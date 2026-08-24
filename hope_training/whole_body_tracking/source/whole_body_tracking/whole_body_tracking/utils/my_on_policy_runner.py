from __future__ import annotations

from collections import namedtuple
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import pathlib
import random
import signal
import stat
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    is_empirical_normalizer,
)
from whole_body_tracking.utils.training_contract import (
    CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY,
    CHECKPOINT_CONTRACT_SCHEMA_KEY,
    CHECKPOINT_CONTRACT_SHA_KEY,
    CHECKPOINT_LAUNCH_CLAIM_SHA_KEY,
    TRAINING_CONTRACT_SCHEMA_VERSION,
    validate_training_launch_claim_sha256,
)
_EXACT_BEHAVIOR_EVENT = "hope_exact_behavior_update"
_RSL_RL_RUNTIME_ABI_EVENT = "hope_rsl_rl_runtime_abi"
_POLICY_STD_UPDATE_EVENT = "hope_policy_std_update"
_POLICY_STD_TELEMETRY_SCHEMA_VERSION = 1
_REWARD_PPO_ECONOMY_EVENT = "hope_action_ball_reward_ppo_economy_update"
_REWARD_PPO_ECONOMY_SCHEMA_VERSION = 1
_REWARD_PPO_ECONOMY_GATE_ENV = (
    "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_GATE"
)
_PRELONG_SEMANTICS_ENABLE_ENV = "HOPE_ACTION_BALL_4096X5_PRELONG_SEMANTICS"
_PRELONG_SEMANTICS_RECIPE_SHA_ENV = (
    "HOPE_ACTION_BALL_4096X5_PRELONG_REWARD_RECIPE_SHA256"
)
_REWARD_PPO_ECONOMY_NUM_ENVS = 4096
_REWARD_PPO_ECONOMY_STEPS_PER_UPDATE = 24
_REWARD_PPO_ECONOMY_ADVANTAGE_TOLERANCE = 5.0e-5
_RSL_RL_RUNTIME_ABI_SCHEMA_VERSION = 1
_A211_ACTOR_WIDTH = 211
_A211_CRITIC_WIDTH = 319
_A211_ACTOR_NORMALIZER_IDENTITY = "action_ball_a211_actor_norm_v2"
_A211_CRITIC_NORMALIZER_IDENTITY = "action_ball_a211_critic_norm_v1"
_C211_ACTOR_WIDTH = 211
_C211_CRITIC_WIDTH = 319
_C211_ACTOR_NORMALIZER_IDENTITY = "action_ball_c211_actor_norm_v2"
_C211_CRITIC_NORMALIZER_IDENTITY = "action_ball_c211_critic_norm_v1"
_ACTION_BALL_211_WAIT_MASK_RANGES = {
    "actor": (_A211_ACTOR_WIDTH, 197, 210),
    "critic": (_A211_CRITIC_WIDTH, 305, 318),
}
_ADAPTIVE_KL_LEARNING_RATE_FLOOR = 1.0e-5
_JOINT_SAFETY_EVENT = "hope_joint_safety_update"
_JOINT_SAFETY_ARTIFACT_SCHEMA_VERSION = 2
_JOINT_SAFETY_COMMIT_EVENT = "hope_joint_safety_optimizer_commit"
_REWARD_EVIDENCE_ARTIFACT_EVENT = "hope_reward_evidence_prepared"
_REWARD_EVIDENCE_COMMIT_EVENT = "hope_reward_evidence_optimizer_commit"
_REWARD_EVIDENCE_ARTIFACT_SCHEMA_VERSION = 1
_REWARD_EVIDENCE_ARTIFACT_MAX_BYTES = 128 * 1024 * 1024
_JOINT_SAFETY_CORE_PAYLOAD_MAX_BYTES = 2 * 1024 * 1024
_JOINT_SAFETY_TERMINAL_PAYLOAD_MAX_BYTES = 24 * 1024 * 1024
_JOINT_SAFETY_NORMAL_ARTIFACT_MAX_BYTES = 4 * 1024 * 1024
_JOINT_SAFETY_FORENSIC_ARTIFACT_MAX_BYTES = 40 * 1024 * 1024
_PLANNER_INITIAL_TTS_BUCKETS = (
    "lt_0p5",
    "eq_0p5",
    "gt_0p5_le_0p9",
    "gt_0p9",
)
# Action families the command term books per-side outcome counters for (hope_commands _clip_names).
_STRIKE_FAMILIES = ("forehand", "backhand")
# Cumulative per-family strike opportunities after which zero legal returns stops being a warm-up
# artefact: a whole side that never once returns is a broken COMMAND (e.g. a target box under the
# table surface), not a policy that needs more samples.  Four HOPE runs sat at exactly 0.0000 for
# thousands of iterations behind a healthy aggregate before anyone looked.
_ZERO_RETURN_ALARM_OPPORTUNITIES = 500
_ZERO_RETURN_ABORT_OPPORTUNITIES = 5000
_EXACT_RESUME_SCHEMA_VERSION = 3
_ENVIRONMENT_RESUME_SCHEMA_VERSION = 4
_SUPPORTED_ENVIRONMENT_RESUME_SCHEMAS = (1, 2, 3, 4)
_SUPPORTED_EXACT_RESUME_SCHEMAS = (1, 2, _EXACT_RESUME_SCHEMA_VERSION)
_ACTION_BALL_FROZEN_EVAL_CONTROL_SCHEMA_VERSION = 1
_RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY = (
    "runtime_bootstrap_receipt_sha256"
)
_RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY = (
    "runtime_bootstrap_lineage_payload_sha256"
)
_RUNTIME_BOOTSTRAP_RECEIPT_KEY = "runtime_bootstrap_receipt"
_EXACT_RESUME_LIVE_STATE_SCHEMA_VERSION = 1
_EXACT_RESUME_LIVE_STATE_KIND = "action_ball_exact_resume_live_state"
_NUMPY_RNG_STATE_SCHEMA_VERSION = 1
_EXACT_RESUME_TELEMETRY_KEYS = (
    "log_dir",
    "wandb_run_id",
    "wandb_run_name",
)

_ACTION_BALL_R10_RUNNER_SCHEMA_VERSION = 1
_ACTION_BALL_R10_RUNNER_HANDOFF_KIND = (
    "action_ball_r10_isaac_runner_boundary_handoff_v1"
)
_ACTION_BALL_R10_RUNNER_SAVE_AUTHORITY_KIND = (
    "action_ball_r10_isaac_runner_save_authority_v1"
)
_ACTION_BALL_R10_COLD_RESTORE_CAPSULE_KIND = (
    "action_ball_r10_isaac_runner_cold_restore_capsule_v1"
)
_ACTION_BALL_R10_RUNNER_INTEGRATION_STATUS = "ISAAC_RUNNER_ADAPTER_ONLY_HOLD"
_ACTION_BALL_R10_RUNNER_OWNER_IDS = (
    "trainer.policy",
    "trainer.optimizer_schedule",
    "trainer.normalizer.actor",
    "trainer.normalizer.critic",
    "trainer.rollout_frontier",
)
_ACTION_BALL_FULL_MDP_DRAIN_OWNER_ORDER = (
    "r05_runtime",
    "motion",
    "racket",
    "physical_ball",
    "r06_landing_outcome",
    "r03_strike_fact",
    "r07_recovery",
)
_ACTION_BALL_FULL_MDP_DRAIN_CHECKPOINT_KIND = (
    "action_ball_full_mdp_ppo_drain_checkpoint_v1"
)
_ACTION_BALL_FULL_MDP_FORMAL_MODE = "formal"
_ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE = (
    "single_action_lean"
)
_ACTION_BALL_EPOCH_UPDATE_ACK_TELEMETRY_SCHEMA_VERSION = 10
_ACTION_BALL_EPOCH_UPDATE_ACK_TELEMETRY_KIND = (
    "action_ball_epoch_optimizer_update_ack_telemetry_v10"
)
_ActionBallFullMdpRunnerDrainChronology = namedtuple(
    "_ActionBallFullMdpRunnerDrainChronology",
    (
        "global_receipt",
        "update_index",
        "completed_environment_steps",
        "optimizer_returned",
        "post_update_acknowledged",
        "operation_sequence",
        "drain_sequence",
        "projection",
        "telemetry_emitted",
    ),
    module=__name__,
)

ActionBallR10ColdRestoreCapsule = namedtuple(
    "ActionBallR10ColdRestoreCapsule",
    (
        "schema_version",
        "kind",
        "checkpoint_bytes",
        "expected_external_pins",
    ),
    module=__name__,
)
ActionBallR10ColdRestoreCapsule.__doc__ = (
    "Externally pinned bytes admitted before RSL-RL asks for observations. "
    "There is deliberately no path fallback or checkpoint-owned pin source."
)

ActionBallR10RunnerBoundaryHandoff = namedtuple(
    "ActionBallR10RunnerBoundaryHandoff",
    (
        "schema_version",
        "kind",
        "integration_status",
        "runtime_wiring",
        "continuation_authorized",
        "completed_update_index",
        "next_learning_iteration",
        "storage_step",
        "transition_empty",
        "environment_step_in_flight",
        "completed_environment_steps",
        "actor_normalizer_calls",
        "critic_normalizer_calls",
        "final_normalized_actor_observations",
        "final_normalized_critic_observations",
        "recurrent_frontier",
        "recurrent_actor_state",
        "recurrent_critic_state",
        "runner_frontier_sha256",
    ),
    module=__name__,
)
ActionBallR10RunnerBoundaryHandoff.__doc__ = (
    "Detached runner frontier handed to the opt-in full-MDP adapter."
)

ActionBallR10RunnerSaveAuthority = namedtuple(
    "ActionBallR10RunnerSaveAuthority",
    (
        "schema_version",
        "kind",
        "family",
        "immutable_pins",
        "boundary",
        "runner_join_claims",
    ),
    module=__name__,
)
ActionBallR10RunnerSaveAuthority.__doc__ = (
    "One adapter decision to seal the complete registry at this boundary."
)


def _action_ball_r10_module():
    """Import the dependency-light pure seam only for an explicit R10 lane."""

    return importlib.import_module("action_ball_full_mdp_checkpoint")


def _action_ball_full_mdp_lean_runtime_module():
    """Import the exact diagnostic epoch owner only for its explicit lane."""

    return importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_lean_runtime"
    )


def _action_ball_full_mdp_durable_wal_module():
    return importlib.import_module(
        "whole_body_tracking.utils.action_ball_full_mdp_durable_wal"
    )


def _action_ball_full_mdp_canonical_sha256(value: object) -> str:
    """Canonicalize the runner writer without accepting another writer's root."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _action_ball_r10_clone_tree(value: object) -> object:
    """Detach tensors and containers without accepting arbitrary pickle types."""

    if torch.is_tensor(value):
        return value.detach().clone()
    if type(value) is dict:
        return {
            key: _action_ball_r10_clone_tree(child)
            for key, child in value.items()
        }
    if isinstance(value, Mapping):
        return {
            key: _action_ball_r10_clone_tree(child)
            for key, child in value.items()
        }
    if type(value) is tuple:
        return tuple(_action_ball_r10_clone_tree(child) for child in value)
    if type(value) is list:
        return [_action_ball_r10_clone_tree(child) for child in value]
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise RuntimeError(
        "R10 runner state contains unsupported clone type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _action_ball_r10_torch_save_bytes(value: object) -> bytes:
    """Serialize a validated tensor tree; the outer R10 package pins the bytes."""

    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _action_ball_r10_torch_load_bytes(value: object) -> object:
    if type(value) is not bytes or not value:
        raise RuntimeError("R10 runner owner payload must be non-empty exact bytes")
    try:
        return torch.load(
            io.BytesIO(value),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError("R10 runner owner payload failed safe tensor loading") from exc


def _action_ball_r10_validate_hidden_tree(value: object, *, label: str) -> None:
    if torch.is_tensor(value):
        if (
            value.numel() <= 0
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise RuntimeError(f"R10 {label} hidden tensor is invalid")
        return
    if type(value) is tuple and value:
        for index, child in enumerate(value):
            _action_ball_r10_validate_hidden_tree(
                child, label=f"{label}[{index}]"
            )
        return
    raise RuntimeError(f"R10 {label} hidden state has an unsupported shape")


def _action_ball_r10_validate_frontier_payload(
    value: object,
    *,
    expected_update_index: Optional[int] = None,
) -> dict:
    """Validate the saved normalized frontier without touching a live runner."""

    if type(value) is not dict:
        raise RuntimeError("R10 rollout frontier payload must be an exact dict")
    required = {
        "schema_version",
        "kind",
        "completed_update_index",
        "next_learning_iteration",
        "num_envs",
        "num_steps_per_env",
        "storage_step",
        "transition_empty",
        "completed_environment_steps",
        "actor_normalizer_calls",
        "critic_normalizer_calls",
        "privileged_obs_type",
        "final_normalized_actor_observations",
        "final_normalized_critic_observations",
        "recurrent_frontier",
        "recurrent_actor_state",
        "recurrent_critic_state",
        "runner_frontier_sha256",
    }
    if set(value) != required:
        raise RuntimeError(
            "R10 rollout frontier keys differ: "
            f"missing={sorted(required - set(value))} "
            f"extra={sorted(set(value) - required)}"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION
        or value["kind"] != "action_ball_r10_trainer_rollout_frontier_v1"
    ):
        raise RuntimeError("R10 rollout frontier schema/kind differs")
    completed = value["completed_update_index"]
    next_iteration = value["next_learning_iteration"]
    if (
        type(completed) is not int
        or completed < 0
        or type(next_iteration) is not int
        or next_iteration != completed + 1
    ):
        raise RuntimeError("R10 rollout frontier update identity differs")
    if expected_update_index is not None and completed != expected_update_index:
        raise RuntimeError("R10 rollout frontier differs from checkpoint boundary")
    num_envs = value["num_envs"]
    num_steps = value["num_steps_per_env"]
    if (
        type(num_envs) is not int
        or num_envs <= 0
        or type(num_steps) is not int
        or num_steps <= 0
        or type(value["completed_environment_steps"]) is not int
        or value["completed_environment_steps"] != num_steps
        or type(value["actor_normalizer_calls"]) is not int
        or value["actor_normalizer_calls"] != num_steps
        or type(value["critic_normalizer_calls"]) is not int
        or value["critic_normalizer_calls"] != num_steps
    ):
        raise RuntimeError("R10 rollout frontier step/normalizer accounting differs")
    if value["storage_step"] != 0 or type(value["storage_step"]) is not int:
        raise RuntimeError("R10 rollout frontier storage is not empty")
    if value["transition_empty"] is not True:
        raise RuntimeError("R10 rollout transition is not empty")
    if value["privileged_obs_type"] != "critic":
        raise RuntimeError("R10 rollout frontier lacks an explicit critic group")
    actor = value["final_normalized_actor_observations"]
    critic = value["final_normalized_critic_observations"]
    for label, tensor in (("actor", actor), ("critic", critic)):
        if (
            not torch.is_tensor(tensor)
            or tensor.ndim != 2
            or tensor.shape[0] <= 0
            or tensor.shape[1] <= 0
            or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(
                f"R10 final normalized {label} observations are invalid"
            )
    if actor.shape[0] != critic.shape[0]:
        raise RuntimeError("R10 actor/critic frontier world counts differ")
    if actor.shape[0] != num_envs:
        raise RuntimeError("R10 rollout frontier num_envs differs")
    r10 = _action_ball_r10_module()
    recurrent = value["recurrent_frontier"]
    if recurrent not in (
        r10.RecurrentFrontierStatus.SEALED.value,
        r10.RecurrentFrontierStatus.NOT_APPLICABLE.value,
    ) or type(recurrent) is not str:
        raise RuntimeError("R10 recurrent frontier status differs")
    actor_hidden = value["recurrent_actor_state"]
    critic_hidden = value["recurrent_critic_state"]
    if recurrent == r10.RecurrentFrontierStatus.NOT_APPLICABLE.value:
        if actor_hidden is not None or critic_hidden is not None:
            raise RuntimeError("non-recurrent R10 frontier retains hidden state")
    elif actor_hidden is None or critic_hidden is None:
        raise RuntimeError("recurrent R10 frontier is missing hidden state")
    else:
        _action_ball_r10_validate_hidden_tree(actor_hidden, label="actor")
        _action_ball_r10_validate_hidden_tree(critic_hidden, label="critic")
    digest = value["runner_frontier_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("R10 runner frontier SHA-256 is invalid")
    unhashed = {key: child for key, child in value.items() if key != "runner_frontier_sha256"}
    if MotionOnPolicyRunner._exact_resume_tree_sha256(unhashed) != digest:
        raise RuntimeError("R10 runner frontier content SHA-256 differs")
    return value


class _ActionBallR10EnvMethodGuard:
    """Temporarily replace reset/get-observations on wrapper and unwrapped env."""

    def __init__(self, env: object) -> None:
        targets = []
        for candidate in (env, getattr(env, "unwrapped", None)):
            if candidate is not None and all(candidate is not current for current in targets):
                targets.append(candidate)
        self._targets = tuple(targets)
        self._patches = []

    def _patch(self, target: object, name: str, replacement: object) -> None:
        if not hasattr(target, name):
            return
        namespace = getattr(target, "__dict__", None)
        had_instance_value = isinstance(namespace, dict) and name in namespace
        prior = namespace[name] if had_instance_value else getattr(target, name)
        try:
            setattr(target, name, replacement)
        except Exception as exc:
            raise RuntimeError(
                f"R10 cold restore cannot guard env.{name}"
            ) from exc
        self._patches.append((target, name, had_instance_value, prior))

    def install_initial_observation_shim(
        self,
        actor: torch.Tensor,
        critic: torch.Tensor,
    ) -> Callable[[], int]:
        calls = {"count": 0}

        def get_saved_frontier():
            calls["count"] += 1
            if calls["count"] != 1:
                raise RuntimeError(
                    "R10 cold construction requested observations more than once"
                )
            return (
                actor.detach().clone(),
                {
                    "observations": {
                        "critic": critic.detach().clone(),
                    }
                },
            )

        def reset_forbidden(*_args, **_kwargs):
            raise RuntimeError("R10 cold construction forbids env.reset()")

        def step_forbidden(*_args, **_kwargs):
            raise RuntimeError("R10 cold construction forbids env.step()")

        try:
            for target in self._targets:
                self._patch(target, "get_observations", get_saved_frontier)
                self._patch(target, "reset", reset_forbidden)
                self._patch(target, "step", step_forbidden)
        except BaseException:
            self.close()
            raise
        return lambda: calls["count"]

    def install_restore_guards(self) -> None:
        def getter_forbidden(*_args, **_kwargs):
            raise RuntimeError("R10 cold restore forbids env.get_observations()")

        def reset_forbidden(*_args, **_kwargs):
            raise RuntimeError("R10 cold restore forbids env.reset()")

        def step_forbidden(*_args, **_kwargs):
            raise RuntimeError("R10 cold restore forbids env.step()")

        try:
            for target in self._targets:
                self._patch(target, "get_observations", getter_forbidden)
                self._patch(target, "reset", reset_forbidden)
                self._patch(target, "step", step_forbidden)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        failures = []
        for target, name, had_instance_value, prior in reversed(self._patches):
            try:
                if had_instance_value:
                    setattr(target, name, prior)
                else:
                    delattr(target, name)
            except BaseException as exc:
                failures.append(
                    f"{name}:{type(exc).__module__}."
                    f"{type(exc).__qualname__}"
                )
        self._patches.clear()
        if failures:
            raise RuntimeError(
                "R10 env method guard could not restore methods: "
                + ",".join(failures)
            )


class _ActionBallR10TrainerOwner:
    """Typed pure-seam owner for one runner-controlled mutation domain."""

    def __init__(
        self,
        *,
        descriptor: object,
        mutation_version: int,
        state_getter: Callable[[], object],
        state_validator: Callable[[object], object],
        state_installer: Callable[[object], None],
        join_claims: Tuple[object, ...],
        poison: Callable[[str], None],
    ) -> None:
        if type(mutation_version) is not int or mutation_version < 0:
            raise RuntimeError("R10 trainer owner mutation version is invalid")
        self.descriptor = descriptor
        self._mutation_version = mutation_version
        self._state_getter = state_getter
        self._state_validator = state_validator
        self._state_installer = state_installer
        self._join_claims = join_claims
        self._poison = poison
        self._token_authority = object()

    def mutation_version(self) -> int:
        return self._mutation_version

    def live_digest(self) -> str:
        state = self._state_getter()
        return MotionOnPolicyRunner._exact_resume_tree_sha256(state)

    def freeze(self, boundary) -> object:
        r10 = _action_ball_r10_module()
        boundary_digest = r10.boundary_sha256(boundary)
        nonce = hashlib.sha256(
            (
                self.descriptor.owner_id
                + ":"
                + str(self._mutation_version)
                + ":"
                + boundary_digest
                + ":"
                + self.live_digest()
            ).encode("ascii")
        ).hexdigest()
        return r10.OwnerFreezeReceipt(
            owner_id=self.descriptor.owner_id,
            descriptor_sha256=r10.descriptor_sha256(self.descriptor),
            boundary_sha256=boundary_digest,
            mutation_version=self._mutation_version,
            seal_nonce_sha256=nonce,
        )

    def export_sealed(self, receipt) -> object:
        r10 = _action_ball_r10_module()
        state = self._state_getter()
        self._state_validator(state)
        return r10.make_opaque_owner_state(
            descriptor=self.descriptor,
            receipt=receipt,
            live_digest_sha256=MotionOnPolicyRunner._exact_resume_tree_sha256(
                state
            ),
            payload=_action_ball_r10_torch_save_bytes(state),
            join_claims=self._join_claims,
        )

    def prepare_restore(
        self,
        envelope,
        immutable_pins,
        owner_root_sha256: str,
    ) -> object:
        del immutable_pins
        r10 = _action_ball_r10_module()
        staged = _action_ball_r10_torch_load_bytes(envelope.payload)
        self._state_validator(staged)
        baseline = _action_ball_r10_torch_save_bytes(self._state_getter())
        token_state = {
            "authority": self._token_authority,
            "staged": staged,
            "baseline": baseline,
            "baseline_version": self._mutation_version,
            "target_version": envelope.mutation_version,
        }
        return r10.PreparedRestoreToken(
            owner_id=self.descriptor.owner_id,
            descriptor_sha256=r10.descriptor_sha256(self.descriptor),
            checkpoint_owner_root_sha256=owner_root_sha256,
            opaque_token=token_state,
        )

    def _token_state(self, token) -> dict:
        value = token.opaque_token
        if type(value) is not dict or value.get("authority") is not self._token_authority:
            raise RuntimeError("R10 trainer restore token authority differs")
        return value

    def commit_restore(self, token) -> None:
        state = self._token_state(token)
        self._state_installer(state["staged"])
        self._mutation_version = state["target_version"]

    def rollback_restore(self, token) -> None:
        state = self._token_state(token)
        baseline = _action_ball_r10_torch_load_bytes(state["baseline"])
        self._state_installer(baseline)
        self._mutation_version = state["baseline_version"]

    def poison_restore(self, reason: str) -> None:
        self._poison(reason)


def _mask_action_ball_211_wait_after_normalization(
    raw_observations: torch.Tensor,
    normalized_observations: torch.Tensor,
    *,
    role: str,
) -> torch.Tensor:
    """Restore the hidden-WAIT zero tuple after empirical normalization.

    The final ``task_valid`` column is deliberately read from the raw tensor.
    Reading the normalized indicator would make the decision depend on running
    moments.  The indicator itself remains normalized; only task9 + base2 +
    clocks2 are restored to exact zero for raw-invalid rows.
    """

    try:
        width, mask_start, mask_stop = _ACTION_BALL_211_WAIT_MASK_RANGES[role]
    except KeyError as exc:
        raise ValueError("ActionBall211 normalizer role must be actor or critic") from exc
    if not isinstance(raw_observations, torch.Tensor) or not isinstance(
        normalized_observations, torch.Tensor
    ):
        raise RuntimeError(
            f"ActionBall211 {role} normalizer must consume and return tensors"
        )
    if (
        raw_observations.shape != normalized_observations.shape
        or raw_observations.ndim < 1
        or raw_observations.shape[-1] != width
    ):
        raise RuntimeError(
            f"ActionBall211 {role} normalizer ABI must remain {width}-D; "
            f"raw={tuple(raw_observations.shape)!r} "
            f"normalized={tuple(normalized_observations.shape)!r}"
        )
    raw_invalid = raw_observations[..., -1] == 0
    masked_region = torch.where(
        raw_invalid.unsqueeze(-1),
        torch.zeros_like(normalized_observations[..., mask_start:mask_stop]),
        normalized_observations[..., mask_start:mask_stop],
    )
    return torch.cat(
        (
            normalized_observations[..., :mask_start],
            masked_region,
            normalized_observations[..., mask_stop:],
        ),
        dim=-1,
    )


def _action_ball_211_wait_forward_hook(role: str):
    """Build one stateless hook without wrapping the normalizer module/state."""

    def hook(_module, inputs, output):
        if not isinstance(inputs, tuple) or len(inputs) != 1:
            raise RuntimeError(
                f"ActionBall211 {role} normalizer requires one raw observation input"
            )
        return _mask_action_ball_211_wait_after_normalization(
            inputs[0], output, role=role
        )

    return hook


def _ratio_or_none(counters: dict, numerator: str, denominator: str):
    """Return an honest derived value; an absent/zero denominator is unavailable, never zero."""

    denom = counters.get(denominator, 0)
    if denom is None or float(denom) <= 0.0:
        return None
    value = float(counters.get(numerator, 0)) / float(denom)
    return value if math.isfinite(value) else None


def _scaled_ratio_or_none(counters: dict, numerator: str, denominator: str, scale: float):
    """``_ratio_or_none`` times a unit conversion (the ledger is integer-only: um -> mm)."""

    value = _ratio_or_none(counters, numerator, denominator)
    return None if value is None else value * float(scale)


def exact_behavior_decision_values(
    counters: dict,
) -> Dict[str, Optional[float]]:
    """Derive dashboard values from one record; windows must sum counters before calling this."""

    values = {
        "swing_completion_rate": _ratio_or_none(
            counters, "swing_completion_count", "swing_outcome_count"
        ),
        "pre_strike_physical_fall_rate": _ratio_or_none(
            counters, "pre_strike_physical_fall_count", "swing_outcome_count"
        ),
        "post_strike_physical_fall_rate": _ratio_or_none(
            counters, "post_strike_physical_fall_count", "swing_outcome_count"
        ),
        "virtual_capture_per_strike": _ratio_or_none(
            counters, "virtual_capture_count", "strike_opportunity_count"
        ),
        "virtual_net_clear_per_capture": _ratio_or_none(
            counters, "virtual_net_clear_count", "virtual_capture_count"
        ),
        "virtual_landing_valid_per_capture": _ratio_or_none(
            counters, "virtual_landing_valid_count", "virtual_capture_count"
        ),
        "virtual_legal_return_per_strike": _ratio_or_none(
            counters, "virtual_legal_return_count", "strike_opportunity_count"
        ),
        "ready_tilt_rad_mean": _ratio_or_none(
            counters, "ready_tilt_rad_sum", "ready_tilt_eligible_sample_count"
        ),
        "ready_base_speed_xy_mps_mean": _ratio_or_none(
            counters,
            "ready_base_speed_xy_mps_sum",
            "ready_base_speed_eligible_sample_count",
        ),
        "ready_station_offset_m_mean": _ratio_or_none(
            counters,
            "ready_station_offset_m_sum",
            "ready_station_offset_eligible_sample_count",
        ),
        "ready_foot_contact_fraction_mean": _ratio_or_none(
            counters,
            "ready_foot_contact_fraction_sum",
            "ready_foot_contact_eligible_sample_count",
        ),
        "ready_foot_slip_speed_mps_mean": _ratio_or_none(
            counters,
            "ready_foot_slip_speed_mps_sum",
            "ready_foot_slip_eligible_sample_count",
        ),
    }
    # CONTINUOUS question production. A bank arm never draws, so `continuous_question_draw_count`
    # is absent/zero there and _ratio_or_none makes every one of these read None — never a lying
    # 0.0 that would look like a healthy continuous run.
    values["continuous_question_exhausted_rate"] = _ratio_or_none(
        counters, "continuous_question_exhausted_count", "continuous_question_draw_count"
    )
    values["continuous_question_admit_rate"] = _ratio_or_none(
        counters, "continuous_question_admitted_count", "continuous_question_draw_count"
    )
    values["continuous_question_resid_mm_mean"] = _scaled_ratio_or_none(
        counters, "continuous_question_resid_um_sum", "continuous_question_admitted_count", 1e-3
    )
    values["continuous_question_closed_loop_mm_mean"] = _scaled_ratio_or_none(
        counters,
        "continuous_question_closed_loop_um_sum",
        "continuous_question_closed_loop_row_count",
        1e-3,
    )
    values["continuous_question_redraw_rounds_mean"] = _ratio_or_none(
        counters, "continuous_question_redraw_round_sum", "continuous_question_refill_count"
    )
    for bucket_name in _PLANNER_INITIAL_TTS_BUCKETS:
        prefix = f"planner_initial_tts_{bucket_name}_"
        values[f"{prefix}swing_completion_rate"] = _ratio_or_none(
            counters,
            f"{prefix}swing_completion_count",
            f"{prefix}swing_outcome_count",
        )
        values[f"{prefix}virtual_capture_per_strike"] = _ratio_or_none(
            counters,
            f"{prefix}virtual_capture_count",
            f"{prefix}strike_opportunity_count",
        )
        values[f"{prefix}virtual_legal_return_per_strike"] = _ratio_or_none(
            counters,
            f"{prefix}virtual_legal_return_count",
            f"{prefix}strike_opportunity_count",
        )
    for family in _STRIKE_FAMILIES:
        values[f"virtual_capture_per_strike_{family}"] = _ratio_or_none(
            counters,
            f"virtual_capture_count_{family}",
            f"strike_opportunity_count_{family}",
        )
        values[f"virtual_legal_return_per_strike_{family}"] = _ratio_or_none(
            counters,
            f"virtual_legal_return_count_{family}",
            f"strike_opportunity_count_{family}",
        )
    return values


def zero_return_alarm_levels(cumulative: dict) -> Dict[str, str]:
    """Families whose cumulative strike opportunities have produced exactly zero legal returns.

    Returns {family: "alarm" | "abort"}; families that returned at least once, or that have not yet
    accumulated enough opportunities, are absent.  This is the ONLY reading that separates "this side
    is never eligible / never satisfiable" from "this side sometimes fails" — the aggregate rate
    averages a dead side against a healthy one into a plausible-looking number.
    """

    levels: Dict[str, str] = {}
    for family in _STRIKE_FAMILIES:
        opportunities = cumulative.get(f"strike_opportunity_count_{family}", 0) or 0
        returns = cumulative.get(f"virtual_legal_return_count_{family}", 0) or 0
        if float(returns) > 0.0:
            continue
        if float(opportunities) >= _ZERO_RETURN_ABORT_OPPORTUNITIES:
            levels[family] = "abort"
        elif float(opportunities) >= _ZERO_RETURN_ALARM_OPPORTUNITIES:
            levels[family] = "alarm"
    return levels


# [已删除 2026-08-06 过期结构清理] class MyOnPolicyRunner(26 行):上游 BeyondMimic 在
# 8a9d329c 一起带进来的 runner,HOPE 从来没有实例化过它(全仓对 ``MyOnPolicyRunner``
# 这个名字的引用数为 0,包括 yaml/json/文档/被 gitignore 的目录)。
# 它的 save() 与下面 MotionOnPolicyRunner.save() 的 ONNX 导出段逐行相同 —— 也就是说
# "checkpoint 落盘时怎么导 ONNX"这件事存了两份,而只有一份在跑。谁去修导出(改文件名规则、
# 改 obs_norm 烘焙、加 metadata 字段),都有一半概率修在这份死的上。现役唯一 runner 是
# MotionOnPolicyRunner:train.py:17641 与 action_ball_frozen_eval_sidecar.py:1938 导入它。
class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: Optional[str] = None,
        device="cpu",
        registry_name=None,
        *,
        training_contract_schema_version: Optional[int] = None,
        training_contract_sha256: Optional[str] = None,
        training_contract_lineage_exact: bool = False,
        training_launch_claim_sha256: Optional[str] = None,
        require_exact_resume_state: bool = False,
        action_ball_r10_checkpoint_adapter: object = None,
        action_ball_r10_cold_restore_capsule: object = None,
        action_ball_full_mdp_runtime_owner: object = None,
        action_ball_full_mdp_run_mode: object = None,
    ):
        if type(require_exact_resume_state) is not bool:
            raise TypeError("require_exact_resume_state must be a bool")
        if type(training_contract_lineage_exact) is not bool:
            raise TypeError(
                "training_contract_lineage_exact must be an exact bool"
            )
        full_mdp_enabled = action_ball_full_mdp_runtime_owner is not None
        r10_requested = action_ball_r10_checkpoint_adapter is not None
        if action_ball_full_mdp_run_mode is None:
            # Preserve the already-reviewed formal constructor ABI.  The new
            # diagnostic exception is never inferred: owner-without-R10 must
            # carry its exact explicit mode.
            full_mdp_run_mode = (
                _ACTION_BALL_FULL_MDP_FORMAL_MODE
                if full_mdp_enabled and r10_requested
                else None
            )
        elif type(action_ball_full_mdp_run_mode) is not str:
            raise TypeError("fresh full-MDP run mode must be an exact string")
        else:
            full_mdp_run_mode = action_ball_full_mdp_run_mode
        if full_mdp_run_mode not in {
            None,
            _ACTION_BALL_FULL_MDP_FORMAL_MODE,
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE,
        }:
            raise RuntimeError("fresh full-MDP run mode is unsupported")
        if full_mdp_run_mode is None and (full_mdp_enabled or r10_requested):
            raise RuntimeError(
                "fresh full-MDP runner requires the runtime owner and R10 "
                "checkpoint adapter together"
            )
        if full_mdp_run_mode == _ACTION_BALL_FULL_MDP_FORMAL_MODE and (
            not full_mdp_enabled or not r10_requested
        ):
            raise RuntimeError(
                "formal fresh full-MDP runner requires the runtime owner and "
                "R10 checkpoint adapter together"
            )
        if full_mdp_run_mode == (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ) and (not full_mdp_enabled or r10_requested):
            raise RuntimeError(
                "single_action_lean requires one runtime owner and forbids "
                "the R10 checkpoint adapter"
            )
        if (
            full_mdp_run_mode
            == _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
            and action_ball_r10_cold_restore_capsule is not None
        ):
            raise RuntimeError(
                "single_action_lean forbids a cold-restore capsule"
            )
        runtime_env = getattr(env, "unwrapped", env)
        if (
            full_mdp_run_mode
            == _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
            and (
                type(getattr(runtime_env, "num_envs", None)) is not int
                or runtime_env.num_envs <= 0
            )
        ):
            raise RuntimeError(
                "single_action_lean requires positive exact-int num_envs"
            )
        full_mdp_lean_runtime_module = None
        full_mdp_lean_epoch_owner = None
        full_mdp_lean_shot_slot_capacity = None
        if full_mdp_run_mode == (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            full_mdp_lean_runtime_module = (
                _action_ball_full_mdp_lean_runtime_module()
            )
            lean_owner_type = getattr(
                full_mdp_lean_runtime_module,
                "ActionBallFullMdpLeanRuntimeOwner",
                None,
            )
            lean_summary_type = getattr(
                full_mdp_lean_runtime_module,
                "ActionEpochPpoBoundarySummary",
                None,
            )
            if (
                type(lean_owner_type) is not type
                or type(lean_summary_type) is not type
                or type(action_ball_full_mdp_runtime_owner)
                is not lean_owner_type
            ):
                raise RuntimeError(
                    "single_action_lean requires the exact code-owned lean "
                    "runtime owner"
                )
            full_mdp_lean_epoch_owner = getattr(
                action_ball_full_mdp_runtime_owner, "epoch_owner", None
            )
            epoch_module = getattr(full_mdp_lean_runtime_module, "epoch_v1", None)
            epoch_owner_type = getattr(epoch_module, "ActionEpochOwner", None)
            if (
                type(epoch_owner_type) is not type
                or type(full_mdp_lean_epoch_owner) is not epoch_owner_type
                or getattr(full_mdp_lean_epoch_owner, "num_envs", None)
                != runtime_env.num_envs
                or getattr(action_ball_full_mdp_runtime_owner, "_ppo_drain", None)
                is not action_ball_full_mdp_runtime_owner
            ):
                raise RuntimeError(
                    "single_action_lean requires one exact ActionEpoch owner "
                    "and no compatibility drain"
                )
            full_mdp_lean_shot_slot_capacity = getattr(
                full_mdp_lean_epoch_owner, "shot_slot_capacity", None
            )
            if (
                type(full_mdp_lean_shot_slot_capacity) is not int
                or full_mdp_lean_shot_slot_capacity != 1
            ):
                raise RuntimeError(
                    "single_action_lean ActionEpoch shot capacity differs"
                )
        full_mdp_callbacks = None
        full_mdp_prepare_summary_callback = None
        full_mdp_drain_owner = None
        full_mdp_projection_callback = None
        if full_mdp_enabled:
            callback_names = (
                "require_healthy",
                "prepare_pre_optimizer_ppo_boundary",
                "mark_optimizer_returned",
                "acknowledge_post_update",
                "poison_optimizer_boundary",
            )
            full_mdp_callbacks = tuple(
                getattr(action_ball_full_mdp_runtime_owner, name, None)
                for name in callback_names
            )
            missing = tuple(
                name
                for name, callback in zip(callback_names, full_mdp_callbacks)
                if not callable(callback)
            )
            if missing:
                raise RuntimeError(
                    "fresh full-MDP runtime owner lacks exact runner callbacks: "
                    + ",".join(missing)
                )
            if any(
                getattr(callback, "__self__", None)
                is not action_ball_full_mdp_runtime_owner
                for callback in full_mdp_callbacks
            ):
                raise RuntimeError(
                    "fresh full-MDP runner callbacks are not exact methods of "
                    "the supplied runtime owner"
                )
            if full_mdp_run_mode == (
                _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
            ):
                full_mdp_prepare_summary_callback = getattr(
                    action_ball_full_mdp_runtime_owner,
                    "prepare_post_update_summary",
                    None,
                )
                if (
                    not callable(full_mdp_prepare_summary_callback)
                    or getattr(
                        full_mdp_prepare_summary_callback, "__self__", None
                    )
                    is not action_ball_full_mdp_runtime_owner
                ):
                    raise RuntimeError(
                        "single_action_lean lean owner lacks its exact "
                        "prepare_post_update_summary method"
                    )
                durable_ack = getattr(
                    action_ball_full_mdp_runtime_owner,
                    "_record_durable_epoch_ack_span",
                    None,
                )
                if (
                    not callable(durable_ack)
                    or getattr(durable_ack, "__self__", None)
                    is not action_ball_full_mdp_runtime_owner
                ):
                    raise RuntimeError(
                        "single_action_lean lean owner lacks its exact durable ACK latch"
                    )
            # This runs before the upstream constructor can request the first
            # observation.  An already-poisoned owner must never be laundered
            # into a newly constructed trainer.
            full_mdp_callbacks[0]()
        validated_launch_claim = None
        if training_launch_claim_sha256 is not None:
            validated_launch_claim = validate_training_launch_claim_sha256(
                training_launch_claim_sha256
            )
        runtime_cfg = getattr(runtime_env, "cfg", None)
        runtime_obs_mode = str(getattr(runtime_cfg, "obs_mode", "") or "")
        if full_mdp_enabled:
            lease = getattr(
                runtime_env, "action_ball_full_mdp_runtime_lease", None
            )
            drain_getter = getattr(
                runtime_env, "action_ball_full_mdp_ppo_drain_owner", None
            )
            owner_env = getattr(
                action_ball_full_mdp_runtime_owner,
                "full_mdp_runtime_env",
                runtime_env,
            )
            owner_lease = getattr(
                action_ball_full_mdp_runtime_owner,
                "full_mdp_runtime_lease",
                lease,
            )
            if lease is None or not callable(drain_getter):
                raise RuntimeError(
                    "fresh full-MDP runner lacks the lease-bound global drain owner"
                )
            if getattr(drain_getter, "__self__", None) is not runtime_env:
                raise RuntimeError(
                    "fresh full-MDP global drain getter is not bound to the "
                    "exact runtime environment"
                )
            if owner_env is not runtime_env or owner_lease is not lease:
                raise RuntimeError(
                    "fresh full-MDP runtime owner/env lease binding differs"
                )
            full_mdp_drain_owner = drain_getter(lease)
            if full_mdp_run_mode == _ACTION_BALL_FULL_MDP_FORMAL_MODE:
                full_mdp_projection_callback = getattr(
                    full_mdp_drain_owner,
                    "require_owned_runner_frontier_projection",
                    None,
                )
                if not callable(full_mdp_projection_callback):
                    raise RuntimeError(
                        "formal fresh full-MDP drain lacks the runner frontier "
                        "projection callback"
                    )
                if (
                    getattr(full_mdp_projection_callback, "__self__", None)
                    is not full_mdp_drain_owner
                ):
                    raise RuntimeError(
                        "formal fresh full-MDP runner frontier projection is not "
                        "bound to the exact global drain owner"
                    )
            if (
                full_mdp_run_mode
                == _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
                and full_mdp_drain_owner
                is not action_ball_full_mdp_runtime_owner
            ):
                raise RuntimeError(
                    "single_action_lean env installed a foreign or "
                    "compatibility PPO drain"
                )
        if runtime_obs_mode in {
            "action_ball_a225",
            "action_ball_c225",
            "action_ball_a210",
            "action_ball_c210",
        }:
            raise RuntimeError(
                f"legacy {runtime_obs_mode} is not consumable by the fresh A211/C211 runner ABI"
            )
        r10_registry = None
        r10_verified_checkpoint = None
        r10_cold_frontier = None
        r10_construction_guard = None
        r10_construction_shim_calls = None
        if (
            action_ball_r10_checkpoint_adapter is not None
            or action_ball_r10_cold_restore_capsule is not None
        ):
            self._action_ball_r10_validate_opt_in_lane(
                runtime_obs_mode=runtime_obs_mode,
                require_exact_resume_state=require_exact_resume_state,
                training_contract_schema_version=training_contract_schema_version,
                training_contract_sha256=training_contract_sha256,
                log_dir=log_dir,
                adapter=action_ball_r10_checkpoint_adapter,
                capsule=action_ball_r10_cold_restore_capsule,
            )
            self._action_ball_r10_validate_adapter_api_authority(
                action_ball_r10_checkpoint_adapter,
                training_launch_claim_sha256=validated_launch_claim,
            )
            r10_registry = self._action_ball_r10_adapter_registry(
                action_ball_r10_checkpoint_adapter
            )
            if action_ball_r10_cold_restore_capsule is not None:
                (
                    r10_verified_checkpoint,
                    r10_cold_frontier,
                ) = self._action_ball_r10_preflight_cold_capsule(
                    runtime_obs_mode=runtime_obs_mode,
                    runtime_num_envs=getattr(runtime_env, "num_envs", None),
                    runtime_num_steps_per_env=(
                        train_cfg.get("num_steps_per_env")
                        if type(train_cfg) is dict
                        else None
                    ),
                    training_contract_sha256=training_contract_sha256,
                    registry=r10_registry,
                    capsule=action_ball_r10_cold_restore_capsule,
                )
                r10_construction_guard = _ActionBallR10EnvMethodGuard(env)
                r10_construction_shim_calls = (
                    r10_construction_guard.install_initial_observation_shim(
                        r10_cold_frontier[
                            "final_normalized_actor_observations"
                        ],
                        r10_cold_frontier[
                            "final_normalized_critic_observations"
                        ],
                    )
                )
        try:
            super().__init__(env, train_cfg, log_dir, device)
        finally:
            if r10_construction_guard is not None:
                r10_construction_guard.close()
        if full_mdp_run_mode == (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            # Upstream persists at iteration zero and again after learn().
            # A diagnostic no-save lane therefore cannot merely rely on a
            # large save_interval.  Suppressing its presentation/log writer is
            # what keeps ordinary completion away from every save call; the
            # checkpoint method itself remains a second fail-closed guard.
            self.disable_logs = True
        if (
            r10_construction_shim_calls is not None
            and r10_construction_shim_calls() != 1
        ):
            raise RuntimeError(
                "R10 cold construction did not consume exactly one saved observation shim"
            )
        if runtime_obs_mode == "action_ball_a211":
            # Import the Isaac task contract only for its dedicated leaf.  This
            # preserves dependency-light runner/receipt audits for every other
            # task while keeping the real A211 constructor fail-closed.
            from whole_body_tracking.tasks.tracking.action_ball_a211_trainability import (
                validate_action_ball_211_runner,
            )

            self.action_ball_a211_trainability_preflight = (
                validate_action_ball_211_runner(self)
            )
        else:
            self.action_ball_a211_trainability_preflight = None
        if runtime_obs_mode == "action_ball_c211":
            from whole_body_tracking.tasks.tracking.action_ball_c211_trainability import (
                validate_action_ball_c211_runner,
            )

            self.action_ball_c211_trainability_preflight = (
                validate_action_ball_c211_runner(self)
            )
        else:
            self.action_ball_c211_trainability_preflight = None
        self._install_action_ball_211_wait_normalizer_masks()
        self.registry_name = registry_name
        self.training_contract_schema_version = training_contract_schema_version
        self.training_contract_sha256 = training_contract_sha256
        self.training_contract_lineage_exact = training_contract_lineage_exact
        self.training_launch_claim_sha256 = validated_launch_claim
        # Formal ActionBall cannot bind these at construction: env.pkl,
        # agent.pkl and the independently reconstructed runtime identity are
        # emitted only after the runner exists.  ``train.py`` must call the
        # one-shot binder after publishing the no-clobber bootstrap receipt
        # and before load/save/learn can cross a checkpoint boundary.
        self.runtime_bootstrap_receipt_sha256 = None
        self.runtime_bootstrap_lineage_payload_sha256 = None
        self.runtime_bootstrap_receipt = None
        self._runtime_bootstrap_content = None
        # This is a construction-time run contract, not a permissive load flag. Task-first
        # training passes True; evaluation and legacy/warm-start runs retain the historical False.
        self.require_exact_resume_state = require_exact_resume_state
        if require_exact_resume_state and self.log_dir is None:
            raise ValueError(
                "require_exact_resume_state is a training-resume contract and requires log_dir"
            )
        if (training_contract_schema_version is None) != (training_contract_sha256 is None):
            raise ValueError("training contract schema and SHA256 must be supplied together")
        if training_contract_schema_version is not None:
            if int(training_contract_schema_version) != TRAINING_CONTRACT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported training contract schema {training_contract_schema_version}; "
                    f"expected {TRAINING_CONTRACT_SCHEMA_VERSION}"
                )
            digest = str(training_contract_sha256).strip().lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("training_contract_sha256 must be 64 lowercase hex characters")
            self.training_contract_sha256 = digest
        if (
            require_exact_resume_state
            and self.training_contract_sha256 is None
        ):
            raise ValueError(
                "require_exact_resume_state requires a training-contract binding"
            )
        # A fresh single_action_lean construction has no checkpoint
        # adapter/capsule and cannot save.  Do not require resume hooks merely
        # because a future load *could* be called: load() below is the exact
        # resume-intent boundary and rechecks before reading checkpoint bytes.
        # Formal and legacy exact-resume construction remains unchanged.
        if full_mdp_run_mode != (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            self._validate_task_first_exact_resume_terms()
        self._action_ball_r10_checkpoint_adapter = None
        self._action_ball_r10_registry = None
        self._action_ball_r10_runtime_poisoned = False
        self._action_ball_r10_poison_reason = None
        self._action_ball_r10_last_boundary_handoff = None
        self._action_ball_r10_last_publication = None
        self._action_ball_r10_last_publish_result = None
        self._action_ball_r10_restore_receipt = None
        self._action_ball_r10_restore_complete_callback = None
        self._action_ball_r10_restore_completion_attempted = False
        self._action_ball_r10_restore_owners = None
        self._action_ball_r10_restore_poison_attempted_owner_ids = ()
        self._action_ball_r10_restore_poison_failures = ()
        self._action_ball_r10_frontier_payload = None
        self._action_ball_r10_restored_initial_frontier = None
        self._action_ball_full_mdp_runtime_owner = (
            action_ball_full_mdp_runtime_owner
        )
        self._action_ball_full_mdp_run_mode = full_mdp_run_mode
        self._action_ball_full_mdp_runtime_owner_identity = (
            action_ball_full_mdp_runtime_owner
        )
        self._action_ball_full_mdp_runtime_owner_type = (
            None
            if action_ball_full_mdp_runtime_owner is None
            else type(action_ball_full_mdp_runtime_owner)
        )
        self._action_ball_full_mdp_runtime_env_identity = (
            runtime_env if full_mdp_enabled else None
        )
        self._action_ball_full_mdp_runtime_lease_identity = (
            lease if full_mdp_enabled else None
        )
        self._action_ball_full_mdp_drain_getter_callback = (
            drain_getter if full_mdp_enabled else None
        )
        self._action_ball_full_mdp_drain_owner = full_mdp_drain_owner
        self._action_ball_full_mdp_drain_owner_identity = full_mdp_drain_owner
        self._action_ball_full_mdp_drain_owner_type = (
            None if full_mdp_drain_owner is None else type(full_mdp_drain_owner)
        )
        self._action_ball_full_mdp_lean_runtime_module = (
            full_mdp_lean_runtime_module
        )
        self._action_ball_full_mdp_lean_epoch_owner_identity = (
            full_mdp_lean_epoch_owner
        )
        self._action_ball_full_mdp_lean_shot_slot_capacity = (
            full_mdp_lean_shot_slot_capacity
        )
        self._action_ball_full_mdp_family = None
        self._action_ball_full_mdp_projection_callback = (
            full_mdp_projection_callback
        )
        self._action_ball_full_mdp_drain_device = (
            None
            if not full_mdp_enabled
            else torch.device(getattr(runtime_env, "device", device))
        )
        self._action_ball_full_mdp_require_healthy_callback = (
            None if full_mdp_callbacks is None else full_mdp_callbacks[0]
        )
        self._action_ball_full_mdp_prepare_callback = (
            None if full_mdp_callbacks is None else full_mdp_callbacks[1]
        )
        self._action_ball_full_mdp_mark_optimizer_callback = (
            None if full_mdp_callbacks is None else full_mdp_callbacks[2]
        )
        self._action_ball_full_mdp_ack_callback = (
            None if full_mdp_callbacks is None else full_mdp_callbacks[3]
        )
        self._action_ball_full_mdp_prepare_summary_callback = (
            full_mdp_prepare_summary_callback
        )
        self._action_ball_full_mdp_poison_callback = (
            None if full_mdp_callbacks is None else full_mdp_callbacks[4]
        )
        self._action_ball_full_mdp_durable_ack_callback = (
            None
            if full_mdp_run_mode != _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
            else action_ball_full_mdp_runtime_owner._record_durable_epoch_ack_span
        )
        self._action_ball_full_mdp_boundary_active = False
        self._action_ball_full_mdp_boundary_prepared = None
        self._action_ball_full_mdp_boundary_poisoned = False
        self._action_ball_full_mdp_boundary_poison_reason = None
        self._action_ball_full_mdp_boundary_poison_callback_failure = None
        self._action_ball_full_mdp_active_drain_chronology = None
        self._action_ball_full_mdp_last_drain_chronology = None
        self._action_ball_full_mdp_durable_wal_path = None
        self._action_ball_full_mdp_durable_wal_identity = None
        self._action_ball_full_mdp_durable_wal_size = None
        self._action_ball_full_mdp_durable_wal_segment_id = None
        self._action_ball_r10_owner_mutation_versions = {
            owner_id: 0 for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS
        }
        if action_ball_r10_checkpoint_adapter is not None:
            self.bind_action_ball_r10_checkpoint_adapter(
                action_ball_r10_checkpoint_adapter,
                _prevalidated_registry=r10_registry,
            )
        if r10_verified_checkpoint is not None:
            self._action_ball_r10_restore_cold_checkpoint(
                verified_checkpoint=r10_verified_checkpoint,
                capsule=action_ball_r10_cold_restore_capsule,
                cold_frontier=r10_cold_frontier,
            )

    @staticmethod
    def _action_ball_r10_validate_opt_in_lane(
        *,
        runtime_obs_mode: str,
        require_exact_resume_state: bool,
        training_contract_schema_version: object,
        training_contract_sha256: object,
        log_dir: object,
        adapter: object,
        capsule: object,
    ) -> None:
        """Reject accidental use outside the exact admitted ActionBall lanes."""

        if adapter is None:
            raise RuntimeError("R10 cold restore requires an explicit full-MDP adapter")
        if runtime_obs_mode not in (
            "action_ball_a211",
            "action_ball_c211",
            "action_ball_full_mdp",
        ):
            raise RuntimeError(
                "R10 runner adapter is restricted to exact A211/C211/full-MDP"
            )
        if require_exact_resume_state is not True or log_dir is None:
            raise RuntimeError(
                "R10 runner adapter requires the strict exact-resume training contract"
            )
        if (
            type(training_contract_schema_version) is not int
            or type(training_contract_sha256) is not str
            or training_contract_schema_version
            != TRAINING_CONTRACT_SCHEMA_VERSION
            or len(training_contract_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in training_contract_sha256
            )
        ):
            raise RuntimeError("R10 runner adapter requires an immutable contract pin")
        if capsule is not None and type(capsule) is not ActionBallR10ColdRestoreCapsule:
            raise RuntimeError("R10 cold restore capsule has the wrong typed schema")

    @staticmethod
    def _action_ball_r10_adapter_registry(adapter: object):
        r10 = _action_ball_r10_module()
        if r10.RUNTIME_WIRING is not False:
            raise RuntimeError("R10 pure seam unexpectedly claims runtime wiring")
        provider = getattr(adapter, "action_ball_r10_registry", None)
        if not callable(provider):
            raise RuntimeError("R10 adapter lacks action_ball_r10_registry()")
        registry = provider()
        r10.validate_r10_owner_registry(
            registry,
            engine=r10.OwnerEngine.ISAAC,
        )
        return registry

    @staticmethod
    def _action_ball_r10_validate_adapter_api_authority(
        adapter: object,
        *,
        training_launch_claim_sha256: object,
    ) -> Callable[[object], object]:
        """Cross-pin the frozen adapter to these loaded runner source bytes."""

        adapter_module = importlib.import_module(
            "action_ball_full_mdp_env_checkpoint_adapter"
        )
        authority_type = getattr(
            adapter_module,
            "RunnerCheckpointApiAuthority",
            None,
        )
        authority = getattr(adapter, "_runner_api", None)
        try:
            runner_source = pathlib.Path(__file__).resolve(strict=True)
            if not runner_source.is_file():
                raise OSError("runner source is not a regular file")
            runner_source_sha256 = hashlib.sha256(
                runner_source.read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise RuntimeError("R10 cannot bind loaded runner source bytes") from exc
        if (
            type(authority_type) is not type
            or type(authority) is not authority_type
            or authority.schema_version
            != getattr(adapter_module, "SCHEMA_VERSION", None)
            or type(authority.schema_version) is not int
            or authority.kind
            != "action_ball_r10_runner_checkpoint_api_authority_v1"
            or authority.source_sha256 != runner_source_sha256
            or type(authority.source_sha256) is not str
            or training_launch_claim_sha256 is None
            or authority.training_launch_claim_sha256
            != training_launch_claim_sha256
            or authority.boundary_handoff_type
            is not ActionBallR10RunnerBoundaryHandoff
            or authority.save_authority_type
            is not ActionBallR10RunnerSaveAuthority
        ):
            raise RuntimeError(
                "R10 adapter runner API/source authority differs from loaded runner"
            )
        callback = getattr(adapter, "action_ball_r10_restore_complete", None)
        if not callable(callback):
            raise RuntimeError(
                "R10 adapter lacks action_ball_r10_restore_complete()"
            )
        return callback

    @staticmethod
    def _action_ball_r10_preflight_cold_capsule(
        *,
        runtime_obs_mode: str,
        runtime_num_envs: object,
        runtime_num_steps_per_env: object,
        training_contract_sha256: object,
        registry: object,
        capsule: ActionBallR10ColdRestoreCapsule,
    ) -> Tuple[object, dict]:
        """Verify every external pin before RSL-RL can call the live getter."""

        if runtime_obs_mode == "action_ball_full_mdp":
            raise RuntimeError(
                "fresh full-MDP R10 cold restore remains fail-closed until its "
                "dynamic observation/noise frontier is durably owned"
            )

        if (
            capsule.schema_version != _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION
            or type(capsule.schema_version) is not int
            or capsule.kind != _ACTION_BALL_R10_COLD_RESTORE_CAPSULE_KIND
            or type(capsule.checkpoint_bytes) is not bytes
            or not capsule.checkpoint_bytes
        ):
            raise RuntimeError("R10 cold restore capsule schema/kind/bytes differ")
        r10 = _action_ball_r10_module()
        verified = r10.verify_checkpoint_candidate(
            capsule.checkpoint_bytes,
            expected_external_pins=capsule.expected_external_pins,
            expected_registry=registry,
        )
        expected_family = (
            r10.PolicyFamily.A
            if runtime_obs_mode == "action_ball_a211"
            else r10.PolicyFamily.C
        )
        if (
            verified.engine is not r10.OwnerEngine.ISAAC
            or verified.family is not expected_family
        ):
            raise RuntimeError("R10 cold checkpoint engine/family differs from runner")
        if verified.immutable_pins.contract_sha256 != training_contract_sha256:
            raise RuntimeError("R10 cold checkpoint differs from runner contract pin")
        by_owner = {state.owner_id: state for state in verified.owner_states}
        envelope = by_owner.get("trainer.rollout_frontier")
        if envelope is None:
            raise RuntimeError("R10 cold checkpoint lacks trainer.rollout_frontier")
        frontier = _action_ball_r10_validate_frontier_payload(
            _action_ball_r10_torch_load_bytes(envelope.payload),
            expected_update_index=verified.boundary.update_index,
        )
        if (
            type(runtime_num_envs) is not int
            or runtime_num_envs <= 0
            or frontier["num_envs"] != runtime_num_envs
            or type(runtime_num_steps_per_env) is not int
            or runtime_num_steps_per_env <= 0
            or frontier["num_steps_per_env"] != runtime_num_steps_per_env
            or tuple(world.world_id for world in verified.boundary.worlds)
            != tuple(range(runtime_num_envs))
        ):
            raise RuntimeError(
                "R10 cold checkpoint rollout/world identity differs from runner"
            )
        expected_actor_width = (
            _A211_ACTOR_WIDTH
            if runtime_obs_mode == "action_ball_a211"
            else _C211_ACTOR_WIDTH
        )
        expected_critic_width = (
            _A211_CRITIC_WIDTH
            if runtime_obs_mode == "action_ball_a211"
            else _C211_CRITIC_WIDTH
        )
        if (
            frontier["final_normalized_actor_observations"].shape[1]
            != expected_actor_width
            or frontier["final_normalized_critic_observations"].shape[1]
            != expected_critic_width
        ):
            raise RuntimeError("R10 cold frontier observation widths differ")
        return verified, frontier

    def bind_action_ball_r10_checkpoint_adapter(
        self,
        adapter: object,
        *,
        _prevalidated_registry: object = None,
    ) -> None:
        """One-shot opt-in; absence preserves the historical runner path."""

        if getattr(self, "_action_ball_r10_checkpoint_adapter", None) is not None:
            raise RuntimeError("R10 checkpoint adapter is already bound")
        runtime_env = getattr(self.env, "unwrapped", self.env)
        runtime_cfg = getattr(runtime_env, "cfg", None)
        runtime_obs_mode = str(getattr(runtime_cfg, "obs_mode", "") or "")
        self._action_ball_r10_validate_opt_in_lane(
            runtime_obs_mode=runtime_obs_mode,
            require_exact_resume_state=self.require_exact_resume_state,
            training_contract_schema_version=self.training_contract_schema_version,
            training_contract_sha256=self.training_contract_sha256,
            log_dir=self.log_dir,
            adapter=adapter,
            capsule=None,
        )
        restore_complete_callback = (
            self._action_ball_r10_validate_adapter_api_authority(
                adapter,
                training_launch_claim_sha256=(
                    self.training_launch_claim_sha256
                ),
            )
        )
        expected_target_mode = (
            "action_ball_full_mdp"
            if runtime_obs_mode == "action_ball_full_mdp"
            else "action_ball"
        )
        if self._strict_exact_resume_target_mode() != expected_target_mode:
            raise RuntimeError(
                "R10 runner adapter target_mode differs from its exact observation lane"
            )
        registry = (
            _prevalidated_registry
            if _prevalidated_registry is not None
            else self._action_ball_r10_adapter_registry(adapter)
        )
        r10 = _action_ball_r10_module()
        r10.validate_r10_owner_registry(
            registry,
            engine=r10.OwnerEngine.ISAAC,
        )
        for name in (
            "action_ball_r10_complete_owners",
            "action_ball_r10_restore_complete",
            "action_ball_r10_post_update_authority",
            "action_ball_r10_publish",
        ):
            if not callable(getattr(adapter, name, None)):
                raise RuntimeError(f"R10 adapter lacks {name}()")
        if getattr(self.alg, "rnd", None) is not None:
            raise RuntimeError("R10 runner adapter does not yet own RND state")
        if getattr(self.alg, "lr_scheduler", None) is not None:
            raise RuntimeError(
                "R10 runner adapter found an unowned scheduler object"
            )
        self._action_ball_r10_checkpoint_adapter = adapter
        self._action_ball_r10_registry = registry
        self._action_ball_r10_restore_complete_callback = (
            restore_complete_callback
        )

    def _action_ball_r10_require_healthy(self) -> None:
        if getattr(self, "_action_ball_r10_runtime_poisoned", False):
            raise RuntimeError(
                "R10 runner runtime is poisoned; retry is forbidden: "
                + str(getattr(self, "_action_ball_r10_poison_reason", None))
            )

    def _action_ball_r10_poison(self, reason: str) -> None:
        self._action_ball_r10_runtime_poisoned = True
        if self._action_ball_r10_poison_reason is None:
            self._action_ball_r10_poison_reason = str(reason)

    def _action_ball_full_mdp_require_healthy(self) -> None:
        """Require the exact construction-bound owner before mutation."""

        owner = getattr(self, "_action_ball_full_mdp_runtime_owner", None)
        if owner is None:
            return
        if getattr(self, "_action_ball_full_mdp_boundary_poisoned", False):
            raise RuntimeError(
                "fresh full-MDP optimizer boundary is poisoned; retry is "
                "forbidden and checkpoint is forbidden: "
                + str(
                    getattr(
                        self,
                        "_action_ball_full_mdp_boundary_poison_reason",
                        None,
                    )
                )
            )
        if type(owner) is not getattr(
            self, "_action_ball_full_mdp_runtime_owner_type", None
        ) or owner is not getattr(
            self, "_action_ball_full_mdp_runtime_owner_identity", None
        ):
            raise RuntimeError("fresh full-MDP runtime owner identity changed")
        runtime_env = getattr(
            self, "_action_ball_full_mdp_runtime_env_identity", None
        )
        lease = getattr(
            self, "_action_ball_full_mdp_runtime_lease_identity", None
        )
        drain_getter = getattr(
            self, "_action_ball_full_mdp_drain_getter_callback", None
        )
        drain_owner = getattr(
            self, "_action_ball_full_mdp_drain_owner_identity", None
        )
        if (
            runtime_env is None
            or lease is None
            or not callable(drain_getter)
            or getattr(drain_getter, "__self__", None) is not runtime_env
            or drain_getter(lease) is not drain_owner
            or type(drain_owner)
            is not getattr(self, "_action_ball_full_mdp_drain_owner_type", None)
        ):
            raise RuntimeError(
                "fresh full-MDP lease-bound global drain identity changed"
            )
        if getattr(self, "_action_ball_full_mdp_run_mode", None) == (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            module = getattr(
                self, "_action_ball_full_mdp_lean_runtime_module", None
            )
            epoch_owner = getattr(
                self,
                "_action_ball_full_mdp_lean_epoch_owner_identity",
                None,
            )
            if (
                module is None
                or type(owner)
                is not getattr(module, "ActionBallFullMdpLeanRuntimeOwner", None)
                or drain_owner is not owner
                or getattr(owner, "epoch_owner", None) is not epoch_owner
                or type(epoch_owner)
                is not getattr(
                    getattr(module, "epoch_v1", None),
                    "ActionEpochOwner",
                    None,
                )
            ):
                raise RuntimeError(
                    "single_action_lean lean owner/epoch identity changed"
                )
        callback = getattr(
            self, "_action_ball_full_mdp_require_healthy_callback", None
        )
        if not callable(callback):
            raise RuntimeError(
                "fresh full-MDP runtime owner health callback is unavailable"
            )
        try:
            callback()
        except BaseException as exc:
            self._action_ball_full_mdp_boundary_poisoned = True
            self._action_ball_full_mdp_boundary_poison_reason = (
                "runtime owner health check failed: "
                f"{type(exc).__module__}.{type(exc).__qualname__}"
            )
            raise RuntimeError(
                "fresh full-MDP runtime owner is unhealthy; retry is "
                "forbidden and checkpoint is forbidden"
            ) from exc

    def _action_ball_full_mdp_poison_optimizer_boundary(
        self,
        prepared: object,
        *,
        update_index: int,
        reason: str,
    ) -> None:
        """Poison once without allowing a broken callback to mask the cause."""

        if getattr(self, "_action_ball_full_mdp_boundary_poisoned", False):
            return
        self._action_ball_full_mdp_boundary_poisoned = True
        self._action_ball_full_mdp_boundary_poison_reason = str(reason)
        callback = getattr(
            self, "_action_ball_full_mdp_poison_callback", None
        )
        if not callable(callback):
            self._action_ball_full_mdp_boundary_poison_callback_failure = (
                "poison callback unavailable"
            )
            return
        try:
            callback(
                prepared,
                update_index=update_index,
                reason=str(reason),
            )
        except BaseException as exc:
            self._action_ball_full_mdp_boundary_poison_callback_failure = (
                f"{type(exc).__module__}.{type(exc).__qualname__}"
            )

    def _action_ball_full_mdp_require_checkpointable(self) -> None:
        """Forbid every checkpoint after or during an uncertain boundary."""

        if getattr(self, "_action_ball_full_mdp_runtime_owner", None) is None:
            return
        if getattr(self, "_action_ball_full_mdp_run_mode", None) == (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            owner = self._action_ball_full_mdp_runtime_owner
            capture = getattr(owner, "_capture_private_carry_for_save", None)
            if not callable(capture) or getattr(capture, "__self__", None) is not owner:
                raise RuntimeError("single_action_lean private carry preflight is unavailable")
            capture()
            raise RuntimeError(
                "single_action_lean private carry cannot authorize persistence"
            )
        if getattr(self, "_action_ball_full_mdp_boundary_poisoned", False):
            raise RuntimeError(
                "fresh full-MDP optimizer boundary is poisoned; checkpoint "
                "is forbidden"
            )
        if getattr(self, "_action_ball_full_mdp_boundary_active", False):
            raise RuntimeError(
                "fresh full-MDP optimizer boundary is active; checkpoint is "
                "forbidden before acknowledgement"
            )
        self._action_ball_full_mdp_require_healthy()

    @staticmethod
    def _action_ball_full_mdp_projection_primitives(
        projection: object,
    ) -> dict:
        """Read only the global owner's documented clone-only primitives."""

        names = (
            "schema_version",
            "kind",
            "num_envs",
            "device_type",
            "device_index",
            "owner_order",
            "schema_identity",
            "next_update_index",
            "operation_sequence",
            "drain_sequence",
            "last_completed_environment_steps",
            "mutation_version_highwaters",
            "update_index",
            "completed_environment_steps",
        )
        try:
            return {name: getattr(projection, name) for name in names}
        except BaseException as exc:
            raise RuntimeError(
                "fresh full-MDP runner frontier projection surface differs"
            ) from exc

    def _action_ball_full_mdp_preflight_durable_wal(self) -> pathlib.Path:
        """Create the per-rank append-only WAL and fsync its directory."""

        if getattr(self, "_action_ball_full_mdp_run_mode", None) != (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            raise RuntimeError(
                "typed ActionEpoch durable WAL is restricted to the lean lane"
            )
        log_dir = getattr(self, "log_dir", None)
        if type(log_dir) is not str or not log_dir:
            raise RuntimeError("typed ActionEpoch durable WAL requires log_dir")
        parent_directory = pathlib.Path(log_dir)
        parent_stat = os.lstat(parent_directory)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RuntimeError(
                "typed ActionEpoch durable WAL log_dir is not an exact directory"
            )
        rank = self._joint_safety_rank()
        directory = parent_directory / "action_ball_epoch_durable_wal"
        path = directory / f"rank_{rank:04d}.jsonl"
        installed = getattr(
            self, "_action_ball_full_mdp_durable_wal_path", None
        )
        if installed is not None:
            if type(installed) is not pathlib.PosixPath or installed != path:
                raise RuntimeError(
                    "typed ActionEpoch durable WAL rank/path changed"
                )
            flags = os.O_RDWR | os.O_APPEND
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags)
            try:
                file_stat = os.fstat(fd)
                expected_identity = getattr(
                    self,
                    "_action_ball_full_mdp_durable_wal_identity",
                    None,
                )
                expected_size = getattr(
                    self,
                    "_action_ball_full_mdp_durable_wal_size",
                    None,
                )
                if (
                    not stat.S_ISREG(file_stat.st_mode)
                    or (file_stat.st_dev, file_stat.st_ino)
                    != expected_identity
                    or type(expected_size) is not int
                    or expected_size < 0
                    or file_stat.st_size != expected_size
                    or (
                        expected_size > 0
                        and os.pread(fd, 1, expected_size - 1) != b"\n"
                    )
                ):
                    raise RuntimeError(
                        "typed ActionEpoch durable WAL inode/content frontier changed"
                    )
            except BaseException:
                try:
                    os.close(fd)
                except BaseException:
                    pass
                raise
            else:
                os.close(fd)
            return installed

        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_stat = os.lstat(directory)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError(
                "typed ActionEpoch durable WAL directory is not a directory"
            )
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(
                    "typed ActionEpoch durable WAL path is not a regular file"
                )
            os.fsync(fd)
        except BaseException:
            try:
                os.close(fd)
            except BaseException:
                pass
            raise
        else:
            os.close(fd)

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        directory_fd = os.open(directory, directory_flags)
        try:
            os.fsync(directory_fd)
        except BaseException:
            try:
                os.close(directory_fd)
            except BaseException:
                pass
            raise
        else:
            os.close(directory_fd)

        parent_fd = os.open(parent_directory, directory_flags)
        try:
            os.fsync(parent_fd)
        except BaseException:
            try:
                os.close(parent_fd)
            except BaseException:
                pass
            raise
        else:
            os.close(parent_fd)
        self._action_ball_full_mdp_durable_wal_path = path
        self._action_ball_full_mdp_durable_wal_identity = (
            file_stat.st_dev,
            file_stat.st_ino,
        )
        self._action_ball_full_mdp_durable_wal_size = 0
        self._action_ball_full_mdp_durable_wal_segment_id = (
            f"{file_stat.st_dev:x}:{file_stat.st_ino:x}"
        )
        return path

    def _action_ball_full_mdp_encode_durable_pending_ack(
        self,
        record: dict,
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> tuple[bytes, bytes]:
        """Canonicalize stdout telemetry and its pre-destructive-ACK row."""

        if type(record) is not dict:
            raise RuntimeError("typed ActionEpoch pending summary is not a dict")
        if (
            record.get("ppo_update") != update_index
            or record.get("completed_environment_steps")
            != completed_environment_steps
        ):
            raise RuntimeError("typed ActionEpoch pending chronology differs")
        return _action_ball_full_mdp_durable_wal_module().encode_pending(
            segment_id=self._action_ball_full_mdp_durable_wal_segment_id,
            rank=self._joint_safety_rank(),
            telemetry=record,
        )

    def _action_ball_full_mdp_append_durable_wal(
        self,
        path: pathlib.Path,
        line: bytes,
    ) -> tuple[int, int]:
        """Append, flush, fsync, and close one exact WAL line before ACK."""

        if (
            type(path) is not pathlib.PosixPath
            or path
            is not getattr(self, "_action_ball_full_mdp_durable_wal_path", None)
            or type(line) is not bytes
            or not line.endswith(b"\n")
            or line.count(b"\n") != 1
        ):
            raise RuntimeError("typed ActionEpoch durable WAL append ABI differs")
        handle = open(path, "ab", buffering=0)
        try:
            file_stat = os.fstat(handle.fileno())
            expected_size = getattr(
                self, "_action_ball_full_mdp_durable_wal_size", None
            )
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or (file_stat.st_dev, file_stat.st_ino)
                != getattr(
                    self,
                    "_action_ball_full_mdp_durable_wal_identity",
                    None,
                )
                or type(expected_size) is not int
                or expected_size < 0
                or file_stat.st_size != expected_size
            ):
                raise RuntimeError(
                    "typed ActionEpoch durable WAL file identity changed"
                )
            written = handle.write(line)
            if type(written) is not int or written != len(line):
                raise OSError(
                    "typed ActionEpoch durable WAL append was a short write"
                )
            handle.flush()
            if os.fstat(handle.fileno()).st_size != expected_size + len(line):
                raise RuntimeError(
                    "typed ActionEpoch durable WAL append frontier changed"
                )
            os.fsync(handle.fileno())
        except BaseException:
            try:
                current = os.fstat(handle.fileno())
                if (
                    type(expected_size) is int
                    and current.st_size > expected_size
                    and (current.st_dev, current.st_ino)
                    == self._action_ball_full_mdp_durable_wal_identity
                ):
                    os.ftruncate(handle.fileno(), expected_size)
            except BaseException:
                pass
            try:
                handle.close()
            except BaseException:
                pass
            raise
        else:
            handle.close()
        end = expected_size + len(line)
        self._action_ball_full_mdp_durable_wal_size = end
        return expected_size, end

    def _action_ball_full_mdp_prepare_epoch_ack_record(
        self,
        summary: object,
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> dict:
        """Purely validate and encode-ready project the still-pending summary."""

        module = getattr(self, "_action_ball_full_mdp_lean_runtime_module", None)
        summary_type = getattr(module, "ActionEpochPpoBoundarySummary", None)
        frontier_type = getattr(module, "EpochDrainFrontier", None)
        active = getattr(
            self, "_action_ball_full_mdp_active_drain_chronology", None
        )
        if (
            type(summary_type) is not type
            or type(frontier_type) is not type
            or type(summary) is not summary_type
            or type(getattr(summary, "frontier", None)) is not frontier_type
            or type(active) is not _ActionBallFullMdpRunnerDrainChronology
            or active.update_index != update_index
            or active.completed_environment_steps != completed_environment_steps
            or active.optimizer_returned is not True
            or active.post_update_acknowledged is not False
            or active.telemetry_emitted is not False
        ):
            raise RuntimeError(
                "diagnostic epoch ACK summary is foreign, stale, or duplicated"
            )
        frontier = summary.frontier
        device = getattr(self, "_action_ball_full_mdp_drain_device", None)
        expected_slots = getattr(
            self, "_action_ball_full_mdp_lean_shot_slot_capacity", None
        )
        exact_ints = (
            frontier.num_envs,
            frontier.shot_slot_capacity,
            frontier.update_index,
            frontier.next_update_index,
            frontier.completed_environment_steps,
            frontier.operation_sequence,
            frontier.drain_sequence,
            frontier.start_commit,
            frontier.end_commit,
        )
        if (
            any(type(value) is not int or value < 0 for value in exact_ints)
            or frontier.schema_version != 10
            or frontier.kind != "action_ball_epoch_ppo_boundary_summary_v10"
            or frontier.num_envs != int(self.env.num_envs)
            or frontier.shot_slot_capacity != expected_slots
            or not isinstance(device, torch.device)
            or frontier.device_type != device.type
            or frontier.device_index != device.index
            or frontier.update_index != update_index
            or frontier.next_update_index != update_index + 1
            or frontier.completed_environment_steps != completed_environment_steps
            or frontier.end_commit < frontier.start_commit
            or frontier.diagnostic_unauthorized is not True
        ):
            raise RuntimeError(
                "diagnostic epoch ACK frontier differs from runner chronology"
            )
        previous = getattr(
            self, "_action_ball_full_mdp_last_drain_chronology", None
        )
        if previous is None:
            exact = (
                frontier.operation_sequence == 1
                and frontier.drain_sequence == 1
                and frontier.start_commit == 0
            )
        else:
            previous_summary = previous.projection
            previous_frontier = getattr(previous_summary, "frontier", None)
            exact = (
                type(previous_summary) is summary_type
                and type(previous_frontier) is frontier_type
                and previous.post_update_acknowledged is True
                and previous.telemetry_emitted is True
                and update_index == previous.update_index + 1
                and completed_environment_steps
                == previous.completed_environment_steps
                + int(self.env.num_envs) * int(self.num_steps_per_env)
                and frontier.operation_sequence == previous.operation_sequence + 1
                and frontier.drain_sequence == previous.drain_sequence + 1
                and frontier.start_commit == previous_frontier.end_commit
            )
        if not exact:
            raise RuntimeError(
                "diagnostic epoch ACK frontier did not advance exactly once"
            )

        completed_type = getattr(
            getattr(module, "drain_v2", None),
            "CompletedActionEpochShot",
            None,
        )
        terminal_shot_type = getattr(
            getattr(module, "drain_v2", None),
            "TerminalActionEpochShot",
            None,
        )
        evidence_type = getattr(
            getattr(module, "drain_v2", None),
            "ActionEpochShotEvidence",
            None,
        )
        lifecycle_flags = getattr(
            getattr(module, "drain_v2", None),
            "SHOT_LIFECYCLE_FLAGS",
            None,
        )
        completed = getattr(summary, "completed_shots", None)
        common_integer_fields = (
            "env_row",
            "slot_index",
            "reset_generation",
            "ball_generation",
            "action_uid",
            "action_slot",
            "shot_index",
            "task_identity",
            "outcome_identity",
            "ball_identity",
            "motion_close_reason",
            "settlement_step",
            "payment_step",
        )

        def encode_evidence(evidence):
            if (
                type(evidence_type) is not type
                or type(lifecycle_flags) is not tuple
                or type(evidence) is not evidence_type
            ):
                raise RuntimeError("diagnostic shot-evidence telemetry ABI differs")
            integer_names = (
                "lifecycle_bits", "r03_valid_bits", "r03_source_step",
                "physical_valid_bits",
                "physical_actor_pair_contact_source_step", "r06_valid_bits",
                "r06_outcome_code", "r06_predicate_bits", "r07_valid_bits",
                "r07_qualified_source_step",
                "r07_first_ready_source_step",
            )
            value = {name: getattr(evidence, name) for name in integer_names}
            if tuple(evidence.__dataclass_fields__) != integer_names or any(
                type(item) is not int for item in value.values()
            ):
                raise RuntimeError("diagnostic shot-evidence telemetry ABI differs")
            lifecycle = {
                name: bool(value["lifecycle_bits"] & (1 << ordinal))
                for ordinal, name in enumerate(lifecycle_flags)
            }
            if (
                value["lifecycle_bits"] < 0
                or value["lifecycle_bits"] >> len(lifecycle_flags)
            ):
                raise RuntimeError("diagnostic shot-evidence telemetry values differ")
            return {
                **value,
                "lifecycle": lifecycle,
                "contact_face": {"availability": "not_produced"},
                "recovery_horizon": {"availability": "not_produced"},
            }

        def encode_shot(shot, *, terminal: bool):
            expected_type = terminal_shot_type if terminal else completed_type
            if type(expected_type) is not type or type(shot) is not expected_type:
                raise RuntimeError("diagnostic shot telemetry ABI differs")
            integers = {
                name: getattr(shot, name) for name in common_integer_fields
            }
            family = shot.stroke_family
            attributed = shot.action_attribution_valid
            target_x_m = shot.target_x_m
            target_y_m = shot.target_y_m
            evidence = encode_evidence(shot.evidence)
            lifecycle = evidence["lifecycle"]
            if (
                any(type(value) is not int for value in integers.values())
                or type(target_x_m) is not float
                or type(target_y_m) is not float
                or type(family) is not str
                or type(attributed) is not bool
                or (family in ("forehand", "backhand")) is not attributed
                or family not in ("unknown", "forehand", "backhand")
                or not math.isfinite(target_x_m)
                or not math.isfinite(target_y_m)
                or not 0 <= integers["env_row"] < frontier.num_envs
                or not 0 <= integers["slot_index"] < expected_slots
                or integers["reset_generation"] < 0
                or integers["ball_generation"] < 0
                or integers["action_uid"] <= 0
                or integers["action_slot"] < 0
                or integers["shot_index"] <= 0
                or integers["task_identity"] <= 0
                or integers["outcome_identity"] <= 0
                or integers["ball_identity"] <= 0
                or integers["motion_close_reason"] not in (-1, 1, 2)
                or integers["settlement_step"] < -1
                or integers["payment_step"] < -1
                or (integers["payment_step"] >= 0 and integers["settlement_step"] < 0)
                or (integers["payment_step"] >= 0
                    and integers["payment_step"] < integers["settlement_step"])
                or lifecycle["reveal_committed"] is not True
                or lifecycle["motion_closed"]
                != (integers["motion_close_reason"] != -1)
                or lifecycle["outcome_settled"]
                != (integers["settlement_step"] >= 0)
                or lifecycle["payment_recorded"]
                != (integers["payment_step"] >= 0)
            ):
                raise RuntimeError("diagnostic shot telemetry values differ")
            return {
                **integers,
                "target_x_m": target_x_m if lifecycle["physical_launched"] else None,
                "target_y_m": target_y_m if lifecycle["physical_launched"] else None,
                "stroke_family": family,
                "action_attribution_valid": attributed,
                "evidence": evidence,
            }

        completed_rows = []
        if type(completed_type) is not type or type(completed) is not tuple:
            raise RuntimeError("diagnostic completed-shot telemetry ABI differs")
        for shot in completed:
            row = encode_shot(shot, terminal=False)
            retirement_step = shot.retirement_step
            if (
                type(retirement_step) is not int
                or row["motion_close_reason"] not in (1, 2)
                or row["settlement_step"] < 0
                or row["payment_step"] < row["settlement_step"]
                or retirement_step < row["payment_step"]
                or not all(row["evidence"]["lifecycle"][name] for name in (
                    "physical_launched", "outcome_settled", "payment_recorded",
                ))
            ):
                raise RuntimeError(
                    "diagnostic completed-shot telemetry values differ"
                )
            completed_rows.append({**row, "retirement_step": retirement_step})
        lifecycle = summary.lifecycle
        lifecycle_values = tuple(
            getattr(lifecycle, name)
            for name in (
                "playback_started_rows", "closed_unplayed_rows",
                "physical_launch_rows", "outcome_settled_rows",
                "payment_recorded_rows", "retired_rows", "terminal_shot_rows",
            )
        ) if type(lifecycle) is getattr(
            getattr(module, "drain_v2", None), "LifecycleEdgeCounts", None
        ) else ()
        if (
            len(lifecycle_values) != 7
            or any(type(value) is not int or value < 0 for value in lifecycle_values)
            or len(completed) != lifecycle.retired_rows
        ):
            raise RuntimeError("diagnostic completed-shot retirement differs")

        terminal_shots = getattr(summary, "terminal_shots", None)
        terminal_shot_rows = []
        if type(terminal_shot_type) is not type or type(terminal_shots) is not tuple:
            raise RuntimeError("diagnostic terminal-shot telemetry ABI differs")
        for shot in terminal_shots:
            row = encode_shot(shot, terminal=True)
            reset_values = {
                name: getattr(shot, name)
                for name in (
                    "reset_generation_after", "reset_common_step",
                    "reset_episode_tick", "reset_reason_bits",
                )
            }
            if (
                any(type(value) is not int for value in reset_values.values())
                or reset_values["reset_generation_after"] <= row["reset_generation"]
                or reset_values["reset_common_step"] < 1
                or reset_values["reset_episode_tick"] < 1
                or reset_values["reset_reason_bits"] == 0
                or reset_values["reset_reason_bits"] & ~31
            ):
                raise RuntimeError("diagnostic terminal-shot telemetry values differ")
            terminal_shot_rows.append({**row, **reset_values})
        if (
            type(lifecycle.terminal_shot_rows) is not int
            or lifecycle.terminal_shot_rows < 0
            or len(terminal_shots) != lifecycle.terminal_shot_rows
        ):
            raise RuntimeError("diagnostic terminal-shot count differs")

        opportunity_type = getattr(
            getattr(module, "drain_v2", None), "D05ActionOpportunity", None
        )
        settlement = summary.settlement
        opportunities = getattr(summary, "action_opportunities", None)
        opportunity_rows = []
        if type(opportunity_type) is not type or type(opportunities) is not tuple:
            raise RuntimeError("diagnostic action-opportunity telemetry ABI differs")
        for row in opportunities:
            if type(row) is not opportunity_type:
                raise RuntimeError("diagnostic action-opportunity telemetry ABI differs")
            integers = (row.env_row, row.slot_index, row.action_uid, row.action_slot)
            flags = (
                row.attribution_valid, row.selected, row.accepted,
                row.censored, row.rejected, row.deferred,
            )
            if (
                any(type(value) is not int for value in integers)
                or any(type(value) is not bool for value in flags)
                or not 0 <= row.env_row < frontier.num_envs
                or not 0 <= row.slot_index < expected_slots
                or sum(flags[2:]) != 1
                or (not row.selected and not row.censored)
                or type(row.stroke_family) is not str
                or (row.stroke_family in ("forehand", "backhand")) is not row.attribution_valid
                or row.stroke_family not in ("unknown", "forehand", "backhand")
            ):
                raise RuntimeError("diagnostic action-opportunity telemetry values differ")
            opportunity_rows.append({name: getattr(row, name) for name in row.__dataclass_fields__})
        settlement_values = (
            settlement.transactions, settlement.due_rows, settlement.selected_rows, settlement.accepted,
            settlement.censored, settlement.rejected, settlement.deferred, settlement.not_ready,
        )
        if (
            type(settlement) is not getattr(
                getattr(module, "drain_v2", None), "D05SettlementCounts", None
            )
            or any(type(value) is not int or value < 0 for value in settlement_values)
            or settlement_values[1:7] != (
                len(opportunities),
                *(sum(getattr(row, field) for row in opportunities) for field in (
                    "selected", "accepted", "censored", "rejected", "deferred",
                )),
            )
            or settlement.not_ready != settlement.deferred
        ):
            raise RuntimeError("diagnostic action-opportunity settlement differs")

        reset_type = getattr(
            getattr(module, "drain_v2", None),
            "ResetTelemetry",
            None,
        )
        terminal_resets = getattr(summary, "terminal_resets", None)
        reset_fields = (
            "env_row",
            "reset_generation",
            "common_step",
            "episode_tick",
            "reason_bits",
        )
        terminal_reset_rows = []
        if type(reset_type) is not type or type(terminal_resets) is not tuple:
            raise RuntimeError("diagnostic terminal-reset telemetry ABI differs")
        for reset in terminal_resets:
            if type(reset) is not reset_type:
                raise RuntimeError(
                    "diagnostic terminal-reset telemetry ABI differs"
                )
            integers = {name: getattr(reset, name) for name in reset_fields}
            if (
                any(type(value) is not int for value in integers.values())
                or not 0 <= integers["env_row"] < frontier.num_envs
                or integers["reset_generation"] < 1
                or integers["common_step"] < 1
                or integers["episode_tick"] < 1
                or integers["reason_bits"] == 0
                or integers["reason_bits"] & ~31
            ):
                raise RuntimeError(
                    "diagnostic terminal-reset telemetry values differ"
                )
            terminal_reset_rows.append(integers)

        reset_reason_counts = {
            "terminal_reset_reason_time_out_count": 0,
            "terminal_reset_reason_base_fell_tilt_count": 0,
            "terminal_reset_reason_base_too_low_count": 0,
            "terminal_reset_reason_joint_qdes_forbidden_count": 0,
            "terminal_reset_reason_robot_hit_table_count": 0,
        }
        for reset in terminal_reset_rows:
            for name, bit in zip(reset_reason_counts, (1, 2, 4, 8, 16)):
                reset_reason_counts[name] += int(bool(reset["reason_bits"] & bit))
        reset_index = {
            (
                reset["env_row"], reset["reset_generation"],
                reset["common_step"], reset["episode_tick"],
                reset["reason_bits"],
            )
            for reset in terminal_reset_rows
        }
        if (
            len({
                (
                    row["env_row"], row["slot_index"],
                    *(row[name] for name in (
                        "reset_generation", "ball_generation", "action_uid",
                        "action_slot", "shot_index", "task_identity",
                        "outcome_identity", "ball_identity",
                    )),
                )
                for row in terminal_shot_rows
            }) != len(terminal_shot_rows)
            or any(
                (
                    row["env_row"], row["reset_generation_after"],
                    row["reset_common_step"], row["reset_episode_tick"],
                    row["reset_reason_bits"],
                ) not in reset_index
                for row in terminal_shot_rows
            )
        ):
            raise RuntimeError("diagnostic terminal-shot/reset relationship differs")
        closure_keys = [
            (
                row["env_row"], row["slot_index"],
                *(row[name] for name in (
                    "reset_generation", "ball_generation", "action_uid",
                    "action_slot", "shot_index", "task_identity",
                    "outcome_identity", "ball_identity",
                )),
            )
            for row in (*completed_rows, *terminal_shot_rows)
        ]
        if len(set(closure_keys)) != len(closure_keys):
            raise RuntimeError("diagnostic shot closure identity differs")

        commits = summary.reveal_commit
        faults = summary.owner_faults
        continuation = summary.continuation
        lean_rewards = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_rewards"
        )
        milestone = summary.milestone.as_json(
            tuple(lean_rewards.MANAGER_NAMES)
        )
        record = {
            "schema_version": (
                _ACTION_BALL_EPOCH_UPDATE_ACK_TELEMETRY_SCHEMA_VERSION
            ),
            "kind": _ACTION_BALL_EPOCH_UPDATE_ACK_TELEMETRY_KIND,
            "diagnostic_unauthorized": True,
            "ppo_update": update_index,
            "completed_environment_steps": completed_environment_steps,
            "epoch_operation_sequence": frontier.operation_sequence,
            "epoch_drain_sequence": frontier.drain_sequence,
            "epoch_commit_start": frontier.start_commit,
            "epoch_commit_end": frontier.end_commit,
            "shot_slot_capacity": frontier.shot_slot_capacity,
            "d05_transactions": settlement.transactions,
            "d05_due_rows": settlement.due_rows,
            "d05_selected_rows": settlement.selected_rows,
            "d05_accepted_rows": settlement.accepted,
            "d05_censored_rows": settlement.censored,
            "d05_rejected_rows": settlement.rejected,
            "d05_deferred_rows": settlement.deferred,
            "d05_not_ready_rows": settlement.not_ready,
            "motion_committed_rows": commits.motion_committed_rows,
            "racket_committed_rows": commits.racket_committed_rows,
            "r05_committed_rows": commits.r05_committed_rows,
            "playback_started_rows": lifecycle.playback_started_rows,
            "closed_unplayed_rows": lifecycle.closed_unplayed_rows,
            "physical_launch_rows": lifecycle.physical_launch_rows,
            "outcome_settled_rows": lifecycle.outcome_settled_rows,
            "payment_recorded_rows": lifecycle.payment_recorded_rows,
            "retired_rows": lifecycle.retired_rows,
            "terminal_shot_rows": lifecycle.terminal_shot_rows,
            "attributed_fault_rows": faults.attributed_fault_rows,
            "active_before": continuation.active_before,
            "active_after": continuation.active_after,
            "awaiting_playback_after": continuation.awaiting_playback_after,
            "awaiting_outcome_after": continuation.awaiting_outcome_after,
            "awaiting_payment_after": continuation.awaiting_payment_after,
            "action_opportunities": opportunity_rows,
            "completed_shots": completed_rows,
            "terminal_shots": terminal_shot_rows,
            "terminal_resets": terminal_reset_rows,
            "terminal_reset_rows": len(terminal_reset_rows),
            "milestone": milestone,
            **reset_reason_counts,
        }
        return record

    def _action_ball_full_mdp_consume_epoch_ack_summary(
        self,
        summary: object,
        *,
        update_index: int,
        completed_environment_steps: int,
        canonical_ack_json: bytes,
    ) -> dict:
        """Install chronology after owner ACK and mirror exact durable content."""

        module = getattr(self, "_action_ball_full_mdp_lean_runtime_module", None)
        summary_type = getattr(module, "ActionEpochPpoBoundarySummary", None)
        frontier_type = getattr(module, "EpochDrainFrontier", None)
        active = getattr(
            self, "_action_ball_full_mdp_active_drain_chronology", None
        )
        frontier = getattr(summary, "frontier", None)
        if (
            type(summary_type) is not type
            or type(frontier_type) is not type
            or type(summary) is not summary_type
            or type(frontier) is not frontier_type
            or type(active) is not _ActionBallFullMdpRunnerDrainChronology
            or active.update_index != update_index
            or active.completed_environment_steps != completed_environment_steps
            or active.optimizer_returned is not True
            or active.post_update_acknowledged is not False
            or active.telemetry_emitted is not False
            or type(canonical_ack_json) is not bytes
            or not canonical_ack_json
        ):
            raise RuntimeError(
                "diagnostic epoch ACK install chronology is foreign or stale"
            )
        try:
            decoded = json.loads(canonical_ack_json.decode("utf-8"))
        except BaseException as exc:
            raise RuntimeError(
                "diagnostic epoch ACK canonical JSON changed after durability"
            ) from exc
        if (
            type(decoded) is not dict
            or decoded.get("schema_version")
            != _ACTION_BALL_EPOCH_UPDATE_ACK_TELEMETRY_SCHEMA_VERSION
            or decoded.get("kind")
            != _ACTION_BALL_EPOCH_UPDATE_ACK_TELEMETRY_KIND
            or decoded.get("ppo_update") != update_index
            or decoded.get("completed_environment_steps")
            != completed_environment_steps
            or decoded.get("epoch_operation_sequence")
            != frontier.operation_sequence
            or decoded.get("epoch_drain_sequence") != frontier.drain_sequence
            or decoded.get("epoch_commit_start") != frontier.start_commit
            or decoded.get("epoch_commit_end") != frontier.end_commit
        ):
            raise RuntimeError(
                "diagnostic epoch ACK canonical JSON chronology differs"
            )
        chronology = active._replace(
            post_update_acknowledged=True,
            operation_sequence=frontier.operation_sequence,
            drain_sequence=frontier.drain_sequence,
            projection=summary,
            telemetry_emitted=False,
        )
        self._action_ball_full_mdp_active_drain_chronology = chronology
        self._action_ball_full_mdp_last_drain_chronology = chronology
        marker_line = (
            "HOPE_ACTION_EPOCH_UPDATE_ACK_JSON="
            + canonical_ack_json.decode("utf-8")
            + "\n"
        )
        written = sys.stdout.write(marker_line)
        if type(written) is not int or written != len(marker_line):
            raise OSError("diagnostic epoch ACK stdout was a short write")
        sys.stdout.flush()
        chronology = chronology._replace(telemetry_emitted=True)
        self._action_ball_full_mdp_active_drain_chronology = chronology
        self._action_ball_full_mdp_last_drain_chronology = chronology
        return decoded

    @staticmethod
    def _action_ball_full_mdp_validate_drain_schema_identity(
        schema_identity: object,
    ) -> tuple:
        """Validate, but never derive, the exact global-owner drain schema.

        Scalar and per-environment fields retain their legacy three-tuple ABI.
        A bounded journal is admitted only as the global drain owner's exact
        four-tuple ``(name, "fixed", minimum, fixed_width)``.  In particular,
        the runner must not infer, flatten, truncate, or otherwise manufacture
        a width from ``num_envs`` or from any payload observed later.
        """

        if type(schema_identity) is not tuple:
            raise RuntimeError(
                "fresh full-MDP drain schema identity owner order differs"
            )
        owner_order = []
        for owner_record in schema_identity:
            if type(owner_record) is not tuple or len(owner_record) != 2:
                raise RuntimeError(
                    "fresh full-MDP drain schema identity differs"
                )
            owner_id, fields = owner_record
            if (
                type(owner_id) is not str
                or not owner_id
                or type(fields) is not tuple
                or not fields
            ):
                raise RuntimeError(
                    "fresh full-MDP drain schema identity differs"
                )
            owner_order.append(owner_id)
            for field in fields:
                if type(field) is not tuple or len(field) not in (3, 4):
                    raise RuntimeError(
                        "fresh full-MDP drain schema field identity differs"
                    )
                name, cardinality, minimum = field[:3]
                if (
                    type(name) is not str
                    or not name
                    or type(cardinality) is not str
                    or cardinality not in ("scalar", "per_env", "fixed")
                    or type(minimum) is not int
                    or minimum < 0
                ):
                    raise RuntimeError(
                        "fresh full-MDP drain schema field identity differs"
                    )
                if cardinality == "fixed":
                    if (
                        len(field) != 4
                        or type(field[3]) is not int
                        or field[3] <= 0
                    ):
                        raise RuntimeError(
                            "fresh full-MDP drain fixed schema width differs"
                        )
                elif len(field) != 3:
                    raise RuntimeError(
                        "fresh full-MDP drain non-fixed schema has a foreign width"
                    )
        if tuple(owner_order) != _ACTION_BALL_FULL_MDP_DRAIN_OWNER_ORDER:
            raise RuntimeError(
                "fresh full-MDP drain schema identity owner order differs"
            )
        return schema_identity

    def _action_ball_full_mdp_validate_projection(
        self,
        projection: object,
        *,
        receipt: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> dict:
        """Join global primitives against chronology witnessed by this runner."""

        if projection is None or receipt is None:
            raise RuntimeError(
                "fresh full-MDP runner frontier lacks an ACKed receipt projection"
            )
        value = self._action_ball_full_mdp_projection_primitives(projection)
        device = getattr(self, "_action_ball_full_mdp_drain_device", None)
        expected_completed = (
            (int(update_index) + 1)
            * int(self.env.num_envs)
            * int(self.num_steps_per_env)
        )
        if (
            type(update_index) is not int
            or update_index < 0
            or type(completed_environment_steps) is not int
            or completed_environment_steps != expected_completed
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["kind"] != _ACTION_BALL_FULL_MDP_DRAIN_CHECKPOINT_KIND
            or type(value["num_envs"]) is not int
            or value["num_envs"] != int(self.env.num_envs)
            or not isinstance(device, torch.device)
            or value["device_type"] != device.type
            or value["device_index"] != device.index
            or value["owner_order"]
            != _ACTION_BALL_FULL_MDP_DRAIN_OWNER_ORDER
            or type(value["next_update_index"]) is not int
            or value["next_update_index"] != update_index + 1
            or type(value["operation_sequence"]) is not int
            or value["operation_sequence"] <= 0
            or type(value["drain_sequence"]) is not int
            or value["drain_sequence"] <= 0
            or type(value["last_completed_environment_steps"]) is not int
            or value["last_completed_environment_steps"]
            != completed_environment_steps
            or type(value["update_index"]) is not int
            or value["update_index"] != update_index
            or type(value["completed_environment_steps"]) is not int
            or value["completed_environment_steps"]
            != completed_environment_steps
        ):
            raise RuntimeError(
                "fresh full-MDP runner/global drain frontier chronology differs"
            )
        self._action_ball_full_mdp_validate_drain_schema_identity(
            value["schema_identity"]
        )
        highwaters = value["mutation_version_highwaters"]
        if (
            type(highwaters) is not tuple
            or tuple(owner_id for owner_id, _version in highwaters)
            != _ACTION_BALL_FULL_MDP_DRAIN_OWNER_ORDER
            or any(
                type(version) is not int or version < 0
                for _owner_id, version in highwaters
            )
        ):
            raise RuntimeError(
                "fresh full-MDP drain mutation highwaters differ"
            )
        previous = getattr(
            self, "_action_ball_full_mdp_last_drain_chronology", None
        )
        if (
            previous is not None
            and value["schema_identity"]
            != self._action_ball_full_mdp_projection_primitives(
                previous.projection
            )["schema_identity"]
        ):
            raise RuntimeError(
                "fresh full-MDP drain schema identity drifted between updates"
            )
        if previous is None:
            if (
                value["operation_sequence"] != update_index + 1
                or value["drain_sequence"] != update_index + 1
            ):
                raise RuntimeError(
                    "fresh full-MDP runner drain chronology lacks its exact "
                    "restored-or-initial frontier"
                )
        elif (
            previous.post_update_acknowledged is not True
            or update_index != previous.update_index + 1
            or completed_environment_steps
            != previous.completed_environment_steps
            + int(self.env.num_envs) * int(self.num_steps_per_env)
            or value["operation_sequence"] != previous.operation_sequence + 1
            or value["drain_sequence"] != previous.drain_sequence + 1
        ):
            raise RuntimeError(
                "fresh full-MDP runner drain sequence did not advance exactly once"
            )
        return value

    def _action_ball_full_mdp_capture_acknowledged_projection(
        self,
        *,
        receipt: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> None:
        """Retain the legacy formal projection path only for R10 checkpoints."""

        callback = getattr(
            self, "_action_ball_full_mdp_projection_callback", None
        )
        drain_owner = getattr(
            self, "_action_ball_full_mdp_drain_owner_identity", None
        )
        active = getattr(
            self, "_action_ball_full_mdp_active_drain_chronology", None
        )
        if (
            getattr(self, "_action_ball_full_mdp_run_mode", None)
            != _ACTION_BALL_FULL_MDP_FORMAL_MODE
            or not callable(callback)
            or getattr(callback, "__self__", None) is not drain_owner
            or type(active) is not _ActionBallFullMdpRunnerDrainChronology
            or active.global_receipt is not receipt
            or active.update_index != update_index
            or active.completed_environment_steps != completed_environment_steps
            or active.optimizer_returned is not True
            or active.post_update_acknowledged is not False
        ):
            raise RuntimeError(
                "formal fresh full-MDP ACK chronology is absent or stale"
            )
        projection = callback(receipt)
        value = self._action_ball_full_mdp_validate_projection(
            projection,
            receipt=receipt,
            update_index=update_index,
            completed_environment_steps=completed_environment_steps,
        )
        chronology = active._replace(
            post_update_acknowledged=True,
            operation_sequence=value["operation_sequence"],
            drain_sequence=value["drain_sequence"],
            projection=projection,
        )
        self._action_ball_full_mdp_active_drain_chronology = chronology
        self._action_ball_full_mdp_last_drain_chronology = chronology

    def action_ball_full_mdp_runner_ppo_drain_frontier(
        self,
        boundary: object,
    ) -> str:
        """Independent trainer writer for the ACKed PPO-drain frontier."""

        self._action_ball_full_mdp_require_checkpointable()
        r10 = _action_ball_r10_module()
        if type(boundary) is not r10.CheckpointBoundary:
            raise RuntimeError(
                "fresh full-MDP runner frontier requires an exact R10 boundary"
            )
        r10.validate_checkpoint_boundary(boundary)
        chronology = getattr(
            self, "_action_ball_full_mdp_last_drain_chronology", None
        )
        handoff = getattr(self, "_action_ball_r10_last_boundary_handoff", None)
        if (
            type(chronology) is not _ActionBallFullMdpRunnerDrainChronology
            or chronology.optimizer_returned is not True
            or chronology.post_update_acknowledged is not True
            or chronology.global_receipt is None
            or chronology.projection is None
            or type(handoff) is not ActionBallR10RunnerBoundaryHandoff
            or handoff.completed_update_index != chronology.update_index
            or handoff.next_learning_iteration != chronology.update_index + 1
            or handoff.completed_environment_steps
            != int(self.num_steps_per_env)
            or boundary.update_index != chronology.update_index
            or boundary.rollout_storage_empty is not True
            or boundary.optimizer_in_flight is not False
            or boundary.gae_in_flight is not False
        ):
            raise RuntimeError(
                "fresh full-MDP runner frontier lacks the exact post-update chronology"
            )
        callback = self._action_ball_full_mdp_projection_callback
        repeated = callback(chronology.global_receipt)
        if repeated is not chronology.projection:
            raise RuntimeError(
                "fresh full-MDP drain projection identity changed on repeat validation"
            )
        value = self._action_ball_full_mdp_projection_primitives(repeated)
        if value != self._action_ball_full_mdp_projection_primitives(
            chronology.projection
        ):
            raise RuntimeError(
                "fresh full-MDP drain projection primitives changed"
            )
        payload = {
            "schema_version": value["schema_version"],
            "kind": value["kind"],
            "num_envs": value["num_envs"],
            "device_type": value["device_type"],
            "device_index": value["device_index"],
            "owner_order": value["owner_order"],
            "schema_identity": value["schema_identity"],
            "checkpoint_boundary_sha256": r10.boundary_sha256(boundary),
            "next_update_index": value["next_update_index"],
            "operation_sequence": value["operation_sequence"],
            "drain_sequence": value["drain_sequence"],
            "last_completed_environment_steps": value[
                "last_completed_environment_steps"
            ],
            "mutation_version_highwaters": value[
                "mutation_version_highwaters"
            ],
        }
        return _action_ball_full_mdp_canonical_sha256(payload)

    def _action_ball_r10_broadcast_restore_poison(
        self,
        owners: object,
        *,
        reason: str,
    ) -> None:
        """Attempt every registered poison callback without short-circuiting."""

        self._action_ball_r10_poison(reason)
        registry = self._action_ball_r10_registry
        attempted = []
        failures = []
        if (
            registry is None
            or type(owners) is not tuple
            or len(owners) != len(registry.owner_ids)
        ):
            failures.append("owner tuple unavailable or incomplete")
        else:
            for owner_id, owner in zip(registry.owner_ids, owners):
                attempted.append(owner_id)
                try:
                    callback = getattr(owner, "poison_restore", None)
                    if not callable(callback):
                        raise RuntimeError("poison_restore is not callable")
                    callback(reason)
                except BaseException as exc:
                    failures.append(
                        f"{owner_id}:{type(exc).__module__}."
                        f"{type(exc).__qualname__}"
                    )
        self._action_ball_r10_restore_poison_attempted_owner_ids = tuple(
            attempted
        )
        self._action_ball_r10_restore_poison_failures = tuple(failures)

    @staticmethod
    def _action_ball_r10_type_identity(value: object) -> str:
        return f"{type(value).__module__}.{type(value).__qualname__}"

    def _action_ball_r10_complete_owners(
        self,
        *,
        operation: str,
        runner_owners: Mapping[str, object],
        verified_checkpoint: object = None,
    ) -> Tuple[object, ...]:
        adapter = self._action_ball_r10_checkpoint_adapter
        provider = adapter.action_ball_r10_complete_owners
        result = provider(
            operation=operation,
            runner_owners=MappingProxyType(dict(runner_owners)),
            verified_checkpoint=verified_checkpoint,
        )
        registry = self._action_ball_r10_registry
        if type(result) is not tuple or len(result) != len(registry.owner_ids):
            raise RuntimeError("R10 adapter returned an incomplete owner tuple")
        if len({id(owner) for owner in result}) != len(result):
            raise RuntimeError("R10 adapter returned aliased owner objects")
        for expected_id, owner in zip(registry.owner_ids, result):
            descriptor = getattr(owner, "descriptor", None)
            if descriptor != registry.descriptor(expected_id):
                raise RuntimeError(
                    f"R10 adapter owner descriptor differs for {expected_id}"
                )
            if expected_id in runner_owners and owner is not runner_owners[expected_id]:
                raise RuntimeError(
                    f"R10 adapter replaced runner-owned state {expected_id}"
                )
        return result

    @staticmethod
    def _action_ball_r10_validate_module_state_dict(
        saved: object,
        current: Mapping[str, object],
        *,
        label: str,
    ) -> None:
        if not isinstance(saved, Mapping):
            raise RuntimeError(f"R10 {label} state_dict must be a mapping")
        if tuple(saved) != tuple(current):
            raise RuntimeError(f"R10 {label} state_dict keys/order differ")
        for name, current_value in current.items():
            saved_value = saved[name]
            if not torch.is_tensor(current_value) or not torch.is_tensor(saved_value):
                raise RuntimeError(f"R10 {label}.{name} is not a tensor")
            if (
                tuple(saved_value.shape) != tuple(current_value.shape)
                or saved_value.dtype != current_value.dtype
            ):
                raise RuntimeError(f"R10 {label}.{name} tensor ABI differs")
            if (
                saved_value.is_floating_point()
                and not bool(torch.isfinite(saved_value).all())
            ):
                raise RuntimeError(f"R10 {label}.{name} is non-finite")

    def _action_ball_r10_policy_state(self) -> dict:
        policy = getattr(getattr(self, "alg", None), "policy", None)
        state_getter = getattr(policy, "state_dict", None)
        if not callable(state_getter):
            raise RuntimeError("R10 trainer.policy lacks state_dict()")
        recurrent = getattr(policy, "is_recurrent", None)
        if type(recurrent) is not bool:
            raise RuntimeError("R10 policy is_recurrent must be an exact bool")
        training = getattr(policy, "training", None)
        if type(training) is not bool:
            raise RuntimeError("R10 policy training mode must be an exact bool")
        return {
            "schema_version": _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            "kind": "action_ball_r10_trainer_policy_v1",
            "policy_type": self._action_ball_r10_type_identity(policy),
            "is_recurrent": recurrent,
            "policy_training": training,
            "std_abi": self._policy_std_abi(),
            "model_state_dict": state_getter(),
        }

    def _action_ball_r10_validate_policy_state(self, value: object) -> dict:
        required = {
            "schema_version",
            "kind",
            "policy_type",
            "is_recurrent",
            "policy_training",
            "std_abi",
            "model_state_dict",
        }
        if type(value) is not dict or set(value) != required:
            raise RuntimeError("R10 trainer.policy payload keys differ")
        policy = self.alg.policy
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION
            or value["kind"] != "action_ball_r10_trainer_policy_v1"
            or value["policy_type"] != self._action_ball_r10_type_identity(policy)
            or type(value["is_recurrent"]) is not bool
            or value["is_recurrent"] is not getattr(policy, "is_recurrent", None)
            or type(value["policy_training"]) is not bool
            or value["std_abi"] != self._policy_std_abi()
        ):
            raise RuntimeError("R10 trainer.policy ABI differs")
        self._action_ball_r10_validate_module_state_dict(
            value["model_state_dict"],
            policy.state_dict(),
            label="trainer.policy",
        )
        return value

    def _action_ball_r10_install_policy_state(self, value: object) -> None:
        state = self._action_ball_r10_validate_policy_state(value)
        self.alg.policy.load_state_dict(state["model_state_dict"], strict=True)
        self.alg.policy.train(state["policy_training"])

    def _action_ball_r10_optimizer_state(self) -> dict:
        algorithm = getattr(self, "alg", None)
        optimizer = getattr(algorithm, "optimizer", None)
        if type(optimizer) is not torch.optim.Adam:
            raise RuntimeError("R10 RSL 2.3.1 lane requires exact torch.optim.Adam")
        if getattr(algorithm, "rnd", None) is not None:
            raise RuntimeError("R10 trainer owner does not support RND state")
        if getattr(algorithm, "lr_scheduler", None) is not None:
            raise RuntimeError(
                "R10 found a real lr_scheduler object without a typed owner"
            )
        schedule = getattr(algorithm, "schedule", None)
        learning_rate = getattr(algorithm, "learning_rate", None)
        desired_kl = getattr(algorithm, "desired_kl", None)
        if type(schedule) is not str or not schedule:
            raise RuntimeError("R10 PPO inline schedule is invalid")
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise RuntimeError("R10 PPO learning_rate is invalid")
        if desired_kl is not None and (
            type(desired_kl) not in (int, float)
            or not math.isfinite(float(desired_kl))
            or float(desired_kl) <= 0.0
        ):
            raise RuntimeError("R10 PPO desired_kl is invalid")
        group_lrs = tuple(
            float(group.get("lr"))
            if isinstance(group, Mapping)
            and type(group.get("lr")) in (int, float)
            and math.isfinite(float(group.get("lr")))
            and float(group.get("lr")) > 0.0
            else float("nan")
            for group in optimizer.param_groups
        )
        if not group_lrs or any(
            not math.isfinite(value) or value != float(learning_rate)
            for value in group_lrs
        ):
            raise RuntimeError("R10 optimizer group LR differs from PPO learning_rate")
        return {
            "schema_version": _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            "kind": "action_ball_r10_trainer_optimizer_schedule_v1",
            "algorithm_type": self._action_ball_r10_type_identity(algorithm),
            "optimizer_type": self._action_ball_r10_type_identity(optimizer),
            "scheduler_representation": "rsl_rl_2_3_1_inline_schedule_and_lr",
            "schedule": schedule,
            "desired_kl": None if desired_kl is None else float(desired_kl),
            "learning_rate": float(learning_rate),
            "param_group_lrs": group_lrs,
            "optimizer_state_dict": optimizer.state_dict(),
        }

    def _action_ball_r10_validate_optimizer_state(
        self,
        value: object,
        *,
        require_complete_adam: bool = True,
    ) -> dict:
        required = {
            "schema_version",
            "kind",
            "algorithm_type",
            "optimizer_type",
            "scheduler_representation",
            "schedule",
            "desired_kl",
            "learning_rate",
            "param_group_lrs",
            "optimizer_state_dict",
        }
        if type(value) is not dict or set(value) != required:
            raise RuntimeError("R10 trainer.optimizer_schedule payload keys differ")
        current = self._action_ball_r10_optimizer_state()
        for key in (
            "schema_version",
            "kind",
            "algorithm_type",
            "optimizer_type",
            "scheduler_representation",
            "schedule",
            "desired_kl",
        ):
            if value[key] != current[key] or type(value[key]) is not type(current[key]):
                raise RuntimeError(f"R10 trainer.optimizer_schedule {key} differs")
        learning_rate = value["learning_rate"]
        group_lrs = value["param_group_lrs"]
        if (
            type(learning_rate) is not float
            or not math.isfinite(learning_rate)
            or learning_rate <= 0.0
            or type(group_lrs) is not tuple
            or len(group_lrs) != len(self.alg.optimizer.param_groups)
            or any(type(item) is not float or item != learning_rate for item in group_lrs)
        ):
            raise RuntimeError("R10 trainer.optimizer_schedule LR frontier differs")
        optimizer_state = value["optimizer_state_dict"]
        if require_complete_adam:
            self._validate_required_adam_state(
                optimizer_state,
                prefix="R10 trainer.optimizer_schedule",
                expected_learning_rate=learning_rate,
            )
        elif not isinstance(optimizer_state, dict) or set(optimizer_state) != {
            "state",
            "param_groups",
        }:
            raise RuntimeError("R10 baseline optimizer state is malformed")
        return value

    def _action_ball_r10_install_optimizer_state(
        self,
        value: object,
        *,
        require_complete_adam: bool = False,
    ) -> None:
        state = self._action_ball_r10_validate_optimizer_state(
            value,
            require_complete_adam=require_complete_adam,
        )
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.alg.schedule = state["schedule"]
        self.alg.desired_kl = state["desired_kl"]
        self.alg.learning_rate = state["learning_rate"]
        for group, saved_lr in zip(
            self.alg.optimizer.param_groups,
            state["param_group_lrs"],
        ):
            group["lr"] = saved_lr

    def _action_ball_r10_normalizer_state(self, role: str) -> dict:
        attribute, normalizer, aliases = self._resolve_runtime_normalizer(role)
        if attribute is None or not is_empirical_normalizer(normalizer):
            raise RuntimeError(f"R10 requires an empirical {role} normalizer")
        if type(getattr(normalizer, "training", None)) is not bool:
            raise RuntimeError(f"R10 {role} normalizer training mode is invalid")
        state = normalizer.state_dict()
        self._validate_empirical_normalizer_state(
            role=role,
            attribute_name=attribute,
            normalizer=normalizer,
        )
        return {
            "schema_version": _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            "kind": f"action_ball_r10_trainer_normalizer_{role}_v1",
            "role": role,
            "attribute_name": attribute,
            "aliases": aliases,
            "normalizer_type": self._action_ball_r10_type_identity(normalizer),
            "normalizer_training": normalizer.training,
            "state_dict": state,
        }

    def _action_ball_r10_validate_normalizer_state(
        self,
        role: str,
        value: object,
    ) -> dict:
        required = {
            "schema_version",
            "kind",
            "role",
            "attribute_name",
            "aliases",
            "normalizer_type",
            "normalizer_training",
            "state_dict",
        }
        if type(value) is not dict or set(value) != required:
            raise RuntimeError(f"R10 trainer.normalizer.{role} payload keys differ")
        current = self._action_ball_r10_normalizer_state(role)
        for key in (
            "schema_version",
            "kind",
            "role",
            "attribute_name",
            "aliases",
            "normalizer_type",
        ):
            if value[key] != current[key] or type(value[key]) is not type(current[key]):
                raise RuntimeError(f"R10 trainer.normalizer.{role} {key} differs")
        if type(value["normalizer_training"]) is not bool:
            raise RuntimeError(f"R10 trainer.normalizer.{role} mode differs")
        self._action_ball_r10_validate_module_state_dict(
            value["state_dict"],
            current["state_dict"],
            label=f"trainer.normalizer.{role}",
        )
        return value

    def _action_ball_r10_install_normalizer_state(
        self,
        role: str,
        value: object,
    ) -> None:
        state = self._action_ball_r10_validate_normalizer_state(role, value)
        _attribute, normalizer, _aliases = self._resolve_runtime_normalizer(role)
        normalizer.load_state_dict(state["state_dict"], strict=True)
        normalizer.train(state["normalizer_training"])

    def _action_ball_r10_recurrent_state(self) -> Tuple[object, object, object]:
        r10 = _action_ball_r10_module()
        policy = self.alg.policy
        recurrent = getattr(policy, "is_recurrent", None)
        if type(recurrent) is not bool:
            raise RuntimeError("R10 policy recurrent ABI is invalid")
        if not recurrent:
            return r10.RecurrentFrontierStatus.NOT_APPLICABLE, None, None
        getter = getattr(policy, "get_hidden_states", None)
        if not callable(getter):
            raise RuntimeError("R10 recurrent policy lacks get_hidden_states()")
        hidden = getter()
        if type(hidden) is not tuple or len(hidden) != 2:
            raise RuntimeError("R10 recurrent policy returned an invalid hidden pair")
        actor_hidden, critic_hidden = hidden
        _action_ball_r10_validate_hidden_tree(actor_hidden, label="actor")
        _action_ball_r10_validate_hidden_tree(critic_hidden, label="critic")
        return (
            r10.RecurrentFrontierStatus.SEALED,
            _action_ball_r10_clone_tree(actor_hidden),
            _action_ball_r10_clone_tree(critic_hidden),
        )

    @staticmethod
    def _action_ball_r10_tree_to_device(value: object, device: object) -> object:
        if torch.is_tensor(value):
            return value.detach().to(device=device).clone()
        if type(value) is tuple:
            return tuple(
                MotionOnPolicyRunner._action_ball_r10_tree_to_device(child, device)
                for child in value
            )
        if value is None:
            return None
        raise RuntimeError("R10 hidden state contains an unsupported device tree")

    def _action_ball_r10_install_recurrent_state(
        self,
        status: str,
        actor_hidden: object,
        critic_hidden: object,
    ) -> None:
        r10 = _action_ball_r10_module()
        policy = self.alg.policy
        recurrent = getattr(policy, "is_recurrent", None)
        if status == r10.RecurrentFrontierStatus.NOT_APPLICABLE.value:
            if recurrent is not False or actor_hidden is not None or critic_hidden is not None:
                raise RuntimeError("R10 non-recurrent frontier differs from live policy")
            return
        if status != r10.RecurrentFrontierStatus.SEALED.value or recurrent is not True:
            raise RuntimeError("R10 recurrent frontier differs from live policy")
        _action_ball_r10_validate_hidden_tree(actor_hidden, label="actor")
        _action_ball_r10_validate_hidden_tree(critic_hidden, label="critic")
        memory_a = getattr(policy, "memory_a", None)
        memory_c = getattr(policy, "memory_c", None)
        if not callable(getattr(memory_a, "reset", None)) or not callable(
            getattr(memory_c, "reset", None)
        ):
            raise RuntimeError("R10 recurrent policy lacks typed memory owners")
        memory_a.reset(
            hidden_states=self._action_ball_r10_tree_to_device(
                actor_hidden, self.device
            )
        )
        memory_c.reset(
            hidden_states=self._action_ball_r10_tree_to_device(
                critic_hidden, self.device
            )
        )

    @staticmethod
    def _action_ball_r10_transition_empty(transition: object) -> bool:
        expected = {
            "observations",
            "privileged_observations",
            "actions",
            "privileged_actions",
            "rewards",
            "dones",
            "values",
            "actions_log_prob",
            "action_mean",
            "action_sigma",
            "hidden_states",
            "rnd_state",
        }
        namespace = getattr(transition, "__dict__", None)
        return (
            type(namespace) is dict
            and set(namespace) == expected
            and all(value is None for value in namespace.values())
        )

    def _action_ball_r10_frontier_from_handoff(
        self,
        handoff: ActionBallR10RunnerBoundaryHandoff,
    ) -> dict:
        value = {
            "schema_version": _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            "kind": "action_ball_r10_trainer_rollout_frontier_v1",
            "completed_update_index": handoff.completed_update_index,
            "next_learning_iteration": handoff.next_learning_iteration,
            "num_envs": int(self.env.num_envs),
            "num_steps_per_env": int(self.num_steps_per_env),
            "storage_step": handoff.storage_step,
            "transition_empty": handoff.transition_empty,
            "completed_environment_steps": handoff.completed_environment_steps,
            "actor_normalizer_calls": handoff.actor_normalizer_calls,
            "critic_normalizer_calls": handoff.critic_normalizer_calls,
            "privileged_obs_type": "critic",
            "final_normalized_actor_observations": (
                handoff.final_normalized_actor_observations.detach().clone()
            ),
            "final_normalized_critic_observations": (
                handoff.final_normalized_critic_observations.detach().clone()
            ),
            "recurrent_frontier": handoff.recurrent_frontier.value,
            "recurrent_actor_state": _action_ball_r10_clone_tree(
                handoff.recurrent_actor_state
            ),
            "recurrent_critic_state": _action_ball_r10_clone_tree(
                handoff.recurrent_critic_state
            ),
        }
        value["runner_frontier_sha256"] = self._exact_resume_tree_sha256(value)
        if value["runner_frontier_sha256"] != handoff.runner_frontier_sha256:
            raise RuntimeError("R10 handoff/frontier SHA-256 differs")
        return _action_ball_r10_validate_frontier_payload(
            value,
            expected_update_index=handoff.completed_update_index,
        )

    def _action_ball_r10_frontier_state(self) -> dict:
        value = self._action_ball_r10_frontier_payload
        if type(value) is not dict:
            raise RuntimeError("R10 trainer.rollout_frontier is not installed")
        return value

    def _action_ball_r10_validate_frontier_state(self, value: object) -> dict:
        state = _action_ball_r10_validate_frontier_payload(value)
        if (
            state["num_envs"] != int(self.env.num_envs)
            or state["num_steps_per_env"] != int(self.num_steps_per_env)
            or state["privileged_obs_type"] != getattr(
                self, "privileged_obs_type", None
            )
        ):
            raise RuntimeError("R10 trainer.rollout_frontier runner ABI differs")
        runtime_obs_mode = getattr(
            getattr(getattr(self.env, "unwrapped", self.env), "cfg", None),
            "obs_mode",
            None,
        )
        if runtime_obs_mode == "action_ball_full_mdp":
            expected_widths = tuple(
                int(
                    self._resolve_runtime_normalizer(role)[1]
                    .state_dict()["mean"]
                    .numel()
                )
                for role in ("actor", "critic")
            )
        else:
            expected_widths = (
                _A211_ACTOR_WIDTH,
                _A211_CRITIC_WIDTH,
            ) if runtime_obs_mode == "action_ball_a211" else (
                _C211_ACTOR_WIDTH,
                _C211_CRITIC_WIDTH,
            )
        if (
            state["final_normalized_actor_observations"].shape[1]
            != expected_widths[0]
            or state["final_normalized_critic_observations"].shape[1]
            != expected_widths[1]
        ):
            raise RuntimeError("R10 trainer.rollout_frontier widths differ")
        live_recurrent = getattr(self.alg.policy, "is_recurrent", None)
        r10 = _action_ball_r10_module()
        if (
            live_recurrent is True
            and state["recurrent_frontier"]
            != r10.RecurrentFrontierStatus.SEALED.value
        ) or (
            live_recurrent is False
            and state["recurrent_frontier"]
            != r10.RecurrentFrontierStatus.NOT_APPLICABLE.value
        ):
            raise RuntimeError("R10 trainer.rollout_frontier recurrence differs")
        return state

    def _action_ball_r10_install_frontier_state(self, value: object) -> None:
        if (
            type(value) is dict
            and value.get("kind")
            == "action_ball_r10_trainer_rollout_frontier_baseline_v1"
        ):
            required = {
                "schema_version",
                "kind",
                "current_learning_iteration",
                "recurrent_frontier",
                "recurrent_actor_state",
                "recurrent_critic_state",
            }
            if (
                set(value) != required
                or value.get("schema_version")
                != _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION
                or type(value.get("current_learning_iteration")) is not int
            ):
                raise RuntimeError("R10 rollback frontier baseline differs")
            self.current_learning_iteration = value["current_learning_iteration"]
            if value["recurrent_frontier"] == "UNINITIALIZED_INTERNAL":
                policy = self.alg.policy
                if getattr(policy, "is_recurrent", None) is not True:
                    raise RuntimeError("R10 rollback recurrence baseline differs")
                policy.memory_a.reset()
                policy.memory_c.reset()
            else:
                self._action_ball_r10_install_recurrent_state(
                    value["recurrent_frontier"],
                    value["recurrent_actor_state"],
                    value["recurrent_critic_state"],
                )
            self._action_ball_r10_frontier_payload = value
            self._action_ball_r10_restored_initial_frontier = None
            return
        state = self._action_ball_r10_validate_frontier_state(value)
        self._action_ball_r10_install_recurrent_state(
            state["recurrent_frontier"],
            state["recurrent_actor_state"],
            state["recurrent_critic_state"],
        )
        self.current_learning_iteration = state["next_learning_iteration"]
        self._action_ball_r10_frontier_payload = state
        self._action_ball_r10_restored_initial_frontier = {
            "actor": state["final_normalized_actor_observations"].detach().clone(),
            "critic": state["final_normalized_critic_observations"].detach().clone(),
            "consumed": False,
        }

    def _action_ball_r10_frontier_baseline(self) -> dict:
        r10 = _action_ball_r10_module()
        recurrent = getattr(self.alg.policy, "is_recurrent", None)
        if recurrent is False:
            status = r10.RecurrentFrontierStatus.NOT_APPLICABLE.value
            actor_hidden = None
            critic_hidden = None
        elif recurrent is True:
            hidden = self.alg.policy.get_hidden_states()
            if type(hidden) is not tuple or len(hidden) != 2:
                raise RuntimeError("R10 cold baseline recurrent ABI differs")
            if hidden[0] is None and hidden[1] is None:
                actor_hidden = None
                critic_hidden = None
                status = "UNINITIALIZED_INTERNAL"
            elif hidden[0] is None or hidden[1] is None:
                raise RuntimeError("R10 cold baseline hidden pair is partial")
            else:
                actor_hidden = _action_ball_r10_clone_tree(hidden[0])
                critic_hidden = _action_ball_r10_clone_tree(hidden[1])
                status = r10.RecurrentFrontierStatus.SEALED.value
        else:
            raise RuntimeError("R10 cold baseline policy recurrence differs")
        return {
            "schema_version": _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            "kind": "action_ball_r10_trainer_rollout_frontier_baseline_v1",
            "current_learning_iteration": int(self.current_learning_iteration),
            "recurrent_frontier": status,
            "recurrent_actor_state": actor_hidden,
            "recurrent_critic_state": critic_hidden,
        }

    def _action_ball_r10_validate_runner_join_claims(
        self,
        claims_by_owner: object,
        *,
        runner_frontier_sha256: str,
        ppo_drain_frontier_sha256: Optional[str] = None,
    ) -> Dict[str, Tuple[object, ...]]:
        r10 = _action_ball_r10_module()
        if type(claims_by_owner) is not dict or set(claims_by_owner) != set(
            _ACTION_BALL_R10_RUNNER_OWNER_IDS
        ):
            raise RuntimeError("R10 runner join-claim owner surface differs")
        expected_ids = {
            owner_id: tuple(
                spec.join_id
                for spec in r10.R10_GLOBAL_JOIN_SPECS
                if owner_id in spec.owner_ids
            )
            for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS
        }
        checked = {}
        for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS:
            claims = claims_by_owner[owner_id]
            if type(claims) is not tuple or tuple(
                getattr(claim, "join_id", None) for claim in claims
            ) != expected_ids[owner_id]:
                raise RuntimeError(f"R10 join-claim surface differs for {owner_id}")
            for claim in claims:
                if type(claim) is not r10.OwnerJoinClaim:
                    raise RuntimeError(f"R10 join claim type differs for {owner_id}")
                digest = claim.value_sha256
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise RuntimeError(f"R10 join claim digest differs for {owner_id}")
                if (
                    claim.join_id == "trainer_update_frontier"
                    and digest != runner_frontier_sha256
                ):
                    raise RuntimeError(
                        f"R10 trainer update frontier differs for {owner_id}"
                    )
            if (
                owner_id == "trainer.rollout_frontier"
                and ppo_drain_frontier_sha256 is not None
            ):
                if (
                    type(ppo_drain_frontier_sha256) is not str
                    or len(ppo_drain_frontier_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in ppo_drain_frontier_sha256
                    )
                ):
                    raise RuntimeError(
                        "R10 runner PPO-drain frontier digest is invalid"
                    )
                claims = tuple(
                    r10.OwnerJoinClaim(
                        claim.join_id,
                        (
                            ppo_drain_frontier_sha256
                            if claim.join_id == "ppo_drain_frontier"
                            else claim.value_sha256
                        ),
                    )
                    for claim in claims
                )
            checked[owner_id] = tuple(claims)
        return checked

    def _action_ball_r10_build_trainer_owners(
        self,
        *,
        mutation_versions: Mapping[str, int],
        join_claims: Mapping[str, Tuple[object, ...]],
    ) -> Dict[str, _ActionBallR10TrainerOwner]:
        if set(mutation_versions) != set(_ACTION_BALL_R10_RUNNER_OWNER_IDS):
            raise RuntimeError("R10 runner mutation-version surface differs")
        registry = self._action_ball_r10_registry
        getters = {
            "trainer.policy": self._action_ball_r10_policy_state,
            "trainer.optimizer_schedule": self._action_ball_r10_optimizer_state,
            "trainer.normalizer.actor": lambda: self._action_ball_r10_normalizer_state(
                "actor"
            ),
            "trainer.normalizer.critic": lambda: self._action_ball_r10_normalizer_state(
                "critic"
            ),
            "trainer.rollout_frontier": self._action_ball_r10_frontier_state,
        }
        validators = {
            "trainer.policy": self._action_ball_r10_validate_policy_state,
            "trainer.optimizer_schedule": self._action_ball_r10_validate_optimizer_state,
            "trainer.normalizer.actor": lambda value: self._action_ball_r10_validate_normalizer_state(
                "actor", value
            ),
            "trainer.normalizer.critic": lambda value: self._action_ball_r10_validate_normalizer_state(
                "critic", value
            ),
            "trainer.rollout_frontier": self._action_ball_r10_validate_frontier_state,
        }
        installers = {
            "trainer.policy": self._action_ball_r10_install_policy_state,
            "trainer.optimizer_schedule": self._action_ball_r10_install_optimizer_state,
            "trainer.normalizer.actor": lambda value: self._action_ball_r10_install_normalizer_state(
                "actor", value
            ),
            "trainer.normalizer.critic": lambda value: self._action_ball_r10_install_normalizer_state(
                "critic", value
            ),
            "trainer.rollout_frontier": self._action_ball_r10_install_frontier_state,
        }
        return {
            owner_id: _ActionBallR10TrainerOwner(
                descriptor=registry.descriptor(owner_id),
                mutation_version=mutation_versions[owner_id],
                state_getter=getters[owner_id],
                state_validator=validators[owner_id],
                state_installer=installers[owner_id],
                join_claims=join_claims[owner_id],
                poison=self._action_ball_r10_poison,
            )
            for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS
        }

    def _action_ball_r10_validate_restore_completion_receipt(
        self,
        receipt: object,
        *,
        verified_checkpoint: object,
    ) -> None:
        r10 = _action_ball_r10_module()
        registry = self._action_ball_r10_registry
        if (
            type(receipt) is not r10.RestoreReceipt
            or receipt.schema_version != r10.SCHEMA_VERSION
            or type(receipt.schema_version) is not int
            or receipt.kind
            != "action_ball_full_mdp_checkpoint_restore_receipt_v1"
            or receipt.integration_status != r10.INTEGRATION_STATUS
            or receipt.runtime_wiring is not False
            or receipt.continuation_authorized is not False
            or receipt.owner_root_sha256
            != verified_checkpoint.owner_root_sha256
            or receipt.prepared_owner_ids != registry.owner_ids
            or receipt.committed_owner_ids != registry.owner_ids
            or receipt.all_prepared_before_first_commit is not True
            or receipt.runtime_poisoned is not False
            or receipt.retry_permitted is not False
        ):
            raise RuntimeError(
                "R10 cold restore completion receipt differs from the exact "
                "21-owner transaction"
            )

    def _action_ball_r10_complete_cold_restore(
        self,
        *,
        receipt: object,
        owners: Tuple[object, ...],
        verified_checkpoint: object,
        cold_frontier: Mapping[str, object],
    ) -> None:
        """One-shot handoff from the pure coordinator to the env adapter."""

        if self._action_ball_r10_restore_completion_attempted:
            poison_owners = self._action_ball_r10_restore_owners
            if poison_owners is None:
                poison_owners = owners
            self._action_ball_r10_broadcast_restore_poison(
                poison_owners,
                reason="R10 cold restore completion was attempted more than once",
            )
            raise RuntimeError(
                "R10 cold restore completion retry is forbidden"
            )
        self._action_ball_r10_restore_completion_attempted = True
        self._action_ball_r10_restore_owners = owners
        try:
            self._action_ball_r10_validate_restore_completion_receipt(
                receipt,
                verified_checkpoint=verified_checkpoint,
            )
            if (
                self.current_learning_iteration
                != verified_checkpoint.boundary.update_index + 1
                or self._action_ball_r10_frontier_payload[
                    "runner_frontier_sha256"
                ]
                != cold_frontier["runner_frontier_sha256"]
            ):
                raise RuntimeError(
                    "R10 cold restore committed a mismatched trainer frontier"
                )
            callback = self._action_ball_r10_restore_complete_callback
            if not callable(callback):
                raise RuntimeError(
                    "R10 cold restore completion callback is unavailable"
                )
            callback_result = callback(receipt)
            if callback_result is not None:
                raise RuntimeError(
                    "R10 cold restore completion callback returned data"
                )
        except BaseException as exc:
            self._action_ball_r10_broadcast_restore_poison(
                owners,
                reason=(
                    "R10 cold restore completion failed; retry is forbidden: "
                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                ),
            )
            raise RuntimeError(
                "R10 cold restore completion failed; retry is forbidden"
            ) from exc
        self._action_ball_r10_restore_receipt = receipt

    def _action_ball_r10_restore_cold_checkpoint(
        self,
        *,
        verified_checkpoint: object,
        capsule: ActionBallR10ColdRestoreCapsule,
        cold_frontier: dict,
    ) -> None:
        """Commit all 21 owners without sampling or resetting the live env."""

        if self._action_ball_r10_restore_completion_attempted:
            owners = self._action_ball_r10_restore_owners
            self._action_ball_r10_broadcast_restore_poison(
                owners,
                reason=(
                    "R10 cold restore transaction was attempted more than "
                    "once"
                ),
            )
            raise RuntimeError("R10 cold restore retry is forbidden")
        self._action_ball_r10_require_healthy()
        if getattr(self, "_action_ball_resume_reset_pending", False):
            raise RuntimeError("R10 cold restore cannot inherit the legacy reset path")
        storage = getattr(self.alg, "storage", None)
        if type(getattr(storage, "step", None)) is not int or storage.step != 0:
            raise RuntimeError("R10 cold restore requires fresh empty rollout storage")
        if not self._action_ball_r10_transition_empty(self.alg.transition):
            raise RuntimeError("R10 cold restore requires a fresh empty transition")
        self._action_ball_r10_frontier_payload = (
            self._action_ball_r10_frontier_baseline()
        )
        states_by_owner = {
            state.owner_id: state for state in verified_checkpoint.owner_states
        }
        claims = self._action_ball_r10_validate_runner_join_claims(
            {
                owner_id: states_by_owner[owner_id].join_claims
                for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS
            },
            runner_frontier_sha256=cold_frontier["runner_frontier_sha256"],
        )
        runner_owners = self._action_ball_r10_build_trainer_owners(
            mutation_versions=self._action_ball_r10_owner_mutation_versions,
            join_claims=claims,
        )
        guard = _ActionBallR10EnvMethodGuard(self.env)
        guard.install_restore_guards()
        normalizer_guards = []
        owners = None

        def forbid_normalizer_forward(_module, _inputs):
            raise RuntimeError("R10 cold restore forbids normalizer.forward()")

        try:
            for role in ("actor", "critic"):
                _attribute, normalizer, _aliases = self._resolve_runtime_normalizer(
                    role
                )
                registrar = getattr(normalizer, "register_forward_pre_hook", None)
                if not callable(registrar):
                    raise RuntimeError(
                        f"R10 cold restore cannot guard {role} normalizer"
                    )
                normalizer_guards.append(registrar(forbid_normalizer_forward))
            owners = self._action_ball_r10_complete_owners(
                operation="cold_restore",
                runner_owners=runner_owners,
                verified_checkpoint=verified_checkpoint,
            )
            r10 = _action_ball_r10_module()
            coordinator = r10.CheckpointRestoreCoordinator(
                engine=r10.OwnerEngine.ISAAC,
                registry=self._action_ball_r10_registry,
                owners=owners,
            )
            receipt = coordinator.restore(
                capsule.checkpoint_bytes,
                expected_external_pins=capsule.expected_external_pins,
            )
            self._action_ball_r10_complete_cold_restore(
                receipt=receipt,
                owners=owners,
                verified_checkpoint=verified_checkpoint,
                cold_frontier=cold_frontier,
            )
        except BaseException:
            if not self._action_ball_r10_runtime_poisoned:
                if owners is None:
                    self._action_ball_r10_poison(
                        "R10 cold restore failed before owner completion; "
                        "retry is forbidden"
                    )
                else:
                    self._action_ball_r10_broadcast_restore_poison(
                        owners,
                        reason=(
                            "R10 cold restore failed; retry is forbidden"
                        ),
                    )
            raise
        finally:
            cleanup_failures = []
            for handle in reversed(normalizer_guards):
                try:
                    handle.remove()
                except BaseException as exc:
                    cleanup_failures.append(
                        "normalizer hook:"
                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                    )
            try:
                guard.close()
            except BaseException as exc:
                cleanup_failures.append(
                    "environment guard:"
                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                )
            if cleanup_failures:
                reason = (
                    "R10 cold restore guard cleanup failed; retry is "
                    "forbidden: "
                    + ",".join(cleanup_failures)
                )
                if owners is None:
                    self._action_ball_r10_poison(reason)
                else:
                    self._action_ball_r10_broadcast_restore_poison(
                        owners,
                        reason=reason,
                    )
                raise RuntimeError(reason)
        for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS:
            self._action_ball_r10_owner_mutation_versions[owner_id] = (
                states_by_owner[owner_id].mutation_version
            )

    def _action_ball_r10_validate_save_authority(
        self,
        value: object,
        *,
        handoff: ActionBallR10RunnerBoundaryHandoff,
    ) -> Tuple[object, Dict[str, Tuple[object, ...]]]:
        if type(value) is not ActionBallR10RunnerSaveAuthority:
            raise RuntimeError("R10 adapter returned an untyped save authority")
        if (
            type(value.schema_version) is not int
            or value.schema_version != _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION
            or value.kind != _ACTION_BALL_R10_RUNNER_SAVE_AUTHORITY_KIND
        ):
            raise RuntimeError("R10 save authority schema/kind differs")
        r10 = _action_ball_r10_module()
        runtime_obs_mode = getattr(
            getattr(getattr(self.env, "unwrapped", self.env), "cfg", None),
            "obs_mode",
            None,
        )
        if runtime_obs_mode == "action_ball_full_mdp":
            retained_family = getattr(
                self, "_action_ball_full_mdp_family", None
            )
            if type(value.family) is not r10.PolicyFamily or (
                retained_family is not None
                and value.family is not retained_family
            ):
                raise RuntimeError("R10 save authority family differs")
            if retained_family is None:
                self._action_ball_full_mdp_family = value.family
        else:
            expected_family = (
                r10.PolicyFamily.A
                if runtime_obs_mode == "action_ball_a211"
                else r10.PolicyFamily.C
            )
            if value.family is not expected_family:
                raise RuntimeError("R10 save authority family differs")
        pins = value.immutable_pins
        if type(pins) is not r10.ImmutableCheckpointPins:
            raise RuntimeError("R10 save authority immutable pins have the wrong type")
        for label, digest in (
            ("code", pins.code_sha256),
            ("config", pins.config_sha256),
            ("contract", pins.contract_sha256),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"R10 save authority {label} pin is invalid")
        if pins.contract_sha256 != self.training_contract_sha256:
            raise RuntimeError(
                "R10 save authority contract pin differs from the runner contract"
            )
        r10.validate_checkpoint_boundary(value.boundary)
        if tuple(world.world_id for world in value.boundary.worlds) != tuple(
            range(int(self.env.num_envs))
        ):
            raise RuntimeError("R10 save authority world identity differs from runner")
        if (
            value.boundary.update_index != handoff.completed_update_index
            or value.boundary.rollout_storage_empty is not True
            or value.boundary.actor_frontier_sealed is not True
            or value.boundary.critic_frontier_sealed is not True
            or value.boundary.recurrent_frontier is not handoff.recurrent_frontier
        ):
            raise RuntimeError("R10 save authority boundary differs from runner")
        claims = self._action_ball_r10_validate_runner_join_claims(
            value.runner_join_claims,
            runner_frontier_sha256=handoff.runner_frontier_sha256,
            ppo_drain_frontier_sha256=(
                self.action_ball_full_mdp_runner_ppo_drain_frontier(
                    value.boundary
                )
            ),
        )
        return value, claims

    def _action_ball_r10_seal_post_update(
        self,
        handoff: ActionBallR10RunnerBoundaryHandoff,
    ) -> object:
        """Ask the full adapter whether this exact boundary should be sealed."""

        self._action_ball_r10_require_healthy()
        self._action_ball_r10_last_boundary_handoff = handoff
        frontier = self._action_ball_r10_frontier_from_handoff(handoff)
        self._action_ball_r10_frontier_payload = frontier
        authority = (
            self._action_ball_r10_checkpoint_adapter.action_ball_r10_post_update_authority(
                handoff
            )
        )
        if authority is None:
            return None
        authority, claims = self._action_ball_r10_validate_save_authority(
            authority,
            handoff=handoff,
        )
        versions = {
            owner_id: handoff.next_learning_iteration
            for owner_id in _ACTION_BALL_R10_RUNNER_OWNER_IDS
        }
        runner_owners = self._action_ball_r10_build_trainer_owners(
            mutation_versions=versions,
            join_claims=claims,
        )
        owners = self._action_ball_r10_complete_owners(
            operation="seal",
            runner_owners=runner_owners,
        )
        r10 = _action_ball_r10_module()
        publication = r10.seal_checkpoint_candidate(
            engine=r10.OwnerEngine.ISAAC,
            family=authority.family,
            immutable_pins=authority.immutable_pins,
            registry=self._action_ball_r10_registry,
            boundary=authority.boundary,
            owners=owners,
        )
        if (
            publication.runtime_wiring is not False
            or publication.continuation_authorized is not False
        ):
            raise RuntimeError("R10 pure publication forged continuation authority")
        publish_result = (
            self._action_ball_r10_checkpoint_adapter.action_ball_r10_publish(
                publication=publication,
                handoff=handoff,
            )
        )
        self._action_ball_r10_last_publication = publication
        self._action_ball_r10_last_publish_result = publish_result
        self._action_ball_r10_owner_mutation_versions.update(versions)
        return publication

    def _action_ball_r10_capture_post_update_handoff(
        self,
        *,
        completed_update_index: int,
        tracker: dict,
    ) -> ActionBallR10RunnerBoundaryHandoff:
        """Validate the real RSL boundary and detach its next-rollout frontier."""

        storage = getattr(self.alg, "storage", None)
        storage_step = getattr(storage, "step", None)
        if type(storage_step) is not int or storage_step != 0:
            raise RuntimeError("R10 post-update rollout storage is not empty")
        if storage is not tracker.get("storage_object"):
            raise RuntimeError("R10 PPO replaced rollout storage during update")
        transition_empty = self._action_ball_r10_transition_empty(
            getattr(self.alg, "transition", None)
        )
        if not transition_empty:
            raise RuntimeError("R10 post-update transition is not empty")
        if (
            tracker.get("environment_step_in_flight") is not False
            or tracker.get("gae_in_flight") is not False
            or tracker.get("optimizer_in_flight") is not False
            or tracker.get("gae_complete") is not True
        ):
            raise RuntimeError("R10 post-update phase still has in-flight state")
        environment_steps = (
            tracker["completed_environment_steps"]
            - tracker["environment_steps_at_boundary"]
        )
        actor_calls = (
            tracker["actor_normalizer_calls"]
            - tracker["actor_calls_at_boundary"]
        )
        critic_calls = (
            tracker["critic_normalizer_calls"]
            - tracker["critic_calls_at_boundary"]
        )
        if (
            environment_steps != int(self.num_steps_per_env)
            or actor_calls != int(self.num_steps_per_env)
            or critic_calls != int(self.num_steps_per_env)
        ):
            raise RuntimeError(
                "R10 rollout step/normalizer counts do not match num_steps_per_env"
            )
        actor = tracker.get("last_normalized_actor")
        critic = tracker.get("last_compute_returns_critic")
        hooked_critic = tracker.get("last_normalized_critic")
        if (
            not torch.is_tensor(actor)
            or not torch.is_tensor(critic)
            or not torch.is_tensor(hooked_critic)
            or not torch.equal(critic, hooked_critic)
        ):
            raise RuntimeError("R10 final normalized actor/critic frontier differs")
        recurrent_status, actor_hidden, critic_hidden = (
            self._action_ball_r10_recurrent_state()
        )
        completed = int(completed_update_index)
        next_iteration = completed + 1
        digest_input = {
            "schema_version": _ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            "kind": "action_ball_r10_trainer_rollout_frontier_v1",
            "completed_update_index": completed,
            "next_learning_iteration": next_iteration,
            "num_envs": int(self.env.num_envs),
            "num_steps_per_env": int(self.num_steps_per_env),
            "storage_step": storage_step,
            "transition_empty": transition_empty,
            "completed_environment_steps": environment_steps,
            "actor_normalizer_calls": actor_calls,
            "critic_normalizer_calls": critic_calls,
            "privileged_obs_type": "critic",
            "final_normalized_actor_observations": actor.detach().clone(),
            "final_normalized_critic_observations": critic.detach().clone(),
            "recurrent_frontier": recurrent_status.value,
            "recurrent_actor_state": _action_ball_r10_clone_tree(actor_hidden),
            "recurrent_critic_state": _action_ball_r10_clone_tree(critic_hidden),
        }
        digest = self._exact_resume_tree_sha256(digest_input)
        handoff = ActionBallR10RunnerBoundaryHandoff(
            schema_version=_ACTION_BALL_R10_RUNNER_SCHEMA_VERSION,
            kind=_ACTION_BALL_R10_RUNNER_HANDOFF_KIND,
            integration_status=_ACTION_BALL_R10_RUNNER_INTEGRATION_STATUS,
            runtime_wiring=False,
            continuation_authorized=False,
            completed_update_index=completed,
            next_learning_iteration=next_iteration,
            storage_step=storage_step,
            transition_empty=transition_empty,
            environment_step_in_flight=False,
            completed_environment_steps=environment_steps,
            actor_normalizer_calls=actor_calls,
            critic_normalizer_calls=critic_calls,
            final_normalized_actor_observations=digest_input[
                "final_normalized_actor_observations"
            ],
            final_normalized_critic_observations=digest_input[
                "final_normalized_critic_observations"
            ],
            recurrent_frontier=recurrent_status,
            recurrent_actor_state=digest_input["recurrent_actor_state"],
            recurrent_critic_state=digest_input["recurrent_critic_state"],
            runner_frontier_sha256=digest,
        )
        self._action_ball_r10_seal_post_update(handoff)
        tracker["environment_steps_at_boundary"] = tracker[
            "completed_environment_steps"
        ]
        tracker["actor_calls_at_boundary"] = tracker[
            "actor_normalizer_calls"
        ]
        tracker["critic_calls_at_boundary"] = tracker[
            "critic_normalizer_calls"
        ]
        tracker["gae_complete"] = False
        tracker["last_compute_returns_critic"] = None
        restored = self._action_ball_r10_restored_initial_frontier
        if type(restored) is dict and restored.get("consumed") is True:
            self._action_ball_r10_restored_initial_frontier = None
        return handoff

    @staticmethod
    def _normalizer_aliases(role: str) -> Tuple[str, ...]:
        """Return the supported RSL-RL attribute names for one input role."""

        if role == "actor":
            return ("obs_normalizer", "actor_obs_normalizer")
        if role == "critic":
            return (
                "privileged_obs_normalizer",
                "critic_obs_normalizer",
            )
        raise ValueError("normalizer role must be actor or critic")

    def _action_ball_211_wait_mask_required(self) -> bool:
        """Return whether this is exactly one fresh A211/C211 runner."""

        enabled = tuple(
            name
            for name in (
                "action_ball_a211_trainability_preflight",
                "action_ball_c211_trainability_preflight",
            )
            if getattr(self, name, None) is not None
        )
        if len(enabled) > 1:
            raise RuntimeError("runner cannot be both fresh A211 and fresh C211")
        return bool(enabled)

    def _install_action_ball_211_wait_normalizer_masks(self) -> None:
        """Attach one post-normalization mask while preserving module identity.

        RSL-RL saves and restores the normalizers' own state dictionaries.
        Forward hooks are therefore intentional here: unlike a wrapper module,
        they do not prefix keys, replace the live object, or change the frozen
        evaluation/checkpoint hashes.  Re-entry is idempotent and also handles
        a test/runtime that deliberately replaces one normalizer object.
        """

        if not self._action_ball_211_wait_mask_required():
            return
        installed = dict(
            getattr(self, "_action_ball_211_wait_normalizer_hooks", {})
        )
        for role in ("actor", "critic"):
            attribute, normalizer, _aliases = self._resolve_runtime_normalizer(role)
            if attribute is None or not is_empirical_normalizer(normalizer):
                raise RuntimeError(
                    f"fresh ActionBall211 requires a live empirical {role} normalizer"
                )
            prior = installed.get(role)
            if prior is not None and prior[0] is normalizer:
                continue
            if prior is not None:
                prior[1].remove()
            registrar = getattr(normalizer, "register_forward_hook", None)
            if not callable(registrar):
                raise RuntimeError(
                    f"fresh ActionBall211 {role} normalizer cannot install WAIT masking"
                )
            installed[role] = (
                normalizer,
                registrar(_action_ball_211_wait_forward_hook(role)),
            )
        self._action_ball_211_wait_normalizer_hooks = installed

    def _normalize_action_ball_211_initial_observations(self, result):
        """Normalize the initial actor/critic pair before its first storage insert.

        RSL-RL 2.3.1 normalizes observations returned by ``env.step`` and reuses
        the final normalized critic tensor for bootstrap, but its initial
        ``get_observations`` result goes directly to ``alg.act``.  A211/C211
        cannot leave that first storage row on a different scale.  Both calls
        below use the same live modules (and therefore the same moment-update
        semantics) as every subsequent rollout row.
        """

        if not self._action_ball_211_wait_mask_required():
            raise RuntimeError(
                "initial ActionBall211 normalization is restricted to fresh A211/C211"
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "ActionBall211 env.get_observations() must return (actor, extras)"
            )
        raw_actor, raw_extras = result
        if not isinstance(raw_actor, torch.Tensor) or not isinstance(
            raw_extras, Mapping
        ):
            raise RuntimeError(
                "ActionBall211 initial actor/extras observation payload is invalid"
            )
        observation_groups = raw_extras.get("observations")
        if not isinstance(observation_groups, Mapping):
            raise RuntimeError(
                "ActionBall211 initial extras lacks observation groups"
            )
        privileged_type = getattr(self, "privileged_obs_type", None)
        if privileged_type != "critic":
            raise RuntimeError(
                "ActionBall211 initial rollout requires an explicit critic group"
            )
        raw_critic = observation_groups.get(privileged_type)
        if not isinstance(raw_critic, torch.Tensor):
            raise RuntimeError(
                "ActionBall211 initial rollout lacks the critic tensor"
            )
        self._install_action_ball_211_wait_normalizer_masks()
        _actor_attribute, actor_normalizer, _actor_aliases = (
            self._resolve_runtime_normalizer("actor")
        )
        _critic_attribute, critic_normalizer, _critic_aliases = (
            self._resolve_runtime_normalizer("critic")
        )
        normalized_actor = actor_normalizer(raw_actor.to(self.device))
        normalized_critic = critic_normalizer(raw_critic.to(self.device))
        normalized_groups = dict(observation_groups)
        normalized_groups[privileged_type] = normalized_critic
        normalized_extras = dict(raw_extras)
        normalized_extras["observations"] = normalized_groups
        return normalized_actor, normalized_extras

    def _resolve_runtime_normalizer(
        self, role: str
    ) -> Tuple[Optional[str], object, Tuple[str, ...]]:
        """Resolve one effective normalizer without guessing across ABI aliases."""

        candidates = self._normalizer_aliases(role)
        present = tuple(name for name in candidates if hasattr(self, name))
        if not present:
            return None, None, ()
        values = tuple(getattr(self, name) for name in present)
        active = tuple(
            (name, value)
            for name, value in zip(present, values)
            if is_empirical_normalizer(value)
        )
        if len(active) > 1 and any(
            value is not active[0][1] for _name, value in active[1:]
        ):
            raise RuntimeError(
                f"{role} observation normalizer aliases disagree"
            )
        if active:
            return active[0][0], active[0][1], present
        # None and Identity are both valid disabled representations.  Prefer
        # the first present name only for the ABI receipt; neither is an
        # effective transform.
        return present[0], values[0], present

    @staticmethod
    def _validate_empirical_normalizer_state(
        *, role: str, attribute_name: str, normalizer: object
    ) -> dict:
        """Validate the live empirical state before the first rollout.

        RSL-RL's empirical normalizer owns mean/std/count buffers.  Merely
        finding a non-Identity module is insufficient: a version mismatch or
        damaged restore can leave empty, shape-inconsistent, or non-finite
        buffers while the runner still advertises normalization as enabled.
        """

        state_dict = getattr(normalizer, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError(
                f"{attribute_name} lacks a deterministic state_dict()"
            )
        state = state_dict()
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(
                f"empirical {role} observation normalizer has no state"
            )

        semantic = {"mean": [], "var": [], "std": [], "count": []}
        semantic_aliases = {
            "mean": "mean",
            "running_mean": "mean",
            "var": "var",
            "variance": "var",
            "running_var": "var",
            "std": "std",
            "running_std": "std",
            "count": "count",
            "running_count": "count",
            "num_batches_tracked": "count",
        }
        shapes = {}
        for key, value in state.items():
            if type(key) is not str or not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"empirical {role} observation normalizer state must "
                    "contain only string-keyed tensors"
                )
            if value.numel() <= 0:
                raise RuntimeError(
                    f"empirical {role} observation normalizer state "
                    f"{key!r} is empty"
                )
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise RuntimeError(
                    f"empirical {role} observation normalizer state "
                    f"{key!r} is non-finite"
                )
            shapes[key] = list(value.shape)
            leaf = key.rsplit(".", 1)[-1].lstrip("_")
            semantic_name = semantic_aliases.get(leaf)
            if semantic_name is not None:
                semantic[semantic_name].append((key, value))

        for required in ("mean", "count"):
            if len(semantic[required]) != 1:
                raise RuntimeError(
                    f"empirical {role} observation normalizer must expose "
                    f"exactly one {required} buffer"
                )
        for optional in ("var", "std"):
            if len(semantic[optional]) > 1:
                raise RuntimeError(
                    f"empirical {role} observation normalizer exposes "
                    f"ambiguous {optional} buffers"
                )
        if not semantic["var"] and not semantic["std"]:
            raise RuntimeError(
                f"empirical {role} observation normalizer lacks a "
                "variance/std buffer"
            )
        mean_key, mean = semantic["mean"][0]
        count_key, count = semantic["count"][0]
        var_entry = semantic["var"][0] if semantic["var"] else None
        std_entry = semantic["std"][0] if semantic["std"] else None
        variance = None if var_entry is None else var_entry[1]
        std = None if std_entry is None else std_entry[1]
        moments = tuple(
            value for value in (variance, std) if value is not None
        )
        if mean.ndim < 1 or any(
            tuple(value.shape) != tuple(mean.shape)
            for value in moments
        ):
            raise RuntimeError(
                f"empirical {role} observation normalizer moment shapes "
                "are invalid"
            )
        if count.numel() != 1:
            raise RuntimeError(
                f"empirical {role} observation normalizer count must be scalar"
            )
        if any(
            not value.is_floating_point()
            or bool((value < 0).any().item())
            for value in moments
        ):
            raise RuntimeError(
                f"empirical {role} observation normalizer variance/std is invalid"
            )
        if count.is_complex() or float(count.item()) < 0.0:
            raise RuntimeError(
                f"empirical {role} observation normalizer count is invalid"
            )
        state_binding = MotionOnPolicyRunner._frozen_eval_state_binding(state)
        return {
            "attribute": attribute_name,
            "aliases_present": list(
                MotionOnPolicyRunner._normalizer_aliases(role)
            ),
            "module_type": (
                f"{type(normalizer).__module__}."
                f"{type(normalizer).__qualname__}"
            ),
            "state_shapes": shapes,
            "state_sha256": state_binding["sha256"],
            "state_byte_count": state_binding["size_bytes"],
            "semantic_width": int(mean.numel()),
            "semantic_buffers": {
                "mean": mean_key,
                "var": None if var_entry is None else var_entry[0],
                "std": None if std_entry is None else std_entry[0],
                "count": count_key,
            },
        }

    def _validate_training_normalizers(self) -> dict:
        """Fail before the first rollout when the configured ABI is not live."""

        empirical = getattr(self, "empirical_normalization", None)
        if empirical is None and not self._uses_real_rsl_rl_runner():
            # A few host-only safety/interrupt tests allocate the subclass via
            # __new__ and intentionally omit all base-runner fields.  The
            # installed RSL-RL class is still fail-closed below.
            empirical = False
        if type(empirical) is not bool:
            raise RuntimeError(
                "runner empirical_normalization must be an exact bool"
            )
        bindings = {}
        for role in ("actor", "critic"):
            attribute, normalizer, aliases_present = (
                self._resolve_runtime_normalizer(role)
            )
            active = is_empirical_normalizer(normalizer)
            if empirical and not active:
                reason = "absent" if attribute is None else "a no-op"
                raise RuntimeError(
                    f"empirical {role} observation normalizer is {reason}"
                )
            if not empirical and active:
                raise RuntimeError(
                    f"disabled {role} observation normalization has a live "
                    "transform"
                )
            if not empirical:
                bindings[role] = {
                    "enabled": False,
                    "attribute": attribute,
                    "aliases_present": list(aliases_present),
                }
                continue
            binding = self._validate_empirical_normalizer_state(
                role=role,
                attribute_name=str(attribute),
                normalizer=normalizer,
            )
            binding["aliases_present"] = list(aliases_present)
            binding["enabled"] = True
            bindings[role] = binding
        a211 = getattr(self, "action_ball_a211_trainability_preflight", None)
        if a211 is not None:
            expected = {
                "actor": (
                    _A211_ACTOR_WIDTH,
                    _A211_ACTOR_NORMALIZER_IDENTITY,
                ),
                "critic": (
                    _A211_CRITIC_WIDTH,
                    _A211_CRITIC_NORMALIZER_IDENTITY,
                ),
            }
            for role, (width, identity) in expected.items():
                binding = bindings.get(role)
                if (
                    not isinstance(binding, dict)
                    or binding.get("enabled") is not True
                    or binding.get("semantic_width") != width
                ):
                    raise RuntimeError(
                        f"A211 {role} normalizer must be a fresh enabled {width}-D transform"
                    )
                binding["contract_identity"] = identity
        c211 = getattr(self, "action_ball_c211_trainability_preflight", None)
        if c211 is not None:
            expected = {
                "actor": (
                    _C211_ACTOR_WIDTH,
                    _C211_ACTOR_NORMALIZER_IDENTITY,
                ),
                "critic": (
                    _C211_CRITIC_WIDTH,
                    _C211_CRITIC_NORMALIZER_IDENTITY,
                ),
            }
            for role, (width, identity) in expected.items():
                binding = bindings.get(role)
                if (
                    not isinstance(binding, dict)
                    or binding.get("enabled") is not True
                    or binding.get("semantic_width") != width
                ):
                    raise RuntimeError(
                        f"C211 {role} normalizer must be a fresh enabled {width}-D transform"
                    )
                binding["contract_identity"] = identity
        result = {
            "empirical_normalization": empirical,
            "normalizers": bindings,
            "a211_trainability": a211,
        }
        if c211 is not None:
            result["c211_trainability"] = c211
        return result

    @staticmethod
    def _uses_real_rsl_rl_runner() -> bool:
        """Distinguish the installed runner from host-only test stand-ins."""

        return OnPolicyRunner.__module__.startswith("rsl_rl.")

    def _policy_std_abi(self) -> Optional[dict]:
        """Resolve the trainable std parameter without reading device values."""

        algorithm = getattr(self, "alg", None)
        policy = None if algorithm is None else getattr(algorithm, "policy", None)
        if policy is None:
            if self._uses_real_rsl_rl_runner():
                raise RuntimeError("RSL-RL algorithm lacks policy for std guard")
            # Host-only runner stand-ins used by the exact-resume tests do not
            # implement an actor.  Production RSL-RL is fail-closed above.
            return None

        raw_std = getattr(policy, "std", None)
        raw_log_std = getattr(policy, "log_std", None)
        has_std = isinstance(raw_std, torch.Tensor)
        has_log_std = isinstance(raw_log_std, torch.Tensor)
        if has_std == has_log_std:
            raise RuntimeError(
                "policy std ABI must expose exactly one of std or log_std"
            )
        claimed_type = getattr(policy, "noise_std_type", None)
        noise_std_type = "log" if has_log_std else "scalar"
        if claimed_type is not None and claimed_type != noise_std_type:
            raise RuntimeError(
                "policy noise_std_type disagrees with its trainable parameter"
            )
        source = raw_log_std if has_log_std else raw_std
        if source.numel() <= 0:
            raise RuntimeError("policy std parameter is empty")
        if not source.is_floating_point():
            raise RuntimeError("policy std parameter must be floating point")
        return {
            "noise_std_type": noise_std_type,
            "parameter_name": "log_std" if has_log_std else "std",
            "parameter_shape": list(source.shape),
            "parameter_count": int(source.numel()),
        }

    def _policy_std_snapshot(self, *, ppo_update: Optional[int]) -> Optional[dict]:
        """Return and validate the optimizer-visible policy std and LR."""

        abi = self._policy_std_abi()
        if abi is None:
            return None
        algorithm = self.alg
        source = getattr(algorithm.policy, abi["parameter_name"])
        has_log_std = abi["noise_std_type"] == "log"
        realized = torch.exp(source.detach()) if has_log_std else source.detach()
        # Exactly one device-to-host synchronization at the PPO update
        # boundary supplies both validation and telemetry.  This avoids
        # reintroducing per-step .item() calls into the rollout hot path.
        summary = torch.stack(
            (realized.min(), realized.mean(), realized.max())
        ).tolist()
        std_min, std_mean, std_max = (float(value) for value in summary)
        if not all(math.isfinite(value) for value in summary):
            raise RuntimeError("realized policy std is non-finite")
        if std_min <= 0.0:
            raise RuntimeError("realized policy std must be strictly positive")

        learning_rate = getattr(algorithm, "learning_rate", None)
        if type(learning_rate) not in (int, float) or not math.isfinite(
            float(learning_rate)
        ) or float(learning_rate) <= 0.0:
            raise RuntimeError("algorithm learning_rate is invalid")
        optimizer = getattr(algorithm, "optimizer", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(param_groups, list) or not param_groups:
            raise RuntimeError("algorithm optimizer has no parameter groups")
        optimizer_lrs = []
        for index, group in enumerate(param_groups):
            value = group.get("lr") if isinstance(group, Mapping) else None
            if type(value) not in (int, float) or not math.isfinite(
                float(value)
            ) or float(value) <= 0.0:
                raise RuntimeError(
                    f"optimizer parameter group {index} learning rate is invalid"
                )
            optimizer_lrs.append(float(value))
        if any(value != float(learning_rate) for value in optimizer_lrs):
            raise RuntimeError(
                "algorithm learning_rate disagrees with optimizer parameter groups"
            )

        return {
            "event": _POLICY_STD_UPDATE_EVENT,
            "schema_version": _POLICY_STD_TELEMETRY_SCHEMA_VERSION,
            "ppo_update": ppo_update,
            "rank": int(self._joint_safety_rank()),
            **abi,
            "policy_std_min": std_min,
            "policy_std_mean": std_mean,
            "policy_std_max": std_max,
            "learning_rate": float(learning_rate),
            "learning_rate_at_floor": bool(
                float(learning_rate) <= _ADAPTIVE_KL_LEARNING_RATE_FLOOR
            ),
        }

    def _emit_policy_std_update(self, *, ppo_update: int) -> Optional[dict]:
        record = self._policy_std_snapshot(ppo_update=int(ppo_update))
        if record is not None:
            print(
                "HOPE_POLICY_STD_UPDATE_JSON="
                + json.dumps(
                    record,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return record

    def _reward_ppo_economy_gate_requested(self) -> bool:
        """Parse the explicit, probe-only reward/PPO economy evidence switch."""

        raw = os.environ.get(_REWARD_PPO_ECONOMY_GATE_ENV)
        if raw is None or raw == "0":
            return False
        if raw != "1":
            raise RuntimeError(
                f"{_REWARD_PPO_ECONOMY_GATE_ENV} must be exactly 0, 1, or absent"
            )
        if not self._action_ball_diagnostic_unauthorized():
            raise RuntimeError(
                f"{_REWARD_PPO_ECONOMY_GATE_ENV}=1 is allowed only for "
                "diagnostic ActionBall"
            )
        if self._joint_safety_rank() != 0:
            raise RuntimeError(
                "reward/PPO economy evidence requires the primary runner rank 0"
            )
        if int(self.num_steps_per_env) != _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE:
            raise RuntimeError(
                "reward/PPO economy evidence requires exactly 24 rollout steps"
            )
        storage = getattr(getattr(self, "alg", None), "storage", None)
        if storage is None:
            raise RuntimeError("reward/PPO economy evidence requires PPO storage")
        if (
            getattr(storage, "training_type", None) != "rl"
            or int(getattr(storage, "num_envs", -1))
            != _REWARD_PPO_ECONOMY_NUM_ENVS
            or int(getattr(storage, "num_transitions_per_env", -1))
            != _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE
        ):
            raise RuntimeError(
                "reward/PPO economy evidence requires exact 4096x24 RL storage"
            )
        return True

    def _prelong_preregistered_reward_recipe(self, expected_sha256: str) -> dict:
        """Read the hash-bound pre-scene recipe used to construct this runner."""

        if self.log_dir is None or self.training_contract_sha256 is None:
            raise RuntimeError(
                "pre-long semantics require a hash-bound training_contract.json"
            )
        contract_path = pathlib.Path(self.log_dir) / "params" / "training_contract.json"
        raw = contract_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.training_contract_sha256:
            raise RuntimeError(
                "training_contract.json changed before pre-long reward binding"
            )

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeError(
                        f"training contract repeats JSON key {key!r}"
                    )
                result[key] = value
            return result

        try:
            contract = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    RuntimeError(
                        "training contract contains non-finite "
                        f"number {token}"
                    )
                ),
            )
            receipt = contract["effective_reward_recipe"]
            action_ball_recipe_sha256 = contract["action_ball_training"][
                "effective_reward_recipe_sha256"
            ]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "training contract lacks the pre-long effective reward recipe"
            ) from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("sha256") != expected_sha256
            or action_ball_recipe_sha256 != expected_sha256
        ):
            raise RuntimeError(
                "pre-long launcher recipe SHA differs from the bound training contract"
            )
        return receipt

    @staticmethod
    def _economy_finite_tensor(value, *, name: str):
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            raise RuntimeError(f"reward/PPO economy {name} tensor is absent or empty")
        if not value.is_floating_point():
            raise RuntimeError(f"reward/PPO economy {name} tensor is not floating point")
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"reward/PPO economy {name} tensor is non-finite")
        return value.detach()

    @classmethod
    def _economy_distribution_stats(cls, value, *, name: str) -> dict:
        """Return one exact finite whole-rollout distribution at the update boundary."""

        flat = cls._economy_finite_tensor(value, name=name).reshape(-1)
        quantiles = torch.quantile(
            flat,
            torch.tensor(
                (0.50, 0.95, 0.99), dtype=flat.dtype, device=flat.device
            ),
        )
        summary = torch.cat(
            (
                torch.stack((flat.min(), flat.mean())),
                quantiles,
                flat.max().reshape(1),
            )
        ).tolist()
        result = {
            key: float(item)
            for key, item in zip(
                ("min", "mean", "p50", "p95", "p99", "max"), summary
            )
        }
        if not all(math.isfinite(item) for item in result.values()):
            raise RuntimeError(
                f"reward/PPO economy {name} distribution summary is non-finite"
            )
        return result

    @classmethod
    def _economy_advantage_stats(cls, value, *, name: str) -> dict:
        flat = cls._economy_finite_tensor(value, name=name).reshape(-1)
        summary = torch.stack(
            (flat.mean(), flat.std(), flat.min(), flat.max())
        ).tolist()
        result = {
            key: float(item)
            for key, item in zip(("mean", "std", "min", "max"), summary)
        }
        if not all(math.isfinite(item) for item in result.values()):
            raise RuntimeError(
                f"reward/PPO economy {name} advantage summary is non-finite"
            )
        return result

    def _prepare_reward_ppo_economy_rollout(
        self, *, activation: Mapping[str, object], ppo_update: int
    ) -> dict:
        """Freeze reward closure and pre-optimizer storage statistics."""

        storage = self.alg.storage
        expected_samples = (
            _REWARD_PPO_ECONOMY_NUM_ENVS
            * _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE
        )
        tensors = {
            "reward": getattr(storage, "rewards", None),
            "returns": getattr(storage, "returns", None),
            "values": getattr(storage, "values", None),
            "post_advantage": getattr(storage, "advantages", None),
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or int(value.numel()) != expected_samples:
                raise RuntimeError(
                    f"reward/PPO economy {name} storage must contain exactly "
                    f"{expected_samples} samples"
                )
        raw_advantage = tensors["returns"] - tensors["values"]
        returns_flat = tensors["returns"].reshape(-1)
        residual_flat = raw_advantage.reshape(-1)
        return_std, return_variance, residual_variance = torch.stack(
            (
                returns_flat.std(),
                returns_flat.var(correction=0),
                residual_flat.var(correction=0),
            )
        ).tolist()
        if (
            not all(
                math.isfinite(float(value))
                for value in (return_std, return_variance, residual_variance)
            )
            or float(return_variance) <= 0.0
        ):
            raise RuntimeError(
                "reward/PPO economy cannot compute finite explained variance"
            )
        explained_variance = 1.0 - (
            float(residual_variance) / float(return_variance)
        )
        if not math.isfinite(explained_variance):
            raise RuntimeError(
                "reward/PPO economy explained variance is non-finite"
            )
        pre_advantage = self._economy_advantage_stats(
            raw_advantage, name="pre-normalization advantage"
        )
        post_advantage = self._economy_advantage_stats(
            tensors["post_advantage"], name="post-normalization advantage"
        )
        tolerance = _REWARD_PPO_ECONOMY_ADVANTAGE_TOLERANCE
        if (
            abs(post_advantage["mean"]) > tolerance
            or abs(post_advantage["std"] - 1.0) > tolerance
        ):
            raise RuntimeError(
                "whole-rollout advantage normalization is not zero-mean/unit-std"
            )

        terms = activation.get("terms")
        cache = activation.get("reward_cache_contract")
        if (
            activation.get("event") != "hope_effective_reward_activation_update"
            or activation.get("task_kind") != "action_ball"
            or activation.get("ppo_update") != int(ppo_update)
            or activation.get("environment_step_count")
            != _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE
            or activation.get("num_envs") != _REWARD_PPO_ECONOMY_NUM_ENVS
            or activation.get("observed_sample_count") != expected_samples
            or not isinstance(terms, list)
            or not terms
            or not isinstance(cache, Mapping)
            or cache.get("total_reward_closure") != "validated"
        ):
            raise RuntimeError("reward/PPO economy activation closure is incomplete")
        term_names = [row.get("name") for row in terms if isinstance(row, Mapping)]
        if (
            len(term_names) != len(terms)
            or any(type(name) is not str or not name for name in term_names)
            or term_names != sorted(term_names)
            or len(term_names) != len(set(term_names))
        ):
            raise RuntimeError("reward/PPO economy reward terms are not exact ordered names")
        per_term_raw = {}
        per_term_weighted = {}
        per_term_denominator = {}
        per_term_error = {}
        for row in terms:
            name = row["name"]
            denominator = row.get("observed_sample_count")
            if denominator != expected_samples:
                raise RuntimeError(
                    f"reward/PPO economy term {name!r} denominator is incomplete"
                )
            per_term_raw[name] = row.get("raw_sum")
            per_term_weighted[name] = row.get("weighted_sum")
            per_term_denominator[name] = denominator
            per_term_error[name] = row.get("raw_recomposition_max_abs_error")

        return {
            "reward": {
                "pre_advantage_reward_min_mean_p50_p95_p99_max": (
                    self._economy_distribution_stats(
                        tensors["reward"], name="pre-advantage reward"
                    )
                ),
                "return_min_mean_p50_p95_p99_max": (
                    self._economy_distribution_stats(
                        tensors["returns"], name="return"
                    )
                ),
                "return_std": float(return_std),
                "explained_variance": float(explained_variance),
                "value_prediction_min_mean_p50_p95_p99_max": (
                    self._economy_distribution_stats(
                        tensors["values"], name="value prediction"
                    )
                ),
                "value_residual_min_mean_p50_p95_p99_max": (
                    self._economy_distribution_stats(
                        raw_advantage, name="value residual"
                    )
                ),
                "per_term_raw_sum": per_term_raw,
                "per_term_weighted_dt_sum": per_term_weighted,
                # This is an exact denominator for each term's contribution
                # distribution, including internally gated zero samples.  It
                # does not pretend that zero contribution means ineligible.
                "per_term_eligible_denominator": per_term_denominator,
                "per_term_denominator_semantics": (
                    "all_rollout_environment_samples_including_gated_zero"
                ),
                "reward_manager_total_sum": activation.get(
                    "total_weighted_reward_sum"
                ),
                "per_term_closure_error": per_term_error,
                "reward_manager_closure_max_abs_error": cache.get(
                    "max_abs_error"
                ),
                "recipe_sha256": activation.get("recipe_sha256"),
                "pre_advantage_reward_semantics": (
                    "ppo_storage_reward_after_timeout_bootstrap"
                ),
                # 2026-08-07 裁定一的前置:深度遥测必须落到 run.log 的 economy JSON 里。
                # 机制层每步都在算这些数,但在此之前从来没有一个消费方把它们写进收据,
                # 于是"38% 的样本各罚 1 个关节"和"1.2% 的样本各罚 31 个"根本分不开 ——
                # 而按深度验收的前提正是先能读到深度。只读快照,不清零,与既有 Live/ 消费方并存。
                "joint_limit_depth": self._reward_ppo_economy_depth_telemetry(),
                "joint_limit_depth_semantics": (
                    "max_intrusion_depth_frac is in band widths: 1.0 sits exactly on "
                    "the soft limit and >1.0 has already crossed it; projection "
                    "distances are fractions of the target-envelope span"
                ),
            },
            "advantage": {
                "pre_normalization_mean_std_min_max": pre_advantage,
                "post_normalization_mean_std_min_max": post_advantage,
                "post_normalization_finite": True,
                "dtype_tolerance": tolerance,
                "normalization_population": "whole_rollout_98304_samples",
            },
        }

    def _reward_ppo_economy_depth_telemetry(self):
        """Snapshot the qdes/actual depth counters for the economy receipt."""

        from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
            peek_qdes_depth_telemetry,
        )

        env = getattr(self.env, "unwrapped", self.env)
        return peek_qdes_depth_telemetry(env)

    @staticmethod
    def _economy_gradient_norm(parameters) -> torch.Tensor:
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not norms:
            raise RuntimeError("reward/PPO economy gradient group has no gradients")
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    def _run_reward_ppo_economy_optimizer(self, original_update):
        """Run the real PPO update while observing, never replacing, its clip."""

        policy = self.alg.policy
        named = list(policy.named_parameters())
        actor = [parameter for name, parameter in named if name.startswith("actor.")]
        critic = [parameter for name, parameter in named if name.startswith("critic.")]
        # 人话:噪声参数在 rsl-rl 里有两个名字 —— noise_std_type="scalar" 叫 std,
        # "log" 才叫 log_std。这里以前写死了 log_std,于是每一个 scalar 策略(四格
        # 全都是 sigma1p0 scalar)都被这道门判成"参数划分不合法"而永远发不出
        # scale4096。名字改成问本类已有的 ABI 权威(_policy_std_abi 自己就是
        # fail-closed:必须恰好暴露 std/log_std 之一、必须与 noise_std_type 自洽、
        # 必须非空浮点),门本身一点没放松:仍要求 actor/critic 非空、噪声参数恰好
        # 一个、并且三组之和穷尽 named_parameters()。
        abi = self._policy_std_abi()
        std_parameter_name = None if abi is None else abi["parameter_name"]
        std = [
            parameter for name, parameter in named if name == std_parameter_name
        ]
        covered = {id(parameter) for parameter in (*actor, *critic, *std)}
        if (
            not actor
            or not critic
            or len(std) != 1
            or len(covered) != len(named)
            or any(id(parameter) not in covered for _name, parameter in named)
        ):
            raise RuntimeError(
                "reward/PPO economy requires exact actor/critic/%s parameter partition"
                % (std_parameter_name or "log_std")
            )

        original_clip = torch.nn.utils.clip_grad_norm_
        captures = []

        def observed_clip(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            expected = list(policy.parameters())
            if [id(item) for item in parameters] != [id(item) for item in expected]:
                raise RuntimeError(
                    "reward/PPO economy observed an unexpected gradient clip parameter set"
                )
            pre = torch.stack(
                (
                    self._economy_gradient_norm(actor),
                    self._economy_gradient_norm(critic),
                    self._economy_gradient_norm(std),
                    self._economy_gradient_norm(parameters),
                )
            ).detach()
            result = original_clip(parameters, max_norm, *args, **kwargs)
            post = self._economy_gradient_norm(parameters).detach()
            captures.append(torch.cat((pre, post.reshape(1))))
            return result

        torch.nn.utils.clip_grad_norm_ = observed_clip
        try:
            result = original_update()
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        expected_minibatches = int(self.alg.num_learning_epochs) * int(
            self.alg.num_mini_batches
        )
        if len(captures) != expected_minibatches or expected_minibatches <= 0:
            raise RuntimeError(
                "reward/PPO economy gradient observation count differs from PPO"
            )
        captured = torch.stack(captures)
        summary = torch.stack(
            (captured.min(dim=0).values, captured.mean(dim=0), captured.max(dim=0).values)
        ).tolist()
        if not all(
            math.isfinite(float(item))
            for row in summary
            for item in row
        ):
            raise RuntimeError("reward/PPO economy gradient summary is non-finite")
        minimum, mean, maximum = summary
        max_grad_norm = float(self.alg.max_grad_norm)
        if maximum[4] > max_grad_norm * (1.0 + 1.0e-5):
            raise RuntimeError(
                "one reward/PPO economy minibatch post-clip norm exceeds max_grad_norm"
            )
        clip_factors = torch.clamp(
            max_grad_norm / (captured[:, 3] + 1.0e-6), max=1.0
        )
        clip_factor_summary = torch.stack(
            (clip_factors.min(), clip_factors.mean(), clip_factors.max())
        ).tolist()
        if not all(math.isfinite(float(value)) for value in clip_factor_summary):
            raise RuntimeError(
                "reward/PPO economy gradient clip-factor summary is non-finite"
            )

        def distribution(index):
            return {
                "min": float(minimum[index]),
                "mean": float(mean[index]),
                "max": float(maximum[index]),
            }

        return result, {
            "pre_clip_actor_mean_parameter_grad_norm": float(mean[0]),
            "pre_clip_critic_parameter_grad_norm": float(mean[1]),
            "pre_clip_std_parameter_grad_norm": float(mean[2]),
            "pre_clip_total_grad_norm": float(mean[3]),
            "post_clip_total_grad_norm": float(mean[4]),
            "pre_clip_actor_mean_parameter_grad_norm_distribution": distribution(0),
            "pre_clip_critic_parameter_grad_norm_distribution": distribution(1),
            "pre_clip_std_parameter_grad_norm_distribution": distribution(2),
            "pre_clip_total_grad_norm_distribution": distribution(3),
            "post_clip_total_grad_norm_distribution": distribution(4),
            "clip_factor_distribution": {
                key: float(value)
                for key, value in zip(
                    ("min", "mean", "max"), clip_factor_summary
                )
            },
            "max_grad_norm": max_grad_norm,
            "aggregation": "arithmetic_mean_over_optimizer_minibatches",
            "optimizer_minibatch_count": expected_minibatches,
        }

    def _reward_ppo_economy_post_update(self, result) -> dict:
        """Measure final-policy whole-rollout KL/clip without sampling RNG."""

        required_losses = {"surrogate", "value_function", "entropy"}
        if not isinstance(result, Mapping) or not required_losses.issubset(result):
            raise RuntimeError("reward/PPO economy PPO loss result is incomplete")
        losses = {name: float(result[name]) for name in required_losses}
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError("reward/PPO economy PPO loss is non-finite")
        storage = self.alg.storage
        observations = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        old_log_prob = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        old_mu = storage.mu.flatten(0, 1)
        old_sigma = storage.sigma.flatten(0, 1)
        kl_sum = torch.zeros((), dtype=observations.dtype, device=observations.device)
        clip_sum = torch.zeros_like(kl_sum)
        count = 0
        chunk_size = 8192
        with torch.inference_mode():
            for start in range(0, observations.shape[0], chunk_size):
                stop = min(start + chunk_size, observations.shape[0])
                self.alg.policy.update_distribution(observations[start:stop])
                new_log_prob = self.alg.policy.get_actions_log_prob(
                    actions[start:stop]
                )
                mu = self.alg.policy.action_mean
                sigma = self.alg.policy.action_std
                old_sigma_chunk = old_sigma[start:stop]
                old_mu_chunk = old_mu[start:stop]
                kl = torch.sum(
                    torch.log(sigma / old_sigma_chunk + 1.0e-5)
                    + (
                        torch.square(old_sigma_chunk)
                        + torch.square(old_mu_chunk - mu)
                    )
                    / (2.0 * torch.square(sigma))
                    - 0.5,
                    dim=-1,
                )
                ratio = torch.exp(new_log_prob - old_log_prob[start:stop])
                kl_sum += kl.sum()
                clip_sum += (
                    (ratio < (1.0 - float(self.alg.clip_param)))
                    | (ratio > (1.0 + float(self.alg.clip_param)))
                ).sum()
                count += int(stop - start)
        if count != _REWARD_PPO_ECONOMY_NUM_ENVS * _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE:
            raise RuntimeError("reward/PPO economy post-update rollout count differs")
        approx_kl, clip_fraction = (
            float(item) for item in torch.stack((kl_sum, clip_sum)).tolist()
        )
        approx_kl /= count
        clip_fraction /= count
        learning_rate = float(self.alg.learning_rate)
        values = (
            losses["surrogate"],
            losses["value_function"],
            losses["entropy"],
            approx_kl,
            learning_rate,
            clip_fraction,
        )
        if (
            not all(math.isfinite(item) for item in values)
            or learning_rate <= 0.0
            or not 0.0 <= clip_fraction <= 1.0
        ):
            raise RuntimeError("reward/PPO economy post-update PPO summary is invalid")
        return {
            "surrogate_loss": losses["surrogate"],
            "value_loss": losses["value_function"],
            "entropy_mean": losses["entropy"],
            "approx_kl": approx_kl,
            "approx_kl_semantics": "final_policy_vs_rollout_policy_whole_rollout",
            "learning_rate": learning_rate,
            "clip_fraction": clip_fraction,
            "clip_fraction_semantics": (
                "final_policy_probability_ratio_outside_ppo_clip_whole_rollout"
            ),
            "loss_entropy_semantics": (
                "arithmetic_mean_over_20_optimizer_minibatches"
            ),
        }

    def _emit_reward_ppo_economy_update(
        self,
        *,
        ppo_update: int,
        rollout: Mapping[str, object],
        ppo: Mapping[str, object],
        gradient: Mapping[str, object],
        policy: Mapping[str, object],
    ) -> dict:
        # 现场重新解析一次参数化(_policy_std_abi 自己 fail-closed),用来核对
        # 这条遥测有没有谎报 noise_std_type。解析不出来就当作不匹配,由下面的
        # 硬门拒绝,而不是默默放行。
        runtime_abi = self._policy_std_abi()
        expected_noise_std_type = (
            None if runtime_abi is None else runtime_abi["noise_std_type"]
        )
        record = {
            "event": _REWARD_PPO_ECONOMY_EVENT,
            "schema_version": _REWARD_PPO_ECONOMY_SCHEMA_VERSION,
            "status": "PASS",
            "ppo_update": int(ppo_update),
            "gate": {
                "num_envs": _REWARD_PPO_ECONOMY_NUM_ENVS,
                "steps_per_env_per_update": (
                    _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE
                ),
                "rollout_samples_per_update": (
                    _REWARD_PPO_ECONOMY_NUM_ENVS
                    * _REWARD_PPO_ECONOMY_STEPS_PER_UPDATE
                ),
            },
            "reward": dict(rollout["reward"]),
            "advantage": dict(rollout["advantage"]),
            "ppo": dict(ppo),
            "gradient": dict(gradient),
            "policy": {
                key: policy[key]
                for key in (
                    "noise_std_type",
                    "policy_std_min",
                    "policy_std_mean",
                    "policy_std_max",
                )
            },
            "checks": {
                "all_required_fields_present": True,
                "all_required_values_finite": True,
                "reward_sum_closure": "PASS",
                "post_advantage_zero_mean_unit_std": "PASS",
                # 记录位,不是准入位。N1 vendor probe 收据(materialize_n1_vendor_probe_
                # _gate_receipt.py)把它钉成 True,所以 N1 那条线照旧只收 log —— 那道门
                # 一点没动。211 四格不同:action_ball_211_four_grid_contract.py 的两个
                # sealed 探索包里,标准初始化那一包写死 noise_std_type="scalar",而
                # 2026-08-05 起"四格全部取标准初始化这一包"。所以对 211 来说,要求
                # noise_std_type=="log" 等于要求四格违反自己的封存合同。211 的真正下游
                # 消费者 action_ball_4096x5_prelong_gate.py 根本不读这个字段,只读
                # policy_std_min/mean/max。
                "noise_std_type_log": policy.get("noise_std_type") == "log",
                # 换上来的准入位:遥测不许谎报自己用的是哪套参数化。
                "noise_std_type_matches_runtime_abi": (
                    policy.get("noise_std_type") == expected_noise_std_type
                ),
                # 这条是真安全不变量,保持硬门。scalar 参数化下 std 是被直接优化的、
                # 没有 exp() 兜底,所以"每个 update 都复查 std>0"从"冗余"变成唯一防线,
                # 绝不能松。
                "policy_std_strictly_positive": float(
                    policy["policy_std_min"]
                )
                > 0.0,
            },
        }
        if not all(
            value is True
            for key, value in record["checks"].items()
            if key
            not in {
                "reward_sum_closure",
                "post_advantage_zero_mean_unit_std",
                "noise_std_type_log",
            }
        ):
            raise RuntimeError("reward/PPO economy update did not pass all checks")
        print(
            "HOPE_ACTION_BALL_REWARD_PPO_ECONOMY_UPDATE_JSON="
            + json.dumps(
                record,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return record

    @staticmethod
    def _rsl_rl_distribution_identity() -> dict:
        """Describe the distribution and import roots used by this process."""

        root = sys.modules.get("rsl_rl")
        runner_module = sys.modules.get(OnPolicyRunner.__module__)
        packages_distributions = getattr(
            importlib_metadata, "packages_distributions", None
        )
        if callable(packages_distributions):
            candidates = tuple(
                sorted(set(packages_distributions().get("rsl_rl", ())))
            )
        else:
            candidates = ()
        if not candidates:
            # Python 3.8's stdlib importlib.metadata predates
            # packages_distributions(), and some editable installs omit the
            # top-level-name index.  These are the historical/current names
            # that install the rsl_rl import package.
            available = []
            for candidate in ("rsl-rl-lib", "rsl-rl", "rsl_rl"):
                try:
                    importlib_metadata.version(candidate)
                except importlib_metadata.PackageNotFoundError:
                    continue
                available.append(candidate)
            candidates = tuple(available)
        distributions = []
        for name in candidates:
            try:
                version = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                version = None
            distributions.append({"name": name, "version": version})
        return {
            "distributions": distributions,
            "package_origin": getattr(root, "__file__", None),
            "runner_module": OnPolicyRunner.__module__,
            "runner_origin": getattr(runner_module, "__file__", None),
        }

    def _emit_rsl_rl_runtime_abi(
        self, *, normalizer_binding: Mapping[str, object]
    ) -> dict:
        """Emit one canonical run-log receipt before any simulator mutation."""

        previous = getattr(self, "_rsl_rl_runtime_abi_record", None)
        if previous is not None:
            return previous
        std_abi = self._policy_std_abi()
        record = {
            "event": _RSL_RL_RUNTIME_ABI_EVENT,
            "schema_version": _RSL_RL_RUNTIME_ABI_SCHEMA_VERSION,
            "runtime": self._rsl_rl_distribution_identity(),
            "capabilities": {
                "empirical_normalization_preflight": True,
                "positive_realized_policy_std_guard": True,
                "normalizer_binding": dict(normalizer_binding),
                # Parameter metadata is host-side.  Numeric std/LR reads are
                # restricted to the one post-optimizer update boundary.
                "policy_std_abi": std_abi,
            },
        }
        print(
            "HOPE_RSL_RL_RUNTIME_ABI_JSON="
            + json.dumps(
                record,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        self._rsl_rl_runtime_abi_record = record
        return record

    @staticmethod
    def _runtime_bootstrap_json_clone(value: object) -> object:
        """Detach one JSON-only receipt value from caller-owned mappings."""

        try:
            return json.loads(
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "runtime bootstrap binding is not a finite JSON value"
            ) from exc

    @staticmethod
    def _exact_resume_source_telemetry(
        state: Mapping[str, object],
    ) -> dict:
        """Copy only three tiny immutable logger/location scalar fields."""

        if not isinstance(state, Mapping):
            raise RuntimeError(
                "exact-resume source state must be a mapping"
            )
        result = {}
        for key in _EXACT_RESUME_TELEMETRY_KEYS:
            value = state.get(key)
            if value is not None and type(value) is not str:
                raise RuntimeError(
                    "exact-resume source telemetry must be None or "
                    f"string: {key}"
                )
            result[key] = value
        return result

    @staticmethod
    def _serialize_numpy_rng_state(state: object) -> dict:
        """Encode NumPy MT19937 state using weights-only-safe primitives."""

        if (
            type(state) is not tuple
            or len(state) != 5
            or state[0] != "MT19937"
            or not isinstance(state[1], np.ndarray)
            or state[1].dtype != np.dtype(np.uint32)
            or tuple(state[1].shape) != (624,)
            or type(state[2]) is not int
            or not 0 <= state[2] <= 624
            or type(state[3]) is not int
            or state[3] not in (0, 1)
            or type(state[4]) not in (int, float)
            or not math.isfinite(float(state[4]))
        ):
            raise RuntimeError(
                "NumPy RNG is not a canonical MT19937 state"
            )
        return {
            "schema_version": _NUMPY_RNG_STATE_SCHEMA_VERSION,
            "bit_generator": "MT19937",
            "state_uint32": [
                int(value) for value in state[1].tolist()
            ],
            "position": state[2],
            "has_gauss": state[3],
            "cached_gaussian": float(state[4]),
        }

    @staticmethod
    def _deserialize_numpy_rng_state(state: object) -> tuple:
        """Validate and reconstruct one private NumPy MT19937 state tuple."""

        expected_keys = {
            "schema_version",
            "bit_generator",
            "state_uint32",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        if (
            type(state) is not dict
            or set(state) != expected_keys
            or state.get("schema_version")
            != _NUMPY_RNG_STATE_SCHEMA_VERSION
            or state.get("bit_generator") != "MT19937"
        ):
            raise RuntimeError(
                "numpy_random_state does not match safe schema 1"
            )
        values = state["state_uint32"]
        if (
            type(values) is not list
            or len(values) != 624
            or any(
                type(value) is not int
                or value < 0
                or value > 0xFFFFFFFF
                for value in values
            )
        ):
            raise RuntimeError(
                "numpy_random_state state_uint32 is invalid"
            )
        position = state["position"]
        has_gauss = state["has_gauss"]
        cached_gaussian = state["cached_gaussian"]
        if (
            type(position) is not int
            or not 0 <= position <= 624
            or type(has_gauss) is not int
            or has_gauss not in (0, 1)
            or type(cached_gaussian) not in (int, float)
            or not math.isfinite(float(cached_gaussian))
        ):
            raise RuntimeError(
                "numpy_random_state cursor/cache is invalid"
            )
        return (
            "MT19937",
            np.asarray(values, dtype=np.uint32),
            position,
            has_gauss,
            float(cached_gaussian),
        )

    @staticmethod
    def _exact_resume_tree_sha256(value: object) -> str:
        """Hash one live state tree without pickle or a whole-tree copy.

        Exact-resume verification runs while the policy and Adam moments can
        already occupy most of a GPU.  Hash one tensor at a time after a
        read-only CPU transfer; never ``deepcopy`` the checkpoint core.
        """

        digest = hashlib.sha256()
        active_containers = set()

        def emit(raw: bytes) -> None:
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)

        def walk(item: object) -> None:
            if torch.is_tensor(item):
                tensor = item.detach().to(device="cpu").contiguous()
                emit(b"tensor")
                emit(str(tensor.dtype).encode("ascii"))
                emit(
                    json.dumps(
                        list(tensor.shape),
                        separators=(",", ":"),
                    ).encode("ascii")
                )
                try:
                    emit(
                        tensor.reshape(-1)
                        .view(torch.uint8)
                        .numpy()
                        .tobytes(order="C")
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "exact-resume tensor cannot be hashed losslessly"
                    ) from exc
                return

            is_container = isinstance(item, (Mapping, list, tuple))
            identity = id(item)
            if is_container:
                if identity in active_containers:
                    raise RuntimeError(
                        "exact-resume state contains a cyclic container"
                    )
                active_containers.add(identity)
            try:
                if item is None:
                    emit(b"none")
                elif type(item) is bool:
                    emit(b"bool1" if item else b"bool0")
                elif type(item) is int:
                    emit(b"int")
                    emit(str(item).encode("ascii"))
                elif type(item) is float:
                    if not math.isfinite(item):
                        raise RuntimeError(
                            "exact-resume state contains a non-finite float"
                        )
                    emit(b"float")
                    emit(item.hex().encode("ascii"))
                elif type(item) is str:
                    emit(b"str")
                    emit(item.encode("utf-8"))
                elif type(item) is bytes:
                    emit(b"bytes")
                    emit(item)
                elif isinstance(item, Mapping):
                    emit(b"mapping")
                    keyed = []
                    for key in item:
                        if type(key) not in (int, str):
                            raise RuntimeError(
                                "exact-resume mapping keys must be exact "
                                "ints or strings"
                            )
                        keyed.append(
                            (
                                MotionOnPolicyRunner._exact_resume_tree_sha256(
                                    key
                                ),
                                key,
                            )
                        )
                    for key_digest, key in sorted(
                        keyed, key=lambda row: row[0]
                    ):
                        emit(key_digest.encode("ascii"))
                        walk(key)
                        walk(item[key])
                elif isinstance(item, tuple):
                    emit(b"tuple")
                    emit(str(len(item)).encode("ascii"))
                    for child in item:
                        walk(child)
                elif isinstance(item, list):
                    emit(b"list")
                    emit(str(len(item)).encode("ascii"))
                    for child in item:
                        walk(child)
                elif isinstance(item, np.ndarray):
                    if item.dtype.hasobject:
                        raise RuntimeError(
                            "exact-resume object ndarray is not hashable"
                        )
                    emit(b"ndarray")
                    emit(str(item.dtype).encode("ascii"))
                    emit(
                        json.dumps(
                            list(item.shape),
                            separators=(",", ":"),
                        ).encode("ascii")
                    )
                    emit(item.tobytes(order="C"))
                elif isinstance(item, np.generic):
                    walk(item.item())
                else:
                    raise RuntimeError(
                        "exact-resume state contains unsupported type "
                        f"{type(item).__module__}."
                        f"{type(item).__qualname__}"
                    )
            finally:
                if is_container:
                    active_containers.remove(identity)

        walk(value)
        return digest.hexdigest()

    @staticmethod
    def _exact_resume_without_telemetry(state: Mapping[str, object]) -> dict:
        """Return a shallow exact-state view without three location fields."""

        if not isinstance(state, Mapping):
            raise RuntimeError("exact-resume state must be a mapping")
        return {
            key: value
            for key, value in state.items()
            if key not in _EXACT_RESUME_TELEMETRY_KEYS
        }

    @staticmethod
    def _exact_resume_live_envelope(content: Mapping[str, object]) -> dict:
        """Seal the small JSON-only live-state receipt."""

        cloned = MotionOnPolicyRunner._runtime_bootstrap_json_clone(
            dict(content)
        )
        raw = json.dumps(
            cloned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return {
            "schema_version": _EXACT_RESUME_LIVE_STATE_SCHEMA_VERSION,
            "kind": _EXACT_RESUME_LIVE_STATE_KIND,
            "content": cloned,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _capture_exact_resume_live_state_content(self) -> dict:
        """Capture the no-step continuation core at source-iteration semantics."""

        if not self._formal_action_ball_runtime_bootstrap_required():
            raise RuntimeError(
                "exact-resume live state requires formal ActionBall"
            )
        source_iteration = getattr(
            self, "_exact_resume_loaded_source_iteration", None
        )
        current_iteration = self.current_learning_iteration
        if (
            type(source_iteration) is not int
            or source_iteration < 0
            or type(current_iteration) is not int
            or current_iteration != source_iteration + 1
        ):
            raise RuntimeError(
                "exact-resume live-state iteration cursor drifted"
            )
        roundtrip_pending = getattr(
            self, "_exact_resume_roundtrip_pending", None
        )
        reset_pending = getattr(
            self, "_action_ball_resume_reset_pending", None
        )
        if roundtrip_pending is not True or reset_pending is not True:
            raise RuntimeError(
                "exact-resume live state requires an unused strict-load "
                "window"
            )
        source_telemetry = getattr(
            self, "_exact_resume_loaded_source_telemetry", None
        )
        if (
            type(source_telemetry) is not dict
            or set(source_telemetry) != set(_EXACT_RESUME_TELEMETRY_KEYS)
        ):
            raise RuntimeError(
                "exact-resume live state lost source telemetry"
            )

        self.current_learning_iteration = source_iteration
        try:
            exact_state = self._build_exact_resume_state()
        finally:
            self.current_learning_iteration = current_iteration
        for key in _EXACT_RESUME_TELEMETRY_KEYS:
            exact_state[key] = source_telemetry[key]
        if exact_state.get("next_learning_iteration") != current_iteration:
            raise RuntimeError(
                "exact-resume live state rebuilt the wrong next iteration"
            )

        environment_state = exact_state.get(
            "environment_resume_state"
        )
        if not isinstance(environment_state, Mapping):
            raise RuntimeError(
                "exact-resume live state lacks environment state"
            )
        common_step_counter = environment_state.get(
            "common_step_counter"
        )
        source_common_step_counter = getattr(
            self,
            "_exact_resume_loaded_source_common_step_counter",
            None,
        )
        if (
            type(common_step_counter) is not int
            or common_step_counter < 0
            or type(source_common_step_counter) is not int
            or source_common_step_counter < 0
        ):
            raise RuntimeError(
                "exact-resume live state has an invalid common-step counter"
            )
        common_step_counter_delta = (
            common_step_counter - source_common_step_counter
        )
        if common_step_counter_delta != 0:
            raise RuntimeError(
                "exact-resume live state crossed an environment step/reset "
                "boundary"
            )

        policy = getattr(self.alg, "policy", None)
        policy_state = getattr(policy, "state_dict", None)
        optimizer = getattr(self.alg, "optimizer", None)
        optimizer_state = getattr(optimizer, "state_dict", None)
        if not callable(policy_state) or not callable(optimizer_state):
            raise RuntimeError(
                "exact-resume live state requires policy and optimizer "
                "state_dict()"
            )
        actor_normalizer = self._frozen_eval_normalizer_payload("actor")
        critic_normalizer = self._frozen_eval_normalizer_payload("critic")
        rng_state = {
            "python_random_state": exact_state["python_random_state"],
            "numpy_random_state": exact_state["numpy_random_state"],
            "torch_random_state": exact_state["torch_random_state"],
            "torch_cuda_random_states": exact_state[
                "torch_cuda_random_states"
            ],
            "torch_cuda_device_count": exact_state[
                "torch_cuda_device_count"
            ],
        }
        bootstrap_binding = {
            key: exact_state[key]
            for key in (
                _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY,
                _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY,
                _RUNTIME_BOOTSTRAP_RECEIPT_KEY,
            )
        }
        content = {
            "schema_version": _EXACT_RESUME_LIVE_STATE_SCHEMA_VERSION,
            "kind": _EXACT_RESUME_LIVE_STATE_KIND,
            "source_embedded_iteration": source_iteration,
            "current_learning_iteration": current_iteration,
            "roundtrip_pending": roundtrip_pending,
            "resume_reset_pending": reset_pending,
            "model_state_sha256": self._exact_resume_tree_sha256(
                policy_state()
            ),
            "optimizer_state_sha256": self._exact_resume_tree_sha256(
                optimizer_state()
            ),
            "actor_normalizer_state_sha256": (
                self._exact_resume_tree_sha256(actor_normalizer)
            ),
            "critic_normalizer_state_sha256": (
                self._exact_resume_tree_sha256(critic_normalizer)
            ),
            "exact_resume_state_sha256": self._exact_resume_tree_sha256(
                self._exact_resume_without_telemetry(exact_state)
            ),
            "environment_resume_state_sha256": (
                self._exact_resume_tree_sha256(environment_state)
            ),
            "rng_state_sha256": self._exact_resume_tree_sha256(
                rng_state
            ),
            "runtime_bootstrap_binding_sha256": (
                self._exact_resume_tree_sha256(bootstrap_binding)
            ),
            "common_step_counter": common_step_counter,
            "common_step_counter_delta": common_step_counter_delta,
        }
        content["live_core_sha256"] = self._exact_resume_tree_sha256(
            content
        )
        return content

    def _install_exact_resume_live_state_baseline(
        self,
        *,
        loaded_checkpoint: Mapping[str, object],
        source_exact_state: Mapping[str, object],
    ) -> None:
        """Prove the strict load restored its source, then freeze a baseline."""

        live_content = self._capture_exact_resume_live_state_content()
        source_model_sha256 = self._exact_resume_tree_sha256(
            loaded_checkpoint["model_state_dict"]
        )
        source_optimizer_sha256 = self._exact_resume_tree_sha256(
            loaded_checkpoint["optimizer_state_dict"]
        )
        if (
            live_content["model_state_sha256"] != source_model_sha256
            or live_content["optimizer_state_sha256"]
            != source_optimizer_sha256
        ):
            raise RuntimeError(
                "strict ActionBall load did not restore policy/optimizer "
                "state exactly"
            )

        # The current runtime may deliberately live in a different no-clobber
        # namespace.  Preflight has already proved its location-free bootstrap
        # lineage; compare every other exact field against the source while
        # substituting the newly minted current receipt.
        expected_exact_state = dict(source_exact_state)
        for key in _EXACT_RESUME_TELEMETRY_KEYS:
            expected_exact_state.pop(key, None)
        current_binding = self._validated_runtime_bootstrap_binding()
        expected_exact_state.update(current_binding)
        if (
            live_content["exact_resume_state_sha256"]
            != self._exact_resume_tree_sha256(expected_exact_state)
        ):
            raise RuntimeError(
                "strict ActionBall load did not restore exact RNG/environment "
                "continuation state"
            )
        self._exact_resume_live_state_baseline = (
            self._runtime_bootstrap_json_clone(live_content)
        )

    def exact_resume_live_state_receipt(self) -> dict:
        """Return a fresh proof that no state changed since strict load."""

        baseline = getattr(
            self, "_exact_resume_live_state_baseline", None
        )
        if not isinstance(baseline, dict):
            raise RuntimeError(
                "exact-resume live-state baseline is unavailable"
            )
        current = self._capture_exact_resume_live_state_content()
        if current != baseline:
            changed = sorted(
                key
                for key in set(current).union(baseline)
                if current.get(key) != baseline.get(key)
            )
            raise RuntimeError(
                "exact-resume live state drifted after strict load: "
                + ", ".join(changed)
            )
        return self._exact_resume_live_envelope(current)

    def _runtime_bootstrap_expected_paths(self) -> dict:
        if self.log_dir is None:
            raise RuntimeError(
                "ActionBall runtime bootstrap requires a training log_dir"
            )
        params = pathlib.Path(self.log_dir).expanduser().resolve() / "params"
        return {
            "training_contract": params / "training_contract.json",
            "environment_config_pickle": params / "env.pkl",
            "agent_config_pickle": params / "agent.pkl",
            "runtime_identity": (
                params / "action_ball_frozen_eval_runtime.json"
            ),
            "receipt": (
                params / "action_ball_runtime_bootstrap_receipt.json"
            ),
        }

    def _formal_action_ball_runtime_bootstrap_required(self) -> bool:
        return (
            self._strict_exact_resume_target_mode() == "action_ball"
            and getattr(
                self, "training_contract_lineage_exact", None
            )
            is True
        )

    def bind_runtime_bootstrap_receipt(
        self,
        *,
        content_sha256: str,
        artifact_receipt: Mapping[str, object],
    ) -> None:
        """Bind the trainer-minted post-dump runtime receipt exactly once.

        Construction is intentionally too early for this binding.  The
        trainer publishes ``env.pkl``, ``agent.pkl``, runtime identity, then
        the immutable receipt and passes the in-memory publication result
        here.  Re-reading a path alone would let a jointly replaced
        checkpoint+receipt choose its own expected identity.
        """

        if not self._formal_action_ball_runtime_bootstrap_required():
            raise RuntimeError(
                "runtime bootstrap receipts require formal exact-lineage "
                "ActionBall"
            )
        if self.training_launch_claim_sha256 is None:
            raise RuntimeError(
                "formal ActionBall runtime bootstrap lacks launch claim"
            )
        if (
            type(content_sha256) is not str
            or len(content_sha256) != 64
            or content_sha256 != content_sha256.lower()
            or any(
                character not in "0123456789abcdef"
                for character in content_sha256
            )
        ):
            raise ValueError(
                "runtime bootstrap content SHA must be 64 lowercase hex"
            )
        if not isinstance(artifact_receipt, Mapping):
            raise TypeError(
                "runtime bootstrap artifact receipt must be a mapping"
            )

        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_evaluation_inbox as inbox_protocol,
            action_ball_runtime_bootstrap as bootstrap_protocol,
        )

        paths = self._runtime_bootstrap_expected_paths()
        supplied_artifact = self._runtime_bootstrap_json_clone(
            dict(artifact_receipt)
        )
        actual_artifact = inbox_protocol.artifact_receipt(paths["receipt"])
        if supplied_artifact != actual_artifact:
            raise RuntimeError(
                "runtime bootstrap receipt artifact differs from the "
                "trainer publication result"
            )
        document = inbox_protocol.strict_read_json(
            paths["receipt"],
            label="ActionBall runtime bootstrap receipt",
        )
        try:
            content = document["content"]
            source = content["source"]
            validated = (
                bootstrap_protocol.validate_runtime_bootstrap_receipt_document(
                    document,
                    expected_repo_root=source["repo_root"],
                    expected_task_id=bootstrap_protocol.TASK_ID,
                    expected_training_launch_claim_sha256=(
                        self.training_launch_claim_sha256
                    ),
                    expected_training_contract_path=paths[
                        "training_contract"
                    ],
                    expected_environment_config_pickle_path=paths[
                        "environment_config_pickle"
                    ],
                    expected_agent_config_pickle_path=paths[
                        "agent_config_pickle"
                    ],
                    expected_runtime_identity_path=paths[
                        "runtime_identity"
                    ],
                    expected_source_commit_oid=source[
                        "head_commit_oid"
                    ],
                )
            )
            lineage_sha256 = (
                bootstrap_protocol.runtime_bootstrap_lineage_payload_sha256(
                    content
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "ActionBall runtime bootstrap receipt failed live "
                "validation"
            ) from exc
        if document.get("content_sha256") != content_sha256:
            raise RuntimeError(
                "runtime bootstrap content SHA differs from publication"
            )
        detached_content = self._runtime_bootstrap_json_clone(validated)
        candidate = {
            _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY: content_sha256,
            _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY: lineage_sha256,
            _RUNTIME_BOOTSTRAP_RECEIPT_KEY: supplied_artifact,
            "content": detached_content,
        }
        existing = {
            _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY: (
                self.runtime_bootstrap_receipt_sha256
            ),
            _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY: (
                self.runtime_bootstrap_lineage_payload_sha256
            ),
            _RUNTIME_BOOTSTRAP_RECEIPT_KEY: (
                self.runtime_bootstrap_receipt
            ),
            "content": self._runtime_bootstrap_content,
        }
        if self.runtime_bootstrap_receipt_sha256 is not None:
            if existing != candidate:
                raise RuntimeError(
                    "runtime bootstrap receipt is already bound to "
                    "different bytes"
                )
            return
        self.runtime_bootstrap_receipt_sha256 = content_sha256
        self.runtime_bootstrap_lineage_payload_sha256 = lineage_sha256
        self.runtime_bootstrap_receipt = supplied_artifact
        self._runtime_bootstrap_content = detached_content

    def _validated_runtime_bootstrap_binding(self) -> dict:
        """Reopen the current run receipt and every artifact it binds."""

        if not self._formal_action_ball_runtime_bootstrap_required():
            return {}
        if (
            self.runtime_bootstrap_receipt_sha256 is None
            or self.runtime_bootstrap_lineage_payload_sha256 is None
            or self.runtime_bootstrap_receipt is None
            or self._runtime_bootstrap_content is None
        ):
            raise RuntimeError(
                "formal ActionBall checkpoint boundary reached before "
                "runtime bootstrap receipt binding"
            )

        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_evaluation_inbox as inbox_protocol,
            action_ball_runtime_bootstrap as bootstrap_protocol,
        )

        paths = self._runtime_bootstrap_expected_paths()
        if (
            inbox_protocol.artifact_receipt(paths["receipt"])
            != self.runtime_bootstrap_receipt
        ):
            raise RuntimeError(
                "ActionBall runtime bootstrap receipt bytes drifted"
            )
        document = inbox_protocol.strict_read_json(
            paths["receipt"],
            label="ActionBall runtime bootstrap receipt",
        )
        try:
            source = document["content"]["source"]
            content = (
                bootstrap_protocol.validate_runtime_bootstrap_receipt_document(
                    document,
                    expected_repo_root=source["repo_root"],
                    expected_task_id=bootstrap_protocol.TASK_ID,
                    expected_training_launch_claim_sha256=(
                        self.training_launch_claim_sha256
                    ),
                    expected_training_contract_path=paths[
                        "training_contract"
                    ],
                    expected_environment_config_pickle_path=paths[
                        "environment_config_pickle"
                    ],
                    expected_agent_config_pickle_path=paths[
                        "agent_config_pickle"
                    ],
                    expected_runtime_identity_path=paths[
                        "runtime_identity"
                    ],
                    expected_source_commit_oid=source[
                        "head_commit_oid"
                    ],
                )
            )
            lineage_sha256 = (
                bootstrap_protocol.runtime_bootstrap_lineage_payload_sha256(
                    document["content"]
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "ActionBall runtime bootstrap receipt or bound artifact "
                "drifted"
            ) from exc
        if (
            document.get("content_sha256")
            != self.runtime_bootstrap_receipt_sha256
            or lineage_sha256
            != self.runtime_bootstrap_lineage_payload_sha256
            or self._runtime_bootstrap_json_clone(content)
            != self._runtime_bootstrap_content
        ):
            raise RuntimeError(
                "ActionBall runtime bootstrap identity differs from its "
                "one-shot runner binding"
            )
        return {
            _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY: (
                self.runtime_bootstrap_receipt_sha256
            ),
            _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY: (
                self.runtime_bootstrap_lineage_payload_sha256
            ),
            _RUNTIME_BOOTSTRAP_RECEIPT_KEY: (
                self._runtime_bootstrap_json_clone(
                    self.runtime_bootstrap_receipt
                )
            ),
        }

    def _checkpoint_infos(self, infos=None) -> dict:
        """Build the one canonical checkpoint-info envelope.

        Curriculum control checkpoints use the same exact-resume payload as
        ordinary periodic checkpoints but deliberately skip ONNX/W&B export.
        Keeping one builder prevents the asynchronous evaluator path from
        accidentally saving a weaker resume contract.
        """

        self._action_ball_full_mdp_require_checkpointable()

        if infos is None:
            infos = {}
        elif not isinstance(infos, dict):
            raise TypeError(
                "runner checkpoint infos must be a dict (contract binding and exact-resume "
                "state are embedded there)"
            )
        else:
            infos = dict(infos)
        if self.training_contract_sha256 is not None:
            infos[CHECKPOINT_CONTRACT_SCHEMA_KEY] = int(
                self.training_contract_schema_version
            )
            infos[CHECKPOINT_CONTRACT_SHA_KEY] = self.training_contract_sha256
            infos[CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY] = (
                1 if self.training_contract_lineage_exact else 0
            )
        if self.training_launch_claim_sha256 is not None:
            infos[CHECKPOINT_LAUNCH_CLAIM_SHA_KEY] = self.training_launch_claim_sha256
        runtime_bootstrap_binding = (
            self._validated_runtime_bootstrap_binding()
        )
        infos.update(runtime_bootstrap_binding)
        # --- 精确续训状态(jiayi hitterobs 9f684ae5 按 main 语义移植)---------------------------
        # 人话:环境的 common_step_counter 驱动所有"随步数渐进"的课程(扰动 ramp、自适应
        # sigma、成功门控扩幅…),但 base rsl_rl 的存档只有权重/优化器/迭代号 —— 不把它和各
        # 命令项的课程状态一起存进 PT,续训时全部课程静默回到第 0 步,对 2 万+ iter 的长训
        # 是真炸弹。main 的既有惯例是把 checkpoint 元数据放进 infos(合同绑定键就在里面),
        # 所以续训状态也走 infos,而不是像 hitterobs 那样整体复刻 base 的 save 再塞顶层键:
        # 复刻会随 rsl_rl 升级悄悄漂移,而且 base save 写完盘才排队上传 W&B,走 infos 让云端
        # 副本从第一份字节起就带状态。键名沿用 jiayi 的 hope_exact_resume_state 便于跨栈对账。
        infos["hope_exact_resume_state"] = self._build_exact_resume_state(
            runtime_bootstrap_binding=runtime_bootstrap_binding
        )
        return infos

    def save(self, path: str, infos=None):
        """Save the model and training information."""

        infos = self._checkpoint_infos(infos)
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            trained_with_obs_norm = bool(self.empirical_normalization)
            obs_norm_baked = export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=self.obs_normalizer if trained_with_obs_norm else None,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename,
                obs_norm_baked=obs_norm_baked,
                trained_with_obs_norm=trained_with_obs_norm,
                source_checkpoint_path=path,
            )
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # Link the input motion artifact(s) to this run (lineage bookkeeping only — a W&B API
            # failure here must not kill the training run). W&B expects registry refs to include an
            # alias (for example, collection:latest).
            if self.registry_name is not None:
                registry_names = (
                    self.registry_name if isinstance(self.registry_name, (list, tuple)) else [self.registry_name]
                )
                for registry_name in registry_names:
                    try:
                        wandb.run.use_artifact(registry_name)
                    except Exception as e:
                        print(f"[MotionOnPolicyRunner] WARNING: use_artifact({registry_name!r}) failed: {e}")
                self.registry_name = None

    @staticmethod
    def _frozen_eval_state_binding(value: object) -> dict:
        """Hash tensor state without relying on pickle/zip serialization.

        The sidecar request binds actor and critic observation-normalizer
        state extracted from the exact runner.  Tensor bytes are hashed in a
        deterministic sorted tree; CUDA tensors are copied read-only to CPU.
        """

        digest = hashlib.sha256()
        byte_count = 0

        def emit(tag: str, payload: bytes = b"") -> None:
            nonlocal byte_count
            encoded_tag = tag.encode("utf-8")
            digest.update(len(encoded_tag).to_bytes(8, "big"))
            digest.update(encoded_tag)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            byte_count += len(payload)

        def visit(item: object) -> None:
            if torch.is_tensor(item):
                tensor = item.detach().to(device="cpu").contiguous()
                emit(
                    "tensor:"
                    + str(tensor.dtype)
                    + ":"
                    + json.dumps(list(tensor.shape), separators=(",", ":")),
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
                )
                return
            if isinstance(item, dict):
                if any(type(key) is not str for key in item):
                    raise TypeError(
                        "normalizer state mappings must use string keys"
                    )
                emit("mapping-begin")
                for key in sorted(item):
                    emit("key", key.encode("utf-8"))
                    visit(item[key])
                emit("mapping-end")
                return
            if isinstance(item, (list, tuple)):
                emit("sequence-begin:" + type(item).__name__)
                for child in item:
                    visit(child)
                emit("sequence-end")
                return
            if item is None or type(item) in (bool, int, float, str):
                emit(
                    "scalar",
                    json.dumps(
                        item,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("ascii"),
                )
                return
            raise TypeError(
                "normalizer state contains unsupported value "
                f"{type(item).__name__}"
            )

        visit(value)
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": byte_count,
        }

    def _frozen_eval_normalizer_payload(self, role: str) -> dict:
        """Return the effective RSL-RL normalizer for one policy input.

        Current RSL-RL names the critic input ``privileged_obs_normalizer``.
        Some older lineages used ``critic_obs_normalizer`` instead.  The
        frozen-evaluation protocol keeps the semantic wire name
        ``critic_obs_normalizer``, but it must hash the module that is
        actually applied to privileged observations rather than silently
        hashing an absent attribute as disabled.
        """

        if role == "actor":
            candidates = ("obs_normalizer",)
        elif role == "critic":
            candidates = (
                "privileged_obs_normalizer",
                "critic_obs_normalizer",
            )
        else:
            raise ValueError(
                "frozen-eval normalizer role must be actor or critic"
            )

        present = [
            (name, getattr(self, name))
            for name in candidates
            if hasattr(self, name)
        ]
        if not present:
            raise RuntimeError(
                f"{role} observation normalizer is absent from the runner"
            )
        normalizer = present[0][1]
        if any(value is not normalizer for _name, value in present[1:]):
            raise RuntimeError(
                f"{role} observation normalizer aliases disagree"
            )

        empirical = getattr(self, "empirical_normalization", None)
        if type(empirical) is not bool:
            raise RuntimeError(
                "runner empirical_normalization must be an exact bool"
            )
        if normalizer is None:
            if empirical:
                raise RuntimeError(
                    f"empirical {role} observation normalizer is missing"
                )
            return {"enabled": False}

        state_dict = getattr(normalizer, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError(
                f"{present[0][0]} lacks a deterministic state_dict()"
            )
        enabled = is_empirical_normalizer(normalizer)
        if empirical and not enabled:
            raise RuntimeError(
                f"empirical {role} observation normalizer is a no-op"
            )
        if not empirical and enabled:
            raise RuntimeError(
                f"disabled {role} observation normalization has a live "
                "transform"
            )
        return {
            # Wire compatibility: ``enabled`` means the checkpoint owns a
            # concrete normalizer module (including RSL-RL's Identity
            # placeholder), while the exact module state and training
            # contract determine whether it is an empirical transform.
            "enabled": True,
            "state": state_dict(),
        }

    def _frozen_eval_runner_bindings(
        self, *, policy_generation: int
    ) -> dict:
        """Read the exact outer training recipe needed by sidecar requests."""

        if type(policy_generation) is not int or policy_generation < 0:
            raise RuntimeError(
                "action-ball frozen evaluation requires a non-negative "
                "policy generation"
            )
        if not self._formal_action_ball_runtime_bootstrap_required():
            raise RuntimeError(
                "frozen evaluation requires formal exact-lineage ActionBall"
            )
        if self.log_dir is None or self.training_contract_sha256 is None:
            raise RuntimeError(
                "action-ball frozen evaluation requires a bound training log"
            )
        contract_path = (
            pathlib.Path(self.log_dir) / "params" / "training_contract.json"
        )
        raw = contract_path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.training_contract_sha256:
            raise RuntimeError(
                "training_contract.json changed before frozen evaluation"
            )

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeError(
                        f"training contract repeats JSON key {key!r}"
                    )
                result[key] = value
            return result

        try:
            contract = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    RuntimeError(
                        "training contract contains non-finite "
                        f"number {token}"
                    )
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "training_contract.json is not strict UTF-8 JSON"
            ) from exc
        if not isinstance(contract, dict):
            raise RuntimeError("training contract must be a JSON object")
        try:
            ppo = contract["action_ball_ppo_runner_recipe"]["sha256"]
            reward = contract["effective_reward_recipe"]["sha256"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "training contract lacks ActionBall PPO/Reward identity"
            ) from exc
        for name, value in (
            ("ppo_recipe_sha256", ppo),
            ("reward_sha256", reward),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or value != value.lower()
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise RuntimeError(f"{name} is not a SHA-256 digest")

        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_evaluation_inbox as inbox_protocol,
            action_ball_frozen_eval_identity as runtime_identity,
        )

        params_dir = contract_path.parent
        env_pickle_path = params_dir / "env.pkl"
        agent_pickle_path = params_dir / "agent.pkl"
        identity_path = (
            params_dir / "action_ball_frozen_eval_runtime.json"
        )
        if self.training_launch_claim_sha256 is None:
            raise RuntimeError(
                "formal frozen evaluation lacks training launch-claim identity"
            )
        runtime_bootstrap_binding = (
            self._validated_runtime_bootstrap_binding()
        )
        try:
            identity_document = inbox_protocol.strict_read_json(
                identity_path,
                label="ActionBall frozen-evaluation runtime identity",
            )
            identity_content = identity_document["content"]
            repo_root = identity_content["source"]["repo_root"]
            task_id = identity_content["task_id"]
            runtime_identity.validate_runtime_identity_document(
                identity_document,
                repo_root=repo_root,
                task_id=task_id,
                training_launch_claim_sha256=(
                    self.training_launch_claim_sha256
                ),
                training_contract_path=contract_path,
                environment_config_pickle_path=env_pickle_path,
                agent_config_pickle_path=agent_pickle_path,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "ActionBall frozen-evaluation runtime identity drifted"
            ) from exc

        return {
            "schema_version": (
                _ACTION_BALL_FROZEN_EVAL_CONTROL_SCHEMA_VERSION
            ),
            "training_contract_sha256": self.training_contract_sha256,
            "training_contract": inbox_protocol.artifact_receipt(
                contract_path
            ),
            "environment_config_pickle": (
                inbox_protocol.artifact_receipt(env_pickle_path)
            ),
            "agent_config_pickle": inbox_protocol.artifact_receipt(
                agent_pickle_path
            ),
            "runtime_identity": inbox_protocol.artifact_receipt(
                identity_path
            ),
            **runtime_bootstrap_binding,
            "training_launch_claim_sha256": (
                self.training_launch_claim_sha256
            ),
            "ppo_recipe_sha256": ppo,
            "reward_sha256": reward,
            "policy_generation": policy_generation,
            "policy_state": self._frozen_eval_state_binding(
                self.alg.policy.state_dict()
            ),
            "actor_obs_normalizer": self._frozen_eval_state_binding(
                self._frozen_eval_normalizer_payload("actor")
            ),
            "critic_obs_normalizer": self._frozen_eval_state_binding(
                self._frozen_eval_normalizer_payload("critic")
            ),
        }

    def _action_ball_frozen_eval_term(self):
        """Return the sole runtime term owning the frozen-eval lifecycle."""

        if self._strict_exact_resume_target_mode() != "action_ball":
            return None
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "command_manager", None)
        if manager is None:
            raise RuntimeError(
                "action-ball frozen evaluation requires command_manager"
            )
        matches = []
        for raw_name in tuple(getattr(manager, "active_terms", ())):
            term = manager.get_term(raw_name)
            if callable(
                getattr(
                    term,
                    "action_ball_frozen_evaluation_boundary",
                    None,
                )
            ):
                matches.append(term)
        if len(matches) != 1:
            raise RuntimeError(
                "action-ball requires exactly one frozen-evaluation "
                f"runtime owner, observed {len(matches)}"
            )
        return matches[0]

    def _action_ball_control_checkpoint(
        self,
        *,
        step: int,
        purpose: str,
        request_seq: int,
    ) -> pathlib.Path:
        """Write one no-clobber exact-resume checkpoint at a PPO boundary."""

        if self.log_dir is None:
            raise RuntimeError(
                "curriculum control checkpoint requires log_dir"
            )
        if (
            type(step) is not int
            or step < 0
            or type(request_seq) is not int
            or request_seq < 0
            or purpose not in (
                "policy_snapshot",
                "request_persisted",
                "evidence_consumed",
            )
        ):
            raise ValueError("invalid curriculum control checkpoint identity")
        if getattr(self, "_action_ball_full_mdp_run_mode", None) == _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE:
            self._action_ball_full_mdp_require_checkpointable()
        root = pathlib.Path(self.log_dir) / "curriculum_control"
        root.mkdir(parents=True, exist_ok=True)
        namespace = root / (
            f"update_{step:020d}_{purpose}_request_{request_seq:020d}"
        )
        try:
            namespace.mkdir()
        except FileExistsError as exc:
            checkpoint = namespace / "resume.pt"
            if (
                purpose == "evidence_consumed"
                and checkpoint.is_file()
                and not checkpoint.is_symlink()
                and getattr(self, "_loaded_checkpoint_path", None)
                == str(checkpoint.resolve())
            ):
                return checkpoint
            raise RuntimeError(
                "curriculum control namespace is already spent: "
                f"{namespace}"
            ) from exc
        checkpoint = namespace / "resume.pt"
        # RSL-RL invokes ``alg.update()`` before assigning
        # ``current_learning_iteration = it``.  This boundary runs from the
        # update wrapper, so on PPO update N the public field can still say
        # N-1.  Persist the completed update named by ``step`` explicitly;
        # otherwise request generation N points at a checkpoint whose top
        # level ``iter`` and exact ``next_learning_iteration`` are stale.
        observed_iteration = self.current_learning_iteration
        self.current_learning_iteration = step
        try:
            infos = self._checkpoint_infos()
            OnPolicyRunner.save(self, str(checkpoint), infos)
        finally:
            self.current_learning_iteration = observed_iteration
        try:
            checkpoint_stat = checkpoint.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "curriculum control checkpoint was not materialized"
            ) from exc
        if (
            not stat.S_ISREG(checkpoint_stat.st_mode)
            or checkpoint_stat.st_nlink != 1
            or checkpoint_stat.st_size <= 0
        ):
            raise RuntimeError(
                "curriculum control checkpoint was not durably materialized"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        checkpoint_fd = os.open(str(checkpoint), flags)
        try:
            opened = os.fstat(checkpoint_fd)
            if (
                opened.st_dev != checkpoint_stat.st_dev
                or opened.st_ino != checkpoint_stat.st_ino
                or opened.st_size != checkpoint_stat.st_size
            ):
                raise RuntimeError(
                    "curriculum control checkpoint changed while opening"
                )
            os.fsync(checkpoint_fd)
        finally:
            os.close(checkpoint_fd)
        self._joint_safety_fsync_directory(namespace)
        self._joint_safety_fsync_directory(root)
        return checkpoint

    def save_exact_resume_roundtrip(self, path: object) -> dict:
        """No-step save preserving source ``iter=N`` / exact ``next=N+1``.

        This production API exists for the independent stage verifier.  A
        normal save immediately after strict load would see the public runner
        cursor at ``N+1`` and incorrectly emit ``iter=N+1,next=N+2`` despite
        no optimizer update.  The source iteration captured during preflight
        is restored only for serialization and the live cursor is put back in
        a ``finally`` block.
        """

        if not self._formal_action_ball_runtime_bootstrap_required():
            raise RuntimeError(
                "exact-resume roundtrip save requires formal ActionBall"
            )
        if not bool(
            getattr(self, "_exact_resume_roundtrip_pending", False)
        ):
            raise RuntimeError(
                "exact-resume roundtrip save requires one fresh strict load "
                "with no intervening learn/update"
            )
        # Re-hash every mutable continuation component before creating any
        # output namespace.  A caller cannot reset/step/update after strict
        # load and still spend the no-step token on a checkpoint.
        self.exact_resume_live_state_receipt()
        source_iteration = getattr(
            self, "_exact_resume_loaded_source_iteration", None
        )
        before_iteration = self.current_learning_iteration
        if (
            type(source_iteration) is not int
            or source_iteration < 0
            or type(before_iteration) is not int
            or before_iteration != source_iteration + 1
        ):
            raise RuntimeError(
                "exact-resume roundtrip iteration cursor drifted"
            )
        try:
            target = pathlib.Path(
                os.path.abspath(os.fspath(path))
            )
        except TypeError as exc:
            raise TypeError(
                "exact-resume roundtrip target must be a filesystem path"
            ) from exc
        parent = target.parent
        parent_info = parent.lstat()
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or parent.is_symlink()
        ):
            raise RuntimeError(
                "exact-resume roundtrip parent must be a real directory"
            )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".action_ball_exact_resume_roundtrip.",
            suffix=".pt.tmp",
            dir=str(parent),
        )
        os.close(descriptor)
        temporary = pathlib.Path(temporary_name)
        installed = False
        try:
            self.current_learning_iteration = source_iteration
            try:
                infos = self._checkpoint_infos()
                source_telemetry = getattr(
                    self,
                    "_exact_resume_loaded_source_telemetry",
                    None,
                )
                if (
                    type(source_telemetry) is not dict
                    or set(source_telemetry)
                    != {
                        "log_dir",
                        "wandb_run_id",
                        "wandb_run_name",
                    }
                ):
                    raise RuntimeError(
                        "exact-resume roundtrip lost source telemetry"
                    )
                live_state = infos.get("hope_exact_resume_state")
                if not isinstance(live_state, dict):
                    raise RuntimeError(
                        "exact-resume roundtrip did not capture live state"
                    )
                # These values are logger/location telemetry rather than
                # simulator or optimizer continuation state.  The independent
                # verifier owns no W&B run, so rebuilding them would turn a
                # zero-step save into a false drift.  Preserve only these
                # explicit source fields; RNG, environment/curriculum, policy,
                # optimizer, normalizers and bootstrap identity all remain
                # freshly captured from the restored live runtime.
                for telemetry_key in (
                    "log_dir",
                    "wandb_run_id",
                    "wandb_run_name",
                ):
                    live_state[telemetry_key] = source_telemetry[
                        telemetry_key
                    ]
                OnPolicyRunner.save(self, str(temporary), infos)
            finally:
                self.current_learning_iteration = before_iteration
            info = temporary.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size <= 0
            ):
                raise RuntimeError(
                    "exact-resume roundtrip checkpoint is not a nonempty "
                    "single-link regular file"
                )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            checkpoint_fd = os.open(str(temporary), flags)
            try:
                opened = os.fstat(checkpoint_fd)
                if (
                    opened.st_dev != info.st_dev
                    or opened.st_ino != info.st_ino
                    or opened.st_size != info.st_size
                ):
                    raise RuntimeError(
                        "exact-resume roundtrip checkpoint changed while "
                        "opening"
                    )
                os.fsync(checkpoint_fd)
            finally:
                os.close(checkpoint_fd)
            try:
                os.link(str(temporary), str(target))
            except FileExistsError as exc:
                raise RuntimeError(
                    "exact-resume roundtrip target namespace is already "
                    f"spent: {target}"
                ) from exc
            os.unlink(str(temporary))
            installed = True
            self._joint_safety_fsync_directory(parent)
        finally:
            self.current_learning_iteration = before_iteration
            if not installed:
                try:
                    os.unlink(str(temporary))
                except FileNotFoundError:
                    pass

        final_info = target.lstat()
        if (
            not stat.S_ISREG(final_info.st_mode)
            or final_info.st_nlink != 1
            or final_info.st_size <= 0
        ):
            raise RuntimeError(
                "installed exact-resume roundtrip checkpoint is invalid"
            )
        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_evaluation_inbox as inbox_protocol,
        )

        receipt = inbox_protocol.artifact_receipt(target)
        self._exact_resume_roundtrip_pending = False
        return {
            "checkpoint": receipt,
            "source_embedded_iteration": source_iteration,
            "before_current_learning_iteration": before_iteration,
            "after_current_learning_iteration": (
                self.current_learning_iteration
            ),
            "output_embedded_iteration": source_iteration,
            "output_next_learning_iteration": source_iteration + 1,
            _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY: (
                self.runtime_bootstrap_receipt_sha256
            ),
            _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY: (
                self.runtime_bootstrap_lineage_payload_sha256
            ),
            _RUNTIME_BOOTSTRAP_RECEIPT_KEY: (
                self._runtime_bootstrap_json_clone(
                    self.runtime_bootstrap_receipt
                )
            ),
        }

    def _service_action_ball_frozen_evaluation(self, step: int) -> bool:
        """Fence one frozen request through evidence, publication and ACK.

        Once a request is published this method intentionally does not return
        to RSL-RL's rollout loop until that exact request is consumed and its
        post-consumption exact checkpoint is ACKed.  The independent sidecar
        remains asynchronous; the trainer is synchronously fenced so neither
        policy weights, normalizers nor the live domain can outrun the frozen
        checkpoint being judged.
        """

        term = self._action_ball_frozen_eval_term()
        if term is None:
            return False
        if type(step) is not int or step < 0:
            raise RuntimeError(
                "action-ball frozen-evaluation boundary step is invalid"
            )
        runner_bindings = None
        published_here = False
        did_global_reset = False
        while True:
            boundary = term.action_ball_frozen_evaluation_boundary(
                step=step,
                phase="poll",
                runner_bindings=runner_bindings,
            )
            if not isinstance(boundary, dict):
                raise RuntimeError(
                    "action-ball frozen-evaluation boundary returned no "
                    "state"
                )
            request_seq = boundary.get("request_seq")
            if type(request_seq) is not int or request_seq < 0:
                raise RuntimeError(
                    "action-ball frozen-evaluation request sequence is "
                    "invalid"
                )
            if boundary.get("diagnostic_unauthorized") is True:
                return did_global_reset
            if boundary.get("requires_runner_binding") is True:
                if runner_bindings is not None:
                    raise RuntimeError(
                        "frozen-eval runtime rejected the already-bound "
                        "policy fence"
                    )
                runner_bindings = self._frozen_eval_runner_bindings(
                    policy_generation=step
                )
                continue

            if boundary.get("needs_global_reset") is True:
                reset_result = (
                    term.action_ball_frozen_evaluation_boundary(
                        step=step,
                        phase="commit_global_reset",
                    )
                )
                if (
                    not isinstance(reset_result, dict)
                    or reset_result.get("global_release_committed")
                    is not True
                ):
                    raise RuntimeError(
                        "action-ball global domain release did not commit"
                    )
                # The release callback ran under the term-owned no-new-work
                # fence. Recreate every environment only after publication so
                # Motion's next birth reads the new domain epoch.
                self.env.reset()
                did_global_reset = True
                after_reset = (
                    term.action_ball_frozen_evaluation_boundary(
                        step=step,
                        phase="after_global_reset",
                    )
                )
                if (
                    not isinstance(after_reset, dict)
                    or after_reset.get("needs_global_reset") is not False
                ):
                    raise RuntimeError(
                        "action-ball global reset did not close its fence"
                    )
                continue

            if boundary.get("needs_ack_checkpoint") is True:
                checkpoint = self._action_ball_control_checkpoint(
                    step=step,
                    purpose="evidence_consumed",
                    request_seq=request_seq,
                )
                state_sha = (
                    term.action_ball_frozen_evaluation_boundary(
                        step=step,
                        phase="consumer_state_sha256",
                        checkpoint_path=str(checkpoint),
                    )
                )
                if (
                    not isinstance(state_sha, dict)
                    or type(
                        state_sha.get("consumer_state_sha256")
                    )
                    is not str
                ):
                    raise RuntimeError(
                        "action-ball consumer state digest was not produced"
                    )
                ack = term.action_ball_frozen_evaluation_boundary(
                    step=step,
                    phase="publish_ack",
                    checkpoint_path=str(checkpoint),
                    consumer_state_sha256=state_sha[
                        "consumer_state_sha256"
                    ],
                )
                if (
                    not isinstance(ack, dict)
                    or ack.get("ack_published") is not True
                ):
                    raise RuntimeError(
                        "action-ball frozen-evaluation ACK was not "
                        "published"
                    )
                continue

            stage = boundary.get("stage")
            if boundary.get("request_due") is True:
                if published_here or runner_bindings is not None:
                    raise RuntimeError(
                        "action-ball frozen-evaluation attempted to publish "
                        "a second request inside one fence"
                    )
                runner_bindings = self._frozen_eval_runner_bindings(
                    policy_generation=step
                )
                checkpoint = self._action_ball_control_checkpoint(
                    step=step,
                    purpose="policy_snapshot",
                    request_seq=request_seq,
                )
                published = (
                    term.action_ball_frozen_evaluation_boundary(
                        step=step,
                        phase="publish_request",
                        checkpoint_path=str(checkpoint),
                        runner_bindings=runner_bindings,
                    )
                )
                if (
                    not isinstance(published, dict)
                    or published.get("published") is not True
                    or published.get("request_seq") != request_seq
                ):
                    raise RuntimeError(
                        "action-ball frozen-evaluation request was not "
                        "published exactly once"
                    )
                published_here = True
                # Persist the newly allocated authority/coordinator tape
                # immediately. A crash between request fsync and this save is
                # reconciled from the preceding policy snapshot.
                self._action_ball_control_checkpoint(
                    step=step,
                    purpose="request_persisted",
                    request_seq=request_seq,
                )
                continue
            if stage == "acked":
                return did_global_reset
            if stage in (
                "published",
                "result_ready",
                "curriculum_consumed",
                "ack_prepared",
            ):
                if runner_bindings is None:
                    raise RuntimeError(
                        "in-flight frozen evaluation has no policy fence"
                    )
                # A dead/failed sidecar never permits another rollout or
                # optimizer update. The external launch supervisor owns
                # process liveness and will fail the run; until then this
                # trainer remains safely fenced.
                poll_interval = float(
                    getattr(
                        self,
                        "_action_ball_frozen_eval_poll_interval_s",
                        1.0,
                    )
                )
                if (
                    not math.isfinite(poll_interval)
                    or poll_interval <= 0.0
                    or poll_interval > 60.0
                ):
                    raise RuntimeError(
                        "action-ball frozen-eval poll interval must be in "
                        "(0, 60] seconds"
                    )
                time.sleep(poll_interval)
                continue
            if stage is None and boundary.get("request_due") is not True:
                return did_global_reset
            raise RuntimeError(
                "action-ball frozen-evaluation entered unknown stage "
                f"{stage!r}"
            )

    # ------------------------------------------------------------------
    # 精确续训包(jiayi hitterobs 9f684ae5 按 main 语义移植)
    # ------------------------------------------------------------------

    # HER 已实现回放环(dict[clip_key -> tensor],RacketTargetCommand._ach_*):main 相对
    # hitterobs 的新增课程宿主,按名字点名整环入档。
    _RESUME_TENSOR_DICT_ATTRS = ("_ach_pos", "_ach_vel", "_ach_spd")
    # 回放环的填充度/写指针(dict[clip_key -> int]):同样点名,不落在下面的后缀规则里。
    _RESUME_SCALAR_DICT_ATTRS = ("_ach_fill", "_ach_ptr")

    def _build_exact_resume_state(
        self,
        *,
        runtime_bootstrap_binding: Optional[Mapping[str, object]] = None,
    ) -> dict:
        """打包"精确续训"所需的训练进度状态(jiayi hitterobs 布局的 main 版)。

        人话:除了权重,续训还需要 (1) 下一个该跑的迭代号 —— base rsl_rl 完成第 N 个迭代后
        存档且记 iter=N,直接续会静默重复一次 PPO 更新;(2) 环境课程主时钟 common_step_counter
        和各命令项的课程状态;(3) 下一批采样所需的 RNG 状态。只要 checkpoint 带这份精确
        续训包,load() 就会在第一次 env.reset() 前恢复 RNG;不带包的 legacy checkpoint
        仍保留旧的 warm-start 语义,绝不根据 seed 或迭代号猜 RNG。
        """
        wandb_run_id = None
        wandb_run_name = None
        if self.logger_type == "wandb" and not self.disable_logs:
            try:
                # Importing W&B for the first time after restoring RNG can
                # itself consume Python/NumPy randomness.  Logger setup owns
                # imports; checkpoint capture only reads an already-loaded
                # module and therefore cannot perturb the continuation stream.
                wandb = sys.modules.get("wandb")
                if wandb is not None and wandb.run is not None:
                    wandb_run_id = wandb.run.id
                    wandb_run_name = wandb.run.name
            except Exception:
                pass
        if runtime_bootstrap_binding is None:
            runtime_bootstrap_binding = (
                self._validated_runtime_bootstrap_binding()
            )
        elif set(runtime_bootstrap_binding) not in (
            set(),
            {
                _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY,
                _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY,
                _RUNTIME_BOOTSTRAP_RECEIPT_KEY,
            },
        ):
            raise RuntimeError(
                "runtime bootstrap exact-state binding has unexpected keys"
            )
        environment_resume_state = (
            self._capture_environment_resume_state()
        )
        bootstrap_state = self._runtime_bootstrap_json_clone(
            dict(runtime_bootstrap_binding)
        )
        resume_context = dict(
            getattr(self, "checkpoint_resume_context", {})
        )
        reserved = {
            "schema_version",
            "next_learning_iteration",
            "target_learning_iterations",
            "tot_timesteps",
            "tot_time",
            "algorithm_learning_rate",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "torch_cuda_random_states",
            "torch_cuda_device_count",
            "log_dir",
            "wandb_run_id",
            "wandb_run_name",
            "environment_resume_state",
            *bootstrap_state,
        }
        overlap = sorted(reserved.intersection(resume_context))
        if overlap:
            raise RuntimeError(
                "checkpoint_resume_context attempts to replace exact state "
                f"fields: {overlap}"
            )
        # RNG is intentionally the final live-state capture.  Everything
        # above is required to be observational; no telemetry import or
        # command-state traversal may occur after this point.
        python_random_state = random.getstate()
        numpy_random_state = self._serialize_numpy_rng_state(
            np.random.get_state()
        )
        torch_random_state = torch.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_random_states = torch.cuda.get_rng_state_all()
            torch_cuda_device_count = int(torch.cuda.device_count())
        else:
            torch_cuda_random_states = []
            torch_cuda_device_count = 0
        return {
            "schema_version": _EXACT_RESUME_SCHEMA_VERSION,
            # Base rsl_rl saves after completing iteration N but stores iter=N. Exact resume must
            # begin at N+1, otherwise it silently performs one duplicate PPO update.
            "next_learning_iteration": int(self.current_learning_iteration) + 1,
            "target_learning_iterations": int(self.cfg.get("max_iterations", 0)),
            "tot_timesteps": int(self.tot_timesteps),
            "tot_time": float(self.tot_time),
            "algorithm_learning_rate": float(self.alg.learning_rate),
            "python_random_state": python_random_state,
            "numpy_random_state": numpy_random_state,
            "torch_random_state": torch_random_state,
            "torch_cuda_random_states": torch_cuda_random_states,
            "torch_cuda_device_count": torch_cuda_device_count,
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
            "wandb_run_id": wandb_run_id,
            "wandb_run_name": wandb_run_name,
            "environment_resume_state": environment_resume_state,
            **bootstrap_state,
            **resume_context,
        }

    @staticmethod
    def _is_scalar_tree(value) -> bool:
        """Return whether a value is cheap, device-independent command-manager state."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return True
        if isinstance(value, (list, tuple)):
            return all(MotionOnPolicyRunner._is_scalar_tree(item) for item in value)
        if isinstance(value, dict):
            return all(
                MotionOnPolicyRunner._is_scalar_tree(key)
                and MotionOnPolicyRunner._is_scalar_tree(item)
                for key, item in value.items()
            )
        return False

    def _ordered_action_resume_terms(
        self, env
    ) -> Tuple[object, Tuple[str, ...], Mapping[str, object]]:
        """Resolve the complete ordered ActionManager tuple for schema-4 state."""

        manager = getattr(env, "action_manager", None)
        if manager is None:
            return None, (), {}
        raw_names = tuple(getattr(manager, "active_terms", ()))
        names = tuple(str(name) for name in raw_names)
        if len(names) != len(set(names)):
            raise RuntimeError("action manager active_terms contains duplicate names")
        getter = getattr(manager, "get_term", None)
        if raw_names and not callable(getter):
            raise RuntimeError(
                "action manager active_terms requires get_term() for exact resume"
            )
        terms = {
            name: getter(raw_name)
            for raw_name, name in zip(raw_names, names)
        }
        return manager, names, terms

    @staticmethod
    def _action_delay_required(action_terms: Mapping[str, object]) -> bool:
        required = False
        for name, term in action_terms.items():
            enabled = getattr(term, "control_step_action_delay_enabled", False)
            if type(enabled) is not bool:
                raise RuntimeError(
                    f"action term {name!r} delay enabled flag must be an exact boolean"
                )
            required = required or enabled
        return required

    @staticmethod
    def _action_runtime_state_required(
        action_terms: Mapping[str, object]
    ) -> bool:
        """Whether any action term owns state that changes the next physics write.

        Older action terms expose only the delay flag, so retain that exact fallback.  New terms
        must expose an exact bool and may require schema-4 even with delay disabled (for example,
        the cross-policy max-inward safety containment latch).
        """

        required = False
        missing = object()
        for name, term in action_terms.items():
            delay_enabled = getattr(
                term, "control_step_action_delay_enabled", False
            )
            if type(delay_enabled) is not bool:
                raise RuntimeError(
                    f"action term {name!r} delay enabled flag must be an exact boolean"
                )
            flag = getattr(term, "action_runtime_state_required", missing)
            if flag is missing:
                flag = delay_enabled
            if type(flag) is not bool:
                raise RuntimeError(
                    f"action term {name!r} runtime-state-required flag must be "
                    "an exact boolean"
                )
            if delay_enabled and not flag:
                raise RuntimeError(
                    f"action term {name!r} runtime-state-required flag cannot "
                    "be false while delay is enabled"
                )
            required = required or flag
        return required

    def _emit_control_step_action_delay_runtime_receipt(self) -> Optional[dict]:
        """Emit the first-reset delay distribution bound to the training contract.

        The receipt is intentionally produced only once, after any exact-resume true reset and
        before the first rollout.  It proves that every live env received one episode lag and lets
        launch supervision compare the instantiated runtime with the immutable static contract.
        """

        previous = getattr(self, "_control_step_action_delay_receipt", None)
        if previous is not None:
            return previous
        env = getattr(self.env, "unwrapped", self.env)
        _manager, names, terms = self._ordered_action_resume_terms(env)
        if not self._action_delay_required(terms):
            return None
        if (
            type(self.training_contract_sha256) is not str
            or len(self.training_contract_sha256) != 64
        ):
            raise RuntimeError(
                "enabled control-step action delay requires a bound training contract"
            )
        rows = []
        for name in names:
            term = terms[name]
            if not getattr(term, "control_step_action_delay_enabled", False):
                continue
            getter = getattr(
                term, "control_step_action_delay_runtime_receipt", None
            )
            if not callable(getter):
                raise RuntimeError(
                    f"enabled action term {name!r} exposes no runtime delay receipt"
                )
            receipt = getter()
            if (
                not isinstance(receipt, dict)
                or receipt.get("schema_version") != 1
                or receipt.get("kind")
                != "whole_body_tracking.policy_control_step_action_delay_receipt"
                or receipt.get("num_envs") != int(getattr(env, "num_envs", -1))
                or receipt.get("initialized_env_count") != receipt.get("num_envs")
                or not isinstance(receipt.get("lag_histogram"), dict)
                or sum(receipt["lag_histogram"].values()) != receipt["num_envs"]
                or not isinstance(receipt.get("contract"), dict)
                or receipt["contract"].get("enabled") is not True
                or receipt["contract"].get("semantic_unit")
                != "policy_control_step"
            ):
                raise RuntimeError(
                    f"enabled action term {name!r} runtime delay receipt is incomplete"
                )
            rows.append({"term_name": name, **receipt})
        if not rows:
            raise RuntimeError(
                "enabled control-step action delay produced no action-term receipt"
            )
        record = {
            "event": "hope_control_step_action_delay_runtime",
            "schema_version": 1,
            "training_contract_sha256": self.training_contract_sha256,
            "active_action_term_names": list(names),
            "delay_terms": rows,
        }
        print(
            "HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON="
            + json.dumps(
                record,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        self._control_step_action_delay_receipt = record
        return record

    def _capture_environment_resume_state(self) -> dict:
        """Capture schedule/curriculum state that affects the next rollout distribution."""
        env = getattr(self.env, "unwrapped", self.env)
        state = {
            # 内层 schema:1 = jiayi/hitterobs 布局(scalars/tensors 两段);2 = main 追加
            # tensor_dicts 段;3 = command term 可用成对显式 hook 接管完整状态。旧 schema
            # 仍走属性扫描兼容,新 schema 的显式状态则按 term 名、term 类型和 strict=True
            # fail-loud 恢复,不允许换动作目录后静默套用旧课程。
            # Preserve exact legacy schema-3/four-key bytes when no action term owns runtime state.
            # Schema 4 covers both a nonzero-capable delay queue and cross-policy safety state.
            "schema_version": 3,
            "common_step_counter": int(getattr(env, "common_step_counter", 0)),
            "active_term_names": [],
            "command_terms": {},
        }
        # Re-check the formal task-first/action-ball contract at every save. This catches runtime
        # drift (for example, a manager or term disappearing after construction) instead of
        # emitting a deceptively complete schema-3 checkpoint.
        self._validate_task_first_exact_resume_terms()
        (
            _action_manager,
            action_term_names,
            action_terms,
        ) = self._ordered_action_resume_terms(env)
        if self._action_runtime_state_required(action_terms):
            if not action_term_names:
                raise RuntimeError(
                    "required action runtime state needs a non-empty ActionManager tuple"
                )
            state["schema_version"] = _ENVIRONMENT_RESUME_SCHEMA_VERSION
            state["active_action_term_names"] = list(action_term_names)
            state["action_terms"] = {}
            for term_name in action_term_names:
                term = action_terms[term_name]
                getter = getattr(
                    term, "action_delay_exact_resume_state_dict", None
                )
                loader = getattr(
                    term, "load_action_delay_exact_resume_state_dict", None
                )
                validator = getattr(
                    term, "validate_action_delay_exact_resume_state_dict", None
                )
                present = tuple(
                    callable(value) for value in (getter, loader, validator)
                )
                if any(present) and not all(present):
                    raise RuntimeError(
                        f"action term {term_name!r} must implement action-runtime exact-resume "
                        "getter/validator/loader as one complete interface"
                    )
                term_type = (
                    f"{type(term).__module__}.{type(term).__qualname__}"
                )
                if all(present):
                    exact_state = getter()
                    if not isinstance(exact_state, dict):
                        raise TypeError(
                            f"action term {term_name!r} runtime exact state must be a dict"
                        )
                    # ``explicit_delay`` is the historical schema-4 wire tag.  Its payload may
                    # now also contain no-delay safety-containment state; retaining the tag keeps
                    # existing delay checkpoints loadable without a schema migration.
                    state["action_terms"][term_name] = {
                        "capture_mode": "explicit_delay",
                        "term_type": term_type,
                        "exact_state": exact_state,
                    }
                else:
                    state["action_terms"][term_name] = {
                        "capture_mode": "identity_only",
                        "term_type": term_type,
                    }
            missing = [
                name
                for name, term in action_terms.items()
                if self._action_runtime_state_required({name: term})
                and state["action_terms"][name]["capture_mode"]
                != "explicit_delay"
            ]
            if missing:
                raise RuntimeError(
                    "required action runtime state lacks exact state on "
                    f"active terms {missing}"
                )

        manager = getattr(env, "command_manager", None)
        if manager is None:
            return state

        raw_term_names = tuple(getattr(manager, "active_terms", ()))
        term_names = tuple(str(name) for name in raw_term_names)
        if len(term_names) != len(set(term_names)):
            raise RuntimeError("command manager active_terms contains duplicate names")
        state["active_term_names"] = list(term_names)
        strict_command_state = self._task_first_exact_resume_required()
        strict_mode = self._strict_exact_resume_target_mode()

        for raw_term_name, term_name in zip(raw_term_names, term_names):
            term = manager.get_term(raw_term_name)
            exact_state_getter = getattr(term, "exact_resume_state_dict", None)
            exact_state_loader = getattr(term, "load_exact_resume_state_dict", None)
            has_exact_state_getter = callable(exact_state_getter)
            has_exact_state_loader = callable(exact_state_loader)
            if has_exact_state_getter != has_exact_state_loader:
                raise RuntimeError(
                    f"command term {term_name!r} must implement exact_resume_state_dict() "
                    "and load_exact_resume_state_dict(state, strict=True) as a pair"
                )
            if strict_command_state and not has_exact_state_getter:
                raise RuntimeError(
                    f"{self._strict_exact_resume_label(strict_mode)} exact resume requires "
                    "every active command term to implement "
                    f"explicit hooks; missing on {term_name!r}"
                )
            if has_exact_state_getter:
                exact_state = exact_state_getter()
                if not isinstance(exact_state, dict):
                    raise TypeError(
                        f"command term {term_name!r} exact_resume_state_dict() must return a dict"
                    )
                state["command_terms"][term_name] = {
                    "capture_mode": "explicit",
                    "term_type": f"{type(term).__module__}.{type(term).__qualname__}",
                    "exact_state": exact_state,
                }
                # The explicit API owns the whole term state. Mixing it with the heuristic scanner
                # would create two competing truths and could overwrite a strict restore.
                continue

            term_state = {"scalars": {}, "tensors": {}, "tensor_dicts": {}}

            # 标量类课程驱动:EMA 成功率/摔倒统计(*_acc / *_acc_c)、成功门控扰动幅度
            # (_curr_perturb_scale)、自适应 sigma 及其误差 EMA(_adaptive_sigma_* /
            # *_err_sum / *_err_sum_c,main 侧新增的两个后缀)。这些决定下一个 rollout 的
            # 目标分布,但不属于策略权重。挑选规则宁多勿漏:多存一个标量无害(恢复端只
            # setattr 认得的),漏存一个课程就悄悄回零。
            for attr, value in vars(term).items():
                is_resume_scalar = (
                    attr == "_curr_perturb_scale"
                    or attr.startswith("_adaptive_sigma_")
                    or attr.endswith("_acc")
                    or attr.endswith("_acc_c")
                    or attr.endswith("_err_sum")
                    or attr.endswith("_err_sum_c")
                    or attr in self._RESUME_SCALAR_DICT_ATTRS
                )
                if is_resume_scalar and self._is_scalar_tree(value):
                    term_state["scalars"][attr] = value

            # RallyV14 samples 35% of true resets from a ring of completed follow-through states.
            # Keeping that ring avoids silently dropping the recovery population after a restart.
            # MotionCommand 的自适应失败分箱统计(bin_failed_count)同理。
            for attr in (
                "bin_failed_count",
                "_current_bin_failed",
                "_post_swing_root",
                "_post_swing_joint_pos",
                "_post_swing_joint_vel",
            ):
                value = getattr(term, attr, None)
                if torch.is_tensor(value):
                    term_state["tensors"][attr] = value.detach().cpu().clone()
            for attr in ("_post_swing_ptr", "_post_swing_count"):
                value = getattr(term, attr, None)
                if isinstance(value, int):
                    term_state["scalars"][attr] = value

            # main 侧新增:HER 已实现回放环本体(dict[clip_key -> tensor])。不存它们,
            # 续训后 achieved-target 回放要从零重新攒,等效于把 35%/HER 采样人口清空。
            for attr in self._RESUME_TENSOR_DICT_ATTRS:
                value = getattr(term, attr, None)
                if isinstance(value, dict) and value and all(
                    torch.is_tensor(item) for item in value.values()
                ):
                    term_state["tensor_dicts"][attr] = {
                        key: item.detach().cpu().clone() for key, item in value.items()
                    }

            # Schema 3 binds the complete ordered active-term tuple even when a legacy term has no
            # recognized persistent attributes.  Omitting an empty term would make a later
            # add/remove/reorder invisible to exact resume.
            state["command_terms"][term_name] = term_state
        return state

    def _strict_exact_resume_target_mode(self) -> Optional[str]:
        """Return the formal command mode that requires complete exact state, if any."""

        env = getattr(self.env, "unwrapped", self.env)
        env_cfg = getattr(env, "cfg", None)
        commands = getattr(env_cfg, "commands", None)
        racket = None if commands is None else getattr(commands, "racket_target", None)
        mode = str(getattr(racket, "target_mode", ""))
        if (
            mode == "reference_perturbed"
            and str(getattr(env_cfg, "obs_mode", ""))
            == "stage1_natural_clip_paddle_world"
        ):
            return "stage1_natural_clip"
        return mode if mode in (
            "task_first",
            "action_ball",
            "action_ball_full_mdp",
        ) else None

    def _action_ball_diagnostic_unauthorized(self) -> bool:
        """Return whether this is the fixed-domain, non-promotable N1 screen."""

        if self._strict_exact_resume_target_mode() != "action_ball":
            return False
        env = getattr(self.env, "unwrapped", self.env)
        commands = getattr(getattr(env, "cfg", None), "commands", None)
        racket = (
            None
            if commands is None
            else getattr(commands, "racket_target", None)
        )
        diagnostic = getattr(
            racket, "action_ball_diagnostic_unauthorized", False
        )
        if type(diagnostic) is not bool:
            raise RuntimeError(
                "action_ball_diagnostic_unauthorized must be an exact boolean"
            )
        return diagnostic

    def _diagnostic_joint_safety_compact_evidence(self) -> bool:
        """Bind compact joint-safety evidence to an unauthorized task identity.

        The original fixed-domain ActionBall diagnostic and the ball-free Stage-1
        natural-clip diagnostic share the same non-promotable device aggregate.  Do
        not infer this from a broad task family: both the exact diagnostic brand and
        the live action producer must opt in.
        """

        mode = self._strict_exact_resume_target_mode()
        if mode == "action_ball":
            diagnostic = self._action_ball_diagnostic_unauthorized()
        elif mode == "stage1_natural_clip":
            env = getattr(self.env, "unwrapped", self.env)
            commands = getattr(getattr(env, "cfg", None), "commands", None)
            racket = (
                None
                if commands is None
                else getattr(commands, "racket_target", None)
            )
            diagnostic = getattr(
                racket, "action_ball_diagnostic_unauthorized", False
            )
            if type(diagnostic) is not bool:
                raise RuntimeError(
                    "action_ball_diagnostic_unauthorized must be an exact boolean"
                )
        else:
            return False
        if not diagnostic:
            return False
        term = self._bind_joint_safety_action_term(required=True)
        if (
            getattr(
                term, "_joint_safety_diagnostic_compact_evidence", None
            )
            is not True
        ):
            raise RuntimeError(
                "diagnostic task requires an explicitly compact joint-safety producer"
            )
        return True

    def _full_mdp_diagnostic_joint_safety_compact_evidence(self) -> bool:
        """Bind only the exact fresh full-MDP diagnostic safety drain.

        Fresh full-MDP shares the compact action-ledger representation, but it
        does not own the legacy ActionBall actual-limit attribution or push
        ledgers.  Keep this predicate separate so enabling the typed
        prepare/acknowledge transaction cannot manufacture those auxiliary
        consumers or their zero-valued records.
        """

        if self._strict_exact_resume_target_mode() != "action_ball_full_mdp":
            return False
        run_mode = getattr(self, "_action_ball_full_mdp_run_mode", None)
        if run_mode == _ACTION_BALL_FULL_MDP_FORMAL_MODE:
            # Formal full-MDP must never inherit the non-promotable compact
            # representation from its diagnostic sibling.
            return False
        if run_mode != _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE:
            raise RuntimeError(
                "fresh full-MDP compact joint-safety evidence requires the "
                "exact single_action_lean runner binding"
            )
        self._action_ball_full_mdp_require_healthy()
        env = getattr(self.env, "unwrapped", self.env)
        if env is not getattr(
            self, "_action_ball_full_mdp_runtime_env_identity", None
        ):
            raise RuntimeError(
                "fresh full-MDP compact joint-safety environment identity changed"
            )
        commands = getattr(getattr(env, "cfg", None), "commands", None)
        racket = (
            None
            if commands is None
            else getattr(commands, "racket_target", None)
        )
        diagnostic = getattr(
            racket, "action_ball_diagnostic_unauthorized", False
        )
        if type(diagnostic) is not bool:
            raise RuntimeError(
                "fresh full-MDP action_ball_diagnostic_unauthorized must be "
                "an exact boolean"
            )
        if diagnostic is not True:
            raise RuntimeError(
                "single_action_lean requires unauthorized fresh full-MDP "
                "joint-safety evidence"
            )
        term = self._bind_joint_safety_action_term(required=True)
        if (
            getattr(
                term, "_joint_safety_diagnostic_compact_evidence", None
            )
            is not True
        ):
            raise RuntimeError(
                "single_action_lean requires an explicitly compact "
                "fresh full-MDP joint-safety producer"
            )
        return True

    def _diagnostic_joint_safety_compact_update_evidence(self) -> bool:
        """Select compact two-phase update evidence without widening auxiliaries."""

        if self._strict_exact_resume_target_mode() == "action_ball_full_mdp":
            return self._full_mdp_diagnostic_joint_safety_compact_evidence()
        return self._diagnostic_joint_safety_compact_evidence()

    def _effective_reward_activation_task_kind(self) -> Optional[str]:
        """Return the two task leaves with a verified RewardManager ledger adapter."""

        if self._strict_exact_resume_target_mode() == "action_ball":
            if self._action_ball_diagnostic_unauthorized():
                # Diagnostic reward screens deliberately cannot mint formal
                # evidence or promotion authority.  Keep the real Reward,
                # clamp, limit/table/fall penalties and terminations in the
                # environment, but do not fence PPO on the formal activation
                # ledger's proof transaction.
                return None
            return "action_ball"
        env = getattr(self.env, "unwrapped", self.env)
        cfg = getattr(env, "cfg", None)
        cfg_mro_names = {
            cls.__name__ for cls in getattr(type(cfg), "__mro__", ())
        }
        if "HOPEPingPongUpperSafeAgibotA3EnvCfg" in cfg_mro_names:
            return "upper_safe"
        return None

    def _bind_action_ball_reward_evidence(self) -> dict:
        """Bind only public, immutable ActionBall/termination evidence APIs."""

        env = getattr(self.env, "unwrapped", self.env)
        command_manager = getattr(env, "command_manager", None)
        termination_manager = getattr(env, "termination_manager", None)
        if command_manager is None or termination_manager is None:
            raise RuntimeError(
                "action-ball Reward evidence requires command and termination managers"
            )
        raw_command_names = tuple(
            getattr(command_manager, "active_terms", ())
        )
        if not raw_command_names:
            raise RuntimeError(
                "action-ball Reward evidence has no active command terms"
            )
        motion_candidates = []
        racket_candidates = []
        for raw_name in raw_command_names:
            term = command_manager.get_term(raw_name)
            if (
                callable(
                    getattr(term, "action_ball_action_uid_for_envs", None)
                )
                and callable(
                    getattr(term, "action_ball_birth_receipt_sha256", None)
                )
                and hasattr(term, "action_ball_reset_generation")
                and hasattr(term, "action_ball_swing_generation")
                and hasattr(term, "action_ball_ordered_action_uids")
            ):
                motion_candidates.append((str(raw_name), term))
            if callable(getattr(term, "action_ball_hard_contract", None)):
                contract = term.action_ball_hard_contract()
                if contract is not None:
                    racket_candidates.append((str(raw_name), term, contract))
        if len(motion_candidates) != 1 or len(racket_candidates) != 1:
            raise RuntimeError(
                "action-ball Reward evidence requires exactly one public "
                "motion identity provider and one Racket hard-contract provider"
            )
        motion_name, motion = motion_candidates[0]
        racket_name, _racket, action_contract = racket_candidates[0]
        if not isinstance(action_contract, dict):
            raise RuntimeError(
                "action-ball Racket hard contract must be a mapping"
            )
        ordered_uids = tuple(motion.action_ball_ordered_action_uids)
        if tuple(action_contract.get("action_uids", ())) != ordered_uids:
            raise RuntimeError(
                "action-ball Motion/Racket action UID orders disagree"
            )
        num_envs = getattr(env, "num_envs", None)
        device = getattr(env, "device", None)
        if type(num_envs) is not int or num_envs <= 0 or device is None:
            raise RuntimeError(
                "action-ball Reward evidence requires num_envs and device"
            )
        all_env_ids = torch.arange(
            num_envs, dtype=torch.long, device=device
        )

        def identity_provider():
            return {
                "action_uid": motion.action_ball_action_uid_for_envs(
                    all_env_ids
                ),
                "reset_generation": motion.action_ball_reset_generation,
                "swing_generation": motion.action_ball_swing_generation,
                "birth_receipt_sha256": tuple(
                    motion.action_ball_birth_receipt_sha256(env_id)
                    for env_id in range(num_envs)
                ),
            }

        raw_termination_names = tuple(
            getattr(termination_manager, "active_terms", ())
        )
        termination_names = tuple(str(name) for name in raw_termination_names)
        required = {
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
            "joint_qdes_forbidden",
            "robot_hit_table",
        }
        if (
            not termination_names
            or len(termination_names) != len(set(termination_names))
            or not required.issubset(set(termination_names))
            or not callable(getattr(termination_manager, "get_term", None))
        ):
            raise RuntimeError(
                "action-ball Reward evidence termination set is incomplete"
            )

        def termination_provider():
            current_names = tuple(
                str(name)
                for name in getattr(
                    termination_manager, "active_terms", ()
                )
            )
            if current_names != termination_names:
                raise RuntimeError(
                    "action-ball termination term order changed after binding"
                )
            return {
                "term_order": termination_names,
                "terminated": termination_manager.terminated,
                "time_outs": termination_manager.time_outs,
                "reason_masks": {
                    str(raw_name): termination_manager.get_term(raw_name)
                    for raw_name in raw_termination_names
                },
            }

        return {
            "action_contract": action_contract,
            "identity_provider": identity_provider,
            "termination_provider": termination_provider,
            "motion_term": motion_name,
            "racket_term": racket_name,
        }

    @staticmethod
    def _strict_exact_resume_label(mode: Optional[str]) -> str:
        if mode == "action_ball":
            return "action-ball"
        if mode == "stage1_natural_clip":
            return "stage1-natural-clip"
        if mode == "action_ball_full_mdp":
            return "action-ball-full-mdp"
        return "task-first"

    def _task_first_exact_resume_required(self) -> bool:
        """Compatibility name for both formal task-first and action-ball paths."""

        return self._strict_exact_resume_target_mode() is not None

    def _validate_task_first_exact_resume_terms(self) -> None:
        """Fail unless every formal command term has paired explicit state hooks."""

        mode = self._strict_exact_resume_target_mode()
        if mode is None:
            return
        label = self._strict_exact_resume_label(mode)
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "command_manager", None)
        if manager is None:
            raise RuntimeError(f"{label} exact resume requires a command manager")
        raw_names = tuple(getattr(manager, "active_terms", ()))
        names = tuple(str(name) for name in raw_names)
        if not names:
            raise RuntimeError(f"{label} exact resume requires active command terms")
        if len(names) != len(set(names)):
            raise RuntimeError("command manager active_terms contains duplicate names")
        missing = []
        for raw_name, name in zip(raw_names, names):
            term = manager.get_term(raw_name)
            getter = getattr(term, "exact_resume_state_dict", None)
            loader = getattr(term, "load_exact_resume_state_dict", None)
            if callable(getter) != callable(loader):
                raise RuntimeError(
                    f"command term {name!r} must implement exact resume hooks as a pair"
                )
            if not callable(getter):
                missing.append(name)
        if missing:
            raise RuntimeError(
                f"{label} exact resume requires explicit hooks on every active command term; "
                f"missing={missing}"
            )

    def _restore_environment_resume_state(
        self, resume_state: dict
    ) -> Tuple[int, str]:
        """Restore saved environment progress, with an iteration-derived fallback for old PTs."""
        env = getattr(self.env, "unwrapped", self.env)
        saved = resume_state.get("environment_resume_state")
        if isinstance(saved, dict) and "common_step_counter" in saved:
            saved_common_step_counter = saved["common_step_counter"]
            common_step_counter = int(saved_common_step_counter)
            source = "checkpoint"
        else:
            saved_common_step_counter = None
            # 老 checkpoint(状态未入档)仍有精确的"下一个迭代号":每完成一个 PPO 迭代,
            # 每个 env 都走 num_steps_per_env 个控制步,所以课程主时钟可以由迭代号精确推算
            # 出来,而不是回零(jiayi 对既有 V14 长跑档的回退推算,原样移植)。
            common_step_counter = int(resume_state["next_learning_iteration"]) * int(
                self.num_steps_per_env
            )
            source = "derived-from-iteration"

        manager = getattr(env, "command_manager", None)
        (
            _action_manager,
            current_action_term_names,
            current_action_terms,
        ) = self._ordered_action_resume_terms(env)
        action_runtime_state_required = self._action_runtime_state_required(
            current_action_terms
        )
        if (
            self._strict_exact_resume_target_mode() == "action_ball"
            and action_runtime_state_required
            and not current_action_term_names
        ):
            raise RuntimeError(
                "ActionBall exact resume requires a non-empty ActionManager tuple"
            )
        # As at capture time, enforce the current formal runtime before considering any saved
        # payload. Schema 1/2 remain readable, but they cannot be loaded into task-first/action-ball
        # environments whose current active tuple lacks explicit hooks.
        self._validate_task_first_exact_resume_terms()
        restored_terms = []
        saved_term_states = {}
        active_terms = {}
        staged_action_restores = []
        restored_action_terms = []
        environment_schema = 1
        if isinstance(saved, dict):
            environment_schema = int(saved.get("schema_version", 1))
            if environment_schema not in _SUPPORTED_ENVIRONMENT_RESUME_SCHEMAS:
                raise RuntimeError(
                    "unsupported environment exact-resume schema "
                    f"{environment_schema}; refusing to guess command state"
                )
            # Legacy schema 1/2 deliberately retain their historical best-effort behavior when no
            # command manager exists. Schema 3 is exact and must validate even in that case: a
            # missing manager is an empty current tuple, never a reason to skip identity checks.
            if manager is not None or environment_schema >= 3:
                saved_term_states = saved.get("command_terms", {})
                if not isinstance(saved_term_states, dict):
                    raise TypeError("environment exact-resume command_terms must be a dict")

                raw_active_term_names = (
                    tuple(getattr(manager, "active_terms", ()))
                    if manager is not None
                    else ()
                )
                active_term_names = tuple(str(name) for name in raw_active_term_names)
                if len(active_term_names) != len(set(active_term_names)):
                    raise RuntimeError("command manager active_terms contains duplicate names")
                active_terms = {
                    name: manager.get_term(raw_name)
                    for raw_name, name in zip(
                        raw_active_term_names, active_term_names
                    )
                }

            if environment_schema >= 3:
                if (
                    type(saved_common_step_counter) is not int
                    or saved_common_step_counter < 0
                ):
                    raise RuntimeError(
                        "schema-3 environment common_step_counter must be a "
                        "nonnegative plain integer"
                    )
                expected_keys = {
                    "schema_version",
                    "common_step_counter",
                    "active_term_names",
                    "command_terms",
                }
                if environment_schema >= 4:
                    expected_keys.update(
                        {"active_action_term_names", "action_terms"}
                    )
                if set(saved) != expected_keys:
                    raise RuntimeError(
                        f"schema-{environment_schema} environment exact-resume "
                        "keys do not match that schema"
                    )
                saved_active_names = saved.get("active_term_names")
                if (
                    type(saved_active_names) is not list
                    or any(type(name) is not str for name in saved_active_names)
                    or len(saved_active_names) != len(set(saved_active_names))
                ):
                    raise RuntimeError(
                        "schema-3 environment active_term_names must be a unique string list"
                    )
                saved_active_names = tuple(saved_active_names)
                if saved_active_names != active_term_names:
                    raise RuntimeError(
                        "exact-resume ordered command term identity mismatch: "
                        f"checkpoint={saved_active_names}, current={active_term_names}"
                    )
                if tuple(saved_term_states) != active_term_names:
                    raise RuntimeError(
                        "schema-3 command_terms must contain every active term in exact order: "
                        f"checkpoint={tuple(saved_term_states)}, current={active_term_names}"
                    )
                saved_explicit = {
                    str(name)
                    for name, term_state in saved_term_states.items()
                    if isinstance(term_state, dict)
                    and term_state.get("capture_mode") == "explicit"
                }
                current_explicit = set()
                for term_name, term in active_terms.items():
                    getter = getattr(term, "exact_resume_state_dict", None)
                    loader = getattr(term, "load_exact_resume_state_dict", None)
                    has_getter = callable(getter)
                    has_loader = callable(loader)
                    if has_getter != has_loader:
                        raise RuntimeError(
                            f"command term {term_name!r} must implement exact resume hooks as a pair"
                        )
                    if has_getter:
                        current_explicit.add(term_name)
                if saved_explicit != current_explicit:
                    raise RuntimeError(
                        "exact-resume command term identity mismatch: "
                        f"checkpoint explicit terms={sorted(saved_explicit)}, "
                        f"current explicit terms={sorted(current_explicit)}"
                    )
                if self._task_first_exact_resume_required() and current_explicit != set(
                    active_term_names
                ):
                    mode = self._strict_exact_resume_target_mode()
                    raise RuntimeError(
                        f"{self._strict_exact_resume_label(mode)} schema-3 restore requires "
                        "explicit state for every active "
                        f"command term; explicit={sorted(current_explicit)}, "
                        f"active={list(active_term_names)}"
                    )
                if self._strict_exact_resume_target_mode() == "action_ball":
                    required_ordered_terms = ("racket_target", "motion")
                    missing_dependency_terms = [
                        name
                        for name in required_ordered_terms
                        if name not in active_terms
                    ]
                    if missing_dependency_terms:
                        raise RuntimeError(
                            "action-ball exact resume requires the Racket -> "
                            "Motion dependency pair; missing="
                            f"{missing_dependency_terms}"
                        )
                    motion_finalize = getattr(
                        active_terms["motion"],
                        "finalize_action_ball_exact_resume",
                        None,
                    )
                    if not callable(motion_finalize):
                        raise RuntimeError(
                            "action-ball exact resume requires Motion."
                            "finalize_action_ball_exact_resume()"
                        )

            if action_runtime_state_required and environment_schema < 4:
                raise RuntimeError(
                    "required action runtime state is fresh-only from "
                    "environment resume schema 1/2/3; resume requires schema 4 "
                    "with exact action-term state"
                )

            if environment_schema >= 4:
                saved_action_names = saved.get("active_action_term_names")
                saved_action_states = saved.get("action_terms")
                if (
                    type(saved_action_names) is not list
                    or any(type(name) is not str for name in saved_action_names)
                    or len(saved_action_names) != len(set(saved_action_names))
                    or not isinstance(saved_action_states, dict)
                ):
                    raise RuntimeError(
                        "schema-4 action term names/state must be a unique list and dict"
                    )
                saved_action_names = tuple(saved_action_names)
                if saved_action_names != current_action_term_names:
                    raise RuntimeError(
                        "exact-resume ordered action term identity mismatch: "
                        f"checkpoint={saved_action_names}, "
                        f"current={current_action_term_names}"
                    )
                if tuple(saved_action_states) != current_action_term_names:
                    raise RuntimeError(
                        "schema-4 action_terms must contain every active term "
                        "in exact order"
                    )
                for term_name in current_action_term_names:
                    term = current_action_terms[term_name]
                    term_state = saved_action_states[term_name]
                    if not isinstance(term_state, dict):
                        raise TypeError(
                            f"action term {term_name!r} resume state must be a dict"
                        )
                    term_type = (
                        f"{type(term).__module__}.{type(term).__qualname__}"
                    )
                    if term_state.get("term_type") != term_type:
                        raise RuntimeError(
                            f"action term {term_name!r} type changed across exact resume"
                        )
                    getter = getattr(
                        term, "action_delay_exact_resume_state_dict", None
                    )
                    validator = getattr(
                        term,
                        "validate_action_delay_exact_resume_state_dict",
                        None,
                    )
                    loader = getattr(
                        term,
                        "load_action_delay_exact_resume_state_dict",
                        None,
                    )
                    has_interface = tuple(
                        callable(value) for value in (getter, validator, loader)
                    )
                    if any(has_interface) and not all(has_interface):
                        raise RuntimeError(
                            f"action term {term_name!r} has a partial "
                            "action-runtime resume interface"
                        )
                    mode = term_state.get("capture_mode")
                    if mode == "identity_only":
                        if set(term_state) != {"capture_mode", "term_type"}:
                            raise RuntimeError(
                                f"identity-only action term {term_name!r} has extra state"
                            )
                        if all(has_interface):
                            raise RuntimeError(
                                f"action term {term_name!r} lost explicit delay state"
                            )
                        continue
                    if mode != "explicit_delay" or set(term_state) != {
                        "capture_mode",
                        "term_type",
                        "exact_state",
                    }:
                        raise RuntimeError(
                            f"action term {term_name!r} has an invalid schema-4 record"
                        )
                    if not all(has_interface):
                        raise RuntimeError(
                            f"action term {term_name!r} cannot restore explicit runtime state"
                        )
                    exact_state = term_state["exact_state"]
                    if not isinstance(exact_state, dict):
                        raise TypeError(
                            f"action term {term_name!r} exact runtime state must be a dict"
                        )
                    # Phase one is read-only: malformed state cannot leave a live queue half
                    # restored.  Every action term stages successfully before the environment
                    # clock or any command term is mutated.
                    validator(exact_state, strict=True)
                    staged_action_restores.append(
                        (term_name, loader, exact_state)
                    )
                staged_action_names = {
                    name for name, _loader, _state in staged_action_restores
                }
                missing_enabled_action_state = [
                    name
                    for name, term in current_action_terms.items()
                    if self._action_runtime_state_required({name: term})
                    and name not in staged_action_names
                ]
                if (
                    action_runtime_state_required
                    and missing_enabled_action_state
                ):
                    raise RuntimeError(
                        "required action runtime state lacks "
                        "schema-4 state on terms "
                        f"{missing_enabled_action_state}"
                    )
        elif action_runtime_state_required:
            raise RuntimeError(
                "required action runtime state cannot resume without "
                "schema-4 environment state; launch fresh"
            )

        # Do not mutate even the curriculum clock until schema-3 structure and active-term identity
        # have passed. Explicit term loaders below remain responsible for their own atomicity.
        env.common_step_counter = common_step_counter

        if isinstance(saved, dict) and manager is not None:
            restore_rows = list(saved_term_states.items())
            action_ball_ordered_restore = (
                environment_schema >= 3
                and self._strict_exact_resume_target_mode()
                == "action_ball"
            )
            if action_ball_ordered_restore:
                by_name = {
                    str(term_name): (term_name, term_state)
                    for term_name, term_state in restore_rows
                }
                restore_rows = [
                    by_name["racket_target"],
                    by_name["motion"],
                    *[
                        row
                        for name, row in by_name.items()
                        if name not in ("racket_target", "motion")
                    ],
                ]
            for term_name, term_state in restore_rows:
                term_name = str(term_name)
                if not isinstance(term_state, dict):
                    raise TypeError(
                        f"command term {term_name!r} exact-resume state must be a dict"
                    )
                if term_state.get("capture_mode") == "explicit":
                    if term_name not in active_terms:
                        raise RuntimeError(
                            f"exact-resume command term {term_name!r} is absent from current config"
                        )
                    term = active_terms[term_name]
                    current_term_type = f"{type(term).__module__}.{type(term).__qualname__}"
                    saved_term_type = term_state.get("term_type")
                    if saved_term_type != current_term_type:
                        raise RuntimeError(
                            f"exact-resume command term {term_name!r} type mismatch: "
                            f"checkpoint={saved_term_type!r}, current={current_term_type!r}"
                        )
                    loader = getattr(term, "load_exact_resume_state_dict", None)
                    if not callable(loader):
                        raise RuntimeError(
                            f"command term {term_name!r} cannot load explicit exact-resume state"
                        )
                    exact_state = term_state.get("exact_state")
                    if not isinstance(exact_state, dict):
                        raise TypeError(
                            f"command term {term_name!r} explicit exact-resume state must be a dict"
                        )
                    # strict=True is an interface contract, not a best-effort hint: manifest SHA,
                    # action order, tensor shapes and any other term identity checks must fail loud.
                    loader(exact_state, strict=True)
                    restored_terms.append(term_name)
                    if action_ball_ordered_restore and term_name == "motion":
                        # Racket is the sole owner of the shared
                        # evaluator/curriculum/provider/domain/broker/pool graph.
                        # Motion first loads only its local state and staged
                        # shared digest; this finalizer then cross-checks the
                        # live Racket-owned graph. Fresh training never enters
                        # this checkpoint-only restore branch.
                        active_terms[
                            "motion"
                        ].finalize_action_ball_exact_resume()
                    continue
                try:
                    term = manager.get_term(term_name)
                except (KeyError, ValueError):
                    # 档里多存的命令项(臂间配置差异):容忍跳过,其余项照常恢复。
                    continue
                restored = False
                for attr, value in term_state.get("scalars", {}).items():
                    if hasattr(term, attr):
                        # 整体 setattr(dict 也整个换):若 clip 集合在续训时变了,后续代码
                        # 按新 clip 取键会 KeyError —— fail-loud,好过静默混用两套课程统计。
                        setattr(term, attr, value)
                        restored = True
                for attr, value in term_state.get("tensors", {}).items():
                    if hasattr(term, attr) and torch.is_tensor(value):
                        current = getattr(term, attr)
                        device = current.device if torch.is_tensor(current) else term.device
                        setattr(term, attr, value.to(device=device))
                        restored = True
                for attr, saved_entries in term_state.get("tensor_dicts", {}).items():
                    current = getattr(term, attr, None)
                    if not isinstance(current, dict) or not isinstance(saved_entries, dict):
                        continue
                    for key, value in saved_entries.items():
                        # 只恢复当前配置也有的 clip 键:档里多出来的键容忍丢弃,新增 clip
                        # 保持新鲜初始化(HER 回放环对新 clip 从零攒起是正确语义)。
                        if key in current and torch.is_tensor(current[key]) and torch.is_tensor(value):
                            current[key] = value.to(device=current[key].device)
                            restored = True
                if restored:
                    restored_terms.append(str(term_name))
        # Phase two: all schema, identity, tensor and range checks above succeeded before any
        # action queue mutation.  Commit the prevalidated action-term payloads in exact manager
        # order.  The delay loader performs no sampling or simulator I/O.
        for term_name, loader, exact_state in staged_action_restores:
            loader(exact_state, strict=True)
            restored_action_terms.append(term_name)
        print(
            "[MotionOnPolicyRunner] exact environment progress restored: "
            f"common_step_counter={common_step_counter} ({source}), "
            f"command_terms={restored_terms}, "
            f"action_terms={restored_action_terms}",
            flush=True,
        )
        return common_step_counter, source

    @staticmethod
    def _validate_rng_tensor(saved, current, *, name: str) -> torch.Tensor:
        """Validate a serialized PyTorch RNG byte vector without mutating global RNG."""

        if not torch.is_tensor(saved):
            raise TypeError(f"{name} must be a torch.Tensor")
        if saved.dtype != current.dtype or tuple(saved.shape) != tuple(current.shape):
            raise RuntimeError(
                f"{name} shape/dtype mismatch: checkpoint={tuple(saved.shape)}/{saved.dtype}, "
                f"current={tuple(current.shape)}/{current.dtype}"
            )
        return saved.detach().cpu()

    def _validated_exact_rng_state(self, resume_state: dict):
        """Validate all sampling RNG payloads without mutating any global generator."""

        required = (
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "torch_cuda_random_states",
        )
        missing = [name for name in required if name not in resume_state]
        if missing:
            raise RuntimeError(
                "exact-resume checkpoint is missing RNG state fields: " + ", ".join(missing)
            )

        python_state = resume_state["python_random_state"]
        numpy_state = self._deserialize_numpy_rng_state(
            resume_state["numpy_random_state"]
        )
        # Validate Python and NumPy payloads on private generators so a malformed CUDA payload
        # cannot leave the process half-restored.
        python_probe = random.Random()
        try:
            python_probe.setstate(python_state)
        except Exception as exc:
            raise RuntimeError("invalid python_random_state in exact-resume checkpoint") from exc
        numpy_probe = np.random.RandomState()
        try:
            numpy_probe.set_state(numpy_state)
        except Exception as exc:
            raise RuntimeError("invalid numpy_random_state in exact-resume checkpoint") from exc

        torch_state = self._validate_rng_tensor(
            resume_state["torch_random_state"],
            torch.get_rng_state(),
            name="torch_random_state",
        )

        saved_cuda_states = resume_state["torch_cuda_random_states"]
        if not isinstance(saved_cuda_states, (list, tuple)):
            raise TypeError("torch_cuda_random_states must be a list or tuple")
        cuda_available = bool(torch.cuda.is_available())
        current_cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and current_cuda_count <= 0:
            raise RuntimeError("torch.cuda reports available but has no visible devices")
        declared_cuda_count = int(
            resume_state.get("torch_cuda_device_count", len(saved_cuda_states))
        )
        if declared_cuda_count != len(saved_cuda_states):
            raise RuntimeError(
                "checkpoint CUDA RNG count is internally inconsistent: "
                f"declared={declared_cuda_count}, states={len(saved_cuda_states)}"
            )
        if declared_cuda_count != current_cuda_count:
            raise RuntimeError(
                "CUDA RNG device-count mismatch: "
                f"checkpoint={declared_cuda_count}, current={current_cuda_count}"
            )
        current_cuda_states = torch.cuda.get_rng_state_all() if current_cuda_count else []
        if len(current_cuda_states) != current_cuda_count:
            raise RuntimeError(
                "torch.cuda.get_rng_state_all() returned an unexpected number of states: "
                f"{len(current_cuda_states)} vs {current_cuda_count} devices"
            )
        cuda_states = [
            self._validate_rng_tensor(saved, current, name=f"torch_cuda_random_states[{idx}]")
            for idx, (saved, current) in enumerate(
                zip(saved_cuda_states, current_cuda_states)
            )
        ]
        return python_state, numpy_state, torch_state, cuda_states

    def _restore_exact_rng_state(self, resume_state: dict) -> None:
        """Restore every sampling RNG atomically after validation and before ``env.reset()``."""

        python_state, numpy_state, torch_state, cuda_states = (
            self._validated_exact_rng_state(resume_state)
        )

        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _checkpoint_exact_resume_state(loaded):
        """Return the canonical exact-resume payload from a loaded checkpoint, if present."""

        if not isinstance(loaded, dict):
            return None
        checkpoint_infos = loaded.get("infos")
        state = (
            checkpoint_infos.get("hope_exact_resume_state")
            if isinstance(checkpoint_infos, dict)
            else None
        )
        if state is None:
            # Cross-stack compatibility: jiayi/hitterobs stored the same payload at the PT top
            # level. Strict mode accepts it only if the payload itself satisfies schema 3.
            state = loaded.get("hope_exact_resume_state")
        return state

    @staticmethod
    def _checkpoint_byte_snapshot(path) -> io.BytesIO:
        """Read one immutable checkpoint snapshot for strict preflight and base loading."""

        try:
            filesystem_path = os.fspath(path)
        except TypeError as exc:
            raise TypeError(
                "required exact resume needs a filesystem checkpoint path"
            ) from exc
        with open(filesystem_path, "rb") as stream:
            return io.BytesIO(stream.read())

    @staticmethod
    def _validate_checkpoint_module_state(
        module: object,
        saved_state: object,
        *,
        prefix: str,
        label: str,
    ) -> Mapping[str, torch.Tensor]:
        """Read-only strict ``state_dict`` compatibility check.

        ``torch.nn.Module.load_state_dict`` is not a validator: it may copy a
        prefix of the checkpoint before a later shape/key error is raised.
        Formal resume therefore compares the complete key/type/shape/dtype
        envelope while all live modules are still untouched.  Formal
        ActionBall deliberately supports the current tensor-only ActorCritic
        state contract; modules using PyTorch ``get_extra_state`` are rejected
        rather than executing an opaque mutation hook during admission.
        """

        state_dict = getattr(module, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError(f"{prefix} live {label} has no state_dict()")
        live_state = state_dict()
        if not isinstance(live_state, Mapping) or not isinstance(
            saved_state, Mapping
        ):
            raise RuntimeError(
                f"{prefix} {label} state must be a mapping"
            )
        if any(type(key) is not str for key in live_state) or any(
            type(key) is not str for key in saved_state
        ):
            raise RuntimeError(
                f"{prefix} {label} state keys must be exact strings"
            )
        live_keys = set(live_state)
        saved_keys = set(saved_state)
        if saved_keys != live_keys:
            missing = sorted(live_keys - saved_keys)
            unexpected = sorted(saved_keys - live_keys)
            raise RuntimeError(
                f"{prefix} {label} state keys differ from the live module; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, live_value in live_state.items():
            saved_value = saved_state[key]
            if not torch.is_tensor(live_value) or not torch.is_tensor(
                saved_value
            ):
                raise RuntimeError(
                    f"{prefix} {label} state {key!r} must be a tensor; "
                    "PyTorch extra_state is unsupported by formal resume"
                )
            if (
                tuple(saved_value.shape) != tuple(live_value.shape)
                or saved_value.dtype != live_value.dtype
            ):
                raise RuntimeError(
                    f"{prefix} {label} state {key!r} shape/dtype differs "
                    "from the live module"
                )
            if (
                (saved_value.is_floating_point() or saved_value.is_complex())
                and not bool(torch.isfinite(saved_value).all().item())
            ):
                raise RuntimeError(
                    f"{prefix} {label} state {key!r} is non-finite"
                )
        return saved_state

    @staticmethod
    def _validate_checkpoint_normalizer_moments(
        saved_state: Mapping[str, torch.Tensor],
        *,
        prefix: str,
        role: str,
        expected_width: int,
    ) -> None:
        """Validate the saved empirical moments, including their semantic width."""

        semantic_aliases = {
            "mean": "mean",
            "running_mean": "mean",
            "var": "var",
            "variance": "var",
            "running_var": "var",
            "std": "std",
            "running_std": "std",
            "count": "count",
            "running_count": "count",
            "num_batches_tracked": "count",
        }
        semantic = {"mean": [], "var": [], "std": [], "count": []}
        for key, value in saved_state.items():
            leaf = key.rsplit(".", 1)[-1].lstrip("_")
            semantic_name = semantic_aliases.get(leaf)
            if semantic_name is not None:
                semantic[semantic_name].append((key, value))
        if len(semantic["mean"]) != 1 or len(semantic["count"]) != 1:
            raise RuntimeError(
                f"{prefix} empirical {role} normalizer must contain one "
                "mean and one count buffer"
            )
        if len(semantic["var"]) > 1 or len(semantic["std"]) > 1 or (
            not semantic["var"] and not semantic["std"]
        ):
            raise RuntimeError(
                f"{prefix} empirical {role} normalizer variance/std buffers "
                "are ambiguous or absent"
            )
        _mean_key, mean = semantic["mean"][0]
        _count_key, count = semantic["count"][0]
        moments = [entry[1] for entry in semantic["var"] + semantic["std"]]
        if (
            mean.ndim < 1
            or mean.numel() != expected_width
            or any(tuple(value.shape) != tuple(mean.shape) for value in moments)
        ):
            raise RuntimeError(
                f"{prefix} empirical {role} normalizer must have exact "
                f"semantic width {expected_width}"
            )
        if count.numel() != 1 or count.dtype == torch.bool or count.is_complex():
            raise RuntimeError(
                f"{prefix} empirical {role} normalizer count is invalid"
            )
        count_value = float(count.detach().cpu().item())
        if not math.isfinite(count_value) or count_value < 0.0:
            raise RuntimeError(
                f"{prefix} empirical {role} normalizer count is invalid"
            )
        if any(
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            or bool((value < 0).any().item())
            for value in moments
        ):
            raise RuntimeError(
                f"{prefix} empirical {role} normalizer variance/std is invalid"
            )

    def _preflight_required_checkpoint_normalizers(
        self, loaded: Mapping[str, object], *, prefix: str
    ) -> None:
        """Validate actor/critic normalizer bytes before any live state copy."""

        empirical = getattr(self, "empirical_normalization", None)
        if type(empirical) is not bool:
            raise RuntimeError(
                f"{prefix} empirical_normalization is not an exact bool"
            )
        fresh_211 = self._action_ball_211_wait_mask_required()
        if fresh_211 and not empirical:
            raise RuntimeError(
                f"{prefix} fresh ActionBall211 requires actor/critic empirical "
                "normalizers"
            )
        fields = (
            ("actor", "obs_norm_state_dict"),
            ("critic", "privileged_obs_norm_state_dict"),
        )
        for role, field in fields:
            attribute, normalizer, _aliases = self._resolve_runtime_normalizer(
                role
            )
            saved = loaded.get(field)
            if not empirical:
                if field in loaded:
                    raise RuntimeError(
                        f"{prefix} contains {role} normalizer state while "
                        "normalization is disabled"
                    )
                continue
            if attribute is None or not is_empirical_normalizer(normalizer):
                raise RuntimeError(
                    f"{prefix} has no live empirical {role} normalizer"
                )
            live_binding = self._validate_empirical_normalizer_state(
                role=role,
                attribute_name=attribute,
                normalizer=normalizer,
            )
            self._validate_checkpoint_module_state(
                normalizer,
                saved,
                prefix=prefix,
                label=f"{role} normalizer",
            )
            expected_width = (
                (_A211_ACTOR_WIDTH if role == "actor" else _A211_CRITIC_WIDTH)
                if fresh_211
                else int(live_binding["semantic_width"])
            )
            self._validate_checkpoint_normalizer_moments(
                saved,
                prefix=prefix,
                role=role,
                expected_width=expected_width,
            )

    @staticmethod
    def _validate_declared_exact_resume_support(
        value: object, *, prefix: str, label: str
    ) -> None:
        """Reject any nested producer declaration that explicitly forbids resume."""

        pending = [(label, value)]
        seen = set()
        visited = 0
        while pending:
            path, current = pending.pop()
            if isinstance(current, Mapping):
                identity = id(current)
                if identity in seen:
                    continue
                seen.add(identity)
                visited += 1
                if visited > 100000:
                    raise RuntimeError(
                        f"{prefix} {label} is too large to preflight safely"
                    )
                if "exact_resume_supported" in current:
                    supported = current["exact_resume_supported"]
                    if type(supported) is not bool:
                        raise RuntimeError(
                            f"{prefix} {path}.exact_resume_supported must be "
                            "an exact bool"
                        )
                    if not supported:
                        raise RuntimeError(
                            f"{prefix} {path} explicitly declares "
                            "exact_resume_supported=false; this checkpoint is "
                            "fresh/warm-start only"
                        )
                for key, child in current.items():
                    pending.append((f"{path}.{key}", child))
            elif isinstance(current, (list, tuple)):
                identity = id(current)
                if identity in seen:
                    continue
                seen.add(identity)
                visited += 1
                if visited > 100000:
                    raise RuntimeError(
                        f"{prefix} {label} is too large to preflight safely"
                    )
                for index, child in enumerate(current):
                    pending.append((f"{path}[{index}]", child))

    def _preflight_required_environment_resume_state(
        self, nested: object, *, prefix: str
    ) -> None:
        """Read-only schema/identity/support validation for the inner state.

        Command terms may expose ``validate_exact_resume_state_dict`` with
        signature ``(state, *, strict=True)``.  Formal ActionBall resume
        requires that read-only hook; the later loader remains the sole commit
        operation.  This deliberately leaves fresh training unchanged while
        giving command payloads a fail-closed admission interface.
        """

        if not isinstance(nested, Mapping):
            raise RuntimeError(
                f"{prefix} requires an environment_resume_state mapping"
            )
        schema = nested.get("schema_version")
        if type(schema) is not int or schema not in (3, 4):
            raise RuntimeError(
                f"{prefix} requires a supported schema-3/4 "
                "environment_resume_state"
            )
        env = getattr(self.env, "unwrapped", self.env)
        (
            _action_manager,
            action_names,
            action_terms,
        ) = self._ordered_action_resume_terms(env)
        runtime_action_state = self._action_runtime_state_required(action_terms)
        fresh_211 = self._action_ball_211_wait_mask_required()
        if (fresh_211 or runtime_action_state) and schema != 4:
            raise RuntimeError(
                f"{prefix} fresh ActionBall211 requires inner action schema 4; "
                "schema 3 is fresh-only"
            )
        expected_keys = {
            "schema_version",
            "common_step_counter",
            "active_term_names",
            "command_terms",
        }
        if schema == 4:
            expected_keys.update({"active_action_term_names", "action_terms"})
        if set(nested) != expected_keys:
            raise RuntimeError(
                f"{prefix} inner environment schema {schema} keys do not "
                "match the supported envelope"
            )
        common_step_counter = nested["common_step_counter"]
        if type(common_step_counter) is not int or common_step_counter < 0:
            raise RuntimeError(
                f"{prefix} environment common_step_counter is invalid"
            )

        manager = getattr(env, "command_manager", None)
        raw_command_names = (
            tuple(getattr(manager, "active_terms", ()))
            if manager is not None
            else ()
        )
        command_names = tuple(str(name) for name in raw_command_names)
        saved_command_names = nested["active_term_names"]
        command_states = nested["command_terms"]
        if (
            type(saved_command_names) is not list
            or any(type(name) is not str for name in saved_command_names)
            or len(saved_command_names) != len(set(saved_command_names))
            or not isinstance(command_states, Mapping)
            or tuple(saved_command_names) != command_names
            or tuple(command_states) != command_names
        ):
            raise RuntimeError(
                f"{prefix} ordered command term identity/state is incomplete"
            )
        strict_action_ball = self._strict_exact_resume_target_mode() == "action_ball"
        if strict_action_ball:
            missing_pair = [
                name
                for name in ("racket_target", "motion")
                if name not in command_names
            ]
            if missing_pair:
                raise RuntimeError(
                    f"{prefix} ActionBall command dependency pair is "
                    f"incomplete; missing={missing_pair}"
                )
            motion_finalize = getattr(
                manager.get_term("motion"),
                "finalize_action_ball_exact_resume",
                None,
            )
            if not callable(motion_finalize):
                raise RuntimeError(
                    f"{prefix} ActionBall Motion command lacks "
                    "finalize_action_ball_exact_resume()"
                )
        for raw_name, name in zip(raw_command_names, command_names):
            term = manager.get_term(raw_name)
            record = command_states[name]
            if not isinstance(record, Mapping):
                raise RuntimeError(
                    f"{prefix} command term {name!r} state is not a mapping"
                )
            mode = record.get("capture_mode")
            if mode != "explicit":
                if strict_action_ball:
                    raise RuntimeError(
                        f"{prefix} ActionBall command term {name!r} lacks "
                        "explicit resume state"
                    )
                continue
            if set(record) != {"capture_mode", "term_type", "exact_state"}:
                raise RuntimeError(
                    f"{prefix} command term {name!r} explicit state has an "
                    "unsupported envelope"
                )
            term_type = f"{type(term).__module__}.{type(term).__qualname__}"
            if record["term_type"] != term_type:
                raise RuntimeError(
                    f"{prefix} command term {name!r} type changed"
                )
            exact_state = record["exact_state"]
            if not isinstance(exact_state, Mapping):
                raise RuntimeError(
                    f"{prefix} command term {name!r} exact state is not a mapping"
                )
            self._validate_declared_exact_resume_support(
                exact_state,
                prefix=prefix,
                label=f"command_terms.{name}.exact_state",
            )
            getter = getattr(term, "exact_resume_state_dict", None)
            loader = getattr(term, "load_exact_resume_state_dict", None)
            if callable(getter) != callable(loader):
                raise RuntimeError(
                    f"{prefix} command term {name!r} has a partial exact "
                    "resume getter/loader interface"
                )
            if strict_action_ball and not callable(getter):
                raise RuntimeError(
                    f"{prefix} ActionBall command term {name!r} cannot "
                    "capture/restore exact state"
                )
            validator = getattr(term, "validate_exact_resume_state_dict", None)
            if validator is not None and not callable(validator):
                raise RuntimeError(
                    f"{prefix} command term {name!r} has a non-callable "
                    "read-only resume validator"
                )
            if strict_action_ball and not callable(validator):
                raise RuntimeError(
                    f"{prefix} ActionBall command term {name!r} lacks "
                    "validate_exact_resume_state_dict(state, strict=True)"
                )
            if callable(validator):
                validator(exact_state, strict=True)

        if strict_action_ball:
            racket_exact = command_states["racket_target"]["exact_state"]
            motion_exact = command_states["motion"]["exact_state"]
            motion_birth = (
                motion_exact.get("action_ball_birth")
                if isinstance(motion_exact, Mapping)
                else None
            )
            racket_digest = (
                racket_exact.get("integrity_sha256")
                if isinstance(racket_exact, Mapping)
                else None
            )
            motion_digest = (
                motion_birth.get("shared_racket_state_sha256")
                if isinstance(motion_birth, Mapping)
                else None
            )
            digests = (racket_digest, motion_digest)
            if any(
                type(digest) is not str
                or len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digests
            ):
                raise RuntimeError(
                    f"{prefix} ActionBall Racket/Motion cross-payload digest "
                    "is absent or invalid"
                )
            if racket_digest != motion_digest:
                raise RuntimeError(
                    f"{prefix} ActionBall Racket/Motion cross-payload digest "
                    "differs"
                )

        if schema == 4:
            saved_action_names = nested["active_action_term_names"]
            action_states = nested["action_terms"]
            if (
                type(saved_action_names) is not list
                or any(type(name) is not str for name in saved_action_names)
                or len(saved_action_names) != len(set(saved_action_names))
                or not isinstance(action_states, Mapping)
                or tuple(saved_action_names) != action_names
                or tuple(action_states) != action_names
            ):
                raise RuntimeError(
                    f"{prefix} ordered action term identity/state is incomplete"
                )
            if fresh_211 and not action_names:
                raise RuntimeError(
                    f"{prefix} fresh ActionBall211 inner schema 4 has no "
                    "active action terms"
                )
            staged_names = set()
            for name in action_names:
                term = action_terms[name]
                record = action_states[name]
                if not isinstance(record, Mapping):
                    raise RuntimeError(
                        f"{prefix} action term {name!r} state is not a mapping"
                    )
                term_type = f"{type(term).__module__}.{type(term).__qualname__}"
                if record.get("term_type") != term_type:
                    raise RuntimeError(
                        f"{prefix} action term {name!r} type changed"
                    )
                mode = record.get("capture_mode")
                getter = getattr(term, "action_delay_exact_resume_state_dict", None)
                validator = getattr(
                    term, "validate_action_delay_exact_resume_state_dict", None
                )
                loader = getattr(
                    term, "load_action_delay_exact_resume_state_dict", None
                )
                interface = tuple(
                    callable(value) for value in (getter, validator, loader)
                )
                if any(interface) and not all(interface):
                    raise RuntimeError(
                        f"{prefix} action term {name!r} has a partial "
                        "schema-4 resume interface"
                    )
                if mode == "identity_only":
                    if set(record) != {"capture_mode", "term_type"} or all(
                        interface
                    ):
                        raise RuntimeError(
                            f"{prefix} action term {name!r} has an invalid "
                            "identity-only schema-4 record"
                        )
                    continue
                if mode != "explicit_delay" or set(record) != {
                    "capture_mode",
                    "term_type",
                    "exact_state",
                }:
                    raise RuntimeError(
                        f"{prefix} action term {name!r} has an invalid "
                        "schema-4 record"
                    )
                if not all(interface):
                    raise RuntimeError(
                        f"{prefix} action term {name!r} cannot validate and "
                        "restore schema-4 state"
                    )
                exact_state = record["exact_state"]
                if not isinstance(exact_state, Mapping):
                    raise RuntimeError(
                        f"{prefix} action term {name!r} exact state is not a "
                        "mapping"
                    )
                validator(exact_state, strict=True)
                staged_names.add(name)
            missing = [
                name
                for name, term in action_terms.items()
                if self._action_runtime_state_required({name: term})
                and name not in staged_names
            ]
            if missing:
                raise RuntimeError(
                    f"{prefix} required action runtime state is missing for "
                    f"terms {missing}"
                )

    def _apply_formal_preloaded_checkpoint(
        self,
        loaded: Mapping[str, object],
        *,
        load_optimizer: bool,
        prefix: str,
    ):
        """Apply an already safely decoded formal checkpoint exactly once.

        Upstream RSL-RL's ``load`` performs its own unrestricted ``torch.load``.
        Formal ActionBall must not call it: doing so would both reopen a path
        after admission and execute arbitrary pickle reducers.  This method
        mirrors the small state-application portion after strict preflight.
        """

        if load_optimizer is not True:
            raise RuntimeError(
                f"{prefix} requires optimizer restoration"
            )
        policy = getattr(getattr(self, "alg", None), "policy", None)
        policy_loader = getattr(policy, "load_state_dict", None)
        optimizer = getattr(getattr(self, "alg", None), "optimizer", None)
        optimizer_loader = getattr(optimizer, "load_state_dict", None)
        if not callable(policy_loader) or not callable(optimizer_loader):
            raise RuntimeError(
                f"{prefix} runner lacks policy/optimizer state loaders"
            )
        model_state = loaded.get("model_state_dict")
        optimizer_state = loaded.get("optimizer_state_dict")
        if not isinstance(model_state, Mapping):
            raise RuntimeError(f"{prefix} lacks model_state_dict")
        policy_loader(model_state, strict=True)
        optimizer_loader(optimizer_state)

        empirical = getattr(self, "empirical_normalization", None)
        if type(empirical) is not bool:
            raise RuntimeError(
                f"{prefix} empirical_normalization is not an exact bool"
            )
        _actor_attribute, actor_normalizer, _actor_aliases = (
            self._resolve_runtime_normalizer("actor")
        )
        _critic_attribute, critic_normalizer, _critic_aliases = (
            self._resolve_runtime_normalizer("critic")
        )
        normalizer_fields = (
            ("obs_norm_state_dict", actor_normalizer, "actor"),
            (
                "privileged_obs_norm_state_dict",
                critic_normalizer,
                "critic",
            ),
        )
        for field, normalizer, role in normalizer_fields:
            saved = loaded.get(field)
            if empirical:
                loader = getattr(normalizer, "load_state_dict", None)
                if not isinstance(saved, Mapping) or not callable(loader):
                    raise RuntimeError(
                        f"{prefix} lacks empirical {role} normalizer state"
                    )
                loader(saved, strict=True)
            elif field in loaded:
                raise RuntimeError(
                    f"{prefix} contains {role} normalizer state while "
                    "normalization is disabled"
                )
        iteration = loaded.get("iter")
        if type(iteration) is not int or iteration < 0:
            raise RuntimeError(f"{prefix} has invalid iteration")
        self.current_learning_iteration = iteration
        return loaded.get("infos")

    def load_formal_action_ball_checkpoint_bytes(
        self,
        checkpoint_bytes: bytes,
        *,
        checkpoint_path: object,
        expected_sha256: str,
        expected_size_bytes: int,
        load_optimizer: bool = True,
    ):
        """Load one admitted immutable byte string with safe deserialization."""

        if not self._formal_action_ball_runtime_bootstrap_required():
            raise RuntimeError(
                "immutable formal checkpoint load requires exact-lineage "
                "ActionBall"
            )
        if type(checkpoint_bytes) is not bytes:
            raise TypeError("formal checkpoint bytes must be exact bytes")
        if (
            type(expected_size_bytes) is not int
            or expected_size_bytes <= 0
            or len(checkpoint_bytes) != expected_size_bytes
        ):
            raise RuntimeError(
                "formal checkpoint byte size differs from admission"
            )
        actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        if (
            type(expected_sha256) is not str
            or expected_sha256 != actual_sha256
        ):
            raise RuntimeError(
                "formal checkpoint bytes differ from admitted SHA-256"
            )
        return self.load(
            os.fspath(checkpoint_path),
            load_optimizer=load_optimizer,
            _formal_immutable_checkpoint_bytes=checkpoint_bytes,
        )

    def _validate_required_adam_state(
        self,
        optimizer_state,
        *,
        prefix: str,
        expected_learning_rate: float,
    ) -> None:
        """Validate a complete Adam continuation state against the live parameter layout."""

        optimizer = getattr(getattr(self, "alg", None), "optimizer", None)
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise RuntimeError(
                f"{prefix} cannot validate optimizer type "
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}; "
                "task-first exact resume currently requires Adam/AdamW"
            )
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
            "state",
            "param_groups",
        }:
            raise RuntimeError(
                f"{prefix} is missing a canonical optimizer_state_dict"
            )
        saved_state = optimizer_state["state"]
        saved_groups = optimizer_state["param_groups"]
        current_groups = optimizer.param_groups
        if (
            not isinstance(saved_state, dict)
            or not isinstance(saved_groups, list)
            or len(saved_groups) != len(current_groups)
            or not saved_groups
        ):
            raise RuntimeError(
                f"{prefix} optimizer state/group structure does not match the live Adam"
            )

        saved_ids = []
        live_parameters = []
        amsgrad_flags = []
        for group_index, (saved_group, current_group) in enumerate(
            zip(saved_groups, current_groups)
        ):
            if (
                type(saved_group) is not dict
                or set(saved_group) != set(current_group)
                or type(saved_group.get("params")) is not list
                or type(current_group.get("params")) is not list
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} is malformed"
                )
            group_ids = saved_group["params"]
            group_parameters = current_group["params"]
            if not group_ids or len(group_ids) != len(group_parameters):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has the wrong size"
                )
            if any(type(param_id) is not int for param_id in group_ids):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has non-integer IDs"
                )
            if any(not torch.is_tensor(parameter) for parameter in group_parameters):
                raise RuntimeError(
                    f"{prefix} live optimizer param group {group_index} is malformed"
                )
            saved_lr = saved_group.get("lr")
            if (
                type(saved_lr) not in (int, float)
                or not math.isfinite(float(saved_lr))
                or float(saved_lr) <= 0.0
                or float(saved_lr) != float(expected_learning_rate)
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has an invalid "
                    "or inconsistent learning rate"
                )
            betas = saved_group.get("betas")
            if (
                type(betas) is not tuple
                or len(betas) != 2
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    or float(value) >= 1.0
                    for value in betas
                )
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has invalid betas"
                )
            eps = saved_group.get("eps")
            weight_decay = saved_group.get("weight_decay")
            if (
                type(eps) not in (int, float)
                or not math.isfinite(float(eps))
                or float(eps) <= 0.0
                or type(weight_decay) not in (int, float)
                or not math.isfinite(float(weight_decay))
                or float(weight_decay) < 0.0
            ):
                raise RuntimeError(
                    f"{prefix} optimizer param group {group_index} has invalid "
                    "epsilon or weight decay"
                )
            for key in set(current_group) - {"params", "lr"}:
                saved_value = saved_group[key]
                current_value = current_group[key]
                if torch.is_tensor(saved_value) or torch.is_tensor(current_value):
                    equal = (
                        torch.is_tensor(saved_value)
                        and torch.is_tensor(current_value)
                        and torch.equal(saved_value, current_value)
                    )
                else:
                    equal = type(saved_value) is type(current_value) and saved_value == current_value
                if not equal:
                    raise RuntimeError(
                        f"{prefix} optimizer param group {group_index} changed {key!r}"
                    )
            saved_ids.extend(group_ids)
            live_parameters.extend(group_parameters)
            amsgrad_flags.extend(
                [bool(saved_group.get("amsgrad", False))] * len(group_ids)
            )

        if (
            len(saved_ids) != len(set(saved_ids))
            or any(type(param_id) is not int for param_id in saved_state)
            or set(saved_state) != set(saved_ids)
        ):
            raise RuntimeError(
                f"{prefix} optimizer state does not cover the exact parameter ID set"
            )

        for param_id, parameter, uses_amsgrad in zip(
            saved_ids, live_parameters, amsgrad_flags
        ):
            entry = saved_state[param_id]
            expected_entry_keys = {"step", "exp_avg", "exp_avg_sq"}
            if uses_amsgrad:
                expected_entry_keys.add("max_exp_avg_sq")
            if type(entry) is not dict or set(entry) != expected_entry_keys:
                raise RuntimeError(
                    f"{prefix} optimizer state for parameter {param_id} lacks Adam moments"
                )
            if any(
                not torch.is_tensor(entry[name])
                or tuple(entry[name].shape) != tuple(parameter.shape)
                or entry[name].dtype != parameter.dtype
                or (
                    torch.is_floating_point(entry[name])
                    and not bool(torch.isfinite(entry[name]).all())
                )
                for name in ("exp_avg", "exp_avg_sq")
            ):
                raise RuntimeError(
                    f"{prefix} optimizer moments for parameter {param_id} "
                    "do not match the live parameter"
                )
            step = entry["step"]
            if torch.is_tensor(step):
                step_value = (
                    float(step.detach().cpu().item())
                    if step.numel() == 1
                    and step.dtype != torch.bool
                    and not torch.is_complex(step)
                    else float("nan")
                )
                valid_step = (
                    step.numel() == 1
                    and step.dtype != torch.bool
                    and not torch.is_complex(step)
                    and bool(torch.isfinite(step).all())
                    and step_value > 0.0
                    and step_value.is_integer()
                )
            else:
                step_value = (
                    float(step) if type(step) in (int, float) else float("nan")
                )
                valid_step = (
                    type(step) in (int, float)
                    and math.isfinite(step_value)
                    and step_value > 0.0
                    and step_value.is_integer()
                )
            if not valid_step:
                raise RuntimeError(
                    f"{prefix} optimizer step for parameter {param_id} is invalid"
                )
            second_moment = entry["exp_avg_sq"]
            if not torch.is_floating_point(second_moment) or bool(
                (second_moment < 0.0).any()
            ):
                raise RuntimeError(
                    f"{prefix} optimizer second moment for parameter {param_id} is invalid"
                )
            if uses_amsgrad:
                maximum = entry.get("max_exp_avg_sq")
                if (
                    not torch.is_tensor(maximum)
                    or tuple(maximum.shape) != tuple(parameter.shape)
                    or maximum.dtype != parameter.dtype
                    or (
                        torch.is_floating_point(maximum)
                        and not bool(torch.isfinite(maximum).all())
                    )
                    or not torch.is_floating_point(maximum)
                    or bool((maximum < 0.0).any())
                    or bool((maximum < second_moment).any())
                ):
                    raise RuntimeError(
                        f"{prefix} optimizer AMSGrad state for parameter {param_id} "
                        "does not match the live parameter"
                    )

    def _validate_checkpoint_runtime_bootstrap_binding(
        self,
        *,
        checkpoint_infos: Mapping[str, object],
        exact_state: Mapping[str, object],
        prefix: str,
    ) -> None:
        """Reopen the source receipt and compare its location-free lineage.

        A same-namespace verifier sees byte-identical receipt bindings.
        A deliberate cross-log resume may relocate the four runtime files,
        but the current trainer-minted receipt must preserve the exact
        location-free lineage payload.  Merely replacing both the source
        checkpoint and the old receipt cannot choose the current in-memory
        expected lineage.
        """

        if not self._formal_action_ball_runtime_bootstrap_required():
            return
        current = self._validated_runtime_bootstrap_binding()
        keys = (
            _RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY,
            _RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY,
            _RUNTIME_BOOTSTRAP_RECEIPT_KEY,
        )
        try:
            infos_binding = {
                key: checkpoint_infos[key] for key in keys
            }
            state_binding = {key: exact_state[key] for key in keys}
        except KeyError as exc:
            raise RuntimeError(
                f"{prefix} lacks runtime bootstrap receipt binding"
            ) from exc
        if infos_binding != state_binding:
            raise RuntimeError(
                f"{prefix} runtime bootstrap infos/exact-state bindings "
                "differ"
            )

        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_evaluation_inbox as inbox_protocol,
            action_ball_runtime_bootstrap as bootstrap_protocol,
        )

        artifact = infos_binding[_RUNTIME_BOOTSTRAP_RECEIPT_KEY]
        try:
            inbox_protocol.verify_artifact_receipt(
                artifact,
                label="source checkpoint runtime bootstrap receipt",
            )
            document = inbox_protocol.strict_read_json(
                artifact["path"],
                label="source checkpoint runtime bootstrap receipt",
            )
            content = document["content"]
            source = content["source"]
            validated = (
                bootstrap_protocol.validate_runtime_bootstrap_receipt_document(
                    document,
                    expected_repo_root=source["repo_root"],
                    expected_task_id=bootstrap_protocol.TASK_ID,
                    expected_training_launch_claim_sha256=content[
                        "training_launch_claim_sha256"
                    ],
                    expected_training_contract_path=content[
                        "training_contract"
                    ]["path"],
                    expected_environment_config_pickle_path=content[
                        "environment_config_pickle"
                    ]["path"],
                    expected_agent_config_pickle_path=content[
                        "agent_config_pickle"
                    ]["path"],
                    expected_runtime_identity_path=content[
                        "runtime_identity"
                    ]["path"],
                    expected_source_commit_oid=source[
                        "head_commit_oid"
                    ],
                )
            )
            lineage_sha256 = (
                bootstrap_protocol.runtime_bootstrap_lineage_payload_sha256(
                    content
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"{prefix} source runtime bootstrap receipt or artifacts "
                "failed live validation"
            ) from exc
        if (
            document.get("content_sha256")
            != infos_binding[_RUNTIME_BOOTSTRAP_RECEIPT_SHA_KEY]
            or lineage_sha256
            != infos_binding[_RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY]
            or self._runtime_bootstrap_json_clone(validated)
            != self._runtime_bootstrap_json_clone(content)
        ):
            raise RuntimeError(
                f"{prefix} source runtime bootstrap binding is internally "
                "inconsistent"
            )
        if (
            content["training_contract"]["sha256"]
            != checkpoint_infos.get(CHECKPOINT_CONTRACT_SHA_KEY)
            or content["training_launch_claim_sha256"]
            != checkpoint_infos.get(CHECKPOINT_LAUNCH_CLAIM_SHA_KEY)
        ):
            raise RuntimeError(
                f"{prefix} runtime bootstrap differs from checkpoint "
                "contract/claim"
            )
        if (
            infos_binding[_RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY]
            != current[_RUNTIME_BOOTSTRAP_LINEAGE_SHA_KEY]
        ):
            raise RuntimeError(
                f"{prefix} runtime bootstrap location-free lineage differs "
                "from the current trainer runtime"
            )

    def _preflight_required_exact_resume_checkpoint(
        self,
        loaded,
        *,
        path: str,
        load_optimizer: bool,
    ) -> dict:
        """Validate the non-negotiable task-first resume envelope before loading model bytes."""

        prefix = f"required exact resume checkpoint {path!r}"
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{prefix} must be a dictionary")
        if self.log_dir is None:
            raise RuntimeError(
                f"{prefix} cannot use the evaluation load path without environment restore"
            )
        if load_optimizer is not True:
            raise RuntimeError(
                f"{prefix} refuses actor-only loading: load_optimizer must be True"
            )
        if getattr(getattr(self, "alg", None), "rnd", None) is not None:
            # Upstream persists a second model/optimizer pair for RND. Until that pair has the
            # same complete-state validator as the policy optimizer, accepting it here would make
            # the "exact" construction contract false.
            raise RuntimeError(
                f"{prefix} does not yet support exact RND resume; disable RND or add "
                "strict rnd_state_dict/rnd_optimizer_state_dict validation"
            )
        checkpoint_infos = loaded.get("infos")
        checkpoint_contract_schema = (
            checkpoint_infos.get(CHECKPOINT_CONTRACT_SCHEMA_KEY)
            if isinstance(checkpoint_infos, dict)
            else None
        )
        checkpoint_lineage_exact = (
            checkpoint_infos.get(CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY)
            if isinstance(checkpoint_infos, dict)
            else None
        )
        expected_lineage_exact = (
            1 if self.training_contract_lineage_exact else 0
        )
        if (
            not isinstance(checkpoint_infos, dict)
            or type(checkpoint_contract_schema) is not int
            or checkpoint_contract_schema != self.training_contract_schema_version
            or checkpoint_infos.get(CHECKPOINT_CONTRACT_SHA_KEY)
            != self.training_contract_sha256
            or type(checkpoint_lineage_exact) is not int
            or checkpoint_lineage_exact != expected_lineage_exact
        ):
            raise RuntimeError(
                f"{prefix} is not bound to this training contract with "
                f"the expected lineage={expected_lineage_exact}"
            )

        checkpoint_iteration = loaded.get("iter")
        if type(checkpoint_iteration) is not int or checkpoint_iteration < 0:
            raise RuntimeError(f"{prefix} has an invalid base iteration")

        state = self._checkpoint_exact_resume_state(loaded)
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != _EXACT_RESUME_SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"{prefix} requires hope_exact_resume_state schema 3; "
                "legacy warm-start fallback is forbidden"
            )
        policy = getattr(getattr(self, "alg", None), "policy", None)
        self._validate_checkpoint_module_state(
            policy,
            loaded.get("model_state_dict"),
            prefix=prefix,
            label="policy",
        )
        self._preflight_required_checkpoint_normalizers(
            loaded,
            prefix=prefix,
        )
        self._validate_checkpoint_runtime_bootstrap_binding(
            checkpoint_infos=checkpoint_infos,
            exact_state=state,
            prefix=prefix,
        )
        required_fields = {
            "next_learning_iteration",
            "tot_timesteps",
            "tot_time",
            "algorithm_learning_rate",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "torch_cuda_random_states",
            "torch_cuda_device_count",
            "environment_resume_state",
        }
        missing = sorted(required_fields - set(state))
        if missing:
            raise RuntimeError(
                f"{prefix} schema-3 state is incomplete; missing={missing}"
            )
        tot_timesteps = state["tot_timesteps"]
        tot_time = state["tot_time"]
        learning_rate = state["algorithm_learning_rate"]
        cuda_count = state["torch_cuda_device_count"]
        if type(tot_timesteps) is not int or tot_timesteps < 0:
            raise RuntimeError(f"{prefix} has invalid tot_timesteps")
        if (
            type(tot_time) not in (int, float)
            or not math.isfinite(float(tot_time))
            or float(tot_time) < 0.0
        ):
            raise RuntimeError(f"{prefix} has invalid tot_time")
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise RuntimeError(f"{prefix} has invalid algorithm_learning_rate")
        if type(cuda_count) is not int or cuda_count < 0:
            raise RuntimeError(f"{prefix} has invalid torch_cuda_device_count")
        next_iteration = state["next_learning_iteration"]
        if (
            type(next_iteration) is not int
            or next_iteration != checkpoint_iteration + 1
        ):
            raise RuntimeError(
                f"{prefix} has stale next_learning_iteration: "
                f"checkpoint={next_iteration!r}, expected={checkpoint_iteration + 1}"
            )
        nested = state["environment_resume_state"]
        if (
            not isinstance(nested, dict)
            or type(nested.get("schema_version")) is not int
            or nested["schema_version"] not in (3, 4)
        ):
            raise RuntimeError(
                f"{prefix} requires a schema-3/4 environment_resume_state"
            )
        self._preflight_required_environment_resume_state(
            nested,
            prefix=prefix,
        )
        live_env = getattr(self.env, "unwrapped", self.env)
        _manager, _names, live_action_terms = self._ordered_action_resume_terms(
            live_env
        )
        if (
            self._action_runtime_state_required(live_action_terms)
            and nested["schema_version"] != 4
        ):
            raise RuntimeError(
                f"{prefix} cannot resume required action runtime state from "
                "schema 3; launch fresh or use a schema-4 checkpoint"
            )
        nested_common_step_counter = nested.get("common_step_counter")
        if (
            type(nested_common_step_counter) is not int
            or nested_common_step_counter < 0
        ):
            raise RuntimeError(
                f"{prefix} schema-3 environment common_step_counter must be a "
                "nonnegative plain integer"
            )
        # Dry validation occurs before policy/optimizer bytes are applied. The same payload is
        # validated once more and committed atomically immediately before the resumed env.reset().
        self._validated_exact_rng_state(state)
        self._validate_required_adam_state(
            loaded.get("optimizer_state_dict"),
            prefix=prefix,
            expected_learning_rate=float(learning_rate),
        )
        return state

    def load(self, path: str, load_optimizer: bool = True, **kwargs):
        """Load a checkpoint; on a training resume also restore curriculum progress.

        人话:base 的 load 只拿回权重/优化器/迭代号,环境课程全部回到第 0 步。这里在训练
        续跑(log_dir 不为 None)时把精确续训包里的课程主时钟和命令项状态一并恢复;老档没
        有状态就按迭代号精确推算主时钟。评测器(isaac_bank_exam / play)也走 runner.load,
        但它们 log_dir=None 且自带确定性调度 —— 那条路保持与移植前逐字节相同的行为。
        ``require_exact_resume_state=True`` 是 task-first/action-ball 训练的构造期铁律:
        optimizer、外层 schema 3、内层 schema 3、迭代连续性或命令项 strict state 任一缺失
        都拒绝,绝不把 actor-only checkpoint 静默解释成 warm start。Action-ball 的 load
        只恢复内存状态；首次 simulator true reset 延迟到 learn(),避免 load 本身采样。
        """
        if getattr(self, "_action_ball_full_mdp_run_mode", None) == (
            _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ):
            raise RuntimeError(
                "single_action_lean forbids checkpoint load"
            )
        immutable_checkpoint_bytes = kwargs.pop(
            "_formal_immutable_checkpoint_bytes", None
        )
        strict_resume = bool(getattr(self, "require_exact_resume_state", False))
        if strict_resume:
            # Calling load(), rather than a caller-authored constructor flag,
            # is the exact proof that checkpoint/resume was requested.  Fail
            # before opening or applying any checkpoint when a live command
            # lacks the paired explicit-state hooks.
            self._validate_task_first_exact_resume_terms()
        formal_safe_resume = (
            strict_resume
            and self._formal_action_ball_runtime_bootstrap_required()
        )
        if immutable_checkpoint_bytes is not None and (
            type(immutable_checkpoint_bytes) is not bytes
            or not strict_resume
            or not formal_safe_resume
        ):
            raise RuntimeError(
                "immutable checkpoint bytes require formal strict ActionBall"
            )
        resolved_checkpoint_path = str(
            pathlib.Path(path).expanduser().resolve()
        )
        snapshot = (
            io.BytesIO(immutable_checkpoint_bytes)
            if immutable_checkpoint_bytes is not None
            else (
                self._checkpoint_byte_snapshot(path)
                if strict_resume
                else None
            )
        )
        load_source = snapshot if snapshot is not None else path
        loaded = torch.load(
            load_source,
            map_location="cpu",
            weights_only=True if formal_safe_resume else False,
        )
        required_state = None
        if strict_resume:
            required_state = self._preflight_required_exact_resume_checkpoint(
                loaded,
                path=path,
                load_optimizer=load_optimizer,
            )
            if formal_safe_resume:
                infos = self._apply_formal_preloaded_checkpoint(
                    loaded,
                    load_optimizer=load_optimizer,
                    prefix=f"required exact resume checkpoint {path!r}",
                )
            else:
                # Preserve historical task-first/diagnostic compatibility.
                # Only formal exact-lineage ActionBall has the immutable
                # weights-only checkpoint contract.
                snapshot.seek(0)
                infos = super().load(
                    load_source,
                    load_optimizer=load_optimizer,
                    **kwargs,
                )
        else:
            infos = super().load(
                load_source,
                load_optimizer=load_optimizer,
                **kwargs,
            )
        if self.log_dir is None:
            self._loaded_checkpoint_path = resolved_checkpoint_path
            return infos
        state = (
            required_state
            if required_state is not None
            else self._checkpoint_exact_resume_state(loaded)
        )
        if state is not None:
            if (
                not isinstance(state, dict)
                or int(state.get("schema_version", 0)) not in _SUPPORTED_EXACT_RESUME_SCHEMAS
            ):
                raise RuntimeError(
                    "unsupported hope_exact_resume_state schema "
                    f"{state.get('schema_version', None) if isinstance(state, dict) else type(state)}; "
                    f"refusing to guess how to resume: {path}"
                )
            if int(state["schema_version"]) >= 3:
                nested = state.get("environment_resume_state")
                if (
                    not isinstance(nested, dict)
                    or type(nested.get("schema_version")) is not int
                    or nested["schema_version"] not in (3, 4)
                ):
                    raise RuntimeError(
                        "schema-3 exact resume requires a schema-3/4 "
                        "environment_resume_state; refusing a partial command restore"
                    )
            # 一致性铁律:状态包写入时恒有 next_learning_iteration == iter+1。checkpoint 外科
            # 手术(make_hitter_warmstart / warm_start_realsensor 把 iter 归零但整份保留 infos)
            # 会打破它 —— 那时状态是"上一世"的,拿来续课程等于劫持一次刻意的全新热启动。
            # 响亮降级成"老档"语义:主时钟按本档 iter 推算,课程统计从头攒。
            expected_next = int(self.current_learning_iteration) + 1
            if int(state.get("next_learning_iteration", -1)) != expected_next:
                if getattr(self, "require_exact_resume_state", False):
                    raise RuntimeError(
                        "required exact resume checkpoint iteration changed during base load: "
                        f"state next={state.get('next_learning_iteration')!r}, "
                        f"base iter+1={expected_next}"
                    )
                print(
                    "[MotionOnPolicyRunner] WARNING: hope_exact_resume_state is stale "
                    f"(next_learning_iteration={state.get('next_learning_iteration')!r} vs "
                    f"checkpoint iter+1={expected_next}); treating as a legacy warm-start "
                    "checkpoint — curriculum statistics start fresh",
                    flush=True,
                )
                state = None
        if state is None:
            if getattr(self, "require_exact_resume_state", False):
                # Defensive invariant: strict preflight above must either return schema 3 or raise.
                raise RuntimeError(
                    "required exact resume state disappeared before environment restore"
                )
            # 老 checkpoint:课程细节无从恢复,但主时钟按迭代号精确推算(见 _restore_…)。
            # 迭代号本身维持 base 语义(iter=N,会重复一次第 N 个更新)——没有状态包时不敢
            # 替它做 N+1 的决定。
            self._restore_environment_resume_state(
                {"next_learning_iteration": int(self.current_learning_iteration) + 1}
            )
        else:
            self.current_learning_iteration = int(state["next_learning_iteration"])
            self.tot_timesteps = int(state.get("tot_timesteps", self.tot_timesteps))
            self.tot_time = float(state.get("tot_time", self.tot_time))
            if getattr(self.alg, "schedule", None) == "adaptive" and "algorithm_learning_rate" in state:
                # 自适应 KL 调度下 lr 是"续"出来的运行状态:不恢复的话第一次 update 会从
                # YAML 初值重新适应,平白抖一下。固定调度(fixed)不恢复 —— YAML 里改 lr
                # 再续训必须生效;这是 main 共享 load() 与 jiayi 独立 exact 路径的取舍差异。
                self.alg.learning_rate = float(state["algorithm_learning_rate"])
                for param_group in self.alg.optimizer.param_groups:
                    param_group["lr"] = self.alg.learning_rate
            self._restore_environment_resume_state(state)
            if int(state["schema_version"]) >= 3:
                # Restoring RNG any earlier lets command-state loading consume the resumed stream;
                # restoring it any later lets env.reset() sample from the constructor seed. This is
                # therefore deliberately the final operation before the first resumed reset.
                self._restore_exact_rng_state(state)
        # The simulator itself is not serialized. Historical/task-first paths retain their reset
        # here. Action-ball has a stronger protocol: load_exact_resume_state_dict() is a pure data
        # restore and load() must neither sample nor write the simulator. Defer its first true reset
        # to the learn boundary, after the exact RNG and all broker/pool/curriculum state are live.
        if self._strict_exact_resume_target_mode() == "action_ball":
            self._action_ball_resume_reset_pending = True
            if strict_resume:
                self._exact_resume_loaded_source_iteration = int(
                    loaded["iter"]
                )
                self._exact_resume_loaded_source_telemetry = (
                    self._exact_resume_source_telemetry(required_state)
                )
                self._exact_resume_roundtrip_pending = True
                source_environment = required_state.get(
                    "environment_resume_state"
                )
                if not isinstance(source_environment, Mapping):
                    raise RuntimeError(
                        "strict ActionBall source lacks environment state"
                    )
                self._exact_resume_loaded_source_common_step_counter = (
                    source_environment.get("common_step_counter")
                )
                self._install_exact_resume_live_state_baseline(
                    loaded_checkpoint=loaded,
                    source_exact_state=required_state,
                )
        else:
            self.env.reset()
        self._loaded_checkpoint_path = resolved_checkpoint_path
        return infos

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Defer SIGINT/SIGTERM to an iteration boundary and save that completed iteration.

        人话:Ctrl-C 不再当场掐死训练(半个迭代的档没有存在价值),而是登记一个"想停"标记,
        等当前 PPO 迭代完整跑完、log() 落账之后存一份带精确续训状态的档再退出。
        第二次 Ctrl-C/SIGTERM = "别等了,立刻死"(2026-07-25):迭代若卡住(env 步进不返回、
        logger 重试死循环),旧版会把后续信号全部静默吞掉,操作员只能另开 shell 找 PID 发
        SIGKILL。现在第二次信号恢复 OS 默认处置并重新对自己投递——不存档,直接终止。
        (老实说明:若主线程死死卡在 native Isaac/CUDA 调用里,Python 层任何 handler 都
        不会被执行,第一次信号同样没反应,那种情况仍然只有 SIGKILL 能救。)
        """
        self._install_action_ball_211_wait_normalizer_masks()
        r10_enabled = (
            getattr(self, "_action_ball_r10_checkpoint_adapter", None)
            is not None
        )
        full_mdp_enabled = (
            getattr(self, "_action_ball_full_mdp_runtime_owner", None)
            is not None
        )
        full_mdp_run_mode = getattr(
            self, "_action_ball_full_mdp_run_mode", None
        )
        expected_r10 = (
            full_mdp_enabled
            and full_mdp_run_mode == _ACTION_BALL_FULL_MDP_FORMAL_MODE
        )
        if r10_enabled != expected_r10 or (
            full_mdp_enabled
            and full_mdp_run_mode
            not in {
                _ACTION_BALL_FULL_MDP_FORMAL_MODE,
                _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE,
            }
        ):
            raise RuntimeError(
                "fresh full-MDP runner lost its exact mode/runtime-owner/R10 "
                "construction binding"
            )
        if full_mdp_enabled:
            self._action_ball_full_mdp_require_healthy()
        if r10_enabled:
            self._action_ball_r10_require_healthy()
            if getattr(self, "_action_ball_resume_reset_pending", False):
                raise RuntimeError(
                    "R10 runner adapter refuses the legacy reset-on-resume path"
                )
        action_ball_update_profile_requested = False
        raw_update_profile = os.environ.get(
            "HOPE_ACTION_BALL_UPDATE_PROFILE"
        )
        if raw_update_profile is not None:
            # Keep the default path import-free.  The profiler installs
            # instance wrappers only for the explicit diagnostic opt-in.
            from whole_body_tracking.utils.action_ball_update_profiler import (
                parse_action_ball_update_profile_request,
            )

            action_ball_update_profile_requested = (
                parse_action_ball_update_profile_request(os.environ)
            )
        if action_ball_update_profile_requested:
            # Reject every unsupported presentation/runtime shape before an
            # exact-resume reset or any other simulator mutation.  Upstream
            # may skip ``log()`` on disable_logs/non-primary ranks, which
            # would silently accumulate several updates into one profile row.
            if not self._action_ball_diagnostic_unauthorized():
                raise RuntimeError(
                    "HOPE_ACTION_BALL_UPDATE_PROFILE=1 is allowed only for "
                    "diagnostic ActionBall; formal profiling is fail-closed"
                )
            if self.disable_logs is not False:
                raise RuntimeError(
                    "HOPE_ACTION_BALL_UPDATE_PROFILE=1 requires "
                    "disable_logs=False so every update emits exactly once"
                )
            if self._joint_safety_rank() != 0:
                raise RuntimeError(
                    "HOPE_ACTION_BALL_UPDATE_PROFILE=1 requires the "
                    "primary runner rank 0"
                )
        reward_ppo_economy_requested = (
            self._reward_ppo_economy_gate_requested()
        )
        prelong_expected_recipe_sha256 = None
        prelong_preregistered_recipe = None
        if (
            _PRELONG_SEMANTICS_ENABLE_ENV in os.environ
            or _PRELONG_SEMANTICS_RECIPE_SHA_ENV in os.environ
        ):
            from whole_body_tracking.utils.action_ball_prelong_semantics import (
                parse_prelong_runtime_request,
            )

            prelong_expected_recipe_sha256 = parse_prelong_runtime_request(
                os.environ,
                reward_ppo_economy_requested=reward_ppo_economy_requested,
            )
            if prelong_expected_recipe_sha256 is not None:
                prelong_preregistered_recipe = (
                    self._prelong_preregistered_reward_recipe(
                        prelong_expected_recipe_sha256
                    )
                )
        normalizer_binding = self._validate_training_normalizers()
        # This stdout receipt is inside the run's durable JSON log boundary.
        # The formal bootstrap receipt itself is minted later by train.py and
        # has an immutable schema, so the runner records its live ABI here
        # without weakening or rewriting that authority document.
        self._emit_rsl_rl_runtime_abi(
            normalizer_binding=normalizer_binding
        )
        # Entering learn consumes the verifier-only no-step window even if a
        # subsequent reset or frozen-evaluation reconciliation fails.
        self._exact_resume_roundtrip_pending = False
        if getattr(self, "_action_ball_resume_reset_pending", False):
            # An exact checkpoint can contain a published frozen-eval request.
            # Resolve that policy fence before the first simulator mutation;
            # otherwise one resumed rollout could update normalizers or serve
            # births from a domain that the old checkpoint is still judging.
            resume_step = max(
                0, int(self.current_learning_iteration) - 1
            )
            reset_by_frozen_eval = (
                self._service_action_ball_frozen_evaluation(resume_step)
            )
            # Clear first so a reset exception cannot be retried implicitly
            # with a partly consumed tape.
            self._action_ball_resume_reset_pending = False
            if not reset_by_frozen_eval:
                self.env.reset()
        # Fresh training has already reset during environment construction; exact resume reaches
        # this point only after its deferred true reset above.  In either case every env must now
        # expose one sampled 0..N control-step lag before the first rollout/update.
        self._emit_control_step_action_delay_runtime_receipt()

        reward_activation_ledger = None
        reward_ppo_economy_ledger = None
        prelong_semantics_ledger = None
        reward_activation_json = None
        reward_ledger_is_action_bound = False
        original_env_step = None
        reward_activation_task_kind = self._effective_reward_activation_task_kind()
        if reward_activation_task_kind is not None:
            # Import lazily so evaluator/legacy/diagnostic runners retain
            # their dependency-light construction path. Formal ActionBall
            # and UpperSafe fail before their first rollout if the exact
            # RewardManager cache contract is unavailable.
            from whole_body_tracking.utils.effective_reward_recipe import (
                ActionBoundRewardEvidenceLedger,
                EffectiveRewardActivationLedger,
                canonical_effective_reward_activation_json,
            )

            unwrapped_env = getattr(self.env, "unwrapped", self.env)
            if reward_activation_task_kind == "action_ball":
                binding = self._bind_action_ball_reward_evidence()
                reward_activation_ledger = (
                    ActionBoundRewardEvidenceLedger(
                        unwrapped_env,
                        expected_environment_step_count=(
                            self.num_steps_per_env
                        ),
                        action_contract=binding["action_contract"],
                        action_identity_provider=binding[
                            "identity_provider"
                        ],
                        termination_snapshot_provider=binding[
                            "termination_provider"
                        ],
                    )
                )
                reward_ledger_is_action_bound = True
            else:
                reward_activation_ledger = (
                    EffectiveRewardActivationLedger(
                        unwrapped_env,
                        task_kind=reward_activation_task_kind,
                        expected_environment_step_count=(
                            self.num_steps_per_env
                        ),
                    )
                )
            reward_activation_json = canonical_effective_reward_activation_json
            original_env_step = getattr(self.env, "step", None)
            if not callable(original_env_step):
                raise RuntimeError(
                    f"{reward_activation_task_kind} runtime reward activation "
                    "requires a callable env.step()"
                )
        if reward_ppo_economy_requested:
            # This is a diagnostic-only numerical observer.  It reuses the
            # verified RewardManager cache adapter but neither persists nor
            # emits the formal activation/episode/action evidence receipts.
            # The integrated gate receipt remains explicitly non-promotable.
            from whole_body_tracking.utils.effective_reward_recipe import (
                EffectiveRewardActivationLedger,
            )

            if reward_activation_ledger is not None:
                raise RuntimeError(
                    "reward/PPO economy diagnostic cannot share a formal Reward ledger"
                )
            unwrapped_env = getattr(self.env, "unwrapped", self.env)
            reward_ppo_economy_ledger = EffectiveRewardActivationLedger(
                unwrapped_env,
                task_kind="action_ball",
                expected_environment_step_count=self.num_steps_per_env,
                batch_host_validation_at_update=True,
            )
            if prelong_expected_recipe_sha256 is not None:
                from whole_body_tracking.utils.action_ball_prelong_semantics import (
                    ActionBallPrelongSemanticsLedger,
                )

                prelong_semantics_ledger = ActionBallPrelongSemanticsLedger(
                    unwrapped_env,
                    preregistered_effective_reward_recipe=(
                        prelong_preregistered_recipe
                    ),
                )
            original_env_step = getattr(self.env, "step", None)
            if not callable(original_env_step):
                raise RuntimeError(
                    "reward/PPO economy diagnostic requires callable env.step()"
                )
        # Diagnostic ActionBall intentionally has no formal Reward activation
        # or promotion authority.  Its joint action keeps the same clamp,
        # physical readbacks, DoneTerms and update-scale counters, but drains a
        # compact device aggregate instead of materializing formal per-step
        # identity transcripts and durable receipt files.
        diagnostic_joint_safety = (
            self._diagnostic_joint_safety_compact_update_evidence()
        )
        joint_safety_action_term = self._bind_joint_safety_action_term(
            required=(
                reward_activation_task_kind is not None
                or diagnostic_joint_safety
            )
        )

        r10_tracker = None
        original_r10_compute_returns = None
        if r10_enabled:
            if (
                getattr(self, "training_type", None) != "rl"
                or getattr(self, "privileged_obs_type", None) != "critic"
            ):
                raise RuntimeError("R10 runner adapter requires PPO with critic observations")
            storage = getattr(self.alg, "storage", None)
            if type(getattr(storage, "step", None)) is not int or storage.step != 0:
                raise RuntimeError("R10 learn entry requires empty rollout storage")
            if not self._action_ball_r10_transition_empty(
                getattr(self.alg, "transition", None)
            ):
                raise RuntimeError("R10 learn entry requires an empty transition")
            original_r10_compute_returns = getattr(
                self.alg, "compute_returns", None
            )
            if not callable(original_r10_compute_returns):
                raise RuntimeError("R10 PPO lacks compute_returns()")
            r10_tracker = {
                "storage_object": storage,
                "environment_step_in_flight": False,
                "completed_environment_steps": 0,
                "environment_steps_at_boundary": 0,
                "actor_normalizer_calls": 0,
                "critic_normalizer_calls": 0,
                "actor_calls_at_boundary": None,
                "critic_calls_at_boundary": None,
                "last_normalized_actor": None,
                "last_normalized_critic": None,
                "last_compute_returns_critic": None,
                "gae_in_flight": False,
                "gae_complete": False,
                "optimizer_in_flight": False,
            }

        original_update = getattr(self.alg, "update", None)
        if not callable(original_update):
            raise RuntimeError(
                "MotionOnPolicyRunner requires a callable alg.update() for exact rollout boundaries"
            )
        if getattr(self, "_rollout_update_wrapper_active", False):
            raise RuntimeError("MotionOnPolicyRunner.learn() cannot be entered recursively")
        pending_signal = None
        previous_handlers = {}

        def request_stop(signum, _frame):
            nonlocal pending_signal
            if pending_signal is None:
                pending_signal = signum
                print(
                    "\n[MotionOnPolicyRunner] interrupt received; finishing the current PPO "
                    "iteration before saving",
                    flush=True,
                )
            else:
                # 第二次信号:升级为立即终止。先恢复该信号的 OS 默认动作再 os.kill 自投——
                # 默认动作由内核执行,不依赖解释器回到字节码循环,比 raise KeyboardInterrupt
                # 硬(后者在 finally/except 链里还可能被吞)。代价:不存档(与提示语一致)。
                print(
                    "\n[MotionOnPolicyRunner] second interrupt — exiting immediately, "
                    "NO checkpoint",
                    flush=True,
                )
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)
        except ValueError:
            # signal.signal is restricted to the main thread. Training normally runs there; retain
            # upstream behavior if an embedding invokes learn() from another thread.
            previous_handlers.clear()

        self._boundary_stop_requested = lambda: pending_signal is not None
        # Upstream RSL-RL calls ``log()`` only on rank 0.  Curriculum advancement is environment
        # state, not presentation, so attach it to the algorithm's one update call per PPO loop.
        # This executes on every rank even when ``disable_logs`` is true and preserves the upstream
        # loop rather than copying a version-sensitive implementation here.
        next_rollout_step = int(self.current_learning_iteration)

        def r10_compute_returns(last_critic_observations, *args, **kwargs):
            if r10_tracker["gae_in_flight"] or r10_tracker["gae_complete"]:
                raise RuntimeError("R10 compute_returns phase re-entered")
            if not torch.is_tensor(last_critic_observations):
                raise RuntimeError("R10 compute_returns critic frontier is not a tensor")
            r10_tracker["gae_in_flight"] = True
            r10_tracker["last_compute_returns_critic"] = (
                last_critic_observations.detach().clone()
            )
            try:
                result = original_r10_compute_returns(
                    last_critic_observations, *args, **kwargs
                )
            except BaseException:
                r10_tracker["last_compute_returns_critic"] = None
                raise
            else:
                r10_tracker["gae_complete"] = True
                return result
            finally:
                r10_tracker["gae_in_flight"] = False

        def call_original_update(*args, **kwargs):
            if r10_enabled:
                if r10_tracker["optimizer_in_flight"]:
                    raise RuntimeError("R10 PPO optimizer phase re-entered")
                r10_tracker["optimizer_in_flight"] = True
            try:
                result = original_update(*args, **kwargs)
            finally:
                if r10_enabled:
                    r10_tracker["optimizer_in_flight"] = False
            if full_mdp_enabled:
                prepared = getattr(
                    self, "_action_ball_full_mdp_boundary_prepared", None
                )
                if prepared is None:
                    raise RuntimeError(
                        "fresh full-MDP optimizer returned without an active boundary"
                    )
                self._action_ball_full_mdp_mark_optimizer_callback(
                    prepared,
                    update_index=next_rollout_step,
                )
                active = getattr(
                    self,
                    "_action_ball_full_mdp_active_drain_chronology",
                    None,
                )
                if (
                    type(active)
                    is not _ActionBallFullMdpRunnerDrainChronology
                    or active.global_receipt is not prepared
                    or active.update_index != next_rollout_step
                    or active.optimizer_returned is not False
                ):
                    raise RuntimeError(
                        "fresh full-MDP optimizer return chronology differs"
                    )
                self._action_ball_full_mdp_active_drain_chronology = (
                    active._replace(optimizer_returned=True)
                )
            return result

        def update_with_rollout_boundary(*args, **kwargs):
            nonlocal next_rollout_step
            full_mdp_prepared = None
            full_mdp_wal_path = None
            full_mdp_pending_summary = None
            full_mdp_ack_json = None
            full_mdp_pending_span = None
            prepared_joint_safety = None
            prepared_reward_evidence = None
            reward_artifact = None
            economy_activation = None
            economy_rollout = None
            prepared_prelong_semantics = None
            if r10_enabled:
                storage = getattr(self.alg, "storage", None)
                if (
                    storage is not r10_tracker["storage_object"]
                    or type(getattr(storage, "step", None)) is not int
                    or storage.step != int(self.num_steps_per_env)
                    or r10_tracker["environment_step_in_flight"] is not False
                    or r10_tracker["gae_in_flight"] is not False
                    or r10_tracker["gae_complete"] is not True
                    or r10_tracker["actor_calls_at_boundary"] is None
                    or r10_tracker["critic_calls_at_boundary"] is None
                    or (
                        r10_tracker["completed_environment_steps"]
                        - r10_tracker["environment_steps_at_boundary"]
                    )
                    != int(self.num_steps_per_env)
                    or (
                        r10_tracker["actor_normalizer_calls"]
                        - r10_tracker["actor_calls_at_boundary"]
                    )
                    != int(self.num_steps_per_env)
                    or (
                        r10_tracker["critic_normalizer_calls"]
                        - r10_tracker["critic_calls_at_boundary"]
                    )
                    != int(self.num_steps_per_env)
                    or not self._action_ball_r10_transition_empty(
                        getattr(self.alg, "transition", None)
                    )
                ):
                    raise RuntimeError("R10 pre-update rollout/GAE boundary differs")
            if joint_safety_action_term is not None:
                # Freeze and validate before PPO may consume the rollout. Formal
                # tasks durably publish the full identity-bound receipt;
                # diagnostic ActionBall validates one device aggregate and
                # remains explicitly non-promotable. The action term keeps
                # either generation live until post-optimizer acknowledgement.
                if diagnostic_joint_safety:
                    prepared_joint_safety = (
                        self._prepare_diagnostic_joint_safety_update(
                            next_rollout_step,
                            expected_action_term=joint_safety_action_term,
                        )
                    )
                else:
                    prepared_joint_safety = self._prepare_joint_safety_update(
                        next_rollout_step,
                        expected_action_term=joint_safety_action_term,
                    )
            if reward_activation_ledger is not None:
                reward_rank = self._joint_safety_rank()
                self._preflight_reward_evidence_update_paths(
                    step=next_rollout_step, rank=reward_rank
                )
                if reward_ledger_is_action_bound:
                    if prepared_joint_safety is None:
                        raise RuntimeError(
                            "action-bound Reward evidence requires the exact "
                            "joint-safety policy-step sequence"
                        )
                    prepared_reward_evidence = (
                        reward_activation_ledger.prepare_update(
                            next_rollout_step,
                            joint_first_policy_step_sequence=(
                                prepared_joint_safety["validated"][
                                    "first_policy_step_sequence"
                                ]
                            ),
                        )
                    )
                else:
                    prepared_reward_evidence = {
                        "ppo_update": next_rollout_step,
                        "activation": (
                            reward_activation_ledger.prepare_update(
                                next_rollout_step
                            )
                        ),
                        "per_action": None,
                        "safety": None,
                        "status": (
                            "frozen_validated_before_optimizer"
                        ),
                    }
                if reward_ledger_is_action_bound:
                    self._require_action_ball_conservation_pass(
                        prepared_reward_evidence,
                        step=next_rollout_step,
                    )
                reward_artifact = self._persist_reward_evidence_update(
                    prepared_reward_evidence,
                    step=next_rollout_step,
                    rank=reward_rank,
                    task_kind=reward_activation_task_kind,
                )
            if reward_ledger_is_action_bound:
                # Keep the optimizer call immediately downstream of the
                # immutable public PASS receipt.  Missing, mutated, or
                # fail-closed conservation evidence is a hard stop.
                self._require_action_ball_conservation_pass(
                    prepared_reward_evidence,
                    step=next_rollout_step,
                )
            if full_mdp_enabled:
                if args or kwargs:
                    raise RuntimeError(
                        "fresh full-MDP requires the zero-argument PPO update ABI"
                    )
                if getattr(
                    self, "_action_ball_full_mdp_boundary_active", False
                ):
                    raise RuntimeError(
                        "fresh full-MDP PPO boundary was re-entered"
                    )
                self._action_ball_full_mdp_require_healthy()
                self._action_ball_full_mdp_boundary_active = True
                try:
                    if self._action_ball_full_mdp_run_mode == (
                        _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
                    ):
                        full_mdp_wal_path = (
                            self._action_ball_full_mdp_preflight_durable_wal()
                        )
                    full_mdp_prepared = (
                        self._action_ball_full_mdp_prepare_callback(
                            update_index=next_rollout_step,
                            completed_environment_steps=(
                                (int(next_rollout_step) + 1)
                                * int(self.env.num_envs)
                                * int(self.num_steps_per_env)
                            ),
                        )
                    )
                    if full_mdp_prepared is None:
                        raise RuntimeError(
                            "fresh full-MDP prepare returned no opaque boundary"
                        )
                    self._action_ball_full_mdp_boundary_prepared = (
                        full_mdp_prepared
                    )
                    completed_rows = (
                        (int(next_rollout_step) + 1)
                        * int(self.env.num_envs)
                        * int(self.num_steps_per_env)
                    )
                    self._action_ball_full_mdp_active_drain_chronology = (
                        _ActionBallFullMdpRunnerDrainChronology(
                            global_receipt=full_mdp_prepared,
                            update_index=next_rollout_step,
                            completed_environment_steps=completed_rows,
                            optimizer_returned=False,
                            post_update_acknowledged=False,
                            operation_sequence=None,
                            drain_sequence=None,
                            projection=None,
                            telemetry_emitted=False,
                        )
                    )
                except BaseException as exc:
                    reason = (
                        "fresh full-MDP pre-optimizer drain failed; retry is "
                        "forbidden: "
                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                    )
                    self._action_ball_full_mdp_poison_optimizer_boundary(
                        full_mdp_prepared,
                        update_index=next_rollout_step,
                        reason=reason,
                    )
                    raise RuntimeError(reason) from exc
            try:
                if reward_ppo_economy_ledger is not None:
                    if args or kwargs:
                        raise RuntimeError(
                            "reward/PPO economy requires the zero-argument PPO update ABI"
                        )
                    economy_activation = reward_ppo_economy_ledger.prepare_update(
                        next_rollout_step
                    )
                    economy_rollout = self._prepare_reward_ppo_economy_rollout(
                        activation=economy_activation,
                        ppo_update=next_rollout_step,
                    )
                    if prelong_semantics_ledger is not None:
                        prepared_prelong_semantics = (
                            prelong_semantics_ledger.prepare_update(
                                next_rollout_step
                            )
                        )
                    if r10_enabled or full_mdp_enabled:
                        result, economy_gradient = (
                            self._run_reward_ppo_economy_optimizer(
                                call_original_update
                            )
                        )
                    else:
                        result, economy_gradient = (
                            self._run_reward_ppo_economy_optimizer(original_update)
                        )
                else:
                    if r10_enabled or full_mdp_enabled:
                        result = call_original_update(*args, **kwargs)
                    else:
                        result = original_update(*args, **kwargs)
                if full_mdp_enabled and self._action_ball_full_mdp_run_mode == (
                    _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
                ):
                    active = getattr(
                        self,
                        "_action_ball_full_mdp_active_drain_chronology",
                        None,
                    )
                    prepare_summary = getattr(
                        self,
                        "_action_ball_full_mdp_prepare_summary_callback",
                        None,
                    )
                    if (
                        type(active)
                        is not _ActionBallFullMdpRunnerDrainChronology
                        or active.global_receipt is not full_mdp_prepared
                        or active.optimizer_returned is not True
                        or active.post_update_acknowledged is not False
                        or not callable(prepare_summary)
                        or type(full_mdp_wal_path) is not pathlib.PosixPath
                    ):
                        raise RuntimeError(
                            "fresh full-MDP pending-summary chronology differs"
                        )
                    full_mdp_pending_summary = prepare_summary(
                        full_mdp_prepared,
                        update_index=next_rollout_step,
                    )
                    pending_record = (
                        self._action_ball_full_mdp_prepare_epoch_ack_record(
                            full_mdp_pending_summary,
                            update_index=next_rollout_step,
                            completed_environment_steps=(
                                active.completed_environment_steps
                            ),
                        )
                    )
                    full_mdp_ack_json, full_mdp_wal_line = (
                        self._action_ball_full_mdp_encode_durable_pending_ack(
                            pending_record,
                            update_index=next_rollout_step,
                            completed_environment_steps=(
                                active.completed_environment_steps
                            ),
                        )
                    )
                    full_mdp_pending_span = (
                        self._action_ball_full_mdp_append_durable_wal(
                        full_mdp_wal_path,
                        full_mdp_wal_line,
                        )
                    )
                    self._action_ball_full_mdp_ack_callback(
                        full_mdp_prepared,
                        full_mdp_pending_summary,
                        update_index=next_rollout_step,
                    )
                    epoch_ack_line = (
                        _action_ball_full_mdp_durable_wal_module().encode_epoch_ack(
                            pending_line=full_mdp_wal_line,
                            pending_byte_start=full_mdp_pending_span[0],
                            pending_byte_end=full_mdp_pending_span[1],
                        )
                    )
                    full_mdp_epoch_ack_span = self._action_ball_full_mdp_append_durable_wal(
                        full_mdp_wal_path, epoch_ack_line
                    )
                    self._action_ball_full_mdp_durable_ack_callback(
                        full_mdp_pending_summary,
                        update_index=next_rollout_step,
                        segment_id=self._action_ball_full_mdp_durable_wal_segment_id,
                        rank=self._joint_safety_rank(),
                        pending_byte_start=full_mdp_pending_span[0],
                        pending_byte_end=full_mdp_pending_span[1],
                        ack_byte_start=full_mdp_epoch_ack_span[0],
                        ack_byte_end=full_mdp_epoch_ack_span[1],
                    )
                    self._action_ball_full_mdp_consume_epoch_ack_summary(
                        full_mdp_pending_summary,
                        update_index=next_rollout_step,
                        completed_environment_steps=(
                            active.completed_environment_steps
                        ),
                        canonical_ack_json=full_mdp_ack_json,
                    )
                    self._action_ball_full_mdp_boundary_prepared = None
                    self._action_ball_full_mdp_boundary_active = False
                # Read the trainable parameter after optimizer.step().  RSL-RL's
                # cached action distribution was constructed before that step and
                # can otherwise hide a newly negative scalar std for one rollout.
                policy_std_record = self._emit_policy_std_update(
                    ppo_update=next_rollout_step
                )
                if reward_ppo_economy_ledger is not None:
                    if policy_std_record is None:
                        raise RuntimeError(
                            "reward/PPO economy lacks post-update policy std telemetry"
                        )
                    economy_ppo = self._reward_ppo_economy_post_update(result)
                    reward_ppo_economy_ledger.acknowledge_update(
                        economy_activation
                    )
                    self._emit_reward_ppo_economy_update(
                        ppo_update=next_rollout_step,
                        rollout=economy_rollout,
                        ppo=economy_ppo,
                        gradient=economy_gradient,
                        policy=policy_std_record,
                    )
                reward_optimizer_commit = None
                if prepared_reward_evidence is not None:
                    reward_optimizer_commit = (
                        self._persist_reward_evidence_optimizer_commit(
                            step=next_rollout_step,
                            rank=reward_rank,
                            artifact=reward_artifact,
                        )
                    )
                if prepared_joint_safety is not None:
                    if diagnostic_joint_safety:
                        self._commit_diagnostic_joint_safety_update(
                            prepared_joint_safety
                        )
                    else:
                        self._commit_joint_safety_update(prepared_joint_safety)
                if prepared_reward_evidence is not None:
                    if reward_ledger_is_action_bound:
                        reward_activation_ledger.acknowledge_update(
                            prepared_reward_evidence
                        )
                    else:
                        reward_activation_ledger.acknowledge_update(
                            prepared_reward_evidence["activation"]
                        )
                    self._emit_reward_evidence_update(
                        prepared_reward_evidence,
                        artifact=reward_artifact,
                        optimizer_commit=reward_optimizer_commit,
                        encoder=reward_activation_json,
                    )
                self._notify_command_terms_rollout_end(next_rollout_step)
                if full_mdp_enabled and self._action_ball_full_mdp_run_mode != (
                    _ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
                ):
                    acknowledged = self._action_ball_full_mdp_ack_callback(
                        full_mdp_prepared,
                        update_index=next_rollout_step,
                    )
                    active = getattr(
                        self,
                        "_action_ball_full_mdp_active_drain_chronology",
                        None,
                    )
                    if (
                        type(active)
                        is not _ActionBallFullMdpRunnerDrainChronology
                    ):
                        raise RuntimeError(
                            "fresh full-MDP ACK lost runner chronology"
                        )
                    self._action_ball_full_mdp_capture_acknowledged_projection(
                        receipt=full_mdp_prepared,
                        update_index=next_rollout_step,
                        completed_environment_steps=(
                            active.completed_environment_steps
                        ),
                    )
                    self._action_ball_full_mdp_boundary_prepared = None
                    self._action_ball_full_mdp_boundary_active = False
                if r10_enabled:
                    self._action_ball_r10_capture_post_update_handoff(
                        completed_update_index=next_rollout_step,
                        tracker=r10_tracker,
                    )
                # Frozen evaluation is deliberately sequenced after the Reward
                # activation and joint-safety two-phase commits.  It may persist
                # control checkpoints or perform a fenced global reset, but it can
                # never change which evidence the just-completed PPO update used.
                # It remains inside the same full-MDP failure domain: after an
                # optimizer acknowledgement there is no rollback, so every later
                # boundary-service failure must poison the owner and forbid retry.
                self._service_action_ball_frozen_evaluation(
                    next_rollout_step
                )
                if prelong_semantics_ledger is not None:
                    # Emit only after every required post-optimizer commit and
                    # boundary service succeeds.  Flush before destructive
                    # acknowledgement so a BrokenPipe cannot silently consume the
                    # sole terminal marker.
                    prelong_ack = (
                        prelong_semantics_ledger.prepare_acknowledgement(
                            prepared_prelong_semantics
                        )
                    )
                    prelong_marker_line = prelong_ack.marker_line
                    print(prelong_marker_line, flush=True)
                    acknowledged_line = prelong_ack.consume()
                    if acknowledged_line != prelong_marker_line:
                        raise RuntimeError(
                            "pre-long marker changed during acknowledgement"
                        )
            except BaseException as exc:
                if full_mdp_enabled:
                    reason = (
                        "fresh full-MDP optimizer/post-boundary failed; retry "
                        "is forbidden and checkpoint is forbidden: "
                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                    )
                    self._action_ball_full_mdp_poison_optimizer_boundary(
                        full_mdp_prepared,
                        update_index=next_rollout_step,
                        reason=reason,
                    )
                raise
            next_rollout_step += 1
            return result

        def step_with_reward_activation(*args, **kwargs):
            active_reward_ledger = (
                reward_activation_ledger
                if reward_activation_ledger is not None
                else reward_ppo_economy_ledger
            )
            step_token = (
                active_reward_ledger.begin_environment_step()
                if reward_ledger_is_action_bound
                else None
            )
            prelong_step_token = (
                prelong_semantics_ledger.begin_environment_step()
                if prelong_semantics_ledger is not None
                else None
            )
            try:
                result = original_env_step(*args, **kwargs)
                if reward_ledger_is_action_bound:
                    active_reward_ledger.observe_after_environment_step(
                        step_token
                    )
                else:
                    active_reward_ledger.observe_after_environment_step()
                if prelong_semantics_ledger is not None:
                    prelong_semantics_ledger.observe_after_environment_step(
                        prelong_step_token
                    )
            except BaseException:
                if reward_ledger_is_action_bound:
                    active_reward_ledger.abort_environment_step()
                if prelong_semantics_ledger is not None:
                    prelong_semantics_ledger.abort_environment_step(
                        prelong_step_token
                    )
                raise
            return result

        r10_original_env_step = None

        def step_with_r10_boundary_tracking(*args, **kwargs):
            if r10_tracker["environment_step_in_flight"]:
                raise RuntimeError("R10 environment step re-entered")
            r10_tracker["environment_step_in_flight"] = True
            try:
                result = r10_original_env_step(*args, **kwargs)
            except BaseException:
                raise
            else:
                r10_tracker["completed_environment_steps"] += 1
                return result
            finally:
                r10_tracker["environment_step_in_flight"] = False

        def capture_r10_normalized_actor(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 2:
                raise RuntimeError("R10 actor normalizer output is invalid")
            r10_tracker["actor_normalizer_calls"] += 1
            r10_tracker["last_normalized_actor"] = output.detach().clone()

        def capture_r10_normalized_critic(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 2:
                raise RuntimeError("R10 critic normalizer output is invalid")
            r10_tracker["critic_normalizer_calls"] += 1
            r10_tracker["last_normalized_critic"] = output.detach().clone()

        self._rollout_update_wrapper_active = True
        reward_activation_step_wrapper_active = False
        r10_env_step_wrapper_active = False
        r10_compute_returns_wrapper_active = False
        r10_normalizer_hook_handles = []
        initial_observation_wrapper_active = False
        original_get_observations = None
        action_ball_update_profiler = None
        learn_primary_error = None
        self._learn_cleanup_secondary_failures = ()
        try:
            if action_ball_update_profile_requested:
                from whole_body_tracking.utils.action_ball_update_profiler import (
                    install_diagnostic_action_ball_update_profiler,
                )

                action_ball_update_profiler = (
                    install_diagnostic_action_ball_update_profiler(
                        self.env,
                        diagnostic_fast_path=diagnostic_joint_safety,
                        emit_line=lambda line: print(line, flush=True),
                    )
                )
                self._action_ball_update_profiler = (
                    action_ball_update_profiler
                )
            if (
                reward_activation_ledger is not None
                or reward_ppo_economy_ledger is not None
            ):
                self.env.step = step_with_reward_activation
                reward_activation_step_wrapper_active = True
            if r10_enabled:
                if init_at_random_ep_len:
                    raise RuntimeError(
                        "R10 exact continuation forbids random initial episode lengths"
                    )
                for role, hook in (
                    ("actor", capture_r10_normalized_actor),
                    ("critic", capture_r10_normalized_critic),
                ):
                    _attribute, normalizer, _aliases = (
                        self._resolve_runtime_normalizer(role)
                    )
                    registrar = getattr(normalizer, "register_forward_hook", None)
                    if not callable(registrar):
                        raise RuntimeError(
                            f"R10 cannot observe the {role} normalizer frontier"
                        )
                    r10_normalizer_hook_handles.append(registrar(hook))
                r10_original_env_step = getattr(self.env, "step", None)
                if not callable(r10_original_env_step):
                    raise RuntimeError("R10 runner requires callable env.step()")
                self.env.step = step_with_r10_boundary_tracking
                r10_env_step_wrapper_active = True
                self.alg.compute_returns = r10_compute_returns
                r10_compute_returns_wrapper_active = True
            r10_initial_frontier_required = (
                self._action_ball_211_wait_mask_required()
                or (
                    r10_enabled
                    and getattr(
                        getattr(
                            getattr(self.env, "unwrapped", self.env),
                            "cfg",
                            None,
                        ),
                        "obs_mode",
                        None,
                    )
                    == "action_ball_full_mdp"
                )
            )
            if r10_initial_frontier_required:
                original_get_observations = getattr(
                    self.env, "get_observations", None
                )
                if not callable(original_get_observations):
                    raise RuntimeError(
                        "fresh ActionBall211 requires env.get_observations()"
                    )

                if r10_enabled:

                    def get_normalized_initial_observations():
                        if r10_tracker["actor_calls_at_boundary"] is not None:
                            raise RuntimeError(
                                "R10 upstream requested initial observations more than once"
                            )
                        restored = self._action_ball_r10_restored_initial_frontier
                        if type(restored) is dict:
                            if restored.get("consumed") is not False:
                                raise RuntimeError(
                                    "R10 restored normalized frontier was already consumed"
                                )
                            restored["consumed"] = True
                            result = (
                                restored["actor"].detach().clone(),
                                {
                                    "observations": {
                                        "critic": restored["critic"].detach().clone(),
                                    }
                                },
                            )
                        else:
                            if self._action_ball_211_wait_mask_required():
                                result = self._normalize_action_ball_211_initial_observations(
                                    original_get_observations()
                                )
                            else:
                                raw_actor, raw_extras = original_get_observations()
                                raw_critic = raw_extras["observations"]["critic"]
                                result = (
                                    self.obs_normalizer(
                                        raw_actor.to(self.device)
                                    ),
                                    {
                                        "observations": {
                                            "critic": self.privileged_obs_normalizer(
                                                raw_critic.to(self.device)
                                            )
                                        }
                                    },
                                )
                        r10_tracker["actor_calls_at_boundary"] = r10_tracker[
                            "actor_normalizer_calls"
                        ]
                        r10_tracker["critic_calls_at_boundary"] = r10_tracker[
                            "critic_normalizer_calls"
                        ]
                        return result

                else:

                    def get_normalized_initial_observations():
                        return self._normalize_action_ball_211_initial_observations(
                            original_get_observations()
                        )

                # Upstream calls this getter exactly at the rollout initial
                # boundary.  Subsequent actor/critic observations flow through
                # the same hooked normalizer modules after env.step(), and the
                # final normalized critic is reused for compute_returns().
                self.env.get_observations = get_normalized_initial_observations
                initial_observation_wrapper_active = True
            self.alg.update = update_with_rollout_boundary
            super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        except BaseException as exc:
            learn_primary_error = exc
            if r10_enabled:
                self._action_ball_r10_poison(
                    "R10 learn boundary failed; exact retry is forbidden: "
                    + f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
                )
            raise
        finally:
            cleanup_failures = []

            def cleanup(label, operation):
                try:
                    operation()
                except BaseException as cleanup_exc:
                    cleanup_failures.append((label, cleanup_exc))

            cleanup(
                "restore_alg_update",
                lambda: setattr(self.alg, "update", original_update),
            )
            if r10_compute_returns_wrapper_active:
                cleanup(
                    "restore_alg_compute_returns",
                    lambda: setattr(
                        self.alg,
                        "compute_returns",
                        original_r10_compute_returns,
                    ),
                )
            if initial_observation_wrapper_active:
                cleanup(
                    "restore_env_get_observations",
                    lambda: setattr(
                        self.env,
                        "get_observations",
                        original_get_observations,
                    ),
                )
            if r10_env_step_wrapper_active:
                cleanup(
                    "restore_r10_env_step",
                    lambda: setattr(self.env, "step", r10_original_env_step),
                )
            for index, handle in enumerate(
                reversed(r10_normalizer_hook_handles)
            ):
                cleanup(
                    f"remove_r10_normalizer_hook_{index}",
                    handle.remove,
                )
            if action_ball_update_profiler is not None:
                cleanup(
                    "close_action_ball_update_profiler",
                    action_ball_update_profiler.close,
                )
                cleanup(
                    "clear_action_ball_update_profiler",
                    lambda: setattr(
                        self, "_action_ball_update_profiler", None
                    ),
                )
            if reward_activation_step_wrapper_active:
                cleanup(
                    "restore_reward_env_step",
                    lambda: setattr(self.env, "step", original_env_step),
                )
            if reward_ledger_is_action_bound:
                cleanup(
                    "close_reward_activation_ledger",
                    reward_activation_ledger.close,
                )
            cleanup(
                "clear_rollout_update_wrapper",
                lambda: setattr(self, "_rollout_update_wrapper_active", False),
            )
            cleanup(
                "clear_boundary_stop_callback",
                lambda: setattr(self, "_boundary_stop_requested", None),
            )
            for signum, handler in previous_handlers.items():
                cleanup(
                    f"restore_signal_{signum}",
                    lambda signum=signum, handler=handler: signal.signal(
                        signum, handler
                    ),
                )
            self._learn_cleanup_secondary_failures = tuple(
                (
                    label,
                    f"{type(cleanup_exc).__module__}."
                    f"{type(cleanup_exc).__qualname__}: {cleanup_exc}",
                )
                for label, cleanup_exc in cleanup_failures
            )
            if cleanup_failures and learn_primary_error is None:
                cleanup_exc = cleanup_failures[0][1]
                raise cleanup_exc.with_traceback(cleanup_exc.__traceback__)

    @staticmethod
    def _action_ball_profile_reset_reason_counters(
        exact_behavior: Mapping[str, Mapping]
    ) -> Dict[str, int]:
        """Select reset strata already owned by the exact behavior ledger.

        This deliberately does not inspect device masks or rescan rollout
        storage.  The profiler's wrappers own true-reset/wrap batch counts;
        this projection adds the same-update terminal reason accounting that
        the diagnostic runner already consumed for its canonical receipt.
        """

        selected: Dict[str, int] = {}
        for record in exact_behavior.values():
            counters = record.get("counters")
            if not isinstance(counters, Mapping):
                raise RuntimeError(
                    "ActionBall profile requires exact behavior counters"
                )
            for name, value in counters.items():
                include = name in {
                    "terminal_reset_count",
                    "timeout_reset_count",
                    "swing_completion_count",
                } or name.startswith("termination_reason_")
                if not include:
                    continue
                if type(value) is not int or value < 0:
                    raise RuntimeError(
                        "ActionBall profile reset reason counters must be "
                        "non-negative integers"
                    )
                if name in selected:
                    raise RuntimeError(
                        "ActionBall profile reset reason counter is duplicated"
                    )
                selected[name] = value
        return dict(sorted(selected.items()))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        step = int(locs["it"])
        super().log(locs, width=width, pad=pad)
        self._consume_actual_joint_forbidden_diagnostic(step)
        self._consume_push_velocity_diagnostic_update(step)
        # Consume/print even when TensorBoard/W&B is disabled: this stdout JSON line is the exact
        # per-update receipt, while dashboard logging is optional presentation only.
        exact_behavior = self._consume_exact_behavior_updates(step)
        action_ball_update_profiler = getattr(
            self, "_action_ball_update_profiler", None
        )
        if action_ball_update_profiler is not None:
            # RSL-RL has already measured these two walls.  Reuse them rather
            # than wrapping/reimplementing PPO or adding a CUDA synchronize.
            # The exact behavior ledger was already consumed above, so adding
            # its scalar reset reasons performs no second rollout/device scan.
            action_ball_update_profiler.emit_update(
                update=step,
                collection_time_s=locs["collection_time"],
                learning_time_s=locs["learn_time"],
                reset_reason_counters=(
                    self._action_ball_profile_reset_reason_counters(
                        exact_behavior
                    )
                ),
            )
        if not self.disable_logs and self.writer is not None:
            self._log_live_metrics(step, exact_behavior=exact_behavior)
        # Ctrl-C 缓冲(见 learn()):恰好在一个 PPO 迭代完整结束、日志落账之后才真正存档退出。
        stop_requested = getattr(self, "_boundary_stop_requested", None)
        if callable(stop_requested) and stop_requested():
            checkpoint = pathlib.Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"
            self.save(str(checkpoint))
            print(
                "[MotionOnPolicyRunner] interrupt checkpoint saved at a completed PPO boundary: "
                f"{checkpoint}",
                flush=True,
            )
            raise KeyboardInterrupt

    def _consume_actual_joint_forbidden_diagnostic(
        self, step: int
    ) -> Optional[dict]:
        """Emit non-promotable reset attribution once per diagnostic PPO update."""

        if not self._diagnostic_joint_safety_compact_evidence():
            return None
        if getattr(
            self, "_actual_joint_forbidden_diagnostic_consumed_step", None
        ) == int(step):
            return getattr(
                self, "_actual_joint_forbidden_diagnostic_consumed_record", None
            )
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "action_manager", None)
        getter = None if manager is None else getattr(manager, "get_term", None)
        if not callable(getter):
            raise RuntimeError(
                "ActionBall diagnostic requires action_manager.get_term()"
            )
        action = getter("joint_pos")
        consumer = getattr(
            action, "consume_actual_joint_forbidden_diagnostic", None
        )
        if not callable(consumer):
            raise RuntimeError(
                "ActionBall diagnostic joint action lacks actual-limit attribution"
            )
        payload = consumer()
        if payload.get("enabled") is not True:
            raise RuntimeError(
                "ActionBall diagnostic actual-limit attribution is not enabled"
            )
        record = {
            "event": "action_ball_actual_joint_forbidden_diagnostic_update",
            "schema_version": 2,
            "ppo_update": int(step),
            **payload,
        }
        print(
            "HOPE_ACTUAL_JOINT_DIAGNOSTIC_UPDATE_JSON="
            + json.dumps(record, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        self._actual_joint_forbidden_diagnostic_consumed_step = int(step)
        self._actual_joint_forbidden_diagnostic_consumed_record = record
        return record

    def _consume_push_velocity_diagnostic_update(
        self, step: int
    ) -> Optional[dict]:
        """Emit/clear the diagnostic velocity-push ledger once per PPO update."""

        if not self._diagnostic_joint_safety_compact_evidence():
            return None
        if getattr(self, "_push_velocity_diagnostic_consumed_step", None) == int(step):
            return getattr(
                self, "_push_velocity_diagnostic_consumed_record", None
            )
        from whole_body_tracking.tasks.tracking.mdp.hope_push_events import (
            PUSH_VELOCITY_DIAGNOSTIC_EVENT,
            PUSH_VELOCITY_DIAGNOSTIC_SCHEMA_VERSION,
            consume_push_velocity_diagnostic_counters,
        )

        env = getattr(self.env, "unwrapped", self.env)
        counters = consume_push_velocity_diagnostic_counters(env)
        record = {
            "event": PUSH_VELOCITY_DIAGNOSTIC_EVENT,
            "schema_version": PUSH_VELOCITY_DIAGNOSTIC_SCHEMA_VERSION,
            "ppo_update": int(step),
            "counters": counters,
            "window_aggregation": "sum_counts_and_extrema_across_ppo_updates",
        }
        print(
            "HOPE_PUSH_VELOCITY_DIAGNOSTIC_UPDATE_JSON="
            + json.dumps(
                record,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        self._push_velocity_diagnostic_consumed_step = int(step)
        self._push_velocity_diagnostic_consumed_record = record
        return record

    def _notify_command_terms_rollout_end(self, step: int) -> None:
        """Notify each active command term once, before any per-update ledger is consumed."""

        if self._action_ball_diagnostic_unauthorized():
            # Reward-screen checkpoints are intentionally fixed-domain and
            # cannot promote curriculum state.  Keep their PPO/checkpoint
            # path independent of the formal rollout-end receipt transaction.
            return
        step = int(step)
        previous_step = getattr(self, "_rollout_end_notified_step", None)
        if previous_step == step:
            return
        if previous_step is not None and step < int(previous_step):
            raise RuntimeError(
                f"PPO update moved backwards at rollout boundary: {step} < {previous_step}"
            )
        # Mark before invoking user code. An exception is fatal and propagates; this guard prevents
        # an outer logger retry from double-advancing an already-mutated curriculum.
        self._rollout_end_notified_step = step

        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "command_manager", None)
        if manager is None:
            return
        raw_term_names = tuple(getattr(manager, "active_terms", ()))
        term_names = tuple(str(name) for name in raw_term_names)
        if len(term_names) != len(set(term_names)):
            raise RuntimeError("command manager active_terms contains duplicate names")
        for raw_term_name, term_name in zip(raw_term_names, term_names):
            term = manager.get_term(raw_term_name)
            callback = getattr(term, "on_rollout_end", None)
            if callable(callback):
                callback(step)

    def _bind_joint_safety_action_term(self, *, required: bool):
        """Resolve the sole protected joint action before the first rollout.

        ActionBall and UpperSafe are fail-closed task leaves: starting even one rollout without the
        guard's read-only snapshot and single-consumer API would create an unaudited safety window.
        Historical tasks do not opt in and therefore return ``None`` without changing behavior.
        """

        if not required:
            return None
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "action_manager", None)
        getter = None if manager is None else getattr(manager, "get_term", None)
        if not callable(getter):
            raise RuntimeError(
                "joint-safety protected task requires action_manager.get_term()"
            )
        try:
            term = getter("joint_pos")
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "joint-safety protected task requires the joint_pos action term"
            ) from exc
        snapshotter = getattr(term, "joint_safety_ledger_snapshot", None)
        preparer = getattr(term, "prepare_joint_safety_ledger_consume", None)
        acknowledger = getattr(term, "acknowledge_joint_safety_ledger", None)
        if (
            not callable(snapshotter)
            or not callable(preparer)
            or not callable(acknowledger)
        ):
            raise RuntimeError(
                "joint-safety protected task requires snapshot plus two-phase "
                "prepare/acknowledge APIs on joint_pos"
            )
        probe = snapshotter()
        if not isinstance(probe, dict) or probe.get("enabled") is not True:
            raise RuntimeError(
                "joint-safety protected task has a disabled pre-apply/substep ledger"
            )
        for prefix in ("policy_step_summary", "terminal_archive"):
            if probe.get(f"{prefix}_overflow_latch") is not False:
                raise RuntimeError(
                    f"joint-safety {prefix.replace('_', ' ')} overflow is already latched"
                )
        return term

    @staticmethod
    def _joint_safety_int(value, *, name: str, minimum: Optional[int] = None) -> int:
        if type(value) is not int:
            raise RuntimeError(f"joint-safety {name} must be a plain integer")
        if minimum is not None and value < minimum:
            raise RuntimeError(
                f"joint-safety {name} must be >= {minimum}; got {value}"
            )
        return value

    @staticmethod
    def _joint_safety_tensor(
        value,
        *,
        name: str,
        shape: tuple,
        boolean: bool = False,
        integer: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise RuntimeError(
                f"joint-safety {name} must be a tensor shaped {tuple(shape)}"
            )
        if boolean:
            if value.dtype != torch.bool:
                raise RuntimeError(f"joint-safety {name} must have bool dtype")
        elif integer:
            if value.dtype == torch.bool or value.dtype.is_floating_point:
                raise RuntimeError(f"joint-safety {name} must have integer dtype")
        elif not value.dtype.is_floating_point:
            raise RuntimeError(f"joint-safety {name} must have floating dtype")
        # prepare_joint_safety_ledger_consume() freezes the action term, so validation may operate
        # on its immutable device-resident view.  Keeping dense 4096 x 24 summaries on-device is
        # essential: only sparse counters/reductions cross to CPU for the durable artifact.
        return value.detach()

    @classmethod
    def _joint_safety_identity(
        cls, identity, *, num_envs: int, name: str
    ) -> dict:
        if not isinstance(identity, dict):
            raise RuntimeError(f"joint-safety {name} must be a mapping")
        enabled = identity.get("action_ball_enabled")
        if not isinstance(enabled, bool):
            raise RuntimeError(
                f"joint-safety {name}.action_ball_enabled must be bool"
            )
        tensors = {}
        for field in (
            "action_episode_sequence",
            "episode_length",
            "action_uid",
            "birth_generation",
            "swing_generation",
        ):
            tensors[field] = cls._joint_safety_tensor(
                identity.get(field),
                name=f"{name}.{field}",
                shape=(num_envs,),
                integer=True,
            )
            if tensors[field].dtype != torch.long:
                raise RuntimeError(
                    f"joint-safety {name}.{field} must have int64 dtype"
                )
        if bool(torch.any(tensors["action_episode_sequence"].lt(0)).item()):
            raise RuntimeError(
                f"joint-safety {name}.action_episode_sequence must be non-negative"
            )
        if bool(torch.any(tensors["episode_length"].lt(-1)).item()):
            raise RuntimeError(
                f"joint-safety {name}.episode_length must be >= -1"
            )
        receipts = identity.get("birth_receipt_sha256")
        if not isinstance(receipts, tuple) or len(receipts) != num_envs:
            raise RuntimeError(
                f"joint-safety {name}.birth_receipt_sha256 must be an env-aligned tuple"
            )
        if enabled:
            for field in ("action_uid", "birth_generation", "swing_generation"):
                if bool(torch.any(tensors[field].lt(0)).item()):
                    raise RuntimeError(
                        f"joint-safety {name}.{field} must be non-negative for action-ball"
                    )
            for receipt in receipts:
                if not isinstance(receipt, str) or len(receipt) != 64:
                    raise RuntimeError(
                        f"joint-safety {name} has an invalid birth receipt"
                    )
                try:
                    int(receipt, 16)
                except ValueError as exc:
                    raise RuntimeError(
                        f"joint-safety {name} birth receipt must be hexadecimal"
                    ) from exc
        else:
            for field in ("action_uid", "birth_generation", "swing_generation"):
                if not bool(torch.all(tensors[field].eq(-1)).item()):
                    raise RuntimeError(
                        f"joint-safety {name}.{field} must use -1 outside action-ball"
                    )
            if any(receipt is not None for receipt in receipts):
                raise RuntimeError(
                    f"joint-safety {name} must not carry birth receipts outside action-ball"
                )
        return {
            **tensors,
            "action_ball_enabled": enabled,
            "birth_receipt_sha256": receipts,
        }

    @staticmethod
    def _assert_joint_safety_identity_transition(
        previous: dict, current: dict, *, name: str
    ) -> None:
        if previous["action_ball_enabled"] != current["action_ball_enabled"]:
            raise RuntimeError(
                f"joint-safety {name} changed action-ball mode inside one run"
            )
        delta = (
            current["action_episode_sequence"]
            - previous["action_episode_sequence"]
        )

        def require(mask: torch.Tensor, message: str) -> None:
            bad = torch.nonzero(~mask, as_tuple=False).reshape(-1)
            if bad.numel():
                env_id = int(bad[0].item())
                raise RuntimeError(
                    f"joint-safety {name} {message} for env {env_id}"
                )

        require(
            delta.eq(0) | delta.eq(1),
            "episode generation must advance by zero or one",
        )
        if not current["action_ball_enabled"]:
            return
        same_episode = delta.eq(0)
        reset_episode = delta.eq(1)
        for field in ("action_uid", "birth_generation"):
            require(
                ~same_episode | current[field].eq(previous[field]),
                f"changed {field} without reset",
            )
        require(
            ~same_episode
            | current["swing_generation"].ge(previous["swing_generation"]),
            "swing generation moved backwards",
        )
        require(
            ~reset_episode
            | current["birth_generation"].gt(previous["birth_generation"]),
            "reset did not advance birth generation",
        )
        receipt_equal = torch.as_tensor(
            [
                before == after
                for before, after in zip(
                    previous["birth_receipt_sha256"],
                    current["birth_receipt_sha256"],
                )
            ],
            dtype=torch.bool,
            device=delta.device,
        )
        require(
            ~same_episode | receipt_equal,
            "changed birth receipt without reset",
        )
        require(
            ~reset_episode | ~receipt_equal,
            "reset reused a birth receipt",
        )

    @staticmethod
    def _joint_safety_identity_sha256(identity: dict) -> str:
        digest = hashlib.sha256()
        digest.update(
            b"action_ball=1"
            if identity["action_ball_enabled"]
            else b"action_ball=0"
        )
        for field in (
            "action_episode_sequence",
            "episode_length",
            "action_uid",
            "birth_generation",
            "swing_generation",
        ):
            tensor = identity[field].detach().to(device="cpu").contiguous()
            digest.update(field.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(
            json.dumps(
                list(identity["birth_receipt_sha256"]),
                sort_keys=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _joint_safety_runtime_contract(self, term) -> dict:
        """Bind the exact hard envelope and substep guard used by the live action term."""

        decimation = self._joint_safety_int(
            getattr(term, "_pre_apply_guard_decimation", None),
            name="bound guard decimation",
            minimum=1,
        )
        physics_dt = getattr(term, "_pre_apply_guard_physics_dt_s", None)
        margin_rad = getattr(term, "_pre_apply_guard_margin_rad", None)
        margin_fraction = getattr(
            term, "_pre_apply_guard_margin_fraction", None
        )
        brake_mode = getattr(term, "_pre_apply_guard_brake_mode", None)
        if brake_mode not in (
            "velocity_horizon_v1",
            "max_inward_until_nonoutward_v1",
        ):
            raise RuntimeError(
                "joint-safety bound guard brake mode is missing or invalid"
            )
        for name, value in (
            ("physics_dt_s", physics_dt),
            ("margin_rad", margin_rad),
            ("margin_fraction", margin_fraction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    f"joint-safety bound guard {name} must be finite"
                )
        if float(physics_dt) <= 0.0:
            raise RuntimeError("joint-safety bound physics_dt_s must be positive")
        if float(margin_rad) < 0.0 or not 0.0 <= float(margin_fraction) < 0.5:
            raise RuntimeError("joint-safety bound hard-envelope margin is invalid")

        asset = getattr(term, "_asset", None)
        data = None if asset is None else getattr(asset, "data", None)
        hard_limits = None if data is None else getattr(data, "joint_pos_limits", None)
        joint_ids = getattr(term, "_joint_ids", None)
        if not torch.is_tensor(hard_limits):
            raise RuntimeError(
                "joint-safety bound action term has no tensor hard joint limits"
            )
        try:
            selected_limits = hard_limits[:, joint_ids, :]
        except (IndexError, TypeError) as exc:
            raise RuntimeError(
                "joint-safety failed to resolve the action joint hard limits"
            ) from exc
        processed = getattr(term, "_processed_actions", None)
        if not torch.is_tensor(processed) or processed.ndim != 2:
            raise RuntimeError(
                "joint-safety bound action term has no env-by-joint action tensor"
            )
        num_envs, joint_count = tuple(processed.shape)
        if tuple(selected_limits.shape) != (num_envs, joint_count, 2):
            raise RuntimeError(
                "joint-safety hard limits do not match the action tensor"
            )
        lower = selected_limits[..., 0]
        upper = selected_limits[..., 1]
        valid = torch.all(
            torch.isfinite(lower)
            & torch.isfinite(upper)
            & lower.lt(upper)
            & lower.eq(lower[0])
            & upper.eq(upper[0])
        )
        if not bool(valid.item()):
            raise RuntimeError(
                "joint-safety requires one finite identical physical hard envelope "
                "across all environments"
            )
        names = getattr(term, "_joint_names", None)
        if not isinstance(names, (list, tuple)) or len(names) != joint_count:
            raise RuntimeError(
                "joint-safety requires the exact action/articulation joint-name order"
            )
        if any(type(name) is not str or not name for name in names):
            raise RuntimeError(
                "joint-safety joint-name order must contain exact non-empty strings"
            )
        joint_names = tuple(names)
        if len(set(joint_names)) != joint_count:
            raise RuntimeError(
                "joint-safety joint-name order must be non-empty and unique"
            )
        hard_lower = lower[0].detach().to(device="cpu").clone()
        hard_upper = upper[0].detach().to(device="cpu").clone()
        digest = hashlib.sha256()
        scalar_contract = {
            "expected_apply_calls": decimation,
            "physics_dt_s": float(physics_dt),
            "margin_rad": float(margin_rad),
            "margin_fraction": float(margin_fraction),
            "brake_mode": brake_mode,
            "num_envs": int(num_envs),
            "joint_count": int(joint_count),
            "joint_names": joint_names,
        }
        digest.update(
            json.dumps(
                scalar_contract, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        for name, tensor in (
            ("hard_lower", hard_lower),
            ("hard_upper", hard_upper),
        ):
            digest.update(name.encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.contiguous().numpy().tobytes(order="C"))
        return {
            **scalar_contract,
            "hard_lower": hard_lower,
            "hard_upper": hard_upper,
            "sha256": digest.hexdigest(),
        }

    @staticmethod
    def _joint_safety_payload_bytes(value) -> int:
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, dict):
            return sum(
                MotionOnPolicyRunner._joint_safety_payload_bytes(item)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return sum(
                MotionOnPolicyRunner._joint_safety_payload_bytes(item)
                for item in value
            )
        return 0

    @staticmethod
    def _joint_safety_finite_number(value, *, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"joint-safety {name} must be a finite number")
        return float(value)

    def _validate_joint_safety_terminal_transcript(
        self,
        transcript: dict,
        *,
        archive_index: int,
        sequence: int,
        env_id: int,
        row: dict,
        contract: dict,
    ) -> dict:
        """Deeply validate one archived env's exact 4+1 physics transcript."""

        prefix = f"terminal archive {archive_index}.transcript"
        expected_keys = {
            "schema_version",
            "policy_step_sequence",
            "policy_start_timestamp_s",
            "expected_apply_calls",
            "physics_dt_s",
            "apply_call_count",
            "post_readback_count",
            "complete",
            "record_count",
            "record_kind",
            "call_index",
            "timestamp_s",
            "joint_pos_timestamp_s",
            "joint_vel_timestamp_s",
            "env_valid",
            "q",
            "qdot",
            "hard_lower_gap",
            "hard_upper_gap",
            "hard_crossing",
            "actual_hard_edge",
            "qdes_env_latch",
            "crossing_env_latch",
            "qdes_joint_latch",
            "crossing_joint_latch",
            "qdes_joint_count",
            "crossing_joint_count",
            "substep_crossing_joint_latch",
            "substep_actual_joint_latch",
            "substep_crossing_joint_count",
            "substep_actual_joint_count",
            "step_qdes_joint_count",
            "step_policy_crossing_joint_count",
        }
        if not isinstance(transcript, dict) or set(transcript) != expected_keys:
            raise RuntimeError(
                f"joint-safety {prefix} has missing or unexpected fields"
            )
        expected_apply = contract["expected_apply_calls"]
        record_count = expected_apply + 1
        if (
            transcript.get("schema_version") != 1
            or transcript.get("policy_step_sequence") != sequence
            or transcript.get("expected_apply_calls") != expected_apply
            or transcript.get("apply_call_count") != expected_apply
            or transcript.get("post_readback_count") != 1
            or transcript.get("complete") is not True
            or transcript.get("record_count") != record_count
        ):
            raise RuntimeError(
                f"joint-safety {prefix} is not a complete bound 4+1 transcript"
            )
        physics_dt = self._joint_safety_finite_number(
            transcript.get("physics_dt_s"), name=f"{prefix}.physics_dt_s"
        )
        if not math.isclose(
            physics_dt,
            contract["physics_dt_s"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"joint-safety {prefix} physics dt drift")
        start = self._joint_safety_finite_number(
            transcript.get("policy_start_timestamp_s"),
            name=f"{prefix}.policy_start_timestamp_s",
        )
        expected_kind = tuple(["apply"] * expected_apply + ["post"])
        expected_index = tuple(range(record_count))
        if (
            transcript.get("record_kind") != expected_kind
            or transcript.get("call_index") != expected_index
        ):
            raise RuntimeError(
                f"joint-safety {prefix} record kind/index order is invalid"
            )
        timestamps = []
        for field in (
            "timestamp_s",
            "joint_pos_timestamp_s",
            "joint_vel_timestamp_s",
        ):
            raw = transcript.get(field)
            if not isinstance(raw, tuple) or len(raw) != record_count:
                raise RuntimeError(
                    f"joint-safety {prefix}.{field} must have {record_count} rows"
                )
            timestamps.append(
                tuple(
                    self._joint_safety_finite_number(
                        value, name=f"{prefix}.{field}[{index}]"
                    )
                    for index, value in enumerate(raw)
                )
            )
        if timestamps[0] != timestamps[1] or timestamps[0] != timestamps[2]:
            raise RuntimeError(
                f"joint-safety {prefix} lazy-buffer timestamps are not exact"
            )
        for index, timestamp in enumerate(timestamps[0]):
            expected_timestamp = start + index * physics_dt
            if not math.isclose(
                timestamp,
                expected_timestamp,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ) or (index and not timestamp > timestamps[0][index - 1]):
                raise RuntimeError(
                    f"joint-safety {prefix} timestamp sequence is invalid"
                )

        joint_count = contract["joint_count"]
        env_valid = self._joint_safety_tensor(
            transcript.get("env_valid"),
            name=f"{prefix}.env_valid",
            shape=(record_count,),
            boolean=True,
        )
        if not bool(torch.all(env_valid).item()):
            raise RuntimeError(
                f"joint-safety {prefix} contains an invalid/incomplete record row"
            )
        float_fields = {}
        for field in (
            "q",
            "qdot",
            "hard_lower_gap",
            "hard_upper_gap",
        ):
            float_fields[field] = self._joint_safety_tensor(
                transcript.get(field),
                name=f"{prefix}.{field}",
                shape=(record_count, joint_count),
            )
        bool_fields = {}
        for field in ("hard_crossing", "actual_hard_edge"):
            bool_fields[field] = self._joint_safety_tensor(
                transcript.get(field),
                name=f"{prefix}.{field}",
                shape=(record_count, joint_count),
                boolean=True,
            )
        q = float_fields["q"]
        qdot = float_fields["qdot"]
        lower_gap = float_fields["hard_lower_gap"]
        upper_gap = float_fields["hard_upper_gap"]
        hard_lower = contract["hard_lower"].to(dtype=q.dtype)[None, :]
        hard_upper = contract["hard_upper"].to(dtype=q.dtype)[None, :]
        finite_q = torch.isfinite(q)
        finite_qdot = torch.isfinite(qdot)
        expected_lower_gap = q - hard_lower
        expected_upper_gap = hard_upper - q
        if bool(
            torch.any(
                finite_q
                & (
                    ~torch.isfinite(lower_gap)
                    | ~torch.isfinite(upper_gap)
                    | ~torch.isclose(
                        lower_gap,
                        expected_lower_gap,
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                    | ~torch.isclose(
                        upper_gap,
                        expected_upper_gap,
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                )
            ).item()
        ):
            raise RuntimeError(
                f"joint-safety {prefix} q and physical hard gaps disagree"
            )
        actual_expected = (
            ~finite_q | lower_gap.le(0.0) | upper_gap.le(0.0)
        )
        if not torch.equal(bool_fields["actual_hard_edge"], actual_expected):
            raise RuntimeError(
                f"joint-safety {prefix} actual-hard-edge mask is forged"
            )
        travel = hard_upper - hard_lower
        inset = (
            contract["margin_rad"]
            + contract["margin_fraction"] * travel
        )
        inner_lower = hard_lower + inset
        inner_upper = hard_upper - inset
        safe_q = torch.where(finite_q, q, torch.zeros_like(q))
        safe_qdot = torch.where(finite_qdot, qdot, torch.zeros_like(qdot))
        # The action term deliberately applies the same full control/reaction
        # horizon at every fresh physics-substep readback.  Recomputing this
        # mask with one physics tick here would reject an honest transcript
        # whenever only the full policy horizon reaches the inner hard guard.
        guard_horizon_s = physics_dt * expected_apply
        ballistic_next = safe_q + safe_qdot * guard_horizon_s
        crossing_expected = (
            ~finite_q
            | ~finite_qdot
            | safe_q.le(inner_lower)
            | safe_q.ge(inner_upper)
            | ballistic_next.le(inner_lower)
            | ballistic_next.ge(inner_upper)
        )
        if not torch.equal(bool_fields["hard_crossing"], crossing_expected):
            raise RuntimeError(
                f"joint-safety {prefix} hard-crossing mask is forged"
            )

        def exact_tensor(
            field: str,
            *,
            shape: tuple,
            boolean: bool = False,
        ) -> torch.Tensor:
            value = self._joint_safety_tensor(
                transcript.get(field),
                name=f"{prefix}.{field}",
                shape=shape,
                boolean=boolean,
                integer=not boolean,
            )
            expected_dtype = torch.bool if boolean else torch.long
            if value.dtype != expected_dtype:
                raise RuntimeError(
                    f"joint-safety {prefix}.{field} has wrong dtype"
                )
            return value

        qdes_env_latch = exact_tensor(
            "qdes_env_latch", shape=(), boolean=True
        )
        crossing_env_latch = exact_tensor(
            "crossing_env_latch", shape=(), boolean=True
        )
        joint_bool_names = (
            "qdes_joint_latch",
            "crossing_joint_latch",
            "substep_crossing_joint_latch",
            "substep_actual_joint_latch",
        )
        joint_int_names = (
            "qdes_joint_count",
            "crossing_joint_count",
            "substep_crossing_joint_count",
            "substep_actual_joint_count",
            "step_qdes_joint_count",
            "step_policy_crossing_joint_count",
        )
        joint_bools = {
            field: exact_tensor(field, shape=(joint_count,), boolean=True)
            for field in joint_bool_names
        }
        joint_counts = {
            field: exact_tensor(field, shape=(joint_count,))
            for field in joint_int_names
        }
        if any(
            bool(torch.any(value.lt(0)).item())
            for value in joint_counts.values()
        ):
            raise RuntimeError(
                f"joint-safety {prefix} contains a negative counter"
            )
        transcript_crossing_count = bool_fields["hard_crossing"].sum(
            dim=0, dtype=torch.long
        )
        transcript_actual_count = bool_fields["actual_hard_edge"].sum(
            dim=0, dtype=torch.long
        )
        # The transcript is one policy step, while these action-term counters
        # and latches are intentionally episode-sticky until reset.  The
        # current-step counts must be a subset of the cumulative episode
        # counts; exact current-step equality is checked against ``row`` below.
        if (
            bool(
                torch.any(
                    transcript_crossing_count
                    > joint_counts["substep_crossing_joint_count"]
                ).item()
            )
            or bool(
                torch.any(
                    transcript_actual_count
                    > joint_counts["substep_actual_joint_count"]
                ).item()
            )
            or not torch.equal(
                joint_bools["substep_crossing_joint_latch"],
                joint_counts["substep_crossing_joint_count"].gt(0),
            )
            or not torch.equal(
                joint_bools["substep_actual_joint_latch"],
                joint_counts["substep_actual_joint_count"].gt(0),
            )
        ):
            raise RuntimeError(
                f"joint-safety {prefix} substep masks/counters disagree"
            )
        if (
            not torch.equal(
                joint_bools["qdes_joint_latch"],
                joint_counts["qdes_joint_count"].gt(0),
            )
            or not torch.equal(
                joint_bools["crossing_joint_latch"],
                joint_counts["crossing_joint_count"].gt(0)
                | joint_counts["substep_crossing_joint_count"].gt(0)
                | joint_counts["substep_actual_joint_count"].gt(0),
            )
            or bool(qdes_env_latch.item())
            != bool(torch.any(joint_bools["qdes_joint_latch"]).item())
            or bool(crossing_env_latch.item())
            != bool(torch.any(joint_bools["crossing_joint_latch"]).item())
        ):
            raise RuntimeError(
                f"joint-safety {prefix} episode latches disagree with counters"
            )
        if (
            not torch.equal(
                joint_counts["step_qdes_joint_count"],
                row["qdes_joint_count"][env_id].to(device="cpu"),
            )
            or not torch.equal(
                joint_counts["step_policy_crossing_joint_count"],
                row["policy_crossing_joint_count"][env_id].to(device="cpu"),
            )
            or not torch.equal(
                transcript_crossing_count,
                row["substep_hard_crossing_joint_count"][env_id].to(
                    device="cpu"
                ),
            )
            or not torch.equal(
                transcript_actual_count,
                row["actual_hard_edge_joint_count"][env_id].to(
                    device="cpu"
                ),
            )
            or bool(
                torch.any(
                    joint_counts["step_qdes_joint_count"]
                    > joint_counts["qdes_joint_count"]
                ).item()
            )
            or bool(
                torch.any(
                    joint_counts["step_policy_crossing_joint_count"]
                    > joint_counts["crossing_joint_count"]
                ).item()
            )
        ):
            raise RuntimeError(
                f"joint-safety {prefix} does not match its policy-step summary"
            )
        finite_lower = torch.where(
            torch.isfinite(lower_gap),
            lower_gap,
            torch.full_like(lower_gap, float("-inf")),
        )
        finite_upper = torch.where(
            torch.isfinite(upper_gap),
            upper_gap,
            torch.full_like(upper_gap, float("-inf")),
        )
        transcript_minimum_gap = torch.minimum(
            finite_lower.amin(dim=0), finite_upper.amin(dim=0)
        )
        if not torch.equal(
            transcript_minimum_gap,
            row["minimum_gap"][env_id].to(device="cpu"),
        ):
            raise RuntimeError(
                f"joint-safety {prefix} minimum hard gap disagrees with summary"
            )
        return {
            "record_count": record_count,
            "actual_hard_edge_count": int(
                transcript_actual_count.sum().item()
            ),
            "hard_crossing_count": int(
                transcript_crossing_count.sum().item()
            ),
        }

    def _validate_joint_safety_update_snapshot(
        self, snapshot: dict, *, step: int, contract: Optional[dict] = None
    ) -> dict:
        """Validate one consumed PPO window before any evidence is published."""

        if not isinstance(snapshot, dict) or snapshot.get("enabled") is not True:
            raise RuntimeError(
                "joint-safety protected update returned a disabled or malformed ledger"
            )
        for prefix in ("policy_step_summary", "terminal_archive"):
            latch = snapshot.get(f"{prefix}_overflow_latch")
            count = snapshot.get(f"{prefix}_overflow_count")
            if not isinstance(latch, bool):
                raise RuntimeError(
                    f"joint-safety {prefix} overflow latch must be bool"
                )
            count = self._joint_safety_int(
                count, name=f"{prefix}_overflow_count", minimum=0
            )
            if latch or count:
                raise RuntimeError(
                    f"joint-safety {prefix.replace('_', ' ')} overflow is sticky"
                )

        since = snapshot.get("since_last_consume")
        if not isinstance(since, dict) or since.get("has_data") is not True:
            raise RuntimeError(
                "joint-safety PPO update has no since-last-consume evidence"
            )
        consume_sequence = self._joint_safety_int(
            since.get("consume_sequence"),
            name="consume_sequence",
            minimum=0,
        )
        expected_steps = self._joint_safety_int(
            getattr(self, "num_steps_per_env", None),
            name="runner num_steps_per_env",
            minimum=1,
        )
        policy_steps_raw = since.get("policy_step_count")
        if not torch.is_tensor(policy_steps_raw) or policy_steps_raw.ndim != 1:
            raise RuntimeError(
                "joint-safety policy_step_count must be a one-dimensional tensor"
            )
        num_envs = int(policy_steps_raw.numel())
        if num_envs <= 0:
            raise RuntimeError("joint-safety ledger cannot have zero environments")
        if (
            not isinstance(contract, dict)
            or contract.get("num_envs") != num_envs
            or type(contract.get("joint_count")) is not int
            or contract["joint_count"] <= 0
            or type(contract.get("expected_apply_calls")) is not int
            or contract["expected_apply_calls"] <= 0
            or not isinstance(contract.get("sha256"), str)
            or len(contract["sha256"]) != 64
        ):
            raise RuntimeError(
                "joint-safety update is missing its exact runtime hard-envelope contract"
            )
        expected_apply_calls = contract["expected_apply_calls"]
        contract_physics_dt = float(contract["physics_dt_s"])
        policy_steps = self._joint_safety_tensor(
            policy_steps_raw,
            name="policy_step_count",
            shape=(num_envs,),
            integer=True,
        )
        if not bool(torch.all(policy_steps.eq(expected_steps)).item()):
            raise RuntimeError(
                "joint-safety PPO update does not contain exactly num_steps_per_env "
                "policy summaries for every environment"
            )

        summaries = snapshot.get("identity_bound_policy_steps")
        if not isinstance(summaries, tuple):
            raise RuntimeError(
                "joint-safety identity_bound_policy_steps must be a tuple"
            )
        used = self._joint_safety_int(
            snapshot.get("policy_step_summary_used"),
            name="policy_step_summary_used",
            minimum=0,
        )
        bound_count = self._joint_safety_int(
            since.get("identity_bound_policy_step_count"),
            name="identity_bound_policy_step_count",
            minimum=0,
        )
        if used != len(summaries) or bound_count != len(summaries):
            raise RuntimeError(
                "joint-safety policy summary counts do not match retained summaries"
            )
        if len(summaries) != expected_steps:
            raise RuntimeError(
                "joint-safety retained summary count does not match PPO rollout length"
            )
        summary_rows = []
        identities = []
        previous_sequence = None
        previous_policy_start = None
        previous_identity = getattr(self, "_joint_safety_last_identity", None)
        for index, summary in enumerate(summaries):
            if not isinstance(summary, dict):
                raise RuntimeError(
                    f"joint-safety policy summary {index} must be a mapping"
                )
            if summary.get("schema_version") != 1:
                raise RuntimeError(
                    f"joint-safety policy summary {index} schema drift"
                )
            if (
                summary.get("included_in_accumulator") is not True
                or summary.get("full_joint_identity_order") is not True
                or summary.get("count_dtype") != "uint8"
            ):
                raise RuntimeError(
                    f"joint-safety policy summary {index} lost its evidence contract"
                )
            if summary.get("accumulator_consume_sequence") != consume_sequence:
                raise RuntimeError(
                    f"joint-safety policy summary {index} belongs to another consume epoch"
                )
            sequence = self._joint_safety_int(
                summary.get("policy_step_sequence"),
                name=f"policy summary {index} sequence",
                minimum=0,
            )
            if previous_sequence is not None and sequence != previous_sequence + 1:
                raise RuntimeError(
                    "joint-safety policy-step sequences are not contiguous"
                )
            previous_sequence = sequence
            policy_start = self._joint_safety_finite_number(
                summary.get("policy_start_timestamp_s"),
                name=f"policy summary {index}.policy_start_timestamp_s",
            )
            if previous_policy_start is not None and not math.isclose(
                policy_start,
                previous_policy_start
                + expected_apply_calls * contract_physics_dt,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise RuntimeError(
                    "joint-safety policy-step timestamps are not contiguous"
                )
            previous_policy_start = policy_start
            if summary.get("expected_apply_calls") != expected_apply_calls:
                raise RuntimeError(
                    f"joint-safety policy summary {index} apply-readback contract drift"
                )
            physics_dt = summary.get("physics_dt_s")
            if (
                isinstance(physics_dt, bool)
                or not isinstance(physics_dt, (int, float))
                or not math.isfinite(float(physics_dt))
                or not math.isclose(
                    float(physics_dt),
                    contract_physics_dt,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise RuntimeError(
                    f"joint-safety policy summary {index} physics dt drift"
                )
            identity = self._joint_safety_identity(
                summary.get("action_identity"),
                num_envs=num_envs,
                name=f"policy summary {index} identity",
            )
            identity_sha256 = self._joint_safety_identity_sha256(identity)
            if previous_identity is not None:
                self._assert_joint_safety_identity_transition(
                    previous_identity,
                    identity,
                    name=f"policy summary {index}",
                )
            previous_identity = identity
            identities.append(identity)
            row_filled = self._joint_safety_tensor(
                summary.get("row_filled"),
                name=f"policy summary {index}.row_filled",
                shape=(num_envs,),
                boolean=True,
            )
            complete = self._joint_safety_tensor(
                summary.get("complete"),
                name=f"policy summary {index}.complete",
                shape=(num_envs,),
                boolean=True,
            )
            apply_count = self._joint_safety_tensor(
                summary.get("apply_readback_count"),
                name=f"policy summary {index}.apply_readback_count",
                shape=(num_envs,),
                integer=True,
            )
            if apply_count.dtype != torch.uint8:
                raise RuntimeError(
                    f"joint-safety policy summary {index}.apply_readback_count "
                    "must have uint8 dtype"
                )
            apply_count = apply_count.to(dtype=torch.long)
            post_count = self._joint_safety_tensor(
                summary.get("post_readback_count"),
                name=f"policy summary {index}.post_readback_count",
                shape=(num_envs,),
                integer=True,
            )
            if post_count.dtype != torch.uint8:
                raise RuntimeError(
                    f"joint-safety policy summary {index}.post_readback_count "
                    "must have uint8 dtype"
                )
            post_count = post_count.to(dtype=torch.long)
            timestamp_pass = self._joint_safety_tensor(
                summary.get("timestamp_invariant_pass"),
                name=f"policy summary {index}.timestamp_invariant_pass",
                shape=(num_envs,),
                boolean=True,
            )
            if not bool(torch.all(row_filled).item()):
                raise RuntimeError(
                    f"joint-safety policy summary {index} has missing environment rows"
                )
            expected_complete = (
                apply_count.eq(expected_apply_calls)
                & post_count.eq(1)
                & timestamp_pass
            )
            if not torch.equal(complete, expected_complete):
                raise RuntimeError(
                    f"joint-safety policy summary {index} completeness is inconsistent"
                )
            if not bool(torch.all(complete).item()):
                raise RuntimeError(
                    f"joint-safety policy summary {index} is incomplete; every "
                    f"environment requires exactly {expected_apply_calls} apply "
                    "readbacks and one "
                    "fresh post-step readback before PPO"
                )
            joint_shape = None
            counters = {}
            for field in (
                "qdes_joint_count",
                "policy_crossing_joint_count",
                "substep_hard_crossing_joint_count",
                "actual_hard_edge_joint_count",
            ):
                value = summary.get(field)
                if (
                    not torch.is_tensor(value)
                    or value.ndim != 2
                    or value.shape[0] != num_envs
                ):
                    raise RuntimeError(
                        f"joint-safety policy summary {index}.{field} has wrong shape"
                    )
                if joint_shape is None:
                    joint_shape = tuple(value.shape)
                if tuple(value.shape) != joint_shape:
                    raise RuntimeError(
                        f"joint-safety policy summary {index} joint shapes disagree"
                    )
                counters[field] = self._joint_safety_tensor(
                    value,
                    name=f"policy summary {index}.{field}",
                    shape=joint_shape,
                    integer=True,
                )
                if counters[field].dtype != torch.uint8:
                    raise RuntimeError(
                        f"joint-safety policy summary {index}.{field} must "
                        "have uint8 dtype"
                    )
                counters[field] = counters[field].to(dtype=torch.long)
                if bool(torch.any(counters[field].lt(0)).item()):
                    raise RuntimeError(
                        f"joint-safety policy summary {index}.{field} is negative"
                    )
            minimum_gap = self._joint_safety_tensor(
                summary.get("minimum_hard_gap"),
                name=f"policy summary {index}.minimum_hard_gap",
                shape=joint_shape,
            )
            if joint_shape != (num_envs, contract["joint_count"]):
                raise RuntimeError(
                    f"joint-safety policy summary {index} joint count does not "
                    "match the bound action/hard-envelope contract"
                )
            if bool(torch.any(torch.isnan(minimum_gap)).item()):
                raise RuntimeError(
                    f"joint-safety policy summary {index} has NaN hard gap"
                )
            summary_rows.append(
                {
                    "sequence": sequence,
                    "policy_start_timestamp_s": policy_start,
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                    "complete": complete,
                    "apply_count": apply_count,
                    "post_count": post_count,
                    "timestamp_pass": timestamp_pass,
                    "minimum_gap": minimum_gap,
                    **counters,
                }
            )

        last_sequence = getattr(self, "_joint_safety_last_policy_step_sequence", None)
        if (
            last_sequence is not None
            and summary_rows[0]["sequence"] != int(last_sequence) + 1
        ):
            raise RuntimeError(
                "joint-safety policy-step sequence is discontinuous across PPO updates"
            )
        last_consume = getattr(self, "_joint_safety_last_consume_sequence", None)
        if last_consume is not None and consume_sequence != int(last_consume) + 1:
            raise RuntimeError(
                "joint-safety consume sequence is discontinuous across PPO updates"
            )

        def since_vector(field: str, *, boolean: bool = False) -> torch.Tensor:
            value = self._joint_safety_tensor(
                since.get(field),
                name=field,
                shape=(num_envs,),
                boolean=boolean,
                integer=not boolean,
            )
            if not boolean and value.dtype != torch.long:
                raise RuntimeError(
                    f"joint-safety accumulator {field} must have int64 dtype"
                )
            return value.to(dtype=torch.bool if boolean else torch.long)

        complete_total = torch.stack(
            [row["complete"].to(dtype=torch.long) for row in summary_rows]
        ).sum(dim=0)
        apply_total = torch.stack(
            [row["apply_count"] for row in summary_rows]
        ).sum(dim=0)
        post_total = torch.stack(
            [row["post_count"] for row in summary_rows]
        ).sum(dim=0)
        for field, expected in (
            ("complete_policy_step_count", complete_total),
            ("incomplete_policy_step_count", policy_steps - complete_total),
            ("apply_readback_count", apply_total),
            ("post_readback_count", post_total),
            ("timestamp_invariant_pass_count", complete_total),
        ):
            if not torch.equal(since_vector(field), expected):
                raise RuntimeError(
                    f"joint-safety accumulator {field} disagrees with per-step evidence"
                )
        if bool(torch.any(since_vector("incomplete_policy_step_count").ne(0)).item()):
            raise RuntimeError(
                "joint-safety PPO update contains an incomplete environment-policy "
                "step; protected training requires exact 4+1 readback coverage"
            )

        joint_shape = tuple(summary_rows[0]["minimum_gap"].shape)
        aggregate_fields = (
            "qdes_joint_count",
            "policy_crossing_joint_count",
            "substep_hard_crossing_joint_count",
            "actual_hard_edge_joint_count",
        )
        aggregate = {}
        for field in aggregate_fields:
            aggregate[field] = torch.stack(
                [row[field] for row in summary_rows]
            ).sum(dim=0)
            observed = self._joint_safety_tensor(
                since.get(field),
                name=f"since_last_consume.{field}",
                shape=joint_shape,
                integer=True,
            )
            if observed.dtype != torch.long:
                raise RuntimeError(
                    f"joint-safety accumulator {field} must have int64 dtype"
                )
            observed = observed.to(dtype=torch.long)
            if not torch.equal(observed, aggregate[field]):
                raise RuntimeError(
                    f"joint-safety accumulator {field} disagrees with per-step evidence"
                )
        for latch_name, field in (
            ("hard_crossing_latch", "substep_hard_crossing_joint_count"),
            ("actual_hard_edge_latch", "actual_hard_edge_joint_count"),
        ):
            observed = since_vector(latch_name, boolean=True)
            expected = torch.any(aggregate[field].gt(0), dim=1)
            if not torch.equal(observed, expected):
                raise RuntimeError(
                    f"joint-safety accumulator {latch_name} disagrees with counts"
                )
        lower_gap = self._joint_safety_tensor(
            since.get("minimum_hard_lower_gap"),
            name="minimum_hard_lower_gap",
            shape=joint_shape,
        )
        upper_gap = self._joint_safety_tensor(
            since.get("minimum_hard_upper_gap"),
            name="minimum_hard_upper_gap",
            shape=joint_shape,
        )
        if bool(
            torch.any(torch.isnan(lower_gap) | torch.isnan(upper_gap)).item()
        ):
            raise RuntimeError("joint-safety accumulator hard gap contains NaN")
        if bool(
            torch.any(torch.isposinf(lower_gap) | torch.isposinf(upper_gap)).item()
        ):
            raise RuntimeError(
                "joint-safety complete PPO update has a missing hard-gap observation"
            )
        combined_gap = torch.minimum(lower_gap, upper_gap)
        per_step_gap = torch.stack(
            [row["minimum_gap"] for row in summary_rows]
        ).amin(dim=0)
        if not torch.equal(combined_gap, per_step_gap):
            raise RuntimeError(
                "joint-safety accumulator hard gap disagrees with per-step evidence"
            )

        archives = snapshot.get("terminal_archives")
        if not isinstance(archives, tuple):
            raise RuntimeError("joint-safety terminal_archives must be a tuple")
        diagnostic_compact_evidence = (
            self._action_ball_diagnostic_unauthorized()
        )
        if diagnostic_compact_evidence and archives:
            raise RuntimeError(
                "diagnostic joint-safety evidence must not materialize "
                "per-reset terminal transcripts"
            )
        archive_used = self._joint_safety_int(
            snapshot.get("terminal_archive_used"),
            name="terminal_archive_used",
            minimum=0,
        )
        if archive_used != len(archives):
            raise RuntimeError(
                "joint-safety terminal archive count does not match retained records"
            )
        summary_by_sequence = {
            row["sequence"]: (row, identity)
            for row, identity in zip(summary_rows, identities)
        }
        expected_archive_keys = {
            "archive_sequence",
            "env_id",
            "policy_step_sequence",
            "action_episode_sequence",
            "episode_length",
            "episode_length_at_policy_start",
            "episode_length_at_reset_hook",
            "action_ball_enabled",
            "action_uid",
            "birth_generation",
            "swing_generation",
            "birth_receipt_sha256",
            "reasons",
            "reset_hook_observed",
            "termination_status_available",
            "terminated",
            "timed_out",
            "included_in_accumulator",
            "accumulator_consume_sequence",
            "transcript",
            "payload_bytes",
        }
        previous_archive_sequence = getattr(
            self, "_joint_safety_last_archive_sequence", None
        )
        seen_archive_keys = set()
        validated_archives = []
        for archive_index, archive in enumerate(archives):
            if (
                not isinstance(archive, dict)
                or set(archive) != expected_archive_keys
            ):
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} has missing "
                    "or unexpected fields"
                )
            archive_sequence = self._joint_safety_int(
                archive.get("archive_sequence"),
                name=f"terminal archive {archive_index}.archive_sequence",
                minimum=0,
            )
            expected_archive_sequence = (
                0
                if previous_archive_sequence is None
                else previous_archive_sequence + 1
            )
            if archive_sequence != expected_archive_sequence:
                raise RuntimeError(
                    "joint-safety terminal archive sequence is discontinuous"
                )
            previous_archive_sequence = archive_sequence
            if (
                archive.get("included_in_accumulator") is not True
                or archive.get("accumulator_consume_sequence")
                != consume_sequence
            ):
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} belongs to another consume epoch"
                )
            env_id = self._joint_safety_int(
                archive.get("env_id"),
                name=f"terminal archive {archive_index}.env_id",
                minimum=0,
            )
            if env_id >= num_envs:
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} env is out of range"
                )
            sequence = self._joint_safety_int(
                archive.get("policy_step_sequence"),
                name=f"terminal archive {archive_index}.policy_step_sequence",
                minimum=0,
            )
            if sequence not in summary_by_sequence:
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} has no policy summary"
                )
            archive_key = (sequence, env_id)
            if archive_key in seen_archive_keys:
                raise RuntimeError(
                    "joint-safety terminal archive repeats an env-policy key"
                )
            seen_archive_keys.add(archive_key)
            row, identity = summary_by_sequence[sequence]
            expected_values = {
                "action_episode_sequence": int(
                    identity["action_episode_sequence"][env_id].item()
                ),
                "action_uid": int(identity["action_uid"][env_id].item()),
                "birth_generation": int(
                    identity["birth_generation"][env_id].item()
                ),
                "swing_generation": int(
                    identity["swing_generation"][env_id].item()
                ),
                "birth_receipt_sha256": identity["birth_receipt_sha256"][
                    env_id
                ],
                "episode_length_at_policy_start": int(
                    identity["episode_length"][env_id].item()
                ),
                "action_ball_enabled": identity["action_ball_enabled"],
            }
            for field, expected in expected_values.items():
                if archive.get(field) != expected:
                    raise RuntimeError(
                        f"joint-safety terminal archive {archive_index} {field} "
                        "does not match its policy-step identity"
                    )
            for field in (
                "reset_hook_observed",
                "termination_status_available",
                "terminated",
                "timed_out",
            ):
                if not isinstance(archive.get(field), bool):
                    raise RuntimeError(
                        f"joint-safety terminal archive {archive_index}.{field} "
                        "must be bool"
                    )
            if (
                archive.get("reset_hook_observed") is not True
                or archive.get("episode_length")
                != archive.get("episode_length_at_reset_hook")
            ):
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} is not bound "
                    "to the reset hook"
                )
            reasons = archive.get("reasons")
            if (
                not isinstance(reasons, tuple)
                or not reasons
                or any(
                    not isinstance(reason, str) or not reason
                    for reason in reasons
                )
                or len(reasons) != len(set(reasons))
                or "reset" not in reasons
                or any(reason not in {"unsafe", "reset"} for reason in reasons)
            ):
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} has invalid reasons"
                )
            if archive["termination_status_available"] and not (
                archive["terminated"] or archive["timed_out"]
            ):
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} has no "
                    "termination outcome"
                )
            transcript = archive.get("transcript")
            transcript_summary = self._validate_joint_safety_terminal_transcript(
                transcript,
                archive_index=archive_index,
                sequence=sequence,
                env_id=env_id,
                row=row,
                contract=contract,
            )
            unsafe = any(
                bool(row[field][env_id].ne(0).any().item())
                for field in aggregate_fields
            )
            if ("unsafe" in reasons) != unsafe:
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} unsafe reason "
                    "does not match its current-step counters"
                )
            payload_bytes = self._joint_safety_int(
                archive.get("payload_bytes"),
                name=f"terminal archive {archive_index}.payload_bytes",
                minimum=0,
            )
            if payload_bytes != self._joint_safety_payload_bytes(archive):
                raise RuntimeError(
                    f"joint-safety terminal archive {archive_index} payload byte "
                    "accounting drift"
                )
            validated_archives.append(
                {
                    "archive": archive,
                    "transcript_summary": transcript_summary,
                    "retain_full_transcript": bool(
                        unsafe or archive["terminated"]
                    ),
                }
            )
        for row in summary_rows:
            # A predicted inner-envelope crossing is a non-terminal brake
            # event, and a finite q_des projection is a non-terminal learning
            # signal.  Only a raw physical hard-edge observation is required
            # to have a matching terminal-reset forensic archive in formal
            # evidence.  The diagnostic screen keeps the same immutable
            # per-step aggregate but deliberately omits reset transcripts.
            if diagnostic_compact_evidence:
                continue
            unsafe_envs = torch.any(
                row["actual_hard_edge_joint_count"].gt(0),
                dim=1,
            )
            for env_id in torch.nonzero(
                unsafe_envs, as_tuple=False
            ).reshape(-1).tolist():
                if (row["sequence"], int(env_id)) not in seen_archive_keys:
                    raise RuntimeError(
                        "joint-safety unsafe env-policy row has no terminal archive"
                    )

        per_step = []
        for row in summary_rows:
            gap = row["minimum_gap"]
            finite_gap = gap[torch.isfinite(gap)]
            per_step.append(
                {
                    "policy_step_sequence": row["sequence"],
                    "identity_sha256": row["identity_sha256"],
                    "complete_env_count": int(row["complete"].sum().item()),
                    "incomplete_env_count": int((~row["complete"]).sum().item()),
                    "minimum_hard_gap_rad": (
                        None
                        if finite_gap.numel() == 0
                        else float(finite_gap.min().item())
                    ),
                    "sparse_counters": {
                        field: {
                            "nonzero_cells": int(row[field].ne(0).sum().item()),
                            "event_count": int(row[field].sum().item()),
                        }
                        for field in aggregate_fields
                    },
                }
            )
        return {
            "ppo_update": step,
            "consume_sequence": consume_sequence,
            "num_envs": num_envs,
            "expected_policy_steps": expected_steps,
            "first_policy_step_sequence": summary_rows[0]["sequence"],
            "last_policy_step_sequence": summary_rows[-1]["sequence"],
            "last_identity": identities[-1],
            "summary_rows": tuple(summary_rows),
            "identities": tuple(identities),
            "per_step": per_step,
            "aggregate": aggregate,
            "minimum_hard_lower_gap": lower_gap,
            "minimum_hard_upper_gap": upper_gap,
            "combined_gap": combined_gap,
            "complete_total": complete_total,
            "archive_count": len(archives),
            "archives": archives,
            "validated_archives": tuple(validated_archives),
            "last_archive_sequence": previous_archive_sequence,
            "contract": contract,
        }

    @staticmethod
    def _joint_safety_value_sha256(value) -> str:
        digest = hashlib.sha256()

        def update(item) -> None:
            if torch.is_tensor(item):
                tensor = item.detach().to(device="cpu").contiguous()
                digest.update(b"tensor\0")
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes(order="C"))
            elif isinstance(item, dict):
                digest.update(b"dict\0")
                for key in sorted(item):
                    update(str(key))
                    update(item[key])
            elif isinstance(item, (tuple, list)):
                digest.update(b"sequence\0")
                digest.update(str(len(item)).encode("ascii"))
                for child in item:
                    update(child)
            elif item is None:
                digest.update(b"none\0")
            elif isinstance(item, bool):
                digest.update(b"true\0" if item else b"false\0")
            elif isinstance(item, int):
                digest.update(f"int:{item}\0".encode("ascii"))
            elif isinstance(item, float):
                digest.update(
                    ("float:" + item.hex() + "\0").encode("ascii")
                )
            elif isinstance(item, str):
                encoded = item.encode("utf-8")
                digest.update(f"str:{len(encoded)}:".encode("ascii"))
                digest.update(encoded)
            else:
                raise RuntimeError(
                    "joint-safety compact artifact cannot hash unsupported "
                    f"type {type(item).__name__}"
                )

        update(value)
        return digest.hexdigest()

    @staticmethod
    def _joint_safety_sparse_coo(value: torch.Tensor) -> dict:
        indices = torch.nonzero(value.ne(0), as_tuple=False).to(
            dtype=torch.int32
        )
        if indices.numel():
            gathered = value[
                indices[:, 0].to(dtype=torch.long),
                indices[:, 1].to(dtype=torch.long),
            ]
        else:
            gathered = torch.empty((0,), dtype=value.dtype)
        if bool(torch.any(gathered.lt(0)).item()):
            raise RuntimeError(
                "joint-safety sparse counter cannot encode a negative value"
            )
        if gathered.numel() and int(gathered.max().item()) <= 255:
            gathered = gathered.to(dtype=torch.uint8)
        else:
            gathered = gathered.to(dtype=torch.long)
        return {
            "index": indices,
            "value": gathered,
            "nonzero_cells": int(indices.shape[0]),
            "event_count": int(gathered.to(dtype=torch.long).sum().item()),
        }

    def _compact_joint_safety_identities(
        self, identities: tuple, identity_sha256: tuple
    ) -> dict:
        """Losslessly retain identity/generation with sparse reset transitions."""

        if not identities:
            raise RuntimeError("joint-safety compact identity has no policy steps")
        num_envs = int(identities[0]["episode_length"].numel())
        episode_length = torch.stack(
            [identity["episode_length"] for identity in identities], dim=0
        )
        swing_generation = torch.stack(
            [identity["swing_generation"] for identity in identities], dim=0
        )
        int32_min = -(2**31)
        int32_max = 2**31 - 1
        for name, value in (
            ("episode_length", episode_length),
            ("swing_generation", swing_generation),
        ):
            if bool(
                torch.any(value.lt(int32_min) | value.gt(int32_max)).item()
            ):
                raise RuntimeError(
                    f"joint-safety {name} exceeds compact int32 range"
                )

        sparse_changes = {}
        for field in (
            "action_episode_sequence",
            "action_uid",
            "birth_generation",
        ):
            stacked = torch.stack(
                [identity[field] for identity in identities], dim=0
            )
            changed = stacked[1:].ne(stacked[:-1])
            index = torch.nonzero(changed, as_tuple=False)
            if index.numel():
                index[:, 0] += 1
                values = stacked[
                    index[:, 0].to(dtype=torch.long),
                    index[:, 1].to(dtype=torch.long),
                ].clone()
            else:
                values = torch.empty((0,), dtype=torch.long)
            sparse_changes[field] = {
                "index": index.to(dtype=torch.int32),
                "value": values,
            }

        initial_receipts = identities[0]["birth_receipt_sha256"]
        receipt_changes = []
        previous_receipts = initial_receipts
        for step_index, identity in enumerate(identities[1:], start=1):
            current_receipts = identity["birth_receipt_sha256"]
            for env_id, (before, after) in enumerate(
                zip(previous_receipts, current_receipts)
            ):
                if before != after:
                    receipt_changes.append((step_index, env_id, after))
            previous_receipts = current_receipts
        return {
            "encoding": (
                "initial_full_plus_sparse_reset_birth_changes_and_dense_"
                "episode_length_swing_generation"
            ),
            "num_envs": num_envs,
            "action_ball_enabled": identities[0]["action_ball_enabled"],
            "initial": {
                field: identities[0][field].clone()
                for field in (
                    "action_episode_sequence",
                    "action_uid",
                    "birth_generation",
                )
            },
            "episode_length_int32": episode_length.to(dtype=torch.int32),
            "swing_generation_int32": swing_generation.to(dtype=torch.int32),
            "changes": sparse_changes,
            "initial_birth_receipt_sha256": initial_receipts,
            "birth_receipt_changes": tuple(receipt_changes),
            "per_step_identity_sha256": identity_sha256,
        }

    def _compact_joint_safety_artifact(
        self, validated: dict, *, step: int, rank: int
    ) -> dict:
        """Build the bounded v2 sidecar; omit the duplicate live batch transcript."""

        aggregate_fields = (
            "qdes_joint_count",
            "policy_crossing_joint_count",
            "substep_hard_crossing_joint_count",
            "actual_hard_edge_joint_count",
        )
        compact_steps = []
        for row, identity in zip(
            validated["summary_rows"], validated["identities"]
        ):
            gap = row["minimum_gap"]
            flat_index = int(torch.argmin(gap.reshape(-1)).item())
            env_id = flat_index // gap.shape[1]
            joint_id = flat_index % gap.shape[1]
            action_uids = identity["action_uid"]
            unique_action_uids = torch.unique(action_uids, sorted=True)
            per_action_minimum = []
            for action_uid in unique_action_uids.tolist():
                mask = action_uids.eq(int(action_uid))
                per_action_minimum.append(gap[mask].amin(dim=0))
            compact_steps.append(
                {
                    "policy_step_sequence": row["sequence"],
                    "policy_start_timestamp_s": row[
                        "policy_start_timestamp_s"
                    ],
                    "identity_sha256": row["identity_sha256"],
                    "minimum_hard_gap_rad": float(
                        gap.reshape(-1)[flat_index].item()
                    ),
                    "minimum_hard_gap_env_id": env_id,
                    "minimum_hard_gap_joint_id": joint_id,
                    "per_action_minimum_hard_gap": {
                        "action_uid": unique_action_uids.clone(),
                        "minimum_gap_rad": torch.stack(
                            per_action_minimum, dim=0
                        ),
                    },
                    "sparse_counters": {
                        field: self._joint_safety_sparse_coo(row[field])
                        for field in aggregate_fields
                    },
                }
            )

        terminal_entries = []
        for item in validated["validated_archives"]:
            archive = item["archive"]
            if item["retain_full_transcript"]:
                terminal_entries.append(
                    {
                        "storage": "full_forensic",
                        "archive": archive,
                    }
                )
                continue
            transcript = archive["transcript"]
            compact_transcript = {
                "storage": "validated_sha256_compact",
                "source_sha256": self._joint_safety_value_sha256(transcript),
                "schema_version": transcript["schema_version"],
                "policy_step_sequence": transcript[
                    "policy_step_sequence"
                ],
                "policy_start_timestamp_s": transcript[
                    "policy_start_timestamp_s"
                ],
                "record_count": transcript["record_count"],
                "record_kind": transcript["record_kind"],
                "call_index": transcript["call_index"],
                "timestamp_s": transcript["timestamp_s"],
                "step_qdes_joint_count": transcript[
                    "step_qdes_joint_count"
                ],
                "step_policy_crossing_joint_count": transcript[
                    "step_policy_crossing_joint_count"
                ],
                "substep_crossing_joint_count": transcript[
                    "substep_crossing_joint_count"
                ],
                "substep_actual_joint_count": transcript[
                    "substep_actual_joint_count"
                ],
            }
            terminal_entries.append(
                {
                    "storage": "compact_timeout_or_nonterminated_reset",
                    "archive": {
                        **{
                            key: value
                            for key, value in archive.items()
                            if key not in {"transcript", "payload_bytes"}
                        },
                        "source_payload_bytes": archive["payload_bytes"],
                        "transcript": compact_transcript,
                    },
                }
            )

        lower = validated["minimum_hard_lower_gap"]
        upper = validated["minimum_hard_upper_gap"]
        lower_min, lower_argmin = lower.min(dim=0)
        upper_min, upper_argmin = upper.min(dim=0)
        actual_count = int(
            validated["aggregate"]["actual_hard_edge_joint_count"]
            .sum()
            .item()
        )
        nonpositive_gap_count = int(
            validated["combined_gap"].le(0.0).sum().item()
        )
        fatal = actual_count > 0 or nonpositive_gap_count > 0
        contract = validated["contract"]
        contract_artifact = {
            key: value
            for key, value in contract.items()
            if key not in {"hard_lower", "hard_upper"}
        }
        contract_artifact["hard_lower"] = contract["hard_lower"].clone()
        contract_artifact["hard_upper"] = contract["hard_upper"].clone()
        core = {
            "event": _JOINT_SAFETY_EVENT,
            "schema_version": _JOINT_SAFETY_ARTIFACT_SCHEMA_VERSION,
            "status": (
                "fatal_actual_hard_edge"
                if fatal
                else "prepared_before_optimizer"
            ),
            "rank": rank,
            "ppo_update": step,
            "contract": contract_artifact,
            "sequence": {
                "consume_sequence": validated["consume_sequence"],
                "first_policy_step_sequence": validated[
                    "first_policy_step_sequence"
                ],
                "last_policy_step_sequence": validated[
                    "last_policy_step_sequence"
                ],
                "last_archive_sequence": validated[
                    "last_archive_sequence"
                ],
            },
            "completeness": {
                "all_rows_present": True,
                "all_policy_steps_complete": True,
                "expected_apply_readbacks": contract[
                    "expected_apply_calls"
                ],
                "expected_post_readbacks": 1,
                "timestamp_invariant": True,
            },
            "identity": self._compact_joint_safety_identities(
                validated["identities"],
                tuple(
                    row["identity_sha256"]
                    for row in validated["summary_rows"]
                ),
            ),
            "policy_steps": tuple(compact_steps),
            "aggregate_sparse_counters": {
                field: self._joint_safety_sparse_coo(
                    validated["aggregate"][field]
                )
                for field in aggregate_fields
            },
            "gaps": {
                "minimum_lower_gap_by_joint": lower_min,
                "minimum_lower_gap_env_id_by_joint": lower_argmin.to(
                    dtype=torch.int32
                ),
                "minimum_upper_gap_by_joint": upper_min,
                "minimum_upper_gap_env_id_by_joint": upper_argmin.to(
                    dtype=torch.int32
                ),
            },
            "fatal_flags": {
                "actual_hard_edge_event_count": actual_count,
                "nonpositive_physical_hard_gap_cell_count": (
                    nonpositive_gap_count
                ),
            },
        }
        terminal = {
            "archive_count": len(terminal_entries),
            "entries": tuple(terminal_entries),
        }
        core_bytes = self._joint_safety_payload_bytes(core)
        terminal_bytes = self._joint_safety_payload_bytes(terminal)
        return {
            **core,
            "terminal": terminal,
            "budgets": {
                "core_payload_bytes": core_bytes,
                "terminal_payload_bytes": terminal_bytes,
                "total_payload_bytes": core_bytes + terminal_bytes,
                "core_payload_max_bytes": (
                    _JOINT_SAFETY_CORE_PAYLOAD_MAX_BYTES
                ),
                "terminal_payload_max_bytes": (
                    _JOINT_SAFETY_TERMINAL_PAYLOAD_MAX_BYTES
                ),
                "normal_serialized_max_bytes": (
                    _JOINT_SAFETY_NORMAL_ARTIFACT_MAX_BYTES
                ),
                "forensic_serialized_max_bytes": (
                    _JOINT_SAFETY_FORENSIC_ARTIFACT_MAX_BYTES
                ),
            },
        }

    @staticmethod
    def _joint_safety_cpu_clone(value):
        if torch.is_tensor(value):
            # prepare_joint_safety_ledger_consume() returns a private detached export.  A CPU
            # tensor is therefore already runner-owned; cloning the 4096 x 24 summaries again
            # would double boundary memory for no isolation gain.  CUDA->CPU still materializes
            # one independent transfer.
            return value.detach().to(device="cpu")
        if isinstance(value, dict):
            return {
                key: MotionOnPolicyRunner._joint_safety_cpu_clone(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                MotionOnPolicyRunner._joint_safety_cpu_clone(item)
                for item in value
            )
        if isinstance(value, list):
            return [
                MotionOnPolicyRunner._joint_safety_cpu_clone(item)
                for item in value
            ]
        return value

    def _persist_joint_safety_update(
        self, payload: dict, *, step: int
    ) -> dict:
        """Durably publish one validated compact receipt before the optimizer."""

        raw_root = getattr(self, "log_dir", None)
        if not isinstance(raw_root, (str, os.PathLike)) or not str(raw_root):
            raise RuntimeError(
                "joint-safety evidence requires a non-empty runner log_dir"
            )
        root = pathlib.Path(raw_root)
        directory = root / "joint_safety_ledgers"
        directory.mkdir(parents=True, exist_ok=True)
        rank = getattr(self, "gpu_global_rank", None)
        if rank is None:
            rank = getattr(self, "rank", 0)
        rank = self._joint_safety_int(rank, name="runner rank", minimum=0)
        filename = (
            f"ppo_update_{step:08d}_rank_{rank:04d}.prepared.pt"
        )
        final_path = directory / filename
        if final_path.exists():
            raise RuntimeError(
                f"joint-safety evidence path already exists: {final_path}"
            )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=str(directory)
        )
        temporary_path = pathlib.Path(temporary_name)
        core_bytes = payload.get("budgets", {}).get("core_payload_bytes")
        terminal_bytes = payload.get("budgets", {}).get(
            "terminal_payload_bytes"
        )
        if (
            type(core_bytes) is not int
            or type(terminal_bytes) is not int
            or core_bytes < 0
            or terminal_bytes < 0
            or core_bytes > _JOINT_SAFETY_CORE_PAYLOAD_MAX_BYTES
            or terminal_bytes > _JOINT_SAFETY_TERMINAL_PAYLOAD_MAX_BYTES
        ):
            os.close(fd)
            temporary_path.unlink()
            raise RuntimeError(
                "joint-safety compact evidence exceeded its pre-registered "
                "core/terminal payload budget"
            )
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            digest = hashlib.sha256()
            size_bytes = 0
            with temporary_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    digest.update(chunk)
            has_full_forensic = any(
                entry.get("storage") == "full_forensic"
                for entry in payload.get("terminal", {}).get("entries", ())
            )
            serialized_limit = (
                _JOINT_SAFETY_FORENSIC_ARTIFACT_MAX_BYTES
                if has_full_forensic
                or payload.get("status") == "fatal_actual_hard_edge"
                else _JOINT_SAFETY_NORMAL_ARTIFACT_MAX_BYTES
            )
            if size_bytes > serialized_limit:
                raise RuntimeError(
                    "joint-safety compact evidence exceeded its pre-registered "
                    f"serialized budget ({size_bytes} > {serialized_limit})"
                )
            try:
                os.link(str(temporary_path), str(final_path))
            except FileExistsError as exc:
                raise RuntimeError(
                    f"joint-safety evidence path already exists: {final_path}"
                ) from exc
            self._joint_safety_fsync_directory(directory)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return {
            "format": "torch_save_cpu",
            "schema_version": _JOINT_SAFETY_ARTIFACT_SCHEMA_VERSION,
            "path": str(final_path.relative_to(root)),
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
            "status": payload["status"],
        }

    @staticmethod
    def _joint_safety_fsync_directory(directory: pathlib.Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(str(directory), flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _preflight_reward_evidence_update_paths(
        self, *, step: int, rank: int
    ) -> pathlib.Path:
        raw_root = getattr(self, "log_dir", None)
        if not isinstance(raw_root, (str, os.PathLike)) or not str(raw_root):
            raise RuntimeError(
                "Reward evidence requires a non-empty runner log_dir"
            )
        directory = pathlib.Path(raw_root) / "reward_evidence_ledgers"
        directory.mkdir(parents=True, exist_ok=True)
        paths = (
            directory
            / f"ppo_update_{step:08d}_rank_{rank:04d}.prepared.json",
            directory
            / (
                f"ppo_update_{step:08d}_rank_{rank:04d}."
                "optimizer_commit.json"
            ),
        )
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise RuntimeError(
                "Reward evidence no-clobber path already exists: "
                + ", ".join(existing)
            )
        return directory

    @staticmethod
    def _require_action_ball_conservation_pass(
        prepared_records: Mapping[str, object], *, step: int
    ) -> Mapping[str, object]:
        """Return the public compact closure or stop before PPO mutation."""

        if not isinstance(prepared_records, Mapping):
            raise RuntimeError(
                "ActionBall optimizer is fenced: Reward evidence is unavailable"
            )
        receipt = prepared_records.get("action_ball_conservation")
        activation = prepared_records.get("activation")
        required_checks = (
            "all_step_reward_buf_equals_all_term_sums",
            "all_episode_sums_equal_captured_term_sums",
            "all_reset_episode_sums_cleared",
            "exact_environment_step_coverage",
        )
        checks = receipt.get("checks") if isinstance(receipt, Mapping) else None
        dashboard = (
            receipt.get("dashboard_normalization")
            if isinstance(receipt, Mapping)
            else None
        )
        completed = (
            receipt.get("completed_episode_segments")
            if isinstance(receipt, Mapping)
            else None
        )
        open_segments = (
            receipt.get("open_episode_segments")
            if isinstance(receipt, Mapping)
            else None
        )
        if (
            not isinstance(receipt, Mapping)
            or prepared_records.get("status")
            != "frozen_validated_before_optimizer"
            or receipt.get("event")
            != "hope_reward_episode_segmented_closure_update"
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "PASS"
            or receipt.get("evidence_source")
            != "live_isaac_reward_manager"
            or receipt.get("capture_mode")
            != "reward_manager_reset_pre_clear_hook"
            or receipt.get("task_kind") != "action_ball"
            or receipt.get("ppo_update") != step
            or not isinstance(activation, Mapping)
            or receipt.get("recipe_sha256")
            != activation.get("recipe_sha256")
            or receipt.get("step_dt_s") != activation.get("step_dt_s")
            or receipt.get("num_envs") != activation.get("num_envs")
            or receipt.get("segment_key_fields")
            != ["env_id", "reset_generation"]
            or not isinstance(
                receipt.get("all_reward_manager_term_names"), list
            )
            or not receipt.get("all_reward_manager_term_names")
            or not isinstance(completed, list)
            or receipt.get("completed_episode_count") != len(completed)
            or not isinstance(open_segments, list)
            or receipt.get("open_episode_count") != len(open_segments)
            or len(open_segments) != receipt.get("num_envs")
            or not isinstance(receipt.get("reset_batches"), list)
            or not isinstance(dashboard, Mapping)
            or dashboard.get("status")
            not in {"PASS", "NOT_OBSERVED_NO_RESET"}
            or type(receipt.get("e2_eligible")) is not bool
            or not isinstance(checks, Mapping)
            or checks.get("status") != "PASS"
            or checks.get("environment_step_count")
            != activation.get("environment_step_count")
            or any(checks.get(name) != "PASS" for name in required_checks)
        ):
            status = (
                receipt.get("status")
                if isinstance(receipt, Mapping)
                else "unavailable"
            )
            raise RuntimeError(
                "ActionBall optimizer is fenced: action_ball_conservation "
                f"is not a source-bound PASS receipt (status={status!r})"
            )
        try:
            json.dumps(
                receipt,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ActionBall optimizer is fenced: conservation receipt is "
                "not finite canonical JSON data"
            ) from exc
        return receipt

    def _persist_reward_evidence_update(
        self,
        prepared_records: dict,
        *,
        step: int,
        rank: int,
        task_kind: str,
    ) -> dict:
        """Fsync the exact Reward records before the optimizer may run."""

        directory = self._preflight_reward_evidence_update_paths(
            step=step, rank=rank
        )
        action_ball_conservation = (
            self._require_action_ball_conservation_pass(
                prepared_records, step=step
            )
            if task_kind == "action_ball"
            else {"status": "NOT_APPLICABLE"}
        )
        payload = {
            "event": _REWARD_EVIDENCE_ARTIFACT_EVENT,
            "schema_version": _REWARD_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            "status": "prepared_before_optimizer",
            "ppo_update": step,
            "rank": rank,
            "task_kind": task_kind,
            "activation": prepared_records["activation"],
            "per_action": prepared_records.get("per_action"),
            "safety": prepared_records.get("safety"),
            "action_ball_conservation": action_ball_conservation,
        }
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > _REWARD_EVIDENCE_ARTIFACT_MAX_BYTES:
            raise RuntimeError(
                "Reward evidence exceeded its pre-registered serialized "
                f"budget ({len(encoded)} > "
                f"{_REWARD_EVIDENCE_ARTIFACT_MAX_BYTES})"
            )
        filename = (
            f"ppo_update_{step:08d}_rank_{rank:04d}.prepared.json"
        )
        final_path = directory / filename
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=str(directory)
        )
        temporary_path = pathlib.Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(str(temporary_path), str(final_path))
            except FileExistsError as exc:
                raise RuntimeError(
                    f"Reward evidence path already exists: {final_path}"
                ) from exc
            self._joint_safety_fsync_directory(directory)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        root = pathlib.Path(str(getattr(self, "log_dir")))
        return {
            "format": "canonical_json",
            "schema_version": _REWARD_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            "path": str(final_path.relative_to(root)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            "status": payload["status"],
            "action_ball_conservation_status": payload[
                "action_ball_conservation"
            ]["status"],
        }

    def _persist_reward_evidence_optimizer_commit(
        self, *, step: int, rank: int, artifact: dict
    ) -> dict:
        """Record optimizer success before either Reward ledger is acknowledged."""

        root = pathlib.Path(str(getattr(self, "log_dir")))
        directory = root / "reward_evidence_ledgers"
        filename = (
            f"ppo_update_{step:08d}_rank_{rank:04d}."
            "optimizer_commit.json"
        )
        final_path = directory / filename
        if final_path.exists():
            raise RuntimeError(
                f"Reward optimizer commit path already exists: {final_path}"
            )
        marker = {
            "event": _REWARD_EVIDENCE_COMMIT_EVENT,
            "schema_version": 1,
            "ppo_update": step,
            "rank": rank,
            "prepared_artifact_path": artifact["path"],
            "prepared_artifact_sha256": artifact["sha256"],
            "status": "optimizer_succeeded_pending_reward_ledger_ack",
        }
        encoded = (
            json.dumps(marker, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=str(directory)
        )
        temporary_path = pathlib.Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(str(temporary_path), str(final_path))
            except FileExistsError as exc:
                raise RuntimeError(
                    "Reward optimizer commit path already exists: "
                    f"{final_path}"
                ) from exc
            self._joint_safety_fsync_directory(directory)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return {
            "format": "canonical_json",
            "schema_version": 1,
            "path": str(final_path.relative_to(root)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        }

    @staticmethod
    def _emit_reward_evidence_update(
        prepared_records: dict,
        *,
        artifact: dict,
        optimizer_commit: dict,
        encoder,
    ) -> None:
        shared = {
            "status": "optimizer_committed_and_ledger_acknowledged",
            "artifact": artifact,
            "optimizer_commit": optimizer_commit,
        }
        outputs = (
            (
                "HOPE_EFFECTIVE_REWARD_ACTIVATION_UPDATE_JSON=",
                prepared_records.get("activation"),
            ),
            (
                "HOPE_EFFECTIVE_REWARD_BY_ACTION_UPDATE_JSON=",
                prepared_records.get("per_action"),
            ),
            (
                "HOPE_REWARD_SAFETY_TRANSITION_UPDATE_JSON=",
                prepared_records.get("safety"),
            ),
        )
        for prefix, record in outputs:
            if record is not None:
                print(prefix + encoder({**record, **shared}), flush=True)
        conservation = prepared_records.get("action_ball_conservation")
        if conservation is not None:
            print(
                "HOPE_REWARD_EPISODE_SEGMENTED_CLOSURE_UPDATE_JSON="
                + encoder(
                    {
                        **conservation,
                        "optimizer_transaction": shared,
                    }
                ),
                flush=True,
            )

    def _persist_joint_safety_optimizer_commit(
        self, prepared: dict
    ) -> dict:
        """Publish optimizer success before destructive ledger acknowledgement."""

        artifact = prepared["artifact"]
        step = prepared["step"]
        raw_root = pathlib.Path(str(getattr(self, "log_dir")))
        directory = raw_root / "joint_safety_ledgers"
        rank = prepared["rank"]
        filename = (
            f"ppo_update_{step:08d}_rank_{rank:04d}.optimizer_commit.json"
        )
        final_path = directory / filename
        if final_path.exists():
            raise RuntimeError(
                f"joint-safety optimizer commit path already exists: {final_path}"
            )
        marker = {
            "event": _JOINT_SAFETY_COMMIT_EVENT,
            "schema_version": 1,
            "ppo_update": step,
            "rank": rank,
            "prepared_artifact_path": artifact["path"],
            "prepared_artifact_sha256": artifact["sha256"],
            "consume_sequence": prepared["validated"]["consume_sequence"],
            "status": "optimizer_succeeded_pending_ledger_ack",
        }
        encoded = (
            json.dumps(marker, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=str(directory)
        )
        temporary_path = pathlib.Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(str(temporary_path), str(final_path))
            except FileExistsError as exc:
                raise RuntimeError(
                    "joint-safety optimizer commit path already exists: "
                    f"{final_path}"
                ) from exc
            self._joint_safety_fsync_directory(directory)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        marker_artifact = {
            "format": "canonical_json",
            "schema_version": 1,
            "path": str(final_path.relative_to(raw_root)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        }
        return marker_artifact

    def _joint_safety_rank(self) -> int:
        rank = getattr(self, "gpu_global_rank", None)
        if rank is None:
            rank = getattr(self, "rank", 0)
        return self._joint_safety_int(
            rank, name="runner rank", minimum=0
        )

    def _preflight_joint_safety_update_paths(
        self, *, step: int, rank: int
    ) -> None:
        raw_root = getattr(self, "log_dir", None)
        if not isinstance(raw_root, (str, os.PathLike)) or not str(raw_root):
            raise RuntimeError(
                "joint-safety evidence requires a non-empty runner log_dir"
            )
        directory = pathlib.Path(raw_root) / "joint_safety_ledgers"
        directory.mkdir(parents=True, exist_ok=True)
        paths = (
            directory
            / f"ppo_update_{step:08d}_rank_{rank:04d}.prepared.pt",
            directory
            / (
                f"ppo_update_{step:08d}_rank_{rank:04d}."
                "optimizer_commit.json"
            ),
        )
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise RuntimeError(
                "joint-safety no-clobber path already exists: "
                + ", ".join(existing)
            )

    def _prepare_diagnostic_joint_safety_update(
        self, step: int, *, expected_action_term=None
    ) -> dict:
        """Freeze and validate one non-promotable device aggregate.

        Diagnostic ActionBall and Stage-1 keep the exact clamp, receding brake,
        fresh q/qdot readbacks, physical hard-edge termination and every
        per-joint counter.  They deliberately do not retain identity-bound
        dense transcripts or publish formal receipt files on every PPO update.
        """

        if not self._diagnostic_joint_safety_compact_update_evidence():
            raise RuntimeError(
                "compact diagnostic joint-safety path requires an unauthorized "
                "task with an explicitly compact producer"
            )
        step = self._joint_safety_int(step, name="PPO update", minimum=0)
        prior_step = getattr(self, "_joint_safety_consumed_step", None)
        if prior_step is not None and step != int(prior_step) + 1:
            raise RuntimeError(
                "diagnostic joint-safety PPO update sequence is not contiguous"
            )
        if getattr(self, "_joint_safety_pending_prepared", None) is not None:
            raise RuntimeError(
                "diagnostic joint-safety has an unacknowledged prepared update"
            )
        term = self._bind_joint_safety_action_term(required=True)
        if expected_action_term is not None and term is not expected_action_term:
            raise RuntimeError(
                "diagnostic joint-safety action term changed after launch binding"
            )
        token, snapshot = term.prepare_joint_safety_ledger_consume()
        prepared = {
            "step": step,
            "term": term,
            "token": token,
            "status": "diagnostic_compact_snapshot_frozen",
        }
        self._joint_safety_pending_prepared = prepared
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("enabled") is not True
            or snapshot.get("diagnostic_compact_evidence") is not True
        ):
            raise RuntimeError(
                "diagnostic joint-safety producer did not expose compact evidence"
            )
        if snapshot.get("terminal_archives") != ():
            raise RuntimeError(
                "diagnostic joint-safety compact evidence retained terminal transcripts"
            )
        if snapshot.get("identity_bound_policy_steps") != ():
            raise RuntimeError(
                "diagnostic joint-safety compact evidence retained per-step identities"
            )
        for prefix in ("policy_step_summary", "terminal_archive"):
            if (
                snapshot.get(f"{prefix}_used") != 0
                or snapshot.get(f"{prefix}_overflow_latch") is not False
                or snapshot.get(f"{prefix}_overflow_count") != 0
            ):
                raise RuntimeError(
                    f"diagnostic joint-safety {prefix} state is not compact/clean"
                )

        since = snapshot.get("since_last_consume")
        if not isinstance(since, dict) or since.get("has_data") is not True:
            raise RuntimeError(
                "diagnostic joint-safety PPO update has no device aggregate"
            )
        consume_sequence = self._joint_safety_int(
            since.get("consume_sequence"),
            name="diagnostic consume_sequence",
            minimum=0,
        )
        expected_steps = self._joint_safety_int(
            getattr(self, "num_steps_per_env", None),
            name="runner num_steps_per_env",
            minimum=1,
        )
        policy_steps_raw = since.get("policy_step_count")
        if not torch.is_tensor(policy_steps_raw) or policy_steps_raw.ndim != 1:
            raise RuntimeError(
                "diagnostic joint-safety policy_step_count must be one-dimensional"
            )
        num_envs = int(policy_steps_raw.numel())
        if num_envs <= 0:
            raise RuntimeError(
                "diagnostic joint-safety cannot have zero environments"
            )
        contract = self._joint_safety_runtime_contract(term)
        joint_count = self._joint_safety_int(
            contract.get("joint_count"),
            name="runtime joint_count",
            minimum=1,
        )
        expected_apply_calls = self._joint_safety_int(
            contract.get("expected_apply_calls"),
            name="runtime expected_apply_calls",
            minimum=1,
        )
        env_shape = (num_envs,)
        joint_shape = (num_envs, joint_count)

        def env_count(name: str) -> torch.Tensor:
            return self._joint_safety_tensor(
                since.get(name), name=name, shape=env_shape, integer=True
            )

        def joint_count_tensor(name: str) -> torch.Tensor:
            return self._joint_safety_tensor(
                since.get(name), name=name, shape=joint_shape, integer=True
            )

        policy_steps = env_count("policy_step_count")
        complete_steps = env_count("complete_policy_step_count")
        incomplete_steps = env_count("incomplete_policy_step_count")
        apply_readbacks = env_count("apply_readback_count")
        post_readbacks = env_count("post_readback_count")
        timestamp_passes = env_count("timestamp_invariant_pass_count")
        qdes_counts = joint_count_tensor("qdes_joint_count")
        policy_crossing_counts = joint_count_tensor(
            "policy_crossing_joint_count"
        )
        substep_crossing_counts = joint_count_tensor(
            "substep_hard_crossing_joint_count"
        )
        actual_hard_counts = joint_count_tensor(
            "actual_hard_edge_joint_count"
        )
        minimum_lower = self._joint_safety_tensor(
            since.get("minimum_hard_lower_gap"),
            name="minimum_hard_lower_gap",
            shape=joint_shape,
        )
        minimum_upper = self._joint_safety_tensor(
            since.get("minimum_hard_upper_gap"),
            name="minimum_hard_upper_gap",
            shape=joint_shape,
        )
        hard_latch = self._joint_safety_tensor(
            since.get("hard_crossing_latch"),
            name="hard_crossing_latch",
            shape=env_shape,
            boolean=True,
        )
        actual_latch = self._joint_safety_tensor(
            since.get("actual_hard_edge_latch"),
            name="actual_hard_edge_latch",
            shape=env_shape,
            boolean=True,
        )

        actual_edge_from_minimum_gap = minimum_lower.le(0.0) | minimum_upper.le(
            0.0
        )
        checks = torch.stack(
            (
                torch.all(policy_steps.eq(expected_steps)),
                torch.all(complete_steps.eq(expected_steps)),
                torch.all(incomplete_steps.eq(0)),
                torch.all(
                    apply_readbacks.eq(
                        expected_steps * expected_apply_calls
                    )
                ),
                torch.all(post_readbacks.eq(expected_steps)),
                torch.all(timestamp_passes.eq(expected_steps)),
                torch.all(torch.isfinite(minimum_lower)),
                torch.all(torch.isfinite(minimum_upper)),
                torch.all(
                    hard_latch.eq(
                        torch.any(substep_crossing_counts.gt(0), dim=1)
                    )
                ),
                torch.all(
                    actual_latch.eq(
                        torch.any(actual_hard_counts.gt(0), dim=1)
                    )
                ),
                torch.all(actual_hard_counts.ge(0)),
                torch.all(
                    actual_hard_counts.gt(0).eq(
                        actual_edge_from_minimum_gap
                    )
                ),
            )
        )
        aggregate_values = torch.stack(
            (
                policy_steps.sum(),
                complete_steps.sum(),
                apply_readbacks.sum(),
                post_readbacks.sum(),
                timestamp_passes.sum(),
                qdes_counts.sum(),
                policy_crossing_counts.sum(),
                substep_crossing_counts.sum(),
                actual_hard_counts.sum(),
            )
        ).to(dtype=torch.float64)
        minimum_gap = torch.minimum(minimum_lower, minimum_upper).amin().reshape(1)
        # This is the sole compact-evidence D2H.  Keep the already-produced
        # env-by-joint count and side-specific minima in the same packed copy
        # as the aggregate checks/totals; never re-read the environment or
        # simulator to localize an edge after the producer is frozen.
        localization_values = torch.cat(
            (
                actual_hard_counts.reshape(-1).to(dtype=torch.float64),
                minimum_lower.reshape(-1).to(dtype=torch.float64),
                minimum_upper.reshape(-1).to(dtype=torch.float64),
            )
        )
        packed = torch.cat(
            (
                checks.to(dtype=torch.float64),
                aggregate_values,
                minimum_gap.to(dtype=torch.float64),
                localization_values,
            )
        ).detach().to(device="cpu").tolist()
        check_names = (
            "policy_step_count",
            "complete_policy_step_count",
            "incomplete_policy_step_count",
            "apply_readback_count",
            "post_readback_count",
            "timestamp_invariant_pass_count",
            "minimum_hard_lower_gap",
            "minimum_hard_upper_gap",
            "hard_crossing_latch",
            "actual_hard_edge_latch",
            "actual_hard_edge_joint_count_nonnegative",
            "actual_hard_edge_count_gap_equivalence",
        )
        failed = [
            name
            for name, value in zip(check_names, packed[: len(check_names)])
            if value != 1.0
        ]
        if failed:
            raise RuntimeError(
                "diagnostic joint-safety aggregate invariant failed: "
                + ", ".join(failed)
            )
        first_sequence = self._joint_safety_int(
            snapshot.get("diagnostic_first_policy_step_sequence"),
            name="diagnostic first policy-step sequence",
            minimum=0,
        )
        last_sequence = self._joint_safety_int(
            snapshot.get("diagnostic_last_policy_step_sequence"),
            name="diagnostic last policy-step sequence",
            minimum=0,
        )
        if last_sequence - first_sequence + 1 != expected_steps:
            raise RuntimeError(
                "diagnostic joint-safety sequence span does not match rollout length"
            )
        previous_sequence = getattr(
            self, "_joint_safety_last_policy_step_sequence", None
        )
        if (
            previous_sequence is not None
            and first_sequence != int(previous_sequence) + 1
        ):
            raise RuntimeError(
                "diagnostic joint-safety sequence is discontinuous across PPO updates"
            )
        totals = packed[
            len(check_names) : len(check_names) + len(aggregate_values)
        ]
        total_names = (
            "policy_steps",
            "complete_policy_steps",
            "apply_readbacks",
            "post_readbacks",
            "timestamp_passes",
            "qdes_events",
            "policy_crossing_events",
            "substep_crossing_events",
            "actual_hard_edge_events",
        )
        minimum_gap_index = len(check_names) + len(total_names)
        localization_offset = minimum_gap_index + 1
        localization_cell_count = num_envs * joint_count
        localization_end = localization_offset + 3 * localization_cell_count
        if len(packed) != localization_end:
            raise RuntimeError(
                "diagnostic joint-safety localization packed length differs"
            )
        localized_counts = packed[
            localization_offset : localization_offset
            + localization_cell_count
        ]
        localized_lower = packed[
            localization_offset
            + localization_cell_count : localization_offset
            + 2 * localization_cell_count
        ]
        localized_upper = packed[
            localization_offset
            + 2 * localization_cell_count : localization_end
        ]
        joint_names = contract.get("joint_names")
        if (
            type(joint_names) is not tuple
            or len(joint_names) != joint_count
            or any(type(name) is not str or not name for name in joint_names)
            or len(set(joint_names)) != joint_count
        ):
            raise RuntimeError(
                "diagnostic joint-safety localization lost exact joint order"
            )
        localization_rows = []
        side_names = ("lower", "upper")
        for env_row in range(num_envs):
            for joint_index, joint_name in enumerate(joint_names):
                flat_index = env_row * joint_count + joint_index
                raw_count = localized_counts[flat_index]
                count = int(raw_count)
                if (
                    isinstance(raw_count, bool)
                    or not math.isfinite(float(raw_count))
                    or float(count) != float(raw_count)
                    or count < 0
                ):
                    raise RuntimeError(
                        "diagnostic joint-safety localization count is invalid"
                    )
                side_gaps = (
                    float(localized_lower[flat_index]),
                    float(localized_upper[flat_index]),
                )
                if any(not math.isfinite(gap) for gap in side_gaps):
                    raise RuntimeError(
                        "diagnostic joint-safety localization gap is non-finite"
                    )
                observed_sides = tuple(
                    side
                    for side, gap in zip(side_names, side_gaps)
                    if gap <= 0.0
                )
                if (count > 0) != bool(observed_sides):
                    raise RuntimeError(
                        "diagnostic joint-safety localization count/gap differs"
                    )
                for side, gap in zip(side_names, side_gaps):
                    if gap > 0.0:
                        continue
                    localization_rows.append(
                        {
                            "env_row": env_row,
                            "joint_index": joint_index,
                            "joint_name": joint_name,
                            "side": side,
                            "joint_readback_count_either_side": count,
                            "minimum_signed_hard_gap_rad": gap,
                        }
                    )
        record = {
            "event": "hope_joint_safety_diagnostic_compact_update",
            "schema_version": 1,
            "status": "diagnostic_compact_prepared_before_optimizer",
            "ppo_update": step,
            "consume_sequence": consume_sequence,
            "num_envs": num_envs,
            "policy_step_count": expected_steps,
            "first_policy_step_sequence": first_sequence,
            "last_policy_step_sequence": last_sequence,
            "counter_totals": {
                name: int(value)
                for name, value in zip(total_names, totals)
            },
            "minimum_hard_gap_rad": float(packed[minimum_gap_index]),
            "actual_hard_edge_localization": {
                "kind": "joint_safety_actual_hard_edge_localization_v1",
                "schema_version": 1,
                "count_semantics": (
                    "per-env-joint readbacks at either mechanical side; "
                    "repeated on each observed side and not a per-side count"
                ),
                "joint_order": list(joint_names),
                "side_order": list(side_names),
                "rows": localization_rows,
            },
            "terminal_archive_count": 0,
            "identity_bound_policy_step_count": 0,
            "formal_authority": False,
        }
        prepared.update(
            {
                "status": "diagnostic_compact_prepared_before_optimizer",
                "record": record,
                "first_policy_step_sequence": first_sequence,
                "last_policy_step_sequence": last_sequence,
                "consume_sequence": consume_sequence,
            }
        )
        return prepared

    def _commit_diagnostic_joint_safety_update(
        self, prepared: dict
    ) -> dict:
        """Acknowledge one compact diagnostic aggregate after PPO succeeds."""

        if (
            prepared is not getattr(
                self, "_joint_safety_pending_prepared", None
            )
            or prepared.get("status")
            != "diagnostic_compact_prepared_before_optimizer"
        ):
            raise RuntimeError(
                "diagnostic joint-safety optimizer commit does not own the pending aggregate"
            )
        term = self._bind_joint_safety_action_term(required=True)
        if term is not prepared["term"]:
            raise RuntimeError(
                "diagnostic joint-safety action term changed before acknowledgement"
            )
        term.acknowledge_joint_safety_ledger(prepared["token"])
        record = {
            **prepared["record"],
            "status": (
                "diagnostic_compact_optimizer_committed_and_ledger_acknowledged"
            ),
        }
        print(
            "HOPE_JOINT_SAFETY_UPDATE_JSON="
            + json.dumps(record, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        self._joint_safety_consumed_step = prepared["step"]
        self._joint_safety_consumed_record = record
        self._joint_safety_last_policy_step_sequence = prepared[
            "last_policy_step_sequence"
        ]
        self._joint_safety_last_consume_sequence = prepared[
            "consume_sequence"
        ]
        self._joint_safety_last_identity = None
        self._joint_safety_last_archive_sequence = None
        self._joint_safety_pending_prepared = None
        return record

    def _prepare_joint_safety_update(
        self, step: int, *, expected_action_term=None
    ) -> dict:
        """Freeze, validate and persist a rollout before PPO may update."""

        step = self._joint_safety_int(step, name="PPO update", minimum=0)
        prior_step = getattr(self, "_joint_safety_consumed_step", None)
        if prior_step is not None and step != int(prior_step) + 1:
            raise RuntimeError(
                "joint-safety PPO update sequence is not contiguous before prepare"
            )
        existing_pending = getattr(
            self, "_joint_safety_pending_prepared", None
        )
        if existing_pending is not None:
            raise RuntimeError(
                "joint-safety has an unacknowledged prepared update; refusing "
                "another PPO boundary"
            )
        term = self._bind_joint_safety_action_term(required=True)
        if expected_action_term is not None and term is not expected_action_term:
            raise RuntimeError(
                "joint-safety joint_pos action term changed after launch binding"
            )
        rank = self._joint_safety_rank()
        self._preflight_joint_safety_update_paths(step=step, rank=rank)
        contract = self._joint_safety_runtime_contract(term)
        token, raw_snapshot = term.prepare_joint_safety_ledger_consume()
        if not isinstance(token, tuple):
            raise RuntimeError(
                "joint-safety action term returned a non-opaque prepare token"
            )
        prepared = {
            "step": step,
            "rank": rank,
            "term": term,
            "token": token,
            "status": "snapshot_frozen",
        }
        # Install the pending owner immediately: validation, CPU transfer, or persistence failure
        # must leave a visible, unacknowledged generation and the action term itself remains frozen.
        self._joint_safety_pending_prepared = prepared
        snapshot = raw_snapshot
        validated = self._validate_joint_safety_update_snapshot(
            snapshot, step=step, contract=contract
        )
        payload = self._joint_safety_cpu_clone(
            self._compact_joint_safety_artifact(
                validated, step=step, rank=rank
            )
        )
        artifact = self._persist_joint_safety_update(payload, step=step)
        aggregate = validated["aggregate"]
        combined_gap = validated["combined_gap"]
        finite_gap = combined_gap[torch.isfinite(combined_gap)]
        reason_counts = {}
        for archive in validated["archives"]:
            for reason in archive["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        record = {
            "event": _JOINT_SAFETY_EVENT,
            "schema_version": _JOINT_SAFETY_ARTIFACT_SCHEMA_VERSION,
            "status": payload["status"],
            "ppo_update": step,
            "consume_sequence": validated["consume_sequence"],
            "num_envs": validated["num_envs"],
            "policy_step_count": validated["expected_policy_steps"],
            "first_policy_step_sequence": validated[
                "first_policy_step_sequence"
            ],
            "last_policy_step_sequence": validated[
                "last_policy_step_sequence"
            ],
            "complete_env_policy_steps": int(
                validated["complete_total"].sum().item()
            ),
            "incomplete_env_policy_steps": 0,
            "minimum_hard_gap_rad": (
                None
                if finite_gap.numel() == 0
                else float(finite_gap.min().item())
            ),
            "counter_totals": {
                field: int(value.sum().item())
                for field, value in aggregate.items()
            },
            "fatal_flags": payload["fatal_flags"],
            "terminal_archive_count": validated["archive_count"],
            "terminal_reason_counts": dict(sorted(reason_counts.items())),
            "per_policy_step_sparse_counters": validated["per_step"],
            "identity_binding": (
                "lossless_initial_per_env_identity_plus_sparse_generation_"
                "transitions_and_per_step_sha256"
            ),
            "artifact": artifact,
        }
        prepared.update(
            {
                "status": payload["status"],
                "validated": validated,
                "payload": payload,
                "artifact": artifact,
                "record": record,
            }
        )
        if payload["status"] == "fatal_actual_hard_edge":
            diagnostic_finite_terminal_sample = (
                self._action_ball_diagnostic_unauthorized()
                and bool(
                    torch.all(
                        torch.isfinite(validated["combined_gap"])
                    ).item()
                )
            )
            if diagnostic_finite_terminal_sample:
                # A finite raw-hard contact remains a terminal, heavily
                # penalized transition for the affected environment.  It is
                # not evidence corruption, however, and discarding the whole
                # rollout would prevent PPO from learning to avoid precisely
                # that failure.  Formal ActionBall remains fail-closed below;
                # non-finite q is never admitted through this diagnostic
                # exception because it produces a non-finite hard gap.
                record["optimizer_disposition"] = (
                    "diagnostic_continue_after_finite_terminal_hard_edge"
                )
                prepared["status"] = "prepared_before_optimizer"
                prepared["record"] = record
            print(
                "HOPE_JOINT_SAFETY_FATAL_JSON="
                + json.dumps(record, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            if diagnostic_finite_terminal_sample:
                print(
                    "HOPE_JOINT_SAFETY_DIAGNOSTIC_CONTINUE_JSON="
                    + json.dumps(
                        {
                            "event": _JOINT_SAFETY_EVENT,
                            "schema_version": (
                                _JOINT_SAFETY_ARTIFACT_SCHEMA_VERSION
                            ),
                            "ppo_update": step,
                            "source_evidence_status": payload["status"],
                            "optimizer_disposition": record[
                                "optimizer_disposition"
                            ],
                            "fatal_flags": payload["fatal_flags"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return prepared
            raise RuntimeError(
                "physical joint hard-edge/non-finite-q evidence was durably "
                "recorded; refusing PPO update and leaving the ledger frozen"
            )
        return prepared

    def _commit_joint_safety_update(self, prepared: dict) -> dict:
        """After optimizer success, durably mark it and acknowledge exact evidence."""

        if (
            prepared is not getattr(
                self, "_joint_safety_pending_prepared", None
            )
            or prepared.get("status") != "prepared_before_optimizer"
        ):
            raise RuntimeError(
                "joint-safety optimizer commit does not own the pending "
                "prepared evidence"
            )
        term = self._bind_joint_safety_action_term(required=True)
        if term is not prepared["term"]:
            raise RuntimeError(
                "joint-safety joint_pos action term changed before acknowledgement"
            )
        commit_artifact = self._persist_joint_safety_optimizer_commit(prepared)
        term.acknowledge_joint_safety_ledger(prepared["token"])
        record = {
            **prepared["record"],
            "status": "optimizer_committed_and_ledger_acknowledged",
            "optimizer_commit": commit_artifact,
        }
        print(
            "HOPE_JOINT_SAFETY_UPDATE_JSON="
            + json.dumps(record, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        validated = prepared["validated"]
        self._joint_safety_consumed_step = prepared["step"]
        self._joint_safety_consumed_record = record
        self._joint_safety_last_policy_step_sequence = validated[
            "last_policy_step_sequence"
        ]
        self._joint_safety_last_consume_sequence = validated[
            "consume_sequence"
        ]
        self._joint_safety_last_identity = validated["last_identity"]
        self._joint_safety_last_archive_sequence = validated[
            "last_archive_sequence"
        ]
        self._joint_safety_pending_prepared = None
        return record

    def _consume_joint_safety_update(
        self, step: int, *, expected_action_term=None
    ) -> Optional[dict]:
        """Test/compatibility boundary: prepare then attest an already-successful optimizer."""

        step = self._joint_safety_int(step, name="PPO update", minimum=0)
        if getattr(self, "_joint_safety_consumed_step", None) == step:
            return getattr(self, "_joint_safety_consumed_record", None)
        if self._diagnostic_joint_safety_compact_update_evidence():
            prepared = self._prepare_diagnostic_joint_safety_update(
                step, expected_action_term=expected_action_term
            )
            return self._commit_diagnostic_joint_safety_update(prepared)
        prepared = self._prepare_joint_safety_update(
            step, expected_action_term=expected_action_term
        )
        return self._commit_joint_safety_update(prepared)

    def _consume_exact_behavior_updates(self, step: int) -> Dict[str, Dict]:
        """Consume the sole behavior ledger once and emit one canonical JSON line per PPO update."""

        if getattr(self, "_exact_behavior_consumed_step", None) == int(step):
            return getattr(self, "_exact_behavior_consumed_records", {})
        records: Dict[str, Dict] = {}
        env = self.env.unwrapped
        if hasattr(env, "command_manager"):
            providers = []
            for term_name in env.command_manager.active_terms:
                term = env.command_manager.get_term(term_name)
                consumer = getattr(term, "consume_exact_behavior_decision_counters", None)
                if callable(consumer):
                    providers.append((str(term_name), consumer))
            if len(providers) > 1:
                names = [name for name, _consumer in providers]
                raise RuntimeError(
                    "exact behavior receipt requires exactly one provider; found "
                    f"{names}"
                )
            for term_name, consumer in providers:
                counters = {
                    name: self._exact_counter_value(value)
                    for name, value in consumer().items()
                }
                record = {
                    "event": _EXACT_BEHAVIOR_EVENT,
                    "schema_version": 1,
                    "ppo_update": int(step),
                    "term": term_name,
                    "counters": dict(sorted(counters.items())),
                    "derived": exact_behavior_decision_values(counters),
                    "window_aggregation": "sum_counters_then_recompute_derived",
                }
                records[term_name] = record
                print(
                    "HOPE_EXACT_BEHAVIOR_UPDATE_JSON="
                    + json.dumps(record, sort_keys=True, separators=(",", ":")),
                    flush=True,
                )
        self._exact_behavior_consumed_step = int(step)
        self._exact_behavior_consumed_records = records
        return records

    def _check_zero_return_alarm(self, term_name: str, counters: dict, step: int) -> None:
        """Accumulate the per-family strike ledger across updates and alarm on a pinned-at-zero side.

        The per-update record is a window, so the decision needs a run-lifetime total; the counters are
        non-decaying integers, so summing them is exact.
        """

        totals = getattr(self, "_zero_return_cumulative", None)
        if totals is None:
            totals = {}
            self._zero_return_cumulative = totals
        term_totals = totals.setdefault(term_name, {})
        for family in _STRIKE_FAMILIES:
            for key in (
                f"strike_opportunity_count_{family}",
                f"virtual_legal_return_count_{family}",
            ):
                if key in counters:
                    term_totals[key] = term_totals.get(key, 0) + float(counters[key])
        levels = zero_return_alarm_levels(term_totals)
        for family in _STRIKE_FAMILIES:
            opportunity_key = f"strike_opportunity_count_{family}"
            if opportunity_key not in term_totals:
                continue
            level = levels.get(family)
            self._log_scalar(
                f"Live/{term_name}/zero_return_alarm_{family}",
                1.0 if level else 0.0,
                step,
            )
            if level is None:
                continue
            message = (
                f"[HOPE ALARM] {term_name}: virtual_legal_return_count_{family} is still 0 after "
                f"{term_totals[opportunity_key]:.0f} cumulative strike opportunities "
                f"(ppo_update {int(step)}). A side that NEVER returns is a broken command, not a "
                f"slow learner — check that the {family} racket target box clears the table "
                f"(z_lo >= vb_table_surface_z + ball radius) before spending more GPU on this run."
            )
            print(message, flush=True)
            if level == "abort":
                raise RuntimeError(message)

    def _log_live_metrics(
        self, step: int, *, exact_behavior: Optional[Dict[str, Dict]] = None
    ) -> None:
        """Log current manager state means every PPO iteration for richer dashboards."""
        env = self.env.unwrapped
        if exact_behavior is None:
            exact_behavior = self._consume_exact_behavior_updates(step)

        if hasattr(env, "command_manager"):
            for term_name in env.command_manager.active_terms:
                term = env.command_manager.get_term(term_name)
                for metric_name, metric_value in term.metrics.items():
                    self._log_scalar(f"Live/{term_name}/{metric_name}", self._mean_tensor(metric_value), step)
                activation_consumer = getattr(
                    term, "consume_training_activation_counters", None
                )
                if not callable(activation_consumer):
                    # Backward-compatible fallback for command terms that expose only the first
                    # post-swing ledger API.  A term exposing both APIs is consumed exactly once
                    # through the aggregate transaction above.
                    activation_consumer = getattr(
                        term, "consume_post_swing_activation_counters", None
                    )
                if callable(activation_consumer):
                    # These are integer event/sample counts accumulated across every environment
                    # step in the just-finished PPO update.  They must be logged as totals, not
                    # averaged over num_envs like instantaneous CommandTerm metrics.  The
                    # aggregate consumer snapshots and resets all ledgers exactly once.
                    for counter_name, counter_value in activation_consumer().items():
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            self._scalar_tensor(counter_value),
                            step,
                        )
                exact_record = exact_behavior.get(str(term_name))
                if exact_record is not None:
                    for counter_name, counter_value in exact_record["counters"].items():
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            counter_value,
                            step,
                        )
                    self._check_zero_return_alarm(
                        str(term_name), exact_record["counters"], step
                    )
                else:
                    sparse_reward_consumer = getattr(
                        term, "consume_sparse_reward_eligibility_counters", None
                    )
                if exact_record is None and callable(sparse_reward_consumer):
                    # Exact non-decayed counts, including per-action denominators.  These are
                    # intentionally a second transaction: MotionCommand owns imitation/replay
                    # activation, while RacketTargetCommand owns virtual strike outcomes.
                    for counter_name, counter_value in sparse_reward_consumer().items():
                        self._log_scalar(
                            f"Live/{term_name}/{counter_name}",
                            self._scalar_tensor(counter_value),
                            step,
                        )
                if hasattr(term, "command_counter"):
                    self._log_scalar(
                        f"Live/{term_name}/command_counter", self._mean_tensor(term.command_counter), step
                    )
        if hasattr(env, "reward_manager"):
            active_reward_terms = tuple(env.reward_manager.active_terms)
            self._log_scalar("Live/Reward/total", self._mean_tensor(getattr(env, "reward_buf", None)), step)
            for idx, term_name in enumerate(active_reward_terms):
                self._log_scalar(
                    f"Live/Reward/{term_name}", self._mean_tensor(env.reward_manager._step_reward[:, idx]), step
                )
            if "base_decel_activation_probe" in active_reward_terms:
                # The nonzero-weight, zero-valued probe is the common RewardManager-stage
                # observer for control and treatment.  Consume its weight-independent ledger
                # exactly once per PPO update; totals/raw sum must not be averaged over num_envs.
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_base_decel_activation_counters,
                )

                command_name = "racket_target"
                for counter_name, counter_value in (
                    consume_base_decel_activation_counters(env, command_name).items()
                ):
                    self._log_scalar(
                        f"Live/{command_name}/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "joint_velocity_limit_hinge_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_joint_velocity_limit_hinge_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_joint_velocity_limit_hinge_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/qdot/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "processed_qdes_slew_hinge_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_processed_qdes_slew_hinge_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_processed_qdes_slew_hinge_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/processed_qdes_slew/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "qdes_limit_barrier_probe" in active_reward_terms:
                # Wave-Q qbar: weight-independent all-joint q_des limit-barrier ledger
                # (above-margin joint counts + max intrusion depth).  Consumed exactly once
                # per PPO update; counts/sums must not be averaged over num_envs.
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_qdes_limit_barrier_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_qdes_limit_barrier_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/qdes_limit_barrier/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if (
                "lower_body_pose_imitation_probe" in active_reward_terms
                or "lower_body_stability_bundle_probe" in active_reward_terms
            ):
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_lower_body_wave_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_lower_body_wave_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/lower_body_wave/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "post_swing_settle_debt_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_post_swing_settle_debt_activation_counters,
                )

                for counter_name, counter_value in (
                    consume_post_swing_settle_debt_activation_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/post_swing_settle_debt/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "action_acc_jerk_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_action_acc_jerk_probe_counters,
                )

                for counter_name, counter_value in (
                    consume_action_acc_jerk_probe_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/action_acc_jerk_probe/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )
            if "implicit_pd_post_step_effort_proxy_probe" in active_reward_terms:
                from whole_body_tracking.tasks.tracking.mdp.hope_rewards import (
                    consume_implicit_pd_post_step_effort_proxy_counters,
                )

                for counter_name, counter_value in (
                    consume_implicit_pd_post_step_effort_proxy_counters(env).items()
                ):
                    self._log_scalar(
                        f"Live/implicit_pd_post_step_effort_proxy/{counter_name}",
                        self._scalar_tensor(counter_value),
                        step,
                    )

        if hasattr(env, "termination_manager"):
            tm = env.termination_manager
            self._log_scalar("Live/Termination/done_rate", self._mean_tensor(tm.dones), step)
            self._log_scalar("Live/Termination/terminated_rate", self._mean_tensor(tm.terminated), step)
            self._log_scalar("Live/Termination/timeout_rate", self._mean_tensor(tm.time_outs), step)
            for term_name in tm.active_terms:
                self._log_scalar(f"Live/Termination/{term_name}", self._mean_tensor(tm.get_term(term_name)), step)

        if hasattr(env, "action_manager"):
            action = getattr(env.action_manager, "action", None)
            prev_action = getattr(env.action_manager, "prev_action", None)
            if action is not None:
                action_abs = torch.abs(action)
                self._log_scalar("Live/Action/mean_abs", self._mean_tensor(action_abs), step)
                self._log_scalar("Live/Action/max_abs", self._mean_tensor(torch.max(action_abs, dim=-1).values), step)
            if action is not None and prev_action is not None:
                action_delta_abs = torch.abs(action - prev_action)
                self._log_scalar("Live/Action/delta_mean_abs", self._mean_tensor(action_delta_abs), step)
                self._log_scalar(
                    "Live/Action/delta_max_abs",
                    self._mean_tensor(torch.max(action_delta_abs, dim=-1).values),
                    step,
                )

        self._log_scalar("Live/Env/episode_length", self._mean_tensor(env.episode_length_buf), step)
        self._log_scalar("Live/Env/common_step_counter", float(getattr(env, "common_step_counter", 0)), step)

    @staticmethod
    def _mean_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.float().mean().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _scalar_tensor(value):
        """Convert one scalar event counter without accidentally averaging a vector."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("per-update activation counter must be a scalar tensor")
            return value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _exact_counter_value(value):
        """Preserve integer counters in JSON and reject vectors/non-finite sums."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("per-update exact behavior counter must be a scalar tensor")
            value = value.item()
        if isinstance(value, bool):
            raise TypeError("per-update exact behavior counter must not be boolean")
        if isinstance(value, int):
            return value
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("per-update exact behavior sum must be finite")
        return numeric

    def _log_scalar(self, tag: str, value, step: int) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.writer.add_scalar(tag, float(value), step)
