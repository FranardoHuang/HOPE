from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import uuid

import pytest
import torch


WBT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WBT_ROOT / "source" / "whole_body_tracking"
RUNNER_PATH = (
    SOURCE_ROOT
    / "whole_body_tracking"
    / "utils"
    / "my_on_policy_runner.py"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import action_ball_full_mdp_checkpoint as C  # noqa: E402
import action_ball_full_mdp_env_checkpoint_adapter as E  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _Memory:
    def __init__(self) -> None:
        self.hidden_states = None

    def reset(self, dones=None, hidden_states=None) -> None:
        if dones is None:
            self.hidden_states = hidden_states
        elif self.hidden_states is not None:
            self.hidden_states[..., dones == 1, :] = 0.0


class _Policy(torch.nn.Module):
    is_recurrent = True
    noise_std_type = "scalar"

    def __init__(self, num_envs: int, num_actions: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((num_actions,), 0.25))
        self.std = torch.nn.Parameter(torch.full((num_actions,), 0.20))
        self.memory_a = _Memory()
        self.memory_c = _Memory()
        self._num_envs = num_envs

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class _Normalizer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.register_buffer("count", torch.zeros(()))
        self.forward_calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.count.add_(1.0)
        return (value - self.mean) / self.std


class _Transition:
    _FIELDS = (
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
    )

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        for name in self._FIELDS:
            setattr(self, name, None)


class _Storage:
    def __init__(self) -> None:
        self.step = 0

    def clear(self) -> None:
        self.step = 0


class _Algorithm:
    def __init__(self, policy: _Policy, *, leave_storage_full: bool) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-2)
        self.schedule = "adaptive"
        self.desired_kl = 0.01
        self.learning_rate = 1.0e-2
        self.rnd = None
        self.storage = _Storage()
        self.transition = _Transition()
        self.leave_storage_full = leave_storage_full
        self.first_act_observations = None
        self.first_act_critic_observations = None
        self.first_act_hidden = None

    def act(self, observations: torch.Tensor, critic: torch.Tensor) -> torch.Tensor:
        if self.first_act_observations is None:
            self.first_act_observations = observations.detach().clone()
            self.first_act_critic_observations = critic.detach().clone()
            hidden = self.policy.get_hidden_states()
            self.first_act_hidden = tuple(
                None if item is None else item.detach().clone() for item in hidden
            )
        actor_hidden, critic_hidden = self.policy.get_hidden_states()
        if actor_hidden is None:
            actor_hidden = torch.zeros(1, observations.shape[0], 3)
            critic_hidden = torch.zeros(1, observations.shape[0], 3)
        increment = observations[:, :1].reshape(1, observations.shape[0], 1)
        self.policy.memory_a.hidden_states = actor_hidden + increment
        self.policy.memory_c.hidden_states = critic_hidden + increment + 0.5
        self.transition.observations = observations
        self.transition.privileged_observations = observations
        self.transition.actions = torch.zeros(observations.shape[0], 2)
        self.transition.hidden_states = self.policy.get_hidden_states()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos) -> None:
        del rewards, infos
        self.storage.step += 1
        self.transition.clear()
        self.policy.memory_a.reset(dones)
        self.policy.memory_c.reset(dones)

    def compute_returns(self, _critic: torch.Tensor) -> None:
        return None

    def update(self):
        self.optimizer.zero_grad()
        loss = sum(parameter.square().sum() for parameter in self.policy.parameters())
        loss.backward()
        self.optimizer.step()
        self.learning_rate *= 0.5
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        if not self.leave_storage_full:
            self.storage.clear()
        return {"loss": float(loss.detach())}


class _Env:
    def __init__(
        self,
        *,
        raw_offset: float = 0.0,
        obs_mode: str = "action_ball_a211",
        target_mode: str = "action_ball",
        actor_width: int = 211,
        critic_width: int = 319,
    ) -> None:
        self.unwrapped = self
        self.num_envs = 2
        self.num_actions = 2
        self.device = "cpu"
        self.getter_calls = 0
        self.noise_calls = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.raw_offset = float(raw_offset)
        self.actor_width = int(actor_width)
        self.critic_width = int(critic_width)
        self._action_ball_full_mdp_runtime_lease = object()
        self._drain_owner = None
        self.cfg = SimpleNamespace(
            obs_mode=obs_mode,
            commands=SimpleNamespace(
                racket_target=SimpleNamespace(target_mode=target_mode)
            ),
        )

    @property
    def action_ball_full_mdp_runtime_lease(self):
        return self._action_ball_full_mdp_runtime_lease

    def action_ball_full_mdp_ppo_drain_owner(self, lease):
        if lease is not self._action_ball_full_mdp_runtime_lease:
            raise RuntimeError("foreign test runtime lease")
        if self._drain_owner is None:
            raise RuntimeError("test global drain is not installed")
        return self._drain_owner

    def _observations(self, value: float):
        actor = torch.full((self.num_envs, self.actor_width), value)
        critic = torch.full((self.num_envs, self.critic_width), value + 0.25)
        actor[:, -1] = 1.0
        critic[:, -1] = 1.0
        return actor, {"observations": {"critic": critic}}

    def get_observations(self):
        self.getter_calls += 1
        self.noise_calls += 1
        return self._observations(self.raw_offset + self.step_calls)

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("test R10 lane must not reset")

    def step(self, _actions):
        self.step_calls += 1
        actor, extras = self._observations(self.raw_offset + self.step_calls)
        rewards = torch.zeros(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        return actor, rewards, dones, extras


class _UnprintableCleanupError(BaseException):
    def __str__(self) -> str:
        raise RuntimeError("cleanup exception text is unavailable")


class _GuardCleanupEnv(_Env):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_attempts = []

    def __delattr__(self, name: str) -> None:
        if name in ("step", "reset", "get_observations"):
            self.cleanup_attempts.append(name)
        if name == "step":
            raise _UnprintableCleanupError()
        super().__delattr__(name)


def _load_runner_module():
    saved_modules: dict[str, object] = {}

    def install(name: str, module: types.ModuleType) -> None:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    rsl_rl = types.ModuleType("rsl_rl")
    rsl_env = types.ModuleType("rsl_rl.env")
    rsl_runners = types.ModuleType("rsl_rl.runners")
    rsl_on_policy = types.ModuleType("rsl_rl.runners.on_policy_runner")

    class VecEnv:
        pass

    class OnPolicyRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env
            self.cfg = train_cfg
            self.log_dir = log_dir
            self.device = device
            self.training_type = "rl"
            observations, extras = env.get_observations()
            assert "critic" in extras["observations"]
            self.privileged_obs_type = "critic"
            self.num_steps_per_env = int(train_cfg["num_steps_per_env"])
            self.save_interval = 100
            self.empirical_normalization = True
            self.obs_normalizer = _Normalizer(observations.shape[1])
            self.privileged_obs_normalizer = _Normalizer(
                extras["observations"]["critic"].shape[1]
            )
            policy = _Policy(env.num_envs, env.num_actions)
            self.alg = _Algorithm(
                policy,
                leave_storage_full=bool(train_cfg.get("leave_storage_full", False)),
            )
            self.disable_logs = True
            self.is_distributed = False
            self.current_learning_iteration = 0
            self.tot_timesteps = 0
            self.tot_time = 0.0

        def train_mode(self):
            self.alg.policy.train()
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            assert init_at_random_ep_len is False
            observations, extras = self.env.get_observations()
            critic = extras["observations"]["critic"]
            observations = observations.to(self.device)
            critic = critic.to(self.device)
            self.train_mode()
            start = int(self.current_learning_iteration)
            for iteration in range(start, start + int(num_learning_iterations)):
                with torch.inference_mode():
                    for _ in range(self.num_steps_per_env):
                        actions = self.alg.act(observations, critic)
                        observations, rewards, dones, infos = self.env.step(actions)
                        observations = self.obs_normalizer(observations.to(self.device))
                        critic = self.privileged_obs_normalizer(
                            infos["observations"]["critic"].to(self.device)
                        )
                        self.alg.process_env_step(rewards, dones, infos)
                    self.alg.compute_returns(critic)
                self.alg.update()
                self.current_learning_iteration = iteration

    OnPolicyRunner.__module__ = "rsl_rl.runners.on_policy_runner"
    rsl_env.VecEnv = VecEnv
    rsl_on_policy.OnPolicyRunner = OnPolicyRunner
    install("rsl_rl", rsl_rl)
    install("rsl_rl.env", rsl_env)
    install("rsl_rl.runners", rsl_runners)
    install("rsl_rl.runners.on_policy_runner", rsl_on_policy)

    isaaclab_rl = types.ModuleType("isaaclab_rl")
    isaaclab_rsl = types.ModuleType("isaaclab_rl.rsl_rl")
    isaaclab_rsl.export_policy_as_onnx = lambda *_args, **_kwargs: None
    install("isaaclab_rl", isaaclab_rl)
    install("isaaclab_rl.rsl_rl", isaaclab_rsl)

    exporter = types.ModuleType("whole_body_tracking.utils.exporter")
    exporter.attach_onnx_metadata = lambda *_args, **_kwargs: None
    exporter.export_motion_policy_as_onnx = lambda *_args, **_kwargs: False
    exporter.is_empirical_normalizer = lambda value: isinstance(value, _Normalizer)
    install("whole_body_tracking.utils.exporter", exporter)

    contract = types.ModuleType("whole_body_tracking.utils.training_contract")
    contract.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY = "lineage"
    contract.CHECKPOINT_CONTRACT_SCHEMA_KEY = "schema"
    contract.CHECKPOINT_CONTRACT_SHA_KEY = "sha"
    contract.CHECKPOINT_LAUNCH_CLAIM_SHA_KEY = "claim"
    contract.TRAINING_CONTRACT_SCHEMA_VERSION = 1
    contract.validate_training_launch_claim_sha256 = lambda value: value
    install("whole_body_tracking.utils.training_contract", contract)

    for family in ("a211", "c211"):
        name = (
            "whole_body_tracking.tasks.tracking."
            f"action_ball_{family}_trainability"
        )
        module = types.ModuleType(name)
        function_name = (
            "validate_action_ball_211_runner"
            if family == "a211"
            else "validate_action_ball_c211_runner"
        )
        setattr(
            module,
            function_name,
            lambda _runner, _family=family: {"family": _family},
        )
        install(name, module)

    module_name = f"_r10_runner_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    install(module_name, module)
    spec.loader.exec_module(module)
    cls = module.MotionOnPolicyRunner
    module._test_original_validate_task_first_exact_resume_terms = (
        cls._validate_task_first_exact_resume_terms
    )
    cls._validate_task_first_exact_resume_terms = lambda _self: None
    cls._emit_rsl_rl_runtime_abi = lambda _self, **_kwargs: None
    cls._emit_control_step_action_delay_runtime_receipt = lambda _self: None
    cls._reward_ppo_economy_gate_requested = lambda _self: False
    cls._effective_reward_activation_task_kind = lambda _self: None
    cls._diagnostic_joint_safety_compact_evidence = lambda _self: False
    cls._bind_joint_safety_action_term = lambda _self, **_kwargs: None
    cls._emit_policy_std_update = lambda _self, **_kwargs: None
    cls._notify_command_terms_rollout_end = lambda _self, _step: None
    cls._service_action_ball_frozen_evaluation = lambda _self, _step: False
    return module, saved_modules


@pytest.fixture(scope="module")
def runner_module():
    module, saved = _load_runner_module()
    try:
        yield module
    finally:
        for name, previous in reversed(tuple(saved.items())):
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_RUN_OWNERS = frozenset(
    (
        "authority.root",
        "trainer.policy",
        "trainer.optimizer_schedule",
        "trainer.normalizer.actor",
        "trainer.normalizer.critic",
        "trainer.rollout_frontier",
        "telemetry.highwater",
        "rng.process",
    )
)


def _registry() -> C.OrderedOwnerRegistry:
    return C.OrderedOwnerRegistry(
        tuple(
            C.OwnerDescriptor(
                owner_id=owner_id,
                schema_version=1,
                state_kind=f"action_ball.r10.{owner_id}.v1",
                scope=(
                    C.OwnerScope.RUN
                    if owner_id in _RUN_OWNERS
                    else C.OwnerScope.WORLD
                ),
                engine=(
                    C.OwnerEngine.ISAAC
                    if owner_id == "env.plant"
                    else C.OwnerEngine.PORTABLE
                ),
                immutable_identity_sha256=_sha("identity:" + owner_id),
                dependencies=C.R10_OWNER_DEPENDENCIES[owner_id],
                restore_rank=index,
            )
            for index, owner_id in enumerate(C.R10_OWNER_ORDER)
        )
    )


class _ExternalOwner:
    def __init__(
        self,
        descriptor: C.OwnerDescriptor,
        *,
        poison_observer=None,
    ) -> None:
        self.descriptor = descriptor
        self.version = 0
        self.payload = ("baseline:" + descriptor.owner_id).encode("ascii")
        self.join_claims = ()
        self.poisoned = False
        self.poison_observer = poison_observer
        self.poison_failure = None
        self.prepare_calls = 0
        self.commit_calls = 0

    def mutation_version(self) -> int:
        return self.version

    def live_digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def freeze(self, boundary: C.CheckpointBoundary):
        return C.OwnerFreezeReceipt(
            owner_id=self.descriptor.owner_id,
            descriptor_sha256=C.descriptor_sha256(self.descriptor),
            boundary_sha256=C.boundary_sha256(boundary),
            mutation_version=self.version,
            seal_nonce_sha256=_sha("nonce:" + self.descriptor.owner_id),
        )

    def export_sealed(self, receipt):
        return C.make_opaque_owner_state(
            descriptor=self.descriptor,
            receipt=receipt,
            live_digest_sha256=self.live_digest(),
            payload=self.payload,
            join_claims=self.join_claims,
        )

    def prepare_restore(self, envelope, _pins, owner_root_sha256):
        self.prepare_calls += 1
        return C.PreparedRestoreToken(
            owner_id=self.descriptor.owner_id,
            descriptor_sha256=C.descriptor_sha256(self.descriptor),
            checkpoint_owner_root_sha256=owner_root_sha256,
            opaque_token={
                "staged": envelope.payload,
                "target_version": envelope.mutation_version,
                "baseline": self.payload,
                "baseline_version": self.version,
            },
        )

    def commit_restore(self, token) -> None:
        self.commit_calls += 1
        self.payload = token.opaque_token["staged"]
        self.version = token.opaque_token["target_version"]

    def rollback_restore(self, token) -> None:
        self.payload = token.opaque_token["baseline"]
        self.version = token.opaque_token["baseline_version"]

    def poison_restore(self, _reason: str) -> None:
        if self.poison_observer is not None:
            self.poison_observer(self.descriptor.owner_id)
        self.poisoned = True
        if self.poison_failure is not None:
            raise self.poison_failure


def _world(world_id: int = 0) -> C.WorldCheckpointPhase:
    return C.WorldCheckpointPhase(
        world_id=world_id,
        reset_generation=1,
        episode_uid_sha256=_sha(f"episode:{world_id}:1"),
        episode_step=7,
        task_birth_snapshot_id=3,
        reset_phase=C.ResetPhase.COMMITTED,
        physics_substep_phase=0,
        physics_in_flight=False,
        r05_phase=C.R05Phase.PREPARED,
        r05_operation_active=False,
        r05_prepared_sealed=True,
        r05_cross_owner_commit_complete=False,
        r03_phase=C.R03Phase.PAID,
        r03_all_consumers_paid=True,
        r03_view_mask=C.R03_FULL_CONSUMER_MASK,
        r03_payment_mask=C.R03_FULL_CONSUMER_MASK,
        r06_flight_phase=C.R06FlightPhase.SETTLED_RETAINED,
        r06_mailbox_phase=C.R06MailboxPhase.PARTIALLY_PAID,
        r06_payment_epoch_open=False,
        r06_view_mask=1,
        r06_payment_mask=1,
        r07_payment_epoch_open=False,
        r07_deadline_ack_pending=False,
    )


class _Adapter:
    def __init__(
        self,
        *,
        save: bool,
        contract_sha256: str | None = None,
        boundary_world_ids: tuple[int, ...] = (0, 1),
    ) -> None:
        self.registry = _registry()
        self.save = save
        self.contract_sha256 = contract_sha256 or _sha("contract")
        self.boundary_world_ids = boundary_world_ids
        self.publication = None
        self.handoffs = []
        self.restore_complete_calls = []
        self.restore_complete_failure = None
        self.restore_complete_result = None
        self.last_complete_owners = None
        self.poison_attempts = []
        self.guard_observations = []
        self.env = None
        self.external = {
            owner_id: _ExternalOwner(
                self.registry.descriptor(owner_id),
                poison_observer=self.poison_attempts.append,
            )
            for owner_id in C.R10_OWNER_ORDER
            if not owner_id.startswith("trainer.")
        }

    def action_ball_r10_registry(self):
        return self.registry

    @staticmethod
    def _join_value(join_id: str, frontier_sha256: str) -> str:
        if join_id == "trainer_update_frontier":
            return frontier_sha256
        return _sha("join:" + join_id)

    def _claims(self, owner_id: str, frontier_sha256: str):
        return tuple(
            C.OwnerJoinClaim(
                spec.join_id,
                self._join_value(spec.join_id, frontier_sha256),
            )
            for spec in C.R10_GLOBAL_JOIN_SPECS
            if owner_id in spec.owner_ids
        )

    def action_ball_r10_post_update_authority(self, handoff):
        self.handoffs.append(handoff)
        if not self.save:
            return None
        for owner_id, owner in self.external.items():
            owner.version = handoff.next_learning_iteration
            owner.payload = (
                f"update:{handoff.completed_update_index}:{owner_id}"
            ).encode("ascii")
            owner.join_claims = self._claims(
                owner_id, handoff.runner_frontier_sha256
            )
        boundary = C.CheckpointBoundary(
            boundary_id_sha256=_sha(
                f"boundary:{handoff.completed_update_index}"
            ),
            update_index=handoff.completed_update_index,
            ppo_phase=C.PPOBoundaryPhase.POST_UPDATE_ROLLOUT_EMPTY,
            environment_step_phase=C.EnvironmentStepPhase.BETWEEN_COMPLETE_STEPS,
            rollout_storage_empty=True,
            actor_frontier_sealed=True,
            critic_frontier_sealed=True,
            recurrent_frontier=handoff.recurrent_frontier,
            gae_in_flight=False,
            optimizer_in_flight=False,
            reset_in_flight=False,
            worlds=tuple(_world(world_id) for world_id in self.boundary_world_ids),
        )
        ppo_drain_frontier = self.runtime_owner.global_drain.frontier(boundary)
        physical_ppo_drain_frontier = getattr(
            self, "physical_ppo_drain_frontier_override", ppo_drain_frontier
        )
        for owner_id, owner in self.external.items():
            owner.join_claims = tuple(
                C.OwnerJoinClaim(
                    claim.join_id,
                    (
                        physical_ppo_drain_frontier
                        if claim.join_id == "ppo_drain_frontier"
                        else claim.value_sha256
                    ),
                )
                for claim in owner.join_claims
            )
        runner_claims = {
            owner_id: tuple(
                C.OwnerJoinClaim(
                    claim.join_id,
                    (
                        physical_ppo_drain_frontier
                        if claim.join_id == "ppo_drain_frontier"
                        else claim.value_sha256
                    ),
                )
                for claim in self._claims(
                    owner_id, handoff.runner_frontier_sha256
                )
            )
            for owner_id in (
                "trainer.policy",
                "trainer.optimizer_schedule",
                "trainer.normalizer.actor",
                "trainer.normalizer.critic",
                "trainer.rollout_frontier",
            )
        }
        return self.module.ActionBallR10RunnerSaveAuthority(
            schema_version=1,
            kind="action_ball_r10_isaac_runner_save_authority_v1",
            family=C.PolicyFamily.A,
            immutable_pins=C.ImmutableCheckpointPins(
                code_sha256=_sha("code"),
                config_sha256=_sha("config"),
                contract_sha256=self.contract_sha256,
            ),
            boundary=boundary,
            runner_join_claims=runner_claims,
        )

    def action_ball_r10_complete_owners(
        self, *, operation, runner_owners, verified_checkpoint
    ):
        assert operation in ("seal", "cold_restore")
        if operation == "cold_restore":
            assert verified_checkpoint is not None
        owners = tuple(
            runner_owners[owner_id]
            if owner_id in runner_owners
            else self.external[owner_id]
            for owner_id in self.registry.owner_ids
        )
        if operation == "cold_restore":
            for owner in owners:
                owner_id = owner.descriptor.owner_id
                if owner_id.startswith("trainer."):
                    original = owner.poison_restore

                    def observed_poison(reason, *, _id=owner_id, _call=original):
                        self.poison_attempts.append(_id)
                        return _call(reason)

                    owner.poison_restore = observed_poison
            self.last_complete_owners = owners
        return owners

    def action_ball_r10_restore_complete(self, receipt):
        self.restore_complete_calls.append(receipt)
        if self.env is not None:
            guarded = []
            for method_name in ("get_observations", "reset", "step"):
                method = getattr(self.env, method_name)
                try:
                    method(None) if method_name == "step" else method()
                except RuntimeError as exc:
                    guarded.append("forbids" in str(exc))
                else:
                    guarded.append(False)
            self.guard_observations.append(tuple(guarded))
        if self.restore_complete_failure is not None:
            raise self.restore_complete_failure
        return self.restore_complete_result

    def action_ball_r10_publish(self, *, publication, handoff):
        self.publication = publication
        return {"update": handoff.completed_update_index}


_DRAIN_OWNER_ORDER = (
    "r05_runtime",
    "motion",
    "racket",
    "physical_ball",
    "r06_landing_outcome",
    "r03_strike_fact",
    "r07_recovery",
)
_DRAIN_SCHEMA_IDENTITY = tuple(
    (owner_id, (("mutation_version", "scalar", 0),))
    for owner_id in _DRAIN_OWNER_ORDER
)


class _Projection:
    pass


def _projection_payload(projection, boundary):
    return {
        "schema_version": projection.schema_version,
        "kind": projection.kind,
        "num_envs": projection.num_envs,
        "device_type": projection.device_type,
        "device_index": projection.device_index,
        "owner_order": projection.owner_order,
        "schema_identity": projection.schema_identity,
        "checkpoint_boundary_sha256": C.boundary_sha256(boundary),
        "next_update_index": projection.next_update_index,
        "operation_sequence": projection.operation_sequence,
        "drain_sequence": projection.drain_sequence,
        "last_completed_environment_steps": (
            projection.last_completed_environment_steps
        ),
        "mutation_version_highwaters": (
            projection.mutation_version_highwaters
        ),
    }


class _GlobalDrainOwner:
    """Independent fake global writer exposing only clone-only primitives."""

    def __init__(self) -> None:
        self.latest = None
        self.projection = None
        self.schema_identity = _DRAIN_SCHEMA_IDENTITY

    def acknowledge(self, token, update_index, completed_environment_steps):
        self.latest = token
        value = object.__new__(_Projection)
        value.schema_version = 1
        value.kind = "action_ball_full_mdp_ppo_drain_checkpoint_v1"
        value.num_envs = 2
        value.device_type = "cpu"
        value.device_index = None
        value.owner_order = _DRAIN_OWNER_ORDER
        value.schema_identity = self.schema_identity
        value.next_update_index = update_index + 1
        value.operation_sequence = update_index + 1
        value.drain_sequence = update_index + 1
        value.last_completed_environment_steps = completed_environment_steps
        value.mutation_version_highwaters = tuple(
            (owner_id, update_index + 1) for owner_id in _DRAIN_OWNER_ORDER
        )
        value.update_index = update_index
        value.completed_environment_steps = completed_environment_steps
        self.projection = value

    def require_owned_runner_frontier_projection(self, receipt):
        if receipt is not self.latest or self.projection is None:
            raise RuntimeError("stale or foreign fake global receipt")
        return self.projection

    def frontier(self, boundary):
        return C.canonical_sha256(_projection_payload(self.projection, boundary))


class _RuntimeDrainOwner:
    """Minimal top owner paired with a separate fake global drain."""

    def __init__(self) -> None:
        self.healthy = True
        self.prepared = []
        self.optimizer_returns = []
        self.acknowledged = []
        self.poisons = []
        self.global_drain = _GlobalDrainOwner()

    def require_healthy(self):
        if not self.healthy:
            raise RuntimeError("test full-MDP owner is poisoned")

    def prepare_pre_optimizer_ppo_boundary(
        self, *, update_index, completed_environment_steps
    ):
        token = object()
        self.prepared.append(
            (token, update_index, completed_environment_steps)
        )
        return token

    def mark_optimizer_returned(self, token, *, update_index):
        assert self.prepared
        assert token is self.prepared[-1][0]
        assert update_index == self.prepared[-1][1]
        self.optimizer_returns.append((token, update_index))

    def acknowledge_post_update(self, token, *, update_index):
        assert self.optimizer_returns
        assert token is self.optimizer_returns[-1][0]
        assert update_index == self.optimizer_returns[-1][1]
        self.acknowledged.append((token, update_index))
        completed = self.prepared[-1][2]
        self.global_drain.acknowledge(token, update_index, completed)
        return object()

    def poison_optimizer_boundary(
        self, token, *, update_index, reason
    ):
        self.healthy = False
        self.poisons.append((token, update_index, reason))


class _IncompleteOwnerAdapter(_Adapter):
    def action_ball_r10_complete_owners(
        self, *, operation, runner_owners, verified_checkpoint
    ):
        owners = super().action_ball_r10_complete_owners(
            operation=operation,
            runner_owners=runner_owners,
            verified_checkpoint=verified_checkpoint,
        )
        return owners[:-1]


def _runner(
    runner_module,
    env: _Env,
    *,
    adapter=None,
    capsule=None,
    leave_storage_full: bool = False,
    contract_schema_version: int = 1,
    contract_sha256: str | None = None,
    num_steps_per_env: int = 2,
    runner_api_source_sha256: str | None = None,
    install_runner_api: bool = True,
    runtime_owner=None,
):
    if adapter is not None and runtime_owner is None:
        runtime_owner = _RuntimeDrainOwner()
    if adapter is not None:
        env._drain_owner = runtime_owner.global_drain
        adapter.runtime_owner = runtime_owner
        adapter._source = SimpleNamespace(
            family=getattr(adapter, "source_family", C.PolicyFamily.A)
        )
        adapter.module = runner_module
        adapter.env = env
        if install_runner_api:
            adapter._runner_api = E.RunnerCheckpointApiAuthority(
                schema_version=E.SCHEMA_VERSION,
                kind="action_ball_r10_runner_checkpoint_api_authority_v1",
                source_sha256=(
                    runner_api_source_sha256
                    or hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest()
                ),
                training_launch_claim_sha256=_sha("launch-claim"),
                boundary_handoff_type=(
                    runner_module.ActionBallR10RunnerBoundaryHandoff
                ),
                save_authority_type=(
                    runner_module.ActionBallR10RunnerSaveAuthority
                ),
            )
    return runner_module.MotionOnPolicyRunner(
        env,
        {
            "num_steps_per_env": num_steps_per_env,
            "leave_storage_full": leave_storage_full,
        },
        log_dir="/tmp/r10-runner-test",
        training_contract_schema_version=contract_schema_version,
        training_contract_sha256=contract_sha256 or _sha("contract"),
        training_contract_lineage_exact=True,
        training_launch_claim_sha256=(
            _sha("launch-claim") if adapter is not None else None
        ),
        require_exact_resume_state=adapter is not None,
        action_ball_r10_checkpoint_adapter=adapter,
        action_ball_r10_cold_restore_capsule=capsule,
        action_ball_full_mdp_runtime_owner=runtime_owner,
    )


def _source_checkpoint(runner_module):
    adapter = _Adapter(save=True)
    env = _Env()
    runner = _runner(runner_module, env, adapter=adapter)
    runner.learn(1)
    assert adapter.publication is not None
    return runner, env, adapter, adapter.publication


def test_legacy_default_path_never_enters_r10(runner_module):
    env = _Env()
    runner = _runner(runner_module, env)
    runner.learn(1)
    assert env.getter_calls == 2
    assert env.reset_calls == 0
    assert env.step_calls == 2
    assert not hasattr(runner, "_action_ball_r10_last_boundary_handoff") or (
        runner._action_ball_r10_last_boundary_handoff is None
    )


def test_restore_guard_cleanup_continues_after_unprintable_base_exception(
    runner_module,
):
    env = _GuardCleanupEnv()
    guard = runner_module._ActionBallR10EnvMethodGuard(env)
    guard.install_restore_guards()
    with pytest.raises(
        RuntimeError,
        match=r"step:.*_UnprintableCleanupError",
    ):
        guard.close()
    assert env.cleanup_attempts == ["step", "reset", "get_observations"]
    assert "reset" not in vars(env)
    assert "get_observations" not in vars(env)
    assert "step" in vars(env)
    object.__delattr__(env, "step")


def test_real_runner_callsite_seals_and_cold_restores_without_sampling(
    runner_module,
):
    source, _source_env, source_adapter, publication = _source_checkpoint(
        runner_module
    )
    verified = C.verify_checkpoint_candidate(
        publication.blob,
        expected_external_pins=publication.external_pins,
        expected_registry=source_adapter.registry,
    )
    assert tuple(state.owner_id for state in verified.owner_states) == (
        C.R10_OWNER_ORDER
    )
    assert len(verified.owner_states) == 21
    assert sum(
        not state.owner_id.startswith("trainer.")
        for state in verified.owner_states
    ) == 16
    assert verified.owner_root_sha256 == (
        publication.external_pins.owner_root_sha256
    )
    saved_by_owner = {state.owner_id: state for state in verified.owner_states}
    saved_policy_payload = runner_module._action_ball_r10_torch_load_bytes(
        saved_by_owner["trainer.policy"].payload
    )
    assert tuple(saved_policy_payload["model_state_dict"]) == ("weight", "std")
    assert torch.equal(
        saved_policy_payload["model_state_dict"]["std"],
        source.alg.policy.std,
    )
    saved_optimizer_payload = runner_module._action_ball_r10_torch_load_bytes(
        saved_by_owner["trainer.optimizer_schedule"].payload
    )
    assert saved_optimizer_payload["scheduler_representation"] == (
        "rsl_rl_2_3_1_inline_schedule_and_lr"
    )
    assert saved_optimizer_payload["schedule"] == source.alg.schedule
    assert saved_optimizer_payload["learning_rate"] == source.alg.learning_rate
    assert saved_optimizer_payload["param_group_lrs"] == (
        source.alg.learning_rate,
    )
    adam_states = saved_optimizer_payload["optimizer_state_dict"]["state"]
    assert len(adam_states) == 2
    assert all(
        {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
        for state in adam_states.values()
    )
    for owner_id, live_normalizer in (
        ("trainer.normalizer.actor", source.obs_normalizer),
        ("trainer.normalizer.critic", source.privileged_obs_normalizer),
    ):
        normalizer_payload = runner_module._action_ball_r10_torch_load_bytes(
            saved_by_owner[owner_id].payload
        )
        assert tuple(normalizer_payload["state_dict"]) == ("mean", "std", "count")
        assert torch.equal(
            normalizer_payload["state_dict"]["count"],
            live_normalizer.state_dict()["count"],
        )
        assert normalizer_payload["state_dict"]["count"].item() > 0.0
    saved_frontier_payload = runner_module._action_ball_r10_torch_load_bytes(
        saved_by_owner["trainer.rollout_frontier"].payload
    )
    assert saved_frontier_payload["storage_step"] == 0
    assert saved_frontier_payload["transition_empty"] is True
    assert saved_frontier_payload["recurrent_frontier"] == (
        C.RecurrentFrontierStatus.SEALED.value
    )
    source_policy_sha = source._exact_resume_tree_sha256(
        source._action_ball_r10_policy_state()
    )
    source_optimizer_sha = source._exact_resume_tree_sha256(
        source._action_ball_r10_optimizer_state()
    )
    source_actor_norm_sha = source._exact_resume_tree_sha256(
        source._action_ball_r10_normalizer_state("actor")
    )
    source_critic_norm_sha = source._exact_resume_tree_sha256(
        source._action_ball_r10_normalizer_state("critic")
    )
    source_hidden = tuple(
        item.detach().clone() for item in source.alg.policy.get_hidden_states()
    )
    saved_actor = source_adapter.handoffs[0].final_normalized_actor_observations
    saved_critic = source_adapter.handoffs[0].final_normalized_critic_observations
    assert publication.runtime_wiring is False
    assert publication.continuation_authorized is False
    assert source_adapter.handoffs[0].runtime_wiring is False

    target_env = _Env(raw_offset=100.0)
    target_adapter = _Adapter(save=False)
    target_adapter.module = runner_module
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    target = _runner(
        runner_module,
        target_env,
        adapter=target_adapter,
        capsule=capsule,
    )

    assert target_adapter.restore_complete_calls == [
        target._action_ball_r10_restore_receipt
    ]
    assert target_adapter.guard_observations == [(True, True, True)]
    assert target_env.getter_calls == 0
    assert target_env.noise_calls == 0
    assert target_env.reset_calls == 0
    assert target_env.step_calls == 0
    assert target.obs_normalizer.forward_calls == 0
    assert target.privileged_obs_normalizer.forward_calls == 0
    assert target.current_learning_iteration == 1
    assert target.alg.learning_rate == pytest.approx(source.alg.learning_rate)
    assert target.alg.schedule == source.alg.schedule
    assert target.alg.desired_kl == source.alg.desired_kl
    assert [group["lr"] for group in target.alg.optimizer.param_groups] == [
        source.alg.learning_rate
    ]
    assert target._exact_resume_tree_sha256(
        target._action_ball_r10_policy_state()
    ) == source_policy_sha
    assert target._exact_resume_tree_sha256(
        target._action_ball_r10_optimizer_state()
    ) == source_optimizer_sha
    assert target._exact_resume_tree_sha256(
        target._action_ball_r10_normalizer_state("actor")
    ) == source_actor_norm_sha
    assert target._exact_resume_tree_sha256(
        target._action_ball_r10_normalizer_state("critic")
    ) == source_critic_norm_sha
    for actual, expected in zip(target.alg.policy.get_hidden_states(), source_hidden):
        assert torch.equal(actual, expected)

    target.learn(1)
    assert target_env.getter_calls == 0
    assert target_env.noise_calls == 0
    assert target_env.reset_calls == 0
    assert target_env.step_calls == 2
    assert target.obs_normalizer.forward_calls == 2
    assert target.privileged_obs_normalizer.forward_calls == 2
    assert torch.equal(target.alg.first_act_observations, saved_actor)
    assert torch.equal(target.alg.first_act_critic_observations, saved_critic)
    for actual, expected in zip(target.alg.first_act_hidden, source_hidden):
        assert torch.equal(actual, expected)
    assert len(target_adapter.handoffs) == 1
    assert target_adapter.handoffs[0].completed_update_index == 1


def test_bind_rejects_missing_or_wrong_runner_api_source_before_sampling(
    runner_module,
):
    for kwargs in (
        {"install_runner_api": False},
        {"runner_api_source_sha256": _sha("wrong-runner-source")},
    ):
        env = _Env(raw_offset=125.0)
        adapter = _Adapter(save=False)
        with pytest.raises(
            RuntimeError,
            match="runner API/source authority differs",
        ):
            _runner(runner_module, env, adapter=adapter, **kwargs)
        assert env.getter_calls == 0
        assert env.noise_calls == 0
        assert env.reset_calls == 0
        assert env.step_calls == 0


def test_missing_restore_complete_callback_fails_before_sampling(
    runner_module,
):
    _source, _source_env, _source_adapter, publication = _source_checkpoint(
        runner_module
    )
    env = _Env(raw_offset=130.0)
    adapter = _Adapter(save=False)
    adapter.action_ball_r10_restore_complete = None
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    with pytest.raises(
        RuntimeError,
        match="lacks action_ball_r10_restore_complete",
    ):
        _runner(runner_module, env, adapter=adapter, capsule=capsule)
    assert env.getter_calls == 0
    assert env.noise_calls == 0
    assert env.reset_calls == 0
    assert env.step_calls == 0


def test_cold_restore_requires_adapter_before_constructor_sampling(
    runner_module,
):
    _source, _source_env, _source_adapter, publication = _source_checkpoint(
        runner_module
    )
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    env = _Env(raw_offset=135.0)
    with pytest.raises(RuntimeError, match="requires an explicit full-MDP adapter"):
        _runner(runner_module, env, capsule=capsule)
    assert env.getter_calls == 0
    assert env.noise_calls == 0
    assert env.reset_calls == 0
    assert env.step_calls == 0


def test_restore_complete_failure_broadcasts_all_21_despite_poison_error(
    runner_module,
):
    _source, _source_env, source_adapter, publication = _source_checkpoint(
        runner_module
    )
    target_env = _Env(raw_offset=140.0)
    target_adapter = _Adapter(save=False)
    target_adapter.restore_complete_failure = RuntimeError("callback failed")
    target_adapter.external["rng.per_world"].poison_failure = RuntimeError(
        "poison failed"
    )
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    with pytest.raises(
        RuntimeError,
        match="completion failed; retry is forbidden",
    ):
        _runner(
            runner_module,
            target_env,
            adapter=target_adapter,
            capsule=capsule,
        )
    assert len(target_adapter.restore_complete_calls) == 1
    assert target_adapter.guard_observations == [(True, True, True)]
    assert tuple(target_adapter.poison_attempts) == source_adapter.registry.owner_ids
    trainer_index = source_adapter.registry.owner_ids.index("trainer.policy")
    failed_runner = (
        target_adapter.last_complete_owners[trainer_index]._poison.__self__
    )
    assert failed_runner._action_ball_r10_runtime_poisoned is True
    assert failed_runner._action_ball_r10_restore_poison_attempted_owner_ids == (
        source_adapter.registry.owner_ids
    )
    assert failed_runner._action_ball_r10_restore_poison_failures == (
        "rng.per_world:builtins.RuntimeError",
    )
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        failed_runner._action_ball_r10_require_healthy()
    assert target_env.getter_calls == 0
    assert target_env.noise_calls == 0
    assert target_env.reset_calls == 0
    assert target_env.step_calls == 0


def test_wrong_restore_receipt_is_rejected_before_adapter_callback(
    runner_module,
    monkeypatch,
):
    _source, _source_env, source_adapter, publication = _source_checkpoint(
        runner_module
    )
    real_restore = C.CheckpointRestoreCoordinator.restore

    def corrupt_receipt(coordinator, *args, **kwargs):
        receipt = real_restore(coordinator, *args, **kwargs)
        external_only = tuple(
            owner_id
            for owner_id in receipt.prepared_owner_ids
            if not owner_id.startswith("trainer.")
        )
        return replace(
            receipt,
            prepared_owner_ids=external_only,
            committed_owner_ids=external_only,
        )

    monkeypatch.setattr(C.CheckpointRestoreCoordinator, "restore", corrupt_receipt)
    target_env = _Env(raw_offset=145.0)
    target_adapter = _Adapter(save=False)
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    with pytest.raises(
        RuntimeError,
        match="completion failed; retry is forbidden",
    ):
        _runner(
            runner_module,
            target_env,
            adapter=target_adapter,
            capsule=capsule,
        )
    assert target_adapter.restore_complete_calls == []
    assert tuple(target_adapter.poison_attempts) == source_adapter.registry.owner_ids
    trainer_index = source_adapter.registry.owner_ids.index("trainer.policy")
    failed_runner = (
        target_adapter.last_complete_owners[trainer_index]._poison.__self__
    )
    assert failed_runner._action_ball_r10_runtime_poisoned is True
    assert failed_runner._action_ball_r10_restore_completion_attempted is True
    assert failed_runner._action_ball_r10_restore_receipt is None
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        failed_runner._action_ball_r10_require_healthy()
    assert target_env.getter_calls == 0
    assert target_env.noise_calls == 0
    assert target_env.reset_calls == 0
    assert target_env.step_calls == 0


def test_restore_complete_is_one_shot_and_second_attempt_poisoned(
    runner_module,
):
    _source, _source_env, source_adapter, publication = _source_checkpoint(
        runner_module
    )
    verified = C.verify_checkpoint_candidate(
        publication.blob,
        expected_external_pins=publication.external_pins,
        expected_registry=source_adapter.registry,
    )
    target_adapter = _Adapter(save=False)
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    target = _runner(
        runner_module,
        _Env(raw_offset=148.0),
        adapter=target_adapter,
        capsule=capsule,
    )
    assert len(target_adapter.restore_complete_calls) == 1
    external_counts = {
        owner_id: (owner.prepare_calls, owner.commit_calls)
        for owner_id, owner in target_adapter.external.items()
    }
    saved_by_owner = {
        state.owner_id: state for state in verified.owner_states
    }
    cold_frontier = runner_module._action_ball_r10_torch_load_bytes(
        saved_by_owner["trainer.rollout_frontier"].payload
    )
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        target._action_ball_r10_restore_cold_checkpoint(
            verified_checkpoint=verified,
            capsule=capsule,
            cold_frontier=cold_frontier,
        )
    assert len(target_adapter.restore_complete_calls) == 1
    assert {
        owner_id: (owner.prepare_calls, owner.commit_calls)
        for owner_id, owner in target_adapter.external.items()
    } == external_counts
    assert tuple(target_adapter.poison_attempts) == source_adapter.registry.owner_ids
    assert target._action_ball_r10_runtime_poisoned is True
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        target.learn(1)


def test_missing_registry_provider_fails_before_base_constructor_sampling(
    runner_module,
):
    env = _Env(raw_offset=150.0)
    adapter = SimpleNamespace(
        action_ball_r10_restore_complete=lambda _receipt: None
    )
    with pytest.raises(RuntimeError, match=r"lacks action_ball_r10_registry\(\)"):
        _runner(runner_module, env, adapter=adapter)
    assert env.getter_calls == 0
    assert env.noise_calls == 0
    assert env.reset_calls == 0
    assert env.step_calls == 0


def test_wrong_external_pin_fails_before_constructor_getter(runner_module):
    _source, _source_env, _source_adapter, publication = _source_checkpoint(
        runner_module
    )
    target_env = _Env(raw_offset=200.0)
    adapter = _Adapter(save=False)
    adapter.module = runner_module
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=replace(
            publication.external_pins,
            checkpoint_size_bytes=publication.external_pins.checkpoint_size_bytes + 1,
        ),
    )
    with pytest.raises(C.ExternalCheckpointPinError, match="length"):
        _runner(
            runner_module,
            target_env,
            adapter=adapter,
            capsule=capsule,
        )
    assert target_env.getter_calls == 0
    assert target_env.noise_calls == 0
    assert target_env.reset_calls == 0


def test_missing_full_environment_owner_set_fails_closed_without_sampling(
    runner_module,
):
    _source, _source_env, _source_adapter, publication = _source_checkpoint(
        runner_module
    )
    target_env = _Env(raw_offset=300.0)
    adapter = _IncompleteOwnerAdapter(save=False)
    adapter.module = runner_module
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    with pytest.raises(RuntimeError, match="incomplete owner tuple"):
        _runner(
            runner_module,
            target_env,
            adapter=adapter,
            capsule=capsule,
        )
    assert target_env.getter_calls == 0
    assert target_env.noise_calls == 0
    assert target_env.reset_calls == 0
    assert target_env.step_calls == 0


def test_save_authority_cannot_relabel_runner_under_another_contract(
    runner_module,
):
    env = _Env()
    adapter = _Adapter(save=True, contract_sha256=_sha("foreign-contract"))
    runner = _runner(runner_module, env, adapter=adapter)
    with pytest.raises(RuntimeError, match="contract pin differs"):
        runner.learn(1)
    assert adapter.publication is None
    assert runner._action_ball_r10_runtime_poisoned is True
    runtime_owner = runner._action_ball_full_mdp_runtime_owner
    assert len(runtime_owner.prepared) == 1
    assert runtime_owner.acknowledged == [
        (runtime_owner.prepared[0][0], 0)
    ]
    assert len(runtime_owner.poisons) == 1
    assert runtime_owner.poisons[0][0] is runtime_owner.prepared[0][0]
    assert runtime_owner.poisons[0][1] == 0
    assert runner._action_ball_full_mdp_boundary_poisoned is True
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        runner.learn(1)
    assert len(runtime_owner.prepared) == 1
    with pytest.raises(RuntimeError, match="checkpoint is forbidden"):
        runner._checkpoint_infos()


@pytest.mark.parametrize(
    ("schema_version", "contract_sha256"),
    (
        (2, _sha("contract")),
        (1, _sha("contract").upper()),
    ),
)
def test_r10_contract_pin_is_rejected_before_base_constructor_sampling(
    runner_module,
    schema_version,
    contract_sha256,
):
    env = _Env()
    adapter = _Adapter(save=False)
    with pytest.raises(RuntimeError, match="immutable contract pin"):
        _runner(
            runner_module,
            env,
            adapter=adapter,
            contract_schema_version=schema_version,
            contract_sha256=contract_sha256,
        )
    assert env.getter_calls == 0
    assert env.noise_calls == 0
    assert env.reset_calls == 0
    assert env.step_calls == 0


def test_save_authority_must_cover_every_runner_world(runner_module):
    env = _Env()
    adapter = _Adapter(save=True, boundary_world_ids=(0,))
    runner = _runner(runner_module, env, adapter=adapter)
    with pytest.raises(RuntimeError, match="world identity differs"):
        runner.learn(1)
    assert adapter.publication is None
    assert runner._action_ball_r10_runtime_poisoned is True


def test_cold_restore_step_horizon_mismatch_fails_before_base_sampling(
    runner_module,
):
    _source, _source_env, _source_adapter, publication = _source_checkpoint(
        runner_module
    )
    env = _Env(raw_offset=400.0)
    adapter = _Adapter(save=False)
    capsule = runner_module.ActionBallR10ColdRestoreCapsule(
        schema_version=1,
        kind="action_ball_r10_isaac_runner_cold_restore_capsule_v1",
        checkpoint_bytes=publication.blob,
        expected_external_pins=publication.external_pins,
    )
    with pytest.raises(RuntimeError, match="rollout/world identity differs"):
        _runner(
            runner_module,
            env,
            adapter=adapter,
            capsule=capsule,
            num_steps_per_env=3,
        )
    assert env.getter_calls == 0
    assert env.noise_calls == 0
    assert env.reset_calls == 0
    assert env.step_calls == 0


def test_post_update_storage_must_really_be_empty(runner_module):
    env = _Env()
    adapter = _Adapter(save=False)
    runner = _runner(
        runner_module,
        env,
        adapter=adapter,
        leave_storage_full=True,
    )
    with pytest.raises(RuntimeError, match="storage is not empty"):
        runner.learn(1)
    assert adapter.handoffs == []
    assert runner._action_ball_r10_runtime_poisoned is True
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        runner.learn(1)


@pytest.mark.parametrize("family", (C.PolicyFamily.A, C.PolicyFamily.C))
def test_fresh_full_mdp_lane_accepts_only_exact_target_and_dynamic_widths(
    runner_module,
    family,
):
    env = _Env(
        obs_mode="action_ball_full_mdp",
        target_mode="action_ball_full_mdp",
        actor_width=263,
        critic_width=263,
    )
    adapter = _Adapter(save=True)
    adapter.source_family = family
    original_post_update = adapter.action_ball_r10_post_update_authority

    def family_authority(handoff):
        value = original_post_update(handoff)
        return value._replace(family=family)

    adapter.action_ball_r10_post_update_authority = family_authority
    adapter._source = SimpleNamespace(family=family)
    runner = _runner(runner_module, env, adapter=adapter)
    adapter._source = SimpleNamespace(family=family)
    runner.learn(1)

    assert adapter.publication is not None
    assert runner._strict_exact_resume_target_mode() == "action_ball_full_mdp"
    assert runner._action_ball_r10_last_boundary_handoff is not None
    callback = runner.action_ball_full_mdp_runner_ppo_drain_frontier
    assert callback.__self__ is runner
    assert callback(
        adapter.publication.external_pins
        and C.verify_checkpoint_candidate(
            adapter.publication.blob,
            expected_external_pins=adapter.publication.external_pins,
            expected_registry=adapter.registry,
        ).boundary
    ) == adapter.runtime_owner.global_drain.frontier(
        C.verify_checkpoint_candidate(
            adapter.publication.blob,
            expected_external_pins=adapter.publication.external_pins,
            expected_registry=adapter.registry,
        ).boundary
    )


def test_fresh_full_mdp_wrong_target_is_not_silently_broadened(runner_module):
    env = _Env(
        obs_mode="action_ball_full_mdp",
        target_mode="action_ball",
        actor_width=263,
        critic_width=371,
    )
    adapter = _Adapter(save=False)
    with pytest.raises(RuntimeError, match="target_mode differs"):
        _runner(runner_module, env, adapter=adapter)
    assert env.getter_calls == 1


def test_physical_ppo_drain_mismatch_poisoned_without_publication(
    runner_module,
):
    env = _Env()
    adapter = _Adapter(save=True)
    adapter.physical_ppo_drain_frontier_override = _sha("forged-physical")
    runner = _runner(runner_module, env, adapter=adapter)

    with pytest.raises(C.CheckpointJoinError, match="ppo_drain_frontier"):
        runner.learn(1)
    assert adapter.publication is None
    assert runner._action_ball_full_mdp_boundary_poisoned is True
    assert len(runner._action_ball_full_mdp_runtime_owner.poisons) == 1
    with pytest.raises(RuntimeError, match="checkpoint is forbidden"):
        runner._checkpoint_infos()
