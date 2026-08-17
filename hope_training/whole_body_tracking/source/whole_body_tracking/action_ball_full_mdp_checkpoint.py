"""Engine-neutral transaction seam for a future ActionBall R10 checkpoint.

This module deliberately has no Isaac, MuJoCo, Torch, runner, or filesystem
dependency.  It can seal and verify *opaque* owner bytes, enforce the only
admissible PPO/environment boundary, and coordinate prepare-all/commit/poison.
It cannot attest that either simulator exported all integration state and it
does not authorize mid-sequence resume by itself.

The distinction is intentional:

* ``SEALED`` here means that typed bytes and joins are internally consistent.
* Exact simulator continuation remains an engine adapter and fresh-process
  fixed-tape obligation outside this module.
* ``RUNTIME_WIRING`` and every launch/continuation authority stay false.

Checkpoint bytes are deterministic JSON containing base64 owner payloads.  A
loader must receive the byte digest, byte length, owner Merkle root, and the
code/config/contract pins from an independent launcher or supervisor.  Values
read from the checkpoint itself are never accepted as their own authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import base64
import hashlib
import inspect
import json
import marshal
import re
import threading
import types
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple
import weakref


SCHEMA_VERSION = 1
CHECKPOINT_KIND = "action_ball_full_mdp_checkpoint_candidate_v1"
CHECKPOINT_RECEIPT_KIND = "action_ball_full_mdp_checkpoint_candidate_receipt_v1"
INTEGRATION_STATUS = "PRE_INTEGRATION_HOLD"
CONTINUATION_CLAIM = "none_pure_transaction_seam_only"
RUNTIME_WIRING = False
ENGINE_CONTINUATION_PROVEN = False
FORMAL_EXACT_RESUME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
PPO_DRAIN_FRONTIER_JOIN_CONTRACT_IMPLEMENTED = True
PPO_DRAIN_FRONTIER_SECOND_WRITER_INTEGRATED = False
PPO_DRAIN_LIVE_HIGHWATER_JOIN_INTEGRATED = False
CHECKPOINT_PUBLICATION_FINALIZER_INTEGRATED = False
PPO_DRAIN_LEAF_OWNER_ORDER = (
    "r05_runtime",
    "motion",
    "racket",
    "physical_ball",
    "r06_landing_outcome",
    "r03_strike_fact",
    "r07_recovery",
)
R03_CONSUMER_COUNT = 10
R03_FULL_CONSUMER_MASK = (1 << R03_CONSUMER_COUNT) - 1
R06_CONSUMER_COUNT = 2
R06_FULL_CONSUMER_MASK = (1 << R06_CONSUMER_COUNT) - 1

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_OWNER_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_JOIN_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


class FullMDPCheckpointError(RuntimeError):
    """Base class for a fail-closed R10 pure-seam refusal."""


class OwnerRegistryError(FullMDPCheckpointError):
    """The typed owner registry is missing, ambiguous, or not topological."""


class CheckpointBoundaryError(FullMDPCheckpointError):
    """The requested save is not at the single authorized R10-v1 boundary."""


class CheckpointJoinError(FullMDPCheckpointError):
    """Opaque owners disagree about a required cross-owner identity."""


class CheckpointSealError(FullMDPCheckpointError):
    """A checkpoint envelope or owner export is malformed or internally stale."""


class CheckpointPublicationFinalizationError(CheckpointSealError):
    """A sealed candidate lacks independent drain or durable-publication proof."""


class ExternalCheckpointPinError(CheckpointSealError):
    """Checkpoint bytes do not match independently supplied authority."""


class CheckpointPrepareError(FullMDPCheckpointError):
    """At least one owner could not prepare observationally."""

    def __init__(self, message: str, *, runtime_poisoned: bool) -> None:
        super().__init__(message)
        self.runtime_poisoned = runtime_poisoned
        self.retry_permitted = not runtime_poisoned


class CheckpointCommitError(FullMDPCheckpointError):
    """Restore commit failed; the coordinator is permanently poisoned."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.runtime_poisoned = True
        self.retry_permitted = False


class CheckpointRuntimePoisonedError(FullMDPCheckpointError):
    """A restore was attempted after a fail-stop commit/rollback failure."""


class OwnerScope(str, Enum):
    RUN = "run"
    WORLD = "world"
    SLOT = "slot"


class OwnerEngine(str, Enum):
    PORTABLE = "portable"
    ISAAC = "isaac"
    MUJOCO = "mujoco"


class PolicyFamily(str, Enum):
    A = "A"
    C = "C"


class SealStatus(str, Enum):
    PREPARING = "PREPARING"
    PARTIAL = "PARTIAL"
    SEALED = "SEALED"


class PPOBoundaryPhase(str, Enum):
    POST_UPDATE_ROLLOUT_EMPTY = "POST_UPDATE_ROLLOUT_EMPTY"
    ROLLOUT_COLLECTING = "ROLLOUT_COLLECTING"
    GAE_RETURNS = "GAE_RETURNS"
    MINIBATCH_UPDATE = "MINIBATCH_UPDATE"
    OPTIMIZER_STEP = "OPTIMIZER_STEP"


class EnvironmentStepPhase(str, Enum):
    BETWEEN_COMPLETE_STEPS = "BETWEEN_COMPLETE_STEPS"
    PRE_PHYSICS = "PRE_PHYSICS"
    PHYSICS = "PHYSICS"
    POST_PHYSICS = "POST_PHYSICS"
    MANAGER_PAYMENT = "MANAGER_PAYMENT"
    RESET = "RESET"


class RecurrentFrontierStatus(str, Enum):
    SEALED = "SEALED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PARTIAL = "PARTIAL"


class ResetPhase(str, Enum):
    COMMITTED = "COMMITTED"
    PREPARED = "PREPARED"
    COMMITTING = "COMMITTING"
    COMMITTED_TAIL_PENDING = "COMMITTED_TAIL_PENDING"


class R05Phase(str, Enum):
    EMPTY = "EMPTY"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    PARTIAL = "PARTIAL"


class R03Phase(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    PENDING = "PENDING"
    PAID = "PAID"


class R06FlightPhase(str, Enum):
    EMPTY = "EMPTY"
    INBOUND = "INBOUND"
    OPEN = "OPEN"
    SETTLED_RETAINED = "SETTLED_RETAINED"


class R06MailboxPhase(str, Enum):
    EMPTY = "EMPTY"
    SETTLED_UNPAID = "SETTLED_UNPAID"
    PARTIALLY_PAID = "PARTIALLY_PAID"
    PAID = "PAID"


@dataclass(frozen=True)
class PhaseMatrixRule:
    """Machine-readable disposition; ``condition`` names the narrow exception."""

    owner: str
    phase: str
    allowed: bool
    condition: str


R10_PHASE_MATRIX = (
    PhaseMatrixRule(
        "ppo",
        PPOBoundaryPhase.POST_UPDATE_ROLLOUT_EMPTY.value,
        True,
        "final_actor_critic_and_recurrent_frontier_sealed",
    ),
    PhaseMatrixRule("ppo", PPOBoundaryPhase.ROLLOUT_COLLECTING.value, False, "none"),
    PhaseMatrixRule("ppo", PPOBoundaryPhase.GAE_RETURNS.value, False, "none"),
    PhaseMatrixRule("ppo", PPOBoundaryPhase.MINIBATCH_UPDATE.value, False, "none"),
    PhaseMatrixRule("ppo", PPOBoundaryPhase.OPTIMIZER_STEP.value, False, "none"),
    PhaseMatrixRule(
        "environment",
        EnvironmentStepPhase.BETWEEN_COMPLETE_STEPS.value,
        True,
        "no_step_reset_physics_or_manager_operation_in_flight",
    ),
    PhaseMatrixRule(
        "environment", EnvironmentStepPhase.PRE_PHYSICS.value, False, "none"
    ),
    PhaseMatrixRule("environment", EnvironmentStepPhase.PHYSICS.value, False, "none"),
    PhaseMatrixRule(
        "environment", EnvironmentStepPhase.POST_PHYSICS.value, False, "none"
    ),
    PhaseMatrixRule(
        "environment", EnvironmentStepPhase.MANAGER_PAYMENT.value, False, "none"
    ),
    PhaseMatrixRule("environment", EnvironmentStepPhase.RESET.value, False, "none"),
    PhaseMatrixRule("reset", ResetPhase.COMMITTED.value, True, "frontier_installed"),
    PhaseMatrixRule("reset", ResetPhase.PREPARED.value, False, "none"),
    PhaseMatrixRule("reset", ResetPhase.COMMITTING.value, False, "none"),
    PhaseMatrixRule(
        "reset", ResetPhase.COMMITTED_TAIL_PENDING.value, False, "none"
    ),
    PhaseMatrixRule("R05", R05Phase.EMPTY.value, True, "operation_inactive"),
    PhaseMatrixRule(
        "R05", R05Phase.PREPARED.value, True, "private_batch_sealed_and_hidden"
    ),
    PhaseMatrixRule(
        "R05", R05Phase.COMMITTED.value, True, "cross_owner_commit_join_complete"
    ),
    PhaseMatrixRule("R05", R05Phase.PARTIAL.value, False, "none"),
    PhaseMatrixRule("R03", R03Phase.IDLE.value, True, "none"),
    PhaseMatrixRule("R03", R03Phase.PAID.value, True, "all_ten_consumers_paid"),
    PhaseMatrixRule("R03", R03Phase.ARMED.value, False, "none"),
    PhaseMatrixRule("R03", R03Phase.PENDING.value, False, "none"),
    PhaseMatrixRule(
        "R06.flight", R06FlightPhase.EMPTY.value, True, "between_complete_steps"
    ),
    PhaseMatrixRule(
        "R06.flight", R06FlightPhase.INBOUND.value, True, "between_complete_steps"
    ),
    PhaseMatrixRule(
        "R06.flight", R06FlightPhase.OPEN.value, True, "between_complete_steps"
    ),
    PhaseMatrixRule(
        "R06.flight",
        R06FlightPhase.SETTLED_RETAINED.value,
        True,
        "between_complete_steps",
    ),
    PhaseMatrixRule(
        "R06.mailbox", R06MailboxPhase.EMPTY.value, True, "payment_epoch_closed"
    ),
    PhaseMatrixRule(
        "R06.mailbox",
        R06MailboxPhase.SETTLED_UNPAID.value,
        True,
        "payment_epoch_closed",
    ),
    PhaseMatrixRule(
        "R06.mailbox",
        R06MailboxPhase.PARTIALLY_PAID.value,
        True,
        "payment_epoch_closed",
    ),
    PhaseMatrixRule(
        "R06.mailbox", R06MailboxPhase.PAID.value, True, "payment_epoch_closed"
    ),
    PhaseMatrixRule("R06", "PAYMENT_EPOCH_OPEN", False, "none"),
    PhaseMatrixRule("R07", "RECOVERY_OR_HOLD", True, "payment_epoch_closed"),
    PhaseMatrixRule("R07", "PAYMENT_EPOCH_OR_DEADLINE_ACK_OPEN", False, "none"),
)


@dataclass(frozen=True)
class OwnerDescriptor:
    """Static ABI and restore dependency for exactly one mutation owner."""

    owner_id: str
    schema_version: int
    state_kind: str
    scope: OwnerScope
    engine: OwnerEngine
    immutable_identity_sha256: str
    dependencies: Tuple[str, ...]
    restore_rank: int


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise CheckpointSealError(label + " must be a lowercase SHA-256")
    return value


def _require_exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise CheckpointSealError(label + " must be an exact bool")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CheckpointSealError(label + " must be a non-negative exact int")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CheckpointSealError("checkpoint value is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _domain_sha256(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + b"\x00" + payload).digest()


def _descriptor_plain(descriptor: OwnerDescriptor) -> Dict[str, object]:
    return {
        "owner_id": descriptor.owner_id,
        "schema_version": descriptor.schema_version,
        "state_kind": descriptor.state_kind,
        "scope": descriptor.scope.value,
        "engine": descriptor.engine.value,
        "immutable_identity_sha256": descriptor.immutable_identity_sha256,
        "dependencies": list(descriptor.dependencies),
        "restore_rank": descriptor.restore_rank,
    }


def descriptor_sha256(descriptor: OwnerDescriptor) -> str:
    _validate_descriptor(descriptor)
    return canonical_sha256(_descriptor_plain(descriptor))


def _validate_descriptor(descriptor: object) -> OwnerDescriptor:
    if type(descriptor) is not OwnerDescriptor:
        raise OwnerRegistryError("owner descriptor must be exact OwnerDescriptor")
    if (
        type(descriptor.owner_id) is not str
        or _OWNER_ID_RE.fullmatch(descriptor.owner_id) is None
    ):
        raise OwnerRegistryError("owner_id has an invalid typed identifier")
    if type(descriptor.schema_version) is not int or descriptor.schema_version < 1:
        raise OwnerRegistryError("owner schema_version must be a positive exact int")
    if (
        type(descriptor.state_kind) is not str
        or not descriptor.state_kind
        or descriptor.state_kind.strip() != descriptor.state_kind
    ):
        raise OwnerRegistryError("owner state_kind must be a non-empty exact string")
    if type(descriptor.scope) is not OwnerScope:
        raise OwnerRegistryError("owner scope must be exact OwnerScope")
    if type(descriptor.engine) is not OwnerEngine:
        raise OwnerRegistryError("owner engine must be exact OwnerEngine")
    try:
        _require_sha256(
            descriptor.immutable_identity_sha256,
            label=descriptor.owner_id + ".immutable_identity_sha256",
        )
    except CheckpointSealError as exc:
        raise OwnerRegistryError(str(exc)) from exc
    if type(descriptor.dependencies) is not tuple or any(
        type(value) is not str for value in descriptor.dependencies
    ):
        raise OwnerRegistryError("owner dependencies must be an exact string tuple")
    if len(set(descriptor.dependencies)) != len(descriptor.dependencies):
        raise OwnerRegistryError("owner dependencies contain a duplicate")
    if descriptor.owner_id in descriptor.dependencies:
        raise OwnerRegistryError("owner cannot depend on itself")
    if type(descriptor.restore_rank) is not int or descriptor.restore_rank < 0:
        raise OwnerRegistryError("owner restore_rank must be a non-negative exact int")
    return descriptor


class OrderedOwnerRegistry:
    """Strict ordered registry; order is also the deterministic commit order."""

    __slots__ = ("_descriptors", "_by_id", "_content_sha256")

    def __init__(self, descriptors: Tuple[OwnerDescriptor, ...]) -> None:
        if type(descriptors) is not tuple or not descriptors:
            raise OwnerRegistryError("owner registry must be a non-empty exact tuple")
        checked = tuple(_validate_descriptor(value) for value in descriptors)
        owner_ids = tuple(value.owner_id for value in checked)
        if len(set(owner_ids)) != len(owner_ids):
            raise OwnerRegistryError("owner registry contains a duplicate owner_id")
        by_id = dict(zip(owner_ids, checked))
        missing_dependencies = sorted(
            {
                dependency
                for value in checked
                for dependency in value.dependencies
                if dependency not in by_id
            }
        )
        if missing_dependencies:
            raise OwnerRegistryError(
                "owner registry has missing dependencies: %s" % missing_dependencies
            )
        self._reject_dependency_cycle(checked, by_id)
        positions = {owner_id: index for index, owner_id in enumerate(owner_ids)}
        previous_rank = -1
        for index, descriptor in enumerate(checked):
            if descriptor.restore_rank < previous_rank:
                raise OwnerRegistryError("owner restore_rank order is not monotone")
            previous_rank = descriptor.restore_rank
            for dependency in descriptor.dependencies:
                if positions[dependency] >= index:
                    raise OwnerRegistryError(
                        "owner registry order is not topological for %s" % descriptor.owner_id
                    )
                if by_id[dependency].restore_rank > descriptor.restore_rank:
                    raise OwnerRegistryError(
                        "dependency restore_rank exceeds dependent restore_rank"
                    )
        self._descriptors = checked
        self._by_id = by_id
        self._content_sha256 = canonical_sha256(
            [_descriptor_plain(value) for value in checked]
        )

    @staticmethod
    def _reject_dependency_cycle(
        descriptors: Tuple[OwnerDescriptor, ...],
        by_id: Mapping[str, OwnerDescriptor],
    ) -> None:
        visiting = set()
        visited = set()

        def visit(owner_id: str) -> None:
            if owner_id in visited:
                return
            if owner_id in visiting:
                raise OwnerRegistryError("owner dependency cycle detected")
            visiting.add(owner_id)
            for dependency in by_id[owner_id].dependencies:
                visit(dependency)
            visiting.remove(owner_id)
            visited.add(owner_id)

        for descriptor in descriptors:
            visit(descriptor.owner_id)

    @property
    def descriptors(self) -> Tuple[OwnerDescriptor, ...]:
        return self._descriptors

    @property
    def owner_ids(self) -> Tuple[str, ...]:
        return tuple(value.owner_id for value in self._descriptors)

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    def descriptor(self, owner_id: str) -> OwnerDescriptor:
        try:
            return self._by_id[owner_id]
        except KeyError as exc:
            raise OwnerRegistryError("unknown owner_id %r" % owner_id) from exc

    def assert_exact(self, other: "OrderedOwnerRegistry") -> None:
        if type(other) is not OrderedOwnerRegistry:
            raise OwnerRegistryError("saved owner registry has the wrong type")
        expected = self.owner_ids
        actual = other.owner_ids
        missing = tuple(value for value in expected if value not in actual)
        extra = tuple(value for value in actual if value not in expected)
        if missing or extra:
            raise OwnerRegistryError(
                "owner registry differs: missing=%s extra=%s" % (missing, extra)
            )
        if actual != expected:
            raise OwnerRegistryError("owner registry order differs")
        for expected_descriptor, actual_descriptor in zip(
            self.descriptors, other.descriptors
        ):
            if actual_descriptor.schema_version != expected_descriptor.schema_version:
                raise OwnerRegistryError(
                    "owner schema differs for %s" % expected_descriptor.owner_id
                )
            if actual_descriptor != expected_descriptor:
                raise OwnerRegistryError(
                    "owner descriptor differs for %s" % expected_descriptor.owner_id
                )


R10_OWNER_ORDER = (
    "authority.root",
    "env.world_reset",
    "env.plant",
    "env.task_authority",
    "env.scheduler_motion",
    "env.reveal_r05",
    "env.ball_physical",
    "env.outcome_r06",
    "env.action_history",
    "env.observation_history_noise",
    "env.strike_r03",
    "env.recovery_r07",
    "env.reward_termination_curriculum",
    "trainer.policy",
    "trainer.optimizer_schedule",
    "trainer.normalizer.actor",
    "trainer.normalizer.critic",
    "trainer.rollout_frontier",
    "telemetry.highwater",
    "rng.process",
    "rng.per_world",
)

R10_OWNER_DEPENDENCIES: Mapping[str, Tuple[str, ...]] = {
    "authority.root": (),
    "env.world_reset": ("authority.root",),
    "env.plant": ("authority.root", "env.world_reset"),
    "env.task_authority": ("authority.root", "env.world_reset"),
    "env.scheduler_motion": ("env.world_reset", "env.task_authority"),
    "env.reveal_r05": (
        "env.world_reset",
        "env.task_authority",
        "env.scheduler_motion",
    ),
    "env.ball_physical": ("env.world_reset", "env.plant", "env.task_authority"),
    "env.outcome_r06": (
        "env.world_reset",
        "env.reveal_r05",
        "env.ball_physical",
    ),
    "env.action_history": ("env.world_reset", "env.plant"),
    "env.observation_history_noise": (
        "env.world_reset",
        "env.action_history",
        "env.task_authority",
        "env.scheduler_motion",
        "env.ball_physical",
        "env.outcome_r06",
    ),
    "env.strike_r03": ("env.world_reset", "env.task_authority", "env.ball_physical"),
    "env.recovery_r07": (
        "env.world_reset",
        "env.task_authority",
        "env.scheduler_motion",
    ),
    "env.reward_termination_curriculum": (
        "env.strike_r03",
        "env.outcome_r06",
        "env.recovery_r07",
    ),
    "trainer.policy": ("authority.root",),
    "trainer.optimizer_schedule": ("trainer.policy",),
    "trainer.normalizer.actor": ("authority.root",),
    "trainer.normalizer.critic": ("authority.root",),
    "trainer.rollout_frontier": (
        "env.observation_history_noise",
        "env.reward_termination_curriculum",
        "trainer.policy",
        "trainer.normalizer.actor",
        "trainer.normalizer.critic",
    ),
    "telemetry.highwater": (
        "env.reveal_r05",
        "env.strike_r03",
        "env.outcome_r06",
        "env.recovery_r07",
        "trainer.rollout_frontier",
    ),
    "rng.process": ("authority.root",),
    "rng.per_world": ("rng.process",),
}

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


def validate_r10_owner_registry(
    registry: OrderedOwnerRegistry, *, engine: OwnerEngine
) -> None:
    """Require the complete R10 surface without fabricating engine state."""

    if type(registry) is not OrderedOwnerRegistry:
        raise OwnerRegistryError("R10 registry must be OrderedOwnerRegistry")
    if type(engine) is not OwnerEngine or engine is OwnerEngine.PORTABLE:
        raise OwnerRegistryError("R10 checkpoint engine must be ISAAC or MUJOCO")
    if registry.owner_ids != R10_OWNER_ORDER:
        missing = tuple(value for value in R10_OWNER_ORDER if value not in registry.owner_ids)
        extra = tuple(value for value in registry.owner_ids if value not in R10_OWNER_ORDER)
        if missing or extra:
            raise OwnerRegistryError(
                "R10 owner surface differs: missing=%s extra=%s" % (missing, extra)
            )
        raise OwnerRegistryError("R10 owner order differs")
    for index, descriptor in enumerate(registry.descriptors):
        owner_id = descriptor.owner_id
        if descriptor.schema_version != 1:
            raise OwnerRegistryError("R10 owner schema differs for %s" % owner_id)
        expected_kind = "action_ball.r10.%s.v1" % owner_id
        if descriptor.state_kind != expected_kind:
            raise OwnerRegistryError("R10 owner state_kind differs for %s" % owner_id)
        expected_scope = OwnerScope.RUN if owner_id in _RUN_OWNERS else OwnerScope.WORLD
        if descriptor.scope is not expected_scope:
            raise OwnerRegistryError("R10 owner scope differs for %s" % owner_id)
        expected_engine = engine if owner_id == "env.plant" else OwnerEngine.PORTABLE
        if descriptor.engine is not expected_engine:
            raise OwnerRegistryError("R10 owner engine differs for %s" % owner_id)
        if descriptor.dependencies != R10_OWNER_DEPENDENCIES[owner_id]:
            raise OwnerRegistryError("R10 owner dependencies differ for %s" % owner_id)
        if descriptor.restore_rank != index:
            raise OwnerRegistryError("R10 owner restore_rank differs for %s" % owner_id)


@dataclass(frozen=True)
class WorldCheckpointPhase:
    """Per-world durable phase receipt; it is not the owner state itself."""

    world_id: int
    reset_generation: int
    episode_uid_sha256: str
    episode_step: int
    task_birth_snapshot_id: int
    reset_phase: ResetPhase
    physics_substep_phase: int
    physics_in_flight: bool
    r05_phase: R05Phase
    r05_operation_active: bool
    r05_prepared_sealed: bool
    r05_cross_owner_commit_complete: bool
    r03_phase: R03Phase
    r03_all_consumers_paid: bool
    r03_view_mask: int
    r03_payment_mask: int
    r06_flight_phase: R06FlightPhase
    r06_mailbox_phase: R06MailboxPhase
    r06_payment_epoch_open: bool
    r06_view_mask: int
    r06_payment_mask: int
    r07_payment_epoch_open: bool
    r07_deadline_ack_pending: bool


@dataclass(frozen=True)
class CheckpointBoundary:
    """The single R10-v1 save frontier supplied by the learn loop."""

    boundary_id_sha256: str
    update_index: int
    ppo_phase: PPOBoundaryPhase
    environment_step_phase: EnvironmentStepPhase
    rollout_storage_empty: bool
    actor_frontier_sealed: bool
    critic_frontier_sealed: bool
    recurrent_frontier: RecurrentFrontierStatus
    gae_in_flight: bool
    optimizer_in_flight: bool
    reset_in_flight: bool
    worlds: Tuple[WorldCheckpointPhase, ...]


def _world_phase_plain(value: WorldCheckpointPhase) -> Dict[str, object]:
    return {
        "world_id": value.world_id,
        "reset_generation": value.reset_generation,
        "episode_uid_sha256": value.episode_uid_sha256,
        "episode_step": value.episode_step,
        "task_birth_snapshot_id": value.task_birth_snapshot_id,
        "reset_phase": value.reset_phase.value,
        "physics_substep_phase": value.physics_substep_phase,
        "physics_in_flight": value.physics_in_flight,
        "r05_phase": value.r05_phase.value,
        "r05_operation_active": value.r05_operation_active,
        "r05_prepared_sealed": value.r05_prepared_sealed,
        "r05_cross_owner_commit_complete": value.r05_cross_owner_commit_complete,
        "r03_phase": value.r03_phase.value,
        "r03_all_consumers_paid": value.r03_all_consumers_paid,
        "r03_view_mask": value.r03_view_mask,
        "r03_payment_mask": value.r03_payment_mask,
        "r06_flight_phase": value.r06_flight_phase.value,
        "r06_mailbox_phase": value.r06_mailbox_phase.value,
        "r06_payment_epoch_open": value.r06_payment_epoch_open,
        "r06_view_mask": value.r06_view_mask,
        "r06_payment_mask": value.r06_payment_mask,
        "r07_payment_epoch_open": value.r07_payment_epoch_open,
        "r07_deadline_ack_pending": value.r07_deadline_ack_pending,
    }


def _boundary_plain(value: CheckpointBoundary) -> Dict[str, object]:
    return {
        "boundary_id_sha256": value.boundary_id_sha256,
        "update_index": value.update_index,
        "ppo_phase": value.ppo_phase.value,
        "environment_step_phase": value.environment_step_phase.value,
        "rollout_storage_empty": value.rollout_storage_empty,
        "actor_frontier_sealed": value.actor_frontier_sealed,
        "critic_frontier_sealed": value.critic_frontier_sealed,
        "recurrent_frontier": value.recurrent_frontier.value,
        "gae_in_flight": value.gae_in_flight,
        "optimizer_in_flight": value.optimizer_in_flight,
        "reset_in_flight": value.reset_in_flight,
        "worlds": [_world_phase_plain(world) for world in value.worlds],
    }


def boundary_sha256(value: CheckpointBoundary) -> str:
    validate_checkpoint_boundary(value)
    return canonical_sha256(_boundary_plain(value))


def validate_checkpoint_boundary(value: object) -> CheckpointBoundary:
    if type(value) is not CheckpointBoundary:
        raise CheckpointBoundaryError("boundary must be exact CheckpointBoundary")
    try:
        _require_sha256(value.boundary_id_sha256, label="boundary_id_sha256")
    except CheckpointSealError as exc:
        raise CheckpointBoundaryError(str(exc)) from exc
    if type(value.update_index) is not int or value.update_index < 0:
        raise CheckpointBoundaryError("update_index must be a non-negative exact int")
    if value.ppo_phase is not PPOBoundaryPhase.POST_UPDATE_ROLLOUT_EMPTY:
        raise CheckpointBoundaryError("checkpoint requires post-update rollout-empty PPO phase")
    if value.environment_step_phase is not EnvironmentStepPhase.BETWEEN_COMPLETE_STEPS:
        raise CheckpointBoundaryError("checkpoint requires a complete env-step boundary")
    if value.rollout_storage_empty is not True:
        raise CheckpointBoundaryError("rollout storage must be empty")
    if value.actor_frontier_sealed is not True or value.critic_frontier_sealed is not True:
        raise CheckpointBoundaryError("final normalized actor/critic frontiers must be sealed")
    if type(value.recurrent_frontier) is not RecurrentFrontierStatus:
        raise CheckpointBoundaryError(
            "recurrent frontier must be exact RecurrentFrontierStatus"
        )
    if value.recurrent_frontier not in (
        RecurrentFrontierStatus.SEALED,
        RecurrentFrontierStatus.NOT_APPLICABLE,
    ):
        raise CheckpointBoundaryError("recurrent frontier is partial")
    if value.gae_in_flight is not False:
        raise CheckpointBoundaryError("GAE/returns state is in flight")
    if value.optimizer_in_flight is not False:
        raise CheckpointBoundaryError("optimizer/minibatch state is in flight")
    if value.reset_in_flight is not False:
        raise CheckpointBoundaryError("reset transaction is in flight")
    if type(value.worlds) is not tuple or not value.worlds:
        raise CheckpointBoundaryError("boundary worlds must be a non-empty exact tuple")
    world_ids = []
    episode_uids = []
    for world in value.worlds:
        _validate_world_phase(world)
        world_ids.append(world.world_id)
        episode_uids.append(world.episode_uid_sha256)
    if len(set(world_ids)) != len(world_ids):
        raise CheckpointBoundaryError("boundary contains a duplicate world_id")
    if tuple(sorted(world_ids)) != tuple(world_ids):
        raise CheckpointBoundaryError("boundary world_id order must be strictly increasing")
    if len(set(episode_uids)) != len(episode_uids):
        raise CheckpointBoundaryError("boundary contains an aliased episode UID")
    return value


def _validate_world_phase(value: object) -> WorldCheckpointPhase:
    if type(value) is not WorldCheckpointPhase:
        raise CheckpointBoundaryError("world phase must be exact WorldCheckpointPhase")
    if type(value.world_id) is not int or value.world_id < 0:
        raise CheckpointBoundaryError("world_id must be a non-negative exact int")
    if type(value.reset_generation) is not int or value.reset_generation < 1:
        raise CheckpointBoundaryError("reset_generation must be a positive exact int")
    try:
        _require_sha256(value.episode_uid_sha256, label="episode_uid_sha256")
    except CheckpointSealError as exc:
        raise CheckpointBoundaryError(str(exc)) from exc
    if type(value.episode_step) is not int or value.episode_step < 0:
        raise CheckpointBoundaryError("episode_step must be a non-negative exact int")
    if type(value.task_birth_snapshot_id) is not int or value.task_birth_snapshot_id < 0:
        raise CheckpointBoundaryError(
            "task_birth_snapshot_id must be a non-negative exact int"
        )
    if type(value.physics_substep_phase) is not int or value.physics_substep_phase < 0:
        raise CheckpointBoundaryError(
            "physics_substep_phase must be a non-negative exact int"
        )
    enum_fields = (
        (value.reset_phase, ResetPhase, "reset_phase"),
        (value.r05_phase, R05Phase, "r05_phase"),
        (value.r03_phase, R03Phase, "r03_phase"),
        (value.r06_flight_phase, R06FlightPhase, "r06_flight_phase"),
        (value.r06_mailbox_phase, R06MailboxPhase, "r06_mailbox_phase"),
    )
    for current, expected_type, label in enum_fields:
        if type(current) is not expected_type:
            raise CheckpointBoundaryError(label + " has the wrong enum type")
    bool_fields = (
        "physics_in_flight",
        "r05_operation_active",
        "r05_prepared_sealed",
        "r05_cross_owner_commit_complete",
        "r03_all_consumers_paid",
        "r06_payment_epoch_open",
        "r07_payment_epoch_open",
        "r07_deadline_ack_pending",
    )
    if any(type(getattr(value, name)) is not bool for name in bool_fields):
        raise CheckpointBoundaryError("world phase bool fields must be exact bools")
    if value.reset_phase is not ResetPhase.COMMITTED:
        raise CheckpointBoundaryError("partial or tail-pending reset is forbidden")
    if value.physics_substep_phase != 0 or value.physics_in_flight:
        raise CheckpointBoundaryError("physics is in flight")
    if value.r05_operation_active:
        raise CheckpointBoundaryError("R05 owner operation is active")
    if value.r05_phase is R05Phase.PARTIAL:
        raise CheckpointBoundaryError("R05 cross-owner partial state is forbidden")
    if value.r05_phase is R05Phase.PREPARED:
        if not value.r05_prepared_sealed:
            raise CheckpointBoundaryError("R05 PREPARED batch is not sealed")
        if value.r05_cross_owner_commit_complete:
            raise CheckpointBoundaryError("R05 PREPARED cannot claim committed visibility")
    elif value.r05_phase is R05Phase.COMMITTED:
        if value.r05_prepared_sealed:
            raise CheckpointBoundaryError(
                "R05 COMMITTED cannot retain a private prepared row"
            )
        if not value.r05_cross_owner_commit_complete:
            raise CheckpointBoundaryError("R05 COMMITTED lacks the atomic owner join")
    else:
        if value.r05_prepared_sealed:
            raise CheckpointBoundaryError("R05 EMPTY cannot retain a prepared batch")
        if not value.r05_cross_owner_commit_complete:
            raise CheckpointBoundaryError("R05 EMPTY owner boundary is incomplete")
    if value.r03_phase in (R03Phase.ARMED, R03Phase.PENDING):
        raise CheckpointBoundaryError("R03 ARMED/PENDING is a partial env.step")
    for name in ("r03_view_mask", "r03_payment_mask"):
        current = getattr(value, name)
        if type(current) is not int or current < 0 or current > R03_FULL_CONSUMER_MASK:
            raise CheckpointBoundaryError(name + " is outside the ordered-ten mask")
    if value.r03_phase is R03Phase.PAID:
        if (
            not value.r03_all_consumers_paid
            or value.r03_view_mask != R03_FULL_CONSUMER_MASK
            or value.r03_payment_mask != R03_FULL_CONSUMER_MASK
        ):
            raise CheckpointBoundaryError("R03 PAID lacks all ten consumer payments")
    if value.r03_phase is R03Phase.IDLE and (
        value.r03_all_consumers_paid
        or value.r03_view_mask != 0
        or value.r03_payment_mask != 0
    ):
        raise CheckpointBoundaryError("R03 IDLE cannot retain consumer masks")
    if value.r06_payment_epoch_open:
        raise CheckpointBoundaryError("R06 payment/view epoch is open")
    for name in ("r06_view_mask", "r06_payment_mask"):
        current = getattr(value, name)
        if type(current) is not int or current < 0 or current > R06_FULL_CONSUMER_MASK:
            raise CheckpointBoundaryError(name + " is outside the dual-consumer mask")
    if value.r06_view_mask != value.r06_payment_mask:
        raise CheckpointBoundaryError("closed R06 epoch has an unpaid viewed consumer")
    if value.r06_mailbox_phase in (
        R06MailboxPhase.EMPTY,
        R06MailboxPhase.SETTLED_UNPAID,
    ):
        if value.r06_payment_mask != 0:
            raise CheckpointBoundaryError("unpaid R06 mailbox retains a payment mask")
    elif value.r06_mailbox_phase is R06MailboxPhase.PARTIALLY_PAID:
        if value.r06_payment_mask not in (1, 2):
            raise CheckpointBoundaryError(
                "R06 PARTIALLY_PAID must contain exactly one durable consumer"
            )
    elif value.r06_payment_mask != R06_FULL_CONSUMER_MASK:
        raise CheckpointBoundaryError("R06 PAID lacks both durable consumers")
    if value.r07_payment_epoch_open:
        raise CheckpointBoundaryError("R07 payment epoch is open")
    if value.r07_deadline_ack_pending:
        raise CheckpointBoundaryError("R07 deadline acknowledgement is partial")
    # PARTIALLY_PAID is intentionally durable when the epoch above is closed.
    return value


@dataclass(frozen=True)
class ImmutableCheckpointPins:
    code_sha256: str
    config_sha256: str
    contract_sha256: str


def _validate_immutable_pins(value: object) -> ImmutableCheckpointPins:
    if type(value) is not ImmutableCheckpointPins:
        raise CheckpointSealError("immutable pins must be exact ImmutableCheckpointPins")
    _require_sha256(value.code_sha256, label="code_sha256")
    _require_sha256(value.config_sha256, label="config_sha256")
    _require_sha256(value.contract_sha256, label="contract_sha256")
    return value


def _immutable_pins_plain(value: ImmutableCheckpointPins) -> Dict[str, str]:
    return {
        "code_sha256": value.code_sha256,
        "config_sha256": value.config_sha256,
        "contract_sha256": value.contract_sha256,
    }


@dataclass(frozen=True)
class ExternalCheckpointPins:
    """Out-of-band authority; never synthesize this from bytes being loaded."""

    checkpoint_bytes_sha256: str
    checkpoint_size_bytes: int
    owner_root_sha256: str
    registry_sha256: str
    code_sha256: str
    config_sha256: str
    contract_sha256: str
    engine: OwnerEngine
    family: PolicyFamily


def _validate_external_pins(value: object) -> ExternalCheckpointPins:
    if type(value) is not ExternalCheckpointPins:
        raise ExternalCheckpointPinError(
            "loader requires exact externally supplied checkpoint pins"
        )
    for name in (
        "checkpoint_bytes_sha256",
        "owner_root_sha256",
        "registry_sha256",
        "code_sha256",
        "config_sha256",
        "contract_sha256",
    ):
        try:
            _require_sha256(getattr(value, name), label=name)
        except CheckpointSealError as exc:
            raise ExternalCheckpointPinError(str(exc)) from exc
    if type(value.checkpoint_size_bytes) is not int or value.checkpoint_size_bytes <= 0:
        raise ExternalCheckpointPinError("checkpoint_size_bytes must be a positive exact int")
    if type(value.engine) is not OwnerEngine or value.engine is OwnerEngine.PORTABLE:
        raise ExternalCheckpointPinError("external engine pin must be ISAAC or MUJOCO")
    if type(value.family) is not PolicyFamily:
        raise ExternalCheckpointPinError("external family pin has the wrong type")
    return value


@dataclass(frozen=True)
class EqualityJoinSpec:
    join_id: str
    owner_ids: Tuple[str, ...]


@dataclass(frozen=True)
class OwnerJoinClaim:
    join_id: str
    value_sha256: str


R10_GLOBAL_JOIN_SPECS = (
    EqualityJoinSpec(
        "per_world_reset_identity",
        (
            "env.world_reset",
            "env.plant",
            "env.task_authority",
            "env.scheduler_motion",
            "env.reveal_r05",
            "env.ball_physical",
            "env.outcome_r06",
            "env.action_history",
            "env.observation_history_noise",
            "env.strike_r03",
            "env.recovery_r07",
            "env.reward_termination_curriculum",
            "trainer.rollout_frontier",
            "telemetry.highwater",
            "rng.per_world",
        ),
    ),
    EqualityJoinSpec(
        "task_scheduler_current",
        ("env.task_authority", "env.scheduler_motion"),
    ),
    EqualityJoinSpec(
        "scheduler_r05_reveal",
        ("env.scheduler_motion", "env.reveal_r05"),
    ),
    EqualityJoinSpec(
        "r05_r06_reservation",
        ("env.reveal_r05", "env.outcome_r06"),
    ),
    EqualityJoinSpec(
        "task_ball_r06_current",
        (
            "env.task_authority",
            "env.ball_physical",
            "env.outcome_r06",
        ),
    ),
    EqualityJoinSpec(
        "action_observation_frontier",
        (
            "env.action_history",
            "env.observation_history_noise",
            "trainer.rollout_frontier",
        ),
    ),
    EqualityJoinSpec(
        "reward_event_frontier",
        (
            "env.outcome_r06",
            "env.strike_r03",
            "env.recovery_r07",
            "env.reward_termination_curriculum",
            "telemetry.highwater",
        ),
    ),
    EqualityJoinSpec(
        "trainer_update_frontier",
        (
            "trainer.policy",
            "trainer.optimizer_schedule",
            "trainer.normalizer.actor",
            "trainer.normalizer.critic",
            "trainer.rollout_frontier",
            "telemetry.highwater",
        ),
    ),
    EqualityJoinSpec(
        "rng_frontier",
        ("trainer.rollout_frontier", "telemetry.highwater", "rng.process", "rng.per_world"),
    ),
    # Equality of two envelope fields is necessary but not sufficient here.
    # ``CheckpointPublicationFinalizationAuthority`` additionally requires the
    # two values to be revalidated by source-pinned writers from distinct
    # causal domains.  In particular, an environment adapter may not copy its
    # physical-owner value into ``trainer.rollout_frontier`` and call that an
    # independent observation.
    EqualityJoinSpec(
        "ppo_drain_frontier",
        ("env.ball_physical", "trainer.rollout_frontier"),
    ),
)


def _validate_join_spec(value: object, registry: OrderedOwnerRegistry) -> EqualityJoinSpec:
    if type(value) is not EqualityJoinSpec:
        raise CheckpointJoinError("join spec must be exact EqualityJoinSpec")
    if type(value.join_id) is not str or _JOIN_ID_RE.fullmatch(value.join_id) is None:
        raise CheckpointJoinError("join_id has an invalid typed identifier")
    if type(value.owner_ids) is not tuple or len(value.owner_ids) < 2:
        raise CheckpointJoinError("join owner_ids must contain at least two owners")
    if any(type(owner_id) is not str for owner_id in value.owner_ids):
        raise CheckpointJoinError("join owner_ids must be exact strings")
    if len(set(value.owner_ids)) != len(value.owner_ids):
        raise CheckpointJoinError("join contains a duplicate owner")
    unknown = tuple(owner_id for owner_id in value.owner_ids if owner_id not in registry.owner_ids)
    if unknown:
        raise CheckpointJoinError("join references unknown owners: %s" % (unknown,))
    positions = {owner_id: index for index, owner_id in enumerate(registry.owner_ids)}
    if tuple(sorted(value.owner_ids, key=positions.__getitem__)) != value.owner_ids:
        raise CheckpointJoinError("join owner order differs from registry order")
    return value


@dataclass(frozen=True)
class OwnerFreezeReceipt:
    owner_id: str
    descriptor_sha256: str
    boundary_sha256: str
    mutation_version: int
    seal_nonce_sha256: str


@dataclass(frozen=True)
class OpaqueOwnerState:
    """Typed envelope; payload bytes remain opaque to the coordinator."""

    owner_id: str
    owner_schema_version: int
    state_kind: str
    descriptor_sha256: str
    boundary_sha256: str
    seal_status: SealStatus
    mutation_version: int
    live_digest_sha256: str
    payload_sha256: str
    payload: bytes
    join_claims: Tuple[OwnerJoinClaim, ...]


@dataclass(frozen=True)
class PreparedRestoreToken:
    owner_id: str
    descriptor_sha256: str
    checkpoint_owner_root_sha256: str
    opaque_token: object


class FullMDPCheckpointOwner(Protocol):
    """Adapter protocol; every state payload is owned and interpreted elsewhere."""

    descriptor: OwnerDescriptor

    def mutation_version(self) -> int:
        """Return a monotone version for all MDP-visible owner state."""

    def live_digest(self) -> str:
        """Digest the full MDP-visible live state without mutation."""

    def freeze(self, boundary: CheckpointBoundary) -> OwnerFreezeReceipt:
        """Produce a typed boundary receipt without changing live state."""

    def export_sealed(self, receipt: OwnerFreezeReceipt) -> OpaqueOwnerState:
        """Return immutable opaque bytes without consuming RNG or receipts."""

    def prepare_restore(
        self,
        envelope: OpaqueOwnerState,
        immutable_pins: ImmutableCheckpointPins,
        owner_root_sha256: str,
    ) -> PreparedRestoreToken:
        """Validate/stage one restore token without changing live state."""

    def commit_restore(self, token: PreparedRestoreToken) -> None:
        """Install the staged state after every owner has prepared."""

    def rollback_restore(self, token: PreparedRestoreToken) -> None:
        """Restore baseline or release staging; must be idempotent for one token."""

    def poison_restore(self, reason: str) -> None:
        """Permanently fail-stop this runtime after uncertain commit."""


def make_opaque_owner_state(
    *,
    descriptor: OwnerDescriptor,
    receipt: OwnerFreezeReceipt,
    live_digest_sha256: str,
    payload: bytes,
    join_claims: Tuple[OwnerJoinClaim, ...],
) -> OpaqueOwnerState:
    """Build an immutable envelope without interpreting the owner payload."""

    _validate_descriptor(descriptor)
    _validate_freeze_receipt(receipt, descriptor=descriptor)
    _require_sha256(live_digest_sha256, label="live_digest_sha256")
    if type(payload) is not bytes or not payload:
        raise CheckpointSealError("opaque owner payload must be non-empty exact bytes")
    if type(join_claims) is not tuple:
        raise CheckpointSealError("owner join claims must be an exact tuple")
    copied = memoryview(payload).tobytes()
    return OpaqueOwnerState(
        owner_id=descriptor.owner_id,
        owner_schema_version=descriptor.schema_version,
        state_kind=descriptor.state_kind,
        descriptor_sha256=descriptor_sha256(descriptor),
        boundary_sha256=receipt.boundary_sha256,
        seal_status=SealStatus.SEALED,
        mutation_version=receipt.mutation_version,
        live_digest_sha256=live_digest_sha256,
        payload_sha256=hashlib.sha256(copied).hexdigest(),
        payload=copied,
        join_claims=join_claims,
    )


def _validate_freeze_receipt(
    value: object,
    *,
    descriptor: OwnerDescriptor,
    expected_boundary_sha256: Optional[str] = None,
    expected_mutation_version: Optional[int] = None,
) -> OwnerFreezeReceipt:
    if type(value) is not OwnerFreezeReceipt:
        raise CheckpointSealError("owner freeze result must be exact OwnerFreezeReceipt")
    if value.owner_id != descriptor.owner_id:
        raise CheckpointSealError("freeze receipt owner_id differs")
    if value.descriptor_sha256 != descriptor_sha256(descriptor):
        raise CheckpointSealError("freeze receipt descriptor differs")
    _require_sha256(value.boundary_sha256, label="freeze boundary_sha256")
    _require_sha256(value.seal_nonce_sha256, label="freeze seal_nonce_sha256")
    _require_nonnegative_int(value.mutation_version, label="freeze mutation_version")
    if expected_boundary_sha256 is not None and value.boundary_sha256 != expected_boundary_sha256:
        raise CheckpointSealError("freeze receipt boundary differs")
    if (
        expected_mutation_version is not None
        and value.mutation_version != expected_mutation_version
    ):
        raise CheckpointSealError("freeze receipt mutation version differs")
    return value


def _expected_claim_ids(owner_id: str) -> Tuple[str, ...]:
    return tuple(
        spec.join_id for spec in R10_GLOBAL_JOIN_SPECS if owner_id in spec.owner_ids
    )


def _validate_owner_state(
    value: object,
    *,
    descriptor: OwnerDescriptor,
    expected_boundary_sha256: str,
    expected_mutation_version: Optional[int] = None,
) -> OpaqueOwnerState:
    if type(value) is not OpaqueOwnerState:
        raise CheckpointSealError("owner export must be exact OpaqueOwnerState")
    if value.owner_id != descriptor.owner_id:
        raise CheckpointSealError("owner export owner_id differs")
    if (
        type(value.owner_schema_version) is not int
        or value.owner_schema_version != descriptor.schema_version
    ):
        raise CheckpointSealError("owner export schema differs")
    if value.state_kind != descriptor.state_kind:
        raise CheckpointSealError("owner export state_kind differs")
    if value.descriptor_sha256 != descriptor_sha256(descriptor):
        raise CheckpointSealError("owner export descriptor differs")
    if value.boundary_sha256 != expected_boundary_sha256:
        raise CheckpointSealError("owner export boundary differs")
    if value.seal_status is not SealStatus.SEALED:
        raise CheckpointSealError("owner export is PARTIAL/unsealed")
    _require_nonnegative_int(value.mutation_version, label="owner mutation_version")
    if (
        expected_mutation_version is not None
        and value.mutation_version != expected_mutation_version
    ):
        raise CheckpointSealError("owner export mutation version differs")
    _require_sha256(value.live_digest_sha256, label="owner live_digest_sha256")
    _require_sha256(value.payload_sha256, label="owner payload_sha256")
    if type(value.payload) is not bytes or not value.payload:
        raise CheckpointSealError("owner payload must be non-empty exact bytes")
    if hashlib.sha256(value.payload).hexdigest() != value.payload_sha256:
        raise CheckpointSealError("owner payload digest differs")
    if type(value.join_claims) is not tuple:
        raise CheckpointJoinError("owner join_claims must be an exact tuple")
    claim_ids = []
    for claim in value.join_claims:
        if type(claim) is not OwnerJoinClaim:
            raise CheckpointJoinError("join claim must be exact OwnerJoinClaim")
        if type(claim.join_id) is not str or _JOIN_ID_RE.fullmatch(claim.join_id) is None:
            raise CheckpointJoinError("owner join_id has an invalid typed identifier")
        _require_sha256(claim.value_sha256, label="join value_sha256")
        claim_ids.append(claim.join_id)
    if len(set(claim_ids)) != len(claim_ids):
        raise CheckpointJoinError("owner contains a duplicate join claim")
    if tuple(claim_ids) != _expected_claim_ids(descriptor.owner_id):
        raise CheckpointJoinError(
            "owner join claim surface differs for %s" % descriptor.owner_id
        )
    return value


def _owner_state_plain(value: OpaqueOwnerState) -> Dict[str, object]:
    return {
        "owner_id": value.owner_id,
        "owner_schema_version": value.owner_schema_version,
        "state_kind": value.state_kind,
        "descriptor_sha256": value.descriptor_sha256,
        "boundary_sha256": value.boundary_sha256,
        "seal_status": value.seal_status.value,
        "mutation_version": value.mutation_version,
        "live_digest_sha256": value.live_digest_sha256,
        "payload_sha256": value.payload_sha256,
        "payload_base64": base64.b64encode(value.payload).decode("ascii"),
        "join_claims": [
            {"join_id": claim.join_id, "value_sha256": claim.value_sha256}
            for claim in value.join_claims
        ],
    }


def owner_merkle_root(states: Tuple[OpaqueOwnerState, ...]) -> str:
    """Return a domain-separated binary Merkle root in registry order."""

    if type(states) is not tuple or not states:
        raise CheckpointSealError("owner Merkle tree requires a non-empty exact tuple")
    leaves = [
        _domain_sha256(
            b"action-ball-r10-owner-leaf-v1",
            _canonical_json_bytes(_owner_state_plain(value)),
        )
        for value in states
    ]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            _domain_sha256(
                b"action-ball-r10-owner-node-v1", leaves[index] + leaves[index + 1]
            )
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def validate_global_joins(
    states: Tuple[OpaqueOwnerState, ...], registry: OrderedOwnerRegistry
) -> None:
    if type(states) is not tuple or len(states) != len(registry.descriptors):
        raise CheckpointJoinError("global join owner cardinality differs")
    by_owner = {state.owner_id: state for state in states}
    if tuple(by_owner) != registry.owner_ids:
        raise CheckpointJoinError("global join owner order differs")
    for spec in R10_GLOBAL_JOIN_SPECS:
        _validate_join_spec(spec, registry)
        values = []
        for owner_id in spec.owner_ids:
            claims = {
                claim.join_id: claim.value_sha256
                for claim in by_owner[owner_id].join_claims
            }
            if spec.join_id not in claims:
                raise CheckpointJoinError(
                    "owner %s lacks join %s" % (owner_id, spec.join_id)
                )
            values.append(claims[spec.join_id])
        if len(set(values)) != 1:
            raise CheckpointJoinError("global join %s differs" % spec.join_id)


def _owner_version(owner: FullMDPCheckpointOwner) -> int:
    value = owner.mutation_version()
    if type(value) is not int or value < 0:
        raise CheckpointSealError("owner mutation_version() must return a non-negative int")
    return value


def _owner_live_digest(owner: FullMDPCheckpointOwner) -> str:
    value = owner.live_digest()
    return _require_sha256(value, label="owner live_digest()")


def _observe_owners(
    owners: Tuple[FullMDPCheckpointOwner, ...]
) -> Tuple[Tuple[int, str], ...]:
    """Globally sandwich digest reads so an accessor cannot mutate any peer."""

    versions_before = tuple(_owner_version(owner) for owner in owners)
    digests_first = tuple(_owner_live_digest(owner) for owner in owners)
    versions_middle = tuple(_owner_version(owner) for owner in owners)
    digests_second = tuple(_owner_live_digest(owner) for owner in owners)
    versions_after = tuple(_owner_version(owner) for owner in owners)
    if (
        versions_before != versions_middle
        or versions_middle != versions_after
        or digests_first != digests_second
    ):
        raise CheckpointSealError(
            "owner observation accessor mutated state or returned an unstable digest"
        )
    return tuple(zip(versions_after, digests_second))


def _bind_owners(
    registry: OrderedOwnerRegistry,
    owners: Tuple[FullMDPCheckpointOwner, ...],
) -> Tuple[FullMDPCheckpointOwner, ...]:
    if type(owners) is not tuple:
        raise OwnerRegistryError("owners must be an exact tuple in registry order")
    if len(owners) != len(registry.descriptors):
        raise OwnerRegistryError("owner adapter cardinality differs from registry")
    if len({id(owner) for owner in owners}) != len(owners):
        raise OwnerRegistryError("owner adapters alias one live object")
    for descriptor, owner in zip(registry.descriptors, owners):
        actual = getattr(owner, "descriptor", None)
        if type(actual) is not OwnerDescriptor or actual != descriptor:
            raise OwnerRegistryError(
                "owner adapter descriptor differs for %s" % descriptor.owner_id
            )
    return owners


def _assert_observational_baseline(
    owners: Tuple[FullMDPCheckpointOwner, ...],
    observations: Tuple[Tuple[int, str], ...],
    *,
    operation: str,
) -> None:
    current = _observe_owners(owners)
    if current != observations:
        raise CheckpointSealError(operation + " mutated an owner during observational phase")


@dataclass(frozen=True)
class CheckpointPublication:
    schema_version: int
    kind: str
    integration_status: str
    runtime_wiring: bool
    continuation_authorized: bool
    blob: bytes
    external_pins: ExternalCheckpointPins


@dataclass(frozen=True)
class VerifiedCheckpoint:
    engine: OwnerEngine
    family: PolicyFamily
    immutable_pins: ImmutableCheckpointPins
    registry: OrderedOwnerRegistry
    boundary: CheckpointBoundary
    owner_states: Tuple[OpaqueOwnerState, ...]
    owner_root_sha256: str
    content_sha256: str
    checkpoint_bytes_sha256: str
    checkpoint_size_bytes: int


def _archive_body_plain(
    *,
    engine: OwnerEngine,
    family: PolicyFamily,
    immutable_pins: ImmutableCheckpointPins,
    registry: OrderedOwnerRegistry,
    boundary: CheckpointBoundary,
    states: Tuple[OpaqueOwnerState, ...],
    root_sha256: str,
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "integration_status": INTEGRATION_STATUS,
        "runtime_wiring": False,
        "engine_continuation_proven": False,
        "formal_exact_resume_integrated": False,
        "launch_authorized": False,
        "continuation_claim": CONTINUATION_CLAIM,
        "seal_status": SealStatus.SEALED.value,
        "engine": engine.value,
        "family": family.value,
        "immutable_pins": _immutable_pins_plain(immutable_pins),
        "registry_sha256": registry.content_sha256,
        "registry": [_descriptor_plain(value) for value in registry.descriptors],
        "boundary": _boundary_plain(boundary),
        "global_join_specs": [
            {"join_id": spec.join_id, "owner_ids": list(spec.owner_ids)}
            for spec in R10_GLOBAL_JOIN_SPECS
        ],
        "owner_root_sha256": root_sha256,
        "owner_states": [_owner_state_plain(value) for value in states],
    }


def seal_checkpoint_candidate(
    *,
    engine: OwnerEngine,
    family: PolicyFamily,
    immutable_pins: ImmutableCheckpointPins,
    registry: OrderedOwnerRegistry,
    boundary: CheckpointBoundary,
    owners: Tuple[FullMDPCheckpointOwner, ...],
) -> CheckpointPublication:
    """Observationally snapshot opaque owners and return bytes plus external pins."""

    if type(engine) is not OwnerEngine or engine is OwnerEngine.PORTABLE:
        raise CheckpointSealError("checkpoint engine must be ISAAC or MUJOCO")
    if type(family) is not PolicyFamily:
        raise CheckpointSealError("checkpoint family must be exact PolicyFamily")
    _validate_immutable_pins(immutable_pins)
    validate_r10_owner_registry(registry, engine=engine)
    validate_checkpoint_boundary(boundary)
    bound = _bind_owners(registry, owners)
    observations = _observe_owners(bound)
    boundary_digest = boundary_sha256(boundary)
    states = []
    state_object_ids = set()
    for descriptor, owner, observation in zip(
        registry.descriptors, bound, observations
    ):
        version, live_digest = observation
        receipt = owner.freeze(boundary)
        _validate_freeze_receipt(
            receipt,
            descriptor=descriptor,
            expected_boundary_sha256=boundary_digest,
            expected_mutation_version=version,
        )
        _assert_observational_baseline(
            bound, observations, operation="owner freeze"
        )
        state = owner.export_sealed(receipt)
        _validate_owner_state(
            state,
            descriptor=descriptor,
            expected_boundary_sha256=boundary_digest,
            expected_mutation_version=version,
        )
        if state.live_digest_sha256 != live_digest:
            raise CheckpointSealError("owner export live digest differs")
        if id(state) in state_object_ids:
            raise CheckpointSealError("two owner exports alias one envelope object")
        state_object_ids.add(id(state))
        states.append(state)
        _assert_observational_baseline(
            bound, observations, operation="owner export"
        )
    state_tuple = tuple(states)
    validate_global_joins(state_tuple, registry)
    _assert_observational_baseline(
        bound, observations, operation="global join validation"
    )
    root = owner_merkle_root(state_tuple)
    body = _archive_body_plain(
        engine=engine,
        family=family,
        immutable_pins=immutable_pins,
        registry=registry,
        boundary=boundary,
        states=state_tuple,
        root_sha256=root,
    )
    archive = dict(body)
    archive["content_sha256"] = canonical_sha256(body)
    blob = _canonical_json_bytes(archive)
    external = ExternalCheckpointPins(
        checkpoint_bytes_sha256=hashlib.sha256(blob).hexdigest(),
        checkpoint_size_bytes=len(blob),
        owner_root_sha256=root,
        registry_sha256=registry.content_sha256,
        code_sha256=immutable_pins.code_sha256,
        config_sha256=immutable_pins.config_sha256,
        contract_sha256=immutable_pins.contract_sha256,
        engine=engine,
        family=family,
    )
    return CheckpointPublication(
        schema_version=SCHEMA_VERSION,
        kind=CHECKPOINT_RECEIPT_KIND,
        integration_status=INTEGRATION_STATUS,
        runtime_wiring=False,
        continuation_authorized=False,
        blob=blob,
        external_pins=external,
    )


def _exact_dict(value: object, keys: Tuple[str, ...], *, label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise CheckpointSealError(label + " must be an exact dict")
    actual = set(value)
    expected = set(keys)
    if actual != expected or any(type(key) is not str for key in value):
        raise CheckpointSealError(
            "%s keys differ: missing=%s extra=%s"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )
    return value


def _no_duplicate_json_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointSealError("checkpoint JSON contains a duplicate key")
        result[key] = value
    return result


def _parse_enum(enum_type: Any, value: object, *, label: str) -> Any:
    if type(value) is not str:
        raise CheckpointSealError(label + " must be an exact enum string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise CheckpointSealError(label + " has an unknown enum value") from exc


def _parse_descriptor(value: object) -> OwnerDescriptor:
    row = _exact_dict(
        value,
        (
            "owner_id",
            "schema_version",
            "state_kind",
            "scope",
            "engine",
            "immutable_identity_sha256",
            "dependencies",
            "restore_rank",
        ),
        label="owner descriptor",
    )
    dependencies = row["dependencies"]
    if type(dependencies) is not list or any(type(item) is not str for item in dependencies):
        raise OwnerRegistryError("saved owner dependencies must be an exact string list")
    return OwnerDescriptor(
        owner_id=row["owner_id"],
        schema_version=row["schema_version"],
        state_kind=row["state_kind"],
        scope=_parse_enum(OwnerScope, row["scope"], label="owner scope"),
        engine=_parse_enum(OwnerEngine, row["engine"], label="owner engine"),
        immutable_identity_sha256=row["immutable_identity_sha256"],
        dependencies=tuple(dependencies),
        restore_rank=row["restore_rank"],
    )


def _parse_world_phase(value: object) -> WorldCheckpointPhase:
    keys = tuple(_world_phase_plain(_example_world_phase()).keys())
    row = _exact_dict(value, keys, label="world phase")
    return WorldCheckpointPhase(
        world_id=row["world_id"],
        reset_generation=row["reset_generation"],
        episode_uid_sha256=row["episode_uid_sha256"],
        episode_step=row["episode_step"],
        task_birth_snapshot_id=row["task_birth_snapshot_id"],
        reset_phase=_parse_enum(ResetPhase, row["reset_phase"], label="reset phase"),
        physics_substep_phase=row["physics_substep_phase"],
        physics_in_flight=row["physics_in_flight"],
        r05_phase=_parse_enum(R05Phase, row["r05_phase"], label="R05 phase"),
        r05_operation_active=row["r05_operation_active"],
        r05_prepared_sealed=row["r05_prepared_sealed"],
        r05_cross_owner_commit_complete=row["r05_cross_owner_commit_complete"],
        r03_phase=_parse_enum(R03Phase, row["r03_phase"], label="R03 phase"),
        r03_all_consumers_paid=row["r03_all_consumers_paid"],
        r03_view_mask=row["r03_view_mask"],
        r03_payment_mask=row["r03_payment_mask"],
        r06_flight_phase=_parse_enum(
            R06FlightPhase, row["r06_flight_phase"], label="R06 flight phase"
        ),
        r06_mailbox_phase=_parse_enum(
            R06MailboxPhase, row["r06_mailbox_phase"], label="R06 mailbox phase"
        ),
        r06_payment_epoch_open=row["r06_payment_epoch_open"],
        r06_view_mask=row["r06_view_mask"],
        r06_payment_mask=row["r06_payment_mask"],
        r07_payment_epoch_open=row["r07_payment_epoch_open"],
        r07_deadline_ack_pending=row["r07_deadline_ack_pending"],
    )


def _example_world_phase() -> WorldCheckpointPhase:
    """Private key-order template; never used as runtime state."""

    zero = "0" * 64
    return WorldCheckpointPhase(
        world_id=0,
        reset_generation=1,
        episode_uid_sha256=zero,
        episode_step=0,
        task_birth_snapshot_id=0,
        reset_phase=ResetPhase.COMMITTED,
        physics_substep_phase=0,
        physics_in_flight=False,
        r05_phase=R05Phase.EMPTY,
        r05_operation_active=False,
        r05_prepared_sealed=False,
        r05_cross_owner_commit_complete=True,
        r03_phase=R03Phase.IDLE,
        r03_all_consumers_paid=False,
        r03_view_mask=0,
        r03_payment_mask=0,
        r06_flight_phase=R06FlightPhase.EMPTY,
        r06_mailbox_phase=R06MailboxPhase.EMPTY,
        r06_payment_epoch_open=False,
        r06_view_mask=0,
        r06_payment_mask=0,
        r07_payment_epoch_open=False,
        r07_deadline_ack_pending=False,
    )


def _parse_boundary(value: object) -> CheckpointBoundary:
    keys = tuple(
        _boundary_plain(
            CheckpointBoundary(
                boundary_id_sha256="0" * 64,
                update_index=0,
                ppo_phase=PPOBoundaryPhase.POST_UPDATE_ROLLOUT_EMPTY,
                environment_step_phase=EnvironmentStepPhase.BETWEEN_COMPLETE_STEPS,
                rollout_storage_empty=True,
                actor_frontier_sealed=True,
                critic_frontier_sealed=True,
                recurrent_frontier=RecurrentFrontierStatus.NOT_APPLICABLE,
                gae_in_flight=False,
                optimizer_in_flight=False,
                reset_in_flight=False,
                worlds=(_example_world_phase(),),
            )
        ).keys()
    )
    row = _exact_dict(value, keys, label="checkpoint boundary")
    worlds = row["worlds"]
    if type(worlds) is not list:
        raise CheckpointBoundaryError("saved boundary worlds must be an exact list")
    boundary = CheckpointBoundary(
        boundary_id_sha256=row["boundary_id_sha256"],
        update_index=row["update_index"],
        ppo_phase=_parse_enum(PPOBoundaryPhase, row["ppo_phase"], label="PPO phase"),
        environment_step_phase=_parse_enum(
            EnvironmentStepPhase,
            row["environment_step_phase"],
            label="environment step phase",
        ),
        rollout_storage_empty=row["rollout_storage_empty"],
        actor_frontier_sealed=row["actor_frontier_sealed"],
        critic_frontier_sealed=row["critic_frontier_sealed"],
        recurrent_frontier=_parse_enum(
            RecurrentFrontierStatus,
            row["recurrent_frontier"],
            label="recurrent frontier",
        ),
        gae_in_flight=row["gae_in_flight"],
        optimizer_in_flight=row["optimizer_in_flight"],
        reset_in_flight=row["reset_in_flight"],
        worlds=tuple(_parse_world_phase(world) for world in worlds),
    )
    return validate_checkpoint_boundary(boundary)


def _parse_owner_state(
    value: object,
    *,
    descriptor: OwnerDescriptor,
    expected_boundary_sha256: str,
) -> OpaqueOwnerState:
    row = _exact_dict(
        value,
        (
            "owner_id",
            "owner_schema_version",
            "state_kind",
            "descriptor_sha256",
            "boundary_sha256",
            "seal_status",
            "mutation_version",
            "live_digest_sha256",
            "payload_sha256",
            "payload_base64",
            "join_claims",
        ),
        label="owner state",
    )
    if type(row["payload_base64"]) is not str:
        raise CheckpointSealError("owner payload_base64 must be an exact string")
    try:
        payload = base64.b64decode(row["payload_base64"].encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise CheckpointSealError("owner payload_base64 is invalid") from exc
    claims = row["join_claims"]
    if type(claims) is not list:
        raise CheckpointJoinError("owner join_claims must be an exact list")
    parsed_claims = []
    for value in claims:
        claim = _exact_dict(
            value, ("join_id", "value_sha256"), label="owner join claim"
        )
        parsed_claims.append(
            OwnerJoinClaim(
                join_id=claim["join_id"], value_sha256=claim["value_sha256"]
            )
        )
    state = OpaqueOwnerState(
        owner_id=row["owner_id"],
        owner_schema_version=row["owner_schema_version"],
        state_kind=row["state_kind"],
        descriptor_sha256=row["descriptor_sha256"],
        boundary_sha256=row["boundary_sha256"],
        seal_status=_parse_enum(SealStatus, row["seal_status"], label="owner seal status"),
        mutation_version=row["mutation_version"],
        live_digest_sha256=row["live_digest_sha256"],
        payload_sha256=row["payload_sha256"],
        payload=payload,
        join_claims=tuple(parsed_claims),
    )
    return _validate_owner_state(
        state,
        descriptor=descriptor,
        expected_boundary_sha256=expected_boundary_sha256,
    )


def verify_checkpoint_candidate(
    blob: bytes,
    *,
    expected_external_pins: ExternalCheckpointPins,
    expected_registry: OrderedOwnerRegistry,
) -> VerifiedCheckpoint:
    """Verify external authority before decoding any checkpoint-owned identity."""

    pins = _validate_external_pins(expected_external_pins)
    if type(expected_registry) is not OrderedOwnerRegistry:
        raise OwnerRegistryError("expected_registry must be exact OrderedOwnerRegistry")
    validate_r10_owner_registry(expected_registry, engine=pins.engine)
    if type(blob) is not bytes or not blob:
        raise ExternalCheckpointPinError("checkpoint blob must be non-empty exact bytes")
    if len(blob) != pins.checkpoint_size_bytes:
        raise ExternalCheckpointPinError("checkpoint byte length differs from external pin")
    if hashlib.sha256(blob).hexdigest() != pins.checkpoint_bytes_sha256:
        raise ExternalCheckpointPinError("checkpoint bytes differ from external SHA-256 pin")
    try:
        raw = json.loads(blob.decode("ascii"), object_pairs_hook=_no_duplicate_json_object)
    except CheckpointSealError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointSealError("checkpoint is not strict ASCII JSON") from exc
    archive = _exact_dict(
        raw,
        (
            "schema_version",
            "kind",
            "integration_status",
            "runtime_wiring",
            "engine_continuation_proven",
            "formal_exact_resume_integrated",
            "launch_authorized",
            "continuation_claim",
            "seal_status",
            "engine",
            "family",
            "immutable_pins",
            "registry_sha256",
            "registry",
            "boundary",
            "global_join_specs",
            "owner_root_sha256",
            "owner_states",
            "content_sha256",
        ),
        label="checkpoint root",
    )
    content_sha256 = archive.pop("content_sha256")
    _require_sha256(content_sha256, label="content_sha256")
    if canonical_sha256(archive) != content_sha256:
        raise CheckpointSealError("checkpoint internal content seal differs")
    if archive["schema_version"] != SCHEMA_VERSION or type(archive["schema_version"]) is not int:
        raise CheckpointSealError("checkpoint schema_version differs")
    if archive["kind"] != CHECKPOINT_KIND:
        raise CheckpointSealError("checkpoint kind differs")
    if archive["integration_status"] != INTEGRATION_STATUS:
        raise CheckpointSealError("checkpoint integration status differs")
    false_fields = (
        "runtime_wiring",
        "engine_continuation_proven",
        "formal_exact_resume_integrated",
        "launch_authorized",
    )
    if any(archive[name] is not False for name in false_fields):
        raise CheckpointSealError("checkpoint forged runtime/exactness authority")
    if archive["continuation_claim"] != CONTINUATION_CLAIM:
        raise CheckpointSealError("checkpoint continuation claim differs")
    if archive["seal_status"] != SealStatus.SEALED.value:
        raise CheckpointSealError("checkpoint root is PARTIAL/unsealed")
    engine = _parse_enum(OwnerEngine, archive["engine"], label="checkpoint engine")
    family = _parse_enum(PolicyFamily, archive["family"], label="checkpoint family")
    if engine is OwnerEngine.PORTABLE:
        raise CheckpointSealError("checkpoint engine cannot be portable")
    if engine is not pins.engine or family is not pins.family:
        raise ExternalCheckpointPinError("checkpoint engine/family differs from external pin")
    immutable_row = _exact_dict(
        archive["immutable_pins"],
        ("code_sha256", "config_sha256", "contract_sha256"),
        label="immutable pins",
    )
    immutable = _validate_immutable_pins(
        ImmutableCheckpointPins(
            code_sha256=immutable_row["code_sha256"],
            config_sha256=immutable_row["config_sha256"],
            contract_sha256=immutable_row["contract_sha256"],
        )
    )
    if (
        immutable.code_sha256 != pins.code_sha256
        or immutable.config_sha256 != pins.config_sha256
        or immutable.contract_sha256 != pins.contract_sha256
    ):
        raise ExternalCheckpointPinError(
            "checkpoint code/config/contract differs from external pin"
        )
    registry_rows = archive["registry"]
    if type(registry_rows) is not list:
        raise OwnerRegistryError("saved registry must be an exact list")
    saved_registry = OrderedOwnerRegistry(
        tuple(_parse_descriptor(value) for value in registry_rows)
    )
    expected_registry.assert_exact(saved_registry)
    validate_r10_owner_registry(saved_registry, engine=engine)
    if archive["registry_sha256"] != saved_registry.content_sha256:
        raise CheckpointSealError("checkpoint registry seal differs")
    if saved_registry.content_sha256 != pins.registry_sha256:
        raise ExternalCheckpointPinError("checkpoint registry differs from external pin")
    join_rows = archive["global_join_specs"]
    if type(join_rows) is not list:
        raise CheckpointJoinError("global_join_specs must be an exact list")
    parsed_specs = []
    for value in join_rows:
        row = _exact_dict(value, ("join_id", "owner_ids"), label="global join spec")
        if type(row["owner_ids"]) is not list:
            raise CheckpointJoinError("global join owner_ids must be an exact list")
        parsed_specs.append(
            EqualityJoinSpec(row["join_id"], tuple(row["owner_ids"]))
        )
    if tuple(parsed_specs) != R10_GLOBAL_JOIN_SPECS:
        raise CheckpointJoinError("global join specification surface differs")
    for spec in parsed_specs:
        _validate_join_spec(spec, saved_registry)
    boundary = _parse_boundary(archive["boundary"])
    boundary_digest = boundary_sha256(boundary)
    owner_rows = archive["owner_states"]
    if type(owner_rows) is not list or len(owner_rows) != len(saved_registry.descriptors):
        raise CheckpointSealError("owner state cardinality differs from registry")
    states = tuple(
        _parse_owner_state(
            row,
            descriptor=descriptor,
            expected_boundary_sha256=boundary_digest,
        )
        for descriptor, row in zip(saved_registry.descriptors, owner_rows)
    )
    if tuple(value.owner_id for value in states) != saved_registry.owner_ids:
        raise CheckpointSealError("owner state order differs from registry")
    validate_global_joins(states, saved_registry)
    root = owner_merkle_root(states)
    _require_sha256(archive["owner_root_sha256"], label="owner_root_sha256")
    if archive["owner_root_sha256"] != root:
        raise CheckpointSealError("checkpoint owner Merkle root differs")
    if root != pins.owner_root_sha256:
        raise ExternalCheckpointPinError("checkpoint owner root differs from external pin")
    return VerifiedCheckpoint(
        engine=engine,
        family=family,
        immutable_pins=immutable,
        registry=saved_registry,
        boundary=boundary,
        owner_states=states,
        owner_root_sha256=root,
        content_sha256=content_sha256,
        checkpoint_bytes_sha256=pins.checkpoint_bytes_sha256,
        checkpoint_size_bytes=pins.checkpoint_size_bytes,
    )


@dataclass(frozen=True)
class PpoDrainRunnerWriterProjection:
    """Runner-writer result made only from retained drain-owner primitives.

    The finalizer never gives the writer the archive root or expected mutation
    versions.  This value is therefore useful only as the return value of the
    exact construction-bound runner callback; constructing an equal value at
    some other callpoint grants no authority.
    """

    schema_version: int
    kind: str
    checkpoint_frontier_sha256: str
    mutation_version_highwaters: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class PpoDrainLeafLiveMutationProjection:
    """One construction-bound leaf's live mutation version observation."""

    schema_version: int
    kind: str
    owner_kind: str
    mutation_version: int


@dataclass(frozen=True)
class _FinalizerConstructionPayload:
    authority_ref: weakref.ReferenceType["CheckpointPublicationFinalizationAuthority"]
    schema_version: int
    kind: str
    registry_sha256: str
    callback_bindings: Tuple[Tuple[str, str, str], ...]
    live_mutation_bindings: Tuple[Tuple[str, str, str], ...]
    diagnostic_allow_missing_live_highwaters: bool
    distinct_identity_count: int
    content_sha256: str


class CheckpointPublicationFinalizerConstructionBundle:
    """Public, immutable, owner-minted pin bundle for one finalizer instance."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("checkpoint finalizer construction bundles are owner-issued only")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("checkpoint finalizer construction bundles are immutable")

    def __copy__(self):
        raise TypeError("checkpoint finalizer construction bundles cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("checkpoint finalizer construction bundles cannot be copied")

    def __reduce__(self):
        raise TypeError("checkpoint finalizer construction bundles cannot be serialized")

    @staticmethod
    def _payload(
        value: "CheckpointPublicationFinalizerConstructionBundle",
    ) -> _FinalizerConstructionPayload:
        payload = _lookup_finalizer_construction_bundle(value)
        authority = None if payload is None else payload.authority_ref()
        if (
            payload is None
            or type(authority) is not CheckpointPublicationFinalizationAuthority
            or authority._construction_bundle is not value
        ):
            raise CheckpointPublicationFinalizationError(
                "construction bundle is not current and finalizer-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return self._payload(self).schema_version

    @property
    def kind(self) -> str:
        return self._payload(self).kind

    @property
    def registry_sha256(self) -> str:
        return self._payload(self).registry_sha256

    @property
    def callback_bindings(self) -> Tuple[Tuple[str, str, str], ...]:
        return self._payload(self).callback_bindings

    @property
    def live_mutation_bindings(self) -> Tuple[Tuple[str, str, str], ...]:
        return self._payload(self).live_mutation_bindings

    @property
    def diagnostic_allow_missing_live_highwaters(self) -> bool:
        return self._payload(self).diagnostic_allow_missing_live_highwaters

    @property
    def production_live_highwater_join(self) -> bool:
        return not self._payload(self).diagnostic_allow_missing_live_highwaters

    @property
    def distinct_identity_count(self) -> int:
        return self._payload(self).distinct_identity_count

    @property
    def content_sha256(self) -> str:
        return self._payload(self).content_sha256


def _make_finalizer_construction_bundle_registry():
    rows: weakref.WeakKeyDictionary[
        CheckpointPublicationFinalizerConstructionBundle,
        _FinalizerConstructionPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _FinalizerConstructionPayload,
    ) -> CheckpointPublicationFinalizerConstructionBundle:
        value = object.__new__(CheckpointPublicationFinalizerConstructionBundle)
        with lock:
            rows[value] = payload
        return value

    def lookup(
        value: CheckpointPublicationFinalizerConstructionBundle,
    ) -> Optional[_FinalizerConstructionPayload]:
        with lock:
            return rows.get(value)

    return mint, lookup


(
    _mint_finalizer_construction_bundle,
    _lookup_finalizer_construction_bundle,
) = _make_finalizer_construction_bundle_registry()
del _make_finalizer_construction_bundle_registry


@dataclass(frozen=True)
class _FinalizationReceiptPayload:
    authority_ref: weakref.ReferenceType["CheckpointPublicationFinalizationAuthority"]
    publication: CheckpointPublication
    schema_version: int
    kind: str
    checkpoint_bytes_sha256: str
    checkpoint_content_sha256: str
    owner_root_sha256: str
    registry_sha256: str
    boundary_sha256: str
    ppo_drain_frontier_sha256: str
    top_audit_claim_validated: bool
    live_highwaters_validated: bool
    runtime_wiring: bool


class CheckpointPublicationFinalizationReceipt:
    """Opaque production proof minted after every pre-ACK fact validates.

    The receipt never acknowledges or garbage-collects the top-owned audit
    claim.  ``validate`` returns the original publication so the top owner can
    perform that state transition itself; this capability is a separately
    retrievable audit artifact.  A diagnostic constructor can never mint it.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("checkpoint finalization receipts are finalizer-issued only")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("checkpoint finalization receipts are immutable")

    def __copy__(self):
        raise TypeError("checkpoint finalization receipts cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("checkpoint finalization receipts cannot be copied")

    def __reduce__(self):
        raise TypeError("checkpoint finalization receipts cannot be serialized")

    @staticmethod
    def _payload(
        value: "CheckpointPublicationFinalizationReceipt",
    ) -> _FinalizationReceiptPayload:
        payload = _lookup_finalization_receipt(value)
        authority = None if payload is None else payload.authority_ref()
        if payload is None or type(authority) is not CheckpointPublicationFinalizationAuthority:
            raise CheckpointPublicationFinalizationError(
                "finalization receipt is not finalizer-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return self._payload(self).schema_version

    @property
    def kind(self) -> str:
        return self._payload(self).kind

    @property
    def checkpoint_bytes_sha256(self) -> str:
        return self._payload(self).checkpoint_bytes_sha256

    @property
    def checkpoint_content_sha256(self) -> str:
        return self._payload(self).checkpoint_content_sha256

    @property
    def owner_root_sha256(self) -> str:
        return self._payload(self).owner_root_sha256

    @property
    def registry_sha256(self) -> str:
        return self._payload(self).registry_sha256

    @property
    def boundary_sha256(self) -> str:
        return self._payload(self).boundary_sha256

    @property
    def ppo_drain_frontier_sha256(self) -> str:
        return self._payload(self).ppo_drain_frontier_sha256

    @property
    def top_audit_claim_validated(self) -> bool:
        return self._payload(self).top_audit_claim_validated

    @property
    def live_highwaters_validated(self) -> bool:
        return self._payload(self).live_highwaters_validated

    @property
    def ready_for_top_ack(self) -> bool:
        payload = self._payload(self)
        return (
            payload.top_audit_claim_validated
            and payload.live_highwaters_validated
        )

    @property
    def runtime_wiring(self) -> bool:
        return self._payload(self).runtime_wiring


def _make_finalization_receipt_registry():
    rows: weakref.WeakKeyDictionary[
        CheckpointPublicationFinalizationReceipt,
        _FinalizationReceiptPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _FinalizationReceiptPayload,
    ) -> CheckpointPublicationFinalizationReceipt:
        value = object.__new__(CheckpointPublicationFinalizationReceipt)
        with lock:
            rows[value] = payload
        return value

    def lookup(
        value: CheckpointPublicationFinalizationReceipt,
    ) -> Optional[_FinalizationReceiptPayload]:
        with lock:
            return rows.get(value)

    return mint, lookup


_mint_finalization_receipt, _lookup_finalization_receipt = (
    _make_finalization_receipt_registry()
)
del _make_finalization_receipt_registry


@dataclass
class _PendingCheckpointPublicationFinalization:
    publication: CheckpointPublication
    verified: VerifiedCheckpoint
    durable_receipt: object
    ppo_drain_frontier_sha256: str
    live_highwaters_validated: bool
    consumed: bool = False
    finalization_receipt: Optional[CheckpointPublicationFinalizationReceipt] = None


def _require_exact_bound_callback(
    owner: object,
    callback: object,
    *,
    label: str,
) -> Callable[..., object]:
    direct = None if owner is None else vars(type(owner)).get(
        getattr(callback, "__name__", "")
    )
    if (
        owner is None
        or type(callback) is not types.MethodType
        or getattr(callback, "__self__", None) is not owner
        or type(direct) is not types.FunctionType
        or callback.__func__ is not direct
        or (
            type(getattr(owner, "__dict__", None)) is dict
            and callback.__name__ in owner.__dict__
        )
    ):
        raise CheckpointPublicationFinalizationError(
            label + " must be an exact construction-bound method"
        )
    return callback


def _callback_sha256(value: Callable[..., object]) -> str:
    function = getattr(value, "__func__", None)
    if type(function) is not types.FunctionType:
        raise CheckpointPublicationFinalizationError(
            "finalizer callback lacks a direct Python function"
        )
    return hashlib.sha256(
        marshal.dumps(function.__code__)
        + str(inspect.signature(function)).encode("ascii")
    ).hexdigest()


def _type_identity(value: object) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _verified_join_value(
    value: VerifiedCheckpoint,
    *,
    join_id: str,
) -> str:
    spec = next(
        (row for row in R10_GLOBAL_JOIN_SPECS if row.join_id == join_id),
        None,
    )
    if spec is None:
        raise CheckpointPublicationFinalizationError(
            "checkpoint lacks the required %s join specification" % join_id
        )
    states = {state.owner_id: state for state in value.owner_states}
    roots = []
    for owner_id in spec.owner_ids:
        claims = {
            claim.join_id: claim.value_sha256
            for claim in states[owner_id].join_claims
        }
        root = claims.get(join_id)
        if root is None:
            raise CheckpointPublicationFinalizationError(
                "checkpoint owner %s lacks %s" % (owner_id, join_id)
            )
        roots.append(root)
    if len(set(roots)) != 1:
        raise CheckpointPublicationFinalizationError(
            "checkpoint %s join differs" % join_id
        )
    return _require_sha256(roots[0], label=join_id)


class CheckpointPublicationFinalizationAuthority:
    """Gate the top R10 cursor on two causal writers and durable publication.

    The physical/runtime writer and runner writer are construction-bound as
    distinct live objects.  Each callback receives only the verified boundary
    and must independently return its retained PPO-drain frontier; neither is
    handed the archive's expected root to echo.  A third, distinct publication
    sink validates the real durable-publication receipt after the full owner
    export/global-join/Merkle/archive verification succeeds.

    This class intentionally has no production constructor callpoint yet.
    ``PPO_DRAIN_FRONTIER_SECOND_WRITER_INTEGRATED`` and
    ``CHECKPOINT_PUBLICATION_FINALIZER_INTEGRATED`` therefore remain false.
    """

    def __init__(
        self,
        *,
        registry: OrderedOwnerRegistry,
        physical_writer: object,
        physical_frontier: object,
        runner_writer: object,
        runner_frontier: object,
        publication_sink: object,
        validate_durable_publication: object,
        audit_authority: object,
        validate_audit_claim: object,
        leaf_live_mutation_writers: Optional[Tuple[object, ...]] = None,
        leaf_live_mutation_callbacks: Optional[Tuple[object, ...]] = None,
        diagnostic_allow_missing_live_highwaters: bool = False,
    ) -> None:
        if type(registry) is not OrderedOwnerRegistry:
            raise CheckpointPublicationFinalizationError(
                "publication finalizer requires an exact owner registry"
            )
        engine = registry.descriptor("env.plant").engine
        validate_r10_owner_registry(registry, engine=engine)
        if type(diagnostic_allow_missing_live_highwaters) is not bool:
            raise CheckpointPublicationFinalizationError(
                "diagnostic_allow_missing_live_highwaters must be an exact bool"
            )
        primary_identities = (
            physical_writer,
            runner_writer,
            publication_sink,
            audit_authority,
        )
        if any(value is None for value in primary_identities) or len(
            {id(value) for value in primary_identities}
        ) != len(primary_identities):
            raise CheckpointPublicationFinalizationError(
                "drain writers, publication sink, and audit authority must be distinct causal owners"
            )
        self._registry = registry
        self._physical_writer = physical_writer
        self._physical_frontier = _require_exact_bound_callback(
            physical_writer,
            physical_frontier,
            label="physical PPO-drain frontier validator",
        )
        self._runner_writer = runner_writer
        self._runner_frontier = _require_exact_bound_callback(
            runner_writer,
            runner_frontier,
            label="runner PPO-drain frontier validator",
        )
        self._publication_sink = publication_sink
        self._validate_durable_publication = _require_exact_bound_callback(
            publication_sink,
            validate_durable_publication,
            label="durable checkpoint publication validator",
        )
        self._audit_authority = audit_authority
        self._validate_audit_claim = _require_exact_bound_callback(
            audit_authority,
            validate_audit_claim,
            label="top R10 audit-claim validator",
        )
        if leaf_live_mutation_writers is None and leaf_live_mutation_callbacks is None:
            if not diagnostic_allow_missing_live_highwaters:
                raise CheckpointPublicationFinalizationError(
                    "production finalizer requires seven exact leaf live-mutation validators"
                )
            leaf_writers: Tuple[object, ...] = ()
            leaf_callbacks: Tuple[Callable[..., object], ...] = ()
        else:
            if diagnostic_allow_missing_live_highwaters:
                raise CheckpointPublicationFinalizationError(
                    "diagnostic live-highwater bypass cannot bind production leaf validators"
                )
            if (
                type(leaf_live_mutation_writers) is not tuple
                or type(leaf_live_mutation_callbacks) is not tuple
                or len(leaf_live_mutation_writers) != len(PPO_DRAIN_LEAF_OWNER_ORDER)
                or len(leaf_live_mutation_callbacks) != len(PPO_DRAIN_LEAF_OWNER_ORDER)
            ):
                raise CheckpointPublicationFinalizationError(
                    "production finalizer requires seven ordered leaf live-mutation validators"
                )
            leaf_writers = leaf_live_mutation_writers
            if any(value is None for value in leaf_writers):
                raise CheckpointPublicationFinalizationError(
                    "leaf live-mutation writer is missing"
                )
            all_identities = primary_identities + leaf_writers
            if len({id(value) for value in all_identities}) != len(all_identities):
                raise CheckpointPublicationFinalizationError(
                    "all drain, sink, audit, and leaf facts require distinct causal owners"
                )
            leaf_callbacks = tuple(
                _require_exact_bound_callback(
                    writer,
                    callback,
                    label=owner_kind + " live-mutation validator",
                )
                for owner_kind, writer, callback in zip(
                    PPO_DRAIN_LEAF_OWNER_ORDER,
                    leaf_writers,
                    leaf_live_mutation_callbacks,
                )
            )
        self._leaf_live_mutation_writers = leaf_writers
        self._leaf_live_mutation_callbacks = leaf_callbacks
        self._diagnostic_allow_missing_live_highwaters = (
            diagnostic_allow_missing_live_highwaters
        )
        callback_bindings = (
            (
                "physical_frontier",
                _type_identity(physical_writer),
                _callback_sha256(self._physical_frontier),
            ),
            (
                "runner_frontier",
                _type_identity(runner_writer),
                _callback_sha256(self._runner_frontier),
            ),
            (
                "durable_publication",
                _type_identity(publication_sink),
                _callback_sha256(self._validate_durable_publication),
            ),
            (
                "top_audit_claim",
                _type_identity(audit_authority),
                _callback_sha256(self._validate_audit_claim),
            ),
        )
        live_mutation_bindings = tuple(
            (
                owner_kind,
                _type_identity(writer),
                _callback_sha256(callback),
            )
            for owner_kind, writer, callback in zip(
                PPO_DRAIN_LEAF_OWNER_ORDER,
                leaf_writers,
                leaf_callbacks,
            )
        )
        all_identities = primary_identities + leaf_writers
        construction_plain = {
            "schema_version": SCHEMA_VERSION,
            "kind": "action_ball_full_mdp_checkpoint_finalizer_construction_v2",
            "registry_sha256": registry.content_sha256,
            "callback_bindings": callback_bindings,
            "live_mutation_bindings": live_mutation_bindings,
            "diagnostic_allow_missing_live_highwaters": (
                diagnostic_allow_missing_live_highwaters
            ),
            "distinct_identity_count": len({id(value) for value in all_identities}),
        }
        self._construction_bundle = _mint_finalizer_construction_bundle(
            _FinalizerConstructionPayload(
                authority_ref=weakref.ref(self),
                **construction_plain,
                content_sha256=canonical_sha256(construction_plain),
            )
        )
        self._pending: Optional[_PendingCheckpointPublicationFinalization] = None
        self._poisoned = False
        self._poison_reason: Optional[str] = None
        self._seen_checkpoint_bytes_sha256: set[str] = set()
        self._lock = threading.RLock()

    @property
    def registry(self) -> OrderedOwnerRegistry:
        return self._registry

    @property
    def construction_bundle(
        self,
    ) -> CheckpointPublicationFinalizerConstructionBundle:
        return self._construction_bundle

    @property
    def construction_receipt(
        self,
    ) -> CheckpointPublicationFinalizerConstructionBundle:
        """Compatibility alias; the returned object is an opaque bundle."""

        return self._construction_bundle

    def require_owned_construction_bundle(
        self,
        bundle: object,
    ) -> CheckpointPublicationFinalizerConstructionBundle:
        if (
            type(bundle) is not CheckpointPublicationFinalizerConstructionBundle
            or bundle is not self._construction_bundle
            or CheckpointPublicationFinalizerConstructionBundle._payload(
                bundle
            ).authority_ref() is not self
        ):
            raise CheckpointPublicationFinalizationError(
                "construction bundle is foreign or not finalizer-issued"
            )
        return bundle

    @property
    def runtime_poisoned(self) -> bool:
        return self._poisoned

    def _poison(self, reason: str) -> None:
        self._poisoned = True
        if self._poison_reason is None:
            self._poison_reason = reason

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise CheckpointPublicationFinalizationError(
                "checkpoint publication finalizer is poisoned; retry is forbidden: "
                + str(self._poison_reason)
            )

    def finalize_publication(
        self,
        publication: object,
        durable_publication_receipt: object,
    ) -> None:
        """Prepare the one publication; no final receipt exists before claim validation."""

        with self._lock:
            self._require_healthy()
            if (
                self._pending is not None
                and not self._pending.consumed
            ):
                self._poison(
                    "a second publication arrived before the prior top R10 ACK"
                )
                raise CheckpointPublicationFinalizationError(
                    "a prior checkpoint publication still awaits the top R10 ACK"
                )
            if (
                type(publication) is not CheckpointPublication
                or publication.schema_version != SCHEMA_VERSION
                or type(publication.schema_version) is not int
                or publication.kind != CHECKPOINT_RECEIPT_KIND
                or publication.integration_status != INTEGRATION_STATUS
                or publication.runtime_wiring is not False
                or publication.continuation_authorized is not False
            ):
                raise CheckpointPublicationFinalizationError(
                    "global checkpoint publication has the wrong exact authority"
                )
            try:
                verified = verify_checkpoint_candidate(
                    publication.blob,
                    expected_external_pins=publication.external_pins,
                    expected_registry=self._registry,
                )
                if (
                    verified.checkpoint_bytes_sha256
                    in self._seen_checkpoint_bytes_sha256
                ):
                    raise CheckpointPublicationFinalizationError(
                        "checkpoint publication was already finalized"
                    )
                drain_root = _verified_join_value(
                    verified,
                    join_id="ppo_drain_frontier",
                )
                physical_root = _require_sha256(
                    self._physical_frontier(verified.boundary),
                    label="physical writer PPO-drain frontier",
                )
                runner_projection = self._runner_frontier(verified.boundary)
                if type(runner_projection) is PpoDrainRunnerWriterProjection:
                    if (
                        runner_projection.schema_version != SCHEMA_VERSION
                        or type(runner_projection.schema_version) is not int
                        or runner_projection.kind
                        != "action_ball_r10_runner_drain_frontier_projection_v1"
                        or type(runner_projection.mutation_version_highwaters)
                        is not tuple
                        or tuple(
                            owner_kind
                            for owner_kind, _value
                            in runner_projection.mutation_version_highwaters
                        )
                        != PPO_DRAIN_LEAF_OWNER_ORDER
                        or any(
                            type(value) is not int or value < 0
                            for _owner_kind, value
                            in runner_projection.mutation_version_highwaters
                        )
                    ):
                        raise CheckpointPublicationFinalizationError(
                            "runner drain frontier projection is malformed"
                        )
                    runner_root = _require_sha256(
                        runner_projection.checkpoint_frontier_sha256,
                        label="runner writer PPO-drain frontier",
                    )
                    expected_highwaters = dict(
                        runner_projection.mutation_version_highwaters
                    )
                elif self._diagnostic_allow_missing_live_highwaters:
                    runner_root = _require_sha256(
                        runner_projection,
                        label="diagnostic runner writer PPO-drain frontier",
                    )
                    expected_highwaters = None
                else:
                    raise CheckpointPublicationFinalizationError(
                        "production runner writer lacks mutation highwaters"
                    )
                if physical_root != drain_root or runner_root != drain_root:
                    raise CheckpointPublicationFinalizationError(
                        "independent PPO-drain frontier writer differs from archive"
                    )
                live_highwaters_validated = False
                if self._leaf_live_mutation_callbacks:
                    if expected_highwaters is None:
                        raise CheckpointPublicationFinalizationError(
                            "leaf live-mutation join lacks runner highwaters"
                        )
                    for owner_kind, callback in zip(
                        PPO_DRAIN_LEAF_OWNER_ORDER,
                        self._leaf_live_mutation_callbacks,
                    ):
                        # Deliberately do not pass the expected value.  The leaf
                        # sees only the independently verified boundary and its
                        # fixed role, then returns its own live observation.
                        projection = callback(verified.boundary, owner_kind)
                        if (
                            type(projection)
                            is not PpoDrainLeafLiveMutationProjection
                            or projection.schema_version != SCHEMA_VERSION
                            or type(projection.schema_version) is not int
                            or projection.kind
                            != "action_ball_r10_leaf_live_mutation_projection_v1"
                            or projection.owner_kind != owner_kind
                            or type(projection.mutation_version) is not int
                            or projection.mutation_version < 0
                            or projection.mutation_version
                            != expected_highwaters[owner_kind]
                        ):
                            raise CheckpointPublicationFinalizationError(
                                owner_kind
                                + " live mutation differs from drained highwater"
                            )
                    live_highwaters_validated = True
                elif not self._diagnostic_allow_missing_live_highwaters:
                    raise CheckpointPublicationFinalizationError(
                        "production finalizer lacks leaf live-mutation validators"
                    )
                validated_durable = self._validate_durable_publication(
                    publication,
                    verified,
                    durable_publication_receipt,
                )
                if validated_durable is not durable_publication_receipt:
                    raise CheckpointPublicationFinalizationError(
                        "durable publication receipt identity differs"
                    )
            except CheckpointPublicationFinalizationError as exc:
                self._poison(str(exc))
                raise
            except BaseException as exc:
                self._poison(
                    "checkpoint publication finalization callback failed"
                )
                raise CheckpointPublicationFinalizationError(
                    "checkpoint publication finalization failed"
                ) from exc
            self._pending = _PendingCheckpointPublicationFinalization(
                publication=publication,
                verified=verified,
                durable_receipt=durable_publication_receipt,
                ppo_drain_frontier_sha256=drain_root,
                live_highwaters_validated=live_highwaters_validated,
            )
            self._seen_checkpoint_bytes_sha256.add(
                verified.checkpoint_bytes_sha256
            )
            return None

    def validate(
        self, publication: object, audit_claim: object
    ) -> Optional[CheckpointPublication]:
        """Validate, mint an audit receipt, and return the exact publication.

        The return identity is the ABI consumed by
        ``finalize_r10_audit_frontier``.  This method does not ACK or retire the
        top-owned audit claim; only that top method performs the cursor change.
        A diagnostic constructor deliberately returns ``None`` and therefore
        cannot satisfy that top identity gate.
        """

        with self._lock:
            self._require_healthy()
            pending = self._pending
            if (
                pending is None
                or pending.consumed
                or publication is not pending.publication
                or type(publication) is not CheckpointPublication
            ):
                raise CheckpointPublicationFinalizationError(
                    "publication is stale, unfinalized, or already consumed"
                )
            try:
                validated_claim = self._validate_audit_claim(
                    audit_claim,
                    pending.verified.boundary,
                    pending.ppo_drain_frontier_sha256,
                )
            except BaseException as exc:
                self._poison("top R10 audit claim validation failed")
                raise CheckpointPublicationFinalizationError(
                    "top R10 audit claim validation failed"
                ) from exc
            if validated_claim is not audit_claim:
                self._poison("top R10 audit claim identity differs")
                raise CheckpointPublicationFinalizationError(
                    "top R10 audit claim identity differs"
                )
            pending.consumed = True
            if pending.live_highwaters_validated:
                pending.finalization_receipt = _mint_finalization_receipt(
                    _FinalizationReceiptPayload(
                        authority_ref=weakref.ref(self),
                        publication=pending.publication,
                        schema_version=SCHEMA_VERSION,
                        kind="action_ball_full_mdp_checkpoint_publication_finalization_v2",
                        checkpoint_bytes_sha256=(
                            pending.verified.checkpoint_bytes_sha256
                        ),
                        checkpoint_content_sha256=pending.verified.content_sha256,
                        owner_root_sha256=pending.verified.owner_root_sha256,
                        registry_sha256=pending.verified.registry.content_sha256,
                        boundary_sha256=boundary_sha256(
                            pending.verified.boundary
                        ),
                        ppo_drain_frontier_sha256=(
                            pending.ppo_drain_frontier_sha256
                        ),
                        top_audit_claim_validated=True,
                        live_highwaters_validated=True,
                        runtime_wiring=False,
                    )
                )
            # A diagnostic constructor can exercise archive/writer/durable/
            # claim rejection without becoming a top ACK authority.  Returning
            # ``None`` makes the top owner's exact-identity gate fail closed.
            return (
                pending.publication
                if pending.live_highwaters_validated
                else None
            )

    def require_finalization_receipt(
        self,
        publication: object,
    ) -> CheckpointPublicationFinalizationReceipt:
        """Return the exact capability only after ``validate`` succeeded."""

        with self._lock:
            pending = self._pending
            if (
                pending is None
                or not pending.consumed
                or publication is not pending.publication
                or pending.finalization_receipt is None
            ):
                raise CheckpointPublicationFinalizationError(
                    "finalization receipt is unavailable before exact validation"
                )
            return pending.finalization_receipt

    def require_owned_finalization_receipt(
        self,
        publication: object,
        receipt: object,
    ) -> CheckpointPublicationFinalizationReceipt:
        expected = self.require_finalization_receipt(publication)
        payload = (
            _lookup_finalization_receipt(receipt)
            if type(receipt) is CheckpointPublicationFinalizationReceipt
            else None
        )
        if (
            receipt is not expected
            or payload is None
            or payload.authority_ref() is not self
            or payload.publication is not publication
        ):
            raise CheckpointPublicationFinalizationError(
                "finalization receipt is foreign or not finalizer-issued"
            )
        return receipt


@dataclass(frozen=True)
class RestoreReceipt:
    schema_version: int
    kind: str
    integration_status: str
    runtime_wiring: bool
    continuation_authorized: bool
    owner_root_sha256: str
    prepared_owner_ids: Tuple[str, ...]
    committed_owner_ids: Tuple[str, ...]
    all_prepared_before_first_commit: bool
    runtime_poisoned: bool
    retry_permitted: bool


class CheckpointRestoreCoordinator:
    """One-shot fail-stop restore coordinator bound to exact live owner objects."""

    def __init__(
        self,
        *,
        engine: OwnerEngine,
        registry: OrderedOwnerRegistry,
        owners: Tuple[FullMDPCheckpointOwner, ...],
    ) -> None:
        validate_r10_owner_registry(registry, engine=engine)
        self._engine = engine
        self._registry = registry
        self._owners = _bind_owners(registry, owners)
        self._poisoned = False
        self._completed = False

    @property
    def runtime_poisoned(self) -> bool:
        return self._poisoned

    @property
    def completed(self) -> bool:
        return self._completed

    def _poison_every_owner(self, reason: str) -> Tuple[str, ...]:
        failures = []
        self._poisoned = True
        for owner, descriptor in zip(self._owners, self._registry.descriptors):
            try:
                owner.poison_restore(reason)
            except BaseException as exc:  # fail-stop includes interrupts
                try:
                    detail = str(exc)
                except BaseException:
                    detail = "<unprintable poison exception>"
                failures.append("%s:%s" % (descriptor.owner_id, detail))
        return tuple(failures)

    @staticmethod
    def _rollback_all(
        prepared: Sequence[Tuple[FullMDPCheckpointOwner, PreparedRestoreToken]]
    ) -> Tuple[str, ...]:
        failures = []
        for owner, token in reversed(tuple(prepared)):
            try:
                owner.rollback_restore(token)
            except BaseException as exc:
                try:
                    detail = str(exc)
                except BaseException:
                    detail = "<unprintable rollback exception>"
                failures.append("%s:%s" % (token.owner_id, detail))
        return tuple(failures)

    def restore(
        self,
        blob: bytes,
        *,
        expected_external_pins: ExternalCheckpointPins,
    ) -> RestoreReceipt:
        if self._poisoned:
            raise CheckpointRuntimePoisonedError(
                "restore coordinator is poisoned; retry is forbidden"
            )
        if self._completed:
            raise FullMDPCheckpointError("restore coordinator already committed once")
        # A restore attempt is one-shot from its first external input read.  A
        # malformed blob or a throwing owner accessor is not evidence that the
        # live runtime stayed untouched: adapters are arbitrary external code.
        # Therefore verification and the first live observation share the same
        # fail-stop boundary as prepare/commit below.
        try:
            verified = verify_checkpoint_candidate(
                blob,
                expected_external_pins=expected_external_pins,
                expected_registry=self._registry,
            )
            if verified.engine is not self._engine:
                raise ExternalCheckpointPinError("live engine differs from checkpoint")
            observations = _observe_owners(self._owners)
        except BaseException:
            self._poison_every_owner(
                "restore verification/initial observation failed; retry is forbidden"
            )
            raise
        prepared = []
        token_object_ids = set()
        try:
            for owner, descriptor, envelope in zip(
                self._owners, self._registry.descriptors, verified.owner_states
            ):
                token = owner.prepare_restore(
                    envelope,
                    verified.immutable_pins,
                    verified.owner_root_sha256,
                )
                if type(token) is not PreparedRestoreToken:
                    raise CheckpointPrepareError(
                        "owner %s returned an untyped restore token" % descriptor.owner_id,
                        runtime_poisoned=False,
                    )
                if (
                    token.owner_id != descriptor.owner_id
                    or token.descriptor_sha256 != descriptor_sha256(descriptor)
                    or token.checkpoint_owner_root_sha256 != verified.owner_root_sha256
                ):
                    raise CheckpointPrepareError(
                        "owner %s returned a mismatched restore token" % descriptor.owner_id,
                        runtime_poisoned=False,
                    )
                if token.opaque_token is None:
                    raise CheckpointPrepareError(
                        "owner %s returned an empty restore token" % descriptor.owner_id,
                        runtime_poisoned=False,
                    )
                if id(token.opaque_token) in token_object_ids:
                    raise CheckpointPrepareError(
                        "prepared owner tokens alias one mutable authority",
                        runtime_poisoned=False,
                    )
                token_object_ids.add(id(token.opaque_token))
                prepared.append((owner, token))
                _assert_observational_baseline(
                    self._owners,
                    observations,
                    operation="owner prepare_restore",
                )
        except BaseException as exc:
            # Only exact, registered tokens are eligible for best-effort
            # rollback.  This is deliberately not called (or reported as) a
            # proof that the whole runtime returned to its baseline: the
            # failing owner may have staged state before returning no trusted
            # token, and any owner callback/accessor may have side effects.
            rollback_attempted_owner_ids = tuple(
                token.owner_id for _, token in reversed(tuple(prepared))
            )
            rollback_failures = self._rollback_all(prepared)
            poison_failures = self._poison_every_owner(
                "restore prepare failed; retry is forbidden"
            )
            raise CheckpointPrepareError(
                "restore prepare failed; rollback_attempted_owner_ids=%s "
                "rollback_failures=%s poison_failures=%s: %s"
                % (
                    rollback_attempted_owner_ids,
                    rollback_failures,
                    poison_failures,
                    exc,
                ),
                runtime_poisoned=True,
            ) from exc

        # No commit is reachable until the loop above has prepared every owner.
        committed_ids = []
        try:
            for owner, token in prepared:
                owner.commit_restore(token)
                committed_ids.append(token.owner_id)
            committed_observations = _observe_owners(self._owners)
            for observation, envelope in zip(
                committed_observations, verified.owner_states
            ):
                version, live_digest = observation
                if version != envelope.mutation_version:
                    raise CheckpointCommitError(
                        "committed owner mutation version differs for %s"
                        % envelope.owner_id
                    )
                if live_digest != envelope.live_digest_sha256:
                    raise CheckpointCommitError(
                        "committed owner live digest differs for %s" % envelope.owner_id
                    )
        except BaseException as exc:
            rollback_attempted_owner_ids = tuple(
                token.owner_id for _, token in reversed(tuple(prepared))
            )
            rollback_failures = self._rollback_all(prepared)
            poison_failures = self._poison_every_owner(
                "restore commit failed; retry is forbidden"
            )
            raise CheckpointCommitError(
                "restore commit failed after %s; "
                "rollback_attempted_owner_ids=%s rollback_failures=%s "
                "poison_failures=%s: %s"
                % (
                    tuple(committed_ids),
                    rollback_attempted_owner_ids,
                    rollback_failures,
                    poison_failures,
                    exc,
                )
            ) from exc
        self._completed = True
        return RestoreReceipt(
            schema_version=SCHEMA_VERSION,
            kind="action_ball_full_mdp_checkpoint_restore_receipt_v1",
            integration_status=INTEGRATION_STATUS,
            runtime_wiring=False,
            continuation_authorized=False,
            owner_root_sha256=verified.owner_root_sha256,
            prepared_owner_ids=self._registry.owner_ids,
            committed_owner_ids=tuple(committed_ids),
            all_prepared_before_first_commit=True,
            runtime_poisoned=False,
            retry_permitted=False,
        )


@dataclass(frozen=True)
class FamilyCommonOutcomeReceipt:
    """One family-side view of the common R06 outcome before final treatment."""

    family: PolicyFamily
    shot_key_sha256: str
    common_fact_sha256: str
    common_view_sha256: str
    common_ack_sha256: str
    common_payment_sha256: str
    placement_fact_sha256: str
    placement_view_sha256: str
    placement_ack_sha256: str
    placement_raw_payment_sha256: str
    common_consumer_accounted: bool
    placement_consumer_accounted: bool
    placement_treatment_gain: int


@dataclass(frozen=True)
class ACCommonJoinReceipt:
    schema_version: int
    kind: str
    common_sha256: str
    a_placement_treatment_gain: int
    c_placement_treatment_gain: int
    both_consumers_accounted: bool
    runtime_wiring: bool
    continuation_authorized: bool


def validate_ac_common_outcome_join(
    a: FamilyCommonOutcomeReceipt,
    c: FamilyCommonOutcomeReceipt,
) -> ACCommonJoinReceipt:
    """Enforce exact-common facts/views/acks; only final A=1/C=0 gain may differ."""

    if type(a) is not FamilyCommonOutcomeReceipt or type(c) is not FamilyCommonOutcomeReceipt:
        raise CheckpointJoinError("A/C join requires exact FamilyCommonOutcomeReceipt values")
    if a.family is not PolicyFamily.A or c.family is not PolicyFamily.C:
        raise CheckpointJoinError("A/C common join family order differs")
    hash_fields = (
        "shot_key_sha256",
        "common_fact_sha256",
        "common_view_sha256",
        "common_ack_sha256",
        "common_payment_sha256",
        "placement_fact_sha256",
        "placement_view_sha256",
        "placement_ack_sha256",
        "placement_raw_payment_sha256",
    )
    for side, label in ((a, "A"), (c, "C")):
        for name in hash_fields:
            _require_sha256(getattr(side, name), label=label + "." + name)
        if type(side.common_consumer_accounted) is not bool:
            raise CheckpointJoinError(label + " common consumer flag must be exact bool")
        if type(side.placement_consumer_accounted) is not bool:
            raise CheckpointJoinError(label + " placement consumer flag must be exact bool")
        if type(side.placement_treatment_gain) is not int:
            raise CheckpointJoinError(label + " treatment gain must be an exact int")
    differing = tuple(name for name in hash_fields if getattr(a, name) != getattr(c, name))
    if differing:
        raise CheckpointJoinError("A/C common outcome differs before treatment: %s" % (differing,))
    if not (
        a.common_consumer_accounted
        and a.placement_consumer_accounted
        and c.common_consumer_accounted
        and c.placement_consumer_accounted
    ):
        raise CheckpointJoinError("A/C did not account both R06 consumers")
    if a.placement_treatment_gain != 1 or c.placement_treatment_gain != 0:
        raise CheckpointJoinError("A/C placement treatment must be exactly A=1,C=0")
    common = {
        name: getattr(a, name) for name in hash_fields
    }
    return ACCommonJoinReceipt(
        schema_version=1,
        kind="action_ball_r10_ac_common_outcome_join_receipt_v1",
        common_sha256=canonical_sha256(common),
        a_placement_treatment_gain=1,
        c_placement_treatment_gain=0,
        both_consumers_accounted=True,
        runtime_wiring=False,
        continuation_authorized=False,
    )


__all__ = [
    "ACCommonJoinReceipt",
    "CHECKPOINT_KIND",
    "CHECKPOINT_RECEIPT_KIND",
    "CONTINUATION_CLAIM",
    "CheckpointBoundary",
    "CheckpointBoundaryError",
    "CheckpointCommitError",
    "CheckpointJoinError",
    "CheckpointPrepareError",
    "CheckpointPublication",
    "CheckpointPublicationFinalizationAuthority",
    "CheckpointPublicationFinalizationError",
    "CheckpointPublicationFinalizationReceipt",
    "CheckpointPublicationFinalizerConstructionBundle",
    "CheckpointRestoreCoordinator",
    "CheckpointRuntimePoisonedError",
    "CheckpointSealError",
    "ENGINE_CONTINUATION_PROVEN",
    "EnvironmentStepPhase",
    "EqualityJoinSpec",
    "ExternalCheckpointPinError",
    "ExternalCheckpointPins",
    "FORMAL_EXACT_RESUME_INTEGRATED",
    "FamilyCommonOutcomeReceipt",
    "FullMDPCheckpointError",
    "FullMDPCheckpointOwner",
    "INTEGRATION_STATUS",
    "ImmutableCheckpointPins",
    "LAUNCH_AUTHORIZED",
    "OpaqueOwnerState",
    "OrderedOwnerRegistry",
    "OwnerDescriptor",
    "OwnerEngine",
    "OwnerFreezeReceipt",
    "OwnerJoinClaim",
    "OwnerRegistryError",
    "OwnerScope",
    "PPOBoundaryPhase",
    "PPO_DRAIN_FRONTIER_JOIN_CONTRACT_IMPLEMENTED",
    "PPO_DRAIN_FRONTIER_SECOND_WRITER_INTEGRATED",
    "PPO_DRAIN_LEAF_OWNER_ORDER",
    "PPO_DRAIN_LIVE_HIGHWATER_JOIN_INTEGRATED",
    "PpoDrainLeafLiveMutationProjection",
    "PpoDrainRunnerWriterProjection",
    "PhaseMatrixRule",
    "PolicyFamily",
    "PreparedRestoreToken",
    "R03Phase",
    "R03_CONSUMER_COUNT",
    "R03_FULL_CONSUMER_MASK",
    "R05Phase",
    "R06FlightPhase",
    "R06MailboxPhase",
    "R06_CONSUMER_COUNT",
    "R06_FULL_CONSUMER_MASK",
    "R10_GLOBAL_JOIN_SPECS",
    "R10_OWNER_DEPENDENCIES",
    "R10_OWNER_ORDER",
    "R10_PHASE_MATRIX",
    "RUNTIME_WIRING",
    "CHECKPOINT_PUBLICATION_FINALIZER_INTEGRATED",
    "RecurrentFrontierStatus",
    "ResetPhase",
    "RestoreReceipt",
    "SCHEMA_VERSION",
    "SealStatus",
    "VerifiedCheckpoint",
    "WorldCheckpointPhase",
    "boundary_sha256",
    "canonical_sha256",
    "descriptor_sha256",
    "make_opaque_owner_state",
    "owner_merkle_root",
    "seal_checkpoint_candidate",
    "validate_ac_common_outcome_join",
    "validate_checkpoint_boundary",
    "validate_global_joins",
    "validate_r10_owner_registry",
    "verify_checkpoint_candidate",
]
