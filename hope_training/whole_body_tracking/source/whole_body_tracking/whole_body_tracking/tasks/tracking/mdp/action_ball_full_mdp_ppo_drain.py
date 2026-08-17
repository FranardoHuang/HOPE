"""One pre-optimizer PPO-boundary drain for the fresh ActionBall runtime.

This module is a device-to-host *coordination* seam, not a producer of facts.
Seven independent hot owners prepare their own device-resident counter rows.
The coordinator concatenates those rows and performs exactly one blocking
device-to-host transfer before an optimizer update.  It then decodes the rows,
rejects leaf faults/invariants, and checks only conservation equations whose
two sides have disjoint causal writers.

The public receipt is an opaque, one-shot capability.  A successful optimizer
step must acknowledge that exact receipt; an optimizer failure, any failure
after transfer starts, or a partial acknowledgement permanently poisons the
coordinator and every leaf.  A clean failure before transfer starts may abort
all prepared leaves and retry the same update.

There is deliberately no packet digest.  A digest made by the same operation
that made the packet would only attest self-consistency, not an external fact.
Likewise, this module does not call the legacy per-leaf ``drain_ppo_boundary``
methods and does not manufacture RewardManager view/payment evidence.  Those
expectations belong to a later host-side join owned by the real top/env path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Callable, Mapping, Protocol, Sequence
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode


SCHEMA_VERSION = 1
RECEIPT_KIND = "action_ball_full_mdp_pre_optimizer_ppo_drain_receipt_v1"
CHECKPOINT_KIND = "action_ball_full_mdp_ppo_drain_checkpoint_v1"

OWNER_ORDER = (
    "r05_runtime",
    "motion",
    "racket",
    "physical_ball",
    "r06_landing_outcome",
    "r03_strike_fact",
    "r07_recovery",
)

RUNTIME_INTEGRATED = False
RUNNER_INTEGRATED = False
CUDA_PROFILED = False
LAUNCH_AUTHORIZED = False

_PREPARE_METHOD = "prepare_pre_optimizer_ppo_boundary_device_pack"
_ABORT_METHOD = "abort_pre_optimizer_ppo_boundary_device_pack"
_ACK_METHOD = "acknowledge_pre_optimizer_ppo_boundary"
_POISON_METHOD = "poison_pre_optimizer_ppo_boundary"


class _PrepareNoHostObservationMode(TorchDispatchMode):
    """Reject Tensor host observation inside every leaf prepare callback."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        keyword_args = kwargs or {}
        if str(func) in (
            "aten._local_scalar_dense.default",
            "aten.item.default",
        ):
            raise ActionBallFullMdpPpoDrainError(
                "leaf prepare attempted a scalar host tensor observation"
            )
        if str(func) == "aten._to_copy.default":
            destination = keyword_args.get("device")
            source = next(
                (value for value in args if isinstance(value, torch.Tensor)),
                None,
            )
            if (
                isinstance(source, torch.Tensor)
                and source.device.type != "cpu"
                and destination is not None
                and torch.device(destination).type == "cpu"
            ):
                raise ActionBallFullMdpPpoDrainError(
                    "leaf prepare attempted an independent device-to-host transfer"
                )
        return func(*args, **keyword_args)


class ActionBallFullMdpPpoDrainError(RuntimeError):
    """Base error for the global pre-optimizer drain."""


class ActionBallFullMdpPpoDrainPrepareError(ActionBallFullMdpPpoDrainError):
    """A preparation failed and reports whether the same update may retry."""

    def __init__(self, message: str, *, retry_permitted: bool) -> None:
        super().__init__(message)
        self.retry_permitted = retry_permitted
        self.runtime_poisoned = not retry_permitted


class ActionBallFullMdpPpoDrainPoisonedError(ActionBallFullMdpPpoDrainError):
    """The drain crossed an irreversible boundary and is fail-stop poisoned."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.retry_permitted = False
        self.runtime_poisoned = True


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ActionBallFullMdpPpoDrainError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _exact_name(value: object, *, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ActionBallFullMdpPpoDrainError(
            f"{label} must be a non-empty ASCII string"
        )
    return value


def _exact_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionBallFullMdpPpoDrainError(
            f"{label} must be one lowercase hexadecimal SHA-256"
        )
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DeviceDrainFieldSpec:
    """One exact int64 device field in a leaf-owned packed row.

    ``cardinality`` may be ``scalar``, ``per_env``, or ``fixed``.  A fixed
    field has an exact positive ``fixed_width`` independent of ``num_envs``;
    it lets a leaf append a bounded device journal to the one global transfer.
    Scalar and per-environment fields deliberately retain their legacy schema
    identity.
    """

    name: str
    cardinality: str = "scalar"
    minimum: int = 0
    fixed_width: int | None = None

    def __post_init__(self) -> None:
        _exact_name(self.name, label="field name")
        if self.cardinality not in ("scalar", "per_env", "fixed"):
            raise ActionBallFullMdpPpoDrainError(
                "field cardinality must be scalar, per_env, or fixed"
            )
        _exact_int(self.minimum, label=f"minimum for {self.name}")
        if self.cardinality == "fixed":
            _exact_int(
                self.fixed_width,
                label=f"fixed_width for {self.name}",
                minimum=1,
            )
        elif self.fixed_width is not None:
            raise ActionBallFullMdpPpoDrainError(
                "fixed_width is allowed only for fixed cardinality"
            )

    def width(self, num_envs: int) -> int:
        if self.cardinality == "scalar":
            return 1
        if self.cardinality == "per_env":
            return num_envs
        assert self.fixed_width is not None
        return self.fixed_width


@dataclass(frozen=True)
class LeafDrainSchema:
    """Ordered device row schema for one independent hot owner."""

    owner_kind: str
    fields: tuple[DeviceDrainFieldSpec, ...]

    def __post_init__(self) -> None:
        _exact_name(self.owner_kind, label="owner kind")
        if not self.fields or any(
            not isinstance(field, DeviceDrainFieldSpec) for field in self.fields
        ):
            raise ActionBallFullMdpPpoDrainError(
                "leaf schema fields must be non-empty DeviceDrainFieldSpec values"
            )
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise ActionBallFullMdpPpoDrainError(
                f"{self.owner_kind} leaf schema has duplicate fields"
            )

    def width(self, num_envs: int) -> int:
        return sum(field.width(num_envs) for field in self.fields)


CheckpointFieldIdentity = (
    tuple[str, str, int] | tuple[str, str, int, int]
)
CheckpointSchemaIdentity = tuple[
    tuple[str, tuple[CheckpointFieldIdentity, ...]], ...
]


def _checkpoint_schema_identity(
    schemas: Sequence[LeafDrainSchema],
) -> CheckpointSchemaIdentity:
    return tuple(
        (
            schema.owner_kind,
            tuple(
                (
                    field.name,
                    field.cardinality,
                    field.minimum,
                    field.fixed_width,
                )
                if field.cardinality == "fixed"
                else (field.name, field.cardinality, field.minimum)
                for field in schema.fields
            ),
        )
        for schema in schemas
    )


@dataclass(frozen=True)
class PpoDrainCheckpointContent:
    """Portable immutable frontier content; it is not restore authority.

    R10 may serialize these fields in its externally sealed envelope.  A
    caller-made or self-consistently re-hashed instance is only data: restore
    accepts it solely through an external VerifiedCheckpoint authority's
    one-shot projection.
    """

    schema_version: int
    kind: str
    num_envs: int
    device_type: str
    device_index: int | None
    owner_order: tuple[str, ...]
    schema_identity: CheckpointSchemaIdentity
    checkpoint_boundary_sha256: str
    next_update_index: int
    operation_sequence: int
    drain_sequence: int
    last_completed_environment_steps: int
    mutation_version_highwaters: tuple[tuple[str, int | None], ...]
    checkpoint_frontier_sha256: str

    def __post_init__(self) -> None:
        _exact_int(self.schema_version, label="checkpoint schema_version", minimum=1)
        if self.kind != CHECKPOINT_KIND:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint content kind differs"
            )
        _exact_int(self.num_envs, label="checkpoint num_envs", minimum=1)
        if self.device_type not in ("cpu", "cuda"):
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint device_type must be cpu or cuda"
            )
        if self.device_index is not None:
            _exact_int(self.device_index, label="checkpoint device_index")
        if self.owner_order != OWNER_ORDER:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint owner order differs"
            )
        if (
            type(self.schema_identity) is not tuple
            or tuple(owner for owner, _fields in self.schema_identity)
            != OWNER_ORDER
        ):
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint schema identity owner order differs"
            )
        for owner_kind, fields in self.schema_identity:
            _exact_name(owner_kind, label="checkpoint schema owner")
            if type(fields) is not tuple or not fields:
                raise ActionBallFullMdpPpoDrainError(
                    "checkpoint schema fields must be a non-empty exact tuple"
                )
            for field in fields:
                if type(field) is not tuple or len(field) not in (3, 4):
                    raise ActionBallFullMdpPpoDrainError(
                        "checkpoint schema field identity differs"
                    )
                name, cardinality, minimum = field[:3]
                _exact_name(name, label="checkpoint schema field")
                if cardinality not in ("scalar", "per_env", "fixed"):
                    raise ActionBallFullMdpPpoDrainError(
                        "checkpoint schema field cardinality differs"
                    )
                _exact_int(
                    minimum,
                    label=f"checkpoint schema minimum for {name}",
                )
                if cardinality == "fixed":
                    if len(field) != 4:
                        raise ActionBallFullMdpPpoDrainError(
                            "checkpoint fixed schema width is absent"
                        )
                    _exact_int(
                        field[3],
                        label=f"checkpoint fixed schema width for {name}",
                        minimum=1,
                    )
                elif len(field) != 3:
                    raise ActionBallFullMdpPpoDrainError(
                        "checkpoint non-fixed schema has a foreign width"
                    )
        _exact_sha256(
            self.checkpoint_boundary_sha256,
            label="checkpoint boundary SHA-256",
        )
        _exact_int(self.next_update_index, label="checkpoint next_update_index")
        _exact_int(self.operation_sequence, label="checkpoint operation_sequence")
        _exact_int(self.drain_sequence, label="checkpoint drain_sequence")
        if (
            type(self.last_completed_environment_steps) is not int
            or self.last_completed_environment_steps < -1
        ):
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint last_completed_environment_steps must be >= -1"
            )
        if tuple(name for name, _value in self.mutation_version_highwaters) != OWNER_ORDER:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint mutation highwater owner order differs"
            )
        for owner_kind, value in self.mutation_version_highwaters:
            _exact_name(owner_kind, label="checkpoint mutation highwater owner")
            if value is not None:
                _exact_int(
                    value,
                    label=f"checkpoint {owner_kind} mutation highwater",
                )
        _exact_sha256(
            self.checkpoint_frontier_sha256,
            label="checkpoint frontier SHA-256",
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return canonical portable data excluding its derived root."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "num_envs": self.num_envs,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "owner_order": self.owner_order,
            "schema_identity": self.schema_identity,
            "checkpoint_boundary_sha256": self.checkpoint_boundary_sha256,
            "next_update_index": self.next_update_index,
            "operation_sequence": self.operation_sequence,
            "drain_sequence": self.drain_sequence,
            "last_completed_environment_steps": (
                self.last_completed_environment_steps
            ),
            "mutation_version_highwaters": self.mutation_version_highwaters,
        }

    def validate_derived_root(self) -> None:
        if _canonical_sha256(self.canonical_payload()) != self.checkpoint_frontier_sha256:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint frontier root differs from canonical content"
            )


@dataclass(frozen=True)
class _CheckpointSnapshotPayload:
    owner_identity: object
    boundary: object
    content: PpoDrainCheckpointContent


class PpoDrainCheckpointSnapshot:
    """Opaque owner-issued snapshot with portable, immutable frontier content."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("PPO drain checkpoint snapshots are owner-issued only")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PPO drain checkpoint snapshots are immutable")

    def __copy__(self):
        raise TypeError("PPO drain checkpoint snapshots cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("PPO drain checkpoint snapshots cannot be copied")

    def __reduce__(self):
        raise TypeError(
            "serialize checkpoint snapshot.content, not its owner authority"
        )

    @staticmethod
    def _payload(value: "PpoDrainCheckpointSnapshot") -> _CheckpointSnapshotPayload:
        payload = _lookup_checkpoint_snapshot(value)
        if payload is None:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint snapshot is not drain-owner-issued"
            )
        return payload

    @property
    def content(self) -> PpoDrainCheckpointContent:
        return self._payload(self).content

    @property
    def schema_version(self) -> int:
        return self.content.schema_version

    @property
    def kind(self) -> str:
        return self.content.kind

    @property
    def checkpoint_boundary_sha256(self) -> str:
        return self.content.checkpoint_boundary_sha256

    @property
    def checkpoint_frontier_sha256(self) -> str:
        return self.content.checkpoint_frontier_sha256

    @property
    def next_update_index(self) -> int:
        return self.content.next_update_index

    @property
    def operation_sequence(self) -> int:
        return self.content.operation_sequence

    @property
    def drain_sequence(self) -> int:
        return self.content.drain_sequence

    @property
    def last_completed_environment_steps(self) -> int:
        return self.content.last_completed_environment_steps

    @property
    def mutation_version_highwaters(self) -> tuple[tuple[str, int | None], ...]:
        return self.content.mutation_version_highwaters


def _make_checkpoint_snapshot_registry():
    rows: weakref.WeakKeyDictionary[
        PpoDrainCheckpointSnapshot,
        _CheckpointSnapshotPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(payload: _CheckpointSnapshotPayload) -> PpoDrainCheckpointSnapshot:
        snapshot = object.__new__(PpoDrainCheckpointSnapshot)
        with lock:
            rows[snapshot] = payload
        return snapshot

    def lookup(
        snapshot: PpoDrainCheckpointSnapshot,
    ) -> _CheckpointSnapshotPayload | None:
        with lock:
            return rows.get(snapshot)

    return mint, lookup


_mint_checkpoint_snapshot, _lookup_checkpoint_snapshot = (
    _make_checkpoint_snapshot_registry()
)
del _make_checkpoint_snapshot_registry


@dataclass(frozen=True)
class _RunnerFrontierProjectionPayload:
    owner_ref: weakref.ReferenceType["ActionBallFullMdpPpoDrainOwner"]
    source_receipt_ref: weakref.ReferenceType["PreOptimizerPpoBoundaryReceipt"]
    schema_version: int
    kind: str
    num_envs: int
    device_type: str
    device_index: int | None
    owner_order: tuple[str, ...]
    schema_identity: CheckpointSchemaIdentity
    next_update_index: int
    operation_sequence: int
    drain_sequence: int
    last_completed_environment_steps: int
    mutation_version_highwaters: tuple[tuple[str, int], ...]
    update_index: int
    completed_environment_steps: int


class PpoDrainRunnerFrontierProjection:
    """Opaque runner-readable projection of the latest ACKed idle frontier.

    It intentionally contains neither a checkpoint boundary root nor a
    derived digest.  The runner/R10 join must obtain its boundary fact from an
    independent owner and combine only these cloned primitive fields.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("PPO drain runner projections are owner-issued only")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PPO drain runner projections are immutable")

    def __copy__(self):
        raise TypeError("PPO drain runner projections cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("PPO drain runner projections cannot be copied")

    def __reduce__(self):
        raise TypeError("PPO drain runner projections cannot be serialized")

    @staticmethod
    def _payload(
        value: "PpoDrainRunnerFrontierProjection",
    ) -> _RunnerFrontierProjectionPayload:
        payload = _lookup_runner_frontier_projection(value)
        owner = None if payload is None else payload.owner_ref()
        if (
            payload is None
            or type(owner) is not ActionBallFullMdpPpoDrainOwner
            or owner._last_runner_frontier_projection is not value
        ):
            raise ActionBallFullMdpPpoDrainError(
                "runner frontier projection is not current and owner-issued"
            )
        return payload

    @property
    def schema_version(self) -> int:
        return self._payload(self).schema_version

    @property
    def kind(self) -> str:
        return self._payload(self).kind

    @property
    def num_envs(self) -> int:
        return self._payload(self).num_envs

    @property
    def device_type(self) -> str:
        return self._payload(self).device_type

    @property
    def device_index(self) -> int | None:
        return self._payload(self).device_index

    @property
    def owner_order(self) -> tuple[str, ...]:
        return self._payload(self).owner_order

    @property
    def schema_identity(self) -> CheckpointSchemaIdentity:
        return self._payload(self).schema_identity

    @property
    def next_update_index(self) -> int:
        return self._payload(self).next_update_index

    @property
    def operation_sequence(self) -> int:
        return self._payload(self).operation_sequence

    @property
    def drain_sequence(self) -> int:
        return self._payload(self).drain_sequence

    @property
    def last_completed_environment_steps(self) -> int:
        return self._payload(self).last_completed_environment_steps

    @property
    def mutation_version_highwaters(self) -> tuple[tuple[str, int], ...]:
        return self._payload(self).mutation_version_highwaters

    @property
    def update_index(self) -> int:
        return self._payload(self).update_index

    @property
    def completed_environment_steps(self) -> int:
        return self._payload(self).completed_environment_steps

def _make_runner_frontier_projection_registry():
    rows: weakref.WeakKeyDictionary[
        PpoDrainRunnerFrontierProjection,
        _RunnerFrontierProjectionPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        payload: _RunnerFrontierProjectionPayload,
    ) -> PpoDrainRunnerFrontierProjection:
        projection = object.__new__(PpoDrainRunnerFrontierProjection)
        with lock:
            rows[projection] = payload
        return projection

    def lookup(
        projection: PpoDrainRunnerFrontierProjection,
    ) -> _RunnerFrontierProjectionPayload | None:
        with lock:
            return rows.get(projection)

    def retire(projection: object) -> None:
        if type(projection) is not PpoDrainRunnerFrontierProjection:
            return
        with lock:
            rows.pop(projection, None)

    return mint, lookup, retire


(
    _mint_runner_frontier_projection,
    _lookup_runner_frontier_projection,
    _retire_runner_frontier_projection,
) = _make_runner_frontier_projection_registry()
del _make_runner_frontier_projection_registry


@dataclass
class _RestoreProjectionPayload:
    owner_identity: object
    content: PpoDrainCheckpointContent
    external_checkpoint_root_sha256: str
    consumed: bool = False


class _PpoDrainRestoreProjection:
    """Private one-shot result of external VerifiedCheckpoint validation."""

    __slots__ = ("__weakref__",)

    def __new__(cls):
        del cls
        raise TypeError("restore projections are externally validated only")


def _make_restore_projection_registry():
    rows: weakref.WeakKeyDictionary[
        _PpoDrainRestoreProjection,
        _RestoreProjectionPayload,
    ] = weakref.WeakKeyDictionary()
    lock = threading.RLock()

    def mint(
        *,
        owner_identity: object,
        content: PpoDrainCheckpointContent,
        external_checkpoint_root_sha256: str,
    ) -> _PpoDrainRestoreProjection:
        if type(content) is not PpoDrainCheckpointContent:
            raise ActionBallFullMdpPpoDrainError(
                "external authority projection requires exact checkpoint content"
            )
        content.validate_derived_root()
        root = _exact_sha256(
            external_checkpoint_root_sha256,
            label="external checkpoint root SHA-256",
        )
        projection = object.__new__(_PpoDrainRestoreProjection)
        with lock:
            rows[projection] = _RestoreProjectionPayload(
                owner_identity=owner_identity,
                content=content,
                external_checkpoint_root_sha256=root,
            )
        return projection

    def consume(
        projection: object,
        *,
        owner_identity: object,
    ) -> _RestoreProjectionPayload | None:
        if type(projection) is not _PpoDrainRestoreProjection:
            return None
        with lock:
            payload = rows.get(projection)
            if (
                payload is None
                or payload.consumed
                or payload.owner_identity is not owner_identity
            ):
                return None
            payload.consumed = True
            return payload

    return mint, consume


_mint_restore_projection, _consume_restore_projection = (
    _make_restore_projection_registry()
)
del _make_restore_projection_registry


_REQUIRED_FIELD_NAMES = {
    "r05_runtime": (
        "mutation_version",
        "fault_count",
        "invariant_count",
        "terminal_resolution_total",
        "policy_opportunity_total",
    ),
    "motion": (
        "mutation_version",
        "fault_count",
        "invariant_count",
        "terminal_resolution_total",
    ),
    "racket": (
        "mutation_version",
        "fault_count",
        "invariant_count",
        "terminal_resolution_total",
    ),
    "physical_ball": (
        "mutation_version",
        "fault_count",
        "invariant_count",
        "terminal_resolution_total",
        "shared_normal_retire_total",
        "physical_only_orphan_park_total",
        "shared_normal_retire_key_summary_0",
        "shared_normal_retire_key_summary_1",
    ),
    "r06_landing_outcome": (
        "mutation_version",
        "fault_count",
        "invariant_count",
        "terminal_resolution_total",
        "shared_normal_retire_total",
        "r06_only_orphan_retire_total",
        "shared_normal_retire_key_summary_0",
        "shared_normal_retire_key_summary_1",
    ),
    "r03_strike_fact": (
        "mutation_version",
        "fault_count",
        "invariant_count",
    ),
    "r07_recovery": (
        "mutation_version",
        "fault_count",
        "invariant_count",
    ),
}


DEFAULT_LEAF_SCHEMAS = tuple(
    LeafDrainSchema(
        owner_kind=owner_kind,
        fields=tuple(
            DeviceDrainFieldSpec(name=name)
            for name in _REQUIRED_FIELD_NAMES[owner_kind]
        ),
    )
    for owner_kind in OWNER_ORDER
)


@dataclass(frozen=True)
class ConservationTerm:
    """One scalar counter and its integer coefficient in a conservation side."""

    owner_kind: str
    field_name: str
    coefficient: int = 1

    def __post_init__(self) -> None:
        _exact_name(self.owner_kind, label="conservation owner kind")
        _exact_name(self.field_name, label="conservation field name")
        if type(self.coefficient) is not int or self.coefficient <= 0:
            raise ActionBallFullMdpPpoDrainError(
                "conservation coefficient must be a positive exact integer"
            )


@dataclass(frozen=True)
class ConservationRule:
    """An equality whose left and right causal-writer sets must be disjoint."""

    name: str
    left: tuple[ConservationTerm, ...]
    right: tuple[ConservationTerm, ...]

    def __post_init__(self) -> None:
        _exact_name(self.name, label="conservation rule name")
        if not self.left or not self.right:
            raise ActionBallFullMdpPpoDrainError(
                "conservation rule sides must both be non-empty"
            )
        if any(not isinstance(term, ConservationTerm) for term in self.left + self.right):
            raise ActionBallFullMdpPpoDrainError(
                "conservation rule terms have the wrong type"
            )
        left_writers = {term.owner_kind for term in self.left}
        right_writers = {term.owner_kind for term in self.right}
        if left_writers & right_writers:
            raise ActionBallFullMdpPpoDrainError(
                f"{self.name} compares a causal writer with itself"
            )


def _one_to_one_rule(name: str, right_owner: str, right_field: str) -> ConservationRule:
    return ConservationRule(
        name=name,
        left=(
            ConservationTerm(
                "r05_runtime", "terminal_resolution_total"
            ),
        ),
        right=(ConservationTerm(right_owner, right_field),),
    )


REQUIRED_CONSERVATION_RULES = (
    _one_to_one_rule(
        "r05_terminal_vs_motion_completion",
        "motion",
        "terminal_resolution_total",
    ),
    _one_to_one_rule(
        "r05_terminal_vs_racket_completion",
        "racket",
        "terminal_resolution_total",
    ),
    _one_to_one_rule(
        "r05_terminal_vs_physical_completion",
        "physical_ball",
        "terminal_resolution_total",
    ),
    _one_to_one_rule(
        "r05_terminal_vs_r06_completion",
        "r06_landing_outcome",
        "terminal_resolution_total",
    ),
    ConservationRule(
        name="physical_vs_r06_shared_normal_retire_count",
        left=(
            ConservationTerm("physical_ball", "shared_normal_retire_total"),
        ),
        right=(
            ConservationTerm(
                "r06_landing_outcome", "shared_normal_retire_total"
            ),
        ),
    ),
    ConservationRule(
        name="physical_vs_r06_shared_normal_retire_key_summary_0",
        left=(
            ConservationTerm(
                "physical_ball", "shared_normal_retire_key_summary_0"
            ),
        ),
        right=(
            ConservationTerm(
                "r06_landing_outcome",
                "shared_normal_retire_key_summary_0",
            ),
        ),
    ),
    ConservationRule(
        name="physical_vs_r06_shared_normal_retire_key_summary_1",
        left=(
            ConservationTerm(
                "physical_ball", "shared_normal_retire_key_summary_1"
            ),
        ),
        right=(
            ConservationTerm(
                "r06_landing_outcome",
                "shared_normal_retire_key_summary_1",
            ),
        ),
    ),
)


class _LeafProtocol(Protocol):
    def prepare_pre_optimizer_ppo_boundary_device_pack(self, **kwargs: object) -> object:
        ...

    def abort_pre_optimizer_ppo_boundary_device_pack(self, **kwargs: object) -> None:
        ...

    def acknowledge_pre_optimizer_ppo_boundary(self, **kwargs: object) -> None:
        ...

    def poison_pre_optimizer_ppo_boundary(self, **kwargs: object) -> None:
        ...


_PACK_TOKEN = object()
_PREPARED_TOKEN = object()
_RECEIPT_TOKEN = object()


class OpaqueLeafDevicePack:
    """Owner-minted capability with no tensor or host-materialization API."""

    __slots__ = ("__owner_kind", "__token")

    def __init__(self, *, owner_kind: str, token: object) -> None:
        if token is not _PACK_TOKEN:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device packs are authority-minted only"
            )
        self.__owner_kind = owner_kind
        self.__token = token

    @property
    def owner_kind(self) -> str:
        return self.__owner_kind

    def __repr__(self) -> str:
        return f"<{type(self).__name__} owner={self.__owner_kind!r}>"


@dataclass
class _OwnedPackState:
    pack: OpaqueLeafDevicePack
    operation_id: int
    values: torch.Tensor


class LeafDevicePackAuthority:
    """A leaf-specific mint bound to one exact owner object and device.

    The authority is passed only during the leaf's prepare call.  It accepts a
    single flat int64 device tensor of the exact schema width and returns an
    opaque pack.  The tensor remains private in this authority's live registry.
    """

    __slots__ = (
        "__owner_kind",
        "__schema",
        "__device",
        "__num_envs",
        "__leaf",
        "__drain_owner",
        "__open_operation_id",
        "__minted_pack",
        "__registry",
    )

    def __init__(
        self,
        *,
        owner_kind: str,
        schema: LeafDrainSchema,
        device: torch.device,
        num_envs: int,
        leaf: _LeafProtocol,
        drain_owner: "ActionBallFullMdpPpoDrainOwner | None" = None,
    ) -> None:
        self.__owner_kind = owner_kind
        self.__schema = schema
        self.__device = device
        self.__num_envs = num_envs
        self.__leaf = leaf
        self.__drain_owner = drain_owner
        self.__open_operation_id: int | None = None
        self.__minted_pack: OpaqueLeafDevicePack | None = None
        self.__registry: dict[int, _OwnedPackState] = {}

    @property
    def owner_kind(self) -> str:
        return self.__owner_kind

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.__schema.fields)

    @property
    def expected_width(self) -> int:
        return self.__schema.width(self.__num_envs)

    def mint_device_pack(
        self,
        *,
        leaf: object,
        values: torch.Tensor,
    ) -> OpaqueLeafDevicePack:
        """Mint one opaque pack inside the currently open prepare callback."""

        operation_id = self.__open_operation_id
        if operation_id is None:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack mint is outside its prepare window"
            )
        if leaf is not self.__leaf:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack mint came from a foreign owner"
            )
        if self.__minted_pack is not None:
            raise ActionBallFullMdpPpoDrainError(
                "leaf prepared more than one device pack"
            )
        if not isinstance(values, torch.Tensor):
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack values must be one torch.Tensor"
            )
        if values.dtype != torch.int64:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack values must have dtype torch.int64"
            )
        if values.device != self.__device:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack values are on the wrong device"
            )
        if values.ndim != 1 or values.shape[0] != self.expected_width:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack values have the wrong flat width"
            )
        pack = OpaqueLeafDevicePack(
            owner_kind=self.__owner_kind,
            token=_PACK_TOKEN,
        )
        self.__registry[id(pack)] = _OwnedPackState(
            pack=pack,
            operation_id=operation_id,
            # Freeze the leaf-owned row on device.  The leaf may keep and
            # mutate its source tensor after prepare; that must not rewrite
            # the already-minted global-boundary evidence.
            values=values.detach().clone().contiguous(),
        )
        self.__minted_pack = pack
        return pack

    def require_owned_ack(
        self,
        *,
        leaf: object,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Validate the exact live global ACK window for this leaf lane.

        This proves only coordinator ownership and call ordering: the exact
        construction-bound leaf, current pack, current receipt, exact decoded
        row, optimizer-return marker, health, and unacknowledged state.  It
        deliberately does not attest that the leaf's business facts are true.
        """

        if type(self.__drain_owner) is not ActionBallFullMdpPpoDrainOwner:
            raise ActionBallFullMdpPpoDrainError(
                "leaf ACK authority is not global-drain-owner-issued"
            )
        self.__drain_owner._require_leaf_ack_authority(
            authority=self,
            leaf=leaf,
            pack=pack,
            receipt=receipt,
            owner_row=owner_row,
        )

    def _open(self, operation_id: int) -> None:
        if self.__open_operation_id is not None:
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack authority is already open"
            )
        self.__open_operation_id = operation_id
        self.__minted_pack = None

    def _close(self) -> OpaqueLeafDevicePack | None:
        pack = self.__minted_pack
        self.__open_operation_id = None
        self.__minted_pack = None
        return pack

    def _require(
        self,
        pack: object,
        *,
        operation_id: int,
    ) -> torch.Tensor:
        if not isinstance(pack, OpaqueLeafDevicePack):
            raise ActionBallFullMdpPpoDrainError(
                "leaf returned a non-capability device pack"
            )
        state = self.__registry.get(id(pack))
        if (
            state is None
            or state.pack is not pack
            or state.operation_id != operation_id
            or pack.owner_kind != self.__owner_kind
        ):
            raise ActionBallFullMdpPpoDrainError(
                "leaf device pack is foreign, stale, or copied"
            )
        return state.values

    def _retire(self, pack: object) -> None:
        state = self.__registry.get(id(pack))
        if state is not None and state.pack is pack:
            del self.__registry[id(pack)]


class PreparedPreOptimizerPpoBoundary:
    """Opaque capability for one prepared, not-yet-transferred boundary."""

    __slots__ = ("__owner", "__operation_id", "__token")

    def __init__(self, *, owner: object, operation_id: int, token: object) -> None:
        if token is not _PREPARED_TOKEN:
            raise ActionBallFullMdpPpoDrainError(
                "prepared PPO boundaries are owner-minted only"
            )
        self.__owner = owner
        self.__operation_id = operation_id
        self.__token = token

    def _matches(self, owner: object, operation_id: int) -> bool:
        return (
            self.__token is _PREPARED_TOKEN
            and self.__owner is owner
            and self.__operation_id == operation_id
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} opaque>"


@dataclass(frozen=True)
class OwnerDrainRow:
    """Decoded immutable host row; it is telemetry, not a capability."""

    owner_kind: str
    values: tuple[tuple[str, int | tuple[int, ...]], ...]

    def scalar(self, field_name: str) -> int:
        for name, value in self.values:
            if name == field_name:
                if type(value) is not int:
                    raise ActionBallFullMdpPpoDrainError(
                        f"{self.owner_kind}.{field_name} is not scalar"
                    )
                return value
        raise ActionBallFullMdpPpoDrainError(
            f"{self.owner_kind}.{field_name} is absent"
        )


class PreOptimizerPpoBoundaryReceipt:
    """Opaque one-shot receipt produced only after the global D2H validates."""

    __slots__ = (
        "__weakref__",
        "__owner",
        "__operation_id",
        "__token",
        "__update_index",
        "__completed_environment_steps",
        "__drain_sequence",
        "__num_envs",
        "__rows",
        "__acknowledged",
    )

    def __init__(
        self,
        *,
        owner: object,
        operation_id: int,
        update_index: int,
        completed_environment_steps: int,
        drain_sequence: int,
        num_envs: int,
        rows: tuple[OwnerDrainRow, ...],
        token: object,
    ) -> None:
        if token is not _RECEIPT_TOKEN:
            raise ActionBallFullMdpPpoDrainError(
                "PPO drain receipts are owner-minted only"
            )
        self.__owner = owner
        self.__operation_id = operation_id
        self.__token = token
        self.__update_index = update_index
        self.__completed_environment_steps = completed_environment_steps
        self.__drain_sequence = drain_sequence
        self.__num_envs = num_envs
        self.__rows = rows
        self.__acknowledged = False

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def kind(self) -> str:
        return RECEIPT_KIND

    @property
    def update_index(self) -> int:
        return self.__update_index

    @property
    def completed_environment_steps(self) -> int:
        return self.__completed_environment_steps

    @property
    def drain_sequence(self) -> int:
        return self.__drain_sequence

    @property
    def num_envs(self) -> int:
        return self.__num_envs

    @property
    def owner_order(self) -> tuple[str, ...]:
        return OWNER_ORDER

    @property
    def owner_rows(self) -> tuple[OwnerDrainRow, ...]:
        return self.__rows

    @property
    def device_to_host_transfers(self) -> int:
        return 1

    @property
    def acknowledged(self) -> bool:
        return self.__acknowledged

    def _matches(self, owner: object, operation_id: int) -> bool:
        return (
            self.__token is _RECEIPT_TOKEN
            and self.__owner is owner
            and self.__operation_id == operation_id
        )

    def _mark_acknowledged(self) -> None:
        self.__acknowledged = True

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} update={self.__update_index} "
            f"sequence={self.__drain_sequence} opaque>"
        )


@dataclass
class _ActiveDrain:
    operation_id: int
    update_index: int
    completed_environment_steps: int
    prepared: PreparedPreOptimizerPpoBoundary
    packs: tuple[OpaqueLeafDevicePack, ...]
    transfer_started: bool = False
    receipt: PreOptimizerPpoBoundaryReceipt | None = None
    optimizer_returned: bool = False


def _validate_schemas(
    schemas: Sequence[LeafDrainSchema],
) -> tuple[LeafDrainSchema, ...]:
    rows = tuple(schemas)
    if tuple(schema.owner_kind for schema in rows) != OWNER_ORDER:
        raise ActionBallFullMdpPpoDrainError(
            "leaf schemas must follow the frozen seven-owner order"
        )
    for schema in rows:
        required = _REQUIRED_FIELD_NAMES[schema.owner_kind]
        actual = tuple(field.name for field in schema.fields)
        if actual[: len(required)] != required:
            raise ActionBallFullMdpPpoDrainError(
                f"{schema.owner_kind} schema must retain its required prefix"
            )
        required_fields = schema.fields[: len(required)]
        if any(field.cardinality != "scalar" for field in required_fields):
            raise ActionBallFullMdpPpoDrainError(
                f"{schema.owner_kind} required fields must remain scalar"
            )
    return rows


def _validate_rules(
    rules: Sequence[ConservationRule],
    schemas: tuple[LeafDrainSchema, ...],
) -> tuple[ConservationRule, ...]:
    rows = tuple(rules)
    if rows[: len(REQUIRED_CONSERVATION_RULES)] != REQUIRED_CONSERVATION_RULES:
        raise ActionBallFullMdpPpoDrainError(
            "conservation rules must retain the required independent-writer prefix"
        )
    schema_by_owner = {schema.owner_kind: schema for schema in schemas}
    for rule in rows:
        if not isinstance(rule, ConservationRule):
            raise ActionBallFullMdpPpoDrainError(
                "conservation rules have the wrong type"
            )
        for term in rule.left + rule.right:
            schema = schema_by_owner.get(term.owner_kind)
            if schema is None:
                raise ActionBallFullMdpPpoDrainError(
                    f"{rule.name} names an unknown owner"
                )
            field = next(
                (value for value in schema.fields if value.name == term.field_name),
                None,
            )
            if field is None or field.cardinality != "scalar":
                raise ActionBallFullMdpPpoDrainError(
                    f"{rule.name} must compare present scalar counters"
                )
    return rows


class ActionBallFullMdpPpoDrainOwner:
    """Seven-owner, single-transfer, pre-optimizer boundary coordinator."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        leaves: Mapping[str, _LeafProtocol],
        leaf_schemas: Sequence[LeafDrainSchema] | None = None,
        conservation_rules: Sequence[ConservationRule] = REQUIRED_CONSERVATION_RULES,
        initial_update_index: int = 0,
        diagnostic_allow_minimal_schemas: bool = False,
        checkpoint_boundary_validator: Callable[[object], str] | None = None,
        checkpoint_restore_validator_factory: Callable[
            [Callable[..., object]], Callable[[object], object]
        ]
        | None = None,
    ) -> None:
        self.num_envs = _exact_int(num_envs, label="num_envs", minimum=1)
        self.device = torch.device(device)
        self._next_update_index = _exact_int(
            initial_update_index,
            label="initial_update_index",
        )
        if not isinstance(leaves, Mapping) or set(leaves) != set(OWNER_ORDER):
            raise ActionBallFullMdpPpoDrainError(
                "leaves must contain exactly the frozen seven owners"
            )
        if type(diagnostic_allow_minimal_schemas) is not bool:
            raise ActionBallFullMdpPpoDrainError(
                "diagnostic_allow_minimal_schemas must be an exact bool"
            )
        selected_schemas = (
            DEFAULT_LEAF_SCHEMAS if leaf_schemas is None else tuple(leaf_schemas)
        )
        if (
            selected_schemas == DEFAULT_LEAF_SCHEMAS
            and not diagnostic_allow_minimal_schemas
        ):
            raise ActionBallFullMdpPpoDrainError(
                "minimal leaf schemas require explicit diagnostic opt-in"
            )
        self._schemas = _validate_schemas(selected_schemas)
        self._rules = _validate_rules(conservation_rules, self._schemas)
        self._leaves = tuple(leaves[name] for name in OWNER_ORDER)
        if len({id(leaf) for leaf in self._leaves}) != len(OWNER_ORDER):
            raise ActionBallFullMdpPpoDrainError(
                "each owner kind must bind a distinct causal leaf object"
            )
        for owner_kind, leaf in zip(OWNER_ORDER, self._leaves):
            missing = tuple(
                name
                for name in (
                    _PREPARE_METHOD,
                    _ABORT_METHOD,
                    _ACK_METHOD,
                    _POISON_METHOD,
                )
                if not callable(getattr(leaf, name, None))
            )
            if missing:
                raise ActionBallFullMdpPpoDrainError(
                    f"{owner_kind} leaf is missing methods {missing}"
                )
        self._authorities = tuple(
            LeafDevicePackAuthority(
                owner_kind=owner_kind,
                schema=schema,
                device=self.device,
                num_envs=self.num_envs,
                leaf=leaf,
                drain_owner=self,
            )
            for owner_kind, schema, leaf in zip(
                OWNER_ORDER, self._schemas, self._leaves
            )
        )
        self._operation_sequence = 0
        self._drain_sequence = 0
        self._last_completed_environment_steps = -1
        self._last_mutation_versions: dict[str, int] | None = None
        self._latest_acknowledged_receipt: (
            PreOptimizerPpoBoundaryReceipt | None
        ) = None
        self._last_runner_frontier_projection: (
            PpoDrainRunnerFrontierProjection | None
        ) = None
        self._active: _ActiveDrain | None = None
        self._poisoned = False
        self._poison_reason: str | None = None
        self._poison_failures: tuple[tuple[str, str], ...] = ()
        self._construction_identity_gate_open = True
        self._exact_leaf_bindings_joined = False
        self._owner_identity = object()
        self._last_checkpoint_snapshot: PpoDrainCheckpointSnapshot | None = None
        if checkpoint_boundary_validator is not None and not callable(
            checkpoint_boundary_validator
        ):
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint_boundary_validator must be callable"
            )
        self._checkpoint_boundary_validator = checkpoint_boundary_validator
        self._checkpoint_restore_validator: Callable[[object], object] | None = None
        self._checkpoint_projection_mint_closed = False
        if checkpoint_restore_validator_factory is not None:
            if not callable(checkpoint_restore_validator_factory):
                raise ActionBallFullMdpPpoDrainError(
                    "checkpoint_restore_validator_factory must be callable"
                )
            def mint_owner_bound_projection(**kwargs: object) -> object:
                if self._checkpoint_projection_mint_closed:
                    raise ActionBallFullMdpPpoDrainError(
                        "restore projection mint is closed after one validation"
                    )
                self._checkpoint_projection_mint_closed = True
                return _mint_restore_projection(
                    owner_identity=self._owner_identity,
                    **kwargs,
                )

            validator = checkpoint_restore_validator_factory(
                mint_owner_bound_projection
            )
            if not callable(validator):
                raise ActionBallFullMdpPpoDrainError(
                    "checkpoint restore validator factory must return callable"
                )
            self._checkpoint_restore_validator = validator
        self._checkpoint_restore_open = True

    def _close_construction_identity_gate(self) -> None:
        self._construction_identity_gate_open = False

    def _retire_last_runner_frontier_projection(self) -> None:
        projection = self._last_runner_frontier_projection
        if projection is not None:
            _retire_runner_frontier_projection(projection)
            self._last_runner_frontier_projection = None

    def _boundary_root(self, boundary: object) -> str:
        validator = self._checkpoint_boundary_validator
        if validator is None:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint boundary validator is not construction-bound"
            )
        return _exact_sha256(
            validator(boundary),
            label="validated checkpoint boundary SHA-256",
        )

    def _checkpoint_content(self, *, boundary_root: str) -> PpoDrainCheckpointContent:
        highwaters = tuple(
            (
                owner_kind,
                None
                if self._last_mutation_versions is None
                else self._last_mutation_versions[owner_kind],
            )
            for owner_kind in OWNER_ORDER
        )
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "num_envs": self.num_envs,
            "device_type": self.device.type,
            "device_index": self.device.index,
            "owner_order": OWNER_ORDER,
            "schema_identity": _checkpoint_schema_identity(self._schemas),
            "checkpoint_boundary_sha256": boundary_root,
            "next_update_index": self._next_update_index,
            "operation_sequence": self._operation_sequence,
            "drain_sequence": self._drain_sequence,
            "last_completed_environment_steps": (
                self._last_completed_environment_steps
            ),
            "mutation_version_highwaters": highwaters,
        }
        return PpoDrainCheckpointContent(
            **payload,
            checkpoint_frontier_sha256=_canonical_sha256(payload),
        )

    def snapshot_for_checkpoint_boundary(
        self,
        boundary: object,
    ) -> PpoDrainCheckpointSnapshot:
        """Mint one exact idle frontier for inclusion in an R10 envelope."""

        self._close_construction_identity_gate()
        self._require_operable()
        if self._active is not None:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint snapshot requires no active PPO drain boundary"
            )
        boundary_root = self._boundary_root(boundary)
        retained = self._last_checkpoint_snapshot
        if retained is not None:
            retained_payload = _lookup_checkpoint_snapshot(retained)
            if (
                retained_payload is not None
                and retained_payload.owner_identity is self._owner_identity
                and retained_payload.boundary is boundary
                and retained_payload.content.checkpoint_boundary_sha256
                == boundary_root
                and retained_payload.content
                == self._checkpoint_content(boundary_root=boundary_root)
            ):
                return retained
        payload = _CheckpointSnapshotPayload(
            owner_identity=self._owner_identity,
            boundary=boundary,
            content=self._checkpoint_content(boundary_root=boundary_root),
        )
        snapshot = _mint_checkpoint_snapshot(payload)
        self._last_checkpoint_snapshot = snapshot
        return snapshot

    def require_owned_checkpoint_snapshot(
        self,
        boundary: object,
        snapshot: object,
    ) -> PpoDrainCheckpointSnapshot:
        """Return only this owner's current exact boundary snapshot identity."""

        self._close_construction_identity_gate()
        self._require_operable()
        payload = (
            _lookup_checkpoint_snapshot(snapshot)
            if type(snapshot) is PpoDrainCheckpointSnapshot
            else None
        )
        boundary_root = self._boundary_root(boundary)
        if (
            payload is None
            or snapshot is not self._last_checkpoint_snapshot
            or payload.owner_identity is not self._owner_identity
            or payload.boundary is not boundary
            or payload.content.checkpoint_boundary_sha256 != boundary_root
            or payload.content
            != self._checkpoint_content(boundary_root=boundary_root)
        ):
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint snapshot is stale, foreign, or boundary-mismatched"
            )
        return snapshot

    def restore_checkpoint(self, external_checkpoint: object) -> None:
        """Restore a fresh owner only through its bound R10 validation factory.

        The external validator must validate a real VerifiedCheckpoint envelope
        and use the one construction-time projection mint passed to its factory.
        Portable ``PpoDrainCheckpointContent`` is intentionally rejected here.
        """

        if not self._checkpoint_restore_open:
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint restore is construction-only and already closed"
            )
        self._checkpoint_restore_open = False
        if (
            not self._construction_identity_gate_open
            or self._exact_leaf_bindings_joined
            or self._active is not None
            or self._poisoned
            or self._operation_sequence != 0
            or self._drain_sequence != 0
            or self._last_completed_environment_steps != -1
            or self._last_mutation_versions is not None
        ):
            raise ActionBallFullMdpPpoDrainError(
                "checkpoint restore requires a fresh idle nonpoisoned owner"
            )
        validator = self._checkpoint_restore_validator
        if validator is None:
            raise ActionBallFullMdpPpoDrainError(
                "external checkpoint restore authority is not construction-bound"
            )
        try:
            projection = validator(external_checkpoint)
        except BaseException as exc:
            raise ActionBallFullMdpPpoDrainError(
                f"external checkpoint authority rejected restore: {type(exc).__name__}: {exc}"
            ) from exc
        projected = _consume_restore_projection(
            projection,
            owner_identity=self._owner_identity,
        )
        if projected is None:
            raise ActionBallFullMdpPpoDrainError(
                "external checkpoint validator returned a foreign or reused projection"
            )
        content = projected.content
        content.validate_derived_root()
        if (
            content.schema_version != SCHEMA_VERSION
            or content.kind != CHECKPOINT_KIND
            or content.num_envs != self.num_envs
            or content.device_type != self.device.type
            or content.device_index != self.device.index
            or content.owner_order != OWNER_ORDER
            or content.schema_identity != _checkpoint_schema_identity(self._schemas)
        ):
            raise ActionBallFullMdpPpoDrainError(
                "restored checkpoint schema, device, or owner inventory differs"
            )
        if content.operation_sequence < content.drain_sequence:
            raise ActionBallFullMdpPpoDrainError(
                "restored operation sequence precedes drain sequence"
            )
        if content.next_update_index < content.drain_sequence:
            raise ActionBallFullMdpPpoDrainError(
                "restored next update index precedes drain sequence"
            )
        if (
            content.drain_sequence == 0
            and content.last_completed_environment_steps != -1
        ):
            raise ActionBallFullMdpPpoDrainError(
                "restored empty drain frontier has completed environment steps"
            )
        highwater_values = tuple(
            value for _owner_kind, value in content.mutation_version_highwaters
        )
        if any(value is None for value in highwater_values):
            if any(value is not None for value in highwater_values):
                raise ActionBallFullMdpPpoDrainError(
                    "restored mutation highwaters are partially absent"
                )
            if (
                content.drain_sequence != 0
                or content.last_completed_environment_steps != -1
            ):
                raise ActionBallFullMdpPpoDrainError(
                    "restored empty mutation highwaters disagree with frontier"
                )
            restored_highwaters = None
        else:
            if (
                content.drain_sequence == 0
                or content.last_completed_environment_steps < 0
            ):
                raise ActionBallFullMdpPpoDrainError(
                    "restored mutation highwaters disagree with drain frontier"
                )
            restored_highwaters = {
                owner_kind: value
                for owner_kind, value in content.mutation_version_highwaters
                if value is not None
            }
        self._next_update_index = content.next_update_index
        self._operation_sequence = content.operation_sequence
        self._drain_sequence = content.drain_sequence
        self._last_completed_environment_steps = (
            content.last_completed_environment_steps
        )
        self._last_mutation_versions = restored_highwaters
        # Deliberately leave the exact leaf binding gate open.  Restore of
        # chronology does not prove the new process bound the intended leaves.

    def require_exact_leaf_bindings(
        self,
        expected_leaves: Mapping[str, object],
    ) -> None:
        """Join exact construction identities once, before any runtime read.

        This is deliberately not a portable receipt or a runtime capability.
        The first attempted join closes the construction gate whether it
        succeeds or fails; every public read/runtime operation also closes it.
        A top owner therefore cannot inspect this coordinator and later retry
        construction with substituted leaves.
        """

        if not self._construction_identity_gate_open:
            raise ActionBallFullMdpPpoDrainError(
                "exact leaf binding join is closed after construction"
            )
        self._close_construction_identity_gate()
        if not isinstance(expected_leaves, Mapping):
            raise ActionBallFullMdpPpoDrainError(
                "expected leaves must be an ordered mapping"
            )
        if tuple(expected_leaves.keys()) != OWNER_ORDER:
            raise ActionBallFullMdpPpoDrainError(
                "expected leaf keys must equal the frozen owner order"
            )
        for owner_kind, actual_leaf in zip(OWNER_ORDER, self._leaves):
            if expected_leaves[owner_kind] is not actual_leaf:
                raise ActionBallFullMdpPpoDrainError(
                    f"{owner_kind} expected leaf identity differs"
                )
        self._exact_leaf_bindings_joined = True

    @property
    def poisoned(self) -> bool:
        self._close_construction_identity_gate()
        return self._poisoned

    @property
    def poison_reason(self) -> str | None:
        self._close_construction_identity_gate()
        return self._poison_reason

    @property
    def poison_failures(self) -> tuple[tuple[str, str], ...]:
        self._close_construction_identity_gate()
        return self._poison_failures

    @property
    def next_update_index(self) -> int:
        self._close_construction_identity_gate()
        return self._next_update_index

    def _require_operable(self) -> None:
        if self._poisoned:
            raise ActionBallFullMdpPpoDrainPoisonedError(
                "global PPO drain is poisoned and requires cold replacement"
            )
        if not self._exact_leaf_bindings_joined:
            raise ActionBallFullMdpPpoDrainError(
                "exact seven-leaf construction binding join is required"
            )

    def _require_leaf_ack_authority(
        self,
        *,
        authority: object,
        leaf: object,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Internal exact-identity join used by a leaf inside its ACK call."""

        self._require_operable()
        active = self._active
        authority_index = next(
            (
                index
                for index, retained in enumerate(self._authorities)
                if retained is authority
            ),
            None,
        )
        if (
            active is None
            or authority_index is None
            or active.receipt is None
            or receipt is not active.receipt
            or type(receipt) is not PreOptimizerPpoBoundaryReceipt
            or not receipt._matches(self, active.operation_id)
            or receipt.acknowledged
            or not active.optimizer_returned
            or leaf is not self._leaves[authority_index]
            or pack is not active.packs[authority_index]
            or owner_row is not receipt.owner_rows[authority_index]
            or owner_row.owner_kind != OWNER_ORDER[authority_index]
        ):
            raise ActionBallFullMdpPpoDrainError(
                "leaf ACK authority is foreign, stale, out of window, or lane-swapped"
            )
        self._authorities[authority_index]._require(
            pack,
            operation_id=active.operation_id,
        )

    def require_owned_runner_frontier_projection(
        self,
        global_receipt: object,
    ) -> PpoDrainRunnerFrontierProjection:
        """Project the latest exact ACKed receipt into runner-owned primitives.

        The projection is available only while this owner is healthy and idle.
        It does not manufacture an R10 boundary root or checkpoint digest.
        """

        self._close_construction_identity_gate()
        self._require_operable()
        receipt = self._latest_acknowledged_receipt
        if (
            self._active is not None
            or receipt is None
            or global_receipt is not receipt
            or type(global_receipt) is not PreOptimizerPpoBoundaryReceipt
            or not receipt.acknowledged
            or receipt.update_index + 1 != self._next_update_index
            or receipt.completed_environment_steps
            != self._last_completed_environment_steps
            or receipt.drain_sequence != self._drain_sequence
            or self._last_mutation_versions is None
        ):
            raise ActionBallFullMdpPpoDrainError(
                "runner frontier requires the latest exact ACKed receipt and an idle owner"
            )
        highwaters = tuple(
            (owner_kind, self._last_mutation_versions[owner_kind])
            for owner_kind in OWNER_ORDER
        )
        payload = _RunnerFrontierProjectionPayload(
            owner_ref=weakref.ref(self),
            source_receipt_ref=weakref.ref(receipt),
            schema_version=SCHEMA_VERSION,
            kind=CHECKPOINT_KIND,
            num_envs=self.num_envs,
            device_type=self.device.type,
            device_index=self.device.index,
            owner_order=OWNER_ORDER,
            schema_identity=_checkpoint_schema_identity(self._schemas),
            next_update_index=self._next_update_index,
            operation_sequence=self._operation_sequence,
            drain_sequence=self._drain_sequence,
            last_completed_environment_steps=(
                self._last_completed_environment_steps
            ),
            mutation_version_highwaters=highwaters,
            update_index=receipt.update_index,
            completed_environment_steps=receipt.completed_environment_steps,
        )
        retained = self._last_runner_frontier_projection
        if retained is not None:
            retained_payload = _lookup_runner_frontier_projection(retained)
            if (
                retained_payload is not None
                and retained_payload.owner_ref() is self
                and retained_payload.source_receipt_ref() is receipt
                and retained_payload.schema_version == payload.schema_version
                and retained_payload.kind == payload.kind
                and retained_payload.num_envs == payload.num_envs
                and retained_payload.device_type == payload.device_type
                and retained_payload.device_index == payload.device_index
                and retained_payload.owner_order == payload.owner_order
                and retained_payload.schema_identity == payload.schema_identity
                and retained_payload.next_update_index == payload.next_update_index
                and retained_payload.operation_sequence == payload.operation_sequence
                and retained_payload.drain_sequence == payload.drain_sequence
                and retained_payload.last_completed_environment_steps
                == payload.last_completed_environment_steps
                and retained_payload.mutation_version_highwaters
                == payload.mutation_version_highwaters
                and retained_payload.update_index == payload.update_index
                and retained_payload.completed_environment_steps
                == payload.completed_environment_steps
            ):
                return retained
            self._retire_last_runner_frontier_projection()
        projection = _mint_runner_frontier_projection(payload)
        self._last_runner_frontier_projection = projection
        return projection

    def _retire_packs(self, active: _ActiveDrain) -> None:
        for authority, pack in zip(self._authorities, active.packs):
            authority._retire(pack)

    def _broadcast_poison(self, reason: str) -> None:
        self._retire_last_runner_frontier_projection()
        if self._poison_reason is None:
            self._poison_reason = reason
        self._poisoned = True
        failures: list[tuple[str, str]] = list(self._poison_failures)
        for owner_kind, leaf in zip(OWNER_ORDER, self._leaves):
            try:
                getattr(leaf, _POISON_METHOD)(reason=self._poison_reason)
            except BaseException as exc:  # poison broadcast must reach later owners
                failures.append((owner_kind, f"{type(exc).__name__}: {exc}"))
        self._poison_failures = tuple(failures)

    def _abort_pack_sequence(
        self,
        packs: Sequence[OpaqueLeafDevicePack],
    ) -> tuple[tuple[str, str], ...]:
        failures: list[tuple[str, str]] = []
        count = len(packs)
        for index in range(count - 1, -1, -1):
            owner_kind = OWNER_ORDER[index]
            try:
                getattr(self._leaves[index], _ABORT_METHOD)(pack=packs[index])
            except BaseException as exc:
                failures.append((owner_kind, f"{type(exc).__name__}: {exc}"))
            finally:
                self._authorities[index]._retire(packs[index])
        return tuple(failures)

    def prepare_pre_optimizer_ppo_boundary(
        self,
        *,
        update_index: int,
        completed_environment_steps: int,
    ) -> PreparedPreOptimizerPpoBoundary:
        """Prepare all seven device rows without performing a host transfer.

        ``completed_environment_steps`` is the cumulative number of completed
        per-environment transition rows.  A normal PPO update therefore adds
        ``num_envs * num_steps_per_env``; it is not a loop-local horizon and
        must be restored from checkpoint rather than reset on resume.
        """

        self._close_construction_identity_gate()
        self._require_operable()
        update = _exact_int(update_index, label="update_index")
        completed = _exact_int(
            completed_environment_steps,
            label="completed_environment_steps",
        )
        if self._active is not None:
            raise ActionBallFullMdpPpoDrainError(
                "a PPO drain boundary is already active"
            )
        if update != self._next_update_index:
            raise ActionBallFullMdpPpoDrainError(
                "update_index must equal the next contiguous update"
            )
        if completed <= self._last_completed_environment_steps:
            raise ActionBallFullMdpPpoDrainError(
                "completed_environment_steps must strictly advance"
            )

        self._retire_last_runner_frontier_projection()
        self._operation_sequence += 1
        operation_id = self._operation_sequence
        packs: list[OpaqueLeafDevicePack] = []
        try:
            for owner_kind, leaf, authority in zip(
                OWNER_ORDER, self._leaves, self._authorities
            ):
                authority._open(operation_id)
                returned: object | None = None
                prepare_error: BaseException | None = None
                try:
                    with _PrepareNoHostObservationMode():
                        returned = getattr(leaf, _PREPARE_METHOD)(
                            authority=authority,
                            update_index=update,
                            completed_environment_steps=completed,
                        )
                except BaseException as exc:
                    prepare_error = exc
                minted = authority._close()
                if minted is not None:
                    packs.append(minted)
                if prepare_error is not None:
                    if minted is None:
                        raise ActionBallFullMdpPpoDrainPoisonedError(
                            f"{owner_kind} failed before minting an abort capability"
                        ) from prepare_error
                    raise prepare_error
                if minted is None or returned is not minted:
                    raise ActionBallFullMdpPpoDrainError(
                        f"{owner_kind} did not return its exact authority-minted pack"
                    )
                authority._require(minted, operation_id=operation_id)
        except BaseException as exc:
            failures = self._abort_pack_sequence(packs)
            unabortable_failure = isinstance(
                exc, ActionBallFullMdpPpoDrainPoisonedError
            )
            if unabortable_failure:
                reason = str(exc)
                if failures:
                    reason += "; prefix abort failures: " + "; ".join(
                        f"{name}={value}" for name, value in failures
                    )
                self._broadcast_poison(reason)
                raise ActionBallFullMdpPpoDrainPrepareError(
                    reason,
                    retry_permitted=False,
                ) from exc
            if failures:
                reason = (
                    "pre-transfer prepare abort failed: "
                    + "; ".join(f"{name}={value}" for name, value in failures)
                )
                self._broadcast_poison(reason)
                raise ActionBallFullMdpPpoDrainPrepareError(
                    reason,
                    retry_permitted=False,
                ) from exc
            raise ActionBallFullMdpPpoDrainPrepareError(
                f"pre-transfer leaf prepare failed: {type(exc).__name__}: {exc}",
                retry_permitted=True,
            ) from exc

        prepared = PreparedPreOptimizerPpoBoundary(
            owner=self,
            operation_id=operation_id,
            token=_PREPARED_TOKEN,
        )
        self._active = _ActiveDrain(
            operation_id=operation_id,
            update_index=update,
            completed_environment_steps=completed,
            prepared=prepared,
            packs=tuple(packs),
        )
        return prepared

    def _require_prepared(
        self,
        prepared: object,
    ) -> _ActiveDrain:
        self._require_operable()
        active = self._active
        if (
            active is None
            or prepared is not active.prepared
            or not isinstance(prepared, PreparedPreOptimizerPpoBoundary)
            or not prepared._matches(self, active.operation_id)
        ):
            if active is not None and active.transfer_started:
                reason = (
                    "foreign or copied prepared capability was presented "
                    "after the global PPO drain transfer started"
                )
                self._broadcast_poison(reason)
                self._retire_packs(active)
                self._active = None
                raise ActionBallFullMdpPpoDrainPoisonedError(reason)
            raise ActionBallFullMdpPpoDrainError(
                "prepared PPO boundary is foreign, stale, or copied"
            )
        return active

    def abort_pre_optimizer_ppo_boundary(
        self,
        prepared: PreparedPreOptimizerPpoBoundary,
    ) -> None:
        """Abort a clean prepare; no abort is allowed after transfer starts."""

        self._close_construction_identity_gate()
        active = self._require_prepared(prepared)
        if active.transfer_started:
            self._broadcast_poison(
                "attempted to abort a PPO drain after transfer started"
            )
            self._retire_packs(active)
            self._active = None
            raise ActionBallFullMdpPpoDrainPoisonedError(
                self._poison_reason or "post-transfer abort poisoned"
            )
        failures = self._abort_pack_sequence(active.packs)
        self._active = None
        if failures:
            reason = "pre-transfer abort failed: " + "; ".join(
                f"{name}={value}" for name, value in failures
            )
            self._broadcast_poison(reason)
            raise ActionBallFullMdpPpoDrainPrepareError(
                reason,
                retry_permitted=False,
            )

    def _decode_rows(self, host_values: object) -> tuple[OwnerDrainRow, ...]:
        if not isinstance(host_values, list) or any(
            type(value) is not int for value in host_values
        ):
            raise ActionBallFullMdpPpoDrainError(
                "global PPO drain host payload is not an exact integer list"
            )
        expected = sum(schema.width(self.num_envs) for schema in self._schemas)
        if len(host_values) != expected:
            raise ActionBallFullMdpPpoDrainError(
                "global PPO drain host payload width differs"
            )
        cursor = 0
        rows: list[OwnerDrainRow] = []
        for schema in self._schemas:
            values: list[tuple[str, int | tuple[int, ...]]] = []
            for field in schema.fields:
                width = field.width(self.num_envs)
                selected = host_values[cursor : cursor + width]
                cursor += width
                if any(value < field.minimum for value in selected):
                    raise ActionBallFullMdpPpoDrainError(
                        f"{schema.owner_kind}.{field.name} is below its minimum"
                    )
                decoded: int | tuple[int, ...]
                decoded = (
                    selected[0]
                    if field.cardinality == "scalar"
                    else tuple(selected)
                )
                values.append((field.name, decoded))
            rows.append(
                OwnerDrainRow(
                    owner_kind=schema.owner_kind,
                    values=tuple(values),
                )
            )
        return tuple(rows)

    def _validate_decoded_rows(
        self,
        rows: tuple[OwnerDrainRow, ...],
    ) -> None:
        row_by_owner = {row.owner_kind: row for row in rows}
        mutation_versions = {
            row.owner_kind: row.scalar("mutation_version") for row in rows
        }
        for row in rows:
            if row.scalar("fault_count") != 0:
                raise ActionBallFullMdpPpoDrainError(
                    f"{row.owner_kind} reported a device fault"
                )
            if row.scalar("invariant_count") != 0:
                raise ActionBallFullMdpPpoDrainError(
                    f"{row.owner_kind} reported an invariant failure"
                )
        if self._last_mutation_versions is not None:
            for owner_kind, version in mutation_versions.items():
                if version < self._last_mutation_versions[owner_kind]:
                    raise ActionBallFullMdpPpoDrainError(
                        f"{owner_kind} mutation_version regressed"
                    )
        for rule in self._rules:
            left = sum(
                term.coefficient
                * row_by_owner[term.owner_kind].scalar(term.field_name)
                for term in rule.left
            )
            right = sum(
                term.coefficient
                * row_by_owner[term.owner_kind].scalar(term.field_name)
                for term in rule.right
            )
            if left != right:
                raise ActionBallFullMdpPpoDrainError(
                    f"conservation {rule.name} differs: left={left}, right={right}"
                )

    def transfer_decode_pre_optimizer_ppo_boundary(
        self,
        prepared: PreparedPreOptimizerPpoBoundary,
    ) -> PreOptimizerPpoBoundaryReceipt:
        """Perform the sole blocking D2H, decode, and validate conservation."""

        self._close_construction_identity_gate()
        active = self._require_prepared(prepared)
        if active.transfer_started:
            self._broadcast_poison(
                "duplicate global PPO drain transfer attempt"
            )
            self._retire_packs(active)
            self._active = None
            raise ActionBallFullMdpPpoDrainPoisonedError(
                self._poison_reason or "duplicate transfer poisoned"
            )
        try:
            tensors = tuple(
                authority._require(pack, operation_id=active.operation_id)
                for authority, pack in zip(self._authorities, active.packs)
            )
            packed = torch.cat(tensors, dim=0).contiguous()
        except BaseException as exc:
            failures = self._abort_pack_sequence(active.packs)
            self._active = None
            if failures:
                reason = "pre-transfer packing abort failed: " + "; ".join(
                    f"{name}={value}" for name, value in failures
                )
                self._broadcast_poison(reason)
                raise ActionBallFullMdpPpoDrainPrepareError(
                    reason,
                    retry_permitted=False,
                ) from exc
            raise ActionBallFullMdpPpoDrainPrepareError(
                f"pre-transfer packing failed: {type(exc).__name__}: {exc}",
                retry_permitted=True,
            ) from exc

        active.transfer_started = True
        try:
            # CUDA->CPU is blocking by default.  This is the sole runtime D2H.
            host_values = packed.to(device="cpu").tolist()
            rows = self._decode_rows(host_values)
            self._validate_decoded_rows(rows)
            receipt = PreOptimizerPpoBoundaryReceipt(
                owner=self,
                operation_id=active.operation_id,
                update_index=active.update_index,
                completed_environment_steps=active.completed_environment_steps,
                drain_sequence=self._drain_sequence + 1,
                num_envs=self.num_envs,
                rows=rows,
                token=_RECEIPT_TOKEN,
            )
        except BaseException as exc:
            reason = (
                "global PPO drain failed after transfer started: "
                f"{type(exc).__name__}: {exc}"
            )
            self._broadcast_poison(reason)
            self._retire_packs(active)
            self._active = None
            raise ActionBallFullMdpPpoDrainPoisonedError(reason) from exc

        active.receipt = receipt
        return receipt

    def _require_receipt(
        self,
        receipt: object,
    ) -> tuple[_ActiveDrain, PreOptimizerPpoBoundaryReceipt]:
        self._require_operable()
        active = self._active
        if (
            active is None
            or active.receipt is None
            or receipt is not active.receipt
            or not isinstance(receipt, PreOptimizerPpoBoundaryReceipt)
            or not receipt._matches(self, active.operation_id)
        ):
            if active is not None and active.transfer_started:
                reason = (
                    "foreign or caller-assembled receipt was presented "
                    "after the global PPO drain transfer started"
                )
                self._broadcast_poison(reason)
                self._retire_packs(active)
                self._active = None
                raise ActionBallFullMdpPpoDrainPoisonedError(reason)
            raise ActionBallFullMdpPpoDrainError(
                "PPO drain receipt is foreign, stale, copied, or not transferred"
            )
        return active, receipt

    def mark_optimizer_returned(
        self,
        receipt: PreOptimizerPpoBoundaryReceipt,
    ) -> None:
        """Record that runner's optimizer call returned before callbacks."""

        self._close_construction_identity_gate()
        active, _owned = self._require_receipt(receipt)
        if active.optimizer_returned:
            reason = "duplicate optimizer-return acknowledgement"
            self._broadcast_poison(reason)
            self._retire_packs(active)
            self._active = None
            raise ActionBallFullMdpPpoDrainPoisonedError(reason)
        active.optimizer_returned = True

    def acknowledge_post_update(
        self,
        receipt: PreOptimizerPpoBoundaryReceipt,
    ) -> None:
        """Release leaf leases after optimizer return and post-update callbacks."""

        self._close_construction_identity_gate()
        active, owned = self._require_receipt(receipt)
        if not active.optimizer_returned:
            reason = "post-update acknowledgement preceded optimizer return"
            self._broadcast_poison(reason)
            self._retire_packs(active)
            self._active = None
            raise ActionBallFullMdpPpoDrainPoisonedError(reason)
        rows = {row.owner_kind: row for row in owned.owner_rows}
        try:
            for owner_kind, leaf, pack in zip(
                OWNER_ORDER, self._leaves, active.packs
            ):
                getattr(leaf, _ACK_METHOD)(
                    pack=pack,
                    receipt=owned,
                    owner_row=rows[owner_kind],
                )
        except BaseException as exc:
            reason = (
                "optimizer acknowledgement was partial: "
                f"{type(exc).__name__}: {exc}"
            )
            self._broadcast_poison(reason)
            self._retire_packs(active)
            self._active = None
            raise ActionBallFullMdpPpoDrainPoisonedError(reason) from exc

        self._retire_packs(active)
        owned._mark_acknowledged()
        self._drain_sequence = owned.drain_sequence
        self._next_update_index += 1
        self._last_completed_environment_steps = (
            active.completed_environment_steps
        )
        self._last_mutation_versions = {
            row.owner_kind: row.scalar("mutation_version")
            for row in owned.owner_rows
        }
        self._active = None
        self._latest_acknowledged_receipt = owned
        self._retire_last_runner_frontier_projection()

    def poison_optimizer_failure(
        self,
        receipt: PreOptimizerPpoBoundaryReceipt,
        *,
        reason: str,
    ) -> None:
        """Fail-stop after an optimizer exception; this path is never retryable."""

        self._close_construction_identity_gate()
        active, _owned = self._require_receipt(receipt)
        message = (
            reason
            if type(reason) is str and bool(reason) and reason.isascii()
            else "unspecified optimizer failure"
        )
        self._broadcast_poison(f"optimizer failed after global drain: {message}")
        self._retire_packs(active)
        self._active = None


__all__ = [
    "ActionBallFullMdpPpoDrainError",
    "ActionBallFullMdpPpoDrainOwner",
    "ActionBallFullMdpPpoDrainPoisonedError",
    "ActionBallFullMdpPpoDrainPrepareError",
    "ConservationRule",
    "ConservationTerm",
    "DEFAULT_LEAF_SCHEMAS",
    "DeviceDrainFieldSpec",
    "LeafDevicePackAuthority",
    "LeafDrainSchema",
    "OpaqueLeafDevicePack",
    "OWNER_ORDER",
    "OwnerDrainRow",
    "PpoDrainCheckpointContent",
    "PpoDrainCheckpointSnapshot",
    "PpoDrainRunnerFrontierProjection",
    "PreOptimizerPpoBoundaryReceipt",
    "PreparedPreOptimizerPpoBoundary",
    "REQUIRED_CONSERVATION_RULES",
]
