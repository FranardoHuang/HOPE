"""Device-resident same-transition ActionBall strike-fact coordinator.

The coordinator is the batched hot-path layer above the semantic strike-fact
contract.  Every hot-path input has a fixed all-environment shape.  R03 freezes
target facts directly from the current ActionEpoch task before physics; the
Racket caller publishes only achieved facts exactly once after physics and
before RewardManager.  All nine guide RewardTerms are installed as named
consumers.  Consumers receive one shared, zero-filled cache object; no consumer
gets a clone or performs a host materialization.

Protocol/value failures are fail-closed device-resident sticky bits.  The
production path contributes its complete portable audit row to the one global
pre-optimizer PPO-boundary transfer.  The legacy :meth:`drain_ppo_boundary`
remains only for isolated tests and old checkpoint migration; once the global
protocol is adopted by an owner, that independent transfer is disabled.
Checkpoint export consumes the portable receipt materialized from the global
row, so an unobserved mutation cannot slip between prepare and serialization.

This module deliberately has no Isaac Lab or MuJoCo dependency.  Runtime
wiring, CUDA profiling, and exact-resume integration remain separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
from typing import Mapping

import torch


SCHEMA_VERSION = 2
SCALE_TARGET_NUM_ENVS = 4096

# Exact RewardTerm names in the shared pre/contact guide recipe.  This portable
# cache has no family branch: matched environments install and pay all of these
# consumers from the same fact.
GUIDE_CONSUMERS = (
    "racket_position",
    "racket_velocity",
    "racket_normal",
    "racket_position_coarse",
    "racket_velocity_coarse",
    "racket_normal_coarse",
    "racket_position_precision",
    "racket_velocity_precision",
    "racket_normal_precision",
)
# The causal capture shaper is a separate RewardTerm and therefore
# owns a separate once-only payment bit.  Append it after the frozen nine-guide
# order so every existing guide index remains stable across schema v1 -> v2.
PADDLE_CENTER_PROXIMITY_CONSUMER = "paddle_center_proximity"
COMMON_CONSUMERS = (PADDLE_CENTER_PROXIMITY_CONSUMER,)
STRIKE_FACT_CONSUMERS = GUIDE_CONSUMERS + COMMON_CONSUMERS
CONSUMER_COUNT = len(STRIKE_FACT_CONSUMERS)
FULL_CONSUMER_MASK = (1 << CONSUMER_COUNT) - 1
FULL_MDP_REWARD_CONSUMERS = STRIKE_FACT_CONSUMERS

RUNTIME_INTEGRATED = False
CUDA_PROFILED = False
FORMAL_EXACT_RESUME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
INTEGRATION_RESIDUALS = (
    "bind fixed-cadence pre-physics arm and reset/wrap ownership in the runtime",
    "wire one post-physics/pre-Reward publish into the constructed Pod1 env graph",
    "wire all ten shared real RewardTerms to this cache",
    "bind coordinator checkpoint state into the runner's whole-checkpoint root",
    "run CUDA fixed-tape parity and profiler-off 4096-env matched-strata timing",
)

_INT_KEY_FIELDS = (
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
)
_DIGEST_KEY_FIELDS = (
    "birth_sha256",
    "sample_sha256",
    "task_sha256",
)
_KEY_FIELDS = _INT_KEY_FIELDS + _DIGEST_KEY_FIELDS
_TARGET_FIELDS = (
    "target_position",
    "target_velocity",
    "target_face_normal",
    "ball_position",
    "ball_velocity",
)
_ACHIEVED_FIELDS = (
    "achieved_position",
    "achieved_velocity",
    "achieved_face_normal",
)
_VECTOR_FIELDS = _TARGET_FIELDS + _ACHIEVED_FIELDS
_MAX_ACTION_UID = (1 << 53) - 1

# Fault bits are per-environment and sticky across reset/wrap.  A row with any
# bit set cannot publish another fact; the boundary receipt fails closed.
FAULT_INVALID_ARM = 1 << 0
FAULT_ARM_COLLISION = 1 << 1
FAULT_STEP_REGRESSION = 1 << 2
FAULT_ARM_EXPIRED = 1 << 3
FAULT_PENDING_EXPIRED = 1 << 4
FAULT_RESET_WRAP_PENDING = 1 << 5
FAULT_UNARMED_PUBLISH = 1 << 6
FAULT_MISSING_PUBLISH = 1 << 7
FAULT_PUBLISH_BINDING = 1 << 8
FAULT_INVALID_PUBLISH = 1 << 9
FAULT_DUPLICATE_VIEW = 1 << 10
FAULT_PAYMENT_BEFORE_VIEW = 1 << 11
FAULT_DUPLICATE_PAYMENT = 1 << 12
FAULT_INVALID_PAYMENT = 1 << 13
FAULT_PAYMENT_OUTSIDE_EVENT = 1 << 14
FAULT_RESET_UNSETTLED = 1 << 15

# Compact R03 producer-fault ABI used by the lean ActionEpoch path.  These are
# facts about the live post-physics source, not Reward values: a normal miss or
# an ineligible row carries no bit.  The bits stay separate from the legacy
# coordinator ledger so the read-only Reward view cannot mutate its producer.
R03_EPOCH_FAULT_STALE_SOURCE_STEP = 1 << 0
R03_EPOCH_FAULT_NONFINITE_FACT = 1 << 1
R03_EPOCH_FAULT_EPOCH_IDENTITY = 1 << 2
R03_EPOCH_FACT_PRESENT = 1 << 0
R03_EPOCH_FACT_PHYSICALLY_VALID = 1 << 1
R03_EPOCH_FACT_VALUE_COUNT = 24

FAULTS = (
    ("invalid_arm", FAULT_INVALID_ARM),
    ("arm_collision", FAULT_ARM_COLLISION),
    ("step_regression", FAULT_STEP_REGRESSION),
    ("arm_expired", FAULT_ARM_EXPIRED),
    ("pending_expired", FAULT_PENDING_EXPIRED),
    ("reset_wrap_pending", FAULT_RESET_WRAP_PENDING),
    ("unarmed_publish", FAULT_UNARMED_PUBLISH),
    ("missing_publish", FAULT_MISSING_PUBLISH),
    ("publish_binding", FAULT_PUBLISH_BINDING),
    ("invalid_publish", FAULT_INVALID_PUBLISH),
    ("duplicate_view", FAULT_DUPLICATE_VIEW),
    ("payment_before_view", FAULT_PAYMENT_BEFORE_VIEW),
    ("duplicate_payment", FAULT_DUPLICATE_PAYMENT),
    ("invalid_payment", FAULT_INVALID_PAYMENT),
    ("payment_outside_event", FAULT_PAYMENT_OUTSIDE_EVENT),
    ("reset_unsettled", FAULT_RESET_UNSETTLED),
)

STATE_NAMES = ("idle", "armed", "pending", "paid")
INVARIANT_NAMES = (
    "state_overlap",
    "arm_storage_dirty",
    "cache_storage_dirty",
    "consumer_bits_out_of_range",
    "paid_without_view",
    "pending_marked_complete",
    "paid_marked_incomplete",
    "unpaid_payment_value",
    "invalid_source_step",
    "invalid_task_key",
    "nonfinite_fact",
    "visible_pending_mismatch",
    "idle_ledger_dirty",
    "accounting_invalid",
)

PRE_OPTIMIZER_PPO_BOUNDARY_OWNER_KIND = "r03_strike_fact"
PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_GLOBAL = "global_pre_optimizer_v1"
PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_LEGACY = "legacy_leaf_drain_v1"
OBSERVATION_PROJECTION_MODE_LEGACY_DIAGNOSTIC = "legacy_or_diagnostic"
OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP = "fresh_full_mdp"

# The first three fields are the frozen global fail-stop prefix.  The remaining
# fields preserve every value in StrikeFactBoundaryReceipt.  They are not a
# same-writer integrity claim: the detailed row exists so the sole global D2H
# can replace the legacy per-leaf drain without deleting portable telemetry.
PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES = (
    "mutation_version",
    "fault_count",
    "invariant_count",
    *(f"fault_{name}_count" for name, _ in FAULTS),
    *(f"state_{name}_count" for name in STATE_NAMES),
    *(f"invariant_{name}_count" for name in INVARIANT_NAMES),
    "armed_total",
    "published_total",
    *(f"payment_{name}_total" for name in STRIKE_FACT_CONSUMERS),
    "reset_total",
)


def make_pre_optimizer_ppo_boundary_leaf_schema(
    *,
    leaf_schema_type: type,
    field_spec_type: type,
) -> object:
    """Build the exact R03 schema using the global drain's public types.

    The types are injected to keep this hot leaf independent of the top-level
    drain module and to avoid an import cycle.  The caller supplies
    ``LeafDrainSchema`` and ``DeviceDrainFieldSpec`` from that module.
    """

    return leaf_schema_type(
        owner_kind=PRE_OPTIMIZER_PPO_BOUNDARY_OWNER_KIND,
        fields=tuple(
            field_spec_type(name=name)
            for name in PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES
        ),
    )


class StrikeFactDeviceError(RuntimeError):
    """A host-side schema, boundary, or checkpoint contract was violated."""


class StrikeFactObservationIdentityHold(StrikeFactDeviceError):
    """R08 cannot consume an R03 publication without a full shot identity."""


class StrikeFactEpochPublicationHold(StrikeFactDeviceError):
    """The lean epoch has not exposed its exact R03 fact-publication API."""


@dataclass(frozen=True)
class DeviceActionTaskKey:
    """Full lossless all-env task key held on one device.

    Integer fields have shape ``[num_envs]`` and dtype ``int64``.  SHA fields
    contain all 32 digest bytes with shape ``[num_envs, 32]`` and dtype
    ``uint8``.  Value/range faults are accumulated on device by the
    coordinator; this dataclass only fixes the ABI.
    """

    env_id: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    birth_sha256: torch.Tensor
    sample_sha256: torch.Tensor
    task_sha256: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "DeviceActionTaskKey":
        if not isinstance(value, Mapping):
            raise StrikeFactDeviceError("task key must be a mapping or DeviceActionTaskKey")
        expected = set(_KEY_FIELDS)
        actual = set(value)
        if actual != expected:
            raise StrikeFactDeviceError(
                "task key fields differ: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )
        return cls(**{name: value[name] for name in _KEY_FIELDS})

    def to_mapping(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in _KEY_FIELDS}


@dataclass(frozen=True)
class EpochR03RacketIdentity:
    """Lean Racket's exact integer join to the D05-owned current epoch.

    These four fields have independent live storage in Racket and therefore
    carry information when joined to the canonical epoch.  Candidate, ball and
    outcome identities remain sole-owned by D05/ActionEpoch; asking Racket to
    echo them would create the same-writer self-check forbidden by the handoff.
    No SHA field exists in this ABI.
    """

    reset_generation: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    task_identity: torch.Tensor


@dataclass(frozen=True)
class SharedStrikeFactDeviceView:
    """One shared read-only-by-contract cache used by every guide consumer.

    The object and every tensor are shared; callers must never mutate them.
    ``eligible`` is true only for a currently published, fault-free fact.
    ``validity`` is a physical/task-fact validity bit, never a family or
    treatment selector.  Rows not carrying a fact are zero-filled in every
    payload tensor.
    """

    eligible: torch.Tensor
    validity: torch.Tensor
    source_step: torch.Tensor
    task_key: DeviceActionTaskKey
    target_position: torch.Tensor
    target_velocity: torch.Tensor
    target_face_normal: torch.Tensor
    ball_position: torch.Tensor
    ball_velocity: torch.Tensor
    achieved_position: torch.Tensor
    achieved_velocity: torch.Tensor
    achieved_face_normal: torch.Tensor


class _OpaqueExactStrikeFkObservationCapability:
    """Empty owner-issued identity; ordinary construction/copy is forbidden."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("exact-strike FK observation capabilities are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("exact-strike FK observation capabilities are immutable")

    def __copy__(self):
        raise TypeError("exact-strike FK observation capabilities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("exact-strike FK observation capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("exact-strike FK observation capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("exact-strike FK observation capabilities cannot be serialized")


class ExactStrikeFkObservationProjection(
    _OpaqueExactStrikeFkObservationCapability
):
    """One-shot opaque identity minted by the exact pre-Reward publication."""

    __slots__ = ()


@dataclass(frozen=True)
class ExactStrikeFkObservationView:
    """Clone-only R03 source facts after a device-side current-identity join.

    ``identity_rejected`` is the same-batch device verdict.  Rejected rows are
    forced false in ``exact_strike_reached`` and zero-filled in every achieved
    FK value, so a consumer cannot misattribute an event by ignoring the
    verdict.  A future FreshObservationOwner must additionally journal the
    verdict before it may remove its integration HOLD.

    ``shot_index`` is deliberately independent of ``swing_generation``.  The
    current production R03 callpoint does not yet provide it; that incomplete
    publication raises :class:`StrikeFactObservationIdentityHold` instead of
    fabricating a join to R05/R06 chronology.
    """

    source_step: torch.Tensor
    task_key: DeviceActionTaskKey
    shot_index: torch.Tensor
    identity_complete: torch.Tensor
    identity_rejected: torch.Tensor
    exact_strike_reached: torch.Tensor
    achieved_position: torch.Tensor
    achieved_velocity: torch.Tensor
    achieved_face_normal: torch.Tensor


class _OpaqueR03FullMdpRewardCapability:
    """Owner-issued production Reward identity; construction/copy is forbidden."""

    __slots__ = ()

    def __new__(cls):
        del cls
        raise TypeError("R03 full-MDP Reward capabilities are owner-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("R03 full-MDP Reward capabilities are immutable")

    def __copy__(self):
        raise TypeError("R03 full-MDP Reward capabilities cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("R03 full-MDP Reward capabilities cannot be copied")

    def __reduce__(self):
        raise TypeError("R03 full-MDP Reward capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        del protocol
        raise TypeError("R03 full-MDP Reward capabilities cannot be serialized")


class R03FullMdpPreRewardPublication(_OpaqueR03FullMdpRewardCapability):
    """Opaque identity for one exact R03 pre-Reward fact publication."""

    __slots__ = ()


class R03FullMdpRewardPaymentVerdict(_OpaqueR03FullMdpRewardCapability):
    """Opaque identity proving one real named ``record_payment`` call."""

    __slots__ = ()


class R03FullMdpRewardCloseReceipt(_OpaqueR03FullMdpRewardCapability):
    """Opaque identity proving the ordered ten-payment R03 epoch closed."""

    __slots__ = ()


@dataclass(frozen=True)
class R03FullMdpPreRewardView:
    """Clone-only view of the causal R03 writer at the pre-Reward seam.

    R03 owns no terminal condition.  Its ``terminated`` and ``time_out`` rows
    are therefore device-local false tensors, not a reinterpretation of a
    normal miss, invalid guide row, or low reward as an infrastructure fault.
    The actual publication identity is the complete strike-fact key plus the
    independently written shot index from :meth:`publish`.
    """

    terminated: torch.Tensor
    time_out: torch.Tensor
    eligible: torch.Tensor
    validity: torch.Tensor
    source_step: torch.Tensor
    task_key: DeviceActionTaskKey
    shot_index: torch.Tensor


@dataclass
class _R03PreRewardRecord:
    publication: R03FullMdpPreRewardPublication
    runtime_owner: object
    control_step: int
    projection_sequence: int
    eligible: torch.Tensor
    validity: torch.Tensor
    source_step: torch.Tensor
    task_key: DeviceActionTaskKey
    shot_index: torch.Tensor
    payment_verdicts: list[R03FullMdpRewardPaymentVerdict]
    stage: str = "issued"


@dataclass
class _R03PaymentRecord:
    verdict: R03FullMdpRewardPaymentVerdict
    publication: R03FullMdpPreRewardPublication
    runtime_owner: object
    control_step: int
    consumer: str
    payment_ordinal: int
    mutation_version: int
    stage: str = "issued"


@dataclass
class _R03CloseRecord:
    receipt: R03FullMdpRewardCloseReceipt
    publication: R03FullMdpPreRewardPublication
    runtime_owner: object
    control_step: int
    payment_verdicts: tuple[R03FullMdpRewardPaymentVerdict, ...]
    stage: str = "issued"


@dataclass
class _ExactStrikeFkObservationProjectionRecord:
    projection: ExactStrikeFkObservationProjection
    publication_sequence: int
    source_step: torch.Tensor
    task_key: DeviceActionTaskKey
    shot_index: torch.Tensor
    identity_complete: torch.Tensor
    exact_strike_reached: torch.Tensor
    achieved_position: torch.Tensor
    achieved_velocity: torch.Tensor
    achieved_face_normal: torch.Tensor
    shot_index_supplied: bool
    stage: str = "issued"


@dataclass(frozen=True)
class StrikeFactBoundaryReceipt:
    """Single-transfer PPO-boundary receipt.

    ``checkpoint_safe`` permits disjoint IDLE/ARMED/PENDING/PAID rows, including
    a partially consumed PENDING row.  It refuses sticky faults or structural
    collisions.  It does not claim runtime integration, CUDA profiling, or
    launch authorization.
    """

    schema_version: int
    update_index: int
    drain_sequence: int
    mutation_version: int
    num_envs: int
    consumers: tuple[str, ...]
    fault_counts: tuple[tuple[str, int], ...]
    state_counts: tuple[tuple[str, int], ...]
    invariant_counts: tuple[tuple[str, int], ...]
    armed_total: int
    published_total: int
    payment_totals: tuple[int, ...]
    reset_total: int
    checkpoint_safe: bool
    device_to_host_transfers: int
    runtime_integrated: bool
    cuda_profiled: bool
    formal_exact_resume_integrated: bool
    launch_authorized: bool


@dataclass
class _PreparedPreOptimizerPpoBoundaryPack:
    """One leaf-local lease over an authority-minted device snapshot."""

    pack: object
    authority: object
    update_index: int
    completed_environment_steps: int
    mutation_version: int


def _expand(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    return mask.reshape((mask.shape[0],) + (1,) * (ndim - 1))


def _masked_copy_(destination: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> None:
    destination.copy_(torch.where(_expand(mask, destination.ndim), source, destination))


def _masked_zero_(tensor: torch.Tensor, mask: torch.Tensor, value: int | float = 0) -> None:
    tensor.masked_fill_(_expand(mask, tensor.ndim), value)


def _exact_action_epoch_owner_type() -> type:
    """Resolve the one epoch implementation in package and focused-test modes."""

    try:
        module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
        )
    except ImportError:
        try:
            module = importlib.import_module("action_ball_full_mdp_epoch")
        except ImportError as exc:
            raise StrikeFactEpochPublicationHold(
                "exact ActionEpochOwner source is unavailable"
            ) from exc
    owner_type = getattr(module, "ActionEpochOwner", None)
    if type(owner_type) is not type:
        raise StrikeFactEpochPublicationHold(
            "exact ActionEpochOwner class is unavailable"
        )
    return owner_type


def _exact_direct_r03_reward_facts_type() -> type:
    """Resolve the Reward graph's exact dataclass without a module cycle."""

    try:
        module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_rewards"
        )
    except ImportError:
        try:
            module = importlib.import_module("action_ball_full_mdp_lean_rewards")
        except ImportError as exc:
            raise StrikeFactEpochPublicationHold(
                "exact DirectR03RewardFacts source is unavailable"
            ) from exc
    facts_type = getattr(module, "DirectR03RewardFacts", None)
    if type(facts_type) is not type:
        raise StrikeFactEpochPublicationHold(
            "exact DirectR03RewardFacts class is unavailable"
        )
    return facts_type


class ActionBallStrikeFactDeviceCoordinator:
    """Fixed-shape all-env strike fact and ten-consumer payment authority."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        observation_projection_mode: str = (
            OBSERVATION_PROJECTION_MODE_LEGACY_DIAGNOSTIC
        ),
        action_epoch_owner: object | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise StrikeFactDeviceError("num_envs must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise StrikeFactDeviceError("dtype must be a floating torch dtype")
        if observation_projection_mode not in (
            OBSERVATION_PROJECTION_MODE_LEGACY_DIAGNOSTIC,
            OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP,
        ):
            raise StrikeFactDeviceError(
                "observation_projection_mode must explicitly select legacy/diagnostic "
                "or fresh full-MDP"
            )

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        if action_epoch_owner is not None:
            owner_type = _exact_action_epoch_owner_type()
            if type(action_epoch_owner) is not owner_type:
                raise StrikeFactDeviceError(
                    "action_epoch_owner must be the exact ActionEpochOwner"
                )
            if (
                action_epoch_owner.num_envs != num_envs
                or action_epoch_owner.device != self.device
            ):
                raise StrikeFactDeviceError(
                    "action_epoch_owner batch/device differs from R03"
                )
        self._action_epoch_owner = action_epoch_owner
        if action_epoch_owner is not None:
            # Cold construction identity is the only publication authority.
            # The epoch does not accept a caller receipt, digest, or bool in
            # place of this exact object join.
            action_epoch_owner.bind_fact_owner("r03_strike_fact", self)
        self._epoch_arm_identity: EpochR03RacketIdentity | None = None
        self._epoch_arm_source_step: torch.Tensor | None = None
        self._epoch_arm_mask: torch.Tensor | None = None
        self._epoch_arm_vectors: dict[str, torch.Tensor] | None = None
        self._epoch_arm_shot_key: object | None = None
        self._epoch_arm_slot: torch.Tensor | None = None
        self._epoch_arm_d05_identity: dict[str, torch.Tensor] | None = None
        self._epoch_racket_owner: object | None = None
        # Construction-fixed: the fresh runtime/factory must explicitly choose
        # its stricter writer ABI.  Runtime mutation cannot downgrade it to the
        # legacy diagnostic behavior.
        self._observation_projection_mode = observation_projection_mode
        self.consumers = STRIKE_FACT_CONSUMERS
        self._consumer_index = {
            name: index for index, name in enumerate(self.consumers)
        }
        self._env_ids = torch.arange(num_envs, dtype=torch.int64, device=self.device)

        self._armed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._pending = torch.zeros_like(self._armed)
        self._paid = torch.zeros_like(self._armed)
        self._visible = torch.zeros_like(self._armed)
        self._fault_bits = torch.zeros(num_envs, dtype=torch.int64, device=self.device)

        self._arm_source_step = torch.full(
            (num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._cache_source_step = torch.full_like(self._arm_source_step, -1)
        self._arm_key_ints = self._new_key_int_buffers()
        self._cache_key_ints = self._new_key_int_buffers()
        self._arm_key_digests = self._new_key_digest_buffers()
        self._cache_key_digests = self._new_key_digest_buffers()

        self._arm_vectors = {
            name: torch.zeros((num_envs, 3), dtype=dtype, device=self.device)
            for name in _TARGET_FIELDS
        }
        self._cache_vectors = {
            name: torch.zeros((num_envs, 3), dtype=dtype, device=self.device)
            for name in _VECTOR_FIELDS
        }
        self._arm_validity = torch.zeros_like(self._armed)
        self._cache_validity = torch.zeros_like(self._armed)
        self._viewed_mask = torch.zeros(
            num_envs, dtype=torch.int64, device=self.device
        )
        self._paid_mask = torch.zeros_like(self._viewed_mask)
        self._payment_values = torch.zeros(
            (num_envs, CONSUMER_COUNT), dtype=dtype, device=self.device
        )

        self._armed_total = torch.zeros((), dtype=torch.int64, device=self.device)
        self._published_total = torch.zeros_like(self._armed_total)
        self._payment_totals = torch.zeros(
            CONSUMER_COUNT, dtype=torch.int64, device=self.device
        )
        self._reset_total = torch.zeros_like(self._armed_total)

        cache_key = DeviceActionTaskKey(
            **{
                **self._cache_key_ints,
                **self._cache_key_digests,
            }
        )
        self._shared_view = SharedStrikeFactDeviceView(
            eligible=self._visible,
            validity=self._cache_validity,
            source_step=self._cache_source_step,
            task_key=cache_key,
            **self._cache_vectors,
        )

        # Host-only lifecycle metadata; it never observes tensor values.
        self._mutation_version = 0
        self._drain_sequence = 0
        self._last_drained_update = -1
        self._last_receipt: StrikeFactBoundaryReceipt | None = None
        self._last_global_receipt: object | None = None
        self._last_receipt_consumed = False
        self._last_globally_acknowledged_mutation_version = -1
        self._checkpoint_requires_global_drain_ack = True
        self._pre_optimizer_global_protocol_adopted = False
        self._active_pre_optimizer_pack: _PreparedPreOptimizerPpoBoundaryPack | None = None
        self._pre_optimizer_poisoned = False
        self._pre_optimizer_poison_reason: str | None = None
        # R08 source-only projection state.  The projection is deliberately
        # outside the guide view/payment protocol: minting it never calls
        # ``view()``, never changes either consumer mask, and never advances
        # the coordinator mutation/drain version.  Records are bounded to the
        # currently published sequence and are invalidated by every lifecycle
        # mutation that can replace or clear the cache.
        self._exact_strike_fk_publication_sequence = 0
        self._exact_strike_fk_current_projection: (
            ExactStrikeFkObservationProjection | None
        ) = None
        self._exact_strike_fk_projection_records: dict[
            ExactStrikeFkObservationProjection,
            _ExactStrikeFkObservationProjectionRecord,
        ] = {}
        self._exact_strike_fk_shot_index: torch.Tensor | None = None
        # Fresh Reward is a separate host-capability lifecycle over the real
        # device cache.  It never replaces the ten per-row payment bits.  The
        # opaque records prove which actual ``record_payment`` mutations paid
        # the exact pre-Reward publication and prevent graph receipts from
        # self-authorizing a close.
        self.full_mdp_reward_consumers = FULL_MDP_REWARD_CONSUMERS
        self._active_full_mdp_reward_publication: (
            R03FullMdpPreRewardPublication | None
        ) = None
        self._full_mdp_pre_reward_records: dict[
            R03FullMdpPreRewardPublication, _R03PreRewardRecord
        ] = {}
        self._full_mdp_payment_records: dict[
            R03FullMdpRewardPaymentVerdict, _R03PaymentRecord
        ] = {}
        self._full_mdp_close_records: dict[
            R03FullMdpRewardCloseReceipt, _R03CloseRecord
        ] = {}

    @property
    def pre_optimizer_poisoned(self) -> bool:
        return self._pre_optimizer_poisoned

    @property
    def pre_optimizer_poison_reason(self) -> str | None:
        return self._pre_optimizer_poison_reason

    @property
    def action_epoch_owner(self) -> object | None:
        """Construction-fixed lean owner identity, if this is the lean path."""

        return self._action_epoch_owner

    def bind_action_epoch_owner(self, action_epoch_owner: object) -> None:
        """One-time cold bind for R03 instances constructed by CommandManager.

        Live Racket constructs the unique R03 before the outer full-MDP factory
        can create the shared ActionEpoch.  This seam permits exactly that cold
        dependency inversion.  It rejects rebinding and any attachment after
        R03 has armed, published, faulted, paid, drained, or otherwise changed
        its initial state; it cannot be used as a hot repair path.
        """

        if self._action_epoch_owner is not None:
            raise StrikeFactDeviceError("action_epoch_owner is already bound")
        owner_type = _exact_action_epoch_owner_type()
        if type(action_epoch_owner) is not owner_type:
            raise StrikeFactDeviceError(
                "action_epoch_owner must be the exact ActionEpochOwner"
            )
        if (
            action_epoch_owner.num_envs != self.num_envs
            or action_epoch_owner.device != self.device
        ):
            raise StrikeFactDeviceError(
                "action_epoch_owner batch/device differs from R03"
            )
        cold = (
            self._mutation_version == 0
            and self._drain_sequence == 0
            and self._last_drained_update == -1
            and self._active_pre_optimizer_pack is None
            and self._active_full_mdp_reward_publication is None
        )
        # Every public cache/fault lifecycle mutation advances the host-local
        # mutation version, so no tensor-to-host check is needed here.
        if not cold:
            raise StrikeFactDeviceError(
                "action_epoch_owner may bind only before any R03 publication"
            )
        action_epoch_owner.bind_fact_owner("r03_strike_fact", self)
        self._action_epoch_owner = action_epoch_owner

    def bind_action_epoch_racket_owner(self, racket_owner: object) -> None:
        """Cold-bind the sole producer of exact-strike eligibility and FK."""

        if self._action_epoch_owner is None:
            raise StrikeFactDeviceError(
                "lean R03 Racket owner requires the ActionEpoch owner first"
            )
        if racket_owner is None or self._epoch_racket_owner is not None:
            raise StrikeFactDeviceError(
                "lean R03 Racket owner is absent or already bound"
            )
        if self._mutation_version != 0 or self._epoch_arm_identity is not None:
            raise StrikeFactDeviceError(
                "lean R03 Racket owner must bind before any R03 mutation"
            )
        active_guard = getattr(
            racket_owner, "require_active_action_epoch_r03_writer", None
        )
        if (
            not callable(active_guard)
            or getattr(active_guard, "__self__", None) is not racket_owner
        ):
            raise StrikeFactDeviceError(
                "lean R03 Racket owner lacks its exact active-writer guard"
            )
        self._epoch_racket_owner = racket_owner

    def _require_active_epoch_racket_writer(self, racket_owner: object) -> None:
        if racket_owner is not self._epoch_racket_owner:
            raise StrikeFactDeviceError(
                "lean R03 write requires the cold-bound Racket owner"
            )
        guard = getattr(
            racket_owner, "require_active_action_epoch_r03_writer", None
        )
        if not callable(guard) or getattr(guard, "__self__", None) is not racket_owner:
            raise StrikeFactDeviceError(
                "lean R03 write lost the bound Racket active-writer guard"
            )
        guard()

    def _latch_action_epoch_producer_faults(
        self,
        *,
        epoch_module: object,
        identity_fault: torch.Tensor,
        stale_source: torch.Tensor,
        nonfinite: torch.Tensor,
    ) -> torch.Tensor:
        """Route every R03 producer cause into Epoch's sole PPO fault drain."""

        if self._action_epoch_owner is None:  # pragma: no cover - caller guard
            raise StrikeFactEpochPublicationHold(
                "lean R03 fault latch requires construction-bound ActionEpochOwner"
            )
        self._action_epoch_owner.latch_runtime_row_fault(
            "r03_strike_fact",
            epoch_module.ROW_FAULT_R03_EPOCH_IDENTITY,
            identity_fault,
            owner=self,
        )
        self._action_epoch_owner.latch_runtime_row_fault(
            "r03_strike_fact",
            epoch_module.ROW_FAULT_R03_STALE_SOURCE_STEP,
            stale_source,
            owner=self,
        )
        return self._action_epoch_owner.latch_runtime_row_fault(
            "r03_strike_fact",
            epoch_module.ROW_FAULT_R03_NONFINITE_FACT,
            nonfinite,
            owner=self,
        )

    def _epoch_racket_identity(
        self, value: EpochR03RacketIdentity
    ) -> EpochR03RacketIdentity:
        if type(value) is not EpochR03RacketIdentity:
            raise StrikeFactDeviceError(
                "lean R03 identity must be exact EpochR03RacketIdentity"
            )
        owned = {}
        for name in value.__dataclass_fields__:
            tensor = getattr(value, name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (self.num_envs,)
                or tensor.dtype != torch.int64
                or tensor.device != self.device
            ):
                raise StrikeFactDeviceError(
                    f"lean R03 identity {name} must be device int64 [num_envs]"
                )
            owned[name] = tensor.detach().clone()
        return EpochR03RacketIdentity(**owned)

    def arm_action_epoch_strike_fact_v1(
        self,
        *,
        racket_owner: object,
        source_step: torch.Tensor,
        racket_identity: EpochR03RacketIdentity,
    ) -> None:
        """Arm from the bound epoch task and Racket's independent integers.

        Eligibility is the intersection of the launched epoch phase, the
        admitted task-valid row, and D05's absolute contact tick.  The latter
        is the integer chronology authority for the shot: deriving the event
        from Racket's manager-metric latch would observe the clock before the
        command update and publish one 20 ms control step late.  This lean path
        never reads or manufactures the legacy birth/sample/task SHA fields.
        Target position, velocity, face normal, ball position and incoming ball
        velocity are frozen only from the selected ActionEpoch ``task_f32``
        Racket slice.  Positions use that slice's environment-local scene frame;
        no Racket-owned target mirror can certify or replace them.
        """

        if self._action_epoch_owner is None:
            raise StrikeFactEpochPublicationHold(
                "lean R03 arm requires construction-bound ActionEpochOwner"
            )
        if self._epoch_racket_owner is None:
            raise StrikeFactEpochPublicationHold(
                "lean R03 arm requires construction-bound Racket owner"
            )
        self._require_active_epoch_racket_writer(racket_owner)
        self._require_hot_mutation_permitted()
        if self._epoch_arm_identity is not None:
            raise StrikeFactDeviceError("previous lean R03 epoch arm is unsettled")
        steps = self._steps(source_step).detach().clone()
        identity = self._epoch_racket_identity(racket_identity)
        epoch = self._action_epoch_owner.current()
        slot = epoch.current_task_slot

        def selected(value: torch.Tensor) -> torch.Tensor:
            suffix = (1,) * (value.ndim - 2)
            index = slot.reshape(self.num_envs, 1, *suffix).expand(
                self.num_envs, 1, *value.shape[2:]
            )
            return torch.gather(value, 1, index).squeeze(1)

        epoch_module = importlib.import_module(type(self._action_epoch_owner).__module__)
        selected_shot_key = epoch_module.ActionEpochShotKey(
            **{
                field.name: selected(
                    getattr(epoch.identity.shot_key, field.name)
                ).detach().clone()
                for field in fields(epoch_module.ActionEpochShotKey)
            }
        )
        shot_key_valid = epoch_module.row_identity.action_epoch_shot_key_valid(
            selected_shot_key
        )
        contact_tick = selected(epoch.clocks.contact_tick)
        task_valid = selected(epoch.task.task_valid)
        exact = steps.eq(contact_tick)
        launched_task = (
            selected(epoch.phase).eq(epoch_module.PHASE_LAUNCH_SETTLED)
            & task_valid
        )
        eligible = launched_task & exact
        identity_fault = eligible & (
            ~shot_key_valid
            | (identity.reset_generation != epoch.reset_generation)
            | (identity.action_uid != selected(epoch.identity.action_uid))
            | (identity.action_slot != selected(epoch.identity.action_slot))
            | (identity.task_identity != selected(epoch.identity.task_identity))
        )
        task_f32 = selected(epoch.task.task_f32)
        racket_start = epoch_module.MOTION_TASK_F32_WIDTH
        racket_task = task_f32[
            :, racket_start : racket_start + epoch_module.RACKET_TASK_F32_WIDTH
        ]
        target_vectors = {
            "target_position": self._vector(
                racket_task[:, 0:3], "epoch.target_position"
            ),
            "target_velocity": self._vector(
                racket_task[:, 3:6], "epoch.target_velocity"
            ),
            "target_face_normal": self._vector(
                racket_task[:, 6:9], "epoch.target_face_normal"
            ),
            "ball_position": self._vector(
                racket_task[:, 9:12], "epoch.ball_position"
            ),
            "ball_velocity": self._vector(
                racket_task[:, 21:24], "epoch.ball_velocity"
            ),
        }
        stale_source = launched_task & steps.lt(0)
        nonfinite = eligible & self._vectors_nonfinite(target_vectors)
        epoch_safe = self._latch_action_epoch_producer_faults(
            epoch_module=epoch_module,
            identity_fault=identity_fault,
            stale_source=stale_source,
            nonfinite=nonfinite,
        )
        fault = (
            identity_fault.to(torch.int64) * R03_EPOCH_FAULT_EPOCH_IDENTITY
            | stale_source.to(torch.int64) * R03_EPOCH_FAULT_STALE_SOURCE_STEP
            | nonfinite.to(torch.int64) * R03_EPOCH_FAULT_NONFINITE_FACT
        )
        fault_slots = torch.zeros(
            (self.num_envs, self._action_epoch_owner.shot_slot_capacity),
            dtype=torch.int64,
            device=self.device,
        )
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        fault_slots[env_ids, slot] = fault
        self._action_epoch_owner.merge_runtime_owner_fault(
            "r03_strike_fact", fault_slots, owner=self
        )
        safe = eligible & fault.eq(0) & epoch_safe
        self._epoch_arm_source_step = steps
        self._epoch_arm_identity = identity
        self._epoch_arm_mask = safe
        self._epoch_arm_shot_key = selected_shot_key
        self._epoch_arm_slot = slot.detach().clone()
        self._epoch_arm_d05_identity = {
            name: selected(getattr(epoch.identity, name)).detach().clone()
            for name in (
                "candidate_identity",
                "ball_identity",
                "outcome_identity",
                "scheduled_ordinal",
                "target_generation",
                "selected_cell",
            )
        }
        self._epoch_arm_vectors = {
            name: value.detach().clone() for name, value in target_vectors.items()
        }

    def publish_action_epoch_strike_fact_v1(
        self,
        *,
        racket_owner: object,
        source_step: torch.Tensor,
        racket_identity: EpochR03RacketIdentity,
        achieved_position: torch.Tensor,
        achieved_velocity: torch.Tensor,
        achieved_face_normal: torch.Tensor,
    ) -> None:
        """Publish lean R03 facts without SHA, caller mask, or shot verdict."""

        self._require_active_epoch_racket_writer(racket_owner)
        self._require_hot_mutation_permitted()
        if (
            self._action_epoch_owner is None
            or self._epoch_arm_identity is None
            or self._epoch_arm_source_step is None
            or self._epoch_arm_mask is None
            or self._epoch_arm_shot_key is None
            or self._epoch_arm_slot is None
            or self._epoch_arm_d05_identity is None
        ):
            raise StrikeFactDeviceError("lean R03 publish has no exact epoch arm")
        steps = self._steps(source_step)
        identity = self._epoch_racket_identity(racket_identity)
        epoch = self._action_epoch_owner.current()
        slot = epoch.current_task_slot
        index = slot[:, None]

        def selected(value: torch.Tensor) -> torch.Tensor:
            return torch.gather(value, 1, index).squeeze(1)

        epoch_module = importlib.import_module(type(self._action_epoch_owner).__module__)
        selected_shot_key = epoch_module.ActionEpochShotKey(
            **{
                field.name: selected(
                    getattr(epoch.identity.shot_key, field.name)
                ).detach().clone()
                for field in fields(epoch_module.ActionEpochShotKey)
            }
        )
        shot_key_matches_arm = epoch_module.row_identity.action_epoch_shot_key_equal(
            selected_shot_key, self._epoch_arm_shot_key
        )
        stale_source = self._epoch_arm_mask & (steps != self._epoch_arm_source_step)
        identity_fault = self._epoch_arm_mask & (
            (slot != self._epoch_arm_slot)
            | ~shot_key_matches_arm
            | (identity.reset_generation != epoch.reset_generation)
            | (identity.action_uid != selected(epoch.identity.action_uid))
            | (identity.action_slot != selected(epoch.identity.action_slot))
            | (identity.task_identity != selected(epoch.identity.task_identity))
        )
        for name, armed in self._epoch_arm_d05_identity.items():
            identity_fault = identity_fault | (
                self._epoch_arm_mask
                & (selected(getattr(epoch.identity, name)) != armed)
            )
        achieved = {
            "achieved_position": self._vector(achieved_position, "achieved_position"),
            "achieved_velocity": self._vector(achieved_velocity, "achieved_velocity"),
            "achieved_face_normal": self._vector(
                achieved_face_normal, "achieved_face_normal"
            ),
        }
        nonfinite = self._epoch_arm_mask & self._vectors_nonfinite(achieved)
        epoch_safe = self._latch_action_epoch_producer_faults(
            epoch_module=epoch_module,
            identity_fault=identity_fault,
            stale_source=stale_source,
            nonfinite=nonfinite,
        )
        fault = (
            identity_fault.to(torch.int64) * R03_EPOCH_FAULT_EPOCH_IDENTITY
            | stale_source.to(torch.int64) * R03_EPOCH_FAULT_STALE_SOURCE_STEP
            | nonfinite.to(torch.int64) * R03_EPOCH_FAULT_NONFINITE_FACT
        )
        slots = self._action_epoch_owner.shot_slot_capacity
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        fault_slots = torch.zeros(
            (self.num_envs, slots), dtype=torch.int64, device=self.device
        )
        fault_slots[env_ids, slot] = fault
        self._action_epoch_owner.merge_runtime_owner_fault(
            "r03_strike_fact", fault_slots, owner=self
        )
        safe = self._epoch_arm_mask & fault.eq(0) & epoch_safe
        valid_bits = torch.zeros_like(fault_slots)
        selected_valid = (
            safe.to(torch.int64) * R03_EPOCH_FACT_PRESENT
            | safe.to(torch.int64) * R03_EPOCH_FACT_PHYSICALLY_VALID
        )
        valid_bits[env_ids, slot] = selected_valid
        fact_step = torch.full_like(fault_slots, -1)
        fact_step[env_ids, slot] = torch.where(
            safe, steps, torch.full_like(steps, -1)
        )
        values = torch.zeros(
            (self.num_envs, slots, epoch_module.OWNER_FACT_F32_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        packed = torch.cat(
            (
                self._epoch_arm_vectors["target_position"],
                self._epoch_arm_vectors["target_velocity"],
                self._epoch_arm_vectors["target_face_normal"],
                self._epoch_arm_vectors["ball_position"],
                self._epoch_arm_vectors["ball_velocity"],
                achieved["achieved_position"],
                achieved["achieved_velocity"],
                achieved["achieved_face_normal"],
            ),
            dim=1,
        )
        packed = torch.where(safe[:, None], packed, torch.zeros_like(packed))
        values[env_ids, slot, :R03_EPOCH_FACT_VALUE_COUNT] = packed
        self._action_epoch_owner.publish_owner_facts(
            "r03_strike_fact",
            owner=self,
            valid_bits=valid_bits,
            source_step=fact_step,
            values=values,
        )
        self._epoch_arm_identity = None
        self._epoch_arm_source_step = None
        self._epoch_arm_mask = None
        self._epoch_arm_vectors = None
        self._epoch_arm_shot_key = None
        self._epoch_arm_slot = None
        self._epoch_arm_d05_identity = None

    def _require_no_armed_epoch_fact(self, *, operation: str) -> None:
        if self._epoch_arm_identity is not None:
            raise StrikeFactDeviceError(
                f"R03 {operation} cannot cross an armed lean epoch fact"
            )

    def action_epoch_reward_facts_v1(self, epoch: object) -> object:
        """Return the exact fixed-slot direct facts expected by lean Reward.

        The passed record must be the current version of this coordinator's
        construction-bound epoch.  Identity/source/finiteness joins already
        happened at the unique producer publication; Reward decodes only the
        immutable R03 fact slice.  Contact chronology remains a separate clock.
        A stale or foreign record is rejected and this method never mutates
        either owner.
        """

        if self._action_epoch_owner is None:
            raise StrikeFactEpochPublicationHold(
                "R03 direct Reward facts require construction-bound ActionEpochOwner"
            )
        owner_type = _exact_action_epoch_owner_type()
        epoch_type = importlib.import_module(owner_type.__module__).ActionEpochRecord
        if type(epoch) is not epoch_type:
            raise StrikeFactDeviceError(
                "epoch must be the exact current ActionEpochRecord"
            )
        current = self._action_epoch_owner.current()
        if (
            epoch.version != current.version
            or epoch.current_task_slot.shape != (self.num_envs,)
            or epoch.current_task_slot.dtype != torch.int64
            or epoch.current_task_slot.device != self.device
        ):
            raise StrikeFactDeviceError(
                "epoch is foreign, stale, or has an invalid current slot ABI"
            )
        # The argument carries only chronology.  Decode the owner's fresh
        # immutable clone so a caller cannot inject alternate payload tensors
        # while repeating only the public snapshot version.
        epoch = current

        epoch_module = importlib.import_module(owner_type.__module__)
        owner_slot = epoch_module.OWNER_ORDER.index("r03_strike_fact")
        valid_bits = epoch.fact_valid_bits[:, :, owner_slot].detach().clone()
        facts = epoch.fact_f32[:, :, owner_slot].detach().clone()
        faults = epoch.owner_fault_bits[:, :, owner_slot].detach().clone()
        eligible = (valid_bits & R03_EPOCH_FACT_PRESENT).ne(0) & faults.eq(0)
        validity = (
            (valid_bits & R03_EPOCH_FACT_PHYSICALLY_VALID).ne(0) & eligible
        )

        def vector(start: int) -> torch.Tensor:
            value = facts[:, :, start : start + 3]
            return torch.where(
                eligible[:, :, None], value, torch.zeros_like(value)
            )

        facts_type = _exact_direct_r03_reward_facts_type()
        return facts_type(
            eligible=eligible,
            validity=validity,
            producer_fault_bits=faults,
            target_position=vector(0),
            target_velocity=vector(3),
            target_face_normal=vector(6),
            ball_position=vector(9),
            achieved_position=vector(15),
            achieved_velocity=vector(18),
            achieved_face_normal=vector(21),
        )

    def _require_pre_optimizer_operable(self) -> None:
        if self._pre_optimizer_poisoned:
            raise StrikeFactDeviceError(
                "R03 pre-optimizer leaf is poisoned and requires cold replacement"
            )

    def _require_hot_mutation_permitted(self) -> None:
        self._require_pre_optimizer_operable()
        if self._active_pre_optimizer_pack is not None:
            self._poison_pre_optimizer(
                "R03 hot mutation attempted while a pre-optimizer pack was active"
            )
            raise StrikeFactDeviceError(
                "R03 hot state cannot mutate while a pre-optimizer pack is active"
            )

    def _require_no_open_full_mdp_reward_cycle(self, *, operation: str) -> None:
        """Keep reset/drain/checkpoint from crossing an unpaid Reward epoch."""

        if self._active_full_mdp_reward_publication is not None:
            raise StrikeFactDeviceError(
                f"R03 {operation} cannot cross an open full-MDP Reward cycle"
            )

    def _advance_mutation_version(self) -> None:
        """Advance the live writer and invalidate the cold checkpoint frontier."""

        self._mutation_version += 1
        self._checkpoint_requires_global_drain_ack = True

    def _poison_pre_optimizer(self, reason: object) -> None:
        if self._pre_optimizer_poison_reason is None:
            self._pre_optimizer_poison_reason = (
                reason
                if type(reason) is str and bool(reason) and reason.isascii()
                else "unspecified R03 pre-optimizer protocol failure"
            )
        self._pre_optimizer_poisoned = True
        self._active_pre_optimizer_pack = None

    def _new_key_int_buffers(self) -> dict[str, torch.Tensor]:
        result = {
            name: torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            for name in _INT_KEY_FIELDS
        }
        result["env_id"].fill_(-1)
        return result

    def _new_key_digest_buffers(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros(
                (self.num_envs, 32), dtype=torch.uint8, device=self.device
            )
            for name in _DIGEST_KEY_FIELDS
        }

    def _mask(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise StrikeFactDeviceError(f"{name} must be a torch.Tensor")
        if (
            value.shape != (self.num_envs,)
            or value.dtype != torch.bool
            or value.device != self.device
        ):
            raise StrikeFactDeviceError(
                f"{name} must be device-local bool [{self.num_envs}]"
            )
        return value.detach()

    def _steps(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise StrikeFactDeviceError("source_step must be a torch.Tensor")
        if (
            value.shape != (self.num_envs,)
            or value.dtype != torch.int64
            or value.device != self.device
        ):
            raise StrikeFactDeviceError(
                f"source_step must be device-local int64 [{self.num_envs}]"
            )
        return value.detach()

    def _key(
        self, value: DeviceActionTaskKey | Mapping[str, torch.Tensor]
    ) -> DeviceActionTaskKey:
        key = value if isinstance(value, DeviceActionTaskKey) else DeviceActionTaskKey.from_mapping(value)
        for name in _INT_KEY_FIELDS:
            tensor = getattr(key, name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (self.num_envs,)
                or tensor.dtype != torch.int64
                or tensor.device != self.device
            ):
                raise StrikeFactDeviceError(
                    f"task key {name} must be device-local int64 [{self.num_envs}]"
                )
        for name in _DIGEST_KEY_FIELDS:
            tensor = getattr(key, name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (self.num_envs, 32)
                or tensor.dtype != torch.uint8
                or tensor.device != self.device
            ):
                raise StrikeFactDeviceError(
                    f"task key {name} must be device-local uint8 [{self.num_envs},32]"
                )
        return key

    def _vector(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise StrikeFactDeviceError(f"{name} must be a torch.Tensor")
        if (
            value.shape != (self.num_envs, 3)
            or value.dtype != self.dtype
            or value.device != self.device
        ):
            raise StrikeFactDeviceError(
                f"{name} must be device-local {self.dtype} [{self.num_envs},3]"
            )
        return value.detach()

    def _payment(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise StrikeFactDeviceError("payment must be a torch.Tensor")
        if (
            value.shape != (self.num_envs,)
            or value.dtype != self.dtype
            or value.device != self.device
        ):
            raise StrikeFactDeviceError(
                f"payment must be device-local {self.dtype} [{self.num_envs}]"
            )
        return value.detach()

    def _consumer(self, name: str) -> tuple[int, int]:
        if name not in self._consumer_index:
            raise StrikeFactDeviceError(f"unknown strike-fact consumer: {name!r}")
        index = self._consumer_index[name]
        return index, 1 << index

    def _key_value_fault(self, key: DeviceActionTaskKey) -> torch.Tensor:
        fault = (
            (key.env_id != self._env_ids)
            | (key.reset_generation < 1)
            | (key.swing_generation < 0)
            | (key.action_uid < 1)
            | (key.action_uid > _MAX_ACTION_UID)
            | (key.action_slot < 0)
        )
        for name in _DIGEST_KEY_FIELDS:
            fault = fault | torch.all(getattr(key, name) == 0, dim=1)
        return fault

    def _vectors_nonfinite(self, vectors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        result = torch.zeros_like(self._armed)
        for value in vectors.values():
            result = result | ~torch.all(torch.isfinite(value), dim=1)
        return result

    def _binding_fault(
        self,
        source_step: torch.Tensor,
        key: DeviceActionTaskKey,
        *,
        arm: bool,
    ) -> torch.Tensor:
        stored_step = self._arm_source_step if arm else self._cache_source_step
        ints = self._arm_key_ints if arm else self._cache_key_ints
        digests = self._arm_key_digests if arm else self._cache_key_digests
        mismatch = stored_step != source_step
        for name in _INT_KEY_FIELDS:
            mismatch = mismatch | (ints[name] != getattr(key, name))
        for name in _DIGEST_KEY_FIELDS:
            mismatch = mismatch | torch.any(
                digests[name] != getattr(key, name), dim=1
            )
        return mismatch

    def _key_binding_fault(
        self, key: DeviceActionTaskKey, *, arm: bool
    ) -> torch.Tensor:
        ints = self._arm_key_ints if arm else self._cache_key_ints
        digests = self._arm_key_digests if arm else self._cache_key_digests
        mismatch = torch.zeros_like(self._armed)
        for name in _INT_KEY_FIELDS:
            mismatch = mismatch | (ints[name] != getattr(key, name))
        for name in _DIGEST_KEY_FIELDS:
            mismatch = mismatch | torch.any(
                digests[name] != getattr(key, name), dim=1
            )
        return mismatch

    def _set_fault(self, mask: torch.Tensor, bit: int) -> None:
        self._fault_bits.bitwise_or_(mask.to(dtype=torch.int64) * bit)
        self._visible.bitwise_and_(~mask)

    def _clear_arm(self, mask: torch.Tensor) -> None:
        self._arm_source_step.masked_fill_(mask, -1)
        for name, tensor in self._arm_key_ints.items():
            tensor.masked_fill_(mask, -1 if name == "env_id" else 0)
        for tensor in self._arm_key_digests.values():
            _masked_zero_(tensor, mask)
        for tensor in self._arm_vectors.values():
            _masked_zero_(tensor, mask)
        self._arm_validity.masked_fill_(mask, False)
        self._armed.bitwise_and_(~mask)

    def _clear_cache(self, mask: torch.Tensor) -> None:
        self._cache_source_step.masked_fill_(mask, -1)
        for name, tensor in self._cache_key_ints.items():
            tensor.masked_fill_(mask, -1 if name == "env_id" else 0)
        for tensor in self._cache_key_digests.values():
            _masked_zero_(tensor, mask)
        for tensor in self._cache_vectors.values():
            _masked_zero_(tensor, mask)
        self._cache_validity.masked_fill_(mask, False)
        self._viewed_mask.masked_fill_(mask, 0)
        self._paid_mask.masked_fill_(mask, 0)
        _masked_zero_(self._payment_values, mask)
        self._pending.bitwise_and_(~mask)
        self._paid.bitwise_and_(~mask)
        self._visible.bitwise_and_(~mask)

    @staticmethod
    def _clone_device_task_key(key: DeviceActionTaskKey) -> DeviceActionTaskKey:
        return DeviceActionTaskKey(
            **{
                name: getattr(key, name).detach().clone()
                for name in _KEY_FIELDS
            }
        )

    def _current_cache_task_key_clone(self) -> DeviceActionTaskKey:
        return DeviceActionTaskKey(
            **{
                **{
                    name: tensor.detach().clone()
                    for name, tensor in self._cache_key_ints.items()
                },
                **{
                    name: tensor.detach().clone()
                    for name, tensor in self._cache_key_digests.items()
                },
            }
        )

    def _invalidate_exact_strike_fk_observation_projection(self) -> None:
        current = self._exact_strike_fk_current_projection
        if current is not None:
            record = self._exact_strike_fk_projection_records.get(current)
            if record is not None:
                record.stage = "stale"
        self._exact_strike_fk_current_projection = None
        self._exact_strike_fk_shot_index = None
        # Keep the owner-private capability registry bounded to the one exact
        # publication.  A removed capability still fails as stale/foreign.
        self._exact_strike_fk_projection_records.clear()

    def _mint_exact_strike_fk_observation_projection(
        self,
        *,
        published_mask: torch.Tensor,
        shot_index: torch.Tensor | None,
    ) -> ExactStrikeFkObservationProjection:
        """Capture the exact pre-Reward R03 writer after successful publish.

        ``shot_index`` is optional only while the production R03 caller is
        being migrated.  Such a projection records that identity is
        incomplete and therefore cannot be required by an R08 consumer.
        """

        supplied = shot_index is not None
        if supplied:
            if (
                not isinstance(shot_index, torch.Tensor)
                or shot_index.shape != (self.num_envs,)
                or shot_index.dtype != torch.int64
                or shot_index.device != self.device
            ):
                raise StrikeFactDeviceError(
                    "shot_index must be device-local int64 [num_envs]"
                )
            copied_shot_index = shot_index.detach().clone()
            identity_complete = published_mask & (copied_shot_index >= 1)
        else:
            copied_shot_index = torch.zeros(
                self.num_envs, dtype=torch.int64, device=self.device
            )
            identity_complete = torch.zeros_like(published_mask)

        self._invalidate_exact_strike_fk_observation_projection()
        self._exact_strike_fk_publication_sequence += 1
        projection = object.__new__(ExactStrikeFkObservationProjection)
        record = _ExactStrikeFkObservationProjectionRecord(
            projection=projection,
            publication_sequence=self._exact_strike_fk_publication_sequence,
            source_step=self._cache_source_step.detach().clone(),
            task_key=self._current_cache_task_key_clone(),
            shot_index=copied_shot_index,
            identity_complete=identity_complete.detach().clone(),
            exact_strike_reached=published_mask.detach().clone(),
            achieved_position=self._cache_vectors["achieved_position"].detach().clone(),
            achieved_velocity=self._cache_vectors["achieved_velocity"].detach().clone(),
            achieved_face_normal=self._cache_vectors[
                "achieved_face_normal"
            ].detach().clone(),
            shot_index_supplied=supplied,
        )
        self._exact_strike_fk_projection_records[projection] = record
        self._exact_strike_fk_current_projection = projection
        self._exact_strike_fk_shot_index = copied_shot_index.detach().clone()
        return projection

    def _clear_exact_strike_fk_observation_rows(
        self, mask: torch.Tensor
    ) -> None:
        """Erase selected true-reset rows without host-observing the mask."""

        projection = self._exact_strike_fk_current_projection
        if projection is None:
            return
        record = self._exact_strike_fk_projection_records.get(projection)
        if record is None or record.stage != "issued":
            return
        record.source_step.masked_fill_(mask, -1)
        for name in _INT_KEY_FIELDS:
            getattr(record.task_key, name).masked_fill_(
                mask, -1 if name == "env_id" else 0
            )
        for name in _DIGEST_KEY_FIELDS:
            _masked_zero_(getattr(record.task_key, name), mask)
        record.shot_index.masked_fill_(mask, 0)
        record.identity_complete.masked_fill_(mask, False)
        record.exact_strike_reached.masked_fill_(mask, False)
        _masked_zero_(record.achieved_position, mask)
        _masked_zero_(record.achieved_velocity, mask)
        _masked_zero_(record.achieved_face_normal, mask)
        if self._exact_strike_fk_shot_index is not None:
            self._exact_strike_fk_shot_index.masked_fill_(mask, 0)

    def _advance_previous(
        self, source_step: torch.Tensor, key: DeviceActionTaskKey
    ) -> None:
        active = self._armed | self._pending | self._paid
        stored_step = torch.where(
            self._armed, self._arm_source_step, self._cache_source_step
        )
        advanced = active & (source_step > stored_step)
        regressed = active & (source_step < stored_step)
        key_mutated = (
            self._armed & self._key_binding_fault(key, arm=True)
        ) | (self._pending & self._key_binding_fault(key, arm=False))

        self._set_fault(regressed, FAULT_STEP_REGRESSION)
        self._set_fault(self._armed & advanced, FAULT_ARM_EXPIRED)
        self._set_fault(self._pending & advanced, FAULT_PENDING_EXPIRED)
        self._set_fault(
            (self._armed | self._pending) & advanced & key_mutated,
            FAULT_RESET_WRAP_PENDING,
        )

        retire = advanced | regressed
        self._clear_arm(retire & self._armed)
        self._clear_cache(retire & (self._pending | self._paid))

    def arm(
        self,
        arm_mask: torch.Tensor,
        *,
        source_step: torch.Tensor,
        task_key: DeviceActionTaskKey | Mapping[str, torch.Tensor],
        target_position: torch.Tensor,
        target_velocity: torch.Tensor,
        target_face_normal: torch.Tensor,
        ball_position: torch.Tensor,
        ball_velocity: torch.Tensor,
        validity: torch.Tensor,
    ) -> None:
        """Stage one all-env pre-physics fact without a host synchronization.

        The runtime owner must call this once on every transition, including
        transitions whose ``arm_mask`` is all false.  That fixed-cadence call
        retires fully paid prior facts and faults stale ARMED/PENDING facts.
        """

        self._require_hot_mutation_permitted()
        # The prior source capability is valid only at its exact post-physics,
        # pre-Reward publication seam.  The next post-Reward arm makes even an
        # otherwise numerically matching replay stale.
        self._invalidate_exact_strike_fk_observation_projection()

        mask = self._mask(arm_mask, "arm_mask")
        steps = self._steps(source_step)
        key = self._key(task_key)
        vectors = {
            "target_position": self._vector(target_position, "target_position"),
            "target_velocity": self._vector(target_velocity, "target_velocity"),
            "target_face_normal": self._vector(
                target_face_normal, "target_face_normal"
            ),
            "ball_position": self._vector(ball_position, "ball_position"),
            "ball_velocity": self._vector(ball_velocity, "ball_velocity"),
        }
        valid = self._mask(validity, "validity")

        self._advance_previous(steps, key)
        collision = mask & (self._armed | self._pending | self._paid)
        invalid = mask & (
            (steps < 0) | self._key_value_fault(key) | self._vectors_nonfinite(vectors)
        )
        self._set_fault(collision, FAULT_ARM_COLLISION)
        self._set_fault(invalid, FAULT_INVALID_ARM)
        safe = mask & ~collision & ~invalid & (self._fault_bits == 0)

        _masked_copy_(self._arm_source_step, steps, safe)
        for name in _INT_KEY_FIELDS:
            _masked_copy_(self._arm_key_ints[name], getattr(key, name), safe)
        for name in _DIGEST_KEY_FIELDS:
            _masked_copy_(self._arm_key_digests[name], getattr(key, name), safe)
        for name, value in vectors.items():
            _masked_copy_(self._arm_vectors[name], value, safe)
        _masked_copy_(self._arm_validity, valid, safe)
        self._armed.bitwise_or_(safe)
        self._armed_total.add_(safe.to(dtype=torch.int64).sum())
        self._advance_mutation_version()

    def publish(
        self,
        publish_mask: torch.Tensor,
        *,
        source_step: torch.Tensor,
        task_key: DeviceActionTaskKey | Mapping[str, torch.Tensor],
        achieved_position: torch.Tensor,
        achieved_velocity: torch.Tensor,
        achieved_face_normal: torch.Tensor,
        observation_shot_index: torch.Tensor | None = None,
    ) -> ExactStrikeFkObservationProjection:
        """Publish one immutable all-env cache after physics and before reward."""

        # In the fresh Full-MDP this field is part of the causal writer ABI,
        # not something a later observation consumer may discover is missing.
        # Reject before cache, lifecycle, counters, faults, or mutation version
        # can change.  Legacy/diagnostic owners retain the optional argument so
        # their old focused contracts do not falsely claim fresh readiness.
        if (
            self._observation_projection_mode
            == OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP
            and observation_shot_index is None
        ):
            raise StrikeFactObservationIdentityHold(
                "fresh full-MDP R03 publication requires independent shot_index "
                "before publication mutation"
            )
        if observation_shot_index is not None and (
            not isinstance(observation_shot_index, torch.Tensor)
            or observation_shot_index.shape != (self.num_envs,)
            or observation_shot_index.dtype != torch.int64
            or observation_shot_index.device != self.device
        ):
            raise StrikeFactDeviceError(
                "observation_shot_index must be device-local int64 [num_envs]"
            )

        self._require_hot_mutation_permitted()

        mask = self._mask(publish_mask, "publish_mask")
        steps = self._steps(source_step)
        key = self._key(task_key)
        achieved = {
            "achieved_position": self._vector(
                achieved_position, "achieved_position"
            ),
            "achieved_velocity": self._vector(
                achieved_velocity, "achieved_velocity"
            ),
            "achieved_face_normal": self._vector(
                achieved_face_normal, "achieved_face_normal"
            ),
        }

        expected = self._armed
        unarmed = mask & ~expected
        missing = expected & ~mask
        binding = mask & expected & self._binding_fault(steps, key, arm=True)
        invalid = mask & (
            (steps < 0) | self._key_value_fault(key) | self._vectors_nonfinite(achieved)
        )
        self._set_fault(unarmed, FAULT_UNARMED_PUBLISH)
        self._set_fault(missing, FAULT_MISSING_PUBLISH)
        self._set_fault(binding, FAULT_PUBLISH_BINDING)
        self._set_fault(invalid, FAULT_INVALID_PUBLISH)
        safe = (
            mask
            & expected
            & ~binding
            & ~invalid
            & (self._fault_bits == 0)
        )

        _masked_copy_(self._cache_source_step, self._arm_source_step, safe)
        for name in _INT_KEY_FIELDS:
            _masked_copy_(self._cache_key_ints[name], self._arm_key_ints[name], safe)
        for name in _DIGEST_KEY_FIELDS:
            _masked_copy_(
                self._cache_key_digests[name], self._arm_key_digests[name], safe
            )
        for name in _TARGET_FIELDS:
            _masked_copy_(self._cache_vectors[name], self._arm_vectors[name], safe)
        for name, value in achieved.items():
            _masked_copy_(self._cache_vectors[name], value, safe)
        _masked_copy_(self._cache_validity, self._arm_validity, safe)
        self._viewed_mask.masked_fill_(safe, 0)
        self._paid_mask.masked_fill_(safe, 0)
        _masked_zero_(self._payment_values, safe)
        self._pending.bitwise_or_(safe)
        self._visible.bitwise_or_(safe)
        self._published_total.add_(safe.to(dtype=torch.int64).sum())

        # ``publish`` is the unique all-env DoneTerm call for this transition.
        # Every armed row either moved to PENDING or failed closed and is erased.
        self._clear_arm(expected)
        projection = self._mint_exact_strike_fk_observation_projection(
            published_mask=safe,
            shot_index=observation_shot_index,
        )
        self._advance_mutation_version()
        return projection

    def require_owned_exact_strike_fk_observation_projection(
        self,
        projection: ExactStrikeFkObservationProjection,
        *,
        current_source_step: torch.Tensor,
        current_task_key: DeviceActionTaskKey | Mapping[str, torch.Tensor],
        current_shot_index: torch.Tensor,
    ) -> ExactStrikeFkObservationView:
        """Authenticate one exact publication and cross-join current identity.

        This is a non-consuming source projection.  It does not invoke or
        simulate :meth:`view`, change visible/pending/paid lifecycle state,
        alter payment bits, or advance the PPO mutation version.  Every tensor
        comparison stays on device.  Foreign, replayed, stale, cloned, and
        wrong-current-identity rows are rejected before facts become visible.
        """

        if type(projection) is not ExactStrikeFkObservationProjection:
            raise StrikeFactDeviceError(
                "exact-strike FK observation projection is forged or cloned"
            )
        record = self._exact_strike_fk_projection_records.get(projection)
        if (
            record is None
            or record.projection is not projection
            or projection is not self._exact_strike_fk_current_projection
            or record.stage != "issued"
            or record.publication_sequence
            != self._exact_strike_fk_publication_sequence
        ):
            raise StrikeFactDeviceError(
                "exact-strike FK observation projection is foreign, replayed, or stale"
            )
        if not record.shot_index_supplied:
            raise StrikeFactObservationIdentityHold(
                "R03 exact-strike FK publication has no independent shot_index; "
                "FreshObservationOwner must remain HOLD"
            )

        steps = self._steps(current_source_step)
        key = self._key(current_task_key)
        if (
            not isinstance(current_shot_index, torch.Tensor)
            or current_shot_index.shape != (self.num_envs,)
            or current_shot_index.dtype != torch.int64
            or current_shot_index.device != self.device
        ):
            raise StrikeFactDeviceError(
                "current_shot_index must be device-local int64 [num_envs]"
            )
        identity_mismatch = steps != record.source_step
        for name in _INT_KEY_FIELDS:
            identity_mismatch = identity_mismatch | (
                getattr(key, name) != getattr(record.task_key, name)
            )
        for name in _DIGEST_KEY_FIELDS:
            identity_mismatch = identity_mismatch | torch.any(
                getattr(key, name) != getattr(record.task_key, name), dim=1
            )
        identity_mismatch = identity_mismatch | (
            current_shot_index != record.shot_index
        )
        # Rows with no exact-strike publication are ordinary negative events,
        # not identity failures.  Counting them as rejects would turn a sparse
        # event into an always-failing gate.
        identity_rejected = record.exact_strike_reached & (
            identity_mismatch | ~record.identity_complete
        )
        reached = record.exact_strike_reached & ~identity_rejected

        def clone_zero_rejected(value: torch.Tensor) -> torch.Tensor:
            result = value.detach().clone()
            result.masked_fill_(_expand(identity_rejected, result.ndim), 0)
            return result

        result = ExactStrikeFkObservationView(
            source_step=record.source_step.detach().clone(),
            task_key=self._clone_device_task_key(record.task_key),
            shot_index=record.shot_index.detach().clone(),
            identity_complete=record.identity_complete.detach().clone(),
            identity_rejected=identity_rejected.detach().clone(),
            exact_strike_reached=reached.detach().clone(),
            achieved_position=clone_zero_rejected(record.achieved_position),
            achieved_velocity=clone_zero_rejected(record.achieved_velocity),
            achieved_face_normal=clone_zero_rejected(record.achieved_face_normal),
        )
        # One exact FreshObservationOwner may latch the source facts.  Repeat
        # use is a replay even though this consumption is intentionally outside
        # the Reward view/payment/mutation protocol.
        record.stage = "consumed"
        return result

    def view(self, consumer: str) -> SharedStrikeFactDeviceView:
        """Book one named read and return the single shared cache object."""

        self._require_hot_mutation_permitted()

        publication = self._active_full_mdp_reward_publication
        if publication is not None:
            record = self._full_mdp_pre_reward_records.get(publication)
            ordinal = len(record.payment_verdicts) if record is not None else -1
            if (
                record is None
                or record.stage != "issued"
                or ordinal >= len(FULL_MDP_REWARD_CONSUMERS)
                or FULL_MDP_REWARD_CONSUMERS[ordinal] != consumer
            ):
                raise StrikeFactDeviceError(
                    "R03 full-MDP Reward views must follow the exact ordered ten"
                )

        _, bit = self._consumer(consumer)
        live = self._pending | self._paid
        already = (self._viewed_mask & bit) != 0
        duplicate = live & already
        self._set_fault(duplicate, FAULT_DUPLICATE_VIEW)
        first = self._pending & ~already & (self._fault_bits == 0)
        self._viewed_mask.bitwise_or_(first.to(dtype=torch.int64) * bit)
        self._advance_mutation_version()
        return self._shared_view

    def record_payment(self, consumer: str, payment: torch.Tensor) -> None:
        """Commit one all-env payment vector to the consumer's once-only bit."""

        self._require_hot_mutation_permitted()

        publication = self._active_full_mdp_reward_publication
        reward_record = (
            self._full_mdp_pre_reward_records.get(publication)
            if publication is not None
            else None
        )
        reward_ordinal = (
            len(reward_record.payment_verdicts)
            if reward_record is not None
            else -1
        )
        if publication is not None and (
            reward_record is None
            or reward_record.stage != "issued"
            or reward_ordinal >= len(FULL_MDP_REWARD_CONSUMERS)
            or FULL_MDP_REWARD_CONSUMERS[reward_ordinal] != consumer
        ):
            raise StrikeFactDeviceError(
                "R03 full-MDP Reward payments must follow the exact ordered ten"
            )

        index, bit = self._consumer(consumer)
        values = self._payment(payment)
        live = self._pending | self._paid
        viewed = (self._viewed_mask & bit) != 0
        previously_paid = (self._paid_mask & bit) != 0
        first_attempt = self._pending & ~previously_paid
        before_view = first_attempt & ~viewed
        duplicate = live & previously_paid
        nonfinite = ~torch.isfinite(values)
        outside = ~live & torch.isfinite(values) & (values != 0)

        self._set_fault(before_view, FAULT_PAYMENT_BEFORE_VIEW)
        self._set_fault(duplicate, FAULT_DUPLICATE_PAYMENT)
        self._set_fault(nonfinite, FAULT_INVALID_PAYMENT)
        self._set_fault(outside, FAULT_PAYMENT_OUTSIDE_EVENT)

        good = (
            first_attempt
            & viewed
            & ~nonfinite
            & (self._fault_bits == 0)
        )
        self._payment_values[:, index].copy_(
            torch.where(good, values, self._payment_values[:, index])
        )
        # The first call consumes the bit even when it is invalid.  The sticky
        # fault denies the boundary while preventing an invalid retry/replay.
        self._paid_mask.bitwise_or_(first_attempt.to(dtype=torch.int64) * bit)
        self._payment_totals[index].add_(
            first_attempt.to(dtype=torch.int64).sum()
        )
        complete = (
            self._pending
            & (self._viewed_mask == FULL_CONSUMER_MASK)
            & (self._paid_mask == FULL_CONSUMER_MASK)
            & (self._fault_bits == 0)
        )
        self._pending.bitwise_and_(~complete)
        self._paid.bitwise_or_(complete)
        self._visible.bitwise_and_(~complete)
        self._advance_mutation_version()

        # A production Reward verdict is minted by the real mutation above,
        # never by the top graph and never from a caller boolean.  Legacy and
        # diagnostic direct calls intentionally continue returning ``None``.
        if publication is None:
            return None
        assert reward_record is not None
        verdict = object.__new__(R03FullMdpRewardPaymentVerdict)
        self._full_mdp_payment_records[verdict] = _R03PaymentRecord(
            verdict=verdict,
            publication=publication,
            runtime_owner=reward_record.runtime_owner,
            control_step=reward_record.control_step,
            consumer=consumer,
            payment_ordinal=reward_ordinal,
            mutation_version=self._mutation_version,
        )
        reward_record.payment_verdicts.append(verdict)
        return verdict

    @staticmethod
    def _full_mdp_control_step(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StrikeFactDeviceError(
                "R03 full-MDP Reward control_step must be a non-negative int"
            )
        return value

    def _require_active_full_mdp_pre_reward_record(
        self,
        publication: object,
        *,
        control_step: object,
        runtime_owner: object,
    ) -> _R03PreRewardRecord:
        record = self._full_mdp_pre_reward_records.get(publication)  # type: ignore[arg-type]
        step = self._full_mdp_control_step(control_step)
        if (
            type(publication) is not R03FullMdpPreRewardPublication
            or publication is not self._active_full_mdp_reward_publication
            or record is None
            or record.publication is not publication
            or record.stage != "issued"
            or record.runtime_owner is not runtime_owner
            or record.control_step != step
            or record.projection_sequence
            != self._exact_strike_fk_publication_sequence
        ):
            raise StrikeFactDeviceError(
                "R03 pre-Reward publication is foreign, stale, or wrong-step"
            )
        return record

    def publish_full_mdp_pre_reward(
        self, *, control_step: int, runtime_owner: object
    ) -> R03FullMdpPreRewardPublication:
        """Publish one exact R03 Reward epoch from the real strike-fact writer."""

        step = self._full_mdp_control_step(control_step)
        if runtime_owner is None:
            raise StrikeFactDeviceError("R03 pre-Reward runtime owner is absent")
        if self._active_full_mdp_reward_publication is not None:
            raise StrikeFactDeviceError(
                "previous R03 full-MDP Reward cycle is still open"
            )
        projection = self._exact_strike_fk_current_projection
        projection_record = self._exact_strike_fk_projection_records.get(projection)
        if (
            self._observation_projection_mode
            != OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP
            or projection is None
            or projection_record is None
            or projection_record.stage != "issued"
            or not projection_record.shot_index_supplied
            or projection_record.publication_sequence
            != self._exact_strike_fk_publication_sequence
            or step < 0
        ):
            raise StrikeFactObservationIdentityHold(
                "R03 pre-Reward requires the real fresh full-key/shot-index publication"
            )
        publication = object.__new__(R03FullMdpPreRewardPublication)
        record = _R03PreRewardRecord(
            publication=publication,
            runtime_owner=runtime_owner,
            control_step=step,
            projection_sequence=projection_record.publication_sequence,
            eligible=self._visible.detach().clone(),
            validity=self._cache_validity.detach().clone(),
            source_step=self._cache_source_step.detach().clone(),
            task_key=self._current_cache_task_key_clone(),
            shot_index=projection_record.shot_index.detach().clone(),
            payment_verdicts=[],
        )
        self._full_mdp_pre_reward_records.clear()
        self._full_mdp_pre_reward_records[publication] = record
        self._full_mdp_payment_records.clear()
        self._full_mdp_close_records.clear()
        self._active_full_mdp_reward_publication = publication
        return publication

    def require_owned_full_mdp_pre_reward(
        self,
        publication: object,
        *,
        control_step: int,
        runtime_owner: object,
    ) -> R03FullMdpPreRewardView:
        """Authenticate and clone only the causal R03 publication facts."""

        record = self._require_active_full_mdp_pre_reward_record(
            publication,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        false = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return R03FullMdpPreRewardView(
            terminated=false,
            time_out=false.clone(),
            eligible=record.eligible.detach().clone(),
            validity=record.validity.detach().clone(),
            source_step=record.source_step.detach().clone(),
            task_key=self._clone_device_task_key(record.task_key),
            shot_index=record.shot_index.detach().clone(),
        )

    def require_owned_full_mdp_reward_payment(
        self,
        verdict: object,
        *,
        consumer: str,
        control_step: int,
        runtime_owner: object,
    ) -> R03FullMdpRewardPaymentVerdict:
        """Validate the opaque verdict minted by the actual payment mutation."""

        try:
            payment = self._full_mdp_payment_records.get(verdict)  # type: ignore[arg-type]
        except TypeError:
            payment = None
        record = self._require_active_full_mdp_pre_reward_record(
            self._active_full_mdp_reward_publication,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        if (
            type(verdict) is not R03FullMdpRewardPaymentVerdict
            or payment is None
            or payment.verdict is not verdict
            or payment.publication is not record.publication
            or payment.runtime_owner is not runtime_owner
            or payment.control_step != record.control_step
            or payment.consumer != consumer
            or payment.stage != "issued"
            or payment.payment_ordinal >= len(record.payment_verdicts)
            or record.payment_verdicts[payment.payment_ordinal] is not verdict
            or FULL_MDP_REWARD_CONSUMERS[payment.payment_ordinal] != consumer
        ):
            raise StrikeFactDeviceError(
                "R03 Reward payment verdict is foreign, stale, or reordered"
            )
        return verdict

    def close_full_mdp_reward_cycle(
        self,
        *,
        control_step: int,
        pre_reward_publication: object,
        ordered_consumers: object,
        ordered_payment_verdicts: object,
        runtime_owner: object,
    ) -> R03FullMdpRewardCloseReceipt:
        """Close only after the exact ten real payment verdicts are present."""

        record = self._require_active_full_mdp_pre_reward_record(
            pre_reward_publication,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        verdicts = tuple(record.payment_verdicts)
        if (
            type(ordered_consumers) is not tuple
            or ordered_consumers != FULL_MDP_REWARD_CONSUMERS
            or type(ordered_payment_verdicts) is not tuple
            or ordered_payment_verdicts != verdicts
            or len(verdicts) != len(FULL_MDP_REWARD_CONSUMERS)
            or len({id(value) for value in verdicts}) != len(verdicts)
        ):
            raise StrikeFactDeviceError(
                "R03 Reward close requires the exact ordered ten real payments"
            )
        for consumer, verdict in zip(FULL_MDP_REWARD_CONSUMERS, verdicts):
            self.require_owned_full_mdp_reward_payment(
                verdict,
                consumer=consumer,
                control_step=record.control_step,
                runtime_owner=runtime_owner,
            )
        receipt = object.__new__(R03FullMdpRewardCloseReceipt)
        self._full_mdp_close_records[receipt] = _R03CloseRecord(
            receipt=receipt,
            publication=record.publication,
            runtime_owner=runtime_owner,
            control_step=record.control_step,
            payment_verdicts=verdicts,
        )
        record.stage = "closed"
        for verdict in verdicts:
            self._full_mdp_payment_records[verdict].stage = "closed"
        self._active_full_mdp_reward_publication = None
        return receipt

    def require_owned_full_mdp_reward_close(
        self,
        receipt: object,
        *,
        control_step: int,
        runtime_owner: object,
    ) -> R03FullMdpRewardCloseReceipt:
        """Authenticate the one close produced from this owner's ten verdicts."""

        record = self._full_mdp_close_records.get(receipt)  # type: ignore[arg-type]
        step = self._full_mdp_control_step(control_step)
        if (
            type(receipt) is not R03FullMdpRewardCloseReceipt
            or record is None
            or record.receipt is not receipt
            or record.runtime_owner is not runtime_owner
            or record.control_step != step
            or record.stage != "issued"
            or len(record.payment_verdicts) != len(FULL_MDP_REWARD_CONSUMERS)
        ):
            raise StrikeFactDeviceError(
                "R03 Reward close receipt is foreign, stale, or wrong-step"
            )
        return receipt

    def reset(self, reset_mask: torch.Tensor) -> None:
        """Clear reset rows without allowing reset/wrap to erase unsettled work."""

        self._require_no_armed_epoch_fact(operation="reset")
        self._require_no_open_full_mdp_reward_cycle(operation="reset")
        self._require_hot_mutation_permitted()

        mask = self._mask(reset_mask, "reset_mask")
        unsettled = mask & (self._armed | self._pending)
        self._set_fault(unsettled, FAULT_RESET_UNSETTLED)
        self._clear_arm(mask & self._armed)
        self._clear_cache(mask & (self._pending | self._paid))
        self._clear_exact_strike_fk_observation_rows(mask)
        self._reset_total.add_(mask.to(dtype=torch.int64).sum())
        self._advance_mutation_version()

    def _key_invalid_from_buffers(
        self,
        active: torch.Tensor,
        ints: Mapping[str, torch.Tensor],
        digests: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        invalid = active & (
            (ints["env_id"] != self._env_ids)
            | (ints["reset_generation"] < 1)
            | (ints["swing_generation"] < 0)
            | (ints["action_uid"] < 1)
            | (ints["action_uid"] > _MAX_ACTION_UID)
            | (ints["action_slot"] < 0)
        )
        for name in _DIGEST_KEY_FIELDS:
            invalid = invalid | (active & torch.all(digests[name] == 0, dim=1))
        return invalid

    def _nonzero_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(self.num_envs, -1)
        return torch.any(flat != 0, dim=1)

    def _nonfinite_rows(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        result = torch.zeros_like(self._armed)
        for tensor in tensors.values():
            result = result | ~torch.all(
                torch.isfinite(tensor.reshape(self.num_envs, -1)), dim=1
            )
        return result

    def _invariant_masks(self) -> tuple[tuple[str, torch.Tensor], ...]:
        idle = ~(self._armed | self._pending | self._paid)
        active_cache = self._pending | self._paid
        overlap = (
            (self._armed & self._pending)
            | (self._armed & self._paid)
            | (self._pending & self._paid)
        )
        arm_dirty = ~self._armed & (
            (self._arm_source_step != -1)
            | (self._arm_key_ints["env_id"] != -1)
            | self._arm_validity
        )
        for name in _INT_KEY_FIELDS[1:]:
            arm_dirty = arm_dirty | (~self._armed & (self._arm_key_ints[name] != 0))
        for tensor in self._arm_key_digests.values():
            arm_dirty = arm_dirty | (~self._armed & self._nonzero_rows(tensor))
        for tensor in self._arm_vectors.values():
            arm_dirty = arm_dirty | (~self._armed & self._nonzero_rows(tensor))

        cache_dirty = ~active_cache & (
            (self._cache_source_step != -1)
            | (self._cache_key_ints["env_id"] != -1)
            | self._cache_validity
            | (self._viewed_mask != 0)
            | (self._paid_mask != 0)
            | self._nonzero_rows(self._payment_values)
        )
        for name in _INT_KEY_FIELDS[1:]:
            cache_dirty = cache_dirty | (
                ~active_cache & (self._cache_key_ints[name] != 0)
            )
        for tensor in self._cache_key_digests.values():
            cache_dirty = cache_dirty | (~active_cache & self._nonzero_rows(tensor))
        for tensor in self._cache_vectors.values():
            cache_dirty = cache_dirty | (~active_cache & self._nonzero_rows(tensor))

        bad_bits = ((self._viewed_mask | self._paid_mask) & ~FULL_CONSUMER_MASK) != 0
        paid_without_view = (self._paid_mask & ~self._viewed_mask) != 0
        pending_complete = self._pending & (
            (self._viewed_mask == FULL_CONSUMER_MASK)
            & (self._paid_mask == FULL_CONSUMER_MASK)
        )
        paid_incomplete = self._paid & (
            (self._viewed_mask != FULL_CONSUMER_MASK)
            | (self._paid_mask != FULL_CONSUMER_MASK)
        )
        bits = torch.tensor(
            [1 << index for index in range(CONSUMER_COUNT)],
            dtype=torch.int64,
            device=self.device,
        )
        paid_columns = (self._paid_mask.unsqueeze(1) & bits.unsqueeze(0)) != 0
        unpaid_value = torch.any(~paid_columns & (self._payment_values != 0), dim=1)
        invalid_step = (self._armed & (self._arm_source_step < 0)) | (
            active_cache & (self._cache_source_step < 0)
        )
        invalid_key = self._key_invalid_from_buffers(
            self._armed, self._arm_key_ints, self._arm_key_digests
        ) | self._key_invalid_from_buffers(
            active_cache, self._cache_key_ints, self._cache_key_digests
        )
        nonfinite = (
            self._armed & self._nonfinite_rows(self._arm_vectors)
        ) | (active_cache & self._nonfinite_rows(self._cache_vectors))
        visible_mismatch = self._visible != self._pending
        idle_ledger = idle & (
            (self._viewed_mask != 0)
            | (self._paid_mask != 0)
            | self._nonzero_rows(self._payment_values)
        )
        accounting_invalid_scalar = (
            (self._armed_total < 0)
            | (self._published_total < 0)
            | (self._published_total > self._armed_total)
            | torch.any(self._payment_totals < 0)
            | torch.any(self._payment_totals > self._published_total)
            | (self._reset_total < 0)
        )
        accounting_invalid = accounting_invalid_scalar.expand(self.num_envs)
        return (
            ("state_overlap", overlap),
            ("arm_storage_dirty", arm_dirty),
            ("cache_storage_dirty", cache_dirty),
            ("consumer_bits_out_of_range", bad_bits),
            ("paid_without_view", paid_without_view),
            ("pending_marked_complete", pending_complete),
            ("paid_marked_incomplete", paid_incomplete),
            ("unpaid_payment_value", unpaid_value),
            ("invalid_source_step", invalid_step),
            ("invalid_task_key", invalid_key),
            ("nonfinite_fact", nonfinite),
            ("visible_pending_mismatch", visible_mismatch),
            ("idle_ledger_dirty", idle_ledger),
            ("accounting_invalid", accounting_invalid),
        )

    @staticmethod
    def _exact_pre_optimizer_index(value: object, *, label: str) -> int:
        if type(value) is not int or value < 0:
            raise StrikeFactDeviceError(
                f"{label} must be an exact non-negative integer"
            )
        return value

    def _pre_optimizer_device_values(self) -> torch.Tensor:
        """Build the complete R03 row without observing a tensor on host."""

        invariant_masks = self._invariant_masks()
        invariant_names = tuple(name for name, _ in invariant_masks)
        if invariant_names != INVARIANT_NAMES:
            raise StrikeFactDeviceError(
                "R03 invariant order differs from the frozen global row schema"
            )
        fault_counts = torch.stack(
            [
                ((self._fault_bits & bit) != 0).to(dtype=torch.int64).sum()
                for _, bit in FAULTS
            ]
        )
        state_counts = torch.stack(
            [
                (~(self._armed | self._pending | self._paid))
                .to(dtype=torch.int64)
                .sum(),
                self._armed.to(dtype=torch.int64).sum(),
                self._pending.to(dtype=torch.int64).sum(),
                self._paid.to(dtype=torch.int64).sum(),
            ]
        )
        invariant_counts = torch.stack(
            [mask.to(dtype=torch.int64).sum() for _, mask in invariant_masks]
        )
        mutation_version = torch.tensor(
            [self._mutation_version],
            dtype=torch.int64,
            device=self.device,
        )
        values = torch.cat(
            (
                mutation_version,
                fault_counts.sum().reshape(1),
                invariant_counts.sum().reshape(1),
                fault_counts,
                state_counts,
                invariant_counts,
                self._armed_total.reshape(1),
                self._published_total.reshape(1),
                self._payment_totals,
                self._reset_total.reshape(1),
            )
        ).contiguous()
        if values.shape != (len(PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES),):
            raise StrikeFactDeviceError("R03 global row width differs")
        return values

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Mint the R03 part of the sole global pre-optimizer device row."""

        self._require_no_armed_epoch_fact(operation="global drain")
        self._require_no_open_full_mdp_reward_cycle(operation="global drain")
        self._require_pre_optimizer_operable()
        update = self._exact_pre_optimizer_index(
            update_index,
            label="update_index",
        )
        completed = self._exact_pre_optimizer_index(
            completed_environment_steps,
            label="completed_environment_steps",
        )
        if update <= self._last_drained_update:
            raise StrikeFactDeviceError(
                "R03 PPO boundary update_index must be strictly increasing"
            )
        if self._active_pre_optimizer_pack is not None:
            self._poison_pre_optimizer(
                "duplicate R03 pre-optimizer prepare while one pack is active"
            )
            raise StrikeFactDeviceError(
                "one R03 pre-optimizer device pack is already active"
            )
        if (
            getattr(authority, "owner_kind", None)
            != PRE_OPTIMIZER_PPO_BOUNDARY_OWNER_KIND
            or tuple(getattr(authority, "field_names", ()))
            != PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES
            or getattr(authority, "expected_width", None)
            != len(PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES)
        ):
            raise StrikeFactDeviceError(
                "R03 global drain authority schema differs from the complete portable row"
            )
        mint = getattr(authority, "mint_device_pack", None)
        require_owned_ack = getattr(authority, "require_owned_ack", None)
        if __package__:
            from . import action_ball_full_mdp_ppo_drain as drain
        else:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_ppo_drain as drain,
            )
        if (
            type(authority) is not drain.LeafDevicePackAuthority
            or not callable(mint)
            or getattr(mint, "__self__", None) is not authority
            or getattr(mint, "__func__", None)
            is not drain.LeafDevicePackAuthority.mint_device_pack
            or not callable(require_owned_ack)
            or getattr(require_owned_ack, "__self__", None) is not authority
            or getattr(require_owned_ack, "__func__", None)
            is not drain.LeafDevicePackAuthority.require_owned_ack
        ):
            raise StrikeFactDeviceError(
                "R03 global drain exact authority API differs"
            )

        # Adoption is monotonic even if this clean prepare is later aborted:
        # one owner must never alternate global and legacy drains.
        self._pre_optimizer_global_protocol_adopted = True
        values = self._pre_optimizer_device_values()
        pack = mint(leaf=self, values=values)
        self._active_pre_optimizer_pack = _PreparedPreOptimizerPpoBoundaryPack(
            pack=pack,
            authority=authority,
            update_index=update,
            completed_environment_steps=completed,
            mutation_version=self._mutation_version,
        )
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack: object) -> None:
        """Release one exact pre-transfer pack without changing R03 facts."""

        self._require_pre_optimizer_operable()
        active = self._active_pre_optimizer_pack
        if active is None or pack is not active.pack:
            raise StrikeFactDeviceError(
                "R03 pre-optimizer abort pack is foreign, stale, or copied"
            )
        self._active_pre_optimizer_pack = None

    @staticmethod
    def _owner_row_values(owner_row: object) -> dict[str, int]:
        if (
            getattr(owner_row, "owner_kind", None)
            != PRE_OPTIMIZER_PPO_BOUNDARY_OWNER_KIND
        ):
            raise StrikeFactDeviceError("R03 global owner row has the wrong kind")
        values = getattr(owner_row, "values", None)
        if not isinstance(values, tuple):
            raise StrikeFactDeviceError("R03 global owner row values must be a tuple")
        names: list[str] = []
        result: dict[str, int] = {}
        for item in values:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
            ):
                raise StrikeFactDeviceError(
                    "R03 global owner row contains a non-scalar field"
                )
            name, value = item
            names.append(name)
            result[name] = value
        if tuple(names) != PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES:
            raise StrikeFactDeviceError(
                "R03 global owner row fields differ from the complete portable schema"
            )
        return result

    def _portable_receipt_from_owner_row(
        self,
        *,
        active: _PreparedPreOptimizerPpoBoundaryPack,
        receipt: object,
        owner_row: object,
    ) -> StrikeFactBoundaryReceipt:
        row = self._owner_row_values(owner_row)
        owner_rows = getattr(receipt, "owner_rows", None)
        if (
            not isinstance(owner_rows, tuple)
            or sum(value is owner_row for value in owner_rows) != 1
            or getattr(receipt, "num_envs", None) != self.num_envs
            or getattr(receipt, "device_to_host_transfers", None) != 1
            or getattr(receipt, "acknowledged", None) is not False
        ):
            raise StrikeFactDeviceError(
                "R03 owner row is not the exact unacknowledged global receipt row"
            )
        update = self._exact_pre_optimizer_index(
            getattr(receipt, "update_index", None),
            label="global receipt update_index",
        )
        completed = self._exact_pre_optimizer_index(
            getattr(receipt, "completed_environment_steps", None),
            label="global receipt completed_environment_steps",
        )
        sequence = self._exact_pre_optimizer_index(
            getattr(receipt, "drain_sequence", None),
            label="global receipt drain_sequence",
        )
        if update != active.update_index or completed != active.completed_environment_steps:
            raise StrikeFactDeviceError(
                "R03 global receipt boundary differs from the prepared pack"
            )
        if sequence != self._drain_sequence + 1:
            raise StrikeFactDeviceError(
                "R03 global drain sequence did not advance exactly once"
            )
        if (
            row["mutation_version"] != active.mutation_version
            or row["mutation_version"] != self._mutation_version
        ):
            raise StrikeFactDeviceError(
                "R03 mutated after its global device row was prepared"
            )

        fault_values = tuple(
            (name, row[f"fault_{name}_count"])
            for name, _ in FAULTS
        )
        state_values = tuple(
            (name, row[f"state_{name}_count"])
            for name in STATE_NAMES
        )
        invariant_names = tuple(name for name, _ in self._invariant_masks())
        invariant_values = tuple(
            (name, row[f"invariant_{name}_count"])
            for name in invariant_names
        )
        if row["fault_count"] != sum(value for _, value in fault_values):
            raise StrikeFactDeviceError(
                "R03 aggregate fault count differs from its portable detail"
            )
        if row["invariant_count"] != sum(
            value for _, value in invariant_values
        ):
            raise StrikeFactDeviceError(
                "R03 aggregate invariant count differs from its portable detail"
            )
        if row["fault_count"] != 0 or row["invariant_count"] != 0:
            raise StrikeFactDeviceError(
                "R03 faulted row cannot acknowledge an optimizer update"
            )

        return StrikeFactBoundaryReceipt(
            schema_version=SCHEMA_VERSION,
            update_index=update,
            drain_sequence=sequence,
            mutation_version=row["mutation_version"],
            num_envs=self.num_envs,
            consumers=self.consumers,
            fault_counts=fault_values,
            state_counts=state_values,
            invariant_counts=invariant_values,
            armed_total=row["armed_total"],
            published_total=row["published_total"],
            payment_totals=tuple(
                row[f"payment_{name}_total"]
                for name in STRIKE_FACT_CONSUMERS
            ),
            reset_total=row["reset_total"],
            checkpoint_safe=True,
            device_to_host_transfers=1,
            runtime_integrated=RUNTIME_INTEGRATED,
            cuda_profiled=CUDA_PROFILED,
            formal_exact_resume_integrated=FORMAL_EXACT_RESUME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Materialize the portable receipt after the optimizer callbacks."""

        self._require_pre_optimizer_operable()
        active = self._active_pre_optimizer_pack
        try:
            if active is None or pack is not active.pack:
                raise StrikeFactDeviceError(
                    "R03 optimizer acknowledgement pack is foreign, stale, or copied"
                )
            # Prove the exact coordinator and optimizer-return window before
            # reading any R03 business facts from the decoded row.
            active.authority.require_owned_ack(
                leaf=self,
                pack=pack,
                receipt=receipt,
                owner_row=owner_row,
            )
            portable = self._portable_receipt_from_owner_row(
                active=active,
                receipt=receipt,
                owner_row=owner_row,
            )
        except BaseException as exc:
            self._poison_pre_optimizer(
                f"R03 optimizer acknowledgement failed: {type(exc).__name__}: {exc}"
            )
            raise

        self._drain_sequence = portable.drain_sequence
        self._last_drained_update = portable.update_index
        self._last_receipt = portable
        self._last_global_receipt = receipt
        self._last_receipt_consumed = False
        self._last_globally_acknowledged_mutation_version = active.mutation_version
        self._checkpoint_requires_global_drain_ack = False
        self._pre_optimizer_global_protocol_adopted = True
        self._active_pre_optimizer_pack = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason: str) -> None:
        """Fail-stop this leaf after any irreversible global boundary failure."""

        self._poison_pre_optimizer(reason)

    @property
    def latest_pre_optimizer_ppo_boundary_receipt(
        self,
    ) -> StrikeFactBoundaryReceipt:
        """Return the exact portable receipt materialized by global ack."""

        self._require_pre_optimizer_operable()
        receipt = self._last_receipt
        if not self._pre_optimizer_global_protocol_adopted or receipt is None:
            raise StrikeFactDeviceError(
                "no acknowledged global R03 boundary receipt is available"
            )
        return receipt

    def require_owned_pre_optimizer_ppo_boundary_receipt(
        self,
        global_receipt: object,
    ) -> StrikeFactBoundaryReceipt:
        """Return the portable R03 audit from this exact global ACK only.

        The global receipt is an independently owned boundary identity.  A
        caller cannot authorize an old or locally materialized R03 receipt by
        repeating its public update/sequence fields.
        """

        self._require_pre_optimizer_operable()
        receipt = self._last_receipt
        if (
            not self._pre_optimizer_global_protocol_adopted
            or receipt is None
            or global_receipt is not self._last_global_receipt
            or getattr(global_receipt, "acknowledged", None) is not True
            or getattr(global_receipt, "update_index", None)
            != receipt.update_index
            or getattr(global_receipt, "drain_sequence", None)
            != receipt.drain_sequence
        ):
            raise StrikeFactDeviceError(
                "global receipt does not own the latest portable R03 audit"
            )
        return receipt

    def drain_ppo_boundary(self, *, update_index: int) -> StrikeFactBoundaryReceipt:
        """Perform the one packed host drain for a strictly newer PPO boundary."""

        self._require_no_armed_epoch_fact(operation="legacy drain")
        self._require_no_open_full_mdp_reward_cycle(operation="legacy drain")
        self._require_pre_optimizer_operable()
        if self._pre_optimizer_global_protocol_adopted:
            raise StrikeFactDeviceError(
                "legacy R03 drain is disabled after global PPO drain adoption"
            )

        if (
            isinstance(update_index, bool)
            or not isinstance(update_index, int)
            or update_index < 0
        ):
            raise StrikeFactDeviceError("update_index must be a non-negative integer")
        if update_index <= self._last_drained_update:
            raise StrikeFactDeviceError("PPO boundary update_index must be strictly increasing")

        invariant_masks = self._invariant_masks()
        fault_counts = torch.stack(
            [
                ((self._fault_bits & bit) != 0).to(dtype=torch.int64).sum()
                for _, bit in FAULTS
            ]
        )
        state_counts = torch.stack(
            [
                (~(self._armed | self._pending | self._paid)).to(dtype=torch.int64).sum(),
                self._armed.to(dtype=torch.int64).sum(),
                self._pending.to(dtype=torch.int64).sum(),
                self._paid.to(dtype=torch.int64).sum(),
            ]
        )
        invariant_counts = torch.stack(
            [mask.to(dtype=torch.int64).sum() for _, mask in invariant_masks]
        )
        packed = torch.cat(
            (
                fault_counts,
                state_counts,
                invariant_counts,
                self._armed_total.reshape(1),
                self._published_total.reshape(1),
                self._payment_totals,
                self._reset_total.reshape(1),
            )
        )
        # The sole runtime D2H observation.  Everything above is device-only.
        host_values = packed.detach().to(device="cpu").tolist()

        cursor = 0
        fault_values = tuple(
            (name, int(host_values[cursor + index]))
            for index, (name, _) in enumerate(FAULTS)
        )
        cursor += len(FAULTS)
        state_values = tuple(
            (name, int(host_values[cursor + index]))
            for index, name in enumerate(STATE_NAMES)
        )
        cursor += len(STATE_NAMES)
        invariant_values = tuple(
            (name, int(host_values[cursor + index]))
            for index, (name, _) in enumerate(invariant_masks)
        )
        cursor += len(invariant_masks)
        armed_total = int(host_values[cursor])
        published_total = int(host_values[cursor + 1])
        cursor += 2
        payment_totals = tuple(
            int(value) for value in host_values[cursor : cursor + CONSUMER_COUNT]
        )
        cursor += CONSUMER_COUNT
        reset_total = int(host_values[cursor])
        safe = all(count == 0 for _, count in fault_values + invariant_values)

        self._drain_sequence += 1
        receipt = StrikeFactBoundaryReceipt(
            schema_version=SCHEMA_VERSION,
            update_index=update_index,
            drain_sequence=self._drain_sequence,
            mutation_version=self._mutation_version,
            num_envs=self.num_envs,
            consumers=self.consumers,
            fault_counts=fault_values,
            state_counts=state_values,
            invariant_counts=invariant_values,
            armed_total=armed_total,
            published_total=published_total,
            payment_totals=payment_totals,
            reset_total=reset_total,
            checkpoint_safe=safe,
            device_to_host_transfers=1,
            runtime_integrated=RUNTIME_INTEGRATED,
            cuda_profiled=CUDA_PROFILED,
            formal_exact_resume_integrated=FORMAL_EXACT_RESUME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )
        self._last_drained_update = update_index
        self._last_receipt = receipt
        self._last_receipt_consumed = False
        return receipt

    def state_dict(self, receipt: StrikeFactBoundaryReceipt) -> dict[str, object]:
        """Export an exact device checkpoint authorized by the latest drain.

        Exact-strike FK observation capabilities are deliberately ephemeral:
        the FreshObservationOwner must latch their clone-only facts at the
        pre-Reward callpoint and checkpoint that latch under its own owner.
        Neither an opaque capability nor a replayable R03 observation record
        crosses this PPO checkpoint.
        """

        self._require_no_armed_epoch_fact(operation="checkpoint")
        self._require_no_open_full_mdp_reward_cycle(operation="checkpoint")
        self._require_pre_optimizer_operable()
        if self._active_pre_optimizer_pack is not None:
            self._poison_pre_optimizer(
                "R03 checkpoint export attempted before global optimizer acknowledgement"
            )
            raise StrikeFactDeviceError(
                "R03 checkpoint export requires completed global acknowledgement"
            )

        if receipt is not self._last_receipt:
            raise StrikeFactDeviceError("checkpoint receipt is foreign or stale")
        if self._last_receipt_consumed:
            raise StrikeFactDeviceError("checkpoint receipt was already consumed")
        if receipt.mutation_version != self._mutation_version:
            raise StrikeFactDeviceError("coordinator mutated after PPO-boundary drain")
        if self._pre_optimizer_global_protocol_adopted and (
            self._checkpoint_requires_global_drain_ack
            or self._last_globally_acknowledged_mutation_version
            != self._mutation_version
        ):
            raise StrikeFactDeviceError(
                "checkpoint lacks the exact globally ACKed R03 mutation frontier"
            )
        if not receipt.checkpoint_safe:
            raise StrikeFactDeviceError("PPO-boundary receipt is not checkpoint-safe")

        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "num_envs": self.num_envs,
            "dtype": str(self.dtype),
            "consumers": self.consumers,
            "drain_update_index": receipt.update_index,
            "drain_sequence": receipt.drain_sequence,
            "mutation_version": self._mutation_version,
            "drain_protocol": (
                PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_GLOBAL
                if self._pre_optimizer_global_protocol_adopted
                else PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_LEGACY
            ),
            "armed": self._armed.detach().clone(),
            "pending": self._pending.detach().clone(),
            "paid": self._paid.detach().clone(),
            "visible": self._visible.detach().clone(),
            "fault_bits": self._fault_bits.detach().clone(),
            "arm_source_step": self._arm_source_step.detach().clone(),
            "cache_source_step": self._cache_source_step.detach().clone(),
            "arm_validity": self._arm_validity.detach().clone(),
            "cache_validity": self._cache_validity.detach().clone(),
            "viewed_mask": self._viewed_mask.detach().clone(),
            "paid_mask": self._paid_mask.detach().clone(),
            "payment_values": self._payment_values.detach().clone(),
            "armed_total": self._armed_total.detach().clone(),
            "published_total": self._published_total.detach().clone(),
            "payment_totals": self._payment_totals.detach().clone(),
            "reset_total": self._reset_total.detach().clone(),
        }
        for name, tensor in self._arm_key_ints.items():
            result[f"arm_key_{name}"] = tensor.detach().clone()
        for name, tensor in self._arm_key_digests.items():
            result[f"arm_key_{name}"] = tensor.detach().clone()
        for name, tensor in self._cache_key_ints.items():
            result[f"cache_key_{name}"] = tensor.detach().clone()
        for name, tensor in self._cache_key_digests.items():
            result[f"cache_key_{name}"] = tensor.detach().clone()
        for name, tensor in self._arm_vectors.items():
            result[f"arm_{name}"] = tensor.detach().clone()
        for name, tensor in self._cache_vectors.items():
            result[f"cache_{name}"] = tensor.detach().clone()
        self._last_receipt_consumed = True
        return result

    checkpoint = state_dict

    def _checkpoint_tensor(
        self,
        value: Mapping[str, object],
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor):
            raise StrikeFactDeviceError(f"checkpoint {name} must be a tensor")
        if tensor.shape != shape or tensor.dtype != dtype:
            raise StrikeFactDeviceError(
                f"checkpoint {name} has wrong shape or dtype"
            )
        return tensor.detach().to(device=self.device).clone()

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        """Atomically load a drained checkpoint after collision validation.

        Restore starts with no valid exact-strike FK projection.  This prevents
        a pre-checkpoint publication capability from being replayed after cold
        restore; the independently checkpointed FreshObservationOwner latch is
        the observation-side source of continuity.
        """

        self._require_no_armed_epoch_fact(operation="checkpoint restore")
        self._require_hot_mutation_permitted()

        if not isinstance(value, Mapping):
            raise StrikeFactDeviceError("checkpoint must be a mapping")
        tensor_names = {
            "armed",
            "pending",
            "paid",
            "visible",
            "fault_bits",
            "arm_source_step",
            "cache_source_step",
            "arm_validity",
            "cache_validity",
            "viewed_mask",
            "paid_mask",
            "payment_values",
            "armed_total",
            "published_total",
            "payment_totals",
            "reset_total",
            *(f"arm_key_{name}" for name in _KEY_FIELDS),
            *(f"cache_key_{name}" for name in _KEY_FIELDS),
            *(f"arm_{name}" for name in _TARGET_FIELDS),
            *(f"cache_{name}" for name in _VECTOR_FIELDS),
        }
        metadata_names = {
            "schema_version",
            "num_envs",
            "dtype",
            "consumers",
            "drain_update_index",
            "drain_sequence",
            "mutation_version",
            "drain_protocol",
        }
        expected = tensor_names | metadata_names
        actual = set(value)
        if actual != expected:
            raise StrikeFactDeviceError(
                "checkpoint fields differ: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )
        if value["schema_version"] != SCHEMA_VERSION:
            raise StrikeFactDeviceError("checkpoint schema_version differs")
        if value["num_envs"] != self.num_envs:
            raise StrikeFactDeviceError("checkpoint num_envs differs")
        if value["dtype"] != str(self.dtype):
            raise StrikeFactDeviceError("checkpoint dtype differs")
        if tuple(value["consumers"]) != self.consumers:  # type: ignore[arg-type]
            raise StrikeFactDeviceError("checkpoint consumers differ")
        for name in ("drain_update_index", "drain_sequence", "mutation_version"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise StrikeFactDeviceError(f"checkpoint {name} must be non-negative int")
        if value["drain_protocol"] not in (
            PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_GLOBAL,
            PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_LEGACY,
        ):
            raise StrikeFactDeviceError("checkpoint drain_protocol differs")
        if self._mutation_version != 0:
            raise StrikeFactDeviceError("load destination must be a fresh coordinator")

        staged: dict[str, torch.Tensor] = {
            "armed": self._checkpoint_tensor(value, "armed", (self.num_envs,), torch.bool),
            "pending": self._checkpoint_tensor(value, "pending", (self.num_envs,), torch.bool),
            "paid": self._checkpoint_tensor(value, "paid", (self.num_envs,), torch.bool),
            "visible": self._checkpoint_tensor(value, "visible", (self.num_envs,), torch.bool),
            "fault_bits": self._checkpoint_tensor(value, "fault_bits", (self.num_envs,), torch.int64),
            "arm_source_step": self._checkpoint_tensor(value, "arm_source_step", (self.num_envs,), torch.int64),
            "cache_source_step": self._checkpoint_tensor(value, "cache_source_step", (self.num_envs,), torch.int64),
            "arm_validity": self._checkpoint_tensor(value, "arm_validity", (self.num_envs,), torch.bool),
            "cache_validity": self._checkpoint_tensor(value, "cache_validity", (self.num_envs,), torch.bool),
            "viewed_mask": self._checkpoint_tensor(value, "viewed_mask", (self.num_envs,), torch.int64),
            "paid_mask": self._checkpoint_tensor(value, "paid_mask", (self.num_envs,), torch.int64),
            "payment_values": self._checkpoint_tensor(value, "payment_values", (self.num_envs, CONSUMER_COUNT), self.dtype),
            "armed_total": self._checkpoint_tensor(value, "armed_total", (), torch.int64),
            "published_total": self._checkpoint_tensor(value, "published_total", (), torch.int64),
            "payment_totals": self._checkpoint_tensor(value, "payment_totals", (CONSUMER_COUNT,), torch.int64),
            "reset_total": self._checkpoint_tensor(value, "reset_total", (), torch.int64),
        }
        for prefix in ("arm", "cache"):
            for name in _INT_KEY_FIELDS:
                staged[f"{prefix}_key_{name}"] = self._checkpoint_tensor(
                    value, f"{prefix}_key_{name}", (self.num_envs,), torch.int64
                )
            for name in _DIGEST_KEY_FIELDS:
                staged[f"{prefix}_key_{name}"] = self._checkpoint_tensor(
                    value,
                    f"{prefix}_key_{name}",
                    (self.num_envs, 32),
                    torch.uint8,
                )
        for name in _TARGET_FIELDS:
            staged[f"arm_{name}"] = self._checkpoint_tensor(
                value, f"arm_{name}", (self.num_envs, 3), self.dtype
            )
        for name in _VECTOR_FIELDS:
            staged[f"cache_{name}"] = self._checkpoint_tensor(
                value, f"cache_{name}", (self.num_envs, 3), self.dtype
            )

        # Validate atomically by installing into a fresh scratch coordinator.
        scratch = ActionBallStrikeFactDeviceCoordinator(
            num_envs=self.num_envs,
            device=self.device,
            dtype=self.dtype,
            observation_projection_mode=self._observation_projection_mode,
        )
        scratch._copy_staged(staged)
        invariant_masks = scratch._invariant_masks()
        load_counts = torch.stack(
            [
                (scratch._fault_bits != 0).to(dtype=torch.int64).sum(),
                *[mask.to(dtype=torch.int64).sum() for _, mask in invariant_masks],
            ]
        )
        load_values = load_counts.detach().to(device="cpu").tolist()
        names = ("sticky_fault_bits",) + tuple(name for name, _ in invariant_masks)
        failures = tuple(
            f"{name}={int(count)}"
            for name, count in zip(names, load_values)
            if count != 0
        )
        if failures:
            raise StrikeFactDeviceError(
                "checkpoint collision/invariant failure: " + ", ".join(failures)
            )

        self._copy_staged(staged)
        self._mutation_version = int(value["mutation_version"])
        self._drain_sequence = int(value["drain_sequence"])
        self._last_drained_update = int(value["drain_update_index"])
        self._last_receipt = None
        self._last_global_receipt = None
        self._last_receipt_consumed = False
        self._pre_optimizer_global_protocol_adopted = (
            value["drain_protocol"]
            == PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_GLOBAL
        )

    def _copy_staged(self, staged: Mapping[str, torch.Tensor]) -> None:
        self._armed.copy_(staged["armed"])
        self._pending.copy_(staged["pending"])
        self._paid.copy_(staged["paid"])
        self._visible.copy_(staged["visible"])
        self._fault_bits.copy_(staged["fault_bits"])
        self._arm_source_step.copy_(staged["arm_source_step"])
        self._cache_source_step.copy_(staged["cache_source_step"])
        self._arm_validity.copy_(staged["arm_validity"])
        self._cache_validity.copy_(staged["cache_validity"])
        self._viewed_mask.copy_(staged["viewed_mask"])
        self._paid_mask.copy_(staged["paid_mask"])
        self._payment_values.copy_(staged["payment_values"])
        self._armed_total.copy_(staged["armed_total"])
        self._published_total.copy_(staged["published_total"])
        self._payment_totals.copy_(staged["payment_totals"])
        self._reset_total.copy_(staged["reset_total"])
        for prefix, ints, digests in (
            ("arm", self._arm_key_ints, self._arm_key_digests),
            ("cache", self._cache_key_ints, self._cache_key_digests),
        ):
            for name in _INT_KEY_FIELDS:
                ints[name].copy_(staged[f"{prefix}_key_{name}"])
            for name in _DIGEST_KEY_FIELDS:
                digests[name].copy_(staged[f"{prefix}_key_{name}"])
        for name in _TARGET_FIELDS:
            self._arm_vectors[name].copy_(staged[f"arm_{name}"])
        for name in _VECTOR_FIELDS:
            self._cache_vectors[name].copy_(staged[f"cache_{name}"])

    load_checkpoint = load_state_dict


__all__ = (
    "SCHEMA_VERSION",
    "SCALE_TARGET_NUM_ENVS",
    "GUIDE_CONSUMERS",
    "PADDLE_CENTER_PROXIMITY_CONSUMER",
    "COMMON_CONSUMERS",
    "STRIKE_FACT_CONSUMERS",
    "FULL_MDP_REWARD_CONSUMERS",
    "CONSUMER_COUNT",
    "FULL_CONSUMER_MASK",
    "RUNTIME_INTEGRATED",
    "CUDA_PROFILED",
    "FORMAL_EXACT_RESUME_INTEGRATED",
    "LAUNCH_AUTHORIZED",
    "INTEGRATION_RESIDUALS",
    "FAULTS",
    "STATE_NAMES",
    "INVARIANT_NAMES",
    "PRE_OPTIMIZER_PPO_BOUNDARY_OWNER_KIND",
    "PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_GLOBAL",
    "PRE_OPTIMIZER_PPO_BOUNDARY_PROTOCOL_LEGACY",
    "OBSERVATION_PROJECTION_MODE_LEGACY_DIAGNOSTIC",
    "OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP",
    "PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES",
    "make_pre_optimizer_ppo_boundary_leaf_schema",
    "StrikeFactDeviceError",
    "StrikeFactObservationIdentityHold",
    "StrikeFactEpochPublicationHold",
    "R03_EPOCH_FAULT_STALE_SOURCE_STEP",
    "R03_EPOCH_FAULT_NONFINITE_FACT",
    "R03_EPOCH_FAULT_EPOCH_IDENTITY",
    "R03_EPOCH_FACT_PRESENT",
    "R03_EPOCH_FACT_PHYSICALLY_VALID",
    "R03_EPOCH_FACT_VALUE_COUNT",
    "DeviceActionTaskKey",
    "EpochR03RacketIdentity",
    "SharedStrikeFactDeviceView",
    "ExactStrikeFkObservationProjection",
    "ExactStrikeFkObservationView",
    "R03FullMdpPreRewardPublication",
    "R03FullMdpPreRewardView",
    "R03FullMdpRewardPaymentVerdict",
    "R03FullMdpRewardCloseReceipt",
    "StrikeFactBoundaryReceipt",
    "ActionBallStrikeFactDeviceCoordinator",
)
