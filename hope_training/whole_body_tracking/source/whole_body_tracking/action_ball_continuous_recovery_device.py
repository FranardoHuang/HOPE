"""Device-resident pure owner for continuous ActionBall recovery facts.

``PRE_INTEGRATION_HOLD`` is part of this module's contract.  The coordinator
has no Isaac, MuJoCo, CommandManager, RewardManager, or environment-config
dependency and cannot authorize a launch.  A future adapter must bind the
portable projections below to the real manager graph and whole-state
checkpoint root, then prove that wiring on Pod1.

The narrow owner implemented here keeps one stable in-place readiness tensor,
the complete C05 fourteen-field outcome key for every committed row, the
completed-action frame-0 reference, and one post-physics recovery fact.  A
scheduled reveal is never delayed by readiness.  Readiness only controls
playback outside this owner; a committed unplayed row still owns its frozen
deadline and subsequent recovery window.  Future task, target, ball and
schedule material are deliberately absent from every public hot-path view.

All plant errors are computed from common tensors in this file.  There is no
backend or A/C branch.  Reference joint/body velocities are literal zero by
construction rather than caller-provided values.  The only score is the
portable additive raw recovery score; a hard readiness conjunction never
gates that dense score.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import importlib
import inspect
from numbers import Integral
from pathlib import Path
import struct
import threading
from types import FunctionType
from typing import Callable, Mapping, NoReturn, Optional, Sequence
import weakref

import torch

import action_ball_continuous_recovery_runtime as _runtime
import action_ball_motion_cadence_device as _motion_cadence
try:
    import action_ball_full_mdp_row_identity as _row_identity
except ImportError:  # pragma: no cover - package-style runtime import
    from whole_body_tracking import action_ball_full_mdp_row_identity as _row_identity


_COMMANDS_SOURCE = (
    Path(__file__).resolve().parent
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "commands.py"
).resolve()
_MOTION_CADENCE_SOURCE = Path(_motion_cadence.__file__).resolve()


SCHEMA_VERSION = 1
INTEGRATION_STATUS = "PRE_INTEGRATION_HOLD"
RUNTIME_WIRING_CONNECTED = False
CUDA_PROFILED = False
FORMAL_EXACT_RESUME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON = "diagnostic_n2_r07_live_inputs_invalid"

# These are an explicit, non-promotable algorithm choice for the N=2 diagnostic
# loop.  They are not a safety certificate and none of their provenance labels
# authorize launch.  The values reuse the already adopted post-swing settle
# margins where that concept exists; the remaining whole-state channels use
# deliberately broad diagnostic scales pending measured recovery evidence.
DIAGNOSTIC_N2_SUPPORT_FORCE_Z_N = 10.0
DIAGNOSTIC_N2_MIN_ROOT_HEIGHT_M = 0.5
DIAGNOSTIC_N2_MAX_ROOT_TILT_RAD = 0.7
DIAGNOSTIC_N2_COMPONENT_WEIGHTS = {
    name: 1.0 for name in _runtime.COMPONENT_NAMES
}
DIAGNOSTIC_N2_COMPONENT_SCALES = {
    "root_position_m": 0.05,
    "root_orientation_rad": 0.10,
    "root_linear_velocity_mps": 0.20,
    "root_angular_velocity_radps": 0.30,
    "joint_position_rad": 0.20,
    "joint_velocity_radps": 1.00,
    "body_position_m": 0.10,
    "body_orientation_rad": 0.20,
    "body_linear_velocity_mps": 0.30,
    "body_angular_velocity_radps": 0.50,
    "station_xy_m": 0.10,
    "foot_slip_mps": 0.10,
    "foot_support_deficit": 1.00,
}
DIAGNOSTIC_N2_READY_TOLERANCES = {
    "root_position_m": 0.10,
    "root_orientation_rad": 0.10,
    "root_linear_velocity_mps": 0.30,
    "root_angular_velocity_radps": 0.50,
    "joint_position_rad": 0.25,
    "joint_velocity_radps": 1.00,
    "body_position_m": 0.15,
    "body_orientation_rad": 0.25,
    "body_linear_velocity_mps": 0.50,
    "body_angular_velocity_radps": 1.00,
    "station_xy_m": 0.10,
    "foot_slip_mps": 0.05,
    "foot_support_deficit": 0.0,
}

# Direct ActionEpoch fact ABI.  These bits describe producer/chronology
# failures only.  A finite fall, insufficient support, or a large recovery
# error is ordinary learning data and must never set one of these bits.
R07_EPOCH_FAULT_STALE_SOURCE_STEP = 1 << 0
R07_EPOCH_FAULT_INVALID_PLANT_FACT = 1 << 1
R07_EPOCH_FAULT_INVALID_REFERENCE = 1 << 2
R07_EPOCH_FAULT_EPOCH_IDENTITY = 1 << 3
R07_EPOCH_FACT_PRESENT = 1 << 0
R07_EPOCH_FACT_NUMERICALLY_VALID = 1 << 1
R07_EPOCH_FACT_VALUE_COUNT = 20
R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0 = 1
R07_REFERENCE_COMPLETED_ACTION_FRAME0 = 2


POLICY_RATE_HZ = 50
RECOVERY_START_AGE_TICK = 10
RECOVERY_END_AGE_TICK = 77
RECOVERY_SAMPLE_COUNT = 68
RECOVERY_REWARD_CONSUMER = "common_recovery_reward_v1"
FULL_MDP_REWARD_CONSUMERS = (RECOVERY_REWARD_CONSUMER,)

PHASE_PRE_REVEAL_HIDDEN = 0
PHASE_ACTIVE_OPPORTUNITY = 1
PHASE_POST_DEADLINE_SUFFIX = 2
PHASE_RECOVERY_HIDDEN = 3
PHASE_READY_HOLD = 4
PHASE_RECOVERY_UNAVAILABLE = 5
PHASE_INFRASTRUCTURE_INVALID = 6
PHASE_CODES = (
    PHASE_PRE_REVEAL_HIDDEN,
    PHASE_ACTIVE_OPPORTUNITY,
    PHASE_POST_DEADLINE_SUFFIX,
    PHASE_RECOVERY_HIDDEN,
    PHASE_READY_HOLD,
    PHASE_RECOVERY_UNAVAILABLE,
    PHASE_INFRASTRUCTURE_INVALID,
)


def _action_epoch_recovery_reward_window(
    phase_code: torch.Tensor,
    recovery_age_tick: torch.Tensor,
    *,
    outcome_settled_phase: int,
) -> torch.Tensor:
    """Return the sole post-shot R07 payment window.

    Readiness remains an all-cycle plant computation.  This predicate owns
    only reward eligibility: the shot outcome must already be settled and the
    deadline-relative age must be one of the 68 adopted recovery cells.
    """

    return (
        phase_code.eq(outcome_settled_phase)
        & recovery_age_tick.ge(RECOVERY_START_AGE_TICK)
        & recovery_age_tick.le(RECOVERY_END_AGE_TICK)
    )

OWNER_NONE = 0
OWNER_SEQUENCE_BIRTH = 1
OWNER_COMMITTED_TASK = 2

# Sticky per-environment protocol/fact faults.  Expected recovery samples are
# counted independently, so an invalid row cannot silently leave a denominator.
FAULT_INVALID_BIND = 1 << 0
FAULT_INVALID_COMMIT = 1 << 1
FAULT_LINEAGE = 1 << 2
FAULT_STEP_REGRESSION = 1 << 3
FAULT_COMMAND_BINDING = 1 << 4
FAULT_SHOT_BOUNDARY_REQUEST = 1 << 5
FAULT_SUFFIX_INCOMPLETE = 1 << 6
FAULT_INVALID_PLANT_FACT = 1 << 7
FAULT_PUBLISH_COLLISION = 1 << 8
FAULT_DUPLICATE_VIEW = 1 << 9
FAULT_PAYMENT_BEFORE_VIEW = 1 << 10
FAULT_DUPLICATE_PAYMENT = 1 << 11
FAULT_PAYMENT_MISMATCH = 1 << 12
FAULT_RESET_UNSETTLED = 1 << 13
FAULT_LEDGER_SEQUENCE = 1 << 14
FAULT_ACTION_EPOCH_MOTION_CHRONOLOGY = 1 << 15
FAULT_MOTION_CADENCE_SUCCESSOR_OVERFLOW = 1 << 16

FAULTS = (
    ("invalid_bind", FAULT_INVALID_BIND),
    ("invalid_commit", FAULT_INVALID_COMMIT),
    ("lineage", FAULT_LINEAGE),
    ("step_regression", FAULT_STEP_REGRESSION),
    ("command_binding", FAULT_COMMAND_BINDING),
    ("shot_boundary_request", FAULT_SHOT_BOUNDARY_REQUEST),
    ("suffix_incomplete", FAULT_SUFFIX_INCOMPLETE),
    ("invalid_plant_fact", FAULT_INVALID_PLANT_FACT),
    ("publish_collision", FAULT_PUBLISH_COLLISION),
    ("duplicate_view", FAULT_DUPLICATE_VIEW),
    ("payment_before_view", FAULT_PAYMENT_BEFORE_VIEW),
    ("duplicate_payment", FAULT_DUPLICATE_PAYMENT),
    ("payment_mismatch", FAULT_PAYMENT_MISMATCH),
    ("reset_unsettled", FAULT_RESET_UNSETTLED),
    ("ledger_sequence", FAULT_LEDGER_SEQUENCE),
    (
        "action_epoch_motion_chronology",
        FAULT_ACTION_EPOCH_MOTION_CHRONOLOGY,
    ),
    (
        "motion_cadence_successor_overflow",
        FAULT_MOTION_CADENCE_SUCCESSOR_OVERFLOW,
    ),
)

R07_GLOBAL_DRAIN_OWNER_KIND = "r07_recovery"
R07_GLOBAL_DRAIN_FAULT_FIELDS = tuple(
    f"fault_{name}_count" for name, _bit in FAULTS
)
R07_GLOBAL_DRAIN_INVARIANT_FIELDS = (
    "invariant_dirty_inactive_count",
    "invariant_ready_without_dwell_count",
    "invariant_eligible_without_owner_count",
    "invariant_payment_required_without_expected_count",
    "invariant_paid_without_view_count",
    "invariant_played_suffix_owner_mismatch_count",
    "invariant_unplayed_marked_suffix_complete_count",
    "invariant_reward_owner_key_mismatch_count",
    "invariant_pending_key_mismatch_count",
    "invariant_reference_kind_invalid_count",
    "invariant_deadline_ack_pending_count",
    "invariant_accounting_invalid_count",
    "invariant_window_cursor_invalid_count",
    "invariant_window_owner_invalid_count",
    "invariant_payment_epoch_closed_unsettled_count",
    "invariant_dirty_invalid_pending_count",
)
R07_GLOBAL_DRAIN_TOTAL_FIELDS = (
    "recovery_expected_total",
    "reward_eligible_total",
    "reward_payment_total",
    "reward_income_total_float64_bits",
    "ready_instant_total",
    "first_ready_total",
    "played_deadline_total",
    "unplayed_deadline_total",
)
R07_GLOBAL_DRAIN_PER_ENV_FIELDS = (
    "window_expected_count",
    "window_eligible_count",
    "window_payment_count",
    "window_first_expected_age_tick_encoded",
    "window_last_expected_age_tick_encoded",
    "window_last_paid_age_tick_encoded",
)
R07_GLOBAL_DRAIN_FIELD_NAMES = (
    "mutation_version",
    "fault_count",
    "invariant_count",
    *R07_GLOBAL_DRAIN_FAULT_FIELDS,
    *R07_GLOBAL_DRAIN_INVARIANT_FIELDS,
    *R07_GLOBAL_DRAIN_TOTAL_FIELDS,
    *R07_GLOBAL_DRAIN_PER_ENV_FIELDS,
)
R07_PPO_DRAIN_LEAF_SCHEMA = (
    R07_GLOBAL_DRAIN_OWNER_KIND,
    tuple(
        (
            name,
            "per_env" if name in R07_GLOBAL_DRAIN_PER_ENV_FIELDS else "scalar",
            0,
        )
        for name in R07_GLOBAL_DRAIN_FIELD_NAMES
    ),
)

_FaultInjector = Optional[Callable[[str], None]]
_ZERO_DIGEST = bytes(32)


class ContinuousRecoveryDeviceError(RuntimeError):
    """A host ABI, immutable identity, or checkpoint contract failed."""


class ContinuousRecoveryConstructionHold(ContinuousRecoveryDeviceError):
    """The code-owned runtime lacks a reviewed R07 construction input."""


class ContinuousRecoveryFullMdpRewardPublication:
    """Opaque owner-issued identity for one post-physics R07 reward epoch."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("R07 full-MDP reward publications are owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("R07 full-MDP reward publications are immutable")


class ContinuousRecoveryFullMdpRewardPaymentVerdict:
    """Opaque proof that the exact R07 RewardTerm paid its owner tensor."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("R07 full-MDP reward verdicts are owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("R07 full-MDP reward verdicts are immutable")


class ContinuousRecoveryFullMdpRewardCloseReceipt:
    """Opaque R07 proof that its sole payment epoch closed exactly once."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("R07 full-MDP reward close receipts are owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("R07 full-MDP reward close receipts are immutable")


class ContinuousRecoveryMotionReadyProjection:
    """Opaque current-epoch readiness capability issued only by R07.

    Motion must retain this handle plus :meth:`require_owned_motion_ready_projection`;
    the writable in-place readiness tensor is deliberately not an authority.
    """

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("R07 Motion-ready projections are owner-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("R07 Motion-ready projections are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("R07 Motion-ready projections cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("R07 Motion-ready projections cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("R07 Motion-ready projections cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("R07 Motion-ready projections cannot be serialized")


@dataclass(frozen=True)
class ContinuousRecoveryMotionReadyView:
    """Clone-only Motion view authenticated by one exact R07 owner."""

    ready_projection: ContinuousRecoveryMotionReadyProjection
    owner_kind: str
    ready_identity: object
    ready: torch.Tensor
    ready_streak: torch.Tensor
    required_dwell: int
    control_tick: torch.Tensor


@dataclass(frozen=True)
class ContinuousRecoveryObservationState:
    """Clone-only R07 post-physics state; never a Motion admission fact."""

    postphysics_valid: torch.Tensor
    source_step: torch.Tensor
    reset_generation: torch.Tensor
    control_tick: torch.Tensor
    ready_streak: torch.Tensor
    required_dwell: int
    foot_supported_lr: torch.Tensor


@dataclass(frozen=True)
class ContinuousRecoveryFullMdpPreRewardView:
    """Authenticated R07 nonterminating DoneTerm projection."""

    terminated: torch.Tensor
    time_out: torch.Tensor


@dataclass(frozen=True)
class _FullMdpPreRewardPayload:
    owner_identity: object
    runtime_owner: object
    control_step: int
    source_step: torch.Tensor
    current_task_key_sha256: torch.Tensor
    recovery_owner_key_sha256: torch.Tensor
    recovery_deadline_tick: torch.Tensor
    plant_facts_valid: torch.Tensor
    publication_mutation_version: int


@dataclass(frozen=True)
class _FullMdpPaymentPayload:
    owner_identity: object
    runtime_owner: object
    publication: ContinuousRecoveryFullMdpRewardPublication
    consumer: str
    control_step: int
    payment_identity: DeviceContinuousRecoveryPaymentIdentity


@dataclass(frozen=True)
class _FullMdpClosePayload:
    owner_identity: object
    runtime_owner: object
    publication: ContinuousRecoveryFullMdpRewardPublication
    payment_verdict: ContinuousRecoveryFullMdpRewardPaymentVerdict
    control_step: int


@dataclass(frozen=True)
class _MotionReadyPayload:
    owner_identity: object
    owner: object
    postphysics_valid: torch.Tensor
    source_step: torch.Tensor
    reset_generation: torch.Tensor
    control_tick: torch.Tensor
    ready: torch.Tensor
    ready_streak: torch.Tensor
    required_dwell: int
    foot_supported_lr: torch.Tensor


_FULL_MDP_REWARD_REGISTRY_LOCK = threading.RLock()
_FULL_MDP_PRE_REWARD_REGISTRY: "weakref.WeakKeyDictionary[object, _FullMdpPreRewardPayload]" = weakref.WeakKeyDictionary()
_FULL_MDP_PAYMENT_REGISTRY: "weakref.WeakKeyDictionary[object, _FullMdpPaymentPayload]" = weakref.WeakKeyDictionary()
_FULL_MDP_CLOSE_REGISTRY: "weakref.WeakKeyDictionary[object, _FullMdpClosePayload]" = weakref.WeakKeyDictionary()
_MOTION_READY_REGISTRY: "weakref.WeakKeyDictionary[object, _MotionReadyPayload]" = weakref.WeakKeyDictionary()


def _mint_full_mdp_identity(cls: type, registry: object, payload: object) -> object:
    value = object.__new__(cls)
    with _FULL_MDP_REWARD_REGISTRY_LOCK:
        registry[value] = payload
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ContinuousRecoveryDeviceError(f"{label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ContinuousRecoveryDeviceError(f"{label} must be >= {minimum}")
    return result


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuousRecoveryDeviceError(
            f"{label} must be one lowercase SHA-256"
        )
    return value


def _digest_tensor(
    values: Sequence[str], *, device: torch.device
) -> torch.Tensor:
    rows = [list(bytes.fromhex(_sha256(value, label="digest"))) for value in values]
    if not rows:
        return torch.empty((0, 32), dtype=torch.uint8, device=device)
    return torch.tensor(rows, dtype=torch.uint8, device=device)


def _host_text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def materialize_r07_ppo_drain_leaf_schema(
    *,
    leaf_schema_type: object,
    field_spec_type: object,
) -> object:
    """Build the exact global schema without importing the top coordinator.

    R07 is a standalone leaf used by focused tests and both simulators.  It
    therefore exports a data-only production descriptor and lets the already
    loaded global drain module supply its own exact classes, avoiding a
    reverse import cycle and duplicate class identities.
    """

    if not callable(leaf_schema_type) or not callable(field_spec_type):
        raise ContinuousRecoveryDeviceError(
            "R07 global drain schema types must be callable"
        )
    owner_kind, fields = R07_PPO_DRAIN_LEAF_SCHEMA
    return leaf_schema_type(
        owner_kind=owner_kind,
        fields=tuple(
            field_spec_type(name=name, cardinality=cardinality, minimum=minimum)
            for name, cardinality, minimum in fields
        ),
    )


@dataclass(frozen=True)
class DiagnosticN2ContinuousRecoveryBundle:
    """Construction-bound R07 leaf, live fact adapter, Motion and epoch.

    The bundle itself is the sole R07 publisher identity cold-bound into the
    epoch.  This avoids both a late mutable binding and the retired portable
    receipt/payment registries.
    """

    owner: "ContinuousRecoveryDeviceCoordinator"
    plant_fact_adapter: "DiagnosticN2ContinuousRecoveryPlantFactAdapter"
    motion_owner: object
    action_epoch_owner: object
    motion_parent_authority: object
    motion_parent_receipt: object

    def _project_parent_action_identity(
        self,
    ) -> _motion_cadence.DiagnosticMotionParentActionIdentity:
        """Read the independent parent's retained action identity each use."""

        authority = self.motion_parent_authority
        receipt = self.motion_parent_receipt
        project = getattr(authority, "project_bound_action_identity", None)
        declared = _exact_motion_parent_projection_definition(authority)
        if (
            type(receipt)
            is not _motion_cadence.DiagnosticMotionProfileReceipt
            or not callable(project)
            or project.__self__ is not authority
            or project.__func__ is not declared
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 diagnostic Motion parent authority/receipt differs"
            )
        try:
            identity = project(receipt, motion_owner=self.motion_owner)
        except Exception as exc:
            raise ContinuousRecoveryDeviceError(
                "R07 diagnostic Motion parent identity projection failed"
            ) from exc
        if (
            type(identity)
            is not _motion_cadence.DiagnosticMotionParentActionIdentity
            or identity.authority is not authority
            or identity.motion_owner is not self.motion_owner
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 diagnostic Motion parent identity differs"
            )
        return identity

    def _require_bound_owner(self) -> "ContinuousRecoveryDeviceCoordinator":
        """Return the one coordinator that minted this construction bundle."""

        owner = self.owner
        adapter = self.plant_fact_adapter
        if (
            type(owner) is not ContinuousRecoveryDeviceCoordinator
            or getattr(owner, "_diagnostic_n2_bundle", None) is not self
            or type(adapter) is not DiagnosticN2ContinuousRecoveryPlantFactAdapter
            or getattr(adapter, "_motion_owner", None) is not self.motion_owner
            or getattr(self.motion_owner, "_env", None)
            is not getattr(adapter, "_env", None)
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 diagnostic bundle/owner identity changed after construction"
            )
        return owner

    def motion_ready_projection(self) -> "ContinuousRecoveryMotionReadyProjection":
        """Forward the newest owner-issued next-tick Motion capability."""

        return self._require_bound_owner().motion_ready_projection()

    def require_owned_motion_ready_projection(
        self,
        projection: object,
        *,
        owner_kind: object,
    ) -> "ContinuousRecoveryMotionReadyView":
        """Authenticate through the bound owner without exposing ``.owner``."""

        return self._require_bound_owner().require_owned_motion_ready_projection(
            projection,
            owner_kind=owner_kind,
        )

    def action_epoch_observation_state(
        self,
    ) -> "ContinuousRecoveryObservationState":
        """Read the existing R07 post-physics state without Motion authority."""

        return self._require_bound_owner().action_epoch_observation_state()

    def publish_epoch_reward_facts(
        self, *, current_source_step: torch.Tensor
    ) -> "R07EpochDirectRewardFacts":
        """Publish R07 readiness and keyed facts for an active ActionEpoch."""

        return self._publish_epoch_reward_facts(
            current_source_step=current_source_step,
            publish_keyed_action_epoch=True,
        )

    def stamp_epoch_idle_observation_without_keyed_facts(
        self, *, current_source_step: int
    ) -> None:
        """Stamp neutral/N/A chronology without reading R07 plant facts.

        This is shared by the unkeyed idle lane and keyed pre-recovery phases;
        neither has an R07 reward/readiness business consumer on this tick.
        """

        owner = self._require_bound_owner()
        epoch_owner = self.action_epoch_owner
        if (
            isinstance(current_source_step, bool)
            or not isinstance(current_source_step, int)
            or current_source_step < 0
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 idle-observation current_source_step is malformed"
            )
        # Construction already bound this exact ActionEpoch owner, and the
        # callee validates this R07 owner before projecting its narrow fact.
        # Re-authenticating the bound Python method here only made the caller
        # prove its own wiring and prevented transparent diagnostic wrapping.
        epoch_facts = epoch_owner.snapshot_idle_observation_chronology(
            owner=self
        )
        motion_cadence_tick = owner._tensor(
            getattr(
                self.motion_owner,
                "_action_ball_continuous_episode_step",
                None,
            ),
            label="R07 idle-observation Motion cadence tick",
            shape=(owner.num_envs,),
            dtype=torch.int64,
        )
        observed_source_step = self.plant_fact_adapter.observe_idle_source_step()
        owner.stamp_action_epoch_idle_observation(
            epoch_facts=epoch_facts,
            current_source_step=current_source_step,
            observed_source_step=observed_source_step,
            motion_cadence_tick=motion_cadence_tick,
            action_epoch_owner=epoch_owner,
        )
        return None

    def _publish_epoch_reward_facts(
        self,
        *,
        current_source_step: torch.Tensor,
        publish_keyed_action_epoch: bool,
    ) -> "R07EpochDirectRewardFacts":
        """Publish one real post-physics R07 fact group into ActionEpoch.

        The live adapter and Motion reference are construction-owned.  The
        caller supplies only the already-owned environment step tensor; it is
        checked as chronology and cannot authorize validity or reward.
        """

        if type(publish_keyed_action_epoch) is not bool:
            raise ContinuousRecoveryDeviceError(
                "R07 keyed publication mode differs"
            )
        owner = self._require_bound_owner()
        epoch_owner = self.action_epoch_owner
        step = owner._tensor(
            current_source_step,
            label="R07 epoch current_source_step",
            shape=(owner.num_envs,),
            dtype=torch.int64,
        )
        method_name = "project_action_ball_full_mdp_recovery_ready_reference"
        project_reference = getattr(self.motion_owner, method_name, None)
        exact_method = _exact_motion_reference_producer_definition(
            self.motion_owner
        )
        if (
            not callable(project_reference)
            or getattr(project_reference, "__self__", None) is not self.motion_owner
            or getattr(project_reference, "__func__", None) is not exact_method
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 Motion ready-reference producer identity differs"
            )

        epoch_snapshot = epoch_owner.current()
        reference = project_reference(action_epoch_snapshot=epoch_snapshot)
        facts = self.plant_fact_adapter.read()
        result = owner.action_epoch_reward_view(
            facts,
            reference=reference,
            epoch=epoch_snapshot,
            current_source_step=step,
            adapter_source_step=self.plant_fact_adapter.last_source_step,
            motion_owner=self.motion_owner,
            action_epoch_owner=epoch_owner,
        )
        owner._require_action_epoch_readiness_chronology(
            observed_source_step=self.plant_fact_adapter.last_source_step,
        )
        owner._require_r07_business_mutation_allowed(
            label="ActionEpoch post-physics readiness"
        )

        if publish_keyed_action_epoch:
            self._publish_keyed_epoch_reward_facts(
                owner=owner,
                epoch_owner=epoch_owner,
                epoch_snapshot=epoch_snapshot,
                result=result,
            )
        n = owner.num_envs
        try:
            reference_shot_key = _row_identity.require_action_epoch_shot_key(
                getattr(reference, "shot_key", None),
                shape=(n,),
                device=owner.device,
                label="R07 Motion reference shot_key",
            ).clone()
        except _row_identity.ActionEpochShotKeyError as exc:
            raise ContinuousRecoveryDeviceError(str(exc)) from exc
        owner._publish_action_epoch_motion_readiness(
            result,
            observed_source_step=self.plant_fact_adapter.last_source_step,
            shot_key=reference_shot_key,
            publish_keyed_first_ready=publish_keyed_action_epoch,
        )
        return result

    def _publish_keyed_epoch_reward_facts(
        self,
        *,
        owner: "ContinuousRecoveryDeviceCoordinator",
        epoch_owner: object,
        epoch_snapshot: object,
        result: "R07EpochDirectRewardFacts",
    ) -> None:
        """Publish only the keyed ActionEpoch planes for a real shot."""

        n = owner.num_envs
        s = epoch_owner.shot_slot_capacity
        env_ids = torch.arange(n, dtype=torch.int64, device=owner.device)
        slots = owner._tensor(
            epoch_snapshot.current_task_slot,
            label="R07 publish current_task_slot",
            shape=(n,),
            dtype=torch.int64,
        )
        slot_valid = slots.ge(0) & slots.lt(s)
        safe_slots = torch.clamp(slots, min=0, max=s - 1)
        lifecycle = epoch_snapshot.recovery_reference_lifecycle_masks()
        _owner_type, _record_type, lifecycle_type, epoch_globals = (
            _exact_action_epoch_types(epoch_owner)
        )
        if type(lifecycle) is not lifecycle_type:
            raise ContinuousRecoveryDeviceError(
                "R07 publish lifecycle type differs"
            )
        completed = owner._tensor(
            lifecycle.completed,
            label="R07 publish completed lifecycle",
            shape=(n, s),
            dtype=torch.bool,
        )[env_ids, safe_slots]
        current_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(epoch_snapshot.identity.shot_key, field.name)[
                    env_ids, safe_slots
                ].detach().clone()
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        current_phase = owner._tensor(
            epoch_snapshot.phase,
            label="R07 publish phase",
            shape=(n, s),
            dtype=torch.int64,
        )[env_ids, safe_slots]
        reveal_phase = epoch_globals.get("PHASE_REVEAL_COMMITTED")
        launch_phase = epoch_globals.get("PHASE_LAUNCH_SETTLED")
        outcome_phase = epoch_globals.get("PHASE_OUTCOME_SETTLED")
        if any(
            type(value) is not int
            for value in (reveal_phase, launch_phase, outcome_phase)
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch public phase constants differ"
            )
        active_phase = (
            current_phase.eq(reveal_phase)
            | current_phase.eq(launch_phase)
            | current_phase.eq(outcome_phase)
        )
        producer_window = (
            result.recovery_age_tick.ge(RECOVERY_START_AGE_TICK)
            & result.recovery_age_tick.le(RECOVERY_END_AGE_TICK)
        )
        publish_rows = (
            slot_valid
            & completed
            & active_phase
            & _row_identity.action_epoch_shot_key_valid(current_key)
            & producer_window
        )
        fault_bits = torch.zeros(
            (n, s), dtype=torch.int64, device=owner.device
        )
        fault_bits[env_ids, safe_slots] = torch.where(
            publish_rows,
            result.producer_fault_bits,
            torch.zeros_like(result.producer_fault_bits),
        )
        epoch_version = getattr(epoch_snapshot, "version", None)
        commit_head = epoch_owner.commit_head
        if (
            type(epoch_version) is not int
            or type(commit_head) is not int
            or epoch_version != commit_head - 1
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch snapshot advanced before publication"
            )
        epoch_owner.merge_runtime_owner_fault(
            "r07_recovery", fault_bits, owner=self
        )

        valid_bits = torch.zeros_like(fault_bits)
        selected_valid = (
            torch.full_like(
                result.producer_fault_bits, R07_EPOCH_FACT_PRESENT
            )
            | result.facts_valid.to(dtype=torch.int64)
            * R07_EPOCH_FACT_NUMERICALLY_VALID
        )
        valid_bits[env_ids, safe_slots] = torch.where(
            publish_rows, selected_valid, torch.zeros_like(selected_valid)
        )
        source_step = torch.full_like(fault_bits, -1)
        source_step[env_ids, safe_slots] = torch.where(
            publish_rows,
            result.source_step,
            torch.full_like(result.source_step, -1),
        )
        values = torch.zeros(
            (n, s, 32), dtype=torch.float32, device=owner.device
        )
        packed = torch.cat(
            (
                result.weighted_reward[:, None],
                result.raw_score[:, None],
                result.reward_eligible.to(torch.float32)[:, None],
                result.facts_valid.to(torch.float32)[:, None],
                result.infrastructure_fault.to(torch.float32)[:, None],
                result.ready_instant.to(torch.float32)[:, None],
                result.recovery_age_tick.to(torch.float32)[:, None],
                result.component_errors,
            ),
            dim=1,
        )
        if tuple(packed.shape) != (n, R07_EPOCH_FACT_VALUE_COUNT):
            raise ContinuousRecoveryDeviceError(
                "R07 packed ActionEpoch fact width differs"
            )
        values[
            env_ids, safe_slots, :R07_EPOCH_FACT_VALUE_COUNT
        ] = torch.where(
            publish_rows[:, None], packed, torch.zeros_like(packed)
        )
        epoch_owner.publish_owner_facts(
            "r07_recovery",
            owner=self,
            valid_bits=valid_bits,
            source_step=source_step,
            values=values,
        )


@dataclass(frozen=True)
class R07EpochDirectRewardFacts:
    """Clone-only direct R07 values; never a payment or admission receipt."""

    source_step: torch.Tensor
    motion_cadence_tick: torch.Tensor
    reset_generation: torch.Tensor
    recovery_age_tick: torch.Tensor
    reward_eligible: torch.Tensor
    facts_valid: torch.Tensor
    foot_supported_lr: torch.Tensor
    infrastructure_fault: torch.Tensor
    producer_fault_bits: torch.Tensor
    component_errors: torch.Tensor
    raw_score: torch.Tensor
    weighted_reward: torch.Tensor
    ready_instant: torch.Tensor
    reference_kind: torch.Tensor
    reference_action_slot: torch.Tensor
    reference_action_uid: torch.Tensor


class DiagnosticN2ContinuousRecoveryPlantFactAdapter:
    """Read R07 facts directly from one bound live Isaac environment.

    This is a numerical producer, not an authority.  It never accepts caller
    masks or readiness decisions.  Unsupported feet and a fallen robot remain
    finite facts with ``hard_safety_ok=False``; missing/malformed/non-finite
    plant channels instead make ``facts_valid=False``.
    """

    def __init__(
        self,
        *,
        env: object,
        motion_owner: object,
        robot: object,
        contact_sensor: object,
        ordered_joint_names: tuple[str, ...],
        ordered_body_names: tuple[str, ...],
        ordered_foot_names: tuple[str, ...],
        body_ids: tuple[int, ...],
        foot_robot_body_ids: tuple[int, ...],
        foot_sensor_body_ids: tuple[int, ...],
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._env = env
        self._motion_owner = motion_owner
        self._robot = robot
        self._contact_sensor = contact_sensor
        self.ordered_joint_names = ordered_joint_names
        self.ordered_body_names = ordered_body_names
        self.ordered_foot_names = ordered_foot_names
        self._body_ids = body_ids
        self._foot_robot_body_ids = foot_robot_body_ids
        self._foot_sensor_body_ids = foot_sensor_body_ids
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self._last_source_step = -1

    @property
    def last_source_step(self) -> int:
        """The exact environment control step observed by the latest read."""

        return self._last_source_step

    def _require_bound_identities(self) -> None:
        if getattr(self._motion_owner, "_env", None) is not self._env:
            raise ContinuousRecoveryDeviceError(
                "R07 Motion/environment identity changed after construction"
            )
        scene = getattr(self._env, "scene", None)
        try:
            robot = scene["robot"]
        except (KeyError, TypeError, AttributeError):
            robot = getattr(scene, "robot", None)
        sensors = getattr(scene, "sensors", None)
        mapped_sensor = None
        if isinstance(sensors, Mapping):
            mapped_sensor = sensors.get("contact_forces")
        try:
            indexed_sensor = scene["contact_forces"]
        except (KeyError, TypeError, AttributeError):
            indexed_sensor = getattr(scene, "contact_forces", None)
        sensor = mapped_sensor if mapped_sensor is not None else indexed_sensor
        if (
            robot is not self._robot
            or getattr(self._motion_owner, "robot", None) is not self._robot
            or sensor is not self._contact_sensor
            or (
                mapped_sensor is not None
                and indexed_sensor is not None
                and mapped_sensor is not indexed_sensor
            )
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 live env/robot/contact-sensor identity changed"
            )
        robot_joint_names = tuple(
            str(name)
            for name in getattr(getattr(self._robot, "data", None), "joint_names", ())
        )
        robot_body_names = tuple(
            str(name) for name in getattr(self._robot, "body_names", ())
        )
        sensor_body_names = tuple(
            str(name) for name in getattr(self._contact_sensor, "body_names", ())
        )
        if (
            robot_joint_names != self.ordered_joint_names
            or any(
                robot_body_names[index] != name
                for index, name in zip(self._body_ids, self.ordered_body_names)
            )
            or any(
                robot_body_names[index] != name
                for index, name in zip(
                    self._foot_robot_body_ids, self.ordered_foot_names
                )
            )
            or any(
                sensor_body_names[index] != name
                for index, name in zip(
                    self._foot_sensor_body_ids, self.ordered_foot_names
                )
            )
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 live joint/body order changed after construction"
            )

    def _fallback(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _channel(
        self,
        value: object,
        *,
        shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.device != self.device
            or value.dtype != self.dtype
        ):
            return self._fallback(shape), torch.zeros_like(valid)
        return value, torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)

    def _observe_source_step(self) -> int:
        source_step = getattr(self._env, "common_step_counter", None)
        if isinstance(source_step, bool) or not isinstance(source_step, int):
            raise ContinuousRecoveryDeviceError(
                "R07 live env common_step_counter must be a plain int"
            )
        if source_step < self._last_source_step:
            raise ContinuousRecoveryDeviceError("R07 source step regressed")
        self._last_source_step = source_step
        return source_step

    def observe_idle_source_step(self) -> int:
        """Read the env chronology without touching robot/contact tensors."""

        return self._observe_source_step()

    def read(self) -> "DeviceContinuousRecoveryPlantFacts":
        """Read one same-tick fact batch; no caller verdict enters this method."""

        self._require_bound_identities()
        self._observe_source_step()

        data = getattr(self._robot, "data", None)
        n = self.num_envs
        j = len(self.ordered_joint_names)
        b = len(self.ordered_body_names)
        f = len(self.ordered_foot_names)
        specs = {
            "root_position_m": (getattr(data, "root_pos_w", None), (n, 3)),
            "root_orientation_wxyz": (getattr(data, "root_quat_w", None), (n, 4)),
            "root_linear_velocity_mps": (getattr(data, "root_lin_vel_w", None), (n, 3)),
            "root_angular_velocity_radps": (getattr(data, "root_ang_vel_w", None), (n, 3)),
            "joint_position_rad": (getattr(data, "joint_pos", None), (n, j)),
            "joint_velocity_radps": (getattr(data, "joint_vel", None), (n, j)),
        }
        values: dict[str, torch.Tensor] = {}
        valid_rows = torch.ones(n, dtype=torch.bool, device=self.device)
        for name, (source, shape) in specs.items():
            values[name], channel_valid = self._channel(source, shape=shape)
            valid_rows &= channel_valid

        body_specs = {
            "body_position_m": (getattr(data, "body_pos_w", None), 3),
            "body_orientation_wxyz": (getattr(data, "body_quat_w", None), 4),
            "body_linear_velocity_mps": (getattr(data, "body_lin_vel_w", None), 3),
            "body_angular_velocity_radps": (getattr(data, "body_ang_vel_w", None), 3),
        }
        robot_body_count = len(tuple(getattr(self._robot, "body_names", ())))
        for name, (source, width) in body_specs.items():
            full, channel_valid = self._channel(
                source, shape=(n, robot_body_count, width)
            )
            values[name] = full[:, self._body_ids, :]
            valid_rows &= channel_valid

        env_origins, origin_valid = self._channel(
            getattr(getattr(self._env, "scene", None), "env_origins", None),
            shape=(n, 3),
        )
        valid_rows &= origin_valid
        station_xy = values["root_position_m"][:, :2]

        forces, force_valid = self._channel(
            getattr(getattr(self._contact_sensor, "data", None), "net_forces_w", None),
            shape=(
                n,
                len(tuple(getattr(self._contact_sensor, "body_names", ()))),
                3,
            ),
        )
        valid_rows &= force_valid
        foot_contact_signal = forces[:, self._foot_sensor_body_ids, 2]
        foot_slip = values["body_linear_velocity_mps"].new_zeros((n, f, 2))
        full_body_velocity, full_velocity_valid = self._channel(
            getattr(data, "body_lin_vel_w", None),
            shape=(n, robot_body_count, 3),
        )
        valid_rows &= full_velocity_valid
        foot_slip.copy_(full_body_velocity[:, self._foot_robot_body_ids, :2])

        root_q = values["root_orientation_wxyz"]
        q_norm = torch.linalg.vector_norm(root_q, dim=1)
        quat_valid = torch.isfinite(q_norm) & (q_norm > torch.finfo(self.dtype).tiny)
        body_q_norm = torch.linalg.vector_norm(
            values["body_orientation_wxyz"], dim=2
        )
        quat_valid &= (
            torch.isfinite(body_q_norm)
            & (body_q_norm > torch.finfo(self.dtype).tiny)
        ).all(dim=1)
        valid_rows &= quat_valid
        safe_q = root_q / q_norm.clamp_min(torch.finfo(self.dtype).tiny).unsqueeze(1)
        root_up_z = 1.0 - 2.0 * (
            torch.square(safe_q[:, 1]) + torch.square(safe_q[:, 2])
        )
        root_tilt = torch.acos(root_up_z.clamp(-1.0, 1.0))
        root_height = values["root_position_m"][:, 2] - env_origins[:, 2]

        limits = getattr(data, "soft_joint_pos_limits", None)
        joint_limit_ok = torch.ones(n, dtype=torch.bool, device=self.device)
        if isinstance(limits, torch.Tensor):
            if tuple(limits.shape) == (j, 2):
                limits = limits.unsqueeze(0).expand(n, -1, -1)
            if (
                tuple(limits.shape) != (n, j, 2)
                or limits.device != self.device
                or limits.dtype != self.dtype
            ):
                valid_rows.zero_()
            else:
                finite_limits = torch.isfinite(limits).reshape(n, -1).all(dim=1)
                ordered_limits = (limits[:, :, 0] <= limits[:, :, 1]).all(dim=1)
                joint_limit_ok = (
                    (values["joint_position_rad"] >= limits[:, :, 0])
                    & (values["joint_position_rad"] <= limits[:, :, 1])
                ).all(dim=1)
                valid_rows &= finite_limits & ordered_limits

        hard_safety_ok = (
            valid_rows
            & (root_height >= DIAGNOSTIC_N2_MIN_ROOT_HEIGHT_M)
            & (root_tilt <= DIAGNOSTIC_N2_MAX_ROOT_TILT_RAD)
            & joint_limit_ok
        )
        return DeviceContinuousRecoveryPlantFacts(
            **values,
            station_xy_m=station_xy,
            foot_contact_signal=foot_contact_signal,
            foot_slip_velocity_xy_mps=foot_slip,
            facts_valid=valid_rows,
            hard_safety_ok=hard_safety_ok,
        )


def _diagnostic_n2_provenance_digest(label: str) -> str:
    return hashlib.sha256(
        f"action_ball_r07_diagnostic_n2_nonpromotable:{label}".encode("utf-8")
    ).hexdigest()


def _exact_action_epoch_types(
    action_epoch_owner: object,
) -> tuple[type, type, type, dict[str, object]]:
    """Recover exact epoch types from the owner's defining class globals."""

    owner_type = type(action_epoch_owner)
    current_method = vars(owner_type).get("current")
    defining_globals = (
        current_method.__globals__
        if type(current_method) is FunctionType
        else None
    )
    record_type = (
        defining_globals.get("ActionEpochRecord")
        if type(defining_globals) is dict
        else None
    )
    lifecycle_type = (
        defining_globals.get("RecoveryReferenceLifecycleMasks")
        if type(defining_globals) is dict
        else None
    )
    if (
        type(defining_globals) is not dict
        or defining_globals.get("__name__") != owner_type.__module__
        or defining_globals.get("ActionEpochOwner") is not owner_type
        or defining_globals.get("row_identity") is not _row_identity
        or type(record_type) is not type
        or type(lifecycle_type) is not type
    ):
        raise ContinuousRecoveryConstructionHold(
            f"{DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON}:"
            "ActionEpochOwner_defining_types_differ"
        )
    return owner_type, record_type, lifecycle_type, defining_globals


def _exact_motion_reference_producer_definition(
    motion_owner: object,
) -> FunctionType:
    """Return only the code-defined Motion frame-0 producer method."""

    motion_type = type(motion_owner)
    declared = vars(motion_type).get(
        "project_action_ball_full_mdp_recovery_ready_reference"
    )
    defining_globals = (
        declared.__globals__ if type(declared) is FunctionType else None
    )
    try:
        owner_source = inspect.getsourcefile(motion_type)
        method_source = inspect.getsourcefile(declared)
    except (TypeError, OSError):
        owner_source = None
        method_source = None
    if (
        motion_type.__name__ != "MotionCommand"
        or motion_type.__qualname__ != "MotionCommand"
        or owner_source is None
        or Path(owner_source).resolve() != _COMMANDS_SOURCE
        or type(declared) is not FunctionType
        or declared.__name__
        != "project_action_ball_full_mdp_recovery_ready_reference"
        or declared.__qualname__
        != (
            "MotionCommand."
            "project_action_ball_full_mdp_recovery_ready_reference"
        )
        or method_source is None
        or Path(method_source).resolve() != _COMMANDS_SOURCE
        or type(defining_globals) is not dict
        or defining_globals.get("MotionCommand") is not motion_type
        or defining_globals.get("__name__") != motion_type.__module__
    ):
        raise ContinuousRecoveryDeviceError(
            "R07 exact Motion frame-0 producer definition differs"
        )
    return declared


def _exact_motion_parent_projection_definition(
    authority: object,
) -> FunctionType:
    """Return the code-defined independent parent projection method."""

    authority_type = type(authority)
    declared = vars(authority_type).get("project_bound_action_identity")
    defining_globals = (
        declared.__globals__ if type(declared) is FunctionType else None
    )
    try:
        owner_source = inspect.getsourcefile(authority_type)
        method_source = inspect.getsourcefile(declared)
    except (TypeError, OSError):
        owner_source = None
        method_source = None
    if (
        authority_type
        is not _motion_cadence.DiagnosticMotionParentScheduleAuthority
        or owner_source is None
        or Path(owner_source).resolve() != _MOTION_CADENCE_SOURCE
        or type(declared) is not FunctionType
        or declared.__name__ != "project_bound_action_identity"
        or declared.__qualname__
        != (
            "DiagnosticMotionParentScheduleAuthority."
            "project_bound_action_identity"
        )
        or method_source is None
        or Path(method_source).resolve() != _MOTION_CADENCE_SOURCE
        or type(defining_globals) is not dict
        or defining_globals.get("DiagnosticMotionParentScheduleAuthority")
        is not authority_type
        or defining_globals.get("__name__") != authority_type.__module__
    ):
        raise ContinuousRecoveryDeviceError(
            "R07 exact Motion parent projection definition differs"
        )
    return declared


def construct_action_ball_full_mdp_diagnostic_n2_recovery_owner(
    *,
    env: object = None,
    motion_owner: object,
    action_epoch_owner: object = None,
    motion_parent_authority: object,
    motion_parent_receipt: object,
) -> DiagnosticN2ContinuousRecoveryBundle:
    """Build the real N=2 diagnostic R07 owner and live plant adapter.

    Digest fields in the legacy profile are deterministic provenance labels,
    never safety gates.  Construction binds one exact live env, Motion term,
    articulation and contact sensor; every readiness input is read directly.
    """

    def hold(reason: str) -> NoReturn:
        raise ContinuousRecoveryConstructionHold(
            f"{DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON}:{reason}"
        )

    if motion_owner is None:
        raise ContinuousRecoveryConstructionHold(
            "diagnostic N2 R07 construction requires the real Motion owner"
        )
    if env is None:
        hold("live_env_required")
    if action_epoch_owner is None:
        hold("exact_cold_ActionEpochOwner_required")
    if (
        type(motion_parent_authority)
        is not _motion_cadence.DiagnosticMotionParentScheduleAuthority
        or type(motion_parent_receipt)
        is not _motion_cadence.DiagnosticMotionProfileReceipt
    ):
        hold("exact_Motion_parent_authority_or_receipt_required")
    try:
        _exact_motion_reference_producer_definition(motion_owner)
    except ContinuousRecoveryDeviceError as exc:
        raise ContinuousRecoveryConstructionHold(
            f"{DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON}:"
            "exact_Motion_reference_producer_required"
        ) from exc
    epoch_owner_type, epoch_record_type, lifecycle_type, _epoch_globals = (
        _exact_action_epoch_types(action_epoch_owner)
    )
    genesis_record = (
        action_epoch_owner.current()
        if type(action_epoch_owner) is epoch_owner_type
        else None
    )
    if (
        type(action_epoch_owner) is not epoch_owner_type
        or action_epoch_owner.num_envs != getattr(env, "num_envs", None)
        or action_epoch_owner.device != torch.device(getattr(env, "device", "cpu"))
        or action_epoch_owner.poisoned
        or action_epoch_owner.commit_head != 1
        or action_epoch_owner.drain_frontier != 0
        or genesis_record is None
        or type(epoch_record_type) is not type
        or type(genesis_record) is not epoch_record_type
        or genesis_record.version != 0
    ):
        hold("canonical_genesis_IDLE_ActionEpochOwner_differs")
    try:
        declared_parent_projection = (
            _exact_motion_parent_projection_definition(
                motion_parent_authority
            )
        )
    except ContinuousRecoveryDeviceError as exc:
        raise ContinuousRecoveryConstructionHold(
            f"{DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON}:"
            "exact_Motion_parent_projection_method_required"
        ) from exc
    project_parent_identity = getattr(
        motion_parent_authority, "project_bound_action_identity", None
    )
    if (
        not callable(project_parent_identity)
        or project_parent_identity.__self__ is not motion_parent_authority
        or project_parent_identity.__func__ is not declared_parent_projection
    ):
        hold("exact_Motion_parent_projection_method_required")
    try:
        parent_identity = project_parent_identity(
            motion_parent_receipt,
            motion_owner=motion_owner,
        )
    except Exception as exc:
        raise ContinuousRecoveryConstructionHold(
            f"{DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON}:"
            "exact_Motion_parent_projection_failed"
        ) from exc
    if (
        type(parent_identity)
        is not _motion_cadence.DiagnosticMotionParentActionIdentity
        or parent_identity.authority is not motion_parent_authority
        or parent_identity.motion_owner is not motion_owner
    ):
        hold("exact_Motion_parent_projection_identity_differs")
    if (
        getattr(motion_owner, "_env", None) is not env
    ):
        hold("exact_live_motion_identity_differs")
    scene = getattr(env, "scene", None)
    try:
        robot = scene["robot"]
    except (KeyError, TypeError, AttributeError):
        robot = getattr(scene, "robot", None)
    if robot is None or getattr(motion_owner, "robot", None) is not robot:
        hold("motion_and_scene_robot_identity_differs")
    sensors = getattr(scene, "sensors", None)
    mapped_sensor = sensors.get("contact_forces") if isinstance(sensors, Mapping) else None
    try:
        indexed_sensor = scene["contact_forces"]
    except (KeyError, TypeError, AttributeError):
        indexed_sensor = getattr(scene, "contact_forces", None)
    if (
        mapped_sensor is not None
        and indexed_sensor is not None
        and mapped_sensor is not indexed_sensor
    ):
        hold("contact_sensor_identity_is_ambiguous")
    contact_sensor = mapped_sensor if mapped_sensor is not None else indexed_sensor
    if contact_sensor is None:
        hold("live_contact_forces_sensor_absent")

    data = getattr(robot, "data", None)
    joint_names = tuple(str(name) for name in getattr(data, "joint_names", ()))
    robot_body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    sensor_body_names = tuple(
        str(name) for name in getattr(contact_sensor, "body_names", ())
    )
    if (
        not joint_names
        or len(set(joint_names)) != len(joint_names)
        or not robot_body_names
        or len(set(robot_body_names)) != len(robot_body_names)
        or not sensor_body_names
        or len(set(sensor_body_names)) != len(sensor_body_names)
    ):
        hold("live_joint_or_body_order_is_missing_or_duplicated")
    try:
        robot_module = importlib.import_module("whole_body_tracking.robots.agibot_a3")
        ordered_body_names = tuple(robot_module.A3_UPPER_TRACKED)
        ordered_foot_names = tuple(robot_module.A3_FEET_BODIES)
    except (ImportError, AttributeError, TypeError) as exc:
        raise ContinuousRecoveryConstructionHold(
            f"{DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON}:A3_name_source_unavailable"
        ) from exc
    if any(name not in robot_body_names for name in ordered_body_names):
        hold("A3_recovery_body_missing_from_live_robot")
    if any(name not in sensor_body_names for name in ordered_foot_names):
        hold("A3_foot_missing_from_live_contact_sensor")

    device = torch.device(getattr(env, "device", "cpu"))
    num_envs = getattr(env, "num_envs", None)
    root_position = getattr(data, "root_pos_w", None)
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or not isinstance(root_position, torch.Tensor)
        or tuple(root_position.shape) != (num_envs, 3)
        or root_position.device != device
        or not root_position.dtype.is_floating_point
    ):
        hold("live_root_tensor_or_environment_shape_differs")
    dtype = root_position.dtype
    profile = _runtime.ContinuousRecoveryProfile(
        continuous_contract_authority_sha256=_diagnostic_n2_provenance_digest("continuous_contract"),
        recovery_contract_authority_sha256=_diagnostic_n2_provenance_digest("recovery_contract"),
        transaction_contract_authority_sha256=_diagnostic_n2_provenance_digest("transaction_contract"),
        source_sha256=_diagnostic_n2_provenance_digest("source"),
        config_sha256=_diagnostic_n2_provenance_digest("numeric_config"),
        plant_fact_schema_sha256=_diagnostic_n2_provenance_digest("live_isaac_plant_fact_schema"),
        ordered_joint_names=joint_names,
        ordered_body_names=ordered_body_names,
        ordered_foot_names=ordered_foot_names,
        position_frame="isaac_world_xyz_m",
        orientation_frame="isaac_world_body_quaternion",
        quaternion_order="wxyz",
        reference_semantics=_runtime.REFERENCE_KIND,
        station_anchor_semantics=_runtime.STATION_ANCHOR_KIND,
        support_signal_semantics=_runtime.SUPPORT_SIGNAL_KIND,
        policy_rate_hz=POLICY_RATE_HZ,
        recovery_start_age_tick=RECOVERY_START_AGE_TICK,
        recovery_end_age_tick=RECOVERY_END_AGE_TICK,
        component_weights=DIAGNOSTIC_N2_COMPONENT_WEIGHTS,
        component_scales=DIAGNOSTIC_N2_COMPONENT_SCALES,
        component_reductions=dict(_runtime.REQUIRED_COMPONENT_REDUCTIONS),
        ready_tolerances=DIAGNOSTIC_N2_READY_TOLERANCES,
        support_contact_threshold=DIAGNOSTIC_N2_SUPPORT_FORCE_Z_N,
        minimum_supported_feet=len(ordered_foot_names),
        ready_dwell_ticks=2,
        reward_weight=0.7,
    )
    owner = ContinuousRecoveryDeviceCoordinator(
        profile=profile,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
    )
    lifecycle_method = getattr(
        epoch_record_type, "recovery_reference_lifecycle_masks", None
    )
    if (
        type(lifecycle_type) is not type
        or not callable(lifecycle_method)
    ):
        hold("ActionEpoch_reference_lifecycle_schema_differs")
    adapter = DiagnosticN2ContinuousRecoveryPlantFactAdapter(
        env=env,
        motion_owner=motion_owner,
        robot=robot,
        contact_sensor=contact_sensor,
        ordered_joint_names=joint_names,
        ordered_body_names=ordered_body_names,
        ordered_foot_names=ordered_foot_names,
        body_ids=tuple(robot_body_names.index(name) for name in ordered_body_names),
        foot_robot_body_ids=tuple(robot_body_names.index(name) for name in ordered_foot_names),
        foot_sensor_body_ids=tuple(sensor_body_names.index(name) for name in ordered_foot_names),
        num_envs=num_envs,
        device=device,
        dtype=dtype,
    )
    bundle = DiagnosticN2ContinuousRecoveryBundle(
        owner=owner,
        plant_fact_adapter=adapter,
        motion_owner=motion_owner,
        action_epoch_owner=action_epoch_owner,
        motion_parent_authority=motion_parent_authority,
        motion_parent_receipt=motion_parent_receipt,
    )
    owner._diagnostic_n2_bundle = bundle
    action_epoch_owner.bind_fact_owner("r07_recovery", bundle)
    return bundle


def _expand(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    return mask.reshape((mask.shape[0],) + (1,) * (ndim - 1))


def _masked_copy_(
    destination: torch.Tensor, source: torch.Tensor, mask: torch.Tensor
) -> None:
    destination.copy_(torch.where(_expand(mask, destination.ndim), source, destination))


def _masked_zero_(tensor: torch.Tensor, mask: torch.Tensor) -> None:
    tensor.masked_fill_(_expand(mask, tensor.ndim), 0)


@dataclass(frozen=True)
class DeviceLandingOutcomeShotKey:
    """Lossless C05 fourteen-field key plus its all-field device digest.

    The six integer and six existing digest fields stay individually visible
    on device.  The two variable-length text fields remain lossless host
    tuples and additionally have UTF-8 SHA-256 tensors for device comparison.
    ``canonical_sha256`` covers all fourteen original fields and is the hot
    path owner token.  Keeping only the legacy eight-field task ref is not a
    valid construction of this type.
    """

    env_id: torch.Tensor
    reset_generation: torch.Tensor
    swing_generation: torch.Tensor
    action_uid: torch.Tensor
    action_slot: torch.Tensor
    birth_sha256: torch.Tensor
    sample_sha256: torch.Tensor
    task_sha256: torch.Tensor
    run_id: tuple[str, ...]
    carry_chain_id: tuple[str, ...]
    shot_index: torch.Tensor
    source_sha256: torch.Tensor
    config_sha256: torch.Tensor
    receipt_content_sha256: torch.Tensor
    run_id_utf8_sha256: torch.Tensor
    carry_chain_id_utf8_sha256: torch.Tensor
    canonical_sha256: torch.Tensor
    host_keys: tuple[object, ...]

    @classmethod
    def from_host_keys(
        cls,
        values: Sequence[object],
        *,
        device: torch.device | str,
    ) -> "DeviceLandingOutcomeShotKey":
        keys = tuple(_runtime.coerce_landing_outcome_shot_key(value) for value in values)
        target = torch.device(device)
        ints = lambda name: torch.tensor(  # noqa: E731
            [getattr(key, name) for key in keys], dtype=torch.int64, device=target
        )
        digests = lambda name: _digest_tensor(  # noqa: E731
            [getattr(key, name) for key in keys], device=target
        )
        run_ids = tuple(key.run_id for key in keys)
        carry_ids = tuple(key.carry_chain_id for key in keys)
        return cls(
            env_id=ints("env_id"),
            reset_generation=ints("reset_generation"),
            swing_generation=ints("swing_generation"),
            action_uid=ints("action_uid"),
            action_slot=ints("action_slot"),
            birth_sha256=digests("birth_sha256"),
            sample_sha256=digests("sample_sha256"),
            task_sha256=digests("task_sha256"),
            run_id=run_ids,
            carry_chain_id=carry_ids,
            shot_index=ints("shot_index"),
            source_sha256=digests("source_sha256"),
            config_sha256=digests("config_sha256"),
            receipt_content_sha256=digests("receipt_content_sha256"),
            run_id_utf8_sha256=_digest_tensor(
                [_host_text_digest(value) for value in run_ids], device=target
            ),
            carry_chain_id_utf8_sha256=_digest_tensor(
                [_host_text_digest(value) for value in carry_ids], device=target
            ),
            canonical_sha256=_digest_tensor(
                [key.canonical_sha256 for key in keys], device=target
            ),
            host_keys=keys,
        )


@dataclass(frozen=True)
class DeviceContinuousRecoveryReference:
    """K-row completed-action frame-0 reference installed at a reveal.

    There are intentionally no target, ball, deadline, or velocity-reference
    fields.  Joint/root/body reference velocities are literal zero in the
    coordinator's error computation.
    """

    reference_sha256: tuple[str, ...]
    root_position_m: torch.Tensor
    root_orientation_wxyz: torch.Tensor
    joint_position_rad: torch.Tensor
    body_position_m: torch.Tensor
    body_orientation_wxyz: torch.Tensor
    station_anchor_xy_m: torch.Tensor


@dataclass(frozen=True)
class DeviceContinuousRecoveryPlantFacts:
    """All-environment post-physics common plant facts."""

    root_position_m: torch.Tensor
    root_orientation_wxyz: torch.Tensor
    root_linear_velocity_mps: torch.Tensor
    root_angular_velocity_radps: torch.Tensor
    joint_position_rad: torch.Tensor
    joint_velocity_radps: torch.Tensor
    body_position_m: torch.Tensor
    body_orientation_wxyz: torch.Tensor
    body_linear_velocity_mps: torch.Tensor
    body_angular_velocity_radps: torch.Tensor
    station_xy_m: torch.Tensor
    foot_contact_signal: torch.Tensor
    foot_slip_velocity_xy_mps: torch.Tensor
    facts_valid: torch.Tensor
    hard_safety_ok: torch.Tensor


@dataclass(frozen=True)
class DeviceContinuousRecoveryCommandProjection:
    """Current-only R04 projection; no future question fields exist."""

    source_step: torch.Tensor
    episode_tick: torch.Tensor
    phase_code: torch.Tensor
    reference_active: torch.Tensor
    motion_active: torch.Tensor
    suffix_complete: torch.Tensor
    deadline_due: torch.Tensor
    scheduled_ordinal: torch.Tensor
    current_deadline_tick: torch.Tensor
    current_task_key_sha256: torch.Tensor


@dataclass(frozen=True)
class DeviceContinuousRecoveryDoneTermProjection:
    """Nonterminating publisher result for a shot/recovery boundary."""

    terminal_requested: torch.Tensor
    truncation_requested: torch.Tensor
    physical_reset_requested: torch.Tensor
    carry_reset_requested: torch.Tensor
    pose_teleport_requested: torch.Tensor


@dataclass(frozen=True)
class DeviceContinuousRecoveryPaymentIdentity:
    """Exact per-tick idempotency tuple; it is not merely a task key."""

    profile_sha256: torch.Tensor
    task_key_sha256: torch.Tensor
    recovery_age_tick: torch.Tensor
    source_step: torch.Tensor
    consumer: str


@dataclass(frozen=True)
class SharedContinuousRecoveryRewardView:
    """One immutable-by-contract cache shared by the recovery RewardTerm."""

    source_step: torch.Tensor
    episode_tick: torch.Tensor
    phase_code: torch.Tensor
    owner_key_sha256: torch.Tensor
    reference_owner_sha256: torch.Tensor
    recovery_age_tick: torch.Tensor
    recovery_expected: torch.Tensor
    reward_eligible: torch.Tensor
    facts_valid: torch.Tensor
    infrastructure_fault: torch.Tensor
    component_errors: torch.Tensor
    component_scores: torch.Tensor
    raw_score: torch.Tensor
    weighted_reward: torch.Tensor
    ready_instant: torch.Tensor
    ready_live: torch.Tensor
    ready_streak: torch.Tensor
    payment_identity: DeviceContinuousRecoveryPaymentIdentity


@dataclass(frozen=True)
class ContinuousRecoveryWindowLedgerRow:
    """One env's exact current/last recovery-window closure evidence."""

    env_id: int
    owner_key_sha256: Optional[str]
    expected_count: int
    eligible_count: int
    payment_count: int
    first_expected_age_tick: Optional[int]
    last_expected_age_tick: Optional[int]
    last_paid_age_tick: Optional[int]
    closed_68_of_68: bool


@dataclass(frozen=True)
class ContinuousRecoveryBoundaryReceipt:
    """The coordinator's single device-to-host PPO-boundary evidence."""

    schema_version: int
    update_index: int
    drain_sequence: int
    mutation_version: int
    num_envs: int
    component_names: tuple[str, ...]
    fault_counts: tuple[tuple[str, int], ...]
    recovery_window_rows: tuple[ContinuousRecoveryWindowLedgerRow, ...]
    recovery_expected_total: int
    reward_eligible_total: int
    reward_payment_total: int
    reward_income_total: float
    ready_instant_total: int
    first_ready_total: int
    played_deadline_total: int
    unplayed_deadline_total: int
    checkpoint_safe: bool
    device_to_host_transfers: int
    integration_status: str
    runtime_wiring_connected: bool
    cuda_profiled: bool
    formal_exact_resume_integrated: bool
    launch_authorized: bool


@dataclass
class _PreparedR07GlobalDrain:
    """Leaf-owned lease retained until the global optimizer ACK."""

    pack: object
    authority: object
    update_index: int
    completed_environment_steps: int
    mutation_version: int
    window_owner_sha256: tuple[Optional[str], ...]
    stage: str = "prepared"


class ContinuousRecoveryDeviceCoordinator:
    """All-environment common recovery/reference/reward authority."""

    def __init__(
        self,
        *,
        profile: _runtime.ContinuousRecoveryProfile,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(profile, _runtime.ContinuousRecoveryProfile):
            raise ContinuousRecoveryDeviceError(
                "profile must be ContinuousRecoveryProfile"
            )
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ContinuousRecoveryDeviceError("num_envs must be a positive int")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ContinuousRecoveryDeviceError("dtype must be a floating torch dtype")
        if profile.policy_rate_hz != POLICY_RATE_HZ:
            raise ContinuousRecoveryDeviceError("profile policy rate differs from 50 Hz")
        if (
            profile.recovery_start_age_tick != RECOVERY_START_AGE_TICK
            or profile.recovery_end_age_tick != RECOVERY_END_AGE_TICK
        ):
            raise ContinuousRecoveryDeviceError("profile recovery age window differs")

        self.profile = profile
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.component_names = tuple(_runtime.COMPONENT_NAMES)
        self.num_components = len(self.component_names)
        self.num_joints = len(profile.ordered_joint_names)
        self.num_bodies = len(profile.ordered_body_names)
        self.num_feet = len(profile.ordered_foot_names)
        self._component_index = {
            name: index for index, name in enumerate(self.component_names)
        }
        self._weights = torch.tensor(
            [profile.component_weights[name] for name in self.component_names],
            dtype=dtype,
            device=self.device,
        )
        self._scales = torch.tensor(
            [profile.component_scales[name] for name in self.component_names],
            dtype=dtype,
            device=self.device,
        )
        self._ready_tolerances = torch.tensor(
            [profile.ready_tolerances[name] for name in self.component_names],
            dtype=dtype,
            device=self.device,
        )
        self._weight_sum = self._weights.sum()
        profile_sha_row = _digest_tensor(
            [profile.canonical_sha256], device=self.device
        )
        self._profile_sha = profile_sha_row.expand(num_envs, -1).clone()

        n = num_envs
        bo = {"dtype": torch.bool, "device": self.device}
        lo = {"dtype": torch.int64, "device": self.device}
        fo = {"dtype": dtype, "device": self.device}
        self._env_ids = torch.arange(n, **lo)
        self._sequence_active = torch.zeros(n, **bo)
        self._reset_generation = torch.zeros(n, **lo)
        self._episode_tick = torch.full((n,), -1, **lo)
        self._source_step = torch.full((n,), -1, **lo)
        self._phase_code = torch.full((n,), PHASE_PRE_REVEAL_HIDDEN, **lo)
        self._scheduled_ordinal = torch.full((n,), -1, **lo)
        self._current_reveal_tick = torch.full((n,), -1, **lo)
        self._current_deadline_tick = torch.full((n,), -1, **lo)
        self._current_key_sha = torch.zeros((n, 32), dtype=torch.uint8, device=self.device)
        self._current_key_valid = torch.zeros(n, **bo)
        self._reward_owner_sha = torch.zeros_like(self._current_key_sha)
        self._reward_owner_valid = torch.zeros(n, **bo)
        self._reward_deadline_tick = torch.full((n,), -1, **lo)
        self._reward_owner_played = torch.zeros(n, **bo)
        # These cursors belong to one exact reward owner per env.  They may
        # never be pooled across envs or shots: age 10 starts the window,
        # ages then advance one-by-one through 77, and even an ineligible
        # zero-reward cell must receive one exact payment acknowledgement.
        self._window_owner_sha = torch.zeros_like(self._current_key_sha)
        self._window_owner_valid = torch.zeros(n, **bo)
        self._window_expected_count = torch.zeros(n, **lo)
        self._window_eligible_count = torch.zeros(n, **lo)
        self._window_payment_count = torch.zeros(n, **lo)
        self._window_first_expected_age = torch.full((n,), -1, **lo)
        self._window_last_expected_age = torch.full((n,), -1, **lo)
        self._window_last_paid_age = torch.full((n,), -1, **lo)
        self._deadline_ack_pending = torch.zeros(n, **bo)
        self._ever_motion_active = torch.zeros(n, **bo)
        self._playback_receipted = torch.zeros(n, **bo)
        self._suffix_complete = torch.zeros(n, **bo)
        self._reference_active = torch.zeros(n, **bo)
        self._motion_active = torch.zeros(n, **bo)

        self._reference_owner_kind = torch.zeros(n, **lo)
        self._reference_owner_sha = torch.zeros_like(self._current_key_sha)
        self._reference_sha = torch.zeros_like(self._current_key_sha)
        self._reference_valid = torch.zeros(n, **bo)
        self._reference_root_position = torch.zeros((n, 3), **fo)
        self._reference_root_orientation = torch.zeros((n, 4), **fo)
        self._reference_joint_position = torch.zeros((n, self.num_joints), **fo)
        self._reference_body_position = torch.zeros((n, self.num_bodies, 3), **fo)
        self._reference_body_orientation = torch.zeros((n, self.num_bodies, 4), **fo)
        self._station_anchor_xy = torch.zeros((n, 2), **fo)

        # A reveal stages the new task's frame-0 reference here.  It is not
        # actor-visible readiness material until the exact played key finishes
        # its full suffix.  Unplayed rows never promote this staging area.
        self._pending_reference_sha = torch.zeros_like(self._current_key_sha)
        self._pending_reference_valid = torch.zeros(n, **bo)
        self._pending_reference_key_sha = torch.zeros_like(self._current_key_sha)
        self._pending_reference_root_position = torch.zeros((n, 3), **fo)
        self._pending_reference_root_orientation = torch.zeros((n, 4), **fo)
        self._pending_reference_joint_position = torch.zeros(
            (n, self.num_joints), **fo
        )
        self._pending_reference_body_position = torch.zeros(
            (n, self.num_bodies, 3), **fo
        )
        self._pending_reference_body_orientation = torch.zeros(
            (n, self.num_bodies, 4), **fo
        )
        self._pending_station_anchor_xy = torch.zeros((n, 2), **fo)

        # This object is bound once by identity to Motion.  It is never replaced.
        self._ready_authority = torch.zeros(n, **bo)
        self._ready_instant = torch.zeros(n, **bo)
        self._ready_streak = torch.zeros(n, **lo)
        self._first_ready_tick = torch.full((n,), -1, **lo)

        # Fresh ActionEpoch readiness is intentionally separate from the
        # legacy sequence/payment ledger above.  It is still owner-private and
        # uses the same two-consecutive-post-physics dwell rule, but it must
        # not fabricate legacy reference/key state merely to satisfy those
        # retired invariants.
        self._action_epoch_ready_instant = torch.zeros(n, **bo)
        self._action_epoch_ready_live = torch.zeros(n, **bo)
        self._action_epoch_ready_streak = torch.zeros(n, **lo)
        self._action_epoch_ready_reference_kind = torch.zeros(n, **lo)
        self._action_epoch_ready_reference_action_slot = torch.full(
            (n,), -1, **lo
        )
        self._action_epoch_ready_reference_action_uid = torch.full(
            (n,), -1, **lo
        )
        self._action_epoch_ready_shot_key = (
            _row_identity.empty_action_epoch_shot_key(
                (n,), device=self.device
            )
        )
        self._action_epoch_first_ready_source_step = torch.full((n,), -1, **lo)
        self._action_epoch_ready_last_motion_cadence_tick = torch.full(
            (n,), -1, **lo
        )
        self._action_epoch_ready_last_reset_generation = torch.full(
            (n,), -1, **lo
        )
        self._action_epoch_ready_last_source_step = -1
        self._idle_observation_published = False
        self._idle_observation_source_step = -1
        self._idle_observation_reset_generation = torch.full((n,), -1, **lo)
        self._idle_observation_motion_cadence_tick = torch.full((n,), -1, **lo)

        self._fault_bits = torch.zeros(n, **lo)
        self._cache_pending = torch.zeros(n, **bo)
        self._cache_viewed = torch.zeros(n, **bo)
        self._cache_paid = torch.zeros(n, **bo)
        self._cache_source_step = torch.full((n,), -1, **lo)
        self._cache_episode_tick = torch.full((n,), -1, **lo)
        self._cache_phase_code = torch.full((n,), PHASE_PRE_REVEAL_HIDDEN, **lo)
        self._cache_owner_sha = torch.zeros_like(self._current_key_sha)
        self._cache_reference_owner_sha = torch.zeros_like(self._current_key_sha)
        self._cache_age_tick = torch.full((n,), -1, **lo)
        self._cache_expected = torch.zeros(n, **bo)
        self._cache_payment_required = torch.zeros(n, **bo)
        self._cache_eligible = torch.zeros(n, **bo)
        self._cache_facts_valid = torch.zeros(n, **bo)
        self._cache_infrastructure_fault = torch.zeros(n, **bo)
        self._cache_errors = torch.zeros((n, self.num_components), **fo)
        self._cache_scores = torch.zeros_like(self._cache_errors)
        self._cache_raw_score = torch.zeros(n, **fo)
        self._cache_weighted_reward = torch.zeros(n, **fo)
        self._cache_ready_instant = torch.zeros(n, **bo)
        self._cache_ready_live = torch.zeros(n, **bo)
        self._cache_ready_streak = torch.zeros(n, **lo)
        self._shared_view = SharedContinuousRecoveryRewardView(
            source_step=self._cache_source_step,
            episode_tick=self._cache_episode_tick,
            phase_code=self._cache_phase_code,
            owner_key_sha256=self._cache_owner_sha,
            reference_owner_sha256=self._cache_reference_owner_sha,
            recovery_age_tick=self._cache_age_tick,
            recovery_expected=self._cache_expected,
            reward_eligible=self._cache_eligible,
            facts_valid=self._cache_facts_valid,
            infrastructure_fault=self._cache_infrastructure_fault,
            component_errors=self._cache_errors,
            component_scores=self._cache_scores,
            raw_score=self._cache_raw_score,
            weighted_reward=self._cache_weighted_reward,
            ready_instant=self._cache_ready_instant,
            ready_live=self._cache_ready_live,
            ready_streak=self._cache_ready_streak,
            payment_identity=DeviceContinuousRecoveryPaymentIdentity(
                profile_sha256=self._profile_sha,
                task_key_sha256=self._cache_owner_sha,
                recovery_age_tick=self._cache_age_tick,
                source_step=self._cache_source_step,
                consumer=RECOVERY_REWARD_CONSUMER,
            ),
        )
        zero = torch.zeros((), dtype=torch.int64, device=self.device)
        self._expected_total = zero.clone()
        self._eligible_total = zero.clone()
        self._payment_total = zero.clone()
        self._income_total = torch.zeros((), **fo)
        self._ready_instant_total = zero.clone()
        self._first_ready_total = zero.clone()
        self._played_deadline_total = zero.clone()
        self._unplayed_deadline_total = zero.clone()
        self._payment_epoch_open = torch.zeros((), **bo)

        self._host_keys: list[object | None] = [None] * n
        self._host_window_owner_keys: list[object | None] = [None] * n
        self._host_ready_owner_keys: list[object | None] = [None] * n
        self._host_played: list[bool] = [False] * n
        self._host_reference_sha: list[str | None] = [None] * n
        self._host_pending_reference_sha: list[str | None] = [None] * n
        self._host_reset_generation: list[int | None] = [None] * n
        self._host_scheduled_ordinal: list[int] = [-1] * n
        self._host_deadline_consumed: list[bool] = [False] * n
        self._host_last_deadline: list[int | None] = [None] * n
        self._host_last_reveal: list[int | None] = [None] * n
        self._host_payment_epoch_open = False
        self._mutation_version = 0
        self._drain_sequence = 0
        self._last_drained_update = -1
        self._last_receipt: ContinuousRecoveryBoundaryReceipt | None = None
        self._last_receipt_consumed = False
        self._last_r07_global_drain_pair: Optional[
            tuple[object, ContinuousRecoveryBoundaryReceipt]
        ] = None
        self._last_globally_acknowledged_mutation_version = -1
        self._checkpoint_requires_global_drain_ack = True
        self._active_r07_global_drain: _PreparedR07GlobalDrain | None = None
        self._r07_global_drain_poisoned = False
        self._r07_global_drain_poison_reason: str | None = None
        self._identity = object()
        self._full_mdp_reward_consumers = FULL_MDP_REWARD_CONSUMERS
        self._active_full_mdp_pre_reward: ContinuousRecoveryFullMdpRewardPublication | None = None
        self._active_full_mdp_pre_reward_payload: _FullMdpPreRewardPayload | None = None
        self._active_full_mdp_payment_verdict: ContinuousRecoveryFullMdpRewardPaymentVerdict | None = None
        self._active_full_mdp_close_receipt: ContinuousRecoveryFullMdpRewardCloseReceipt | None = None
        self._latest_motion_ready_projection: ContinuousRecoveryMotionReadyProjection | None = None
        self._diagnostic_n2_bundle: DiagnosticN2ContinuousRecoveryBundle | None = None
        self._full_mdp_reward_poisoned = False
        self._full_mdp_reward_poison_reason: str | None = None

    @property
    def ready_authority(self) -> torch.Tensor:
        """Stable in-place bool tensor consumed by the Motion bridge."""

        return self._ready_authority

    @property
    def shared_reward_view(self) -> SharedContinuousRecoveryRewardView:
        return self._shared_view

    @property
    def full_mdp_reward_consumers(self) -> tuple[str, ...]:
        return self._full_mdp_reward_consumers

    def _require_full_mdp_reward_operable(self) -> None:
        if self._full_mdp_reward_poisoned:
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward epoch is poisoned and requires cold replacement"
            )

    def _poison_full_mdp_reward(self, reason: str) -> None:
        self._full_mdp_reward_poisoned = True
        if self._full_mdp_reward_poison_reason is None:
            self._full_mdp_reward_poison_reason = reason

    @staticmethod
    def _require_runtime_owner(runtime_owner: object) -> object:
        if runtime_owner is None:
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward requires the real top runtime owner"
            )
        required = (
            "require_healthy",
            "publish_full_mdp_pre_reward",
            "require_owned_full_mdp_pre_reward",
            "close_full_mdp_reward_cycle",
        )
        if any(not callable(getattr(runtime_owner, name, None)) for name in required):
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward runtime owner API differs"
            )
        return runtime_owner

    def _ids(self, env_ids: Sequence[int], *, label: str) -> tuple[int, ...]:
        if isinstance(env_ids, (str, bytes)) or not isinstance(env_ids, Sequence):
            raise ContinuousRecoveryDeviceError(f"{label} must be an env-id sequence")
        ids = tuple(_exact_int(value, label=f"{label} row") for value in env_ids)
        if not ids or len(set(ids)) != len(ids) or any(value >= self.num_envs for value in ids):
            raise ContinuousRecoveryDeviceError(f"{label} ids are empty, duplicate, or out of range")
        return ids

    def _index(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor(tuple(ids), dtype=torch.int64, device=self.device)

    def _mask_from_ids(self, ids: Sequence[int]) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[self._index(ids)] = True
        return mask

    def _tensor(
        self,
        value: torch.Tensor,
        *,
        label: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != self.device
        ):
            raise ContinuousRecoveryDeviceError(
                f"{label} must have shape {shape}, dtype {dtype}, device {self.device}"
            )
        return value

    def _reference(
        self, value: DeviceContinuousRecoveryReference, *, rows: int
    ) -> DeviceContinuousRecoveryReference:
        if not isinstance(value, DeviceContinuousRecoveryReference):
            raise ContinuousRecoveryDeviceError(
                "reference must be DeviceContinuousRecoveryReference"
            )
        if len(value.reference_sha256) != rows:
            raise ContinuousRecoveryDeviceError("reference SHA row count differs")
        for index, digest in enumerate(value.reference_sha256):
            _sha256(digest, label=f"reference_sha256[{index}]")
        tensors = (
            ("root_position_m", value.root_position_m, (rows, 3)),
            ("root_orientation_wxyz", value.root_orientation_wxyz, (rows, 4)),
            ("joint_position_rad", value.joint_position_rad, (rows, self.num_joints)),
            ("body_position_m", value.body_position_m, (rows, self.num_bodies, 3)),
            ("body_orientation_wxyz", value.body_orientation_wxyz, (rows, self.num_bodies, 4)),
            ("station_anchor_xy_m", value.station_anchor_xy_m, (rows, 2)),
        )
        for label, tensor, shape in tensors:
            self._tensor(tensor, label=f"reference.{label}", shape=shape, dtype=self.dtype)
        return value

    def bind_sequence_birth(
        self,
        env_ids: Sequence[int],
        *,
        reset_generations: Sequence[int],
        sequence_origin_ticks: Sequence[int],
        reference: DeviceContinuousRecoveryReference,
        fault_injector: _FaultInjector = None,
    ) -> None:
        """Atomically install the birth ready reference for selected rows."""

        self._require_r07_business_mutation_allowed(label="sequence birth")
        ids = self._ids(env_ids, label="bind_sequence_birth.env_ids")
        if len(reset_generations) != len(ids) or len(sequence_origin_ticks) != len(ids):
            raise ContinuousRecoveryDeviceError(
                "birth generations/ticks must have one value per env"
            )
        generations = tuple(
            _exact_int(value, label="birth reset_generation", minimum=1)
            for value in reset_generations
        )
        origins = tuple(
            _exact_int(value, label="birth sequence_origin_tick")
            for value in sequence_origin_ticks
        )
        reference = self._reference(reference, rows=len(ids))
        if any(self._host_reset_generation[env_id] is not None for env_id in ids):
            raise ContinuousRecoveryDeviceError(
                "sequence birth may bind only a fresh or true-boundary row"
            )

        index = self._index(ids)
        birth_owner_sha = tuple(
            hashlib.sha256(
                (
                    "action_ball_continuous_birth_ready_owner_v1|"
                    f"{env_id}|{generation}|{digest}"
                ).encode("ascii")
            ).hexdigest()
            for env_id, generation, digest in zip(
                ids, generations, reference.reference_sha256
            )
        )
        generation_tensor = torch.tensor(generations, dtype=torch.int64, device=self.device)
        origin_tensor = torch.tensor(origins, dtype=torch.int64, device=self.device)
        reference_sha = _digest_tensor(reference.reference_sha256, device=self.device)
        owner_sha = _digest_tensor(birth_owner_sha, device=self.device)

        if fault_injector is not None:
            fault_injector("birth_validated_before_publish")

        self._sequence_active.index_fill_(0, index, True)
        self._reset_generation.index_copy_(0, index, generation_tensor)
        self._episode_tick.index_copy_(0, index, origin_tensor - 1)
        self._source_step.index_fill_(0, index, -1)
        self._reference_owner_kind.index_fill_(0, index, OWNER_SEQUENCE_BIRTH)
        self._reference_owner_sha.index_copy_(0, index, owner_sha)
        self._reference_sha.index_copy_(0, index, reference_sha)
        self._reference_valid.index_fill_(0, index, True)
        self._reference_root_position.index_copy_(0, index, reference.root_position_m)
        self._reference_root_orientation.index_copy_(
            0, index, reference.root_orientation_wxyz
        )
        self._reference_joint_position.index_copy_(0, index, reference.joint_position_rad)
        self._reference_body_position.index_copy_(0, index, reference.body_position_m)
        self._reference_body_orientation.index_copy_(
            0, index, reference.body_orientation_wxyz
        )
        self._station_anchor_xy.index_copy_(0, index, reference.station_anchor_xy_m)
        self._ready_authority.index_fill_(0, index, False)
        self._ready_instant.index_fill_(0, index, False)
        self._ready_streak.index_fill_(0, index, 0)
        self._first_ready_tick.index_fill_(0, index, -1)

        staged_reset = list(self._host_reset_generation)
        staged_ref = list(self._host_reference_sha)
        for env_id, generation, digest in zip(ids, generations, reference.reference_sha256):
            staged_reset[env_id] = generation
            staged_ref[env_id] = digest
        self._host_reset_generation = staged_reset
        self._host_reference_sha = staged_ref
        self._advance_mutation_version()

    @staticmethod
    def _validate_adjacent_key(previous: object, current: object) -> None:
        same_lineage_fields = (
            "env_id",
            "reset_generation",
            "action_uid",
            "action_slot",
            "birth_sha256",
            "run_id",
            "carry_chain_id",
            "source_sha256",
            "config_sha256",
        )
        if any(
            getattr(previous, name) != getattr(current, name)
            for name in same_lineage_fields
        ):
            raise ContinuousRecoveryDeviceError(
                "adjacent committed keys differ in carry/action lineage"
            )
        if current.swing_generation != previous.swing_generation + 1:
            raise ContinuousRecoveryDeviceError(
                "adjacent swing_generation must advance by one"
            )
        if current.shot_index != previous.shot_index + 1:
            raise ContinuousRecoveryDeviceError(
                "adjacent outcome shot_index must advance by one"
            )
        for name in (
            "sample_sha256",
            "task_sha256",
            "receipt_content_sha256",
        ):
            if getattr(current, name) == getattr(previous, name):
                raise ContinuousRecoveryDeviceError(
                    f"adjacent committed keys reuse {name}"
                )

    def commit_reveal(
        self,
        env_ids: Sequence[int],
        *,
        task_keys: Sequence[object],
        scheduled_ordinals: Sequence[int],
        scheduled_reveal_ticks: Sequence[int],
        scheduled_deadline_ticks: Sequence[int],
        task_frame0_reference: DeviceContinuousRecoveryReference,
        fault_injector: _FaultInjector = None,
    ) -> None:
        """Commit current identity and stage, but never expose, its frame 0.

        Readiness does not participate in this admission.  In particular this
        method neither changes ``ready_authority`` nor the active ready owner.
        """

        self._require_r07_business_mutation_allowed(label="reveal commit")
        ids = self._ids(env_ids, label="commit_reveal.env_ids")
        if not (
            len(task_keys)
            == len(scheduled_ordinals)
            == len(scheduled_reveal_ticks)
            == len(scheduled_deadline_ticks)
            == len(ids)
        ):
            raise ContinuousRecoveryDeviceError(
                "commit reveal fields must have one row per env"
            )
        keys = tuple(
            _runtime.coerce_landing_outcome_shot_key(value) for value in task_keys
        )
        ordinals = tuple(
            _exact_int(value, label="scheduled_ordinal")
            for value in scheduled_ordinals
        )
        reveals = tuple(
            _exact_int(value, label="scheduled_reveal_tick")
            for value in scheduled_reveal_ticks
        )
        deadlines = tuple(
            _exact_int(value, label="scheduled_deadline_tick")
            for value in scheduled_deadline_ticks
        )
        reference = self._reference(task_frame0_reference, rows=len(ids))

        for row, (env_id, key, ordinal, reveal, deadline) in enumerate(
            zip(ids, keys, ordinals, reveals, deadlines)
        ):
            generation = self._host_reset_generation[env_id]
            if generation is None:
                raise ContinuousRecoveryDeviceError(
                    "commit reveal precedes sequence birth"
                )
            if key.env_id != env_id or key.reset_generation != generation:
                raise ContinuousRecoveryDeviceError(
                    "committed key env/reset differs from sequence"
                )
            if (
                key.source_sha256 != self.profile.source_sha256
                or key.config_sha256 != self.profile.config_sha256
            ):
                raise ContinuousRecoveryDeviceError(
                    "committed key source/config differs from recovery profile"
                )
            if key.swing_generation != ordinal or key.shot_index != ordinal + 1:
                raise ContinuousRecoveryDeviceError(
                    "committed key violates ordinal/generation mapping"
                )
            if deadline <= reveal:
                raise ContinuousRecoveryDeviceError(
                    "committed deadline must follow reveal"
                )
            previous = self._host_keys[env_id]
            previous_ordinal = self._host_scheduled_ordinal[env_id]
            if previous is None:
                if ordinal != 0:
                    raise ContinuousRecoveryDeviceError(
                        "first committed scheduled ordinal must be zero"
                    )
            else:
                if ordinal != previous_ordinal + 1:
                    raise ContinuousRecoveryDeviceError(
                        "scheduled ordinal must advance by one"
                    )
                if not self._host_deadline_consumed[env_id]:
                    raise ContinuousRecoveryDeviceError(
                        "successor reveal precedes prior deadline consumption"
                    )
                self._validate_adjacent_key(previous, key)
                prior_deadline = self._host_last_deadline[env_id]
                if prior_deadline is None or reveal < prior_deadline + 78:
                    raise ContinuousRecoveryDeviceError(
                        "successor reveal truncates prior age 10..77 window"
                    )
            previous_reveal = self._host_last_reveal[env_id]
            if previous_reveal is not None and reveal <= previous_reveal:
                raise ContinuousRecoveryDeviceError(
                    "scheduled reveal ticks must strictly advance"
                )
            _sha256(
                reference.reference_sha256[row], label="task frame0 reference SHA"
            )

        device_keys = DeviceLandingOutcomeShotKey.from_host_keys(
            keys, device=self.device
        )
        index = self._index(ids)
        requires_closure = torch.tensor(
            [self._host_keys[env_id] is not None for env_id in ids],
            dtype=torch.bool,
            device=self.device,
        )
        selected_window_owner_match = torch.all(
            self._window_owner_sha.index_select(0, index)
            == self._current_key_sha.index_select(0, index),
            dim=1,
        )
        selected_window_closed = (
            self._window_owner_valid.index_select(0, index)
            & selected_window_owner_match
            & (
                self._window_expected_count.index_select(0, index)
                == RECOVERY_SAMPLE_COUNT
            )
            & (
                self._window_payment_count.index_select(0, index)
                == RECOVERY_SAMPLE_COUNT
            )
            & (
                self._window_first_expected_age.index_select(0, index)
                == RECOVERY_START_AGE_TICK
            )
            & (
                self._window_last_expected_age.index_select(0, index)
                == RECOVERY_END_AGE_TICK
            )
            & (
                self._window_last_paid_age.index_select(0, index)
                == RECOVERY_END_AGE_TICK
            )
        )
        closure_bad_rows = requires_closure & ~selected_window_closed
        closure_bad = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        closure_bad.index_copy_(0, index, closure_bad_rows)
        ordinal_tensor = torch.tensor(ordinals, dtype=torch.int64, device=self.device)
        reveal_tensor = torch.tensor(reveals, dtype=torch.int64, device=self.device)
        deadline_tensor = torch.tensor(deadlines, dtype=torch.int64, device=self.device)
        pending_reference_sha = _digest_tensor(
            reference.reference_sha256, device=self.device
        )

        if fault_injector is not None:
            fault_injector("reveal_validated_before_publish")

        self._fault_bits.bitwise_or_(
            closure_bad.to(dtype=torch.int64) * FAULT_LEDGER_SEQUENCE
        )
        self._ready_authority.bitwise_and_(self._fault_bits == 0)
        self._scheduled_ordinal.index_copy_(0, index, ordinal_tensor)
        self._current_reveal_tick.index_copy_(0, index, reveal_tensor)
        self._current_deadline_tick.index_copy_(0, index, deadline_tensor)
        self._current_key_sha.index_copy_(0, index, device_keys.canonical_sha256)
        self._current_key_valid.index_fill_(0, index, True)
        self._ever_motion_active.index_fill_(0, index, False)
        self._playback_receipted.index_fill_(0, index, False)
        self._suffix_complete.index_fill_(0, index, False)
        self._pending_reference_sha.index_copy_(0, index, pending_reference_sha)
        self._pending_reference_key_sha.index_copy_(
            0, index, device_keys.canonical_sha256
        )
        self._pending_reference_valid.index_fill_(0, index, True)
        self._pending_reference_root_position.index_copy_(
            0, index, reference.root_position_m
        )
        self._pending_reference_root_orientation.index_copy_(
            0, index, reference.root_orientation_wxyz
        )
        self._pending_reference_joint_position.index_copy_(
            0, index, reference.joint_position_rad
        )
        self._pending_reference_body_position.index_copy_(
            0, index, reference.body_position_m
        )
        self._pending_reference_body_orientation.index_copy_(
            0, index, reference.body_orientation_wxyz
        )
        self._pending_station_anchor_xy.index_copy_(
            0, index, reference.station_anchor_xy_m
        )
        # Any prior owner is outside its frozen window by the reveal check.
        self._reward_owner_valid.index_fill_(0, index, False)
        self._reward_owner_sha.index_fill_(0, index, 0)
        self._reward_deadline_tick.index_fill_(0, index, -1)
        self._reward_owner_played.index_fill_(0, index, False)
        self._deadline_ack_pending.index_fill_(0, index, False)

        staged_keys = list(self._host_keys)
        staged_played = list(self._host_played)
        staged_pending = list(self._host_pending_reference_sha)
        staged_ordinals = list(self._host_scheduled_ordinal)
        staged_deadlines = list(self._host_last_deadline)
        staged_consumed = list(self._host_deadline_consumed)
        staged_reveals = list(self._host_last_reveal)
        for env_id, key, digest, ordinal, reveal, deadline in zip(
            ids,
            keys,
            reference.reference_sha256,
            ordinals,
            reveals,
            deadlines,
        ):
            staged_keys[env_id] = key
            staged_played[env_id] = False
            staged_pending[env_id] = digest
            staged_ordinals[env_id] = ordinal
            staged_deadlines[env_id] = deadline
            staged_consumed[env_id] = False
            staged_reveals[env_id] = reveal
        self._host_keys = staged_keys
        self._host_played = staged_played
        self._host_pending_reference_sha = staged_pending
        self._host_scheduled_ordinal = staged_ordinals
        self._host_last_deadline = staged_deadlines
        self._host_deadline_consumed = staged_consumed
        self._host_last_reveal = staged_reveals
        self._advance_mutation_version()

    def mark_playback_started(
        self,
        env_ids: Sequence[int],
        *,
        task_keys: Sequence[object],
    ) -> None:
        """Bind the rare R04 playback-release event to the exact full key."""

        self._require_r07_business_mutation_allowed(label="playback start")
        ids = self._ids(env_ids, label="mark_playback_started.env_ids")
        if len(task_keys) != len(ids):
            raise ContinuousRecoveryDeviceError(
                "playback start requires one full key per env"
            )
        keys = tuple(
            _runtime.coerce_landing_outcome_shot_key(value) for value in task_keys
        )
        staged_played = list(self._host_played)
        for env_id, key in zip(ids, keys):
            if self._host_keys[env_id] != key:
                raise ContinuousRecoveryDeviceError(
                    "playback start key differs from committed full key"
                )
            if staged_played[env_id]:
                raise ContinuousRecoveryDeviceError(
                    "playback start was already recorded"
                )
            staged_played[env_id] = True
        self._ever_motion_active.index_fill_(0, self._index(ids), True)
        self._playback_receipted.index_fill_(0, self._index(ids), True)
        self._host_played = staged_played
        self._advance_mutation_version()

    def complete_suffix(
        self,
        env_ids: Sequence[int],
        *,
        task_keys: Sequence[object],
        completed_at_episode_ticks: Sequence[int],
        fault_injector: _FaultInjector = None,
    ) -> None:
        """Promote the exact played task's staged frame 0 to ready owner."""

        self._require_r07_business_mutation_allowed(label="suffix completion")
        ids = self._ids(env_ids, label="complete_suffix.env_ids")
        if len(task_keys) != len(ids) or len(completed_at_episode_ticks) != len(ids):
            raise ContinuousRecoveryDeviceError(
                "complete_suffix requires one full key and tick per env"
            )
        keys = tuple(
            _runtime.coerce_landing_outcome_shot_key(value) for value in task_keys
        )
        completion_ticks = tuple(
            _exact_int(value, label="completed_at_episode_tick")
            for value in completed_at_episode_ticks
        )
        for env_id, key, completion_tick in zip(ids, keys, completion_ticks):
            if self._host_keys[env_id] != key:
                raise ContinuousRecoveryDeviceError(
                    "suffix completion key differs from committed full key"
                )
            if self._host_pending_reference_sha[env_id] is None:
                raise ContinuousRecoveryDeviceError(
                    "suffix completion has no staged frame0 reference"
                )
            if not self._host_played[env_id]:
                raise ContinuousRecoveryDeviceError(
                    "suffix completion precedes playback start"
                )
            if not self._host_deadline_consumed[env_id]:
                raise ContinuousRecoveryDeviceError(
                    "suffix completion precedes deadline consumption"
                )
            deadline = self._host_last_deadline[env_id]
            if (
                deadline is None
                or completion_tick < deadline
                or completion_tick > deadline + RECOVERY_START_AGE_TICK
            ):
                raise ContinuousRecoveryDeviceError(
                    "suffix completion lies outside deadline..age10"
                )
        index = self._index(ids)
        key_digest = DeviceLandingOutcomeShotKey.from_host_keys(
            keys, device=self.device
        ).canonical_sha256
        selected_ever_played = self._ever_motion_active.index_select(0, index)
        selected_pending = self._pending_reference_valid.index_select(0, index)
        selected_key_match = torch.all(
            self._pending_reference_key_sha.index_select(0, index) == key_digest,
            dim=1,
        )
        selected_reward_valid = self._reward_owner_valid.index_select(0, index)
        selected_reward_played = self._reward_owner_played.index_select(0, index)
        selected_reward_match = torch.all(
            self._reward_owner_sha.index_select(0, index) == key_digest,
            dim=1,
        )
        selected_age = self._episode_tick.index_select(
            0, index
        ) - self._reward_deadline_tick.index_select(0, index)
        completion_tensor = torch.tensor(
            completion_ticks, dtype=torch.int64, device=self.device
        )
        tick_match = self._episode_tick.index_select(0, index) == completion_tensor
        timing_ok = (selected_age >= 0) & (
            selected_age <= RECOVERY_START_AGE_TICK
        )
        safe_rows = (
            selected_ever_played
            & selected_pending
            & selected_key_match
            & selected_reward_valid
            & selected_reward_played
            & selected_reward_match
            & timing_ok
            & tick_match
        )
        bad_rows = ~safe_rows
        full_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        full_mask.index_copy_(0, index, safe_rows)
        bad_mask = torch.zeros_like(full_mask)
        bad_mask.index_copy_(0, index, bad_rows)

        if fault_injector is not None:
            fault_injector("suffix_validated_before_publish")

        self._fault_bits.bitwise_or_(
            bad_mask.to(dtype=torch.int64) * FAULT_SUFFIX_INCOMPLETE
        )
        _masked_copy_(
            self._reference_owner_sha,
            self._pending_reference_key_sha,
            full_mask,
        )
        self._reference_owner_kind.copy_(
            torch.where(
                full_mask,
                torch.full_like(self._reference_owner_kind, OWNER_COMMITTED_TASK),
                self._reference_owner_kind,
            )
        )
        _masked_copy_(self._reference_sha, self._pending_reference_sha, full_mask)
        _masked_copy_(
            self._reference_root_position,
            self._pending_reference_root_position,
            full_mask,
        )
        _masked_copy_(
            self._reference_root_orientation,
            self._pending_reference_root_orientation,
            full_mask,
        )
        _masked_copy_(
            self._reference_joint_position,
            self._pending_reference_joint_position,
            full_mask,
        )
        _masked_copy_(
            self._reference_body_position,
            self._pending_reference_body_position,
            full_mask,
        )
        _masked_copy_(
            self._reference_body_orientation,
            self._pending_reference_body_orientation,
            full_mask,
        )
        _masked_copy_(
            self._station_anchor_xy, self._pending_station_anchor_xy, full_mask
        )
        self._suffix_complete.bitwise_or_(full_mask)
        self._pending_reference_valid.bitwise_and_(~full_mask)
        _masked_zero_(self._pending_reference_sha, full_mask)
        _masked_zero_(self._pending_reference_key_sha, full_mask)
        for tensor in (
            self._pending_reference_root_position,
            self._pending_reference_root_orientation,
            self._pending_reference_joint_position,
            self._pending_reference_body_position,
            self._pending_reference_body_orientation,
            self._pending_station_anchor_xy,
        ):
            _masked_zero_(tensor, full_mask)
        self._ready_authority.bitwise_and_(~full_mask)
        self._ready_instant.bitwise_and_(~full_mask)
        self._ready_streak.masked_fill_(full_mask, 0)
        self._first_ready_tick.masked_fill_(full_mask, -1)

        staged_ready_keys = list(self._host_ready_owner_keys)
        staged_refs = list(self._host_reference_sha)
        staged_pending = list(self._host_pending_reference_sha)
        for env_id, key in zip(ids, keys):
            staged_ready_keys[env_id] = key
            staged_refs[env_id] = staged_pending[env_id]
            staged_pending[env_id] = None
        self._host_ready_owner_keys = staged_ready_keys
        self._host_reference_sha = staged_refs
        self._host_pending_reference_sha = staged_pending
        self._advance_mutation_version()

    def _command_projection(
        self, value: DeviceContinuousRecoveryCommandProjection
    ) -> DeviceContinuousRecoveryCommandProjection:
        if not isinstance(value, DeviceContinuousRecoveryCommandProjection):
            raise ContinuousRecoveryDeviceError(
                "command projection must be DeviceContinuousRecoveryCommandProjection"
            )
        n = self.num_envs
        for name in (
            "source_step",
            "episode_tick",
            "phase_code",
            "scheduled_ordinal",
            "current_deadline_tick",
        ):
            self._tensor(
                getattr(value, name),
                label=f"command.{name}",
                shape=(n,),
                dtype=torch.int64,
            )
        for name in (
            "reference_active",
            "motion_active",
            "suffix_complete",
            "deadline_due",
        ):
            self._tensor(
                getattr(value, name),
                label=f"command.{name}",
                shape=(n,),
                dtype=torch.bool,
            )
        self._tensor(
            value.current_task_key_sha256,
            label="command.current_task_key_sha256",
            shape=(n, 32),
            dtype=torch.uint8,
        )
        return value

    def reconcile_command_projection(
        self, value: DeviceContinuousRecoveryCommandProjection
    ) -> None:
        """Consume only current R04 identity/phase facts on the device.

        A completed suffix must be promoted with :meth:`complete_suffix`
        before this projection reports it.  Readiness is neither read nor used
        to move a reveal/deadline.
        """

        self._require_r07_business_mutation_allowed(
            label="command reconciliation"
        )
        value = self._command_projection(value)
        active = self._sequence_active
        first_step = self._source_step < 0
        step_ok = (first_step & (value.source_step >= 0)) | (
            ~first_step & (value.source_step > self._source_step)
        )
        tick_ok = value.episode_tick == self._episode_tick + 1
        phase_ok = (value.phase_code >= PHASE_CODES[0]) & (
            value.phase_code <= PHASE_CODES[-1]
        )
        key_match = torch.all(
            value.current_task_key_sha256 == self._current_key_sha, dim=1
        )
        empty_key = torch.all(value.current_task_key_sha256 == 0, dim=1)
        binding_ok = torch.where(self._current_key_valid, key_match, empty_key)
        timing_ok = torch.where(
            self._current_key_valid,
            (value.scheduled_ordinal == self._scheduled_ordinal)
            & (value.current_deadline_tick == self._current_deadline_tick),
            (value.scheduled_ordinal == -1)
            & (value.current_deadline_tick == -1),
        )
        mutually_exclusive = ~(value.reference_active & value.motion_active)
        deadline_ok = ~value.deadline_due | (
            self._current_key_valid
            & key_match
            & (value.episode_tick == self._current_deadline_tick)
        )
        regression = active & (~step_ok | ~tick_ok | ~phase_ok)
        binding_fault = active & (
            ~binding_ok | ~timing_ok | ~mutually_exclusive | ~deadline_ok
        )
        suffix_fault = active & value.suffix_complete & value.motion_active
        self._fault_bits.bitwise_or_(
            regression.to(dtype=torch.int64) * FAULT_STEP_REGRESSION
        )
        self._fault_bits.bitwise_or_(
            binding_fault.to(dtype=torch.int64) * FAULT_COMMAND_BINDING
        )
        self._fault_bits.bitwise_or_(
            suffix_fault.to(dtype=torch.int64) * FAULT_SUFFIX_INCOMPLETE
        )

        safe = active & ~regression & ~binding_fault & ~suffix_fault
        newly_played = safe & value.motion_active
        missing_playback_receipt = newly_played & ~self._playback_receipted
        self._fault_bits.bitwise_or_(
            missing_playback_receipt.to(dtype=torch.int64)
            * FAULT_COMMAND_BINDING
        )
        self._ever_motion_active.bitwise_or_(newly_played)
        deadline = safe & value.deadline_due
        self._deadline_ack_pending.bitwise_or_(deadline)

        _masked_copy_(self._source_step, value.source_step, safe)
        _masked_copy_(self._episode_tick, value.episode_tick, safe)
        _masked_copy_(self._phase_code, value.phase_code, safe)
        _masked_copy_(self._reference_active, value.reference_active, safe)
        _masked_copy_(self._motion_active, value.motion_active, safe)
        self._ready_authority.bitwise_and_(self._fault_bits == 0)
        self._advance_mutation_version()

    def latch_deadline_consumed(
        self,
        env_ids: Sequence[int],
        *,
        task_keys: Sequence[object],
        deadline_ticks: Sequence[int],
        fault_injector: _FaultInjector = None,
    ) -> None:
        """Atomically bind one R04 deadline event to reward ownership.

        This rare receipt is deliberately separate from the all-env phase
        projection so the exact C05 host key remains checkpointable without a
        hot-path device-to-host extraction.
        """

        self._require_r07_business_mutation_allowed(label="deadline latch")
        ids = self._ids(env_ids, label="latch_deadline_consumed.env_ids")
        if len(task_keys) != len(ids) or len(deadline_ticks) != len(ids):
            raise ContinuousRecoveryDeviceError(
                "deadline latch requires one key and tick per env"
            )
        keys = tuple(
            _runtime.coerce_landing_outcome_shot_key(value) for value in task_keys
        )
        deadlines = tuple(
            _exact_int(value, label="deadline_tick") for value in deadline_ticks
        )
        for env_id, key, deadline in zip(ids, keys, deadlines):
            if self._host_keys[env_id] != key:
                raise ContinuousRecoveryDeviceError(
                    "deadline latch key differs from committed full key"
                )
            if self._host_last_deadline[env_id] != deadline:
                raise ContinuousRecoveryDeviceError(
                    "deadline latch tick differs from committed timing"
                )
            if self._host_deadline_consumed[env_id]:
                raise ContinuousRecoveryDeviceError(
                    "deadline was already consumed"
                )

        index = self._index(ids)
        key_digest = DeviceLandingOutcomeShotKey.from_host_keys(
            keys, device=self.device
        ).canonical_sha256
        deadline_tensor = torch.tensor(
            deadlines, dtype=torch.int64, device=self.device
        )
        key_match = torch.all(
            self._current_key_sha.index_select(0, index) == key_digest, dim=1
        )
        timing_match = (
            self._current_deadline_tick.index_select(0, index) == deadline_tensor
        ) & (self._episode_tick.index_select(0, index) == deadline_tensor)
        pending = self._deadline_ack_pending.index_select(0, index)
        safe_rows = key_match & timing_match & pending
        full_safe = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        full_safe.index_copy_(0, index, safe_rows)
        bad = torch.zeros_like(full_safe)
        bad.index_copy_(0, index, ~safe_rows)
        played = full_safe & self._ever_motion_active
        unplayed = full_safe & ~self._ever_motion_active

        if fault_injector is not None:
            fault_injector("deadline_validated_before_publish")

        self._fault_bits.bitwise_or_(
            bad.to(dtype=torch.int64) * FAULT_COMMAND_BINDING
        )
        _masked_copy_(self._reward_owner_sha, self._current_key_sha, full_safe)
        _masked_copy_(
            self._reward_deadline_tick, self._current_deadline_tick, full_safe
        )
        self._reward_owner_valid.bitwise_or_(full_safe)
        self._reward_owner_played.copy_(
            torch.where(full_safe, self._ever_motion_active, self._reward_owner_played)
        )
        _masked_copy_(self._window_owner_sha, self._current_key_sha, full_safe)
        self._window_owner_valid.bitwise_or_(full_safe)
        for tensor in (
            self._window_expected_count,
            self._window_eligible_count,
            self._window_payment_count,
        ):
            tensor.masked_fill_(full_safe, 0)
        for tensor in (
            self._window_first_expected_age,
            self._window_last_expected_age,
            self._window_last_paid_age,
        ):
            tensor.masked_fill_(full_safe, -1)
        self._deadline_ack_pending.bitwise_and_(~full_safe)
        # An unplayed task never owns a ready reference; discard its private
        # frame0 while retaining its full key as the reward owner.
        self._pending_reference_valid.bitwise_and_(~unplayed)
        _masked_zero_(self._pending_reference_sha, unplayed)
        _masked_zero_(self._pending_reference_key_sha, unplayed)
        for tensor in (
            self._pending_reference_root_position,
            self._pending_reference_root_orientation,
            self._pending_reference_joint_position,
            self._pending_reference_body_position,
            self._pending_reference_body_orientation,
            self._pending_station_anchor_xy,
        ):
            _masked_zero_(tensor, unplayed)
        self._played_deadline_total.add_(played.to(dtype=torch.int64).sum())
        self._unplayed_deadline_total.add_(unplayed.to(dtype=torch.int64).sum())
        self._ready_authority.bitwise_and_(self._fault_bits == 0)

        staged_consumed = list(self._host_deadline_consumed)
        staged_pending = list(self._host_pending_reference_sha)
        staged_window_owners = list(self._host_window_owner_keys)
        for env_id, key in zip(ids, keys):
            staged_consumed[env_id] = True
            staged_window_owners[env_id] = key
            if not self._host_played[env_id]:
                staged_pending[env_id] = None
        self._host_deadline_consumed = staged_consumed
        self._host_pending_reference_sha = staged_pending
        self._host_window_owner_keys = staged_window_owners
        self._advance_mutation_version()

    def _plant_facts(
        self, value: DeviceContinuousRecoveryPlantFacts
    ) -> DeviceContinuousRecoveryPlantFacts:
        if not isinstance(value, DeviceContinuousRecoveryPlantFacts):
            raise ContinuousRecoveryDeviceError(
                "plant facts must be DeviceContinuousRecoveryPlantFacts"
            )
        n, j, b, f = (
            self.num_envs,
            self.num_joints,
            self.num_bodies,
            self.num_feet,
        )
        shapes = {
            "root_position_m": (n, 3),
            "root_orientation_wxyz": (n, 4),
            "root_linear_velocity_mps": (n, 3),
            "root_angular_velocity_radps": (n, 3),
            "joint_position_rad": (n, j),
            "joint_velocity_radps": (n, j),
            "body_position_m": (n, b, 3),
            "body_orientation_wxyz": (n, b, 4),
            "body_linear_velocity_mps": (n, b, 3),
            "body_angular_velocity_radps": (n, b, 3),
            "station_xy_m": (n, 2),
            "foot_contact_signal": (n, f),
            "foot_slip_velocity_xy_mps": (n, f, 2),
        }
        for name, shape in shapes.items():
            self._tensor(
                getattr(value, name),
                label=f"plant.{name}",
                shape=shape,
                dtype=self.dtype,
            )
        for name in ("facts_valid", "hard_safety_ok"):
            self._tensor(
                getattr(value, name),
                label=f"plant.{name}",
                shape=(n,),
                dtype=torch.bool,
            )
        return value

    @staticmethod
    def _finite_rows(value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)

    @staticmethod
    def _safe(value: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def _reduce_flat(self, value: torch.Tensor, kind: str) -> torch.Tensor:
        rows = value.reshape(value.shape[0], -1).abs()
        if kind == "root_mean_square_v1":
            return torch.sqrt(torch.mean(rows.square(), dim=1))
        raise ContinuousRecoveryDeviceError(
            f"unsupported flat reduction {kind!r}"
        )

    def _reduce_l2(self, value: torch.Tensor, kind: str) -> torch.Tensor:
        norms = torch.linalg.vector_norm(value, dim=-1)
        flat = norms.reshape(norms.shape[0], -1)
        if kind == "l2_norm_v1" and flat.shape[1] == 1:
            return flat[:, 0]
        if kind == "root_mean_square_l2_v1":
            return torch.sqrt(torch.mean(flat.square(), dim=1))
        raise ContinuousRecoveryDeviceError(f"unsupported L2 reduction {kind!r}")

    def _reduce_angles(self, value: torch.Tensor, kind: str) -> torch.Tensor:
        flat = value.reshape(value.shape[0], -1)
        if kind == "quaternion_geodesic_rad_v1" and flat.shape[1] == 1:
            return flat[:, 0]
        if kind == "root_mean_square_geodesic_rad_v1":
            return torch.sqrt(torch.mean(flat.square(), dim=1))
        raise ContinuousRecoveryDeviceError(
            f"unsupported angle reduction {kind!r}"
        )

    def _quat_error(
        self, current: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_norm = torch.linalg.vector_norm(current, dim=-1)
        reference_norm = torch.linalg.vector_norm(reference, dim=-1)
        valid = (
            torch.isfinite(current).all(dim=-1)
            & torch.isfinite(reference).all(dim=-1)
            & (current_norm > 0)
            & (reference_norm > 0)
        )
        safe_current = self._safe(current) / current_norm.clamp_min(
            torch.finfo(self.dtype).tiny
        ).unsqueeze(-1)
        safe_reference = self._safe(reference) / reference_norm.clamp_min(
            torch.finfo(self.dtype).tiny
        ).unsqueeze(-1)
        dot = torch.sum(safe_current * safe_reference, dim=-1).abs().clamp(0.0, 1.0)
        error = 2.0 * torch.acos(dot)
        return torch.where(valid, error, torch.zeros_like(error)), valid

    def _component_errors(
        self, facts: DeviceContinuousRecoveryPlantFacts
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._component_errors_against(
            facts,
            root_position=self._reference_root_position,
            root_orientation=self._reference_root_orientation,
            joint_position=self._reference_joint_position,
            body_position=self._reference_body_position,
            body_orientation=self._reference_body_orientation,
            station_anchor=self._station_anchor_xy,
        )

    def _component_errors_against(
        self,
        facts: DeviceContinuousRecoveryPlantFacts,
        *,
        root_position: torch.Tensor,
        root_orientation: torch.Tensor,
        joint_position: torch.Tensor,
        body_position: torch.Tensor,
        body_orientation: torch.Tensor,
        station_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reductions = self.profile.component_reductions
        current_root_q = self._safe(facts.root_orientation_wxyz)
        reference_root_q = self._safe(root_orientation)
        root_angle, root_q_valid = self._quat_error(
            current_root_q, reference_root_q
        )
        current_body_q = self._safe(facts.body_orientation_wxyz)
        reference_body_q = self._safe(body_orientation)
        body_angles, body_q_valid = self._quat_error(
            current_body_q, reference_body_q
        )

        support = self._safe(facts.foot_contact_signal) >= float(
            self.profile.support_contact_threshold
        )
        support_count = support.to(dtype=torch.int64).sum(dim=1)
        slip_norms = torch.linalg.vector_norm(
            self._safe(facts.foot_slip_velocity_xy_mps), dim=-1
        )
        supported_slip = torch.where(support, slip_norms, torch.zeros_like(slip_norms))
        slip_kind = reductions["foot_slip_mps"]
        if slip_kind == "support_conditioned_max_l2_v1":
            slip_error = supported_slip.amax(dim=1)
        else:
            raise ContinuousRecoveryDeviceError(
                f"unsupported supported-foot reduction {slip_kind!r}"
            )
        support_deficit = (
            int(self.profile.minimum_supported_feet) - support_count
        ).clamp_min(0).to(dtype=self.dtype)

        values = {
            "root_position_m": self._reduce_l2(
                self._safe(facts.root_position_m)
                - self._safe(root_position),
                reductions["root_position_m"],
            ),
            "root_orientation_rad": self._reduce_angles(
                root_angle.unsqueeze(-1), reductions["root_orientation_rad"]
            ),
            "root_linear_velocity_mps": self._reduce_l2(
                self._safe(facts.root_linear_velocity_mps),
                reductions["root_linear_velocity_mps"],
            ),
            "root_angular_velocity_radps": self._reduce_l2(
                self._safe(facts.root_angular_velocity_radps),
                reductions["root_angular_velocity_radps"],
            ),
            "joint_position_rad": self._reduce_flat(
                self._safe(facts.joint_position_rad)
                - self._safe(joint_position),
                reductions["joint_position_rad"],
            ),
            "joint_velocity_radps": self._reduce_flat(
                self._safe(facts.joint_velocity_radps),
                reductions["joint_velocity_radps"],
            ),
            "body_position_m": self._reduce_l2(
                self._safe(facts.body_position_m)
                - self._safe(body_position),
                reductions["body_position_m"],
            ),
            "body_orientation_rad": self._reduce_angles(
                body_angles, reductions["body_orientation_rad"]
            ),
            "body_linear_velocity_mps": self._reduce_l2(
                self._safe(facts.body_linear_velocity_mps),
                reductions["body_linear_velocity_mps"],
            ),
            "body_angular_velocity_radps": self._reduce_l2(
                self._safe(facts.body_angular_velocity_radps),
                reductions["body_angular_velocity_radps"],
            ),
            "station_xy_m": self._reduce_l2(
                self._safe(facts.station_xy_m) - self._safe(station_anchor),
                reductions["station_xy_m"],
            ),
            "foot_slip_mps": slip_error,
            "foot_support_deficit": support_deficit,
        }
        errors = torch.stack([values[name] for name in self.component_names], dim=1)

        tensor_finite = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for name in (
            "root_position_m",
            "root_orientation_wxyz",
            "root_linear_velocity_mps",
            "root_angular_velocity_radps",
            "joint_position_rad",
            "joint_velocity_radps",
            "body_position_m",
            "body_orientation_wxyz",
            "body_linear_velocity_mps",
            "body_angular_velocity_radps",
            "station_xy_m",
            "foot_contact_signal",
            "foot_slip_velocity_xy_mps",
        ):
            tensor_finite.bitwise_and_(self._finite_rows(getattr(facts, name)))
        reference_finite = (
            self._finite_rows(root_position)
            & self._finite_rows(root_orientation)
            & self._finite_rows(joint_position)
            & self._finite_rows(body_position)
            & self._finite_rows(body_orientation)
            & self._finite_rows(station_anchor)
        )
        quat_valid = root_q_valid & body_q_valid.all(dim=1)
        finite = tensor_finite & reference_finite & quat_valid & torch.isfinite(errors).all(dim=1)
        return errors, finite, support_count

    def action_epoch_reward_view(
        self,
        facts: DeviceContinuousRecoveryPlantFacts,
        *,
        reference: object,
        epoch: object,
        current_source_step: torch.Tensor,
        adapter_source_step: object,
        motion_owner: object,
        action_epoch_owner: object,
    ) -> R07EpochDirectRewardFacts:
        """Compute a non-mutating R07 view from live facts and Motion frame 0.

        This is intentionally separate from :meth:`publish_after_physics` and
        its legacy payment ledger.  It changes no coordinator tensor.  Missing
        or stale producers become typed fault bits; finite low support, a fall,
        and large recovery errors remain ordinary finite learning values.
        """

        facts = self._plant_facts(facts)
        step = self._tensor(
            current_source_step,
            label="R07 epoch current_source_step",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        (
            epoch_owner_type,
            epoch_record_type,
            lifecycle_type,
            epoch_globals,
        ) = (
            _exact_action_epoch_types(action_epoch_owner)
        )
        bound_bundle = self._diagnostic_n2_bundle
        parent_identity_valid = (
            type(bound_bundle) is DiagnosticN2ContinuousRecoveryBundle
            and bound_bundle.owner is self
            and bound_bundle.motion_owner is motion_owner
            and bound_bundle.action_epoch_owner is action_epoch_owner
        )
        if not parent_identity_valid:
            raise ContinuousRecoveryDeviceError(
                "R07 construction-bound parent/Motion/epoch identity differs"
            )
        motion_parent_identity = (
            bound_bundle._project_parent_action_identity()
        )
        epoch_version = getattr(epoch, "version", None)
        reference_epoch_version = getattr(reference, "epoch_version", None)
        commit_head = getattr(action_epoch_owner, "commit_head", None)
        if (
            type(epoch_version) is not int
            or type(reference_epoch_version) is not int
            or type(commit_head) is not int
            or epoch_version != reference_epoch_version
            or epoch_version != commit_head - 1
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch snapshot/reference is stale or foreign"
            )
        required_epoch = (
            type(epoch) is epoch_record_type
            and type(action_epoch_owner) is epoch_owner_type
            and action_epoch_owner.num_envs == self.num_envs
            and action_epoch_owner.device == self.device
            and getattr(reference, "epoch_owner", None) is action_epoch_owner
            and getattr(reference, "motion_owner", None) is motion_owner
        )
        reference_fields = (
            ("root_position_m", (self.num_envs, 3)),
            ("root_orientation_wxyz", (self.num_envs, 4)),
            ("joint_position_rad", (self.num_envs, self.num_joints)),
            ("body_position_m", (self.num_envs, self.num_bodies, 3)),
            ("body_orientation_wxyz", (self.num_envs, self.num_bodies, 4)),
            ("station_anchor_xy_m", (self.num_envs, 2)),
        )
        reference_shape_ok = required_epoch
        for name, shape in reference_fields:
            value = getattr(reference, name, None)
            reference_shape_ok = reference_shape_ok and (
                isinstance(value, torch.Tensor)
                and tuple(value.shape) == shape
                and value.device == self.device
                and value.dtype == self.dtype
            )
        validity = getattr(reference, "validity", None)
        reference_faults = getattr(reference, "producer_fault_bits", None)
        reference_kind = getattr(reference, "reference_kind", None)
        reference_action_slot = getattr(
            reference, "reference_action_slot", None
        )
        reference_action_uid = getattr(
            reference, "reference_action_uid", None
        )
        motion_cadence_tick = getattr(reference, "cadence_tick", None)
        try:
            reference_shot_key = _row_identity.require_action_epoch_shot_key(
                getattr(reference, "shot_key", None),
                shape=(self.num_envs,),
                device=self.device,
                label="R07 Motion reference shot_key",
            ).clone()
        except _row_identity.ActionEpochShotKeyError as exc:
            raise ContinuousRecoveryDeviceError(str(exc)) from exc
        reference_shape_ok = reference_shape_ok and (
            isinstance(validity, torch.Tensor)
            and tuple(validity.shape) == (self.num_envs,)
            and validity.device == self.device
            and validity.dtype == torch.bool
            and isinstance(reference_faults, torch.Tensor)
            and tuple(reference_faults.shape) == (self.num_envs,)
            and reference_faults.device == self.device
            and reference_faults.dtype == torch.int64
            and isinstance(reference_kind, torch.Tensor)
            and tuple(reference_kind.shape) == (self.num_envs,)
            and reference_kind.device == self.device
            and reference_kind.dtype == torch.int64
            and isinstance(reference_action_slot, torch.Tensor)
            and tuple(reference_action_slot.shape) == (self.num_envs,)
            and reference_action_slot.device == self.device
            and reference_action_slot.dtype == torch.int64
            and isinstance(reference_action_uid, torch.Tensor)
            and tuple(reference_action_uid.shape) == (self.num_envs,)
            and reference_action_uid.device == self.device
            and reference_action_uid.dtype == torch.int64
            and isinstance(motion_cadence_tick, torch.Tensor)
            and tuple(motion_cadence_tick.shape) == (self.num_envs,)
            and motion_cadence_tick.device == self.device
            and motion_cadence_tick.dtype == torch.int64
        )
        if not reference_shape_ok:
            raise ContinuousRecoveryDeviceError(
                "R07 Motion frame-0 reference ABI/identity differs"
            )
        reset_generation = self._tensor(
            getattr(epoch, "reset_generation", None),
            label="R07 epoch reset_generation",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        _generation_changed, motion_chronology_valid = (
            self._action_epoch_motion_cadence_chronology(
                reset_generation=reset_generation,
                motion_cadence_tick=motion_cadence_tick,
                allow_same_generation_hold=False,
            )
        )

        parent_action_slot = getattr(
            motion_parent_identity, "action_slot", None
        )
        parent_action_uid = getattr(
            motion_parent_identity, "action_uid", None
        )
        parent_action_uids = getattr(
            motion_parent_identity, "action_uids", None
        )
        if (
            not parent_identity_valid
            or type(parent_action_uids) is not tuple
            or not parent_action_uids
            or any(
                type(uid) is not int or uid <= 0
                for uid in parent_action_uids
            )
            or len(set(parent_action_uids)) != len(parent_action_uids)
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 Motion parent action authority differs"
            )
        action_count = len(parent_action_uids)
        if (
            type(parent_action_slot) is not int
            or parent_action_slot < 0
            or parent_action_slot >= action_count
            or type(parent_action_uid) is not int
            or parent_action_uid <= 0
            or parent_action_uids[parent_action_slot]
            != parent_action_uid
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 Motion parent schedule identity differs"
            )
        motion_clip_id = self._tensor(
            getattr(motion_owner, "clip_id", None),
            label="R07 Motion current action slot",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        code_uid_table = torch.as_tensor(
            parent_action_uids,
            dtype=torch.int64,
            device=self.device,
        )
        clip_slot_valid = motion_clip_id.ge(0) & motion_clip_id.lt(action_count)
        safe_motion_clip_id = torch.clamp(
            motion_clip_id, min=0, max=action_count - 1
        )
        parent_slot_row = torch.full_like(
            motion_clip_id, parent_action_slot
        )
        parent_uid_row = torch.full_like(motion_clip_id, parent_action_uid)
        parent_current_identity_valid = (
            clip_slot_valid
            & motion_clip_id.eq(parent_slot_row)
            & code_uid_table[safe_motion_clip_id].eq(parent_uid_row)
        )

        reference_values = {
            name: getattr(reference, name).detach().clone()
            for name, _shape in reference_fields
        }
        all_reference_finite = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for value in reference_values.values():
            all_reference_finite &= self._finite_rows(value)
        env_ids = torch.arange(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        shot_slot_capacity = action_epoch_owner.shot_slot_capacity
        current_slots = self._tensor(
            getattr(epoch, "current_task_slot"),
            label="R07 epoch current_task_slot",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        slot_valid = current_slots.ge(0) & current_slots.lt(
            shot_slot_capacity
        )
        safe_slots = torch.clamp(
            current_slots, min=0, max=shot_slot_capacity - 1
        )
        current_phase = self._tensor(
            getattr(epoch, "phase"),
            label="R07 epoch current phase",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.int64,
        )[env_ids, safe_slots]
        outcome_phase = epoch_globals.get("PHASE_OUTCOME_SETTLED")
        if type(outcome_phase) is not int:
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch outcome phase constant differs"
            )
        lifecycle = epoch.recovery_reference_lifecycle_masks()
        if type(lifecycle) is not lifecycle_type:
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch reference lifecycle type differs"
            )
        lifecycle_upcoming = self._tensor(
            lifecycle.upcoming,
            label="R07 epoch lifecycle upcoming",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.bool,
        )
        lifecycle_completed = self._tensor(
            lifecycle.completed,
            label="R07 epoch lifecycle completed",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.bool,
        )
        bootstrap_lifecycle = (
            slot_valid & lifecycle_upcoming[env_ids, safe_slots]
        )
        completed_lifecycle = (
            slot_valid & lifecycle_completed[env_ids, safe_slots]
        )
        selected_rows = self._tensor(
            getattr(epoch, "selected_mask"),
            label="R07 epoch selected_mask",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.bool,
        )[env_ids, safe_slots]
        epoch_action_slots = self._tensor(
            getattr(getattr(epoch, "identity"), "action_slot"),
            label="R07 epoch action_slot",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.int64,
        )[env_ids, safe_slots]
        epoch_action_uids = self._tensor(
            getattr(getattr(epoch, "identity"), "action_uid"),
            label="R07 epoch action_uid",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.int64,
        )[env_ids, safe_slots]
        current_shot_key = _row_identity.ActionEpochShotKey(
            **{
                field.name: getattr(epoch.identity.shot_key, field.name)[
                    env_ids, safe_slots
                ].detach().clone()
                for field in fields(_row_identity.ActionEpochShotKey)
            }
        )
        empty_shot_key = _row_identity.empty_action_epoch_shot_key(
            (self.num_envs,), device=self.device
        )
        current_key_valid = _row_identity.action_epoch_shot_key_valid(
            current_shot_key
        )
        reference_key_valid = _row_identity.action_epoch_shot_key_valid(
            reference_shot_key
        )
        current_key_empty = _row_identity.action_epoch_shot_key_equal(
            current_shot_key, empty_shot_key
        )
        reference_key_empty = _row_identity.action_epoch_shot_key_equal(
            reference_shot_key, empty_shot_key
        )
        reference_key_matches_current = _row_identity.action_epoch_shot_key_equal(
            reference_shot_key, current_shot_key
        )
        reference_matches_parent = (
            reference_action_slot.eq(parent_slot_row)
            & reference_action_uid.eq(parent_uid_row)
            & parent_current_identity_valid
        )
        epoch_slot_valid = epoch_action_slots.ge(0) & epoch_action_slots.lt(
            action_count
        )
        safe_epoch_action_slots = torch.clamp(
            epoch_action_slots, min=0, max=action_count - 1
        )
        bootstrap_identity_valid = (
            bootstrap_lifecycle
            & ~selected_rows
            & current_key_empty
            & reference_key_empty
            & reference_matches_parent
        )
        completed_identity_valid = (
            completed_lifecycle
            & selected_rows
            & current_key_valid
            & reference_key_valid
            & reference_key_matches_current
            & epoch_slot_valid
            & motion_clip_id.eq(epoch_action_slots)
            & epoch_action_uids.eq(code_uid_table[safe_epoch_action_slots])
            & reference_action_slot.eq(epoch_action_slots)
            & reference_action_uid.eq(epoch_action_uids)
        )
        independent_identity_valid = (
            bootstrap_identity_valid
            | completed_identity_valid
        )
        lifecycle_valid = bootstrap_lifecycle | completed_lifecycle
        expected_reference_kind = torch.where(
            bootstrap_lifecycle,
            torch.full_like(
                reference_kind,
                R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0,
            ),
            torch.where(
                completed_lifecycle,
                torch.full_like(
                    reference_kind,
                    R07_REFERENCE_COMPLETED_ACTION_FRAME0,
                ),
                torch.zeros_like(reference_kind),
            ),
        )
        reference_kind = reference_kind.detach().clone()
        reference_action_slot = reference_action_slot.detach().clone()
        reference_action_uid = reference_action_uid.detach().clone()
        reference_valid = (
            validity.detach().clone()
            & all_reference_finite
            & lifecycle_valid
            & independent_identity_valid
            & reference_kind.eq(expected_reference_kind)
            & reference_action_slot.ge(0)
            & reference_action_uid.gt(0)
        )

        errors, arithmetic_valid, support_count = self._component_errors_against(
            facts,
            root_position=reference_values["root_position_m"],
            root_orientation=reference_values["root_orientation_wxyz"],
            joint_position=reference_values["joint_position_rad"],
            body_position=reference_values["body_position_m"],
            body_orientation=reference_values["body_orientation_wxyz"],
            station_anchor=reference_values["station_anchor_xy_m"],
        )

        adapter_step_valid = (
            type(adapter_source_step) is int
            and adapter_source_step >= 0
        )
        adapter_step = torch.full_like(
            step, adapter_source_step if adapter_step_valid else -1
        )
        chronology_valid = step.eq(adapter_step) & step.ge(0)
        deadline_tick = self._tensor(
            getattr(getattr(epoch, "clocks"), "deadline_tick"),
            label="R07 epoch deadline_tick",
            shape=(self.num_envs, shot_slot_capacity),
            dtype=torch.int64,
        )[env_ids, safe_slots]
        recovery_age_valid = (
            completed_lifecycle
            & current_key_valid
            & deadline_tick.ge(0)
            & motion_cadence_tick.ge(deadline_tick)
        )
        recovery_age_tick = torch.where(
            recovery_age_valid,
            motion_cadence_tick - deadline_tick,
            torch.full_like(step, -1),
        )
        plant_valid = facts.facts_valid & arithmetic_valid
        producer_fault_bits = reference_faults.detach().clone()
        producer_fault_bits |= (
            (~chronology_valid).to(torch.int64)
            * R07_EPOCH_FAULT_STALE_SOURCE_STEP
        )
        producer_fault_bits |= (
            (~plant_valid).to(torch.int64)
            * R07_EPOCH_FAULT_INVALID_PLANT_FACT
        )
        producer_fault_bits |= (
            (~reference_valid).to(torch.int64)
            * R07_EPOCH_FAULT_INVALID_REFERENCE
        )
        if not required_epoch:
            producer_fault_bits |= torch.full_like(
                producer_fault_bits, R07_EPOCH_FAULT_EPOCH_IDENTITY
            )
        # This view remains pure: the exact publication writer below owns the
        # sticky R07 fault.  Still make the invalid independent-owner join
        # unusable here so reward/ready values cannot escape before that write.
        r07_row_safe = self._fault_bits.eq(0) & motion_chronology_valid
        facts_valid = (
            plant_valid & reference_valid & chronology_valid & r07_row_safe
        )
        infrastructure_fault = producer_fault_bits.ne(0) | ~r07_row_safe
        scores = torch.reciprocal(
            1.0 + torch.square(errors / self._scales.unsqueeze(0))
        )
        raw_score = torch.sum(scores * self._weights.unsqueeze(0), dim=1) / self._weight_sum
        # Plant errors and readiness remain dense throughout the control-step
        # cycle.  Payment is narrower: R07 is post-shot recovery, never a
        # frame-zero pull during reveal/swing and never retained-row income.
        reward_window = _action_epoch_recovery_reward_window(
            current_phase,
            recovery_age_tick,
            outcome_settled_phase=outcome_phase,
        )
        reward_eligible = facts_valid & reward_window
        weighted_reward = torch.where(
            reward_eligible,
            raw_score * float(self.profile.reward_weight),
            torch.zeros_like(raw_score),
        )
        support_ok = support_count >= int(self.profile.minimum_supported_feet)
        ready_instant = (
            facts_valid
            & facts.hard_safety_ok
            & support_ok
            & torch.all(errors <= self._ready_tolerances.unsqueeze(0), dim=1)
        )
        return R07EpochDirectRewardFacts(
            source_step=torch.where(
                facts_valid, step, torch.full_like(step, -1)
            ),
            motion_cadence_tick=motion_cadence_tick.detach().clone(),
            reset_generation=reset_generation.detach().clone(),
            recovery_age_tick=recovery_age_tick,
            reward_eligible=reward_eligible,
            facts_valid=facts_valid,
            foot_supported_lr=(
                facts.facts_valid[:, None]
                & (
                    facts.foot_contact_signal
                    >= float(self.profile.support_contact_threshold)
                )
            ),
            infrastructure_fault=infrastructure_fault,
            producer_fault_bits=producer_fault_bits,
            component_errors=torch.where(
                facts_valid[:, None], errors, torch.zeros_like(errors)
            ),
            raw_score=torch.where(
                facts_valid, raw_score, torch.zeros_like(raw_score)
            ),
            weighted_reward=weighted_reward,
            ready_instant=ready_instant,
            reference_kind=reference_kind,
            reference_action_slot=reference_action_slot,
            reference_action_uid=reference_action_uid,
        )

    def stamp_action_epoch_idle_observation(
        self,
        *,
        epoch_facts: object,
        current_source_step: int,
        observed_source_step: int,
        motion_cadence_tick: torch.Tensor,
        action_epoch_owner: object,
    ) -> None:
        """Retain only neutral chronology needed by the current critic view."""

        if (
            isinstance(current_source_step, bool)
            or not isinstance(current_source_step, int)
            or current_source_step < 0
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 idle-observation current_source_step is malformed"
            )
        motion_tick = self._tensor(
            motion_cadence_tick,
            label="R07 idle-observation Motion cadence tick",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        epoch_owner_type, _record_type, _lifecycle_type, epoch_globals = (
            _exact_action_epoch_types(action_epoch_owner)
        )
        epoch_facts_type = epoch_globals.get(
            "ActionEpochIdleObservationChronology"
        )
        bound_bundle = self._diagnostic_n2_bundle
        if (
            type(bound_bundle) is not DiagnosticN2ContinuousRecoveryBundle
            or bound_bundle.owner is not self
            or bound_bundle.action_epoch_owner is not action_epoch_owner
            or type(action_epoch_owner) is not epoch_owner_type
            or type(epoch_facts_type) is not type
            or type(epoch_facts) is not epoch_facts_type
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 idle-observation owner or fact type differs"
            )

        reset_generation = self._tensor(
            getattr(epoch_facts, "reset_generation", None),
            label="R07 idle-observation reset_generation",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        epoch_version = getattr(epoch_facts, "epoch_version", None)
        if (
            type(epoch_version) is not int
            or epoch_version != action_epoch_owner.commit_head - 1
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 idle-observation ActionEpoch snapshot is stale or foreign"
            )
        if (
            isinstance(observed_source_step, bool)
            or not isinstance(observed_source_step, int)
            or observed_source_step < 0
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 idle-observation source step is malformed"
            )
        if observed_source_step != current_source_step:
            raise ContinuousRecoveryDeviceError(
                "R07 idle-observation source step differs from runtime"
            )

        self._require_action_epoch_readiness_chronology(
            observed_source_step=observed_source_step,
        )
        self._require_r07_business_mutation_allowed(
            label="ActionEpoch idle post-physics support"
        )
        _generation_changed, chronology_valid = (
            self._action_epoch_motion_cadence_chronology(
                reset_generation=reset_generation,
                motion_cadence_tick=motion_tick,
                allow_same_generation_hold=True,
            )
        )
        writable_rows = self._latch_r07_row_fault(
            ~chronology_valid,
            reason_bit=FAULT_ACTION_EPOCH_MOTION_CHRONOLOGY,
        )
        writable_rows = self._latch_r07_row_fault(
            writable_rows
            & motion_tick.ge(torch.iinfo(torch.int64).max),
            reason_bit=FAULT_MOTION_CADENCE_SUCCESSOR_OVERFLOW,
        )
        self._action_epoch_ready_last_motion_cadence_tick.copy_(
            torch.where(
                writable_rows,
                motion_tick,
                self._action_epoch_ready_last_motion_cadence_tick,
            )
        )
        self._action_epoch_ready_last_reset_generation.copy_(
            torch.where(
                writable_rows,
                reset_generation,
                self._action_epoch_ready_last_reset_generation,
            )
        )
        self._action_epoch_ready_last_source_step = observed_source_step
        # A neutral/N/A tick is a real gap in R07's consecutive-readiness
        # chronology.  Preserve the keyed reference and first-ready history,
        # but never let a later full recovery sample inherit dwell accumulated
        # before this gap.
        self._action_epoch_ready_instant.bitwise_and_(~writable_rows)
        self._action_epoch_ready_live.bitwise_and_(~writable_rows)
        self._action_epoch_ready_streak.copy_(
            torch.where(
                writable_rows,
                torch.zeros_like(self._action_epoch_ready_streak),
                self._action_epoch_ready_streak,
            )
        )
        self._idle_observation_source_step = observed_source_step
        self._idle_observation_reset_generation.copy_(
            torch.where(
                writable_rows,
                reset_generation,
                self._idle_observation_reset_generation,
            )
        )
        self._idle_observation_motion_cadence_tick.copy_(
            torch.where(
                writable_rows,
                motion_tick,
                self._idle_observation_motion_cadence_tick,
            )
        )
        self._idle_observation_published = True
        self._latest_motion_ready_projection = None
        self._advance_mutation_version()
        return None

    def _action_epoch_motion_cadence_chronology(
        self,
        *,
        reset_generation: torch.Tensor,
        motion_cadence_tick: torch.Tensor,
        allow_same_generation_hold: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return generation-change and validity for the independent join.

        The no-key idle path samples the global source clock even when Motion's
        row-local sequence is inactive, so its cadence may legally hold for one
        or more source ticks.  A keyed recovery sample is stricter: Motion is
        active and must publish the exact next cadence tick.  Keep that semantic
        choice explicit at each callsite instead of inferring one clock from the
        other.
        """

        if type(allow_same_generation_hold) is not bool:
            raise ContinuousRecoveryDeviceError(
                "R07 cadence hold mode differs"
            )

        reset_generation = self._tensor(
            reset_generation,
            label="R07 readiness reset_generation",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        motion_cadence_tick = self._tensor(
            motion_cadence_tick,
            label="R07 readiness Motion cadence_tick",
            shape=(self.num_envs,),
            dtype=torch.int64,
        )
        prior_generation = self._action_epoch_ready_last_reset_generation
        prior_tick = self._action_epoch_ready_last_motion_cadence_tick
        has_history = prior_generation.ge(0)
        same_generation = reset_generation.eq(prior_generation)
        generation_advanced = (
            prior_generation < torch.iinfo(torch.int64).max
        ) & reset_generation.eq(prior_generation + 1)
        tick_advanced = (
            prior_tick < torch.iinfo(torch.int64).max
        ) & motion_cadence_tick.eq(prior_tick + 1)
        tick_held = motion_cadence_tick.eq(prior_tick)
        same_generation_tick_valid = tick_advanced
        if allow_same_generation_hold:
            same_generation_tick_valid = tick_held | tick_advanced
        # ``-1`` is owner-private initialization state, never a Motion fact
        # that R07 may publish or turn into a next-tick capability.
        chronology_valid = motion_cadence_tick.ge(0) & (
            ((~has_history) & reset_generation.ge(0))
            | (
                has_history
                & (
                    (same_generation & same_generation_tick_valid)
                    | generation_advanced
                )
            )
        )
        return has_history & generation_advanced, chronology_valid

    def _latch_r07_row_fault(
        self,
        fault_rows: torch.Tensor,
        *,
        reason_bit: int,
    ) -> torch.Tensor:
        """Latch one named device fault and return the global writable rows."""

        rows = self._tensor(
            fault_rows,
            label="R07 runtime fault rows",
            shape=(self.num_envs,),
            dtype=torch.bool,
        )
        known_bits = frozenset(bit for _name, bit in FAULTS)
        if (
            type(reason_bit) is not int
            or reason_bit <= 0
            or reason_bit & (reason_bit - 1)
            or reason_bit not in known_bits
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 runtime fault reason bit is unknown or compound"
            )
        # The first root fault owns the row's causal diagnosis.  Later checks
        # see the returned sticky safe mask and cannot manufacture compounds.
        root_rows = rows & self._fault_bits.eq(0)
        self._fault_bits.bitwise_or_(
            root_rows.to(dtype=torch.int64) * int(reason_bit)
        )
        return self._fault_bits.eq(0)

    def _require_action_epoch_readiness_chronology(
        self,
        *,
        observed_source_step: object,
    ) -> None:
        """Reject a skipped/replayed live step before either owner mutates."""

        if (
            isinstance(observed_source_step, bool)
            or not isinstance(observed_source_step, int)
            or observed_source_step < 0
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch readiness chronology is malformed"
            )
        prior = self._action_epoch_ready_last_source_step
        if prior >= 0 and observed_source_step != prior + 1:
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch readiness source step is skipped, stale, or replayed"
            )

    def _next_ready_dwell(
        self,
        ready_instant: torch.Tensor,
        prior_streak: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the single owner-owned consecutive-readiness definition."""

        next_streak = torch.where(
            ready_instant,
            prior_streak + 1,
            torch.zeros_like(prior_streak),
        )
        ready_live = ready_instant & (
            next_streak >= int(self.profile.ready_dwell_ticks)
        )
        return next_streak, ready_live

    def _publish_action_epoch_motion_readiness(
        self,
        result: R07EpochDirectRewardFacts,
        *,
        observed_source_step: object,
        shot_key: _row_identity.ActionEpochShotKey | None,
        publish_keyed_first_ready: bool = True,
    ) -> None:
        """Commit lean dwell state and mint only the next-tick Motion view."""

        if type(publish_keyed_first_ready) is not bool:
            raise ContinuousRecoveryDeviceError(
                "R07 keyed first-ready publication mode differs"
            )
        self._require_r07_business_mutation_allowed(
            label="ActionEpoch post-physics readiness"
        )
        self._require_action_epoch_readiness_chronology(
            observed_source_step=observed_source_step,
        )
        key = None
        if shot_key is None:
            if publish_keyed_first_ready:
                raise ContinuousRecoveryDeviceError(
                    "R07 keyed readiness requires a shot key"
                )
        else:
            try:
                key = _row_identity.require_action_epoch_shot_key(
                    shot_key,
                    shape=(self.num_envs,),
                    device=self.device,
                    label="R07 readiness shot_key",
                ).clone()
            except _row_identity.ActionEpochShotKeyError as exc:
                raise ContinuousRecoveryDeviceError(str(exc)) from exc
        if type(result) is not R07EpochDirectRewardFacts:
            raise ContinuousRecoveryDeviceError(
                "R07 ActionEpoch readiness facts have foreign type"
            )
        for name, dtype in (
            ("source_step", torch.int64),
            ("motion_cadence_tick", torch.int64),
            ("reset_generation", torch.int64),
            ("facts_valid", torch.bool),
            ("infrastructure_fault", torch.bool),
            ("producer_fault_bits", torch.int64),
            ("ready_instant", torch.bool),
            ("reference_kind", torch.int64),
            ("reference_action_slot", torch.int64),
            ("reference_action_uid", torch.int64),
        ):
            value = getattr(result, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (self.num_envs,)
                or value.dtype != dtype
                or value.device != self.device
            ):
                raise ContinuousRecoveryDeviceError(
                    f"R07 ActionEpoch readiness {name} shape differs"
                )
        foot_supported_lr = self._tensor(
            result.foot_supported_lr,
            label="R07 readiness foot_supported_lr",
            shape=(self.num_envs, self.num_feet),
            dtype=torch.bool,
        )
        generation_changed, chronology_valid = (
            self._action_epoch_motion_cadence_chronology(
                reset_generation=result.reset_generation,
                motion_cadence_tick=result.motion_cadence_tick,
                allow_same_generation_hold=False,
            )
        )
        writable_rows = self._latch_r07_row_fault(
            ~chronology_valid,
            reason_bit=FAULT_ACTION_EPOCH_MOTION_CHRONOLOGY,
        )
        writable_rows = self._latch_r07_row_fault(
            writable_rows
            & result.motion_cadence_tick.ge(torch.iinfo(torch.int64).max),
            reason_bit=FAULT_MOTION_CADENCE_SUCCESSOR_OVERFLOW,
        )

        if key is None:
            key_changed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            for field in fields(_row_identity.ActionEpochShotKey):
                key_changed |= getattr(
                    self._action_epoch_ready_shot_key, field.name
                ).ne(-1)
        else:
            key_changed = ~_row_identity.action_epoch_shot_key_equal(
                key, self._action_epoch_ready_shot_key
            )
        reference_identity_changed = (
            result.reference_kind.ne(self._action_epoch_ready_reference_kind)
            | result.reference_action_slot.ne(
                self._action_epoch_ready_reference_action_slot
            )
            | result.reference_action_uid.ne(
                self._action_epoch_ready_reference_action_uid
            )
            | key_changed
        )
        reference_identity_changed |= generation_changed
        reference_identity_changed &= writable_rows
        prior_streak = torch.where(
            reference_identity_changed,
            torch.zeros_like(self._action_epoch_ready_streak),
            self._action_epoch_ready_streak,
        )
        first_ready_source_step = torch.where(
            reference_identity_changed,
            torch.full_like(self._action_epoch_first_ready_source_step, -1),
            self._action_epoch_first_ready_source_step,
        )
        ready_instant = (
            result.ready_instant
            & result.facts_valid
            & ~result.infrastructure_fault
            & result.producer_fault_bits.eq(0)
            & result.source_step.eq(int(observed_source_step))
            & writable_rows
        )
        next_streak, ready_live = self._next_ready_dwell(
            ready_instant,
            prior_streak,
        )
        first_ready = ready_live & first_ready_source_step.lt(0)

        self._action_epoch_ready_instant.copy_(
            torch.where(
                writable_rows,
                ready_instant,
                self._action_epoch_ready_instant,
            )
        )
        self._action_epoch_ready_live.copy_(
            torch.where(
                writable_rows,
                ready_live,
                self._action_epoch_ready_live,
            )
        )
        self._action_epoch_ready_streak.copy_(
            torch.where(
                writable_rows,
                next_streak,
                self._action_epoch_ready_streak,
            )
        )
        self._action_epoch_ready_reference_kind.copy_(
            torch.where(
                writable_rows,
                result.reference_kind,
                self._action_epoch_ready_reference_kind,
            )
        )
        self._action_epoch_ready_reference_action_slot.copy_(
            torch.where(
                writable_rows,
                result.reference_action_slot,
                self._action_epoch_ready_reference_action_slot,
            )
        )
        self._action_epoch_ready_reference_action_uid.copy_(
            torch.where(
                writable_rows,
                result.reference_action_uid,
                self._action_epoch_ready_reference_action_uid,
            )
        )
        for field in fields(_row_identity.ActionEpochShotKey):
            destination = getattr(self._action_epoch_ready_shot_key, field.name)
            source = (
                torch.full_like(destination, -1)
                if key is None
                else getattr(key, field.name)
            )
            destination.copy_(
                torch.where(writable_rows, source, destination)
            )
        self._action_epoch_first_ready_source_step.copy_(
            torch.where(
                writable_rows,
                torch.where(
                    first_ready,
                    torch.full_like(
                        first_ready_source_step,
                        int(observed_source_step),
                    ),
                    first_ready_source_step,
                ),
                self._action_epoch_first_ready_source_step,
            )
        )
        self._action_epoch_ready_last_motion_cadence_tick.copy_(
            torch.where(
                writable_rows,
                result.motion_cadence_tick,
                self._action_epoch_ready_last_motion_cadence_tick,
            )
        )
        self._action_epoch_ready_last_reset_generation.copy_(
            torch.where(
                writable_rows,
                result.reset_generation,
                self._action_epoch_ready_last_reset_generation,
            )
        )
        self._action_epoch_ready_last_source_step = int(observed_source_step)
        self._ready_instant_total.add_(ready_instant.to(torch.int64).sum())
        self._first_ready_total.add_(first_ready.to(torch.int64).sum())
        bundle = self._diagnostic_n2_bundle
        # Bootstrap readiness is a Motion admission fact for the upcoming
        # action, not an R07 fact owned by a current full ActionEpoch shot key.
        # Keep the owner-private dwell/first-ready chronology above, but only
        # publish keyed telemetry for a completed action reference.
        epoch_first_ready = (
            first_ready
            & writable_rows
            & result.reference_kind.eq(
                R07_REFERENCE_COMPLETED_ACTION_FRAME0
            )
        )
        if publish_keyed_first_ready:
            bundle.action_epoch_owner.publish_r07_first_ready(
                owner=bundle,
                first_ready=epoch_first_ready,
                shot_key=key,
                source_step=self._action_epoch_first_ready_source_step,
            )

        neutral_i64 = torch.full_like(result.source_step, -1)
        control_cadence = torch.where(
            writable_rows,
            result.motion_cadence_tick,
            torch.zeros_like(result.motion_cadence_tick),
        )
        motion_ready_payload = _MotionReadyPayload(
            owner_identity=self._identity,
            owner=self,
            postphysics_valid=(
                result.facts_valid & writable_rows
            ).detach().clone(),
            source_step=torch.where(
                writable_rows, result.source_step, neutral_i64
            ).detach().clone(),
            reset_generation=torch.where(
                writable_rows, result.reset_generation, neutral_i64
            ).detach().clone(),
            control_tick=torch.where(
                writable_rows,
                control_cadence + 1,
                neutral_i64,
            ),
            ready=(ready_live & writable_rows).detach().clone(),
            ready_streak=torch.where(
                writable_rows,
                next_streak,
                torch.zeros_like(next_streak),
            ).detach().clone(),
            required_dwell=int(self.profile.ready_dwell_ticks),
            foot_supported_lr=(
                foot_supported_lr & writable_rows[:, None]
            ).detach().clone(),
        )
        self._latest_motion_ready_projection = _mint_full_mdp_identity(
            ContinuousRecoveryMotionReadyProjection,
            _MOTION_READY_REGISTRY,
            motion_ready_payload,
        )
        self._idle_observation_published = False
        self._advance_mutation_version()

    def publish_after_physics(
        self,
        facts: DeviceContinuousRecoveryPlantFacts,
    ) -> DeviceContinuousRecoveryDoneTermProjection:
        """Publish one common fact and return a literal nonterminating projection."""

        self._require_r07_business_mutation_allowed(label="post-physics publish")
        facts = self._plant_facts(facts)
        active = self._sequence_active
        missed_deadline_ack = active & self._deadline_ack_pending
        self._fault_bits.bitwise_or_(
            missed_deadline_ack.to(dtype=torch.int64) * FAULT_COMMAND_BINDING
        )
        unsettled = active & self._cache_pending & ~self._cache_paid
        duplicate_publish = (
            active
            & self._cache_pending
            & (self._cache_source_step == self._source_step)
            & (self._cache_episode_tick == self._episode_tick)
        )
        publish_collision = unsettled | duplicate_publish
        self._fault_bits.bitwise_or_(
            publish_collision.to(dtype=torch.int64) * FAULT_PUBLISH_COLLISION
        )

        errors, arithmetic_valid, support_count = self._component_errors(facts)
        facts_valid = (
            active
            & facts.facts_valid
            & arithmetic_valid
            & self._reference_valid
        )
        # ``facts_valid=False`` means the construction-bound producer could
        # not publish the required plant row; it is infrastructure, not a
        # poor recovery outcome.  A finite but fallen/unsupported row remains
        # valid learning data and is handled only by readiness/reward values.
        invalid_fact = active & (~facts.facts_valid | ~arithmetic_valid)
        self._fault_bits.bitwise_or_(
            invalid_fact.to(dtype=torch.int64) * FAULT_INVALID_PLANT_FACT
        )

        age = self._episode_tick - self._reward_deadline_tick
        expected = (
            active
            & self._reward_owner_valid
            & (age >= RECOVERY_START_AGE_TICK)
            & (age <= RECOVERY_END_AGE_TICK)
        )
        window_owner_match = torch.all(
            self._window_owner_sha == self._reward_owner_sha, dim=1
        )
        first_window_cell = self._window_expected_count == 0
        expected_age_is_next = torch.where(
            first_window_cell,
            age == RECOVERY_START_AGE_TICK,
            age == self._window_last_expected_age + 1,
        )
        prior_cell_paid = (
            self._window_payment_count == self._window_expected_count
        )
        expected_sequence_ok = (
            self._window_owner_valid
            & window_owner_match
            & expected_age_is_next
            & prior_cell_paid
            & ~publish_collision
        )
        ledger_sequence_fault = expected & ~expected_sequence_ok
        self._fault_bits.bitwise_or_(
            ledger_sequence_fault.to(dtype=torch.int64)
            * FAULT_LEDGER_SEQUENCE
        )
        accepted_expected = expected & expected_sequence_ok
        suffix_missing = (
            self._reward_owner_valid
            & self._reward_owner_played
            & (age >= RECOVERY_START_AGE_TICK)
            & ~self._suffix_complete
        )
        played_reference_match = torch.all(
            self._reference_owner_sha == self._reward_owner_sha, dim=1
        )
        played_reference_fault = (
            expected
            & self._reward_owner_played
            & self._suffix_complete
            & ~played_reference_match
        )
        infrastructure_fault = (
            suffix_missing
            | played_reference_fault
            | ledger_sequence_fault
            | publish_collision
            | invalid_fact
        )
        self._fault_bits.bitwise_or_(
            (suffix_missing | played_reference_fault).to(dtype=torch.int64)
            * FAULT_SUFFIX_INCOMPLETE
        )

        hidden = (
            active
            & self._reference_active
            & ~self._motion_active
            & facts_valid
        )
        owner_matches_current = torch.all(
            self._reward_owner_sha == self._current_key_sha, dim=1
        )
        suffix_gate = ~self._reward_owner_played | self._suffix_complete
        eligible = (
            accepted_expected
            & hidden
            & owner_matches_current
            & suffix_gate
            & ~infrastructure_fault
            & (self._fault_bits == 0)
        )

        scores = 1.0 / (1.0 + torch.square(errors / self._scales.unsqueeze(0)))
        raw_score = torch.sum(scores * self._weights.unsqueeze(0), dim=1) / self._weight_sum
        weighted_reward = torch.where(
            eligible,
            raw_score * float(self.profile.reward_weight),
            torch.zeros_like(raw_score),
        )

        support_ok = support_count >= int(self.profile.minimum_supported_feet)
        all_tolerances = torch.all(
            errors <= self._ready_tolerances.unsqueeze(0), dim=1
        )
        suffix_ready = ~(
            self._reward_owner_valid
            & self._reward_owner_played
            & ~self._suffix_complete
        )
        ready_instant = (
            hidden
            & facts.hard_safety_ok
            & support_ok
            & all_tolerances
            & suffix_ready
            & (self._fault_bits == 0)
        )
        next_streak, ready_live = self._next_ready_dwell(
            ready_instant,
            self._ready_streak,
        )
        first_ready = ready_live & (self._first_ready_tick < 0)
        self._ready_streak.copy_(next_streak)
        self._ready_instant.copy_(ready_instant)
        self._ready_authority.copy_(ready_live)
        self._first_ready_tick.copy_(
            torch.where(first_ready, self._episode_tick, self._first_ready_tick)
        )

        self._cache_pending.copy_(active)
        self._cache_viewed.zero_()
        self._cache_paid.zero_()
        self._cache_source_step.copy_(
            torch.where(active, self._source_step, torch.full_like(self._source_step, -1))
        )
        self._cache_episode_tick.copy_(
            torch.where(active, self._episode_tick, torch.full_like(self._episode_tick, -1))
        )
        self._cache_phase_code.copy_(
            torch.where(
                active,
                self._phase_code,
                torch.full_like(self._phase_code, PHASE_PRE_REVEAL_HIDDEN),
            )
        )
        self._cache_owner_sha.copy_(
            torch.where(
                _expand(self._reward_owner_valid, 2),
                self._reward_owner_sha,
                torch.zeros_like(self._reward_owner_sha),
            )
        )
        self._cache_reference_owner_sha.copy_(
            torch.where(
                _expand(self._reference_valid, 2),
                self._reference_owner_sha,
                torch.zeros_like(self._reference_owner_sha),
            )
        )
        self._cache_age_tick.copy_(
            torch.where(self._reward_owner_valid, age, torch.full_like(age, -1))
        )
        self._cache_expected.copy_(expected)
        self._cache_payment_required.copy_(accepted_expected)
        self._cache_eligible.copy_(eligible)
        self._cache_facts_valid.copy_(facts_valid)
        self._cache_infrastructure_fault.copy_(infrastructure_fault)
        self._cache_errors.copy_(
            torch.where(_expand(active, 2), errors, torch.zeros_like(errors))
        )
        self._cache_scores.copy_(
            torch.where(_expand(active, 2), scores, torch.zeros_like(scores))
        )
        self._cache_raw_score.copy_(
            torch.where(active, raw_score, torch.zeros_like(raw_score))
        )
        self._cache_weighted_reward.copy_(weighted_reward)
        self._cache_ready_instant.copy_(ready_instant)
        self._cache_ready_live.copy_(ready_live)
        self._cache_ready_streak.copy_(self._ready_streak)

        self._host_payment_epoch_open = True
        self._payment_epoch_open.fill_(True)

        prior_expected_count = self._window_expected_count.clone()
        self._window_expected_count.add_(
            accepted_expected.to(dtype=torch.int64)
        )
        self._window_eligible_count.add_(eligible.to(dtype=torch.int64))
        self._window_first_expected_age.copy_(
            torch.where(
                accepted_expected & (prior_expected_count == 0),
                age,
                self._window_first_expected_age,
            )
        )
        self._window_last_expected_age.copy_(
            torch.where(
                accepted_expected,
                age,
                self._window_last_expected_age,
            )
        )
        self._expected_total.add_(
            accepted_expected.to(dtype=torch.int64).sum()
        )
        self._eligible_total.add_(eligible.to(dtype=torch.int64).sum())
        self._ready_instant_total.add_(ready_instant.to(dtype=torch.int64).sum())
        self._first_ready_total.add_(first_ready.to(dtype=torch.int64).sum())
        self._advance_mutation_version()

        # This post-physics fact is applicable to Motion's next policy tick.
        # Keep only the newest opaque handle: raw bool aliases and a replayed
        # older projection are never readiness authority.
        motion_ready_payload = _MotionReadyPayload(
            owner_identity=self._identity,
            owner=self,
            postphysics_valid=self._cache_facts_valid.detach().clone(),
            source_step=self._cache_source_step.detach().clone(),
            reset_generation=self._reset_generation.detach().clone(),
            control_tick=(self._cache_episode_tick + 1).detach().clone(),
            ready=self._cache_ready_live.detach().clone(),
            ready_streak=self._cache_ready_streak.detach().clone(),
            required_dwell=int(self.profile.ready_dwell_ticks),
            foot_supported_lr=(
                facts.facts_valid[:, None]
                & (
                    facts.foot_contact_signal
                    >= float(self.profile.support_contact_threshold)
                )
            ).detach().clone(),
        )
        self._latest_motion_ready_projection = _mint_full_mdp_identity(
            ContinuousRecoveryMotionReadyProjection,
            _MOTION_READY_REGISTRY,
            motion_ready_payload,
        )
        self._idle_observation_published = False

        zeros = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return DeviceContinuousRecoveryDoneTermProjection(
            terminal_requested=zeros,
            truncation_requested=zeros.clone(),
            physical_reset_requested=zeros.clone(),
            carry_reset_requested=zeros.clone(),
            pose_teleport_requested=zeros.clone(),
        )

    def motion_ready_projection(self) -> ContinuousRecoveryMotionReadyProjection:
        """Return the sole post-physics readiness capability for Motion."""

        projection = self._latest_motion_ready_projection
        if projection is None:
            raise ContinuousRecoveryDeviceError(
                "R07 Motion readiness has no real post-physics publication"
            )
        return projection

    def action_epoch_observation_state(
        self,
    ) -> ContinuousRecoveryObservationState:
        """Clone the current direct post-physics state for Observation V2.

        Cold genesis has no simulated post-physics edge, so its only truthful
        state is an explicit invalid/zero row set.  Once direct publication
        has started, losing the retained payload is an error rather than a
        reusable zero fallback.
        """

        if (
            type(self._diagnostic_n2_bundle)
            is not DiagnosticN2ContinuousRecoveryBundle
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 observation state requires the bound direct runtime"
            )
        projection = self._latest_motion_ready_projection
        if projection is None:
            if self._idle_observation_published:
                source_step = torch.full(
                    (self.num_envs,),
                    self._idle_observation_source_step,
                    dtype=torch.int64,
                    device=self.device,
                )
                cadence_tick = self._idle_observation_motion_cadence_tick
                valid_rows = (
                    self._fault_bits.eq(0)
                    & cadence_tick.ge(0)
                    & cadence_tick.lt(torch.iinfo(torch.int64).max)
                )
                safe_cadence = torch.where(
                    valid_rows,
                    cadence_tick,
                    torch.zeros_like(cadence_tick),
                )
                neutral_i64 = torch.full_like(cadence_tick, -1)
                return ContinuousRecoveryObservationState(
                    postphysics_valid=valid_rows.detach().clone(),
                    source_step=torch.where(
                        valid_rows, source_step, neutral_i64
                    ),
                    reset_generation=torch.where(
                        valid_rows,
                        self._idle_observation_reset_generation,
                        neutral_i64,
                    ).detach().clone(),
                    control_tick=torch.where(
                        valid_rows,
                        safe_cadence + 1,
                        neutral_i64,
                    ),
                    ready_streak=(
                        torch.zeros_like(cadence_tick)
                    ),
                    required_dwell=int(self.profile.ready_dwell_ticks),
                    foot_supported_lr=torch.zeros(
                        (self.num_envs, self.num_feet),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                )
            if self._action_epoch_ready_last_source_step >= 0:
                raise ContinuousRecoveryDeviceError(
                    "R07 observation state lost its post-physics publication"
                )
            return ContinuousRecoveryObservationState(
                postphysics_valid=torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                ),
                source_step=torch.full(
                    (self.num_envs,), -1, dtype=torch.int64, device=self.device
                ),
                reset_generation=torch.full(
                    (self.num_envs,), -1, dtype=torch.int64, device=self.device
                ),
                control_tick=torch.full(
                    (self.num_envs,), -1, dtype=torch.int64, device=self.device
                ),
                ready_streak=torch.zeros(
                    self.num_envs, dtype=torch.int64, device=self.device
                ),
                required_dwell=int(self.profile.ready_dwell_ticks),
                foot_supported_lr=torch.zeros(
                    (self.num_envs, self.num_feet),
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        with _FULL_MDP_REWARD_REGISTRY_LOCK:
            payload = _MOTION_READY_REGISTRY.get(projection)
        if (
            payload is None
            or payload.owner_identity is not self._identity
            or payload.owner is not self
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 observation state publication is not owner-issued"
            )
        return ContinuousRecoveryObservationState(
            postphysics_valid=payload.postphysics_valid.detach().clone(),
            source_step=payload.source_step.detach().clone(),
            reset_generation=payload.reset_generation.detach().clone(),
            control_tick=payload.control_tick.detach().clone(),
            ready_streak=payload.ready_streak.detach().clone(),
            required_dwell=payload.required_dwell,
            foot_supported_lr=payload.foot_supported_lr.detach().clone(),
        )

    def require_owned_motion_ready_projection(
        self,
        projection: object,
        *,
        owner_kind: object,
    ) -> ContinuousRecoveryMotionReadyView:
        """Authenticate and clone the current R07->Motion projection."""

        if owner_kind != "motion":
            raise ContinuousRecoveryDeviceError(
                "R07 readiness projection consumer must be exact Motion"
            )
        if (
            type(projection) is not ContinuousRecoveryMotionReadyProjection
            or projection is not self._latest_motion_ready_projection
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 Motion readiness projection is stale or foreign"
            )
        with _FULL_MDP_REWARD_REGISTRY_LOCK:
            payload = _MOTION_READY_REGISTRY.get(projection)
        if (
            payload is None
            or payload.owner_identity is not self._identity
            or payload.owner is not self
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 Motion readiness projection is not owner-issued"
            )
        return ContinuousRecoveryMotionReadyView(
            ready_projection=projection,
            owner_kind="motion",
            ready_identity=self._identity,
            ready=payload.ready.detach().clone(),
            ready_streak=payload.ready_streak.detach().clone(),
            required_dwell=payload.required_dwell,
            control_tick=payload.control_tick.detach().clone(),
        )

    def publish_full_mdp_pre_reward(
        self,
        *,
        control_step: int,
        runtime_owner: object,
    ) -> ContinuousRecoveryFullMdpRewardPublication:
        """Freeze the exact current R07 cache for the top pre-Reward join.

        This is identity publication, not a replacement fact producer.  The
        plant validity, current shot key and deadline all come from the real
        post-physics writer above.  No caller eligibility boolean is accepted.
        """

        self._require_r07_business_mutation_allowed(label="R07 pre-reward publish")
        self._require_full_mdp_reward_operable()
        top = self._require_runtime_owner(runtime_owner)
        top.require_healthy()
        step = _exact_int(control_step, label="control_step")
        if self._active_full_mdp_pre_reward is not None:
            raise ContinuousRecoveryDeviceError(
                "previous R07 full-MDP Reward epoch is still open"
            )
        if not self._host_payment_epoch_open:
            raise ContinuousRecoveryDeviceError(
                "R07 pre-Reward publish requires the real current post-physics epoch"
            )
        expected_source = torch.full_like(self._cache_source_step, step)
        wrong_step = self._cache_source_step != expected_source
        # Keep the hot path device-resident.  A wrong-step row is made
        # non-paying in this same batch and its sticky writer fault is consumed
        # by the sole pre-optimizer global drain before optimizer mutation.
        self._fault_bits.bitwise_or_(
            wrong_step.to(dtype=torch.int64) * FAULT_COMMAND_BINDING
        )
        self._cache_weighted_reward.masked_fill_(wrong_step, 0)
        self._ready_authority.bitwise_and_(~wrong_step)
        payload = _FullMdpPreRewardPayload(
            owner_identity=self._identity,
            runtime_owner=top,
            control_step=step,
            source_step=self._cache_source_step.detach().clone(),
            current_task_key_sha256=self._current_key_sha.detach().clone(),
            recovery_owner_key_sha256=self._cache_owner_sha.detach().clone(),
            recovery_deadline_tick=self._reward_deadline_tick.detach().clone(),
            plant_facts_valid=self._cache_facts_valid.detach().clone(),
            publication_mutation_version=self._mutation_version,
        )
        publication = _mint_full_mdp_identity(
            ContinuousRecoveryFullMdpRewardPublication,
            _FULL_MDP_PRE_REWARD_REGISTRY,
            payload,
        )
        self._active_full_mdp_pre_reward = publication
        self._active_full_mdp_pre_reward_payload = payload
        self._active_full_mdp_payment_verdict = None
        self._active_full_mdp_close_receipt = None
        return publication

    def _require_owned_full_mdp_publication_payload(
        self,
        publication: object,
        *,
        control_step: int,
        runtime_owner: object,
    ) -> _FullMdpPreRewardPayload:
        self._require_full_mdp_reward_operable()
        top = self._require_runtime_owner(runtime_owner)
        step = _exact_int(control_step, label="control_step")
        try:
            payload = _FULL_MDP_PRE_REWARD_REGISTRY.get(publication)
        except TypeError:
            payload = None
        if (
            type(publication) is not ContinuousRecoveryFullMdpRewardPublication
            or publication is not self._active_full_mdp_pre_reward
            or payload is None
            or payload is not self._active_full_mdp_pre_reward_payload
            or payload.owner_identity is not self._identity
            or payload.runtime_owner is not top
            or payload.control_step != step
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 pre-Reward publication is foreign, stale, or wrong-step"
            )
        return payload

    def require_owned_full_mdp_pre_reward(
        self,
        publication: object,
        *,
        control_step: int,
        runtime_owner: object,
    ) -> ContinuousRecoveryFullMdpPreRewardView:
        """Authenticate publication without re-deriving its writer-owned facts."""

        self._require_owned_full_mdp_publication_payload(
            publication,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        zeros = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return ContinuousRecoveryFullMdpPreRewardView(
            terminated=zeros,
            time_out=zeros.clone(),
        )

    def reward_view(self, consumer: str) -> SharedContinuousRecoveryRewardView:
        self._require_r07_business_mutation_allowed(label="reward view")
        if consumer != RECOVERY_REWARD_CONSUMER:
            raise ContinuousRecoveryDeviceError(
                "unknown continuous recovery reward consumer"
            )
        duplicate = self._cache_pending & self._cache_viewed
        self._fault_bits.bitwise_or_(
            duplicate.to(dtype=torch.int64) * FAULT_DUPLICATE_VIEW
        )
        first = self._cache_pending & ~self._cache_viewed
        self._cache_viewed.bitwise_or_(first)
        self._ready_authority.bitwise_and_(self._fault_bits == 0)
        self._advance_mutation_version()
        return self._shared_view

    view = reward_view

    def record_reward_payment(
        self,
        consumer: str,
        payment: torch.Tensor,
    ) -> ContinuousRecoveryFullMdpRewardPaymentVerdict | None:
        self._require_r07_business_mutation_allowed(label="reward payment")
        if consumer != RECOVERY_REWARD_CONSUMER:
            raise ContinuousRecoveryDeviceError(
                "unknown continuous recovery reward consumer"
            )
        values = self._tensor(
            payment,
            label="reward payment",
            shape=(self.num_envs,),
            dtype=self.dtype,
        )
        pending = self._cache_pending
        before_view = pending & ~self._cache_viewed & ~self._cache_paid
        duplicate = pending & self._cache_paid
        nonfinite = ~torch.isfinite(values)
        mismatch = pending & torch.isfinite(values) & (
            values != self._cache_weighted_reward
        )
        outside = ~pending & torch.isfinite(values) & (values != 0)
        self._fault_bits.bitwise_or_(
            before_view.to(dtype=torch.int64) * FAULT_PAYMENT_BEFORE_VIEW
        )
        self._fault_bits.bitwise_or_(
            duplicate.to(dtype=torch.int64) * FAULT_DUPLICATE_PAYMENT
        )
        self._fault_bits.bitwise_or_(
            (nonfinite | mismatch | outside).to(dtype=torch.int64)
            * FAULT_PAYMENT_MISMATCH
        )
        first_attempt = pending & ~self._cache_paid
        good = (
            first_attempt
            & self._cache_viewed
            & ~nonfinite
            & ~mismatch
        )
        self._cache_paid.bitwise_or_(first_attempt)
        paid_expected = good & self._cache_payment_required
        paid_eligible = paid_expected & self._cache_eligible
        self._window_payment_count.add_(
            paid_expected.to(dtype=torch.int64)
        )
        self._window_last_paid_age.copy_(
            torch.where(
                paid_expected,
                self._cache_age_tick,
                self._window_last_paid_age,
            )
        )
        self._payment_total.add_(paid_expected.to(dtype=torch.int64).sum())
        self._income_total.add_(
            torch.where(paid_eligible, values, torch.zeros_like(values)).sum()
        )
        self._ready_authority.bitwise_and_(self._fault_bits == 0)
        self._host_payment_epoch_open = False
        self._payment_epoch_open.fill_(False)
        self._advance_mutation_version()
        publication = self._active_full_mdp_pre_reward
        payload = self._active_full_mdp_pre_reward_payload
        if publication is None or payload is None:
            return None
        if self._active_full_mdp_payment_verdict is not None:
            self._poison_full_mdp_reward("duplicate R07 full-MDP Reward payment")
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward payment was already verdict-issued"
            )
        verdict_payload = _FullMdpPaymentPayload(
            owner_identity=self._identity,
            runtime_owner=payload.runtime_owner,
            publication=publication,
            consumer=consumer,
            control_step=payload.control_step,
            payment_identity=self._shared_view.payment_identity,
        )
        verdict = _mint_full_mdp_identity(
            ContinuousRecoveryFullMdpRewardPaymentVerdict,
            _FULL_MDP_PAYMENT_REGISTRY,
            verdict_payload,
        )
        self._active_full_mdp_payment_verdict = verdict
        return verdict

    record_payment = record_reward_payment

    def require_owned_full_mdp_reward_payment(
        self,
        verdict: object,
        *,
        consumer: str,
        control_step: int,
        runtime_owner: object,
    ) -> ContinuousRecoveryFullMdpRewardPaymentVerdict:
        """Validate the exact owner verdict; caller booleans are never verdicts."""

        payload = self._require_owned_full_mdp_publication_payload(
            self._active_full_mdp_pre_reward,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        try:
            verdict_payload = _FULL_MDP_PAYMENT_REGISTRY.get(verdict)
        except TypeError:
            verdict_payload = None
        if (
            type(verdict) is not ContinuousRecoveryFullMdpRewardPaymentVerdict
            or verdict is not self._active_full_mdp_payment_verdict
            or verdict_payload is None
            or verdict_payload.owner_identity is not self._identity
            or verdict_payload.runtime_owner is not payload.runtime_owner
            or verdict_payload.publication is not self._active_full_mdp_pre_reward
            or consumer != RECOVERY_REWARD_CONSUMER
            or verdict_payload.consumer != consumer
            or verdict_payload.control_step != payload.control_step
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward payment verdict is foreign or mismatched"
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
    ) -> ContinuousRecoveryFullMdpRewardCloseReceipt:
        """Close only the exact one-consumer R07 epoch after owner validation."""

        payload = self._require_owned_full_mdp_publication_payload(
            pre_reward_publication,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        if (
            type(ordered_consumers) is not tuple
            or ordered_consumers != FULL_MDP_REWARD_CONSUMERS
            or type(ordered_payment_verdicts) is not tuple
            or len(ordered_payment_verdicts) != 1
        ):
            self._poison_full_mdp_reward("R07 close order/payment count differs")
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward close order/payment count differs"
            )
        verdict = self.require_owned_full_mdp_reward_payment(
            ordered_payment_verdicts[0],
            consumer=RECOVERY_REWARD_CONSUMER,
            control_step=control_step,
            runtime_owner=runtime_owner,
        )
        if self._active_full_mdp_close_receipt is not None:
            self._poison_full_mdp_reward("duplicate R07 full-MDP Reward close")
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward epoch was already closed"
            )
        close_payload = _FullMdpClosePayload(
            owner_identity=self._identity,
            runtime_owner=payload.runtime_owner,
            publication=pre_reward_publication,
            payment_verdict=verdict,
            control_step=payload.control_step,
        )
        receipt = _mint_full_mdp_identity(
            ContinuousRecoveryFullMdpRewardCloseReceipt,
            _FULL_MDP_CLOSE_REGISTRY,
            close_payload,
        )
        self._active_full_mdp_close_receipt = receipt
        self._active_full_mdp_pre_reward = None
        self._active_full_mdp_pre_reward_payload = None
        self._active_full_mdp_payment_verdict = None
        return receipt

    def require_owned_full_mdp_reward_close(
        self,
        receipt: object,
        *,
        control_step: int,
        runtime_owner: object,
    ) -> ContinuousRecoveryFullMdpRewardCloseReceipt:
        """Authenticate the most recent exact R07 close once top joins it."""

        top = self._require_runtime_owner(runtime_owner)
        step = _exact_int(control_step, label="control_step")
        try:
            payload = _FULL_MDP_CLOSE_REGISTRY.get(receipt)
        except TypeError:
            payload = None
        if (
            type(receipt) is not ContinuousRecoveryFullMdpRewardCloseReceipt
            or receipt is not self._active_full_mdp_close_receipt
            or payload is None
            or payload.owner_identity is not self._identity
            or payload.runtime_owner is not top
            or payload.control_step != step
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 full-MDP Reward close receipt is foreign, stale, or wrong-step"
            )
        return receipt

    def _invariant_masks(self) -> tuple[tuple[str, torch.Tensor], ...]:
        dirty_inactive = ~self._sequence_active & (
            self._current_key_valid
            | self._reward_owner_valid
            | self._reference_valid
            | self._ready_authority
            | self._cache_pending
        )
        ready_without_dwell = self._ready_authority & (
            (~self._ready_instant)
            | (self._ready_streak < int(self.profile.ready_dwell_ticks))
            | ~self._reference_valid
        )
        eligible_without_owner = self._cache_eligible & (
            ~self._cache_expected | ~self._reward_owner_valid
        )
        payment_required_without_expected = (
            self._cache_payment_required & ~self._cache_expected
        )
        paid_without_view = self._cache_paid & ~self._cache_viewed
        played_suffix_owner_mismatch = (
            self._reward_owner_valid
            & self._reward_owner_played
            & self._suffix_complete
            & ~torch.all(
                self._reference_owner_sha == self._reward_owner_sha, dim=1
            )
        )
        unplayed_forces_owner_match = (
            self._reward_owner_valid
            & ~self._reward_owner_played
            & self._suffix_complete
        )
        reward_owner_key_mismatch = self._reward_owner_valid & ~torch.all(
            self._reward_owner_sha == self._current_key_sha, dim=1
        )
        pending_key_mismatch = self._pending_reference_valid & (
            ~self._current_key_valid
            | ~torch.all(
                self._pending_reference_key_sha == self._current_key_sha,
                dim=1,
            )
        )
        reference_kind_invalid = self._reference_valid & (
            (self._reference_owner_kind != OWNER_SEQUENCE_BIRTH)
            & (self._reference_owner_kind != OWNER_COMMITTED_TASK)
        )
        maximum_income = (
            self._eligible_total.to(dtype=self.dtype)
            * float(self.profile.reward_weight)
        )
        summation_slack = (
            torch.finfo(self.dtype).eps
            * torch.maximum(maximum_income.abs(), torch.ones_like(maximum_income))
            * self._eligible_total.clamp_min(1).to(dtype=self.dtype)
            * 4.0
        )
        accounting_invalid = (
            (self._eligible_total > self._expected_total)
            | (self._payment_total > self._expected_total)
            | (self._first_ready_total > self._ready_instant_total)
            | ~torch.isfinite(self._income_total)
            | (self._income_total < 0)
            | (self._income_total > maximum_income + summation_slack)
        ).reshape(1)
        expected_count = self._window_expected_count
        eligible_count = self._window_eligible_count
        payment_count = self._window_payment_count
        empty_window = expected_count == 0
        empty_payment = payment_count == 0
        window_cursor_invalid = (
            (expected_count < 0)
            | (expected_count > RECOVERY_SAMPLE_COUNT)
            | (eligible_count < 0)
            | (eligible_count > expected_count)
            | (payment_count < 0)
            | (payment_count > expected_count)
            | (expected_count - payment_count > 1)
            | (
                empty_window
                & (
                    (self._window_first_expected_age != -1)
                    | (self._window_last_expected_age != -1)
                )
            )
            | (
                ~empty_window
                & (
                    (self._window_first_expected_age != RECOVERY_START_AGE_TICK)
                    | (
                        self._window_last_expected_age
                        != RECOVERY_START_AGE_TICK + expected_count - 1
                    )
                )
            )
            | (empty_payment & (self._window_last_paid_age != -1))
            | (
                ~empty_payment
                & (
                    self._window_last_paid_age
                    != RECOVERY_START_AGE_TICK + payment_count - 1
                )
            )
        )
        window_owner_payload_nonzero = torch.any(
            self._window_owner_sha != 0, dim=1
        )
        window_owner_invalid = (
            (~self._window_owner_valid)
            & (
                window_owner_payload_nonzero
                | (expected_count != 0)
                | (eligible_count != 0)
                | (payment_count != 0)
            )
        ) | (
            self._reward_owner_valid
            & (
                ~self._window_owner_valid
                | ~torch.all(
                    self._window_owner_sha == self._reward_owner_sha, dim=1
                )
            )
        )
        payment_epoch_closed_unsettled = (
            (~self._payment_epoch_open)
            & torch.any(self._cache_pending & ~self._cache_paid)
        ).reshape(1)
        pending_payload_nonzero = (
            torch.any(self._pending_reference_sha != 0, dim=1)
            | torch.any(self._pending_reference_key_sha != 0, dim=1)
            | self._pending_reference_root_position.reshape(self.num_envs, -1).ne(0).any(dim=1)
            | self._pending_reference_root_orientation.reshape(self.num_envs, -1).ne(0).any(dim=1)
            | self._pending_reference_joint_position.reshape(self.num_envs, -1).ne(0).any(dim=1)
            | self._pending_reference_body_position.reshape(self.num_envs, -1).ne(0).any(dim=1)
            | self._pending_reference_body_orientation.reshape(self.num_envs, -1).ne(0).any(dim=1)
            | self._pending_station_anchor_xy.reshape(self.num_envs, -1).ne(0).any(dim=1)
        )
        dirty_invalid_pending = ~self._pending_reference_valid & pending_payload_nonzero
        return (
            ("dirty_inactive", dirty_inactive),
            ("ready_without_dwell", ready_without_dwell),
            ("eligible_without_owner", eligible_without_owner),
            (
                "payment_required_without_expected",
                payment_required_without_expected,
            ),
            ("paid_without_view", paid_without_view),
            ("played_suffix_owner_mismatch", played_suffix_owner_mismatch),
            ("unplayed_marked_suffix_complete", unplayed_forces_owner_match),
            ("reward_owner_key_mismatch", reward_owner_key_mismatch),
            ("pending_key_mismatch", pending_key_mismatch),
            ("reference_kind_invalid", reference_kind_invalid),
            ("deadline_ack_pending", self._deadline_ack_pending),
            ("accounting_invalid", accounting_invalid),
            ("window_cursor_invalid", window_cursor_invalid),
            ("window_owner_invalid", window_owner_invalid),
            (
                "payment_epoch_closed_unsettled",
                payment_epoch_closed_unsettled,
            ),
            ("dirty_invalid_pending", dirty_invalid_pending),
        )

    @staticmethod
    def _global_owner_row_values(owner_row: object) -> dict[str, object]:
        if getattr(owner_row, "owner_kind", None) != R07_GLOBAL_DRAIN_OWNER_KIND:
            raise ContinuousRecoveryDeviceError(
                "global drain owner row is not the R07 row"
            )
        raw = getattr(owner_row, "values", None)
        if not isinstance(raw, tuple):
            raise ContinuousRecoveryDeviceError(
                "global drain R07 row values must be a tuple"
            )
        names = tuple(
            value[0]
            for value in raw
            if isinstance(value, tuple) and len(value) == 2
        )
        if len(names) != len(raw) or names != R07_GLOBAL_DRAIN_FIELD_NAMES:
            raise ContinuousRecoveryDeviceError(
                "global drain R07 row schema differs"
            )
        return dict(raw)

    def _require_r07_global_drain_operable(self) -> None:
        if self._r07_global_drain_poisoned:
            raise ContinuousRecoveryDeviceError(
                "R07 global drain is poisoned and requires cold replacement"
            )

    def _require_r07_business_mutation_allowed(self, *, label: str) -> None:
        self._require_r07_global_drain_operable()
        if self._active_r07_global_drain is not None:
            raise ContinuousRecoveryDeviceError(
                f"{label} cannot mutate an active R07 global drain lease"
            )
        if (
            self._active_full_mdp_pre_reward is not None
            and label
            not in (
                "reward view",
                "reward payment",
                "R07 pre-reward publish",
            )
        ):
            raise ContinuousRecoveryDeviceError(
                f"{label} cannot cross an open R07 full-MDP Reward epoch"
            )

    def _advance_mutation_version(self) -> None:
        """Advance the live writer and invalidate its cold ACK frontier."""

        self._mutation_version += 1
        self._checkpoint_requires_global_drain_ack = True

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority: object,
        update_index: int,
        completed_environment_steps: int,
    ) -> object:
        """Freeze R07 evidence on device for the sole global PPO transfer.

        This method deliberately does not call :meth:`drain_ppo_ledger` and
        performs no host tensor observation.  The global authority clones the
        packed row on device.  Portable audit rows are materialized only from
        that decoded row after the optimizer and its callbacks succeed.
        """

        self._require_r07_business_mutation_allowed(label="global PPO drain")

        self._require_r07_global_drain_operable()
        update = _exact_int(update_index, label="update_index")
        completed = _exact_int(
            completed_environment_steps,
            label="completed_environment_steps",
            minimum=1,
        )
        if update <= self._last_drained_update:
            raise ContinuousRecoveryDeviceError(
                "update_index must strictly advance the R07 drain highwater"
            )
        if self._active_r07_global_drain is not None:
            raise ContinuousRecoveryDeviceError(
                "an R07 global drain lease is already active"
            )
        if getattr(authority, "owner_kind", None) != R07_GLOBAL_DRAIN_OWNER_KIND:
            raise ContinuousRecoveryDeviceError(
                "global drain authority is not bound to R07"
            )
        if tuple(getattr(authority, "field_names", ())) != R07_GLOBAL_DRAIN_FIELD_NAMES:
            raise ContinuousRecoveryDeviceError(
                "global drain R07 authority schema differs"
            )
        mint = getattr(authority, "mint_device_pack", None)
        require_owned_ack = getattr(authority, "require_owned_ack", None)
        try:
            from whole_body_tracking.tasks.tracking.mdp import (
                action_ball_full_mdp_ppo_drain as drain,
            )
        except (ImportError, ModuleNotFoundError):
            import action_ball_full_mdp_ppo_drain as drain
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
            raise ContinuousRecoveryDeviceError(
                "global drain R07 exact authority API differs"
            )
        invariant_masks = self._invariant_masks()
        if tuple(name for name, _mask in invariant_masks) != tuple(
            name.removeprefix("invariant_").removesuffix("_count")
            for name in R07_GLOBAL_DRAIN_INVARIANT_FIELDS
        ):
            raise ContinuousRecoveryDeviceError(
                "R07 invariant schema differs from the global drain ABI"
            )
        fault_counts = tuple(
            ((self._fault_bits & bit) != 0).to(dtype=torch.int64).sum()
            for _name, bit in FAULTS
        )
        invariant_counts = tuple(
            mask.to(dtype=torch.int64).sum()
            for _name, mask in invariant_masks
        )
        # Window closure is intentionally not packed: it is a deterministic
        # projection of the six transferred cursors, so transmitting it would
        # create a same-writer self-check rather than new evidence.
        income_bits = (
            self._income_total.to(dtype=torch.float64).reshape(1).view(torch.int64)
        )
        scalar_values = (
            torch.tensor(
                (self._mutation_version,),
                dtype=torch.int64,
                device=self.device,
            ),
            torch.stack(fault_counts).sum().reshape(1),
            torch.stack(invariant_counts).sum().reshape(1),
            *(value.reshape(1) for value in fault_counts),
            *(value.reshape(1) for value in invariant_counts),
            self._expected_total.reshape(1),
            self._eligible_total.reshape(1),
            self._payment_total.reshape(1),
            income_bits,
            self._ready_instant_total.reshape(1),
            self._first_ready_total.reshape(1),
            self._played_deadline_total.reshape(1),
            self._unplayed_deadline_total.reshape(1),
        )
        per_env_values = (
            self._window_expected_count,
            self._window_eligible_count,
            self._window_payment_count,
            self._window_first_expected_age + 1,
            self._window_last_expected_age + 1,
            self._window_last_paid_age + 1,
        )
        values = torch.cat(
            tuple(value.to(dtype=torch.int64).reshape(-1) for value in scalar_values)
            + tuple(
                value.to(dtype=torch.int64).reshape(-1)
                for value in per_env_values
            ),
            dim=0,
        ).contiguous()
        pack = mint(leaf=self, values=values)
        self._active_r07_global_drain = _PreparedR07GlobalDrain(
            pack=pack,
            authority=authority,
            update_index=update,
            completed_environment_steps=completed,
            mutation_version=self._mutation_version,
            window_owner_sha256=tuple(
                None if key is None else key.canonical_sha256
                for key in self._host_window_owner_keys
            ),
        )
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack: object) -> None:
        """Release a pre-transfer lease without changing recovery state."""

        active = self._active_r07_global_drain
        if active is None or pack is not active.pack or active.stage != "prepared":
            raise ContinuousRecoveryDeviceError(
                "R07 global drain abort pack is stale or foreign"
            )
        if self._mutation_version != active.mutation_version:
            self._r07_global_drain_poisoned = True
            self._r07_global_drain_poison_reason = (
                "R07 mutated while a global drain lease was active"
            )
            active.stage = "poisoned"
            raise ContinuousRecoveryDeviceError(
                self._r07_global_drain_poison_reason
            )
        active.stage = "aborted"
        self._active_r07_global_drain = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack: object,
        receipt: object,
        owner_row: object,
    ) -> None:
        """Materialize the portable R07 receipt from the one global D2H."""

        self._require_r07_global_drain_operable()
        active = self._active_r07_global_drain
        if active is None or pack is not active.pack or active.stage != "prepared":
            raise ContinuousRecoveryDeviceError(
                "R07 global drain acknowledgement pack is stale or foreign"
            )
        # Authenticate the exact construction-bound coordinator and its
        # optimizer-return window before inspecting recovery facts.
        active.authority.require_owned_ack(
            leaf=self,
            pack=pack,
            receipt=receipt,
            owner_row=owner_row,
        )
        if self._mutation_version != active.mutation_version:
            raise ContinuousRecoveryDeviceError(
                "R07 mutated after its global drain device snapshot"
            )
        if (
            getattr(receipt, "update_index", None) != active.update_index
            or getattr(receipt, "completed_environment_steps", None)
            != active.completed_environment_steps
            or getattr(receipt, "device_to_host_transfers", None) != 1
            or getattr(receipt, "drain_sequence", None) != self._drain_sequence + 1
        ):
            raise ContinuousRecoveryDeviceError(
                "global drain receipt does not match the R07 lease"
            )
        receipt_rows = getattr(receipt, "owner_rows", None)
        if not isinstance(receipt_rows, tuple) or not any(
            row is owner_row for row in receipt_rows
        ):
            raise ContinuousRecoveryDeviceError(
                "global drain R07 owner row is not owned by the receipt"
            )
        values = self._global_owner_row_values(owner_row)
        active.stage = "acknowledging"
        if values["mutation_version"] != active.mutation_version:
            raise ContinuousRecoveryDeviceError(
                "global drain R07 mutation version differs"
            )
        fault_counts = tuple(
            (name, values[field])
            for (name, _bit), field in zip(FAULTS, R07_GLOBAL_DRAIN_FAULT_FIELDS)
        )
        invariant_counts = tuple(
            (
                field.removeprefix("invariant_").removesuffix("_count"),
                values[field],
            )
            for field in R07_GLOBAL_DRAIN_INVARIANT_FIELDS
        )
        def per_env(field: str) -> tuple[int, ...]:
            value = values[field]
            if (
                not isinstance(value, tuple)
                or len(value) != self.num_envs
                or any(type(item) is not int for item in value)
            ):
                raise ContinuousRecoveryDeviceError(
                    f"global drain R07 {field} row differs"
                )
            return value

        expected = per_env("window_expected_count")
        eligible = per_env("window_eligible_count")
        payment = per_env("window_payment_count")
        first_encoded = per_env("window_first_expected_age_tick_encoded")
        last_encoded = per_env("window_last_expected_age_tick_encoded")
        paid_encoded = per_env("window_last_paid_age_tick_encoded")
        window_rows = tuple(
            ContinuousRecoveryWindowLedgerRow(
                env_id=env_id,
                owner_key_sha256=active.window_owner_sha256[env_id],
                expected_count=expected[env_id],
                eligible_count=eligible[env_id],
                payment_count=payment[env_id],
                first_expected_age_tick=(
                    None if first_encoded[env_id] == 0 else first_encoded[env_id] - 1
                ),
                last_expected_age_tick=(
                    None if last_encoded[env_id] == 0 else last_encoded[env_id] - 1
                ),
                last_paid_age_tick=(
                    None if paid_encoded[env_id] == 0 else paid_encoded[env_id] - 1
                ),
                closed_68_of_68=(
                    expected[env_id] == RECOVERY_SAMPLE_COUNT
                    and payment[env_id] == RECOVERY_SAMPLE_COUNT
                    and first_encoded[env_id] == RECOVERY_START_AGE_TICK + 1
                    and last_encoded[env_id] == RECOVERY_END_AGE_TICK + 1
                    and paid_encoded[env_id] == RECOVERY_END_AGE_TICK + 1
                ),
            )
            for env_id in range(self.num_envs)
        )
        income_bits = values["reward_income_total_float64_bits"]
        if type(income_bits) is not int or income_bits < 0:
            raise ContinuousRecoveryDeviceError(
                "global drain R07 reward-income bits differ"
            )
        income = struct.unpack("!d", struct.pack("!Q", income_bits))[0]
        if not (income >= 0.0 and income < float("inf")):
            raise ContinuousRecoveryDeviceError(
                "global drain R07 reward income is not finite nonnegative"
            )
        portable = ContinuousRecoveryBoundaryReceipt(
            schema_version=SCHEMA_VERSION,
            update_index=active.update_index,
            drain_sequence=getattr(receipt, "drain_sequence"),
            mutation_version=active.mutation_version,
            num_envs=self.num_envs,
            component_names=self.component_names,
            fault_counts=fault_counts + invariant_counts,
            recovery_window_rows=window_rows,
            recovery_expected_total=values["recovery_expected_total"],
            reward_eligible_total=values["reward_eligible_total"],
            reward_payment_total=values["reward_payment_total"],
            reward_income_total=income,
            ready_instant_total=values["ready_instant_total"],
            first_ready_total=values["first_ready_total"],
            played_deadline_total=values["played_deadline_total"],
            unplayed_deadline_total=values["unplayed_deadline_total"],
            checkpoint_safe=True,
            device_to_host_transfers=1,
            integration_status=INTEGRATION_STATUS,
            runtime_wiring_connected=RUNTIME_WIRING_CONNECTED,
            cuda_profiled=CUDA_PROFILED,
            formal_exact_resume_integrated=FORMAL_EXACT_RESUME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )
        self._drain_sequence = portable.drain_sequence
        self._last_drained_update = active.update_index
        self._last_receipt = portable
        self._last_receipt_consumed = False
        self._last_r07_global_drain_pair = (receipt, portable)
        self._last_globally_acknowledged_mutation_version = active.mutation_version
        self._checkpoint_requires_global_drain_ack = False
        active.stage = "acknowledged"
        self._active_r07_global_drain = None

    def require_owned_pre_optimizer_ppo_boundary_receipt(
        self,
        global_receipt: object,
    ) -> ContinuousRecoveryBoundaryReceipt:
        """Return the exact portable R07 audit made by this global ACK."""

        self._require_r07_global_drain_operable()
        pair = self._last_r07_global_drain_pair
        if (
            pair is None
            or global_receipt is not pair[0]
            or getattr(global_receipt, "acknowledged", None) is not True
        ):
            raise ContinuousRecoveryDeviceError(
                "global receipt is foreign, stale, or not acknowledged by R07"
            )
        return pair[1]

    def poison_pre_optimizer_ppo_boundary(self, *, reason: object) -> None:
        """Sticky fail-stop after any post-transfer or partial-ACK failure."""

        if self._r07_global_drain_poison_reason is None:
            self._r07_global_drain_poison_reason = (
                reason
                if type(reason) is str and bool(reason)
                else "unspecified global PPO drain failure"
            )
        self._r07_global_drain_poisoned = True
        if self._active_r07_global_drain is not None:
            self._active_r07_global_drain.stage = "poisoned"

    def drain_ppo_ledger(
        self, *, update_index: int
    ) -> ContinuousRecoveryBoundaryReceipt:
        """Perform the sole hot-path device-to-host boundary transfer."""

        self._require_r07_global_drain_operable()
        if self._active_r07_global_drain is not None:
            raise ContinuousRecoveryDeviceError(
                "legacy R07 drain cannot overlap the global drain"
            )
        update = _exact_int(update_index, label="update_index")
        if update <= self._last_drained_update:
            raise ContinuousRecoveryDeviceError(
                "update_index must strictly advance"
            )
        invariant_masks = self._invariant_masks()
        count_scalars = [
            *[
                ((self._fault_bits & bit) != 0).to(dtype=torch.int64).sum()
                for _, bit in FAULTS
            ],
            *[mask.to(dtype=torch.int64).sum() for _, mask in invariant_masks],
            self._expected_total,
            self._eligible_total,
            self._payment_total,
            self._ready_instant_total,
            self._first_ready_total,
            self._played_deadline_total,
            self._unplayed_deadline_total,
        ]
        scalar_packed = torch.stack(
            [
                *[value.to(dtype=torch.float64) for value in count_scalars],
                self._income_total.to(dtype=torch.float64),
            ]
        )
        window_closed = (
            self._window_owner_valid
            & (self._window_expected_count == RECOVERY_SAMPLE_COUNT)
            & (self._window_payment_count == RECOVERY_SAMPLE_COUNT)
            & (
                self._window_first_expected_age == RECOVERY_START_AGE_TICK
            )
            & (self._window_last_expected_age == RECOVERY_END_AGE_TICK)
            & (self._window_last_paid_age == RECOVERY_END_AGE_TICK)
        )
        window_packed = torch.stack(
            (
                self._window_expected_count,
                self._window_eligible_count,
                self._window_payment_count,
                self._window_first_expected_age,
                self._window_last_expected_age,
                self._window_last_paid_age,
                window_closed.to(dtype=torch.int64),
            ),
            dim=1,
        ).to(dtype=torch.float64)
        packed = torch.cat((scalar_packed, window_packed.reshape(-1)))
        host_values = packed.detach().to(device="cpu").tolist()
        scalar_count = len(count_scalars)
        host_counts = host_values[:scalar_count]
        income = float(host_values[scalar_count])
        host_window_values = host_values[scalar_count + 1 :]
        cursor = 0
        fault_counts = tuple(
            (name, int(host_counts[cursor + index]))
            for index, (name, _) in enumerate(FAULTS)
        )
        cursor += len(FAULTS)
        invariant_counts = tuple(
            (name, int(host_counts[cursor + index]))
            for index, (name, _) in enumerate(invariant_masks)
        )
        cursor += len(invariant_masks)
        totals = tuple(int(value) for value in host_counts[cursor:])
        safe = all(
            count == 0 for _, count in fault_counts + invariant_counts
        )
        recovery_window_rows = []
        width = 7
        for env_id in range(self.num_envs):
            row = host_window_values[env_id * width : (env_id + 1) * width]
            owner_key = self._host_window_owner_keys[env_id]
            recovery_window_rows.append(
                ContinuousRecoveryWindowLedgerRow(
                    env_id=env_id,
                    owner_key_sha256=(
                        None
                        if owner_key is None
                        else owner_key.canonical_sha256
                    ),
                    expected_count=int(row[0]),
                    eligible_count=int(row[1]),
                    payment_count=int(row[2]),
                    first_expected_age_tick=(
                        None if int(row[3]) < 0 else int(row[3])
                    ),
                    last_expected_age_tick=(
                        None if int(row[4]) < 0 else int(row[4])
                    ),
                    last_paid_age_tick=(
                        None if int(row[5]) < 0 else int(row[5])
                    ),
                    closed_68_of_68=bool(int(row[6])),
                )
            )

        self._drain_sequence += 1
        receipt = ContinuousRecoveryBoundaryReceipt(
            schema_version=SCHEMA_VERSION,
            update_index=update,
            drain_sequence=self._drain_sequence,
            mutation_version=self._mutation_version,
            num_envs=self.num_envs,
            component_names=self.component_names,
            fault_counts=fault_counts + invariant_counts,
            recovery_window_rows=tuple(recovery_window_rows),
            recovery_expected_total=totals[0],
            reward_eligible_total=totals[1],
            reward_payment_total=totals[2],
            reward_income_total=income,
            ready_instant_total=totals[3],
            first_ready_total=totals[4],
            played_deadline_total=totals[5],
            unplayed_deadline_total=totals[6],
            checkpoint_safe=safe,
            device_to_host_transfers=1,
            integration_status=INTEGRATION_STATUS,
            runtime_wiring_connected=RUNTIME_WIRING_CONNECTED,
            cuda_profiled=CUDA_PROFILED,
            formal_exact_resume_integrated=FORMAL_EXACT_RESUME_INTEGRATED,
            launch_authorized=LAUNCH_AUTHORIZED,
        )
        self._last_drained_update = update
        self._last_receipt = receipt
        self._last_receipt_consumed = False
        return receipt

    drain_ppo_boundary = drain_ppo_ledger

    _CONFIG_TENSORS = frozenset(
        (
            "_env_ids",
            "_weights",
            "_scales",
            "_ready_tolerances",
            "_weight_sum",
            "_profile_sha",
            # Fresh diagnostic ActionEpoch dwell state is not part of the
            # legacy exact-resume schema.  Its live epoch owner is separately
            # non-resumable and reconstructs these rows from post-physics
            # facts after a cold start.
            "_action_epoch_ready_instant",
            "_action_epoch_ready_live",
            "_action_epoch_ready_streak",
            "_action_epoch_ready_reference_kind",
            "_action_epoch_ready_reference_action_slot",
            "_action_epoch_ready_reference_action_uid",
            "_action_epoch_first_ready_source_step",
            "_action_epoch_ready_last_motion_cadence_tick",
            "_action_epoch_ready_last_reset_generation",
        )
    )

    def _state_tensors(self) -> dict[str, torch.Tensor]:
        return {
            name[1:]: value
            for name, value in vars(self).items()
            if name.startswith("_")
            and isinstance(value, torch.Tensor)
            and name not in self._CONFIG_TENSORS
        }

    def _host_state(self) -> dict[str, object]:
        def key(value: object | None) -> object:
            return None if value is None else value.to_mapping()

        return {
            "current_keys": [key(value) for value in self._host_keys],
            "window_owner_keys": [
                key(value) for value in self._host_window_owner_keys
            ],
            "ready_owner_keys": [
                key(value) for value in self._host_ready_owner_keys
            ],
            "played": list(self._host_played),
            "reference_sha256": list(self._host_reference_sha),
            "pending_reference_sha256": list(
                self._host_pending_reference_sha
            ),
            "reset_generation": list(self._host_reset_generation),
            "scheduled_ordinal": list(self._host_scheduled_ordinal),
            "deadline_consumed": list(self._host_deadline_consumed),
            "last_deadline_tick": list(self._host_last_deadline),
            "last_reveal_tick": list(self._host_last_reveal),
            "payment_epoch_open": self._host_payment_epoch_open,
        }

    def _validate_host_checkpoint_semantics(self) -> None:
        """Reject a re-sealed host row before comparing its device mirror."""

        for env_id in range(self.num_envs):
            generation = self._host_reset_generation[env_id]
            current = self._host_keys[env_id]
            window = self._host_window_owner_keys[env_id]
            ready = self._host_ready_owner_keys[env_id]
            played = self._host_played[env_id]
            reference_sha = self._host_reference_sha[env_id]
            pending_sha = self._host_pending_reference_sha[env_id]
            ordinal = self._host_scheduled_ordinal[env_id]
            consumed = self._host_deadline_consumed[env_id]
            deadline = self._host_last_deadline[env_id]
            reveal = self._host_last_reveal[env_id]
            active = generation is not None
            if not active:
                if any(
                    value is not None
                    for value in (
                        current,
                        window,
                        ready,
                        reference_sha,
                        pending_sha,
                        deadline,
                        reveal,
                    )
                ) or played or consumed or ordinal != -1:
                    raise ContinuousRecoveryDeviceError(
                        "inactive checkpoint host row owns semantic state"
                    )
                continue
            if reference_sha is None:
                raise ContinuousRecoveryDeviceError(
                    "active checkpoint host row lacks a reference"
                )
            if current is None:
                if any(
                    value is not None
                    for value in (pending_sha, deadline, reveal)
                ) or played or consumed or ordinal != -1:
                    raise ContinuousRecoveryDeviceError(
                        "checkpoint host row has task facts without a current key"
                    )
            else:
                if (
                    current.env_id != env_id
                    or current.reset_generation != generation
                    or current.source_sha256 != self.profile.source_sha256
                    or current.config_sha256 != self.profile.config_sha256
                    or current.swing_generation != ordinal
                    or current.shot_index != ordinal + 1
                    or deadline is None
                    or reveal is None
                    or deadline <= reveal
                ):
                    raise ContinuousRecoveryDeviceError(
                        "checkpoint current host key/timing binding differs"
                    )
                if consumed and window != current:
                    raise ContinuousRecoveryDeviceError(
                        "consumed checkpoint deadline lost its full window key"
                    )
                if pending_sha is not None and consumed and not played:
                    raise ContinuousRecoveryDeviceError(
                        "unplayed consumed checkpoint retained private frame0"
                    )
            for label, key in (("window", window), ("ready", ready)):
                if key is None:
                    continue
                if (
                    key.env_id != env_id
                    or key.reset_generation != generation
                    or key.source_sha256 != self.profile.source_sha256
                    or key.config_sha256 != self.profile.config_sha256
                    or key.swing_generation + 1 != key.shot_index
                    or current is None
                    or key.shot_index > current.shot_index
                ):
                    raise ContinuousRecoveryDeviceError(
                        f"checkpoint {label} host key lineage differs"
                    )
                for name in (
                    "action_uid",
                    "action_slot",
                    "birth_sha256",
                    "run_id",
                    "carry_chain_id",
                ):
                    if getattr(key, name) != getattr(current, name):
                        raise ContinuousRecoveryDeviceError(
                            f"checkpoint {label} host key carry lineage differs"
                        )

    def _host_device_checkpoint_mismatch_masks(
        self,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Cross-bind every host authority field to its device mirror."""

        zero_sha = "00" * 32
        active_rows = [value is not None for value in self._host_reset_generation]
        current_valid_rows = [value is not None for value in self._host_keys]
        window_valid_rows = [
            value is not None for value in self._host_window_owner_keys
        ]
        pending_valid_rows = [
            value is not None for value in self._host_pending_reference_sha
        ]
        active = torch.tensor(
            active_rows, dtype=torch.bool, device=self.device
        )
        current_valid = torch.tensor(
            current_valid_rows, dtype=torch.bool, device=self.device
        )
        window_valid = torch.tensor(
            window_valid_rows, dtype=torch.bool, device=self.device
        )
        pending_valid = torch.tensor(
            pending_valid_rows, dtype=torch.bool, device=self.device
        )
        reset_generation = torch.tensor(
            [value or 0 for value in self._host_reset_generation],
            dtype=torch.int64,
            device=self.device,
        )
        ordinals = torch.tensor(
            self._host_scheduled_ordinal,
            dtype=torch.int64,
            device=self.device,
        )
        reveals = torch.tensor(
            [value if value is not None else -1 for value in self._host_last_reveal],
            dtype=torch.int64,
            device=self.device,
        )
        deadlines = torch.tensor(
            [value if value is not None else -1 for value in self._host_last_deadline],
            dtype=torch.int64,
            device=self.device,
        )
        played = torch.tensor(
            self._host_played, dtype=torch.bool, device=self.device
        )
        consumed = torch.tensor(
            self._host_deadline_consumed,
            dtype=torch.bool,
            device=self.device,
        )

        def key_digests(values: Sequence[object | None]) -> torch.Tensor:
            return _digest_tensor(
                [
                    zero_sha if value is None else value.canonical_sha256
                    for value in values
                ],
                device=self.device,
            )

        current_sha = key_digests(self._host_keys)
        window_sha = key_digests(self._host_window_owner_keys)
        pending_key_sha = current_sha
        reference_sha = _digest_tensor(
            [value if value is not None else zero_sha for value in self._host_reference_sha],
            device=self.device,
        )
        pending_reference_sha = _digest_tensor(
            [
                value if value is not None else zero_sha
                for value in self._host_pending_reference_sha
            ],
            device=self.device,
        )
        reference_owner_kind_rows: list[int] = []
        reference_owner_sha_rows: list[str] = []
        for env_id, (generation, ref_sha, ready_key) in enumerate(
            zip(
                self._host_reset_generation,
                self._host_reference_sha,
                self._host_ready_owner_keys,
            )
        ):
            if generation is None or ref_sha is None:
                reference_owner_kind_rows.append(OWNER_NONE)
                reference_owner_sha_rows.append(zero_sha)
            elif ready_key is None:
                reference_owner_kind_rows.append(OWNER_SEQUENCE_BIRTH)
                reference_owner_sha_rows.append(
                    hashlib.sha256(
                        (
                            "action_ball_continuous_birth_ready_owner_v1|"
                            f"{env_id}|{generation}|{ref_sha}"
                        ).encode("ascii")
                    ).hexdigest()
                )
            else:
                reference_owner_kind_rows.append(OWNER_COMMITTED_TASK)
                reference_owner_sha_rows.append(ready_key.canonical_sha256)
        reference_owner_kind = torch.tensor(
            reference_owner_kind_rows,
            dtype=torch.int64,
            device=self.device,
        )
        reference_owner_sha = _digest_tensor(
            reference_owner_sha_rows, device=self.device
        )
        suffix_complete = torch.tensor(
            [
                bool(
                    current is not None
                    and ready is not None
                    and current == ready
                    and was_played
                    and was_consumed
                )
                for current, ready, was_played, was_consumed in zip(
                    self._host_keys,
                    self._host_ready_owner_keys,
                    self._host_played,
                    self._host_deadline_consumed,
                )
            ],
            dtype=torch.bool,
            device=self.device,
        )
        reward_sha = torch.where(
            _expand(consumed, 2), current_sha, torch.zeros_like(current_sha)
        )
        reward_deadline = torch.where(
            consumed, deadlines, torch.full_like(deadlines, -1)
        )
        return (
            ("sequence_active", self._sequence_active != active),
            ("reset_generation", self._reset_generation != reset_generation),
            ("current_key_valid", self._current_key_valid != current_valid),
            (
                "current_key_sha",
                torch.any(self._current_key_sha != current_sha, dim=1),
            ),
            ("scheduled_ordinal", self._scheduled_ordinal != ordinals),
            ("current_reveal_tick", self._current_reveal_tick != reveals),
            ("current_deadline_tick", self._current_deadline_tick != deadlines),
            ("host_played_ever", self._ever_motion_active != played),
            ("host_played_receipt", self._playback_receipted != played),
            ("reward_owner_valid", self._reward_owner_valid != consumed),
            (
                "reward_owner_sha",
                torch.any(self._reward_owner_sha != reward_sha, dim=1),
            ),
            (
                "reward_deadline_tick",
                self._reward_deadline_tick != reward_deadline,
            ),
            ("reward_owner_played", self._reward_owner_played != (played & consumed)),
            ("window_owner_valid", self._window_owner_valid != window_valid),
            (
                "window_owner_sha",
                torch.any(self._window_owner_sha != window_sha, dim=1),
            ),
            ("pending_reference_valid", self._pending_reference_valid != pending_valid),
            (
                "pending_reference_sha",
                torch.any(
                    self._pending_reference_sha != pending_reference_sha,
                    dim=1,
                ),
            ),
            (
                "pending_reference_key_sha",
                pending_valid
                & torch.any(
                    self._pending_reference_key_sha != pending_key_sha,
                    dim=1,
                ),
            ),
            ("reference_valid", self._reference_valid != active),
            (
                "reference_sha",
                torch.any(self._reference_sha != reference_sha, dim=1),
            ),
            (
                "reference_owner_kind",
                self._reference_owner_kind != reference_owner_kind,
            ),
            (
                "reference_owner_sha",
                torch.any(
                    self._reference_owner_sha != reference_owner_sha, dim=1
                ),
            ),
            ("suffix_complete", self._suffix_complete != suffix_complete),
            (
                "payment_epoch_open",
                (self._payment_epoch_open != self._host_payment_epoch_open).reshape(1),
            ),
        )

    @staticmethod
    def _tensor_manifest(
        tensors: Mapping[str, torch.Tensor]
    ) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "dtype": str(tensors[name].dtype),
                "shape": list(tensors[name].shape),
                "numel": tensors[name].numel(),
                "element_size": tensors[name].element_size(),
            }
            for name in sorted(tensors)
        ]

    @staticmethod
    def _tensor_bytes_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
        byte_rows = [
            tensors[name]
            .detach()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            for name in sorted(tensors)
        ]
        if byte_rows:
            packed = torch.cat(byte_rows)
            raw = bytes(packed.to(device="cpu").tolist())
        else:  # pragma: no cover - the coordinator always owns tensors
            raw = b""
        return hashlib.sha256(raw).hexdigest()

    def checkpoint_state(
        self,
        receipt: ContinuousRecoveryBoundaryReceipt,
    ) -> dict[str, object]:
        """Export an exact checkpoint authorized by the latest safe drain."""

        if (
            self._active_r07_global_drain is None
            and self._active_full_mdp_pre_reward is not None
        ):
            raise ContinuousRecoveryDeviceError(
                "checkpoint cannot cross an open R07 full-MDP Reward epoch"
            )
        self._require_r07_global_drain_operable()
        if self._active_r07_global_drain is not None:
            raise ContinuousRecoveryDeviceError(
                "checkpoint cannot overlap an active R07 global drain lease"
            )
        if receipt is not self._last_receipt:
            raise ContinuousRecoveryDeviceError(
                "checkpoint receipt is foreign or stale"
            )
        if self._last_receipt_consumed:
            raise ContinuousRecoveryDeviceError(
                "checkpoint receipt was already consumed"
            )
        if receipt.mutation_version != self._mutation_version:
            raise ContinuousRecoveryDeviceError(
                "coordinator mutated after PPO-boundary drain"
            )
        if self._last_r07_global_drain_pair is not None and (
            self._checkpoint_requires_global_drain_ack
            or self._last_globally_acknowledged_mutation_version
            != self._mutation_version
        ):
            raise ContinuousRecoveryDeviceError(
                "checkpoint lacks the exact globally ACKed R07 mutation frontier"
            )
        if not receipt.checkpoint_safe:
            raise ContinuousRecoveryDeviceError(
                "PPO-boundary receipt is not checkpoint-safe"
            )
        tensors = {
            name: tensor.detach().clone()
            for name, tensor in self._state_tensors().items()
        }
        host = self._host_state()
        manifest = self._tensor_manifest(tensors)
        tensor_sha = self._tensor_bytes_sha256(tensors)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "kind": "action_ball_continuous_recovery_device_checkpoint_v1",
            "integration_status": INTEGRATION_STATUS,
            "profile_sha256": self.profile.canonical_sha256,
            "num_envs": self.num_envs,
            "dtype": str(self.dtype),
            "component_names": list(self.component_names),
            "drain_update_index": receipt.update_index,
            "drain_sequence": receipt.drain_sequence,
            "mutation_version": self._mutation_version,
            "host_state": host,
            "tensor_manifest": manifest,
            "tensor_bytes_sha256": tensor_sha,
        }
        checkpoint_sha = _runtime.canonical_sha256(identity)
        self._last_receipt_consumed = True
        return {
            **identity,
            "state_tensors": tensors,
            "checkpoint_sha256": checkpoint_sha,
        }

    checkpoint = checkpoint_state

    @classmethod
    def from_checkpoint(
        cls,
        *,
        profile: _runtime.ContinuousRecoveryProfile,
        checkpoint: Mapping[str, object],
        expected_checkpoint_sha256: str,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "ContinuousRecoveryDeviceCoordinator":
        """Restore only from an independently retained checkpoint digest."""

        expected_sha = _sha256(
            expected_checkpoint_sha256, label="expected_checkpoint_sha256"
        )
        if not isinstance(checkpoint, Mapping):
            raise ContinuousRecoveryDeviceError("checkpoint must be a mapping")
        expected_fields = {
            "schema_version",
            "kind",
            "integration_status",
            "profile_sha256",
            "num_envs",
            "dtype",
            "component_names",
            "drain_update_index",
            "drain_sequence",
            "mutation_version",
            "host_state",
            "tensor_manifest",
            "tensor_bytes_sha256",
            "state_tensors",
            "checkpoint_sha256",
        }
        if set(checkpoint) != expected_fields:
            raise ContinuousRecoveryDeviceError("checkpoint fields differ")
        declared_sha = _sha256(
            checkpoint["checkpoint_sha256"], label="checkpoint_sha256"
        )
        if declared_sha != expected_sha:
            raise ContinuousRecoveryDeviceError(
                "checkpoint differs from externally pinned SHA"
            )
        identity = {
            name: checkpoint[name]
            for name in expected_fields
            if name not in ("state_tensors", "checkpoint_sha256")
        }
        if _runtime.canonical_sha256(identity) != declared_sha:
            raise ContinuousRecoveryDeviceError(
                "checkpoint canonical identity SHA differs"
            )
        if checkpoint["schema_version"] != SCHEMA_VERSION:
            raise ContinuousRecoveryDeviceError("checkpoint schema differs")
        if checkpoint["kind"] != "action_ball_continuous_recovery_device_checkpoint_v1":
            raise ContinuousRecoveryDeviceError("checkpoint kind differs")
        if checkpoint["integration_status"] != INTEGRATION_STATUS:
            raise ContinuousRecoveryDeviceError(
                "checkpoint integration status differs"
            )
        if checkpoint["profile_sha256"] != profile.canonical_sha256:
            raise ContinuousRecoveryDeviceError("checkpoint profile differs")
        num_envs = _exact_int(
            checkpoint["num_envs"], label="checkpoint.num_envs", minimum=1
        )
        if checkpoint["dtype"] != str(dtype):
            raise ContinuousRecoveryDeviceError("checkpoint dtype differs")
        if tuple(checkpoint["component_names"]) != tuple(_runtime.COMPONENT_NAMES):
            raise ContinuousRecoveryDeviceError(
                "checkpoint component names differ"
            )
        owner = cls(
            profile=profile,
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )
        raw_tensors = checkpoint["state_tensors"]
        if not isinstance(raw_tensors, Mapping):
            raise ContinuousRecoveryDeviceError(
                "checkpoint state_tensors must be a mapping"
            )
        live = owner._state_tensors()
        if set(raw_tensors) != set(live):
            raise ContinuousRecoveryDeviceError(
                "checkpoint tensor names differ"
            )
        staged: dict[str, torch.Tensor] = {}
        for name, target in live.items():
            value = raw_tensors[name]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(target.shape)
                or value.dtype != target.dtype
            ):
                raise ContinuousRecoveryDeviceError(
                    f"checkpoint tensor {name} has wrong shape/dtype"
                )
            staged[name] = value.detach().to(device=owner.device).clone()
        if owner._tensor_manifest(staged) != checkpoint["tensor_manifest"]:
            raise ContinuousRecoveryDeviceError(
                "checkpoint tensor manifest differs"
            )
        actual_tensor_sha = owner._tensor_bytes_sha256(staged)
        if actual_tensor_sha != checkpoint["tensor_bytes_sha256"]:
            raise ContinuousRecoveryDeviceError(
                "checkpoint tensor bytes differ"
            )

        host = checkpoint["host_state"]
        if not isinstance(host, Mapping):
            raise ContinuousRecoveryDeviceError(
                "checkpoint host_state must be a mapping"
            )
        host_fields = {
            "current_keys",
            "window_owner_keys",
            "ready_owner_keys",
            "played",
            "reference_sha256",
            "pending_reference_sha256",
            "reset_generation",
            "scheduled_ordinal",
            "deadline_consumed",
            "last_deadline_tick",
            "last_reveal_tick",
            "payment_epoch_open",
        }
        if set(host) != host_fields:
            raise ContinuousRecoveryDeviceError(
                "checkpoint host-state fields differ"
            )
        for name in host_fields - {"payment_epoch_open"}:
            if not isinstance(host[name], list) or len(host[name]) != num_envs:
                raise ContinuousRecoveryDeviceError(
                    f"checkpoint host {name} row count differs"
                )

        def keys(rows: Sequence[object]) -> list[object | None]:
            return [
                None
                if value is None
                else _runtime.coerce_landing_outcome_shot_key(value)
                for value in rows
            ]

        current_keys = keys(host["current_keys"])
        window_keys = keys(host["window_owner_keys"])
        ready_keys = keys(host["ready_owner_keys"])
        played = list(host["played"])
        if any(type(value) is not bool for value in played):
            raise ContinuousRecoveryDeviceError(
                "checkpoint host played values must be exact bools"
            )
        payment_epoch_open = host["payment_epoch_open"]
        if type(payment_epoch_open) is not bool:
            raise ContinuousRecoveryDeviceError(
                "checkpoint payment_epoch_open must be exact bool"
            )
        reference_sha = [
            None if value is None else _sha256(value, label="reference SHA")
            for value in host["reference_sha256"]
        ]
        pending_sha = [
            None if value is None else _sha256(value, label="pending reference SHA")
            for value in host["pending_reference_sha256"]
        ]
        reset_generations = [
            None
            if value is None
            else _exact_int(value, label="host reset_generation", minimum=1)
            for value in host["reset_generation"]
        ]
        ordinals = [
            _exact_int(value, label="host scheduled_ordinal", minimum=-1)
            for value in host["scheduled_ordinal"]
        ]
        consumed = list(host["deadline_consumed"])
        if any(type(value) is not bool for value in consumed):
            raise ContinuousRecoveryDeviceError(
                "checkpoint deadline_consumed values must be exact bools"
            )
        deadlines = [
            None
            if value is None
            else _exact_int(value, label="host last_deadline_tick")
            for value in host["last_deadline_tick"]
        ]
        reveals = [
            None
            if value is None
            else _exact_int(value, label="host last_reveal_tick")
            for value in host["last_reveal_tick"]
        ]

        scratch = cls(
            profile=profile,
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )
        for name, tensor in scratch._state_tensors().items():
            tensor.copy_(staged[name])
        scratch._host_keys = current_keys
        scratch._host_window_owner_keys = window_keys
        scratch._host_ready_owner_keys = ready_keys
        scratch._host_played = played
        scratch._host_reference_sha = reference_sha
        scratch._host_pending_reference_sha = pending_sha
        scratch._host_reset_generation = reset_generations
        scratch._host_scheduled_ordinal = ordinals
        scratch._host_deadline_consumed = consumed
        scratch._host_last_deadline = deadlines
        scratch._host_last_reveal = reveals
        scratch._host_payment_epoch_open = payment_epoch_open
        scratch._validate_host_checkpoint_semantics()
        binding_counts = torch.stack(
            [
                mask.to(dtype=torch.int64).sum()
                for _, mask in scratch._host_device_checkpoint_mismatch_masks()
            ]
        ).to(device="cpu").tolist()
        if any(int(value) != 0 for value in binding_counts):
            raise ContinuousRecoveryDeviceError(
                "checkpoint host/device cross-binding failure"
            )
        invariant_counts = torch.stack(
            [
                (scratch._fault_bits != 0).to(dtype=torch.int64).sum(),
                *[
                    mask.to(dtype=torch.int64).sum()
                    for _, mask in scratch._invariant_masks()
                ],
            ]
        ).to(device="cpu").tolist()
        if any(int(value) != 0 for value in invariant_counts):
            raise ContinuousRecoveryDeviceError(
                "checkpoint collision/invariant failure"
            )

        owner = scratch
        owner._mutation_version = _exact_int(
            checkpoint["mutation_version"], label="checkpoint mutation_version"
        )
        owner._drain_sequence = _exact_int(
            checkpoint["drain_sequence"], label="checkpoint drain_sequence"
        )
        owner._last_drained_update = _exact_int(
            checkpoint["drain_update_index"],
            label="checkpoint drain_update_index",
        )
        owner._last_receipt = None
        owner._last_receipt_consumed = False
        owner._last_r07_global_drain_pair = None
        return owner

    restore_from_checkpoint = from_checkpoint

    def reset_true_boundary(self, env_ids: Sequence[int]) -> None:
        """Clear selected semantic rows only at a real episode boundary."""

        self._require_r07_business_mutation_allowed(label="true boundary")
        ids = self._ids(env_ids, label="reset_true_boundary.env_ids")
        if self._host_payment_epoch_open:
            raise ContinuousRecoveryDeviceError(
                "true boundary cannot clear an open reward-payment epoch"
            )
        # A readiness capability is a whole-batch current-epoch statement.
        # Any selected true reset revokes it; Motion must wait for the next
        # real post-physics publication instead of retaining mixed epochs.
        self._latest_motion_ready_projection = None
        index = self._index(ids)
        selected = self._mask_from_ids(ids)
        unsettled = selected & self._cache_pending & ~self._cache_paid
        self._fault_bits.bitwise_or_(
            unsettled.to(dtype=torch.int64) * FAULT_RESET_UNSETTLED
        )
        safe = selected & ~unsettled
        for tensor in (
            self._action_epoch_ready_instant,
            self._action_epoch_ready_live,
        ):
            tensor.bitwise_and_(~safe)
        self._action_epoch_ready_streak.masked_fill_(safe, 0)
        self._action_epoch_ready_reference_kind.masked_fill_(safe, 0)
        for tensor in (
            self._action_epoch_ready_reference_action_slot,
            self._action_epoch_ready_reference_action_uid,
            self._action_epoch_first_ready_source_step,
        ):
            tensor.masked_fill_(safe, -1)
        for field in fields(_row_identity.ActionEpochShotKey):
            getattr(
                self._action_epoch_ready_shot_key, field.name
            ).masked_fill_(safe, -1)
        for tensor in (
            self._sequence_active,
            self._current_key_valid,
            self._reward_owner_valid,
            self._reward_owner_played,
            self._window_owner_valid,
            self._deadline_ack_pending,
            self._ever_motion_active,
            self._playback_receipted,
            self._suffix_complete,
            self._reference_valid,
            self._pending_reference_valid,
            self._ready_authority,
            self._ready_instant,
            self._cache_pending,
            self._cache_viewed,
            self._cache_paid,
            self._cache_expected,
            self._cache_payment_required,
            self._cache_eligible,
            self._cache_facts_valid,
            self._cache_infrastructure_fault,
            self._cache_ready_instant,
            self._cache_ready_live,
        ):
            tensor.bitwise_and_(~safe)
        for tensor in (
            self._reset_generation,
            self._reference_owner_kind,
            self._ready_streak,
            self._cache_ready_streak,
            self._window_expected_count,
            self._window_eligible_count,
            self._window_payment_count,
        ):
            tensor.masked_fill_(safe, 0)
        for tensor in (
            self._episode_tick,
            self._source_step,
            self._phase_code,
            self._scheduled_ordinal,
            self._current_reveal_tick,
            self._current_deadline_tick,
            self._reward_deadline_tick,
            self._first_ready_tick,
            self._cache_source_step,
            self._cache_episode_tick,
            self._cache_phase_code,
            self._cache_age_tick,
            self._window_first_expected_age,
            self._window_last_expected_age,
            self._window_last_paid_age,
            self._action_epoch_ready_last_motion_cadence_tick,
            self._action_epoch_ready_last_reset_generation,
        ):
            tensor.masked_fill_(safe, -1)
        for tensor in (
            self._current_key_sha,
            self._reward_owner_sha,
            self._window_owner_sha,
            self._reference_owner_sha,
            self._reference_sha,
            self._reference_root_position,
            self._reference_root_orientation,
            self._reference_joint_position,
            self._reference_body_position,
            self._reference_body_orientation,
            self._station_anchor_xy,
            self._pending_reference_sha,
            self._pending_reference_key_sha,
            self._pending_reference_root_position,
            self._pending_reference_root_orientation,
            self._pending_reference_joint_position,
            self._pending_reference_body_position,
            self._pending_reference_body_orientation,
            self._pending_station_anchor_xy,
            self._cache_owner_sha,
            self._cache_reference_owner_sha,
            self._cache_errors,
            self._cache_scores,
            self._cache_raw_score,
            self._cache_weighted_reward,
        ):
            _masked_zero_(tensor, safe)
        self._motion_active.bitwise_and_(~safe)
        self._reference_active.bitwise_and_(~safe)

        staged_keys = list(self._host_keys)
        staged_window_owners = list(self._host_window_owner_keys)
        staged_ready = list(self._host_ready_owner_keys)
        staged_played = list(self._host_played)
        staged_ref = list(self._host_reference_sha)
        staged_pending = list(self._host_pending_reference_sha)
        staged_generation = list(self._host_reset_generation)
        staged_ordinal = list(self._host_scheduled_ordinal)
        staged_consumed = list(self._host_deadline_consumed)
        staged_deadline = list(self._host_last_deadline)
        staged_reveal = list(self._host_last_reveal)
        for env_id in ids:
            staged_keys[env_id] = None
            staged_window_owners[env_id] = None
            staged_ready[env_id] = None
            staged_played[env_id] = False
            staged_ref[env_id] = None
            staged_pending[env_id] = None
            staged_generation[env_id] = None
            staged_ordinal[env_id] = -1
            staged_consumed[env_id] = False
            staged_deadline[env_id] = None
            staged_reveal[env_id] = None
        self._host_keys = staged_keys
        self._host_window_owner_keys = staged_window_owners
        self._host_ready_owner_keys = staged_ready
        self._host_played = staged_played
        self._host_reference_sha = staged_ref
        self._host_pending_reference_sha = staged_pending
        self._host_reset_generation = staged_generation
        self._host_scheduled_ordinal = staged_ordinal
        self._host_deadline_consumed = staged_consumed
        self._host_last_deadline = staged_deadline
        self._host_last_reveal = staged_reveal
        self._advance_mutation_version()


__all__ = (
    "SCHEMA_VERSION",
    "INTEGRATION_STATUS",
    "RUNTIME_WIRING_CONNECTED",
    "CUDA_PROFILED",
    "FORMAL_EXACT_RESUME_INTEGRATED",
    "LAUNCH_AUTHORIZED",
    "DIAGNOSTIC_UNAUTHORIZED",
    "DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON",
    "DIAGNOSTIC_N2_SUPPORT_FORCE_Z_N",
    "DIAGNOSTIC_N2_MIN_ROOT_HEIGHT_M",
    "DIAGNOSTIC_N2_MAX_ROOT_TILT_RAD",
    "DIAGNOSTIC_N2_COMPONENT_WEIGHTS",
    "DIAGNOSTIC_N2_COMPONENT_SCALES",
    "DIAGNOSTIC_N2_READY_TOLERANCES",
    "R07_EPOCH_FAULT_STALE_SOURCE_STEP",
    "R07_EPOCH_FAULT_INVALID_PLANT_FACT",
    "R07_EPOCH_FAULT_INVALID_REFERENCE",
    "R07_EPOCH_FAULT_EPOCH_IDENTITY",
    "R07_EPOCH_FACT_PRESENT",
    "R07_EPOCH_FACT_NUMERICALLY_VALID",
    "R07_EPOCH_FACT_VALUE_COUNT",
    "POLICY_RATE_HZ",
    "RECOVERY_START_AGE_TICK",
    "RECOVERY_END_AGE_TICK",
    "RECOVERY_SAMPLE_COUNT",
    "RECOVERY_REWARD_CONSUMER",
    "FULL_MDP_REWARD_CONSUMERS",
    "PHASE_PRE_REVEAL_HIDDEN",
    "PHASE_ACTIVE_OPPORTUNITY",
    "PHASE_POST_DEADLINE_SUFFIX",
    "PHASE_RECOVERY_HIDDEN",
    "PHASE_READY_HOLD",
    "PHASE_RECOVERY_UNAVAILABLE",
    "PHASE_INFRASTRUCTURE_INVALID",
    "FAULT_ACTION_EPOCH_MOTION_CHRONOLOGY",
    "FAULT_MOTION_CADENCE_SUCCESSOR_OVERFLOW",
    "FAULTS",
    "ContinuousRecoveryDeviceError",
    "ContinuousRecoveryConstructionHold",
    "ContinuousRecoveryFullMdpRewardPublication",
    "ContinuousRecoveryFullMdpRewardPaymentVerdict",
    "ContinuousRecoveryFullMdpRewardCloseReceipt",
    "ContinuousRecoveryMotionReadyProjection",
    "ContinuousRecoveryMotionReadyView",
    "ContinuousRecoveryObservationState",
    "ContinuousRecoveryFullMdpPreRewardView",
    "DeviceLandingOutcomeShotKey",
    "DeviceContinuousRecoveryReference",
    "DeviceContinuousRecoveryPlantFacts",
    "DeviceContinuousRecoveryCommandProjection",
    "DeviceContinuousRecoveryDoneTermProjection",
    "DeviceContinuousRecoveryPaymentIdentity",
    "SharedContinuousRecoveryRewardView",
    "ContinuousRecoveryWindowLedgerRow",
    "ContinuousRecoveryBoundaryReceipt",
    "DiagnosticN2ContinuousRecoveryBundle",
    "DiagnosticN2ContinuousRecoveryPlantFactAdapter",
    "R07EpochDirectRewardFacts",
    "ContinuousRecoveryDeviceCoordinator",
    "construct_action_ball_full_mdp_diagnostic_n2_recovery_owner",
)
