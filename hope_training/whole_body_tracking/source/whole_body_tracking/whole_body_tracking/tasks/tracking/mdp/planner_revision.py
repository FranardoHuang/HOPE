"""Dependency-light phase governor for revisions of one immutable planner task.

The planner may improve its estimate while a swing is active, but a revision is not a new
question.  This module therefore keeps two facts separate:

* :class:`LatentTaskTruth` identifies the immutable question and strike phase.
* :class:`PlannerTaskRevision` is one atomic actor-visible estimate of position, velocity,
  signed racket normal, and time-to-strike (TTS).

Accepted revisions may move the deadline and target within a SHA-bound training envelope.  The
reference phase is governed rather than assigned from TTS directly: it never moves backwards,
its rate and acceleration are bounded, a later deadline can only hold or slow the next phase
step, and an earlier deadline that cannot be reached inside the envelope is rejected.  Rejection
returns the exact input ledger, so callers cannot accidentally commit half of the target tuple.

There are deliberately no Isaac, torch, ROS, or wall-clock imports.  Training and deployment can
feed the same recorded revision trace through these pure functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence


PHASE_GOVERNOR_CONTRACT_VERSION = "phase_governor_v1"
PHASE_GOVERNOR_PROFILE_SCHEMA_VERSION = 1
PLANNER_TASK_REVISION_SCHEMA_VERSION = 1
INITIAL_TTS_MIXTURE_CONTRACT_VERSION = "initial_tts_mixture_v1"


def _plain_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a plain int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    if (
        len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 string")
    return value


def _vector3(value: object, *, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return tuple(
        _finite_float(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _normal_angle(a: Sequence[float], b: Sequence[float]) -> float:
    norm_a = math.sqrt(sum(component * component for component in a))
    norm_b = math.sqrt(sum(component * component for component in b))
    dot = sum(left * right for left, right in zip(a, b, strict=True)) / (norm_a * norm_b)
    return math.acos(max(-1.0, min(1.0, dot)))


def canonical_profile_bytes(document: Mapping[str, Any]) -> bytes:
    """Return the single canonical byte representation used by ``profile_sha256``."""

    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class InitialTtsMixtureComponent:
    """One explicitly weighted preparation-time stratum.

    ``lo_s == hi_s`` is intentionally allowed: the deployment baseline needs a
    real point mass at exactly 0.5 s rather than hoping a continuous uniform draw
    happens to land there.  The overall model envelope remains non-degenerate.
    """

    name: str
    lo_s: float
    hi_s: float
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ValueError("initial TTS mixture component name must be a non-empty trimmed string")
        lo = _finite_float(self.lo_s, name=f"{self.name}.lo_s", minimum=0.0)
        hi = _finite_float(self.hi_s, name=f"{self.name}.hi_s", minimum=lo)
        if hi < lo:
            raise ValueError(f"{self.name}.hi_s must be >= lo_s")
        _finite_float(self.weight, name=f"{self.name}.weight", minimum=math.ulp(0.0))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> InitialTtsMixtureComponent:
        if not isinstance(value, Mapping):
            raise ValueError("initial TTS mixture component must be a mapping")
        expected = {"name", "range_s", "weight"}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing:
            raise ValueError(f"initial TTS mixture component is missing fields: {missing}")
        if unknown:
            raise ValueError(f"initial TTS mixture component contains unknown fields: {unknown}")
        interval = value["range_s"]
        if isinstance(interval, (str, bytes)) or not isinstance(interval, Sequence) or len(interval) != 2:
            raise ValueError("initial TTS mixture component range_s must contain two numbers")
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            lo_s=interval[0],  # type: ignore[arg-type]
            hi_s=interval[1],  # type: ignore[arg-type]
            weight=value["weight"],  # type: ignore[arg-type]
        )

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "range_s": [float(self.lo_s), float(self.hi_s)],
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class InitialTtsMixture:
    """Content-stable preparation-time distribution used only by training.

    Runtime does not sample this distribution; it enforces the enclosing
    checkpoint-bound ``initial_tts_range_s``.  Keeping the mixture in the hard
    training contract prevents a broad envelope from hiding a narrow or absent
    0.5-second curriculum.
    """

    components: tuple[InitialTtsMixtureComponent, ...]
    contract_version: str = INITIAL_TTS_MIXTURE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != INITIAL_TTS_MIXTURE_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported initial TTS mixture contract {self.contract_version!r}"
            )
        if not self.components:
            raise ValueError("initial TTS mixture requires at least one component")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("initial TTS mixture component names must be unique")
        total = math.fsum(float(component.weight) for component in self.components)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"initial TTS mixture weights must sum to 1, got {total}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> InitialTtsMixture:
        if not isinstance(value, Mapping):
            raise ValueError("initial TTS mixture must be a mapping")
        expected = {"contract_version", "components"}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing:
            raise ValueError(f"initial TTS mixture is missing fields: {missing}")
        if unknown:
            raise ValueError(f"initial TTS mixture contains unknown fields: {unknown}")
        raw_components = value["components"]
        if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, Sequence):
            raise ValueError("initial TTS mixture components must be a sequence")
        return cls(
            components=tuple(
                InitialTtsMixtureComponent.from_mapping(component)  # type: ignore[arg-type]
                for component in raw_components
            ),
            contract_version=value["contract_version"],  # type: ignore[arg-type]
        )

    @property
    def support_range_s(self) -> tuple[float, float]:
        return (
            min(float(component.lo_s) for component in self.components),
            max(float(component.hi_s) for component in self.components),
        )

    def validate_support(self, *, lo_s: float, hi_s: float) -> None:
        expected = (float(lo_s), float(hi_s))
        if self.support_range_s != expected:
            raise ValueError(
                "initial TTS mixture support must exactly equal initial_tts_range_s: "
                f"mixture={self.support_range_s}, range={expected}"
            )

    def document(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "components": [component.document() for component in self.components],
        }


@dataclass(frozen=True)
class PhaseGovernorProfile:
    """Versioned envelope shared by training, export, and deployment."""

    policy_dt_s: float
    min_tts_s: float
    max_tts_s: float
    max_phase_rate_per_s: float
    max_phase_acceleration_per_s2: float
    max_deadline_revision_delta_s: float
    max_position_revision_delta_m: float
    max_velocity_revision_delta_mps: float
    max_normal_revision_delta_rad: float
    normal_unit_tolerance: float = 1e-4
    early_deadline_tolerance_s: float = 1e-9
    contract_version: str = PHASE_GOVERNOR_CONTRACT_VERSION
    schema_version: int = PHASE_GOVERNOR_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _plain_int(self.schema_version, name="schema_version", minimum=1)
        if self.contract_version != PHASE_GOVERNOR_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported phase governor contract {self.contract_version!r}; "
                f"expected {PHASE_GOVERNOR_CONTRACT_VERSION!r}"
            )
        if self.schema_version != PHASE_GOVERNOR_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported phase governor profile schema {self.schema_version}; "
                f"expected {PHASE_GOVERNOR_PROFILE_SCHEMA_VERSION}"
            )
        _finite_float(self.policy_dt_s, name="policy_dt_s", minimum=math.ulp(0.0))
        minimum_tts = _finite_float(self.min_tts_s, name="min_tts_s", minimum=math.ulp(0.0))
        maximum_tts = _finite_float(self.max_tts_s, name="max_tts_s", minimum=minimum_tts)
        if maximum_tts <= minimum_tts:
            raise ValueError("max_tts_s must be greater than min_tts_s")
        _finite_float(
            self.max_phase_rate_per_s,
            name="max_phase_rate_per_s",
            minimum=math.ulp(0.0),
        )
        _finite_float(
            self.max_phase_acceleration_per_s2,
            name="max_phase_acceleration_per_s2",
            minimum=math.ulp(0.0),
        )
        _finite_float(
            self.max_deadline_revision_delta_s,
            name="max_deadline_revision_delta_s",
            minimum=0.0,
        )
        _finite_float(
            self.max_position_revision_delta_m,
            name="max_position_revision_delta_m",
            minimum=0.0,
        )
        _finite_float(
            self.max_velocity_revision_delta_mps,
            name="max_velocity_revision_delta_mps",
            minimum=0.0,
        )
        normal_delta = _finite_float(
            self.max_normal_revision_delta_rad,
            name="max_normal_revision_delta_rad",
            minimum=0.0,
        )
        if normal_delta > math.pi:
            raise ValueError("max_normal_revision_delta_rad must be <= pi")
        tolerance = _finite_float(
            self.normal_unit_tolerance,
            name="normal_unit_tolerance",
            minimum=0.0,
        )
        if tolerance >= 1.0:
            raise ValueError("normal_unit_tolerance must be < 1")
        _finite_float(
            self.early_deadline_tolerance_s,
            name="early_deadline_tolerance_s",
            minimum=0.0,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PhaseGovernorProfile:
        """Load the complete profile without accepting silent defaults or extensions."""

        if not isinstance(value, Mapping):
            raise ValueError("phase governor profile must be a mapping")
        fields = {
            "policy_dt_s",
            "min_tts_s",
            "max_tts_s",
            "max_phase_rate_per_s",
            "max_phase_acceleration_per_s2",
            "max_deadline_revision_delta_s",
            "max_position_revision_delta_m",
            "max_velocity_revision_delta_mps",
            "max_normal_revision_delta_rad",
            "normal_unit_tolerance",
            "early_deadline_tolerance_s",
            "contract_version",
            "schema_version",
        }
        missing = sorted(fields - set(value))
        unknown = sorted(set(value) - fields)
        if missing:
            raise ValueError(f"phase governor profile is missing fields: {missing}")
        if unknown:
            raise ValueError(f"phase governor profile contains unknown fields: {unknown}")
        return cls(**dict(value))  # type: ignore[arg-type]

    def document(self) -> dict[str, Any]:
        """Return the complete executable profile; no defaults are omitted."""

        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return hashlib.sha256(canonical_profile_bytes(self.document())).hexdigest()

    def hard_contract(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "schema_version": self.schema_version,
            "profile_sha256": self.profile_sha256,
            "profile": self.document(),
        }


@dataclass(frozen=True)
class LatentTaskTruth:
    """Immutable task identity and reference strike phase (normally normalized to 1)."""

    control_epoch: int
    task_id: int
    truth_sha256: str
    strike_phase: float = 1.0

    def __post_init__(self) -> None:
        _plain_int(self.control_epoch, name="control_epoch")
        _plain_int(self.task_id, name="task_id", minimum=1)
        _sha256(self.truth_sha256, name="truth_sha256")
        _finite_float(self.strike_phase, name="strike_phase", minimum=math.ulp(0.0))


@dataclass(frozen=True)
class PlannerTaskRevision:
    """One all-or-nothing visible estimate for the active latent task."""

    control_epoch: int
    task_id: int
    task_revision: int
    command_sequence: int
    source_monotonic_s: float
    truth_sha256: str
    target_position_m: tuple[float, float, float]
    target_velocity_mps: tuple[float, float, float]
    target_normal: tuple[float, float, float]
    desired_tts_s: float
    schema_version: int = PLANNER_TASK_REVISION_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PlannerTaskRevision:
        """Parse a strict wire-neutral mapping, rejecting partial or extension tuples."""

        if not isinstance(value, Mapping):
            raise ValueError("planner task revision must be a mapping")
        allowed = {
            "schema_version",
            "control_epoch",
            "task_id",
            "task_revision",
            "command_sequence",
            "source_monotonic_s",
            "truth_sha256",
            "target_position_m",
            "target_velocity_mps",
            "target_normal",
            "desired_tts_s",
        }
        missing = sorted(allowed - set(value))
        unknown = sorted(set(value) - allowed)
        if missing:
            raise ValueError(f"planner task revision is missing fields: {missing}")
        if unknown:
            raise ValueError(f"planner task revision contains unknown fields: {unknown}")
        return cls(
            schema_version=_plain_int(value["schema_version"], name="schema_version"),
            control_epoch=_plain_int(value["control_epoch"], name="control_epoch"),
            task_id=_plain_int(value["task_id"], name="task_id", minimum=1),
            task_revision=_plain_int(
                value["task_revision"], name="task_revision", minimum=1
            ),
            command_sequence=_plain_int(
                value["command_sequence"], name="command_sequence", minimum=1
            ),
            source_monotonic_s=_finite_float(
                value["source_monotonic_s"], name="source_monotonic_s", minimum=0.0
            ),
            truth_sha256=_sha256(value["truth_sha256"], name="truth_sha256"),
            target_position_m=_vector3(value["target_position_m"], name="target_position_m"),
            target_velocity_mps=_vector3(
                value["target_velocity_mps"], name="target_velocity_mps"
            ),
            target_normal=_vector3(value["target_normal"], name="target_normal"),
            desired_tts_s=_finite_float(
                value["desired_tts_s"], name="desired_tts_s", minimum=math.ulp(0.0)
            ),
        )


@dataclass(frozen=True)
class ActivePhaseState:
    truth: LatentTaskTruth
    phase: float
    phase_rate_per_s: float
    local_monotonic_s: float
    deadline_local_s: float
    # A latest-value transport may omit any intermediate revision.  Keep the
    # trained envelope relative to the tuple that began this consumer task;
    # otherwise the allowance is transport-order dependent and can ratchet.
    baseline_deadline_local_s: float
    baseline_revision: PlannerTaskRevision
    visible_revision: PlannerTaskRevision
    slow_only_next_step: bool = False


@dataclass(frozen=True)
class PhaseGovernorLedger:
    """Immutable ledger; callers commit by replacing their prior value with a decision value."""

    active: ActivePhaseState | None = None
    last_control_epoch: int = -1
    last_task_id: int = -1
    last_command_sequence: int = -1
    last_source_monotonic_s: float = -1.0


@dataclass(frozen=True)
class RevisionDecision:
    accepted: bool
    reason: str
    ledger: PhaseGovernorLedger


def _canonicalize_revision_tts(
    profile: PhaseGovernorProfile, desired_tts_s: float
) -> float:
    """Validate one deadline against the numeric envelope and snap its grid edges.

    Training transports this field as float32 while the reference governor and
    deployment consumer use doubles.  Repeatedly subtracting ``policy_dt_s``
    can therefore put the final legal value a few ulps below ``min_tts_s``.
    The profile-bound tolerance accepts only that narrow numeric band; snapping
    the accepted value makes all three consumers commit the same canonical
    deadline instead of carrying representation-specific drift forward.
    """

    tts = _finite_float(desired_tts_s, name="desired_tts_s")
    tolerance = profile.early_deadline_tolerance_s
    if tts + tolerance < profile.min_tts_s or tts - tolerance > profile.max_tts_s:
        raise ValueError(
            f"desired_tts_s {tts} is outside [{profile.min_tts_s}, "
            f"{profile.max_tts_s}] with tolerance {tolerance}"
        )
    if abs(tts - profile.min_tts_s) <= tolerance:
        return profile.min_tts_s
    if abs(tts - profile.max_tts_s) <= tolerance:
        return profile.max_tts_s
    return tts


def _validate_revision(
    profile: PhaseGovernorProfile, revision: PlannerTaskRevision
) -> float:
    _plain_int(revision.schema_version, name="schema_version", minimum=1)
    if revision.schema_version != PLANNER_TASK_REVISION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported planner revision schema {revision.schema_version}; "
            f"expected {PLANNER_TASK_REVISION_SCHEMA_VERSION}"
        )
    _plain_int(revision.control_epoch, name="control_epoch")
    _plain_int(revision.task_id, name="task_id", minimum=1)
    _plain_int(revision.task_revision, name="task_revision", minimum=1)
    _plain_int(revision.command_sequence, name="command_sequence", minimum=1)
    _finite_float(revision.source_monotonic_s, name="source_monotonic_s", minimum=0.0)
    _sha256(revision.truth_sha256, name="truth_sha256")
    _vector3(revision.target_position_m, name="target_position_m")
    _vector3(revision.target_velocity_mps, name="target_velocity_mps")
    normal = _vector3(revision.target_normal, name="target_normal")
    normal_norm = math.sqrt(sum(component * component for component in normal))
    if abs(normal_norm - 1.0) > profile.normal_unit_tolerance:
        raise ValueError(
            f"target_normal must be unit length within {profile.normal_unit_tolerance}, "
            f"got norm {normal_norm}"
        )
    return _canonicalize_revision_tts(profile, revision.desired_tts_s)


def _minimum_finish_time(
    distance: float,
    initial_rate: float,
    maximum_rate: float,
    maximum_acceleration: float,
) -> float:
    """Continuous lower bound with no terminal-rate requirement."""

    if distance <= 0.0:
        return 0.0
    rate = min(maximum_rate, max(0.0, initial_rate))
    accelerate_time = max(0.0, (maximum_rate - rate) / maximum_acceleration)
    accelerate_distance = (
        rate * accelerate_time + 0.5 * maximum_acceleration * accelerate_time**2
    )
    if distance <= accelerate_distance:
        return (-rate + math.sqrt(rate * rate + 2.0 * maximum_acceleration * distance)) / (
            maximum_acceleration
        )
    return accelerate_time + (distance - accelerate_distance) / maximum_rate


def begin_task(
    profile: PhaseGovernorProfile,
    ledger: PhaseGovernorLedger,
    truth: LatentTaskTruth,
    revision: PlannerTaskRevision,
    *,
    local_monotonic_s: float,
) -> RevisionDecision:
    """Begin a strictly newer task, or return the exact old ledger on rejection."""

    try:
        if ledger.active is not None:
            raise ValueError("a task is already active")
        local_now = _finite_float(
            local_monotonic_s, name="local_monotonic_s", minimum=0.0
        )
        canonical_tts = _validate_revision(profile, revision)
        revision = replace(revision, desired_tts_s=canonical_tts)
        if revision.control_epoch != truth.control_epoch or revision.task_id != truth.task_id:
            raise ValueError("revision (control_epoch, task_id) does not match latent task")
        if revision.truth_sha256 != truth.truth_sha256:
            raise ValueError("revision truth_sha256 does not match latent task")
        if revision.task_revision != 1:
            raise ValueError("a new task must begin at task_revision=1")
        if truth.control_epoch < ledger.last_control_epoch:
            raise ValueError("control_epoch cannot move backwards")
        same_epoch = truth.control_epoch == ledger.last_control_epoch
        if same_epoch:
            if truth.task_id <= ledger.last_task_id:
                raise ValueError("task_id must strictly increase within one control_epoch")
            if revision.command_sequence <= ledger.last_command_sequence:
                raise ValueError("command_sequence must strictly increase within one control_epoch")
            if revision.source_monotonic_s <= ledger.last_source_monotonic_s:
                raise ValueError("source_monotonic_s must strictly increase within one control_epoch")
        minimum_tts = _minimum_finish_time(
            truth.strike_phase,
            0.0,
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        if revision.desired_tts_s + profile.early_deadline_tolerance_s < minimum_tts:
            raise ValueError(
                f"initial deadline is outside the reachable phase envelope: "
                f"tts={revision.desired_tts_s}, minimum={minimum_tts}"
            )
    except ValueError as exc:
        return RevisionDecision(False, str(exc), ledger)

    active = ActivePhaseState(
        truth=truth,
        phase=0.0,
        phase_rate_per_s=0.0,
        local_monotonic_s=local_now,
        deadline_local_s=local_now + revision.desired_tts_s,
        baseline_deadline_local_s=local_now + revision.desired_tts_s,
        baseline_revision=revision,
        visible_revision=revision,
    )
    return RevisionDecision(
        True,
        "accepted",
        replace(
            ledger,
            active=active,
            last_control_epoch=truth.control_epoch,
            last_task_id=truth.task_id,
            last_command_sequence=revision.command_sequence,
            last_source_monotonic_s=revision.source_monotonic_s,
        ),
    )


def revise_task(
    profile: PhaseGovernorProfile,
    ledger: PhaseGovernorLedger,
    revision: PlannerTaskRevision,
) -> RevisionDecision:
    """Atomically revise the active task without changing its latent truth or phase."""

    active = ledger.active
    try:
        if active is None:
            raise ValueError("no task is active")
        if active.phase >= active.truth.strike_phase:
            raise ValueError("post-contact revisions are closed")
        canonical_tts = _validate_revision(profile, revision)
        revision = replace(revision, desired_tts_s=canonical_tts)
        previous = active.visible_revision
        baseline = active.baseline_revision
        if (
            revision.control_epoch != active.truth.control_epoch
            or revision.task_id != active.truth.task_id
        ):
            raise ValueError("a revision cannot replace the active task")
        if revision.truth_sha256 != active.truth.truth_sha256:
            raise ValueError("a revision cannot replace latent task truth")
        if revision.task_revision <= previous.task_revision:
            raise ValueError("task_revision must strictly increase")
        if revision.command_sequence <= ledger.last_command_sequence:
            raise ValueError("command_sequence must strictly increase")
        if revision.source_monotonic_s <= ledger.last_source_monotonic_s:
            raise ValueError("source_monotonic_s must strictly increase")
        if (
            _distance(revision.target_position_m, baseline.target_position_m)
            > profile.max_position_revision_delta_m
        ):
            raise ValueError("target position revision exceeds the trained envelope")
        if (
            _distance(revision.target_velocity_mps, baseline.target_velocity_mps)
            > profile.max_velocity_revision_delta_mps
        ):
            raise ValueError("target velocity revision exceeds the trained envelope")
        if (
            _normal_angle(revision.target_normal, baseline.target_normal)
            > profile.max_normal_revision_delta_rad
        ):
            raise ValueError("target normal revision exceeds the trained envelope")

        proposed_deadline = active.local_monotonic_s + revision.desired_tts_s
        baseline_deadline_delta = (
            proposed_deadline - active.baseline_deadline_local_s
        )
        if abs(baseline_deadline_delta) > profile.max_deadline_revision_delta_s:
            raise ValueError("deadline revision exceeds the trained envelope")
        deadline_delta = proposed_deadline - active.deadline_local_s
        remaining_phase = max(0.0, active.truth.strike_phase - active.phase)
        minimum_tts = _minimum_finish_time(
            remaining_phase,
            active.phase_rate_per_s,
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        if revision.desired_tts_s + profile.early_deadline_tolerance_s < minimum_tts:
            raise ValueError(
                f"earlier deadline is outside the reachable phase envelope: "
                f"tts={revision.desired_tts_s}, minimum={minimum_tts}"
            )
    except ValueError as exc:
        return RevisionDecision(False, str(exc), ledger)

    revised = replace(
        active,
        deadline_local_s=proposed_deadline,
        visible_revision=revision,
        slow_only_next_step=deadline_delta > profile.early_deadline_tolerance_s,
    )
    return RevisionDecision(
        True,
        "accepted",
        replace(
            ledger,
            active=revised,
            last_command_sequence=revision.command_sequence,
            last_source_monotonic_s=revision.source_monotonic_s,
        ),
    )


def advance_phase(
    profile: PhaseGovernorProfile,
    ledger: PhaseGovernorLedger,
) -> PhaseGovernorLedger:
    """Advance one fixed policy tick with monotonic phase and bounded rate/acceleration."""

    active = ledger.active
    if active is None:
        return ledger
    if active.phase >= active.truth.strike_phase:
        # Phase is clamped at the strike, but the rate state still decelerates through the same
        # envelope.  Snapping the rate to zero here would hide an acceleration discontinuity from
        # a deployment consumer.
        decelerated_rate = max(
            0.0,
            active.phase_rate_per_s
            - profile.max_phase_acceleration_per_s2 * profile.policy_dt_s,
        )
        return replace(
            ledger,
            active=replace(
                active,
                phase_rate_per_s=decelerated_rate,
                local_monotonic_s=active.local_monotonic_s + profile.policy_dt_s,
                slow_only_next_step=False,
            ),
        )

    dt = profile.policy_dt_s
    new_time = active.local_monotonic_s + dt
    remaining_phase = active.truth.strike_phase - active.phase
    remaining_time_after_step = max(active.deadline_local_s - new_time, 0.0)
    if remaining_time_after_step <= profile.early_deadline_tolerance_s:
        requested_rate = profile.max_phase_rate_per_s
    else:
        requested_rate = min(
            profile.max_phase_rate_per_s,
            remaining_phase / remaining_time_after_step,
        )
        earliest = _minimum_finish_time(
            remaining_phase,
            active.phase_rate_per_s,
            profile.max_phase_rate_per_s,
            profile.max_phase_acceleration_per_s2,
        )
        if remaining_time_after_step <= earliest + dt:
            requested_rate = profile.max_phase_rate_per_s
    if active.slow_only_next_step:
        requested_rate = min(requested_rate, active.phase_rate_per_s)

    maximum_rate_delta = profile.max_phase_acceleration_per_s2 * dt
    rate_delta = max(
        -maximum_rate_delta,
        min(maximum_rate_delta, requested_rate - active.phase_rate_per_s),
    )
    new_rate = min(
        profile.max_phase_rate_per_s,
        max(0.0, active.phase_rate_per_s + rate_delta),
    )
    phase_delta = max(0.0, 0.5 * (active.phase_rate_per_s + new_rate) * dt)
    new_phase = min(active.truth.strike_phase, active.phase + phase_delta)
    if remaining_time_after_step > profile.early_deadline_tolerance_s:
        # Preserve one complete actor interval before contact.  Without this
        # cap the phase can land exactly on the strike while the immutable
        # task still has one policy tick (normally 0.02 s) remaining, making
        # the final real-time target revision impossible to consume.
        next_rate = min(
            profile.max_phase_rate_per_s,
            new_rate + profile.max_phase_acceleration_per_s2 * dt,
        )
        reserved_distance = 0.5 * (new_rate + next_rate) * dt
        precontact_cap = max(
            active.phase,
            active.truth.strike_phase - reserved_distance,
        )
        new_phase = min(new_phase, precontact_cap)

    return replace(
        ledger,
        active=replace(
            active,
            phase=new_phase,
            phase_rate_per_s=new_rate,
            local_monotonic_s=new_time,
            slow_only_next_step=False,
        ),
    )


def complete_task(ledger: PhaseGovernorLedger) -> PhaseGovernorLedger:
    """Explicitly rearm after the caller has completed/aborted the active physical ball."""

    if ledger.active is None:
        raise ValueError("no task is active")
    return replace(ledger, active=None)
