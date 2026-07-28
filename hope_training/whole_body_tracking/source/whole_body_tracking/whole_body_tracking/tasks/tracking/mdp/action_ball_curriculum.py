"""Auditable asymmetric curriculum for action-conditioned ball-first training.

The curriculum does not use the sampler's old seven grouped widths.  It owns a
versioned catalog of 32 signed *arms*.  Positive and negative support can
therefore grow independently, including time-to-contact and incoming/spin
direction tangent coordinates.

Online rolling observations only schedule which arm should be evaluated next.
They never authorize a frontier change.  Every authoritative change requires
one frozen canary followed by a disjoint frozen heldout window from the same
policy snapshot and exact domain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple


ARM_KEYS = (
    "time_to_contact_lower",
    "time_to_contact_upper",
    "contact_x_lower",
    "contact_x_upper",
    "contact_y_lower",
    "contact_y_upper",
    "contact_z_lower",
    "contact_z_upper",
    "incoming_speed_lower",
    "incoming_speed_upper",
    "spin_magnitude_lower",
    "spin_magnitude_upper",
    "base_spawn_x_lower",
    "base_spawn_x_upper",
    "base_spawn_y_lower",
    "base_spawn_y_upper",
    "base_travel_x_lower",
    "base_travel_x_upper",
    "base_travel_y_lower",
    "base_travel_y_upper",
    "landing_aim_x_lower",
    "landing_aim_x_upper",
    "landing_aim_y_lower",
    "landing_aim_y_upper",
    "incoming_direction_u_neg",
    "incoming_direction_u_pos",
    "incoming_direction_v_neg",
    "incoming_direction_v_pos",
    "spin_direction_u_neg",
    "spin_direction_u_pos",
    "spin_direction_v_neg",
    "spin_direction_v_pos",
)
BASE_TRAVEL_ARMS = tuple(
    arm for arm in ARM_KEYS if arm.startswith("base_travel_")
)
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
JOINT_RHOS = LEVELS
MOBILITIES = ("no_move", "move")
EVIDENCE_ROLES = (
    "scheduler",
    "frozen_canary",
    "frozen_heldout",
)
STATE_SCHEMA_VERSION = 7
EVIDENCE_SCHEMA_VERSION = 3
INT64_MAX = (1 << 63) - 1
CANARY_MIN = 256
HELDOUT_MIN = 768
_ZERO_SHA = "0" * 64


def _canonical_sha256(document: object) -> str:
    raw = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


ARM_CATALOG_DOCUMENT = {
    # This document intentionally duplicates action_ball_sampling's
    # dependency-light contract.  A focused cross-module test prevents drift.
    "schema_version": 3,
    "arm_keys": list(ARM_KEYS),
}
ARM_CATALOG_SHA256 = _canonical_sha256(ARM_CATALOG_DOCUMENT)


def _plain_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a plain integer")
    if value < minimum or value > INT64_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {INT64_MAX}]")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a plain finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _exact_keys(
    value: object,
    expected: Sequence[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ValueError(
            f"{name} has invalid keys "
            f"(missing={sorted(expected_set - actual)}, "
            f"unknown={sorted(actual - expected_set)})"
        )
    return value


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


@dataclass(frozen=True, order=True)
class ActionProfileKey:
    action_uid: int
    profile_sha256: str
    mobility: str

    def __post_init__(self) -> None:
        _plain_int(self.action_uid, name="action_uid", minimum=1)
        if self.action_uid > (1 << 53) - 1:
            raise ValueError("action_uid must be in [1, 2**53-1]")
        _sha256(self.profile_sha256, name="profile_sha256")
        if self.mobility not in MOBILITIES:
            raise ValueError(f"mobility must be one of {MOBILITIES!r}")

    @property
    def enabled_arms(self) -> Tuple[str, ...]:
        if self.mobility == "no_move":
            return tuple(
                arm for arm in ARM_KEYS if arm not in BASE_TRAVEL_ARMS
            )
        return ARM_KEYS

    def as_dict(self) -> Dict[str, object]:
        return {
            "action_uid": self.action_uid,
            "profile_sha256": self.profile_sha256,
            "mobility": self.mobility,
        }


@dataclass(frozen=True)
class BallOutcomeLedger:
    P: int
    A: int
    I: int
    S: int
    C: int
    L: int
    F: int
    U_table: int
    U_fall: int
    U_collision: int
    X: int
    NB: int = 0
    NB_F: int = 0

    def __post_init__(self) -> None:
        for field in self.as_dict():
            _plain_int(getattr(self, field), name=field)
        if not self.P >= self.A >= self.I >= self.S >= self.C:
            raise ValueError("ledger must satisfy P >= A >= I >= S >= C")
        terminal = (
            self.L
            + self.F
            + self.U_table
            + self.U_fall
            + self.U_collision
        )
        if terminal != self.C:
            raise ValueError(
                "closed outcomes must conserve exactly: "
                "C = L + F + U_table + U_fall + U_collision"
            )
        if self.X > self.P:
            raise ValueError("X must not exceed proposed attempts P")
        if self.NB > self.L + self.F:
            raise ValueError(
                "new-band safe-closed NB cannot exceed safe closures L + F"
            )
        if self.NB_F > self.NB or self.NB_F > self.F:
            raise ValueError(
                "new-band failures NB_F cannot exceed NB or total failures F"
            )

    @property
    def safe_closed(self) -> int:
        return self.L + self.F

    @property
    def other_unsafe(self) -> int:
        return self.U_fall + self.U_collision

    @property
    def unsafe(self) -> int:
        return self.U_table + self.U_fall + self.U_collision

    def as_dict(self) -> Dict[str, int]:
        return {
            field: getattr(self, field)
            for field in (
                "P",
                "A",
                "I",
                "S",
                "C",
                "L",
                "F",
                "U_table",
                "U_fall",
                "U_collision",
                "X",
                "NB",
                "NB_F",
            )
        }


BallOutcomeCounts = BallOutcomeLedger


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float

    def as_dict(self) -> Dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


def wilson_interval(
    successes: int,
    attempts: int,
    *,
    z: float,
) -> ConfidenceInterval:
    successes = _plain_int(successes, name="successes")
    attempts = _plain_int(attempts, name="attempts")
    z = _finite_float(z, name="z", minimum=0.0)
    if successes > attempts:
        raise ValueError("successes must not exceed attempts")
    if attempts == 0:
        return ConfidenceInterval(0.0, 1.0)
    n = float(attempts)
    p = float(successes) / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
        / denominator
    )
    return ConfidenceInterval(
        max(0.0, center - radius),
        min(1.0, center + radius),
    )


@dataclass(frozen=True)
class BallCurriculumConfig:
    min_proposals: int = CANARY_MIN
    min_safe_closed: int = CANARY_MIN
    target_failure_rate: float = 0.10
    failure_band_half_width: float = 0.025
    min_solver_admit_rate: float = 0.95
    min_install_rate: float = 0.95
    min_start_rate: float = 0.95
    min_close_rate: float = 0.95
    max_other_unsafe_rate: float = 0.02
    confidence_z: float = 1.96
    max_center_failures: int = 8

    def __post_init__(self) -> None:
        for field, floor in (
            ("min_proposals", CANARY_MIN),
            ("min_safe_closed", CANARY_MIN),
        ):
            value = _plain_int(getattr(self, field), name=field, minimum=1)
            if value < floor:
                raise ValueError(f"{field} cannot be below {floor}")
        target = _finite_float(
            self.target_failure_rate,
            name="target_failure_rate",
            minimum=0.0,
            maximum=1.0,
        )
        half = _finite_float(
            self.failure_band_half_width,
            name="failure_band_half_width",
            minimum=0.0,
            maximum=0.5,
        )
        if target - half < 0.0 or target + half > 1.0:
            raise ValueError("target failure band must lie inside [0, 1]")
        for field in (
            "min_solver_admit_rate",
            "min_install_rate",
            "min_start_rate",
            "min_close_rate",
            "max_other_unsafe_rate",
        ):
            _finite_float(
                getattr(self, field),
                name=field,
                minimum=0.0,
                maximum=1.0,
            )
        _finite_float(self.confidence_z, name="confidence_z", minimum=0.0)
        _plain_int(
            self.max_center_failures,
            name="max_center_failures",
            minimum=1,
        )

    @classmethod
    def formal(cls, **overrides: object) -> "BallCurriculumConfig":
        return cls(**overrides)

    @property
    def heldout_min_proposals(self) -> int:
        """Code-frozen heldout floor, deliberately outside manifest config."""

        return HELDOUT_MIN

    @property
    def heldout_min_safe_closed(self) -> int:
        """Code-frozen heldout safe-closed floor."""

        return HELDOUT_MIN

    @property
    def failure_band(self) -> Tuple[float, float]:
        return (
            self.target_failure_rate - self.failure_band_half_width,
            self.target_failure_rate + self.failure_band_half_width,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "min_proposals",
                "min_safe_closed",
                "target_failure_rate",
                "failure_band_half_width",
                "min_solver_admit_rate",
                "min_install_rate",
                "min_start_rate",
                "min_close_rate",
                "max_other_unsafe_rate",
                "confidence_z",
                "max_center_failures",
            )
        }


@dataclass(frozen=True)
class ArmSchedulerConfig:
    rolling_window: int = 100
    min_history: int = 20
    forced_every: int = 5
    max_gap_factor: int = 2
    new_band_ring_size: int = 30

    def __post_init__(self) -> None:
        rolling = _plain_int(
            self.rolling_window, name="rolling_window", minimum=1
        )
        history = _plain_int(
            self.min_history, name="min_history", minimum=1
        )
        if rolling != 100:
            raise ValueError("rolling_window is contractually fixed at 100")
        if history > rolling:
            raise ValueError("min_history cannot exceed rolling_window")
        _plain_int(self.forced_every, name="forced_every", minimum=1)
        _plain_int(self.max_gap_factor, name="max_gap_factor", minimum=1)
        ring = _plain_int(
            self.new_band_ring_size,
            name="new_band_ring_size",
            minimum=1,
        )
        if ring != 30:
            raise ValueError(
                "new_band_ring_size is contractually fixed at 30"
            )
        if ring > rolling:
            raise ValueError(
                "new_band_ring_size cannot exceed rolling_window"
            )

    def as_dict(self) -> Dict[str, int]:
        return {
            "rolling_window": self.rolling_window,
            "min_history": self.min_history,
            "forced_every": self.forced_every,
            "max_gap_factor": self.max_gap_factor,
            "new_band_ring_size": self.new_band_ring_size,
        }

    @property
    def contract_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 2,
                "kind": "action_ball_arm_scheduler",
                "arm_catalog_sha256": ARM_CATALOG_SHA256,
                "config": self.as_dict(),
                "score": {
                    "stage_gates": (
                        "Wilson LCB for A/P, I/A, S/I, C/S"
                    ),
                    "unsafe_gates": (
                        "X=0, U_table=0, and point "
                        "(U_fall+U_collision)/C <= configured maximum"
                    ),
                    "objective": "minimize Wilson UCB of F/(L+F)",
                    "window": "latest 100 matching arm-epoch rows only",
                },
                "new_band_ring": {
                    # Franco 2026-07-28 ruling: the whole-window failure rate
                    # over a widened uniform range dilutes the newly added
                    # band; marginal promotion must therefore be judged on a
                    # dedicated per-(arm, candidate-level) ring of attempts
                    # actually drawn inside the new band.
                    "ring_size": 30,
                    "membership": (
                        "attempt drawn value inside the candidate-minus-"
                        "frontier new band, closed, safe terminal (L or F), "
                        "not infrastructure-invalid; newest ring_size rows "
                        "of the matching arm-epoch strata"
                    ),
                    "promotion": (
                        "marginal frontier changes require a full ring; the "
                        "new-range verdict is a direct failure count over "
                        "exactly these ring rows: <=floor(band_lower*ring) "
                        "too easy, >floor(band_upper*ring) too hard, "
                        "otherwise in band"
                    ),
                    "incomplete": (
                        "a ring below ring_size keeps collecting and is "
                        "never a promotion failure"
                    ),
                },
                "forced": (
                    "warmup minimum, periodic oldest, hard max-gap, "
                    "catalog-order ties"
                ),
            }
        )


@dataclass(frozen=True)
class BallDomainEvidence:
    key: ActionProfileKey
    arm_catalog_sha256: str
    scheduler_contract_sha256: str
    sampler_sha256: str
    solver_sha256: str
    policy_contract_sha256: str
    policy_checkpoint_sha256: str
    policy_generation: int
    evidence_role: str
    domain_epoch: int
    stratum: str
    selected_arm_key: str
    selection_round: int
    arm_levels: Tuple[float, ...]
    rho: float
    seed_block_start: int
    seed_block_end_exclusive: int
    sample_id_start: int
    sample_id_end_exclusive: int
    sample_receipt_root_sha256: str
    unique_birth_count: int
    birth_receipt_root_sha256: str
    seq: int
    window_id: str
    ledger: BallOutcomeLedger
    window_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ActionProfileKey):
            raise TypeError("key must be ActionProfileKey")
        for field in (
            "arm_catalog_sha256",
            "scheduler_contract_sha256",
            "sampler_sha256",
            "solver_sha256",
            "policy_contract_sha256",
            "policy_checkpoint_sha256",
            "sample_receipt_root_sha256",
            "birth_receipt_root_sha256",
        ):
            _sha256(getattr(self, field), name=field)
        if self.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("evidence arm catalog mismatch")
        if self.evidence_role not in EVIDENCE_ROLES:
            raise ValueError("invalid evidence_role")
        _plain_int(
            self.policy_generation,
            name="policy_generation",
            minimum=1,
        )
        _plain_int(self.domain_epoch, name="domain_epoch")
        _plain_int(self.selection_round, name="selection_round")
        _plain_int(self.seq, name="seq", minimum=1)
        _plain_int(
            self.unique_birth_count,
            name="unique_birth_count",
            minimum=1,
        )
        if type(self.stratum) is not str or not self.stratum:
            raise ValueError("stratum must be a non-empty string")
        if type(self.selected_arm_key) is not str:
            raise ValueError("selected_arm_key must be a string")
        if (
            self.selected_arm_key
            and self.selected_arm_key not in ARM_KEYS
        ):
            raise ValueError("selected_arm_key is outside ARM_KEYS")
        if (
            not isinstance(self.arm_levels, tuple)
            or len(self.arm_levels) != len(ARM_KEYS)
        ):
            raise ValueError(
                f"arm_levels must be a {len(ARM_KEYS)}-tuple"
            )
        for index, level in enumerate(self.arm_levels):
            _finite_float(
                level,
                name=f"arm_levels[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
        _finite_float(self.rho, name="rho", minimum=0.0, maximum=1.0)
        for field in (
            "seed_block_start",
            "seed_block_end_exclusive",
            "sample_id_start",
            "sample_id_end_exclusive",
        ):
            _plain_int(getattr(self, field), name=field)
        if self.seed_block_end_exclusive <= self.seed_block_start:
            raise ValueError("seed block must be non-empty")
        if self.sample_id_end_exclusive <= self.sample_id_start:
            raise ValueError("sample id range must be non-empty")
        if not isinstance(self.ledger, BallOutcomeLedger):
            raise TypeError("ledger must be BallOutcomeLedger")
        if (
            self.sample_id_end_exclusive - self.sample_id_start
            != self.ledger.P
            or self.seed_block_end_exclusive - self.seed_block_start
            != self.ledger.P
        ):
            raise ValueError(
                "sample and seed ranges must both have ledger.P rows"
            )
        if self.unique_birth_count > self.ledger.P:
            raise ValueError("unique_birth_count cannot exceed P")
        if type(self.window_id) is not str or not self.window_id:
            raise ValueError("window_id must be non-empty")
        _sha256(self.window_sha256, name="window_sha256")
        if self.window_sha256 != self.compute_window_sha256():
            raise ValueError("window_sha256 does not match evidence contents")

    def _hash_document(self) -> Dict[str, object]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "key": self.key.as_dict(),
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "scheduler_contract_sha256": (
                self.scheduler_contract_sha256
            ),
            "sampler_sha256": self.sampler_sha256,
            "solver_sha256": self.solver_sha256,
            "policy_contract_sha256": self.policy_contract_sha256,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "policy_generation": self.policy_generation,
            "evidence_role": self.evidence_role,
            "domain_epoch": self.domain_epoch,
            "stratum": self.stratum,
            "selected_arm_key": self.selected_arm_key,
            "selection_round": self.selection_round,
            "arm_levels": list(self.arm_levels),
            "rho": self.rho,
            "seed_block_start": self.seed_block_start,
            "seed_block_end_exclusive": self.seed_block_end_exclusive,
            "sample_id_start": self.sample_id_start,
            "sample_id_end_exclusive": self.sample_id_end_exclusive,
            "sample_receipt_root_sha256": (
                self.sample_receipt_root_sha256
            ),
            "unique_birth_count": self.unique_birth_count,
            "birth_receipt_root_sha256": (
                self.birth_receipt_root_sha256
            ),
            "seq": self.seq,
            "window_id": self.window_id,
            "ledger": self.ledger.as_dict(),
        }

    def compute_window_sha256(self) -> str:
        return _canonical_sha256(self._hash_document())

    @classmethod
    def create(cls, **kwargs: object) -> "BallDomainEvidence":
        required = {
            "key",
            "arm_catalog_sha256",
            "scheduler_contract_sha256",
            "sampler_sha256",
            "solver_sha256",
            "policy_contract_sha256",
            "policy_checkpoint_sha256",
            "policy_generation",
            "evidence_role",
            "domain_epoch",
            "stratum",
            "selected_arm_key",
            "selection_round",
            "arm_levels",
            "rho",
            "seed_block_start",
            "seed_block_end_exclusive",
            "sample_id_start",
            "sample_id_end_exclusive",
            "sample_receipt_root_sha256",
            "unique_birth_count",
            "birth_receipt_root_sha256",
            "seq",
            "window_id",
            "ledger",
        }
        if set(kwargs) != required:
            raise TypeError(
                "create evidence fields mismatch "
                f"(missing={sorted(required - set(kwargs))}, "
                f"unknown={sorted(set(kwargs) - required)})"
            )
        ledger = kwargs["ledger"]
        if not isinstance(ledger, BallOutcomeLedger):
            raise TypeError("ledger must be BallOutcomeLedger")
        document = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "key": kwargs["key"].as_dict(),
            "arm_catalog_sha256": kwargs["arm_catalog_sha256"],
            "scheduler_contract_sha256": kwargs[
                "scheduler_contract_sha256"
            ],
            "sampler_sha256": kwargs["sampler_sha256"],
            "solver_sha256": kwargs["solver_sha256"],
            "policy_contract_sha256": kwargs["policy_contract_sha256"],
            "policy_checkpoint_sha256": kwargs[
                "policy_checkpoint_sha256"
            ],
            "policy_generation": kwargs["policy_generation"],
            "evidence_role": kwargs["evidence_role"],
            "domain_epoch": kwargs["domain_epoch"],
            "stratum": kwargs["stratum"],
            "selected_arm_key": kwargs["selected_arm_key"],
            "selection_round": kwargs["selection_round"],
            "arm_levels": list(kwargs["arm_levels"]),
            "rho": kwargs["rho"],
            "seed_block_start": kwargs["seed_block_start"],
            "seed_block_end_exclusive": kwargs[
                "seed_block_end_exclusive"
            ],
            "sample_id_start": kwargs["sample_id_start"],
            "sample_id_end_exclusive": kwargs[
                "sample_id_end_exclusive"
            ],
            "sample_receipt_root_sha256": kwargs[
                "sample_receipt_root_sha256"
            ],
            "unique_birth_count": kwargs["unique_birth_count"],
            "birth_receipt_root_sha256": kwargs[
                "birth_receipt_root_sha256"
            ],
            "seq": kwargs["seq"],
            "window_id": kwargs["window_id"],
            "ledger": ledger.as_dict(),
        }
        return cls(window_sha256=_canonical_sha256(document), **kwargs)


@dataclass(frozen=True)
class ExpectedDomain:
    stratum: str
    domain_epoch: int
    selected_arm_key: str
    selection_round: int
    arm_catalog_sha256: str
    scheduler_contract_sha256: str
    arm_levels: Tuple[float, ...]
    rho: float


@dataclass(frozen=True)
class BallCurriculumDecision:
    key: ActionProfileKey
    kind: str
    evidence_role: str
    stratum: str
    arm_key: Optional[str]
    domain_epoch_before: int
    domain_epoch_after: int
    frontier_before: Tuple[float, ...]
    frontier_after: Tuple[float, ...]
    rho_before: float
    rho_after: float
    solver_admit: ConfidenceInterval
    install: ConfidenceInterval
    start: ConfidenceInterval
    close: ConfidenceInterval
    policy_failure: ConfidenceInterval
    other_unsafe: ConfidenceInterval
    blockers: Tuple[str, ...]
    window_sha256: str


_ATTEMPT_FIELDS = (
    "sample_receipt_sha256",
    "birth_receipt_sha256",
    "solver_admitted",
    "installed",
    "started",
    "closed",
    "terminal_outcome",
    "infrastructure_invalid",
    "in_new_band",
)
_TERMINALS = (
    "legal_return",
    "safe_nonreturn",
    "table_hit",
    "fall",
    "collision",
)
_RING_SAFE_TERMINALS = ("legal_return", "safe_nonreturn")


def _ring_eligible(row: Mapping[str, object]) -> bool:
    """One new-band ring member: drawn in the new band and safely closed."""

    return bool(
        row["in_new_band"]
        and row["closed"]
        and not row["infrastructure_invalid"]
        and row["terminal_outcome"] in _RING_SAFE_TERMINALS
    )


def _validated_attempt_row(value: object, *, name: str) -> Dict[str, object]:
    row = _exact_keys(value, _ATTEMPT_FIELDS, name=name)
    sample_sha = _sha256(
        row["sample_receipt_sha256"],
        name=f"{name}.sample_receipt_sha256",
    )
    birth_sha = _sha256(
        row["birth_receipt_sha256"],
        name=f"{name}.birth_receipt_sha256",
    )
    flags = {}
    for field in (
        "solver_admitted",
        "installed",
        "started",
        "closed",
        "infrastructure_invalid",
        "in_new_band",
    ):
        if type(row[field]) is not bool:
            raise ValueError(f"{name}.{field} must be bool")
        flags[field] = row[field]
    if not (
        flags["solver_admitted"]
        >= flags["installed"]
        >= flags["started"]
        >= flags["closed"]
    ):
        raise ValueError(f"{name} stage flags are not contained")
    terminal = row["terminal_outcome"]
    if flags["closed"]:
        if terminal not in _TERMINALS:
            raise ValueError(f"{name} closed row needs terminal outcome")
    elif terminal is not None:
        raise ValueError(f"{name} non-closed row cannot have terminal")
    return {
        "sample_receipt_sha256": sample_sha,
        "birth_receipt_sha256": birth_sha,
        **flags,
        "terminal_outcome": terminal,
    }


def _ledger_from_attempt_rows(
    attempts: Sequence[Mapping[str, object]],
) -> BallOutcomeLedger:
    terminals = {name: 0 for name in _TERMINALS}
    new_band = 0
    new_band_failures = 0
    for attempt in attempts:
        terminal = attempt["terminal_outcome"]
        if terminal is not None:
            terminals[terminal] += 1
        if _ring_eligible(attempt):
            new_band += 1
            if terminal == "safe_nonreturn":
                new_band_failures += 1
    return BallOutcomeLedger(
        P=len(attempts),
        A=sum(bool(item["solver_admitted"]) for item in attempts),
        I=sum(bool(item["installed"]) for item in attempts),
        S=sum(bool(item["started"]) for item in attempts),
        C=sum(bool(item["closed"]) for item in attempts),
        L=terminals["legal_return"],
        F=terminals["safe_nonreturn"],
        U_table=terminals["table_hit"],
        U_fall=terminals["fall"],
        U_collision=terminals["collision"],
        X=sum(
            bool(item["infrastructure_invalid"]) for item in attempts
        ),
        NB=new_band,
        NB_F=new_band_failures,
    )


@dataclass(frozen=True)
class _Receipt:
    evidence: BallDomainEvidence
    certified: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "evidence": self.evidence._hash_document(),
            "window_sha256": self.evidence.window_sha256,
            "certified": self.certified,
        }


@dataclass(frozen=True)
class _SchedulerReceipt:
    evidence: BallDomainEvidence
    attempts: Tuple[Dict[str, object], ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "evidence": self.evidence._hash_document(),
            "window_sha256": self.evidence.window_sha256,
            "attempts": [dict(row) for row in self.attempts],
        }


@dataclass
class _Progress:
    phase: str
    arm_frontier_indices: Tuple[int, ...]
    arm_status: Tuple[str, ...]
    arm_probe_indices: Tuple[int, ...]
    arm_epochs: Tuple[int, ...]
    selected_arm_key: str
    selection_round: int
    last_selected_round: Tuple[int, ...]
    center_epoch: int
    joint_epoch: int
    joint_probe_index: int
    joint_rho_index: int
    center_failures: int
    pending_canary: Optional[_Receipt]
    formal_receipts: Tuple[_Receipt, ...]
    scheduler_receipts: Tuple[_SchedulerReceipt, ...]
    last_certified: Optional[Dict[str, object]]

    def clone(self) -> "_Progress":
        return replace(
            self,
            arm_frontier_indices=tuple(self.arm_frontier_indices),
            arm_status=tuple(self.arm_status),
            arm_probe_indices=tuple(self.arm_probe_indices),
            arm_epochs=tuple(self.arm_epochs),
            last_selected_round=tuple(self.last_selected_round),
            formal_receipts=tuple(self.formal_receipts),
            scheduler_receipts=tuple(self.scheduler_receipts),
            last_certified=(
                None
                if self.last_certified is None
                else dict(self.last_certified)
            ),
        )


class BallCurriculumStalledError(RuntimeError):
    pass


class ActionBallCurriculum:
    def __init__(
        self,
        *,
        contract_sha256: str,
        profile_order: Sequence[ActionProfileKey],
        sampler_sha256: str,
        solver_sha256: str,
        policy_contract_sha256: str,
        config: BallCurriculumConfig,
        scheduler_config: ArmSchedulerConfig = ArmSchedulerConfig(),
        evaluator_authority: object | None = None,
    ) -> None:
        self._contract_sha256 = _sha256(
            contract_sha256, name="contract_sha256"
        )
        self._sampler_sha256 = _sha256(
            sampler_sha256, name="sampler_sha256"
        )
        self._solver_sha256 = _sha256(
            solver_sha256, name="solver_sha256"
        )
        self._policy_contract_sha256 = _sha256(
            policy_contract_sha256, name="policy_contract_sha256"
        )
        if isinstance(profile_order, (str, bytes)):
            raise ValueError("profile_order must be a sequence")
        order = tuple(profile_order)
        if (
            not order
            or any(not isinstance(key, ActionProfileKey) for key in order)
            or len(order) != len(set(order))
        ):
            raise ValueError(
                "profile_order must contain unique ActionProfileKey values"
            )
        if not isinstance(config, BallCurriculumConfig):
            raise TypeError("config must be BallCurriculumConfig")
        if not isinstance(scheduler_config, ArmSchedulerConfig):
            raise TypeError("scheduler_config must be ArmSchedulerConfig")
        self._profile_order = order
        self._config = config
        self._scheduler_config = scheduler_config
        self._scheduler_contract_sha256 = (
            scheduler_config.contract_sha256
        )
        self._progress = {
            key: self._new_progress(key) for key in order
        }
        self._evaluator_authority: object | None = None
        if evaluator_authority is not None:
            self.bind_evaluator_authority(evaluator_authority)

    @staticmethod
    def _new_progress(key: ActionProfileKey) -> _Progress:
        status = tuple(
            "pending" if arm in key.enabled_arms else "disabled"
            for arm in ARM_KEYS
        )
        return _Progress(
            phase="center",
            arm_frontier_indices=(0,) * len(ARM_KEYS),
            arm_status=status,
            arm_probe_indices=(1,) * len(ARM_KEYS),
            arm_epochs=(0,) * len(ARM_KEYS),
            selected_arm_key="",
            selection_round=0,
            last_selected_round=(0,) * len(ARM_KEYS),
            center_epoch=0,
            joint_epoch=0,
            joint_probe_index=1,
            joint_rho_index=0,
            center_failures=0,
            pending_canary=None,
            formal_receipts=(),
            scheduler_receipts=(),
            last_certified=None,
        )

    @staticmethod
    def _evaluation_types() -> Tuple[type, type, type]:
        try:
            from .action_ball_evaluation import (
                FrozenEvaluationAuthorityError,
                FrozenEvaluationCapability,
                FrozenEvaluatorAuthority,
            )
        except ImportError:
            from action_ball_evaluation import (  # type: ignore
                FrozenEvaluationAuthorityError,
                FrozenEvaluationCapability,
                FrozenEvaluatorAuthority,
            )
        return (
            FrozenEvaluatorAuthority,
            FrozenEvaluationCapability,
            FrozenEvaluationAuthorityError,
        )

    def bind_evaluator_authority(self, authority: object) -> None:
        authority_type, _, authority_error = self._evaluation_types()
        if self._evaluator_authority is not None:
            raise authority_error(
                "frozen evaluator authority may be bound only once"
            )
        if type(authority) is not authority_type:
            raise authority_error(
                "formal curriculum requires exact evaluator authority"
            )
        authority.assert_binding(
            curriculum_contract_sha256=self._contract_sha256,
            profile_order=self._profile_order,
            arm_catalog_sha256=ARM_CATALOG_SHA256,
            scheduler_contract_sha256=self._scheduler_contract_sha256,
            sampler_sha256=self._sampler_sha256,
            solver_sha256=self._solver_sha256,
            policy_contract_sha256=self._policy_contract_sha256,
        )
        self._evaluator_authority = authority

    @property
    def profile_order(self) -> Tuple[ActionProfileKey, ...]:
        return self._profile_order

    @property
    def config(self) -> BallCurriculumConfig:
        return self._config

    @property
    def scheduler_config(self) -> ArmSchedulerConfig:
        return self._scheduler_config

    @property
    def scheduler_contract_sha256(self) -> str:
        return self._scheduler_contract_sha256

    def phase(self, key: ActionProfileKey) -> str:
        return self._progress[key].phase

    def frontiers(self, key: ActionProfileKey) -> Dict[str, float]:
        return {
            arm: LEVELS[index]
            for arm, index in zip(
                ARM_KEYS, self._progress[key].arm_frontier_indices
            )
        }

    def levels(self, key: ActionProfileKey) -> Dict[str, float]:
        progress = self._progress[key]
        rho = JOINT_RHOS[progress.joint_rho_index]
        if progress.phase in ("joint", "steady", "stalled"):
            return {
                arm: LEVELS[index] * rho
                for arm, index in zip(
                    ARM_KEYS, progress.arm_frontier_indices
                )
            }
        return self.frontiers(key)

    def selected_arm(self, key: ActionProfileKey) -> str:
        return self._progress[key].selected_arm_key

    def joint_rho(self, key: ActionProfileKey) -> float:
        return JOINT_RHOS[self._progress[key].joint_rho_index]

    def _domain(
        self,
        progress: _Progress,
        *,
        stratum: str,
        epoch: int,
        arm_levels: Tuple[float, ...],
        rho: float,
        selected_arm_key: str,
    ) -> ExpectedDomain:
        return ExpectedDomain(
            stratum=stratum,
            domain_epoch=epoch,
            selected_arm_key=selected_arm_key,
            selection_round=progress.selection_round,
            arm_catalog_sha256=ARM_CATALOG_SHA256,
            scheduler_contract_sha256=(
                self._scheduler_contract_sha256
            ),
            arm_levels=arm_levels,
            rho=rho,
        )

    def selected_formal_domain(
        self, key: ActionProfileKey
    ) -> Optional[ExpectedDomain]:
        return self._selected_formal_domain_from_progress(
            self._progress[key]
        )

    def expected_domains(
        self, key: ActionProfileKey
    ) -> Tuple[ExpectedDomain, ...]:
        domain = self.selected_formal_domain(key)
        return () if domain is None else (domain,)

    def _selected_formal_domain_from_progress(
        self, progress: _Progress
    ) -> Optional[ExpectedDomain]:
        zero = (0.0,) * len(ARM_KEYS)
        if progress.phase == "center":
            return self._domain(
                progress,
                stratum="center",
                epoch=progress.center_epoch,
                arm_levels=zero,
                rho=0.0,
                selected_arm_key="",
            )
        if progress.phase == "marginal":
            if not progress.selected_arm_key:
                return None
            index = ARM_KEYS.index(progress.selected_arm_key)
            levels = [0.0] * len(ARM_KEYS)
            levels[index] = LEVELS[progress.arm_probe_indices[index]]
            return self._domain(
                progress,
                stratum=f"marginal:{progress.selected_arm_key}",
                epoch=progress.arm_epochs[index],
                arm_levels=tuple(levels),
                rho=0.0,
                selected_arm_key=progress.selected_arm_key,
            )
        if progress.phase == "joint":
            rho = JOINT_RHOS[progress.joint_probe_index]
            return self._domain(
                progress,
                stratum="joint",
                epoch=progress.joint_epoch,
                arm_levels=tuple(
                    LEVELS[index] * rho
                    for index in progress.arm_frontier_indices
                ),
                rho=rho,
                selected_arm_key="",
            )
        if progress.phase == "steady":
            rho = JOINT_RHOS[progress.joint_rho_index]
            return self._domain(
                progress,
                stratum="steady",
                epoch=progress.joint_epoch,
                arm_levels=tuple(
                    LEVELS[index] * rho
                    for index in progress.arm_frontier_indices
                ),
                rho=rho,
                selected_arm_key="",
            )
        return None

    def scheduler_domains(
        self, key: ActionProfileKey
    ) -> Tuple[ExpectedDomain, ...]:
        progress = self._progress[key]
        if progress.phase != "marginal" or progress.pending_canary:
            return ()
        result = []
        for index, status in enumerate(progress.arm_status):
            if status != "probing":
                continue
            levels = [0.0] * len(ARM_KEYS)
            levels[index] = LEVELS[progress.arm_probe_indices[index]]
            result.append(
                self._domain(
                    progress,
                    stratum=f"marginal:{ARM_KEYS[index]}",
                    epoch=progress.arm_epochs[index],
                    arm_levels=tuple(levels),
                    rho=0.0,
                    selected_arm_key=ARM_KEYS[index],
                )
            )
        return tuple(result)

    def domain_epoch(
        self, key: ActionProfileKey, stratum: Optional[str] = None
    ) -> int:
        domains = (
            self.expected_domains(key)
            if stratum is None
            else self.expected_domains(key) + self.scheduler_domains(key)
        )
        if stratum is None:
            if len(domains) != 1:
                raise ValueError("no unique selected formal domain")
            return domains[0].domain_epoch
        matches = [d for d in domains if d.stratum == stratum]
        if len(matches) != 1:
            raise ValueError(f"stratum {stratum!r} is not expected")
        return matches[0].domain_epoch

    def _all_evidence(self) -> Tuple[BallDomainEvidence, ...]:
        result = []
        for progress in self._progress.values():
            result.extend(
                receipt.evidence for receipt in progress.formal_receipts
            )
            result.extend(
                receipt.evidence for receipt in progress.scheduler_receipts
            )
        return tuple(result)

    def _retained_canary_window_sha256(
        self,
        progress_by_key: Mapping[ActionProfileKey, _Progress],
    ) -> Tuple[str, ...]:
        return tuple(
            progress_by_key[key].pending_canary.evidence.window_sha256
            for key in self._profile_order
            if progress_by_key[key].pending_canary is not None
        )

    def _validate_global_batch(
        self,
        evidence_by_profile: Mapping[
            ActionProfileKey, BallDomainEvidence
        ],
    ) -> None:
        history = self._all_evidence()
        batch = tuple(evidence_by_profile.values())
        ids = [item.window_id for item in batch]
        hashes = [item.window_sha256 for item in batch]
        sequences = [item.seq for item in batch]
        if (
            len(ids) != len(set(ids))
            or len(hashes) != len(set(hashes))
            or len(sequences) != len(set(sequences))
        ):
            raise ValueError("atomic evidence identity is duplicated")
        history_ids = {item.window_id for item in history}
        history_hashes = {item.window_sha256 for item in history}
        if history_ids.intersection(ids) or history_hashes.intersection(hashes):
            raise ValueError("evidence window was already consumed")
        global_last_seq = max((item.seq for item in history), default=0)
        if min(sequences) <= global_last_seq:
            raise ValueError("evidence seq must be globally monotonic")
        if len({item.policy_generation for item in batch}) != 1:
            raise ValueError(
                "one atomic update requires one policy generation"
            )
        checkpoint_by_generation: Dict[int, str] = {}
        previous_generation = 0
        for item in sorted(history + batch, key=lambda row: row.seq):
            if item.policy_generation < previous_generation:
                raise ValueError("policy generation regressed globally")
            previous_generation = item.policy_generation
            previous = checkpoint_by_generation.setdefault(
                item.policy_generation,
                item.policy_checkpoint_sha256,
            )
            if previous != item.policy_checkpoint_sha256:
                raise ValueError(
                    "one policy generation cannot name multiple checkpoints"
                )
        keyed = tuple(
            (item.key, item) for item in history
        ) + tuple(evidence_by_profile.items())
        for index, (left_key, left) in enumerate(keyed):
            for right_key, right in keyed[index + 1 :]:
                if left_key.action_uid != right_key.action_uid:
                    continue
                if _intervals_overlap(
                    left.sample_id_start,
                    left.sample_id_end_exclusive,
                    right.sample_id_start,
                    right.sample_id_end_exclusive,
                ):
                    raise ValueError("same-action sample ranges overlap")
                if _intervals_overlap(
                    left.seed_block_start,
                    left.seed_block_end_exclusive,
                    right.seed_block_start,
                    right.seed_block_end_exclusive,
                ):
                    raise ValueError("same-action seed ranges overlap")

    def _validate_common(
        self,
        key: ActionProfileKey,
        evidence: BallDomainEvidence,
    ) -> None:
        if evidence.key != key:
            raise ValueError("evidence profile key mismatch")
        for field, expected in (
            ("arm_catalog_sha256", ARM_CATALOG_SHA256),
            (
                "scheduler_contract_sha256",
                self._scheduler_contract_sha256,
            ),
            ("sampler_sha256", self._sampler_sha256),
            ("solver_sha256", self._solver_sha256),
            ("policy_contract_sha256", self._policy_contract_sha256),
        ):
            if getattr(evidence, field) != expected:
                raise ValueError(f"evidence {field} mismatch")

    @staticmethod
    def _domain_matches(
        evidence: BallDomainEvidence, domain: ExpectedDomain
    ) -> bool:
        return (
            evidence.stratum == domain.stratum
            and evidence.domain_epoch == domain.domain_epoch
            and evidence.selected_arm_key == domain.selected_arm_key
            and evidence.selection_round == domain.selection_round
            and evidence.arm_levels == domain.arm_levels
            and evidence.rho == domain.rho
            and evidence.arm_catalog_sha256
            == domain.arm_catalog_sha256
            and evidence.scheduler_contract_sha256
            == domain.scheduler_contract_sha256
        )

    def observe_scheduler(
        self,
        capability_by_profile: Mapping[ActionProfileKey, object],
    ) -> Tuple[str, ...]:
        _, capability_type, authority_error = self._evaluation_types()
        if self._evaluator_authority is None:
            raise authority_error(
                "curriculum hold: frozen evaluator authority is not bound"
            )
        if not capability_by_profile or any(
            type(value) is not capability_type
            for value in capability_by_profile.values()
        ):
            raise authority_error(
                "scheduler observations require opaque capabilities"
            )
        evidence_by_profile = self._evaluator_authority.inspect_many(
            capability_by_profile
        )
        attempt_rows = self._evaluator_authority.attempt_rows_many(
            capability_by_profile
        )
        self._validate_global_batch(evidence_by_profile)
        staged = {
            key: progress.clone()
            for key, progress in self._progress.items()
        }
        selected = []
        for key in self._profile_order:
            if key not in evidence_by_profile:
                continue
            evidence = evidence_by_profile[key]
            self._validate_common(key, evidence)
            if evidence.evidence_role != "scheduler":
                raise ValueError("observe_scheduler requires scheduler role")
            domains = {
                domain.stratum: domain
                for domain in self._scheduler_domains_from_progress(
                    staged[key]
                )
            }
            domain = domains.get(evidence.stratum)
            if domain is None or not self._domain_matches(evidence, domain):
                raise ValueError(
                    "scheduler evidence does not match current arm candidate"
                )
            rows = tuple(
                _validated_attempt_row(
                    row,
                    name=f"scheduler[{key.action_uid}][{index}]",
                )
                for index, row in enumerate(attempt_rows[key])
            )
            if _ledger_from_attempt_rows(rows) != evidence.ledger:
                raise ValueError(
                    "scheduler attempt rows do not derive evidence ledger"
                )
            staged[key].scheduler_receipts += (
                _SchedulerReceipt(evidence=evidence, attempts=rows),
            )
            if staged[key].pending_canary is None:
                self._reselect_arm(key, staged[key])
            selected.append(staged[key].selected_arm_key)
        retained = self._retained_canary_window_sha256(staged)
        consumed = self._evaluator_authority.consume_many(
            capability_by_profile,
            retain_formal_window_sha256=retained,
        )
        if consumed != evidence_by_profile:
            raise authority_error("authority changed during scheduler update")
        self._evaluator_authority.assert_formal_retention(retained)
        self._progress = staged
        return tuple(selected)

    def _scheduler_domains_from_progress(
        self, progress: _Progress
    ) -> Tuple[ExpectedDomain, ...]:
        if progress.phase != "marginal" or progress.pending_canary:
            return ()
        result = []
        for index, status in enumerate(progress.arm_status):
            if status != "probing":
                continue
            levels = [0.0] * len(ARM_KEYS)
            levels[index] = LEVELS[progress.arm_probe_indices[index]]
            result.append(
                self._domain(
                    progress,
                    stratum=f"marginal:{ARM_KEYS[index]}",
                    epoch=progress.arm_epochs[index],
                    arm_levels=tuple(levels),
                    rho=0.0,
                    selected_arm_key=ARM_KEYS[index],
                )
            )
        return tuple(result)

    def _recent_ledger(
        self, progress: _Progress, arm_index: int
    ) -> BallOutcomeLedger:
        arm = ARM_KEYS[arm_index]
        stratum = f"marginal:{arm}"
        epoch = progress.arm_epochs[arm_index]
        level = LEVELS[progress.arm_probe_indices[arm_index]]
        rows = []
        for receipt in progress.scheduler_receipts:
            evidence = receipt.evidence
            if (
                evidence.stratum == stratum
                and evidence.domain_epoch == epoch
                and evidence.arm_levels[arm_index] == level
            ):
                rows.extend(receipt.attempts)
        rows = rows[-self._scheduler_config.rolling_window :]
        return _ledger_from_attempt_rows(rows)

    def _new_band_ring_rows(
        self, progress: _Progress, arm_index: int
    ) -> Tuple[Mapping[str, object], ...]:
        """Newest ring-size new-band safe-closed rows for the arm candidate.

        The ring is a pure function of the retained scheduler receipts, so it
        rides the existing deterministic checkpoint replay: the persisted
        attempt rows (which now declare ``in_new_band``) are the ring state,
        and every contributing window is identified by its window SHA.
        """

        arm = ARM_KEYS[arm_index]
        stratum = f"marginal:{arm}"
        epoch = progress.arm_epochs[arm_index]
        level = LEVELS[progress.arm_probe_indices[arm_index]]
        rows = []
        for receipt in progress.scheduler_receipts:
            evidence = receipt.evidence
            if (
                evidence.stratum == stratum
                and evidence.domain_epoch == epoch
                and evidence.arm_levels[arm_index] == level
            ):
                for row in receipt.attempts:
                    if _ring_eligible(row):
                        rows.append(row)
        return tuple(
            rows[-self._scheduler_config.new_band_ring_size :]
        )

    def _scheduler_eligible(
        self, ledger: BallOutcomeLedger
    ) -> bool:
        cfg = self._config
        if ledger.P < self._scheduler_config.min_history:
            return False
        if ledger.X or ledger.U_table:
            return False
        gates = (
            wilson_interval(ledger.A, ledger.P, z=cfg.confidence_z).lower
            >= cfg.min_solver_admit_rate,
            wilson_interval(ledger.I, ledger.A, z=cfg.confidence_z).lower
            >= cfg.min_install_rate,
            wilson_interval(ledger.S, ledger.I, z=cfg.confidence_z).lower
            >= cfg.min_start_rate,
            wilson_interval(ledger.C, ledger.S, z=cfg.confidence_z).lower
            >= cfg.min_close_rate,
            (
                ledger.C > 0
                and ledger.other_unsafe / ledger.C
                <= cfg.max_other_unsafe_rate
            ),
        )
        return all(gates) and ledger.safe_closed > 0

    def _reselect_arm(
        self, key: ActionProfileKey, progress: _Progress
    ) -> None:
        if progress.phase != "marginal" or progress.pending_canary:
            return
        eligible_indices = [
            index
            for index, status in enumerate(progress.arm_status)
            if status == "probing"
        ]
        if not eligible_indices:
            progress.selected_arm_key = ""
            return
        progress.selection_round += 1
        round_id = progress.selection_round
        ledgers = {
            index: self._recent_ledger(progress, index)
            for index in eligible_indices
        }
        under_sampled = [
            index
            for index in eligible_indices
            if ledgers[index].P < self._scheduler_config.min_history
        ]
        ages = {
            index: round_id - progress.last_selected_round[index]
            for index in eligible_indices
        }
        max_gap = (
            self._scheduler_config.max_gap_factor
            * len(eligible_indices)
        )
        overdue = [
            index
            for index in eligible_indices
            if ages[index] >= max_gap
        ]
        forced = (
            round_id % self._scheduler_config.forced_every == 0
        )
        if under_sampled:
            chosen = min(
                under_sampled,
                key=lambda index: (
                    ledgers[index].P,
                    -ages[index],
                    index,
                ),
            )
        elif overdue or forced:
            pool = overdue if overdue else eligible_indices
            chosen = min(pool, key=lambda index: (-ages[index], index))
        else:
            qualified = [
                index
                for index in eligible_indices
                if self._scheduler_eligible(ledgers[index])
            ]
            if qualified:
                chosen = min(
                    qualified,
                    key=lambda index: (
                        wilson_interval(
                            ledgers[index].F,
                            ledgers[index].safe_closed,
                            z=self._config.confidence_z,
                        ).upper,
                        index,
                    ),
                )
            else:
                chosen = min(
                    eligible_indices,
                    key=lambda index: (-ages[index], index),
                )
        rounds = list(progress.last_selected_round)
        rounds[chosen] = round_id
        progress.last_selected_round = tuple(rounds)
        progress.selected_arm_key = ARM_KEYS[chosen]

    def update_selected(
        self,
        capability_by_profile: Mapping[ActionProfileKey, object],
    ) -> Tuple[BallCurriculumDecision, ...]:
        _, capability_type, authority_error = self._evaluation_types()
        if self._evaluator_authority is None:
            raise authority_error(
                "curriculum hold: frozen evaluator authority is not bound"
            )
        if not capability_by_profile or any(
            type(value) is not capability_type
            for value in capability_by_profile.values()
        ):
            raise authority_error(
                "formal curriculum accepts only opaque capabilities"
            )
        evidence_by_profile = self._evaluator_authority.inspect_many(
            capability_by_profile
        )
        self._validate_global_batch(evidence_by_profile)
        staged = {
            key: progress.clone()
            for key, progress in self._progress.items()
        }
        decisions = []
        for key in self._profile_order:
            if key not in evidence_by_profile:
                continue
            evidence = evidence_by_profile[key]
            self._validate_common(key, evidence)
            decisions.append(
                self._apply_formal_evidence(
                    key, staged[key], evidence
                )
            )
        retained = self._retained_canary_window_sha256(staged)
        consumed = self._evaluator_authority.consume_many(
            capability_by_profile,
            retain_formal_window_sha256=retained,
        )
        if consumed != evidence_by_profile:
            raise authority_error("authority changed during formal update")
        self._evaluator_authority.assert_formal_retention(retained)
        self._progress = staged
        return tuple(decisions)

    def _metrics(
        self, ledger: BallOutcomeLedger
    ) -> Tuple[
        ConfidenceInterval,
        ConfidenceInterval,
        ConfidenceInterval,
        ConfidenceInterval,
        ConfidenceInterval,
        ConfidenceInterval,
        Tuple[str, ...],
        bool,
        bool,
        bool,
    ]:
        cfg = self._config
        admit = wilson_interval(ledger.A, ledger.P, z=cfg.confidence_z)
        install = wilson_interval(ledger.I, ledger.A, z=cfg.confidence_z)
        start = wilson_interval(ledger.S, ledger.I, z=cfg.confidence_z)
        close = wilson_interval(ledger.C, ledger.S, z=cfg.confidence_z)
        failure = wilson_interval(
            ledger.F, ledger.safe_closed, z=cfg.confidence_z
        )
        unsafe = wilson_interval(
            ledger.other_unsafe, ledger.C, z=cfg.confidence_z
        )
        blockers = []
        if admit.lower < cfg.min_solver_admit_rate:
            blockers.append("solver_admit_below_gate")
        if install.lower < cfg.min_install_rate:
            blockers.append("install_below_gate")
        if start.lower < cfg.min_start_rate:
            blockers.append("start_below_gate")
        if close.lower < cfg.min_close_rate:
            blockers.append("close_below_gate")
        if unsafe.upper > cfg.max_other_unsafe_rate:
            blockers.append("other_unsafe_above_gate")
        if ledger.U_table:
            blockers.append("table_hit_zero_tolerance")
        if ledger.X:
            blockers.append("attribution_exceptions_present")
        quality_bad = bool(blockers)
        too_hard = failure.lower > cfg.failure_band[1]
        too_easy = failure.upper < cfg.failure_band[0]
        return (
            admit,
            install,
            start,
            close,
            failure,
            unsafe,
            tuple(blockers),
            quality_bad,
            too_hard,
            too_easy,
        )

    def _apply_formal_evidence(
        self,
        key: ActionProfileKey,
        progress: _Progress,
        evidence: BallDomainEvidence,
    ) -> BallCurriculumDecision:
        if evidence.evidence_role not in (
            "frozen_canary",
            "frozen_heldout",
        ):
            raise ValueError("formal update requires frozen role")
        domain = self._selected_formal_domain_from_progress(progress)
        if domain is None or not self._domain_matches(evidence, domain):
            raise ValueError(
                "formal evidence does not match selected domain"
            )
        if evidence.unique_birth_count != evidence.ledger.P:
            raise ValueError(
                "formal windows require one unique birth per proposal"
            )
        (
            admit,
            install,
            start,
            close,
            failure,
            unsafe,
            blockers,
            quality_bad,
            too_hard,
            too_easy,
        ) = self._metrics(evidence.ledger)
        frontier_before = tuple(
            LEVELS[index] for index in progress.arm_frontier_indices
        )
        rho_before = JOINT_RHOS[progress.joint_rho_index]
        arm_key = (
            evidence.selected_arm_key
            if evidence.selected_arm_key
            else None
        )
        kind = "hold"
        certified = False

        if evidence.evidence_role == "frozen_canary":
            if progress.pending_canary is not None:
                raise ValueError("a frozen canary is already pending heldout")
            if (
                evidence.ledger.P < self._config.min_proposals
                or evidence.ledger.safe_closed
                < self._config.min_safe_closed
            ):
                raise ValueError("frozen canary is below 256-row gates")
            receipt = _Receipt(evidence=evidence, certified=False)
            progress.formal_receipts += (receipt,)
            if quality_bad:
                kind = "canary_blocked"
            else:
                progress.pending_canary = receipt
                kind = "canary_pass"
            return self._decision(
                key,
                progress,
                evidence,
                kind,
                arm_key,
                frontier_before,
                rho_before,
                admit,
                install,
                start,
                close,
                failure,
                unsafe,
                blockers,
            )

        pending = progress.pending_canary
        if pending is None:
            raise ValueError("heldout requires a prior frozen canary")
        canary = pending.evidence
        for field in (
            "stratum",
            "domain_epoch",
            "selected_arm_key",
            "selection_round",
            "arm_levels",
            "rho",
            "policy_checkpoint_sha256",
            "policy_generation",
        ):
            if getattr(canary, field) != getattr(evidence, field):
                raise ValueError(
                    f"heldout does not match frozen canary {field}"
                )
        if (
            evidence.ledger.P < self._config.heldout_min_proposals
            or evidence.ledger.safe_closed
            < self._config.heldout_min_safe_closed
        ):
            raise ValueError("frozen heldout is below 768-row gates")

        progress.pending_canary = None
        if progress.phase == "center":
            if quality_bad or too_hard:
                progress.center_failures += 1
                kind = "blocked_at_center"
                if (
                    progress.center_failures
                    >= self._config.max_center_failures
                ):
                    progress.phase = "stalled"
                    kind = "stalled_at_center"
            else:
                certified = True
                progress.phase = "marginal"
                progress.arm_status = tuple(
                    "disabled" if status == "disabled" else "probing"
                    for status in progress.arm_status
                )
                progress.last_certified = self._certificate(evidence)
                self._reselect_arm(key, progress)
                kind = "center_pass"
        elif progress.phase == "marginal":
            assert arm_key is not None
            index = ARM_KEYS.index(arm_key)
            frontiers = list(progress.arm_frontier_indices)
            statuses = list(progress.arm_status)
            probes = list(progress.arm_probe_indices)
            epochs = list(progress.arm_epochs)
            candidate = probes[index]
            # Franco 2026-07-28 ruling: the new-range verdict for one
            # (arm, candidate-level) promotion is computed from the
            # dedicated new-band ring, never from the diluted whole-window
            # failure rate.  An unfilled ring keeps collecting and is not a
            # promotion failure.
            ring_rows = self._new_band_ring_rows(progress, index)
            ring_size = self._scheduler_config.new_band_ring_size
            if quality_bad:
                statuses[index] = "decided"
                kind = "bound_marginal"
            elif len(ring_rows) < ring_size:
                kind = "new_band_ring_incomplete"
            else:
                ring_failures = sum(
                    1
                    for row in ring_rows
                    if row["terminal_outcome"] == "safe_nonreturn"
                )
                # Franco 2026-07-28 second ruling: the ring verdict is a
                # DIRECT COUNT against the band edges scaled by the ring
                # length, not a Wilson interval.  For the f10 band
                # [0.075, 0.125] with ring 30 this reads: <=2 too easy,
                # exactly 3 in band (lock), >=4 too hard; f20 derives its
                # own thresholds from the same formula.  The 1e-9 guards a
                # downward float error on exact-integer band edges only.
                band_low, band_high = self._config.failure_band
                easy_max = math.floor(band_low * ring_size + 1.0e-9)
                hard_min = (
                    math.floor(band_high * ring_size + 1.0e-9) + 1
                )
                if ring_failures >= hard_min:
                    statuses[index] = "decided"
                    kind = "bound_marginal"
                else:
                    certified = True
                    frontiers[index] = candidate
                    progress.last_certified = self._certificate(evidence)
                    if (
                        ring_failures <= easy_max
                        and candidate < len(LEVELS) - 1
                    ):
                        probes[index] += 1
                        epochs[index] += 1
                        kind = "expand_marginal"
                    else:
                        statuses[index] = "decided"
                        kind = "lock_marginal"
            progress.arm_frontier_indices = tuple(frontiers)
            progress.arm_status = tuple(statuses)
            progress.arm_probe_indices = tuple(probes)
            progress.arm_epochs = tuple(epochs)
            if all(
                status in ("decided", "disabled")
                for status in progress.arm_status
            ):
                progress.phase = "joint"
                progress.selected_arm_key = ""
                progress.joint_epoch = 0
                progress.joint_probe_index = 1
                kind += "_enter_joint"
            else:
                self._reselect_arm(key, progress)
        elif progress.phase == "joint":
            candidate = progress.joint_probe_index
            if quality_bad or too_hard:
                progress.joint_rho_index = max(0, candidate - 1)
                progress.phase = "steady"
                progress.joint_epoch += 1
                kind = "bound_joint"
            else:
                certified = True
                progress.joint_rho_index = candidate
                progress.last_certified = self._certificate(evidence)
                if too_easy and candidate < len(JOINT_RHOS) - 1:
                    progress.joint_probe_index += 1
                    progress.joint_epoch += 1
                    kind = "expand_joint"
                else:
                    progress.phase = "steady"
                    progress.joint_epoch += 1
                    kind = "enter_steady"
        elif progress.phase == "steady":
            if quality_bad or too_hard:
                if progress.joint_rho_index == 0:
                    progress.phase = "stalled"
                    kind = "stalled_at_joint_center"
                else:
                    progress.joint_rho_index -= 1
                    progress.joint_probe_index = (
                        progress.joint_rho_index
                    )
                    progress.joint_epoch += 1
                    kind = "retreat_joint"
            else:
                certified = True
                progress.last_certified = self._certificate(evidence)
                if (
                    too_easy
                    and progress.joint_rho_index < len(JOINT_RHOS) - 1
                ):
                    progress.joint_probe_index = (
                        progress.joint_rho_index + 1
                    )
                    progress.phase = "joint"
                    progress.joint_epoch += 1
                    kind = "reopen_joint"
                elif progress.joint_rho_index == len(JOINT_RHOS) - 1:
                    kind = "full_domain"
                else:
                    kind = "steady"

        progress.formal_receipts += (
            _Receipt(evidence=evidence, certified=certified),
        )
        return self._decision(
            key,
            progress,
            evidence,
            kind,
            arm_key,
            frontier_before,
            rho_before,
            admit,
            install,
            start,
            close,
            failure,
            unsafe,
            blockers,
        )

    @staticmethod
    def _certificate(evidence: BallDomainEvidence) -> Dict[str, object]:
        return {
            "window_id": evidence.window_id,
            "window_sha256": evidence.window_sha256,
            "seq": evidence.seq,
            "stratum": evidence.stratum,
            "domain_epoch": evidence.domain_epoch,
            "selected_arm_key": evidence.selected_arm_key,
            "selection_round": evidence.selection_round,
            "arm_levels": list(evidence.arm_levels),
            "rho": evidence.rho,
            "policy_contract_sha256": evidence.policy_contract_sha256,
            "policy_checkpoint_sha256": (
                evidence.policy_checkpoint_sha256
            ),
            "policy_generation": evidence.policy_generation,
        }

    def _decision(
        self,
        key: ActionProfileKey,
        progress: _Progress,
        evidence: BallDomainEvidence,
        kind: str,
        arm_key: Optional[str],
        frontier_before: Tuple[float, ...],
        rho_before: float,
        admit: ConfidenceInterval,
        install: ConfidenceInterval,
        start: ConfidenceInterval,
        close: ConfidenceInterval,
        failure: ConfidenceInterval,
        unsafe: ConfidenceInterval,
        blockers: Tuple[str, ...],
    ) -> BallCurriculumDecision:
        domain_after = self._selected_formal_domain_from_progress(progress)
        return BallCurriculumDecision(
            key=key,
            kind=kind,
            evidence_role=evidence.evidence_role,
            stratum=evidence.stratum,
            arm_key=arm_key,
            domain_epoch_before=evidence.domain_epoch,
            domain_epoch_after=(
                evidence.domain_epoch
                if domain_after is None
                else domain_after.domain_epoch
            ),
            frontier_before=frontier_before,
            frontier_after=tuple(
                LEVELS[index]
                for index in progress.arm_frontier_indices
            ),
            rho_before=rho_before,
            rho_after=JOINT_RHOS[progress.joint_rho_index],
            solver_admit=admit,
            install=install,
            start=start,
            close=close,
            policy_failure=failure,
            other_unsafe=unsafe,
            blockers=blockers,
            window_sha256=evidence.window_sha256,
        )

    def state_dict(self) -> Dict[str, object]:
        if self._evaluator_authority is not None:
            self._evaluator_authority.assert_formal_retention(
                self._retained_canary_window_sha256(self._progress)
            )
        authority_state = (
            None
            if self._evaluator_authority is None
            else self._evaluator_authority.state_dict()
        )
        document = {
            "schema_version": STATE_SCHEMA_VERSION,
            "contract_sha256": self._contract_sha256,
            "profile_order": [key.as_dict() for key in self._profile_order],
            "arm_catalog": ARM_CATALOG_DOCUMENT,
            "arm_catalog_sha256": ARM_CATALOG_SHA256,
            "scheduler_config": self._scheduler_config.as_dict(),
            "scheduler_contract_sha256": (
                self._scheduler_contract_sha256
            ),
            "sampler_sha256": self._sampler_sha256,
            "solver_sha256": self._solver_sha256,
            "policy_contract_sha256": self._policy_contract_sha256,
            "config": self._config.as_dict(),
            "evaluator_authority_contract_sha256": (
                None
                if self._evaluator_authority is None
                else self._evaluator_authority.authority_contract_sha256
            ),
            "evaluator_authority_state_owner_sha256": (
                None
                if self._evaluator_authority is None
                else self._evaluator_authority.state_owner_sha256
            ),
            "evaluator_authority_state": authority_state,
            "evaluator_authority_state_sha256": (
                None
                if authority_state is None
                else _canonical_sha256(authority_state)
            ),
            "progress": [
                self._progress_row(key, self._progress[key])
                for key in self._profile_order
            ],
        }
        document["state_sha256"] = _canonical_sha256(document)
        return document

    @staticmethod
    def _progress_row(
        key: ActionProfileKey, progress: _Progress
    ) -> Dict[str, object]:
        all_windows = [
            receipt.evidence
            for receipt in progress.formal_receipts
        ] + [
            receipt.evidence
            for receipt in progress.scheduler_receipts
        ]
        chain = _ZERO_SHA
        for evidence in sorted(all_windows, key=lambda item: item.seq):
            chain = hashlib.sha256(
                (chain + evidence.window_sha256).encode("ascii")
            ).hexdigest()
        return {
            "key": key.as_dict(),
            "phase": progress.phase,
            "arm_frontier_indices": list(
                progress.arm_frontier_indices
            ),
            "arm_status": list(progress.arm_status),
            "arm_probe_indices": list(progress.arm_probe_indices),
            "arm_epochs": list(progress.arm_epochs),
            "selected_arm_key": progress.selected_arm_key,
            "selection_round": progress.selection_round,
            "last_selected_round": list(progress.last_selected_round),
            "center_epoch": progress.center_epoch,
            "joint_epoch": progress.joint_epoch,
            "joint_probe_index": progress.joint_probe_index,
            "joint_rho_index": progress.joint_rho_index,
            "center_failures": progress.center_failures,
            "pending_canary_window_sha256": (
                None
                if progress.pending_canary is None
                else progress.pending_canary.evidence.window_sha256
            ),
            "formal_receipts": [
                receipt.as_dict() for receipt in progress.formal_receipts
            ],
            "scheduler_receipts": [
                receipt.as_dict()
                for receipt in progress.scheduler_receipts
            ],
            "event_hash_chain_sha256": chain,
            "last_certified": progress.last_certified,
        }

    def load_state_dict(self, state: object) -> None:
        top_fields = (
            "schema_version",
            "contract_sha256",
            "profile_order",
            "arm_catalog",
            "arm_catalog_sha256",
            "scheduler_config",
            "scheduler_contract_sha256",
            "sampler_sha256",
            "solver_sha256",
            "policy_contract_sha256",
            "config",
            "evaluator_authority_contract_sha256",
            "evaluator_authority_state_owner_sha256",
            "evaluator_authority_state",
            "evaluator_authority_state_sha256",
            "progress",
            "state_sha256",
        )
        document = _exact_keys(
            state, top_fields, name="action-ball curriculum state"
        )
        digest = _sha256(document["state_sha256"], name="state_sha256")
        unsigned = dict(document)
        del unsigned["state_sha256"]
        if _canonical_sha256(unsigned) != digest:
            raise ValueError("action-ball curriculum state digest mismatch")
        if document["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported action-ball curriculum state schema; "
                "legacy seven-axis checkpoints cannot migrate"
            )
        constants = {
            "contract_sha256": self._contract_sha256,
            "profile_order": [key.as_dict() for key in self._profile_order],
            "arm_catalog": ARM_CATALOG_DOCUMENT,
            "arm_catalog_sha256": ARM_CATALOG_SHA256,
            "scheduler_config": self._scheduler_config.as_dict(),
            "scheduler_contract_sha256": (
                self._scheduler_contract_sha256
            ),
            "sampler_sha256": self._sampler_sha256,
            "solver_sha256": self._solver_sha256,
            "policy_contract_sha256": self._policy_contract_sha256,
            "config": self._config.as_dict(),
        }
        for field, expected in constants.items():
            if document[field] != expected:
                raise ValueError(f"curriculum {field} mismatch")
        rows = document["progress"]
        if not isinstance(rows, list) or len(rows) != len(self._profile_order):
            raise ValueError("progress must align with profile_order")

        replay = ActionBallCurriculum(
            contract_sha256=self._contract_sha256,
            profile_order=self._profile_order,
            sampler_sha256=self._sampler_sha256,
            solver_sha256=self._solver_sha256,
            policy_contract_sha256=self._policy_contract_sha256,
            config=self._config,
            scheduler_config=self._scheduler_config,
        )
        events = []
        persisted_rows = []
        for index, key in enumerate(self._profile_order):
            row = _exact_keys(
                rows[index],
                (
                    "key",
                    "phase",
                    "arm_frontier_indices",
                    "arm_status",
                    "arm_probe_indices",
                    "arm_epochs",
                    "selected_arm_key",
                    "selection_round",
                    "last_selected_round",
                    "center_epoch",
                    "joint_epoch",
                    "joint_probe_index",
                    "joint_rho_index",
                    "center_failures",
                    "pending_canary_window_sha256",
                    "formal_receipts",
                    "scheduler_receipts",
                    "event_hash_chain_sha256",
                    "last_certified",
                ),
                name=f"progress[{index}]",
            )
            if row["key"] != key.as_dict():
                raise ValueError("progress key/order mismatch")
            persisted_rows.append(dict(row))
            formal_raw = row["formal_receipts"]
            scheduler_raw = row["scheduler_receipts"]
            if not isinstance(formal_raw, list) or not isinstance(
                scheduler_raw, list
            ):
                raise ValueError("receipt collections must be lists")
            for item in formal_raw:
                receipt = self._parse_formal_receipt(item)
                events.append(
                    (receipt.evidence.seq, "formal", key, receipt)
                )
            for item in scheduler_raw:
                receipt = self._parse_scheduler_receipt(item)
                events.append(
                    (receipt.evidence.seq, "scheduler", key, receipt)
                )
        if len({seq for seq, _, _, _ in events}) != len(events):
            raise ValueError("checkpoint contains duplicate global seq")
        checkpoint_by_generation: Dict[int, str] = {}
        previous_generation = 0
        for _, kind, key, receipt in sorted(events):
            evidence = receipt.evidence
            if evidence.policy_generation < previous_generation:
                raise ValueError("checkpoint policy generation regressed")
            previous_generation = evidence.policy_generation
            previous = checkpoint_by_generation.setdefault(
                evidence.policy_generation,
                evidence.policy_checkpoint_sha256,
            )
            if previous != evidence.policy_checkpoint_sha256:
                raise ValueError(
                    "checkpoint aliases one policy generation"
                )
            progress = replay._progress[key]
            if kind == "scheduler":
                domains = {
                    domain.stratum: domain
                    for domain in replay._scheduler_domains_from_progress(
                        progress
                    )
                }
                domain = domains.get(evidence.stratum)
                if (
                    domain is None
                    or not replay._domain_matches(evidence, domain)
                ):
                    raise ValueError(
                        "scheduler receipt is not replay-reachable"
                    )
                progress.scheduler_receipts += (receipt,)
                if progress.pending_canary is None:
                    replay._reselect_arm(key, progress)
            else:
                decision = replay._apply_formal_evidence(
                    key, progress, evidence
                )
                if (
                    progress.formal_receipts[-1].certified
                    != receipt.certified
                ):
                    raise ValueError(
                        "formal receipt certification is forged"
                    )
                del decision
        replay_rows = [
            replay._progress_row(key, replay._progress[key])
            for key in self._profile_order
        ]
        if replay_rows != persisted_rows:
            raise ValueError(
                "curriculum state is not reachable by deterministic replay"
            )

        if self._evaluator_authority is None:
            authority_fields = (
                document["evaluator_authority_contract_sha256"],
                document["evaluator_authority_state_owner_sha256"],
                document["evaluator_authority_state"],
                document["evaluator_authority_state_sha256"],
            )
            if authority_fields != (None, None, None, None):
                raise ValueError(
                    "checkpoint requires bound evaluator authority"
                )
        else:
            if (
                document["evaluator_authority_contract_sha256"]
                != self._evaluator_authority.authority_contract_sha256
                or document[
                    "evaluator_authority_state_owner_sha256"
                ]
                != self._evaluator_authority.state_owner_sha256
            ):
                raise ValueError("evaluator authority binding mismatch")
            authority_state = document["evaluator_authority_state"]
            if (
                _canonical_sha256(authority_state)
                != document["evaluator_authority_state_sha256"]
            ):
                raise ValueError("evaluator authority state digest mismatch")
            expected_consumed = tuple(
                receipt.evidence.window_sha256
                for _, _, _, receipt in sorted(events)
            )
            if not isinstance(authority_state, Mapping):
                raise ValueError("authority state must be a mapping")
            raw_consumed = authority_state.get("consumed")
            if not isinstance(raw_consumed, list):
                raise ValueError("authority consumed state must be a list")
            actual_consumed = tuple(
                item.get("window_sha256")
                if isinstance(item, Mapping)
                else None
                for item in raw_consumed
            )
            if actual_consumed != expected_consumed:
                raise ValueError(
                    "authority and curriculum consumed windows differ"
                )
            self._evaluator_authority.load_state_dict(authority_state)
            self._evaluator_authority.assert_formal_retention(
                self._retained_canary_window_sha256(replay._progress)
            )
        self._progress = replay._progress

    def _evidence_from_document(
        self, document: object, window_sha256: object
    ) -> BallDomainEvidence:
        fields = (
            "schema_version",
            "key",
            "arm_catalog_sha256",
            "scheduler_contract_sha256",
            "sampler_sha256",
            "solver_sha256",
            "policy_contract_sha256",
            "policy_checkpoint_sha256",
            "policy_generation",
            "evidence_role",
            "domain_epoch",
            "stratum",
            "selected_arm_key",
            "selection_round",
            "arm_levels",
            "rho",
            "seed_block_start",
            "seed_block_end_exclusive",
            "sample_id_start",
            "sample_id_end_exclusive",
            "sample_receipt_root_sha256",
            "unique_birth_count",
            "birth_receipt_root_sha256",
            "seq",
            "window_id",
            "ledger",
        )
        row = _exact_keys(document, fields, name="evidence document")
        if row["schema_version"] != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("legacy evidence schema is unsupported")
        key_row = _exact_keys(
            row["key"],
            ("action_uid", "profile_sha256", "mobility"),
            name="evidence key",
        )
        ledger_row = _exact_keys(
            row["ledger"],
            tuple(BallOutcomeLedger.__dataclass_fields__),
            name="evidence ledger",
        )
        return BallDomainEvidence(
            key=ActionProfileKey(**key_row),
            arm_catalog_sha256=row["arm_catalog_sha256"],
            scheduler_contract_sha256=row[
                "scheduler_contract_sha256"
            ],
            sampler_sha256=row["sampler_sha256"],
            solver_sha256=row["solver_sha256"],
            policy_contract_sha256=row["policy_contract_sha256"],
            policy_checkpoint_sha256=row[
                "policy_checkpoint_sha256"
            ],
            policy_generation=row["policy_generation"],
            evidence_role=row["evidence_role"],
            domain_epoch=row["domain_epoch"],
            stratum=row["stratum"],
            selected_arm_key=row["selected_arm_key"],
            selection_round=row["selection_round"],
            arm_levels=tuple(row["arm_levels"]),
            rho=row["rho"],
            seed_block_start=row["seed_block_start"],
            seed_block_end_exclusive=row["seed_block_end_exclusive"],
            sample_id_start=row["sample_id_start"],
            sample_id_end_exclusive=row["sample_id_end_exclusive"],
            sample_receipt_root_sha256=row[
                "sample_receipt_root_sha256"
            ],
            unique_birth_count=row["unique_birth_count"],
            birth_receipt_root_sha256=row[
                "birth_receipt_root_sha256"
            ],
            seq=row["seq"],
            window_id=row["window_id"],
            ledger=BallOutcomeLedger(**ledger_row),
            window_sha256=window_sha256,
        )

    def _parse_formal_receipt(self, value: object) -> _Receipt:
        row = _exact_keys(
            value,
            ("evidence", "window_sha256", "certified"),
            name="formal receipt",
        )
        if type(row["certified"]) is not bool:
            raise ValueError("receipt certified must be bool")
        evidence = self._evidence_from_document(
            row["evidence"], row["window_sha256"]
        )
        if evidence.evidence_role == "scheduler":
            raise ValueError("scheduler evidence in formal receipts")
        return _Receipt(evidence=evidence, certified=row["certified"])

    def _parse_scheduler_receipt(
        self, value: object
    ) -> _SchedulerReceipt:
        row = _exact_keys(
            value,
            ("evidence", "window_sha256", "attempts"),
            name="scheduler receipt",
        )
        evidence = self._evidence_from_document(
            row["evidence"], row["window_sha256"]
        )
        if evidence.evidence_role != "scheduler":
            raise ValueError("formal evidence in scheduler receipts")
        raw_attempts = row["attempts"]
        if not isinstance(raw_attempts, list):
            raise ValueError("scheduler attempts must be a list")
        attempts = tuple(
            _validated_attempt_row(
                item, name=f"scheduler receipt attempt[{index}]"
            )
            for index, item in enumerate(raw_attempts)
        )
        if _ledger_from_attempt_rows(attempts) != evidence.ledger:
            raise ValueError("scheduler receipt ledger mismatch")
        return _SchedulerReceipt(evidence=evidence, attempts=attempts)
