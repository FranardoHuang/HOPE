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
from pathlib import PurePosixPath, PureWindowsPath
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
STATE_SCHEMA_VERSION = 10
EVIDENCE_SCHEMA_VERSION = 4
INT64_MAX = (1 << 63) - 1
CANARY_MIN = 256
HELDOUT_MIN = 768
# Schema-4 formal heldout windows reserve 20% of 960 proposals for the
# selected action-axis-side frontier.  Requiring ceil(20% * 768) safe-closed
# frontier rows preserves the same 80% completion floor inside that causal
# slice instead of allowing the 80% center/interior mass to hide an empty
# frontier.
HELDOUT_NEW_BAND_MIN = 154
_ZERO_SHA = "0" * 64
_DRAIN_RESET_MINT_SENTINEL = object()


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


def _relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    if "\\" in value:
        raise ValueError(f"{name} must use POSIX separators")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or value != posix.as_posix()
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise ValueError(f"{name} must be a normalized relative path")
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


DRAIN_RESET_AUTHORITY_CONTRACT_DOCUMENT = {
    "schema_version": 1,
    "kind": "action_ball_drain_reset_authority",
    "launch": "exact code-pinned canonical receipt",
    "source": (
        "code-pinned runtime coordinator reads broker, attempt pool, task "
        "receipt pool, and environment reset state; callers never supply "
        "drain counters or reset counts"
    ),
    "request": (
        "complete pending release set, old published-domain roots and epochs, "
        "canary and heldout roots, and one frozen policy checkpoint"
    ),
    "snapshot": (
        "one shared reset generation across broker/pool/task/env, zero active "
        "attempts/reservations/births/tasks, and an exact N-of-N reset bitmap"
    ),
    "fence": (
        "the coordinator holds a live no-new-work fence from snapshot through "
        "one no-fail publication callback, then records consumption and unlocks"
    ),
    "receipt": "opaque same-process single-use capability",
    "resume": (
        "unconsumed receipts are never serialized and must be re-drained "
        "after restore"
    ),
}
DRAIN_RESET_AUTHORITY_CONTRACT_SHA256 = _canonical_sha256(
    DRAIN_RESET_AUTHORITY_CONTRACT_DOCUMENT
)

# Production remains fail-closed until a reviewed runtime coordinator launch
# receipt is pinned here.  Tests may extend the set only inside their process.
TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256 = frozenset()


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


def canonical_action_profile_key(value: object) -> ActionProfileKey:
    """Validate one equivalent dataclass key without relying on module identity.

    Production imports and dependency-light contract tests can load the same
    curriculum source under different module aliases.  In that case Python
    class identity is deliberately different even though the immutable
    dataclass contract is byte-for-byte the same.  Accept only that exact
    frozen ordered dataclass shape, then reconstruct it through this module's
    constructor so every scalar/type/domain invariant is rechecked locally.
    Plain mappings and duck-typed objects remain invalid API inputs.
    """

    value_type = type(value)
    dataclass_fields = getattr(value_type, "__dataclass_fields__", None)
    dataclass_params = getattr(value_type, "__dataclass_params__", None)
    expected_fields = tuple(ActionProfileKey.__dataclass_fields__)
    if (
        value_type.__name__ != "ActionProfileKey"
        or type(dataclass_fields) is not dict
        or tuple(dataclass_fields) != expected_fields
        or dataclass_params is None
        or dataclass_params.frozen is not True
        or dataclass_params.eq is not True
        or dataclass_params.order is not True
        or type(value) is dict
    ):
        raise ValueError("value must be the exact frozen ActionProfileKey dataclass shape")
    serializer = getattr(value, "as_dict", None)
    if not callable(serializer):
        raise ValueError("ActionProfileKey equivalent must expose as_dict()")
    row = serializer()
    if type(row) is not dict or tuple(row) != expected_fields:
        raise ValueError("ActionProfileKey as_dict() is not canonical")
    if any(getattr(value, field, object()) != row[field] for field in expected_fields):
        raise ValueError("ActionProfileKey fields disagree with as_dict()")
    canonical = ActionProfileKey(**row)
    if canonical.as_dict() != row:
        raise ValueError("ActionProfileKey canonical roundtrip failed")
    return canonical


def drain_reset_launch_receipt_document(
    *,
    curriculum_contract_sha256: str,
    profile_order: Sequence[ActionProfileKey],
    arm_catalog_sha256: str,
    scheduler_contract_sha256: str,
    sampler_sha256: str,
    solver_sha256: str,
    policy_contract_sha256: str,
    runtime_source_contract_sha256: str,
    runtime_source_path: str,
    runtime_source_sha256: str,
    broker_contract_sha256: str,
    attempt_pool_contract_sha256: str,
    task_receipt_pool_contract_sha256: str,
    env_reset_contract_sha256: str,
) -> Dict[str, object]:
    """Build launch-finalizer input; construction alone never authorizes."""

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
    catalog = _sha256(
        arm_catalog_sha256, name="arm_catalog_sha256"
    )
    if catalog != ARM_CATALOG_SHA256:
        raise ValueError("launch arm catalog does not match code")
    return {
        "schema_version": 1,
        "kind": "action_ball_drain_reset_launch",
        "authority_contract_sha256": (
            DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
        ),
        "curriculum_contract_sha256": _sha256(
            curriculum_contract_sha256,
            name="curriculum_contract_sha256",
        ),
        "profile_order": [key.as_dict() for key in order],
        "arm_catalog_sha256": catalog,
        "scheduler_contract_sha256": _sha256(
            scheduler_contract_sha256,
            name="scheduler_contract_sha256",
        ),
        "sampler_sha256": _sha256(
            sampler_sha256, name="sampler_sha256"
        ),
        "solver_sha256": _sha256(
            solver_sha256, name="solver_sha256"
        ),
        "policy_contract_sha256": _sha256(
            policy_contract_sha256,
            name="policy_contract_sha256",
        ),
        "runtime_source_contract_sha256": _sha256(
            runtime_source_contract_sha256,
            name="runtime_source_contract_sha256",
        ),
        "runtime_source_path": _relative_path(
            runtime_source_path, name="runtime_source_path"
        ),
        "runtime_source_sha256": _sha256(
            runtime_source_sha256, name="runtime_source_sha256"
        ),
        "broker_contract_sha256": _sha256(
            broker_contract_sha256, name="broker_contract_sha256"
        ),
        "attempt_pool_contract_sha256": _sha256(
            attempt_pool_contract_sha256,
            name="attempt_pool_contract_sha256",
        ),
        "task_receipt_pool_contract_sha256": _sha256(
            task_receipt_pool_contract_sha256,
            name="task_receipt_pool_contract_sha256",
        ),
        "env_reset_contract_sha256": _sha256(
            env_reset_contract_sha256,
            name="env_reset_contract_sha256",
        ),
    }


class DrainResetAuthorityError(RuntimeError):
    """The runtime drain/reset authority is absent, stale, or inconsistent."""


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
    U_joint_qdes: int = 0
    U_joint_actual: int = 0
    NB: int = 0
    NB_F: int = 0

    def __post_init__(self) -> None:
        for field in self.as_dict():
            _plain_int(getattr(self, field), name=field)
        if not self.P >= self.A >= self.I >= self.S >= self.C:
            raise ValueError("ledger must satisfy P >= A >= I >= S >= C")
        unsafe_closures = self.C - self.L - self.F
        if unsafe_closures < 0:
            raise ValueError(
                "closed outcomes must conserve exactly: "
                "C >= L + F"
            )
        raw_unsafe = (
            self.U_table,
            self.U_fall,
            self.U_collision,
            self.U_joint_qdes,
            self.U_joint_actual,
        )
        if unsafe_closures:
            if max(raw_unsafe) > unsafe_closures:
                raise ValueError(
                    "one raw unsafe channel cannot exceed unique unsafe "
                    "closures C - L - F"
                )
            if sum(raw_unsafe) < unsafe_closures:
                raise ValueError(
                    "every unique unsafe closure needs at least one raw "
                    "sticky safety signal"
                )
        elif any(raw_unsafe):
            raise ValueError(
                "raw unsafe signals require at least one unique unsafe closure"
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
        # Table and joint-limit channels are separate zero-tolerance gates.
        # When they are zero, every unique unsafe closure is necessarily a
        # fall/collision closure.  Returning the unique closure count avoids
        # double-counting coincident raw fall/collision signals.
        return self.C - self.L - self.F

    @property
    def unsafe(self) -> int:
        return self.C - self.L - self.F

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
                "U_joint_qdes",
                "U_joint_actual",
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
    objective_inactive_arms: Tuple[str, ...] = ()

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
        inactive = self.objective_inactive_arms
        if (
            not isinstance(inactive, tuple)
            or len(inactive) != len(set(inactive))
            or any(
                type(arm) is not str or arm not in ARM_KEYS
                for arm in inactive
            )
        ):
            raise ValueError(
                "objective_inactive_arms must be a unique tuple of ARM_KEYS"
            )
        if inactive and set(inactive) != {
            "landing_aim_y_lower",
            "landing_aim_y_upper",
        }:
            raise ValueError(
                "the only reviewed objective arm mask disables both "
                "counter-rally landing-y sides"
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
    def heldout_min_new_band(self) -> int:
        """Code-frozen action-axis-side heldout new-band floor."""

        return HELDOUT_NEW_BAND_MIN

    @property
    def failure_band(self) -> Tuple[float, float]:
        return (
            self.target_failure_rate - self.failure_band_half_width,
            self.target_failure_rate + self.failure_band_half_width,
        )

    def as_dict(self) -> Dict[str, object]:
        result = {
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
        # Preserve every legacy N=5/N=73 config byte.  The extra field exists
        # only for the explicit N=1 counter-rally objective.
        if self.objective_inactive_arms:
            result["objective_inactive_arms"] = list(
                self.objective_inactive_arms
            )
        return result

    def active_arm_keys(self, *, mobility: str) -> Tuple[str, ...]:
        if mobility not in MOBILITIES:
            raise ValueError(f"mobility must be one of {MOBILITIES!r}")
        return tuple(
            arm
            for arm in ARM_KEYS
            if arm not in self.objective_inactive_arms
            and not (
                mobility == "no_move" and arm in BASE_TRAVEL_ARMS
            )
        )


@dataclass(frozen=True)
class ArmSchedulerConfig:
    rolling_window: int = 100
    min_history: int = 20
    forced_every: int = 5
    max_gap_factor: int = 2

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

    def as_dict(self) -> Dict[str, int]:
        return {
            "rolling_window": self.rolling_window,
            "min_history": self.min_history,
            "forced_every": self.forced_every,
            "max_gap_factor": self.max_gap_factor,
        }

    @property
    def contract_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": 5,
                "kind": "action_ball_arm_scheduler",
                "arm_catalog_sha256": ARM_CATALOG_SHA256,
                "config": self.as_dict(),
                "score": {
                    "stage_gates": (
                        "Wilson LCB for A/P, I/A, S/I, C/S"
                    ),
                    "unsafe_gates": (
                        "X=0, U_table=0, U_joint_qdes=0, "
                        "U_joint_actual=0, and point "
                        "(U_fall+U_collision)/C <= configured maximum"
                    ),
                    "objective": "minimize Wilson UCB of F/(L+F)",
                    "window": "latest 100 matching arm-epoch rows only",
                },
                "formal_new_band_gate": {
                    "scope": (
                        "frozen heldout rows for exactly the selected "
                        "action-axis-side frontier arm"
                    ),
                    "minimum_safe_closed": HELDOUT_NEW_BAND_MIN,
                    "verdict": (
                        "Wilson interval of NB_F/NB against the configured "
                        "policy-failure band; never the diluted whole-domain "
                        "F/(L+F)"
                    ),
                    "whole_domain_gates": (
                        "solver admission, install, start, close, other "
                        "unsafe, table, joint-limit, and attribution blockers"
                    ),
                    "scheduler_authority": (
                        "the recent-100 stream only schedules candidate "
                        "collection and never releases a frontier"
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
class _ProgressTarget:
    """Immutable domain-affecting state staged for one atomic release."""

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
    domain_release_epoch: int
    last_certified_json: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "phase": self.phase,
            "arm_frontier_indices": list(self.arm_frontier_indices),
            "arm_status": list(self.arm_status),
            "arm_probe_indices": list(self.arm_probe_indices),
            "arm_epochs": list(self.arm_epochs),
            "selected_arm_key": self.selected_arm_key,
            "selection_round": self.selection_round,
            "last_selected_round": list(self.last_selected_round),
            "center_epoch": self.center_epoch,
            "joint_epoch": self.joint_epoch,
            "joint_probe_index": self.joint_probe_index,
            "joint_rho_index": self.joint_rho_index,
            "center_failures": self.center_failures,
            "domain_release_epoch": self.domain_release_epoch,
            "last_certified_json": self.last_certified_json,
        }


@dataclass(frozen=True)
class PendingDomainRelease:
    """Certified decision waiting for a global pre-reset publish barrier.

    The evaluator's heldout verdict can create this object, but it cannot
    mutate the domain seen by samplers.  Publication is a separate CAS-style
    operation against the exact old state and a process-local barrier token.
    """

    key: ActionProfileKey
    release_id_sha256: str
    from_state_root_sha256: str
    from_domain_epoch: int
    from_levels_sha256: str
    to_state_root_sha256: str
    to_domain_epoch: int
    to_levels_sha256: str
    policy_checkpoint_sha256: str
    policy_generation: int
    canary_window_sha256: str
    heldout_window_sha256: str
    heldout_seq: int
    target: _ProgressTarget

    def __post_init__(self) -> None:
        if not isinstance(self.key, ActionProfileKey):
            raise TypeError("pending release key must be ActionProfileKey")
        for field in (
            "release_id_sha256",
            "from_state_root_sha256",
            "from_levels_sha256",
            "to_state_root_sha256",
            "to_levels_sha256",
            "policy_checkpoint_sha256",
            "canary_window_sha256",
            "heldout_window_sha256",
        ):
            _sha256(getattr(self, field), name=field)
        _plain_int(self.from_domain_epoch, name="from_domain_epoch")
        _plain_int(self.to_domain_epoch, name="to_domain_epoch")
        _plain_int(
            self.policy_generation, name="policy_generation", minimum=1
        )
        _plain_int(self.heldout_seq, name="heldout_seq", minimum=1)
        if self.to_domain_epoch != self.from_domain_epoch + 1:
            raise ValueError("pending release epoch must advance exactly once")
        if not isinstance(self.target, _ProgressTarget):
            raise TypeError("pending release target has invalid type")
        if self.target.domain_release_epoch != self.to_domain_epoch:
            raise ValueError("pending release target epoch mismatch")
        if self.release_id_sha256 != self.compute_release_id_sha256():
            raise ValueError("pending release id does not match contents")

    def _hash_document(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "key": self.key.as_dict(),
            "from_state_root_sha256": self.from_state_root_sha256,
            "from_domain_epoch": self.from_domain_epoch,
            "from_levels_sha256": self.from_levels_sha256,
            "to_state_root_sha256": self.to_state_root_sha256,
            "to_domain_epoch": self.to_domain_epoch,
            "to_levels_sha256": self.to_levels_sha256,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "policy_generation": self.policy_generation,
            "canary_window_sha256": self.canary_window_sha256,
            "heldout_window_sha256": self.heldout_window_sha256,
            "heldout_seq": self.heldout_seq,
            "target": self.target.as_dict(),
        }

    def compute_release_id_sha256(self) -> str:
        return _canonical_sha256(self._hash_document())

    def as_dict(self) -> Dict[str, object]:
        result = self._hash_document()
        result["release_id_sha256"] = self.release_id_sha256
        return result

    @classmethod
    def create(cls, **kwargs: object) -> "PendingDomainRelease":
        target = kwargs.get("target")
        key = kwargs.get("key")
        if not isinstance(target, _ProgressTarget):
            raise TypeError("pending release target has invalid type")
        if not isinstance(key, ActionProfileKey):
            raise TypeError("pending release key has invalid type")
        document = {
            "schema_version": 1,
            "key": key.as_dict(),
            "from_state_root_sha256": kwargs[
                "from_state_root_sha256"
            ],
            "from_domain_epoch": kwargs["from_domain_epoch"],
            "from_levels_sha256": kwargs["from_levels_sha256"],
            "to_state_root_sha256": kwargs["to_state_root_sha256"],
            "to_domain_epoch": kwargs["to_domain_epoch"],
            "to_levels_sha256": kwargs["to_levels_sha256"],
            "policy_checkpoint_sha256": kwargs[
                "policy_checkpoint_sha256"
            ],
            "policy_generation": kwargs["policy_generation"],
            "canary_window_sha256": kwargs[
                "canary_window_sha256"
            ],
            "heldout_window_sha256": kwargs[
                "heldout_window_sha256"
            ],
            "heldout_seq": kwargs["heldout_seq"],
            "target": target.as_dict(),
        }
        return cls(
            release_id_sha256=_canonical_sha256(document),
            **kwargs,
        )


@dataclass(frozen=True)
class GlobalPreResetBarrierToken:
    """Persistent audit document carried by an opaque drain/reset receipt."""

    barrier_serial: int
    authority_contract_sha256: str
    launch_receipt_sha256: str
    runtime_source_contract_sha256: str
    runtime_source_sha256: str
    request_sha256: str
    old_global_state_root_sha256: str
    target_global_state_root_sha256: str
    published_domain_set_root_sha256: str
    release_set_root_sha256: str
    evidence_set_root_sha256: str
    release_ids: Tuple[str, ...]
    policy_checkpoint_sha256: str
    policy_generation: int
    broker_reset_generation: int
    attempt_pool_reset_generation: int
    task_receipt_pool_reset_generation: int
    env_reset_generation: int
    active_attempts: int
    reserved_attempts: int
    active_births: int
    pending_task_receipts: int
    reset_count: int
    env_count: int
    reset_participant_ids: Tuple[int, ...]
    reset_bitmap_sha256: str
    fence_id_sha256: str
    broker_state_root_sha256: str
    attempt_pool_state_root_sha256: str
    task_receipt_pool_state_root_sha256: str
    env_reset_state_root_sha256: str
    snapshot_sha256: str
    token_sha256: str

    def __post_init__(self) -> None:
        _plain_int(self.barrier_serial, name="barrier_serial", minimum=1)
        for field in (
            "authority_contract_sha256",
            "launch_receipt_sha256",
            "runtime_source_contract_sha256",
            "runtime_source_sha256",
            "request_sha256",
            "old_global_state_root_sha256",
            "target_global_state_root_sha256",
            "published_domain_set_root_sha256",
            "release_set_root_sha256",
            "evidence_set_root_sha256",
            "policy_checkpoint_sha256",
            "reset_bitmap_sha256",
            "fence_id_sha256",
            "broker_state_root_sha256",
            "attempt_pool_state_root_sha256",
            "task_receipt_pool_state_root_sha256",
            "env_reset_state_root_sha256",
            "snapshot_sha256",
            "token_sha256",
        ):
            _sha256(getattr(self, field), name=field)
        if (
            self.authority_contract_sha256
            != DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
        ):
            raise ValueError("barrier authority contract does not match code")
        if not self.release_ids:
            raise ValueError("barrier must cover at least one release")
        for index, value in enumerate(self.release_ids):
            _sha256(value, name=f"release_ids[{index}]")
        if len(set(self.release_ids)) != len(self.release_ids):
            raise ValueError("barrier release ids must be unique")
        _plain_int(
            self.policy_generation, name="policy_generation", minimum=1
        )
        for field in (
            "broker_reset_generation",
            "attempt_pool_reset_generation",
            "task_receipt_pool_reset_generation",
            "env_reset_generation",
        ):
            _plain_int(getattr(self, field), name=field, minimum=1)
        if len(
            {
                self.broker_reset_generation,
                self.attempt_pool_reset_generation,
                self.task_receipt_pool_reset_generation,
                self.env_reset_generation,
            }
        ) != 1:
            raise ValueError(
                "broker/pool/task/env must attest one reset generation"
            )
        for field in (
            "active_attempts",
            "reserved_attempts",
            "active_births",
            "pending_task_receipts",
            "reset_count",
            "env_count",
        ):
            _plain_int(getattr(self, field), name=field)
        if any(
            getattr(self, field)
            for field in (
                "active_attempts",
                "reserved_attempts",
                "active_births",
                "pending_task_receipts",
            )
        ):
            raise ValueError("global pre-reset barrier has active work")
        if self.env_count < 1 or self.reset_count != self.env_count:
            raise ValueError("global pre-reset barrier is a partial reset")
        if any(
            type(value) is not int or value < 0 or value > INT64_MAX
            for value in self.reset_participant_ids
        ):
            raise ValueError("reset participant ids must be plain integers")
        if self.reset_participant_ids != tuple(range(self.env_count)):
            raise ValueError(
                "global pre-reset barrier requires exact N-of-N participants"
            )
        expected_bitmap = _canonical_sha256(
            {
                "schema_version": 1,
                "reset_generation": self.env_reset_generation,
                "env_count": self.env_count,
                "reset_participant_ids": list(
                    self.reset_participant_ids
                ),
            }
        )
        if self.reset_bitmap_sha256 != expected_bitmap:
            raise ValueError("global pre-reset reset bitmap mismatch")
        if self.fence_id_sha256 == _ZERO_SHA:
            raise ValueError("drain/reset fence id must not be zero")
        for field in (
            "broker_state_root_sha256",
            "attempt_pool_state_root_sha256",
            "task_receipt_pool_state_root_sha256",
            "env_reset_state_root_sha256",
        ):
            if getattr(self, field) == _ZERO_SHA:
                raise ValueError(f"{field} must not be the zero digest")
        if self.snapshot_sha256 != _canonical_sha256(
            self._snapshot_document()
        ):
            raise ValueError("barrier runtime snapshot digest mismatch")
        if self.token_sha256 != self.compute_token_sha256():
            raise ValueError("barrier token digest mismatch")

    def _snapshot_document(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "action_ball_global_pre_reset_snapshot",
            "request_sha256": self.request_sha256,
            "old_global_state_root_sha256": (
                self.old_global_state_root_sha256
            ),
            "target_global_state_root_sha256": (
                self.target_global_state_root_sha256
            ),
            "published_domain_set_root_sha256": (
                self.published_domain_set_root_sha256
            ),
            "release_set_root_sha256": self.release_set_root_sha256,
            "evidence_set_root_sha256": self.evidence_set_root_sha256,
            "policy_checkpoint_sha256": (
                self.policy_checkpoint_sha256
            ),
            "policy_generation": self.policy_generation,
            "broker_reset_generation": self.broker_reset_generation,
            "attempt_pool_reset_generation": (
                self.attempt_pool_reset_generation
            ),
            "task_receipt_pool_reset_generation": (
                self.task_receipt_pool_reset_generation
            ),
            "env_reset_generation": self.env_reset_generation,
            "active_attempts": self.active_attempts,
            "reserved_attempts": self.reserved_attempts,
            "active_births": self.active_births,
            "pending_task_receipts": self.pending_task_receipts,
            "reset_count": self.reset_count,
            "env_count": self.env_count,
            "reset_participant_ids": list(self.reset_participant_ids),
            "reset_bitmap_sha256": self.reset_bitmap_sha256,
            "fence_id_sha256": self.fence_id_sha256,
            "broker_state_root_sha256": (
                self.broker_state_root_sha256
            ),
            "attempt_pool_state_root_sha256": (
                self.attempt_pool_state_root_sha256
            ),
            "task_receipt_pool_state_root_sha256": (
                self.task_receipt_pool_state_root_sha256
            ),
            "env_reset_state_root_sha256": (
                self.env_reset_state_root_sha256
            ),
        }

    def _hash_document(self) -> Dict[str, object]:
        return {
            "schema_version": 2,
            "barrier_serial": self.barrier_serial,
            "authority_contract_sha256": (
                self.authority_contract_sha256
            ),
            "launch_receipt_sha256": self.launch_receipt_sha256,
            "runtime_source_contract_sha256": (
                self.runtime_source_contract_sha256
            ),
            "runtime_source_sha256": self.runtime_source_sha256,
            "request_sha256": self.request_sha256,
            "old_global_state_root_sha256": (
                self.old_global_state_root_sha256
            ),
            "target_global_state_root_sha256": (
                self.target_global_state_root_sha256
            ),
            "published_domain_set_root_sha256": (
                self.published_domain_set_root_sha256
            ),
            "release_set_root_sha256": self.release_set_root_sha256,
            "evidence_set_root_sha256": self.evidence_set_root_sha256,
            "release_ids": list(self.release_ids),
            "policy_checkpoint_sha256": (
                self.policy_checkpoint_sha256
            ),
            "policy_generation": self.policy_generation,
            "broker_reset_generation": self.broker_reset_generation,
            "attempt_pool_reset_generation": (
                self.attempt_pool_reset_generation
            ),
            "task_receipt_pool_reset_generation": (
                self.task_receipt_pool_reset_generation
            ),
            "env_reset_generation": self.env_reset_generation,
            "active_attempts": self.active_attempts,
            "reserved_attempts": self.reserved_attempts,
            "active_births": self.active_births,
            "pending_task_receipts": self.pending_task_receipts,
            "reset_count": self.reset_count,
            "env_count": self.env_count,
            "reset_participant_ids": list(self.reset_participant_ids),
            "reset_bitmap_sha256": self.reset_bitmap_sha256,
            "fence_id_sha256": self.fence_id_sha256,
            "broker_state_root_sha256": (
                self.broker_state_root_sha256
            ),
            "attempt_pool_state_root_sha256": (
                self.attempt_pool_state_root_sha256
            ),
            "task_receipt_pool_state_root_sha256": (
                self.task_receipt_pool_state_root_sha256
            ),
            "env_reset_state_root_sha256": (
                self.env_reset_state_root_sha256
            ),
            "snapshot_sha256": self.snapshot_sha256,
        }

    def compute_token_sha256(self) -> str:
        return _canonical_sha256(self._hash_document())

    def as_dict(self) -> Dict[str, object]:
        result = self._hash_document()
        result["token_sha256"] = self.token_sha256
        return result

    @classmethod
    def create(cls, **kwargs: object) -> "GlobalPreResetBarrierToken":
        participants = tuple(kwargs["reset_participant_ids"])
        bitmap = _canonical_sha256(
            {
                "schema_version": 1,
                "reset_generation": kwargs["env_reset_generation"],
                "env_count": kwargs["env_count"],
                "reset_participant_ids": list(participants),
            }
        )
        normalized = {
            **kwargs,
            "reset_participant_ids": participants,
            "reset_bitmap_sha256": bitmap,
        }
        # Compute the two self-digests in dependency order; only the returned
        # instance is constructed and validated.
        snapshot_document = {
            "schema_version": 1,
            "kind": "action_ball_global_pre_reset_snapshot",
            "request_sha256": normalized["request_sha256"],
            "old_global_state_root_sha256": normalized[
                "old_global_state_root_sha256"
            ],
            "target_global_state_root_sha256": normalized[
                "target_global_state_root_sha256"
            ],
            "published_domain_set_root_sha256": normalized[
                "published_domain_set_root_sha256"
            ],
            "release_set_root_sha256": normalized[
                "release_set_root_sha256"
            ],
            "evidence_set_root_sha256": normalized[
                "evidence_set_root_sha256"
            ],
            "policy_checkpoint_sha256": normalized[
                "policy_checkpoint_sha256"
            ],
            "policy_generation": normalized["policy_generation"],
            "broker_reset_generation": normalized[
                "broker_reset_generation"
            ],
            "attempt_pool_reset_generation": normalized[
                "attempt_pool_reset_generation"
            ],
            "task_receipt_pool_reset_generation": normalized[
                "task_receipt_pool_reset_generation"
            ],
            "env_reset_generation": normalized["env_reset_generation"],
            "active_attempts": normalized["active_attempts"],
            "reserved_attempts": normalized["reserved_attempts"],
            "active_births": normalized["active_births"],
            "pending_task_receipts": normalized[
                "pending_task_receipts"
            ],
            "reset_count": normalized["reset_count"],
            "env_count": normalized["env_count"],
            "reset_participant_ids": list(participants),
            "reset_bitmap_sha256": bitmap,
            "fence_id_sha256": normalized["fence_id_sha256"],
            "broker_state_root_sha256": normalized[
                "broker_state_root_sha256"
            ],
            "attempt_pool_state_root_sha256": normalized[
                "attempt_pool_state_root_sha256"
            ],
            "task_receipt_pool_state_root_sha256": normalized[
                "task_receipt_pool_state_root_sha256"
            ],
            "env_reset_state_root_sha256": normalized[
                "env_reset_state_root_sha256"
            ],
        }
        snapshot_sha = _canonical_sha256(snapshot_document)
        hash_document = {
            "schema_version": 2,
            "barrier_serial": normalized["barrier_serial"],
            "authority_contract_sha256": normalized[
                "authority_contract_sha256"
            ],
            "launch_receipt_sha256": normalized[
                "launch_receipt_sha256"
            ],
            "runtime_source_contract_sha256": normalized[
                "runtime_source_contract_sha256"
            ],
            "runtime_source_sha256": normalized[
                "runtime_source_sha256"
            ],
            "request_sha256": normalized["request_sha256"],
            "old_global_state_root_sha256": normalized[
                "old_global_state_root_sha256"
            ],
            "target_global_state_root_sha256": normalized[
                "target_global_state_root_sha256"
            ],
            "published_domain_set_root_sha256": normalized[
                "published_domain_set_root_sha256"
            ],
            "release_set_root_sha256": normalized[
                "release_set_root_sha256"
            ],
            "evidence_set_root_sha256": normalized[
                "evidence_set_root_sha256"
            ],
            "release_ids": list(normalized["release_ids"]),
            "policy_checkpoint_sha256": normalized[
                "policy_checkpoint_sha256"
            ],
            "policy_generation": normalized["policy_generation"],
            "broker_reset_generation": normalized[
                "broker_reset_generation"
            ],
            "attempt_pool_reset_generation": normalized[
                "attempt_pool_reset_generation"
            ],
            "task_receipt_pool_reset_generation": normalized[
                "task_receipt_pool_reset_generation"
            ],
            "env_reset_generation": normalized["env_reset_generation"],
            "active_attempts": normalized["active_attempts"],
            "reserved_attempts": normalized["reserved_attempts"],
            "active_births": normalized["active_births"],
            "pending_task_receipts": normalized[
                "pending_task_receipts"
            ],
            "reset_count": normalized["reset_count"],
            "env_count": normalized["env_count"],
            "reset_participant_ids": list(participants),
            "reset_bitmap_sha256": bitmap,
            "fence_id_sha256": normalized["fence_id_sha256"],
            "broker_state_root_sha256": normalized[
                "broker_state_root_sha256"
            ],
            "attempt_pool_state_root_sha256": normalized[
                "attempt_pool_state_root_sha256"
            ],
            "task_receipt_pool_state_root_sha256": normalized[
                "task_receipt_pool_state_root_sha256"
            ],
            "env_reset_state_root_sha256": normalized[
                "env_reset_state_root_sha256"
            ],
            "snapshot_sha256": snapshot_sha,
        }
        return cls(
            snapshot_sha256=snapshot_sha,
            token_sha256=_canonical_sha256(hash_document),
            **normalized,
        )


class DrainResetReceipt:
    """Opaque, same-process, single-use drain/reset authorization."""

    __slots__ = ("_authority", "_lifetime", "_token")

    def __init__(
        self,
        sentinel: object,
        authority: "DrainResetAuthority",
        lifetime: object,
        token: GlobalPreResetBarrierToken,
    ) -> None:
        if sentinel is not _DRAIN_RESET_MINT_SENTINEL:
            raise TypeError(
                "DrainResetReceipt is minted only by its authority"
            )
        self._authority = authority
        self._lifetime = lifetime
        self._token = token

    @property
    def token_sha256(self) -> str:
        return self._token.token_sha256

    @property
    def barrier_serial(self) -> int:
        return self._token.barrier_serial

    def __copy__(self) -> "DrainResetReceipt":
        return self

    def __deepcopy__(self, memo: object) -> "DrainResetReceipt":
        del memo
        return self


class DrainResetAuthority:
    """Code-rooted adapter over a trusted runtime coordinator snapshot."""

    __slots__ = (
        "_launch",
        "_launch_receipt_sha256",
        "_runtime_source",
        "_state_owner_sha256",
        "_lifetime",
        "_pending",
        "_consumed",
    )

    _LAUNCH_FIELDS = (
        "schema_version",
        "kind",
        "authority_contract_sha256",
        "curriculum_contract_sha256",
        "profile_order",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "runtime_source_contract_sha256",
        "runtime_source_path",
        "runtime_source_sha256",
        "broker_contract_sha256",
        "attempt_pool_contract_sha256",
        "task_receipt_pool_contract_sha256",
        "env_reset_contract_sha256",
    )
    _SOURCE_FIELDS = (
        "runtime_source_contract_sha256",
        "runtime_source_path",
        "runtime_source_sha256",
        "broker_contract_sha256",
        "attempt_pool_contract_sha256",
        "task_receipt_pool_contract_sha256",
        "env_reset_contract_sha256",
    )
    _REQUEST_FIELDS = (
        "schema_version",
        "kind",
        "barrier_serial",
        "curriculum_contract_sha256",
        "profile_order",
        "old_global_state_root_sha256",
        "target_global_state_root_sha256",
        "published_domain_set_root_sha256",
        "release_set_root_sha256",
        "evidence_set_root_sha256",
        "release_ids",
        "policy_checkpoint_sha256",
        "policy_generation",
    )
    _SNAPSHOT_FIELDS = (
        "schema_version",
        "kind",
        "request_sha256",
        "old_global_state_root_sha256",
        "target_global_state_root_sha256",
        "published_domain_set_root_sha256",
        "release_set_root_sha256",
        "evidence_set_root_sha256",
        "policy_checkpoint_sha256",
        "policy_generation",
        "broker_reset_generation",
        "attempt_pool_reset_generation",
        "task_receipt_pool_reset_generation",
        "env_reset_generation",
        "active_attempts",
        "reserved_attempts",
        "active_births",
        "pending_task_receipts",
        "reset_count",
        "env_count",
        "reset_participant_ids",
        "reset_bitmap_sha256",
        "fence_id_sha256",
        "broker_state_root_sha256",
        "attempt_pool_state_root_sha256",
        "task_receipt_pool_state_root_sha256",
        "env_reset_state_root_sha256",
    )

    def __init__(
        self,
        *,
        launch: Mapping[str, object],
        launch_receipt_sha256: str,
        runtime_source: object,
    ) -> None:
        self._launch = dict(launch)
        self._launch_receipt_sha256 = launch_receipt_sha256
        self._runtime_source = runtime_source
        self._state_owner_sha256 = _canonical_sha256(
            {
                "schema_version": 1,
                "kind": "action_ball_drain_reset_state_owner",
                "authority_contract_sha256": (
                    DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
                ),
                "launch_receipt_sha256": launch_receipt_sha256,
                "runtime_source_binding": {
                    field: launch[field] for field in self._SOURCE_FIELDS
                },
            }
        )
        self._lifetime = object()
        self._pending: Dict[str, GlobalPreResetBarrierToken] = {}
        self._consumed: Dict[str, GlobalPreResetBarrierToken] = {}

    @classmethod
    def from_trusted_launch_receipt(
        cls,
        launch_receipt: object,
        *,
        runtime_source: object,
    ) -> "DrainResetAuthority":
        row = _exact_keys(
            launch_receipt,
            cls._LAUNCH_FIELDS,
            name="drain/reset launch receipt",
        )
        if (
            row["schema_version"] != 1
            or row["kind"] != "action_ball_drain_reset_launch"
            or row["authority_contract_sha256"]
            != DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
        ):
            raise DrainResetAuthorityError(
                "drain/reset launch receipt does not match code"
            )
        profile_rows = row["profile_order"]
        if not isinstance(profile_rows, list):
            raise DrainResetAuthorityError(
                "drain/reset launch profile_order must be a list"
            )
        keys = []
        for index, item in enumerate(profile_rows):
            key_row = _exact_keys(
                item,
                ("action_uid", "profile_sha256", "mobility"),
                name=f"drain/reset launch profile_order[{index}]",
            )
            keys.append(ActionProfileKey(**key_row))
        normalized = drain_reset_launch_receipt_document(
            curriculum_contract_sha256=row[
                "curriculum_contract_sha256"
            ],
            profile_order=tuple(keys),
            arm_catalog_sha256=row["arm_catalog_sha256"],
            scheduler_contract_sha256=row[
                "scheduler_contract_sha256"
            ],
            sampler_sha256=row["sampler_sha256"],
            solver_sha256=row["solver_sha256"],
            policy_contract_sha256=row["policy_contract_sha256"],
            runtime_source_contract_sha256=row[
                "runtime_source_contract_sha256"
            ],
            runtime_source_path=row["runtime_source_path"],
            runtime_source_sha256=row["runtime_source_sha256"],
            broker_contract_sha256=row["broker_contract_sha256"],
            attempt_pool_contract_sha256=row[
                "attempt_pool_contract_sha256"
            ],
            task_receipt_pool_contract_sha256=row[
                "task_receipt_pool_contract_sha256"
            ],
            env_reset_contract_sha256=row[
                "env_reset_contract_sha256"
            ],
        )
        if dict(row) != normalized:
            raise DrainResetAuthorityError(
                "drain/reset launch receipt is not canonical"
            )
        launch_sha = _canonical_sha256(normalized)
        if launch_sha not in TRUSTED_DRAIN_RESET_LAUNCH_RECEIPT_SHA256:
            raise DrainResetAuthorityError(
                "drain/reset launch receipt is not code-pinned"
            )
        binding_method = getattr(runtime_source, "binding_document", None)
        if not callable(binding_method):
            raise DrainResetAuthorityError(
                "runtime coordinator lacks binding_document"
            )
        source_binding = _exact_keys(
            binding_method(),
            cls._SOURCE_FIELDS,
            name="drain/reset runtime source binding",
        )
        expected_source = {
            field: normalized[field] for field in cls._SOURCE_FIELDS
        }
        if dict(source_binding) != expected_source:
            raise DrainResetAuthorityError(
                "runtime coordinator code/contract binding mismatch"
            )
        return cls(
            launch=normalized,
            launch_receipt_sha256=launch_sha,
            runtime_source=runtime_source,
        )

    @property
    def authority_contract_sha256(self) -> str:
        return DRAIN_RESET_AUTHORITY_CONTRACT_SHA256

    @property
    def release_authorized(self) -> bool:
        return True

    @property
    def launch_receipt_sha256(self) -> str:
        return self._launch_receipt_sha256

    @property
    def state_owner_sha256(self) -> str:
        return self._state_owner_sha256

    def assert_binding(
        self,
        *,
        curriculum_contract_sha256: str,
        profile_order: Sequence[ActionProfileKey],
        arm_catalog_sha256: str,
        scheduler_contract_sha256: str,
        sampler_sha256: str,
        solver_sha256: str,
        policy_contract_sha256: str,
    ) -> None:
        expected = {
            "curriculum_contract_sha256": curriculum_contract_sha256,
            "profile_order": [key.as_dict() for key in profile_order],
            "arm_catalog_sha256": arm_catalog_sha256,
            "scheduler_contract_sha256": scheduler_contract_sha256,
            "sampler_sha256": sampler_sha256,
            "solver_sha256": solver_sha256,
            "policy_contract_sha256": policy_contract_sha256,
        }
        if any(self._launch[field] != value for field, value in expected.items()):
            raise DrainResetAuthorityError(
                "drain/reset authority binding mismatch"
            )

    def state_dict(self) -> Dict[str, object]:
        if self._pending:
            raise DrainResetAuthorityError(
                "cannot checkpoint drain/reset authority with a live fence"
            )
        consumed = sorted(
            self._consumed.values(),
            key=lambda item: item.barrier_serial,
        )
        documents = tuple(item.as_dict() for item in consumed)
        self._assert_source_consumed(documents)
        chain = _ZERO_SHA
        for token in consumed:
            chain = hashlib.sha256(
                (chain + token.token_sha256).encode("ascii")
            ).hexdigest()
        document = {
            "schema_version": 1,
            "authority_contract_sha256": (
                DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
            ),
            "launch_receipt_sha256": self._launch_receipt_sha256,
            "state_owner_sha256": self._state_owner_sha256,
            "consumed": list(documents),
            "consumed_hash_chain_sha256": chain,
        }
        document["state_sha256"] = _canonical_sha256(document)
        return document

    def load_state_dict(
        self,
        state: object,
        *,
        _prevalidated_tokens: Optional[
            Sequence[GlobalPreResetBarrierToken]
        ] = None,
    ) -> None:
        if self._pending:
            raise DrainResetAuthorityError(
                "cannot restore drain/reset authority with a live fence"
            )
        row = _exact_keys(
            state,
            (
                "schema_version",
                "authority_contract_sha256",
                "launch_receipt_sha256",
                "state_owner_sha256",
                "consumed",
                "consumed_hash_chain_sha256",
                "state_sha256",
            ),
            name="drain/reset authority state",
        )
        digest = _sha256(row["state_sha256"], name="state_sha256")
        unsigned = dict(row)
        del unsigned["state_sha256"]
        if _canonical_sha256(unsigned) != digest:
            raise DrainResetAuthorityError(
                "drain/reset authority state digest mismatch"
            )
        if (
            row["schema_version"] != 1
            or row["authority_contract_sha256"]
            != DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
            or row["launch_receipt_sha256"]
            != self._launch_receipt_sha256
            or row["state_owner_sha256"] != self._state_owner_sha256
        ):
            raise DrainResetAuthorityError(
                "drain/reset authority state binding mismatch"
            )
        raw_consumed = row["consumed"]
        if not isinstance(raw_consumed, list):
            raise DrainResetAuthorityError(
                "drain/reset authority consumed must be a list"
            )
        if _prevalidated_tokens is None:
            consumed = [
                ActionBallCurriculum._parse_barrier_token(item)
                for item in raw_consumed
            ]
        else:
            consumed = list(_prevalidated_tokens)
            if (
                any(
                    type(item) is not GlobalPreResetBarrierToken
                    for item in consumed
                )
                or raw_consumed
                != [item.as_dict() for item in consumed]
            ):
                raise DrainResetAuthorityError(
                    "prevalidated drain/reset transcript mismatch"
                )
        if any(
            (
                item.launch_receipt_sha256
                != self._launch_receipt_sha256
                or item.runtime_source_contract_sha256
                != self._launch["runtime_source_contract_sha256"]
                or item.runtime_source_sha256
                != self._launch["runtime_source_sha256"]
            )
            for item in consumed
        ):
            raise DrainResetAuthorityError(
                "consumed token code/source binding mismatch"
            )
        if len({item.token_sha256 for item in consumed}) != len(consumed):
            raise DrainResetAuthorityError(
                "drain/reset authority contains duplicate tokens"
            )
        serials = [item.barrier_serial for item in consumed]
        if serials != sorted(serials) or len(serials) != len(set(serials)):
            raise DrainResetAuthorityError(
                "drain/reset commit serials are not strictly ordered"
            )
        chain = _ZERO_SHA
        for token in consumed:
            chain = hashlib.sha256(
                (chain + token.token_sha256).encode("ascii")
            ).hexdigest()
        if chain != row["consumed_hash_chain_sha256"]:
            raise DrainResetAuthorityError(
                "drain/reset consumed hash chain mismatch"
            )
        documents = tuple(item.as_dict() for item in consumed)
        self._assert_source_consumed(documents)
        self._consumed = {
            item.token_sha256: item for item in consumed
        }

    def _assert_source_consumed(
        self, ordered_token_documents: Sequence[Mapping[str, object]]
    ) -> None:
        checker = getattr(
            self._runtime_source,
            "assert_consumed_drain_reset",
            None,
        )
        if not callable(checker):
            raise DrainResetAuthorityError(
                "runtime coordinator lacks consumed-transcript assertion"
            )
        if checker(tuple(ordered_token_documents)) is not True:
            raise DrainResetAuthorityError(
                "runtime coordinator consumed transcript mismatch"
            )

    def issue(self, request_document: object) -> DrainResetReceipt:
        request = _exact_keys(
            request_document,
            self._REQUEST_FIELDS,
            name="drain/reset release request",
        )
        if (
            request["schema_version"] != 1
            or request["kind"] != "action_ball_domain_release_request"
        ):
            raise DrainResetAuthorityError(
                "unsupported drain/reset request schema"
            )
        self.assert_binding(
            curriculum_contract_sha256=request[
                "curriculum_contract_sha256"
            ],
            profile_order=tuple(
                ActionProfileKey(**item)
                for item in request["profile_order"]
            ),
            arm_catalog_sha256=self._launch["arm_catalog_sha256"],
            scheduler_contract_sha256=self._launch[
                "scheduler_contract_sha256"
            ],
            sampler_sha256=self._launch["sampler_sha256"],
            solver_sha256=self._launch["solver_sha256"],
            policy_contract_sha256=self._launch[
                "policy_contract_sha256"
            ],
        )
        _plain_int(
            request["barrier_serial"],
            name="barrier_serial",
            minimum=1,
        )
        for field in (
            "old_global_state_root_sha256",
            "target_global_state_root_sha256",
            "published_domain_set_root_sha256",
            "release_set_root_sha256",
            "evidence_set_root_sha256",
            "policy_checkpoint_sha256",
        ):
            _sha256(request[field], name=field)
        _plain_int(
            request["policy_generation"],
            name="policy_generation",
            minimum=1,
        )
        release_ids = request["release_ids"]
        if not isinstance(release_ids, list) or not release_ids:
            raise DrainResetAuthorityError(
                "drain/reset request release_ids must be a non-empty list"
            )
        for index, release_id in enumerate(release_ids):
            _sha256(release_id, name=f"release_ids[{index}]")
        if len(release_ids) != len(set(release_ids)):
            raise DrainResetAuthorityError(
                "drain/reset request release ids must be unique"
            )
        detached_request = json.loads(
            json.dumps(
                dict(request),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        request_sha = _canonical_sha256(detached_request)
        capture = getattr(
            self._runtime_source, "capture_drain_reset", None
        )
        if not callable(capture):
            raise DrainResetAuthorityError(
                "runtime coordinator lacks capture_drain_reset"
            )
        snapshot = _exact_keys(
            capture(detached_request),
            self._SNAPSHOT_FIELDS,
            name="runtime drain/reset snapshot",
        )
        if (
            snapshot["schema_version"] != 1
            or snapshot["kind"]
            != "action_ball_global_pre_reset_snapshot"
            or snapshot["request_sha256"] != request_sha
        ):
            raise DrainResetAuthorityError(
                "runtime drain/reset snapshot request mismatch"
            )
        for field in (
            "old_global_state_root_sha256",
            "published_domain_set_root_sha256",
            "release_set_root_sha256",
            "evidence_set_root_sha256",
            "policy_checkpoint_sha256",
            "policy_generation",
        ):
            if snapshot[field] != request[field]:
                raise DrainResetAuthorityError(
                    f"runtime drain/reset snapshot {field} mismatch"
                )
        participants = snapshot["reset_participant_ids"]
        if not isinstance(participants, list):
            raise DrainResetAuthorityError(
                "reset_participant_ids must be a list"
            )
        try:
            token = GlobalPreResetBarrierToken.create(
                barrier_serial=request["barrier_serial"],
                authority_contract_sha256=(
                    DRAIN_RESET_AUTHORITY_CONTRACT_SHA256
                ),
                launch_receipt_sha256=self._launch_receipt_sha256,
                runtime_source_contract_sha256=self._launch[
                    "runtime_source_contract_sha256"
                ],
                runtime_source_sha256=self._launch[
                    "runtime_source_sha256"
                ],
                request_sha256=request_sha,
                old_global_state_root_sha256=request[
                    "old_global_state_root_sha256"
                ],
                target_global_state_root_sha256=request[
                    "target_global_state_root_sha256"
                ],
                published_domain_set_root_sha256=request[
                    "published_domain_set_root_sha256"
                ],
                release_set_root_sha256=request[
                    "release_set_root_sha256"
                ],
                evidence_set_root_sha256=request[
                    "evidence_set_root_sha256"
                ],
                release_ids=tuple(release_ids),
                policy_checkpoint_sha256=request[
                    "policy_checkpoint_sha256"
                ],
                policy_generation=request["policy_generation"],
                broker_reset_generation=snapshot[
                    "broker_reset_generation"
                ],
                attempt_pool_reset_generation=snapshot[
                    "attempt_pool_reset_generation"
                ],
                task_receipt_pool_reset_generation=snapshot[
                    "task_receipt_pool_reset_generation"
                ],
                env_reset_generation=snapshot[
                    "env_reset_generation"
                ],
                active_attempts=snapshot["active_attempts"],
                reserved_attempts=snapshot["reserved_attempts"],
                active_births=snapshot["active_births"],
                pending_task_receipts=snapshot[
                    "pending_task_receipts"
                ],
                reset_count=snapshot["reset_count"],
                env_count=snapshot["env_count"],
                reset_participant_ids=tuple(participants),
                fence_id_sha256=snapshot["fence_id_sha256"],
                broker_state_root_sha256=snapshot[
                    "broker_state_root_sha256"
                ],
                attempt_pool_state_root_sha256=snapshot[
                    "attempt_pool_state_root_sha256"
                ],
                task_receipt_pool_state_root_sha256=snapshot[
                    "task_receipt_pool_state_root_sha256"
                ],
                env_reset_state_root_sha256=snapshot[
                    "env_reset_state_root_sha256"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise DrainResetAuthorityError(
                f"runtime drain/reset snapshot is unsafe: {exc}"
            ) from exc
        if snapshot["reset_bitmap_sha256"] != token.reset_bitmap_sha256:
            raise DrainResetAuthorityError(
                "runtime drain/reset bitmap does not attest exact N-of-N"
            )
        if token.token_sha256 in self._pending:
            raise DrainResetAuthorityError(
                "runtime drain/reset receipt was already issued"
            )
        self._pending[token.token_sha256] = token
        return DrainResetReceipt(
            _DRAIN_RESET_MINT_SENTINEL,
            self,
            self._lifetime,
            token,
        )

    def assert_receipt(
        self, receipt: object
    ) -> GlobalPreResetBarrierToken:
        if type(receipt) is not DrainResetReceipt:
            raise DrainResetAuthorityError(
                "release requires an opaque DrainResetReceipt"
            )
        if (
            receipt._authority is not self
            or receipt._lifetime is not self._lifetime
        ):
            raise DrainResetAuthorityError(
                "drain/reset receipt belongs to another or restored authority"
            )
        token = self._pending.get(receipt._token.token_sha256)
        if token is None or token is not receipt._token:
            raise DrainResetAuthorityError(
                "drain/reset receipt is stale, forged, or consumed"
            )
        return token

    def commit_receipt(
        self,
        receipt: object,
        *,
        publish_noexcept: object,
    ) -> GlobalPreResetBarrierToken:
        """Recheck and publish once while the source still owns its fence."""

        token = self.assert_receipt(receipt)
        if not callable(publish_noexcept):
            raise TypeError("publish_noexcept must be callable")
        commit = getattr(
            self._runtime_source, "commit_drain_reset", None
        )
        if not callable(commit):
            raise DrainResetAuthorityError(
                "runtime coordinator lacks commit_drain_reset"
            )
        calls = 0

        def guarded_publish() -> None:
            nonlocal calls
            if calls:
                raise DrainResetAuthorityError(
                    "runtime coordinator repeated publication callback"
                )
            calls += 1
            publish_noexcept()

        acknowledgement = _exact_keys(
            commit(token.as_dict(), guarded_publish),
            (
                "schema_version",
                "kind",
                "token_sha256",
                "fence_id_sha256",
                "published",
            ),
            name="drain/reset commit acknowledgement",
        )
        if (
            acknowledgement["schema_version"] != 1
            or acknowledgement["kind"]
            != "action_ball_drain_reset_commit"
            or acknowledgement["token_sha256"] != token.token_sha256
            or acknowledgement["fence_id_sha256"]
            != token.fence_id_sha256
            or acknowledgement["published"] is not True
            or calls != 1
        ):
            raise DrainResetAuthorityError(
                "runtime coordinator violated fenced publication contract"
            )
        del self._pending[token.token_sha256]
        self._consumed[token.token_sha256] = token
        return token

    def abort_receipt(self, receipt: object) -> None:
        token = self.assert_receipt(receipt)
        abort = getattr(self._runtime_source, "abort_drain_reset", None)
        if not callable(abort):
            raise DrainResetAuthorityError(
                "runtime coordinator lacks abort_drain_reset"
            )
        acknowledgement = _exact_keys(
            abort(token.as_dict()),
            (
                "schema_version",
                "kind",
                "token_sha256",
                "fence_id_sha256",
                "aborted",
            ),
            name="drain/reset abort acknowledgement",
        )
        if (
            acknowledgement["schema_version"] != 1
            or acknowledgement["kind"]
            != "action_ball_drain_reset_abort"
            or acknowledgement["token_sha256"] != token.token_sha256
            or acknowledgement["fence_id_sha256"]
            != token.fence_id_sha256
            or acknowledgement["aborted"] is not True
        ):
            raise DrainResetAuthorityError(
                "runtime coordinator violated fenced abort contract"
            )
        del self._pending[token.token_sha256]


@dataclass(frozen=True)
class DomainReleaseReceipt:
    key: ActionProfileKey
    release_id_sha256: str
    barrier_token: GlobalPreResetBarrierToken
    commit_serial: int
    from_state_root_sha256: str
    from_domain_epoch: int
    from_levels_sha256: str
    to_state_root_sha256: str
    to_domain_epoch: int
    to_levels_sha256: str
    policy_checkpoint_sha256: str
    policy_generation: int
    canary_window_sha256: str
    heldout_window_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ActionProfileKey):
            raise TypeError("release receipt key must be ActionProfileKey")
        for field in (
            "release_id_sha256",
            "from_state_root_sha256",
            "from_levels_sha256",
            "to_state_root_sha256",
            "to_levels_sha256",
            "policy_checkpoint_sha256",
            "canary_window_sha256",
            "heldout_window_sha256",
            "receipt_sha256",
        ):
            _sha256(getattr(self, field), name=field)
        if not isinstance(
            self.barrier_token, GlobalPreResetBarrierToken
        ):
            raise TypeError("release receipt barrier token has invalid type")
        _plain_int(self.commit_serial, name="commit_serial", minimum=1)
        _plain_int(self.from_domain_epoch, name="from_domain_epoch")
        _plain_int(self.to_domain_epoch, name="to_domain_epoch")
        _plain_int(
            self.policy_generation, name="policy_generation", minimum=1
        )
        if self.to_domain_epoch != self.from_domain_epoch + 1:
            raise ValueError("release receipt epoch must advance exactly once")
        if self.receipt_sha256 != self.compute_receipt_sha256():
            raise ValueError("release receipt digest mismatch")

    def _hash_document(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "key": self.key.as_dict(),
            "release_id_sha256": self.release_id_sha256,
            "barrier_token": self.barrier_token.as_dict(),
            "commit_serial": self.commit_serial,
            "from_state_root_sha256": self.from_state_root_sha256,
            "from_domain_epoch": self.from_domain_epoch,
            "from_levels_sha256": self.from_levels_sha256,
            "to_state_root_sha256": self.to_state_root_sha256,
            "to_domain_epoch": self.to_domain_epoch,
            "to_levels_sha256": self.to_levels_sha256,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "policy_generation": self.policy_generation,
            "canary_window_sha256": self.canary_window_sha256,
            "heldout_window_sha256": self.heldout_window_sha256,
        }

    def compute_receipt_sha256(self) -> str:
        return _canonical_sha256(self._hash_document())

    @property
    def barrier_token_sha256(self) -> str:
        return self.barrier_token.token_sha256

    def as_dict(self) -> Dict[str, object]:
        result = self._hash_document()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    @classmethod
    def create(cls, **kwargs: object) -> "DomainReleaseReceipt":
        key = kwargs.get("key")
        if not isinstance(key, ActionProfileKey):
            raise TypeError("release receipt key has invalid type")
        document = {
            "schema_version": 1,
            "key": key.as_dict(),
            "release_id_sha256": kwargs["release_id_sha256"],
            "barrier_token": kwargs["barrier_token"].as_dict(),
            "commit_serial": kwargs["commit_serial"],
            "from_state_root_sha256": kwargs[
                "from_state_root_sha256"
            ],
            "from_domain_epoch": kwargs["from_domain_epoch"],
            "from_levels_sha256": kwargs["from_levels_sha256"],
            "to_state_root_sha256": kwargs["to_state_root_sha256"],
            "to_domain_epoch": kwargs["to_domain_epoch"],
            "to_levels_sha256": kwargs["to_levels_sha256"],
            "policy_checkpoint_sha256": kwargs[
                "policy_checkpoint_sha256"
            ],
            "policy_generation": kwargs["policy_generation"],
            "canary_window_sha256": kwargs[
                "canary_window_sha256"
            ],
            "heldout_window_sha256": kwargs[
                "heldout_window_sha256"
            ],
        }
        return cls(
            receipt_sha256=_canonical_sha256(document),
            **kwargs,
        )


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
_ATTEMPT_FIELDS_V4 = _ATTEMPT_FIELDS + ("terminal_signals",)
_TERMINALS = (
    "legal_return",
    "safe_nonreturn",
    "table_hit",
    "fall",
    "collision",
    "joint_qdes_limit",
    "joint_actual_limit",
)
_NEW_BAND_SAFE_TERMINALS = ("legal_return", "safe_nonreturn")
_TERMINAL_SIGNAL_KEYS = (
    "infrastructure_invalid",
    "joint_actual_limit",
    "joint_qdes_limit",
    "fall",
    "table_hit",
    "collision",
    "legal_return",
)


def _new_band_safe_closed_eligible(
    row: Mapping[str, object],
) -> bool:
    """Whether one attempt is safe-closed evidence in the new band."""

    return bool(
        row["in_new_band"]
        and row["closed"]
        and not row["infrastructure_invalid"]
        and row["terminal_outcome"] in _NEW_BAND_SAFE_TERMINALS
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


def _classify_terminal_signals(
    signals: Mapping[str, bool],
) -> Optional[str]:
    """Apply the frozen runtime's primary-outcome precedence to raw signals."""

    if signals["infrastructure_invalid"]:
        return None
    if signals["joint_actual_limit"]:
        return "joint_actual_limit"
    if signals["joint_qdes_limit"]:
        return "joint_qdes_limit"
    if signals["fall"]:
        return "fall"
    if signals["table_hit"]:
        return "table_hit"
    if signals["collision"]:
        return "collision"
    if signals["legal_return"]:
        return "legal_return"
    return "safe_nonreturn"


def _validated_attempt_row_v4(
    value: object, *, name: str
) -> Dict[str, object]:
    """Retain schema-4 raw sticky safety signals for exact scheduler replay."""

    row = _exact_keys(value, _ATTEMPT_FIELDS_V4, name=name)
    normalized = _validated_attempt_row(
        {field: row[field] for field in _ATTEMPT_FIELDS},
        name=name,
    )
    raw_signals = row["terminal_signals"]
    if raw_signals is None:
        signals = None
    else:
        signal_row = _exact_keys(
            raw_signals,
            _TERMINAL_SIGNAL_KEYS,
            name=f"{name}.terminal_signals",
        )
        signals = {}
        for field in _TERMINAL_SIGNAL_KEYS:
            if type(signal_row[field]) is not bool:
                raise ValueError(
                    f"{name}.terminal_signals.{field} must be bool"
                )
            signals[field] = signal_row[field]
        if (
            signals["infrastructure_invalid"]
            != normalized["infrastructure_invalid"]
        ):
            raise ValueError(
                f"{name} infrastructure flag differs from raw signals"
            )
        if normalized["closed"]:
            if (
                signals["infrastructure_invalid"]
                or _classify_terminal_signals(signals)
                != normalized["terminal_outcome"]
            ):
                raise ValueError(
                    f"{name} terminal outcome differs from raw signals"
                )
        elif normalized["infrastructure_invalid"]:
            if any(
                signals[field]
                for field in _TERMINAL_SIGNAL_KEYS
                if field != "infrastructure_invalid"
            ):
                raise ValueError(
                    f"{name} infrastructure burn has physical signals"
                )
        elif any(signals.values()):
            raise ValueError(
                f"{name} unsettled/rejected row has terminal signals"
            )
    if normalized["closed"] and signals is None:
        raise ValueError(f"{name} closed row needs raw terminal signals")
    if (
        signals is None
        and normalized["infrastructure_invalid"]
        and normalized["closed"]
    ):
        raise ValueError(
            f"{name} closed infrastructure row is not representable"
        )
    normalized["terminal_signals"] = signals
    return normalized


def _ledger_from_attempt_rows(
    attempts: Sequence[Mapping[str, object]],
) -> BallOutcomeLedger:
    terminals = {name: 0 for name in _TERMINALS}
    admitted = 0
    installed = 0
    started = 0
    closed = 0
    infrastructure_invalid = 0
    raw_signals = {
        "table_hit": 0,
        "fall": 0,
        "collision": 0,
        "joint_qdes_limit": 0,
        "joint_actual_limit": 0,
    }
    new_band = 0
    new_band_failures = 0
    for attempt in attempts:
        terminal = attempt["terminal_outcome"]
        if terminal is not None:
            terminals[terminal] += 1
        admitted += bool(attempt["solver_admitted"])
        installed += bool(attempt["installed"])
        started += bool(attempt["started"])
        closed += bool(attempt["closed"])
        infrastructure_invalid += bool(
            attempt["infrastructure_invalid"]
        )
        signals = attempt.get("terminal_signals")
        if signals is None:
            for signal in raw_signals:
                raw_signals[signal] += terminal == signal
        else:
            for signal in raw_signals:
                raw_signals[signal] += bool(signals[signal])
        if _new_band_safe_closed_eligible(attempt):
            new_band += 1
            if terminal == "safe_nonreturn":
                new_band_failures += 1

    return BallOutcomeLedger(
        P=len(attempts),
        A=admitted,
        I=installed,
        S=started,
        C=closed,
        L=terminals["legal_return"],
        F=terminals["safe_nonreturn"],
        U_table=raw_signals["table_hit"],
        U_fall=raw_signals["fall"],
        U_collision=raw_signals["collision"],
        X=infrastructure_invalid,
        U_joint_qdes=raw_signals["joint_qdes_limit"],
        U_joint_actual=raw_signals["joint_actual_limit"],
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
    domain_release_epoch: int
    pending_canary: Optional[_Receipt]
    pending_release: Optional[PendingDomainRelease]
    release_receipts: Tuple[DomainReleaseReceipt, ...]
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
            release_receipts=tuple(self.release_receipts),
            formal_receipts=tuple(self.formal_receipts),
            scheduler_receipts=tuple(self.scheduler_receipts),
            last_certified=(
                None
                if self.last_certified is None
                else dict(self.last_certified)
            ),
        )


# ``BallCurriculumStalledError`` was deleted 2026-08-06.  It was declared here
# and then never raised, never caught and never imported -- an exception type
# for a failure mode nothing reports.  The curriculum's live fail-closed paths
# raise ``DrainResetAuthorityError`` / ``ValueError`` instead.


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
        drain_reset_authority: object | None = None,
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
        self._next_barrier_serial = 1
        self._issued_barriers: Dict[
            str, DrainResetReceipt
        ] = {}
        self._evaluator_authority: object | None = None
        self._drain_reset_authority: DrainResetAuthority | None = None
        if evaluator_authority is not None:
            self.bind_evaluator_authority(evaluator_authority)
        if drain_reset_authority is not None:
            self.bind_drain_reset_authority(drain_reset_authority)

    def _new_progress(self, key: ActionProfileKey) -> _Progress:
        enabled_arms = self._config.active_arm_keys(
            mobility=key.mobility
        )
        status = tuple(
            "pending" if arm in enabled_arms else "disabled"
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
            domain_release_epoch=0,
            pending_canary=None,
            pending_release=None,
            release_receipts=(),
            formal_receipts=(),
            scheduler_receipts=(),
            last_certified=None,
        )

    @staticmethod
    def _target_from_progress(progress: _Progress) -> _ProgressTarget:
        last_certified_json = (
            None
            if progress.last_certified is None
            else json.dumps(
                progress.last_certified,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return _ProgressTarget(
            phase=progress.phase,
            arm_frontier_indices=tuple(progress.arm_frontier_indices),
            arm_status=tuple(progress.arm_status),
            arm_probe_indices=tuple(progress.arm_probe_indices),
            arm_epochs=tuple(progress.arm_epochs),
            selected_arm_key=progress.selected_arm_key,
            selection_round=progress.selection_round,
            last_selected_round=tuple(progress.last_selected_round),
            center_epoch=progress.center_epoch,
            joint_epoch=progress.joint_epoch,
            joint_probe_index=progress.joint_probe_index,
            joint_rho_index=progress.joint_rho_index,
            center_failures=progress.center_failures,
            domain_release_epoch=progress.domain_release_epoch,
            last_certified_json=last_certified_json,
        )

    @staticmethod
    def _apply_target(
        progress: _Progress, target: _ProgressTarget
    ) -> None:
        progress.phase = target.phase
        progress.arm_frontier_indices = target.arm_frontier_indices
        progress.arm_status = target.arm_status
        progress.arm_probe_indices = target.arm_probe_indices
        progress.arm_epochs = target.arm_epochs
        progress.selected_arm_key = target.selected_arm_key
        progress.selection_round = target.selection_round
        progress.last_selected_round = target.last_selected_round
        progress.center_epoch = target.center_epoch
        progress.joint_epoch = target.joint_epoch
        progress.joint_probe_index = target.joint_probe_index
        progress.joint_rho_index = target.joint_rho_index
        progress.center_failures = target.center_failures
        progress.domain_release_epoch = target.domain_release_epoch
        progress.last_certified = (
            None
            if target.last_certified_json is None
            else json.loads(target.last_certified_json)
        )

    @classmethod
    def _published_state_document(
        cls, key: ActionProfileKey, progress: _Progress
    ) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "key": key.as_dict(),
            "target": cls._target_from_progress(progress).as_dict(),
        }

    @classmethod
    def _published_state_root(
        cls, key: ActionProfileKey, progress: _Progress
    ) -> str:
        return _canonical_sha256(
            cls._published_state_document(key, progress)
        )

    @staticmethod
    def _domain_levels_sha256(
        domain: Optional[ExpectedDomain],
        *,
        release_epoch: int,
    ) -> str:
        return _canonical_sha256(
            {
                "schema_version": 1,
                "release_epoch": release_epoch,
                "domain": (
                    None
                    if domain is None
                    else {
                        "stratum": domain.stratum,
                        "selected_arm_key": domain.selected_arm_key,
                        "selection_round": domain.selection_round,
                        "arm_levels": list(domain.arm_levels),
                        "rho": domain.rho,
                    }
                ),
            }
        )

    def _global_published_state_root(
        self,
        progress_by_key: Optional[
            Mapping[ActionProfileKey, _Progress]
        ] = None,
    ) -> str:
        source = self._progress if progress_by_key is None else progress_by_key
        return _canonical_sha256(
            {
                "schema_version": 1,
                "profile_order": [
                    {
                        "key": key.as_dict(),
                        "published_state_root_sha256": (
                            self._published_state_root(key, source[key])
                        ),
                    }
                    for key in self._profile_order
                ],
            }
        )

    def pending_domain_release(
        self, key: ActionProfileKey
    ) -> Optional[PendingDomainRelease]:
        return self._progress[key].pending_release

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

    @staticmethod
    def _evaluation_v4_types() -> Tuple[type, type, type, type]:
        try:
            from .action_ball_evaluation import (
                FrozenEvaluationAuthorityError,
                FrozenEvaluationCapabilityV4,
                FrozenEvaluationReleaseReceipt,
                FrozenEvaluatorV4Authority,
            )
        except ImportError:
            from action_ball_evaluation import (  # type: ignore
                FrozenEvaluationAuthorityError,
                FrozenEvaluationCapabilityV4,
                FrozenEvaluationReleaseReceipt,
                FrozenEvaluatorV4Authority,
            )
        return (
            FrozenEvaluatorV4Authority,
            FrozenEvaluationCapabilityV4,
            FrozenEvaluationReleaseReceipt,
            FrozenEvaluationAuthorityError,
        )

    def bind_evaluator_authority(self, authority: object) -> None:
        legacy_type, _, authority_error = self._evaluation_types()
        v4_type, _, _, _ = self._evaluation_v4_types()
        if self._evaluator_authority is not None:
            raise authority_error(
                "frozen evaluator authority may be bound only once"
            )
        if type(authority) not in (legacy_type, v4_type):
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

    def bind_drain_reset_authority(self, authority: object) -> None:
        if self._drain_reset_authority is not None:
            raise DrainResetAuthorityError(
                "drain/reset authority may be bound only once"
            )
        if type(authority) is not DrainResetAuthority:
            raise DrainResetAuthorityError(
                "release requires exact DrainResetAuthority"
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
        if authority.release_authorized is not True:
            raise DrainResetAuthorityError(
                "drain/reset authority is not release-authorized"
            )
        self._drain_reset_authority = authority

    @property
    def release_authorized(self) -> bool:
        return bool(
            self._drain_reset_authority is not None
            and self._drain_reset_authority.release_authorized is True
        )

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
                epoch=progress.domain_release_epoch,
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
                epoch=progress.domain_release_epoch,
                arm_levels=tuple(levels),
                rho=0.0,
                selected_arm_key=progress.selected_arm_key,
            )
        if progress.phase == "joint":
            rho = JOINT_RHOS[progress.joint_probe_index]
            return self._domain(
                progress,
                stratum="joint",
                epoch=progress.domain_release_epoch,
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
                epoch=progress.domain_release_epoch,
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
        if (
            progress.phase != "marginal"
            or progress.pending_canary
            or progress.pending_release
        ):
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

    def _assert_v4_authority_alignment(
        self,
        progress_by_key: Mapping[ActionProfileKey, _Progress],
        authority_state: Mapping[str, object],
    ) -> None:
        """Cross-check exact V4 authority tapes against curriculum receipts."""

        raw_windows = authority_state.get("windows")
        raw_consumed_releases = authority_state.get("consumed_releases")
        raw_consumed_scheduler = authority_state.get(
            "consumed_scheduler_capabilities"
        )
        if (
            not isinstance(raw_windows, list)
            or not isinstance(raw_consumed_releases, list)
            or not isinstance(raw_consumed_scheduler, list)
        ):
            raise ValueError("V4 authority state lacks exact receipt sets")
        capability_by_window = {}
        role_by_capability = {}
        for index, raw in enumerate(raw_windows):
            if not isinstance(raw, Mapping):
                raise ValueError(f"V4 window[{index}] must be a mapping")
            evidence_document = raw.get("evidence")
            capability_id = raw.get("capability_id")
            if evidence_document is None or capability_id is None:
                continue
            capability_sha = _sha256(
                capability_id,
                name=f"V4 window[{index}].capability_id",
            )
            window_sha = _canonical_sha256(evidence_document)
            if window_sha in capability_by_window:
                raise ValueError("V4 authority repeats one evidence window")
            capability_by_window[window_sha] = capability_sha
            role_by_capability[capability_sha] = raw.get("evidence_role")

        scheduler_capabilities = set()
        expected_release_ids = set()
        authority_owner = _sha256(
            self._evaluator_authority.state_owner_sha256,
            name="V4 evaluator state_owner_sha256",
        )
        for key in self._profile_order:
            progress = progress_by_key[key]
            for receipt in progress.scheduler_receipts:
                try:
                    capability = capability_by_window[
                        receipt.evidence.window_sha256
                    ]
                except KeyError as exc:
                    raise ValueError(
                        "curriculum scheduler window is absent from V4 tape"
                    ) from exc
                if role_by_capability.get(capability) != "scheduler":
                    raise ValueError(
                        "curriculum scheduler window has wrong V4 role"
                    )
                scheduler_capabilities.add(capability)
            formal = progress.formal_receipts
            if len(formal) % 2:
                raise ValueError(
                    "V4 curriculum formal receipts are not complete pairs"
                )
            for offset in range(0, len(formal), 2):
                canary = formal[offset].evidence
                heldout = formal[offset + 1].evidence
                if (
                    canary.evidence_role != "frozen_canary"
                    or heldout.evidence_role != "frozen_heldout"
                ):
                    raise ValueError(
                        "V4 curriculum formal receipt order is invalid"
                    )
                for evidence in (canary, heldout):
                    if (
                        evidence.window_sha256
                        not in capability_by_window
                    ):
                        raise ValueError(
                            "curriculum formal window is absent from V4 tape"
                        )
                expected_release_ids.add(
                    _canonical_sha256(
                        {
                            "schema_version": 4,
                            "kind": (
                                "action_ball_frozen_evaluation_release"
                            ),
                            "state_owner_sha256": authority_owner,
                            "policy_checkpoint_sha256": (
                                canary.policy_checkpoint_sha256
                            ),
                            "policy_generation": (
                                canary.policy_generation
                            ),
                            "canary_window_sha256": (
                                canary.window_sha256
                            ),
                            "heldout_window_sha256": (
                                heldout.window_sha256
                            ),
                            "optional_stopping": False,
                        }
                    )
                )
        consumed_scheduler = {
            _sha256(value, name="V4 consumed scheduler capability")
            for value in raw_consumed_scheduler
        }
        consumed_releases = {
            _sha256(value, name="V4 consumed release")
            for value in raw_consumed_releases
        }
        if consumed_scheduler != scheduler_capabilities:
            raise ValueError(
                "V4 authority and curriculum scheduler receipts differ"
            )
        if consumed_releases != expected_release_ids:
            raise ValueError(
                "V4 authority and curriculum release receipts differ"
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

    def _observe_scheduler_legacy_for_test(
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
                    {
                        field: row[field]
                        for field in _ATTEMPT_FIELDS
                    },
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

    def observe_scheduler(
        self,
        capability_by_profile: Mapping[ActionProfileKey, object],
    ) -> Tuple[str, ...]:
        """Consume exact schema-4 scheduler transcripts for arm selection."""

        (
            v4_authority_type,
            scheduler_capability_type,
            _,
            authority_error,
        ) = self._evaluation_v4_types()
        if (
            self._evaluator_authority is None
            or type(self._evaluator_authority) is not v4_authority_type
        ):
            raise authority_error(
                "scheduler ingest requires schema-4 evaluator authority"
            )
        if not isinstance(capability_by_profile, Mapping) or not (
            capability_by_profile
        ) or any(
            type(value) is not scheduler_capability_type
            for value in capability_by_profile.values()
        ):
            raise authority_error(
                "scheduler ingest accepts only opaque schema-4 capabilities"
            )
        exact = (
            self._evaluator_authority.assert_scheduler_capabilities_many(
                capability_by_profile
            )
        )
        evidence_by_profile = {
            key: value[0] for key, value in exact.items()
        }
        self._validate_global_batch(evidence_by_profile)
        staged = {
            key: progress.clone()
            for key, progress in self._progress.items()
        }
        selected = []
        for key in self._profile_order:
            if key not in exact:
                continue
            evidence, raw_rows = exact[key]
            self._validate_common(key, evidence)
            if evidence.evidence_role != "scheduler":
                raise ValueError("scheduler evidence role mismatch")
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
                _validated_attempt_row_v4(
                    {
                        field: row[field]
                        for field in _ATTEMPT_FIELDS_V4
                    },
                    name=f"scheduler[{key.action_uid}][{index}]",
                )
                for index, row in enumerate(raw_rows)
            )
            if _ledger_from_attempt_rows(rows) != evidence.ledger:
                raise ValueError(
                    "schema-4 scheduler transcript ledger mismatch"
                )
            staged[key].scheduler_receipts += (
                _SchedulerReceipt(evidence=evidence, attempts=rows),
            )
            if (
                staged[key].pending_canary is None
                and staged[key].pending_release is None
            ):
                self._reselect_arm(key, staged[key])
            selected.append(staged[key].selected_arm_key)
        consumed = (
            self._evaluator_authority.consume_scheduler_capabilities_many(
                capability_by_profile
            )
        )
        if consumed != exact:
            raise authority_error(
                "schema-4 evaluator changed during scheduler ingest"
            )
        self._progress = staged
        return tuple(selected)

    def _scheduler_domains_from_progress(
        self, progress: _Progress
    ) -> Tuple[ExpectedDomain, ...]:
        if (
            progress.phase != "marginal"
            or progress.pending_canary
            or progress.pending_release
        ):
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
        return self._recent_ledgers(progress, (arm_index,))[arm_index]

    def _recent_ledgers(
        self,
        progress: _Progress,
        arm_indices: Sequence[int],
    ) -> Dict[int, BallOutcomeLedger]:
        """Build all requested rolling ledgers in one receipt scan.

        Replaying one scheduler window reselects across every probing arm.
        Scanning the full receipt tuple once per arm made exact-resume cost
        quadratic in the arm count despite every receipt belonging to only
        one stratum.  The grouped scan below retains the exact receipt/attempt
        order and the same per-arm 100-row suffix.
        """

        indices = tuple(arm_indices)
        rows_by_index = {index: [] for index in indices}
        stratum_to_index = {
            f"marginal:{ARM_KEYS[index]}": index for index in indices
        }
        for receipt in progress.scheduler_receipts:
            evidence = receipt.evidence
            index = stratum_to_index.get(evidence.stratum)
            if (
                index is not None
                and evidence.domain_epoch == progress.arm_epochs[index]
                and evidence.arm_levels[index]
                == LEVELS[progress.arm_probe_indices[index]]
            ):
                rows_by_index[index].extend(receipt.attempts)
        window = self._scheduler_config.rolling_window
        return {
            index: _ledger_from_attempt_rows(rows[-window:])
            for index, rows in rows_by_index.items()
        }

    def _scheduler_eligible(
        self, ledger: BallOutcomeLedger
    ) -> bool:
        cfg = self._config
        if ledger.P < self._scheduler_config.min_history:
            return False
        if (
            ledger.X
            or ledger.U_table
            or ledger.U_joint_qdes
            or ledger.U_joint_actual
        ):
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
        if (
            progress.phase != "marginal"
            or progress.pending_canary
            or progress.pending_release
        ):
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
        ledgers = self._recent_ledgers(progress, eligible_indices)
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

    def _stage_legacy_capabilities_for_test(
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

    def stage_selected(
        self,
        release_receipt_by_profile: Mapping[ActionProfileKey, object],
    ) -> Tuple[BallCurriculumDecision, ...]:
        """Stage schema-4 canary+heldout decisions without publishing them."""

        (
            v4_authority_type,
            _,
            release_receipt_type,
            authority_error,
        ) = self._evaluation_v4_types()
        if (
            self._evaluator_authority is None
            or type(self._evaluator_authority) is not v4_authority_type
        ):
            raise authority_error(
                "formal release requires bound schema-4 evaluator authority"
            )
        if not isinstance(release_receipt_by_profile, Mapping) or not (
            release_receipt_by_profile
        ) or any(
            type(value) is not release_receipt_type
            for value in release_receipt_by_profile.values()
        ):
            raise authority_error(
                "formal release accepts only opaque schema-4 receipts"
            )
        evidence_pairs = (
            self._evaluator_authority.assert_release_receipts_many(
                release_receipt_by_profile
            )
        )
        existing_pending = {
            key: progress.pending_release
            for key, progress in self._progress.items()
            if progress.pending_release is not None
        }
        if set(existing_pending).intersection(evidence_pairs):
            raise ValueError(
                "profile already has a pending domain release"
            )
        if existing_pending:
            pending_snapshots = {
                (
                    pending.policy_generation,
                    pending.policy_checkpoint_sha256,
                )
                for pending in existing_pending.values()
            }
            if len(pending_snapshots) != 1:
                raise ValueError(
                    "pending domain releases have mixed policy snapshots"
                )
            required_snapshot = next(iter(pending_snapshots))
            if any(
                (
                    pair[0].policy_generation,
                    pair[0].policy_checkpoint_sha256,
                )
                != required_snapshot
                or (
                    pair[1].policy_generation,
                    pair[1].policy_checkpoint_sha256,
                )
                != required_snapshot
                for pair in evidence_pairs.values()
            ):
                raise ValueError(
                    "new release snapshot differs from pending releases"
                )
        canary_by_profile = {
            key: pair[0] for key, pair in evidence_pairs.items()
        }
        heldout_by_profile = {
            key: pair[1] for key, pair in evidence_pairs.items()
        }
        self._validate_global_batch(canary_by_profile)
        # V4 allocates all proposal/seed/birth ranges privately and proves
        # pairwise canary-heldout disjointness before minting the receipt.
        # Validate heldout global identity independently against prior state.
        self._validate_global_batch(heldout_by_profile)
        staged = {
            key: progress.clone()
            for key, progress in self._progress.items()
        }
        decisions = []
        for key in self._profile_order:
            if key not in evidence_pairs:
                continue
            canary, heldout = evidence_pairs[key]
            self._validate_common(key, canary)
            self._validate_common(key, heldout)
            canary_decision = self._apply_formal_evidence(
                key, staged[key], canary
            )
            if canary_decision.kind == "canary_blocked":
                domain = self._selected_formal_domain_from_progress(
                    staged[key]
                )
                if (
                    domain is None
                    or not self._domain_matches(heldout, domain)
                ):
                    raise ValueError(
                        "blocked canary heldout domain is inconsistent"
                    )
                staged[key].formal_receipts += (
                    _Receipt(evidence=heldout, certified=False),
                )
                decisions.append(canary_decision)
                continue
            decisions.append(
                self._apply_formal_evidence(
                    key, staged[key], heldout
                )
            )
        consumed = self._evaluator_authority.consume_releases_many(
            release_receipt_by_profile
        )
        if consumed != evidence_pairs:
            raise authority_error(
                "schema-4 evaluator changed during atomic stage"
            )
        self._progress = staged
        return tuple(decisions)

    def update_selected(
        self,
        release_receipt_by_profile: Mapping[ActionProfileKey, object],
    ) -> Tuple[BallCurriculumDecision, ...]:
        """Compatibility spelling for schema-4 staging, never publication."""

        return self.stage_selected(release_receipt_by_profile)

    def issue_global_pre_reset_barrier(
        self,
    ) -> DrainResetReceipt:
        """Acquire one fenced receipt from the bound runtime coordinator.

        This API intentionally accepts no counters, reset bitmap, environment
        count, pending mapping, checkpoint, or roots.  Every input is derived
        from current curriculum state or read by the code-pinned coordinator.
        """

        if self._issued_barriers:
            raise ValueError("an unconsumed global pre-reset barrier exists")
        if self._drain_reset_authority is None:
            raise DrainResetAuthorityError(
                "curriculum hold: drain/reset authority is not bound"
            )
        ordered = [
            self._progress[key].pending_release
            for key in self._profile_order
            if self._progress[key].pending_release is not None
        ]
        if not ordered:
            raise ValueError("no pending domain release")
        checkpoints = {
            (
                item.policy_generation,
                item.policy_checkpoint_sha256,
            )
            for item in ordered
        }
        if len(checkpoints) != 1:
            raise ValueError(
                "one global release requires one frozen policy checkpoint"
            )
        policy_generation, checkpoint_sha256 = next(iter(checkpoints))
        release_set_root = _canonical_sha256(
            {
                "schema_version": 1,
                "releases": [item.as_dict() for item in ordered],
            }
        )
        domain_heads = []
        for key in self._profile_order:
            progress = self._progress[key]
            domain = self._selected_formal_domain_from_progress(progress)
            domain_heads.append(
                {
                    "key": key.as_dict(),
                    "published_state_root_sha256": (
                        self._published_state_root(key, progress)
                    ),
                    "domain_epoch": progress.domain_release_epoch,
                    "levels_sha256": self._domain_levels_sha256(
                        domain,
                        release_epoch=progress.domain_release_epoch,
                    ),
                }
            )
        published_domain_set_root = _canonical_sha256(
            {
                "schema_version": 1,
                "ordered_domain_heads": domain_heads,
            }
        )
        evidence_set_root = _canonical_sha256(
            {
                "schema_version": 1,
                "ordered_release_evidence": [
                    {
                        "release_id_sha256": item.release_id_sha256,
                        "canary_window_sha256": (
                            item.canary_window_sha256
                        ),
                        "heldout_window_sha256": (
                            item.heldout_window_sha256
                        ),
                    }
                    for item in ordered
                ],
            }
        )
        target_progress = {
            key: progress.clone()
            for key, progress in self._progress.items()
        }
        for item in ordered:
            self._apply_target(target_progress[item.key], item.target)
            target_progress[item.key].pending_release = None
        request = {
            "schema_version": 1,
            "kind": "action_ball_domain_release_request",
            "barrier_serial": self._next_barrier_serial,
            "curriculum_contract_sha256": self._contract_sha256,
            "profile_order": [
                key.as_dict() for key in self._profile_order
            ],
            "old_global_state_root_sha256": (
                self._global_published_state_root()
            ),
            "target_global_state_root_sha256": (
                self._global_published_state_root(target_progress)
            ),
            "published_domain_set_root_sha256": (
                published_domain_set_root
            ),
            "release_set_root_sha256": release_set_root,
            "evidence_set_root_sha256": evidence_set_root,
            "release_ids": [
                item.release_id_sha256 for item in ordered
            ],
            "policy_checkpoint_sha256": checkpoint_sha256,
            "policy_generation": policy_generation,
        }
        receipt = self._drain_reset_authority.issue(request)
        token = self._drain_reset_authority.assert_receipt(receipt)
        if token.barrier_serial != self._next_barrier_serial:
            raise DrainResetAuthorityError(
                "drain/reset authority changed barrier serial"
            )
        if token.request_sha256 != _canonical_sha256(request):
            raise DrainResetAuthorityError(
                "drain/reset authority changed release request"
            )
        self._next_barrier_serial += 1
        self._issued_barriers[token.token_sha256] = receipt
        return receipt

    def issued_global_pre_reset_barriers(
        self,
    ) -> Tuple[DrainResetReceipt, ...]:
        return tuple(
            self._issued_barriers[key]
            for key in sorted(self._issued_barriers)
        )

    def commit_release(
        self,
        drain_reset_receipt: DrainResetReceipt,
    ) -> Tuple[DomainReleaseReceipt, ...]:
        """Publish once inside the coordinator's still-live drain fence."""

        if self._drain_reset_authority is None:
            raise DrainResetAuthorityError(
                "curriculum hold: drain/reset authority is not bound"
            )
        token = self._drain_reset_authority.assert_receipt(
            drain_reset_receipt
        )
        registered = self._issued_barriers.get(token.token_sha256)
        if registered is not drain_reset_receipt:
            raise DrainResetAuthorityError(
                "drain/reset receipt is foreign, stale, or consumed"
            )
        if (
            token.old_global_state_root_sha256
            != self._global_published_state_root()
        ):
            raise ValueError("published global state changed after barrier")
        ordered_pending = []
        for key in self._profile_order:
            pending = self._progress[key].pending_release
            if pending is not None:
                ordered_pending.append(pending)
        if tuple(
            item.release_id_sha256 for item in ordered_pending
        ) != token.release_ids:
            raise ValueError("pending release set changed after barrier")
        release_set_root = _canonical_sha256(
            {
                "schema_version": 1,
                "releases": [
                    item.as_dict() for item in ordered_pending
                ],
            }
        )
        if release_set_root != token.release_set_root_sha256:
            raise ValueError("pending release contents changed after barrier")
        published_domain_set_root = _canonical_sha256(
            {
                "schema_version": 1,
                "ordered_domain_heads": [
                    {
                        "key": key.as_dict(),
                        "published_state_root_sha256": (
                            self._published_state_root(
                                key, self._progress[key]
                            )
                        ),
                        "domain_epoch": self._progress[
                            key
                        ].domain_release_epoch,
                        "levels_sha256": self._domain_levels_sha256(
                            self._selected_formal_domain_from_progress(
                                self._progress[key]
                            ),
                            release_epoch=self._progress[
                                key
                            ].domain_release_epoch,
                        ),
                    }
                    for key in self._profile_order
                ],
            }
        )
        if (
            published_domain_set_root
            != token.published_domain_set_root_sha256
        ):
            raise ValueError("published domain heads changed after barrier")
        evidence_set_root = _canonical_sha256(
            {
                "schema_version": 1,
                "ordered_release_evidence": [
                    {
                        "release_id_sha256": item.release_id_sha256,
                        "canary_window_sha256": (
                            item.canary_window_sha256
                        ),
                        "heldout_window_sha256": (
                            item.heldout_window_sha256
                        ),
                    }
                    for item in ordered_pending
                ],
            }
        )
        if evidence_set_root != token.evidence_set_root_sha256:
            raise ValueError("pending release evidence set changed after barrier")
        if any(
            (
                item.policy_generation != token.policy_generation
                or item.policy_checkpoint_sha256
                != token.policy_checkpoint_sha256
            )
            for item in ordered_pending
        ):
            raise ValueError("barrier frozen-policy identity mismatch")

        staged = {
            key: progress.clone()
            for key, progress in self._progress.items()
        }
        receipts = []
        for key in self._profile_order:
            progress = staged[key]
            pending = progress.pending_release
            if pending is None:
                continue
            current_domain = self._selected_formal_domain_from_progress(
                progress
            )
            if (
                self._published_state_root(key, progress)
                != pending.from_state_root_sha256
                or progress.domain_release_epoch
                != pending.from_domain_epoch
                or self._domain_levels_sha256(
                    current_domain,
                    release_epoch=progress.domain_release_epoch,
                )
                != pending.from_levels_sha256
            ):
                raise ValueError("pending release old-state CAS failed")
            if not any(
                receipt.evidence.window_sha256
                == pending.canary_window_sha256
                for receipt in progress.formal_receipts
            ) or not any(
                receipt.evidence.window_sha256
                == pending.heldout_window_sha256
                for receipt in progress.formal_receipts
            ):
                raise ValueError("pending release evidence roots are absent")
            self._apply_target(progress, pending.target)
            if (
                self._published_state_root(key, progress)
                != pending.to_state_root_sha256
                or progress.domain_release_epoch
                != pending.to_domain_epoch
            ):
                raise ValueError("pending release target root mismatch")
            to_domain = self._selected_formal_domain_from_progress(progress)
            if (
                self._domain_levels_sha256(
                    to_domain,
                    release_epoch=progress.domain_release_epoch,
                )
                != pending.to_levels_sha256
            ):
                raise ValueError("pending release target levels mismatch")
            receipt = DomainReleaseReceipt.create(
                key=key,
                release_id_sha256=pending.release_id_sha256,
                barrier_token=token,
                commit_serial=token.barrier_serial,
                from_state_root_sha256=pending.from_state_root_sha256,
                from_domain_epoch=pending.from_domain_epoch,
                from_levels_sha256=pending.from_levels_sha256,
                to_state_root_sha256=pending.to_state_root_sha256,
                to_domain_epoch=pending.to_domain_epoch,
                to_levels_sha256=pending.to_levels_sha256,
                policy_checkpoint_sha256=(
                    pending.policy_checkpoint_sha256
                ),
                policy_generation=pending.policy_generation,
                canary_window_sha256=pending.canary_window_sha256,
                heldout_window_sha256=pending.heldout_window_sha256,
            )
            progress.pending_release = None
            progress.release_receipts += (receipt,)
            receipts.append(receipt)
        if not receipts:
            raise ValueError("barrier has no live pending release")
        if (
            self._global_published_state_root(staged)
            != token.target_global_state_root_sha256
        ):
            raise ValueError("barrier target global state root mismatch")

        def publish_noexcept() -> None:
            self._progress = staged

        committed = self._drain_reset_authority.commit_receipt(
            drain_reset_receipt,
            publish_noexcept=publish_noexcept,
        )
        if committed is not token:
            raise DrainResetAuthorityError(
                "drain/reset authority changed committed token"
            )
        del self._issued_barriers[token.token_sha256]
        return tuple(receipts)

    def abort_global_pre_reset_barrier(
        self, drain_reset_receipt: DrainResetReceipt
    ) -> None:
        if self._drain_reset_authority is None:
            raise DrainResetAuthorityError(
                "curriculum hold: drain/reset authority is not bound"
            )
        token = self._drain_reset_authority.assert_receipt(
            drain_reset_receipt
        )
        if self._issued_barriers.get(token.token_sha256) is not (
            drain_reset_receipt
        ):
            raise DrainResetAuthorityError(
                "drain/reset receipt is foreign, stale, or consumed"
            )
        self._drain_reset_authority.abort_receipt(drain_reset_receipt)
        del self._issued_barriers[token.token_sha256]

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
        if ledger.U_joint_qdes:
            blockers.append("joint_qdes_limit_zero_tolerance")
        if ledger.U_joint_actual:
            blockers.append("joint_actual_limit_zero_tolerance")
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
        if progress.pending_release is not None:
            raise ValueError(
                "domain release is pending global pre-reset commit"
            )
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

        if progress.phase == "marginal":
            # The selected formal domain already binds one action and one
            # signed axis-side arm.  Only its heldout frontier rows measure the
            # causal effect of adding that band; the center/interior rows must
            # not dilute the expand/lock/bound verdict.  Admission and every
            # safety blocker above intentionally remain whole-domain gates.
            failure = wilson_interval(
                evidence.ledger.NB_F,
                evidence.ledger.NB,
                z=self._config.confidence_z,
            )
            too_hard = failure.lower > self._config.failure_band[1]
            too_easy = failure.upper < self._config.failure_band[0]
            if evidence.ledger.NB < self._config.heldout_min_new_band:
                blockers = blockers + (
                    "new_band_safe_closed_below_gate",
                )
                quality_bad = True

        published = progress
        progress = progress.clone()
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
            # The rolling scheduler stream only chooses which candidate arm
            # to explore.  It is adaptive training data and therefore has no
            # release authority.  This frontier transition is decided solely
            # by the selected action-axis-side new-band slice of the frozen
            # heldout: NB_F/NB LCB above the band bounds, UCB below the band
            # expands, and an overlapping/in-band interval locks.
            if quality_bad or too_hard:
                statuses[index] = "decided"
                kind = "bound_marginal"
            else:
                certified = True
                frontiers[index] = candidate
                progress.last_certified = self._certificate(evidence)
                if too_easy and candidate < len(LEVELS) - 1:
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

        progress.domain_release_epoch = (
            published.domain_release_epoch + 1
        )
        progress.formal_receipts += (
            _Receipt(evidence=evidence, certified=certified),
        )
        decision = self._decision(
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
        target = self._target_from_progress(progress)
        to_domain = self._selected_formal_domain_from_progress(progress)
        pending_release = PendingDomainRelease.create(
            key=key,
            from_state_root_sha256=self._published_state_root(
                key, published
            ),
            from_domain_epoch=published.domain_release_epoch,
            from_levels_sha256=self._domain_levels_sha256(
                domain,
                release_epoch=published.domain_release_epoch,
            ),
            to_state_root_sha256=self._published_state_root(key, progress),
            to_domain_epoch=progress.domain_release_epoch,
            to_levels_sha256=self._domain_levels_sha256(
                to_domain,
                release_epoch=progress.domain_release_epoch,
            ),
            policy_checkpoint_sha256=(
                evidence.policy_checkpoint_sha256
            ),
            policy_generation=evidence.policy_generation,
            canary_window_sha256=canary.window_sha256,
            heldout_window_sha256=evidence.window_sha256,
            heldout_seq=evidence.seq,
            target=target,
        )
        published.pending_canary = None
        published.pending_release = pending_release
        published.formal_receipts = progress.formal_receipts
        return decision

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
        return BallCurriculumDecision(
            key=key,
            kind=kind,
            evidence_role=evidence.evidence_role,
            stratum=evidence.stratum,
            arm_key=arm_key,
            domain_epoch_before=evidence.domain_epoch,
            domain_epoch_after=progress.domain_release_epoch,
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
        if self._issued_barriers:
            raise DrainResetAuthorityError(
                "cannot checkpoint a live drain/reset fence; commit or abort "
                "it and re-drain after resume"
            )
        if self._evaluator_authority is not None:
            legacy_type, _, _ = self._evaluation_types()
            if type(self._evaluator_authority) is legacy_type:
                self._evaluator_authority.assert_formal_retention(
                    self._retained_canary_window_sha256(self._progress)
                )
        authority_state = (
            None
            if self._evaluator_authority is None
            else self._evaluator_authority.state_dict()
        )
        drain_reset_authority_state = (
            None
            if self._drain_reset_authority is None
            else self._drain_reset_authority.state_dict()
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
            "drain_reset_authority_contract_sha256": (
                None
                if self._drain_reset_authority is None
                else self._drain_reset_authority.authority_contract_sha256
            ),
            "drain_reset_launch_receipt_sha256": (
                None
                if self._drain_reset_authority is None
                else self._drain_reset_authority.launch_receipt_sha256
            ),
            "drain_reset_authority_state_owner_sha256": (
                None
                if self._drain_reset_authority is None
                else self._drain_reset_authority.state_owner_sha256
            ),
            "drain_reset_authority_state": (
                drain_reset_authority_state
            ),
            "drain_reset_authority_state_sha256": (
                None
                if drain_reset_authority_state is None
                else _canonical_sha256(drain_reset_authority_state)
            ),
            "next_barrier_serial": self._next_barrier_serial,
            "issued_global_pre_reset_barriers": [],
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
        for receipt in progress.release_receipts:
            chain = hashlib.sha256(
                (chain + receipt.receipt_sha256).encode("ascii")
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
            "domain_release_epoch": progress.domain_release_epoch,
            "pending_canary_window_sha256": (
                None
                if progress.pending_canary is None
                else progress.pending_canary.evidence.window_sha256
            ),
            "pending_release": (
                None
                if progress.pending_release is None
                else progress.pending_release.as_dict()
            ),
            "release_receipts": [
                receipt.as_dict()
                for receipt in progress.release_receipts
            ],
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
            "drain_reset_authority_contract_sha256",
            "drain_reset_launch_receipt_sha256",
            "drain_reset_authority_state_owner_sha256",
            "drain_reset_authority_state",
            "drain_reset_authority_state_sha256",
            "next_barrier_serial",
            "issued_global_pre_reset_barriers",
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
        release_receipts_by_id: Dict[str, DomainReleaseReceipt] = {}
        committed_barriers: Dict[
            str, Tuple[GlobalPreResetBarrierToken, list]
        ] = {}
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
                    "domain_release_epoch",
                    "pending_canary_window_sha256",
                    "pending_release",
                    "release_receipts",
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
            release_raw = row["release_receipts"]
            if not isinstance(formal_raw, list) or not isinstance(
                scheduler_raw, list
            ) or not isinstance(release_raw, list):
                raise ValueError("receipt collections must be lists")
            parsed_pending = (
                None
                if row["pending_release"] is None
                else self._parse_pending_release(row["pending_release"])
            )
            if parsed_pending is not None and parsed_pending.key != key:
                raise ValueError("pending release key/order mismatch")
            for item in release_raw:
                release_receipt = self._parse_release_receipt(item)
                if release_receipt.key != key:
                    raise ValueError("release receipt key/order mismatch")
                if (
                    release_receipt.release_id_sha256
                    in release_receipts_by_id
                ):
                    raise ValueError("duplicate domain release id")
                release_receipts_by_id[
                    release_receipt.release_id_sha256
                ] = release_receipt
                barrier = release_receipt.barrier_token
                existing = committed_barriers.get(barrier.token_sha256)
                if existing is None:
                    committed_barriers[barrier.token_sha256] = (
                        barrier,
                        [release_receipt],
                    )
                else:
                    if existing[0] != barrier:
                        raise ValueError(
                            "one barrier digest names different contents"
                        )
                    existing[1].append(release_receipt)
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
        formal_seq_by_window = {
            receipt.evidence.window_sha256: receipt.evidence.seq
            for _, kind, _, receipt in events
            if kind == "formal"
        }
        barriers_by_trigger_seq: Dict[
            int,
            list,
        ] = {}
        for group in committed_barriers.values():
            try:
                trigger_seq = max(
                    formal_seq_by_window[
                        receipt.heldout_window_sha256
                    ]
                    for receipt in group[1]
                )
            except KeyError as exc:
                raise ValueError(
                    "release receipt heldout window is absent"
                ) from exc
            barriers_by_trigger_seq.setdefault(
                trigger_seq, []
            ).append(group)

        # A full N=93 replay contains thousands of barrier checks.  The
        # published root of one profile changes only when scheduler selection
        # changes or a staged target commits; recomputing all N per-profile
        # canonical hashes for every barrier made restore quadratic in N.
        # Keep the exact canonical aggregate documents live and refresh only
        # the row whose deterministic replay state changed.  The aggregate
        # SHA itself is still recomputed and checked at every historical
        # barrier, so no integrity gate is skipped or weakened.
        profile_key_documents = tuple(
            key.as_dict() for key in self._profile_order
        )
        profile_index_by_key = {
            key: index
            for index, key in enumerate(self._profile_order)
        }
        published_state_rows = []
        published_domain_head_rows = []
        for key, key_document in zip(
            self._profile_order, profile_key_documents
        ):
            progress = replay._progress[key]
            published_root = replay._published_state_root(key, progress)
            release_epoch = progress.domain_release_epoch
            domain = replay._selected_formal_domain_from_progress(
                progress
            )
            published_state_rows.append(
                {
                    "key": key_document,
                    "published_state_root_sha256": published_root,
                }
            )
            published_domain_head_rows.append(
                {
                    "key": key_document,
                    "published_state_root_sha256": published_root,
                    "domain_epoch": release_epoch,
                    "levels_sha256": replay._domain_levels_sha256(
                        domain, release_epoch=release_epoch
                    ),
                }
            )
        published_state_document = {
            "schema_version": 1,
            "profile_order": published_state_rows,
        }
        published_domain_head_document = {
            "schema_version": 1,
            "ordered_domain_heads": published_domain_head_rows,
        }

        def refresh_published_row(key: ActionProfileKey) -> str:
            index = profile_index_by_key[key]
            progress = replay._progress[key]
            published_root = replay._published_state_root(key, progress)
            release_epoch = progress.domain_release_epoch
            published_state_rows[index][
                "published_state_root_sha256"
            ] = published_root
            domain_row = published_domain_head_rows[index]
            domain_row["published_state_root_sha256"] = published_root
            domain_row["domain_epoch"] = release_epoch
            domain_row["levels_sha256"] = (
                replay._domain_levels_sha256(
                    replay._selected_formal_domain_from_progress(
                        progress
                    ),
                    release_epoch=release_epoch,
                )
            )
            return published_root

        checkpoint_by_generation: Dict[int, str] = {}
        previous_generation = 0
        replayed_release_ids = set()
        pending_release_by_id: Dict[
            str, Tuple[ActionProfileKey, PendingDomainRelease]
        ] = {}
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
                    refresh_published_row(key)
            else:
                blocked_pair = bool(
                    evidence.evidence_role == "frozen_heldout"
                    and progress.pending_canary is None
                    and progress.formal_receipts
                    and progress.formal_receipts[-1]
                    .evidence.evidence_role
                    == "frozen_canary"
                    and not progress.formal_receipts[-1].certified
                )
                if blocked_pair:
                    canary = progress.formal_receipts[-1].evidence
                    replay._validate_common(key, evidence)
                    domain = (
                        replay._selected_formal_domain_from_progress(
                            progress
                        )
                    )
                    if (
                        domain is None
                        or not replay._domain_matches(evidence, domain)
                        or evidence.unique_birth_count
                        != evidence.ledger.P
                        or evidence.ledger.P
                        < replay._config.heldout_min_proposals
                        or evidence.ledger.safe_closed
                        < replay._config.heldout_min_safe_closed
                        or receipt.certified
                        or not replay._metrics(canary.ledger)[7]
                    ):
                        raise ValueError(
                            "blocked V4 formal pair is invalid"
                        )
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
                        if getattr(canary, field) != getattr(
                            evidence, field
                        ):
                            raise ValueError(
                                "blocked V4 pair identity mismatch"
                            )
                    progress.formal_receipts += (receipt,)
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
            pending = progress.pending_release
            if pending is not None:
                existing_pending = pending_release_by_id.setdefault(
                    pending.release_id_sha256, (key, pending)
                )
                if existing_pending != (key, pending):
                    raise ValueError("duplicate pending domain release id")
            for group in barriers_by_trigger_seq.get(evidence.seq, ()):
                barrier, group_receipts = group
                if not all(
                    release_id in pending_release_by_id
                    for release_id in barrier.release_ids
                ):
                    continue
                ordered_pending_ids = tuple(
                    sorted(
                        pending_release_by_id,
                        key=lambda release_id: profile_index_by_key[
                            pending_release_by_id[release_id][0]
                        ],
                    )
                )
                if ordered_pending_ids != barrier.release_ids:
                    raise ValueError(
                        "committed barrier omitted a pending release"
                    )
                if (
                    _canonical_sha256(published_state_document)
                    != barrier.old_global_state_root_sha256
                ):
                    raise ValueError(
                        "committed barrier old global root mismatch"
                    )
                ordered_pending = [
                    pending_release_by_id[release_id][1]
                    for release_id in barrier.release_ids
                ]
                release_set_root = _canonical_sha256(
                    {
                        "schema_version": 1,
                        "releases": [
                            pending.as_dict()
                            for pending in ordered_pending
                        ],
                    }
                )
                if release_set_root != barrier.release_set_root_sha256:
                    raise ValueError(
                        "committed barrier release root mismatch"
                    )
                published_domain_set_root = _canonical_sha256(
                    published_domain_head_document
                )
                if (
                    published_domain_set_root
                    != barrier.published_domain_set_root_sha256
                ):
                    raise ValueError(
                        "committed barrier domain-head root mismatch"
                    )
                evidence_set_root = _canonical_sha256(
                    {
                        "schema_version": 1,
                        "ordered_release_evidence": [
                            {
                                "release_id_sha256": (
                                    pending.release_id_sha256
                                ),
                                "canary_window_sha256": (
                                    pending.canary_window_sha256
                                ),
                                "heldout_window_sha256": (
                                    pending.heldout_window_sha256
                                ),
                            }
                            for pending in ordered_pending
                        ],
                    }
                )
                if evidence_set_root != barrier.evidence_set_root_sha256:
                    raise ValueError(
                        "committed barrier evidence root mismatch"
                    )
                checkpoints = {
                    (
                        pending.policy_generation,
                        pending.policy_checkpoint_sha256,
                    )
                    for pending in ordered_pending
                }
                if checkpoints != {
                    (
                        barrier.policy_generation,
                        barrier.policy_checkpoint_sha256,
                    )
                }:
                    raise ValueError(
                        "committed barrier policy identity mismatch"
                    )
                request_document = {
                    "schema_version": 1,
                    "kind": "action_ball_domain_release_request",
                    "barrier_serial": barrier.barrier_serial,
                    "curriculum_contract_sha256": self._contract_sha256,
                    "profile_order": list(profile_key_documents),
                    "old_global_state_root_sha256": (
                        barrier.old_global_state_root_sha256
                    ),
                    "target_global_state_root_sha256": (
                        barrier.target_global_state_root_sha256
                    ),
                    "published_domain_set_root_sha256": (
                        published_domain_set_root
                    ),
                    "release_set_root_sha256": release_set_root,
                    "evidence_set_root_sha256": evidence_set_root,
                    "release_ids": list(barrier.release_ids),
                    "policy_checkpoint_sha256": (
                        barrier.policy_checkpoint_sha256
                    ),
                    "policy_generation": barrier.policy_generation,
                }
                if (
                    _canonical_sha256(request_document)
                    != barrier.request_sha256
                ):
                    raise ValueError(
                        "committed barrier request digest mismatch"
                    )
                receipt_by_id = {
                    item.release_id_sha256: item
                    for item in group_receipts
                }
                if tuple(receipt_by_id) != barrier.release_ids:
                    raise ValueError(
                        "committed barrier receipt set mismatch"
                    )
                for release_id in barrier.release_ids:
                    pending_key, pending = pending_release_by_id[
                        release_id
                    ]
                    release_receipt = receipt_by_id[release_id]
                    fields = (
                        "release_id_sha256",
                        "from_state_root_sha256",
                        "from_domain_epoch",
                        "from_levels_sha256",
                        "to_state_root_sha256",
                        "to_domain_epoch",
                        "to_levels_sha256",
                        "policy_checkpoint_sha256",
                        "policy_generation",
                        "canary_window_sha256",
                        "heldout_window_sha256",
                    )
                    if any(
                        getattr(release_receipt, field)
                        != getattr(pending, field)
                        for field in fields
                    ):
                        raise ValueError(
                            "release receipt does not match staged release"
                        )
                    target_progress = replay._progress[pending_key]
                    self._apply_target(
                        target_progress, pending.target
                    )
                    if (
                        refresh_published_row(pending_key)
                        != pending.to_state_root_sha256
                    ):
                        raise ValueError(
                            "replayed release target root mismatch"
                        )
                    target_progress.pending_release = None
                    target_progress.release_receipts += (
                        release_receipt,
                    )
                    replayed_release_ids.add(release_id)
                    del pending_release_by_id[release_id]
                if (
                    _canonical_sha256(published_state_document)
                    != barrier.target_global_state_root_sha256
                ):
                    raise ValueError(
                        "committed barrier target global root mismatch"
                    )
        if replayed_release_ids != set(release_receipts_by_id):
            raise ValueError("release receipt is not replay-reachable")
        if (
            _canonical_sha256(published_state_document)
            != replay._global_published_state_root()
        ):
            raise ValueError(
                "cached published-state replay root is inconsistent"
            )
        replay_rows = [
            replay._progress_row(key, replay._progress[key])
            for key in self._profile_order
        ]
        if replay_rows != persisted_rows:
            raise ValueError(
                "curriculum state is not reachable by deterministic replay"
            )
        next_barrier_serial = _plain_int(
            document["next_barrier_serial"],
            name="next_barrier_serial",
            minimum=1,
        )
        raw_barriers = document["issued_global_pre_reset_barriers"]
        if raw_barriers != []:
            raise ValueError(
                "live drain/reset receipts cannot survive checkpoint resume"
            )
        serials = [
            receipt.commit_serial
            for progress in replay._progress.values()
            for receipt in progress.release_receipts
        ]
        if next_barrier_serial != max(serials, default=0) + 1:
            raise ValueError("next barrier serial is not exact")
        replay._next_barrier_serial = next_barrier_serial
        replay._issued_barriers = {}

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
            if not isinstance(authority_state, Mapping):
                raise ValueError("authority state must be a mapping")
            legacy_type, _, _ = self._evaluation_types()
            v4_type, _, _, _ = self._evaluation_v4_types()
            if type(self._evaluator_authority) is v4_type:
                self._assert_v4_authority_alignment(
                    replay._progress, authority_state
                )
                self._evaluator_authority.load_state_dict(authority_state)
                if self._evaluator_authority.state_dict() != authority_state:
                    raise ValueError(
                        "V4 evaluator authority resume is not exact"
                    )
            elif type(self._evaluator_authority) is legacy_type:
                expected_consumed = tuple(
                    receipt.evidence.window_sha256
                    for _, _, _, receipt in sorted(events)
                )
                raw_consumed = authority_state.get("consumed")
                if not isinstance(raw_consumed, list):
                    raise ValueError(
                        "authority consumed state must be a list"
                    )
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
            else:
                raise ValueError("unsupported evaluator authority type")
        if self._drain_reset_authority is None:
            drain_fields = (
                document["drain_reset_authority_contract_sha256"],
                document["drain_reset_launch_receipt_sha256"],
                document["drain_reset_authority_state_owner_sha256"],
                document["drain_reset_authority_state"],
                document["drain_reset_authority_state_sha256"],
            )
            if drain_fields != (None, None, None, None, None):
                raise ValueError(
                    "checkpoint requires bound drain/reset authority"
                )
            if committed_barriers:
                raise ValueError(
                    "committed releases require drain/reset authority history"
                )
        else:
            if (
                document["drain_reset_authority_contract_sha256"]
                != self._drain_reset_authority.authority_contract_sha256
                or document["drain_reset_launch_receipt_sha256"]
                != self._drain_reset_authority.launch_receipt_sha256
                or document[
                    "drain_reset_authority_state_owner_sha256"
                ]
                != self._drain_reset_authority.state_owner_sha256
            ):
                raise ValueError("drain/reset authority binding mismatch")
            drain_state = document["drain_reset_authority_state"]
            if not isinstance(drain_state, Mapping) or (
                _canonical_sha256(drain_state)
                != document["drain_reset_authority_state_sha256"]
            ):
                raise ValueError("drain/reset authority state digest mismatch")
            raw_consumed = drain_state.get("consumed")
            if not isinstance(raw_consumed, list):
                raise ValueError(
                    "drain/reset authority consumed state must be a list"
                )
            expected_barriers = tuple(
                group[0]
                for group in sorted(
                    committed_barriers.values(),
                    key=lambda item: item[0].barrier_serial,
                )
            )
            expected_tokens = tuple(
                item.token_sha256 for item in expected_barriers
            )
            actual_tokens = []
            for index, item in enumerate(raw_consumed):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        "drain/reset consumed token must be a mapping"
                    )
                actual_tokens.append(
                    _sha256(
                        item.get("token_sha256"),
                        name=f"drain consumed[{index}].token_sha256",
                    )
                )
            actual_tokens = tuple(actual_tokens)
            if actual_tokens != expected_tokens:
                raise ValueError(
                    "drain/reset authority and release history differ"
                )
            self._drain_reset_authority.load_state_dict(
                drain_state,
                _prevalidated_tokens=expected_barriers,
            )
            if (
                self._drain_reset_authority.state_dict()
                != drain_state
            ):
                raise ValueError(
                    "drain/reset authority resume is not exact"
                )
        self._progress = replay._progress
        self._next_barrier_serial = replay._next_barrier_serial
        self._issued_barriers = replay._issued_barriers

    @staticmethod
    def _parse_progress_target(value: object) -> _ProgressTarget:
        fields = tuple(_ProgressTarget.__dataclass_fields__)
        row = _exact_keys(value, fields, name="release target")
        for field in (
            "arm_frontier_indices",
            "arm_status",
            "arm_probe_indices",
            "arm_epochs",
            "last_selected_round",
        ):
            if not isinstance(row[field], list):
                raise ValueError(f"release target {field} must be a list")
        last_json = row["last_certified_json"]
        if last_json is not None:
            if type(last_json) is not str:
                raise ValueError("last_certified_json must be string or null")
            decoded = json.loads(last_json)
            if not isinstance(decoded, Mapping):
                raise ValueError("last_certified_json must encode a mapping")
            canonical = json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            if canonical != last_json:
                raise ValueError("last_certified_json is not canonical")
        target = _ProgressTarget(
            phase=row["phase"],
            arm_frontier_indices=tuple(row["arm_frontier_indices"]),
            arm_status=tuple(row["arm_status"]),
            arm_probe_indices=tuple(row["arm_probe_indices"]),
            arm_epochs=tuple(row["arm_epochs"]),
            selected_arm_key=row["selected_arm_key"],
            selection_round=row["selection_round"],
            last_selected_round=tuple(row["last_selected_round"]),
            center_epoch=row["center_epoch"],
            joint_epoch=row["joint_epoch"],
            joint_probe_index=row["joint_probe_index"],
            joint_rho_index=row["joint_rho_index"],
            center_failures=row["center_failures"],
            domain_release_epoch=row["domain_release_epoch"],
            last_certified_json=last_json,
        )
        if any(
            len(getattr(target, field)) != len(ARM_KEYS)
            for field in (
                "arm_frontier_indices",
                "arm_status",
                "arm_probe_indices",
                "arm_epochs",
                "last_selected_round",
            )
        ):
            raise ValueError("release target arm vectors have wrong length")
        return target

    @classmethod
    def _parse_pending_release(
        cls, value: object
    ) -> PendingDomainRelease:
        fields = (
            "schema_version",
            "key",
            "from_state_root_sha256",
            "from_domain_epoch",
            "from_levels_sha256",
            "to_state_root_sha256",
            "to_domain_epoch",
            "to_levels_sha256",
            "policy_checkpoint_sha256",
            "policy_generation",
            "canary_window_sha256",
            "heldout_window_sha256",
            "heldout_seq",
            "target",
            "release_id_sha256",
        )
        row = _exact_keys(value, fields, name="pending domain release")
        if row["schema_version"] != 1:
            raise ValueError("unsupported pending release schema")
        key_row = _exact_keys(
            row["key"],
            ("action_uid", "profile_sha256", "mobility"),
            name="pending release key",
        )
        return PendingDomainRelease(
            key=ActionProfileKey(**key_row),
            release_id_sha256=row["release_id_sha256"],
            from_state_root_sha256=row["from_state_root_sha256"],
            from_domain_epoch=row["from_domain_epoch"],
            from_levels_sha256=row["from_levels_sha256"],
            to_state_root_sha256=row["to_state_root_sha256"],
            to_domain_epoch=row["to_domain_epoch"],
            to_levels_sha256=row["to_levels_sha256"],
            policy_checkpoint_sha256=row[
                "policy_checkpoint_sha256"
            ],
            policy_generation=row["policy_generation"],
            canary_window_sha256=row["canary_window_sha256"],
            heldout_window_sha256=row["heldout_window_sha256"],
            heldout_seq=row["heldout_seq"],
            target=cls._parse_progress_target(row["target"]),
        )

    @staticmethod
    def _parse_release_receipt(value: object) -> DomainReleaseReceipt:
        fields = (
            "schema_version",
            "key",
            "release_id_sha256",
            "barrier_token",
            "commit_serial",
            "from_state_root_sha256",
            "from_domain_epoch",
            "from_levels_sha256",
            "to_state_root_sha256",
            "to_domain_epoch",
            "to_levels_sha256",
            "policy_checkpoint_sha256",
            "policy_generation",
            "canary_window_sha256",
            "heldout_window_sha256",
            "receipt_sha256",
        )
        row = _exact_keys(value, fields, name="domain release receipt")
        if row["schema_version"] != 1:
            raise ValueError("unsupported release receipt schema")
        key_row = _exact_keys(
            row["key"],
            ("action_uid", "profile_sha256", "mobility"),
            name="release receipt key",
        )
        return DomainReleaseReceipt(
            key=ActionProfileKey(**key_row),
            release_id_sha256=row["release_id_sha256"],
            barrier_token=ActionBallCurriculum._parse_barrier_token(
                row["barrier_token"]
            ),
            commit_serial=row["commit_serial"],
            from_state_root_sha256=row["from_state_root_sha256"],
            from_domain_epoch=row["from_domain_epoch"],
            from_levels_sha256=row["from_levels_sha256"],
            to_state_root_sha256=row["to_state_root_sha256"],
            to_domain_epoch=row["to_domain_epoch"],
            to_levels_sha256=row["to_levels_sha256"],
            policy_checkpoint_sha256=row[
                "policy_checkpoint_sha256"
            ],
            policy_generation=row["policy_generation"],
            canary_window_sha256=row["canary_window_sha256"],
            heldout_window_sha256=row["heldout_window_sha256"],
            receipt_sha256=row["receipt_sha256"],
        )

    @staticmethod
    def _parse_barrier_token(
        value: object,
    ) -> GlobalPreResetBarrierToken:
        fields = (
            "schema_version",
            "barrier_serial",
            "authority_contract_sha256",
            "launch_receipt_sha256",
            "runtime_source_contract_sha256",
            "runtime_source_sha256",
            "request_sha256",
            "old_global_state_root_sha256",
            "target_global_state_root_sha256",
            "published_domain_set_root_sha256",
            "release_set_root_sha256",
            "evidence_set_root_sha256",
            "release_ids",
            "policy_checkpoint_sha256",
            "policy_generation",
            "broker_reset_generation",
            "attempt_pool_reset_generation",
            "task_receipt_pool_reset_generation",
            "env_reset_generation",
            "active_attempts",
            "reserved_attempts",
            "active_births",
            "pending_task_receipts",
            "reset_count",
            "env_count",
            "reset_participant_ids",
            "reset_bitmap_sha256",
            "fence_id_sha256",
            "broker_state_root_sha256",
            "attempt_pool_state_root_sha256",
            "task_receipt_pool_state_root_sha256",
            "env_reset_state_root_sha256",
            "snapshot_sha256",
            "token_sha256",
        )
        row = _exact_keys(value, fields, name="global pre-reset barrier")
        if row["schema_version"] != 2:
            raise ValueError("unsupported barrier token schema")
        if not isinstance(row["release_ids"], list) or not isinstance(
            row["reset_participant_ids"], list
        ):
            raise ValueError(
                "barrier release_ids and reset participants must be lists"
            )
        return GlobalPreResetBarrierToken(
            barrier_serial=row["barrier_serial"],
            authority_contract_sha256=row[
                "authority_contract_sha256"
            ],
            launch_receipt_sha256=row["launch_receipt_sha256"],
            runtime_source_contract_sha256=row[
                "runtime_source_contract_sha256"
            ],
            runtime_source_sha256=row["runtime_source_sha256"],
            request_sha256=row["request_sha256"],
            old_global_state_root_sha256=row[
                "old_global_state_root_sha256"
            ],
            target_global_state_root_sha256=row[
                "target_global_state_root_sha256"
            ],
            published_domain_set_root_sha256=row[
                "published_domain_set_root_sha256"
            ],
            release_set_root_sha256=row["release_set_root_sha256"],
            evidence_set_root_sha256=row["evidence_set_root_sha256"],
            release_ids=tuple(row["release_ids"]),
            policy_checkpoint_sha256=row[
                "policy_checkpoint_sha256"
            ],
            policy_generation=row["policy_generation"],
            broker_reset_generation=row["broker_reset_generation"],
            attempt_pool_reset_generation=row[
                "attempt_pool_reset_generation"
            ],
            task_receipt_pool_reset_generation=row[
                "task_receipt_pool_reset_generation"
            ],
            env_reset_generation=row["env_reset_generation"],
            active_attempts=row["active_attempts"],
            reserved_attempts=row["reserved_attempts"],
            active_births=row["active_births"],
            pending_task_receipts=row["pending_task_receipts"],
            reset_count=row["reset_count"],
            env_count=row["env_count"],
            reset_participant_ids=tuple(row["reset_participant_ids"]),
            reset_bitmap_sha256=row["reset_bitmap_sha256"],
            fence_id_sha256=row["fence_id_sha256"],
            broker_state_root_sha256=row[
                "broker_state_root_sha256"
            ],
            attempt_pool_state_root_sha256=row[
                "attempt_pool_state_root_sha256"
            ],
            task_receipt_pool_state_root_sha256=row[
                "task_receipt_pool_state_root_sha256"
            ],
            env_reset_state_root_sha256=row[
                "env_reset_state_root_sha256"
            ],
            snapshot_sha256=row["snapshot_sha256"],
            token_sha256=row["token_sha256"],
        )

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
        v4_authority_type, _, _, _ = self._evaluation_v4_types()
        validator = (
            _validated_attempt_row_v4
            if type(self._evaluator_authority) is v4_authority_type
            else _validated_attempt_row
        )
        attempts = tuple(
            validator(
                item, name=f"scheduler receipt attempt[{index}]"
            )
            for index, item in enumerate(raw_attempts)
        )
        if _ledger_from_attempt_rows(attempts) != evidence.ledger:
            raise ValueError("scheduler receipt ledger mismatch")
        return _SchedulerReceipt(evidence=evidence, attempts=attempts)
