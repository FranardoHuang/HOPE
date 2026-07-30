"""Code-rooted evaluation authority for action-conditioned ball-first training.

The authority turns exact ordered attempt transcripts into opaque, single-use
capabilities.  Online scheduler windows and frozen certification windows use
the same transport, but only frozen canary/heldout evidence may authorize a
curriculum frontier change.  Every window binds both sample and birth
receipts, the signed-arm catalog, and the deterministic scheduler contract.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

try:  # Package import in production.
    from .action_ball_curriculum import (
        ARM_CATALOG_SHA256,
        EVIDENCE_SCHEMA_VERSION,
        ActionProfileKey,
        BallDomainEvidence,
        BallOutcomeLedger,
        canonical_action_profile_key,
    )
except ImportError:  # Dependency-light direct-file tests.
    from action_ball_curriculum import (  # type: ignore
        ARM_CATALOG_SHA256,
        EVIDENCE_SCHEMA_VERSION,
        ActionProfileKey,
        BallDomainEvidence,
        BallOutcomeLedger,
        canonical_action_profile_key,
    )

try:  # Package import in production.
    from . import action_ball_runtime as _attempt_runtime
except ImportError:  # Direct-file tests may load it into sys.modules first.
    _attempt_runtime = sys.modules.get("action_ball_runtime")


SCHEMA_VERSION = 3
STATE_SCHEMA_VERSION = 4
INT64_MAX = (1 << 63) - 1
_MINT_SENTINEL = object()
_FORMAL_SENTINEL = object()
_ZERO_SHA = "0" * 64
_FORMAL_ROLES = frozenset(("frozen_canary", "frozen_heldout"))
TERMINAL_OUTCOMES = (
    "legal_return",
    "safe_nonreturn",
    "table_hit",
    "fall",
    "collision",
    "joint_qdes_limit",
    "joint_actual_limit",
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _plain_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a plain integer")
    if value < minimum or value > INT64_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {INT64_MAX}]")
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


def ordered_sample_receipt_root(
    ordered_sample_receipt_sha256: Sequence[str],
) -> str:
    """Hash an ordered, within-window unique sample receipt list."""

    if isinstance(ordered_sample_receipt_sha256, (str, bytes)):
        raise ValueError("ordered sample receipts must be a sequence")
    receipts = tuple(
        _sha256(value, name=f"sample_receipt_sha256[{index}]")
        for index, value in enumerate(ordered_sample_receipt_sha256)
    )
    if not receipts:
        raise ValueError("ordered sample receipts must not be empty")
    if len(receipts) != len(set(receipts)):
        raise ValueError("one evaluation window cannot reuse a sample receipt")
    return _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "action_ball_ordered_sample_receipts",
            "count": len(receipts),
            "ordered_sample_receipt_sha256": list(receipts),
        }
    )


def ordered_birth_receipt_root(
    ordered_birth_receipt_sha256: Sequence[str],
) -> str:
    """Hash the exact ordered birth list; scheduler multiplicity is preserved."""

    if isinstance(ordered_birth_receipt_sha256, (str, bytes)):
        raise ValueError("ordered birth receipts must be a sequence")
    receipts = tuple(
        _sha256(value, name=f"birth_receipt_sha256[{index}]")
        for index, value in enumerate(ordered_birth_receipt_sha256)
    )
    if not receipts:
        raise ValueError("ordered birth receipts must not be empty")
    return _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "action_ball_ordered_birth_receipts",
            "count": len(receipts),
            "ordered_birth_receipt_sha256": list(receipts),
        }
    )


_AUTHORITY_CONTRACT_DOCUMENT = {
    "schema_version": SCHEMA_VERSION,
    "kind": "action_ball_frozen_evaluator_authority",
    "identity": (
        "curriculum/arm-catalog/scheduler/sampler/solver/policy/"
        "profile-order/attempt-source"
    ),
    "window": (
        "role/profile/signed-arm-domain/policy snapshot/seed and sample "
        "ranges/ordered sample and birth roots/unique births/full ledger"
    ),
    "formal": (
        "canary and heldout require one unique birth per proposal and "
        "same-action formal windows never reuse births"
    ),
    "capability": "exact-authority opaque single-use object",
    "launch": "code-pinned exact canonical receipt",
    "resume": (
        "pending and retained in-flight canaries keep exact attempts; "
        "completed formal windows retain canonical aggregate evidence and "
        "capability hash-chain events without duplicating attempt rows"
    ),
    "new_band": (
        "every attempt declares whether its drawn value fell inside the "
        "candidate-minus-frontier new band; ledgers conserve the safe-closed "
        "new-band count NB and its failures NB_F (Franco 2026-07-28 ring)"
    ),
}
FROZEN_EVALUATOR_AUTHORITY_CONTRACT_SHA256 = _canonical_sha256(
    _AUTHORITY_CONTRACT_DOCUMENT
)

# Production remains fail-closed until a reviewed finalizer digest is pinned.
TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256 = frozenset()


def launch_receipt_document(
    *,
    curriculum_contract_sha256: str,
    profile_order: Sequence[ActionProfileKey],
    arm_catalog_sha256: str,
    scheduler_contract_sha256: str,
    sampler_sha256: str,
    solver_sha256: str,
    policy_contract_sha256: str,
    attempt_source_contract_sha256: str,
    attempt_source_path: str,
    attempt_source_sha256: str,
) -> Dict[str, object]:
    """Return exact launch-finalizer input; constructing it does not authorize."""

    if isinstance(profile_order, (str, bytes)):
        raise ValueError("profile_order must be a sequence")
    order = tuple(profile_order)
    if (
        not order
        or any(not isinstance(key, ActionProfileKey) for key in order)
        or len(order) != len(set(order))
    ):
        raise ValueError("profile_order must contain unique ActionProfileKey values")
    catalog = _sha256(arm_catalog_sha256, name="arm_catalog_sha256")
    if catalog != ARM_CATALOG_SHA256:
        raise ValueError("launch arm catalog does not match code")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_ball_frozen_evaluator_launch",
        "authority_contract_sha256": (
            FROZEN_EVALUATOR_AUTHORITY_CONTRACT_SHA256
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
        "sampler_sha256": _sha256(sampler_sha256, name="sampler_sha256"),
        "solver_sha256": _sha256(solver_sha256, name="solver_sha256"),
        "policy_contract_sha256": _sha256(
            policy_contract_sha256,
            name="policy_contract_sha256",
        ),
        "attempt_source_contract_sha256": _sha256(
            attempt_source_contract_sha256,
            name="attempt_source_contract_sha256",
        ),
        "attempt_source_path": _relative_path(
            attempt_source_path,
            name="attempt_source_path",
        ),
        "attempt_source_sha256": _sha256(
            attempt_source_sha256,
            name="attempt_source_sha256",
        ),
    }


class FrozenEvaluationAuthorityError(RuntimeError):
    """A capability is absent, foreign, stale, replayed, or inconsistent."""


@dataclass(frozen=True)
class _AttemptData:
    sample_receipt_sha256: str
    birth_receipt_sha256: str
    solver_admitted: bool
    installed: bool
    started: bool
    closed: bool
    terminal_outcome: str | None
    infrastructure_invalid: bool
    in_new_band: bool

    def __post_init__(self) -> None:
        _sha256(self.sample_receipt_sha256, name="sample_receipt_sha256")
        _sha256(self.birth_receipt_sha256, name="birth_receipt_sha256")
        for field in (
            "solver_admitted",
            "installed",
            "started",
            "closed",
            "infrastructure_invalid",
            "in_new_band",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be bool")
        if not (
            self.solver_admitted
            >= self.installed
            >= self.started
            >= self.closed
        ):
            raise ValueError(
                "attempt must satisfy admitted >= installed >= started >= closed"
            )
        if self.closed:
            if self.terminal_outcome not in TERMINAL_OUTCOMES:
                raise ValueError("closed attempt needs exactly one terminal outcome")
        elif self.terminal_outcome is not None:
            raise ValueError("non-closed attempt cannot have a terminal outcome")

    def as_dict(self) -> Dict[str, object]:
        return {
            "sample_receipt_sha256": self.sample_receipt_sha256,
            "birth_receipt_sha256": self.birth_receipt_sha256,
            "solver_admitted": self.solver_admitted,
            "installed": self.installed,
            "started": self.started,
            "closed": self.closed,
            "terminal_outcome": self.terminal_outcome,
            "infrastructure_invalid": self.infrastructure_invalid,
            "in_new_band": self.in_new_band,
        }


class FrozenAttemptReceipt:
    """Opaque attempt record emitted by the admitted runtime source."""

    __slots__ = ("_authority", "_lifetime", "_data", "_receipt_sha256")

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorAuthority",
        lifetime: object,
        data: _AttemptData,
    ) -> None:
        if sentinel is not _MINT_SENTINEL:
            raise TypeError("FrozenAttemptReceipt is minted only by its authority")
        self._authority = authority
        self._lifetime = lifetime
        self._data = data
        self._receipt_sha256 = _canonical_sha256(data.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return self._receipt_sha256

    @property
    def sample_receipt_sha256(self) -> str:
        return self._data.sample_receipt_sha256

    @property
    def birth_receipt_sha256(self) -> str:
        return self._data.birth_receipt_sha256


def _ledger_from_attempts(
    attempts: Sequence[_AttemptData],
) -> BallOutcomeLedger:
    terminal = {name: 0 for name in TERMINAL_OUTCOMES}
    new_band = 0
    new_band_failures = 0
    for attempt in attempts:
        if attempt.terminal_outcome is not None:
            terminal[attempt.terminal_outcome] += 1
        if (
            attempt.in_new_band
            and attempt.closed
            and not attempt.infrastructure_invalid
            and attempt.terminal_outcome
            in ("legal_return", "safe_nonreturn")
        ):
            new_band += 1
            if attempt.terminal_outcome == "safe_nonreturn":
                new_band_failures += 1
    return BallOutcomeLedger(
        P=len(attempts),
        A=sum(item.solver_admitted for item in attempts),
        I=sum(item.installed for item in attempts),
        S=sum(item.started for item in attempts),
        C=sum(item.closed for item in attempts),
        L=terminal["legal_return"],
        F=terminal["safe_nonreturn"],
        U_table=terminal["table_hit"],
        U_fall=terminal["fall"],
        U_collision=terminal["collision"],
        U_joint_qdes=terminal["joint_qdes_limit"],
        U_joint_actual=terminal["joint_actual_limit"],
        X=sum(item.infrastructure_invalid for item in attempts),
        NB=new_band,
        NB_F=new_band_failures,
    )


class FrozenEvaluationCapability:
    """Opaque, single-use authority handle."""

    __slots__ = ("_authority", "_lifetime", "_capability_id", "_evidence")

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorAuthority",
        lifetime: object,
        capability_id: str,
        evidence: BallDomainEvidence,
    ) -> None:
        if sentinel is not _MINT_SENTINEL:
            raise TypeError(
                "FrozenEvaluationCapability is minted only by its authority"
            )
        self._authority = authority
        self._lifetime = lifetime
        self._capability_id = capability_id
        self._evidence = evidence

    @property
    def capability_id(self) -> str:
        return self._capability_id

    @property
    def evidence(self) -> BallDomainEvidence:
        return self._evidence

    @property
    def release_authorized(self) -> bool:
        """Schema-3 capabilities are permanently diagnostic-only.

        They predate authority-owned checkpoint/range allocation and trusted
        terminal classification.  They remain readable for legacy diagnostics
        but can never authorize a schema-4 curriculum release.
        """

        return False

    def __getattr__(self, name: str) -> object:
        return getattr(self._evidence, name)

    def __copy__(self) -> "FrozenEvaluationCapability":
        return self

    def __deepcopy__(
        self,
        memo: Dict[int, object],
    ) -> "FrozenEvaluationCapability":
        return self


@dataclass(frozen=True)
class _Window:
    evidence: BallDomainEvidence
    attempts: Tuple[_AttemptData, ...]
    capability_id: str
    attempt_storage: str = "full"

    def __post_init__(self) -> None:
        if self.attempt_storage not in ("full", "formal_compact"):
            raise ValueError("invalid evaluator attempt_storage")
        if self.attempt_storage == "formal_compact":
            if not self.is_formal or self.attempts:
                raise ValueError(
                    "formal_compact windows require formal evidence and "
                    "no retained attempts"
                )
        elif not self.attempts:
            raise ValueError("full evaluator windows require attempts")

    @property
    def ordered_sample_receipt_sha256(self) -> Tuple[str, ...]:
        return tuple(item.sample_receipt_sha256 for item in self.attempts)

    @property
    def ordered_birth_receipt_sha256(self) -> Tuple[str, ...]:
        return tuple(item.birth_receipt_sha256 for item in self.attempts)

    @property
    def is_formal(self) -> bool:
        return self.evidence.evidence_role in _FORMAL_ROLES

    @property
    def is_compact(self) -> bool:
        return self.attempt_storage == "formal_compact"

    def compact_formal(self) -> "_Window":
        if not self.is_formal:
            raise ValueError("only formal windows may be compacted")
        return _Window(
            evidence=self.evidence,
            attempts=(),
            capability_id=self.capability_id,
            attempt_storage="formal_compact",
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "evidence": self.evidence._hash_document(),
            "window_sha256": self.evidence.window_sha256,
            "attempt_storage": self.attempt_storage,
            "ordered_attempts": (
                None
                if self.is_compact
                else [item.as_dict() for item in self.attempts]
            ),
        }


class FrozenEvaluatorAuthority:
    """Stateful issuer and single-use ledger for evaluation windows."""

    _STATE_KEYS = (
        "schema_version",
        "authority_contract_sha256",
        "state_owner_sha256",
        "curriculum_contract_sha256",
        "profile_order",
        "arm_catalog_sha256",
        "scheduler_contract_sha256",
        "sampler_sha256",
        "solver_sha256",
        "policy_contract_sha256",
        "attempt_source_contract_sha256",
        "attempt_source_path",
        "attempt_source_sha256",
        "launch_receipt_sha256",
        "pending",
        "consumed",
        "consumed_hash_chain_sha256",
        "state_sha256",
    )

    def __init__(
        self,
        *,
        curriculum_contract_sha256: str,
        profile_order: Sequence[ActionProfileKey],
        arm_catalog_sha256: str,
        scheduler_contract_sha256: str,
        sampler_sha256: str,
        solver_sha256: str,
        policy_contract_sha256: str,
        attempt_source_contract_sha256: str = _ZERO_SHA,
        attempt_source_path: str = "diagnostic/unbound.py",
        attempt_source_sha256: str = _ZERO_SHA,
        _formal_sentinel: object | None = None,
        _launch_receipt_sha256: str | None = None,
    ) -> None:
        self._curriculum_contract_sha256 = _sha256(
            curriculum_contract_sha256,
            name="curriculum_contract_sha256",
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
        self._profile_order = order
        self._profile_set = frozenset(order)
        self._arm_catalog_sha256 = _sha256(
            arm_catalog_sha256,
            name="arm_catalog_sha256",
        )
        if self._arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("authority arm catalog does not match code")
        self._scheduler_contract_sha256 = _sha256(
            scheduler_contract_sha256,
            name="scheduler_contract_sha256",
        )
        self._sampler_sha256 = _sha256(sampler_sha256, name="sampler_sha256")
        self._solver_sha256 = _sha256(solver_sha256, name="solver_sha256")
        self._policy_contract_sha256 = _sha256(
            policy_contract_sha256,
            name="policy_contract_sha256",
        )
        self._attempt_source_contract_sha256 = _sha256(
            attempt_source_contract_sha256,
            name="attempt_source_contract_sha256",
        )
        self._attempt_source_path = _relative_path(
            attempt_source_path,
            name="attempt_source_path",
        )
        self._attempt_source_sha256 = _sha256(
            attempt_source_sha256,
            name="attempt_source_sha256",
        )
        self._formal = _formal_sentinel is _FORMAL_SENTINEL
        self._launch_receipt_sha256 = (
            _sha256(_launch_receipt_sha256, name="launch_receipt_sha256")
            if self._formal
            else _ZERO_SHA
        )
        self._state_owner_sha256 = _canonical_sha256(self.binding_document())
        self._pending: Dict[str, _Window] = {}
        self._consumed: Dict[str, _Window] = {}
        self._lifetime = object()

    @classmethod
    def from_trusted_launch_receipt(
        cls,
        receipt: object,
    ) -> "FrozenEvaluatorAuthority":
        row = _exact_keys(
            receipt,
            (
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
                "attempt_source_contract_sha256",
                "attempt_source_path",
                "attempt_source_sha256",
            ),
            name="frozen evaluator launch receipt",
        )
        if (
            row["schema_version"] != SCHEMA_VERSION
            or row["kind"] != "action_ball_frozen_evaluator_launch"
            or row["authority_contract_sha256"]
            != FROZEN_EVALUATOR_AUTHORITY_CONTRACT_SHA256
            or row["arm_catalog_sha256"] != ARM_CATALOG_SHA256
        ):
            raise FrozenEvaluationAuthorityError(
                "frozen evaluator launch receipt contract mismatch"
            )
        receipt_sha256 = _canonical_sha256(row)
        if receipt_sha256 not in TRUSTED_FROZEN_EVALUATOR_LAUNCH_RECEIPT_SHA256:
            raise FrozenEvaluationAuthorityError(
                "frozen evaluator launch receipt is not code-pinned"
            )
        raw_order = row["profile_order"]
        if not isinstance(raw_order, list):
            raise FrozenEvaluationAuthorityError(
                "launch receipt profile_order must be a list"
            )
        order = []
        for index, item in enumerate(raw_order):
            key_row = _exact_keys(
                item,
                ("action_uid", "profile_sha256", "mobility"),
                name=f"launch profile_order[{index}]",
            )
            order.append(ActionProfileKey(**key_row))
        return cls(
            curriculum_contract_sha256=row["curriculum_contract_sha256"],
            profile_order=tuple(order),
            arm_catalog_sha256=row["arm_catalog_sha256"],
            scheduler_contract_sha256=row["scheduler_contract_sha256"],
            sampler_sha256=row["sampler_sha256"],
            solver_sha256=row["solver_sha256"],
            policy_contract_sha256=row["policy_contract_sha256"],
            attempt_source_contract_sha256=row[
                "attempt_source_contract_sha256"
            ],
            attempt_source_path=row["attempt_source_path"],
            attempt_source_sha256=row["attempt_source_sha256"],
            _formal_sentinel=_FORMAL_SENTINEL,
            _launch_receipt_sha256=receipt_sha256,
        )

    @property
    def authority_contract_sha256(self) -> str:
        return FROZEN_EVALUATOR_AUTHORITY_CONTRACT_SHA256

    @property
    def release_authorized(self) -> bool:
        """Legacy schema-3 authority is never a release authority."""

        return False

    def assert_release_receipt(self, receipt: object) -> None:
        del receipt
        raise FrozenEvaluationAuthorityError(
            "legacy schema-3 evaluator cannot authorize schema-4 release"
        )

    @property
    def state_owner_sha256(self) -> str:
        return self._state_owner_sha256

    @property
    def profile_order(self) -> Tuple[ActionProfileKey, ...]:
        return self._profile_order

    @property
    def consumed_window_sha256(self) -> Tuple[str, ...]:
        return tuple(
            item.evidence.window_sha256
            for item in sorted(
                self._consumed.values(),
                key=lambda window: window.evidence.seq,
            )
        )

    def assert_formal_retention(
        self,
        expected_window_sha256: Sequence[str],
    ) -> None:
        """Verify that only in-flight canaries retain raw attempt rows."""

        if isinstance(expected_window_sha256, (str, bytes)):
            raise FrozenEvaluationAuthorityError(
                "expected formal retention must be a sequence"
            )
        expected_rows = tuple(
            _sha256(
                item,
                name=f"expected_window_sha256[{index}]",
            )
            for index, item in enumerate(expected_window_sha256)
        )
        if len(expected_rows) != len(set(expected_rows)):
            raise FrozenEvaluationAuthorityError(
                "expected formal retention contains duplicates"
            )
        actual = {
            window.evidence.window_sha256
            for window in self._consumed.values()
            if window.is_formal and not window.is_compact
        }
        if actual != set(expected_rows):
            raise FrozenEvaluationAuthorityError(
                "formal attempt retention does not match in-flight canaries"
            )

    def binding_document(self) -> Dict[str, object]:
        return {
            "authority_contract_sha256": (
                FROZEN_EVALUATOR_AUTHORITY_CONTRACT_SHA256
            ),
            "curriculum_contract_sha256": self._curriculum_contract_sha256,
            "profile_order": [key.as_dict() for key in self._profile_order],
            "arm_catalog_sha256": self._arm_catalog_sha256,
            "scheduler_contract_sha256": self._scheduler_contract_sha256,
            "sampler_sha256": self._sampler_sha256,
            "solver_sha256": self._solver_sha256,
            "policy_contract_sha256": self._policy_contract_sha256,
            "attempt_source_contract_sha256": (
                self._attempt_source_contract_sha256
            ),
            "attempt_source_path": self._attempt_source_path,
            "attempt_source_sha256": self._attempt_source_sha256,
            "launch_receipt_sha256": self._launch_receipt_sha256,
        }

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
        if not self._formal:
            raise FrozenEvaluationAuthorityError(
                "diagnostic evaluator authority is not formal; a code-pinned "
                "launch finalizer receipt is required"
            )
        expected = {
            **self.binding_document(),
            "curriculum_contract_sha256": curriculum_contract_sha256,
            "profile_order": [key.as_dict() for key in profile_order],
            "arm_catalog_sha256": arm_catalog_sha256,
            "scheduler_contract_sha256": scheduler_contract_sha256,
            "sampler_sha256": sampler_sha256,
            "solver_sha256": solver_sha256,
            "policy_contract_sha256": policy_contract_sha256,
        }
        if self.binding_document() != expected:
            raise FrozenEvaluationAuthorityError(
                "frozen evaluator authority binding mismatch"
            )

    def record_attempt(
        self,
        *,
        sample_receipt_sha256: str,
        birth_receipt_sha256: str,
        solver_admitted: bool,
        installed: bool,
        started: bool,
        closed: bool,
        terminal_outcome: str | None,
        in_new_band: bool,
        infrastructure_invalid: bool = False,
    ) -> FrozenAttemptReceipt:
        if not self._formal:
            raise FrozenEvaluationAuthorityError(
                "diagnostic authority cannot record formal attempts"
            )
        data = _AttemptData(
            sample_receipt_sha256=sample_receipt_sha256,
            birth_receipt_sha256=birth_receipt_sha256,
            solver_admitted=solver_admitted,
            installed=installed,
            started=started,
            closed=closed,
            terminal_outcome=terminal_outcome,
            infrastructure_invalid=infrastructure_invalid,
            in_new_band=in_new_band,
        )
        return FrozenAttemptReceipt(_MINT_SENTINEL, self, self._lifetime, data)

    def issue_window(
        self,
        *,
        key: ActionProfileKey,
        policy_checkpoint_sha256: str,
        policy_generation: int,
        evidence_role: str,
        domain_epoch: int,
        stratum: str,
        selected_arm_key: str,
        selection_round: int,
        arm_levels: Tuple[float, ...],
        rho: float,
        seed_block_start: int,
        seed_block_end_exclusive: int,
        sample_id_start: int,
        sample_id_end_exclusive: int,
        seq: int,
        window_id: str,
        ordered_attempt_receipts: Sequence[FrozenAttemptReceipt],
    ) -> FrozenEvaluationCapability:
        if not self._formal:
            raise FrozenEvaluationAuthorityError(
                "diagnostic authority cannot issue formal capabilities"
            )
        if key not in self._profile_set:
            raise FrozenEvaluationAuthorityError(
                "evaluation key is outside the frozen profile order"
            )
        if isinstance(ordered_attempt_receipts, (str, bytes)):
            raise FrozenEvaluationAuthorityError(
                "ordered attempt receipts must be a sequence"
            )
        attempts = []
        attempt_ids = set()
        for index, receipt in enumerate(ordered_attempt_receipts):
            if type(receipt) is not FrozenAttemptReceipt:
                raise FrozenEvaluationAuthorityError(
                    f"attempt[{index}] is not an opaque attempt receipt"
                )
            if (
                receipt._authority is not self
                or receipt._lifetime is not self._lifetime
            ):
                raise FrozenEvaluationAuthorityError(
                    f"attempt[{index}] belongs to another or restored authority"
                )
            if receipt.receipt_sha256 in attempt_ids:
                raise FrozenEvaluationAuthorityError(
                    "one evaluation window cannot reuse an attempt receipt"
                )
            attempt_ids.add(receipt.receipt_sha256)
            attempts.append(receipt._data)
        attempts_tuple = tuple(attempts)
        ledger = _ledger_from_attempts(attempts_tuple)
        sample_receipts = tuple(
            item.sample_receipt_sha256 for item in attempts_tuple
        )
        birth_receipts = tuple(
            item.birth_receipt_sha256 for item in attempts_tuple
        )
        sample_root = ordered_sample_receipt_root(sample_receipts)
        birth_root = ordered_birth_receipt_root(birth_receipts)
        unique_birth_count = len(set(birth_receipts))
        if evidence_role in _FORMAL_ROLES and unique_birth_count != ledger.P:
            raise FrozenEvaluationAuthorityError(
                "formal windows require one unique birth per proposal"
            )
        evidence = BallDomainEvidence.create(
            key=key,
            arm_catalog_sha256=self._arm_catalog_sha256,
            scheduler_contract_sha256=self._scheduler_contract_sha256,
            sampler_sha256=self._sampler_sha256,
            solver_sha256=self._solver_sha256,
            policy_contract_sha256=self._policy_contract_sha256,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
            policy_generation=policy_generation,
            evidence_role=evidence_role,
            domain_epoch=domain_epoch,
            stratum=stratum,
            selected_arm_key=selected_arm_key,
            selection_round=selection_round,
            arm_levels=arm_levels,
            rho=rho,
            seed_block_start=seed_block_start,
            seed_block_end_exclusive=seed_block_end_exclusive,
            sample_id_start=sample_id_start,
            sample_id_end_exclusive=sample_id_end_exclusive,
            sample_receipt_root_sha256=sample_root,
            unique_birth_count=unique_birth_count,
            birth_receipt_root_sha256=birth_root,
            seq=seq,
            window_id=window_id,
            ledger=ledger,
        )
        all_windows = tuple(self._pending.values()) + tuple(
            self._consumed.values()
        )
        if all_windows and evidence.seq <= max(
            item.evidence.seq for item in all_windows
        ):
            raise FrozenEvaluationAuthorityError(
                "evaluation seq must increase monotonically"
            )
        checkpoint_by_generation = {
            item.evidence.policy_generation: (
                item.evidence.policy_checkpoint_sha256
            )
            for item in all_windows
        }
        if all_windows and evidence.policy_generation < max(
            item.evidence.policy_generation for item in all_windows
        ):
            raise FrozenEvaluationAuthorityError(
                "policy generation cannot regress"
            )
        prior_checkpoint = checkpoint_by_generation.get(
            evidence.policy_generation
        )
        if (
            prior_checkpoint is not None
            and prior_checkpoint != evidence.policy_checkpoint_sha256
        ):
            raise FrozenEvaluationAuthorityError(
                "one policy generation cannot name multiple checkpoints"
            )
        for existing in all_windows:
            prior = existing.evidence
            if prior.window_id == evidence.window_id:
                raise FrozenEvaluationAuthorityError(
                    "evaluation window_id was already issued"
                )
            if prior.key.action_uid != key.action_uid:
                continue
            if set(existing.ordered_sample_receipt_sha256).intersection(
                sample_receipts
            ):
                raise FrozenEvaluationAuthorityError(
                    "same-action sample receipt was already issued"
                )
            if _intervals_overlap(
                prior.sample_id_start,
                prior.sample_id_end_exclusive,
                evidence.sample_id_start,
                evidence.sample_id_end_exclusive,
            ):
                raise FrozenEvaluationAuthorityError(
                    "same-action sample ranges overlap"
                )
            if _intervals_overlap(
                prior.seed_block_start,
                prior.seed_block_end_exclusive,
                evidence.seed_block_start,
                evidence.seed_block_end_exclusive,
            ):
                raise FrozenEvaluationAuthorityError(
                    "same-action seed ranges overlap"
                )
            if (
                evidence_role in _FORMAL_ROLES
                and existing.is_formal
                and set(existing.ordered_birth_receipt_sha256).intersection(
                    birth_receipts
                )
            ):
                raise FrozenEvaluationAuthorityError(
                    "same-action frozen windows reuse a birth receipt"
                )
        capability_id = _canonical_sha256(
            {
                "state_owner_sha256": self._state_owner_sha256,
                "window_sha256": evidence.window_sha256,
                "sample_receipt_root_sha256": sample_root,
                "birth_receipt_root_sha256": birth_root,
                "unique_birth_count": unique_birth_count,
            }
        )
        if capability_id in self._pending or capability_id in self._consumed:
            raise FrozenEvaluationAuthorityError(
                "evaluation window was already issued"
            )
        window = _Window(evidence, attempts_tuple, capability_id)
        self._pending[capability_id] = window
        return self._mint(window)

    def _mint(self, window: _Window) -> FrozenEvaluationCapability:
        return FrozenEvaluationCapability(
            _MINT_SENTINEL,
            self,
            self._lifetime,
            window.capability_id,
            window.evidence,
        )

    def pending_capability(
        self,
        capability_id: str,
    ) -> FrozenEvaluationCapability:
        digest = _sha256(capability_id, name="capability_id")
        try:
            return self._mint(self._pending[digest])
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "capability is not pending"
            ) from exc

    def _window_for(
        self,
        capability: FrozenEvaluationCapability,
    ) -> _Window:
        if type(capability) is not FrozenEvaluationCapability:
            raise FrozenEvaluationAuthorityError(
                "formal curriculum requires an opaque evaluation capability"
            )
        if (
            capability._authority is not self
            or capability._lifetime is not self._lifetime
        ):
            raise FrozenEvaluationAuthorityError(
                "capability belongs to another or restored authority"
            )
        window = self._pending.get(capability._capability_id)
        if window is None or capability._evidence != window.evidence:
            raise FrozenEvaluationAuthorityError(
                "capability is stale, consumed, or inconsistent"
            )
        return window

    def inspect_many(
        self,
        capabilities: Mapping[
            ActionProfileKey,
            FrozenEvaluationCapability,
        ],
    ) -> Dict[ActionProfileKey, BallDomainEvidence]:
        if not isinstance(capabilities, Mapping) or not capabilities:
            raise FrozenEvaluationAuthorityError(
                "capabilities must be a non-empty mapping"
            )
        result: Dict[ActionProfileKey, BallDomainEvidence] = {}
        ids = set()
        for key, capability in capabilities.items():
            window = self._window_for(capability)
            if key != window.evidence.key:
                raise FrozenEvaluationAuthorityError(
                    "capability profile key mismatch"
                )
            if window.capability_id in ids:
                raise FrozenEvaluationAuthorityError(
                    "duplicate capability in atomic update"
                )
            ids.add(window.capability_id)
            result[key] = window.evidence
        return result

    def attempt_rows_many(
        self,
        capabilities: Mapping[
            ActionProfileKey,
            FrozenEvaluationCapability,
        ],
    ) -> Dict[ActionProfileKey, Tuple[Dict[str, object], ...]]:
        """Return detached exact rows for rolling scheduler bookkeeping."""

        self.inspect_many(capabilities)
        return {
            key: tuple(
                item.as_dict()
                for item in self._window_for(capability).attempts
            )
            for key, capability in capabilities.items()
        }

    def consume_many(
        self,
        capabilities: Mapping[
            ActionProfileKey,
            FrozenEvaluationCapability,
        ],
        *,
        retain_formal_window_sha256: Sequence[str] | None = None,
    ) -> Dict[ActionProfileKey, BallDomainEvidence]:
        """Consume atomically and compact completed formal transcripts.

        ``retain_formal_window_sha256`` names the exact canaries that remain
        in-flight while a disjoint heldout is collected.  ``None`` is the
        conservative standalone default: retain every consumed canary.
        """

        evidence = self.inspect_many(capabilities)
        windows = [
            self._pending[capability.capability_id]
            for capability in capabilities.values()
        ]
        existing_samples: Dict[int, set[str]] = {}
        existing_formal_births: Dict[int, set[str]] = {}
        for window in self._consumed.values():
            uid = window.evidence.key.action_uid
            existing_samples.setdefault(uid, set()).update(
                window.ordered_sample_receipt_sha256
            )
            if window.is_formal:
                existing_formal_births.setdefault(uid, set()).update(
                    window.ordered_birth_receipt_sha256
                )
        staged_samples: Dict[int, set[str]] = {}
        staged_formal_births: Dict[int, set[str]] = {}
        for window in windows:
            uid = window.evidence.key.action_uid
            samples = set(window.ordered_sample_receipt_sha256)
            if samples & existing_samples.get(uid, set()):
                raise FrozenEvaluationAuthorityError(
                    "same-action sample receipt was already consumed"
                )
            if samples & staged_samples.setdefault(uid, set()):
                raise FrozenEvaluationAuthorityError(
                    "atomic update reuses a same-action sample receipt"
                )
            staged_samples[uid].update(samples)
            if window.is_formal:
                births = set(window.ordered_birth_receipt_sha256)
                if births & existing_formal_births.get(uid, set()):
                    raise FrozenEvaluationAuthorityError(
                        "same-action frozen birth receipt was already consumed"
                    )
                if births & staged_formal_births.setdefault(uid, set()):
                    raise FrozenEvaluationAuthorityError(
                        "atomic frozen update reuses a same-action birth receipt"
                    )
                staged_formal_births[uid].update(births)
        proposed_consumed = dict(self._consumed)
        proposed_consumed.update(
            (window.capability_id, window) for window in windows
        )
        if retain_formal_window_sha256 is None:
            retained = {
                window.evidence.window_sha256
                for window in proposed_consumed.values()
                if (
                    window.evidence.evidence_role == "frozen_canary"
                    and not window.is_compact
                )
            }
        else:
            if isinstance(retain_formal_window_sha256, (str, bytes)):
                raise FrozenEvaluationAuthorityError(
                    "retained formal windows must be a sequence"
                )
            retained_rows = tuple(
                _sha256(
                    item,
                    name=f"retain_formal_window_sha256[{index}]",
                )
                for index, item in enumerate(
                    retain_formal_window_sha256
                )
            )
            if len(retained_rows) != len(set(retained_rows)):
                raise FrozenEvaluationAuthorityError(
                    "retained formal window identities are duplicated"
                )
            retained = set(retained_rows)
        retainable = {
            window.evidence.window_sha256
            for window in proposed_consumed.values()
            if (
                window.evidence.evidence_role == "frozen_canary"
                and not window.is_compact
            )
        }
        if not retained.issubset(retainable):
            raise FrozenEvaluationAuthorityError(
                "only full consumed canaries may remain in-flight"
            )
        compacted_consumed = {
            capability_id: (
                window
                if (
                    not window.is_formal
                    or window.evidence.window_sha256 in retained
                    or window.is_compact
                )
                else window.compact_formal()
            )
            for capability_id, window in proposed_consumed.items()
        }
        proposed_pending = dict(self._pending)
        for window in windows:
            del proposed_pending[window.capability_id]
        self._pending = proposed_pending
        self._consumed = compacted_consumed
        return evidence

    def state_dict(self) -> Dict[str, object]:
        consumed = sorted(
            self._consumed.values(),
            key=lambda item: item.evidence.seq,
        )
        chain = _ZERO_SHA
        for window in consumed:
            chain = hashlib.sha256(
                (chain + window.capability_id).encode("ascii")
            ).hexdigest()
        document = {
            "schema_version": STATE_SCHEMA_VERSION,
            **self.binding_document(),
            "state_owner_sha256": self._state_owner_sha256,
            "pending": [
                item.as_dict()
                for item in sorted(
                    self._pending.values(),
                    key=lambda window: window.evidence.seq,
                )
            ],
            "consumed": [item.as_dict() for item in consumed],
            "consumed_hash_chain_sha256": chain,
        }
        document["state_sha256"] = _canonical_sha256(document)
        return document

    def load_state_dict(self, state: object) -> None:
        row = _exact_keys(
            state,
            self._STATE_KEYS,
            name="frozen evaluator authority state",
        )
        digest = _sha256(row["state_sha256"], name="state_sha256")
        unsigned = dict(row)
        del unsigned["state_sha256"]
        if _canonical_sha256(unsigned) != digest:
            raise FrozenEvaluationAuthorityError(
                "frozen evaluator authority state digest mismatch"
            )
        if (
            _plain_int(row["schema_version"], name="schema_version", minimum=1)
            != STATE_SCHEMA_VERSION
        ):
            raise FrozenEvaluationAuthorityError(
                "unsupported frozen evaluator authority state schema"
            )
        for field, expected in {
            **self.binding_document(),
            "state_owner_sha256": self._state_owner_sha256,
        }.items():
            if row[field] != expected:
                raise FrozenEvaluationAuthorityError(
                    f"frozen evaluator authority {field} mismatch"
                )
        pending = self._parse_windows(
            row["pending"],
            name="pending",
            allow_compact=False,
        )
        consumed = self._parse_windows(
            row["consumed"],
            name="consumed",
            allow_compact=True,
        )
        all_windows = tuple(pending.values()) + tuple(consumed.values())
        ids = [item.capability_id for item in all_windows]
        window_ids = [item.evidence.window_id for item in all_windows]
        seqs = [item.evidence.seq for item in all_windows]
        if (
            len(ids) != len(set(ids))
            or len(window_ids) != len(set(window_ids))
            or len(seqs) != len(set(seqs))
        ):
            raise FrozenEvaluationAuthorityError(
                "authority state contains duplicate window identity"
            )
        checkpoint_by_generation: Dict[int, str] = {}
        previous_generation = 0
        for window in sorted(all_windows, key=lambda item: item.evidence.seq):
            evidence = window.evidence
            if evidence.policy_generation < previous_generation:
                raise FrozenEvaluationAuthorityError(
                    "policy generation regressed in authority state"
                )
            previous_generation = evidence.policy_generation
            previous = checkpoint_by_generation.setdefault(
                evidence.policy_generation,
                evidence.policy_checkpoint_sha256,
            )
            if previous != evidence.policy_checkpoint_sha256:
                raise FrozenEvaluationAuthorityError(
                    "one policy generation names multiple checkpoints"
                )
        self._validate_global_windows(all_windows)
        chain = _ZERO_SHA
        for window in sorted(
            consumed.values(),
            key=lambda item: item.evidence.seq,
        ):
            chain = hashlib.sha256(
                (chain + window.capability_id).encode("ascii")
            ).hexdigest()
        if (
            _sha256(
                row["consumed_hash_chain_sha256"],
                name="consumed_hash_chain_sha256",
            )
            != chain
        ):
            raise FrozenEvaluationAuthorityError(
                "consumed authority hash chain mismatch"
            )
        self._pending = pending
        self._consumed = consumed
        self._lifetime = object()

    @staticmethod
    def _evidence_fields() -> Tuple[str, ...]:
        return (
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

    def _parse_windows(
        self,
        value: object,
        *,
        name: str,
        allow_compact: bool,
    ) -> Dict[str, _Window]:
        if not isinstance(value, list):
            raise FrozenEvaluationAuthorityError(f"{name} windows must be a list")
        result: Dict[str, _Window] = {}
        for index, item in enumerate(value):
            row = _exact_keys(
                item,
                (
                    "capability_id",
                    "evidence",
                    "window_sha256",
                    "attempt_storage",
                    "ordered_attempts",
                ),
                name=f"{name}[{index}]",
            )
            evidence_row = _exact_keys(
                row["evidence"],
                self._evidence_fields(),
                name=f"{name}[{index}].evidence",
            )
            if evidence_row["schema_version"] != EVIDENCE_SCHEMA_VERSION:
                raise FrozenEvaluationAuthorityError(
                    "legacy evidence schema is unsupported"
                )
            key_row = _exact_keys(
                evidence_row["key"],
                ("action_uid", "profile_sha256", "mobility"),
                name=f"{name}[{index}].evidence.key",
            )
            key = ActionProfileKey(**key_row)
            if key not in self._profile_set:
                raise FrozenEvaluationAuthorityError(
                    "authority state contains an unknown profile"
                )
            ledger_row = _exact_keys(
                evidence_row["ledger"],
                (
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
                    "U_joint_qdes",
                    "U_joint_actual",
                    "X",
                    "NB",
                    "NB_F",
                ),
                name=f"{name}[{index}].evidence.ledger",
            )
            ledger = BallOutcomeLedger(**ledger_row)
            evidence = BallDomainEvidence.create(
                key=key,
                arm_catalog_sha256=evidence_row["arm_catalog_sha256"],
                scheduler_contract_sha256=evidence_row[
                    "scheduler_contract_sha256"
                ],
                sampler_sha256=evidence_row["sampler_sha256"],
                solver_sha256=evidence_row["solver_sha256"],
                policy_contract_sha256=evidence_row["policy_contract_sha256"],
                policy_checkpoint_sha256=evidence_row[
                    "policy_checkpoint_sha256"
                ],
                policy_generation=evidence_row["policy_generation"],
                evidence_role=evidence_row["evidence_role"],
                domain_epoch=evidence_row["domain_epoch"],
                stratum=evidence_row["stratum"],
                selected_arm_key=evidence_row["selected_arm_key"],
                selection_round=evidence_row["selection_round"],
                arm_levels=tuple(evidence_row["arm_levels"]),
                rho=evidence_row["rho"],
                seed_block_start=evidence_row["seed_block_start"],
                seed_block_end_exclusive=evidence_row[
                    "seed_block_end_exclusive"
                ],
                sample_id_start=evidence_row["sample_id_start"],
                sample_id_end_exclusive=evidence_row[
                    "sample_id_end_exclusive"
                ],
                sample_receipt_root_sha256=evidence_row[
                    "sample_receipt_root_sha256"
                ],
                unique_birth_count=evidence_row["unique_birth_count"],
                birth_receipt_root_sha256=evidence_row[
                    "birth_receipt_root_sha256"
                ],
                seq=evidence_row["seq"],
                window_id=evidence_row["window_id"],
                ledger=ledger,
            )
            if evidence.window_sha256 != _sha256(
                row["window_sha256"],
                name=f"{name}[{index}].window_sha256",
            ):
                raise FrozenEvaluationAuthorityError(
                    "authority window evidence hash mismatch"
                )
            storage = row["attempt_storage"]
            if storage not in ("full", "formal_compact"):
                raise FrozenEvaluationAuthorityError(
                    "invalid authority attempt_storage"
                )
            attempts_raw = row["ordered_attempts"]
            if storage == "formal_compact":
                if (
                    not allow_compact
                    or evidence.evidence_role not in _FORMAL_ROLES
                    or attempts_raw is not None
                ):
                    raise FrozenEvaluationAuthorityError(
                        "formal_compact is valid only for consumed formal "
                        "windows without attempt rows"
                    )
                attempts_tuple: Tuple[_AttemptData, ...] = ()
                sample_root = evidence.sample_receipt_root_sha256
                birth_root = evidence.birth_receipt_root_sha256
                unique_birth_count = evidence.unique_birth_count
            else:
                if not isinstance(attempts_raw, list):
                    raise FrozenEvaluationAuthorityError(
                        "full ordered attempts must be a list"
                    )
                attempts = []
                for attempt_index, attempt_raw in enumerate(attempts_raw):
                    attempt_row = _exact_keys(
                        attempt_raw,
                        (
                            "sample_receipt_sha256",
                            "birth_receipt_sha256",
                            "solver_admitted",
                            "installed",
                            "started",
                            "closed",
                            "terminal_outcome",
                            "infrastructure_invalid",
                            "in_new_band",
                        ),
                        name=(
                            f"{name}[{index}].attempt[{attempt_index}]"
                        ),
                    )
                    attempts.append(_AttemptData(**attempt_row))
                attempts_tuple = tuple(attempts)
                if _ledger_from_attempts(attempts_tuple) != ledger:
                    raise FrozenEvaluationAuthorityError(
                        "authority window ledger is not derived from attempts"
                    )
                sample_root = ordered_sample_receipt_root(
                    tuple(
                        item.sample_receipt_sha256
                        for item in attempts_tuple
                    )
                )
                births = tuple(
                    item.birth_receipt_sha256 for item in attempts_tuple
                )
                birth_root = ordered_birth_receipt_root(births)
                unique_birth_count = len(set(births))
                if (
                    sample_root != evidence.sample_receipt_root_sha256
                    or birth_root != evidence.birth_receipt_root_sha256
                    or unique_birth_count != evidence.unique_birth_count
                ):
                    raise FrozenEvaluationAuthorityError(
                        "ordered sample/birth receipt evidence mismatch"
                    )
                if (
                    allow_compact
                    and evidence.evidence_role == "frozen_heldout"
                ):
                    raise FrozenEvaluationAuthorityError(
                        "consumed heldout must use formal_compact storage"
                    )
            if (
                evidence.evidence_role in _FORMAL_ROLES
                and unique_birth_count != ledger.P
            ):
                raise FrozenEvaluationAuthorityError(
                    "formal authority state reuses a birth"
                )
            capability_id = _sha256(
                row["capability_id"],
                name=f"{name}[{index}].capability_id",
            )
            expected_id = _canonical_sha256(
                {
                    "state_owner_sha256": self._state_owner_sha256,
                    "window_sha256": evidence.window_sha256,
                    "sample_receipt_root_sha256": sample_root,
                    "birth_receipt_root_sha256": birth_root,
                    "unique_birth_count": unique_birth_count,
                }
            )
            if capability_id != expected_id:
                raise FrozenEvaluationAuthorityError(
                    "authority capability identity mismatch"
                )
            if capability_id in result:
                raise FrozenEvaluationAuthorityError(
                    "duplicate capability identity"
                )
            result[capability_id] = _Window(
                evidence,
                attempts_tuple,
                capability_id,
                storage,
            )
        return result

    @staticmethod
    def _validate_global_windows(windows: Sequence[_Window]) -> None:
        samples_by_action: Dict[int, set[str]] = {}
        formal_births_by_action: Dict[int, set[str]] = {}
        prior_by_action: Dict[int, list[_Window]] = {}
        for window in sorted(windows, key=lambda item: item.evidence.seq):
            evidence = window.evidence
            uid = evidence.key.action_uid
            samples = set(window.ordered_sample_receipt_sha256)
            if samples & samples_by_action.setdefault(uid, set()):
                raise FrozenEvaluationAuthorityError(
                    "authority state reuses a same-action sample receipt"
                )
            samples_by_action[uid].update(samples)
            if window.is_formal:
                births = set(window.ordered_birth_receipt_sha256)
                if births & formal_births_by_action.setdefault(uid, set()):
                    raise FrozenEvaluationAuthorityError(
                        "authority state reuses a same-action frozen birth"
                    )
                formal_births_by_action[uid].update(births)
            for prior in prior_by_action.setdefault(uid, []):
                if _intervals_overlap(
                    prior.evidence.sample_id_start,
                    prior.evidence.sample_id_end_exclusive,
                    evidence.sample_id_start,
                    evidence.sample_id_end_exclusive,
                ):
                    raise FrozenEvaluationAuthorityError(
                        "authority state overlaps same-action sample ranges"
                    )
                if _intervals_overlap(
                    prior.evidence.seed_block_start,
                    prior.evidence.seed_block_end_exclusive,
                    evidence.seed_block_start,
                    evidence.seed_block_end_exclusive,
                ):
                    raise FrozenEvaluationAuthorityError(
                        "authority state overlaps same-action seed ranges"
                    )
            prior_by_action[uid].append(window)


# ---------------------------------------------------------------------------
# Schema 4: authority-owned frozen evaluation
# ---------------------------------------------------------------------------

V4_SCHEMA_VERSION = 4
V4_SCHEDULER_PROPOSALS = 100
V4_CANARY_PROPOSALS = 320
V4_CANARY_SAFE_CLOSED_MIN = 256
V4_HELDOUT_PROPOSALS = 960
V4_HELDOUT_SAFE_CLOSED_MIN = 768
V4_SAMPLING_MIXTURE = {
    "center": 0.20,
    "interior": 0.60,
    "frontier": 0.20,
}
_V4_MINT_SENTINEL = object()
_V4_FORMAL_SENTINEL = object()

_V4_AUTHORITY_DOCUMENT = {
    "schema_version": V4_SCHEMA_VERSION,
    "kind": "action_ball_frozen_evaluator_v4_authority",
    "scope": "same-process opaque capabilities only",
    "policy": (
        "authority hashes immutable checkpoint bytes and privately allocates "
        "monotonic generation"
    ),
    "allocation": (
        "authority privately allocates disjoint seed/sample/birth ranges; "
        "100 scheduler, 320 canary, and 960 heldout proposals; optional "
        "stopping is forbidden"
    ),
    "sampling_mixture": (
        "every fixed window is exactly 20 percent center, 60 percent "
        "interior, and 20 percent selected frontier"
    ),
    "attempt": (
        "proposal -> exact sampler issue -> solver reject/admit -> install -> "
        "start -> trusted raw sensor close; unresolved crash reservations "
        "are burned as X"
    ),
    "new_band": (
        "exact source replays the issued sampler row; evaluator recomputes "
        "frontier-arm membership and accepts no caller boolean"
    ),
    "terminal_precedence": (
        "X > joint_actual_limit > joint_qdes_limit > fall > table_hit > "
        "collision > legal_return > safe_nonreturn"
    ),
    "release": (
        "opaque schema-4 canary+heldout receipt; safe-closed floors 256/768; "
        "legacy schema-3 evidence is permanently release_authorized=false"
    ),
}
FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256 = _canonical_sha256(
    _V4_AUTHORITY_DOCUMENT
)
TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256 = frozenset()


def _runtime_v4():
    global _attempt_runtime
    if _attempt_runtime is None:
        _attempt_runtime = sys.modules.get("action_ball_runtime")
    if _attempt_runtime is None:
        raise FrozenEvaluationAuthorityError(
            "schema-4 evaluator requires action_ball_runtime in the same "
            "process"
        )
    return _attempt_runtime


def launch_receipt_document_v4(
    *,
    curriculum_contract_sha256: str,
    profile_order: Sequence[ActionProfileKey],
    arm_catalog_sha256: str,
    scheduler_contract_sha256: str,
    sampler_sha256: str,
    solver_sha256: str,
    policy_contract_sha256: str,
    attempt_source_contract_sha256: str,
    attempt_source_path: str,
    attempt_source_sha256: str,
) -> Dict[str, object]:
    """Build the exact code-pinned launch input for the V4 authority."""

    if isinstance(profile_order, (str, bytes)):
        raise ValueError("profile_order must be a sequence")
    raw_order = tuple(profile_order)
    if not raw_order:
        raise ValueError(
            "profile_order must contain unique ActionProfileKey values"
        )
    try:
        order = tuple(
            canonical_action_profile_key(key) for key in raw_order
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "profile_order must contain unique ActionProfileKey values"
        ) from exc
    if len(order) != len(set(order)):
        raise ValueError(
            "profile_order must contain unique ActionProfileKey values"
        )
    catalog = _sha256(arm_catalog_sha256, name="arm_catalog_sha256")
    if catalog != ARM_CATALOG_SHA256:
        raise ValueError("launch arm catalog does not match code")
    return {
        "schema_version": V4_SCHEMA_VERSION,
        "kind": "action_ball_frozen_evaluator_v4_launch",
        "authority_contract_sha256": (
            FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
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
            sampler_sha256,
            name="sampler_sha256",
        ),
        "solver_sha256": _sha256(
            solver_sha256,
            name="solver_sha256",
        ),
        "policy_contract_sha256": _sha256(
            policy_contract_sha256,
            name="policy_contract_sha256",
        ),
        "attempt_source_contract_sha256": _sha256(
            attempt_source_contract_sha256,
            name="attempt_source_contract_sha256",
        ),
        "attempt_source_path": _relative_path(
            attempt_source_path,
            name="attempt_source_path",
        ),
        "attempt_source_sha256": _sha256(
            attempt_source_sha256,
            name="attempt_source_sha256",
        ),
        "window_contract": {
            "optional_stopping": False,
            "scheduler_proposals": V4_SCHEDULER_PROPOSALS,
            "canary_proposals": V4_CANARY_PROPOSALS,
            "canary_safe_closed_min": V4_CANARY_SAFE_CLOSED_MIN,
            "heldout_proposals": V4_HELDOUT_PROPOSALS,
            "heldout_safe_closed_min": V4_HELDOUT_SAFE_CLOSED_MIN,
            "sampling_mixture": dict(V4_SAMPLING_MIXTURE),
        },
    }


class FrozenPolicySnapshotV4:
    """Opaque immutable checkpoint identity minted from raw bytes."""

    __slots__ = (
        "_authority",
        "_lifetime",
        "_checkpoint_sha256",
        "_generation",
    )

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorV4Authority",
        lifetime: object,
        checkpoint_sha256: str,
        generation: int,
    ) -> None:
        if sentinel is not _V4_MINT_SENTINEL:
            raise TypeError("policy snapshots are authority-minted")
        self._authority = authority
        self._lifetime = lifetime
        self._checkpoint_sha256 = checkpoint_sha256
        self._generation = generation

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def generation(self) -> int:
        return self._generation


@dataclass
class _V4Attempt:
    request: object
    proposal: object | None = None
    solver: object | None = None
    install: object | None = None
    start: object | None = None
    terminal: object | None = None
    in_new_band: bool = False
    status: str = "reserved"

    @property
    def settled(self) -> bool:
        return self.status in ("solver_rejected", "closed", "burned_x")

    def sample_receipt_sha256(self) -> str:
        if self.proposal is not None:
            return self.proposal.sample_receipt_sha256
        return _canonical_sha256(
            {
                "schema_version": V4_SCHEMA_VERSION,
                "kind": "burned_sample_reservation",
                "reservation_sha256": self.request.reservation_sha256,
                "sample_index": self.request.sample_index,
            }
        )

    def birth_receipt_sha256(self) -> str:
        if self.proposal is not None:
            return self.proposal.birth_receipt_sha256
        return _canonical_sha256(
            {
                "schema_version": V4_SCHEMA_VERSION,
                "kind": "burned_birth_reservation",
                "reservation_sha256": self.request.reservation_sha256,
                "birth_index": self.request.birth_index,
            }
        )

    def row(self) -> Dict[str, object]:
        terminal_outcome = None
        terminal_signals = None
        infrastructure_invalid = self.status == "burned_x"
        if self.terminal is not None:
            terminal_outcome = self.terminal.terminal_outcome
            terminal_signals = self.terminal.signals.to_dict()
            infrastructure_invalid = (
                self.terminal.signals.infrastructure_invalid
            )
        reject_reason = ""
        if self.solver is not None:
            reject_reason = self.solver.reject_reason
        return {
            "schema_version": V4_SCHEMA_VERSION,
            "reservation_sha256": self.request.reservation_sha256,
            "policy_checkpoint_sha256": (
                self.request.policy_checkpoint_sha256
            ),
            "policy_generation": self.request.policy_generation,
            "seed": self.request.seed,
            "sample_index": self.request.sample_index,
            "birth_index": self.request.birth_index,
            "action_uid": self.request.action_uid,
            "profile_sha256": self.request.profile_sha256,
            "mobility_mode": self.request.mobility_mode,
            "domain_epoch": self.request.domain_epoch,
            "levels_sha256": (
                self.request.domain_levels.canonical_sha256
            ),
            "sampling_stratum": (
                ""
                if self.proposal is None
                else self.proposal.sampling_stratum
            ),
            "frontier_arm": (
                "" if self.proposal is None else self.proposal.frontier_arm
            ),
            "sample_receipt_sha256": self.sample_receipt_sha256(),
            "birth_receipt_sha256": self.birth_receipt_sha256(),
            "solver_admitted": bool(
                self.solver is not None
                and self.solver.disposition == "admitted"
            ),
            "reject_reason": reject_reason,
            "installed": self.install is not None,
            "started": self.start is not None,
            "closed": (
                self.terminal is not None
                and not infrastructure_invalid
            ),
            "terminal_outcome": terminal_outcome,
            "terminal_signals": terminal_signals,
            "infrastructure_invalid": infrastructure_invalid,
            "in_new_band": self.in_new_band,
            "status": self.status,
        }


@dataclass
class _V4Window:
    allocation_sha256: str
    snapshot: FrozenPolicySnapshotV4
    key: ActionProfileKey
    evidence_role: str
    domain_epoch: int
    stratum: str
    selected_arm_key: str
    selection_round: int
    arm_levels: Tuple[float, ...]
    rho: float
    seq: int
    seed_start: int
    sample_start: int
    birth_start: int
    attempts: list[_V4Attempt]
    finalized: bool = False
    evidence: BallDomainEvidence | None = None
    capability_id: str = ""

    @property
    def proposal_count(self) -> int:
        return len(self.attempts)


class FrozenEvaluationSessionV4:
    """Opaque live window handle; valid only inside the minting process."""

    __slots__ = ("_authority", "_lifetime", "_allocation_sha256")

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorV4Authority",
        lifetime: object,
        allocation_sha256: str,
    ) -> None:
        if sentinel is not _V4_MINT_SENTINEL:
            raise TypeError("evaluation sessions are authority-minted")
        self._authority = authority
        self._lifetime = lifetime
        self._allocation_sha256 = allocation_sha256


class FrozenProposalHandleV4:
    """Opaque single-attempt handle; never a JSON/wire capability."""

    __slots__ = (
        "_authority",
        "_lifetime",
        "_allocation_sha256",
        "_offset",
    )

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorV4Authority",
        lifetime: object,
        allocation_sha256: str,
        offset: int,
    ) -> None:
        if sentinel is not _V4_MINT_SENTINEL:
            raise TypeError("proposal handles are authority-minted")
        self._authority = authority
        self._lifetime = lifetime
        self._allocation_sha256 = allocation_sha256
        self._offset = offset


class FrozenEvaluationCapabilityV4:
    """Opaque completed fixed-size window capability."""

    __slots__ = (
        "_authority",
        "_lifetime",
        "_capability_id",
        "_evidence",
    )

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorV4Authority",
        lifetime: object,
        capability_id: str,
        evidence: BallDomainEvidence,
    ) -> None:
        if sentinel is not _V4_MINT_SENTINEL:
            raise TypeError("V4 capabilities are authority-minted")
        self._authority = authority
        self._lifetime = lifetime
        self._capability_id = capability_id
        self._evidence = evidence

    @property
    def capability_id(self) -> str:
        return self._capability_id

    @property
    def evidence(self) -> BallDomainEvidence:
        return self._evidence

    @property
    def release_authorized(self) -> bool:
        return self._evidence.evidence_role in (
            "frozen_canary",
            "frozen_heldout",
        )


class FrozenEvaluationReleaseReceipt:
    """Opaque schema-4 canary+heldout receipt for curriculum staging."""

    __slots__ = (
        "_authority",
        "_lifetime",
        "_release_id",
        "_canary",
        "_heldout",
    )

    def __init__(
        self,
        sentinel: object,
        authority: "FrozenEvaluatorV4Authority",
        lifetime: object,
        release_id: str,
        canary: BallDomainEvidence,
        heldout: BallDomainEvidence,
    ) -> None:
        if sentinel is not _V4_MINT_SENTINEL:
            raise TypeError("release receipts are authority-minted")
        self._authority = authority
        self._lifetime = lifetime
        self._release_id = release_id
        self._canary = canary
        self._heldout = heldout

    @property
    def schema_version(self) -> int:
        return V4_SCHEMA_VERSION

    @property
    def release_id(self) -> str:
        return self._release_id

    @property
    def release_authorized(self) -> bool:
        return True

    @property
    def policy_generation(self) -> int:
        return self._canary.policy_generation

    @property
    def policy_checkpoint_sha256(self) -> str:
        return self._canary.policy_checkpoint_sha256

    @property
    def canary_evidence(self) -> BallDomainEvidence:
        return self._canary

    @property
    def heldout_evidence(self) -> BallDomainEvidence:
        return self._heldout

    @property
    def canary_window_sha256(self) -> str:
        return self._canary.window_sha256

    @property
    def heldout_window_sha256(self) -> str:
        return self._heldout.window_sha256


class FrozenEvaluatorV4Authority:
    """Same-process, fixed-window frozen evaluator.

    The class deliberately has no API accepting checkpoint SHA, generation,
    seed/sample/birth range, terminal outcome, or ``in_new_band``.  Those
    values are either allocated here, derived from raw checkpoint bytes, or
    obtained from the code-pinned in-process attempt source.
    """

    def __init__(
        self,
        *,
        launch_receipt: Mapping[str, object],
        attempt_source: object,
        _formal_sentinel: object,
    ) -> None:
        if _formal_sentinel is not _V4_FORMAL_SENTINEL:
            raise FrozenEvaluationAuthorityError(
                "V4 authority requires a trusted launch finalizer"
            )
        self._launch = dict(launch_receipt)
        self._launch_sha256 = _canonical_sha256(self._launch)
        self._profile_order = self._parse_profile_order(
            self._launch["profile_order"]
        )
        self._profile_set = frozenset(self._profile_order)
        self._source = attempt_source
        self._validate_source_binding()
        self._source_state_owner_sha256 = _sha256(
            getattr(self._source, "state_owner_sha256", None),
            name="attempt_source.state_owner_sha256",
        )
        self._state_owner_sha256 = _canonical_sha256(
            {
                "launch_receipt_sha256": self._launch_sha256,
                "source_state_owner_sha256": (
                    self._source_state_owner_sha256
                ),
            }
        )
        self._lifetime = object()
        self._policy_generation = 0
        self._seq = 0
        self._seed_cursor = 0
        self._sample_cursor = 0
        self._birth_cursor = 0
        self._snapshots: Dict[int, Tuple[str, bytes]] = {}
        self._windows: Dict[str, _V4Window] = {}
        self._pending_releases: Dict[
            str,
            Tuple[str, str],
        ] = {}
        self._consumed_releases: set[str] = set()
        self._used_window_capabilities: set[str] = set()
        self._consumed_scheduler_capabilities: set[str] = set()

    @classmethod
    def from_trusted_launch_receipt(
        cls,
        receipt: object,
        *,
        attempt_source: object,
    ) -> "FrozenEvaluatorV4Authority":
        row = _exact_keys(
            receipt,
            (
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
                "attempt_source_contract_sha256",
                "attempt_source_path",
                "attempt_source_sha256",
                "window_contract",
            ),
            name="V4 frozen evaluator launch receipt",
        )
        expected_window_contract = {
            "optional_stopping": False,
            "scheduler_proposals": V4_SCHEDULER_PROPOSALS,
            "canary_proposals": V4_CANARY_PROPOSALS,
            "canary_safe_closed_min": V4_CANARY_SAFE_CLOSED_MIN,
            "heldout_proposals": V4_HELDOUT_PROPOSALS,
            "heldout_safe_closed_min": V4_HELDOUT_SAFE_CLOSED_MIN,
            "sampling_mixture": dict(V4_SAMPLING_MIXTURE),
        }
        if (
            row["schema_version"] != V4_SCHEMA_VERSION
            or row["kind"] != "action_ball_frozen_evaluator_v4_launch"
            or row["authority_contract_sha256"]
            != FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
            or row["arm_catalog_sha256"] != ARM_CATALOG_SHA256
            or row["window_contract"] != expected_window_contract
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 frozen evaluator launch receipt contract mismatch"
            )
        receipt_sha256 = _canonical_sha256(row)
        if (
            receipt_sha256
            not in TRUSTED_FROZEN_EVALUATOR_V4_LAUNCH_RECEIPT_SHA256
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 frozen evaluator launch receipt is not code-pinned"
            )
        return cls(
            launch_receipt=row,
            attempt_source=attempt_source,
            _formal_sentinel=_V4_FORMAL_SENTINEL,
        )

    @staticmethod
    def _parse_profile_order(
        value: object,
    ) -> Tuple[ActionProfileKey, ...]:
        if not isinstance(value, list) or not value:
            raise FrozenEvaluationAuthorityError(
                "V4 profile_order must be a non-empty list"
            )
        result = []
        for index, raw in enumerate(value):
            row = _exact_keys(
                raw,
                ("action_uid", "profile_sha256", "mobility"),
                name=f"V4 profile_order[{index}]",
            )
            result.append(ActionProfileKey(**row))
        converted = tuple(result)
        if len(converted) != len(set(converted)):
            raise FrozenEvaluationAuthorityError(
                "V4 profile_order contains duplicates"
            )
        return converted

    @property
    def authority_contract_sha256(self) -> str:
        return FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256

    @property
    def state_owner_sha256(self) -> str:
        return self._state_owner_sha256

    @property
    def release_authorized(self) -> bool:
        return True

    def binding_document(self) -> Dict[str, object]:
        return {
            "schema_version": V4_SCHEMA_VERSION,
            "authority_contract_sha256": (
                FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
            ),
            "launch_receipt_sha256": self._launch_sha256,
            "curriculum_contract_sha256": self._launch[
                "curriculum_contract_sha256"
            ],
            "profile_order": list(self._launch["profile_order"]),
            "arm_catalog_sha256": self._launch[
                "arm_catalog_sha256"
            ],
            "scheduler_contract_sha256": self._launch[
                "scheduler_contract_sha256"
            ],
            "sampler_sha256": self._launch["sampler_sha256"],
            "solver_sha256": self._launch["solver_sha256"],
            "policy_contract_sha256": self._launch[
                "policy_contract_sha256"
            ],
            "attempt_source_contract_sha256": self._launch[
                "attempt_source_contract_sha256"
            ],
            "attempt_source_path": self._launch[
                "attempt_source_path"
            ],
            "attempt_source_sha256": self._launch[
                "attempt_source_sha256"
            ],
            "source_state_owner_sha256": (
                self._source_state_owner_sha256
            ),
            "state_owner_sha256": self._state_owner_sha256,
        }

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
            "curriculum_contract_sha256": (
                curriculum_contract_sha256
            ),
            "profile_order": [
                key.as_dict() for key in profile_order
            ],
            "arm_catalog_sha256": arm_catalog_sha256,
            "scheduler_contract_sha256": (
                scheduler_contract_sha256
            ),
            "sampler_sha256": sampler_sha256,
            "solver_sha256": solver_sha256,
            "policy_contract_sha256": policy_contract_sha256,
        }
        actual = {
            name: self._launch[name] for name in expected
        }
        if actual != expected:
            raise FrozenEvaluationAuthorityError(
                "V4 frozen evaluator authority binding mismatch"
            )

    def _validate_source_binding(self) -> None:
        expected = {
            "source_contract_sha256": self._launch[
                "attempt_source_contract_sha256"
            ],
            "source_code_sha256": self._launch[
                "attempt_source_sha256"
            ],
            "source_path": self._launch["attempt_source_path"],
        }
        for field, wanted in expected.items():
            if getattr(self._source, field, None) != wanted:
                raise FrozenEvaluationAuthorityError(
                    f"attempt source {field} differs from trusted launch"
                )
        for method in (
            "state_dict",
            "load_state_dict",
            "issue_proposal",
            "assert_exact_proposal",
            "solver_event",
            "assert_solver_event",
            "lifecycle_event",
            "assert_lifecycle_event",
            "terminal_event",
            "assert_terminal_event",
        ):
            if not callable(getattr(self._source, method, None)):
                raise FrozenEvaluationAuthorityError(
                    f"attempt source lacks {method}()"
                )
        self._source_state()

    def _source_state(self) -> object:
        try:
            encoded = json.dumps(
                self._source.state_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            return json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise FrozenEvaluationAuthorityError(
                "attempt source state must be finite JSON data"
            ) from exc

    def _pure_source_call(self, method: str, *args: object) -> object:
        fingerprint = getattr(
            self._source,
            "state_fingerprint",
            None,
        )
        before = (
            fingerprint()
            if callable(fingerprint)
            else self._source_state()
        )
        result = getattr(self._source, method)(*args)
        after = (
            fingerprint()
            if callable(fingerprint)
            else self._source_state()
        )
        if after != before:
            if not callable(fingerprint):
                self._source.load_state_dict(before)
            raise FrozenEvaluationAuthorityError(
                f"attempt source {method}() mutated authority state"
            )
        return result

    def _pure_source_assert(self, method: str, *args: object) -> None:
        self._pure_source_call(method, *args)

    def freeze_checkpoint(
        self,
        checkpoint_bytes: object,
        *,
        policy_generation: Optional[int] = None,
    ) -> FrozenPolicySnapshotV4:
        if not isinstance(checkpoint_bytes, (bytes, bytearray, memoryview)):
            raise FrozenEvaluationAuthorityError(
                "freeze_checkpoint requires raw checkpoint bytes"
            )
        raw = bytes(checkpoint_bytes)
        if not raw:
            raise FrozenEvaluationAuthorityError(
                "checkpoint bytes must not be empty"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if policy_generation is None:
            generation = max(self._snapshots, default=0) + 1
        else:
            generation = _plain_int(
                policy_generation,
                name="policy generation",
            )
            if (
                self._snapshots
                and generation <= max(self._snapshots)
            ):
                raise FrozenEvaluationAuthorityError(
                    "policy generation must increase monotonically"
                )
        self._policy_generation = generation
        self._snapshots[generation] = (digest, raw)
        return FrozenPolicySnapshotV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            digest,
            generation,
        )

    def _snapshot(
        self,
        snapshot: object,
    ) -> FrozenPolicySnapshotV4:
        if type(snapshot) is not FrozenPolicySnapshotV4:
            raise FrozenEvaluationAuthorityError(
                "V4 window requires an opaque policy snapshot"
            )
        if (
            snapshot._authority is not self
            or snapshot._lifetime is not self._lifetime
        ):
            raise FrozenEvaluationAuthorityError(
                "policy snapshot belongs to another or restored authority"
            )
        expected = self._snapshots.get(snapshot.generation)
        if (
            expected is None
            or expected[0] != snapshot.checkpoint_sha256
            or hashlib.sha256(expected[1]).hexdigest() != expected[0]
        ):
            raise FrozenEvaluationAuthorityError(
                "policy snapshot bytes or identity drifted"
            )
        return snapshot

    def open_window(
        self,
        *,
        snapshot: FrozenPolicySnapshotV4,
        key: ActionProfileKey,
        evidence_role: str,
        domain_epoch: int,
        stratum: str,
        selected_arm_key: str,
        selection_round: int,
        arm_levels: Tuple[float, ...],
        rho: float,
    ) -> FrozenEvaluationSessionV4:
        frozen = self._snapshot(snapshot)
        if key not in self._profile_set:
            raise FrozenEvaluationAuthorityError(
                "V4 window key is outside the frozen profile order"
            )
        if evidence_role == "scheduler":
            count = V4_SCHEDULER_PROPOSALS
        elif evidence_role == "frozen_canary":
            count = V4_CANARY_PROPOSALS
        elif evidence_role == "frozen_heldout":
            count = V4_HELDOUT_PROPOSALS
        else:
            raise FrozenEvaluationAuthorityError(
                "V4 supports scheduler/frozen_canary/frozen_heldout windows"
            )
        epoch = _plain_int(domain_epoch, name="domain_epoch")
        round_index = _plain_int(
            selection_round,
            name="selection_round",
        )
        if type(stratum) is not str or not stratum:
            raise FrozenEvaluationAuthorityError(
                "stratum must be non-empty text"
            )
        if type(selected_arm_key) is not str:
            raise FrozenEvaluationAuthorityError(
                "selected_arm_key must be text"
            )
        runtime = _runtime_v4()
        if (
            selected_arm_key
            and selected_arm_key not in runtime.ARM_KEYS
        ):
            raise FrozenEvaluationAuthorityError(
                "selected_arm_key is outside the signed-arm catalog"
            )
        if (
            not isinstance(arm_levels, tuple)
            or len(arm_levels) != len(runtime.ARM_KEYS)
        ):
            raise FrozenEvaluationAuthorityError(
                "arm_levels has the wrong signed-arm shape"
            )
        normalized_levels = tuple(
            float(value) for value in arm_levels
        )
        if any(
            type(value) not in (int, float)
            or not 0.0 <= float(value) <= 1.0
            for value in arm_levels
        ):
            raise FrozenEvaluationAuthorityError(
                "arm_levels must be finite values in [0,1]"
            )
        rho_value = float(rho)
        if (
            type(rho) not in (int, float)
            or not 0.0 <= rho_value <= 1.0
        ):
            raise FrozenEvaluationAuthorityError(
                "rho must be in [0,1]"
            )
        levels = runtime.ActionDomainLevels(
            **dict(zip(runtime.ARM_KEYS, normalized_levels))
        )
        self._seq += 1
        seq = self._seq
        seed_start = self._seed_cursor
        sample_start = self._sample_cursor
        birth_start = self._birth_cursor
        allocation = {
            "schema_version": V4_SCHEMA_VERSION,
            "kind": "frozen_window_allocation",
            "state_owner_sha256": self._state_owner_sha256,
            "seq": seq,
            "key": key.as_dict(),
            "evidence_role": evidence_role,
            "domain_epoch": epoch,
            "stratum": stratum,
            "selected_arm_key": selected_arm_key,
            "selection_round": round_index,
            "arm_levels": list(normalized_levels),
            "rho": rho_value,
            "policy_checkpoint_sha256": frozen.checkpoint_sha256,
            "policy_generation": frozen.generation,
            "seed_start": seed_start,
            "sample_start": sample_start,
            "birth_start": birth_start,
            "proposal_count": count,
            "optional_stopping": False,
        }
        allocation_sha256 = _canonical_sha256(allocation)
        attempts = []
        for offset in range(count):
            request = runtime.FrozenEvaluationProposalRequest.create(
                policy_checkpoint_sha256=frozen.checkpoint_sha256,
                policy_generation=frozen.generation,
                window_sha256=allocation_sha256,
                evidence_role=evidence_role,
                proposal_offset=offset,
                seed=seed_start + offset,
                sample_index=sample_start + offset,
                birth_index=birth_start + offset,
                action_uid=key.action_uid,
                profile_sha256=key.profile_sha256,
                mobility_mode=key.mobility,
                domain_epoch=epoch,
                domain_levels=levels,
                selected_arm_key=selected_arm_key,
            )
            attempts.append(_V4Attempt(request=request))
        self._seed_cursor += count
        self._sample_cursor += count
        self._birth_cursor += count
        window = _V4Window(
            allocation_sha256=allocation_sha256,
            snapshot=frozen,
            key=key,
            evidence_role=evidence_role,
            domain_epoch=epoch,
            stratum=stratum,
            selected_arm_key=selected_arm_key,
            selection_round=round_index,
            arm_levels=normalized_levels,
            rho=rho_value,
            seq=seq,
            seed_start=seed_start,
            sample_start=sample_start,
            birth_start=birth_start,
            attempts=attempts,
        )
        self._windows[allocation_sha256] = window
        return FrozenEvaluationSessionV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            allocation_sha256,
        )

    def _window(self, session: object) -> _V4Window:
        if type(session) is not FrozenEvaluationSessionV4:
            raise FrozenEvaluationAuthorityError(
                "V4 operation requires an opaque evaluation session"
            )
        if (
            session._authority is not self
            or session._lifetime is not self._lifetime
        ):
            raise FrozenEvaluationAuthorityError(
                "evaluation session belongs to another or restored authority"
            )
        try:
            return self._windows[session._allocation_sha256]
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "evaluation session is unknown"
            ) from exc

    def issue_next(
        self,
        session: FrozenEvaluationSessionV4,
    ) -> FrozenProposalHandleV4:
        window = self._window(session)
        if window.finalized:
            raise FrozenEvaluationAuthorityError(
                "cannot issue from a finalized window"
            )
        offset = next(
            (
                index
                for index, attempt in enumerate(window.attempts)
                if attempt.status == "reserved"
            ),
            None,
        )
        if offset is None:
            raise FrozenEvaluationAuthorityError(
                "window has no unissued proposal reservation"
            )
        attempt = window.attempts[offset]
        proposal = self._source.issue_proposal(attempt.request)
        runtime = _runtime_v4()
        if not isinstance(proposal, runtime.FrozenIssuedProposal):
            raise FrozenEvaluationAuthorityError(
                "attempt source returned a non-canonical proposal receipt"
            )
        proposal.assert_request(attempt.request)
        if (
            proposal.source_contract_sha256
            != self._launch["attempt_source_contract_sha256"]
        ):
            raise FrozenEvaluationAuthorityError(
                "proposal source contract differs from launch"
            )
        self._pure_source_assert(
            "assert_exact_proposal",
            attempt.request,
            proposal,
        )
        attempt.proposal = proposal
        attempt.in_new_band = bool(
            window.selected_arm_key
            and proposal.sampling_stratum == "frontier"
            and proposal.frontier_arm == window.selected_arm_key
        )
        attempt.status = "proposed"
        return FrozenProposalHandleV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            window.allocation_sha256,
            offset,
        )

    def _attempt(
        self,
        handle: object,
    ) -> Tuple[_V4Window, _V4Attempt]:
        if type(handle) is not FrozenProposalHandleV4:
            raise FrozenEvaluationAuthorityError(
                "V4 attempt operation requires an opaque proposal handle"
            )
        if (
            handle._authority is not self
            or handle._lifetime is not self._lifetime
        ):
            raise FrozenEvaluationAuthorityError(
                "proposal handle belongs to another or restored authority"
            )
        try:
            window = self._windows[handle._allocation_sha256]
            attempt = window.attempts[handle._offset]
        except (KeyError, IndexError) as exc:
            raise FrozenEvaluationAuthorityError(
                "proposal handle is unknown"
            ) from exc
        return window, attempt

    def capture_solver(
        self,
        handle: FrozenProposalHandleV4,
    ) -> str:
        _, attempt = self._attempt(handle)
        if attempt.status != "proposed" or attempt.proposal is None:
            raise FrozenEvaluationAuthorityError(
                "solver event requires exactly one proposed attempt"
            )
        event = self._source.solver_event(
            attempt.request,
            attempt.proposal,
        )
        runtime = _runtime_v4()
        if not isinstance(event, runtime.FrozenSolverEvent):
            raise FrozenEvaluationAuthorityError(
                "attempt source returned a non-canonical solver event"
            )
        if (
            event.proposal_receipt_sha256
            != attempt.proposal.source_receipt_sha256
            or event.source_contract_sha256
            != self._launch["attempt_source_contract_sha256"]
        ):
            raise FrozenEvaluationAuthorityError(
                "solver event does not bind the exact proposal/source"
            )
        self._pure_source_assert(
            "assert_solver_event",
            attempt.request,
            attempt.proposal,
            event,
        )
        attempt.solver = event
        if event.disposition == "rejected":
            attempt.status = "solver_rejected"
        else:
            attempt.status = "solver_admitted"
        return event.disposition

    def capture_install(
        self,
        handle: FrozenProposalHandleV4,
    ) -> None:
        _, attempt = self._attempt(handle)
        if (
            attempt.status != "solver_admitted"
            or attempt.proposal is None
            or attempt.solver is None
        ):
            raise FrozenEvaluationAuthorityError(
                "install requires one admitted solver event"
            )
        event = self._source.lifecycle_event(
            attempt.request,
            attempt.proposal,
            attempt.solver,
            "installed",
        )
        runtime = _runtime_v4()
        if not isinstance(event, runtime.FrozenLifecycleEvent):
            raise FrozenEvaluationAuthorityError(
                "attempt source returned a non-canonical install event"
            )
        self._assert_lifecycle_binding(attempt, event, "installed")
        attempt.install = event
        attempt.status = "installed"

    def capture_start(
        self,
        handle: FrozenProposalHandleV4,
    ) -> None:
        _, attempt = self._attempt(handle)
        if (
            attempt.status != "installed"
            or attempt.proposal is None
            or attempt.solver is None
        ):
            raise FrozenEvaluationAuthorityError(
                "start requires one installed admitted task"
            )
        event = self._source.lifecycle_event(
            attempt.request,
            attempt.proposal,
            attempt.solver,
            "started",
        )
        runtime = _runtime_v4()
        if not isinstance(event, runtime.FrozenLifecycleEvent):
            raise FrozenEvaluationAuthorityError(
                "attempt source returned a non-canonical start event"
            )
        self._assert_lifecycle_binding(attempt, event, "started")
        attempt.start = event
        attempt.status = "started"

    def _assert_lifecycle_binding(
        self,
        attempt: _V4Attempt,
        event: object,
        stage: str,
    ) -> None:
        if (
            event.stage != stage
            or event.proposal_receipt_sha256
            != attempt.proposal.source_receipt_sha256
            or event.task_receipt_sha256
            != attempt.solver.task_receipt_sha256
            or event.source_contract_sha256
            != self._launch["attempt_source_contract_sha256"]
        ):
            raise FrozenEvaluationAuthorityError(
                f"{stage} event does not bind the exact admitted proposal"
            )
        self._pure_source_assert(
            "assert_lifecycle_event",
            attempt.request,
            attempt.proposal,
            attempt.solver,
            event,
        )

    def capture_terminal(
        self,
        handle: FrozenProposalHandleV4,
    ) -> str | None:
        _, attempt = self._attempt(handle)
        if (
            attempt.status != "started"
            or attempt.proposal is None
            or attempt.solver is None
        ):
            raise FrozenEvaluationAuthorityError(
                "terminal event requires one started admitted task"
            )
        event = self._source.terminal_event(
            attempt.request,
            attempt.proposal,
            attempt.solver,
        )
        runtime = _runtime_v4()
        if not isinstance(event, runtime.FrozenTerminalEvent):
            raise FrozenEvaluationAuthorityError(
                "attempt source returned a non-canonical terminal event"
            )
        if (
            event.proposal_receipt_sha256
            != attempt.proposal.source_receipt_sha256
            or event.task_receipt_sha256
            != attempt.solver.task_receipt_sha256
            or event.source_contract_sha256
            != self._launch["attempt_source_contract_sha256"]
        ):
            raise FrozenEvaluationAuthorityError(
                "terminal event does not bind the exact started task"
            )
        self._pure_source_assert(
            "assert_terminal_event",
            attempt.request,
            attempt.proposal,
            attempt.solver,
            event,
        )
        # Classification is code-rooted in action_ball_runtime.  No caller
        # value is accepted or compared.
        outcome = runtime.classify_frozen_terminal(event.signals)
        if outcome != event.terminal_outcome:
            raise FrozenEvaluationAuthorityError(
                "terminal event classifier drift"
            )
        attempt.terminal = event
        attempt.status = (
            "burned_x"
            if event.signals.infrastructure_invalid
            else "closed"
        )
        return outcome

    def capture_infrastructure_invalid(
        self,
        handle: FrozenProposalHandleV4,
    ) -> None:
        """Close an admitted attempt that failed before a normal terminal."""

        _, attempt = self._attempt(handle)
        if (
            attempt.status
            not in ("solver_admitted", "installed", "started")
            or attempt.proposal is None
            or attempt.solver is None
        ):
            raise FrozenEvaluationAuthorityError(
                "infrastructure burn requires one admitted live attempt"
            )
        event = self._source.terminal_event(
            attempt.request,
            attempt.proposal,
            attempt.solver,
        )
        runtime = _runtime_v4()
        if not isinstance(event, runtime.FrozenTerminalEvent):
            raise FrozenEvaluationAuthorityError(
                "attempt source returned a non-canonical terminal event"
            )
        if (
            not event.signals.infrastructure_invalid
            or event.terminal_outcome is not None
            or event.proposal_receipt_sha256
            != attempt.proposal.source_receipt_sha256
            or event.task_receipt_sha256
            != attempt.solver.task_receipt_sha256
            or event.source_contract_sha256
            != self._launch["attempt_source_contract_sha256"]
        ):
            raise FrozenEvaluationAuthorityError(
                "pre-terminal infrastructure burn is not exact"
            )
        self._pure_source_assert(
            "assert_terminal_event",
            attempt.request,
            attempt.proposal,
            attempt.solver,
            event,
        )
        attempt.terminal = event
        attempt.status = "burned_x"

    def replay_window_from_source(
        self,
        session: FrozenEvaluationSessionV4,
    ) -> FrozenEvaluationCapabilityV4:
        """Replay one complete code-pinned sidecar transcript."""

        if not callable(
            getattr(self._source, "next_event_stage", None)
        ):
            raise FrozenEvaluationAuthorityError(
                "attempt source lacks next_event_stage() replay adapter"
            )
        window = self._window(session)
        if window.finalized:
            raise FrozenEvaluationAuthorityError(
                "cannot replay a finalized V4 window"
            )
        while any(
            attempt.status == "reserved"
            for attempt in window.attempts
        ):
            handle = self.issue_next(session)
            disposition = self.capture_solver(handle)
            if disposition == "rejected":
                continue
            _, attempt = self._attempt(handle)
            while not attempt.settled:
                stage = self._pure_source_call(
                    "next_event_stage",
                    attempt.request,
                )
                if stage == "installed":
                    self.capture_install(handle)
                elif stage == "started":
                    self.capture_start(handle)
                elif stage == "terminal":
                    if attempt.status == "started":
                        self.capture_terminal(handle)
                    else:
                        self.capture_infrastructure_invalid(handle)
                elif stage == "settled":
                    break
                else:
                    raise FrozenEvaluationAuthorityError(
                        "attempt source returned an invalid next event stage"
                    )
        return self.finalize_window(session)

    def burn_unfinished_after_crash(
        self,
        session: FrozenEvaluationSessionV4,
    ) -> int:
        """Burn every unresolved reservation as X without reusing its range."""

        window = self._window(session)
        if window.finalized:
            raise FrozenEvaluationAuthorityError(
                "cannot burn a finalized window"
            )
        count = 0
        for attempt in window.attempts:
            if not attempt.settled:
                attempt.status = "burned_x"
                count += 1
        return count

    @staticmethod
    def _ledger(attempts: Sequence[_V4Attempt]) -> BallOutcomeLedger:
        rows = [attempt.row() for attempt in attempts]
        terminal = {
            name: 0
            for name in (
                "legal_return",
                "safe_nonreturn",
                "table_hit",
                "fall",
                "collision",
                "joint_qdes_limit",
                "joint_actual_limit",
            )
        }
        for row in rows:
            outcome = row["terminal_outcome"]
            if outcome in terminal:
                terminal[outcome] += 1
        new_band_rows = [
            row
            for row in rows
            if (
                row["in_new_band"]
                and row["closed"]
                and row["terminal_outcome"]
                in ("legal_return", "safe_nonreturn")
            )
        ]
        kwargs = {
            "P": len(rows),
            "A": sum(bool(row["solver_admitted"]) for row in rows),
            "I": sum(bool(row["installed"]) for row in rows),
            "S": sum(bool(row["started"]) for row in rows),
            "C": sum(bool(row["closed"]) for row in rows),
            "L": terminal["legal_return"],
            "F": terminal["safe_nonreturn"],
            "U_table": sum(
                bool(
                    row["terminal_signals"] is not None
                    and row["terminal_signals"]["table_hit"]
                )
                for row in rows
            ),
            "U_fall": sum(
                bool(
                    row["terminal_signals"] is not None
                    and row["terminal_signals"]["fall"]
                )
                for row in rows
            ),
            "U_collision": sum(
                bool(
                    row["terminal_signals"] is not None
                    and row["terminal_signals"]["collision"]
                )
                for row in rows
            ),
            "X": sum(
                bool(row["infrastructure_invalid"]) for row in rows
            ),
            "NB": len(new_band_rows),
            "NB_F": sum(
                row["terminal_outcome"] == "safe_nonreturn"
                for row in new_band_rows
            ),
        }
        fields = getattr(BallOutcomeLedger, "__dataclass_fields__", {})
        if "U_joint_qdes" in fields:
            kwargs["U_joint_qdes"] = sum(
                bool(
                    row["terminal_signals"] is not None
                    and row["terminal_signals"]["joint_qdes_limit"]
                )
                for row in rows
            )
            kwargs["U_joint_actual"] = sum(
                bool(
                    row["terminal_signals"] is not None
                    and row["terminal_signals"]["joint_actual_limit"]
                )
                for row in rows
            )
        elif (
            terminal["joint_qdes_limit"]
            or terminal["joint_actual_limit"]
        ):
            raise FrozenEvaluationAuthorityError(
                "curriculum ledger schema lacks joint-limit terminals"
            )
        return BallOutcomeLedger(**kwargs)

    @staticmethod
    def _assert_sampling_mixture(window: _V4Window) -> None:
        expected = {
            "scheduler": V4_SCHEDULER_PROPOSALS,
            "frozen_canary": V4_CANARY_PROPOSALS,
            "frozen_heldout": V4_HELDOUT_PROPOSALS,
        }[window.evidence_role]
        expected_mixture = {
            stratum: int(expected * fraction)
            for stratum, fraction in V4_SAMPLING_MIXTURE.items()
        }
        if sum(expected_mixture.values()) != expected:
            raise FrozenEvaluationAuthorityError(
                "V4 sampling mixture does not partition the fixed window"
            )
        observed_mixture = {
            stratum: sum(
                attempt.proposal is not None
                and attempt.proposal.sampling_stratum == stratum
                for attempt in window.attempts
            )
            for stratum in expected_mixture
        }
        if observed_mixture != expected_mixture:
            raise FrozenEvaluationAuthorityError(
                "V4 window sampling mixture must be exact 20/60/20"
            )
        if window.selected_arm_key and any(
            attempt.proposal is not None
            and attempt.proposal.sampling_stratum == "frontier"
            and attempt.proposal.frontier_arm
            != window.selected_arm_key
            for attempt in window.attempts
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 frontier proposal differs from the selected "
                "action-axis-side arm"
            )

    def finalize_window(
        self,
        session: FrozenEvaluationSessionV4,
    ) -> FrozenEvaluationCapabilityV4:
        window = self._window(session)
        if window.finalized:
            raise FrozenEvaluationAuthorityError(
                "V4 window was already finalized"
            )
        if not all(attempt.settled for attempt in window.attempts):
            raise FrozenEvaluationAuthorityError(
                "optional stopping is forbidden; every fixed proposal "
                "reservation must settle or burn"
            )
        ledger = self._ledger(window.attempts)
        expected = {
            "scheduler": V4_SCHEDULER_PROPOSALS,
            "frozen_canary": V4_CANARY_PROPOSALS,
            "frozen_heldout": V4_HELDOUT_PROPOSALS,
        }[window.evidence_role]
        floor = {
            "scheduler": 0,
            "frozen_canary": V4_CANARY_SAFE_CLOSED_MIN,
            "frozen_heldout": V4_HELDOUT_SAFE_CLOSED_MIN,
        }[window.evidence_role]
        if ledger.P != expected:
            raise FrozenEvaluationAuthorityError(
                "V4 fixed proposal count drifted"
            )
        if ledger.safe_closed < floor:
            raise FrozenEvaluationAuthorityError(
                f"{window.evidence_role} safe-closed floor is {floor}; "
                f"observed {ledger.safe_closed}"
            )
        self._assert_sampling_mixture(window)
        sample_rows = tuple(
            attempt.sample_receipt_sha256()
            for attempt in window.attempts
        )
        birth_rows = tuple(
            attempt.birth_receipt_sha256()
            for attempt in window.attempts
        )
        if len(set(sample_rows)) != expected:
            raise FrozenEvaluationAuthorityError(
                "V4 window reused a sample receipt"
            )
        if len(set(birth_rows)) != expected:
            raise FrozenEvaluationAuthorityError(
                "V4 window requires one unique birth per proposal"
            )
        admitted_task_rows = tuple(
            attempt.solver.task_receipt_sha256
            for attempt in window.attempts
            if (
                attempt.solver is not None
                and attempt.solver.disposition == "admitted"
            )
        )
        if len(admitted_task_rows) != len(set(admitted_task_rows)):
            raise FrozenEvaluationAuthorityError(
                "V4 window reused one admitted task receipt"
            )
        evidence = BallDomainEvidence.create(
            key=window.key,
            arm_catalog_sha256=self._launch["arm_catalog_sha256"],
            scheduler_contract_sha256=self._launch[
                "scheduler_contract_sha256"
            ],
            sampler_sha256=self._launch["sampler_sha256"],
            solver_sha256=self._launch["solver_sha256"],
            policy_contract_sha256=self._launch[
                "policy_contract_sha256"
            ],
            policy_checkpoint_sha256=(
                window.snapshot.checkpoint_sha256
            ),
            policy_generation=window.snapshot.generation,
            evidence_role=window.evidence_role,
            domain_epoch=window.domain_epoch,
            stratum=window.stratum,
            selected_arm_key=window.selected_arm_key,
            selection_round=window.selection_round,
            arm_levels=window.arm_levels,
            rho=window.rho,
            seed_block_start=window.seed_start,
            seed_block_end_exclusive=(
                window.seed_start + expected
            ),
            sample_id_start=window.sample_start,
            sample_id_end_exclusive=(
                window.sample_start + expected
            ),
            sample_receipt_root_sha256=(
                ordered_sample_receipt_root(sample_rows)
            ),
            unique_birth_count=expected,
            birth_receipt_root_sha256=(
                ordered_birth_receipt_root(birth_rows)
            ),
            seq=window.seq,
            window_id=window.allocation_sha256,
            ledger=ledger,
        )
        capability_id = _canonical_sha256(
            {
                "schema_version": V4_SCHEMA_VERSION,
                "state_owner_sha256": self._state_owner_sha256,
                "allocation_sha256": window.allocation_sha256,
                "window_sha256": evidence.window_sha256,
                "attempt_transcript_sha256": _canonical_sha256(
                    [attempt.row() for attempt in window.attempts]
                ),
            }
        )
        window.evidence = evidence
        window.capability_id = capability_id
        window.finalized = True
        return FrozenEvaluationCapabilityV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            capability_id,
            evidence,
        )

    def attempt_rows(
        self,
        session: FrozenEvaluationSessionV4,
    ) -> Tuple[Dict[str, object], ...]:
        window = self._window(session)
        return tuple(attempt.row() for attempt in window.attempts)

    def _capability_window(
        self,
        capability: object,
    ) -> _V4Window:
        if type(capability) is not FrozenEvaluationCapabilityV4:
            raise FrozenEvaluationAuthorityError(
                "V4 release requires opaque schema-4 window capabilities"
            )
        if (
            capability._authority is not self
            or capability._lifetime is not self._lifetime
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 window capability belongs to another authority"
            )
        matches = [
            window
            for window in self._windows.values()
            if window.capability_id == capability.capability_id
        ]
        if (
            len(matches) != 1
            or matches[0].evidence != capability.evidence
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 window capability is stale or inconsistent"
            )
        return matches[0]

    def assert_scheduler_capabilities_many(
        self,
        capabilities: Mapping[
            ActionProfileKey,
            FrozenEvaluationCapabilityV4,
        ],
    ) -> Dict[
        ActionProfileKey,
        Tuple[
            BallDomainEvidence,
            Tuple[Dict[str, object], ...],
        ],
    ]:
        """Validate exact scheduler transcripts without consuming them."""

        if not isinstance(capabilities, Mapping) or not capabilities:
            raise FrozenEvaluationAuthorityError(
                "V4 scheduler capabilities must be a non-empty mapping"
            )
        result = {}
        capability_ids = set()
        for key, capability in capabilities.items():
            window = self._capability_window(capability)
            if (
                window.evidence_role != "scheduler"
                or capability.evidence.evidence_role != "scheduler"
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 scheduler ingest rejects canary/heldout capability"
                )
            if capability.evidence.key != key:
                raise FrozenEvaluationAuthorityError(
                    "V4 scheduler capability key mismatch"
                )
            if (
                capability.capability_id
                in self._consumed_scheduler_capabilities
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 scheduler capability is stale or consumed"
                )
            if capability.capability_id in capability_ids:
                raise FrozenEvaluationAuthorityError(
                    "V4 scheduler batch reuses one capability"
                )
            capability_ids.add(capability.capability_id)
            result[key] = (
                capability.evidence,
                tuple(attempt.row() for attempt in window.attempts),
            )
        return result

    def consume_scheduler_capabilities_many(
        self,
        capabilities: Mapping[
            ActionProfileKey,
            FrozenEvaluationCapabilityV4,
        ],
    ) -> Dict[
        ActionProfileKey,
        Tuple[
            BallDomainEvidence,
            Tuple[Dict[str, object], ...],
        ],
    ]:
        """Atomically consume one exact scheduler transcript per key."""

        evidence = self.assert_scheduler_capabilities_many(
            capabilities
        )
        proposed = set(self._consumed_scheduler_capabilities)
        proposed.update(
            capability.capability_id
            for capability in capabilities.values()
        )
        self._consumed_scheduler_capabilities = proposed
        return evidence

    def issue_release(
        self,
        *,
        canary: FrozenEvaluationCapabilityV4,
        heldout: FrozenEvaluationCapabilityV4,
    ) -> FrozenEvaluationReleaseReceipt:
        canary_window = self._capability_window(canary)
        heldout_window = self._capability_window(heldout)
        if (
            canary_window.evidence_role != "frozen_canary"
            or heldout_window.evidence_role != "frozen_heldout"
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 release requires canary then heldout roles"
            )
        if (
            canary.capability_id in self._used_window_capabilities
            or heldout.capability_id in self._used_window_capabilities
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 window capability was already paired"
            )
        same = (
            canary.evidence.key == heldout.evidence.key
            and canary.evidence.policy_checkpoint_sha256
            == heldout.evidence.policy_checkpoint_sha256
            and canary.evidence.policy_generation
            == heldout.evidence.policy_generation
            and canary.evidence.domain_epoch
            == heldout.evidence.domain_epoch
            and canary.evidence.stratum == heldout.evidence.stratum
            and canary.evidence.selected_arm_key
            == heldout.evidence.selected_arm_key
            and canary.evidence.selection_round
            == heldout.evidence.selection_round
            and canary.evidence.arm_levels
            == heldout.evidence.arm_levels
            and canary.evidence.rho == heldout.evidence.rho
        )
        if not same:
            raise FrozenEvaluationAuthorityError(
                "V4 canary/heldout release identity mismatch"
            )
        if _intervals_overlap(
            canary.evidence.seed_block_start,
            canary.evidence.seed_block_end_exclusive,
            heldout.evidence.seed_block_start,
            heldout.evidence.seed_block_end_exclusive,
        ) or _intervals_overlap(
            canary.evidence.sample_id_start,
            canary.evidence.sample_id_end_exclusive,
            heldout.evidence.sample_id_start,
            heldout.evidence.sample_id_end_exclusive,
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 canary/heldout ranges overlap"
            )
        release_id = _canonical_sha256(
            {
                "schema_version": V4_SCHEMA_VERSION,
                "kind": "action_ball_frozen_evaluation_release",
                "state_owner_sha256": self._state_owner_sha256,
                "policy_checkpoint_sha256": (
                    canary.evidence.policy_checkpoint_sha256
                ),
                "policy_generation": (
                    canary.evidence.policy_generation
                ),
                "canary_window_sha256": (
                    canary.evidence.window_sha256
                ),
                "heldout_window_sha256": (
                    heldout.evidence.window_sha256
                ),
                "optional_stopping": False,
            }
        )
        if (
            release_id in self._pending_releases
            or release_id in self._consumed_releases
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 release was already issued"
            )
        self._pending_releases[release_id] = (
            canary.capability_id,
            heldout.capability_id,
        )
        self._used_window_capabilities.update(
            (canary.capability_id, heldout.capability_id)
        )
        return FrozenEvaluationReleaseReceipt(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            release_id,
            canary.evidence,
            heldout.evidence,
        )

    def assert_release_receipt(
        self,
        receipt: object,
    ) -> Tuple[BallDomainEvidence, BallDomainEvidence]:
        if type(receipt) is not FrozenEvaluationReleaseReceipt:
            raise FrozenEvaluationAuthorityError(
                "curriculum release requires an opaque schema-4 receipt"
            )
        if (
            receipt._authority is not self
            or receipt._lifetime is not self._lifetime
            or receipt.release_id not in self._pending_releases
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 release receipt is foreign, stale, or consumed"
            )
        pair = self._pending_releases[receipt.release_id]
        windows = {
            window.capability_id: window
            for window in self._windows.values()
            if window.capability_id
        }
        try:
            canary = windows[pair[0]].evidence
            heldout = windows[pair[1]].evidence
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "V4 release window disappeared"
            ) from exc
        if (
            canary is None
            or heldout is None
            or canary != receipt.canary_evidence
            or heldout != receipt.heldout_evidence
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 release evidence was tampered"
            )
        return canary, heldout

    def consume_release(
        self,
        receipt: FrozenEvaluationReleaseReceipt,
    ) -> Tuple[BallDomainEvidence, BallDomainEvidence]:
        evidence = self.assert_release_receipt(receipt)
        del self._pending_releases[receipt.release_id]
        self._consumed_releases.add(receipt.release_id)
        return evidence

    def assert_release_receipts_many(
        self,
        receipts: Mapping[
            ActionProfileKey,
            FrozenEvaluationReleaseReceipt,
        ],
    ) -> Dict[
        ActionProfileKey,
        Tuple[BallDomainEvidence, BallDomainEvidence],
    ]:
        """Validate an exact unique release batch without mutating state."""

        if not isinstance(receipts, Mapping) or not receipts:
            raise FrozenEvaluationAuthorityError(
                "V4 release batch must be a non-empty mapping"
            )
        result = {}
        release_ids = set()
        for key, receipt in receipts.items():
            if not isinstance(key, ActionProfileKey):
                raise FrozenEvaluationAuthorityError(
                    "V4 release batch key must be ActionProfileKey"
                )
            evidence = self.assert_release_receipt(receipt)
            if evidence[0].key != key or evidence[1].key != key:
                raise FrozenEvaluationAuthorityError(
                    "V4 release batch key/evidence mismatch"
                )
            if receipt.release_id in release_ids:
                raise FrozenEvaluationAuthorityError(
                    "V4 release batch reuses one receipt"
                )
            release_ids.add(receipt.release_id)
            result[key] = evidence
        return result

    def consume_releases_many(
        self,
        receipts: Mapping[
            ActionProfileKey,
            FrozenEvaluationReleaseReceipt,
        ],
    ) -> Dict[
        ActionProfileKey,
        Tuple[BallDomainEvidence, BallDomainEvidence],
    ]:
        """Atomically consume a prevalidated schema-4 release batch."""

        evidence = self.assert_release_receipts_many(receipts)
        release_ids = {
            receipt.release_id for receipt in receipts.values()
        }
        proposed_pending = dict(self._pending_releases)
        for release_id in release_ids:
            del proposed_pending[release_id]
        proposed_consumed = set(self._consumed_releases)
        proposed_consumed.update(release_ids)
        self._pending_releases = proposed_pending
        self._consumed_releases = proposed_consumed
        return evidence

    @staticmethod
    def _attempt_state(attempt: _V4Attempt) -> Dict[str, object]:
        request = {
            "reservation_sha256": attempt.request.reservation_sha256,
            **attempt.request.identity_payload(),
        }
        proposal = None
        if attempt.proposal is not None:
            proposal = {
                "source_receipt_sha256": (
                    attempt.proposal.source_receipt_sha256
                ),
                **attempt.proposal.receipt_payload(),
            }
        solver = None
        if attempt.solver is not None:
            solver = {
                "event_receipt_sha256": (
                    attempt.solver.event_receipt_sha256
                ),
                **attempt.solver.event_payload(),
            }

        def lifecycle(event: object | None) -> object:
            if event is None:
                return None
            return {
                "event_receipt_sha256": event.event_receipt_sha256,
                **event.event_payload(),
            }

        terminal = None
        if attempt.terminal is not None:
            terminal = {
                "event_receipt_sha256": (
                    attempt.terminal.event_receipt_sha256
                ),
                **attempt.terminal.event_payload(),
            }
        return {
            "request": request,
            "proposal": proposal,
            "solver": solver,
            "install": lifecycle(attempt.install),
            "start": lifecycle(attempt.start),
            "terminal": terminal,
            "in_new_band": attempt.in_new_band,
            "status": attempt.status,
        }

    @staticmethod
    def _parse_attempt_state(value: object) -> _V4Attempt:
        runtime = _runtime_v4()
        row = _exact_keys(
            value,
            (
                "request",
                "proposal",
                "solver",
                "install",
                "start",
                "terminal",
                "in_new_band",
                "status",
            ),
            name="V4 attempt state",
        )
        request_row = dict(
            _exact_keys(
                row["request"],
                (
                    "reservation_sha256",
                    "schema_version",
                    "kind",
                    "policy_checkpoint_sha256",
                    "policy_generation",
                    "window_sha256",
                    "evidence_role",
                    "proposal_offset",
                    "seed",
                    "sample_index",
                    "birth_index",
                    "action_uid",
                    "profile_sha256",
                    "mobility_mode",
                    "domain_epoch",
                    "domain_levels",
                    "selected_arm_key",
                ),
                name="V4 attempt request state",
            )
        )
        if (
            request_row.pop("schema_version") != V4_SCHEMA_VERSION
            or request_row.pop("kind")
            != "frozen_evaluation_proposal_reservation"
        ):
            raise FrozenEvaluationAuthorityError(
                "unsupported V4 proposal reservation schema"
            )
        request_row["domain_levels"] = runtime.ActionDomainLevels.from_dict(
            request_row["domain_levels"]
        )
        request = runtime.FrozenEvaluationProposalRequest(
            **request_row
        )

        proposal = None
        if row["proposal"] is not None:
            proposal_row = dict(
                _exact_keys(
                    row["proposal"],
                    (
                        "source_receipt_sha256",
                        "schema_version",
                        "kind",
                        "reservation_sha256",
                        "source_contract_sha256",
                        "sample_receipt_sha256",
                        "birth_receipt_sha256",
                        "action_uid",
                        "profile_sha256",
                        "mobility_mode",
                        "domain_epoch",
                        "levels_sha256",
                        "sample_index",
                        "birth_index",
                        "sampling_stratum",
                        "frontier_arm",
                    ),
                    name="V4 issued proposal state",
                )
            )
            if (
                proposal_row.pop("schema_version")
                != V4_SCHEMA_VERSION
                or proposal_row.pop("kind")
                != "frozen_issued_proposal"
            ):
                raise FrozenEvaluationAuthorityError(
                    "unsupported V4 issued proposal schema"
                )
            proposal = runtime.FrozenIssuedProposal(**proposal_row)
            proposal.assert_request(request)

        solver = None
        if row["solver"] is not None:
            solver_row = dict(
                _exact_keys(
                    row["solver"],
                    (
                        "event_receipt_sha256",
                        "schema_version",
                        "kind",
                        "proposal_receipt_sha256",
                        "source_contract_sha256",
                        "disposition",
                        "reject_reason",
                        "task_receipt_sha256",
                    ),
                    name="V4 solver event state",
                )
            )
            if (
                solver_row.pop("schema_version") != V4_SCHEMA_VERSION
                or solver_row.pop("kind") != "frozen_solver_event"
            ):
                raise FrozenEvaluationAuthorityError(
                    "unsupported V4 solver event schema"
                )
            solver = runtime.FrozenSolverEvent(**solver_row)

        def parse_lifecycle(
            raw: object,
            *,
            name: str,
        ) -> object | None:
            if raw is None:
                return None
            event_row = dict(
                _exact_keys(
                    raw,
                    (
                        "event_receipt_sha256",
                        "schema_version",
                        "kind",
                        "proposal_receipt_sha256",
                        "task_receipt_sha256",
                        "source_contract_sha256",
                        "stage",
                    ),
                    name=name,
                )
            )
            if (
                event_row.pop("schema_version")
                != V4_SCHEMA_VERSION
                or event_row.pop("kind")
                != "frozen_lifecycle_event"
            ):
                raise FrozenEvaluationAuthorityError(
                    "unsupported V4 lifecycle event schema"
                )
            return runtime.FrozenLifecycleEvent(**event_row)

        install = parse_lifecycle(
            row["install"],
            name="V4 install event state",
        )
        start = parse_lifecycle(
            row["start"],
            name="V4 start event state",
        )
        terminal = None
        if row["terminal"] is not None:
            terminal_row = dict(
                _exact_keys(
                    row["terminal"],
                    (
                        "event_receipt_sha256",
                        "schema_version",
                        "kind",
                        "proposal_receipt_sha256",
                        "task_receipt_sha256",
                        "source_contract_sha256",
                        "signals",
                    ),
                    name="V4 terminal event state",
                )
            )
            if (
                terminal_row.pop("schema_version")
                != V4_SCHEMA_VERSION
                or terminal_row.pop("kind")
                != "frozen_terminal_event"
            ):
                raise FrozenEvaluationAuthorityError(
                    "unsupported V4 terminal event schema"
                )
            signal_row = _exact_keys(
                terminal_row["signals"],
                (
                    "infrastructure_invalid",
                    "joint_actual_limit",
                    "joint_qdes_limit",
                    "fall",
                    "table_hit",
                    "collision",
                    "legal_return",
                ),
                name="V4 terminal signal state",
            )
            terminal_row["signals"] = runtime.FrozenTerminalSignals(
                **signal_row
            )
            terminal = runtime.FrozenTerminalEvent(**terminal_row)
        if type(row["in_new_band"]) is not bool:
            raise FrozenEvaluationAuthorityError(
                "V4 in_new_band state must be bool"
            )
        status = row["status"]
        if status not in (
            "reserved",
            "proposed",
            "solver_rejected",
            "solver_admitted",
            "installed",
            "started",
            "closed",
            "burned_x",
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 attempt state status is invalid"
            )
        result = _V4Attempt(
            request=request,
            proposal=proposal,
            solver=solver,
            install=install,
            start=start,
            terminal=terminal,
            in_new_band=row["in_new_band"],
            status=status,
        )
        expected_presence = {
            "reserved": (False, False, False, False, False),
            "proposed": (True, False, False, False, False),
            "solver_rejected": (True, True, False, False, False),
            "solver_admitted": (True, True, False, False, False),
            "installed": (True, True, True, False, False),
            "started": (True, True, True, True, False),
            "closed": (True, True, True, True, True),
        }
        if status in expected_presence:
            actual_presence = (
                proposal is not None,
                solver is not None,
                install is not None,
                start is not None,
                terminal is not None,
            )
            if actual_presence != expected_presence[status]:
                raise FrozenEvaluationAuthorityError(
                    "V4 attempt lifecycle fields disagree with status"
                )
        if (
            status == "solver_rejected"
            and solver.disposition != "rejected"
        ) or (
            status in (
                "solver_admitted",
                "installed",
                "started",
                "closed",
            )
            and solver.disposition != "admitted"
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 solver disposition disagrees with lifecycle status"
            )
        if (
            status == "closed"
            and terminal.signals.infrastructure_invalid
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 closed attempt cannot be infrastructure invalid"
            )
        if solver is not None and (
            proposal is None
            or solver.proposal_receipt_sha256
            != proposal.source_receipt_sha256
            or solver.source_contract_sha256
            != proposal.source_contract_sha256
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 solver event chain binding mismatch"
            )
        for event in (install, start, terminal):
            if event is not None and (
                solver is None
                or proposal is None
                or event.proposal_receipt_sha256
                != proposal.source_receipt_sha256
                or event.task_receipt_sha256
                != solver.task_receipt_sha256
                or event.source_contract_sha256
                != proposal.source_contract_sha256
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 lifecycle event chain binding mismatch"
                )
        if install is not None and install.stage != "installed":
            raise FrozenEvaluationAuthorityError(
                "V4 install event has the wrong stage"
            )
        if start is not None and start.stage != "started":
            raise FrozenEvaluationAuthorityError(
                "V4 start event has the wrong stage"
            )
        return result

    @staticmethod
    def _parse_v4_evidence(value: object) -> BallDomainEvidence:
        row = dict(
            _exact_keys(
                value,
                FrozenEvaluatorAuthority._evidence_fields(),
                name="V4 evidence state",
            )
        )
        if row.pop("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise FrozenEvaluationAuthorityError(
                "unsupported V4 evidence schema"
            )
        key_row = _exact_keys(
            row.pop("key"),
            ("action_uid", "profile_sha256", "mobility"),
            name="V4 evidence key",
        )
        ledger_row = row.pop("ledger")
        ledger_fields = tuple(
            BallOutcomeLedger.__dataclass_fields__
        )
        ledger = BallOutcomeLedger(
            **_exact_keys(
                ledger_row,
                ledger_fields,
                name="V4 evidence ledger",
            )
        )
        row["key"] = ActionProfileKey(**key_row)
        row["arm_levels"] = tuple(row["arm_levels"])
        row["ledger"] = ledger
        return BallDomainEvidence.create(**row)

    def state_dict(self) -> Dict[str, object]:
        """Serialize exact V4 state, including burned reservations.

        Raw checkpoint bytes are retained because accepting a caller-supplied
        digest on resume would reintroduce the forged-checkpoint hole.  The
        enclosing training checkpoint may deduplicate/compress this JSON
        field, but it must restore it byte-exactly.
        """

        snapshots = []
        for generation, (digest, raw) in sorted(
            self._snapshots.items()
        ):
            snapshots.append(
                {
                    "generation": generation,
                    "checkpoint_sha256": digest,
                    "checkpoint_bytes_base64": base64.b64encode(
                        raw
                    ).decode("ascii"),
                }
            )
        windows = []
        for window in sorted(
            self._windows.values(),
            key=lambda item: item.seq,
        ):
            windows.append(
                {
                    "allocation_sha256": window.allocation_sha256,
                    "policy_generation": (
                        window.snapshot.generation
                    ),
                    "key": window.key.as_dict(),
                    "evidence_role": window.evidence_role,
                    "domain_epoch": window.domain_epoch,
                    "stratum": window.stratum,
                    "selected_arm_key": window.selected_arm_key,
                    "selection_round": window.selection_round,
                    "arm_levels": list(window.arm_levels),
                    "rho": window.rho,
                    "seq": window.seq,
                    "seed_start": window.seed_start,
                    "sample_start": window.sample_start,
                    "birth_start": window.birth_start,
                    "attempts": [
                        self._attempt_state(attempt)
                        for attempt in window.attempts
                    ],
                    "finalized": window.finalized,
                    "evidence": (
                        None
                        if window.evidence is None
                        else window.evidence._hash_document()
                    ),
                    "capability_id": window.capability_id,
                }
            )
        source_state = self._source_state()
        document = {
            "schema_version": V4_SCHEMA_VERSION,
            "authority_contract_sha256": (
                FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
            ),
            "binding": self.binding_document(),
            "policy_generation": self._policy_generation,
            "seq": self._seq,
            "seed_cursor": self._seed_cursor,
            "sample_cursor": self._sample_cursor,
            "birth_cursor": self._birth_cursor,
            "snapshots": snapshots,
            "windows": windows,
            "pending_releases": [
                [release_id, pair[0], pair[1]]
                for release_id, pair in sorted(
                    self._pending_releases.items()
                )
            ],
            "consumed_releases": sorted(self._consumed_releases),
            "used_window_capabilities": sorted(
                self._used_window_capabilities
            ),
            "consumed_scheduler_capabilities": sorted(
                self._consumed_scheduler_capabilities
            ),
            "source_state": source_state,
            "source_state_sha256": _canonical_sha256(source_state),
        }
        document["state_sha256"] = _canonical_sha256(document)
        return document

    def load_state_dict(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise FrozenEvaluationAuthorityError(
                "V4 authority state must be a mapping"
            )
        if state.get("schema_version") != V4_SCHEMA_VERSION:
            raise FrozenEvaluationAuthorityError(
                "legacy/unsupported evaluator state cannot migrate into V4"
            )
        row = _exact_keys(
            state,
            (
                "schema_version",
                "authority_contract_sha256",
                "binding",
                "policy_generation",
                "seq",
                "seed_cursor",
                "sample_cursor",
                "birth_cursor",
                "snapshots",
                "windows",
                "pending_releases",
                "consumed_releases",
                "used_window_capabilities",
                "consumed_scheduler_capabilities",
                "source_state",
                "source_state_sha256",
                "state_sha256",
            ),
            name="V4 evaluator authority state",
        )
        unsigned = dict(row)
        digest = _sha256(
            unsigned.pop("state_sha256"),
            name="V4 state_sha256",
        )
        if _canonical_sha256(unsigned) != digest:
            raise FrozenEvaluationAuthorityError(
                "V4 evaluator authority state digest mismatch"
            )
        if (
            row["authority_contract_sha256"]
            != FROZEN_EVALUATOR_V4_AUTHORITY_CONTRACT_SHA256
            or row["binding"] != self.binding_document()
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 evaluator authority state binding mismatch"
            )
        source_state = row["source_state"]
        if (
            _canonical_sha256(source_state)
            != row["source_state_sha256"]
        ):
            raise FrozenEvaluationAuthorityError(
                "V4 attempt source state digest mismatch"
            )
        previous_source_state = self._source_state()
        try:
            self._source.load_state_dict(source_state)
            if self._source_state() != source_state:
                raise FrozenEvaluationAuthorityError(
                    "V4 attempt source did not restore exact state"
                )
            snapshots_raw = row["snapshots"]
            if not isinstance(snapshots_raw, list):
                raise FrozenEvaluationAuthorityError(
                    "V4 snapshots state must be a list"
                )
            snapshots: Dict[int, Tuple[str, bytes]] = {}
            for index, raw_snapshot in enumerate(snapshots_raw):
                snapshot_row = _exact_keys(
                    raw_snapshot,
                    (
                        "generation",
                        "checkpoint_sha256",
                        "checkpoint_bytes_base64",
                    ),
                    name=f"V4 snapshot[{index}]",
                )
                generation = _plain_int(
                    snapshot_row["generation"],
                    name=f"V4 snapshot[{index}].generation",
                )
                checkpoint_sha = _sha256(
                    snapshot_row["checkpoint_sha256"],
                    name=f"V4 snapshot[{index}].checkpoint_sha256",
                )
                encoded = snapshot_row[
                    "checkpoint_bytes_base64"
                ]
                if type(encoded) is not str:
                    raise FrozenEvaluationAuthorityError(
                        "V4 checkpoint bytes must be canonical base64"
                    )
                try:
                    checkpoint_bytes = base64.b64decode(
                        encoded.encode("ascii"),
                        validate=True,
                    )
                except (ValueError, UnicodeEncodeError) as exc:
                    raise FrozenEvaluationAuthorityError(
                        "V4 checkpoint bytes are not canonical base64"
                    ) from exc
                if (
                    base64.b64encode(checkpoint_bytes).decode("ascii")
                    != encoded
                    or hashlib.sha256(checkpoint_bytes).hexdigest()
                    != checkpoint_sha
                ):
                    raise FrozenEvaluationAuthorityError(
                        "V4 checkpoint bytes/hash mismatch"
                    )
                if generation in snapshots:
                    raise FrozenEvaluationAuthorityError(
                        "V4 snapshot generation is duplicated"
                    )
                snapshots[generation] = (
                    checkpoint_sha,
                    checkpoint_bytes,
                )
            windows_raw = row["windows"]
            if not isinstance(windows_raw, list):
                raise FrozenEvaluationAuthorityError(
                    "V4 windows state must be a list"
                )
            lifetime = object()
            windows: Dict[str, _V4Window] = {}
            for index, raw_window in enumerate(windows_raw):
                window_row = _exact_keys(
                    raw_window,
                    (
                        "allocation_sha256",
                        "policy_generation",
                        "key",
                        "evidence_role",
                        "domain_epoch",
                        "stratum",
                        "selected_arm_key",
                        "selection_round",
                        "arm_levels",
                        "rho",
                        "seq",
                        "seed_start",
                        "sample_start",
                        "birth_start",
                        "attempts",
                        "finalized",
                        "evidence",
                        "capability_id",
                    ),
                    name=f"V4 window[{index}]",
                )
                generation = _plain_int(
                    window_row["policy_generation"],
                    name=f"V4 window[{index}].policy_generation",
                    minimum=1,
                )
                try:
                    checkpoint = snapshots[generation]
                except KeyError as exc:
                    raise FrozenEvaluationAuthorityError(
                        "V4 window names a missing policy snapshot"
                    ) from exc
                snapshot = FrozenPolicySnapshotV4(
                    _V4_MINT_SENTINEL,
                    self,
                    lifetime,
                    checkpoint[0],
                    generation,
                )
                key_row = _exact_keys(
                    window_row["key"],
                    ("action_uid", "profile_sha256", "mobility"),
                    name=f"V4 window[{index}].key",
                )
                key = ActionProfileKey(**key_row)
                if key not in self._profile_set:
                    raise FrozenEvaluationAuthorityError(
                        "V4 window state has an unknown profile key"
                    )
                attempts_raw = window_row["attempts"]
                if not isinstance(attempts_raw, list):
                    raise FrozenEvaluationAuthorityError(
                        "V4 window attempts must be a list"
                    )
                attempts = [
                    self._parse_attempt_state(raw_attempt)
                    for raw_attempt in attempts_raw
                ]
                allocation_sha = _sha256(
                    window_row["allocation_sha256"],
                    name=f"V4 window[{index}].allocation_sha256",
                )
                if any(
                    attempt.request.window_sha256 != allocation_sha
                    or attempt.request.proposal_offset
                    != attempt_index
                    for attempt_index, attempt in enumerate(attempts)
                ):
                    raise FrozenEvaluationAuthorityError(
                        "V4 window attempt allocation/order mismatch"
                    )
                evidence = (
                    None
                    if window_row["evidence"] is None
                    else self._parse_v4_evidence(
                        window_row["evidence"]
                    )
                )
                finalized = window_row["finalized"]
                if type(finalized) is not bool:
                    raise FrozenEvaluationAuthorityError(
                        "V4 window finalized flag must be bool"
                    )
                capability_id = window_row["capability_id"]
                if finalized:
                    capability_id = _sha256(
                        capability_id,
                        name=(
                            f"V4 window[{index}].capability_id"
                        ),
                    )
                    if evidence is None:
                        raise FrozenEvaluationAuthorityError(
                            "finalized V4 window lacks evidence"
                        )
                elif evidence is not None or capability_id != "":
                    raise FrozenEvaluationAuthorityError(
                        "unfinished V4 window carries final evidence"
                    )
                role = window_row["evidence_role"]
                expected_count = {
                    "scheduler": V4_SCHEDULER_PROPOSALS,
                    "frozen_canary": V4_CANARY_PROPOSALS,
                    "frozen_heldout": V4_HELDOUT_PROPOSALS,
                }.get(role)
                if expected_count is None or len(attempts) != expected_count:
                    raise FrozenEvaluationAuthorityError(
                        "V4 restored window fixed proposal count drifted"
                    )
                domain_epoch = _plain_int(
                    window_row["domain_epoch"],
                    name=f"V4 window[{index}].domain_epoch",
                )
                selection_round = _plain_int(
                    window_row["selection_round"],
                    name=f"V4 window[{index}].selection_round",
                )
                seq_value = _plain_int(
                    window_row["seq"],
                    name=f"V4 window[{index}].seq",
                    minimum=1,
                )
                seed_start = _plain_int(
                    window_row["seed_start"],
                    name=f"V4 window[{index}].seed_start",
                )
                sample_start = _plain_int(
                    window_row["sample_start"],
                    name=f"V4 window[{index}].sample_start",
                )
                birth_start = _plain_int(
                    window_row["birth_start"],
                    name=f"V4 window[{index}].birth_start",
                )
                arm_levels = tuple(window_row["arm_levels"])
                if len(arm_levels) != len(_runtime_v4().ARM_KEYS):
                    raise FrozenEvaluationAuthorityError(
                        "V4 restored window arm-level shape drifted"
                    )
                rho_value = window_row["rho"]
                allocation_document = {
                    "schema_version": V4_SCHEMA_VERSION,
                    "kind": "frozen_window_allocation",
                    "state_owner_sha256": self._state_owner_sha256,
                    "seq": seq_value,
                    "key": key.as_dict(),
                    "evidence_role": role,
                    "domain_epoch": domain_epoch,
                    "stratum": window_row["stratum"],
                    "selected_arm_key": window_row[
                        "selected_arm_key"
                    ],
                    "selection_round": selection_round,
                    "arm_levels": list(arm_levels),
                    "rho": rho_value,
                    "policy_checkpoint_sha256": checkpoint[0],
                    "policy_generation": generation,
                    "seed_start": seed_start,
                    "sample_start": sample_start,
                    "birth_start": birth_start,
                    "proposal_count": expected_count,
                    "optional_stopping": False,
                }
                if (
                    _canonical_sha256(allocation_document)
                    != allocation_sha
                ):
                    raise FrozenEvaluationAuthorityError(
                        "V4 restored window allocation hash mismatch"
                    )
                for attempt_index, attempt in enumerate(attempts):
                    request = attempt.request
                    expected_request = (
                        checkpoint[0],
                        generation,
                        role,
                        attempt_index,
                        seed_start + attempt_index,
                        sample_start + attempt_index,
                        birth_start + attempt_index,
                        key.action_uid,
                        key.profile_sha256,
                        key.mobility,
                        domain_epoch,
                        tuple(arm_levels),
                        window_row["selected_arm_key"],
                    )
                    actual_request = (
                        request.policy_checkpoint_sha256,
                        request.policy_generation,
                        request.evidence_role,
                        request.proposal_offset,
                        request.seed,
                        request.sample_index,
                        request.birth_index,
                        request.action_uid,
                        request.profile_sha256,
                        request.mobility_mode,
                        request.domain_epoch,
                        tuple(
                            getattr(
                                request.domain_levels,
                                arm,
                            )
                            for arm in _runtime_v4().ARM_KEYS
                        ),
                        request.selected_arm_key,
                    )
                    if actual_request != expected_request:
                        raise FrozenEvaluationAuthorityError(
                            "V4 restored proposal differs from window "
                            "allocation"
                        )
                    expected_new_band = bool(
                        window_row["selected_arm_key"]
                        and attempt.proposal is not None
                        and attempt.proposal.sampling_stratum
                        == "frontier"
                        and attempt.proposal.frontier_arm
                        == window_row["selected_arm_key"]
                    )
                    if attempt.in_new_band != expected_new_band:
                        raise FrozenEvaluationAuthorityError(
                            "V4 restored new-band flag is not code-derived"
                        )
                    if attempt.proposal is not None and (
                        attempt.proposal.source_contract_sha256
                        != self._launch[
                            "attempt_source_contract_sha256"
                        ]
                    ):
                        raise FrozenEvaluationAuthorityError(
                            "V4 restored proposal source binding mismatch"
                        )
                window = _V4Window(
                    allocation_sha256=allocation_sha,
                    snapshot=snapshot,
                    key=key,
                    evidence_role=role,
                    domain_epoch=domain_epoch,
                    stratum=window_row["stratum"],
                    selected_arm_key=window_row[
                        "selected_arm_key"
                    ],
                    selection_round=selection_round,
                    arm_levels=arm_levels,
                    rho=rho_value,
                    seq=seq_value,
                    seed_start=seed_start,
                    sample_start=sample_start,
                    birth_start=birth_start,
                    attempts=attempts,
                    finalized=finalized,
                    evidence=evidence,
                    capability_id=capability_id,
                )
                if allocation_sha in windows:
                    raise FrozenEvaluationAuthorityError(
                        "V4 window allocation is duplicated"
                    )
                if finalized:
                    ledger = self._ledger(attempts)
                    if evidence.ledger != ledger:
                        raise FrozenEvaluationAuthorityError(
                            "V4 evidence ledger differs from transcript"
                        )
                    self._assert_sampling_mixture(window)
                    sample_rows = tuple(
                        attempt.sample_receipt_sha256()
                        for attempt in attempts
                    )
                    birth_rows = tuple(
                        attempt.birth_receipt_sha256()
                        for attempt in attempts
                    )
                    evidence_binding = (
                        evidence.key,
                        evidence.policy_checkpoint_sha256,
                        evidence.policy_generation,
                        evidence.evidence_role,
                        evidence.domain_epoch,
                        evidence.stratum,
                        evidence.selected_arm_key,
                        evidence.selection_round,
                        evidence.arm_levels,
                        evidence.rho,
                        evidence.seed_block_start,
                        evidence.seed_block_end_exclusive,
                        evidence.sample_id_start,
                        evidence.sample_id_end_exclusive,
                        evidence.sample_receipt_root_sha256,
                        evidence.unique_birth_count,
                        evidence.birth_receipt_root_sha256,
                        evidence.seq,
                        evidence.window_id,
                    )
                    expected_evidence_binding = (
                        key,
                        checkpoint[0],
                        generation,
                        role,
                        domain_epoch,
                        window_row["stratum"],
                        window_row["selected_arm_key"],
                        selection_round,
                        arm_levels,
                        rho_value,
                        seed_start,
                        seed_start + expected_count,
                        sample_start,
                        sample_start + expected_count,
                        ordered_sample_receipt_root(sample_rows),
                        expected_count,
                        ordered_birth_receipt_root(birth_rows),
                        seq_value,
                        allocation_sha,
                    )
                    if evidence_binding != expected_evidence_binding:
                        raise FrozenEvaluationAuthorityError(
                            "V4 evidence differs from exact restored "
                            "transcript/allocation"
                        )
                    admitted_tasks = [
                        attempt.solver.task_receipt_sha256
                        for attempt in attempts
                        if (
                            attempt.solver is not None
                            and attempt.solver.disposition
                            == "admitted"
                        )
                    ]
                    if len(admitted_tasks) != len(
                        set(admitted_tasks)
                    ):
                        raise FrozenEvaluationAuthorityError(
                            "V4 restored window reuses an admitted task"
                        )
                    expected_capability = _canonical_sha256(
                        {
                            "schema_version": V4_SCHEMA_VERSION,
                            "state_owner_sha256": (
                                self._state_owner_sha256
                            ),
                            "allocation_sha256": allocation_sha,
                            "window_sha256": (
                                evidence.window_sha256
                            ),
                            "attempt_transcript_sha256": (
                                _canonical_sha256(
                                    [
                                        attempt.row()
                                        for attempt in attempts
                                    ]
                                )
                            ),
                        }
                    )
                    if capability_id != expected_capability:
                        raise FrozenEvaluationAuthorityError(
                            "V4 window capability identity mismatch"
                        )
                windows[allocation_sha] = window
            pending_raw = row["pending_releases"]
            if not isinstance(pending_raw, list):
                raise FrozenEvaluationAuthorityError(
                    "V4 pending releases must be a list"
                )
            pending: Dict[str, Tuple[str, str]] = {}
            for index, raw_release in enumerate(pending_raw):
                if (
                    not isinstance(raw_release, list)
                    or len(raw_release) != 3
                ):
                    raise FrozenEvaluationAuthorityError(
                        f"V4 pending release[{index}] is invalid"
                    )
                release_id = _sha256(
                    raw_release[0],
                    name=f"V4 pending release[{index}].id",
                )
                pair = (
                    _sha256(
                        raw_release[1],
                        name=(
                            f"V4 pending release[{index}].canary"
                        ),
                    ),
                    _sha256(
                        raw_release[2],
                        name=(
                            f"V4 pending release[{index}].heldout"
                        ),
                    ),
                )
                if release_id in pending:
                    raise FrozenEvaluationAuthorityError(
                        "V4 pending release is duplicated"
                    )
                pending[release_id] = pair
            consumed_raw = row["consumed_releases"]
            used_raw = row["used_window_capabilities"]
            consumed_scheduler_raw = row[
                "consumed_scheduler_capabilities"
            ]
            if not isinstance(consumed_raw, list) or not isinstance(
                used_raw, list
            ) or not isinstance(
                consumed_scheduler_raw, list
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 release identity sets must be lists"
                )
            consumed = {
                _sha256(value, name="V4 consumed release")
                for value in consumed_raw
            }
            used = {
                _sha256(value, name="V4 used window capability")
                for value in used_raw
            }
            consumed_scheduler = {
                _sha256(
                    value,
                    name="V4 consumed scheduler capability",
                )
                for value in consumed_scheduler_raw
            }
            if (
                len(consumed) != len(consumed_raw)
                or len(used) != len(used_raw)
                or len(consumed_scheduler)
                != len(consumed_scheduler_raw)
                or set(pending) & consumed
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 release identity sets overlap or duplicate"
                )
            capabilities = {
                window.capability_id: window
                for window in windows.values()
                if window.capability_id
            }
            if not consumed_scheduler.issubset(capabilities):
                raise FrozenEvaluationAuthorityError(
                    "V4 consumed scheduler capability is missing"
                )
            if any(
                capabilities[capability_id].evidence_role
                != "scheduler"
                for capability_id in consumed_scheduler
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 consumed scheduler set contains a formal window"
                )
            for release_id, pair in pending.items():
                if not set(pair).issubset(capabilities):
                    raise FrozenEvaluationAuthorityError(
                        "V4 pending release names a missing capability"
                    )
                canary = capabilities[pair[0]].evidence
                heldout = capabilities[pair[1]].evidence
                expected_release = _canonical_sha256(
                    {
                        "schema_version": V4_SCHEMA_VERSION,
                        "kind": (
                            "action_ball_frozen_evaluation_release"
                        ),
                        "state_owner_sha256": (
                            self._state_owner_sha256
                        ),
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
                if (
                    release_id != expected_release
                    or canary.evidence_role != "frozen_canary"
                    or heldout.evidence_role != "frozen_heldout"
                ):
                    raise FrozenEvaluationAuthorityError(
                        "V4 pending release identity mismatch"
                    )
            policy_generation = _plain_int(
                row["policy_generation"],
                name="V4 policy_generation",
            )
            seq = _plain_int(row["seq"], name="V4 seq")
            seed_cursor = _plain_int(
                row["seed_cursor"],
                name="V4 seed_cursor",
            )
            sample_cursor = _plain_int(
                row["sample_cursor"],
                name="V4 sample_cursor",
            )
            birth_cursor = _plain_int(
                row["birth_cursor"],
                name="V4 birth_cursor",
            )
            ordered_windows = sorted(
                windows.values(),
                key=lambda item: item.seq,
            )
            if [
                window.seq for window in ordered_windows
            ] != list(range(1, len(ordered_windows) + 1)):
                raise FrozenEvaluationAuthorityError(
                    "V4 window seq tape is not contiguous"
                )
            expected_seed_start = 0
            expected_sample_start = 0
            expected_birth_start = 0
            for window in ordered_windows:
                if (
                    window.seed_start != expected_seed_start
                    or window.sample_start != expected_sample_start
                    or window.birth_start != expected_birth_start
                ):
                    raise FrozenEvaluationAuthorityError(
                        "V4 allocation ranges overlap or contain gaps"
                    )
                expected_seed_start += window.proposal_count
                expected_sample_start += window.proposal_count
                expected_birth_start += window.proposal_count
            if snapshots and policy_generation != max(snapshots):
                raise FrozenEvaluationAuthorityError(
                    "V4 policy generation high-water mismatch"
                )
            if not snapshots and policy_generation != 0:
                raise FrozenEvaluationAuthorityError(
                    "V4 empty policy tape has a nonzero high-water"
                )
            if windows and seq != max(
                window.seq for window in windows.values()
            ):
                raise FrozenEvaluationAuthorityError(
                    "V4 window seq high-water mismatch"
                )
            for cursor_name, cursor, field in (
                ("seed", seed_cursor, "seed"),
                ("sample", sample_cursor, "sample_index"),
                ("birth", birth_cursor, "birth_index"),
            ):
                highwater = max(
                    (
                        getattr(attempt.request, field) + 1
                        for window in windows.values()
                        for attempt in window.attempts
                    ),
                    default=0,
                )
                if cursor != highwater:
                    raise FrozenEvaluationAuthorityError(
                        f"V4 {cursor_name} cursor high-water mismatch"
                    )
            # Validate every restored event against the exact source
            # transcript after source state has been restored.
            for window in windows.values():
                for attempt in window.attempts:
                    if attempt.proposal is not None:
                        self._pure_source_assert(
                            "assert_exact_proposal",
                            attempt.request,
                            attempt.proposal,
                        )
                    if attempt.solver is not None:
                        self._pure_source_assert(
                            "assert_solver_event",
                            attempt.request,
                            attempt.proposal,
                            attempt.solver,
                        )
                    for lifecycle in (
                        attempt.install,
                        attempt.start,
                    ):
                        if lifecycle is not None:
                            self._pure_source_assert(
                                "assert_lifecycle_event",
                                attempt.request,
                                attempt.proposal,
                                attempt.solver,
                                lifecycle,
                            )
                    if attempt.terminal is not None:
                        self._pure_source_assert(
                            "assert_terminal_event",
                            attempt.request,
                            attempt.proposal,
                            attempt.solver,
                            attempt.terminal,
                        )
        except Exception:
            self._source.load_state_dict(previous_source_state)
            raise
        self._policy_generation = policy_generation
        self._seq = seq
        self._seed_cursor = seed_cursor
        self._sample_cursor = sample_cursor
        self._birth_cursor = birth_cursor
        self._snapshots = snapshots
        self._windows = windows
        self._pending_releases = pending
        self._consumed_releases = consumed
        self._used_window_capabilities = used
        self._consumed_scheduler_capabilities = consumed_scheduler
        self._lifetime = lifetime

    def policy_snapshot(
        self,
        generation: int,
    ) -> FrozenPolicySnapshotV4:
        generation = _plain_int(
            generation,
            name="policy generation",
        )
        try:
            digest, _ = self._snapshots[generation]
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "V4 policy snapshot generation is unknown"
            ) from exc
        return FrozenPolicySnapshotV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            digest,
            generation,
        )

    def pending_session(
        self,
        allocation_sha256: str,
    ) -> FrozenEvaluationSessionV4:
        digest = _sha256(
            allocation_sha256,
            name="V4 allocation_sha256",
        )
        try:
            window = self._windows[digest]
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "V4 evaluation window is unknown"
            ) from exc
        if window.finalized:
            raise FrozenEvaluationAuthorityError(
                "V4 evaluation window is already finalized"
            )
        return FrozenEvaluationSessionV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            digest,
        )

    def sidecar_request_plan(
        self,
        sessions: Sequence[FrozenEvaluationSessionV4],
    ) -> Dict[str, object]:
        """Expose only authority-allocated fields needed by the file adapter."""

        if isinstance(sessions, (str, bytes)) or not isinstance(
            sessions, Sequence
        ):
            raise FrozenEvaluationAuthorityError(
                "sidecar request sessions must be a sequence"
            )
        windows = tuple(self._window(session) for session in sessions)
        roles = tuple(window.evidence_role for window in windows)
        if roles == ("scheduler",):
            request_kind = "scheduler"
        elif roles == ("frozen_canary", "frozen_heldout"):
            request_kind = "formal"
        else:
            raise FrozenEvaluationAuthorityError(
                "sidecar request must bind one scheduler or canary/heldout"
            )
        first = windows[0]
        identity = (
            first.snapshot.checkpoint_sha256,
            first.snapshot.generation,
            first.key,
            first.domain_epoch,
            first.stratum,
            first.selected_arm_key,
            first.selection_round,
            first.arm_levels,
            first.rho,
        )
        if any(
            (
                window.snapshot.checkpoint_sha256,
                window.snapshot.generation,
                window.key,
                window.domain_epoch,
                window.stratum,
                window.selected_arm_key,
                window.selection_round,
                window.arm_levels,
                window.rho,
            )
            != identity
            for window in windows[1:]
        ):
            raise FrozenEvaluationAuthorityError(
                "sidecar request windows do not share one frozen target"
            )
        if len(windows) == 2 and any(
            getattr(windows[1], f"{axis}_start")
            != getattr(windows[0], f"{axis}_start")
            + windows[0].proposal_count
            for axis in ("seed", "sample", "birth")
        ):
            raise FrozenEvaluationAuthorityError(
                "sidecar heldout allocation is not contiguous after canary"
            )
        return {
            "schema_version": V4_SCHEMA_VERSION,
            "request_kind": request_kind,
            "checkpoint_sha256": first.snapshot.checkpoint_sha256,
            "policy_generation": first.snapshot.generation,
            "key": first.key.as_dict(),
            "target": {
                "action_uid": first.key.action_uid,
                "profile_sha256": first.key.profile_sha256,
                "mobility_mode": first.key.mobility,
                "domain_epoch": first.domain_epoch,
                "stratum": first.stratum,
                "selected_arm_key": first.selected_arm_key,
                "selection_round": first.selection_round,
                "arm_levels": list(first.arm_levels),
                "rho": first.rho,
            },
            "seed_start": first.seed_start,
            "sample_start": first.sample_start,
            "birth_start": first.birth_start,
            "allocation_sha256": [
                window.allocation_sha256 for window in windows
            ],
        }

    def pending_capability(
        self,
        capability_id: str,
    ) -> FrozenEvaluationCapabilityV4:
        digest = _sha256(
            capability_id,
            name="V4 capability_id",
        )
        matches = [
            window
            for window in self._windows.values()
            if window.capability_id == digest
        ]
        if len(matches) != 1 or matches[0].evidence is None:
            raise FrozenEvaluationAuthorityError(
                "V4 evaluation capability is unknown"
            )
        return FrozenEvaluationCapabilityV4(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            digest,
            matches[0].evidence,
        )

    def complete_sidecar_window(
        self,
        allocation_sha256: str,
    ) -> FrozenEvaluationCapabilityV4:
        """Resume or complete one exact source-backed window."""

        digest = _sha256(
            allocation_sha256,
            name="V4 sidecar allocation_sha256",
        )
        try:
            window = self._windows[digest]
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "V4 sidecar window is unknown"
            ) from exc
        if window.finalized:
            return self.pending_capability(window.capability_id)
        return self.replay_window_from_source(
            self.pending_session(digest)
        )

    def issue_or_resume_sidecar_release(
        self,
        *,
        canary: FrozenEvaluationCapabilityV4,
        heldout: FrozenEvaluationCapabilityV4,
    ) -> FrozenEvaluationReleaseReceipt:
        """Idempotently recover the exact pending release after a crash."""

        canary_window = self._capability_window(canary)
        heldout_window = self._capability_window(heldout)
        pair = (
            canary_window.capability_id,
            heldout_window.capability_id,
        )
        matches = [
            release_id
            for release_id, existing in self._pending_releases.items()
            if existing == pair
        ]
        if len(matches) > 1:
            raise FrozenEvaluationAuthorityError(
                "V4 pending release pair is duplicated"
            )
        if matches:
            return self.pending_release(matches[0])
        return self.issue_release(canary=canary, heldout=heldout)

    def pending_release(
        self,
        release_id: str,
    ) -> FrozenEvaluationReleaseReceipt:
        digest = _sha256(
            release_id,
            name="V4 release_id",
        )
        try:
            pair = self._pending_releases[digest]
        except KeyError as exc:
            raise FrozenEvaluationAuthorityError(
                "V4 release is not pending"
            ) from exc
        capabilities = {
            window.capability_id: window
            for window in self._windows.values()
            if window.capability_id
        }
        canary = capabilities[pair[0]].evidence
        heldout = capabilities[pair[1]].evidence
        return FrozenEvaluationReleaseReceipt(
            _V4_MINT_SENTINEL,
            self,
            self._lifetime,
            digest,
            canary,
            heldout,
        )

    def assert_sidecar_result_consumed(
        self,
        *,
        request_kind: str,
        result_id: str,
    ) -> None:
        """Prove curriculum ingestion consumed one sidecar result."""

        digest = _sha256(result_id, name="V4 sidecar result_id")
        if request_kind == "scheduler":
            if digest not in self._consumed_scheduler_capabilities:
                raise FrozenEvaluationAuthorityError(
                    "scheduler sidecar capability is not consumed"
                )
        elif request_kind == "formal":
            if digest not in self._consumed_releases:
                raise FrozenEvaluationAuthorityError(
                    "formal sidecar release is not consumed"
                )
        else:
            raise FrozenEvaluationAuthorityError(
                "sidecar request_kind must be scheduler or formal"
            )

    def assert_sidecar_request_consumed(self, request_seq: int) -> str:
        if not callable(
            getattr(self._source, "assert_request_consumed", None)
        ):
            raise FrozenEvaluationAuthorityError(
                "attempt source lacks request-consumption proof"
            )
        result = self._pure_source_call(
            "assert_request_consumed",
            request_seq,
        )
        return _sha256(
            result,
            name="sidecar consumed evidence SHA",
        )


# Alternate spelling kept explicit for discoverability in integration code.
FrozenEvaluatorAuthorityV4 = FrozenEvaluatorV4Authority
