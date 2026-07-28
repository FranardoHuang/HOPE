"""Code-rooted evaluation authority for action-conditioned ball-first training.

The authority turns exact ordered attempt transcripts into opaque, single-use
capabilities.  Online scheduler windows and frozen certification windows use
the same transport, but only frozen canary/heldout evidence may authorize a
curriculum frontier change.  Every window binds both sample and birth
receipts, the signed-arm catalog, and the deterministic scheduler contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Dict, Mapping, Sequence, Tuple

try:  # Package import in production.
    from .action_ball_curriculum import (
        ARM_CATALOG_SHA256,
        EVIDENCE_SCHEMA_VERSION,
        ActionProfileKey,
        BallDomainEvidence,
        BallOutcomeLedger,
    )
except ImportError:  # Dependency-light direct-file tests.
    from action_ball_curriculum import (  # type: ignore
        ARM_CATALOG_SHA256,
        EVIDENCE_SCHEMA_VERSION,
        ActionProfileKey,
        BallDomainEvidence,
        BallOutcomeLedger,
    )


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
