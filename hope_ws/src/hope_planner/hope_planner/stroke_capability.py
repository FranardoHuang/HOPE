"""Fail-closed capability selection over an ordered action catalog.

The physical planner remains responsible for producing one hard-safe candidate per
catalog action.  This module only ranks those candidates using held-out capability
evidence from one *specific* policy/checkpoint and task/reward definition.

Every identity that can change the meaning of a probability is bound into
``CapabilityArtifact``.  Selection refuses catalog reordering, missing candidates,
or a stale artifact.  A bad numeric value invalidates only its own candidate; it
must never turn into a process-wide ``nan`` comparison or make an unsafe action
selectable.

Pure standard-library Python 3.8; this file is shared by ROS-side tests and training
artifact tooling.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


CAPABILITY_SCHEMA_VERSION = 1
_SHA256_HEX_LEN = 64
# Exact integer range shared by the flat JSON/ROS/C++ action catalog.  Staying at
# or below 2**53-1 preserves identity through IEEE-754 JSON consumers.
_ACTION_UID_MAX = (1 << 53) - 1


class CapabilityContractError(ValueError):
    """A caller crossed an identity/schema boundary and selection must not run."""


def _is_exact_int(value: object) -> bool:
    # Identity/control integers cross JSON and C++; accepting bool, float, or a
    # library scalar here would make the Python contract wider than the catalog.
    return type(value) is int


def _require_action_uid(value: object, field: str = "action_uid") -> int:
    if not _is_exact_int(value):
        raise CapabilityContractError(f"{field} must be an integer, got {value!r}")
    uid = int(value)
    if not (0 < uid <= _ACTION_UID_MAX):
        raise CapabilityContractError(
            f"{field} must be in [1, {_ACTION_UID_MAX}], got {uid!r}; "
            "zero is the abstain sentinel"
        )
    return uid


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str:
        raise CapabilityContractError(f"{field} must be a lowercase sha256 string")
    if len(value) != _SHA256_HEX_LEN:
        raise CapabilityContractError(
            f"{field} must contain exactly {_SHA256_HEX_LEN} lowercase hex characters"
        )
    if value != value.lower() or any(ch not in "0123456789abcdef" for ch in value):
        raise CapabilityContractError(f"{field} must be lowercase hexadecimal")
    return value


def _canonical_sha256(mapping: Mapping[str, object]) -> str:
    payload = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _catalog_rows(catalog: object) -> Tuple[object, ...]:
    try:
        rows = tuple(catalog.actions)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise CapabilityContractError("catalog.actions must be an ordered iterable") from exc
    if not rows:
        raise CapabilityContractError("catalog must contain at least one action")

    seen_uids = set()
    seen_ids = set()
    for expected_slot, row in enumerate(rows):
        try:
            uid = _require_action_uid(row.action_uid, "catalog action_uid")
            action_id = row.action_id
            slot = row.slot
        except AttributeError as exc:
            raise CapabilityContractError(
                "each catalog action must expose action_id, action_uid, and slot"
            ) from exc
        if (
            not isinstance(action_id, str)
            or not action_id.strip()
            or action_id != action_id.strip()
        ):
            raise CapabilityContractError(
                f"catalog slot {expected_slot}: action_id must be a non-empty trimmed string"
            )
        if not _is_exact_int(slot) or int(slot) != expected_slot:
            raise CapabilityContractError(
                f"catalog action {action_id!r}: slot {slot!r} does not match ordered "
                f"position {expected_slot}"
            )
        if uid in seen_uids:
            raise CapabilityContractError(f"catalog contains duplicate action_uid {uid}")
        if action_id in seen_ids:
            raise CapabilityContractError(f"catalog contains duplicate action_id {action_id!r}")
        seen_uids.add(uid)
        seen_ids.add(action_id)

        try:
            found = catalog.by_uid(uid)  # type: ignore[attr-defined]
        except (AttributeError, KeyError, TypeError) as exc:
            raise CapabilityContractError(
                f"catalog.by_uid({uid}) does not resolve its slot-{expected_slot} action"
            ) from exc
        if getattr(found, "action_uid", None) != uid or \
                getattr(found, "action_id", None) != action_id or \
                getattr(found, "slot", None) != expected_slot:
            raise CapabilityContractError(
                f"catalog.by_uid({uid}) returned a different action identity or slot"
            )

    _require_sha256(
        getattr(catalog, "catalog_sha256", None), "catalog.catalog_sha256"
    )
    return rows


def _catalog_action_uids(catalog: object) -> Tuple[int, ...]:
    return tuple(int(row.action_uid) for row in _catalog_rows(catalog))


@dataclass(frozen=True)
class CapabilityArtifact:
    """Identity receipt for one calibrated, held-out capability model."""

    schema_version: int
    catalog_sha256: str
    policy_sha256: str
    task_sha256: str
    reward_sha256: str
    heldout_sha256: str
    model_sha256: str
    calibration_sha256: str
    action_uids: Tuple[int, ...]
    artifact_sha256: str

    _FIELDS = frozenset(
        (
            "schema_version",
            "catalog_sha256",
            "policy_sha256",
            "task_sha256",
            "reward_sha256",
            "heldout_sha256",
            "model_sha256",
            "calibration_sha256",
            "action_uids",
            "artifact_sha256",
        )
    )

    def __post_init__(self) -> None:
        if not _is_exact_int(self.schema_version) or int(self.schema_version) != \
                CAPABILITY_SCHEMA_VERSION:
            raise CapabilityContractError(
                f"capability schema_version must be {CAPABILITY_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )
        for name in (
            "catalog_sha256",
            "policy_sha256",
            "task_sha256",
            "reward_sha256",
            "heldout_sha256",
            "model_sha256",
            "calibration_sha256",
            "artifact_sha256",
        ):
            _require_sha256(getattr(self, name), name)

        try:
            uids = tuple(
                _require_action_uid(uid, f"action_uids[{index}]")
                for index, uid in enumerate(self.action_uids)
            )
        except TypeError as exc:
            raise CapabilityContractError("action_uids must be an ordered sequence") from exc
        if not uids:
            raise CapabilityContractError("action_uids must not be empty")
        if len(set(uids)) != len(uids):
            raise CapabilityContractError("action_uids contains a duplicate identity")
        object.__setattr__(self, "action_uids", uids)

        expected = _canonical_sha256(self._unsigned_mapping())
        if self.artifact_sha256 != expected:
            raise CapabilityContractError(
                "artifact_sha256 does not match the canonical capability metadata: "
                f"declared {self.artifact_sha256}, computed {expected}"
            )

    def _unsigned_mapping(self) -> Dict[str, object]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "policy_sha256": self.policy_sha256,
            "task_sha256": self.task_sha256,
            "reward_sha256": self.reward_sha256,
            "heldout_sha256": self.heldout_sha256,
            "model_sha256": self.model_sha256,
            "calibration_sha256": self.calibration_sha256,
            "action_uids": list(self.action_uids),
        }

    @classmethod
    def create(
        cls,
        *,
        catalog: object,
        policy_sha256: str,
        task_sha256: str,
        reward_sha256: str,
        heldout_sha256: str,
        model_sha256: str,
        calibration_sha256: str,
    ) -> "CapabilityArtifact":
        rows = _catalog_rows(catalog)
        unsigned = {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "catalog_sha256": _require_sha256(
                getattr(catalog, "catalog_sha256", None), "catalog.catalog_sha256"
            ),
            "policy_sha256": _require_sha256(policy_sha256, "policy_sha256"),
            "task_sha256": _require_sha256(task_sha256, "task_sha256"),
            "reward_sha256": _require_sha256(reward_sha256, "reward_sha256"),
            "heldout_sha256": _require_sha256(heldout_sha256, "heldout_sha256"),
            "model_sha256": _require_sha256(model_sha256, "model_sha256"),
            "calibration_sha256": _require_sha256(
                calibration_sha256, "calibration_sha256"
            ),
            "action_uids": [int(row.action_uid) for row in rows],
        }
        return cls(
            artifact_sha256=_canonical_sha256(unsigned),
            action_uids=tuple(unsigned["action_uids"]),  # type: ignore[arg-type]
            schema_version=CAPABILITY_SCHEMA_VERSION,
            catalog_sha256=str(unsigned["catalog_sha256"]),
            policy_sha256=str(unsigned["policy_sha256"]),
            task_sha256=str(unsigned["task_sha256"]),
            reward_sha256=str(unsigned["reward_sha256"]),
            heldout_sha256=str(unsigned["heldout_sha256"]),
            model_sha256=str(unsigned["model_sha256"]),
            calibration_sha256=str(unsigned["calibration_sha256"]),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CapabilityArtifact":
        if not isinstance(raw, Mapping):
            raise CapabilityContractError("capability artifact must be a mapping")
        keys = set(raw.keys())
        missing = cls._FIELDS - keys
        extra = keys - cls._FIELDS
        if missing or extra:
            raise CapabilityContractError(
                "capability artifact fields mismatch: "
                f"missing={sorted(str(key) for key in missing)}, "
                f"extra={sorted(str(key) for key in extra)}"
            )
        if not isinstance(raw["action_uids"], list):
            raise CapabilityContractError(
                "action_uids must be a JSON list in a capability artifact mapping"
            )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            catalog_sha256=raw["catalog_sha256"],  # type: ignore[arg-type]
            policy_sha256=raw["policy_sha256"],  # type: ignore[arg-type]
            task_sha256=raw["task_sha256"],  # type: ignore[arg-type]
            reward_sha256=raw["reward_sha256"],  # type: ignore[arg-type]
            heldout_sha256=raw["heldout_sha256"],  # type: ignore[arg-type]
            model_sha256=raw["model_sha256"],  # type: ignore[arg-type]
            calibration_sha256=raw["calibration_sha256"],  # type: ignore[arg-type]
            action_uids=tuple(raw["action_uids"]),  # type: ignore[arg-type]
            artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> Dict[str, object]:
        out = self._unsigned_mapping()
        out["artifact_sha256"] = self.artifact_sha256
        return out

    def assert_compatible(
        self,
        *,
        catalog: object,
        policy_sha256: str,
        task_sha256: str,
        reward_sha256: str,
        heldout_sha256: str,
        model_sha256: str,
        calibration_sha256: str,
    ) -> None:
        rows = _catalog_rows(catalog)
        expected = {
            "catalog_sha256": _require_sha256(
                getattr(catalog, "catalog_sha256", None), "catalog.catalog_sha256"
            ),
            "policy_sha256": _require_sha256(policy_sha256, "policy_sha256"),
            "task_sha256": _require_sha256(task_sha256, "task_sha256"),
            "reward_sha256": _require_sha256(reward_sha256, "reward_sha256"),
            "heldout_sha256": _require_sha256(heldout_sha256, "heldout_sha256"),
            "model_sha256": _require_sha256(model_sha256, "model_sha256"),
            "calibration_sha256": _require_sha256(
                calibration_sha256, "calibration_sha256"
            ),
        }
        for field, wanted in expected.items():
            got = getattr(self, field)
            if got != wanted:
                raise CapabilityContractError(
                    f"capability {field} mismatch: artifact={got}, current={wanted}"
                )
        catalog_uids = tuple(int(row.action_uid) for row in rows)
        if self.action_uids != catalog_uids:
            raise CapabilityContractError(
                "capability action_uids do not match catalog slot order: "
                f"artifact={self.action_uids}, catalog={catalog_uids}"
            )


@dataclass(frozen=True)
class SelectorProfile:
    """Runtime gates.  Smaller integer priority is stronger, matching stroke_select."""

    min_support: int
    max_ood_score: float
    min_lcb_success: float
    delta_tie: float
    priority_by_uid: Mapping[int, int]

    def __post_init__(self) -> None:
        if not _is_exact_int(self.min_support) or int(self.min_support) < 1:
            raise CapabilityContractError("min_support must be an integer >= 1")
        for field in ("max_ood_score", "min_lcb_success", "delta_tie"):
            raw = getattr(self, field)
            if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
                raise CapabilityContractError(f"{field} must be a finite number in [0, 1]")
            value = float(raw)
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise CapabilityContractError(f"{field} must be a finite number in [0, 1]")
            object.__setattr__(self, field, value)
        if not isinstance(self.priority_by_uid, Mapping):
            raise CapabilityContractError("priority_by_uid must be a mapping")
        copied: Dict[int, int] = {}
        for raw_uid, raw_priority in self.priority_by_uid.items():
            uid = _require_action_uid(raw_uid, "priority_by_uid key")
            if not _is_exact_int(raw_priority) or int(raw_priority) < 0:
                raise CapabilityContractError(
                    f"priority_by_uid[{uid}] must be an integer >= 0"
                )
            copied[uid] = int(raw_priority)
        if not copied:
            raise CapabilityContractError("priority_by_uid must not be empty")
        object.__setattr__(self, "priority_by_uid", MappingProxyType(copied))

    def assert_for_catalog(self, catalog: object) -> None:
        wanted = set(_catalog_action_uids(catalog))
        got = set(self.priority_by_uid.keys())
        if got != wanted:
            raise CapabilityContractError(
                "priority_by_uid must cover the catalog exactly: "
                f"missing={sorted(wanted - got)}, extra={sorted(got - wanted)}"
            )


@dataclass(frozen=True)
class CandidateEvidence:
    """One adapter candidate plus optional learned capability evidence.

    Hard failures are rejected before numeric evidence is inspected.  They may carry
    diagnostics (including an optimistic finite score) without ever becoming eligible.
    Hard passes must provide all three learned-capability fields; malformed numeric
    values are intentionally handled per candidate by :func:`select_action`.
    """

    action_uid: int
    hard_ok: bool
    hard_reason: str
    support_count: Optional[int] = None
    ood_score: Optional[float] = None
    lcb_success: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_uid", _require_action_uid(self.action_uid, "candidate action_uid")
        )
        if not isinstance(self.hard_ok, bool):
            raise CapabilityContractError("hard_ok must be a bool")
        if not isinstance(self.hard_reason, str):
            raise CapabilityContractError("hard_reason must be a string")
        if self.hard_ok:
            if self.hard_reason:
                raise CapabilityContractError("hard-pass candidate hard_reason must be empty")
            if self.support_count is None or self.ood_score is None or \
                    self.lcb_success is None:
                raise CapabilityContractError(
                    "hard-pass candidate must provide support_count, ood_score, and lcb_success"
                )
        elif not self.hard_reason.strip():
            raise CapabilityContractError(
                "hard-fail candidate must provide a non-empty hard_reason"
            )


@dataclass(frozen=True)
class ActionAssessment:
    """Stable per-slot audit row returned for every non-contract selection result."""

    action_uid: int
    action_id: str
    slot: int
    priority: int
    eligible: bool
    reason: str
    hard_reason: str
    support_count: Optional[int]
    ood_score: Optional[float]
    lcb_success: Optional[float]


@dataclass(frozen=True)
class Decision:
    """Selection or an explicit abstention.

    ``0 / "" / -1`` is the only abstention identity.  Contract violations raise
    :class:`CapabilityContractError` instead of returning an apparently ordinary
    abstention.
    """

    selected_action_uid: int
    selected_action_id: str
    selected_slot: int
    reason: str
    assessments: Tuple[ActionAssessment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise CapabilityContractError("decision reason must be a non-empty string")
        if type(self.assessments) is not tuple:
            raise CapabilityContractError("decision assessments must be a tuple")
        if any(not isinstance(row, ActionAssessment) for row in self.assessments):
            raise CapabilityContractError(
                "decision assessments must contain only ActionAssessment rows"
            )
        if self.selected_action_uid == 0:
            if self.selected_action_id != "" or self.selected_slot != -1:
                raise CapabilityContractError(
                    "abstention identity must be exactly action_uid=0, action_id='', slot=-1"
                )
        else:
            _require_action_uid(self.selected_action_uid, "selected_action_uid")
            if not isinstance(self.selected_action_id, str) or not self.selected_action_id:
                raise CapabilityContractError(
                    "selected_action_id must be a non-empty string for a selection"
                )
            if not _is_exact_int(self.selected_slot) or self.selected_slot < 0:
                raise CapabilityContractError(
                    "selected_slot must be an integer >= 0 for a selection"
                )

    @property
    def selected_uid(self) -> int:
        return self.selected_action_uid

    @property
    def selected_id(self) -> str:
        return self.selected_action_id

    @property
    def abstained(self) -> bool:
        return self.selected_action_uid == 0


def _valid_support(value: object) -> bool:
    return _is_exact_int(value) and int(value) >= 0


def _valid_unit_interval(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    number = float(value)
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _abstain(reason: str, rows: Sequence[ActionAssessment]) -> Decision:
    return Decision(
        selected_action_uid=0,
        selected_action_id="",
        selected_slot=-1,
        reason=reason,
        assessments=tuple(rows),
    )


def _no_eligible_reason(rows: Sequence[ActionAssessment]) -> str:
    reasons = {row.reason for row in rows}
    if reasons == {"hard_reject"}:
        return "abstain_all_hard_rejected"
    if reasons == {"invalid_evidence"}:
        return "abstain_all_invalid_evidence"
    if reasons == {"low_support"}:
        return "abstain_all_low_support"
    if reasons == {"ood"}:
        return "abstain_all_ood"
    if reasons == {"below_min_lcb"}:
        return "abstain_below_min_lcb"
    return "abstain_no_eligible_candidate"


def select_action(
    catalog: object,
    artifact: CapabilityArtifact,
    profile: SelectorProfile,
    candidates: Sequence[CandidateEvidence],
    *,
    policy_sha256: str,
    task_sha256: str,
    reward_sha256: str,
    heldout_sha256: str,
    model_sha256: str,
    calibration_sha256: str,
) -> Decision:
    """Choose one action, or abstain, under strict identity and safety ordering."""

    if not isinstance(artifact, CapabilityArtifact):
        raise CapabilityContractError("artifact must be a CapabilityArtifact")
    if not isinstance(profile, SelectorProfile):
        raise CapabilityContractError("profile must be a SelectorProfile")
    artifact.assert_compatible(
        catalog=catalog,
        policy_sha256=policy_sha256,
        task_sha256=task_sha256,
        reward_sha256=reward_sha256,
        heldout_sha256=heldout_sha256,
        model_sha256=model_sha256,
        calibration_sha256=calibration_sha256,
    )
    profile.assert_for_catalog(catalog)
    actions = _catalog_rows(catalog)

    try:
        candidate_rows = tuple(candidates)
    except TypeError as exc:
        raise CapabilityContractError("candidates must be an ordered sequence") from exc
    for index, candidate in enumerate(candidate_rows):
        if not isinstance(candidate, CandidateEvidence):
            raise CapabilityContractError(
                f"candidates[{index}] must be a CandidateEvidence"
            )
    candidate_uids = tuple(candidate.action_uid for candidate in candidate_rows)
    catalog_uids = tuple(int(action.action_uid) for action in actions)
    if len(set(candidate_uids)) != len(candidate_uids):
        raise CapabilityContractError(
            f"candidate sequence contains duplicate action_uid(s): {candidate_uids}"
        )
    unknown = sorted(set(candidate_uids) - set(catalog_uids))
    if unknown:
        raise CapabilityContractError(
            f"candidate sequence contains unknown action_uid(s): {unknown}"
        )
    if len(candidate_uids) != len(catalog_uids):
        raise CapabilityContractError(
            "candidate sequence must contain exactly one row per catalog slot: "
            f"got {len(candidate_uids)}, expected {len(catalog_uids)}"
        )
    if candidate_uids != catalog_uids:
        raise CapabilityContractError(
            "candidate action_uid sequence must equal catalog slot order: "
            f"got {candidate_uids}, expected {catalog_uids}"
        )

    assessments = []
    for action, candidate in zip(actions, candidate_rows):
        uid = int(action.action_uid)
        priority = profile.priority_by_uid[uid]

        # P0: physical/adapter safety always wins.  Do not even inspect capability
        # numerics on this branch.
        if not candidate.hard_ok:
            assessments.append(
                ActionAssessment(
                    action_uid=uid,
                    action_id=str(action.action_id),
                    slot=int(action.slot),
                    priority=priority,
                    eligible=False,
                    reason="hard_reject",
                    hard_reason=candidate.hard_reason,
                    support_count=candidate.support_count,
                    ood_score=candidate.ood_score,
                    lcb_success=candidate.lcb_success,
                )
            )
            continue

        # Invalid learned evidence fails closed for this row, without poisoning
        # the other actions or turning into an exception-driven global outage.
        if not _valid_support(candidate.support_count) or \
                not _valid_unit_interval(candidate.ood_score) or \
                not _valid_unit_interval(candidate.lcb_success):
            assessments.append(
                ActionAssessment(
                    action_uid=uid,
                    action_id=str(action.action_id),
                    slot=int(action.slot),
                    priority=priority,
                    eligible=False,
                    reason="invalid_evidence",
                    hard_reason="",
                    support_count=candidate.support_count,
                    ood_score=candidate.ood_score,
                    lcb_success=candidate.lcb_success,
                )
            )
            continue

        support = int(candidate.support_count)  # type: ignore[arg-type]
        ood = float(candidate.ood_score)  # type: ignore[arg-type]
        lcb = float(candidate.lcb_success)  # type: ignore[arg-type]
        if support < profile.min_support:
            reason = "low_support"
            eligible = False
        elif ood > profile.max_ood_score:
            reason = "ood"
            eligible = False
        elif lcb < profile.min_lcb_success:
            # The success floor is an admission condition for every action,
            # not merely a global check on the best row.  Otherwise a wide
            # ``delta_tie`` plus a strong manual priority could revive a row
            # whose conservative success estimate is below the abstain floor.
            reason = "below_min_lcb"
            eligible = False
        else:
            reason = "eligible"
            eligible = True
        assessments.append(
            ActionAssessment(
                action_uid=uid,
                action_id=str(action.action_id),
                slot=int(action.slot),
                priority=priority,
                eligible=eligible,
                reason=reason,
                hard_reason="",
                support_count=support,
                ood_score=ood,
                lcb_success=lcb,
            )
        )

    eligible_rows = [row for row in assessments if row.eligible]
    if not eligible_rows:
        return _abstain(_no_eligible_reason(assessments), assessments)

    best_lcb = max(float(row.lcb_success) for row in eligible_rows)

    tie_floor = best_lcb - profile.delta_tie
    tie_rows = [
        row for row in eligible_rows if float(row.lcb_success) >= tie_floor
    ]
    selected = min(
        tie_rows,
        key=lambda row: (
            row.priority,
            -float(row.lcb_success),
            row.action_uid,
        ),
    )
    final_rows = [
        replace(row, reason="selected")
        if row.action_uid == selected.action_uid
        else replace(row, reason="eligible_not_selected")
        if row.eligible
        else row
        for row in assessments
    ]
    return Decision(
        selected_action_uid=selected.action_uid,
        selected_action_id=selected.action_id,
        selected_slot=selected.slot,
        reason="selected",
        assessments=tuple(final_rows),
    )
