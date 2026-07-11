"""Shared, immutable schema-v3 BankExam schedule artifacts.

This module deliberately has no Isaac Lab, MuJoCo, or torch import.  It is the common paper
builder for evaluator-owned BankExam adapters: both simulators can load the same canonical JSON
artifact, verify it against the same schema-v3 exam bank, and consume the same finite sequence.

The existing schedule code in :mod:`venue_ball_sampler` is physics-contract-bound and must not be
edited merely to share a paper with another evaluator.  ``install_schedule_on_sampler`` therefore
adapts a validated artifact onto an already constructed ``BankExamSampler`` without changing that
module.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 3
BANK_SCHEMA_VERSION = 3
ARTIFACT_TYPE = "bank-exam-schedule"
HOLD_SEMANTICS = "stand-policy-actions-then-raw-frame0-v1"
_HEX = frozenset("0123456789abcdef")
_UINT64_LIMIT = 1 << 64
_INT32_LIMIT = 1 << 31

__all__ = (
    "ARTIFACT_TYPE",
    "BANK_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "BankExamScheduleArtifact",
    "BankExamScheduleItem",
    "HOLD_SEMANTICS",
    "artifact_document",
    "atomic_bank_question_id",
    "canonical_json_bytes",
    "canonical_schedule_sha256",
    "derive_atomic_question_ids",
    "derive_question_bank_question_ids",
    "derive_sampler_question_ids",
    "install_schedule_on_sampler",
    "load_schedule_artifact",
    "materialize_balanced_bank_exam_schedule",
    "sha256_file",
    "validate_schedule_artifact",
    "write_schedule_artifact",
)


@dataclass(frozen=True)
class BankExamScheduleItem:
    """One no-wrap question on the shared paper.

    The field names intentionally match ``venue_ball_sampler.BankExamScheduleItem`` so an item is
    a structural drop-in for ``BankExamSampler.sample``.
    """

    schedule_index: int
    clip: int
    bank_row: int
    question_id: str
    hold_steps: int
    attempt_seed: int
    repeat: int = 0


@dataclass(frozen=True)
class BankExamScheduleArtifact:
    """Validated in-memory representation of one canonical schedule artifact."""

    schema_version: int
    bank_schema_version: int
    artifact_type: str
    bank_sha256: str
    clip_order: tuple[str, ...]
    question_counts: tuple[int, ...]
    per_clip_quota: int
    schedule_seed: int
    hold_range: tuple[int, int]
    hold_semantics: str
    no_wrap: bool
    items: tuple[BankExamScheduleItem, ...]
    schedule_sha256: str

    @property
    def schedule(self) -> tuple[BankExamScheduleItem, ...]:
        """Compatibility spelling used by evaluator adapters."""
        return self.items

    @property
    def schedule_k(self) -> int:
        return len(self.items)

    @property
    def schedule_question_ids(self) -> tuple[str, ...]:
        return tuple(item.question_id for item in self.items)


def canonical_json_bytes(value) -> bytes:
    """Return the single JSON byte representation used for artifact hashing and writing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_payload(artifact: BankExamScheduleArtifact) -> dict:
    return {
        "artifact_type": artifact.artifact_type,
        "bank_schema_version": artifact.bank_schema_version,
        "bank_sha256": artifact.bank_sha256,
        "clip_order": list(artifact.clip_order),
        "hold_range": list(artifact.hold_range),
        "hold_semantics": artifact.hold_semantics,
        "items": [asdict(item) for item in artifact.items],
        "no_wrap": artifact.no_wrap,
        "per_clip_quota": artifact.per_clip_quota,
        "question_counts": list(artifact.question_counts),
        "schedule_seed": artifact.schedule_seed,
        "schema_version": artifact.schema_version,
    }


def artifact_document(artifact: BankExamScheduleArtifact) -> dict:
    """Return a fresh JSON-compatible document, including its declared schedule SHA."""
    document = _artifact_payload(artifact)
    document["schedule_sha256"] = artifact.schedule_sha256
    return document


def canonical_schedule_sha256(value: BankExamScheduleArtifact | Mapping) -> str:
    """Hash every semantic schedule field using canonical JSON.

    ``schedule_sha256`` is excluded from the hash to avoid a self-reference.  Mapping input is
    useful to artifact producers in other runtimes and to integrity tooling.
    """
    if isinstance(value, BankExamScheduleArtifact):
        payload = _artifact_payload(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
        payload.pop("schedule_sha256", None)
    else:
        raise TypeError("schedule hash input must be an artifact or mapping")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _plain_int(name: str, value, *, minimum: int = 0, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}], got {result}")
    return result


def _sha256_hex(name: str, value) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256, got {value!r}")
    return value


def _clip_order(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("clip_order must be a non-empty sequence of clip names")
    names = tuple(value)
    if not names:
        raise ValueError("clip_order must be non-empty")
    for name in names:
        if (not isinstance(name, str) or not name or ":" in name or "\0" in name
                or name.strip() != name):
            raise ValueError(f"invalid clip name in clip_order: {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"clip_order contains duplicates: {names!r}")
    return names


def _base_question_id(value, *, location: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise ValueError(
            f"{location} must be a 64-character lowercase atomic question id, got {value!r}"
        )
    return value


def _normalize_question_ids(
    clip_names: tuple[str, ...], question_ids: Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], ...]:
    if isinstance(question_ids, (str, bytes)) or len(question_ids) != len(clip_names):
        raise ValueError(
            "question_ids must have exactly one row-id sequence per clip; "
            f"clips={len(clip_names)}, sequences={len(question_ids)}"
        )
    result = []
    for clip, (name, values) in enumerate(zip(clip_names, question_ids)):
        if isinstance(values, (str, bytes)):
            raise ValueError(f"question_ids[{clip}] for {name!r} must be a sequence")
        ids = tuple(
            _base_question_id(value, location=f"question_ids[{clip}][{row}]")
            for row, value in enumerate(values)
        )
        if not ids:
            raise ValueError(f"clip {name!r} has no questions")
        if len(set(ids)) != len(ids):
            raise ValueError(f"clip {name!r} has duplicate atomic question ids")
        result.append(ids)
    return tuple(result)


def atomic_bank_question_id(
    *,
    bank_sha256,
    clip,
    bank_row,
    incoming_vel,
    incoming_spin,
    demanded_vel,
    demanded_normal,
) -> str:
    """Content-address one complete atomic bank row.

    This intentionally preserves the production ``bank-exam-question-v1`` byte contract so the
    shared artifact IDs exactly match IDs independently derived by the existing MuJoCo sampler.
    """
    bank_sha = _sha256_hex("bank_sha256", bank_sha256)
    clip_id = _plain_int("clip", clip, maximum=_INT32_LIMIT - 1)
    row_id = _plain_int("bank_row", bank_row, maximum=_INT32_LIMIT - 1)
    digest = hashlib.sha256()
    digest.update(b"bank-exam-question-v1\0")
    digest.update(bank_sha.encode("ascii"))
    digest.update(np.asarray([clip_id, row_id], dtype="<i8").tobytes())
    for name, value in (
        ("incoming_vel", incoming_vel),
        ("incoming_spin", incoming_spin),
        ("demanded_vel", demanded_vel),
        ("demanded_normal", demanded_normal),
    ):
        array = np.asarray(value, dtype="<f8")
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite shape-(3,) atomic vector")
        array = array.copy()
        array[array == 0.0] = 0.0
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _numpy(value) -> np.ndarray:
    """Convert numpy/torch-like CPU values without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def derive_atomic_question_ids(
    *,
    bank_sha256,
    q_counts,
    incoming_vel,
    incoming_spin,
    demanded_vel,
    demanded_normal,
) -> tuple[tuple[str, ...], ...]:
    """Derive exact production atomic IDs from padded QuestionBank/sampler arrays."""
    bank_sha = _sha256_hex("bank_sha256", bank_sha256)
    counts_array = _numpy(q_counts)
    if counts_array.ndim != 1 or counts_array.size == 0:
        raise ValueError(f"q_counts must be a non-empty one-dimensional array, got {counts_array}")
    counts = tuple(
        _plain_int(f"q_counts[{clip}]", value, minimum=1, maximum=_INT32_LIMIT - 1)
        for clip, value in enumerate(counts_array.tolist())
    )
    arrays = {}
    q_max = max(counts)
    for name, value in (
        ("incoming_vel", incoming_vel),
        ("incoming_spin", incoming_spin),
        ("demanded_vel", demanded_vel),
        ("demanded_normal", demanded_normal),
    ):
        array = _numpy(value)
        if (array.ndim != 3 or array.shape[0] != len(counts) or array.shape[1] < q_max
                or array.shape[2] != 3):
            raise ValueError(
                f"{name} must have padded shape (C,Q>=max(counts),3), got {array.shape}"
            )
        arrays[name] = array

    return tuple(
        tuple(
            atomic_bank_question_id(
                bank_sha256=bank_sha,
                clip=clip,
                bank_row=row,
                incoming_vel=arrays["incoming_vel"][clip, row],
                incoming_spin=arrays["incoming_spin"][clip, row],
                demanded_vel=arrays["demanded_vel"][clip, row],
                demanded_normal=arrays["demanded_normal"][clip, row],
            )
            for row in range(count)
        )
        for clip, count in enumerate(counts)
    )


def _require_exam_bank_metadata(metadata, *, owner: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{owner} has no schema-v3 bank metadata")
    if metadata.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError(
            f"{owner} bank schema_version={metadata.get('schema_version')!r}, "
            f"expected {BANK_SCHEMA_VERSION}"
        )
    if metadata.get("split") != "exam":
        raise ValueError(f"{owner} bank split={metadata.get('split')!r}, expected 'exam'")


def derive_question_bank_question_ids(
    question_bank, *, bank_sha256: str | None = None
) -> tuple[tuple[str, ...], ...]:
    """Derive IDs from a loaded ``stage1_question_bank.QuestionBank``.

    The bank file itself is re-hashed, so a caller cannot accidentally pair loaded arrays with the
    SHA of a different serialization.
    """
    _require_exam_bank_metadata(getattr(question_bank, "metadata", None), owner="QuestionBank")
    source_path = getattr(question_bank, "source_path", None)
    if not source_path or not os.path.isfile(source_path):
        raise ValueError(f"QuestionBank source_path is missing or unreadable: {source_path!r}")
    actual_sha = sha256_file(source_path)
    if bank_sha256 is not None and _sha256_hex("bank_sha256", bank_sha256) != actual_sha:
        raise ValueError(
            f"QuestionBank source SHA {actual_sha} does not match expected bank SHA {bank_sha256}"
        )
    return derive_atomic_question_ids(
        bank_sha256=actual_sha,
        q_counts=question_bank.counts,
        incoming_vel=question_bank.incoming_vel,
        incoming_spin=question_bank.incoming_spin,
        demanded_vel=question_bank.demanded_vel,
        demanded_normal=question_bank.demanded_normal,
    )


def derive_sampler_question_ids(
    sampler, *, allow_inexact_contract: bool = False
) -> tuple[tuple[str, ...], ...]:
    """Derive and verify IDs from an existing schema-v3 exam ``BankExamSampler``.

    The default remains formal/fail-closed.  Historical checkpoint canaries may explicitly allow
    an otherwise valid schema-v3 exam sampler whose only missing link is train/checkpoint lineage;
    those callers must also stamp the resulting evaluation contract inexact.
    """
    _require_exam_bank_metadata(getattr(sampler, "bank_meta", None), owner="BankExamSampler")
    if getattr(sampler, "contract_exact", None) is not True and not allow_inexact_contract:
        raise ValueError(
            "BankExamSampler is not exact-bound to the schema-v3 train/exam family; "
            "refusing to install a formal shared schedule (historical diagnostic canaries must "
            "opt in explicitly)"
        )
    bank_sha = _sha256_hex("sampler.bank_sha256", getattr(sampler, "bank_sha256", None))
    bank_path = getattr(sampler, "bank_path", None)
    if not bank_path or not os.path.isfile(bank_path):
        raise ValueError(f"BankExamSampler bank_path is missing or unreadable: {bank_path!r}")
    actual_sha = sha256_file(bank_path)
    if actual_sha != bank_sha:
        raise ValueError(
            f"BankExamSampler bank file SHA {actual_sha} != recorded bank SHA {bank_sha}"
        )
    return derive_atomic_question_ids(
        bank_sha256=bank_sha,
        q_counts=getattr(sampler, "q_counts", None),
        incoming_vel=getattr(sampler, "q_income", None),
        incoming_spin=getattr(sampler, "q_spin", None),
        demanded_vel=getattr(sampler, "q_vel", None),
        demanded_normal=getattr(sampler, "q_nrm", None),
    )


def _versioned_digest(domain: bytes, value: Mapping) -> bytes:
    return hashlib.sha256(domain + b"\0" + canonical_json_bytes(value)).digest()


def _row_order_key(
    *, bank_sha256: str, schedule_seed: int, clip: int, clip_name: str, row: int, question_id: str
) -> tuple[bytes, int]:
    digest = _versioned_digest(
        b"bank-exam-row-order-v3",
        {
            "bank_sha256": bank_sha256,
            "bank_row": row,
            "clip": clip,
            "clip_name": clip_name,
            "question_id": question_id,
            "schedule_seed": schedule_seed,
        },
    )
    return digest, row


def _attempt_seed(
    *, bank_sha256: str, schedule_seed: int, clip: int, row: int, question_id: str
) -> int:
    digest = _versioned_digest(
        b"bank-exam-attempt-v3",
        {
            "bank_sha256": bank_sha256,
            "bank_row": row,
            "clip": clip,
            "question_id": question_id,
            "repeat": 0,
            "schedule_seed": schedule_seed,
        },
    )
    return int.from_bytes(digest[:8], "little")


def _hold_steps(attempt_seed: int, hold_range: tuple[int, int]) -> int:
    lo, hi = hold_range
    digest = hashlib.sha256(
        b"bank-exam-hold-v3\0" + attempt_seed.to_bytes(8, "little")
    ).digest()
    return lo + int.from_bytes(digest[:8], "little") % (hi - lo + 1)


def materialize_balanced_bank_exam_schedule(
    *,
    bank_sha256,
    clip_names,
    question_ids,
    per_clip_quota,
    schedule_seed=0,
    hold_range=(0, 100),
) -> BankExamScheduleArtifact:
    """Build an exact, balanced, deterministic and finite schema-v3 paper.

    ``question_ids`` contains the unqualified 64-hex atomic IDs for every row of every clip.  The
    same scalar quota is selected without replacement from every clip.  Items are strict
    round-robin by clip, so every prefix of one full round is balanced as well.
    """
    bank_sha = _sha256_hex("bank_sha256", bank_sha256)
    names = _clip_order(clip_names)
    ids_by_clip = _normalize_question_ids(names, question_ids)
    quota = _plain_int(
        "per_clip_quota", per_clip_quota, minimum=1, maximum=_INT32_LIMIT - 1
    )
    seed = _plain_int("schedule_seed", schedule_seed, maximum=_UINT64_LIMIT - 1)
    if isinstance(hold_range, (str, bytes)) or len(hold_range) != 2:
        raise ValueError(f"hold_range must contain exactly two integers, got {hold_range!r}")
    hold = (
        _plain_int("hold_range[0]", hold_range[0], maximum=_INT32_LIMIT - 1),
        _plain_int("hold_range[1]", hold_range[1], maximum=_INT32_LIMIT - 1),
    )
    if hold[0] > hold[1]:
        raise ValueError(f"hold_range lower bound exceeds upper bound: {hold}")
    counts = tuple(len(ids) for ids in ids_by_clip)
    too_small = {name: count for name, count in zip(names, counts) if count < quota}
    if too_small:
        raise ValueError(
            f"per_clip_quota={quota} exceeds available schema-v3 exam questions {too_small}"
        )

    selected_rows = []
    for clip, (name, ids) in enumerate(zip(names, ids_by_clip)):
        rows = sorted(
            range(len(ids)),
            key=lambda row: _row_order_key(
                bank_sha256=bank_sha,
                schedule_seed=seed,
                clip=clip,
                clip_name=name,
                row=row,
                question_id=ids[row],
            ),
        )
        selected_rows.append(tuple(rows[:quota]))

    items = []
    for rank in range(quota):
        for clip, name in enumerate(names):
            row = selected_rows[clip][rank]
            qualified_id = f"{name}:{ids_by_clip[clip][row]}"
            attempt_seed = _attempt_seed(
                bank_sha256=bank_sha,
                schedule_seed=seed,
                clip=clip,
                row=row,
                question_id=qualified_id,
            )
            items.append(
                BankExamScheduleItem(
                    schedule_index=len(items),
                    clip=clip,
                    bank_row=row,
                    question_id=qualified_id,
                    hold_steps=_hold_steps(attempt_seed, hold),
                    attempt_seed=attempt_seed,
                    repeat=0,
                )
            )

    artifact = BankExamScheduleArtifact(
        schema_version=SCHEMA_VERSION,
        bank_schema_version=BANK_SCHEMA_VERSION,
        artifact_type=ARTIFACT_TYPE,
        bank_sha256=bank_sha,
        clip_order=names,
        question_counts=counts,
        per_clip_quota=quota,
        schedule_seed=seed,
        hold_range=hold,
        hold_semantics=HOLD_SEMANTICS,
        no_wrap=True,
        items=tuple(items),
        schedule_sha256="",
    )
    return BankExamScheduleArtifact(
        **{
            **artifact.__dict__,
            "schedule_sha256": canonical_schedule_sha256(artifact),
        }
    )


def _validate_artifact_self_consistency(artifact: BankExamScheduleArtifact) -> None:
    if artifact.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schedule schema_version={artifact.schema_version}, expected {SCHEMA_VERSION}"
        )
    if artifact.bank_schema_version != BANK_SCHEMA_VERSION:
        raise ValueError(
            f"schedule bank_schema_version={artifact.bank_schema_version}, "
            f"expected {BANK_SCHEMA_VERSION}"
        )
    if artifact.artifact_type != ARTIFACT_TYPE:
        raise ValueError(
            f"schedule artifact_type={artifact.artifact_type!r}, expected {ARTIFACT_TYPE!r}"
        )
    _sha256_hex("schedule bank_sha256", artifact.bank_sha256)
    names = _clip_order(artifact.clip_order)
    if len(artifact.question_counts) != len(names):
        raise ValueError("schedule question_counts length does not match clip_order")
    counts = tuple(
        _plain_int(
            f"question_counts[{clip}]", count, minimum=1, maximum=_INT32_LIMIT - 1
        )
        for clip, count in enumerate(artifact.question_counts)
    )
    quota = _plain_int(
        "per_clip_quota", artifact.per_clip_quota, minimum=1, maximum=_INT32_LIMIT - 1
    )
    if any(count < quota for count in counts):
        raise ValueError("schedule per_clip_quota exceeds its declared question_counts")
    _plain_int("schedule_seed", artifact.schedule_seed, maximum=_UINT64_LIMIT - 1)
    if len(artifact.hold_range) != 2:
        raise ValueError("schedule hold_range must contain exactly two integers")
    hold = (
        _plain_int("hold_range[0]", artifact.hold_range[0], maximum=_INT32_LIMIT - 1),
        _plain_int("hold_range[1]", artifact.hold_range[1], maximum=_INT32_LIMIT - 1),
    )
    if hold[0] > hold[1]:
        raise ValueError("schedule hold_range lower bound exceeds upper bound")
    if artifact.hold_semantics != HOLD_SEMANTICS:
        raise ValueError(
            f"schedule hold_semantics={artifact.hold_semantics!r}, "
            f"expected {HOLD_SEMANTICS!r}"
        )
    if artifact.no_wrap is not True:
        raise ValueError("schema-v3 BankExam schedule must declare no_wrap=true")
    if len(artifact.items) != len(names) * quota:
        raise ValueError(
            f"schedule has {len(artifact.items)} items, expected "
            f"{len(names)} clips * {quota} quota"
        )

    seen_rows = set()
    seen_ids = set()
    selected = [0] * len(names)
    for index, item in enumerate(artifact.items):
        if item.schedule_index != index:
            raise ValueError(
                f"schedule index/order is not contiguous at item {index}: "
                f"declared {item.schedule_index}"
            )
        expected_clip = index % len(names)
        if item.clip != expected_clip:
            raise ValueError(
                f"schedule clip order is not exact round-robin at item {index}: "
                f"clip={item.clip}, expected={expected_clip}"
            )
        if not 0 <= item.bank_row < counts[item.clip]:
            raise ValueError(
                f"schedule item {index} bank_row={item.bank_row} is outside clip "
                f"{item.clip} count={counts[item.clip]}"
            )
        prefix = f"{names[item.clip]}:"
        if not isinstance(item.question_id, str) or not item.question_id.startswith(prefix):
            raise ValueError(
                f"schedule item {index} question id is not qualified for clip {names[item.clip]!r}"
            )
        _base_question_id(
            item.question_id[len(prefix):], location=f"items[{index}].question_id"
        )
        key = (item.clip, item.bank_row)
        if key in seen_rows or item.question_id in seen_ids:
            raise ValueError(f"schedule contains a duplicate row/question at item {index}")
        seen_rows.add(key)
        seen_ids.add(item.question_id)
        selected[item.clip] += 1
        if item.repeat != 0:
            raise ValueError(f"schedule item {index} repeat must be zero, got {item.repeat}")
        expected_seed = _attempt_seed(
            bank_sha256=artifact.bank_sha256,
            schedule_seed=artifact.schedule_seed,
            clip=item.clip,
            row=item.bank_row,
            question_id=item.question_id,
        )
        if item.attempt_seed != expected_seed:
            raise ValueError(
                f"schedule item {index} attempt seed mismatch: "
                f"{item.attempt_seed} != {expected_seed}"
            )
        expected_hold = _hold_steps(expected_seed, hold)
        if item.hold_steps != expected_hold:
            raise ValueError(
                f"schedule item {index} hold mismatch: {item.hold_steps} != {expected_hold}"
            )
    if selected != [quota] * len(names):
        raise ValueError(
            f"schedule violates exact per-clip quota: selected={selected}, quota={quota}"
        )
    expected_sha = canonical_schedule_sha256(artifact)
    if artifact.schedule_sha256 != expected_sha:
        raise ValueError(
            f"schedule SHA mismatch: declared {artifact.schedule_sha256!r}, computed {expected_sha}"
        )


def validate_schedule_artifact(
    artifact: BankExamScheduleArtifact,
    *,
    expected_bank_sha256,
    expected_clip_names,
    expected_question_ids,
) -> BankExamScheduleArtifact:
    """Strictly bind a parsed artifact to a complete, independently loaded exam bank."""
    if not isinstance(artifact, BankExamScheduleArtifact):
        raise TypeError("artifact must be a BankExamScheduleArtifact")
    _validate_artifact_self_consistency(artifact)
    bank_sha = _sha256_hex("expected_bank_sha256", expected_bank_sha256)
    if artifact.bank_sha256 != bank_sha:
        raise ValueError(
            f"schedule bank SHA {artifact.bank_sha256} != expected bank SHA {bank_sha}"
        )
    names = _clip_order(expected_clip_names)
    if artifact.clip_order != names:
        raise ValueError(
            f"schedule clip order {artifact.clip_order!r} != expected clip order {names!r}"
        )
    ids = _normalize_question_ids(names, expected_question_ids)
    counts = tuple(len(values) for values in ids)
    if artifact.question_counts != counts:
        raise ValueError(
            f"schedule question counts {artifact.question_counts} != bank question counts {counts}"
        )
    for index, item in enumerate(artifact.items):
        expected_id = f"{names[item.clip]}:{ids[item.clip][item.bank_row]}"
        if item.question_id != expected_id:
            raise ValueError(
                f"schedule question id mismatch at item {index}: "
                f"{item.question_id!r} != bank {expected_id!r}"
            )

    regenerated = materialize_balanced_bank_exam_schedule(
        bank_sha256=bank_sha,
        clip_names=names,
        question_ids=ids,
        per_clip_quota=artifact.per_clip_quota,
        schedule_seed=artifact.schedule_seed,
        hold_range=artifact.hold_range,
    )
    actual_order = tuple((item.clip, item.bank_row) for item in artifact.items)
    expected_order = tuple((item.clip, item.bank_row) for item in regenerated.items)
    if actual_order != expected_order:
        raise ValueError(
            "schedule item/order does not match the deterministic schema-v3 materialization"
        )
    if artifact != regenerated:
        raise ValueError(
            "schedule contents do not match deterministic question IDs, attempt seeds, or holds"
        )
    return artifact


_DOCUMENT_KEYS = frozenset(
    {
        "artifact_type",
        "bank_schema_version",
        "bank_sha256",
        "clip_order",
        "hold_range",
        "hold_semantics",
        "items",
        "no_wrap",
        "per_clip_quota",
        "question_counts",
        "schedule_seed",
        "schedule_sha256",
        "schema_version",
    }
)
_ITEM_KEYS = frozenset(
    {
        "attempt_seed",
        "bank_row",
        "clip",
        "hold_steps",
        "question_id",
        "repeat",
        "schedule_index",
    }
)


def _exact_keys(value: Mapping, expected: frozenset[str], *, location: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{location} keys differ from schema: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _artifact_from_document(document) -> BankExamScheduleArtifact:
    if not isinstance(document, dict):
        raise ValueError("schedule artifact root must be a JSON object")
    _exact_keys(document, _DOCUMENT_KEYS, location="schedule document")
    declared_sha = _sha256_hex("schedule_sha256", document["schedule_sha256"])
    computed_sha = canonical_schedule_sha256(document)
    if declared_sha != computed_sha:
        raise ValueError(
            f"schedule SHA mismatch: declared {declared_sha}, computed {computed_sha}"
        )
    if not isinstance(document["clip_order"], list):
        raise ValueError("schedule clip_order must be a JSON array")
    if not isinstance(document["question_counts"], list):
        raise ValueError("schedule question_counts must be a JSON array")
    if not isinstance(document["hold_range"], list):
        raise ValueError("schedule hold_range must be a JSON array")
    if not isinstance(document["items"], list):
        raise ValueError("schedule items must be a JSON array")

    items = []
    for index, raw in enumerate(document["items"]):
        if not isinstance(raw, dict):
            raise ValueError(f"schedule items[{index}] must be a JSON object")
        _exact_keys(raw, _ITEM_KEYS, location=f"schedule items[{index}]")
        items.append(
            BankExamScheduleItem(
                schedule_index=_plain_int(
                    f"items[{index}].schedule_index",
                    raw["schedule_index"],
                    maximum=_INT32_LIMIT - 1,
                ),
                clip=_plain_int(
                    f"items[{index}].clip", raw["clip"], maximum=_INT32_LIMIT - 1
                ),
                bank_row=_plain_int(
                    f"items[{index}].bank_row",
                    raw["bank_row"],
                    maximum=_INT32_LIMIT - 1,
                ),
                question_id=raw["question_id"],
                hold_steps=_plain_int(
                    f"items[{index}].hold_steps",
                    raw["hold_steps"],
                    maximum=_INT32_LIMIT - 1,
                ),
                attempt_seed=_plain_int(
                    f"items[{index}].attempt_seed",
                    raw["attempt_seed"],
                    maximum=_UINT64_LIMIT - 1,
                ),
                repeat=_plain_int(
                    f"items[{index}].repeat", raw["repeat"], maximum=_INT32_LIMIT - 1
                ),
            )
        )

    artifact = BankExamScheduleArtifact(
        schema_version=_plain_int(
            "schema_version", document["schema_version"], maximum=_INT32_LIMIT - 1
        ),
        bank_schema_version=_plain_int(
            "bank_schema_version",
            document["bank_schema_version"],
            maximum=_INT32_LIMIT - 1,
        ),
        artifact_type=document["artifact_type"],
        bank_sha256=document["bank_sha256"],
        clip_order=tuple(document["clip_order"]),
        question_counts=tuple(document["question_counts"]),
        per_clip_quota=_plain_int(
            "per_clip_quota", document["per_clip_quota"], maximum=_INT32_LIMIT - 1
        ),
        schedule_seed=_plain_int(
            "schedule_seed", document["schedule_seed"], maximum=_UINT64_LIMIT - 1
        ),
        hold_range=tuple(document["hold_range"]),
        hold_semantics=document["hold_semantics"],
        no_wrap=document["no_wrap"],
        items=tuple(items),
        schedule_sha256=declared_sha,
    )
    _validate_artifact_self_consistency(artifact)
    return artifact


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in schedule artifact: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant in schedule artifact: {value}")


def write_schedule_artifact(
    artifact: BankExamScheduleArtifact, path: str | os.PathLike[str]
) -> str:
    """Atomically write canonical JSON plus one trailing newline; return the absolute path."""
    if not isinstance(artifact, BankExamScheduleArtifact):
        raise TypeError("artifact must be a BankExamScheduleArtifact")
    _validate_artifact_self_consistency(artifact)
    output = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(output)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(output)}.", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(artifact_document(artifact)))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output


def load_schedule_artifact(
    path: str | os.PathLike[str],
    *,
    expected_bank_sha256,
    expected_clip_names,
    expected_question_ids,
) -> BankExamScheduleArtifact:
    """Load a JSON paper and strictly re-derive it against an independently loaded bank."""
    with open(path, "rb") as stream:
        raw = stream.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"schedule artifact is not UTF-8: {path}") from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid schedule JSON in {path}: {exc}") from exc
    artifact = _artifact_from_document(document)
    return validate_schedule_artifact(
        artifact,
        expected_bank_sha256=expected_bank_sha256,
        expected_clip_names=expected_clip_names,
        expected_question_ids=expected_question_ids,
    )


def install_schedule_on_sampler(
    sampler,
    artifact: BankExamScheduleArtifact,
    *,
    allow_inexact_contract: bool = False,
):
    """Validate and install a shared artifact on an existing ``BankExamSampler``.

    Validation completes before any sampler field is mutated.  Installing rewinds the paper and
    clears its counters, leaving the inherited finite/no-wrap ``sample`` implementation in charge.
    """
    names = _clip_order(getattr(sampler, "clip_names", ()))
    question_ids = derive_sampler_question_ids(
        sampler, allow_inexact_contract=allow_inexact_contract
    )
    validate_schedule_artifact(
        artifact,
        expected_bank_sha256=getattr(sampler, "bank_sha256", None),
        expected_clip_names=names,
        expected_question_ids=question_ids,
    )
    if callable(getattr(sampler, "reset_counters", None)):
        sampler.reset_counters()
    sampler.schedule = artifact.items
    sampler.schedule_sha256 = artifact.schedule_sha256
    sampler.schedule_seed = artifact.schedule_seed
    sampler.schedule_k = len(artifact.items)
    sampler.schedule_hold_range = artifact.hold_range
    sampler._schedule_next = 0
    sampler.selected = [artifact.per_clip_quota] * len(names)
    sampler.asked = [0] * len(names)
    sampler.wrapped = [0] * len(names)
    sampler.schedule_artifact = artifact
    sampler.schedule_artifact_schema_version = artifact.schema_version
    return sampler
