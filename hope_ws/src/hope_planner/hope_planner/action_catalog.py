"""Strict, content-bound action identities for planner/deploy catalogs.

``action_uid`` is derived from canonical UTF-8 JSON containing only
``action_id``, ``family`` and ``content_sha256``.  A local dense ``slot`` is
deliberately not part of that identity, so a valid catalog reorder preserves
every UID while reassigning slots and recomputing ``catalog_sha256``.

The 53-bit-positive UID range is exactly representable by a float64 wire.  A
cryptographic truncation collision is extremely unlikely, but it is not
silently resolved: duplicate UIDs in one catalog are a hard validation error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple


ACTION_CATALOG_SCHEMA_V1 = 1
MAX_ACTION_UID = (1 << 53) - 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = frozenset(
    ("action_id", "action_uid", "slot", "family", "content_sha256")
)
_IDENTITY_KEYS = frozenset(("action_id", "family", "content_sha256"))
_CATALOG_KEYS = frozenset(("schema_version", "actions", "catalog_sha256"))


def _canonical_bytes(value: object) -> bytes:
    """Return the one canonical JSON encoding used by all hashes in this module."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_keys(
    value: object, expected: frozenset, *, name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(name))
    actual = set(value.keys())
    if actual != set(expected):
        missing = sorted(str(key) for key in expected - actual)
        unknown = sorted(str(key) for key in actual - expected)
        raise ValueError(
            "{} has invalid keys (missing={}, unknown={})".format(
                name, missing, unknown
            )
        )
    return value


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("{} must be a non-empty trimmed string".format(name))
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("{} must be exactly 64 lowercase hexadecimal characters".format(name))
    return value


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int = None,
) -> int:
    # ``bool`` and integral floats are intentionally rejected.  JSON numbers
    # must be decoded as Python ints to enter this identity/control contract.
    if type(value) is not int:
        raise ValueError("{} must be an integer (not bool or float)".format(name))
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError("{} must be >= {}".format(name, minimum))
        raise ValueError("{} must be in [{},{}]".format(name, minimum, maximum))
    return value


def derive_action_uid(
    action_id: str, family: str, content_sha256: str
) -> int:
    """Derive a stable positive, float64-exact UID from action content identity.

    The full SHA-256 digest of the canonical identity payload is interpreted as
    one unsigned big-endian integer, reduced into ``[1, MAX_ACTION_UID]``.
    ``slot`` is intentionally absent.
    """

    action_id = _require_string(action_id, name="action_id")
    family = _require_string(family, name="family")
    content_sha256 = _require_sha256(content_sha256, name="content_sha256")
    identity = {
        "action_id": action_id,
        "content_sha256": content_sha256,
        "family": family,
    }
    digest = hashlib.sha256(_canonical_bytes(identity)).digest()
    return 1 + (int.from_bytes(digest, byteorder="big") % MAX_ACTION_UID)


@dataclass(frozen=True)
class ActionRecord:
    """One action's stable identity plus its catalog-local dense slot."""

    action_id: str
    action_uid: int
    slot: int
    family: str
    content_sha256: str

    def __post_init__(self) -> None:
        action_id = _require_string(self.action_id, name="action_id")
        family = _require_string(self.family, name="family")
        content_sha256 = _require_sha256(
            self.content_sha256, name="content_sha256"
        )
        action_uid = _require_int(
            self.action_uid,
            name="action_uid",
            minimum=1,
            maximum=MAX_ACTION_UID,
        )
        _require_int(self.slot, name="slot", minimum=0)
        expected_uid = derive_action_uid(action_id, family, content_sha256)
        if action_uid != expected_uid:
            raise ValueError(
                "action_uid {} does not match canonical action identity (expected {})".format(
                    action_uid, expected_uid
                )
            )

    @classmethod
    def build(
        cls,
        *,
        action_id: str,
        slot: int,
        family: str,
        content_sha256: str
    ) -> "ActionRecord":
        """Create a record while deriving its content-bound UID."""

        return cls(
            action_id=action_id,
            action_uid=derive_action_uid(action_id, family, content_sha256),
            slot=slot,
            family=family,
            content_sha256=content_sha256,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "ActionRecord":
        """Parse one full serialized row, rejecting missing or unknown keys."""

        row = _require_exact_keys(value, _RECORD_KEYS, name="action row")
        return cls(
            action_id=row["action_id"],  # type: ignore[arg-type]
            action_uid=row["action_uid"],  # type: ignore[arg-type]
            slot=row["slot"],  # type: ignore[arg-type]
            family=row["family"],  # type: ignore[arg-type]
            content_sha256=row["content_sha256"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> Dict[str, object]:
        """Return this record in its canonical schema-v1 mapping shape."""

        return {
            "action_id": self.action_id,
            "action_uid": self.action_uid,
            "slot": self.slot,
            "family": self.family,
            "content_sha256": self.content_sha256,
        }

    def with_slot(self, slot: int) -> "ActionRecord":
        """Return the same stable action identity assigned to another local slot."""

        return ActionRecord(
            action_id=self.action_id,
            action_uid=self.action_uid,
            slot=slot,
            family=self.family,
            content_sha256=self.content_sha256,
        )


def _validate_dense_records(records: Tuple[ActionRecord, ...]) -> None:
    if not records:
        raise ValueError("action catalog must contain at least one action")
    for index, record in enumerate(records):
        if not isinstance(record, ActionRecord):
            raise ValueError("actions[{}] must be an ActionRecord".format(index))

    ids = [record.action_id for record in records]
    uids = [record.action_uid for record in records]
    slots = [record.slot for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("action catalog contains duplicate action_id values")
    if len(set(uids)) != len(uids):
        raise ValueError("action catalog contains duplicate action_uid values")
    if len(set(slots)) != len(slots):
        raise ValueError("action catalog contains duplicate slot values")
    expected_slots = list(range(len(records)))
    if slots != expected_slots:
        raise ValueError(
            "action rows must be ordered by dense slots 0..N-1; got {}".format(slots)
        )


def _catalog_payload(records: Tuple[ActionRecord, ...]) -> Dict[str, object]:
    """Build the canonical catalog payload without its self hash."""

    return {
        "schema_version": ACTION_CATALOG_SCHEMA_V1,
        "actions": [record.to_mapping() for record in records],
    }


def _catalog_sha256(records: Tuple[ActionRecord, ...]) -> str:
    return hashlib.sha256(_canonical_bytes(_catalog_payload(records))).hexdigest()


@dataclass(frozen=True)
class ActionCatalog:
    """Schema-v1 ordered action catalog with strict content/self validation."""

    schema_version: int
    actions: Tuple[ActionRecord, ...]
    catalog_sha256: str

    def __post_init__(self) -> None:
        schema_version = _require_int(
            self.schema_version, name="schema_version", minimum=1
        )
        if schema_version != ACTION_CATALOG_SCHEMA_V1:
            raise ValueError(
                "unsupported action catalog schema_version {}".format(schema_version)
            )
        if type(self.actions) is not tuple:
            raise ValueError("actions must be a tuple; use create/build/from_mapping")
        _validate_dense_records(self.actions)
        catalog_sha256 = _require_sha256(
            self.catalog_sha256, name="catalog_sha256"
        )
        expected_sha256 = _catalog_sha256(self.actions)
        if catalog_sha256 != expected_sha256:
            raise ValueError(
                "catalog_sha256 does not match canonical payload (expected {})".format(
                    expected_sha256
                )
            )

    @classmethod
    def create(cls, actions: Sequence[ActionRecord]) -> "ActionCatalog":
        """Create from already-slotted records.

        Records must arrive in exact dense slot order ``0..N-1``.  This method
        never repairs, sorts or silently deduplicates caller input.
        """

        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
            raise ValueError("actions must be a sequence of ActionRecord values")
        records = tuple(actions)
        _validate_dense_records(records)
        return cls(
            schema_version=ACTION_CATALOG_SCHEMA_V1,
            actions=records,
            catalog_sha256=_catalog_sha256(records),
        )

    @classmethod
    def build(cls, rows: Sequence[Mapping[str, object]]) -> "ActionCatalog":
        """Build from ordered identity rows, deriving UID and dense slot.

        Each input row has exactly ``action_id``, ``family`` and
        ``content_sha256``.  Its position becomes its local slot.
        """

        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("rows must be a sequence of identity mappings")
        records: List[ActionRecord] = []
        for slot, value in enumerate(rows):
            row = _require_exact_keys(
                value, _IDENTITY_KEYS, name="identity row {}".format(slot)
            )
            records.append(
                ActionRecord.build(
                    action_id=row["action_id"],  # type: ignore[arg-type]
                    slot=slot,
                    family=row["family"],  # type: ignore[arg-type]
                    content_sha256=row["content_sha256"],  # type: ignore[arg-type]
                )
            )
        return cls.create(records)

    @classmethod
    def from_mapping(cls, value: object) -> "ActionCatalog":
        """Parse and fully authenticate one serialized schema-v1 catalog."""

        mapping = _require_exact_keys(value, _CATALOG_KEYS, name="action catalog")
        schema_version = _require_int(
            mapping["schema_version"], name="schema_version", minimum=1
        )
        if schema_version != ACTION_CATALOG_SCHEMA_V1:
            raise ValueError(
                "unsupported action catalog schema_version {}".format(schema_version)
            )
        raw_actions = mapping["actions"]
        if type(raw_actions) is not list:
            raise ValueError("actions must be a JSON-style list")
        records = tuple(
            ActionRecord.from_mapping(row) for row in raw_actions
        )
        _validate_dense_records(records)
        catalog_sha256 = _require_sha256(
            mapping["catalog_sha256"], name="catalog_sha256"
        )
        expected_sha256 = _catalog_sha256(records)
        if catalog_sha256 != expected_sha256:
            raise ValueError(
                "catalog_sha256 does not match canonical payload (expected {})".format(
                    expected_sha256
                )
            )
        return cls(
            schema_version=schema_version,
            actions=records,
            catalog_sha256=catalog_sha256,
        )

    def to_mapping(self) -> Dict[str, object]:
        """Return a detached JSON-compatible mapping including the self hash."""

        payload = _catalog_payload(self.actions)
        payload["catalog_sha256"] = self.catalog_sha256
        return payload

    def reorder(self, action_ids: Sequence[str]) -> "ActionCatalog":
        """Reorder by exact action ID, rebuilding dense slots and catalog hash."""

        if isinstance(action_ids, (str, bytes)) or not isinstance(
            action_ids, Sequence
        ):
            raise ValueError("action_ids must be a sequence of strings")
        requested = tuple(action_ids)
        for index, action_id in enumerate(requested):
            _require_string(action_id, name="action_ids[{}]".format(index))
        if len(requested) != len(self.actions):
            raise ValueError(
                "reorder must name every action exactly once (expected {}, got {})".format(
                    len(self.actions), len(requested)
                )
            )
        if len(set(requested)) != len(requested):
            raise ValueError("reorder contains duplicate action_id values")
        known = set(self.action_ids)
        supplied = set(requested)
        if supplied != known:
            missing = sorted(known - supplied)
            unknown = sorted(supplied - known)
            raise KeyError(
                "reorder action IDs differ from catalog (missing={}, unknown={})".format(
                    missing, unknown
                )
            )
        return ActionCatalog.create(
            [self.by_id(action_id).with_slot(slot)
             for slot, action_id in enumerate(requested)]
        )

    def by_id(self, action_id: str) -> ActionRecord:
        action_id = _require_string(action_id, name="action_id")
        for record in self.actions:
            if record.action_id == action_id:
                return record
        raise KeyError("unknown action_id {!r}".format(action_id))

    def by_uid(self, action_uid: int) -> ActionRecord:
        action_uid = _require_int(
            action_uid, name="action_uid", minimum=1, maximum=MAX_ACTION_UID
        )
        for record in self.actions:
            if record.action_uid == action_uid:
                return record
        raise KeyError("unknown action_uid {!r}".format(action_uid))

    def by_slot(self, slot: int) -> ActionRecord:
        slot = _require_int(slot, name="slot", minimum=0)
        if slot >= len(self.actions):
            raise KeyError("unknown slot {!r}".format(slot))
        # Dense-slot validation makes this indexing identity exact.
        return self.actions[slot]

    @property
    def action_ids(self) -> Tuple[str, ...]:
        return tuple(record.action_id for record in self.actions)

    @property
    def action_uids(self) -> Tuple[int, ...]:
        return tuple(record.action_uid for record in self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[ActionRecord]:
        return iter(self.actions)

    def __getitem__(self, slot: int) -> ActionRecord:
        return self.by_slot(slot)


__all__ = [
    "ACTION_CATALOG_SCHEMA_V1",
    "MAX_ACTION_UID",
    "ActionCatalog",
    "ActionRecord",
    "derive_action_uid",
]
