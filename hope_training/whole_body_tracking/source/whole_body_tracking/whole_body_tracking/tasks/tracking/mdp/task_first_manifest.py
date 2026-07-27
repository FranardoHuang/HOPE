"""Strict schema-v1 contract for task-first training manifests.

The manifest binds an arbitrary ordered action bank to each action's full
task-generalization envelope and to the curriculum evidence gate.  Parsing is
fail-closed: missing fields, unknown fields, duplicate JSON keys, non-finite
numbers, ambiguous paths, and bool-as-number values are rejected.

Two hashes intentionally have different jobs:

* ``file_sha256`` binds the exact bytes supplied at process startup and is the
  value checkpoints and launch commands must pin.
* ``canonical_sha256`` is a formatting-independent content digest useful for
  review and comparison; it must not replace the byte hash at startup.

This module is dependency-light and does not read any motion asset.  Runtime
integration must independently verify each declared motion path and SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Dict, Mapping, Sequence, Tuple

if __package__:
    from .task_first_curriculum import GateConfig
else:  # pragma: no cover - exercised by host-only spec loaders
    from task_first_curriculum import GateConfig


SCHEMA_VERSION = 1
MAX_ACTION_UID = (1 << 53) - 1
MAX_HOLDOUT_SEED = (1 << 63) - 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "manifest_id",
        "training_authorized",
        "action_order",
        "actions",
        "gate",
        "holdout",
        "notes",
    )
)
_ACTION_KEYS = frozenset(
    (
        "action_id",
        "action_uid",
        "motion_path",
        "motion_sha256",
        "strike_phase",
        "family_sign",
        "mount_normal_sign",
        "position_half_extent_m",
        "speed_delta_mps",
        "face_cone_deg",
        "base_center_shift_xy_m",
        "base_half_extent_xy_m",
    )
)
_GATE_KEYS = frozenset(
    (
        "min_attempts",
        "enter_success_lower_bound",
        "exit_success_lower_bound",
        "enter_unsafe_upper_bound",
        "exit_unsafe_upper_bound",
        "enter_dwell_updates",
        "exit_dwell_updates",
        "max_stall_updates",
        "stall_policy",
        "confidence_z",
    )
)
_HOLDOUT_KEYS = frozenset(("seed", "samples_per_action", "split_id"))


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(str(key) for key in expected - actual)
        unknown = sorted(str(key) for key in actual - expected)
        raise ValueError(
            f"{name} has invalid keys (missing={missing}, unknown={unknown})"
        )
    return value


def _require_string(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    if not allow_empty and (not value or value.strip() != value):
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a plain integer")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be >= {minimum}")
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
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


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _require_vector(
    value: object,
    *,
    name: str,
    length: int,
    minimum: float | None = None,
) -> Tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return tuple(
        _require_finite(
            component,
            name=f"{name}[{index}]",
            minimum=minimum,
        )
        for index, component in enumerate(value)
    )


def _require_motion_path(value: object) -> str:
    path = _require_string(value, name="motion_path")
    if "\x00" in path:
        raise ValueError("motion_path must not contain NUL")
    if "\\" in path:
        raise ValueError("motion_path must use unambiguous POSIX separators")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError("motion_path must be relative")
    if ".." in posix.parts:
        raise ValueError("motion_path must not contain '..'")
    if not posix.parts or all(part in ("", ".") for part in posix.parts):
        raise ValueError("motion_path must identify a relative asset")
    return path


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant {value!r} is not allowed")


def _strict_json_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TaskFirstAction:
    """One action identity and its full task-centered generalization envelope."""

    action_id: str
    action_uid: int
    motion_path: str
    motion_sha256: str
    strike_phase: float
    family_sign: int
    mount_normal_sign: int
    position_half_extent_m: Tuple[float, float, float]
    speed_delta_mps: float
    face_cone_deg: float
    base_center_shift_xy_m: Tuple[float, float]
    base_half_extent_xy_m: Tuple[float, float]

    @classmethod
    def from_mapping(cls, value: object) -> "TaskFirstAction":
        """Parse one action row while rejecting every schema ambiguity."""

        row = _require_exact_keys(value, _ACTION_KEYS, name="action")
        family_sign = _require_int(
            row["family_sign"], name="family_sign", minimum=-1, maximum=1
        )
        mount_normal_sign = _require_int(
            row["mount_normal_sign"],
            name="mount_normal_sign",
            minimum=-1,
            maximum=1,
        )
        if family_sign not in (-1, 1):
            raise ValueError("family_sign must be +1 or -1")
        if mount_normal_sign not in (-1, 1):
            raise ValueError("mount_normal_sign must be +1 or -1")
        return cls(
            action_id=_require_string(row["action_id"], name="action_id"),
            action_uid=_require_int(
                row["action_uid"],
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
            motion_path=_require_motion_path(row["motion_path"]),
            motion_sha256=_require_sha256(
                row["motion_sha256"], name="motion_sha256"
            ),
            strike_phase=_require_finite(
                row["strike_phase"],
                name="strike_phase",
                minimum=0.0,
                maximum=1.0,
            ),
            family_sign=family_sign,
            mount_normal_sign=mount_normal_sign,
            position_half_extent_m=_require_vector(
                row["position_half_extent_m"],
                name="position_half_extent_m",
                length=3,
                minimum=0.0,
            ),  # type: ignore[arg-type]
            speed_delta_mps=_require_finite(
                row["speed_delta_mps"],
                name="speed_delta_mps",
                minimum=0.0,
            ),
            face_cone_deg=_require_finite(
                row["face_cone_deg"],
                name="face_cone_deg",
                minimum=0.0,
                maximum=90.0,
            ),
            base_center_shift_xy_m=_require_vector(
                row["base_center_shift_xy_m"],
                name="base_center_shift_xy_m",
                length=2,
            ),  # type: ignore[arg-type]
            base_half_extent_xy_m=_require_vector(
                row["base_half_extent_xy_m"],
                name="base_half_extent_xy_m",
                length=2,
                minimum=0.0,
            ),  # type: ignore[arg-type]
        )

    def to_mapping(self) -> Dict[str, object]:
        """Return the exact schema-v1 JSON shape for this action."""

        return {
            "action_id": self.action_id,
            "action_uid": self.action_uid,
            "motion_path": self.motion_path,
            "motion_sha256": self.motion_sha256,
            "strike_phase": self.strike_phase,
            "family_sign": self.family_sign,
            "mount_normal_sign": self.mount_normal_sign,
            "position_half_extent_m": list(self.position_half_extent_m),
            "speed_delta_mps": self.speed_delta_mps,
            "face_cone_deg": self.face_cone_deg,
            "base_center_shift_xy_m": list(self.base_center_shift_xy_m),
            "base_half_extent_xy_m": list(self.base_half_extent_xy_m),
        }


@dataclass(frozen=True)
class HoldoutConfig:
    """Deterministic held-out evaluation split shared by every action."""

    seed: int
    samples_per_action: int
    split_id: str

    @classmethod
    def from_mapping(cls, value: object) -> "HoldoutConfig":
        row = _require_exact_keys(value, _HOLDOUT_KEYS, name="holdout")
        return cls(
            seed=_require_int(
                row["seed"],
                name="holdout.seed",
                minimum=0,
                maximum=MAX_HOLDOUT_SEED,
            ),
            samples_per_action=_require_int(
                row["samples_per_action"],
                name="holdout.samples_per_action",
                minimum=1,
            ),
            split_id=_require_string(
                row["split_id"], name="holdout.split_id"
            ),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "samples_per_action": self.samples_per_action,
            "split_id": self.split_id,
        }


@dataclass(frozen=True)
class TaskFirstManifest:
    """Validated immutable schema-v1 task-first training declaration."""

    schema_version: int
    manifest_id: str
    training_authorized: bool
    action_order: Tuple[str, ...]
    actions: Tuple[TaskFirstAction, ...]
    gate: GateConfig
    holdout: HoldoutConfig
    notes: str

    @classmethod
    def from_mapping(cls, value: object) -> "TaskFirstManifest":
        """Validate and normalize a decoded JSON document."""

        document = _require_exact_keys(
            value, _TOP_LEVEL_KEYS, name="task-first manifest"
        )
        schema_version = _require_int(
            document["schema_version"],
            name="schema_version",
            minimum=SCHEMA_VERSION,
            maximum=SCHEMA_VERSION,
        )
        if type(document["training_authorized"]) is not bool:
            raise ValueError("training_authorized must be a bool")

        raw_order = document["action_order"]
        if (
            isinstance(raw_order, (str, bytes))
            or not isinstance(raw_order, Sequence)
        ):
            raise ValueError("action_order must be an array")
        action_order = tuple(
            _require_string(action_id, name=f"action_order[{index}]")
            for index, action_id in enumerate(raw_order)
        )
        if not action_order:
            raise ValueError("action_order must contain at least one action")
        if len(set(action_order)) != len(action_order):
            raise ValueError("action_order must not contain duplicate action IDs")

        raw_actions = document["actions"]
        if (
            isinstance(raw_actions, (str, bytes))
            or not isinstance(raw_actions, Sequence)
        ):
            raise ValueError("actions must be an array")
        actions = tuple(TaskFirstAction.from_mapping(row) for row in raw_actions)
        action_ids = tuple(action.action_id for action in actions)
        if action_ids != action_order:
            raise ValueError(
                "actions must have exactly the same IDs and order as action_order"
            )
        action_uids = tuple(action.action_uid for action in actions)
        if len(set(action_uids)) != len(action_uids):
            raise ValueError("actions must not contain duplicate action_uid values")

        gate_mapping = _require_exact_keys(
            document["gate"], _GATE_KEYS, name="gate"
        )
        gate = GateConfig.from_dict(gate_mapping)
        return cls(
            schema_version=schema_version,
            manifest_id=_require_string(
                document["manifest_id"], name="manifest_id"
            ),
            training_authorized=document["training_authorized"],
            action_order=action_order,
            actions=actions,
            gate=gate,
            holdout=HoldoutConfig.from_mapping(document["holdout"]),
            notes=_require_string(
                document["notes"], name="notes", allow_empty=True
            ),
        )

    def to_mapping(self) -> Dict[str, object]:
        """Return this manifest's exact normalized schema-v1 JSON shape."""

        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "training_authorized": self.training_authorized,
            "action_order": list(self.action_order),
            "actions": [action.to_mapping() for action in self.actions],
            "gate": self.gate.as_dict(),
            "holdout": self.holdout.to_mapping(),
            "notes": self.notes,
        }


def canonical_manifest_bytes(manifest: TaskFirstManifest) -> bytes:
    """Return the stable content representation used by ``canonical_sha256``."""

    if not isinstance(manifest, TaskFirstManifest):
        raise TypeError("manifest must be a TaskFirstManifest")
    return (
        json.dumps(
            manifest.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_manifest_sha256(manifest: TaskFirstManifest) -> str:
    """Return the formatting-independent digest of a validated manifest."""

    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


@dataclass(frozen=True)
class LoadedTaskFirstManifest:
    """Validated manifest plus exact-byte and canonical-content receipts."""

    manifest: TaskFirstManifest
    source_path: Path
    file_sha256: str
    canonical_sha256: str


def load_task_first_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_training_authorized: bool = False,
) -> LoadedTaskFirstManifest:
    """Read, byte-bind, and strictly validate one task-first manifest.

    ``expected_sha256`` is matched against the original file bytes before JSON
    parsing.  Set ``require_training_authorized`` at executable launch
    boundaries; review tools may leave it false to inspect preregistered but
    intentionally blocked manifests.
    """

    if type(require_training_authorized) is not bool:
        raise ValueError("require_training_authorized must be a bool")
    expected = (
        None
        if expected_sha256 is None
        else _require_sha256(expected_sha256, name="expected_sha256")
    )
    source_path = Path(path)
    raw = source_path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected is not None and file_sha256 != expected:
        raise ValueError(
            "task-first manifest file SHA-256 mismatch: "
            f"expected {expected}, got {file_sha256}"
        )

    try:
        document = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except UnicodeDecodeError as error:
        raise ValueError("task-first manifest must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError("task-first manifest is not valid JSON") from error

    manifest = TaskFirstManifest.from_mapping(document)
    if require_training_authorized and not manifest.training_authorized:
        raise ValueError(
            "task-first manifest is metadata-only: training_authorized is false"
        )
    return LoadedTaskFirstManifest(
        manifest=manifest,
        source_path=source_path,
        file_sha256=file_sha256,
        canonical_sha256=canonical_manifest_sha256(manifest),
    )
