"""Dependency-light runtime contracts for action-conditioned ball-first training.

This module is deliberately independent of Isaac, Torch, NumPy, and the
question solver.  It owns two narrow seams:

* a true-reset birth broker.  Motion selects an action, reserves an immutable
  base-birth receipt, writes the canonical root from that receipt, commits the
  reservation, and Racket consumes it exactly once;
* a lazy per-action solved-task pool.  Only the action requested by the
  runtime is refilled, so a 93-action registry does not allocate 93 dense
  buffers up front.

The provider and solver callbacks receive only immutable frozen contract
objects whose fields are JSON-safe data (the dataclass wrapper itself is an
in-process seam, not a wire encoding).  No simulator object or cached root
pose is exposed here.  In
particular, episode birth must never be derived from ``base_pos_w`` or another
pre-reset simulation cache.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import threading
from typing import Callable, Dict, Mapping, Protocol, Sequence, Tuple
import weakref

try:
    from . import racket_contact_geometry as _contact_geometry
except ImportError:
    # Several dependency-light contract tests deliberately load this file by
    # exact path, outside its package, so importing the sibling explicitly is
    # the only faithful standalone fallback.  Production package imports take
    # the normal relative path above.
    _geometry_path = Path(__file__).with_name(
        "racket_contact_geometry.py"
    )
    _geometry_spec = importlib.util.spec_from_file_location(
        "_action_ball_racket_contact_geometry", _geometry_path
    )
    if _geometry_spec is None or _geometry_spec.loader is None:
        raise
    _contact_geometry = importlib.util.module_from_spec(_geometry_spec)
    sys.modules[_geometry_spec.name] = _contact_geometry
    _geometry_spec.loader.exec_module(_contact_geometry)


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

SCHEMA_VERSION = 3
TASK_RECEIPT_SCHEMA_VERSION = 5
TASK_RECEIPT_TIMING_AUTHORITY = (
    f"per_swing_task_receipt_v{TASK_RECEIPT_SCHEMA_VERSION}"
    "_exact_face_contact"
)
BROKER_STATE_SCHEMA_VERSION = 4
POOL_STATE_SCHEMA_VERSION = 3
MAX_ACTION_UID = (1 << 53) - 1
MAX_COUNTER = (1 << 63) - 1
SAMPLER_BIRTH_DRAW_COUNT = 3
SAMPLER_SAMPLE_DRAW_COUNT = 18
SAMPLER_SCHEMA_VERSION = 3
UNIT_VECTOR_TOLERANCE = 1.0e-6
MAX_PRE_SWING_WAIT_S = 1.0

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
ARM_CATALOG_SHA256 = (
    "2cbc6673119e0a816b0ee5081b403e5f4598437e4e8bf2eaa1e8a3db88f91d1b"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_DESCRIPTION = {
    "schema_version": SCHEMA_VERSION,
    "task_receipt_schema_version": TASK_RECEIPT_SCHEMA_VERSION,
    "sampler_schema_version": SAMPLER_SCHEMA_VERSION,
    "arm_catalog_sha256": ARM_CATALOG_SHA256,
    "birth": (
        "env/reset_generation/action_uid/action_slot/domain_epoch/mode/"
        "levels/sampler_birth/draw_range/base_spawn/base_yaw and exact "
        "runtime pins"
    ),
    "task": (
        "birth/sample/ball-centre/aim plus exact_face_contact_v2 selected-"
        "face full orientation, site target, face/site velocities, coupled "
        "angular teacher retiming proof/residual and the same runtime pins"
    ),
    "racket_contact_geometry_source_sha256": (
        _contact_geometry.GEOMETRY_SOURCE_SHA256
    ),
    "broker": (
        "reserve true reset -> pure full-catalog domain/provider tape "
        "authorities -> exact provider birth authority -> commit root write "
        "-> consume once; retain contiguous full consumed history"
    ),
    "pool": (
        "lazy per-action/per-birth FIFO with pure exact sampler and solved-"
        "task authorities, per-action sample/task high-water authorities, "
        "append-only active/retired birth lifecycle transcripts, atomic "
        "vectorized refill/retire, and exact JSON state"
    ),
    "state_schemas": {
        "broker": BROKER_STATE_SCHEMA_VERSION,
        "pool": POOL_STATE_SCHEMA_VERSION,
    },
    "sampler": {
        "birth_draws": SAMPLER_BIRTH_DRAW_COUNT,
        "sample_draws": SAMPLER_SAMPLE_DRAW_COUNT,
        "arm_catalog_sha256": ARM_CATALOG_SHA256,
    },
    "teacher_timing": (
        "required=norm(racket_site_velocity); "
        "rate=required/reference_site_speed after exact face-centre to site "
        "omega-cross-r coupling; "
        "scaled_hit=reference_hit/rate; scaled_cycle=reference_cycle/rate; "
        "wait=TTC-scaled_hit; no clipping; certified rate bounds and "
        "reaction_margin<=wait<=1s"
    ),
    "resume_threat_model": (
        "internal assignment/lifecycle/transcript authorities reject partial "
        "drift, rollback, and wrong-birth replay; formal resume additionally "
        "requires an independently pre-pinned raw checkpoint/shared-state "
        "root and does not claim resistance to coordinated component re-sign"
    ),
}


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class _WeakIdentityCachedCanonicalSha256:
    """Cache a frozen receipt digest without mutating the receipt.

    ``functools.cached_property`` writes into the instance ``__dict__`` on
    first access.  That would make ``vars()``, pickle, deepcopy, and any
    future exact-resume fingerprint depend on access history.  This
    descriptor instead keys a weak external cache by object identity.  The
    id check protects against reuse, and the weakref callback drops entries
    as soon as the immutable receipt leaves its owner's lifecycle.
    """

    def __init__(self, function: Callable[[object], str]) -> None:
        self._function = function
        self.__doc__ = function.__doc__
        self._entries: Dict[int, Tuple[object, str]] = {}
        self._lock = threading.RLock()

    def __get__(self, instance: object, owner: object = None) -> object:
        if instance is None:
            return self
        key = id(instance)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached[0]() is instance:
                return cached[1]
            digest = self._function(instance)
            descriptor_ref = weakref.ref(self)

            def discard(
                receipt_ref: object,
                *,
                identity: int = key,
                owner_ref: object = descriptor_ref,
            ) -> None:
                descriptor = owner_ref()
                if descriptor is None:
                    return
                with descriptor._lock:
                    current = descriptor._entries.get(identity)
                    if (
                        current is not None
                        and current[0] is receipt_ref
                    ):
                        descriptor._entries.pop(identity, None)

            receipt_ref = weakref.ref(instance, discard)
            self._entries[key] = (receipt_ref, digest)
            return digest

    def __set__(self, instance: object, value: object) -> None:
        # Match the former read-only ``property`` data-descriptor contract.  In particular,
        # ``object.__setattr__`` and an accidental same-name instance attribute must never shadow
        # the canonical digest lookup.
        raise AttributeError("canonical_sha256 is read-only")

    def __delete__(self, instance: object) -> None:
        raise AttributeError("canonical_sha256 is read-only")


# ``canonical_sha256`` is cached only on deeply immutable frozen dataclasses.
# The descriptor above is deliberately external to dataclass state, equality,
# repr, deepcopy, pickle, and the explicit JSON wire payloads.  A contract
# test rejects adding a mutable child to any cached class.
TASK_TRANSCRIPT_SCHEMA_VERSION = 1
TASK_LIFECYCLE_SCHEMA_VERSION = 1
_LIFECYCLE_REJECTED = 0
_LIFECYCLE_PENDING = 1
_LIFECYCLE_ISSUED = 2
_LIFECYCLE_DISCARDED = 3


def extend_task_transcript_sha256(
    prior_sha256: str, task_sha256: str
) -> str:
    """Append one canonical task SHA to an existing transcript root."""

    return _sha256_json(
        {
            "schema_version": TASK_TRANSCRIPT_SCHEMA_VERSION,
            "prior_sha256": _sha256(
                prior_sha256, name="task transcript prior_sha256"
            ),
            "task_sha256": _sha256(
                task_sha256, name="task transcript task_sha256"
            ),
        }
    )


_task_transcript_extend = extend_task_transcript_sha256


def task_transcript_sha256(
    birth_sha256: str, ordered_task_sha256: Sequence[str]
) -> str:
    """Return the canonical append-only chain root for one birth's tasks."""

    root = _sha256_json(
        {
            "schema_version": TASK_TRANSCRIPT_SCHEMA_VERSION,
            "birth_sha256": _sha256(
                birth_sha256, name="task transcript birth_sha256"
            ),
        }
    )
    if isinstance(ordered_task_sha256, (str, bytes)) or not isinstance(
        ordered_task_sha256, Sequence
    ):
        raise ActionBallContractError(
            "ordered_task_sha256 must be a sequence"
        )
    for digest in ordered_task_sha256:
        root = extend_task_transcript_sha256(root, digest)
    return root


def _pack_lifecycle_2bit(statuses: Sequence[int]) -> str:
    packed = bytearray((len(statuses) + 3) // 4)
    for index, status in enumerate(statuses):
        if type(status) is not int or not 0 <= status <= 3:
            raise ActionBallContractError(
                "task lifecycle status must be a 2-bit integer"
            )
        packed[index // 4] |= status << (2 * (index % 4))
    return base64.b64encode(bytes(packed)).decode("ascii")


def _unpack_lifecycle_2bit(value: object, *, count: int) -> list[int]:
    if type(value) is not str:
        raise ActionBallContractError(
            "task lifecycle bitmap must be base64 text"
        )
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ActionBallContractError(
            "task lifecycle bitmap is not canonical base64"
        ) from exc
    if len(raw) != (count + 3) // 4:
        raise ActionBallContractError(
            "task lifecycle bitmap length disagrees with sample count"
        )
    statuses = [
        (raw[index // 4] >> (2 * (index % 4))) & 0b11
        for index in range(count)
    ]
    if count % 4 and raw:
        unused_mask = ~((1 << (2 * (count % 4))) - 1) & 0xFF
        if raw[-1] & unused_mask:
            raise ActionBallContractError(
                "task lifecycle bitmap has non-zero padding bits"
            )
    if _pack_lifecycle_2bit(statuses) != value:
        raise ActionBallContractError(
            "task lifecycle bitmap is not canonical"
        )
    return statuses


def _task_lifecycle_sha256(
    action_uid: int, statuses: Sequence[int]
) -> str:
    return _sha256_json(
        {
            "schema_version": TASK_LIFECYCLE_SCHEMA_VERSION,
            "action_uid": action_uid,
            "sample_count": len(statuses),
            "lifecycle_2bit_base64": _pack_lifecycle_2bit(statuses),
        }
    )


def _json_data(value: object, *, name: str) -> object:
    """Return a detached JSON-safe copy or fail on opaque/non-finite state."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ActionBallContractError(
            f"{name} must contain only JSON-safe finite data"
        ) from exc


RUNTIME_CONTRACT_SHA256 = _sha256_json(_CONTRACT_DESCRIPTION)

if ARM_CATALOG_SHA256 != _sha256_json(
    {
        "schema_version": SAMPLER_SCHEMA_VERSION,
        "arm_keys": list(ARM_KEYS),
    }
):
    raise RuntimeError("action-ball arm catalog constant drifted")


class ActionBallContractError(ValueError):
    """A receipt, binding, pin, or serialized state violates the contract."""


class CounterRallyTaskIdentityError(ActionBallContractError):
    """Counter-rally task identity drift; never a difficulty failure."""


class BirthProtocolError(RuntimeError):
    """The true-reset reserve/commit/consume protocol was used incorrectly."""


class PoolProtocolError(RuntimeError):
    """The lazy solver pool could not safely fulfill a request."""


_DOMAIN_LEVEL_NAMES = ARM_KEYS


def _plain_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = MAX_COUNTER,
) -> int:
    if type(value) is not int:
        raise ActionBallContractError(f"{name} must be a plain integer")
    if value < minimum or value > maximum:
        raise ActionBallContractError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise ActionBallContractError(
            f"{name} must be a plain finite number"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ActionBallContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ActionBallContractError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ActionBallContractError(f"{name} must be <= {maximum}")
    return result


def _vec(
    value: object,
    *,
    name: str,
    length: int,
) -> Tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (tuple, list))
        or len(value) != length
    ):
        raise ActionBallContractError(
            f"{name} must be a length-{length} tuple/list"
        )
    return tuple(
        _finite(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )


def _vec2(value: object, *, name: str) -> Vec2:
    return _vec(value, name=name, length=2)  # type: ignore[return-value]


def _vec3(value: object, *, name: str) -> Vec3:
    return _vec(value, name=name, length=3)  # type: ignore[return-value]


def _unit_vec3(value: object, *, name: str) -> Vec3:
    result = _vec3(value, name=name)
    norm = math.sqrt(sum(component * component for component in result))
    if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
        raise ActionBallContractError(
            f"{name} must already be unit length within "
            f"{UNIT_VECTOR_TOLERANCE}; got norm {norm}"
        )
    return result


def _unit_quat_wxyz(value: object, *, name: str) -> Tuple[float, float, float, float]:
    result = _vec(value, name=name, length=4)
    norm = math.sqrt(sum(component * component for component in result))
    if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
        raise ActionBallContractError(
            f"{name} must already be unit length within "
            f"{UNIT_VECTOR_TOLERANCE}; got norm {norm}"
        )
    return result  # type: ignore[return-value]


def _rotate_yaw(value: Vec3, yaw_rad: float) -> Vec3:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * value[0] - sine * value[1],
        sine * value[0] + cosine * value[1],
        value[2],
    )


def _vec3_add(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _vec3_scale(value: Vec3, scale: float) -> Vec3:
    return (value[0] * scale, value[1] * scale, value[2] * scale)


def _vec3_close(left: Vec3, right: Vec3, *, tolerance: float = 1.0e-9) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def _assert_yaw_quaternion(
    yaw_rad: float,
    quat_wxyz: Tuple[float, float, float, float],
    *,
    name: str,
) -> None:
    expected = (
        math.cos(0.5 * yaw_rad),
        0.0,
        0.0,
        math.sin(0.5 * yaw_rad),
    )
    direct = max(abs(a - b) for a, b in zip(quat_wxyz, expected))
    negated = max(abs(a + b) for a, b in zip(quat_wxyz, expected))
    if min(direct, negated) > UNIT_VECTOR_TOLERANCE:
        raise ActionBallContractError(
            f"{name} must be the yaw-only wxyz quaternion for base_yaw_rad"
        )


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ActionBallContractError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _identity(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ActionBallContractError(
            f"{name} must be a non-empty trimmed identity string"
        )
    return value


def _relative_path(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if "\\" in result or "\x00" in result:
        raise ActionBallContractError(
            f"{name} must be a normalized relative POSIX path"
        )
    posix = PurePosixPath(result)
    windows = PureWindowsPath(result)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or posix.as_posix() != result
    ):
        raise ActionBallContractError(
            f"{name} must be a normalized relative POSIX path without '..'"
        )
    return result


def _mode(value: object, *, name: str = "mobility_mode") -> str:
    if value not in ("no_move", "move"):
        raise ActionBallContractError(
            f"{name} must be exactly 'no_move' or 'move'"
        )
    return str(value)


def _exact_mapping(
    value: object,
    expected: Sequence[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ActionBallContractError(f"{name} must be a mapping")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ActionBallContractError(
            f"{name} has invalid keys "
            f"(missing={sorted(wanted - actual)}, "
            f"unknown={sorted(actual - wanted)})"
        )
    return value


def _true_reset(reset_kind: object) -> None:
    if reset_kind != "true_reset":
        raise BirthProtocolError(
            "birth receipts exist only at reset_kind='true_reset'; "
            "wrap/midswing paths must not reserve, commit, or consume one"
        )


COUNTER_RALLY_TASK_IDENTITY_SCHEMA_VERSION = 1
_COUNTER_RALLY_TASK_IDENTITY_PAYLOAD_KEYS = (
    "schema_version",
    "objective_profile_sha256",
    "return_direction_env_xy",
    "target_baseline_speed_mps",
)


@dataclass(frozen=True)
class CounterRallyTaskIdentity:
    """Exact N=1 objective values bound inside one solved task receipt."""

    objective_profile_sha256: str
    return_direction_env_xy: Vec2
    target_baseline_speed_mps: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_profile_sha256",
            _sha256(
                self.objective_profile_sha256,
                name="counter_rally_task.objective_profile_sha256",
            ),
        )
        direction = _vec2(
            self.return_direction_env_xy,
            name="counter_rally_task.return_direction_env_xy",
        )
        norm = math.hypot(direction[0], direction[1])
        if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
            raise ActionBallContractError(
                "counter_rally_task.return_direction_env_xy must already "
                f"be unit length within {UNIT_VECTOR_TOLERANCE}; "
                f"got norm {norm}"
            )
        if direction[0] <= 0.0:
            raise ActionBallContractError(
                "counter_rally_task.return_direction_env_xy must be "
                "opponent-bound"
            )
        object.__setattr__(
            self, "return_direction_env_xy", direction
        )
        target_speed = _finite(
            self.target_baseline_speed_mps,
            name="counter_rally_task.target_baseline_speed_mps",
            minimum=0.0,
        )
        if target_speed <= 0.0:
            raise ActionBallContractError(
                "counter_rally_task.target_baseline_speed_mps must be > 0"
            )
        object.__setattr__(
            self, "target_baseline_speed_mps", target_speed
        )

    def payload_dict(self) -> Dict[str, object]:
        return {
            "schema_version": COUNTER_RALLY_TASK_IDENTITY_SCHEMA_VERSION,
            "objective_profile_sha256": self.objective_profile_sha256,
            "return_direction_env_xy": list(
                self.return_direction_env_xy
            ),
            "target_baseline_speed_mps": (
                self.target_baseline_speed_mps
            ),
        }

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.payload_dict())

    def to_dict(self) -> Dict[str, object]:
        result = self.payload_dict()
        result["canonical_sha256"] = self.canonical_sha256
        return result

    @classmethod
    def from_dict(cls, value: object) -> "CounterRallyTaskIdentity":
        row = _exact_mapping(
            value,
            (
                *_COUNTER_RALLY_TASK_IDENTITY_PAYLOAD_KEYS,
                "canonical_sha256",
            ),
            name="counter-rally task identity",
        )
        if (
            row["schema_version"]
            != COUNTER_RALLY_TASK_IDENTITY_SCHEMA_VERSION
        ):
            raise ActionBallContractError(
                "unsupported counter-rally task identity schema_version"
            )
        identity = cls(
            objective_profile_sha256=row["objective_profile_sha256"],
            return_direction_env_xy=row["return_direction_env_xy"],
            target_baseline_speed_mps=row[
                "target_baseline_speed_mps"
            ],
        )
        declared = _sha256(
            row["canonical_sha256"],
            name="counter_rally_task.canonical_sha256",
        )
        if declared != identity.canonical_sha256:
            raise CounterRallyTaskIdentityError(
                "counter-rally task identity canonical SHA mismatch"
            )
        return identity


@dataclass(frozen=True)
class ActionDomainLevels:
    """Frozen normalized widths for the 32 asymmetric curriculum arms."""

    time_to_contact_lower: float = 0.0
    time_to_contact_upper: float = 0.0
    contact_x_lower: float = 0.0
    contact_x_upper: float = 0.0
    contact_y_lower: float = 0.0
    contact_y_upper: float = 0.0
    contact_z_lower: float = 0.0
    contact_z_upper: float = 0.0
    incoming_speed_lower: float = 0.0
    incoming_speed_upper: float = 0.0
    spin_magnitude_lower: float = 0.0
    spin_magnitude_upper: float = 0.0
    base_spawn_x_lower: float = 0.0
    base_spawn_x_upper: float = 0.0
    base_spawn_y_lower: float = 0.0
    base_spawn_y_upper: float = 0.0
    base_travel_x_lower: float = 0.0
    base_travel_x_upper: float = 0.0
    base_travel_y_lower: float = 0.0
    base_travel_y_upper: float = 0.0
    landing_aim_x_lower: float = 0.0
    landing_aim_x_upper: float = 0.0
    landing_aim_y_lower: float = 0.0
    landing_aim_y_upper: float = 0.0
    incoming_direction_u_neg: float = 0.0
    incoming_direction_u_pos: float = 0.0
    incoming_direction_v_neg: float = 0.0
    incoming_direction_v_pos: float = 0.0
    spin_direction_u_neg: float = 0.0
    spin_direction_u_pos: float = 0.0
    spin_direction_v_neg: float = 0.0
    spin_direction_v_pos: float = 0.0

    def __post_init__(self) -> None:
        for name in _DOMAIN_LEVEL_NAMES:
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=f"domain_levels.{name}",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            name: getattr(self, name) for name in _DOMAIN_LEVEL_NAMES
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActionDomainLevels":
        row = _exact_mapping(
            value, _DOMAIN_LEVEL_NAMES, name="action domain levels"
        )
        return cls(
            **{name: row[name] for name in _DOMAIN_LEVEL_NAMES}
        )  # type: ignore[arg-type]

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class ActionSamplingMixture:
    """Runtime copy of the sampler's exact deterministic quota receipt."""

    center_slots: int
    interior_slots: int
    frontier_slots: int
    interior_level_scale: float
    frontier_band_fraction: float
    schedule: Tuple[str, ...]

    def __post_init__(self) -> None:
        weights = []
        for name in (
            "center_slots",
            "interior_slots",
            "frontier_slots",
        ):
            value = _plain_int(
                getattr(self, name),
                name=f"sampling_mixture.{name}",
                minimum=1,
            )
            object.__setattr__(self, name, value)
            weights.append(value)
        interior = _finite(
            self.interior_level_scale,
            name="sampling_mixture.interior_level_scale",
            minimum=0.0,
            maximum=1.0,
        )
        band = _finite(
            self.frontier_band_fraction,
            name="sampling_mixture.frontier_band_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < band < 1.0:
            raise ActionBallContractError(
                "sampling mixture frontier band must lie in (0, 1)"
            )
        if interior > 1.0 - band + 1.0e-15:
            raise ActionBallContractError(
                "sampling mixture interior overlaps frontier"
            )
        object.__setattr__(self, "interior_level_scale", interior)
        object.__setattr__(self, "frontier_band_fraction", band)
        total = sum(weights)
        current = [0, 0, 0]
        names = ("center", "interior", "frontier")
        expected = []
        for _ in range(total):
            for index, weight in enumerate(weights):
                current[index] += weight
            chosen = max(
                range(len(names)),
                key=lambda index: (current[index], -index),
            )
            expected.append(names[chosen])
            current[chosen] -= total
        schedule = tuple(self.schedule)
        if schedule != tuple(expected):
            raise ActionBallContractError(
                "sampling mixture schedule differs from exact quota"
            )
        object.__setattr__(self, "schedule", schedule)

    def to_dict(self) -> Dict[str, object]:
        return {
            "center_slots": self.center_slots,
            "interior_slots": self.interior_slots,
            "frontier_slots": self.frontier_slots,
            "interior_level_scale": self.interior_level_scale,
            "frontier_band_fraction": self.frontier_band_fraction,
            "schedule": list(self.schedule),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActionSamplingMixture":
        row = _exact_mapping(
            value,
            (
                "center_slots",
                "interior_slots",
                "frontier_slots",
                "interior_level_scale",
                "frontier_band_fraction",
                "schedule",
            ),
            name="action sampling mixture",
        )
        raw_schedule = row["schedule"]
        if not isinstance(raw_schedule, (tuple, list)):
            raise ActionBallContractError(
                "sampling mixture schedule must be a sequence"
            )
        return cls(
            center_slots=row["center_slots"],
            interior_slots=row["interior_slots"],
            frontier_slots=row["frontier_slots"],
            interior_level_scale=row["interior_level_scale"],
            frontier_band_fraction=row["frontier_band_fraction"],
            schedule=tuple(raw_schedule),
        )


FROZEN_ATTEMPT_SCHEMA_VERSION = 4
FROZEN_TERMINAL_OUTCOMES = (
    "legal_return",
    "safe_nonreturn",
    "table_hit",
    "fall",
    "collision",
    "joint_qdes_limit",
    "joint_actual_limit",
)


@dataclass(frozen=True)
class FrozenEvaluationProposalRequest:
    """Authority-owned allocation for one frozen-policy proposal.

    This is an in-process request, not an authorization capability.  The
    evaluator allocates ``policy_generation``, ``seed``, ``sample_index`` and
    ``birth_index``; callers never submit those values to the evaluator.
    """

    reservation_sha256: str
    policy_checkpoint_sha256: str
    policy_generation: int
    window_sha256: str
    evidence_role: str
    proposal_offset: int
    seed: int
    sample_index: int
    birth_index: int
    action_uid: int
    profile_sha256: str
    mobility_mode: str
    domain_epoch: int
    domain_levels: ActionDomainLevels
    selected_arm_key: str

    def __post_init__(self) -> None:
        for name in (
            "reservation_sha256",
            "policy_checkpoint_sha256",
            "window_sha256",
            "profile_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        for name, minimum, maximum in (
            ("policy_generation", 1, MAX_COUNTER),
            ("proposal_offset", 0, MAX_COUNTER),
            ("seed", 0, MAX_COUNTER),
            ("sample_index", 0, MAX_COUNTER),
            ("birth_index", 0, MAX_COUNTER),
            ("action_uid", 1, MAX_ACTION_UID),
            ("domain_epoch", 0, MAX_COUNTER),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if self.evidence_role not in (
            "scheduler",
            "frozen_canary",
            "frozen_heldout",
        ):
            raise ActionBallContractError(
                "frozen proposal evidence_role must be scheduler, "
                "frozen_canary, or frozen_heldout"
            )
        object.__setattr__(
            self,
            "mobility_mode",
            _mode(self.mobility_mode),
        )
        if not isinstance(self.domain_levels, ActionDomainLevels):
            raise ActionBallContractError(
                "frozen proposal domain_levels must be ActionDomainLevels"
            )
        if self.selected_arm_key and self.selected_arm_key not in ARM_KEYS:
            raise ActionBallContractError(
                "frozen proposal selected_arm_key is outside ARM_KEYS"
            )
        expected = _sha256_json(self.identity_payload())
        if self.reservation_sha256 != expected:
            raise ActionBallContractError(
                "frozen proposal reservation SHA does not match its "
                "authority allocation"
            )

    def identity_payload(self) -> Dict[str, object]:
        return {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_evaluation_proposal_reservation",
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "policy_generation": self.policy_generation,
            "window_sha256": self.window_sha256,
            "evidence_role": self.evidence_role,
            "proposal_offset": self.proposal_offset,
            "seed": self.seed,
            "sample_index": self.sample_index,
            "birth_index": self.birth_index,
            "action_uid": self.action_uid,
            "profile_sha256": self.profile_sha256,
            "mobility_mode": self.mobility_mode,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.to_dict(),
            "selected_arm_key": self.selected_arm_key,
        }

    @classmethod
    def create(cls, **kwargs: object) -> "FrozenEvaluationProposalRequest":
        raw_levels = kwargs.get("domain_levels")
        if not isinstance(raw_levels, ActionDomainLevels):
            raise ActionBallContractError(
                "frozen proposal domain_levels must be ActionDomainLevels"
            )
        normalized = dict(kwargs)
        normalized["domain_levels"] = raw_levels.to_dict()
        payload = {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_evaluation_proposal_reservation",
            **normalized,
        }
        return cls(
            reservation_sha256=_sha256_json(payload),
            **kwargs,
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FrozenIssuedProposal:
    """Exact sampler/birth identity returned by the pinned attempt source."""

    reservation_sha256: str
    source_contract_sha256: str
    source_receipt_sha256: str
    sample_receipt_sha256: str
    birth_receipt_sha256: str
    action_uid: int
    profile_sha256: str
    mobility_mode: str
    domain_epoch: int
    levels_sha256: str
    sample_index: int
    birth_index: int
    sampling_stratum: str
    frontier_arm: str

    def __post_init__(self) -> None:
        for name in (
            "reservation_sha256",
            "source_contract_sha256",
            "source_receipt_sha256",
            "sample_receipt_sha256",
            "birth_receipt_sha256",
            "profile_sha256",
            "levels_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        for name, minimum, maximum in (
            ("action_uid", 1, MAX_ACTION_UID),
            ("domain_epoch", 0, MAX_COUNTER),
            ("sample_index", 0, MAX_COUNTER),
            ("birth_index", 0, MAX_COUNTER),
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        object.__setattr__(
            self,
            "mobility_mode",
            _mode(self.mobility_mode),
        )
        if self.sampling_stratum not in (
            "center",
            "interior",
            "frontier",
        ):
            raise ActionBallContractError(
                "frozen proposal sampling_stratum is invalid"
            )
        if self.frontier_arm:
            if (
                self.sampling_stratum != "frontier"
                or self.frontier_arm not in ARM_KEYS
            ):
                raise ActionBallContractError(
                    "frozen proposal frontier arm/stratum mismatch"
                )
        elif self.sampling_stratum == "frontier":
            raise ActionBallContractError(
                "frontier proposal must name its signed arm"
            )
        expected = _sha256_json(self.receipt_payload())
        if self.source_receipt_sha256 != expected:
            raise ActionBallContractError(
                "frozen issued-proposal receipt SHA mismatch"
            )

    def receipt_payload(self) -> Dict[str, object]:
        return {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_issued_proposal",
            "reservation_sha256": self.reservation_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "sample_receipt_sha256": self.sample_receipt_sha256,
            "birth_receipt_sha256": self.birth_receipt_sha256,
            "action_uid": self.action_uid,
            "profile_sha256": self.profile_sha256,
            "mobility_mode": self.mobility_mode,
            "domain_epoch": self.domain_epoch,
            "levels_sha256": self.levels_sha256,
            "sample_index": self.sample_index,
            "birth_index": self.birth_index,
            "sampling_stratum": self.sampling_stratum,
            "frontier_arm": self.frontier_arm,
        }

    @classmethod
    def create(cls, **kwargs: object) -> "FrozenIssuedProposal":
        payload = {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_issued_proposal",
            **kwargs,
        }
        return cls(
            source_receipt_sha256=_sha256_json(payload),
            **kwargs,
        )  # type: ignore[arg-type]

    def assert_request(
        self,
        request: FrozenEvaluationProposalRequest,
    ) -> None:
        if not isinstance(request, FrozenEvaluationProposalRequest):
            raise ActionBallContractError(
                "issued proposal request must be frozen authority data"
            )
        expected = (
            request.reservation_sha256,
            request.action_uid,
            request.profile_sha256,
            request.mobility_mode,
            request.domain_epoch,
            request.domain_levels.canonical_sha256,
            request.sample_index,
            request.birth_index,
        )
        actual = (
            self.reservation_sha256,
            self.action_uid,
            self.profile_sha256,
            self.mobility_mode,
            self.domain_epoch,
            self.levels_sha256,
            self.sample_index,
            self.birth_index,
        )
        if actual != expected:
            raise ActionBallContractError(
                "issued proposal differs from its authority reservation"
            )


@dataclass(frozen=True)
class FrozenSolverEvent:
    """Pinned solver disposition; rejection keeps an explicit reason."""

    proposal_receipt_sha256: str
    source_contract_sha256: str
    event_receipt_sha256: str
    disposition: str
    reject_reason: str
    task_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "proposal_receipt_sha256",
            "source_contract_sha256",
            "event_receipt_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        if self.disposition not in ("rejected", "admitted"):
            raise ActionBallContractError(
                "solver disposition must be rejected or admitted"
            )
        if self.disposition == "rejected":
            object.__setattr__(
                self,
                "reject_reason",
                _identity(self.reject_reason, name="reject_reason"),
            )
            if self.task_receipt_sha256:
                raise ActionBallContractError(
                    "rejected solver event cannot name a task receipt"
                )
        else:
            if self.reject_reason:
                raise ActionBallContractError(
                    "admitted solver event cannot name a reject reason"
                )
            object.__setattr__(
                self,
                "task_receipt_sha256",
                _sha256(
                    self.task_receipt_sha256,
                    name="task_receipt_sha256",
                ),
            )
        expected = _sha256_json(self.event_payload())
        if self.event_receipt_sha256 != expected:
            raise ActionBallContractError(
                "frozen solver event receipt SHA mismatch"
            )

    def event_payload(self) -> Dict[str, object]:
        return {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_solver_event",
            "proposal_receipt_sha256": self.proposal_receipt_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "disposition": self.disposition,
            "reject_reason": self.reject_reason,
            "task_receipt_sha256": self.task_receipt_sha256,
        }

    @classmethod
    def create(cls, **kwargs: object) -> "FrozenSolverEvent":
        payload = {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_solver_event",
            **kwargs,
        }
        return cls(
            event_receipt_sha256=_sha256_json(payload),
            **kwargs,
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FrozenLifecycleEvent:
    """Exact install/start event emitted from the in-process runtime seam."""

    proposal_receipt_sha256: str
    task_receipt_sha256: str
    source_contract_sha256: str
    stage: str
    event_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "proposal_receipt_sha256",
            "task_receipt_sha256",
            "source_contract_sha256",
            "event_receipt_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        if self.stage not in ("installed", "started"):
            raise ActionBallContractError(
                "frozen lifecycle event stage is invalid"
            )
        expected = _sha256_json(self.event_payload())
        if self.event_receipt_sha256 != expected:
            raise ActionBallContractError(
                "frozen lifecycle event receipt SHA mismatch"
            )

    def event_payload(self) -> Dict[str, object]:
        return {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_lifecycle_event",
            "proposal_receipt_sha256": self.proposal_receipt_sha256,
            "task_receipt_sha256": self.task_receipt_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "stage": self.stage,
        }

    @classmethod
    def create(cls, **kwargs: object) -> "FrozenLifecycleEvent":
        payload = {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_lifecycle_event",
            **kwargs,
        }
        return cls(
            event_receipt_sha256=_sha256_json(payload),
            **kwargs,
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FrozenTerminalSignals:
    """Raw trusted sensor facts; the outcome is always code-classified."""

    infrastructure_invalid: bool = False
    joint_actual_limit: bool = False
    joint_qdes_limit: bool = False
    fall: bool = False
    table_hit: bool = False
    collision: bool = False
    legal_return: bool = False

    def __post_init__(self) -> None:
        for name in (
            "infrastructure_invalid",
            "joint_actual_limit",
            "joint_qdes_limit",
            "fall",
            "table_hit",
            "collision",
            "legal_return",
        ):
            if type(getattr(self, name)) is not bool:
                raise ActionBallContractError(
                    f"frozen terminal signal {name} must be bool"
                )

    def to_dict(self) -> Dict[str, bool]:
        return {
            name: getattr(self, name)
            for name in (
                "infrastructure_invalid",
                "joint_actual_limit",
                "joint_qdes_limit",
                "fall",
                "table_hit",
                "collision",
                "legal_return",
            )
        }


def classify_frozen_terminal(
    signals: FrozenTerminalSignals,
) -> str | None:
    """Apply the single canonical terminal precedence.

    Infrastructure invalidity is ``X`` and therefore returns ``None``: it
    burns the reservation but is not a policy closure.  Hard actual-joint
    limit comes before commanded-limit saturation, then fall/table/collision,
    legal return, and finally a safe non-return.
    """

    if not isinstance(signals, FrozenTerminalSignals):
        raise ActionBallContractError(
            "terminal classifier needs FrozenTerminalSignals"
        )
    if signals.infrastructure_invalid:
        return None
    if signals.joint_actual_limit:
        return "joint_actual_limit"
    if signals.joint_qdes_limit:
        return "joint_qdes_limit"
    if signals.fall:
        return "fall"
    if signals.table_hit:
        return "table_hit"
    if signals.collision:
        return "collision"
    if signals.legal_return:
        return "legal_return"
    return "safe_nonreturn"


@dataclass(frozen=True)
class FrozenTerminalEvent:
    """Pinned sensor receipt.  It never carries a caller-selected outcome."""

    proposal_receipt_sha256: str
    task_receipt_sha256: str
    source_contract_sha256: str
    signals: FrozenTerminalSignals
    event_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "proposal_receipt_sha256",
            "task_receipt_sha256",
            "source_contract_sha256",
            "event_receipt_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        if not isinstance(self.signals, FrozenTerminalSignals):
            raise ActionBallContractError(
                "terminal event signals must be FrozenTerminalSignals"
            )
        expected = _sha256_json(self.event_payload())
        if self.event_receipt_sha256 != expected:
            raise ActionBallContractError(
                "frozen terminal event receipt SHA mismatch"
            )

    def event_payload(self) -> Dict[str, object]:
        return {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_terminal_event",
            "proposal_receipt_sha256": self.proposal_receipt_sha256,
            "task_receipt_sha256": self.task_receipt_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "signals": self.signals.to_dict(),
        }

    @property
    def terminal_outcome(self) -> str | None:
        return classify_frozen_terminal(self.signals)

    @classmethod
    def create(cls, **kwargs: object) -> "FrozenTerminalEvent":
        signals = kwargs.get("signals")
        if not isinstance(signals, FrozenTerminalSignals):
            raise ActionBallContractError(
                "terminal event signals must be FrozenTerminalSignals"
            )
        payload = {
            "schema_version": FROZEN_ATTEMPT_SCHEMA_VERSION,
            "kind": "frozen_terminal_event",
            **kwargs,
            "signals": signals.to_dict(),
        }
        return cls(
            event_receipt_sha256=_sha256_json(payload),
            **kwargs,
        )  # type: ignore[arg-type]


class FrozenAttemptSource(Protocol):
    """Same-process, code-pinned source of sampler/solver/runtime events.

    Opaque Python capabilities deliberately do not cross process boundaries.
    A cross-process evaluator must instead introduce a separately reviewed
    authenticated transport; JSON rehydration is not equivalent authority.
    """

    source_contract_sha256: str
    source_code_sha256: str
    source_path: str
    state_owner_sha256: str

    def state_dict(self) -> Mapping[str, object]:
        ...

    def load_state_dict(self, state: object) -> None:
        ...

    def issue_proposal(
        self,
        request: FrozenEvaluationProposalRequest,
    ) -> FrozenIssuedProposal:
        ...

    def assert_exact_proposal(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
    ) -> None:
        ...

    def solver_event(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
    ) -> FrozenSolverEvent:
        ...

    def assert_solver_event(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
        event: FrozenSolverEvent,
    ) -> None:
        ...

    def lifecycle_event(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
        solver: FrozenSolverEvent,
        stage: str,
    ) -> FrozenLifecycleEvent:
        ...

    def assert_lifecycle_event(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
        solver: FrozenSolverEvent,
        event: FrozenLifecycleEvent,
    ) -> None:
        ...

    def terminal_event(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
        solver: FrozenSolverEvent,
    ) -> FrozenTerminalEvent:
        ...

    def assert_terminal_event(
        self,
        request: FrozenEvaluationProposalRequest,
        proposal: FrozenIssuedProposal,
        solver: FrozenSolverEvent,
        event: FrozenTerminalEvent,
    ) -> None:
        ...


@dataclass(frozen=True)
class ActionDomainClaim:
    """One authority-minted frozen domain snapshot for an action reset."""

    authority_contract_sha256: str
    arm_catalog_sha256: str
    action_uid: int
    domain_epoch: int
    domain_levels: ActionDomainLevels
    levels_sha256: str
    profile_sha256: str
    mobility_mode: str

    def __post_init__(self) -> None:
        for name in (
            "authority_contract_sha256",
            "arm_catalog_sha256",
            "levels_sha256",
            "profile_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "domain_epoch",
            _plain_int(self.domain_epoch, name="domain_epoch"),
        )
        if not isinstance(self.domain_levels, ActionDomainLevels):
            raise ActionBallContractError(
                "domain claim levels must be ActionDomainLevels"
            )
        if self.levels_sha256 != self.domain_levels.canonical_sha256:
            raise ActionBallContractError(
                "domain claim levels SHA does not match its frozen payload"
            )
        if self.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ActionBallContractError(
                "domain claim arm catalog SHA mismatch"
            )
        object.__setattr__(
            self, "mobility_mode", _mode(self.mobility_mode)
        )

    def payload_dict(self) -> Dict[str, object]:
        return {
            "authority_contract_sha256": self.authority_contract_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.to_dict(),
            "levels_sha256": self.levels_sha256,
            "profile_sha256": self.profile_sha256,
            "mobility_mode": self.mobility_mode,
        }

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.payload_dict())


@dataclass(frozen=True)
class RuntimePins:
    """Run-global exact identities shared by every action receipt."""

    manifest_sha256: str
    sampler_sha256: str
    domain_authority_sha256: str
    physics_sha256: str
    solver_sha256: str
    counter_rally_objective_profile_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "sampler_sha256",
            "domain_authority_sha256",
            "physics_sha256",
            "solver_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        if self.counter_rally_objective_profile_sha256 is not None:
            object.__setattr__(
                self,
                "counter_rally_objective_profile_sha256",
                _sha256(
                    self.counter_rally_objective_profile_sha256,
                    name=(
                        "counter_rally_objective_profile_sha256"
                    ),
                ),
            )

    def to_dict(self) -> Dict[str, str]:
        result = {
            "manifest_sha256": self.manifest_sha256,
            "sampler_sha256": self.sampler_sha256,
            "domain_authority_sha256": self.domain_authority_sha256,
            "physics_sha256": self.physics_sha256,
            "solver_sha256": self.solver_sha256,
        }
        if self.counter_rally_objective_profile_sha256 is not None:
            result["counter_rally_objective_profile_sha256"] = (
                self.counter_rally_objective_profile_sha256
            )
        return result

    @classmethod
    def from_dict(cls, value: object) -> "RuntimePins":
        has_counter_rally_objective = (
            isinstance(value, Mapping)
            and "counter_rally_objective_profile_sha256" in value
        )
        row = _exact_mapping(
            value,
            (
                "manifest_sha256",
                "sampler_sha256",
                "domain_authority_sha256",
                "physics_sha256",
                "solver_sha256",
                *(
                    ("counter_rally_objective_profile_sha256",)
                    if has_counter_rally_objective
                    else ()
                ),
            ),
            name="runtime pins",
        )
        return cls(**{name: row[name] for name in row})  # type: ignore[arg-type]

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class ActionBinding:
    """One manifest-ordered action and its exact motion/profile identities."""

    action_uid: int
    action_slot: int
    motion_path: str
    motion_sha256: str
    profile_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )
        object.__setattr__(
            self,
            "motion_path",
            _relative_path(self.motion_path, name="motion_path"),
        )
        object.__setattr__(
            self,
            "motion_sha256",
            _sha256(self.motion_sha256, name="motion_sha256"),
        )
        object.__setattr__(
            self,
            "profile_sha256",
            _sha256(self.profile_sha256, name="profile_sha256"),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_uid": self.action_uid,
            "action_slot": self.action_slot,
            "motion_path": self.motion_path,
            "motion_sha256": self.motion_sha256,
            "profile_sha256": self.profile_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActionBinding":
        row = _exact_mapping(
            value,
            (
                "action_uid",
                "action_slot",
                "motion_path",
                "motion_sha256",
                "profile_sha256",
            ),
            name="action binding",
        )
        return cls(**{name: row[name] for name in row})  # type: ignore[arg-type]


def _validate_bindings(
    bindings: Sequence[ActionBinding],
) -> Tuple[ActionBinding, ...]:
    if isinstance(bindings, (str, bytes)) or not isinstance(
        bindings, Sequence
    ):
        raise ActionBallContractError("bindings must be a non-empty sequence")
    converted = tuple(bindings)
    if not converted:
        raise ActionBallContractError("bindings must be non-empty")
    if any(not isinstance(binding, ActionBinding) for binding in converted):
        raise ActionBallContractError(
            "every binding must be an ActionBinding"
        )
    ordered = tuple(sorted(converted, key=lambda binding: binding.action_slot))
    slots = tuple(binding.action_slot for binding in ordered)
    if slots != tuple(range(len(ordered))):
        raise ActionBallContractError(
            "action slots must be unique and contiguous in [0, N)"
        )
    uids = tuple(binding.action_uid for binding in ordered)
    if len(set(uids)) != len(uids):
        raise ActionBallContractError("action_uid values must be unique")
    return ordered


def _registry_sha256(
    bindings: Sequence[ActionBinding],
    pins: RuntimePins,
    mobility_mode: str,
) -> str:
    return _sha256_json(
        {
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "pins": pins.to_dict(),
            "mobility_mode": mobility_mode,
            "bindings": [binding.to_dict() for binding in bindings],
        }
    )


@dataclass(frozen=True)
class ActionBirthRequest:
    """Pure provider input; deliberately contains no environment/sim object."""

    env_id: int
    reset_generation: int
    action_uid: int
    action_slot: int
    domain_claim: ActionDomainClaim
    registry_sha256: str
    mobility_mode: str
    binding: ActionBinding
    pins: RuntimePins

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )
        if not isinstance(self.domain_claim, ActionDomainClaim):
            raise ActionBallContractError(
                "birth request requires an ActionDomainClaim"
            )
        object.__setattr__(
            self,
            "registry_sha256",
            _sha256(self.registry_sha256, name="registry_sha256"),
        )
        object.__setattr__(
            self,
            "mobility_mode",
            _mode(self.mobility_mode),
        )
        if not isinstance(self.binding, ActionBinding):
            raise ActionBallContractError(
                "birth request binding must be ActionBinding"
            )
        if not isinstance(self.pins, RuntimePins):
            raise ActionBallContractError(
                "birth request pins must be RuntimePins"
            )
        if (
            self.binding.action_uid != self.action_uid
            or self.binding.action_slot != self.action_slot
            or self.domain_claim.action_uid != self.action_uid
            or self.domain_claim.profile_sha256
            != self.binding.profile_sha256
            or self.domain_claim.mobility_mode != self.mobility_mode
            or self.domain_claim.authority_contract_sha256
            != self.pins.domain_authority_sha256
        ):
            raise ActionBallContractError(
                "birth request action/domain claim does not match binding "
                "or run pins"
            )


class ActionDomainClaimAuthority(Protocol):
    domain_authority_contract_sha256: str
    state_owner_sha256: str

    def claim_for_action(self, action_uid: int) -> ActionDomainClaim:
        """Mint the next frozen reset-barrier domain claim."""

    def state_dict(self) -> Mapping[str, object]:
        """Return all domain-schedule state as JSON data."""

    def load_state_dict(self, state: object) -> None:
        """Atomically restore the exact domain-schedule state."""

    def domain_cursor_for(self, action_uid: int) -> int:
        """Return the exact next-claim cursor for one action."""


class ActionBirthProvider(Protocol):
    sampler_contract_sha256: str
    state_owner_sha256: str

    def __call__(self, request: ActionBirthRequest) -> "ActionBirthReceipt":
        """Create a receipt from pure contract data, never a sim cache."""

    def state_dict(self) -> Mapping[str, object]:
        """Return every behaviorally relevant provider counter as JSON data."""

    def load_state_dict(self, state: object) -> None:
        """Atomically restore an exact provider state."""

    def assert_issued_birth(self, receipt: "ActionBirthReceipt") -> None:
        """Fail unless this exact runtime birth belongs to provider state."""

    def birth_highwater_for(
        self, action_uid: int
    ) -> Tuple[int, int]:
        """Return exact ``(last_birth_index, last_birth_draw_end)``."""


_BIRTH_PAYLOAD_KEYS = (
    "schema_version",
    "runtime_contract_sha256",
    "registry_sha256",
    "env_id",
    "reset_generation",
    "action_uid",
    "action_slot",
    "domain_epoch",
    "domain_claim_sha256",
    "domain_authority_sha256",
    "domain_levels",
    "arm_catalog_sha256",
    "levels_sha256",
    "sampler_birth_sha256",
    "sampler_birth_index",
    "sampler_draw_start",
    "sampler_draw_end",
    "mobility_mode",
    "base_yaw_rad",
    "base_quat_wxyz",
    "base_spawn_w_m",
    "manifest_sha256",
    "sampler_sha256",
    "profile_sha256",
    "motion_sha256",
    "physics_sha256",
    "solver_sha256",
)
_MIXTURE_BIRTH_PAYLOAD_KEYS = (
    *_BIRTH_PAYLOAD_KEYS[:13],
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    *_BIRTH_PAYLOAD_KEYS[13:],
)
# The initial-center collapse is written down only when the sampler actually
# ran under that law, so every receipt produced before this field existed keeps
# its exact bytes and canonical SHA.  ``False`` therefore means "absent", which
# is also what every legacy row on disk asserts.
_INITIAL_CENTER_MIXTURE_BIRTH_PAYLOAD_KEYS = (
    *_BIRTH_PAYLOAD_KEYS[:13],
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "initial_center_single_question",
    *_BIRTH_PAYLOAD_KEYS[13:],
)


@dataclass(frozen=True)
class ActionBirthReceipt:
    """Immutable true-reset root/base contract for one env generation."""

    env_id: int
    reset_generation: int
    action_uid: int
    action_slot: int
    domain_epoch: int
    domain_claim_sha256: str
    domain_authority_sha256: str
    domain_levels: ActionDomainLevels
    arm_catalog_sha256: str
    levels_sha256: str
    sampler_birth_sha256: str
    sampler_birth_index: int
    sampler_draw_start: int
    sampler_draw_end: int
    mobility_mode: str
    base_yaw_rad: float
    base_quat_wxyz: Tuple[float, float, float, float]
    base_spawn_w_m: Vec3
    manifest_sha256: str
    sampler_sha256: str
    profile_sha256: str
    motion_sha256: str
    physics_sha256: str
    solver_sha256: str
    registry_sha256: str
    sampling_mixture: ActionSamplingMixture | None = None
    sampling_stratum: str = "domain"
    sampling_levels: ActionDomainLevels | None = None
    frontier_arm: str | None = None
    initial_center_single_question: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )
        object.__setattr__(
            self,
            "domain_epoch",
            _plain_int(self.domain_epoch, name="domain_epoch"),
        )
        for name in (
            "domain_claim_sha256",
            "domain_authority_sha256",
            "arm_catalog_sha256",
            "levels_sha256",
            "sampler_birth_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        if not isinstance(self.domain_levels, ActionDomainLevels):
            raise ActionBallContractError(
                "birth receipt domain_levels must be ActionDomainLevels"
            )
        if self.levels_sha256 != self.domain_levels.canonical_sha256:
            raise ActionBallContractError(
                "birth receipt levels SHA does not match its frozen payload"
            )
        if self.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ActionBallContractError(
                "birth receipt arm catalog SHA mismatch"
            )
        if type(self.initial_center_single_question) is not bool:
            raise ActionBallContractError(
                "birth initial_center_single_question must be a plain bool"
            )
        if self.sampling_mixture is None:
            if (
                self.sampling_stratum != "domain"
                or self.frontier_arm is not None
                or self.initial_center_single_question
            ):
                raise ActionBallContractError(
                    "legacy birth cannot carry mixture sampling metadata"
                )
            object.__setattr__(
                self,
                "sampling_levels",
                ActionDomainLevels(),
            )
        else:
            if not isinstance(
                self.sampling_mixture, ActionSamplingMixture
            ):
                raise ActionBallContractError(
                    "birth sampling_mixture has invalid type"
                )
            if self.sampling_stratum not in (
                "center",
                "interior",
                "frontier",
            ):
                raise ActionBallContractError(
                    "birth sampling_stratum is invalid"
                )
            if not isinstance(
                self.sampling_levels, ActionDomainLevels
            ):
                raise ActionBallContractError(
                    "birth sampling_levels has invalid type"
                )
            expected_stratum = self.sampling_mixture.schedule[
                self.sampler_birth_index
                % len(self.sampling_mixture.schedule)
            ]
            inactive_birth_frontier = (
                expected_stratum == "frontier"
                and self.sampling_stratum == "center"
                and all(
                    getattr(self.domain_levels, arm) == 0.0
                    and getattr(self.sampling_levels, arm) == 0.0
                    for arm in (
                        "base_spawn_x_lower",
                        "base_spawn_x_upper",
                        "base_spawn_y_lower",
                        "base_spawn_y_upper",
                    )
                )
            )
            # ``ActionBallSampler._literal_initial_center_active`` collapses the
            # whole plan to the literal centre point while the sampler runs in
            # initial-center single-question mode and all 32 curriculum arms are
            # exactly zero.  Under that law the quota slot no longer selects the
            # stratum, so comparing against the schedule would reject the only
            # plan the sampler is allowed to emit.  The gate stays fail-closed:
            # it now asserts the *unique* legal row instead of the quota slot,
            # and rejects any non-centre row the collapse cannot have produced.
            initial_center_collapse = (
                self.initial_center_single_question
                and all(
                    getattr(self.domain_levels, arm) == 0.0
                    for arm in ARM_KEYS
                )
            )
            if initial_center_collapse:
                if (
                    self.sampling_stratum != "center"
                    or self.frontier_arm is not None
                    or any(
                        getattr(self.sampling_levels, arm) != 0.0
                        for arm in ARM_KEYS
                    )
                ):
                    raise ActionBallContractError(
                        "birth initial-center collapse is not the literal "
                        "all-zero center row"
                    )
            elif (
                self.sampling_stratum != expected_stratum
                and not inactive_birth_frontier
            ):
                raise ActionBallContractError(
                    "birth sampling stratum differs from quota schedule"
                )
            if (self.sampling_stratum == "frontier") != (
                self.frontier_arm is not None
            ):
                raise ActionBallContractError(
                    "birth frontier arm must exist exactly at frontier"
                )
            if self.frontier_arm is not None and self.frontier_arm not in (
                "base_spawn_x_lower",
                "base_spawn_x_upper",
                "base_spawn_y_lower",
                "base_spawn_y_upper",
            ):
                raise ActionBallContractError(
                    "birth frontier arm is not a base-spawn arm"
                )
            for arm in ARM_KEYS:
                if getattr(self.sampling_levels, arm) > (
                    getattr(self.domain_levels, arm) + 1.0e-15
                ):
                    raise ActionBallContractError(
                        "birth sampling levels exceed frozen domain"
                    )
        object.__setattr__(
            self,
            "sampler_birth_index",
            _plain_int(
                self.sampler_birth_index,
                name="sampler_birth_index",
            ),
        )
        object.__setattr__(
            self,
            "sampler_draw_start",
            _plain_int(
                self.sampler_draw_start,
                name="sampler_draw_start",
            ),
        )
        object.__setattr__(
            self,
            "sampler_draw_end",
            _plain_int(
                self.sampler_draw_end,
                name="sampler_draw_end",
                minimum=1,
            ),
        )
        if (
            self.sampler_draw_end - self.sampler_draw_start
            != SAMPLER_BIRTH_DRAW_COUNT
        ):
            raise ActionBallContractError(
                "sampler birth draw range must consume exactly "
                f"{SAMPLER_BIRTH_DRAW_COUNT} draws"
            )
        object.__setattr__(
            self, "mobility_mode", _mode(self.mobility_mode)
        )
        object.__setattr__(
            self,
            "base_yaw_rad",
            _finite(self.base_yaw_rad, name="base_yaw_rad"),
        )
        object.__setattr__(
            self,
            "base_quat_wxyz",
            _unit_quat_wxyz(
                self.base_quat_wxyz, name="base_quat_wxyz"
            ),
        )
        _assert_yaw_quaternion(
            self.base_yaw_rad,
            self.base_quat_wxyz,
            name="base_quat_wxyz",
        )
        object.__setattr__(
            self,
            "base_spawn_w_m",
            _vec3(self.base_spawn_w_m, name="base_spawn_w_m"),
        )
        for name in (
            "registry_sha256",
            "manifest_sha256",
            "sampler_sha256",
            "profile_sha256",
            "motion_sha256",
            "physics_sha256",
            "solver_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        domain_claim = ActionDomainClaim(
            authority_contract_sha256=self.domain_authority_sha256,
            arm_catalog_sha256=self.arm_catalog_sha256,
            action_uid=self.action_uid,
            domain_epoch=self.domain_epoch,
            domain_levels=self.domain_levels,
            levels_sha256=self.levels_sha256,
            profile_sha256=self.profile_sha256,
            mobility_mode=self.mobility_mode,
        )
        if domain_claim.canonical_sha256 != self.domain_claim_sha256:
            raise ActionBallContractError(
                "birth receipt domain claim SHA does not match its fields"
            )
        sampler_birth_identity = {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "kind": "base_birth",
            "sampler_contract_sha256": self.sampler_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "levels_sha256": self.levels_sha256,
            "profile_sha256": self.profile_sha256,
            "birth_index": self.sampler_birth_index,
            "draw_start": self.sampler_draw_start,
            "draw_end": self.sampler_draw_end,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_start_w_m": self.base_spawn_w_m,
        }
        if self.sampling_mixture is not None:
            sampler_birth_identity.update(
                {
                    "sampling_mixture": (
                        self.sampling_mixture.to_dict()
                    ),
                    "sampling_stratum": self.sampling_stratum,
                    "sampling_levels": self.sampling_levels.to_dict(),
                    "frontier_arm": self.frontier_arm,
                }
            )
        if _sha256_json(sampler_birth_identity) != self.sampler_birth_sha256:
            raise ActionBallContractError(
                "sampler birth SHA does not match its exact identity fields"
            )

    def payload_dict(self) -> Dict[str, object]:
        result = {
            "schema_version": SCHEMA_VERSION,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "registry_sha256": self.registry_sha256,
            "env_id": self.env_id,
            "reset_generation": self.reset_generation,
            "action_uid": self.action_uid,
            "action_slot": self.action_slot,
            "domain_epoch": self.domain_epoch,
            "domain_claim_sha256": self.domain_claim_sha256,
            "domain_authority_sha256": self.domain_authority_sha256,
            "domain_levels": self.domain_levels.to_dict(),
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "levels_sha256": self.levels_sha256,
            "sampler_birth_sha256": self.sampler_birth_sha256,
            "sampler_birth_index": self.sampler_birth_index,
            "sampler_draw_start": self.sampler_draw_start,
            "sampler_draw_end": self.sampler_draw_end,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_quat_wxyz": list(self.base_quat_wxyz),
            "base_spawn_w_m": list(self.base_spawn_w_m),
            "manifest_sha256": self.manifest_sha256,
            "sampler_sha256": self.sampler_sha256,
            "profile_sha256": self.profile_sha256,
            "motion_sha256": self.motion_sha256,
            "physics_sha256": self.physics_sha256,
            "solver_sha256": self.solver_sha256,
        }
        if self.sampling_mixture is not None:
            result.update(
                {
                    "sampling_mixture": self.sampling_mixture.to_dict(),
                    "sampling_stratum": self.sampling_stratum,
                    "sampling_levels": self.sampling_levels.to_dict(),
                    "frontier_arm": self.frontier_arm,
                }
            )
            if self.initial_center_single_question:
                result["initial_center_single_question"] = True
        return result

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.payload_dict())

    def to_dict(self) -> Dict[str, object]:
        result = self.payload_dict()
        result["canonical_sha256"] = self.canonical_sha256
        return result

    def sampler_identity_receipt(self) -> Dict[str, object]:
        """Return the exact sampler birth identity without runtime wrappers."""

        result = {
            "birth_id": self.sampler_birth_sha256,
            "sampler_contract_sha256": self.sampler_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.to_dict(),
            "profile_sha256": self.profile_sha256,
            "levels_sha256": self.levels_sha256,
            "birth_index": self.sampler_birth_index,
            "draw_start": self.sampler_draw_start,
            "draw_end": self.sampler_draw_end,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_start_w_m": list(self.base_spawn_w_m),
        }
        if self.sampling_mixture is not None:
            result.update(
                {
                    "sampling_mixture": (
                        self.sampling_mixture.to_dict()
                    ),
                    "sampling_stratum": self.sampling_stratum,
                    "sampling_levels": self.sampling_levels.to_dict(),
                    "frontier_arm": self.frontier_arm,
                }
            )
        return result

    @classmethod
    def from_dict(cls, value: object) -> "ActionBirthReceipt":
        if not isinstance(value, Mapping):
            raise ActionBallContractError(
                "action birth receipt must be a mapping"
            )
        has_mixture = "sampling_mixture" in value
        has_initial_center = "initial_center_single_question" in value
        if has_initial_center and not has_mixture:
            raise ActionBallContractError(
                "legacy birth cannot carry mixture sampling metadata"
            )
        if has_initial_center:
            payload_keys = _INITIAL_CENTER_MIXTURE_BIRTH_PAYLOAD_KEYS
        elif has_mixture:
            payload_keys = _MIXTURE_BIRTH_PAYLOAD_KEYS
        else:
            payload_keys = _BIRTH_PAYLOAD_KEYS
        row = _exact_mapping(
            value,
            (*payload_keys, "canonical_sha256"),
            name="action birth receipt",
        )
        if row["schema_version"] != SCHEMA_VERSION:
            raise ActionBallContractError(
                "unsupported action birth receipt schema_version"
            )
        if row["runtime_contract_sha256"] != RUNTIME_CONTRACT_SHA256:
            raise ActionBallContractError(
                "action birth receipt runtime contract SHA mismatch"
            )
        if has_initial_center and row[
            "initial_center_single_question"
        ] is not True:
            # The key exists only to record a collapse that actually happened;
            # a serialized ``False`` would be an unwritten default masquerading
            # as evidence, and would also fork the canonical bytes.
            raise ActionBallContractError(
                "birth initial_center_single_question is only written when true"
            )
        fields = {
            name: row[name]
            for name in payload_keys
            if name not in ("schema_version", "runtime_contract_sha256")
        }
        fields["domain_levels"] = ActionDomainLevels.from_dict(
            row["domain_levels"]
        )
        if has_mixture:
            fields["sampling_mixture"] = (
                ActionSamplingMixture.from_dict(
                    row["sampling_mixture"]
                )
            )
            fields["sampling_levels"] = ActionDomainLevels.from_dict(
                row["sampling_levels"]
            )
        receipt = cls(**fields)  # type: ignore[arg-type]
        declared = _sha256(
            row["canonical_sha256"], name="canonical_sha256"
        )
        if declared != receipt.canonical_sha256:
            raise ActionBallContractError(
                "action birth receipt canonical SHA mismatch"
            )
        return receipt

    def assert_contract(
        self,
        *,
        binding: ActionBinding,
        pins: RuntimePins,
        mobility_mode: str,
        registry_sha256: str,
    ) -> None:
        if (
            self.action_uid != binding.action_uid
            or self.action_slot != binding.action_slot
            or self.motion_sha256 != binding.motion_sha256
            or self.profile_sha256 != binding.profile_sha256
        ):
            raise ActionBallContractError(
                "birth receipt does not match its manifest action binding"
            )
        if (
            self.manifest_sha256 != pins.manifest_sha256
            or self.sampler_sha256 != pins.sampler_sha256
            or self.domain_authority_sha256
            != pins.domain_authority_sha256
            or self.physics_sha256 != pins.physics_sha256
            or self.solver_sha256 != pins.solver_sha256
        ):
            raise ActionBallContractError(
                "birth receipt does not match run-global pins"
            )
        if self.mobility_mode != _mode(mobility_mode):
            raise ActionBallContractError(
                "birth receipt mobility mode differs from frozen run mode"
            )
        if self.registry_sha256 != _sha256(
            registry_sha256, name="registry_sha256"
        ):
            raise ActionBallContractError(
                "birth receipt registry SHA differs from the bound registry"
            )


@dataclass(frozen=True)
class ActionTeacherTiming:
    """Canonical unclipped teacher-retiming values for one solved swing."""

    required_racket_site_speed_mps: float
    teacher_rate: float
    scaled_t_hit_s: float
    scaled_t_cycle_s: float
    pre_swing_wait_s: float


BASE_PREPARATION_SCHEMA_VERSION = 1
BASE_PREPARATION_REJECT_REASON = (
    "base_preparation_time_insufficient"
)
_BASE_PREPARATION_MOTION_MODEL = (
    "rest_to_rest_planar_trapezoid_with_settle_margin"
)


@dataclass(frozen=True)
class BasePreparationContract:
    """Pinned conservative planar base-motion envelope for ``move``.

    The speed and acceleration values are hard limits used by the admission
    proof, not controller observations or policy inputs.  A producer must pin
    this exact object in its solver contract before admitting any ``move``
    sample.
    """

    max_planar_speed_mps: float
    max_planar_acceleration_mps2: float
    settle_margin_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_planar_speed_mps",
            _finite(
                self.max_planar_speed_mps,
                name="base_preparation.max_planar_speed_mps",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "max_planar_acceleration_mps2",
            _finite(
                self.max_planar_acceleration_mps2,
                name=(
                    "base_preparation."
                    "max_planar_acceleration_mps2"
                ),
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "settle_margin_s",
            _finite(
                self.settle_margin_s,
                name="base_preparation.settle_margin_s",
                minimum=0.0,
            ),
        )
        if self.max_planar_speed_mps <= 0.0:
            raise ActionBallContractError(
                "base preparation max planar speed must be > 0"
            )
        if self.max_planar_acceleration_mps2 <= 0.0:
            raise ActionBallContractError(
                "base preparation max planar acceleration must be > 0"
            )

    def payload_dict(self) -> Dict[str, object]:
        return {
            "schema_version": BASE_PREPARATION_SCHEMA_VERSION,
            "kind": "base_preparation_contract",
            "motion_model": _BASE_PREPARATION_MOTION_MODEL,
            "max_planar_speed_mps": self.max_planar_speed_mps,
            "max_planar_acceleration_mps2": (
                self.max_planar_acceleration_mps2
            ),
            "settle_margin_s": self.settle_margin_s,
        }

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.payload_dict())

    def to_dict(self) -> Dict[str, object]:
        result = self.payload_dict()
        result["canonical_sha256"] = self.canonical_sha256
        return result

    @classmethod
    def from_dict(cls, value: object) -> "BasePreparationContract":
        row = _exact_mapping(
            value,
            (
                "schema_version",
                "kind",
                "motion_model",
                "max_planar_speed_mps",
                "max_planar_acceleration_mps2",
                "settle_margin_s",
                "canonical_sha256",
            ),
            name="base preparation contract",
        )
        if row["schema_version"] != BASE_PREPARATION_SCHEMA_VERSION:
            raise ActionBallContractError(
                "unsupported base preparation contract schema_version"
            )
        if row["kind"] != "base_preparation_contract":
            raise ActionBallContractError(
                "base preparation contract kind mismatch"
            )
        if row["motion_model"] != _BASE_PREPARATION_MOTION_MODEL:
            raise ActionBallContractError(
                "base preparation motion model mismatch"
            )
        result = cls(
            max_planar_speed_mps=row["max_planar_speed_mps"],
            max_planar_acceleration_mps2=(
                row["max_planar_acceleration_mps2"]
            ),
            settle_margin_s=row["settle_margin_s"],
        )
        declared = _sha256(
            row["canonical_sha256"],
            name="base preparation contract canonical_sha256",
        )
        if declared != result.canonical_sha256:
            raise ActionBallContractError(
                "base preparation contract canonical SHA mismatch"
            )
        return result


def _base_motion_time_required_s(
    distance_m: float,
    contract: BasePreparationContract,
) -> float:
    """Return the rest-to-rest triangular/trapezoidal minimum time."""

    if distance_m == 0.0:
        return 0.0
    speed = contract.max_planar_speed_mps
    acceleration = contract.max_planar_acceleration_mps2
    distance_to_reach_speed = speed * speed / acceleration
    if distance_m <= distance_to_reach_speed:
        return 2.0 * math.sqrt(distance_m / acceleration)
    return (
        2.0 * speed / acceleration
        + (distance_m - distance_to_reach_speed) / speed
    )


@dataclass(frozen=True)
class BasePreparationReceipt:
    """Per-proposal admission proof kept outside policy difficulty.

    Each receipt accounts for exactly one sampled solver proposal and zero
    policy attempts.  Rejected rows therefore remain in the proposal
    denominator while never entering return-failure or curriculum-difficulty
    statistics.
    """

    proposal_sample_sha256: str
    proposal_sample_index: int
    mobility_mode: str
    preparation_contract_sha256: str | None
    executed_base_travel_b_yaw_m: Vec3
    planar_travel_distance_m: float
    max_planar_speed_mps: float
    max_planar_acceleration_mps2: float
    settle_margin_s: float
    motion_time_required_s: float
    move_preparation_required_s: float
    reaction_margin_s: float
    required_preparation_s: float
    available_preparation_s: float
    admitted: bool
    reject_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_sample_sha256",
            _sha256(
                self.proposal_sample_sha256,
                name="base preparation proposal_sample_sha256",
            ),
        )
        object.__setattr__(
            self,
            "proposal_sample_index",
            _plain_int(
                self.proposal_sample_index,
                name="base preparation proposal_sample_index",
            ),
        )
        object.__setattr__(
            self,
            "mobility_mode",
            _mode(self.mobility_mode),
        )
        travel = _vec3(
            self.executed_base_travel_b_yaw_m,
            name="base preparation executed_base_travel_b_yaw_m",
        )
        if travel[2] != 0.0:
            raise ActionBallContractError(
                "base preparation travel must be planar with exact z=0"
            )
        object.__setattr__(
            self,
            "executed_base_travel_b_yaw_m",
            travel,
        )
        for name in (
            "planar_travel_distance_m",
            "max_planar_speed_mps",
            "max_planar_acceleration_mps2",
            "settle_margin_s",
            "motion_time_required_s",
            "move_preparation_required_s",
            "reaction_margin_s",
            "required_preparation_s",
            "available_preparation_s",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=f"base preparation {name}",
                    minimum=0.0,
                ),
            )
        if type(self.admitted) is not bool:
            raise ActionBallContractError(
                "base preparation admitted must be bool"
            )
        if type(self.reject_reason) is not str:
            raise ActionBallContractError(
                "base preparation reject_reason must be str"
            )
        expected_distance = math.hypot(travel[0], travel[1])
        if self.planar_travel_distance_m != expected_distance:
            raise ActionBallContractError(
                "base preparation distance differs from executed travel"
            )
        if self.mobility_mode == "no_move":
            if (
                travel != (0.0, 0.0, 0.0)
                or self.preparation_contract_sha256 is not None
                or self.max_planar_speed_mps != 0.0
                or self.max_planar_acceleration_mps2 != 0.0
                or self.settle_margin_s != 0.0
            ):
                raise ActionBallContractError(
                    "no_move preparation must have zero executed travel "
                    "and no motion contract"
                )
            expected_motion = 0.0
            expected_move_required = 0.0
        else:
            contract = BasePreparationContract(
                max_planar_speed_mps=self.max_planar_speed_mps,
                max_planar_acceleration_mps2=(
                    self.max_planar_acceleration_mps2
                ),
                settle_margin_s=self.settle_margin_s,
            )
            declared_contract = _sha256(
                self.preparation_contract_sha256,
                name="base preparation contract SHA",
            )
            if declared_contract != contract.canonical_sha256:
                raise ActionBallContractError(
                    "base preparation receipt contract SHA mismatch"
                )
            object.__setattr__(
                self,
                "preparation_contract_sha256",
                declared_contract,
            )
            expected_motion = _base_motion_time_required_s(
                expected_distance,
                contract,
            )
            expected_move_required = (
                0.0
                if expected_distance == 0.0
                else expected_motion + contract.settle_margin_s
            )
        expected_required = max(
            self.reaction_margin_s,
            expected_move_required,
        )
        if self.available_preparation_s < self.reaction_margin_s:
            raise ActionBallContractError(
                "base preparation receipt requires a wait already above "
                "the reaction margin"
            )
        if self.available_preparation_s > MAX_PRE_SWING_WAIT_S:
            raise ActionBallContractError(
                "base preparation receipt requires a wait at or below 1s"
            )
        expected_admitted = (
            self.available_preparation_s >= expected_required
        )
        expected_reason = (
            "" if expected_admitted else BASE_PREPARATION_REJECT_REASON
        )
        expected_values = (
            expected_motion,
            expected_move_required,
            expected_required,
            expected_admitted,
            expected_reason,
        )
        declared_values = (
            self.motion_time_required_s,
            self.move_preparation_required_s,
            self.required_preparation_s,
            self.admitted,
            self.reject_reason,
        )
        if declared_values != expected_values:
            raise ActionBallContractError(
                "base preparation receipt differs from exact motion "
                "envelope/admission formula"
            )

    @property
    def proposal_count_delta(self) -> int:
        return 1

    @property
    def policy_attempt_count_delta(self) -> int:
        return 0

    @property
    def solver_rejection_count_delta(self) -> int:
        return 0 if self.admitted else 1

    def payload_dict(self) -> Dict[str, object]:
        return {
            "schema_version": BASE_PREPARATION_SCHEMA_VERSION,
            "kind": "base_preparation_receipt",
            "proposal_sample_sha256": self.proposal_sample_sha256,
            "proposal_sample_index": self.proposal_sample_index,
            "mobility_mode": self.mobility_mode,
            "preparation_contract_sha256": (
                self.preparation_contract_sha256
            ),
            "executed_base_travel_b_yaw_m": list(
                self.executed_base_travel_b_yaw_m
            ),
            "planar_travel_distance_m": self.planar_travel_distance_m,
            "max_planar_speed_mps": self.max_planar_speed_mps,
            "max_planar_acceleration_mps2": (
                self.max_planar_acceleration_mps2
            ),
            "settle_margin_s": self.settle_margin_s,
            "motion_time_required_s": self.motion_time_required_s,
            "move_preparation_required_s": (
                self.move_preparation_required_s
            ),
            "reaction_margin_s": self.reaction_margin_s,
            "required_preparation_s": self.required_preparation_s,
            "available_preparation_s": self.available_preparation_s,
            "admitted": self.admitted,
            "reject_reason": self.reject_reason,
            "proposal_count_delta": self.proposal_count_delta,
            "policy_attempt_count_delta": (
                self.policy_attempt_count_delta
            ),
            "solver_rejection_count_delta": (
                self.solver_rejection_count_delta
            ),
        }

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.payload_dict())

    def to_dict(self) -> Dict[str, object]:
        result = self.payload_dict()
        result["canonical_sha256"] = self.canonical_sha256
        return result

    @classmethod
    def evaluate(
        cls,
        *,
        proposal_sample_sha256: str,
        proposal_sample_index: int,
        mobility_mode: str,
        base_travel_b_yaw_m: Sequence[float],
        reaction_margin_s: float,
        available_preparation_s: float,
        contract: BasePreparationContract | None,
    ) -> "BasePreparationReceipt":
        mode = _mode(mobility_mode)
        latent_travel = _vec3(
            base_travel_b_yaw_m,
            name="base preparation base_travel_b_yaw_m",
        )
        if mode == "no_move":
            if contract is not None:
                raise ActionBallContractError(
                    "no_move base preparation must not bind a motion "
                    "contract"
                )
            travel = (0.0, 0.0, 0.0)
            contract_sha256 = None
            speed = 0.0
            acceleration = 0.0
            settle = 0.0
            motion_time = 0.0
            move_required = 0.0
        else:
            if not isinstance(contract, BasePreparationContract):
                raise ActionBallContractError(
                    "move base preparation requires an exact pinned "
                    "BasePreparationContract"
                )
            if latent_travel[2] != 0.0:
                raise ActionBallContractError(
                    "move base preparation travel must have exact z=0"
                )
            travel = latent_travel
            contract_sha256 = contract.canonical_sha256
            speed = contract.max_planar_speed_mps
            acceleration = contract.max_planar_acceleration_mps2
            settle = contract.settle_margin_s
            distance = math.hypot(travel[0], travel[1])
            motion_time = _base_motion_time_required_s(
                distance,
                contract,
            )
            move_required = (
                0.0
                if distance == 0.0
                else motion_time + settle
            )
        distance = math.hypot(travel[0], travel[1])
        reaction = _finite(
            reaction_margin_s,
            name="base preparation reaction_margin_s",
            minimum=0.0,
        )
        available = _finite(
            available_preparation_s,
            name="base preparation available_preparation_s",
            minimum=0.0,
        )
        required = max(reaction, move_required)
        admitted = available >= required
        return cls(
            proposal_sample_sha256=proposal_sample_sha256,
            proposal_sample_index=proposal_sample_index,
            mobility_mode=mode,
            preparation_contract_sha256=contract_sha256,
            executed_base_travel_b_yaw_m=travel,
            planar_travel_distance_m=distance,
            max_planar_speed_mps=speed,
            max_planar_acceleration_mps2=acceleration,
            settle_margin_s=settle,
            motion_time_required_s=motion_time,
            move_preparation_required_s=move_required,
            reaction_margin_s=reaction,
            required_preparation_s=required,
            available_preparation_s=available,
            admitted=admitted,
            reject_reason=(
                "" if admitted else BASE_PREPARATION_REJECT_REASON
            ),
        )

    @classmethod
    def from_dict(cls, value: object) -> "BasePreparationReceipt":
        keys = (
            "schema_version",
            "kind",
            "proposal_sample_sha256",
            "proposal_sample_index",
            "mobility_mode",
            "preparation_contract_sha256",
            "executed_base_travel_b_yaw_m",
            "planar_travel_distance_m",
            "max_planar_speed_mps",
            "max_planar_acceleration_mps2",
            "settle_margin_s",
            "motion_time_required_s",
            "move_preparation_required_s",
            "reaction_margin_s",
            "required_preparation_s",
            "available_preparation_s",
            "admitted",
            "reject_reason",
            "proposal_count_delta",
            "policy_attempt_count_delta",
            "solver_rejection_count_delta",
        )
        row = _exact_mapping(
            value,
            (*keys, "canonical_sha256"),
            name="base preparation receipt",
        )
        if row["schema_version"] != BASE_PREPARATION_SCHEMA_VERSION:
            raise ActionBallContractError(
                "unsupported base preparation receipt schema_version"
            )
        if row["kind"] != "base_preparation_receipt":
            raise ActionBallContractError(
                "base preparation receipt kind mismatch"
            )
        expected_deltas = (
            1,
            0,
            0 if row["admitted"] is True else 1,
        )
        declared_deltas = (
            row["proposal_count_delta"],
            row["policy_attempt_count_delta"],
            row["solver_rejection_count_delta"],
        )
        if declared_deltas != expected_deltas:
            raise ActionBallContractError(
                "base preparation receipt accounting deltas mismatch"
            )
        result = cls(
            **{
                name: row[name]
                for name in keys
                if name
                not in (
                    "schema_version",
                    "kind",
                    "proposal_count_delta",
                    "policy_attempt_count_delta",
                    "solver_rejection_count_delta",
                )
            }
        )
        declared_sha256 = _sha256(
            row["canonical_sha256"],
            name="base preparation receipt canonical_sha256",
        )
        if declared_sha256 != result.canonical_sha256:
            raise ActionBallContractError(
                "base preparation receipt canonical SHA mismatch"
            )
        return result


class BasePreparationAdmissionError(ActionBallContractError):
    """Named solver-admission rejection carrying its full proposal proof."""

    def __init__(self, receipt: BasePreparationReceipt) -> None:
        if (
            not isinstance(receipt, BasePreparationReceipt)
            or receipt.admitted
            or receipt.reject_reason
            != BASE_PREPARATION_REJECT_REASON
        ):
            raise ActionBallContractError(
                "base preparation rejection requires a rejected receipt"
            )
        self.receipt = receipt
        self.reject_reason = receipt.reject_reason
        super().__init__(
            f"{receipt.reject_reason}: sample="
            f"{receipt.proposal_sample_index}, distance_m="
            f"{receipt.planar_travel_distance_m}, required_s="
            f"{receipt.required_preparation_s}, available_s="
            f"{receipt.available_preparation_s}"
        )


@dataclass(frozen=True)
class ActionTeacherTimingWithBasePreparation:
    """Admitted timing plus its separately hashable solver proof."""

    timing: ActionTeacherTiming
    base_preparation: BasePreparationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.timing, ActionTeacherTiming):
            raise ActionBallContractError(
                "prepared teacher timing requires ActionTeacherTiming"
            )
        if (
            not isinstance(
                self.base_preparation,
                BasePreparationReceipt,
            )
            or not self.base_preparation.admitted
        ):
            raise ActionBallContractError(
                "prepared teacher timing requires an admitted base "
                "preparation receipt"
            )
        if (
            self.base_preparation.available_preparation_s
            != self.timing.pre_swing_wait_s
        ):
            raise ActionBallContractError(
                "base preparation available time differs from teacher wait"
            )


def derive_action_teacher_timing(
    *,
    racket_velocity_w_mps: Sequence[float],
    time_to_contact_s: float,
    reference_t_hit_s: float,
    reference_t_cycle_s: float,
    reference_racket_site_speed_mps: float,
    reaction_margin_s: float,
    teacher_rate_min: float,
    teacher_rate_max: float,
) -> ActionTeacherTiming:
    """Derive the exact no-clipping TTC/teacher timing contract."""

    velocity = _vec3(
        racket_velocity_w_mps, name="racket_velocity_w_mps"
    )
    ttc = _finite(
        time_to_contact_s, name="time_to_contact_s", minimum=0.0
    )
    reference_hit = _finite(
        reference_t_hit_s, name="reference_t_hit_s", minimum=0.0
    )
    reference_cycle = _finite(
        reference_t_cycle_s, name="reference_t_cycle_s", minimum=0.0
    )
    reference_speed = _finite(
        reference_racket_site_speed_mps,
        name="reference_racket_site_speed_mps",
        minimum=0.0,
    )
    reaction_margin = _finite(
        reaction_margin_s, name="reaction_margin_s", minimum=0.0
    )
    rate_min = _finite(
        teacher_rate_min, name="teacher_rate_min", minimum=0.0
    )
    rate_max = _finite(
        teacher_rate_max, name="teacher_rate_max", minimum=0.0
    )
    if reference_hit <= 0.0 or reference_cycle <= reference_hit:
        raise ActionBallContractError(
            "reference timing requires 0 < reference_t_hit_s < "
            "reference_t_cycle_s"
        )
    if reference_speed <= 0.0:
        raise ActionBallContractError(
            "reference_racket_site_speed_mps must be > 0"
        )
    if rate_min <= 0.0 or not rate_min <= 1.0 <= rate_max:
        raise ActionBallContractError(
            "teacher rate bounds must satisfy 0 < min <= 1 <= max"
        )
    required_speed = math.sqrt(
        sum(component * component for component in velocity)
    )
    try:
        teacher_rate = (
            _contact_geometry.canonical_teacher_rate_from_site_speed(
                required_speed,
                reference_speed,
                rate_min,
                rate_max,
            )
        )
    except _contact_geometry.ExactFaceContactGeometryError as error:
        raise ActionBallContractError(str(error)) from error
    scaled_hit = reference_hit / teacher_rate
    scaled_cycle = reference_cycle / teacher_rate
    wait = ttc - scaled_hit
    if not reaction_margin <= wait <= MAX_PRE_SWING_WAIT_S:
        raise ActionBallContractError(
            "pre-swing wait lies outside reaction-margin/1s bounds"
        )
    return ActionTeacherTiming(
        required_racket_site_speed_mps=required_speed,
        teacher_rate=teacher_rate,
        scaled_t_hit_s=scaled_hit,
        scaled_t_cycle_s=scaled_cycle,
        pre_swing_wait_s=wait,
    )


def derive_action_teacher_site_timing(
    *,
    racket_site_velocity_w_mps: Sequence[float],
    time_to_contact_s: float,
    reference_t_hit_s: float,
    reference_t_cycle_s: float,
    reference_racket_site_speed_mps: float,
    reaction_margin_s: float,
    teacher_rate_min: float,
    teacher_rate_max: float,
) -> ActionTeacherTiming:
    """Explicit v2 name for the canonical-site teacher clock.

    ``derive_action_teacher_timing`` remains as a legacy dependency-light
    primitive for older non-v2 callers.  Fresh ActionBall receipts may only
    call this wrapper after the exact face-centre/angular-rate solve.
    """

    return derive_action_teacher_timing(
        racket_velocity_w_mps=racket_site_velocity_w_mps,
        time_to_contact_s=time_to_contact_s,
        reference_t_hit_s=reference_t_hit_s,
        reference_t_cycle_s=reference_t_cycle_s,
        reference_racket_site_speed_mps=(
            reference_racket_site_speed_mps
        ),
        reaction_margin_s=reaction_margin_s,
        teacher_rate_min=teacher_rate_min,
        teacher_rate_max=teacher_rate_max,
    )


def derive_action_teacher_timing_with_base_preparation(
    *,
    racket_velocity_w_mps: Sequence[float],
    time_to_contact_s: float,
    reference_t_hit_s: float,
    reference_t_cycle_s: float,
    reference_racket_site_speed_mps: float,
    reaction_margin_s: float,
    teacher_rate_min: float,
    teacher_rate_max: float,
    proposal_sample_sha256: str,
    proposal_sample_index: int,
    mobility_mode: str,
    base_travel_b_yaw_m: Sequence[float],
    base_preparation_contract: BasePreparationContract | None,
) -> ActionTeacherTimingWithBasePreparation:
    """Derive teacher timing, then admit only physically preparable travel.

    The original :func:`derive_action_teacher_timing` remains untouched so
    legacy ``no_move`` receipts and exact-resume identities stay byte-exact.
    This explicit seam is mandatory for ``move``: omitting its pinned motion
    contract fails closed.
    """

    mode = _mode(mobility_mode)
    if mode == "move" and not isinstance(
        base_preparation_contract,
        BasePreparationContract,
    ):
        raise ActionBallContractError(
            "move teacher timing requires an exact pinned "
            "BasePreparationContract"
        )
    if mode == "no_move" and base_preparation_contract is not None:
        raise ActionBallContractError(
            "no_move teacher timing must not bind a base preparation "
            "contract"
        )
    timing = derive_action_teacher_timing(
        racket_velocity_w_mps=racket_velocity_w_mps,
        time_to_contact_s=time_to_contact_s,
        reference_t_hit_s=reference_t_hit_s,
        reference_t_cycle_s=reference_t_cycle_s,
        reference_racket_site_speed_mps=(
            reference_racket_site_speed_mps
        ),
        reaction_margin_s=reaction_margin_s,
        teacher_rate_min=teacher_rate_min,
        teacher_rate_max=teacher_rate_max,
    )
    preparation = BasePreparationReceipt.evaluate(
        proposal_sample_sha256=proposal_sample_sha256,
        proposal_sample_index=proposal_sample_index,
        mobility_mode=mode,
        base_travel_b_yaw_m=base_travel_b_yaw_m,
        reaction_margin_s=reaction_margin_s,
        available_preparation_s=timing.pre_swing_wait_s,
        contract=base_preparation_contract,
    )
    if not preparation.admitted:
        raise BasePreparationAdmissionError(preparation)
    return ActionTeacherTimingWithBasePreparation(
        timing=timing,
        base_preparation=preparation,
    )


_TASK_PAYLOAD_KEYS = (
    "schema_version",
    "runtime_contract_sha256",
    "registry_sha256",
    "birth_sha256",
    "sample_sha256",
    "env_id",
    "reset_generation",
    "swing_generation",
    "action_uid",
    "action_slot",
    "domain_epoch",
    "domain_claim_sha256",
    "domain_authority_sha256",
    "domain_levels",
    "arm_catalog_sha256",
    "levels_sha256",
    "sampler_birth_sha256",
    "mobility_mode",
    "base_yaw_rad",
    "base_quat_wxyz",
    "base_spawn_w_m",
    "base_goal_w_m",
    "sample_index",
    "sample_draw_start",
    "sample_draw_end",
    "base_spawn_latent_w_m",
    "base_travel_latent_b_yaw_m",
    "contact_offset_from_base_goal_b_yaw_m",
    "ball_contact_w_m",
    "racket_site_target_w_m",
    "time_to_contact_s",
    "incoming_speed_mps",
    "incoming_direction_b_yaw",
    "incoming_velocity_w_mps",
    "spin_magnitude_radps",
    "spin_direction_b_yaw",
    "incoming_spin_w_radps",
    "landing_aim_w_xy_m",
    "mount_normal_sign",
    "racket_normal_w",
    "reference_racket_quat_wxyz",
    "reference_racket_angular_velocity_w_radps",
    "racket_command_quat_wxyz",
    "racket_face_center_velocity_w_mps",
    "racket_site_velocity_w_mps",
    "racket_command_angular_velocity_w_radps",
    "geometry_source_sha256",
    "reference_t_hit_s",
    "reference_t_cycle_s",
    "reference_racket_site_speed_mps",
    "required_racket_site_speed_mps",
    "reaction_margin_s",
    "teacher_rate_min",
    "teacher_rate_max",
    "teacher_rate",
    "scaled_t_hit_s",
    "scaled_t_cycle_s",
    "pre_swing_wait_s",
    "solver_residual_m",
    "manifest_sha256",
    "sampler_sha256",
    "profile_sha256",
    "motion_sha256",
    "physics_sha256",
    "solver_sha256",
)
_MIXTURE_TASK_PAYLOAD_KEYS = (
    *_TASK_PAYLOAD_KEYS,
    "birth_index",
    "birth_sampling_stratum",
    "birth_sampling_levels",
    "birth_frontier_arm",
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "contact_time_step_s",
    "time_to_contact_tick",
)
_COUNTER_RALLY_TASK_PAYLOAD_KEYS = (
    *_TASK_PAYLOAD_KEYS,
    "counter_rally_task",
)
_COUNTER_RALLY_MIXTURE_TASK_PAYLOAD_KEYS = (
    *_MIXTURE_TASK_PAYLOAD_KEYS,
    "counter_rally_task",
)
# Same rule as the birth row: the initial-center collapse is only ever written
# down when it happened, so pre-existing task receipts keep their exact bytes.
_INITIAL_CENTER_MIXTURE_TASK_PAYLOAD_KEYS = (
    *_MIXTURE_TASK_PAYLOAD_KEYS,
    "initial_center_single_question",
)
_INITIAL_CENTER_COUNTER_RALLY_MIXTURE_TASK_PAYLOAD_KEYS = (
    *_COUNTER_RALLY_MIXTURE_TASK_PAYLOAD_KEYS,
    "initial_center_single_question",
)


@dataclass(frozen=True)
class ActionTaskReceiptRef:
    """Small immutable Motion↔Racket reference to one exact task receipt."""

    env_id: int
    reset_generation: int
    swing_generation: int
    action_uid: int
    action_slot: int
    birth_sha256: str
    sample_sha256: str
    task_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="ref.env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="ref.reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "swing_generation",
            _plain_int(
                self.swing_generation, name="ref.swing_generation"
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="ref.action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="ref.action_slot"),
        )
        for name in (
            "birth_sha256",
            "sample_sha256",
            "task_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=f"ref.{name}"),
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "env_id": self.env_id,
            "reset_generation": self.reset_generation,
            "swing_generation": self.swing_generation,
            "action_uid": self.action_uid,
            "action_slot": self.action_slot,
            "birth_sha256": self.birth_sha256,
            "sample_sha256": self.sample_sha256,
            "task_sha256": self.task_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ActionTaskReceiptRef":
        row = _exact_mapping(
            value,
            (
                "env_id",
                "reset_generation",
                "swing_generation",
                "action_uid",
                "action_slot",
                "birth_sha256",
                "sample_sha256",
                "task_sha256",
            ),
            name="action task receipt ref",
        )
        return cls(**{name: row[name] for name in row})  # type: ignore[arg-type]


_FULL_TASK_RECEIPT_VALIDATION = object()
_DIAGNOSTIC_PREVALIDATED_TASK_RECEIPT = object()


@dataclass(frozen=True)
class ActionBallTaskReceipt:
    """One admitted ball plus the solved racket task installed for it."""

    birth_sha256: str
    sample_sha256: str
    env_id: int
    reset_generation: int
    swing_generation: int
    action_uid: int
    action_slot: int
    domain_epoch: int
    domain_claim_sha256: str
    domain_authority_sha256: str
    domain_levels: ActionDomainLevels
    arm_catalog_sha256: str
    levels_sha256: str
    sampler_birth_sha256: str
    mobility_mode: str
    base_yaw_rad: float
    base_quat_wxyz: Tuple[float, float, float, float]
    base_spawn_w_m: Vec3
    base_goal_w_m: Vec3
    sample_index: int
    sample_draw_start: int
    sample_draw_end: int
    base_spawn_latent_w_m: Vec3
    base_travel_latent_b_yaw_m: Vec3
    contact_offset_from_base_goal_b_yaw_m: Vec3
    ball_contact_w_m: Vec3
    racket_site_target_w_m: Vec3
    time_to_contact_s: float
    incoming_speed_mps: float
    incoming_direction_b_yaw: Vec3
    incoming_velocity_w_mps: Vec3
    spin_magnitude_radps: float
    spin_direction_b_yaw: Vec3
    incoming_spin_w_radps: Vec3
    landing_aim_w_xy_m: Vec2
    mount_normal_sign: int
    racket_normal_w: Vec3
    reference_racket_quat_wxyz: Tuple[float, float, float, float]
    reference_racket_angular_velocity_w_radps: Vec3
    racket_command_quat_wxyz: Tuple[float, float, float, float]
    racket_face_center_velocity_w_mps: Vec3
    racket_site_velocity_w_mps: Vec3
    racket_command_angular_velocity_w_radps: Vec3
    geometry_source_sha256: str
    reference_t_hit_s: float
    reference_t_cycle_s: float
    reference_racket_site_speed_mps: float
    required_racket_site_speed_mps: float
    reaction_margin_s: float
    teacher_rate_min: float
    teacher_rate_max: float
    teacher_rate: float
    scaled_t_hit_s: float
    scaled_t_cycle_s: float
    pre_swing_wait_s: float
    solver_residual_m: float
    manifest_sha256: str
    sampler_sha256: str
    profile_sha256: str
    motion_sha256: str
    physics_sha256: str
    solver_sha256: str
    registry_sha256: str
    birth_index: int = -1
    birth_sampling_stratum: str = "domain"
    birth_sampling_levels: ActionDomainLevels | None = None
    birth_frontier_arm: str | None = None
    sampling_mixture: ActionSamplingMixture | None = None
    sampling_stratum: str = "domain"
    sampling_levels: ActionDomainLevels | None = None
    frontier_arm: str | None = None
    contact_time_step_s: float | None = None
    time_to_contact_tick: int | None = None
    counter_rally_task: CounterRallyTaskIdentity | None = None
    initial_center_single_question: bool = False
    _validation_mode: InitVar[object] = _FULL_TASK_RECEIPT_VALIDATION

    def __post_init__(
        self, _validation_mode: object = _FULL_TASK_RECEIPT_VALIDATION
    ) -> None:
        if (
            _validation_mode is not _FULL_TASK_RECEIPT_VALIDATION
            and _validation_mode
            is not _DIAGNOSTIC_PREVALIDATED_TASK_RECEIPT
        ):
            raise ActionBallContractError(
                "task receipt validation mode is not an internal authority"
            )
        for name in (
            "birth_sha256",
            "sample_sha256",
            "registry_sha256",
            "domain_claim_sha256",
            "domain_authority_sha256",
            "arm_catalog_sha256",
            "levels_sha256",
            "sampler_birth_sha256",
            "manifest_sha256",
            "sampler_sha256",
            "profile_sha256",
            "motion_sha256",
            "physics_sha256",
            "solver_sha256",
            "geometry_source_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "swing_generation",
            _plain_int(
                self.swing_generation,
                name="swing_generation",
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )
        object.__setattr__(
            self,
            "domain_epoch",
            _plain_int(self.domain_epoch, name="domain_epoch"),
        )
        if not isinstance(self.domain_levels, ActionDomainLevels):
            raise ActionBallContractError(
                "task receipt domain_levels must be ActionDomainLevels"
            )
        if self.levels_sha256 != self.domain_levels.canonical_sha256:
            raise ActionBallContractError(
                "task receipt levels SHA does not match its frozen payload"
            )
        if self.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ActionBallContractError(
                "task receipt arm catalog SHA mismatch"
            )
        if (
            self.counter_rally_task is not None
            and not isinstance(
                self.counter_rally_task,
                CounterRallyTaskIdentity,
            )
        ):
            raise ActionBallContractError(
                "task receipt counter_rally_task must be "
                "CounterRallyTaskIdentity or None"
            )
        object.__setattr__(
            self, "mobility_mode", _mode(self.mobility_mode)
        )
        object.__setattr__(
            self,
            "base_yaw_rad",
            _finite(self.base_yaw_rad, name="base_yaw_rad"),
        )
        object.__setattr__(
            self,
            "base_quat_wxyz",
            _unit_quat_wxyz(
                self.base_quat_wxyz, name="base_quat_wxyz"
            ),
        )
        for name in (
            "base_spawn_w_m",
            "base_goal_w_m",
            "base_spawn_latent_w_m",
            "base_travel_latent_b_yaw_m",
            "contact_offset_from_base_goal_b_yaw_m",
            "ball_contact_w_m",
            "racket_site_target_w_m",
            "incoming_velocity_w_mps",
            "incoming_spin_w_radps",
            "reference_racket_angular_velocity_w_radps",
            "racket_face_center_velocity_w_mps",
            "racket_site_velocity_w_mps",
            "racket_command_angular_velocity_w_radps",
        ):
            object.__setattr__(
                self, name, _vec3(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "mount_normal_sign",
            _plain_int(
                self.mount_normal_sign,
                name="mount_normal_sign",
                minimum=-1,
                maximum=1,
            ),
        )
        if self.mount_normal_sign not in (-1, 1):
            raise ActionBallContractError(
                "task receipt mount_normal_sign must be +1 or -1"
            )
        object.__setattr__(
            self,
            "reference_racket_quat_wxyz",
            _contact_geometry.canonical_quat_wxyz(
                _unit_quat_wxyz(
                    self.reference_racket_quat_wxyz,
                    name="reference_racket_quat_wxyz",
                )
            ),
        )
        object.__setattr__(
            self,
            "racket_command_quat_wxyz",
            _contact_geometry.canonical_quat_wxyz(
                _unit_quat_wxyz(
                    self.racket_command_quat_wxyz,
                    name="racket_command_quat_wxyz",
                )
            ),
        )
        if (
            self.geometry_source_sha256
            != _contact_geometry.GEOMETRY_SOURCE_SHA256
        ):
            raise ActionBallContractError(
                "task receipt exact-face geometry source SHA mismatch"
            )
        object.__setattr__(
            self,
            "sample_index",
            _plain_int(self.sample_index, name="sample_index"),
        )
        object.__setattr__(
            self,
            "sample_draw_start",
            _plain_int(
                self.sample_draw_start, name="sample_draw_start"
            ),
        )
        object.__setattr__(
            self,
            "sample_draw_end",
            _plain_int(
                self.sample_draw_end,
                name="sample_draw_end",
                minimum=1,
            ),
        )
        if (
            self.sample_draw_end - self.sample_draw_start
            != SAMPLER_SAMPLE_DRAW_COUNT
        ):
            raise ActionBallContractError(
                "sampler sample draw range must consume exactly "
                f"{SAMPLER_SAMPLE_DRAW_COUNT} draws"
            )
        if type(self.initial_center_single_question) is not bool:
            raise ActionBallContractError(
                "task initial_center_single_question must be a plain bool"
            )
        if self.sampling_mixture is None:
            if (
                self.birth_index != -1
                or self.birth_sampling_stratum != "domain"
                or self.birth_frontier_arm is not None
                or self.sampling_stratum != "domain"
                or self.frontier_arm is not None
                or self.contact_time_step_s is not None
                or self.time_to_contact_tick is not None
                or self.initial_center_single_question
            ):
                raise ActionBallContractError(
                    "legacy task cannot carry mixture/tick metadata"
                )
            object.__setattr__(
                self, "birth_sampling_levels", self.domain_levels
            )
            object.__setattr__(
                self, "sampling_levels", self.domain_levels
            )
        else:
            if not isinstance(
                self.sampling_mixture, ActionSamplingMixture
            ):
                raise ActionBallContractError(
                    "task sampling_mixture has invalid type"
                )
            object.__setattr__(
                self,
                "birth_index",
                _plain_int(
                    self.birth_index,
                    name="birth_index",
                ),
            )
            if not isinstance(
                self.birth_sampling_levels, ActionDomainLevels
            ) or not isinstance(
                self.sampling_levels, ActionDomainLevels
            ):
                raise ActionBallContractError(
                    "task sampling levels have invalid type"
                )
            schedule = self.sampling_mixture.schedule
            expected_birth_stratum = schedule[
                self.birth_index % len(schedule)
            ]
            expected_sample_stratum = schedule[
                self.sample_index % len(schedule)
            ]
            inactive_birth_frontier = (
                expected_birth_stratum == "frontier"
                and self.birth_sampling_stratum == "center"
                and all(
                    getattr(self.domain_levels, arm) == 0.0
                    and getattr(self.birth_sampling_levels, arm) == 0.0
                    for arm in (
                        "base_spawn_x_lower",
                        "base_spawn_x_upper",
                        "base_spawn_y_lower",
                        "base_spawn_y_upper",
                    )
                )
            )
            # ``_literal_initial_center_active`` keys off the frozen domain
            # levels, which this receipt copies verbatim from its birth, so the
            # collapse binds the birth plan and the swing plan identically.
            # Both must therefore be the literal centre row, and nothing else.
            initial_center_collapse = (
                self.initial_center_single_question
                and all(
                    getattr(self.domain_levels, arm) == 0.0
                    for arm in ARM_KEYS
                )
            )
            if initial_center_collapse:
                if (
                    self.birth_sampling_stratum != "center"
                    or self.sampling_stratum != "center"
                    or self.birth_frontier_arm is not None
                    or self.frontier_arm is not None
                    or any(
                        getattr(self.birth_sampling_levels, arm) != 0.0
                        or getattr(self.sampling_levels, arm) != 0.0
                        for arm in ARM_KEYS
                    )
                ):
                    raise ActionBallContractError(
                        "task initial-center collapse is not the literal "
                        "all-zero center row"
                    )
            elif (
                (
                    self.birth_sampling_stratum
                    != expected_birth_stratum
                    and not inactive_birth_frontier
                )
                or self.sampling_stratum != expected_sample_stratum
            ):
                raise ActionBallContractError(
                    "task sampling stratum differs from quota schedule"
                )
            if (self.birth_sampling_stratum == "frontier") != (
                self.birth_frontier_arm is not None
            ) or (self.sampling_stratum == "frontier") != (
                self.frontier_arm is not None
            ):
                raise ActionBallContractError(
                    "task frontier arm presence disagrees with stratum"
                )
            if self.birth_frontier_arm is not None and (
                self.birth_frontier_arm
                not in (
                    "base_spawn_x_lower",
                    "base_spawn_x_upper",
                    "base_spawn_y_lower",
                    "base_spawn_y_upper",
                )
            ):
                raise ActionBallContractError(
                    "task birth frontier is not a base-spawn arm"
                )
            if self.frontier_arm is not None and (
                self.frontier_arm not in ARM_KEYS
                or self.frontier_arm.startswith("base_spawn_")
            ):
                raise ActionBallContractError(
                    "task swing frontier arm is invalid"
                )
            for arm in ARM_KEYS:
                if (
                    getattr(self.birth_sampling_levels, arm)
                    > getattr(self.domain_levels, arm) + 1.0e-15
                    or getattr(self.sampling_levels, arm)
                    > getattr(self.domain_levels, arm) + 1.0e-15
                ):
                    raise ActionBallContractError(
                        "task sampling levels exceed frozen domain"
                    )
            step = _finite(
                self.contact_time_step_s,
                name="contact_time_step_s",
                minimum=0.0,
            )
            if step <= 0.0:
                raise ActionBallContractError(
                    "task contact_time_step_s must be > 0"
                )
            tick = _plain_int(
                self.time_to_contact_tick,
                name="time_to_contact_tick",
                minimum=1,
            )
            object.__setattr__(self, "contact_time_step_s", step)
            object.__setattr__(self, "time_to_contact_tick", tick)
        object.__setattr__(
            self,
            "time_to_contact_s",
            _finite(
                self.time_to_contact_s,
                name="time_to_contact_s",
                minimum=0.0,
            ),
        )
        if (
            self.contact_time_step_s is not None
            and self.time_to_contact_s
            != self.time_to_contact_tick * self.contact_time_step_s
        ):
            raise ActionBallContractError(
                "task TTC is not exactly its policy-step tick"
            )
        object.__setattr__(
            self,
            "incoming_speed_mps",
            _finite(
                self.incoming_speed_mps,
                name="incoming_speed_mps",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "incoming_direction_b_yaw",
            _unit_vec3(
                self.incoming_direction_b_yaw,
                name="incoming_direction_b_yaw",
            ),
        )
        if self.counter_rally_task is not None:
            incoming_direction_w = _rotate_yaw(
                self.incoming_direction_b_yaw,
                self.base_yaw_rad,
            )
            horizontal_norm = math.hypot(
                incoming_direction_w[0],
                incoming_direction_w[1],
            )
            if horizontal_norm <= 1.0e-12:
                raise CounterRallyTaskIdentityError(
                    "counter-rally incoming horizontal direction is zero"
                )
            expected_return_direction = (
                -incoming_direction_w[0] / horizontal_norm,
                -incoming_direction_w[1] / horizontal_norm,
            )
            if any(
                not math.isclose(
                    declared,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                )
                for declared, expected in zip(
                    self.counter_rally_task.return_direction_env_xy,
                    expected_return_direction,
                )
            ):
                raise CounterRallyTaskIdentityError(
                    "counter-rally return direction is not the exact "
                    "horizontal reverse of the sampled incoming ball"
                )
        object.__setattr__(
            self,
            "spin_magnitude_radps",
            _finite(
                self.spin_magnitude_radps,
                name="spin_magnitude_radps",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "spin_direction_b_yaw",
            _unit_vec3(
                self.spin_direction_b_yaw,
                name="spin_direction_b_yaw",
            ),
        )
        object.__setattr__(
            self,
            "landing_aim_w_xy_m",
            _vec2(self.landing_aim_w_xy_m, name="landing_aim_w_xy_m"),
        )
        object.__setattr__(
            self,
            "racket_normal_w",
            _unit_vec3(self.racket_normal_w, name="racket_normal_w"),
        )
        object.__setattr__(
            self,
            "solver_residual_m",
            _finite(
                self.solver_residual_m,
                name="solver_residual_m",
                minimum=0.0,
            ),
        )
        for name in (
            "reference_t_hit_s",
            "reference_t_cycle_s",
            "reference_racket_site_speed_mps",
            "required_racket_site_speed_mps",
            "reaction_margin_s",
            "teacher_rate_min",
            "teacher_rate_max",
            "teacher_rate",
            "scaled_t_hit_s",
            "scaled_t_cycle_s",
            "pre_swing_wait_s",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                ),
            )
        if (
            _validation_mode
            is _DIAGNOSTIC_PREVALIDATED_TASK_RECEIPT
        ):
            self._assert_sample_relations_without_rehash()
            return
        try:
            geometry = _contact_geometry.solve_exact_face_contact(
                ball_contact_w_m=self.ball_contact_w_m,
                racket_face_center_velocity_w_mps=(
                    self.racket_face_center_velocity_w_mps
                ),
                solved_raw_a_normal_w=self.racket_normal_w,
                mount_normal_sign=self.mount_normal_sign,
                reference_racket_quat_wxyz=(
                    self.reference_racket_quat_wxyz
                ),
                reference_racket_angular_velocity_w_radps=(
                    self.reference_racket_angular_velocity_w_radps
                ),
                reference_racket_site_speed_mps=(
                    self.reference_racket_site_speed_mps
                ),
                teacher_rate_min=self.teacher_rate_min,
                teacher_rate_max=self.teacher_rate_max,
            )
        except _contact_geometry.ExactFaceContactGeometryError as error:
            raise ActionBallContractError(
                "task receipt exact-face geometry proof is invalid: "
                f"{error}"
            ) from error
        geometry_vectors = (
            (
                self.racket_site_target_w_m,
                geometry.racket_site_target_w_m,
                "racket_site_target_w_m",
            ),
            (
                self.racket_face_center_velocity_w_mps,
                geometry.racket_face_center_velocity_w_mps,
                "racket_face_center_velocity_w_mps",
            ),
            (
                self.racket_site_velocity_w_mps,
                geometry.racket_site_velocity_w_mps,
                "racket_site_velocity_w_mps",
            ),
            (
                self.racket_command_angular_velocity_w_radps,
                geometry.racket_command_angular_velocity_w_radps,
                "racket_command_angular_velocity_w_radps",
            ),
        )
        for declared, expected, name in geometry_vectors:
            if not _vec3_close(declared, expected, tolerance=1.0e-10):
                raise ActionBallContractError(
                    f"task receipt {name} differs from exact-face geometry"
                )
        if any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-12)
            for a, b in zip(
                self.racket_command_quat_wxyz,
                geometry.racket_command_quat_wxyz,
            )
        ):
            raise ActionBallContractError(
                "task receipt racket command quaternion does not preserve "
                "the reference twist through the deterministic minimal rotation"
            )
        if not math.isclose(
            self.teacher_rate,
            geometry.teacher_rate,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ActionBallContractError(
                "task teacher rate differs from exact face/site angular solve"
            )
        timing = derive_action_teacher_site_timing(
            racket_site_velocity_w_mps=self.racket_site_velocity_w_mps,
            time_to_contact_s=self.time_to_contact_s,
            reference_t_hit_s=self.reference_t_hit_s,
            reference_t_cycle_s=self.reference_t_cycle_s,
            reference_racket_site_speed_mps=(
                self.reference_racket_site_speed_mps
            ),
            reaction_margin_s=self.reaction_margin_s,
            teacher_rate_min=self.teacher_rate_min,
            teacher_rate_max=self.teacher_rate_max,
        )
        declared_timing = (
            self.required_racket_site_speed_mps,
            self.teacher_rate,
            self.scaled_t_hit_s,
            self.scaled_t_cycle_s,
            self.pre_swing_wait_s,
        )
        expected_timing = (
            timing.required_racket_site_speed_mps,
            timing.teacher_rate,
            timing.scaled_t_hit_s,
            timing.scaled_t_cycle_s,
            timing.pre_swing_wait_s,
        )
        if declared_timing != expected_timing:
            raise ActionBallContractError(
                "task teacher timing proof differs from exact unclipped "
                "formula"
            )
        self._assert_sample_relations_without_rehash()
        domain_claim = ActionDomainClaim(
            authority_contract_sha256=self.domain_authority_sha256,
            arm_catalog_sha256=self.arm_catalog_sha256,
            action_uid=self.action_uid,
            domain_epoch=self.domain_epoch,
            domain_levels=self.domain_levels,
            levels_sha256=self.levels_sha256,
            profile_sha256=self.profile_sha256,
            mobility_mode=self.mobility_mode,
        )
        if domain_claim.canonical_sha256 != self.domain_claim_sha256:
            raise ActionBallContractError(
                "task receipt domain claim SHA does not match its fields"
            )
        sample_identity = self._sampler_identity_payload()
        if _sha256_json(sample_identity) != self.sample_sha256:
            raise ActionBallContractError(
                "sampler sample SHA does not match exact ball/base payload"
            )

    def _assert_sample_relations_without_rehash(self) -> None:
        """Check cheap install-critical relations without rebuilding proofs."""

        if (
            self.mobility_mode == "no_move"
            and self.base_goal_w_m != self.base_spawn_w_m
        ):
            raise ActionBallContractError(
                "no_move task requires base_goal_w_m == base_spawn_w_m"
            )
        _assert_yaw_quaternion(
            self.base_yaw_rad,
            self.base_quat_wxyz,
            name="base_quat_wxyz",
        )
        if self.mobility_mode == "move":
            expected_goal = _vec3_add(
                self.base_spawn_w_m,
                _rotate_yaw(
                    self.base_travel_latent_b_yaw_m,
                    self.base_yaw_rad,
                ),
            )
            if not _vec3_close(self.base_goal_w_m, expected_goal):
                raise ActionBallContractError(
                    "move task base goal disagrees with sampled travel"
                )
        expected_contact = _vec3_add(
            self.base_goal_w_m,
            _rotate_yaw(
                self.contact_offset_from_base_goal_b_yaw_m,
                self.base_yaw_rad,
            ),
        )
        if not _vec3_close(self.ball_contact_w_m, expected_contact):
            raise ActionBallContractError(
                "task contact disagrees with sampled base-relative offset"
            )
        expected_velocity = _vec3_scale(
            _rotate_yaw(
                self.incoming_direction_b_yaw, self.base_yaw_rad
            ),
            self.incoming_speed_mps,
        )
        if not _vec3_close(
            self.incoming_velocity_w_mps, expected_velocity
        ):
            raise ActionBallContractError(
                "task incoming velocity disagrees with sampler identity"
            )
        expected_spin = _vec3_scale(
            _rotate_yaw(
                self.spin_direction_b_yaw, self.base_yaw_rad
            ),
            self.spin_magnitude_radps,
        )
        if not _vec3_close(self.incoming_spin_w_radps, expected_spin):
            raise ActionBallContractError(
                "task incoming spin disagrees with sampler identity"
            )

    def _sampler_identity_payload(self) -> Dict[str, object]:
        payload = {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "kind": "swing_sample",
            "sampler_contract_sha256": self.sampler_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "sample_index": self.sample_index,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.to_dict(),
            "birth_id": self.sampler_birth_sha256,
            "profile_sha256": self.profile_sha256,
            "levels_sha256": self.levels_sha256,
            "draw_start": self.sample_draw_start,
            "draw_end": self.sample_draw_end,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_start_w_m": self.base_spawn_w_m,
            "base_spawn_latent_w_m": self.base_spawn_latent_w_m,
            "base_travel_latent_b_yaw_m": (
                self.base_travel_latent_b_yaw_m
            ),
            "base_goal_w_m": self.base_goal_w_m,
            "contact_offset_from_base_goal_b_yaw_m": (
                self.contact_offset_from_base_goal_b_yaw_m
            ),
            "contact_w_m": self.ball_contact_w_m,
            "time_to_contact_s": self.time_to_contact_s,
            "incoming_speed_mps": self.incoming_speed_mps,
            "incoming_direction_b_yaw": self.incoming_direction_b_yaw,
            "incoming_direction_w": _rotate_yaw(
                self.incoming_direction_b_yaw, self.base_yaw_rad
            ),
            "incoming_velocity_w_mps": self.incoming_velocity_w_mps,
            "spin_magnitude_radps": self.spin_magnitude_radps,
            "spin_direction_b_yaw": self.spin_direction_b_yaw,
            "spin_direction_w": _rotate_yaw(
                self.spin_direction_b_yaw, self.base_yaw_rad
            ),
            "spin_w_radps": self.incoming_spin_w_radps,
            "landing_aim_w_xy_m": self.landing_aim_w_xy_m,
        }
        if self.sampling_mixture is None:
            return payload
        return {
            **{
                key: payload[key]
                for key in (
                    "schema_version",
                    "kind",
                    "sampler_contract_sha256",
                    "arm_catalog_sha256",
                    "sample_index",
                    "action_uid",
                    "domain_epoch",
                    "domain_levels",
                )
            },
            "birth_index": self.birth_index,
            "birth_sampling_stratum": (
                self.birth_sampling_stratum
            ),
            "birth_sampling_levels": (
                self.birth_sampling_levels.to_dict()
            ),
            "birth_frontier_arm": self.birth_frontier_arm,
            "sampling_mixture": self.sampling_mixture.to_dict(),
            "sampling_stratum": self.sampling_stratum,
            "sampling_levels": self.sampling_levels.to_dict(),
            "frontier_arm": self.frontier_arm,
            **{
                key: payload[key]
                for key in (
                    "birth_id",
                    "profile_sha256",
                    "levels_sha256",
                    "draw_start",
                    "draw_end",
                    "mobility_mode",
                    "base_yaw_rad",
                    "base_start_w_m",
                    "base_spawn_latent_w_m",
                    "base_travel_latent_b_yaw_m",
                    "base_goal_w_m",
                    "contact_offset_from_base_goal_b_yaw_m",
                    "contact_w_m",
                    "time_to_contact_s",
                    "incoming_speed_mps",
                    "incoming_direction_b_yaw",
                    "incoming_direction_w",
                    "incoming_velocity_w_mps",
                    "spin_magnitude_radps",
                    "spin_direction_b_yaw",
                    "spin_direction_w",
                    "spin_w_radps",
                    "landing_aim_w_xy_m",
                )
            },
            "contact_time_step_s": self.contact_time_step_s,
            "time_to_contact_tick": self.time_to_contact_tick,
        }

    def sampler_identity_receipt(self) -> Dict[str, object]:
        """Return the exact flat proof accepted by ActionBallSampler."""

        return {
            "sample_id": self.sample_sha256,
            **self._sampler_identity_payload(),
        }

    @classmethod
    def from_birth(
        cls,
        birth: ActionBirthReceipt,
        *,
        sample_sha256: str,
        sample_index: int,
        sample_draw_start: int,
        sample_draw_end: int,
        swing_generation: int,
        base_goal_w_m: Sequence[float] | None = None,
        base_spawn_latent_w_m: Sequence[float],
        base_travel_latent_b_yaw_m: Sequence[float],
        contact_offset_from_base_goal_b_yaw_m: Sequence[float],
        ball_contact_w_m: Sequence[float],
        racket_site_target_w_m: Sequence[float],
        time_to_contact_s: float,
        incoming_speed_mps: float,
        incoming_direction_b_yaw: Sequence[float],
        incoming_velocity_w_mps: Sequence[float],
        spin_magnitude_radps: float,
        spin_direction_b_yaw: Sequence[float],
        incoming_spin_w_radps: Sequence[float],
        landing_aim_w_xy_m: Sequence[float],
        mount_normal_sign: int,
        racket_normal_w: Sequence[float],
        reference_racket_quat_wxyz: Sequence[float],
        reference_racket_angular_velocity_w_radps: Sequence[float],
        racket_command_quat_wxyz: Sequence[float],
        racket_face_center_velocity_w_mps: Sequence[float],
        racket_site_velocity_w_mps: Sequence[float],
        racket_command_angular_velocity_w_radps: Sequence[float],
        geometry_source_sha256: str,
        reference_t_hit_s: float,
        reference_t_cycle_s: float,
        reference_racket_site_speed_mps: float,
        required_racket_site_speed_mps: float,
        reaction_margin_s: float,
        teacher_rate_min: float,
        teacher_rate_max: float,
        teacher_rate: float,
        scaled_t_hit_s: float,
        scaled_t_cycle_s: float,
        pre_swing_wait_s: float,
        solver_residual_m: float,
        contact_time_step_s: float | None = None,
        time_to_contact_tick: int | None = None,
        birth_index: int = -1,
        birth_sampling_stratum: str = "domain",
        birth_sampling_levels: ActionDomainLevels | None = None,
        birth_frontier_arm: str | None = None,
        sampling_mixture: ActionSamplingMixture | None = None,
        sampling_stratum: str = "domain",
        sampling_levels: ActionDomainLevels | None = None,
        frontier_arm: str | None = None,
        counter_rally_task: CounterRallyTaskIdentity | None = None,
        _validation_mode: object = _FULL_TASK_RECEIPT_VALIDATION,
    ) -> "ActionBallTaskReceipt":
        if not isinstance(birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "from_birth requires an ActionBirthReceipt"
            )
        if birth.mobility_mode == "move" and base_goal_w_m is None:
            raise ActionBallContractError(
                "move task requires an explicit per-swing base_goal_w_m "
                "from the sampled base travel"
            )
        return cls(
            birth_sha256=birth.canonical_sha256,
            registry_sha256=birth.registry_sha256,
            sample_sha256=sample_sha256,
            env_id=birth.env_id,
            reset_generation=birth.reset_generation,
            swing_generation=swing_generation,
            action_uid=birth.action_uid,
            action_slot=birth.action_slot,
            domain_epoch=birth.domain_epoch,
            domain_claim_sha256=birth.domain_claim_sha256,
            domain_authority_sha256=birth.domain_authority_sha256,
            domain_levels=birth.domain_levels,
            arm_catalog_sha256=birth.arm_catalog_sha256,
            levels_sha256=birth.levels_sha256,
            sampler_birth_sha256=birth.sampler_birth_sha256,
            mobility_mode=birth.mobility_mode,
            base_yaw_rad=birth.base_yaw_rad,
            base_quat_wxyz=birth.base_quat_wxyz,
            base_spawn_w_m=birth.base_spawn_w_m,
            base_goal_w_m=(
                birth.base_spawn_w_m
                if base_goal_w_m is None
                else tuple(base_goal_w_m)
            ),
            sample_index=sample_index,
            sample_draw_start=sample_draw_start,
            sample_draw_end=sample_draw_end,
            base_spawn_latent_w_m=tuple(base_spawn_latent_w_m),
            base_travel_latent_b_yaw_m=tuple(
                base_travel_latent_b_yaw_m
            ),
            contact_offset_from_base_goal_b_yaw_m=tuple(
                contact_offset_from_base_goal_b_yaw_m
            ),
            ball_contact_w_m=tuple(ball_contact_w_m),
            racket_site_target_w_m=tuple(racket_site_target_w_m),
            time_to_contact_s=time_to_contact_s,
            contact_time_step_s=contact_time_step_s,
            time_to_contact_tick=time_to_contact_tick,
            incoming_speed_mps=incoming_speed_mps,
            incoming_direction_b_yaw=tuple(
                incoming_direction_b_yaw
            ),
            incoming_velocity_w_mps=tuple(incoming_velocity_w_mps),
            spin_magnitude_radps=spin_magnitude_radps,
            spin_direction_b_yaw=tuple(spin_direction_b_yaw),
            incoming_spin_w_radps=tuple(incoming_spin_w_radps),
            landing_aim_w_xy_m=tuple(landing_aim_w_xy_m),
            mount_normal_sign=mount_normal_sign,
            racket_normal_w=tuple(racket_normal_w),
            reference_racket_quat_wxyz=tuple(
                reference_racket_quat_wxyz
            ),
            reference_racket_angular_velocity_w_radps=tuple(
                reference_racket_angular_velocity_w_radps
            ),
            racket_command_quat_wxyz=tuple(
                racket_command_quat_wxyz
            ),
            racket_face_center_velocity_w_mps=tuple(
                racket_face_center_velocity_w_mps
            ),
            racket_site_velocity_w_mps=tuple(
                racket_site_velocity_w_mps
            ),
            racket_command_angular_velocity_w_radps=tuple(
                racket_command_angular_velocity_w_radps
            ),
            geometry_source_sha256=geometry_source_sha256,
            reference_t_hit_s=reference_t_hit_s,
            reference_t_cycle_s=reference_t_cycle_s,
            reference_racket_site_speed_mps=(
                reference_racket_site_speed_mps
            ),
            required_racket_site_speed_mps=(
                required_racket_site_speed_mps
            ),
            reaction_margin_s=reaction_margin_s,
            teacher_rate_min=teacher_rate_min,
            teacher_rate_max=teacher_rate_max,
            teacher_rate=teacher_rate,
            scaled_t_hit_s=scaled_t_hit_s,
            scaled_t_cycle_s=scaled_t_cycle_s,
            pre_swing_wait_s=pre_swing_wait_s,
            solver_residual_m=solver_residual_m,
            manifest_sha256=birth.manifest_sha256,
            sampler_sha256=birth.sampler_sha256,
            profile_sha256=birth.profile_sha256,
            motion_sha256=birth.motion_sha256,
            physics_sha256=birth.physics_sha256,
            solver_sha256=birth.solver_sha256,
            birth_index=birth_index,
            birth_sampling_stratum=birth_sampling_stratum,
            birth_sampling_levels=birth_sampling_levels,
            birth_frontier_arm=birth_frontier_arm,
            sampling_mixture=sampling_mixture,
            sampling_stratum=sampling_stratum,
            sampling_levels=sampling_levels,
            frontier_arm=frontier_arm,
            counter_rally_task=counter_rally_task,
            initial_center_single_question=(
                birth.initial_center_single_question
            ),
            _validation_mode=_validation_mode,
        )

    def payload_dict(self) -> Dict[str, object]:
        payload = {
            "schema_version": TASK_RECEIPT_SCHEMA_VERSION,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "registry_sha256": self.registry_sha256,
            "birth_sha256": self.birth_sha256,
            "sample_sha256": self.sample_sha256,
            "env_id": self.env_id,
            "reset_generation": self.reset_generation,
            "swing_generation": self.swing_generation,
            "action_uid": self.action_uid,
            "action_slot": self.action_slot,
            "domain_epoch": self.domain_epoch,
            "domain_claim_sha256": self.domain_claim_sha256,
            "domain_authority_sha256": self.domain_authority_sha256,
            "domain_levels": self.domain_levels.to_dict(),
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "levels_sha256": self.levels_sha256,
            "sampler_birth_sha256": self.sampler_birth_sha256,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_quat_wxyz": list(self.base_quat_wxyz),
            "base_spawn_w_m": list(self.base_spawn_w_m),
            "base_goal_w_m": list(self.base_goal_w_m),
            "sample_index": self.sample_index,
            "sample_draw_start": self.sample_draw_start,
            "sample_draw_end": self.sample_draw_end,
            "base_spawn_latent_w_m": list(
                self.base_spawn_latent_w_m
            ),
            "base_travel_latent_b_yaw_m": list(
                self.base_travel_latent_b_yaw_m
            ),
            "contact_offset_from_base_goal_b_yaw_m": list(
                self.contact_offset_from_base_goal_b_yaw_m
            ),
            "ball_contact_w_m": list(self.ball_contact_w_m),
            "racket_site_target_w_m": list(
                self.racket_site_target_w_m
            ),
            "time_to_contact_s": self.time_to_contact_s,
            "incoming_speed_mps": self.incoming_speed_mps,
            "incoming_direction_b_yaw": list(
                self.incoming_direction_b_yaw
            ),
            "incoming_velocity_w_mps": list(
                self.incoming_velocity_w_mps
            ),
            "spin_magnitude_radps": self.spin_magnitude_radps,
            "spin_direction_b_yaw": list(
                self.spin_direction_b_yaw
            ),
            "incoming_spin_w_radps": list(self.incoming_spin_w_radps),
            "landing_aim_w_xy_m": list(self.landing_aim_w_xy_m),
            "mount_normal_sign": self.mount_normal_sign,
            "racket_normal_w": list(self.racket_normal_w),
            "reference_racket_quat_wxyz": list(
                self.reference_racket_quat_wxyz
            ),
            "reference_racket_angular_velocity_w_radps": list(
                self.reference_racket_angular_velocity_w_radps
            ),
            "racket_command_quat_wxyz": list(
                self.racket_command_quat_wxyz
            ),
            "racket_face_center_velocity_w_mps": list(
                self.racket_face_center_velocity_w_mps
            ),
            "racket_site_velocity_w_mps": list(
                self.racket_site_velocity_w_mps
            ),
            "racket_command_angular_velocity_w_radps": list(
                self.racket_command_angular_velocity_w_radps
            ),
            "geometry_source_sha256": self.geometry_source_sha256,
            "reference_t_hit_s": self.reference_t_hit_s,
            "reference_t_cycle_s": self.reference_t_cycle_s,
            "reference_racket_site_speed_mps": (
                self.reference_racket_site_speed_mps
            ),
            "required_racket_site_speed_mps": (
                self.required_racket_site_speed_mps
            ),
            "reaction_margin_s": self.reaction_margin_s,
            "teacher_rate_min": self.teacher_rate_min,
            "teacher_rate_max": self.teacher_rate_max,
            "teacher_rate": self.teacher_rate,
            "scaled_t_hit_s": self.scaled_t_hit_s,
            "scaled_t_cycle_s": self.scaled_t_cycle_s,
            "pre_swing_wait_s": self.pre_swing_wait_s,
            "solver_residual_m": self.solver_residual_m,
            "manifest_sha256": self.manifest_sha256,
            "sampler_sha256": self.sampler_sha256,
            "profile_sha256": self.profile_sha256,
            "motion_sha256": self.motion_sha256,
            "physics_sha256": self.physics_sha256,
            "solver_sha256": self.solver_sha256,
        }
        if self.counter_rally_task is not None:
            payload["counter_rally_task"] = (
                self.counter_rally_task.to_dict()
            )
        if self.sampling_mixture is None:
            return payload
        payload = {
            **payload,
            "birth_index": self.birth_index,
            "birth_sampling_stratum": (
                self.birth_sampling_stratum
            ),
            "birth_sampling_levels": (
                self.birth_sampling_levels.to_dict()
            ),
            "birth_frontier_arm": self.birth_frontier_arm,
            "sampling_mixture": self.sampling_mixture.to_dict(),
            "sampling_stratum": self.sampling_stratum,
            "sampling_levels": self.sampling_levels.to_dict(),
            "frontier_arm": self.frontier_arm,
            "contact_time_step_s": self.contact_time_step_s,
            "time_to_contact_tick": self.time_to_contact_tick,
        }
        if self.initial_center_single_question:
            payload["initial_center_single_question"] = True
        return payload

    @_WeakIdentityCachedCanonicalSha256
    def canonical_sha256(self) -> str:
        return _sha256_json(self.payload_dict())

    def task_ref(self) -> ActionTaskReceiptRef:
        return ActionTaskReceiptRef(
            env_id=self.env_id,
            reset_generation=self.reset_generation,
            swing_generation=self.swing_generation,
            action_uid=self.action_uid,
            action_slot=self.action_slot,
            birth_sha256=self.birth_sha256,
            sample_sha256=self.sample_sha256,
            task_sha256=self.canonical_sha256,
        )

    def to_dict(self) -> Dict[str, object]:
        result = self.payload_dict()
        result["canonical_sha256"] = self.canonical_sha256
        return result

    @classmethod
    def from_dict(cls, value: object) -> "ActionBallTaskReceipt":
        has_mixture = (
            isinstance(value, Mapping)
            and "sampling_mixture" in value
        )
        has_counter_rally_task = (
            isinstance(value, Mapping)
            and "counter_rally_task" in value
        )
        has_initial_center = (
            isinstance(value, Mapping)
            and "initial_center_single_question" in value
        )
        if has_initial_center and not has_mixture:
            raise ActionBallContractError(
                "legacy task cannot carry mixture/tick metadata"
            )
        if has_initial_center and has_counter_rally_task:
            payload_keys = (
                _INITIAL_CENTER_COUNTER_RALLY_MIXTURE_TASK_PAYLOAD_KEYS
            )
        elif has_initial_center:
            payload_keys = _INITIAL_CENTER_MIXTURE_TASK_PAYLOAD_KEYS
        elif has_mixture and has_counter_rally_task:
            payload_keys = _COUNTER_RALLY_MIXTURE_TASK_PAYLOAD_KEYS
        elif has_mixture:
            payload_keys = _MIXTURE_TASK_PAYLOAD_KEYS
        elif has_counter_rally_task:
            payload_keys = _COUNTER_RALLY_TASK_PAYLOAD_KEYS
        else:
            payload_keys = _TASK_PAYLOAD_KEYS
        row = _exact_mapping(
            value,
            (*payload_keys, "canonical_sha256"),
            name="action-ball task receipt",
        )
        if row["schema_version"] != TASK_RECEIPT_SCHEMA_VERSION:
            raise ActionBallContractError(
                "unsupported action-ball task receipt schema_version"
            )
        if has_initial_center and row[
            "initial_center_single_question"
        ] is not True:
            raise ActionBallContractError(
                "task initial_center_single_question is only written when true"
            )
        if row["runtime_contract_sha256"] != RUNTIME_CONTRACT_SHA256:
            raise ActionBallContractError(
                "action-ball task receipt runtime contract SHA mismatch"
            )
        fields = {
            name: row[name]
            for name in payload_keys
            if name not in ("schema_version", "runtime_contract_sha256")
        }
        fields["domain_levels"] = ActionDomainLevels.from_dict(
            row["domain_levels"]
        )
        if has_counter_rally_task:
            fields["counter_rally_task"] = (
                CounterRallyTaskIdentity.from_dict(
                    row["counter_rally_task"]
                )
            )
        if has_mixture:
            fields["birth_sampling_levels"] = (
                ActionDomainLevels.from_dict(
                    row["birth_sampling_levels"]
                )
            )
            fields["sampling_mixture"] = (
                ActionSamplingMixture.from_dict(
                    row["sampling_mixture"]
                )
            )
            fields["sampling_levels"] = ActionDomainLevels.from_dict(
                row["sampling_levels"]
            )
        receipt = cls(**fields)  # type: ignore[arg-type]
        declared = _sha256(
            row["canonical_sha256"], name="canonical_sha256"
        )
        if declared != receipt.canonical_sha256:
            raise ActionBallContractError(
                "action-ball task receipt canonical SHA mismatch"
            )
        return receipt

    def require_counter_rally_task(
        self,
        *,
        expected_objective_profile_sha256: str,
    ) -> CounterRallyTaskIdentity:
        """Return the exact N=1 task identity or hard-stop on drift."""

        expected = _sha256(
            expected_objective_profile_sha256,
            name="expected_objective_profile_sha256",
        )
        identity = self.counter_rally_task
        if identity is None:
            raise CounterRallyTaskIdentityError(
                "counter-rally task identity is missing"
            )
        if identity.objective_profile_sha256 != expected:
            raise CounterRallyTaskIdentityError(
                "counter-rally objective profile SHA mismatch"
            )
        return identity

    def assert_contract(
        self,
        *,
        binding: ActionBinding,
        pins: RuntimePins,
        mobility_mode: str,
        registry_sha256: str,
    ) -> None:
        if (
            self.action_uid != binding.action_uid
            or self.action_slot != binding.action_slot
            or self.motion_sha256 != binding.motion_sha256
            or self.profile_sha256 != binding.profile_sha256
        ):
            raise ActionBallContractError(
                "task receipt does not match its manifest action binding"
            )
        if (
            self.manifest_sha256 != pins.manifest_sha256
            or self.sampler_sha256 != pins.sampler_sha256
            or self.domain_authority_sha256
            != pins.domain_authority_sha256
            or self.physics_sha256 != pins.physics_sha256
            or self.solver_sha256 != pins.solver_sha256
        ):
            raise ActionBallContractError(
                "task receipt does not match run-global pins"
            )
        expected_counter_rally_objective = (
            pins.counter_rally_objective_profile_sha256
        )
        if expected_counter_rally_objective is None:
            if self.counter_rally_task is not None:
                raise CounterRallyTaskIdentityError(
                    "ordinary task/run pins cannot carry counter-rally identity"
                )
        else:
            self.require_counter_rally_task(
                expected_objective_profile_sha256=(
                    expected_counter_rally_objective
                )
            )
        if self.mobility_mode != _mode(mobility_mode):
            raise ActionBallContractError(
                "task receipt mobility mode differs from frozen run mode"
            )
        if self.registry_sha256 != _sha256(
            registry_sha256, name="registry_sha256"
        ):
            raise ActionBallContractError(
                "task receipt registry SHA differs from the bound registry"
            )

    def assert_birth(self, birth: ActionBirthReceipt) -> None:
        if not isinstance(birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "task birth binding requires ActionBirthReceipt"
            )
        if self.birth_sha256 != birth.canonical_sha256:
            raise ActionBallContractError("task receipt birth SHA mismatch")
        expected = (
            birth.env_id,
            birth.reset_generation,
            birth.action_uid,
            birth.action_slot,
            birth.domain_epoch,
            birth.domain_claim_sha256,
            birth.domain_authority_sha256,
            birth.domain_levels,
            birth.arm_catalog_sha256,
            birth.levels_sha256,
            birth.sampler_birth_sha256,
            birth.mobility_mode,
            birth.base_yaw_rad,
            birth.base_quat_wxyz,
            birth.base_spawn_w_m,
            birth.manifest_sha256,
            birth.sampler_sha256,
            birth.profile_sha256,
            birth.motion_sha256,
            birth.physics_sha256,
            birth.solver_sha256,
            birth.registry_sha256,
        )
        actual = (
            self.env_id,
            self.reset_generation,
            self.action_uid,
            self.action_slot,
            self.domain_epoch,
            self.domain_claim_sha256,
            self.domain_authority_sha256,
            self.domain_levels,
            self.arm_catalog_sha256,
            self.levels_sha256,
            self.sampler_birth_sha256,
            self.mobility_mode,
            self.base_yaw_rad,
            self.base_quat_wxyz,
            self.base_spawn_w_m,
            self.manifest_sha256,
            self.sampler_sha256,
            self.profile_sha256,
            self.motion_sha256,
            self.physics_sha256,
            self.solver_sha256,
            self.registry_sha256,
        )
        if actual != expected:
            raise ActionBallContractError(
                "task receipt duplicates fields that disagree with birth"
            )
        if self.sample_draw_start < birth.sampler_draw_end:
            raise ActionBallContractError(
                "task sample draw range predates/overlaps its sampler birth"
            )


_DIAGNOSTIC_TASK_RECEIPT_KWARG_NAMES = (
    "sample_sha256",
    "sample_index",
    "sample_draw_start",
    "sample_draw_end",
    "swing_generation",
    "base_goal_w_m",
    "base_spawn_latent_w_m",
    "base_travel_latent_b_yaw_m",
    "contact_offset_from_base_goal_b_yaw_m",
    "ball_contact_w_m",
    "racket_site_target_w_m",
    "time_to_contact_s",
    "incoming_speed_mps",
    "incoming_direction_b_yaw",
    "incoming_velocity_w_mps",
    "spin_magnitude_radps",
    "spin_direction_b_yaw",
    "incoming_spin_w_radps",
    "landing_aim_w_xy_m",
    "mount_normal_sign",
    "racket_normal_w",
    "reference_racket_quat_wxyz",
    "reference_racket_angular_velocity_w_radps",
    "racket_command_quat_wxyz",
    "racket_face_center_velocity_w_mps",
    "racket_site_velocity_w_mps",
    "racket_command_angular_velocity_w_radps",
    "geometry_source_sha256",
    "reference_t_hit_s",
    "reference_t_cycle_s",
    "reference_racket_site_speed_mps",
    "required_racket_site_speed_mps",
    "reaction_margin_s",
    "teacher_rate_min",
    "teacher_rate_max",
    "teacher_rate",
    "scaled_t_hit_s",
    "scaled_t_cycle_s",
    "pre_swing_wait_s",
    "solver_residual_m",
    "contact_time_step_s",
    "time_to_contact_tick",
    "birth_index",
    "birth_sampling_stratum",
    "birth_sampling_levels",
    "birth_frontier_arm",
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    "counter_rally_task",
)
_DIAGNOSTIC_TASK_RECEIPT_KWARG_SET = frozenset(
    _DIAGNOSTIC_TASK_RECEIPT_KWARG_NAMES
)
_DIAGNOSTIC_TASK_RECEIPT_BIRTH_FIELD_NAMES = (
    "birth_sha256",
    "env_id",
    "reset_generation",
    "action_uid",
    "action_slot",
    "domain_epoch",
    "domain_claim_sha256",
    "domain_authority_sha256",
    "domain_levels",
    "arm_catalog_sha256",
    "levels_sha256",
    "sampler_birth_sha256",
    "mobility_mode",
    "base_yaw_rad",
    "base_quat_wxyz",
    "base_spawn_w_m",
    "manifest_sha256",
    "sampler_sha256",
    "profile_sha256",
    "motion_sha256",
    "physics_sha256",
    "solver_sha256",
    "registry_sha256",
    "initial_center_single_question",
)
_DIAGNOSTIC_TASK_RECEIPT_SEQUENCE_KWARGS = (
    "base_goal_w_m",
    "base_spawn_latent_w_m",
    "base_travel_latent_b_yaw_m",
    "contact_offset_from_base_goal_b_yaw_m",
    "ball_contact_w_m",
    "racket_site_target_w_m",
    "incoming_direction_b_yaw",
    "incoming_velocity_w_mps",
    "spin_direction_b_yaw",
    "incoming_spin_w_radps",
    "landing_aim_w_xy_m",
    "racket_normal_w",
    "reference_racket_quat_wxyz",
    "reference_racket_angular_velocity_w_radps",
    "racket_command_quat_wxyz",
    "racket_face_center_velocity_w_mps",
    "racket_site_velocity_w_mps",
    "racket_command_angular_velocity_w_radps",
)
_DIAGNOSTIC_TASK_RECEIPT_STORAGE_FIELDS = tuple(
    name
    for name in ActionBallTaskReceipt.__annotations__
    if name != "_validation_mode"
)
if (
    _DIAGNOSTIC_TASK_RECEIPT_KWARG_SET
    | frozenset(_DIAGNOSTIC_TASK_RECEIPT_BIRTH_FIELD_NAMES)
    != frozenset(_DIAGNOSTIC_TASK_RECEIPT_STORAGE_FIELDS)
):
    raise RuntimeError(
        "diagnostic task receipt constructor fields drifted from dataclass"
    )


def _diagnostic_prevalidated_task_receipt_from_birth(
    birth: ActionBirthReceipt,
    **kwargs: object,
) -> ActionBallTaskReceipt:
    """Materialize one producer-validated diagnostic row without re-proving it.

    The sole production caller has already admitted the immutable sampler row,
    solved exact-face geometry/timing, and conserved proposal reasons in one
    batch.  Re-entering ``ActionBallTaskReceipt.__post_init__`` here repeated
    all of that work once per admitted environment.  This constructor retains
    the exact frozen base type and wire payload while making producer drift
    fail closed through an exact keyword contract.
    """

    if not isinstance(birth, ActionBirthReceipt):
        raise ActionBallContractError(
            "diagnostic task receipt requires an ActionBirthReceipt"
        )
    if kwargs.keys() != _DIAGNOSTIC_TASK_RECEIPT_KWARG_SET:
        provided = frozenset(kwargs)
        missing = sorted(
            _DIAGNOSTIC_TASK_RECEIPT_KWARG_SET - provided
        )
        extra = sorted(provided - _DIAGNOSTIC_TASK_RECEIPT_KWARG_SET)
        raise ActionBallContractError(
            "diagnostic task receipt requires the exact producer keyword "
            f"set; missing={missing!r}; extra={extra!r}"
        )
    if birth.mobility_mode == "move" and kwargs["base_goal_w_m"] is None:
        raise ActionBallContractError(
            "move task requires an explicit per-swing base_goal_w_m "
            "from the sampled base travel"
        )

    if kwargs["base_goal_w_m"] is None:
        kwargs["base_goal_w_m"] = birth.base_spawn_w_m
    for name in _DIAGNOSTIC_TASK_RECEIPT_SEQUENCE_KWARGS:
        kwargs[name] = tuple(kwargs[name])  # type: ignore[arg-type]
    if kwargs["sampling_mixture"] is None:
        # Match the legacy normalization performed by the formal constructor.
        kwargs["birth_sampling_levels"] = birth.domain_levels
        kwargs["sampling_levels"] = birth.domain_levels

    kwargs.update(
        birth_sha256=birth.canonical_sha256,
        env_id=birth.env_id,
        reset_generation=birth.reset_generation,
        action_uid=birth.action_uid,
        action_slot=birth.action_slot,
        domain_epoch=birth.domain_epoch,
        domain_claim_sha256=birth.domain_claim_sha256,
        domain_authority_sha256=birth.domain_authority_sha256,
        domain_levels=birth.domain_levels,
        arm_catalog_sha256=birth.arm_catalog_sha256,
        levels_sha256=birth.levels_sha256,
        sampler_birth_sha256=birth.sampler_birth_sha256,
        mobility_mode=birth.mobility_mode,
        base_yaw_rad=birth.base_yaw_rad,
        base_quat_wxyz=birth.base_quat_wxyz,
        base_spawn_w_m=birth.base_spawn_w_m,
        manifest_sha256=birth.manifest_sha256,
        sampler_sha256=birth.sampler_sha256,
        profile_sha256=birth.profile_sha256,
        motion_sha256=birth.motion_sha256,
        physics_sha256=birth.physics_sha256,
        solver_sha256=birth.solver_sha256,
        registry_sha256=birth.registry_sha256,
        initial_center_single_question=(
            birth.initial_center_single_question
        ),
    )
    receipt = object.__new__(ActionBallTaskReceipt)
    for name in _DIAGNOSTIC_TASK_RECEIPT_STORAGE_FIELDS:
        object.__setattr__(receipt, name, kwargs[name])
    return receipt


@dataclass(frozen=True)
class _PendingBirth:
    receipt: ActionBirthReceipt
    status: str

    def __post_init__(self) -> None:
        if self.status not in ("reserved", "committed"):
            raise ActionBallContractError(
                "pending birth status must be reserved or committed"
            )


@dataclass(frozen=True)
class BirthReserveRequest:
    """One Motion-side birth reservation claim for an atomic env batch."""

    env_id: int
    reset_generation: int
    action_uid: int
    action_slot: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )


@dataclass(frozen=True)
class BirthCommitRequest:
    """One Motion-side post-root-write commit claim."""

    env_id: int
    reset_generation: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _sha256(self.receipt_sha256, name="receipt_sha256"),
        )


@dataclass(frozen=True)
class BirthConsumeRequest:
    """One Racket-side consume claim, suitable for atomic env batches."""

    env_id: int
    reset_generation: int
    action_uid: int
    action_slot: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        BirthReserveRequest(
            env_id=self.env_id,
            reset_generation=self.reset_generation,
            action_uid=self.action_uid,
            action_slot=self.action_slot,
        )
        object.__setattr__(
            self, "env_id", _plain_int(self.env_id, name="env_id")
        )
        object.__setattr__(
            self,
            "reset_generation",
            _plain_int(
                self.reset_generation,
                name="reset_generation",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _sha256(self.receipt_sha256, name="receipt_sha256"),
        )


class ActionBirthBroker:
    """Per-env single-consumer broker for true-reset base births."""

    _STATE_KEYS = (
        "schema_version",
        "runtime_contract_sha256",
        "registry_sha256",
        "pins",
        "mobility_mode",
        "bindings",
        "domain_authority_contract_sha256",
        "domain_authority_state_owner_sha256",
        "domain_authority_state",
        "domain_authority_state_sha256",
        "provider_contract_sha256",
        "provider_state_owner_sha256",
        "provider_state",
        "provider_state_sha256",
        "domain_claim_counts",
        "last_sampler_birth_indices",
        "last_sampler_draw_ends",
        "last_generations",
        "consumed_generations",
        "consumed_receipts",
        "pending",
        "integrity_sha256",
    )

    def __init__(
        self,
        bindings: Sequence[ActionBinding],
        pins: RuntimePins,
        mobility_mode: str,
        *,
        diagnostic_unauthorized: bool = False,
    ) -> None:
        self._bindings = _validate_bindings(bindings)
        if not isinstance(pins, RuntimePins):
            raise ActionBallContractError("pins must be RuntimePins")
        self._pins = pins
        if type(diagnostic_unauthorized) is not bool:
            raise ActionBallContractError(
                "diagnostic_unauthorized must be an exact boolean"
            )
        self._diagnostic_fast_path = diagnostic_unauthorized
        if (
            pins.counter_rally_objective_profile_sha256 is not None
            and len(self._bindings) != 1
        ):
            raise CounterRallyTaskIdentityError(
                "counter-rally objective pin requires exact N=1 bindings"
            )
        self._mobility_mode = _mode(mobility_mode)
        self._by_uid = {
            binding.action_uid: binding for binding in self._bindings
        }
        self._by_slot = {
            binding.action_slot: binding for binding in self._bindings
        }
        self._registry_sha256 = _registry_sha256(
            self._bindings, self._pins, self._mobility_mode
        )
        self._provider: Callable[
            [ActionBirthRequest], ActionBirthReceipt
        ] | None = None
        self._domain_authority: ActionDomainClaimAuthority | None = None
        self._last_sampler_birth_index: Dict[int, int] = {}
        self._last_sampler_draw_end: Dict[int, int] = {}
        self._domain_claim_count: Dict[int, int] = {}
        self._last_generation: Dict[int, int] = {}
        self._consumed_generation: Dict[int, int] = {}
        self._consumed_receipts: Dict[
            Tuple[int, int], ActionBirthReceipt
        ] = {}
        # Diagnostic checkpoints deliberately do not contain exact ActionBall
        # resume state.  Keep only the live env's latest immutable receipt in
        # that mode instead of maintaining both an env/generation transcript
        # and a second env->tuple-key index on every short episode.
        self._diagnostic_consumed_receipt_by_env: Dict[
            int, ActionBirthReceipt
        ] = {}
        self._pending: Dict[int, _PendingBirth] = {}

    @property
    def ordered_action_uids(self) -> Tuple[int, ...]:
        return tuple(binding.action_uid for binding in self._bindings)

    @property
    def action_count(self) -> int:
        return len(self._bindings)

    @property
    def diagnostic_fast_path(self) -> bool:
        return self._diagnostic_fast_path

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256

    @property
    def provider_state_owner_sha256(self) -> str:
        return self._provider_state_owner_sha256()

    def provider_state_snapshot(self) -> object:
        return self._provider_state()

    def binding_for_slot(self, action_slot: int) -> ActionBinding:
        slot = _plain_int(action_slot, name="action_slot")
        try:
            return self._by_slot[slot]
        except KeyError as exc:
            raise ActionBallContractError(
                f"unknown action_slot {slot}"
            ) from exc

    def binding_for_uid(self, action_uid: int) -> ActionBinding:
        uid = _plain_int(
            action_uid,
            name="action_uid",
            minimum=1,
            maximum=MAX_ACTION_UID,
        )
        try:
            return self._by_uid[uid]
        except KeyError as exc:
            raise ActionBallContractError(
                f"unknown action_uid {uid}"
            ) from exc

    def _binding(self, action_uid: int, action_slot: int) -> ActionBinding:
        by_uid = self.binding_for_uid(action_uid)
        by_slot = self.binding_for_slot(action_slot)
        if by_uid != by_slot:
            raise ActionBallContractError(
                f"action_uid {action_uid} is not bound to slot {action_slot}"
            )
        return by_uid

    def bind_provider(self, provider: ActionBirthProvider) -> None:
        if self._provider is not None:
            raise BirthProtocolError("birth provider may be bound only once")
        if not callable(provider):
            raise ActionBallContractError("birth provider must be callable")
        if (
            _sha256(
                getattr(provider, "sampler_contract_sha256", None),
                name="provider.sampler_contract_sha256",
            )
            != self._pins.sampler_sha256
        ):
            raise ActionBallContractError(
                "birth provider sampler contract differs from runtime pins"
            )
        _sha256(
            getattr(provider, "state_owner_sha256", None),
            name="provider.state_owner_sha256",
        )
        if (
            not callable(getattr(provider, "state_dict", None))
            or not callable(getattr(provider, "load_state_dict", None))
            or not callable(
                getattr(provider, "assert_issued_birth", None)
            )
            or not callable(
                getattr(provider, "birth_highwater_for", None)
            )
        ):
            raise ActionBallContractError(
                "birth provider must implement atomic state_dict/"
                "load_state_dict, exact assert_issued_birth, and "
                "birth_highwater_for"
            )
        _json_data(provider.state_dict(), name="birth provider state")
        self._provider = provider

    def bind_domain_claim_authority(
        self, authority: ActionDomainClaimAuthority
    ) -> None:
        if self._domain_authority is not None:
            raise BirthProtocolError(
                "domain claim authority may be bound only once"
            )
        if (
            not callable(getattr(authority, "claim_for_action", None))
            or not callable(
                getattr(authority, "domain_cursor_for", None)
            )
        ):
            raise ActionBallContractError(
                "domain claim authority must implement claim_for_action() "
                "and domain_cursor_for()"
            )
        if (
            _sha256(
                getattr(
                    authority,
                    "domain_authority_contract_sha256",
                    None,
                ),
                name="authority.domain_authority_contract_sha256",
            )
            != self._pins.domain_authority_sha256
        ):
            raise ActionBallContractError(
                "domain claim authority contract differs from runtime pins"
            )
        _sha256(
            getattr(authority, "state_owner_sha256", None),
            name="authority.state_owner_sha256",
        )
        if not callable(getattr(authority, "state_dict", None)) or not callable(
            getattr(authority, "load_state_dict", None)
        ):
            raise ActionBallContractError(
                "domain claim authority must implement atomic "
                "state_dict/load_state_dict"
            )
        _json_data(
            authority.state_dict(), name="domain claim authority state"
        )
        self._domain_authority = authority

    def _provider_state(self) -> object:
        if self._provider is None:
            raise BirthProtocolError("birth provider is not bound")
        return _json_data(
            self._provider.state_dict(), name="birth provider state"
        )

    def _restore_provider_state(self, state: object) -> None:
        if self._provider is None:
            raise BirthProtocolError("birth provider is not bound")
        detached = _json_data(state, name="birth provider state")
        self._provider.load_state_dict(detached)
        if self._provider_state() != detached:
            raise ActionBallContractError(
                "birth provider load_state_dict did not restore exact state"
            )

    def _domain_authority_state(self) -> object:
        if self._domain_authority is None:
            raise BirthProtocolError("domain claim authority is not bound")
        return _json_data(
            self._domain_authority.state_dict(),
            name="domain claim authority state",
        )

    def _restore_domain_authority_state(self, state: object) -> None:
        if self._domain_authority is None:
            raise BirthProtocolError("domain claim authority is not bound")
        detached = _json_data(
            state, name="domain claim authority state"
        )
        self._domain_authority.load_state_dict(detached)
        if self._domain_authority_state() != detached:
            raise ActionBallContractError(
                "domain claim authority load_state_dict did not restore "
                "exact state"
            )

    def _provider_state_owner_sha256(self) -> str:
        if self._provider is None:
            raise BirthProtocolError("birth provider is not bound")
        return _sha256(
            getattr(self._provider, "state_owner_sha256", None),
            name="provider.state_owner_sha256",
        )

    def _domain_authority_state_owner_sha256(self) -> str:
        if self._domain_authority is None:
            raise BirthProtocolError("domain claim authority is not bound")
        return _sha256(
            getattr(self._domain_authority, "state_owner_sha256", None),
            name="authority.state_owner_sha256",
        )

    def _callback_states(self) -> Tuple[object, object]:
        provider_state = self._provider_state()
        authority_state = self._domain_authority_state()
        if (
            self._provider_state_owner_sha256()
            == self._domain_authority_state_owner_sha256()
            and provider_state != authority_state
        ):
            raise ActionBallContractError(
                "callbacks sharing one state owner must expose byte-identical "
                "JSON states"
            )
        return provider_state, authority_state

    def _restore_callback_states(
        self, provider_state: object, authority_state: object
    ) -> None:
        shared_owner = (
            self._provider_state_owner_sha256()
            == self._domain_authority_state_owner_sha256()
        )
        if shared_owner:
            if provider_state != authority_state:
                raise ActionBallContractError(
                    "shared callback owner checkpoint states disagree"
                )
            self._restore_domain_authority_state(authority_state)
            if self._provider_state() != provider_state:
                raise ActionBallContractError(
                    "shared callback owner restore is not visible through "
                    "the provider"
                )
            return
        provider_error = None
        authority_error = None
        try:
            self._restore_provider_state(provider_state)
        except Exception as exc:  # pragma: no cover - catastrophic adapter bug
            provider_error = exc
        try:
            self._restore_domain_authority_state(authority_state)
        except Exception as exc:  # pragma: no cover - catastrophic adapter bug
            authority_error = exc
        if provider_error is not None or authority_error is not None:
            raise BirthProtocolError(
                "provider/domain authority exact state restore failed"
            ) from (provider_error or authority_error)

    def _provider_birth_highwater(
        self, action_uid: int
    ) -> Tuple[int, int]:
        if self._provider is None:
            raise BirthProtocolError("birth provider is not bound")
        uid = self.binding_for_uid(action_uid).action_uid
        raw = self._provider.birth_highwater_for(uid)
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ActionBallContractError(
                "provider birth high-water must be a length-2 tuple/list"
            )
        index, draw_end = raw
        if type(index) is not int or type(draw_end) is not int:
            raise ActionBallContractError(
                "provider birth high-water values must be plain integers"
            )
        if index == -1 and draw_end == 0:
            return (-1, 0)
        if (
            index < 0
            or index > MAX_COUNTER
            or draw_end < 1
            or draw_end > MAX_COUNTER
        ):
            raise ActionBallContractError(
                "provider birth high-water values are out of range"
            )
        return (index, draw_end)

    def _domain_cursor(self, action_uid: int) -> int:
        if self._domain_authority is None:
            raise BirthProtocolError(
                "domain claim authority is not bound"
            )
        uid = self.binding_for_uid(action_uid).action_uid
        return _plain_int(
            self._domain_authority.domain_cursor_for(uid),
            name="domain authority cursor",
        )

    def _callback_highwaters(
        self,
    ) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, int]]:
        """Read every callback tape atomically and fail on hidden mutation."""

        provider_state, authority_state = self._callback_states()
        try:
            provider = {
                binding.action_uid: self._provider_birth_highwater(
                    binding.action_uid
                )
                for binding in self._bindings
            }
            domain = {
                binding.action_uid: self._domain_cursor(
                    binding.action_uid
                )
                for binding in self._bindings
            }
            if self._callback_states() != (
                provider_state,
                authority_state,
            ):
                raise ActionBallContractError(
                    "provider/domain high-water authorities must be pure"
                )
            return provider, domain
        except Exception:
            self._restore_callback_states(
                provider_state, authority_state
            )
            raise

    def _broker_callback_highwaters(
        self,
    ) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, int]]:
        provider = {
            binding.action_uid: (
                self._last_sampler_birth_index.get(
                    binding.action_uid, -1
                ),
                self._last_sampler_draw_end.get(binding.action_uid, 0),
            )
            for binding in self._bindings
        }
        domain = {
            binding.action_uid: self._domain_claim_count.get(
                binding.action_uid, 0
            )
            for binding in self._bindings
        }
        return provider, domain

    def _assert_complete_birth_transcript(
        self,
        receipts: Sequence[ActionBirthReceipt],
        *,
        sampler_birth_indices: Mapping[int, int],
        sampler_draw_ends: Mapping[int, int],
    ) -> None:
        """Require broker receipts to exhaust every provider-issued birth.

        A provider high-water proves only the last issued birth.  Without
        this coverage check, a re-signed checkpoint could delete an older
        env/generation receipt while retaining the provider state and final
        high-water, then reuse the deleted generation after restore.
        """

        by_action: Dict[int, Dict[int, ActionBirthReceipt]] = {
            binding.action_uid: {} for binding in self._bindings
        }
        sampler_sha256: set[str] = set()
        for receipt in receipts:
            if not isinstance(receipt, ActionBirthReceipt):
                raise ActionBallContractError(
                    "birth transcript requires ActionBirthReceipt rows"
                )
            try:
                action_rows = by_action[receipt.action_uid]
            except KeyError as exc:
                raise ActionBallContractError(
                    "birth transcript has an unknown action_uid"
                ) from exc
            if (
                receipt.sampler_birth_sha256 in sampler_sha256
                or receipt.sampler_birth_index in action_rows
            ):
                raise ActionBallContractError(
                    "birth transcript replays one sampler birth"
                )
            sampler_sha256.add(receipt.sampler_birth_sha256)
            action_rows[receipt.sampler_birth_index] = receipt

        for binding in self._bindings:
            action_uid = binding.action_uid
            last_index = sampler_birth_indices.get(action_uid, -1)
            last_draw_end = sampler_draw_ends.get(action_uid, 0)
            action_rows = by_action[action_uid]
            if last_index == -1:
                if last_draw_end != 0 or action_rows:
                    raise ActionBallContractError(
                        "birth transcript disagrees with empty provider "
                        "high-water"
                    )
                continue
            if (
                last_index < 0
                or last_draw_end < SAMPLER_BIRTH_DRAW_COUNT
                or len(action_rows) != last_index + 1
                or min(action_rows, default=-1) != 0
                or max(action_rows, default=-1) != last_index
            ):
                raise ActionBallContractError(
                    "birth transcript does not exhaust provider-issued "
                    "birth indices"
                )
            if (
                action_rows[last_index].sampler_draw_end
                != last_draw_end
            ):
                raise ActionBallContractError(
                    "birth transcript final draw end differs from provider "
                    "high-water"
                )

    def _reserve_many_diagnostic_batched(
        self,
        validated_requests: Sequence[
            Tuple[BirthReserveRequest, ActionBinding]
        ],
        *,
        claim_many: Callable[[Sequence[int]], object],
        provide_many: Callable[
            [Sequence[ActionBirthRequest]], object
        ],
    ) -> Tuple[ActionBirthReceipt, ...]:
        """Run the unauthorized diagnostic birth callbacks once per batch.

        Formal training deliberately never enters this seam.  Both callback
        outputs are validate-all/commit-all batches; broker pending state is
        untouched until every claim and receipt validates.  A callback fault
        is terminal for the diagnostic run because its private provider/RNG
        tape may already have advanced, but Motion and Racket have not yet
        received or installed any row.
        """

        raw_claims = claim_many(
            tuple(
                binding.action_uid
                for _request, binding in validated_requests
            )
        )
        if isinstance(raw_claims, (str, bytes)) or not isinstance(
            raw_claims, Sequence
        ):
            raise ActionBallContractError(
                "batched domain authority must return a claim sequence"
            )
        claims = tuple(raw_claims)
        if len(claims) != len(validated_requests):
            raise ActionBallContractError(
                "batched domain authority returned a partial claim batch"
            )

        projected_domain_counts = dict(self._domain_claim_count)
        provider_requests = []
        for (request, binding), domain_claim in zip(
            validated_requests, claims
        ):
            if not isinstance(domain_claim, ActionDomainClaim):
                raise ActionBallContractError(
                    "domain claim authority must return ActionDomainClaim"
                )
            if (
                domain_claim.action_uid != binding.action_uid
                or domain_claim.profile_sha256
                != binding.profile_sha256
                or domain_claim.mobility_mode != self._mobility_mode
                or domain_claim.authority_contract_sha256
                != self._pins.domain_authority_sha256
            ):
                raise ActionBallContractError(
                    "domain claim does not match action binding/run pins"
                )
            projected_domain_counts[binding.action_uid] = (
                projected_domain_counts.get(binding.action_uid, 0) + 1
            )
            provider_requests.append(
                ActionBirthRequest(
                    env_id=request.env_id,
                    reset_generation=request.reset_generation,
                    action_uid=binding.action_uid,
                    action_slot=binding.action_slot,
                    domain_claim=domain_claim,
                    registry_sha256=self._registry_sha256,
                    mobility_mode=self._mobility_mode,
                    binding=binding,
                    pins=self._pins,
                )
            )

        raw_receipts = provide_many(tuple(provider_requests))
        if isinstance(raw_receipts, (str, bytes)) or not isinstance(
            raw_receipts, Sequence
        ):
            raise ActionBallContractError(
                "batched birth provider must return a receipt sequence"
            )
        receipts = tuple(raw_receipts)
        if len(receipts) != len(validated_requests):
            raise ActionBallContractError(
                "batched birth provider returned a partial receipt batch"
            )

        receipt_digests: set[str] = set()
        sampler_birth_digests: set[str] = set()
        projected_birth_indices = dict(
            self._last_sampler_birth_index
        )
        projected_draw_ends = dict(self._last_sampler_draw_end)
        for (
            (request, binding),
            domain_claim,
            receipt,
        ) in zip(validated_requests, claims, receipts):
            if not isinstance(receipt, ActionBirthReceipt):
                raise ActionBallContractError(
                    "birth provider must return ActionBirthReceipt"
                )
            receipt.assert_contract(
                binding=binding,
                pins=self._pins,
                mobility_mode=self._mobility_mode,
                registry_sha256=self._registry_sha256,
            )
            if (
                receipt.env_id != request.env_id
                or receipt.reset_generation
                != request.reset_generation
                or receipt.domain_epoch != domain_claim.domain_epoch
                or receipt.domain_levels != domain_claim.domain_levels
                or receipt.levels_sha256
                != domain_claim.levels_sha256
                or receipt.domain_claim_sha256
                != domain_claim.canonical_sha256
                or receipt.registry_sha256 != self._registry_sha256
            ):
                raise ActionBallContractError(
                    "birth provider returned wrong env/reset/domain claim"
                )
            receipt_digest = receipt.canonical_sha256
            if receipt_digest in receipt_digests:
                raise ActionBallContractError(
                    "birth provider replayed a receipt within one batch"
                )
            if (
                receipt.sampler_birth_sha256
                in sampler_birth_digests
            ):
                raise ActionBallContractError(
                    "birth provider replayed a sampler birth within one "
                    "batch"
                )
            expected_birth_index = (
                projected_birth_indices.get(binding.action_uid, -1) + 1
            )
            if receipt.sampler_birth_index != expected_birth_index:
                raise ActionBallContractError(
                    "birth provider returned a non-contiguous sampler "
                    "birth index"
                )
            if receipt.sampler_draw_start < projected_draw_ends.get(
                binding.action_uid, 0
            ):
                raise ActionBallContractError(
                    "birth provider sampler draw range replayed/overlapped "
                    "a prior birth"
                )
            projected_birth_indices[
                binding.action_uid
            ] = receipt.sampler_birth_index
            projected_draw_ends[
                binding.action_uid
            ] = receipt.sampler_draw_end
            receipt_digests.add(receipt_digest)
            sampler_birth_digests.add(
                receipt.sampler_birth_sha256
            )

        # Broker state publishes only after the complete batch validates.
        for (request, _binding), receipt in zip(
            validated_requests, receipts
        ):
            self._pending[request.env_id] = _PendingBirth(
                receipt, "reserved"
            )
            self._last_generation[
                request.env_id
            ] = request.reset_generation
        self._last_sampler_birth_index = projected_birth_indices
        self._last_sampler_draw_end = projected_draw_ends
        self._domain_claim_count = projected_domain_counts
        return receipts

    def reserve_true_reset(
        self,
        *,
        env_id: int,
        reset_generation: int,
        action_uid: int,
        action_slot: int,
        reset_kind: str = "true_reset",
    ) -> ActionBirthReceipt:
        request = BirthReserveRequest(
            env_id=env_id,
            reset_generation=reset_generation,
            action_uid=action_uid,
            action_slot=action_slot,
        )
        return self.reserve_many_true_reset(
            (request,), reset_kind=reset_kind
        )[0]

    def reserve_many_true_reset(
        self,
        requests: Sequence[BirthReserveRequest],
        *,
        reset_kind: str = "true_reset",
    ) -> Tuple[ActionBirthReceipt, ...]:
        """Atomically reserve provider births for a reset env batch."""

        _true_reset(reset_kind)
        if self._provider is None:
            raise BirthProtocolError("birth provider is not bound")
        if self._domain_authority is None:
            raise BirthProtocolError(
                "domain claim authority is not bound"
            )
        if isinstance(requests, (str, bytes)) or not isinstance(
            requests, Sequence
        ):
            raise ActionBallContractError(
                "reserve requests must be a non-empty sequence"
            )
        converted = tuple(requests)
        if not converted or any(
            not isinstance(request, BirthReserveRequest)
            for request in converted
        ):
            raise ActionBallContractError(
                "reserve requests must be non-empty BirthReserveRequest "
                "objects"
            )
        env_ids = [request.env_id for request in converted]
        if len(set(env_ids)) != len(env_ids):
            raise BirthProtocolError(
                "reserve batch must not repeat an env_id"
            )

        validated_requests: list[
            Tuple[BirthReserveRequest, ActionBinding]
        ] = []
        for request in converted:
            env = request.env_id
            binding = self._binding(
                request.action_uid, request.action_slot
            )
            if env in self._pending:
                raise BirthProtocolError(
                    f"env {env} already has an unconsumed birth"
                )
            expected_generation = self._last_generation.get(env, 0) + 1
            if request.reset_generation != expected_generation:
                raise BirthProtocolError(
                    f"env {env} reset generation must be exactly "
                    f"{expected_generation}, got "
                    f"{request.reset_generation}"
                )
            validated_requests.append((request, binding))

        if self._diagnostic_fast_path:
            claim_many = getattr(
                self._domain_authority,
                "claim_many_for_actions",
                None,
            )
            provide_many = getattr(
                self._provider, "provide_many", None
            )
            if callable(claim_many) and callable(provide_many):
                return self._reserve_many_diagnostic_batched(
                    tuple(validated_requests),
                    claim_many=claim_many,
                    provide_many=provide_many,
                )

        receipts: list[ActionBirthReceipt] = []
        receipt_digests: set[str] = set()
        sampler_birth_digests: set[str] = set()
        projected_birth_indices = dict(self._last_sampler_birth_index)
        projected_draw_ends = dict(self._last_sampler_draw_end)
        projected_domain_counts = dict(self._domain_claim_count)
        provider_state = None
        authority_state = None
        if not self._diagnostic_fast_path:
            provider_state, authority_state = self._callback_states()
        try:
            if (
                not self._diagnostic_fast_path
                and
                self._callback_highwaters()
                != self._broker_callback_highwaters()
            ):
                raise ActionBallContractError(
                    "broker callback high-water differs from provider/"
                    "domain authority"
                )
            for request, binding in validated_requests:
                domain_claim = self._domain_authority.claim_for_action(
                    binding.action_uid
                )
                projected_domain_counts[binding.action_uid] = (
                    projected_domain_counts.get(binding.action_uid, 0) + 1
                )
                if not isinstance(domain_claim, ActionDomainClaim):
                    raise ActionBallContractError(
                        "domain claim authority must return "
                        "ActionDomainClaim"
                    )
                if (
                    domain_claim.action_uid != binding.action_uid
                    or domain_claim.profile_sha256
                    != binding.profile_sha256
                    or domain_claim.mobility_mode != self._mobility_mode
                    or domain_claim.authority_contract_sha256
                    != self._pins.domain_authority_sha256
                ):
                    raise ActionBallContractError(
                        "domain claim does not match action binding/run pins"
                    )
                provider_request = ActionBirthRequest(
                    env_id=request.env_id,
                    reset_generation=request.reset_generation,
                    action_uid=binding.action_uid,
                    action_slot=binding.action_slot,
                    domain_claim=domain_claim,
                    registry_sha256=self._registry_sha256,
                    mobility_mode=self._mobility_mode,
                    binding=binding,
                    pins=self._pins,
                )
                receipt = self._provider(provider_request)
                if not isinstance(receipt, ActionBirthReceipt):
                    raise ActionBallContractError(
                        "birth provider must return ActionBirthReceipt"
                    )
                receipt.assert_contract(
                    binding=binding,
                    pins=self._pins,
                    mobility_mode=self._mobility_mode,
                    registry_sha256=self._registry_sha256,
                )
                if (
                    receipt.env_id != request.env_id
                    or receipt.reset_generation
                    != request.reset_generation
                    or receipt.domain_epoch != domain_claim.domain_epoch
                    or receipt.domain_levels != domain_claim.domain_levels
                    or receipt.levels_sha256
                    != domain_claim.levels_sha256
                    or receipt.domain_claim_sha256
                    != domain_claim.canonical_sha256
                    or receipt.registry_sha256 != self._registry_sha256
                ):
                    raise ActionBallContractError(
                        "birth provider returned wrong env/reset/domain claim"
                    )
                receipt_digest = receipt.canonical_sha256
                if receipt_digest in receipt_digests:
                    raise ActionBallContractError(
                        "birth provider replayed a receipt within one batch"
                    )
                if (
                    receipt.sampler_birth_sha256
                    in sampler_birth_digests
                ):
                    raise ActionBallContractError(
                        "birth provider replayed a sampler birth within one "
                        "batch"
                    )
                expected_birth_index = (
                    projected_birth_indices.get(binding.action_uid, -1) + 1
                )
                if receipt.sampler_birth_index != expected_birth_index:
                    raise ActionBallContractError(
                        "birth provider returned a non-contiguous sampler "
                        "birth index"
                    )
                if receipt.sampler_draw_start < projected_draw_ends.get(
                    binding.action_uid, 0
                ):
                    raise ActionBallContractError(
                        "birth provider sampler draw range replayed/overlapped "
                        "a prior birth"
                    )
                projected_birth_indices[
                    binding.action_uid
                ] = receipt.sampler_birth_index
                projected_draw_ends[
                    binding.action_uid
                ] = receipt.sampler_draw_end
                receipt_digests.add(receipt_digest)
                sampler_birth_digests.add(
                    receipt.sampler_birth_sha256
                )
                receipts.append(receipt)
            if not self._diagnostic_fast_path:
                provider_state_after_issue, authority_state_after_issue = (
                    self._callback_states()
                )
                for receipt in receipts:
                    self._provider.assert_issued_birth(receipt)
                if self._callback_states() != (
                    provider_state_after_issue,
                    authority_state_after_issue,
                ):
                    raise ActionBallContractError(
                        "birth provider authority assertion must be pure"
                    )
                expected_highwaters = (
                    {
                        binding.action_uid: (
                            projected_birth_indices.get(
                                binding.action_uid, -1
                            ),
                            projected_draw_ends.get(binding.action_uid, 0),
                        )
                        for binding in self._bindings
                    },
                    {
                        binding.action_uid: projected_domain_counts.get(
                            binding.action_uid, 0
                        )
                        for binding in self._bindings
                    },
                )
                if self._callback_highwaters() != expected_highwaters:
                    raise ActionBallContractError(
                        "provider/domain authority advanced an unstaged action "
                        "tape"
                    )
        except Exception:
            if not self._diagnostic_fast_path:
                try:
                    self._restore_callback_states(
                        provider_state, authority_state
                    )
                except Exception as rollback_error:
                    raise BirthProtocolError(
                        "birth provider failed and its exact state rollback "
                        "failed"
                    ) from rollback_error
            raise

        # Broker state commits only after all provider outputs validate.
        for (request, _binding), receipt in zip(
            validated_requests, receipts
        ):
            self._pending[request.env_id] = _PendingBirth(
                receipt, "reserved"
            )
            self._last_generation[
                request.env_id
            ] = request.reset_generation
        self._last_sampler_birth_index = projected_birth_indices
        self._last_sampler_draw_end = projected_draw_ends
        self._domain_claim_count = projected_domain_counts
        return tuple(receipts)

    def pending_receipt(
        self,
        *,
        env_id: int,
        reset_generation: int,
        action_uid: int,
        action_slot: int,
        reset_kind: str = "true_reset",
    ) -> ActionBirthReceipt:
        _true_reset(reset_kind)
        env = _plain_int(env_id, name="env_id")
        generation = _plain_int(
            reset_generation, name="reset_generation", minimum=1
        )
        binding = self._binding(action_uid, action_slot)
        pending = self._pending.get(env)
        if pending is None:
            raise BirthProtocolError(f"env {env} has no pending birth")
        receipt = pending.receipt
        if (
            receipt.reset_generation != generation
            or receipt.action_uid != binding.action_uid
            or receipt.action_slot != binding.action_slot
        ):
            raise BirthProtocolError(
                "pending birth generation/action identity mismatch"
            )
        return receipt

    def commit_true_reset(
        self,
        *,
        env_id: int,
        reset_generation: int,
        receipt_sha256: str,
        reset_kind: str = "true_reset",
    ) -> None:
        request = BirthCommitRequest(
            env_id=env_id,
            reset_generation=reset_generation,
            receipt_sha256=receipt_sha256,
        )
        self.commit_many_true_reset((request,), reset_kind=reset_kind)

    def commit_many_true_reset(
        self,
        requests: Sequence[BirthCommitRequest],
        *,
        reset_kind: str = "true_reset",
    ) -> None:
        """Atomically mark a reset batch committed after one root write."""

        _true_reset(reset_kind)
        if isinstance(requests, (str, bytes)) or not isinstance(
            requests, Sequence
        ):
            raise ActionBallContractError(
                "commit requests must be a non-empty sequence"
            )
        converted = tuple(requests)
        if not converted or any(
            not isinstance(request, BirthCommitRequest)
            for request in converted
        ):
            raise ActionBallContractError(
                "commit requests must be non-empty BirthCommitRequest "
                "objects"
            )
        env_ids = [request.env_id for request in converted]
        if len(set(env_ids)) != len(env_ids):
            raise BirthProtocolError(
                "commit batch must not repeat an env_id"
            )
        validated: list[Tuple[int, _PendingBirth]] = []
        for request in converted:
            pending = self._pending.get(request.env_id)
            if pending is None:
                raise BirthProtocolError(
                    f"env {request.env_id} has no pending birth"
                )
            if pending.status != "reserved":
                raise BirthProtocolError(
                    f"env {request.env_id} birth was already committed"
                )
            if (
                pending.receipt.reset_generation
                != request.reset_generation
                or pending.receipt.canonical_sha256
                != request.receipt_sha256
            ):
                raise BirthProtocolError(
                    "commit generation/receipt SHA does not match "
                    "reservation"
                )
            validated.append((request.env_id, pending))
        for env, pending in validated:
            self._pending[env] = _PendingBirth(
                pending.receipt, "committed"
            )

    def consume_true_reset(
        self,
        *,
        env_id: int,
        reset_generation: int,
        action_uid: int,
        action_slot: int,
        receipt_sha256: str,
        reset_kind: str = "true_reset",
    ) -> ActionBirthReceipt:
        request = BirthConsumeRequest(
            env_id=env_id,
            reset_generation=reset_generation,
            action_uid=action_uid,
            action_slot=action_slot,
            receipt_sha256=receipt_sha256,
        )
        return self.consume_many_true_reset(
            (request,), reset_kind=reset_kind
        )[0]

    def consume_many_true_reset(
        self,
        requests: Sequence[BirthConsumeRequest],
        *,
        reset_kind: str = "true_reset",
    ) -> Tuple[ActionBirthReceipt, ...]:
        """Atomically consume one committed birth for every requested env.

        All generation, action, slot, and digest claims are checked before any
        pending receipt is removed.  Racket can therefore validate an
        ``env_ids`` batch without partially consuming the prefix when a later
        row is stale.
        """

        _true_reset(reset_kind)
        if isinstance(requests, (str, bytes)) or not isinstance(
            requests, Sequence
        ):
            raise ActionBallContractError(
                "consume requests must be a non-empty sequence"
            )
        converted = tuple(requests)
        if not converted:
            raise ActionBallContractError(
                "consume requests must be non-empty"
            )
        if any(
            not isinstance(request, BirthConsumeRequest)
            for request in converted
        ):
            raise ActionBallContractError(
                "every consume request must be BirthConsumeRequest"
            )
        env_ids = [request.env_id for request in converted]
        if len(set(env_ids)) != len(env_ids):
            raise BirthProtocolError(
                "consume batch must not repeat an env_id"
            )

        validated: list[Tuple[int, int, ActionBirthReceipt]] = []
        for request in converted:
            env = request.env_id
            generation = request.reset_generation
            binding = self._binding(
                request.action_uid, request.action_slot
            )
            pending = self._pending.get(env)
            if pending is None:
                consumed = self._consumed_generation.get(env, 0)
                if generation <= consumed:
                    raise BirthProtocolError(
                        f"env {env} generation {generation} birth was "
                        "already consumed (replay/stale)"
                    )
                raise BirthProtocolError(
                    f"env {env} has no pending birth"
                )
            receipt = pending.receipt
            if pending.status != "committed":
                raise BirthProtocolError(
                    "Racket cannot consume birth before Motion commits "
                    "root write"
                )
            if (
                receipt.reset_generation != generation
                or receipt.action_uid != binding.action_uid
                or receipt.action_slot != binding.action_slot
                or receipt.canonical_sha256
                != request.receipt_sha256
            ):
                raise BirthProtocolError(
                    "consume generation/action/receipt SHA mismatch"
                )
            validated.append((env, generation, receipt))

        if self._diagnostic_fast_path:
            for env, generation, receipt in validated:
                del self._pending[env]
                self._consumed_generation[env] = generation
                self._diagnostic_consumed_receipt_by_env[env] = receipt
        else:
            for env, generation, _receipt in validated:
                del self._pending[env]
                self._consumed_generation[env] = generation
            for env, _generation, receipt in validated:
                key = (env, receipt.reset_generation)
                self._consumed_receipts[key] = receipt
        return tuple(receipt for _env, _generation, receipt in validated)

    def assert_consumed_birth(
        self, birth: ActionBirthReceipt
    ) -> None:
        """Fail unless Racket has consumed this exact committed birth."""

        if not isinstance(birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "consumed birth authority requires ActionBirthReceipt"
            )
        binding = self._binding(birth.action_uid, birth.action_slot)
        birth.assert_contract(
            binding=binding,
            pins=self._pins,
            mobility_mode=self._mobility_mode,
            registry_sha256=self._registry_sha256,
        )
        consumed_receipt = (
            self._diagnostic_consumed_receipt_by_env.get(birth.env_id)
            if self._diagnostic_fast_path
            else self._consumed_receipts.get(
                (birth.env_id, birth.reset_generation)
            )
        )
        if (
            self._consumed_generation.get(birth.env_id, 0)
            < birth.reset_generation
            or consumed_receipt != birth
        ):
            raise BirthProtocolError(
                "birth is not the env's exact consumed generation"
            )

    def assert_known_generation(
        self, *, env_id: int, reset_generation: int
    ) -> None:
        """Fail unless the broker transcript has reached this generation.

        This deliberately accepts an older already-consumed generation: pool
        retirement can lag behind a subsequent true reset, but a checkpoint
        must never invent retirement provenance for an env/generation the
        broker has not issued.
        """

        env = _plain_int(env_id, name="env_id")
        generation = _plain_int(
            reset_generation, name="reset_generation", minimum=1
        )
        if self._last_generation.get(env, 0) < generation:
            raise BirthProtocolError(
                "generation is absent from the birth broker transcript"
            )

    def state_dict(self) -> Dict[str, object]:
        provider_state, authority_state = self._callback_states()
        consumed_receipt_items = (
            tuple(
                (
                    (env, receipt.reset_generation),
                    receipt,
                )
                for env, receipt in sorted(
                    self._diagnostic_consumed_receipt_by_env.items()
                )
            )
            if self._diagnostic_fast_path
            else tuple(self._consumed_receipts.items())
        )
        transcript = (
            *(receipt for _key, receipt in consumed_receipt_items),
            *(
                pending.receipt
                for pending in self._pending.values()
            ),
        )
        try:
            if self._provider is None:
                raise BirthProtocolError("birth provider is not bound")
            self._assert_complete_birth_transcript(
                transcript,
                sampler_birth_indices=(
                    self._last_sampler_birth_index
                ),
                sampler_draw_ends=self._last_sampler_draw_end,
            )
            for receipt in transcript:
                self._provider.assert_issued_birth(receipt)
            if self._callback_states() != (
                provider_state,
                authority_state,
            ):
                raise ActionBallContractError(
                    "birth provider authority assertion must be pure"
                )
            if (
                self._callback_highwaters()
                != self._broker_callback_highwaters()
            ):
                raise ActionBallContractError(
                    "broker callback high-water differs from provider/domain "
                    "authority"
                )
        except Exception:
            self._restore_callback_states(
                provider_state, authority_state
            )
            raise
        payload: Dict[str, object] = {
            "schema_version": BROKER_STATE_SCHEMA_VERSION,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "registry_sha256": self._registry_sha256,
            "pins": self._pins.to_dict(),
            "mobility_mode": self._mobility_mode,
            "bindings": [
                binding.to_dict() for binding in self._bindings
            ],
            "domain_authority_contract_sha256": (
                self._pins.domain_authority_sha256
            ),
            "domain_authority_state_owner_sha256": (
                self._domain_authority_state_owner_sha256()
            ),
            "domain_authority_state": authority_state,
            "domain_authority_state_sha256": _sha256_json(
                authority_state
            ),
            "provider_contract_sha256": self._pins.sampler_sha256,
            "provider_state_owner_sha256": (
                self._provider_state_owner_sha256()
            ),
            "provider_state": provider_state,
            "provider_state_sha256": _sha256_json(provider_state),
            "domain_claim_counts": [
                [action_uid, count]
                for action_uid, count in sorted(
                    self._domain_claim_count.items()
                )
            ],
            "last_sampler_birth_indices": [
                [action_uid, birth_index]
                for action_uid, birth_index in sorted(
                    self._last_sampler_birth_index.items()
                )
            ],
            "last_sampler_draw_ends": [
                [action_uid, draw_end]
                for action_uid, draw_end in sorted(
                    self._last_sampler_draw_end.items()
                )
            ],
            "last_generations": [
                [env, generation]
                for env, generation in sorted(self._last_generation.items())
            ],
            "consumed_generations": [
                [env, generation]
                for env, generation in sorted(
                    self._consumed_generation.items()
                )
            ],
            "consumed_receipts": [
                receipt.to_dict()
                for _key, receipt in sorted(consumed_receipt_items)
            ],
            "pending": [
                {
                    "env_id": env,
                    "status": pending.status,
                    "receipt": pending.receipt.to_dict(),
                }
                for env, pending in sorted(self._pending.items())
            ],
        }
        payload["integrity_sha256"] = _sha256_json(payload)
        return payload

    def diagnostic_state_dict_with_consumed_history(
        self, receipts: Sequence[ActionBirthReceipt]
    ) -> Dict[str, object]:
        """Snapshot a complete diagnostic consumed transcript, read-only.

        The diagnostic live broker deliberately retains only the latest
        consumed receipt per environment.  Immutable-N1 exact resume keeps
        the complete provider-issued birth history separately and may supply
        it here at a stable (no-pending-transaction) checkpoint.  This method
        neither changes the legacy ``state_dict()`` payload nor mutates broker
        or callback state.
        """

        if not self._diagnostic_fast_path:
            raise ActionBallContractError(
                "diagnostic consumed-history state requires diagnostic mode"
            )
        if (
            isinstance(receipts, (str, bytes))
            or not isinstance(receipts, Sequence)
        ):
            raise TypeError("diagnostic consumed history must be a sequence")
        if self._pending:
            raise BirthProtocolError(
                "diagnostic consumed-history state forbids pending births"
            )
        converted = tuple(receipts)
        if any(
            not isinstance(receipt, ActionBirthReceipt)
            for receipt in converted
        ):
            raise TypeError(
                "diagnostic consumed history requires ActionBirthReceipt rows"
            )
        keys = tuple(
            (receipt.env_id, receipt.reset_generation)
            for receipt in converted
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ActionBallContractError(
                "diagnostic consumed history must be unique and sorted by "
                "env/generation"
            )
        expected_keys = tuple(
            (env, generation)
            for env, last_generation in sorted(
                self._consumed_generation.items()
            )
            for generation in range(1, last_generation + 1)
        )
        if keys != expected_keys:
            raise ActionBallContractError(
                "diagnostic consumed history is missing or adds a generation"
            )
        by_key = dict(zip(keys, converted))
        for env, generation in self._consumed_generation.items():
            current = self._diagnostic_consumed_receipt_by_env.get(env)
            if current is None or by_key[(env, generation)] != current:
                raise ActionBallContractError(
                    "diagnostic consumed history changed the live env receipt"
                )

        provider_state, authority_state = self._callback_states()
        transcript = converted
        try:
            if self._provider is None:
                raise BirthProtocolError("birth provider is not bound")
            # Historical rows arrive through a diagnostic-only read seam,
            # rather than through reserve/consume.  Re-run the complete
            # immutable broker contract for every generation before trusting
            # provider high-waters; a permissive provider adapter must not
            # allow an older, re-signed receipt to drift from bindings/pins.
            for receipt in transcript:
                binding = self._binding(
                    receipt.action_uid, receipt.action_slot
                )
                receipt.assert_contract(
                    binding=binding,
                    pins=self._pins,
                    mobility_mode=self._mobility_mode,
                    registry_sha256=self._registry_sha256,
                )
            self._assert_complete_birth_transcript(
                transcript,
                sampler_birth_indices=self._last_sampler_birth_index,
                sampler_draw_ends=self._last_sampler_draw_end,
            )
            for receipt in transcript:
                self._provider.assert_issued_birth(receipt)
            if self._callback_states() != (
                provider_state,
                authority_state,
            ):
                raise ActionBallContractError(
                    "birth provider authority assertion must be pure"
                )
            if (
                self._callback_highwaters()
                != self._broker_callback_highwaters()
            ):
                raise ActionBallContractError(
                    "broker callback high-water differs from provider/domain "
                    "authority"
                )
        except Exception:
            self._restore_callback_states(
                provider_state, authority_state
            )
            raise

        payload: Dict[str, object] = {
            "schema_version": BROKER_STATE_SCHEMA_VERSION,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "registry_sha256": self._registry_sha256,
            "pins": self._pins.to_dict(),
            "mobility_mode": self._mobility_mode,
            "bindings": [
                binding.to_dict() for binding in self._bindings
            ],
            "domain_authority_contract_sha256": (
                self._pins.domain_authority_sha256
            ),
            "domain_authority_state_owner_sha256": (
                self._domain_authority_state_owner_sha256()
            ),
            "domain_authority_state": authority_state,
            "domain_authority_state_sha256": _sha256_json(
                authority_state
            ),
            "provider_contract_sha256": self._pins.sampler_sha256,
            "provider_state_owner_sha256": (
                self._provider_state_owner_sha256()
            ),
            "provider_state": provider_state,
            "provider_state_sha256": _sha256_json(provider_state),
            "domain_claim_counts": [
                [action_uid, count]
                for action_uid, count in sorted(
                    self._domain_claim_count.items()
                )
            ],
            "last_sampler_birth_indices": [
                [action_uid, birth_index]
                for action_uid, birth_index in sorted(
                    self._last_sampler_birth_index.items()
                )
            ],
            "last_sampler_draw_ends": [
                [action_uid, draw_end]
                for action_uid, draw_end in sorted(
                    self._last_sampler_draw_end.items()
                )
            ],
            "last_generations": [
                [env, generation]
                for env, generation in sorted(
                    self._last_generation.items()
                )
            ],
            "consumed_generations": [
                [env, generation]
                for env, generation in sorted(
                    self._consumed_generation.items()
                )
            ],
            "consumed_receipts": [
                receipt.to_dict() for receipt in converted
            ],
            "pending": [],
        }
        payload["integrity_sha256"] = _sha256_json(payload)
        if self._callback_states() != (
            provider_state,
            authority_state,
        ):
            raise ActionBallContractError(
                "diagnostic consumed-history snapshot mutated callback state"
            )
        return payload

    @staticmethod
    def _generation_rows(
        value: object, *, name: str
    ) -> Dict[int, int]:
        if not isinstance(value, (tuple, list)):
            raise ActionBallContractError(f"{name} must be a list")
        result: Dict[int, int] = {}
        for index, row in enumerate(value):
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 2
            ):
                raise ActionBallContractError(
                    f"{name}[{index}] must be [env_id, generation]"
                )
            env = _plain_int(row[0], name=f"{name}[{index}].env_id")
            generation = _plain_int(
                row[1],
                name=f"{name}[{index}].generation",
                minimum=1,
            )
            if env in result:
                raise ActionBallContractError(
                    f"{name} repeats env_id {env}"
                )
            result[env] = generation
        if list(result) != sorted(result):
            raise ActionBallContractError(
                f"{name} must be sorted by env_id"
            )
        return result

    @staticmethod
    def _action_counter_rows(
        value: object, *, name: str
    ) -> Dict[int, int]:
        if not isinstance(value, (tuple, list)):
            raise ActionBallContractError(f"{name} must be a list")
        result: Dict[int, int] = {}
        for index, row in enumerate(value):
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ActionBallContractError(
                    f"{name}[{index}] must be [action_uid, counter]"
                )
            action_uid = _plain_int(
                row[0],
                name=f"{name}[{index}].action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            )
            counter = _plain_int(
                row[1], name=f"{name}[{index}].counter"
            )
            if action_uid in result:
                raise ActionBallContractError(
                    f"{name} repeats action_uid {action_uid}"
                )
            result[action_uid] = counter
        if list(result) != sorted(result):
            raise ActionBallContractError(
                f"{name} must be sorted by action_uid"
            )
        return result

    def load_state_dict(self, state: object) -> None:
        row = _exact_mapping(
            state, self._STATE_KEYS, name="birth broker state"
        )
        if row["schema_version"] != BROKER_STATE_SCHEMA_VERSION:
            raise ActionBallContractError(
                "unsupported birth broker state schema_version"
            )
        if row["runtime_contract_sha256"] != RUNTIME_CONTRACT_SHA256:
            raise ActionBallContractError(
                "birth broker runtime contract SHA mismatch"
            )
        declared_integrity = _sha256(
            row["integrity_sha256"], name="integrity_sha256"
        )
        payload = {
            key: row[key]
            for key in self._STATE_KEYS
            if key != "integrity_sha256"
        }
        if _sha256_json(payload) != declared_integrity:
            raise ActionBallContractError(
                "birth broker state integrity mismatch"
            )
        pins = RuntimePins.from_dict(row["pins"])
        mode = _mode(row["mobility_mode"])
        if not isinstance(row["bindings"], (tuple, list)):
            raise ActionBallContractError("bindings state must be a list")
        bindings = _validate_bindings(
            tuple(
                ActionBinding.from_dict(binding)
                for binding in row["bindings"]
            )
        )
        registry_sha = _sha256(
            row["registry_sha256"], name="registry_sha256"
        )
        if (
            pins != self._pins
            or mode != self._mobility_mode
            or bindings != self._bindings
            or registry_sha != self._registry_sha256
            or registry_sha
            != _registry_sha256(bindings, pins, mode)
        ):
            raise ActionBallContractError(
                "birth broker state belongs to a different run registry"
            )
        authority_contract_sha = _sha256(
            row["domain_authority_contract_sha256"],
            name="domain_authority_contract_sha256",
        )
        if authority_contract_sha != self._pins.domain_authority_sha256:
            raise ActionBallContractError(
                "birth broker domain authority contract SHA mismatch"
            )
        authority_owner_sha = _sha256(
            row["domain_authority_state_owner_sha256"],
            name="domain_authority_state_owner_sha256",
        )
        if (
            authority_owner_sha
            != self._domain_authority_state_owner_sha256()
        ):
            raise ActionBallContractError(
                "birth broker domain authority state owner mismatch"
            )
        authority_state = _json_data(
            row["domain_authority_state"],
            name="domain claim authority state",
        )
        if _sha256(
            row["domain_authority_state_sha256"],
            name="domain_authority_state_sha256",
        ) != _sha256_json(authority_state):
            raise ActionBallContractError(
                "birth broker domain authority state SHA mismatch"
            )
        provider_contract_sha = _sha256(
            row["provider_contract_sha256"],
            name="provider_contract_sha256",
        )
        if provider_contract_sha != self._pins.sampler_sha256:
            raise ActionBallContractError(
                "birth broker provider contract SHA mismatch"
            )
        provider_owner_sha = _sha256(
            row["provider_state_owner_sha256"],
            name="provider_state_owner_sha256",
        )
        if provider_owner_sha != self._provider_state_owner_sha256():
            raise ActionBallContractError(
                "birth broker provider state owner mismatch"
            )
        provider_state = _json_data(
            row["provider_state"], name="birth provider state"
        )
        if _sha256(
            row["provider_state_sha256"],
            name="provider_state_sha256",
        ) != _sha256_json(provider_state):
            raise ActionBallContractError(
                "birth broker provider state SHA mismatch"
            )
        domain_claim_counts = self._action_counter_rows(
            row["domain_claim_counts"],
            name="domain_claim_counts",
        )
        sampler_birth_indices = self._action_counter_rows(
            row["last_sampler_birth_indices"],
            name="last_sampler_birth_indices",
        )
        sampler_draw_ends = self._action_counter_rows(
            row["last_sampler_draw_ends"],
            name="last_sampler_draw_ends",
        )
        if (
            set(sampler_birth_indices) != set(sampler_draw_ends)
            or set(sampler_birth_indices) != set(domain_claim_counts)
            or any(
                action_uid not in self._by_uid
                for action_uid in sampler_birth_indices
            )
            or any(
                domain_claim_counts[action_uid] != birth_index + 1
                for action_uid, birth_index in sampler_birth_indices.items()
            )
            or any(draw_end < SAMPLER_BIRTH_DRAW_COUNT for draw_end in sampler_draw_ends.values())
        ):
            raise ActionBallContractError(
                "birth broker sampler counters disagree with the action "
                "registry"
            )
        last = self._generation_rows(
            row["last_generations"], name="last_generations"
        )
        consumed = self._generation_rows(
            row["consumed_generations"],
            name="consumed_generations",
        )
        if not isinstance(row["consumed_receipts"], (tuple, list)):
            raise ActionBallContractError(
                "consumed_receipts must be a list"
            )
        consumed_receipts: Dict[
            Tuple[int, int], ActionBirthReceipt
        ] = {}
        for index, raw_receipt in enumerate(row["consumed_receipts"]):
            receipt = ActionBirthReceipt.from_dict(raw_receipt)
            env = receipt.env_id
            generation = receipt.reset_generation
            key = (env, generation)
            if key in consumed_receipts:
                raise ActionBallContractError(
                    "consumed_receipts repeats an env/generation"
                )
            if generation > consumed.get(env, 0):
                raise ActionBallContractError(
                    "consumed receipt exceeds generation ledger"
                )
            try:
                binding = self._by_uid[receipt.action_uid]
            except KeyError as exc:
                raise ActionBallContractError(
                    "consumed receipt has unknown action_uid"
                ) from exc
            receipt.assert_contract(
                binding=binding,
                pins=self._pins,
                mobility_mode=self._mobility_mode,
                registry_sha256=self._registry_sha256,
            )
            if receipt.action_slot != binding.action_slot:
                raise ActionBallContractError(
                    "consumed receipt action slot mismatch"
                )
            consumed_receipts[key] = receipt
        if list(consumed_receipts) != sorted(consumed_receipts):
            raise ActionBallContractError(
                "consumed_receipts must be sorted by env/generation"
            )
        for env, generation in consumed.items():
            if generation > last.get(env, 0):
                raise ActionBallContractError(
                    "consumed generation exceeds last generation"
                )
            if {
                receipt_generation
                for receipt_env, receipt_generation in consumed_receipts
                if receipt_env == env
            } != set(range(1, generation + 1)):
                raise ActionBallContractError(
                    "consumed receipt history must be contiguous"
                )
        if any(
            env not in consumed for env, _generation in consumed_receipts
        ):
            raise ActionBallContractError(
                "consumed receipt history has no generation ledger"
            )
        if not isinstance(row["pending"], (tuple, list)):
            raise ActionBallContractError("pending state must be a list")
        pending_result: Dict[int, _PendingBirth] = {}
        pending_sampler_sha256: set[str] = set()
        pending_sampler_indices: Dict[int, set[int]] = {}
        pending_sampler_ranges: Dict[int, list[Tuple[int, int]]] = {}
        for receipt in consumed_receipts.values():
            indices = pending_sampler_indices.setdefault(
                receipt.action_uid, set()
            )
            ranges = pending_sampler_ranges.setdefault(
                receipt.action_uid, []
            )
            if (
                receipt.sampler_birth_sha256 in pending_sampler_sha256
                or receipt.sampler_birth_index in indices
                or any(
                    receipt.sampler_draw_start < prior_end
                    and prior_start < receipt.sampler_draw_end
                    for prior_start, prior_end in ranges
                )
            ):
                raise ActionBallContractError(
                    "consumed state replays one sampler birth"
                )
            if (
                receipt.sampler_birth_index
                > sampler_birth_indices.get(receipt.action_uid, -1)
                or receipt.sampler_draw_end
                > sampler_draw_ends.get(receipt.action_uid, 0)
            ):
                raise ActionBallContractError(
                    "consumed sampler birth exceeds broker high-water"
                )
            pending_sampler_sha256.add(
                receipt.sampler_birth_sha256
            )
            indices.add(receipt.sampler_birth_index)
            ranges.append(
                (
                    receipt.sampler_draw_start,
                    receipt.sampler_draw_end,
                )
            )
        for index, raw_pending in enumerate(row["pending"]):
            pending_row = _exact_mapping(
                raw_pending,
                ("env_id", "status", "receipt"),
                name=f"pending[{index}]",
            )
            env = _plain_int(
                pending_row["env_id"], name=f"pending[{index}].env_id"
            )
            if env in pending_result:
                raise ActionBallContractError(
                    f"pending repeats env_id {env}"
                )
            receipt = ActionBirthReceipt.from_dict(
                pending_row["receipt"]
            )
            if receipt.env_id != env:
                raise ActionBallContractError(
                    "pending env_id disagrees with receipt"
                )
            try:
                binding = self._by_uid[receipt.action_uid]
            except KeyError as exc:
                raise ActionBallContractError(
                    "pending receipt has unknown action_uid"
                ) from exc
            receipt.assert_contract(
                binding=binding,
                pins=self._pins,
                mobility_mode=self._mobility_mode,
                registry_sha256=self._registry_sha256,
            )
            if receipt.action_slot != binding.action_slot:
                raise ActionBallContractError(
                    "pending receipt action slot mismatch"
                )
            if receipt.reset_generation != last.get(env):
                raise ActionBallContractError(
                    "pending receipt is not the env's last generation"
                )
            if receipt.reset_generation <= consumed.get(env, 0):
                raise ActionBallContractError(
                    "pending receipt generation was already consumed"
                )
            if (
                receipt.sampler_birth_sha256 in pending_sampler_sha256
                or receipt.sampler_birth_index
                in pending_sampler_indices.setdefault(
                    receipt.action_uid, set()
                )
            ):
                raise ActionBallContractError(
                    "pending state replays one sampler birth"
                )
            pending_sampler_sha256.add(receipt.sampler_birth_sha256)
            pending_sampler_indices[receipt.action_uid].add(
                receipt.sampler_birth_index
            )
            ranges = pending_sampler_ranges.setdefault(
                receipt.action_uid, []
            )
            if any(
                receipt.sampler_draw_start < prior_end
                and prior_start < receipt.sampler_draw_end
                for prior_start, prior_end in ranges
            ):
                raise ActionBallContractError(
                    "pending sampler birth draw ranges overlap"
                )
            ranges.append(
                (
                    receipt.sampler_draw_start,
                    receipt.sampler_draw_end,
                )
            )
            if (
                receipt.sampler_birth_index
                > sampler_birth_indices.get(receipt.action_uid, -1)
                or receipt.sampler_draw_end
                > sampler_draw_ends.get(receipt.action_uid, 0)
            ):
                raise ActionBallContractError(
                    "pending sampler birth exceeds broker high-water"
                )
            status = pending_row["status"]
            if status not in ("reserved", "committed"):
                raise ActionBallContractError(
                    "pending status must be reserved or committed"
                )
            pending_result[env] = _PendingBirth(receipt, status)
        if list(pending_result) != sorted(pending_result):
            raise ActionBallContractError(
                "pending state must be sorted by env_id"
            )
        transcript = (
            *consumed_receipts.values(),
            *(
                pending.receipt
                for pending in pending_result.values()
            ),
        )
        self._assert_complete_birth_transcript(
            transcript,
            sampler_birth_indices=sampler_birth_indices,
            sampler_draw_ends=sampler_draw_ends,
        )
        for env, generation in last.items():
            if env in pending_result:
                if consumed.get(env, 0) != generation - 1:
                    raise ActionBallContractError(
                        "pending generation must immediately follow the "
                        "last consumed generation"
                    )
            elif consumed.get(env, 0) != generation:
                raise ActionBallContractError(
                    "an unconsumed last generation is missing its pending "
                    "receipt"
                )
        if any(env not in last for env in consumed):
            raise ActionBallContractError(
                "consumed generation has no matching last generation"
            )
        previous_provider_state, previous_authority_state = (
            self._callback_states()
        )
        try:
            self._restore_callback_states(
                provider_state, authority_state
            )
            if self._provider is None:
                raise BirthProtocolError("birth provider is not bound")
            for receipt in transcript:
                self._provider.assert_issued_birth(receipt)
            if self._callback_states() != (
                provider_state,
                authority_state,
            ):
                raise ActionBallContractError(
                    "birth provider authority assertion must be pure"
                )
            expected_highwaters = (
                {
                    binding.action_uid: (
                        sampler_birth_indices.get(
                            binding.action_uid, -1
                        ),
                        sampler_draw_ends.get(binding.action_uid, 0),
                    )
                    for binding in self._bindings
                },
                {
                    binding.action_uid: domain_claim_counts.get(
                        binding.action_uid, 0
                    )
                    for binding in self._bindings
                },
            )
            if self._callback_highwaters() != expected_highwaters:
                raise ActionBallContractError(
                    "broker callback high-water differs from restored "
                    "provider/domain authority"
                )
        except Exception:
            self._restore_callback_states(
                previous_provider_state, previous_authority_state
            )
            raise
        # Atomic commit: no broker field changes until every row and the
        # bound stateful provider restore have validated.
        self._last_generation = last
        self._consumed_generation = consumed
        if self._diagnostic_fast_path:
            self._consumed_receipts = {}
            self._diagnostic_consumed_receipt_by_env = {
                env: consumed_receipts[(env, generation)]
                for env, generation in consumed.items()
                if (env, generation) in consumed_receipts
            }
        else:
            self._consumed_receipts = consumed_receipts
            self._diagnostic_consumed_receipt_by_env = {}
        self._pending = pending_result
        self._last_sampler_birth_index = sampler_birth_indices
        self._last_sampler_draw_end = sampler_draw_ends
        self._domain_claim_count = domain_claim_counts


@dataclass(frozen=True)
class ActionPoolRefillRequest:
    """Pure callback input for refilling exactly one selected action."""

    action_uid: int
    action_slot: int
    refill_index: int
    minimum_receipts: int
    swing_generation_start: int
    mobility_mode: str
    registry_sha256: str
    binding: ActionBinding
    pins: RuntimePins
    birth: ActionBirthReceipt

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "action_slot",
            _plain_int(self.action_slot, name="action_slot"),
        )
        object.__setattr__(
            self,
            "refill_index",
            _plain_int(
                self.refill_index, name="refill_index", minimum=1
            ),
        )
        object.__setattr__(
            self,
            "minimum_receipts",
            _plain_int(
                self.minimum_receipts,
                name="minimum_receipts",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "swing_generation_start",
            _plain_int(
                self.swing_generation_start,
                name="swing_generation_start",
            ),
        )
        object.__setattr__(
            self, "mobility_mode", _mode(self.mobility_mode)
        )
        object.__setattr__(
            self,
            "registry_sha256",
            _sha256(self.registry_sha256, name="registry_sha256"),
        )
        if (
            not isinstance(self.binding, ActionBinding)
            or self.binding.action_uid != self.action_uid
            or self.binding.action_slot != self.action_slot
        ):
            raise ActionBallContractError(
                "pool refill request action binding mismatch"
            )
        if not isinstance(self.pins, RuntimePins):
            raise ActionBallContractError(
                "pool refill request pins must be RuntimePins"
            )
        if not isinstance(self.birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "pool refill request birth must be ActionBirthReceipt"
            )
        self.birth.assert_contract(
            binding=self.binding,
            pins=self.pins,
            mobility_mode=self.mobility_mode,
            registry_sha256=self.registry_sha256,
        )


@dataclass(frozen=True)
class ActionPoolRefillBatch:
    """Solver callback output, including rejected-proposal accounting."""

    action_uid: int
    proposed_count: int
    proposal_sample_indices: Tuple[int, ...]
    receipts: Tuple[ActionBallTaskReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(
                self.action_uid,
                name="action_uid",
                minimum=1,
                maximum=MAX_ACTION_UID,
            ),
        )
        object.__setattr__(
            self,
            "proposed_count",
            _plain_int(self.proposed_count, name="proposed_count"),
        )
        if not isinstance(
            self.proposal_sample_indices, (tuple, list)
        ):
            raise ActionBallContractError(
                "proposal_sample_indices must be a tuple/list"
            )
        proposal_indices = tuple(
            _plain_int(index, name="proposal_sample_index")
            for index in self.proposal_sample_indices
        )
        object.__setattr__(
            self, "proposal_sample_indices", proposal_indices
        )
        if (
            len(proposal_indices) != self.proposed_count
            or tuple(sorted(set(proposal_indices)))
            != proposal_indices
        ):
            raise ActionBallContractError(
                "proposal sample indices must be strictly increasing, "
                "unique, and match proposed_count"
            )
        if not isinstance(self.receipts, (tuple, list)):
            raise ActionBallContractError(
                "refill receipts must be a tuple/list"
            )
        receipts = tuple(self.receipts)
        if any(
            not isinstance(receipt, ActionBallTaskReceipt)
            for receipt in receipts
        ):
            raise ActionBallContractError(
                "refill receipts must be ActionBallTaskReceipt objects"
            )
        object.__setattr__(self, "receipts", receipts)
        if self.proposed_count < len(receipts):
            raise ActionBallContractError(
                "proposed_count must cover every admitted receipt"
            )
        if not {
            receipt.sample_index for receipt in receipts
        }.issubset(proposal_indices):
            raise ActionBallContractError(
                "every admitted receipt must belong to this refill's "
                "proposal sample indices"
            )


@dataclass(frozen=True)
class ActionSampleAssignment:
    """Compact exact sample-index assignment for one birth refill."""

    birth: ActionBirthReceipt
    refill_index: int
    proposal_sample_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "sample assignment birth must be ActionBirthReceipt"
            )
        object.__setattr__(
            self,
            "refill_index",
            _plain_int(
                self.refill_index,
                name="sample assignment refill_index",
                minimum=1,
            ),
        )
        if not isinstance(
            self.proposal_sample_indices, (tuple, list)
        ):
            raise ActionBallContractError(
                "sample assignment indices must be a tuple/list"
            )
        indices = tuple(
            _plain_int(index, name="sample assignment index")
            for index in self.proposal_sample_indices
        )
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ActionBallContractError(
                "sample assignment indices must be non-empty, strictly "
                "increasing, and unique"
            )
        object.__setattr__(self, "proposal_sample_indices", indices)


def _encode_sample_index_segments(
    indices: Sequence[int],
) -> list[list[int]]:
    """Canonically run-length encode one sorted arithmetic index tape."""

    converted = tuple(
        _plain_int(index, name="proposal sample index")
        for index in indices
    )
    if not converted or tuple(sorted(set(converted))) != converted:
        raise ActionBallContractError(
            "proposal sample indices must be non-empty, strictly "
            "increasing, and unique"
        )
    result: list[list[int]] = []
    cursor = 0
    while cursor < len(converted):
        start = converted[cursor]
        if cursor + 1 == len(converted):
            result.append([start, 1, 1])
            break
        step = converted[cursor + 1] - start
        end = cursor + 2
        while (
            end < len(converted)
            and converted[end] - converted[end - 1] == step
        ):
            end += 1
        result.append([start, step, end - cursor])
        cursor = end
    return result


def _decode_sample_index_segments(
    value: object,
    *,
    name: str,
) -> Tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ActionBallContractError(
            f"{name} must be a non-empty segment list"
        )
    raw_segments: list[list[int]] = []
    indices: list[int] = []
    for segment_index, raw_segment in enumerate(value):
        if not isinstance(raw_segment, (tuple, list)) or len(
            raw_segment
        ) != 3:
            raise ActionBallContractError(
                f"{name}[{segment_index}] must be [start, step, count]"
            )
        start = _plain_int(
            raw_segment[0],
            name=f"{name}[{segment_index}].start",
        )
        step = _plain_int(
            raw_segment[1],
            name=f"{name}[{segment_index}].step",
            minimum=1,
        )
        count = _plain_int(
            raw_segment[2],
            name=f"{name}[{segment_index}].count",
            minimum=1,
        )
        raw_segments.append([start, step, count])
        indices.extend(start + step * offset for offset in range(count))
    converted = tuple(indices)
    if (
        tuple(sorted(set(converted))) != converted
        or _encode_sample_index_segments(converted) != raw_segments
    ):
        raise ActionBallContractError(
            f"{name} is not a canonical strictly increasing index tape"
        )
    return converted


def _sample_assignment_rows(
    assignments: Sequence[ActionSampleAssignment],
) -> list[Dict[str, object]]:
    return [
        {
            "refill_index": assignment.refill_index,
            "proposal_index_segments": _encode_sample_index_segments(
                assignment.proposal_sample_indices
            ),
        }
        for assignment in assignments
    ]


def _sample_assignments_from_rows(
    value: object,
    *,
    birth: ActionBirthReceipt,
    name: str,
) -> Tuple[ActionSampleAssignment, ...]:
    if not isinstance(value, (tuple, list)):
        raise ActionBallContractError(f"{name} must be a list")
    result: list[ActionSampleAssignment] = []
    for index, raw_assignment in enumerate(value):
        row = _exact_mapping(
            raw_assignment,
            ("refill_index", "proposal_index_segments"),
            name=f"{name}[{index}]",
        )
        result.append(
            ActionSampleAssignment(
                birth=birth,
                refill_index=row["refill_index"],  # type: ignore[arg-type]
                proposal_sample_indices=(
                    _decode_sample_index_segments(
                        row["proposal_index_segments"],
                        name=(
                            f"{name}[{index}]"
                            ".proposal_index_segments"
                        ),
                    )
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class ActionTaskIssueRequest:
    """One exact birth/swing claim for scalar or vectorized pool issue."""

    birth: ActionBirthReceipt
    swing_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "task issue birth must be ActionBirthReceipt"
            )
        object.__setattr__(
            self,
            "swing_generation",
            _plain_int(
                self.swing_generation, name="swing_generation"
            ),
        )


@dataclass(frozen=True)
class PoolLedger:
    requests: int = 0
    refill_calls: int = 0
    proposed: int = 0
    admitted: int = 0
    issued: int = 0
    discarded: int = 0

    def __post_init__(self) -> None:
        for name in (
            "requests",
            "refill_calls",
            "proposed",
            "admitted",
            "issued",
            "discarded",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(getattr(self, name), name=f"ledger.{name}"),
            )
        if (
            self.proposed < self.admitted
            or self.admitted < self.issued + self.discarded
        ):
            raise ActionBallContractError(
                "pool ledger requires proposed >= admitted >= "
                "issued + discarded"
            )
        if self.requests != self.issued:
            raise ActionBallContractError(
                "pool ledger requires requests == issued"
            )

    def to_dict(self) -> Dict[str, int]:
        return {
            "requests": self.requests,
            "refill_calls": self.refill_calls,
            "proposed": self.proposed,
            "admitted": self.admitted,
            "issued": self.issued,
            "discarded": self.discarded,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PoolLedger":
        row = _exact_mapping(
            value,
            (
                "requests",
                "refill_calls",
                "proposed",
                "admitted",
                "issued",
                "discarded",
            ),
            name="pool ledger",
        )
        return cls(**{name: row[name] for name in row})  # type: ignore[arg-type]


@dataclass(frozen=True)
class _RetiredPoolBirth:
    """Compact append-only lifecycle evidence for one retired birth."""

    birth: ActionBirthReceipt
    refill_index: int
    proposed_count: int
    admitted_count: int
    issued_count: int
    discarded_count: int
    task_transcript_sha256: str
    sample_assignments: Tuple[ActionSampleAssignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "retired lifecycle birth must be ActionBirthReceipt"
            )
        object.__setattr__(
            self,
            "refill_index",
            _plain_int(self.refill_index, name="retired.refill_index"),
        )
        object.__setattr__(
            self,
            "proposed_count",
            _plain_int(
                self.proposed_count, name="retired.proposed_count"
            ),
        )
        for name in (
            "admitted_count",
            "issued_count",
            "discarded_count",
        ):
            object.__setattr__(
                self,
                name,
                _plain_int(
                    getattr(self, name), name=f"retired.{name}"
                ),
            )
        object.__setattr__(
            self,
            "task_transcript_sha256",
            _sha256(
                self.task_transcript_sha256,
                name="retired.task_transcript_sha256",
            ),
        )
        if not isinstance(self.sample_assignments, (tuple, list)):
            raise ActionBallContractError(
                "retired sample assignments must be a tuple/list"
            )
        assignments = tuple(self.sample_assignments)
        if any(
            not isinstance(assignment, ActionSampleAssignment)
            or assignment.birth != self.birth
            for assignment in assignments
        ):
            raise ActionBallContractError(
                "retired sample assignments must bind the exact birth"
            )
        if [
            assignment.refill_index for assignment in assignments
        ] != list(range(1, self.refill_index + 1)):
            raise ActionBallContractError(
                "retired sample assignment refill indices are not "
                "contiguous"
            )
        object.__setattr__(self, "sample_assignments", assignments)
        if sum(
            len(assignment.proposal_sample_indices)
            for assignment in assignments
        ) != self.proposed_count:
            raise ActionBallContractError(
                "retired proposal count differs from sample assignments"
            )
        if self.proposed_count < self.admitted_count:
            raise ActionBallContractError(
                "retired proposed_count must cover admitted receipts"
            )
        if (
            self.admitted_count
            != self.issued_count + self.discarded_count
        ):
            raise ActionBallContractError(
                "retired admitted count must equal issued + discarded"
            )
        if self.admitted_count and self.refill_index == 0:
            raise ActionBallContractError(
                "retired admitted tasks require at least one refill"
            )
        if self.admitted_count == 0 and (
            self.task_transcript_sha256
            != task_transcript_sha256(
                self.birth.canonical_sha256, ()
            )
        ):
            raise ActionBallContractError(
                "empty retired task transcript has the wrong root"
            )


class ActionTaskSolver(Protocol):
    solver_contract_sha256: str
    state_owner_sha256: str

    def __call__(
        self, request: ActionPoolRefillRequest
    ) -> ActionPoolRefillBatch:
        """Sample/solve one requested action and return admitted receipts."""

    def state_dict(self) -> Mapping[str, object]:
        """Return all sampler/solver mutable state as pure JSON data."""

    def load_state_dict(self, state: object) -> None:
        """Atomically validate and restore a prior pure-data state."""

    def assert_emitted_sample(
        self, receipt: ActionBallTaskReceipt
    ) -> None:
        """Fail unless this exact canonical sample was emitted by the sampler."""

    def assert_emitted_tasks(
        self, receipts: Sequence[ActionBallTaskReceipt]
    ) -> None:
        """Pure batch proof of exact pinned solver outputs."""

    def emitted_task_count_for(self, action_uid: int) -> int:
        """Return the exact admitted-task transcript count for one action."""

    def task_transcript_for_birth(
        self, birth_sha256: str
    ) -> Tuple[int, str]:
        """Return exact ``(task_count, ordered_task_chain_sha256)``."""

    def assert_proposal_assignments(
        self, assignments: Sequence[ActionSampleAssignment]
    ) -> None:
        """Pure batch proof that sample indices belong to exact births."""

    def sample_highwater_for(
        self, action_uid: int
    ) -> Tuple[int, int]:
        """Return exact ``(last_sample_index, last_sample_draw_end)``."""


class LazyActionTaskPool:
    """Exact-resume FIFO pools materialized only for requested action UIDs."""

    _STATE_KEYS = (
        "schema_version",
        "runtime_contract_sha256",
        "registry_sha256",
        "pins",
        "mobility_mode",
        "refill_size",
        "bindings",
        "birth_authority_state_sha256",
        "solver_contract_sha256",
        "solver_state_owner_sha256",
        "solver_state",
        "solver_state_sha256",
        "retired_generations",
        "actions",
        "integrity_sha256",
    )

    def __init__(
        self,
        bindings: Sequence[ActionBinding],
        pins: RuntimePins,
        mobility_mode: str,
        *,
        refill_size: int = 1,
        diagnostic_unauthorized: bool = False,
    ) -> None:
        self._bindings = _validate_bindings(bindings)
        if not isinstance(pins, RuntimePins):
            raise ActionBallContractError("pins must be RuntimePins")
        self._pins = pins
        if type(diagnostic_unauthorized) is not bool:
            raise ActionBallContractError(
                "diagnostic_unauthorized must be an exact boolean"
            )
        self._diagnostic_fast_path = diagnostic_unauthorized
        if (
            pins.counter_rally_objective_profile_sha256 is not None
            and len(self._bindings) != 1
        ):
            raise CounterRallyTaskIdentityError(
                "counter-rally objective pin requires exact N=1 bindings"
            )
        self._mobility_mode = _mode(mobility_mode)
        self._refill_size = _plain_int(
            refill_size, name="refill_size", minimum=1
        )
        if self._diagnostic_fast_path and self._refill_size != 1:
            raise ActionBallContractError(
                "diagnostic_unauthorized task pools require refill_size=1"
            )
        self._by_uid = {
            binding.action_uid: binding for binding in self._bindings
        }
        self._registry_sha256 = _registry_sha256(
            self._bindings, self._pins, self._mobility_mode
        )
        self._solver: Callable[
            [ActionPoolRefillRequest], ActionPoolRefillBatch
        ] | None = None
        self._birth_authority: object | None = None
        # All of these maps are intentionally empty at construction, even N=93.
        self._births: Dict[int, Dict[str, ActionBirthReceipt]] = {}
        self._pending: Dict[
            int, Dict[str, list[ActionBallTaskReceipt]]
        ] = {}
        self._issued_task_transcript_sha256: Dict[
            int, Dict[str, str]
        ] = {}
        self._cursor: Dict[int, Dict[str, int]] = {}
        self._refill_index: Dict[int, Dict[str, int]] = {}
        self._proposed_by_birth: Dict[int, Dict[str, int]] = {}
        self._sample_assignments: Dict[
            int, Dict[str, list[ActionSampleAssignment]]
        ] = {}
        self._ledger: Dict[int, PoolLedger] = {}
        self._seen_sha256: Dict[int, Dict[str, set[str]]] = {}
        self._seen_sample_sha256: Dict[int, Dict[str, set[str]]] = {}
        self._retired_births: Dict[
            int, Dict[str, _RetiredPoolBirth]
        ] = {}
        self._task_lifecycle: Dict[int, list[int]] = {}
        self._last_sample_index: Dict[int, int] = {}
        self._last_sample_draw_end: Dict[int, int] = {}
        self._retired_generation: Dict[int, int] = {}
        # Diagnostic-only O(1) replacement for the formal cross-action scan.
        # It is bounded by live environments and is cleared on retirement.
        self._diagnostic_birth_by_env: Dict[int, Tuple[int, str]] = {}
        self._diagnostic_active_sample_sha256: set[str] = set()

    @property
    def materialized_action_uids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._births))

    @property
    def ordered_action_uids(self) -> Tuple[int, ...]:
        return tuple(binding.action_uid for binding in self._bindings)

    @property
    def action_count(self) -> int:
        return len(self._bindings)

    @property
    def diagnostic_fast_path(self) -> bool:
        return self._diagnostic_fast_path

    def pending_count(
        self,
        action_uid: int,
        *,
        birth_sha256: str | None = None,
    ) -> int:
        binding = self._binding(action_uid)
        uid = binding.action_uid
        if birth_sha256 is None:
            return sum(
                len(queue)
                for queue in self._pending.get(uid, {}).values()
            )
        digest = _sha256(birth_sha256, name="birth_sha256")
        return len(self._pending.get(uid, {}).get(digest, ()))

    def ledger(self, action_uid: int) -> PoolLedger:
        binding = self._binding(action_uid)
        return self._ledger.get(binding.action_uid, PoolLedger())

    def _binding(self, action_uid: int) -> ActionBinding:
        uid = _plain_int(
            action_uid,
            name="action_uid",
            minimum=1,
            maximum=MAX_ACTION_UID,
        )
        try:
            return self._by_uid[uid]
        except KeyError as exc:
            raise ActionBallContractError(
                f"unknown action_uid {uid}"
            ) from exc

    def bind_solver(self, solver: ActionTaskSolver) -> None:
        if self._solver is not None:
            raise PoolProtocolError("task solver may be bound only once")
        if not callable(solver):
            raise ActionBallContractError("task solver must be callable")
        if (
            _sha256(
                getattr(solver, "solver_contract_sha256", None),
                name="solver.solver_contract_sha256",
            )
            != self._pins.solver_sha256
        ):
            raise ActionBallContractError(
                "task solver contract SHA differs from runtime pins"
            )
        _sha256(
            getattr(solver, "state_owner_sha256", None),
            name="solver.state_owner_sha256",
        )
        if not callable(getattr(solver, "state_dict", None)) or not callable(
            getattr(solver, "load_state_dict", None)
        ):
            raise ActionBallContractError(
                "task solver must implement atomic state_dict/load_state_dict"
            )
        if not callable(getattr(solver, "assert_emitted_sample", None)):
            raise ActionBallContractError(
                "task solver must implement assert_emitted_sample()"
            )
        if not callable(getattr(solver, "assert_emitted_tasks", None)):
            raise ActionBallContractError(
                "task solver must implement assert_emitted_tasks()"
            )
        if not callable(
            getattr(solver, "emitted_task_count_for", None)
        ):
            raise ActionBallContractError(
                "task solver must implement emitted_task_count_for()"
            )
        if not callable(
            getattr(solver, "task_transcript_for_birth", None)
        ):
            raise ActionBallContractError(
                "task solver must implement task_transcript_for_birth()"
            )
        if not callable(
            getattr(solver, "assert_proposal_assignments", None)
        ):
            raise ActionBallContractError(
                "task solver must implement assert_proposal_assignments()"
            )
        if not callable(getattr(solver, "sample_highwater_for", None)):
            raise ActionBallContractError(
                "task solver must implement sample_highwater_for()"
            )
        _json_data(solver.state_dict(), name="solver state")
        self._solver = solver

    def _solver_state(self) -> object:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        return _json_data(
            self._solver.state_dict(), name="solver state"
        )

    def _solver_state_owner_sha256(self) -> str:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        return _sha256(
            getattr(self._solver, "state_owner_sha256", None),
            name="solver.state_owner_sha256",
        )

    def _solver_sample_highwater(
        self, action_uid: int
    ) -> Tuple[int, int]:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        uid = self._binding(action_uid).action_uid
        raw = self._solver.sample_highwater_for(uid)
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ActionBallContractError(
                "solver sample high-water must be a length-2 tuple/list"
            )
        index, draw_end = raw
        if type(index) is not int or type(draw_end) is not int:
            raise ActionBallContractError(
                "solver sample high-water values must be plain integers"
            )
        if index == -1 and draw_end == 0:
            return (-1, 0)
        if index < 0 or index > MAX_COUNTER or draw_end < 1 or draw_end > MAX_COUNTER:
            raise ActionBallContractError(
                "solver sample high-water values are out of range"
            )
        return (index, draw_end)

    def _solver_sample_highwaters(self) -> Dict[int, Tuple[int, int]]:
        """Read every action tape atomically and enforce a pure authority."""

        solver_state = self._solver_state()
        try:
            result = {
                binding.action_uid: self._solver_sample_highwater(
                    binding.action_uid
                )
                for binding in self._bindings
            }
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver sample high-water authority must be pure"
                )
            return result
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def _pool_sample_highwaters(self) -> Dict[int, Tuple[int, int]]:
        return {
            binding.action_uid: (
                self._last_sample_index.get(binding.action_uid, -1),
                self._last_sample_draw_end.get(binding.action_uid, 0),
            )
            for binding in self._bindings
        }

    def _solver_emitted_task_count(self, action_uid: int) -> int:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        uid = self._binding(action_uid).action_uid
        return _plain_int(
            self._solver.emitted_task_count_for(uid),
            name="solver emitted task count",
        )

    def _solver_emitted_task_counts(self) -> Dict[int, int]:
        """Read the full task authority catalog atomically and purely."""

        solver_state = self._solver_state()
        try:
            result = {
                binding.action_uid: self._solver_emitted_task_count(
                    binding.action_uid
                )
                for binding in self._bindings
            }
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver emitted-task count authority must be pure"
                )
            return result
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def _all_task_receipts_by_action(
        self,
    ) -> Dict[int, Tuple[ActionBallTaskReceipt, ...]]:
        """Return full receipts that must remain immediately issuable."""

        result: Dict[int, Tuple[ActionBallTaskReceipt, ...]] = {}
        for binding in self._bindings:
            uid = binding.action_uid
            receipts: list[ActionBallTaskReceipt] = []
            for digest in sorted(self._births.get(uid, {})):
                receipts.extend(self._pending[uid][digest])
            result[uid] = tuple(receipts)
        return result

    def _pool_emitted_task_counts(self) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for binding in self._bindings:
            uid = binding.action_uid
            active = sum(
                self._cursor[uid][digest]
                + len(self._pending[uid][digest])
                for digest in self._births.get(uid, {})
            )
            retired = sum(
                record.admitted_count
                for record in self._retired_births.get(uid, {}).values()
            )
            result[uid] = active + retired
        return result

    def _solver_task_transcript_for_birth(
        self, birth_sha256: str
    ) -> Tuple[int, str]:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        digest = _sha256(birth_sha256, name="birth_sha256")
        raw = self._solver.task_transcript_for_birth(digest)
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ActionBallContractError(
                "solver birth task transcript must be a length-2 "
                "tuple/list"
            )
        count = _plain_int(
            raw[0], name="solver birth task transcript count"
        )
        root = _sha256(
            raw[1], name="solver birth task transcript root"
        )
        return count, root

    def _solver_delegates_birth_task_transcripts(self) -> bool:
        """Whether the bounded solver deliberately leaves birth roots to this pool.

        Precomputed band sources can revalidate every receipt from immutable
        cache rows but must not retain one duplicate state entry per historical
        birth.  Existing online and immutable-tape solvers keep their original
        solver-owned transcript contract.
        """

        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        value = getattr(
            self._solver, "pool_owns_birth_task_transcripts", False
        )
        if type(value) is not bool:
            raise ActionBallContractError(
                "solver pool_owns_birth_task_transcripts must be an exact boolean"
            )
        return value

    def _solver_task_transcript_for_birth_pure(
        self, birth_sha256: str
    ) -> Tuple[int, str]:
        return self._solver_task_transcripts_for_births_pure(
            (birth_sha256,)
        )[0]

    def _solver_task_transcripts_for_births_pure(
        self, birth_sha256s: Sequence[str]
    ) -> Tuple[Tuple[int, str], ...]:
        """Read a birth batch under one solver purity envelope.

        ``state_dict()`` contains the complete sampler/task authority and can
        grow with the number of live births.  Taking that snapshot around
        every individual transcript therefore makes an otherwise vectorized
        request/retirement batch quadratic.  One pre/post snapshot enforces
        batch-net purity and rollback: persistent mutation and exceptions are
        rejected, while a transient mutation restored inside the same batch is
        outside this cheaper contract.  Per-row transcript count/root checks
        still bind every returned authority value.
        """

        if not birth_sha256s:
            return ()
        solver_state = self._solver_state()
        try:
            result = tuple(
                self._solver_task_transcript_for_birth(birth_sha256)
                for birth_sha256 in birth_sha256s
            )
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver birth task transcript authority must be pure"
                )
            return result
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def _expected_task_transcript_for_active_birth(
        self, action_uid: int, birth_sha256: str
    ) -> Tuple[int, str]:
        uid = self._binding(action_uid).action_uid
        digest = _sha256(birth_sha256, name="birth_sha256")
        pending = self._pending[uid][digest]
        root = self._issued_task_transcript_sha256[uid][digest]
        for receipt in pending:
            root = _task_transcript_extend(
                root, receipt.canonical_sha256
            )
        return self._cursor[uid][digest] + len(pending), root

    def _assert_all_task_transcripts_pure(self) -> None:
        """Cross-check compact active/retired roots with solver authority."""

        if self._solver_delegates_birth_task_transcripts():
            return

        solver_state = self._solver_state()
        try:
            for binding in self._bindings:
                uid = binding.action_uid
                for digest in self._births.get(uid, {}):
                    if (
                        self._solver_task_transcript_for_birth(digest)
                        != self._expected_task_transcript_for_active_birth(
                            uid, digest
                        )
                    ):
                        raise ActionBallContractError(
                            "active birth task transcript differs from "
                            "solver authority"
                        )
                for digest, retired in self._retired_births.get(
                    uid, {}
                ).items():
                    if self._solver_task_transcript_for_birth(digest) != (
                        retired.admitted_count,
                        retired.task_transcript_sha256,
                    ):
                        raise ActionBallContractError(
                            "retired birth task transcript differs from "
                            "solver authority"
                        )
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver birth task transcript authority must be pure"
                )
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def _validate_assignment_partition_for_action(
        self,
        *,
        action_uid: int,
        lifecycle: Sequence[int],
        births: Mapping[str, ActionBirthReceipt],
        pending: Mapping[str, Sequence[ActionBallTaskReceipt]],
        cursor: Mapping[str, int],
        refill_index: Mapping[str, int],
        proposed_by_birth: Mapping[str, int],
        sample_assignments: Mapping[
            str, Sequence[ActionSampleAssignment]
        ],
        retired_births: Mapping[str, _RetiredPoolBirth],
    ) -> None:
        """Prove every sampler index belongs to exactly one birth/refill."""

        uid = self._binding(action_uid).action_uid
        active_keys = set(births)
        if (
            set(pending) != active_keys
            or set(cursor) != active_keys
            or set(refill_index) != active_keys
            or set(proposed_by_birth) != active_keys
            or set(sample_assignments) != active_keys
        ):
            raise ActionBallContractError(
                "active birth lifecycle maps have different key sets"
            )
        if active_keys.intersection(retired_births):
            raise ActionBallContractError(
                "one birth cannot be both active and retired"
            )

        owner_by_index: Dict[int, str] = {}

        def claim_indices(
            birth_digest: str,
            assignments: Sequence[ActionSampleAssignment],
        ) -> Tuple[int, ...]:
            flattened: list[int] = []
            for assignment in assignments:
                if (
                    assignment.birth.canonical_sha256 != birth_digest
                    or assignment.birth.action_uid != uid
                ):
                    raise ActionBallContractError(
                        "sample assignment binds the wrong birth/action"
                    )
                flattened.extend(assignment.proposal_sample_indices)
            if tuple(sorted(set(flattened))) != tuple(flattened):
                raise ActionBallContractError(
                    "one birth's refill assignment tape is not strictly "
                    "increasing and unique"
                )
            for sample_index in flattened:
                if sample_index >= len(lifecycle):
                    raise ActionBallContractError(
                        "sample assignment exceeds lifecycle high-water"
                    )
                if sample_index in owner_by_index:
                    raise ActionBallContractError(
                        "sample index is assigned to multiple births/refills"
                    )
                owner_by_index[sample_index] = birth_digest
            return tuple(flattened)

        for birth_digest, birth in births.items():
            assignments = tuple(sample_assignments[birth_digest])
            expected_refills = list(
                range(1, refill_index[birth_digest] + 1)
            )
            if [
                assignment.refill_index for assignment in assignments
            ] != expected_refills:
                raise ActionBallContractError(
                    "active sample assignment refill indices are not "
                    "contiguous"
                )
            if any(assignment.birth != birth for assignment in assignments):
                raise ActionBallContractError(
                    "active sample assignment birth payload mismatch"
                )
            indices = claim_indices(birth_digest, assignments)
            if len(indices) != proposed_by_birth[birth_digest]:
                raise ActionBallContractError(
                    "active proposal count differs from sample assignments"
                )
            pending_indices = [
                receipt.sample_index
                for receipt in pending[birth_digest]
            ]
            if len(pending_indices) != len(set(pending_indices)):
                raise ActionBallContractError(
                    "active pending receipts repeat a sample index"
                )
            owned_pending = {
                sample_index
                for sample_index in indices
                if lifecycle[sample_index] == _LIFECYCLE_PENDING
            }
            if owned_pending != set(pending_indices):
                raise ActionBallContractError(
                    "active pending lifecycle differs from receipt samples"
                )
            status_counts = {
                status: sum(
                    lifecycle[sample_index] == status
                    for sample_index in indices
                )
                for status in range(4)
            }
            admitted = cursor[birth_digest] + len(
                pending[birth_digest]
            )
            if (
                status_counts[_LIFECYCLE_REJECTED]
                != len(indices) - admitted
                or status_counts[_LIFECYCLE_PENDING]
                != len(pending[birth_digest])
                or status_counts[_LIFECYCLE_ISSUED]
                != cursor[birth_digest]
                or status_counts[_LIFECYCLE_DISCARDED] != 0
            ):
                raise ActionBallContractError(
                    "active birth assignment statuses disagree with "
                    "cursor/pending/proposed counts"
                )

        for birth_digest, retired in retired_births.items():
            indices = claim_indices(
                birth_digest, retired.sample_assignments
            )
            status_counts = {
                status: sum(
                    lifecycle[sample_index] == status
                    for sample_index in indices
                )
                for status in range(4)
            }
            if (
                len(indices) != retired.proposed_count
                or status_counts[_LIFECYCLE_REJECTED]
                != retired.proposed_count - retired.admitted_count
                or status_counts[_LIFECYCLE_PENDING] != 0
                or status_counts[_LIFECYCLE_ISSUED]
                != retired.issued_count
                or status_counts[_LIFECYCLE_DISCARDED]
                != retired.discarded_count
            ):
                raise ActionBallContractError(
                    "retired birth assignment statuses disagree with "
                    "compact lifecycle counts"
                )

        if set(owner_by_index) != set(range(len(lifecycle))):
            raise ActionBallContractError(
                "sample assignments must partition every lifecycle index"
            )

    def _assert_compact_lifecycle_invariants(self) -> None:
        for binding in self._bindings:
            uid = binding.action_uid
            lifecycle = self._task_lifecycle.get(uid, [])
            last_sample_index = self._last_sample_index.get(uid, -1)
            if len(lifecycle) != last_sample_index + 1:
                raise ActionBallContractError(
                    "task lifecycle must cover sample indices "
                    "0..high-water"
                )
            active_refills = sum(
                self._refill_index[uid][digest]
                for digest in self._births.get(uid, {})
            )
            active_proposed = sum(
                self._proposed_by_birth[uid][digest]
                for digest in self._births.get(uid, {})
            )
            active_admitted = sum(
                self._cursor[uid][digest]
                + len(self._pending[uid][digest])
                for digest in self._births.get(uid, {})
            )
            active_issued = sum(
                self._cursor[uid][digest]
                for digest in self._births.get(uid, {})
            )
            retired_records = self._retired_births.get(uid, {}).values()
            retired_refills = sum(
                record.refill_index for record in retired_records
            )
            retired_records = self._retired_births.get(uid, {}).values()
            retired_proposed = sum(
                record.proposed_count for record in retired_records
            )
            retired_records = self._retired_births.get(uid, {}).values()
            retired_admitted = sum(
                record.admitted_count for record in retired_records
            )
            retired_records = self._retired_births.get(uid, {}).values()
            retired_issued = sum(
                record.issued_count for record in retired_records
            )
            retired_records = self._retired_births.get(uid, {}).values()
            retired_discarded = sum(
                record.discarded_count for record in retired_records
            )
            expected = PoolLedger(
                requests=active_issued + retired_issued,
                refill_calls=active_refills + retired_refills,
                proposed=active_proposed + retired_proposed,
                admitted=active_admitted + retired_admitted,
                issued=active_issued + retired_issued,
                discarded=retired_discarded,
            )
            actual = self._ledger.get(uid, PoolLedger())
            if actual != expected or actual.proposed != len(lifecycle):
                raise ActionBallContractError(
                    "pool ledger differs from compact per-birth lifecycle"
                )
            status_counts = {
                status: lifecycle.count(status)
                for status in range(4)
            }
            active_pending = sum(
                len(self._pending[uid][digest])
                for digest in self._births.get(uid, {})
            )
            if (
                status_counts[_LIFECYCLE_REJECTED]
                != actual.proposed - actual.admitted
                or status_counts[_LIFECYCLE_PENDING]
                != active_pending
                or status_counts[_LIFECYCLE_ISSUED] != actual.issued
                or status_counts[_LIFECYCLE_DISCARDED]
                != actual.discarded
            ):
                raise ActionBallContractError(
                    "2-bit task lifecycle statuses disagree with ledger"
                )
            self._validate_assignment_partition_for_action(
                action_uid=uid,
                lifecycle=lifecycle,
                births=self._births.get(uid, {}),
                pending=self._pending.get(uid, {}),
                cursor=self._cursor.get(uid, {}),
                refill_index=self._refill_index.get(uid, {}),
                proposed_by_birth=self._proposed_by_birth.get(uid, {}),
                sample_assignments=self._sample_assignments.get(uid, {}),
                retired_births=self._retired_births.get(uid, {}),
            )

    def _assert_emitted_tasks_pure(
        self, receipts: Sequence[ActionBallTaskReceipt]
    ) -> None:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        solver_state = self._solver_state()
        try:
            self._solver.assert_emitted_tasks(tuple(receipts))
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver exact-task authority assertion must be pure"
                )
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def _assert_proposal_assignments_pure(
        self, assignments: Sequence[ActionSampleAssignment]
    ) -> None:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        solver_state = self._solver_state()
        try:
            self._solver.assert_proposal_assignments(
                tuple(assignments)
            )
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver proposal-assignment authority must be pure"
                )
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def _restore_solver_state(self, state: object) -> None:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        detached = _json_data(state, name="solver state")
        self._solver.load_state_dict(detached)
        if self._solver_state() != detached:
            raise ActionBallContractError(
                "task solver load_state_dict did not restore exact state"
            )

    def bind_birth_authority(self, authority: object) -> None:
        """Bind the broker that proves Motion committed and Racket consumed."""

        if self._birth_authority is not None:
            raise PoolProtocolError(
                "birth authority may be bound only once"
            )
        if type(authority) is not ActionBirthBroker:
            raise ActionBallContractError(
                "birth authority must be the exact ActionBirthBroker type"
            )
        if getattr(authority, "registry_sha256", None) != self._registry_sha256:
            raise ActionBallContractError(
                "birth authority registry differs from task pool registry"
            )
        if authority.diagnostic_fast_path != self._diagnostic_fast_path:
            raise ActionBallContractError(
                "birth authority and task pool diagnostic modes differ"
            )
        self._birth_authority = authority

    def _birth_authority_state_sha256(self) -> str | None:
        if self._birth_authority is None:
            return None
        return _sha256_json(self._birth_authority.state_dict())

    def _validate_birth(
        self, birth: ActionBirthReceipt
    ) -> ActionBinding:
        if not isinstance(birth, ActionBirthReceipt):
            raise ActionBallContractError(
                "task pool requests require ActionBirthReceipt"
            )
        binding = self._binding(birth.action_uid)
        birth.assert_contract(
            binding=binding,
            pins=self._pins,
            mobility_mode=self._mobility_mode,
            registry_sha256=self._registry_sha256,
        )
        if self._birth_authority is None:
            raise PoolProtocolError("birth authority is not bound")
        self._birth_authority.assert_consumed_birth(birth)
        return binding

    def _ensure_birth(
        self,
        binding: ActionBinding,
        birth: ActionBirthReceipt,
    ) -> str:
        uid = binding.action_uid
        digest = birth.canonical_sha256
        if birth.reset_generation <= self._retired_generation.get(
            birth.env_id, 0
        ):
            raise PoolProtocolError(
                "cannot reactivate a retired/stale birth generation"
            )
        existing = self._births.get(uid, {}).get(digest)
        if existing is not None:
            if existing != birth:
                raise ActionBallContractError(
                    "birth SHA collision in task pool"
                )
            return digest
        # The broker guarantees one action per env/reset generation.  Preserve
        # that invariant even if a caller accidentally presents two different
        # birth payloads with the same logical identity.
        if self._diagnostic_fast_path:
            if birth.env_id in self._diagnostic_birth_by_env:
                raise ActionBallContractError(
                    "task pool already has an active birth for this env; "
                    "retire it before the next true reset"
                )
        else:
            for action_births in self._births.values():
                for active in action_births.values():
                    if active.env_id == birth.env_id:
                        raise ActionBallContractError(
                            "task pool already has an active birth for this "
                            "env; retire it before the next true reset"
                        )
        self._births.setdefault(uid, {})[digest] = birth
        self._pending.setdefault(uid, {})[digest] = []
        self._cursor.setdefault(uid, {})[digest] = 0
        self._refill_index.setdefault(uid, {})[digest] = 0
        if not self._diagnostic_fast_path:
            self._issued_task_transcript_sha256.setdefault(uid, {})[
                digest
            ] = task_transcript_sha256(digest, ())
            self._proposed_by_birth.setdefault(uid, {})[digest] = 0
            self._sample_assignments.setdefault(uid, {})[digest] = []
        self._seen_sha256.setdefault(uid, {})[digest] = set()
        self._seen_sample_sha256.setdefault(uid, {})[digest] = set()
        if uid not in self._ledger:
            self._ledger[uid] = PoolLedger()
        if self._diagnostic_fast_path:
            self._diagnostic_birth_by_env[birth.env_id] = (uid, digest)
        return digest

    def _build_refill_request(
        self,
        binding: ActionBinding,
        birth: ActionBirthReceipt,
        birth_digest: str,
    ) -> ActionPoolRefillRequest:
        uid = binding.action_uid
        next_refill = self._refill_index[uid][birth_digest] + 1
        seen = self._seen_sha256[uid][birth_digest]
        swing_generation_start = len(seen)
        return ActionPoolRefillRequest(
            action_uid=uid,
            action_slot=binding.action_slot,
            refill_index=next_refill,
            minimum_receipts=self._refill_size,
            swing_generation_start=swing_generation_start,
            mobility_mode=self._mobility_mode,
            binding=binding,
            pins=self._pins,
            birth=birth,
            registry_sha256=self._registry_sha256,
        )

    def _validate_refill_batch(
        self,
        *,
        binding: ActionBinding,
        birth: ActionBirthReceipt,
        birth_digest: str,
        request: ActionPoolRefillRequest,
        batch: ActionPoolRefillBatch,
        unavailable_sample_sha256: set[str] | None = None,
        sample_index_floor: int | None = None,
        sample_draw_floor: int | None = None,
        staged_sample_indices: set[int] | None = None,
        staged_sample_draw_highwater: int | None = None,
        staged_sample_draw_starts: set[int] | None = None,
        staged_sample_draw_ranges: Sequence[Tuple[int, int]] = (),
        verify_solver_provenance: bool = True,
    ) -> Tuple[Tuple[str, ...], int, int]:
        uid = binding.action_uid
        if not isinstance(batch, ActionPoolRefillBatch):
            raise ActionBallContractError(
                "task solver must return ActionPoolRefillBatch"
            )
        if batch.action_uid != uid:
            raise ActionBallContractError(
                "task solver refilled a different action_uid"
            )
        if not batch.receipts:
            raise PoolProtocolError(
                f"task solver admitted no receipts for action_uid {uid}"
            )
        if len(batch.receipts) < request.minimum_receipts:
            raise PoolProtocolError(
                f"task solver admitted {len(batch.receipts)} receipts for "
                f"action_uid {uid}, below requested minimum "
                f"{request.minimum_receipts}"
            )
        seen = self._seen_sha256[uid][birth_digest]
        seen_samples = self._seen_sample_sha256[uid][birth_digest]
        unavailable_samples = (
            {
                sample_digest
                for action_births in self._seen_sample_sha256.values()
                for birth_samples in action_births.values()
                for sample_digest in birth_samples
            }
            if unavailable_sample_sha256 is None
            else unavailable_sample_sha256
        )
        new_digests: list[str] = []
        new_sample_digests: list[str] = []
        last_sample_index = self._last_sample_index.get(uid, -1)
        last_sample_draw_end = self._last_sample_draw_end.get(uid, 0)
        if sample_index_floor is not None:
            last_sample_index = sample_index_floor
        if sample_draw_floor is not None:
            last_sample_draw_end = sample_draw_floor
        floor_sample_index = last_sample_index
        floor_sample_draw_end = last_sample_draw_end
        batch_sample_index = last_sample_index
        batch_sample_draw_end = last_sample_draw_end
        staged_indices = (
            set() if staged_sample_indices is None else staged_sample_indices
        )
        for offset, receipt in enumerate(batch.receipts):
            receipt.assert_contract(
                binding=binding,
                pins=self._pins,
                mobility_mode=self._mobility_mode,
                registry_sha256=self._registry_sha256,
            )
            receipt.assert_birth(birth)
            if verify_solver_provenance:
                if self._solver is None:
                    raise PoolProtocolError("task solver is not bound")
                self._solver.assert_emitted_sample(receipt)
            if (
                receipt.sample_index <= floor_sample_index
                or receipt.sample_index <= batch_sample_index
                or receipt.sample_index in staged_indices
            ):
                raise ActionBallContractError(
                    "task solver sample index replayed/went backwards"
                )
            if (
                receipt.sample_draw_start < floor_sample_draw_end
                or receipt.sample_draw_start < batch_sample_draw_end
                # Diagnostic solve-many batches normally advance in tape
                # order.  A later redraw can still return one earlier,
                # disjoint admitted range after a higher range, so retain
                # exact overlap semantics with constant-width start probes.
                or (
                    staged_sample_draw_highwater is not None
                    and staged_sample_draw_starts is not None
                    and receipt.sample_draw_start
                    < staged_sample_draw_highwater
                    and any(
                        prior_start in staged_sample_draw_starts
                        for prior_start in range(
                            receipt.sample_draw_start
                            - SAMPLER_SAMPLE_DRAW_COUNT
                            + 1,
                            receipt.sample_draw_end,
                        )
                    )
                )
                or any(
                    receipt.sample_draw_start < prior_end
                    and prior_start < receipt.sample_draw_end
                    for prior_start, prior_end in staged_sample_draw_ranges
                )
            ):
                raise ActionBallContractError(
                    "task solver sample draw range replayed/overlapped"
                )
            batch_sample_index = receipt.sample_index
            batch_sample_draw_end = receipt.sample_draw_end
            last_sample_index = max(
                last_sample_index, receipt.sample_index
            )
            last_sample_draw_end = max(
                last_sample_draw_end, receipt.sample_draw_end
            )
            if (
                receipt.swing_generation
                != request.swing_generation_start + offset
            ):
                raise ActionBallContractError(
                    "task solver returned non-contiguous/wrong swing "
                    "generation"
                )
            digest = receipt.canonical_sha256
            if digest in seen or digest in new_digests:
                raise ActionBallContractError(
                    "task solver returned a replayed/duplicate receipt"
                )
            new_digests.append(digest)
            if (
                receipt.sample_sha256 in seen_samples
                or receipt.sample_sha256 in unavailable_samples
                or receipt.sample_sha256 in new_sample_digests
            ):
                raise ActionBallContractError(
                    "task solver reused one sampler sample receipt"
                )
            new_sample_digests.append(receipt.sample_sha256)
        return (
            tuple(new_digests),
            last_sample_index,
            last_sample_draw_end,
        )

    def _install_refill_batch(
        self,
        *,
        binding: ActionBinding,
        birth_digest: str,
        request: ActionPoolRefillRequest,
        batch: ActionPoolRefillBatch,
        new_digests: Sequence[str],
        last_sample_index: int,
        last_sample_draw_end: int,
    ) -> None:
        uid = binding.action_uid
        current = self._ledger.get(uid, PoolLedger())
        updated = PoolLedger(
            requests=current.requests,
            refill_calls=current.refill_calls + 1,
            proposed=current.proposed + batch.proposed_count,
            admitted=current.admitted + len(batch.receipts),
            issued=current.issued,
            discarded=current.discarded,
        )
        self._pending[uid][birth_digest].extend(batch.receipts)
        self._seen_sha256[uid][birth_digest].update(new_digests)
        self._seen_sample_sha256[uid][birth_digest].update(
            receipt.sample_sha256 for receipt in batch.receipts
        )
        self._refill_index[uid][
            birth_digest
        ] = request.refill_index
        self._proposed_by_birth[uid][birth_digest] += (
            batch.proposed_count
        )
        self._sample_assignments[uid][birth_digest].append(
            ActionSampleAssignment(
                birth=self._births[uid][birth_digest],
                refill_index=request.refill_index,
                proposal_sample_indices=batch.proposal_sample_indices,
            )
        )
        self._ledger[uid] = updated
        self._last_sample_index[uid] = last_sample_index
        self._last_sample_draw_end[uid] = last_sample_draw_end

    def _install_refill_batch_diagnostic(
        self,
        *,
        binding: ActionBinding,
        birth_digest: str,
        request: ActionPoolRefillRequest,
        batch: ActionPoolRefillBatch,
        new_digests: Sequence[str],
        last_sample_index: int,
        last_sample_draw_end: int,
    ) -> None:
        """Install one diagnostic refill without formal proof scaffolding.

        Diagnostic pools cannot publish an exact-resume ``state_dict`` and
        deliberately omit the compact lifecycle.  Keeping a per-birth
        transcript root, proposal-assignment object, and proposed counter in
        that mode therefore paid JSON/SHA and dataclass costs on every reset
        without being consumed.  FIFO, replay detection, sampler high-water,
        and the aggregate P/A/issued/discarded ledger remain authoritative.
        """

        if not self._diagnostic_fast_path:
            raise ActionBallContractError(
                "diagnostic refill installer requires diagnostic mode"
            )
        uid = binding.action_uid
        self._pending[uid][birth_digest].extend(batch.receipts)
        self._seen_sha256[uid][birth_digest].update(new_digests)
        self._seen_sample_sha256[uid][birth_digest].update(
            receipt.sample_sha256 for receipt in batch.receipts
        )
        self._refill_index[uid][birth_digest] = request.refill_index
        self._last_sample_index[uid] = last_sample_index
        self._last_sample_draw_end[uid] = last_sample_draw_end

    def _install_lifecycle_samples(
        self,
        *,
        action_uid: int,
        batches: Sequence[ActionPoolRefillBatch],
        authority_sample_index: int,
    ) -> None:
        """Append every proposal once, then mark admitted samples pending."""

        uid = self._binding(action_uid).action_uid
        current = self._task_lifecycle.get(uid, [])
        proposed = sum(batch.proposed_count for batch in batches)
        expected_new = authority_sample_index + 1 - len(current)
        if expected_new != proposed:
            raise ActionBallContractError(
                "solver proposed_count must exactly equal newly issued "
                "sampler sample indices"
            )
        proposal_indices = [
            sample_index
            for batch in batches
            for sample_index in batch.proposal_sample_indices
        ]
        expected_indices = list(
            range(len(current), authority_sample_index + 1)
        )
        if sorted(proposal_indices) != expected_indices or len(
            proposal_indices
        ) != len(set(proposal_indices)):
            raise ActionBallContractError(
                "refill proposal sample indices must uniquely and "
                "completely cover the new sampler tape"
            )
        lifecycle = [
            *current,
            *([_LIFECYCLE_REJECTED] * proposed),
        ]
        admitted_indices: set[int] = set()
        for batch in batches:
            for receipt in batch.receipts:
                sample_index = receipt.sample_index
                if (
                    sample_index in admitted_indices
                    or sample_index < 0
                    or sample_index >= len(lifecycle)
                    or lifecycle[sample_index] != _LIFECYCLE_REJECTED
                ):
                    raise ActionBallContractError(
                        "admitted sample cannot be installed into lifecycle"
                    )
                admitted_indices.add(sample_index)
                lifecycle[sample_index] = _LIFECYCLE_PENDING
        self._task_lifecycle[uid] = lifecycle

    def _refill(
        self,
        binding: ActionBinding,
        birth: ActionBirthReceipt,
        birth_digest: str,
    ) -> None:
        if self._solver is None:
            raise PoolProtocolError("task solver is not bound")
        request = self._build_refill_request(
            binding, birth, birth_digest
        )
        solver_state = self._solver_state()
        try:
            previous_highwaters = self._solver_sample_highwaters()
            if previous_highwaters != self._pool_sample_highwaters():
                raise ActionBallContractError(
                    "pool sample high-water differs from solver authority"
                )
            previous_task_counts = self._solver_emitted_task_counts()
            if previous_task_counts != self._pool_emitted_task_counts():
                raise ActionBallContractError(
                    "pool admitted-task counts differ from solver authority"
                )
            batch = self._solver(request)
            emitted_solver_state = self._solver_state()
            authority_highwaters = self._solver_sample_highwaters()
            authority_task_counts = self._solver_emitted_task_counts()
            authority_highwater = authority_highwaters[
                binding.action_uid
            ]
            if any(
                authority_highwaters[uid] != prior
                for uid, prior in previous_highwaters.items()
                if uid != binding.action_uid
            ):
                raise ActionBallContractError(
                    "solver advanced an unstaged action sample tape"
                )
            if any(
                authority_task_counts[uid] != prior
                for uid, prior in previous_task_counts.items()
                if uid != binding.action_uid
            ) or authority_task_counts[binding.action_uid] != (
                previous_task_counts[binding.action_uid]
                + len(batch.receipts)
            ):
                raise ActionBallContractError(
                    "solver admitted-task transcript advanced outside the "
                    "staged callback result"
                )
            (
                new_digests,
                last_sample_index,
                last_sample_draw_end,
            ) = self._validate_refill_batch(
                binding=binding,
                birth=birth,
                birth_digest=birth_digest,
                request=request,
                batch=batch,
            )
            self._assert_emitted_tasks_pure(batch.receipts)
            self._assert_proposal_assignments_pure(
                (
                    ActionSampleAssignment(
                        birth=birth,
                        refill_index=request.refill_index,
                        proposal_sample_indices=(
                            batch.proposal_sample_indices
                        ),
                    ),
                )
            )
            expected_count, expected_root = (
                self._expected_task_transcript_for_active_birth(
                    binding.action_uid, birth_digest
                )
            )
            for receipt in batch.receipts:
                expected_root = _task_transcript_extend(
                    expected_root, receipt.canonical_sha256
                )
            expected_count += len(batch.receipts)
            if (
                not self._solver_delegates_birth_task_transcripts()
                and self._solver_task_transcript_for_birth_pure(
                    birth_digest
                )
                != (expected_count, expected_root)
            ):
                raise ActionBallContractError(
                    "solver birth task transcript differs from staged "
                    "callback result"
                )
            previous_highwater = (
                self._last_sample_index.get(binding.action_uid, -1),
                self._last_sample_draw_end.get(binding.action_uid, 0),
            )
            if (
                authority_highwater[0] < last_sample_index
                or authority_highwater[1] < last_sample_draw_end
                or authority_highwater[0] < previous_highwater[0]
                or authority_highwater[1] < previous_highwater[1]
            ):
                raise ActionBallContractError(
                    "solver sample high-water disagrees with emitted samples"
                )
            if self._solver_state() != emitted_solver_state:
                raise ActionBallContractError(
                    "solver sample authority assertion must be pure"
                )
            # Commit only after the whole callback result validates.
            self._install_lifecycle_samples(
                action_uid=binding.action_uid,
                batches=(batch,),
                authority_sample_index=authority_highwater[0],
            )
            self._install_refill_batch(
                binding=binding,
                birth_digest=birth_digest,
                request=request,
                batch=batch,
                new_digests=new_digests,
                last_sample_index=authority_highwater[0],
                last_sample_draw_end=authority_highwater[1],
            )
        except Exception:
            self._restore_solver_state(solver_state)
            raise

    def request(
        self,
        birth: ActionBirthReceipt,
        *,
        swing_generation: int,
    ) -> ActionBallTaskReceipt:
        """Issue the next task for one exact env/episode birth.

        The top-level allocation remains lazy by action UID, while concurrent
        environments using that action get independent birth-SHA subqueues.
        """

        return self.request_many(
            (
                ActionTaskIssueRequest(
                    birth=birth,
                    swing_generation=swing_generation,
                ),
            )
        )[0]

    def _rollback_empty_birth(
        self, uid: int, birth_digest: str
    ) -> None:
        if (
            self._cursor[uid][birth_digest] != 0
            or self._refill_index[uid][birth_digest] != 0
            or self._pending[uid][birth_digest]
            or self._seen_sha256[uid][birth_digest]
            or self._seen_sample_sha256[uid][birth_digest]
        ):
            return
        if not self._diagnostic_fast_path and (
            self._proposed_by_birth[uid][birth_digest] != 0
            or self._sample_assignments[uid][birth_digest]
            or self._issued_task_transcript_sha256[uid][birth_digest]
            != task_transcript_sha256(birth_digest, ())
        ):
            return
        birth_env_id = self._births[uid][birth_digest].env_id
        active_tables = (
            self._births,
            self._pending,
            self._cursor,
            self._refill_index,
            self._seen_sha256,
            self._seen_sample_sha256,
        )
        if not self._diagnostic_fast_path:
            active_tables = (
                *active_tables,
                self._issued_task_transcript_sha256,
                self._proposed_by_birth,
                self._sample_assignments,
            )
        for table in active_tables:
            del table[uid][birth_digest]
            if not table[uid]:
                del table[uid]
        if self._diagnostic_fast_path:
            self._diagnostic_birth_by_env.pop(birth_env_id, None)
        if self._ledger.get(uid) == PoolLedger():
            del self._ledger[uid]

    def _request_many_diagnostic(
        self,
        converted: Tuple[ActionTaskIssueRequest, ...],
    ) -> Tuple[ActionBallTaskReceipt, ...]:
        """Issue diagnostic tasks without formal proof/replay hot-path work.

        The same solver callback, receipt contracts, fixed action identity,
        and sample/task counters remain authoritative.  This path deliberately
        does not snapshot or restore the solver: a malformed diagnostic
        callback fails the run instead of attempting an exact-resume rollback.
        """

        validated: list[
            Tuple[
                ActionTaskIssueRequest,
                ActionBinding,
                int,
                str,
            ]
        ] = []
        request_births: set[str] = set()
        for request in converted:
            binding = self._validate_birth(request.birth)
            uid = binding.action_uid
            digest = request.birth.canonical_sha256
            if digest in request_births:
                raise PoolProtocolError(
                    "task issue batch repeats one birth"
                )
            request_births.add(digest)
            expected_generation = self._cursor.get(uid, {}).get(
                digest, 0
            )
            if request.swing_generation != expected_generation:
                raise PoolProtocolError(
                    f"birth task swing generation must be exactly "
                    f"{expected_generation}, got "
                    f"{request.swing_generation}"
                )
            validated.append((request, binding, uid, digest))

        registered: list[Tuple[int, str]] = []
        try:
            for request, binding, uid, _digest in validated:
                before = request.birth.canonical_sha256 in self._births.get(
                    uid, {}
                )
                digest = self._ensure_birth(binding, request.birth)
                if not before:
                    registered.append((uid, digest))

            refills: list[
                Tuple[
                    ActionBinding,
                    ActionBirthReceipt,
                    str,
                    ActionPoolRefillRequest,
                ]
            ] = []
            for request, binding, uid, digest in validated:
                if not self._pending[uid][digest]:
                    refills.append(
                        (
                            binding,
                            request.birth,
                            digest,
                            self._build_refill_request(
                                binding, request.birth, digest
                            ),
                        )
                    )

            batches: Tuple[ActionPoolRefillBatch, ...]
            if not refills:
                batches = ()
            elif len(refills) == 1:
                if self._solver is None:
                    raise PoolProtocolError("task solver is not bound")
                batches = (self._solver(refills[0][3]),)
            else:
                solve_many = (
                    None
                    if self._solver is None
                    else getattr(self._solver, "solve_many", None)
                )
                if not callable(solve_many):
                    raise PoolProtocolError(
                        "multi-birth request requires solver.solve_many()"
                    )
                raw_batches = solve_many(
                    tuple(refill[3] for refill in refills)
                )
                if not isinstance(raw_batches, (tuple, list)):
                    raise ActionBallContractError(
                        "solver.solve_many() must return a tuple/list"
                    )
                batches = tuple(raw_batches)
                if len(batches) != len(refills):
                    raise ActionBallContractError(
                        "solver.solve_many() returned wrong batch count"
                    )

            unavailable_samples = (
                self._diagnostic_active_sample_sha256
            )
            staged_sample_digests: set[str] = set()
            staged_sample_indices_by_uid: Dict[int, set[int]] = {}
            staged_sample_draw_highwater_by_uid: Dict[int, int] = {}
            staged_sample_draw_starts_by_uid: Dict[int, set[int]] = {}
            proposal_indices_by_uid: Dict[int, list[int]] = {}
            staged_refills = []
            for refill, batch in zip(refills, batches):
                binding, birth, digest, refill_request = refill
                uid = binding.action_uid
                staged_indices = staged_sample_indices_by_uid.setdefault(
                    uid, set()
                )
                staged_draw_highwater = (
                    staged_sample_draw_highwater_by_uid.get(
                        uid, self._last_sample_draw_end.get(uid, 0)
                    )
                )
                staged_draw_starts = (
                    staged_sample_draw_starts_by_uid.setdefault(uid, set())
                )
                sample_draw_floor = self._last_sample_draw_end.get(uid, 0)
                (
                    new_digests,
                    last_sample_index,
                    last_sample_draw_end,
                ) = self._validate_refill_batch(
                    binding=binding,
                    birth=birth,
                    birth_digest=digest,
                    request=refill_request,
                    batch=batch,
                    unavailable_sample_sha256=unavailable_samples,
                    sample_index_floor=self._last_sample_index.get(uid, -1),
                    sample_draw_floor=sample_draw_floor,
                    staged_sample_indices=staged_indices,
                    staged_sample_draw_highwater=staged_draw_highwater,
                    staged_sample_draw_starts=staged_draw_starts,
                    verify_solver_provenance=False,
                )
                staged_indices.update(
                    receipt.sample_index for receipt in batch.receipts
                )
                staged_sample_draw_highwater_by_uid[uid] = max(
                    staged_draw_highwater, last_sample_draw_end
                )
                staged_draw_starts.update(
                    receipt.sample_draw_start
                    for receipt in batch.receipts
                )
                for receipt in batch.receipts:
                    if receipt.sample_sha256 in staged_sample_digests:
                        raise ActionBallContractError(
                            "diagnostic refill reused one staged sampler "
                            "sample receipt"
                        )
                    staged_sample_digests.add(receipt.sample_sha256)
                proposal_indices_by_uid.setdefault(uid, []).extend(
                    batch.proposal_sample_indices
                )
                staged_refills.append(
                    (
                        binding,
                        digest,
                        refill_request,
                        batch,
                        new_digests,
                        last_sample_index,
                        last_sample_draw_end,
                    )
                )

            authority_highwaters: Dict[int, Tuple[int, int]] = {}
            for uid, proposal_indices in proposal_indices_by_uid.items():
                authority = self._solver_sample_highwater(uid)
                previous = (
                    self._last_sample_index.get(uid, -1),
                    self._last_sample_draw_end.get(uid, 0),
                )
                if tuple(sorted(proposal_indices)) != tuple(
                    range(previous[0] + 1, authority[0] + 1)
                ):
                    raise ActionBallContractError(
                        "diagnostic refill proposals do not exactly advance "
                        "the action sample tape"
                    )
                if authority[1] < previous[1]:
                    raise ActionBallContractError(
                        "diagnostic solver sample draw high-water went "
                        "backwards"
                    )
                authority_highwaters[uid] = authority

            refill_deltas: Dict[int, Tuple[int, int, int]] = {}
            for (
                binding,
                digest,
                refill_request,
                batch,
                new_digests,
                last_sample_index,
                last_sample_draw_end,
            ) in staged_refills:
                authority = authority_highwaters[binding.action_uid]
                if (
                    authority[0] < last_sample_index
                    or authority[1] < last_sample_draw_end
                ):
                    raise ActionBallContractError(
                        "diagnostic solver sample high-water does not cover "
                        "its admitted receipts"
                    )
                self._install_refill_batch_diagnostic(
                    binding=binding,
                    birth_digest=digest,
                    request=refill_request,
                    batch=batch,
                    new_digests=new_digests,
                    last_sample_index=last_sample_index,
                    last_sample_draw_end=last_sample_draw_end,
                )
                calls, proposed, admitted = refill_deltas.get(
                    binding.action_uid, (0, 0, 0)
                )
                refill_deltas[binding.action_uid] = (
                    calls + 1,
                    proposed + batch.proposed_count,
                    admitted + len(batch.receipts),
                )
            for uid, (calls, proposed, admitted) in (
                refill_deltas.items()
            ):
                current = self._ledger[uid]
                self._ledger[uid] = PoolLedger(
                    requests=current.requests,
                    refill_calls=current.refill_calls + calls,
                    proposed=current.proposed + proposed,
                    admitted=current.admitted + admitted,
                    issued=current.issued,
                    discarded=current.discarded,
                )
            for uid, (sample_index, draw_end) in (
                authority_highwaters.items()
            ):
                self._last_sample_index[uid] = sample_index
                self._last_sample_draw_end[uid] = draw_end
            self._diagnostic_active_sample_sha256.update(
                staged_sample_digests
            )
        except Exception:
            for uid, digest in reversed(registered):
                if digest in self._births.get(uid, {}):
                    self._rollback_empty_birth(uid, digest)
            raise

        issued: list[ActionBallTaskReceipt] = []
        for request, _binding, uid, digest in validated:
            pending = self._pending[uid][digest]
            if not pending:
                raise PoolProtocolError(
                    "diagnostic solver left no admitted task to issue"
                )
            receipt = pending[0]
            if receipt.swing_generation != request.swing_generation:
                raise ActionBallContractError(
                    "pending task receipt swing generation disagrees with "
                    "pool cursor"
                )
            issued.append(receipt)
        issued_by_uid: Dict[int, int] = {}
        for (
            _request,
            _binding,
            uid,
            digest,
        ), receipt in zip(validated, issued):
            self._pending[uid][digest].pop(0)
            self._cursor[uid][digest] += 1
            issued_by_uid[uid] = issued_by_uid.get(uid, 0) + 1
        for uid, issued_count in issued_by_uid.items():
            current = self._ledger[uid]
            self._ledger[uid] = PoolLedger(
                requests=current.requests + issued_count,
                refill_calls=current.refill_calls,
                proposed=current.proposed,
                admitted=current.admitted,
                issued=current.issued + issued_count,
                discarded=current.discarded,
            )
        return tuple(issued)

    def request_many(
        self,
        requests: Sequence[ActionTaskIssueRequest],
    ) -> Tuple[ActionBallTaskReceipt, ...]:
        """Issue an env batch with at most one vectorized solver callback.

        A solver used with more than one empty birth must implement
        ``solve_many(tuple[ActionPoolRefillRequest, ...])``.  All callback
        batches validate before any pool refill or issue commits.
        """

        if isinstance(requests, (str, bytes)) or not isinstance(
            requests, Sequence
        ):
            raise ActionBallContractError(
                "task issue requests must be a non-empty sequence"
            )
        converted = tuple(requests)
        if not converted or any(
            not isinstance(request, ActionTaskIssueRequest)
            for request in converted
        ):
            raise ActionBallContractError(
                "task issue requests must be non-empty "
                "ActionTaskIssueRequest objects"
            )
        if self._diagnostic_fast_path:
            return self._request_many_diagnostic(converted)
        validated: list[
            Tuple[
                ActionTaskIssueRequest,
                ActionBinding,
                int,
                str,
            ]
        ] = []
        request_births: set[str] = set()
        for request in converted:
            binding = self._validate_birth(request.birth)
            uid = binding.action_uid
            digest = request.birth.canonical_sha256
            if digest in request_births:
                raise PoolProtocolError(
                    "task issue batch repeats one birth"
                )
            request_births.add(digest)
            expected_generation = self._cursor.get(uid, {}).get(
                digest, 0
            )
            if request.swing_generation != expected_generation:
                raise PoolProtocolError(
                    f"birth task swing generation must be exactly "
                    f"{expected_generation}, got "
                    f"{request.swing_generation}"
                )
            validated.append((request, binding, uid, digest))

        solver_state = self._solver_state()
        registered: list[Tuple[int, str]] = []
        try:
            previous_highwaters = self._solver_sample_highwaters()
            if previous_highwaters != self._pool_sample_highwaters():
                raise ActionBallContractError(
                    "pool sample high-water differs from solver authority"
                )
            previous_task_counts = self._solver_emitted_task_counts()
            if previous_task_counts != self._pool_emitted_task_counts():
                raise ActionBallContractError(
                    "pool admitted-task counts differ from solver authority"
                )
            for request, binding, uid, _digest in validated:
                before = request.birth.canonical_sha256 in self._births.get(
                    uid, {}
                )
                digest = self._ensure_birth(binding, request.birth)
                if not before:
                    registered.append((uid, digest))

            refills: list[
                Tuple[
                    ActionBinding,
                    ActionBirthReceipt,
                    str,
                    ActionPoolRefillRequest,
                ]
            ] = []
            for request, binding, uid, digest in validated:
                if not self._pending[uid][digest]:
                    refills.append(
                        (
                            binding,
                            request.birth,
                            digest,
                            self._build_refill_request(
                                binding, request.birth, digest
                            ),
                        )
                    )
            batches: Tuple[ActionPoolRefillBatch, ...]
            if not refills:
                batches = ()
            elif len(refills) == 1:
                if self._solver is None:
                    raise PoolProtocolError("task solver is not bound")
                batches = (self._solver(refills[0][3]),)
            else:
                solve_many = (
                    None
                    if self._solver is None
                    else getattr(self._solver, "solve_many", None)
                )
                if not callable(solve_many):
                    raise PoolProtocolError(
                        "multi-birth request requires solver.solve_many()"
                    )
                raw_batches = solve_many(
                    tuple(refill[3] for refill in refills)
                )
                if not isinstance(raw_batches, (tuple, list)):
                    raise ActionBallContractError(
                        "solver.solve_many() must return a tuple/list"
                    )
                batches = tuple(raw_batches)
                if len(batches) != len(refills):
                    raise ActionBallContractError(
                        "solver.solve_many() returned wrong batch count"
                    )
            emitted_solver_state = self._solver_state()
            authority_highwaters = self._solver_sample_highwaters()
            authority_task_counts = self._solver_emitted_task_counts()
            staged_uids = {
                binding.action_uid
                for binding, _birth, _digest, _request in refills
            }
            if any(
                authority_highwaters[uid] != prior
                for uid, prior in previous_highwaters.items()
                if uid not in staged_uids
            ):
                raise ActionBallContractError(
                    "solver advanced an unstaged action sample tape"
                )

            staged_refills = []
            unavailable_samples = {
                sample_digest
                for action_births in self._seen_sample_sha256.values()
                for birth_samples in action_births.values()
                for sample_digest in birth_samples
            }
            projected_sample_highwater = {
                uid: (
                    self._last_sample_index.get(uid, -1),
                    self._last_sample_draw_end.get(uid, 0),
                )
                for _request, _binding, uid, _digest in validated
            }
            staged_sample_indices_by_uid: Dict[int, set[int]] = {}
            staged_sample_draw_ranges_by_uid: Dict[
                int, list[Tuple[int, int]]
            ] = {}
            for refill, batch in zip(refills, batches):
                binding, birth, digest, refill_request = refill
                uid = binding.action_uid
                sample_index_floor = self._last_sample_index.get(uid, -1)
                sample_draw_floor = self._last_sample_draw_end.get(uid, 0)
                staged_indices = staged_sample_indices_by_uid.setdefault(
                    uid, set()
                )
                staged_ranges = (
                    staged_sample_draw_ranges_by_uid.setdefault(uid, [])
                )
                (
                    new_digests,
                    last_sample_index,
                    last_sample_draw_end,
                ) = self._validate_refill_batch(
                    binding=binding,
                    birth=birth,
                    birth_digest=digest,
                    request=refill_request,
                    batch=batch,
                    unavailable_sample_sha256=unavailable_samples,
                    sample_index_floor=sample_index_floor,
                    sample_draw_floor=sample_draw_floor,
                    staged_sample_indices=staged_indices,
                    staged_sample_draw_ranges=staged_ranges,
                )
                old_index, old_draw_end = projected_sample_highwater[uid]
                projected_sample_highwater[uid] = (
                    max(old_index, last_sample_index),
                    max(old_draw_end, last_sample_draw_end),
                )
                staged_indices.update(
                    receipt.sample_index for receipt in batch.receipts
                )
                staged_ranges.extend(
                    (
                        receipt.sample_draw_start,
                        receipt.sample_draw_end,
                    )
                    for receipt in batch.receipts
                )
                unavailable_samples.update(
                    receipt.sample_sha256 for receipt in batch.receipts
                )
                staged_refills.append(
                    (
                        binding,
                        digest,
                        refill_request,
                        batch,
                        new_digests,
                        last_sample_index,
                        last_sample_draw_end,
                    )
                )
            returned_task_counts: Dict[int, int] = {}
            returned_receipts: list[ActionBallTaskReceipt] = []
            for (
                binding,
                _digest,
                _request,
                batch,
                _digests,
                _sample_index,
                _draw_end,
            ) in staged_refills:
                returned_task_counts[binding.action_uid] = (
                    returned_task_counts.get(binding.action_uid, 0)
                    + len(batch.receipts)
                )
                returned_receipts.extend(batch.receipts)
            if any(
                authority_task_counts[uid]
                != previous_task_counts[uid]
                + returned_task_counts.get(uid, 0)
                for uid in previous_task_counts
            ):
                raise ActionBallContractError(
                    "solver admitted-task transcript advanced outside the "
                    "staged callback results"
                )
            self._assert_emitted_tasks_pure(returned_receipts)
            self._assert_proposal_assignments_pure(
                tuple(
                    ActionSampleAssignment(
                        birth=refill_request.birth,
                        refill_index=refill_request.refill_index,
                        proposal_sample_indices=(
                            batch.proposal_sample_indices
                        ),
                    )
                    for (
                        _binding,
                        _digest,
                        refill_request,
                        batch,
                        _digests,
                        _sample_index,
                        _draw_end,
                    ) in staged_refills
                )
            )
            if not self._solver_delegates_birth_task_transcripts():
                authority_transcripts = (
                    self._solver_task_transcripts_for_births_pure(
                        tuple(
                            digest
                            for (
                                _binding,
                                digest,
                                _request,
                                _batch,
                                _digests,
                                _sample_index,
                                _draw_end,
                            ) in staged_refills
                        )
                    )
                )
                for (
                    (
                        binding,
                        digest,
                        _request,
                        batch,
                        _digests,
                        _sample_index,
                        _draw_end,
                    ),
                    authority_transcript,
                ) in zip(staged_refills, authority_transcripts):
                    expected_count, expected_root = (
                        self._expected_task_transcript_for_active_birth(
                            binding.action_uid, digest
                        )
                    )
                    for receipt in batch.receipts:
                        expected_root = _task_transcript_extend(
                            expected_root, receipt.canonical_sha256
                        )
                    expected_count += len(batch.receipts)
                    if authority_transcript != (
                        expected_count,
                        expected_root,
                    ):
                        raise ActionBallContractError(
                            "solver birth task transcript differs from staged "
                            "callback result"
                        )
            for uid in staged_sample_indices_by_uid:
                authority_highwater = authority_highwaters[uid]
                emitted_highwater = projected_sample_highwater[uid]
                previous_highwater = (
                    self._last_sample_index.get(uid, -1),
                    self._last_sample_draw_end.get(uid, 0),
                )
                if (
                    authority_highwater[0] < emitted_highwater[0]
                    or authority_highwater[1] < emitted_highwater[1]
                    or authority_highwater[0] < previous_highwater[0]
                    or authority_highwater[1] < previous_highwater[1]
                ):
                    raise ActionBallContractError(
                        "solver sample high-water disagrees with emitted "
                        "samples"
                    )
                projected_sample_highwater[uid] = (
                    authority_highwater
                )
            if self._solver_state() != emitted_solver_state:
                raise ActionBallContractError(
                    "solver sample authority assertion must be pure"
                )
            projected: Dict[int, Tuple[int, int, int]] = {}
            for (
                binding,
                _digest,
                _request,
                batch,
                _digests,
                _sample_index,
                _draw_end,
            ) in staged_refills:
                calls, proposed, admitted = projected.get(
                    binding.action_uid, (0, 0, 0)
                )
                projected[binding.action_uid] = (
                    calls + 1,
                    proposed + batch.proposed_count,
                    admitted + len(batch.receipts),
                )
            for uid, (calls, proposed, admitted) in projected.items():
                current = self._ledger.get(uid, PoolLedger())
                PoolLedger(
                    requests=current.requests,
                    refill_calls=current.refill_calls + calls,
                    proposed=current.proposed + proposed,
                    admitted=current.admitted + admitted,
                    issued=current.issued,
                    discarded=current.discarded,
                )
            for uid in staged_sample_indices_by_uid:
                uid_batches = tuple(
                    batch
                    for (
                        binding,
                        _digest,
                        _request,
                        batch,
                        _digests,
                        _sample_index,
                        _draw_end,
                    ) in staged_refills
                    if binding.action_uid == uid
                )
                self._install_lifecycle_samples(
                    action_uid=uid,
                    batches=uid_batches,
                    authority_sample_index=(
                        projected_sample_highwater[uid][0]
                    ),
                )
            for (
                binding,
                digest,
                refill_request,
                batch,
                new_digests,
                last_sample_index,
                last_sample_draw_end,
            ) in staged_refills:
                self._install_refill_batch(
                    binding=binding,
                    birth_digest=digest,
                    request=refill_request,
                    batch=batch,
                    new_digests=new_digests,
                    last_sample_index=last_sample_index,
                    last_sample_draw_end=last_sample_draw_end,
                )
            for uid, (
                last_sample_index,
                last_sample_draw_end,
            ) in projected_sample_highwater.items():
                if uid in staged_sample_indices_by_uid:
                    self._last_sample_index[uid] = last_sample_index
                    self._last_sample_draw_end[uid] = (
                        last_sample_draw_end
                    )
        except Exception:
            for uid, digest in reversed(registered):
                if digest in self._births.get(uid, {}):
                    self._rollback_empty_birth(uid, digest)
            self._restore_solver_state(solver_state)
            raise

        issued: list[ActionBallTaskReceipt] = []
        for request, _binding, uid, digest in validated:
            receipt = self._pending[uid][digest][0]
            if receipt.swing_generation != request.swing_generation:
                raise ActionBallContractError(
                    "pending task receipt swing generation disagrees with "
                    "pool cursor"
                )
            if (
                self._task_lifecycle[uid][receipt.sample_index]
                != _LIFECYCLE_PENDING
            ):
                raise ActionBallContractError(
                    "pending task receipt lifecycle is not pending"
                )
            issued.append(receipt)
        for (
            request,
            _binding,
            uid,
            digest,
        ), receipt in zip(validated, issued):
            self._pending[uid][digest].pop(0)
            self._task_lifecycle[uid][
                receipt.sample_index
            ] = _LIFECYCLE_ISSUED
            self._issued_task_transcript_sha256[uid][digest] = (
                _task_transcript_extend(
                    self._issued_task_transcript_sha256[uid][digest],
                    receipt.canonical_sha256,
                )
            )
            current = self._ledger[uid]
            self._cursor[uid][digest] += 1
            self._ledger[uid] = PoolLedger(
                requests=current.requests + 1,
                refill_calls=current.refill_calls,
                proposed=current.proposed,
                admitted=current.admitted,
                issued=current.issued + 1,
                discarded=current.discarded,
            )
        return tuple(issued)

    def retire_birth(
        self,
        birth: ActionBirthReceipt,
    ) -> int:
        """Retire one finished episode and count every unused solved task."""

        return self.retire_many((birth,))[0]

    def _retire_many_diagnostic(
        self,
        converted: Tuple[ActionBirthReceipt, ...],
    ) -> Tuple[int, ...]:
        """Retire active diagnostic queues without retaining proof history."""

        validated: list[Tuple[ActionBirthReceipt, int, str, int]] = []
        discarded_by_uid: Dict[int, int] = {}
        for birth in converted:
            binding = self._binding(birth.action_uid)
            birth.assert_contract(
                binding=binding,
                pins=self._pins,
                mobility_mode=self._mobility_mode,
                registry_sha256=self._registry_sha256,
            )
            uid = binding.action_uid
            digest = birth.canonical_sha256
            active = self._births.get(uid, {}).get(digest)
            if active is None or active != birth:
                raise PoolProtocolError(
                    "cannot retire an unknown or already retired birth"
                )
            previous_retired = self._retired_generation.get(
                birth.env_id, 0
            )
            if birth.reset_generation <= previous_retired:
                raise PoolProtocolError(
                    "birth retirement generation is stale"
                )
            discarded = len(self._pending[uid][digest])
            discarded_by_uid[uid] = (
                discarded_by_uid.get(uid, 0) + discarded
            )
            validated.append((birth, uid, digest, discarded))

        projected_ledgers: Dict[int, PoolLedger] = {}
        for uid, discarded in discarded_by_uid.items():
            current = self._ledger[uid]
            projected_ledgers[uid] = PoolLedger(
                requests=current.requests,
                refill_calls=current.refill_calls,
                proposed=current.proposed,
                admitted=current.admitted,
                issued=current.issued,
                discarded=current.discarded + discarded,
            )
        for uid, ledger in projected_ledgers.items():
            self._ledger[uid] = ledger
        for birth, uid, digest, _discarded in validated:
            self._diagnostic_active_sample_sha256.difference_update(
                self._seen_sample_sha256[uid][digest]
            )
            for table in (
                self._births,
                self._pending,
                self._cursor,
                self._refill_index,
                self._seen_sha256,
                self._seen_sample_sha256,
            ):
                del table[uid][digest]
                if not table[uid]:
                    del table[uid]
            self._diagnostic_birth_by_env.pop(birth.env_id, None)
            self._retired_generation[
                birth.env_id
            ] = birth.reset_generation
        return tuple(row[3] for row in validated)

    def retire_many(
        self,
        births: Sequence[ActionBirthReceipt],
    ) -> Tuple[int, ...]:
        """Atomically retire a true-reset env batch.

        Retirement validates the pool-owned active receipt, rather than the
        broker's *latest* consumed generation.  A caller may therefore safely
        retire generation N even if the broker has already consumed N+1,
        without creating an unrecoverable active-birth deadlock.
        """

        if isinstance(births, (str, bytes)) or not isinstance(
            births, Sequence
        ):
            raise ActionBallContractError(
                "retire births must be a non-empty sequence"
            )
        converted = tuple(births)
        if not converted or any(
            not isinstance(birth, ActionBirthReceipt)
            for birth in converted
        ):
            raise ActionBallContractError(
                "retire births must be non-empty ActionBirthReceipt objects"
            )
        env_ids = [birth.env_id for birth in converted]
        digests = [birth.canonical_sha256 for birth in converted]
        if len(set(env_ids)) != len(env_ids) or len(set(digests)) != len(
            digests
        ):
            raise PoolProtocolError(
                "retire batch must not repeat an env or birth"
            )
        if self._diagnostic_fast_path:
            return self._retire_many_diagnostic(converted)

        validated: list[
            Tuple[
                ActionBirthReceipt,
                int,
                str,
                int,
                _RetiredPoolBirth,
            ]
        ] = []
        discarded_by_uid: Dict[int, int] = {}
        for birth in converted:
            binding = self._binding(birth.action_uid)
            birth.assert_contract(
                binding=binding,
                pins=self._pins,
                mobility_mode=self._mobility_mode,
                registry_sha256=self._registry_sha256,
            )
            uid = binding.action_uid
            digest = birth.canonical_sha256
            active = self._births.get(uid, {}).get(digest)
            if active is None or active != birth:
                raise PoolProtocolError(
                    "cannot retire an unknown or already retired birth"
                )
            if digest in self._retired_births.get(uid, {}):
                raise PoolProtocolError(
                    "retired birth transcript already exists"
                )
            previous_retired = self._retired_generation.get(
                birth.env_id, 0
            )
            if birth.reset_generation <= previous_retired:
                raise PoolProtocolError(
                    "birth retirement generation is stale"
                )
            discarded = len(self._pending[uid][digest])
            if any(
                self._task_lifecycle[uid][receipt.sample_index]
                != _LIFECYCLE_PENDING
                for receipt in self._pending[uid][digest]
            ):
                raise ActionBallContractError(
                    "discarded task lifecycle is not pending"
                )
            admitted_count, transcript_root = (
                self._expected_task_transcript_for_active_birth(
                    uid, digest
                )
            )
            retired = _RetiredPoolBirth(
                birth=birth,
                refill_index=self._refill_index[uid][digest],
                proposed_count=self._proposed_by_birth[uid][digest],
                admitted_count=admitted_count,
                issued_count=self._cursor[uid][digest],
                discarded_count=discarded,
                task_transcript_sha256=transcript_root,
                sample_assignments=tuple(
                    self._sample_assignments[uid][digest]
                ),
            )
            discarded_by_uid[uid] = (
                discarded_by_uid.get(uid, 0) + discarded
            )
            validated.append(
                (birth, uid, digest, discarded, retired)
            )

        authority_transcripts = (
            self._solver_task_transcripts_for_births_pure(
                tuple(row[2] for row in validated)
            )
        )
        for row, authority_transcript in zip(
            validated, authority_transcripts
        ):
            retired = row[4]
            if authority_transcript != (
                retired.admitted_count,
                retired.task_transcript_sha256,
            ):
                raise ActionBallContractError(
                    "retired birth task transcript differs from solver "
                    "authority"
                )

        projected_ledgers: Dict[int, PoolLedger] = {}
        for uid, discarded in discarded_by_uid.items():
            current = self._ledger[uid]
            projected_ledgers[uid] = PoolLedger(
                requests=current.requests,
                refill_calls=current.refill_calls,
                proposed=current.proposed,
                admitted=current.admitted,
                issued=current.issued,
                discarded=current.discarded + discarded,
            )

        for uid, ledger in projected_ledgers.items():
            self._ledger[uid] = ledger
        for birth, uid, digest, _discarded, retired in validated:
            for receipt in self._pending[uid][digest]:
                self._task_lifecycle[uid][
                    receipt.sample_index
                ] = _LIFECYCLE_DISCARDED
            self._retired_births.setdefault(uid, {})[
                digest
            ] = retired
            for table in (
                self._births,
                self._pending,
                self._issued_task_transcript_sha256,
                self._cursor,
                self._refill_index,
                self._proposed_by_birth,
                self._sample_assignments,
                self._seen_sha256,
                self._seen_sample_sha256,
            ):
                del table[uid][digest]
                if not table[uid]:
                    del table[uid]
            self._retired_generation[
                birth.env_id
            ] = birth.reset_generation
        return tuple(row[3] for row in validated)

    def state_dict(self) -> Dict[str, object]:
        solver_state = self._solver_state()
        try:
            self._assert_compact_lifecycle_invariants()
            for binding in self._bindings:
                uid = binding.action_uid
                expected = (
                    self._last_sample_index.get(uid, -1),
                    self._last_sample_draw_end.get(uid, 0),
                )
                if self._solver_sample_highwater(uid) != expected:
                    raise ActionBallContractError(
                        "pool sample high-water differs from solver "
                        "authority"
                    )
            if (
                self._solver_emitted_task_counts()
                != self._pool_emitted_task_counts()
            ):
                raise ActionBallContractError(
                    "pool admitted-task counts differ from solver authority"
                )
            pending_receipts = tuple(
                receipt
                for receipts in self._all_task_receipts_by_action().values()
                for receipt in receipts
            )
            self._assert_emitted_tasks_pure(pending_receipts)
            self._assert_proposal_assignments_pure(
                tuple(
                    assignment
                    for action_assignments in (
                        self._sample_assignments.values()
                    )
                    for birth_assignments in (
                        action_assignments.values()
                    )
                    for assignment in birth_assignments
                )
                + tuple(
                    assignment
                    for action_retired in self._retired_births.values()
                    for retired in action_retired.values()
                    for assignment in retired.sample_assignments
                )
            )
            self._assert_all_task_transcripts_pure()
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver checkpoint authorities must be pure"
                )
        except Exception:
            self._restore_solver_state(solver_state)
            raise
        if (
            self._birth_authority is not None
            and self._solver_state_owner_sha256()
            == self._birth_authority.provider_state_owner_sha256
            and solver_state
            != self._birth_authority.provider_state_snapshot()
        ):
            raise ActionBallContractError(
                "shared sampler owner must expose byte-identical provider "
                "and solver states"
            )
        actions = []
        for uid in sorted(self._ledger):
            birth_rows = []
            for birth_digest in sorted(self._births.get(uid, {})):
                pending = self._pending[uid][birth_digest]
                order = [
                    receipt.canonical_sha256 for receipt in pending
                ]
                birth_rows.append(
                    {
                        "birth": self._births[uid][
                            birth_digest
                        ].to_dict(),
                        "cursor": self._cursor[uid][birth_digest],
                        "issued_task_transcript_sha256": (
                            self._issued_task_transcript_sha256[uid][
                                birth_digest
                            ]
                        ),
                        "refill_index": self._refill_index[uid][
                            birth_digest
                        ],
                        "proposed_count": self._proposed_by_birth[uid][
                            birth_digest
                        ],
                        "sample_assignments": _sample_assignment_rows(
                            self._sample_assignments[uid][birth_digest]
                        ),
                        "pending_order": order,
                        "pending_receipts": [
                            receipt.to_dict() for receipt in pending
                        ],
                        "seen_sha256": sorted(
                            self._seen_sha256[uid][birth_digest]
                        ),
                        "seen_sample_sha256": sorted(
                            self._seen_sample_sha256[uid][birth_digest]
                        ),
                    }
                )
            retired_rows = []
            for birth_digest in sorted(
                self._retired_births.get(uid, {})
            ):
                retired = self._retired_births[uid][birth_digest]
                retired_rows.append(
                    {
                        "birth": retired.birth.to_dict(),
                        "refill_index": retired.refill_index,
                        "proposed_count": retired.proposed_count,
                        "admitted_count": retired.admitted_count,
                        "issued_count": retired.issued_count,
                        "discarded_count": retired.discarded_count,
                        "task_transcript_sha256": (
                            retired.task_transcript_sha256
                        ),
                        "sample_assignments": _sample_assignment_rows(
                            retired.sample_assignments
                        ),
                    }
                )
            lifecycle = self._task_lifecycle.get(uid, [])
            actions.append(
                {
                    "action_uid": uid,
                    "ledger": self._ledger[uid].to_dict(),
                    "last_sample_index": self._last_sample_index.get(uid),
                    "last_sample_draw_end": self._last_sample_draw_end.get(
                        uid
                    ),
                    "lifecycle_sample_count": len(lifecycle),
                    "lifecycle_2bit_base64": (
                        _pack_lifecycle_2bit(lifecycle)
                    ),
                    "lifecycle_sha256": _task_lifecycle_sha256(
                        uid, lifecycle
                    ),
                    "births": birth_rows,
                    "retired_births": retired_rows,
                }
            )
        payload: Dict[str, object] = {
            "schema_version": POOL_STATE_SCHEMA_VERSION,
            "runtime_contract_sha256": RUNTIME_CONTRACT_SHA256,
            "registry_sha256": self._registry_sha256,
            "pins": self._pins.to_dict(),
            "mobility_mode": self._mobility_mode,
            "refill_size": self._refill_size,
            "bindings": [
                binding.to_dict() for binding in self._bindings
            ],
            "birth_authority_state_sha256": (
                self._birth_authority_state_sha256()
            ),
            "solver_contract_sha256": self._pins.solver_sha256,
            "solver_state_owner_sha256": (
                self._solver_state_owner_sha256()
            ),
            "solver_state": solver_state,
            "solver_state_sha256": _sha256_json(solver_state),
            "retired_generations": [
                [env, generation]
                for env, generation in sorted(
                    self._retired_generation.items()
                )
            ],
            "actions": actions,
        }
        payload["integrity_sha256"] = _sha256_json(payload)
        return payload

    def load_state_dict(self, state: object) -> None:
        row = _exact_mapping(
            state, self._STATE_KEYS, name="lazy action pool state"
        )
        if row["schema_version"] != POOL_STATE_SCHEMA_VERSION:
            raise ActionBallContractError(
                "unsupported lazy action pool state schema_version"
            )
        if row["runtime_contract_sha256"] != RUNTIME_CONTRACT_SHA256:
            raise ActionBallContractError(
                "lazy action pool runtime contract SHA mismatch"
            )
        declared_integrity = _sha256(
            row["integrity_sha256"], name="integrity_sha256"
        )
        payload = {
            key: row[key]
            for key in self._STATE_KEYS
            if key != "integrity_sha256"
        }
        if _sha256_json(payload) != declared_integrity:
            raise ActionBallContractError(
                "lazy action pool state integrity mismatch"
            )
        pins = RuntimePins.from_dict(row["pins"])
        mode = _mode(row["mobility_mode"])
        refill_size = _plain_int(
            row["refill_size"], name="refill_size", minimum=1
        )
        if not isinstance(row["bindings"], (tuple, list)):
            raise ActionBallContractError("bindings state must be a list")
        bindings = _validate_bindings(
            tuple(
                ActionBinding.from_dict(binding)
                for binding in row["bindings"]
            )
        )
        registry_sha = _sha256(
            row["registry_sha256"], name="registry_sha256"
        )
        if (
            pins != self._pins
            or mode != self._mobility_mode
            or refill_size != self._refill_size
            or bindings != self._bindings
            or registry_sha != self._registry_sha256
            or registry_sha
            != _registry_sha256(bindings, pins, mode)
        ):
            raise ActionBallContractError(
                "lazy action pool state belongs to a different run registry"
            )
        raw_authority_sha = row["birth_authority_state_sha256"]
        if raw_authority_sha is None:
            authority_state_sha = None
        else:
            authority_state_sha = _sha256(
                raw_authority_sha,
                name="birth_authority_state_sha256",
            )
        solver_contract_sha = _sha256(
            row["solver_contract_sha256"],
            name="solver_contract_sha256",
        )
        if solver_contract_sha != self._pins.solver_sha256:
            raise ActionBallContractError(
                "lazy action pool solver contract SHA mismatch"
            )
        solver_owner_sha = _sha256(
            row["solver_state_owner_sha256"],
            name="solver_state_owner_sha256",
        )
        if solver_owner_sha != self._solver_state_owner_sha256():
            raise ActionBallContractError(
                "lazy action pool solver state owner mismatch"
            )
        solver_state = _json_data(
            row["solver_state"], name="solver state"
        )
        if _sha256(
            row["solver_state_sha256"],
            name="solver_state_sha256",
        ) != _sha256_json(solver_state):
            raise ActionBallContractError(
                "lazy action pool solver state SHA mismatch"
            )
        if not isinstance(row["actions"], (tuple, list)):
            raise ActionBallContractError("actions state must be a list")
        retired_generations = ActionBirthBroker._generation_rows(
            row["retired_generations"],
            name="retired_generations",
        )

        births_result: Dict[int, Dict[str, ActionBirthReceipt]] = {}
        pending_result: Dict[
            int, Dict[str, list[ActionBallTaskReceipt]]
        ] = {}
        cursor_result: Dict[int, Dict[str, int]] = {}
        issued_transcript_result: Dict[int, Dict[str, str]] = {}
        refill_result: Dict[int, Dict[str, int]] = {}
        proposed_result: Dict[int, Dict[str, int]] = {}
        sample_assignments_result: Dict[
            int, Dict[str, list[ActionSampleAssignment]]
        ] = {}
        retired_births_result: Dict[
            int, Dict[str, _RetiredPoolBirth]
        ] = {}
        lifecycle_result: Dict[int, list[int]] = {}
        ledger_result: Dict[int, PoolLedger] = {}
        seen_result: Dict[int, Dict[str, set[str]]] = {}
        sample_seen_result: Dict[int, Dict[str, set[str]]] = {}
        last_sample_index_result: Dict[int, int] = {}
        last_sample_draw_end_result: Dict[int, int] = {}
        active_envs: set[int] = set()
        active_generation_by_env: Dict[int, int] = {}
        task_transcript_expectations: Dict[str, Tuple[int, str]] = {}
        task_digests: set[str] = set()
        sample_digests: set[str] = set()
        for index, raw_action in enumerate(row["actions"]):
            action_row = _exact_mapping(
                raw_action,
                (
                    "action_uid",
                    "ledger",
                    "last_sample_index",
                    "last_sample_draw_end",
                    "lifecycle_sample_count",
                    "lifecycle_2bit_base64",
                    "lifecycle_sha256",
                    "births",
                    "retired_births",
                ),
                name=f"actions[{index}]",
            )
            binding = self._binding(action_row["action_uid"])  # type: ignore[arg-type]
            uid = binding.action_uid
            if uid in ledger_result:
                raise ActionBallContractError(
                    f"actions state repeats action_uid {uid}"
                )
            if not isinstance(action_row["births"], (tuple, list)):
                raise ActionBallContractError(
                    f"actions[{index}].births must be a list"
                )
            if not isinstance(
                action_row["retired_births"], (tuple, list)
            ):
                raise ActionBallContractError(
                    f"actions[{index}].retired_births must be a list"
                )
            action_births: Dict[str, ActionBirthReceipt] = {}
            action_pending: Dict[
                str, list[ActionBallTaskReceipt]
            ] = {}
            action_cursor: Dict[str, int] = {}
            action_issued_transcript: Dict[str, str] = {}
            action_refill: Dict[str, int] = {}
            action_proposed: Dict[str, int] = {}
            action_assignments: Dict[
                str, list[ActionSampleAssignment]
            ] = {}
            action_seen: Dict[str, set[str]] = {}
            action_sample_seen: Dict[str, set[str]] = {}
            action_retired: Dict[str, _RetiredPoolBirth] = {}
            active_admitted = 0
            active_issued = 0
            active_refills = 0
            active_proposed = 0
            retired_admitted = 0
            retired_issued = 0
            retired_discarded = 0
            retired_refills = 0
            retired_proposed = 0
            for birth_index, raw_birth in enumerate(
                action_row["births"]
            ):
                birth_row = _exact_mapping(
                    raw_birth,
                    (
                        "birth",
                        "cursor",
                        "issued_task_transcript_sha256",
                        "refill_index",
                        "proposed_count",
                        "sample_assignments",
                        "pending_order",
                        "pending_receipts",
                        "seen_sha256",
                        "seen_sample_sha256",
                    ),
                    name=f"actions[{index}].births[{birth_index}]",
                )
                birth = ActionBirthReceipt.from_dict(birth_row["birth"])
                birth.assert_contract(
                    binding=binding,
                    pins=self._pins,
                    mobility_mode=self._mobility_mode,
                    registry_sha256=self._registry_sha256,
                )
                if (
                    birth.action_uid != uid
                    or birth.action_slot != binding.action_slot
                ):
                    raise ActionBallContractError(
                        "active birth does not match action row"
                    )
                birth_digest = birth.canonical_sha256
                if birth_digest in action_births:
                    raise ActionBallContractError(
                        "action repeats an active birth SHA"
                    )
                if birth.env_id in active_envs:
                    raise ActionBallContractError(
                        "two active births share an env_id"
                    )
                if birth.reset_generation <= retired_generations.get(
                    birth.env_id, 0
                ):
                    raise ActionBallContractError(
                        "active birth is not newer than retired generation"
                    )
                active_envs.add(birth.env_id)
                active_generation_by_env[
                    birth.env_id
                ] = birth.reset_generation
                cursor = _plain_int(
                    birth_row["cursor"],
                    name=(
                        f"actions[{index}].births[{birth_index}].cursor"
                    ),
                )
                issued_transcript = _sha256(
                    birth_row["issued_task_transcript_sha256"],
                    name=(
                        f"actions[{index}].births[{birth_index}]"
                        ".issued_task_transcript_sha256"
                    ),
                )
                refill_index = _plain_int(
                    birth_row["refill_index"],
                    name=(
                        f"actions[{index}].births[{birth_index}]"
                        ".refill_index"
                    ),
                )
                proposed_count = _plain_int(
                    birth_row["proposed_count"],
                    name=(
                        f"actions[{index}].births[{birth_index}]"
                        ".proposed_count"
                    ),
                )
                assignments = _sample_assignments_from_rows(
                    birth_row["sample_assignments"],
                    birth=birth,
                    name=(
                        f"actions[{index}].births[{birth_index}]"
                        ".sample_assignments"
                    ),
                )
                if [
                    assignment.refill_index
                    for assignment in assignments
                ] != list(range(1, refill_index + 1)):
                    raise ActionBallContractError(
                        "active sample assignment refill indices are not "
                        "contiguous"
                    )
                if sum(
                    len(assignment.proposal_sample_indices)
                    for assignment in assignments
                ) != proposed_count:
                    raise ActionBallContractError(
                        "active proposal count differs from sample "
                        "assignments"
                    )
                if not isinstance(
                    birth_row["pending_receipts"], (tuple, list)
                ) or not isinstance(
                    birth_row["pending_order"], (tuple, list)
                ):
                    raise ActionBallContractError(
                        "pending receipts/order must be lists"
                    )
                receipts = [
                    ActionBallTaskReceipt.from_dict(receipt)
                    for receipt in birth_row["pending_receipts"]
                ]
                for receipt in receipts:
                    receipt.assert_contract(
                        binding=binding,
                        pins=self._pins,
                        mobility_mode=self._mobility_mode,
                        registry_sha256=self._registry_sha256,
                    )
                    receipt.assert_birth(birth)
                order = [
                    _sha256(digest, name="pending_order digest")
                    for digest in birth_row["pending_order"]
                ]
                actual_order = [
                    receipt.canonical_sha256 for receipt in receipts
                ]
                if order != actual_order:
                    raise ActionBallContractError(
                        "pending_order does not match pending receipt order"
                    )
                expected_swing_generations = list(
                    range(cursor, cursor + len(receipts))
                )
                if [
                    receipt.swing_generation for receipt in receipts
                ] != expected_swing_generations:
                    raise ActionBallContractError(
                        "pending task swing generations do not match cursor "
                        "order"
                    )
                if not isinstance(
                    birth_row["seen_sha256"], (tuple, list)
                ):
                    raise ActionBallContractError(
                        "seen_sha256 must be a list"
                    )
                seen_list = [
                    _sha256(digest, name="seen_sha256 digest")
                    for digest in birth_row["seen_sha256"]
                ]
                if seen_list != sorted(set(seen_list)):
                    raise ActionBallContractError(
                        "seen_sha256 must be sorted and unique"
                    )
                seen = set(seen_list)
                if task_digests.intersection(seen):
                    raise ActionBallContractError(
                        "task receipt SHA is replayed across active births"
                    )
                task_digests.update(seen)
                if not set(order).issubset(seen):
                    raise ActionBallContractError(
                        "pending receipt is missing from seen_sha256"
                    )
                if len(seen) != cursor + len(receipts):
                    raise ActionBallContractError(
                        "birth cursor/pending/seen invariants disagree"
                    )
                expected_transcript = issued_transcript
                for receipt in receipts:
                    expected_transcript = _task_transcript_extend(
                        expected_transcript,
                        receipt.canonical_sha256,
                    )
                if proposed_count < len(seen):
                    raise ActionBallContractError(
                        "active birth proposed_count is below admitted "
                        "task count"
                    )
                if cursor == 0 and issued_transcript != (
                    task_transcript_sha256(birth_digest, ())
                ):
                    raise ActionBallContractError(
                        "zero-cursor active birth has a non-empty issued "
                        "task transcript"
                    )
                if birth_digest in task_transcript_expectations:
                    raise ActionBallContractError(
                        "pool lifecycle repeats one birth transcript"
                    )
                task_transcript_expectations[birth_digest] = (
                    len(seen),
                    expected_transcript,
                )
                if not isinstance(
                    birth_row["seen_sample_sha256"], (tuple, list)
                ):
                    raise ActionBallContractError(
                        "seen_sample_sha256 must be a list"
                    )
                sample_seen_list = [
                    _sha256(
                        digest,
                        name="seen_sample_sha256 digest",
                    )
                    for digest in birth_row["seen_sample_sha256"]
                ]
                if sample_seen_list != sorted(set(sample_seen_list)):
                    raise ActionBallContractError(
                        "seen_sample_sha256 must be sorted and unique"
                    )
                sample_seen = set(sample_seen_list)
                receipt_sample_digests = {
                    receipt.sample_sha256 for receipt in receipts
                }
                if not receipt_sample_digests.issubset(sample_seen):
                    raise ActionBallContractError(
                        "pending receipt sample is missing from "
                        "seen_sample_sha256"
                    )
                if len(sample_seen) != len(seen):
                    raise ActionBallContractError(
                        "seen task/sample SHA cardinalities disagree"
                    )
                if sample_digests.intersection(sample_seen):
                    raise ActionBallContractError(
                        "ball sample SHA is replayed across active births"
                    )
                sample_digests.update(sample_seen)
                if refill_index == 0 and seen:
                    raise ActionBallContractError(
                        "birth has admitted tasks without a refill"
                    )
                if len(seen) < refill_index * self._refill_size:
                    raise ActionBallContractError(
                        "birth refill index exceeds admitted task history"
                    )
                action_births[birth_digest] = birth
                action_pending[birth_digest] = receipts
                action_cursor[birth_digest] = cursor
                action_issued_transcript[
                    birth_digest
                ] = issued_transcript
                action_refill[birth_digest] = refill_index
                action_proposed[birth_digest] = proposed_count
                action_assignments[birth_digest] = list(assignments)
                action_seen[birth_digest] = seen
                action_sample_seen[birth_digest] = sample_seen
                active_admitted += len(seen)
                active_issued += cursor
                active_refills += refill_index
                active_proposed += proposed_count
            if list(action_births) != sorted(action_births):
                raise ActionBallContractError(
                    "birth rows must be sorted by canonical SHA"
                )
            for retired_index, raw_retired in enumerate(
                action_row["retired_births"]
            ):
                retired_row = _exact_mapping(
                    raw_retired,
                    (
                        "birth",
                        "refill_index",
                        "proposed_count",
                        "admitted_count",
                        "issued_count",
                        "discarded_count",
                        "task_transcript_sha256",
                        "sample_assignments",
                    ),
                    name=(
                        f"actions[{index}].retired_births"
                        f"[{retired_index}]"
                    ),
                )
                retired_birth = ActionBirthReceipt.from_dict(
                    retired_row["birth"]
                )
                retired_birth.assert_contract(
                    binding=binding,
                    pins=self._pins,
                    mobility_mode=self._mobility_mode,
                    registry_sha256=self._registry_sha256,
                )
                if (
                    retired_birth.action_uid != uid
                    or retired_birth.action_slot
                    != binding.action_slot
                ):
                    raise ActionBallContractError(
                        "retired birth does not match action row"
                    )
                retired_digest = retired_birth.canonical_sha256
                if (
                    retired_digest in action_births
                    or retired_digest in action_retired
                    or retired_digest in task_transcript_expectations
                ):
                    raise ActionBallContractError(
                        "pool lifecycle repeats one birth transcript"
                    )
                if (
                    retired_birth.env_id in active_generation_by_env
                    and active_generation_by_env[retired_birth.env_id]
                    <= retired_birth.reset_generation
                ):
                    raise ActionBallContractError(
                        "retired birth is not older than active env birth"
                    )
                if retired_birth.reset_generation > (
                    retired_generations.get(retired_birth.env_id, 0)
                ):
                    raise ActionBallContractError(
                        "retired birth exceeds retired generation ledger"
                    )
                retired = _RetiredPoolBirth(
                    birth=retired_birth,
                    refill_index=retired_row["refill_index"],
                    proposed_count=retired_row["proposed_count"],
                    admitted_count=retired_row["admitted_count"],
                    issued_count=retired_row["issued_count"],
                    discarded_count=retired_row["discarded_count"],
                    task_transcript_sha256=retired_row[
                        "task_transcript_sha256"
                    ],
                    sample_assignments=_sample_assignments_from_rows(
                        retired_row["sample_assignments"],
                        birth=retired_birth,
                        name=(
                            f"actions[{index}].retired_births"
                            f"[{retired_index}].sample_assignments"
                        ),
                    ),
                )
                if (
                    retired.refill_index
                    > retired.admitted_count // self._refill_size
                ):
                    raise ActionBallContractError(
                        "retired refill index exceeds admitted-task "
                        "reachability"
                    )
                action_retired[retired_digest] = retired
                task_transcript_expectations[retired_digest] = (
                    retired.admitted_count,
                    retired.task_transcript_sha256,
                )
                retired_admitted += retired.admitted_count
                retired_issued += retired.issued_count
                retired_discarded += retired.discarded_count
                retired_refills += retired.refill_index
                retired_proposed += retired.proposed_count
            if list(action_retired) != sorted(action_retired):
                raise ActionBallContractError(
                    "retired birth rows must be sorted by canonical SHA"
                )
            ledger = PoolLedger.from_dict(action_row["ledger"])
            if (
                ledger.refill_calls
                > ledger.admitted // self._refill_size
            ):
                raise ActionBallContractError(
                    "pool ledger refill calls exceed admitted-task "
                    "reachability"
                )
            if action_row["last_sample_index"] is None:
                last_sample_index = None
            else:
                last_sample_index = _plain_int(
                    action_row["last_sample_index"],
                    name=f"actions[{index}].last_sample_index",
                )
            if action_row["last_sample_draw_end"] is None:
                last_sample_draw_end = None
            else:
                last_sample_draw_end = _plain_int(
                    action_row["last_sample_draw_end"],
                    name=f"actions[{index}].last_sample_draw_end",
                    minimum=1,
                )
            if (last_sample_index is None) != (
                last_sample_draw_end is None
            ):
                raise ActionBallContractError(
                    "sample index/draw high-water must both be null or set"
                )
            if ledger.admitted > 0 and last_sample_index is None:
                raise ActionBallContractError(
                    "admitted pool ledger requires sample high-water"
                )
            if ledger.admitted == 0 and last_sample_index is not None:
                raise ActionBallContractError(
                    "empty pool ledger cannot have sample high-water"
                )
            active_receipts = [
                receipt
                for receipts in action_pending.values()
                for receipt in receipts
            ]
            if active_receipts and (
                max(receipt.sample_index for receipt in active_receipts)
                > last_sample_index  # type: ignore[operator]
                or max(
                    receipt.sample_draw_end for receipt in active_receipts
                )
                > last_sample_draw_end  # type: ignore[operator]
            ):
                raise ActionBallContractError(
                    "active sample exceeds action sample high-water"
                )
            expected_ledger = PoolLedger(
                requests=active_issued + retired_issued,
                refill_calls=active_refills + retired_refills,
                proposed=active_proposed + retired_proposed,
                admitted=active_admitted + retired_admitted,
                issued=active_issued + retired_issued,
                discarded=retired_discarded,
            )
            if ledger != expected_ledger:
                raise ActionBallContractError(
                    "pool action ledger differs from compact per-birth "
                    "lifecycle"
                )
            lifecycle_count = _plain_int(
                action_row["lifecycle_sample_count"],
                name=f"actions[{index}].lifecycle_sample_count",
            )
            if lifecycle_count != ledger.proposed or (
                last_sample_index is None
                and lifecycle_count != 0
            ) or (
                last_sample_index is not None
                and lifecycle_count != last_sample_index + 1
            ):
                raise ActionBallContractError(
                    "task lifecycle must cover every proposal sample index"
                )
            lifecycle = _unpack_lifecycle_2bit(
                action_row["lifecycle_2bit_base64"],
                count=lifecycle_count,
            )
            if _sha256(
                action_row["lifecycle_sha256"],
                name=f"actions[{index}].lifecycle_sha256",
            ) != _task_lifecycle_sha256(uid, lifecycle):
                raise ActionBallContractError(
                    "task lifecycle SHA mismatch"
                )
            status_counts = {
                status: lifecycle.count(status)
                for status in range(4)
            }
            pending_count = sum(
                len(receipts)
                for receipts in action_pending.values()
            )
            if (
                status_counts[_LIFECYCLE_REJECTED]
                != ledger.proposed - ledger.admitted
                or status_counts[_LIFECYCLE_PENDING] != pending_count
                or status_counts[_LIFECYCLE_ISSUED] != ledger.issued
                or status_counts[_LIFECYCLE_DISCARDED]
                != ledger.discarded
            ):
                raise ActionBallContractError(
                    "2-bit task lifecycle statuses disagree with ledger"
                )
            for receipt in active_receipts:
                if (
                    lifecycle[receipt.sample_index]
                    != _LIFECYCLE_PENDING
                ):
                    raise ActionBallContractError(
                        "pending receipt lifecycle status is not pending"
                    )
            self._validate_assignment_partition_for_action(
                action_uid=uid,
                lifecycle=lifecycle,
                births=action_births,
                pending=action_pending,
                cursor=action_cursor,
                refill_index=action_refill,
                proposed_by_birth=action_proposed,
                sample_assignments=action_assignments,
                retired_births=action_retired,
            )
            if action_births:
                births_result[uid] = action_births
                pending_result[uid] = action_pending
                cursor_result[uid] = action_cursor
                issued_transcript_result[
                    uid
                ] = action_issued_transcript
                refill_result[uid] = action_refill
                proposed_result[uid] = action_proposed
                sample_assignments_result[uid] = action_assignments
                seen_result[uid] = action_seen
                sample_seen_result[uid] = action_sample_seen
            if action_retired:
                retired_births_result[uid] = action_retired
            lifecycle_result[uid] = lifecycle
            ledger_result[uid] = ledger
            if last_sample_index is not None:
                last_sample_index_result[uid] = last_sample_index
                last_sample_draw_end_result[uid] = last_sample_draw_end  # type: ignore[assignment]
        if list(ledger_result) != sorted(ledger_result):
            raise ActionBallContractError(
                "actions state must be sorted by action_uid"
            )
        expected_retired_generations: Dict[int, int] = {}
        for action_retired in retired_births_result.values():
            for retired in action_retired.values():
                expected_retired_generations[retired.birth.env_id] = max(
                    expected_retired_generations.get(
                        retired.birth.env_id, 0
                    ),
                    retired.birth.reset_generation,
                )
        if retired_generations != expected_retired_generations:
            raise ActionBallContractError(
                "retired generation ledger differs from compact retired "
                "birth records"
            )
        if authority_state_sha is None:
            if births_result or retired_births_result:
                raise ActionBallContractError(
                    "non-pristine pool state must pin a birth authority state"
                )
        else:
            if self._birth_authority is None:
                raise PoolProtocolError(
                    "load the exact birth broker state and bind it before "
                    "loading a non-pristine task pool"
                )
            if (
                self._birth_authority_state_sha256()
                != authority_state_sha
            ):
                raise ActionBallContractError(
                    "task pool checkpoint birth authority state mismatch"
                )
            for action_births in births_result.values():
                for birth in action_births.values():
                    self._birth_authority.assert_consumed_birth(birth)
            for action_retired in retired_births_result.values():
                for retired in action_retired.values():
                    self._birth_authority.assert_consumed_birth(
                        retired.birth
                    )
            if (
                solver_owner_sha
                == self._birth_authority.provider_state_owner_sha256
                and solver_state
                != self._birth_authority.provider_state_snapshot()
            ):
                raise ActionBallContractError(
                    "shared sampler owner checkpoint states disagree "
                    "between broker and pool"
                )
        previous_solver_state = self._solver_state()
        try:
            self._restore_solver_state(solver_state)
            if self._solver is None:
                raise PoolProtocolError("task solver is not bound")
            pending_task_receipts: list[
                ActionBallTaskReceipt
            ] = []
            for action_pending in pending_result.values():
                for receipts in action_pending.values():
                    for receipt in receipts:
                        self._solver.assert_emitted_sample(receipt)
                        pending_task_receipts.append(receipt)
            self._solver.assert_emitted_tasks(
                tuple(pending_task_receipts)
            )
            self._solver.assert_proposal_assignments(
                tuple(
                    assignment
                    for action_assignments in (
                        sample_assignments_result.values()
                    )
                    for birth_assignments in (
                        action_assignments.values()
                    )
                    for assignment in birth_assignments
                )
                + tuple(
                    assignment
                    for action_retired in retired_births_result.values()
                    for retired in action_retired.values()
                    for assignment in retired.sample_assignments
                )
            )
            for binding in self._bindings:
                uid = binding.action_uid
                expected_count = ledger_result.get(
                    uid, PoolLedger()
                ).admitted
                if (
                    self._solver_emitted_task_count(uid)
                    != expected_count
                ):
                    raise ActionBallContractError(
                        "pool admitted-task count differs from solver "
                        "authority"
                    )
            if not self._solver_delegates_birth_task_transcripts():
                for birth_digest, expectation in (
                    task_transcript_expectations.items()
                ):
                    if (
                        self._solver_task_transcript_for_birth(
                            birth_digest
                        )
                        != expectation
                    ):
                        raise ActionBallContractError(
                            "pool birth task transcript differs from solver "
                            "authority"
                        )
            for binding in self._bindings:
                uid = binding.action_uid
                saved_highwater = (
                    last_sample_index_result.get(uid, -1),
                    last_sample_draw_end_result.get(uid, 0),
                )
                if (
                    self._solver_sample_highwater(uid)
                    != saved_highwater
                ):
                    raise ActionBallContractError(
                        "pool sample high-water differs from solver "
                        "authority"
                    )
            if self._solver_state() != solver_state:
                raise ActionBallContractError(
                    "solver sample authority assertion must be pure"
                )
        except Exception:
            self._restore_solver_state(previous_solver_state)
            raise
        # Atomic commit.  The bound runtime solver deliberately stays bound.
        self._births = births_result
        self._pending = pending_result
        self._cursor = cursor_result
        self._issued_task_transcript_sha256 = (
            issued_transcript_result
        )
        self._refill_index = refill_result
        self._proposed_by_birth = proposed_result
        self._sample_assignments = sample_assignments_result
        self._retired_births = retired_births_result
        self._task_lifecycle = lifecycle_result
        self._ledger = ledger_result
        self._seen_sha256 = seen_result
        self._seen_sample_sha256 = sample_seen_result
        self._last_sample_index = last_sample_index_result
        self._last_sample_draw_end = last_sample_draw_end_result
        self._retired_generation = retired_generations


__all__ = [
    "ActionBallContractError",
    "CounterRallyTaskIdentityError",
    "BirthProtocolError",
    "PoolProtocolError",
    "RuntimePins",
    "ActionBinding",
    "ActionDomainLevels",
    "ActionDomainClaim",
    "ActionDomainClaimAuthority",
    "ActionBirthProvider",
    "ActionBirthRequest",
    "ActionBirthReceipt",
    "ActionBallTaskReceipt",
    "CounterRallyTaskIdentity",
    "ActionTaskReceiptRef",
    "ActionTeacherTiming",
    "ActionTeacherTimingWithBasePreparation",
    "BasePreparationContract",
    "BasePreparationReceipt",
    "BasePreparationAdmissionError",
    "derive_action_teacher_timing",
    "derive_action_teacher_site_timing",
    "derive_action_teacher_timing_with_base_preparation",
    "BirthReserveRequest",
    "BirthCommitRequest",
    "BirthConsumeRequest",
    "ActionBirthBroker",
    "ActionPoolRefillRequest",
    "ActionPoolRefillBatch",
    "ActionSampleAssignment",
    "ActionTaskIssueRequest",
    "PoolLedger",
    "ActionTaskSolver",
    "LazyActionTaskPool",
    "TASK_TRANSCRIPT_SCHEMA_VERSION",
    "extend_task_transcript_sha256",
    "task_transcript_sha256",
    "ARM_KEYS",
    "ARM_CATALOG_SHA256",
    "SAMPLER_SCHEMA_VERSION",
    "SAMPLER_BIRTH_DRAW_COUNT",
    "SAMPLER_SAMPLE_DRAW_COUNT",
    "TASK_RECEIPT_SCHEMA_VERSION",
    "TASK_RECEIPT_TIMING_AUTHORITY",
    "COUNTER_RALLY_TASK_IDENTITY_SCHEMA_VERSION",
    "MAX_PRE_SWING_WAIT_S",
    "BASE_PREPARATION_SCHEMA_VERSION",
    "BASE_PREPARATION_REJECT_REASON",
    "RUNTIME_CONTRACT_SHA256",
]
