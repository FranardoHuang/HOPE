"""Deterministic action-conditioned incoming-ball and base sampling.

This module intentionally has no Isaac, NumPy, Torch, or manifest dependency.
It implements the data-producing part of the training order

``action -> ball/base sample -> downstream ball-to-task solver``.

Every swing sample consumes exactly :data:`DRAWS_PER_SAMPLE` counter-based
random draws and every episode-birth reservation consumes exactly
:data:`DRAWS_PER_BIRTH`.  In particular, ``no_move`` still samples (and
records) latent base spawn/travel before forcing ``base_goal == base_start``.
Exact resume is therefore a small, strict per-action state instead of an opaque
framework RNG blob.

Coordinate contract
-------------------

* ``W`` is the environment-local world frame.
* ``B_yaw`` is the sampled base-yaw frame (roll and pitch are zero).
* episode-birth base spawn is sampled in ``W`` exactly once;
* base travel, contact offset, incoming direction, and spin direction are
  sampled in ``B_yaw``;
* the returned receipt contains both relative quantities and their ``W``
  realization.

The contact position is always derived from the *goal* base pose.  A downstream
solver must fill the task; this module never guesses or authorizes a task.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from statistics import NormalDist
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

SCHEMA_VERSION = 3
STATE_SCHEMA_VERSION = 5
DRAWS_PER_BIRTH = 3
DRAWS_PER_SAMPLE = 18
INT64_MAX = (1 << 63) - 1
UNIT_VECTOR_TOLERANCE = 1.0e-6
_TWO_POW_53 = float(1 << 53)
_SMALLEST_POSITIVE_FLOAT = float.fromhex("0x0.0000000000001p-1022")
_LARGEST_FLOAT_BELOW_ONE = float.fromhex("0x1.fffffffffffffp-1")
_NORMAL = NormalDist()

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
_LEVEL_NAMES = ARM_KEYS
_STATE_KEYS = (
    "schema_version",
    "sampler_contract_sha256",
    "arm_catalog_sha256",
    "seed",
    "draws_per_birth",
    "draws_per_sample",
    "action_uids",
    "per_action",
    "issued_births",
    "issued_sample_birth_indices",
    "compaction_segments",
    "compaction_head_sha256",
    "total_birth_count",
    "total_sample_count",
    "total_draw_count",
    "integrity_sha256",
)
_PER_ACTION_STATE_KEYS = (
    "birth_count",
    "sample_count",
    "draw_count",
    "retired_birth_count",
    "retired_sample_count",
    "retired_birth_highwater_draw_end",
    "retired_sample_highwater_draw_end",
    "retired_assignment_head_sha256",
    "assignment_head_sha256",
    "compaction_segment_count",
    "compaction_segment_head_sha256",
)
_BIRTH_STATE_KEYS = (
    "birth_id",
    "sampler_contract_sha256",
    "arm_catalog_sha256",
    "action_uid",
    "domain_epoch",
    "domain_levels",
    "profile_sha256",
    "levels_sha256",
    "birth_index",
    "draw_start",
    "draw_end",
    "mobility_mode",
    "base_yaw_rad",
    "base_start_w_m",
)
_SAMPLE_IDENTITY_KEYS = (
    "schema_version",
    "kind",
    "sampler_contract_sha256",
    "arm_catalog_sha256",
    "sample_index",
    "action_uid",
    "domain_epoch",
    "domain_levels",
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
_COMPACTION_SEGMENT_KEYS = (
    "action_uid",
    "sampler_contract_sha256",
    "segment_index",
    "retired_birth_count",
    "retired_sample_count",
    "retained_birth_start_inclusive",
    "retained_sample_start_inclusive",
    "issued_birth_count",
    "issued_sample_count",
    "issued_draw_count",
    "retired_birth_highwater_draw_end",
    "retired_sample_highwater_draw_end",
    "prior_segment_head_sha256",
    "birth_transcript_sha256",
    "sample_assignment_transcript_sha256",
    "retired_assignment_head_sha256",
    "assignment_head_sha256",
    "segment_sha256",
    "segment_head_sha256",
)


def _plain_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = INT64_MAX,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a plain integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _finite(
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


def _vec3(
    value: object,
    *,
    name: str,
    minimum: Optional[float] = None,
) -> Vec3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 tuple/list")
    return tuple(
        _finite(component, name=f"{name}[{index}]", minimum=minimum)
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _vec2(
    value: object,
    *,
    name: str,
    minimum: Optional[float] = None,
) -> Vec2:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 tuple/list")
    return tuple(
        _finite(component, name=f"{name}[{index}]", minimum=minimum)
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _validate_interval(
    lower: Vec3,
    center: Vec3,
    upper: Vec3,
    *,
    name: str,
) -> None:
    for index, (lo, middle, hi) in enumerate(zip(lower, center, upper)):
        if hi < lo:
            raise ValueError(f"{name}[{index}] has upper below lower")
        if not lo <= middle <= hi:
            raise ValueError(
                f"{name}[{index}] center must lie inside its bounds"
            )


def _validate_interval2(
    lower: Vec2,
    center: Vec2,
    upper: Vec2,
    *,
    name: str,
) -> None:
    for index, (lo, middle, hi) in enumerate(zip(lower, center, upper)):
        if hi < lo:
            raise ValueError(f"{name}[{index}] has upper below lower")
        if not lo <= middle <= hi:
            raise ValueError(
                f"{name}[{index}] center must lie inside its bounds"
            )


def _validate_std_pair(
    initial: Vec3,
    maximum: Vec3,
    *,
    name: str,
) -> None:
    for index, (lo, hi) in enumerate(zip(initial, maximum)):
        if lo < 0.0 or hi < 0.0:
            raise ValueError(f"{name}[{index}] std must be non-negative")
        if lo > hi:
            raise ValueError(
                f"{name}[{index}] initial std must not exceed maximum std"
            )


def _validate_std_pair2(
    initial: Vec2,
    maximum: Vec2,
    *,
    name: str,
) -> None:
    for index, (lo, hi) in enumerate(zip(initial, maximum)):
        if lo < 0.0 or hi < 0.0:
            raise ValueError(f"{name}[{index}] std must be non-negative")
        if lo > hi:
            raise ValueError(
                f"{name}[{index}] initial std must not exceed maximum std"
            )


def _strict_unit(value: Vec3, *, name: str) -> Vec3:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be a non-zero finite direction")
    if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
        raise ValueError(
            f"{name} must already be unit length within "
            f"{UNIT_VECTOR_TOLERANCE}; got norm {norm}"
        )
    # Preserve the declared bytes/floats.  Profile validation must not silently
    # alter an action's physical direction.
    return value


def _normalize_generated(value: Vec3, *, name: str) -> Vec3:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be a non-zero finite direction")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


ARM_CATALOG_SHA256 = _sha256_json(
    {
        "schema_version": SCHEMA_VERSION,
        "arm_keys": list(ARM_KEYS),
    }
)


def _sha256_hex(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hex")
    return value


def _exact_mapping(
    value: object,
    expected: Sequence[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise ValueError(
            f"{name} has invalid keys "
            f"(missing={sorted(expected_set - actual_set)}, "
            f"unknown={sorted(actual_set - expected_set)})"
        )
    return value


def _highwater_pair(value: object, *, name: str) -> Tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an (index, draw_end) pair")
    index = _plain_int(
        value[0],
        name=f"{name}[0]",
        minimum=-1,
    )
    draw_end = _plain_int(value[1], name=f"{name}[1]")
    if (index == -1) != (draw_end == 0):
        raise ValueError(
            f"{name} must use exactly (-1, 0) for an empty high-water"
        )
    return (index, draw_end)


def _lerp(initial: float, maximum: float, level: float) -> float:
    return initial + (maximum - initial) * level


def _vec_lerp(initial: Vec3, maximum: Vec3, level: float) -> Vec3:
    return tuple(_lerp(a, b, level) for a, b in zip(initial, maximum))  # type: ignore[return-value]


def _vec2_lerp(initial: Vec2, maximum: Vec2, level: float) -> Vec2:
    return tuple(_lerp(a, b, level) for a, b in zip(initial, maximum))  # type: ignore[return-value]


def _vec3_lerp_levels(
    initial: Vec3,
    maximum: Vec3,
    levels: Vec3,
) -> Vec3:
    return tuple(
        _lerp(a, b, level)
        for a, b, level in zip(initial, maximum, levels)
    )  # type: ignore[return-value]


def _vec2_lerp_levels(
    initial: Vec2,
    maximum: Vec2,
    levels: Vec2,
) -> Vec2:
    return tuple(
        _lerp(a, b, level)
        for a, b, level in zip(initial, maximum, levels)
    )  # type: ignore[return-value]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(value: Vec3, scalar: float) -> Vec3:
    return tuple(component * scalar for component in value)  # type: ignore[return-value]


def _rotate_yaw(value: Vec3, yaw_rad: float) -> Vec3:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    x, y, z = value
    return (
        cosine * x - sine * y,
        sine * x + cosine * y,
        z,
    )


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def _validate_tangent_frame(
    center: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    *,
    name: str,
) -> None:
    if abs(_dot(center, tangent_u)) > UNIT_VECTOR_TOLERANCE:
        raise ValueError(f"{name}.tangent_u must be orthogonal to center")
    if abs(_dot(center, tangent_v)) > UNIT_VECTOR_TOLERANCE:
        raise ValueError(f"{name}.tangent_v must be orthogonal to center")
    if abs(_dot(tangent_u, tangent_v)) > UNIT_VECTOR_TOLERANCE:
        raise ValueError(
            f"{name}.tangent_u and tangent_v must be orthogonal"
        )
    handed = _cross(tangent_u, tangent_v)
    if any(
        abs(actual - expected) > UNIT_VECTOR_TOLERANCE
        for actual, expected in zip(handed, center)
    ):
        raise ValueError(
            f"{name} tangent frame must be right-handed"
        )


def _validate_side_std(
    *,
    center: float,
    lower_bound: float,
    upper_bound: float,
    lower_initial: float,
    lower_maximum: float,
    upper_initial: float,
    upper_maximum: float,
    name: str,
) -> None:
    if not lower_bound <= center <= upper_bound:
        raise ValueError(f"{name} center must lie inside bounds")
    for side_name, initial, maximum in (
        ("lower", lower_initial, lower_maximum),
        ("upper", upper_initial, upper_maximum),
    ):
        if initial < 0.0 or maximum < 0.0:
            raise ValueError(f"{name} {side_name} std must be non-negative")
        if initial > maximum:
            raise ValueError(
                f"{name} {side_name} initial std must not exceed maximum"
            )
    if center == lower_bound and lower_maximum != 0.0:
        raise ValueError(
            f"{name} lower std must be zero at the lower hard bound"
        )
    if center == upper_bound and upper_maximum != 0.0:
        raise ValueError(
            f"{name} upper std must be zero at the upper hard bound"
        )
    if lower_maximum > center - lower_bound + 1.0e-12:
        raise ValueError(
            f"{name} lower max exceeds center-to-min support"
        )
    if upper_maximum > upper_bound - center + 1.0e-12:
        raise ValueError(
            f"{name} upper max exceeds center-to-max support"
        )


def _validate_width_pair(
    initial: float,
    maximum: float,
    *,
    name: str,
) -> None:
    if initial < 0.0 or maximum < 0.0:
        raise ValueError(f"{name} widths must be non-negative")
    if initial > maximum:
        raise ValueError(f"{name} initial width must not exceed maximum")
    if maximum > 180.0:
        raise ValueError(f"{name} maximum width must be <= 180 degrees")


@dataclass(frozen=True)
class DomainLevels:
    """Exact normalized widths for the 32 independently promoted arms."""

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
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _finite(
                    getattr(self, field.name),
                    name=f"levels.{field.name}",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

    @classmethod
    def from_mapping(cls, value: object) -> "DomainLevels":
        row = _exact_mapping(value, _LEVEL_NAMES, name="domain levels")
        return cls(**{name: row[name] for name in _LEVEL_NAMES})

    def as_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in ARM_KEYS}

    @property
    def sha256(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True)
class SamplingProfile:
    """Strict internal profile; a manifest adapter may construct this later."""

    action_uid: int

    contact_offset_center_b_yaw_m: Vec3
    contact_offset_std_lower_initial_m: Vec3
    contact_offset_std_lower_max_m: Vec3
    contact_offset_std_upper_initial_m: Vec3
    contact_offset_std_upper_max_m: Vec3
    contact_offset_min_b_yaw_m: Vec3
    contact_offset_max_b_yaw_m: Vec3

    time_to_contact_center_s: float
    time_to_contact_std_lower_initial_s: float
    time_to_contact_std_lower_max_s: float
    time_to_contact_std_upper_initial_s: float
    time_to_contact_std_upper_max_s: float
    time_to_contact_min_s: float
    time_to_contact_max_s: float

    incoming_direction_center_b_yaw: Vec3
    incoming_direction_tangent_u_b_yaw: Vec3
    incoming_direction_tangent_v_b_yaw: Vec3
    incoming_direction_tangent_u_neg_initial_deg: float
    incoming_direction_tangent_u_neg_max_deg: float
    incoming_direction_tangent_u_pos_initial_deg: float
    incoming_direction_tangent_u_pos_max_deg: float
    incoming_direction_tangent_v_neg_initial_deg: float
    incoming_direction_tangent_v_neg_max_deg: float
    incoming_direction_tangent_v_pos_initial_deg: float
    incoming_direction_tangent_v_pos_max_deg: float
    incoming_inbound_axis_b_yaw: Vec3
    incoming_inbound_min_cosine: float
    incoming_speed_center_mps: float
    incoming_speed_std_lower_initial_mps: float
    incoming_speed_std_lower_max_mps: float
    incoming_speed_std_upper_initial_mps: float
    incoming_speed_std_upper_max_mps: float
    incoming_speed_min_mps: float
    incoming_speed_max_mps: float

    spin_direction_center_b_yaw: Vec3
    spin_direction_tangent_u_b_yaw: Vec3
    spin_direction_tangent_v_b_yaw: Vec3
    spin_direction_tangent_u_neg_initial_deg: float
    spin_direction_tangent_u_neg_max_deg: float
    spin_direction_tangent_u_pos_initial_deg: float
    spin_direction_tangent_u_pos_max_deg: float
    spin_direction_tangent_v_neg_initial_deg: float
    spin_direction_tangent_v_neg_max_deg: float
    spin_direction_tangent_v_pos_initial_deg: float
    spin_direction_tangent_v_pos_max_deg: float
    spin_magnitude_center_radps: float
    spin_magnitude_std_lower_initial_radps: float
    spin_magnitude_std_lower_max_radps: float
    spin_magnitude_std_upper_initial_radps: float
    spin_magnitude_std_upper_max_radps: float
    spin_magnitude_min_radps: float
    spin_magnitude_max_radps: float

    base_spawn_center_w_m: Vec3
    base_spawn_std_lower_initial_m: Vec3
    base_spawn_std_lower_max_m: Vec3
    base_spawn_std_upper_initial_m: Vec3
    base_spawn_std_upper_max_m: Vec3
    base_spawn_min_w_m: Vec3
    base_spawn_max_w_m: Vec3

    base_travel_center_b_yaw_m: Vec3
    base_travel_std_lower_initial_m: Vec3
    base_travel_std_lower_max_m: Vec3
    base_travel_std_upper_initial_m: Vec3
    base_travel_std_upper_max_m: Vec3
    base_travel_min_b_yaw_m: Vec3
    base_travel_max_b_yaw_m: Vec3

    landing_aim_center_w_xy_m: Vec2
    landing_aim_std_lower_initial_m: Vec2
    landing_aim_std_lower_max_m: Vec2
    landing_aim_std_upper_initial_m: Vec2
    landing_aim_std_upper_max_m: Vec2
    landing_aim_min_w_xy_m: Vec2
    landing_aim_max_w_xy_m: Vec2

    reference_t_hit_s: float
    reference_t_cycle_s: float
    reference_racket_site_speed_mps: float
    reaction_margin_s: float
    teacher_rate_min: float
    teacher_rate_max: float

    mobility_mode: str = "no_move"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_uid",
            _plain_int(self.action_uid, name="action_uid", minimum=1),
        )
        vector_names = (
            "contact_offset_center_b_yaw_m",
            "contact_offset_min_b_yaw_m",
            "contact_offset_max_b_yaw_m",
            "incoming_direction_center_b_yaw",
            "incoming_direction_tangent_u_b_yaw",
            "incoming_direction_tangent_v_b_yaw",
            "incoming_inbound_axis_b_yaw",
            "spin_direction_center_b_yaw",
            "spin_direction_tangent_u_b_yaw",
            "spin_direction_tangent_v_b_yaw",
            "base_spawn_center_w_m",
            "base_spawn_min_w_m",
            "base_spawn_max_w_m",
            "base_travel_center_b_yaw_m",
            "base_travel_min_b_yaw_m",
            "base_travel_max_b_yaw_m",
        )
        vector_std_names = (
            "contact_offset_std_lower_initial_m",
            "contact_offset_std_lower_max_m",
            "contact_offset_std_upper_initial_m",
            "contact_offset_std_upper_max_m",
            "base_spawn_std_lower_initial_m",
            "base_spawn_std_lower_max_m",
            "base_spawn_std_upper_initial_m",
            "base_spawn_std_upper_max_m",
            "base_travel_std_lower_initial_m",
            "base_travel_std_lower_max_m",
            "base_travel_std_upper_initial_m",
            "base_travel_std_upper_max_m",
        )
        for name in vector_names:
            object.__setattr__(
                self, name, _vec3(getattr(self, name), name=name)
            )
        for name in vector_std_names:
            object.__setattr__(
                self,
                name,
                _vec3(getattr(self, name), name=name, minimum=0.0),
            )

        aim_vector_names = (
            "landing_aim_center_w_xy_m",
            "landing_aim_min_w_xy_m",
            "landing_aim_max_w_xy_m",
        )
        aim_std_names = (
            "landing_aim_std_lower_initial_m",
            "landing_aim_std_lower_max_m",
            "landing_aim_std_upper_initial_m",
            "landing_aim_std_upper_max_m",
        )
        for name in aim_vector_names:
            object.__setattr__(
                self, name, _vec2(getattr(self, name), name=name)
            )
        for name in aim_std_names:
            object.__setattr__(
                self,
                name,
                _vec2(getattr(self, name), name=name, minimum=0.0),
            )

        unit_names = (
            "incoming_direction_center_b_yaw",
            "incoming_direction_tangent_u_b_yaw",
            "incoming_direction_tangent_v_b_yaw",
            "incoming_inbound_axis_b_yaw",
            "spin_direction_center_b_yaw",
            "spin_direction_tangent_u_b_yaw",
            "spin_direction_tangent_v_b_yaw",
        )
        for name in unit_names:
            object.__setattr__(
                self,
                name,
                _strict_unit(getattr(self, name), name=name),
            )
        _validate_tangent_frame(
            self.incoming_direction_center_b_yaw,
            self.incoming_direction_tangent_u_b_yaw,
            self.incoming_direction_tangent_v_b_yaw,
            name="incoming_direction",
        )
        _validate_tangent_frame(
            self.spin_direction_center_b_yaw,
            self.spin_direction_tangent_u_b_yaw,
            self.spin_direction_tangent_v_b_yaw,
            name="spin_direction",
        )

        scalar_names = (
            "time_to_contact_center_s",
            "time_to_contact_std_lower_initial_s",
            "time_to_contact_std_lower_max_s",
            "time_to_contact_std_upper_initial_s",
            "time_to_contact_std_upper_max_s",
            "time_to_contact_min_s",
            "time_to_contact_max_s",
            "incoming_speed_center_mps",
            "incoming_speed_std_lower_initial_mps",
            "incoming_speed_std_lower_max_mps",
            "incoming_speed_std_upper_initial_mps",
            "incoming_speed_std_upper_max_mps",
            "incoming_speed_min_mps",
            "incoming_speed_max_mps",
            "spin_magnitude_center_radps",
            "spin_magnitude_std_lower_initial_radps",
            "spin_magnitude_std_lower_max_radps",
            "spin_magnitude_std_upper_initial_radps",
            "spin_magnitude_std_upper_max_radps",
            "spin_magnitude_min_radps",
            "spin_magnitude_max_radps",
            "reference_t_hit_s",
            "reference_t_cycle_s",
            "reference_racket_site_speed_mps",
            "reaction_margin_s",
            "teacher_rate_min",
            "teacher_rate_max",
        )
        for name in scalar_names:
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name, minimum=0.0),
            )
        object.__setattr__(
            self,
            "incoming_inbound_min_cosine",
            _finite(
                self.incoming_inbound_min_cosine,
                name="incoming_inbound_min_cosine",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.incoming_inbound_min_cosine >= 1.0:
            raise ValueError("incoming_inbound_min_cosine must be < 1")

        width_names = (
            "incoming_direction_tangent_u_neg",
            "incoming_direction_tangent_u_pos",
            "incoming_direction_tangent_v_neg",
            "incoming_direction_tangent_v_pos",
            "spin_direction_tangent_u_neg",
            "spin_direction_tangent_u_pos",
            "spin_direction_tangent_v_neg",
            "spin_direction_tangent_v_pos",
        )
        for prefix in width_names:
            initial_name = f"{prefix}_initial_deg"
            maximum_name = f"{prefix}_max_deg"
            initial = _finite(
                getattr(self, initial_name),
                name=initial_name,
                minimum=0.0,
                maximum=180.0,
            )
            maximum = _finite(
                getattr(self, maximum_name),
                name=maximum_name,
                minimum=0.0,
                maximum=180.0,
            )
            object.__setattr__(self, initial_name, initial)
            object.__setattr__(self, maximum_name, maximum)
            _validate_width_pair(
                initial, maximum, name=prefix
            )
        for direction_name in ("incoming_direction", "spin_direction"):
            radial_max_deg = math.hypot(
                max(
                    getattr(
                        self,
                        f"{direction_name}_tangent_u_neg_max_deg",
                    ),
                    getattr(
                        self,
                        f"{direction_name}_tangent_u_pos_max_deg",
                    ),
                ),
                max(
                    getattr(
                        self,
                        f"{direction_name}_tangent_v_neg_max_deg",
                    ),
                    getattr(
                        self,
                        f"{direction_name}_tangent_v_pos_max_deg",
                    ),
                ),
            )
            if radial_max_deg > 180.0:
                raise ValueError(
                    f"{direction_name} maximum tangent envelope "
                    "must be <= 180 degrees"
                )
        center_to_inbound_axis_deg = math.degrees(
            math.acos(
                max(
                    -1.0,
                    min(
                        1.0,
                        _dot(
                            self.incoming_direction_center_b_yaw,
                            self.incoming_inbound_axis_b_yaw,
                        ),
                    ),
                )
            )
        )
        incoming_radius_max_deg = math.hypot(
            max(
                self.incoming_direction_tangent_u_neg_max_deg,
                self.incoming_direction_tangent_u_pos_max_deg,
            ),
            max(
                self.incoming_direction_tangent_v_neg_max_deg,
                self.incoming_direction_tangent_v_pos_max_deg,
            ),
        )
        inbound_limit_deg = math.degrees(
            math.acos(self.incoming_inbound_min_cosine)
        )
        if center_to_inbound_axis_deg + incoming_radius_max_deg > (
            inbound_limit_deg + UNIT_VECTOR_TOLERANCE
        ):
            raise ValueError(
                "incoming_direction maximum tangent support violates "
                "the inbound cone contract"
            )

        for index, axis in enumerate(("x", "y", "z")):
            _validate_side_std(
                center=self.contact_offset_center_b_yaw_m[index],
                lower_bound=self.contact_offset_min_b_yaw_m[index],
                upper_bound=self.contact_offset_max_b_yaw_m[index],
                lower_initial=self.contact_offset_std_lower_initial_m[index],
                lower_maximum=self.contact_offset_std_lower_max_m[index],
                upper_initial=self.contact_offset_std_upper_initial_m[index],
                upper_maximum=self.contact_offset_std_upper_max_m[index],
                name=f"contact_offset.{axis}",
            )
        if (
            self.contact_offset_std_lower_max_m[0]
            > self.contact_offset_std_lower_max_m[1]
            or self.contact_offset_std_upper_max_m[0]
            > self.contact_offset_std_upper_max_m[1]
        ):
            raise ValueError("contact x std must not exceed y std")

        _validate_side_std(
            center=self.time_to_contact_center_s,
            lower_bound=self.time_to_contact_min_s,
            upper_bound=self.time_to_contact_max_s,
            lower_initial=self.time_to_contact_std_lower_initial_s,
            lower_maximum=self.time_to_contact_std_lower_max_s,
            upper_initial=self.time_to_contact_std_upper_initial_s,
            upper_maximum=self.time_to_contact_std_upper_max_s,
            name="time_to_contact",
        )
        _validate_side_std(
            center=self.incoming_speed_center_mps,
            lower_bound=self.incoming_speed_min_mps,
            upper_bound=self.incoming_speed_max_mps,
            lower_initial=self.incoming_speed_std_lower_initial_mps,
            lower_maximum=self.incoming_speed_std_lower_max_mps,
            upper_initial=self.incoming_speed_std_upper_initial_mps,
            upper_maximum=self.incoming_speed_std_upper_max_mps,
            name="incoming_speed",
        )
        if self.incoming_speed_max_mps <= self.incoming_speed_min_mps:
            raise ValueError("incoming speed bounds must have positive width")
        if not math.isclose(
            self.incoming_speed_min_mps,
            0.4 * self.incoming_speed_center_mps,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "incoming_speed_min_mps must equal exactly 0.4 times "
                "incoming_speed_center_mps"
            )
        _validate_side_std(
            center=self.spin_magnitude_center_radps,
            lower_bound=self.spin_magnitude_min_radps,
            upper_bound=self.spin_magnitude_max_radps,
            lower_initial=self.spin_magnitude_std_lower_initial_radps,
            lower_maximum=self.spin_magnitude_std_lower_max_radps,
            upper_initial=self.spin_magnitude_std_upper_initial_radps,
            upper_maximum=self.spin_magnitude_std_upper_max_radps,
            name="spin_magnitude",
        )

        for index, axis in enumerate(("x", "y", "z")):
            _validate_side_std(
                center=self.base_spawn_center_w_m[index],
                lower_bound=self.base_spawn_min_w_m[index],
                upper_bound=self.base_spawn_max_w_m[index],
                lower_initial=self.base_spawn_std_lower_initial_m[index],
                lower_maximum=self.base_spawn_std_lower_max_m[index],
                upper_initial=self.base_spawn_std_upper_initial_m[index],
                upper_maximum=self.base_spawn_std_upper_max_m[index],
                name=f"base_spawn.{axis}",
            )
            _validate_side_std(
                center=self.base_travel_center_b_yaw_m[index],
                lower_bound=self.base_travel_min_b_yaw_m[index],
                upper_bound=self.base_travel_max_b_yaw_m[index],
                lower_initial=self.base_travel_std_lower_initial_m[index],
                lower_maximum=self.base_travel_std_lower_max_m[index],
                upper_initial=self.base_travel_std_upper_initial_m[index],
                upper_maximum=self.base_travel_std_upper_max_m[index],
                name=f"base_travel.{axis}",
            )
        for name in (
            "base_spawn_center_w_m",
            "base_spawn_std_lower_initial_m",
            "base_spawn_std_lower_max_m",
            "base_spawn_std_upper_initial_m",
            "base_spawn_std_upper_max_m",
            "base_spawn_min_w_m",
            "base_spawn_max_w_m",
            "base_travel_center_b_yaw_m",
            "base_travel_std_lower_initial_m",
            "base_travel_std_lower_max_m",
            "base_travel_std_upper_initial_m",
            "base_travel_std_upper_max_m",
            "base_travel_min_b_yaw_m",
            "base_travel_max_b_yaw_m",
        ):
            if getattr(self, name)[2] != 0.0:
                raise ValueError(f"{name} z must be exactly zero")
        for index, axis in enumerate(("x", "y")):
            _validate_side_std(
                center=self.landing_aim_center_w_xy_m[index],
                lower_bound=self.landing_aim_min_w_xy_m[index],
                upper_bound=self.landing_aim_max_w_xy_m[index],
                lower_initial=self.landing_aim_std_lower_initial_m[index],
                lower_maximum=self.landing_aim_std_lower_max_m[index],
                upper_initial=self.landing_aim_std_upper_initial_m[index],
                upper_maximum=self.landing_aim_std_upper_max_m[index],
                name=f"landing_aim.{axis}",
            )

        if self.reference_t_hit_s <= 0.0:
            raise ValueError("reference_t_hit_s must be > 0")
        if self.reference_t_cycle_s <= self.reference_t_hit_s:
            raise ValueError(
                "reference_t_cycle_s must be > reference_t_hit_s"
            )
        if self.reference_racket_site_speed_mps <= 0.0:
            raise ValueError(
                "reference_racket_site_speed_mps must be > 0"
            )
        if self.teacher_rate_min <= 0.0:
            raise ValueError("teacher_rate_min must be > 0")
        if not self.teacher_rate_min <= 1.0 <= self.teacher_rate_max:
            raise ValueError(
                "teacher rate range must contain native rate 1.0"
            )
        if self.time_to_contact_min_s < (
            self.reference_t_hit_s / self.teacher_rate_min
            + self.reaction_margin_s
        ):
            raise ValueError(
                "time_to_contact_min_s violates reaction margin"
            )
        if (
            self.time_to_contact_max_s
            - self.reference_t_hit_s / self.teacher_rate_max
            > 1.0 + 1.0e-12
        ):
            raise ValueError("maximum pre_swing_wait must be <= 1.0 s")
        if self.mobility_mode not in ("no_move", "move"):
            raise ValueError("mobility_mode must be 'no_move' or 'move'")

    def as_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = list(value) if isinstance(value, tuple) else value
        return result

    @property
    def sha256(self) -> str:
        return _sha256_json(self.as_dict())


@dataclass(frozen=True)
class BaseBirthReceipt:
    """Verified true-reset base spawn, reusable by all swings in an episode."""

    birth_id: str
    sampler_contract_sha256: str
    arm_catalog_sha256: str
    action_uid: int
    domain_epoch: int
    domain_levels: DomainLevels
    profile_sha256: str
    levels_sha256: str
    birth_index: int
    draw_start: int
    draw_end: int
    mobility_mode: str
    base_yaw_rad: float
    base_start_w_m: Vec3

    def to_state_dict(self) -> Dict[str, object]:
        """Canonical flat checkpoint row used by deterministic replay."""

        return {
            "birth_id": self.birth_id,
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.as_dict(),
            "profile_sha256": self.profile_sha256,
            "levels_sha256": self.levels_sha256,
            "birth_index": self.birth_index,
            "draw_start": self.draw_start,
            "draw_end": self.draw_end,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_start_w_m": list(self.base_start_w_m),
        }

    def to_identity_receipt(self) -> Dict[str, object]:
        """Flat strict receipt accepted by ``assert_issued_birth``."""

        return self.to_state_dict()

    @classmethod
    def from_identity_receipt(
        cls,
        value: object,
    ) -> "BaseBirthReceipt":
        row = _exact_mapping(
            value,
            _BIRTH_STATE_KEYS,
            name="birth identity receipt",
        )
        levels = DomainLevels.from_mapping(row["domain_levels"])
        levels_sha256 = _sha256_hex(
            row["levels_sha256"], name="birth.levels_sha256"
        )
        if levels_sha256 != levels.sha256:
            raise ValueError("birth levels hash mismatch")
        mode = row["mobility_mode"]
        if mode not in ("no_move", "move"):
            raise ValueError(
                "birth.mobility_mode must be 'no_move' or 'move'"
            )
        result = cls(
            birth_id=_sha256_hex(
                row["birth_id"], name="birth.birth_id"
            ),
            sampler_contract_sha256=_sha256_hex(
                row["sampler_contract_sha256"],
                name="birth.sampler_contract_sha256",
            ),
            arm_catalog_sha256=_sha256_hex(
                row["arm_catalog_sha256"],
                name="birth.arm_catalog_sha256",
            ),
            action_uid=_plain_int(
                row["action_uid"], name="birth.action_uid", minimum=1
            ),
            domain_epoch=_plain_int(
                row["domain_epoch"], name="birth.domain_epoch"
            ),
            domain_levels=levels,
            profile_sha256=_sha256_hex(
                row["profile_sha256"], name="birth.profile_sha256"
            ),
            levels_sha256=levels_sha256,
            birth_index=_plain_int(
                row["birth_index"], name="birth.birth_index"
            ),
            draw_start=_plain_int(
                row["draw_start"], name="birth.draw_start"
            ),
            draw_end=_plain_int(
                row["draw_end"], name="birth.draw_end"
            ),
            mobility_mode=mode,
            base_yaw_rad=_finite(
                row["base_yaw_rad"], name="birth.base_yaw_rad"
            ),
            base_start_w_m=_vec3(
                row["base_start_w_m"], name="birth.base_start_w_m"
            ),
        )
        if result.draw_end - result.draw_start != DRAWS_PER_BIRTH:
            raise ValueError("birth receipt has invalid draw range")
        if result.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("birth arm catalog hash mismatch")
        payload = _birth_identity_payload(
            sampler_contract_sha256=result.sampler_contract_sha256,
            arm_catalog_sha256=result.arm_catalog_sha256,
            action_uid=result.action_uid,
            domain_epoch=result.domain_epoch,
            levels_sha256=result.levels_sha256,
            profile_sha256=result.profile_sha256,
            birth_index=result.birth_index,
            draw_start=result.draw_start,
            draw_end=result.draw_end,
            mobility_mode=result.mobility_mode,
            base_yaw_rad=result.base_yaw_rad,
            base_start_w_m=result.base_start_w_m,
        )
        if result.birth_id != _sha256_json(payload):
            raise ValueError("birth_id does not match canonical identity")
        return result

    def to_receipt(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "birth_id": self.birth_id,
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.as_dict(),
            "profile_sha256": self.profile_sha256,
            "levels_sha256": self.levels_sha256,
            "birth_index": self.birth_index,
            "draw_range": {
                "action_tape": self.action_uid,
                "start_inclusive": self.draw_start,
                "end_exclusive": self.draw_end,
            },
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_start_w_m": list(self.base_start_w_m),
        }


# Public semantic name used by compact runtime ledgers.  Keep the historical
# BaseBirthReceipt spelling as the canonical class so existing receipts retain
# exact dataclass equality and serialization.
EpisodeBirthReceipt = BaseBirthReceipt


class CompactedSampleError(ValueError):
    """Raw replay is unavailable after its authority prefix is folded.

    A compaction segment head proves checkpoint-chain continuity only.  It is
    deliberately not presented as a membership proof for one retired receipt.
    Callers must compare the segment/global head with their externally pinned
    rollout checkpoint.
    """

    def __init__(
        self,
        *,
        action_uid: int,
        authority_kind: str,
        authority_index: int,
        segment_head_sha256: str,
    ) -> None:
        self.action_uid = action_uid
        self.authority_kind = authority_kind
        self.authority_index = authority_index
        self.segment_head_sha256 = segment_head_sha256
        super().__init__(
            f"{authority_kind}_index {authority_index} for action_uid "
            f"{action_uid} is compacted; raw receipt replay/assertion is "
            "unavailable; verify the externally pinned compaction segment "
            f"head {segment_head_sha256}"
        )


@dataclass(frozen=True)
class SamplerRetirePrefixBarrier:
    """Rollout-boundary compare-and-compact request for one action tape."""

    action_uid: int
    retire_birth_through_inclusive: int
    retire_sample_through_inclusive: int
    expected_birth_highwater: Tuple[int, int]
    expected_sample_highwater: Tuple[int, int]
    expected_assignment_head_sha256: str

    def __post_init__(self) -> None:
        _plain_int(self.action_uid, name="barrier.action_uid", minimum=1)
        _plain_int(
            self.retire_birth_through_inclusive,
            name="barrier.retire_birth_through_inclusive",
            minimum=-1,
        )
        _plain_int(
            self.retire_sample_through_inclusive,
            name="barrier.retire_sample_through_inclusive",
            minimum=-1,
        )
        object.__setattr__(
            self,
            "expected_birth_highwater",
            _highwater_pair(
                self.expected_birth_highwater,
                name="barrier.expected_birth_highwater",
            ),
        )
        object.__setattr__(
            self,
            "expected_sample_highwater",
            _highwater_pair(
                self.expected_sample_highwater,
                name="barrier.expected_sample_highwater",
            ),
        )
        _sha256_hex(
            self.expected_assignment_head_sha256,
            name="barrier.expected_assignment_head_sha256",
        )


@dataclass(frozen=True)
class SamplerCompactionReceipt:
    """One compact checkpoint segment, safe to pin outside sampler state."""

    action_uid: int
    sampler_contract_sha256: str
    segment_index: int
    retired_birth_count: int
    retired_sample_count: int
    retained_birth_start_inclusive: int
    retained_sample_start_inclusive: int
    issued_birth_count: int
    issued_sample_count: int
    issued_draw_count: int
    retired_birth_highwater_draw_end: int
    retired_sample_highwater_draw_end: int
    prior_segment_head_sha256: str
    birth_transcript_sha256: str
    sample_assignment_transcript_sha256: str
    retired_assignment_head_sha256: str
    assignment_head_sha256: str
    segment_sha256: str
    segment_head_sha256: str

    def __post_init__(self) -> None:
        _plain_int(
            self.action_uid,
            name="compaction_receipt.action_uid",
            minimum=1,
        )
        _sha256_hex(
            self.sampler_contract_sha256,
            name="compaction_receipt.sampler_contract_sha256",
        )
        _plain_int(
            self.segment_index,
            name="compaction_receipt.segment_index",
        )
        retired_birth_count = _plain_int(
            self.retired_birth_count,
            name="compaction_receipt.retired_birth_count",
        )
        retired_sample_count = _plain_int(
            self.retired_sample_count,
            name="compaction_receipt.retired_sample_count",
        )
        if retired_birth_count + retired_sample_count == 0:
            raise ValueError(
                "compaction receipt must retire at least one authority row"
            )
        retained_birth_start = _plain_int(
            self.retained_birth_start_inclusive,
            name=(
                "compaction_receipt.retained_birth_start_inclusive"
            ),
        )
        retained_sample_start = _plain_int(
            self.retained_sample_start_inclusive,
            name=(
                "compaction_receipt.retained_sample_start_inclusive"
            ),
        )
        issued_birth_count = _plain_int(
            self.issued_birth_count,
            name="compaction_receipt.issued_birth_count",
        )
        issued_sample_count = _plain_int(
            self.issued_sample_count,
            name="compaction_receipt.issued_sample_count",
        )
        issued_draw_count = _plain_int(
            self.issued_draw_count,
            name="compaction_receipt.issued_draw_count",
        )
        if retained_birth_start > issued_birth_count:
            raise ValueError(
                "compaction receipt retained birth start exceeds "
                "issued_birth_count"
            )
        if retained_sample_start > issued_sample_count:
            raise ValueError(
                "compaction receipt retained sample start exceeds "
                "issued_sample_count"
            )
        if issued_draw_count != (
            issued_birth_count * DRAWS_PER_BIRTH
            + issued_sample_count * DRAWS_PER_SAMPLE
        ):
            raise ValueError(
                "compaction receipt issued_draw_count is inconsistent"
            )
        _plain_int(
            self.retired_birth_highwater_draw_end,
            name=(
                "compaction_receipt."
                "retired_birth_highwater_draw_end"
            ),
        )
        _plain_int(
            self.retired_sample_highwater_draw_end,
            name=(
                "compaction_receipt."
                "retired_sample_highwater_draw_end"
            ),
        )
        for field_name in (
            "prior_segment_head_sha256",
            "birth_transcript_sha256",
            "sample_assignment_transcript_sha256",
            "retired_assignment_head_sha256",
            "assignment_head_sha256",
            "segment_sha256",
            "segment_head_sha256",
        ):
            _sha256_hex(
                getattr(self, field_name),
                name=f"compaction_receipt.{field_name}",
            )
        expected_segment_sha256 = _sha256_json(
            self._segment_payload()
        )
        if self.segment_sha256 != expected_segment_sha256:
            raise ValueError("compaction receipt segment hash mismatch")
        expected_head = _sha256_json(
            {
                "kind": "action_ball_compaction_segment_head",
                "state_schema_version": STATE_SCHEMA_VERSION,
                "action_uid": self.action_uid,
                "segment_index": self.segment_index,
                "prior_segment_head_sha256": (
                    self.prior_segment_head_sha256
                ),
                "segment_sha256": self.segment_sha256,
            }
        )
        if self.segment_head_sha256 != expected_head:
            raise ValueError("compaction receipt segment head mismatch")

    def _segment_payload(self) -> Dict[str, object]:
        return {
            "kind": "action_ball_compaction_segment",
            "state_schema_version": STATE_SCHEMA_VERSION,
            "action_uid": self.action_uid,
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "segment_index": self.segment_index,
            "retired_birth_count": self.retired_birth_count,
            "retired_sample_count": self.retired_sample_count,
            "retained_birth_start_inclusive": (
                self.retained_birth_start_inclusive
            ),
            "retained_sample_start_inclusive": (
                self.retained_sample_start_inclusive
            ),
            "issued_birth_count": self.issued_birth_count,
            "issued_sample_count": self.issued_sample_count,
            "issued_draw_count": self.issued_draw_count,
            "retired_birth_highwater_draw_end": (
                self.retired_birth_highwater_draw_end
            ),
            "retired_sample_highwater_draw_end": (
                self.retired_sample_highwater_draw_end
            ),
            "prior_segment_head_sha256": (
                self.prior_segment_head_sha256
            ),
            "birth_transcript_sha256": self.birth_transcript_sha256,
            "sample_assignment_transcript_sha256": (
                self.sample_assignment_transcript_sha256
            ),
            "retired_assignment_head_sha256": (
                self.retired_assignment_head_sha256
            ),
            "assignment_head_sha256": self.assignment_head_sha256,
        }

    def to_state_dict(self) -> Dict[str, object]:
        return {
            "action_uid": self.action_uid,
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "segment_index": self.segment_index,
            "retired_birth_count": self.retired_birth_count,
            "retired_sample_count": self.retired_sample_count,
            "retained_birth_start_inclusive": (
                self.retained_birth_start_inclusive
            ),
            "retained_sample_start_inclusive": (
                self.retained_sample_start_inclusive
            ),
            "issued_birth_count": self.issued_birth_count,
            "issued_sample_count": self.issued_sample_count,
            "issued_draw_count": self.issued_draw_count,
            "retired_birth_highwater_draw_end": (
                self.retired_birth_highwater_draw_end
            ),
            "retired_sample_highwater_draw_end": (
                self.retired_sample_highwater_draw_end
            ),
            "prior_segment_head_sha256": (
                self.prior_segment_head_sha256
            ),
            "birth_transcript_sha256": self.birth_transcript_sha256,
            "sample_assignment_transcript_sha256": (
                self.sample_assignment_transcript_sha256
            ),
            "retired_assignment_head_sha256": (
                self.retired_assignment_head_sha256
            ),
            "assignment_head_sha256": self.assignment_head_sha256,
            "segment_sha256": self.segment_sha256,
            "segment_head_sha256": self.segment_head_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: object,
    ) -> "SamplerCompactionReceipt":
        row = _exact_mapping(
            value,
            _COMPACTION_SEGMENT_KEYS,
            name="compaction segment",
        )
        return cls(
            action_uid=_plain_int(
                row["action_uid"],
                name="compaction_segment.action_uid",
                minimum=1,
            ),
            sampler_contract_sha256=_sha256_hex(
                row["sampler_contract_sha256"],
                name=(
                    "compaction_segment.sampler_contract_sha256"
                ),
            ),
            segment_index=_plain_int(
                row["segment_index"],
                name="compaction_segment.segment_index",
            ),
            retired_birth_count=_plain_int(
                row["retired_birth_count"],
                name="compaction_segment.retired_birth_count",
            ),
            retired_sample_count=_plain_int(
                row["retired_sample_count"],
                name="compaction_segment.retired_sample_count",
            ),
            retained_birth_start_inclusive=_plain_int(
                row["retained_birth_start_inclusive"],
                name=(
                    "compaction_segment."
                    "retained_birth_start_inclusive"
                ),
            ),
            retained_sample_start_inclusive=_plain_int(
                row["retained_sample_start_inclusive"],
                name=(
                    "compaction_segment."
                    "retained_sample_start_inclusive"
                ),
            ),
            issued_birth_count=_plain_int(
                row["issued_birth_count"],
                name="compaction_segment.issued_birth_count",
            ),
            issued_sample_count=_plain_int(
                row["issued_sample_count"],
                name="compaction_segment.issued_sample_count",
            ),
            issued_draw_count=_plain_int(
                row["issued_draw_count"],
                name="compaction_segment.issued_draw_count",
            ),
            retired_birth_highwater_draw_end=_plain_int(
                row["retired_birth_highwater_draw_end"],
                name=(
                    "compaction_segment."
                    "retired_birth_highwater_draw_end"
                ),
            ),
            retired_sample_highwater_draw_end=_plain_int(
                row["retired_sample_highwater_draw_end"],
                name=(
                    "compaction_segment."
                    "retired_sample_highwater_draw_end"
                ),
            ),
            prior_segment_head_sha256=_sha256_hex(
                row["prior_segment_head_sha256"],
                name=(
                    "compaction_segment."
                    "prior_segment_head_sha256"
                ),
            ),
            birth_transcript_sha256=_sha256_hex(
                row["birth_transcript_sha256"],
                name=(
                    "compaction_segment.birth_transcript_sha256"
                ),
            ),
            sample_assignment_transcript_sha256=_sha256_hex(
                row["sample_assignment_transcript_sha256"],
                name=(
                    "compaction_segment."
                    "sample_assignment_transcript_sha256"
                ),
            ),
            retired_assignment_head_sha256=_sha256_hex(
                row["retired_assignment_head_sha256"],
                name=(
                    "compaction_segment."
                    "retired_assignment_head_sha256"
                ),
            ),
            assignment_head_sha256=_sha256_hex(
                row["assignment_head_sha256"],
                name="compaction_segment.assignment_head_sha256",
            ),
            segment_sha256=_sha256_hex(
                row["segment_sha256"],
                name="compaction_segment.segment_sha256",
            ),
            segment_head_sha256=_sha256_hex(
                row["segment_head_sha256"],
                name="compaction_segment.segment_head_sha256",
            ),
        )


@dataclass(frozen=True)
class BallBaseSample:
    """One immutable sample and its downstream-solver receipt."""

    sample_id: str
    sampler_contract_sha256: str
    arm_catalog_sha256: str
    action_uid: int
    domain_epoch: int
    domain_levels: DomainLevels
    sample_index: int
    birth_id: str
    profile_sha256: str
    levels_sha256: str
    draw_start: int
    draw_end: int
    mobility_mode: str
    base_yaw_rad: float
    base_start_w_m: Vec3
    base_spawn_latent_w_m: Vec3
    base_travel_latent_b_yaw_m: Vec3
    base_goal_w_m: Vec3
    contact_offset_from_base_goal_b_yaw_m: Vec3
    contact_w_m: Vec3
    time_to_contact_s: float
    incoming_speed_mps: float
    incoming_direction_b_yaw: Vec3
    incoming_direction_w: Vec3
    incoming_velocity_w_mps: Vec3
    spin_magnitude_radps: float
    spin_direction_b_yaw: Vec3
    spin_direction_w: Vec3
    spin_w_radps: Vec3
    landing_aim_w_xy_m: Vec2

    def identity_payload(self) -> Dict[str, object]:
        """Return every immutable field covered by ``sample_id``."""

        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "swing_sample",
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "sample_index": self.sample_index,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.as_dict(),
            "birth_id": self.birth_id,
            "profile_sha256": self.profile_sha256,
            "levels_sha256": self.levels_sha256,
            "draw_start": self.draw_start,
            "draw_end": self.draw_end,
            "mobility_mode": self.mobility_mode,
            "base_yaw_rad": self.base_yaw_rad,
            "base_start_w_m": self.base_start_w_m,
            "base_spawn_latent_w_m": self.base_spawn_latent_w_m,
            "base_travel_latent_b_yaw_m": (
                self.base_travel_latent_b_yaw_m
            ),
            "base_goal_w_m": self.base_goal_w_m,
            "contact_offset_from_base_goal_b_yaw_m": (
                self.contact_offset_from_base_goal_b_yaw_m
            ),
            "contact_w_m": self.contact_w_m,
            "time_to_contact_s": self.time_to_contact_s,
            "incoming_speed_mps": self.incoming_speed_mps,
            "incoming_direction_b_yaw": self.incoming_direction_b_yaw,
            "incoming_direction_w": self.incoming_direction_w,
            "incoming_velocity_w_mps": self.incoming_velocity_w_mps,
            "spin_magnitude_radps": self.spin_magnitude_radps,
            "spin_direction_b_yaw": self.spin_direction_b_yaw,
            "spin_direction_w": self.spin_direction_w,
            "spin_w_radps": self.spin_w_radps,
            "landing_aim_w_xy_m": self.landing_aim_w_xy_m,
        }

    def verify_sample_id(self) -> None:
        """Fail closed if any identity field no longer matches ``sample_id``."""

        if (
            type(self.sample_id) is not str
            or self.sample_id != _sha256_json(self.identity_payload())
        ):
            raise ValueError("sample_id does not match canonical identity")

    def to_identity_receipt(self) -> Dict[str, object]:
        """Return the strict flat receipt accepted by provenance assertions."""

        return {
            "sample_id": self.sample_id,
            **self.identity_payload(),
        }

    @classmethod
    def from_identity_receipt(
        cls,
        value: object,
    ) -> "BallBaseSample":
        row = _exact_mapping(
            value,
            ("sample_id", *_SAMPLE_IDENTITY_KEYS),
            name="sample identity receipt",
        )
        if (
            _plain_int(
                row["schema_version"], name="sample.schema_version"
            )
            != SCHEMA_VERSION
        ):
            raise ValueError(
                f"sample.schema_version must be {SCHEMA_VERSION}"
            )
        if row["kind"] != "swing_sample":
            raise ValueError("sample.kind must be 'swing_sample'")
        levels = DomainLevels.from_mapping(row["domain_levels"])
        levels_sha256 = _sha256_hex(
            row["levels_sha256"], name="sample.levels_sha256"
        )
        if levels_sha256 != levels.sha256:
            raise ValueError("sample levels hash mismatch")
        mode = row["mobility_mode"]
        if mode not in ("no_move", "move"):
            raise ValueError(
                "sample.mobility_mode must be 'no_move' or 'move'"
            )
        result = cls(
            sample_id=_sha256_hex(
                row["sample_id"], name="sample.sample_id"
            ),
            sampler_contract_sha256=_sha256_hex(
                row["sampler_contract_sha256"],
                name="sample.sampler_contract_sha256",
            ),
            arm_catalog_sha256=_sha256_hex(
                row["arm_catalog_sha256"],
                name="sample.arm_catalog_sha256",
            ),
            action_uid=_plain_int(
                row["action_uid"],
                name="sample.action_uid",
                minimum=1,
            ),
            domain_epoch=_plain_int(
                row["domain_epoch"], name="sample.domain_epoch"
            ),
            domain_levels=levels,
            sample_index=_plain_int(
                row["sample_index"], name="sample.sample_index"
            ),
            birth_id=_sha256_hex(
                row["birth_id"], name="sample.birth_id"
            ),
            profile_sha256=_sha256_hex(
                row["profile_sha256"], name="sample.profile_sha256"
            ),
            levels_sha256=levels_sha256,
            draw_start=_plain_int(
                row["draw_start"], name="sample.draw_start"
            ),
            draw_end=_plain_int(
                row["draw_end"], name="sample.draw_end"
            ),
            mobility_mode=mode,
            base_yaw_rad=_finite(
                row["base_yaw_rad"], name="sample.base_yaw_rad"
            ),
            base_start_w_m=_vec3(
                row["base_start_w_m"], name="sample.base_start_w_m"
            ),
            base_spawn_latent_w_m=_vec3(
                row["base_spawn_latent_w_m"],
                name="sample.base_spawn_latent_w_m",
            ),
            base_travel_latent_b_yaw_m=_vec3(
                row["base_travel_latent_b_yaw_m"],
                name="sample.base_travel_latent_b_yaw_m",
            ),
            base_goal_w_m=_vec3(
                row["base_goal_w_m"], name="sample.base_goal_w_m"
            ),
            contact_offset_from_base_goal_b_yaw_m=_vec3(
                row["contact_offset_from_base_goal_b_yaw_m"],
                name=(
                    "sample.contact_offset_from_base_goal_b_yaw_m"
                ),
            ),
            contact_w_m=_vec3(
                row["contact_w_m"], name="sample.contact_w_m"
            ),
            time_to_contact_s=_finite(
                row["time_to_contact_s"],
                name="sample.time_to_contact_s",
                minimum=0.0,
            ),
            incoming_speed_mps=_finite(
                row["incoming_speed_mps"],
                name="sample.incoming_speed_mps",
                minimum=0.0,
            ),
            incoming_direction_b_yaw=_vec3(
                row["incoming_direction_b_yaw"],
                name="sample.incoming_direction_b_yaw",
            ),
            incoming_direction_w=_vec3(
                row["incoming_direction_w"],
                name="sample.incoming_direction_w",
            ),
            incoming_velocity_w_mps=_vec3(
                row["incoming_velocity_w_mps"],
                name="sample.incoming_velocity_w_mps",
            ),
            spin_magnitude_radps=_finite(
                row["spin_magnitude_radps"],
                name="sample.spin_magnitude_radps",
                minimum=0.0,
            ),
            spin_direction_b_yaw=_vec3(
                row["spin_direction_b_yaw"],
                name="sample.spin_direction_b_yaw",
            ),
            spin_direction_w=_vec3(
                row["spin_direction_w"],
                name="sample.spin_direction_w",
            ),
            spin_w_radps=_vec3(
                row["spin_w_radps"], name="sample.spin_w_radps"
            ),
            landing_aim_w_xy_m=_vec2(
                row["landing_aim_w_xy_m"],
                name="sample.landing_aim_w_xy_m",
            ),
        )
        if result.draw_end - result.draw_start != DRAWS_PER_SAMPLE:
            raise ValueError("sample receipt has invalid draw range")
        if result.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("sample arm catalog hash mismatch")
        result.verify_sample_id()
        return result

    def to_receipt(self) -> Dict[str, object]:
        """Return a JSON-safe receipt with an explicitly unresolved task."""

        return {
            "schema_version": SCHEMA_VERSION,
            "canonical_identity": self.to_identity_receipt(),
            "sample_id": self.sample_id,
            "sampler_contract_sha256": self.sampler_contract_sha256,
            "arm_catalog_sha256": self.arm_catalog_sha256,
            "action_uid": self.action_uid,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.as_dict(),
            "sample_index": self.sample_index,
            "profile_sha256": self.profile_sha256,
            "levels_sha256": self.levels_sha256,
            "draw_range": {
                "action_tape": self.action_uid,
                "start_inclusive": self.draw_start,
                "end_exclusive": self.draw_end,
            },
            "birth_id": self.birth_id,
            "mobility_mode": self.mobility_mode,
            "frames": {
                "world": "W",
                "base_relative": "B_yaw",
            },
            "base_yaw_rad": self.base_yaw_rad,
            "base": {
                "start_w_m": list(self.base_start_w_m),
                "spawn_latent_w_m": list(self.base_spawn_latent_w_m),
                "travel_latent_b_yaw_m": list(
                    self.base_travel_latent_b_yaw_m
                ),
                "goal_w_m": list(self.base_goal_w_m),
            },
            "ball": {
                "contact_offset_from_base_goal_b_yaw_m": list(
                    self.contact_offset_from_base_goal_b_yaw_m
                ),
                "contact_w_m": list(self.contact_w_m),
                "time_to_contact_s": self.time_to_contact_s,
                "incoming_speed_mps": self.incoming_speed_mps,
                "incoming_direction_b_yaw": list(
                    self.incoming_direction_b_yaw
                ),
                "incoming_direction_w": list(self.incoming_direction_w),
                "incoming_velocity_w_mps": list(
                    self.incoming_velocity_w_mps
                ),
                "spin_magnitude_radps": self.spin_magnitude_radps,
                "spin_direction_b_yaw": list(
                    self.spin_direction_b_yaw
                ),
                "spin_direction_w": list(self.spin_direction_w),
                "spin_w_radps": list(self.spin_w_radps),
            },
            "solver_input": {
                "landing_aim_w_xy_m": list(self.landing_aim_w_xy_m),
            },
            # A solver must replace this with its own receipt.  Keeping the key
            # explicit prevents a sampled ball from being mistaken for a task.
            "task": None,
        }


def _birth_identity_payload(
    *,
    sampler_contract_sha256: str,
    arm_catalog_sha256: str,
    action_uid: int,
    domain_epoch: int,
    levels_sha256: str,
    profile_sha256: str,
    birth_index: int,
    draw_start: int,
    draw_end: int,
    mobility_mode: str,
    base_yaw_rad: float,
    base_start_w_m: Vec3,
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "base_birth",
        "sampler_contract_sha256": sampler_contract_sha256,
        "arm_catalog_sha256": arm_catalog_sha256,
        "action_uid": action_uid,
        "domain_epoch": domain_epoch,
        "levels_sha256": levels_sha256,
        "profile_sha256": profile_sha256,
        "birth_index": birth_index,
        "draw_start": draw_start,
        "draw_end": draw_end,
        "mobility_mode": mobility_mode,
        "base_yaw_rad": base_yaw_rad,
        "base_start_w_m": base_start_w_m,
    }


class _CounterRng:
    """Stable open-interval uniforms keyed by seed, counter, and request."""

    def __init__(self, seed: int, draw_count: int = 0) -> None:
        self.seed = _plain_int(seed, name="seed")
        self.draw_count = _plain_int(draw_count, name="draw_count")

    def uniform_open(self, request_digest: bytes) -> float:
        if self.draw_count >= INT64_MAX:
            raise OverflowError("random draw counter exhausted")
        payload = (
            b"action-ball-sampling/counter/v1\0"
            + self.seed.to_bytes(8, byteorder="big", signed=False)
            + self.draw_count.to_bytes(8, byteorder="big", signed=False)
            + request_digest
        )
        bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        mantissa = bits >> 11
        self.draw_count += 1
        return (float(mantissa) + 0.5) / _TWO_POW_53


class _ReplaySampleBirthIndexLedger:
    """Constant-space stand-in used only by an isolated replay sampler."""

    def __init__(self, expected_length: int) -> None:
        self._length = expected_length

    def __len__(self) -> int:
        return self._length

    def append(self, _birth_index: int) -> None:
        self._length += 1


def _sample_truncated_normal(
    *,
    center: float,
    std: float,
    lower: float,
    upper: float,
    uniform: float,
    name: str,
) -> float:
    if not lower <= center <= upper:
        raise ValueError(f"{name} center lies outside bounds")
    if upper < lower:
        raise ValueError(f"{name} has invalid bounds")
    if std == 0.0 or upper == lower:
        return center
    low_probability = _NORMAL.cdf((lower - center) / std)
    high_probability = _NORMAL.cdf((upper - center) / std)
    if not high_probability > low_probability:
        raise ValueError(
            f"{name} truncated probability collapsed; widen bounds or std"
        )
    probability = low_probability + uniform * (
        high_probability - low_probability
    )
    probability = min(
        _LARGEST_FLOAT_BELOW_ONE,
        max(_SMALLEST_POSITIVE_FLOAT, probability),
    )
    result = center + std * _NORMAL.inv_cdf(probability)
    # Floating-point inverse-CDF roundoff can escape by a few ulps.
    return min(upper, max(lower, result))


def _sample_vector_truncated(
    *,
    center: Vec3,
    std: Vec3,
    lower: Vec3,
    upper: Vec3,
    uniforms: Sequence[float],
    name: str,
) -> Vec3:
    return tuple(
        _sample_truncated_normal(
            center=center[index],
            std=std[index],
            lower=lower[index],
            upper=upper[index],
            uniform=uniforms[index],
            name=f"{name}[{index}]",
        )
        for index in range(3)
    )  # type: ignore[return-value]


def _sample_vector2_truncated(
    *,
    center: Vec2,
    std: Vec2,
    lower: Vec2,
    upper: Vec2,
    uniforms: Sequence[float],
    name: str,
) -> Vec2:
    return tuple(
        _sample_truncated_normal(
            center=center[index],
            std=std[index],
            lower=lower[index],
            upper=upper[index],
            uniform=uniforms[index],
            name=f"{name}[{index}]",
        )
        for index in range(2)
    )  # type: ignore[return-value]


def _sample_asymmetric_truncated(
    *,
    center: float,
    lower_std: float,
    upper_std: float,
    lower_bound: float,
    upper_bound: float,
    uniform: float,
    name: str,
) -> float:
    """Single UNIFORM draw over [center-lower_width, center+upper_width].

    Franco 2026-07-28 二次裁定:不选侧——就是一次均匀采样,区间两端由两侧当前支撑
    半宽(`*_std` 字段,随课程档位独立进退)决定,硬 bounds 封顶。一侧扩张会按宽度
    比例分走采样量,这是设计意图(区间即任务域,均匀覆盖)。
    """

    if not lower_bound <= center <= upper_bound:
        raise ValueError(f"{name} center lies outside bounds")
    lower_width = min(lower_std, center - lower_bound)
    upper_width = min(upper_std, upper_bound - center)
    if lower_width < 0.0 or upper_width < 0.0:
        raise ValueError(f"{name} has negative support width")
    low = center - lower_width
    return low + uniform * (lower_width + upper_width)


def _sample_asymmetric_vector3(
    *,
    center: Vec3,
    lower_std: Vec3,
    upper_std: Vec3,
    lower_bound: Vec3,
    upper_bound: Vec3,
    uniforms: Sequence[float],
    name: str,
) -> Vec3:
    return tuple(
        _sample_asymmetric_truncated(
            center=center[index],
            lower_std=lower_std[index],
            upper_std=upper_std[index],
            lower_bound=lower_bound[index],
            upper_bound=upper_bound[index],
            uniform=uniforms[index],
            name=f"{name}[{index}]",
        )
        for index in range(3)
    )  # type: ignore[return-value]


def _sample_asymmetric_vector2(
    *,
    center: Vec2,
    lower_std: Vec2,
    upper_std: Vec2,
    lower_bound: Vec2,
    upper_bound: Vec2,
    uniforms: Sequence[float],
    name: str,
) -> Vec2:
    return tuple(
        _sample_asymmetric_truncated(
            center=center[index],
            lower_std=lower_std[index],
            upper_std=upper_std[index],
            lower_bound=lower_bound[index],
            upper_bound=upper_bound[index],
            uniform=uniforms[index],
            name=f"{name}[{index}]",
        )
        for index in range(2)
    )  # type: ignore[return-value]


def _sample_signed_tangent_angle_rad(
    *,
    negative_width_deg: float,
    positive_width_deg: float,
    uniform: float,
) -> float:
    """Sample a tangent angle uniformly over [-negative_width, +positive_width].

    Franco 2026-07-28 二次裁定:不选侧,单次均匀采样覆盖整个不对称角区间;两侧宽度
    随课程档位独立进退。
    """

    negative_width = math.radians(negative_width_deg)
    positive_width = math.radians(positive_width_deg)
    total = negative_width + positive_width
    if total == 0.0:
        return 0.0
    return -negative_width + uniform * total


def _direction_from_tangent_angles(
    *,
    center: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    angle_u_rad: float,
    angle_v_rad: float,
) -> Vec3:
    delta = _add(
        _scale(tangent_u, angle_u_rad),
        _scale(tangent_v, angle_v_rad),
    )
    radius = math.hypot(angle_u_rad, angle_v_rad)
    if radius == 0.0:
        return center
    direction = _add(
        _scale(center, math.cos(radius)),
        _scale(delta, math.sin(radius) / radius),
    )
    return _normalize_generated(
        direction, name="sampled tangent direction"
    )


def _sample_asymmetric_direction(
    *,
    center: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    u_negative_width_deg: float,
    u_positive_width_deg: float,
    v_negative_width_deg: float,
    v_positive_width_deg: float,
    uniforms: Sequence[float],
) -> Vec3:
    return _direction_from_tangent_angles(
        center=center,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        angle_u_rad=_sample_signed_tangent_angle_rad(
            negative_width_deg=u_negative_width_deg,
            positive_width_deg=u_positive_width_deg,
            uniform=uniforms[0],
        ),
        angle_v_rad=_sample_signed_tangent_angle_rad(
            negative_width_deg=v_negative_width_deg,
            positive_width_deg=v_positive_width_deg,
            uniform=uniforms[1],
        ),
    )


def _sample_direction_cone(
    center: Vec3,
    cone_deg: float,
    radial_uniform: float,
    azimuth_uniform: float,
) -> Vec3:
    """Uniformly sample solid angle inside a spherical cap."""

    center = _strict_unit(center, name="cone center")
    cone_rad = math.radians(cone_deg)
    cosine_theta = 1.0 - radial_uniform * (1.0 - math.cos(cone_rad))
    cosine_theta = min(1.0, max(-1.0, cosine_theta))
    sine_theta = math.sqrt(max(0.0, 1.0 - cosine_theta * cosine_theta))
    azimuth = 2.0 * math.pi * azimuth_uniform

    # Pick the coordinate axis least aligned with the center to avoid a
    # near-zero cross product.
    abs_components = tuple(abs(component) for component in center)
    auxiliary: Vec3
    if abs_components[0] <= abs_components[1] and abs_components[0] <= abs_components[2]:
        auxiliary = (1.0, 0.0, 0.0)
    elif abs_components[1] <= abs_components[2]:
        auxiliary = (0.0, 1.0, 0.0)
    else:
        auxiliary = (0.0, 0.0, 1.0)
    tangent_u = _normalize_generated(
        _cross(center, auxiliary), name="cone tangent"
    )
    tangent_v = _cross(center, tangent_u)
    direction = _add(
        _scale(center, cosine_theta),
        _scale(
            _add(
                _scale(tangent_u, math.cos(azimuth)),
                _scale(tangent_v, math.sin(azimuth)),
            ),
            sine_theta,
        ),
    )
    return _normalize_generated(direction, name="sampled cone direction")


class ActionBallSampler:
    """Arbitrary-N sampler with one independent random tape per action UID."""

    def __init__(
        self,
        profiles: Sequence[SamplingProfile],
        *,
        seed: int,
    ) -> None:
        if not isinstance(profiles, (tuple, list)) or not profiles:
            raise ValueError("profiles must be a non-empty tuple/list")
        ordered: Dict[int, SamplingProfile] = {}
        for index, profile in enumerate(profiles):
            if not isinstance(profile, SamplingProfile):
                raise TypeError(
                    f"profiles[{index}] must be a SamplingProfile"
                )
            if profile.action_uid in ordered:
                raise ValueError(
                    f"duplicate action_uid {profile.action_uid}"
                )
            ordered[profile.action_uid] = profile
        self._profiles = dict(sorted(ordered.items()))
        self._seed = _plain_int(seed, name="seed")
        self._rng_by_action = {
            uid: _CounterRng(self._seed) for uid in self.action_uids
        }
        self._birth_count_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._sample_count_by_action = {
            uid: 0 for uid in self.action_uids
        }
        # Compact active sample authority: one birth_index integer for each
        # retained sample_index.  Its list position is relative to the retired
        # sample prefix, not absolute sample index.
        self._issued_sample_birth_indices_by_action: Dict[
            int, List[int]
        ] = {
            uid: [] for uid in self.action_uids
        }
        # Authoritative issuance transcript.  A caller-recomputed receipt hash
        # proves only self-consistency, not that this sampler issued the birth.
        # Keeping the exact ID by action/index closes that gap for active and
        # pending rollout authority.  Retired prefixes are folded only through
        # an explicit atomic rollout-boundary barrier below.
        self._issued_births_by_action: Dict[
            int, Dict[int, BaseBirthReceipt]
        ] = {
            uid: {} for uid in self.action_uids
        }
        self._contract_sha256 = _sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "state_schema_version": STATE_SCHEMA_VERSION,
                "arm_catalog_sha256": ARM_CATALOG_SHA256,
                "seed": self._seed,
                "draws_per_birth": DRAWS_PER_BIRTH,
                "draws_per_sample": DRAWS_PER_SAMPLE,
                "profiles": [
                    self._profiles[uid].as_dict() for uid in self.action_uids
                ],
            }
        )
        self._retired_birth_count_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._retired_sample_count_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._retired_birth_highwater_draw_end_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._retired_sample_highwater_draw_end_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._retired_assignment_head_by_action = {
            uid: self._assignment_genesis_sha256(uid)
            for uid in self.action_uids
        }
        self._assignment_head_by_action = dict(
            self._retired_assignment_head_by_action
        )
        self._compaction_segments_by_action: Dict[
            int, List[SamplerCompactionReceipt]
        ] = {
            uid: [] for uid in self.action_uids
        }

    @property
    def action_uids(self) -> Tuple[int, ...]:
        return tuple(self._profiles)

    @property
    def sampler_contract_sha256(self) -> str:
        return self._contract_sha256

    @property
    def draw_count(self) -> int:
        return sum(rng.draw_count for rng in self._rng_by_action.values())

    @property
    def birth_count(self) -> int:
        return sum(self._birth_count_by_action.values())

    @property
    def sample_count(self) -> int:
        return sum(self._sample_count_by_action.values())

    def draw_count_for(self, action_uid: int) -> int:
        return self._rng_for(action_uid).draw_count

    def birth_count_for(self, action_uid: int) -> int:
        action_uid = self._validated_action_uid(action_uid)
        return self._birth_count_by_action[action_uid]

    def sample_count_for(self, action_uid: int) -> int:
        action_uid = self._validated_action_uid(action_uid)
        return self._sample_count_by_action[action_uid]

    def retired_prefix_for(
        self,
        action_uid: int,
    ) -> Tuple[int, int]:
        """Return cumulative retired ``(birth_count, sample_count)``."""

        action_uid = self._validated_action_uid(action_uid)
        return (
            self._retired_birth_count_by_action[action_uid],
            self._retired_sample_count_by_action[action_uid],
        )

    def assignment_head_for(self, action_uid: int) -> str:
        """Return the append-only sample-to-birth assignment head."""

        action_uid = self._validated_action_uid(action_uid)
        return self._assignment_head_by_action[action_uid]

    def action_compaction_head_for(self, action_uid: int) -> str:
        """Return the action segment head suitable for external pinning."""

        action_uid = self._validated_action_uid(action_uid)
        segments = self._compaction_segments_by_action[action_uid]
        if segments:
            return segments[-1].segment_head_sha256
        return self._compaction_genesis_sha256(action_uid)

    @property
    def compaction_head_sha256(self) -> str:
        """Canonical global head over all per-action compacted prefixes."""

        return _sha256_json(
            {
                "kind": "action_ball_global_compaction_head",
                "state_schema_version": STATE_SCHEMA_VERSION,
                "sampler_contract_sha256": self._contract_sha256,
                "actions": [
                    {
                        "action_uid": uid,
                        "retired_birth_count": (
                            self._retired_birth_count_by_action[uid]
                        ),
                        "retired_sample_count": (
                            self._retired_sample_count_by_action[uid]
                        ),
                        "segment_count": len(
                            self._compaction_segments_by_action[uid]
                        ),
                        "segment_head_sha256": (
                            self.action_compaction_head_for(uid)
                        ),
                    }
                    for uid in self.action_uids
                ],
            }
        )

    def birth_highwater_for(
        self,
        action_uid: int,
    ) -> Tuple[int, int]:
        """Return exact ``(last_birth_index, last_birth_draw_end)``.

        Samples issued after the latest birth advance the action tape but must
        not change this birth-only authority. ``(-1, 0)`` means no birth has
        been issued for the action.
        """

        action_uid = self._validated_action_uid(action_uid)
        count = self._birth_count_by_action[action_uid]
        if count == 0:
            return (-1, 0)
        last_index = count - 1
        receipt = self._issued_births_by_action[action_uid].get(
            last_index
        )
        if receipt is None:
            if last_index < self._retired_birth_count_by_action[action_uid]:
                return (
                    last_index,
                    self._retired_birth_highwater_draw_end_by_action[
                        action_uid
                    ],
                )
            raise RuntimeError(
                "retained birth transcript is inconsistent with birth_count"
            )
        return (last_index, receipt.draw_end)

    def sample_highwater_for(
        self,
        action_uid: int,
    ) -> Tuple[int, int]:
        """Return exact ``(last_sample_index, last_sample_draw_end)``.

        A birth may be reserved after the most recent sample, so total action
        draw count is not a valid sample high-water.  ``(-1, 0)`` denotes an
        action for which no sample has ever been issued.
        """

        action_uid = self._validated_action_uid(action_uid)
        count = self._sample_count_by_action[action_uid]
        if count == 0:
            return (-1, 0)
        last_index = count - 1
        if last_index < self._retired_sample_count_by_action[action_uid]:
            return (
                last_index,
                self._retired_sample_highwater_draw_end_by_action[
                    action_uid
                ],
            )
        draw_start = self._sample_draw_start_for_index(
            action_uid=action_uid,
            sample_index=last_index,
        )
        return (last_index, draw_start + DRAWS_PER_SAMPLE)

    def _validated_action_uid(self, action_uid: object) -> int:
        action_uid = _plain_int(
            action_uid, name="action_uid", minimum=1
        )
        if action_uid not in self._profiles:
            raise ValueError(f"unknown action_uid {action_uid}")
        return action_uid

    def _rng_for(self, action_uid: object) -> _CounterRng:
        return self._rng_by_action[
            self._validated_action_uid(action_uid)
        ]

    @staticmethod
    def _validated_levels(
        levels: Union[DomainLevels, Mapping[str, object]],
    ) -> DomainLevels:
        if isinstance(levels, DomainLevels):
            return levels
        return DomainLevels.from_mapping(levels)

    @staticmethod
    def _request_digest(
        *,
        kind: str,
        action_uid: int,
        domain_epoch: int,
        levels: DomainLevels,
    ) -> bytes:
        return bytes.fromhex(
            _sha256_json(
                {
                    "kind": kind,
                    "action_uid": action_uid,
                    "domain_epoch": domain_epoch,
                    "levels_sha256": levels.sha256,
                }
            )
        )

    def _assignment_genesis_sha256(
        self,
        action_uid: int,
    ) -> str:
        return _sha256_json(
            {
                "kind": "action_ball_sample_assignment_genesis",
                "state_schema_version": STATE_SCHEMA_VERSION,
                "sampler_contract_sha256": self._contract_sha256,
                "action_uid": action_uid,
            }
        )

    def _assignment_step_sha256(
        self,
        *,
        action_uid: int,
        previous_head_sha256: str,
        sample_index: int,
        birth_index: int,
    ) -> str:
        return _sha256_json(
            {
                "kind": "action_ball_sample_assignment_step",
                "state_schema_version": STATE_SCHEMA_VERSION,
                "sampler_contract_sha256": self._contract_sha256,
                "action_uid": action_uid,
                "previous_head_sha256": previous_head_sha256,
                "sample_index": sample_index,
                "birth_index": birth_index,
            }
        )

    def _assignment_suffix_head_sha256(
        self,
        *,
        action_uid: int,
        previous_head_sha256: str,
        sample_start_index: int,
        birth_indices: Sequence[int],
    ) -> str:
        head = previous_head_sha256
        for offset, birth_index in enumerate(birth_indices):
            head = self._assignment_step_sha256(
                action_uid=action_uid,
                previous_head_sha256=head,
                sample_index=sample_start_index + offset,
                birth_index=birth_index,
            )
        return head

    def _compaction_genesis_sha256(self, action_uid: int) -> str:
        return _sha256_json(
            {
                "kind": "action_ball_compaction_genesis",
                "state_schema_version": STATE_SCHEMA_VERSION,
                "sampler_contract_sha256": self._contract_sha256,
                "action_uid": action_uid,
            }
        )

    def reserve_birth(
        self,
        *,
        action_uid: int,
        domain_epoch: int,
        levels: Union[DomainLevels, Mapping[str, object]],
        base_yaw_rad: float = 0.0,
    ) -> BaseBirthReceipt:
        """Sample the true-reset base spawn once for a new episode."""

        action_uid = self._validated_action_uid(action_uid)
        domain_epoch = _plain_int(domain_epoch, name="domain_epoch")
        levels = self._validated_levels(levels)
        base_yaw_rad = _finite(base_yaw_rad, name="base_yaw_rad")
        profile = self._profiles[action_uid]
        rng = self._rng_by_action[action_uid]
        birth_index = self._birth_count_by_action[action_uid]
        if (
            rng.draw_count > INT64_MAX - DRAWS_PER_BIRTH
            or birth_index >= INT64_MAX
        ):
            raise OverflowError("action birth tape exhausted")

        draw_start = rng.draw_count
        request_digest = self._request_digest(
            kind="base_birth",
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            levels=levels,
        )
        uniforms = [
            rng.uniform_open(request_digest)
            for _ in range(DRAWS_PER_BIRTH)
        ]
        base_start_w_m = _sample_asymmetric_vector3(
            center=profile.base_spawn_center_w_m,
            lower_std=_vec3_lerp_levels(
                profile.base_spawn_std_lower_initial_m,
                profile.base_spawn_std_lower_max_m,
                (
                    levels.base_spawn_x_lower,
                    levels.base_spawn_y_lower,
                    0.0,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.base_spawn_std_upper_initial_m,
                profile.base_spawn_std_upper_max_m,
                (
                    levels.base_spawn_x_upper,
                    levels.base_spawn_y_upper,
                    0.0,
                ),
            ),
            lower_bound=profile.base_spawn_min_w_m,
            upper_bound=profile.base_spawn_max_w_m,
            uniforms=uniforms,
            name="base_birth_spawn",
        )
        draw_end = rng.draw_count
        if draw_end - draw_start != DRAWS_PER_BIRTH:
            raise AssertionError("internal birth fixed-draw contract violated")
        payload = _birth_identity_payload(
            sampler_contract_sha256=self._contract_sha256,
            arm_catalog_sha256=ARM_CATALOG_SHA256,
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            levels_sha256=levels.sha256,
            profile_sha256=profile.sha256,
            birth_index=birth_index,
            draw_start=draw_start,
            draw_end=draw_end,
            mobility_mode=profile.mobility_mode,
            base_yaw_rad=base_yaw_rad,
            base_start_w_m=base_start_w_m,
        )
        receipt = BaseBirthReceipt(
            birth_id=_sha256_json(payload),
            sampler_contract_sha256=self._contract_sha256,
            arm_catalog_sha256=ARM_CATALOG_SHA256,
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            domain_levels=levels,
            profile_sha256=profile.sha256,
            levels_sha256=levels.sha256,
            birth_index=birth_index,
            draw_start=draw_start,
            draw_end=draw_end,
            mobility_mode=profile.mobility_mode,
            base_yaw_rad=base_yaw_rad,
            base_start_w_m=base_start_w_m,
        )
        self._issued_births_by_action[action_uid][
            birth_index
        ] = receipt
        self._birth_count_by_action[action_uid] = birth_index + 1
        return receipt

    def _validate_birth(
        self,
        birth: BaseBirthReceipt,
        *,
        action_uid: int,
        domain_epoch: int,
        levels: DomainLevels,
        base_yaw_rad: float,
    ) -> SamplingProfile:
        if not isinstance(birth, BaseBirthReceipt):
            raise TypeError("birth must be a BaseBirthReceipt")
        profile = self._profiles[action_uid]
        mismatches = []
        expected_fields = (
            ("sampler_contract_sha256", self._contract_sha256),
            ("arm_catalog_sha256", ARM_CATALOG_SHA256),
            ("action_uid", action_uid),
            ("domain_epoch", domain_epoch),
            ("domain_levels", levels),
            ("profile_sha256", profile.sha256),
            ("levels_sha256", levels.sha256),
            ("mobility_mode", profile.mobility_mode),
            ("base_yaw_rad", base_yaw_rad),
        )
        for name, expected in expected_fields:
            if getattr(birth, name) != expected:
                mismatches.append(name)
        if mismatches:
            raise ValueError(
                "birth receipt does not match sample request/profile: "
                + ", ".join(mismatches)
            )
        _plain_int(birth.birth_index, name="birth.birth_index")
        _plain_int(birth.draw_start, name="birth.draw_start")
        _plain_int(birth.draw_end, name="birth.draw_end")
        base_start = _vec3(
            birth.base_start_w_m, name="birth.base_start_w_m"
        )
        if birth.draw_end - birth.draw_start != DRAWS_PER_BIRTH:
            raise ValueError("birth receipt has invalid draw range")
        if (
            birth.birth_index
            < self._retired_birth_count_by_action[action_uid]
        ):
            raise CompactedSampleError(
                action_uid=action_uid,
                authority_kind="birth",
                authority_index=birth.birth_index,
                segment_head_sha256=(
                    self.action_compaction_head_for(action_uid)
                ),
            )
        issued_birth = self._issued_births_by_action[action_uid].get(
            birth.birth_index
        )
        if issued_birth is None or birth != issued_birth:
            raise ValueError(
                "birth receipt does not match the exact issued transcript"
            )
        if (
            birth.birth_index
            >= self._birth_count_by_action[action_uid]
            or birth.draw_end > self._rng_by_action[action_uid].draw_count
        ):
            raise ValueError(
                "birth receipt is not present in this sampler state"
            )
        payload = _birth_identity_payload(
            sampler_contract_sha256=birth.sampler_contract_sha256,
            arm_catalog_sha256=birth.arm_catalog_sha256,
            action_uid=birth.action_uid,
            domain_epoch=birth.domain_epoch,
            levels_sha256=birth.levels_sha256,
            profile_sha256=birth.profile_sha256,
            birth_index=birth.birth_index,
            draw_start=birth.draw_start,
            draw_end=birth.draw_end,
            mobility_mode=birth.mobility_mode,
            base_yaw_rad=birth.base_yaw_rad,
            base_start_w_m=base_start,
        )
        if type(birth.birth_id) is not str or birth.birth_id != _sha256_json(
            payload
        ):
            raise ValueError("birth receipt identity check failed")
        return profile

    def sample(
        self,
        *,
        birth: BaseBirthReceipt,
        action_uid: int,
        domain_epoch: int,
        levels: Union[DomainLevels, Mapping[str, object]],
        base_yaw_rad: float = 0.0,
    ) -> BallBaseSample:
        """Sample a new ball/aim against a verified episode birth."""

        action_uid = self._validated_action_uid(action_uid)
        domain_epoch = _plain_int(domain_epoch, name="domain_epoch")
        levels = self._validated_levels(levels)
        base_yaw_rad = _finite(base_yaw_rad, name="base_yaw_rad")
        profile = self._validate_birth(
            birth,
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            levels=levels,
            base_yaw_rad=base_yaw_rad,
        )
        rng = self._rng_by_action[action_uid]
        sample_index = self._sample_count_by_action[action_uid]
        sample_birth_indices = (
            self._issued_sample_birth_indices_by_action[action_uid]
        )
        retained_sample_start = (
            self._retired_sample_count_by_action[action_uid]
        )
        if len(sample_birth_indices) != (
            sample_index - retained_sample_start
        ):
            raise RuntimeError(
                "sample authority ledger is inconsistent with sample_count"
            )
        if (
            rng.draw_count > INT64_MAX - DRAWS_PER_SAMPLE
            or sample_index >= INT64_MAX
        ):
            raise OverflowError("action swing tape exhausted")

        draw_start = rng.draw_count
        request_digest = self._request_digest(
            kind="swing_sample",
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            levels=levels,
        )
        # Public draw order: latent spawn 3, latent travel 3, contact 3,
        # time-to-contact 1, speed 1, incoming direction 2, spin magnitude 1,
        # spin direction 2, landing aim 2.  No branch may alter this budget.
        uniforms = [
            rng.uniform_open(request_digest)
            for _ in range(DRAWS_PER_SAMPLE)
        ]
        base_spawn_latent_w_m = _sample_asymmetric_vector3(
            center=profile.base_spawn_center_w_m,
            lower_std=_vec3_lerp_levels(
                profile.base_spawn_std_lower_initial_m,
                profile.base_spawn_std_lower_max_m,
                (
                    levels.base_spawn_x_lower,
                    levels.base_spawn_y_lower,
                    0.0,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.base_spawn_std_upper_initial_m,
                profile.base_spawn_std_upper_max_m,
                (
                    levels.base_spawn_x_upper,
                    levels.base_spawn_y_upper,
                    0.0,
                ),
            ),
            lower_bound=profile.base_spawn_min_w_m,
            upper_bound=profile.base_spawn_max_w_m,
            uniforms=uniforms[0:3],
            name="base_spawn_latent",
        )
        base_travel_latent_b_yaw_m = _sample_asymmetric_vector3(
            center=profile.base_travel_center_b_yaw_m,
            lower_std=_vec3_lerp_levels(
                profile.base_travel_std_lower_initial_m,
                profile.base_travel_std_lower_max_m,
                (
                    levels.base_travel_x_lower,
                    levels.base_travel_y_lower,
                    0.0,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.base_travel_std_upper_initial_m,
                profile.base_travel_std_upper_max_m,
                (
                    levels.base_travel_x_upper,
                    levels.base_travel_y_upper,
                    0.0,
                ),
            ),
            lower_bound=profile.base_travel_min_b_yaw_m,
            upper_bound=profile.base_travel_max_b_yaw_m,
            uniforms=uniforms[3:6],
            name="base_travel",
        )
        base_start_w_m = birth.base_start_w_m
        if profile.mobility_mode == "no_move":
            base_goal_w_m = base_start_w_m
        else:
            base_goal_w_m = _add(
                base_start_w_m,
                _rotate_yaw(base_travel_latent_b_yaw_m, base_yaw_rad),
            )

        contact_offset_b_yaw_m = _sample_asymmetric_vector3(
            center=profile.contact_offset_center_b_yaw_m,
            lower_std=_vec3_lerp_levels(
                profile.contact_offset_std_lower_initial_m,
                profile.contact_offset_std_lower_max_m,
                (
                    levels.contact_x_lower,
                    levels.contact_y_lower,
                    levels.contact_z_lower,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.contact_offset_std_upper_initial_m,
                profile.contact_offset_std_upper_max_m,
                (
                    levels.contact_x_upper,
                    levels.contact_y_upper,
                    levels.contact_z_upper,
                ),
            ),
            lower_bound=profile.contact_offset_min_b_yaw_m,
            upper_bound=profile.contact_offset_max_b_yaw_m,
            uniforms=uniforms[6:9],
            name="contact_offset",
        )
        contact_w_m = _add(
            base_goal_w_m,
            _rotate_yaw(contact_offset_b_yaw_m, base_yaw_rad),
        )

        time_to_contact_s = _sample_asymmetric_truncated(
            center=profile.time_to_contact_center_s,
            lower_std=_lerp(
                profile.time_to_contact_std_lower_initial_s,
                profile.time_to_contact_std_lower_max_s,
                levels.time_to_contact_lower,
            ),
            upper_std=_lerp(
                profile.time_to_contact_std_upper_initial_s,
                profile.time_to_contact_std_upper_max_s,
                levels.time_to_contact_upper,
            ),
            lower_bound=profile.time_to_contact_min_s,
            upper_bound=profile.time_to_contact_max_s,
            uniform=uniforms[9],
            name="time_to_contact",
        )
        speed_mps = _sample_asymmetric_truncated(
            center=profile.incoming_speed_center_mps,
            lower_std=_lerp(
                profile.incoming_speed_std_lower_initial_mps,
                profile.incoming_speed_std_lower_max_mps,
                levels.incoming_speed_lower,
            ),
            upper_std=_lerp(
                profile.incoming_speed_std_upper_initial_mps,
                profile.incoming_speed_std_upper_max_mps,
                levels.incoming_speed_upper,
            ),
            lower_bound=profile.incoming_speed_min_mps,
            upper_bound=profile.incoming_speed_max_mps,
            uniform=uniforms[10],
            name="incoming_speed",
        )
        incoming_direction_b_yaw = _sample_asymmetric_direction(
            center=profile.incoming_direction_center_b_yaw,
            tangent_u=profile.incoming_direction_tangent_u_b_yaw,
            tangent_v=profile.incoming_direction_tangent_v_b_yaw,
            u_negative_width_deg=_lerp(
                profile.incoming_direction_tangent_u_neg_initial_deg,
                profile.incoming_direction_tangent_u_neg_max_deg,
                levels.incoming_direction_u_neg,
            ),
            u_positive_width_deg=_lerp(
                profile.incoming_direction_tangent_u_pos_initial_deg,
                profile.incoming_direction_tangent_u_pos_max_deg,
                levels.incoming_direction_u_pos,
            ),
            v_negative_width_deg=_lerp(
                profile.incoming_direction_tangent_v_neg_initial_deg,
                profile.incoming_direction_tangent_v_neg_max_deg,
                levels.incoming_direction_v_neg,
            ),
            v_positive_width_deg=_lerp(
                profile.incoming_direction_tangent_v_pos_initial_deg,
                profile.incoming_direction_tangent_v_pos_max_deg,
                levels.incoming_direction_v_pos,
            ),
            uniforms=uniforms[11:13],
        )
        if (
            _dot(
                incoming_direction_b_yaw,
                profile.incoming_inbound_axis_b_yaw,
            )
            + 1.0e-12
            < profile.incoming_inbound_min_cosine
        ):
            raise AssertionError(
                "sampled incoming direction violates inbound cone contract"
            )
        incoming_direction_w = _rotate_yaw(
            incoming_direction_b_yaw, base_yaw_rad
        )
        incoming_velocity_w_mps = _scale(
            incoming_direction_w, speed_mps
        )

        spin_magnitude_radps = _sample_asymmetric_truncated(
            center=profile.spin_magnitude_center_radps,
            lower_std=_lerp(
                profile.spin_magnitude_std_lower_initial_radps,
                profile.spin_magnitude_std_lower_max_radps,
                levels.spin_magnitude_lower,
            ),
            upper_std=_lerp(
                profile.spin_magnitude_std_upper_initial_radps,
                profile.spin_magnitude_std_upper_max_radps,
                levels.spin_magnitude_upper,
            ),
            lower_bound=profile.spin_magnitude_min_radps,
            upper_bound=profile.spin_magnitude_max_radps,
            uniform=uniforms[13],
            name="spin_magnitude",
        )
        spin_direction_b_yaw = _sample_asymmetric_direction(
            center=profile.spin_direction_center_b_yaw,
            tangent_u=profile.spin_direction_tangent_u_b_yaw,
            tangent_v=profile.spin_direction_tangent_v_b_yaw,
            u_negative_width_deg=_lerp(
                profile.spin_direction_tangent_u_neg_initial_deg,
                profile.spin_direction_tangent_u_neg_max_deg,
                levels.spin_direction_u_neg,
            ),
            u_positive_width_deg=_lerp(
                profile.spin_direction_tangent_u_pos_initial_deg,
                profile.spin_direction_tangent_u_pos_max_deg,
                levels.spin_direction_u_pos,
            ),
            v_negative_width_deg=_lerp(
                profile.spin_direction_tangent_v_neg_initial_deg,
                profile.spin_direction_tangent_v_neg_max_deg,
                levels.spin_direction_v_neg,
            ),
            v_positive_width_deg=_lerp(
                profile.spin_direction_tangent_v_pos_initial_deg,
                profile.spin_direction_tangent_v_pos_max_deg,
                levels.spin_direction_v_pos,
            ),
            uniforms=uniforms[14:16],
        )
        spin_direction_w = _rotate_yaw(
            spin_direction_b_yaw, base_yaw_rad
        )
        spin_w_radps = _scale(
            spin_direction_w, spin_magnitude_radps
        )
        landing_aim_w_xy_m = _sample_asymmetric_vector2(
            center=profile.landing_aim_center_w_xy_m,
            lower_std=_vec2_lerp_levels(
                profile.landing_aim_std_lower_initial_m,
                profile.landing_aim_std_lower_max_m,
                (
                    levels.landing_aim_x_lower,
                    levels.landing_aim_y_lower,
                ),
            ),
            upper_std=_vec2_lerp_levels(
                profile.landing_aim_std_upper_initial_m,
                profile.landing_aim_std_upper_max_m,
                (
                    levels.landing_aim_x_upper,
                    levels.landing_aim_y_upper,
                ),
            ),
            lower_bound=profile.landing_aim_min_w_xy_m,
            upper_bound=profile.landing_aim_max_w_xy_m,
            uniforms=uniforms[16:18],
            name="landing_aim",
        )

        draw_end = rng.draw_count
        if draw_end - draw_start != DRAWS_PER_SAMPLE:
            raise AssertionError("internal fixed-draw contract violated")
        candidate = BallBaseSample(
            sample_id="",
            sampler_contract_sha256=self._contract_sha256,
            arm_catalog_sha256=ARM_CATALOG_SHA256,
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            domain_levels=levels,
            sample_index=sample_index,
            birth_id=birth.birth_id,
            profile_sha256=profile.sha256,
            levels_sha256=levels.sha256,
            draw_start=draw_start,
            draw_end=draw_end,
            mobility_mode=profile.mobility_mode,
            base_yaw_rad=base_yaw_rad,
            base_start_w_m=base_start_w_m,
            base_spawn_latent_w_m=base_spawn_latent_w_m,
            base_travel_latent_b_yaw_m=base_travel_latent_b_yaw_m,
            base_goal_w_m=base_goal_w_m,
            contact_offset_from_base_goal_b_yaw_m=contact_offset_b_yaw_m,
            contact_w_m=contact_w_m,
            time_to_contact_s=time_to_contact_s,
            incoming_speed_mps=speed_mps,
            incoming_direction_b_yaw=incoming_direction_b_yaw,
            incoming_direction_w=incoming_direction_w,
            incoming_velocity_w_mps=incoming_velocity_w_mps,
            spin_magnitude_radps=spin_magnitude_radps,
            spin_direction_b_yaw=spin_direction_b_yaw,
            spin_direction_w=spin_direction_w,
            spin_w_radps=spin_w_radps,
            landing_aim_w_xy_m=landing_aim_w_xy_m,
        )
        completed = replace(
            candidate,
            sample_id=_sha256_json(candidate.identity_payload()),
        )
        completed.verify_sample_id()
        sample_birth_indices.append(birth.birth_index)
        self._assignment_head_by_action[action_uid] = (
            self._assignment_step_sha256(
                action_uid=action_uid,
                previous_head_sha256=(
                    self._assignment_head_by_action[action_uid]
                ),
                sample_index=sample_index,
                birth_index=birth.birth_index,
            )
        )
        self._sample_count_by_action[action_uid] = sample_index + 1
        return completed

    def assert_issued_birth(
        self,
        birth_or_receipt: Union[BaseBirthReceipt, Mapping[str, object]],
    ) -> None:
        """Pure exact-authority check for a provider-issued episode birth."""

        if isinstance(birth_or_receipt, BaseBirthReceipt):
            birth = birth_or_receipt
        elif isinstance(birth_or_receipt, Mapping):
            birth = BaseBirthReceipt.from_identity_receipt(
                birth_or_receipt
            )
        else:
            raise TypeError(
                "birth_or_receipt must be BaseBirthReceipt or mapping"
            )
        self._validate_birth(
            birth,
            action_uid=birth.action_uid,
            domain_epoch=birth.domain_epoch,
            levels=birth.domain_levels,
            base_yaw_rad=birth.base_yaw_rad,
        )

    def _sample_draw_start_for_index(
        self,
        *,
        action_uid: int,
        sample_index: int,
    ) -> int:
        action_uid = self._validated_action_uid(action_uid)
        sample_index = _plain_int(
            sample_index, name="sample_index"
        )
        if sample_index >= self._sample_count_by_action[action_uid]:
            raise ValueError(
                "sample_index is not below the issued high-water mark"
            )
        if (
            sample_index
            < self._retired_sample_count_by_action[action_uid]
        ):
            raise CompactedSampleError(
                action_uid=action_uid,
                authority_kind="sample",
                authority_index=sample_index,
                segment_head_sha256=(
                    self.action_compaction_head_for(action_uid)
                ),
            )
        return self._sample_draw_start_from_births(
            sample_index=sample_index,
            births=self._issued_births_by_action[action_uid].values(),
            retired_birth_count=(
                self._retired_birth_count_by_action[action_uid]
            ),
        )

    @staticmethod
    def _sample_draw_start_from_births(
        *,
        sample_index: int,
        births: Iterable[BaseBirthReceipt],
        retired_birth_count: int = 0,
    ) -> int:
        """Map an action-local sample index around signed birth draw blocks."""

        # Start from the nth 18-draw block, then insert each 3-draw birth
        # event that occurs at or before the moving candidate position.
        # Every retired birth is earlier than every retained sample: the
        # compaction barrier rejects a retained assignment to a retired birth.
        candidate = (
            sample_index * DRAWS_PER_SAMPLE
            + retired_birth_count * DRAWS_PER_BIRTH
        )
        births = sorted(
            births,
            key=lambda receipt: receipt.draw_start,
        )
        for birth in births:
            if birth.draw_start <= candidate:
                candidate += DRAWS_PER_BIRTH
            else:
                break
        return candidate

    def replay_issued_sample(
        self,
        birth_or_receipt: Union[
            EpisodeBirthReceipt, Mapping[str, object]
        ],
        sample_index: int,
    ) -> BallBaseSample:
        """Purely reconstruct one genuinely issued sample.

        ``sample_index`` is action-local.  The compact authority ledger proves
        which exact issued episode birth owned that index; the birth transcript
        in turn owns the original domain epoch, all 32 levels, base yaw,
        profile, arm catalog, and sampler contract.  No caller-supplied sample
        fields are trusted and no sampler state is advanced.
        """

        if isinstance(birth_or_receipt, BaseBirthReceipt):
            birth = birth_or_receipt
        elif isinstance(birth_or_receipt, Mapping):
            birth = BaseBirthReceipt.from_identity_receipt(
                birth_or_receipt
            )
        else:
            raise TypeError(
                "birth_or_receipt must be an EpisodeBirthReceipt "
                "or mapping"
            )
        action_uid = self._validated_action_uid(birth.action_uid)
        sample_index = _plain_int(
            sample_index, name="sample_index"
        )
        if sample_index >= self._sample_count_by_action[action_uid]:
            raise ValueError(
                "sample_index is not below the issued high-water mark"
            )
        retained_sample_start = (
            self._retired_sample_count_by_action[action_uid]
        )
        if sample_index < retained_sample_start:
            raise CompactedSampleError(
                action_uid=action_uid,
                authority_kind="sample",
                authority_index=sample_index,
                segment_head_sha256=(
                    self.action_compaction_head_for(action_uid)
                ),
            )
        profile = self._validate_birth(
            birth,
            action_uid=action_uid,
            domain_epoch=birth.domain_epoch,
            levels=birth.domain_levels,
            base_yaw_rad=birth.base_yaw_rad,
        )
        draw_start = self._sample_draw_start_for_index(
            action_uid=action_uid,
            sample_index=sample_index,
        )
        sample_birth_indices = (
            self._issued_sample_birth_indices_by_action[action_uid]
        )
        if len(sample_birth_indices) != (
            self._sample_count_by_action[action_uid]
            - retained_sample_start
        ):
            raise RuntimeError(
                "sample authority ledger is inconsistent with sample_count"
            )
        ledger_offset = sample_index - retained_sample_start
        if sample_birth_indices[ledger_offset] != birth.birth_index:
            raise ValueError(
                "sample_index was issued under a different episode birth"
            )
        if birth.draw_end > draw_start:
            raise ValueError(
                "sample_index precedes its issued episode birth"
            )
        result = self._replay_sample_at(
            profile=profile,
            birth=birth,
            action_uid=action_uid,
            domain_epoch=birth.domain_epoch,
            levels=birth.domain_levels,
            base_yaw_rad=birth.base_yaw_rad,
            sample_index=sample_index,
            draw_start=draw_start,
        )
        result.verify_sample_id()
        return result

    def replay_issued_samples(
        self,
        birth_sample_indices: Sequence[
            Tuple[
                Union[
                    EpisodeBirthReceipt, Mapping[str, object]
                ],
                int,
            ]
        ],
    ) -> Tuple[BallBaseSample, ...]:
        """Pure mixed-birth batch form of :meth:`replay_issued_sample`."""

        if (
            isinstance(birth_sample_indices, (str, bytes))
            or not isinstance(birth_sample_indices, Sequence)
        ):
            raise TypeError(
                "birth_sample_indices must be a non-string sequence"
            )
        results = []
        for index, request in enumerate(birth_sample_indices):
            if not isinstance(request, (tuple, list)) or len(request) != 2:
                raise TypeError(
                    f"birth_sample_indices[{index}] must be a "
                    "(birth, sample_index) pair"
                )
            results.append(
                self.replay_issued_sample(request[0], request[1])
            )
        return tuple(results)

    def assert_issued_sample(
        self,
        sample_or_receipt: Union[BallBaseSample, Mapping[str, object]],
    ) -> None:
        """Pure deterministic provenance assertion for one emitted sample.

        This does not trust a self-consistent ``sample_id``.  It maps
        ``sample_index`` to the exact 18-draw action-tape block after excluding
        all signed 3-draw birth events, then replays that block from the
        sampler seed/profile/request and compares every immutable field.
        """

        if isinstance(sample_or_receipt, BallBaseSample):
            sample = sample_or_receipt
            sample.verify_sample_id()
        elif isinstance(sample_or_receipt, Mapping):
            identity: object = sample_or_receipt
            outer_receipt: Optional[Mapping[str, object]] = None
            if "canonical_identity" in sample_or_receipt:
                identity = sample_or_receipt["canonical_identity"]
                outer_receipt = sample_or_receipt
            sample = BallBaseSample.from_identity_receipt(identity)
            if outer_receipt is not None:
                expected_outer = sample.to_receipt()
                if set(outer_receipt) != set(expected_outer):
                    raise ValueError(
                        "sample outer receipt keys do not match canonical "
                        "receipt"
                    )
                for key, expected_value in expected_outer.items():
                    # The downstream solver owns this one slot; every emitted
                    # sample/base field remains exact authority.
                    if key == "task":
                        continue
                    if outer_receipt[key] != expected_value:
                        raise ValueError(
                            "sample outer receipt disagrees with canonical "
                            f"identity at {key!r}"
                        )
        else:
            raise TypeError(
                "sample_or_receipt must be BallBaseSample or mapping"
            )

        action_uid = self._validated_action_uid(sample.action_uid)
        if (
            sample.sample_index
            < self._retired_sample_count_by_action[action_uid]
        ):
            raise CompactedSampleError(
                action_uid=action_uid,
                authority_kind="sample",
                authority_index=sample.sample_index,
                segment_head_sha256=(
                    self.action_compaction_head_for(action_uid)
                ),
            )
        profile = self._profiles[action_uid]
        if sample.sampler_contract_sha256 != self._contract_sha256:
            raise ValueError("sample sampler contract mismatch")
        if sample.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("sample arm catalog hash mismatch")
        if sample.profile_sha256 != profile.sha256:
            raise ValueError("sample profile mismatch")
        if sample.levels_sha256 != sample.domain_levels.sha256:
            raise ValueError("sample levels hash mismatch")
        if sample.mobility_mode != profile.mobility_mode:
            raise ValueError("sample mobility mismatch")

        matching_births = [
            birth
            for birth in self._issued_births_by_action[action_uid].values()
            if birth.birth_id == sample.birth_id
        ]
        if len(matching_births) != 1:
            raise ValueError(
                "sample birth_id does not name one exact issued birth"
            )
        birth = matching_births[0]
        expected = self.replay_issued_sample(
            birth,
            sample.sample_index,
        )
        if (
            sample.draw_start != expected.draw_start
            or sample.draw_end != expected.draw_end
        ):
            raise ValueError(
                "sample draw range does not match its issued action-tape "
                "index"
            )
        if sample != expected:
            raise ValueError(
                "sample does not match deterministic issued replay"
            )

    def _replay_sample_at(
        self,
        *,
        profile: SamplingProfile,
        birth: BaseBirthReceipt,
        action_uid: int,
        domain_epoch: int,
        levels: DomainLevels,
        base_yaw_rad: float,
        sample_index: int,
        draw_start: int,
    ) -> BallBaseSample:
        """Replay one action-tape block without rebuilding the N-profile bank.

        ``ActionBallSampler.__init__`` canonicalizes and hashes every profile.
        Doing that for every assertion made N93 pool verification orders of
        magnitude slower.  This isolated single-action context reuses the
        already-validated profile and contract and advances only a temporary
        counter.  No authority state is mutated.
        """

        replay = object.__new__(ActionBallSampler)
        replay._profiles = {action_uid: profile}
        replay._seed = self._seed
        replay._contract_sha256 = self._contract_sha256
        replay._rng_by_action = {
            action_uid: _CounterRng(self._seed, draw_start)
        }
        replay._birth_count_by_action = {
            action_uid: birth.birth_index + 1
        }
        replay._sample_count_by_action = {
            action_uid: sample_index
        }
        replay._retired_birth_count_by_action = {
            action_uid: 0
        }
        replay._retired_sample_count_by_action = {
            action_uid: sample_index
        }
        replay._retired_birth_highwater_draw_end_by_action = {
            action_uid: 0
        }
        replay._retired_sample_highwater_draw_end_by_action = {
            action_uid: 0
        }
        assignment_genesis = replay._assignment_genesis_sha256(
            action_uid
        )
        replay._retired_assignment_head_by_action = {
            action_uid: assignment_genesis
        }
        replay._assignment_head_by_action = {
            action_uid: assignment_genesis
        }
        replay._compaction_segments_by_action = {
            action_uid: []
        }
        replay._issued_sample_birth_indices_by_action = {
            action_uid: _ReplaySampleBirthIndexLedger(0)
        }
        replay._issued_births_by_action = {
            action_uid: {birth.birth_index: birth}
        }
        return replay.sample(
            birth=birth,
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            levels=levels,
            base_yaw_rad=base_yaw_rad,
        )

    def assert_issued_samples(
        self,
        samples_or_receipts: Sequence[
            Union[BallBaseSample, Mapping[str, object]]
        ],
    ) -> None:
        """Pure batch spelling for pool/refill boundaries."""

        if (
            isinstance(samples_or_receipts, (str, bytes))
            or not isinstance(samples_or_receipts, Sequence)
        ):
            raise TypeError(
                "samples_or_receipts must be a non-string sequence"
            )
        for sample_or_receipt in samples_or_receipts:
            self.assert_issued_sample(sample_or_receipt)

    def assert_emitted_sample(
        self,
        sample_or_receipt: Union[
            BallBaseSample, Mapping[str, object]
        ],
    ) -> None:
        """Compatibility spelling for :meth:`assert_issued_sample`."""

        self.assert_issued_sample(sample_or_receipt)

    def compact_retired_prefix(
        self,
        barriers: Sequence[SamplerRetirePrefixBarrier],
    ) -> Tuple[SamplerCompactionReceipt, ...]:
        """Atomically fold rollout-safe authority prefixes.

        The runtime must call this at a rollout boundary after it has found the
        largest continuous birth/sample prefixes that no active or pending
        rollout can reference.  Every barrier is compare-and-swap guarded by
        exact birth/sample high-waters and the append-only assignment head.

        Folded transcript hashes and segment heads prove checkpoint continuity;
        they intentionally do not prove random membership of an individual
        retired receipt.  Raw replay/assertion for a folded index therefore
        fails closed with :class:`CompactedSampleError`.
        """

        if (
            isinstance(barriers, (str, bytes))
            or not isinstance(barriers, Sequence)
            or not barriers
        ):
            raise ValueError(
                "barriers must be a non-empty non-string sequence"
            )
        seen = set()
        staged = []
        for barrier_index, barrier in enumerate(barriers):
            if not isinstance(barrier, SamplerRetirePrefixBarrier):
                raise TypeError(
                    f"barriers[{barrier_index}] must be a "
                    "SamplerRetirePrefixBarrier"
                )
            action_uid = self._validated_action_uid(barrier.action_uid)
            if action_uid in seen:
                raise ValueError(
                    f"duplicate compaction barrier for action_uid "
                    f"{action_uid}"
                )
            seen.add(action_uid)
            if (
                barrier.expected_birth_highwater
                != self.birth_highwater_for(action_uid)
            ):
                raise ValueError(
                    f"action_uid {action_uid} birth high-water changed "
                    "before compaction barrier"
                )
            if (
                barrier.expected_sample_highwater
                != self.sample_highwater_for(action_uid)
            ):
                raise ValueError(
                    f"action_uid {action_uid} sample high-water changed "
                    "before compaction barrier"
                )
            if (
                barrier.expected_assignment_head_sha256
                != self._assignment_head_by_action[action_uid]
            ):
                raise ValueError(
                    f"action_uid {action_uid} assignment head changed "
                    "before compaction barrier"
                )

            old_retired_birth_count = (
                self._retired_birth_count_by_action[action_uid]
            )
            old_retired_sample_count = (
                self._retired_sample_count_by_action[action_uid]
            )
            new_retired_birth_count = (
                barrier.retire_birth_through_inclusive + 1
            )
            new_retired_sample_count = (
                barrier.retire_sample_through_inclusive + 1
            )
            birth_count = self._birth_count_by_action[action_uid]
            sample_count = self._sample_count_by_action[action_uid]
            if not (
                old_retired_birth_count
                <= new_retired_birth_count
                <= birth_count
            ):
                raise ValueError(
                    f"action_uid {action_uid} retired birth prefix must "
                    "advance monotonically within the issued high-water"
                )
            if not (
                old_retired_sample_count
                <= new_retired_sample_count
                <= sample_count
            ):
                raise ValueError(
                    f"action_uid {action_uid} retired sample prefix must "
                    "advance monotonically within the issued high-water"
                )
            retired_birth_count = (
                new_retired_birth_count - old_retired_birth_count
            )
            retired_sample_count = (
                new_retired_sample_count - old_retired_sample_count
            )
            if retired_birth_count + retired_sample_count == 0:
                raise ValueError(
                    f"action_uid {action_uid} compaction barrier does "
                    "not advance either retired prefix"
                )

            births = self._issued_births_by_action[action_uid]
            expected_birth_indices = set(
                range(old_retired_birth_count, birth_count)
            )
            if set(births) != expected_birth_indices:
                raise RuntimeError(
                    "retained birth transcript is inconsistent with "
                    "retired/birth counts"
                )
            assignments = (
                self._issued_sample_birth_indices_by_action[action_uid]
            )
            if len(assignments) != (
                sample_count - old_retired_sample_count
            ):
                raise RuntimeError(
                    "sample authority ledger is inconsistent with "
                    "retired/sample counts"
                )
            if any(birth_index not in births for birth_index in assignments):
                raise RuntimeError(
                    "retained sample authority references a non-retained "
                    "birth"
                )
            assignment_cut = (
                new_retired_sample_count - old_retired_sample_count
            )
            retired_assignments = assignments[:assignment_cut]
            retained_assignments = assignments[assignment_cut:]
            blocking_birth_indices = sorted(
                {
                    birth_index
                    for birth_index in retained_assignments
                    if birth_index < new_retired_birth_count
                }
            )
            if blocking_birth_indices:
                raise ValueError(
                    f"action_uid {action_uid} cannot retire birth prefix; "
                    "retained samples still reference birth indices "
                    f"{blocking_birth_indices[:8]}"
                )

            retired_births = [
                births[index]
                for index in range(
                    old_retired_birth_count,
                    new_retired_birth_count,
                )
            ]
            retired_birth_highwater_draw_end = (
                self._retired_birth_highwater_draw_end_by_action[
                    action_uid
                ]
            )
            if retired_births:
                retired_birth_highwater_draw_end = (
                    retired_births[-1].draw_end
                )
            retired_sample_highwater_draw_end = (
                self._retired_sample_highwater_draw_end_by_action[
                    action_uid
                ]
            )
            if retired_sample_count:
                last_retired_sample_index = new_retired_sample_count - 1
                retired_sample_highwater_draw_end = (
                    self._sample_draw_start_from_births(
                        sample_index=last_retired_sample_index,
                        births=births.values(),
                        retired_birth_count=old_retired_birth_count,
                    )
                    + DRAWS_PER_SAMPLE
                )

            birth_transcript_sha256 = _sha256_json(
                {
                    "kind": "action_ball_retired_birth_transcript",
                    "state_schema_version": STATE_SCHEMA_VERSION,
                    "sampler_contract_sha256": self._contract_sha256,
                    "action_uid": action_uid,
                    "start_inclusive": old_retired_birth_count,
                    "end_exclusive": new_retired_birth_count,
                    "births": [
                        birth.to_state_dict()
                        for birth in retired_births
                    ],
                }
            )
            sample_assignment_transcript_sha256 = _sha256_json(
                {
                    "kind": (
                        "action_ball_retired_sample_assignment_transcript"
                    ),
                    "state_schema_version": STATE_SCHEMA_VERSION,
                    "sampler_contract_sha256": self._contract_sha256,
                    "action_uid": action_uid,
                    "start_inclusive": old_retired_sample_count,
                    "end_exclusive": new_retired_sample_count,
                    "birth_indices": list(retired_assignments),
                }
            )
            retired_assignment_head_sha256 = (
                self._assignment_suffix_head_sha256(
                    action_uid=action_uid,
                    previous_head_sha256=(
                        self._retired_assignment_head_by_action[action_uid]
                    ),
                    sample_start_index=old_retired_sample_count,
                    birth_indices=retired_assignments,
                )
            )
            segments = self._compaction_segments_by_action[action_uid]
            prior_segment_head_sha256 = (
                segments[-1].segment_head_sha256
                if segments
                else self._compaction_genesis_sha256(action_uid)
            )
            receipt_values = {
                "action_uid": action_uid,
                "sampler_contract_sha256": self._contract_sha256,
                "segment_index": len(segments),
                "retired_birth_count": retired_birth_count,
                "retired_sample_count": retired_sample_count,
                "retained_birth_start_inclusive": (
                    new_retired_birth_count
                ),
                "retained_sample_start_inclusive": (
                    new_retired_sample_count
                ),
                "issued_birth_count": birth_count,
                "issued_sample_count": sample_count,
                "issued_draw_count": (
                    self._rng_by_action[action_uid].draw_count
                ),
                "retired_birth_highwater_draw_end": (
                    retired_birth_highwater_draw_end
                ),
                "retired_sample_highwater_draw_end": (
                    retired_sample_highwater_draw_end
                ),
                "prior_segment_head_sha256": (
                    prior_segment_head_sha256
                ),
                "birth_transcript_sha256": birth_transcript_sha256,
                "sample_assignment_transcript_sha256": (
                    sample_assignment_transcript_sha256
                ),
                "retired_assignment_head_sha256": (
                    retired_assignment_head_sha256
                ),
                "assignment_head_sha256": (
                    self._assignment_head_by_action[action_uid]
                ),
            }
            segment_payload = {
                "kind": "action_ball_compaction_segment",
                "state_schema_version": STATE_SCHEMA_VERSION,
                **receipt_values,
            }
            segment_sha256 = _sha256_json(segment_payload)
            segment_head_sha256 = _sha256_json(
                {
                    "kind": "action_ball_compaction_segment_head",
                    "state_schema_version": STATE_SCHEMA_VERSION,
                    "action_uid": action_uid,
                    "segment_index": len(segments),
                    "prior_segment_head_sha256": (
                        prior_segment_head_sha256
                    ),
                    "segment_sha256": segment_sha256,
                }
            )
            receipt = SamplerCompactionReceipt(
                **receipt_values,
                segment_sha256=segment_sha256,
                segment_head_sha256=segment_head_sha256,
            )
            staged.append(
                (
                    action_uid,
                    new_retired_birth_count,
                    new_retired_sample_count,
                    retired_birth_highwater_draw_end,
                    retired_sample_highwater_draw_end,
                    retired_assignment_head_sha256,
                    {
                        index: birth
                        for index, birth in births.items()
                        if index >= new_retired_birth_count
                    },
                    list(retained_assignments),
                    receipt,
                )
            )

        # Commit only after every barrier has validated and every receipt has
        # been constructed, making a multi-action rollout boundary atomic.
        receipts = []
        for (
            action_uid,
            retired_birth_count,
            retired_sample_count,
            retired_birth_highwater_draw_end,
            retired_sample_highwater_draw_end,
            retired_assignment_head_sha256,
            retained_births,
            retained_assignments,
            receipt,
        ) in staged:
            self._retired_birth_count_by_action[action_uid] = (
                retired_birth_count
            )
            self._retired_sample_count_by_action[action_uid] = (
                retired_sample_count
            )
            self._retired_birth_highwater_draw_end_by_action[action_uid] = (
                retired_birth_highwater_draw_end
            )
            self._retired_sample_highwater_draw_end_by_action[action_uid] = (
                retired_sample_highwater_draw_end
            )
            self._retired_assignment_head_by_action[action_uid] = (
                retired_assignment_head_sha256
            )
            self._issued_births_by_action[action_uid] = retained_births
            self._issued_sample_birth_indices_by_action[action_uid] = (
                retained_assignments
            )
            self._compaction_segments_by_action[action_uid].append(
                receipt
            )
            receipts.append(receipt)
        return tuple(receipts)

    def state_dict(self) -> Dict[str, object]:
        per_action = {}
        issued_sample_birth_indices = {}
        issued_births = {}
        compaction_segments = {}
        for uid in self.action_uids:
            indices = self._issued_sample_birth_indices_by_action[uid]
            retired_birth_count = (
                self._retired_birth_count_by_action[uid]
            )
            retired_sample_count = (
                self._retired_sample_count_by_action[uid]
            )
            if len(indices) != (
                self._sample_count_by_action[uid]
                - retired_sample_count
            ):
                raise RuntimeError(
                    "sample authority ledger is inconsistent with "
                    "retired/sample counts"
                )
            births = self._issued_births_by_action[uid]
            if set(births) != set(
                range(
                    retired_birth_count,
                    self._birth_count_by_action[uid],
                )
            ):
                raise RuntimeError(
                    "retained birth transcript is inconsistent with "
                    "retired/birth counts"
                )
            detached_indices = list(indices)
            issued_sample_birth_indices[str(uid)] = detached_indices
            issued_births[str(uid)] = [
                births[index].to_state_dict()
                for index in range(
                    retired_birth_count,
                    self._birth_count_by_action[uid],
                )
            ]
            segments = self._compaction_segments_by_action[uid]
            compaction_segments[str(uid)] = [
                segment.to_state_dict() for segment in segments
            ]
            expected_assignment_head = (
                self._assignment_suffix_head_sha256(
                    action_uid=uid,
                    previous_head_sha256=(
                        self._retired_assignment_head_by_action[uid]
                    ),
                    sample_start_index=retired_sample_count,
                    birth_indices=detached_indices,
                )
            )
            if (
                expected_assignment_head
                != self._assignment_head_by_action[uid]
            ):
                raise RuntimeError(
                    "sample assignment append chain is inconsistent"
                )
            per_action[str(uid)] = {
                "birth_count": self._birth_count_by_action[uid],
                "sample_count": self._sample_count_by_action[uid],
                "draw_count": self._rng_by_action[uid].draw_count,
                "retired_birth_count": retired_birth_count,
                "retired_sample_count": retired_sample_count,
                "retired_birth_highwater_draw_end": (
                    self._retired_birth_highwater_draw_end_by_action[uid]
                ),
                "retired_sample_highwater_draw_end": (
                    self._retired_sample_highwater_draw_end_by_action[uid]
                ),
                "retired_assignment_head_sha256": (
                    self._retired_assignment_head_by_action[uid]
                ),
                "assignment_head_sha256": (
                    self._assignment_head_by_action[uid]
                ),
                "compaction_segment_count": len(segments),
                "compaction_segment_head_sha256": (
                    self.action_compaction_head_for(uid)
                ),
            }
        payload: Dict[str, object] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "sampler_contract_sha256": self._contract_sha256,
            "arm_catalog_sha256": ARM_CATALOG_SHA256,
            "seed": self._seed,
            "draws_per_birth": DRAWS_PER_BIRTH,
            "draws_per_sample": DRAWS_PER_SAMPLE,
            "action_uids": list(self.action_uids),
            "per_action": per_action,
            "issued_births": issued_births,
            "issued_sample_birth_indices": issued_sample_birth_indices,
            "compaction_segments": compaction_segments,
            "compaction_head_sha256": self.compaction_head_sha256,
            "total_birth_count": self.birth_count,
            "total_sample_count": self.sample_count,
            "total_draw_count": self.draw_count,
        }
        return {**payload, "integrity_sha256": _sha256_json(payload)}

    def load_state_dict(self, state: object) -> None:
        """Strict, atomic restore; malformed or altered states change nothing."""

        row = _exact_mapping(state, _STATE_KEYS, name="sampler state")
        payload = {key: row[key] for key in _STATE_KEYS[:-1]}
        integrity = row["integrity_sha256"]
        if (
            type(integrity) is not str
            or len(integrity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in integrity
            )
        ):
            raise ValueError("sampler state integrity_sha256 is invalid")
        if _sha256_json(payload) != integrity:
            raise ValueError("sampler state integrity check failed")
        if (
            _plain_int(row["schema_version"], name="schema_version")
            != STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"sampler state schema_version must be {STATE_SCHEMA_VERSION}"
            )
        if row["sampler_contract_sha256"] != self._contract_sha256:
            raise ValueError("sampler state contract does not match profiles")
        if row["arm_catalog_sha256"] != ARM_CATALOG_SHA256:
            raise ValueError("sampler state arm catalog mismatch")
        if _plain_int(row["seed"], name="seed") != self._seed:
            raise ValueError("sampler state seed does not match sampler")
        if (
            _plain_int(
                row["draws_per_birth"],
                name="draws_per_birth",
                minimum=1,
            )
            != DRAWS_PER_BIRTH
        ):
            raise ValueError("sampler state draws_per_birth mismatch")
        if (
            _plain_int(
                row["draws_per_sample"],
                name="draws_per_sample",
                minimum=1,
            )
            != DRAWS_PER_SAMPLE
        ):
            raise ValueError("sampler state draws_per_sample mismatch")

        raw_uids = row["action_uids"]
        if not isinstance(raw_uids, list):
            raise ValueError("sampler state action_uids must be a list")
        action_uids = tuple(
            _plain_int(
                uid,
                name=f"action_uids[{index}]",
                minimum=1,
            )
            for index, uid in enumerate(raw_uids)
        )
        if action_uids != self.action_uids:
            raise ValueError("sampler state action_uids do not match profiles")
        raw_per_action = row["per_action"]
        if not isinstance(raw_per_action, Mapping):
            raise ValueError("sampler state per_action must be a mapping")
        expected_keys = {str(uid) for uid in self.action_uids}
        actual_keys = set(raw_per_action)
        if actual_keys != expected_keys:
            raise ValueError(
                "sampler state per_action keys do not match action_uids"
            )

        restored_counts: Dict[int, Tuple[int, int, int]] = {}
        restored_retired_counts: Dict[int, Tuple[int, int]] = {}
        restored_retired_highwaters: Dict[int, Tuple[int, int]] = {}
        declared_retired_assignment_heads: Dict[int, str] = {}
        declared_assignment_heads: Dict[int, str] = {}
        declared_segment_counts: Dict[int, int] = {}
        declared_segment_heads: Dict[int, str] = {}
        for uid in self.action_uids:
            action_row = _exact_mapping(
                raw_per_action[str(uid)],
                _PER_ACTION_STATE_KEYS,
                name=f"per_action[{uid}]",
            )
            births = _plain_int(
                action_row["birth_count"],
                name=f"per_action[{uid}].birth_count",
            )
            samples = _plain_int(
                action_row["sample_count"],
                name=f"per_action[{uid}].sample_count",
            )
            draws = _plain_int(
                action_row["draw_count"],
                name=f"per_action[{uid}].draw_count",
            )
            expected_draws = (
                births * DRAWS_PER_BIRTH
                + samples * DRAWS_PER_SAMPLE
            )
            if draws != expected_draws:
                raise ValueError(
                    f"per_action[{uid}] draw_count is inconsistent "
                    "with birth/sample counts"
                )
            restored_counts[uid] = (births, samples, draws)
            retired_births = _plain_int(
                action_row["retired_birth_count"],
                name=f"per_action[{uid}].retired_birth_count",
            )
            retired_samples = _plain_int(
                action_row["retired_sample_count"],
                name=f"per_action[{uid}].retired_sample_count",
            )
            if retired_births > births or retired_samples > samples:
                raise ValueError(
                    f"per_action[{uid}] retired prefix exceeds issued "
                    "counts"
                )
            if samples and not births:
                raise ValueError(
                    f"per_action[{uid}] cannot have samples without a "
                    "birth"
                )
            retired_birth_draw_end = _plain_int(
                action_row["retired_birth_highwater_draw_end"],
                name=(
                    f"per_action[{uid}]."
                    "retired_birth_highwater_draw_end"
                ),
            )
            retired_sample_draw_end = _plain_int(
                action_row["retired_sample_highwater_draw_end"],
                name=(
                    f"per_action[{uid}]."
                    "retired_sample_highwater_draw_end"
                ),
            )
            if (retired_births == 0) != (retired_birth_draw_end == 0):
                raise ValueError(
                    f"per_action[{uid}] retired birth high-water is "
                    "inconsistent"
                )
            if (retired_samples == 0) != (retired_sample_draw_end == 0):
                raise ValueError(
                    f"per_action[{uid}] retired sample high-water is "
                    "inconsistent"
                )
            if (
                retired_birth_draw_end > draws
                or retired_sample_draw_end > draws
            ):
                raise ValueError(
                    f"per_action[{uid}] retired high-water exceeds "
                    "draw_count"
                )
            restored_retired_counts[uid] = (
                retired_births,
                retired_samples,
            )
            restored_retired_highwaters[uid] = (
                retired_birth_draw_end,
                retired_sample_draw_end,
            )
            declared_retired_assignment_heads[uid] = _sha256_hex(
                action_row["retired_assignment_head_sha256"],
                name=(
                    f"per_action[{uid}]."
                    "retired_assignment_head_sha256"
                ),
            )
            declared_assignment_heads[uid] = _sha256_hex(
                action_row["assignment_head_sha256"],
                name=f"per_action[{uid}].assignment_head_sha256",
            )
            declared_segment_counts[uid] = _plain_int(
                action_row["compaction_segment_count"],
                name=f"per_action[{uid}].compaction_segment_count",
            )
            declared_segment_heads[uid] = _sha256_hex(
                action_row["compaction_segment_head_sha256"],
                name=(
                    f"per_action[{uid}]."
                    "compaction_segment_head_sha256"
                ),
            )

        raw_issued = row["issued_births"]
        if not isinstance(raw_issued, Mapping):
            raise ValueError(
                "sampler state issued_births must be a mapping"
            )
        if set(raw_issued) != expected_keys:
            raise ValueError(
                "sampler state issued_births keys do not match action_uids"
            )
        raw_sample_birth_indices = row["issued_sample_birth_indices"]
        if not isinstance(raw_sample_birth_indices, Mapping):
            raise ValueError(
                "sampler state issued_sample_birth_indices must be a mapping"
            )
        if set(raw_sample_birth_indices) != expected_keys:
            raise ValueError(
                "sampler state issued_sample_birth_indices keys do not "
                "match action_uids"
            )
        raw_compaction_segments = row["compaction_segments"]
        if not isinstance(raw_compaction_segments, Mapping):
            raise ValueError(
                "sampler state compaction_segments must be a mapping"
            )
        if set(raw_compaction_segments) != expected_keys:
            raise ValueError(
                "sampler state compaction_segments keys do not match "
                "action_uids"
            )
        restored_segments: Dict[
            int, List[SamplerCompactionReceipt]
        ] = {}
        for uid in self.action_uids:
            raw_segments = raw_compaction_segments[str(uid)]
            if not isinstance(raw_segments, list):
                raise ValueError(
                    f"compaction_segments[{uid}] must be a list"
                )
            if len(raw_segments) != declared_segment_counts[uid]:
                raise ValueError(
                    f"compaction_segments[{uid}] length is inconsistent"
                )
            segments: List[SamplerCompactionReceipt] = []
            cumulative_births = 0
            cumulative_samples = 0
            prior_segment_head = self._compaction_genesis_sha256(uid)
            prior_retired_assignment_head = (
                self._assignment_genesis_sha256(uid)
            )
            prior_birth_draw_end = 0
            prior_sample_draw_end = 0
            prior_issued_births = 0
            prior_issued_samples = 0
            prior_issued_draws = 0
            for segment_index, raw_segment in enumerate(raw_segments):
                segment = SamplerCompactionReceipt.from_state_dict(
                    raw_segment
                )
                if segment.action_uid != uid:
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "action mismatch"
                    )
                if segment.sampler_contract_sha256 != (
                    self._contract_sha256
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "contract mismatch"
                    )
                if segment.segment_index != segment_index:
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "index mismatch"
                    )
                if (
                    segment.prior_segment_head_sha256
                    != prior_segment_head
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "chain discontinuity"
                    )
                cumulative_births += segment.retired_birth_count
                cumulative_samples += segment.retired_sample_count
                if segment.retained_birth_start_inclusive != (
                    cumulative_births
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "birth prefix discontinuity"
                    )
                if segment.retained_sample_start_inclusive != (
                    cumulative_samples
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "sample prefix discontinuity"
                    )
                if (
                    segment.issued_birth_count < prior_issued_births
                    or segment.issued_sample_count < prior_issued_samples
                    or segment.issued_draw_count < prior_issued_draws
                    or segment.issued_birth_count
                    > restored_counts[uid][0]
                    or segment.issued_sample_count
                    > restored_counts[uid][1]
                    or segment.issued_draw_count
                    > restored_counts[uid][2]
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}][{segment_index}] "
                        "issued high-water is not monotonic"
                    )
                if segment.retired_birth_count == 0:
                    if (
                        segment.retired_birth_highwater_draw_end
                        != prior_birth_draw_end
                    ):
                        raise ValueError(
                            f"compaction_segments[{uid}] birth "
                            "high-water changed without retiring births"
                        )
                elif (
                    segment.retired_birth_highwater_draw_end
                    <= prior_birth_draw_end
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}] birth high-water "
                        "is not monotonic"
                    )
                if segment.retired_sample_count == 0:
                    if (
                        segment.retired_sample_highwater_draw_end
                        != prior_sample_draw_end
                    ):
                        raise ValueError(
                            f"compaction_segments[{uid}] sample "
                            "high-water changed without retiring samples"
                        )
                    if (
                        segment.retired_assignment_head_sha256
                        != prior_retired_assignment_head
                    ):
                        raise ValueError(
                            f"compaction_segments[{uid}] retired "
                            "assignment head changed without samples"
                        )
                elif (
                    segment.retired_sample_highwater_draw_end
                    <= prior_sample_draw_end
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}] sample high-water "
                        "is not monotonic"
                    )
                if (
                    segment.issued_sample_count
                    == segment.retained_sample_start_inclusive
                    and segment.assignment_head_sha256
                    != segment.retired_assignment_head_sha256
                ):
                    raise ValueError(
                        f"compaction_segments[{uid}] full retired "
                        "assignment head mismatch"
                    )
                segments.append(segment)
                prior_segment_head = segment.segment_head_sha256
                prior_retired_assignment_head = (
                    segment.retired_assignment_head_sha256
                )
                prior_birth_draw_end = (
                    segment.retired_birth_highwater_draw_end
                )
                prior_sample_draw_end = (
                    segment.retired_sample_highwater_draw_end
                )
                prior_issued_births = segment.issued_birth_count
                prior_issued_samples = segment.issued_sample_count
                prior_issued_draws = segment.issued_draw_count
            if cumulative_births != restored_retired_counts[uid][0]:
                raise ValueError(
                    f"compaction_segments[{uid}] final birth prefix "
                    "mismatch"
                )
            if cumulative_samples != restored_retired_counts[uid][1]:
                raise ValueError(
                    f"compaction_segments[{uid}] final sample prefix "
                    "mismatch"
                )
            if (
                prior_birth_draw_end,
                prior_sample_draw_end,
            ) != restored_retired_highwaters[uid]:
                raise ValueError(
                    f"compaction_segments[{uid}] final retired "
                    "high-water mismatch"
                )
            if prior_retired_assignment_head != (
                declared_retired_assignment_heads[uid]
            ):
                raise ValueError(
                    f"compaction_segments[{uid}] final retired "
                    "assignment head mismatch"
                )
            if prior_segment_head != declared_segment_heads[uid]:
                raise ValueError(
                    f"compaction_segments[{uid}] final segment head "
                    "mismatch"
                )
            restored_segments[uid] = segments

        restored_issued: Dict[
            int, Dict[int, BaseBirthReceipt]
        ] = {}
        restored_sample_birth_indices: Dict[int, List[int]] = {}
        for uid in self.action_uids:
            action_births = raw_issued[str(uid)]
            if not isinstance(action_births, list):
                raise ValueError(
                    f"issued_births[{uid}] must be a list"
                )
            expected_birth_count = restored_counts[uid][0]
            retired_birth_count = restored_retired_counts[uid][0]
            retained_birth_count = (
                expected_birth_count - retired_birth_count
            )
            if len(action_births) != retained_birth_count:
                raise ValueError(
                    f"issued_births[{uid}] length is inconsistent "
                    "with retired/birth counts"
                )
            if (
                expected_birth_count == 0
                and restored_counts[uid][1] != 0
            ):
                raise ValueError(
                    f"issued_births[{uid}] cannot have samples without "
                    "an episode birth"
                )
            indexed: Dict[int, BaseBirthReceipt] = {}
            retired_birth_draw_end = (
                restored_retired_highwaters[uid][0]
            )
            retired_birth_sample_draws = (
                retired_birth_draw_end
                - retired_birth_count * DRAWS_PER_BIRTH
            )
            if (
                retired_birth_sample_draws < 0
                or retired_birth_sample_draws % DRAWS_PER_SAMPLE != 0
            ):
                raise ValueError(
                    f"per_action[{uid}] retired birth high-water has "
                    "an impossible event lattice"
                )
            previous_draw_end = retired_birth_draw_end
            inferred_samples = (
                retired_birth_sample_draws // DRAWS_PER_SAMPLE
            )
            profile = self._profiles[uid]
            for offset, raw_birth in enumerate(action_births):
                index = retired_birth_count + offset
                birth_row = _exact_mapping(
                    raw_birth,
                    _BIRTH_STATE_KEYS,
                    name=f"issued_births[{uid}][{index}]",
                )
                birth_id = birth_row["birth_id"]
                if (
                    type(birth_id) is not str
                    or len(birth_id) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in birth_id
                    )
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] must be "
                        "64 lowercase hex"
                    )
                if (
                    birth_row["sampler_contract_sha256"]
                    != self._contract_sha256
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] contract mismatch"
                    )
                if birth_row["arm_catalog_sha256"] != ARM_CATALOG_SHA256:
                    raise ValueError(
                        f"issued_births[{uid}][{index}] arm catalog "
                        "mismatch"
                    )
                if (
                    _plain_int(
                        birth_row["action_uid"],
                        name=(
                            f"issued_births[{uid}][{index}].action_uid"
                        ),
                        minimum=1,
                    )
                    != uid
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] action mismatch"
                    )
                domain_epoch = _plain_int(
                    birth_row["domain_epoch"],
                    name=(
                        f"issued_births[{uid}][{index}].domain_epoch"
                    ),
                )
                levels = DomainLevels.from_mapping(
                    birth_row["domain_levels"]
                )
                levels_sha256 = birth_row["levels_sha256"]
                if (
                    type(levels_sha256) is not str
                    or levels_sha256 != levels.sha256
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] levels hash "
                        "mismatch"
                    )
                if birth_row["profile_sha256"] != profile.sha256:
                    raise ValueError(
                        f"issued_births[{uid}][{index}] profile mismatch"
                    )
                if (
                    _plain_int(
                        birth_row["birth_index"],
                        name=(
                            f"issued_births[{uid}][{index}].birth_index"
                        ),
                    )
                    != index
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] index mismatch"
                    )
                draw_start = _plain_int(
                    birth_row["draw_start"],
                    name=f"issued_births[{uid}][{index}].draw_start",
                )
                draw_end = _plain_int(
                    birth_row["draw_end"],
                    name=f"issued_births[{uid}][{index}].draw_end",
                )
                if draw_end - draw_start != DRAWS_PER_BIRTH:
                    raise ValueError(
                        f"issued_births[{uid}][{index}] draw range "
                        "must equal DRAWS_PER_BIRTH"
                    )
                if index == 0 and draw_start != 0:
                    raise ValueError(
                        f"issued_births[{uid}][0] must begin at draw 0"
                    )
                inter_birth_gap = draw_start - previous_draw_end
                if (
                    inter_birth_gap < 0
                    or inter_birth_gap % DRAWS_PER_SAMPLE != 0
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] has an "
                        "impossible event gap"
                    )
                inferred_samples += inter_birth_gap // DRAWS_PER_SAMPLE
                previous_draw_end = draw_end
                if birth_row["mobility_mode"] != profile.mobility_mode:
                    raise ValueError(
                        f"issued_births[{uid}][{index}] mobility mismatch"
                    )
                base_yaw_rad = _finite(
                    birth_row["base_yaw_rad"],
                    name=(
                        f"issued_births[{uid}][{index}].base_yaw_rad"
                    ),
                )
                base_start_w_m = _vec3(
                    birth_row["base_start_w_m"],
                    name=(
                        f"issued_births[{uid}][{index}].base_start_w_m"
                    ),
                )

                # Replay the counter-based birth draw.  The checkpoint row
                # cannot authorize its own base or ID merely by recomputing
                # the outer state checksum.
                replay_rng = _CounterRng(self._seed, draw_start)
                request_digest = self._request_digest(
                    kind="base_birth",
                    action_uid=uid,
                    domain_epoch=domain_epoch,
                    levels=levels,
                )
                replay_uniforms = [
                    replay_rng.uniform_open(request_digest)
                    for _ in range(DRAWS_PER_BIRTH)
                ]
                replayed_base_start = _sample_asymmetric_vector3(
                    center=profile.base_spawn_center_w_m,
                    lower_std=_vec3_lerp_levels(
                        profile.base_spawn_std_lower_initial_m,
                        profile.base_spawn_std_lower_max_m,
                        (
                            levels.base_spawn_x_lower,
                            levels.base_spawn_y_lower,
                            0.0,
                        ),
                    ),
                    upper_std=_vec3_lerp_levels(
                        profile.base_spawn_std_upper_initial_m,
                        profile.base_spawn_std_upper_max_m,
                        (
                            levels.base_spawn_x_upper,
                            levels.base_spawn_y_upper,
                            0.0,
                        ),
                    ),
                    lower_bound=profile.base_spawn_min_w_m,
                    upper_bound=profile.base_spawn_max_w_m,
                    uniforms=replay_uniforms,
                    name="replayed_base_birth_spawn",
                )
                if (
                    replay_rng.draw_count != draw_end
                    or replayed_base_start != base_start_w_m
                ):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] does not replay "
                        "from the sampler seed/profile"
                    )
                identity_payload = _birth_identity_payload(
                    sampler_contract_sha256=self._contract_sha256,
                    arm_catalog_sha256=ARM_CATALOG_SHA256,
                    action_uid=uid,
                    domain_epoch=domain_epoch,
                    levels_sha256=levels.sha256,
                    profile_sha256=profile.sha256,
                    birth_index=index,
                    draw_start=draw_start,
                    draw_end=draw_end,
                    mobility_mode=profile.mobility_mode,
                    base_yaw_rad=base_yaw_rad,
                    base_start_w_m=base_start_w_m,
                )
                if birth_id != _sha256_json(identity_payload):
                    raise ValueError(
                        f"issued_births[{uid}][{index}] identity mismatch"
                    )
                indexed[index] = BaseBirthReceipt(
                    birth_id=birth_id,
                    sampler_contract_sha256=self._contract_sha256,
                    arm_catalog_sha256=ARM_CATALOG_SHA256,
                    action_uid=uid,
                    domain_epoch=domain_epoch,
                    domain_levels=levels,
                    profile_sha256=profile.sha256,
                    levels_sha256=levels.sha256,
                    birth_index=index,
                    draw_start=draw_start,
                    draw_end=draw_end,
                    mobility_mode=profile.mobility_mode,
                    base_yaw_rad=base_yaw_rad,
                    base_start_w_m=base_start_w_m,
                )
            tail_gap = restored_counts[uid][2] - previous_draw_end
            if (
                tail_gap < 0
                or tail_gap % DRAWS_PER_SAMPLE != 0
            ):
                raise ValueError(
                    f"issued_births[{uid}] has an impossible tail gap"
                )
            inferred_samples += tail_gap // DRAWS_PER_SAMPLE
            if inferred_samples != restored_counts[uid][1]:
                raise ValueError(
                    f"issued_births[{uid}] event gaps do not account "
                    "for sample_count"
                )
            ids = [receipt.birth_id for receipt in indexed.values()]
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"issued_births[{uid}] must not contain duplicate IDs"
                )
            raw_assignments = raw_sample_birth_indices[str(uid)]
            if not isinstance(raw_assignments, list):
                raise ValueError(
                    f"issued_sample_birth_indices[{uid}] must be a list"
                )
            expected_sample_count = restored_counts[uid][1]
            retired_sample_count = restored_retired_counts[uid][1]
            retained_sample_count = (
                expected_sample_count - retired_sample_count
            )
            if len(raw_assignments) != retained_sample_count:
                raise ValueError(
                    f"issued_sample_birth_indices[{uid}] length is "
                    "inconsistent with retired/sample counts"
                )
            retired_sample_draw_end = (
                restored_retired_highwaters[uid][1]
            )
            retired_sample_birth_draws = (
                retired_sample_draw_end
                - retired_sample_count * DRAWS_PER_SAMPLE
            )
            if (
                retired_sample_birth_draws < 0
                or retired_sample_birth_draws % DRAWS_PER_BIRTH != 0
                or (
                    retired_sample_birth_draws
                    // DRAWS_PER_BIRTH
                )
                > expected_birth_count
            ):
                raise ValueError(
                    f"per_action[{uid}] retired sample high-water has "
                    "an impossible event lattice"
                )
            assignments = []
            for offset, raw_birth_index in enumerate(
                raw_assignments
            ):
                sample_index = retired_sample_count + offset
                birth_index = _plain_int(
                    raw_birth_index,
                    name=(
                        f"issued_sample_birth_indices[{uid}]"
                        f"[{sample_index}]"
                    ),
                )
                if birth_index not in indexed:
                    raise ValueError(
                        f"issued_sample_birth_indices[{uid}]"
                        f"[{sample_index}] references an unknown birth"
                    )
                sample_draw_start = (
                    self._sample_draw_start_from_births(
                        sample_index=sample_index,
                        births=indexed.values(),
                        retired_birth_count=retired_birth_count,
                    )
                )
                if indexed[birth_index].draw_end > sample_draw_start:
                    raise ValueError(
                        f"issued_sample_birth_indices[{uid}]"
                        f"[{sample_index}] assigns a birth issued after "
                        "the sample"
                    )
                assignments.append(birth_index)
            replayed_assignment_head = (
                self._assignment_suffix_head_sha256(
                    action_uid=uid,
                    previous_head_sha256=(
                        declared_retired_assignment_heads[uid]
                    ),
                    sample_start_index=retired_sample_count,
                    birth_indices=assignments,
                )
            )
            if declared_assignment_heads[uid] != replayed_assignment_head:
                raise ValueError(
                    f"issued_sample_birth_indices[{uid}] assignment "
                    "hash mismatch"
                )
            restored_issued[uid] = indexed
            restored_sample_birth_indices[uid] = assignments

        declared_compaction_head = _sha256_hex(
            row["compaction_head_sha256"],
            name="compaction_head_sha256",
        )
        expected_compaction_head = _sha256_json(
            {
                "kind": "action_ball_global_compaction_head",
                "state_schema_version": STATE_SCHEMA_VERSION,
                "sampler_contract_sha256": self._contract_sha256,
                "actions": [
                    {
                        "action_uid": uid,
                        "retired_birth_count": (
                            restored_retired_counts[uid][0]
                        ),
                        "retired_sample_count": (
                            restored_retired_counts[uid][1]
                        ),
                        "segment_count": len(restored_segments[uid]),
                        "segment_head_sha256": (
                            declared_segment_heads[uid]
                        ),
                    }
                    for uid in self.action_uids
                ],
            }
        )
        if declared_compaction_head != expected_compaction_head:
            raise ValueError("sampler state compaction head mismatch")

        total_births = _plain_int(
            row["total_birth_count"], name="total_birth_count"
        )
        total_samples = _plain_int(
            row["total_sample_count"], name="total_sample_count"
        )
        total_draws = _plain_int(
            row["total_draw_count"], name="total_draw_count"
        )
        if total_births != sum(value[0] for value in restored_counts.values()):
            raise ValueError("total_birth_count is inconsistent")
        if total_samples != sum(
            value[1] for value in restored_counts.values()
        ):
            raise ValueError("total_sample_count is inconsistent")
        if total_draws != sum(value[2] for value in restored_counts.values()):
            raise ValueError("total_draw_count is inconsistent")

        self._rng_by_action = {
            uid: _CounterRng(self._seed, restored_counts[uid][2])
            for uid in self.action_uids
        }
        self._birth_count_by_action = {
            uid: restored_counts[uid][0] for uid in self.action_uids
        }
        self._sample_count_by_action = {
            uid: restored_counts[uid][1] for uid in self.action_uids
        }
        self._retired_birth_count_by_action = {
            uid: restored_retired_counts[uid][0]
            for uid in self.action_uids
        }
        self._retired_sample_count_by_action = {
            uid: restored_retired_counts[uid][1]
            for uid in self.action_uids
        }
        self._retired_birth_highwater_draw_end_by_action = {
            uid: restored_retired_highwaters[uid][0]
            for uid in self.action_uids
        }
        self._retired_sample_highwater_draw_end_by_action = {
            uid: restored_retired_highwaters[uid][1]
            for uid in self.action_uids
        }
        self._retired_assignment_head_by_action = dict(
            declared_retired_assignment_heads
        )
        self._assignment_head_by_action = dict(
            declared_assignment_heads
        )
        self._compaction_segments_by_action = restored_segments
        self._issued_births_by_action = restored_issued
        self._issued_sample_birth_indices_by_action = (
            restored_sample_birth_indices
        )


__all__ = [
    "ARM_CATALOG_SHA256",
    "ARM_KEYS",
    "ActionBallSampler",
    "BallBaseSample",
    "BaseBirthReceipt",
    "CompactedSampleError",
    "DRAWS_PER_BIRTH",
    "DRAWS_PER_SAMPLE",
    "DomainLevels",
    "EpisodeBirthReceipt",
    "SamplerCompactionReceipt",
    "SamplerRetirePrefixBarrier",
    "SamplingProfile",
]
