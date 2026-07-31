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

from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
from statistics import NormalDist
import struct
import sys
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
_DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY = object()

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
_MIXTURE_BIRTH_STATE_KEYS = (
    *_BIRTH_STATE_KEYS[:6],
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    *_BIRTH_STATE_KEYS[6:],
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
_MIXTURE_SAMPLE_IDENTITY_KEYS = (
    *_SAMPLE_IDENTITY_KEYS[:8],
    "birth_index",
    "birth_sampling_stratum",
    "birth_sampling_levels",
    "birth_frontier_arm",
    "sampling_mixture",
    "sampling_stratum",
    "sampling_levels",
    "frontier_arm",
    *_SAMPLE_IDENTITY_KEYS[8:],
    "contact_time_step_s",
    "time_to_contact_tick",
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

_FROZEN_EVALUATION_PROPOSAL_SAMPLER_SEMANTICS = {
    "schema_version": 1,
    "kind": "action_ball_frozen_evaluation_proposal_sampler",
    "random_access": (
        "one external allocation seed owns exactly one independent "
        "birth-plus-swing 3+18 draw tape"
    ),
    "training_state_isolation": (
        "never reads, advances, restores, or compacts the training sampler "
        "tape"
    ),
    "sampling_core": (
        "ActionBallSampler reserve_birth/sample geometry with an explicit "
        "authority-owned center/interior/frontier plan"
    ),
    "mixture": "exact repeating SamplingMixture 1/3/1 quota",
    "frontier": (
        "the requested signed arm alone receives its candidate domain "
        "level; every other arm in both birth and ball-task components is "
        "held at center level zero"
    ),
    "component_strata": (
        "outer evaluation stratum is distinct from birth and ball-task "
        "component strata; a frontier belongs to exactly one component and "
        "the other component is center"
    ),
    "proposal_accounting": (
        "one call returns one birth receipt, one sample receipt, and one "
        "self-hashed proposal receipt; no redraw exists"
    ),
}


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


def frozen_evaluation_proposal_sampler_contract() -> Dict[str, object]:
    """Return the code-pinned stateless evaluator sampler contract.

    The whole source file is hashed deliberately: the formal evaluator shares
    the training sampler's private geometry helpers, so changing any of those
    helpers must invalidate a pending evaluator request even when the small
    public wrapper below did not change text.
    """

    try:
        with open(__file__, "rb") as stream:
            source_sha256 = hashlib.sha256(stream.read()).hexdigest()
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot bind frozen-evaluation sampler to its source bytes"
        ) from exc
    payload = {
        **_FROZEN_EVALUATION_PROPOSAL_SAMPLER_SEMANTICS,
        "sampling_schema_version": SCHEMA_VERSION,
        "arm_catalog_sha256": ARM_CATALOG_SHA256,
        "draws_per_birth": DRAWS_PER_BIRTH,
        "draws_per_sample": DRAWS_PER_SAMPLE,
        "implementation_source_sha256": source_sha256,
    }
    return {
        "payload": payload,
        "sha256": _sha256_json(payload),
    }


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


def _counter_rally_geometry_helper(objective: object):
    """Resolve the helper from the exact module that created the objective."""

    module = sys.modules.get(type(objective).__module__)
    helper = getattr(module, "counter_rally_reverse_ray_geometry", None)
    if not callable(helper):
        raise ValueError(
            "counter-rally objective module is missing the shared "
            "reverse-ray helper"
        )
    return helper


def _counter_rally_reverse_ray_geometry(
    *,
    contact_w_m: Vec3,
    incoming_direction_w: Vec3,
    landing_x_w_m: float,
    objective: object,
) -> Tuple[Vec2, Optional[str]]:
    """Call the canonical helper without suppressing the proposal receipt.

    The objective's defining module is already present because the strict
    manifest/profile loader created that exact object.  Resolving the pure
    helper from that module avoids both a copied formula and Python class
    identity problems in dependency-light spec-loader tests.
    """

    return _counter_rally_geometry_helper(objective)(
        contact_env_m=contact_w_m,
        return_direction_env_xy=(
            -incoming_direction_w[0],
            -incoming_direction_w[1],
        ),
        landing_depth_env_x_m=landing_x_w_m,
        profile=objective,
    )


def _validate_counter_rally_profile_support(
    profile: object,
    *,
    base_yaw_rad: Optional[float],
) -> None:
    """Validate static support and, when known, its actual-yaw cone."""

    objective = getattr(profile, "counter_rally_objective", None)
    if objective is None:
        return
    _counter_rally_geometry_helper(objective)
    if base_yaw_rad is not None:
        yaw = _finite(base_yaw_rad, name="base_yaw_rad")
        center_w = _rotate_yaw(
            getattr(profile, "incoming_direction_center_b_yaw"), yaw
        )
        center_horizontal_norm = math.hypot(center_w[0], center_w[1])
        if center_horizontal_norm <= 1.0e-12:
            raise ValueError(
                "counter-rally incoming center has no horizontal ray"
            )
        center_return_x = -center_w[0] / center_horizontal_norm
        center_error_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, center_return_x)))
        )
        support_radius_deg = math.hypot(
            max(
                getattr(
                    profile,
                    "incoming_direction_tangent_u_neg_max_deg",
                ),
                getattr(
                    profile,
                    "incoming_direction_tangent_u_pos_max_deg",
                ),
            ),
            max(
                getattr(
                    profile,
                    "incoming_direction_tangent_v_neg_max_deg",
                ),
                getattr(
                    profile,
                    "incoming_direction_tangent_v_pos_max_deg",
                ),
            ),
        )
        cone_half_angle_deg = math.degrees(
            math.acos(
                _finite(
                    getattr(
                        objective, "minimum_opponent_x_component"
                    ),
                    name="counter_rally.minimum_opponent_x_component",
                    minimum=0.0,
                    maximum=1.0,
                )
            )
        )
        if (
            center_error_deg + support_radius_deg
            > cone_half_angle_deg + UNIT_VECTOR_TOLERANCE
        ):
            raise ValueError(
                "counter-rally incoming-direction support leaves the "
                "opponent cone"
            )

    table_near_x = _finite(
        getattr(objective, "table_near_x_env_m"),
        name="counter_rally.table_near_x_env_m",
    )
    table_length = _finite(
        getattr(objective, "table_length_m"),
        name="counter_rally.table_length_m",
        minimum=0.0,
    )
    edge_margin = _finite(
        getattr(objective, "table_edge_margin_m"),
        name="counter_rally.table_edge_margin_m",
        minimum=0.0,
    )
    opponent_baseline_x = _finite(
        getattr(objective, "opponent_baseline_x_env_m"),
        name="counter_rally.opponent_baseline_x_env_m",
    )
    support_lo = getattr(profile, "landing_aim_min_w_xy_m")[0]
    support_hi = getattr(profile, "landing_aim_max_w_xy_m")[0]
    net_x = table_near_x + 0.5 * table_length
    if not (
        support_lo > net_x
        and support_hi <= opponent_baseline_x - edge_margin
    ):
        raise ValueError(
            "counter-rally landing-x support must lie on the bounded "
            "opponent half"
        )
    incoming_speed_lo = getattr(profile, "incoming_speed_min_mps")
    incoming_speed_hi = getattr(profile, "incoming_speed_max_mps")
    venue_speed_lo = getattr(
        objective, "minimum_supported_ball_speed_mps"
    )
    venue_speed_hi = getattr(
        objective, "maximum_supported_ball_speed_mps"
    )
    speed_ratio = getattr(objective, "target_baseline_speed_ratio")
    if not (
        venue_speed_lo
        <= incoming_speed_lo
        <= incoming_speed_hi
        <= venue_speed_hi
        and venue_speed_lo
        <= speed_ratio * incoming_speed_lo
        <= speed_ratio * incoming_speed_hi
        <= venue_speed_hi
    ):
        raise ValueError(
            "counter-rally incoming/target speed support leaves venue "
            "bounds"
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
class SamplingMixture:
    """Deterministic center/interior/frontier sampling recipe.

    The integer slots form an exact repeating quota rather than an IID
    categorical draw.  The default ``1/3/1`` recipe is therefore exactly
    ``20/60/20`` in every complete five-sample block and gives every stratum a
    finite starvation bound.  ``interior_level_scale`` is joint ``rho`` over
    the complete physical support width (including initial width), not a
    multiplier on the normalized curriculum level.  The center stratum always
    uses the profile's non-degenerate level-zero support.

    This object is opt-in at :class:`ActionBallSampler` construction so older
    center/marginal callers do not silently change distribution.  A joint
    curriculum launch must pass ``SamplingMixture()`` explicitly.
    """

    center_slots: int = 1
    interior_slots: int = 3
    frontier_slots: int = 1
    interior_level_scale: float = 0.8
    frontier_band_fraction: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "center_slots",
            "interior_slots",
            "frontier_slots",
        ):
            value = _plain_int(
                getattr(self, name),
                name=f"sampling_mixture.{name}",
                minimum=1,
                maximum=1000,
            )
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "interior_level_scale",
            _finite(
                self.interior_level_scale,
                name="sampling_mixture.interior_level_scale",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        band = _finite(
            self.frontier_band_fraction,
            name="sampling_mixture.frontier_band_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if band <= 0.0 or band >= 1.0:
            raise ValueError(
                "sampling_mixture.frontier_band_fraction must be in "
                "(0, 1)"
            )
        if self.interior_level_scale > 1.0 - band + 1.0e-15:
            raise ValueError(
                "sampling_mixture.interior_level_scale must be <= "
                "1 - frontier_band_fraction so interior support cannot "
                "overlap the frontier band"
            )
        object.__setattr__(self, "frontier_band_fraction", band)

    @property
    def cycle_length(self) -> int:
        return (
            self.center_slots
            + self.interior_slots
            + self.frontier_slots
        )

    @property
    def schedule(self) -> Tuple[str, ...]:
        """Return a deterministic, evenly interleaved quota cycle."""

        weights = (
            self.center_slots,
            self.interior_slots,
            self.frontier_slots,
        )
        names = ("center", "interior", "frontier")
        current = [0, 0, 0]
        schedule = []
        for _ in range(self.cycle_length):
            for index, weight in enumerate(weights):
                current[index] += weight
            chosen = max(
                range(len(names)),
                key=lambda index: (current[index], -index),
            )
            schedule.append(names[chosen])
            current[chosen] -= self.cycle_length
        result = tuple(schedule)
        if tuple(result.count(name) for name in names) != weights:
            raise AssertionError("sampling mixture schedule lost a quota")
        return result

    def stratum_for(self, proposal_index: int) -> str:
        """Return the quota stratum for one independent proposal cursor.

        Birth and swing cursors both use this schedule, but each action owns
        separate ``birth_count`` and ``sample_count`` cursors.  In particular,
        rejected birth proposals still advance only the birth cursor.
        """

        proposal_index = _plain_int(
            proposal_index, name="proposal_index"
        )
        return self.schedule[proposal_index % self.cycle_length]

    def frontier_ordinal_before(self, proposal_index: int) -> int:
        """Count frontier slots preceding one proposal cursor exactly."""

        proposal_index = _plain_int(
            proposal_index, name="proposal_index"
        )
        cycles, offset = divmod(proposal_index, self.cycle_length)
        return (
            cycles * self.frontier_slots
            + self.schedule[:offset].count("frontier")
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "center_slots": self.center_slots,
            "interior_slots": self.interior_slots,
            "frontier_slots": self.frontier_slots,
            "interior_level_scale": self.interior_level_scale,
            "frontier_band_fraction": self.frontier_band_fraction,
            "schedule": list(self.schedule),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SamplingMixture":
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
            name="sampling mixture",
        )
        result = cls(
            center_slots=row["center_slots"],
            interior_slots=row["interior_slots"],
            frontier_slots=row["frontier_slots"],
            interior_level_scale=row["interior_level_scale"],
            frontier_band_fraction=row["frontier_band_fraction"],
        )
        declared_schedule = row["schedule"]
        if (
            not isinstance(declared_schedule, (tuple, list))
            or tuple(declared_schedule) != result.schedule
        ):
            raise ValueError("sampling mixture schedule mismatch")
        return result

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
    counter_rally_objective: Optional[object] = None

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
        # base_spawn 的 z 不是课程轴,但它承载该动作 canonical-ready 的 root Z(由
        # runtime 传入 adapter),所以 center/min/max 必须是同一个常数、std 全零;
        # 独立造 profile(z=0)仍然合法。base_travel 是平面位移,z 必须严格为零。
        spawn_z = self.base_spawn_center_w_m[2]
        if not math.isfinite(spawn_z):
            raise ValueError("base_spawn_center_w_m z must be finite")
        for name in ("base_spawn_min_w_m", "base_spawn_max_w_m"):
            if getattr(self, name)[2] != spawn_z:
                raise ValueError(
                    f"{name} z must equal base_spawn_center_w_m z exactly "
                    "(base spawn z is one constant, not a curriculum axis)"
                )
        for name in (
            "base_spawn_std_lower_initial_m",
            "base_spawn_std_lower_max_m",
            "base_spawn_std_upper_initial_m",
            "base_spawn_std_upper_max_m",
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
        if self.counter_rally_objective is not None:
            objective = self.counter_rally_objective
            if (
                getattr(objective, "mode", None) != "counter_rally_v1"
                or not isinstance(getattr(objective, "sha256", None), str)
                or len(objective.sha256) != 64
            ):
                raise ValueError(
                    "counter_rally_objective must be a validated "
                    "CounterRallyObjectiveProfile"
                )
            _validate_counter_rally_profile_support(
                self, base_yaw_rad=None
            )

    def as_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "counter_rally_objective":
                if value is not None:
                    result[field.name] = value.to_mapping()
                continue
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
    sampling_mixture: Optional[SamplingMixture] = None
    sampling_stratum: str = "domain"
    sampling_levels: DomainLevels = field(default_factory=DomainLevels)
    frontier_arm: Optional[str] = None

    def to_state_dict(self) -> Dict[str, object]:
        """Canonical flat checkpoint row used by deterministic replay."""

        payload = {
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
        if self.sampling_mixture is None:
            return payload
        return {
            **{
                key: payload[key]
                for key in _BIRTH_STATE_KEYS[:6]
            },
            "sampling_mixture": self.sampling_mixture.as_dict(),
            "sampling_stratum": self.sampling_stratum,
            "sampling_levels": self.sampling_levels.as_dict(),
            "frontier_arm": self.frontier_arm,
            **{
                key: payload[key]
                for key in _BIRTH_STATE_KEYS[6:]
            },
        }

    def to_identity_receipt(self) -> Dict[str, object]:
        """Flat strict receipt accepted by ``assert_issued_birth``."""

        return self.to_state_dict()

    @classmethod
    def from_identity_receipt(
        cls,
        value: object,
    ) -> "BaseBirthReceipt":
        if not isinstance(value, Mapping):
            raise ValueError("birth identity receipt must be a mapping")
        has_mixture = "sampling_mixture" in value
        row = _exact_mapping(
            value,
            (
                _MIXTURE_BIRTH_STATE_KEYS
                if has_mixture
                else _BIRTH_STATE_KEYS
            ),
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
        if has_mixture:
            mixture = SamplingMixture.from_mapping(
                row["sampling_mixture"]
            )
            stratum = row["sampling_stratum"]
            if stratum not in ("center", "interior", "frontier"):
                raise ValueError("birth.sampling_stratum is invalid")
            sampling_levels = DomainLevels.from_mapping(
                row["sampling_levels"]
            )
            raw_frontier_arm = row["frontier_arm"]
            if raw_frontier_arm is not None and (
                type(raw_frontier_arm) is not str
                or raw_frontier_arm
                not in (
                    "base_spawn_x_lower",
                    "base_spawn_x_upper",
                    "base_spawn_y_lower",
                    "base_spawn_y_upper",
                )
            ):
                raise ValueError("birth.frontier_arm is invalid")
            if (stratum == "frontier") != (
                raw_frontier_arm is not None
            ):
                raise ValueError(
                    "birth.frontier_arm must be present exactly for "
                    "frontier"
                )
        else:
            mixture = None
            stratum = "domain"
            sampling_levels = DomainLevels()
            raw_frontier_arm = None
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
            sampling_mixture=mixture,
            sampling_stratum=stratum,
            sampling_levels=sampling_levels,
            frontier_arm=raw_frontier_arm,
        )
        if result.draw_end - result.draw_start != DRAWS_PER_BIRTH:
            raise ValueError("birth receipt has invalid draw range")
        if result.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("birth arm catalog hash mismatch")
        if (
            result.sampling_mixture is not None
            and result.sampling_stratum
            != result.sampling_mixture.stratum_for(result.birth_index)
        ):
            raise ValueError(
                "birth sampling_stratum disagrees with mixture schedule"
            )
        if result.sampling_mixture is not None:
            for arm in ARM_KEYS:
                effective = getattr(result.sampling_levels, arm)
                if arm not in _BASE_SPAWN_ARMS and effective != 0.0:
                    raise ValueError(
                        "birth sampling_levels contain a swing-only arm"
                    )
                if effective > getattr(result.domain_levels, arm) + 1.0e-15:
                    raise ValueError(
                        "birth sampling_levels exceed domain levels"
                    )
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
            sampling_mixture=result.sampling_mixture,
            sampling_stratum=result.sampling_stratum,
            sampling_levels=result.sampling_levels,
            frontier_arm=result.frontier_arm,
        )
        if result.birth_id != _sha256_json(payload):
            raise ValueError("birth_id does not match canonical identity")
        return result

    def to_receipt(self) -> Dict[str, object]:
        receipt = {
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
        if self.sampling_mixture is not None:
            receipt["sampling"] = {
                "mixture": self.sampling_mixture.as_dict(),
                "stratum": self.sampling_stratum,
                "effective_levels": self.sampling_levels.as_dict(),
                "frontier_arm": self.frontier_arm,
            }
        return receipt


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
    birth_index: int
    birth_sampling_stratum: str
    birth_sampling_levels: DomainLevels
    birth_frontier_arm: Optional[str]
    sampling_mixture: Optional[SamplingMixture]
    sampling_stratum: str
    sampling_levels: DomainLevels
    frontier_arm: Optional[str]
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
    contact_time_step_s: Optional[float]
    time_to_contact_tick: Optional[int]
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

        payload = {
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
        if self.sampling_mixture is None:
            # Preserve the exact pre-mixture sample SHA for legacy
            # center/marginal/runtime consumers.  The opt-in joint mixture is
            # a distinct sampler contract and extends the identity below.
            return payload
        ordered = {}
        for key in _MIXTURE_SAMPLE_IDENTITY_KEYS:
            if key == "birth_index":
                ordered[key] = self.birth_index
            elif key == "birth_sampling_stratum":
                ordered[key] = self.birth_sampling_stratum
            elif key == "birth_sampling_levels":
                ordered[key] = self.birth_sampling_levels.as_dict()
            elif key == "birth_frontier_arm":
                ordered[key] = self.birth_frontier_arm
            elif key == "sampling_mixture":
                ordered[key] = self.sampling_mixture.as_dict()
            elif key == "sampling_stratum":
                ordered[key] = self.sampling_stratum
            elif key == "sampling_levels":
                ordered[key] = self.sampling_levels.as_dict()
            elif key == "frontier_arm":
                ordered[key] = self.frontier_arm
            elif key == "contact_time_step_s":
                ordered[key] = self.contact_time_step_s
            elif key == "time_to_contact_tick":
                ordered[key] = self.time_to_contact_tick
            else:
                ordered[key] = payload[key]
        return ordered

    def verify_sample_id(self) -> None:
        """Fail closed if any identity field no longer matches ``sample_id``."""

        if (
            type(self.sample_id) is not str
            or self.sample_id != _sha256_json(self.identity_payload())
        ):
            raise ValueError("sample_id does not match canonical identity")

    @property
    def executed_planar_base_travel_distance_m(self) -> float:
        """Return actual planar travel without extending sample identity.

        ``no_move`` deliberately records a latent travel draw for exact RNG
        parity, but does not execute it.  Keeping this as a derived property
        lets the move-preparation admission proof consume the right distance
        while preserving every legacy sample payload and SHA byte-for-byte.
        """

        if self.mobility_mode == "no_move":
            return 0.0
        return math.hypot(
            self.base_travel_latent_b_yaw_m[0],
            self.base_travel_latent_b_yaw_m[1],
        )

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
        if not isinstance(value, Mapping):
            raise ValueError("sample identity receipt must be a mapping")
        has_mixture = "sampling_mixture" in value
        row = _exact_mapping(
            value,
            (
                "sample_id",
                *(
                    _MIXTURE_SAMPLE_IDENTITY_KEYS
                    if has_mixture
                    else _SAMPLE_IDENTITY_KEYS
                ),
            ),
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
        if has_mixture:
            mixture = SamplingMixture.from_mapping(
                row["sampling_mixture"]
            )
            birth_index = _plain_int(
                row["birth_index"], name="sample.birth_index"
            )
            birth_stratum = row["birth_sampling_stratum"]
            if birth_stratum not in (
                "center",
                "interior",
                "frontier",
            ):
                raise ValueError(
                    "sample.birth_sampling_stratum is invalid"
                )
            if birth_stratum != mixture.stratum_for(birth_index):
                raise ValueError(
                    "sample birth_sampling_stratum disagrees with "
                    "mixture birth schedule"
                )
            birth_sampling_levels = DomainLevels.from_mapping(
                row["birth_sampling_levels"]
            )
            raw_birth_frontier_arm = row["birth_frontier_arm"]
            if raw_birth_frontier_arm is not None and (
                type(raw_birth_frontier_arm) is not str
                or raw_birth_frontier_arm not in _BASE_SPAWN_ARMS
            ):
                raise ValueError(
                    "sample.birth_frontier_arm is invalid"
                )
            if (birth_stratum == "frontier") != (
                raw_birth_frontier_arm is not None
            ):
                raise ValueError(
                    "sample.birth_frontier_arm must be present exactly "
                    "for frontier"
                )
            stratum = row["sampling_stratum"]
            if stratum not in ("center", "interior", "frontier"):
                raise ValueError("sample.sampling_stratum is invalid")
            sampling_levels = DomainLevels.from_mapping(
                row["sampling_levels"]
            )
            raw_frontier_arm = row["frontier_arm"]
            if raw_frontier_arm is not None and (
                type(raw_frontier_arm) is not str
                or raw_frontier_arm not in ARM_KEYS
            ):
                raise ValueError("sample.frontier_arm is invalid")
            if (stratum == "frontier") != (
                raw_frontier_arm is not None
            ):
                raise ValueError(
                    "sample.frontier_arm must be present exactly for "
                    "frontier"
                )
            expected_stratum = mixture.stratum_for(
                _plain_int(
                    row["sample_index"],
                    name="sample.sample_index",
                )
            )
            if stratum != expected_stratum:
                raise ValueError(
                    "sample sampling_stratum disagrees with mixture schedule"
                )
            # Exact physical-rho inversion requires the sampler's pinned
            # profile.  Public coercion can still reject impossible level
            # escalation; ``assert_issued_sample`` later recomputes the exact
            # profile-aware plan and deterministic draw.
            for arm in ARM_KEYS:
                if getattr(sampling_levels, arm) > (
                    getattr(levels, arm) + 1.0e-15
                ):
                    raise ValueError(
                        "sample sampling_levels exceed domain levels"
                    )
            raw_contact_step = row["contact_time_step_s"]
            raw_contact_tick = row["time_to_contact_tick"]
            if raw_contact_step is None:
                if raw_contact_tick is not None:
                    raise ValueError(
                        "sample time_to_contact_tick requires "
                        "contact_time_step_s"
                    )
                contact_time_step_s = None
                time_to_contact_tick = None
            else:
                contact_time_step_s = _finite(
                    raw_contact_step,
                    name="sample.contact_time_step_s",
                    minimum=0.0,
                )
                if contact_time_step_s <= 0.0:
                    raise ValueError(
                        "sample.contact_time_step_s must be > 0"
                    )
                time_to_contact_tick = _plain_int(
                    raw_contact_tick,
                    name="sample.time_to_contact_tick",
                    minimum=1,
                )
        else:
            mixture = None
            birth_index = -1
            birth_stratum = "domain"
            birth_sampling_levels = levels
            raw_birth_frontier_arm = None
            stratum = "domain"
            sampling_levels = levels
            raw_frontier_arm = None
            contact_time_step_s = None
            time_to_contact_tick = None
        mode = row["mobility_mode"]
        if mode not in ("no_move", "move"):
            raise ValueError(
                "sample.mobility_mode must be 'no_move' or 'move'"
            )
        if has_mixture:
            for arm in ARM_KEYS:
                birth_effective = getattr(
                    birth_sampling_levels, arm
                )
                if (
                    arm not in _BASE_SPAWN_ARMS
                    and birth_effective != 0.0
                ):
                    raise ValueError(
                        "sample birth_sampling_levels contain a "
                        "swing-only arm"
                    )
                if birth_effective > getattr(levels, arm) + 1.0e-15:
                    raise ValueError(
                        "sample birth_sampling_levels exceed domain "
                        "levels"
                    )
            if any(
                getattr(sampling_levels, arm) != 0.0
                for arm in _BASE_SPAWN_ARMS
            ):
                raise ValueError(
                    "sample sampling_levels contain a base-birth arm"
                )
            if (
                mode == "no_move"
                and any(
                    getattr(sampling_levels, arm) != 0.0
                    for arm in (
                        "base_travel_x_lower",
                        "base_travel_x_upper",
                        "base_travel_y_lower",
                        "base_travel_y_upper",
                    )
                )
            ):
                raise ValueError(
                    "no_move sample sampling_levels contain base travel"
                )
            if raw_frontier_arm in _BASE_SPAWN_ARMS:
                raise ValueError(
                    "sample frontier_arm cannot be a base-birth arm"
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
            birth_index=birth_index,
            birth_sampling_stratum=birth_stratum,
            birth_sampling_levels=birth_sampling_levels,
            birth_frontier_arm=raw_birth_frontier_arm,
            sampling_mixture=mixture,
            sampling_stratum=stratum,
            sampling_levels=sampling_levels,
            frontier_arm=raw_frontier_arm,
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
            contact_time_step_s=contact_time_step_s,
            time_to_contact_tick=time_to_contact_tick,
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
        if (
            result.contact_time_step_s is not None
            and result.time_to_contact_s
            != result.time_to_contact_tick * result.contact_time_step_s
        ):
            raise ValueError(
                "sample time_to_contact_s is not its exact policy tick"
            )
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
            **(
                {
                    "sampling": {
                        "mixture": self.sampling_mixture.as_dict(),
                        "stratum": self.sampling_stratum,
                        "effective_levels": (
                            self.sampling_levels.as_dict()
                        ),
                        "frontier_arm": self.frontier_arm,
                        "birth": {
                            "birth_index": self.birth_index,
                            "stratum": self.birth_sampling_stratum,
                            "effective_levels": (
                                self.birth_sampling_levels.as_dict()
                            ),
                            "frontier_arm": self.birth_frontier_arm,
                        },
                    }
                }
                if self.sampling_mixture is not None
                else {}
            ),
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
                **(
                    {
                        "contact_time_step_s": (
                            self.contact_time_step_s
                        ),
                        "time_to_contact_tick": (
                            self.time_to_contact_tick
                        ),
                    }
                    if self.sampling_mixture is not None
                    else {}
                ),
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


@dataclass(frozen=True)
class FrozenEvaluationProposal:
    """One stateless, authority-indexed birth and incoming-ball proposal."""

    proposal_sampler_contract_sha256: str
    proposal_receipt_sha256: str
    evaluation_seed: int
    external_sample_index: int
    external_birth_index: int
    action_uid: int
    profile_sha256: str
    domain_epoch: int
    domain_levels: DomainLevels
    rho: float
    sampling_stratum: str
    selected_arm: Optional[str]
    birth_component_stratum: str
    ball_task_component_stratum: str
    base_yaw_rad: float
    policy_dt_s: float
    birth: BaseBirthReceipt
    sample: BallBaseSample

    def receipt_payload(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "action_ball_frozen_evaluation_proposal",
            "proposal_sampler_contract_sha256": (
                self.proposal_sampler_contract_sha256
            ),
            "evaluation_seed": self.evaluation_seed,
            "external_sample_index": self.external_sample_index,
            "external_birth_index": self.external_birth_index,
            "action_uid": self.action_uid,
            "profile_sha256": self.profile_sha256,
            "domain_epoch": self.domain_epoch,
            "domain_levels": self.domain_levels.as_dict(),
            "rho": self.rho,
            "sampling_stratum": self.sampling_stratum,
            "selected_arm": self.selected_arm,
            "component_strata": {
                "birth": self.birth_component_stratum,
                "ball_task": self.ball_task_component_stratum,
            },
            "base_yaw_rad": self.base_yaw_rad,
            "policy_dt_s": self.policy_dt_s,
            "birth_receipt_sha256": self.birth.birth_id,
            "sample_receipt_sha256": self.sample.sample_id,
        }

    def verify(self) -> None:
        contract = frozen_evaluation_proposal_sampler_contract()
        if (
            self.proposal_sampler_contract_sha256
            != contract["sha256"]
        ):
            raise ValueError(
                "frozen proposal sampler contract differs from live code"
            )
        _plain_int(self.evaluation_seed, name="evaluation_seed")
        _plain_int(
            self.external_sample_index, name="external_sample_index"
        )
        _plain_int(
            self.external_birth_index, name="external_birth_index"
        )
        if self.action_uid != self.sample.action_uid:
            raise ValueError("frozen proposal sample changed action identity")
        if self.action_uid != self.birth.action_uid:
            raise ValueError("frozen proposal birth changed action identity")
        if self.profile_sha256 != self.sample.profile_sha256:
            raise ValueError("frozen proposal sample changed profile")
        if self.profile_sha256 != self.birth.profile_sha256:
            raise ValueError("frozen proposal birth changed profile")
        if (
            self.external_sample_index != self.sample.sample_index
            or self.external_birth_index != self.birth.birth_index
        ):
            raise ValueError(
                "frozen proposal receipts changed authority indices"
            )
        if (
            self.domain_epoch != self.sample.domain_epoch
            or self.domain_epoch != self.birth.domain_epoch
            or self.domain_levels != self.sample.domain_levels
            or self.domain_levels != self.birth.domain_levels
        ):
            raise ValueError(
                "frozen proposal receipts changed the frozen domain"
            )
        if (
            self.sample.birth_id != self.birth.birth_id
            or self.sample.base_start_w_m != self.birth.base_start_w_m
        ):
            raise ValueError(
                "frozen proposal sample is detached from its birth"
            )
        if self.sampling_stratum not in (
            "center",
            "interior",
            "frontier",
        ):
            raise ValueError("frozen proposal sampling stratum is invalid")
        if (self.sampling_stratum == "frontier") != (
            self.selected_arm is not None
        ):
            raise ValueError(
                "frozen proposal frontier arm/stratum mismatch"
            )
        if (
            self.selected_arm is not None
            and self.selected_arm not in ARM_KEYS
        ):
            raise ValueError("frozen proposal selected arm is unknown")
        if (
            self.birth_component_stratum
            != self.birth.sampling_stratum
            or self.ball_task_component_stratum
            != self.sample.sampling_stratum
        ):
            raise ValueError(
                "frozen proposal component strata differ from receipts"
            )
        if self.sampling_stratum == "frontier":
            selected_is_birth = self.selected_arm in _BASE_SPAWN_ARMS
            expected = (
                ("frontier", "center")
                if selected_is_birth
                else ("center", "frontier")
            )
            if (
                self.birth_component_stratum,
                self.ball_task_component_stratum,
            ) != expected:
                raise ValueError(
                    "frozen frontier must belong to exactly one component"
                )
            for arm in ARM_KEYS:
                expected_level = (
                    getattr(self.domain_levels, arm)
                    if arm == self.selected_arm
                    else 0.0
                )
                actual_level = (
                    getattr(self.birth.sampling_levels, arm)
                    if arm in _BASE_SPAWN_ARMS
                    else getattr(self.sample.sampling_levels, arm)
                )
                if actual_level != expected_level:
                    raise ValueError(
                        "frozen frontier changed an unselected arm"
                    )
        else:
            if (
                self.birth_component_stratum,
                self.ball_task_component_stratum,
            ) != (self.sampling_stratum, self.sampling_stratum):
                raise ValueError(
                    "non-frontier component strata must equal outer stratum"
                )
        _finite(self.rho, name="rho", minimum=0.0, maximum=1.0)
        step = _finite(
            self.policy_dt_s, name="policy_dt_s", minimum=0.0
        )
        if step <= 0.0:
            raise ValueError("policy_dt_s must be > 0")
        self.sample.verify_sample_id()
        if (
            type(self.proposal_receipt_sha256) is not str
            or self.proposal_receipt_sha256
            != _sha256_json(self.receipt_payload())
        ):
            raise ValueError(
                "frozen proposal receipt SHA does not match identity"
            )


def _direction_tangent_coordinates(
    direction: Vec3,
    *,
    center: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
) -> Tuple[float, float]:
    """Invert ``_direction_from_tangent_angles`` on its certified support."""

    cosine = max(-1.0, min(1.0, _dot(direction, center)))
    radius = math.acos(cosine)
    if radius <= 1.0e-15:
        return (0.0, 0.0)
    sine = math.sin(radius)
    if abs(sine) <= 1.0e-15:
        raise ValueError(
            "sampled direction is singular in the tangent chart"
        )
    scale = radius / sine
    return (
        _dot(direction, tangent_u) * scale,
        _dot(direction, tangent_v) * scale,
    )


def _frontier_coordinate_delta(
    sample: BallBaseSample,
    profile: SamplingProfile,
    arm: str,
) -> float:
    """Recover the selected signed coordinate from receipt-visible values."""

    if arm.startswith("time_to_contact_"):
        return (
            sample.time_to_contact_s
            - profile.time_to_contact_center_s
        )
    if arm.startswith("contact_"):
        _, axis, _ = arm.split("_")
        index = {"x": 0, "y": 1, "z": 2}[axis]
        return (
            sample.contact_offset_from_base_goal_b_yaw_m[index]
            - profile.contact_offset_center_b_yaw_m[index]
        )
    if arm.startswith("incoming_speed_"):
        return (
            sample.incoming_speed_mps
            - profile.incoming_speed_center_mps
        )
    if arm.startswith("spin_magnitude_"):
        return (
            sample.spin_magnitude_radps
            - profile.spin_magnitude_center_radps
        )
    if arm.startswith("base_travel_"):
        _, _, axis, _ = arm.split("_")
        index = {"x": 0, "y": 1}[axis]
        return (
            sample.base_travel_latent_b_yaw_m[index]
            - profile.base_travel_center_b_yaw_m[index]
        )
    if arm.startswith("landing_aim_"):
        _, _, axis, _ = arm.split("_")
        index = {"x": 0, "y": 1}[axis]
        return (
            sample.landing_aim_w_xy_m[index]
            - profile.landing_aim_center_w_xy_m[index]
        )
    if arm.startswith("incoming_direction_"):
        _, _, axis, _ = arm.split("_")
        coordinates = _direction_tangent_coordinates(
            sample.incoming_direction_b_yaw,
            center=profile.incoming_direction_center_b_yaw,
            tangent_u=profile.incoming_direction_tangent_u_b_yaw,
            tangent_v=profile.incoming_direction_tangent_v_b_yaw,
        )
        return coordinates[0 if axis == "u" else 1]
    if arm.startswith("spin_direction_"):
        _, _, axis, _ = arm.split("_")
        coordinates = _direction_tangent_coordinates(
            sample.spin_direction_b_yaw,
            center=profile.spin_direction_center_b_yaw,
            tangent_u=profile.spin_direction_tangent_u_b_yaw,
            tangent_v=profile.spin_direction_tangent_v_b_yaw,
        )
        return coordinates[0 if axis == "u" else 1]
    raise ValueError(
        f"{arm!r} is not a receipt-visible per-swing frontier arm"
    )


def _birth_frontier_coordinate_delta(
    birth: BaseBirthReceipt,
    profile: SamplingProfile,
    arm: str,
) -> float:
    """Recover one base-spawn axis from an issued birth receipt."""

    if arm not in _BASE_SPAWN_ARMS:
        raise ValueError(
            f"{arm!r} is not a base-birth frontier arm"
        )
    _, _, axis, _ = arm.split("_")
    index = {"x": 0, "y": 1}[axis]
    return (
        birth.base_start_w_m[index]
        - profile.base_spawn_center_w_m[index]
    )


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
    sampling_mixture: Optional[SamplingMixture] = None,
    sampling_stratum: str = "domain",
    sampling_levels: Optional[DomainLevels] = None,
    frontier_arm: Optional[str] = None,
) -> Dict[str, object]:
    payload = {
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
    if sampling_mixture is not None:
        if sampling_levels is None:
            raise ValueError(
                "mixture birth identity requires sampling_levels"
            )
        payload.update(
            {
                "sampling_mixture": sampling_mixture.as_dict(),
                "sampling_stratum": sampling_stratum,
                "sampling_levels": sampling_levels.as_dict(),
                "frontier_arm": frontier_arm,
            }
        )
    return payload


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

    def _uniform_open_many_diagnostic(
        self,
        request_digest: bytes,
        count: int,
        *,
        _authority: object,
    ) -> List[float]:
        """Return the exact scalar tape while hashing invariant bytes once.

        This is deliberately private to the diagnostic prevalidated sampler.
        Formal sampling keeps calling :meth:`uniform_open` one draw at a time.
        """

        if _authority is not _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY:
            raise RuntimeError(
                "diagnostic counter batch has no internal authority"
            )
        count = _plain_int(count, name="count")
        if count < 1:
            raise ValueError("count must be >= 1")
        if self.draw_count > INT64_MAX - count:
            raise OverflowError("random draw counter exhausted")
        prefix = hashlib.sha256()
        prefix.update(b"action-ball-sampling/counter/v1\0")
        prefix.update(self.seed.to_bytes(8, byteorder="big", signed=False))
        start = self.draw_count
        values = []
        for draw_count in range(start, start + count):
            digest = prefix.copy()
            digest.update(
                draw_count.to_bytes(8, byteorder="big", signed=False)
            )
            digest.update(request_digest)
            bits = int.from_bytes(digest.digest()[:8], "big")
            mantissa = bits >> 11
            values.append((float(mantissa) + 0.5) / _TWO_POW_53)
        self.draw_count = start + count
        return values


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


@dataclass(frozen=True)
class _ContactTimeTickGrid:
    """Native integer policy-tick support for one action profile."""

    step_s: float
    center_tick: int
    minimum_tick: int
    maximum_tick: int
    lower_initial_ticks: int
    lower_maximum_ticks: int
    upper_initial_ticks: int
    upper_maximum_ticks: int

    def width_ticks(
        self, levels: DomainLevels
    ) -> Tuple[int, int]:
        def _width(initial: int, maximum: int, level: float) -> int:
            promoted = initial + level * (maximum - initial)
            return min(
                maximum,
                max(initial, math.floor(promoted + 0.5)),
            )

        return (
            _width(
                self.lower_initial_ticks,
                self.lower_maximum_ticks,
                levels.time_to_contact_lower,
            ),
            _width(
                self.upper_initial_ticks,
                self.upper_maximum_ticks,
                levels.time_to_contact_upper,
            ),
        )

    def sample_tick(
        self,
        *,
        uniform: float,
        levels: DomainLevels,
        frontier_side: Optional[str],
        frontier_band_fraction: Optional[float],
    ) -> int:
        first, last = self.tick_bounds(
            levels=levels,
            frontier_side=frontier_side,
            frontier_band_fraction=frontier_band_fraction,
        )
        if first > last:
            raise ValueError(
                "time-to-contact frontier contains no policy tick"
            )
        count = last - first + 1
        offset = min(count - 1, math.floor(uniform * count))
        tick = first + offset
        if not self.minimum_tick <= tick <= self.maximum_tick:
            raise AssertionError(
                "time-to-contact integer sampler escaped certified bounds"
            )
        return tick

    def tick_bounds(
        self,
        *,
        levels: DomainLevels,
        frontier_side: Optional[str],
        frontier_band_fraction: Optional[float],
    ) -> Tuple[int, int]:
        """Return the closed native-tick set for one TTC stratum request."""

        lower, upper = self.width_ticks(levels)
        if frontier_side is None:
            return (
                self.center_tick - lower,
                self.center_tick + upper,
            )
        band = _finite(
            frontier_band_fraction,
            name="time_to_contact.frontier_band_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < band < 1.0:
            raise ValueError(
                "time-to-contact frontier band must lie in (0, 1)"
            )
        if frontier_side == "negative":
            outer_start = max(
                1, math.floor((1.0 - band) * lower) + 1
            )
            return (
                self.center_tick - lower,
                self.center_tick - outer_start,
            )
        if frontier_side == "positive":
            outer_start = max(
                1, math.floor((1.0 - band) * upper) + 1
            )
            return (
                self.center_tick + outer_start,
                self.center_tick + upper,
            )
        raise ValueError(
            "time-to-contact frontier side must be negative or positive"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "center_quantization": "nearest_tick_ties_up_then_clamp",
            "width_quantization": "ceil_seconds_minimum_one_tick",
            "step_s": self.step_s,
            "center_tick": self.center_tick,
            "center_s": self.center_tick * self.step_s,
            "minimum_tick": self.minimum_tick,
            "maximum_tick": self.maximum_tick,
            "lower_initial_ticks": self.lower_initial_ticks,
            "lower_maximum_ticks": self.lower_maximum_ticks,
            "upper_initial_ticks": self.upper_initial_ticks,
            "upper_maximum_ticks": self.upper_maximum_ticks,
        }


def _contact_time_tick_grid(
    profile: SamplingProfile,
    *,
    step_s: float,
    frontier_band_fraction: float,
) -> _ContactTimeTickGrid:
    """Quantize one profile once; no per-proposal continuous TTC exists."""

    step = _finite(
        step_s, name="contact_time_step_s", minimum=0.0
    )
    if step <= 0.0:
        raise ValueError("contact_time_step_s must be > 0")
    frontier_band = _finite(
        frontier_band_fraction,
        name="time_to_contact.frontier_band_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < frontier_band < 1.0:
        raise ValueError(
            "time-to-contact frontier band must lie in (0, 1)"
        )
    epsilon = 1.0e-12
    strict_reaction_floor_s = (
        profile.reference_t_hit_s / profile.teacher_rate_min
        + profile.reaction_margin_s
    )
    minimum_tick = max(
        math.ceil(profile.time_to_contact_min_s / step - epsilon),
        math.floor(strict_reaction_floor_s / step + epsilon) + 1,
    )
    maximum_wait_s = (
        profile.reference_t_hit_s / profile.teacher_rate_max + 1.0
    )
    maximum_tick = min(
        math.floor(profile.time_to_contact_max_s / step + epsilon),
        math.floor(maximum_wait_s / step + epsilon),
    )
    if minimum_tick > maximum_tick:
        raise ValueError(
            "time-to-contact profile has no tick satisfying strict "
            "reaction and one-second wait bounds"
        )
    center_tick = math.floor(
        profile.time_to_contact_center_s / step + 0.5
    )
    center_tick = min(max(center_tick, minimum_tick), maximum_tick)
    lower_capacity = center_tick - minimum_tick
    upper_capacity = maximum_tick - center_tick
    if lower_capacity < 1 or upper_capacity < 1:
        raise ValueError(
            "time-to-contact profile must admit at least one policy tick "
            "on each side of its quantized center"
        )

    def _tick_widths(
        initial_s: float,
        maximum_s: float,
        capacity: int,
        *,
        side: str,
    ) -> Tuple[int, int]:
        initial = min(
            capacity,
            max(1, math.ceil(initial_s / step - epsilon)),
        )
        maximum = min(
            capacity,
            max(initial, math.ceil(maximum_s / step - epsilon)),
        )
        if not 1 <= initial <= maximum <= capacity:
            raise ValueError(
                f"time-to-contact {side} tick widths are invalid"
            )
        return initial, maximum

    lower_initial, lower_maximum = _tick_widths(
        profile.time_to_contact_std_lower_initial_s,
        profile.time_to_contact_std_lower_max_s,
        lower_capacity,
        side="lower",
    )
    upper_initial, upper_maximum = _tick_widths(
        profile.time_to_contact_std_upper_initial_s,
        profile.time_to_contact_std_upper_max_s,
        upper_capacity,
        side="upper",
    )
    result = _ContactTimeTickGrid(
        step_s=step,
        center_tick=center_tick,
        minimum_tick=minimum_tick,
        maximum_tick=maximum_tick,
        lower_initial_ticks=lower_initial,
        lower_maximum_ticks=lower_maximum,
        upper_initial_ticks=upper_initial,
        upper_maximum_ticks=upper_maximum,
    )
    # Exhaust every curriculum level and both sides at construction.  This
    # proves center/interior/frontier are non-empty before a random tape can
    # advance; the actual proposal path merely indexes these certified sets.
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        levels = DomainLevels(
            time_to_contact_lower=level,
            time_to_contact_upper=level,
        )
        lower, upper = result.width_ticks(levels)
        if lower < 1 or upper < 1:
            raise AssertionError(
                "time-to-contact reachable level lost one side"
            )
        for side in ("negative", "positive"):
            result.sample_tick(
                uniform=0.5,
                levels=levels,
                frontier_side=side,
                frontier_band_fraction=frontier_band,
            )
    if not (
        result.minimum_tick * step > strict_reaction_floor_s
        and result.maximum_tick * step
        - profile.reference_t_hit_s / profile.teacher_rate_max
        <= 1.0 + 1.0e-12
    ):
        raise AssertionError(
            "time-to-contact tick grid violated certified timing bounds"
        )
    return result


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


_BASE_SPAWN_ARMS = (
    "base_spawn_x_lower",
    "base_spawn_x_upper",
    "base_spawn_y_lower",
    "base_spawn_y_upper",
)


def _arm_support_parameters(
    profile: SamplingProfile,
    arm: str,
) -> Tuple[float, float, float]:
    """Return selected-side ``(initial, maximum, hard_cap)`` widths.

    ``DomainLevels`` parameterize promotion between the first two values.
    Sampling clips the resulting physical support at ``hard_cap``.  Keeping
    these three quantities separate is necessary for joint ``rho``: scaling a
    normalized level scales only the promoted increment and can accidentally
    push the interior distribution into the outer frontier band whenever the
    initial support is non-zero.
    """

    if arm not in ARM_KEYS:
        raise ValueError(f"unknown support arm {arm!r}")
    axis_index = {"x": 0, "y": 1, "z": 2}
    aim_axis_index = {"x": 0, "y": 1}

    if arm.startswith("time_to_contact_"):
        if arm.endswith("_lower"):
            return (
                profile.time_to_contact_std_lower_initial_s,
                profile.time_to_contact_std_lower_max_s,
                profile.time_to_contact_center_s
                - profile.time_to_contact_min_s,
            )
        return (
            profile.time_to_contact_std_upper_initial_s,
            profile.time_to_contact_std_upper_max_s,
            profile.time_to_contact_max_s
            - profile.time_to_contact_center_s,
        )
    if arm.startswith("contact_"):
        _, axis, side = arm.split("_")
        index = axis_index[axis]
        if side == "lower":
            return (
                profile.contact_offset_std_lower_initial_m[index],
                profile.contact_offset_std_lower_max_m[index],
                profile.contact_offset_center_b_yaw_m[index]
                - profile.contact_offset_min_b_yaw_m[index],
            )
        return (
            profile.contact_offset_std_upper_initial_m[index],
            profile.contact_offset_std_upper_max_m[index],
            profile.contact_offset_max_b_yaw_m[index]
            - profile.contact_offset_center_b_yaw_m[index],
        )
    if arm.startswith("incoming_speed_"):
        if arm.endswith("_lower"):
            return (
                profile.incoming_speed_std_lower_initial_mps,
                profile.incoming_speed_std_lower_max_mps,
                profile.incoming_speed_center_mps
                - profile.incoming_speed_min_mps,
            )
        return (
            profile.incoming_speed_std_upper_initial_mps,
            profile.incoming_speed_std_upper_max_mps,
            profile.incoming_speed_max_mps
            - profile.incoming_speed_center_mps,
        )
    if arm.startswith("spin_magnitude_"):
        if arm.endswith("_lower"):
            return (
                profile.spin_magnitude_std_lower_initial_radps,
                profile.spin_magnitude_std_lower_max_radps,
                profile.spin_magnitude_center_radps
                - profile.spin_magnitude_min_radps,
            )
        return (
            profile.spin_magnitude_std_upper_initial_radps,
            profile.spin_magnitude_std_upper_max_radps,
            profile.spin_magnitude_max_radps
            - profile.spin_magnitude_center_radps,
        )
    for prefix in ("base_spawn", "base_travel"):
        if arm.startswith(prefix + "_"):
            _, _, axis, side = arm.split("_")
            index = axis_index[axis]
            if prefix == "base_spawn":
                center = profile.base_spawn_center_w_m
                lower_bound = profile.base_spawn_min_w_m
                upper_bound = profile.base_spawn_max_w_m
            else:
                center = profile.base_travel_center_b_yaw_m
                lower_bound = profile.base_travel_min_b_yaw_m
                upper_bound = profile.base_travel_max_b_yaw_m
            if side == "lower":
                return (
                    getattr(
                        profile, f"{prefix}_std_lower_initial_m"
                    )[index],
                    getattr(profile, f"{prefix}_std_lower_max_m")[
                        index
                    ],
                    center[index] - lower_bound[index],
                )
            return (
                getattr(profile, f"{prefix}_std_upper_initial_m")[
                    index
                ],
                getattr(profile, f"{prefix}_std_upper_max_m")[index],
                upper_bound[index] - center[index],
            )
    if arm.startswith("landing_aim_"):
        _, _, axis, side = arm.split("_")
        index = aim_axis_index[axis]
        if side == "lower":
            return (
                profile.landing_aim_std_lower_initial_m[index],
                profile.landing_aim_std_lower_max_m[index],
                profile.landing_aim_center_w_xy_m[index]
                - profile.landing_aim_min_w_xy_m[index],
            )
        return (
            profile.landing_aim_std_upper_initial_m[index],
            profile.landing_aim_std_upper_max_m[index],
            profile.landing_aim_max_w_xy_m[index]
            - profile.landing_aim_center_w_xy_m[index],
        )
    for prefix in ("incoming_direction", "spin_direction"):
        if arm.startswith(prefix + "_"):
            _, _, axis, side = arm.split("_")
            suffix = "neg" if side == "neg" else "pos"
            initial = math.radians(
                getattr(
                    profile,
                    f"{prefix}_tangent_{axis}_{suffix}_initial_deg",
                )
            )
            maximum = math.radians(
                getattr(
                    profile,
                    f"{prefix}_tangent_{axis}_{suffix}_max_deg",
                )
            )
            return (initial, maximum, maximum)
    raise AssertionError(f"unhandled support arm {arm!r}")


def _arm_physical_width(
    profile: SamplingProfile,
    levels: DomainLevels,
    arm: str,
) -> float:
    initial, maximum, hard_cap = _arm_support_parameters(
        profile, arm
    )
    return min(
        _lerp(initial, maximum, getattr(levels, arm)),
        hard_cap,
    )


def _rho_scaled_levels(
    profile: SamplingProfile,
    levels: DomainLevels,
    rho: float,
    *,
    active_arms: Sequence[str],
    full_width_arm: Optional[str] = None,
) -> DomainLevels:
    """Scale complete physical support widths, including level-zero width.

    A profile cannot represent support narrower than its initial width, so
    ``max(initial_width, rho * current_width)`` is used.  Inverting that
    physical target back to a normalized level keeps downstream samplers
    unchanged while making the mixture's ``rho`` interpretation exact.
    Inactive axes are zeroed so a swing receipt cannot claim a base-birth arm.
    """

    rho = _finite(rho, name="rho", minimum=0.0, maximum=1.0)
    active = tuple(active_arms)
    if len(set(active)) != len(active) or any(
        arm not in ARM_KEYS for arm in active
    ):
        raise ValueError("active_arms must be unique known arms")
    if full_width_arm is not None and full_width_arm not in active:
        raise ValueError("full_width_arm must be active")
    values = {arm: 0.0 for arm in ARM_KEYS}
    for arm in active:
        source_level = getattr(levels, arm)
        if arm == full_width_arm or rho == 1.0:
            values[arm] = source_level
            continue
        initial, maximum, hard_cap = _arm_support_parameters(
            profile, arm
        )
        initial_width = min(initial, hard_cap)
        current_width = min(
            _lerp(initial, maximum, source_level),
            hard_cap,
        )
        target_width = max(initial_width, rho * current_width)
        if (
            target_width <= initial_width
            or maximum <= initial
            or hard_cap <= initial
        ):
            values[arm] = 0.0
            continue
        effective_level = (
            (target_width - initial) / (maximum - initial)
        )
        values[arm] = min(
            source_level, max(0.0, effective_level)
        )
    return DomainLevels(**values)


def _frontier_width_pair(
    profile: SamplingProfile,
    levels: DomainLevels,
    arm: str,
) -> Tuple[float, float, str, int]:
    """Return negative/positive widths, selected side, and draw index.

    Widths are the exact support half-widths consumed by the sampler, not a
    caller-provided new-band boolean.  The returned draw index identifies the
    already-budgeted open uniform that can be remapped into the selected outer
    band without changing :data:`DRAWS_PER_SAMPLE`.
    """

    if arm not in ARM_KEYS:
        raise ValueError(f"unknown frontier arm {arm!r}")
    axis_index = {"x": 0, "y": 1, "z": 2}
    aim_axis_index = {"x": 0, "y": 1}

    if arm.startswith("time_to_contact_"):
        negative = _lerp(
            profile.time_to_contact_std_lower_initial_s,
            profile.time_to_contact_std_lower_max_s,
            levels.time_to_contact_lower,
        )
        positive = _lerp(
            profile.time_to_contact_std_upper_initial_s,
            profile.time_to_contact_std_upper_max_s,
            levels.time_to_contact_upper,
        )
        negative = min(
            negative,
            profile.time_to_contact_center_s
            - profile.time_to_contact_min_s,
        )
        positive = min(
            positive,
            profile.time_to_contact_max_s
            - profile.time_to_contact_center_s,
        )
        return (
            negative,
            positive,
            "negative" if arm.endswith("_lower") else "positive",
            9,
        )

    if arm.startswith("contact_"):
        _, axis, side = arm.split("_")
        index = axis_index[axis]
        negative = _lerp(
            profile.contact_offset_std_lower_initial_m[index],
            profile.contact_offset_std_lower_max_m[index],
            getattr(levels, f"contact_{axis}_lower"),
        )
        positive = _lerp(
            profile.contact_offset_std_upper_initial_m[index],
            profile.contact_offset_std_upper_max_m[index],
            getattr(levels, f"contact_{axis}_upper"),
        )
        negative = min(
            negative,
            profile.contact_offset_center_b_yaw_m[index]
            - profile.contact_offset_min_b_yaw_m[index],
        )
        positive = min(
            positive,
            profile.contact_offset_max_b_yaw_m[index]
            - profile.contact_offset_center_b_yaw_m[index],
        )
        return (
            negative,
            positive,
            "negative" if side == "lower" else "positive",
            6 + index,
        )

    if arm.startswith("incoming_speed_"):
        negative = _lerp(
            profile.incoming_speed_std_lower_initial_mps,
            profile.incoming_speed_std_lower_max_mps,
            levels.incoming_speed_lower,
        )
        positive = _lerp(
            profile.incoming_speed_std_upper_initial_mps,
            profile.incoming_speed_std_upper_max_mps,
            levels.incoming_speed_upper,
        )
        negative = min(
            negative,
            profile.incoming_speed_center_mps
            - profile.incoming_speed_min_mps,
        )
        positive = min(
            positive,
            profile.incoming_speed_max_mps
            - profile.incoming_speed_center_mps,
        )
        return (
            negative,
            positive,
            "negative" if arm.endswith("_lower") else "positive",
            10,
        )

    if arm.startswith("spin_magnitude_"):
        negative = _lerp(
            profile.spin_magnitude_std_lower_initial_radps,
            profile.spin_magnitude_std_lower_max_radps,
            levels.spin_magnitude_lower,
        )
        positive = _lerp(
            profile.spin_magnitude_std_upper_initial_radps,
            profile.spin_magnitude_std_upper_max_radps,
            levels.spin_magnitude_upper,
        )
        negative = min(
            negative,
            profile.spin_magnitude_center_radps
            - profile.spin_magnitude_min_radps,
        )
        positive = min(
            positive,
            profile.spin_magnitude_max_radps
            - profile.spin_magnitude_center_radps,
        )
        return (
            negative,
            positive,
            "negative" if arm.endswith("_lower") else "positive",
            13,
        )

    for prefix, draw_start in (
        ("base_spawn", 0),
        ("base_travel", 3),
    ):
        marker = prefix + "_"
        if arm.startswith(marker):
            _, _, axis, side = arm.split("_")
            index = axis_index[axis]
            center = getattr(profile, f"{prefix}_center_w_m", None)
            if prefix == "base_travel":
                center = profile.base_travel_center_b_yaw_m
                lower_bound = profile.base_travel_min_b_yaw_m
                upper_bound = profile.base_travel_max_b_yaw_m
            else:
                center = profile.base_spawn_center_w_m
                lower_bound = profile.base_spawn_min_w_m
                upper_bound = profile.base_spawn_max_w_m
            negative = _lerp(
                getattr(profile, f"{prefix}_std_lower_initial_m")[index],
                getattr(profile, f"{prefix}_std_lower_max_m")[index],
                getattr(levels, f"{prefix}_{axis}_lower"),
            )
            positive = _lerp(
                getattr(profile, f"{prefix}_std_upper_initial_m")[index],
                getattr(profile, f"{prefix}_std_upper_max_m")[index],
                getattr(levels, f"{prefix}_{axis}_upper"),
            )
            negative = min(negative, center[index] - lower_bound[index])
            positive = min(positive, upper_bound[index] - center[index])
            return (
                negative,
                positive,
                "negative" if side == "lower" else "positive",
                draw_start + index,
            )

    if arm.startswith("landing_aim_"):
        _, _, axis, side = arm.split("_")
        index = aim_axis_index[axis]
        negative = _lerp(
            profile.landing_aim_std_lower_initial_m[index],
            profile.landing_aim_std_lower_max_m[index],
            getattr(levels, f"landing_aim_{axis}_lower"),
        )
        positive = _lerp(
            profile.landing_aim_std_upper_initial_m[index],
            profile.landing_aim_std_upper_max_m[index],
            getattr(levels, f"landing_aim_{axis}_upper"),
        )
        negative = min(
            negative,
            profile.landing_aim_center_w_xy_m[index]
            - profile.landing_aim_min_w_xy_m[index],
        )
        positive = min(
            positive,
            profile.landing_aim_max_w_xy_m[index]
            - profile.landing_aim_center_w_xy_m[index],
        )
        return (
            negative,
            positive,
            "negative" if side == "lower" else "positive",
            16 + index,
        )

    for prefix, draw_start in (
        ("incoming_direction", 11),
        ("spin_direction", 14),
    ):
        marker = prefix + "_"
        if arm.startswith(marker):
            _, _, axis, side = arm.split("_")
            negative = math.radians(
                _lerp(
                    getattr(
                        profile,
                        f"{prefix}_tangent_{axis}_neg_initial_deg",
                    ),
                    getattr(
                        profile,
                        f"{prefix}_tangent_{axis}_neg_max_deg",
                    ),
                    getattr(levels, f"{prefix}_{axis}_neg"),
                )
            )
            positive = math.radians(
                _lerp(
                    getattr(
                        profile,
                        f"{prefix}_tangent_{axis}_pos_initial_deg",
                    ),
                    getattr(
                        profile,
                        f"{prefix}_tangent_{axis}_pos_max_deg",
                    ),
                    getattr(levels, f"{prefix}_{axis}_pos"),
                )
            )
            return (
                negative,
                positive,
                "negative" if side == "neg" else "positive",
                draw_start + (0 if axis == "u" else 1),
            )

    raise AssertionError(f"unhandled frontier arm {arm!r}")


def _eligible_swing_frontier_arms(
    profile: SamplingProfile,
    levels: DomainLevels,
    mixture: Optional[SamplingMixture] = None,
    contact_time_tick_grid: Optional[_ContactTimeTickGrid] = None,
) -> Tuple[str, ...]:
    """Return physically distinct per-swing frontier arms."""

    active = tuple(
        arm
        for arm in ARM_KEYS
        if not arm.startswith("base_spawn_")
        and not (
            profile.counter_rally_objective is not None
            and arm
            in profile.counter_rally_objective.inactive_curriculum_arms
        )
        and not (
            profile.mobility_mode == "no_move"
            and arm.startswith("base_travel_")
        )
    )
    if mixture is None:
        mixture = SamplingMixture()
    return _eligible_frontier_arms(
        profile=profile,
        levels=levels,
        mixture=mixture,
        active_arms=active,
        contact_time_tick_grid=contact_time_tick_grid,
    )


def _eligible_birth_frontier_arms(
    profile: SamplingProfile,
    levels: DomainLevels,
    mixture: SamplingMixture,
) -> Tuple[str, ...]:
    """Return base-spawn axis/side arms with a distinct outer band."""

    return _eligible_frontier_arms(
        profile=profile,
        levels=levels,
        mixture=mixture,
        active_arms=_BASE_SPAWN_ARMS,
    )


def _eligible_frontier_arms(
    *,
    profile: SamplingProfile,
    levels: DomainLevels,
    mixture: SamplingMixture,
    active_arms: Sequence[str],
    contact_time_tick_grid: Optional[_ContactTimeTickGrid] = None,
) -> Tuple[str, ...]:
    """Choose promoted frontiers, or the current support's outer band.

    Level-zero support is physical support, not a point.  An arm is therefore
    a promoted frontier only when its selected outer band begins beyond both
    center and joint-interior support.  When no arm is promoted (notably at
    level zero), the frontier stratum still has a valid meaning: the outer band
    of any non-zero side in the current physical support.  Preserve that
    stratum and its arm accounting instead of rejecting a legal initial
    domain.  A truly zero-width domain still fails closed.
    """

    promoted = []
    current_support = []
    interior = _rho_scaled_levels(
        profile,
        levels,
        mixture.interior_level_scale,
        active_arms=active_arms,
    )
    center = DomainLevels()
    frontier_start_scale = 1.0 - mixture.frontier_band_fraction
    tolerance = 1.0e-12
    for arm in active_arms:
        if (
            contact_time_tick_grid is not None
            and arm.startswith("time_to_contact_")
        ):
            lower_ticks, upper_ticks = (
                contact_time_tick_grid.width_ticks(levels)
            )
            center_lower_ticks, center_upper_ticks = (
                contact_time_tick_grid.width_ticks(center)
            )
            interior_lower_ticks, interior_upper_ticks = (
                contact_time_tick_grid.width_ticks(interior)
            )
            side_ticks = (
                lower_ticks
                if arm.endswith("_lower")
                else upper_ticks
            )
            if side_ticks >= 1:
                current_support.append(arm)
            center_side_ticks = (
                center_lower_ticks
                if arm.endswith("_lower")
                else center_upper_ticks
            )
            interior_side_ticks = (
                interior_lower_ticks
                if arm.endswith("_lower")
                else interior_upper_ticks
            )
            if side_ticks > max(
                center_side_ticks, interior_side_ticks
            ):
                promoted.append(arm)
            continue
        full_width = _arm_physical_width(profile, levels, arm)
        if full_width > tolerance:
            current_support.append(arm)
        center_width = _arm_physical_width(profile, center, arm)
        interior_width = _arm_physical_width(
            profile, interior, arm
        )
        frontier_start = frontier_start_scale * full_width
        if (
            full_width > center_width + tolerance
            and center_width <= frontier_start + tolerance
            and interior_width <= frontier_start + tolerance
        ):
            promoted.append(arm)
    return tuple(promoted if promoted else current_support)


def _frontier_band_uniform(
    uniform: float,
    *,
    negative_width: float,
    positive_width: float,
    side: str,
    band_fraction: float,
) -> float:
    """Remap one open uniform into an exact asymmetric outer-side band."""

    total = negative_width + positive_width
    if total <= 0.0:
        raise ValueError("frontier coordinate has zero total support")
    if side == "negative":
        if negative_width <= 0.0:
            raise ValueError("negative frontier arm has zero support")
        return uniform * band_fraction * negative_width / total
    if side == "positive":
        if positive_width <= 0.0:
            raise ValueError("positive frontier arm has zero support")
        return (
            negative_width
            + (1.0 - band_fraction) * positive_width
            + uniform * band_fraction * positive_width
        ) / total
    raise ValueError("frontier side must be negative or positive")


def _sampling_plan(
    *,
    profile: SamplingProfile,
    levels: DomainLevels,
    mixture: Optional[SamplingMixture],
    proposal_index: int,
    scope: str,
    contact_time_tick_grid: Optional[_ContactTimeTickGrid] = None,
) -> Tuple[str, DomainLevels, Optional[str]]:
    if mixture is None:
        return ("domain", levels, None)
    if scope == "birth":
        active_arms = _BASE_SPAWN_ARMS
        eligible = _eligible_birth_frontier_arms(
            profile, levels, mixture
        )
    elif scope == "swing":
        active_arms = tuple(
            arm
            for arm in ARM_KEYS
            if not arm.startswith("base_spawn_")
            and not (
                profile.counter_rally_objective is not None
                and arm
                in profile.counter_rally_objective.inactive_curriculum_arms
            )
            and not (
                profile.mobility_mode == "no_move"
                and arm.startswith("base_travel_")
            )
        )
        eligible = _eligible_swing_frontier_arms(
            profile,
            levels,
            mixture,
            contact_time_tick_grid,
        )
    else:
        raise ValueError("sampling scope must be 'birth' or 'swing'")
    stratum = mixture.stratum_for(proposal_index)
    if stratum == "center":
        return (
            "center",
            _rho_scaled_levels(
                profile,
                levels,
                0.0,
                active_arms=active_arms,
            ),
            None,
        )
    if stratum == "interior":
        return (
            "interior",
            _rho_scaled_levels(
                profile,
                levels,
                mixture.interior_level_scale,
                active_arms=active_arms,
            ),
            None,
        )
    if stratum != "frontier":
        raise AssertionError(f"unknown sampling stratum {stratum!r}")
    if not eligible:
        if scope == "swing":
            raise ValueError(
                "frontier stratum has no non-zero per-swing arm "
                "distinct from center/interior at the current physical "
                "support widths"
            )
        raise ValueError(
            "frontier stratum has no distinct base-birth arm at the "
            "current physical support widths"
        )
    ordinal = mixture.frontier_ordinal_before(proposal_index)
    arm = eligible[ordinal % len(eligible)]
    return (
        "frontier",
        _rho_scaled_levels(
            profile,
            levels,
            mixture.interior_level_scale,
            active_arms=active_arms,
            full_width_arm=arm,
        ),
        arm,
    )


def _validated_sampling_plan_override(
    *,
    profile: SamplingProfile,
    domain_levels: DomainLevels,
    mixture: Optional[SamplingMixture],
    scope: str,
    plan: Tuple[str, DomainLevels, Optional[str]],
) -> Tuple[str, DomainLevels, Optional[str]]:
    """Validate one evaluator-owned plan before any counter draw."""

    if mixture is None:
        raise ValueError(
            "sampling plan override requires the production mixture"
        )
    if (
        not isinstance(plan, tuple)
        or len(plan) != 3
        or plan[0] not in ("center", "interior", "frontier")
        or not isinstance(plan[1], DomainLevels)
    ):
        raise ValueError("sampling plan override is malformed")
    stratum, effective, arm = plan
    for name in ARM_KEYS:
        if getattr(effective, name) > (
            getattr(domain_levels, name) + 1.0e-15
        ):
            raise ValueError(
                "sampling plan override exceeds the frozen domain"
            )
    if scope == "birth":
        active = set(_BASE_SPAWN_ARMS)
    elif scope == "swing":
        active = {
            name
            for name in ARM_KEYS
            if name not in _BASE_SPAWN_ARMS
            and not (
                profile.mobility_mode == "no_move"
                and name.startswith("base_travel_")
            )
        }
    else:
        raise ValueError("sampling scope must be 'birth' or 'swing'")
    for name in ARM_KEYS:
        if name not in active and getattr(effective, name) != 0.0:
            raise ValueError(
                "sampling plan override activates an out-of-scope arm"
            )
    if stratum == "frontier":
        if arm not in active:
            raise ValueError(
                "frontier sampling override names an out-of-scope arm"
            )
        if _arm_physical_width(profile, effective, arm) <= 0.0:
            raise ValueError(
                "frontier sampling override has zero physical width"
            )
    elif arm is not None:
        raise ValueError(
            "non-frontier sampling override cannot name an arm"
        )
    return plan


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
        sampling_mixture: Optional[SamplingMixture] = None,
        contact_time_step_s: Optional[float] = None,
        diagnostic_unauthorized: bool = False,
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
        if type(diagnostic_unauthorized) is not bool:
            raise TypeError(
                "diagnostic_unauthorized must be an exact boolean"
            )
        # Runtime bookkeeping only: deliberately excluded from the sampler
        # identity so the same seed/request tape emits byte-identical samples.
        self._diagnostic_fast_path = diagnostic_unauthorized
        if (
            sampling_mixture is not None
            and not isinstance(sampling_mixture, SamplingMixture)
        ):
            raise TypeError(
                "sampling_mixture must be SamplingMixture or None"
            )
        self._sampling_mixture = sampling_mixture
        if contact_time_step_s is None:
            self._contact_time_step_s = None
            self._contact_time_grid_by_action: Dict[
                int, _ContactTimeTickGrid
            ] = {}
        else:
            if sampling_mixture is None:
                raise ValueError(
                    "contact_time_step_s is a production-mixture contract"
                )
            step = _finite(
                contact_time_step_s,
                name="contact_time_step_s",
                minimum=0.0,
            )
            if step <= 0.0:
                raise ValueError("contact_time_step_s must be > 0")
            self._contact_time_step_s = step
            self._contact_time_grid_by_action = {
                uid: _contact_time_tick_grid(
                    self._profiles[uid],
                    step_s=step,
                    frontier_band_fraction=(
                        sampling_mixture.frontier_band_fraction
                    ),
                )
                for uid in self.action_uids
            }
        self._rng_by_action = {
            uid: _CounterRng(self._seed) for uid in self.action_uids
        }
        # Pure, bounded run-local memoization.  DomainLevels is frozen, but
        # the cache key still snapshots every float by its exact IEEE-754
        # bytes so equal-value reconstructions hit while +0.0/-0.0 retain the
        # canonical-JSON distinction.  The request key then includes the same
        # four fields as the historical JSON payload.  These caches authorize
        # nothing, consume no random draws, and are deliberately absent from
        # state_dict: clearing them before/after an exact resume can only
        # change CPU work, never samples or receipt bytes.
        self._request_digest_cache_limit = max(
            64, 8 * len(self._profiles)
        )
        self._levels_sha256_cache: Dict[bytes, str] = {}
        self._request_digest_cache: Dict[
            Tuple[str, int, int, str], bytes
        ] = {}
        self._birth_count_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._sample_count_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._diagnostic_last_birth_draw_end_by_action = {
            uid: 0 for uid in self.action_uids
        }
        self._diagnostic_last_sample_draw_end_by_action = {
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
        contract_payload = {
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
        if self._sampling_mixture is not None:
            contract_payload["sampling_mixture"] = (
                self._sampling_mixture.as_dict()
            )
            contract_payload["contact_time_step_s"] = (
                self._contact_time_step_s
            )
            if self._contact_time_step_s is not None:
                contract_payload["contact_time_tick_grids"] = [
                    {
                        "action_uid": uid,
                        **self._contact_time_grid_by_action[uid].to_dict(),
                    }
                    for uid in self.action_uids
                ]
        self._contract_sha256 = _sha256_json(contract_payload)
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
    def sampling_mixture(self) -> Optional[SamplingMixture]:
        return self._sampling_mixture

    @property
    def contact_time_step_s(self) -> Optional[float]:
        return self._contact_time_step_s

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
        if self._diagnostic_fast_path:
            return (
                last_index,
                self._diagnostic_last_birth_draw_end_by_action[action_uid],
            )
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
        if self._diagnostic_fast_path:
            return (
                last_index,
                self._diagnostic_last_sample_draw_end_by_action[action_uid],
            )
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

    def _ensure_request_digest_caches(self) -> None:
        """Lazily install disposable digest caches on replay views.

        Deterministic assertion replay deliberately constructs a narrow
        ``ActionBallSampler`` view with ``object.__new__`` so it can avoid
        re-hashing the full profile bank.  These caches are derived,
        non-authoritative state, so replay views may create them on first use
        without changing samples, receipts, or exact-resume state.
        """

        if not hasattr(self, "_request_digest_cache_limit"):
            self._request_digest_cache_limit = max(
                64, 8 * len(self._profiles)
            )
        if not hasattr(self, "_levels_sha256_cache"):
            self._levels_sha256_cache = {}
        if not hasattr(self, "_request_digest_cache"):
            self._request_digest_cache = {}

    def _request_digest(
        self,
        *,
        kind: str,
        action_uid: int,
        domain_epoch: int,
        levels: DomainLevels,
    ) -> bytes:
        self._ensure_request_digest_caches()
        levels_fingerprint = struct.pack(
            f"!{len(ARM_KEYS)}d",
            *(getattr(levels, name) for name in ARM_KEYS),
        )
        levels_sha256 = self._levels_sha256_cache.get(
            levels_fingerprint
        )
        if levels_sha256 is None:
            levels_sha256 = levels.sha256
            if (
                len(self._levels_sha256_cache)
                >= self._request_digest_cache_limit
            ):
                # Both maps are disposable derived state.  Clearing together
                # bounds memory without introducing an eviction cursor that
                # would itself need exact-resume authority.
                self._levels_sha256_cache.clear()
                self._request_digest_cache.clear()
            self._levels_sha256_cache[
                levels_fingerprint
            ] = levels_sha256

        cache_key = (
            kind,
            action_uid,
            domain_epoch,
            levels_sha256,
        )
        digest = self._request_digest_cache.get(cache_key)
        if digest is None:
            digest = bytes.fromhex(
                _sha256_json(
                    {
                        "kind": kind,
                        "action_uid": action_uid,
                        "domain_epoch": domain_epoch,
                        "levels_sha256": levels_sha256,
                    }
                )
            )
            if (
                len(self._request_digest_cache)
                >= self._request_digest_cache_limit
            ):
                self._request_digest_cache.clear()
            self._request_digest_cache[cache_key] = digest
        return digest

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
        _sampling_plan_override: Optional[
            Tuple[str, DomainLevels, Optional[str]]
        ] = None,
    ) -> BaseBirthReceipt:
        """Sample the true-reset base spawn once for a new episode."""

        action_uid = self._validated_action_uid(action_uid)
        domain_epoch = _plain_int(domain_epoch, name="domain_epoch")
        levels = self._validated_levels(levels)
        base_yaw_rad = _finite(base_yaw_rad, name="base_yaw_rad")
        profile = self._profiles[action_uid]
        _validate_counter_rally_profile_support(
            profile, base_yaw_rad=base_yaw_rad
        )
        rng = self._rng_by_action[action_uid]
        birth_index = self._birth_count_by_action[action_uid]
        if (
            rng.draw_count > INT64_MAX - DRAWS_PER_BIRTH
            or birth_index >= INT64_MAX
        ):
            raise OverflowError("action birth tape exhausted")

        # The episode-base mixture owns an independent per-action proposal
        # cursor.  Freeze its plan before consuming the random tape so an
        # invalid frontier leaves counters/transcripts byte-identical.
        if _sampling_plan_override is None:
            (
                sampling_stratum,
                sampling_levels,
                frontier_arm,
            ) = _sampling_plan(
                profile=profile,
                levels=levels,
                mixture=self._sampling_mixture,
                proposal_index=birth_index,
                scope="birth",
            )
        else:
            (
                sampling_stratum,
                sampling_levels,
                frontier_arm,
            ) = _validated_sampling_plan_override(
                profile=profile,
                domain_levels=levels,
                mixture=self._sampling_mixture,
                scope="birth",
                plan=_sampling_plan_override,
            )
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
        if frontier_arm is not None:
            (
                negative_width,
                positive_width,
                frontier_side,
                frontier_draw_index,
            ) = _frontier_width_pair(
                profile, sampling_levels, frontier_arm
            )
            if frontier_draw_index >= DRAWS_PER_BIRTH:
                raise AssertionError(
                    "base-birth frontier selected a swing-only draw"
                )
            uniforms[frontier_draw_index] = _frontier_band_uniform(
                uniforms[frontier_draw_index],
                negative_width=negative_width,
                positive_width=positive_width,
                side=frontier_side,
                band_fraction=(
                    self._sampling_mixture.frontier_band_fraction
                ),
            )
        base_start_w_m = _sample_asymmetric_vector3(
            center=profile.base_spawn_center_w_m,
            lower_std=_vec3_lerp_levels(
                profile.base_spawn_std_lower_initial_m,
                profile.base_spawn_std_lower_max_m,
                (
                    sampling_levels.base_spawn_x_lower,
                    sampling_levels.base_spawn_y_lower,
                    0.0,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.base_spawn_std_upper_initial_m,
                profile.base_spawn_std_upper_max_m,
                (
                    sampling_levels.base_spawn_x_upper,
                    sampling_levels.base_spawn_y_upper,
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
            sampling_mixture=self._sampling_mixture,
            sampling_stratum=sampling_stratum,
            sampling_levels=(
                sampling_levels
                if self._sampling_mixture is not None
                else DomainLevels()
            ),
            frontier_arm=frontier_arm,
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
            sampling_mixture=self._sampling_mixture,
            sampling_stratum=sampling_stratum,
            sampling_levels=(
                sampling_levels
                if self._sampling_mixture is not None
                else DomainLevels()
            ),
            frontier_arm=frontier_arm,
        )
        self._issued_births_by_action[action_uid][
            birth_index
        ] = receipt
        self._birth_count_by_action[action_uid] = birth_index + 1
        if self._diagnostic_fast_path:
            self._diagnostic_last_birth_draw_end_by_action[
                action_uid
            ] = draw_end
        return receipt

    def _validate_birth(
        self,
        birth: BaseBirthReceipt,
        *,
        action_uid: int,
        domain_epoch: int,
        levels: DomainLevels,
        base_yaw_rad: float,
        sampling_plan_override: Optional[
            Tuple[str, DomainLevels, Optional[str]]
        ] = None,
    ) -> SamplingProfile:
        if not isinstance(birth, BaseBirthReceipt):
            raise TypeError("birth must be a BaseBirthReceipt")
        profile = self._profiles[action_uid]
        if sampling_plan_override is None:
            (
                expected_birth_stratum,
                expected_birth_levels,
                expected_birth_frontier,
            ) = _sampling_plan(
                profile=profile,
                levels=levels,
                mixture=self._sampling_mixture,
                proposal_index=birth.birth_index,
                scope="birth",
            )
        else:
            (
                expected_birth_stratum,
                expected_birth_levels,
                expected_birth_frontier,
            ) = _validated_sampling_plan_override(
                profile=profile,
                domain_levels=levels,
                mixture=self._sampling_mixture,
                scope="birth",
                plan=sampling_plan_override,
            )
        mismatches = []
        expected_fields = [
            ("sampler_contract_sha256", self._contract_sha256),
            ("arm_catalog_sha256", ARM_CATALOG_SHA256),
            ("action_uid", action_uid),
            ("domain_epoch", domain_epoch),
            ("domain_levels", levels),
            ("profile_sha256", profile.sha256),
            ("levels_sha256", levels.sha256),
            ("mobility_mode", profile.mobility_mode),
            ("base_yaw_rad", base_yaw_rad),
        ]
        if self._sampling_mixture is not None:
            expected_fields.extend(
                (
                    ("sampling_mixture", self._sampling_mixture),
                    ("sampling_stratum", expected_birth_stratum),
                    ("sampling_levels", expected_birth_levels),
                    ("frontier_arm", expected_birth_frontier),
                )
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
            sampling_mixture=birth.sampling_mixture,
            sampling_stratum=birth.sampling_stratum,
            sampling_levels=birth.sampling_levels,
            frontier_arm=birth.frontier_arm,
        )
        if type(birth.birth_id) is not str or birth.birth_id != _sha256_json(
            payload
        ):
            raise ValueError("birth receipt identity check failed")
        return profile

    def sample_many_prevalidated(
        self,
        *,
        birth: BaseBirthReceipt,
        action_uid: int,
        domain_epoch: int,
        levels: Union[DomainLevels, Mapping[str, object]],
        base_yaw_rad: float = 0.0,
        count: int,
    ) -> Tuple[BallBaseSample, ...]:
        """Emit one diagnostic proposal batch from an exact live birth.

        Birth issuance already performed the expensive canonical proof.  The
        diagnostic runtime retains that exact immutable object, so this path
        binds the request to the sampler's active birth index once and then
        advances the unchanged per-action scalar tape ``count`` times.
        """

        if not self._diagnostic_fast_path:
            raise RuntimeError(
                "sample_many_prevalidated requires diagnostic_unauthorized"
            )
        count = _plain_int(count, name="count", minimum=1)
        action_uid = self._validated_action_uid(action_uid)
        domain_epoch = _plain_int(domain_epoch, name="domain_epoch")
        levels = self._validated_levels(levels)
        base_yaw_rad = _finite(base_yaw_rad, name="base_yaw_rad")
        if not isinstance(birth, BaseBirthReceipt):
            raise TypeError("birth must be a BaseBirthReceipt")
        profile = self._profiles[action_uid]
        issued_birth = self._issued_births_by_action[action_uid].get(
            birth.birth_index
        )
        if issued_birth is not birth:
            raise ValueError(
                "diagnostic prevalidated birth is not the exact live "
                "sampler object"
            )
        if (
            birth.sampler_contract_sha256 != self._contract_sha256
            or birth.arm_catalog_sha256 != ARM_CATALOG_SHA256
            or birth.action_uid != action_uid
            or birth.domain_epoch != domain_epoch
            or birth.domain_levels != levels
            or birth.mobility_mode != profile.mobility_mode
            or birth.base_yaw_rad != base_yaw_rad
            or birth.birth_index
            < self._retired_birth_count_by_action[action_uid]
            or birth.birth_index
            >= self._birth_count_by_action[action_uid]
            or birth.draw_end
            > self._rng_by_action[action_uid].draw_count
        ):
            raise ValueError(
                "diagnostic prevalidated birth/request binding drifted"
            )
        _validate_counter_rally_profile_support(
            profile, base_yaw_rad=base_yaw_rad
        )
        return tuple(
            self.sample(
                birth=birth,
                action_uid=action_uid,
                domain_epoch=domain_epoch,
                levels=levels,
                base_yaw_rad=base_yaw_rad,
                _diagnostic_prevalidated_profile=profile,
                _diagnostic_prevalidated_authority=(
                    _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY
                ),
            )
            for _ in range(count)
        )

    def sample(
        self,
        *,
        birth: BaseBirthReceipt,
        action_uid: int,
        domain_epoch: int,
        levels: Union[DomainLevels, Mapping[str, object]],
        base_yaw_rad: float = 0.0,
        _birth_sampling_plan_override: Optional[
            Tuple[str, DomainLevels, Optional[str]]
        ] = None,
        _sampling_plan_override: Optional[
            Tuple[str, DomainLevels, Optional[str]]
        ] = None,
        _diagnostic_prevalidated_profile: Optional[
            SamplingProfile
        ] = None,
        _diagnostic_prevalidated_authority: object = None,
    ) -> BallBaseSample:
        """Sample a new ball/aim against a verified episode birth."""

        if _diagnostic_prevalidated_authority is None:
            action_uid = self._validated_action_uid(action_uid)
            domain_epoch = _plain_int(
                domain_epoch, name="domain_epoch"
            )
            levels = self._validated_levels(levels)
            base_yaw_rad = _finite(
                base_yaw_rad, name="base_yaw_rad"
            )
            profile = self._validate_birth(
                birth,
                action_uid=action_uid,
                domain_epoch=domain_epoch,
                levels=levels,
                base_yaw_rad=base_yaw_rad,
                sampling_plan_override=_birth_sampling_plan_override,
            )
            _validate_counter_rally_profile_support(
                profile, base_yaw_rad=base_yaw_rad
            )
        elif (
            _diagnostic_prevalidated_authority
            is _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY
            and self._diagnostic_fast_path
            and _diagnostic_prevalidated_profile
            is self._profiles.get(action_uid)
        ):
            profile = _diagnostic_prevalidated_profile
        else:
            raise RuntimeError(
                "sample diagnostic prevalidation has no internal authority"
            )
        rng = self._rng_by_action[action_uid]
        sample_index = self._sample_count_by_action[action_uid]
        sample_birth_indices = (
            self._issued_sample_birth_indices_by_action[action_uid]
        )
        retained_sample_start = (
            self._retired_sample_count_by_action[action_uid]
        )
        if (
            not self._diagnostic_fast_path
            and len(sample_birth_indices)
            != (sample_index - retained_sample_start)
        ):
            raise RuntimeError(
                "sample authority ledger is inconsistent with sample_count"
            )
        if (
            rng.draw_count > INT64_MAX - DRAWS_PER_SAMPLE
            or sample_index >= INT64_MAX
        ):
            raise OverflowError("action swing tape exhausted")

        # Validate and freeze the stratum before consuming the counter tape.
        # In particular, a malformed/zero-width frontier request must leave
        # RNG, assignment, and sample counters byte-identical for retry.
        if _sampling_plan_override is None:
            (
                sampling_stratum,
                sampling_levels,
                frontier_arm,
            ) = _sampling_plan(
                profile=profile,
                levels=levels,
                mixture=self._sampling_mixture,
                proposal_index=sample_index,
                scope="swing",
                contact_time_tick_grid=(
                    self._contact_time_grid_by_action.get(action_uid)
                ),
            )
        else:
            (
                sampling_stratum,
                sampling_levels,
                frontier_arm,
            ) = _validated_sampling_plan_override(
                profile=profile,
                domain_levels=levels,
                mixture=self._sampling_mixture,
                scope="swing",
                plan=_sampling_plan_override,
            )
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
        if (
            _diagnostic_prevalidated_authority
            is _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY
        ):
            uniforms = rng._uniform_open_many_diagnostic(
                request_digest,
                DRAWS_PER_SAMPLE,
                _authority=(
                    _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY
                ),
            )
        else:
            uniforms = [
                rng.uniform_open(request_digest)
                for _ in range(DRAWS_PER_SAMPLE)
            ]
        native_ttc_grid = self._contact_time_grid_by_action.get(
            action_uid
        )
        if (
            frontier_arm is not None
            and not (
                native_ttc_grid is not None
                and frontier_arm.startswith("time_to_contact_")
            )
        ):
            (
                negative_width,
                positive_width,
                frontier_side,
                frontier_draw_index,
            ) = _frontier_width_pair(
                profile, sampling_levels, frontier_arm
            )
            uniforms[frontier_draw_index] = _frontier_band_uniform(
                uniforms[frontier_draw_index],
                negative_width=negative_width,
                positive_width=positive_width,
                side=frontier_side,
                band_fraction=(
                    self._sampling_mixture.frontier_band_fraction
                ),
            )
        if self._sampling_mixture is None:
            # Legacy receipts retain their historical, explicitly unused
            # per-swing latent spawn bytes.
            base_spawn_latent_w_m = _sample_asymmetric_vector3(
                center=profile.base_spawn_center_w_m,
                lower_std=_vec3_lerp_levels(
                    profile.base_spawn_std_lower_initial_m,
                    profile.base_spawn_std_lower_max_m,
                    (
                        sampling_levels.base_spawn_x_lower,
                        sampling_levels.base_spawn_y_lower,
                        0.0,
                    ),
                ),
                upper_std=_vec3_lerp_levels(
                    profile.base_spawn_std_upper_initial_m,
                    profile.base_spawn_std_upper_max_m,
                    (
                        sampling_levels.base_spawn_x_upper,
                        sampling_levels.base_spawn_y_upper,
                        0.0,
                    ),
                ),
                lower_bound=profile.base_spawn_min_w_m,
                upper_bound=profile.base_spawn_max_w_m,
                uniforms=uniforms[0:3],
                name="base_spawn_latent",
            )
        else:
            # Base spawn is a birth proposal, never a swing arm.  Keep the
            # fixed draw budget but bind the receipt-visible value to the
            # actual birth instead of exposing an unused latent as capability.
            base_spawn_latent_w_m = birth.base_start_w_m
        base_travel_latent_b_yaw_m = _sample_asymmetric_vector3(
            center=profile.base_travel_center_b_yaw_m,
            lower_std=_vec3_lerp_levels(
                profile.base_travel_std_lower_initial_m,
                profile.base_travel_std_lower_max_m,
                (
                    sampling_levels.base_travel_x_lower,
                    sampling_levels.base_travel_y_lower,
                    0.0,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.base_travel_std_upper_initial_m,
                profile.base_travel_std_upper_max_m,
                (
                    sampling_levels.base_travel_x_upper,
                    sampling_levels.base_travel_y_upper,
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
                    sampling_levels.contact_x_lower,
                    sampling_levels.contact_y_lower,
                    sampling_levels.contact_z_lower,
                ),
            ),
            upper_std=_vec3_lerp_levels(
                profile.contact_offset_std_upper_initial_m,
                profile.contact_offset_std_upper_max_m,
                (
                    sampling_levels.contact_x_upper,
                    sampling_levels.contact_y_upper,
                    sampling_levels.contact_z_upper,
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

        if native_ttc_grid is None:
            time_to_contact_tick = None
            time_to_contact_s = _sample_asymmetric_truncated(
                center=profile.time_to_contact_center_s,
                lower_std=_lerp(
                    profile.time_to_contact_std_lower_initial_s,
                    profile.time_to_contact_std_lower_max_s,
                    sampling_levels.time_to_contact_lower,
                ),
                upper_std=_lerp(
                    profile.time_to_contact_std_upper_initial_s,
                    profile.time_to_contact_std_upper_max_s,
                    sampling_levels.time_to_contact_upper,
                ),
                lower_bound=profile.time_to_contact_min_s,
                upper_bound=profile.time_to_contact_max_s,
                uniform=uniforms[9],
                name="time_to_contact",
            )
        else:
            ttc_frontier_side = None
            ttc_frontier_band = None
            if (
                frontier_arm is not None
                and frontier_arm.startswith("time_to_contact_")
            ):
                ttc_frontier_side = (
                    "negative"
                    if frontier_arm.endswith("_lower")
                    else "positive"
                )
                if self._sampling_mixture is None:
                    raise AssertionError(
                        "TTC frontier arm has no sampling mixture"
                    )
                ttc_frontier_band = (
                    self._sampling_mixture.frontier_band_fraction
                )
            time_to_contact_tick = native_ttc_grid.sample_tick(
                uniform=uniforms[9],
                levels=sampling_levels,
                frontier_side=ttc_frontier_side,
                frontier_band_fraction=ttc_frontier_band,
            )
            time_to_contact_s = (
                time_to_contact_tick * native_ttc_grid.step_s
            )
        speed_mps = _sample_asymmetric_truncated(
            center=profile.incoming_speed_center_mps,
            lower_std=_lerp(
                profile.incoming_speed_std_lower_initial_mps,
                profile.incoming_speed_std_lower_max_mps,
                sampling_levels.incoming_speed_lower,
            ),
            upper_std=_lerp(
                profile.incoming_speed_std_upper_initial_mps,
                profile.incoming_speed_std_upper_max_mps,
                sampling_levels.incoming_speed_upper,
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
                sampling_levels.incoming_direction_u_neg,
            ),
            u_positive_width_deg=_lerp(
                profile.incoming_direction_tangent_u_pos_initial_deg,
                profile.incoming_direction_tangent_u_pos_max_deg,
                sampling_levels.incoming_direction_u_pos,
            ),
            v_negative_width_deg=_lerp(
                profile.incoming_direction_tangent_v_neg_initial_deg,
                profile.incoming_direction_tangent_v_neg_max_deg,
                sampling_levels.incoming_direction_v_neg,
            ),
            v_positive_width_deg=_lerp(
                profile.incoming_direction_tangent_v_pos_initial_deg,
                profile.incoming_direction_tangent_v_pos_max_deg,
                sampling_levels.incoming_direction_v_pos,
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
                sampling_levels.spin_magnitude_lower,
            ),
            upper_std=_lerp(
                profile.spin_magnitude_std_upper_initial_radps,
                profile.spin_magnitude_std_upper_max_radps,
                sampling_levels.spin_magnitude_upper,
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
                sampling_levels.spin_direction_u_neg,
            ),
            u_positive_width_deg=_lerp(
                profile.spin_direction_tangent_u_pos_initial_deg,
                profile.spin_direction_tangent_u_pos_max_deg,
                sampling_levels.spin_direction_u_pos,
            ),
            v_negative_width_deg=_lerp(
                profile.spin_direction_tangent_v_neg_initial_deg,
                profile.spin_direction_tangent_v_neg_max_deg,
                sampling_levels.spin_direction_v_neg,
            ),
            v_positive_width_deg=_lerp(
                profile.spin_direction_tangent_v_pos_initial_deg,
                profile.spin_direction_tangent_v_pos_max_deg,
                sampling_levels.spin_direction_v_pos,
            ),
            uniforms=uniforms[14:16],
        )
        spin_direction_w = _rotate_yaw(
            spin_direction_b_yaw, base_yaw_rad
        )
        spin_w_radps = _scale(
            spin_direction_w, spin_magnitude_radps
        )
        if profile.counter_rally_objective is None:
            landing_aim_w_xy_m = _sample_asymmetric_vector2(
                center=profile.landing_aim_center_w_xy_m,
                lower_std=_vec2_lerp_levels(
                    profile.landing_aim_std_lower_initial_m,
                    profile.landing_aim_std_lower_max_m,
                    (
                        sampling_levels.landing_aim_x_lower,
                        sampling_levels.landing_aim_y_lower,
                    ),
                ),
                upper_std=_vec2_lerp_levels(
                    profile.landing_aim_std_upper_initial_m,
                    profile.landing_aim_std_upper_max_m,
                    (
                        sampling_levels.landing_aim_x_upper,
                        sampling_levels.landing_aim_y_upper,
                    ),
                ),
                lower_bound=profile.landing_aim_min_w_xy_m,
                upper_bound=profile.landing_aim_max_w_xy_m,
                uniforms=uniforms[16:18],
                name="landing_aim",
            )
        else:
            # Draw 17 remains deliberately reserved.  It is consumed by the
            # fixed 18-draw tape and therefore covered by draw_start/draw_end,
            # but it cannot independently perturb landing y: counter-rally y
            # is a deterministic consequence of the reverse incoming ray.
            _reserved_landing_y_draw = uniforms[17]
            landing_x = _sample_asymmetric_truncated(
                center=profile.landing_aim_center_w_xy_m[0],
                lower_std=_lerp(
                    profile.landing_aim_std_lower_initial_m[0],
                    profile.landing_aim_std_lower_max_m[0],
                    sampling_levels.landing_aim_x_lower,
                ),
                upper_std=_lerp(
                    profile.landing_aim_std_upper_initial_m[0],
                    profile.landing_aim_std_upper_max_m[0],
                    sampling_levels.landing_aim_x_upper,
                ),
                lower_bound=profile.landing_aim_min_w_xy_m[0],
                upper_bound=profile.landing_aim_max_w_xy_m[0],
                uniform=uniforms[16],
                name="landing_aim.x",
            )
            (
                landing_aim_w_xy_m,
                _counter_rally_proposal_rejection_reason,
            ) = _counter_rally_reverse_ray_geometry(
                contact_w_m=contact_w_m,
                incoming_direction_w=incoming_direction_w,
                landing_x_w_m=landing_x,
                objective=profile.counter_rally_objective,
            )
            # A per-proposal miss remains a proposal.  The fixed-action solver
            # calls the same helper and owns named rejection accounting; the
            # sampler never loses this row or its fixed draw transcript.

        draw_end = rng.draw_count
        if draw_end - draw_start != DRAWS_PER_SAMPLE:
            raise AssertionError("internal fixed-draw contract violated")
        if (
            _diagnostic_prevalidated_authority
            is _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY
        ):
            # The exact live immutable birth already pins both hashes.  Avoid
            # repeating their canonical JSON/SHA construction for every row.
            profile_sha256 = birth.profile_sha256
            levels_sha256 = birth.levels_sha256
        else:
            profile_sha256 = profile.sha256
            levels_sha256 = levels.sha256
        candidate = BallBaseSample(
            sample_id="",
            sampler_contract_sha256=self._contract_sha256,
            arm_catalog_sha256=ARM_CATALOG_SHA256,
            action_uid=action_uid,
            domain_epoch=domain_epoch,
            domain_levels=levels,
            birth_index=(
                birth.birth_index
                if self._sampling_mixture is not None
                else -1
            ),
            birth_sampling_stratum=(
                birth.sampling_stratum
                if self._sampling_mixture is not None
                else "domain"
            ),
            birth_sampling_levels=(
                birth.sampling_levels
                if self._sampling_mixture is not None
                else levels
            ),
            birth_frontier_arm=(
                birth.frontier_arm
                if self._sampling_mixture is not None
                else None
            ),
            sampling_mixture=self._sampling_mixture,
            sampling_stratum=sampling_stratum,
            sampling_levels=sampling_levels,
            frontier_arm=frontier_arm,
            sample_index=sample_index,
            birth_id=birth.birth_id,
            profile_sha256=profile_sha256,
            levels_sha256=levels_sha256,
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
            contact_time_step_s=self._contact_time_step_s,
            time_to_contact_tick=time_to_contact_tick,
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
        if (
            _diagnostic_prevalidated_authority
            is not _DIAGNOSTIC_PREVALIDATED_SAMPLE_AUTHORITY
        ):
            completed.verify_sample_id()
        if self._diagnostic_fast_path:
            self._diagnostic_last_sample_draw_end_by_action[
                action_uid
            ] = draw_end
        else:
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

    def forget_diagnostic_births(
        self,
        births: Sequence[BaseBirthReceipt],
    ) -> None:
        """Forget exact retired births without touching RNG/high-water state.

        Diagnostic runs do not claim replay or exact sampler resume.  Their
        live authority is therefore bounded by active episode births instead
        of retaining an append-only transcript.  Validation is atomic and the
        operation is O(k) in the number of retired environments.
        """

        if not self._diagnostic_fast_path:
            raise RuntimeError(
                "forget_diagnostic_births requires diagnostic_unauthorized"
            )
        if (
            isinstance(births, (str, bytes))
            or not isinstance(births, Sequence)
        ):
            raise TypeError("births must be a non-string sequence")
        validated = []
        seen = set()
        for index, birth in enumerate(births):
            if not isinstance(birth, BaseBirthReceipt):
                raise TypeError(
                    f"births[{index}] must be a BaseBirthReceipt"
                )
            action_uid = self._validated_action_uid(birth.action_uid)
            key = (action_uid, birth.birth_index)
            if key in seen:
                raise ValueError("diagnostic retirement repeated one birth")
            seen.add(key)
            if (
                self._issued_births_by_action[action_uid].get(
                    birth.birth_index
                )
                != birth
            ):
                raise ValueError(
                    "diagnostic retirement birth is not live sampler authority"
                )
            validated.append((action_uid, birth.birth_index))
        for action_uid, birth_index in validated:
            del self._issued_births_by_action[action_uid][birth_index]

    def _verify_sampling_membership(
        self,
        sample: BallBaseSample,
    ) -> None:
        """Recompute stratum/frontier geometry after identity coercion."""

        sample.verify_sample_id()
        action_uid = self._validated_action_uid(sample.action_uid)
        profile = self._profiles[action_uid]
        if sample.sampler_contract_sha256 != self._contract_sha256:
            raise ValueError("sample sampler contract mismatch")
        if sample.profile_sha256 != profile.sha256:
            raise ValueError("sample profile mismatch")
        if sample.arm_catalog_sha256 != ARM_CATALOG_SHA256:
            raise ValueError("sample arm catalog mismatch")
        if sample.contact_time_step_s != self._contact_time_step_s:
            raise ValueError(
                "sample contact-time lattice differs from sampler contract"
            )
        if self._contact_time_step_s is None:
            if sample.time_to_contact_tick is not None:
                raise ValueError(
                    "continuous-time sampler received a contact tick"
                )
        else:
            tick = _plain_int(
                sample.time_to_contact_tick,
                name="sample.time_to_contact_tick",
                minimum=1,
            )
            if sample.time_to_contact_s != (
                tick * self._contact_time_step_s
            ):
                raise ValueError(
                    "sample TTC is not exactly its policy-step tick"
                )
        expected = _sampling_plan(
            profile=profile,
            levels=sample.domain_levels,
            mixture=self._sampling_mixture,
            proposal_index=sample.sample_index,
            scope="swing",
            contact_time_tick_grid=(
                self._contact_time_grid_by_action.get(action_uid)
            ),
        )
        actual = (
            sample.sampling_stratum,
            sample.sampling_levels,
            sample.frontier_arm,
        )
        if actual != expected:
            raise ValueError(
                "sample sampling metadata disagrees with deterministic plan"
            )
        if self._sampling_mixture is not None:
            birth = self._issued_births_by_action[action_uid].get(
                sample.birth_index
            )
            if birth is None:
                raise ValueError(
                    "sample references an unavailable birth transcript"
                )
            bound_birth = (
                sample.birth_id,
                sample.birth_sampling_stratum,
                sample.birth_sampling_levels,
                sample.birth_frontier_arm,
            )
            expected_birth = (
                birth.birth_id,
                birth.sampling_stratum,
                birth.sampling_levels,
                birth.frontier_arm,
            )
            if bound_birth != expected_birth:
                raise ValueError(
                    "sample birth sampling metadata disagrees with "
                    "the issued birth"
                )
        if sample.frontier_arm is None:
            return
        mixture = self._sampling_mixture
        if mixture is None:
            raise ValueError("frontier sample has no sampler mixture")
        native_ttc_grid = self._contact_time_grid_by_action.get(
            action_uid
        )
        if (
            native_ttc_grid is not None
            and sample.frontier_arm.startswith("time_to_contact_")
        ):
            side = (
                "negative"
                if sample.frontier_arm.endswith("_lower")
                else "positive"
            )
            first, last = native_ttc_grid.tick_bounds(
                levels=sample.sampling_levels,
                frontier_side=side,
                frontier_band_fraction=(
                    mixture.frontier_band_fraction
                ),
            )
            tick = _plain_int(
                sample.time_to_contact_tick,
                name="sample.time_to_contact_tick",
                minimum=1,
            )
            if not first <= tick <= last:
                raise ValueError(
                    "sample TTC tick does not lie in its native frontier set"
                )
            return
        negative, positive, side, _ = _frontier_width_pair(
            profile, sample.sampling_levels, sample.frontier_arm
        )
        delta = _frontier_coordinate_delta(
            sample, profile, sample.frontier_arm
        )
        if side == "negative":
            width = negative
            normalized = -delta / width if width > 0.0 else -1.0
        else:
            width = positive
            normalized = delta / width if width > 0.0 else -1.0
        lower = 1.0 - mixture.frontier_band_fraction
        tolerance = 1.0e-10
        if (
            width <= 0.0
            or normalized + tolerance < lower
            or normalized > 1.0 + tolerance
        ):
            raise ValueError(
                "sample does not lie in its recomputed frontier band"
            )

    def verify_sampling_membership(
        self,
        sample_or_receipt: Union[
            BallBaseSample, Mapping[str, object]
        ],
    ) -> None:
        """Verify exact issuance, then recompute frontier membership.

        A self-consistent public sample hash is not sampling authority.  This
        method intentionally requires the receipt to match this sampler's
        exact issued action-tape row before its stratum is accepted.  The
        replay path then calls :meth:`_verify_sampling_membership`, which
        derives the selected outer band from profile, levels, and sample
        coordinates; no caller-provided ``in_new_band`` boolean exists.
        """

        self.assert_issued_sample(sample_or_receipt)

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

    def verify_birth_sampling_membership(
        self,
        birth_or_receipt: Union[
            BaseBirthReceipt, Mapping[str, object]
        ],
    ) -> None:
        """Verify issuance and recompute a base-spawn frontier band."""

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
        self.assert_issued_birth(birth)
        if birth.frontier_arm is None:
            return
        mixture = self._sampling_mixture
        if mixture is None:
            raise ValueError("frontier birth has no sampler mixture")
        profile = self._profiles[birth.action_uid]
        negative, positive, side, _ = _frontier_width_pair(
            profile, birth.sampling_levels, birth.frontier_arm
        )
        delta = _birth_frontier_coordinate_delta(
            birth, profile, birth.frontier_arm
        )
        if side == "negative":
            width = negative
            normalized = -delta / width if width > 0.0 else -1.0
        else:
            width = positive
            normalized = delta / width if width > 0.0 else -1.0
        lower = 1.0 - mixture.frontier_band_fraction
        if (
            width <= 0.0
            or normalized + 1.0e-10 < lower
            or normalized > 1.0 + 1.0e-10
        ):
            raise ValueError(
                "birth does not lie in its recomputed frontier band"
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
        self._verify_sampling_membership(sample)

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
        replay._diagnostic_fast_path = False
        replay._sampling_mixture = self._sampling_mixture
        replay._contact_time_step_s = self._contact_time_step_s
        replay._contact_time_grid_by_action = {
            action_uid: self._contact_time_grid_by_action[action_uid]
        } if action_uid in self._contact_time_grid_by_action else {}
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
                    (
                        _MIXTURE_BIRTH_STATE_KEYS
                        if self._sampling_mixture is not None
                        else _BIRTH_STATE_KEYS
                    ),
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
                (
                    sampling_stratum,
                    sampling_levels,
                    frontier_arm,
                ) = _sampling_plan(
                    profile=profile,
                    levels=levels,
                    mixture=self._sampling_mixture,
                    proposal_index=index,
                    scope="birth",
                )
                if self._sampling_mixture is not None:
                    declared_mixture = SamplingMixture.from_mapping(
                        birth_row["sampling_mixture"]
                    )
                    declared_sampling_levels = (
                        DomainLevels.from_mapping(
                            birth_row["sampling_levels"]
                        )
                    )
                    if (
                        declared_mixture != self._sampling_mixture
                        or birth_row["sampling_stratum"]
                        != sampling_stratum
                        or declared_sampling_levels != sampling_levels
                        or birth_row["frontier_arm"] != frontier_arm
                    ):
                        raise ValueError(
                            f"issued_births[{uid}][{index}] sampling "
                            "plan mismatch"
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
                if frontier_arm is not None:
                    (
                        negative_width,
                        positive_width,
                        frontier_side,
                        frontier_draw_index,
                    ) = _frontier_width_pair(
                        profile, sampling_levels, frontier_arm
                    )
                    if frontier_draw_index >= DRAWS_PER_BIRTH:
                        raise ValueError(
                            f"issued_births[{uid}][{index}] frontier "
                            "uses a swing-only draw"
                        )
                    replay_uniforms[frontier_draw_index] = (
                        _frontier_band_uniform(
                            replay_uniforms[frontier_draw_index],
                            negative_width=negative_width,
                            positive_width=positive_width,
                            side=frontier_side,
                            band_fraction=(
                                self._sampling_mixture
                                .frontier_band_fraction
                            ),
                        )
                    )
                replayed_base_start = _sample_asymmetric_vector3(
                    center=profile.base_spawn_center_w_m,
                    lower_std=_vec3_lerp_levels(
                        profile.base_spawn_std_lower_initial_m,
                        profile.base_spawn_std_lower_max_m,
                        (
                            sampling_levels.base_spawn_x_lower,
                            sampling_levels.base_spawn_y_lower,
                            0.0,
                        ),
                    ),
                    upper_std=_vec3_lerp_levels(
                        profile.base_spawn_std_upper_initial_m,
                        profile.base_spawn_std_upper_max_m,
                        (
                            sampling_levels.base_spawn_x_upper,
                            sampling_levels.base_spawn_y_upper,
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
                    sampling_mixture=self._sampling_mixture,
                    sampling_stratum=sampling_stratum,
                    sampling_levels=(
                        sampling_levels
                        if self._sampling_mixture is not None
                        else DomainLevels()
                    ),
                    frontier_arm=frontier_arm,
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
                    sampling_mixture=self._sampling_mixture,
                    sampling_stratum=sampling_stratum,
                    sampling_levels=(
                        sampling_levels
                        if self._sampling_mixture is not None
                        else DomainLevels()
                    ),
                    frontier_arm=frontier_arm,
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


def sample_frozen_evaluation_proposal(
    profile: SamplingProfile,
    *,
    evaluation_seed: int,
    external_sample_index: int,
    external_birth_index: int,
    domain_epoch: int,
    domain_levels: Union[DomainLevels, Mapping[str, object]],
    rho: float,
    sampling_stratum: str,
    selected_arm: Optional[str],
    base_yaw_rad: float,
    policy_dt_s: float,
) -> FrozenEvaluationProposal:
    """Random-access one formal proposal without touching training state.

    ``evaluation_seed`` is allocated independently for every proposal.  The
    temporary sampler therefore starts at draw zero, uses draws ``[0, 3)`` for
    that proposal's birth and ``[3, 21)`` for its swing, and is discarded.
    External sample/birth indices are receipt identity, not a request to burn
    all preceding rows.
    """

    if not isinstance(profile, SamplingProfile):
        raise TypeError("profile must be SamplingProfile")
    evaluation_seed = _plain_int(
        evaluation_seed, name="evaluation_seed"
    )
    external_sample_index = _plain_int(
        external_sample_index, name="external_sample_index"
    )
    external_birth_index = _plain_int(
        external_birth_index, name="external_birth_index"
    )
    domain_epoch = _plain_int(domain_epoch, name="domain_epoch")
    levels = (
        domain_levels
        if isinstance(domain_levels, DomainLevels)
        else DomainLevels.from_mapping(domain_levels)
    )
    rho = _finite(rho, name="rho", minimum=0.0, maximum=1.0)
    base_yaw_rad = _finite(base_yaw_rad, name="base_yaw_rad")
    policy_dt_s = _finite(
        policy_dt_s, name="policy_dt_s", minimum=0.0
    )
    if policy_dt_s <= 0.0:
        raise ValueError("policy_dt_s must be > 0")
    if sampling_stratum not in (
        "center",
        "interior",
        "frontier",
    ):
        raise ValueError("sampling_stratum is invalid")
    if (sampling_stratum == "frontier") != (
        selected_arm is not None
    ):
        raise ValueError(
            "selected_arm must be present exactly for frontier"
        )
    if selected_arm is not None and selected_arm not in ARM_KEYS:
        raise ValueError("selected_arm is outside ARM_KEYS")

    mixture = SamplingMixture()
    expected_sample_stratum = mixture.stratum_for(
        external_sample_index
    )
    expected_birth_stratum = mixture.stratum_for(
        external_birth_index
    )
    if (
        sampling_stratum != expected_sample_stratum
        or sampling_stratum != expected_birth_stratum
    ):
        raise ValueError(
            "authority stratum disagrees with the exact 1/3/1 allocation "
            "schedule"
        )

    birth_arms = tuple(_BASE_SPAWN_ARMS)
    swing_arms = tuple(
        arm
        for arm in ARM_KEYS
        if arm not in _BASE_SPAWN_ARMS
        and not (
            profile.mobility_mode == "no_move"
            and arm.startswith("base_travel_")
        )
    )
    if (
        selected_arm is not None
        and selected_arm not in birth_arms
        and selected_arm not in swing_arms
    ):
        raise ValueError(
            "selected_arm is inactive for this mobility mode"
        )
    def _scope_plan(
        active_arms: Tuple[str, ...],
    ) -> Tuple[str, DomainLevels, Optional[str]]:
        owns_frontier = (
            sampling_stratum == "frontier"
            and selected_arm in active_arms
        )
        if owns_frontier:
            selected_only = DomainLevels(
                **{
                    arm: (
                        getattr(levels, arm)
                        if arm == selected_arm
                        else 0.0
                    )
                    for arm in ARM_KEYS
                }
            )
            return ("frontier", selected_only, selected_arm)
        if sampling_stratum in ("center", "frontier"):
            # For a one-arm frontier the non-owning component is held at
            # center too; this is the isolation proof that prevents a hidden
            # base-spawn plus swing double-frontier proposal.
            return ("center", DomainLevels(), None)
        return (
            "interior",
            _rho_scaled_levels(
                profile,
                levels,
                rho,
                active_arms=active_arms,
            ),
            None,
        )

    birth_plan = _scope_plan(birth_arms)
    swing_plan = _scope_plan(swing_arms)
    sampler = ActionBallSampler(
        (profile,),
        seed=evaluation_seed,
        sampling_mixture=mixture,
        contact_time_step_s=policy_dt_s,
    )
    uid = profile.action_uid
    # Random access: external indices are installed directly as receipt
    # cursors, while this proposal's independent seed starts at draw zero.
    sampler._birth_count_by_action[uid] = external_birth_index
    sampler._retired_birth_count_by_action[uid] = external_birth_index
    sampler._sample_count_by_action[uid] = external_sample_index
    sampler._retired_sample_count_by_action[uid] = (
        external_sample_index
    )
    sampler._rng_by_action[uid] = _CounterRng(evaluation_seed, 0)
    birth = sampler.reserve_birth(
        action_uid=uid,
        domain_epoch=domain_epoch,
        levels=levels,
        base_yaw_rad=base_yaw_rad,
        _sampling_plan_override=birth_plan,
    )
    sample = sampler.sample(
        birth=birth,
        action_uid=uid,
        domain_epoch=domain_epoch,
        levels=levels,
        base_yaw_rad=base_yaw_rad,
        _birth_sampling_plan_override=birth_plan,
        _sampling_plan_override=swing_plan,
    )
    if birth.draw_start != 0 or birth.draw_end != DRAWS_PER_BIRTH:
        raise AssertionError(
            "frozen proposal birth did not own exact local draws [0,3)"
        )
    if (
        sample.draw_start != DRAWS_PER_BIRTH
        or sample.draw_end != DRAWS_PER_BIRTH + DRAWS_PER_SAMPLE
    ):
        raise AssertionError(
            "frozen proposal swing did not own exact local draws [3,21)"
        )
    contract_sha256 = (
        frozen_evaluation_proposal_sampler_contract()["sha256"]
    )
    candidate = FrozenEvaluationProposal(
        proposal_sampler_contract_sha256=contract_sha256,
        proposal_receipt_sha256="",
        evaluation_seed=evaluation_seed,
        external_sample_index=external_sample_index,
        external_birth_index=external_birth_index,
        action_uid=uid,
        profile_sha256=profile.sha256,
        domain_epoch=domain_epoch,
        domain_levels=levels,
        rho=rho,
        sampling_stratum=sampling_stratum,
        selected_arm=selected_arm,
        birth_component_stratum=birth_plan[0],
        ball_task_component_stratum=swing_plan[0],
        base_yaw_rad=base_yaw_rad,
        policy_dt_s=policy_dt_s,
        birth=birth,
        sample=sample,
    )
    result = replace(
        candidate,
        proposal_receipt_sha256=_sha256_json(
            candidate.receipt_payload()
        ),
    )
    result.verify()
    return result


FROZEN_EVALUATION_PROPOSAL_SAMPLER_CONTRACT_SHA256 = (
    frozen_evaluation_proposal_sampler_contract()["sha256"]
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
    "FROZEN_EVALUATION_PROPOSAL_SAMPLER_CONTRACT_SHA256",
    "FrozenEvaluationProposal",
    "SamplerCompactionReceipt",
    "SamplerRetirePrefixBarrier",
    "SamplingMixture",
    "SamplingProfile",
    "frozen_evaluation_proposal_sampler_contract",
    "sample_frozen_evaluation_proposal",
]
