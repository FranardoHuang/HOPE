"""Strict metadata contract for action-conditioned ball-first training.

The producer order bound by this manifest is:

``action -> incoming-ball sample -> frozen ball-to-task solve -> attempt``.

Each ordered action owns one incoming-ball distribution.  The distribution is
centred on that action's best known ball and can be widened asynchronously by
the runtime curriculum.  This file intentionally contains no Torch or Isaac
imports so identity, units, bounds, and exact-resume inputs can be reviewed on
a host-only machine.

Two digests have deliberately different meanings:

* ``file_sha256`` binds the exact UTF-8 JSON bytes supplied at launch.
* ``canonical_sha256`` is formatting-independent and is useful for comparing
  two validated manifests.  It must not replace the exact-byte launch pin.

Schema v3 is metadata-only and deliberately has no self-reported authorization
field.  Passing ``require_formal_admission=True`` fails closed: code-rooted
motion admission remains the responsibility of the executable launch boundary,
separate from the referenced-byte checks provided here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Dict, Mapping, Optional, Sequence, Tuple
import unicodedata


SCHEMA_VERSION = 3
MAX_ACTION_UID = (1 << 53) - 1
MAX_HOLDOUT_SEED = (1 << 63) - 1

# The fore-aft contact uncertainty must remain small.  The table depth and
# timing solve make x qualitatively different from lateral/vertical reach, so
# silently using an isotropic large standard deviation is a schema error.
MAX_CONTACT_X_STD_M = 0.10
UNIT_VECTOR_TOLERANCE = 1.0e-6

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "manifest_id",
        "mobility_mode",
        "action_order",
        "prototype",
        "solver_profile_sha256",
        "physics_profile_sha256",
        "landing_aim",
        "actions",
        "curriculum",
        "holdout",
        "notes",
    )
)
_PROTOTYPE_KEYS = frozenset(("path", "sha256", "scope"))
_LANDING_AIM_KEYS = frozenset(
    (
        "center_w_xy_m",
        "std_lower_initial_m",
        "std_lower_max_m",
        "std_upper_initial_m",
        "std_upper_max_m",
        "min_w_xy_m",
        "max_w_xy_m",
    )
)
_ACTION_KEYS = frozenset(
    (
        "action_id",
        "action_uid",
        "motion_path",
        "motion_sha256",
        "strike_phase",
        "reference_t_hit_s",
        "reference_t_cycle_s",
        "reference_racket_site_speed_mps",
        "reaction_margin_s",
        "teacher_rate_min",
        "teacher_rate_max",
        "family",
        "mount_normal_sign",
        "ball_profile",
    )
)
_BALL_PROFILE_KEYS = frozenset(
    (
        "contact_offset_center_b_yaw_m",
        "contact_offset_std_lower_initial_m",
        "contact_offset_std_lower_max_m",
        "contact_offset_std_upper_initial_m",
        "contact_offset_std_upper_max_m",
        "contact_offset_min_b_yaw_m",
        "contact_offset_max_b_yaw_m",
        "time_to_contact_center_s",
        "time_to_contact_std_lower_initial_s",
        "time_to_contact_std_lower_max_s",
        "time_to_contact_std_upper_initial_s",
        "time_to_contact_std_upper_max_s",
        "time_to_contact_min_s",
        "time_to_contact_max_s",
        "incoming_direction_center_b_yaw",
        "incoming_direction_tangent_u_b_yaw",
        "incoming_direction_tangent_v_b_yaw",
        "incoming_direction_tangent_u_neg_initial_deg",
        "incoming_direction_tangent_u_neg_max_deg",
        "incoming_direction_tangent_u_pos_initial_deg",
        "incoming_direction_tangent_u_pos_max_deg",
        "incoming_direction_tangent_v_neg_initial_deg",
        "incoming_direction_tangent_v_neg_max_deg",
        "incoming_direction_tangent_v_pos_initial_deg",
        "incoming_direction_tangent_v_pos_max_deg",
        "incoming_inbound_axis_b_yaw",
        "incoming_inbound_min_cosine",
        "incoming_speed_center_mps",
        "incoming_speed_std_lower_initial_mps",
        "incoming_speed_std_lower_max_mps",
        "incoming_speed_std_upper_initial_mps",
        "incoming_speed_std_upper_max_mps",
        "incoming_speed_min_mps",
        "incoming_speed_max_mps",
        "spin_direction_center_b_yaw",
        "spin_direction_tangent_u_b_yaw",
        "spin_direction_tangent_v_b_yaw",
        "spin_direction_tangent_u_neg_initial_deg",
        "spin_direction_tangent_u_neg_max_deg",
        "spin_direction_tangent_u_pos_initial_deg",
        "spin_direction_tangent_u_pos_max_deg",
        "spin_direction_tangent_v_neg_initial_deg",
        "spin_direction_tangent_v_neg_max_deg",
        "spin_direction_tangent_v_pos_initial_deg",
        "spin_direction_tangent_v_pos_max_deg",
        "spin_magnitude_center_radps",
        "spin_magnitude_std_lower_initial_radps",
        "spin_magnitude_std_lower_max_radps",
        "spin_magnitude_std_upper_initial_radps",
        "spin_magnitude_std_upper_max_radps",
        "spin_magnitude_min_radps",
        "spin_magnitude_max_radps",
        "base_spawn_center_w_xy_m",
        "base_spawn_std_lower_initial_m",
        "base_spawn_std_lower_max_m",
        "base_spawn_std_upper_initial_m",
        "base_spawn_std_upper_max_m",
        "base_spawn_min_w_xy_m",
        "base_spawn_max_w_xy_m",
        "base_travel_center_b_yaw_xy_m",
        "base_travel_std_lower_initial_m",
        "base_travel_std_lower_max_m",
        "base_travel_std_upper_initial_m",
        "base_travel_std_upper_max_m",
        "base_travel_min_b_yaw_xy_m",
        "base_travel_max_b_yaw_xy_m",
    )
)
_CURRICULUM_KEYS = frozenset(
    (
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
)
_HOLDOUT_KEYS = frozenset(("seed", "samples_per_action", "split_id"))


class ActionBallManifestAdmissionError(ValueError):
    """Raised when metadata is mistaken for formal launch authorization."""


def _require_exact_keys(
    value: object,
    expected: frozenset,
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


def _require_identity_text(value: object, *, name: str) -> str:
    result = _require_string(value, name=name)
    if result != unicodedata.normalize("NFC", result):
        raise ValueError(f"{name} must use NFC Unicode normalization")
    if any(
        unicodedata.category(character) in ("Cc", "Cf", "Cs")
        for character in result
    ):
        raise ValueError(
            f"{name} must not contain control, format, or surrogate characters"
        )
    return result


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: Optional[int] = None,
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
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    # bool is a subclass of int and must be rejected explicitly at this
    # scientific/control boundary.
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
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
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
            maximum=maximum,
        )
        for index, component in enumerate(value)
    )


def _require_unit_vector(
    value: object,
    *,
    name: str,
) -> Tuple[float, float, float]:
    vector = _require_vector(value, name=name, length=3)
    norm = math.sqrt(sum(component * component for component in vector))
    if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
        raise ValueError(
            f"{name} must be unit length within "
            f"{UNIT_VECTOR_TOLERANCE:g}; got norm {norm:.12g}"
        )
    return vector  # type: ignore[return-value]


def _require_relative_posix_path(value: object, *, name: str) -> str:
    path = _require_string(value, name=name)
    if "\x00" in path:
        raise ValueError(f"{name} must not contain NUL")
    if "\\" in path:
        raise ValueError(f"{name} must use POSIX separators")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{name} must be relative")
    if ".." in posix.parts:
        raise ValueError(f"{name} must not contain '..'")
    if not posix.parts or all(part in ("", ".") for part in posix.parts):
        raise ValueError(f"{name} must identify a relative asset")
    if posix.as_posix() != path:
        raise ValueError(
            f"{name} must be a normalized relative POSIX path"
        )
    return path


def _require_initial_not_above_max(
    initial: Sequence[float],
    maximum: Sequence[float],
    *,
    name: str,
) -> None:
    for index, (initial_value, maximum_value) in enumerate(
        zip(initial, maximum)
    ):
        if initial_value > maximum_value:
            raise ValueError(
                f"{name} initial[{index}] must be <= max[{index}]"
            )


def _require_asymmetric_widths(
    lower_initial: Sequence[float],
    lower_maximum: Sequence[float],
    upper_initial: Sequence[float],
    upper_maximum: Sequence[float],
    *,
    name: str,
) -> None:
    _require_initial_not_above_max(
        lower_initial, lower_maximum, name=f"{name}.lower"
    )
    _require_initial_not_above_max(
        upper_initial, upper_maximum, name=f"{name}.upper"
    )


def _require_widths_inside_bounds(
    center: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    lower_maximum: Sequence[float],
    upper_maximum: Sequence[float],
    *,
    name: str,
) -> None:
    for index, values in enumerate(
        zip(center, lower, upper, lower_maximum, upper_maximum)
    ):
        center_value, lower_value, upper_value, lower_width, upper_width = (
            values
        )
        if lower_width > center_value - lower_value + 1.0e-12:
            raise ValueError(
                f"{name}.lower max[{index}] exceeds center-to-min support"
            )
        if upper_width > upper_value - center_value + 1.0e-12:
            raise ValueError(
                f"{name}.upper max[{index}] exceeds center-to-max support"
            )


def _require_center_inside_bounds(
    center: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    name: str,
) -> None:
    for index, (center_value, lower_value, upper_value) in enumerate(
        zip(center, lower, upper)
    ):
        if upper_value < lower_value:
            raise ValueError(
                f"{name}[{index}] upper bound must be >= lower bound"
            )
        if not lower_value <= center_value <= upper_value:
            raise ValueError(
                f"{name}[{index}] center must lie inside bounds"
            )


def _dot(
    left: Sequence[float], right: Sequence[float]
) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(
    left: Sequence[float], right: Sequence[float]
) -> Tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _require_tangent_frame(
    center: Tuple[float, float, float],
    tangent_u: Tuple[float, float, float],
    tangent_v: Tuple[float, float, float],
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
            f"{name} tangent frame must be right-handed: cross(u,v)=center"
        )


def _require_direction_widths(
    *,
    u_neg_initial: float,
    u_neg_max: float,
    u_pos_initial: float,
    u_pos_max: float,
    v_neg_initial: float,
    v_neg_max: float,
    v_pos_initial: float,
    v_pos_max: float,
    name: str,
) -> None:
    for side, initial, maximum in (
        ("u_neg", u_neg_initial, u_neg_max),
        ("u_pos", u_pos_initial, u_pos_max),
        ("v_neg", v_neg_initial, v_neg_max),
        ("v_pos", v_pos_initial, v_pos_max),
    ):
        if initial > maximum:
            raise ValueError(
                f"{name}.{side}_initial_deg must be <= "
                f"{side}_max_deg"
            )
    radial_max = math.hypot(
        max(u_neg_max, u_pos_max), max(v_neg_max, v_pos_max)
    )
    if radial_max > 180.0:
        raise ValueError(
            f"{name} maximum tangent envelope must be <= 180 degrees"
        )


def derive_action_ball_action_uid(
    action_id: str,
    family: str,
    motion_sha256: str,
) -> int:
    """Derive the planner-compatible positive, float64-exact action UID.

    The canonical payload is byte-for-byte equivalent to
    ``hope_planner.action_catalog.derive_action_uid``.  A manifest therefore
    cannot rename an action, change its family, or replace its motion bytes
    while keeping an old wire identity.
    """

    action_id_value = _require_identity_text(action_id, name="action_id")
    family_value = _require_identity_text(family, name="family")
    motion_sha256_value = _require_sha256(
        motion_sha256, name="motion_sha256"
    )
    identity = {
        "action_id": action_id_value,
        "content_sha256": motion_sha256_value,
        "family": family_value,
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return 1 + (int.from_bytes(digest, byteorder="big") % MAX_ACTION_UID)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant {value!r} is not allowed")


def _strict_json_object(
    pairs: Sequence[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class PrototypeBinding:
    """Exact shared action-prototype artifact used by the fixed solver."""

    path: str
    sha256: str
    scope: str

    @classmethod
    def from_mapping(cls, value: object) -> "PrototypeBinding":
        row = _require_exact_keys(
            value, _PROTOTYPE_KEYS, name="prototype"
        )
        return cls(
            path=_require_relative_posix_path(
                row["path"], name="prototype.path"
            ),
            sha256=_require_sha256(
                row["sha256"], name="prototype.sha256"
            ),
            scope=_require_identity_text(
                row["scope"], name="prototype.scope"
            ),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class LandingAimProfile:
    """Global opponent-side landing domain in environment-local ``W`` x/y."""

    center_w_xy_m: Tuple[float, float]
    std_lower_initial_m: Tuple[float, float]
    std_lower_max_m: Tuple[float, float]
    std_upper_initial_m: Tuple[float, float]
    std_upper_max_m: Tuple[float, float]
    min_w_xy_m: Tuple[float, float]
    max_w_xy_m: Tuple[float, float]

    @classmethod
    def from_mapping(cls, value: object) -> "LandingAimProfile":
        row = _require_exact_keys(
            value, _LANDING_AIM_KEYS, name="landing_aim"
        )
        center = _require_vector(
            row["center_w_xy_m"],
            name="landing_aim.center_w_xy_m",
            length=2,
        )
        lower_initial = _require_vector(
            row["std_lower_initial_m"],
            name="landing_aim.std_lower_initial_m",
            length=2,
            minimum=0.0,
        )
        lower_max = _require_vector(
            row["std_lower_max_m"],
            name="landing_aim.std_lower_max_m",
            length=2,
            minimum=0.0,
        )
        upper_initial = _require_vector(
            row["std_upper_initial_m"],
            name="landing_aim.std_upper_initial_m",
            length=2,
            minimum=0.0,
        )
        upper_max = _require_vector(
            row["std_upper_max_m"],
            name="landing_aim.std_upper_max_m",
            length=2,
            minimum=0.0,
        )
        lower = _require_vector(
            row["min_w_xy_m"],
            name="landing_aim.min_w_xy_m",
            length=2,
        )
        upper = _require_vector(
            row["max_w_xy_m"],
            name="landing_aim.max_w_xy_m",
            length=2,
        )
        _require_asymmetric_widths(
            lower_initial,
            lower_max,
            upper_initial,
            upper_max,
            name="landing_aim.std",
        )
        _require_center_inside_bounds(
            center, lower, upper, name="landing_aim"
        )
        _require_widths_inside_bounds(
            center,
            lower,
            upper,
            lower_max,
            upper_max,
            name="landing_aim.std",
        )
        return cls(
            center_w_xy_m=center,  # type: ignore[arg-type]
            std_lower_initial_m=lower_initial,  # type: ignore[arg-type]
            std_lower_max_m=lower_max,  # type: ignore[arg-type]
            std_upper_initial_m=upper_initial,  # type: ignore[arg-type]
            std_upper_max_m=upper_max,  # type: ignore[arg-type]
            min_w_xy_m=lower,  # type: ignore[arg-type]
            max_w_xy_m=upper,  # type: ignore[arg-type]
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "center_w_xy_m": list(self.center_w_xy_m),
            "std_lower_initial_m": list(self.std_lower_initial_m),
            "std_lower_max_m": list(self.std_lower_max_m),
            "std_upper_initial_m": list(self.std_upper_initial_m),
            "std_upper_max_m": list(self.std_upper_max_m),
            "min_w_xy_m": list(self.min_w_xy_m),
            "max_w_xy_m": list(self.max_w_xy_m),
        }


@dataclass(frozen=True)
class ActionBallProfile:
    """Best-ball centre plus initial and maximum per-action domain widths.

    Contact is a full three-dimensional offset from the sampled base *goal* in
    ``B_yaw``.  Base spawn is environment-local ``W`` x/y; canonical-ready
    fixes base z outside this sampler.  Travel is a latent goal displacement
    in ``B_yaw`` x/y.
    """

    contact_offset_center_b_yaw_m: Tuple[float, float, float]
    contact_offset_std_lower_initial_m: Tuple[float, float, float]
    contact_offset_std_lower_max_m: Tuple[float, float, float]
    contact_offset_std_upper_initial_m: Tuple[float, float, float]
    contact_offset_std_upper_max_m: Tuple[float, float, float]
    contact_offset_min_b_yaw_m: Tuple[float, float, float]
    contact_offset_max_b_yaw_m: Tuple[float, float, float]
    time_to_contact_center_s: float
    time_to_contact_std_lower_initial_s: float
    time_to_contact_std_lower_max_s: float
    time_to_contact_std_upper_initial_s: float
    time_to_contact_std_upper_max_s: float
    time_to_contact_min_s: float
    time_to_contact_max_s: float
    incoming_direction_center_b_yaw: Tuple[float, float, float]
    incoming_direction_tangent_u_b_yaw: Tuple[float, float, float]
    incoming_direction_tangent_v_b_yaw: Tuple[float, float, float]
    incoming_direction_tangent_u_neg_initial_deg: float
    incoming_direction_tangent_u_neg_max_deg: float
    incoming_direction_tangent_u_pos_initial_deg: float
    incoming_direction_tangent_u_pos_max_deg: float
    incoming_direction_tangent_v_neg_initial_deg: float
    incoming_direction_tangent_v_neg_max_deg: float
    incoming_direction_tangent_v_pos_initial_deg: float
    incoming_direction_tangent_v_pos_max_deg: float
    incoming_inbound_axis_b_yaw: Tuple[float, float, float]
    incoming_inbound_min_cosine: float
    incoming_speed_center_mps: float
    incoming_speed_std_lower_initial_mps: float
    incoming_speed_std_lower_max_mps: float
    incoming_speed_std_upper_initial_mps: float
    incoming_speed_std_upper_max_mps: float
    incoming_speed_min_mps: float
    incoming_speed_max_mps: float
    spin_direction_center_b_yaw: Tuple[float, float, float]
    spin_direction_tangent_u_b_yaw: Tuple[float, float, float]
    spin_direction_tangent_v_b_yaw: Tuple[float, float, float]
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
    base_spawn_center_w_xy_m: Tuple[float, float]
    base_spawn_std_lower_initial_m: Tuple[float, float]
    base_spawn_std_lower_max_m: Tuple[float, float]
    base_spawn_std_upper_initial_m: Tuple[float, float]
    base_spawn_std_upper_max_m: Tuple[float, float]
    base_spawn_min_w_xy_m: Tuple[float, float]
    base_spawn_max_w_xy_m: Tuple[float, float]
    base_travel_center_b_yaw_xy_m: Tuple[float, float]
    base_travel_std_lower_initial_m: Tuple[float, float]
    base_travel_std_lower_max_m: Tuple[float, float]
    base_travel_std_upper_initial_m: Tuple[float, float]
    base_travel_std_upper_max_m: Tuple[float, float]
    base_travel_min_b_yaw_xy_m: Tuple[float, float]
    base_travel_max_b_yaw_xy_m: Tuple[float, float]

    @classmethod
    def from_mapping(cls, value: object) -> "ActionBallProfile":
        row = _require_exact_keys(
            value, _BALL_PROFILE_KEYS, name="ball_profile"
        )
        contact_center = _require_vector(
            row["contact_offset_center_b_yaw_m"],
            name="contact_offset_center_b_yaw_m",
            length=3,
        )
        contact_lower_initial = _require_vector(
            row["contact_offset_std_lower_initial_m"],
            name="contact_offset_std_lower_initial_m",
            length=3,
            minimum=0.0,
        )
        contact_lower_max = _require_vector(
            row["contact_offset_std_lower_max_m"],
            name="contact_offset_std_lower_max_m",
            length=3,
            minimum=0.0,
        )
        contact_upper_initial = _require_vector(
            row["contact_offset_std_upper_initial_m"],
            name="contact_offset_std_upper_initial_m",
            length=3,
            minimum=0.0,
        )
        contact_upper_max = _require_vector(
            row["contact_offset_std_upper_max_m"],
            name="contact_offset_std_upper_max_m",
            length=3,
            minimum=0.0,
        )
        contact_lower = _require_vector(
            row["contact_offset_min_b_yaw_m"],
            name="contact_offset_min_b_yaw_m",
            length=3,
        )
        contact_upper = _require_vector(
            row["contact_offset_max_b_yaw_m"],
            name="contact_offset_max_b_yaw_m",
            length=3,
        )
        _require_asymmetric_widths(
            contact_lower_initial,
            contact_lower_max,
            contact_upper_initial,
            contact_upper_max,
            name="contact_offset_std",
        )
        _require_center_inside_bounds(
            contact_center,
            contact_lower,
            contact_upper,
            name="contact_offset",
        )
        _require_widths_inside_bounds(
            contact_center,
            contact_lower,
            contact_upper,
            contact_lower_max,
            contact_upper_max,
            name="contact_offset_std",
        )
        if max(contact_lower_max[0], contact_upper_max[0]) > (
            MAX_CONTACT_X_STD_M
        ):
            raise ValueError(
                "contact_offset x maximum std must be <= "
                f"{MAX_CONTACT_X_STD_M} m"
            )
        for side_name, initial, maximum in (
            ("lower", contact_lower_initial, contact_lower_max),
            ("upper", contact_upper_initial, contact_upper_max),
        ):
            if initial[0] > initial[1] or maximum[0] > maximum[1]:
                raise ValueError(
                    f"contact_offset {side_name} x std must not exceed y"
                )

        time_center = _require_finite(
            row["time_to_contact_center_s"],
            name="time_to_contact_center_s",
            minimum=0.0,
        )
        time_lower_initial = _require_finite(
            row["time_to_contact_std_lower_initial_s"],
            name="time_to_contact_std_lower_initial_s",
            minimum=0.0,
        )
        time_lower_max = _require_finite(
            row["time_to_contact_std_lower_max_s"],
            name="time_to_contact_std_lower_max_s",
            minimum=0.0,
        )
        time_upper_initial = _require_finite(
            row["time_to_contact_std_upper_initial_s"],
            name="time_to_contact_std_upper_initial_s",
            minimum=0.0,
        )
        time_upper_max = _require_finite(
            row["time_to_contact_std_upper_max_s"],
            name="time_to_contact_std_upper_max_s",
            minimum=0.0,
        )
        time_min = _require_finite(
            row["time_to_contact_min_s"],
            name="time_to_contact_min_s",
            minimum=0.0,
        )
        time_max = _require_finite(
            row["time_to_contact_max_s"],
            name="time_to_contact_max_s",
            minimum=0.0,
        )
        _require_asymmetric_widths(
            (time_lower_initial,),
            (time_lower_max,),
            (time_upper_initial,),
            (time_upper_max,),
            name="time_to_contact_std",
        )
        _require_center_inside_bounds(
            (time_center,), (time_min,), (time_max,),
            name="time_to_contact",
        )
        _require_widths_inside_bounds(
            (time_center,),
            (time_min,),
            (time_max,),
            (time_lower_max,),
            (time_upper_max,),
            name="time_to_contact_std",
        )

        incoming_center = _require_unit_vector(
            row["incoming_direction_center_b_yaw"],
            name="incoming_direction_center_b_yaw",
        )
        incoming_tangent_u = _require_unit_vector(
            row["incoming_direction_tangent_u_b_yaw"],
            name="incoming_direction_tangent_u_b_yaw",
        )
        incoming_tangent_v = _require_unit_vector(
            row["incoming_direction_tangent_v_b_yaw"],
            name="incoming_direction_tangent_v_b_yaw",
        )
        _require_tangent_frame(
            incoming_center,
            incoming_tangent_u,
            incoming_tangent_v,
            name="incoming_direction",
        )
        incoming_direction_widths = tuple(
            _require_finite(
                row[key], name=key, minimum=0.0, maximum=180.0
            )
            for key in (
                "incoming_direction_tangent_u_neg_initial_deg",
                "incoming_direction_tangent_u_neg_max_deg",
                "incoming_direction_tangent_u_pos_initial_deg",
                "incoming_direction_tangent_u_pos_max_deg",
                "incoming_direction_tangent_v_neg_initial_deg",
                "incoming_direction_tangent_v_neg_max_deg",
                "incoming_direction_tangent_v_pos_initial_deg",
                "incoming_direction_tangent_v_pos_max_deg",
            )
        )
        _require_direction_widths(
            u_neg_initial=incoming_direction_widths[0],
            u_neg_max=incoming_direction_widths[1],
            u_pos_initial=incoming_direction_widths[2],
            u_pos_max=incoming_direction_widths[3],
            v_neg_initial=incoming_direction_widths[4],
            v_neg_max=incoming_direction_widths[5],
            v_pos_initial=incoming_direction_widths[6],
            v_pos_max=incoming_direction_widths[7],
            name="incoming_direction",
        )
        inbound_axis = _require_unit_vector(
            row["incoming_inbound_axis_b_yaw"],
            name="incoming_inbound_axis_b_yaw",
        )
        inbound_min_cosine = _require_finite(
            row["incoming_inbound_min_cosine"],
            name="incoming_inbound_min_cosine",
            minimum=0.0,
            maximum=1.0,
        )
        if inbound_min_cosine >= 1.0:
            raise ValueError("incoming_inbound_min_cosine must be < 1")
        center_to_axis_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, _dot(
                incoming_center, inbound_axis
            ))))
        )
        max_tangent_radius_deg = math.hypot(
            max(incoming_direction_widths[1], incoming_direction_widths[3]),
            max(incoming_direction_widths[5], incoming_direction_widths[7]),
        )
        inbound_limit_deg = math.degrees(math.acos(inbound_min_cosine))
        if center_to_axis_deg + max_tangent_radius_deg > (
            inbound_limit_deg + UNIT_VECTOR_TOLERANCE
        ):
            raise ValueError(
                "incoming_direction maximum tangent support violates "
                "the inbound cone contract"
            )

        speed_center = _require_finite(
            row["incoming_speed_center_mps"],
            name="incoming_speed_center_mps",
            minimum=0.0,
        )
        speed_lower_initial = _require_finite(
            row["incoming_speed_std_lower_initial_mps"],
            name="incoming_speed_std_lower_initial_mps",
            minimum=0.0,
        )
        speed_lower_max = _require_finite(
            row["incoming_speed_std_lower_max_mps"],
            name="incoming_speed_std_lower_max_mps",
            minimum=0.0,
        )
        speed_upper_initial = _require_finite(
            row["incoming_speed_std_upper_initial_mps"],
            name="incoming_speed_std_upper_initial_mps",
            minimum=0.0,
        )
        speed_upper_max = _require_finite(
            row["incoming_speed_std_upper_max_mps"],
            name="incoming_speed_std_upper_max_mps",
            minimum=0.0,
        )
        speed_min = _require_finite(
            row["incoming_speed_min_mps"],
            name="incoming_speed_min_mps",
            minimum=0.0,
        )
        speed_max = _require_finite(
            row["incoming_speed_max_mps"],
            name="incoming_speed_max_mps",
            minimum=0.0,
        )
        if speed_max <= speed_min:
            raise ValueError(
                "incoming speed range must have max strictly above min"
            )
        if speed_center <= 0.0:
            raise ValueError("incoming_speed_center_mps must be > 0")
        if not speed_min <= speed_center <= speed_max:
            raise ValueError(
                "incoming_speed_center_mps must lie inside the speed range"
            )
        if not math.isclose(
            speed_min,
            0.4 * speed_center,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "incoming_speed_min_mps must equal exactly 0.4 times "
                "incoming_speed_center_mps"
            )
        _require_asymmetric_widths(
            (speed_lower_initial,),
            (speed_lower_max,),
            (speed_upper_initial,),
            (speed_upper_max,),
            name="incoming_speed_std",
        )
        _require_widths_inside_bounds(
            (speed_center,),
            (speed_min,),
            (speed_max,),
            (speed_lower_max,),
            (speed_upper_max,),
            name="incoming_speed_std",
        )

        spin_direction_center = _require_unit_vector(
            row["spin_direction_center_b_yaw"],
            name="spin_direction_center_b_yaw",
        )
        spin_tangent_u = _require_unit_vector(
            row["spin_direction_tangent_u_b_yaw"],
            name="spin_direction_tangent_u_b_yaw",
        )
        spin_tangent_v = _require_unit_vector(
            row["spin_direction_tangent_v_b_yaw"],
            name="spin_direction_tangent_v_b_yaw",
        )
        _require_tangent_frame(
            spin_direction_center,
            spin_tangent_u,
            spin_tangent_v,
            name="spin_direction",
        )
        spin_direction_widths = tuple(
            _require_finite(
                row[key], name=key, minimum=0.0, maximum=180.0
            )
            for key in (
                "spin_direction_tangent_u_neg_initial_deg",
                "spin_direction_tangent_u_neg_max_deg",
                "spin_direction_tangent_u_pos_initial_deg",
                "spin_direction_tangent_u_pos_max_deg",
                "spin_direction_tangent_v_neg_initial_deg",
                "spin_direction_tangent_v_neg_max_deg",
                "spin_direction_tangent_v_pos_initial_deg",
                "spin_direction_tangent_v_pos_max_deg",
            )
        )
        _require_direction_widths(
            u_neg_initial=spin_direction_widths[0],
            u_neg_max=spin_direction_widths[1],
            u_pos_initial=spin_direction_widths[2],
            u_pos_max=spin_direction_widths[3],
            v_neg_initial=spin_direction_widths[4],
            v_neg_max=spin_direction_widths[5],
            v_pos_initial=spin_direction_widths[6],
            v_pos_max=spin_direction_widths[7],
            name="spin_direction",
        )

        spin_center = _require_finite(
            row["spin_magnitude_center_radps"],
            name="spin_magnitude_center_radps",
            minimum=0.0,
        )
        spin_lower_initial = _require_finite(
            row["spin_magnitude_std_lower_initial_radps"],
            name="spin_magnitude_std_lower_initial_radps",
            minimum=0.0,
        )
        spin_lower_max = _require_finite(
            row["spin_magnitude_std_lower_max_radps"],
            name="spin_magnitude_std_lower_max_radps",
            minimum=0.0,
        )
        spin_upper_initial = _require_finite(
            row["spin_magnitude_std_upper_initial_radps"],
            name="spin_magnitude_std_upper_initial_radps",
            minimum=0.0,
        )
        spin_upper_max = _require_finite(
            row["spin_magnitude_std_upper_max_radps"],
            name="spin_magnitude_std_upper_max_radps",
            minimum=0.0,
        )
        spin_min = _require_finite(
            row["spin_magnitude_min_radps"],
            name="spin_magnitude_min_radps",
            minimum=0.0,
        )
        spin_max = _require_finite(
            row["spin_magnitude_max_radps"],
            name="spin_magnitude_max_radps",
            minimum=0.0,
        )
        if spin_max < spin_min:
            raise ValueError(
                "spin magnitude range must have max >= min"
            )
        if not spin_min <= spin_center <= spin_max:
            raise ValueError(
                "spin_magnitude_center_radps must lie inside the spin range"
            )
        _require_asymmetric_widths(
            (spin_lower_initial,),
            (spin_lower_max,),
            (spin_upper_initial,),
            (spin_upper_max,),
            name="spin_magnitude_std",
        )
        _require_widths_inside_bounds(
            (spin_center,),
            (spin_min,),
            (spin_max,),
            (spin_lower_max,),
            (spin_upper_max,),
            name="spin_magnitude_std",
        )

        base_spawn_center = _require_vector(
            row["base_spawn_center_w_xy_m"],
            name="base_spawn_center_w_xy_m",
            length=2,
        )
        base_spawn_lower_initial = _require_vector(
            row["base_spawn_std_lower_initial_m"],
            name="base_spawn_std_lower_initial_m",
            length=2,
            minimum=0.0,
        )
        base_spawn_lower_max = _require_vector(
            row["base_spawn_std_lower_max_m"],
            name="base_spawn_std_lower_max_m",
            length=2,
            minimum=0.0,
        )
        base_spawn_upper_initial = _require_vector(
            row["base_spawn_std_upper_initial_m"],
            name="base_spawn_std_upper_initial_m",
            length=2,
            minimum=0.0,
        )
        base_spawn_upper_max = _require_vector(
            row["base_spawn_std_upper_max_m"],
            name="base_spawn_std_upper_max_m",
            length=2,
            minimum=0.0,
        )
        base_spawn_lower = _require_vector(
            row["base_spawn_min_w_xy_m"],
            name="base_spawn_min_w_xy_m",
            length=2,
        )
        base_spawn_upper = _require_vector(
            row["base_spawn_max_w_xy_m"],
            name="base_spawn_max_w_xy_m",
            length=2,
        )
        _require_asymmetric_widths(
            base_spawn_lower_initial,
            base_spawn_lower_max,
            base_spawn_upper_initial,
            base_spawn_upper_max,
            name="base_spawn_std",
        )
        _require_center_inside_bounds(
            base_spawn_center,
            base_spawn_lower,
            base_spawn_upper,
            name="base_spawn",
        )
        _require_widths_inside_bounds(
            base_spawn_center,
            base_spawn_lower,
            base_spawn_upper,
            base_spawn_lower_max,
            base_spawn_upper_max,
            name="base_spawn_std",
        )

        base_travel_center = _require_vector(
            row["base_travel_center_b_yaw_xy_m"],
            name="base_travel_center_b_yaw_xy_m",
            length=2,
        )
        base_travel_lower_initial = _require_vector(
            row["base_travel_std_lower_initial_m"],
            name="base_travel_std_lower_initial_m",
            length=2,
            minimum=0.0,
        )
        base_travel_lower_max = _require_vector(
            row["base_travel_std_lower_max_m"],
            name="base_travel_std_lower_max_m",
            length=2,
            minimum=0.0,
        )
        base_travel_upper_initial = _require_vector(
            row["base_travel_std_upper_initial_m"],
            name="base_travel_std_upper_initial_m",
            length=2,
            minimum=0.0,
        )
        base_travel_upper_max = _require_vector(
            row["base_travel_std_upper_max_m"],
            name="base_travel_std_upper_max_m",
            length=2,
            minimum=0.0,
        )
        base_travel_lower = _require_vector(
            row["base_travel_min_b_yaw_xy_m"],
            name="base_travel_min_b_yaw_xy_m",
            length=2,
        )
        base_travel_upper = _require_vector(
            row["base_travel_max_b_yaw_xy_m"],
            name="base_travel_max_b_yaw_xy_m",
            length=2,
        )
        _require_asymmetric_widths(
            base_travel_lower_initial,
            base_travel_lower_max,
            base_travel_upper_initial,
            base_travel_upper_max,
            name="base_travel_std",
        )
        _require_center_inside_bounds(
            base_travel_center,
            base_travel_lower,
            base_travel_upper,
            name="base_travel",
        )
        _require_widths_inside_bounds(
            base_travel_center,
            base_travel_lower,
            base_travel_upper,
            base_travel_lower_max,
            base_travel_upper_max,
            name="base_travel_std",
        )

        return cls(
            contact_offset_center_b_yaw_m=contact_center,  # type: ignore[arg-type]
            contact_offset_std_lower_initial_m=contact_lower_initial,  # type: ignore[arg-type]
            contact_offset_std_lower_max_m=contact_lower_max,  # type: ignore[arg-type]
            contact_offset_std_upper_initial_m=contact_upper_initial,  # type: ignore[arg-type]
            contact_offset_std_upper_max_m=contact_upper_max,  # type: ignore[arg-type]
            contact_offset_min_b_yaw_m=contact_lower,  # type: ignore[arg-type]
            contact_offset_max_b_yaw_m=contact_upper,  # type: ignore[arg-type]
            time_to_contact_center_s=time_center,
            time_to_contact_std_lower_initial_s=time_lower_initial,
            time_to_contact_std_lower_max_s=time_lower_max,
            time_to_contact_std_upper_initial_s=time_upper_initial,
            time_to_contact_std_upper_max_s=time_upper_max,
            time_to_contact_min_s=time_min,
            time_to_contact_max_s=time_max,
            incoming_direction_center_b_yaw=incoming_center,
            incoming_direction_tangent_u_b_yaw=incoming_tangent_u,
            incoming_direction_tangent_v_b_yaw=incoming_tangent_v,
            incoming_direction_tangent_u_neg_initial_deg=incoming_direction_widths[0],
            incoming_direction_tangent_u_neg_max_deg=incoming_direction_widths[1],
            incoming_direction_tangent_u_pos_initial_deg=incoming_direction_widths[2],
            incoming_direction_tangent_u_pos_max_deg=incoming_direction_widths[3],
            incoming_direction_tangent_v_neg_initial_deg=incoming_direction_widths[4],
            incoming_direction_tangent_v_neg_max_deg=incoming_direction_widths[5],
            incoming_direction_tangent_v_pos_initial_deg=incoming_direction_widths[6],
            incoming_direction_tangent_v_pos_max_deg=incoming_direction_widths[7],
            incoming_inbound_axis_b_yaw=inbound_axis,
            incoming_inbound_min_cosine=inbound_min_cosine,
            incoming_speed_center_mps=speed_center,
            incoming_speed_std_lower_initial_mps=speed_lower_initial,
            incoming_speed_std_lower_max_mps=speed_lower_max,
            incoming_speed_std_upper_initial_mps=speed_upper_initial,
            incoming_speed_std_upper_max_mps=speed_upper_max,
            incoming_speed_min_mps=speed_min,
            incoming_speed_max_mps=speed_max,
            spin_direction_center_b_yaw=spin_direction_center,
            spin_direction_tangent_u_b_yaw=spin_tangent_u,
            spin_direction_tangent_v_b_yaw=spin_tangent_v,
            spin_direction_tangent_u_neg_initial_deg=spin_direction_widths[0],
            spin_direction_tangent_u_neg_max_deg=spin_direction_widths[1],
            spin_direction_tangent_u_pos_initial_deg=spin_direction_widths[2],
            spin_direction_tangent_u_pos_max_deg=spin_direction_widths[3],
            spin_direction_tangent_v_neg_initial_deg=spin_direction_widths[4],
            spin_direction_tangent_v_neg_max_deg=spin_direction_widths[5],
            spin_direction_tangent_v_pos_initial_deg=spin_direction_widths[6],
            spin_direction_tangent_v_pos_max_deg=spin_direction_widths[7],
            spin_magnitude_center_radps=spin_center,
            spin_magnitude_std_lower_initial_radps=spin_lower_initial,
            spin_magnitude_std_lower_max_radps=spin_lower_max,
            spin_magnitude_std_upper_initial_radps=spin_upper_initial,
            spin_magnitude_std_upper_max_radps=spin_upper_max,
            spin_magnitude_min_radps=spin_min,
            spin_magnitude_max_radps=spin_max,
            base_spawn_center_w_xy_m=base_spawn_center,  # type: ignore[arg-type]
            base_spawn_std_lower_initial_m=base_spawn_lower_initial,  # type: ignore[arg-type]
            base_spawn_std_lower_max_m=base_spawn_lower_max,  # type: ignore[arg-type]
            base_spawn_std_upper_initial_m=base_spawn_upper_initial,  # type: ignore[arg-type]
            base_spawn_std_upper_max_m=base_spawn_upper_max,  # type: ignore[arg-type]
            base_spawn_min_w_xy_m=base_spawn_lower,  # type: ignore[arg-type]
            base_spawn_max_w_xy_m=base_spawn_upper,  # type: ignore[arg-type]
            base_travel_center_b_yaw_xy_m=base_travel_center,  # type: ignore[arg-type]
            base_travel_std_lower_initial_m=base_travel_lower_initial,  # type: ignore[arg-type]
            base_travel_std_lower_max_m=base_travel_lower_max,  # type: ignore[arg-type]
            base_travel_std_upper_initial_m=base_travel_upper_initial,  # type: ignore[arg-type]
            base_travel_std_upper_max_m=base_travel_upper_max,  # type: ignore[arg-type]
            base_travel_min_b_yaw_xy_m=base_travel_lower,  # type: ignore[arg-type]
            base_travel_max_b_yaw_xy_m=base_travel_upper,  # type: ignore[arg-type]
        )

    def to_mapping(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for name in _BALL_PROFILE_KEYS:
            value = getattr(self, name)
            result[name] = list(value) if isinstance(value, tuple) else value
        return result


@dataclass(frozen=True)
class ActionBallAction:
    """One ordered, content-bound action and its incoming-ball profile."""

    action_id: str
    action_uid: int
    motion_path: str
    motion_sha256: str
    strike_phase: float
    reference_t_hit_s: float
    reference_t_cycle_s: float
    reference_racket_site_speed_mps: float
    reaction_margin_s: float
    teacher_rate_min: float
    teacher_rate_max: float
    family: str
    mount_normal_sign: int
    ball_profile: ActionBallProfile

    @classmethod
    def from_mapping(cls, value: object) -> "ActionBallAction":
        row = _require_exact_keys(value, _ACTION_KEYS, name="action")
        action_id = _require_identity_text(
            row["action_id"], name="action_id"
        )
        family = _require_identity_text(row["family"], name="family")
        motion_sha256 = _require_sha256(
            row["motion_sha256"], name="motion_sha256"
        )
        action_uid = _require_int(
            row["action_uid"],
            name="action_uid",
            minimum=1,
            maximum=MAX_ACTION_UID,
        )
        expected_uid = derive_action_ball_action_uid(
            action_id, family, motion_sha256
        )
        if action_uid != expected_uid:
            raise ValueError(
                f"action_uid {action_uid} does not match canonical action "
                f"identity (expected {expected_uid})"
            )
        strike_phase = _require_finite(
            row["strike_phase"],
            name="strike_phase",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < strike_phase < 1.0:
            raise ValueError(
                "strike_phase must lie strictly inside (0, 1)"
            )
        reference_t_hit_s = _require_finite(
            row["reference_t_hit_s"],
            name="reference_t_hit_s",
            minimum=0.0,
        )
        reference_t_cycle_s = _require_finite(
            row["reference_t_cycle_s"],
            name="reference_t_cycle_s",
            minimum=0.0,
        )
        if reference_t_hit_s <= 0.0:
            raise ValueError("reference_t_hit_s must be > 0")
        if reference_t_cycle_s <= reference_t_hit_s:
            raise ValueError(
                "reference_t_cycle_s must be > reference_t_hit_s"
            )
        reference_racket_site_speed_mps = _require_finite(
            row["reference_racket_site_speed_mps"],
            name="reference_racket_site_speed_mps",
            minimum=0.0,
        )
        if reference_racket_site_speed_mps <= 0.0:
            raise ValueError(
                "reference_racket_site_speed_mps must be > 0"
            )
        reaction_margin_s = _require_finite(
            row["reaction_margin_s"],
            name="reaction_margin_s",
            minimum=0.0,
        )
        teacher_rate_min = _require_finite(
            row["teacher_rate_min"],
            name="teacher_rate_min",
            minimum=0.0,
        )
        teacher_rate_max = _require_finite(
            row["teacher_rate_max"],
            name="teacher_rate_max",
            minimum=0.0,
        )
        if teacher_rate_min <= 0.0:
            raise ValueError("teacher_rate_min must be > 0")
        if not teacher_rate_min <= 1.0 <= teacher_rate_max:
            raise ValueError(
                "teacher rate range must contain the native rate 1.0"
            )
        ball_profile = ActionBallProfile.from_mapping(row["ball_profile"])
        slowest_t_hit_s = reference_t_hit_s / teacher_rate_min
        if ball_profile.time_to_contact_min_s < (
            slowest_t_hit_s + reaction_margin_s
        ):
            raise ValueError(
                "time_to_contact_min_s must be >= "
                "reference_t_hit_s / teacher_rate_min + reaction_margin_s"
            )
        fastest_t_hit_s = reference_t_hit_s / teacher_rate_max
        if ball_profile.time_to_contact_max_s - fastest_t_hit_s > (
            1.0 + 1.0e-12
        ):
            raise ValueError(
                "maximum pre_swing_wait must be <= 1.0 s for every "
                "allowed teacher rate"
            )
        mount_normal_sign = _require_int(
            row["mount_normal_sign"],
            name="mount_normal_sign",
            minimum=-1,
            maximum=1,
        )
        if mount_normal_sign not in (-1, 1):
            raise ValueError("mount_normal_sign must be +1 or -1")
        return cls(
            action_id=action_id,
            action_uid=action_uid,
            motion_path=_require_relative_posix_path(
                row["motion_path"], name="motion_path"
            ),
            motion_sha256=motion_sha256,
            strike_phase=strike_phase,
            reference_t_hit_s=reference_t_hit_s,
            reference_t_cycle_s=reference_t_cycle_s,
            reference_racket_site_speed_mps=(
                reference_racket_site_speed_mps
            ),
            reaction_margin_s=reaction_margin_s,
            teacher_rate_min=teacher_rate_min,
            teacher_rate_max=teacher_rate_max,
            family=family,
            mount_normal_sign=mount_normal_sign,
            ball_profile=ball_profile,
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_uid": self.action_uid,
            "motion_path": self.motion_path,
            "motion_sha256": self.motion_sha256,
            "strike_phase": self.strike_phase,
            "reference_t_hit_s": self.reference_t_hit_s,
            "reference_t_cycle_s": self.reference_t_cycle_s,
            "reference_racket_site_speed_mps": (
                self.reference_racket_site_speed_mps
            ),
            "reaction_margin_s": self.reaction_margin_s,
            "teacher_rate_min": self.teacher_rate_min,
            "teacher_rate_max": self.teacher_rate_max,
            "family": self.family,
            "mount_normal_sign": self.mount_normal_sign,
            "ball_profile": self.ball_profile.to_mapping(),
        }


@dataclass(frozen=True)
class ActionBallCurriculumConfig:
    """Per-action feedback rule; failures mean safe-valid policy failures.

    The default target is 10% with a +/-2.5 percentage-point band.  A 20%
    ablation is represented by another manifest, so it necessarily receives a
    different exact-byte and canonical digest.
    """

    min_proposals: int = 256
    min_safe_closed: int = 256
    target_failure_rate: float = 0.10
    failure_band_half_width: float = 0.025
    min_solver_admit_rate: float = 0.95
    min_install_rate: float = 0.95
    min_start_rate: float = 0.95
    min_close_rate: float = 0.95
    max_other_unsafe_rate: float = 0.02
    confidence_z: float = 1.96
    max_center_failures: int = 8

    def __post_init__(self) -> None:
        _require_int(
            self.min_proposals,
            name="curriculum.min_proposals",
            minimum=1,
        )
        _require_int(
            self.min_safe_closed,
            name="curriculum.min_safe_closed",
            minimum=1,
        )
        target = _require_finite(
            self.target_failure_rate,
            name="curriculum.target_failure_rate",
            minimum=0.0,
            maximum=1.0,
        )
        half_width = _require_finite(
            self.failure_band_half_width,
            name="curriculum.failure_band_half_width",
            minimum=0.0,
            maximum=0.5,
        )
        if target - half_width < 0.0 or target + half_width > 1.0:
            raise ValueError(
                "curriculum target failure band must lie inside [0, 1]"
            )
        for name in (
            "min_solver_admit_rate",
            "min_install_rate",
            "min_start_rate",
            "min_close_rate",
            "max_other_unsafe_rate",
        ):
            _require_finite(
                getattr(self, name),
                name=f"curriculum.{name}",
                minimum=0.0,
                maximum=1.0,
            )
        _require_finite(
            self.confidence_z,
            name="curriculum.confidence_z",
            minimum=0.0,
        )
        _require_int(
            self.max_center_failures,
            name="curriculum.max_center_failures",
            minimum=1,
        )

    @classmethod
    def from_mapping(
        cls, value: object
    ) -> "ActionBallCurriculumConfig":
        row = _require_exact_keys(
            value, _CURRICULUM_KEYS, name="curriculum"
        )
        return cls(
            min_proposals=_require_int(
                row["min_proposals"],
                name="curriculum.min_proposals",
                minimum=1,
            ),
            min_safe_closed=_require_int(
                row["min_safe_closed"],
                name="curriculum.min_safe_closed",
                minimum=1,
            ),
            target_failure_rate=_require_finite(
                row["target_failure_rate"],
                name="curriculum.target_failure_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            failure_band_half_width=_require_finite(
                row["failure_band_half_width"],
                name="curriculum.failure_band_half_width",
                minimum=0.0,
                maximum=0.5,
            ),
            min_solver_admit_rate=_require_finite(
                row["min_solver_admit_rate"],
                name="curriculum.min_solver_admit_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            min_install_rate=_require_finite(
                row["min_install_rate"],
                name="curriculum.min_install_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            min_start_rate=_require_finite(
                row["min_start_rate"],
                name="curriculum.min_start_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            min_close_rate=_require_finite(
                row["min_close_rate"],
                name="curriculum.min_close_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            max_other_unsafe_rate=_require_finite(
                row["max_other_unsafe_rate"],
                name="curriculum.max_other_unsafe_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            confidence_z=_require_finite(
                row["confidence_z"],
                name="curriculum.confidence_z",
                minimum=0.0,
            ),
            max_center_failures=_require_int(
                row["max_center_failures"],
                name="curriculum.max_center_failures",
                minimum=1,
            ),
        )

    @property
    def failure_band(self) -> Tuple[float, float]:
        return (
            self.target_failure_rate - self.failure_band_half_width,
            self.target_failure_rate + self.failure_band_half_width,
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "min_proposals": self.min_proposals,
            "min_safe_closed": self.min_safe_closed,
            "target_failure_rate": self.target_failure_rate,
            "failure_band_half_width": self.failure_band_half_width,
            "min_solver_admit_rate": self.min_solver_admit_rate,
            "min_install_rate": self.min_install_rate,
            "min_start_rate": self.min_start_rate,
            "min_close_rate": self.min_close_rate,
            "max_other_unsafe_rate": self.max_other_unsafe_rate,
            "confidence_z": self.confidence_z,
            "max_center_failures": self.max_center_failures,
        }


@dataclass(frozen=True)
class HoldoutConfig:
    """Deterministic held-out ball split evaluated for every action."""

    seed: int
    samples_per_action: int
    split_id: str

    @classmethod
    def from_mapping(cls, value: object) -> "HoldoutConfig":
        row = _require_exact_keys(
            value, _HOLDOUT_KEYS, name="holdout"
        )
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
            split_id=_require_identity_text(
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
class ActionBallManifest:
    """Validated immutable action-conditioned ball-first declaration."""

    schema_version: int
    manifest_id: str
    mobility_mode: str
    action_order: Tuple[str, ...]
    prototype: PrototypeBinding
    solver_profile_sha256: str
    physics_profile_sha256: str
    landing_aim: LandingAimProfile
    actions: Tuple[ActionBallAction, ...]
    curriculum: ActionBallCurriculumConfig
    holdout: HoldoutConfig
    notes: str

    @classmethod
    def from_mapping(cls, value: object) -> "ActionBallManifest":
        document = _require_exact_keys(
            value,
            _TOP_LEVEL_KEYS,
            name="action-conditioned ball-first manifest",
        )
        schema_version = _require_int(
            document["schema_version"],
            name="schema_version",
            minimum=SCHEMA_VERSION,
            maximum=SCHEMA_VERSION,
        )
        mobility_mode = _require_string(
            document["mobility_mode"], name="mobility_mode"
        )
        if mobility_mode not in ("no_move", "move"):
            raise ValueError("mobility_mode must be 'no_move' or 'move'")

        raw_order = document["action_order"]
        if (
            isinstance(raw_order, (str, bytes))
            or not isinstance(raw_order, Sequence)
        ):
            raise ValueError("action_order must be an array")
        action_order = tuple(
            _require_identity_text(
                action_id, name=f"action_order[{index}]"
            )
            for index, action_id in enumerate(raw_order)
        )
        if not action_order:
            raise ValueError(
                "action_order must contain at least one action"
            )
        if len(set(action_order)) != len(action_order):
            raise ValueError(
                "action_order must not contain duplicate action IDs"
            )

        raw_actions = document["actions"]
        if (
            isinstance(raw_actions, (str, bytes))
            or not isinstance(raw_actions, Sequence)
        ):
            raise ValueError("actions must be an array")
        actions = tuple(
            ActionBallAction.from_mapping(row) for row in raw_actions
        )
        action_ids = tuple(action.action_id for action in actions)
        if action_ids != action_order:
            raise ValueError(
                "actions must have exactly the same IDs and order as "
                "action_order"
            )
        action_uids = tuple(action.action_uid for action in actions)
        if len(set(action_uids)) != len(action_uids):
            raise ValueError(
                "actions must not contain duplicate action_uid values"
            )

        curriculum = ActionBallCurriculumConfig.from_mapping(
            document["curriculum"]
        )
        holdout = HoldoutConfig.from_mapping(document["holdout"])
        required_holdout = max(
            curriculum.min_proposals,
            curriculum.min_safe_closed,
        )
        if holdout.samples_per_action < required_holdout:
            raise ValueError(
                "holdout.samples_per_action must cover the manifest "
                "curriculum decision window: "
                f"need at least {required_holdout}, got "
                f"{holdout.samples_per_action}"
            )

        return cls(
            schema_version=schema_version,
            manifest_id=_require_identity_text(
                document["manifest_id"], name="manifest_id"
            ),
            mobility_mode=mobility_mode,
            action_order=action_order,
            prototype=PrototypeBinding.from_mapping(
                document["prototype"]
            ),
            solver_profile_sha256=_require_sha256(
                document["solver_profile_sha256"],
                name="solver_profile_sha256",
            ),
            physics_profile_sha256=_require_sha256(
                document["physics_profile_sha256"],
                name="physics_profile_sha256",
            ),
            landing_aim=LandingAimProfile.from_mapping(
                document["landing_aim"]
            ),
            actions=actions,
            curriculum=curriculum,
            holdout=holdout,
            notes=_require_string(
                document["notes"], name="notes", allow_empty=True
            ),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "mobility_mode": self.mobility_mode,
            "action_order": list(self.action_order),
            "prototype": self.prototype.to_mapping(),
            "solver_profile_sha256": self.solver_profile_sha256,
            "physics_profile_sha256": self.physics_profile_sha256,
            "landing_aim": self.landing_aim.to_mapping(),
            "actions": [action.to_mapping() for action in self.actions],
            "curriculum": self.curriculum.to_mapping(),
            "holdout": self.holdout.to_mapping(),
            "notes": self.notes,
        }


def canonical_manifest_bytes(manifest: ActionBallManifest) -> bytes:
    """Return the stable content encoding used only for comparison."""

    if not isinstance(manifest, ActionBallManifest):
        raise TypeError("manifest must be an ActionBallManifest")
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


def canonical_manifest_sha256(manifest: ActionBallManifest) -> str:
    """Return a formatting-independent digest of validated content."""

    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


@dataclass(frozen=True)
class VerifiedReferencedAsset:
    """One referenced regular file verified below the trusted repository root."""

    label: str
    relative_path: str
    resolved_path: Path
    sha256: str


@dataclass(frozen=True)
class VerifiedActionBallAssets:
    """Ordered byte-verification receipt for prototype and motion assets."""

    repo_root: Path
    prototype: VerifiedReferencedAsset
    motions: Tuple[VerifiedReferencedAsset, ...]


def _verified_referenced_asset(
    *,
    repo_root: Path,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> VerifiedReferencedAsset:
    relative = _require_relative_posix_path(
        relative_path, name=f"{label}.path"
    )
    expected = _require_sha256(
        expected_sha256, name=f"{label}.sha256"
    )
    candidate = repo_root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"{label} referenced asset does not resolve: {relative!r}"
        ) from error
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(
            f"{label} referenced asset escapes repo_root through a "
            f"symlink: {relative!r}"
        ) from error

    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise ValueError(
            f"{label} referenced asset cannot be stat'ed: {relative!r}"
        ) from error
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"{label} referenced asset must resolve to a regular file: "
            f"{relative!r}"
        )

    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            # Recheck the opened object, rather than trusting only the path
            # lookup above.  This closes the common final-component swap.
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError(
                    f"{label} referenced asset must be a regular file"
                )
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(
            f"{label} referenced asset cannot be read: {relative!r}"
        ) from error
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"{label} referenced asset SHA-256 mismatch: "
            f"expected {expected}, got {actual}"
        )
    return VerifiedReferencedAsset(
        label=label,
        relative_path=relative,
        resolved_path=resolved,
        sha256=actual,
    )


def verify_action_ball_referenced_assets(
    manifest: ActionBallManifest,
    *,
    repo_root: object,
) -> VerifiedActionBallAssets:
    """Verify prototype and every ordered motion against repository bytes.

    The manifest paths are exact normalized POSIX-relative paths.  Each path is
    resolved under ``repo_root``; a symlink may point elsewhere inside that
    root, but a symlink escape is rejected.  The opened target must be a
    regular file and its exact bytes must match the manifest SHA.

    This proves only byte identity.  It does not mint the code-rooted motion
    admission that the executable launch boundary must require separately.
    """

    if not isinstance(manifest, ActionBallManifest):
        raise TypeError("manifest must be an ActionBallManifest")
    try:
        root = Path(repo_root).resolve(strict=True)  # type: ignore[arg-type]
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("repo_root must resolve to an existing directory") from error
    if not root.is_dir():
        raise ValueError("repo_root must resolve to an existing directory")

    prototype = _verified_referenced_asset(
        repo_root=root,
        relative_path=manifest.prototype.path,
        expected_sha256=manifest.prototype.sha256,
        label="prototype",
    )
    motions = tuple(
        _verified_referenced_asset(
            repo_root=root,
            relative_path=action.motion_path,
            expected_sha256=action.motion_sha256,
            label=f"motion[{action.action_id}]",
        )
        for action in manifest.actions
    )
    return VerifiedActionBallAssets(
        repo_root=root,
        prototype=prototype,
        motions=motions,
    )


@dataclass(frozen=True)
class LoadedActionBallManifest:
    """Validated manifest plus exact-byte and canonical-content receipts."""

    manifest: ActionBallManifest
    source_path: Path
    file_sha256: str
    canonical_sha256: str
    referenced_assets: Optional[VerifiedActionBallAssets] = None


def load_action_ball_manifest(
    path: object,
    *,
    expected_sha256: Optional[str] = None,
    verify_referenced_assets: bool = False,
    repo_root: Optional[object] = None,
    require_formal_admission: bool = False,
) -> LoadedActionBallManifest:
    """Read, exact-byte bind, and strictly validate one schema-v3 manifest.

    Review tooling may omit ``expected_sha256`` and referenced-asset checks.
    Preflight callers can set ``verify_referenced_assets=True`` and provide the
    trusted ``repo_root`` to bind prototype and motion bytes.  An executable
    launch must additionally obtain code-rooted motion admission outside this
    metadata schema; requesting it here deliberately fails closed.
    """

    if type(require_formal_admission) is not bool:
        raise ValueError("require_formal_admission must be a bool")
    if type(verify_referenced_assets) is not bool:
        raise ValueError("verify_referenced_assets must be a bool")
    if verify_referenced_assets and repo_root is None:
        raise ValueError(
            "verify_referenced_assets=True requires repo_root"
        )
    if not verify_referenced_assets and repo_root is not None:
        raise ValueError(
            "repo_root is only accepted when "
            "verify_referenced_assets=True"
        )
    expected = (
        None
        if expected_sha256 is None
        else _require_sha256(
            expected_sha256, name="expected_sha256"
        )
    )
    source_path = Path(path)  # type: ignore[arg-type]
    raw = source_path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected is not None and file_sha256 != expected:
        raise ValueError(
            "action-ball manifest file SHA-256 mismatch: "
            f"expected {expected}, got {file_sha256}"
        )

    try:
        document = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except UnicodeDecodeError as error:
        raise ValueError(
            "action-ball manifest must be UTF-8 JSON"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            "action-ball manifest is not valid JSON"
        ) from error

    manifest = ActionBallManifest.from_mapping(document)
    assets = (
        verify_action_ball_referenced_assets(
            manifest, repo_root=repo_root
        )
        if verify_referenced_assets
        else None
    )
    if require_formal_admission:
        raise ActionBallManifestAdmissionError(
            "action-ball schema v2 is metadata-only: "
            "referenced-byte verification is not code-rooted motion "
            "admission; the executable launch boundary must verify its "
            "opaque motion admission capability separately"
        )
    return LoadedActionBallManifest(
        manifest=manifest,
        source_path=source_path,
        file_sha256=file_sha256,
        canonical_sha256=canonical_manifest_sha256(manifest),
        referenced_assets=assets,
    )
