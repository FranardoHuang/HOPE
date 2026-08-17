"""Engine-neutral landing-placement reward semantics.

This module is the canonical, standard-library-only definition of the next
ActionBall landing objective.  It intentionally consumes no return-direction
vector, ball speed, baseline state, or post-bounce state.  A producer supplies
only the first ball-centre crossing of the landing plane plus selected-rubber
contact and ball-centre net facts.

The scientific parameters have no module default.  A caller must explicitly
construct :class:`LandingPlacementProfile`, whose canonical SHA binds the
coordinate frame, table/net geometry, mixture weight, kernel scales, and the
fixed ``1.0``/``0.5`` table gates.  The target is per-attempt question state;
it is bound by :class:`LandingPlacementTaskIdentity`, while outcome facts
reference that identity's canonical SHA.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from numbers import Real
from typing import Mapping, Optional


SCHEMA_VERSION = 1
PROFILE_KIND = "action_ball_landing_placement_profile_v1"
TASK_IDENTITY_KIND = "action_ball_landing_placement_task_identity_v1"
FACTS_KIND = "action_ball_landing_placement_facts_v1"
SCORE_KIND = "action_ball_landing_placement_score_v1"
CAUCHY_DEFINITION = "1/(1+(distance_m/sigma_broad_m)^2)"
GAUSSIAN_DEFINITION = "exp(-(distance_m/sigma_narrow_m)^2)"
SELECTED_RUBBER_CONTACT_AUTHORITY = "selected_rubber_contact_authority_v1"
GEOMETRY_IDENTITY_ABS_TOL_M = 1.0e-12
SCORE_REASONS = (
    "no_contact",
    "nonfinite",
    "crossing_contract_fault",
    "no_crossing",
    "net_not_crossed",
    "net_not_clear",
    "not_opponent_bound",
    "scored_off_table",
    "scored_on_table",
)


class LandingPlacementIdentityError(ValueError):
    """Profile/frame identity drift, rather than a scoreable task failure."""


def canonical_sha256(value: object) -> str:
    """Hash repository-style canonical JSON; non-finite values fail closed."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _optional_number(value: object, *, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number or null")
    result = float(value)
    return 0.0 if result == 0.0 else result


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an exact bool")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _sealed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["canonical_sha256"] = canonical_sha256(payload)
    return result


def _verified_payload(
    value: object,
    *,
    expected_keys: frozenset[str],
    kind: str,
    label: str,
) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected = expected_keys | {"canonical_sha256"}
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    declared = _sha256(value["canonical_sha256"], label="canonical_sha256")
    payload = {key: value[key] for key in expected_keys}
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version differs")
    if payload["kind"] != kind:
        raise ValueError(f"{label} kind differs")
    if canonical_sha256(payload) != declared:
        raise ValueError(f"{label} canonical SHA differs")
    return payload, declared


@dataclass(frozen=True)
class LandingPlacementProfile:
    """Explicit scientific profile; deliberately has no adopted defaults."""

    frame_id: str
    frame_binding_sha256: str
    contact_source_semantics: str
    table_surface_z_m: float
    ball_radius_m: float
    ball_center_landing_plane_z_m: float
    net_x_m: float
    net_mesh_top_z_m: float
    ball_center_net_clear_z_m: float
    opponent_table_x_min_m: float
    opponent_table_x_max_m: float
    table_y_min_m: float
    table_y_max_m: float
    alpha_broad: float
    sigma_broad_m: float
    sigma_narrow_m: float
    on_table_gate: float
    off_table_gate: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", _nonempty_text(self.frame_id, label="frame_id")
        )
        object.__setattr__(
            self,
            "frame_binding_sha256",
            _sha256(self.frame_binding_sha256, label="frame_binding_sha256"),
        )
        object.__setattr__(
            self,
            "contact_source_semantics",
            _nonempty_text(
                self.contact_source_semantics,
                label="contact_source_semantics",
            ),
        )
        for field in fields(self):
            if field.name in (
                "frame_id",
                "frame_binding_sha256",
                "contact_source_semantics",
            ):
                continue
            object.__setattr__(
                self,
                field.name,
                _finite_number(getattr(self, field.name), label=field.name),
            )

        if self.opponent_table_x_min_m != self.net_x_m:
            raise ValueError(
                "opponent_table_x_min_m must equal net_x_m in this profile"
            )
        if self.opponent_table_x_max_m <= self.opponent_table_x_min_m:
            raise ValueError("opponent table x interval must be non-empty")
        if self.table_y_max_m <= self.table_y_min_m:
            raise ValueError("table y interval must be non-empty")
        if self.contact_source_semantics != SELECTED_RUBBER_CONTACT_AUTHORITY:
            raise ValueError(
                "contact_source_semantics must bind selected-rubber authority"
            )
        if self.ball_radius_m <= 0.0:
            raise ValueError("ball_radius_m must be positive")
        if not math.isclose(
            self.ball_center_landing_plane_z_m,
            self.table_surface_z_m + self.ball_radius_m,
            rel_tol=0.0,
            abs_tol=GEOMETRY_IDENTITY_ABS_TOL_M,
        ):
            raise ValueError(
                "ball-center landing plane must equal table surface plus radius"
            )
        if not math.isclose(
            self.ball_center_net_clear_z_m,
            self.net_mesh_top_z_m + self.ball_radius_m,
            rel_tol=0.0,
            abs_tol=GEOMETRY_IDENTITY_ABS_TOL_M,
        ):
            raise ValueError(
                "ball-center net clearance must equal net mesh top plus radius"
            )
        if self.net_mesh_top_z_m <= self.table_surface_z_m:
            raise ValueError("net mesh top must be above table surface")
        if not 0.0 < self.alpha_broad < 1.0:
            raise ValueError("alpha_broad must be in (0,1)")
        if not self.sigma_broad_m > self.sigma_narrow_m > 0.0:
            raise ValueError(
                "kernel scales must satisfy sigma_broad_m > sigma_narrow_m > 0"
            )
        if self.on_table_gate != 1.0 or self.off_table_gate != 0.5:
            raise ValueError("landing gates are frozen at on-table=1.0/off-table=0.5")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PROFILE_KIND,
            "frame_id": self.frame_id,
            "frame_binding_sha256": self.frame_binding_sha256,
            "contact_source_semantics": self.contact_source_semantics,
            "table_surface_z_m": self.table_surface_z_m,
            "ball_radius_m": self.ball_radius_m,
            "ball_center_landing_plane_z_m": (
                self.ball_center_landing_plane_z_m
            ),
            "net_x_m": self.net_x_m,
            "net_mesh_top_z_m": self.net_mesh_top_z_m,
            "ball_center_net_clear_z_m": self.ball_center_net_clear_z_m,
            "opponent_table_x_min_m": self.opponent_table_x_min_m,
            "opponent_table_x_max_m": self.opponent_table_x_max_m,
            "table_y_min_m": self.table_y_min_m,
            "table_y_max_m": self.table_y_max_m,
            "alpha_broad": self.alpha_broad,
            "sigma_broad_m": self.sigma_broad_m,
            "sigma_narrow_m": self.sigma_narrow_m,
            "on_table_gate": self.on_table_gate,
            "off_table_gate": self.off_table_gate,
            "cauchy_definition": CAUCHY_DEFINITION,
            "gaussian_definition": GAUSSIAN_DEFINITION,
            "geometry_identity_abs_tol_m": GEOMETRY_IDENTITY_ABS_TOL_M,
        }

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "LandingPlacementProfile":
        payload, declared = _verified_payload(
            value,
            expected_keys=_PROFILE_KEYS,
            kind=PROFILE_KIND,
            label="landing-placement profile",
        )
        if payload["cauchy_definition"] != CAUCHY_DEFINITION:
            raise ValueError("landing-placement Cauchy definition differs")
        if payload["gaussian_definition"] != GAUSSIAN_DEFINITION:
            raise ValueError("landing-placement Gaussian definition differs")
        if (
            payload["geometry_identity_abs_tol_m"]
            != GEOMETRY_IDENTITY_ABS_TOL_M
        ):
            raise ValueError("landing-placement geometry tolerance differs")
        result = cls(
            **{
                key: payload[key]
                for key in _PROFILE_FIELDS
            }
        )
        if result.canonical_sha256 != declared:
            raise ValueError(
                "landing-placement profile normalization changed canonical SHA"
            )
        return result


_PROFILE_FIELDS = tuple(field.name for field in fields(LandingPlacementProfile))
_PROFILE_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "cauchy_definition",
        "gaussian_definition",
        "geometry_identity_abs_tol_m",
    )
    + _PROFILE_FIELDS
)


@dataclass(frozen=True)
class LandingPlacementTaskIdentity:
    """Canonical per-attempt target and opaque semantic/instance identity."""

    frame_id: str
    frame_binding_sha256: str
    profile_sha256: str
    task_receipt_sha256: str
    semantic_binding_sha256: str
    instance_binding_sha256: str
    target_x_m: float
    target_y_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", _nonempty_text(self.frame_id, label="frame_id")
        )
        for name in (
            "frame_binding_sha256",
            "profile_sha256",
            "task_receipt_sha256",
            "semantic_binding_sha256",
            "instance_binding_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in ("target_x_m", "target_y_m"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), label=name),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": TASK_IDENTITY_KIND,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @property
    def canonical_token_bytes(self) -> bytes:
        """Raw 32-byte token used by the device-side identity comparison."""

        return bytes.fromhex(self.canonical_sha256)

    @classmethod
    def from_mapping(cls, value: object) -> "LandingPlacementTaskIdentity":
        payload, declared = _verified_payload(
            value,
            expected_keys=_TASK_IDENTITY_KEYS,
            kind=TASK_IDENTITY_KIND,
            label="landing-placement task identity",
        )
        result = cls(
            **{key: payload[key] for key in _TASK_IDENTITY_FIELDS}
        )
        if result.canonical_sha256 != declared:
            raise ValueError(
                "landing-placement task identity normalization changed "
                "canonical SHA"
            )
        return result


_TASK_IDENTITY_FIELDS = tuple(
    field.name for field in fields(LandingPlacementTaskIdentity)
)
_TASK_IDENTITY_KEYS = frozenset(
    ("schema_version", "kind") + _TASK_IDENTITY_FIELDS
)


@dataclass(frozen=True)
class LandingPlacementFacts:
    """Canonical ball-centre landing-plane facts for one immutable task."""

    frame_id: str
    profile_sha256: str
    task_identity_sha256: str
    contact_valid: bool
    first_plane_crossing_valid: bool
    first_plane_crossing_nonfinite: bool
    first_plane_crossing_contract_fault: bool
    first_plane_crossing_x_m: Optional[float]
    first_plane_crossing_y_m: Optional[float]
    ball_center_net_crossed: bool
    ball_center_net_clear: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", _nonempty_text(self.frame_id, label="frame_id")
        )
        for name in ("profile_sha256", "task_identity_sha256"):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in (
            "contact_valid",
            "first_plane_crossing_valid",
            "first_plane_crossing_nonfinite",
            "first_plane_crossing_contract_fault",
            "ball_center_net_crossed",
            "ball_center_net_clear",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )

        x = _optional_number(
            self.first_plane_crossing_x_m,
            label="first_plane_crossing_x_m",
        )
        y = _optional_number(
            self.first_plane_crossing_y_m,
            label="first_plane_crossing_y_m",
        )
        raw_nonfinite = any(
            value is not None and not math.isfinite(value) for value in (x, y)
        )
        nonfinite = self.first_plane_crossing_nonfinite or raw_nonfinite
        missing_coordinate = x is None or y is None
        contract_fault = self.first_plane_crossing_contract_fault or (
            self.first_plane_crossing_valid
            and missing_coordinate
            and not nonfinite
        )
        finite_pair = (
            x is not None
            and y is not None
            and math.isfinite(x)
            and math.isfinite(y)
        )
        if nonfinite or contract_fault or not finite_pair:
            x = None
            y = None
            object.__setattr__(self, "first_plane_crossing_valid", False)
        object.__setattr__(self, "first_plane_crossing_nonfinite", nonfinite)
        object.__setattr__(
            self,
            "first_plane_crossing_contract_fault",
            contract_fault,
        )
        object.__setattr__(self, "first_plane_crossing_x_m", x)
        object.__setattr__(self, "first_plane_crossing_y_m", y)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": FACTS_KIND,
            "frame_id": self.frame_id,
            "profile_sha256": self.profile_sha256,
            "task_identity_sha256": self.task_identity_sha256,
            "contact_valid": self.contact_valid,
            "first_plane_crossing_valid": self.first_plane_crossing_valid,
            "first_plane_crossing_nonfinite": (
                self.first_plane_crossing_nonfinite
            ),
            "first_plane_crossing_contract_fault": (
                self.first_plane_crossing_contract_fault
            ),
            "first_plane_crossing_x_m": self.first_plane_crossing_x_m,
            "first_plane_crossing_y_m": self.first_plane_crossing_y_m,
            "ball_center_net_crossed": self.ball_center_net_crossed,
            "ball_center_net_clear": self.ball_center_net_clear,
        }

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "LandingPlacementFacts":
        payload, declared = _verified_payload(
            value,
            expected_keys=_FACTS_KEYS,
            kind=FACTS_KIND,
            label="landing-placement facts",
        )
        result = cls(**{key: payload[key] for key in _FACTS_FIELDS})
        if result.canonical_sha256 != declared:
            raise ValueError(
                "landing-placement facts normalization changed canonical SHA"
            )
        return result


_FACTS_FIELDS = tuple(field.name for field in fields(LandingPlacementFacts))
_FACTS_KEYS = frozenset(("schema_version", "kind") + _FACTS_FIELDS)


@dataclass(frozen=True)
class LandingPlacementScore:
    """Canonical scored result with validity and on-table state kept separate."""

    frame_id: str
    profile_sha256: str
    facts_sha256: str
    task_identity_sha256: str
    task_receipt_sha256: str
    semantic_binding_sha256: str
    instance_binding_sha256: str
    target_x_m: float
    target_y_m: float
    contact_valid: bool
    first_plane_crossing_valid: bool
    first_plane_crossing_nonfinite: bool
    first_plane_crossing_contract_fault: bool
    ball_center_net_crossed: bool
    ball_center_net_clear: bool
    opponent_bound: bool
    on_opponent_table: bool
    reason: str
    placement_error_m: Optional[float]
    broad_kernel: float
    narrow_kernel: float
    blended_kernel: float
    table_gate: float
    total: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", _nonempty_text(self.frame_id, label="frame_id")
        )
        for name in (
            "profile_sha256",
            "facts_sha256",
            "task_identity_sha256",
            "task_receipt_sha256",
            "semantic_binding_sha256",
            "instance_binding_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), label=name)
            )
        for name in ("target_x_m", "target_y_m"):
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), label=name),
            )
        for name in (
            "contact_valid",
            "first_plane_crossing_valid",
            "first_plane_crossing_nonfinite",
            "first_plane_crossing_contract_fault",
            "ball_center_net_crossed",
            "ball_center_net_clear",
            "opponent_bound",
            "on_opponent_table",
        ):
            object.__setattr__(
                self, name, _exact_bool(getattr(self, name), label=name)
            )
        object.__setattr__(
            self, "reason", _nonempty_text(self.reason, label="reason")
        )
        if self.reason not in SCORE_REASONS:
            raise ValueError("unknown landing-placement score reason")
        if self.placement_error_m is not None:
            error = _finite_number(
                self.placement_error_m, label="placement_error_m"
            )
            if error < 0.0:
                raise ValueError("placement_error_m must be nonnegative")
            object.__setattr__(self, "placement_error_m", error)
        for name in (
            "broad_kernel",
            "narrow_kernel",
            "blended_kernel",
            "table_gate",
            "total",
        ):
            value = _finite_number(getattr(self, name), label=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
            object.__setattr__(self, name, value)
        if self.on_opponent_table and not (
            self.first_plane_crossing_valid and self.opponent_bound
        ):
            raise ValueError("on_opponent_table requires a valid opponent crossing")
        if (
            self.first_plane_crossing_nonfinite
            and self.first_plane_crossing_valid
        ):
            raise ValueError("non-finite crossing cannot be valid")
        if (
            self.first_plane_crossing_contract_fault
            and self.first_plane_crossing_valid
        ):
            raise ValueError("crossing contract fault cannot be valid")
        if not self.first_plane_crossing_valid and self.placement_error_m is not None:
            raise ValueError("invalid first crossing cannot carry placement_error_m")
        if not self.contact_valid:
            expected_reason = "no_contact"
        elif self.first_plane_crossing_nonfinite:
            expected_reason = "nonfinite"
        elif self.first_plane_crossing_contract_fault:
            expected_reason = "crossing_contract_fault"
        elif not self.first_plane_crossing_valid:
            expected_reason = "no_crossing"
        elif not self.ball_center_net_crossed:
            expected_reason = "net_not_crossed"
        elif not self.ball_center_net_clear:
            expected_reason = "net_not_clear"
        elif not self.opponent_bound:
            expected_reason = "not_opponent_bound"
        elif self.on_opponent_table:
            expected_reason = "scored_on_table"
        else:
            expected_reason = "scored_off_table"
        if self.reason != expected_reason:
            raise ValueError("landing-placement primary reason is inconsistent")
        if self.reason == "scored_on_table":
            if self.table_gate != 1.0:
                raise ValueError("scored_on_table requires table_gate=1.0")
        elif self.reason == "scored_off_table":
            if self.on_opponent_table or self.table_gate != 0.5:
                raise ValueError("scored_off_table requires off-table gate=0.5")
        elif self.table_gate != 0.0 or self.total != 0.0:
            raise ValueError("unscored reason requires zero gate and total")
        if not math.isclose(
            self.total,
            self.table_gate * self.blended_kernel,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("total must equal table_gate * blended_kernel")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": SCORE_KIND,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    def to_mapping(self) -> dict[str, object]:
        return _sealed(self.payload())

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_mapping(cls, value: object) -> "LandingPlacementScore":
        payload, declared = _verified_payload(
            value,
            expected_keys=_SCORE_KEYS,
            kind=SCORE_KIND,
            label="landing-placement score",
        )
        result = cls(**{key: payload[key] for key in _SCORE_FIELDS})
        if result.canonical_sha256 != declared:
            raise ValueError(
                "landing-placement score normalization changed canonical SHA"
            )
        return result


_SCORE_FIELDS = tuple(field.name for field in fields(LandingPlacementScore))
_SCORE_KEYS = frozenset(("schema_version", "kind") + _SCORE_FIELDS)


def score_landing_placement(
    profile: LandingPlacementProfile,
    task_identity: LandingPlacementTaskIdentity,
    facts: LandingPlacementFacts,
) -> LandingPlacementScore:
    """Score ``gate * (alpha*Cauchy + (1-alpha)*Gaussian)``.

    The scorer first verifies the profile/task/facts identity chain.  Task
    identity supplies the only authoritative target.  Eligibility then
    requires selected-rubber contact, a declared and finite first
    ball-centre landing-plane crossing, both net facts, and an opponent-side
    crossing.  The
    opponent-table rectangle yields gate ``1.0``; any other opponent-side
    crossing yields ``0.5``.  Own-side/backwards and all invalid cases yield
    zero.  Table membership is derived here and is not conflated with crossing
    validity.
    """

    if not isinstance(profile, LandingPlacementProfile):
        raise TypeError("profile must be a LandingPlacementProfile")
    if not isinstance(task_identity, LandingPlacementTaskIdentity):
        raise TypeError(
            "task_identity must be a LandingPlacementTaskIdentity"
        )
    if not isinstance(facts, LandingPlacementFacts):
        raise TypeError("facts must be LandingPlacementFacts")
    if task_identity.profile_sha256 != profile.canonical_sha256:
        raise LandingPlacementIdentityError("task identity profile SHA differs")
    if task_identity.frame_id != profile.frame_id:
        raise LandingPlacementIdentityError("task identity frame_id differs")
    if task_identity.frame_binding_sha256 != profile.frame_binding_sha256:
        raise LandingPlacementIdentityError(
            "task identity frame binding SHA differs"
        )
    if facts.profile_sha256 != profile.canonical_sha256:
        raise LandingPlacementIdentityError("facts profile SHA differs")
    if facts.frame_id != profile.frame_id:
        raise LandingPlacementIdentityError("facts frame_id differs")
    if facts.task_identity_sha256 != task_identity.canonical_sha256:
        raise LandingPlacementIdentityError("facts task identity SHA differs")
    if not (
        profile.opponent_table_x_min_m
        <= task_identity.target_x_m
        <= profile.opponent_table_x_max_m
        and profile.table_y_min_m
        <= task_identity.target_y_m
        <= profile.table_y_max_m
    ):
        raise LandingPlacementIdentityError(
            "facts target lies outside the opponent table"
        )

    first_valid = (
        facts.first_plane_crossing_valid
        and not facts.first_plane_crossing_nonfinite
        and not facts.first_plane_crossing_contract_fault
        and facts.first_plane_crossing_x_m is not None
        and facts.first_plane_crossing_y_m is not None
    )
    opponent_bound = bool(
        first_valid
        and facts.first_plane_crossing_x_m > profile.net_x_m
    )
    on_table = bool(
        opponent_bound
        and profile.opponent_table_x_min_m
        <= facts.first_plane_crossing_x_m
        <= profile.opponent_table_x_max_m
        and profile.table_y_min_m
        <= facts.first_plane_crossing_y_m
        <= profile.table_y_max_m
    )
    net_ok = facts.ball_center_net_crossed and facts.ball_center_net_clear

    if first_valid:
        error = math.hypot(
            facts.first_plane_crossing_x_m - task_identity.target_x_m,
            facts.first_plane_crossing_y_m - task_identity.target_y_m,
        )
        broad = 1.0 / (1.0 + (error / profile.sigma_broad_m) ** 2)
        narrow = math.exp(-((error / profile.sigma_narrow_m) ** 2))
        blended = (
            profile.alpha_broad * broad
            + (1.0 - profile.alpha_broad) * narrow
        )
    else:
        error = None
        broad = 0.0
        narrow = 0.0
        blended = 0.0

    eligible = facts.contact_valid and first_valid and net_ok and opponent_bound
    if not facts.contact_valid:
        reason = "no_contact"
    elif facts.first_plane_crossing_nonfinite:
        reason = "nonfinite"
    elif facts.first_plane_crossing_contract_fault:
        reason = "crossing_contract_fault"
    elif not first_valid:
        reason = "no_crossing"
    elif not facts.ball_center_net_crossed:
        reason = "net_not_crossed"
    elif not facts.ball_center_net_clear:
        reason = "net_not_clear"
    elif not opponent_bound:
        reason = "not_opponent_bound"
    elif on_table:
        reason = "scored_on_table"
    else:
        reason = "scored_off_table"

    if eligible:
        gate = profile.on_table_gate if on_table else profile.off_table_gate
    else:
        gate = 0.0
    total = gate * blended
    return LandingPlacementScore(
        frame_id=profile.frame_id,
        profile_sha256=profile.canonical_sha256,
        facts_sha256=facts.canonical_sha256,
        task_identity_sha256=task_identity.canonical_sha256,
        task_receipt_sha256=task_identity.task_receipt_sha256,
        semantic_binding_sha256=task_identity.semantic_binding_sha256,
        instance_binding_sha256=task_identity.instance_binding_sha256,
        target_x_m=task_identity.target_x_m,
        target_y_m=task_identity.target_y_m,
        contact_valid=facts.contact_valid,
        first_plane_crossing_valid=bool(first_valid),
        first_plane_crossing_nonfinite=(
            facts.first_plane_crossing_nonfinite
        ),
        first_plane_crossing_contract_fault=(
            facts.first_plane_crossing_contract_fault
        ),
        ball_center_net_crossed=facts.ball_center_net_crossed,
        ball_center_net_clear=facts.ball_center_net_clear,
        opponent_bound=opponent_bound,
        on_opponent_table=on_table,
        reason=reason,
        placement_error_m=error,
        broad_kernel=broad,
        narrow_kernel=narrow,
        blended_kernel=blended,
        table_gate=gate,
        total=total,
    )


__all__ = (
    "CAUCHY_DEFINITION",
    "FACTS_KIND",
    "GAUSSIAN_DEFINITION",
    "GEOMETRY_IDENTITY_ABS_TOL_M",
    "LandingPlacementFacts",
    "LandingPlacementIdentityError",
    "LandingPlacementProfile",
    "LandingPlacementScore",
    "LandingPlacementTaskIdentity",
    "PROFILE_KIND",
    "SCHEMA_VERSION",
    "SCORE_REASONS",
    "SCORE_KIND",
    "SELECTED_RUBBER_CONTACT_AUTHORITY",
    "TASK_IDENTITY_KIND",
    "canonical_sha256",
    "score_landing_placement",
)
