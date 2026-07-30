"""Pure-CPU counter-rally objective and fitted venue-ball reference.

This module is deliberately independent of Isaac, MuJoCo and Torch.  It gives
the single-action backhand experiments one frozen, reviewable meaning:

* express the incoming ball and contact in the sampled base-yaw frame;
* send the ball horizontally opposite to the incoming direction;
* choose the first landing on that ray;
* apply the same fitted table impulse used by the venue physics model; and
* require a speed-matched crossing of the opponent baseline.

It is a reference/gate implementation, not a policy-time selector.  Production
wiring must bind its profile and venue-physics hashes into the launch receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence, Tuple


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

COUNTER_RALLY_MODE = "counter_rally_v1"
COUNTER_RALLY_INACTIVE_ARMS = (
    "landing_aim_y_lower",
    "landing_aim_y_upper",
)
COUNTER_RALLY_SOLVER_REJECTION_REASON_SCHEMA = (
    "reverse_ray_not_opponent_bound",
    "landing_depth_outside_table",
    "landing_depth_not_opponent_half",
    "landing_behind_contact",
    "reverse_ray_misses_table",
    "incoming_speed_outside_venue_support",
    "target_speed_outside_venue_support",
)
COUNTER_RALLY_SOLVER_REJECTION_REASONS = frozenset(
    COUNTER_RALLY_SOLVER_REJECTION_REASON_SCHEMA
)
_PROFILE_KEYS = frozenset(
    (
        "mode",
        "opponent_baseline_x_env_m",
        "table_near_x_env_m",
        "table_length_m",
        "table_half_width_m",
        "table_surface_z_env_m",
        "net_height_m",
        "table_edge_margin_m",
        "target_baseline_speed_ratio",
        "target_speed_abs_tolerance_mps",
        "target_speed_rel_tolerance",
        "reverse_direction_tolerance_deg",
        "baseline_direction_tolerance_deg",
        "landing_tolerance_m",
        "minimum_opponent_x_component",
        "minimum_supported_ball_speed_mps",
        "maximum_supported_ball_speed_mps",
        "reward_legal_fraction",
        "reward_landing_fraction",
        "reward_reverse_fraction",
        "reward_speed_fraction",
        "inactive_curriculum_arms",
    )
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _unit2(value: Sequence[float], *, name: str) -> Vec2:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise CounterRallyRejected(f"{name}_shape")
    x = _finite(value[0], name=f"{name}[0]")
    y = _finite(value[1], name=f"{name}[1]")
    norm = math.hypot(x, y)
    if norm <= 1.0e-9:
        raise CounterRallyRejected(f"{name}_horizontal_zero")
    return (x / norm, y / norm)


def _rotate2(value: Vec2, yaw_rad: float) -> Vec2:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (c * value[0] - s * value[1], s * value[0] + c * value[1])


@dataclass(frozen=True)
class CounterRallyObjectiveProfile:
    mode: str = COUNTER_RALLY_MODE
    opponent_baseline_x_env_m: float = 3.24
    table_near_x_env_m: float = 0.50
    table_length_m: float = 2.74
    table_half_width_m: float = 0.7625
    table_surface_z_env_m: float = 0.76
    net_height_m: float = 0.1525
    table_edge_margin_m: float = 0.025
    target_baseline_speed_ratio: float = 1.0
    target_speed_abs_tolerance_mps: float = 0.35
    target_speed_rel_tolerance: float = 0.15
    reverse_direction_tolerance_deg: float = 8.0
    baseline_direction_tolerance_deg: float = 12.0
    landing_tolerance_m: float = 0.03
    minimum_opponent_x_component: float = 0.85
    minimum_supported_ball_speed_mps: float = 1.0
    maximum_supported_ball_speed_mps: float = 7.0
    reward_legal_fraction: float = 0.60
    reward_landing_fraction: float = 0.05
    reward_reverse_fraction: float = 0.10
    reward_speed_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.mode != COUNTER_RALLY_MODE:
            raise ValueError(f"mode must be {COUNTER_RALLY_MODE!r}")
        for field_name in _PROFILE_KEYS - {
            "mode",
            "inactive_curriculum_arms",
        }:
            value = _finite(getattr(self, field_name), name=field_name)
            object.__setattr__(self, field_name, value)
        if self.table_length_m <= 0.0 or self.table_half_width_m <= 0.0:
            raise ValueError("table dimensions must be positive")
        expected_far = self.table_near_x_env_m + self.table_length_m
        if abs(self.opponent_baseline_x_env_m - expected_far) > 1.0e-9:
            raise ValueError(
                "opponent baseline must equal table_near_x + table_length"
            )
        if not 0.0 <= self.table_edge_margin_m < self.table_half_width_m:
            raise ValueError("table_edge_margin_m is outside the table")
        if not 0.0 < self.minimum_opponent_x_component <= 1.0:
            raise ValueError("minimum_opponent_x_component must be in (0, 1]")
        if not (
            0.0 < self.minimum_supported_ball_speed_mps
            < self.maximum_supported_ball_speed_mps
        ):
            raise ValueError("invalid supported ball-speed interval")
        if self.target_baseline_speed_ratio <= 0.0:
            raise ValueError("target_baseline_speed_ratio must be positive")
        for name in (
            "target_speed_abs_tolerance_mps",
            "target_speed_rel_tolerance",
            "reverse_direction_tolerance_deg",
            "baseline_direction_tolerance_deg",
            "landing_tolerance_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        reward_sum = sum(
            getattr(self, name)
            for name in (
                "reward_legal_fraction",
                "reward_landing_fraction",
                "reward_reverse_fraction",
                "reward_speed_fraction",
            )
        )
        if abs(reward_sum - 1.0) > 1.0e-12:
            raise ValueError("counter-rally reward fractions must sum to one")
        if any(
            getattr(self, name) < 0.0
            for name in (
                "reward_legal_fraction",
                "reward_landing_fraction",
                "reward_reverse_fraction",
                "reward_speed_fraction",
            )
        ):
            raise ValueError("counter-rally reward fractions must be nonnegative")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "CounterRallyObjectiveProfile":
        if not isinstance(value, Mapping):
            raise ValueError("counter_rally_objective must be a mapping")
        if frozenset(value) != _PROFILE_KEYS:
            raise ValueError(
                "counter_rally_objective keys mismatch: "
                f"expected {sorted(_PROFILE_KEYS)!r}, got {sorted(value)!r}"
            )
        inactive = value["inactive_curriculum_arms"]
        if (
            not isinstance(inactive, list)
            or tuple(inactive) != COUNTER_RALLY_INACTIVE_ARMS
        ):
            raise ValueError(
                "counter-rally inactive_curriculum_arms must disable "
                "both landing-y sides in canonical order"
            )
        return cls(
            **{
                name: value[name]
                for name in _PROFILE_KEYS
                if name != "inactive_curriculum_arms"
            }
        )

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "mode": self.mode,
            "opponent_baseline_x_env_m": self.opponent_baseline_x_env_m,
            "table_near_x_env_m": self.table_near_x_env_m,
            "table_length_m": self.table_length_m,
            "table_half_width_m": self.table_half_width_m,
            "table_surface_z_env_m": self.table_surface_z_env_m,
            "net_height_m": self.net_height_m,
            "table_edge_margin_m": self.table_edge_margin_m,
            "target_baseline_speed_ratio": self.target_baseline_speed_ratio,
            "target_speed_abs_tolerance_mps": self.target_speed_abs_tolerance_mps,
            "target_speed_rel_tolerance": self.target_speed_rel_tolerance,
            "reverse_direction_tolerance_deg": self.reverse_direction_tolerance_deg,
            "baseline_direction_tolerance_deg": self.baseline_direction_tolerance_deg,
            "landing_tolerance_m": self.landing_tolerance_m,
            "minimum_opponent_x_component": self.minimum_opponent_x_component,
            "minimum_supported_ball_speed_mps": self.minimum_supported_ball_speed_mps,
            "maximum_supported_ball_speed_mps": self.maximum_supported_ball_speed_mps,
            "reward_legal_fraction": self.reward_legal_fraction,
            "reward_landing_fraction": self.reward_landing_fraction,
            "reward_reverse_fraction": self.reward_reverse_fraction,
            "reward_speed_fraction": self.reward_speed_fraction,
            "inactive_curriculum_arms": list(
                self.inactive_curriculum_arms
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())

    @property
    def inactive_curriculum_arms(self) -> Tuple[str, ...]:
        return COUNTER_RALLY_INACTIVE_ARMS


class CounterRallyRejected(ValueError):
    """Named fail-closed rejection of an invalid counter-rally question."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


class CounterRallyIdentityError(RuntimeError):
    """Frozen action/objective identity drift; never a difficulty failure."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def counter_rally_reverse_ray_geometry(
    *,
    contact_env_m: Sequence[float],
    return_direction_env_xy: Sequence[float],
    landing_depth_env_x_m: float,
    profile: object,
) -> Tuple[Vec2, str | None]:
    """Return one reverse-ray aim plus its canonical admission reason.

    This helper is deliberately non-throwing for ordinary proposal geometry:
    the sampler must preserve every fixed-draw proposal in its denominator.
    ``derive_counter_rally_task`` turns a non-``None`` reason into a strict
    rejection, and the production fixed-action solver records the same reason.
    Invalid scalar/profile shapes remain programmer/configuration errors.
    """

    if getattr(profile, "mode", None) != COUNTER_RALLY_MODE:
        raise ValueError("counter-rally objective mode is invalid")
    if isinstance(contact_env_m, (str, bytes)) or len(contact_env_m) != 3:
        raise ValueError("contact_env_m must contain three numbers")
    contact_x = _finite(contact_env_m[0], name="contact_env_m[0]")
    contact_y = _finite(contact_env_m[1], name="contact_env_m[1]")
    landing_x = _finite(
        landing_depth_env_x_m, name="landing_depth_env_x_m"
    )
    try:
        return_direction = _unit2(
            return_direction_env_xy,
            name="return_direction_env_xy",
        )
    except CounterRallyRejected:
        return (landing_x, contact_y), "reverse_ray_not_opponent_bound"

    reason = None
    if return_direction[0] < getattr(
        profile, "minimum_opponent_x_component"
    ):
        reason = "reverse_ray_not_opponent_bound"

    table_lo = (
        getattr(profile, "table_near_x_env_m")
        + getattr(profile, "table_edge_margin_m")
    )
    table_hi = (
        getattr(profile, "opponent_baseline_x_env_m")
        - getattr(profile, "table_edge_margin_m")
    )
    if not table_lo <= landing_x <= table_hi:
        reason = reason or "landing_depth_outside_table"
    net_x = (
        getattr(profile, "table_near_x_env_m")
        + 0.5 * getattr(profile, "table_length_m")
    )
    if landing_x <= net_x:
        reason = reason or "landing_depth_not_opponent_half"
    if return_direction[0] <= 1.0e-12:
        return (landing_x, contact_y), reason

    ray_scale = (landing_x - contact_x) / return_direction[0]
    if ray_scale <= 0.0:
        reason = reason or "landing_behind_contact"
    landing_y = contact_y + ray_scale * return_direction[1]
    half_width = (
        getattr(profile, "table_half_width_m")
        - getattr(profile, "table_edge_margin_m")
    )
    if abs(landing_y) > half_width:
        reason = reason or "reverse_ray_misses_table"
    return (landing_x, landing_y), reason


@dataclass(frozen=True)
class CounterRallyTask:
    contact_env_m: Vec3
    incoming_direction_b_yaw_xy: Vec2
    return_direction_b_yaw_xy: Vec2
    return_direction_env_xy: Vec2
    landing_aim_env_xy_m: Vec2
    opponent_baseline_x_env_m: float
    target_baseline_speed_mps: float
    objective_profile_sha256: str


@dataclass(frozen=True)
class CounterRallyFixedSolverPrecheck:
    """One fixed-action proposal outcome before the ordinary task solver.

    Identity/malformed-input drift raises :class:`CounterRallyIdentityError`
    and therefore never enters the proposal/difficulty ledger.  A physically
    invalid question remains an honest proposal with ``P=1, A=0`` and one
    canonical rejection reason.  A passing precheck is merely eligible for
    the ordinary racket solver; it must not increment ``A`` early.
    """

    frozen_action_uid: int
    objective_profile_sha256: str
    proposal_count: int
    eligible_for_solver: bool
    task: CounterRallyTask | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if (
            type(self.frozen_action_uid) is not int
            or self.frozen_action_uid <= 0
        ):
            raise ValueError("frozen_action_uid must be a positive integer")
        if (
            type(self.objective_profile_sha256) is not str
            or len(self.objective_profile_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.objective_profile_sha256
            )
        ):
            raise ValueError(
                "objective_profile_sha256 must be 64 lowercase hex characters"
            )
        if self.proposal_count != 1:
            raise ValueError(
                "counter-rally fixed-solver precheck requires P=1"
            )
        if type(self.eligible_for_solver) is not bool:
            raise ValueError(
                "eligible_for_solver must be bool"
            )
        if self.eligible_for_solver != (self.task is not None):
            raise ValueError(
                "counter-rally eligibility disagrees with task presence"
            )
        if self.eligible_for_solver == (
            self.rejection_reason is not None
        ):
            raise ValueError(
                "counter-rally rejection reason must exist exactly when "
                "the proposal is ineligible"
            )
        if (
            self.task is not None
            and self.task.objective_profile_sha256
            != self.objective_profile_sha256
        ):
            raise CounterRallyIdentityError(
                "objective_profile_sha256_mismatch"
            )

    @property
    def rejected_ledger_counts(self) -> Tuple[int, int]:
        """Return the only ledger delta owned by this precheck: ``P=1/A=0``."""

        if self.eligible_for_solver:
            raise ValueError(
                "a passing precheck has no solver-admission ledger delta"
            )
        return (1, 0)


def derive_counter_rally_task(
    *,
    base_goal_env_xy_m: Sequence[float],
    base_yaw_env_rad: float,
    contact_offset_b_yaw_m: Sequence[float],
    incoming_direction_b_yaw: Sequence[float],
    incoming_ball_speed_at_contact_mps: float,
    landing_depth_env_x_m: float,
    profile: CounterRallyObjectiveProfile,
) -> CounterRallyTask:
    """Derive one base-relative reverse-ray task without any selector."""

    if not isinstance(profile, CounterRallyObjectiveProfile):
        raise TypeError("profile must be CounterRallyObjectiveProfile")
    if len(base_goal_env_xy_m) != 2 or len(contact_offset_b_yaw_m) != 3:
        raise CounterRallyRejected("base_or_contact_shape")
    base_x = _finite(base_goal_env_xy_m[0], name="base_goal_env_xy_m[0]")
    base_y = _finite(base_goal_env_xy_m[1], name="base_goal_env_xy_m[1]")
    yaw = _finite(base_yaw_env_rad, name="base_yaw_env_rad")
    offset_xy = (
        _finite(contact_offset_b_yaw_m[0], name="contact_offset_b_yaw_m[0]"),
        _finite(contact_offset_b_yaw_m[1], name="contact_offset_b_yaw_m[1]"),
    )
    offset_w = _rotate2(offset_xy, yaw)
    contact = (
        base_x + offset_w[0],
        base_y + offset_w[1],
        _finite(contact_offset_b_yaw_m[2], name="contact_offset_b_yaw_m[2]"),
    )
    incoming_b = _unit2(
        incoming_direction_b_yaw, name="incoming_direction_b_yaw"
    )
    return_b = (-incoming_b[0], -incoming_b[1])
    return_w = _rotate2(return_b, yaw)
    landing_aim, rejection_reason = counter_rally_reverse_ray_geometry(
        contact_env_m=contact,
        return_direction_env_xy=return_w,
        landing_depth_env_x_m=landing_depth_env_x_m,
        profile=profile,
    )
    if rejection_reason is not None:
        raise CounterRallyRejected(rejection_reason)
    incoming_speed = _finite(
        incoming_ball_speed_at_contact_mps,
        name="incoming_ball_speed_at_contact_mps",
    )
    if not (
        profile.minimum_supported_ball_speed_mps
        <= incoming_speed
        <= profile.maximum_supported_ball_speed_mps
    ):
        raise CounterRallyRejected("incoming_speed_outside_venue_support")
    target_speed = profile.target_baseline_speed_ratio * incoming_speed
    if not (
        profile.minimum_supported_ball_speed_mps
        <= target_speed
        <= profile.maximum_supported_ball_speed_mps
    ):
        raise CounterRallyRejected("target_speed_outside_venue_support")
    return CounterRallyTask(
        contact_env_m=contact,
        incoming_direction_b_yaw_xy=incoming_b,
        return_direction_b_yaw_xy=return_b,
        return_direction_env_xy=return_w,
        landing_aim_env_xy_m=landing_aim,
        opponent_baseline_x_env_m=profile.opponent_baseline_x_env_m,
        target_baseline_speed_mps=target_speed,
        objective_profile_sha256=profile.sha256,
    )


def precheck_counter_rally_fixed_solver_proposal(
    *,
    frozen_action_uid: int,
    solver_action_uid: int,
    expected_objective_profile_sha256: str,
    base_goal_env_xy_m: Sequence[float],
    base_yaw_env_rad: float,
    contact_offset_b_yaw_m: Sequence[float],
    incoming_direction_b_yaw: Sequence[float],
    incoming_ball_speed_at_contact_mps: float,
    landing_depth_env_x_m: float,
    profile: CounterRallyObjectiveProfile,
) -> CounterRallyFixedSolverPrecheck:
    """Precheck one proposal without selector fallback or action switching."""

    if (
        type(frozen_action_uid) is not int
        or frozen_action_uid <= 0
        or type(solver_action_uid) is not int
        or solver_action_uid <= 0
    ):
        raise ValueError("action UIDs must be positive integers")
    if solver_action_uid != frozen_action_uid:
        raise CounterRallyIdentityError("frozen_action_uid_mismatch")
    if (
        type(expected_objective_profile_sha256) is not str
        or len(expected_objective_profile_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_objective_profile_sha256
        )
    ):
        raise ValueError(
            "expected_objective_profile_sha256 must be 64 lowercase hex characters"
        )
    if profile.sha256 != expected_objective_profile_sha256:
        raise CounterRallyIdentityError(
            "objective_profile_sha256_mismatch"
        )
    try:
        task = derive_counter_rally_task(
            base_goal_env_xy_m=base_goal_env_xy_m,
            base_yaw_env_rad=base_yaw_env_rad,
            contact_offset_b_yaw_m=contact_offset_b_yaw_m,
            incoming_direction_b_yaw=incoming_direction_b_yaw,
            incoming_ball_speed_at_contact_mps=(
                incoming_ball_speed_at_contact_mps
            ),
            landing_depth_env_x_m=landing_depth_env_x_m,
            profile=profile,
        )
    except CounterRallyRejected as error:
        if error.reason not in COUNTER_RALLY_SOLVER_REJECTION_REASONS:
            raise CounterRallyIdentityError(
                "malformed_counter_rally_proposal:" + error.reason
            ) from error
        return CounterRallyFixedSolverPrecheck(
            frozen_action_uid=frozen_action_uid,
            objective_profile_sha256=expected_objective_profile_sha256,
            proposal_count=1,
            eligible_for_solver=False,
            task=None,
            rejection_reason=error.reason,
        )
    except (TypeError, ValueError) as error:
        raise CounterRallyIdentityError(
            "malformed_counter_rally_proposal:" + str(error)
        ) from error
    return CounterRallyFixedSolverPrecheck(
        frozen_action_uid=frozen_action_uid,
        objective_profile_sha256=expected_objective_profile_sha256,
        proposal_count=1,
        eligible_for_solver=True,
        task=task,
        rejection_reason=None,
    )


@dataclass(frozen=True)
class VenueBallPhysics:
    ball_radius_m: float = 0.020
    ball_mass_kg: float = 0.0034
    ball_inertia_coeff: float = 2.0 / 3.0
    gravity_mps2: float = 9.81
    drag_k_d_per_m: float = 0.1261
    magnus_k_m: float = 0.00444
    table_e_eff: float = 0.9215
    table_a_t: float = 0.369
    table_b_t: float = 0.0
    table_mu_safety: float = 2.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _finite(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.ball_radius_m <= 0.0 or self.ball_mass_kg <= 0.0:
            raise ValueError("ball radius and mass must be positive")
        if self.ball_inertia_coeff <= 0.0:
            raise ValueError("ball inertia coefficient must be positive")

    def to_mapping(self) -> Mapping[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())

    @classmethod
    def from_venue_yaml(cls, path: str | Path) -> "VenueBallPhysics":
        """Load only the reviewed scalar venue parameters.

        PyYAML is deliberately imported lazily so importing this pure contract
        module does not add a manifest-time dependency.
        """

        import yaml  # type: ignore

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            ball_radius_m=raw["ball"]["radius"],
            ball_mass_kg=raw["ball"]["mass"],
            ball_inertia_coeff=raw["ball"]["inertia_coeff"],
            gravity_mps2=raw["flight"]["g"],
            drag_k_d_per_m=raw["flight"]["k_d"],
            magnus_k_m=raw["flight"]["k_m"],
            table_e_eff=raw["contact"]["table"]["e_eff"],
            table_a_t=raw["contact"]["table"]["a_t"],
            table_b_t=raw["contact"]["table"]["b_t"],
            table_mu_safety=raw["contact"]["table"]["mu_safety"],
        )


def fitted_table_impulse(
    *,
    velocity_before_mps: Sequence[float],
    spin_before_radps: Sequence[float],
    physics: VenueBallPhysics,
) -> Tuple[Vec3, Vec3]:
    """Apply the fitted table normal/tangential impulse to one ball state."""

    if len(velocity_before_mps) != 3 or len(spin_before_radps) != 3:
        raise ValueError("velocity and spin must be 3-vectors")
    v = tuple(_finite(x, name="velocity_before_mps") for x in velocity_before_mps)
    omega = tuple(_finite(x, name="spin_before_radps") for x in spin_before_radps)
    if v[2] >= 0.0:
        raise CounterRallyRejected("table_contact_not_descending")
    radius = physics.ball_radius_m
    mass = physics.ball_mass_kg
    inertia = physics.ball_inertia_coeff * mass * radius * radius
    # Contact-point tangential velocity: v + omega x (-r z_hat).
    u_t = (v[0] - radius * omega[1], v[1] + radius * omega[0])
    gain = physics.table_a_t + physics.table_b_t
    desired_j_t = (-mass * gain * u_t[0], -mass * gain * u_t[1])
    j_n = -mass * (1.0 + physics.table_e_eff) * v[2]
    cap = physics.table_mu_safety * abs(j_n)
    norm_j_t = math.hypot(*desired_j_t)
    if norm_j_t > cap > 0.0:
        scale = cap / norm_j_t
        j_t = (desired_j_t[0] * scale, desired_j_t[1] * scale)
    else:
        j_t = desired_j_t
    v_after = (
        v[0] + j_t[0] / mass,
        v[1] + j_t[1] / mass,
        v[2] + j_n / mass,
    )
    # r_contact x J / I, with r_contact=(0,0,-r).
    omega_after = (
        omega[0] + radius * j_t[1] / inertia,
        omega[1] - radius * j_t[0] / inertia,
        omega[2],
    )
    return v_after, omega_after


@dataclass(frozen=True)
class CounterRallyOutcome:
    net_crossed: bool
    net_clear: bool
    first_landing_valid: bool
    first_landing_env_xy_m: Vec2 | None
    first_landing_time_s: float | None
    table_bounce_count: int
    opponent_baseline_crossed: bool
    baseline_cross_env_yz_m: Vec2 | None
    baseline_velocity_mps: Vec3 | None
    baseline_time_s: float | None
    post_hit_direction_env_xy: Vec2 | None
    rejection_reason: str | None


def _flight_acceleration(
    velocity: "object", spin: "object", physics: VenueBallPhysics
) -> "object":
    import numpy as np

    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    gravity = np.zeros_like(velocity)
    gravity[..., 2] = -physics.gravity_mps2
    return (
        gravity
        - physics.drag_k_d_per_m * speed * velocity
        + physics.magnus_k_m * np.cross(spin, velocity)
    )


def _rk4_step(
    position: "object",
    velocity: "object",
    spin: "object",
    dt: float,
    physics: VenueBallPhysics,
) -> Tuple["object", "object"]:
    k1p = velocity
    k1v = _flight_acceleration(velocity, spin, physics)
    k2p = velocity + 0.5 * dt * k1v
    k2v = _flight_acceleration(velocity + 0.5 * dt * k1v, spin, physics)
    k3p = velocity + 0.5 * dt * k2v
    k3v = _flight_acceleration(velocity + 0.5 * dt * k2v, spin, physics)
    k4p = velocity + dt * k3v
    k4v = _flight_acceleration(velocity + dt * k3v, spin, physics)
    return (
        position + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        velocity + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )


def rollout_counter_rally_batch(
    *,
    position_after_paddle_env_m: Sequence[Sequence[float]],
    velocity_after_paddle_mps: Sequence[Sequence[float]],
    spin_after_paddle_radps: Sequence[Sequence[float]],
    profile: CounterRallyObjectiveProfile,
    physics: VenueBallPhysics,
    dt_s: float = 0.001,
    max_time_s: float = 2.0,
) -> Tuple[CounterRallyOutcome, ...]:
    """Batched NumPy reference from post-paddle state through baseline."""

    import numpy as np

    dt = _finite(dt_s, name="dt_s")
    max_time = _finite(max_time_s, name="max_time_s")
    if dt <= 0.0 or max_time <= dt:
        raise ValueError("invalid rollout horizon")
    p = np.asarray(position_after_paddle_env_m, dtype=np.float64)
    v = np.asarray(velocity_after_paddle_mps, dtype=np.float64)
    omega = np.asarray(spin_after_paddle_radps, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or v.shape != p.shape or omega.shape != p.shape:
        raise ValueError("post-paddle state arrays must all have shape [N,3]")
    if not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(omega).all():
        raise ValueError("post-paddle state arrays must be finite")
    n = p.shape[0]
    active = np.ones(n, dtype=bool)
    net_crossed = np.zeros(n, dtype=bool)
    net_clear = np.zeros(n, dtype=bool)
    landed = np.zeros(n, dtype=bool)
    landing_valid = np.zeros(n, dtype=bool)
    bounced = np.zeros(n, dtype=np.int64)
    baseline_crossed = np.zeros(n, dtype=bool)
    landing_xy = np.full((n, 2), np.nan)
    landing_time = np.full(n, np.nan)
    baseline_yz = np.full((n, 2), np.nan)
    baseline_velocity = np.full((n, 3), np.nan)
    baseline_time = np.full(n, np.nan)
    reasons = [None] * n
    initial_direction = np.full((n, 2), np.nan)
    horizontal_norm = np.linalg.norm(v[:, :2], axis=1)
    valid_dir = horizontal_norm > 1.0e-9
    initial_direction[valid_dir] = v[valid_dir, :2] / horizontal_norm[valid_dir, None]
    threshold_z = profile.table_surface_z_env_m + physics.ball_radius_m
    net_x = profile.table_near_x_env_m + 0.5 * profile.table_length_m
    net_threshold = (
        profile.table_surface_z_env_m
        + profile.net_height_m
        + physics.ball_radius_m
    )
    table_lo = profile.table_near_x_env_m + profile.table_edge_margin_m
    table_hi = profile.opponent_baseline_x_env_m - profile.table_edge_margin_m
    half_width = profile.table_half_width_m - profile.table_edge_margin_m
    time = 0.0
    for _ in range(int(math.ceil(max_time / dt))):
        if not active.any():
            break
        old_p = p.copy()
        old_v = v.copy()
        new_p, new_v = _rk4_step(p, v, omega, dt, physics)
        time += dt
        # Interpolate net crossing before the first landing.
        cross_net = (
            active
            & ~net_crossed
            & (old_p[:, 0] < net_x)
            & (new_p[:, 0] >= net_x)
        )
        for i in np.flatnonzero(cross_net):
            alpha = (net_x - old_p[i, 0]) / (new_p[i, 0] - old_p[i, 0])
            z = old_p[i, 2] + alpha * (new_p[i, 2] - old_p[i, 2])
            net_crossed[i] = True
            net_clear[i] = bool(z >= net_threshold)
            if not net_clear[i]:
                active[i] = False
                reasons[i] = "net_not_clear"
        # First/second descending crossing of the table plane.
        cross_surface = (
            active
            & (old_p[:, 2] > threshold_z)
            & (new_p[:, 2] <= threshold_z)
            & (new_v[:, 2] < 0.0)
        )
        for i in np.flatnonzero(cross_surface):
            alpha = (old_p[i, 2] - threshold_z) / (
                old_p[i, 2] - new_p[i, 2]
            )
            hit_xy = old_p[i, :2] + alpha * (new_p[i, :2] - old_p[i, :2])
            if not landed[i]:
                landed[i] = True
                landing_xy[i] = hit_xy
                landing_time[i] = time - dt + alpha * dt
                inside = (
                    table_lo <= hit_xy[0] <= table_hi
                    and abs(hit_xy[1]) <= half_width
                )
                landing_valid[i] = inside
                if not inside:
                    active[i] = False
                    reasons[i] = "first_landing_outside_table"
                    continue
                if hit_xy[0] <= net_x or not net_crossed[i]:
                    active[i] = False
                    reasons[i] = "first_landing_own_half"
                    continue
                hit_v = old_v[i] + alpha * (new_v[i] - old_v[i])
                try:
                    out_v, out_omega = fitted_table_impulse(
                        velocity_before_mps=hit_v,
                        spin_before_radps=omega[i],
                        physics=physics,
                    )
                except CounterRallyRejected as exc:
                    active[i] = False
                    reasons[i] = exc.reason
                    continue
                p[i] = np.array((hit_xy[0], hit_xy[1], threshold_z + 1.0e-9))
                v[i] = np.asarray(out_v)
                omega[i] = np.asarray(out_omega)
                bounced[i] = 1
                continue
            active[i] = False
            reasons[i] = "second_table_bounce_before_baseline"
        # Preserve bounce-updated rows; advance all others.
        advanced = active & ~cross_surface
        p[advanced] = new_p[advanced]
        v[advanced] = new_v[advanced]
        # Opponent baseline crossing after exactly one legal bounce.
        cross_base = (
            active
            & landed
            & (bounced == 1)
            & (old_p[:, 0] < profile.opponent_baseline_x_env_m)
            & (p[:, 0] >= profile.opponent_baseline_x_env_m)
        )
        for i in np.flatnonzero(cross_base):
            denominator = p[i, 0] - old_p[i, 0]
            alpha = (
                1.0
                if abs(denominator) <= 1.0e-12
                else (profile.opponent_baseline_x_env_m - old_p[i, 0])
                / denominator
            )
            baseline_yz[i] = old_p[i, 1:] + alpha * (p[i, 1:] - old_p[i, 1:])
            baseline_velocity[i] = old_v[i] + alpha * (v[i] - old_v[i])
            baseline_time[i] = time - dt + alpha * dt
            baseline_crossed[i] = True
            active[i] = False
        # Fail closed on leaving the forward venue or horizon.
        backward = active & (p[:, 0] < profile.table_near_x_env_m - 0.5)
        for i in np.flatnonzero(backward):
            active[i] = False
            reasons[i] = "trajectory_left_venue"
    for i in np.flatnonzero(active):
        reasons[i] = "rollout_horizon_exceeded"
    result = []
    for i in range(n):
        result.append(
            CounterRallyOutcome(
                net_crossed=bool(net_crossed[i]),
                net_clear=bool(net_clear[i]),
                first_landing_valid=bool(landing_valid[i]),
                first_landing_env_xy_m=(
                    None
                    if not landed[i]
                    else (float(landing_xy[i, 0]), float(landing_xy[i, 1]))
                ),
                first_landing_time_s=(
                    None if not landed[i] else float(landing_time[i])
                ),
                table_bounce_count=int(bounced[i]),
                opponent_baseline_crossed=bool(baseline_crossed[i]),
                baseline_cross_env_yz_m=(
                    None
                    if not baseline_crossed[i]
                    else (float(baseline_yz[i, 0]), float(baseline_yz[i, 1]))
                ),
                baseline_velocity_mps=(
                    None
                    if not baseline_crossed[i]
                    else tuple(float(x) for x in baseline_velocity[i])
                ),
                baseline_time_s=(
                    None if not baseline_crossed[i] else float(baseline_time[i])
                ),
                post_hit_direction_env_xy=(
                    None
                    if not valid_dir[i]
                    else (
                        float(initial_direction[i, 0]),
                        float(initial_direction[i, 1]),
                    )
                ),
                rejection_reason=(
                    None if baseline_crossed[i] else reasons[i]
                ),
            )
        )
    return tuple(result)


def rollout_counter_rally_eager(
    *,
    position_after_paddle_env_m: Sequence[float],
    velocity_after_paddle_mps: Sequence[float],
    spin_after_paddle_radps: Sequence[float],
    profile: CounterRallyObjectiveProfile,
    physics: VenueBallPhysics,
    dt_s: float = 0.001,
    max_time_s: float = 2.0,
) -> CounterRallyOutcome:
    return rollout_counter_rally_batch(
        position_after_paddle_env_m=(position_after_paddle_env_m,),
        velocity_after_paddle_mps=(velocity_after_paddle_mps,),
        spin_after_paddle_radps=(spin_after_paddle_radps,),
        profile=profile,
        physics=physics,
        dt_s=dt_s,
        max_time_s=max_time_s,
    )[0]


def _angle_deg(a: Vec2, b: Vec2) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    cross = a[0] * b[1] - a[1] * b[0]
    return abs(math.degrees(math.atan2(cross, dot)))


@dataclass(frozen=True)
class CounterRallyAssessment:
    accepted: bool
    reasons: Tuple[str, ...]
    landing_error_m: float | None
    reverse_direction_error_deg: float | None
    baseline_direction_error_deg: float | None
    baseline_speed_mps: float | None
    baseline_speed_error_mps: float | None


def assess_counter_rally_outcome(
    *,
    task: CounterRallyTask,
    outcome: CounterRallyOutcome,
    profile: CounterRallyObjectiveProfile,
) -> CounterRallyAssessment:
    if task.objective_profile_sha256 != profile.sha256:
        raise CounterRallyIdentityError(
            "objective_profile_sha256_mismatch"
        )
    reasons = []
    landing_error = None
    reverse_error = None
    baseline_direction_error = None
    baseline_speed = None
    baseline_speed_error = None
    if not outcome.net_crossed:
        reasons.append("net_not_crossed")
    if not outcome.net_clear:
        reasons.append("net_not_clear")
    net_x = profile.table_near_x_env_m + 0.5 * profile.table_length_m
    landing_valid = bool(
        outcome.first_landing_valid
        and outcome.first_landing_env_xy_m is not None
    )
    if not landing_valid:
        reasons.append("first_landing_invalid")
    elif outcome.first_landing_env_xy_m[0] <= net_x:
        reasons.append("first_landing_not_opponent_half")
    landing_stage_valid = bool(
        outcome.net_crossed
        and outcome.net_clear
        and landing_valid
        and outcome.first_landing_env_xy_m[0] > net_x
    )
    if landing_stage_valid:
        landing_error = math.dist(
            outcome.first_landing_env_xy_m, task.landing_aim_env_xy_m
        )
        if landing_error > profile.landing_tolerance_m:
            reasons.append("landing_aim_miss")
    if outcome.table_bounce_count != 1:
        reasons.append("table_bounce_count_not_one")
    legal_first_landing = bool(
        landing_stage_valid
        and outcome.table_bounce_count == 1
    )
    baseline_valid = bool(
        legal_first_landing
        and outcome.opponent_baseline_crossed
        and outcome.baseline_velocity_mps is not None
    )
    if legal_first_landing and not baseline_valid:
        reasons.append("opponent_baseline_not_crossed")
    if baseline_valid:
        if outcome.post_hit_direction_env_xy is None:
            reasons.append("post_hit_direction_missing")
        else:
            reverse_error = _angle_deg(
                outcome.post_hit_direction_env_xy,
                task.return_direction_env_xy,
            )
            if reverse_error > profile.reverse_direction_tolerance_deg:
                reasons.append("post_hit_reverse_direction_miss")
        velocity = outcome.baseline_velocity_mps
        baseline_speed = math.sqrt(sum(component * component for component in velocity))
        baseline_speed_error = abs(
            baseline_speed - task.target_baseline_speed_mps
        )
        speed_tolerance = max(
            profile.target_speed_abs_tolerance_mps,
            profile.target_speed_rel_tolerance
            * task.target_baseline_speed_mps,
        )
        if baseline_speed_error > speed_tolerance:
            reasons.append("baseline_speed_miss")
        horizontal = _unit2(velocity[:2], name="baseline_velocity")
        baseline_direction_error = _angle_deg(
            horizontal, task.return_direction_env_xy
        )
        if baseline_direction_error > profile.baseline_direction_tolerance_deg:
            reasons.append("baseline_direction_miss")
    return CounterRallyAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        landing_error_m=landing_error,
        reverse_direction_error_deg=reverse_error,
        baseline_direction_error_deg=baseline_direction_error,
        baseline_speed_mps=baseline_speed,
        baseline_speed_error_mps=baseline_speed_error,
    )


def counter_rally_reward_raw(
    *,
    task: CounterRallyTask,
    outcome: CounterRallyOutcome,
    profile: CounterRallyObjectiveProfile,
) -> Mapping[str, float]:
    """Return bounded raw [0,1] terms; weighting remains an external recipe."""

    if task.objective_profile_sha256 != profile.sha256:
        raise CounterRallyIdentityError(
            "objective_profile_sha256_mismatch"
        )
    net_x = profile.table_near_x_env_m + 0.5 * profile.table_length_m
    legal_first_landing = bool(
        outcome.net_crossed
        and outcome.net_clear
        and outcome.first_landing_valid
        and outcome.first_landing_env_xy_m is not None
        and outcome.first_landing_env_xy_m[0] > net_x
        and outcome.table_bounce_count == 1
    )
    baseline_valid = bool(
        legal_first_landing
        and outcome.opponent_baseline_crossed
        and outcome.baseline_velocity_mps is not None
    )
    legal = float(legal_first_landing)
    landing = 0.0
    if legal_first_landing:
        error = math.dist(
            outcome.first_landing_env_xy_m, task.landing_aim_env_xy_m
        )
        landing = math.exp(-((error / profile.landing_tolerance_m) ** 2))
    reverse = 0.0
    if baseline_valid and outcome.post_hit_direction_env_xy is not None:
        angle = _angle_deg(
            outcome.post_hit_direction_env_xy,
            task.return_direction_env_xy,
        )
        reverse = math.exp(
            -((angle / profile.reverse_direction_tolerance_deg) ** 2)
        )
    speed = 0.0
    if baseline_valid:
        measured = math.sqrt(
            sum(component * component for component in outcome.baseline_velocity_mps)
        )
        tolerance = max(
            profile.target_speed_abs_tolerance_mps,
            profile.target_speed_rel_tolerance
            * task.target_baseline_speed_mps,
        )
        speed = math.exp(
            -((abs(measured - task.target_baseline_speed_mps) / tolerance) ** 2)
        )
    total = (
        profile.reward_legal_fraction * legal
        + profile.reward_landing_fraction * landing
        + profile.reward_reverse_fraction * reverse
        + profile.reward_speed_fraction * speed
    )
    return {
        "legal": legal,
        "landing": landing,
        "reverse": reverse,
        "speed": speed,
        "total": max(0.0, min(1.0, total)),
    }
