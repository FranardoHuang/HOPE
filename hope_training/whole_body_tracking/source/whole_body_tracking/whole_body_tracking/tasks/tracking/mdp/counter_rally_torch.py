"""Batched Torch mirror of :mod:`counter_rally` venue physics.

The implementation is intentionally self-contained: it does not call the
NumPy/CPU oracle and it does not depend on an Isaac scene.  This lets launch
preflight compare the trainer-side batched result against the pure CPU oracle
at both 1 ms and 0.5 ms before the objective is allowed into a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Mapping, Sequence, Tuple

import torch


REJECTION_REASONS: Tuple[str, ...] = (
    "none",
    "net_not_clear",
    "first_landing_outside_table",
    "first_landing_own_half",
    "table_contact_not_descending",
    "second_table_bounce_before_baseline",
    "trajectory_left_venue",
    "rollout_horizon_exceeded",
)
_REASON = {name: index for index, name in enumerate(REJECTION_REASONS)}
OUTCOME_PRIMARY_REASONS: Tuple[str, ...] = (
    "accepted",
    "paddle_contact_missing",
    "net_not_crossed",
    "net_not_clear",
    "first_landing_invalid",
    "first_landing_not_opponent_half",
    "landing_aim_miss",
    "table_bounce_count_not_one",
    "opponent_baseline_not_crossed",
    "post_hit_direction_missing",
    "post_hit_reverse_direction_miss",
    "baseline_speed_miss",
    "baseline_direction_miss",
)
_OUTCOME_REASON = {
    name: index for index, name in enumerate(OUTCOME_PRIMARY_REASONS)
}
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
_PHYSICS_KEYS = frozenset(
    (
        "ball_radius_m",
        "ball_mass_kg",
        "ball_inertia_coeff",
        "gravity_mps2",
        "drag_k_d_per_m",
        "magnus_k_m",
        "table_e_eff",
        "table_a_t",
        "table_b_t",
        "table_mu_safety",
    )
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return value


def _finite_mapping_number(
    mapping: Mapping[str, object],
    key: str,
) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


@dataclass(frozen=True)
class CounterRallyTorchBinding:
    """One canonical objective+physics contract for every Torch hot path."""

    objective_profile_sha256: str
    venue_physics_sha256: str
    profile_canonical_json: str
    physics_canonical_json: str
    table_near_x_env_m: float
    table_length_m: float
    table_half_width_m: float
    table_surface_z_env_m: float
    net_height_m: float
    table_edge_margin_m: float
    landing_tolerance_m: float
    reverse_direction_tolerance_deg: float
    baseline_direction_tolerance_deg: float
    target_speed_abs_tolerance_mps: float
    target_speed_rel_tolerance: float
    reward_legal_fraction: float
    reward_landing_fraction: float
    reward_reverse_fraction: float
    reward_speed_fraction: float
    ball_radius_m: float
    ball_mass_kg: float
    ball_inertia_coeff: float
    gravity_mps2: float
    drag_k_d_per_m: float
    magnus_k_m: float
    table_e_eff: float
    table_a_t: float
    table_b_t: float
    table_mu_safety: float

    def __post_init__(self) -> None:
        objective_sha = _valid_sha256(
            self.objective_profile_sha256,
            name="objective_profile_sha256",
        )
        physics_sha = _valid_sha256(
            self.venue_physics_sha256,
            name="venue_physics_sha256",
        )
        try:
            profile = json.loads(self.profile_canonical_json)
            physics = json.loads(self.physics_canonical_json)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "counter-rally canonical mappings are invalid JSON"
            ) from error
        if (
            not isinstance(profile, Mapping)
            or frozenset(profile) != _PROFILE_KEYS
            or _canonical_json(profile) != self.profile_canonical_json
        ):
            raise ValueError(
                "counter-rally profile canonical JSON/schema mismatch"
            )
        if (
            not isinstance(physics, Mapping)
            or frozenset(physics) != _PHYSICS_KEYS
            or _canonical_json(physics) != self.physics_canonical_json
        ):
            raise ValueError(
                "counter-rally physics canonical JSON/schema mismatch"
            )
        if _canonical_sha256(profile) != objective_sha:
            raise CounterRallyTorchIdentityError(
                "objective_profile_sha256_mismatch"
            )
        if _canonical_sha256(physics) != physics_sha:
            raise CounterRallyTorchIdentityError(
                "venue_physics_sha256_mismatch"
            )
        if profile["mode"] != "counter_rally_v1":
            raise ValueError("counter-rally objective mode mismatch")
        if profile["inactive_curriculum_arms"] != [
            "landing_aim_y_lower",
            "landing_aim_y_upper",
        ]:
            raise ValueError(
                "counter-rally inactive curriculum arms mismatch"
            )
        profile_fields = (
            "table_near_x_env_m",
            "table_length_m",
            "table_half_width_m",
            "table_surface_z_env_m",
            "net_height_m",
            "table_edge_margin_m",
            "landing_tolerance_m",
            "reverse_direction_tolerance_deg",
            "baseline_direction_tolerance_deg",
            "target_speed_abs_tolerance_mps",
            "target_speed_rel_tolerance",
            "reward_legal_fraction",
            "reward_landing_fraction",
            "reward_reverse_fraction",
            "reward_speed_fraction",
        )
        physics_fields = tuple(sorted(_PHYSICS_KEYS))
        for name in profile_fields:
            expected = _finite_mapping_number(profile, name)
            if getattr(self, name) != expected:
                raise CounterRallyTorchIdentityError(
                    "objective_profile_scalar_mismatch:" + name
                )
        for name in physics_fields:
            expected = _finite_mapping_number(physics, name)
            if getattr(self, name) != expected:
                raise CounterRallyTorchIdentityError(
                    "venue_physics_scalar_mismatch:" + name
                )
        weights = (
            self.reward_legal_fraction,
            self.reward_landing_fraction,
            self.reward_reverse_fraction,
            self.reward_speed_fraction,
        )
        if any(value < 0.0 for value in weights) or abs(
            sum(weights) - 1.0
        ) > 1.0e-12:
            raise ValueError(
                "counter-rally reward fractions must be nonnegative "
                "and sum to one"
            )

    @classmethod
    def from_mappings(
        cls,
        *,
        objective_profile: Mapping[str, object],
        venue_physics: Mapping[str, object],
        expected_objective_profile_sha256: str,
        expected_venue_physics_sha256: str,
    ) -> "CounterRallyTorchBinding":
        if (
            not isinstance(objective_profile, Mapping)
            or frozenset(objective_profile) != _PROFILE_KEYS
        ):
            raise ValueError(
                "objective_profile must have the canonical strict schema"
            )
        if (
            not isinstance(venue_physics, Mapping)
            or frozenset(venue_physics) != _PHYSICS_KEYS
        ):
            raise ValueError(
                "venue_physics must have the canonical strict schema"
            )
        expected_objective = _valid_sha256(
            expected_objective_profile_sha256,
            name="expected_objective_profile_sha256",
        )
        expected_physics = _valid_sha256(
            expected_venue_physics_sha256,
            name="expected_venue_physics_sha256",
        )
        if _canonical_sha256(objective_profile) != expected_objective:
            raise CounterRallyTorchIdentityError(
                "objective_profile_sha256_mismatch"
            )
        if _canonical_sha256(venue_physics) != expected_physics:
            raise CounterRallyTorchIdentityError(
                "venue_physics_sha256_mismatch"
            )
        values = {
            name: _finite_mapping_number(objective_profile, name)
            for name in (
                "table_near_x_env_m",
                "table_length_m",
                "table_half_width_m",
                "table_surface_z_env_m",
                "net_height_m",
                "table_edge_margin_m",
                "landing_tolerance_m",
                "reverse_direction_tolerance_deg",
                "baseline_direction_tolerance_deg",
                "target_speed_abs_tolerance_mps",
                "target_speed_rel_tolerance",
                "reward_legal_fraction",
                "reward_landing_fraction",
                "reward_reverse_fraction",
                "reward_speed_fraction",
            )
        }
        values.update(
            {
                name: _finite_mapping_number(venue_physics, name)
                for name in _PHYSICS_KEYS
            }
        )
        return cls(
            objective_profile_sha256=expected_objective,
            venue_physics_sha256=expected_physics,
            profile_canonical_json=_canonical_json(objective_profile),
            physics_canonical_json=_canonical_json(venue_physics),
            **values,
        )


@dataclass(frozen=True)
class CounterRallyTorchOutcome:
    net_crossed: torch.Tensor
    net_clear: torch.Tensor
    first_landing_valid: torch.Tensor
    first_landing_env_xy_m: torch.Tensor
    first_landing_time_s: torch.Tensor
    table_bounce_count: torch.Tensor
    opponent_baseline_crossed: torch.Tensor
    baseline_cross_env_yz_m: torch.Tensor
    baseline_velocity_mps: torch.Tensor
    baseline_time_s: torch.Tensor
    post_hit_direction_env_xy: torch.Tensor
    rejection_reason_code: torch.Tensor

    @property
    def rejection_reasons(self) -> Tuple[str, ...]:
        return tuple(
            REJECTION_REASONS[int(code)]
            for code in self.rejection_reason_code.detach().cpu().tolist()
        )


class CounterRallyTorchIdentityError(RuntimeError):
    """Frozen objective identity drift; never a policy/difficulty failure."""


@dataclass(frozen=True)
class CounterRallyTorchOutcomeGates:
    """Full production admission assessment under one verified objective."""

    objective_profile_sha256: str
    legal_first_landing: torch.Tensor
    baseline_valid: torch.Tensor
    accepted: torch.Tensor
    primary_reason_code: torch.Tensor
    landing_error_m: torch.Tensor
    reverse_direction_error_deg: torch.Tensor
    baseline_direction_error_deg: torch.Tensor
    baseline_speed_mps: torch.Tensor
    baseline_speed_error_mps: torch.Tensor

    @property
    def primary_reasons(self) -> Tuple[str, ...]:
        return tuple(
            OUTCOME_PRIMARY_REASONS[int(code)]
            for code in self.primary_reason_code.detach().cpu().tolist()
        )


def _validate_objective_identity(
    *,
    task_objective_profile_sha256: Sequence[str],
    objective_profile_sha256: str,
    batch_size: int,
) -> None:
    if (
        type(objective_profile_sha256) is not str
        or len(objective_profile_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in objective_profile_sha256
        )
    ):
        raise ValueError(
            "objective_profile_sha256 must be 64 lowercase hex characters"
        )
    if isinstance(task_objective_profile_sha256, (str, bytes)):
        raise ValueError(
            "task_objective_profile_sha256 must contain one SHA per row"
        )
    task_profile_shas = tuple(task_objective_profile_sha256)
    if len(task_profile_shas) != batch_size or any(
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in task_profile_shas
    ):
        raise ValueError(
            "task_objective_profile_sha256 must contain one valid SHA per row"
        )
    mismatched_rows = tuple(
        index
        for index, value in enumerate(task_profile_shas)
        if value != objective_profile_sha256
    )
    if mismatched_rows:
        raise CounterRallyTorchIdentityError(
            "objective_profile_sha256_mismatch at rows "
            + ",".join(str(index) for index in mismatched_rows)
        )


def counter_rally_outcome_gates_torch(
    *,
    outcome: CounterRallyTorchOutcome,
    binding: CounterRallyTorchBinding,
    landing_aim_env_xy_m: torch.Tensor,
    return_direction_env_xy: torch.Tensor,
    target_baseline_speed_mps: torch.Tensor,
    paddle_contact_valid: torch.Tensor,
    task_objective_profile_sha256: Sequence[str],
) -> CounterRallyTorchOutcomeGates:
    """Validate identity and return CPU-parity hard admission diagnostics."""

    if not isinstance(binding, CounterRallyTorchBinding):
        raise TypeError("binding must be CounterRallyTorchBinding")
    batch_size = int(outcome.net_crossed.numel())
    vector_shapes = (
        outcome.net_crossed.shape,
        outcome.net_clear.shape,
        outcome.first_landing_valid.shape,
        outcome.table_bounce_count.shape,
        outcome.opponent_baseline_crossed.shape,
    )
    if any(shape != (batch_size,) for shape in vector_shapes):
        raise ValueError("counter-rally outcome masks must all have shape [N]")
    if outcome.first_landing_env_xy_m.shape != (batch_size, 2):
        raise ValueError(
            "counter-rally first landing positions must have shape [N,2]"
        )
    if (
        landing_aim_env_xy_m.shape != (batch_size, 2)
        or return_direction_env_xy.shape != (batch_size, 2)
        or target_baseline_speed_mps.shape != (batch_size,)
        or paddle_contact_valid.shape != (batch_size,)
    ):
        raise ValueError(
            "counter-rally task tensors must have [N,2]/[N,2]/[N] shapes"
        )
    task_tensors = (
        landing_aim_env_xy_m,
        return_direction_env_xy,
        target_baseline_speed_mps,
        paddle_contact_valid,
    )
    outcome_tensors = (
        outcome.net_crossed,
        outcome.net_clear,
        outcome.first_landing_valid,
        outcome.first_landing_env_xy_m,
        outcome.table_bounce_count,
        outcome.opponent_baseline_crossed,
        outcome.baseline_velocity_mps,
        outcome.post_hit_direction_env_xy,
    )
    if any(
        tensor.device != landing_aim_env_xy_m.device
        for tensor in (*task_tensors, *outcome_tensors)
    ):
        raise ValueError(
            "counter-rally task/outcome tensors must share one device"
        )
    if (
        return_direction_env_xy.dtype != landing_aim_env_xy_m.dtype
        or target_baseline_speed_mps.dtype
        != landing_aim_env_xy_m.dtype
    ):
        raise ValueError(
            "counter-rally task tensors must share one floating dtype"
        )
    if not landing_aim_env_xy_m.dtype.is_floating_point:
        raise ValueError("counter-rally task tensors must be floating")
    if paddle_contact_valid.dtype != torch.bool:
        raise ValueError("paddle_contact_valid must be a bool [N] tensor")
    if not bool(
        torch.isfinite(landing_aim_env_xy_m).all()
        and torch.isfinite(return_direction_env_xy).all()
        and torch.isfinite(target_baseline_speed_mps).all()
    ):
        raise ValueError("counter-rally task tensors must be finite")
    direction_norm = torch.linalg.norm(
        return_direction_env_xy, dim=-1
    )
    if not bool(
        (
            (direction_norm - 1.0).abs() <= 1.0e-6
        ).all()
        and (return_direction_env_xy[:, 0] > 0.0).all()
    ):
        raise ValueError(
            "counter-rally return directions must be unit and opponent-bound"
        )
    if not bool((target_baseline_speed_mps > 0.0).all()):
        raise ValueError(
            "counter-rally target baseline speeds must be positive"
        )
    _validate_objective_identity(
        task_objective_profile_sha256=task_objective_profile_sha256,
        objective_profile_sha256=binding.objective_profile_sha256,
        batch_size=batch_size,
    )
    net_x = float(
        binding.table_near_x_env_m + 0.5 * binding.table_length_m
    )
    landing_stage_valid = (
        paddle_contact_valid
        & outcome.net_crossed
        & outcome.net_clear
        & outcome.first_landing_valid
        & (outcome.first_landing_env_xy_m[:, 0] > net_x)
    )
    legal = landing_stage_valid & (outcome.table_bounce_count == 1)
    baseline_data_valid = torch.isfinite(
        outcome.baseline_velocity_mps
    ).all(dim=-1)
    baseline_valid = legal & outcome.opponent_baseline_crossed
    baseline_valid = baseline_valid & baseline_data_valid

    nan = torch.full(
        (batch_size,),
        float("nan"),
        dtype=landing_aim_env_xy_m.dtype,
        device=landing_aim_env_xy_m.device,
    )
    landing_error_raw = torch.linalg.norm(
        outcome.first_landing_env_xy_m - landing_aim_env_xy_m,
        dim=-1,
    )
    landing_error = torch.where(
        landing_stage_valid, landing_error_raw, nan
    )
    post_hit_valid = torch.isfinite(
        outcome.post_hit_direction_env_xy
    ).all(dim=-1)
    post_hit_direction = torch.nan_to_num(
        outcome.post_hit_direction_env_xy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    reverse_dot = torch.sum(
        post_hit_direction * return_direction_env_xy,
        dim=-1,
    ).clamp(-1.0, 1.0)
    reverse_cross = (
        post_hit_direction[:, 0] * return_direction_env_xy[:, 1]
        - post_hit_direction[:, 1] * return_direction_env_xy[:, 0]
    )
    reverse_error_raw = torch.rad2deg(
        torch.atan2(reverse_cross.abs(), reverse_dot)
    )
    reverse_error = torch.where(
        baseline_valid & post_hit_valid,
        reverse_error_raw,
        nan,
    )
    baseline_velocity = torch.nan_to_num(
        outcome.baseline_velocity_mps,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    baseline_speed_raw = torch.linalg.norm(
        baseline_velocity, dim=-1
    )
    baseline_speed = torch.where(
        baseline_valid, baseline_speed_raw, nan
    )
    baseline_speed_error = torch.where(
        baseline_valid,
        (baseline_speed_raw - target_baseline_speed_mps).abs(),
        nan,
    )
    baseline_horizontal_norm = torch.linalg.norm(
        baseline_velocity[:, :2], dim=-1
    )
    baseline_direction_valid = (
        baseline_valid & (baseline_horizontal_norm > 1.0e-12)
    )
    baseline_direction = (
        baseline_velocity[:, :2]
        / baseline_horizontal_norm.clamp_min(
            torch.finfo(baseline_velocity.dtype).tiny
        )[:, None]
    )
    baseline_dot = torch.sum(
        baseline_direction * return_direction_env_xy,
        dim=-1,
    ).clamp(-1.0, 1.0)
    baseline_cross = (
        baseline_direction[:, 0] * return_direction_env_xy[:, 1]
        - baseline_direction[:, 1] * return_direction_env_xy[:, 0]
    )
    baseline_direction_error_raw = torch.rad2deg(
        torch.atan2(baseline_cross.abs(), baseline_dot)
    )
    baseline_direction_error = torch.where(
        baseline_direction_valid,
        baseline_direction_error_raw,
        nan,
    )
    speed_tolerance = torch.maximum(
        torch.full_like(
            target_baseline_speed_mps,
            binding.target_speed_abs_tolerance_mps,
        ),
        binding.target_speed_rel_tolerance
        * target_baseline_speed_mps,
    )

    primary_reason = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=landing_aim_env_xy_m.device,
    )

    def set_primary(mask: torch.Tensor, reason: str) -> None:
        nonlocal primary_reason
        select = (primary_reason == 0) & mask
        primary_reason = torch.where(
            select,
            torch.full_like(
                primary_reason, _OUTCOME_REASON[reason]
            ),
            primary_reason,
        )

    set_primary(~paddle_contact_valid, "paddle_contact_missing")
    set_primary(
        paddle_contact_valid & ~outcome.net_crossed,
        "net_not_crossed",
    )
    set_primary(~outcome.net_clear, "net_not_clear")
    set_primary(
        ~outcome.first_landing_valid, "first_landing_invalid"
    )
    set_primary(
        outcome.first_landing_valid
        & (outcome.first_landing_env_xy_m[:, 0] <= net_x),
        "first_landing_not_opponent_half",
    )
    set_primary(
        landing_stage_valid
        & (landing_error_raw > binding.landing_tolerance_m),
        "landing_aim_miss",
    )
    set_primary(
        outcome.table_bounce_count != 1,
        "table_bounce_count_not_one",
    )
    set_primary(
        legal
        & (~outcome.opponent_baseline_crossed | ~baseline_data_valid),
        "opponent_baseline_not_crossed",
    )
    set_primary(
        baseline_valid & ~post_hit_valid,
        "post_hit_direction_missing",
    )
    set_primary(
        baseline_valid
        & post_hit_valid
        & (
            reverse_error_raw
            > binding.reverse_direction_tolerance_deg
        ),
        "post_hit_reverse_direction_miss",
    )
    set_primary(
        baseline_valid
        & (
            (baseline_speed_raw - target_baseline_speed_mps).abs()
            > speed_tolerance
        ),
        "baseline_speed_miss",
    )
    set_primary(
        baseline_valid
        & (
            ~baseline_direction_valid
            | (
                baseline_direction_error_raw
                > binding.baseline_direction_tolerance_deg
            )
        ),
        "baseline_direction_miss",
    )
    accepted = primary_reason == _OUTCOME_REASON["accepted"]
    return CounterRallyTorchOutcomeGates(
        objective_profile_sha256=binding.objective_profile_sha256,
        legal_first_landing=legal,
        baseline_valid=baseline_valid,
        accepted=accepted,
        primary_reason_code=primary_reason,
        landing_error_m=landing_error,
        reverse_direction_error_deg=reverse_error,
        baseline_direction_error_deg=baseline_direction_error,
        baseline_speed_mps=baseline_speed,
        baseline_speed_error_mps=baseline_speed_error,
    )


def _acceleration(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    *,
    gravity_mps2: float,
    drag_k_d_per_m: float,
    magnus_k_m: float,
) -> torch.Tensor:
    speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
    gravity = torch.zeros_like(velocity)
    gravity[:, 2] = -float(gravity_mps2)
    return (
        gravity
        - float(drag_k_d_per_m) * speed * velocity
        + float(magnus_k_m) * torch.cross(spin, velocity, dim=-1)
    )


def _rk4_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    *,
    dt_s: float,
    gravity_mps2: float,
    drag_k_d_per_m: float,
    magnus_k_m: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = float(dt_s)

    def acceleration(value: torch.Tensor) -> torch.Tensor:
        return _acceleration(
            value,
            spin,
            gravity_mps2=gravity_mps2,
            drag_k_d_per_m=drag_k_d_per_m,
            magnus_k_m=magnus_k_m,
        )

    k1p = velocity
    k1v = acceleration(velocity)
    k2p = velocity + 0.5 * dt * k1v
    k2v = acceleration(velocity + 0.5 * dt * k1v)
    k3p = velocity + 0.5 * dt * k2v
    k3v = acceleration(velocity + 0.5 * dt * k2v)
    k4p = velocity + dt * k3v
    k4v = acceleration(velocity + dt * k3v)
    return (
        position + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        velocity + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )


def fitted_table_impulse_torch(
    velocity_before_mps: torch.Tensor,
    spin_before_radps: torch.Tensor,
    *,
    ball_radius_m: float,
    ball_mass_kg: float,
    ball_inertia_coeff: float,
    table_e_eff: float,
    table_a_t: float,
    table_b_t: float,
    table_mu_safety: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized fitted table impulse plus a descending-contact mask."""

    if (
        velocity_before_mps.ndim != 2
        or velocity_before_mps.shape[-1] != 3
        or spin_before_radps.shape != velocity_before_mps.shape
    ):
        raise ValueError("velocity/spin must have matching [N,3] shapes")
    descending = velocity_before_mps[:, 2] < 0.0
    radius = float(ball_radius_m)
    mass = float(ball_mass_kg)
    inertia = float(ball_inertia_coeff) * mass * radius * radius
    u_t = torch.stack(
        (
            velocity_before_mps[:, 0] - radius * spin_before_radps[:, 1],
            velocity_before_mps[:, 1] + radius * spin_before_radps[:, 0],
        ),
        dim=-1,
    )
    desired_j_t = -mass * float(table_a_t + table_b_t) * u_t
    j_n = -mass * (1.0 + float(table_e_eff)) * velocity_before_mps[:, 2]
    cap = float(table_mu_safety) * j_n.abs()
    norm = torch.linalg.norm(desired_j_t, dim=-1)
    scale = torch.minimum(
        torch.ones_like(norm),
        cap / norm.clamp_min(torch.finfo(norm.dtype).tiny),
    )
    j_t = desired_j_t * scale[:, None]
    velocity_after = torch.stack(
        (
            velocity_before_mps[:, 0] + j_t[:, 0] / mass,
            velocity_before_mps[:, 1] + j_t[:, 1] / mass,
            velocity_before_mps[:, 2] + j_n / mass,
        ),
        dim=-1,
    )
    spin_after = torch.stack(
        (
            spin_before_radps[:, 0] + radius * j_t[:, 1] / inertia,
            spin_before_radps[:, 1] - radius * j_t[:, 0] / inertia,
            spin_before_radps[:, 2],
        ),
        dim=-1,
    )
    return velocity_after, spin_after, descending


@torch.no_grad()
def rollout_counter_rally_torch(
    position_after_paddle_env_m: torch.Tensor,
    velocity_after_paddle_mps: torch.Tensor,
    spin_after_paddle_radps: torch.Tensor,
    *,
    binding: CounterRallyTorchBinding,
    dt_s: float = 0.001,
    max_time_s: float = 2.0,
) -> CounterRallyTorchOutcome:
    """Fused-row Torch rollout with the CPU oracle's event ordering."""

    if not isinstance(binding, CounterRallyTorchBinding):
        raise TypeError("binding must be CounterRallyTorchBinding")
    table_near_x_env_m = binding.table_near_x_env_m
    table_length_m = binding.table_length_m
    table_half_width_m = binding.table_half_width_m
    table_surface_z_env_m = binding.table_surface_z_env_m
    net_height_m = binding.net_height_m
    table_edge_margin_m = binding.table_edge_margin_m
    ball_radius_m = binding.ball_radius_m
    ball_mass_kg = binding.ball_mass_kg
    ball_inertia_coeff = binding.ball_inertia_coeff
    gravity_mps2 = binding.gravity_mps2
    drag_k_d_per_m = binding.drag_k_d_per_m
    magnus_k_m = binding.magnus_k_m
    table_e_eff = binding.table_e_eff
    table_a_t = binding.table_a_t
    table_b_t = binding.table_b_t
    table_mu_safety = binding.table_mu_safety
    p = position_after_paddle_env_m.clone()
    v = velocity_after_paddle_mps.clone()
    omega = spin_after_paddle_radps.clone()
    if (
        p.ndim != 2
        or p.shape[-1] != 3
        or v.shape != p.shape
        or omega.shape != p.shape
    ):
        raise ValueError("post-paddle state tensors must all have shape [N,3]")
    if not p.dtype.is_floating_point or v.dtype != p.dtype or omega.dtype != p.dtype:
        raise ValueError("post-paddle state tensors must share a floating dtype")
    if p.device != v.device or p.device != omega.device:
        raise ValueError("post-paddle state tensors must share one device")
    if not bool(torch.isfinite(p).all() and torch.isfinite(v).all() and torch.isfinite(omega).all()):
        raise ValueError("post-paddle state tensors must be finite")
    dt = float(dt_s)
    horizon = float(max_time_s)
    if not math.isfinite(dt) or not math.isfinite(horizon) or dt <= 0.0 or horizon <= dt:
        raise ValueError("invalid rollout horizon")

    n = int(p.shape[0])
    device, dtype = p.device, p.dtype
    active = torch.ones(n, dtype=torch.bool, device=device)
    net_crossed = torch.zeros_like(active)
    net_clear = torch.zeros_like(active)
    landed = torch.zeros_like(active)
    landing_valid = torch.zeros_like(active)
    bounced = torch.zeros(n, dtype=torch.long, device=device)
    baseline_crossed = torch.zeros_like(active)
    nan = float("nan")
    landing_xy = torch.full((n, 2), nan, dtype=dtype, device=device)
    landing_time = torch.full((n,), nan, dtype=dtype, device=device)
    baseline_yz = torch.full((n, 2), nan, dtype=dtype, device=device)
    baseline_velocity = torch.full((n, 3), nan, dtype=dtype, device=device)
    baseline_time = torch.full((n,), nan, dtype=dtype, device=device)
    reason = torch.zeros(n, dtype=torch.long, device=device)
    horizontal_norm = torch.linalg.norm(v[:, :2], dim=-1)
    direction_valid = horizontal_norm > 1.0e-9
    initial_direction = torch.full((n, 2), nan, dtype=dtype, device=device)
    initial_direction[direction_valid] = (
        v[direction_valid, :2] / horizontal_norm[direction_valid, None]
    )

    threshold_z = float(table_surface_z_env_m + ball_radius_m)
    net_x = float(table_near_x_env_m + 0.5 * table_length_m)
    far_x = float(table_near_x_env_m + table_length_m)
    net_threshold = float(
        table_surface_z_env_m + net_height_m + ball_radius_m
    )
    table_lo = float(table_near_x_env_m + table_edge_margin_m)
    table_hi = float(far_x - table_edge_margin_m)
    half_width = float(table_half_width_m - table_edge_margin_m)
    time_s = 0.0
    for _ in range(int(math.ceil(horizon / dt))):
        if not bool(active.any()):
            break
        old_p = p.clone()
        old_v = v.clone()
        new_p, new_v = _rk4_step(
            p,
            v,
            omega,
            dt_s=dt,
            gravity_mps2=gravity_mps2,
            drag_k_d_per_m=drag_k_d_per_m,
            magnus_k_m=magnus_k_m,
        )
        time_s += dt

        cross_net = (
            active
            & ~net_crossed
            & (old_p[:, 0] < net_x)
            & (new_p[:, 0] >= net_x)
        )
        net_denominator = new_p[:, 0] - old_p[:, 0]
        net_alpha = (net_x - old_p[:, 0]) / net_denominator.clamp_min(
            torch.finfo(dtype).tiny
        )
        net_z = old_p[:, 2] + net_alpha * (new_p[:, 2] - old_p[:, 2])
        net_crossed = net_crossed | cross_net
        this_net_clear = cross_net & (net_z >= net_threshold)
        net_clear = net_clear | this_net_clear
        net_fail = cross_net & ~this_net_clear
        reason[net_fail] = _REASON["net_not_clear"]
        active[net_fail] = False

        cross_surface = (
            active
            & (old_p[:, 2] > threshold_z)
            & (new_p[:, 2] <= threshold_z)
            & (new_v[:, 2] < 0.0)
        )
        surface_denominator = old_p[:, 2] - new_p[:, 2]
        surface_alpha = (old_p[:, 2] - threshold_z) / surface_denominator.clamp_min(
            torch.finfo(dtype).tiny
        )
        hit_xy = old_p[:, :2] + surface_alpha[:, None] * (
            new_p[:, :2] - old_p[:, :2]
        )
        first = cross_surface & ~landed
        landing_xy[first] = hit_xy[first]
        landing_time[first] = time_s - dt + surface_alpha[first] * dt
        landed = landed | first
        inside = (
            (hit_xy[:, 0] >= table_lo)
            & (hit_xy[:, 0] <= table_hi)
            & (hit_xy[:, 1].abs() <= half_width)
        )
        landing_valid[first] = inside[first]
        outside = first & ~inside
        reason[outside] = _REASON["first_landing_outside_table"]
        active[outside] = False
        own_half = first & inside & (
            (hit_xy[:, 0] <= net_x) | ~net_crossed
        )
        reason[own_half] = _REASON["first_landing_own_half"]
        active[own_half] = False
        legal_first = first & inside & ~own_half

        hit_velocity = old_v + surface_alpha[:, None] * (new_v - old_v)
        bounced_velocity, bounced_spin, descending = fitted_table_impulse_torch(
            hit_velocity,
            omega,
            ball_radius_m=ball_radius_m,
            ball_mass_kg=ball_mass_kg,
            ball_inertia_coeff=ball_inertia_coeff,
            table_e_eff=table_e_eff,
            table_a_t=table_a_t,
            table_b_t=table_b_t,
            table_mu_safety=table_mu_safety,
        )
        non_descending = legal_first & ~descending
        reason[non_descending] = _REASON["table_contact_not_descending"]
        active[non_descending] = False
        apply_bounce = legal_first & descending
        p[apply_bounce, :2] = hit_xy[apply_bounce]
        p[apply_bounce, 2] = threshold_z + 1.0e-9
        v[apply_bounce] = bounced_velocity[apply_bounce]
        omega[apply_bounce] = bounced_spin[apply_bounce]
        bounced[apply_bounce] = 1

        second = cross_surface & landed & ~first
        reason[second] = _REASON["second_table_bounce_before_baseline"]
        active[second] = False
        advance = active & ~cross_surface
        p[advance] = new_p[advance]
        v[advance] = new_v[advance]

        cross_baseline = (
            active
            & landed
            & (bounced == 1)
            & (old_p[:, 0] < far_x)
            & (p[:, 0] >= far_x)
        )
        baseline_denominator = p[:, 0] - old_p[:, 0]
        baseline_alpha = torch.where(
            baseline_denominator.abs() <= 1.0e-12,
            torch.ones_like(baseline_denominator),
            (far_x - old_p[:, 0]) / baseline_denominator,
        )
        baseline_yz[cross_baseline] = (
            old_p[cross_baseline, 1:]
            + baseline_alpha[cross_baseline, None]
            * (p[cross_baseline, 1:] - old_p[cross_baseline, 1:])
        )
        baseline_velocity[cross_baseline] = (
            old_v[cross_baseline]
            + baseline_alpha[cross_baseline, None]
            * (v[cross_baseline] - old_v[cross_baseline])
        )
        baseline_time[cross_baseline] = (
            time_s - dt + baseline_alpha[cross_baseline] * dt
        )
        baseline_crossed = baseline_crossed | cross_baseline
        active[cross_baseline] = False

        backward = active & (p[:, 0] < table_near_x_env_m - 0.5)
        reason[backward] = _REASON["trajectory_left_venue"]
        active[backward] = False

    reason[active] = _REASON["rollout_horizon_exceeded"]
    return CounterRallyTorchOutcome(
        net_crossed=net_crossed,
        net_clear=net_clear,
        first_landing_valid=landing_valid,
        first_landing_env_xy_m=landing_xy,
        first_landing_time_s=landing_time,
        table_bounce_count=bounced,
        opponent_baseline_crossed=baseline_crossed,
        baseline_cross_env_yz_m=baseline_yz,
        baseline_velocity_mps=baseline_velocity,
        baseline_time_s=baseline_time,
        post_hit_direction_env_xy=initial_direction,
        rejection_reason_code=reason,
    )


def counter_rally_reward_raw_torch(
    *,
    binding: CounterRallyTorchBinding,
    landing_aim_env_xy_m: torch.Tensor,
    return_direction_env_xy: torch.Tensor,
    target_baseline_speed_mps: torch.Tensor,
    paddle_contact_valid: torch.Tensor,
    task_objective_profile_sha256: Sequence[str],
    outcome: CounterRallyTorchOutcome,
) -> torch.Tensor:
    """Return columns legal/landing/reverse/speed/total with staged gates."""

    batch_size = landing_aim_env_xy_m.shape[0]
    if (
        landing_aim_env_xy_m.ndim != 2
        or landing_aim_env_xy_m.shape[-1] != 2
        or return_direction_env_xy.shape != landing_aim_env_xy_m.shape
        or target_baseline_speed_mps.shape != (batch_size,)
    ):
        raise ValueError(
            "landing aim/return direction/target speed must have "
            "matching [N,2]/[N,2]/[N] shapes"
        )
    gates = counter_rally_outcome_gates_torch(
        outcome=outcome,
        binding=binding,
        landing_aim_env_xy_m=landing_aim_env_xy_m,
        return_direction_env_xy=return_direction_env_xy,
        target_baseline_speed_mps=target_baseline_speed_mps,
        paddle_contact_valid=paddle_contact_valid,
        task_objective_profile_sha256=task_objective_profile_sha256,
    )
    legal = gates.legal_first_landing
    baseline_valid = gates.baseline_valid
    landing_error = torch.linalg.norm(
        outcome.first_landing_env_xy_m - landing_aim_env_xy_m,
        dim=-1,
    )
    landing = torch.exp(
        -torch.square(
            landing_error / binding.landing_tolerance_m
        )
    )
    landing = torch.where(legal, landing, torch.zeros_like(landing))
    dot = torch.sum(
        outcome.post_hit_direction_env_xy * return_direction_env_xy,
        dim=-1,
    ).clamp(-1.0, 1.0)
    cross = (
        outcome.post_hit_direction_env_xy[:, 0]
        * return_direction_env_xy[:, 1]
        - outcome.post_hit_direction_env_xy[:, 1]
        * return_direction_env_xy[:, 0]
    )
    angle_deg = torch.rad2deg(torch.atan2(cross.abs(), dot))
    reverse = torch.exp(
        -torch.square(
            angle_deg / binding.reverse_direction_tolerance_deg
        )
    )
    reverse = torch.where(
        baseline_valid
        & torch.isfinite(
            outcome.post_hit_direction_env_xy
        ).all(dim=-1),
        reverse,
        torch.zeros_like(reverse),
    )
    baseline_speed = torch.linalg.norm(
        outcome.baseline_velocity_mps, dim=-1
    )
    speed_tolerance = torch.maximum(
        torch.full_like(
            target_baseline_speed_mps,
            binding.target_speed_abs_tolerance_mps,
        ),
        binding.target_speed_rel_tolerance
        * target_baseline_speed_mps,
    )
    speed = torch.exp(
        -torch.square(
            (baseline_speed - target_baseline_speed_mps).abs()
            / speed_tolerance
        )
    )
    speed = torch.where(baseline_valid, speed, torch.zeros_like(speed))
    legal_float = legal.to(landing.dtype)
    total = (
        binding.reward_legal_fraction * legal_float
        + binding.reward_landing_fraction * landing
        + binding.reward_reverse_fraction * reverse
        + binding.reward_speed_fraction * speed
    ).clamp(0.0, 1.0)
    return torch.stack(
        (legal_float, landing, reverse, speed, total), dim=-1
    )
