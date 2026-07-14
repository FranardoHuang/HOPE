"""Fail-closed source core for a bounded lateral-balance perturbation ablation.

The first experiment cell deliberately perturbs only ``recovery/hold`` states.  It does not
change root velocity and it does not implement the later ``anytime`` axis.  The physical command
is a WORLD-Y force applied at ``torso_link``'s centre of mass.  Its impulse is normalized by the
*total articulation mass* so that link-mass randomization does not silently change the intended
whole-robot momentum disturbance::

    F_y = total_mass * sampled_delta_v_y / pulse_duration

This module is intentionally split in two:

* :class:`LateralPulseScheduler` is a pure, deterministic torch scheduler/kernel.
* :func:`dispatch_lateral_wrench_fail_closed` is an adapter seam.  No Isaac adapter is supplied
  here.  A future runtime adapter must transform the WORLD wrench to the frame expected by
  ``Articulation.set_external_force_and_torque``, stage without changing the live buffer, return
  an exact preflight receipt, then atomically overwrite the complete torso wrench buffer on every
  simulator step (including zeros after a pulse) at the COM.  Until that adapter and its ledger
  consumer are implemented and tested, this feature is not runtime-ready.

Keeping the seam explicit prevents two common but scientifically invalid shortcuts: using
``push_by_setting_velocity`` instead of a force, or leaving a non-zero external-force buffer alive
after the nominal pulse.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol, runtime_checkable

import torch


_COUNT_PREFIX = "lateral_perturbation_"
_U16_MASK = 0xFFFF
_U32_MASK = 0xFFFFFFFF
_U32_SCALE = float(1 << 32)
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_PHILOX_KEY1_NAMESPACE = 0xA3C59AC3
_RANDOM_GENERATOR_ID = "philox4x32-10-domain-separated-v1"
_RANDOM_DOMAINS = {
    "phase_offset": 0x50484153,
    "selection": 0x53454C45,
    "direction": 0x44495245,
    "unit_magnitude": 0x4D41474E,
}

# Immutable source-level backstop.  These limits deliberately contain the preregistered
# treatment (0.08 m/s over 0.10 s) and held-out stress paper (0.14 m/s over 0.10 s), while
# preventing a malformed config or randomized mass from turning this simulation-only probe into
# an arbitrarily large wrench.  They are not a real-robot safety certificate.
_HARD_MAX_ABS_NORMALIZED_IMPULSE_MPS = 0.15
_HARD_MAX_ABS_NORMALIZED_ACCEL_MPS2 = 2.0
_HARD_MIN_PULSE_DURATION_S = 0.02
_HARD_MAX_PULSE_DURATION_S = 0.20
_HARD_MAX_ABS_FORCE_N = 200.0
_DISPATCH_APPLICATION_CAPABILITY = object()


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _is_plain_number(value: object) -> bool:
    return type(value) in (int, float)


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def lateral_hard_safety_contract() -> dict[str, object]:
    """Return the immutable physical-command envelope enforced by this source core."""

    return {
        "schema_version": 1,
        "max_abs_normalized_impulse_mps": _HARD_MAX_ABS_NORMALIZED_IMPULSE_MPS,
        "max_abs_normalized_accel_mps2": _HARD_MAX_ABS_NORMALIZED_ACCEL_MPS2,
        "min_pulse_duration_s": _HARD_MIN_PULSE_DURATION_S,
        "max_pulse_duration_s": _HARD_MAX_PULSE_DURATION_S,
        "max_abs_world_force_y_N": _HARD_MAX_ABS_FORCE_N,
        "world_force_xz_N": [0.0, 0.0],
        "explicit_torque_Nm": [0.0, 0.0, 0.0],
    }


def lateral_hard_safety_identity_sha256() -> str:
    payload = json.dumps(
        lateral_hard_safety_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_all_async(condition: torch.Tensor, message: str) -> None:
    """Fail without a deliberate CUDA-to-host synchronization in the hot path."""

    predicate = torch.all(condition)
    if predicate.device.type == "cpu":
        if not bool(predicate):
            raise RuntimeError(message)
        return
    assert_fn = getattr(torch, "_assert_async", None)
    if callable(assert_fn):
        # torch 2.0 exposes the one-argument form; newer versions may accept a message, but using
        # the common signature keeps the runtime source portable without a CUDA synchronization.
        assert_fn(predicate)
    else:  # pragma: no cover - only for older torch than the supported runtime
        raise RuntimeError(
            "torch._assert_async is required for CUDA lateral-perturbation validation"
        )


def _assert_all_prewrite(condition: torch.Tensor, message: str) -> None:
    """Make a tensor safety predicate host-visible before any backend side effect.

    CUDA asynchronous assertions are useful inside the pure scheduler, but they cannot guard a
    Python writer: the writer could run before the device failure is observed.  Every mass,
    cast, wrench and adapter-preflight predicate that protects a physical buffer therefore uses
    this deliberately synchronous boundary.  It is correctness-first source code and remains a
    blocker for the pending no-host-sync throughput gate.
    """

    predicate = torch.all(condition)
    if not torch.equal(predicate, torch.ones_like(predicate, dtype=torch.bool)):
        raise RuntimeError(message)


def _mulhilo_u32_const(value: torch.Tensor, multiplier: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact unsigned-32 multiply split into high/low lanes using safe int64 arithmetic."""

    value_lo = torch.bitwise_and(value, _U16_MASK)
    value_hi = torch.bitwise_right_shift(value, 16)
    multiplier_lo = multiplier & _U16_MASK
    multiplier_hi = multiplier >> 16
    product_00 = value_lo * multiplier_lo
    product_01 = value_lo * multiplier_hi
    product_10 = value_hi * multiplier_lo
    product_11 = value_hi * multiplier_hi
    middle = (
        torch.bitwise_right_shift(product_00, 16)
        + torch.bitwise_and(product_01, _U16_MASK)
        + torch.bitwise_and(product_10, _U16_MASK)
    )
    low = torch.bitwise_or(
        torch.bitwise_and(product_00, _U16_MASK),
        torch.bitwise_left_shift(torch.bitwise_and(middle, _U16_MASK), 16),
    )
    high = (
        product_11
        + torch.bitwise_right_shift(product_01, 16)
        + torch.bitwise_right_shift(product_10, 16)
        + torch.bitwise_right_shift(middle, 16)
    )
    return torch.bitwise_and(high, _U32_MASK), torch.bitwise_and(low, _U32_MASK)


def _philox4x32_10(
    counter: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    key: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random123 Philox4x32-10, represented as non-negative int64 uint32 lanes."""

    c0, c1, c2, c3 = (torch.bitwise_and(lane, _U32_MASK) for lane in counter)
    k0 = key[0] & _U32_MASK
    k1 = key[1] & _U32_MASK
    for round_index in range(10):
        hi0, lo0 = _mulhilo_u32_const(c0, _PHILOX_M0)
        hi1, lo1 = _mulhilo_u32_const(c2, _PHILOX_M1)
        c0, c1, c2, c3 = (
            torch.bitwise_and(torch.bitwise_xor(torch.bitwise_xor(hi1, c1), k0), _U32_MASK),
            lo1,
            torch.bitwise_and(torch.bitwise_xor(torch.bitwise_xor(hi0, c3), k1), _U32_MASK),
            lo0,
        )
        if round_index != 9:
            k0 = (k0 + _PHILOX_W0) & _U32_MASK
            k1 = (k1 + _PHILOX_W1) & _U32_MASK
    return c0, c1, c2, c3


def _counter_uniform(
    *,
    seed: int,
    env_ids: torch.Tensor,
    episode_indices: torch.Tensor,
    opportunity_indices: torch.Tensor,
    domain: str,
) -> torch.Tensor:
    """Domain-separated counter-based U(0,1) values independent of call order."""

    if domain not in _RANDOM_DOMAINS:
        raise ValueError(f"unknown lateral perturbation random domain: {domain!r}")
    domain_lane = torch.full_like(env_ids, _RANDOM_DOMAINS[domain])
    output = _philox4x32_10(
        (
            env_ids,
            episode_indices,
            opportunity_indices,
            domain_lane,
        ),
        (seed, seed ^ _PHILOX_KEY1_NAMESPACE),
    )[0]
    return (output.to(dtype=torch.float64) + 0.5) / _U32_SCALE


def random_schedule_contract(cfg: "LateralPerturbationConfig") -> dict[str, object]:
    """Canonical CRN contract shared by zero-control and non-zero treatment."""

    return {
        "schema_version": 1,
        "generator": _RANDOM_GENERATOR_ID,
        "domains_u32_hex": {
            name: f"0x{value:08x}" for name, value in sorted(_RANDOM_DOMAINS.items())
        },
        "seed": cfg.seed,
        "policy_dt_s": float(cfg.policy_dt_s),
        "opportunity_interval_steps": cfg.opportunity_interval_steps,
        "pulse_duration_steps": cfg.pulse_duration_steps,
        "selection_probability": float(cfg.selection_probability),
        "eligibility_mode": cfg.eligibility_mode,
        "body_name": cfg.body_name,
        "force_frame": cfg.force_frame,
        "application_point": cfg.application_point,
    }


def random_schedule_identity_sha256(cfg: "LateralPerturbationConfig") -> str:
    payload = json.dumps(
        random_schedule_contract(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LateralPerturbationConfig:
    """Immutable contract for the recovery/hold-only first perturbation cell.

    ``normalized_impulse_*_mps`` is a whole-robot delta-velocity-equivalent budget, not a direct
    velocity write.  A treatment obtains its unit variate from the open interval ``(0, 1)`` and
    maps it between the configured magnitude bounds (a point mass when the bounds are equal), then
    samples direction with equal probability.  A matched control uses exactly ``0, 0`` while
    retaining the same opportunity/selection schedule.
    """

    policy_dt_s: float
    opportunity_interval_steps: int
    pulse_duration_steps: int
    selection_probability: float
    normalized_impulse_min_mps: float
    normalized_impulse_max_mps: float
    seed: int
    eligibility_mode: str = "recovery_hold"
    body_name: str = "torso_link"
    force_frame: str = "world"
    application_point: str = "center_of_mass"

    def __post_init__(self) -> None:
        if not _is_plain_number(self.policy_dt_s) or not math.isfinite(
            float(self.policy_dt_s)
        ) or float(self.policy_dt_s) <= 0.0:
            raise ValueError("policy_dt_s must be a finite number > 0")
        for name in ("opportunity_interval_steps", "pulse_duration_steps"):
            value = getattr(self, name)
            if not _is_plain_int(value) or value <= 0:
                raise ValueError(f"{name} must be a positive plain int")
        if self.opportunity_interval_steps < self.pulse_duration_steps:
            raise ValueError(
                "opportunity_interval_steps must be >= pulse_duration_steps so pulses cannot "
                "overlap by construction"
            )
        if not _is_plain_number(self.selection_probability) or not math.isfinite(
            float(self.selection_probability)
        ) or not 0.0 <= float(self.selection_probability) <= 1.0:
            raise ValueError("selection_probability must be finite and in [0, 1]")
        for name in (
            "normalized_impulse_min_mps",
            "normalized_impulse_max_mps",
        ):
            value = getattr(self, name)
            if (
                not _is_plain_number(value)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        if self.normalized_impulse_min_mps > self.normalized_impulse_max_mps:
            raise ValueError(
                "normalized_impulse_min_mps cannot exceed normalized_impulse_max_mps"
            )
        duration_s = float(self.policy_dt_s) * self.pulse_duration_steps
        if not math.isfinite(duration_s):
            raise ValueError("derived pulse_duration_s must be finite")
        if not _HARD_MIN_PULSE_DURATION_S <= duration_s <= _HARD_MAX_PULSE_DURATION_S:
            raise ValueError(
                "pulse_duration_s is outside the immutable hard safety envelope "
                f"[{_HARD_MIN_PULSE_DURATION_S}, {_HARD_MAX_PULSE_DURATION_S}]"
            )
        if float(self.normalized_impulse_max_mps) > _HARD_MAX_ABS_NORMALIZED_IMPULSE_MPS:
            raise ValueError(
                "normalized impulse exceeds the immutable hard safety envelope "
                f"{_HARD_MAX_ABS_NORMALIZED_IMPULSE_MPS} m/s"
            )
        max_normalized_accel = float(self.normalized_impulse_max_mps) / duration_s
        if not math.isfinite(max_normalized_accel):
            raise ValueError("derived normalized acceleration must be finite")
        if max_normalized_accel > _HARD_MAX_ABS_NORMALIZED_ACCEL_MPS2:
            raise ValueError(
                "derived normalized acceleration exceeds the immutable hard safety envelope "
                f"{_HARD_MAX_ABS_NORMALIZED_ACCEL_MPS2} m/s^2"
            )
        if not _is_plain_int(self.seed) or not 0 <= self.seed <= _U32_MASK:
            raise ValueError(f"seed must be a plain int in [0, {_U32_MASK}]")
        if self.eligibility_mode != "recovery_hold":
            raise ValueError(
                "the source-core v1 only permits eligibility_mode='recovery_hold'; "
                "an anytime perturbation is a separate future causal axis"
            )
        if self.body_name != "torso_link":
            raise ValueError("the first cell is frozen to body_name='torso_link'")
        if self.force_frame != "world":
            raise ValueError("the first cell is frozen to force_frame='world'")
        if self.application_point != "center_of_mass":
            raise ValueError("the first cell must apply at the torso centre of mass")

    @property
    def pulse_duration_s(self) -> float:
        return float(self.policy_dt_s) * self.pulse_duration_steps

    @property
    def is_zero_control(self) -> bool:
        return (
            float(self.normalized_impulse_min_mps) == 0.0
            and float(self.normalized_impulse_max_mps) == 0.0
        )

    @property
    def random_schedule_identity_sha256(self) -> str:
        return random_schedule_identity_sha256(self)

    @property
    def hard_safety_identity_sha256(self) -> str:
        return lateral_hard_safety_identity_sha256()


@dataclass(frozen=True)
class LateralPerturbationStep:
    """Per-environment command and sample ledger for one unique simulator step."""

    step_token: int
    random_schedule_identity_sha256: str
    hard_safety_identity_sha256: str
    potential_phase_offset_steps: torch.Tensor
    opportunity_indices: torch.Tensor
    potential_selection_u01: torch.Tensor
    potential_direction_u01: torch.Tensor
    potential_unit_magnitude_u01: torch.Tensor
    normalized_accel_y_mps2: torch.Tensor
    opportunity_mask: torch.Tensor
    eligible_opportunity_mask: torch.Tensor
    selected_start_mask: torch.Tensor
    nonzero_start_mask: torch.Tensor
    sampled_normalized_impulse_y_mps: torch.Tensor
    active_force_mask: torch.Tensor
    interrupted_for_strike_mask: torch.Tensor
    interrupted_for_window_mask: torch.Tensor
    interrupted_for_reset_mask: torch.Tensor
    strike_interrupted_sampled_impulse_y_mps: torch.Tensor
    strike_interrupted_commanded_impulse_y_mps: torch.Tensor
    strike_interrupted_applied_impulse_y_mps: torch.Tensor
    strike_abandoned_uncommanded_impulse_y_mps: torch.Tensor
    strike_abandoned_unapplied_impulse_y_mps: torch.Tensor
    window_interrupted_sampled_impulse_y_mps: torch.Tensor
    window_interrupted_commanded_impulse_y_mps: torch.Tensor
    window_interrupted_applied_impulse_y_mps: torch.Tensor
    window_abandoned_uncommanded_impulse_y_mps: torch.Tensor
    window_abandoned_unapplied_impulse_y_mps: torch.Tensor
    reset_interrupted_sampled_impulse_y_mps: torch.Tensor
    reset_interrupted_commanded_impulse_y_mps: torch.Tensor
    reset_interrupted_applied_impulse_y_mps: torch.Tensor
    reset_abandoned_uncommanded_impulse_y_mps: torch.Tensor
    reset_abandoned_unapplied_impulse_y_mps: torch.Tensor
    remaining_steps_after_step: torch.Tensor

    def clone(self) -> "LateralPerturbationStep":
        return LateralPerturbationStep(
            step_token=self.step_token,
            random_schedule_identity_sha256=self.random_schedule_identity_sha256,
            hard_safety_identity_sha256=self.hard_safety_identity_sha256,
            potential_phase_offset_steps=self.potential_phase_offset_steps.clone(),
            opportunity_indices=self.opportunity_indices.clone(),
            potential_selection_u01=self.potential_selection_u01.clone(),
            potential_direction_u01=self.potential_direction_u01.clone(),
            potential_unit_magnitude_u01=self.potential_unit_magnitude_u01.clone(),
            normalized_accel_y_mps2=self.normalized_accel_y_mps2.clone(),
            opportunity_mask=self.opportunity_mask.clone(),
            eligible_opportunity_mask=self.eligible_opportunity_mask.clone(),
            selected_start_mask=self.selected_start_mask.clone(),
            nonzero_start_mask=self.nonzero_start_mask.clone(),
            sampled_normalized_impulse_y_mps=self.sampled_normalized_impulse_y_mps.clone(),
            active_force_mask=self.active_force_mask.clone(),
            interrupted_for_strike_mask=self.interrupted_for_strike_mask.clone(),
            interrupted_for_window_mask=self.interrupted_for_window_mask.clone(),
            interrupted_for_reset_mask=self.interrupted_for_reset_mask.clone(),
            strike_interrupted_sampled_impulse_y_mps=(
                self.strike_interrupted_sampled_impulse_y_mps.clone()
            ),
            strike_interrupted_commanded_impulse_y_mps=(
                self.strike_interrupted_commanded_impulse_y_mps.clone()
            ),
            strike_interrupted_applied_impulse_y_mps=(
                self.strike_interrupted_applied_impulse_y_mps.clone()
            ),
            strike_abandoned_uncommanded_impulse_y_mps=(
                self.strike_abandoned_uncommanded_impulse_y_mps.clone()
            ),
            strike_abandoned_unapplied_impulse_y_mps=(
                self.strike_abandoned_unapplied_impulse_y_mps.clone()
            ),
            window_interrupted_sampled_impulse_y_mps=(
                self.window_interrupted_sampled_impulse_y_mps.clone()
            ),
            window_interrupted_commanded_impulse_y_mps=(
                self.window_interrupted_commanded_impulse_y_mps.clone()
            ),
            window_interrupted_applied_impulse_y_mps=(
                self.window_interrupted_applied_impulse_y_mps.clone()
            ),
            window_abandoned_uncommanded_impulse_y_mps=(
                self.window_abandoned_uncommanded_impulse_y_mps.clone()
            ),
            window_abandoned_unapplied_impulse_y_mps=(
                self.window_abandoned_unapplied_impulse_y_mps.clone()
            ),
            reset_interrupted_sampled_impulse_y_mps=(
                self.reset_interrupted_sampled_impulse_y_mps.clone()
            ),
            reset_interrupted_commanded_impulse_y_mps=(
                self.reset_interrupted_commanded_impulse_y_mps.clone()
            ),
            reset_interrupted_applied_impulse_y_mps=(
                self.reset_interrupted_applied_impulse_y_mps.clone()
            ),
            reset_abandoned_uncommanded_impulse_y_mps=(
                self.reset_abandoned_uncommanded_impulse_y_mps.clone()
            ),
            reset_abandoned_unapplied_impulse_y_mps=(
                self.reset_abandoned_unapplied_impulse_y_mps.clone()
            ),
            remaining_steps_after_step=self.remaining_steps_after_step.clone(),
        )


@dataclass(frozen=True)
class LateralWrenchPreflightReceipt:
    """Side-effect-free preflight receipt returned by a reviewed simulator adapter.

    Receipt validation happens before the backend write.  ``preflight_token`` is opaque adapter
    state consumed by one reviewed, atomic, no-throw commit.  This receipt is not itself proof of
    simulator behaviour; a runtime consumer must persist the resulting application ledger.
    """

    step_token: int
    body_name: str
    input_force_frame: str
    application_point: str
    full_batch_overwrite: bool
    inactive_zero_overwrite: bool
    zero_torque: bool
    world_to_backend_transform_identity_sha256: str
    application_backend_identity_sha256: str
    actual_total_mass_kg: torch.Tensor
    commanded_force_w: torch.Tensor
    commanded_torque_w: torch.Tensor
    applied_force_mask: torch.Tensor
    preflight_token: object

    def __post_init__(self) -> None:
        if not _is_plain_int(self.step_token) or self.step_token < 0:
            raise ValueError("receipt step_token must be a non-negative plain int")
        for name in (
            "full_batch_overwrite",
            "inactive_zero_overwrite",
            "zero_torque",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"receipt {name} must be a bool")
        if not isinstance(self.applied_force_mask, torch.Tensor):
            raise ValueError("receipt applied_force_mask must be a torch.Tensor")
        if self.applied_force_mask.ndim != 1 or self.applied_force_mask.dtype != torch.bool:
            raise ValueError("receipt applied_force_mask must be a 1-D bool tensor")
        if not _is_sha256_hex(self.world_to_backend_transform_identity_sha256):
            raise ValueError(
                "receipt world_to_backend_transform_identity_sha256 must be lowercase SHA-256"
            )
        if not _is_sha256_hex(self.application_backend_identity_sha256):
            raise ValueError(
                "receipt application_backend_identity_sha256 must be lowercase SHA-256"
            )
        for name in ("actual_total_mass_kg", "commanded_force_w", "commanded_torque_w"):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise ValueError(f"receipt {name} must be a torch.Tensor")
        if self.actual_total_mass_kg.ndim != 1 or not torch.is_floating_point(
            self.actual_total_mass_kg
        ):
            raise ValueError("receipt actual_total_mass_kg must be a 1-D floating tensor")
        expected_wrench_shape = (self.actual_total_mass_kg.shape[0], 1, 3)
        if self.commanded_force_w.shape != expected_wrench_shape:
            raise ValueError("receipt commanded_force_w has the wrong shape")
        if self.commanded_torque_w.shape != expected_wrench_shape:
            raise ValueError("receipt commanded_torque_w has the wrong shape")
        for name in ("commanded_force_w", "commanded_torque_w"):
            value = getattr(self, name)
            if not torch.is_floating_point(value):
                raise ValueError(f"receipt {name} must use a floating dtype")
            if value.dtype != self.actual_total_mass_kg.dtype:
                raise ValueError(f"receipt {name} must use the actual-mass dtype")
            if value.device != self.actual_total_mass_kg.device:
                raise ValueError(f"receipt {name} must use the actual-mass device")
        if self.applied_force_mask.shape != self.actual_total_mass_kg.shape:
            raise ValueError("receipt applied_force_mask must match actual_total_mass_kg shape")
        if self.applied_force_mask.device != self.actual_total_mass_kg.device:
            raise ValueError("receipt applied_force_mask must use the actual-mass device")
        if self.preflight_token is None:
            raise ValueError("receipt preflight_token cannot be None")
        _assert_all_prewrite(
            torch.isfinite(self.actual_total_mass_kg) & self.actual_total_mass_kg.gt(0.0),
            "receipt actual_total_mass_kg must be finite and strictly positive",
        )
        _assert_all_prewrite(
            torch.isfinite(self.commanded_force_w) & torch.isfinite(self.commanded_torque_w),
            "receipt commanded wrench must be finite",
        )


@dataclass(frozen=True)
class LateralApplicationLedgerRow:
    """Validated scheduler-to-adapter application record for one simulator step."""

    step_token: int
    body_name: str
    hard_safety_identity_sha256: str
    world_to_backend_transform_identity_sha256: str
    application_backend_identity_sha256: str
    actual_total_mass_kg: torch.Tensor
    commanded_normalized_accel_y_mps2: torch.Tensor
    commanded_world_force_y_N: torch.Tensor
    commanded_world_impulse_y_Ns: torch.Tensor
    applied_force_mask: torch.Tensor
    selected_start_count: torch.Tensor
    applied_nonzero_start_count: torch.Tensor
    nonzero_force_env_count: torch.Tensor
    commanded_normalized_impulse_abs_mps: torch.Tensor

    def clone(self) -> "LateralApplicationLedgerRow":
        """Deep-clone all tensor fields so callers cannot mutate scheduler-owned state."""

        return LateralApplicationLedgerRow(
            step_token=self.step_token,
            body_name=self.body_name,
            hard_safety_identity_sha256=self.hard_safety_identity_sha256,
            world_to_backend_transform_identity_sha256=(
                self.world_to_backend_transform_identity_sha256
            ),
            application_backend_identity_sha256=(
                self.application_backend_identity_sha256
            ),
            actual_total_mass_kg=self.actual_total_mass_kg.clone(),
            commanded_normalized_accel_y_mps2=(
                self.commanded_normalized_accel_y_mps2.clone()
            ),
            commanded_world_force_y_N=self.commanded_world_force_y_N.clone(),
            commanded_world_impulse_y_Ns=self.commanded_world_impulse_y_Ns.clone(),
            applied_force_mask=self.applied_force_mask.clone(),
            selected_start_count=self.selected_start_count.clone(),
            applied_nonzero_start_count=self.applied_nonzero_start_count.clone(),
            nonzero_force_env_count=self.nonzero_force_env_count.clone(),
            commanded_normalized_impulse_abs_mps=(
                self.commanded_normalized_impulse_abs_mps.clone()
            ),
        )


@dataclass(frozen=True)
class _PreparedApplication:
    """Scheduler-private, one-use application transaction prepared before adapter commit."""

    nonce: object
    step_token: int
    total_mass_kg: torch.Tensor
    force_w: torch.Tensor
    torque_w: torch.Tensor
    active_force_mask: torch.Tensor
    applied_impulse_per_env: torch.Tensor
    applied_starts: torch.Tensor
    applied_force_env_count: torch.Tensor
    applied_step_impulse: torch.Tensor
    private_ledger: LateralApplicationLedgerRow
    public_ledger: LateralApplicationLedgerRow
    application_backend_token: object | None = None
    already_committed: bool = False


@runtime_checkable
class LateralWrenchAdapter(Protocol):
    """Seam a future Isaac adapter must implement; no implementation is provided here."""

    body_name: str
    input_force_frame: str
    application_point: str
    full_batch_overwrite: bool
    inactive_zero_overwrite: bool
    preflight_side_effect_free: bool
    commit_is_atomic_and_noexcept: bool
    discard_is_noexcept: bool
    world_to_backend_transform_identity_sha256: str
    application_backend_identity_sha256: str
    application_backend_token: object

    def preflight_world_wrench_at_body_com(
        self,
        *,
        step_token: int,
        total_mass_kg: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        preflight_token: object,
    ) -> LateralWrenchPreflightReceipt:
        """Validate/stage an exact command without changing the live backend wrench buffer."""

    def commit_preflighted_world_wrench_at_body_com(
        self, *, preflight_token: object
    ) -> None:
        """Atomically commit one staged full-buffer overwrite and never raise."""

    def discard_preflighted_world_wrench_at_body_com(
        self, *, preflight_token: object
    ) -> None:
        """Discard staged data without touching the backend buffer and never raise."""


def _counter_zeros(device: torch.device) -> dict[str, torch.Tensor]:
    count_names = (
        "opportunity_count",
        "eligible_opportunity_count",
        "selected_start_count",
        "selected_left_count",
        "selected_right_count",
        "nonzero_pulse_command_count",
        "skipped_strike_window_count",
        "skipped_ineligible_phase_count",
        "skipped_short_window_count",
        "skipped_active_pulse_count",
        "interrupted_for_strike_count",
        "interrupted_for_window_count",
        "interrupted_for_reset_count",
        "active_force_env_step_count",
        "wrench_write_step_count",
        "applied_pulse_count",
        "applied_force_env_step_count",
    )
    state = {
        _COUNT_PREFIX + name: torch.zeros((), dtype=torch.long, device=device)
        for name in count_names
    }
    for name in (
        "sampled_normalized_impulse_abs_sum_mps",
        "commanded_normalized_impulse_abs_sum_mps",
        "applied_normalized_impulse_abs_sum_mps",
        "reset_interrupted_sampled_impulse_abs_sum_mps",
        "reset_interrupted_commanded_impulse_abs_sum_mps",
        "reset_interrupted_applied_impulse_abs_sum_mps",
        "reset_abandoned_uncommanded_impulse_abs_sum_mps",
        "reset_abandoned_unapplied_impulse_abs_sum_mps",
        "strike_interrupted_sampled_impulse_abs_sum_mps",
        "strike_interrupted_commanded_impulse_abs_sum_mps",
        "strike_interrupted_applied_impulse_abs_sum_mps",
        "strike_abandoned_uncommanded_impulse_abs_sum_mps",
        "strike_abandoned_unapplied_impulse_abs_sum_mps",
        "window_interrupted_sampled_impulse_abs_sum_mps",
        "window_interrupted_commanded_impulse_abs_sum_mps",
        "window_interrupted_applied_impulse_abs_sum_mps",
        "window_abandoned_uncommanded_impulse_abs_sum_mps",
        "window_abandoned_unapplied_impulse_abs_sum_mps",
    ):
        state[_COUNT_PREFIX + name] = torch.zeros(
            (), dtype=torch.float64, device=device
        )
    return state


class LateralPulseScheduler:
    """Deterministic pulse scheduler with activation and application accounting.

    The caller supplies the true recovery/hold mask, strike window, and number of safe policy
    steps remaining in that window.  A pulse only starts when the complete configured duration
    fits.  If the window unexpectedly closes or a strike starts while a pulse is active, the next
    command is zero and an interruption counter is charged.

    ``require_application_ack=True`` is mandatory for a future runtime hook.  It prevents the
    scheduler from advancing while the preceding wrench command has not received a validated
    committed full-buffer-overwrite ledger.  Pure source tests may use plan-only mode (the default).
    """

    def __init__(
        self,
        num_envs: int,
        cfg: LateralPerturbationConfig,
        *,
        device: str | torch.device = "cpu",
        require_application_ack: bool = False,
    ) -> None:
        if not _is_plain_int(num_envs) or num_envs <= 0:
            raise ValueError("num_envs must be a positive plain int")
        if num_envs - 1 > _U32_MASK:
            raise ValueError("num_envs exceed the Philox uint32 environment counter lane")
        if type(require_application_ack) is not bool:
            raise ValueError("require_application_ack must be a bool")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.require_application_ack = require_application_ack
        self.random_schedule_identity_sha256 = cfg.random_schedule_identity_sha256
        self.hard_safety_identity_sha256 = cfg.hard_safety_identity_sha256
        self._env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        self._remaining_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._active_impulse_y = torch.zeros(
            num_envs, dtype=torch.float64, device=self.device
        )
        self._active_commanded_impulse_y = torch.zeros_like(self._active_impulse_y)
        self._active_applied_impulse_y = torch.zeros_like(self._active_impulse_y)
        self._last_episode_indices = torch.full(
            (num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._last_episode_steps = torch.full(
            (num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._counters = _counter_zeros(self.device)
        self._last_step_token: int | None = None
        self._last_inputs: tuple[torch.Tensor, ...] | None = None
        self._last_result: LateralPerturbationStep | None = None
        self._last_application_ledger: LateralApplicationLedgerRow | None = None
        self._pending_application: _PreparedApplication | None = None
        self._application_dirty_unknown = False
        self._last_application_backend_token: object | None = None

    def _check_vector(
        self, name: str, value: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (self.num_envs,):
            raise ValueError(
                f"{name} must have shape ({self.num_envs},), got {tuple(value.shape)}"
            )
        if value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
        if value.device != self.device:
            raise ValueError(
                f"{name} must be on {self.device}, got {value.device}"
            )
        return value

    def _same_inputs(self, current: tuple[torch.Tensor, ...]) -> bool:
        return self._last_inputs is not None and all(
            torch.equal(old, new) for old, new in zip(self._last_inputs, current)
        )

    def step(
        self,
        *,
        step_token: int,
        episode_indices: torch.Tensor,
        episode_steps: torch.Tensor,
        recovery_hold_eligible: torch.Tensor,
        strike_window: torch.Tensor,
        safe_window_remaining_steps: torch.Tensor,
    ) -> LateralPerturbationStep:
        """Advance one unique policy/simulator step and return the per-env sample ledger."""

        if not _is_plain_int(step_token) or step_token < 0:
            raise ValueError("step_token must be a non-negative plain int")
        episode_indices = self._check_vector(
            "episode_indices", episode_indices, dtype=torch.long
        )
        episode_steps = self._check_vector(
            "episode_steps", episode_steps, dtype=torch.long
        )
        recovery_hold_eligible = self._check_vector(
            "recovery_hold_eligible", recovery_hold_eligible, dtype=torch.bool
        )
        strike_window = self._check_vector(
            "strike_window", strike_window, dtype=torch.bool
        )
        safe_window_remaining_steps = self._check_vector(
            "safe_window_remaining_steps",
            safe_window_remaining_steps,
            dtype=torch.long,
        )
        _assert_all_async(episode_indices >= 0, "episode_indices cannot be negative")
        _assert_all_async(episode_steps >= 0, "episode_steps cannot be negative")
        _assert_all_async(
            safe_window_remaining_steps >= 0,
            "safe_window_remaining_steps cannot be negative",
        )
        _assert_all_async(
            episode_indices <= _U32_MASK,
            "episode_indices exceed the Philox uint32 counter lane",
        )

        current_inputs = (
            episode_indices,
            episode_steps,
            recovery_hold_eligible,
            strike_window,
            safe_window_remaining_steps,
        )
        if self._application_dirty_unknown:
            raise RuntimeError(
                "lateral adapter backend is DIRTY/UNKNOWN after an atomic-commit contract "
                "violation; terminate the run or use an independently reviewed zero-clear/readback"
            )
        if self._last_step_token == step_token:
            if not self._same_inputs(current_inputs):
                raise RuntimeError(
                    "same step_token was reused with different perturbation inputs"
                )
            assert self._last_result is not None
            return self._last_result.clone()
        if self._last_step_token is not None:
            if step_token != self._last_step_token + 1:
                raise RuntimeError(
                    "lateral perturbation step tokens must be consecutive; a missing hook could "
                    "leave a stale external force alive"
                )
            if self.require_application_ack and (
                self._last_application_ledger is None
                or self._last_application_ledger.step_token != self._last_step_token
            ):
                raise RuntimeError(
                    "cannot advance lateral perturbation without the previous full-wrench "
                    "application receipt"
                )

        seen_episode = self._last_episode_indices.ge(0)
        changed_episode = episode_indices.ne(self._last_episode_indices)
        actual_reset = seen_episode & changed_episode
        invalid_same_episode_step = (
            seen_episode
            & ~changed_episode
            & episode_steps.ne(self._last_episode_steps + 1)
        )
        invalid_reset_step = seen_episode & changed_episode & episode_steps.ne(0)
        invalid_episode_order = (
            seen_episode
            & changed_episode
            & episode_indices.le(self._last_episode_indices)
        )
        _assert_all_async(
            ~invalid_same_episode_step,
            "episode_steps must advance by exactly one inside an episode",
        )
        _assert_all_async(
            ~invalid_reset_step,
            "a changed episode index must restart episode_steps at zero",
        )
        _assert_all_async(
            ~invalid_episode_order,
            "episode_indices must increase monotonically on reset",
        )

        interrupted_reset = actual_reset & self._remaining_steps.gt(0)
        reset_sampled_impulse = torch.where(
            interrupted_reset,
            self._active_impulse_y,
            torch.zeros_like(self._active_impulse_y),
        )
        reset_commanded_impulse = torch.where(
            interrupted_reset,
            self._active_commanded_impulse_y,
            torch.zeros_like(self._active_commanded_impulse_y),
        )
        reset_applied_impulse = torch.where(
            interrupted_reset,
            self._active_applied_impulse_y,
            torch.zeros_like(self._active_applied_impulse_y),
        )
        reset_abandoned_uncommanded = reset_sampled_impulse - reset_commanded_impulse
        reset_abandoned_unapplied = reset_commanded_impulse - reset_applied_impulse

        self._remaining_steps.masked_fill_(actual_reset, 0)
        self._active_impulse_y.masked_fill_(actual_reset, 0.0)
        self._active_commanded_impulse_y.masked_fill_(actual_reset, 0.0)
        self._active_applied_impulse_y.masked_fill_(actual_reset, 0.0)
        self._last_episode_indices.copy_(episode_indices)
        self._last_episode_steps.copy_(episode_steps)

        active_before = self._remaining_steps.gt(0)
        interrupted_strike = active_before & strike_window
        interrupted_window = active_before & (
            ~recovery_hold_eligible
            | self._remaining_steps.gt(safe_window_remaining_steps)
        ) & ~strike_window
        interrupted = interrupted_strike | interrupted_window

        def interrupted_values(mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
            sampled = torch.where(
                mask, self._active_impulse_y, torch.zeros_like(self._active_impulse_y)
            )
            commanded = torch.where(
                mask,
                self._active_commanded_impulse_y,
                torch.zeros_like(self._active_commanded_impulse_y),
            )
            applied = torch.where(
                mask,
                self._active_applied_impulse_y,
                torch.zeros_like(self._active_applied_impulse_y),
            )
            return (
                sampled,
                commanded,
                applied,
                sampled - commanded,
                commanded - applied,
            )

        (
            strike_sampled_impulse,
            strike_commanded_impulse,
            strike_applied_impulse,
            strike_abandoned_uncommanded,
            strike_abandoned_unapplied,
        ) = interrupted_values(interrupted_strike)
        (
            window_sampled_impulse,
            window_commanded_impulse,
            window_applied_impulse,
            window_abandoned_uncommanded,
            window_abandoned_unapplied,
        ) = interrupted_values(interrupted_window)
        self._remaining_steps.masked_fill_(interrupted, 0)
        self._active_impulse_y.masked_fill_(interrupted, 0.0)
        self._active_commanded_impulse_y.masked_fill_(interrupted, 0.0)
        self._active_applied_impulse_y.masked_fill_(interrupted, 0.0)

        zero_opp = torch.zeros_like(episode_steps)
        offsets_u = _counter_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=zero_opp,
            domain="phase_offset",
        )
        offsets = torch.floor(
            offsets_u * self.cfg.opportunity_interval_steps
        ).to(dtype=torch.long)
        opportunity = episode_steps.remainder(
            self.cfg.opportunity_interval_steps
        ).eq(offsets) & ~actual_reset
        opportunity_indices = torch.div(
            (episode_steps - offsets).clamp_min(0),
            self.cfg.opportunity_interval_steps,
            rounding_mode="floor",
        )
        _assert_all_async(
            opportunity_indices <= _U32_MASK,
            "opportunity_indices exceed the Philox uint32 counter lane",
        )

        active_now = self._remaining_steps.gt(0)
        enough_window = safe_window_remaining_steps.ge(
            self.cfg.pulse_duration_steps
        )
        eligible = (
            opportunity
            & recovery_hold_eligible
            & ~strike_window
            & enough_window
            & ~active_now
        )
        skipped_strike = opportunity & strike_window
        skipped_phase = opportunity & ~strike_window & ~recovery_hold_eligible
        skipped_short = (
            opportunity
            & recovery_hold_eligible
            & ~strike_window
            & ~enough_window
        )
        skipped_active = (
            opportunity
            & recovery_hold_eligible
            & ~strike_window
            & enough_window
            & active_now
        )

        select_u = _counter_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            domain="selection",
        )
        selected = eligible & select_u.lt(float(self.cfg.selection_probability))
        direction_u = _counter_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            domain="direction",
        )
        sign = torch.where(
            direction_u.lt(0.5),
            torch.full_like(direction_u, -1.0),
            torch.ones_like(direction_u),
        )
        magnitude_u = _counter_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            domain="unit_magnitude",
        )
        magnitude = float(self.cfg.normalized_impulse_min_mps) + magnitude_u * (
            float(self.cfg.normalized_impulse_max_mps)
            - float(self.cfg.normalized_impulse_min_mps)
        )
        sampled_impulse = torch.where(
            selected, sign * magnitude, torch.zeros_like(magnitude)
        )
        nonzero_start = selected & sampled_impulse.ne(0.0)

        # A zero control occupies the same virtual pulse duration as treatment.  This preserves
        # the stateless opportunity schedule instead of giving control extra selection chances.
        self._remaining_steps[selected] = self.cfg.pulse_duration_steps
        self._active_impulse_y[selected] = sampled_impulse[selected]
        self._active_commanded_impulse_y[selected] = 0.0
        self._active_applied_impulse_y[selected] = 0.0

        active_force = self._remaining_steps.gt(0) & self._active_impulse_y.ne(0.0)
        normalized_accel = torch.where(
            self._remaining_steps.gt(0),
            self._active_impulse_y / self.cfg.pulse_duration_s,
            torch.zeros_like(self._active_impulse_y),
        )
        self._active_commanded_impulse_y.add_(
            normalized_accel * float(self.cfg.policy_dt_s)
        )
        self._remaining_steps.sub_(self._remaining_steps.gt(0).to(torch.long))

        def add_count(name: str, mask: torch.Tensor) -> None:
            self._counters[_COUNT_PREFIX + name].add_(
                mask.detach().sum(dtype=torch.long)
            )

        add_count("opportunity_count", opportunity)
        add_count("eligible_opportunity_count", eligible)
        add_count("selected_start_count", selected)
        add_count("selected_left_count", selected & sign.lt(0.0))
        add_count("selected_right_count", selected & sign.gt(0.0))
        add_count("nonzero_pulse_command_count", nonzero_start)
        add_count("skipped_strike_window_count", skipped_strike)
        add_count("skipped_ineligible_phase_count", skipped_phase)
        add_count("skipped_short_window_count", skipped_short)
        add_count("skipped_active_pulse_count", skipped_active)
        add_count("interrupted_for_strike_count", interrupted_strike)
        add_count("interrupted_for_window_count", interrupted_window)
        add_count("interrupted_for_reset_count", interrupted_reset)
        add_count("active_force_env_step_count", active_force)
        self._counters[
            _COUNT_PREFIX + "sampled_normalized_impulse_abs_sum_mps"
        ].add_(sampled_impulse.detach().abs().sum(dtype=torch.float64))
        self._counters[
            _COUNT_PREFIX + "commanded_normalized_impulse_abs_sum_mps"
        ].add_(
            normalized_accel.detach().abs().sum(dtype=torch.float64)
            * float(self.cfg.policy_dt_s)
        )
        reset_ledger = {
            "reset_interrupted_sampled_impulse_abs_sum_mps": reset_sampled_impulse,
            "reset_interrupted_commanded_impulse_abs_sum_mps": reset_commanded_impulse,
            "reset_interrupted_applied_impulse_abs_sum_mps": reset_applied_impulse,
            "reset_abandoned_uncommanded_impulse_abs_sum_mps": (
                reset_abandoned_uncommanded
            ),
            "reset_abandoned_unapplied_impulse_abs_sum_mps": (
                reset_abandoned_unapplied
            ),
            "strike_interrupted_sampled_impulse_abs_sum_mps": (
                strike_sampled_impulse
            ),
            "strike_interrupted_commanded_impulse_abs_sum_mps": (
                strike_commanded_impulse
            ),
            "strike_interrupted_applied_impulse_abs_sum_mps": (
                strike_applied_impulse
            ),
            "strike_abandoned_uncommanded_impulse_abs_sum_mps": (
                strike_abandoned_uncommanded
            ),
            "strike_abandoned_unapplied_impulse_abs_sum_mps": (
                strike_abandoned_unapplied
            ),
            "window_interrupted_sampled_impulse_abs_sum_mps": (
                window_sampled_impulse
            ),
            "window_interrupted_commanded_impulse_abs_sum_mps": (
                window_commanded_impulse
            ),
            "window_interrupted_applied_impulse_abs_sum_mps": (
                window_applied_impulse
            ),
            "window_abandoned_uncommanded_impulse_abs_sum_mps": (
                window_abandoned_uncommanded
            ),
            "window_abandoned_unapplied_impulse_abs_sum_mps": (
                window_abandoned_unapplied
            ),
        }
        for counter_name, values in reset_ledger.items():
            self._counters[_COUNT_PREFIX + counter_name].add_(
                values.detach().abs().sum(dtype=torch.float64)
            )

        result = LateralPerturbationStep(
            step_token=step_token,
            random_schedule_identity_sha256=self.random_schedule_identity_sha256,
            hard_safety_identity_sha256=self.hard_safety_identity_sha256,
            potential_phase_offset_steps=offsets,
            opportunity_indices=opportunity_indices,
            potential_selection_u01=select_u,
            potential_direction_u01=direction_u,
            potential_unit_magnitude_u01=magnitude_u,
            normalized_accel_y_mps2=normalized_accel,
            opportunity_mask=opportunity,
            eligible_opportunity_mask=eligible,
            selected_start_mask=selected,
            nonzero_start_mask=nonzero_start,
            sampled_normalized_impulse_y_mps=sampled_impulse,
            active_force_mask=active_force,
            interrupted_for_strike_mask=interrupted_strike,
            interrupted_for_window_mask=interrupted_window,
            interrupted_for_reset_mask=interrupted_reset,
            strike_interrupted_sampled_impulse_y_mps=strike_sampled_impulse,
            strike_interrupted_commanded_impulse_y_mps=strike_commanded_impulse,
            strike_interrupted_applied_impulse_y_mps=strike_applied_impulse,
            strike_abandoned_uncommanded_impulse_y_mps=(
                strike_abandoned_uncommanded
            ),
            strike_abandoned_unapplied_impulse_y_mps=(
                strike_abandoned_unapplied
            ),
            window_interrupted_sampled_impulse_y_mps=window_sampled_impulse,
            window_interrupted_commanded_impulse_y_mps=window_commanded_impulse,
            window_interrupted_applied_impulse_y_mps=window_applied_impulse,
            window_abandoned_uncommanded_impulse_y_mps=(
                window_abandoned_uncommanded
            ),
            window_abandoned_unapplied_impulse_y_mps=(
                window_abandoned_unapplied
            ),
            reset_interrupted_sampled_impulse_y_mps=reset_sampled_impulse,
            reset_interrupted_commanded_impulse_y_mps=reset_commanded_impulse,
            reset_interrupted_applied_impulse_y_mps=reset_applied_impulse,
            reset_abandoned_uncommanded_impulse_y_mps=(
                reset_abandoned_uncommanded
            ),
            reset_abandoned_unapplied_impulse_y_mps=(
                reset_abandoned_unapplied
            ),
            remaining_steps_after_step=self._remaining_steps.clone(),
        )
        self._last_step_token = step_token
        self._last_inputs = tuple(value.clone() for value in current_inputs)
        self._last_result = result.clone()
        self._last_application_ledger = None
        self._pending_application = None
        self._last_application_backend_token = None
        return result

    def cached_application_ledger(
        self, step_token: int
    ) -> LateralApplicationLedgerRow | None:
        if (
            self._last_application_ledger is not None
            and self._last_application_ledger.step_token == step_token
        ):
            return self._last_application_ledger.clone()
        return None

    def _validate_application_result(self, result: LateralPerturbationStep) -> None:
        """Fail if a caller mutates or substitutes the scheduler's typed step ledger."""

        if self._last_result is None or result.step_token != self._last_step_token:
            raise RuntimeError("application receipt does not belong to the current scheduler step")
        last = self._last_result
        if result.random_schedule_identity_sha256 != last.random_schedule_identity_sha256:
            raise RuntimeError("application result has the wrong random schedule identity")
        if result.hard_safety_identity_sha256 != last.hard_safety_identity_sha256:
            raise RuntimeError("application result has the wrong hard safety identity")
        comparable = (
            "potential_phase_offset_steps",
            "opportunity_indices",
            "potential_selection_u01",
            "potential_direction_u01",
            "potential_unit_magnitude_u01",
            "normalized_accel_y_mps2",
            "opportunity_mask",
            "eligible_opportunity_mask",
            "selected_start_mask",
            "nonzero_start_mask",
            "sampled_normalized_impulse_y_mps",
            "active_force_mask",
            "interrupted_for_strike_mask",
            "interrupted_for_window_mask",
            "interrupted_for_reset_mask",
            "strike_interrupted_sampled_impulse_y_mps",
            "strike_interrupted_commanded_impulse_y_mps",
            "strike_interrupted_applied_impulse_y_mps",
            "strike_abandoned_uncommanded_impulse_y_mps",
            "strike_abandoned_unapplied_impulse_y_mps",
            "window_interrupted_sampled_impulse_y_mps",
            "window_interrupted_commanded_impulse_y_mps",
            "window_interrupted_applied_impulse_y_mps",
            "window_abandoned_uncommanded_impulse_y_mps",
            "window_abandoned_unapplied_impulse_y_mps",
            "reset_interrupted_sampled_impulse_y_mps",
            "reset_interrupted_commanded_impulse_y_mps",
            "reset_interrupted_applied_impulse_y_mps",
            "reset_abandoned_uncommanded_impulse_y_mps",
            "reset_abandoned_unapplied_impulse_y_mps",
            "remaining_steps_after_step",
        )
        equality_predicates = []
        for name in comparable:
            canonical_tensor = getattr(last, name)
            public_tensor = getattr(result, name)
            if (
                canonical_tensor.shape != public_tensor.shape
                or canonical_tensor.dtype != public_tensor.dtype
                or canonical_tensor.device != public_tensor.device
            ):
                raise RuntimeError(
                    f"application result has the wrong tensor contract for field {name}"
                )
            equality_predicates.append(torch.all(canonical_tensor == public_tensor))
        predicate_vector = torch.stack(equality_predicates)
        # A physical writer cannot be protected by an asynchronous assertion: Python could call
        # the adapter before a CUDA-side failure becomes host-visible.  This deliberate one-sync
        # barrier is correctness-first and must be included in the pending GPU-throughput gate.
        if not torch.equal(
            predicate_vector,
            torch.ones_like(predicate_vector, dtype=torch.bool),
        ):
            for name in comparable:
                if not torch.equal(getattr(last, name), getattr(result, name)):
                    raise RuntimeError(
                        "application result does not match scheduler ledger field "
                        f"{name}"
                    )
            raise RuntimeError("application result does not match scheduler ledger")

    def _validated_private_result_clone(
        self, result: LateralPerturbationStep
    ) -> LateralPerturbationStep:
        """Validate a public step, then return only scheduler-owned canonical data."""

        self._validate_application_result(result)
        assert self._last_result is not None
        return self._last_result.clone()

    @staticmethod
    def _require_dispatch_capability(capability: object) -> None:
        if capability is not _DISPATCH_APPLICATION_CAPABILITY:
            raise RuntimeError("application bookkeeping requires the internal dispatch capability")

    def _prepare_application_from_dispatch(
        self,
        *,
        capability: object,
        total_mass_kg: torch.Tensor,
        transform_identity_sha256: str,
        application_backend_identity_sha256: str | None = None,
        application_backend_token: object | None = None,
    ) -> _PreparedApplication:
        """Prepare all command/ledger state before a backend side effect.

        This is intentionally not a public receipt API.  A one-use module capability is required,
        and every command/mask/count is derived from ``self._last_result`` rather than caller-
        supplied expected tensors or an adapter receipt.
        """

        self._require_dispatch_capability(capability)
        if self._last_result is None or self._last_step_token is None:
            raise RuntimeError("cannot prepare an application before a scheduler step")
        if self._pending_application is not None:
            raise RuntimeError("a lateral application transaction is already pending")
        if self._application_dirty_unknown:
            raise RuntimeError(
                "cannot prepare while lateral adapter backend is DIRTY/UNKNOWN"
            )
        if not _is_sha256_hex(transform_identity_sha256):
            raise RuntimeError("dispatch transform identity must be a lowercase SHA-256")
        if not _is_sha256_hex(application_backend_identity_sha256):
            raise RuntimeError(
                "dispatch application backend identity must be a lowercase SHA-256"
            )
        if application_backend_token is None:
            raise RuntimeError("dispatch application backend token cannot be None")

        canonical = self._last_result
        expected_total_mass_kg = total_mass_kg.detach().clone()
        force_w, torque_w = lateral_world_wrench_from_total_mass(
            canonical.normalized_accel_y_mps2,
            expected_total_mass_kg,
        )
        _assert_all_prewrite(
            (force_w[:, :, 0] == 0.0)
            & (force_w[:, :, 2] == 0.0)
            & torch.all(torque_w == 0.0, dim=-1),
            "lateral wrench kernel emitted forbidden X/Z force or torque",
        )

        cached = self.cached_application_ledger(canonical.step_token)
        if cached is not None:
            if (
                cached.world_to_backend_transform_identity_sha256
                != transform_identity_sha256
            ):
                raise RuntimeError("same-step dispatch reused a different transform identity")
            if (
                cached.application_backend_identity_sha256
                != application_backend_identity_sha256
                or self._last_application_backend_token is not application_backend_token
            ):
                raise RuntimeError(
                    "same-step dispatch reused a different live application backend"
                )
            cached_tensors = {
                "actual_total_mass_kg": expected_total_mass_kg,
                "commanded_normalized_accel_y_mps2": (
                    canonical.normalized_accel_y_mps2.to(
                        dtype=expected_total_mass_kg.dtype
                    )
                ),
                "commanded_world_force_y_N": force_w[:, 0, 1],
                "commanded_world_impulse_y_Ns": (
                    force_w[:, 0, 1] * float(self.cfg.policy_dt_s)
                ),
            }
            for name, expected_tensor in cached_tensors.items():
                actual_tensor = getattr(cached, name)
                if (
                    actual_tensor.shape != expected_tensor.shape
                    or actual_tensor.dtype != expected_tensor.dtype
                    or actual_tensor.device != expected_tensor.device
                ):
                    raise RuntimeError(f"same-step dispatch changed {name} tensor contract")
                _assert_all_prewrite(
                    actual_tensor == expected_tensor,
                    f"same-step dispatch changed {name}",
                )
            zeros = torch.zeros_like(canonical.normalized_accel_y_mps2)
            zero_count = torch.zeros((), dtype=torch.long, device=self.device)
            zero_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            return _PreparedApplication(
                nonce=object(),
                step_token=canonical.step_token,
                total_mass_kg=expected_total_mass_kg,
                force_w=force_w,
                torque_w=torque_w,
                active_force_mask=canonical.active_force_mask.clone(),
                applied_impulse_per_env=zeros,
                applied_starts=zero_count,
                applied_force_env_count=zero_count.clone(),
                applied_step_impulse=zero_sum,
                private_ledger=cached.clone(),
                public_ledger=cached.clone(),
                application_backend_token=application_backend_token,
                already_committed=True,
            )

        active_force_mask = canonical.active_force_mask.clone()
        applied_starts = (
            canonical.nonzero_start_mask & active_force_mask
        ).sum(dtype=torch.long)
        applied_force_env_count = active_force_mask.sum(dtype=torch.long)
        applied_impulse_per_env = torch.where(
            active_force_mask,
            canonical.normalized_accel_y_mps2 * float(self.cfg.policy_dt_s),
            torch.zeros_like(canonical.normalized_accel_y_mps2),
        )
        applied_step_impulse = applied_impulse_per_env.abs().sum(dtype=torch.float64)
        ledger = LateralApplicationLedgerRow(
            step_token=canonical.step_token,
            body_name=self.cfg.body_name,
            hard_safety_identity_sha256=self.hard_safety_identity_sha256,
            world_to_backend_transform_identity_sha256=transform_identity_sha256,
            application_backend_identity_sha256=(
                application_backend_identity_sha256
            ),
            actual_total_mass_kg=expected_total_mass_kg.clone(),
            commanded_normalized_accel_y_mps2=(
                canonical.normalized_accel_y_mps2.to(
                    dtype=expected_total_mass_kg.dtype
                ).detach().clone()
            ),
            commanded_world_force_y_N=force_w[:, 0, 1].detach().clone(),
            commanded_world_impulse_y_Ns=(
                force_w[:, 0, 1].detach().clone() * float(self.cfg.policy_dt_s)
            ),
            applied_force_mask=active_force_mask.clone(),
            selected_start_count=canonical.selected_start_mask.sum(
                dtype=torch.long
            ).detach().clone(),
            applied_nonzero_start_count=applied_starts.detach().clone(),
            nonzero_force_env_count=applied_force_env_count.detach().clone(),
            commanded_normalized_impulse_abs_mps=applied_step_impulse.detach().clone(),
        )
        prepared = _PreparedApplication(
            nonce=object(),
            step_token=canonical.step_token,
            total_mass_kg=expected_total_mass_kg,
            force_w=force_w.detach().clone(),
            torque_w=torque_w.detach().clone(),
            active_force_mask=active_force_mask,
            applied_impulse_per_env=applied_impulse_per_env.detach().clone(),
            applied_starts=applied_starts.detach().clone(),
            applied_force_env_count=applied_force_env_count.detach().clone(),
            applied_step_impulse=applied_step_impulse.detach().clone(),
            private_ledger=ledger.clone(),
            public_ledger=ledger.clone(),
            application_backend_token=application_backend_token,
        )
        self._pending_application = prepared
        return prepared

    def _abort_application_from_dispatch(
        self, *, capability: object, prepared: _PreparedApplication
    ) -> None:
        """Discard a pre-write transaction after rejected adapter preflight."""

        self._require_dispatch_capability(capability)
        if self._pending_application is not prepared:
            raise RuntimeError("cannot abort a foreign lateral application transaction")
        self._pending_application = None

    def _commit_application_from_dispatch(
        self, *, capability: object, prepared: _PreparedApplication
    ) -> LateralApplicationLedgerRow:
        """Commit preallocated bookkeeping after the adapter's atomic no-throw write.

        All checks and allocations occur in ``_prepare_application_from_dispatch``.  Once the
        backend commit returns, this method performs only scheduler-owned deterministic updates.
        """

        self._require_dispatch_capability(capability)
        if prepared.already_committed:
            return prepared.public_ledger
        if self._pending_application is not prepared:
            raise RuntimeError("cannot commit a foreign lateral application transaction")
        self._active_applied_impulse_y.add_(prepared.applied_impulse_per_env)
        self._counters[_COUNT_PREFIX + "wrench_write_step_count"].add_(1)
        self._counters[_COUNT_PREFIX + "applied_pulse_count"].add_(
            prepared.applied_starts
        )
        self._counters[_COUNT_PREFIX + "applied_force_env_step_count"].add_(
            prepared.applied_force_env_count
        )
        self._counters[
            _COUNT_PREFIX + "applied_normalized_impulse_abs_sum_mps"
        ].add_(prepared.applied_step_impulse)
        self._last_application_ledger = prepared.private_ledger
        self._last_application_backend_token = prepared.application_backend_token
        self._pending_application = None
        return prepared.public_ledger

    def _mark_application_dirty_from_dispatch(
        self, *, capability: object, prepared: _PreparedApplication
    ) -> None:
        """Permanently block ordinary retry after an impossible no-throw commit violation."""

        self._require_dispatch_capability(capability)
        if self._pending_application is not prepared:
            raise RuntimeError("cannot dirty-mark a foreign lateral application transaction")
        self._pending_application = None
        self._application_dirty_unknown = True

    def consume_counters(self) -> dict[str, torch.Tensor]:
        """Snapshot/reset counters while retaining step/application idempotence tokens."""

        snapshot = {
            name: value.detach().clone() for name, value in self._counters.items()
        }
        for value in self._counters.values():
            value.zero_()
        return snapshot


def lateral_world_wrench_from_total_mass(
    normalized_accel_y_mps2: torch.Tensor,
    total_mass_kg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized lateral acceleration to a WORLD wrench at the torso COM.

    The returned tensors have Isaac's batch/body/vector layout ``(N, 1, 3)``.  Force X/Z and all
    torque components are allocated as exact zeros.  ``total_mass_kg`` must reflect the complete
    articulation after mass randomization; using torso-link mass would violate the experiment's
    normalization contract.
    """

    if not isinstance(normalized_accel_y_mps2, torch.Tensor):
        raise TypeError("normalized_accel_y_mps2 must be a torch.Tensor")
    if not isinstance(total_mass_kg, torch.Tensor):
        raise TypeError("total_mass_kg must be a torch.Tensor")
    if normalized_accel_y_mps2.ndim != 1:
        raise ValueError("normalized_accel_y_mps2 must have shape (N,)")
    if total_mass_kg.shape != normalized_accel_y_mps2.shape:
        raise ValueError("total_mass_kg must have the same shape as normalized acceleration")
    if total_mass_kg.device != normalized_accel_y_mps2.device:
        raise ValueError("mass and acceleration must be on the same device")
    if not torch.is_floating_point(normalized_accel_y_mps2):
        raise TypeError("normalized acceleration must use a floating dtype")
    if not torch.is_floating_point(total_mass_kg):
        raise TypeError("total_mass_kg must use a floating dtype")
    _assert_all_prewrite(
        torch.isfinite(normalized_accel_y_mps2),
        "normalized acceleration must be finite",
    )
    _assert_all_prewrite(
        torch.isfinite(total_mass_kg) & total_mass_kg.gt(0.0),
        "total_mass_kg must be finite and strictly positive",
    )

    # The simulator mass tensor owns the runtime wrench dtype (normally float32).  Sampling and
    # ledger arithmetic stay float64, but the actual command is rounded exactly once here instead
    # of asking an adapter to make an undocumented cast.
    dtype = total_mass_kg.dtype
    accel = normalized_accel_y_mps2.to(dtype=dtype)
    mass = total_mass_kg
    _assert_all_prewrite(
        torch.isfinite(accel),
        "normalized acceleration must remain finite after cast to the runtime mass dtype",
    )
    _assert_all_prewrite(
        normalized_accel_y_mps2.abs().le(_HARD_MAX_ABS_NORMALIZED_ACCEL_MPS2),
        "normalized acceleration exceeds the immutable hard safety envelope before cast",
    )
    _assert_all_prewrite(
        accel.abs().le(_HARD_MAX_ABS_NORMALIZED_ACCEL_MPS2),
        "normalized acceleration exceeds the immutable hard safety envelope",
    )
    force_w = torch.zeros(
        (accel.shape[0], 1, 3), dtype=dtype, device=accel.device
    )
    torque_w = torch.zeros_like(force_w)
    force_w[:, 0, 1] = mass * accel
    _assert_all_prewrite(
        torch.isfinite(force_w) & torch.isfinite(torque_w),
        "derived runtime wrench must remain finite after mass multiplication",
    )
    _assert_all_prewrite(
        force_w[:, 0, 1].abs().le(_HARD_MAX_ABS_FORCE_N),
        "derived WORLD-Y force exceeds the immutable hard safety envelope",
    )
    return force_w, torque_w


def dispatch_lateral_wrench_fail_closed(
    *,
    scheduler: LateralPulseScheduler,
    result: LateralPerturbationStep,
    total_mass_kg: torch.Tensor,
    adapter: LateralWrenchAdapter,
) -> LateralApplicationLedgerRow:
    """Preflight, atomically write/clear, then ledger one full-batch torso wrench.

    The adapter uses a two-phase contract: preflight may stage data but must not touch the live
    backend buffer; after every receipt predicate is synchronously visible, commit performs one
    atomic full-buffer overwrite and is required never to throw.  The commit is called even when
    every force is zero, which clears a completed or interrupted pulse.
    """

    # This must happen before even inspecting the adapter.  The public dataclass is frozen, but its
    # tensors are mutable; deriving a wrench from it before comparison would let an attacker cause
    # a physical write and only then trip the acknowledgement check.  All command derivation below
    # uses the scheduler-owned canonical clone returned here.
    canonical_result = scheduler._validated_private_result_clone(result)

    required_adapter_fields = {
        "body_name": scheduler.cfg.body_name,
        "input_force_frame": scheduler.cfg.force_frame,
        "application_point": scheduler.cfg.application_point,
        "full_batch_overwrite": True,
        "inactive_zero_overwrite": True,
        "preflight_side_effect_free": True,
        "commit_is_atomic_and_noexcept": True,
        "discard_is_noexcept": True,
    }
    for name, expected in required_adapter_fields.items():
        if not hasattr(adapter, name) or getattr(adapter, name) != expected:
            raise RuntimeError(
                f"lateral wrench adapter is not runtime-safe: {name} must be {expected!r}"
            )
    preflight = getattr(adapter, "preflight_world_wrench_at_body_com", None)
    if not callable(preflight):
        raise RuntimeError(
            "lateral wrench adapter lacks preflight_world_wrench_at_body_com"
        )
    commit = getattr(adapter, "commit_preflighted_world_wrench_at_body_com", None)
    if not callable(commit):
        raise RuntimeError(
            "lateral wrench adapter lacks commit_preflighted_world_wrench_at_body_com"
        )
    discard = getattr(adapter, "discard_preflighted_world_wrench_at_body_com", None)
    if not callable(discard):
        raise RuntimeError(
            "lateral wrench adapter lacks discard_preflighted_world_wrench_at_body_com"
        )
    transform_identity = getattr(
        adapter, "world_to_backend_transform_identity_sha256", None
    )
    if not _is_sha256_hex(transform_identity):
        raise RuntimeError(
            "lateral wrench adapter must expose a lowercase SHA-256 transform identity"
        )
    backend_identity = getattr(
        adapter, "application_backend_identity_sha256", None
    )
    if not _is_sha256_hex(backend_identity):
        raise RuntimeError(
            "lateral wrench adapter must expose a lowercase SHA-256 application backend identity"
        )
    backend_token = getattr(adapter, "application_backend_token", None)
    if backend_token is None:
        raise RuntimeError(
            "lateral wrench adapter must expose a stable live application backend token"
        )
    prepared = scheduler._prepare_application_from_dispatch(
        capability=_DISPATCH_APPLICATION_CAPABILITY,
        total_mass_kg=total_mass_kg,
        transform_identity_sha256=transform_identity,
        application_backend_identity_sha256=backend_identity,
        application_backend_token=backend_token,
    )
    if prepared.already_committed:
        return prepared.public_ledger

    expected_metadata = {
        "step_token": prepared.step_token,
        "body_name": scheduler.cfg.body_name,
        "input_force_frame": scheduler.cfg.force_frame,
        "application_point": scheduler.cfg.application_point,
        "full_batch_overwrite": True,
        "inactive_zero_overwrite": True,
        "zero_torque": True,
        "world_to_backend_transform_identity_sha256": transform_identity,
        "application_backend_identity_sha256": backend_identity,
    }
    try:
        receipt = preflight(
            step_token=prepared.step_token,
            total_mass_kg=prepared.total_mass_kg.clone(),
            force_w=prepared.force_w.clone(),
            torque_w=prepared.torque_w.clone(),
            preflight_token=prepared.nonce,
        )
        if not isinstance(receipt, LateralWrenchPreflightReceipt):
            raise RuntimeError("lateral wrench adapter returned no typed preflight receipt")
        if receipt.preflight_token is not prepared.nonce:
            raise RuntimeError(
                "lateral wrench preflight receipt returned a stale or foreign source token"
            )
        for name, expected in expected_metadata.items():
            if getattr(receipt, name) != expected:
                if name == "world_to_backend_transform_identity_sha256":
                    raise RuntimeError(
                        "lateral wrench preflight receipt transform identity does not match "
                        "the adapter contract"
                    )
                raise RuntimeError(
                    f"lateral wrench preflight receipt mismatch for {name}: "
                    f"expected {expected!r}, got {getattr(receipt, name)!r}"
                )
        tensor_receipt_fields = {
            "actual_total_mass_kg": prepared.total_mass_kg,
            "commanded_force_w": prepared.force_w,
            "commanded_torque_w": prepared.torque_w,
        }
        for name, expected_tensor in tensor_receipt_fields.items():
            actual_tensor = getattr(receipt, name)
            if (
                actual_tensor.shape != expected_tensor.shape
                or actual_tensor.dtype != expected_tensor.dtype
                or actual_tensor.device != expected_tensor.device
            ):
                raise RuntimeError(
                    f"adapter preflight receipt {name} has the wrong tensor contract"
                )
            _assert_all_prewrite(
                actual_tensor == expected_tensor,
                f"adapter preflight receipt {name} does not match the canonical command",
            )
        if receipt.applied_force_mask.shape != (scheduler.num_envs,):
            raise RuntimeError("adapter preflight applied_force_mask has the wrong shape")
        if receipt.applied_force_mask.dtype != torch.bool:
            raise RuntimeError("adapter preflight applied_force_mask has the wrong dtype")
        if receipt.applied_force_mask.device != scheduler.device:
            raise RuntimeError("adapter preflight applied_force_mask has the wrong device")
        _assert_all_prewrite(
            receipt.applied_force_mask == prepared.active_force_mask,
            "adapter preflight applied_force_mask does not match the canonical force mask",
        )
    except BaseException:
        try:
            discard(preflight_token=prepared.nonce)
        finally:
            scheduler._abort_application_from_dispatch(
                capability=_DISPATCH_APPLICATION_CAPABILITY,
                prepared=prepared,
            )
        raise

    # No validation, allocation, receipt processing or caller-controlled branch may occur between
    # this atomic no-throw commit and the preallocated scheduler bookkeeping below.
    try:
        commit_result = commit(preflight_token=prepared.nonce)
    except BaseException as exc:  # pragma: no cover - a reviewed adapter contract violation
        scheduler._mark_application_dirty_from_dispatch(
            capability=_DISPATCH_APPLICATION_CAPABILITY,
            prepared=prepared,
        )
        raise RuntimeError(
            "adapter violated atomic no-throw commit; backend state is DIRTY/UNKNOWN and the run "
            "must terminate or use an independently reviewed zero-clear/readback path"
        ) from exc
    if commit_result is not None:
        scheduler._mark_application_dirty_from_dispatch(
            capability=_DISPATCH_APPLICATION_CAPABILITY,
            prepared=prepared,
        )
        raise RuntimeError(
            "adapter violated the None-returning atomic commit contract; backend state is "
            "DIRTY/UNKNOWN"
        )
    return scheduler._commit_application_from_dispatch(
        capability=_DISPATCH_APPLICATION_CAPABILITY,
        prepared=prepared,
    )


__all__ = [
    "LateralApplicationLedgerRow",
    "LateralPerturbationConfig",
    "LateralPerturbationStep",
    "LateralPulseScheduler",
    "LateralWrenchAdapter",
    "LateralWrenchPreflightReceipt",
    "dispatch_lateral_wrench_fail_closed",
    "lateral_hard_safety_contract",
    "lateral_hard_safety_identity_sha256",
    "lateral_world_wrench_from_total_mass",
    "random_schedule_contract",
    "random_schedule_identity_sha256",
]
