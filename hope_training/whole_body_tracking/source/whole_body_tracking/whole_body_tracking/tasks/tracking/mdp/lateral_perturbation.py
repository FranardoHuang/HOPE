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
  ``Articulation.set_external_force_and_torque``, overwrite the complete torso wrench buffer on
  every simulator step (including zeros after a pulse), apply at the COM, and return an exact
  receipt.  Until that adapter and its ledger consumer are implemented and tested, this feature is
  not runtime-ready.

Keeping the seam explicit prevents two common but scientifically invalid shortcuts: using
``push_by_setting_velocity`` instead of a force, or leaving a non-zero external-force buffer alive
after the nominal pulse.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import torch


_LCG_MODULUS = 2_147_483_647
_COUNT_PREFIX = "lateral_perturbation_"


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def _is_plain_number(value: object) -> bool:
    return type(value) in (int, float)


@dataclass(frozen=True)
class LateralPerturbationConfig:
    """Immutable contract for the recovery/hold-only first perturbation cell.

    ``normalized_impulse_*_mps`` is a whole-robot delta-velocity-equivalent budget, not a direct
    velocity write.  A treatment samples the magnitude uniformly inside the closed interval and
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
        if not _is_plain_int(self.seed) or not 0 <= self.seed < _LCG_MODULUS:
            raise ValueError(f"seed must be a plain int in [0, {_LCG_MODULUS})")
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


@dataclass(frozen=True)
class LateralPerturbationStep:
    """Per-environment command and sample ledger for one unique simulator step."""

    step_token: int
    normalized_accel_y_mps2: torch.Tensor
    opportunity_mask: torch.Tensor
    eligible_opportunity_mask: torch.Tensor
    selected_start_mask: torch.Tensor
    nonzero_start_mask: torch.Tensor
    sampled_normalized_impulse_y_mps: torch.Tensor
    active_force_mask: torch.Tensor
    interrupted_for_strike_mask: torch.Tensor
    interrupted_for_window_mask: torch.Tensor
    remaining_steps_after_step: torch.Tensor

    def clone(self) -> "LateralPerturbationStep":
        return LateralPerturbationStep(
            step_token=self.step_token,
            normalized_accel_y_mps2=self.normalized_accel_y_mps2.clone(),
            opportunity_mask=self.opportunity_mask.clone(),
            eligible_opportunity_mask=self.eligible_opportunity_mask.clone(),
            selected_start_mask=self.selected_start_mask.clone(),
            nonzero_start_mask=self.nonzero_start_mask.clone(),
            sampled_normalized_impulse_y_mps=self.sampled_normalized_impulse_y_mps.clone(),
            active_force_mask=self.active_force_mask.clone(),
            interrupted_for_strike_mask=self.interrupted_for_strike_mask.clone(),
            interrupted_for_window_mask=self.interrupted_for_window_mask.clone(),
            remaining_steps_after_step=self.remaining_steps_after_step.clone(),
        )


@dataclass(frozen=True)
class LateralWrenchWriteReceipt:
    """Minimum acknowledgement returned by a reviewed simulator adapter.

    This is an application ledger row, not proof of simulator behaviour by itself.  A runtime
    consumer must persist it together with the per-step sample ledger above.
    """

    step_token: int
    body_name: str
    input_force_frame: str
    application_point: str
    full_batch_overwrite: bool
    inactive_zero_overwrite: bool
    zero_torque: bool
    nonzero_force_env_count: int

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
        if (
            not _is_plain_int(self.nonzero_force_env_count)
            or self.nonzero_force_env_count < 0
        ):
            raise ValueError(
                "receipt nonzero_force_env_count must be a non-negative plain int"
            )


@dataclass(frozen=True)
class LateralApplicationLedgerRow:
    """Validated scheduler-to-adapter application record for one simulator step."""

    step_token: int
    body_name: str
    selected_start_count: int
    applied_nonzero_start_count: int
    nonzero_force_env_count: int
    commanded_normalized_impulse_abs_mps: float


@runtime_checkable
class LateralWrenchAdapter(Protocol):
    """Seam a future Isaac adapter must implement; no implementation is provided here."""

    body_name: str
    input_force_frame: str
    application_point: str
    full_batch_overwrite: bool
    inactive_zero_overwrite: bool

    def overwrite_world_wrench_at_body_com(
        self,
        *,
        step_token: int,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
    ) -> LateralWrenchWriteReceipt:
        """Transform/write the full batch and acknowledge the exact application contract."""


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
    ):
        state[_COUNT_PREFIX + name] = torch.zeros(
            (), dtype=torch.float64, device=device
        )
    return state


def _stateless_uniform(
    *,
    seed: int,
    env_ids: torch.Tensor,
    episode_indices: torch.Tensor,
    opportunity_indices: torch.Tensor,
    stream: int,
) -> torch.Tensor:
    """Counter-based U(0,1) variates with stable CPU/CUDA integer arithmetic.

    This is intentionally stateless: control and treatment receive the same timing, selection,
    sign and unit-magnitude variates even after their trajectories diverge.  It is not a security
    RNG.  All products stay far below signed-int64 overflow after each modulus.
    """

    if not _is_plain_int(stream) or stream < 0:
        raise ValueError("stream must be a non-negative plain int")
    modulus = _LCG_MODULUS
    x = torch.remainder(seed + 1 + (env_ids + 1) * 48_271, modulus)
    x = torch.remainder(x + (episode_indices + 1) * 69_621, modulus)
    x = torch.remainder(x + (opportunity_indices + 1) * 40_699, modulus)
    x = torch.remainder(x + (stream + 1) * 104_729, modulus)
    x = torch.remainder(x * 48_271 + 12_820_163, modulus)
    x = torch.remainder(x * 40_699 + 1_234_567, modulus)
    return (x.to(dtype=torch.float64) + 0.5) / float(modulus)


class LateralPulseScheduler:
    """Deterministic pulse scheduler with activation and application accounting.

    The caller supplies the true recovery/hold mask, strike window, and number of safe policy
    steps remaining in that window.  A pulse only starts when the complete configured duration
    fits.  If the window unexpectedly closes or a strike starts while a pulse is active, the next
    command is zero and an interruption counter is charged.

    ``require_application_ack=True`` is mandatory for a future runtime hook.  It prevents the
    scheduler from advancing while the preceding wrench command has not received a validated
    full-buffer-overwrite receipt.  Pure source tests may use plan-only mode (the default).
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
        if type(require_application_ack) is not bool:
            raise ValueError("require_application_ack must be a bool")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.require_application_ack = require_application_ack
        self._env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        self._remaining_steps = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self._active_impulse_y = torch.zeros(
            num_envs, dtype=torch.float64, device=self.device
        )
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
        if bool(torch.any(episode_indices < 0)):
            raise ValueError("episode_indices cannot be negative")
        if bool(torch.any(episode_steps < 0)):
            raise ValueError("episode_steps cannot be negative")
        if bool(torch.any(safe_window_remaining_steps < 0)):
            raise ValueError("safe_window_remaining_steps cannot be negative")

        current_inputs = (
            episode_indices,
            episode_steps,
            recovery_hold_eligible,
            strike_window,
            safe_window_remaining_steps,
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
        if bool(torch.any(invalid_same_episode_step)):
            raise RuntimeError(
                "episode_steps must advance by exactly one inside an episode"
            )
        if bool(torch.any(invalid_reset_step)):
            raise RuntimeError("a changed episode index must restart episode_steps at zero")
        if bool(torch.any(invalid_episode_order)):
            raise RuntimeError("episode_indices must increase monotonically on reset")
        self._remaining_steps.masked_fill_(changed_episode, 0)
        self._active_impulse_y.masked_fill_(changed_episode, 0.0)
        self._last_episode_indices.copy_(episode_indices)
        self._last_episode_steps.copy_(episode_steps)

        active_before = self._remaining_steps.gt(0)
        interrupted_strike = active_before & strike_window
        interrupted_window = active_before & (
            ~recovery_hold_eligible
            | self._remaining_steps.gt(safe_window_remaining_steps)
        ) & ~strike_window
        interrupted = interrupted_strike | interrupted_window
        self._remaining_steps.masked_fill_(interrupted, 0)
        self._active_impulse_y.masked_fill_(interrupted, 0.0)

        zero_opp = torch.zeros_like(episode_steps)
        offsets_u = _stateless_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=zero_opp,
            stream=0,
        )
        offsets = torch.floor(
            offsets_u * self.cfg.opportunity_interval_steps
        ).to(dtype=torch.long)
        opportunity = episode_steps.remainder(
            self.cfg.opportunity_interval_steps
        ).eq(offsets)
        opportunity_indices = torch.div(
            (episode_steps - offsets).clamp_min(0),
            self.cfg.opportunity_interval_steps,
            rounding_mode="floor",
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

        select_u = _stateless_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            stream=1,
        )
        selected = eligible & select_u.lt(float(self.cfg.selection_probability))
        direction_u = _stateless_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            stream=2,
        )
        sign = torch.where(
            direction_u.lt(0.5),
            torch.full_like(direction_u, -1.0),
            torch.ones_like(direction_u),
        )
        magnitude_u = _stateless_uniform(
            seed=self.cfg.seed,
            env_ids=self._env_ids,
            episode_indices=episode_indices,
            opportunity_indices=opportunity_indices,
            stream=3,
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

        active_force = self._remaining_steps.gt(0) & self._active_impulse_y.ne(0.0)
        normalized_accel = torch.where(
            self._remaining_steps.gt(0),
            self._active_impulse_y / self.cfg.pulse_duration_s,
            torch.zeros_like(self._active_impulse_y),
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

        result = LateralPerturbationStep(
            step_token=step_token,
            normalized_accel_y_mps2=normalized_accel,
            opportunity_mask=opportunity,
            eligible_opportunity_mask=eligible,
            selected_start_mask=selected,
            nonzero_start_mask=nonzero_start,
            sampled_normalized_impulse_y_mps=sampled_impulse,
            active_force_mask=active_force,
            interrupted_for_strike_mask=interrupted_strike,
            interrupted_for_window_mask=interrupted_window,
            remaining_steps_after_step=self._remaining_steps.clone(),
        )
        self._last_step_token = step_token
        self._last_inputs = tuple(value.clone() for value in current_inputs)
        self._last_result = result.clone()
        self._last_application_ledger = None
        return result

    def cached_application_ledger(
        self, step_token: int
    ) -> LateralApplicationLedgerRow | None:
        if (
            self._last_application_ledger is not None
            and self._last_application_ledger.step_token == step_token
        ):
            return self._last_application_ledger
        return None

    def acknowledge_application(
        self,
        result: LateralPerturbationStep,
        receipt: LateralWrenchWriteReceipt,
    ) -> LateralApplicationLedgerRow:
        """Validate one full-buffer write receipt and charge application counters once."""

        if self._last_result is None or result.step_token != self._last_step_token:
            raise RuntimeError("application receipt does not belong to the current scheduler step")
        last = self._last_result
        comparable = (
            "normalized_accel_y_mps2",
            "opportunity_mask",
            "eligible_opportunity_mask",
            "selected_start_mask",
            "nonzero_start_mask",
            "sampled_normalized_impulse_y_mps",
            "active_force_mask",
            "interrupted_for_strike_mask",
            "interrupted_for_window_mask",
            "remaining_steps_after_step",
        )
        if any(
            not torch.equal(getattr(last, name), getattr(result, name))
            for name in comparable
        ):
            raise RuntimeError("application result does not match the current scheduler ledger")
        if receipt.step_token != result.step_token:
            raise RuntimeError("adapter receipt step_token does not match the command")
        cached = self.cached_application_ledger(result.step_token)
        if cached is not None:
            return cached
        expected_nonzero = int(result.active_force_mask.sum().item())
        expected = {
            "body_name": self.cfg.body_name,
            "input_force_frame": self.cfg.force_frame,
            "application_point": self.cfg.application_point,
            "full_batch_overwrite": True,
            "inactive_zero_overwrite": True,
            "zero_torque": True,
            "nonzero_force_env_count": expected_nonzero,
        }
        for name, value in expected.items():
            if getattr(receipt, name) != value:
                raise RuntimeError(
                    f"lateral wrench receipt mismatch for {name}: "
                    f"expected {value!r}, got {getattr(receipt, name)!r}"
                )

        applied_starts = int(result.nonzero_start_mask.sum().item())
        applied_step_impulse = float(
            result.normalized_accel_y_mps2.abs().sum().item()
            * float(self.cfg.policy_dt_s)
        )
        self._counters[_COUNT_PREFIX + "wrench_write_step_count"].add_(1)
        self._counters[_COUNT_PREFIX + "applied_pulse_count"].add_(applied_starts)
        self._counters[_COUNT_PREFIX + "applied_force_env_step_count"].add_(
            expected_nonzero
        )
        self._counters[
            _COUNT_PREFIX + "applied_normalized_impulse_abs_sum_mps"
        ].add_(applied_step_impulse)
        ledger = LateralApplicationLedgerRow(
            step_token=result.step_token,
            body_name=self.cfg.body_name,
            selected_start_count=int(result.selected_start_mask.sum().item()),
            applied_nonzero_start_count=applied_starts,
            nonzero_force_env_count=expected_nonzero,
            commanded_normalized_impulse_abs_mps=applied_step_impulse,
        )
        self._last_application_ledger = ledger
        return ledger

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
    if bool(torch.any(~torch.isfinite(normalized_accel_y_mps2))):
        raise ValueError("normalized acceleration must be finite")
    if bool(torch.any(~torch.isfinite(total_mass_kg))) or bool(
        torch.any(total_mass_kg <= 0.0)
    ):
        raise ValueError("total_mass_kg must be finite and strictly positive")

    # The simulator mass tensor owns the runtime wrench dtype (normally float32).  Sampling and
    # ledger arithmetic stay float64, but the actual command is rounded exactly once here instead
    # of asking an adapter to make an undocumented cast.
    dtype = total_mass_kg.dtype
    accel = normalized_accel_y_mps2.to(dtype=dtype)
    mass = total_mass_kg
    force_w = torch.zeros(
        (accel.shape[0], 1, 3), dtype=dtype, device=accel.device
    )
    torque_w = torch.zeros_like(force_w)
    force_w[:, 0, 1] = mass * accel
    return force_w, torque_w


def dispatch_lateral_wrench_fail_closed(
    *,
    scheduler: LateralPulseScheduler,
    result: LateralPerturbationStep,
    total_mass_kg: torch.Tensor,
    adapter: LateralWrenchAdapter,
) -> LateralApplicationLedgerRow:
    """Write/clear one full-batch torso wrench through a reviewed adapter seam.

    The adapter is called even when every force is zero.  That zero overwrite is the mechanism
    that prevents a completed pulse from becoming a persistent external force.
    """

    cached = scheduler.cached_application_ledger(result.step_token)
    if cached is not None:
        return cached
    required_adapter_fields = {
        "body_name": scheduler.cfg.body_name,
        "input_force_frame": scheduler.cfg.force_frame,
        "application_point": scheduler.cfg.application_point,
        "full_batch_overwrite": True,
        "inactive_zero_overwrite": True,
    }
    for name, expected in required_adapter_fields.items():
        if not hasattr(adapter, name) or getattr(adapter, name) != expected:
            raise RuntimeError(
                f"lateral wrench adapter is not runtime-safe: {name} must be {expected!r}"
            )
    writer = getattr(adapter, "overwrite_world_wrench_at_body_com", None)
    if not callable(writer):
        raise RuntimeError(
            "lateral wrench adapter lacks overwrite_world_wrench_at_body_com"
        )
    force_w, torque_w = lateral_world_wrench_from_total_mass(
        result.normalized_accel_y_mps2, total_mass_kg
    )
    if bool(torch.any(force_w[:, :, 0] != 0.0)) or bool(
        torch.any(force_w[:, :, 2] != 0.0)
    ) or bool(torch.any(torque_w != 0.0)):
        raise RuntimeError("lateral wrench kernel emitted forbidden X/Z force or torque")
    receipt = writer(
        step_token=result.step_token,
        force_w=force_w,
        torque_w=torque_w,
    )
    if not isinstance(receipt, LateralWrenchWriteReceipt):
        raise RuntimeError("lateral wrench adapter returned no typed write receipt")
    return scheduler.acknowledge_application(result, receipt)


__all__ = [
    "LateralApplicationLedgerRow",
    "LateralPerturbationConfig",
    "LateralPerturbationStep",
    "LateralPulseScheduler",
    "LateralWrenchAdapter",
    "LateralWrenchWriteReceipt",
    "dispatch_lateral_wrench_fail_closed",
    "lateral_world_wrench_from_total_mass",
]
