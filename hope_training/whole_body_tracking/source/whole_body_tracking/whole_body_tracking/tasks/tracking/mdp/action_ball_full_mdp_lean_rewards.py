"""Lean twenty-term RewardManager graph over one immutable ActionEpoch.

Post-physics producers publish once into their fixed ActionEpoch fact slices.
RewardManager then captures one immutable record at ordinal zero and decodes
the fourteen lifecycle values from that snapshot.  Six common imitation terms
then reuse the existing Motion reward functions and record into the same
milestone accumulator.  Reward terms never requery a lifecycle producer,
consume a receipt, compare source digests, or read producer-private state.

Ordinary no-contact, miss, low reward, or not-ready facts return finite learning
values (usually zero).  Typed producer faults are already present in the epoch
owner slot and suppress that row; the sole global drain remains responsible
for decoding them as infrastructure evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import types
from typing import Mapping

import torch

try:
    from . import action_ball_full_mdp_epoch as epoch_v1
except ImportError:  # Focused source-file tests do not import Isaac Lab.
    import action_ball_full_mdp_epoch as epoch_v1

try:
    from . import action_ball_full_mdp_lean_checkpoint_txn as carry_txn
except ImportError:
    import action_ball_full_mdp_lean_checkpoint_txn as carry_txn


DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

GRAPH_ATTR = "action_ball_full_mdp_lean_reward_graph"
ORDERED_CONSUMERS = epoch_v1.REWARD_CONSUMER_ORDER
LIFECYCLE_PAYMENT_COUNT = 14
if len(ORDERED_CONSUMERS) != LIFECYCLE_PAYMENT_COUNT:
    raise RuntimeError("lean Reward ABI is not exactly fourteen consumers")

R03_NAMES = tuple(name.split(":", 1)[1] for name in ORDERED_CONSUMERS[:10])
_R03_ERROR_COMPONENT_BY_ORDINAL = (0, 1, 2, 0, 1, 2, 0, 1, 2, 3)
LIFECYCLE_MANAGER_NAMES = tuple(
    name.split(":", 1)[1] for name in ORDERED_CONSUMERS
)
COMMON_DENSE_SPECS = (
    ("motion_global_anchor_pos", "motion_global_anchor_position_error_exp", 0.5, 0.3),
    ("motion_global_anchor_ori", "motion_global_anchor_orientation_error_exp", 0.5, 0.4),
    ("motion_body_pos", "motion_relative_body_position_error_exp", 1.0, 0.3),
    ("motion_body_ori", "motion_relative_body_orientation_error_exp", 1.0, 0.4),
    ("motion_body_lin_vel", "motion_global_body_linear_velocity_error_exp", 1.0, 1.0),
    ("motion_body_ang_vel", "motion_global_body_angular_velocity_error_exp", 1.0, 3.14),
)
COMMON_DENSE_NAMES = tuple(spec[0] for spec in COMMON_DENSE_SPECS)
BODY_ORIENTATION_COARSE_STD = 1.0
MANAGER_NAMES = LIFECYCLE_MANAGER_NAMES + COMMON_DENSE_NAMES
MANAGER_TERM_COUNT = len(MANAGER_NAMES)

R03_PRESENT = 1 << 0
R03_PHYSICALLY_VALID = 1 << 1
PHYSICAL_PRESENT = 1 << 0
PHYSICAL_SELECTED_CONTACT = 1 << 1
R06_PRESENT = 1 << 0
R06_POLICY_ELIGIBLE = 1 << 1
R06_SOURCE_VALID = 1 << 2
R07_PRESENT = 1 << 0
R07_NUMERICALLY_VALID = 1 << 1

# One-update diagnostic values only.  They are deliberately local, immutable,
# and unauthorized: this profile can exercise real RewardManager/PPO wiring,
# but cannot stand in for the launcher-pinned scientific numeric selection.
DIAGNOSTIC_N2_REWARD_PROFILE_KIND = (
    "action_ball_full_mdp_diagnostic_n2_reward_profile_v2"
)
DIAGNOSTIC_N2_WEIGHTS = {
    **{name: 1.0 for name in R03_NAMES},
    "physical_selected_contact": 1.0,
    "common_on_table_outcome": 20.0,
    "post_contact_placement_guidance": 1.0,
    # R07 fact is already weighted by its construction-owned profile.
    "common_recovery_reward_v1": 1.0,
    **{name: weight for name, _func, weight, _std in COMMON_DENSE_SPECS},
}
DIAGNOSTIC_N2_R03_SCALES = {
    "racket_position": 0.2,
    "racket_velocity": 1.0,
    "racket_normal": 0.5,
    "racket_position_coarse": 0.5,
    "racket_velocity_coarse": 2.0,
    "racket_normal_coarse": 1.0,
    "racket_position_precision": 0.1,
    "racket_velocity_precision": 0.5,
    "racket_normal_precision": 0.25,
    "paddle_center_proximity": 0.15,
}


@dataclass(frozen=True)
class DirectR03RewardFacts:
    """Compatibility decode of the immutable R03 epoch fact slice."""

    eligible: torch.Tensor
    validity: torch.Tensor
    producer_fault_bits: torch.Tensor
    target_position: torch.Tensor
    target_velocity: torch.Tensor
    target_face_normal: torch.Tensor
    ball_position: torch.Tensor
    achieved_position: torch.Tensor
    achieved_velocity: torch.Tensor
    achieved_face_normal: torch.Tensor


@dataclass(frozen=True)
class _PackedR03RewardCycle:
    """One frozen selected-slot decode shared by the ten R03 consumers."""

    clean_errors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    finite: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    admitted: torch.Tensor


@dataclass(frozen=True)
class DirectR06RewardFacts:
    """Compatibility decode of the immutable R06 epoch fact slice."""

    eligible: torch.Tensor
    policy_eligible: torch.Tensor
    producer_fault_bits: torch.Tensor
    common_on_table_outcome: torch.Tensor
    canonical_total: torch.Tensor
    placement_treatment_gain: torch.Tensor


@dataclass(frozen=True)
class DiagnosticN2RewardManagerBundle:
    """Exact graph plus real manager cfg for the disposable N=2 smoke."""

    profile_kind: str
    graph: "LeanActionEpochRewardGraph"
    manager_cfg: dict[str, object]
    diagnostic_unauthorized: bool = True


class LeanRewardError(RuntimeError):
    """Base error for the ActionEpoch Reward graph."""


class LeanRewardConstructionHold(LeanRewardError):
    """The exact epoch or real RewardManager import seam is unavailable."""


class LeanRewardCycleError(LeanRewardError):
    """A manager term was skipped, duplicated, reordered, or malformed."""


def _positive_host_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise LeanRewardConstructionHold(label + " must be a host number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise LeanRewardConstructionHold(label + " must be finite and positive")
    return result


def _finite_host_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise LeanRewardConstructionHold(label + " must be a host number")
    result = float(value)
    if not math.isfinite(result):
        raise LeanRewardConstructionHold(label + " must be finite")
    return result


class LeanActionEpochRewardGraph:
    """Pay lifecycle ordinals 0..13, then record common dense ordinals 14..19."""

    def __init__(self, *, epoch_owner: epoch_v1.ActionEpochOwner) -> None:
        if type(epoch_owner) is not epoch_v1.ActionEpochOwner:
            raise LeanRewardConstructionHold(
                "epoch_owner must be exact ActionEpochOwner"
            )
        if epoch_owner.shot_slot_capacity < 1:
            raise LeanRewardConstructionHold("epoch shot-slot capacity is empty")
        self.epoch_owner = epoch_owner
        self.num_envs = epoch_owner.num_envs
        self.shot_slot_capacity = epoch_owner.shot_slot_capacity
        self.device = epoch_owner.device
        self._cycle_epoch: epoch_v1.ActionEpochRecord | None = None
        self._r03_cycle_cache: _PackedR03RewardCycle | None = None
        self._next_ordinal = 0
        self._next_dense_ordinal = LIFECYCLE_PAYMENT_COUNT
        self._dense_cycle_open = False
        self._completed_cycle_count = 0
        self._actual_closed_cycle_count = 0
        self._poisoned = False
        self._poison_reason: str | None = None
        self._milestone = epoch_owner.milestone
        self._lean_carry_coordinator = None
        self._milestone_configured_income_scale_host = (1.0,) * MANAGER_TERM_COUNT
        self._milestone_configured_income_scale = torch.ones(
            MANAGER_TERM_COUNT, dtype=torch.float64, device=self.device
        )

    def configure_milestone_configured_income(
        self, manager_cfg: Mapping[str, object], step_dt: float
    ) -> None:
        """Bind the same immutable term config later consumed by RewardManager."""

        carry_txn._require_leaf_mutable(self)
        dt = _positive_host_number(step_dt, label="step_dt")
        weights = tuple(
            _finite_host_number(getattr(manager_cfg[name], "weight", None), label=name + " weight")
            for name in MANAGER_NAMES
        )
        configured_host = tuple(weight * dt for weight in weights)
        configured_device = torch.tensor(
            configured_host,
            dtype=torch.float64,
            device=self.device,
        )
        # The Lean root registers this graph before the env installs the
        # RewardManager config.  Preserve the construction-attested tensor
        # identity across that final configuration seam.
        self._milestone_configured_income_scale.copy_(configured_device)
        self._milestone_configured_income_scale_host = configured_host

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def cycle_open(self) -> bool:
        return self._cycle_epoch is not None or self._dense_cycle_open

    @property
    def completed_cycle_count(self) -> int:
        """Monotonic host chronology; increments only after ordinal 19 records."""

        return self._completed_cycle_count

    @property
    def actual_closed_cycle_count(self) -> int:
        """Cycles closed once with RewardManager's independent reward buffer."""

        return self._actual_closed_cycle_count

    def _poison(self, reason: object) -> None:
        self._invalidate_r03_cycle_cache()
        clean = reason.strip() if type(reason) is str else ""
        if not clean:
            clean = "lean Reward cycle failed"
        if not self._poisoned:
            self._poison_reason = clean
        self._poisoned = True

    def _record(self) -> epoch_v1.ActionEpochRecord:
        if self._cycle_epoch is None:
            raise LeanRewardCycleError("Reward fact requested outside a cycle")
        return self._cycle_epoch

    def _selected(self, value: torch.Tensor) -> torch.Tensor:
        record = self._record()
        index = record.current_task_slot
        suffix = (1,) * (value.ndim - 2)
        gather_index = index.reshape(self.num_envs, 1, *suffix).expand(
            self.num_envs, 1, *value.shape[2:]
        )
        return torch.gather(value, 1, gather_index).squeeze(1)

    def _owner_rows(self, owner_kind: str) -> tuple[torch.Tensor, ...]:
        record = self._record()
        try:
            slot = epoch_v1.OWNER_ORDER.index(owner_kind)
        except ValueError as exc:
            raise LeanRewardConstructionHold(
                "epoch owner order lacks " + owner_kind
            ) from exc
        return (
            self._selected(record.fact_valid_bits[:, :, slot]),
            self._selected(record.fact_source_step[:, :, slot]),
            self._selected(record.fact_f32[:, :, slot]),
            self._selected(record.owner_fault_bits[:, :, slot]),
        )

    def _invalidate_r03_cycle_cache(self) -> None:
        self._r03_cycle_cache = None

    def _decode_r03_cycle(self) -> _PackedR03RewardCycle:
        cached = self._r03_cycle_cache
        if cached is not None:
            return cached
        valid, source_step, fact, faults = self._owner_rows("r03_strike_fact")
        record = self._record()
        if fact.shape != (self.num_envs, epoch_v1.OWNER_FACT_F32_WIDTH):
            raise LeanRewardCycleError("R03 epoch fact width differs")
        target_position = fact[:, 0:3]
        target_velocity = fact[:, 3:6]
        target_normal = fact[:, 6:9]
        ball_position = fact[:, 9:12]
        achieved_position = fact[:, 15:18]
        achieved_velocity = fact[:, 18:21]
        achieved_normal = fact[:, 21:24]
        errors = (
            torch.linalg.vector_norm(
                achieved_position - target_position, dim=-1
            ),
            torch.linalg.vector_norm(
                achieved_velocity - target_velocity, dim=-1
            ),
            torch.acos(
                torch.sum(achieved_normal * target_normal, dim=-1).clamp(
                    -1.0, 1.0
                )
            ),
            torch.linalg.vector_norm(
                achieved_position - ball_position, dim=-1
            ),
        )
        finite = tuple(torch.isfinite(error) for error in errors)
        clean_errors = tuple(
            torch.where(valid_error, error, torch.zeros_like(error))
            for error, valid_error in zip(errors, finite)
        )
        present = torch.bitwise_and(valid, R03_PRESENT).ne(0) & faults.eq(0)
        admitted = (
            present
            & torch.bitwise_and(valid, R03_PHYSICALLY_VALID).ne(0)
            # The fact image is sticky; only its source Reward tick is payable.
            & record.reward_cycle_fault.eq(0)
            & source_step.eq(record.reward_cycle_age)
        )
        cached = _PackedR03RewardCycle(
            clean_errors=clean_errors,
            finite=finite,
            admitted=admitted,
        )
        self._r03_cycle_cache = cached
        return cached

    def _r03(self, ordinal: int, scale: float) -> torch.Tensor:
        cached = self._decode_r03_cycle()
        consumer = R03_NAMES[ordinal]
        component = _R03_ERROR_COMPONENT_BY_ORDINAL[ordinal]
        clean_error = cached.clean_errors[component]
        finite = cached.finite[component]
        ratio_sq = torch.square(clean_error / scale)
        if consumer.endswith("_coarse") or consumer == "paddle_center_proximity":
            raw = torch.reciprocal(1.0 + ratio_sq)
        else:
            raw = torch.exp(-ratio_sq)
        self._milestone.add_reward(
            ordinal, raw, raw, cached.admitted, finite,
            self._milestone_configured_income_scale[ordinal],
        )
        return torch.where(
            cached.admitted & finite, raw, torch.zeros_like(raw)
        )

    def _physical(self) -> torch.Tensor:
        valid, source_step, _fact, faults = self._owner_rows("physical_ball")
        record = self._record()
        present = (
            torch.bitwise_and(valid, PHYSICAL_PRESENT).ne(0)
            & faults.eq(0)
            & record.reward_cycle_fault.eq(0)
            & source_step.eq(record.reward_cycle_age)
        )
        contact = torch.bitwise_and(valid, PHYSICAL_SELECTED_CONTACT).ne(0)
        raw = (present & contact).to(torch.float32)
        finite = torch.ones_like(present)
        self._milestone.add_reward(
            10, raw, raw, present, finite,
            self._milestone_configured_income_scale[10],
        )
        return raw

    def _r06(self, ordinal: int) -> torch.Tensor:
        valid, _source_step, fact, faults = self._owner_rows(
            "r06_landing_outcome"
        )
        record = self._record()
        phase = self._selected(record.phase)
        settlement_step = self._selected(record.settlement_step)
        payment_step = self._selected(record.payment_step)
        primitive = fact[:, 0] if ordinal == 11 else fact[:, 1]
        payment = primitive if ordinal == 11 else primitive * fact[:, 2]
        finite = torch.isfinite(primitive) & torch.isfinite(payment)
        eligible = (
            torch.bitwise_and(valid, R06_PRESENT).ne(0)
            & torch.bitwise_and(valid, R06_POLICY_ELIGIBLE).ne(0)
            & torch.bitwise_and(valid, R06_SOURCE_VALID).ne(0)
            & faults.eq(0)
            # Both R06 ordinals share the ordinal-zero frozen before-image.
            # The payment is recorded only after the complete Reward cycle.
            & phase.eq(epoch_v1.PHASE_OUTCOME_SETTLED)
            & settlement_step.ge(0)
            & payment_step.eq(-1)
        )
        self._milestone.add_reward(
            ordinal, primitive, payment, eligible, finite,
            self._milestone_configured_income_scale[ordinal],
        )
        return torch.where(
            eligible & finite,
            torch.where(finite, payment, torch.zeros_like(payment)),
            torch.zeros_like(payment),
        )

    def _r07(self) -> torch.Tensor:
        valid, source_step, fact, faults = self._owner_rows("r07_recovery")
        record = self._record()
        phase = self._selected(record.phase)
        reward_cycle_age = record.reward_cycle_age
        reward = fact[:, 0]
        reward_eligible = fact[:, 2].eq(1.0)
        facts_valid = fact[:, 3].eq(1.0)
        infrastructure_fault = fact[:, 4].ne(0.0)
        raw_score = fact[:, 1]
        components = fact[:, 7:20]
        finite = (
            torch.isfinite(reward) & torch.isfinite(raw_score)
            & torch.isfinite(components).all(dim=1)
        )
        eligible = (
            torch.bitwise_and(valid, R07_PRESENT).ne(0)
            & torch.bitwise_and(valid, R07_NUMERICALLY_VALID).ne(0)
            & reward_eligible
            & facts_valid
            & ~infrastructure_fault
            & faults.eq(0)
            & record.reward_cycle_fault.eq(0)
            # R07 is dense across recovery ticks, but each tick still requires
            # the producer's immediately preceding post-physics publication.
            & reward_cycle_age.gt(0)
            & source_step.eq(reward_cycle_age - 1)
            # RETIRED deliberately retains its last immutable fact image.
            # Current phase is therefore the non-redundant boundary that
            # prevents replaying the final recovery payment.
            & phase.eq(epoch_v1.PHASE_OUTCOME_SETTLED)
        )
        self._milestone.add_reward(
            13, raw_score, reward, eligible, finite,
            self._milestone_configured_income_scale[13],
        )
        return torch.where(
            eligible & finite,
            torch.where(finite, reward, torch.zeros_like(reward)),
            torch.zeros_like(reward),
        )

    def pay(self, ordinal: int, *, scale: float | None = None) -> torch.Tensor:
        carry_txn._require_leaf_mutable(self)
        """Decode and pay the next manager ordinal exactly once."""

        if self._poisoned:
            self._invalidate_r03_cycle_cache()
            raise LeanRewardCycleError(
                "lean Reward graph is poisoned: " + str(self._poison_reason)
            )
        if type(ordinal) is not int or ordinal != self._next_ordinal:
            self._invalidate_r03_cycle_cache()
            raise LeanRewardCycleError(
                f"Reward consumer order differs: expected {self._next_ordinal}, got {ordinal!r}"
            )
        try:
            if ordinal == 0:
                self._invalidate_r03_cycle_cache()
                if self._cycle_epoch is not None or self._dense_cycle_open:
                    raise LeanRewardCycleError("Reward cycle was already open")
                if self._actual_closed_cycle_count != self._completed_cycle_count:
                    raise LeanRewardCycleError(
                        "prior Reward cycle lacks its actual-buffer close"
                    )
                # The returned record includes the exact newly opened cycle
                # and one frozen post-physics fact image for every ordinal.
                self._cycle_epoch = self.epoch_owner.open_reward_cycle()
                self._dense_cycle_open = True
            elif self._cycle_epoch is None:
                raise LeanRewardCycleError("Reward term ran before ordinal zero")

            if ordinal < 10:
                if scale is None:
                    raise LeanRewardCycleError("R03 Reward scale is absent")
                value = self._r03(
                    ordinal, _positive_host_number(scale, label="scale")
                )
            elif ordinal == 10:
                if scale is not None:
                    raise LeanRewardCycleError(
                        "Physical Reward cannot accept a scale"
                    )
                value = self._physical()
            elif ordinal < 13:
                if scale is not None:
                    raise LeanRewardCycleError("R06 Reward cannot accept a scale")
                value = self._r06(ordinal)
            else:
                if scale is not None:
                    raise LeanRewardCycleError("R07 Reward cannot accept a scale")
                value = self._r07()
            self.epoch_owner.pay_reward(ordinal)
            if ordinal == len(R03_NAMES) - 1:
                self._invalidate_r03_cycle_cache()
        except BaseException as exc:
            self._poison(
                "ordinal " + str(ordinal) + " failed: " + type(exc).__name__
            )
            raise

        self._next_ordinal += 1
        if self._next_ordinal == LIFECYCLE_PAYMENT_COUNT:
            self._next_ordinal = 0
            self._cycle_epoch = None
            self._invalidate_r03_cycle_cache()
        return value

    def record_common_dense(
        self, ordinal: int, value: torch.Tensor
    ) -> torch.Tensor:
        """Record one existing Motion reward without extending lifecycle payment bits."""

        carry_txn._require_leaf_mutable(self)
        if self._poisoned:
            self._invalidate_r03_cycle_cache()
            raise LeanRewardCycleError(
                "lean Reward graph is poisoned: " + str(self._poison_reason)
            )
        try:
            if (
                type(ordinal) is not int
                or ordinal != self._next_dense_ordinal
                or not self._dense_cycle_open
                or self._cycle_epoch is not None
                or self._next_ordinal != 0
            ):
                raise LeanRewardCycleError(
                    "common dense Reward order differs: expected "
                    f"{self._next_dense_ordinal}, got {ordinal!r}"
                )
            if (
                type(value) is not torch.Tensor
                or value.device != self.device
                or tuple(value.shape) != (self.num_envs,)
                or value.dtype is not torch.float32
            ):
                raise LeanRewardCycleError("common dense Reward tensor ABI differs")
            finite = torch.isfinite(value)
            eligible = torch.ones_like(finite)
            self._milestone.add_reward(
                ordinal,
                value,
                value,
                eligible,
                finite,
                self._milestone_configured_income_scale[ordinal],
            )
        except BaseException as exc:
            self._poison(
                "common dense ordinal " + str(ordinal) + " failed: "
                + type(exc).__name__
            )
            raise
        self._next_dense_ordinal += 1
        if self._next_dense_ordinal == MANAGER_TERM_COUNT:
            self._next_dense_ordinal = LIFECYCLE_PAYMENT_COUNT
            self._dense_cycle_open = False
            self._completed_cycle_count += 1
        return value

    def close_milestone_actual_reward(self, reward: torch.Tensor) -> None:
        carry_txn._require_leaf_mutable(self)
        """Close one completed cycle with RewardManager's exact output tensor."""

        self._invalidate_r03_cycle_cache()
        if self._poisoned:
            raise LeanRewardCycleError(
                "lean Reward graph is poisoned: " + str(self._poison_reason)
            )
        try:
            if (
                self._cycle_epoch is not None
                or self._next_ordinal != 0
                or self._dense_cycle_open
                or self._next_dense_ordinal != LIFECYCLE_PAYMENT_COUNT
            ):
                raise LeanRewardCycleError(
                    "actual Reward close ran before all twenty terms"
                )
            if self._completed_cycle_count != self._actual_closed_cycle_count + 1:
                raise LeanRewardCycleError(
                    "actual Reward close was skipped, duplicated, or replayed"
                )
            self._milestone.close_actual_step(reward)
        except BaseException as exc:
            self._poison(
                "actual Reward close failed: " + type(exc).__name__
            )
            raise
        self._actual_closed_cycle_count += 1

    def _require_checkpoint_boundary(self, *, dormant: bool) -> None:
        if self._poisoned:
            self._invalidate_r03_cycle_cache()
            raise LeanRewardCycleError("poisoned Reward graph cannot checkpoint")
        if (
            self._cycle_epoch is not None
            or self._next_ordinal != 0
            or self._dense_cycle_open
            or self._next_dense_ordinal != LIFECYCLE_PAYMENT_COUNT
            or self._completed_cycle_count != self._actual_closed_cycle_count
        ):
            self._invalidate_r03_cycle_cache()
            raise LeanRewardCycleError("Reward checkpoint boundary is not closed")
        if dormant and self._completed_cycle_count != 0:
            raise LeanRewardCycleError("Reward restore target is not dormant")

    def _lean_carry_schema(self) -> carry_txn._LeanCarrySchema:
        return carry_txn._LeanCarrySchema(
            "reward",
            (
                ("completed_cycle_count", int),
                ("actual_closed_cycle_count", int),
                ("manager_names", tuple),
                ("configured_income_scale", tuple),
            ),
            (
                carry_txn._LeanCarryTensorSpec(
                    "configured_income_scale_device",
                    (MANAGER_TERM_COUNT,), torch.float64, "attest",
                ),
            ),
        )

    def _lean_carry_construction_views(self):
        return (self._milestone_configured_income_scale,)

    def _lean_carry_capture(self, lease):
        if getattr(lease, "coordinator", None) is not self._lean_carry_coordinator:
            raise LeanRewardCycleError("Reward carry lease differs")
        self._require_checkpoint_boundary(dormant=False)
        host = self._milestone_configured_income_scale_host
        device = self._milestone_configured_income_scale
        if (
            tuple(MANAGER_NAMES) != MANAGER_NAMES
            or type(host) is not tuple or len(host) != MANAGER_TERM_COUNT
        ):
            raise LeanRewardCycleError("Reward configured-income semantic binding differs")
        return carry_txn._LeanCarryCapture((
            self._completed_cycle_count, self._actual_closed_cycle_count,
            tuple(MANAGER_NAMES), host,
        ), (device,))

    def _lean_carry_stage(self, lease, scalars, host_tensors):
        if getattr(lease, "coordinator", None) is not self._lean_carry_coordinator:
            raise LeanRewardCycleError("Reward target carry lease differs")
        self._require_checkpoint_boundary(dormant=True)
        completed, actual, names, configured = scalars
        if (
            completed < 0 or actual != completed or names != tuple(MANAGER_NAMES)
            or type(configured) is not tuple or len(configured) != MANAGER_TERM_COUNT
            or any(type(value) is not float or not math.isfinite(value) for value in configured)
            or configured != self._milestone_configured_income_scale_host
            or tuple(float(value) for value in host_tensors[0].tolist()) != configured
        ):
            raise LeanRewardCycleError("Reward carry semantic ABI differs")
        staging = (host_tensors[0].to(device=self.device, copy=True).contiguous(),)
        return carry_txn._LeanCarryStage(
            scalars, staging, (self._milestone_configured_income_scale,)
        )

    def _lean_carry_target_views(self, lease, stage):
        if lease is not self._lean_carry_coordinator._active_lease:
            raise LeanRewardCycleError("Reward carry target lease differs")
        return (self._milestone_configured_income_scale,)

    def _lean_carry_apply_scalars(self, lease, stage) -> None:
        if not stage.commit_started or lease is not self._lean_carry_coordinator._active_lease:
            raise LeanRewardCycleError("Reward carry commit was not armed")
        self._completed_cycle_count, self._actual_closed_cycle_count = stage.scalars[:2]


ENV_REWARD_DISPATCHER_NAME = "_action_ball_full_mdp_lean_reward_term"


def _env_reward_dispatcher(env: object):
    """Resolve the env class's own bound Reward dispatcher, without graph access."""

    env_type = type(env)
    descriptor = vars(env_type).get(ENV_REWARD_DISPATCHER_NAME)
    instance_state = getattr(env, "__dict__", None)
    if (
        type(instance_state) is not dict
        or ENV_REWARD_DISPATCHER_NAME in instance_state
        or type(descriptor) is not types.FunctionType
    ):
        raise LeanRewardConstructionHold(
            "env has no exact class-owned lean Reward dispatcher"
        )
    bound = descriptor.__get__(env, env_type)
    live = getattr(env, ENV_REWARD_DISPATCHER_NAME, None)
    if (
        not callable(bound)
        or getattr(bound, "__self__", None) is not env
        or getattr(bound, "__func__", None) is not descriptor
        or not callable(live)
        or getattr(live, "__self__", None) is not env
        or getattr(live, "__func__", None) is not descriptor
    ):
        raise LeanRewardConstructionHold(
            "env lean Reward dispatcher binding differs from its class descriptor"
        )
    return bound


def _dispatch_reward_term(
    env: object, *, ordinal: int, scale: float | None
) -> torch.Tensor:
    return _env_reward_dispatcher(env)(ordinal=ordinal, scale=scale)


def common_dense_reward(
    env: object,
    *,
    ordinal: int,
    command_name: str,
    std: float,
    coarse_std: float | None = None,
) -> torch.Tensor:
    """Evaluate one adopted Motion imitation function and record its manager row."""

    if type(ordinal) is not int or not (
        LIFECYCLE_PAYMENT_COUNT <= ordinal < MANAGER_TERM_COUNT
    ):
        raise LeanRewardConstructionHold("common dense ordinal differs")
    name, evaluator_name, _weight, expected_std = COMMON_DENSE_SPECS[
        ordinal - LIFECYCLE_PAYMENT_COUNT
    ]
    if command_name != "motion" or _positive_host_number(
        std, label=name + " std"
    ).hex() != expected_std.hex():
        raise LeanRewardConstructionHold("common dense term parameters differ")
    expected_coarse = (
        BODY_ORIENTATION_COARSE_STD if name == "motion_body_ori" else None
    )
    if coarse_std != expected_coarse:
        raise LeanRewardConstructionHold("common dense coarse kernel differs")
    try:
        evaluator_module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.rewards"
        )
        evaluator = getattr(evaluator_module, evaluator_name)
    except Exception as exc:
        raise LeanRewardConstructionHold(
            "common dense Motion evaluator is unavailable: " + name
        ) from exc
    evaluator_kwargs = {"command_name": command_name, "std": expected_std}
    if expected_coarse is not None:
        evaluator_kwargs["coarse_std"] = expected_coarse
    value = evaluator(env, **evaluator_kwargs)
    return _env_reward_dispatcher(env)(ordinal=ordinal, value=value)


def racket_position(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=0, scale=scale)


def racket_velocity(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=1, scale=scale)


def racket_normal(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=2, scale=scale)


def racket_position_coarse(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=3, scale=scale)


def racket_velocity_coarse(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=4, scale=scale)


def racket_normal_coarse(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=5, scale=scale)


def racket_position_precision(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=6, scale=scale)


def racket_velocity_precision(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=7, scale=scale)


def racket_normal_precision(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=8, scale=scale)


def paddle_center_proximity(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=9, scale=scale)


def physical_selected_contact(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=10, scale=scale)


def common_on_table_outcome(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=11, scale=scale)


def post_contact_placement_guidance(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=12, scale=scale)


def common_recovery_reward_v1(
    env: object, *, scale: float | None = None
) -> torch.Tensor:
    return _dispatch_reward_term(env, ordinal=13, scale=scale)


# Isaac's config serialization pickles callables by ``module + global name``.
# These are therefore deliberate source-level module globals, not closures with
# rewritten metadata.  Their tuple order is the RewardManager ABI above.
LIFECYCLE_REWARD_TERM_CALLABLES = (
    racket_position,
    racket_velocity,
    racket_normal,
    racket_position_coarse,
    racket_velocity_coarse,
    racket_normal_coarse,
    racket_position_precision,
    racket_velocity_precision,
    racket_normal_precision,
    paddle_center_proximity,
    physical_selected_contact,
    common_on_table_outcome,
    post_contact_placement_guidance,
    common_recovery_reward_v1,
)
REWARD_TERM_CALLABLES = LIFECYCLE_REWARD_TERM_CALLABLES + (
    common_dense_reward,
) * len(COMMON_DENSE_SPECS)


def materialize_reward_manager_cfg(
    *,
    weights: Mapping[str, object],
    r03_scales: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact ordered real Isaac ``RewardTermCfg`` mapping."""

    try:
        reward_term_type = importlib.import_module(
            "isaaclab.managers"
        ).RewardTermCfg
    except Exception as exc:
        raise LeanRewardConstructionHold(
            "Isaac RewardTermCfg import surface is unavailable"
        ) from exc
    if set(weights) != set(MANAGER_NAMES):
        raise LeanRewardConstructionHold(
            "Reward weights must name exact ordered twenty"
        )
    if set(r03_scales) != set(R03_NAMES):
        raise LeanRewardConstructionHold("R03 scales must name exact ordered ten")
    result: dict[str, object] = {}
    for ordinal, name in enumerate(MANAGER_NAMES):
        weight = _positive_host_number(weights[name], label=name + " weight")
        if ordinal == 13 and weight != 1.0:
            raise LeanRewardConstructionHold(
                "R07 manager weight must equal one because its epoch fact is weighted"
            )
        params: dict[str, object] = {}
        if ordinal < 10:
            params["scale"] = _positive_host_number(
                r03_scales[name], label=name + " scale"
            )
        elif ordinal >= LIFECYCLE_PAYMENT_COUNT:
            spec = COMMON_DENSE_SPECS[ordinal - LIFECYCLE_PAYMENT_COUNT]
            if weight.hex() != float(spec[2]).hex():
                raise LeanRewardConstructionHold(
                    name + " weight must equal the common dense contract"
                )
            params.update(
                ordinal=ordinal,
                command_name="motion",
                std=float(spec[3]),
            )
            if name == "motion_body_ori":
                params["coarse_std"] = BODY_ORIENTATION_COARSE_STD
        result[name] = reward_term_type(
            func=REWARD_TERM_CALLABLES[ordinal], weight=weight, params=params
        )
    return result


def materialize_diagnostic_n2_reward_manager_cfg(
    *, epoch_owner: epoch_v1.ActionEpochOwner
) -> DiagnosticN2RewardManagerBundle:
    """Materialize the sole code-owned disposable N=2 Reward profile.

    This narrow entry point accepts no caller weights/scales and is the only
    factory-facing API for the first optimizer-update diagnostic.  The generic
    materializer remains an internal/testable mechanism; it is not a runtime
    numeric-authority seam.
    """

    graph = LeanActionEpochRewardGraph(epoch_owner=epoch_owner)
    manager_cfg = materialize_reward_manager_cfg(
        weights=DIAGNOSTIC_N2_WEIGHTS,
        r03_scales=DIAGNOSTIC_N2_R03_SCALES,
    )
    return DiagnosticN2RewardManagerBundle(
        profile_kind=DIAGNOSTIC_N2_REWARD_PROFILE_KIND,
        graph=graph,
        manager_cfg=manager_cfg,
    )


__all__ = [
    "DIAGNOSTIC_UNAUTHORIZED",
    "RUNTIME_INTEGRATED",
    "LAUNCH_AUTHORIZED",
    "GRAPH_ATTR",
    "ENV_REWARD_DISPATCHER_NAME",
    "ORDERED_CONSUMERS",
    "LIFECYCLE_PAYMENT_COUNT",
    "LIFECYCLE_MANAGER_NAMES",
    "COMMON_DENSE_SPECS",
    "COMMON_DENSE_NAMES",
    "BODY_ORIENTATION_COARSE_STD",
    "MANAGER_NAMES",
    "MANAGER_TERM_COUNT",
    "DIAGNOSTIC_N2_REWARD_PROFILE_KIND",
    "DIAGNOSTIC_N2_WEIGHTS",
    "DIAGNOSTIC_N2_R03_SCALES",
    "DirectR03RewardFacts",
    "DirectR06RewardFacts",
    "DiagnosticN2RewardManagerBundle",
    "LeanRewardError",
    "LeanRewardConstructionHold",
    "LeanRewardCycleError",
    "LeanActionEpochRewardGraph",
    *LIFECYCLE_MANAGER_NAMES,
    "common_dense_reward",
    "LIFECYCLE_REWARD_TERM_CALLABLES",
    "REWARD_TERM_CALLABLES",
    "materialize_reward_manager_cfg",
    "materialize_diagnostic_n2_reward_manager_cfg",
]
