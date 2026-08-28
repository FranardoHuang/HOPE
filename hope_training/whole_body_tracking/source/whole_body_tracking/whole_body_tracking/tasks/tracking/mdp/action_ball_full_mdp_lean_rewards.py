"""Lean RewardManager graph over one immutable ActionEpoch.

Post-physics producers publish once into their fixed ActionEpoch fact slices.
RewardManager then captures one immutable record at ordinal zero and decodes
the fourteen lifecycle values from that snapshot.  Six common imitation terms
and four measured-paddle motion priors then reuse the existing reward functions
and record into the same milestone accumulator.  Reward terms never requery a
lifecycle producer, consume a receipt, compare source digests, or read
producer-private state.

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

try:
    from . import action_ball_full_mdp_reward_contract as reward_contract
except ImportError:
    import action_ball_full_mdp_reward_contract as reward_contract

try:
    from . import action_ball_full_mdp_regularization as regularization
except ImportError:
    import action_ball_full_mdp_regularization as regularization

try:
    from . import action_ball_full_mdp_paddle_prior as paddle_prior
except ImportError:
    import action_ball_full_mdp_paddle_prior as paddle_prior


DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

GRAPH_ATTR = "action_ball_full_mdp_lean_reward_graph"
ENV_REWARD_HOT_PATH_ATTR = "_action_ball_full_mdp_reward_hot_path"
ORDERED_CONSUMERS = epoch_v1.REWARD_CONSUMER_ORDER
LIFECYCLE_PAYMENT_COUNT = reward_contract.LIFECYCLE_PAYMENT_COUNT

R03_NAMES = tuple(name.split(":", 1)[1] for name in ORDERED_CONSUMERS[:10])
_R03_ERROR_COMPONENT_BY_ORDINAL = (0, 1, 2, 0, 1, 2, 0, 1, 2, 3)
LIFECYCLE_MANAGER_NAMES = tuple(
    name.split(":", 1)[1] for name in ORDERED_CONSUMERS
)
if LIFECYCLE_MANAGER_NAMES != reward_contract.LIFECYCLE_MANAGER_NAMES:
    raise RuntimeError("lean lifecycle Reward order differs from shared contract")
COMMON_DENSE_SPECS = reward_contract.COMMON_DENSE_SPECS
PADDLE_MOTION_PRIOR_SPECS = reward_contract.PADDLE_MOTION_PRIOR_SPECS
REGULARIZATION_SPECS = reward_contract.REGULARIZATION_SPECS
ALL_DENSE_SPECS = COMMON_DENSE_SPECS + PADDLE_MOTION_PRIOR_SPECS
COMMON_DENSE_NAMES = reward_contract.COMMON_DENSE_NAMES
PADDLE_MOTION_PRIOR_NAMES = reward_contract.PADDLE_MOTION_PRIOR_NAMES
REGULARIZATION_NAMES = reward_contract.REGULARIZATION_NAMES
BODY_ORIENTATION_COARSE_STD = next(
    spec.coarse_std
    for spec in COMMON_DENSE_SPECS
    if spec.manager_name == "motion_body_ori"
)
MANAGER_NAMES = reward_contract.MANAGER_NAMES
MANAGER_TERM_COUNT = reward_contract.REWARD_TERM_COUNT
PADDLE_MOTION_PRIOR_FIRST_ORDINAL = (
    LIFECYCLE_PAYMENT_COUNT + len(COMMON_DENSE_SPECS)
)
REGULARIZATION_FIRST_ORDINAL = (
    PADDLE_MOTION_PRIOR_FIRST_ORDINAL + len(PADDLE_MOTION_PRIOR_SPECS)
)
if (
    tuple(spec.manager_name for spec in PADDLE_MOTION_PRIOR_SPECS)
    != (
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
        "motion_racket_long_axis",
    )
    or len({spec.command_name for spec in PADDLE_MOTION_PRIOR_SPECS}) != 1
    or any(
        float(spec.scale_in_strike_window) != 1.0
        for spec in PADDLE_MOTION_PRIOR_SPECS
    )
):
    raise RuntimeError("paddle pack requires the full-strength ordered contract")

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
    "action_ball_full_mdp_diagnostic_n2_reward_profile_v3"
)
DIAGNOSTIC_N2_WEIGHTS = {
    **{name: 1.0 for name in R03_NAMES},
    "physical_selected_contact": 1.0,
    "common_on_table_outcome": 20.0,
    "post_contact_placement_guidance": 1.0,
    # R07 fact is already weighted by its construction-owned profile.
    "common_recovery_reward_v1": 1.0,
    **{spec.manager_name: spec.manager_weight for spec in ALL_DENSE_SPECS},
    **{
        spec.manager_name: spec.manager_weight
        for spec in REGULARIZATION_SPECS
    },
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
class _FrozenR06RewardCycle:
    """One frozen selected-slot source shared by both R06 consumers."""

    valid: torch.Tensor
    fact: torch.Tensor
    faults: torch.Tensor
    phase: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor


@dataclass(frozen=True)
class _SealedEnvRewardHotPath:
    """Construction-bound dispatcher and evaluator identities for one env."""

    graph: "LeanActionEpochRewardGraph"
    graph_type: type
    dispatcher: object
    common_evaluators: tuple[object, ...]
    paddle_command_lookup: object
    paddle_tracking_errors: object
    lifecycle_payment_count: int
    paddle_first_ordinal: int
    regularization_first_ordinal: int


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
    """Pay lifecycle rows, then record every shared-contract dense row."""

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
        self._r06_cycle_cache: _FrozenR06RewardCycle | None = None
        self._paddle_reward_cycle_cache: tuple[torch.Tensor, ...] | None = None
        self._paddle_playback_cycle_mask: torch.Tensor | None = None
        self._paddle_error_cycle_matrix: torch.Tensor | None = None
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
        """Monotonic host chronology; increments only after the final row."""

        return self._completed_cycle_count

    @property
    def actual_closed_cycle_count(self) -> int:
        """Cycles closed once with RewardManager's independent reward buffer."""

        return self._actual_closed_cycle_count

    def _poison(self, reason: object) -> None:
        self._invalidate_r03_cycle_cache()
        self._invalidate_r06_cycle_cache()
        self._clear_paddle_playback_cycle_cache()
        clean = reason.strip() if type(reason) is str else ""
        if not clean:
            clean = "lean Reward cycle failed"
        if not self._poisoned:
            self._poison_reason = clean
        self._poisoned = True

    def _clear_paddle_playback_cycle_cache(self) -> None:
        self._paddle_reward_cycle_cache = None
        self._paddle_playback_cycle_mask = None
        self._paddle_error_cycle_matrix = None

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

    def _invalidate_r06_cycle_cache(self) -> None:
        self._r06_cycle_cache = None

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
        cached = self._r06_cycle_cache
        if cached is None:
            cached = self._decode_r06_cycle()
            self._r06_cycle_cache = cached
        primitive = cached.fact[:, 0] if ordinal == 11 else cached.fact[:, 1]
        payment = (
            primitive if ordinal == 11 else primitive * cached.fact[:, 2]
        )
        finite = torch.isfinite(primitive) & torch.isfinite(payment)
        eligible = (
            torch.bitwise_and(cached.valid, R06_PRESENT).ne(0)
            & torch.bitwise_and(cached.valid, R06_POLICY_ELIGIBLE).ne(0)
            & torch.bitwise_and(cached.valid, R06_SOURCE_VALID).ne(0)
            & cached.faults.eq(0)
            # Both R06 ordinals share the ordinal-zero frozen before-image.
            # The payment is recorded only after the complete Reward cycle.
            & cached.phase.eq(epoch_v1.PHASE_OUTCOME_SETTLED)
            & cached.settlement_step.ge(0)
            & cached.payment_step.eq(-1)
        )
        self._milestone.add_reward(
            ordinal,
            primitive,
            payment,
            eligible,
            finite,
            self._milestone_configured_income_scale[ordinal],
        )
        return torch.where(
            eligible & finite,
            torch.where(finite, payment, torch.zeros_like(payment)),
            torch.zeros_like(payment),
        )

    def _decode_r06_cycle(self) -> _FrozenR06RewardCycle:
        valid, _source_step, fact, faults = self._owner_rows(
            "r06_landing_outcome"
        )
        record = self._record()
        phase = self._selected(record.phase)
        settlement_step = self._selected(record.settlement_step)
        payment_step = self._selected(record.payment_step)
        return _FrozenR06RewardCycle(
            valid=valid,
            fact=fact,
            faults=faults,
            phase=phase,
            settlement_step=settlement_step,
            payment_step=payment_step,
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

    def paddle_motion_prior_cycle_value(
        self,
        env: object,
        *,
        ordinal: int,
        command_lookup: object,
        tracking_errors: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return one column from the same-cycle four-channel paddle pack."""

        carry_txn._require_leaf_mutable(self)
        first = PADDLE_MOTION_PRIOR_FIRST_ORDINAL
        if (
            type(ordinal) is not int
            or not first <= ordinal < REGULARIZATION_FIRST_ORDINAL
            or not self._dense_cycle_open
            or ordinal != self._next_dense_ordinal
        ):
            raise LeanRewardCycleError("paddle pack requested outside its dense row")
        cached = self._paddle_reward_cycle_cache
        playback_active = None
        telemetry_errors = None
        if cached is None:
            if ordinal != first:
                raise LeanRewardCycleError("paddle pack first row was skipped")
            cmd = command_lookup(
                env, PADDLE_MOTION_PRIOR_SPECS[0].command_name
            )
            errors = tracking_errors(cmd)
            # Keep the adopted four scalar kernel expressions bitwise exact on
            # CUDA while sharing the materially larger teacher/error decode.
            value = tuple(
                paddle_prior.coarse_precision_kernel(
                    errors[:, column],
                    precision_std=spec.std,
                    coarse_std=spec.coarse_std,
                )
                for column, spec in enumerate(PADDLE_MOTION_PRIOR_SPECS)
            )
            try:
                motion = cmd._motion()
                accessor = getattr(
                    motion, "action_ball_full_mdp_playback_active_mask", None
                )
                if not callable(accessor):
                    raise RuntimeError(
                        "Motion lacks its public FullMDP playback-active accessor"
                    )
                candidate = accessor()
                if (
                    type(candidate) is not torch.Tensor
                    or tuple(candidate.shape) != (self.num_envs,)
                    or candidate.device != self.device
                    or candidate.dtype is not torch.bool
                    or not candidate.is_contiguous()
                ):
                    raise RuntimeError("Motion playback telemetry ABI differs")
                if (
                    type(errors) is not torch.Tensor
                    or tuple(errors.shape)
                    != (self.num_envs, len(PADDLE_MOTION_PRIOR_SPECS))
                    or errors.device != self.device
                    or errors.dtype is not torch.float32
                    or not errors.is_contiguous()
                ):
                    raise RuntimeError("Motion paddle-error telemetry ABI differs")
                playback_active, telemetry_errors = candidate, errors
            except Exception:
                # This denominator is diagnostic only.  Reward remains the
                # exact measured-paddle pack even when telemetry is absent.
                playback_active = None
                telemetry_errors = None
            cached = value
            self._paddle_reward_cycle_cache = cached
        column = ordinal - first
        return (
            cached[column],
            playback_active,
            telemetry_errors,
        )

    def pay(self, ordinal: int, *, scale: float | None = None) -> torch.Tensor:
        carry_txn._require_leaf_mutable(self)
        """Decode and pay the next manager ordinal exactly once."""

        if self._poisoned:
            self._invalidate_r03_cycle_cache()
            self._invalidate_r06_cycle_cache()
            raise LeanRewardCycleError(
                "lean Reward graph is poisoned: " + str(self._poison_reason)
            )
        if type(ordinal) is not int or ordinal != self._next_ordinal:
            self._invalidate_r03_cycle_cache()
            self._invalidate_r06_cycle_cache()
            raise LeanRewardCycleError(
                f"Reward consumer order differs: expected {self._next_ordinal}, got {ordinal!r}"
            )
        try:
            if ordinal == 0:
                self._invalidate_r03_cycle_cache()
                self._invalidate_r06_cycle_cache()
                self._clear_paddle_playback_cycle_cache()
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
            if ordinal == 12:
                self._invalidate_r06_cycle_cache()
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
            self._invalidate_r06_cycle_cache()
        return value

    def record_common_dense(
        self,
        ordinal: int,
        value: torch.Tensor,
        *,
        paddle_playback_active: torch.Tensor | None = None,
        paddle_error_components: torch.Tensor | None = None,
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
            if (
                PADDLE_MOTION_PRIOR_FIRST_ORDINAL
                <= ordinal
                < REGULARIZATION_FIRST_ORDINAL
            ):
                if ordinal == PADDLE_MOTION_PRIOR_FIRST_ORDINAL:
                    if paddle_playback_active is not None and (
                        type(paddle_playback_active) is not torch.Tensor
                        or tuple(paddle_playback_active.shape)
                        != (self.num_envs,)
                        or paddle_playback_active.device != self.device
                        or paddle_playback_active.dtype is not torch.bool
                        or not paddle_playback_active.is_contiguous()
                    ):
                        # Telemetry is optional, but an explicitly injected
                        # tensor still has to obey the local ABI.
                        raise LeanRewardCycleError(
                            "paddle motion-prior playback telemetry ABI differs"
                        )
                    if paddle_error_components is not None and (
                        type(paddle_error_components) is not torch.Tensor
                        or tuple(paddle_error_components.shape)
                        != (self.num_envs, len(PADDLE_MOTION_PRIOR_SPECS))
                        or paddle_error_components.device != self.device
                        or paddle_error_components.dtype is not torch.float32
                        or not paddle_error_components.is_contiguous()
                    ):
                        raise LeanRewardCycleError(
                            "paddle motion-prior error telemetry ABI differs"
                        )
                    if (paddle_playback_active is None) != (
                        paddle_error_components is None
                    ):
                        raise LeanRewardCycleError(
                            "paddle motion-prior telemetry resolved partially"
                        )
                    self._paddle_playback_cycle_mask = paddle_playback_active
                    self._paddle_error_cycle_matrix = paddle_error_components
                elif (
                    paddle_playback_active is not None
                    or paddle_error_components is not None
                ):
                    raise LeanRewardCycleError(
                        "paddle telemetry must be resolved once"
                    )
            elif (
                paddle_playback_active is not None
                or paddle_error_components is not None
            ):
                raise LeanRewardCycleError(
                    "non-paddle dense row received paddle telemetry"
                )
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
            if (
                PADDLE_MOTION_PRIOR_FIRST_ORDINAL
                <= ordinal
                < REGULARIZATION_FIRST_ORDINAL
            ):
                if (
                    self._paddle_playback_cycle_mask is None
                    or self._paddle_error_cycle_matrix is None
                ):
                    self._milestone.add_paddle_motion_prior_unavailable(ordinal)
                else:
                    self._milestone.add_paddle_motion_prior_playback(
                        ordinal,
                        value,
                        self._paddle_playback_cycle_mask,
                        self._paddle_error_cycle_matrix[
                            :, ordinal - PADDLE_MOTION_PRIOR_FIRST_ORDINAL
                        ],
                    )
                if ordinal + 1 == REGULARIZATION_FIRST_ORDINAL:
                    self._clear_paddle_playback_cycle_cache()
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
            self._clear_paddle_playback_cycle_cache()
        return value

    def close_milestone_actual_reward(self, reward: torch.Tensor) -> None:
        carry_txn._require_leaf_mutable(self)
        """Close one completed cycle with RewardManager's exact output tensor."""

        self._invalidate_r03_cycle_cache()
        self._invalidate_r06_cycle_cache()
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
                    "actual Reward close ran before all shared-contract terms"
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
            self._invalidate_r06_cycle_cache()
            raise LeanRewardCycleError("poisoned Reward graph cannot checkpoint")
        if (
            self._cycle_epoch is not None
            or self._next_ordinal != 0
            or self._dense_cycle_open
            or self._next_dense_ordinal != LIFECYCLE_PAYMENT_COUNT
            or self._completed_cycle_count != self._actual_closed_cycle_count
            or self._r03_cycle_cache is not None
            or self._r06_cycle_cache is not None
            or self._paddle_reward_cycle_cache is not None
            or self._paddle_playback_cycle_mask is not None
            or self._paddle_error_cycle_matrix is not None
        ):
            self._invalidate_r03_cycle_cache()
            self._invalidate_r06_cycle_cache()
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


def seal_env_reward_hot_path(
    env: object, graph: LeanActionEpochRewardGraph
) -> _SealedEnvRewardHotPath:
    """Resolve immutable Reward callables once at the construction boundary."""

    if type(graph) is not LeanActionEpochRewardGraph:
        raise LeanRewardConstructionHold("sealed Reward graph type differs")
    env_type = type(env)
    descriptor = vars(env_type).get(ENV_REWARD_DISPATCHER_NAME)
    instance_state = getattr(env, "__dict__", None)
    if (
        type(instance_state) is not dict
        or ENV_REWARD_DISPATCHER_NAME in instance_state
        or ENV_REWARD_HOT_PATH_ATTR in instance_state
        or type(descriptor) is not types.FunctionType
    ):
        raise LeanRewardConstructionHold(
            "env has no unique class-owned lean Reward dispatcher"
        )
    evaluator_module = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.rewards"
    )
    common_evaluators = tuple(
        getattr(evaluator_module, spec.evaluator_name)
        for spec in COMMON_DENSE_SPECS
    )
    paddle_evaluator_module = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.hope_rewards"
    )
    paddle_command_lookup = getattr(paddle_evaluator_module, "_cmd")
    paddle_tracking_errors = getattr(
        paddle_evaluator_module, "motion_racket_tracking_errors_now"
    )
    if not callable(paddle_command_lookup) or not callable(paddle_tracking_errors):
        raise LeanRewardConstructionHold("paddle evaluator callables differ")
    return _SealedEnvRewardHotPath(
        graph=graph,
        graph_type=LeanActionEpochRewardGraph,
        dispatcher=descriptor,
        common_evaluators=common_evaluators,
        paddle_command_lookup=paddle_command_lookup,
        paddle_tracking_errors=paddle_tracking_errors,
        lifecycle_payment_count=LIFECYCLE_PAYMENT_COUNT,
        paddle_first_ordinal=PADDLE_MOTION_PRIOR_FIRST_ORDINAL,
        regularization_first_ordinal=REGULARIZATION_FIRST_ORDINAL,
    )


def _env_reward_hot_path(env: object) -> _SealedEnvRewardHotPath:
    return env.__dict__[ENV_REWARD_HOT_PATH_ATTR]


def _env_reward_dispatcher(env: object):
    """Return the construction-bound exact unbound env dispatcher."""

    return _env_reward_hot_path(env).dispatcher


def _dispatch_reward_term(
    env: object, *, ordinal: int, scale: float | None
) -> torch.Tensor:
    return _env_reward_dispatcher(env)(env, ordinal=ordinal, scale=scale)


def common_dense_reward(
    env: object,
    *,
    ordinal: int,
    command_name: str,
    std: float,
    coarse_std: float | None = None,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Evaluate one adopted Motion imitation function and record its manager row."""

    if type(ordinal) is not int or not (
        LIFECYCLE_PAYMENT_COUNT
        <= ordinal
        < LIFECYCLE_PAYMENT_COUNT + len(COMMON_DENSE_SPECS)
    ):
        raise LeanRewardConstructionHold("common dense ordinal differs")
    spec = COMMON_DENSE_SPECS[
        ordinal - LIFECYCLE_PAYMENT_COUNT
    ]
    if command_name != spec.command_name or _positive_host_number(
        std, label=spec.manager_name + " std"
    ).hex() != spec.std.hex():
        raise LeanRewardConstructionHold("common dense term parameters differ")
    if coarse_std != spec.coarse_std:
        raise LeanRewardConstructionHold("common dense coarse kernel differs")
    if spec.body_scope is None:
        if body_names is not None:
            raise LeanRewardConstructionHold("anchor dense body scope differs")
    elif spec.body_scope == reward_contract.UPPER_EXCEPT_HELD_WRIST:
        if (
            type(body_names) is not tuple
            or not body_names
            or reward_contract.HELD_RACKET_WRIST_BODY_NAME in body_names
            or len(set(body_names)) != len(body_names)
        ):
            raise LeanRewardConstructionHold("common dense held-wrist scope differs")
    else:
        raise LeanRewardConstructionHold("common dense body scope is unknown")
    binding = _env_reward_hot_path(env)
    evaluator = binding.common_evaluators[
        ordinal - LIFECYCLE_PAYMENT_COUNT
    ]
    evaluator_kwargs = {"command_name": command_name, "std": spec.std}
    if spec.coarse_std is not None:
        evaluator_kwargs["coarse_std"] = spec.coarse_std
    if body_names is not None:
        evaluator_kwargs["body_names"] = body_names
    value = evaluator(env, **evaluator_kwargs)
    return binding.dispatcher(env, ordinal=ordinal, value=value)


def paddle_motion_prior_reward(
    env: object,
    *,
    ordinal: int,
    command_name: str,
    std: float,
    coarse_std: float,
    scale_in_strike_window: float,
) -> torch.Tensor:
    """Evaluate one full-phase official-site motion prior and record its row."""

    first = PADDLE_MOTION_PRIOR_FIRST_ORDINAL
    if type(ordinal) is not int or not first <= ordinal < REGULARIZATION_FIRST_ORDINAL:
        raise LeanRewardConstructionHold("paddle motion-prior ordinal differs")
    spec = PADDLE_MOTION_PRIOR_SPECS[ordinal - first]
    if (
        command_name != spec.command_name
        or _positive_host_number(
            std, label=spec.manager_name + " std"
        ).hex()
        != spec.std.hex()
        or spec.coarse_std is None
        or _positive_host_number(
            coarse_std, label=spec.manager_name + " coarse std"
        ).hex()
        != spec.coarse_std.hex()
        or _positive_host_number(
            scale_in_strike_window,
            label=spec.manager_name + " strike-window scale",
        ).hex()
        != float(spec.scale_in_strike_window).hex()
    ):
        raise LeanRewardConstructionHold(
            "paddle motion-prior term parameters differ"
        )
    binding = _env_reward_hot_path(env)
    value, playback_active, paddle_error_components = (
        binding.graph.paddle_motion_prior_cycle_value(
            env,
            ordinal=ordinal,
            command_lookup=binding.paddle_command_lookup,
            tracking_errors=binding.paddle_tracking_errors,
        )
    )
    return binding.dispatcher(
        env,
        ordinal=ordinal,
        value=value,
        paddle_playback_active=playback_active,
        paddle_error_components=paddle_error_components,
    )


def _regularization_action_term(env: object) -> object:
    try:
        action = env.action_manager.get_term("joint_pos")
    except Exception as exc:
        raise LeanRewardConstructionHold(
            "FullMDP regularization lacks joint_pos action term"
        ) from exc
    names = tuple(getattr(action, "_joint_names", ()))
    raw_ids = getattr(action, "_joint_ids", slice(None))
    if isinstance(raw_ids, slice):
        ids = tuple(range(regularization.JOINT_COUNT))[raw_ids]
    else:
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        ids = tuple(int(value) for value in raw_ids)
    if (
        len(names) != regularization.JOINT_COUNT
        or len(set(names)) != regularization.JOINT_COUNT
        or ids != tuple(range(regularization.JOINT_COUNT))
    ):
        raise LeanRewardConstructionHold(
            "FullMDP regularization requires identity 31-joint action order"
        )
    return action


def _regularization_robot_data(env: object, action: object) -> object:
    asset = getattr(action, "_asset", None)
    data = getattr(asset, "data", None)
    names = tuple(
        getattr(data, "joint_names", getattr(asset, "joint_names", ()))
    )
    if (
        data is None
        or len(names) != regularization.JOINT_COUNT
        or len(set(names)) != regularization.JOINT_COUNT
        or tuple(getattr(action, "_joint_names", ())) != names
    ):
        raise LeanRewardConstructionHold(
            "FullMDP regularization requires identity 31-joint robot order"
        )
    return data


def regularization_reward(env: object, *, ordinal: int) -> torch.Tensor:
    """Evaluate one shared negative-cost kernel and record its manager row."""

    if (
        type(ordinal) is not int
        or not REGULARIZATION_FIRST_ORDINAL <= ordinal < MANAGER_TERM_COUNT
    ):
        raise LeanRewardConstructionHold("regularization ordinal differs")
    spec = REGULARIZATION_SPECS[ordinal - REGULARIZATION_FIRST_ORDINAL]
    action = _regularization_action_term(env)
    data = _regularization_robot_data(env, action)
    if spec.evaluator_name == "action_rate_l2":
        value = regularization.action_rate_l2(
            env.action_manager.action, env.action_manager.prev_action
        )
    elif spec.evaluator_name == "qdes_limit_barrier_v2":
        value = regularization.soft_limit_barrier_v2(
            action.processed_actions,
            data.soft_joint_pos_limits,
            data.default_joint_pos,
            data.joint_pos_limits,
        )
    elif spec.evaluator_name == "qdes_projection_penalty":
        value = regularization.qdes_projection_penalty(
            action.pre_clamp_qdes,
            action.nominal_projected_qdes,
            action.nominal_projection_span,
            action.pre_clamp_qdes_valid,
            action.nominal_projected_qdes_valid,
        )
    elif spec.evaluator_name == "actual_joint_limit_barrier_v2":
        value = regularization.soft_limit_barrier_v2(
            data.joint_pos,
            data.soft_joint_pos_limits,
            data.default_joint_pos,
            data.joint_pos_limits,
        )
    else:
        raise LeanRewardConstructionHold(
            "regularization evaluator differs: " + spec.evaluator_name
        )
    if (
        type(value) is not torch.Tensor
        or tuple(value.shape) != (int(env.num_envs),)
        or value.dtype is not torch.float32
        or not value.is_contiguous()
    ):
        raise LeanRewardConstructionHold(
            "regularization evaluator tensor ABI differs: " + spec.manager_name
        )
    return _env_reward_dispatcher(env)(env, ordinal=ordinal, value=value)


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
) * len(COMMON_DENSE_SPECS) + (
    paddle_motion_prior_reward,
) * len(PADDLE_MOTION_PRIOR_SPECS) + (
    regularization_reward,
) * len(REGULARIZATION_SPECS)


def _a3_upper_except_held_wrist_body_names() -> tuple[str, ...]:
    try:
        robot_module = importlib.import_module(
            "whole_body_tracking.robots.agibot_a3"
        )
        upper = robot_module.A3_UPPER_TRACKED
        return reward_contract.upper_except_held_wrist_body_names(upper)
    except Exception as exc:
        raise LeanRewardConstructionHold(
            "A3 non-wrist upper-body scope is unavailable"
        ) from exc


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
            "Reward weights must name the exact shared ordered contract"
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
            if ordinal >= REGULARIZATION_FIRST_ORDINAL:
                spec = REGULARIZATION_SPECS[
                    ordinal - REGULARIZATION_FIRST_ORDINAL
                ]
                if weight.hex() != spec.manager_weight.hex():
                    raise LeanRewardConstructionHold(
                        name + " weight must equal shared regularization contract"
                    )
                params["ordinal"] = ordinal
                result[name] = reward_term_type(
                    func=REWARD_TERM_CALLABLES[ordinal],
                    weight=weight,
                    params=params,
                )
                continue
            spec = ALL_DENSE_SPECS[ordinal - LIFECYCLE_PAYMENT_COUNT]
            if weight.hex() != spec.manager_weight.hex():
                raise LeanRewardConstructionHold(
                    name + " weight must equal the shared dense contract"
                )
            params.update(
                ordinal=ordinal,
                command_name=spec.command_name,
                std=spec.std,
            )
            if spec.coarse_std is not None:
                params["coarse_std"] = spec.coarse_std
            if spec.body_scope == reward_contract.UPPER_EXCEPT_HELD_WRIST:
                params["body_names"] = (
                    _a3_upper_except_held_wrist_body_names()
                )
            elif spec.body_scope is not None:
                raise LeanRewardConstructionHold(
                    name + " dense body scope is unknown"
                )
            if spec.scale_in_strike_window is not None:
                params["scale_in_strike_window"] = (
                    spec.scale_in_strike_window
                )
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
    "PADDLE_MOTION_PRIOR_SPECS",
    "REGULARIZATION_SPECS",
    "ALL_DENSE_SPECS",
    "COMMON_DENSE_NAMES",
    "PADDLE_MOTION_PRIOR_NAMES",
    "REGULARIZATION_NAMES",
    "BODY_ORIENTATION_COARSE_STD",
    "MANAGER_NAMES",
    "MANAGER_TERM_COUNT",
    "PADDLE_MOTION_PRIOR_FIRST_ORDINAL",
    "REGULARIZATION_FIRST_ORDINAL",
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
    "paddle_motion_prior_reward",
    "regularization_reward",
    "LIFECYCLE_REWARD_TERM_CALLABLES",
    "REWARD_TERM_CALLABLES",
    "materialize_reward_manager_cfg",
    "materialize_diagnostic_n2_reward_manager_cfg",
]
