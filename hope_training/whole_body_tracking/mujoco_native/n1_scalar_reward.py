"""Pure diagnostic scalar reward for the fixed-question native N1 C-lite arm.

The function in this module combines caller-owned motion and balance terms
with two task layers:

* a one-sample Cauchy kernel on the achieved selected-rubber centre to ball
  centre distance, which preserves a bounded miss gradient; and
* a pay-once observed-landing term gated by the existing native reward/event
  eligibility contract.

This module does not derive a desired contact state, solve an inverse problem,
predict ball flight, classify contact, or decide table geometry.  Centres,
event eligibility, and the optional opponent-side-out fact must come from
their respective upstream authorities.  Contradictory, missing, or non-finite
facts fail closed with :class:`N1ScalarRewardError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .n1_reward_event_kernel import N1RewardEligibility


Vector3 = Tuple[float, float, float]


class N1ScalarRewardError(ValueError):
    """The scalar-reward inputs cannot authorize a finite diagnostic value."""


@dataclass(frozen=True)
class N1ScalarRewardSpec:
    """Fixed-question C-lite reward magnitudes.

    ``motion`` and ``balance`` are already-scaled caller terms.  The remaining
    fields are diagnostic magnitudes, not physics parameters.  An observed
    legal landing is a fixed ``1.0`` event at the default scale: its lower
    bound and peak are therefore both ``1.0``.  An observed opponent-side out
    is capped at ``0.5``, no more than half that legal value.
    """

    miss_distance_scale_m: float = 0.15
    miss_proximity_weight: float = 1.0
    legal_landing_reward: float = 1.0
    opponent_side_out_reward: float = 0.5


@dataclass(frozen=True)
class N1ScalarRewardLatch:
    """Caller-owned, per-swing pay-once state for the observed outcome."""

    observed_outcome_paid: bool = False


@dataclass(frozen=True)
class N1ScalarRewardInput:
    """Facts for one pure scalar-reward evaluation.

    The two centres are required only when ``miss_sample_eligible`` is true.
    They must represent the achieved selected-rubber centre and ball centre in
    the same frame at the same externally selected sample; this function does
    not time-propagate either value.

    ``observed_opponent_side_out`` is an explicit upstream outcome fact.  It is
    not inferred from a position here and is valid only for a resolved,
    net-clearing, non-legal outcome in ``event_eligibility``.
    """

    motion_reward: float
    balance_reward: float
    miss_sample_eligible: bool
    selected_rubber_center_w_m: Optional[Vector3]
    ball_center_w_m: Optional[Vector3]
    event_eligibility: N1RewardEligibility
    observed_opponent_side_out: bool
    latch: N1ScalarRewardLatch


@dataclass(frozen=True)
class N1ScalarRewardOutput:
    """Per-term accounting plus the next caller-owned event latch."""

    motion_reward: float
    balance_reward: float
    miss_proximity_reward: float
    observed_legal_landing_reward: float
    observed_opponent_side_out_reward: float
    total_reward: float
    observed_outcome_paid_now: bool
    next_latch: N1ScalarRewardLatch

    def term_sum(self) -> float:
        """Recompute the scalar from its five independently reported terms."""

        return math.fsum(
            (
                self.motion_reward,
                self.balance_reward,
                self.miss_proximity_reward,
                self.observed_legal_landing_reward,
                self.observed_opponent_side_out_reward,
            )
        )


_ELIGIBILITY_BOOL_FIELDS = (
    "motion_mimic_denominator",
    "contact_target_denominator",
    "closed_swing_denominator",
    "actual_contact_numerator",
    "achieved_outgoing_flight_denominator",
    "predicted_outcome_denominator",
    "predicted_net_clear_numerator",
    "predicted_legal_landing_numerator",
    "observed_outcome_denominator",
    "observed_net_clear_numerator",
    "observed_legal_landing_numerator",
    "unresolved_achieved_flight",
    "motion_mimic_pay_eligible",
    "contact_target_pay_eligible",
    "actual_contact_pay_eligible",
    "predicted_outcome_pay_eligible",
    "observed_outcome_pay_eligible",
)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise N1ScalarRewardError("%s must be a finite scalar" % name)
    result = float(value)
    if not math.isfinite(result):
        raise N1ScalarRewardError("%s must be a finite scalar" % name)
    return result


def _plain_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise N1ScalarRewardError("%s must be bool" % name)
    return value


def _vector3(value: object, name: str) -> Vector3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise N1ScalarRewardError("%s must contain exactly three values" % name)
    return tuple(
        _finite_scalar(item, "%s[%d]" % (name, index))
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _validate_spec(spec: N1ScalarRewardSpec) -> N1ScalarRewardSpec:
    if type(spec) is not N1ScalarRewardSpec:
        raise N1ScalarRewardError("spec must be N1ScalarRewardSpec")
    scale = _finite_scalar(spec.miss_distance_scale_m, "miss_distance_scale_m")
    miss_weight = _finite_scalar(
        spec.miss_proximity_weight, "miss_proximity_weight"
    )
    legal_reward = _finite_scalar(spec.legal_landing_reward, "legal_landing_reward")
    out_reward = _finite_scalar(
        spec.opponent_side_out_reward, "opponent_side_out_reward"
    )
    if scale <= 0.0:
        raise N1ScalarRewardError("miss_distance_scale_m must be positive")
    if min(miss_weight, legal_reward, out_reward) < 0.0:
        raise N1ScalarRewardError("diagnostic reward magnitudes must be non-negative")
    if legal_reward <= 0.0:
        raise N1ScalarRewardError("legal_landing_reward must be positive")
    if out_reward > 0.5:
        raise N1ScalarRewardError("opponent_side_out_reward must be at most 0.5")
    if out_reward > 0.5 * legal_reward:
        raise N1ScalarRewardError(
            "opponent_side_out_reward must be at most half legal_landing_reward"
        )
    return spec


def _validate_latch(latch: N1ScalarRewardLatch) -> N1ScalarRewardLatch:
    if type(latch) is not N1ScalarRewardLatch:
        raise N1ScalarRewardError("latch must be N1ScalarRewardLatch")
    _plain_bool(
        latch.observed_outcome_paid,
        "latch.observed_outcome_paid",
    )
    return latch


def _validate_eligibility(value: N1RewardEligibility) -> N1RewardEligibility:
    if type(value) is not N1RewardEligibility:
        raise N1ScalarRewardError(
            "event_eligibility must be N1RewardEligibility"
        )
    for name in _ELIGIBILITY_BOOL_FIELDS:
        _plain_bool(getattr(value, name), "event_eligibility.%s" % name)

    contact = value.actual_contact_pay_eligible
    flight = value.achieved_outgoing_flight_denominator
    observed = value.observed_outcome_denominator
    net_clear = value.observed_net_clear_numerator
    legal = value.observed_legal_landing_numerator
    if flight and not contact:
        raise N1ScalarRewardError(
            "achieved outgoing flight requires actual selected-rubber contact"
        )
    if observed and not flight:
        raise N1ScalarRewardError(
            "observed outcome requires valid achieved outgoing flight"
        )
    if net_clear and not observed:
        raise N1ScalarRewardError("observed net clear requires resolved outcome")
    if legal and not net_clear:
        raise N1ScalarRewardError("observed legal landing requires observed net clear")
    if value.observed_outcome_pay_eligible is not legal:
        raise N1ScalarRewardError(
            "observed outcome pay eligibility must equal observed legal landing"
        )
    if value.unresolved_achieved_flight and (
        not flight or observed or not value.closed_swing_denominator
    ):
        raise N1ScalarRewardError(
            "unresolved achieved flight requires a closed unresolved valid flight"
        )
    return value


def evaluate_n1_c_lite_scalar_reward(
    sample: N1ScalarRewardInput,
    *,
    spec: N1ScalarRewardSpec = N1ScalarRewardSpec(),
) -> N1ScalarRewardOutput:
    """Evaluate one finite C-lite diagnostic scalar and its next event latch.

    Motion, balance, and the eligible strike-distance kernel are per-call
    terms.  Actual selected-rubber contact never pays a separate bonus; it is
    only a prerequisite for achieved flight and the first resolved observed
    outcome, which pays at most once per caller-owned swing latch.
    """

    if type(sample) is not N1ScalarRewardInput:
        raise N1ScalarRewardError("sample must be N1ScalarRewardInput")
    spec = _validate_spec(spec)
    motion = _finite_scalar(sample.motion_reward, "motion_reward")
    balance = _finite_scalar(sample.balance_reward, "balance_reward")
    miss_eligible = _plain_bool(
        sample.miss_sample_eligible, "miss_sample_eligible"
    )
    opponent_out = _plain_bool(
        sample.observed_opponent_side_out,
        "observed_opponent_side_out",
    )
    event = _validate_eligibility(sample.event_eligibility)
    latch = _validate_latch(sample.latch)

    proximity = 0.0
    centres_present = (
        sample.selected_rubber_center_w_m is not None
        or sample.ball_center_w_m is not None
    )
    if miss_eligible or centres_present:
        paddle = _vector3(
            sample.selected_rubber_center_w_m,
            "selected_rubber_center_w_m",
        )
        ball = _vector3(sample.ball_center_w_m, "ball_center_w_m")
    if miss_eligible:
        distance = math.sqrt(
            math.fsum((paddle[index] - ball[index]) ** 2 for index in range(3))
        )
        ratio = distance / float(spec.miss_distance_scale_m)
        proximity = float(spec.miss_proximity_weight) / (1.0 + ratio * ratio)
        if not math.isfinite(proximity):
            raise N1ScalarRewardError("miss proximity reward is not finite")

    observed = event.observed_outcome_denominator
    legal = event.observed_legal_landing_numerator
    if opponent_out and (
        not observed or not event.observed_net_clear_numerator or legal
    ):
        raise N1ScalarRewardError(
            "opponent-side out requires a resolved net-clearing non-legal outcome"
        )

    outcome_now = observed and not latch.observed_outcome_paid
    if outcome_now and not event.actual_contact_pay_eligible:
        raise N1ScalarRewardError(
            "observed outcome cannot pay without selected-rubber contact"
        )

    legal_term = float(spec.legal_landing_reward) if outcome_now and legal else 0.0
    out_term = (
        float(spec.opponent_side_out_reward)
        if outcome_now and opponent_out
        else 0.0
    )
    next_latch = N1ScalarRewardLatch(
        observed_outcome_paid=(latch.observed_outcome_paid or observed),
    )
    total = math.fsum((motion, balance, proximity, legal_term, out_term))
    if not math.isfinite(total):
        raise N1ScalarRewardError("total reward is not finite")

    result = N1ScalarRewardOutput(
        motion_reward=motion,
        balance_reward=balance,
        miss_proximity_reward=proximity,
        observed_legal_landing_reward=legal_term,
        observed_opponent_side_out_reward=out_term,
        total_reward=total,
        observed_outcome_paid_now=outcome_now,
        next_latch=next_latch,
    )
    if not math.isfinite(result.term_sum()) or result.term_sum() != total:
        raise N1ScalarRewardError("per-term scalar reward does not close")
    return result


__all__ = [
    "N1ScalarRewardError",
    "N1ScalarRewardInput",
    "N1ScalarRewardLatch",
    "N1ScalarRewardOutput",
    "N1ScalarRewardSpec",
    "evaluate_n1_c_lite_scalar_reward",
]
