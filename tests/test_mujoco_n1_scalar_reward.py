"""Truth-table tests for the pure native-N1 C-lite scalar reward."""

from __future__ import annotations

import ast
import dataclasses
import math
import sys
from pathlib import Path

import pytest

WBT_ROOT = Path(__file__).resolve().parents[1] / "hope_training/whole_body_tracking"
sys.path.insert(0, str(WBT_ROOT))

from mujoco_native import n1_reward_event_kernel as events  # noqa: E402
from mujoco_native import n1_scalar_reward as reward  # noqa: E402


MODULE_PATH = Path(reward.__file__)


def _eligibility(
    *,
    contact=False,
    flight=False,
    observed=False,
    net_clear=False,
    legal=False,
):
    return events.N1RewardEligibility(
        motion_mimic_denominator=True,
        contact_target_denominator=False,
        closed_swing_denominator=observed,
        actual_contact_numerator=observed and contact,
        achieved_outgoing_flight_denominator=flight,
        predicted_outcome_denominator=False,
        predicted_net_clear_numerator=False,
        predicted_legal_landing_numerator=False,
        observed_outcome_denominator=observed,
        observed_net_clear_numerator=net_clear,
        observed_legal_landing_numerator=legal,
        unresolved_achieved_flight=False,
        motion_mimic_pay_eligible=True,
        contact_target_pay_eligible=False,
        actual_contact_pay_eligible=contact,
        predicted_outcome_pay_eligible=False,
        observed_outcome_pay_eligible=legal,
    )


def _sample(
    *,
    eligibility=None,
    opponent_out=False,
    latch=reward.N1ScalarRewardLatch(),
    distance=0.15,
    miss_eligible=True,
    motion=0.2,
    balance=-0.1,
):
    return reward.N1ScalarRewardInput(
        motion_reward=motion,
        balance_reward=balance,
        miss_sample_eligible=miss_eligible,
        selected_rubber_center_w_m=(distance, 0.0, 0.0),
        ball_center_w_m=(0.0, 0.0, 0.0),
        event_eligibility=eligibility or _eligibility(),
        observed_opponent_side_out=opponent_out,
        latch=latch,
    )


@pytest.mark.parametrize(
    "eligibility,opponent_out,expected_legal,expected_out",
    (
        (_eligibility(), False, 0.0, 0.0),
        (_eligibility(contact=True), False, 0.0, 0.0),
        (
            _eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=True,
                legal=True,
            ),
            False,
            1.0,
            0.0,
        ),
        (
            _eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=True,
                legal=False,
            ),
            True,
            0.0,
            0.5,
        ),
        (
            _eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=False,
                legal=False,
            ),
            False,
            0.0,
            0.0,
        ),
    ),
)
def test_reward_truth_table(
    eligibility,
    opponent_out,
    expected_legal,
    expected_out,
):
    result = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(eligibility=eligibility, opponent_out=opponent_out)
    )
    assert result.miss_proximity_reward == pytest.approx(0.5)
    assert result.observed_legal_landing_reward == expected_legal
    assert result.observed_opponent_side_out_reward == expected_out
    assert result.observed_opponent_side_out_reward <= 0.5
    assert result.total_reward == result.term_sum()


def test_cauchy_miss_signal_is_positive_and_strictly_decreases_with_distance():
    distances = (0.0, 0.05, 0.15, 0.30, 1.0, 100.0)
    values = [
        reward.evaluate_n1_c_lite_scalar_reward(
            _sample(distance=distance, motion=0.0, balance=0.0)
        ).miss_proximity_reward
        for distance in distances
    ]
    assert values[0] == 1.0
    assert all(left > right > 0.0 for left, right in zip(values, values[1:]))
    assert values[2] == pytest.approx(0.5)


def test_default_term_scale_has_no_contact_bonus_and_out_is_half_legal():
    spec = reward.N1ScalarRewardSpec()
    assert spec.miss_distance_scale_m == 0.15
    assert spec.miss_proximity_weight == 1.0
    assert spec.legal_landing_reward == 1.0
    assert spec.opponent_side_out_reward == 0.5

    contact_only = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=_eligibility(contact=True),
            miss_eligible=False,
            motion=0.0,
            balance=0.0,
        )
    )
    legal = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=_eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=True,
                legal=True,
            ),
            miss_eligible=False,
            motion=0.0,
            balance=0.0,
        )
    )
    opponent_out = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=_eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=True,
                legal=False,
            ),
            opponent_out=True,
            miss_eligible=False,
            motion=0.0,
            balance=0.0,
        )
    )
    assert contact_only.total_reward == 0.0
    assert legal.total_reward == 1.0
    assert opponent_out.total_reward == 0.5 * legal.total_reward
    assert not hasattr(contact_only, "selected_rubber_contact_reward")


def test_event_terms_pay_once_and_latch_is_explicit_input_output():
    eligibility = _eligibility(
        contact=True,
        flight=True,
        observed=True,
        net_clear=True,
        legal=True,
    )
    first = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(eligibility=eligibility, miss_eligible=False)
    )
    assert first.observed_outcome_paid_now is True
    assert first.next_latch == reward.N1ScalarRewardLatch(True)

    second = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=eligibility,
            latch=first.next_latch,
            miss_eligible=False,
        )
    )
    assert second.observed_outcome_paid_now is False
    assert second.observed_legal_landing_reward == 0.0
    assert second.total_reward == pytest.approx(0.1)


def test_contact_never_pays_and_only_later_observed_outcome_is_latched():
    contact = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=_eligibility(contact=True),
            miss_eligible=False,
            motion=0.0,
            balance=0.0,
        )
    )
    assert contact.total_reward == 0.0
    assert contact.observed_outcome_paid_now is False
    assert contact.next_latch == reward.N1ScalarRewardLatch(False)

    landing = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=_eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=True,
                legal=True,
            ),
            latch=contact.next_latch,
            miss_eligible=False,
            motion=0.0,
            balance=0.0,
        )
    )
    assert landing.observed_legal_landing_reward == 1.0
    assert landing.next_latch == reward.N1ScalarRewardLatch(True)


@pytest.mark.parametrize(
    "sample,match",
    (
        (
            _sample(motion=float("nan")),
            "motion_reward",
        ),
        (
            dataclasses.replace(
                _sample(), selected_rubber_center_w_m=(float("inf"), 0.0, 0.0)
            ),
            "selected_rubber_center",
        ),
        (
            _sample(
                eligibility=_eligibility(
                    contact=True,
                    flight=True,
                    observed=True,
                    net_clear=False,
                ),
                opponent_out=True,
            ),
            "opponent-side out",
        ),
    ),
)
def test_nonfinite_and_contradictory_inputs_fail_closed(sample, match):
    with pytest.raises(reward.N1ScalarRewardError, match=match):
        reward.evaluate_n1_c_lite_scalar_reward(sample)


@pytest.mark.parametrize(
    "spec,match",
    (
        (
            reward.N1ScalarRewardSpec(miss_distance_scale_m=0.0),
            "scale",
        ),
        (
            reward.N1ScalarRewardSpec(opponent_side_out_reward=0.500001),
            "at most 0.5",
        ),
        (
            reward.N1ScalarRewardSpec(legal_landing_reward=0.5),
            "at most half",
        ),
        (
            reward.N1ScalarRewardSpec(miss_proximity_weight=float("nan")),
            "finite",
        ),
    ),
)
def test_invalid_reward_spec_fails_closed(spec, match):
    with pytest.raises(reward.N1ScalarRewardError, match=match):
        reward.evaluate_n1_c_lite_scalar_reward(_sample(), spec=spec)


def test_all_reported_terms_and_total_are_finite_and_close():
    result = reward.evaluate_n1_c_lite_scalar_reward(
        _sample(
            eligibility=_eligibility(
                contact=True,
                flight=True,
                observed=True,
                net_clear=True,
                legal=True,
            )
        )
    )
    terms = (
        result.motion_reward,
        result.balance_reward,
        result.miss_proximity_reward,
        result.observed_legal_landing_reward,
        result.observed_opponent_side_out_reward,
    )
    assert all(math.isfinite(value) for value in terms + (result.total_reward,))
    assert result.total_reward == math.fsum(terms) == result.term_sum()


def test_source_has_no_desired_contact_or_inverse_physics_path():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    forbidden = {
        "desired_at_contact",
        "desired_contact",
        "racket_target_pos_w",
        "racket_target_vel_w",
        "racket_target_normal_w",
        "inverse_solve",
        "predict_paddle_contact",
        "predict_flight",
    }
    assert not forbidden & (names | attributes)
