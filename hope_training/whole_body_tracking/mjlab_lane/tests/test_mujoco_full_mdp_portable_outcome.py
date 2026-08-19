"""Counterexamples for the engine-neutral MuJoCo R06/R07 kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

import mujoco_full_mdp_portable_outcome as outcome


def test_observed_flight_crosses_real_net_then_descending_landing_plane():
    tracked = torch.ones(1, dtype=torch.bool)
    net, clear, landing, _xy, _table, _bound, _opponent = outcome.observe_flight_step(
        torch=torch,
        previous=torch.tensor([[1.70, 0.10, 1.00]]),
        current=torch.tensor([[2.00, 0.10, 0.96]]),
        tracking=tracked,
        target_positive_x=True,
        net_x=1.87,
        net_clear_z=0.93,
        landing_plane_z=0.78,
        table_bounds=(0.50, 3.24, -0.7625, 0.7625),
    )
    assert net.all() and clear.all() and not landing.any()

    net, clear, landing, xy, on_table, opponent_bound, on_opponent = (
        outcome.observe_flight_step(
            torch=torch,
            previous=torch.tensor([[2.40, 0.10, 0.90]]),
            current=torch.tensor([[2.60, 0.10, 0.70]]),
            tracking=tracked,
            target_positive_x=True,
            net_x=1.87,
            net_clear_z=0.93,
            landing_plane_z=0.78,
            table_bounds=(0.50, 3.24, -0.7625, 0.7625),
        )
    )
    assert not net.any() and not clear.any()
    assert landing.all() and on_table.all() and opponent_bound.all() and on_opponent.all()
    torch.testing.assert_close(xy, torch.tensor([[2.52, 0.10]]))


def test_outcome_and_r06_rows_cover_legal_own_out_invalid_and_no_contact():
    active = torch.ones(5, dtype=torch.bool)
    selected = torch.tensor([True, True, True, True, False])
    finite = torch.tensor([True, True, True, False, True])
    landing = torch.tensor([True, True, True, False, False])
    on_table = torch.tensor([True, True, False, False, False])
    on_opponent = torch.tensor([True, False, False, False, False])
    net = torch.tensor([True, False, True, False, False])
    clear = torch.tensor([True, False, True, False, False])
    settled, codes = outcome.classify_outcome(
        torch=torch,
        active=active,
        selected_contact=selected,
        finite=finite,
        landing_present=landing,
        landing_on_table=on_table,
        landing_on_opponent=on_opponent,
        net_crossed=net,
        net_clear=clear,
        dead=torch.zeros(5, dtype=torch.bool),
        expired=torch.tensor([False, False, False, False, True]),
        codes=torch.zeros(5, dtype=torch.long),
    )
    assert settled.all()
    assert torch.equal(codes, torch.tensor([3, 4, 5, 6, 1]))

    crossing_xy = torch.tensor(
        [
            [2.55, 0.00],
            [1.20, 0.10],
            [3.50, 1.00],
            [float("nan"), 0.00],
            [0.00, 0.00],
        ]
    )
    eligible, valid, common, facts = outcome.r06_rows(
        torch=torch,
        settled=settled,
        selected_contact=selected,
        invalid_outcome=codes.eq(6),
        crossing_present=landing,
        crossing_xy=crossing_xy,
        target_xy=torch.tensor([2.55, 0.00]),
        opponent_bound=torch.tensor([True, False, True, False, False]),
        on_opponent=on_opponent,
        net_crossed=net,
        net_clear=clear,
        broad_sigma=0.65,
        narrow_sigma=0.04,
    )
    assert torch.equal(eligible, torch.tensor([True, True, True, True, True]))
    assert torch.equal(valid, torch.tensor([True, True, True, False, True]))
    assert torch.equal(common, torch.tensor([True, False, False, False, False]))
    assert facts[0, 0] == 1.0 and facts[0, 1] == 1.0
    assert facts[1, 1] == 0.0
    assert 0.0 < facts[2, 1] < 1.0
    assert torch.count_nonzero(facts[3, :3]) == 0
    assert facts[4, 2] == 1.0
    assert torch.count_nonzero(facts[4, :2]) == 0
    assert torch.isfinite(facts).all()


def test_recovery_rows_keep_expected_denominator_separate_from_numeric_fault():
    expected = torch.tensor([True, True, False])
    errors = torch.zeros((3, 13))
    errors[1, 4] = float("nan")
    eligible, valid, ready, facts = outcome.r07_rows(
        torch=torch,
        expected=expected,
        age=torch.tensor([10, 10, -1]),
        errors=errors,
        hard_safety_ok=torch.tensor([True, True, True]),
        scales=torch.ones((1, 13)),
        ready_tolerances=torch.ones((1, 13)),
        weight=0.7,
    )
    assert torch.equal(eligible, torch.tensor([True, False, False]))
    assert torch.equal(valid, torch.tensor([True, False, True]))
    assert torch.equal(ready, torch.tensor([True, False, False]))
    torch.testing.assert_close(facts[:, 0], torch.tensor([0.7, 0.0, 0.0]))
    assert torch.equal(facts[:, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(facts[:, 4], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.isfinite(facts).all()


def test_recovery_ready_requires_hard_safety_but_reward_stays_eligible():
    eligible, valid, ready, facts = outcome.r07_rows(
        torch=torch,
        expected=torch.tensor([True]),
        age=torch.tensor([10]),
        errors=torch.zeros((1, 13)),
        hard_safety_ok=torch.tensor([False]),
        scales=torch.ones((1, 13)),
        ready_tolerances=torch.ones((1, 13)),
        weight=0.7,
    )
    assert eligible.item() and valid.item()
    assert not ready.item()
    assert facts[0, 0] == 0.7
    assert facts[0, 5] == 0.0
