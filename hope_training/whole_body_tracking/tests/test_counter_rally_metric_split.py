"""Counter-rally objective acceptance must not redefine virtual legal returns."""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_metric_sync_fix import DECAY, _make_rally_cmd  # noqa: E402


def _counter_metric_rig():
    clip_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    command = _make_rally_cmd(4, clip_ids=clip_ids)
    command._action_ball_enabled = True
    command._action_ball_attempt_active = torch.ones(4, dtype=torch.bool)
    command._action_ball_attempt_legal = torch.zeros(4, dtype=torch.bool)
    # S0 起 _vb_book_strike_step 除了 L(legal) 还要闩 H(hit), 并且在 A211/C211
    # 的 RESET_WAIT 日程开着时用 task_valid 把隐藏任务挡在 C/H/L/F 分母外面。
    # 本 rig 是老的 counter-rally 臂: 没有 WAIT 日程(=None), 所以 task_valid
    # 这条口径对它是恒等的, 分母语义与 S0 之前逐位一致。
    command._action_ball_attempt_hit = torch.zeros(4, dtype=torch.bool)
    command._action_ball_task_wait_schedule = None
    return command


def test_counter_acceptance_has_its_own_exact_count_and_keeps_landing_metric():
    command = _counter_metric_rig()
    exact = torch.ones(4, dtype=torch.bool)
    legal_first_landing = torch.tensor([True, True, True, False])
    counter_accepted = torch.tensor([True, False, True, False])

    command._vb_book_strike_step(
        DECAY,
        exact,
        exact,
        legal_first_landing,
        legal_first_landing,
        legal_first_landing,
        counter_rally_accepted=counter_accepted,
    )
    snapshot = command.consume_sparse_reward_eligibility_counters()

    assert snapshot["virtual_legal_return_count"].item() == 3
    assert snapshot["counter_rally_accepted_count"].item() == 2
    assert snapshot["virtual_legal_return_count_forehand"].item() == 2
    assert snapshot["counter_rally_accepted_count_forehand"].item() == 1
    assert snapshot["virtual_legal_return_count_backhand"].item() == 1
    assert snapshot["counter_rally_accepted_count_backhand"].item() == 1
    assert command._vb_inb_acc == pytest.approx(3.0)
    assert torch.equal(command._rally_returned, legal_first_landing)
    # Counter acceptance still drives the ActionBall attempt/curriculum latch.
    assert torch.equal(command._action_ball_attempt_legal, counter_accepted)


def test_counter_acceptance_without_legal_first_landing_fails_loud():
    command = _counter_metric_rig()
    exact = torch.ones(4, dtype=torch.bool)
    legal_first_landing = torch.tensor([True, False, False, False])
    counter_accepted = torch.tensor([True, True, False, False])

    with pytest.raises(
        RuntimeError,
        match="cannot occur without a legal first opponent-table landing",
    ):
        command._vb_book_strike_step(
            DECAY,
            exact,
            exact,
            legal_first_landing,
            legal_first_landing,
            legal_first_landing,
            counter_rally_accepted=counter_accepted,
        )
