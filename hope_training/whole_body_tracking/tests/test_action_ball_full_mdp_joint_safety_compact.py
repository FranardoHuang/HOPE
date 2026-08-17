"""Direct producer counterexamples for fresh full-MDP compact safety evidence."""

from __future__ import annotations

import pytest
import torch

from test_joint_limit_safety import (
    _action_and_env,
    _finish_guarded_policy_step,
)


def _full_mdp_action(*, compact: bool, capacity: int):
    return _action_and_env(
        guard=True,
        guard_policy_dt_s=0.1,
        runtime_step_dt=0.1,
        terminal_archive_capacity=capacity,
        target_mode="action_ball_full_mdp",
        action_ball_diagnostic_unauthorized=True,
        diagnostic_compact_evidence=compact,
    )


def test_noncompact_full_mdp_overflows_on_real_third_summary_at_capacity_two():
    action, _env, asset = _full_mdp_action(compact=False, capacity=2)
    zeros = torch.zeros(2, 2)

    for _ in range(2):
        action.process_actions(zeros)
        _finish_guarded_policy_step(action, asset)
    with pytest.raises(RuntimeError, match="policy-step summary overflow"):
        action.process_actions(zeros)
        _finish_guarded_policy_step(action, asset)

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["policy_step_summary_capacity"] == 2
    assert snapshot["policy_step_summary_used"] == 2
    assert snapshot["policy_step_summary_overflow_latch"] is True
    assert snapshot["policy_step_summary_overflow_count"] == 1
    with pytest.raises(RuntimeError, match="overflow is sticky"):
        action.process_actions(zeros)


def test_compact_full_mdp_crosses_old_4096_boundary_without_dense_archive():
    action, _env, asset = _full_mdp_action(compact=True, capacity=2)
    zeros = torch.zeros(2, 2)

    for _ in range(4097):
        action.process_actions(zeros)
        _finish_guarded_policy_step(action, asset)

    snapshot = action.joint_safety_ledger_snapshot()
    assert action._joint_safety_diagnostic_compact_evidence is True
    assert snapshot["identity_bound_policy_steps"] == ()
    assert snapshot["terminal_archives"] == ()
    for prefix in ("policy_step_summary", "terminal_archive"):
        assert snapshot[f"{prefix}_used"] == 0
        assert snapshot[f"{prefix}_overflow_latch"] is False
        assert snapshot[f"{prefix}_overflow_count"] == 0
    assert snapshot["since_last_consume"]["policy_step_count"].tolist() == [
        4097,
        4097,
    ]

    token, prepared = action.prepare_joint_safety_ledger_consume()
    assert prepared["since_last_consume"]["has_data"] is True
    action.acknowledge_joint_safety_ledger(token)
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"][
            "has_data"
        ]
        is False
    )
