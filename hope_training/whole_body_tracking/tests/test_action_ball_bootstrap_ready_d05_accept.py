"""Fresh first-task exposure must not depend on R07 recovery readiness."""

from __future__ import annotations

import torch

import action_ball_full_mdp_portable_catalog as catalog
import test_action_ball_continuous_recovery_live_facts as live


def test_fresh_motion_reveals_at_catalog_tick_without_r07_publication_or_install(
    monkeypatch,
) -> None:
    device = torch.device("cpu")
    env, motion, _robot, _sensor = live._subject(monkeypatch, device=device)

    assert motion._action_ball_continuous_fresh_motion_lane_bound is True
    assert motion._action_ball_continuous_r07_ready_owner is None
    assert motion._action_ball_continuous_r07_ready_projection is None
    # This is the legacy ready source.  Fresh training must ignore even an
    # explicitly false value instead of requiring an R07 bootstrap verdict.
    motion._action_ball_continuous_ready_authority = torch.zeros(
        motion.num_envs, dtype=torch.bool, device=device
    )
    first_reveal_step = int(
        motion._action_ball_continuous_schedule_projection[
            "first_reveal_step"
        ]
    )
    assert first_reveal_step == catalog.FRESH_FIRST_REVEAL_TICK == 295
    # This test isolates the curriculum seam rather than replaying the full
    # balance prefix.  Motion still owns both the due tick and its transition.
    motion._action_ball_continuous_episode_step.fill_(first_reveal_step - 1)
    env.common_step_counter = first_reveal_step
    motion._advance_action_ball_continuous_motion_cadence()
    assert motion._action_ball_continuous_r07_ready_projection is None
    assert motion._action_ball_continuous_reveal_due.tolist() == [True, True]
    assert motion._action_ball_continuous_ready_at_reveal.tolist() == [True, True]
