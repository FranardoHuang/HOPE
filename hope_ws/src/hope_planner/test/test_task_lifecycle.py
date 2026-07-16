import pytest

from hope_planner.flat_command_wire import MAX_EXACT_FLOAT64_INTEGER
from hope_planner.task_lifecycle import (
    FormalBallTrackBoundary,
    FormalBallTrackDecision,
    FormalTaskCounterExhaustion,
    FormalTaskEpochError,
    FormalTaskLifecycle,
    FormalTaskState,
    FormalTaskTransitionError,
)


def test_ball_track_requires_gap_or_inbound_discontinuity_to_rearm():
    boundary = FormalBallTrackBoundary(no_ball_rearm_s=0.10)
    assert boundary.observe_absent(1.00, task_active=False) is FormalBallTrackDecision.NONE
    assert boundary.observe_absent(1.11, task_active=False) is FormalBallTrackDecision.NONE
    # A visible outbound ball cannot consume the completed gap proof.
    assert boundary.observe_present(
        1.12,
        ball_x_m=0.5,
        ball_vx_mps=1.0,
        discontinuity_detected=False,
        task_active=False,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=None,
    ) is FormalBallTrackDecision.NONE
    assert boundary.observe_present(
        1.13,
        ball_x_m=0.5,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=False,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=1.50,
    ) is FormalBallTrackDecision.SAFE_REARM

    fresh = FormalBallTrackBoundary()
    assert fresh.observe_present(
        2.0,
        ball_x_m=0.8,
        ball_vx_mps=-2.0,
        discontinuity_detected=True,
        task_active=False,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=2.5,
    ) is FormalBallTrackDecision.SAFE_REARM


def test_active_task_gap_then_new_inbound_closes_and_rearms_atomically():
    boundary = FormalBallTrackBoundary(no_ball_rearm_s=0.10)
    assert boundary.observe_present(
        1.0,
        ball_x_m=0.5,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=2.0,
    ) is FormalBallTrackDecision.NONE
    assert boundary.observe_absent(
        1.01, task_active=True
    ) is FormalBallTrackDecision.NONE
    assert boundary.observe_absent(
        1.12, task_active=True
    ) is FormalBallTrackDecision.NONE

    # The old deadline has not elapsed and the new ball is still before the
    # plane.  The completed no-ball proof nevertheless owns the physical task
    # boundary; returning NONE here would merge two balls into one task id.
    assert boundary.observe_present(
        1.13,
        ball_x_m=0.8,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=1.8,
    ) is FormalBallTrackDecision.CLOSE_AND_REARM


def test_active_task_short_gap_and_inbound_noise_do_not_split_ball():
    boundary = FormalBallTrackBoundary(no_ball_rearm_s=0.10)
    boundary.observe_present(
        1.0,
        ball_x_m=0.5,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=2.0,
    )
    assert boundary.observe_absent(
        1.01, task_active=True
    ) is FormalBallTrackDecision.NONE
    assert boundary.observe_absent(
        1.05, task_active=True
    ) is FormalBallTrackDecision.NONE
    assert boundary.observe_present(
        1.06,
        ball_x_m=0.45,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=1.9,
    ) is FormalBallTrackDecision.NONE


def test_ball_track_closes_after_plane_deadline_or_outbound_contact():
    boundary = FormalBallTrackBoundary(
        plane_close_margin_m=0.02, deadline_close_grace_s=0.08
    )
    assert boundary.observe_present(
        3.0,
        ball_x_m=0.5,
        ball_vx_mps=-2.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=3.4,
    ) is FormalBallTrackDecision.NONE
    assert boundary.observe_present(
        3.3,
        ball_x_m=-0.03,
        ball_vx_mps=-2.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=3.4,
    ) is FormalBallTrackDecision.CLOSE_ACTIVE

    deadline = FormalBallTrackBoundary(deadline_close_grace_s=0.05)
    deadline.observe_present(
        4.0,
        ball_x_m=0.4,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=4.2,
    )
    assert deadline.observe_absent(
        4.26, task_active=True
    ) is FormalBallTrackDecision.CLOSE_ACTIVE

    outbound = FormalBallTrackBoundary()
    assert outbound.observe_present(
        5.0,
        ball_x_m=0.1,
        ball_vx_mps=2.0,
        discontinuity_detected=True,
        task_active=True,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=None,
    ) is FormalBallTrackDecision.CLOSE_ACTIVE


def test_ball_track_epoch_reset_discards_old_gap_and_deadline_proofs():
    boundary = FormalBallTrackBoundary(no_ball_rearm_s=0.0)
    boundary.observe_absent(1.0, task_active=False)
    boundary.observe_present(
        1.1,
        ball_x_m=0.5,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=False,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=1.5,
    )
    boundary.reset_epoch()
    assert boundary.observe_present(
        1.2,
        ball_x_m=0.4,
        ball_vx_mps=-1.0,
        discontinuity_detected=False,
        task_active=False,
        strike_plane_x_m=0.0,
        predicted_strike_time_s=1.5,
    ) is FormalBallTrackDecision.NONE


def armed_lifecycle(epoch: int = 7) -> FormalTaskLifecycle:
    lifecycle = FormalTaskLifecycle()
    lifecycle.explicit_rearm(
        epoch, no_ball_or_new_serve_confirmed=True
    )
    return lifecycle


def test_restart_stays_disarmed_until_explicit_safe_rearm():
    lifecycle = FormalTaskLifecycle()
    assert lifecycle.publish(
        7, inbound_track_ready=True, solver_valid=True
    ) is None
    assert lifecycle.state is FormalTaskState.DISARMED
    assert lifecycle.last_task_id == 0

    with pytest.raises(FormalTaskTransitionError, match="confirmed"):
        lifecycle.explicit_rearm(
            7, no_ball_or_new_serve_confirmed=False
        )
    assert lifecycle.state is FormalTaskState.DISARMED

    lifecycle.explicit_rearm(7, no_ball_or_new_serve_confirmed=True)
    assert lifecycle.state is FormalTaskState.ARMED
    assert lifecycle.publish(
        7, inbound_track_ready=False, solver_valid=True
    ) is None


def test_same_ball_valid_invalid_valid_keeps_task_and_advances_revision():
    lifecycle = armed_lifecycle()
    valid1 = lifecycle.publish(
        7, inbound_track_ready=True, solver_valid=True
    )
    invalid = lifecycle.publish(
        7, inbound_track_ready=False, solver_valid=False
    )
    valid2 = lifecycle.publish(
        7, inbound_track_ready=True, solver_valid=True
    )

    assert valid1 is not None and invalid is not None and valid2 is not None
    assert [row.task_id for row in (valid1, invalid, valid2)] == [1, 1, 1]
    assert [row.task_revision for row in (valid1, invalid, valid2)] == [1, 2, 3]
    assert [row.valid for row in (valid1, invalid, valid2)] == [True, False, True]
    assert lifecycle.state is FormalTaskState.ACTIVE


def test_active_inbound_predicate_jitter_never_splits_one_task():
    lifecycle = armed_lifecycle()
    rows = [
        lifecycle.publish(7, inbound_track_ready=inbound, solver_valid=True)
        for inbound in (True, False, True, False, False, True)
    ]
    assert all(row is not None for row in rows)
    assert [row.task_id for row in rows if row is not None] == [1] * 6
    assert [row.task_revision for row in rows if row is not None] == list(range(1, 7))


def test_close_then_explicit_rearm_is_only_path_to_next_task():
    lifecycle = armed_lifecycle()
    first = lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)
    assert first is not None
    terminal = lifecycle.close(7)
    assert terminal is not None
    assert (terminal.task_id, terminal.task_revision, terminal.valid) == (1, 2, False)
    assert lifecycle.state is FormalTaskState.CLOSED_WAIT_REARM

    # A new inbound track cannot allocate while the close barrier is latched.
    assert lifecycle.publish(
        7, inbound_track_ready=True, solver_valid=True
    ) is None
    with pytest.raises(FormalTaskTransitionError, match="confirmed"):
        lifecycle.explicit_rearm(
            7, no_ball_or_new_serve_confirmed=False
        )
    assert lifecycle.publish(
        7, inbound_track_ready=True, solver_valid=True
    ) is None

    lifecycle.explicit_rearm(7, no_ball_or_new_serve_confirmed=True)
    second = lifecycle.publish(7, inbound_track_ready=True, solver_valid=False)
    assert second is not None
    assert (second.task_id, second.task_revision, second.valid) == (2, 1, False)


def test_duplicate_close_and_rearm_while_active_fail_closed():
    lifecycle = armed_lifecycle()
    lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)
    with pytest.raises(FormalTaskTransitionError, match="active"):
        lifecycle.explicit_rearm(7, no_ball_or_new_serve_confirmed=True)
    lifecycle.close(7)
    with pytest.raises(FormalTaskTransitionError, match="active"):
        lifecycle.close(7)


def test_epoch_change_disarms_and_requires_a_new_explicit_barrier():
    lifecycle = armed_lifecycle(epoch=7)
    before = lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)
    assert before is not None and before.task_id == 1

    # The first row carrying epoch 8 cannot itself rearm or allocate.
    assert lifecycle.publish(
        8, inbound_track_ready=True, solver_valid=True
    ) is None
    assert lifecycle.control_epoch == 8
    assert lifecycle.state is FormalTaskState.DISARMED
    assert lifecycle.last_task_id == 0
    assert lifecycle.publish(
        8, inbound_track_ready=True, solver_valid=True
    ) is None

    lifecycle.explicit_rearm(8, no_ball_or_new_serve_confirmed=True)
    after = lifecycle.publish(8, inbound_track_ready=True, solver_valid=True)
    assert after is not None
    assert (after.control_epoch, after.task_id, after.task_revision) == (8, 1, 1)
    with pytest.raises(FormalTaskEpochError, match="regressed"):
        lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)


def test_close_at_new_epoch_does_not_fabricate_old_task_terminal_row():
    lifecycle = armed_lifecycle(epoch=7)
    lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)
    assert lifecycle.close(8) is None
    assert lifecycle.control_epoch == 8
    assert lifecycle.state is FormalTaskState.DISARMED


def test_malformed_new_epoch_event_still_disarms_old_active_task():
    lifecycle = armed_lifecycle(epoch=7)
    lifecycle.publish(7, inbound_track_ready=True, solver_valid=True)
    with pytest.raises(TypeError, match="exact boolean"):
        lifecycle.publish(8, inbound_track_ready=1, solver_valid=True)
    assert lifecycle.control_epoch == 8
    assert lifecycle.state is FormalTaskState.DISARMED
    assert lifecycle.active_task_id is None

    lifecycle.explicit_rearm(8, no_ball_or_new_serve_confirmed=True)
    lifecycle.publish(8, inbound_track_ready=True, solver_valid=True)
    with pytest.raises(TypeError, match="exact boolean"):
        lifecycle.explicit_rearm(
            9, no_ball_or_new_serve_confirmed=1
        )
    assert lifecycle.control_epoch == 9
    assert lifecycle.state is FormalTaskState.DISARMED


@pytest.mark.parametrize("bad", [1, 0, None, "true"])
def test_lifecycle_requires_exact_boolean_events(bad):
    lifecycle = armed_lifecycle()
    with pytest.raises(TypeError, match="exact boolean"):
        lifecycle.publish(7, inbound_track_ready=bad, solver_valid=True)
    with pytest.raises(TypeError, match="exact boolean"):
        lifecycle.publish(7, inbound_track_ready=True, solver_valid=bad)


@pytest.mark.parametrize(
    "bad",
    [True, -1, 1.5, float("nan"), MAX_EXACT_FLOAT64_INTEGER + 1],
)
def test_lifecycle_rejects_nonexact_epoch(bad):
    lifecycle = FormalTaskLifecycle()
    with pytest.raises(FormalTaskEpochError):
        lifecycle.publish(bad, inbound_track_ready=True, solver_valid=True)


def test_task_id_and_revision_exhaustion_permanently_disarm_current_epoch():
    task_exhausted = FormalTaskLifecycle(
        control_epoch=7,
        state=FormalTaskState.ARMED,
        last_task_id=MAX_EXACT_FLOAT64_INTEGER,
    )
    with pytest.raises(FormalTaskCounterExhaustion, match="task_id"):
        task_exhausted.publish(7, inbound_track_ready=True, solver_valid=True)
    assert task_exhausted.state is FormalTaskState.DISARMED
    assert task_exhausted.exhausted
    with pytest.raises(FormalTaskCounterExhaustion, match="exhausted"):
        task_exhausted.explicit_rearm(
            7, no_ball_or_new_serve_confirmed=True
        )

    revision_exhausted = FormalTaskLifecycle(
        control_epoch=7,
        state=FormalTaskState.ACTIVE,
        last_task_id=1,
        active_task_id=1,
        active_revision=MAX_EXACT_FLOAT64_INTEGER,
    )
    with pytest.raises(FormalTaskCounterExhaustion, match="task_revision"):
        revision_exhausted.publish(
            7, inbound_track_ready=False, solver_valid=False
        )
    assert revision_exhausted.state is FormalTaskState.DISARMED
    assert revision_exhausted.exhausted
