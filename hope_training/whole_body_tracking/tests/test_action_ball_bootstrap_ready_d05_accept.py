"""Fresh first-task exposure must not depend on R07 recovery readiness."""

from __future__ import annotations

import types

import torch

import test_action_ball_continuous_recovery_live_facts as live
import test_action_ball_motion_rowwise_accept_writer as rowwise


D05 = rowwise.D05
EPOCH = D05._require_action_epoch_module()


def _arm_production_d05_from_fresh_motion(
    motion,
    epoch_owner,
    d05_owner: D05.DeviceR05Owner,
):
    """Build one full-N D05 row from Motion's retained reveal fact."""

    device = torch.device(motion.device)
    n = motion.num_envs
    ready_at_reveal = motion._action_ball_continuous_ready_at_reveal.clone()
    due = motion._action_ball_continuous_reveal_due.clone()
    assert due.tolist() == [True, True]
    assert ready_at_reveal.tolist() == [True, True]

    candidate = rowwise._candidate(
        n,
        device,
        candidate_valid=torch.ones(n, dtype=torch.bool, device=device),
    )
    key = candidate.identity.shot_key
    key.reset_generation[:, 0].copy_(motion._action_ball_reset_generation)
    key.ball_generation[:, 0].copy_(motion._action_ball_swing_generation)
    key.action_slot[:, 0].copy_(motion.clip_id)
    key.action_uid[:, 0].copy_(
        torch.as_tensor(
            motion._action_ball_action_uids,
            dtype=torch.int64,
            device=device,
        )[motion.clip_id]
    )
    candidate.playback_admissible[:, 0].copy_(ready_at_reveal)

    prepared = types.SimpleNamespace(
        owner_fault_free=torch.ones(n, dtype=torch.bool, device=device),
        admissible=due.clone(),
        projection=types.SimpleNamespace(ready_at_reveal=ready_at_reveal),
        selected_target_xy_m=torch.arange(
            n * 2, dtype=torch.float32, device=device
        ).reshape(n, 2),
    )
    rows = EPOCH.ActionEpochDueRows(
        common_step=2,
        due_mask=due,
        construct_mask=due.clone(),
    )
    token = object.__new__(D05.DeviceR05RowTransaction)
    d05_owner._action_epoch_candidate = lambda _prepared: candidate
    record = d05_owner._build_row_transaction(
        token,
        rows,
        prepared,
        types.SimpleNamespace(prepared=prepared),
    )
    assert record.accept_mask.tolist() == [True, True]
    assert record.candidate.playback_admissible[:, 0].tolist() == [True, True]
    record.stage = "settling"
    d05_owner._row_transaction_records[token] = record
    d05_owner._active_row_transaction = token

    racket_peer = rowwise._RealD05AcceptedPeer(d05_owner, "racket")
    physical_peer = rowwise._RealD05AcceptedPeer(d05_owner, "physical_ball")
    epoch_owner._active_d05 = EPOCH._ActiveD05Transaction(
        rows=rows,
        publication_ordinal=17,
        base_version=epoch_owner.current().version,
    )
    epoch_owner._d05_owner = d05_owner
    epoch_owner._d05_candidate_projector = (
        d05_owner.require_owned_action_epoch_candidate
    )
    epoch_owner._d05_accept_writers = (
        motion.commit_action_ball_full_mdp_motion_epoch_rows,
        racket_peer.commit,
        physical_peer.commit,
    )
    return token, record


def test_fresh_first_d05_accepts_without_r07_publication_or_install(
    monkeypatch,
) -> None:
    device = torch.device("cpu")
    env, motion, _robot, _sensor = live._subject(monkeypatch, device=device)

    epoch_owner = motion._diagnostic_test_epoch_owner
    d05_owner = motion._diagnostic_test_device_r05_owner
    assert motion._action_ball_continuous_fresh_motion_lane_bound is True
    assert motion._action_ball_continuous_r07_ready_owner is None
    assert motion._action_ball_continuous_r07_ready_projection is None
    # This is the legacy ready source.  Fresh training must ignore even an
    # explicitly false value instead of requiring an R07 bootstrap verdict.
    motion._action_ball_continuous_ready_authority = torch.zeros(
        motion.num_envs, dtype=torch.bool, device=device
    )
    env.common_step_counter = 2
    motion._advance_action_ball_continuous_motion_cadence()
    assert motion._action_ball_continuous_r07_ready_projection is None
    assert motion._action_ball_continuous_ready_at_reveal.tolist() == [True, True]

    token, d05_record = _arm_production_d05_from_fresh_motion(
        motion,
        epoch_owner,
        d05_owner,
    )
    epoch_owner.settle_d05_transaction(token)
    settled = epoch_owner.current()

    d05_entry = next(
        entry
        for entry in reversed(epoch_owner._publication.pending_log)
        if entry.transition == "D05_SETTLED"
    )
    decision = d05_entry.delta.values[
        d05_entry.delta.names.index("decision")
    ][:, 0]
    assert (
        epoch_owner._undecoded_overflow.tolist(),
        decision.tolist(),
        settled.phase[:, 0].tolist(),
    ) == (
        [False, False],
        [EPOCH.D05_DECISION_ACCEPT, EPOCH.D05_DECISION_ACCEPT],
        [EPOCH.PHASE_REVEAL_COMMITTED, EPOCH.PHASE_REVEAL_COMMITTED],
    )
    assert d05_record.accepted_consumers == {
        "motion",
        "racket",
        "physical_ball",
    }
