"""Focused counterexamples for the one-pass row-wise epoch drain."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (str(SOURCE), str(MDP)):
    if path not in sys.path:
        sys.path.insert(0, path)

import action_ball_full_mdp_drain_summary as D  # noqa: E402
import action_ball_full_mdp_epoch as E  # noqa: E402


SHAPE = (2, 1)


def _milestone():
    return (
        torch.zeros(D.milestone_tensors.I64_NUMEL, dtype=torch.int64),
        torch.zeros(D.milestone_tensors.F64_NUMEL, dtype=torch.float64),
    )


def _key(*, valid_rows=(), seed=10_000):
    values = {}
    for offset, name in enumerate(D.SHOT_KEY_FIELDS):
        tensor = torch.full(SHAPE, -1, dtype=torch.int64)
        for row in valid_rows:
            tensor[row, 0] = seed + 100 * row + offset
        values[name] = tensor
    return values


def _event(mask, key):
    return (mask, *(key[name] for name in D.SHOT_KEY_FIELDS))


def _entry(sequence, transition, names, values, *, epoch=0):
    return E.CommitEntry(
        sequence=sequence,
        epoch=epoch,
        transition=transition,
        before_version=sequence - 1,
        after_version=sequence,
        delta=E.PackedDelta(names=names, values=values),
    )


def _selected_reset_entry(
    sequence,
    *,
    selected=(True, False),
    generations=(1, 0),
    facts=((10, 2, 1), (-1, -1, 0)),
):
    return _entry(
        sequence,
        "RESET_SELECTED",
        (
            "selected_mask",
            "reset_generation",
            "terminal_reset_facts_i64",
        ),
        (
            torch.tensor(selected, dtype=torch.bool),
            torch.tensor(generations, dtype=torch.int64),
            torch.tensor(facts, dtype=torch.int64),
        ),
        epoch=-1,
    )


def _append_d05(
    entries, *, due_row, decision, valid_key, epoch, selected=True, key_seed=10_000,
    family_code=0, attribution_valid=False,
):
    due = torch.zeros(SHAPE, dtype=torch.bool)
    due[due_row, 0] = True
    selected_mask = torch.zeros(SHAPE, dtype=torch.bool)
    selected_mask[due_row, 0] = selected
    key = _key(
        valid_rows=(due_row,) if valid_key else (), seed=key_seed
    )
    decisions = torch.zeros(SHAPE, dtype=torch.int64)
    decisions[due] = decision
    construction = torch.zeros(SHAPE, dtype=torch.bool)
    playback = torch.zeros(SHAPE, dtype=torch.bool)
    faults = torch.zeros((*SHAPE, 7), dtype=torch.int64)
    family = torch.zeros(SHAPE, dtype=torch.int64)
    attributed = torch.zeros(SHAPE, dtype=torch.bool)
    family[due] = family_code
    attributed[due] = attribution_valid
    if decision == D.D05_DECISION_ACCEPT:
        construction[due] = True
        playback[due] = True
    elif decision == D.D05_DECISION_DEFER:
        construction[due] = True
    elif decision == D.D05_DECISION_CENSOR:
        faults[..., 0][due] = 1
    entries.append(
        _entry(
            len(entries),
            "D05_SETTLED",
            D.SHOT_EVENT_PREFIX
            + (
                "due_mask",
                "selected_mask",
                "decision",
                "construction_admissible",
                "playback_admissible",
                "owner_fault_bits",
                "stroke_family_code",
                "action_attribution_valid",
            ),
            (
                *_event(due, key),
                due,
                selected_mask,
                decisions,
                construction,
                playback,
                faults,
                family,
                attributed,
            ),
            epoch=epoch,
        )
    )
    accepted = due & decisions.eq(D.D05_DECISION_ACCEPT)
    transitions = (
        "WRITES_STARTED:motion",
        "WRITES_COMMITTED:motion",
        "WRITES_STARTED:racket",
        "WRITES_COMMITTED:racket",
        "WRITES_STARTED:r05_runtime",
        "WRITES_COMMITTED:r05_runtime",
    )
    for transition in transitions:
        entries.append(
            _entry(
                len(entries),
                transition,
                D.SHOT_EVENT_PREFIX,
                _event(accepted, key),
                epoch=epoch,
            )
        )
    publication = torch.full(SHAPE, epoch, dtype=torch.int64)
    entries.append(
        _entry(
            len(entries),
            "D05_ACCEPT_PUBLISHED",
            D.SHOT_EVENT_PREFIX + ("publication_ordinal",),
            (*_event(accepted, key), publication),
            epoch=epoch,
        )
    )
    return key


def _decode(entries, previous=None, *, start=0, milestone=None):
    previous = previous or D.ActionEpochDrainContinuation.empty(
        num_envs=SHAPE[0], shot_slot_capacity=SHAPE[1]
    )
    sequenced = tuple(
        _entry(
            start + offset,
            entry.transition,
            entry.delta.names,
            entry.delta.values,
            epoch=entry.epoch,
        )
        for offset, entry in enumerate(entries)
    )
    milestone_i64, milestone_f64 = (
        milestone if milestone is not None else _milestone()
    )
    return D.decode_epoch_drain_suffix(
        sequenced,
        start_commit=start,
        end_commit=start + len(entries),
        previous=previous,
        milestone_i64=milestone_i64,
        milestone_f64=milestone_f64,
    )


def _lifecycle_entry(sequence, transition, mask, key, remainder_names=(), values=()):
    return _entry(
        sequence,
        transition,
        D.SHOT_EVENT_PREFIX + remainder_names,
        (*_event(mask, key), *values),
    )


def _append_completed_shot(
    entries, *, row, epoch, key_seed, target_xy, settlement_step
):
    key = _append_d05(
        entries,
        due_row=row,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=epoch,
        key_seed=key_seed,
    )
    # Four attempts may deliberately reuse the policy uid/slot; the remaining
    # six key fields retain physical-shot identity.
    key["action_uid"][row, 0] = 77
    key["action_slot"][row, 0] = 0
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[row, 0] = True
    publication = torch.full(SHAPE, epoch, dtype=torch.int64)
    target = torch.zeros((*SHAPE, 2), dtype=torch.float32)
    target[row, 0] = torch.tensor(target_xy, dtype=torch.float32)
    settlement = torch.full(SHAPE, -1, dtype=torch.int64)
    payment = torch.full(SHAPE, -1, dtype=torch.int64)
    retirement = torch.full(SHAPE, -1, dtype=torch.int64)
    reason = torch.full(SHAPE, -1, dtype=torch.int64)
    settlement[row, 0] = settlement_step
    payment[row, 0] = settlement_step + 1
    retirement[row, 0] = settlement_step + 2
    reason[row, 0] = 1
    specs = (
        ("MOTION_PLAYBACK_STARTED", (), ()),
        (
            "PHYSICAL_LAUNCH_ROWS",
            (
                "publication_ordinal",
                "launch_succeeded",
                "late_launch",
                "owner_fault_bits",
                "target_xy_m",
            ),
            (
                publication,
                mask.clone(),
                torch.zeros(SHAPE, dtype=torch.bool),
                torch.zeros(SHAPE, dtype=torch.int64),
                target,
            ),
        ),
        (
            "R06_OUTCOME_ROWS",
            (
                "publication_ordinal",
                "settlement_step",
                "valid_bits",
                "outcome_code",
                "owner_fault_bits",
                "predicate_bits",
            ),
            (
                publication,
                settlement,
                torch.full(SHAPE, 7, dtype=torch.int64),
                torch.ones(SHAPE, dtype=torch.int64),
                torch.zeros(SHAPE, dtype=torch.int64),
                torch.full(SHAPE, 63, dtype=torch.int64),
            ),
        ),
        ("PAYMENT_RECORDED", ("payment_step",), (payment,)),
        ("MOTION_CLOSED", ("motion_close_reason",), (reason,)),
        (
            "RETIRED",
            ("motion_close_reason", "payment_step", "retirement_step"),
            (reason, payment, retirement),
        ),
    )
    for transition, names, values in specs:
        entries.append(
            _lifecycle_entry(
                len(entries), transition, mask, key, names, values
            )
        )
    return key


def test_update_zero_empty_suffix_is_a_general_zero_transaction_abi():
    decoded = _decode(())
    assert decoded.settlement.transactions == 0
    assert decoded.settlement.selected_rows == 0
    assert decoded.continuation.active_before == 0
    assert decoded.continuation.active_after == 0


def test_event_telemetry_and_business_journal_remain_separately_visible():
    entries = []
    _append_completed_shot(
        entries, row=0, epoch=0, key_seed=7000,
        target_xy=(0.1, 0.2), settlement_step=10,
    )
    telemetry_i64, telemetry_f64 = _milestone()
    base = D.milestone_tensors._EI
    known = torch.arange(
        1, 1 + len(D.milestone_tensors.EVENT_NAMES), dtype=torch.int64
    )
    telemetry_i64[base:base + known.numel()] = known
    decoded = _decode(
        tuple(entries), milestone=(telemetry_i64, telemetry_f64)
    )
    assert decoded.settlement.due_rows == 1
    assert decoded.lifecycle.outcome_settled_rows == 1
    assert decoded.milestone.i64[base:base + known.numel()] == tuple(known.tolist())


def test_episode_window_matches_exact_selected_reset_suffix():
    milestone_i64, milestone_f64 = _milestone()
    milestone_i64[-7:] = torch.tensor([1, 2, 1, 0, 0, 0, 0])
    decoded = _decode(
        (_selected_reset_entry(0),),
        milestone=(milestone_i64, milestone_f64),
    )
    assert len(decoded.terminal_resets) == 1

    mismatched = milestone_i64.clone()
    mismatched[-6] = 3
    with pytest.raises(
        D.ActionEpochDrainDecodeError,
        match="episode/reset suffix relationship differs",
    ):
        _decode(
            (_selected_reset_entry(0),),
            milestone=(mismatched, milestone_f64),
        )


def test_frozen_producer_negative_unpublished_epoch_rows_decode():
    owner = E.ActionEpochOwner(num_envs=SHAPE[0], device="cpu")
    owner.activate_reset_genesis(
        selected_mask=torch.ones(SHAPE[0], dtype=torch.bool),
        reset_generation=torch.zeros(SHAPE[0], dtype=torch.int64),
    )
    owner.open_reward_cycle()
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        owner.pay_reward(ordinal)
    start, end = owner.prepare_drain()
    entries = owner.materialize_drain(start=start, end=end).entries

    assert [
        (
            entry.sequence,
            entry.epoch,
            entry.transition,
            entry.before_version,
            entry.after_version,
        )
        for entry in entries[:2]
    ] == [
        (0, -1, "RESET_GENESIS_IDLE", -1, 0),
        (1, -1, "REWARD_CYCLE_OPEN", 0, 1),
    ]
    assert all(entry.epoch == -1 for entry in entries)

    milestone_i64, milestone_f64 = _milestone()
    decoded = D.decode_epoch_drain_suffix(
        entries,
        start_commit=start,
        end_commit=end,
        previous=D.ActionEpochDrainContinuation.empty(
            num_envs=SHAPE[0], shot_slot_capacity=SHAPE[1]
        ),
        milestone_i64=milestone_i64,
        milestone_f64=milestone_f64,
    )
    assert decoded.settlement.transactions == 0
    assert decoded.continuation.active_after == 0
    assert decoded.terminal_resets == ()


def test_negative_d05_publication_fails_closed():
    d05_entries = []
    _append_d05(
        d05_entries,
        due_row=0,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=-1,
    )
    with pytest.raises(
        D.ActionEpochDrainDecodeError,
        match="D05 publication chronology differs",
    ):
        _decode(d05_entries)


def test_continuation_key_fields_have_independent_storage():
    continuation = D.ActionEpochDrainContinuation.empty(
        num_envs=SHAPE[0], shot_slot_capacity=SHAPE[1]
    )
    clone = continuation.clone()
    original_pointers = {
        getattr(continuation.key, name).untyped_storage().data_ptr()
        for name in D.SHOT_KEY_FIELDS
    }
    clone_pointers = {
        getattr(clone.key, name).untyped_storage().data_ptr()
        for name in D.SHOT_KEY_FIELDS
    }
    assert len(original_pointers) == len(clone_pointers) == len(D.SHOT_KEY_FIELDS)
    assert original_pointers.isdisjoint(clone_pointers)


def test_journal_key_cross_field_storage_overlap_fails_closed():
    entries = []
    _append_d05(
        entries,
        due_row=0,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=4,
    )
    first = entries[0]
    values = list(first.delta.values)
    values[2] = values[1]
    entries[0] = _entry(
        0,
        first.transition,
        first.delta.names,
        tuple(values),
        epoch=first.epoch,
    )
    with pytest.raises(D.ActionEpochDrainDecodeError, match="overlaps storage"):
        _decode(entries)


def test_two_rowwise_transactions_conserve_due_not_global_n():
    entries = []
    _append_d05(
        entries,
        due_row=0,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=4,
    )
    _append_d05(
        entries,
        due_row=1,
        decision=D.D05_DECISION_CENSOR,
        valid_key=False,
        epoch=5,
        selected=False,
    )
    decoded = _decode(entries)
    assert decoded.settlement.transactions == 2
    assert decoded.settlement.due_rows == 2
    assert decoded.settlement.selected_rows == 1
    assert decoded.settlement.accepted == 1
    assert decoded.settlement.censored == 1
    assert decoded.reveal_commit.motion_committed_rows == 1
    assert decoded.reveal_commit.racket_committed_rows == 1
    assert decoded.reveal_commit.r05_committed_rows == 1
    assert decoded.continuation.active_after == 1


@pytest.mark.parametrize(
    ("decision", "selected", "field"),
    (
        (D.D05_DECISION_ACCEPT, True, "accepted"),
        (D.D05_DECISION_CENSOR, False, "censored"),
        (D.D05_DECISION_REJECT, True, "rejected"),
        (D.D05_DECISION_DEFER, True, "deferred"),
    ),
)
def test_each_due_row_emits_one_typed_bh_action_decision(decision, selected, field):
    entries = []
    _append_d05(
        entries, due_row=0, decision=decision, valid_key=True, epoch=4,
        selected=selected, family_code=2, attribution_valid=True,
    )
    row = _decode(entries).action_opportunities[0]
    assert row.stroke_family == "backhand" and row.attribution_valid is True
    assert row.selected is selected
    assert sum((row.accepted, row.censored, row.rejected, row.deferred)) == 1
    assert getattr(row, field) is True


def test_playback_outcome_payment_and_retire_may_cross_suffixes_by_full_key():
    first = []
    key = _append_d05(
        first,
        due_row=0,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=4,
        family_code=2,
        attribution_valid=True,
    )
    decoded1 = _decode(first)
    assert decoded1.continuation.awaiting_playback_after == 1
    assert decoded1.action_opportunities[0].stroke_family == "backhand"

    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0] = True
    i64 = torch.full(SHAPE, -1, dtype=torch.int64)
    publication = i64.clone()
    publication[mask] = 4
    second = [
        _lifecycle_entry(0, "MOTION_PLAYBACK_STARTED", mask, key),
        _lifecycle_entry(
            1,
            "PHYSICAL_LAUNCH_ROWS",
            mask,
            key,
            (
                "publication_ordinal",
                "launch_succeeded",
                "late_launch",
                "owner_fault_bits",
                "target_xy_m",
            ),
            (
                publication,
                mask.clone(),
                torch.zeros(SHAPE, dtype=torch.bool),
                torch.zeros(SHAPE, dtype=torch.int64),
                torch.tensor([[[0.21, -0.13]], [[0.0, 0.0]]], dtype=torch.float32),
            ),
        ),
        _lifecycle_entry(
            2,
            "R06_OUTCOME_ROWS",
            mask,
            key,
            (
                "publication_ordinal",
                "settlement_step",
                "valid_bits",
                "outcome_code",
                "owner_fault_bits",
                "predicate_bits",
            ),
            (
                publication,
                torch.full(SHAPE, 7, dtype=torch.int64),
                torch.ones(SHAPE, dtype=torch.int64),
                torch.full(SHAPE, 2, dtype=torch.int64),
                torch.zeros(SHAPE, dtype=torch.int64),
                torch.zeros(SHAPE, dtype=torch.int64),
            ),
        ),
    ]
    second.append(
        _entry(
            len(second),
            "REWARD_CYCLE_OPEN",
            ("reward_cycle_age", "reward_cycle_fault"),
            (
                torch.ones(SHAPE[0], dtype=torch.int64),
                torch.zeros(SHAPE[0], dtype=torch.int64),
            ),
        )
    )
    paid = torch.zeros((SHAPE[0], 14), dtype=torch.bool)
    for ordinal in range(14):
        paid[:, ordinal] = True
        second.append(
            _entry(
                len(second),
                "REWARD_CONSUMER_PAID",
                ("reward_consumer_ordinal", "reward_paid"),
                (
                    torch.full((SHAPE[0],), ordinal, dtype=torch.int64),
                    paid.clone(),
                ),
            )
        )
    second.append(
        _lifecycle_entry(
            len(second),
            "PAYMENT_RECORDED",
            mask,
            key,
            ("payment_step",),
            (torch.full(SHAPE, 8, dtype=torch.int64),),
        )
    )
    decoded2 = _decode(
        second, decoded1.next_continuation, start=len(first)
    )
    assert decoded2.settlement.transactions == 0
    assert decoded2.lifecycle.playback_started_rows == 1
    assert decoded2.lifecycle.physical_launch_rows == 1
    assert decoded2.lifecycle.outcome_settled_rows == 1
    assert decoded2.lifecycle.payment_recorded_rows == 1
    assert decoded2.completed_shots == ()
    assert decoded2.next_continuation.physical_target_xy_m[0, 0].tolist() == pytest.approx(
        [0.21, -0.13]
    )

    empty = torch.zeros(SHAPE, dtype=torch.bool)
    reason = torch.full(SHAPE, -1, dtype=torch.int64)
    reason[mask] = 1
    payment = torch.full(SHAPE, 8, dtype=torch.int64)
    retirement = torch.full(SHAPE, -1, dtype=torch.int64)
    retirement[mask] = 9
    third = [
        _lifecycle_entry(
            0,
            "MOTION_CLOSED",
            empty,
            _key(),
            ("motion_close_reason",),
            (torch.full(SHAPE, -1, dtype=torch.int64),),
        ),
        _lifecycle_entry(
            1,
            "MOTION_CLOSED",
            mask,
            key,
            ("motion_close_reason",),
            (reason,),
        ),
        _lifecycle_entry(
            2,
            "RETIRED",
            mask,
            key,
            ("motion_close_reason", "payment_step", "retirement_step"),
            (reason, payment, retirement),
        ),
        _lifecycle_entry(
            3,
            "RETIRED",
            empty,
            _key(),
            ("motion_close_reason", "payment_step", "retirement_step"),
            (
                torch.full(SHAPE, -1, dtype=torch.int64),
                torch.full(SHAPE, -1, dtype=torch.int64),
                torch.full(SHAPE, -1, dtype=torch.int64),
            ),
        ),
    ]
    decoded3 = _decode(
        third,
        decoded2.next_continuation,
        start=len(first) + len(second),
    )
    assert decoded3.lifecycle.retired_rows == 1
    assert decoded3.continuation.active_after == 0
    assert decoded3.completed_shots == (
        D.CompletedActionEpochShot(
            env_row=0,
            slot_index=0,
            **{name: int(key[name][0, 0]) for name in D.SHOT_KEY_FIELDS},
            target_x_m=pytest.approx(0.21),
            target_y_m=pytest.approx(-0.13),
            motion_close_reason=1,
            settlement_step=7,
            payment_step=8,
            retirement_step=9,
            stroke_family="backhand",
            action_attribution_valid=True,
            evidence=D.ActionEpochShotEvidence(
                lifecycle_bits=sum(
                    D.SHOT_LIFECYCLE_BITS[name]
                    for name in (
                        "reveal_committed", "playback_started", "motion_closed",
                        "physical_launched", "outcome_settled",
                        "payment_recorded",
                    )
                ),
                r03_valid_bits=0,
                r03_source_step=-1,
                physical_valid_bits=0,
                physical_actor_pair_contact_source_step=-1,
                r06_valid_bits=1,
                r06_outcome_code=2,
                r06_predicate_bits=0,
                r07_valid_bits=0,
                r07_qualified_source_step=-1,
                r07_first_ready_source_step=-1,
            ),
        ),
    )


def test_selected_reset_clears_only_the_selected_continuation_row():
    entries = []
    _append_d05(
        entries,
        due_row=0,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=4,
    )
    _append_d05(
        entries,
        due_row=1,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=5,
    )
    first = _decode(entries)
    selected = torch.tensor([True, False], dtype=torch.bool)
    reset = _selected_reset_entry(0, generations=(11_000, 0))
    milestone_i64, milestone_f64 = _milestone()
    milestone_i64[-7:] = torch.tensor([1, 2, 1, 0, 0, 0, 0])
    second = _decode(
        (reset,), first.next_continuation, start=len(entries),
        milestone=(milestone_i64, milestone_f64),
    )
    assert second.continuation.active_before == 2
    assert second.continuation.active_after == 1
    assert not bool(second.next_continuation.occupied[0, 0])
    assert bool(second.next_continuation.occupied[1, 0])
    assert second.terminal_resets == (
        D.ResetTelemetry(0, 11_000, 10, 2, 1),
    )
    assert len(second.terminal_shots) == 1
    assert second.terminal_shots[0].reset_generation_after == 11_000
    assert second.terminal_shots[0].reset_reason_bits == 1
    assert second.lifecycle.terminal_shot_rows == 1


def test_r07_invalid_then_qualified_and_first_ready_preserve_true_sources():
    first = []
    key = _append_d05(
        first, due_row=0, decision=D.D05_DECISION_ACCEPT,
        valid_key=True, epoch=4,
    )
    decoded = _decode(first)
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0] = True

    def facts(valid_bits, source_step, qualified):
        valid = torch.zeros(SHAPE, dtype=torch.int64)
        source = torch.full(SHAPE, -1, dtype=torch.int64)
        selected = torch.zeros(SHAPE, dtype=torch.bool)
        valid[mask] = valid_bits
        source[mask] = source_step
        selected[mask] = qualified
        return _lifecycle_entry(
            0, "OWNER_FACTS:r07_recovery", mask, key,
            ("valid_bits", "source_step", "qualified"),
            (valid, source, selected),
        )

    r03_valid = torch.zeros(SHAPE, dtype=torch.int64)
    r03_step = torch.full(SHAPE, -1, dtype=torch.int64)
    r03_qualified = torch.zeros(SHAPE, dtype=torch.bool)
    r03_valid[mask], r03_step[mask], r03_qualified[mask] = 3, 7, True
    r03 = _lifecycle_entry(
        0, "OWNER_FACTS:r03_strike_fact", mask, key,
        ("valid_bits", "source_step", "qualified"),
        (r03_valid, r03_step, r03_qualified),
    )
    physical_valid = torch.zeros(SHAPE, dtype=torch.int64)
    physical_step = torch.full(SHAPE, -1, dtype=torch.int64)
    physical_valid[mask], physical_step[mask] = 3, 9
    physical = _lifecycle_entry(
        1, "PHYSICAL_POSTPHYSICS_ROWS", mask, key,
        ("publication_ordinal", "owner_fault_bits", "fact_valid_bits",
         "fact_source_step"),
        (torch.full(SHAPE, 4, dtype=torch.int64),
         torch.zeros(SHAPE, dtype=torch.int64), physical_valid, physical_step),
    )
    decoded = _decode(
        (r03, physical, facts(1, -1, False)),
        decoded.next_continuation,
        start=len(first),
    )
    assert decoded.next_continuation.r07_valid_bits[0, 0].item() == 1
    assert decoded.next_continuation.r07_qualified_source_step[0, 0].item() == -1
    decoded = _decode(
        (facts(3, 11, True),), decoded.next_continuation, start=len(first) + 3
    )
    step = torch.full(SHAPE, -1, dtype=torch.int64)
    step[mask] = 13
    ready = _lifecycle_entry(
        0, "R07_FIRST_READY", mask, key, ("source_step",), (step,)
    )
    decoded = _decode(
        (ready,), decoded.next_continuation, start=len(first) + 4
    )
    milestone_i64, milestone_f64 = _milestone()
    milestone_i64[-7:] = torch.tensor([1, 2, 1, 0, 0, 0, 0])
    closed = _decode(
        (_selected_reset_entry(0, generations=(11_000, 0)),),
        decoded.next_continuation,
        start=len(first) + 5,
        milestone=(milestone_i64, milestone_f64),
    )
    evidence = closed.terminal_shots[0].evidence
    assert evidence.r03_valid_bits == 3 and evidence.r03_source_step == 7
    assert evidence.physical_valid_bits == 3
    assert evidence.physical_actor_pair_contact_source_step == 9
    assert evidence.r07_valid_bits == 3
    assert evidence.r07_qualified_source_step == 11
    assert evidence.r07_first_ready_source_step == 13


@pytest.mark.parametrize(
    ("valid_bits", "outcome_code", "predicates"),
    ((3, 1, 0), (4, 1, 0), (1, 1, 1), (7, 1, 4), (1, 0, 0)),
)
def test_r06_raw_bit_relationship_mutants_fail_closed(
    valid_bits, outcome_code, predicates,
):
    first = []
    key = _append_d05(
        first, due_row=0, decision=D.D05_DECISION_ACCEPT,
        valid_key=True, epoch=4,
    )
    decoded = _decode(first)
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0] = True
    publication = torch.full(SHAPE, 4, dtype=torch.int64)
    target = torch.zeros((*SHAPE, 2), dtype=torch.float32)
    launch = _lifecycle_entry(
        0, "PHYSICAL_LAUNCH_ROWS", mask, key,
        ("publication_ordinal", "launch_succeeded", "late_launch",
         "owner_fault_bits", "target_xy_m"),
        (publication, mask.clone(), torch.zeros(SHAPE, dtype=torch.bool),
         torch.zeros(SHAPE, dtype=torch.int64), target),
    )
    decoded = _decode(
        (launch,), decoded.next_continuation, start=len(first)
    )
    i64 = torch.full(SHAPE, -1, dtype=torch.int64)
    settlement, valid, outcome, predicate = (i64.clone() for _ in range(4))
    settlement[mask] = 7
    valid[mask] = valid_bits
    outcome[mask] = outcome_code
    predicate[mask] = predicates
    event = _lifecycle_entry(
        0, "R06_OUTCOME_ROWS", mask, key,
        ("publication_ordinal", "settlement_step", "valid_bits",
         "outcome_code", "owner_fault_bits", "predicate_bits"),
        (publication, settlement, valid, outcome,
         torch.zeros(SHAPE, dtype=torch.int64), predicate),
    )
    with pytest.raises(D.ActionEpochDrainDecodeError, match="R06 outcome"):
        _decode((event,), decoded.next_continuation, start=len(first) + 1)


def test_r06_present_producer_fault_is_not_misreported_as_eligible():
    entries = []
    key = _append_d05(
        entries, due_row=0, decision=D.D05_DECISION_ACCEPT,
        valid_key=True, epoch=4,
    )
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0] = True
    publication = torch.full(SHAPE, 4, dtype=torch.int64)
    target = torch.zeros((*SHAPE, 2), dtype=torch.float32)
    entries.append(_lifecycle_entry(
        len(entries), "PHYSICAL_LAUNCH_ROWS", mask, key,
        ("publication_ordinal", "launch_succeeded", "late_launch",
         "owner_fault_bits", "target_xy_m"),
        (publication, mask.clone(), torch.zeros(SHAPE, dtype=torch.bool),
         torch.zeros(SHAPE, dtype=torch.int64), target),
    ))
    settlement = torch.full(SHAPE, -1, dtype=torch.int64)
    valid = torch.full(SHAPE, -1, dtype=torch.int64)
    outcome = torch.full(SHAPE, -1, dtype=torch.int64)
    settlement[mask], valid[mask], outcome[mask] = 7, 1, 5
    entries.append(_lifecycle_entry(
        len(entries), "R06_OUTCOME_ROWS", mask, key,
        ("publication_ordinal", "settlement_step", "valid_bits",
         "outcome_code", "owner_fault_bits", "predicate_bits"),
        (publication, settlement, valid, outcome,
         torch.ones(SHAPE, dtype=torch.int64),
         torch.zeros(SHAPE, dtype=torch.int64)),
    ))
    decoded = _decode(entries)
    evidence = D._shot_evidence(decoded.next_continuation, 0, 0)
    assert evidence.r06_valid_bits == 1
    assert evidence.r06_predicate_bits == 0


@pytest.mark.parametrize(
    "facts",
    (
        ((10, 2, 1), (9, -1, 0)),
        ((10, 2, 1), (-1, -1, 2)),
        ((10, 2, 32), (-1, -1, 0)),
        ((10, 2, 0), (-1, -1, 0)),
        ((0, 2, 1), (-1, -1, 0)),
        ((10, 0, 1), (-1, -1, 0)),
    ),
)
def test_selected_reset_telemetry_illegal_counterexamples_fail_closed(facts):
    with pytest.raises(
        D.ActionEpochDrainDecodeError,
        match="selected-reset telemetry differs",
    ):
        _decode((_selected_reset_entry(0, facts=facts),))


def test_same_env_multiple_resets_and_multibit_reasons_remain_distinct():
    milestone_i64, milestone_f64 = _milestone()
    milestone_i64[-7:] = torch.tensor([2, 8, 1, 0, 2, 0, 1])
    decoded = _decode(
        (
            _selected_reset_entry(
                0,
                generations=(4, 0),
                facts=((30, 7, 1 | 4), (-1, -1, 0)),
            ),
            _selected_reset_entry(
                1,
                generations=(5, 0),
                facts=((31, 1, 4 | 16), (-1, -1, 0)),
            ),
        ),
        milestone=(milestone_i64, milestone_f64),
    )
    assert decoded.terminal_resets == (
        D.ResetTelemetry(0, 4, 30, 7, 1 | 4),
        D.ResetTelemetry(0, 5, 31, 1, 4 | 16),
    )


def test_partial_n2_four_completed_shots_keep_distinct_keys_and_adjacent_targets():
    entries = []
    keys = []
    targets = []
    for index in range(4):
        target = (0.30 + index * 0.0001, -0.20 - index * 0.0001)
        targets.append(target)
        keys.append(
            _append_completed_shot(
                entries,
                row=0,
                epoch=10 + index,
                key_seed=20_000 + 1_000 * index,
                target_xy=target,
                settlement_step=100 + 3 * index,
            )
        )
    decoded = _decode(entries)
    assert decoded.lifecycle.retired_rows == 4
    assert len(decoded.completed_shots) == 4
    assert {(shot.env_row, shot.slot_index) for shot in decoded.completed_shots} == {
        (0, 0)
    }
    assert {shot.action_uid for shot in decoded.completed_shots} == {77}
    assert {shot.action_slot for shot in decoded.completed_shots} == {0}
    assert len({shot.shot_index for shot in decoded.completed_shots}) == 4
    for shot, key in zip(decoded.completed_shots, keys):
        assert tuple(getattr(shot, field) for field in D.SHOT_KEY_FIELDS) == tuple(
            int(key[field][0, 0]) for field in D.SHOT_KEY_FIELDS
        )
    assert [shot.target_x_m for shot in decoded.completed_shots] == pytest.approx(
        [target[0] for target in targets]
    )
    assert [shot.target_y_m for shot in decoded.completed_shots] == pytest.approx(
        [target[1] for target in targets]
    )
    assert decoded.continuation.active_after == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong_key", "full shot key"),
        ("nan_target", "Physical launch"),
        ("neutral_target", "Physical launch"),
        ("rollback", "retirement"),
        ("neutral_retirement", "retirement"),
    ),
)
def test_completed_shot_wrong_key_nan_and_retirement_rollback_fail_closed(
    mutation, message
):
    entries = []
    _append_completed_shot(
        entries,
        row=0,
        epoch=10,
        key_seed=20_000,
        target_xy=(0.30, -0.20),
        settlement_step=100,
    )
    transition = (
        "PHYSICAL_LAUNCH_ROWS"
        if mutation in ("nan_target", "neutral_target")
        else "RETIRED"
    )
    index = next(i for i, entry in enumerate(entries) if entry.transition == transition)
    entry = entries[index]
    values = list(entry.delta.values)
    if mutation == "wrong_key":
        value_index = entry.delta.names.index("shot_key.action_uid")
        values[value_index] = values[value_index].clone()
        values[value_index][0, 0] += 1
    elif mutation == "nan_target":
        value_index = entry.delta.names.index("target_xy_m")
        values[value_index] = values[value_index].clone()
        values[value_index][0, 0, 0] = float("nan")
    elif mutation == "neutral_target":
        value_index = entry.delta.names.index("target_xy_m")
        values[value_index] = values[value_index].clone()
        values[value_index][1, 0, 0] = 0.5
    elif mutation == "neutral_retirement":
        value_index = entry.delta.names.index("retirement_step")
        values[value_index] = values[value_index].clone()
        values[value_index][1, 0] = 999
    else:
        value_index = entry.delta.names.index("retirement_step")
        values[value_index] = values[value_index].clone()
        values[value_index][0, 0] = 99
    entries[index] = _entry(
        index,
        transition,
        entry.delta.names,
        tuple(values),
        epoch=entry.epoch,
    )
    with pytest.raises(D.ActionEpochDrainDecodeError, match=message):
        _decode(entries)


def test_writer_reorder_and_wrong_key_lifecycle_fail_closed():
    entries = []
    key = _append_d05(
        entries,
        due_row=0,
        decision=D.D05_DECISION_ACCEPT,
        valid_key=True,
        epoch=4,
    )
    reordered = list(entries)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    reordered = [
        _entry(i, row.transition, row.delta.names, row.delta.values, epoch=row.epoch)
        for i, row in enumerate(reordered)
    ]
    with pytest.raises(D.ActionEpochDrainDecodeError, match="order"):
        _decode(reordered)

    decoded = _decode(entries)
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[0, 0] = True
    wrong = _key(valid_rows=(0,))
    wrong["action_uid"][0, 0] += 1
    event = _lifecycle_entry(0, "MOTION_PLAYBACK_STARTED", mask, wrong)
    with pytest.raises(D.ActionEpochDrainDecodeError, match="full shot key"):
        _decode((event,), decoded.next_continuation, start=len(entries))
