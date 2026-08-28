import ast
import inspect
import importlib
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (SOURCE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import action_ball_continuous_runtime_transaction_device as d05  # noqa: E402


epoch = d05._require_action_epoch_module()


@dataclass(frozen=True)
class _PreviousPaidRows:
    valid: torch.Tensor
    shot_key: object
    publication_ordinal: torch.Tensor
    settlement_step: torch.Tensor
    payment_step: torch.Tensor


_FAKE_R06 = ModuleType("action_ball_landing_outcome_device")
_FAKE_R06.PreviousPaidActionEpochRows = _PreviousPaidRows


def _candidate(
    n: int, *, invalid: torch.Tensor | None = None, epoch_module=None
):
    current_epoch = epoch if epoch_module is None else epoch_module
    shape = (n, 1)
    row = torch.arange(n, dtype=torch.int64).reshape(shape)
    positive = row + 1
    invalid = (
        torch.zeros(n, dtype=torch.bool) if invalid is None else invalid
    ).reshape(shape)

    def keyed(value: torch.Tensor) -> torch.Tensor:
        return torch.where(invalid, torch.full_like(value, -1), value).contiguous()

    key = current_epoch.ActionEpochShotKey(
        reset_generation=keyed(row),
        ball_generation=keyed(positive + 10),
        action_uid=keyed(positive + 20),
        action_slot=keyed(row),
        shot_index=keyed(positive + 30),
        task_identity=keyed(positive + 40),
        outcome_identity=keyed(positive + 50),
        ball_identity=keyed(positive + 60),
    )
    identity = current_epoch.EpochIdentityPayload(
        shot_key=key,
        scheduled_ordinal=(positive + 70).contiguous(),
        target_generation=(positive + 80).contiguous(),
        selected_cell=row.remainder(3).contiguous(),
        candidate_identity=(positive + 90).contiguous(),
    )
    clocks = current_epoch.EpochClockPayload(
        reveal_tick=(positive + 100).contiguous(),
        contact_tick=(positive + 110).contiguous(),
        launch_tick=(positive + 120).contiguous(),
        deadline_tick=(positive + 130).contiguous(),
        next_reveal_tick=(positive + 140).contiguous(),
    )
    return current_epoch.ActionEpochD05CandidateProjection(
        identity=identity,
        clocks=clocks,
        task=current_epoch.EpochTaskPayload(
            task_f32=torch.ones(n, 1, current_epoch.TASK_F32_WIDTH),
            task_valid=torch.ones(shape, dtype=torch.bool),
        ),
        rng_counter=(positive + 150).contiguous(),
        construction_admissible=torch.ones(shape, dtype=torch.bool),
        playback_admissible=torch.ones(shape, dtype=torch.bool),
        owner_fault_bits=torch.zeros(
            n, 1, len(current_epoch.OWNER_ORDER), dtype=torch.int64
        ),
    )


def _private_owner(n: int, candidate):
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = n
    owner._device = torch.device("cpu")
    owner._action_epoch_candidate = lambda prepared: candidate
    return owner


def _pairwise_disjoint(*masks: torch.Tensor) -> bool:
    for left_index, left in enumerate(masks):
        for right in masks[left_index + 1 :]:
            if bool((left & right).any()):
                return False
    return True


@pytest.mark.parametrize("n", (1, 2, 64))
def test_row_transaction_masks_are_full_n_disjoint_and_cover_due(n):
    lane = torch.arange(n, dtype=torch.int64)
    kind = lane.remainder(6)
    due = kind.ne(0)
    construct = due & kind.ne(1)
    invalid = kind.eq(5)
    admitted = ~kind.eq(3)
    playable = ~kind.eq(4)
    clean = ~kind.eq(2)
    candidate = _candidate(n, invalid=invalid)
    owner = _private_owner(n, candidate)
    prepared = SimpleNamespace(
        owner_fault_free=clean,
        admissible=admitted,
        projection=SimpleNamespace(ready_at_reveal=playable),
        selected_target_xy_m=torch.ones(n, 2),
    )
    token = object.__new__(d05.DeviceR05RowTransaction)
    record = owner._build_row_transaction(
        token,
        epoch.ActionEpochDueRows(7, due, construct),
        prepared,
        SimpleNamespace(prepared=prepared),
    )
    masks = (
        record.accept_mask,
        record.reject_mask,
        record.defer_mask,
        record.censor_mask,
    )
    assert all(mask.dtype is torch.bool and mask.shape == (n,) for mask in masks)
    assert _pairwise_disjoint(*masks)
    assert torch.equal(masks[0] | masks[1] | masks[2] | masks[3], due)
    assert not bool(record.candidate.task.task_valid[~construct].any())
    for field in record.candidate.identity.shot_key.__dataclass_fields__:
        assert bool(getattr(record.candidate.identity.shot_key, field)[~construct].eq(-1).all())


def test_all_false_due_is_neutral_and_has_no_selected_shape():
    n = 64
    due = torch.zeros(n, dtype=torch.bool)
    candidate = _candidate(n)
    owner = _private_owner(n, candidate)
    prepared = SimpleNamespace(
        owner_fault_free=torch.ones(n, dtype=torch.bool),
        admissible=torch.ones(n, dtype=torch.bool),
        projection=SimpleNamespace(
            ready_at_reveal=torch.ones(n, dtype=torch.bool)
        ),
        selected_target_xy_m=torch.ones(n, 2),
    )
    record = owner._build_row_transaction(
        object.__new__(d05.DeviceR05RowTransaction),
        epoch.ActionEpochDueRows(8, due, due.clone()),
        prepared,
        SimpleNamespace(prepared=prepared),
    )
    assert not any(bool(mask.any()) for mask in (
        record.accept_mask,
        record.reject_mask,
        record.defer_mask,
        record.censor_mask,
    ))
    assert record.candidate.task.task_f32.shape == (
        n, 1, epoch.TASK_F32_WIDTH
    )
    assert not bool(record.candidate.task.task_f32.any())


def test_second_due_after_unaccepted_first_due_has_clean_chronology():
    """A settled-but-unaccepted Q0 must not poison Q1 with bit 50."""

    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    owner._row_axis = torch.arange(2, dtype=torch.int64)
    owner._reset_generation = torch.tensor([1, 1], dtype=torch.int64)
    owner._scheduled_ordinal = torch.tensor([0, 0], dtype=torch.int64)
    owner._outcome_shot_index = torch.tensor([1, 1], dtype=torch.int64)
    owner._target_generation = torch.tensor([0, 0], dtype=torch.int64)
    due_mask = torch.ones(2, dtype=torch.bool)
    due = epoch.ActionEpochDueRows(588, due_mask, due_mask.clone())
    motion = SimpleNamespace(
        common_step=588,
        episode_tick=torch.tensor([588, 588], dtype=torch.int64),
        scheduled_ordinal=torch.tensor([1, 1], dtype=torch.int64),
        reveal_tick=torch.tensor([588, 588], dtype=torch.int64),
        deadline_tick=torch.tensor([733, 733], dtype=torch.int64),
        next_reveal_tick=torch.tensor([881, 881], dtype=torch.int64),
        reset_generation=torch.tensor([1, 1], dtype=torch.int64),
        swing_generation=torch.tensor([1, 1], dtype=torch.int64),
        reveal_due=due_mask.clone(),
        ready_at_reveal=torch.zeros(2, dtype=torch.bool),
    )

    projection = owner._current_row_cadence(due, motion)

    assert projection.scheduled_ordinal.tolist() == [1, 1]
    assert projection.outcome_shot_index.tolist() == [2, 2]
    assert not bool(projection.cadence_producer_fault.any())


class _EpochWriterWindow:
    def __init__(self, active: str, *, accept=(True, False)):
        self.active = active
        self.required = []
        self.accept_mask = torch.tensor(accept, dtype=torch.bool).reshape(2, 1)

    def require_active_d05_accepted_rows(
        self, token: object, *, owner_kind: str
    ):
        del token
        self.required.append(owner_kind)
        if owner_kind != self.active:
            raise RuntimeError("writer reordered")
        current_epoch = d05._require_action_epoch_module()
        return current_epoch.ActionEpochD05AcceptedRows(
            accept_mask=self.accept_mask.clone(),
            publication_ordinal=torch.full((2, 1), 17, dtype=torch.int64),
        )


def _armed_record(owner, *, accept=(True, False), epoch_module=None):
    candidate = _candidate(2, epoch_module=epoch_module)
    token = object.__new__(d05.DeviceR05RowTransaction)
    prepared = SimpleNamespace(selected_target_xy_m=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    record = d05._RowTransactionRecord(
        capability=token,
        candidate=candidate,
        prepared=prepared,
        preview=SimpleNamespace(prepared=prepared),
        due_mask=torch.tensor(accept),
        construct_mask=torch.tensor(accept),
        accept_mask=torch.tensor(accept),
        reject_mask=torch.zeros(2, dtype=torch.bool),
        defer_mask=torch.zeros(2, dtype=torch.bool),
        censor_mask=torch.zeros(2, dtype=torch.bool),
        candidate_consumed=False,
        accepted_consumers=set(),
        stage="settling",
    )
    owner._row_transaction_records = {token: record}
    owner._active_row_transaction = token
    return token, record


def test_candidate_and_accepted_views_are_opaque_one_time_and_neutral():
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    window = _EpochWriterWindow("motion")
    owner._diagnostic_epoch_owner = window
    token, record = _armed_record(owner)
    candidate = owner.require_owned_action_epoch_candidate(token)
    assert candidate is not record.candidate
    with pytest.raises(d05.DeviceR05ConflictError):
        owner.require_owned_action_epoch_candidate(token)
    accepted = owner.require_owned_action_epoch_accepted(
        token, owner_kind="motion"
    )
    assert type(accepted) is d05.DeviceR05AcceptedRowsView
    assert accepted.publication_ordinal.tolist() == [[17], [-1]]
    assert accepted.target_xy_m.tolist() == [[[1.0, 2.0]], [[0.0, 0.0]]]
    assert accepted.task.task_valid.tolist() == [[True], [False]]
    assert accepted.identity.shot_key.action_uid[:, 0].tolist()[1] == -1
    with pytest.raises(d05.DeviceR05ConflictError):
        owner.require_owned_action_epoch_accepted(token, owner_kind="motion")


def test_physical_consumer_maps_to_r05_writer_and_reorder_fails():
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    window = _EpochWriterWindow("r05_runtime")
    owner._diagnostic_epoch_owner = window
    token, _ = _armed_record(owner)
    owner.require_owned_action_epoch_accepted(token, owner_kind="physical_ball")
    assert window.required == ["r05_runtime"]
    token, _ = _armed_record(owner)
    with pytest.raises(RuntimeError, match="reordered"):
        owner.require_owned_action_epoch_accepted(token, owner_kind="racket")


@pytest.mark.parametrize(
    ("owner_kind", "epoch_kind"),
    (("motion", "motion"), ("racket", "racket"), ("physical_ball", "r05_runtime")),
)
def test_epoch_accept_mask_is_sole_leaf_authority(owner_kind, epoch_kind):
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    window = _EpochWriterWindow(epoch_kind, accept=(False, False))
    owner._diagnostic_epoch_owner = window
    token, record = _armed_record(owner, accept=(True, True))
    assert bool(record.accept_mask.all())
    accepted = owner.require_owned_action_epoch_accepted(
        token, owner_kind=owner_kind
    )
    assert window.required == [epoch_kind]
    assert accepted.publication_ordinal.eq(-1).all()
    assert accepted.target_xy_m.eq(0).all()
    assert accepted.task.task_valid.logical_not().all()
    assert accepted.task.task_f32.eq(0).all()
    for field in accepted.identity.shot_key.__dataclass_fields__:
        assert getattr(accepted.identity.shot_key, field).eq(-1).all()


class _DueMotion:
    def project_current_action_epoch_rows(self):
        return SimpleNamespace(
            common_step=1,
            reveal_due=torch.ones(2, dtype=torch.bool),
            closed_mask=torch.zeros(2, dtype=torch.bool),
            close_reason=torch.zeros(2, dtype=torch.int64),
        )


class _EmptyR06:
    def __init__(self, epoch_owner, epoch_module, paid_rows_type):
        self.epoch_owner = epoch_owner
        self.epoch_module = epoch_module
        self.paid_rows_type = paid_rows_type

    def project_previous_paid_action_epoch_rows(self):
        invalid = torch.full((2,), -1, dtype=torch.int64)
        return self.paid_rows_type(
            valid=torch.zeros(2, dtype=torch.bool),
            shot_key=self.epoch_module.ActionEpochShotKey(
                **{
                    name: invalid.clone()
                    for name in self.epoch_module.ActionEpochShotKey.__dataclass_fields__
                }
            ),
            publication_ordinal=invalid.clone(),
            settlement_step=invalid.clone(),
            payment_step=invalid.clone(),
        )

    def consume_closed_action_epoch_rows(self):
        self.epoch_owner.project_current_closed_action_epoch_rows(owner=self)


class _MotionLeaf:
    def __init__(self, owner):
        self.owner = owner
        self.state = torch.arange(2, dtype=torch.int64)
        self.view = None

    def commit_action_ball_full_mdp_motion_epoch_rows(self, token):
        self.view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind="motion"
        )
        self.state = torch.where(
            self.view.task.task_valid[:, 0],
            torch.full_like(self.state, 99),
            self.state,
        )


class _RacketLeaf:
    def __init__(self, owner):
        self.owner = owner
        self.state = torch.arange(2, dtype=torch.int64)
        self.view = None

    def commit_action_ball_full_mdp_racket_epoch_rows(self, token):
        self.view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind="racket"
        )
        self.state = torch.where(
            self.view.task.task_valid[:, 0],
            torch.full_like(self.state, 99),
            self.state,
        )


class _PhysicalLeaf:
    def __init__(self, owner):
        self.owner = owner
        self.state = torch.arange(2, dtype=torch.int64)
        self.view = None

    def retain_action_epoch_launch(self, token):
        self.view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind="physical_ball"
        )
        self.state = torch.where(
            self.view.task.task_valid[:, 0],
            torch.full_like(self.state, 99),
            self.state,
        )


def test_real_epoch_stale_reset_censors_mutated_d05_accept_mask(monkeypatch):
    monkeypatch.setitem(sys.modules, _FAKE_R06.__name__, _FAKE_R06)
    current_epoch = d05._require_action_epoch_module()
    paid_rows_type = _PreviousPaidRows
    if current_epoch.__package__:
        paid_rows_type = importlib.import_module(
            current_epoch.__package__ + ".action_ball_landing_outcome_device"
        ).PreviousPaidActionEpochRows
    epoch_owner = current_epoch.ActionEpochOwner(num_envs=2, device="cpu")
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.tensor([1, 0], dtype=torch.int64),
    )
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    owner._diagnostic_epoch_owner = epoch_owner
    owner._active_diagnostic_epoch_leaf_writer = None
    token, record = _armed_record(
        owner, accept=(True, True), epoch_module=current_epoch
    )
    motion, racket, physical = (
        _MotionLeaf(owner), _RacketLeaf(owner), _PhysicalLeaf(owner)
    )
    owner._diagnostic_motion_owner = motion
    owner._diagnostic_racket_owner = racket
    owner._diagnostic_physical_owner = physical
    published = []
    owner._publish_action_epoch_afterimage = (
        lambda preview, *, accepted, settled_due: published.append(
            (accepted.clone(), settled_due.clone())
        )
    )
    epoch_owner.bind_d05_accept_writers(
        motion_write=owner._commit_action_epoch_motion_write,
        racket_write=owner._commit_action_epoch_racket_write,
        r05_write=owner._commit_action_epoch_r05_write,
    )
    epoch_owner.bind_motion_cadence_owner(_DueMotion())
    r06 = _EmptyR06(epoch_owner, current_epoch, paid_rows_type)
    epoch_owner.bind_fact_owner("r06_landing_outcome", r06)
    epoch_owner.bind_async_owner("r06_landing_outcome", r06)
    before = epoch_owner.current()
    epoch_owner.prepare_after_command_rows()
    epoch_owner.settle_d05_transaction(token)
    after = epoch_owner.current()

    assert bool(record.accept_mask.all())
    assert all(leaf.state.tolist() == [0, 1] for leaf in (motion, racket, physical))
    assert all(not bool(leaf.view.task.task_valid.any()) for leaf in (motion, racket, physical))
    assert len(published) == 1 and not bool(published[0][0].any())
    assert bool(published[0][1].all())
    assert torch.equal(after.phase, before.phase)
    start, end = epoch_owner.prepare_drain()
    materialized = epoch_owner.materialize_drain(start=start, end=end)
    accepted_event = next(
        entry for entry in materialized.entries
        if entry.transition == "D05_ACCEPT_PUBLISHED"
    )
    assert not bool(accepted_event.delta.values[0].any())
    with pytest.raises(d05.DeviceR05ConflictError):
        owner.require_owned_action_epoch_accepted(token, owner_kind="motion")
    with pytest.raises(d05.DeviceR05ConflictError):
        owner.require_owned_action_epoch_accepted(
            object.__new__(d05.DeviceR05RowTransaction), owner_kind="motion"
        )


def test_real_r05_defer_advances_only_due_cursor_and_consumes_tape():
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    prepared_capability, preview_capability = object(), object()
    base = torch.tensor([3, 5], dtype=torch.int64)
    business_names = (
        "_target_generation", "_previous_cell_index", "_scheduled_ordinal",
        "_outcome_shot_index", "_task_identity", "_outcome_identity",
        "_ball_identity", "_sequence_kind", "_policy_opportunity",
    )
    live = {
        "_rng_lo": base.clone(), "_rng_hi": base.add(10),
        "_draw_count": base.add(20),
        **{name: base.add(100 + index) for index, name in enumerate(business_names)},
        "_next_outcome_identity": torch.tensor(40, dtype=torch.int64),
        "_next_ball_identity": torch.tensor(50, dtype=torch.int64),
    }
    owner._publication = d05._PublicationState(
        live=live,
        registries={
            "_preview_records": {preview_capability: object()},
            "_prepared_records": {prepared_capability: object()},
        },
        counters={"_active": preview_capability},
        journal_rows={},
    )
    prepared = SimpleNamespace(
        capability=prepared_capability,
        rng_advance_mask=torch.ones(2, dtype=torch.bool),
        rng_before_lo=live["_rng_lo"].clone(),
        rng_after_lo=live["_rng_lo"].add(1),
        rng_before_hi=live["_rng_hi"].clone(),
        rng_after_hi=live["_rng_hi"].add(1),
        draw_before=live["_draw_count"].clone(),
        draw_after=live["_draw_count"].add(1),
        generation_before=live["_target_generation"].clone(),
        generation_after=live["_target_generation"].add(1),
        previous_before=live["_previous_cell_index"].clone(),
        selected_cell=live["_previous_cell_index"].add(1),
        ordinal_before=live["_scheduled_ordinal"].clone(),
        outcome_before=live["_outcome_shot_index"].clone(),
        projection=SimpleNamespace(
            scheduled_ordinal=live["_scheduled_ordinal"].add(1),
            outcome_shot_index=live["_outcome_shot_index"].add(1),
            task_identity=live["_task_identity"].add(1),
        ),
        reserved_outcome_identity=live["_outcome_identity"].add(1),
        reserved_ball_identity=live["_ball_identity"].add(1),
        outcome_identity_highwater_before=live["_next_outcome_identity"].clone(),
        ball_identity_highwater_before=live["_next_ball_identity"].clone(),
        identity_advance_count=torch.tensor(2, dtype=torch.int64),
        stage="previewed",
    )
    preview = SimpleNamespace(
        capability=preview_capability, prepared=prepared, stage="previewed"
    )
    before = {
        name: owner._publication.live[name].view(torch.uint8).clone()
        for name in business_names
    }
    owner._publish_action_epoch_afterimage(
        preview,
        accepted=torch.zeros(2, dtype=torch.bool),
        settled_due=torch.tensor([True, False], dtype=torch.bool),
    )
    identity_names = tuple(
        name
        for name in business_names
        if name not in {"_scheduled_ordinal", "_outcome_shot_index"}
    )
    assert all(
        torch.equal(owner._publication.live[name].view(torch.uint8), value)
        for name, value in before.items()
        if name in identity_names
    )
    assert owner._publication.live["_scheduled_ordinal"].tolist() == [106, 107]
    assert owner._publication.live["_outcome_shot_index"].tolist() == [107, 108]
    assert owner._publication.live["_rng_lo"].tolist() == [4, 6]
    assert owner._publication.live["_next_outcome_identity"].item() == 42
    assert owner._publication.counters["_active"] is None
    source = inspect.getsource(d05.DeviceR05Owner._publish_action_epoch_afterimage)
    assert "_mutation_version" not in source
    assert "_checkpoint_requires_global_drain_ack" not in source


def test_public_signatures_expose_no_mask_index_or_verdict():
    assert tuple(inspect.signature(
        d05.DeviceR05Owner.advance_action_ball_full_mdp_rows
    ).parameters) == ("self",)
    assert tuple(inspect.signature(
        d05.DeviceR05Owner.require_owned_action_epoch_candidate
    ).parameters) == ("self", "token")
    accepted = inspect.signature(
        d05.DeviceR05Owner.require_owned_action_epoch_accepted
    ).parameters
    assert tuple(accepted) == ("self", "token", "owner_kind")
    assert accepted["owner_kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "DeviceCadenceProjection" not in d05.__all__
    assert (
        "selected_env_index"
        in d05.DeviceCadenceProjection.__dataclass_fields__
    )


def test_current_row_cadence_publishes_canonical_full_n_row_identity():
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._num_envs = 2
    owner._device = torch.device("cpu")
    owner._row_axis = torch.arange(2, dtype=torch.int64)
    owner._reset_generation = torch.tensor([3, 4], dtype=torch.int64)
    owner._scheduled_ordinal = torch.tensor([10, 20], dtype=torch.int64)
    owner._outcome_shot_index = torch.tensor([30, 40], dtype=torch.int64)
    owner._target_generation = torch.tensor([50, 60], dtype=torch.int64)
    due_mask = torch.tensor([True, False], dtype=torch.bool)
    due = epoch.ActionEpochDueRows(7, due_mask, due_mask.clone())
    motion = SimpleNamespace(
        common_step=7,
        episode_tick=torch.tensor([8, 8], dtype=torch.int64),
        scheduled_ordinal=torch.tensor([11, 21], dtype=torch.int64),
        reveal_tick=torch.tensor([8, 8], dtype=torch.int64),
        deadline_tick=torch.tensor([12, 12], dtype=torch.int64),
        next_reveal_tick=torch.tensor([48, 48], dtype=torch.int64),
        reset_generation=torch.tensor([3, 4], dtype=torch.int64),
        swing_generation=torch.tensor([1, 1], dtype=torch.int64),
        reveal_due=due_mask.clone(),
        ready_at_reveal=torch.tensor([True, False], dtype=torch.bool),
    )

    cadence = owner._current_row_cadence(due, motion)

    assert cadence.selected_count == 2
    assert cadence.selected_env_index.tolist() == [0, 1]
    assert cadence.selected_env_index.dtype == torch.int64
    assert cadence.selected_env_index.is_contiguous()
    assert cadence.selected_env_index.data_ptr() != owner._row_axis.data_ptr()
    assert cadence.cadence_producer_fault.tolist() == [0, 0]


def _owner_with_active_prepared_during_internal_composition():
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._authority_callback_active = False
    owner._authority_reentry_detected = False
    owner._question_composition_in_progress = True
    token = object.__new__(d05.DeviceR05PreparedToken)
    prepared = SimpleNamespace(capability=token, stage="prepared")
    owner._prepared_records = {token: prepared}
    owner._preview_records = {}
    owner._active = token
    return owner, token, prepared


def test_public_preview_still_rejects_during_internal_composition():
    owner, token, prepared = (
        _owner_with_active_prepared_during_internal_composition()
    )

    with pytest.raises(
        d05.DeviceR05ConflictError,
        match="D05-internal question composition is in progress",
    ):
        owner.preview(token)

    assert prepared.stage == "prepared"
    assert owner._active is token
    assert owner._preview_records == {}


def test_full_n_internal_composition_uses_private_preview_transition():
    owner, token, prepared = (
        _owner_with_active_prepared_during_internal_composition()
    )

    preview = owner._preview_impl(token)

    assert type(preview) is d05.DeviceR05PreviewToken
    assert prepared.stage == "previewed"
    assert owner._active is preview
    assert owner._preview_records[preview].prepared is prepared

    source = textwrap.dedent(inspect.getsource(
        d05.DeviceR05Owner.advance_action_ball_full_mdp_rows
    ))
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "_preview_impl" in calls
    assert "preview" not in calls
    assert "_preview_impl" not in d05.__all__


def test_d05_poison_calls_are_exact_owner_attributed():
    source = textwrap.dedent(inspect.getsource(
        d05.DeviceR05Owner.advance_action_ball_full_mdp_rows
    ))
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "poison_owner_write"
    ]
    assert len(calls) == 2
    assert {call.args[1].value for call in calls} == {22, 24}
    for call in calls:
        assert call.args[0].value == "r05_runtime"
        assert len(call.keywords) == 1
        keyword = call.keywords[0]
        assert keyword.arg == "owner"
        assert isinstance(keyword.value, ast.Name) and keyword.value.id == "self"


def test_epoch_poison_runtime_rejects_foreign_d05_owner():
    current_epoch = d05._require_action_epoch_module()
    epoch_owner = current_epoch.ActionEpochOwner(num_envs=1, device="cpu")
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(1, dtype=torch.bool),
        reset_generation=torch.zeros(1, dtype=torch.int64),
    )
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    epoch_owner.bind_d05_accept_writers(
        motion_write=owner._commit_action_epoch_motion_write,
        racket_write=owner._commit_action_epoch_racket_write,
        r05_write=owner._commit_action_epoch_r05_write,
    )
    with pytest.raises(current_epoch.ActionEpochError):
        epoch_owner.poison_owner_write("r05_runtime", 22, owner=object())
    assert not epoch_owner.poisoned
    epoch_owner.poison_owner_write("r05_runtime", 22, owner=owner)
    assert epoch_owner.poisoned


def test_question_rng_is_unreachable_before_epoch_due_freeze():
    current_epoch = d05._require_action_epoch_module()
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._question_prepare_lock = RLock()
    owner._require_idle = lambda: None
    calls = []

    def bomb(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Question/RNG ran before due freeze")

    owner._internal_question_compose = bomb
    epoch_owner = current_epoch.ActionEpochOwner(
        num_envs=1,
        device=torch.device("cpu"),
        shot_slot_capacity=1,
        initial_reset_generation=torch.ones(1, dtype=torch.int64),
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(1, dtype=torch.bool),
        reset_generation=torch.ones(1, dtype=torch.int64),
    )
    owner._diagnostic_epoch_owner = epoch_owner
    with pytest.raises(current_epoch.ActionEpochError):
        owner.advance_action_ball_full_mdp_rows()
    assert calls == []


def test_k0_epoch_return_skips_question_numeric_writers_and_second_projection(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, _FAKE_R06.__name__, _FAKE_R06)
    current_epoch = d05._require_action_epoch_module()
    paid_rows_type = _PreviousPaidRows
    epoch_owner = current_epoch.ActionEpochOwner(num_envs=2, device="cpu")
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    owner = d05.DeviceR05Owner.__new__(d05.DeviceR05Owner)
    owner._question_prepare_lock = RLock()
    owner._require_idle = lambda: None
    owner._diagnostic_epoch_owner = epoch_owner
    epoch_owner.bind_d05_accept_writers(
        motion_write=owner._commit_action_epoch_motion_write,
        racket_write=owner._commit_action_epoch_racket_write,
        r05_write=owner._commit_action_epoch_r05_write,
    )

    class _IdleMotion:
        def __init__(self):
            self.calls = 0

        def project_current_action_epoch_rows(self):
            self.calls += 1
            if self.calls != 1:
                raise AssertionError("idle path requested a second Motion projection")
            return SimpleNamespace(
                common_step=1,
                reveal_due=torch.zeros(2, dtype=torch.bool),
                closed_mask=torch.zeros(2, dtype=torch.bool),
                close_reason=torch.zeros(2, dtype=torch.int64),
            )

    motion = _IdleMotion()
    epoch_owner.bind_motion_cadence_owner(motion)
    r06 = _EmptyR06(epoch_owner, current_epoch, paid_rows_type)
    epoch_owner.bind_fact_owner("r06_landing_outcome", r06)
    epoch_owner.bind_async_owner("r06_landing_outcome", r06)
    def bomb(*_args, **_kwargs):
        raise AssertionError("K0 entered Question/RNG numeric composition")

    owner._internal_question_compose = bomb
    owner._cadence_authority = motion
    for name in (
        "_current_row_cadence",
        "_prepare_many_impl",
        "_preview_impl",
        "_build_row_transaction",
    ):
        setattr(owner, name, bomb)
    head_before = epoch_owner.commit_head

    assert owner.advance_action_ball_full_mdp_rows() is None

    assert motion.calls == 1
    assert epoch_owner.commit_head == head_before


def test_canonical_epoch_idle_gate_needs_no_pre_materialized_commit_log():
    current_epoch = d05._require_action_epoch_module()
    epoch_owner = current_epoch.ActionEpochOwner(
        num_envs=2,
        device=torch.device("cpu"),
        shot_slot_capacity=1,
        initial_reset_generation=torch.ones(2, dtype=torch.int64),
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.ones(2, dtype=torch.int64),
    )
    assert not hasattr(epoch_owner, "commit_log")
    d05._require_canonical_action_epoch_idle(
        epoch_owner,
        epoch_module=current_epoch,
        device=torch.device("cpu"),
        num_envs=2,
    )
    gate_source = inspect.getsource(d05._require_canonical_action_epoch_idle)
    assert "commit_log" not in gate_source


def test_hot_sources_allow_only_the_profiled_question_compaction_boundary():
    transaction_path = (
        SOURCE / "action_ball_continuous_runtime_transaction_device.py"
    )
    cadence_path = SOURCE / "action_ball_motion_cadence_device.py"
    question_path = SOURCE / "action_ball_full_mdp_canary_question_owner.py"
    transaction_source = transaction_path.read_text(encoding="utf-8")
    cadence_source = cadence_path.read_text(encoding="utf-8")
    question_source = question_path.read_text(encoding="utf-8")

    # Cadence and transaction ownership remain static full-row surfaces.  The
    # sole exception is the question composer's explicit mask-first seam: one
    # dynamic row list is allowed to avoid sending all 4096 rows through the
    # much larger LM/exact/Physical numeric stack.  Its synchronization cost is
    # intentionally visible to the bounded real-run profiler and no second
    # compaction may spread into the hot path unnoticed.
    for source in (transaction_source, cadence_source):
        for forbidden in ("masked_select", "torch.nonzero", ".nonzero("):
            assert forbidden not in source
    assert "masked_select" not in question_source
    assert "torch.nonzero" not in question_source
    assert question_source.count(".nonzero(") == 1
    assert (
        "active_index = construction_mask.nonzero(as_tuple=False).reshape(-1)"
        in question_source
    )

    assert "def issue_current_r05_cadence_if_due" not in cadence_source
    assert "def project_r05_cadence" not in cadence_source
    d05_source = transaction_source
    assert "def prepare_many(" not in d05_source
    assert "def prepare_many_from_internal_question" not in d05_source
    assert "def settle_action_ball_full_mdp_epoch" not in d05_source
