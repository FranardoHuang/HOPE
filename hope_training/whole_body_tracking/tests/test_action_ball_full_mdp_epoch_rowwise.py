"""Focused CPU semantics for the row-wise ActionEpoch replacement."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, replace
import importlib
import inspect
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (SOURCE_ROOT, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import canonical package-qualified classes without executing the repository
# package initializer (which imports IsaacLab).  These are namespace shells,
# not replacement owner modules; each production module below is still loaded
# from its exact source path and registered under its canonical key.
_PACKAGE_PATHS = {
    "whole_body_tracking": SOURCE_ROOT / "whole_body_tracking",
    "whole_body_tracking.tasks": SOURCE_ROOT / "whole_body_tracking" / "tasks",
    "whole_body_tracking.tasks.tracking": (
        SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking"
    ),
    "whole_body_tracking.tasks.tracking.mdp": MDP,
}
for _package_name, _package_path in _PACKAGE_PATHS.items():
    _package = sys.modules.get(_package_name)
    if _package is None:
        _package = types.ModuleType(_package_name)
        sys.modules[_package_name] = _package
    if not getattr(_package, "__path__", None):
        _package.__path__ = [str(_package_path)]

D05 = importlib.import_module(
    "action_ball_continuous_runtime_transaction_device"
)
E = D05._require_action_epoch_module()
AS = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_action_strata"
)
LEAN = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_runtime"
)
LEAN_REWARDS = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_rewards"
)
R06_MODULE = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_landing_outcome_device"
)
PHYSICAL_MODULE = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_physical_flight_device"
)
PreviousPaidActionEpochRows = R06_MODULE.PreviousPaidActionEpochRows
ActionEpochR06OutcomeRows = R06_MODULE.ActionEpochR06OutcomeRows
ActionEpochR06LaunchProjection = PHYSICAL_MODULE.ActionEpochR06LaunchProjection
ActionEpochR06PostPhysicsProjection = (
    PHYSICAL_MODULE.ActionEpochR06PostPhysicsProjection
)
FAKE_R06_MODULE = R06_MODULE
FAKE_PHYSICAL_MODULE = PHYSICAL_MODULE


def _key(values: torch.Tensor, valid: torch.Tensor) -> E.ActionEpochShotKey:
    def field(offset: int) -> torch.Tensor:
        return torch.where(valid, values + offset, torch.full_like(values, -1))

    return E.ActionEpochShotKey(
        reset_generation=torch.where(valid, torch.zeros_like(values), torch.full_like(values, -1)),
        ball_generation=field(10),
        action_uid=field(20),
        action_slot=torch.where(valid, torch.zeros_like(values), torch.full_like(values, -1)),
        shot_index=field(30),
        task_identity=field(40),
        outcome_identity=field(50),
        ball_identity=field(60),
    )


def _candidate(device: torch.device) -> E.ActionEpochD05CandidateProjection:
    values = torch.tensor([[1], [2]], dtype=torch.int64, device=device)
    construction = torch.tensor([[True], [False]], dtype=torch.bool, device=device)
    identity = E.EpochIdentityPayload(
        shot_key=_key(values, construction),
        scheduled_ordinal=values + 100,
        target_generation=values + 200,
        selected_cell=torch.zeros_like(values),
        candidate_identity=values + 300,
    )
    clocks = E.EpochClockPayload(
        reveal_tick=torch.full_like(values, 10),
        contact_tick=torch.full_like(values, 12),
        launch_tick=torch.full_like(values, 13),
        deadline_tick=torch.full_like(values, 20),
        next_reveal_tick=torch.full_like(values, 30),
    )
    return E.ActionEpochD05CandidateProjection(
        identity=identity,
        clocks=clocks,
        task=E.EpochTaskPayload(
            task_f32=torch.zeros((2, 1, E.TASK_F32_WIDTH), dtype=torch.float32, device=device),
            task_valid=construction.clone(),
        ),
        rng_counter=values + 400,
        construction_admissible=construction,
        playback_admissible=torch.ones_like(construction),
        owner_fault_bits=torch.zeros((2, 1, E.OWNER_COUNT), dtype=torch.int64, device=device),
    )


class _MotionLeaf:
    def __init__(self, owner):
        self.owner = owner
        self.calls = 0
        self.accepted_masks: list[torch.Tensor] = []
        self.fail = False

    def commit_action_ball_full_mdp_motion_epoch_rows(self, token: object) -> None:
        view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind="motion"
        )
        self.accepted_masks.append(view.task.task_valid.clone())
        self.calls += 1
        if self.fail:
            raise RuntimeError("leaf mutated then failed")

    def publish_action_ball_full_mdp_post_d05_observation(self) -> None:
        return None


class _RacketLeaf:
    def __init__(self, owner):
        self.owner = owner
        self.calls = 0
        self.accepted_masks: list[torch.Tensor] = []

    def commit_action_ball_full_mdp_racket_epoch_rows(self, token: object) -> None:
        view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind="racket"
        )
        self.accepted_masks.append(view.task.task_valid.clone())
        self.calls += 1


class _PhysicalLeaf:
    def __init__(self, owner):
        self.owner = owner
        self.calls = 0
        self.accepted_masks: list[torch.Tensor] = []

    def retain_action_epoch_launch(self, token: object) -> None:
        view = self.owner.require_owned_action_epoch_accepted(
            token, owner_kind="physical_ball"
        )
        self.accepted_masks.append(view.task.task_valid.clone())
        self.calls += 1


class _RealD05Harness:
    """Exact DeviceR05 callbacks around one private row record at a time."""

    def __init__(
        self,
        epoch: E.ActionEpochOwner,
        candidate: E.ActionEpochD05CandidateProjection,
    ) -> None:
        owner = D05.DeviceR05Owner.__new__(D05.DeviceR05Owner)
        owner._num_envs = 2
        owner._device = epoch.device
        owner._diagnostic_epoch_owner = epoch
        owner._active_diagnostic_epoch_leaf_writer = None
        owner._row_transaction_records = {}
        owner._active_row_transaction = None
        self.owner = owner
        self.candidate = candidate
        self.calls: list[str] = []
        self.accepted_masks: list[torch.Tensor] = []
        self.motion = _MotionLeaf(owner)
        self.racket = _RacketLeaf(owner)
        self.physical = _PhysicalLeaf(owner)
        owner._diagnostic_motion_owner = self.motion
        owner._diagnostic_racket_owner = self.racket
        owner._diagnostic_physical_owner = self.physical
        owner._publish_action_epoch_afterimage = self._publish_afterimage

    def _publish_afterimage(
        self,
        _preview,
        *,
        accepted: torch.Tensor,
        settled_due: torch.Tensor,
    ) -> None:
        assert torch.equal(accepted & settled_due, accepted)
        self.accepted_masks.append(accepted[:, None].clone())
        self.calls.append("r05_runtime")

    def bind(self) -> None:
        self.owner._diagnostic_epoch_owner.bind_d05_accept_writers(
            motion_write=self.owner._commit_action_epoch_motion_write,
            racket_write=self.owner._commit_action_epoch_racket_write,
            r05_write=self.owner._commit_action_epoch_r05_write,
        )

    def arm(self) -> object:
        token = object.__new__(D05.DeviceR05RowTransaction)
        prepared = types.SimpleNamespace(
            selected_target_xy_m=torch.ones(
                2, 2, dtype=torch.float32, device=self.owner._device
            )
        )
        record = D05._RowTransactionRecord(
            capability=token,
            candidate=self.candidate,
            prepared=prepared,
            preview=types.SimpleNamespace(prepared=prepared),
            due_mask=torch.ones(2, dtype=torch.bool, device=self.owner._device),
            construct_mask=torch.ones(2, dtype=torch.bool, device=self.owner._device),
            accept_mask=torch.ones(2, dtype=torch.bool, device=self.owner._device),
            reject_mask=torch.zeros(2, dtype=torch.bool, device=self.owner._device),
            defer_mask=torch.zeros(2, dtype=torch.bool, device=self.owner._device),
            censor_mask=torch.zeros(2, dtype=torch.bool, device=self.owner._device),
            candidate_consumed=False,
            accepted_consumers=set(),
            stage="settling",
        )
        self.owner._row_transaction_records = {token: record}
        self.owner._active_row_transaction = token
        return token


@dataclass
class _MotionProjection:
    common_step: int
    reveal_due: torch.Tensor
    closed_mask: torch.Tensor
    close_reason: torch.Tensor


class _MotionCadence:
    def __init__(
        self, device: torch.device, action_uids=(21,), family_codes=(1,)
    ):
        self.projection = _MotionProjection(
            1,
            torch.ones(2, dtype=torch.bool, device=device),
            torch.zeros(2, dtype=torch.bool, device=device),
            torch.zeros(2, dtype=torch.int64, device=device),
        )
        self.catalog = AS.ActionStrokeFamilyCatalog(action_uids, family_codes)

    def project_current_action_epoch_rows(self):
        return self.projection

    def project_action_stroke_family_catalog(self):
        return self.catalog.clone()


class _R06:
    def __init__(self, epoch: E.ActionEpochOwner, device: torch.device):
        self.epoch = epoch
        self.previous = PreviousPaidActionEpochRows(
            valid=torch.zeros(2, dtype=torch.bool, device=device),
            shot_key=_key(torch.zeros(2, dtype=torch.int64, device=device), torch.zeros(2, dtype=torch.bool, device=device)),
            publication_ordinal=torch.full((2,), -1, dtype=torch.int64, device=device),
            settlement_step=torch.full((2,), -1, dtype=torch.int64, device=device),
            payment_step=torch.full((2,), -1, dtype=torch.int64, device=device),
        )
        self.outcome: ActionEpochR06OutcomeRows | None = None
        self.consumed: E.ActionEpochClosedRows | None = None

    def project_previous_paid_action_epoch_rows(self):
        return self.previous

    def project_current_action_epoch_outcome_rows(self):
        assert self.outcome is not None
        return self.outcome

    def consume_closed_action_epoch_rows(self):
        self.consumed = self.epoch.project_current_closed_action_epoch_rows(owner=self)


class _Physical:
    def __init__(self):
        self.launch: ActionEpochR06LaunchProjection | None = None
        self.launch_calls = 0

    def action_epoch_r06_launch_projection(self):
        self.launch_calls += 1
        assert self.launch is not None
        return self.launch


class _PhysicalProjectionOwner(_Physical):
    def __init__(self):
        super().__init__()
        self.projection: ActionEpochR06PostPhysicsProjection | None = None

    def require_owned_action_epoch_r06_postphysics_projection(self):
        assert self.projection is not None
        return self.projection


def _launch_packet(
    record: E.ActionEpochRecord,
    *,
    due: torch.Tensor,
    late: torch.Tensor | None = None,
    target_xy_m: torch.Tensor | None = None,
) -> ActionEpochR06LaunchProjection:
    key = E.ActionEpochShotKey(
        **{
            field.name: getattr(record.identity.shot_key, field.name)[:, 0].clone()
            for field in fields(E.ActionEpochShotKey)
        }
    )
    device = due.device
    exact_late = torch.zeros_like(due) if late is None else late.clone()
    exact_target = (
        torch.zeros((2, 2), dtype=torch.float32, device=device)
        if target_xy_m is None
        else target_xy_m.clone()
    )
    return ActionEpochR06LaunchProjection(
        selected_mask=due.clone(),
        due=due.clone(),
        late_launch=exact_late,
        flight_slot=torch.where(
            due,
            torch.zeros(2, dtype=torch.int64, device=device),
            torch.full((2,), -1, dtype=torch.int64, device=device),
        ),
        shot_key=key,
        publication_ordinal=record.publication_ordinal[:, 0].clone(),
        target_xy_m=exact_target,
        launch_control_step=torch.where(
            due,
            torch.ones(2, dtype=torch.int64, device=device),
            torch.full((2,), -1, dtype=torch.int64, device=device),
        ),
        contact_deadline_control_step=torch.where(
            due,
            torch.full((2,), 2, dtype=torch.int64, device=device),
            torch.full((2,), -1, dtype=torch.int64, device=device),
        ),
        crossing_horizon_control_step=torch.where(
            due,
            torch.full((2,), 3, dtype=torch.int64, device=device),
            torch.full((2,), -1, dtype=torch.int64, device=device),
        ),
        physical_owner=object(),
        epoch_owner=object(),
        owner_identity=object(),
        _token=PHYSICAL_MODULE._ACTION_EPOCH_R06_LAUNCH_TOKEN,
    )


class _PlaybackOwner:
    def __init__(self, epoch: E.ActionEpochOwner):
        self.epoch = epoch
        self.mask = torch.tensor(
            [[True], [False]], dtype=torch.bool, device=epoch.device
        )
        self.fail = False

    def action_epoch_playback_transition_mask(self, _kind, _record):
        if self.fail:
            raise E.ActionEpochError("clip identity differs")
        return self.mask


class _NoHostTensorObservation(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        if str(func) in ("aten._local_scalar_dense.default", "aten.item.default"):
            raise AssertionError("hot ActionEpoch path observed a tensor scalar")
        return func(*args, **(kwargs or {}))


def _ready_epoch(
    *, reward_age: int = 0, bind_playback: bool = False, device="cpu",
    catalog_uids=(21,), family_codes=(1,),
):
    exact_device = torch.device(device)
    epoch = E.ActionEpochOwner(
        num_envs=2, device=exact_device, initial_reward_cycle_age=reward_age
    )
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=exact_device),
        reset_generation=torch.zeros(2, dtype=torch.int64, device=exact_device),
    )
    playback = _PlaybackOwner(epoch)
    if bind_playback:
        epoch.bind_motion_playback_owner(playback)
    candidate = _candidate(exact_device)
    d05 = _RealD05Harness(epoch, candidate)
    d05.bind()
    cadence = _MotionCadence(exact_device, catalog_uids, family_codes)
    epoch.bind_motion_cadence_owner(cadence)
    r06 = _R06(epoch, exact_device)
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    physical = _Physical()
    epoch.bind_fact_owner("physical_ball", physical)
    epoch.bind_async_owner("physical_ball", physical)
    return (
        epoch, d05, cadence, r06, playback, d05.motion, d05.racket, physical
    )


@pytest.mark.parametrize(
    ("catalog_uids", "family_codes", "family", "attributed"),
    (((21,), (2,), 2, True), ((999,), (1,), 0, False)),
)
def test_d05_action_family_requires_exact_slot_uid_join(
    catalog_uids, family_codes, family, attributed
):
    epoch, d05, cadence, *_ = _ready_epoch(
        catalog_uids=catalog_uids, family_codes=family_codes
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    settled = next(
        entry for entry in _materialized_entries(epoch)
        if entry.transition == "D05_SETTLED"
    )
    values = dict(zip(settled.delta.names, settled.delta.values))
    assert int(values["stroke_family_code"][0, 0]) == family
    assert bool(values["action_attribution_valid"][0, 0]) is attributed
    assert "motion_file" not in inspect.getsource(type(cadence))


def _materialized_entries(epoch: E.ActionEpochOwner) -> tuple[E.CommitEntry, ...]:
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert not bool(materialized.overflow.any())
    epoch.acknowledge_drain(start=start, end=end)
    return materialized.entries


def test_k0_after_command_advances_only_scalar_chronology():
    epoch, d05, cadence, r06, *_ = _ready_epoch()
    cadence.projection = _MotionProjection(
        1,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    before = epoch.current()
    head_before = epoch.commit_head

    with _NoHostTensorObservation():
        rows = epoch.prepare_after_command_rows()

    assert rows is None
    after = epoch.current()
    assert (after.epoch, after.version) == (before.epoch, before.version)
    _assert_record_tensors_equal(after, before)
    assert epoch.commit_head == head_before
    assert epoch._next_epoch == 1
    assert epoch._last_motion_common_step == 1
    assert d05.motion.calls == d05.racket.calls == 0
    assert d05.calls == []
    assert r06.consumed is None


def test_k0_then_kpositive_fixed_tape_preserves_counters_reason_and_rng():
    epoch, d05, cadence, r06, *_ = _ready_epoch()
    cadence.projection = _MotionProjection(
        1,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    assert epoch.prepare_after_command_rows() is None

    cadence.projection = _MotionProjection(
        2,
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([False, True]),
        torch.tensor([E.MOTION_CLOSE_NONE, E.MOTION_CLOSE_UNPLAYED]),
    )
    assert epoch.prepare_after_command_rows() is not None
    assert epoch._undecoded_overflow.tolist() == [False, True]
    assert r06.consumed is not None
    epoch.abort_d05_transaction(owner=d05.owner)

    cadence.projection = _MotionProjection(
        3,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    assert epoch.prepare_after_command_rows() is None
    assert epoch._undecoded_overflow.tolist() == [False, True]

    cadence.projection = _MotionProjection(
        4,
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )

    rows = epoch.prepare_after_command_rows()
    assert rows is not None
    assert rows.common_step == 4
    assert rows.due_mask.tolist() == [True, False]
    assert rows.construct_mask.tolist() == [True, False]
    assert epoch.settle_d05_transaction(d05.arm()) is None

    current = epoch.current()
    assert current.publication_ordinal[:, 0].tolist() == [3, -1]
    assert current.rng_counter[:, 0].tolist() == [401, -1]
    base = E.milestone_tensors._EI
    assert epoch.milestone.i64[base:base + 4].tolist() == [1, 1, 1, 1]
    settled = next(
        entry
        for entry in epoch._publication.pending_log
        if entry.transition == "D05_SETTLED"
    )
    decision = dict(zip(settled.delta.names, settled.delta.values))["decision"]
    assert decision[:, 0].tolist() == [E.D05_DECISION_ACCEPT, E.D05_DECISION_NONE]
    assert settled.epoch == 3
    assert d05.motion.calls == d05.racket.calls == 1
    assert d05.calls == ["r05_runtime"]


def test_previous_paid_row_is_business_even_when_motion_is_not_due():
    epoch, d05, cadence, r06, *_ = _ready_epoch()
    cadence.projection = _MotionProjection(
        1,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    r06.previous = replace(
        r06.previous,
        valid=torch.tensor([True, False]),
        payment_step=torch.tensor([1, -1], dtype=torch.int64),
    )
    head_before = epoch.commit_head

    rows = epoch.prepare_after_command_rows()

    assert rows is not None
    assert not bool(rows.due_mask.any())
    assert epoch.commit_head > head_before
    assert epoch._last_r06_paid_payment_step.tolist() == [1, -1]
    assert r06.consumed is not None
    assert r06.consumed.valid.tolist() == [False, False]
    epoch.abort_d05_transaction(owner=d05.owner)


def test_accept_is_masked_reject_is_event_only_and_payment_uses_control_step(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    device = torch.device("cpu")
    epoch = E.ActionEpochOwner(num_envs=2, device=device)
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    genesis = epoch.current()
    candidate = _candidate(device)
    d05 = _RealD05Harness(epoch, candidate)
    d05.bind()
    cadence = _MotionCadence(device)
    epoch.bind_motion_cadence_owner(cadence)
    r06 = _R06(epoch, device)
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    physical = _Physical()
    epoch.bind_fact_owner("physical_ball", physical)
    epoch.bind_async_owner("physical_ball", physical)
    playback = _PlaybackOwner(epoch)
    epoch.bind_motion_playback_owner(playback)

    due = epoch.prepare_after_command_rows()
    assert torch.equal(due.due_mask, torch.tensor([True, True]))
    epoch.settle_d05_transaction(d05.arm())
    published = epoch.current()
    assert published.epoch == -1  # scalar compatibility value is never a join key
    assert published.phase[:, 0].tolist() == [E.PHASE_REVEAL_COMMITTED, E.PHASE_IDLE]
    assert published.publication_ordinal[:, 0].tolist() == [0, -1]
    assert torch.equal(published.identity.shot_key.action_uid[1], genesis.identity.shot_key.action_uid[1])
    settled = next(
        entry for entry in _materialized_entries(epoch)
        if entry.transition == "D05_SETTLED"
    )
    decision = settled.delta.values[11]
    assert decision[:, 0].tolist() == [E.D05_DECISION_ACCEPT, E.D05_DECISION_REJECT]
    assert d05.motion.calls == d05.racket.calls == 1
    assert d05.calls == ["r05_runtime"]
    with pytest.raises(E.ActionEpochError):
        epoch.require_active_d05_accepted_rows(
            d05, owner_kind="r05_runtime"
        )
    epoch.publish_motion_playback_started(owner=playback)

    physical.launch = _launch_packet(
        epoch.current(), due=torch.tensor([True, False])
    )
    epoch.refresh_physical_launch_rows()
    epoch.merge_runtime_owner_fault(
        "r06_landing_outcome",
        torch.tensor([[4], [0]], dtype=torch.int64),
        owner=r06,
    )
    current = epoch.current()
    key_rows = E.ActionEpochShotKey(**{
        name: getattr(current.identity.shot_key, name)[:, 0].clone()
        for name in current.identity.shot_key.__dataclass_fields__
    })
    r06.outcome = ActionEpochR06OutcomeRows(
        valid=torch.tensor([True, False]),
        shot_key=key_rows,
        publication_ordinal=current.publication_ordinal[:, 0].clone(),
        settlement_step=torch.tensor([25, -1], dtype=torch.int64),
        valid_bits=torch.tensor([1, 0], dtype=torch.int64),
        fact_values=torch.zeros((2, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32),
        outcome_code=torch.tensor([2, -1], dtype=torch.int64),
        owner_fault_bits=torch.zeros(2, dtype=torch.int64),
    )
    refresh_result = epoch.refresh_r06_outcome_rows()
    assert refresh_result is None
    outcome_record = epoch.current()
    r06_slot = E.OWNER_ORDER.index("r06_landing_outcome")
    assert outcome_record.owner_fault_bits[0, 0, r06_slot].item() == 4
    epoch.open_reward_cycle()
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    payment = epoch.publish_reward_payment(77)
    assert payment.valid.tolist() == [True, False]
    assert payment.payment_step.tolist() == [77, -1]
    paid_record = epoch.current()
    r06.previous = PreviousPaidActionEpochRows(
        payment.valid.clone(), payment.shot_key.clone(),
        paid_record.publication_ordinal[:, 0].clone(),
        paid_record.settlement_step[:, 0].clone(),
        payment.payment_step.clone(),
    )
    # Advance only the peer row to a newer publication before delayed row-0 close.
    peer_construct = torch.tensor([[False], [True]], dtype=torch.bool)
    peer_values = torch.tensor([[9], [10]], dtype=torch.int64)
    peer = _candidate(device)
    d05.candidate = replace(
        peer,
        identity=replace(peer.identity, shot_key=_key(peer_values, peer_construct)),
        task=replace(peer.task, task_valid=peer_construct),
        construction_admissible=peer_construct,
    )
    cadence.projection = _MotionProjection(
        80,
        torch.tensor([False, True]),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    assert epoch.current().publication_ordinal[:, 0].tolist() == [0, 1]
    retire_peer_before = _peer_bytes(epoch.current())
    cadence.projection = _MotionProjection(
        100,
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([True, False]),
        torch.tensor([E.MOTION_CLOSE_PLAYED_SUFFIX, E.MOTION_CLOSE_NONE], dtype=torch.int64),
    )
    epoch.prepare_after_command_rows()
    retire_peer_after = _peer_bytes(epoch.current())
    assert retire_peer_before.keys() == retire_peer_after.keys()
    assert all(
        torch.equal(retire_peer_before[name], retire_peer_after[name])
        for name in retire_peer_before
    )
    assert epoch.current().phase[:, 0].tolist() == [E.PHASE_RETIRED, E.PHASE_REVEAL_COMMITTED]
    assert r06.consumed is not None and r06.consumed.valid.tolist() == [True, False]
    assert epoch.project_current_reward_payment_rows().valid.tolist() == [False, False]
    epoch.settle_d05_transaction(d05.arm())
    assert d05.motion.calls == d05.racket.calls == 3  # peer + neutral callbacks

    retirement_entries = _materialized_entries(epoch)
    retired = next(
        entry
        for entry in retirement_entries
        if entry.transition == "RETIRED" and bool(entry.delta.values[0].any())
    )
    retired_delta = dict(zip(retired.delta.names, retired.delta.values))
    assert retired.delta.names[-3:] == (
        "motion_close_reason", "payment_step", "retirement_step"
    )
    assert retired_delta["event_mask"].tolist() == [[True], [False]]
    assert retired_delta["payment_step"].tolist() == [[77], [-1]]
    assert retired_delta["retirement_step"].tolist() == [[100], [-1]]
    assert torch.all(
        retired_delta["payment_step"][retired_delta["event_mask"]]
        <= retired_delta["retirement_step"][retired_delta["event_mask"]]
    )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_physical_launch_journals_keyed_target_and_preserves_peer_bytes(
    monkeypatch, device
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, _cadence, _r06, _playback, *_middle, physical = _ready_epoch(
        device=device
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    before = epoch.current()
    peer_before = _peer_bytes(before)
    exact_device = torch.device(device)
    target = torch.tensor(
        [[0.31, -0.27], [9.0, 8.0]],
        dtype=torch.float32,
        device=exact_device,
    )
    physical.launch = _launch_packet(
        before,
        due=torch.tensor([True, False], dtype=torch.bool, device=exact_device),
        target_xy_m=target,
    )
    with _NoHostTensorObservation():
        refresh_result = epoch.refresh_physical_launch_rows()
    assert refresh_result is None
    after = epoch.current()
    peer_after = _peer_bytes(after)
    assert peer_before.keys() == peer_after.keys()
    assert all(
        torch.equal(peer_before[name], peer_after[name]) for name in peer_before
    )

    launch = next(
        entry for entry in _materialized_entries(epoch)
        if entry.transition == "PHYSICAL_LAUNCH_ROWS"
    )
    delta = dict(zip(launch.delta.names, launch.delta.values))
    assert launch.delta.names[-5:] == (
        "publication_ordinal",
        "launch_succeeded",
        "late_launch",
        "owner_fault_bits",
        "target_xy_m",
    )
    assert delta["event_mask"].tolist() == [[True], [False]]
    assert delta["target_xy_m"].shape == (2, 1, 2)
    assert delta["target_xy_m"].is_contiguous()
    assert torch.equal(delta["target_xy_m"][0, 0], target[0].cpu())
    assert torch.equal(
        delta["target_xy_m"][1, 0], torch.zeros(2, dtype=torch.float32)
    )
    for field in fields(E.ActionEpochShotKey):
        assert torch.equal(
            delta["shot_key." + field.name][0, 0],
            getattr(before.identity.shot_key, field.name)[0, 0].cpu(),
        )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_nonfinite_physical_target_latches_before_launch_publication(
    monkeypatch, device
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, _cadence, _r06, _playback, *_middle, physical = _ready_epoch(
        device=device
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    before = epoch.current()
    before_rows = (_peer_bytes(before, row=0), _peer_bytes(before))
    exact_device = torch.device(device)
    target = torch.zeros((2, 2), dtype=torch.float32, device=exact_device)
    target[0, 0] = float("nan")
    physical.launch = _launch_packet(
        before,
        due=torch.tensor([True, False], dtype=torch.bool, device=exact_device),
        target_xy_m=target,
    )
    with _NoHostTensorObservation():
        epoch.refresh_physical_launch_rows()
    after = epoch.current()
    after_rows = (_peer_bytes(after, row=0), _peer_bytes(after))
    assert all(
        lhs.keys() == rhs.keys()
        and all(torch.equal(lhs[name], rhs[name]) for name in lhs)
        for lhs, rhs in zip(before_rows, after_rows)
    )
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.overflow.tolist() == [True, False]
    launch = next(
        entry for entry in materialized.entries
        if entry.transition == "PHYSICAL_LAUNCH_ROWS"
    )
    delta = dict(zip(launch.delta.names, launch.delta.values))
    assert not delta["event_mask"].any()
    assert torch.equal(delta["target_xy_m"], torch.zeros((2, 1, 2)))


@pytest.mark.parametrize(
    "malformed", ["shape", "dtype", "device", "noncontiguous"]
)
def test_structurally_malformed_physical_target_fails_before_any_mutation(
    monkeypatch, malformed
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, _cadence, _r06, _playback, *_middle, physical = _ready_epoch()
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    before = epoch.current()
    before_head = epoch.commit_head
    before_overflow = epoch._undecoded_overflow.clone()
    before_rows = (_peer_bytes(before, row=0), _peer_bytes(before))
    if malformed == "shape":
        target = torch.zeros((2, 3), dtype=torch.float32)
    elif malformed == "dtype":
        target = torch.zeros((2, 2), dtype=torch.float64)
    elif malformed == "noncontiguous":
        target = torch.zeros((2, 4), dtype=torch.float32)[:, ::2]
        assert not target.is_contiguous()
    else:
        target = torch.zeros((2, 2), dtype=torch.float32, device="meta")
    physical.launch = _launch_packet(
        before, due=torch.tensor([True, False])
    )
    physical.launch.target_xy_m = target
    with pytest.raises(E.ActionEpochError, match="target_xy_m"):
        epoch.refresh_physical_launch_rows()
    after = epoch.current()
    after_rows = (_peer_bytes(after, row=0), _peer_bytes(after))
    assert epoch.commit_head == before_head
    assert torch.equal(epoch._undecoded_overflow, before_overflow)
    assert all(
        lhs.keys() == rhs.keys()
        and all(torch.equal(lhs[name], rhs[name]) for name in lhs)
        for lhs, rhs in zip(before_rows, after_rows)
    )


class _RealSelectedResetHarness:
    def __init__(self, epoch: E.ActionEpochOwner) -> None:
        reward = LEAN_REWARDS.LeanActionEpochRewardGraph(epoch_owner=epoch)
        self.owner = LEAN.ActionBallFullMdpLeanRuntimeOwner(
            env=object(),
            runtime_lease=object(),
            epoch_owner=epoch,
            reward_graph=reward,
            r05_runtime=object(),
            motion=object(),
            racket=object(),
            physical_ball=object(),
            r06_landing_outcome=object(),
            r03_strike_fact=object(),
            r07_recovery=object(),
        )

    def arm_preflight(
        self,
        *,
        selected_env_index: torch.Tensor,
        selected_mask: torch.Tensor,
        generation_before: torch.Tensor,
        generation_after: torch.Tensor,
        generation_overflow_fault: torch.Tensor,
        terminal_reset_facts_i64: torch.Tensor,
    ) -> object:
        preflight = object.__new__(LEAN._SelectedResetPackedPreflight)
        self.owner._selected_reset_projection = types.SimpleNamespace(
            selected_env_index=selected_env_index,
            selected_mask=selected_mask,
            generation_before=generation_before,
            generation_after=generation_after,
            generation_overflow_fault=generation_overflow_fault,
            terminal_reset_facts_i64=terminal_reset_facts_i64,
        )
        self.owner._selected_reset_prepared = object()
        self.owner._selected_reset_epoch_prepared = None
        self.owner._selected_reset_packed_preflight = preflight
        return preflight

    def arm_commit(self, lease: object) -> None:
        self.owner._selected_reset_epoch_prepared = lease
        self.owner._selected_reset_r05_receipt = object()
        self.owner._selected_reset_completions = {}
        self.owner._selected_reset_leaf_completions_consumed = True


class _ForeignResetComposite:
    def selected_true_reset(self, _event, _projection):
        return None

    def require_owned_epoch_selected_reset_preflight(self, value, **_kwargs):
        return value

    def require_owned_epoch_selected_reset_commit(self, _value, lease):
        return lease


def _terminal_reset_facts(selected_mask: torch.Tensor) -> torch.Tensor:
    facts = torch.full(
        (selected_mask.shape[0], 3),
        -1,
        dtype=torch.int64,
        device=selected_mask.device,
    )
    facts[:, 2] = 0
    facts[selected_mask] = torch.tensor(
        [101, 7, 1 | 4], dtype=torch.int64, device=selected_mask.device
    )
    return facts.contiguous()


def _peer_bytes(
    record: E.ActionEpochRecord, *, row: int = 1
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for field in fields(record):
        value = getattr(record, field.name)
        if type(value) is torch.Tensor:
            result[field.name] = value[row].reshape(-1).view(torch.uint8).clone()
    for prefix, value in (
        ("identity", record.identity),
        ("shot_key", record.identity.shot_key),
        ("clocks", record.clocks),
        ("task", record.task),
    ):
        for field in fields(value):
            tensor = getattr(value, field.name)
            if type(tensor) is torch.Tensor:
                result[prefix + "." + field.name] = (
                    tensor[row].reshape(-1).view(torch.uint8).clone()
                )
    return result


def test_selected_reset_preserves_every_unselected_peer_byte(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    device = torch.device("cpu")
    epoch = E.ActionEpochOwner(num_envs=2, device=device)
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    reset = _RealSelectedResetHarness(epoch)
    epoch.bind_selected_reset_owner(reset.owner)
    base = _candidate(device)
    values = torch.tensor([[1], [2]], dtype=torch.int64)
    both = torch.ones((2, 1), dtype=torch.bool)
    task_values = base.task.task_f32.clone()
    special = torch.tensor([-2147483648, 2143294004], dtype=torch.int32).view(torch.float32)
    task_values[1, 0, :2] = special
    candidate = replace(
        base,
        identity=replace(base.identity, shot_key=_key(values, both)),
        task=replace(base.task, task_f32=task_values, task_valid=both),
        construction_admissible=both,
    )
    d05 = _RealD05Harness(epoch, candidate)
    d05.bind()
    cadence = _MotionCadence(device)
    epoch.bind_motion_cadence_owner(cadence)
    r06 = _R06(epoch, device)
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    before = _peer_bytes(epoch.current())

    selected_env_index = torch.tensor([0], dtype=torch.int64)
    selected_mask = torch.tensor([True, False])
    generation_before = torch.tensor([0, 0], dtype=torch.int64)
    generation_after = torch.tensor([1, 0], dtype=torch.int64)
    overflow = torch.tensor([False, False])
    terminal_reset_facts_i64 = _terminal_reset_facts(selected_mask)
    top = reset.arm_preflight(
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    lease = epoch.prepare_selected_true_reset(
        owner=reset.owner,
        top_preflight=top,
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    reset.arm_commit(lease)
    after = epoch.commit_selected_true_reset(
        owner=reset.owner, prepared_reset=lease
    )
    assert after.phase[:, 0].tolist() == [E.PHASE_IDLE, E.PHASE_REVEAL_COMMITTED]
    assert after.publication_ordinal[:, 0].tolist() == [-1, 0]
    assert not E.row_identity.action_epoch_shot_key_valid(
        E.ActionEpochShotKey(**{
            name: getattr(after.identity.shot_key, name)[0].clone()
            for name in after.identity.shot_key.__dataclass_fields__
        })
    ).any()
    reset_entries = tuple(
        entry
        for entry in _materialized_entries(epoch)
        if entry.transition == "RESET_SELECTED"
    )
    assert len(reset_entries) == 1
    reset_delta = reset_entries[0].delta
    assert reset_delta.names == (
        "selected_mask",
        "reset_generation",
        "terminal_reset_facts_i64",
    )
    assert torch.equal(reset_delta.values[2], terminal_reset_facts_i64)
    peer = _peer_bytes(after)
    assert before.keys() == peer.keys()
    assert all(torch.equal(before[name], peer[name]) for name in before)


def test_selected_reset_cannot_bypass_real_top_preflight():
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    bypass = _ForeignResetComposite()
    with pytest.raises(E.ActionEpochError):
        epoch.bind_selected_reset_owner(bypass)
    exact = _RealSelectedResetHarness(epoch)
    epoch.bind_selected_reset_owner(exact.owner)
    with pytest.raises(LEAN.ActionBallFullMdpLeanRuntimeError):
        epoch.prepare_selected_true_reset(
            owner=exact.owner,
            top_preflight=object(),
            selected_env_index=torch.tensor([0], dtype=torch.int64),
            selected_mask=torch.tensor([True, False]),
            generation_before=torch.tensor([0, 0], dtype=torch.int64),
            generation_after=torch.tensor([1, 0], dtype=torch.int64),
            generation_overflow_fault=torch.tensor([False, False]),
            terminal_reset_facts_i64=_terminal_reset_facts(
                torch.tensor([True, False])
            ),
        )
    assert epoch.commit_head == 1


def test_active_selected_reset_lease_blocks_reward_drain_and_checkpoint():
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    reset = _RealSelectedResetHarness(epoch)
    epoch.bind_selected_reset_owner(reset.owner)
    start, end = epoch.prepare_drain()
    epoch.materialize_drain(start=start, end=end)
    epoch.acknowledge_drain(start=start, end=end)
    selected_env_index = torch.tensor([0], dtype=torch.int64)
    selected_mask = torch.tensor([True, False])
    generation_before = torch.tensor([0, 0], dtype=torch.int64)
    generation_after = torch.tensor([1, 0], dtype=torch.int64)
    overflow = torch.tensor([False, False])
    terminal_reset_facts_i64 = _terminal_reset_facts(selected_mask)
    top = reset.arm_preflight(
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    lease = epoch.prepare_selected_true_reset(
        owner=reset.owner,
        top_preflight=top,
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    for operation in (
        epoch.open_reward_cycle,
        lambda: epoch.publish_reward_payment(1),
        epoch.prepare_drain,
        epoch.checkpoint,
    ):
        with pytest.raises(E.ActionEpochError):
            operation()
    epoch.abort_selected_true_reset(owner=reset.owner, prepared_reset=lease)
    assert epoch.checkpoint().current.phase.eq(E.PHASE_IDLE).all()


def _physical_packet(
    current: E.ActionEpochRecord, *, flight_index: int, contact: bool
) -> ActionEpochR06PostPhysicsProjection:
    device = current.phase.device
    key = E.row_identity.empty_action_epoch_shot_key((2, 2), device=device)
    for field in fields(key):
        getattr(key, field.name)[0, flight_index] = getattr(
            current.identity.shot_key, field.name
        )[0, 0]
    publication = torch.full((2, 2), -1, dtype=torch.int64, device=device)
    publication[0, flight_index] = current.publication_ordinal[0, 0]
    observe = torch.zeros((2, 2), dtype=torch.bool, device=device)
    observe[0, flight_index] = True
    bits = torch.zeros((2, 2), dtype=torch.int64, device=device)
    bits[0, flight_index] = 3 if contact else 1
    facts = torch.zeros(
        (2, 2, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32, device=device
    )
    if contact:
        facts[0, flight_index, 0:10] = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32, device=device
        )
    else:
        facts[0, flight_index, 0:3] = torch.tensor(
            [11, 12, 13], dtype=torch.float32, device=device
        )
        facts[0, flight_index, 9] = 20
    source = torch.full((2, 2), -1, dtype=torch.int64, device=device)
    source[0, flight_index] = 50 if contact else 60
    shape = (2, 2)
    zeros_bool = torch.zeros(shape, dtype=torch.bool, device=device)
    zeros_i64 = torch.zeros(shape, dtype=torch.int64, device=device)
    centers = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
    selected_contact = torch.zeros(shape, dtype=torch.bool, device=device)
    selected_contact[0, flight_index] = contact
    return ActionEpochR06PostPhysicsProjection(
        observe_mask=observe,
        flight_slot=torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.int64, device=device
        ),
        shot_key=key,
        publication_ordinal=publication,
        observation_ordinal=torch.where(observe, zeros_i64 + 1, zeros_i64),
        previous_ball_center_m=centers.clone(),
        current_ball_center_m=centers.clone(),
        observation_stamp=source.clone(),
        selected_contact_event=selected_contact,
        selected_contact_ball_center_m=centers.clone(),
        selected_contact_outgoing_segment_anchor_m=centers.clone(),
        selected_contact_stamp=source.clone(),
        net_crossing_event=zeros_bool.clone(),
        net_clear_at_crossing=zeros_bool.clone(),
        net_crossing_stamp=torch.full_like(source, -1),
        crossing_report_delivered=zeros_bool.clone(),
        first_descending_crossing_event=zeros_bool.clone(),
        first_descending_crossing_xy_m=torch.zeros(
            (*shape, 2), dtype=torch.float32, device=device
        ),
        first_descending_crossing_stamp=torch.full_like(source, -1),
        nonfinite_observation=zeros_bool.clone(),
        producer_contract_fault=zeros_bool.clone(),
        engine_overflow=zeros_bool.clone(),
        owner_fault_bits=zeros_i64.clone(),
        fact_valid_bits=bits,
        fact_source_step=source,
        fact_f32=facts,
        physical_owner=object(),
        epoch_owner=object(),
        _owner_identity=object(),
        _token=PHYSICAL_MODULE._ACTION_EPOCH_R06_POSTPHYSICS_TOKEN,
    )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_physical_launch_pull_rejects_stale_key_before_publication(
    monkeypatch, device
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, _cadence, _r06, _playback, _motion, _racket, physical = (
        _ready_epoch(device=device)
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    before = epoch.current()
    stale = _launch_packet(
        before, due=torch.tensor([True, False], device=torch.device(device))
    )
    stale.shot_key.ball_generation[0] += 1
    physical.launch = stale
    with _NoHostTensorObservation():
        epoch.refresh_physical_launch_rows()
    after = epoch.current()
    assert torch.equal(after.phase, before.phase)
    assert torch.equal(after.launch_succeeded, before.launch_succeeded)
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.overflow.tolist() == [True, False]


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_physical_k_grid_join_preserves_first_contact_across_later_miss(
    monkeypatch, device
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    exact_device = torch.device(device)
    epoch = E.ActionEpochOwner(num_envs=2, device=exact_device)
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=exact_device),
        reset_generation=torch.zeros(2, dtype=torch.int64, device=exact_device),
    )
    base = _candidate(exact_device)
    values = torch.tensor([[1], [2]], dtype=torch.int64, device=exact_device)
    both = torch.ones((2, 1), dtype=torch.bool, device=exact_device)
    candidate = replace(
        base,
        identity=replace(base.identity, shot_key=_key(values, both)),
        task=replace(base.task, task_valid=both),
        construction_admissible=both,
    )
    d05 = _RealD05Harness(epoch, candidate)
    d05.bind()
    cadence = _MotionCadence(exact_device)
    epoch.bind_motion_cadence_owner(cadence)
    r06 = _R06(epoch, exact_device)
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    physical = _PhysicalProjectionOwner()
    epoch.bind_fact_owner("physical_ball", physical)
    epoch.bind_async_owner("physical_ball", physical)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    physical.launch = _launch_packet(
        epoch.current(), due=torch.tensor([True, True], device=exact_device)
    )
    epoch.refresh_physical_launch_rows()

    physical.projection = _physical_packet(epoch.current(), flight_index=1, contact=True)
    epoch.refresh_physical_postphysics_rows()
    physical.projection = _physical_packet(epoch.current(), flight_index=0, contact=False)
    refresh_result = epoch.refresh_physical_postphysics_rows()
    assert refresh_result is None
    after = epoch.current()
    slot = E.OWNER_ORDER.index("physical_ball")
    assert after.fact_valid_bits[0, 0, slot].item() == 3
    assert after.fact_source_step[0, 0, slot].item() == 50
    facts = after.fact_f32[0, 0, slot]
    assert torch.equal(
        facts[0:3],
        torch.tensor([11, 12, 13], dtype=torch.float32, device=exact_device),
    )
    assert torch.equal(
        facts[3:9],
        torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.float32, device=exact_device),
    )
    assert facts[9].item() == 20
    duplicate = _physical_packet(epoch.current(), flight_index=0, contact=False)
    duplicate.observe_mask[0, 1] = True
    duplicate.publication_ordinal[0, 1] = duplicate.publication_ordinal[0, 0]
    for field in fields(duplicate.shot_key):
        getattr(duplicate.shot_key, field.name)[0, 1] = getattr(
            duplicate.shot_key, field.name
        )[0, 0]
    physical.projection = duplicate
    before_duplicate = epoch.current()
    with _NoHostTensorObservation():
        epoch.refresh_physical_postphysics_rows()
    after_duplicate = epoch.current()
    assert torch.equal(after_duplicate.fact_f32, before_duplicate.fact_f32)
    assert epoch.milestone.i64[E.milestone_tensors._EI + 5].item() == 1
    assert epoch.milestone.i64[E.milestone_tensors._EI + 6].item() == 1
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.overflow.tolist() == [True, False]


def test_drain_has_one_materialization_and_no_pre_materialized_tensor_surface():
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    prepared = epoch.prepare_drain()
    assert prepared == (0, 1)
    materialized = epoch.materialize_drain(start=0, end=1)
    assert type(materialized) is E.ActionEpochMaterializedDrain
    assert materialized.overflow.device.type == "cpu"
    assert materialized.milestone_i64.device.type == "cpu"
    assert materialized.milestone_f64.device.type == "cpu"
    assert materialized.entries[0].delta.names == (
        "reset_generation", "reset_selected_mask"
    )
    assert all(
        tensor.device.type == "cpu"
        for entry in materialized.entries
        for tensor in entry.delta.values
    )
    try:
        epoch.materialize_drain(start=0, end=1)
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("a frozen drain was materialized twice")
    epoch.acknowledge_drain(start=0, end=1)
    assert epoch.prepare_drain() == (1, 1)


def _pending_host_reference(epoch: E.ActionEpochOwner):
    entries = tuple(
        replace(
            entry,
            delta=E.PackedDelta(
                entry.delta.names,
                tuple(value.detach().clone().contiguous() for value in entry.delta.values),
            ),
        )
        for entry in epoch._publication.pending_log
    )
    return entries, epoch._undecoded_overflow.detach().clone().contiguous()


def _assert_materialized_bytes_equal(materialized, expected_entries, expected_overflow):
    assert tuple(
        (entry.sequence, entry.epoch, entry.transition, entry.before_version, entry.after_version)
        for entry in materialized.entries
    ) == tuple(
        (entry.sequence, entry.epoch, entry.transition, entry.before_version, entry.after_version)
        for entry in expected_entries
    )
    for actual, expected in zip(materialized.entries, expected_entries):
        assert actual.delta.names == expected.delta.names
        for actual_value, expected_value in zip(actual.delta.values, expected.delta.values):
            assert actual_value.dtype is expected_value.dtype
            assert actual_value.shape == expected_value.shape
            assert torch.equal(
                actual_value.reshape(-1).view(torch.uint8),
                expected_value.reshape(-1).view(torch.uint8),
            )
    assert torch.equal(
        materialized.overflow.reshape(-1).view(torch.uint8),
        expected_overflow.reshape(-1).view(torch.uint8),
    )
    assert materialized.milestone_i64.shape == (E.milestone_tensors.I64_NUMEL,)
    assert materialized.milestone_f64.shape == (E.milestone_tensors.F64_NUMEL,)


def test_drain_one_d2h_contains_every_byte_and_source_has_one_cpu_transfer(
    monkeypatch,
):
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    expected_entries, expected_overflow = _pending_host_reference(epoch)
    expected_bytes = torch.cat(
        tuple(
            value.contiguous().view(torch.uint8).reshape(-1)
            for entry in expected_entries
            for value in entry.delta.values
        )
        + tuple(value.view(torch.uint8).reshape(-1) for value in epoch.milestone.pack_views())
        + (expected_overflow.view(torch.uint8).reshape(-1),),
        dim=0,
    )
    calls = []
    real_transfer = E._single_d2h_packed_bytes

    def checked_transfer(device_bytes):
        calls.append(device_bytes)
        assert device_bytes.dtype is torch.uint8
        assert device_bytes.ndim == 1 and device_bytes.is_contiguous()
        assert torch.equal(device_bytes, expected_bytes)
        return real_transfer(device_bytes)

    monkeypatch.setattr(E, "_single_d2h_packed_bytes", checked_transfer)
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert len(calls) == 1
    _assert_materialized_bytes_equal(
        materialized, expected_entries, expected_overflow
    )

    tree = ast.parse(inspect.getsource(E))
    cpu_transfers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "to":
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "device"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "cpu"
            ):
                cpu_transfers.append(node)
    assert len(cpu_transfers) == 1
    materialize_source = inspect.getsource(E.ActionEpochOwner.materialize_drain)
    assert all(
        token not in materialize_source
        for token in (".cpu(", ".item(", ".tolist(", ".numpy(")
    )


def test_same_drain_carries_milestone_window_and_only_ack_clears_it():
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    epoch.milestone.add_step_return(torch.tensor([1.0, 2.0]))
    actual = torch.zeros(2)
    for ordinal in range(14):
        payment = torch.tensor([0.0, 1.0]) if ordinal == 0 else torch.zeros(2)
        epoch.milestone.add_reward(
            ordinal, payment, payment,
            torch.tensor([True, True]), torch.tensor([True, True]),
            torch.tensor(0.5, dtype=torch.float64),
        )
        actual.add_(payment * 0.5)
    epoch.milestone.close_actual_step(actual)
    expected_i64 = epoch.milestone.i64.clone()
    expected_f64 = epoch.milestone.f64.clone()
    start, end = epoch.prepare_drain()
    with pytest.raises(RuntimeError, match="frozen for drain"):
        epoch.milestone.add_step_return(torch.ones(2))
    materialized = epoch.materialize_drain(start=start, end=end)
    assert torch.equal(materialized.milestone_i64, expected_i64)
    assert torch.equal(materialized.milestone_f64, expected_f64)
    assert torch.equal(epoch.milestone.i64, expected_i64)
    with pytest.raises(RuntimeError, match="frozen for drain"):
        epoch.milestone.add_step_return(torch.ones(2))
    epoch.acknowledge_drain(start=start, end=end)
    assert not bool(epoch.milestone.i64.any())
    assert not bool(epoch.milestone.f64.any())
    assert torch.equal(
        epoch.milestone.open_episode_return,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
    )


def test_epoch_fact_and_ready_events_are_once_per_full_key_and_reset_rearms():
    epoch, d05, cadence, _r06, _playback, *_rest = _ready_epoch()
    reset = _RealSelectedResetHarness(epoch)
    epoch.bind_selected_reset_owner(reset.owner)
    r03_owner, r07_owner = object(), object()
    epoch.bind_fact_owner("r03_strike_fact", r03_owner)
    epoch.bind_fact_owner("r07_recovery", r07_owner)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    assert epoch.milestone.i64[E.milestone_tensors._EI:E.milestone_tensors._EI + 4].tolist() == [2, 2, 1, 1]

    step = torch.tensor([[5], [-1]], dtype=torch.int64)
    valid = torch.tensor([[3], [0]], dtype=torch.int64)
    invalid = torch.tensor([[1], [0]], dtype=torch.int64)
    r03 = torch.zeros((2, 1, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32)
    r07 = torch.zeros_like(r03)
    r07[:, :, 2:4] = 1.0
    for bits in (valid, invalid, valid):
        epoch.publish_owner_facts(
            "r03_strike_fact", owner=r03_owner,
            valid_bits=bits, source_step=step, values=r03,
        )
        epoch.publish_owner_facts(
            "r07_recovery", owner=r07_owner,
            valid_bits=bits, source_step=step, values=r07,
        )
    current = epoch.current()
    key = E.ActionEpochShotKey(**{
        name: getattr(current.identity.shot_key, name)[:, 0].clone()
        for name in current.identity.shot_key.__dataclass_fields__
    })
    ready = torch.tensor([True, False])
    epoch.publish_r07_first_ready(
        owner=r07_owner, first_ready=ready, shot_key=key, source_step=step[:, 0]
    )
    epoch.publish_r07_first_ready(
        owner=r07_owner, first_ready=ready, shot_key=key, source_step=step[:, 0]
    )
    old_key = key.clone()
    base = E.milestone_tensors._EI
    assert epoch.milestone.i64[base + 4].item() == 1
    assert epoch.milestone.i64[base + 14].item() == 1
    assert epoch.milestone.i64[base + 15].item() == 1

    selected_index = torch.tensor([0], dtype=torch.int64)
    selected = torch.tensor([True, False])
    generation_before = torch.tensor([0, 0], dtype=torch.int64)
    generation_after = torch.tensor([1, 0], dtype=torch.int64)
    overflow = torch.tensor([False, False])
    terminal = _terminal_reset_facts(selected)
    top = reset.arm_preflight(
        selected_env_index=selected_index, selected_mask=selected,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal,
    )
    lease = epoch.prepare_selected_true_reset(
        owner=reset.owner, top_preflight=top,
        selected_env_index=selected_index, selected_mask=selected,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal,
    )
    reset.arm_commit(lease)
    epoch.commit_selected_true_reset(owner=reset.owner, prepared_reset=lease)

    candidate_key = d05.candidate.identity.shot_key.clone()
    candidate_key.reset_generation[0, 0] = 1
    d05.candidate = replace(
        d05.candidate,
        identity=replace(d05.candidate.identity, shot_key=candidate_key),
    )
    cadence.projection = _MotionProjection(
        2, selected.clone(), torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    epoch.publish_owner_facts(
        "r03_strike_fact", owner=r03_owner,
        valid_bits=valid, source_step=step + 1, values=r03,
    )
    epoch.publish_owner_facts(
        "r07_recovery", owner=r07_owner,
        valid_bits=valid, source_step=step + 1, values=r07,
    )
    current = epoch.current()
    key = E.ActionEpochShotKey(**{
        name: getattr(current.identity.shot_key, name)[:, 0].clone()
        for name in current.identity.shot_key.__dataclass_fields__
    })
    epoch.publish_r07_first_ready(
        owner=r07_owner, first_ready=ready, shot_key=key,
        source_step=(step + 1)[:, 0],
    )
    assert epoch.milestone.i64[base + 4].item() == 2
    assert epoch.milestone.i64[base + 14].item() == 2
    assert epoch.milestone.i64[base + 15].item() == 2
    epoch.publish_r07_first_ready(
        owner=r07_owner, first_ready=ready, shot_key=old_key,
        source_step=(step + 1)[:, 0],
    )
    assert epoch.milestone.i64[base + 15].item() == 2
    assert epoch._undecoded_overflow.tolist() == [True, False]


def test_r03_and_r07_first_fact_owner_offsets_are_independently_wired():
    base = E.milestone_tensors._EI
    for owner_kind, expected in (
        ("r03_strike_fact", (1, 0)),
        ("r07_recovery", (0, 1)),
    ):
        epoch, d05, *_rest = _ready_epoch()
        producer = object()
        epoch.bind_fact_owner(owner_kind, producer)
        epoch.prepare_after_command_rows()
        epoch.settle_d05_transaction(d05.arm())
        values = torch.zeros(
            (2, 1, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32
        )
        if owner_kind == "r07_recovery":
            values[:, :, 2:4] = 1.0
        epoch.publish_owner_facts(
            owner_kind, owner=producer,
            valid_bits=torch.tensor([[3], [0]], dtype=torch.int64),
            source_step=torch.tensor([[5], [-1]], dtype=torch.int64),
            values=values,
        )
        assert (
            epoch.milestone.i64[base + 4].item(),
            epoch.milestone.i64[base + 14].item(),
        ) == expected


def test_milestone_failure_after_business_append_sticky_poisons_epoch(monkeypatch):
    epoch, d05, *_rest = _ready_epoch()
    epoch.prepare_after_command_rows()

    def fail_after_append(_self, *_args):
        raise RuntimeError("synthetic milestone failure")

    monkeypatch.setattr(
        E.milestone_tensors.MilestoneTensorAccumulator,
        "add_d05_events",
        fail_after_append,
    )
    with pytest.raises(RuntimeError, match="synthetic milestone failure"):
        epoch.settle_d05_transaction(d05.arm())
    transitions = tuple(
        entry.transition for entry in epoch._publication.pending_log
    )
    assert "D05_SETTLED" in transitions
    assert transitions[-1] == "OWNER_WRITE_POISON:r05_runtime"
    assert epoch.poisoned is True


def test_d05_event_vector_distinguishes_due_from_slot_opportunity_censor():
    epoch, d05, cadence, *_rest = _ready_epoch()
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    cadence.projection = _MotionProjection(
        2,
        torch.tensor([True, False]),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.int64),
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    base = E.milestone_tensors._EI
    assert epoch.milestone.i64[base:base + 4].tolist() == [3, 2, 1, 1]


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_r06_event_tuple_counts_unique_settlement_not_reprojection_or_wrong_key(
    monkeypatch, device,
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    assert (
        R06_MODULE.R06_ACTION_EPOCH_CONTACT_VALID_F32,
        R06_MODULE.R06_ACTION_EPOCH_NET_CROSSED_F32,
        R06_MODULE.R06_ACTION_EPOCH_NET_CLEAR_F32,
        R06_MODULE.R06_ACTION_EPOCH_CROSSING_VALID_F32,
        R06_MODULE.R06_ACTION_EPOCH_ON_TABLE_F32,
        R06_MODULE.R06_ACTION_EPOCH_COMMON_ON_TABLE_F32,
    ) == (7, 9, 10, 8, 6, 0)
    offsets = (7, 9, 10, 8, 6, 0)
    exact_device = torch.device(device)

    def run(
        pattern, *, wrong_key=False, publication_delta=0, replay_count=0
    ):
        epoch, d05, _cadence, r06, _playback, *_middle, physical = _ready_epoch(
            device=exact_device
        )
        epoch.prepare_after_command_rows()
        epoch.settle_d05_transaction(d05.arm())
        physical.launch = _launch_packet(
            epoch.current(),
            due=torch.tensor([True, False], device=exact_device),
        )
        epoch.refresh_physical_launch_rows()
        current = epoch.current()
        key = E.ActionEpochShotKey(**{
            name: getattr(current.identity.shot_key, name)[:, 0].clone()
            for name in current.identity.shot_key.__dataclass_fields__
        })
        if wrong_key:
            key.action_uid[0] += 1
        facts = torch.zeros(
            (2, E.OWNER_FACT_F32_WIDTH),
            dtype=torch.float32,
            device=exact_device,
        )
        for value, offset in zip(pattern, offsets):
            facts[0, offset] = float(value)
        publication_ordinal = current.publication_ordinal[:, 0].clone()
        publication_ordinal[0] += publication_delta
        r06.outcome = ActionEpochR06OutcomeRows(
            valid=torch.tensor([True, False], device=exact_device), shot_key=key,
            publication_ordinal=publication_ordinal,
            settlement_step=torch.tensor(
                [12, -1], dtype=torch.int64, device=exact_device
            ),
            valid_bits=torch.tensor(
                [1, 0], dtype=torch.int64, device=exact_device
            ),
            fact_values=facts,
            outcome_code=torch.tensor(
                [2, -1], dtype=torch.int64, device=exact_device
            ),
            owner_fault_bits=torch.zeros(
                2, dtype=torch.int64, device=exact_device
            ),
        )
        epoch.refresh_r06_outcome_rows()
        for _ in range(replay_count):
            epoch.refresh_r06_outcome_rows()
        return epoch, r06

    patterns = (
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (1, 1, 1, 1, 1, 1),
    )
    base = E.milestone_tensors._EI + 7
    for pattern in patterns:
        epoch, _r06 = run(pattern)
        assert epoch.milestone.i64[base:base + 7].tolist() == [1, *pattern]

    # One settlement followed by the three legal decimation re-reads has one
    # positive business incidence, one event, and no device overflow.
    epoch, _r06 = run(patterns[0], replay_count=3)
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    outcome_masks = tuple(
        entry.delta.values[0]
        for entry in materialized.entries
        if entry.transition == "R06_OUTCOME_ROWS"
    )
    assert len(outcome_masks) == 4  # one positive plus three zero-mask observations
    assert sum(int(mask.to(torch.int64).sum()) for mask in outcome_masks) == 1
    assert materialized.milestone_i64[base:base + 7].tolist() == [
        1, *patterns[0]
    ]
    assert materialized.overflow.tolist() == [False, False]

    wrong, _r06 = run(patterns[-1], wrong_key=True)
    assert not bool(wrong.milestone.i64[base:base + 7].any())
    start, end = wrong.prepare_drain()
    assert wrong.materialize_drain(start=start, end=end).overflow.tolist() == [
        True, False
    ]

    for delta in (-1, 1):
        wrong_publication, _r06 = run(
            patterns[-1], publication_delta=delta
        )
        assert not bool(
            wrong_publication.milestone.i64[base:base + 7].any()
        )
        start, end = wrong_publication.prepare_drain()
        assert wrong_publication.materialize_drain(
            start=start, end=end
        ).overflow.tolist() == [True, False]

    # Every retained payload field participates in the replay after-image.
    for changed in (
        "shot_key", "publication_ordinal", "settlement_step", "valid_bits",
        "fact_values", "outcome_code", "owner_fault_bits",
    ):
        mutant, r06 = run(patterns[-1])
        packet = r06.outcome
        assert packet is not None
        if changed == "shot_key":
            action_uid = packet.shot_key.action_uid.clone()
            action_uid[0] += 1
            packet = replace(
                packet,
                shot_key=replace(packet.shot_key, action_uid=action_uid),
            )
        else:
            value = getattr(packet, changed).clone()
            if changed == "fact_values":
                value[0, 3] += 1.0
            else:
                value[0] += 1
            packet = replace(packet, **{changed: value})
        r06.outcome = packet
        mutant.refresh_r06_outcome_rows()
        start, end = mutant.prepare_drain()
        materialized = mutant.materialize_drain(start=start, end=end)
        assert materialized.overflow.tolist() == [True, False]
        assert materialized.milestone_i64[base:base + 7].tolist() == [
            1, *patterns[-1]
        ]


def test_physical_event_pair_distinguishes_observation_then_contact_and_replay(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    d05 = _RealD05Harness(epoch, _candidate(torch.device("cpu")))
    d05.bind()
    cadence = _MotionCadence(torch.device("cpu"))
    epoch.bind_motion_cadence_owner(cadence)
    r06 = _R06(epoch, torch.device("cpu"))
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    physical = _PhysicalProjectionOwner()
    epoch.bind_fact_owner("physical_ball", physical)
    epoch.bind_async_owner("physical_ball", physical)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    physical.launch = _launch_packet(
        epoch.current(), due=torch.tensor([True, False])
    )
    epoch.refresh_physical_launch_rows()
    base = E.milestone_tensors._EI + 5

    physical.projection = _physical_packet(
        epoch.current(), flight_index=0, contact=False
    )
    epoch.refresh_physical_postphysics_rows()
    assert epoch.milestone.i64[base:base + 2].tolist() == [1, 0]

    physical.projection = _physical_packet(
        epoch.current(), flight_index=0, contact=True
    )
    epoch.refresh_physical_postphysics_rows()
    assert epoch.milestone.i64[base:base + 2].tolist() == [1, 1]
    epoch.refresh_physical_postphysics_rows()
    assert epoch.milestone.i64[base:base + 2].tolist() == [1, 1]


def test_drain_decoder_rejects_truncation_extension_and_schema_mutations():
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    entries = epoch._publication.pending_log
    layout = epoch._drain_layout(entries)
    payload = torch.cat(
        tuple(
            value.detach().contiguous().view(torch.uint8).reshape(-1)
            for entry in entries
            for value in entry.delta.values
        )
        + tuple(value.view(torch.uint8).reshape(-1) for value in epoch.milestone.pack_views())
        + (epoch._undecoded_overflow.view(torch.uint8).reshape(-1),),
        dim=0,
    ).contiguous()
    malformed = (
        (payload[:-1].contiguous(), layout),
        (torch.cat((payload, torch.zeros(1, dtype=torch.uint8))), layout),
        (payload, (replace(layout[0], offset=1), *layout[1:])),
        (payload, (replace(layout[0], dtype=torch.float32), *layout[1:])),
        (payload, (replace(layout[0], shape=(1, 2)), *layout[1:])),
    )
    for host_bytes, bad_layout in malformed:
        with pytest.raises(E.ActionEpochError):
            epoch._decode_drain_bytes(
                host_bytes=host_bytes, entries=entries, layout=bad_layout
            )


def _decode_test_entries(epoch, entries):
    layout = epoch._drain_layout(entries)
    payload = torch.cat(
        tuple(
            value.detach().contiguous().view(torch.uint8).reshape(-1)
            for entry in entries
            for value in entry.delta.values
        )
        + tuple(value.view(torch.uint8).reshape(-1) for value in epoch.milestone.pack_views())
        + (torch.zeros(epoch.num_envs, dtype=torch.uint8),),
        dim=0,
    ).contiguous()
    return epoch._decode_drain_bytes(
        host_bytes=payload, entries=entries, layout=layout
    )


def _assert_known_physical_launch_semantics(entry):
    values = dict(zip(entry.delta.names, entry.delta.values))
    assert torch.equal(
        values["event_mask"], torch.tensor([[True], [False]])
    )
    assert torch.equal(
        values["launch_succeeded"], torch.tensor([[True], [False]])
    )
    assert torch.equal(
        values["late_launch"], torch.tensor([[False], [False]])
    )
    assert torch.equal(
        values["target_xy_m"],
        torch.tensor([[[1.25, -2.5]], [[0.0, 0.0]]], dtype=torch.float32),
    )


def _assert_known_d05_settled_semantics(entry):
    values = dict(zip(entry.delta.names, entry.delta.values))
    assert torch.equal(
        values["construction_admissible"], torch.tensor([[True], [False]])
    )
    assert torch.equal(
        values["playback_admissible"], torch.tensor([[True], [True]])
    )
    assert torch.equal(
        values["decision"],
        torch.tensor([[E.D05_DECISION_ACCEPT], [E.D05_DECISION_REJECT]]),
    )


def test_drain_decoder_known_d05_vector_rejects_same_shape_field_swap(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, *_rest = _ready_epoch()
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    settled_entry = next(
        entry
        for entry in epoch._publication.pending_log
        if entry.transition == "D05_SETTLED"
    )
    decoded = _decode_test_entries(epoch, (settled_entry,))
    _assert_known_d05_settled_semantics(decoded.entries[0])

    construction_index = settled_entry.delta.names.index(
        "construction_admissible"
    )
    playback_index = settled_entry.delta.names.index("playback_admissible")
    mutant_values = list(settled_entry.delta.values)
    mutant_values[construction_index], mutant_values[playback_index] = (
        mutant_values[playback_index],
        mutant_values[construction_index],
    )
    mutant = replace(
        settled_entry,
        delta=E.PackedDelta(settled_entry.delta.names, tuple(mutant_values)),
    )
    mutant_decoded = _decode_test_entries(epoch, (mutant,))
    with pytest.raises(AssertionError):
        _assert_known_d05_settled_semantics(mutant_decoded.entries[0])


def test_drain_decoder_known_launch_vector_rejects_same_shape_field_swap(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, *_middle, physical = _ready_epoch()
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    physical.launch = _launch_packet(
        epoch.current(),
        due=torch.tensor([True, False]),
        target_xy_m=torch.tensor([[1.25, -2.5], [8.0, 9.0]]),
    )
    epoch.refresh_physical_launch_rows()
    launch_entry = next(
        entry
        for entry in epoch._publication.pending_log
        if entry.transition == "PHYSICAL_LAUNCH_ROWS"
    )
    decoded = _decode_test_entries(epoch, (launch_entry,))
    _assert_known_physical_launch_semantics(decoded.entries[0])

    launch_index = launch_entry.delta.names.index("launch_succeeded")
    late_index = launch_entry.delta.names.index("late_launch")
    mutant_values = list(launch_entry.delta.values)
    mutant_values[launch_index], mutant_values[late_index] = (
        mutant_values[late_index],
        mutant_values[launch_index],
    )
    mutant = replace(
        launch_entry,
        delta=E.PackedDelta(launch_entry.delta.names, tuple(mutant_values)),
    )
    mutant_decoded = _decode_test_entries(epoch, (mutant,))
    with pytest.raises(AssertionError):
        _assert_known_physical_launch_semantics(mutant_decoded.entries[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profiler unavailable")
def test_materialize_drain_cuda_profiler_observes_one_physical_d2h():
    device = torch.device("cuda:0")
    epoch = E.ActionEpochOwner(num_envs=2, device=device)
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(2, dtype=torch.int64, device=device),
    )
    start, end = epoch.prepare_drain()
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        materialized = epoch.materialize_drain(start=start, end=end)
        torch.cuda.synchronize(device)
    dtoh_events = [
        event
        for event in profiler.events()
        if event.name.startswith("Memcpy DtoH")
    ]
    assert len(dtoh_events) == 1
    assert materialized.overflow.device.type == "cpu"
    assert all(
        value.device.type == "cpu"
        for entry in materialized.entries
        for value in entry.delta.values
    )


def _assert_failed_materialization_is_retryable(epoch, *, start, end, pending_log):
    assert epoch._pending_drain == (start, end)
    assert not epoch._pending_drain_materialized
    assert epoch._publication.pending_log is pending_log
    assert epoch.drain_frontier == start
    with pytest.raises(E.ActionEpochError):
        epoch.acknowledge_drain(start=start, end=end)
    assert epoch._pending_drain == (start, end)
    assert not epoch._pending_drain_materialized
    assert epoch._publication.pending_log is pending_log
    assert epoch.drain_frontier == start


def test_drain_transfer_failure_rejects_ack_and_allows_exact_retry(monkeypatch):
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    pending_log = epoch._publication.pending_log
    start, end = epoch.prepare_drain()
    real_transfer = E._single_d2h_packed_bytes

    def fail_transfer(_device_bytes):
        raise RuntimeError("synthetic packed transfer failure")

    monkeypatch.setattr(E, "_single_d2h_packed_bytes", fail_transfer)
    with pytest.raises(RuntimeError, match="synthetic packed transfer failure"):
        epoch.materialize_drain(start=start, end=end)
    with pytest.raises(RuntimeError, match="frozen for drain"):
        epoch.milestone.add_step_return(torch.ones(2))
    _assert_failed_materialization_is_retryable(
        epoch, start=start, end=end, pending_log=pending_log
    )

    monkeypatch.setattr(E, "_single_d2h_packed_bytes", real_transfer)
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.entries[0].transition == "RESET_GENESIS_IDLE"
    epoch.acknowledge_drain(start=start, end=end)
    assert epoch.drain_frontier == end


def test_drain_decode_failure_rejects_ack_and_allows_exact_retry(monkeypatch):
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    pending_log = epoch._publication.pending_log
    start, end = epoch.prepare_drain()
    real_decode = epoch._decode_drain_bytes

    def fail_decode(**_kwargs):
        raise E.ActionEpochError("synthetic packed decode failure")

    monkeypatch.setattr(epoch, "_decode_drain_bytes", fail_decode)
    with pytest.raises(E.ActionEpochError, match="synthetic packed decode failure"):
        epoch.materialize_drain(start=start, end=end)
    _assert_failed_materialization_is_retryable(
        epoch, start=start, end=end, pending_log=pending_log
    )

    monkeypatch.setattr(epoch, "_decode_drain_bytes", real_decode)
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.entries[0].transition == "RESET_GENESIS_IDLE"
    epoch.acknowledge_drain(start=start, end=end)
    assert epoch.drain_frontier == end


def test_single_d2h_preserves_full_key_payment_and_selected_reset_bytes(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, _cadence, r06, playback, *_middle, physical = _ready_epoch(
        bind_playback=True
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    epoch.publish_motion_playback_started(owner=playback)
    physical.launch = _launch_packet(
        epoch.current(), due=torch.tensor([True, False])
    )
    epoch.refresh_physical_launch_rows()
    current = epoch.current()
    key = E.ActionEpochShotKey(
        **{
            name: getattr(current.identity.shot_key, name)[:, 0].clone()
            for name in current.identity.shot_key.__dataclass_fields__
        }
    )
    r06.outcome = ActionEpochR06OutcomeRows(
        valid=torch.tensor([True, False]),
        shot_key=key,
        publication_ordinal=current.publication_ordinal[:, 0].clone(),
        settlement_step=torch.tensor([12, -1], dtype=torch.int64),
        valid_bits=torch.tensor([1, 0], dtype=torch.int64),
        fact_values=torch.zeros((2, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32),
        outcome_code=torch.tensor([2, -1], dtype=torch.int64),
        owner_fault_bits=torch.zeros(2, dtype=torch.int64),
    )
    epoch.refresh_r06_outcome_rows()
    epoch.open_reward_cycle()
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    epoch.publish_reward_payment(20)
    expected_entries, expected_overflow = _pending_host_reference(epoch)
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    _assert_materialized_bytes_equal(
        materialized, expected_entries, expected_overflow
    )
    assert any(entry.transition == "D05_SETTLED" for entry in materialized.entries)
    assert any(entry.transition == "PAYMENT_RECORDED" for entry in materialized.entries)

    reset_epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    reset_epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    reset = _RealSelectedResetHarness(reset_epoch)
    reset_epoch.bind_selected_reset_owner(reset.owner)
    selected_env_index = torch.tensor([0], dtype=torch.int64)
    selected_mask = torch.tensor([True, False])
    generation_before = torch.tensor([0, 0], dtype=torch.int64)
    generation_after = torch.tensor([1, 0], dtype=torch.int64)
    generation_overflow_fault = torch.tensor([False, False])
    terminal_reset_facts_i64 = _terminal_reset_facts(selected_mask)
    top = reset.arm_preflight(
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=generation_overflow_fault,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    lease = reset_epoch.prepare_selected_true_reset(
        owner=reset.owner,
        top_preflight=top,
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=generation_overflow_fault,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    reset.arm_commit(lease)
    reset_epoch.commit_selected_true_reset(
        owner=reset.owner, prepared_reset=lease
    )
    expected_entries, expected_overflow = _pending_host_reference(reset_epoch)
    start, end = reset_epoch.prepare_drain()
    materialized = reset_epoch.materialize_drain(start=start, end=end)
    _assert_materialized_bytes_equal(
        materialized, expected_entries, expected_overflow
    )
    assert any(entry.transition == "RESET_SELECTED" for entry in materialized.entries)


def test_frozen_drain_rejects_before_producer_pull_or_device_fault_latch(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, _d05, _cadence, _r06, _playback, *_middle, physical = _ready_epoch()
    fact_owner = object()
    epoch.bind_fact_owner("r03_strike_fact", fact_owner)
    baseline = epoch._undecoded_overflow.clone()
    start, end = epoch.prepare_drain()
    with pytest.raises(E.ActionEpochError):
        epoch.refresh_physical_launch_rows()
    assert physical.launch_calls == 0
    assert torch.equal(epoch._undecoded_overflow, baseline)
    epoch.abort_drain(start=start, end=end)
    epoch.milestone.add_step_return(torch.ones(2))

    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert not bool(materialized.overflow.any())
    with pytest.raises(E.ActionEpochError):
        epoch.publish_owner_facts(
            "r03_strike_fact",
            owner=fact_owner,
            valid_bits=torch.tensor([[1], [0]], dtype=torch.int64),
            source_step=torch.tensor([[1], [-1]], dtype=torch.int64),
            values=torch.zeros(
                (2, 1, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32
            ),
        )
    assert torch.equal(epoch._undecoded_overflow, baseline)
    epoch.acknowledge_drain(start=start, end=end)
    assert torch.equal(epoch._undecoded_overflow, baseline)


def test_drain_abort_overflow_and_inference_replacement_are_exact(monkeypatch):
    with torch.inference_mode():
        epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
        epoch.activate_reset_genesis(
            selected_mask=torch.ones(2, dtype=torch.bool),
            reset_generation=torch.zeros(2, dtype=torch.int64),
        )
        start, end = epoch.prepare_drain()
    inference_storage = epoch._undecoded_overflow
    assert torch.is_inference(inference_storage)
    epoch.abort_drain(start=start, end=end)
    start, end = epoch.prepare_drain()
    epoch.materialize_drain(start=start, end=end)
    try:
        epoch.abort_drain(start=start, end=end)
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("post-materialization drain aborted")
    original_zeros = E.torch.zeros

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("synthetic normal-storage allocation failure")

    monkeypatch.setattr(E.torch, "zeros", fail_allocation)
    try:
        epoch.acknowledge_drain(start=start, end=end)
    except RuntimeError:
        pass
    else:
        raise AssertionError("ACK committed despite replacement allocation failure")
    assert epoch.drain_frontier == start
    assert epoch._undecoded_overflow is inference_storage
    monkeypatch.setattr(E.torch, "zeros", original_zeros)
    epoch.acknowledge_drain(start=start, end=end)
    assert epoch._undecoded_overflow is not inference_storage
    assert not torch.is_inference(epoch._undecoded_overflow)
    try:
        epoch.acknowledge_drain(start=start, end=end)
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("drain ACK replayed")

    overflowed = E.ActionEpochOwner(num_envs=2, device="cpu")
    overflowed.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.ones(2, dtype=torch.int64),
    )
    start, end = overflowed.prepare_drain()
    materialized = overflowed.materialize_drain(start=start, end=end)
    assert bool(materialized.overflow.all()) and overflowed.poisoned
    storage = overflowed._undecoded_overflow
    try:
        overflowed.acknowledge_drain(start=start, end=end)
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("overflowed drain ACKed")
    assert overflowed._undecoded_overflow is storage
    assert overflowed.drain_frontier == start


def test_inference_constructed_owner_accepts_genesis_after_context_exit():
    with torch.inference_mode():
        epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    inference_storage = epoch._undecoded_overflow
    assert torch.is_inference(inference_storage)

    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    assert epoch._undecoded_overflow is not inference_storage
    assert not torch.is_inference(epoch._undecoded_overflow)
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert not bool(materialized.overflow.any())
    epoch.acknowledge_drain(start=start, end=end)
    assert epoch.drain_frontier == end


def test_genesis_and_exact_bindings_are_single_flight(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    empty = E.ActionEpochOwner(num_envs=2, device="cpu")
    try:
        empty.bind_motion_cadence_owner(_MotionCadence(torch.device("cpu")))
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("Motion cadence bound before canonical genesis")
    epoch, *_rest = _ready_epoch()
    try:
        epoch.bind_motion_cadence_owner(_MotionCadence(torch.device("cpu")))
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("Motion cadence rebound")
    assert epoch.current().epoch == -1
    assert torch.all(epoch.current().identity.shot_key.action_uid.eq(-1))
    patched_epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    patched_epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    patched = type("PatchedCadence", (), {})()
    patched.project_current_action_epoch_rows = lambda: None
    with pytest.raises(E.ActionEpochError):
        patched_epoch.bind_motion_cadence_owner(patched)
    writer_epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    writer_epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    class _ForeignD05Composite:
        def require_owned_action_epoch_candidate(self, _token):
            return _candidate(torch.device("cpu"))

        def _commit_action_epoch_motion_write(self, _token):
            return None

        def _commit_action_epoch_racket_write(self, _token):
            return None

        def _commit_action_epoch_r05_write(self, _token):
            return None

    foreign_d05 = _ForeignD05Composite()
    with pytest.raises(E.ActionEpochError):
        writer_epoch.bind_d05_accept_writers(
            motion_write=foreign_d05._commit_action_epoch_motion_write,
            racket_write=foreign_d05._commit_action_epoch_racket_write,
            r05_write=foreign_d05._commit_action_epoch_r05_write,
        )
    assert E.ActionEpochOwner._exact_d05_owner_type() is D05.DeviceR05Owner
    physical_fact = _Physical()
    writer_epoch.bind_fact_owner("physical_ball", physical_fact)
    with pytest.raises(E.ActionEpochError):
        writer_epoch.bind_async_owner("physical_ball", _Physical())
    writer_epoch.bind_async_owner("physical_ball", physical_fact)


def test_motion_playback_edge_is_monotonic_and_owner_failure_poison_is_sticky(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, _cadence, _r06, playback, *_ = _ready_epoch(bind_playback=True)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    first = epoch.publish_motion_playback_started(owner=playback)
    assert first.motion_playback_started[:, 0].tolist() == [True, False]
    replay = epoch.publish_motion_playback_started(owner=playback)
    assert replay.motion_playback_started[:, 0].tolist() == [True, False]
    try:
        epoch.publish_motion_playback_started(owner=object())
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("foreign Motion owner published an edge")
    playback.fail = True
    try:
        epoch.publish_motion_playback_started(owner=playback)
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("Motion owner failure did not fail closed")
    assert epoch.poisoned
    try:
        epoch.open_reward_cycle()
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("poisoned Epoch continued")


@pytest.mark.parametrize(
    ("started", "reason"),
    [
        (False, E.MOTION_CLOSE_PLAYED_SUFFIX),
        (True, E.MOTION_CLOSE_UNPLAYED),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_motion_close_causality_is_device_latched_without_host_observation(
    monkeypatch, device, started, reason
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, cadence, _r06, playback, *_ = _ready_epoch(
        bind_playback=started, device=device
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    if started:
        epoch.publish_motion_playback_started(owner=playback)
    exact_device = torch.device(device)
    cadence.projection = _MotionProjection(
        2,
        torch.tensor([True, False], dtype=torch.bool, device=exact_device),
        torch.tensor([True, False], dtype=torch.bool, device=exact_device),
        torch.tensor([reason, E.MOTION_CLOSE_NONE], dtype=torch.int64, device=exact_device),
    )
    with _NoHostTensorObservation():
        rows = epoch.prepare_after_command_rows()
    assert not rows.due_mask[0]
    assert epoch.current().motion_close_reason[0, 0] == E.MOTION_CLOSE_NONE
    epoch.settle_d05_transaction(d05.arm())
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.overflow.tolist() == [True, False]


def test_owner_fault_plane_is_typed_and_malformed_input_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, *_ = _ready_epoch()
    r07_owner = object()
    epoch.bind_fact_owner("r07_recovery", r07_owner)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    before = epoch.current()
    with pytest.raises(E.ActionEpochError):
        epoch.merge_runtime_owner_fault(
            "r07_recovery",
            torch.tensor([[4], [0]], dtype=torch.int64),
            owner=object(),
        )
    assert epoch.current().version == before.version
    try:
        epoch.merge_runtime_owner_fault(
            "r07_recovery", torch.ones(2, dtype=torch.int64), owner=r07_owner
        )
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("rank-1 owner fault was accepted")
    bits = torch.tensor([[4], [0]], dtype=torch.int64)
    merge_result = epoch.merge_runtime_owner_fault(
        "r07_recovery", bits, owner=r07_owner
    )
    assert merge_result is None
    record = epoch.current()
    slot = E.OWNER_ORDER.index("r07_recovery")
    assert record.owner_fault_bits[:, 0, slot].tolist() == [4, 0]


def test_owner_write_poison_requires_exact_attributed_owner_without_mutation(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, _cadence, r06, *_middle, physical = _ready_epoch()
    before_head = epoch.commit_head
    before_rows = (_peer_bytes(epoch.current(), row=0), _peer_bytes(epoch.current()))
    for owner_kind, foreign in (
        ("r05_runtime", object()),
        ("physical_ball", r06),
        ("r06_landing_outcome", physical),
    ):
        with pytest.raises(E.ActionEpochError):
            epoch.poison_owner_write(owner_kind, 11, owner=foreign)
    assert epoch.commit_head == before_head and not epoch.poisoned
    after_rows = (_peer_bytes(epoch.current(), row=0), _peer_bytes(epoch.current()))
    assert all(
        before.keys() == after.keys()
        and all(torch.equal(before[name], after[name]) for name in before)
        for before, after in zip(before_rows, after_rows)
    )
    poisoned = epoch.poison_owner_write(
        "r05_runtime", 11, owner=d05.owner
    )
    assert epoch.poisoned and poisoned.version == epoch.current().version
    assert poisoned.phase.eq(E.PHASE_IDLE).all()


def test_owner_facts_mask_idle_rows_and_keep_active_owner_planes_distinct(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    idle = E.ActionEpochOwner(num_envs=2, device="cpu")
    idle.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    idle_owner = object()
    idle.bind_fact_owner("r03_strike_fact", idle_owner)
    idle.publish_owner_facts(
        "r03_strike_fact",
        owner=idle_owner,
        valid_bits=torch.tensor([[1], [0]], dtype=torch.int64),
        source_step=torch.tensor([[3], [-1]], dtype=torch.int64),
        values=torch.ones((2, 1, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32),
    )
    assert idle.current().fact_valid_bits.eq(0).all()
    start, end = idle.prepare_drain()
    assert idle.materialize_drain(start=start, end=end).overflow.tolist() == [True, False]

    epoch, d05, _cadence, _r06, _playback, *_middle, physical = _ready_epoch()
    r03_owner, r07_owner = object(), object()
    epoch.bind_fact_owner("r03_strike_fact", r03_owner)
    epoch.bind_fact_owner("r07_recovery", r07_owner)
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    valid = torch.tensor([[1], [0]], dtype=torch.int64)
    step = torch.tensor([[5], [-1]], dtype=torch.int64)
    r03_values = torch.ones((2, 1, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32)
    r07_values = torch.full_like(r03_values, 7.0)
    epoch.publish_owner_facts(
        "r03_strike_fact", owner=r03_owner,
        valid_bits=valid, source_step=step, values=r03_values,
    )
    physical.launch = _launch_packet(epoch.current(), due=torch.tensor([True, False]))
    epoch.refresh_physical_launch_rows()
    publish_result = epoch.publish_owner_facts(
        "r07_recovery", owner=r07_owner,
        valid_bits=valid, source_step=step, values=r07_values,
    )
    assert publish_result is None
    record = epoch.current()
    r03_slot = E.OWNER_ORDER.index("r03_strike_fact")
    r07_slot = E.OWNER_ORDER.index("r07_recovery")
    assert record.fact_f32[0, 0, r03_slot, 0].item() == 1.0
    assert record.fact_f32[0, 0, r07_slot, 0].item() == 7.0
    assert not record.fact_valid_bits[1].any()


def test_fixed_writer_order_zero_masks_and_post_write_failure_are_sticky(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    candidate = _candidate(torch.device("cpu"))
    candidate.construction_admissible.zero_()
    d05 = _RealD05Harness(epoch, candidate)
    d05.motion.fail = True
    d05.bind()
    cadence = _MotionCadence(torch.device("cpu"))
    epoch.bind_motion_cadence_owner(cadence)
    r06 = _R06(epoch, torch.device("cpu"))
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    epoch.prepare_after_command_rows()
    try:
        epoch.settle_d05_transaction(d05.arm())
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed irreversible writer returned")
    assert d05.motion.calls == 1 and epoch.poisoned


def test_d05_private_abort_is_zero_write_and_owner_exact(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, _cadence, _r06, _playback, motion, racket, _physical = (
        _ready_epoch()
    )
    epoch.prepare_after_command_rows()
    try:
        epoch.abort_d05_transaction(owner=object())
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("foreign D05 owner aborted a transaction")
    head = epoch.commit_head
    epoch.abort_d05_transaction(owner=d05.owner)
    assert epoch.commit_head == head
    assert motion.calls == racket.calls == 0 and d05.calls == []


def test_d05_old_reset_generation_is_censored_and_device_latched(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, _cadence, _r06, _playback, motion, racket, _physical = (
        _ready_epoch()
    )
    reset = _RealSelectedResetHarness(epoch)
    epoch.bind_selected_reset_owner(reset.owner)
    selected_env_index = torch.tensor([0], dtype=torch.int64)
    selected_mask = torch.tensor([True, False])
    generation_before = torch.tensor([0, 0], dtype=torch.int64)
    generation_after = torch.tensor([1, 0], dtype=torch.int64)
    overflow = torch.tensor([False, False])
    terminal_reset_facts_i64 = _terminal_reset_facts(selected_mask)
    top = reset.arm_preflight(
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    lease = epoch.prepare_selected_true_reset(
        owner=reset.owner,
        top_preflight=top,
        selected_env_index=selected_env_index,
        selected_mask=selected_mask,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal_reset_facts_i64,
    )
    reset.arm_commit(lease)
    epoch.commit_selected_true_reset(
        owner=reset.owner, prepared_reset=lease
    )
    epoch.prepare_after_command_rows()
    before = epoch.current()
    epoch.settle_d05_transaction(d05.arm())
    after = epoch.current()
    assert torch.equal(after.phase, before.phase)
    assert motion.calls == racket.calls == 1 and d05.calls == ["r05_runtime"]
    assert all(
        not bool(mask.any())
        for mask in (
            *motion.accepted_masks,
            *racket.accepted_masks,
            *d05.accepted_masks,
        )
    )
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    settled = next(
        entry for entry in materialized.entries
        if entry.transition == "D05_SETTLED"
    )
    assert settled.delta.values[11][:, 0].tolist() == [
        E.D05_DECISION_CENSOR,
        E.D05_DECISION_REJECT,
    ]
    assert materialized.overflow.tolist() == [True, False]


def test_d05_defer_and_censor_are_event_only_with_exact_zero_writer_log(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, _cadence, _r06, _playback, motion, racket, _physical = (
        _ready_epoch()
    )
    base = d05.candidate
    both = torch.ones((2, 1), dtype=torch.bool)
    faults = base.owner_fault_bits.clone()
    faults[1, 0, 0] = 8
    d05.candidate = replace(
        base,
        identity=replace(
            base.identity,
            shot_key=_key(torch.tensor([[21], [22]], dtype=torch.int64), both),
        ),
        task=replace(base.task, task_valid=both),
        construction_admissible=both,
        playback_admissible=torch.tensor([[False], [True]]),
        owner_fault_bits=faults,
    )
    epoch.prepare_after_command_rows()
    before = epoch.current()
    epoch.settle_d05_transaction(d05.arm())
    after = epoch.current()
    assert torch.equal(before.phase, after.phase)
    entries = _materialized_entries(epoch)
    settled = next(
        entry for entry in entries if entry.transition == "D05_SETTLED"
    )
    assert settled.delta.values[11][:, 0].tolist() == [
        E.D05_DECISION_DEFER,
        E.D05_DECISION_CENSOR,
    ]
    tail = [
        entry
        for entry in entries
        if entry.transition.startswith("WRITES_")
        or entry.transition == "D05_ACCEPT_PUBLISHED"
    ]
    assert [entry.transition for entry in tail] == [
        "WRITES_STARTED:motion", "WRITES_COMMITTED:motion",
        "WRITES_STARTED:racket", "WRITES_COMMITTED:racket",
        "WRITES_STARTED:r05_runtime", "WRITES_COMMITTED:r05_runtime",
        "D05_ACCEPT_PUBLISHED",
    ]
    assert all(not bool(entry.delta.values[0].any()) for entry in tail)
    assert motion.calls == racket.calls == 1 and d05.calls == ["r05_runtime"]


def test_recovery_reference_masks_preserve_idle_and_carry_across_event_only_d05(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, cadence, _r06, _playback, *_ = _ready_epoch()

    genesis = epoch.current().recovery_reference_lifecycle_masks()
    assert genesis.upcoming[:, 0].tolist() == [True, True]
    assert genesis.completed[:, 0].tolist() == [False, False]

    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    accepted = epoch.current().recovery_reference_lifecycle_masks()
    assert accepted.upcoming[:, 0].tolist() == [False, True]
    assert accepted.completed[:, 0].tolist() == [True, False]

    both = torch.ones((2, 1), dtype=torch.bool)
    base = d05.candidate
    d05.candidate = replace(
        base,
        identity=replace(
            base.identity,
            shot_key=_key(torch.tensor([[31], [32]], dtype=torch.int64), both),
        ),
        task=replace(base.task, task_valid=both),
        construction_admissible=both,
        playback_admissible=torch.zeros_like(both),
        owner_fault_bits=torch.zeros_like(base.owner_fault_bits),
    )
    cadence.projection = _MotionProjection(
        common_step=2,
        reveal_due=torch.ones(2, dtype=torch.bool),
        closed_mask=torch.zeros(2, dtype=torch.bool),
        close_reason=torch.zeros(2, dtype=torch.int64),
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())

    unchanged = epoch.current().recovery_reference_lifecycle_masks()
    assert unchanged.upcoming[:, 0].tolist() == [False, True]
    assert unchanged.completed[:, 0].tolist() == [True, False]
    assert not bool((unchanged.upcoming & unchanged.completed).any())
    entries = _materialized_entries(epoch)
    decisions = [
        entry.delta.values[11][:, 0].tolist()
        for entry in entries
        if entry.transition == "D05_SETTLED"
    ]
    assert decisions == [
        [E.D05_DECISION_ACCEPT, E.D05_DECISION_REJECT],
        [E.D05_DECISION_CENSOR, E.D05_DECISION_DEFER],
    ]


def test_reward_order_overflow_and_checkpoint_are_bounded():
    epoch = E.ActionEpochOwner(
        num_envs=2, device="cpu", initial_reward_cycle_age=2**63 - 1
    )
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    epoch.open_reward_cycle()
    try:
        epoch.pay_reward(1)
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("out-of-order Reward consumer was accepted")
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert not bool(materialized.overflow.any())
    epoch.acknowledge_drain(start=start, end=end)
    checkpoint = epoch.checkpoint()
    assert checkpoint.reward_cycle_age.tolist() == [2**63 - 1, 2**63 - 1]
    assert checkpoint.current.reward_cycle_fault.ne(0).all()
    assert not hasattr(checkpoint, "commit_log")


def test_reward_payment_mutates_publication_without_return_clone(monkeypatch):
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    epoch.open_reward_cycle()
    before = epoch._publication.current
    assert before is not None
    commit_head = epoch.commit_head

    clone_calls = 0
    original_clone = E.ActionEpochRecord.clone

    def counted_clone(record):
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(record)

    monkeypatch.setattr(E.ActionEpochRecord, "clone", counted_clone)

    assert epoch.pay_reward(0) is None
    paid = epoch._publication.current
    assert paid is not None and paid is not before
    assert paid.identity is before.identity
    assert paid.version == before.version + 1
    assert epoch.commit_head == commit_head + 1
    assert paid.reward_paid[:, 0].tolist() == [True, True]
    assert not bool(paid.reward_paid[:, 1:].any())
    assert clone_calls == 0

    for wrong_ordinal in (0, 2):
        with pytest.raises(E.ActionEpochError, match="chronology differs"):
            epoch.pay_reward(wrong_ordinal)
        assert epoch._publication.current is paid
        assert epoch.commit_head == commit_head + 1
        assert clone_calls == 0

    assert epoch.pay_reward(1) is None
    assert epoch._publication.current is not paid
    assert epoch._publication.current.reward_paid[:, :2].all()
    assert clone_calls == 0


def test_open_reward_debt_blocks_reset_drain_and_checkpoint():
    epoch = E.ActionEpochOwner(num_envs=2, device="cpu")
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    reset = _RealSelectedResetHarness(epoch)
    epoch.bind_selected_reset_owner(reset.owner)
    epoch.open_reward_cycle()
    for operation in (epoch.prepare_drain, epoch.checkpoint):
        try:
            operation()
        except E.ActionEpochError:
            pass
        else:
            raise AssertionError("open Reward debt crossed a boundary")
    try:
        epoch.prepare_selected_true_reset(
            owner=reset.owner,
            top_preflight=object(),
            selected_env_index=torch.tensor([0], dtype=torch.int64),
            selected_mask=torch.tensor([True, False]),
            generation_before=torch.tensor([0, 0], dtype=torch.int64),
            generation_after=torch.tensor([1, 0], dtype=torch.int64),
            generation_overflow_fault=torch.tensor([False, False]),
            terminal_reset_facts_i64=_terminal_reset_facts(
                torch.tensor([True, False])
            ),
        )
    except E.ActionEpochError:
        pass
    else:
        raise AssertionError("selected reset crossed open Reward debt")
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    start, end = epoch.prepare_drain()
    epoch.materialize_drain(start=start, end=end)
    epoch.acknowledge_drain(start=start, end=end)
    checkpoint = epoch.checkpoint()
    checkpoint.current.phase.fill_(123)
    assert torch.all(epoch.current().phase.eq(E.PHASE_IDLE))


def _portable_epoch_source():
    epoch, d05, *_ = _ready_epoch()
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    epoch.open_reward_cycle()
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    epoch.publish_reward_payment(9)
    epoch.milestone.add_step_return(torch.tensor([1.25, -0.5]))
    epoch.milestone.add_first_fact_event(
        "r03_strike_fact", torch.tensor([[True], [False]])
    )
    epoch.milestone.add_r07_first_ready(torch.tensor([[False], [True]]))
    _materialized_entries(epoch)
    return epoch


def _portable_paid_epoch_source(monkeypatch):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(
        sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE
    )
    epoch, d05, _cadence, r06, playback, *_middle, physical = _ready_epoch(
        bind_playback=True
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    epoch.publish_motion_playback_started(owner=playback)
    physical.launch = _launch_packet(
        epoch.current(), due=torch.tensor([True, False])
    )
    epoch.refresh_physical_launch_rows()
    current = epoch.current()
    key = E.ActionEpochShotKey(**{
        field.name: getattr(current.identity.shot_key, field.name)[:, 0].clone()
        for field in E.fields(E.ActionEpochShotKey)
    })
    r06.outcome = ActionEpochR06OutcomeRows(
        valid=torch.tensor([True, False]),
        shot_key=key,
        publication_ordinal=current.publication_ordinal[:, 0].clone(),
        settlement_step=torch.tensor([12, -1], dtype=torch.int64),
        valid_bits=torch.tensor([1, 0], dtype=torch.int64),
        fact_values=torch.zeros(
            (2, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32
        ),
        outcome_code=torch.tensor([2, -1], dtype=torch.int64),
        owner_fault_bits=torch.zeros(2, dtype=torch.int64),
    )
    epoch.refresh_r06_outcome_rows()
    epoch.open_reward_cycle()
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    epoch.publish_reward_payment(20)
    _materialized_entries(epoch)
    return epoch


def _assert_record_tensors_equal(left, right):
    left_items = E._record_tensor_items(left)
    right_items = E._record_tensor_items(right)
    assert tuple(name for name, _ in left_items) == tuple(
        name for name, _ in right_items
    )
    assert all(
        torch.equal(left_value, right_value)
        for (_, left_value), (_, right_value) in zip(left_items, right_items)
    )


def test_epoch_private_carry_leaf_exposes_typed_values_without_milestone_pack():
    source = _portable_epoch_source()
    marker = object()
    source._lean_carry_coordinator = marker
    lease = types.SimpleNamespace(coordinator=marker, kind="capture")
    capture = source._lean_carry_capture(lease)
    schema = source._lean_carry_schema()
    assert schema.role == "epoch"
    assert len(capture.tensors) == len(schema.tensor_fields)
    assert "milestone" not in " ".join(field.name for field in schema.tensor_fields)
    host = tuple(value.detach().clone().contiguous() for value in capture.tensors)
    state = source._lean_carry_state_from_host(capture.scalars, host)
    assert state.commit_head == state.drain_frontier == source.commit_head
    assert not hasattr(E, "_single_d2h_checkpoint_views")


@pytest.mark.skip(reason="superseded by Lean-root composite carry transaction")
def test_private_epoch_carry_restores_record_frontier_payment_and_milestone():
    source = _portable_epoch_source()
    transfer_calls = []
    original_transfer = E._single_d2h_packed_bytes
    original_milestone_transfer = E.milestone_tensors._single_d2h_checkpoint_carry

    def counted_transfer(value):
        transfer_calls.append(value)
        return original_transfer(value)

    def forbidden_separate_milestone_transfer(_views):
        raise AssertionError("Epoch capture must include milestone in its packed D2H")

    E._single_d2h_packed_bytes = counted_transfer
    E.milestone_tensors._single_d2h_checkpoint_carry = (
        forbidden_separate_milestone_transfer
    )
    try:
        state = source._checkpoint_carry_state()
    finally:
        E._single_d2h_packed_bytes = original_transfer
        E.milestone_tensors._single_d2h_checkpoint_carry = (
            original_milestone_transfer
        )
    assert len(transfer_calls) == 1
    assert state.current.phase.device.type == "cpu"
    assert state.current.phase[0, 0] == E.PHASE_REVEAL_COMMITTED

    target, *_ = _ready_epoch()
    token = target._prepare_restore_carry(state)
    assert not hasattr(token, "_state")
    assert not hasattr(token, "_milestone_token")
    action_catalog = target._action_uids_by_slot
    family_catalog = target._family_codes_by_slot
    state.current.phase.fill_(123)  # prepared target must not alias its input
    target._commit_restore_carry(token)

    _assert_record_tensors_equal(source.current(), target.current())
    assert target.commit_head == source.commit_head
    assert target.drain_frontier == source.drain_frontier
    assert target._next_epoch == source._next_epoch
    assert target._reward_ordinal == E.REWARD_CONSUMER_COUNT
    assert torch.equal(
        target.project_current_reward_payment_rows().shot_key.action_uid,
        source.project_current_reward_payment_rows().shot_key.action_uid,
    )
    assert target.milestone.open_episode_return.tolist() == [1.25, -0.5]
    assert target.milestone.r03_seen[:, 0].tolist() == [True, False]
    assert target.milestone.r07_ready_seen[:, 0].tolist() == [False, True]
    assert target._action_uids_by_slot is action_catalog
    assert target._family_codes_by_slot is family_catalog
    assert target._reset_generation is target._publication.current.reset_generation
    assert target._reward_cycle_age is target._publication.current.reward_cycle_age
    assert target._reward_cycle_fault is target._publication.current.reward_cycle_fault
    with pytest.raises(E.ActionEpochError):
        target._commit_restore_carry(token)
    with pytest.raises(E.ActionEpochError):
        target._abort_restore_carry(token)
    target.open_reward_cycle()  # restored closed chronology starts the next cycle
    target.pay_reward(0)


@pytest.mark.skip(reason="superseded by central opaque authority and alias checks")
def test_private_epoch_restore_rejects_foreign_catalog_alias_and_token():
    source = _portable_epoch_source()
    state = source._checkpoint_carry_state()
    target, *_ = _ready_epoch()
    before = target.current()

    foreign_catalog = replace(
        state, family_codes_by_slot=torch.tensor([2], dtype=torch.int64)
    )
    with pytest.raises(E.ActionEpochError):
        target._prepare_restore_carry(foreign_catalog)
    _assert_record_tensors_equal(before, target.current())

    aliased = replace(
        state,
        last_r06_paid_payment_step=state.current_payment_rows.payment_step,
    )
    with pytest.raises(E.ActionEpochError):
        target._prepare_restore_carry(aliased)
    _assert_record_tensors_equal(before, target.current())

    non_tensor_catalog = replace(state, action_uids_by_slot=(21,))
    with pytest.raises(E.ActionEpochError):
        target._prepare_restore_carry(non_tensor_catalog)

    foreign_version = replace(
        state, current=replace(state.current, version=state.current.version + 1)
    )
    with pytest.raises(E.ActionEpochError):
        target._prepare_restore_carry(foreign_version)

    malformed_target, *_ = _ready_epoch()
    malformed_current = malformed_target._publication.current
    malformed_target._publication = E._Publication(
        replace(malformed_current, phase=malformed_current.phase[:, :0]),
        malformed_target._publication.pending_log,
    )
    with pytest.raises(E.ActionEpochError):
        malformed_target._prepare_restore_carry(state)

    overflow_target, *_ = _ready_epoch()
    overflow_target._undecoded_overflow[0] = True
    with pytest.raises(E.ActionEpochError):
        overflow_target._prepare_restore_carry(state)

    aliased_target, *_ = _ready_epoch()
    aliased_target.milestone.r07_ready_seen = aliased_target.milestone.r03_seen
    with pytest.raises(RuntimeError, match="tensors alias"):
        aliased_target._prepare_restore_carry(state)

    cross_aliased_target, *_ = _ready_epoch()
    cross_aliased_target.milestone.r03_seen = (
        cross_aliased_target._undecoded_overflow[:, None]
    )
    with pytest.raises(E.ActionEpochError, match="aliases"):
        cross_aliased_target._prepare_restore_carry(state)

    peer, *_ = _ready_epoch()
    token = target._prepare_restore_carry(state)
    with pytest.raises(E.ActionEpochError):
        peer._commit_restore_carry(token)
    target._abort_restore_carry(token)
    with pytest.raises(E.ActionEpochError):
        target._commit_restore_carry(token)
    with pytest.raises(E.ActionEpochError):
        target._abort_restore_carry(token)
    target.prepare_after_command_rows()  # abort reopens the original dormant owner


@pytest.mark.skip(reason="superseded by central typed schema validation")
def test_epoch_carry_rejects_foreign_chronology_and_nested_aliases():
    source = _portable_epoch_source()
    state = source._checkpoint_carry_state()
    target, *_ = _ready_epoch()

    slot = state.current.current_task_slot.clone()
    slot[0] = state.shot_slot_capacity
    publication = state.current.publication_ordinal.clone()
    publication[0, 0] = state.next_epoch
    key_reset = state.current.identity.shot_key.reset_generation.clone()
    key_reset[0, 0] += 1
    mutants = (
        replace(state, next_epoch=state.current.version + 1),
        replace(state, last_motion_common_step=-1),
        replace(state, current=replace(state.current, current_task_slot=slot)),
        replace(
            state, current=replace(state.current, publication_ordinal=publication)
        ),
        replace(state, current=replace(state.current, epoch=0)),
        replace(
            state,
            current=replace(
                state.current,
                identity=replace(
                    state.current.identity,
                    shot_key=replace(
                        state.current.identity.shot_key,
                        reset_generation=key_reset,
                    ),
                ),
            ),
        ),
        replace(
            state,
            milestone_carry=replace(
                state.milestone_carry,
                r03_seen=state.current.launch_succeeded,
            ),
        ),
        replace(
            state,
            milestone_carry=replace(
                state.milestone_carry,
                open_episode_return=state.current.reset_generation.view(
                    torch.float64
                ),
            ),
        ),
    )
    for mutant in mutants:
        with pytest.raises((E.ActionEpochError, RuntimeError)):
            target._prepare_restore_carry(mutant)

    live_alias = _portable_epoch_source()
    live_alias.milestone.r03_seen = (
        live_alias._publication.current.launch_succeeded
    )
    with pytest.raises(E.ActionEpochError):
        live_alias._checkpoint_carry_state()

    mirror_drift = _portable_epoch_source()
    mirror_drift._reset_generation = mirror_drift._reset_generation.clone()
    mirror_drift._reset_generation[0] += 1
    with pytest.raises(E.ActionEpochError, match="owner mirror"):
        mirror_drift._checkpoint_carry_state()


@pytest.mark.skip(reason="superseded by root cross-owner invariant phase")
def test_epoch_carry_rejects_foreign_payment_chronology(monkeypatch):
    state = _portable_paid_epoch_source(monkeypatch)._checkpoint_carry_state()
    assert state.current_payment_rows.valid.tolist() == [True, False]
    target, *_ = _ready_epoch()

    foreign_key = state.current_payment_rows.shot_key.clone()
    foreign_key.shot_index[0] += 1
    wrong_phase = state.current.phase.clone()
    wrong_phase[0, 0] = E.PHASE_REVEAL_COMMITTED
    wrong_step = state.current_payment_rows.payment_step.clone()
    wrong_step[0] += 1
    late_settlement = state.current.settlement_step.clone()
    late_settlement[0, 0] = state.current_payment_rows.payment_step[0] + 1
    absent_settlement = state.current.settlement_step.clone()
    absent_settlement[0, 0] = -1
    absent_due = state.current.reward_due.clone()
    absent_due[0].zero_()
    absent_paid = state.current.reward_paid.clone()
    absent_paid[0].zero_()
    missing_projection_valid = state.current_payment_rows.valid.clone()
    missing_projection_valid[0] = False
    missing_projection_step = state.current_payment_rows.payment_step.clone()
    missing_projection_step[0] = -1
    mutants = (
        replace(
            state,
            current_payment_rows=replace(
                state.current_payment_rows, shot_key=foreign_key
            ),
        ),
        replace(state, current=replace(state.current, phase=wrong_phase)),
        replace(
            state,
            current_payment_rows=replace(
                state.current_payment_rows, payment_step=wrong_step
            ),
        ),
        replace(
            state,
            current=replace(state.current, settlement_step=late_settlement),
        ),
        replace(
            state,
            current=replace(state.current, settlement_step=absent_settlement),
        ),
        replace(
            state,
            current=replace(
                state.current,
                reward_due=absent_due,
                reward_paid=absent_paid,
            ),
        ),
        replace(
            state,
            current_payment_rows=replace(
                state.current_payment_rows,
                valid=missing_projection_valid,
                payment_step=missing_projection_step,
            ),
        ),
    )
    for mutant in mutants:
        with pytest.raises(E.ActionEpochError):
            target._prepare_restore_carry(mutant)


@pytest.mark.skip(reason="superseded by graph-wide lease and process fail-stop")
def test_prepared_epoch_restore_blocks_writers_and_partial_commit_is_no_retry(
    monkeypatch,
):
    state = _portable_epoch_source()._checkpoint_carry_state()
    target, *_ = _ready_epoch()
    token = target._prepare_restore_carry(state)
    with pytest.raises(E.ActionEpochError):
        target.prepare_after_command_rows()

    milestone_type = type(target.milestone)
    original_commit = milestone_type._commit_restore_carry

    def fail_after_milestone_commit(owner, milestone_token):
        original_commit(owner, milestone_token)
        raise RuntimeError("injected child commit failure")

    monkeypatch.setattr(
        milestone_type, "_commit_restore_carry", fail_after_milestone_commit
    )
    with pytest.raises(RuntimeError, match="injected child commit failure"):
        target._commit_restore_carry(token)
    assert target.poisoned
    assert target.milestone.open_episode_return.tolist() == [1.25, -0.5]
    with pytest.raises(E.ActionEpochError):
        target._commit_restore_carry(token)
    with pytest.raises(E.ActionEpochError):
        target._abort_restore_carry(token)


@pytest.mark.skip(reason="superseded by root durable-ACK capture preflight")
def test_epoch_carry_capture_requires_ack_boundary_and_exact_live_abi():
    epoch, *_ = _ready_epoch()
    with pytest.raises(E.ActionEpochError):
        epoch._checkpoint_carry_state()  # genesis journal is not ACKed
    _materialized_entries(epoch)
    epoch._action_uids_by_slot = torch.empty(0, dtype=torch.int64)
    epoch._family_codes_by_slot = torch.empty(0, dtype=torch.int64)
    with pytest.raises(E.ActionEpochError):
        epoch._checkpoint_carry_state()


@pytest.mark.parametrize(
    "paid_fault",
    ["wrong_key", "future_payment", "current_future"],
)
def test_wrong_full_key_or_payment_chronology_cannot_retire(
    monkeypatch, paid_fault
):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    monkeypatch.setitem(sys.modules, FAKE_PHYSICAL_MODULE.__name__, FAKE_PHYSICAL_MODULE)
    epoch, d05, cadence, r06, playback, *_middle, physical = _ready_epoch(
        bind_playback=True
    )
    epoch.prepare_after_command_rows()
    epoch.settle_d05_transaction(d05.arm())
    epoch.publish_motion_playback_started(owner=playback)
    physical.launch = _launch_packet(
        epoch.current(), due=torch.tensor([True, False])
    )
    epoch.refresh_physical_launch_rows()
    current = epoch.current()
    key = E.ActionEpochShotKey(**{
        name: getattr(current.identity.shot_key, name)[:, 0].clone()
        for name in current.identity.shot_key.__dataclass_fields__
    })
    r06.outcome = ActionEpochR06OutcomeRows(
        valid=torch.tensor([True, False]),
        shot_key=key.clone(),
        publication_ordinal=current.publication_ordinal[:, 0].clone(),
        settlement_step=torch.tensor([12, -1], dtype=torch.int64),
        valid_bits=torch.tensor([1, 0], dtype=torch.int64),
        fact_values=torch.zeros((2, E.OWNER_FACT_F32_WIDTH), dtype=torch.float32),
        outcome_code=torch.tensor([2, -1], dtype=torch.int64),
        owner_fault_bits=torch.zeros(2, dtype=torch.int64),
    )
    epoch.refresh_r06_outcome_rows()
    epoch.open_reward_cycle()
    for ordinal in range(E.REWARD_CONSUMER_COUNT):
        epoch.pay_reward(ordinal)
    current_payment_step = 22 if paid_fault == "current_future" else 20
    paid = epoch.publish_reward_payment(current_payment_step)
    if paid_fault == "wrong_key":
        key.shot_index[0] += 1
    paid_step = 22 if paid_fault == "future_payment" else 20
    r06.previous = PreviousPaidActionEpochRows(
        valid=torch.tensor([True, False]),
        shot_key=key,
        publication_ordinal=epoch.current().publication_ordinal[:, 0].clone(),
        settlement_step=torch.tensor([12, -1], dtype=torch.int64),
        payment_step=torch.tensor([paid_step, -1], dtype=torch.int64),
    )
    cadence.projection = _MotionProjection(
        21,
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([True, False]),
        torch.tensor([E.MOTION_CLOSE_PLAYED_SUFFIX, 0], dtype=torch.int64),
    )
    epoch.prepare_after_command_rows()
    assert epoch.current().phase[0, 0] == E.PHASE_OUTCOME_SETTLED
    assert r06.consumed is not None and not r06.consumed.valid.any()
    if paid_fault != "wrong_key":
        epoch.settle_d05_transaction(d05.arm())
        start, end = epoch.prepare_drain()
        materialized = epoch.materialize_drain(start=start, end=end)
        assert materialized.overflow.tolist() == [True, False]


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
)
def test_core_d05_path_does_not_observe_tensor_scalars(monkeypatch, device):
    monkeypatch.setitem(sys.modules, FAKE_R06_MODULE.__name__, FAKE_R06_MODULE)
    epoch, d05, *_ = _ready_epoch(device=device)
    with _NoHostTensorObservation():
        epoch.prepare_after_command_rows()
        epoch.settle_d05_transaction(d05.arm())
