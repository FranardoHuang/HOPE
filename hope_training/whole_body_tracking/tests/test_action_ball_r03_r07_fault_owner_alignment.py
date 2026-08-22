"""Exact R03/R07 caller identity at the shared ActionEpoch fault plane."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (SOURCE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import action_ball_continuous_recovery_device as R07  # noqa: E402
import action_ball_strike_fact_device as R03  # noqa: E402
import test_action_ball_continuous_recovery_device as r07_fixture  # noqa: E402
import test_action_ball_motion_rowwise_accept_writer as motion_row  # noqa: E402


def _epoch():
    epoch_module = importlib.import_module(
        R03._exact_action_epoch_owner_type().__module__
    )
    owner = epoch_module.ActionEpochOwner(num_envs=2, device="cpu")
    owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    return owner


def _raw_bytes(value):
    if type(value) is torch.Tensor:
        tensor = value.detach().contiguous().cpu()
        return (str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes())
    if is_dataclass(value):
        return tuple(
            (field.name, _raw_bytes(getattr(value, field.name)))
            for field in fields(value)
        )
    if type(value) is tuple:
        return tuple(_raw_bytes(item) for item in value)
    return value


class _Racket:
    def __init__(self):
        self._action_ball_strike_fact_exact_eligibility = torch.ones(
            2, dtype=torch.bool
        )
        self._action_ball_strike_fact_target_validity = torch.ones(
            2, dtype=torch.bool
        )
        self.active = False

    def require_active_action_epoch_r03_writer(self):
        if not self.active:
            raise RuntimeError("R03 writer is inactive")


@contextmanager
def _active(racket):
    racket.active = True
    try:
        yield
    finally:
        racket.active = False


def _r03_identity(epoch):
    current = epoch.current()
    slot = current.current_task_slot[:, None]
    return R03.EpochR03RacketIdentity(
        reset_generation=current.reset_generation.clone(),
        action_uid=current.identity.action_uid.gather(1, slot).squeeze(1),
        action_slot=current.identity.action_slot.gather(1, slot).squeeze(1),
        task_identity=current.identity.task_identity.gather(1, slot).squeeze(1),
    )


def _real_settled_r03():
    (
        _motion,
        _d05_owner,
        epoch,
        d05_token,
        _row_record,
        _racket_peer,
        _physical_peer,
    ) = motion_row._install_real_d05_record(
        device=torch.device("cpu"), corrupt_accept_mask=False
    )
    active_d05 = epoch._active_d05
    epoch._active_d05 = None
    try:
        owner = R03.ActionBallStrikeFactDeviceCoordinator(
            num_envs=2, device="cpu", action_epoch_owner=epoch
        )
    finally:
        epoch._active_d05 = active_d05
    racket = _Racket()
    racket._action_ball_strike_fact_exact_eligibility[1] = False
    owner.bind_action_epoch_racket_owner(racket)
    epoch.settle_d05_transaction(d05_token)
    epoch_module = importlib.import_module(type(epoch).__module__)
    # This fixture owns fault-plane alignment, not launch chronology.  The
    # production Physical -> Epoch launch path is covered by the row-wise R03
    # test; set only the prerequisite phase here to keep this seam narrow.
    epoch._publication.current.phase[0, 0] = epoch_module.PHASE_LAUNCH_SETTLED
    assert epoch.current().phase[:, 0].tolist() == [
        epoch_module.PHASE_LAUNCH_SETTLED,
        epoch_module.PHASE_IDLE,
    ]
    return owner, epoch, racket, _r03_identity(epoch)


def _shot_key(*, shot_index=(11, 12)):
    values = torch.tensor(shot_index, dtype=torch.int64)
    return R07._row_identity.ActionEpochShotKey(
        reset_generation=torch.zeros(2, dtype=torch.int64),
        ball_generation=values + 10,
        action_uid=torch.full((2,), 101, dtype=torch.int64),
        action_slot=torch.zeros(2, dtype=torch.int64),
        shot_index=values.clone(),
        task_identity=values + 20,
        outcome_identity=values + 30,
        ball_identity=values + 40,
    )


def _ready_result(owner, *, step: int, kind: int):
    ready = torch.ones(2, dtype=torch.bool)
    return R07.R07EpochDirectRewardFacts(
        source_step=torch.full((2,), step, dtype=torch.int64),
        motion_cadence_tick=torch.full((2,), step, dtype=torch.int64),
        reset_generation=torch.zeros(2, dtype=torch.int64),
        recovery_age_tick=torch.full((2,), -1, dtype=torch.int64),
        reward_eligible=ready.clone(),
        facts_valid=ready.clone(),
        foot_supported_lr=torch.ones(
            (2, owner.num_feet), dtype=torch.bool
        ),
        infrastructure_fault=torch.zeros(2, dtype=torch.bool),
        producer_fault_bits=torch.zeros(2, dtype=torch.int64),
        component_errors=torch.zeros(
            (2, owner.num_components), dtype=owner.dtype
        ),
        raw_score=torch.ones(2, dtype=owner.dtype),
        weighted_reward=torch.ones(2, dtype=owner.dtype),
        ready_instant=ready,
        reference_kind=torch.full((2,), kind, dtype=torch.int64),
        reference_action_slot=torch.zeros(2, dtype=torch.int64),
        reference_action_uid=torch.full((2,), 101, dtype=torch.int64),
    )


def _ready_row_bytes(owner, row: int):
    tensors = (
        owner._action_epoch_ready_instant,
        owner._action_epoch_ready_live,
        owner._action_epoch_ready_streak,
        owner._action_epoch_ready_reference_kind,
        owner._action_epoch_ready_reference_action_slot,
        owner._action_epoch_ready_reference_action_uid,
        owner._action_epoch_first_ready_source_step,
        owner._action_epoch_ready_last_motion_cadence_tick,
        owner._action_epoch_ready_last_reset_generation,
        *(
            getattr(owner._action_epoch_ready_shot_key, field.name)
            for field in fields(R07._row_identity.ActionEpochShotKey)
        ),
    )
    return tuple(tensor[row].contiguous().numpy().tobytes() for tensor in tensors)


def test_r03_real_publish_and_epoch_gate_require_the_bound_coordinator():
    owner, epoch, racket, identity = _real_settled_r03()
    epoch_module = importlib.import_module(type(epoch).__module__)
    foreign = R03.ActionBallStrikeFactDeviceCoordinator(
        num_envs=2, device="cpu"
    )
    bits = torch.tensor([[4], [0]], dtype=torch.int64)
    before_head = epoch.commit_head
    before = _raw_bytes(epoch.current())

    for caller in (foreign, object()):
        with pytest.raises(
            epoch_module.ActionEpochError, match="owner identity differs"
        ):
            epoch.merge_runtime_owner_fault(
                "r03_strike_fact", bits, owner=caller
            )
        assert epoch.commit_head == before_head
        assert _raw_bytes(epoch.current()) == before

    step = torch.full((2,), 7, dtype=torch.int64)
    vectors = {
        name: torch.arange(offset, offset + 6, dtype=torch.float32).reshape(2, 3)
        for offset, name in enumerate(
            (
                "target_position",
                "target_velocity",
                "target_face_normal",
                "ball_position",
                "ball_velocity",
                "achieved_position",
                "achieved_velocity",
                "achieved_face_normal",
            ),
            start=1,
        )
    }
    with _active(racket):
        owner.arm_action_epoch_strike_fact_v1(
            racket_owner=racket,
            source_step=step,
            racket_identity=identity,
            target_position=vectors["target_position"],
            target_velocity=vectors["target_velocity"],
            target_face_normal=vectors["target_face_normal"],
            ball_position=vectors["ball_position"],
            ball_velocity=vectors["ball_velocity"],
        )
    after_arm = epoch.commit_head
    assert after_arm == before_head + 1
    with _active(racket):
        owner.publish_action_epoch_strike_fact_v1(
            racket_owner=racket,
            source_step=step,
            racket_identity=identity,
            achieved_position=vectors["achieved_position"],
            achieved_velocity=vectors["achieved_velocity"],
            achieved_face_normal=vectors["achieved_face_normal"],
        )
    assert epoch.commit_head == after_arm + 2

    record = epoch.current()
    owner_slot = epoch_module.OWNER_ORDER.index("r03_strike_fact")
    row = torch.arange(2, dtype=torch.int64)
    slot = record.current_task_slot
    published = record.fact_f32[row, slot, owner_slot]
    expected = torch.cat(tuple(vectors.values()), dim=1)
    assert published.shape[1] == epoch_module.OWNER_FACT_F32_WIDTH
    assert torch.equal(published[0, : R03.R03_EPOCH_FACT_VALUE_COUNT], expected[0])
    assert torch.count_nonzero(
        published[0, R03.R03_EPOCH_FACT_VALUE_COUNT :]
    ).item() == 0
    assert torch.count_nonzero(published[1]).item() == 0
    assert record.fact_valid_bits[row, slot, owner_slot].tolist() == [
        R03.R03_EPOCH_FACT_PRESENT | R03.R03_EPOCH_FACT_PHYSICALLY_VALID,
        0,
    ]
    assert record.fact_source_step[row, slot, owner_slot].tolist() == [7, -1]
    assert record.owner_fault_bits[row, slot, owner_slot].tolist() == [0, 0]

    epoch.merge_runtime_owner_fault("r03_strike_fact", bits, owner=owner)
    record = epoch.current()
    assert record.owner_fault_bits[row, slot, owner_slot].tolist() == [4, 0]


def test_r03_stale_arm_rejects_next_full_key_with_same_action_slot():
    owner, epoch, racket, identity = _real_settled_r03()
    step = torch.ones(2, dtype=torch.int64)
    zero = torch.zeros((2, 3), dtype=torch.float32)
    with _active(racket):
        owner.arm_action_epoch_strike_fact_v1(
            racket_owner=racket,
            source_step=step,
            racket_identity=identity,
            target_position=zero,
            target_velocity=zero,
            target_face_normal=zero,
            ball_position=zero,
            ball_velocity=zero,
        )
    epoch_module = importlib.import_module(type(epoch).__module__)
    r03_slot = epoch_module.OWNER_ORDER.index("r03_strike_fact")
    peer_before = tuple(
        tensor[1].contiguous().numpy().tobytes()
        for tensor in (
            epoch.current().owner_fault_bits[:, :, r03_slot],
            epoch.current().fact_valid_bits[:, :, r03_slot],
            epoch.current().fact_source_step[:, :, r03_slot],
            epoch.current().fact_f32[:, :, r03_slot],
        )
    )

    # Adversarially replace only shot_index.  action_uid/action_slot remain
    # identical, so a legacy two-field or scalar-epoch join would accept it.
    retained_uid = epoch._publication.current.identity.action_uid[0, 0].item()
    retained_slot = epoch._publication.current.identity.action_slot[0, 0].item()
    epoch._publication.current.identity.shot_key.shot_index[0, 0].add_(1)
    assert epoch._publication.current.identity.action_uid[0, 0].item() == retained_uid
    assert epoch._publication.current.identity.action_slot[0, 0].item() == retained_slot
    with _active(racket):
        owner.publish_action_epoch_strike_fact_v1(
            racket_owner=racket,
            source_step=step,
            racket_identity=identity,
            achieved_position=zero,
            achieved_velocity=zero,
            achieved_face_normal=zero,
        )
    record = epoch.current()
    assert (
        record.owner_fault_bits[0, 0, r03_slot].item()
        & R03.R03_EPOCH_FAULT_EPOCH_IDENTITY
    )
    assert record.fact_valid_bits[0, 0, r03_slot].item() == 0
    peer_after = tuple(
        tensor[1].contiguous().numpy().tobytes()
        for tensor in (
            record.owner_fault_bits[:, :, r03_slot],
            record.fact_valid_bits[:, :, r03_slot],
            record.fact_source_step[:, :, r03_slot],
            record.fact_f32[:, :, r03_slot],
        )
    )
    assert peer_after == peer_before


@pytest.mark.parametrize(
    ("fault_kind", "expected_fault"),
    (
        ("stale_source", R03.R03_EPOCH_FAULT_STALE_SOURCE_STEP),
        ("nonfinite", R03.R03_EPOCH_FAULT_NONFINITE_FACT),
    ),
)
def test_r03_real_arm_types_stale_and_nonfinite_without_touching_peer(
    fault_kind, expected_fault
):
    owner, epoch, racket, identity = _real_settled_r03()
    step = torch.ones(2, dtype=torch.int64)
    target_position = torch.zeros((2, 3), dtype=torch.float32)
    if fault_kind == "stale_source":
        step[0] = -1
    else:
        target_position[0, 0] = torch.nan
    zero = torch.zeros((2, 3), dtype=torch.float32)
    owner_slot = importlib.import_module(type(epoch).__module__).OWNER_ORDER.index(
        "r03_strike_fact"
    )
    peer_before = tuple(
        tensor[1].contiguous().numpy().tobytes()
        for tensor in (
            epoch.current().owner_fault_bits[:, :, owner_slot],
            epoch.current().fact_valid_bits[:, :, owner_slot],
            epoch.current().fact_source_step[:, :, owner_slot],
            epoch.current().fact_f32[:, :, owner_slot],
        )
    )
    with _active(racket):
        owner.arm_action_epoch_strike_fact_v1(
            racket_owner=racket,
            source_step=step,
            racket_identity=identity,
            target_position=target_position,
            target_velocity=zero,
            target_face_normal=zero,
            ball_position=zero,
            ball_velocity=zero,
        )
    record = epoch.current()
    assert record.owner_fault_bits[0, 0, owner_slot].item() & expected_fault
    assert record.fact_valid_bits[0, 0, owner_slot].item() == 0
    peer_after = tuple(
        tensor[1].contiguous().numpy().tobytes()
        for tensor in (
            record.owner_fault_bits[:, :, owner_slot],
            record.fact_valid_bits[:, :, owner_slot],
            record.fact_source_step[:, :, owner_slot],
            record.fact_f32[:, :, owner_slot],
        )
    )
    assert peer_after == peer_before


def test_r07_epoch_gate_rejects_same_type_foreign_bundle_without_mutation():
    epoch = _epoch()
    epoch_module = importlib.import_module(type(epoch).__module__)
    coordinator = r07_fixture._owner(num_envs=2)
    bundle = R07.DiagnosticN2ContinuousRecoveryBundle(
        owner=coordinator,
        plant_fact_adapter=object(),
        motion_owner=object(),
        action_epoch_owner=epoch,
        motion_parent_authority=object(),
        motion_parent_receipt=object(),
    )
    coordinator._diagnostic_n2_bundle = bundle
    epoch.bind_fact_owner("r07_recovery", bundle)
    foreign = replace(bundle)
    assert type(foreign) is type(bundle) and foreign is not bundle
    bits = torch.tensor([[0], [2]], dtype=torch.int64)
    before_head = epoch.commit_head
    before = _raw_bytes(epoch.current())

    for caller in (foreign, object()):
        with pytest.raises(
            epoch_module.ActionEpochError, match="owner identity differs"
        ):
            epoch.merge_runtime_owner_fault(
                "r07_recovery", bits, owner=caller
            )
        assert epoch.commit_head == before_head
        assert _raw_bytes(epoch.current()) == before

    epoch.merge_runtime_owner_fault("r07_recovery", bits, owner=bundle)
    record = epoch.current()
    slot = epoch_module.OWNER_ORDER.index("r07_recovery")
    assert record.owner_fault_bits[:, 0, slot].tolist() == [0, 2]


def test_r07_readiness_uses_full_key_and_selected_reset_preserves_peer_bytes():
    epoch = _epoch()
    owner = r07_fixture._owner(num_envs=2)
    bundle = R07.DiagnosticN2ContinuousRecoveryBundle(
        owner=owner,
        plant_fact_adapter=object(),
        motion_owner=object(),
        action_epoch_owner=epoch,
        motion_parent_authority=object(),
        motion_parent_receipt=object(),
    )
    owner._diagnostic_n2_bundle = bundle
    epoch.bind_fact_owner("r07_recovery", bundle)
    empty = R07._row_identity.empty_action_epoch_shot_key(
        (2,), device=owner.device
    )
    owner._publish_action_epoch_motion_readiness(
        _ready_result(
            owner,
            step=0,
            kind=R07.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0,
        ),
        observed_source_step=0,
        shot_key=empty,
    )
    owner._publish_action_epoch_motion_readiness(
        _ready_result(
            owner,
            step=1,
            kind=R07.R07_REFERENCE_BOOTSTRAP_UPCOMING_ACTION_FRAME0,
        ),
        observed_source_step=1,
        shot_key=empty,
    )
    assert owner._action_epoch_ready_streak.tolist() == [2, 2]

    first = _shot_key()
    owner._publish_action_epoch_motion_readiness(
        _ready_result(
            owner,
            step=2,
            kind=R07.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
        ),
        observed_source_step=2,
        shot_key=first,
    )
    assert owner._action_epoch_ready_streak.tolist() == [1, 1]
    owner._publish_action_epoch_motion_readiness(
        _ready_result(
            owner,
            step=3,
            kind=R07.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
        ),
        observed_source_step=3,
        shot_key=first,
    )
    assert owner._action_epoch_ready_streak.tolist() == [2, 2]

    # The next row-0 shot deliberately keeps action_uid=101/action_slot=0;
    # only the full business key reveals that it is a different shot.
    second = _shot_key(shot_index=(13, 12))
    owner._publish_action_epoch_motion_readiness(
        _ready_result(
            owner,
            step=4,
            kind=R07.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
        ),
        observed_source_step=4,
        shot_key=second,
    )
    assert owner._action_epoch_ready_streak.tolist() == [1, 3]
    before_replay = _raw_bytes(owner._action_epoch_ready_shot_key)
    before_streak = owner._action_epoch_ready_streak.clone()
    with pytest.raises(
        R07.ContinuousRecoveryDeviceError, match="skipped, stale, or replayed"
    ):
        owner._publish_action_epoch_motion_readiness(
            _ready_result(
                owner,
                step=4,
                kind=R07.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
            ),
            observed_source_step=4,
            shot_key=second,
        )
    assert torch.equal(owner._action_epoch_ready_streak, before_streak)
    assert _raw_bytes(owner._action_epoch_ready_shot_key) == before_replay

    peer_before = _ready_row_bytes(owner, 1)
    owner.reset_true_boundary([0])
    assert _ready_row_bytes(owner, 1) == peer_before
    assert not owner._action_epoch_ready_instant[0]
    assert not owner._action_epoch_ready_live[0]
    assert owner._action_epoch_ready_streak[0].item() == 0
    for field in fields(R07._row_identity.ActionEpochShotKey):
        assert getattr(owner._action_epoch_ready_shot_key, field.name)[0].item() == -1
