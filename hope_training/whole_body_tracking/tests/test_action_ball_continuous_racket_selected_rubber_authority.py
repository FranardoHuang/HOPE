"""Focused contracts for legacy and direct ActionEpoch selected-rubber views.

The legacy token publisher remains fail-closed.  The lean ActionEpoch lane
instead pulls an exact live Physical allocation directly during its launch
transaction and joins it to Racket's cold mount-sign table without a caller
mask, slot, digest, or verdict.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import test_action_ball_continuous_racket_observation_projection as obs_test


HC = obs_test.HC


def _load_scene_module():
    name = "whole_body_tracking.tasks.tracking.config.agibot_a3.action_ball_full_mdp_ball_scene"
    if name in sys.modules:
        return sys.modules[name]
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3"
        / "action_ball_full_mdp_ball_scene.py"
    )
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _direct_epoch_racket(device: str, *, selected=(True, True)):
    physical = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_physical_flight_device"
    )
    epoch = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
    )
    target = torch.device(device)
    owner = epoch.ActionEpochOwner(
        num_envs=2, device=target, shot_slot_capacity=1,
        initial_reset_generation=torch.ones(2, dtype=torch.int64, device=target),
    )
    action_uid = torch.tensor([[101], [202]], dtype=torch.int64, device=target)
    action_slot = torch.tensor([[0], [1]], dtype=torch.int64, device=target)
    selected_tensor = torch.tensor(
        selected, dtype=torch.bool, device=target
    )
    published = selected_tensor[:, None]

    def published_i64(value: torch.Tensor) -> torch.Tensor:
        return torch.where(published, value, torch.full_like(value, -1))

    shot_key = epoch.ActionEpochShotKey(
        reset_generation=published_i64(torch.ones_like(action_uid)),
        ball_generation=published_i64(
            torch.tensor([[7], [8]], dtype=torch.int64, device=target)
        ),
        action_uid=published_i64(action_uid),
        action_slot=published_i64(action_slot),
        shot_index=published_i64(
            torch.tensor([[1], [2]], dtype=torch.int64, device=target)
        ),
        task_identity=published_i64(action_uid + 1000),
        outcome_identity=published_i64(action_uid + 3000),
        ball_identity=published_i64(action_uid + 2000),
    )
    identity = epoch.EpochIdentityPayload(
        shot_key=shot_key,
        scheduled_ordinal=published_i64(action_uid + 4000),
        target_generation=published_i64(action_uid + 5000),
        selected_cell=published_i64(torch.zeros_like(action_uid)),
        candidate_identity=published_i64(action_uid + 6000),
    )
    owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=target),
        reset_generation=torch.ones(2, dtype=torch.int64, device=target),
    )
    record = owner.current()
    mixed_phase = torch.where(
        selected_tensor[:, None],
        torch.full_like(record.phase, epoch.PHASE_REVEAL_COMMITTED),
        torch.full_like(record.phase, epoch.PHASE_IDLE),
    )
    mixed_slot = torch.where(
        selected_tensor,
        record.current_task_slot,
        torch.full_like(record.current_task_slot, -1),
    )
    owner._publication = epoch._Publication(
        current=replace(
            record,
            identity=identity,
            phase=mixed_phase,
            current_task_slot=mixed_slot,
            publication_ordinal=published_i64(
                torch.tensor([[1], [2]], dtype=torch.int64, device=target)
            ),
        ),
        pending_log=owner._publication.pending_log,
    )

    physical_owner = physical.ActionBallPhysicalFlightDeviceOwner.__new__(
        physical.ActionBallPhysicalFlightDeviceOwner
    )
    physical_owner.num_envs = 2
    physical_owner.flight_capacity = 2
    physical_owner.device = target
    physical_owner._action_epoch_owner = owner
    physical_owner._owner_identity = object()
    physical_owner._action_epoch_active_physics_fact_allocation = None

    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = target
    racket.cfg = SimpleNamespace(
        mount_normal_sign_per_clip=(1, -1), mount_normal_sign=1
    )
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    racket._action_ball_continuous_racket_poisoned = False
    racket._action_ball_full_mdp_racket_epoch_owner = owner
    racket._action_ball_full_mdp_racket_physical_owner = None
    racket._action_ball_full_mdp_racket_mount_sign_table = None
    racket.bind_action_ball_full_mdp_racket_selected_rubber_physical_owner(
        physical_owner
    )
    assert racket._action_ball_full_mdp_racket_physical_owner is physical_owner
    assert torch.equal(
        racket._action_ball_full_mdp_racket_mount_sign_table,
        torch.tensor([1, -1], dtype=torch.int8, device=target),
    )
    return racket, physical_owner, owner, physical


def _allocation_identity(owner, active):
    record = owner.current()
    source = record.identity.shot_key
    slot = record.current_task_slot.clamp(min=0)
    index = slot[:, None]

    def expand_selected(value: torch.Tensor) -> torch.Tensor:
        selected = torch.gather(value, 1, index).expand_as(active)
        return torch.where(active, selected, torch.full_like(selected, -1))

    shot_key = type(source)(
        reset_generation=expand_selected(source.reset_generation),
        ball_generation=expand_selected(source.ball_generation),
        action_uid=expand_selected(source.action_uid),
        action_slot=expand_selected(source.action_slot),
        shot_index=expand_selected(source.shot_index),
        task_identity=expand_selected(source.task_identity),
        outcome_identity=expand_selected(source.outcome_identity),
        ball_identity=expand_selected(source.ball_identity),
    )
    publication_ordinal = expand_selected(record.publication_ordinal)
    return shot_key, publication_ordinal


def _set_direct_allocation(
    *, racket, physical_owner, owner, physical, active
):
    target = torch.device(racket.device)
    full_key = torch.zeros(2, 2, 32, dtype=torch.uint8, device=target)
    full_key[active] = 17
    shot_key, publication_ordinal = _allocation_identity(owner, active)
    physical_owner._action_epoch_active_physics_fact_allocation = (
        physical.ActionEpochPhysicsFactAllocationProjection(
            active_mask=active,
            launch_due_mask=active.clone(),
            flight_slot=torch.tensor(
                [[0, 1], [0, 1]], dtype=torch.int64, device=target
            ),
            shot_key=shot_key,
            publication_ordinal=publication_ordinal,
            full_key_sha256=full_key,
            physical_owner=physical_owner,
            epoch_owner=owner,
            owner_identity=physical_owner._owner_identity,
            _token=physical._ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN,
        )
    )


def _direct_scene_port(
    *, scene, target, active, physical_owner, owner, racket
):
    fact_owner = scene.IsaacPhysxBallFactOwner.__new__(
        scene.IsaacPhysxBallFactOwner
    )
    fact_owner.num_envs = 2
    fact_owner.flight_capacity = 2
    fact_owner.device = target
    fact_owner._bound_authority = None
    fact_owner._action_epoch_direct_binding = False
    fact_owner._expected_active = torch.zeros_like(active)
    fact_owner._expected_rubber = torch.full(
        (2, 2), -1, dtype=torch.int8, device=target
    )
    fact_owner._expected_generation = torch.full(
        (2, 2), -1, dtype=torch.int64, device=target
    )
    fact_owner._expected_key = torch.zeros(
        2, 2, 32, dtype=torch.uint8, device=target
    )
    fact_owner._expected_action_epoch_shot_key = {
        name: torch.full((2, 2), -1, dtype=torch.int64, device=target)
        for name in scene._ACTION_EPOCH_SHOT_KEY_FIELDS
    }
    fact_owner._expected_action_epoch_publication_ordinal = torch.full(
        (2, 2), -1, dtype=torch.int64, device=target
    )
    fact_owner._previous_valid = torch.zeros_like(active)
    fact_owner._contact_latch = torch.zeros_like(active)
    fact_owner._ordinary_invalid_contact_latch = torch.zeros_like(active)
    fact_owner._net_latch = torch.zeros_like(active)
    fact_owner._landing_latch = torch.zeros_like(active)
    fact_owner._contact_candidate_event = torch.zeros_like(active)
    fact_owner._known_non_rubber_candidate_event = torch.zeros_like(active)
    fact_owner._racket_invalid_contact_candidate_event = torch.zeros_like(active)
    fact_owner._binding_fault = torch.zeros_like(active)
    port = scene.IsaacLabPhysicalFlightScenePort.__new__(
        scene.IsaacLabPhysicalFlightScenePort
    )
    port.num_envs = 2
    port.flight_capacity = 2
    port.device = target
    port._action_epoch_physical_owner = physical_owner
    port._action_epoch_owner = owner
    port._action_epoch_racket_owner = racket
    port._physx_fact_owner = fact_owner
    return port, fact_owner


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize(
    "selected,active,expected",
    (
        (
            (False, False),
            ((False, False), (False, False)),
            ((-1, -1), (-1, -1)),
        ),
        (
            (True, False),
            ((True, False), (False, False)),
            ((0, -1), (-1, -1)),
        ),
    ),
)
def test_direct_epoch_selected_rubber_accepts_idle_empty_rows(
    device, selected, active, expected
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    racket, physical_owner, owner, physical = _direct_epoch_racket(
        device, selected=selected
    )
    target = torch.device(device)
    active_tensor = torch.tensor(active, dtype=torch.bool, device=target)
    _set_direct_allocation(
        racket=racket,
        physical_owner=physical_owner,
        owner=owner,
        physical=physical,
        active=active_tensor,
    )

    view = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()

    assert torch.equal(view.active_mask, active_tensor)
    assert torch.equal(
        view.expected_rubber,
        torch.tensor(expected, dtype=torch.int8, device=target),
    )


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize(
    "selected,active,expected",
    (
        (
            (False, False),
            ((False, False), (False, False)),
            ((-1, -1), (-1, -1)),
        ),
        (
            (True, False),
            ((True, False), (False, False)),
            ((0, -1), (-1, -1)),
        ),
    ),
)
def test_scene_noarg_chain_accepts_idle_empty_rows(
    device, selected, active, expected
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    scene = _load_scene_module()
    racket, physical_owner, owner, physical = _direct_epoch_racket(
        device, selected=selected
    )
    target = torch.device(device)
    active_tensor = torch.tensor(active, dtype=torch.bool, device=target)
    _set_direct_allocation(
        racket=racket,
        physical_owner=physical_owner,
        owner=owner,
        physical=physical,
        active=active_tensor,
    )
    port, fact_owner = _direct_scene_port(
        scene=scene,
        target=target,
        active=active_tensor,
        physical_owner=physical_owner,
        owner=owner,
        racket=racket,
    )

    port.arm_action_epoch_physics_fact_source()

    assert fact_owner._action_epoch_direct_binding is True
    assert not torch.any(fact_owner._binding_fault)
    assert torch.equal(fact_owner._expected_active, active_tensor)
    assert torch.equal(
        fact_owner._expected_rubber,
        torch.tensor(expected, dtype=torch.int8, device=target),
    )


def test_scene_noarg_chain_marks_only_changed_identity_row_as_binding_fault():
    scene = _load_scene_module()
    racket, physical_owner, owner, physical = _direct_epoch_racket("cpu")
    active = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    _set_direct_allocation(
        racket=racket,
        physical_owner=physical_owner,
        owner=owner,
        physical=physical,
        active=active,
    )
    allocation = physical_owner._action_epoch_active_physics_fact_allocation
    allocation.reset_generation[0, 0] = 2
    port, fact_owner = _direct_scene_port(
        scene=scene,
        target=torch.device("cpu"),
        active=active,
        physical_owner=physical_owner,
        owner=owner,
        racket=racket,
    )

    port.arm_action_epoch_physics_fact_source()

    assert torch.equal(
        fact_owner._binding_fault,
        torch.tensor([[True, False], [False, False]], dtype=torch.bool),
    )
    assert torch.equal(fact_owner._expected_active, active)
    assert torch.equal(
        fact_owner._expected_rubber,
        torch.tensor([[-1, -1], [-1, 1]], dtype=torch.int8),
    )


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_direct_epoch_selected_rubber_mixed_red_black_and_inactive(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    racket, physical_owner, owner, physical = _direct_epoch_racket(device)
    target = torch.device(device)
    active = torch.tensor(
        [[True, False], [False, True]], dtype=torch.bool, device=target
    )
    shot_key, publication_ordinal = _allocation_identity(owner, active)
    physical_owner._action_epoch_active_physics_fact_allocation = (
        physical.ActionEpochPhysicsFactAllocationProjection(
            active_mask=active,
            launch_due_mask=active.clone(),
            flight_slot=torch.tensor(
                [[0, 1], [0, 1]], dtype=torch.int64, device=target
            ),
            shot_key=shot_key,
            publication_ordinal=publication_ordinal,
            full_key_sha256=torch.zeros(
                2, 2, 32, dtype=torch.uint8, device=target
            ),
            physical_owner=physical_owner,
            epoch_owner=owner,
            owner_identity=physical_owner._owner_identity,
            _token=physical._ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN,
        )
    )

    view = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()
    assert type(view) is HC.ActionBallFullMdpActionEpochSelectedRubberView
    assert view.racket_owner is racket
    assert view.physical_owner is physical_owner
    assert view.epoch_owner is owner
    assert torch.equal(view.active_mask, active)
    assert torch.equal(
        view.expected_rubber,
        torch.tensor([[0, -1], [-1, 1]], dtype=torch.int8, device=target),
    )

    allocation = physical_owner._action_epoch_active_physics_fact_allocation
    allocation.reset_generation[0, 0] = 2
    neutral = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()
    assert torch.equal(neutral.active_mask, active)
    assert torch.equal(
        neutral.expected_rubber,
        torch.tensor([[-1, -1], [-1, 1]], dtype=torch.int8, device=target),
    )
    # The contradictory row becomes an inactive-face sentinel; the independent
    # healthy row keeps its selected black face byte-for-byte.
    assert neutral.expected_rubber[0, 0].item() == -1
    assert (
        neutral.expected_rubber[1, 1].item()
        == view.expected_rubber[1, 1].item()
    )
    allocation.reset_generation[0, 0] = 1

    allocation.active_mask.zero_()
    allocation.launch_due_mask.zero_()
    for name in (
        "reset_generation",
        "ball_generation",
        "action_uid",
        "action_slot",
        "shot_index",
        "task_identity",
        "outcome_identity",
        "ball_identity",
    ):
        getattr(allocation.shot_key, name).fill_(-1)
    allocation.publication_ordinal.fill_(-1)
    empty = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()
    assert not torch.any(empty.active_mask)
    assert torch.equal(
        empty.expected_rubber,
        torch.full((2, 2), -1, dtype=torch.int8, device=target),
    )

    physical_owner._action_epoch_active_physics_fact_allocation = None
    with pytest.raises(
        HC.ActionBallContinuousRacketSelectedRubberHold,
        match="inactive or stale",
    ):
        racket.action_ball_full_mdp_action_epoch_selected_rubber_view()


def test_direct_epoch_selected_rubber_avoids_nonportable_async_assert(
    monkeypatch,
):
    """Supported host Torch 2.0.1 accepts only the condition argument."""

    racket, physical_owner, owner, physical = _direct_epoch_racket("cpu")
    active = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    _set_direct_allocation(
        racket=racket,
        physical_owner=physical_owner,
        owner=owner,
        physical=physical,
        active=active,
    )
    calls = []

    def torch20_assert_async(condition):
        calls.append(condition)

    monkeypatch.setattr(torch, "_assert_async", torch20_assert_async)
    view = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()

    assert calls == []
    assert torch.equal(view.active_mask, active)
    assert torch.equal(
        view.expected_rubber,
        torch.tensor([[0, -1], [-1, 1]], dtype=torch.int8),
    )


def test_direct_epoch_selected_rubber_uses_one_narrow_projection_not_record_clone(
    monkeypatch,
):
    racket, physical_owner, owner, physical = _direct_epoch_racket("cpu")
    active = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    _set_direct_allocation(
        racket=racket,
        physical_owner=physical_owner,
        owner=owner,
        physical=physical,
        active=active,
    )
    epoch = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
    )
    project = owner.project_current_shot
    calls = []

    def counted_projection():
        calls.append(True)
        return project()

    def forbidden_record_clone(_record):
        raise AssertionError("selected-rubber must not clone the full Epoch record")

    monkeypatch.setattr(owner, "project_current_shot", counted_projection)
    monkeypatch.setattr(epoch.ActionEpochRecord, "clone", forbidden_record_clone)

    view = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()

    assert calls == [True]
    assert torch.equal(view.active_mask, active)
    assert torch.equal(
        view.expected_rubber,
        torch.tensor([[0, -1], [-1, 1]], dtype=torch.int8),
    )


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_direct_epoch_selected_rubber_rejects_foreign_physical(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    racket, physical_owner, owner, physical = _direct_epoch_racket(device)
    target = torch.device(device)
    foreign = physical.ActionBallPhysicalFlightDeviceOwner.__new__(
        physical.ActionBallPhysicalFlightDeviceOwner
    )
    foreign.num_envs = 2
    foreign.flight_capacity = 2
    foreign.device = target
    foreign._action_epoch_owner = owner
    foreign._owner_identity = object()
    foreign._action_epoch_active_physics_fact_allocation = None
    inactive = torch.zeros(2, 2, dtype=torch.bool, device=target)
    shot_key, publication_ordinal = _allocation_identity(owner, inactive)
    physical_owner._action_epoch_active_physics_fact_allocation = (
        physical.ActionEpochPhysicsFactAllocationProjection(
            active_mask=inactive,
            launch_due_mask=inactive.clone(),
            flight_slot=torch.tensor(
                [[0, 1], [0, 1]], dtype=torch.int64, device=target
            ),
            shot_key=shot_key,
            publication_ordinal=publication_ordinal,
            full_key_sha256=torch.zeros(
                2, 2, 32, dtype=torch.uint8, device=target
            ),
            physical_owner=foreign,
            epoch_owner=owner,
            owner_identity=physical_owner._owner_identity,
            _token=physical._ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN,
        )
    )
    with pytest.raises(
        HC.ActionBallContinuousRacketSelectedRubberHold,
        match="inactive or stale",
    ):
        racket.action_ball_full_mdp_action_epoch_selected_rubber_view()


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("reset_generation", 2),
        ("action_uid", 999),
        ("action_slot", 1),
        ("task_identity", 999),
    ),
)
def test_direct_epoch_selected_rubber_neutralizes_changed_identity_only(
    field, replacement
):
    racket, physical_owner, owner, physical = _direct_epoch_racket("cpu")
    active = torch.tensor([[True, False], [False, True]])
    shot_key, publication_ordinal = _allocation_identity(owner, active)
    getattr(shot_key, field)[0, 0] = replacement
    physical_owner._action_epoch_active_physics_fact_allocation = (
        physical.ActionEpochPhysicsFactAllocationProjection(
            active_mask=active,
            launch_due_mask=active.clone(),
            flight_slot=torch.tensor([[0, 1], [0, 1]]),
            shot_key=shot_key,
            publication_ordinal=publication_ordinal,
            full_key_sha256=torch.zeros(2, 2, 32, dtype=torch.uint8),
            physical_owner=physical_owner,
            epoch_owner=owner,
            owner_identity=physical_owner._owner_identity,
            _token=physical._ACTION_EPOCH_PHYSICS_FACT_ALLOCATION_TOKEN,
        )
    )
    view = racket.action_ball_full_mdp_action_epoch_selected_rubber_view()

    assert torch.equal(view.active_mask, active)
    assert torch.equal(
        view.expected_rubber,
        torch.tensor([[-1, -1], [-1, 1]], dtype=torch.int8),
    )


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_selected_rubber_mapping_is_device_only_and_marks_invalid_rows(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    target = torch.device(device)
    slots = torch.tensor([[0, 1], [1, 7]], dtype=torch.int64, device=target)
    active = torch.tensor(
        [[True, True], [False, True]], dtype=torch.bool, device=target
    )
    signs = torch.tensor([1, -1], dtype=torch.int8, device=target)

    rubber, invalid = HC._action_ball_full_mdp_expected_rubber_from_action_slot(
        slots, active, signs
    )

    assert rubber.dtype == torch.int8
    assert torch.equal(
        rubber,
        torch.tensor([[0, 1], [-1, 1]], dtype=torch.int8, device=target),
    )
    assert torch.equal(
        invalid,
        torch.tensor(
            [[False, False], [False, True]], dtype=torch.bool, device=target
        ),
    )


def _racket(*, committed: bool):
    racket = HC.RacketTargetCommand.__new__(HC.RacketTargetCommand)
    racket.num_envs = 2
    racket.device = "cpu"
    racket._action_ball_full_mdp_enabled = True
    racket._action_ball_enabled = False
    racket._action_ball_continuous_racket_poisoned = False
    racket._action_ball_full_mdp_device_r05_owner = object()
    racket._action_ball_continuous_racket_terminal_epoch_committed = committed
    racket._action_ball_full_mdp_racket_physical_owner = object()
    racket._action_ball_full_mdp_racket_selected_rubber_sequence = 0
    racket._action_ball_full_mdp_racket_selected_rubber_token = None
    racket._action_ball_full_mdp_racket_selected_rubber_view = None
    return racket


def test_capability_is_nonconstructible_and_unserializable():
    token_type = HC.ActionBallFullMdpRacketSelectedRubberToken
    with pytest.raises(TypeError, match="owner-issued"):
        token_type()
    token = object.__new__(token_type)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(token)
    with pytest.raises(TypeError, match="cannot be serialized"):
        token.__reduce__()


def test_production_publish_holds_before_any_mutation_without_real_accept():
    racket = _racket(committed=False)
    before = (
        racket._action_ball_full_mdp_racket_selected_rubber_sequence,
        racket._action_ball_full_mdp_racket_selected_rubber_token,
        racket._action_ball_full_mdp_racket_selected_rubber_view,
    )

    with pytest.raises(
        HC.ActionBallContinuousRacketSelectedRubberHold,
        match="real Device-R05 ACCEPT",
    ):
        racket.publish_action_ball_full_mdp_selected_rubber_authority()

    assert before == (
        racket._action_ball_full_mdp_racket_selected_rubber_sequence,
        racket._action_ball_full_mdp_racket_selected_rubber_token,
        racket._action_ball_full_mdp_racket_selected_rubber_view,
    )


def test_production_publication_accepts_no_caller_identity_or_boolean():
    method = HC.RacketTargetCommand.publish_action_ball_full_mdp_selected_rubber_authority
    assert tuple(inspect.signature(method).parameters) == ("self",)
    source = inspect.getsource(method)
    for forbidden in ("projection_sha256", "_assert_async"):
        assert forbidden not in source


def test_direct_epoch_surface_has_no_caller_authority_or_host_verdict():
    method = (
        HC.RacketTargetCommand.
        action_ball_full_mdp_action_epoch_selected_rubber_view
    )
    assert tuple(inspect.signature(method).parameters) == ("self",)
    source = inspect.getsource(method)
    for forbidden in (
        ".item(",
        ".tolist(",
        ".cpu(",
        "bool(",
        "_assert_async",
        "sha256",
        "receipt",
        "token",
        "scene_snapshot",
    ):
        assert forbidden not in source
    assert "epoch_owner.current()" not in source
    assert "epoch_owner.project_current_shot()" in source


def _published_view(racket):
    token_type = HC.ActionBallFullMdpRacketSelectedRubberToken
    token = object.__new__(token_type)
    active = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    key = torch.zeros((2, 2, 32), dtype=torch.uint8)
    key[0, 0, 0] = 1
    key[1, 1, 0] = 2
    generation = torch.tensor([[4, -1], [-1, 8]], dtype=torch.int64)
    view = HC.ActionBallFullMdpRacketSelectedRubberView(
        token=token,
        racket_owner=racket,
        publication_identity=object(),
        publication_sequence=1,
        physical_owner_mutation_version=9,
        active_mask=active,
        expected_rubber=torch.tensor([[0, -1], [-1, 1]], dtype=torch.int8),
        full_key_sha256=key,
        ball_generation=generation,
        flight_slot=torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
        reset_generation=torch.tensor([[1, 1], [2, 2]], dtype=torch.int64),
        swing_generation=torch.tensor([[4, 4], [8, 8]], dtype=torch.int64),
        action_uid=torch.tensor([[10, 10], [20, 20]], dtype=torch.int64),
        action_slot=torch.tensor([[0, 0], [1, 1]], dtype=torch.int64),
        task_receipt_sha256=key.clone(),
    )
    racket._action_ball_full_mdp_racket_selected_rubber_token = token
    racket._action_ball_full_mdp_racket_selected_rubber_view = view
    racket._action_ball_full_mdp_racket_physical_owner = SimpleNamespace(
        scene_snapshot=lambda: SimpleNamespace(
            owner_mutation_version=9,
            published_to_runtime=active.clone(),
            physically_parked=torch.zeros_like(active),
            outcome_key_sha256=key.clone(),
            ball_generation=generation.clone(),
        )
    )
    return token, view


def test_owned_view_is_clone_only_and_stale_physical_state_is_rejected():
    racket = _racket(committed=True)
    token, retained = _published_view(racket)

    projected = (
        racket.require_owned_action_ball_full_mdp_selected_rubber_authority(
            token
        )
    )
    projected.expected_rubber.fill_(1)
    projected.full_key_sha256.zero_()
    assert torch.equal(
        retained.expected_rubber,
        torch.tensor([[0, -1], [-1, 1]], dtype=torch.int8),
    )
    assert torch.count_nonzero(retained.full_key_sha256) == 2

    racket._action_ball_full_mdp_racket_physical_owner.scene_snapshot = (
        lambda: SimpleNamespace(
            owner_mutation_version=10,
            published_to_runtime=retained.active_mask.clone(),
            physically_parked=torch.zeros_like(retained.active_mask),
            outcome_key_sha256=retained.full_key_sha256.clone(),
            ball_generation=retained.ball_generation.clone(),
        )
    )
    with pytest.raises(
        HC.ActionBallContinuousRacketSelectedRubberHold,
        match="changed after publication",
    ):
        racket.require_owned_action_ball_full_mdp_selected_rubber_authority(
            token
        )


def test_cuda_production_publish_holds_before_snapshot_or_token_mutation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    racket = _racket(committed=True)
    racket.device = "cuda:0"
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        raise AssertionError("CUDA HOLD must precede Physical snapshot")

    racket._action_ball_full_mdp_racket_physical_owner = SimpleNamespace(
        scene_snapshot=snapshot
    )
    before = (
        racket._action_ball_full_mdp_racket_selected_rubber_sequence,
        racket._action_ball_full_mdp_racket_selected_rubber_token,
        racket._action_ball_full_mdp_racket_selected_rubber_view,
    )
    with pytest.raises(
        HC.ActionBallContinuousRacketSelectedRubberHold,
        match="packed global boundary",
    ):
        racket.publish_action_ball_full_mdp_selected_rubber_authority()
    assert calls == 0
    assert before == (
        racket._action_ball_full_mdp_racket_selected_rubber_sequence,
        racket._action_ball_full_mdp_racket_selected_rubber_token,
        racket._action_ball_full_mdp_racket_selected_rubber_view,
    )
