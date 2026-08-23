"""Focused value/lifetime contract for the narrow current-shot projection."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import inspect
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (SOURCE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import action_ball_full_mdp_epoch as epoch  # noqa: E402


def _owner_with_mixed_current_slots(device: torch.device):
    owner = epoch.ActionEpochOwner(
        num_envs=3,
        device=device,
        shot_slot_capacity=2,
        initial_reset_generation=torch.ones(
            3, dtype=torch.int64, device=device
        ),
    )
    owner.activate_reset_genesis(
        selected_mask=torch.ones(3, dtype=torch.bool, device=device),
        reset_generation=torch.ones(3, dtype=torch.int64, device=device),
    )
    record = owner._publication.current
    assert record is not None
    shape = (3, 2)
    shot_key = epoch.ActionEpochShotKey(
        **{
            field.name: (
                torch.arange(6, dtype=torch.int64, device=device)
                .reshape(shape)
                .add_(1000 * (index + 1))
                .contiguous()
            )
            for index, field in enumerate(fields(epoch.ActionEpochShotKey))
        }
    )
    current = replace(
        record,
        identity=replace(record.identity, shot_key=shot_key),
        phase=torch.tensor(
            [[10, 11], [20, 21], [30, 31]],
            dtype=torch.int64,
            device=device,
        ),
        current_task_slot=torch.tensor(
            [1, 0, -1], dtype=torch.int64, device=device
        ),
        publication_ordinal=torch.tensor(
            [[100, 101], [200, 201], [300, 301]],
            dtype=torch.int64,
            device=device,
        ),
    )
    owner._publication = epoch._Publication(
        current=current,
        pending_log=owner._publication.pending_log,
    )
    return owner


@pytest.mark.parametrize("device_name", ("cpu", "cuda:0"))
def test_current_shot_projection_gathers_neutralizes_and_never_clones_record(
    monkeypatch, device_name
):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    owner = _owner_with_mixed_current_slots(device)

    def forbidden_record_clone(_record):
        raise AssertionError("narrow projection must not clone ActionEpochRecord")

    monkeypatch.setattr(epoch.ActionEpochRecord, "clone", forbidden_record_clone)
    projection = owner.project_current_shot()

    assert type(projection) is epoch.ActionEpochCurrentShotProjection
    assert tuple(field.name for field in fields(type(projection))) == (
        "slot_valid",
        "phase",
        "shot_key",
        "publication_ordinal",
    )
    assert projection.slot_valid.tolist() == [True, True, False]
    assert projection.phase.tolist() == [11, 20, -1]
    assert projection.publication_ordinal.tolist() == [101, 200, -1]
    epoch.row_identity.require_action_epoch_shot_key(
        projection.shot_key,
        shape=(3,),
        device=device,
        label="current-shot projection",
    )
    for index, field in enumerate(fields(epoch.ActionEpochShotKey)):
        expected = [1001 + 1000 * index, 1002 + 1000 * index, -1]
        assert getattr(projection.shot_key, field.name).tolist() == expected


def test_current_shot_projection_has_no_live_alias_and_becomes_stale_by_value():
    owner = _owner_with_mixed_current_slots(torch.device("cpu"))
    live = owner._publication.current
    assert live is not None
    live_phase = live.phase.clone()
    live_publication = live.publication_ordinal.clone()
    live_key = live.identity.shot_key.clone()
    first = owner.project_current_shot()
    frozen_first_phase = first.phase.clone()

    first.slot_valid.zero_()
    first.phase.fill_(777)
    first.publication_ordinal.fill_(888)
    for field in fields(epoch.ActionEpochShotKey):
        getattr(first.shot_key, field.name).fill_(999)

    assert torch.equal(live.phase, live_phase)
    assert torch.equal(live.publication_ordinal, live_publication)
    for field in fields(epoch.ActionEpochShotKey):
        assert torch.equal(
            getattr(live.identity.shot_key, field.name),
            getattr(live_key, field.name),
        )

    second = owner.project_current_shot()
    assert second.phase.tolist() == [11, 20, -1]
    assert second.phase.data_ptr() != first.phase.data_ptr()
    for field in fields(epoch.ActionEpochShotKey):
        assert (
            getattr(second.shot_key, field.name).data_ptr()
            != getattr(first.shot_key, field.name).data_ptr()
        )

    live.phase[0, 1] = 42
    third = owner.project_current_shot()
    assert frozen_first_phase.tolist() == [11, 20, -1]
    assert second.phase.tolist() == [11, 20, -1]
    assert third.phase.tolist() == [42, 20, -1]


def test_current_shot_projection_source_has_no_host_verdict_or_caller_choice():
    method = epoch.ActionEpochOwner.project_current_shot
    assert tuple(inspect.signature(method).parameters) == ("self",)
    source = inspect.getsource(method)
    for forbidden in (
        ".item(",
        ".cpu(",
        ".tolist(",
        ".numpy(",
        ".nonzero(",
        "synchronize(",
        "_assert_async",
        "bool(",
        ".clone()",
    ):
        assert forbidden not in source
