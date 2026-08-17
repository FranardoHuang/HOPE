"""Focused exact-source tests for the lean R03 -> ActionEpoch Reward seam."""

from __future__ import annotations

from dataclasses import replace
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

import action_ball_strike_fact_device as D  # noqa: E402


def _epoch_module():
    return importlib.import_module(D._exact_action_epoch_owner_type().__module__)


def _devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available():
        result.append(torch.device("cuda", torch.cuda.current_device()))
    return result


@pytest.fixture(params=_devices())
def device(request):
    return request.param


def _epoch(*, num_envs: int, device: torch.device):
    module = _epoch_module()
    owner = module.ActionEpochOwner(num_envs=num_envs, device=device)
    owner.activate_reset_genesis(
        selected_mask=torch.ones(num_envs, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(num_envs, dtype=torch.int64, device=device),
    )
    return owner


def test_exact_epoch_binding_rejects_foreign_type_and_batch(device):
    with pytest.raises(D.StrikeFactDeviceError, match="exact ActionEpochOwner"):
        D.ActionBallStrikeFactDeviceCoordinator(
            num_envs=3, device=device, action_epoch_owner=object()
        )
    wrong_batch = _epoch(num_envs=2, device=device)
    with pytest.raises(D.StrikeFactDeviceError, match="batch/device"):
        D.ActionBallStrikeFactDeviceCoordinator(
            num_envs=3, device=device, action_epoch_owner=wrong_batch
        )


def test_one_time_cold_epoch_bind_uses_the_canonical_module_identity(device):
    owner = D.ActionBallStrikeFactDeviceCoordinator(num_envs=3, device=device)
    epoch = _epoch(num_envs=3, device=device)
    owner.bind_action_epoch_owner(epoch)
    assert owner.action_epoch_owner is epoch
    with pytest.raises(D.StrikeFactDeviceError, match="already bound"):
        owner.bind_action_epoch_owner(epoch)


def test_reward_snapshot_uses_version_only_for_freshness(device):
    epoch = _epoch(num_envs=3, device=device)
    owner = D.ActionBallStrikeFactDeviceCoordinator(
        num_envs=3, device=device, action_epoch_owner=epoch
    )
    current = epoch.current()
    facts = owner.action_epoch_reward_facts_v1(current)
    assert not facts.eligible.any()
    assert not facts.validity.any()
    assert not facts.producer_fault_bits.any()

    stale = replace(current, version=current.version - 1)
    with pytest.raises(D.StrikeFactDeviceError, match="foreign, stale"):
        owner.action_epoch_reward_facts_v1(stale)
