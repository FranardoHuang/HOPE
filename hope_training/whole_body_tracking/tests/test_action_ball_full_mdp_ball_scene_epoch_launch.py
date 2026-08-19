"""Focused counterexamples for the lean Physical -> ActionEpoch scene writer."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from test_action_ball_full_mdp_ball_scene import S, _Asset, _ReplicatedScene, _spec


_ROOT = Path(__file__).resolve().parents[1]
_MDP = (
    _ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("action_ball_full_mdp_epoch", _MDP / "action_ball_full_mdp_epoch.py")


class _Projection:
    def __init__(self, *, kind, state, selected, owner):
        self.kind = kind
        self.state_env_f32 = state
        self.selected_mask = selected
        self.physical_owner = owner


class _Physical:
    def action_epoch_scene_write_projection(self):
        if self._action_epoch_active_scene_write is None:
            raise RuntimeError("no active scene write")
        return self._action_epoch_active_scene_write

    def require_owned_action_epoch_scene_write_projection(self, view):
        if view is not self._action_epoch_active_scene_write:
            raise RuntimeError("foreign scene write")
        return view


def _port(device="cpu"):
    device = torch.device(device)
    _, spec = _spec(cadence=5, horizon=5)
    origins = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]],
        dtype=torch.float32,
        device=device,
    )
    scene = _ReplicatedScene(num_envs=2)
    for name in spec.scene_entity_names:
        root = torch.zeros((2, 13), dtype=torch.float32, device=device)
        root[:, :3] = origins + torch.tensor(
            S.PARK_POSITION_ENV_M, dtype=torch.float32, device=device
        )
        root[:, 3] = 1.0
        scene[name] = _Asset(root)
    return S.IsaacLabPhysicalFlightScenePort(
        scene=scene, spec=spec, env_origins=origins
    ), scene, origins


def _physical_shell(port, epoch_owner):
    owner = _Physical()
    owner.num_envs = port.num_envs
    owner.flight_capacity = port.flight_capacity
    owner.device = port.device
    owner.scene_port = port
    owner._action_epoch_owner = epoch_owner
    owner._action_epoch_active_scene_write = None
    return owner


def _view(owner, *, kind="launch", selected_cells=((1, 0),), mutate_cells=None):
    state = owner.scene_port.read_state_env().contiguous()
    selected = torch.zeros(
        (owner.num_envs, owner.flight_capacity),
        dtype=torch.bool,
        device=owner.device,
    )
    for env_index, slot_index in selected_cells:
        selected[env_index, slot_index] = True
        state[env_index, slot_index, :3] = torch.tensor(
            [0.3, -0.2, 1.1], dtype=torch.float32, device=owner.device
        )
        state[env_index, slot_index, 3] = 1.0
        state[env_index, slot_index, 7:10] = torch.tensor(
            [4.0, 0.0, -2.0], dtype=torch.float32, device=owner.device
        )
    for env_index, slot_index in mutate_cells or ():
        state[env_index, slot_index, 0] += 7.0
    view = _Projection(
        kind=kind, state=state, selected=selected.contiguous(), owner=owner
    )
    owner._action_epoch_active_scene_write = view
    return view


@pytest.fixture(autouse=True)
def _direct_module_bindings(monkeypatch):
    # The production package import needs Isaac Lab on the Pod.  Focused host
    # tests bind the exact already-loaded source classes under their direct
    # fallback names; the scene still verifies exact class and bound-method
    # identity rather than accepting a duck-typed fixture.
    monkeypatch.setitem(sys.modules, "action_ball_full_mdp_epoch", E)
    module = ModuleType("action_ball_physical_flight_device")
    module.ActionBallPhysicalFlightDeviceOwner = _Physical
    monkeypatch.setitem(sys.modules, "action_ball_physical_flight_device", module)


def test_exact_bound_physical_projection_is_the_only_scene_launch_payload():
    port, scene, origins = _port()
    epoch_owner = E.ActionEpochOwner(num_envs=2, device="cpu")
    physical = _physical_shell(port, epoch_owner)

    signature = inspect.signature(port.preflight_action_epoch_write)
    assert tuple(signature.parameters) == ()
    source = inspect.getsource(S.IsaacLabPhysicalFlightScenePort.preflight_action_epoch_write)
    assert "receipt" not in source
    assert "sha256" not in source.lower()
    assert "verdict" not in source
    assert "contact" not in source

    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="not construction-bound"):
        port.preflight_action_epoch_write()
    port.bind_action_epoch_scene_writer(physical, epoch_owner)
    _view(physical)
    handle = port.preflight_action_epoch_write()
    receipt = port.apply_prevalidated_write(handle)
    port.require_owned_apply_receipt(handle, receipt)
    assert torch.equal(
        scene[port.spec.scene_entity_names[0]].data.root_state_w[1, :3],
        origins[1] + torch.tensor([0.3, -0.2, 1.1]),
    )
    assert torch.equal(
        scene[port.spec.scene_entity_names[1]].data.root_state_w,
        torch.tensor(
            [
                [1.0, 2.0, -17.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-2.0, 0.5, -16.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_fact_source_binder_normalizes_manager_term_device_without_host_tensor_verdict():
    source = inspect.getsource(
        S.IsaacLabPhysicalFlightScenePort.bind_action_epoch_physics_fact_source
    )
    assert "type(racket_device_raw) not in (str, torch.device)" in source
    assert "racket_device = torch.device(racket_device_raw)" in source
    assert "racket_device != self.device" in source
    assert "torch.equal" not in source
    assert ".any()" not in source


def test_foreign_writer_rebind_fails_and_mask_prevents_inactive_slot_mutation():
    port, scene, _ = _port()
    epoch_owner = E.ActionEpochOwner(num_envs=2, device="cpu")
    physical = _physical_shell(port, epoch_owner)
    foreign = _physical_shell(port, epoch_owner)
    port.bind_action_epoch_scene_writer(physical, epoch_owner)

    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="identity"):
        port.bind_action_epoch_scene_writer(foreign, epoch_owner)
    _view(physical, selected_cells=((1, 0),), mutate_cells=((0, 1),))
    before = scene[port.spec.scene_entity_names[1]].data.root_state_w.clone()
    handle = port.preflight_action_epoch_write()
    port.apply_prevalidated_write(handle)
    assert torch.equal(
        scene[port.spec.scene_entity_names[1]].data.root_state_w, before
    )


def test_launch_slot_count_nonfinite_and_missing_active_writer_fail_closed():
    port, scene, _ = _port()
    epoch_owner = E.ActionEpochOwner(num_envs=2, device="cpu")
    physical = _physical_shell(port, epoch_owner)
    port.bind_action_epoch_scene_writer(physical, epoch_owner)

    _view(physical, selected_cells=((0, 0), (0, 1)))
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="multiple slots"):
        port.preflight_action_epoch_write()
    view = _view(physical, selected_cells=((0, 0),))
    view.state_env_f32[0, 0, 0] = float("nan")
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="nonfinite"):
        port.preflight_action_epoch_write()
    physical._action_epoch_active_scene_write = None
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="no exact active"):
        port.preflight_action_epoch_write()
    assert all(asset.pose_writes == 0 for asset in scene.values())


def test_retire_can_park_multiple_physical_owned_slots():
    port, scene, _ = _port()
    epoch_owner = E.ActionEpochOwner(num_envs=2, device="cpu")
    physical = _physical_shell(port, epoch_owner)
    port.bind_action_epoch_scene_writer(physical, epoch_owner)

    _view(physical, kind="retire", selected_cells=((0, 0), (0, 1)))
    handle = port.preflight_action_epoch_write()
    port.apply_prevalidated_write(handle)
    assert scene[port.spec.scene_entity_names[0]].pose_writes == 1
    assert scene[port.spec.scene_entity_names[1]].pose_writes == 1



@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["launch", "retire"])
def test_repeated_empty_mask_is_a_branchless_noop_not_safety_evidence(device, kind):
    port, scene, _ = _port(device)
    epoch_owner = E.ActionEpochOwner(num_envs=2, device=device)
    physical = _physical_shell(port, epoch_owner)
    port.bind_action_epoch_scene_writer(physical, epoch_owner)
    before = {
        name: asset.data.root_state_w.clone() for name, asset in scene.items()
    }

    for _ in range(3):
        view = _view(
            physical,
            kind=kind,
            selected_cells=(),
            mutate_cells=((0, 0), (0, 1), (1, 0), (1, 1)),
        )
        assert not bool(view.selected_mask.any())
        handle = port.preflight_action_epoch_write()
        receipt = port.apply_prevalidated_write(handle)
        port.require_owned_apply_receipt(handle, receipt)
        physical._action_epoch_active_scene_write = None

    for name, asset in scene.items():
        assert torch.equal(asset.data.root_state_w, before[name])
        assert asset.pose_writes == 3
        assert asset.velocity_writes == 3
