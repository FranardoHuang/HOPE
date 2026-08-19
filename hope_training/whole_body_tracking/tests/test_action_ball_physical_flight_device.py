"""Focused Physical owner negative boundaries independent of retired R05 ingress.

The portable/compact R05 preview construction path was removed from production.
Its helper graph and downstream integration tests belong to that deleted ABI and
must not be reconstructed here.  Current full-N ActionEpoch launch,
postphysics, selected-reset, and checkpoint-HOLD coverage lives in
``test_action_ball_physical_epoch_hot_lane.py`` and is exercised with this file
in the same-process integration command.
"""

from __future__ import annotations

from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
import types
from typing import NamedTuple

import pytest
import torch


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import action_ball_full_mdp_diagnostic_capacity as Q  # noqa: E402
import action_ball_full_mdp_reset_genesis as G  # noqa: E402
from test_action_ball_physical_flight_contract import (  # noqa: E402
    _prepare,
    _sha,
)


def _load_device_module():
    path = (
        _SOURCE_ROOT
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "mdp"
        / "action_ball_physical_flight_device.py"
    )
    name = "_test_action_ball_physical_flight_device"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load_device_module()


class _FocusedPostPhysicsEventPhase(IntEnum):
    POST_SCENE_UPDATE = 1


class _FocusedPostPhysicsStamp(NamedTuple):
    control_step: int
    physics_substep: int
    physics_substeps_per_control: int
    sim_step: int
    event_phase: _FocusedPostPhysicsEventPhase

    def exact_tuple(self):
        return (
            self.control_step,
            self.physics_substep,
            self.physics_substeps_per_control,
            self.sim_step,
            int(self.event_phase),
        )


_POSTPHYSICS_ENV_MODULE = "whole_body_tracking.tasks.tracking.full_mdp_env"
_FocusedPostPhysicsStamp.__module__ = _POSTPHYSICS_ENV_MODULE
_FocusedPostPhysicsStamp.__qualname__ = "FullMdpPhysicsSubstepStamp"
_FocusedPostPhysicsEventPhase.__module__ = _POSTPHYSICS_ENV_MODULE
_FocusedPostPhysicsEventPhase.__qualname__ = "FullMdpPhysicsEventPhase"


def _install_focused_postphysics_stamp_module(monkeypatch):
    module = types.ModuleType(_POSTPHYSICS_ENV_MODULE)
    module.FullMdpPhysicsSubstepStamp = _FocusedPostPhysicsStamp
    module.FullMdpPhysicsEventPhase = _FocusedPostPhysicsEventPhase
    monkeypatch.setitem(sys.modules, _POSTPHYSICS_ENV_MODULE, module)


def _focused_postphysics_stamp(*, control=1, substep=0, decimation=1, sim_step=1):
    return _FocusedPostPhysicsStamp(
        control_step=control,
        physics_substep=substep,
        physics_substeps_per_control=decimation,
        sim_step=sim_step,
        event_phase=_FocusedPostPhysicsEventPhase.POST_SCENE_UPDATE,
    )


def _owner(*, num_envs: int = 1, device: torch.device | str = "cpu"):
    template = _prepare()
    install = template.rows[0].install_payload
    capacity = install.capacity_receipt
    scene = D.TensorPhysicalFlightScenePort(
        num_envs=num_envs,
        flight_capacity=capacity.configured_flight_capacity,
        device=device,
    )
    issue = G.issue_action_ball_full_mdp_reset_genesis(
        num_envs=num_envs,
        device=torch.device(device),
    )
    owner = D.ActionBallPhysicalFlightDeviceOwner(
        num_envs=num_envs,
        capacity_receipt=capacity,
        expected_capacity_receipt_sha256=capacity.canonical_sha256,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
        scene_body_names=tuple(
            f"action_ball_flight_ball_{index:03d}"
            for index in range(capacity.configured_flight_capacity)
        ),
        scene_port=scene,
    )
    return owner, scene, template, install


def _stamp(shape, *, control: int, substep: int, phase: int):
    return D.PhysicsStampGrid(
        control_step=torch.full(shape, control, dtype=torch.int64),
        physics_substep=torch.full(shape, substep, dtype=torch.int32),
        event_phase=torch.full(shape, phase, dtype=torch.int8),
    )


def _facts(shape, *, control: int):
    zeros = torch.zeros(shape, dtype=torch.bool)
    return D.IsaacPostPhysicsFacts(
        observation_stamp=_stamp(shape, control=control, substep=0, phase=2),
        current_state_env_f32=torch.zeros(shape + (13,), dtype=torch.float32),
        selected_contact_event=zeros.clone(),
        selected_contact_ball_center_m=torch.zeros(
            shape + (3,), dtype=torch.float32
        ),
        selected_contact_outgoing_segment_anchor_m=torch.zeros(
            shape + (3,), dtype=torch.float32
        ),
        selected_contact_stamp=_stamp(
            shape, control=-1, substep=-1, phase=-1
        ),
        net_crossing_event=zeros.clone(),
        net_clear_at_crossing=zeros.clone(),
        net_crossing_stamp=_stamp(shape, control=-1, substep=-1, phase=-1),
        crossing_report_delivered=zeros.clone(),
        first_descending_crossing_event=zeros.clone(),
        first_descending_crossing_xy_m=torch.zeros(
            shape + (2,), dtype=torch.float32
        ),
        first_descending_crossing_stamp=_stamp(
            shape, control=-1, substep=-1, phase=-1
        ),
        nonfinite_observation=zeros.clone(),
        producer_contract_fault=zeros.clone(),
        engine_overflow=zeros.clone(),
    )


def test_legacy_hot_binder_remains_explicit_hold_without_mutation():
    owner, _, _, _ = _owner()
    before = owner.scene_snapshot()
    with pytest.raises(D.PhysicalLateLaunchProductionHold, match="remains HOLD"):
        owner.bind_device_r05_hot_reveal_owner(
            object(),
            physical_question_owner=object(),
            motion_contact_launch_authority=object(),
            full_key_authority=object(),
        )
    after = owner.scene_snapshot()
    assert torch.equal(before.state_env_f32, after.state_env_f32)
    assert before.owner_mutation_version == after.owner_mutation_version


def test_capture_postphysics_wrong_stamp_and_missing_exact_scene_producer_are_sticky(
    monkeypatch,
):
    _install_focused_postphysics_stamp_module(monkeypatch)
    owner, _, _, _ = _owner()

    with pytest.raises(D.PhysicalFlightOwnerPoisonedError, match="wrong stamp"):
        owner.capture_post_physics_facts(object())
    with pytest.raises(D.PhysicalFlightOwnerPoisonedError, match="wrong stamp"):
        owner.capture_post_physics_facts(_focused_postphysics_stamp())

    owner, _, _, _ = _owner()
    with pytest.raises(
        D.PhysicalFlightOwnerPoisonedError,
        match="exact concrete Isaac postphysics producer is absent",
    ):
        owner.capture_post_physics_facts(_focused_postphysics_stamp())
    assert owner._poisoned
    assert owner._active_postphysics_capture is None
    assert owner._active_postphysics is None


def test_capture_postphysics_exact_stamp_chronology_rejects_first_stale_substep(
    monkeypatch,
):
    _install_focused_postphysics_stamp_module(monkeypatch)
    owner, _, _, _ = _owner()
    with pytest.raises(
        D.PhysicalFlightOwnerPoisonedError,
        match="did not start at substep zero",
    ):
        owner.capture_post_physics_facts(
            _focused_postphysics_stamp(
                control=1,
                substep=1,
                decimation=2,
                sim_step=2,
            )
        )


def test_production_direct_postphysics_build_is_not_fact_authority(monkeypatch):
    owner, _, _, _ = _owner()
    monkeypatch.setattr(owner, "scene_port", object())
    with pytest.raises(
        D.PhysicalFlightOwnerPoisonedError,
        match="exact one-shot capture",
    ):
        owner.build_post_physics_publication(facts=_facts((1, 1), control=1))
    assert owner._poisoned
    assert owner._active_postphysics is None


def test_cuda_capture_postphysics_missing_exact_producer_fails_before_host_sync(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("focused CUDA postphysics gate requires CUDA")
    _install_focused_postphysics_stamp_module(monkeypatch)
    indexed_cuda = torch.device("cuda", torch.cuda.current_device())
    owner, _, _, _ = _owner(device=indexed_cuda)
    original_item = torch.Tensor.item

    def forbidden_item(value):
        if value.is_cuda:
            raise AssertionError("capture synchronized CUDA to host")
        return original_item(value)

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    with pytest.raises(
        D.PhysicalFlightOwnerPoisonedError,
        match="exact concrete Isaac postphysics producer is absent",
    ):
        owner.capture_post_physics_facts(_focused_postphysics_stamp())


def test_constructor_rejects_duck_typed_or_subclassed_scene_port():
    class AliasingPort(D.TensorPhysicalFlightScenePort):
        def read_state_env(self):
            return self._state

    template = _prepare()
    capacity = template.rows[0].install_payload.capacity_receipt
    port = AliasingPort(
        num_envs=1,
        flight_capacity=capacity.configured_flight_capacity,
        device="cpu",
    )
    issue = G.issue_action_ball_full_mdp_reset_genesis(
        num_envs=1,
        device=torch.device("cpu"),
    )
    with pytest.raises(
        D.PhysicalFlightDeviceError,
        match="exact reviewed concrete implementation",
    ):
        D.ActionBallPhysicalFlightDeviceOwner(
            num_envs=1,
            capacity_receipt=capacity,
            expected_capacity_receipt_sha256=capacity.canonical_sha256,
            reset_genesis_authority=issue.authority,
            reset_genesis_receipt=issue.receipt,
            scene_body_names=tuple(
                f"action_ball_flight_ball_{index:03d}"
                for index in range(capacity.configured_flight_capacity)
            ),
            scene_port=port,
        )


def test_isaac_scene_exact_type_accepts_nonsemantic_source_comment_without_sha_gate(
    monkeypatch,
    tmp_path,
):
    """Exact live class identity, not a source hash, binds the scene owner."""

    source_path = (
        _SOURCE_ROOT
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3"
        / "action_ball_full_mdp_ball_scene.py"
    )
    mutated_path = tmp_path / source_path.name
    mutated_path.write_bytes(
        source_path.read_bytes() + b"\n# independent source-pin mutation\n"
    )
    module_name = (
        "whole_body_tracking.tasks.tracking.config.agibot_a3."
        "action_ball_full_mdp_ball_scene"
    )
    spec = importlib.util.spec_from_file_location(module_name, mutated_path)
    assert spec is not None and spec.loader is not None
    scene_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, scene_module)
    spec.loader.exec_module(scene_module)

    scene_spec = scene_module.ActionBallFullMdpDiagnosticBallSceneSpec(
        schema_version=scene_module.SCHEMA_VERSION,
        kind=scene_module.DIAGNOSTIC_SCENE_SPEC_KIND,
        capacity_authority_kind=Q.DIAGNOSTIC_CAPACITY_KIND,
        formal_capacity_receipt_sha256=None,
        flight_capacity=2,
        scene_entity_names=(
            "action_ball_flight_ball_000",
            "action_ball_flight_ball_001",
        ),
        prim_paths=(
            "{ENV_REGEX_NS}/ActionBallFlightBall_000",
            "{ENV_REGEX_NS}/ActionBallFlightBall_001",
        ),
        ball_radius_m=0.02,
        ball_mass_kg=0.0034,
        park_position_env_m=scene_module.PARK_POSITION_ENV_M,
        collision_enabled=True,
        gravity_enabled=True,
    )

    class Data:
        def __init__(self):
            self.root_state_w = torch.zeros((2, 13), dtype=torch.float32)
            self.root_state_w[:, 2] = scene_module.PARK_POSITION_ENV_M[2]
            self.root_state_w[:, 3] = 1.0

    class Asset:
        def __init__(self):
            self.data = Data()

        def write_root_pose_to_sim(self, value, *, env_ids):
            self.data.root_state_w[env_ids, :7] = value

        def write_root_velocity_to_sim(self, value, *, env_ids):
            self.data.root_state_w[env_ids, 7:] = value

    class Scene(dict):
        pass

    scene = Scene({name: Asset() for name in scene_spec.scene_entity_names})
    scene.cfg = types.SimpleNamespace(replicate_physics=True)
    scene.env_prim_paths = tuple(f"/World/envs/env_{index}" for index in range(2))
    port = scene_module.IsaacLabPhysicalFlightScenePort(
        scene=scene,
        spec=scene_spec,
        env_origins=torch.zeros((2, 3), dtype=torch.float32),
    )
    binding = Q.construct_diagnostic_n2_capacity_binding(scene_spec)
    issue = G.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2,
        device=torch.device("cpu"),
    )

    owner = D.ActionBallPhysicalFlightDeviceOwner(
        num_envs=2,
        scene_body_names=scene_spec.scene_entity_names,
        scene_port=port,
        diagnostic_n2_capacity_binding=binding,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
    )
    assert owner.scene_port is port
    assert owner._diagnostic_n2_no_save is True


def test_settle_retire_rejects_caller_constructed_r06_ack_before_scene_write():
    owner, scene, _, install = _owner()
    before_apply_count = scene.apply_count
    mask = torch.zeros((1, owner.flight_capacity), dtype=torch.bool)
    mask[0, install.flight_slot] = True
    fake_ack = D.AcknowledgedR06PhysicalSnapshot(
        _snapshot_root_sha256=_sha("fake-r06-after"),
        _owner_mutation_version=2,
        _owner_identity=owner._owner_identity,
        _token=object(),
    )
    with pytest.raises(D.PhysicalFlightDeviceError, match="tombstoned"):
        owner.settle_retire(
            D.PhysicalSettleRetireInput(
                retire_mask=mask,
                r06_ack=fake_ack,
            )
        )
    assert scene.apply_count == before_apply_count
