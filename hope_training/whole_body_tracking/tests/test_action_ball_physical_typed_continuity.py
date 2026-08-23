"""Cross-file regression for the direct Physical typed-identity continuity."""
from dataclasses import fields
import sys
from types import ModuleType, SimpleNamespace
import pytest
import torch
import test_action_ball_full_mdp_ball_scene_postphysics as scene_test
import test_action_ball_physical_epoch_hot_lane as hot
import test_action_ball_physical_flight_device as device_test
P = hot.physical
R06 = hot._canonical_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_landing_outcome_device",
    hot.MDP / "action_ball_landing_outcome_device.py",
)
class _Epoch(hot._EpochLaunchStub):
    def refresh_physical_postphysics_rows(self):
        self.owner.require_owned_action_epoch_r06_postphysics_projection()
class _Racket:
    def __init__(self, owner, epoch_owner):
        self.owner, self.epoch_owner = owner, epoch_owner
    def action_ball_full_mdp_action_epoch_selected_rubber_view(self):
        allocation = self.owner.require_owned_action_epoch_physics_fact_allocation(
            self.owner.action_epoch_physics_fact_allocation()
        )
        active = allocation.active_mask
        return SimpleNamespace(
            racket_owner=self, physical_owner=self.owner,
            epoch_owner=self.epoch_owner, active_mask=active.clone(),
            expected_rubber=torch.where(
                active, torch.zeros_like(active, dtype=torch.int8),
                torch.full_like(active, -1, dtype=torch.int8),
            ),
        )
class _R06(hot._R06InstallProbe):
    def __init__(self, owner):
        super().__init__(owner)
        self.postphysics = []
        self.retire = False
    def publish_action_ball_full_mdp_epoch_post_physics(self):
        view = self.owner.require_owned_action_epoch_r06_postphysics_projection()
        self.postphysics.append(SimpleNamespace(
            publication=view.publication_ordinal.clone(),
            observation=view.observation_ordinal.clone(),
            previous=view.previous_ball_center_m.clone(),
            current=view.current_ball_center_m.clone()))
        accepted = view.observe_mask.clone()
        settled = accepted & self.retire
        zeros = torch.zeros_like(view.observation_ordinal)
        return R06.ActionEpochR06PostPhysicsResult(
            accepted=accepted, rejected=torch.zeros_like(accepted), fault_bits=zeros,
            settled_mask=settled, settlement_cause=zeros,
            new_valid_contact_mask=view.selected_contact_event.clone(),
            observation_ordinal=view.observation_ordinal.clone(),
            mutation_version=zeros, flight_slot=view.flight_slot.clone(),
        )
    def retire_action_ball_full_mdp_epoch_post_physics(self):
        view = self.owner.require_owned_action_epoch_r06_postphysics_projection()
        retired = view.observe_mask & self.retire
        return R06.ActionEpochR06RetireResult(
            retired_mask=retired, mailbox_retired_mask=torch.zeros_like(retired),
            flight_slot=view.flight_slot.clone(),
            mutation_version=torch.zeros_like(view.observation_ordinal),
        )
    def publish_action_ball_full_mdp_epoch_facts(self): return None
    def poison_global_reveal_epoch(self, *_args): return None
@pytest.mark.parametrize("legacy_value", (0, 23))
def test_typed_launch_three_substeps_and_retire_preserve_exact_continuity(monkeypatch, legacy_value):
    scene_class = scene_test.S.IsaacLabPhysicalFlightScenePort
    scene_module_name = ("whole_body_tracking.tasks.tracking.config.agibot_a3."
                         "action_ball_full_mdp_ball_scene")
    monkeypatch.setattr(scene_class, "__module__", scene_module_name)
    scene_module = ModuleType(scene_module_name)
    scene_module.__file__ = scene_test.S.__file__
    scene_module.IsaacLabPhysicalFlightScenePort = scene_class
    monkeypatch.setitem(sys.modules, scene_module_name, scene_module)
    face_module = ModuleType("whole_body_tracking.tasks.tracking.mdp.hope_commands")
    face_module.ActionBallFullMdpActionEpochSelectedRubberView = SimpleNamespace
    monkeypatch.setitem(sys.modules, face_module.__name__, face_module)
    monkeypatch.setattr(
        sys.modules["whole_body_tracking.tasks.tracking.mdp"],
        "hope_commands", face_module, raising=False,
    )
    port = scene_test._port(torch.device("cpu"))
    capacity = hot._capacity(cadence=5, horizon=5)
    issue = hot.genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2, device=torch.device("cpu"))
    owner = P.ActionBallPhysicalFlightDeviceOwner(
        num_envs=2, capacity_receipt=capacity,
        expected_capacity_receipt_sha256=capacity.canonical_sha256,
        reset_genesis_authority=issue.authority, reset_genesis_receipt=issue.receipt,
        scene_body_names=port.spec.scene_entity_names, scene_port=port,
    )
    key_grid = hot._shot_key(num_envs=2, device=owner.device, width=1)
    assert len(fields(P._row_identity.ActionEpochShotKey)) == 8
    epoch_owner = _Epoch(owner, key_grid)
    key = P._row_identity.ActionEpochShotKey(**{
        field.name: getattr(key_grid, field.name)[:, 0].clone()
        for field in fields(P._row_identity.ActionEpochShotKey)
    })
    motion = hot._MotionLaunchStub(owner, key)
    r06 = _R06(owner)
    owner._action_epoch_owner, owner._action_epoch_motion_owner = epoch_owner, motion
    owner._r06_owner = r06
    port._action_epoch_physical_owner = owner
    port._action_epoch_owner = epoch_owner
    port._action_epoch_racket_owner = _Racket(owner, epoch_owner)
    fact_owner, *_ = scene_test._fact_owner(device=torch.device("cpu"))
    scene_test._mark_live_subscriptions(fact_owner)
    port._physx_fact_owner = fact_owner
    owner._action_epoch_pending_launch = hot._pending(
        owner, key=key, pending=torch.tensor((True, False)),
        ordinal=torch.tensor((8, 9), dtype=torch.int64),
    )
    device_test._install_focused_postphysics_stamp_module(monkeypatch)
    owner._outcome_sha.fill_(legacy_value)
    legacy_sha = owner._outcome_sha.clone()
    legacy_observation = owner._observation_ordinal.clone()
    peer = owner._clone_action_epoch_direct_state()
    for substep, delta in enumerate((0.01, 0.02, 0.03)):
        owner.launch_action_epoch()
        port.assets[0].data.root_state_w[0, 0] += delta
        fact_owner.on_post_step_heartbeat(0.005)
        owner.publish_action_epoch_post_physics(device_test._focused_postphysics_stamp(
            control=1, substep=substep, decimation=4, sim_step=substep + 1,
        ))
    assert [view.observation[0, 0].item() for view in r06.postphysics] == [0, 1, 2]
    assert [view.publication[0, 0].item() for view in r06.postphysics] == [8, 8, 8]
    assert torch.equal(r06.postphysics[1].previous[0, 0], r06.postphysics[0].current[0, 0])
    assert torch.equal(r06.postphysics[2].previous[0, 0], r06.postphysics[1].current[0, 0])
    r06.retire = True
    owner.launch_action_epoch()
    fact_owner.on_post_step_heartbeat(0.005)
    owner.publish_action_epoch_post_physics(device_test._focused_postphysics_stamp(
        control=1, substep=3, decimation=4, sim_step=4,
    ))
    for field in fields(P._row_identity.ActionEpochShotKey):
        lane = getattr(owner._action_epoch_flight_shot_key, field.name)
        assert lane[0].eq(-1).all()
        assert torch.equal(lane[1], getattr(peer.flight_shot_key, field.name)[1])
    assert owner._action_epoch_flight_publication_ordinal[0].eq(-1).all()
    assert owner._action_epoch_flight_observation_ordinal[0].eq(-1).all()
    assert owner._action_epoch_flight_previous_ball_center_m[0].eq(0).all()
    for lane, before in (
        (owner._action_epoch_flight_publication_ordinal, peer.flight_publication_ordinal),
        (owner._action_epoch_flight_observation_ordinal, peer.flight_observation_ordinal),
        (owner._action_epoch_flight_previous_ball_center_m, peer.flight_previous_ball_center_m),
    ):
        assert torch.equal(lane[1], before[1])
    assert torch.equal(owner._outcome_sha, legacy_sha)
    assert torch.equal(owner._observation_ordinal, legacy_observation)
