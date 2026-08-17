from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.util
import inspect
import copy
import json
import pickle
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_SOURCE_ROOT = _HERE.parent / "source" / "whole_body_tracking"
_MDP_ROOT = (
    _SOURCE_ROOT
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
for _path in (_SOURCE_ROOT, _MDP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _install_namespace(name: str, path: Path) -> None:
    """Install package identity without executing unrelated Isaac imports."""

    module = sys.modules.get(name)
    if module is not None:
        return
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


_install_namespace("whole_body_tracking", _SOURCE_ROOT / "whole_body_tracking")
_install_namespace(
    "whole_body_tracking.tasks",
    _SOURCE_ROOT / "whole_body_tracking" / "tasks",
)
_install_namespace(
    "whole_body_tracking.tasks.tracking",
    _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking",
)
_install_namespace(
    "whole_body_tracking.tasks.tracking.mdp",
    _MDP_ROOT,
)
_CANONICAL_R06 = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp."
    "action_ball_landing_outcome_device"
)


class _ExistingR06Loader:
    """Let the legacy helper reuse, not reload, the canonical package module."""

    @staticmethod
    def create_module(_spec):
        return _CANONICAL_R06

    @staticmethod
    def exec_module(module):
        assert module is _CANONICAL_R06


_HELPER_PATH = _HERE / "test_action_ball_landing_outcome_device.py"
_SPEC = importlib.util.spec_from_file_location(
    "_landing_outcome_selected_reset_helper", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
H = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = H
_original_spec_from_file_location = importlib.util.spec_from_file_location


def _helper_spec_from_file_location(name, location, *args, **kwargs):
    if (
        name == "action_ball_landing_outcome_device_under_test"
        and Path(location).resolve()
        == (_MDP_ROOT / "action_ball_landing_outcome_device.py").resolve()
    ):
        return importlib.util.spec_from_loader(
            _CANONICAL_R06.__name__, _ExistingR06Loader()
        )
    return _original_spec_from_file_location(
        name, location, *args, **kwargs
    )


try:
    importlib.util.spec_from_file_location = _helper_spec_from_file_location
    _SPEC.loader.exec_module(H)
finally:
    importlib.util.spec_from_file_location = _original_spec_from_file_location
D = _CANONICAL_R06
H.D = D
DEVICE_R05 = D._r05_device

# The exact local CPU torch environment is Python 3.8.  Load the checked-in
# global-drain/device fixture with only its Python >=3.9 type-alias syntax
# postponed; production modules and runtime behavior are unchanged.
if sys.version_info < (3, 9):
    _drain_path = _MDP_ROOT / "action_ball_full_mdp_ppo_drain.py"
    _drain_source = _drain_path.read_text(encoding="utf-8").replace(
        "CheckpointFieldIdentity = (\n"
        "    tuple[str, str, int] | tuple[str, str, int, int]\n"
        ")\n"
        "CheckpointSchemaIdentity = tuple[\n"
        "    tuple[str, tuple[CheckpointFieldIdentity, ...]], ...\n"
        "]",
        "CheckpointFieldIdentity = object\nCheckpointSchemaIdentity = object",
    )
    _drain_name = (
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_ppo_drain"
    )
    _drain_module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(_drain_name, loader=None)
    )
    _drain_module.__file__ = str(_drain_path)
    sys.modules[_drain_name] = _drain_module
    exec(compile(_drain_source, str(_drain_path), "exec"), _drain_module.__dict__)

    _device_helper_path = (
        _HERE / "test_action_ball_continuous_runtime_transaction_device.py"
    )
    _device_helper_source = _device_helper_path.read_text(encoding="utf-8").replace(
        "_DRAIN_SPEC = importlib.util.spec_from_file_location(_DRAIN_NAME, _GLOBAL_DRAIN_PATH)\n"
        "assert _DRAIN_SPEC is not None and _DRAIN_SPEC.loader is not None\n"
        "global_drain = importlib.util.module_from_spec(_DRAIN_SPEC)\n"
        "sys.modules[_DRAIN_NAME] = global_drain\n"
        "setattr(sys.modules[\"whole_body_tracking.tasks.tracking.mdp\"],\n"
        "        \"action_ball_full_mdp_ppo_drain\", global_drain)\n"
        "_DRAIN_SPEC.loader.exec_module(global_drain)",
        "global_drain = sys.modules[_DRAIN_NAME]",
    )
    _device_helper = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(
            "_selected_reset_device_r05_helper", loader=None
        )
    )
    _device_helper.__file__ = str(_device_helper_path)
    sys.modules["_selected_reset_device_r05_helper"] = _device_helper
    exec(
        compile(_device_helper_source, str(_device_helper_path), "exec"),
        _device_helper.__dict__,
    )
    H._HELPER_MODULES[_device_helper_path.name] = _device_helper

    _genesis_path = _SOURCE_ROOT / "action_ball_full_mdp_reset_genesis.py"
    _genesis_source = _genesis_path.read_text(encoding="utf-8").replace(
        ", slots=True", ""
    ).replace("@dataclass(slots=True)", "@dataclass")
    _genesis_module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(
            "action_ball_full_mdp_reset_genesis", loader=None
        )
    )
    _genesis_module.__file__ = str(_genesis_path)
    sys.modules["action_ball_full_mdp_reset_genesis"] = _genesis_module
    exec(
        compile(_genesis_source, str(_genesis_path), "exec"),
        _genesis_module.__dict__,
    )


def _global_drain_module():
    # R06 validates exact class/method identity.  Loading the drain source a
    # second time under a test-only module name would create lookalike classes
    # that the production leaf must reject.
    try:
        return importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_ppo_drain"
        )
    except (ImportError, ModuleNotFoundError):
        return importlib.import_module("action_ball_full_mdp_ppo_drain")


def _physical_module():
    """Return the production Physical owner, never a lookalike test class."""

    return importlib.import_module("action_ball_physical_flight_device")


def _construct_selected_physical(owner):
    physical_module = _physical_module()
    contract_helper = H._test_helper(
        "test_action_ball_physical_flight_contract.py"
    )
    capacity = contract_helper._capacity(
        cadence=1,
        horizon=owner.flight_slot_capacity - 1,
    )
    assert capacity.configured_flight_capacity == owner.flight_slot_capacity
    scene = physical_module.TensorPhysicalFlightScenePort(
        num_envs=owner.num_envs,
        flight_capacity=owner.flight_slot_capacity,
        device=owner.device,
    )
    genesis = physical_module._reset_genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=owner.num_envs, device=torch.device(owner.device)
    )
    physical = physical_module.ActionBallPhysicalFlightDeviceOwner(
        num_envs=owner.num_envs,
        capacity_receipt=capacity,
        expected_capacity_receipt_sha256=capacity.canonical_sha256,
        reset_genesis_authority=genesis.authority,
        reset_genesis_receipt=genesis.receipt,
        scene_body_names=tuple(
            f"action_ball_flight_ball_{index:03d}"
            for index in range(owner.flight_slot_capacity)
        ),
        scene_port=scene,
    )
    return physical


def _row_tensors(owner):
    return {
        name: tensor.detach().clone()
        for name, tensor in owner._checkpoint_tensors().items()
        if (
            name.startswith(("flight_", "mailbox_", "replay_"))
            or name
            in {
                "ingress_fault_bits",
                "post_fault_bits",
                "lifecycle_fault_bits",
                "device_sticky_poison",
                "reset_generation_highwater",
                "selected_reset_count",
            }
        )
    }


def _coordinator_on(
    device,
    *,
    rows=2,
    flight_slots=2,
    mailbox_slots=3,
):
    profile = H._profile()
    registry = H._registry()
    payment_authority = H._payment_authority("A")
    capacity_authority = H._capacity_authority(
        flight_slots=flight_slots,
        mailbox_slots=mailbox_slots,
    )
    owner = D.ActionBallLandingOutcomeDeviceCoordinator(
        num_envs=rows,
        flight_slot_capacity=flight_slots,
        mailbox_capacity=mailbox_slots,
        device=device,
        dtype=torch.float32,
        profile=profile,
        runtime_binding=H._binding(
            profile=profile,
            registry=registry,
            payment_authority=payment_authority,
            capacity_authority=capacity_authority,
        ),
        payment_authority=payment_authority,
        capacity_authority=capacity_authority,
        text_registry=registry,
    )
    owner._test_text_registry = registry
    owner._test_payment_authority = payment_authority
    return owner


def _assert_unselected_row_unchanged(before, after, env_id=1):
    assert before.keys() == after.keys()
    for name in before:
        expected = before[name][env_id].reshape(-1).contiguous().view(
            torch.uint8
        )
        actual = after[name][env_id].reshape(-1).contiguous().view(
            torch.uint8
        )
        assert torch.equal(expected, actual), name


class _ZeroDrainLeaf:
    """Exact zero-valued peer for a real seven-leaf coordinator test."""

    def __init__(self, schema, *, num_envs, device):
        self.schema = schema
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.active_pack = None
        self.active_authority = None
        self.poison_reason = None

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self, *, authority, update_index, completed_environment_steps
    ):
        del update_index, completed_environment_steps
        if self.active_pack is not None or self.poison_reason is not None:
            raise RuntimeError("zero drain leaf is not idle")
        values = torch.zeros(
            self.schema.width(self.num_envs),
            dtype=torch.int64,
            device=self.device,
        )
        pack = authority.mint_device_pack(leaf=self, values=values)
        self.active_pack = pack
        self.active_authority = authority
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack):
        if pack is not self.active_pack:
            raise RuntimeError("zero drain leaf abort pack differs")
        self.active_pack = None
        self.active_authority = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self, *, pack, receipt, owner_row
    ):
        if pack is not self.active_pack or self.active_authority is None:
            raise RuntimeError("zero drain leaf ACK pack differs")
        self.active_authority.require_owned_ack(
            leaf=self,
            pack=pack,
            receipt=receipt,
            owner_row=owner_row,
        )
        self.active_pack = None
        self.active_authority = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason):
        if self.poison_reason is None:
            self.poison_reason = reason


def _global_drain_owner_with_r06(owner):
    drain = _global_drain_module()
    r06_schema = D.materialize_r06_ppo_drain_leaf_schema(
        leaf_schema_type=drain.LeafDrainSchema,
        field_spec_type=drain.DeviceDrainFieldSpec,
    )
    schemas = tuple(
        r06_schema
        if schema.owner_kind == "r06_landing_outcome"
        else schema
        for schema in drain.DEFAULT_LEAF_SCHEMAS
    )
    leaves = {
        schema.owner_kind: _ZeroDrainLeaf(
            schema,
            num_envs=owner.num_envs,
            device=owner.device,
        )
        for schema in schemas
    }
    leaves["r06_landing_outcome"] = owner
    coordinator = drain.ActionBallFullMdpPpoDrainOwner(
        num_envs=owner.num_envs,
        device=owner.device,
        leaves=leaves,
        leaf_schemas=schemas,
    )
    coordinator.require_exact_leaf_bindings(
        {name: leaves[name] for name in drain.OWNER_ORDER}
    )
    return coordinator, leaves


def _prepare_global_drain(coordinator, owner, update_index=0):
    prepared = coordinator.prepare_pre_optimizer_ppo_boundary(
        update_index=update_index,
        completed_environment_steps=owner.num_envs * 24,
    )
    active = coordinator._active
    assert active is not None
    lane = _global_drain_module().OWNER_ORDER.index(
        "r06_landing_outcome"
    )
    pack = active.packs[lane]
    authority = coordinator._authorities[lane]
    values = authority._require(pack, operation_id=active.operation_id)
    assert owner._active_r06_global_drain is not None
    assert owner._active_r06_global_drain.pack is pack
    return prepared, pack, values


@dataclass
class _DeviceResetFixture:
    authority: object
    selected: tuple[int, ...]
    event: object | None = None


def _bind_device_reset_fixture(
    owner,
    selected=(0,),
    initial_generation=1,
    *,
    physical=None,
    bind_physical_r06=True,
):
    helper = H._test_helper(
        "test_action_ball_continuous_runtime_transaction_device.py"
    )
    device = torch.device(owner.device)
    initial = (initial_generation,) * owner.num_envs
    profile = helper._ProfileAuthority(device)
    if physical is None:
        physical = _construct_selected_physical(owner)
    genesis = helper._Genesis(device, owner.num_envs, values=initial)
    genesis.projection = replace(
        genesis.projection,
        world_reset_identity=physical._genesis_world_reset_identity,
    )
    cadence = helper._Cadence(device, owner.num_envs)
    question = helper._Question(device)
    reveal = helper._Reveal(device)
    children = tuple(
        helper._Child(kind) for kind in DEVICE_R05.CHILD_OWNER_ORDER
    )
    reveal.bind_children(children)
    drain = helper._Drain()
    reset = helper._Reset(device, owner.num_envs)
    device_owner = DEVICE_R05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=12345,
        num_envs=owner.num_envs,
        journal_capacity=8,
        max_reveal_epochs_per_drain=8,
        genesis_authority=genesis,
        genesis_receipt=genesis.receipt,
        cadence_authority=cadence,
        question_authority=question,
        reveal_boundary_authority=reveal,
        child_completion_authorities=children,
        drain_authority=drain,
        true_reset_authority=reset,
    )
    reveal.bind_owner(device_owner)
    reset.bind_owner(device_owner)
    owner.bind_device_r05_reset_owner(
        device_owner,
        prepared_reset_validator=(
            device_owner.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            device_owner.require_owned_true_reset_receipt
        ),
    )
    fixture = _DeviceResetFixture(
        authority=reset,
        selected=tuple(selected),
    )
    physical.bind_device_r05_reset_owner(
        device_owner,
        prepared_reset_validator=(device_owner.require_owned_prepared_true_reset),
        r05_receipt_validator=(device_owner.require_owned_true_reset_receipt),
    )
    if bind_physical_r06:
        physical.bind_r06_owner(owner)
    return device_owner, fixture, physical


def _prepare_device_reset(device_owner, fixture):
    event = fixture.authority.issue(device_owner, fixture.selected)
    fixture.event = event
    prepared = device_owner.prepare_true_reset_many(event)
    fixture.authority.allow_commit(prepared)
    return prepared


def _device_reset_fixture(owner, selected=(0,), initial_generation=1):
    device_owner, fixture, physical = _bind_device_reset_fixture(
        owner,
        selected=selected,
        initial_generation=initial_generation,
    )
    prepared = _prepare_device_reset(device_owner, fixture)
    return device_owner, fixture.event, prepared, physical


def _commit_r06_reset(owner, physical, prepared_r05):
    prepared = owner.prepare_selected_reset(prepared_r05)
    physical_staged = physical.stage_selected_true_reset(prepared)
    physical_prepared = physical.finalize_selected_true_reset(
        physical_staged
    )
    armed = owner.arm_prevalidated_selected_reset(
        prepared, physical_prepared
    )
    physical_armed = physical.prearm_selected_true_reset(
        physical_prepared, armed
    )
    physical_commit = physical.commit_prevalidated_selected_true_reset(
        physical_armed
    )
    commit = owner.commit_prevalidated_selected_reset(
        armed, physical_commit
    )
    physical.acknowledge_r06_selected_reset_commit(
        physical_commit, commit
    )
    return prepared, physical_commit, commit


def _complete_device_reset(
    owner,
    physical,
    device_owner,
    fixture,
    prepared_r05,
    physical_commit,
    r06_commit,
):
    receipt = device_owner.commit_true_reset_many(prepared_r05)
    r06_completion = owner.complete_selected_reset_after_r05(
        r06_commit, receipt
    )
    physical_completion = physical.complete_selected_true_reset_after_r05(
        physical_commit, r06_commit, receipt
    )
    actual = {
        "physical_ball": physical_completion,
        "r06_flight": r06_completion,
    }
    for kind in DEVICE_R05.CHILD_OWNER_ORDER:
        child_receipt = actual.get(kind)
        if child_receipt is None:
            child_receipt = fixture.authority.issue_child_completion(
                receipt, kind
            )
        else:
            fixture.authority.committable[
                (receipt, kind, child_receipt)
            ] = True
        device_owner.record_true_reset_child_completion(
            receipt,
            child_kind=kind,
            child_receipt=child_receipt,
        )
    return receipt, r06_completion, physical_completion


def _assert_empty_noncopyable_capability(token, capability_type):
    assert type(token) is capability_type
    assert tuple(capability_type.__slots__) == ()
    assert not hasattr(token, "__dict__")
    assert not any(
        isinstance(getattr(token, name), torch.Tensor)
        for name in dir(token)
        if not name.startswith("__")
    )
    with pytest.raises(TypeError, match="owner-issued"):
        capability_type()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(token)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(token)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(token)


def _settle_selected_env0_while_env1_remains_live(owner, *, retire=True):
    """Publish one real N=2 post-physics row and retire only env0."""

    assert owner.num_envs == 2
    assert owner.flight_slot_capacity == 2
    shape = owner._flight_shape
    observe = (owner._flight_state == D.FLIGHT_INBOUND) | (
        owner._flight_state == D.FLIGHT_OPEN
    )
    assert torch.equal(
        observe,
        torch.tensor(
            ((True, False), (True, False)),
            dtype=torch.bool,
            device=owner.device,
        ),
    )

    def stamp(mask, phase):
        return D.PhysicsStampBatch(
            control_step=torch.where(
                mask,
                torch.ones(shape, dtype=torch.int64, device=owner.device),
                torch.full(
                    shape, -1, dtype=torch.int64, device=owner.device
                ),
            ),
            physics_substep=torch.where(
                mask,
                torch.full(
                    shape, 3, dtype=torch.int32, device=owner.device
                ),
                torch.full(
                    shape, -1, dtype=torch.int32, device=owner.device
                ),
            ),
            event_phase=torch.where(
                mask,
                torch.full(
                    shape, phase, dtype=torch.int8, device=owner.device
                ),
                torch.full(
                    shape, -1, dtype=torch.int8, device=owner.device
                ),
            ),
        )

    previous = torch.zeros(
        shape + (3,), dtype=owner.dtype, device=owner.device
    )
    current = torch.zeros_like(previous)
    previous[0, 0] = torch.tensor(
        (2.1, 0.0, 1.1), dtype=owner.dtype, device=owner.device
    )
    current[0, 0] = torch.tensor(
        (2.2, 0.0, 0.7), dtype=owner.dtype, device=owner.device
    )
    previous[1, 0] = torch.tensor(
        (2.1, 0.0, 1.1), dtype=owner.dtype, device=owner.device
    )
    current[1, 0] = torch.tensor(
        (2.15, 0.0, 1.05), dtype=owner.dtype, device=owner.device
    )
    contact = torch.zeros(shape, dtype=torch.bool, device=owner.device)
    contact[0, 0] = True
    contact_center = torch.zeros_like(previous)
    contact_center[0, 0] = torch.tensor(
        (2.2, 0.0, 0.9), dtype=owner.dtype, device=owner.device
    )
    net = contact.clone()
    crossing = contact.clone()
    crossing_xy = torch.zeros(
        shape + (2,), dtype=owner.dtype, device=owner.device
    )
    crossing_xy[0, 0] = torch.tensor(
        (2.2, 0.0), dtype=owner.dtype, device=owner.device
    )
    physical_publication_identity = object()
    result = owner.publish_post_physics(
        D.PostPhysicsFlightBatch(
            observe_mask=observe,
            full_key_sha256=owner._flight_full_key_sha256.detach().clone(),
            ball_generation=owner._flight_ball_generation.detach().clone(),
            observation_ordinal=torch.where(
                observe,
                owner._flight_observation_ordinal + 1,
                torch.full(
                    shape, -1, dtype=torch.int64, device=owner.device
                ),
            ),
            previous_ball_center_m=previous,
            current_ball_center_m=current,
            observation_stamp=stamp(observe, D.PHASE_LANDING),
            selected_contact_event=contact,
            selected_contact_ball_center_m=contact_center,
            selected_contact_outgoing_segment_anchor_m=(
                contact_center.detach().clone()
            ),
            selected_contact_stamp=stamp(contact, D.PHASE_CONTACT),
            net_crossing_event=net,
            net_clear_at_crossing=net.detach().clone(),
            net_crossing_stamp=stamp(net, D.PHASE_NET),
            crossing_report_delivered=crossing.detach().clone(),
            first_descending_crossing_event=crossing,
            first_descending_crossing_xy_m=crossing_xy,
            first_descending_crossing_stamp=stamp(
                crossing, D.PHASE_LANDING
            ),
            nonfinite_observation=torch.zeros_like(observe),
            producer_contract_fault=torch.zeros_like(observe),
            engine_overflow=torch.zeros_like(observe),
            physical_publication_identity=physical_publication_identity,
        )
    )
    assert torch.equal(
        result.settled_mask,
        torch.tensor(
            ((True, False), (False, False)),
            dtype=torch.bool,
            device=owner.device,
        ),
    )
    assert torch.equal(result.accepted, observe)
    assert not bool(result.rejected.any())
    assert not bool(result.fault_bits.any())
    contact = owner.consume_owned_post_physics_contact_authority(
        result.contact_authority,
        expected_publication_identity=physical_publication_identity,
    )
    assert contact.publication_identity is physical_publication_identity
    assert torch.equal(
        contact.new_valid_contact_mask,
        torch.tensor(
            ((True, False), (False, False)),
            dtype=torch.bool,
            device=owner.device,
        ),
    )
    if retire:
        retired = H._commit_physical_retire(owner, result)
        assert torch.equal(retired.accepted, result.settled_mask)
        assert torch.equal(retired.normal_mask, result.settled_mask)
        assert not bool(retired.cleanup_mask.any())
        assert not bool(retired.fault_bits.any())
    return result


def test_n2_selected_reset_clears_only_selected_rows_and_acks_device_r05():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    harness, reset_authority, physical = _bind_device_reset_fixture(owner)
    owner._previous_paid_payment_step.copy_(
        torch.tensor((13, 17), dtype=torch.int64, device=owner.device)
    )
    owner._previous_paid_payment_step_highwater.copy_(
        torch.tensor((13, 17), dtype=torch.int64, device=owner.device)
    )
    before = _row_tensors(owner)
    prepared_r05 = _prepare_device_reset(harness, reset_authority)

    prepared, physical_commit, commit = _commit_r06_reset(
        owner, physical, prepared_r05
    )
    lease = owner._active_selected_reset_lease
    assert lease is not None
    capability = owner.selected_reset_mask_capability(prepared)
    armed = lease.armed_reset
    assert armed is not None
    _assert_empty_noncopyable_capability(
        prepared, D.PreparedLandingOutcomeSelectedReset
    )
    _assert_empty_noncopyable_capability(
        capability, D.LandingOutcomeSelectedResetMaskCapability
    )
    _assert_empty_noncopyable_capability(
        armed, D.ArmedLandingOutcomeSelectedReset
    )
    _assert_empty_noncopyable_capability(
        commit, D.LandingOutcomeSelectedResetCommitToken
    )
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale or foreign"):
        owner.require_owned_selected_reset_mask_capability(
            object(),
            expected_prepared_reset=prepared,
        )
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale, forged"):
        owner.require_owned_selected_reset_arm(
            object(),
            lease.physical_prepared_token,
        )
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale, forged"):
        owner.require_owned_selected_reset_commit(
            object(),
            expected_prepared_true_reset=prepared_r05,
        )
    after_commit = _row_tensors(owner)
    _assert_unselected_row_unchanged(before, after_commit)
    assert not bool(owner._flight_state[0].ne(D.FLIGHT_EMPTY).any())
    assert not bool(owner._mailbox_reserved[0].any())
    assert not bool(owner._mailbox_history_valid[0].any())
    assert not bool(owner._replay_valid[0])
    assert int(owner._previous_paid_payment_step[0]) == -1
    assert int(owner._previous_paid_payment_step[1]) == 17
    assert int(owner._previous_paid_payment_step_highwater[0]) == -1
    assert int(owner._previous_paid_payment_step_highwater[1]) == 17
    assert int(owner._reset_generation_highwater[0]) == 2
    assert int(owner._selected_reset_count[0]) == 1
    assert int(owner._selected_reset_retired_flight_total) == 0
    assert sum(int(value) for value in owner._invariant_counts()) == 0
    assert owner._active_selected_reset_lease is not None

    receipt, completion, physical_completion = _complete_device_reset(
        owner,
        physical,
        harness,
        reset_authority,
        prepared_r05,
        physical_commit,
        commit,
    )
    _assert_empty_noncopyable_capability(
        completion, D.LandingOutcomeSelectedResetCompletionToken
    )
    assert owner.require_owned_selected_reset_commit(
        commit,
        expected_prepared_true_reset=prepared_r05,
    ) is commit
    assert owner.require_owned_selected_reset_physical_commit(
        commit,
        expected_prepared_true_reset=prepared_r05,
        expected_device_r05_owner=harness,
    ) is commit
    assert owner._active_selected_reset_lease is None
    assert not hasattr(completion, "canonical_sha256")
    assert not hasattr(completion, "to_mapping")
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale or foreign"):
        owner.require_owned_selected_reset_completion(
            object()
        )
    assert owner.require_owned_selected_reset_completion(completion) is completion
    assert owner.consume_owned_selected_reset_completion(completion) is completion
    assert (
        physical.consume_owned_selected_reset_completion(
            physical_completion
        )
        is physical_completion
    )
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale or foreign"):
        owner.require_owned_selected_reset_completion(completion)


def test_partial_payment_debt_blocks_current_d05_to_physical_selected_reset():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    harness, reset_authority, physical = _bind_device_reset_fixture(owner)
    full_key, _key, installed = H._install(owner)
    assert installed.accepted.tolist() == [True, False]
    full_keys = torch.zeros(
        (2, D.TOKEN_BYTES), dtype=torch.uint8, device=owner.device
    )
    full_keys[0].copy_(full_key[0])
    settlement = H._settle_batch_rows(
        owner,
        full_keys=full_keys,
        ball_generations=torch.tensor(
            (0, -1), dtype=torch.int64, device=owner.device
        ),
        observe_envs=(0,),
        step=1,
        retire=False,
    )
    # This test targets Reward debt before selected-reset staging.  Release the
    # test-only legacy-shaped settlement handle without inventing a second
    # Physical owner; the exact D05/Physical pair below remains untouched.
    assert settlement.settled_mask[0, 0].item() is True
    owner._latest_post_physics_settlement = None
    _view, payment, _raw = H._pay(
        owner, D.COMMON_ON_TABLE_CONSUMER, reward_epoch=1
    )
    assert payment.accepted.tolist() == [[True, False, False], [False, False, False]]
    assert owner._mailbox_state[0, 0].item() == D.MAILBOX_PARTIALLY_PAID
    physical_before = physical.scene_snapshot()
    prepared_r05 = _prepare_device_reset(harness, reset_authority)

    with pytest.raises(D.LandingOutcomeDeviceError, match="stale debt"):
        owner.prepare_selected_reset(prepared_r05)

    physical_after = physical.scene_snapshot()
    assert torch.equal(
        physical_before.state_env_f32, physical_after.state_env_f32
    )
    assert owner._active_selected_reset_lease is None
    assert physical._active_selected_reset_stage is None
    assert physical._active_selected_reset_finalize is None


def test_selected_reset_generation_regression_becomes_global_fault_debt():
    drain = _global_drain_module()
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    harness, reset_authority, physical = _bind_device_reset_fixture(
        owner, initial_generation=1
    )
    # Counterexample only: emulate an independently observed R06 generation
    # ahead of Device-R05 before either owner has reset.  This cannot be a
    # positive fixture because no valid same-genesis run should construct it.
    owner._reset_generation_highwater[0] = 2

    prepared_r05 = _prepare_device_reset(harness, reset_authority)
    _prepared, physical_commit, commit = _commit_r06_reset(
        owner, physical, prepared_r05
    )
    assert bool(owner._device_sticky_poison[0])
    assert not bool(owner._device_sticky_poison[1])
    assert int(owner._lifecycle_fault_bits[0]) & D.FAULT_GENERATION_BINDING
    assert torch.equal(
        owner._reset_generation_highwater,
        torch.tensor((2, 1), dtype=torch.int64, device=owner.device),
    )
    _receipt, completion, physical_completion = _complete_device_reset(
        owner,
        physical,
        harness,
        reset_authority,
        prepared_r05,
        physical_commit,
        commit,
    )
    owner.consume_owned_selected_reset_completion(completion)
    physical.consume_owned_selected_reset_completion(physical_completion)

    coordinator, _leaves = _global_drain_owner_with_r06(owner)
    prepared, _pack, values = _prepare_global_drain(coordinator, owner)
    packed = dict(zip(D.R06_GLOBAL_DRAIN_FIELD_NAMES, values.tolist()))
    assert packed["fault_generation_binding_count"] == 1
    assert packed["fault_count"] == 1
    assert packed["invariant_count"] == 0
    with pytest.raises(
        drain.ActionBallFullMdpPpoDrainPoisonedError,
        match="r06_landing_outcome reported a device fault",
    ):
        coordinator.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert coordinator.poisoned is True
    assert owner._r06_global_drain_poisoned is True
    assert owner._poisoned is True


def test_selected_reset_generation_max_never_wraps_and_blocks_optimizer_ack():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    with pytest.raises(
        DEVICE_R05.DeviceR05Error,
        match="no positive int64 continuation",
    ):
        _bind_device_reset_fixture(
            owner,
            initial_generation=torch.iinfo(torch.int64).max,
        )
    assert torch.equal(
        owner._reset_generation_highwater,
        torch.zeros(2, dtype=torch.int64, device=owner.device),
    )


def test_selected_reset_prepare_is_exclusive_and_abort_is_prearm_only():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    _harness, _event, prepared_r05, physical = _device_reset_fixture(owner)
    before = _row_tensors(owner)
    prepared = owner.prepare_selected_reset(prepared_r05)
    capability = owner.selected_reset_mask_capability(prepared)
    view = owner.require_owned_selected_reset_mask_capability(
        capability,
        expected_prepared_reset=prepared,
    )
    view.device_mask.data.logical_not_()
    view.generation_before.data.add_(99)
    view.generation_after.add_(77)
    repeated = owner.require_owned_selected_reset_mask_capability(
        capability,
        expected_prepared_reset=prepared,
    )
    assert torch.equal(
        repeated.device_mask,
        torch.tensor((True, False), device=owner.device),
    )
    assert torch.equal(
        repeated.generation_before,
        torch.ones(2, dtype=torch.int64, device=owner.device),
    )
    assert torch.equal(
        repeated.generation_after,
        torch.tensor((2, 1), dtype=torch.int64, device=owner.device),
    )
    _assert_empty_noncopyable_capability(
        prepared, D.PreparedLandingOutcomeSelectedReset
    )
    _assert_empty_noncopyable_capability(
        capability, D.LandingOutcomeSelectedResetMaskCapability
    )
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale, forged"):
        owner.require_owned_selected_reset_prepare(object())
    with pytest.raises(D.LandingOutcomeDeviceError, match="selected-reset lease"):
        owner.drain_ppo_boundary(update_index=0)
    owner.abort_selected_reset(prepared)
    _assert_unselected_row_unchanged(before, _row_tensors(owner), env_id=0)

    prepared = owner.prepare_selected_reset(prepared_r05)
    physical_staged = physical.stage_selected_true_reset(prepared)
    physical_prepared = physical.finalize_selected_true_reset(
        physical_staged
    )
    owner.arm_prevalidated_selected_reset(prepared, physical_prepared)
    with pytest.raises(D.LandingOutcomeDeviceError, match="crossed prearm"):
        owner.abort_selected_reset(prepared)


def test_wrong_r05_ack_after_physical_and_r06_commit_is_sticky_debt():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    harness, reset_authority, physical = _bind_device_reset_fixture(owner)
    prepared_r05 = _prepare_device_reset(harness, reset_authority)
    _prepared, _physical_commit, commit = _commit_r06_reset(
        owner, physical, prepared_r05
    )
    committed = {
        name: value.detach().clone()
        for name, value in owner._checkpoint_tensors().items()
    }

    with pytest.raises(
        DEVICE_R05.DeviceR05ConflictError,
        match="receipt type differs",
    ):
        owner.complete_selected_reset_after_r05(commit, object())
    assert owner._poisoned is True
    assert owner._active_selected_reset_lease is not None
    assert owner._active_selected_reset_lease.commit_token is commit
    assert owner._latest_selected_reset_completion is None
    for name, expected in committed.items():
        actual = owner._checkpoint_tensors()[name]
        assert torch.equal(
            expected.reshape(-1).contiguous().view(torch.uint8),
            actual.reshape(-1).contiguous().view(torch.uint8),
        ), name

    receipt = harness.commit_true_reset_many(prepared_r05)
    with pytest.raises(D.LandingOutcomeDeviceError, match="stale, forged"):
        owner.complete_selected_reset_after_r05(commit, receipt)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for byte-exact device-row reset coverage",
)
def test_cuda_selected_reset_preserves_unselected_row_bitwise():
    owner = _coordinator_on(
        torch.device("cuda", torch.cuda.current_device())
    )
    harness, _event, prepared_r05, physical = _device_reset_fixture(owner)
    owner._flight_target_xy_m[1, 0, 0] = torch.tensor(
        float("nan"), dtype=owner.dtype, device=owner.device
    )
    owner._flight_target_xy_m[1, 0, 1] = torch.tensor(
        -0.0, dtype=owner.dtype, device=owner.device
    )
    nan_bits = (
        owner._flight_target_xy_m[1, 0, 0:1]
        .contiguous()
        .view(torch.uint8)
        .detach()
        .clone()
    )
    negative_zero_bits = (
        owner._flight_target_xy_m[1, 0, 1:2]
        .contiguous()
        .view(torch.uint8)
        .detach()
        .clone()
    )
    before = _row_tensors(owner)
    _prepared, physical_commit, commit = _commit_r06_reset(
        owner, physical, prepared_r05
    )
    after = _row_tensors(owner)
    _assert_unselected_row_unchanged(before, after)
    assert torch.equal(
        owner._flight_target_xy_m[1, 0, 0:1]
        .contiguous()
        .view(torch.uint8),
        nan_bits,
    )
    assert torch.equal(
        owner._flight_target_xy_m[1, 0, 1:2]
        .contiguous()
        .view(torch.uint8),
        negative_zero_bits,
    )
    assert torch.equal(
        owner._reset_generation_highwater,
        torch.tensor((2, 1), dtype=torch.int64, device=owner.device),
    )
    fixture = _DeviceResetFixture(
        authority=harness._true_reset_authority,
        selected=(0,),
        event=_event,
    )
    _receipt, completion, physical_completion = _complete_device_reset(
        owner,
        physical,
        harness,
        fixture,
        prepared_r05,
        physical_commit,
        commit,
    )
    assert owner.consume_owned_selected_reset_completion(completion) is completion
    assert (
        physical.consume_owned_selected_reset_completion(
            physical_completion
        )
        is physical_completion
    )


def test_global_drain_packs_complete_r06_audit_and_exact_ack():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    coordinator, leaves = _global_drain_owner_with_r06(owner)
    prepared, _pack, values = _prepare_global_drain(coordinator, owner)
    assert tuple(values.shape) == (len(D.R06_GLOBAL_DRAIN_FIELD_NAMES),)
    assert tuple(D.R06_GLOBAL_DRAIN_FIELD_NAMES[:8]) == (
        "mutation_version",
        "fault_count",
        "invariant_count",
        "terminal_resolution_total",
        "shared_normal_retire_total",
        "r06_only_orphan_retire_total",
        "shared_normal_retire_key_summary_0",
        "shared_normal_retire_key_summary_1",
    )
    receipt = coordinator.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert receipt.acknowledged is False
    with pytest.raises(
        D.LandingOutcomeDeviceError,
        match="lease blocks|no acknowledged",
    ):
        owner.require_owned_pre_optimizer_ppo_boundary_receipt(receipt)
    coordinator.mark_optimizer_returned(receipt)
    coordinator.acknowledge_post_update(receipt)
    assert receipt.acknowledged is True
    assert leaves["r06_landing_outcome"] is owner
    portable = owner.latest_pre_optimizer_ppo_boundary_receipt
    assert portable.update_index == 0
    assert portable.device_to_host_transfers == 1
    assert (
        owner.require_owned_pre_optimizer_ppo_boundary_receipt(receipt)
        is portable
    )
    with pytest.raises(D.LandingOutcomeDeviceError, match="legacy R06"):
        owner.drain_ppo_boundary(update_index=1)

    checkpoint = owner.state_dict(portable)
    assert checkpoint["drain_protocol"] == D._R06_GLOBAL_DRAIN_PROTOCOL
    assert checkpoint["global_drain_adopted"] is True
    restored = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    restored.load_state_dict(
        checkpoint,
        expected_checkpoint_content_sha256=checkpoint[
            "checkpoint_content_sha256"
        ],
    )
    assert restored._r06_global_drain_adopted is True
    with pytest.raises(D.LandingOutcomeDeviceError, match="legacy R06"):
        restored.drain_ppo_boundary(update_index=1)


def test_global_drain_clean_abort_is_write_free_and_adoption_is_monotonic():
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    coordinator, _leaves = _global_drain_owner_with_r06(owner)
    before = {
        name: value.detach().clone()
        for name, value in owner._checkpoint_tensors().items()
    }
    prepared, _pack, _values = _prepare_global_drain(coordinator, owner)
    coordinator.abort_pre_optimizer_ppo_boundary(prepared)
    for name, expected in before.items():
        assert torch.equal(owner._checkpoint_tensors()[name], expected), name
    assert owner._active_r06_global_drain is None
    with pytest.raises(D.LandingOutcomeDeviceError, match="legacy R06"):
        owner.drain_ppo_boundary(update_index=0)


def test_global_drain_forged_row_sticky_poisons_without_new_d2h():
    drain = _global_drain_module()
    owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    coordinator, _leaves = _global_drain_owner_with_r06(owner)
    prepared, pack, _values = _prepare_global_drain(coordinator, owner)
    receipt = coordinator.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    coordinator.mark_optimizer_returned(receipt)
    row = next(
        value
        for value in receipt.owner_rows
        if value.owner_kind == "r06_landing_outcome"
    )

    foreign_owner = H._coordinator(
        rows=2,
        flight_slots=2,
        mailbox_slots=3,
        bind_physical_park=False,
    )
    foreign_coordinator, _foreign_leaves = _global_drain_owner_with_r06(
        foreign_owner
    )
    foreign_prepared, _foreign_pack, _foreign_values = _prepare_global_drain(
        foreign_coordinator, foreign_owner
    )
    foreign_receipt = (
        foreign_coordinator.transfer_decode_pre_optimizer_ppo_boundary(
            foreign_prepared
        )
    )
    foreign_coordinator.mark_optimizer_returned(foreign_receipt)
    foreign_row = next(
        value
        for value in foreign_receipt.owner_rows
        if value.owner_kind == "r06_landing_outcome"
    )
    assert foreign_row.values == row.values

    with pytest.raises(
        drain.ActionBallFullMdpPpoDrainError,
        match="foreign|stale|out of window|lane-swapped",
    ):
        owner.acknowledge_pre_optimizer_ppo_boundary(
            pack=pack,
            receipt=foreign_receipt,
            owner_row=foreign_row,
        )
    assert owner._r06_global_drain_poisoned is True
    assert owner._poisoned is True
    assert owner._active_r06_global_drain is not None
    with pytest.raises(D.LandingOutcomeDeviceError, match="poisoned"):
        owner.acknowledge_pre_optimizer_ppo_boundary(
            pack=pack,
            receipt=receipt,
            owner_row=row,
        )


def test_global_drain_source_and_ack_api_pins_match_frozen_authority():
    drain = _global_drain_module()
    source = inspect.getsource(drain)
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == (
        D._R06_GLOBAL_DRAIN_SOURCE_SHA256
    )

    tree = ast.parse(source)
    authority_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LeafDevicePackAuthority"
    )
    method = next(
        node
        for node in authority_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "require_owned_ack"
    )
    method_source = ast.get_source_segment(source, method, padded=False)
    encoded = json.dumps(
        {
            "fields": (),
            "methods": (("require_owned_ack", method_source),),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == (
        D._R06_GLOBAL_DRAIN_ACK_AUTHORITY_API_SHA256
    )


def test_selected_reset_hot_methods_have_no_host_tensor_materialization():
    names = (
        "prepare_selected_reset",
        "require_owned_selected_reset_prepare",
        "selected_reset_mask_capability",
        "require_owned_selected_reset_mask_capability",
        "arm_prevalidated_selected_reset",
        "require_owned_selected_reset_arm",
        "commit_prevalidated_selected_reset",
        "require_owned_selected_reset_commit",
        "require_owned_selected_reset_physical_commit",
        "complete_selected_reset_after_r05",
        "require_owned_selected_reset_completion",
        "consume_owned_selected_reset_completion",
    )
    forbidden = (".item(", ".cpu(", ".tolist(", "bool(")
    for name in names:
        source = inspect.getsource(
            getattr(D.ActionBallLandingOutcomeDeviceCoordinator, name)
        )
        assert not any(token in source for token in forbidden), name
    signature = inspect.signature(
        D.ActionBallLandingOutcomeDeviceCoordinator.prepare_selected_reset
    )
    assert tuple(signature.parameters) == ("self", "prepared_true_reset")

    global_prepare = inspect.getsource(
        D.ActionBallLandingOutcomeDeviceCoordinator.prepare_pre_optimizer_ppo_boundary_device_pack
    )
    for token in (".item(", ".cpu(", ".tolist(", ".numpy("):
        assert token not in global_prepare
