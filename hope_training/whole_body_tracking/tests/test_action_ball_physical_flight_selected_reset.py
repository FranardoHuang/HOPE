from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import inspect
import pickle
from pathlib import Path
import sys

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_WBT_ROOT = _HERE.parent
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
_MDP_ROOT = (
    _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
)
for _path in (_SOURCE_ROOT, _MDP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load(
    "_physical_selected_reset_r06_helper",
    _HERE / "test_action_ball_landing_outcome_device.py",
)
D = _load(
    "action_ball_physical_flight_device",
    _MDP_ROOT / "action_ball_physical_flight_device.py",
)
R06 = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_landing_outcome_device"
)
H.D = R06
R05D = D._r05_device


def _global_drain_module():
    canonical = sys.modules.get(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_ppo_drain"
    )
    return canonical or importlib.import_module("action_ball_full_mdp_ppo_drain")


def _global_drain_helper():
    return H._test_helper("test_action_ball_full_mdp_ppo_drain.py")

from test_action_ball_physical_flight_contract import _capacity  # noqa: E402


class _GenesisAuthority:
    def __init__(self, generations):
        self.receipt = object()
        self.generations = tuple(generations)

    def require_owned_r05_genesis(self, receipt, *, device, num_envs):
        if receipt is not self.receipt or num_envs != len(self.generations):
            raise RuntimeError("foreign reset-generation genesis")
        return R05D.DeviceGenesisProjection(
            world_reset_identity=self.receipt,
            reset_generations=torch.tensor(
                self.generations,
                dtype=torch.int64,
                device=device,
            ),
        )


class _UnusedCadenceAuthority:
    """Current full-N Motion source; selected-reset never pulls a row."""

    def __init__(self, device, num_envs):
        self.device = torch.device(device)
        self.num_envs = num_envs

    def project_current_action_epoch_rows(self):
        raise AssertionError("Motion rows are outside selected-reset coverage")


class _UnusedQuestionAuthority:
    def project_r05_candidate_bank(self, *_args, **_kwargs):
        raise AssertionError("candidate bank is outside selected-reset coverage")


class _UnusedRevealAuthority:
    def project_owned_r05_reveal_boundary(self, *_args, **_kwargs):
        raise AssertionError("reveal is outside selected-reset coverage")

    def require_owned_r05_reveal_boundary(self, *_args, **_kwargs):
        raise AssertionError("reveal is outside selected-reset coverage")

    def require_owned_r05_terminal_arm(self, *_args, **_kwargs):
        raise AssertionError("terminal arm is outside selected-reset coverage")

    def require_owned_r05_terminal_commit(self, *_args, **_kwargs):
        raise AssertionError("terminal commit is outside selected-reset coverage")


class _UnusedChildAuthority:
    def require_owned_r05_child_completion(self, *_args, **_kwargs):
        raise AssertionError("reveal child ACK is outside selected-reset coverage")


class _UnusedDrainAuthority:
    def materialize_r05_device_drain(self, *_args, **_kwargs):
        raise AssertionError("portable drain is outside selected-reset coverage")

    def require_owned_r05_drain_ack(self, *_args, **_kwargs):
        raise AssertionError("portable drain is outside selected-reset coverage")


class _TrueResetAuthority:
    def __init__(self, selected, generations):
        self.selected = tuple(selected)
        self.generations = tuple(generations)
        self.event = object()
        self.prepared = None
        self.preflight = None
        self.children_committed = False
        self.live_reset_ledger_identity = None

    def project_r05_true_reset(
        self,
        receipt,
        *,
        device,
        num_envs,
        live_reset_ledger_identity,
        live_reset_generation,
    ):
        if (
            receipt is not self.event
            or num_envs != len(self.generations)
            or live_reset_ledger_identity is not self.live_reset_ledger_identity
        ):
            raise RuntimeError("foreign selected-reset event")
        assert tuple(live_reset_generation.shape) == (num_envs,)
        index = torch.tensor(self.selected, dtype=torch.int64, device=device)
        mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        mask[index] = True
        return R05D.DeviceTrueResetEventProjection(
            reset_event_identity=self.event,
            selected_env_index=index,
            selected_mask=mask,
        )

    def require_owned_r05_true_reset_preflight(
        self, prepared, *, preflight_capability
    ):
        if prepared is not self.prepared or preflight_capability is not self.preflight:
            raise RuntimeError("foreign selected-reset preflight")
        return R05D.DeviceTrueResetPreflightProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.event,
            preflight_capability=preflight_capability,
        )

    def require_owned_r05_true_reset_commit(self, prepared, *, owner_view):
        if prepared is not self.prepared or not self.children_committed:
            raise RuntimeError("four child commits are not retained")
        if (
            type(owner_view) is not R05D.DeviceR05TrueResetCommitInput
            or owner_view.prepared_true_reset is not prepared
            or owner_view.reset_event_identity is not self.event
        ):
            raise RuntimeError("foreign Device-R05 reset commit view")
        return R05D.DeviceTrueResetCommitProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.event,
            child_kinds=R05D.CHILD_OWNER_ORDER,
            child_commit_identities=tuple(object() for _ in R05D.CHILD_OWNER_ORDER),
            preflight_capability=self.preflight,
        )

    def require_owned_r05_true_reset_abort(self, prepared):
        if prepared is not self.prepared:
            raise RuntimeError("foreign selected-reset prepare")
        return R05D.DeviceTrueResetAbortProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.event,
            child_commits_started=self.children_committed,
        )

    def require_owned_r05_true_reset_child_completion(
        self, receipt, *, child_kind, child_receipt
    ):
        return R05D.DeviceTrueResetChildCompletionProjection(
            true_reset_receipt=receipt,
            child_kind=child_kind,
            child_receipt=child_receipt,
        )


def _device_r05(device, genesis, *, generations=(1, 1), selected=(0,)):
    helper = H._test_helper(
        "test_action_ball_continuous_runtime_transaction_device.py"
    )
    profile_authority = helper._ProfileAuthority(device)
    reset = _TrueResetAuthority(selected, generations)
    owner = R05D.DeviceR05Owner(
        profile_authority,
        profile_authority.receipt,
        seed=12345,
        num_envs=len(generations),
        journal_capacity=8,
        max_reveal_epochs_per_drain=8,
        genesis_authority=genesis.authority,
        genesis_receipt=genesis.receipt,
        cadence_authority=_UnusedCadenceAuthority(
            device, len(generations)
        ),
        question_authority=_UnusedQuestionAuthority(),
        reveal_boundary_authority=_UnusedRevealAuthority(),
        child_completion_authorities=tuple(
            _UnusedChildAuthority() for _ in R05D.CHILD_OWNER_ORDER
        ),
        drain_authority=_UnusedDrainAuthority(),
    )
    owner._reset_generation.copy_(
        torch.tensor(generations, dtype=torch.int64, device=owner.device)
    )
    env_binding = owner.project_full_mdp_env_reset_binding()
    env_view = owner.require_owned_full_mdp_env_reset_binding(env_binding)
    reset.live_reset_ledger_identity = env_view.live_reset_ledger_identity
    return owner, reset


def _owners(device, *, generations=(1, 1), selected=(0,)):
    capacity = _capacity(cadence=1, horizon=0)
    exact_device = torch.device(device)
    if exact_device.type == "cuda" and exact_device.index is None:
        exact_device = torch.device("cuda", torch.cuda.current_device())
    scene = D.TensorPhysicalFlightScenePort(
        num_envs=2,
        flight_capacity=1,
        device=exact_device,
    )
    genesis = D._reset_genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=2, device=exact_device
    )
    physical = D.ActionBallPhysicalFlightDeviceOwner(
        num_envs=2,
        capacity_receipt=capacity,
        expected_capacity_receipt_sha256=capacity.canonical_sha256,
        reset_genesis_authority=genesis.authority,
        reset_genesis_receipt=genesis.receipt,
        scene_body_names=("action_ball_flight_ball_000",),
        scene_port=scene,
    )
    # The real genesis owner fixes production at generation one.  This focused
    # component counterexample may then place the already-constructed device
    # owner at an int64 boundary before any reset binding or write.
    physical._device_reset_generation.copy_(
        torch.tensor(generations, dtype=torch.int64, device=physical.device)
    )
    r06 = H._coordinator(
        rows=2,
        flight_slots=1,
        mailbox_slots=1,
        bind_physical_park=False,
    )
    if r06.device != physical.device:
        # H._coordinator is a CPU fixture.  Construct its exact device twin.
        profile = H._profile()
        registry = H._registry()
        payment = H._payment_authority("A")
        capacity_authority = H._capacity_authority(
            flight_slots=1, mailbox_slots=1
        )
        r06 = R06.ActionBallLandingOutcomeDeviceCoordinator(
            num_envs=2,
            flight_slot_capacity=1,
            mailbox_capacity=1,
            device=physical.device,
            dtype=torch.float32,
            profile=profile,
            runtime_binding=H._binding(
                profile=profile,
                registry=registry,
                payment_authority=payment,
                capacity_authority=capacity_authority,
            ),
            payment_authority=payment,
            capacity_authority=capacity_authority,
            text_registry=registry,
        )
    device_r05, reset = _device_r05(
        physical.device,
        genesis,
        generations=generations,
        selected=selected,
    )
    physical.bind_device_r05_reset_owner(
        device_r05,
        prepared_reset_validator=(
            device_r05.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            device_r05.require_owned_true_reset_receipt
        ),
    )
    physical._device_reset_generation.copy_(
        torch.tensor(generations, dtype=torch.int64, device=physical.device)
    )
    r06.bind_device_r05_reset_owner(
        device_r05,
        prepared_reset_validator=(
            device_r05.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            device_r05.require_owned_true_reset_receipt
        ),
    )
    device_r05.bind_true_reset_authority(reset)
    physical.bind_r06_owner(r06)
    prepared_r05 = device_r05.prepare_true_reset_many(reset.event)
    reset.prepared = prepared_r05
    reset.preflight = object()
    device_r05.register_true_reset_preflight(prepared_r05, reset.preflight)
    return physical, scene, r06, device_r05, reset, prepared_r05


def _physical_row_bytes(owner, scene, env_id):
    tensors = (
        scene.read_state_env(),
        owner._lifecycle,
        owner._generation,
        owner._outcome_sha,
        owner._install_sha,
        owner._installed_state_sha,
        owner._reveal_step,
        owner._observation_ordinal,
        owner._previous_ball_center,
        owner._parked,
        owner._published,
        owner._slot_version,
        owner._device_fault,
        owner._device_reset_generation,
        owner._action_epoch_active_flight_slot,
        *(
            getattr(owner._action_epoch_flight_shot_key, field.name)
            for field in D.fields(D._row_identity.ActionEpochShotKey)
        ),
        owner._action_epoch_flight_publication_ordinal,
    )
    return tuple(tensor[env_id].detach().cpu().numpy().tobytes() for tensor in tensors)


def _float32_from_bits(bits, *, device):
    unsigned = torch.tensor(bits, dtype=torch.int64, device=device)
    signed = torch.where(unsigned >= (1 << 31), unsigned - (1 << 32), unsigned)
    return signed.to(torch.int32).view(torch.float32)


def _stage_and_arm(physical, r06, prepared_r05):
    prepared_r06 = r06.prepare_selected_reset(prepared_r05)
    staged = physical.stage_selected_true_reset(prepared_r06)
    finalized = physical.finalize_selected_true_reset(staged)
    armed_r06 = r06.arm_prevalidated_selected_reset(
        prepared_r06, finalized
    )
    armed_physical = physical.prearm_selected_true_reset(
        finalized, armed_r06
    )
    return prepared_r06, staged, finalized, armed_r06, armed_physical


@pytest.mark.parametrize("runtime_device", ["cpu", "cuda"])
def test_n2_mixed_selected_parks_first_preserves_unselected_and_acks_last(
    runtime_device,
):
    if runtime_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    physical, scene, r06, r05, reset, prepared_r05 = _owners(runtime_device)
    scene._state[0, 0, 0] = 2.25
    scene._state[1, 0, 0] = 9.5
    physical._lifecycle.fill_(D.R06_FLIGHT_INBOUND)
    physical._parked.zero_()
    physical._published.fill_(True)
    physical._generation.fill_(11)
    physical._slot_version.copy_(
        torch.tensor([[3], [7]], dtype=torch.int64, device=physical.device)
    )
    before_unselected = _physical_row_bytes(physical, scene, 1)

    _, _, _, armed_r06, armed_physical = _stage_and_arm(
        physical, r06, prepared_r05
    )
    park = physical.commit_prevalidated_selected_true_reset(armed_physical)
    assert physical.require_owned_selected_reset_commit(park) is park
    assert physical.require_owned_selected_reset_commit(park) is park
    assert scene.apply_count == 1
    assert physical._parked[0].all()
    assert not physical._published[0].any()
    assert torch.equal(
        physical._device_reset_generation,
        torch.tensor([2, 1], dtype=torch.int64, device=physical.device),
    )
    assert _physical_row_bytes(physical, scene, 1) == before_unselected

    r06_commit = r06.commit_prevalidated_selected_reset(armed_r06, park)
    physical.acknowledge_r06_selected_reset_commit(park, r06_commit)
    reset.children_committed = True
    receipt = r05.commit_true_reset_many(prepared_r05)
    completion = physical.complete_selected_true_reset_after_r05(
        park, r06_commit, receipt
    )
    assert physical.require_owned_selected_reset_completion(completion) is completion
    assert physical.consume_owned_selected_reset_completion(completion) is completion
    with pytest.raises(D.PhysicalFlightDeviceError, match="stale or foreign"):
        physical.consume_owned_selected_reset_completion(completion)
    assert not hasattr(completion, "canonical_sha256")
    assert not hasattr(completion, "to_mapping")


@pytest.mark.parametrize("runtime_device", ["cpu", "cuda"])
def test_unselected_nan_payload_and_signed_zero_are_bit_identical(
    runtime_device,
):
    if runtime_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    physical, scene, r06, _r05, _reset, prepared_r05 = _owners(
        runtime_device
    )
    # Quiet NaN with a non-default payload and the negative-zero sign bit.
    scene._state[1, 0, :2].copy_(
        _float32_from_bits(
            [0x7FC12345, 0x80000000],
            device=physical.device,
        )
    )
    physical._previous_ball_center[1, 0, :2].copy_(
        _float32_from_bits(
            [0x7FC54321, 0x80000000],
            device=physical.device,
        )
    )
    before = _physical_row_bytes(physical, scene, 1)
    _, _, _, _armed_r06, armed_physical = _stage_and_arm(
        physical, r06, prepared_r05
    )
    physical.commit_prevalidated_selected_true_reset(armed_physical)
    assert _physical_row_bytes(physical, scene, 1) == before


def test_prearm_abort_is_zero_write_and_prearm_crosses_fail_stop_boundary():
    physical, scene, r06, _r05, _reset, prepared_r05 = _owners("cpu")
    before = _physical_row_bytes(physical, scene, 0)
    prepared_r06 = r06.prepare_selected_reset(prepared_r05)
    staged = physical.stage_selected_true_reset(prepared_r06)
    finalized = physical.finalize_selected_true_reset(staged)
    physical.abort_selected_true_reset(finalized)
    r06.abort_selected_reset(prepared_r06)
    assert scene.apply_count == 0
    assert _physical_row_bytes(physical, scene, 0) == before

    # A top failure broadcast after prearm is sticky and performs no rollback.
    _, _, _, _, _ = _stage_and_arm(physical, r06, prepared_r05)
    with pytest.raises(D.PhysicalFlightDeviceError, match="crossed prearm"):
        physical.abort_selected_true_reset(
            physical._active_selected_reset_finalize
        )
    physical.poison_selected_reset("injected peer failure after prearm")
    with pytest.raises(D.PhysicalFlightOwnerPoisonedError):
        physical.scene_snapshot()


def test_apply_then_raise_keeps_parked_scene_and_poison_never_rolls_back():
    physical, scene, r06, _r05, _reset, prepared_r05 = _owners("cpu")
    scene._state[0, 0, 0] = 2.25
    before_unselected = _physical_row_bytes(physical, scene, 1)
    _, _, _, _armed_r06, armed_physical = _stage_and_arm(
        physical, r06, prepared_r05
    )
    original = scene.apply_prevalidated_write

    def apply_then_raise(handle):
        original(handle)
        raise RuntimeError("writer failed after physical bytes became visible")

    scene.apply_prevalidated_write = apply_then_raise
    with pytest.raises(
        D.PhysicalFlightOwnerPoisonedError, match="rollback is untrusted"
    ):
        physical.commit_prevalidated_selected_true_reset(armed_physical)
    assert scene.apply_count == 1
    assert torch.equal(scene._state[0], physical._park_state_template[0])
    assert _physical_row_bytes(physical, scene, 1) == before_unselected
    with pytest.raises(D.PhysicalFlightOwnerPoisonedError):
        physical.scene_snapshot()


def test_selected_reset_hot_path_source_forbids_host_tensor_observation():
    for name in (
        "stage_selected_true_reset",
        "finalize_selected_true_reset",
        "prearm_selected_true_reset",
        "commit_prevalidated_selected_true_reset",
        "acknowledge_r06_selected_reset_commit",
        "complete_selected_true_reset_after_r05",
    ):
        source = inspect.getsource(
            getattr(D.ActionBallPhysicalFlightDeviceOwner, name)
        )
        assert not any(
            marker in source
            for marker in (".item(", ".tolist(", ".cpu(", ".numpy(", "bool(")
        ), name
    signature = inspect.signature(
        D.ActionBallPhysicalFlightDeviceOwner.stage_selected_true_reset
    )
    assert tuple(signature.parameters) == ("self", "r06_prepared_reset")
    stage_source = inspect.getsource(
        D.ActionBallPhysicalFlightDeviceOwner.stage_selected_true_reset
    )
    finalize_source = inspect.getsource(
        D.ActionBallPhysicalFlightDeviceOwner.finalize_selected_true_reset
    )
    assert '_prepare_action_epoch_scene_write(' in finalize_source
    assert 'kind="retire"' in finalize_source
    assert "scene_port.preflight_write" not in finalize_source
    require_operable_source = inspect.getsource(
        D.ActionBallPhysicalFlightDeviceOwner._require_operable
    )
    assert "diagnostic_selected_reset_capability" in require_operable_source
    checkpoint_source = inspect.getsource(
        D.ActionBallPhysicalFlightDeviceOwner.
        _pending_r06_settlement_ack_for_legacy_checkpoint_rejection
    )
    assert "diagnostic_selected_reset_capability" not in checkpoint_source
    for forbidden_private_alias in (
        "_device_r05_prepared_true_reset",
        "_selected_env_mask",
        "_device_mask",
    ):
        assert forbidden_private_alias not in stage_source
    assert "require_owned_selected_reset_mask_capability" in stage_source
    assert '"_diagnostic_n2_construction_record", None' in stage_source
    assert "_action_ball_full_mdp_epoch_owner" in stage_source
    assert "_selected_reset_physical_park_token_authority" in stage_source
    assert "_action_epoch_physical_owner" in stage_source
    assert "_action_epoch_device_r05_owner" in stage_source
    assert "_diagnostic_physical_owner" in stage_source
    assert "_diagnostic_epoch_owner" in stage_source
    bind_source = inspect.getsource(
        D.ActionBallPhysicalFlightDeviceOwner.bind_device_r05_reset_owner
    )
    assert "device_r05_owner.num_envs" not in bind_source
    assert "device_r05_owner.device" not in bind_source


@pytest.mark.parametrize("runtime_device", ["cpu", "cuda"])
def test_generation_fault_settles_safely_but_opaque_ack_cannot_claim_health(
    runtime_device,
):
    if runtime_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    physical, scene, r06, r05, reset, prepared_r05 = _owners(
        runtime_device
    )
    # Simulate a stale physical chronology without changing Device-R05's exact
    # prepared generation.  The reset must still park first and complete last;
    # the mismatch is a retained device fault, never a rollback trigger.
    physical._device_reset_generation[0] = 9
    _, _, _, armed_r06, armed_physical = _stage_and_arm(
        physical, r06, prepared_r05
    )
    park = physical.commit_prevalidated_selected_true_reset(armed_physical)
    assert scene.apply_count == 1
    assert physical._parked[0].all()
    r06_commit = r06.commit_prevalidated_selected_reset(armed_r06, park)
    physical.acknowledge_r06_selected_reset_commit(park, r06_commit)
    reset.children_committed = True
    receipt = r05.commit_true_reset_many(prepared_r05)
    completion = physical.complete_selected_true_reset_after_r05(
        park, r06_commit, receipt
    )
    assert physical._device_fault[0].all()
    assert not hasattr(completion, "__dict__")
    assert type(completion).__slots__ == ()
    assert not hasattr(completion, "success")
    assert not hasattr(completion, "healthy")
    physical.consume_owned_selected_reset_completion(completion)

    # The only optimizer-bound materialization carries the exact fault count;
    # the sole global D2H rejects it before minting any healthy receipt.
    drained_physical, drain_owner = _real_physical_drain(runtime_device)
    drained_physical._device_fault.copy_(physical._device_fault)
    prepared_drain = drain_owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    values = _active_physical_drain_values(drained_physical)
    row_values = dict(zip(D.PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES, values))
    assert row_values["fault_count"] == 1
    with pytest.raises(Exception, match="device fault"):
        drain_owner.transfer_decode_pre_optimizer_ppo_boundary(prepared_drain)
    assert drained_physical._physical_global_drain_poisoned


@pytest.mark.parametrize("runtime_device", ["cpu", "cuda"])
def test_generation_max_settles_without_wrap_and_blocks_global_ack(
    runtime_device,
):
    if runtime_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    maximum = torch.iinfo(torch.int64).max
    physical, scene, r06, r05, reset, prepared_r05 = _owners(
        runtime_device,
        generations=(maximum, maximum),
    )
    _, _, _, armed_r06, armed_physical = _stage_and_arm(
        physical, r06, prepared_r05
    )
    park = physical.commit_prevalidated_selected_true_reset(armed_physical)
    assert scene.apply_count == 1
    assert torch.equal(
        physical._device_reset_generation,
        torch.full(
            (2,), maximum, dtype=torch.int64, device=physical.device
        ),
    )
    assert physical._device_fault[0].all()
    assert not physical._device_fault[1].any()
    r06_commit = r06.commit_prevalidated_selected_reset(armed_r06, park)
    physical.acknowledge_r06_selected_reset_commit(park, r06_commit)
    reset.children_committed = True
    receipt = r05.commit_true_reset_many(prepared_r05)
    completion = physical.complete_selected_true_reset_after_r05(
        park, r06_commit, receipt
    )
    physical.consume_owned_selected_reset_completion(completion)
    drained_physical, drain_owner = _real_physical_drain(runtime_device)
    drained_physical._device_fault.copy_(physical._device_fault)
    prepared_drain = drain_owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    values = _active_physical_drain_values(drained_physical)
    packed = dict(zip(D.PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES, values))
    assert packed["fault_count"] == 1
    with pytest.raises(Exception, match="device fault"):
        drain_owner.transfer_decode_pre_optimizer_ppo_boundary(prepared_drain)
    assert drained_physical._physical_global_drain_poisoned


def test_selected_reset_never_erases_a_preexisting_undrained_fault():
    physical, _scene, r06, r05, reset, prepared_r05 = _owners("cpu")
    physical._device_fault[0, 0] = True
    _, _, _, armed_r06, armed_physical = _stage_and_arm(
        physical, r06, prepared_r05
    )
    park = physical.commit_prevalidated_selected_true_reset(armed_physical)
    r06_commit = r06.commit_prevalidated_selected_reset(armed_r06, park)
    physical.acknowledge_r06_selected_reset_commit(park, r06_commit)
    reset.children_committed = True
    receipt = r05.commit_true_reset_many(prepared_r05)
    completion = physical.complete_selected_true_reset_after_r05(
        park, r06_commit, receipt
    )
    assert physical._device_fault[0, 0]
    physical.consume_owned_selected_reset_completion(completion)
    drained_physical, drain_owner = _real_physical_drain("cpu")
    drained_physical._device_fault.copy_(physical._device_fault)
    drain_owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    values = dict(
        zip(
            D.PHYSICAL_GLOBAL_DRAIN_FIELD_NAMES,
            _active_physical_drain_values(drained_physical),
        )
    )
    assert values["fault_count"] == 1


def test_fault_raised_after_finalize_is_joined_before_physical_commit():
    physical, _scene, r06, r05, reset, prepared_r05 = _owners("cpu")
    prepared_r06 = r06.prepare_selected_reset(prepared_r05)
    staged = physical.stage_selected_true_reset(prepared_r06)
    finalized = physical.finalize_selected_true_reset(staged)
    # Another device-side witness may surface after allocation but before the
    # no-fail arm.  Stale detection must retain it, never overwrite it with
    # the earlier after-image.
    physical._device_fault[0, 0] = True
    armed_r06 = r06.arm_prevalidated_selected_reset(
        prepared_r06, finalized
    )
    armed_physical = physical.prearm_selected_true_reset(
        finalized, armed_r06
    )
    park = physical.commit_prevalidated_selected_true_reset(armed_physical)
    r06_commit = r06.commit_prevalidated_selected_reset(armed_r06, park)
    physical.acknowledge_r06_selected_reset_commit(park, r06_commit)
    reset.children_committed = True
    receipt = r05.commit_true_reset_many(prepared_r05)
    completion = physical.complete_selected_true_reset_after_r05(
        park, r06_commit, receipt
    )
    assert physical._device_fault[0, 0]
    physical.consume_owned_selected_reset_completion(completion)


def test_selected_reset_capabilities_are_empty_exact_identities():
    physical, _scene, r06, r05, reset, prepared_r05 = _owners("cpu")
    prepared_r06 = r06.prepare_selected_reset(prepared_r05)
    staged = physical.stage_selected_true_reset(prepared_r06)
    finalized = physical.finalize_selected_true_reset(staged)
    armed_r06 = r06.arm_prevalidated_selected_reset(
        prepared_r06, finalized
    )
    armed = physical.prearm_selected_true_reset(finalized, armed_r06)
    park = physical.commit_prevalidated_selected_true_reset(armed)
    for capability in (staged, finalized, armed, park):
        assert not hasattr(capability, "__dict__")
        assert type(capability).__slots__ == ()
        with pytest.raises(AttributeError):
            capability.device_mask = torch.ones(2, dtype=torch.bool)
        with pytest.raises(TypeError, match="owner-issued"):
            type(capability)()
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(capability)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(capability)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(capability)
    with pytest.raises(TypeError):
        replace(park)
    forged = object.__new__(D.PhysicalSelectedTrueResetParkCommitToken)
    with pytest.raises(D.PhysicalFlightDeviceError, match="stale or foreign"):
        physical.require_owned_selected_reset_commit(forged)

    r06_commit = r06.commit_prevalidated_selected_reset(armed_r06, park)
    physical.acknowledge_r06_selected_reset_commit(park, r06_commit)
    reset.children_committed = True
    receipt = r05.commit_true_reset_many(prepared_r05)
    completion = physical.complete_selected_true_reset_after_r05(
        park, r06_commit, receipt
    )
    assert not hasattr(completion, "__dict__")
    assert type(completion).__slots__ == ()
    with pytest.raises(AttributeError):
        completion.success = True
    with pytest.raises(TypeError, match="owner-issued"):
        type(completion)()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(completion)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(completion)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(completion)
    with pytest.raises(TypeError):
        replace(completion)
    forged_completion = object.__new__(
        D.PhysicalSelectedTrueResetCompletionToken
    )
    with pytest.raises(D.PhysicalFlightDeviceError, match="stale or foreign"):
        physical.require_owned_selected_reset_completion(forged_completion)


def _real_physical_drain(runtime_device="cpu"):
    helper = _global_drain_helper()
    drain = _global_drain_module()
    physical, _scene, _r06, _r05, _reset, _prepared = _owners(
        runtime_device
    )
    drain_device = physical.device
    schemas = list(drain.DEFAULT_LEAF_SCHEMAS)
    physical_schema = D.materialize_physical_ppo_drain_leaf_schema(
        leaf_schema_type=drain.LeafDrainSchema,
        field_spec_type=drain.DeviceDrainFieldSpec,
    )
    schemas[drain.OWNER_ORDER.index("physical_ball")] = physical_schema
    leaves = {
        schema.owner_kind: helper.FakeLeaf(
            schema.owner_kind,
            num_envs=2,
            schema=schema,
            device=drain_device,
            total=0,
        )
        for schema in schemas
    }
    leaves["physical_ball"] = physical
    owner = drain.ActionBallFullMdpPpoDrainOwner(
        num_envs=2,
        device=drain_device,
        leaves=leaves,
        leaf_schemas=tuple(schemas),
        diagnostic_allow_minimal_schemas=True,
    )
    owner.require_exact_leaf_bindings(
        {name: leaves[name] for name in drain.OWNER_ORDER}
    )
    return physical, owner


def _active_physical_drain_values(physical):
    active = physical._active_physical_global_drain
    assert active is not None
    authority = active.authority
    registry = getattr(authority, "_LeafDevicePackAuthority__registry")
    state = registry[id(active.pack)]
    assert state.pack is active.pack
    return tuple(state.values.tolist())


def test_physical_global_drain_pack_exact_ack_abort_and_poison():
    physical, owner = _real_physical_drain("cpu")
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    with pytest.raises(D.PhysicalFlightDeviceError, match="active global PPO drain"):
        physical.bind_r06_owner(physical._r06_owner)
    owner.abort_pre_optimizer_ppo_boundary(prepared)

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)
    assert physical._physical_global_drain_sequence == 1
    assert (
        physical._physical_global_drain_last_acknowledged_mutation_version
        == physical._mutation_version
    )
    assert not physical._physical_checkpoint_requires_global_drain_ack
    physical._require_globally_acknowledged_checkpoint_frontier()

    physical._advance_owner_mutation_version()
    assert physical._physical_checkpoint_requires_global_drain_ack
    with pytest.raises(D.PhysicalFlightDeviceError, match="globally ACKed"):
        physical._require_globally_acknowledged_checkpoint_frontier()

    physical.poison_pre_optimizer_ppo_boundary(reason="injected post-D2H failure")
    physical.poison_pre_optimizer_ppo_boundary(reason="ignored replacement")
    assert physical._physical_global_drain_poison_reason == "injected post-D2H failure"
    with pytest.raises(D.PhysicalFlightOwnerPoisonedError):
        physical.scene_snapshot()


def test_physical_global_drain_source_and_ack_method_are_exactly_pinned():
    drain = _global_drain_module()
    assert hashlib.sha256(Path(drain.__file__).read_bytes()).hexdigest() == (
        D.PHYSICAL_GLOBAL_DRAIN_SOURCE_SHA256
    )
    assert D._global_drain_ack_authority_api_sha256(drain) == (
        D.PHYSICAL_GLOBAL_DRAIN_ACK_AUTHORITY_API_SHA256
    )


def test_physical_global_drain_rejects_foreign_exact_coordinator_ack():
    physical, owner = _real_physical_drain("cpu")
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    physical_row = next(
        row for row in receipt.owner_rows if row.owner_kind == "physical_ball"
    )
    _foreign_physical, foreign_owner = _real_physical_drain("cpu")
    foreign_prepared = foreign_owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    foreign_receipt = foreign_owner.transfer_decode_pre_optimizer_ppo_boundary(
        foreign_prepared
    )
    foreign_owner.mark_optimizer_returned(foreign_receipt)
    active = physical._active_physical_global_drain
    assert active is not None
    with pytest.raises(Exception, match="foreign|stale|window|lane"):
        physical.acknowledge_pre_optimizer_ppo_boundary(
            pack=active.pack,
            receipt=foreign_receipt,
            owner_row=physical_row,
        )
    assert physical._physical_global_drain_poisoned


def test_bound_owner_tombstones_legacy_host_selected_reset():
    physical, _scene, _r06, _r05, _reset, _prepared = _owners("cpu")
    with pytest.raises(D.PhysicalFlightDeviceError, match="diagnostic-only"):
        physical.true_reset_many(
            selected_env_ids=(0,),
            prior_reset_generations=(1,),
            zero_open_all_owner_closure=object(),
            expected_zero_open_all_owner_closure_sha256="0" * 64,
        )
