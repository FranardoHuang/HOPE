from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import os
from pathlib import Path
import struct
import sys
import types

import pytest
import torch


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import action_ball_continuous_recovery_device as device_recovery  # noqa: E402
import action_ball_continuous_recovery_runtime as recovery  # noqa: E402
import action_ball_landing_outcome_mailbox as mailbox  # noqa: E402

_DRAIN_SOURCE = (
    _WBT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
if str(_DRAIN_SOURCE) not in sys.path:
    sys.path.insert(0, str(_DRAIN_SOURCE))

_DRAIN_PACKAGE = "whole_body_tracking.tasks.tracking.mdp"
_DRAIN_FLAT_NAME = "action_ball_full_mdp_ppo_drain"
_DRAIN_CANONICAL_NAME = f"{_DRAIN_PACKAGE}.{_DRAIN_FLAT_NAME}"
global_drain = sys.modules.get(_DRAIN_CANONICAL_NAME)
if global_drain is None:
    import action_ball_full_mdp_ppo_drain as global_drain  # noqa: E402

    if _DRAIN_PACKAGE in sys.modules:
        sys.modules.setdefault(_DRAIN_CANONICAL_NAME, global_drain)
sys.modules[_DRAIN_FLAT_NAME] = global_drain
if _DRAIN_PACKAGE in sys.modules:
    setattr(
        sys.modules[_DRAIN_PACKAGE],
        _DRAIN_FLAT_NAME,
        global_drain,
    )


_DEVICE = torch.device(
    os.environ.get("ACTION_BALL_RECOVERY_DEVICE_TEST_DEVICE", "cpu")
)
_DTYPE = torch.float64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _profile_kwargs() -> dict[str, object]:
    return {
        "continuous_contract_authority_sha256": _sha("continuous-contract"),
        "recovery_contract_authority_sha256": _sha("recovery-contract"),
        "transaction_contract_authority_sha256": _sha("transaction-contract"),
        "source_sha256": _sha("source"),
        "config_sha256": _sha("config"),
        "plant_fact_schema_sha256": _sha("plant-fact-schema"),
        "ordered_joint_names": ("hip", "knee", "shoulder", "elbow"),
        "ordered_body_names": ("pelvis", "torso", "racket"),
        "ordered_foot_names": ("left_foot", "right_foot"),
        "position_frame": "hope_world_table_xyz_m",
        "orientation_frame": "hope_world_body_quaternion",
        "quaternion_order": "wxyz",
        "reference_semantics": recovery.REFERENCE_KIND,
        "station_anchor_semantics": recovery.STATION_ANCHOR_KIND,
        "support_signal_semantics": recovery.SUPPORT_SIGNAL_KIND,
        "policy_rate_hz": 50,
        "recovery_start_age_tick": 10,
        "recovery_end_age_tick": 77,
        "component_weights": {
            name: 1.0 for name in recovery.COMPONENT_NAMES
        },
        "component_scales": {
            name: 1.0 for name in recovery.COMPONENT_NAMES
        },
        "component_reductions": dict(recovery.REQUIRED_COMPONENT_REDUCTIONS),
        "ready_tolerances": {
            name: (0.0 if name == "foot_support_deficit" else 0.1)
            for name in recovery.COMPONENT_NAMES
        },
        "support_contact_threshold": 1.0,
        "minimum_supported_feet": 2,
        "ready_dwell_ticks": 2,
        "reward_weight": 0.7,
    }


def _profile(**updates: object) -> recovery.ContinuousRecoveryProfile:
    values = _profile_kwargs()
    values.update(updates)
    return recovery.ContinuousRecoveryProfile(**values)


def _owner(
    *,
    num_envs: int = 1,
    profile: recovery.ContinuousRecoveryProfile | None = None,
) -> device_recovery.ContinuousRecoveryDeviceCoordinator:
    return device_recovery.ContinuousRecoveryDeviceCoordinator(
        profile=profile or _profile(),
        num_envs=num_envs,
        device=_DEVICE,
        dtype=_DTYPE,
    )


class _ZeroGlobalDrainLeaf:
    def __init__(self, owner_kind: str, device: torch.device) -> None:
        self.owner_kind = owner_kind
        self.device = device
        self._active_pack = None
        self.poisoned = False

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self, *, authority, update_index, completed_environment_steps
    ):
        values = torch.zeros(
            authority.expected_width,
            dtype=torch.int64,
            device=self.device,
        )
        pack = authority.mint_device_pack(leaf=self, values=values)
        self._active_pack = pack
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack):
        assert pack is self._active_pack
        self._active_pack = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self, *, pack, receipt, owner_row
    ):
        assert pack is self._active_pack
        self._active_pack = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason):
        self.poisoned = True


def _global_drain_owner(
    r07: device_recovery.ContinuousRecoveryDeviceCoordinator,
    *,
    initial_update_index: int = 0,
) -> global_drain.ActionBallFullMdpPpoDrainOwner:
    r07_schema = device_recovery.materialize_r07_ppo_drain_leaf_schema(
        leaf_schema_type=global_drain.LeafDrainSchema,
        field_spec_type=global_drain.DeviceDrainFieldSpec,
    )
    schemas = tuple(
        (
            r07_schema
            if schema.owner_kind == "r07_recovery"
            else schema
        )
        for schema in global_drain.DEFAULT_LEAF_SCHEMAS
    )
    leaves = {
        owner_kind: _ZeroGlobalDrainLeaf(owner_kind, r07.device)
        for owner_kind in global_drain.OWNER_ORDER
    }
    leaves["r07_recovery"] = r07
    drain = global_drain.ActionBallFullMdpPpoDrainOwner(
        num_envs=r07.num_envs,
        device=r07.device,
        leaves=leaves,
        leaf_schemas=schemas,
        initial_update_index=initial_update_index,
    )
    drain.require_exact_leaf_bindings(
        {owner_kind: leaves[owner_kind] for owner_kind in global_drain.OWNER_ORDER}
    )
    return drain


def test_r07_exports_dependency_neutral_exact_global_drain_schema():
    assert type(device_recovery.R07_PPO_DRAIN_LEAF_SCHEMA) is tuple
    owner_kind, fields = device_recovery.R07_PPO_DRAIN_LEAF_SCHEMA
    assert owner_kind == "r07_recovery"
    assert type(fields) is tuple
    assert tuple(name for name, _cardinality, _minimum in fields) == (
        device_recovery.R07_GLOBAL_DRAIN_FIELD_NAMES
    )
    schema = device_recovery.materialize_r07_ppo_drain_leaf_schema(
        leaf_schema_type=global_drain.LeafDrainSchema,
        field_spec_type=global_drain.DeviceDrainFieldSpec,
    )
    assert type(schema) is global_drain.LeafDrainSchema
    assert schema.owner_kind == owner_kind
    assert tuple(
        (field.name, field.cardinality, field.minimum)
        for field in schema.fields
    ) == fields
    assert schema.width(2) == len(fields) + len(
        device_recovery.R07_GLOBAL_DRAIN_PER_ENV_FIELDS
    )


def _key(
    *,
    env_id: int = 0,
    ordinal: int = 0,
    **updates: object,
) -> mailbox.LandingOutcomeShotKey:
    values: dict[str, object] = {
        "env_id": env_id,
        "reset_generation": 1,
        "swing_generation": ordinal,
        "action_uid": 7000 + env_id,
        "action_slot": 4,
        "birth_sha256": _sha(f"birth:{env_id}"),
        "sample_sha256": _sha(f"sample:{env_id}:{ordinal}"),
        "task_sha256": _sha(f"task:{env_id}:{ordinal}"),
        "run_id": "r07-device-focused",
        "carry_chain_id": f"carry-env-{env_id}",
        "shot_index": ordinal + 1,
        "source_sha256": _sha("source"),
        "config_sha256": _sha("config"),
        "receipt_content_sha256": _sha(f"receipt:{env_id}:{ordinal}"),
    }
    values.update(updates)
    return mailbox.LandingOutcomeShotKey(**values)


def _reference(
    rows: int,
    *,
    label: str,
    value: float = 0.0,
    profile: recovery.ContinuousRecoveryProfile | None = None,
) -> device_recovery.DeviceContinuousRecoveryReference:
    selected = profile or _profile()
    root_q = torch.zeros((rows, 4), dtype=_DTYPE, device=_DEVICE)
    root_q[:, 0] = 1.0
    body_q = torch.zeros(
        (rows, len(selected.ordered_body_names), 4),
        dtype=_DTYPE,
        device=_DEVICE,
    )
    body_q[:, :, 0] = 1.0
    return device_recovery.DeviceContinuousRecoveryReference(
        reference_sha256=tuple(_sha(f"{label}:{row}") for row in range(rows)),
        root_position_m=torch.full(
            (rows, 3), value, dtype=_DTYPE, device=_DEVICE
        ),
        root_orientation_wxyz=root_q,
        joint_position_rad=torch.full(
            (rows, len(selected.ordered_joint_names)),
            value,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
        body_position_m=torch.full(
            (rows, len(selected.ordered_body_names), 3),
            value,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
        body_orientation_wxyz=body_q,
        station_anchor_xy_m=torch.full(
            (rows, 2), value, dtype=_DTYPE, device=_DEVICE
        ),
    )


def _bind(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    env_ids: tuple[int, ...] | None = None,
) -> None:
    ids = tuple(range(owner.num_envs)) if env_ids is None else env_ids
    owner.bind_sequence_birth(
        ids,
        reset_generations=(1,) * len(ids),
        sequence_origin_ticks=(0,) * len(ids),
        reference=_reference(
            len(ids), label="birth-reference", profile=owner.profile
        ),
    )


def _device_key_digest(
    key: mailbox.LandingOutcomeShotKey,
) -> torch.Tensor:
    return device_recovery.DeviceLandingOutcomeShotKey.from_host_keys(
        (key,), device=_DEVICE
    ).canonical_sha256[0]


def _commit(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    key: mailbox.LandingOutcomeShotKey,
    *,
    reveal_tick: int,
    deadline_tick: int,
    reference_value: float = 1.0,
) -> None:
    owner.commit_reveal(
        (key.env_id,),
        task_keys=(key,),
        scheduled_ordinals=(key.swing_generation,),
        scheduled_reveal_ticks=(reveal_tick,),
        scheduled_deadline_ticks=(deadline_tick,),
        task_frame0_reference=_reference(
            1,
            label=f"task-reference:{key.env_id}:{key.swing_generation}",
            value=reference_value,
            profile=owner.profile,
        ),
    )


def _latch_deadline(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    key: mailbox.LandingOutcomeShotKey,
    *,
    deadline_tick: int,
) -> None:
    owner.latch_deadline_consumed(
        (key.env_id,),
        task_keys=(key,),
        deadline_ticks=(deadline_tick,),
    )


def _command(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    *,
    tick: int,
    source_step: int | None = None,
    phase_code: int = device_recovery.PHASE_RECOVERY_HIDDEN,
    reference_active: bool = True,
    motion_active: bool = False,
    suffix_complete: bool | None = None,
    deadline_due: bool = False,
) -> device_recovery.DeviceContinuousRecoveryCommandProjection:
    n = owner.num_envs
    suffix = (
        owner._suffix_complete.clone()
        if suffix_complete is None
        else torch.full(
            (n,), suffix_complete, dtype=torch.bool, device=owner.device
        )
    )
    return device_recovery.DeviceContinuousRecoveryCommandProjection(
        source_step=torch.full(
            (n,),
            tick if source_step is None else source_step,
            dtype=torch.int64,
            device=owner.device,
        ),
        episode_tick=torch.full(
            (n,), tick, dtype=torch.int64, device=owner.device
        ),
        phase_code=torch.full(
            (n,), phase_code, dtype=torch.int64, device=owner.device
        ),
        reference_active=torch.full(
            (n,), reference_active, dtype=torch.bool, device=owner.device
        ),
        motion_active=torch.full(
            (n,), motion_active, dtype=torch.bool, device=owner.device
        ),
        suffix_complete=suffix,
        deadline_due=torch.full(
            (n,), deadline_due, dtype=torch.bool, device=owner.device
        ),
        scheduled_ordinal=owner._scheduled_ordinal.clone(),
        current_deadline_tick=owner._current_deadline_tick.clone(),
        current_task_key_sha256=owner._current_key_sha.clone(),
    )


def _reconcile(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    *,
    tick: int,
    source_step: int | None = None,
    phase_code: int = device_recovery.PHASE_RECOVERY_HIDDEN,
    reference_active: bool = True,
    motion_active: bool = False,
    suffix_complete: bool | None = None,
    deadline_due: bool = False,
) -> None:
    owner.reconcile_command_projection(
        _command(
            owner,
            tick=tick,
            source_step=source_step,
            phase_code=phase_code,
            reference_active=reference_active,
            motion_active=motion_active,
            suffix_complete=suffix_complete,
            deadline_due=deadline_due,
        )
    )


def _facts(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
) -> device_recovery.DeviceContinuousRecoveryPlantFacts:
    n = owner.num_envs
    return device_recovery.DeviceContinuousRecoveryPlantFacts(
        root_position_m=owner._reference_root_position.clone(),
        root_orientation_wxyz=owner._reference_root_orientation.clone(),
        root_linear_velocity_mps=torch.zeros(
            (n, 3), dtype=owner.dtype, device=owner.device
        ),
        root_angular_velocity_radps=torch.zeros(
            (n, 3), dtype=owner.dtype, device=owner.device
        ),
        joint_position_rad=owner._reference_joint_position.clone(),
        joint_velocity_radps=torch.zeros(
            (n, owner.num_joints), dtype=owner.dtype, device=owner.device
        ),
        body_position_m=owner._reference_body_position.clone(),
        body_orientation_wxyz=owner._reference_body_orientation.clone(),
        body_linear_velocity_mps=torch.zeros(
            (n, owner.num_bodies, 3),
            dtype=owner.dtype,
            device=owner.device,
        ),
        body_angular_velocity_radps=torch.zeros(
            (n, owner.num_bodies, 3),
            dtype=owner.dtype,
            device=owner.device,
        ),
        station_xy_m=owner._station_anchor_xy.clone(),
        foot_contact_signal=torch.full(
            (n, owner.num_feet),
            float(owner.profile.support_contact_threshold),
            dtype=owner.dtype,
            device=owner.device,
        ),
        foot_slip_velocity_xy_mps=torch.zeros(
            (n, owner.num_feet, 2),
            dtype=owner.dtype,
            device=owner.device,
        ),
        facts_valid=torch.ones(n, dtype=torch.bool, device=owner.device),
        hard_safety_ok=torch.ones(n, dtype=torch.bool, device=owner.device),
    )


def _snapshot_view(
    value: device_recovery.SharedContinuousRecoveryRewardView,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, torch.Tensor):
            result[field.name] = item.detach().clone()
        elif isinstance(
            item, device_recovery.DeviceContinuousRecoveryPaymentIdentity
        ):
            result[field.name] = {
                identity_field.name: (
                    identity_item.detach().clone()
                    if isinstance(identity_item, torch.Tensor)
                    else identity_item
                )
                for identity_field in fields(item)
                for identity_item in (getattr(item, identity_field.name),)
            }
        else:  # pragma: no cover - new public fields must choose a copy contract.
            raise AssertionError(
                f"unsupported reward-view field {field.name}: {type(item)!r}"
            )
    return result


def _publish_and_settle(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    *,
    facts: device_recovery.DeviceContinuousRecoveryPlantFacts | None = None,
) -> tuple[
    dict[str, object],
    device_recovery.DeviceContinuousRecoveryDoneTermProjection,
]:
    done = owner.publish_after_physics(_facts(owner) if facts is None else facts)
    view = owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    snapshot = _snapshot_view(view)
    owner.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER,
        view.weighted_reward.clone(),
    )
    return snapshot, done


class _ExactTopRewardOwner:
    def require_healthy(self):
        return None

    def publish_full_mdp_pre_reward(self, **kwargs):
        raise AssertionError("R07 must not recursively call the top publisher")

    def require_owned_full_mdp_pre_reward(self, *args, **kwargs):
        raise AssertionError("R07 must not recursively call the top validator")

    def close_full_mdp_reward_cycle(self, **kwargs):
        raise AssertionError("R07 must not recursively call the top closer")


def _open_full_mdp_reward_epoch(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    *,
    control_step: int = 0,
    top: _ExactTopRewardOwner | None = None,
):
    authority = top or _ExactTopRewardOwner()
    publication = owner.publish_full_mdp_pre_reward(
        control_step=control_step,
        runtime_owner=authority,
    )
    owned = owner.require_owned_full_mdp_pre_reward(
        publication,
        control_step=control_step,
        runtime_owner=authority,
    )
    return authority, publication, owned


def _settle_complete_recovery_window(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    *,
    deadline_tick: int,
    first_unreconciled_tick: int,
) -> list[int]:
    """Advance contiguously and pay every exact age-10..77 denominator cell."""

    observed_ages: list[int] = []
    for tick in range(first_unreconciled_tick, deadline_tick + 78):
        _reconcile(owner, tick=tick)
        age = tick - deadline_tick
        if device_recovery.RECOVERY_START_AGE_TICK <= age <= (
            device_recovery.RECOVERY_END_AGE_TICK
        ):
            view, _ = _publish_and_settle(owner)
            assert view["recovery_expected"].tolist() == [True]
            observed_ages.append(int(view["recovery_age_tick"][0]))
    assert observed_ages == list(
        range(
            device_recovery.RECOVERY_START_AGE_TICK,
            device_recovery.RECOVERY_END_AGE_TICK + 1,
        )
    )
    return observed_ages


def _fault_count(
    receipt: device_recovery.ContinuousRecoveryBoundaryReceipt,
    name: str,
) -> int:
    return dict(receipt.fault_counts)[name]


def _row_state(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    env_id: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in owner._state_tensors().items():
        if tensor.ndim > 0 and tensor.shape[0] == owner.num_envs:
            result[name] = tensor[env_id].detach().clone()
    return result


def _assert_row_equal(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    env_id: int,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = _row_state(owner, env_id)
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.equal(actual[name], expected[name]), name


def _host_row(
    owner: device_recovery.ContinuousRecoveryDeviceCoordinator,
    env_id: int,
) -> dict[str, object]:
    return {
        name: (rows[env_id] if isinstance(rows, list) else rows)
        for name, rows in owner._host_state().items()
    }


def test_device_first_grid_is_explicit_pre_integration_hold_and_device_portable():
    owner = _owner()
    assert owner.device == _DEVICE
    assert device_recovery.INTEGRATION_STATUS == "PRE_INTEGRATION_HOLD"
    assert device_recovery.RUNTIME_WIRING_CONNECTED is False
    assert device_recovery.CUDA_PROFILED is False
    assert device_recovery.FORMAL_EXACT_RESUME_INTEGRATED is False
    assert device_recovery.LAUNCH_AUTHORIZED is False
    assert device_recovery.POLICY_RATE_HZ == 50
    assert device_recovery.RECOVERY_START_AGE_TICK == 10
    assert device_recovery.RECOVERY_END_AGE_TICK == 77
    assert device_recovery.RECOVERY_SAMPLE_COUNT == 68


def test_device_c05_owner_retains_all_fourteen_fields_losslessly():
    key = _key()
    value = device_recovery.DeviceLandingOutcomeShotKey.from_host_keys(
        (key,), device=_DEVICE
    )
    assert len(fields(mailbox.LandingOutcomeShotKey)) == 14
    for name in (
        "env_id",
        "reset_generation",
        "swing_generation",
        "action_uid",
        "action_slot",
        "shot_index",
    ):
        assert int(getattr(value, name)[0]) == getattr(key, name)
    for name in (
        "birth_sha256",
        "sample_sha256",
        "task_sha256",
        "source_sha256",
        "config_sha256",
        "receipt_content_sha256",
    ):
        assert bytes(getattr(value, name)[0].tolist()).hex() == getattr(key, name)
    assert value.run_id == (key.run_id,)
    assert value.carry_chain_id == (key.carry_chain_id,)
    assert value.host_keys == (key,)
    assert bytes(value.canonical_sha256[0].tolist()).hex() == key.canonical_sha256


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("env_id", 1),
        ("reset_generation", 2),
        ("swing_generation", 1),
        ("action_uid", 7001),
        ("action_slot", 5),
        ("birth_sha256", _sha("mutated-birth")),
        ("sample_sha256", _sha("mutated-sample")),
        ("task_sha256", _sha("mutated-task")),
        ("run_id", "mutated-run"),
        ("carry_chain_id", "mutated-carry"),
        ("shot_index", 2),
        ("source_sha256", _sha("mutated-source")),
        ("config_sha256", _sha("mutated-config")),
        ("receipt_content_sha256", _sha("mutated-receipt")),
    ),
)
def test_each_c05_field_changes_device_canonical_owner_token(
    name: str,
    replacement: object,
):
    original = _key()
    mutated = replace(original, **{name: replacement})
    values = device_recovery.DeviceLandingOutcomeShotKey.from_host_keys(
        (original, mutated), device=_DEVICE
    )
    assert not torch.equal(values.canonical_sha256[0], values.canonical_sha256[1])
    assert bytes(values.canonical_sha256[1].tolist()).hex() == mutated.canonical_sha256


@pytest.mark.parametrize(
    "missing",
    (
        "run_id",
        "carry_chain_id",
        "shot_index",
        "source_sha256",
        "config_sha256",
        "receipt_content_sha256",
    ),
)
def test_legacy_eight_field_or_partial_key_cannot_enter_device_owner(missing: str):
    value = _key().full_key_dict()
    del value[missing]
    with pytest.raises(recovery.ContinuousRecoveryError):
        device_recovery.DeviceLandingOutcomeShotKey.from_host_keys(
            (value,), device=_DEVICE
        )


def test_n3_selected_birth_is_byte_invariant_and_ready_tensor_identity_is_stable():
    owner = _owner(num_envs=3)
    ready = owner.ready_authority
    pointer = ready.data_ptr()
    untouched_device = _row_state(owner, 1)
    untouched_host = _host_row(owner, 1)

    _bind(owner, (0, 2))

    assert owner.ready_authority is ready
    assert owner.ready_authority.data_ptr() == pointer
    _assert_row_equal(owner, 1, untouched_device)
    assert _host_row(owner, 1) == untouched_host
    assert owner._sequence_active.tolist() == [True, False, True]


def test_selected_true_reset_preserves_every_unselected_device_and_host_byte():
    owner = _owner(num_envs=3)
    pointer = owner.ready_authority.data_ptr()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    untouched_device = _row_state(owner, 1)
    untouched_host = _host_row(owner, 1)

    owner.reset_true_boundary((0, 2))

    assert owner.ready_authority.data_ptr() == pointer
    _assert_row_equal(owner, 1, untouched_device)
    assert _host_row(owner, 1) == untouched_host
    assert owner._sequence_active.tolist() == [False, True, False]


def test_true_reset_during_open_payment_epoch_rejects_before_any_write():
    owner = _owner(num_envs=3)
    _bind(owner)
    _reconcile(owner, tick=0)
    owner.publish_after_physics(_facts(owner))
    owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    before_tensors = {
        name: tensor.clone() for name, tensor in owner._state_tensors().items()
    }
    before_host = owner._host_state()

    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.reset_true_boundary((0, 2))

    for name, tensor in owner._state_tensors().items():
        assert torch.equal(tensor, before_tensors[name]), name
    assert owner._host_state() == before_host
    owner.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER,
        owner.shared_reward_view.weighted_reward.clone(),
    )


def test_birth_reference_can_become_ready_but_cannot_create_recovery_reward():
    owner = _owner()
    pointer = owner.ready_authority.data_ptr()
    _bind(owner)

    _reconcile(owner, tick=0)
    first, _ = _publish_and_settle(owner)
    _reconcile(owner, tick=1)
    second, _ = _publish_and_settle(owner)

    assert first["ready_instant"].tolist() == [True]
    assert first["ready_live"].tolist() == [False]
    assert second["ready_live"].tolist() == [True]
    assert owner.ready_authority.tolist() == [True]
    assert owner.ready_authority.data_ptr() == pointer
    assert second["recovery_expected"].tolist() == [False]
    assert second["reward_eligible"].tolist() == [False]
    assert second["weighted_reward"].tolist() == [0.0]
    assert torch.all(second["owner_key_sha256"] == 0)
    assert torch.any(second["reference_owner_sha256"] != 0)


def test_motion_consumes_only_owner_issued_current_epoch_ready_projection():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)

    first = owner.motion_ready_projection()
    view = owner.require_owned_motion_ready_projection(
        first, owner_kind="motion"
    )
    assert type(first) is device_recovery.ContinuousRecoveryMotionReadyProjection
    assert view.ready_projection is first
    assert view.owner_kind == "motion"
    assert view.ready.tolist() == [False]
    assert view.ready_streak.tolist() == [1]
    assert view.required_dwell == 2
    assert view.control_tick.tolist() == [1]
    view.ready.fill_(True)
    view.ready_streak.zero_()
    again = owner.require_owned_motion_ready_projection(
        first, owner_kind="motion"
    )
    assert again.ready.tolist() == [False]
    assert again.ready_streak.tolist() == [1]
    assert again.required_dwell == 2

    _reconcile(owner, tick=1)
    _publish_and_settle(owner)
    second = owner.motion_ready_projection()
    assert second is not first
    second_view = owner.require_owned_motion_ready_projection(
        second, owner_kind="motion"
    )
    assert second_view.ready.tolist() == [True]
    assert second_view.ready_streak.tolist() == [2]
    assert second_view.required_dwell == 2
    assert second_view.control_tick.tolist() == [2]
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="stale or foreign",
    ):
        owner.require_owned_motion_ready_projection(
            first, owner_kind="motion"
        )
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="exact Motion",
    ):
        owner.require_owned_motion_ready_projection(
            second, owner_kind="reward"
        )
    with pytest.raises(TypeError, match="owner-issued"):
        device_recovery.ContinuousRecoveryMotionReadyProjection()


def test_true_reset_revokes_current_motion_ready_projection():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    projection = owner.motion_ready_projection()

    owner.reset_true_boundary((0,))

    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="no real post-physics publication",
    ):
        owner.motion_ready_projection()
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="stale or foreign",
    ):
        owner.require_owned_motion_ready_projection(
            projection, owner_kind="motion"
        )


def test_diagnostic_n2_constructor_holds_before_fixture_profile_or_publication():
    with pytest.raises(
        device_recovery.ContinuousRecoveryConstructionHold,
        match=device_recovery.DIAGNOSTIC_N2_CONSTRUCTION_HOLD_REASON,
    ):
        device_recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner(
            motion_owner=object(),
            action_epoch_owner=object(),
            motion_parent_authority=object(),
            motion_parent_receipt=object(),
        )
    with pytest.raises(
        device_recovery.ContinuousRecoveryConstructionHold,
        match="real Motion owner",
    ):
        device_recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner(
            motion_owner=None,
            action_epoch_owner=object(),
            motion_parent_authority=object(),
            motion_parent_receipt=object(),
        )
    assert device_recovery.DIAGNOSTIC_UNAUTHORIZED is True
    assert device_recovery.RUNTIME_WIRING_CONNECTED is False
    assert device_recovery.LAUNCH_AUTHORIZED is False


def test_action_epoch_readiness_true_writer_publishes_only_monotonic_first_ready():
    owner = _owner()
    key = device_recovery._row_identity.empty_action_epoch_shot_key(
        (1,), device=owner.device
    )
    for offset, field in enumerate(fields(key)):
        getattr(key, field.name).fill_(1 + offset)
    key.reset_generation.fill_(1)
    key.action_slot.fill_(0)

    calls = []

    class Epoch:
        def publish_r07_first_ready(
            self, *, owner, first_ready, shot_key, source_step
        ):
            calls.append(
                (owner, first_ready.clone(), shot_key.clone(), source_step.clone())
            )

    bundle = types.SimpleNamespace(action_epoch_owner=Epoch())
    owner._diagnostic_n2_bundle = bundle

    def facts(step):
        one_i64 = torch.tensor([step], dtype=torch.int64, device=owner.device)
        return device_recovery.R07EpochDirectRewardFacts(
            source_step=one_i64.clone(),
            motion_cadence_tick=one_i64.clone(),
            reset_generation=torch.ones(1, dtype=torch.int64, device=owner.device),
            recovery_age_tick=one_i64.clone(),
            reward_eligible=torch.ones(1, dtype=torch.bool, device=owner.device),
            facts_valid=torch.ones(1, dtype=torch.bool, device=owner.device),
            foot_supported_lr=torch.ones(
                (1, owner.num_feet), dtype=torch.bool, device=owner.device
            ),
            infrastructure_fault=torch.zeros(1, dtype=torch.bool, device=owner.device),
            producer_fault_bits=torch.zeros(1, dtype=torch.int64, device=owner.device),
            component_errors=torch.zeros(
                (1, len(recovery.COMPONENT_NAMES)), dtype=owner.dtype,
                device=owner.device,
            ),
            raw_score=torch.ones(1, dtype=owner.dtype, device=owner.device),
            weighted_reward=torch.ones(1, dtype=owner.dtype, device=owner.device),
            ready_instant=torch.ones(1, dtype=torch.bool, device=owner.device),
            reference_kind=torch.full(
                (1,),
                device_recovery.R07_REFERENCE_COMPLETED_ACTION_FRAME0,
                dtype=torch.int64,
                device=owner.device,
            ),
            reference_action_slot=torch.zeros(1, dtype=torch.int64, device=owner.device),
            reference_action_uid=torch.ones(1, dtype=torch.int64, device=owner.device),
        )

    owner._publish_action_epoch_motion_readiness(
        facts(0), observed_source_step=0, shot_key=key
    )
    owner._publish_action_epoch_motion_readiness(
        facts(1), observed_source_step=1, shot_key=key
    )
    assert len(calls) == 2
    assert calls[0][0] is bundle and calls[0][1].tolist() == [False]
    assert calls[1][0] is bundle and calls[1][1].tolist() == [True]
    assert calls[0][3].tolist() == [-1]
    assert calls[1][3].tolist() == [1]
    assert int(owner._first_ready_total) == 1


def test_commit_reveal_neither_switches_active_ready_reference_nor_clears_streak():
    owner = _owner()
    _bind(owner)
    for tick in (0, 1):
        _reconcile(owner, tick=tick)
        _publish_and_settle(owner)
    ready = owner.ready_authority
    pointer = ready.data_ptr()
    owner_sha = owner._reference_owner_sha.clone()
    reference_sha = owner._reference_sha.clone()
    streak = owner._ready_streak.clone()

    _commit(owner, _key(), reveal_tick=2, deadline_tick=3)

    assert owner.ready_authority is ready
    assert owner.ready_authority.data_ptr() == pointer
    assert owner.ready_authority.tolist() == [True]
    assert torch.equal(owner._reference_owner_sha, owner_sha)
    assert torch.equal(owner._reference_sha, reference_sha)
    assert torch.equal(owner._ready_streak, streak)
    assert owner._pending_reference_valid.tolist() == [True]


def test_unplayed_deadline_rewards_current_key_while_birth_stays_ready_owner():
    owner = _owner()
    _bind(owner)
    key = _key()
    birth_owner = owner._reference_owner_sha.clone()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)

    for tick in range(12):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)
    view, _ = _publish_and_settle(owner)

    assert view["recovery_age_tick"].tolist() == [10]
    assert view["recovery_expected"].tolist() == [True]
    assert view["reward_eligible"].tolist() == [True]
    assert torch.equal(view["owner_key_sha256"][0], _device_key_digest(key))
    assert torch.equal(view["reference_owner_sha256"], birth_owner)
    assert not torch.equal(
        view["owner_key_sha256"], view["reference_owner_sha256"]
    )
    assert owner._pending_reference_valid.tolist() == [False]


def test_unplayed_successor_keeps_previous_played_reference_owner():
    owner = _owner()
    _bind(owner)
    q0 = _key(ordinal=0)
    _commit(owner, q0, reveal_tick=0, deadline_tick=1)
    owner.mark_playback_started((0,), task_keys=(q0,))
    _reconcile(
        owner,
        tick=0,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=1,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, q0, deadline_tick=1)
    _reconcile(owner, tick=2, suffix_complete=True)
    owner.complete_suffix(
        (0,), task_keys=(q0,), completed_at_episode_ticks=(2,)
    )
    q0_owner = owner._reference_owner_sha.clone()
    assert torch.equal(q0_owner[0], _device_key_digest(q0))
    settled_q0_ages = _settle_complete_recovery_window(
        owner,
        deadline_tick=1,
        first_unreconciled_tick=3,
    )
    assert owner._window_expected_count.tolist() == [68]
    assert owner._window_payment_count.tolist() == [68]
    assert owner._window_first_expected_age.tolist() == [10]
    assert owner._window_last_expected_age.tolist() == [77]
    assert owner._window_last_paid_age.tolist() == [77]

    q1 = _key(ordinal=1)
    _commit(owner, q1, reveal_tick=79, deadline_tick=80)
    _reconcile(owner, tick=79)
    _reconcile(owner, tick=80, deadline_due=True)
    _latch_deadline(owner, q1, deadline_tick=80)
    for tick in range(81, 91):
        _reconcile(owner, tick=tick)
    view, _ = _publish_and_settle(owner)

    assert view["recovery_age_tick"].tolist() == [10]
    assert torch.equal(view["owner_key_sha256"][0], _device_key_digest(q1))
    assert torch.equal(view["reference_owner_sha256"], q0_owner)
    assert owner._host_ready_owner_keys == [q0]
    assert settled_q0_ages == list(range(10, 78))


def test_played_full_suffix_requires_exact_full_key_then_promotes_and_clears_streak():
    owner = _owner()
    _bind(owner)
    for tick in (0, 1):
        _reconcile(owner, tick=tick)
        _publish_and_settle(owner)
    key = _key()
    _commit(owner, key, reveal_tick=2, deadline_tick=3, reference_value=2.0)
    owner.mark_playback_started((0,), task_keys=(key,))
    _reconcile(
        owner,
        tick=2,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=3,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, key, deadline_tick=3)
    _reconcile(owner, tick=4, suffix_complete=True)
    before = {name: tensor.clone() for name, tensor in owner._state_tensors().items()}
    wrong = replace(key, receipt_content_sha256=_sha("wrong-receipt"))
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.complete_suffix(
            (0,), task_keys=(wrong,), completed_at_episode_ticks=(4,)
        )
    for name, tensor in owner._state_tensors().items():
        assert torch.equal(tensor, before[name]), name

    owner.complete_suffix(
        (0,), task_keys=(key,), completed_at_episode_ticks=(4,)
    )

    assert torch.equal(owner._reference_owner_sha[0], _device_key_digest(key))
    assert owner._reference_owner_kind.tolist() == [device_recovery.OWNER_COMMITTED_TASK]
    assert owner._suffix_complete.tolist() == [True]
    assert owner._pending_reference_valid.tolist() == [False]
    assert owner._ready_streak.tolist() == [0]
    assert owner.ready_authority.tolist() == [False]
    assert torch.all(owner._reference_root_position == 2.0)

    view, _ = _publish_and_settle(owner)
    assert view["reference_owner_sha256"].tolist() == [
        list(_device_key_digest(key).tolist())
    ]


def test_real_manager_order_commit_motion_deadline_suffix_command_publish_is_accepted():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    owner.mark_playback_started((0,), task_keys=(key,))
    _reconcile(
        owner,
        tick=0,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=1,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, key, deadline_tick=1)
    _reconcile(owner, tick=2, suffix_complete=True)
    owner.complete_suffix(
        (0,), task_keys=(key,), completed_at_episode_ticks=(2,)
    )
    view, done = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)

    assert view["recovery_age_tick"].tolist() == [1]
    assert view["infrastructure_fault"].tolist() == [False]
    payment_identity = view["payment_identity"]
    assert bytes(payment_identity["profile_sha256"][0].tolist()).hex() == (
        owner.profile.canonical_sha256
    )
    assert torch.equal(
        payment_identity["task_key_sha256"][0], _device_key_digest(key)
    )
    assert payment_identity["recovery_age_tick"].tolist() == [1]
    assert payment_identity["source_step"].tolist() == [2]
    assert payment_identity["consumer"] == device_recovery.RECOVERY_REWARD_CONSUMER
    assert receipt.checkpoint_safe is True, {
        name: count for name, count in receipt.fault_counts if count
    }
    assert receipt.played_deadline_total == 1
    assert all(count == 0 for _, count in receipt.fault_counts)
    for field in fields(done):
        assert getattr(done, field.name).tolist() == [False]


def test_deadline_projection_without_exact_latch_faults_before_publish():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    _reconcile(owner, tick=0)
    _reconcile(owner, tick=1, deadline_due=True)

    view, _ = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)

    assert view["recovery_age_tick"].tolist() == [-1]
    assert view["recovery_expected"].tolist() == [False]
    assert view["reward_eligible"].tolist() == [False]
    assert torch.all(view["owner_key_sha256"] == 0)
    assert _fault_count(receipt, "command_binding") == 1
    assert receipt.checkpoint_safe is False


def test_failed_deadline_safe_row_is_sticky_poison_without_device_owner_or_payment():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    _reconcile(owner, tick=0)
    # The exact tick/key are present, but Motion never emitted deadline_due,
    # so the device acknowledgement predicate is deliberately false.
    _reconcile(owner, tick=1, deadline_due=False)

    _latch_deadline(owner, key, deadline_tick=1)

    assert owner._reward_owner_valid.tolist() == [False]
    assert owner._window_owner_valid.tolist() == [False]
    view, _ = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)
    assert view["recovery_expected"].tolist() == [False]
    assert view["reward_eligible"].tolist() == [False]
    assert view["weighted_reward"].tolist() == [0.0]
    assert owner.ready_authority.tolist() == [False]
    assert receipt.reward_payment_total == 0
    assert _fault_count(receipt, "command_binding") == 1
    assert receipt.checkpoint_safe is False
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.checkpoint_state(receipt)


def test_failed_suffix_safe_row_is_sticky_poison_without_device_promotion_or_payment():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    owner.mark_playback_started((0,), task_keys=(key,))
    _reconcile(
        owner,
        tick=0,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=1,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, key, deadline_tick=1)

    # The host-supplied completion tick is in the broad allowed interval, but
    # the device clock has not reconciled tick 2.  No authority may promote.
    owner.complete_suffix(
        (0,), task_keys=(key,), completed_at_episode_ticks=(2,)
    )

    assert owner._suffix_complete.tolist() == [False]
    assert owner._pending_reference_valid.tolist() == [True]
    assert owner._reference_owner_kind.tolist() == [
        device_recovery.OWNER_SEQUENCE_BIRTH
    ]
    view, _ = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)
    assert view["recovery_expected"].tolist() == [False]
    assert view["reward_eligible"].tolist() == [False]
    assert view["weighted_reward"].tolist() == [0.0]
    assert owner.ready_authority.tolist() == [False]
    assert receipt.reward_payment_total == 0
    assert _fault_count(receipt, "suffix_incomplete") == 1
    assert receipt.checkpoint_safe is False
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.checkpoint_state(receipt)


def test_exact_age_window_has_68_expected_eligible_and_paid_samples():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    expected_ages: list[int] = []
    eligible_ages: list[int] = []
    boundary: dict[int, tuple[bool, bool]] = {}

    for tick in range(80):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)
        view, _ = _publish_and_settle(owner)
        age = int(view["recovery_age_tick"][0])
        expected = bool(view["recovery_expected"][0])
        eligible = bool(view["reward_eligible"][0])
        if expected:
            expected_ages.append(age)
        if eligible:
            eligible_ages.append(age)
        if age in (9, 10, 77, 78):
            boundary[age] = (expected, eligible)

    receipt = owner.drain_ppo_ledger(update_index=0)
    row = receipt.recovery_window_rows[0]
    assert expected_ages == list(range(10, 78))
    assert eligible_ages == list(range(10, 78))
    assert boundary == {
        9: (False, False),
        10: (True, True),
        77: (True, True),
        78: (False, False),
    }
    assert receipt.recovery_expected_total == 68
    assert receipt.reward_eligible_total == 68
    assert receipt.reward_payment_total == 68
    assert receipt.reward_income_total > 0.0
    assert receipt.unplayed_deadline_total == 1
    assert row.owner_key_sha256 == key.canonical_sha256
    assert row.expected_count == 68
    assert row.eligible_count == 68
    assert row.payment_count == 68
    assert row.first_expected_age_tick == 10
    assert row.last_expected_age_tick == 77
    assert row.last_paid_age_tick == 77
    assert row.closed_68_of_68 is True
    assert int(owner._eligible_total) <= int(owner._expected_total)
    assert int(owner._payment_total) <= int(owner._expected_total)
    assert int(owner._first_ready_total) <= int(owner._ready_instant_total)
    maximum_income = float(owner._eligible_total) * owner.profile.reward_weight
    assert float(owner._income_total) <= maximum_income + 1.0e-10, (
        float(owner._income_total),
        maximum_income,
    )
    assert receipt.checkpoint_safe is True, {
        "nonzero": {
            name: count for name, count in receipt.fault_counts if count
        },
        "expected": int(owner._expected_total),
        "eligible": int(owner._eligible_total),
        "payment": int(owner._payment_total),
        "income": float(owner._income_total),
        "ready_instant": int(owner._ready_instant_total),
        "first_ready": int(owner._first_ready_total),
    }


def test_global_drain_r07_leaf_materializes_full_window_from_one_transfer():
    owner = _owner(num_envs=2)
    _bind(owner)
    keys = (_key(env_id=0), _key(env_id=1))
    for key in keys:
        _commit(owner, key, reveal_tick=0, deadline_tick=1)
    for tick in range(12):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            for key in keys:
                _latch_deadline(owner, key, deadline_tick=1)
        _publish_and_settle(owner)

    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    assert owner._last_drained_update == -1
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert receipt.device_to_host_transfers == 1
    assert owner._last_receipt is None

    drain.mark_optimizer_returned(receipt)
    drain.acknowledge_post_update(receipt)

    portable = owner._last_receipt
    assert portable is not None
    assert portable.device_to_host_transfers == 1
    assert portable.update_index == 0
    assert portable.drain_sequence == 1
    assert portable.recovery_expected_total == 2
    assert portable.reward_eligible_total == 2
    assert portable.reward_payment_total == 2
    assert portable.reward_income_total > 0.0
    assert tuple(row.owner_key_sha256 for row in portable.recovery_window_rows) == (
        keys[0].canonical_sha256,
        keys[1].canonical_sha256,
    )
    assert tuple(row.expected_count for row in portable.recovery_window_rows) == (
        1,
        1,
    )
    assert tuple(row.first_expected_age_tick for row in portable.recovery_window_rows) == (
        10,
        10,
    )
    assert tuple(row.last_expected_age_tick for row in portable.recovery_window_rows) == (
        10,
        10,
    )
    assert owner._last_drained_update == 0
    assert owner._active_r07_global_drain is None
    assert owner.require_owned_pre_optimizer_ppo_boundary_receipt(receipt) is portable
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="foreign, stale",
    ):
        owner.require_owned_pre_optimizer_ppo_boundary_receipt(object())
    checkpoint = owner.checkpoint_state(portable)
    assert checkpoint["drain_update_index"] == 0

    # A later explicitly invoked legacy diagnostic drain may update the
    # checkpoint candidate, but cannot rebind this global ACK to another
    # portable receipt.
    _reconcile(owner, tick=12)
    _publish_and_settle(owner)
    legacy = owner.drain_ppo_ledger(update_index=1)
    assert legacy is owner._last_receipt
    assert legacy is not portable
    assert owner.require_owned_pre_optimizer_ppo_boundary_receipt(receipt) is portable


def test_global_drain_r07_clean_abort_has_no_business_mutation_and_can_retry():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    drain = _global_drain_owner(owner)
    before_tensors = {
        name: tensor.clone() for name, tensor in owner._state_tensors().items()
    }
    before_host = owner._host_state()
    before_version = owner._mutation_version

    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    drain.abort_pre_optimizer_ppo_boundary(prepared)

    assert owner._mutation_version == before_version
    assert owner._last_drained_update == -1
    assert owner._last_receipt is None
    assert owner._host_state() == before_host
    for name, value in owner._state_tensors().items():
        assert torch.equal(value, before_tensors[name]), name
    retried = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    drain.abort_pre_optimizer_ppo_boundary(retried)


def test_global_drain_r07_blocks_business_mutation_until_ack():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    before_version = owner._mutation_version
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="reward view cannot mutate an active R07 global drain lease",
    ):
        owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    assert owner._mutation_version == before_version
    assert owner._last_drained_update == -1
    assert owner._last_receipt is None

    drain.mark_optimizer_returned(receipt)
    drain.acknowledge_post_update(receipt)
    assert owner._last_drained_update == 0
    assert owner._last_receipt is not None


def test_global_drain_r07_all_business_writers_refuse_active_lease():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    key = _key()
    writers = (
        lambda: owner.bind_sequence_birth(
            (0,),
            reset_generations=(2,),
            sequence_origin_ticks=(1,),
            reference=_reference(1, label="blocked-birth"),
        ),
        lambda: _commit(owner, key, reveal_tick=1, deadline_tick=2),
        lambda: owner.mark_playback_started((0,), task_keys=(key,)),
        lambda: owner.complete_suffix(
            (0,), task_keys=(key,), completed_at_episode_ticks=(1,)
        ),
        lambda: owner.reconcile_command_projection(_command(owner, tick=1)),
        lambda: _latch_deadline(owner, key, deadline_tick=1),
        lambda: owner.publish_after_physics(_facts(owner)),
        lambda: owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER),
        lambda: owner.record_reward_payment(
            device_recovery.RECOVERY_REWARD_CONSUMER,
            torch.zeros(owner.num_envs, dtype=owner.dtype, device=owner.device),
        ),
        lambda: owner.reset_true_boundary((0,)),
    )
    before_version = owner._mutation_version
    before_tensors = {
        name: tensor.clone() for name, tensor in owner._state_tensors().items()
    }
    before_host = owner._host_state()
    for writer in writers:
        with pytest.raises(
            device_recovery.ContinuousRecoveryDeviceError,
            match="active R07 global drain lease",
        ):
            writer()
        assert owner._mutation_version == before_version
        assert owner._host_state() == before_host
        for name, value in owner._state_tensors().items():
            assert torch.equal(value, before_tensors[name]), name

    drain.abort_pre_optimizer_ppo_boundary(prepared)


def test_global_drain_r07_true_reset_is_blocked_while_lease_is_active():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )

    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="active R07 global drain lease",
    ):
        owner.reset_true_boundary((0,))

    drain.abort_pre_optimizer_ppo_boundary(prepared)
    owner.reset_true_boundary((0,))
    assert owner._host_reset_generation == [None]


def test_global_drain_r07_checkpoint_and_legacy_drain_are_mutually_exclusive():
    owner, legacy_receipt = _safe_checkpoint_fixture()
    # Continue from the next update after the legacy diagnostic drain.
    drain = _global_drain_owner(owner, initial_update_index=1)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=1,
        completed_environment_steps=24,
    )

    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="legacy R07 drain cannot overlap",
    ):
        owner.drain_ppo_ledger(update_index=1)
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="checkpoint cannot overlap",
    ):
        owner.checkpoint_state(legacy_receipt)

    drain.abort_pre_optimizer_ppo_boundary(prepared)
    checkpoint = owner.checkpoint_state(legacy_receipt)
    assert checkpoint["drain_update_index"] == 0


def test_global_drain_r07_device_fault_reaches_real_global_consumer():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    # Exercise the real device fault producer, rather than fabricating a host
    # receipt or overriding the global consumer's decoded row.
    owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    assert torch.any(owner._fault_bits != 0)
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )

    with pytest.raises(
        global_drain.ActionBallFullMdpPpoDrainPoisonedError,
        match="device fault",
    ):
        drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert drain.poisoned is True
    assert owner._r07_global_drain_poisoned is True
    assert owner._last_drained_update == -1
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError, match="poisoned"):
        owner.prepare_pre_optimizer_ppo_boundary_device_pack(
            authority=object(),
            update_index=0,
            completed_environment_steps=24,
        )


def test_global_drain_r07_income_float64_bits_roundtrip_and_nan_is_rejected():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    for tick in range(12):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)
        _publish_and_settle(owner)
    expected_income = float(owner._income_total)
    assert expected_income > 0.0
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    row = next(
        value for value in receipt.owner_rows if value.owner_kind == "r07_recovery"
    )
    bits = dict(row.values)["reward_income_total_float64_bits"]
    assert struct.pack("!Q", bits) == struct.pack("!d", expected_income)
    drain.mark_optimizer_returned(receipt)
    drain.acknowledge_post_update(receipt)
    portable = owner.require_owned_pre_optimizer_ppo_boundary_receipt(receipt)
    assert struct.pack("!d", portable.reward_income_total) == struct.pack("!Q", bits)

    poisoned = _owner()
    _bind(poisoned)
    poisoned._income_total.fill_(float("nan"))
    nan_drain = _global_drain_owner(poisoned)
    nan_prepared = nan_drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    with pytest.raises(
        global_drain.ActionBallFullMdpPpoDrainPoisonedError,
        match="invariant failure",
    ):
        nan_drain.transfer_decode_pre_optimizer_ppo_boundary(nan_prepared)
    assert poisoned._r07_global_drain_poisoned is True


def test_global_drain_r07_malformed_owner_row_cannot_retry_after_ack_starts():
    owner = _owner(num_envs=2)
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    r07_row = next(
        row for row in receipt.owner_rows if row.owner_kind == "r07_recovery"
    )
    foreign = global_drain.OwnerDrainRow(
        owner_kind=r07_row.owner_kind, values=r07_row.values
    )
    with pytest.raises(global_drain.ActionBallFullMdpPpoDrainError, match="foreign"):
        owner.acknowledge_pre_optimizer_ppo_boundary(
            pack=owner._active_r07_global_drain.pack,
            receipt=receipt,
            owner_row=foreign,
        )
    assert owner._active_r07_global_drain.stage == "prepared"

    drain.mark_optimizer_returned(receipt)
    drain.acknowledge_post_update(receipt)
    assert owner._active_r07_global_drain is None


def test_global_drain_r07_rejects_same_value_foreign_real_coordinator_receipt():
    owner = _owner(num_envs=2)
    foreign_owner = _owner(num_envs=2)
    for value in (owner, foreign_owner):
        _bind(value)
        _reconcile(value, tick=0)
        _publish_and_settle(value)
    drain = _global_drain_owner(owner)
    foreign_drain = _global_drain_owner(foreign_owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=48
    )
    foreign_prepared = foreign_drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=48
    )
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    foreign_receipt = foreign_drain.transfer_decode_pre_optimizer_ppo_boundary(
        foreign_prepared
    )
    drain.mark_optimizer_returned(receipt)
    foreign_drain.mark_optimizer_returned(foreign_receipt)
    own_row = next(
        row for row in receipt.owner_rows if row.owner_kind == "r07_recovery"
    )
    foreign_row = next(
        row
        for row in foreign_receipt.owner_rows
        if row.owner_kind == "r07_recovery"
    )
    assert own_row.values == foreign_row.values
    active = owner._active_r07_global_drain
    assert active is not None
    with pytest.raises(global_drain.ActionBallFullMdpPpoDrainError, match="foreign"):
        owner.acknowledge_pre_optimizer_ppo_boundary(
            pack=active.pack,
            receipt=foreign_receipt,
            owner_row=foreign_row,
        )


def test_global_drain_r07_post_ack_mutation_requires_new_ack_before_checkpoint():
    owner = _owner(num_envs=2)
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=48
    )
    global_receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    drain.mark_optimizer_returned(global_receipt)
    drain.acknowledge_post_update(global_receipt)
    portable = owner.require_owned_pre_optimizer_ppo_boundary_receipt(
        global_receipt
    )
    assert owner._last_globally_acknowledged_mutation_version == portable.mutation_version
    assert owner._checkpoint_requires_global_drain_ack is False
    owner._advance_mutation_version()
    assert owner._checkpoint_requires_global_drain_ack is True
    with pytest.raises(
        device_recovery.ContinuousRecoveryDeviceError,
        match="mutated after",
    ):
        owner.checkpoint_state(portable)


def test_same_source_and_age_cannot_be_republished_or_paid_twice():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    for tick in range(12):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)

    first, _ = _publish_and_settle(owner)
    assert first["recovery_age_tick"].tolist() == [10]
    assert first["recovery_expected"].tolist() == [True]
    assert first["reward_eligible"].tolist() == [True]
    assert int(owner._window_expected_count[0]) == 1
    assert int(owner._window_payment_count[0]) == 1

    # No command reconciliation occurs here: this is the exact same immutable
    # (profile, key, age, source, consumer) payment identity.
    owner.publish_after_physics(_facts(owner))
    replay_view = owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    replay = _snapshot_view(replay_view)
    assert torch.equal(
        replay["payment_identity"]["task_key_sha256"],
        first["payment_identity"]["task_key_sha256"],
    )
    assert torch.equal(
        replay["payment_identity"]["recovery_age_tick"],
        first["payment_identity"]["recovery_age_tick"],
    )
    assert torch.equal(
        replay["payment_identity"]["source_step"],
        first["payment_identity"]["source_step"],
    )
    owner.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER,
        replay_view.weighted_reward.clone(),
    )

    receipt = owner.drain_ppo_ledger(update_index=0)
    assert receipt.recovery_expected_total == 1
    assert receipt.reward_payment_total == 1
    assert int(owner._window_expected_count[0]) == 1
    assert int(owner._window_payment_count[0]) == 1
    assert _fault_count(receipt, "publish_collision") == 1
    assert _fault_count(receipt, "ledger_sequence") == 1
    assert receipt.checkpoint_safe is False
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.checkpoint_state(receipt)


def test_skipped_age10_through_76_cannot_close_at_77_or_reveal_safe_successor():
    owner = _owner()
    _bind(owner)
    q0 = _key(ordinal=0)
    _commit(owner, q0, reveal_tick=0, deadline_tick=1)
    for tick in range(79):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, q0, deadline_tick=1)

    late, _ = _publish_and_settle(owner)
    assert late["recovery_age_tick"].tolist() == [77]
    assert late["recovery_expected"].tolist() == [True]
    assert late["reward_eligible"].tolist() == [False]
    assert late["infrastructure_fault"].tolist() == [True]
    assert int(owner._window_expected_count[0]) == 0
    assert int(owner._window_payment_count[0]) == 0

    q1 = _key(ordinal=1)
    _commit(owner, q1, reveal_tick=79, deadline_tick=80)
    receipt = owner.drain_ppo_ledger(update_index=0)
    row = receipt.recovery_window_rows[0]
    assert row.owner_key_sha256 == q0.canonical_sha256
    assert row.expected_count == 0
    assert row.payment_count == 0
    assert row.closed_68_of_68 is False
    assert receipt.recovery_expected_total == 0
    assert receipt.reward_payment_total == 0
    assert _fault_count(receipt, "ledger_sequence") == 1
    assert receipt.checkpoint_safe is False
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.checkpoint_state(receipt)


def test_expected_producer_absent_zero_counts_denominator_and_is_faulted():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    for tick in range(12):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)
    facts = _facts(owner)
    facts = replace(facts, facts_valid=torch.zeros_like(facts.facts_valid))

    view, _ = _publish_and_settle(owner, facts=facts)
    receipt = owner.drain_ppo_ledger(update_index=0)
    row = receipt.recovery_window_rows[0]
    assert view["recovery_age_tick"].tolist() == [10]
    assert view["recovery_expected"].tolist() == [True]
    assert view["reward_eligible"].tolist() == [False]
    assert view["weighted_reward"].tolist() == [0.0]
    assert receipt.recovery_expected_total == 1
    assert receipt.reward_eligible_total == 0
    assert receipt.reward_payment_total == 1
    assert receipt.reward_income_total == 0.0
    assert row.owner_key_sha256 == key.canonical_sha256
    assert row.expected_count == 1
    assert row.eligible_count == 0
    assert row.payment_count == 1
    assert row.first_expected_age_tick == 10
    assert row.last_expected_age_tick == 10
    assert row.last_paid_age_tick == 10
    assert row.closed_68_of_68 is False
    assert view["infrastructure_fault"].tolist() == [True]
    assert _fault_count(receipt, "invalid_plant_fact") == 1
    assert receipt.checkpoint_safe is False


def test_played_age11_without_suffix_is_sticky_fault_but_still_expected():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    owner.mark_playback_started((0,), task_keys=(key,))
    for tick in range(12):
        _reconcile(
            owner,
            tick=tick,
            phase_code=(
                device_recovery.PHASE_ACTIVE_OPPORTUNITY
                if tick == 0
                else device_recovery.PHASE_RECOVERY_HIDDEN
            ),
            reference_active=(tick != 0),
            motion_active=(tick == 0),
            deadline_due=(tick == 1),
        )
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)
    at_age10, _ = _publish_and_settle(owner)
    _reconcile(owner, tick=12)
    at_age11, _ = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)
    assert at_age10["recovery_age_tick"].tolist() == [10]
    assert at_age11["recovery_age_tick"].tolist() == [11]
    for view in (at_age10, at_age11):
        assert view["recovery_expected"].tolist() == [True]
        assert view["reward_eligible"].tolist() == [False]
        assert view["infrastructure_fault"].tolist() == [True]
        assert view["weighted_reward"].tolist() == [0.0]
    assert receipt.recovery_expected_total == 2
    assert receipt.reward_eligible_total == 0
    assert receipt.reward_payment_total == 2
    assert _fault_count(receipt, "suffix_incomplete") == 1
    assert receipt.checkpoint_safe is False


@pytest.mark.parametrize("failed", ("component", "support", "safety"))
def test_hard_ready_and_cannot_be_compensated_while_additive_score_stays_positive(
    failed: str,
):
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    for tick in range(12):
        _reconcile(owner, tick=tick, deadline_due=(tick == 1))
        if tick == 1:
            _latch_deadline(owner, key, deadline_tick=1)
    facts = _facts(owner)
    if failed == "component":
        changed = facts.root_position_m.clone()
        changed[0, 0] += 1.0
        facts = replace(facts, root_position_m=changed)
    elif failed == "support":
        contacts = facts.foot_contact_signal.clone()
        contacts[0, 0] = 0.0
        facts = replace(facts, foot_contact_signal=contacts)
    else:
        hard_safety = facts.hard_safety_ok.clone()
        hard_safety[0] = False
        facts = replace(facts, hard_safety_ok=hard_safety)

    view, _ = _publish_and_settle(owner, facts=facts)

    assert view["recovery_expected"].tolist() == [True]
    assert view["reward_eligible"].tolist() == [True]
    assert view["ready_instant"].tolist() == [False]
    assert view["ready_live"].tolist() == [False]
    assert 0.0 < float(view["raw_score"][0]) <= 1.0
    assert float(view["weighted_reward"][0]) > 0.0


@pytest.mark.parametrize("bad", ("nan_position", "zero_root_quat", "zero_body_quat"))
def test_nonfinite_or_zero_quaternion_fails_closed_and_never_requests_done(
    bad: str,
):
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    facts = _facts(owner)
    if bad == "nan_position":
        value = facts.root_position_m.clone()
        value[0, 0] = float("nan")
        facts = replace(facts, root_position_m=value)
    elif bad == "zero_root_quat":
        facts = replace(
            facts,
            root_orientation_wxyz=torch.zeros_like(
                facts.root_orientation_wxyz
            ),
        )
    else:
        facts = replace(
            facts,
            body_orientation_wxyz=torch.zeros_like(
                facts.body_orientation_wxyz
            ),
        )

    view, done = _publish_and_settle(owner, facts=facts)
    receipt = owner.drain_ppo_ledger(update_index=0)

    assert view["facts_valid"].tolist() == [False]
    assert view["reward_eligible"].tolist() == [False]
    assert view["ready_instant"].tolist() == [False]
    assert view["weighted_reward"].tolist() == [0.0]
    assert _fault_count(receipt, "invalid_plant_fact") == 1
    for field in fields(done):
        assert getattr(done, field.name).tolist() == [False]


def test_public_reference_and_reward_views_have_no_future_question_material():
    reference_names = {field.name for field in fields(
        device_recovery.DeviceContinuousRecoveryReference
    )}
    reward_names = {field.name for field in fields(
        device_recovery.SharedContinuousRecoveryRewardView
    )}
    for names in (reference_names, reward_names):
        assert not any("target" in name for name in names)
        assert not any("future" in name for name in names)
        assert not any("next" in name for name in names)
    assert not any("deadline" in name for name in reward_names)
    kwargs = {
        field.name: getattr(_reference(1, label="public"), field.name)
        for field in fields(device_recovery.DeviceContinuousRecoveryReference)
    }
    kwargs["target_xy_m"] = torch.zeros((1, 2), dtype=_DTYPE)
    with pytest.raises(TypeError):
        device_recovery.DeviceContinuousRecoveryReference(**kwargs)


def test_reward_view_and_payment_are_exactly_once_and_one_ulp_is_a_mismatch():
    duplicate_view = _owner()
    _bind(duplicate_view)
    _reconcile(duplicate_view, tick=0)
    duplicate_view.publish_after_physics(_facts(duplicate_view))
    first = duplicate_view.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    assert duplicate_view.reward_view(
        device_recovery.RECOVERY_REWARD_CONSUMER
    ) is first
    duplicate_view.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER,
        first.weighted_reward.clone(),
    )
    receipt = duplicate_view.drain_ppo_ledger(update_index=0)
    assert _fault_count(receipt, "duplicate_view") == 1

    before_view = _owner()
    _bind(before_view)
    _reconcile(before_view, tick=0)
    before_view.publish_after_physics(_facts(before_view))
    before_view.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER,
        before_view.shared_reward_view.weighted_reward.clone(),
    )
    receipt = before_view.drain_ppo_ledger(update_index=0)
    assert _fault_count(receipt, "payment_before_view") == 1

    duplicate_payment = _owner()
    _bind(duplicate_payment)
    _reconcile(duplicate_payment, tick=0)
    duplicate_payment.publish_after_physics(_facts(duplicate_payment))
    view = duplicate_payment.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    payment = view.weighted_reward.clone()
    duplicate_payment.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER, payment
    )
    duplicate_payment.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER, payment
    )
    receipt = duplicate_payment.drain_ppo_ledger(update_index=0)
    assert _fault_count(receipt, "duplicate_payment") == 1

    ulp = _owner()
    _bind(ulp)
    _reconcile(ulp, tick=0)
    ulp.publish_after_physics(_facts(ulp))
    view = ulp.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    wrong = torch.nextafter(
        view.weighted_reward,
        torch.full_like(view.weighted_reward, float("inf")),
    )
    ulp.record_reward_payment(device_recovery.RECOVERY_REWARD_CONSUMER, wrong)
    receipt = ulp.drain_ppo_ledger(update_index=0)
    assert _fault_count(receipt, "payment_mismatch") == 1


def test_r07_full_mdp_reward_epoch_requires_real_plant_identity_and_exact_close():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    owner.publish_after_physics(_facts(owner))
    top, publication, owned = _open_full_mdp_reward_epoch(owner)
    assert owned.terminated.tolist() == [False]
    assert owned.time_out.tolist() == [False]
    assert owner.full_mdp_reward_consumers == (
        device_recovery.RECOVERY_REWARD_CONSUMER,
    )

    view = owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    verdict = owner.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER,
        view.weighted_reward,
    )
    assert type(verdict) is (
        device_recovery.ContinuousRecoveryFullMdpRewardPaymentVerdict
    )
    assert owner.require_owned_full_mdp_reward_payment(
        verdict,
        consumer=device_recovery.RECOVERY_REWARD_CONSUMER,
        control_step=0,
        runtime_owner=top,
    ) is verdict
    close = owner.close_full_mdp_reward_cycle(
        control_step=0,
        pre_reward_publication=publication,
        ordered_consumers=(device_recovery.RECOVERY_REWARD_CONSUMER,),
        ordered_payment_verdicts=(verdict,),
        runtime_owner=top,
    )
    assert owner.require_owned_full_mdp_reward_close(
        close,
        control_step=0,
        runtime_owner=top,
    ) is close


def test_r07_full_mdp_epoch_blocks_reset_checkpoint_and_global_drain_until_close():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    owner.publish_after_physics(_facts(owner))
    top, publication, _ = _open_full_mdp_reward_epoch(owner)
    drain = _global_drain_owner(owner)

    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError, match="open R07"):
        owner.reset_true_boundary((0,))
    with pytest.raises(global_drain.ActionBallFullMdpPpoDrainPrepareError):
        drain.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=1,
        )
    fake_receipt = object()
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError, match="open R07"):
        owner.checkpoint_state(fake_receipt)

    assert owner._r07_global_drain_poisoned is True


def test_r07_full_mdp_close_then_allows_fresh_global_drain_graph():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    owner.publish_after_physics(_facts(owner))
    top, publication, _ = _open_full_mdp_reward_epoch(owner)
    view = owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    verdict = owner.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER, view.weighted_reward
    )
    owner.close_full_mdp_reward_cycle(
        control_step=0,
        pre_reward_publication=publication,
        ordered_consumers=(device_recovery.RECOVERY_REWARD_CONSUMER,),
        ordered_payment_verdicts=(verdict,),
        runtime_owner=top,
    )
    # A leaf that rejects before minting an abort capability correctly poisons
    # that attempted global coordinator; a fresh exact graph is required.
    drain = _global_drain_owner(owner)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=1,
    )
    drain.abort_pre_optimizer_ppo_boundary(prepared)


def test_r07_full_mdp_rejects_wrong_step_foreign_and_self_authored_payment():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    owner.publish_after_physics(_facts(owner))
    top, publication, _ = _open_full_mdp_reward_epoch(owner)
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError, match="wrong-step"):
        owner.require_owned_full_mdp_pre_reward(
            publication,
            control_step=1,
            runtime_owner=top,
        )
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError, match="foreign"):
        owner.require_owned_full_mdp_reward_payment(
            True,
            consumer=device_recovery.RECOVERY_REWARD_CONSUMER,
            control_step=0,
            runtime_owner=top,
        )
    with pytest.raises(TypeError, match="owner-issued"):
        device_recovery.ContinuousRecoveryFullMdpRewardPublication()
    with pytest.raises(TypeError, match="owner-issued"):
        device_recovery.ContinuousRecoveryFullMdpRewardPaymentVerdict()

    wrong_step = _owner()
    _bind(wrong_step)
    _reconcile(wrong_step, tick=0, source_step=0)
    wrong_step.publish_after_physics(_facts(wrong_step))
    wrong_top = _ExactTopRewardOwner()
    wrong_publication = wrong_step.publish_full_mdp_pre_reward(
        control_step=1,
        runtime_owner=wrong_top,
    )
    wrong_owned = wrong_step.require_owned_full_mdp_pre_reward(
        wrong_publication,
        control_step=1,
        runtime_owner=wrong_top,
    )
    assert wrong_owned.terminated.tolist() == [False]
    wrong_view = wrong_step.reward_view(
        device_recovery.RECOVERY_REWARD_CONSUMER
    )
    assert wrong_view.weighted_reward.tolist() == [0.0]


def test_r07_full_mdp_zero_ineligible_still_issues_owner_payment_verdict():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    facts = _facts(owner)
    facts.facts_valid.zero_()
    owner.publish_after_physics(facts)
    top, publication, _ = _open_full_mdp_reward_epoch(owner)
    view = owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    assert view.weighted_reward.tolist() == [0.0]
    verdict = owner.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER, view.weighted_reward
    )
    assert type(verdict) is (
        device_recovery.ContinuousRecoveryFullMdpRewardPaymentVerdict
    )
    close = owner.close_full_mdp_reward_cycle(
        control_step=0,
        pre_reward_publication=publication,
        ordered_consumers=(device_recovery.RECOVERY_REWARD_CONSUMER,),
        ordered_payment_verdicts=(verdict,),
        runtime_owner=top,
    )
    assert owner.require_owned_full_mdp_reward_close(
        close, control_step=0, runtime_owner=top
    ) is close


def test_missing_plant_producer_is_a_fault_not_an_ordinary_zero_cell():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    facts = _facts(owner)
    facts.facts_valid.zero_()

    view, done = _publish_and_settle(owner, facts=facts)
    receipt = owner.drain_ppo_ledger(update_index=0)

    assert view["facts_valid"].tolist() == [False]
    assert view["weighted_reward"].tolist() == [0.0]
    assert _fault_count(receipt, "invalid_plant_fact") == 1
    for field in fields(done):
        assert getattr(done, field.name).tolist() == [False]


def test_r07_full_mdp_manager_adapter_uses_exact_one_weight_and_owner_verdict():
    reward_path = (
        _WBT_ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
        / "action_ball_full_mdp_rewards.py"
    )
    source = reward_path.read_text()
    assert 'R07_CONSUMER = "common_recovery_reward_v1"' in source
    assert "float(manager_weight) != 1.0" in source
    assert "result = owner.record_reward_payment(R07_CONSUMER, payment)" in source
    assert "owner_payment_result=result" in source


@pytest.mark.parametrize("kind", ("duplicate_source", "skipped_episode_tick"))
def test_duplicate_source_or_skipped_episode_tick_fails_closed(kind: str):
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=0)
    if kind == "duplicate_source":
        projection = _command(owner, tick=1, source_step=0)
    else:
        projection = _command(owner, tick=2, source_step=1)
    owner.reconcile_command_projection(projection)
    view, _ = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)

    assert view["source_step"].tolist() == [0]
    assert view["episode_tick"].tolist() == [0]
    assert view["reward_eligible"].tolist() == [False]
    assert owner.ready_authority.tolist() == [False]
    assert _fault_count(receipt, "step_regression") == 1


def test_source_clock_may_jump_when_episode_clock_remains_contiguous():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0, source_step=10)
    _reconcile(owner, tick=1, source_step=100)
    view, _ = _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)
    assert view["source_step"].tolist() == [100]
    assert view["episode_tick"].tolist() == [1]
    assert _fault_count(receipt, "step_regression") == 0
    assert receipt.checkpoint_safe is True


def _safe_checkpoint_fixture():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    _publish_and_settle(owner)
    receipt = owner.drain_ppo_ledger(update_index=0)
    assert receipt.checkpoint_safe is True
    return owner, receipt


def _clone_checkpoint_tensors(
    checkpoint: dict[str, object],
) -> dict[str, object]:
    result = dict(checkpoint)
    result["state_tensors"] = {
        name: tensor.clone()
        for name, tensor in checkpoint["state_tensors"].items()
    }
    return result


def _rich_checkpoint_fixture() -> tuple[
    device_recovery.ContinuousRecoveryDeviceCoordinator,
    dict[str, object],
]:
    """Checkpoint a state with distinct current, ready, pending and window owners."""

    owner = _owner()
    _bind(owner)
    q0 = _key(ordinal=0)
    _commit(owner, q0, reveal_tick=0, deadline_tick=1)
    owner.mark_playback_started((0,), task_keys=(q0,))
    _reconcile(
        owner,
        tick=0,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=1,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, q0, deadline_tick=1)
    _reconcile(owner, tick=2, suffix_complete=True)
    owner.complete_suffix(
        (0,), task_keys=(q0,), completed_at_episode_ticks=(2,)
    )
    _settle_complete_recovery_window(
        owner,
        deadline_tick=1,
        first_unreconciled_tick=3,
    )

    q1 = _key(ordinal=1)
    _commit(owner, q1, reveal_tick=79, deadline_tick=80)
    owner.mark_playback_started((0,), task_keys=(q1,))
    _reconcile(
        owner,
        tick=79,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=80,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, q1, deadline_tick=80)
    owner.publish_after_physics(_facts(owner))
    owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)

    assert owner._host_keys == [q1]
    assert owner._host_ready_owner_keys == [q0]
    assert owner._host_window_owner_keys == [q1]
    assert owner._host_played == [True]
    assert owner._host_deadline_consumed == [True]
    assert owner._host_reference_sha[0] is not None
    assert owner._host_pending_reference_sha[0] is not None
    assert owner._host_payment_epoch_open is True
    receipt = owner.drain_ppo_ledger(update_index=0)
    assert receipt.checkpoint_safe is True, {
        name: count for name, count in receipt.fault_counts if count
    }
    return owner, owner.checkpoint_state(receipt)


def _reseal_host_only_checkpoint_forge(
    checkpoint: dict[str, object],
    *,
    field: str,
    replacement: object,
) -> tuple[dict[str, object], str]:
    forged = dict(checkpoint)
    forged_host: dict[str, object] = {}
    for name, value in checkpoint["host_state"].items():
        if isinstance(value, list):
            forged_host[name] = [
                dict(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            forged_host[name] = value
    forged_host[field] = replacement
    forged["host_state"] = forged_host
    identity = {
        name: value
        for name, value in forged.items()
        if name not in ("state_tensors", "checkpoint_sha256")
    }
    new_pin = recovery.canonical_sha256(identity)
    forged["checkpoint_sha256"] = new_pin
    assert forged["state_tensors"] is checkpoint["state_tensors"]
    assert forged["tensor_bytes_sha256"] == checkpoint["tensor_bytes_sha256"]
    return forged, new_pin


def test_checkpoint_requires_latest_one_use_receipt_and_external_pin():
    owner, stale = _safe_checkpoint_fixture()
    _reconcile(owner, tick=1)
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.checkpoint_state(stale)
    _publish_and_settle(owner)
    latest = owner.drain_ppo_ledger(update_index=1)
    checkpoint = owner.checkpoint_state(latest)
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        owner.checkpoint_state(latest)
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        device_recovery.ContinuousRecoveryDeviceCoordinator.from_checkpoint(
            profile=owner.profile,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=_sha("foreign-checkpoint-authority"),
            device=_DEVICE,
            dtype=_DTYPE,
        )


def test_checkpoint_tensor_mutation_is_rejected_and_restore_is_unaliased():
    owner, receipt = _safe_checkpoint_fixture()
    checkpoint = owner.checkpoint_state(receipt)
    pinned = checkpoint["checkpoint_sha256"]
    tampered = _clone_checkpoint_tensors(checkpoint)
    tampered["state_tensors"]["ready_authority"][0].logical_not_()
    with pytest.raises(device_recovery.ContinuousRecoveryDeviceError):
        device_recovery.ContinuousRecoveryDeviceCoordinator.from_checkpoint(
            profile=owner.profile,
            checkpoint=tampered,
            expected_checkpoint_sha256=pinned,
            device=_DEVICE,
            dtype=_DTYPE,
        )

    restored = device_recovery.ContinuousRecoveryDeviceCoordinator.from_checkpoint(
        profile=owner.profile,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=pinned,
        device=_DEVICE,
        dtype=_DTYPE,
    )
    for name, tensor in owner._state_tensors().items():
        assert torch.equal(restored._state_tensors()[name], tensor), name
    assert restored._host_state() == owner._host_state()
    frozen_tick = checkpoint["state_tensors"]["episode_tick"].clone()
    restored._episode_tick.add_(1)
    assert torch.equal(checkpoint["state_tensors"]["episode_tick"], frozen_tick)
    restored_tick = restored._episode_tick.clone()
    checkpoint["state_tensors"]["episode_tick"].add_(7)
    assert torch.equal(restored._episode_tick, restored_tick)


def test_checkpoint_rejects_resealed_host_only_cross_authority_forges():
    owner, checkpoint = _rich_checkpoint_fixture()
    host = checkpoint["host_state"]

    def changed_row(field: str, replacement: object) -> list[object]:
        rows = list(host[field])
        rows[0] = replacement
        return rows

    def changed_full_key(field: str, label: str) -> list[object]:
        rows = list(host[field])
        original = mailbox.LandingOutcomeShotKey.from_mapping(rows[0])
        mutated = replace(original, receipt_content_sha256=_sha(label))
        # This remains a syntactically complete and individually valid C05 key;
        # only its cross-authority binding to device tensors is forged.
        assert len(fields(mutated)) == 14
        rows[0] = mutated.to_mapping()
        return rows

    cases: list[tuple[str, str, object]] = [
        (
            "current_full14_key",
            "current_keys",
            changed_full_key("current_keys", "forged-current-key"),
        ),
        (
            "ready_full14_key",
            "ready_owner_keys",
            changed_full_key("ready_owner_keys", "forged-ready-key"),
        ),
        (
            "ready_reference",
            "reference_sha256",
            changed_row("reference_sha256", _sha("forged-ready-reference")),
        ),
        (
            "pending_reference",
            "pending_reference_sha256",
            changed_row(
                "pending_reference_sha256", _sha("forged-pending-reference")
            ),
        ),
        (
            "reset_generation",
            "reset_generation",
            changed_row("reset_generation", 2),
        ),
        (
            "scheduled_ordinal",
            "scheduled_ordinal",
            changed_row("scheduled_ordinal", 2),
        ),
        (
            "deadline",
            "last_deadline_tick",
            changed_row("last_deadline_tick", 81),
        ),
        (
            "reveal",
            "last_reveal_tick",
            changed_row("last_reveal_tick", 78),
        ),
        ("played", "played", changed_row("played", False)),
        (
            "deadline_consumed",
            "deadline_consumed",
            changed_row("deadline_consumed", False),
        ),
        ("payment_epoch", "payment_epoch_open", False),
    ]
    if "window_owner_keys" in host:
        cases.append(
            (
                "window_full14_key",
                "window_owner_keys",
                changed_full_key("window_owner_keys", "forged-window-key"),
            )
        )

    for label, field, replacement in cases:
        forged, new_pin = _reseal_host_only_checkpoint_forge(
            checkpoint,
            field=field,
            replacement=replacement,
        )
        try:
            device_recovery.ContinuousRecoveryDeviceCoordinator.from_checkpoint(
                profile=owner.profile,
                checkpoint=forged,
                expected_checkpoint_sha256=new_pin,
                device=_DEVICE,
                dtype=_DTYPE,
            )
        except device_recovery.ContinuousRecoveryDeviceError:
            continue
        pytest.fail(f"accepted resealed host-only checkpoint forge: {label}")


def test_view_before_pay_checkpoint_restores_open_payment_epoch_exactly():
    owner = _owner()
    _bind(owner)
    _reconcile(owner, tick=0)
    owner.publish_after_physics(_facts(owner))
    view = owner.reward_view(device_recovery.RECOVERY_REWARD_CONSUMER)
    payment = view.weighted_reward.clone()
    boundary = owner.drain_ppo_ledger(update_index=0)
    assert boundary.checkpoint_safe is True
    checkpoint = owner.checkpoint_state(boundary)
    restored = device_recovery.ContinuousRecoveryDeviceCoordinator.from_checkpoint(
        profile=owner.profile,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint["checkpoint_sha256"],
        device=_DEVICE,
        dtype=_DTYPE,
    )
    assert restored._cache_pending.tolist() == [True]
    assert restored._cache_viewed.tolist() == [True]
    assert restored._cache_paid.tolist() == [False]
    assert restored._host_payment_epoch_open is True
    assert torch.equal(restored._cache_weighted_reward, payment)
    restored.record_reward_payment(
        device_recovery.RECOVERY_REWARD_CONSUMER, payment
    )
    assert restored._cache_paid.tolist() == [True]
    assert restored._host_payment_epoch_open is False
    paid = restored.drain_ppo_ledger(update_index=1)
    assert paid.checkpoint_safe is True


def test_fault_injected_birth_is_atomic_for_all_device_and_host_state():
    owner = _owner(num_envs=3)
    before_tensors = {
        name: tensor.clone() for name, tensor in owner._state_tensors().items()
    }
    before_host = owner._host_state()

    def fail(stage: str) -> None:
        assert stage == "birth_validated_before_publish"
        raise RuntimeError("fault injection")

    with pytest.raises(RuntimeError, match="fault injection"):
        owner.bind_sequence_birth(
            (0, 2),
            reset_generations=(1, 1),
            sequence_origin_ticks=(0, 0),
            reference=_reference(2, label="atomic", profile=owner.profile),
            fault_injector=fail,
        )
    for name, tensor in owner._state_tensors().items():
        assert torch.equal(tensor, before_tensors[name]), name
    assert owner._host_state() == before_host


def test_fault_injected_deadline_latch_is_atomic_for_all_device_and_host_state():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    _reconcile(owner, tick=0)
    _reconcile(owner, tick=1, deadline_due=True)
    before_tensors = {
        name: tensor.clone() for name, tensor in owner._state_tensors().items()
    }
    before_host = owner._host_state()

    def fail(stage: str) -> None:
        assert stage == "deadline_validated_before_publish"
        raise RuntimeError("fault injection")

    with pytest.raises(RuntimeError, match="fault injection"):
        owner.latch_deadline_consumed(
            (0,),
            task_keys=(key,),
            deadline_ticks=(1,),
            fault_injector=fail,
        )
    for name, tensor in owner._state_tensors().items():
        assert torch.equal(tensor, before_tensors[name]), name
    assert owner._host_state() == before_host


def test_fault_injected_suffix_completion_is_atomic_for_all_device_and_host_state():
    owner = _owner()
    _bind(owner)
    key = _key()
    _commit(owner, key, reveal_tick=0, deadline_tick=1)
    owner.mark_playback_started((0,), task_keys=(key,))
    _reconcile(
        owner,
        tick=0,
        phase_code=device_recovery.PHASE_ACTIVE_OPPORTUNITY,
        reference_active=False,
        motion_active=True,
    )
    _reconcile(
        owner,
        tick=1,
        phase_code=device_recovery.PHASE_POST_DEADLINE_SUFFIX,
        reference_active=False,
        deadline_due=True,
    )
    _latch_deadline(owner, key, deadline_tick=1)
    _reconcile(owner, tick=2, suffix_complete=True)
    before_tensors = {
        name: tensor.clone() for name, tensor in owner._state_tensors().items()
    }
    before_host = owner._host_state()

    def fail(stage: str) -> None:
        assert stage == "suffix_validated_before_publish"
        raise RuntimeError("fault injection")

    with pytest.raises(RuntimeError, match="fault injection"):
        owner.complete_suffix(
            (0,),
            task_keys=(key,),
            completed_at_episode_ticks=(2,),
            fault_injector=fail,
        )
    for name, tensor in owner._state_tensors().items():
        assert torch.equal(tensor, before_tensors[name]), name
    assert owner._host_state() == before_host
