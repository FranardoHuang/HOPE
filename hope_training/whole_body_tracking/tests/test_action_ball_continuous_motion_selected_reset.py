"""Focused fresh Motion selected-reset leaf tests.

Run on the exact Pod1 Isaac environment.  This harness binds only the explicit
pin-pending diagnostic authority; it cannot authorize runtime integration or a
training launch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import struct
import sys

import pytest
import torch

import test_action_ball_continuous_motion_bridge as bridge


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import action_ball_continuous_runtime_transaction_device as device_r05
import action_ball_continuous_target_sampler as target_sampler


_DRAIN_SOURCE_PATH = (
    _SOURCE_ROOT
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_full_mdp_ppo_drain.py"
)
if str(_DRAIN_SOURCE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_DRAIN_SOURCE_PATH.parent))
# Motion imports this construction authority lazily.  Bind the same module
# object under both normal names so exact class identity is preserved in the
# synthetic IsaacLab package harness.
_DRAIN_PACKAGE = "whole_body_tracking.tasks.tracking.mdp"
_DRAIN_CANONICAL_NAME = f"{_DRAIN_PACKAGE}.action_ball_full_mdp_ppo_drain"
if _DRAIN_CANONICAL_NAME in sys.modules:
    D = sys.modules[_DRAIN_CANONICAL_NAME]
else:
    import action_ball_full_mdp_ppo_drain as D

    sys.modules[_DRAIN_CANONICAL_NAME] = D
sys.modules["action_ball_full_mdp_ppo_drain"] = D
if _DRAIN_PACKAGE in sys.modules:
    setattr(sys.modules[_DRAIN_PACKAGE], "action_ball_full_mdp_ppo_drain", D)


C = bridge.C


@dataclass(frozen=True)
class _PreparedProjection:
    owner_kind: str
    prepared_true_reset: object
    prepared_identity: object
    reset_event_identity: object
    selected_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    generation_overflow_fault: torch.Tensor


class _DeviceR05Authority:
    def __init__(self, *, command, mask: torch.Tensor):
        self.prepared = object()
        self.receipt = object()
        self.mask = mask
        self.prepared_identity = object()
        self.reset_event_identity = object()
        self.generation_before = command._action_ball_reset_generation.clone()
        self.generation_overflow_fault = mask & (
            self.generation_before == torch.iinfo(torch.int64).max
        )
        self.generation_after = self.generation_before + (
            mask & ~self.generation_overflow_fault
        ).to(dtype=torch.int64)

    def require_owned_prepared_true_reset(self, prepared, *, owner_kind):
        if prepared is not self.prepared or owner_kind != "motion":
            raise RuntimeError("prepared true reset differs")
        return _PreparedProjection(
            owner_kind=owner_kind,
            prepared_true_reset=prepared,
            prepared_identity=self.prepared_identity,
            reset_event_identity=self.reset_event_identity,
            selected_mask=self.mask,
            generation_before=self.generation_before,
            generation_after=self.generation_after,
            generation_overflow_fault=self.generation_overflow_fault,
        )

    def require_owned_true_reset_receipt(
        self,
        receipt,
        *,
        expected_prepared_true_reset,
    ):
        if (
            receipt is not self.receipt
            or expected_prepared_true_reset is not self.prepared
        ):
            raise RuntimeError("R05 reset acknowledgement differs")
        return receipt


class _GenesisAuthority:
    def __init__(self, reset_generations):
        self.receipt = object()
        self.world_reset_identity = object()
        self.reset_generations = tuple(reset_generations)
        self.live_reset_ledger_identity = None

    def require_owned_r05_genesis(self, receipt, *, device, num_envs):
        if (
            receipt is not self.receipt
            or device != torch.device("cpu")
            or num_envs != len(self.reset_generations)
        ):
            raise RuntimeError("Device-R05 genesis differs")
        return device_r05.DeviceGenesisProjection(
            world_reset_identity=self.world_reset_identity,
            reset_generations=torch.tensor(
                self.reset_generations,
                dtype=torch.int64,
                device=device,
            ),
        )


class _ProfileAuthority:
    """Concrete cold-profile authority matching current Device-R05 ABI."""

    def __init__(self, portable, *, device):
        self.receipt = object()
        cell_ids = tuple(cell.cell_id for cell in portable.cells)
        semantic_sha256s = tuple(
            portable.semantic_sha256(cell) for cell in portable.cells
        )
        targets = tuple(tuple(cell.target) for cell in portable.cells)
        digest = hashlib.sha256()
        digest.update(portable.profile_sha256.encode("ascii"))
        for cell_id, semantic_sha256 in zip(cell_ids, semantic_sha256s):
            digest.update(len(cell_id).to_bytes(8, "big"))
            digest.update(cell_id.encode("utf-8"))
            digest.update(bytes.fromhex(semantic_sha256))
        for row in targets:
            for value in row:
                digest.update(struct.pack(">f", value))
        self.projection = device_r05.DeviceProfileProjection(
            profile_sha256=portable.profile_sha256,
            profile_binding_sha256=digest.hexdigest(),
            cell_ids=cell_ids,
            semantic_sha256s=semantic_sha256s,
            targets_xy_m=torch.tensor(
                targets, dtype=torch.float32, device=device
            ),
        )

    def require_owned_r05_profile(self, receipt):
        if receipt is not self.receipt:
            raise RuntimeError("Device-R05 profile receipt differs")
        return self.projection


class _UnusedCadenceAuthority:
    def project_current_action_epoch_rows(self):
        raise AssertionError("cadence is outside this focused reset test")


class _UnusedQuestionAuthority:
    def project_r05_candidate_bank(self, *_args, **_kwargs):
        raise AssertionError("question bank is outside this focused reset test")


class _UnusedRevealAuthority:
    def project_owned_r05_reveal_boundary(self, *_args, **_kwargs):
        raise AssertionError("reveal is outside this focused reset test")

    def require_owned_r05_reveal_boundary(self, *_args, **_kwargs):
        raise AssertionError("reveal is outside this focused reset test")

    def require_owned_r05_terminal_arm(self, *_args, **_kwargs):
        raise AssertionError("reveal is outside this focused reset test")

    def require_owned_r05_terminal_commit(self, *_args, **_kwargs):
        raise AssertionError("reveal is outside this focused reset test")


class _UnusedChildCompletionAuthority:
    def require_owned_r05_child_completion(self, *_args, **_kwargs):
        raise AssertionError("reveal completion is outside this reset test")


class _UnusedDrainAuthority:
    def materialize_r05_device_drain(self, *_args, **_kwargs):
        raise AssertionError("drain is outside this focused reset test")

    def require_owned_r05_drain_ack(self, *_args, **_kwargs):
        raise AssertionError("drain is outside this focused reset test")


class _TrueResetAuthority:
    """Minimal exact top authority for a real Device-R05 interop test."""

    def __init__(self, *, selected_env_ids, reset_generations):
        self.event = object()
        self.event_identity = object()
        self.selected_env_ids = tuple(selected_env_ids)
        self.reset_generations = tuple(reset_generations)
        self.live_reset_ledger_identity = None
        self.prepared = None
        self.child_commit_identities = None
        self.preflight_capability = None

    def project_r05_true_reset(
        self,
        receipt,
        *,
        device,
        num_envs,
        live_reset_ledger_identity,
        live_reset_generation,
    ):
        expected = torch.tensor(
            self.reset_generations,
            dtype=torch.int64,
            device=device,
        )
        if (
            receipt is not self.event
            or type(live_reset_ledger_identity)
            is not device_r05.DeviceR05LiveResetLedger
            or num_envs != len(self.reset_generations)
            or not torch.equal(live_reset_generation, expected)
        ):
            raise RuntimeError("Device-R05 reset event differs")
        if self.live_reset_ledger_identity is None:
            self.live_reset_ledger_identity = live_reset_ledger_identity
        elif self.live_reset_ledger_identity is not live_reset_ledger_identity:
            raise RuntimeError("Device-R05 live reset ledger differs")
        index = torch.tensor(
            self.selected_env_ids,
            dtype=torch.int64,
            device=device,
        )
        mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        mask[index] = True
        return device_r05.DeviceTrueResetEventProjection(
            reset_event_identity=self.event_identity,
            selected_env_index=index,
            selected_mask=mask,
        )

    def authorize_child_commits(self, prepared, motion_commit_token):
        if (
            type(motion_commit_token)
            is not C.ActionBallContinuousMotionSelectedResetChildTerminalToken
        ):
            raise RuntimeError("Motion reset commit token differs")
        self.prepared = prepared
        self.child_commit_identities = (
            motion_commit_token,
            object(),
            object(),
            object(),
        )

    def authorize_preflight(self, owner, prepared):
        self.prepared = prepared
        self.preflight_capability = object()
        owner.register_true_reset_preflight(
            prepared, self.preflight_capability
        )

    def require_owned_r05_true_reset_preflight(
        self, prepared, *, preflight_capability
    ):
        if (
            prepared is not self.prepared
            or preflight_capability is not self.preflight_capability
        ):
            raise RuntimeError("true-reset preflight differs")
        return device_r05.DeviceTrueResetPreflightProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.event_identity,
            preflight_capability=preflight_capability,
        )

    def require_owned_r05_true_reset_commit(self, prepared, *, owner_view):
        if (
            prepared is not self.prepared
            or self.child_commit_identities is None
            or owner_view.prepared_true_reset is not prepared
            or bool(torch.any(owner_view.generation_overflow_fault))
        ):
            raise RuntimeError("four-child reset commit authority differs")
        return device_r05.DeviceTrueResetCommitProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.event_identity,
            child_kinds=device_r05.CHILD_OWNER_ORDER,
            child_commit_identities=self.child_commit_identities,
            preflight_capability=self.preflight_capability,
        )

    def require_owned_r05_true_reset_abort(self, prepared):
        return device_r05.DeviceTrueResetAbortProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.event_identity,
            child_commits_started=self.child_commit_identities is not None,
        )

    def require_owned_r05_true_reset_child_completion(
        self, receipt, *, child_kind, child_receipt
    ):
        if (
            receipt is None
            or child_kind not in device_r05.CHILD_OWNER_ORDER
            or child_receipt is None
        ):
            raise RuntimeError("true-reset child completion differs")
        return device_r05.DeviceTrueResetChildCompletionProjection(
            true_reset_receipt=receipt,
            child_kind=child_kind,
            child_receipt=child_receipt,
        )


def _portable_profile():
    return target_sampler.ContinuousTargetProfile(
        frame_id="fixture_env_frame",
        frame_binding_sha256="9" * 64,
        runtime_dtype=target_sampler.RUNTIME_DTYPE,
        quantization_contract=target_sampler.QUANTIZATION_CONTRACT,
        components=("landing_x_m", "landing_y_m"),
        cells=(
            target_sampler.TargetCell("center", (2.5, 0.0)),
            target_sampler.TargetCell("left", (2.5, -0.2)),
        ),
    )


def _real_device_r05_owner(command):
    reset_generations = tuple(
        int(value)
        for value in (
            command._action_ball_reset_generation.detach().cpu().tolist()
        )
    )
    genesis = _GenesisAuthority(reset_generations)
    true_reset = _TrueResetAuthority(
        selected_env_ids=(1,),
        reset_generations=reset_generations,
    )
    profile = _ProfileAuthority(_portable_profile(), device="cpu")
    owner = device_r05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=71,
        num_envs=command.num_envs,
        journal_capacity=8,
        max_reveal_epochs_per_drain=8,
        genesis_authority=genesis,
        genesis_receipt=genesis.receipt,
        cadence_authority=_UnusedCadenceAuthority(),
        question_authority=_UnusedQuestionAuthority(),
        reveal_boundary_authority=_UnusedRevealAuthority(),
        child_completion_authorities=tuple(
            _UnusedChildCompletionAuthority() for _ in range(4)
        ),
        drain_authority=_UnusedDrainAuthority(),
        true_reset_authority=true_reset,
    )
    return owner, true_reset


def _bind_real_device_r05_owner(command):
    command.time_left = torch.zeros(
        command.num_envs,
        dtype=torch.float32,
        device=command.device,
    ).contiguous()
    device_owner, authority = _real_device_r05_owner(command)
    command.bind_action_ball_continuous_motion_device_r05_reveal(device_owner)
    return device_owner, authority


def _command(*, bind_stub=True):
    command, env_ids = bridge._configure_unbound_command(num_envs=3)
    schedule = bridge._schedule_projection(cadence_steps=81)
    schedule.update(
        upcoming_action_slot=0,
        upcoming_action_uid=(
            command._action_ball_continuous_code_owned_action_uids()[0]
        ),
    )
    command.bind_action_ball_continuous_parent_authorities(
        **bridge._parent_binding_kwargs(schedule)
    )
    command._reset_action_ball_continuous_motion_cadence(env_ids)
    device_owner, authority = _bind_real_device_r05_owner(command)
    # ``_configure_unbound_command`` bypasses ``MotionCommand.__init__``.
    # Production always owns these two cached reward tensors before the fresh
    # checkpoint lane binds, so spell them out in this focused harness too.
    if not hasattr(command, "body_pos_relative_w"):
        command.body_pos_relative_w = torch.zeros_like(
            command._action_ball_safe_ready_body_pos_w
        )
        command.body_quat_relative_w = torch.zeros_like(
            command._action_ball_safe_ready_body_quat_w
        )
        command.body_quat_relative_w[..., 0] = 1.0
    # The focused harness bypasses the production FullMDP table binding that
    # allocates the checkpointed physical/task scene-frame tensors.
    n = command.num_envs
    dtype = command.motion.body_pos_w.dtype
    command._action_ball_full_mdp_frozen_root_pos_w = torch.zeros(
        n, 3, dtype=dtype, device=command.device
    )
    command._action_ball_full_mdp_frozen_root_quat_wxyz = torch.zeros(
        n, 4, dtype=dtype, device=command.device
    )
    command._action_ball_full_mdp_frozen_root_quat_wxyz[:, 0] = 1.0
    command._action_ball_full_mdp_frozen_root_valid = torch.ones(
        n, dtype=torch.bool, device=command.device
    )
    command._action_ball_full_mdp_task_yaw_wxyz = torch.zeros(
        n, 4, dtype=dtype, device=command.device
    )
    command._action_ball_full_mdp_task_yaw_wxyz[:, 0] = 1.0
    command._action_ball_full_mdp_task_translation_w = torch.zeros(
        n, 3, dtype=dtype, device=command.device
    )
    r05_owner = _DeviceR05Authority(
        command=command,
        mask=torch.tensor([False, True, False], dtype=torch.bool)
    )
    if bind_stub:
        command.bind_action_ball_continuous_motion_selected_reset(
            r05_owner,
            prepared_reset_validator=r05_owner.require_owned_prepared_true_reset,
            r05_receipt_validator=r05_owner.require_owned_true_reset_receipt,
            diagnostic=True,
        )
    return command, device_owner, r05_owner if bind_stub else authority


def _tensor_snapshot(command):
    names = (
        "_action_ball_continuous_sequence_active",
        "_action_ball_continuous_episode_step",
        "_action_ball_continuous_scheduled_ordinal",
        "_action_ball_continuous_current_reveal_step",
        "_action_ball_continuous_current_deadline_step",
        "_action_ball_continuous_next_reveal_step",
        "_action_ball_continuous_last_closed_ordinal",
        "_action_ball_continuous_opportunities_consumed",
        "_action_ball_continuous_policy_opportunities_created",
        "_action_ball_continuous_infrastructure_censors_consumed",
        "_action_ball_continuous_current_policy_opportunity",
        "_action_ball_continuous_motion_active",
        "_action_ball_continuous_suffix_complete",
        "_action_ball_continuous_ready_reference_active",
        "_action_ball_continuous_ready_at_reveal",
        "_action_ball_continuous_reveal_due",
        "_action_ball_continuous_deadline_due",
        "_action_ball_continuous_recovery_unavailable",
        "_action_ball_continuous_task_commit_pending",
        "_action_ball_continuous_task_commit_missed",
        "_action_ball_continuous_task_committed",
        "_action_ball_continuous_motion_release_pending",
        "_action_ball_continuous_motion_release_missed",
        "_action_ball_continuous_phase",
        "_action_ball_continuous_canonical_phase",
        "_action_ball_continuous_canonical_phase_start_tick",
        "_action_ball_continuous_canonical_task_identity",
        "_action_ball_continuous_canonical_cadence_identity",
        "_action_ball_continuous_canonical_action_uid",
        "_action_ball_continuous_canonical_shot_index",
        "_action_ball_continuous_canonical_outcome_identity",
        "_action_ball_continuous_canonical_candidate_identity",
        "_action_ball_continuous_canonical_task_valid",
        "_action_ball_continuous_canonical_playback_started",
        "_action_ball_task_timing_active",
        "_action_ball_task_pending_elapsed_s",
        "_action_ball_task_age_s",
        "_action_ball_time_to_contact_s",
        "_action_ball_teacher_rate",
        "_action_ball_scaled_t_hit_s",
        "_action_ball_scaled_t_cycle_s",
        "_action_ball_pre_swing_wait_s",
        "_action_ball_continuous_motion_reset_pending",
        "_action_ball_reset_generation",
        "_action_ball_swing_generation",
        "time_steps",
        "time_steps_f",
        "speed_scale",
        "hold_counter",
        "_action_ball_continuous_motion_device_mutation_version",
    )
    return {name: getattr(command, name).clone() for name in names}


def _seed_nontrivial_live_state(command):
    snapshot = _tensor_snapshot(command)
    for index, (name, value) in enumerate(snapshot.items()):
        if value.dtype == torch.bool:
            value = torch.tensor(
                [bool((index + row) % 2) for row in range(value.numel())],
                dtype=torch.bool,
            ).reshape(value.shape)
        elif value.is_floating_point():
            value = torch.arange(
                value.numel(), dtype=value.dtype
            ).reshape(value.shape) + index + 0.25
        else:
            value = torch.arange(
                value.numel(), dtype=value.dtype
            ).reshape(value.shape) + index + 2
        getattr(command, name).copy_(value)
    command._action_ball_continuous_motion_device_mutation_version.zero_()


def _refresh_selection_generations(command, r05_owner):
    r05_owner.generation_before = command._action_ball_reset_generation.clone()
    r05_owner.generation_overflow_fault = r05_owner.mask & (
        r05_owner.generation_before == torch.iinfo(torch.int64).max
    )
    r05_owner.generation_after = r05_owner.generation_before + (
        r05_owner.mask & ~r05_owner.generation_overflow_fault
    ).to(dtype=torch.int64)


class _DrainPeer:
    """Minimal independent peer used only to exercise Motion's real row."""

    def __init__(self, *, owner_kind, schema, num_envs, device, total=0):
        self.owner_kind = owner_kind
        self.schema = schema
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.total = total
        self.pack = None
        self.poisoned = False

    def _value(self, field_name):
        if field_name == "mutation_version":
            return 0
        if field_name in ("fault_count", "invariant_count"):
            return 0
        if field_name in (
            "terminal_resolution_total",
            "policy_opportunity_total",
        ):
            return self.total
        return 0

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority,
        update_index,
        completed_environment_steps,
    ):
        assert type(update_index) is int
        assert type(completed_environment_steps) is int
        values = []
        for field in self.schema.fields:
            value = self._value(field.name)
            values.extend((value,) * field.width(self.num_envs))
        self.pack = authority.mint_device_pack(
            leaf=self,
            values=torch.tensor(
                values, dtype=torch.int64, device=self.device
            ),
        )
        return self.pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack):
        if pack is not self.pack:
            raise RuntimeError("peer drain pack differs")
        self.pack = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack,
        receipt,
        owner_row,
    ):
        if (
            pack is not self.pack
            or owner_row.owner_kind != self.owner_kind
            or receipt.device_to_host_transfers != 1
        ):
            raise RuntimeError("peer drain acknowledgement differs")
        self.pack = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason):
        assert type(reason) is str and reason
        self.poisoned = True


def _global_drain_owner(command, *, terminal_total=0, initial_update_index=0):
    schema_by_owner = {
        schema.owner_kind: schema for schema in D.DEFAULT_LEAF_SCHEMAS
    }
    leaves = {
        owner_kind: (
            command
            if owner_kind == "motion"
            else _DrainPeer(
                owner_kind=owner_kind,
                schema=schema_by_owner[owner_kind],
                num_envs=command.num_envs,
                device=command.device,
                total=terminal_total,
            )
        )
        for owner_kind in D.OWNER_ORDER
    }
    assert tuple(leaves) == D.OWNER_ORDER
    assert len({id(leaf) for leaf in leaves.values()}) == len(D.OWNER_ORDER)
    owner = D.ActionBallFullMdpPpoDrainOwner(
        num_envs=command.num_envs,
        device=command.device,
        leaves=leaves,
        diagnostic_allow_minimal_schemas=True,
        initial_update_index=initial_update_index,
    )
    # Construction joins the seven independently retained leaf identities
    # before the first property read or runtime operation closes the gate.
    owner.require_exact_leaf_bindings(leaves)
    return owner, leaves


def test_k2_motion_old_plus_one_total_is_blocked_by_real_seven_leaf_conservation():
    command, _owner_unused, _r05_owner = _command()
    # Model one completed K=2 epoch but mutate Motion to the old per-batch +1
    # accounting.  Every independent peer retains the correct selected-env
    # cardinality, so the real global coordinator must fail before optimizer.
    command._action_ball_continuous_motion_mutation_version = 1
    command._action_ball_continuous_motion_device_mutation_version.fill_(1)
    command._action_ball_continuous_motion_terminal_resolution_total = 1
    command._action_ball_continuous_motion_terminal_resolution_total_device.fill_(1)
    drain, _leaves = _global_drain_owner(command, terminal_total=2)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=2,
    )
    with pytest.raises(
        D.ActionBallFullMdpPpoDrainPoisonedError,
        match="r05_terminal_vs_motion_completion",
    ):
        drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)


def test_selected_reset_is_two_phase_selected_only_and_r05_last():
    command, _owner_unused, r05_owner = _command()
    _seed_nontrivial_live_state(command)
    _refresh_selection_generations(command, r05_owner)
    before = _tensor_snapshot(command)

    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    assert type(stage) is C.ActionBallContinuousMotionSelectedResetStage
    assert type(armed) is C.ActionBallContinuousMotionSelectedResetPrevalidated
    assert set(field.name for field in fields(type(stage))) == {
        "_owner_nonce",
        "serial",
        "owner_mutation_version",
        "stage_sha256",
    }
    assert not any(
        torch.is_tensor(getattr(stage, field.name))
        for field in fields(type(stage))
    )
    assert _tensor_snapshot(command).keys() == before.keys()
    for name, value in before.items():
        assert torch.equal(getattr(command, name), value), name

    terminal = command.commit_prevalidated_selected_reset(armed)
    assert type(terminal) is (
        C.ActionBallContinuousMotionSelectedResetChildTerminalToken
    )
    assert not hasattr(terminal, "canonical_sha256")
    assert command.require_owned_selected_reset_commit(
        terminal,
        expected_prepared_true_reset=r05_owner.prepared,
    ) is terminal
    assert command.require_owned_selected_reset_commit(
        terminal,
        expected_prepared_true_reset=r05_owner.prepared,
    ) is terminal
    with pytest.raises(RuntimeError, match="stale or foreign"):
        command.require_owned_selected_reset_commit(
            terminal,
            expected_prepared_true_reset=object(),
        )
    assert command._action_ball_continuous_motion_selected_reset_committed
    assert command._action_ball_continuous_motion_selected_reset_stage is stage
    for name, value in before.items():
        live = getattr(command, name)
        if name == "_action_ball_continuous_motion_device_mutation_version":
            assert live.tolist() == [1]
        else:
            assert torch.equal(live[[0, 2]], value[[0, 2]]), name
    assert command._action_ball_reset_generation[1] == (
        before["_action_ball_reset_generation"][1] + 1
    )
    assert command._action_ball_swing_generation[1] == 0
    assert command._action_ball_continuous_episode_step[1] == -1
    assert command._action_ball_continuous_scheduled_ordinal[1] == -1
    assert command._action_ball_continuous_next_reveal_step[1] == 2
    assert command._action_ball_continuous_motion_reset_pending[1]
    assert not command._action_ball_task_timing_active[1]

    with pytest.raises(RuntimeError, match="selected-reset lease"):
        command.exact_resume_state_dict()
    with pytest.raises(RuntimeError, match="not owner-issued"):
        command.complete_selected_reset_after_r05(terminal, object())
    assert command._action_ball_continuous_motion_poisoned

    command, _owner_unused, r05_owner = _command()
    _seed_nontrivial_live_state(command)
    _refresh_selection_generations(command, r05_owner)
    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    assert type(completion) is (
        C.ActionBallContinuousMotionSelectedResetCompletionToken
    )
    assert not hasattr(completion, "selected_env_ids")
    assert not hasattr(completion, "canonical_sha256")
    assert command.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    ) is completion
    with pytest.raises(RuntimeError, match="selected-reset lease"):
        command.exact_resume_state_dict()
    assert command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    ) is completion
    with pytest.raises(RuntimeError, match="stale or foreign"):
        command.require_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=r05_owner.prepared,
        )
    assert command._action_ball_continuous_motion_selected_reset_stage is None

    # A later selected reset cannot reuse either prior leaf token family.
    r05_owner.prepared = object()
    r05_owner.receipt = object()
    _refresh_selection_generations(command, r05_owner)
    next_stage = command.prepare_selected_reset(r05_owner.prepared)
    with pytest.raises(RuntimeError, match="forged or stale"):
        command.arm_prevalidated_selected_reset(stage)
    command.abort_prevalidated_selected_reset(next_stage)
    with pytest.raises(RuntimeError, match="stale or duplicated"):
        command.complete_selected_reset_after_r05(
            terminal, r05_owner.receipt
        )


def test_selected_reset_preserves_unselected_motion_raw_history_bits():
    command, _owner_unused, r05_owner = _command()
    _seed_nontrivial_live_state(command)
    r05_owner.mask = torch.tensor(
        [True, False, False], dtype=torch.bool
    )
    _refresh_selection_generations(command, r05_owner)
    command.time_steps_f.view(torch.int32)[1] = (
        -2147483648
    )
    command.speed_scale.view(torch.int32)[1] = 2143294004

    before = {
        name: value[1]
        .detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
        for name, value in _tensor_snapshot(command).items()
        if value.ndim >= 1 and value.shape[0] == command.num_envs
    }
    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    )

    after = {
        name: value[1]
        .detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
        for name, value in _tensor_snapshot(command).items()
        if value.ndim >= 1 and value.shape[0] == command.num_envs
    }
    assert after == before
    assert (
        command.time_steps_f[1]
        .reshape(1)
        .view(torch.int32)
        [0].item()
        == -2147483648
    )
    assert (
        command.speed_scale[1]
        .reshape(1)
        .view(torch.int32)[0]
        .item()
        == 2143294004
    )


def test_selected_reset_forgery_abort_and_legacy_tombstone():
    command, _owner_unused, r05_owner = _command()
    _seed_nontrivial_live_state(command)
    _refresh_selection_generations(command, r05_owner)
    before = _tensor_snapshot(command)

    with pytest.raises(RuntimeError, match="not owner-issued"):
        command.prepare_selected_reset(object())
    stage = command.prepare_selected_reset(r05_owner.prepared)
    with pytest.raises(RuntimeError, match="forged or stale"):
        command.arm_prevalidated_selected_reset(
            replace(stage, serial=stage.serial + 1)
        )
    command.abort_prevalidated_selected_reset(stage)
    for name, value in before.items():
        assert torch.equal(getattr(command, name), value), name

    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    command.abort_prevalidated_selected_reset(armed)
    for name, value in before.items():
        assert torch.equal(getattr(command, name), value), name

    with pytest.raises(RuntimeError, match="legacy Motion command resample"):
        command._resample_command(torch.tensor([1], dtype=torch.long))
    for name, value in before.items():
        assert torch.equal(getattr(command, name), value), name


def test_selected_reset_generation_mismatch_sets_fault_and_settles_tombstone():
    command, _owner_unused, r05_owner = _command()
    _seed_nontrivial_live_state(command)
    _refresh_selection_generations(command, r05_owner)
    r05_owner.generation_after = r05_owner.generation_after + 1
    before = _tensor_snapshot(command)

    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    assert command._action_ball_continuous_motion_fault_count_device.tolist() == [1]
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    )
    assert command._action_ball_reset_generation[1] == (
        before["_action_ball_reset_generation"][1] + 1
    )
    assert not command._action_ball_continuous_canonical_task_valid[1]
    assert not command._action_ball_continuous_motion_active[1]
    assert command._action_ball_continuous_motion_reset_pending[1]
    with pytest.raises(RuntimeError, match="globally ACKed mutation frontier"):
        command._action_ball_continuous_motion_checkpoint_payload()


def test_selected_reset_generation_max_never_wraps_and_global_drain_sees_fault():
    command, _owner_unused, r05_owner = _command()
    _seed_nontrivial_live_state(command)
    command._action_ball_reset_generation[1] = torch.iinfo(torch.int64).max
    _refresh_selection_generations(command, r05_owner)
    before = _tensor_snapshot(command)

    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    assert command._action_ball_continuous_motion_fault_count_device.tolist() == [1]
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    )
    assert command._action_ball_reset_generation[1] == torch.iinfo(torch.int64).max
    assert not command._action_ball_continuous_canonical_task_valid[1]
    assert not command._action_ball_continuous_motion_active[1]
    assert command._action_ball_continuous_motion_reset_pending[1]
    assert torch.equal(
        command._action_ball_reset_generation[[0, 2]],
        before["_action_ball_reset_generation"][[0, 2]],
    )

    drain, _peers = _global_drain_owner(command)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=command.num_envs * 24,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError, match="fault"):
        drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert command._action_ball_continuous_motion_global_drain_poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        command.exact_resume_state_dict()


def test_selected_reset_interoperates_with_real_device_r05_owner():
    command, device_owner, authority = _command(bind_stub=False)
    _seed_nontrivial_live_state(command)
    command.bind_action_ball_continuous_motion_selected_reset(
        device_owner,
        prepared_reset_validator=(
            device_owner.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            device_owner.require_owned_true_reset_receipt
        ),
        diagnostic=True,
    )
    before = _tensor_snapshot(command)
    r05_before = device_owner.reset_generation.clone()
    prepared = device_owner.prepare_true_reset_many(authority.event)
    authority.authorize_preflight(device_owner, prepared)

    stage = command.prepare_selected_reset(prepared)
    prevalidated = command.arm_prevalidated_selected_reset(stage)
    for name, value in before.items():
        assert torch.equal(getattr(command, name), value), name
    assert torch.equal(device_owner.reset_generation, r05_before)

    motion_commit = command.commit_prevalidated_selected_reset(prevalidated)
    assert command._action_ball_continuous_motion_reset_pending[1]
    assert torch.equal(device_owner.reset_generation, r05_before)

    assert command.require_owned_selected_reset_commit(
        motion_commit,
        expected_prepared_true_reset=prepared,
    ) is motion_commit
    assert command.require_owned_selected_reset_commit(
        motion_commit,
        expected_prepared_true_reset=prepared,
    ) is motion_commit

    authority.authorize_child_commits(prepared, motion_commit)
    r05_receipt = device_owner.commit_true_reset_many(prepared)
    assert torch.equal(
        device_owner.reset_generation,
        r05_before + torch.tensor([0, 1, 0], dtype=torch.int64),
    )
    completion = command.complete_selected_reset_after_r05(
        motion_commit,
        r05_receipt,
    )
    assert command.require_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=prepared,
    ) is completion
    assert command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=prepared,
    ) is completion


def test_selected_reset_real_device_r05_runs_under_inference_mode():
    command, device_owner, authority = _command(bind_stub=False)
    _seed_nontrivial_live_state(command)
    # The live reset join starts from one shared generation fact.  Keep the
    # other Motion fields nontrivial, but do not manufacture a mismatch
    # between Motion and the independently owned Device-R05 ledger in this
    # clean-path inference-mode regression.
    command._action_ball_reset_generation.copy_(
        device_owner.reset_generation
    )
    command.bind_action_ball_continuous_motion_selected_reset(
        device_owner,
        prepared_reset_validator=(
            device_owner.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            device_owner.require_owned_true_reset_receipt
        ),
        diagnostic=True,
    )
    before = command._action_ball_reset_generation.clone()

    # IsaacLab enters reset from its inference-mode environment step.  The
    # Device-R05 clone-only projection therefore contains inference tensors,
    # which intentionally have no Tensor._version counter.
    with torch.inference_mode():
        prepared = device_owner.prepare_true_reset_many(authority.event)
        authority.authorize_preflight(device_owner, prepared)
        stage = command.prepare_selected_reset(prepared)
        prevalidated = command.arm_prevalidated_selected_reset(stage)
        motion_commit = command.commit_prevalidated_selected_reset(
            prevalidated
        )
        authority.authorize_child_commits(prepared, motion_commit)
        r05_receipt = device_owner.commit_true_reset_many(prepared)
        completion = command.complete_selected_reset_after_r05(
            motion_commit,
            r05_receipt,
        )
        command.consume_owned_selected_reset_completion(
            completion,
            expected_prepared_true_reset=prepared,
        )

    expected = before + torch.tensor([0, 1, 0], dtype=torch.int64)
    assert torch.equal(command._action_ball_reset_generation, expected)
    assert torch.equal(device_owner.reset_generation, expected)


def test_production_device_r05_reveal_binder_is_exact_and_idempotent():
    command, device_owner, _authority = _command(bind_stub=False)
    command.bind_action_ball_continuous_motion_device_r05_reveal(
        device_owner
    )
    command.bind_action_ball_continuous_motion_device_r05_reveal(
        device_owner
    )
    assert command._action_ball_continuous_motion_device_r05_owner is (
        device_owner
    )
    assert command._action_ball_continuous_transaction_owner is None
    assert command._action_ball_reset_generation.tolist() == [1, 1, 1]
    # The genesis join must precede Device-R05's construction-window close;
    # reading the business-state property first would make this bind fail.
    assert device_owner._construction_window_open is True
    command.bind_action_ball_continuous_motion_selected_reset(
        device_owner,
        prepared_reset_validator=(
            device_owner.require_owned_prepared_true_reset
        ),
        r05_receipt_validator=(
            device_owner.require_owned_true_reset_receipt
        ),
        authority_source_sha256="bogus-sha-cannot-grant-or-deny-authority",
        diagnostic=False,
    )


def test_expected_selected_reset_sha_cannot_authorize_named_lookalikes():
    command, _env_ids = bridge._continuous_command(
        num_envs=3, cadence_steps=81
    )
    authority = _DeviceR05Authority(
        command=command,
        mask=torch.tensor([False, True, False], dtype=torch.bool),
    )
    command._action_ball_continuous_fresh_motion_lane_bound = True
    command._action_ball_continuous_motion_device_r05_owner = authority

    with pytest.raises(RuntimeError, match="exact Device-R05 reveal binder"):
        command.bind_action_ball_continuous_motion_selected_reset(
            authority,
            prepared_reset_validator=(
                authority.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                authority.require_owned_true_reset_receipt
            ),
            authority_source_sha256=(
                C._ACTION_BALL_CONTINUOUS_MOTION_SELECTED_RESET_AUTHORITY_API_SHA256
            ),
            diagnostic=False,
        )


def test_selected_reset_contract_has_no_per_env_host_materialization():
    prepare_source = inspect.getsource(
        C.MotionCommand.prepare_action_ball_continuous_motion_selected_reset
    )
    arm_source = inspect.getsource(
        C.MotionCommand.arm_prevalidated_action_ball_continuous_motion_selected_reset
    )
    commit_source = inspect.getsource(
        C.MotionCommand.commit_prevalidated_action_ball_continuous_motion_selected_reset
    )
    for source in (prepare_source, arm_source, commit_source):
        for forbidden in (".cpu(", ".item(", ".tolist("):
            assert forbidden not in source
    assert "_assert_async" not in arm_source
    assert "_tensor_identity_version_receipt" not in prepare_source
    assert "_tensor_matches_identity_version_receipt" not in arm_source
    assert "_action_ball_continuous_motion_swap_receipts" not in arm_source
    assert ".copy_(" in commit_source
    assert "swaps_match_receipts" not in commit_source
    assert "torch.where" not in commit_source
    assert "_assert_async" not in commit_source
    assert "_canonical_json_bytes" not in commit_source
    assert "validator(" not in commit_source
    assert ".add_(" not in commit_source
    assert set(
        field.name
        for field in fields(
            C.ActionBallContinuousMotionSelectedResetChildTerminalToken
        )
    ) == {"_owner_nonce", "serial", "stage_sha256"}
    assert set(
        field.name
        for field in fields(
            C.ActionBallContinuousMotionSelectedResetCompletionToken
        )
    ) == {"_owner_nonce", "serial", "stage_sha256"}
    assert (
        C._ACTION_BALL_CONTINUOUS_MOTION_SELECTED_RESET_AUTHORITY_API_SHA256
        == "c54c56ecc5fce051dfadd3e2bb6d90d68acedd3c7095c88508c704ac052da6da"
    )


def test_motion_global_drain_uses_one_transfer_exact_ack_and_persists_highwater():
    command, _owner_unused, _reset_unused = _command()
    drain, _peers = _global_drain_owner(command)

    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=command.num_envs * 24,
    )
    with pytest.raises(RuntimeError, match="global drain lease"):
        command._advance_action_ball_continuous_motion_cadence()
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    motion_row = next(
        row for row in receipt.owner_rows if row.owner_kind == "motion"
    )
    assert motion_row.values == (
        ("mutation_version", 0),
        ("fault_count", 0),
        ("invariant_count", 0),
        ("terminal_resolution_total", 0),
    )
    assert receipt.device_to_host_transfers == 1
    drain.mark_optimizer_returned(receipt)
    drain.acknowledge_post_update(receipt)

    assert command._action_ball_continuous_motion_global_drain_sequence == 1
    assert command._action_ball_continuous_motion_global_drain_last_update == 0
    assert (
        command._action_ball_continuous_motion_global_drain_last_completed_steps
        == command.num_envs * 24
    )
    assert (
        command._action_ball_continuous_motion_global_drain_last_acknowledged_mutation_version
        == command._action_ball_continuous_motion_mutation_version
    )
    assert (
        command._action_ball_continuous_motion_checkpoint_requires_global_drain_ack
        is False
    )
    assert command._action_ball_continuous_motion_terminal_resolution_total == 0

    leaf = command._action_ball_continuous_motion_checkpoint_payload()
    assert leaf["schema_version"] == 8
    assert leaf["terminal_resolution_total"] == 0
    assert leaf["terminal_resolution_total_device"].tolist() == [0]
    assert leaf["global_drain_sequence"] == 1
    assert leaf["global_drain_last_update"] == 0
    assert leaf["global_drain_last_completed_steps"] == command.num_envs * 24
    assert leaf["global_drain_last_acknowledged_mutation_version"] == 0
    assert leaf["checkpoint_requires_global_drain_ack"] is False
    command._prepare_action_ball_continuous_motion_checkpoint(leaf)

    missing_task_close = deepcopy(leaf)
    del missing_task_close["tensors"]["task_close_tick"]
    with pytest.raises(ValueError, match="tensor fields differ"):
        command._prepare_action_ball_continuous_motion_checkpoint(
            missing_task_close
        )


def _checkpointable_fresh_exact_command():
    """Return one globally ACKed fresh owner with coherent legacy refs."""

    command, _owner_unused, _reset_unused = _command()
    command.clip_id.copy_(
        torch.tensor(
            [
                task_ref.action_slot
                for task_ref in command._action_ball_active_task_refs
            ],
            dtype=torch.int64,
            device=command.device,
        )
    )
    command.time_steps.copy_(command.motion.seg_start[command.clip_id])
    command.time_steps_f.copy_(command.time_steps.float())

    body_pos = command._action_ball_safe_ready_body_pos_w
    body_quat = command._action_ball_safe_ready_body_quat_w
    relative_pos = command.body_pos_relative_w
    relative_quat = command.body_quat_relative_w
    body_pos.copy_(
        torch.arange(
            body_pos.numel(), dtype=body_pos.dtype, device=body_pos.device
        ).reshape_as(body_pos)
        / 10.0
    )
    relative_pos.copy_(body_pos + 20.0)
    body_quat.zero_()
    body_quat[..., 0] = 1.0
    relative_quat.zero_()
    relative_quat[..., 0] = 1.0
    command._action_ball_safe_ready_reference_pending.copy_(
        torch.tensor(
            [False, True, False],
            dtype=torch.bool,
            device=command.device,
        )
    )
    # This is a conservative work cache, not a second row-count authority.
    command._action_ball_safe_ready_pending_count = command.num_envs

    drain, _peers = _global_drain_owner(command)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=command.num_envs * 24,
    )
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    drain.mark_optimizer_returned(receipt)
    drain.acknowledge_post_update(receipt)
    return command


_READY_CHECKPOINT_FIELDS = (
    "_action_ball_safe_ready_body_pos_w",
    "_action_ball_safe_ready_body_quat_w",
    "_action_ball_safe_ready_reference_pending",
    "_action_ball_full_mdp_frozen_root_pos_w",
    "_action_ball_full_mdp_frozen_root_quat_wxyz",
    "_action_ball_full_mdp_frozen_root_valid",
    "_action_ball_full_mdp_task_yaw_wxyz",
    "_action_ball_full_mdp_task_translation_w",
    "body_pos_relative_w",
    "body_quat_relative_w",
)


def _ready_checkpoint_snapshot(command):
    return {
        **{
            name: getattr(command, name).detach().clone()
            for name in _READY_CHECKPOINT_FIELDS
        },
        "pending_count": command._action_ball_safe_ready_pending_count,
    }


def _assert_ready_checkpoint_snapshot(command, expected) -> None:
    for name in _READY_CHECKPOINT_FIELDS:
        assert torch.equal(getattr(command, name), expected[name]), name
    assert (
        command._action_ball_safe_ready_pending_count
        == expected["pending_count"]
    )


def test_schema8_exact_resume_roundtrip_restores_complete_ready_teacher():
    command = _checkpointable_fresh_exact_command()
    saved = command.exact_resume_state_dict()
    leaf = saved["action_ball_birth"]["continuous_motion_leaf"]

    assert saved["schema_version"] == 5
    assert leaf["schema_version"] == 8
    assert "safe_ready_pending_count" not in leaf
    for name in (
        "reset_ready_body_pos_w",
        "reset_ready_body_quat_w",
        "reset_ready_reference_pending",
        "frozen_root_pos_w",
        "frozen_root_quat_wxyz",
        "frozen_root_valid",
        "accepted_task_yaw_wxyz",
        "accepted_task_translation_w",
        "body_pos_relative_w",
        "body_quat_relative_w",
    ):
        assert name in leaf["tensors"]

    expected = _ready_checkpoint_snapshot(command)
    command._action_ball_safe_ready_body_pos_w.fill_(-101.0)
    command._action_ball_safe_ready_body_quat_w.fill_(-102.0)
    command._action_ball_safe_ready_reference_pending.zero_()
    command._action_ball_full_mdp_frozen_root_pos_w.fill_(-105.0)
    command._action_ball_full_mdp_frozen_root_quat_wxyz.fill_(-106.0)
    command._action_ball_full_mdp_frozen_root_valid.zero_()
    command._action_ball_full_mdp_task_yaw_wxyz.fill_(-107.0)
    command._action_ball_full_mdp_task_translation_w.fill_(-108.0)
    command.body_pos_relative_w.fill_(-103.0)
    command.body_quat_relative_w.fill_(-104.0)
    command._action_ball_safe_ready_pending_count = 0

    command.load_exact_resume_state_dict(saved, strict=True)
    _assert_ready_checkpoint_snapshot(command, expected)
    command.finalize_action_ball_exact_resume()
    bridge.motion_birth._assert_nested_equal(
        command.exact_resume_state_dict(), saved
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "tensor fields differ"),
        ("type", "must be a torch.Tensor"),
        ("dtype", "shape/dtype mismatch"),
        ("shape", "shape/dtype mismatch"),
        ("drift", "checkpoint root differs"),
    ),
)
def test_schema8_ready_teacher_missing_type_shape_and_drift_fail_closed(
    mutation,
    message,
):
    command = _checkpointable_fresh_exact_command()
    saved = command.exact_resume_state_dict()
    before = _ready_checkpoint_snapshot(command)
    damaged = deepcopy(saved)
    tensors = damaged["action_ball_birth"]["continuous_motion_leaf"][
        "tensors"
    ]
    if mutation == "missing":
        del tensors["reset_ready_body_pos_w"]
    elif mutation == "type":
        tensors["reset_ready_body_pos_w"] = tensors[
            "reset_ready_body_pos_w"
        ].tolist()
    elif mutation == "dtype":
        tensors["reset_ready_body_pos_w"] = tensors[
            "reset_ready_body_pos_w"
        ].to(torch.float64)
    elif mutation == "shape":
        tensors["reset_ready_body_pos_w"] = tensors[
            "reset_ready_body_pos_w"
        ][:-1]
    else:
        tensors["reset_ready_body_pos_w"] = tensors[
            "reset_ready_body_pos_w"
        ].clone()
        tensors["reset_ready_body_pos_w"][0, 0, 0] += 1.0

    with pytest.raises(ValueError, match=message):
        command.validate_exact_resume_state_dict(damaged, strict=True)
    _assert_ready_checkpoint_snapshot(command, before)


def test_schema8_pending_work_cache_cannot_disagree_with_device_mask():
    command = _checkpointable_fresh_exact_command()
    assert bool(command._action_ball_safe_ready_reference_pending.any())
    command._action_ball_safe_ready_pending_count = 0

    with pytest.raises(RuntimeError, match="pending cache differs from its mask"):
        command.exact_resume_state_dict()


def test_schema8_recomputed_sha_cannot_authorize_wrong_valid_se2():
    command = _checkpointable_fresh_exact_command()
    leaf = command._action_ball_continuous_motion_checkpoint_payload()
    leaf["tensors"]["canonical_phase"][0] = 0
    leaf["tensors"]["frozen_root_valid"][0] = True
    leaf["tensors"]["action_slot"][0] = 0
    leaf["tensors"]["frozen_root_pos_w"][0, :2] = torch.tensor([0.4, -0.3])
    leaf["tensors"]["frozen_root_quat_wxyz"][0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    leaf["tensors"]["accepted_task_yaw_wxyz"][0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    leaf["tensors"]["accepted_task_translation_w"][0] = torch.tensor([0.1, 0.2, 0.0])
    command._action_ball_full_mdp_source_strike_root_xy = torch.zeros((1, 2))
    command._action_ball_full_mdp_source_strike_yaw_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    payload = {}
    for key, value in leaf.items():
        if key == "canonical_sha256":
            continue
        if key == "tensors":
            payload[key] = {name: tensor.tolist() for name, tensor in value.items()}
        elif isinstance(value, torch.Tensor):
            payload[key] = value.tolist()
        else:
            payload[key] = value
    leaf["canonical_sha256"] = hashlib.sha256(C._canonical_json_bytes(payload)).hexdigest()
    with pytest.raises(ValueError, match="frozen task frame is invalid"):
        command._prepare_action_ball_continuous_motion_checkpoint(leaf)


def test_schema8_ready_teacher_load_rollback_restores_prior_live_state(
    monkeypatch,
):
    command = _checkpointable_fresh_exact_command()
    saved = command.exact_resume_state_dict()

    command._action_ball_safe_ready_body_pos_w.add_(101.0)
    command._action_ball_safe_ready_body_quat_w.mul_(-1.0)
    command._action_ball_safe_ready_reference_pending.zero_()
    command._action_ball_full_mdp_frozen_root_pos_w.add_(105.0)
    command._action_ball_full_mdp_frozen_root_quat_wxyz.mul_(-1.0)
    command._action_ball_full_mdp_frozen_root_valid.logical_not_()
    command._action_ball_full_mdp_task_yaw_wxyz.mul_(-1.0)
    command._action_ball_full_mdp_task_translation_w.sub_(108.0)
    command.body_pos_relative_w.sub_(55.0)
    command.body_quat_relative_w.mul_(-1.0)
    command._action_ball_safe_ready_pending_count = 0
    before = _ready_checkpoint_snapshot(command)

    def fail_after_leaf_copy():
        raise RuntimeError("injected post-leaf failure")

    monkeypatch.setattr(
        command,
        "_invalidate_action_ball_continuous_observation_publication",
        fail_after_leaf_copy,
    )
    with pytest.raises(RuntimeError, match="injected post-leaf failure"):
        command.load_exact_resume_state_dict(saved, strict=True)

    _assert_ready_checkpoint_snapshot(command, before)


def test_motion_rejects_same_value_foreign_real_coordinator_and_post_ack_checkpoint_mutation():
    command, _owner_unused, _reset_unused = _command()
    foreign_command, _foreign_owner_unused, _foreign_reset_unused = _command()
    drain, _peers = _global_drain_owner(command)
    foreign_drain, _foreign_peers = _global_drain_owner(foreign_command)

    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=command.num_envs * 24,
    )
    foreign_prepared = foreign_drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=foreign_command.num_envs * 24,
    )
    receipt = drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    foreign_receipt = foreign_drain.transfer_decode_pre_optimizer_ppo_boundary(
        foreign_prepared
    )
    drain.mark_optimizer_returned(receipt)
    foreign_drain.mark_optimizer_returned(foreign_receipt)
    foreign_row = next(
        row for row in foreign_receipt.owner_rows if row.owner_kind == "motion"
    )
    own_row = next(
        row for row in receipt.owner_rows if row.owner_kind == "motion"
    )
    assert foreign_row.values == own_row.values
    active = command._action_ball_continuous_motion_global_drain_active
    assert active is not None
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign"):
        command.acknowledge_pre_optimizer_ppo_boundary(
            pack=active.pack,
            receipt=foreign_receipt,
            owner_row=foreign_row,
        )

    drain.acknowledge_post_update(receipt)
    command._action_ball_continuous_motion_checkpoint_payload()
    command._increment_action_ball_continuous_motion_mutation_version()
    with pytest.raises(RuntimeError, match="globally ACKed mutation frontier"):
        command._action_ball_continuous_motion_checkpoint_payload()


def test_motion_global_drain_clean_abort_and_device_invariant_are_real():
    command, _owner_unused, _reset_unused = _command()
    drain, peers = _global_drain_owner(command)
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=command.num_envs * 24,
    )
    drain.abort_pre_optimizer_ppo_boundary(prepared)
    assert command._action_ball_continuous_motion_global_drain_active is None
    assert not command._action_ball_continuous_motion_poisoned

    # This is a device chronology contradiction, not a fixture-authored
    # invariant_count.  Motion itself derives the nonzero row in prepare.
    command._action_ball_continuous_scheduled_ordinal[1] = 0
    command._action_ball_continuous_policy_opportunities_created[1] = 2
    prepared = drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=command.num_envs * 24,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError, match="invariant"):
        drain.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert command._action_ball_continuous_motion_poisoned
    assert command._action_ball_continuous_motion_global_drain_poisoned
    assert all(
        peer.poisoned
        for name, peer in peers.items()
        if name != "motion"
    )


def test_motion_terminal_resolution_total_is_durable_and_selected_reset_does_not_clear_it():
    command, owner, r05_owner = _command()
    command._action_ball_continuous_motion_terminal_resolution_total = 7
    command._action_ball_continuous_motion_terminal_resolution_total_device.fill_(7)
    _refresh_selection_generations(command, r05_owner)
    stage = command.prepare_selected_reset(r05_owner.prepared)
    armed = command.arm_prevalidated_selected_reset(stage)
    terminal = command.commit_prevalidated_selected_reset(armed)
    completion = command.complete_selected_reset_after_r05(
        terminal, r05_owner.receipt
    )
    command.consume_owned_selected_reset_completion(
        completion,
        expected_prepared_true_reset=r05_owner.prepared,
    )
    assert command._action_ball_continuous_motion_terminal_resolution_total == 7
    assert (
        command._action_ball_continuous_motion_terminal_resolution_total_device.tolist()
        == [7]
    )

    checkpoint_command, _checkpoint_owner, _checkpoint_reset = _command()
    checkpoint_command._action_ball_continuous_motion_terminal_resolution_total = 7
    checkpoint_command._action_ball_continuous_motion_terminal_resolution_total_device.fill_(
        7
    )
    checkpoint_command._action_ball_continuous_motion_mutation_version = 7
    checkpoint_command._action_ball_continuous_motion_device_mutation_version.fill_(
        7
    )
    checkpoint_drain, _checkpoint_peers = _global_drain_owner(
        checkpoint_command,
        terminal_total=7,
    )
    checkpoint_prepared = checkpoint_drain.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=checkpoint_command.num_envs * 24,
    )
    checkpoint_receipt = checkpoint_drain.transfer_decode_pre_optimizer_ppo_boundary(
        checkpoint_prepared
    )
    checkpoint_drain.mark_optimizer_returned(checkpoint_receipt)
    checkpoint_drain.acknowledge_post_update(checkpoint_receipt)
    leaf = checkpoint_command._action_ball_continuous_motion_checkpoint_payload()
    checkpoint_command._prepare_action_ball_continuous_motion_checkpoint(leaf)
    assert leaf["terminal_resolution_total"] == 7
    assert leaf["terminal_resolution_total_device"].tolist() == [7]


def test_motion_global_drain_prepare_has_no_host_tensor_observation():
    source = inspect.getsource(
        C.MotionCommand.prepare_pre_optimizer_ppo_boundary_device_pack
    )
    for forbidden in (".cpu(", ".item(", ".tolist(", "bool(torch"):
        assert forbidden not in source
    assert "torch.zeros(1" not in source
    assert "lane_failure" in source
