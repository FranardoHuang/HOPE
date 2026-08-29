"""Focused contracts for the lean row-wise PPO boundary and callpoints."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import replace
from enum import IntEnum
from pathlib import Path
import sys
import types

import pytest
import torch

import test_action_ball_full_mdp_epoch_rowwise as epoch_rowwise


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "whole_body_tracking"
MDP = SOURCE / "whole_body_tracking" / "tasks" / "tracking" / "mdp"
for path in (str(SOURCE), str(MDP)):
    if path not in sys.path:
        sys.path.insert(0, path)


# Reuse the one production module graph established by the row-wise fixture.
# Choosing a namespace from ambient ``sys.modules`` made exact dataclass checks
# depend on collection order whenever the same source had also been imported
# through its supported flat fallback.
_ready_epoch = epoch_rowwise._ready_epoch
E = epoch_rowwise.E
R = epoch_rowwise.LEAN_REWARDS
L = epoch_rowwise.LEAN
T = L.carry_txn


class _CarryRoot:
    def __init__(self):
        self._lean_carry_coordinator = None

    def _lean_carry_schema(self):
        return T._LeanCarrySchema("root", (), ())

    def _lean_carry_construction_views(self):
        return ()

    def _lean_carry_capture(self, lease):
        return T._LeanCarryCapture((), ())

    def _lean_carry_stage(self, lease, scalars, host):
        return T._LeanCarryStage((), (), ())

    def _lean_carry_target_views(self, lease, stage):
        return ()

    def _lean_carry_apply_scalars(self, lease, stage):
        assert stage.commit_started

    def _lean_carry_cross_validate(self, lease, source, host, staged):
        assert source == staged


class _CarryLeaf:
    def __init__(self, role, value, *, field_name="value"):
        self.role = role
        self.value = value
        self.field_name = field_name
        self._lean_carry_coordinator = None

    def _lean_carry_schema(self):
        return T._LeanCarrySchema(
            self.role, (),
            (T._LeanCarryTensorSpec(
                self.field_name, tuple(self.value.shape), self.value.dtype
            ),),
        )

    def _lean_carry_construction_views(self):
        return (self.value,)

    def _lean_carry_capture(self, lease):
        return T._LeanCarryCapture((), (self.value,))

    def _lean_carry_stage(self, lease, scalars, host):
        return T._LeanCarryStage(
            (), (host[0].to(device=self.value.device, copy=True).contiguous(),),
            (self.value,),
        )

    def _lean_carry_target_views(self, lease, stage):
        return (self.value,)

    def _lean_carry_apply_scalars(self, lease, stage):
        assert stage.commit_started


def _carry_graph(a, b):
    root = _CarryRoot()
    coordinator = T._LeanCarryCoordinator(
        root=root, mandatory_roles=("root", "a", "b")
    )
    coordinator._register("root", root)
    coordinator._register("a", a)
    coordinator._register("b", b)
    return coordinator


class _Env:
    def __init__(self):
        self.action_ball_full_mdp_runtime_lease = object()
        self.graph = None

    def action_ball_full_mdp_lean_reward_graph(self, lease):
        assert lease is self.action_ball_full_mdp_runtime_lease
        return self.graph


class _R05:
    def __init__(self, calls):
        self.calls = calls

    def advance_action_ball_full_mdp_rows(self):
        self.calls.append("r05")
        return None


class _FailingR05(_R05):
    def advance_action_ball_full_mdp_rows(self):
        self.calls.append("r05")
        raise RuntimeError("D05 failed")


class _Racket:
    def __init__(self, calls):
        self.calls = calls

    def arm_action_ball_full_mdp_epoch_strike_fact(self):
        self.calls.append("racket")
        return None


class _Physical:
    def __init__(
        self,
        calls,
        *,
        transport_work=True,
        recovery_epoch_work=True,
    ):
        self.calls = calls
        self.transport_work = transport_work
        self.recovery_epoch_work = recovery_epoch_work

    def refresh_action_epoch_host_activity(self, *, next_control_step):
        self.calls.append(("physical", next_control_step))
        return None

    def action_epoch_host_activity_verdict(self, *, control_step):
        return type("Activity", (), {
            "control_step": control_step,
            "transport_work": self.transport_work,
            "recovery_epoch_work": self.recovery_epoch_work,
        })()


class _R06:
    def __init__(self, calls, *, fail=False):
        self.calls = calls
        self.fail = fail
        self.epoch = None
        self.projected_payment = None

    def close_action_ball_full_mdp_epoch_reward_rows(self):
        self.calls.append("r06")
        if self.fail:
            raise RuntimeError("r06 close failed")
        if self.epoch is not None:
            self.projected_payment = (
                self.epoch.project_current_reward_payment_rows()
            )
        return None


class _PostPhysicsPhysical:
    def __init__(
        self,
        *,
        transport_work: bool,
        recovery_epoch_work: bool,
    ):
        self.transport_work = transport_work
        self.recovery_epoch_work = recovery_epoch_work
        self.launch_count = 0
        self.publish_count = 0

    def launch_action_epoch(self):
        self.launch_count += 1
        return None

    def publish_action_epoch_post_physics(self, _stamp):
        self.publish_count += 1
        return None

    def action_epoch_host_activity_verdict(self, *, control_step):
        return types.SimpleNamespace(
            control_step=control_step,
            transport_work=self.transport_work,
            recovery_epoch_work=self.recovery_epoch_work,
        )


class _PostPhysicsRacket:
    def __init__(self):
        self.publish_count = 0

    def publish_action_ball_full_mdp_epoch_strike_fact(self, *, source_step):
        assert source_step == 1
        self.publish_count += 1
        return None


class _PostPhysicsR07:
    def __init__(self):
        self.returned_facts = object()
        self.full_source_steps = []
        self.idle_source_steps = []
        self.projection_count = 0

    def publish_epoch_reward_facts(self, *, current_source_step):
        assert type(current_source_step) is torch.Tensor
        self.full_source_steps.append(current_source_step.clone())
        return self.returned_facts

    def stamp_epoch_idle_observation_without_keyed_facts(
        self, *, current_source_step
    ):
        assert type(current_source_step) is int
        self.idle_source_steps.append(current_source_step)
        return None

    def motion_ready_projection(self):
        self.projection_count += 1
        return self.returned_facts


class _PostPhysicsMotion:
    def __init__(self):
        self.installed = []

    def install_action_ball_continuous_r07_ready_projection(self, projection):
        self.installed.append(projection)
        return None


def _exact_physics_stamp_types(monkeypatch):
    module_name = "whole_body_tracking.tasks.tracking.full_mdp_env"
    module = types.ModuleType(module_name)
    phase_type = IntEnum(
        "FullMdpPhysicsEventPhase",
        {"POST_SCENE_UPDATE": 1},
        module=module_name,
    )
    pre_type = namedtuple(
        "FullMdpPrePhysicsSubstepStamp",
        (
            "control_step",
            "physics_substep",
            "physics_substeps_per_control",
            "sim_step_before",
        ),
        module=module_name,
    )
    post_type = namedtuple(
        "FullMdpPhysicsSubstepStamp",
        (
            "control_step",
            "physics_substep",
            "physics_substeps_per_control",
            "sim_step",
            "event_phase",
        ),
        module=module_name,
    )
    module.FullMdpPhysicsEventPhase = phase_type
    module.FullMdpPrePhysicsSubstepStamp = pre_type
    module.FullMdpPhysicsSubstepStamp = post_type
    monkeypatch.setitem(sys.modules, module_name, module)
    return pre_type, post_type, phase_type


def _devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available():
        result.append(torch.device("cuda:0"))
    return result


@pytest.fixture(params=_devices())
def device(request):
    return request.param


def _owner(
    *,
    device=torch.device("cpu"),
    r05=None,
    motion=None,
    racket=None,
    physical=None,
    r06=None,
    r07=None,
):
    env = _Env()
    epoch = E.ActionEpochOwner(num_envs=2, device=device)
    epoch.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool, device=device),
        reset_generation=torch.zeros(2, dtype=torch.int64, device=device),
    )
    graph = R.LeanActionEpochRewardGraph(epoch_owner=epoch)
    env.graph = graph
    owner = L.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=env.action_ball_full_mdp_runtime_lease,
        epoch_owner=epoch,
        reward_graph=graph,
        r05_runtime=object() if r05 is None else r05,
        motion=object() if motion is None else motion,
        racket=object() if racket is None else racket,
        physical_ball=object() if physical is None else physical,
        r06_landing_outcome=object() if r06 is None else r06,
        r03_strike_fact=object(),
        r07_recovery=object() if r07 is None else r07,
    )
    return owner, epoch, graph


def test_semantic_observation_v3_is_the_only_live_forwarding_surface(
    monkeypatch,
):
    owner, epoch, _graph = _owner()
    record = epoch.current()
    cached_observation = object()
    calls = []

    def build(*, runtime_owner, record):
        calls.append((runtime_owner, record))
        return cached_observation

    class ObservationModule:
        build_direct_action_epoch_observation_facts = staticmethod(build)

    monkeypatch.setattr(
        L.importlib,
        "import_module",
        lambda name: ObservationModule,
    )

    assert not hasattr(L.ActionBallFullMdpLeanRuntimeOwner, "action_epoch_observation_v1")
    assert not hasattr(owner, "action_epoch_observation_v1")
    first = owner.semantic_action_epoch_observation_v3(record)
    second = owner.semantic_action_epoch_observation_v3(record)
    assert first is cached_observation
    assert second is cached_observation
    assert calls == [(owner, record), (owner, record)]


def test_real_root_registers_all_five_mandatory_roles():
    epoch, *_ = _ready_epoch(device="cpu")
    graph = R.LeanActionEpochRewardGraph(epoch_owner=epoch)
    env = _Env()
    env.graph = graph
    d05 = _CarryLeaf(
        "d05", torch.zeros(2, dtype=torch.int64),
        field_name="reset_generation",
    )
    root = L.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=env.action_ball_full_mdp_runtime_lease,
        epoch_owner=epoch,
        reward_graph=graph,
        r05_runtime=d05,
        motion=object(),
        racket=object(),
        physical_ball=object(),
        r06_landing_outcome=object(),
        r03_strike_fact=object(),
        r07_recovery=object(),
    )
    coordinator = root._lean_carry_coordinator
    assert tuple(coordinator._owners) == (
        "root", "epoch", "milestone", "reward", "d05"
    )
    assert tuple(coordinator._owners.values()) == (
        root, epoch, epoch.milestone, graph, d05
    )


def test_real_root_cross_phase_rejects_d05_only_reset_generation_drift():
    epoch, *_ = _ready_epoch(device="cpu")
    graph = R.LeanActionEpochRewardGraph(epoch_owner=epoch)
    env = _Env()
    env.graph = graph
    d05 = _CarryLeaf(
        "d05", torch.zeros(2, dtype=torch.int64),
        field_name="reset_generation",
    )
    root = L.ActionBallFullMdpLeanRuntimeOwner(
        env=env, runtime_lease=env.action_ball_full_mdp_runtime_lease,
        epoch_owner=epoch, reward_graph=graph, r05_runtime=d05,
        motion=object(), racket=object(), physical_ball=object(),
        r06_landing_outcome=object(), r03_strike_fact=object(),
        r07_recovery=object(),
    )
    coordinator = root._lean_carry_coordinator
    lease = T._LeanCarryLease(coordinator, 1, "prepare")
    coordinator._active_lease = lease
    host = []
    for role in coordinator._mandatory_roles:
        host.append(tuple(
            torch.zeros(field.shape, dtype=field.dtype)
            for field in coordinator._schemas[role].tensor_fields
        ))
    epoch_reset_index = tuple(
        field.name for field in coordinator._schemas["epoch"].tensor_fields
    ).index("reset_generation")
    host[1] = tuple(
        torch.tensor([3, 4], dtype=torch.int64)
        if index == epoch_reset_index else value
        for index, value in enumerate(host[1])
    )
    d05_reset_index = tuple(
        field.name for field in coordinator._schemas["d05"].tensor_fields
    ).index("reset_generation")
    host[4] = tuple(
        torch.tensor([3, 5], dtype=torch.int64)
        if index == d05_reset_index else value
        for index, value in enumerate(host[4])
    )
    scalars = (
        (0, 0, 0, 0, 7, 11), (2, 7, 7), (),
        (11, 11, tuple(R.MANAGER_NAMES), (0.0,) * R.MANAGER_TERM_COUNT),
        (),
    )
    before = epoch.current().reset_generation.clone()
    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError,
        match="cross-owner ACK/reward chronology",
    ):
        root._lean_carry_cross_validate(
            lease, scalars, tuple(host), scalars
        )
    assert torch.equal(epoch.current().reset_generation, before)
    coordinator._active_lease = None


def test_prepare_lease_blocks_every_root_mutator_before_state_change():
    root, _epoch, _graph = _owner()
    coordinator = root._lean_carry_coordinator
    coordinator._active_lease = T._LeanCarryLease(coordinator, 1, "prepare")
    before = (
        root._business_generation,
        root._last_before_policy_control_step,
        root._pending_after_command_control_step,
        root._durable_ack_update_index,
        root._selected_reset_live_ledger_identity,
        root._poisoned,
    )
    calls = (
        lambda: root.before_policy_step(1, torch.zeros((2, 1))),
        lambda: root.selected_true_reset(object(), object()),
        lambda: root._record_durable_epoch_ack_span(
            object(), update_index=0, segment_id="s", rank=0,
            pending_byte_start=0, pending_byte_end=1,
            ack_byte_start=1, ack_byte_end=2,
        ),
        lambda: root.poison_optimizer_boundary(
            None, update_index=0, reason="injected"
        ),
        lambda: root.project_r05_true_reset(
            object(), device=torch.device("cpu"), num_envs=2,
            live_reset_ledger_identity=object(),
            live_reset_generation=torch.zeros(2, dtype=torch.int64),
        ),
    )
    for call in calls:
        with pytest.raises(T._LeanCarryError, match="overlaps"):
            call()
        assert (
            root._business_generation,
            root._last_before_policy_control_step,
            root._pending_after_command_control_step,
            root._durable_ack_update_index,
            root._selected_reset_live_ledger_identity,
            root._poisoned,
        ) == before
    coordinator._active_lease = None


def _complete_boundary(owner, *, update, completed):
    boundary = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=update,
        completed_environment_steps=completed,
    )
    owner.mark_optimizer_returned(boundary, update_index=update)
    summary = owner.prepare_post_update_summary(
        boundary, update_index=update
    )
    owner.acknowledge_post_update(
        boundary, summary, update_index=update
    )
    return summary


def test_preceding_import_same_process_reuses_canonical_module_identities():
    # A preceding collected copy of this test must not be invalidated when a
    # later copy imports its fixtures in the same process.  In particular, the
    # runtime's exact graph-type check remains meaningful rather than being
    # weakened to accommodate two spec-loaded copies of the same class.
    assert epoch_rowwise.E is E
    assert epoch_rowwise.LEAN_REWARDS is R
    assert epoch_rowwise.LEAN is L
    assert E.carry_txn is T
    assert L.carry_txn is T
    owner, _epoch, graph = _owner()
    assert owner._reward_graph_identity is graph
    assert type(graph) is R.LeanActionEpochRewardGraph


def test_prepare_materializes_once_and_ack_commits_the_typed_summary(
    device, monkeypatch
):
    owner, epoch, _graph = _owner(device=device)
    original = E.ActionEpochOwner.materialize_drain
    calls = []

    def counted(self, *, start, end):
        calls.append((start, end))
        return original(self, start=start, end=end)

    monkeypatch.setattr(E.ActionEpochOwner, "materialize_drain", counted)
    summary = _complete_boundary(owner, update=0, completed=8)
    assert type(summary) is L.ActionEpochPpoBoundarySummary
    assert calls == [(0, 1)]
    assert summary.frontier.start_commit == 0
    assert summary.frontier.end_commit == 1
    assert summary.frontier.update_index == 0
    assert summary.frontier.completed_environment_steps == 8
    assert summary.frontier.due_terminal_overlap_rows == 0
    assert summary.settlement.transactions == 0
    assert epoch.drain_frontier == 1
    assert not hasattr(owner, "require_owned_runner_frontier_projection")


def test_post_update_summary_is_non_destructive_until_exact_ack():
    owner, epoch, _graph = _owner()
    boundary = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=8
    )
    owner.mark_optimizer_returned(boundary, update_index=0)
    summary = owner.prepare_post_update_summary(boundary, update_index=0)

    assert owner._active_post_update_summary is summary
    assert epoch.drain_frontier == 0
    assert owner._acked_commit_end == 0
    assert owner._next_update_index == 0

    returned = owner.acknowledge_post_update(
        boundary, summary, update_index=0
    )
    assert returned is None
    assert epoch.drain_frontier == 1
    assert owner._active_post_update_summary is None


def test_optimizer_boundary_idle_guard_rejects_active_and_pending_durable_ack():
    owner, _epoch, _graph = _owner()
    owner.require_optimizer_boundary_idle()
    boundary = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=8
    )

    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError,
        match="optimizer boundary is active",
    ):
        owner.require_optimizer_boundary_idle()

    owner.mark_optimizer_returned(boundary, update_index=0)
    summary = owner.prepare_post_update_summary(boundary, update_index=0)
    owner.acknowledge_post_update(boundary, summary, update_index=0)
    assert owner._active_boundary is None
    assert owner._pending_durable_ack_summary is summary
    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError,
        match="awaiting durable ACK",
    ):
        owner.require_optimizer_boundary_idle()


def test_post_update_ack_rejects_equal_copy_before_destructive_epoch_ack(
    monkeypatch,
):
    owner, epoch, _graph = _owner()
    epoch_ack_calls = []

    def record_epoch_ack(self, *, start, end):
        epoch_ack_calls.append((self, start, end))

    monkeypatch.setattr(E.ActionEpochOwner, "acknowledge_drain", record_epoch_ack)
    boundary = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=8
    )
    owner.mark_optimizer_returned(boundary, update_index=0)
    summary = owner.prepare_post_update_summary(boundary, update_index=0)
    foreign = replace(summary)
    assert foreign == summary and foreign is not summary

    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError, match="foreign, stale, or unprepared"
    ):
        owner.acknowledge_post_update(boundary, foreign, update_index=0)
    assert owner.poisoned
    assert owner._active_post_update_summary is summary
    assert epoch.drain_frontier == 0
    assert epoch_ack_calls == []


def test_update_zero_and_later_use_the_same_zero_or_many_abi():
    owner, _epoch, _graph = _owner()
    first = _complete_boundary(owner, update=0, completed=4)
    later = _complete_boundary(owner, update=1, completed=8)
    assert first.settlement.transactions == later.settlement.transactions == 0
    assert first.frontier.next_update_index == 1
    assert later.frontier.next_update_index == 2
    assert later.frontier.start_commit == later.frontier.end_commit == 1


def test_failed_journal_ack_does_not_install_provisional_continuation(monkeypatch):
    owner, _epoch, _graph = _owner()
    original_continuation = owner._acked_continuation
    boundary = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=4
    )
    owner.mark_optimizer_returned(boundary, update_index=0)

    def fail_ack(self, *, start, end):
        del self, start, end
        raise RuntimeError("journal ACK failed")

    monkeypatch.setattr(E.ActionEpochOwner, "acknowledge_drain", fail_ack)
    summary = owner.prepare_post_update_summary(boundary, update_index=0)
    with pytest.raises(RuntimeError, match="journal ACK failed"):
        owner.acknowledge_post_update(boundary, summary, update_index=0)
    assert owner.poisoned
    assert owner._active_post_update_summary is summary
    assert owner._acked_continuation is original_continuation
    assert owner._acked_commit_end == 0
    assert owner._next_update_index == 0


def test_host_named_row_fault_stops_before_optimizer_and_materializes_once(monkeypatch):
    owner, epoch, _graph = _owner()
    epoch._undrained_row_fault_bits[1] = E.ROW_FAULT_MOTION_CLOSE_CONTRACT
    original = E.ActionEpochOwner.materialize_drain
    calls = []

    def counted(self, *, start, end):
        calls.append((start, end))
        materialized = original(self, start=start, end=end)
        assert materialized.row_fault_bits.device.type == "cpu"
        return materialized

    monkeypatch.setattr(E.ActionEpochOwner, "materialize_drain", counted)
    with pytest.raises(
        L.ActionBallFullMdpEpochRowFaultError,
        match=r"motion_close_contract\(rows=1,envs=\[1\]\)",
    ):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0, completed_environment_steps=4
        )
    assert calls == [(0, 1)]
    assert owner.poisoned


def test_host_selected_reset_generation_overflow_fault_is_named_exactly():
    owner, epoch, _graph = _owner()
    epoch._undrained_row_fault_bits[0] = (
        E.ROW_FAULT_SELECTED_RESET_GENERATION_OVERFLOW
    )

    with pytest.raises(
        L.ActionBallFullMdpEpochRowFaultError,
        match=(
            r"selected_reset_generation_overflow\(rows=1,envs=\[0\]\)"
        ),
    ) as raised:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0, completed_environment_steps=4
        )

    assert raised.value.row_fault_bits.tolist() == [
        E.ROW_FAULT_SELECTED_RESET_GENERATION_OVERFLOW,
        0,
    ]
    assert owner.poisoned


def test_exact_bound_runtime_faults_pack_and_decode_each_named_cause():
    owner, epoch, _graph = _owner()
    r06 = epoch_rowwise._R06(epoch, epoch.device)
    motion = epoch_rowwise._PlaybackOwner(epoch)
    epoch.bind_fact_owner("r06_landing_outcome", r06)
    epoch.bind_async_owner("r06_landing_outcome", r06)
    epoch.bind_motion_playback_owner(motion)
    rows = torch.tensor([True, False], dtype=torch.bool, device=epoch.device)
    other_rows = ~rows
    epoch.latch_runtime_row_fault(
        "r06_landing_outcome",
        E.ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE,
        rows,
        owner=r06,
    )
    epoch.latch_runtime_row_fault(
        "motion",
        E.ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT,
        other_rows,
        owner=motion,
    )
    epoch.latch_runtime_row_fault(
        "motion",
        E.ROW_FAULT_MOTION_TASK_TIMING_CONTRACT,
        other_rows,
        owner=motion,
    )
    epoch.latch_runtime_row_fault(
        "r06_landing_outcome",
        E.ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE,
        rows,
        owner=r06,
    )

    with pytest.raises(L.ActionBallFullMdpEpochRowFaultError) as raised:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0, completed_environment_steps=4
        )

    expected = (
        E.ROW_FAULT_R06_OUTCOME_PROJECTION_DUPLICATE
        | E.ROW_FAULT_R06_PAYMENT_MAILBOX_DUPLICATE
    )
    assert raised.value.row_fault_bits.tolist() == [
        expected,
        (
            E.ROW_FAULT_MOTION_REVEAL_REFERENCE_CONTRACT
            | E.ROW_FAULT_MOTION_TASK_TIMING_CONTRACT
        ),
    ]
    assert (
        "r06_outcome_projection_duplicate(rows=1,envs=[0])"
        in str(raised.value)
    )
    assert (
        "r06_payment_mailbox_duplicate(rows=1,envs=[0])"
        in str(raised.value)
    )
    assert (
        "motion_reveal_reference_contract(rows=1,envs=[1])"
        in str(raised.value)
    )
    assert "motion_task_timing_contract(rows=1,envs=[1])" in str(
        raised.value
    )
    assert owner.poisoned


def test_host_row_fault_reports_compound_named_and_unknown_bits_exactly():
    owner, epoch, _graph = _owner()
    unknown = 1 << 40
    epoch._undrained_row_fault_bits.copy_(torch.tensor([
        E.ROW_FAULT_MOTION_CLOSE_CONTRACT | unknown,
        E.ROW_FAULT_RESET_GENESIS_CONTRACT,
    ], dtype=torch.int64))

    with pytest.raises(L.ActionBallFullMdpEpochRowFaultError) as raised:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0, completed_environment_steps=4
        )

    fault = raised.value
    assert fault.row_fault_bits.tolist() == [
        E.ROW_FAULT_MOTION_CLOSE_CONTRACT | unknown,
        E.ROW_FAULT_RESET_GENESIS_CONTRACT,
    ]
    message = str(fault)
    assert "reset_genesis_contract(rows=1,envs=[1])" in message
    assert "motion_close_contract(rows=1,envs=[0])" in message
    assert (
        "unknown_action_epoch_row_fault_bits"
        f"(rows=1,envs=[0],values=[{unknown}])"
    ) in message
    assert owner.poisoned


def test_after_command_orders_r05_then_racket_without_caller_rows():
    calls = []
    owner, _epoch, _graph = _owner(
        r05=_R05(calls),
        racket=_Racket(calls),
        physical=_Physical(calls),
    )
    owner.after_command_compute_before_observation(0)
    assert calls == ["r05", ("physical", 1), "racket"]
    owner.before_policy_step(1, torch.zeros((2, 1)))
    owner.after_command_compute_before_observation(1)
    assert calls[-3:] == ["r05", ("physical", 2), "racket"]
    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError, match="stale, skipped, or replayed"
    ):
        owner.after_command_compute_before_observation(1)


@pytest.mark.parametrize("recovery_epoch_work", (False, True))
def test_after_command_transport_idle_skips_r03_arm(recovery_epoch_work):
    calls = []
    physical = _Physical(
        calls,
        transport_work=False,
        recovery_epoch_work=recovery_epoch_work,
    )
    owner, _epoch, _graph = _owner(
        r05=_R05(calls),
        racket=_Racket(calls),
        physical=physical,
    )
    owner.after_command_compute_before_observation(0)
    assert calls == ["r05", ("physical", 1)]
    verdict = physical.action_epoch_host_activity_verdict(control_step=1)
    assert verdict.transport_work is False
    assert verdict.recovery_epoch_work is recovery_epoch_work


@pytest.mark.parametrize(
    (
        "transport_work",
        "recovery_epoch_work",
        "expected_racket",
        "expected_full_r07",
        "expected_idle_r07",
    ),
    (
        pytest.param(False, False, 0, 0, 1, id="idle"),
        pytest.param(True, False, 1, 0, 1, id="transport-pre-recovery"),
        pytest.param(False, True, 0, 1, 0, id="actual-recovery"),
    ),
)
def test_post_physics_reads_r07_plant_only_for_actual_recovery_epoch(
    monkeypatch,
    transport_work,
    recovery_epoch_work,
    expected_racket,
    expected_full_r07,
    expected_idle_r07,
):
    pre_type, post_type, phase_type = _exact_physics_stamp_types(monkeypatch)
    physical = _PostPhysicsPhysical(
        transport_work=transport_work,
        recovery_epoch_work=recovery_epoch_work,
    )
    racket = _PostPhysicsRacket()
    r07 = _PostPhysicsR07()
    motion = _PostPhysicsMotion()
    owner, _epoch, _graph = _owner(
        physical=physical,
        racket=racket,
        r07=r07,
        motion=motion,
    )

    owner.before_policy_step(1, torch.zeros((2, 1)))
    owner.before_physics_substep(pre_type(1, 0, 1, 0))
    assert owner.publish_post_physics_substep(
        post_type(1, 0, 1, 1, phase_type.POST_SCENE_UPDATE)
    ) is None

    assert physical.launch_count == 1
    assert physical.publish_count == 1
    assert racket.publish_count == expected_racket
    assert len(r07.full_source_steps) == expected_full_r07
    assert r07.idle_source_steps == [0] * expected_idle_r07
    # Fresh Motion never consumes R07 readiness; R07 remains the independent
    # keyed recovery/critic producer above.
    assert r07.projection_count == 0
    assert motion.installed == []
    if expected_full_r07:
        assert torch.equal(
            r07.full_source_steps[0], torch.zeros(2, dtype=torch.int64)
        )
    assert not owner.poisoned


def test_after_command_rejects_the_removed_r05_surface():
    owner, _epoch, _graph = _owner(r05=object(), racket=_Racket([]))
    with pytest.raises(L.ActionBallFullMdpLeanRuntimeError, match="advance"):
        owner.after_command_compute_before_observation(0)


def test_after_command_failure_poison_is_attributed_to_exact_r05_owner(
    monkeypatch,
):
    calls = []
    r05 = _FailingR05(calls)
    owner, epoch, _graph = _owner(r05=r05, racket=_Racket(calls))
    poison_calls = []

    def capture_poison(self, owner_kind, reason_code, *, owner):
        assert self is epoch
        poison_calls.append((owner_kind, reason_code, owner))
        return self.current()

    monkeypatch.setattr(E.ActionEpochOwner, "poison_owner_write", capture_poison)
    with pytest.raises(RuntimeError, match="D05 failed"):
        owner.after_command_compute_before_observation(0)
    assert calls == ["r05"]
    assert poison_calls == [("r05_runtime", 24, r05)]
    assert owner.poisoned


def test_selected_reset_global_poison_reaches_epoch_with_exact_r05_owner():
    calls = []
    r05 = _R05(calls)
    owner, epoch, _graph = _owner(r05=r05)
    # The focused harness does not construct DeviceR05; install the same exact
    # owner identity that production bind_d05_accept_writers records.
    epoch._d05_owner = r05
    with owner._lock:
        owner._poison_selected_reset_locked("selected-reset commit failed")
    assert owner.poisoned
    assert epoch.poisoned


def test_after_reward_ignores_same_writer_echo_then_r06_pulls_independent_projection(
    monkeypatch,
):
    calls = []
    r06 = _R06(calls)
    owner, epoch, graph = _owner(r06=r06)
    r06.epoch = epoch
    owner._last_before_policy_control_step = 1
    graph._completed_cycle_count = 1
    graph._actual_closed_cycle_count = 1
    commit_before = epoch.commit_head
    publish_payment = E.ActionEpochOwner.publish_reward_payment
    same_writer_echo = object()

    def publish_then_echo(self, control_step):
        assert publish_payment(self, control_step) is None
        return same_writer_echo

    monkeypatch.setattr(
        E.ActionEpochOwner, "publish_reward_payment", publish_then_echo
    )
    owner.after_reward_close(1)
    assert calls == ["r06"]
    assert epoch.commit_head == commit_before + 1
    assert type(r06.projected_payment) is E.ActionEpochRewardPaymentRows
    assert r06.projected_payment.payment_step.tolist() == [-1, -1]
    start, end = epoch.prepare_drain()
    materialized = epoch.materialize_drain(start=start, end=end)
    assert materialized.entries[-1].transition == "PAYMENT_RECORDED"
    epoch.acknowledge_drain(start=start, end=end)
    assert owner._last_reward_control_step == 1


def test_after_reward_rejects_wrong_control_step_before_payment_or_r06():
    calls = []
    owner, epoch, graph = _owner(r06=_R06(calls))
    owner._last_before_policy_control_step = 1
    graph._completed_cycle_count = 1
    graph._actual_closed_cycle_count = 1
    commit_before = epoch.commit_head
    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError, match="skipped, duplicated, or replayed"
    ):
        owner.after_reward_close(2)
    assert calls == []
    assert epoch.commit_head == commit_before
    assert owner._last_reward_control_step == 0


def test_runtime_rejects_reward_cycle_without_exact_actual_buffer_close():
    calls = []
    owner, epoch, graph = _owner(r06=_R06(calls))
    owner._last_before_policy_control_step = 1
    graph._completed_cycle_count = 1
    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError, match="actual-buffer close"
    ):
        owner.after_reward_close(1)
    assert calls == []
    assert epoch.commit_head == 1

    boundary_owner, _epoch, boundary_graph = _owner()
    boundary_graph._completed_cycle_count = 1
    with pytest.raises(
        L.ActionBallFullMdpLeanRuntimeError,
        match="unfinished actual-buffer accounting",
    ):
        boundary_owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0, completed_environment_steps=4
        )
    assert boundary_owner.poisoned is True


def test_r06_failure_after_payment_is_sticky_and_not_replayable():
    calls = []
    owner, epoch, graph = _owner(r06=_R06(calls, fail=True))
    owner._last_before_policy_control_step = 1
    graph._completed_cycle_count = 1
    graph._actual_closed_cycle_count = 1
    commit_before = epoch.commit_head
    with pytest.raises(RuntimeError, match="r06 close failed"):
        owner.after_reward_close(1)
    assert calls == ["r06"]
    assert epoch.commit_head == commit_before + 1
    assert owner.poisoned
    with pytest.raises(L.ActionBallFullMdpLeanRuntimeError, match="poisoned"):
        owner.after_reward_close(1)
    assert calls == ["r06"]


def test_carry_core_one_composite_d2h_handles_unaligned_zero_dim_mixed_dtype(monkeypatch):
    source = _carry_graph(
        _CarryLeaf("a", torch.tensor(True)),
        _CarryLeaf("b", torch.tensor(17, dtype=torch.int64)),
    )
    calls = []
    original = T._single_composite_d2h

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(T, "_single_composite_d2h", counted)
    image = source._capture()
    assert len(calls) == 1
    target_a = _CarryLeaf("a", torch.tensor(False))
    target_b = _CarryLeaf("b", torch.tensor(0, dtype=torch.int64))
    target = _carry_graph(target_a, target_b)
    prepared = target._prepare(image)
    assert not hasattr(prepared, "stages") and not hasattr(image, "host_tensors")
    target._commit(prepared)
    assert bool(target_a.value) and int(target_b.value) == 17


def test_carry_core_rejects_source_overlap_before_transfer_and_same_writer(monkeypatch):
    shared = torch.arange(2, dtype=torch.int64)
    root = _CarryRoot()
    coordinator = T._LeanCarryCoordinator(
        root=root, mandatory_roles=("root", "a", "b")
    )
    coordinator._register("root", root)
    leaf = _CarryLeaf("a", shared[:1])
    coordinator._register("a", leaf)
    with pytest.raises(T._LeanCarryError, match="foreign, duplicate, or null"):
        coordinator._register("b", leaf)

    graph = _carry_graph(
        _CarryLeaf("a", shared[:1]), _CarryLeaf("b", shared[:1])
    )
    original_d2h = T._single_composite_d2h
    monkeypatch.setattr(
        T, "_single_composite_d2h",
        lambda _value: (_ for _ in ()).throw(AssertionError("D2H ran")),
    )
    with pytest.raises(T._LeanCarryError, match="aliases"):
        graph._capture()
    monkeypatch.setattr(T, "_single_composite_d2h", original_d2h)

    source_a = _CarryLeaf("a", torch.tensor([3]))
    source_b = _CarryLeaf("b", torch.tensor([4]))
    source_graph = _carry_graph(source_a, source_b)
    image = source_graph._capture()
    source_a._lean_carry_coordinator = None
    mixed_target = _carry_graph(source_a, _CarryLeaf("b", torch.tensor([0])))
    with pytest.raises(T._LeanCarryError, match="owner identities overlap"):
        mixed_target._prepare(image)
    mixed_target._discard(image)

    image = source_graph._capture()
    aliased_target = _carry_graph(
        _CarryLeaf("a", source_a.value), _CarryLeaf("b", torch.tensor([0]))
    )
    with pytest.raises(T._LeanCarryError, match="source.*target|target.*source"):
        aliased_target._prepare(image)
    aliased_target._discard(image)


def test_carry_core_freezes_construction_target_and_abort_allows_retry():
    source = _carry_graph(
        _CarryLeaf("a", torch.tensor([1])),
        _CarryLeaf("b", torch.tensor([2])),
    )
    image = source._capture()
    target_a = _CarryLeaf("a", torch.tensor([0]))
    target = _carry_graph(target_a, _CarryLeaf("b", torch.tensor([0])))
    prepared = target._prepare(image)
    target._abort(prepared)
    prepared = target._prepare(image)
    target._abort(prepared)
    target_a.value = torch.tensor([0])
    with pytest.raises(T._LeanCarryError, match="construction target identity"):
        target._prepare(image)
    target._discard(image)
    with pytest.raises(T._LeanCarryError, match="discard authority"):
        target._discard(image)


@pytest.mark.parametrize("alias_kind", ("cross_target", "retained_source"))
def test_carry_commit_rejects_storage_alias_introduced_after_prepare(alias_kind):
    source_a = _CarryLeaf("a", torch.tensor([1]))
    source_b = _CarryLeaf("b", torch.tensor([2]))
    source = _carry_graph(source_a, source_b)
    image = source._capture()
    target_a = _CarryLeaf("a", torch.tensor([0]))
    target_b = _CarryLeaf("b", torch.tensor([0]))
    target = _carry_graph(target_a, target_b)
    prepared = target._prepare(image)
    if alias_kind == "cross_target":
        target_a.value.set_(target_b.value)
    else:
        target_a.value.set_(source_a.value)

    with pytest.raises(T._LeanCarryError, match="aliases"):
        target._commit(prepared)

    assert int(target_b.value) == 0
    assert int(source_a.value) == 1 and int(source_b.value) == 2
    target._abort(prepared)
    target._discard(image)


def test_carry_discard_rejects_busy_image_without_leaking_table():
    before = len(T._IMAGE_STATES)
    source = _carry_graph(
        _CarryLeaf("a", torch.tensor([1])),
        _CarryLeaf("b", torch.tensor([2])),
    )
    image = source._capture()
    target = _carry_graph(
        _CarryLeaf("a", torch.tensor([0])),
        _CarryLeaf("b", torch.tensor([0])),
    )
    prepared = target._prepare(image)
    with pytest.raises(T._LeanCarryError, match="discard authority"):
        target._discard(image)
    target._abort(prepared)
    target._discard(image)
    assert len(T._IMAGE_STATES) == before


def test_carry_save_style_capture_discard_repeats_without_image_leak():
    before = len(T._IMAGE_STATES)
    source = _carry_graph(
        _CarryLeaf("a", torch.tensor([1])),
        _CarryLeaf("b", torch.tensor([2])),
    )
    for _ in range(2):
        image = source._capture()
        assert len(T._IMAGE_STATES) == before + 1
        source._discard(image)
        assert len(T._IMAGE_STATES) == before


def test_carry_partial_commit_poison_is_process_wide(monkeypatch):
    original_apply = _CarryLeaf._lean_carry_apply_scalars

    def fail_b(self, lease, stage):
        assert stage.commit_started
        if self.role == "b":
            raise RuntimeError("injected post-copy failure")
        return original_apply(self, lease, stage)

    source = _carry_graph(
        _CarryLeaf("a", torch.tensor([1])),
        _CarryLeaf("b", torch.tensor([2])),
    )
    image = source._capture()
    monkeypatch.setattr(_CarryLeaf, "_lean_carry_apply_scalars", fail_b)
    target = _carry_graph(
        _CarryLeaf("a", torch.tensor([0])),
        _CarryLeaf("b", torch.tensor([0])),
    )
    prepared = target._prepare(image)
    try:
        with pytest.raises(RuntimeError, match="post-copy"):
            target._commit(prepared)
        with pytest.raises(T._LeanCarryError, match="process is poisoned"):
            source._capture()
    finally:
        T._PROCESS_POISON_REASON = None
        T._IMAGE_BUSY.discard(image)
        T._IMAGE_STATES.pop(image, None)
