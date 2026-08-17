from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_full_mdp_runtime_owner.py"
)
CHECKPOINT_SOURCE = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "action_ball_full_mdp_checkpoint.py"
)
REWARD_SOURCE = SOURCE.with_name("action_ball_full_mdp_rewards.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "action_ball_full_mdp_runtime_owner_focused", SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def _load_rewards():
    name = (
        "whole_body_tracking.tasks.tracking.mdp."
        "action_ball_full_mdp_rewards"
    )
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, REWARD_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RW = _load_rewards()


def _load_checkpoint():
    spec = importlib.util.spec_from_file_location(
        "action_ball_full_mdp_checkpoint_provider_focused",
        CHECKPOINT_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = _load_checkpoint()


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _world(*, reset_generation=3):
    return C.WorldCheckpointPhase(
        world_id=0,
        reset_generation=reset_generation,
        episode_uid_sha256=_sha(f"episode:0:{reset_generation}"),
        episode_step=81,
        task_birth_snapshot_id=17,
        reset_phase=C.ResetPhase.COMMITTED,
        physics_substep_phase=0,
        physics_in_flight=False,
        r05_phase=C.R05Phase.EMPTY,
        r05_operation_active=False,
        r05_prepared_sealed=False,
        r05_cross_owner_commit_complete=True,
        r03_phase=C.R03Phase.IDLE,
        r03_all_consumers_paid=False,
        r03_view_mask=0,
        r03_payment_mask=0,
        r06_flight_phase=C.R06FlightPhase.EMPTY,
        r06_mailbox_phase=C.R06MailboxPhase.EMPTY,
        r06_payment_epoch_open=False,
        r06_view_mask=0,
        r06_payment_mask=0,
        r07_payment_epoch_open=False,
        r07_deadline_ack_pending=False,
    )


def _boundary(*, reset_generation=3):
    return C.CheckpointBoundary(
        boundary_id_sha256=_sha(f"boundary:{reset_generation}"),
        update_index=17,
        ppo_phase=C.PPOBoundaryPhase.POST_UPDATE_ROLLOUT_EMPTY,
        environment_step_phase=C.EnvironmentStepPhase.BETWEEN_COMPLETE_STEPS,
        rollout_storage_empty=True,
        actor_frontier_sealed=True,
        critic_frontier_sealed=True,
        recurrent_frontier=C.RecurrentFrontierStatus.SEALED,
        gae_in_flight=False,
        optimizer_in_flight=False,
        reset_in_flight=False,
        worlds=(_world(reset_generation=reset_generation),),
    )


def _join_state(
    provider,
    *,
    sequence,
    reset_generation,
    reset_identity_label=None,
    current_label=None,
):
    runtime_identity = provider.runtime_owner_identity
    reset_label = (
        f"reset:0:{reset_generation}"
        if reset_identity_label is None
        else reset_identity_label
    )
    task_current_label = (
        f"task-ball-r06:0:{reset_generation}:{sequence}"
        if current_label is None
        else current_label
    )
    reset_rows = (
        M.ActionBallFullMdpWorldResetJoinRow(
            world_id=0,
            reset_generation=reset_generation,
            reset_identity_sha256=_sha(reset_label),
        ),
    )
    current_rows = (
        M.ActionBallFullMdpTaskBallR06JoinRow(
            world_id=0,
            reset_generation=reset_generation,
            task_ball_r06_current_sha256=_sha(task_current_label),
        ),
    )
    reset_root = M._canonical_sha256(
        {
            "kind": "action_ball_per_world_reset_identity_join_v1",
            "rows": [
                {
                    "world_id": row.world_id,
                    "reset_generation": row.reset_generation,
                    "reset_identity_sha256": row.reset_identity_sha256,
                }
                for row in reset_rows
            ],
        }
    )
    current_root = M._canonical_sha256(
        {
            "kind": "action_ball_task_ball_r06_current_join_v1",
            "rows": [
                {
                    "world_id": row.world_id,
                    "reset_generation": row.reset_generation,
                    "task_ball_r06_current_sha256": (
                        row.task_ball_r06_current_sha256
                    ),
                }
                for row in current_rows
            ],
        }
    )
    canonical = M._canonical_sha256(
        {
            "schema_version": M.RUNTIME_OWNER_SCHEMA_VERSION,
            "kind": M.CHECKPOINT_JOIN_STATE_KIND,
            "sequence": sequence,
            "per_world_reset_identity_sha256": reset_root,
            "task_ball_r06_current_sha256": current_root,
        }
    )
    state = M._ActionBallFullMdpCheckpointJoinState(
        schema_version=M.RUNTIME_OWNER_SCHEMA_VERSION,
        kind=M.CHECKPOINT_JOIN_STATE_KIND,
        sequence=sequence,
        world_reset_rows=reset_rows,
        task_ball_r06_rows=current_rows,
        per_world_reset_identity_sha256=reset_root,
        task_ball_r06_current_sha256=current_root,
        canonical_sha256=canonical,
        _runtime_owner_identity=runtime_identity,
    )
    return state


def _publish_state(provider, state):
    provider._publish_runtime_join_state(
        state,
        runtime_owner_identity=provider.runtime_owner_identity,
        _token=M._PROVIDER_STATE_TOKEN,
    )


class _CheckpointDrainSnapshot:
    def __init__(
        self,
        *,
        drain_sequence=0,
        next_update_index=0,
        completed_environment_steps=-1,
    ):
        self.drain_sequence = drain_sequence
        self.next_update_index = next_update_index
        self.last_completed_environment_steps = completed_environment_steps
        value = None if drain_sequence == 0 else 1
        self.mutation_version_highwaters = tuple(
            (name, value)
            for name in (
                "r05_runtime",
                "motion",
                "racket",
                "physical_ball",
                "r06_landing_outcome",
                "r03_strike_fact",
                "r07_recovery",
            )
        )
        self.checkpoint_frontier_sha256 = _sha(
            f"drain:{drain_sequence}:{next_update_index}:"
            f"{completed_environment_steps}"
        )


class _CheckpointDrain:
    def __init__(self, snapshot=None):
        self.snapshot = (
            _CheckpointDrainSnapshot() if snapshot is None else snapshot
        )
        self.boundary = None

    def snapshot_for_checkpoint_boundary(self, boundary):
        self.boundary = boundary
        return self.snapshot

    def require_owned_checkpoint_snapshot(self, boundary, snapshot):
        assert boundary is self.boundary
        assert snapshot is self.snapshot
        return snapshot


class _AuditRuntimeAuthority:
    def __init__(self):
        self.provider = None
        self.claim = None

    def _claim_r10_audit_frontier(self, drain_snapshot):
        if self.claim is None:
            self.claim = object()
        self.drain_snapshot = drain_snapshot
        return self.claim

    def _require_owned_audit_frontier_claim(self, claim, consumer_kind):
        assert consumer_kind == "r10"
        assert claim is self.claim
        return claim


def _provider_with_state(*, reset_generation=3, drain_snapshot=None):
    runtime_identity = object()
    runtime_owner = _AuditRuntimeAuthority()
    drain_owner = _CheckpointDrain(drain_snapshot)
    provider = M.ActionBallFullMdpCheckpointJoinSnapshotProvider(
        num_envs=1,
        runtime_owner_identity=runtime_identity,
        runtime_owner=runtime_owner,
        ppo_drain_owner=drain_owner,
        checkpoint_module=C,
        _token=M._PROVIDER_CONSTRUCTION_TOKEN,
    )
    runtime_owner.provider = provider
    _publish_state(
        provider,
        _join_state(
            provider,
            sequence=1,
            reset_generation=reset_generation,
        ),
    )
    return provider


class _ExplodingOwner:
    def __getattribute__(self, name):
        raise AssertionError("dependency gate inspected a caller object: " + name)


def test_dependency_inventory_is_truthfully_pin_pending_and_live_ordered():
    inventory = M.action_ball_full_mdp_runtime_dependency_inventory()
    assert inventory.frozen is False
    assert inventory.runtime_integrated is False
    assert inventory.post_physics_integrated is False
    assert inventory.selected_reset_integrated is False
    assert inventory.ppo_drain_bindings_integrated is False
    assert inventory.launch_authorized is False
    assert inventory.diagnostic_unauthorized is True
    assert M.R10_SHARED_JOIN_PROVIDER_IMPLEMENTED is True
    assert M.R10_SHARED_JOIN_PROVIDER_INTEGRATED is False
    assert inventory.child_completion_order == (
        "motion",
        "racket",
        "r06_flight",
        "physical_ball",
    )
    assert inventory.global_poison_order == (
        "motion",
        "racket",
        "physical_ball",
        "r06_flight",
        "r05",
    )
    by_role = {row.role: row for row in inventory.rows}
    assert by_role["r10_checkpoint_contract"].observed_source_sha256 == (
        hashlib.sha256(CHECKPOINT_SOURCE.read_bytes()).hexdigest()
    )
    assert by_role["r10_checkpoint_contract"].frozen is True
    assert by_role["r10_checkpoint_contract"].blocker is None
    assert len(by_role["r10_checkpoint_contract"].observed_api_sha256) == 64
    assert by_role["r10_checkpoint_contract"].observed_api_sha256 != (
        M._canonical_sha256(())
    )
    assert by_role["r10_checkpoint_finalizer_contract"].frozen is True
    assert (
        by_role["r10_checkpoint_finalizer_contract"].observed_api_sha256
        == "2384305a1584159b1193de240599650e7a5634f9cfa7d39a2e2fea6268de384c"
    )
    assert (
        by_role[
            "r10_checkpoint_finalizer_construction_bundle"
        ].observed_api_sha256
        == "5c94c90af265c8dd6c625164f44438490b5a81f6aba6f28a239cf29c85dfc95d"
    )
    assert (
        by_role["r10_checkpoint_finalizer_construction_bundle"].frozen
        is True
    )
    assert any("final source SHA is not frozen" in value for value in inventory.blockers)
    assert {
        row.role for row in inventory.rows if row.expected_api_sha256 is None
    } == {
        "r05_reveal_owner",
        "device_r05_owner",
        "motion_child",
        "racket_child",
        "r06_child",
        "physical_ball_child",
        "physical_checkpoint_adapter",
        "r03_child",
        "r07_child",
    }
    assert by_role["ppo_drain_owner"].frozen is True
    assert by_role["ppo_drain_owner"].blocker is None
    assert by_role["ppo_drain_checkpoint_contract"].frozen is True
    assert by_role["ppo_drain_runner_frontier_contract"].frozen is True
    assert by_role["ppo_drain_leaf_ack_contract"].frozen is True
    required_by_role = {
        spec.role: spec.required_methods for spec in M._DEPENDENCY_SPECS
    }
    assert "bind_action_ball_continuous_motion_device_r05_reveal" in (
        required_by_role["motion_child"]
    )
    assert "bind_action_ball_continuous_motion_staging" not in (
        required_by_role["motion_child"]
    )
    assert "bind_action_ball_full_mdp_racket_staging" in (
        required_by_role["racket_child"]
    )
    assert "bind_action_ball_continuous_racket_staging" not in (
        required_by_role["racket_child"]
    )
    assert "bind_physical_park_token_authority" in (
        required_by_role["r06_child"]
    )
    assert "bind_reveal_boundary" not in required_by_role["r06_child"]
    assert "bind_reveal_boundary_owner" not in (
        required_by_role["r06_child"]
    )
    assert "bind_r06_owner" in required_by_role["physical_ball_child"]
    assert required_by_role["r05_reveal_owner"] == (
        "poison_global_reveal_epoch",
    )
    for role in ("motion_child", "racket_child", "r06_child"):
        assert "require_owned_selected_reset_commit" in required_by_role[role]
    assert "require_committed_selected_reset_park_token" in (
        required_by_role["physical_ball_child"]
    )


def test_create_rejects_before_reading_shape_compatible_owner_objects():
    exploding = _ExplodingOwner()
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="runtime dependency DAG is not frozen",
    ):
        M.ActionBallFullMdpRuntimeOwner.create(
            num_envs=3,
            device=exploding,
            r05_owner=exploding,
            device_r05_owner=exploding,
            motion_owner=exploding,
            racket_owner=exploding,
            r06_owner=exploding,
            physical_owner=exploding,
            r03_owner=exploding,
            r07_owner=exploding,
            ppo_drain_owner=exploding,
            env=exploding,
            env_lease=exploding,
        )


def test_create_from_env_rejects_before_reading_env_or_lease():
    exploding = _ExplodingOwner()
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="runtime dependency DAG is not frozen",
    ):
        M.ActionBallFullMdpRuntimeOwner.create_from_env(exploding, exploding)


def test_create_from_env_protocol_appends_r03_r07_before_global_drain():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpRuntimeOwner"
    )
    factory = next(
        node
        for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "create_from_env"
    )
    getter_assignment = next(
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "getter_names"
            for target in node.targets
        )
    )
    assert ast.literal_eval(getter_assignment.value) == (
        "action_ball_full_mdp_num_envs",
        "action_ball_full_mdp_device",
        "action_ball_full_mdp_r05_owner",
        "action_ball_full_mdp_device_r05_owner",
        "action_ball_full_mdp_motion_owner",
        "action_ball_full_mdp_racket_owner",
        "action_ball_full_mdp_r06_owner",
        "action_ball_full_mdp_physical_owner",
        "action_ball_full_mdp_r03_owner",
        "action_ball_full_mdp_r07_owner",
        "action_ball_full_mdp_ppo_drain_owner",
    )


def test_top_constructs_exact_global_drain_identity_mapping():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "require_exact_leaf_bindings"
    ]
    assert len(calls) == 1
    mapping = calls[0].args[0]
    assert isinstance(mapping, ast.Dict)
    assert tuple(ast.literal_eval(key) for key in mapping.keys) == (
        "r05_runtime",
        "motion",
        "racket",
        "physical_ball",
        "r06_landing_outcome",
        "r03_strike_fact",
        "r07_recovery",
    )
    assert tuple(value.id for value in mapping.values) == (
        "device_r05_owner",
        "motion_owner",
        "racket_owner",
        "physical_owner",
        "r06_owner",
        "r03_owner",
        "r07_owner",
    )


def _runtime_shell(*, drain=None):
    owner = object.__new__(M.ActionBallFullMdpRuntimeOwner)
    owner._identity = object()
    owner._inventory = type("Inventory", (), {"content_sha256": _sha("dag")})()
    owner._num_envs = 2
    owner._device = "cuda:0"
    owner._env = object()
    owner._env_lease = object()
    owner._motion = _PoisonLeaf("motion")
    owner._racket = _PoisonLeaf("racket")
    owner._physical = _PoisonLeaf("physical_ball")
    owner._r06 = _PoisonLeaf("r06_flight")
    owner._r05 = _PoisonLeaf("r05")
    owner._device_r05 = _DeviceR05PoisonLeaf()
    owner._r03 = object()
    owner._r07 = object()
    owner._ppo_drain = drain if drain is not None else _DrainRecorder([])
    owner._active_optimizer_receipt = None
    owner._active_optimizer_update_index = None
    owner._audit_frontier_ring = []
    owner._r10_audit_highwater = 0
    owner._r11_audit_highwater = 0
    owner._active_r10_audit_claim = None
    owner._active_r11_audit_claim = None
    owner._r10_checkpoint_publication_validator = None
    owner._r10_checkpoint_publication_consumer = None
    owner._r11_audit_consumer = None
    owner._r11_audit_consumer_validator = None
    owner._audit_consumer_binding_open = True
    owner._poisoned = False
    owner._poison_reason = None
    owner._poison_failures = ()
    owner._selected_reset_event = None
    owner._selected_reset_prepared = None
    owner._selected_reset_child_commits = None
    owner._selected_reset_child_commits_started = False
    owner._selected_reset_projection = None
    owner._selected_reset_r05_receipt = None
    owner._selected_reset_completions = None
    owner._selected_reset_env_binding = None
    owner._selected_reset_env_binding_view = None
    owner._selected_reset_receipt = None
    owner._selected_reset_receipt_consumed = False
    owner._selected_reset_sequence = 0
    owner._reward_owner_binding_open = True
    owner._reward_owner_binding = None
    owner._full_mdp_reward_graph = None
    owner._active_pre_reward_publication = None
    owner._active_pre_reward_payload = None
    owner._reward_poisoned = False
    owner._reward_poison_reason = None
    owner._last_final_postphysics_control_step = None
    owner._lock = threading.RLock()
    owner._checkpoint_join_snapshot_provider = object()
    owner._r10_checkpoint_adapter = object()
    return owner


class _PoisonLeaf:
    def __init__(self, name, events=None, *, fail=False):
        self.name = name
        self.events = [] if events is None else events
        self.fail = fail

    def poison_global_reveal_epoch(self, reason):
        self.events.append(("poison", self.name, reason))
        if self.fail:
            raise RuntimeError(self.name + " poison failed")


class _DeviceR05PoisonLeaf:
    def __init__(self, events=None):
        self.events = [] if events is None else events

    def poison_from_external_failure(self, reason_code):
        self.events.append(("device_r05_poison", reason_code))

    def require_healthy(self):
        return None


class _AuditDrainLeaf:
    def __init__(self, receipt):
        self.receipt = receipt
        self.portable = _PortableAuditReceipt(
            update_index=receipt.update_index,
            drain_sequence=receipt.drain_sequence,
            counts=(1, 2),
        )
        self.calls = []

    def require_owned_pre_optimizer_ppo_boundary_receipt(self, receipt):
        assert receipt is self.receipt
        self.calls.append(receipt)
        return self.portable


class _DrainRecorder:
    def __init__(self, events):
        self.events = events
        self.poisoned = False
        self.poison_reason = None
        self.receipt = object()
        self.sequence = 0
        self.update_index = None
        self.completed_environment_steps = None

    def prepare_pre_optimizer_ppo_boundary(
        self, *, update_index, completed_environment_steps
    ):
        self.update_index = update_index
        self.completed_environment_steps = completed_environment_steps
        self.events.append(("prepare", update_index, completed_environment_steps))
        return object()

    def transfer_decode_pre_optimizer_ppo_boundary(self, prepared):
        self.events.append(("transfer", prepared))
        self.sequence += 1
        self.receipt = type(
            "DrainReceipt",
            (),
            {
                "drain_sequence": self.sequence,
                "update_index": self.update_index,
                "completed_environment_steps": self.completed_environment_steps,
            },
        )()
        return self.receipt

    def mark_optimizer_returned(self, receipt):
        assert receipt is self.receipt
        self.events.append(("optimizer_returned", receipt))

    def acknowledge_post_update(self, receipt):
        assert receipt is self.receipt
        self.events.append(("post_update", receipt))

    def poison_optimizer_failure(self, receipt, *, reason):
        assert receipt is self.receipt
        self.events.append(("optimizer_poison", receipt, reason))
        self.poisoned = True
        self.poison_reason = reason


def test_runtime_surface_exposes_exact_env_lease_dag_and_adapter():
    owner = _runtime_shell()
    assert owner.full_mdp_runtime_dependency_dag_sha256 == _sha("dag")
    assert owner.full_mdp_runtime_env is owner._env
    assert owner.full_mdp_runtime_lease is owner._env_lease
    assert owner.launch_authorized is False
    assert owner.diagnostic_unauthorized is True
    assert owner.launch_authorized is (not owner.diagnostic_unauthorized)
    assert owner.full_mdp_post_physics_dependency_dag_sha256 == _sha("dag")
    assert owner.full_mdp_post_physics_env is owner._env
    assert owner.full_mdp_post_physics_lease is owner._env_lease
    assert owner.checkpoint_join_snapshot_provider is (
        owner._checkpoint_join_snapshot_provider
    )
    assert owner.action_ball_r10_checkpoint_adapter is owner._r10_checkpoint_adapter


def test_device_join_accepts_only_the_same_canonical_device_spelling():
    class Device:
        def __str__(self):
            return "cuda:0"

    assert M._same_device("cuda:0", Device()) is True
    assert M._same_device("cuda:0", "cuda:1") is False
    assert M._same_device("cuda", "cuda:0") is False
    assert M._same_device("", "") is False


@dataclass(frozen=True)
class _PortableAuditReceipt:
    update_index: int
    drain_sequence: int
    counts: tuple[int, ...]


def test_runner_drain_is_one_prepare_transfer_then_two_phase_ack():
    events = []
    drain = _DrainRecorder(events)
    owner = _runtime_shell(drain=drain)
    receipt = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=4,
        completed_environment_steps=8192,
    )
    owner._r03 = _AuditDrainLeaf(receipt)
    owner._r07 = _AuditDrainLeaf(receipt)
    assert receipt is drain.receipt
    assert [row[0] for row in events] == ["prepare", "transfer"]
    with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError, match="already active"):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=4,
            completed_environment_steps=8192,
        )
    with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError, match="differs"):
        owner.mark_optimizer_returned(object(), update_index=4)
    owner.mark_optimizer_returned(receipt, update_index=4)
    owner.acknowledge_post_update(receipt, update_index=4)
    assert [row[0] for row in events] == [
        "prepare",
        "transfer",
        "optimizer_returned",
        "post_update",
    ]
    assert owner._active_optimizer_receipt is None
    assert len(owner._audit_frontier_ring) == 1
    row = owner._audit_frontier_ring[0]
    assert row.drain_sequence == 1
    assert row.update_index == 4
    assert row.completed_environment_steps == 8192
    assert row.global_receipt is receipt
    assert row.r03_receipt is owner._r03.portable
    assert row.r07_receipt is owner._r07.portable


class _ExactR11Consumer:
    def __init__(self):
        self.rows = []

    def validate(self, claim, r03_receipt, r07_receipt):
        self.rows.append((claim, r03_receipt, r07_receipt))
        return claim


class _ExactR10PublicationAuthority:
    def __init__(self):
        self.calls = []

    def validate(self, publication, claim):
        self.calls.append((publication, claim))
        return publication


def _append_audit_row(owner, *, sequence):
    update_index = sequence - 1
    completed = sequence * 2048
    global_receipt = type(
        "DrainReceipt",
        (),
        {
            "drain_sequence": sequence,
            "update_index": update_index,
            "completed_environment_steps": completed,
        },
    )()
    r03 = _PortableAuditReceipt(update_index, sequence, (sequence,))
    r07 = _PortableAuditReceipt(update_index, sequence, (sequence + 1,))
    r03_root = M._portable_receipt_sha256(r03, label="r03")
    r07_root = M._portable_receipt_sha256(r07, label="r07")
    row_root = M._canonical_sha256(
        {
            "kind": "action_ball_full_mdp_audit_frontier_row_v1",
            "drain_sequence": sequence,
            "update_index": update_index,
            "completed_environment_steps": completed,
            "r03_receipt_sha256": r03_root,
            "r07_receipt_sha256": r07_root,
        }
    )
    row = M._AuditFrontierRow(
        drain_sequence=sequence,
        update_index=update_index,
        completed_environment_steps=completed,
        global_receipt=global_receipt,
        r03_receipt=r03,
        r07_receipt=r07,
        r03_receipt_sha256=r03_root,
        r07_receipt_sha256=r07_root,
        canonical_sha256=row_root,
    )
    owner._audit_frontier_ring.append(row)
    return row


def test_bounded_audit_ring_has_independent_r10_r11_ack_and_dual_gc():
    owner = _runtime_shell()
    owner._checkpoint_join_snapshot_provider = object()
    r11 = _ExactR11Consumer()
    publication_authority = _ExactR10PublicationAuthority()
    owner.bind_r11_audit_consumer(r11, r11.validate)
    owner.bind_r10_checkpoint_publication_authority(
        publication_authority,
        publication_authority.validate,
    )
    owner._audit_consumer_binding_open = False
    row = _append_audit_row(owner, sequence=1)
    snapshot = _CheckpointDrainSnapshot(
        drain_sequence=1,
        next_update_index=1,
        completed_environment_steps=2048,
    )
    r10_claim = owner._claim_r10_audit_frontier(snapshot)
    r11_claim = owner.claim_r11_audit_frontier(r11)

    owner.acknowledge_r11_audit_frontier(r11, r11_claim)
    assert owner._r11_audit_highwater == 1
    assert owner._r10_audit_highwater == 0
    assert owner._audit_frontier_ring == [row]

    publication = object()
    owner.finalize_r10_audit_frontier(publication, r10_claim)
    assert owner._r10_audit_highwater == 1
    assert owner._audit_frontier_ring == []
    assert publication_authority.calls == [(publication, r10_claim)]


def test_r10_frontier_requires_exact_latest_drain_chronology_and_authority():
    owner = _runtime_shell()
    owner._checkpoint_join_snapshot_provider = object()
    _append_audit_row(owner, sequence=1)
    with pytest.raises(
        M.ActionBallFullMdpRuntimeOwnerError,
        match="chronology differs",
    ):
        owner._claim_r10_audit_frontier(
            _CheckpointDrainSnapshot(
                drain_sequence=1,
                next_update_index=2,
                completed_environment_steps=2048,
            )
        )
    snapshot = _CheckpointDrainSnapshot(
        drain_sequence=1,
        next_update_index=1,
        completed_environment_steps=2048,
    )
    claim = owner._claim_r10_audit_frontier(snapshot)
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="publication validator is not bound",
    ):
        owner.finalize_r10_audit_frontier(object(), claim)
    assert owner._r10_audit_highwater == 0
    assert len(owner._audit_frontier_ring) == 1


def test_audit_ring_blocks_only_at_capacity_not_after_first_update(monkeypatch):
    owner = _runtime_shell()
    for sequence in range(1, M.AUDIT_FRONTIER_RING_CAPACITY):
        _append_audit_row(owner, sequence=sequence)
    receipt = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=1,
    )
    owner._active_optimizer_receipt = None
    owner._active_optimizer_update_index = None
    _append_audit_row(owner, sequence=M.AUDIT_FRONTIER_RING_CAPACITY)
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="would overwrite",
    ):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=1,
            completed_environment_steps=2,
        )
    assert receipt is not None


def test_optimizer_failure_poison_is_sticky_and_broadcast_ordered():
    events = []
    drain = _DrainRecorder(events)
    owner = _runtime_shell(drain=drain)
    for name in ("motion", "racket", "physical", "r06", "r05"):
        setattr(
            owner,
            "_" + ("physical" if name == "physical" else name),
            _PoisonLeaf(name, events, fail=name == "racket"),
        )
    owner._device_r05 = _DeviceR05PoisonLeaf(events)
    receipt = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=2,
        completed_environment_steps=4096,
    )
    owner.poison_optimizer_boundary(
        receipt,
        update_index=2,
        reason="optimizer exploded",
    )
    assert owner.poisoned is True
    assert owner.poison_reason == "optimizer exploded"
    assert [row[1] for row in events if row[0] == "poison"] == [
        "motion",
        "racket",
        "physical",
        "r06",
        "r05",
    ]
    assert events[-1] == (
        "optimizer_poison",
        receipt,
        "optimizer exploded",
    )
    assert ("device_r05_poison", 10) in events
    assert owner.poison_failures == (("racket", "RuntimeError"),)
    with pytest.raises(M.ActionBallFullMdpRuntimePoisonedError):
        owner.require_healthy()


def test_selected_reset_authority_schema_is_small_and_opaque_ledger_bound():
    source = SOURCE.read_bytes()
    schema = M._selected_reset_authority_api_sha256(source)
    assert len(schema) == 64
    assert M.SELECTED_RESET_AUTHORITY_API_METHODS == (
        "project_r05_true_reset",
        "require_owned_r05_true_reset_commit",
        "require_owned_r05_true_reset_abort",
    )
    tree = ast.parse(source.decode("utf-8"))
    owner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpRuntimeOwner"
    )
    projector = next(
        node
        for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "project_r05_true_reset"
    )
    assert tuple(arg.arg for arg in projector.args.kwonlyargs) == (
        "device",
        "num_envs",
        "live_reset_ledger_identity",
        "live_reset_generation",
    )
    assert not projector.args.defaults
    assert all(value is None for value in projector.args.kw_defaults)
    commit = next(
        node
        for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "require_owned_r05_true_reset_commit"
    )
    assert tuple(arg.arg for arg in commit.args.kwonlyargs) == ("owner_view",)
    assert not commit.args.defaults
    assert all(value is None for value in commit.args.kw_defaults)


def test_selected_reset_packed_preflight_rejects_drift_before_leaf_commit(
    monkeypatch,
):
    """One real packed verdict rejects value drift before four writers."""

    torch = pytest.importorskip("torch")
    env_module = types.ModuleType("runtime_owner_reset_preflight_env")
    device_module = types.ModuleType("runtime_owner_reset_preflight_device")

    class _Projection:
        def __init__(
            self,
            *,
            reset_event_identity,
            selected_env_index,
            selected_mask,
            generation_before,
            generation_after,
            generation_overflow_fault,
        ):
            self.reset_event_identity = reset_event_identity
            self.selected_env_index = selected_env_index
            self.selected_mask = selected_mask
            self.generation_before = generation_before
            self.generation_after = generation_after
            self.generation_overflow_fault = generation_overflow_fault

    class _EventProjection:
        def __init__(self, **values):
            self.__dict__.update(values)

    env_module.FullMdpSelectedResetProjection = _Projection
    device_module.DeviceTrueResetEventProjection = _EventProjection
    monkeypatch.setitem(sys.modules, env_module.__name__, env_module)
    monkeypatch.setitem(sys.modules, device_module.__name__, device_module)

    class _Env:
        def project_action_ball_full_mdp_selected_reset_event(self, *args, **kwargs):
            del args, kwargs
            return self.projection

    class _Device:
        pass

    _Env.__module__ = env_module.__name__
    _Device.__module__ = device_module.__name__
    owner = _runtime_shell()
    env = _Env()
    device_owner = _Device()
    owner._env = env
    owner._device_r05 = device_owner
    owner._device = torch.device("cpu")
    owner._num_envs = 2
    event = object()
    identity = object()
    live_identity = object()
    owner._selected_reset_event = event
    owner._selected_reset_env_binding_view = types.SimpleNamespace(
        device_r05_owner=device_owner,
        live_reset_ledger_identity=live_identity,
    )
    live = torch.tensor([7, 11], dtype=torch.int64)

    def make_projection(
        *,
        index=(0,),
        mask=(True, False),
        before=(7, 11),
        after=(8, 11),
        overflow=(False, False),
    ):
        return _Projection(
            reset_event_identity=identity,
            selected_env_index=torch.tensor(index, dtype=torch.int64),
            selected_mask=torch.tensor(mask, dtype=torch.bool),
            generation_before=torch.tensor(before, dtype=torch.int64),
            generation_after=torch.tensor(after, dtype=torch.int64),
            generation_overflow_fault=torch.tensor(overflow, dtype=torch.bool),
        )

    env.projection = make_projection()
    result = owner.project_r05_true_reset(
        event,
        device=torch.device("cpu"),
        num_envs=2,
        live_reset_ledger_identity=live_identity,
        live_reset_generation=live,
    )
    assert type(result) is _EventProjection
    assert result.selected_env_index.tolist() == [0]

    bad = (
        make_projection(mask=(False, True)),
        make_projection(before=(8, 11), after=(9, 11)),
        make_projection(after=(9, 11)),
        make_projection(index=(0, 0)),
        make_projection(
            before=(torch.iinfo(torch.int64).max, 11),
            after=(torch.iinfo(torch.int64).max, 11),
            overflow=(True, False),
        ),
    )
    for candidate in bad:
        env.projection = candidate
        with pytest.raises(
            M.ActionBallFullMdpRuntimeOwnerError,
            match="selected-reset packed preflight",
        ):
            owner.project_r05_true_reset(
                event,
                device=torch.device("cpu"),
                num_envs=2,
                live_reset_ledger_identity=live_identity,
                live_reset_generation=live,
            )


def test_selected_reset_success_is_physical_r06_motion_racket_r05_last(
    monkeypatch,
):
    events = []
    event = object()
    event_identity = object()
    prepared = object()
    r05_receipt = object()
    commits = {
        name: object()
        for name in ("motion", "racket", "physical", "r06")
    }
    completions = {
        name: object()
        for name in ("motion", "racket", "r06", "physical")
    }

    class _Tensor:
        def __init__(self, *, dtype):
            self.shape = (2,)
            self.device = "cuda:0"
            self.dtype = dtype

    selected_mask = _Tensor(dtype="bool")
    generation_before = _Tensor(dtype="int64")
    generation_after = _Tensor(dtype="int64")
    overflow_fault = _Tensor(dtype="bool")
    projection = type(
        "EnvProjection",
        (),
        {
            "reset_event_identity": event_identity,
            "selected_mask": selected_mask,
            "generation_before": generation_before,
            "generation_after": generation_after,
            "generation_overflow_fault": overflow_fault,
        },
    )()

    fake_device_module = types.ModuleType(
        "runtime_owner_selected_reset_fake_device"
    )

    class _CommitInput:
        def __init__(self):
            self.prepared_true_reset = prepared
            self.reset_event_identity = event_identity
            self.selected_mask = selected_mask
            self.generation_before = generation_before
            self.generation_after = generation_after
            self.generation_overflow_fault = overflow_fault

    class _CommitProjection:
        def __init__(
            self,
            *,
            prepared_true_reset,
            reset_event_identity,
            child_kinds,
            child_commit_identities,
        ):
            self.prepared_true_reset = prepared_true_reset
            self.reset_event_identity = reset_event_identity
            self.child_kinds = child_kinds
            self.child_commit_identities = child_commit_identities

    class _AbortProjection:
        def __init__(self, **values):
            self.__dict__.update(values)

    class _ChildCompletionProjection:
        def __init__(self, **values):
            self.__dict__.update(values)

    fake_device_module.DeviceR05TrueResetCommitInput = _CommitInput
    fake_device_module.DeviceTrueResetCommitProjection = _CommitProjection
    fake_device_module.DeviceTrueResetAbortProjection = _AbortProjection
    fake_device_module.DeviceTrueResetChildCompletionProjection = (
        _ChildCompletionProjection
    )
    monkeypatch.setitem(
        sys.modules,
        fake_device_module.__name__,
        fake_device_module,
    )

    class _DeviceR05:
        def require_healthy(self):
            return None

        def prepare_true_reset_many(self, value):
            assert value is event
            events.append("d05.prepare")
            owner._selected_reset_projection = projection
            return prepared

        def commit_true_reset_many(self, value):
            assert value is prepared
            events.append("d05.commit.authority")
            proof = owner.require_owned_r05_true_reset_commit(
                prepared,
                owner_view=_CommitInput(),
            )
            assert proof.prepared_true_reset is prepared
            assert proof.reset_event_identity is event_identity
            assert proof.child_kinds == M.SELECTED_RESET_COMMIT_PROOF_ORDER
            assert proof.child_commit_identities == (
                commits["motion"],
                commits["racket"],
                commits["physical"],
                commits["r06"],
            )
            events.append("d05.commit.last")
            return r05_receipt

        def record_true_reset_child_completion(
            self, receipt, *, child_kind, child_receipt
        ):
            proof = owner.require_owned_r05_true_reset_child_completion(
                receipt,
                child_kind=child_kind,
                child_receipt=child_receipt,
            )
            assert proof.true_reset_receipt is receipt
            assert proof.child_kind == child_kind
            assert proof.child_receipt is child_receipt
            events.append(f"d05.record_completion.{child_kind}")

    _DeviceR05.__module__ = fake_device_module.__name__
    fake_device_module.DeviceR05 = _DeviceR05

    class _Motion:
        def prepare_action_ball_continuous_motion_selected_reset(self, value):
            assert value is prepared
            events.append("motion.prepare")
            return "motion-stage"

        def arm_prevalidated_action_ball_continuous_motion_selected_reset(
            self, value
        ):
            assert value == "motion-stage"
            events.append("motion.arm")
            return "motion-prevalidated"

        def commit_prevalidated_action_ball_continuous_motion_selected_reset(
            self, value
        ):
            assert value == "motion-prevalidated"
            events.append("motion.commit")
            return commits["motion"]

        def require_owned_selected_reset_commit(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is commits["motion"]
            assert expected_prepared_true_reset is prepared
            events.append("motion.require_commit")
            return value

        def complete_action_ball_continuous_motion_selected_reset_after_r05(
            self, commit, receipt
        ):
            assert commit is commits["motion"] and receipt is r05_receipt
            events.append("motion.complete")
            return completions["motion"]

        def require_owned_selected_reset_completion(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is completions["motion"]
            assert expected_prepared_true_reset is prepared
            events.append("motion.require_completion")
            return value

        def consume_owned_selected_reset_completion(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is completions["motion"]
            assert expected_prepared_true_reset is prepared
            events.append("motion.consume_completion")
            return value

    class _Racket:
        def stage_action_ball_continuous_racket_selected_reset(self, value):
            assert value is prepared
            events.append("racket.stage")
            return "racket-stage"

        def finalize_action_ball_continuous_racket_selected_reset(self, value):
            assert value == "racket-stage"
            events.append("racket.finalize")
            return "racket-prevalidated"

        def commit_prevalidated_action_ball_continuous_racket_selected_reset(
            self, value
        ):
            assert value == "racket-prevalidated"
            events.append("racket.commit")
            return commits["racket"]

        def require_owned_selected_reset_commit(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is commits["racket"]
            assert expected_prepared_true_reset is prepared
            events.append("racket.require_commit")
            return value

        def complete_action_ball_continuous_racket_selected_reset_after_r05(
            self, commit, receipt
        ):
            assert commit is commits["racket"] and receipt is r05_receipt
            events.append("racket.complete")
            return completions["racket"]

        def require_owned_selected_reset_completion(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is completions["racket"]
            assert expected_prepared_true_reset is prepared
            events.append("racket.require_completion")
            return value

        def consume_owned_selected_reset_completion(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is completions["racket"]
            assert expected_prepared_true_reset is prepared
            events.append("racket.consume_completion")
            return value

    class _R06:
        def prepare_selected_reset(self, value):
            assert value is prepared
            events.append("r06.prepare")
            return "r06-prepared"

        def arm_prevalidated_selected_reset(self, value, physical_finalized):
            assert value == "r06-prepared"
            assert physical_finalized == "physical-finalized"
            events.append("r06.arm")
            return "r06-armed"

        def commit_prevalidated_selected_reset(self, armed, physical_commit):
            assert armed == "r06-armed"
            assert physical_commit is commits["physical"]
            events.append("r06.commit")
            return commits["r06"]

        def require_owned_selected_reset_commit(
            self, value, *, expected_prepared_true_reset
        ):
            assert value is commits["r06"]
            assert expected_prepared_true_reset is prepared
            events.append("r06.require_commit")
            return value

        def complete_selected_reset_after_r05(self, commit, receipt):
            assert commit is commits["r06"] and receipt is r05_receipt
            events.append("r06.complete")
            return completions["r06"]

        def require_owned_selected_reset_completion(self, value):
            assert value is completions["r06"]
            events.append("r06.require_completion")
            return value

        def consume_owned_selected_reset_completion(self, value):
            assert value is completions["r06"]
            events.append("r06.consume_completion")
            return value

    class _Physical:
        def stage_selected_true_reset(self, value):
            assert value == "r06-prepared"
            events.append("physical.stage")
            return "physical-stage"

        def finalize_selected_true_reset(self, value):
            assert value == "physical-stage"
            events.append("physical.finalize")
            return "physical-finalized"

        def prearm_selected_true_reset(self, finalized, r06_armed):
            assert finalized == "physical-finalized"
            assert r06_armed == "r06-armed"
            events.append("physical.prearm")
            return "physical-armed"

        def commit_prevalidated_selected_true_reset(self, value):
            assert value == "physical-armed"
            events.append("physical.commit")
            return commits["physical"]

        def require_owned_selected_reset_commit(self, value):
            assert value is commits["physical"]
            events.append("physical.require_commit")
            return value

        def acknowledge_r06_selected_reset_commit(
            self, physical_commit, r06_commit
        ):
            assert physical_commit is commits["physical"]
            assert r06_commit is commits["r06"]
            events.append("physical.ack_r06")

        def complete_selected_true_reset_after_r05(
            self, physical_commit, r06_commit, receipt
        ):
            assert physical_commit is commits["physical"]
            assert r06_commit is commits["r06"]
            assert receipt is r05_receipt
            events.append("physical.complete")
            return completions["physical"]

        def require_owned_selected_reset_completion(self, value):
            assert value is completions["physical"]
            events.append("physical.require_completion")
            return value

        def consume_owned_selected_reset_completion(self, value):
            assert value is completions["physical"]
            events.append("physical.consume_completion")
            return value

    owner = _runtime_shell()
    owner._device_r05 = _DeviceR05()
    owner._motion = _Motion()
    owner._racket = _Racket()
    owner._r06 = _R06()
    owner._physical = _Physical()

    receipt = owner.selected_true_reset(event)
    assert type(receipt) is M.ActionBallFullMdpSelectedTrueResetReceipt
    assert events == [
        "d05.prepare",
        "motion.prepare",
        "motion.arm",
        "racket.stage",
        "racket.finalize",
        "r06.prepare",
        "physical.stage",
        "physical.finalize",
        "r06.arm",
        "physical.prearm",
        "physical.commit",
        "physical.require_commit",
        "r06.commit",
        "r06.require_commit",
        "physical.ack_r06",
        "motion.commit",
        "racket.commit",
        "motion.require_commit",
        "racket.require_commit",
        "d05.commit.authority",
        "motion.require_commit",
        "racket.require_commit",
        "physical.require_commit",
        "r06.require_commit",
        "d05.commit.last",
        "motion.complete",
        "racket.complete",
        "r06.complete",
        "physical.complete",
        "motion.require_completion",
        "racket.require_completion",
        "r06.require_completion",
        "physical.require_completion",
        "d05.record_completion.motion",
        "d05.record_completion.racket",
        "d05.record_completion.r06_flight",
        "d05.record_completion.physical_ball",
        "motion.consume_completion",
        "racket.consume_completion",
        "r06.consume_completion",
        "physical.consume_completion",
    ]
    assert owner.require_owned_selected_true_reset_receipt(
        receipt,
        event,
    ) is receipt
    with pytest.raises(
        M.ActionBallFullMdpRuntimeOwnerError,
        match="stale, foreign, or replayed",
    ):
        owner.require_owned_selected_true_reset_receipt(receipt, event)


def test_selected_reset_r06_arm_failure_is_poison_only_and_never_aborts():
    owner = _runtime_shell()
    events = []
    event = object()
    prepared = object()

    class _Device(_DeviceR05PoisonLeaf):
        def prepare_true_reset_many(self, value):
            assert value is event
            events.append("d05.prepare")
            return prepared

        def abort_true_reset_many(self, value):
            events.append("d05.abort")

    class _Motion(_PoisonLeaf):
        def prepare_action_ball_continuous_motion_selected_reset(self, value):
            assert value is prepared
            events.append("motion.prepare")
            return object()

        def arm_prevalidated_action_ball_continuous_motion_selected_reset(
            self, value
        ):
            events.append("motion.prevalidate")
            return object()

        def abort_prevalidated_action_ball_continuous_motion_selected_reset(
            self, value
        ):
            events.append("motion.abort")

    class _Racket(_PoisonLeaf):
        def stage_action_ball_continuous_racket_selected_reset(self, value):
            assert value is prepared
            events.append("racket.stage")
            return object()

        def finalize_action_ball_continuous_racket_selected_reset(self, value):
            events.append("racket.finalize")
            return object()

        def abort_prevalidated_action_ball_continuous_racket_selected_reset(
            self, value
        ):
            events.append("racket.abort")

    class _R06:
        def prepare_selected_reset(self, value):
            assert value is prepared
            events.append("r06.prepare")
            return object()

        def arm_prevalidated_selected_reset(self, value, physical):
            events.append("r06.arm.fail")
            raise RuntimeError("arm failed")

        def abort_selected_reset(self, value):
            events.append("r06.abort")

        def poison_selected_reset(self, reason):
            events.append("r06.poison")

    class _Physical:
        def stage_selected_true_reset(self, value):
            events.append("physical.stage")
            return object()

        def finalize_selected_true_reset(self, value):
            events.append("physical.finalize")
            return object()

        def abort_selected_true_reset(self, value):
            events.append("physical.abort")

        def poison_selected_reset(self, reason):
            events.append("physical.poison")

    owner._device_r05 = _Device(events)
    owner._motion = _Motion("motion", events)
    owner._racket = _Racket("racket", events)
    owner._r06 = _R06()
    owner._physical = _Physical()
    with pytest.raises(RuntimeError, match="arm failed"):
        owner.selected_true_reset(event)
    assert owner.poisoned is True
    assert "r06.arm.fail" in events
    assert "physical.poison" in events
    assert "r06.poison" in events
    assert "d05.abort" not in events
    assert "motion.abort" not in events
    assert "racket.abort" not in events
    assert "r06.abort" not in events
    assert "physical.abort" not in events


def test_selected_reset_precommit_abort_is_reverse_and_r05_last():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpRuntimeOwner"
    )
    method = next(
        node
        for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_abort_selected_reset_precommit"
    )
    owner = _runtime_shell()
    events = []

    class _AbortLeaf:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            assert name.startswith("abort_")

            def abort(value):
                events.append((self.name, value))

            return abort

    class _AbortR05(_DeviceR05PoisonLeaf):
        def abort_true_reset_many(self, value):
            events.append(("device_r05", value))

    owner._motion = _AbortLeaf("motion")
    owner._racket = _AbortLeaf("racket")
    owner._r06 = _AbortLeaf("r06")
    owner._physical = _AbortLeaf("physical")
    owner._device_r05 = _AbortR05()
    values = tuple(object() for _ in range(5))
    owner._selected_reset_event = object()
    owner._selected_reset_projection = object()
    owner._selected_reset_prepared = values[0]
    owner._abort_selected_reset_precommit(
        prepared=values[0],
        motion_value=values[1],
        racket_value=values[2],
        r06_prepared=values[3],
        physical_value=values[4],
    )
    assert [name for name, _value in events] == [
        "physical",
        "r06",
        "racket",
        "motion",
        "device_r05",
    ]
    assert owner._selected_reset_event is None
    assert owner._selected_reset_prepared is None
    assert method is not None


def test_selected_reset_poison_uses_selected_leaf_seams():
    owner = _runtime_shell()
    events = []

    class _Global:
        def __init__(self, name):
            self.name = name

        def poison_global_reveal_epoch(self, reason):
            events.append((self.name, reason))

    class _Selected:
        def __init__(self, name):
            self.name = name

        def poison_selected_reset(self, reason):
            events.append((self.name, reason))

    owner._motion = _Global("motion")
    owner._racket = _Global("racket")
    owner._physical = _Selected("physical")
    owner._r06 = _Selected("r06")
    owner._device_r05 = _DeviceR05PoisonLeaf(events)
    owner._poison_selected_reset("irreversible")
    assert [name for name, _reason in events[:-1]] == [
        "motion",
        "racket",
        "physical",
        "r06",
    ]
    assert events[-1] == ("device_r05_poison", 11)
    assert owner.poisoned is True


def test_unconsumed_selected_reset_receipt_blocks_policy_and_ppo():
    owner = _runtime_shell()
    event = object()
    receipt = object()
    owner._selected_reset_event = event
    owner._selected_reset_receipt = receipt
    owner._selected_reset_prepared = object()
    with pytest.raises(
        M.ActionBallFullMdpRuntimeOwnerError,
        match="unsettled selected reset",
    ):
        owner.before_policy_step(0, object())
    with pytest.raises(
        M.ActionBallFullMdpRuntimeOwnerError,
        match="unsettled selected reset",
    ):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=2,
        )


def test_production_construction_never_binds_selected_reset_diagnostic_mode():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected_bind_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "bind_action_ball_continuous_motion_selected_reset",
            "bind_action_ball_continuous_racket_selected_reset",
        }
    ]
    assert len(selected_bind_calls) == 2
    for call in selected_bind_calls:
        keywords = {value.arg: value.value for value in call.keywords}
        assert isinstance(keywords["diagnostic"], ast.Constant)
        assert keywords["diagnostic"].value is False
        assert isinstance(keywords["authority_source_sha256"], ast.Name)
        assert keywords["authority_source_sha256"].id == "authority_api_sha256"


def test_production_construction_preserves_device_r05_genesis_window_order():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpRuntimeOwner"
    )
    factory = next(
        node
        for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_with_inventory"
    )
    calls = [
        (node.lineno, node.func.attr)
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "bind_action_ball_continuous_motion_device_r05_reveal",
            "bind_action_ball_full_mdp_racket_staging",
            "bind_device_r05_reset_owner",
            "bind_r06_owner",
            "project_full_mdp_env_reset_binding",
            "require_owned_full_mdp_env_reset_binding",
            "bind_true_reset_authority",
        }
    ]
    by_name = {}
    for lineno, name in calls:
        by_name.setdefault(name, []).append(lineno)
    child_genesis = (
        by_name["bind_action_ball_continuous_motion_device_r05_reveal"][0],
        by_name["bind_action_ball_full_mdp_racket_staging"][0],
        *by_name["bind_device_r05_reset_owner"],
        by_name["bind_r06_owner"][0],
        by_name["project_full_mdp_env_reset_binding"][0],
        by_name["require_owned_full_mdp_env_reset_binding"][0],
    )
    close = by_name["bind_true_reset_authority"][0]
    assert all(lineno < close for lineno in child_genesis)
    env_bind_line = next(
        node.lineno
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "env_reset_binder"
    )
    assert env_bind_line < close
    # The adapter constructor is a direct class call, not an attribute call.
    adapter_line = next(
        node.lineno
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PhysicalFlightCheckpointAdapter"
    )
    assert adapter_line < close


def test_production_construction_does_not_bind_portable_r05_hot_reveal():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ActionBallFullMdpRuntimeOwner"
    )
    constructor = next(
        node
        for node in owner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_with_inventory"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes.isdisjoint(
        {
            "bind_terminal_boundary_authority",
            "bind_action_ball_continuous_motion_staging",
            "bind_action_ball_continuous_racket_staging",
            "bind_action_ball_continuous_motion_reveal_boundary",
            "bind_action_ball_continuous_racket_reveal_boundary",
            "bind_reveal_boundary_owner",
            "bind_reveal_boundary",
            "bind_r05_terminal_owner",
        }
    )


def test_postphysics_capture_is_owner_issued_and_failure_poisoned():
    events = []

    class Physical(_PoisonLeaf):
        def capture_post_physics_facts(self, stamp):
            events.append(("capture", stamp))
            return "facts"

        def build_post_physics_publication(self, *, facts):
            events.append(("build", facts))
            return "publication"

        def publish_post_physics_to_r06(self, publication):
            events.append(("publish", publication))
            return "ack"

        def retire_post_physics_to_r06(self, result):
            events.append(("retire", result))

    owner = _runtime_shell()
    owner._physical = Physical("physical", events)
    stamp = object()
    owner.publish_post_physics_substep(stamp)
    assert events[:4] == [
        ("capture", stamp),
        ("build", "facts"),
        ("publish", "publication"),
        ("retire", "ack"),
    ]

    owner = _runtime_shell()
    owner._physical = _PoisonLeaf("physical", events)
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="fact producer",
    ):
        owner.publish_post_physics_substep(object())
    assert owner.poisoned is False


def test_postphysics_failure_after_capture_poison_broadcasts_all_owners():
    events = []

    class Physical(_PoisonLeaf):
        def capture_post_physics_facts(self, stamp):
            events.append(("capture", stamp))
            return "facts"

        def build_post_physics_publication(self, *, facts):
            events.append(("build", facts))
            raise RuntimeError("materialization failed")

    owner = _runtime_shell()
    owner._motion = _PoisonLeaf("motion", events)
    owner._racket = _PoisonLeaf("racket", events)
    owner._physical = Physical("physical", events)
    owner._r06 = _PoisonLeaf("r06", events)
    owner._r05 = _PoisonLeaf("r05", events)
    with pytest.raises(RuntimeError, match="materialization failed"):
        owner.publish_post_physics_substep(object())
    assert owner.poisoned is True
    assert [row[1] for row in events if row[0] == "poison"] == [
        "motion",
        "racket",
        "physical",
        "r06",
        "r05",
    ]


class _RewardOwner(_PoisonLeaf):
    def __init__(
        self,
        name,
        consumers,
        events,
        *,
        n=2,
        device=torch.device("cpu"),
        fail_close=False,
    ):
        super().__init__(name, events)
        self.full_mdp_reward_consumers = tuple(consumers)
        self.num_envs = n
        self.device = torch.device(device)
        self.fail_close = fail_close
        self.publication = object()
        self.close_receipt = object()
        self.verdicts = {}
        self.bound = False
        self.active_cycle = None

    def _bind_full_mdp_reward_graph_from_top(
        self, *, runtime_owner, ordered_consumers
    ):
        assert tuple(ordered_consumers) == self.full_mdp_reward_consumers
        self.runtime_owner = runtime_owner
        self.bound = True
        self.events.append(("reward_bind", self.name))

    def open_full_mdp_reward_cycle(
        self, publication, *, control_step, runtime_owner
    ):
        assert self.bound
        assert runtime_owner is self.runtime_owner
        cycle = object()
        self.active_cycle = cycle
        self.events.append(("reward_open", self.name, control_step))
        return cycle

    def publish_full_mdp_pre_reward(self, *, control_step, runtime_owner):
        self.events.append(("reward_publish", self.name, control_step))
        self.runtime_owner = runtime_owner
        return self.publication

    def require_owned_full_mdp_pre_reward(
        self, publication, *, control_step, runtime_owner
    ):
        assert publication is self.publication
        assert runtime_owner is self.runtime_owner
        self.events.append(("reward_require", self.name, control_step))
        return types.SimpleNamespace(
            terminated=torch.tensor(
                [False, self.name == "r03"], device=self.device
            ),
            time_out=torch.zeros(2, dtype=torch.bool, device=self.device),
        )

    def require_owned_full_mdp_reward_payment(
        self, verdict, *, consumer, control_step, runtime_owner
    ):
        assert runtime_owner is self.runtime_owner
        assert verdict is self.verdicts[consumer]
        self.events.append(("reward_verdict", self.name, consumer, control_step))
        return verdict

    def close_full_mdp_reward_cycle(
        self,
        *,
        control_step,
        pre_reward_publication,
        ordered_consumers,
        ordered_payment_verdicts,
        runtime_owner,
    ):
        assert runtime_owner is self.runtime_owner
        assert ordered_consumers == self.full_mdp_reward_consumers
        assert ordered_payment_verdicts == tuple(
            self.verdicts[name] for name in ordered_consumers
        )
        if self.name in ("r03", "r07"):
            assert pre_reward_publication is self.publication
        self.events.append(("reward_close", self.name, control_step))
        if self.fail_close:
            raise RuntimeError(self.name + " close failed")
        return self.close_receipt

    def require_owned_full_mdp_reward_close(
        self, receipt, *, control_step, runtime_owner
    ):
        assert receipt is self.close_receipt
        assert runtime_owner is self.runtime_owner
        self.events.append(("reward_closed", self.name, control_step))
        return receipt


def _reward_runtime(device_name="cpu"):
    owner = _runtime_shell()
    owner._device = torch.empty((), device=device_name).device
    events = []
    by_owner = dict(M.FULL_MDP_REWARD_OWNER_CONSUMERS)
    owner._r03 = _RewardOwner(
        "r03", by_owner["r03"], events, device=owner._device
    )
    owner._physical = _RewardOwner(
        "physical", by_owner["physical"], events, device=owner._device
    )
    owner._r06 = _RewardOwner(
        "r06", by_owner["r06"], events, device=owner._device
    )
    owner._r07 = _RewardOwner(
        "r07", by_owner["r07"], events, device=owner._device
    )
    for leaf in (owner._r03, owner._physical, owner._r06, owner._r07):
        leaf.runtime_owner = owner
    for consumer in M.FULL_MDP_REWARD_ORDERED_CONSUMERS:
        owner_name, leaf_consumer = consumer.split(":", 1)
        getattr(owner, "_" + owner_name).verdicts[leaf_consumer] = object()
    return owner, events


def _reward_graph(owner):
    return RW.FreshFullMdpRewardGraph(
        runtime_owner=owner,
        runtime_lease=owner._env_lease,
        owners=RW._RewardOwners(
            r03=owner._r03,
            physical=owner._physical,
            r06=owner._r06,
            r07=owner._r07,
            num_envs=owner._num_envs,
            device=torch.device(owner._device),
        ),
        diagnostic_unauthorized=False,
        _construction_token=RW._PRODUCTION_GRAPH_CONSTRUCTION_TOKEN,
    )


def test_reward_binding_rejects_public_diagnostic_graph_before_leaf_bind():
    owner, events = _reward_runtime()
    graph = RW.FreshFullMdpRewardGraph(
        runtime_owner=owner,
        runtime_lease=owner._env_lease,
        owners=RW._RewardOwners(
            r03=owner._r03,
            physical=owner._physical,
            r06=owner._r06,
            r07=owner._r07,
            num_envs=owner._num_envs,
            device=torch.device(owner._device),
        ),
        diagnostic_unauthorized=True,
    )
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="exact fresh graph",
    ):
        owner.bind_full_mdp_reward_owners(
            runtime_lease=owner._env_lease,
            ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
            reward_graph=graph,
        )
    assert not any(row[0] == "reward_bind" for row in events)
    assert owner._reward_owner_binding is None


def test_reward_binding_rejects_foreign_graph_leaf_before_leaf_bind():
    owner, events = _reward_runtime()
    foreign, _ = _reward_runtime()
    graph = RW.FreshFullMdpRewardGraph(
        runtime_owner=owner,
        runtime_lease=owner._env_lease,
        owners=RW._RewardOwners(
            r03=foreign._r03,
            physical=owner._physical,
            r06=owner._r06,
            r07=owner._r07,
            num_envs=owner._num_envs,
            device=torch.device(owner._device),
        ),
        diagnostic_unauthorized=False,
        _construction_token=RW._PRODUCTION_GRAPH_CONSTRUCTION_TOKEN,
    )
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="causal owner identity",
    ):
        owner.bind_full_mdp_reward_owners(
            runtime_lease=owner._env_lease,
            ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
            reward_graph=graph,
        )
    assert not any(row[0] == "reward_bind" for row in events)
    assert owner._reward_owner_binding is None


def _bind_and_publish_reward(owner, *, step=7):
    reward_graph = _reward_graph(owner)
    binding = owner.bind_full_mdp_reward_owners(
        runtime_lease=owner._env_lease,
        ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
        reward_graph=reward_graph,
    )
    owner._last_final_postphysics_control_step = step
    publication = owner.publish_full_mdp_pre_reward(
        runtime_lease=owner._env_lease,
        control_step=step,
    )
    return binding, publication


def test_top_after_reward_close_owns_the_same_graph_cycle():
    owner, events = _reward_runtime()
    graph = _reward_graph(owner)
    owner.bind_full_mdp_reward_owners(
        runtime_lease=owner._env_lease,
        ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
        reward_graph=graph,
    )
    owner._last_final_postphysics_control_step = 7
    terminated = graph.begin_pre_reward(control_step=7)
    assert terminated.tolist() == [False, True]
    for consumer in M.FULL_MDP_REWARD_ORDERED_CONSUMERS:
        owner_name, leaf_consumer = consumer.split(":", 1)
        graph._record_payment(
            consumer,
            owner_payment_result=getattr(owner, "_" + owner_name).verdicts[
                leaf_consumer
            ],
        )
    assert owner.after_reward_close(7) is None
    assert graph.active_cycle is None
    assert owner._active_pre_reward_publication is None
    assert [row[1] for row in events if row[0] == "reward_open"] == [
        "physical",
        "r06",
    ]


def test_top_after_reward_close_rejects_wrong_control_step_before_graph_close():
    owner, _ = _reward_runtime()
    graph = _reward_graph(owner)
    owner.bind_full_mdp_reward_owners(
        runtime_lease=owner._env_lease,
        ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
        reward_graph=graph,
    )
    owner._last_final_postphysics_control_step = 7
    graph.begin_pre_reward(control_step=7)
    with pytest.raises(
        M.ActionBallFullMdpRuntimeOwnerError, match="control-step graph"
    ):
        owner.after_reward_close(8)
    assert graph.active_cycle is not None
    assert owner._active_pre_reward_publication is not None
    assert owner.poisoned


def test_top_after_reward_close_missing_payment_poison_retains_cycle():
    owner, _ = _reward_runtime()
    graph = _reward_graph(owner)
    owner.bind_full_mdp_reward_owners(
        runtime_lease=owner._env_lease,
        ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
        reward_graph=graph,
    )
    owner._last_final_postphysics_control_step = 7
    graph.begin_pre_reward(control_step=7)
    for consumer in M.FULL_MDP_REWARD_ORDERED_CONSUMERS[:-1]:
        owner_name, leaf_consumer = consumer.split(":", 1)
        graph._record_payment(
            consumer,
            owner_payment_result=getattr(owner, "_" + owner_name).verdicts[
                leaf_consumer
            ],
        )
    with pytest.raises(RW.FreshFullMdpRewardCycleError, match="missing"):
        owner.after_reward_close(7)
    assert graph.active_cycle is not None
    assert owner._active_pre_reward_publication is not None
    assert owner.poisoned


def _reward_close_args(owner, publication):
    verdicts = tuple(
        getattr(owner, "_" + name.split(":", 1)[0]).verdicts[
            name.split(":", 1)[1]
        ]
        for name in M.FULL_MDP_REWARD_ORDERED_CONSUMERS
    )
    return dict(
        runtime_lease=owner._env_lease,
        pre_reward_publication=publication,
        ordered_owner_payment_results=verdicts,
        ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_reward_binding_publish_require_and_close_use_real_owner_order(
    device_name,
):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    owner, events = _reward_runtime(device_name)
    binding, publication = _bind_and_publish_reward(owner)
    assert (binding.r03, binding.physical, binding.r06, binding.r07) == (
        owner._r03,
        owner._physical,
        owner._r06,
        owner._r07,
    )
    view = owner.require_owned_full_mdp_pre_reward(
        publication,
        runtime_lease=owner._env_lease,
        control_step=7,
    )
    assert torch.equal(
        view.terminated,
        torch.tensor([False, True], device=owner._device),
    )
    view.terminated.fill_(False)
    assert owner.require_owned_full_mdp_pre_reward(
        publication,
        runtime_lease=owner._env_lease,
        control_step=7,
    ).terminated.tolist() == [False, True]
    close = owner.close_full_mdp_reward_cycle(
        **_reward_close_args(owner, publication)
    )
    assert type(close) is M.ActionBallFullMdpRewardCloseReceipt
    assert owner._active_pre_reward_publication is None
    assert [row[:2] for row in events if row[0] == "reward_publish"] == [
        ("reward_publish", "r03"),
        ("reward_publish", "r07"),
    ]
    assert [row[1] for row in events if row[0] == "reward_close"] == [
        "r03",
        "physical",
        "r06",
        "r07",
    ]


def test_reward_binding_wrong_order_missing_method_and_repeat_hold(monkeypatch):
    owner, _ = _reward_runtime()
    wrong = tuple(reversed(M.FULL_MDP_REWARD_ORDERED_CONSUMERS))
    with pytest.raises(M.ActionBallFullMdpRuntimeDependencyError, match="order"):
        owner.bind_full_mdp_reward_owners(
            runtime_lease=owner._env_lease,
            ordered_consumers=wrong,
            reward_graph=object(),
        )
    monkeypatch.setattr(
        owner._physical, "require_owned_full_mdp_reward_payment", None
    )
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="exact fresh graph",
    ):
        owner.bind_full_mdp_reward_owners(
            runtime_lease=owner._env_lease,
            ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
            reward_graph=object(),
        )


def test_reward_publish_requires_final_postphysics_and_exact_identity():
    owner, _ = _reward_runtime()
    graph = RW.FreshFullMdpRewardGraph(
        runtime_owner=owner,
        runtime_lease=owner._env_lease,
        owners=RW._RewardOwners(
            r03=owner._r03,
            physical=owner._physical,
            r06=owner._r06,
            r07=owner._r07,
            num_envs=owner._num_envs,
            device=torch.device(owner._device),
        ),
        diagnostic_unauthorized=False,
        _construction_token=RW._PRODUCTION_GRAPH_CONSTRUCTION_TOKEN,
    )
    owner.bind_full_mdp_reward_owners(
        runtime_lease=owner._env_lease,
        ordered_consumers=M.FULL_MDP_REWARD_ORDERED_CONSUMERS,
        reward_graph=graph,
    )
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError, match="final postphysics"
    ):
        owner.publish_full_mdp_pre_reward(
            runtime_lease=owner._env_lease,
            control_step=7,
        )
    owner._last_final_postphysics_control_step = 7
    publication = owner.publish_full_mdp_pre_reward(
        runtime_lease=owner._env_lease,
        control_step=7,
    )
    for candidate, step, lease in (
        (object(), 7, owner._env_lease),
        (publication, 8, owner._env_lease),
        (publication, 7, object()),
    ):
        with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError, match="foreign"):
            owner.require_owned_full_mdp_pre_reward(
                candidate,
                runtime_lease=lease,
                control_step=step,
            )


@pytest.mark.parametrize("mutation", ["skip", "duplicate", "foreign"])
def test_reward_bad_verdict_graph_poison_blocks_reset_and_optimizer(mutation):
    owner, _ = _reward_runtime()
    _, publication = _bind_and_publish_reward(owner)
    kwargs = _reward_close_args(owner, publication)
    if mutation == "skip":
        kwargs["ordered_owner_payment_results"] = kwargs[
            "ordered_owner_payment_results"
        ][:-1]
    elif mutation == "duplicate":
        verdicts = list(kwargs["ordered_owner_payment_results"])
        verdicts[-1] = verdicts[0]
        kwargs["ordered_owner_payment_results"] = tuple(verdicts)
    else:
        verdicts = list(kwargs["ordered_owner_payment_results"])
        verdicts[3] = object()
        kwargs["ordered_owner_payment_results"] = tuple(verdicts)
    with pytest.raises((M.ActionBallFullMdpRuntimeOwnerError, AssertionError)):
        owner.close_full_mdp_reward_cycle(**kwargs)
    assert owner._active_pre_reward_publication is publication
    assert owner.poisoned
    with pytest.raises(M.ActionBallFullMdpRuntimePoisonedError):
        owner.selected_true_reset(object())
    with pytest.raises(M.ActionBallFullMdpRuntimePoisonedError):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0, completed_environment_steps=1
        )


def test_reward_mid_close_exception_keeps_debt_and_poison():
    owner, events = _reward_runtime()
    owner._r06.fail_close = True
    _, publication = _bind_and_publish_reward(owner)
    with pytest.raises(RuntimeError, match="r06 close failed"):
        owner.close_full_mdp_reward_cycle(
            **_reward_close_args(owner, publication)
        )
    assert [row[1] for row in events if row[0] == "reward_close"] == [
        "r03",
        "physical",
        "r06",
    ]
    assert owner._active_pre_reward_publication is publication
    assert owner.poisoned


def test_reward_cycle_debt_blocks_reset_before_any_payment_close():
    owner, _ = _reward_runtime()
    _, publication = _bind_and_publish_reward(owner)
    with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError, match="Reward transaction"):
        owner._require_no_selected_reset_debt(operation="selected reset")
    assert owner._active_pre_reward_publication is publication


def test_direct_reveal_is_tombstoned_before_request_inspection():
    owner = _runtime_shell()
    exploding = _ExplodingOwner()
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="direct reveal execution is a tombstone",
    ):
        owner.execute_reveal(exploding)


def test_before_policy_step_holds_before_legacy_request_or_action_inspection():
    owner = _runtime_shell()
    exploding = _ExplodingOwner()
    with pytest.raises(
        M.ActionBallFullMdpRuntimeDependencyError,
        match="children do not consume Device-R05 hot tokens",
    ):
        owner.before_policy_step(exploding, exploding)

    motion_required = next(
        spec.required_methods
        for spec in M._DEPENDENCY_SPECS
        if spec.role == "motion_child"
    )
    assert "prepare_action_ball_full_mdp_reveal_request" not in motion_required
    assert (
        "require_owned_action_ball_full_mdp_reveal_request"
        not in motion_required
    )


def test_device_r05_inventory_names_hot_reveal_and_selected_reset_surface():
    required = next(
        spec.required_methods
        for spec in M._DEPENDENCY_SPECS
        if spec.role == "device_r05_owner"
    )
    assert required == (
        "project_owned_genesis_for_child",
        "require_owned_genesis_projection",
        "project_full_mdp_env_reset_binding",
        "require_owned_full_mdp_env_reset_binding",
        "bind_true_reset_authority",
        "prepare_many",
        "preview",
        "abort_prepared",
        "abort_preview",
        "stage_terminal",
        "arm_terminal",
        "commit_terminal",
        "record_child_completion",
        "poison_from_external_failure",
        "require_healthy",
        "prepare_true_reset_many",
        "require_owned_prepared_true_reset",
        "abort_true_reset_many",
        "commit_true_reset_many",
        "require_owned_true_reset_receipt",
    )


def test_dependency_inventory_exposes_physical_provider_cross_pin_blocker():
    inventory = M.action_ball_full_mdp_runtime_dependency_inventory()
    assert M.PROVIDER_API_SCHEMA_SHA256 == (
        "dc1cc5540bd73612e9677930bba14d83a809dd43c7c68fd48795642b40aa92d2"
    )
    assert any(
        "physical_checkpoint_adapter" in blocker
        for blocker in inventory.blockers
    )


def test_direct_runtime_and_provider_construction_are_tombstoned():
    with pytest.raises(TypeError, match="use ActionBallFullMdpRuntimeOwner.create"):
        M.ActionBallFullMdpRuntimeOwner()
    with pytest.raises(
        M.ActionBallFullMdpRuntimeOwnerError,
        match="runtime-owner constructed only",
    ):
        M.ActionBallFullMdpCheckpointJoinSnapshotProvider(
            num_envs=1,
            runtime_owner_identity=object(),
            runtime_owner=object(),
            ppo_drain_owner=object(),
            checkpoint_module=object(),
            _token=object(),
        )


def test_opaque_snapshot_type_has_no_public_constructor():
    with pytest.raises(TypeError, match="provider-issued only"):
        M.ActionBallFullMdpCheckpointJoinSnapshot()


def test_provider_api_schema_is_stable_and_does_not_reverse_pin_source_bytes():
    source_bytes = SOURCE.read_bytes()
    assert M.PROVIDER_API_SCHEMA_SHA256 == (
        M.ActionBallFullMdpCheckpointJoinSnapshotProvider.API_SCHEMA_SHA256
    )
    assert M.PROVIDER_API_SCHEMA_SHA256 == M._provider_api_schema_sha256(
        source_bytes
    )
    assert len(M.PROVIDER_API_SCHEMA_SHA256) == 64
    changed = source_bytes.replace(
        b'if owner_id != "env.ball_physical":',
        b'if owner_id not in ("env.ball_physical",):',
        1,
    )
    assert changed != source_bytes
    assert M._provider_api_schema_sha256(changed) != (
        M.PROVIDER_API_SCHEMA_SHA256
    )
    changed_snapshot_slots = source_bytes.replace(
        b'    __slots__ = ("__weakref__",)\n',
        b'    __slots__ = ("__weakref__", "_unexpected")\n',
        1,
    )
    assert changed_snapshot_slots != source_bytes
    assert M._provider_api_schema_sha256(changed_snapshot_slots) != (
        M.PROVIDER_API_SCHEMA_SHA256
    )
    source = source_bytes.decode("utf-8")
    assert "source_pin_direction" in source
    assert "runtime_owner_to_physical_only" in source


def test_runtime_source_has_no_numeric_host_transfer_or_legacy_commit_calls():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_attributes = {"item", "cpu", "tolist", "numpy"}
    observed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    }
    assert observed == set()
    forbidden_calls = {
        "commit_many",
        "censor_many",
        "arm_preview_for_all_owner",
        "commit_prevalidated",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden_calls)


def test_dependency_inventory_content_root_changes_with_semantic_rows(
    monkeypatch,
):
    first = M.action_ball_full_mdp_runtime_dependency_inventory()
    specs = list(M._DEPENDENCY_SPECS)
    owner_index = next(
        index
        for index, spec in enumerate(specs)
        if spec.role == "r05_reveal_owner"
    )
    specs[owner_index] = replace(
        specs[owner_index],
        expected_api_sha256="0" * 64,
    )
    monkeypatch.setattr(M, "_DEPENDENCY_SPECS", tuple(specs))
    changed = M.action_ball_full_mdp_runtime_dependency_inventory()
    assert first.rows != changed.rows
    assert first.content_sha256 != changed.content_sha256
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() != first.content_sha256


def test_checkpoint_api_pin_covers_boundary_fields_not_an_empty_surface():
    source = CHECKPOINT_SOURCE.read_bytes()
    changed = source.replace(
        b"    reset_in_flight: bool\n",
        b"    reset_in_flight: int\n",
        1,
    )
    assert changed != source
    original_sha = M._method_surface_sha256(
        source,
        class_name="CheckpointBoundary",
        method_names=(),
        field_names=next(
            spec.required_fields
            for spec in M._DEPENDENCY_SPECS
            if spec.role == "r10_checkpoint_contract"
        ),
    )
    changed_sha = M._method_surface_sha256(
        changed,
        class_name="CheckpointBoundary",
        method_names=(),
        field_names=next(
            spec.required_fields
            for spec in M._DEPENDENCY_SPECS
            if spec.role == "r10_checkpoint_contract"
        ),
    )
    assert original_sha != changed_sha


def test_provider_retains_one_exact_snapshot_and_rejects_wrong_authority():
    provider = _provider_with_state()
    boundary = _boundary()
    snapshot = provider.snapshot_for_checkpoint_boundary(boundary)
    assert provider.snapshot_for_checkpoint_boundary(boundary) is snapshot
    assert provider.require_owned_snapshot(
        boundary,
        snapshot,
        snapshot.canonical_sha256,
    ) is snapshot
    audit_claim = provider.prepare_checkpoint_audit_claim(
        boundary,
        snapshot,
        "env.ball_physical",
    )
    claims = provider.checkpoint_join_claims(
        boundary,
        snapshot,
        "env.ball_physical",
        audit_claim,
    )
    assert tuple(type(row) for row in claims) == (
        C.OwnerJoinClaim,
        C.OwnerJoinClaim,
        C.OwnerJoinClaim,
    )
    assert tuple(row.join_id for row in claims) == (
        "per_world_reset_identity",
        "task_ball_r06_current",
        "ppo_drain_frontier",
    )
    assert tuple(row.value_sha256 for row in claims) == (
        snapshot.per_world_reset_identity,
        snapshot.task_ball_r06_current,
        snapshot.ppo_drain_frontier,
    )

    with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError):
        provider.require_owned_snapshot(boundary, snapshot, "0" * 64)
    forged = object.__new__(M.ActionBallFullMdpCheckpointJoinSnapshot)
    with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError):
        provider.require_owned_snapshot(
            boundary,
            forged,
            snapshot.canonical_sha256,
        )
    with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError):
        provider.snapshot_for_checkpoint_boundary(
            replace(boundary, worlds=(_world(reset_generation=4),))
        )


def test_provider_requires_exact_sequence_and_per_world_reset_history():
    provider = _provider_with_state(reset_generation=3)
    original_reset_label = "reset:0:3"

    rejected = (
        _join_state(
            provider,
            sequence=3,
            reset_generation=3,
            reset_identity_label=original_reset_label,
        ),
        _join_state(
            provider,
            sequence=2,
            reset_generation=2,
        ),
        _join_state(
            provider,
            sequence=2,
            reset_generation=3,
            reset_identity_label="same-generation-substitution",
        ),
        _join_state(
            provider,
            sequence=2,
            reset_generation=4,
            reset_identity_label=original_reset_label,
        ),
    )
    for state in rejected:
        with pytest.raises(M.ActionBallFullMdpRuntimeOwnerError):
            _publish_state(provider, state)

    same_generation = _join_state(
        provider,
        sequence=2,
        reset_generation=3,
        reset_identity_label=original_reset_label,
        current_label="task-current-advanced-without-reset",
    )
    _publish_state(provider, same_generation)
    next_generation = _join_state(
        provider,
        sequence=3,
        reset_generation=4,
        reset_identity_label="reset:0:4",
    )
    _publish_state(provider, next_generation)
    assert provider.current_runtime_join_state_sha256 == (
        next_generation.canonical_sha256
    )


def test_checkpoint_claim_validation_and_mint_share_one_provider_lock(
    monkeypatch,
):
    provider = _provider_with_state(reset_generation=3)
    boundary = _boundary(reset_generation=3)
    snapshot = provider.snapshot_for_checkpoint_boundary(boundary)
    audit_claim = provider.prepare_checkpoint_audit_claim(
        boundary,
        snapshot,
        "env.ball_physical",
    )
    original_claim_type = C.OwnerJoinClaim
    claim_constructor_entered = threading.Event()
    release_claim_constructor = threading.Event()
    publisher_started = threading.Event()
    publisher_finished = threading.Event()
    claim_result = []
    claim_errors = []

    def blocking_claim(*, join_id, value_sha256):
        claim_constructor_entered.set()
        assert release_claim_constructor.wait(timeout=5.0)
        return original_claim_type(
            join_id=join_id,
            value_sha256=value_sha256,
        )

    monkeypatch.setattr(C, "OwnerJoinClaim", blocking_claim)

    def read_claims():
        try:
            claim_result.extend(
                    provider.checkpoint_join_claims(
                        boundary,
                        snapshot,
                        "env.ball_physical",
                        audit_claim,
                )
            )
        except BaseException as exc:
            claim_errors.append(exc)

    def publish_next_state():
        publisher_started.set()
        _publish_state(
            provider,
            _join_state(
                provider,
                sequence=2,
                reset_generation=3,
                reset_identity_label="reset:0:3",
            ),
        )
        publisher_finished.set()

    reader = threading.Thread(target=read_claims)
    reader.start()
    assert claim_constructor_entered.wait(timeout=5.0)
    publisher = threading.Thread(target=publish_next_state)
    publisher.start()
    assert publisher_started.wait(timeout=5.0)
    assert not publisher_finished.wait(timeout=0.1)
    release_claim_constructor.set()
    reader.join(timeout=5.0)
    publisher.join(timeout=5.0)

    assert not reader.is_alive()
    assert not publisher.is_alive()
    assert claim_errors == []
    assert publisher_finished.is_set()
    assert tuple(row.value_sha256 for row in claim_result) == (
        snapshot.per_world_reset_identity,
        snapshot.task_ball_r06_current,
        snapshot.ppo_drain_frontier,
    )
