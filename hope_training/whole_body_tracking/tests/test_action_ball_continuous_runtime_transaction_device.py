from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import action_ball_continuous_runtime_transaction_device as r05  # noqa: E402
import action_ball_continuous_target_sampler as c03  # noqa: E402
_GLOBAL_DRAIN_PATH = (
    _WBT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking"
    / "tasks" / "tracking" / "mdp" / "action_ball_full_mdp_ppo_drain.py"
)
if str(_GLOBAL_DRAIN_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_GLOBAL_DRAIN_PATH.parent))
_DRAIN_NAME = "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_ppo_drain"
_PACKAGE_PATHS = (
    ("whole_body_tracking", _SOURCE_ROOT / "whole_body_tracking"),
    (
        "whole_body_tracking.tasks",
        _SOURCE_ROOT / "whole_body_tracking" / "tasks",
    ),
    (
        "whole_body_tracking.tasks.tracking",
        _SOURCE_ROOT / "whole_body_tracking" / "tasks" / "tracking",
    ),
    ("whole_body_tracking.tasks.tracking.mdp", _GLOBAL_DRAIN_PATH.parent),
)
for _package, _package_path in _PACKAGE_PATHS:
    _module = sys.modules.get(_package)
    if _module is None:
        _module = types.ModuleType(_package)
        sys.modules[_package] = _module
    _paths = list(getattr(_module, "__path__", ()))
    if str(_package_path) not in _paths:
        _module.__path__ = [*_paths, str(_package_path)]
global_drain = sys.modules.get(_DRAIN_NAME)
if global_drain is None:
    _DRAIN_SPEC = importlib.util.spec_from_file_location(
        _DRAIN_NAME, _GLOBAL_DRAIN_PATH
    )
    assert _DRAIN_SPEC is not None and _DRAIN_SPEC.loader is not None
    global_drain = importlib.util.module_from_spec(_DRAIN_SPEC)
    sys.modules[_DRAIN_NAME] = global_drain
    _DRAIN_SPEC.loader.exec_module(global_drain)
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_full_mdp_ppo_drain",
    global_drain,
)
_EPOCH_PATH = (
    _WBT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking"
    / "tasks" / "tracking" / "mdp" / "action_ball_full_mdp_epoch.py"
)
_EPOCH_NAME = "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_epoch"
epoch = sys.modules.get(_EPOCH_NAME) or sys.modules.get(
    "action_ball_full_mdp_epoch"
)
if epoch is None:
    _EPOCH_SPEC = importlib.util.spec_from_file_location(_EPOCH_NAME, _EPOCH_PATH)
    assert _EPOCH_SPEC is not None and _EPOCH_SPEC.loader is not None
    epoch = importlib.util.module_from_spec(_EPOCH_SPEC)
    sys.modules[_EPOCH_NAME] = epoch
    _EPOCH_SPEC.loader.exec_module(epoch)
sys.modules["action_ball_full_mdp_epoch"] = epoch
sys.modules[_EPOCH_NAME] = epoch
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_full_mdp_epoch",
    epoch,
)
_CARRY_NAME = (
    "whole_body_tracking.tasks.tracking.mdp."
    "action_ball_full_mdp_lean_checkpoint_txn"
)
carry = getattr(epoch, "carry_txn", None)
if carry is None:
    carry = sys.modules.get(_CARRY_NAME) or sys.modules.get(
        "action_ball_full_mdp_lean_checkpoint_txn"
    )
if carry is None:
    _CARRY_SPEC = importlib.util.spec_from_file_location(
        _CARRY_NAME,
        _GLOBAL_DRAIN_PATH.parent / "action_ball_full_mdp_lean_checkpoint_txn.py",
    )
    assert _CARRY_SPEC is not None and _CARRY_SPEC.loader is not None
    carry = importlib.util.module_from_spec(_CARRY_SPEC)
    sys.modules[_CARRY_NAME] = carry
    _CARRY_SPEC.loader.exec_module(carry)
sys.modules[_CARRY_NAME] = carry
sys.modules["action_ball_full_mdp_lean_checkpoint_txn"] = carry
setattr(
    sys.modules["whole_body_tracking.tasks.tracking.mdp"],
    "action_ball_full_mdp_lean_checkpoint_txn",
    carry,
)




def _profile():
    return c03.ContinuousTargetProfile(
        frame_id="hope_world_table_xy_m",
        frame_binding_sha256="a" * 64,
        runtime_dtype=c03.RUNTIME_DTYPE,
        quantization_contract=c03.QUANTIZATION_CONTRACT,
        components=("landing_x_m", "landing_y_m"),
        cells=(
            c03.TargetCell("near_left", (2.10, -0.20)),
            c03.TargetCell("near_right", (2.10, 0.20)),
            c03.TargetCell("deep_center", (2.80, 0.0)),
        ),
    )


class _ProfileAuthority:
    def __init__(self, device):
        portable = _profile()
        self.receipt = object()
        cell_ids = tuple(cell.cell_id for cell in portable.cells)
        semantics = tuple(
            portable.semantic_sha256(cell) for cell in portable.cells
        )
        targets_host = tuple(tuple(cell.target) for cell in portable.cells)
        import hashlib
        import struct

        digest = hashlib.sha256()
        digest.update(portable.profile_sha256.encode("ascii"))
        for cell_id, semantic in zip(cell_ids, semantics):
            digest.update(len(cell_id).to_bytes(8, "big"))
            digest.update(cell_id.encode("utf-8"))
            digest.update(bytes.fromhex(semantic))
        for row in targets_host:
            for value in row:
                digest.update(struct.pack(">f", value))
        self.projection = r05.DeviceProfileProjection(
            profile_sha256=portable.profile_sha256,
            profile_binding_sha256=digest.hexdigest(),
            cell_ids=cell_ids,
            semantic_sha256s=semantics,
            targets_xy_m=torch.tensor(
                targets_host, dtype=torch.float32, device=device
            ),
        )

    def require_owned_r05_profile(self, receipt):
        assert receipt is self.receipt
        return self.projection


class _Genesis:
    def __init__(self, device, n, values=None):
        self.receipt = object()
        self.projection = r05.DeviceGenesisProjection(
            world_reset_identity=object(),
            reset_generations=torch.tensor(
                values if values is not None else (1,) * n,
                dtype=torch.int64,
                device=device,
            ),
        )

    def require_owned_r05_genesis(self, receipt, *, device, num_envs):
        assert receipt is self.receipt
        assert self.projection.reset_generations.device == device
        assert self.projection.reset_generations.shape == (num_envs,)
        return self.projection


class _Cadence:
    """Construction-bound fixed-N Motion source; no caller row selection."""

    def __init__(self, device, n):
        self.device = torch.device(device)
        self.n = n

    def project_current_action_epoch_rows(self):
        raise RuntimeError("cold-only fixture has no armed Motion tick")


class _Question:
    """Cold-only Question binding with no compact/public construction seam."""

    def __init__(self, device):
        self.device = torch.device(device)

    def project_r05_candidate_bank(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("cold-only fixture cannot compose a candidate")


class _Reveal:
    """Legacy boundary shell retained only for cold/reset invariants."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.children = None
        self.owner = None

    def bind_children(self, children):
        assert self.children is None
        self.children = children

    def bind_owner(self, owner):
        assert self.owner is None
        self.owner = owner

    def project_owned_r05_reveal_boundary(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("cold-only fixture has no reveal")

    def require_owned_r05_terminal_arm(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("cold-only fixture has no terminal arm")

    def require_owned_r05_terminal_commit(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("cold-only fixture has no terminal commit")


class _Child:
    def __init__(self, kind):
        self.kind = kind

    def require_owned_r05_child_completion(self, receipt):
        del receipt
        raise RuntimeError("cold-only fixture has no child completion")


class _Drain:
    def __init__(self):
        self.materialize_calls = 0


class _Reset:
    def __init__(self, device, n):
        self.device = device
        self.n = n
        self.events = {}
        self.committable = {}
        self.abortable = {}
        self.preflights = {}
        self.overflow = False
        self.owner = None
        self.project_calls = 0

    def bind_owner(self, owner):
        self.owner = owner

    def issue(self, owner, ids):
        receipt = object()
        index = torch.tensor(ids, dtype=torch.int64, device=self.device)
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        # Keep adversarial raw indices intact without letting the diagnostic
        # authority itself launch an invalid CUDA indexing kernel.
        valid_ids = tuple(value for value in ids if 0 <= value < self.n)
        if valid_ids:
            mask[
                torch.tensor(
                    valid_ids, dtype=torch.int64, device=self.device
                )
            ] = True
        self.events[receipt] = r05.DeviceTrueResetEventProjection(
            reset_event_identity=object(),
            selected_env_index=index,
            selected_mask=mask,
        )
        return receipt

    def project_r05_true_reset(
        self, receipt, *, device, num_envs, live_reset_ledger_identity,
        live_reset_generation
    ):
        del device, num_envs, live_reset_ledger_identity, live_reset_generation
        self.project_calls += 1
        return self.events[receipt]

    def allow_commit(self, prepared):
        self.committable[prepared] = self.owner.require_owned_prepared_true_reset(
            prepared, owner_kind="motion"
        ).reset_event_identity
        capability = object()
        self.preflights[prepared] = capability
        self.owner.register_true_reset_preflight(prepared, capability)

    def require_owned_r05_true_reset_preflight(
        self, prepared, *, preflight_capability
    ):
        assert self.preflights[prepared] is preflight_capability
        return r05.DeviceTrueResetPreflightProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.committable[prepared],
            preflight_capability=preflight_capability,
        )

    def allow_abort(self, prepared):
        self.abortable[prepared] = self.owner.require_owned_prepared_true_reset(
            prepared, owner_kind="motion"
        ).reset_event_identity

    def require_owned_r05_true_reset_commit(
        self, prepared, *, owner_view
    ):
        assert prepared in self.committable
        assert owner_view.prepared_true_reset is prepared
        if bool(torch.any(owner_view.generation_overflow_fault)):
            raise RuntimeError("reset generation overflow")
        return r05.DeviceTrueResetCommitProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.committable[prepared],
            child_kinds=r05.CHILD_OWNER_ORDER,
            child_commit_identities=tuple(object() for _ in r05.CHILD_OWNER_ORDER),
            preflight_capability=self.preflights[prepared],
        )

    def require_owned_r05_true_reset_abort(self, prepared):
        assert prepared in self.abortable
        return r05.DeviceTrueResetAbortProjection(
            prepared_true_reset=prepared,
            reset_event_identity=self.abortable[prepared],
            child_commits_started=False,
        )

    def issue_child_completion(self, receipt, kind):
        child_receipt = object()
        self.committable[(receipt, kind, child_receipt)] = True
        return child_receipt

    def require_owned_r05_true_reset_child_completion(
        self, receipt, *, child_kind, child_receipt
    ):
        assert self.committable[(receipt, child_kind, child_receipt)] is True
        return r05.DeviceTrueResetChildCompletionProjection(
            true_reset_receipt=receipt,
            child_kind=child_kind,
            child_receipt=child_receipt,
        )


class _Checkpoint:
    def __init__(self, checkpoint):
        self.receipt = object()
        self.checkpoint = checkpoint

    def require_owned_r05_device_checkpoint(self, receipt):
        assert receipt is self.receipt
        return self.checkpoint


class _GlobalPeerLeaf:
    def __init__(self, owner_kind, device):
        self.owner_kind = owner_kind
        self.device = device
        self.terminal_total = 0
        self.active = None
        self.poisoned = False

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self, *, authority, update_index, completed_environment_steps
    ):
        del update_index, completed_environment_steps
        values = []
        for name in authority.field_names:
            if name == "terminal_resolution_total":
                value = self.terminal_total
            else:
                value = 0
            values.append(value)
        tensor = torch.tensor(values, dtype=torch.int64, device=self.device)
        pack = authority.mint_device_pack(leaf=self, values=tensor)
        self.active = (authority, pack)
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack):
        assert self.active is not None and pack is self.active[1]
        self.active = None

    def acknowledge_pre_optimizer_ppo_boundary(
        self, *, pack, receipt, owner_row
    ):
        authority, retained = self.active
        authority.require_owned_ack(
            leaf=self, pack=pack, receipt=receipt, owner_row=owner_row
        )
        assert pack is retained
        self.active = None

    def poison_pre_optimizer_ppo_boundary(self, *, reason):
        del reason
        self.poisoned = True


def _global_schema(h):
    schemas = list(global_drain.DEFAULT_LEAF_SCHEMAS)
    schemas[0] = r05.materialize_pre_optimizer_ppo_boundary_leaf_schema(
        leaf_schema_type=global_drain.LeafDrainSchema,
        field_spec_type=global_drain.DeviceDrainFieldSpec,
        journal_capacity=h.owner.journal_capacity,
        num_envs=h.owner.num_envs,
        support_size=h.owner.profile.support_size,
    )
    return tuple(schemas)


def _ensure_global(h):
    coordinator = getattr(h, "global_drain", None)
    if coordinator is not None:
        return coordinator
    peers = {
        name: _GlobalPeerLeaf(name, h.device)
        for name in global_drain.OWNER_ORDER[1:]
    }
    leaves = {"r05_runtime": h.owner, **peers}
    coordinator = global_drain.ActionBallFullMdpPpoDrainOwner(
        num_envs=h.owner.num_envs,
        device=h.device,
        leaves=leaves,
        leaf_schemas=_global_schema(h),
    )
    coordinator.require_exact_leaf_bindings(leaves)
    h.global_drain = coordinator
    h.global_peers = peers
    return coordinator


def _journal_view(h):
    start, end = h.owner._journal_tail, h.owner._journal_head
    parts, schema, offset = [], [], 0
    for name in r05._JOURNAL_FIELD_NAMES:
        tensor = h.owner._journal_source(name)
        flat = tensor.to(torch.int64).reshape(-1)
        schema.append((name, offset, offset + flat.numel(), tuple(tensor.shape)))
        parts.append(flat)
        offset += flat.numel()
    return r05.DeviceR05DrainView(
        drain_identity=object(), schema_version=2,
        num_envs=h.owner.num_envs, support_size=h.owner.profile.support_size,
        row_count=end - start, start_sequence=start, end_sequence=end,
        packed=torch.cat(parts), packed_schema=tuple(schema),
    )


@dataclass
class _Harness:
    owner: r05.DeviceR05Owner
    device: torch.device
    profile: _ProfileAuthority
    genesis: _Genesis
    cadence: _Cadence
    question: _Question
    reveal: _Reveal
    children: tuple[_Child, ...]
    drain: _Drain
    reset: _Reset


def _harness(
    n,
    *,
    device="cpu",
    seed=12345,
    capacity=64,
    max_epochs=64,
    cadence=None,
    question=None,
):
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    profile = _ProfileAuthority(dev)
    genesis = _Genesis(dev, n)
    cadence = _Cadence(dev, n) if cadence is None else cadence
    question = _Question(dev) if question is None else question
    reveal = _Reveal(dev)
    children = tuple(_Child(kind) for kind in r05.CHILD_OWNER_ORDER)
    reveal.bind_children(children)
    drain = _Drain()
    reset = _Reset(dev, n)
    owner = r05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=seed,
        num_envs=n,
        journal_capacity=capacity,
        max_reveal_epochs_per_drain=max_epochs,
        genesis_authority=genesis,
        genesis_receipt=genesis.receipt,
        cadence_authority=cadence,
        question_authority=question,
        reveal_boundary_authority=reveal,
        child_completion_authorities=children,
        true_reset_authority=reset,
    )
    reveal.bind_owner(owner)
    reset.bind_owner(owner)
    return _Harness(
        owner, dev, profile, genesis, cadence, question, reveal, children, drain, reset
    )


class _SingleRowFaultQuestion:
    """Internal round-bank mutation with faults confined to one row."""

    _SOURCE_FAULT = 1 << 48

    def __init__(self, bad_row):
        self.bad_row = bad_row

    def compose_r05_candidate_bank_inside_prepare(self, internal_context):
        (
            cadence_receipt,
            cadence,
            _profile_projection,
            device,
            support,
            _draw_u01,
            candidate_identity,
            _construction_mask,
            _previous_cell_index,
            bank_sequence,
        ) = r05._consume_internal_question_context(internal_context, self)
        rounds = r05.INTERNAL_QUESTION_REDRAW_ROUNDS
        prefix = (cadence.selected_count, rounds, support)
        mutated_identity = candidate_identity.clone()
        mutated_identity[self.bad_row, 1, 0] += 1
        source_fault = torch.zeros(
            cadence.selected_count, dtype=torch.int64, device=device
        )
        source_fault[self.bad_row] = self._SOURCE_FAULT
        attempted = torch.ones(
            (cadence.selected_count, 1, support),
            dtype=torch.bool,
            device=device,
        )
        attempted[self.bad_row] = False
        attempted = torch.cat(
            (
                attempted,
                torch.zeros(
                    (cadence.selected_count, rounds - 1, support),
                    dtype=torch.bool,
                    device=device,
                ),
            ),
            dim=1,
        )
        chronology = r05.DeviceQuestionRoundChronology(
            action_uid=torch.where(
                attempted,
                torch.ones(prefix, dtype=torch.int64, device=device),
                torch.full(prefix, -1, dtype=torch.int64, device=device),
            ).contiguous(),
            contact_tick=torch.where(
                attempted,
                torch.full(prefix, 12, dtype=torch.int64, device=device),
                torch.full(prefix, -1, dtype=torch.int64, device=device),
            ).contiguous(),
            launch_tick=torch.where(
                attempted,
                torch.full(prefix, 11, dtype=torch.int64, device=device),
                torch.full(prefix, -1, dtype=torch.int64, device=device),
            ).contiguous(),
            chosen_horizon_ticks=torch.where(
                attempted,
                torch.ones(prefix, dtype=torch.int64, device=device),
                torch.full(prefix, -1, dtype=torch.int64, device=device),
            ).contiguous(),
            task_close_tick=torch.where(
                attempted,
                torch.full(prefix, 20, dtype=torch.int64, device=device),
                torch.full(prefix, -1, dtype=torch.int64, device=device),
            ).contiguous(),
        )
        construction_reason = torch.full(
            prefix,
            r05.QUESTION_CONSTRUCTION_REASON_INVALID_PRODUCER,
            dtype=torch.int64,
            device=device,
        )
        construction_reason[:, 0] = (
            r05.QUESTION_CONSTRUCTION_REASON_ADMITTED
        )
        construction_reason[self.bad_row] = (
            r05.QUESTION_CONSTRUCTION_REASON_INVALID_PRODUCER
        )
        return r05.DeviceQuestionProjection(
            cadence_receipt_identity=cadence_receipt,
            bank_identity=object(),
            bank_sequence=bank_sequence,
            bank=None,
            producer_fault=source_fault.contiguous(),
            selected_count=cadence.selected_count,
            support_size=support,
            round_bank=r05.DeviceR05CandidateRoundBank(
                candidate_identity=mutated_identity.contiguous(),
                construction_reason=construction_reason.contiguous(),
                producer_fault=torch.zeros(
                    (cadence.selected_count, rounds),
                    dtype=torch.int64,
                    device=device,
                ),
                motion_task_f32=torch.zeros(
                    (*prefix, len(r05.MOTION_TASK_F32_FIELDS)),
                    dtype=torch.float32,
                    device=device,
                ),
                racket_task_f32=torch.zeros(
                    (*prefix, len(r05.RACKET_F32_FIELDS)),
                    dtype=torch.float32,
                    device=device,
                ),
                physical_state_f32=torch.zeros(
                    (*prefix, len(r05.PHYSICAL_STATE_F32_FIELDS)),
                    dtype=torch.float32,
                    device=device,
                ),
            ),
            round_chronology=chronology,
        )


def test_internal_question_one_bad_row_does_not_censor_peers():
    """Source and round mutations censor only their owning environment."""

    n, bad_row = 3, 1
    question = _SingleRowFaultQuestion(bad_row)
    h = _harness(n, question=question)
    cadence_receipt = object()
    i64_zero = torch.zeros(n, dtype=torch.int64, device=h.device)
    i64_one = torch.ones(n, dtype=torch.int64, device=h.device)
    cadence = r05.DeviceCadenceProjection(
        selected_count=n,
        selected_env_index=torch.arange(n, dtype=torch.int64, device=h.device),
        episode_tick=torch.full_like(i64_zero, 10),
        reveal_tick=torch.full_like(i64_zero, 10),
        deadline_tick=torch.full_like(i64_zero, 30),
        next_reveal_tick=torch.full_like(i64_zero, 100),
        swing_generation=i64_one.clone(),
        ready_at_reveal=torch.ones(n, dtype=torch.bool, device=h.device),
        action_slot=i64_zero.clone(),
        pending_elapsed_s=torch.zeros(
            n, dtype=torch.float32, device=h.device
        ),
        reset_generation=i64_one.clone(),
        scheduled_ordinal=i64_zero.clone(),
        outcome_shot_index=i64_one.clone(),
        sampler_generation=i64_zero.clone(),
        task_identity=torch.full_like(i64_zero, -1),
        cadence_identity=torch.full_like(i64_zero, -1),
        cadence_producer_fault=i64_zero.clone(),
        cadence_owner_receipt_identity=cadence_receipt,
    )
    callback_started = [False]
    prepared_token = h.owner._prepare_many_impl(
        cadence_receipt,
        question_receipt=None,
        internal_question_compose=(
            question.compose_r05_candidate_bank_inside_prepare
        ),
        internal_callback_started=callback_started,
        owned_projection=cadence,
        construction_mask=torch.ones(
            n, dtype=torch.bool, device=h.device
        ),
        transaction_ordinal=0,
    )
    prepared = h.owner._require_prepared(prepared_token)

    assert callback_started == [True]
    assert prepared.owner_fault_free.tolist() == [True, False, True]
    assert prepared.question_producer_fault[[0, 2]].eq(0).all()
    assert (
        prepared.question_producer_fault[bad_row]
        == question._SOURCE_FAULT | r05.PRODUCER_FAULT_QUESTION_CHRONOLOGY
    )

    preview_token = h.owner._preview_impl(prepared_token)
    preview = h.owner._require_preview(preview_token)
    all_rows = torch.ones(n, dtype=torch.bool, device=h.device)
    transaction = h.owner._build_row_transaction(
        object.__new__(r05.DeviceR05RowTransaction),
        epoch.ActionEpochDueRows(0, all_rows, all_rows.clone()),
        prepared,
        preview,
    )
    assert transaction.accept_mask.tolist() == [True, False, True]
    assert transaction.censor_mask.tolist() == [False, True, False]
    r05_fault = transaction.candidate.owner_fault_bits[
        :, 0, epoch.OWNER_ORDER.index("r05_runtime")
    ]
    assert r05_fault[[0, 2]].eq(0).all()
    assert r05_fault[bad_row].ne(0)


def _drain_ack(h):
    coordinator = _ensure_global(h)
    for peer in h.global_peers.values():
        peer.terminal_total = sum(
            int(h.owner._journal_source("meta")[slot, h.owner._META_SELECTED_COUNT])
            for slot in range(h.owner.journal_capacity)
            if int(h.owner._journal_source("meta")[slot, h.owner._META_OPERATION])
            in (r05.JOURNAL_ACCEPT, r05.JOURNAL_CENSOR)
        )
    view = _journal_view(h)
    assert view.packed.ndim == 1 and view.packed.is_contiguous()
    prepared = coordinator.prepare_pre_optimizer_ppo_boundary(
        update_index=coordinator.next_update_index,
        completed_environment_steps=(coordinator.next_update_index + 1) * 48,
    )
    receipt = coordinator.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    coordinator.mark_optimizer_returned(receipt)
    coordinator.acknowledge_post_update(receipt)
    return view


def _prepare_global_receipt(h, *, terminal_total):
    """Prepare and perform the coordinator-owned sole D2H for a test row."""

    coordinator = _ensure_global(h)
    for peer in h.global_peers.values():
        peer.terminal_total = terminal_total
    prepared = coordinator.prepare_pre_optimizer_ppo_boundary(
        update_index=coordinator.next_update_index,
        completed_environment_steps=(coordinator.next_update_index + 1) * 48,
    )
    receipt = coordinator.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner_row = next(
        row
        for row in receipt.owner_rows
        if row.owner_kind == r05.GLOBAL_DRAIN_OWNER_KIND
    )
    return coordinator, prepared, receipt, owner_row


def test_action_epoch_owned_d05_rejects_the_exact_legacy_global_drain_type():
    current_epoch = r05._require_action_epoch_module()
    device = torch.device("cpu")
    epoch_owner = current_epoch.ActionEpochOwner(
        num_envs=2,
        device=device,
        shot_slot_capacity=1,
        initial_reset_generation=1,
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.ones(2, dtype=torch.int64),
    )
    profile, genesis = _ProfileAuthority(device), _Genesis(device, 2)
    owner = r05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=20260804,
        num_envs=2,
        journal_capacity=64,
        max_reveal_epochs_per_drain=64,
        genesis_authority=genesis,
        genesis_receipt=genesis.receipt,
        cadence_authority=_Cadence(device, 2),
        question_authority=_Question(device),
        diagnostic_epoch_owner=epoch_owner,
    )
    schema = r05.materialize_pre_optimizer_ppo_boundary_leaf_schema(
        leaf_schema_type=global_drain.LeafDrainSchema,
        field_spec_type=global_drain.DeviceDrainFieldSpec,
        journal_capacity=owner.journal_capacity,
        num_envs=owner.num_envs,
        support_size=owner.profile.support_size,
    )
    authority = global_drain.LeafDevicePackAuthority(
        owner_kind=r05.GLOBAL_DRAIN_OWNER_KIND,
        schema=schema,
        device=owner.device,
        num_envs=owner.num_envs,
        leaf=owner,
    )
    with pytest.raises(
        r05.DeviceR05ConflictError, match="rejects a second global drain"
    ):
        owner.prepare_pre_optimizer_ppo_boundary_device_pack(
            authority=authority,
            update_index=0,
            completed_environment_steps=2,
        )
    assert owner._active_drain is None
    assert owner._journal_head == owner._journal_tail == 0


@pytest.mark.parametrize("runtime_device", ("cpu", "cuda"))
@pytest.mark.parametrize("mutation", ("empty", "reward_debt"))
def test_direct_action_epoch_constructor_requires_exact_genesis_idle(
    runtime_device, mutation
):
    current_epoch = r05._require_action_epoch_module()
    if runtime_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(runtime_device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    epoch_owner = current_epoch.ActionEpochOwner(
        num_envs=2,
        device=device,
        shot_slot_capacity=1,
        initial_reset_generation=1,
    )
    if mutation != "empty":
        epoch_owner.activate_reset_genesis(
            selected_mask=torch.ones(2, dtype=torch.bool, device=device),
            reset_generation=torch.ones(2, dtype=torch.int64, device=device),
        )
        epoch_owner.open_reward_cycle()
    with pytest.raises(r05.DeviceR05Error, match="canonical genesis IDLE"):
        r05._require_canonical_action_epoch_idle(
            epoch_owner,
            epoch_module=current_epoch,
            device=device,
            num_envs=2,
        )


def test_checkpoint_restore_requires_same_numeric_reset_genesis_afterimage():
    left = _harness(1, seed=78)
    _drain_ack(left)
    checkpoint = left.owner.checkpoint_device()
    authority = _Checkpoint(checkpoint)
    right_parts = _harness(1, seed=78)
    right_parts.genesis.projection = r05.DeviceGenesisProjection(
        world_reset_identity=object(),
        reset_generations=checkpoint.reset_generation.clone() + 1,
    )
    with pytest.raises(r05.DeviceR05ConflictError, match="fresh genesis"):
        r05.DeviceR05Owner.from_device_checkpoint(
            profile_authority=right_parts.profile,
            profile_receipt=right_parts.profile.receipt,
            checkpoint_authority=authority,
            authority_receipt=authority.receipt,
            genesis_authority=right_parts.genesis,
            genesis_receipt=right_parts.genesis.receipt,
            cadence_authority=right_parts.cadence,
            question_authority=right_parts.question,
            reveal_boundary_authority=right_parts.reveal,
            child_completion_authorities=right_parts.children,
            drain_authority=right_parts.drain,
            true_reset_authority=right_parts.reset,
        )


@pytest.mark.parametrize("mutation", ("binding", "nan"))
def test_profile_authority_cannot_self_attest_wrong_or_nonfinite_content(mutation):
    dev = torch.device("cpu")
    profile = _ProfileAuthority(dev)
    if mutation == "binding":
        profile.projection = r05.DeviceProfileProjection(
            **{
                **profile.projection.__dict__,
                "profile_binding_sha256": "f" * 64,
            }
        )
        match = "content binding"
    else:
        targets = profile.projection.targets_xy_m.clone()
        targets[0, 0] = float("nan")
        profile.projection = r05.DeviceProfileProjection(
            **{**profile.projection.__dict__, "targets_xy_m": targets}
        )
        match = "structural binding"
    parts = _harness(1)
    with pytest.raises(r05.DeviceR05Error, match=match):
        r05.DeviceR05Owner(
            profile,
            profile.receipt,
            seed=1,
            num_envs=1,
            journal_capacity=2,
            max_reveal_epochs_per_drain=2,
            genesis_authority=parts.genesis,
            genesis_receipt=parts.genesis.receipt,
            cadence_authority=parts.cadence,
            question_authority=parts.question,
            reveal_boundary_authority=parts.reveal,
            child_completion_authorities=parts.children,
            drain_authority=parts.drain,
            true_reset_authority=parts.reset,
        )


def test_checkpoint_rejects_noncontinuable_int64_frontier():
    h = _harness(1)
    _drain_ack(h)
    h.owner._epoch = (1 << 63) - 1
    with pytest.raises(r05.DeviceR05ConflictError, match="continuation frontier"):
        h.owner.checkpoint_device()


def test_true_reset_generation_max_is_exposed_before_any_commit_and_aborts():
    h = _harness(2)
    maximum = (1 << 63) - 1
    h.owner._reset_generation[0] = maximum
    h.reset.events.clear()
    event = h.reset.issue(h.owner, (0,))
    prepared = h.owner.prepare_true_reset_many(event)
    for kind in r05.CHILD_OWNER_ORDER:
        projection = h.owner.require_owned_prepared_true_reset(
            prepared, owner_kind=kind
        )
        assert projection.generation_before.tolist() == [maximum, 1]
        assert projection.generation_after.tolist() == [maximum, 1]
        assert projection.writer_fault.tolist() is True
    h.reset.allow_abort(prepared)
    h.owner.abort_true_reset_many(prepared)
    assert h.owner.reset_generation.tolist() == [maximum, 1]
    assert h.owner.poisoned.tolist() is False
    view = _journal_view(h)
    schema = {name: (start, end) for name, start, end, _ in view.packed_schema}
    assert view.packed[slice(*schema["primary_fault"])][:2].tolist() == [17, 0]


def test_true_reset_mutation_max_is_in_prepared_writer_fault():
    h = _harness(2)
    h.owner._mutation_version.fill_((1 << 63) - 1)
    event = h.reset.issue(h.owner, (0,))
    prepared = h.owner.prepare_true_reset_many(event)
    projection = h.owner.require_owned_prepared_true_reset(
        prepared, owner_kind="motion"
    )
    assert projection.generation_overflow_fault.tolist() == [False, False]
    assert projection.writer_fault.tolist() is True
    h.reset.allow_abort(prepared)
    h.owner.abort_true_reset_many(prepared)


@pytest.mark.parametrize("ids", ((0, 0), (-1,), (2,)))
def test_true_reset_invalid_raw_selection_is_safe_writer_fault(ids):
    h = _harness(2)
    event = h.reset.issue(h.owner, ids)
    prepared = h.owner.prepare_true_reset_many(event)
    projection = h.owner.require_owned_prepared_true_reset(
        prepared, owner_kind="motion"
    )
    assert projection.generation_overflow_fault.tolist() == [False, False]
    assert projection.writer_fault.tolist() is True
    h.reset.allow_abort(prepared)
    h.owner.abort_true_reset_many(prepared)
    assert h.owner.poisoned.tolist() is False


def test_true_reset_host_mutation_max_rejects_before_authority_or_capability():
    h = _harness(2)
    event = h.reset.issue(h.owner, (0,))
    h.owner._mutation_version_host = (1 << 63) - 1
    before_epoch = h.owner._epoch
    before_head = h.owner._journal_head
    with pytest.raises(
        r05.DeviceR05PoisonedError, match="mutation chronology"
    ):
        h.owner.prepare_true_reset_many(event)
    assert h.owner._active is None
    assert h.owner._epoch == before_epoch
    assert h.owner._journal_head == before_head
    assert h.reset.project_calls == 0


def test_true_reset_preflight_registration_is_exact_once_and_required():
    h = _harness(2)
    event = h.reset.issue(h.owner, (0,))
    prepared = h.owner.prepare_true_reset_many(event)
    with pytest.raises(r05.DeviceR05ConflictError, match="registration failed"):
        h.owner.register_true_reset_preflight(prepared, object())
    h.reset.allow_commit(prepared)
    with pytest.raises(r05.DeviceR05ConflictError, match="already registered"):
        h.owner.register_true_reset_preflight(
            prepared, h.reset.preflights[prepared]
        )
    h.reset.committable.pop(prepared)
    with pytest.raises(r05.DeviceR05PoisonedError, match="child proof failed"):
        h.owner.commit_true_reset_many(prepared)


def test_true_reset_afterimage_failure_has_zero_live_partial_write(monkeypatch):
    h = _harness(2)
    event = h.reset.issue(h.owner, (0,))
    prepared = h.owner.prepare_true_reset_many(event)
    h.reset.allow_commit(prepared)
    live_names = (
        "reset_generation", "scheduled_ordinal", "outcome_shot_index",
        "sequence_kind", "task_identity", "outcome_identity",
        "ball_identity", "policy_opportunity", "mutation_version",
    )
    before = {name: getattr(h.owner, name) for name in live_names}
    publication = h.owner._publication
    build = h.owner._build_true_reset_journal_afterimage

    def fail_after_live_clone(*args, **kwargs):
        if kwargs["committed"]:
            raise RuntimeError("injected after cloned reset writes")
        return build(*args, **kwargs)

    monkeypatch.setattr(
        h.owner, "_build_true_reset_journal_afterimage", fail_after_live_clone
    )
    with pytest.raises(r05.DeviceR05PoisonedError, match="publication failed"):
        h.owner.commit_true_reset_many(prepared)
    assert h.owner._publication is not publication
    assert all(
        torch.equal(getattr(h.owner, name), value)
        for name, value in before.items()
    )
    view = _journal_view(h)
    schema = {name: (start, end) for name, start, end, _ in view.packed_schema}
    meta = view.packed[slice(*schema["meta"])].reshape(1, -1)
    assert int(meta[0, h.owner._META_OPERATION]) == r05.JOURNAL_ABORT
    with pytest.raises(r05.DeviceR05ConflictError, match="stale or foreign"):
        h.owner.commit_true_reset_many(prepared)
    assert view.end_sequence == 1


def test_construction_window_bind_is_one_time_and_read_closes_unbound_owner():
    dev = torch.device("cpu")
    genesis = _Genesis(dev, 1)
    cadence, reveal = _Cadence(dev, 1), _Reveal(dev)
    children, drain, reset = (
        tuple(_Child(kind) for kind in r05.CHILD_OWNER_ORDER), _Drain(), _Reset(dev, 1)
    )
    profile = _ProfileAuthority(dev)
    owner = r05.DeviceR05Owner(
        profile,
        profile.receipt,
        seed=1, num_envs=1, journal_capacity=2, max_reveal_epochs_per_drain=2,
        genesis_authority=genesis, genesis_receipt=genesis.receipt,
        cadence_authority=cadence, question_authority=_Question(dev),
        reveal_boundary_authority=reveal,
        child_completion_authorities=children, drain_authority=drain,
    )
    genesis_projection = owner.project_owned_genesis_for_child(
        owner_kind="physical_ball"
    )
    genesis_view = owner.require_owned_genesis_projection(
        genesis_projection, owner_kind="physical_ball"
    )
    assert genesis_view.device_r05_owner is owner
    assert genesis_view.world_reset_identity is genesis.projection.world_reset_identity
    assert genesis_view.reset_generation.tolist() == [1]
    forged = object.__new__(r05.DeviceR05GenesisProjection)
    with pytest.raises(r05.DeviceR05ConflictError, match="foreign"):
        owner.require_owned_genesis_projection(
            forged, owner_kind="physical_ball"
        )
    genesis_view.reset_generation.fill_(99)
    assert genesis.projection.reset_generations.tolist() == [1]
    with pytest.raises(r05.DeviceR05Error, match="kind"):
        owner.project_owned_genesis_for_child(owner_kind="caller")
    env_projection = owner.project_owned_genesis_for_child(
        owner_kind="full_mdp_env"
    )
    assert owner.require_owned_genesis_projection(
        env_projection, owner_kind="full_mdp_env"
    ).reset_generation.tolist() == [1]
    owner.bind_true_reset_authority(reset)
    with pytest.raises(r05.DeviceR05ConflictError, match="closed"):
        owner.project_owned_genesis_for_child(owner_kind="physical_ball")
    with pytest.raises(r05.DeviceR05ConflictError, match="closed"):
        owner.bind_true_reset_authority(reset)

    genesis2 = _Genesis(dev, 1)
    profile2 = _ProfileAuthority(dev)
    owner2 = r05.DeviceR05Owner(
        profile2, profile2.receipt,
        seed=1, num_envs=1, journal_capacity=2,
        max_reveal_epochs_per_drain=2, genesis_authority=genesis2,
        genesis_receipt=genesis2.receipt, cadence_authority=cadence,
        question_authority=_Question(dev),
        reveal_boundary_authority=reveal,
        child_completion_authorities=children, drain_authority=drain,
    )
    with pytest.raises(r05.DeviceR05ConflictError, match="before"):
        _ = owner2.draw_count
    with pytest.raises(r05.DeviceR05ConflictError, match="closed"):
        owner2.bind_true_reset_authority(reset)


def test_global_leaf_empty_journal_has_zero_count_and_fixed_sentinel():
    h = _harness(1, capacity=2, max_epochs=2)
    coordinator, _prepared, receipt, row = _prepare_global_receipt(
        h, terminal_total=0
    )
    assert row.scalar("journal_count") == 0
    assert row.scalar("journal_start_sequence") == 0
    assert row.scalar("journal_end_sequence") == 0
    assert row.scalar("terminal_resolution_total") == 0
    assert row.scalar("policy_opportunity_total") == 0
    assert len(dict(row.values)["journal_meta"]) == 2 * 2 * h.owner._META_WIDTH
    assert set(dict(row.values)["journal_meta"]) <= {0, 1}
    coordinator.mark_optimizer_returned(receipt)
    coordinator.acknowledge_post_update(receipt)
    assert h.owner.checkpoint_device().last_global_update_index == 0


def test_global_leaf_ack_before_optimizer_return_fail_stops():
    h = _harness(1)
    coordinator, _prepared, receipt, _row = _prepare_global_receipt(
        h, terminal_total=0
    )
    with pytest.raises(
        global_drain.ActionBallFullMdpPpoDrainPoisonedError,
        match="preceded optimizer return",
    ):
        coordinator.acknowledge_post_update(receipt)
    assert h.owner._poisoned_python


def test_global_leaf_rejects_foreign_real_coordinator_ack():
    h = _harness(1)
    coordinator, _prepared, receipt, _row = _prepare_global_receipt(
        h, terminal_total=0
    )
    foreign = _harness(1)
    _foreign_coordinator, _foreign_prepared, foreign_receipt, foreign_row = (
        _prepare_global_receipt(foreign, terminal_total=0)
    )
    coordinator.mark_optimizer_returned(receipt)
    active = h.owner._active_drain
    assert active is not None
    with pytest.raises(
        global_drain.ActionBallFullMdpPpoDrainError,
        match="foreign, stale, out of window",
    ):
        h.owner.acknowledge_pre_optimizer_ppo_boundary(
            pack=active.capability,
            receipt=foreign_receipt,
            owner_row=foreign_row,
        )
    assert h.owner._poisoned_python


def test_global_leaf_prepare_abort_ack_have_no_tensor_host_observation(monkeypatch):
    h = _harness(1)
    coordinator = _ensure_global(h)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden host tensor observation")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    prepared = coordinator.prepare_pre_optimizer_ppo_boundary(
        update_index=0, completed_environment_steps=48
    )
    coordinator.abort_pre_optimizer_ppo_boundary(prepared)

    # The single coordinator transfer is intentionally outside the tripwire.
    monkeypatch.undo()
    coordinator, _prepared, receipt, row = _prepare_global_receipt(
        h, terminal_total=0
    )
    coordinator.mark_optimizer_returned(receipt)
    active = h.owner._active_drain
    assert active is not None
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    h.owner.acknowledge_pre_optimizer_ppo_boundary(
        pack=active.capability,
        receipt=receipt,
        owner_row=row,
    )
    assert h.owner._active_drain is None


class _CarryRoot:
    def __init__(self, value):
        self.value = torch.tensor((value,), dtype=torch.int64)

    def _lean_carry_schema(self):
        return carry._LeanCarrySchema(
            "root", (), (carry._LeanCarryTensorSpec("sentinel", (1,), torch.int64),)
        )

    def _lean_carry_construction_views(self):
        return (self.value,)

    def _lean_carry_capture(self, lease):
        assert lease.coordinator is self._lean_carry_coordinator
        return carry._LeanCarryCapture((), (self.value,))

    def _lean_carry_stage(self, lease, scalars, host_tensors):
        assert lease.coordinator is self._lean_carry_coordinator
        return carry._LeanCarryStage(
            scalars,
            tuple(value.to(self.value.device, copy=True) for value in host_tensors),
            (self.value,),
        )

    def _lean_carry_target_views(self, lease, stage):
        assert lease.coordinator is self._lean_carry_coordinator
        assert type(stage) is carry._LeanCarryStage
        return (self.value,)

    def _lean_carry_apply_scalars(self, lease, stage):
        assert lease.coordinator is self._lean_carry_coordinator
        assert type(stage) is carry._LeanCarryStage and stage.commit_started

    def _lean_carry_cross_validate(
        self, lease, source_scalars_by_role, host_tensors_by_role,
        staged_scalars_by_role,
    ):
        assert lease.coordinator is self._lean_carry_coordinator
        assert source_scalars_by_role == staged_scalars_by_role
        assert all(value.device.type == "cpu" for row in host_tensors_by_role for value in row)


def _lean_harness(n=2, *, seed=20260817, bind_reset=True):
    device = torch.device("cpu")
    epoch_owner = epoch.ActionEpochOwner(
        num_envs=n, device=device, shot_slot_capacity=1,
        initial_reset_generation=1,
    )
    epoch_owner.activate_reset_genesis(
        selected_mask=torch.ones(n, dtype=torch.bool),
        reset_generation=torch.ones(n, dtype=torch.int64),
    )
    profile, genesis = _ProfileAuthority(device), _Genesis(device, n)
    reset = _Reset(device, n)
    owner = r05.DeviceR05Owner(
        profile, profile.receipt, seed=seed, num_envs=n,
        journal_capacity=64, max_reveal_epochs_per_drain=64,
        genesis_authority=genesis, genesis_receipt=genesis.receipt,
        cadence_authority=_Cadence(device, n),
        question_authority=_Question(device), diagnostic_epoch_owner=epoch_owner,
    )
    reset.bind_owner(owner)
    if bind_reset:
        owner.bind_true_reset_authority(reset)
    return types.SimpleNamespace(
        owner=owner, reset=reset, profile=profile, genesis=genesis,
        epoch_owner=epoch_owner,
    )


def _settle_lean_true_reset(h, ids):
    prepared = h.owner.prepare_true_reset_many(h.reset.issue(h.owner, ids))
    h.reset.allow_commit(prepared)
    receipt = h.owner.commit_true_reset_many(prepared)
    for kind in r05.CHILD_OWNER_ORDER:
        child = h.reset.issue_child_completion(receipt, kind)
        h.owner.record_true_reset_child_completion(
            receipt, child_kind=kind, child_receipt=child
        )
    assert h.owner._active is None
    assert not any(getattr(h.owner, name) for name in r05._PUBLICATION_REGISTRY_NAMES)


def _carry_coordinator(owner, *, root_value):
    root = _CarryRoot(root_value)
    coordinator = carry._LeanCarryCoordinator(
        root=root, mandatory_roles=("root", "d05")
    )
    coordinator._register("root", root)
    coordinator._register("d05", owner)
    return root, coordinator


def test_lean_carry_schema_has_only_business_copy_and_construction_attest():
    owner = _lean_harness().owner
    schema = owner._lean_carry_schema()
    fields = {field.name: field.disposition for field in schema.tensor_fields}
    assert schema.role == "d05"
    assert set(fields) == set(r05._LEAN_CARRY_COPY_NAMES + r05._LEAN_CARRY_ATTEST_NAMES)
    assert all(fields[name] == "copy" for name in r05._LEAN_CARRY_COPY_NAMES)
    assert all(fields[name] == "attest" for name in r05._LEAN_CARRY_ATTEST_NAMES)
    assert "last_candidate_bank_sequence" not in fields
    assert "next_candidate_identity" not in fields
    assert "full_key_sha256" not in fields
    assert all(field.placement == "device" for field in schema.tensor_fields)


def test_lean_carry_registration_precedes_exact_final_reset_bind():
    h = _lean_harness(bind_reset=False)
    _root, coordinator = _carry_coordinator(h.owner, root_value=1)
    h.owner.bind_true_reset_authority(h.reset)
    image = coordinator._capture()
    coordinator._discard(image)
    h.owner._question_authority = object()
    with pytest.raises(r05.DeviceR05ConflictError, match="binding drifted"):
        h.owner._lean_carry_construction_views()
    with pytest.raises(r05.DeviceR05ConflictError, match="quiescent"):
        coordinator._capture()


def test_lean_carry_final_reset_bind_cannot_resign_prior_authority_drift():
    h = _lean_harness(bind_reset=False)
    _root, _coordinator = _carry_coordinator(h.owner, root_value=1)
    h.owner._question_authority = object()
    with pytest.raises(r05.DeviceR05ConflictError, match="changed before final"):
        h.owner.bind_true_reset_authority(h.reset)
    assert h.owner._construction_window_open is False
    assert h.owner._true_reset_authority is None


def _install_counterexample_committed(owner):
    lo, hi = owner._rng_lo.clone(), owner._rng_hi.clone()
    for _ in range(r05.INTERNAL_QUESTION_TOTAL_DRAW_WIDTH):
        lo, hi, _draw_lo, _draw_hi = r05._splitmix64_lanes(lo, hi)
    owner._rng_lo.copy_(lo)
    owner._rng_hi.copy_(hi)
    owner._draw_count.fill_(r05.INTERNAL_QUESTION_TOTAL_DRAW_WIDTH)
    owner._target_generation.fill_(1)
    owner._previous_cell_index.fill_(0)
    owner._scheduled_ordinal.fill_(0)
    owner._outcome_shot_index.fill_(1)
    owner._sequence_kind.fill_(r05.SEQUENCE_COMMITTED)
    owner._policy_opportunity.fill_(True)
    identities = torch.arange(1, owner._num_envs + 1, dtype=torch.int64)
    for value in (owner._task_identity, owner._outcome_identity, owner._ball_identity):
        value.copy_(identities)
    owner._next_outcome_identity.fill_(owner._num_envs + 1)
    owner._next_ball_identity.fill_(owner._num_envs + 1)


def test_lean_carry_real_true_reset_roundtrip_and_next_public_afterimage(monkeypatch):
    source, target = _lean_harness(), _lean_harness()
    _settle_lean_true_reset(source, (0,))
    source_root, source_coordinator = _carry_coordinator(source.owner, root_value=7)
    target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
    calls = 0
    direct = carry._single_composite_d2h

    def counted(packed):
        nonlocal calls
        calls += 1
        return direct(packed)

    monkeypatch.setattr(carry, "_single_composite_d2h", counted)
    image = source_coordinator._capture()
    target_identities = tuple(id(value) for value in target.owner._lean_carry_views())
    prepared = target_coordinator._prepare(image)
    target_coordinator._commit(prepared)
    assert calls == 1 and target_root.value.tolist() == source_root.value.tolist()
    assert tuple(id(value) for value in target.owner._lean_carry_views()) == target_identities
    assert target.owner._lean_carry_scalars()[10:] == source.owner._lean_carry_scalars()[10:]
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            source.owner._lean_carry_views()[: len(r05._LEAN_CARRY_COPY_NAMES)],
            target.owner._lean_carry_views()[: len(r05._LEAN_CARRY_COPY_NAMES)],
        )
    )
    _settle_lean_true_reset(source, (1,))
    _settle_lean_true_reset(target, (1,))
    assert source.owner._lean_carry_scalars()[10:] == target.owner._lean_carry_scalars()[10:]
    assert all(
        torch.equal(left, right)
        for left, right in zip(source.owner._lean_carry_views(), target.owner._lean_carry_views())
    )


def test_lean_carry_roundtrips_due_consumed_without_accepted_task():
    """A deferred first due consumes chronology without inventing a task."""

    source, target = _lean_harness(), _lean_harness()
    source.owner._scheduled_ordinal.fill_(0)
    source.owner._outcome_shot_index.fill_(1)
    assert source.owner._sequence_kind.eq(r05.SEQUENCE_EMPTY).all()
    assert source.owner._task_identity.eq(-1).all()
    source_root, source_coordinator = _carry_coordinator(
        source.owner, root_value=7
    )
    target_root, target_coordinator = _carry_coordinator(
        target.owner, root_value=0
    )

    image = source_coordinator._capture()
    prepared = target_coordinator._prepare(image)
    target_coordinator._commit(prepared)

    assert target_root.value.tolist() == source_root.value.tolist()
    assert target.owner._scheduled_ordinal.eq(0).all()
    assert target.owner._outcome_shot_index.eq(1).all()
    assert target.owner._sequence_kind.eq(r05.SEQUENCE_EMPTY).all()
    assert target.owner._task_identity.eq(-1).all()


@pytest.mark.parametrize(
    "mutation",
    (
        "rng", "generation", "identity", "committed_without_generation",
        "committed_negative", "censored_positive", "censored_empty_identity",
        "duplicate_identity", "profile", "source_alias", "source_dtype",
    ),
)
def test_lean_carry_rejects_malformed_source_state(mutation):
    source, target = _lean_harness(), _lean_harness()
    if mutation == "rng":
        source.owner._rng_lo.add_(1)
    elif mutation == "generation":
        source.owner._target_generation.add_(1)
    elif mutation == "identity":
        source.owner._task_identity.fill_(1)
        source.owner._outcome_identity.fill_(1)
        source.owner._ball_identity.fill_(1)
    elif mutation in (
        "committed_without_generation", "committed_negative",
        "censored_positive", "censored_empty_identity", "duplicate_identity",
    ):
        _install_counterexample_committed(source.owner)
        if mutation == "committed_without_generation":
            source.owner._target_generation.zero_()
            source.owner._previous_cell_index.fill_(-1)
        elif mutation == "committed_negative":
            for value in (
                source.owner._task_identity, source.owner._outcome_identity,
                source.owner._ball_identity,
            ):
                value.fill_(-1)
        elif mutation == "censored_positive":
            source.owner._sequence_kind.fill_(r05.SEQUENCE_INFRA_CENSORED)
            source.owner._policy_opportunity.zero_()
        elif mutation == "censored_empty_identity":
            source.owner._sequence_kind.fill_(r05.SEQUENCE_INFRA_CENSORED)
            source.owner._policy_opportunity.zero_()
            for value in (
                source.owner._task_identity, source.owner._outcome_identity,
                source.owner._ball_identity,
            ):
                value.fill_(-1)
        elif mutation == "duplicate_identity":
            for value in (
                source.owner._task_identity, source.owner._outcome_identity,
                source.owner._ball_identity,
            ):
                value.fill_(1)
    elif mutation == "profile":
        source.owner._profile.targets_xy_m[0, 0].add_(0.25)
    elif mutation == "source_alias":
        source.owner._rng_hi = source.owner._rng_lo
    else:
        source.owner._rng_lo = source.owner._rng_lo.to(torch.float64)
    if mutation == "source_dtype":
        with pytest.raises(r05.DeviceR05Error, match="device/dtype/shape"):
            _carry_coordinator(source.owner, root_value=1)
        return
    _source_root, source_coordinator = _carry_coordinator(source.owner, root_value=1)
    _target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
    if mutation == "source_alias":
        with pytest.raises(carry._LeanCarryError):
            source_coordinator._capture()
    else:
        image = source_coordinator._capture()
        with pytest.raises((carry._LeanCarryError, r05.DeviceR05ConflictError)):
            target_coordinator._prepare(image)


def test_lean_carry_rejects_real_duplicate_reset_device_poison():
    source, target = _lean_harness(), _lean_harness()
    _settle_lean_true_reset(source, (0, 0))
    assert source.owner._poisoned_python is False and bool(source.owner._poisoned)
    _source_root, source_coordinator = _carry_coordinator(source.owner, root_value=1)
    _target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
    image = source_coordinator._capture()
    with pytest.raises(r05.DeviceR05ConflictError, match="attestation"):
        target_coordinator._prepare(image)


@pytest.mark.parametrize("mutation", ("profile", "not_fresh"))
def test_lean_carry_rejects_target_construction_drift(mutation):
    source, target = _lean_harness(), _lean_harness()
    if mutation == "profile":
        target.owner._profile.targets_xy_m[0, 0].add_(0.25)
    else:
        _settle_lean_true_reset(target, (0,))
    _source_root, source_coordinator = _carry_coordinator(source.owner, root_value=1)
    _target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
    with pytest.raises((carry._LeanCarryError, r05.DeviceR05ConflictError)):
        target_coordinator._prepare(source_coordinator._capture())


def test_lean_carry_rejects_source_target_storage_overlap():
    source, target = _lean_harness(), _lean_harness()
    target.owner._rng_lo = source.owner._rng_lo
    _source_root, source_coordinator = _carry_coordinator(source.owner, root_value=1)
    _target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
    with pytest.raises(carry._LeanCarryError, match="aliases"):
        target_coordinator._prepare(source_coordinator._capture())


def test_lean_carry_commit_rechecks_attest_target_and_construction_bindings():
    for mutation in ("profile_value", "tensor_rebind", "scalar", "authority"):
        source, target = _lean_harness(), _lean_harness()
        _source_root, source_coordinator = _carry_coordinator(source.owner, root_value=1)
        _target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
        prepared = target_coordinator._prepare(source_coordinator._capture())
        if mutation == "profile_value":
            target.owner._profile.targets_xy_m[0, 0].add_(0.25)
        elif mutation == "tensor_rebind":
            target.owner._rng_lo = target.owner._rng_lo.clone()
        elif mutation == "scalar":
            target.owner._seed += 1
        else:
            target.owner._question_authority = object()
        with pytest.raises((carry._LeanCarryError, r05.DeviceR05ConflictError)):
            target_coordinator._commit(prepared)
        target_coordinator._abort(prepared)


def test_lean_carry_commit_rejects_draw_generation_storage_alias_after_prepare():
    source, target = _lean_harness(), _lean_harness()
    _settle_lean_true_reset(source, (0,))
    _source_root, source_coordinator = _carry_coordinator(
        source.owner, root_value=1
    )
    _target_root, target_coordinator = _carry_coordinator(
        target.owner, root_value=0
    )
    prepared = target_coordinator._prepare(source_coordinator._capture())
    reset_before = target.owner._reset_generation.clone()
    target.owner._draw_count.set_(target.owner._target_generation)

    with pytest.raises(carry._LeanCarryError, match="aliases"):
        target_coordinator._commit(prepared)

    assert torch.equal(target.owner._reset_generation, reset_before)
    target_coordinator._abort(prepared)


def test_lean_carry_lease_blocks_public_mutation_and_lean_legacy_checkpoint():
    source, target = _lean_harness(), _lean_harness()
    _source_root, source_coordinator = _carry_coordinator(source.owner, root_value=1)
    _target_root, target_coordinator = _carry_coordinator(target.owner, root_value=0)
    prepared = target_coordinator._prepare(source_coordinator._capture())
    with pytest.raises(carry._LeanCarryError, match="overlaps"):
        target.owner.poison_pre_optimizer_ppo_boundary(reason="injected")
    assert target.owner._global_drain_poison_reason is None
    assert target.owner._poisoned_python is False
    assert not bool(target.owner._poisoned) and int(target.owner._poison_reason) == 0
    with pytest.raises(carry._LeanCarryError, match="overlaps"):
        target.owner.prepare_true_reset_many(target.reset.issue(target.owner, (0,)))
    target_coordinator._abort(prepared)
    with pytest.raises(r05.DeviceR05ConflictError, match="tombstoned"):
        target.owner.checkpoint_device()
