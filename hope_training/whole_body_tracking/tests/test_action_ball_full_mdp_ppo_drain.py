"""Portable tests for the one global pre-optimizer ActionBall drain."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import gc
import importlib.util
import os
from pathlib import Path
import pickle
import sys
import weakref

import pytest
import torch


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "action_ball_full_mdp_ppo_drain.py"
)
SPEC = importlib.util.spec_from_file_location(
    "action_ball_full_mdp_ppo_drain_test_target", SOURCE_PATH
)
assert SPEC is not None and SPEC.loader is not None
D = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = D
SPEC.loader.exec_module(D)


def _load_module(name, source_path):
    spec = importlib.util.spec_from_file_location(name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLeaf:
    def __init__(
        self,
        owner_kind: str,
        *,
        num_envs: int,
        schema,
        total: int = 7,
        device: str = "cpu",
    ) -> None:
        self.owner_kind = owner_kind
        self.num_envs = num_envs
        self.schema = schema
        self.total = total
        self.device = device
        self.prepare_calls = []
        self.abort_calls = []
        self.ack_calls = []
        self.poison_calls = []
        self.prepare_failure = None
        self.abort_failure = None
        self.ack_failure = None
        self.poison_failure = None
        self.fault_count = 0
        self.invariant_count = 0
        self.mutation_version = 3
        self.counter_overrides = {}
        self.extra_values = {}
        self.last_pack = None
        self.last_authority = None
        self.last_values = None
        self.expected_ack_sequence = None

    def _values(self, update_index, completed_environment_steps):
        values = []
        for field in self.schema.fields:
            if field.name == "mutation_version":
                value = self.mutation_version
            elif field.name == "fault_count":
                value = self.fault_count
            elif field.name == "invariant_count":
                value = self.invariant_count
            elif field.name in (
                "terminal_resolution_total",
                "policy_opportunity_total",
                "shared_normal_retire_total",
                "physical_only_orphan_park_total",
                "r06_only_orphan_retire_total",
                "shared_normal_retire_key_summary_0",
                "shared_normal_retire_key_summary_1",
            ):
                value = self.counter_overrides.get(field.name, self.total)
            else:
                value = self.extra_values.get(field.name, field.minimum)
            width = field.width(self.num_envs)
            if field.cardinality == "scalar":
                values.append(value)
            else:
                row = value if isinstance(value, tuple) else (value,) * width
                values.extend(row)
        return torch.tensor(values, dtype=torch.int64, device=self.device)

    def prepare_pre_optimizer_ppo_boundary_device_pack(
        self,
        *,
        authority,
        update_index,
        completed_environment_steps,
    ):
        self.prepare_calls.append((update_index, completed_environment_steps))
        if self.prepare_failure is not None and self.prepare_failure[0] == "before":
            raise self.prepare_failure[1]
        values = self._values(update_index, completed_environment_steps)
        self.last_values = values
        self.last_authority = authority
        pack = authority.mint_device_pack(
            leaf=self,
            values=values,
        )
        self.last_pack = pack
        if self.prepare_failure is not None and self.prepare_failure[0] == "after":
            raise self.prepare_failure[1]
        return pack

    def abort_pre_optimizer_ppo_boundary_device_pack(self, *, pack):
        self.abort_calls.append(pack)
        if self.abort_failure is not None:
            raise self.abort_failure

    def acknowledge_pre_optimizer_ppo_boundary(
        self,
        *,
        pack,
        receipt,
        owner_row,
    ):
        self.ack_calls.append((pack, receipt, owner_row))
        if (
            self.expected_ack_sequence is not None
            and receipt.drain_sequence != self.expected_ack_sequence
        ):
            raise RuntimeError("drain sequence differs")
        if self.ack_failure is not None:
            raise self.ack_failure

    def poison_pre_optimizer_ppo_boundary(self, *, reason):
        self.poison_calls.append(reason)
        if self.poison_failure is not None:
            raise self.poison_failure


def make_owner(
    num_envs,
    *,
    schemas=None,
    initial_update_index=0,
    device="cpu",
    checkpoint_boundary_validator=None,
    checkpoint_restore_validator_factory=None,
    join_leaf_bindings=True,
):
    schemas = D.DEFAULT_LEAF_SCHEMAS if schemas is None else tuple(schemas)
    leaves = {
        schema.owner_kind: FakeLeaf(
            schema.owner_kind,
            num_envs=num_envs,
            schema=schema,
            device=device,
        )
        for schema in schemas
    }
    owner = D.ActionBallFullMdpPpoDrainOwner(
        num_envs=num_envs,
        device=device,
        leaves=leaves,
        leaf_schemas=schemas,
        initial_update_index=initial_update_index,
        diagnostic_allow_minimal_schemas=(schemas == D.DEFAULT_LEAF_SCHEMAS),
        checkpoint_boundary_validator=checkpoint_boundary_validator,
        checkpoint_restore_validator_factory=(
            checkpoint_restore_validator_factory
        ),
    )
    if join_leaf_bindings:
        owner.require_exact_leaf_bindings(
            {name: leaves[name] for name in D.OWNER_ORDER}
        )
    return owner, leaves


@pytest.mark.parametrize("num_envs", (2, 64))
def test_n2_n64_single_global_transfer_and_optimizer_ack(num_envs):
    device = os.environ.get("ACTION_BALL_PPO_DRAIN_TEST_DEVICE", "cpu")
    if device == "cuda":
        device = "cuda:0"
    owner, leaves = make_owner(num_envs, device=device)

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=num_envs * 24,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert receipt.schema_version == 1
    assert receipt.kind == D.RECEIPT_KIND
    assert receipt.update_index == 0
    assert receipt.completed_environment_steps == num_envs * 24
    assert receipt.owner_order == D.OWNER_ORDER
    assert tuple(row.owner_kind for row in receipt.owner_rows) == D.OWNER_ORDER
    assert receipt.device_to_host_transfers == 1
    assert receipt.acknowledged is False
    assert all(len(leaf.ack_calls) == 0 for leaf in leaves.values())

    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)

    assert receipt.acknowledged is True
    assert owner.next_update_index == 1
    assert all(len(leaf.ack_calls) == 1 for leaf in leaves.values())
    assert all(not leaf.abort_calls for leaf in leaves.values())
    assert all(not leaf.poison_calls for leaf in leaves.values())


def test_pack_is_a_device_snapshot_not_a_leaf_mutable_alias():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    for leaf in leaves.values():
        leaf.last_values.add_(1000)

    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert all(
        row.scalar("mutation_version") == 3 for row in receipt.owner_rows
    )
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_source_has_exactly_one_explicit_cpu_transfer_and_no_legacy_drain_call():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    executable = source.split("__all__ =", 1)[0]
    assert executable.count('.to(device="cpu")') == 1
    assert ".cpu(" not in executable
    assert "drain_ppo_boundary(" not in executable
    assert "packet_sha256" not in executable


@pytest.mark.skipif(
    os.environ.get("ACTION_BALL_PPO_DRAIN_TEST_DEVICE", "cpu") == "cpu",
    reason="requires the exact Pod CUDA lane",
)
@pytest.mark.parametrize("host_observation", ("item", "cpu", "tolist"))
def test_leaf_prepare_cannot_hide_an_independent_host_observation(host_observation):
    device = os.environ["ACTION_BALL_PPO_DRAIN_TEST_DEVICE"]
    if device == "cuda":
        device = "cuda:0"
    owner, leaves = make_owner(2, device=device)
    leaf = leaves["motion"]
    original = leaf.prepare_pre_optimizer_ppo_boundary_device_pack

    def observed_prepare(**kwargs):
        probe = torch.ones(1, dtype=torch.int64, device=device)
        if host_observation == "item":
            probe[0].item()
        elif host_observation == "cpu":
            probe.cpu()
        else:
            probe.tolist()
        return original(**kwargs)

    leaf.prepare_pre_optimizer_ppo_boundary_device_pack = observed_prepare
    with pytest.raises(D.ActionBallFullMdpPpoDrainPrepareError) as caught:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )

    assert caught.value.retry_permitted is False
    assert owner.poisoned is True


def test_fixed_owner_order_and_required_independent_writer_conservation():
    assert D.OWNER_ORDER == (
        "r05_runtime",
        "motion",
        "racket",
        "physical_ball",
        "r06_landing_outcome",
        "r03_strike_fact",
        "r07_recovery",
    )
    assert tuple(rule.name for rule in D.REQUIRED_CONSERVATION_RULES) == (
        "r05_terminal_vs_motion_completion",
        "r05_terminal_vs_racket_completion",
        "r05_terminal_vs_physical_completion",
        "r05_terminal_vs_r06_completion",
        "physical_vs_r06_shared_normal_retire_count",
        "physical_vs_r06_shared_normal_retire_key_summary_0",
        "physical_vs_r06_shared_normal_retire_key_summary_1",
    )
    for rule in D.REQUIRED_CONSERVATION_RULES:
        assert {term.owner_kind for term in rule.left}.isdisjoint(
            term.owner_kind for term in rule.right
        )


def test_same_writer_conservation_is_rejected_as_self_attestation():
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="itself"):
        D.ConservationRule(
            name="same_writer",
            left=(D.ConservationTerm("r05_runtime", "a"),),
            right=(D.ConservationTerm("r05_runtime", "b"),),
        )


def test_distinct_owner_kinds_cannot_alias_one_leaf_object():
    _owner, leaves = make_owner(2)
    leaves["motion"] = leaves["r05_runtime"]
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="distinct"):
        D.ActionBallFullMdpPpoDrainOwner(
            num_envs=2,
            device="cpu",
            leaves=leaves,
            diagnostic_allow_minimal_schemas=True,
        )


def test_minimal_default_schema_requires_explicit_diagnostic_opt_in():
    _owner, leaves = make_owner(2)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="diagnostic opt-in"):
        D.ActionBallFullMdpPpoDrainOwner(
            num_envs=2,
            device="cpu",
            leaves=leaves,
        )


def test_construction_exact_leaf_binding_join_accepts_only_exact_ordered_identities():
    schemas = D.DEFAULT_LEAF_SCHEMAS
    leaves = {
        schema.owner_kind: FakeLeaf(
            schema.owner_kind,
            num_envs=2,
            schema=schema,
        )
        for schema in schemas
    }
    owner = D.ActionBallFullMdpPpoDrainOwner(
        num_envs=2,
        device="cpu",
        leaves=leaves,
        diagnostic_allow_minimal_schemas=True,
    )
    ordered = {name: leaves[name] for name in D.OWNER_ORDER}
    assert owner.require_exact_leaf_bindings(ordered) is None
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="closed"):
        owner.require_exact_leaf_bindings(ordered)


@pytest.mark.parametrize("mutation", ("foreign", "swapped", "missing"))
def test_construction_exact_leaf_binding_join_rejects_substitution_once(mutation):
    schemas = D.DEFAULT_LEAF_SCHEMAS
    leaves = {
        schema.owner_kind: FakeLeaf(
            schema.owner_kind,
            num_envs=2,
            schema=schema,
        )
        for schema in schemas
    }
    owner = D.ActionBallFullMdpPpoDrainOwner(
        num_envs=2,
        device="cpu",
        leaves=leaves,
        diagnostic_allow_minimal_schemas=True,
    )
    ordered = {name: leaves[name] for name in D.OWNER_ORDER}
    if mutation == "foreign":
        ordered["motion"] = object()
    elif mutation == "swapped":
        ordered["motion"], ordered["racket"] = (
            ordered["racket"],
            ordered["motion"],
        )
    else:
        del ordered["r03_strike_fact"]

    with pytest.raises(D.ActionBallFullMdpPpoDrainError):
        owner.require_exact_leaf_bindings(ordered)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="closed"):
        owner.require_exact_leaf_bindings(
            {name: leaves[name] for name in D.OWNER_ORDER}
        )


def test_construction_leaf_binding_join_closes_after_read_or_runtime_operation():
    def unjoined_owner():
        schemas = D.DEFAULT_LEAF_SCHEMAS
        leaves = {
            schema.owner_kind: FakeLeaf(
                schema.owner_kind,
                num_envs=2,
                schema=schema,
            )
            for schema in schemas
        }
        owner = D.ActionBallFullMdpPpoDrainOwner(
            num_envs=2,
            device="cpu",
            leaves=leaves,
            diagnostic_allow_minimal_schemas=True,
        )
        return owner, leaves

    read_owner, read_leaves = unjoined_owner()
    assert read_owner.next_update_index == 0
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="closed"):
        read_owner.require_exact_leaf_bindings(
            {name: read_leaves[name] for name in D.OWNER_ORDER}
        )

    runtime_owner, runtime_leaves = unjoined_owner()
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="binding join"):
        runtime_owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="closed"):
        runtime_owner.require_exact_leaf_bindings(
            {name: runtime_leaves[name] for name in D.OWNER_ORDER}
        )


def test_future_r05_per_env_fields_extend_schema_without_reordering_owner():
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    first = schemas[0]
    schemas[0] = D.LeafDrainSchema(
        owner_kind=first.owner_kind,
        fields=first.fields
        + (
            D.DeviceDrainFieldSpec(
                "portable_row_ordinal",
                cardinality="per_env",
            ),
        ),
    )
    owner, leaves = make_owner(64, schemas=schemas)
    leaves["r05_runtime"].extra_values["portable_row_ordinal"] = tuple(range(64))

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=1536,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    row = receipt.owner_rows[0]

    assert row.values[-1] == ("portable_row_ordinal", tuple(range(64)))
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


@pytest.mark.parametrize(
    ("logical_count", "journal", "expected"),
    (
        (0, (0, 0, 0, 0), (0, 0, 0, 0)),
        (3, (11, 12, 13, 0), (11, 12, 13, 0)),
    ),
)
def test_fixed_width_field_accepts_empty_or_populated_bounded_tuple(
    logical_count,
    journal,
    expected,
):
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    first = schemas[0]
    schemas[0] = D.LeafDrainSchema(
        owner_kind=first.owner_kind,
        fields=first.fields
        + (
            D.DeviceDrainFieldSpec("journal_count"),
            D.DeviceDrainFieldSpec(
                "journal_values",
                cardinality="fixed",
                fixed_width=4,
            ),
        ),
    )
    owner, leaves = make_owner(64, schemas=schemas)
    leaves["r05_runtime"].extra_values.update(
        journal_count=logical_count,
        journal_values=journal,
    )

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=1536,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert receipt.owner_rows[0].values[-2:] == (
        ("journal_count", logical_count),
        ("journal_values", expected),
    )
    assert D._checkpoint_schema_identity(schemas)[0][1][-1] == (
        "journal_values",
        "fixed",
        0,
        4,
    )
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_legacy_scalar_and_per_env_schema_identities_remain_three_tuples():
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    first = schemas[0]
    schemas[0] = D.LeafDrainSchema(
        owner_kind=first.owner_kind,
        fields=first.fields
        + (D.DeviceDrainFieldSpec("portable_row", cardinality="per_env"),),
    )

    identity = D._checkpoint_schema_identity(schemas)

    assert all(
        len(field) == 3
        for _owner, fields in identity
        for field in fields
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cardinality": "fixed"},
        {"cardinality": "fixed", "fixed_width": 0},
        {"cardinality": "fixed", "fixed_width": True},
        {"cardinality": "scalar", "fixed_width": 2},
        {"cardinality": "per_env", "fixed_width": 2},
    ),
)
def test_fixed_width_schema_rejects_absent_invalid_or_foreign_width(kwargs):
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="fixed_width"):
        D.DeviceDrainFieldSpec("journal", **kwargs)


@pytest.mark.parametrize("width_delta", (-1, 1))
def test_fixed_width_authority_rejects_wrong_pack_width(width_delta):
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    first = schemas[0]
    schemas[0] = D.LeafDrainSchema(
        owner_kind=first.owner_kind,
        fields=first.fields
        + (
            D.DeviceDrainFieldSpec(
                "journal_values",
                cardinality="fixed",
                fixed_width=4,
            ),
        ),
    )
    owner, leaves = make_owner(2, schemas=schemas)
    leaf = leaves["r05_runtime"]

    def wrong_width_prepare(*, authority, update_index, completed_environment_steps):
        del update_index, completed_environment_steps
        values = torch.zeros(
            authority.expected_width + width_delta,
            dtype=torch.int64,
            device=leaf.device,
        )
        return authority.mint_device_pack(leaf=leaf, values=values)

    leaf.prepare_pre_optimizer_ppo_boundary_device_pack = wrong_width_prepare
    with pytest.raises(D.ActionBallFullMdpPpoDrainPrepareError) as caught:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )
    assert caught.value.retry_permitted is False
    assert owner.poisoned is True


def test_fixed_width_decode_rejects_foreign_below_minimum_value():
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    first = schemas[0]
    schemas[0] = D.LeafDrainSchema(
        owner_kind=first.owner_kind,
        fields=first.fields
        + (
            D.DeviceDrainFieldSpec(
                "journal_values",
                cardinality="fixed",
                minimum=0,
                fixed_width=3,
            ),
        ),
    )
    owner, leaves = make_owner(2, schemas=schemas)
    leaves["r05_runtime"].extra_values["journal_values"] = (7, -1, 9)

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError, match="below"):
        owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert owner.poisoned is True


def test_real_r03_r07_factories_replace_minimal_schemas_without_copying_abi():
    source_root = SOURCE_PATH.parents[4]
    sys.path.insert(0, str(source_root))
    r03 = _load_module(
        "action_ball_strike_fact_device_schema_test_target",
        SOURCE_PATH.with_name("action_ball_strike_fact_device.py"),
    )
    try:
        r07 = _load_module(
            "action_ball_continuous_recovery_device_schema_test_target",
            source_root / "action_ball_continuous_recovery_device.py",
        )
    finally:
        sys.path.remove(str(source_root))
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    schemas[D.OWNER_ORDER.index("r03_strike_fact")] = (
        r03.make_pre_optimizer_ppo_boundary_leaf_schema(
            leaf_schema_type=D.LeafDrainSchema,
            field_spec_type=D.DeviceDrainFieldSpec,
        )
    )
    schemas[D.OWNER_ORDER.index("r07_recovery")] = (
        r07.materialize_r07_ppo_drain_leaf_schema(
            leaf_schema_type=D.LeafDrainSchema,
            field_spec_type=D.DeviceDrainFieldSpec,
        )
    )

    owner, leaves = make_owner(2, schemas=schemas)
    assert tuple(field.name for field in schemas[-2].fields) == (
        r03.PRE_OPTIMIZER_PPO_BOUNDARY_FIELD_NAMES
    )
    assert tuple(field.name for field in schemas[-1].fields) == (
        r07.R07_GLOBAL_DRAIN_FIELD_NAMES
    )
    assert tuple(
        field.name
        for field in schemas[-1].fields
        if field.cardinality == "per_env"
    ) == r07.R07_GLOBAL_DRAIN_PER_ENV_FIELDS
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assert receipt.owner_rows[-1].values[-1] == (
        "window_last_paid_age_tick_encoded",
        (0, 0),
    )
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_n1_per_env_extension_preserves_tuple_abi():
    schemas = list(D.DEFAULT_LEAF_SCHEMAS)
    first = schemas[0]
    schemas[0] = D.LeafDrainSchema(
        owner_kind=first.owner_kind,
        fields=first.fields
        + (D.DeviceDrainFieldSpec("portable_row", cardinality="per_env"),),
    )
    owner, leaves = make_owner(1, schemas=schemas)
    leaves["r05_runtime"].extra_values["portable_row"] = (9,)

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=24,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert receipt.owner_rows[0].values[-1] == ("portable_row", (9,))
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_clean_pretransfer_abort_is_reverse_order_and_same_update_can_retry():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    owner.abort_pre_optimizer_ppo_boundary(prepared)

    assert owner.poisoned is False
    assert all(len(leaf.abort_calls) == 1 for leaf in leaves.values())

    retried = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(retried)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)
    assert owner.next_update_index == 1


def test_prepare_failure_after_mint_aborts_prepared_prefix_and_retry_is_permitted():
    owner, leaves = make_owner(2)
    leaves["physical_ball"].prepare_failure = (
        "after",
        RuntimeError("post-mint preflight"),
    )

    with pytest.raises(
        D.ActionBallFullMdpPpoDrainPrepareError,
        match="prepare failed",
    ) as caught:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )

    assert caught.value.retry_permitted is True
    assert owner.poisoned is False
    for name in ("r05_runtime", "motion", "racket"):
        assert len(leaves[name].abort_calls) == 1
    assert len(leaves["physical_ball"].abort_calls) == 1
    assert not leaves["r06_landing_outcome"].prepare_calls

    leaves["physical_ball"].prepare_failure = None
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_pretransfer_abort_failure_sticky_poisons_all_leaves():
    owner, leaves = make_owner(2)
    leaves["racket"].prepare_failure = ("after", RuntimeError("preflight"))
    leaves["motion"].abort_failure = RuntimeError("cannot abort")

    with pytest.raises(D.ActionBallFullMdpPpoDrainPrepareError) as caught:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )

    assert caught.value.retry_permitted is False
    assert owner.poisoned is True
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )


def test_failure_before_abort_capability_sticky_poisons_no_retry():
    owner, leaves = make_owner(2)
    leaves["physical_ball"].prepare_failure = (
        "before",
        RuntimeError("mutated then failed before mint"),
    )

    with pytest.raises(D.ActionBallFullMdpPpoDrainPrepareError) as caught:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )

    assert caught.value.retry_permitted is False
    assert owner.poisoned is True
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())


@pytest.mark.parametrize(
    ("owner_kind", "mutation", "match"),
    (
        ("r03_strike_fact", "fault", "device fault"),
        ("r07_recovery", "invariant", "invariant failure"),
        ("motion", "mutation_regression", "mutation_version regressed"),
        ("physical_ball", "conservation", "conservation"),
    ),
)
def test_posttransfer_decode_failure_is_sticky_poison_no_retry(
    owner_kind,
    mutation,
    match,
):
    owner, leaves = make_owner(2)
    leaf = leaves[owner_kind]
    if mutation == "fault":
        leaf.fault_count = 1
    elif mutation == "invariant":
        leaf.invariant_count = 1
    elif mutation == "mutation_regression":
        owner._last_mutation_versions = {
            name: 4 for name in D.OWNER_ORDER
        }
        leaf.mutation_version = 2
    else:
        leaf.counter_overrides["terminal_resolution_total"] = leaf.total - 1

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError, match=match):
        owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert owner.poisoned is True
    assert all(len(value.poison_calls) == 1 for value in leaves.values())
    assert all(not value.abort_calls for value in leaves.values())
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )


def test_legal_censor_zero_policy_opportunity_keeps_resolution_conservation():
    owner, leaves = make_owner(2)
    leaves["r05_runtime"].counter_overrides["policy_opportunity_total"] = 0

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    r05 = receipt.owner_rows[0]
    assert r05.scalar("policy_opportunity_total") == 0
    assert r05.scalar("terminal_resolution_total") == 7
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


@pytest.mark.parametrize(
    ("owner_kind", "field_name"),
    (
        ("physical_ball", "physical_only_orphan_park_total"),
        ("r06_landing_outcome", "r06_only_orphan_retire_total"),
    ),
)
def test_legal_single_sided_orphan_cleanup_is_telemetry_not_false_conservation(
    owner_kind,
    field_name,
):
    owner, leaves = make_owner(2)
    leaves[owner_kind].counter_overrides[field_name] = 1

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    row = next(
        value for value in receipt.owner_rows if value.owner_kind == owner_kind
    )
    assert row.scalar(field_name) == 1
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_shared_normal_retire_key_summary_mismatch_sticky_poisons():
    owner, leaves = make_owner(2)
    leaves["physical_ball"].counter_overrides[
        "shared_normal_retire_key_summary_1"
    ] = 8

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    with pytest.raises(
        D.ActionBallFullMdpPpoDrainPoisonedError,
        match="key_summary_1",
    ):
        owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert owner.poisoned is True


def test_duplicate_and_out_of_order_update_are_rejected_without_transfer():
    owner, _leaves = make_owner(2)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="contiguous"):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=1,
            completed_environment_steps=48,
        )

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="contiguous"):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=96,
        )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="contiguous"):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=2,
            completed_environment_steps=96,
        )


def test_duplicate_transfer_poisons_and_no_abort_is_claimed():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError, match="duplicate"):
        owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    assert owner.poisoned is True
    assert all(not leaf.abort_calls for leaf in leaves.values())


def test_partial_optimizer_ack_poisons_all_and_cannot_retry():
    owner, leaves = make_owner(2)
    leaves["physical_ball"].ack_failure = RuntimeError("ack stopped")
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError, match="partial"):
        owner.mark_optimizer_returned(receipt)
        owner.acknowledge_post_update(receipt)

    assert owner.poisoned is True
    assert len(leaves["r05_runtime"].ack_calls) == 1
    assert len(leaves["physical_ball"].ack_calls) == 1
    assert not leaves["r06_landing_outcome"].ack_calls
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())


def test_post_update_ack_before_optimizer_return_sticky_poisons():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    with pytest.raises(
        D.ActionBallFullMdpPpoDrainPoisonedError,
        match="preceded optimizer return",
    ):
        owner.acknowledge_post_update(receipt)

    assert owner.poisoned is True
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())


def test_duplicate_optimizer_return_sticky_poisons():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)

    with pytest.raises(
        D.ActionBallFullMdpPpoDrainPoisonedError,
        match="duplicate optimizer-return",
    ):
        owner.mark_optimizer_returned(receipt)

    assert owner.poisoned is True
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())


def test_optimizer_failure_after_valid_drain_sticky_poisons():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    owner.poison_optimizer_failure(receipt, reason="Adam step raised")

    assert owner.poisoned is True
    assert "Adam step raised" in owner.poison_reason
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError):
        owner.mark_optimizer_returned(receipt)
        owner.acknowledge_post_update(receipt)


def test_poison_broadcast_continues_after_leaf_poison_failure():
    owner, leaves = make_owner(2)
    leaves["motion"].poison_failure = RuntimeError("local poison failed")
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    owner.poison_optimizer_failure(receipt, reason="optimizer")

    assert owner.poisoned is True
    assert owner.poison_failures[0][0] == "motion"
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())


def test_receipt_and_prepared_are_exact_owner_capabilities():
    first, _first_leaves = make_owner(2)
    second, _second_leaves = make_owner(2)
    prepared = first.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="owner-minted"):
        D.PreparedPreOptimizerPpoBoundary(
            owner=first,
            operation_id=1,
            token=object(),
        )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign"):
        second.transfer_decode_pre_optimizer_ppo_boundary(prepared)

    receipt = first.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="owner-minted"):
        D.PreOptimizerPpoBoundaryReceipt(
            owner=first,
            operation_id=1,
            update_index=0,
            completed_environment_steps=48,
            drain_sequence=1,
            num_envs=2,
            rows=receipt.owner_rows,
            token=object(),
        )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign"):
        second.acknowledge_post_update(receipt)

    row = receipt.owner_rows[0]
    with pytest.raises(FrozenInstanceError):
        row.owner_kind = "motion"
    first.mark_optimizer_returned(receipt)
    first.acknowledge_post_update(receipt)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign"):
        first.acknowledge_post_update(receipt)


def test_caller_assembled_lookalike_receipt_posttransfer_sticky_poisons():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    real = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    assembled = D.PreOptimizerPpoBoundaryReceipt(
        owner=owner,
        operation_id=1,
        update_index=real.update_index,
        completed_environment_steps=real.completed_environment_steps,
        drain_sequence=real.drain_sequence,
        num_envs=real.num_envs,
        rows=real.owner_rows,
        token=D._RECEIPT_TOKEN,
    )

    with pytest.raises(
        D.ActionBallFullMdpPpoDrainPoisonedError,
        match="caller-assembled",
    ):
        owner.acknowledge_post_update(assembled)

    assert owner.poisoned is True
    assert all(len(leaf.poison_calls) == 1 for leaf in leaves.values())
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError):
        owner.acknowledge_post_update(real)


def test_leaf_pack_hides_tensor_and_host_observation_apis():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    pack = leaves["r05_runtime"].last_pack

    assert repr(pack) == "<OpaqueLeafDevicePack owner='r05_runtime'>"
    for name in (
        "tensor",
        "values",
        "to",
        "cpu",
        "item",
        "tolist",
        "numpy",
        "to_mapping",
    ):
        assert not hasattr(pack, name)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="authority-minted"):
        D.OpaqueLeafDevicePack(owner_kind="r05_runtime", token=object())

    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)


def test_leaf_ack_authority_requires_exact_live_lane_and_optimizer_return():
    owner, leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    motion = leaves["motion"]
    racket = leaves["racket"]
    motion_row = receipt.owner_rows[D.OWNER_ORDER.index("motion")]
    racket_row = receipt.owner_rows[D.OWNER_ORDER.index("racket")]

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="out of window"):
        motion.last_authority.require_owned_ack(
            leaf=motion,
            pack=motion.last_pack,
            receipt=receipt,
            owner_row=motion_row,
        )

    owner.mark_optimizer_returned(receipt)
    assert (
        motion.last_authority.require_owned_ack(
            leaf=motion,
            pack=motion.last_pack,
            receipt=receipt,
            owner_row=motion_row,
        )
        is None
    )
    # Validation is repeatable only while this exact global ACK window lives.
    assert (
        motion.last_authority.require_owned_ack(
            leaf=motion,
            pack=motion.last_pack,
            receipt=receipt,
            owner_row=motion_row,
        )
        is None
    )
    for mutation in (
        {"leaf": racket},
        {"pack": racket.last_pack},
        {"owner_row": racket_row},
        {
            "owner_row": D.OwnerDrainRow(
                owner_kind=motion_row.owner_kind,
                values=motion_row.values,
            )
        },
    ):
        arguments = {
            "leaf": motion,
            "pack": motion.last_pack,
            "receipt": receipt,
            "owner_row": motion_row,
        }
        arguments.update(mutation)
        with pytest.raises(
            D.ActionBallFullMdpPpoDrainError,
            match="lane-swapped",
        ):
            motion.last_authority.require_owned_ack(**arguments)

    owner.acknowledge_post_update(receipt)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="stale"):
        motion.last_authority.require_owned_ack(
            leaf=motion,
            pack=motion.last_pack,
            receipt=receipt,
            owner_row=motion_row,
        )


def test_leaf_ack_authority_rejects_foreign_same_value_coordinator_receipt():
    first, first_leaves = make_owner(2)
    second, _second_leaves = make_owner(2)
    first_prepared = first.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    second_prepared = second.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    first_receipt = first.transfer_decode_pre_optimizer_ppo_boundary(first_prepared)
    second_receipt = second.transfer_decode_pre_optimizer_ppo_boundary(second_prepared)
    first.mark_optimizer_returned(first_receipt)
    second.mark_optimizer_returned(second_receipt)
    motion = first_leaves["motion"]

    assert second_receipt.owner_rows == first_receipt.owner_rows
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign"):
        motion.last_authority.require_owned_ack(
            leaf=motion,
            pack=motion.last_pack,
            receipt=second_receipt,
            owner_row=second_receipt.owner_rows[D.OWNER_ORDER.index("motion")],
        )

    first.acknowledge_post_update(first_receipt)
    second.acknowledge_post_update(second_receipt)


def test_caller_constructed_leaf_ack_authority_cannot_delegate_to_fake_owner():
    class FakeDrainOwner:
        def _require_leaf_ack_authority(self, **kwargs):
            del kwargs
            return None

    schema = D.DEFAULT_LEAF_SCHEMAS[0]
    leaf = FakeLeaf(
        schema.owner_kind,
        num_envs=2,
        schema=schema,
    )
    authority = D.LeafDevicePackAuthority(
        owner_kind=schema.owner_kind,
        schema=schema,
        device=torch.device("cpu"),
        num_envs=2,
        leaf=leaf,
        drain_owner=FakeDrainOwner(),
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="owner-issued"):
        authority.require_owned_ack(
            leaf=leaf,
            pack=object(),
            receipt=object(),
            owner_row=object(),
        )


def test_wrong_pack_return_and_foreign_leaf_mint_are_pretransfer_retryable():
    owner, leaves = make_owner(2)
    motion = leaves["motion"]
    original = motion.prepare_pre_optimizer_ppo_boundary_device_pack

    def wrong_return(**kwargs):
        original(**kwargs)
        return object()

    motion.prepare_pre_optimizer_ppo_boundary_device_pack = wrong_return
    with pytest.raises(D.ActionBallFullMdpPpoDrainPrepareError) as caught:
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=0,
            completed_environment_steps=48,
        )
    assert caught.value.retry_permitted is True
    assert owner.poisoned is False


def test_completed_environment_steps_must_strictly_advance_after_ack():
    owner, _leaves = make_owner(2)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="strictly advance"):
        owner.prepare_pre_optimizer_ppo_boundary(
            update_index=1,
            completed_environment_steps=48,
        )


class _CheckpointBoundary:
    def __init__(self, root):
        self.root = root


def _validate_checkpoint_boundary(boundary):
    if type(boundary) is not _CheckpointBoundary:
        raise RuntimeError("foreign boundary")
    return boundary.root


class _ExternalVerifiedCheckpoint:
    def __init__(self, *, root, content):
        self.root = root
        self.content = content


class _ExternalCheckpointAuthority:
    def __init__(self):
        self._verified = {}

    def register(self, verified):
        self._verified[id(verified)] = verified

    def validator_factory(self, mint_projection):
        def validate(candidate):
            retained = self._verified.get(id(candidate))
            if retained is None or retained is not candidate:
                raise RuntimeError("checkpoint is not externally verified")
            return mint_projection(
                content=retained.content,
                external_checkpoint_root_sha256=retained.root,
            )

        return validate


def _content_with(content, **updates):
    payload = content.canonical_payload()
    payload.update(updates)
    return D.PpoDrainCheckpointContent(
        **payload,
        checkpoint_frontier_sha256=D._canonical_sha256(payload),
    )


def _restore_owner_for(authority, *, num_envs=2, device="cpu", schemas=None):
    return make_owner(
        num_envs,
        device=device,
        schemas=schemas,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )[0]


def _ack_one(owner, *, update_index, completed_environment_steps):
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=update_index,
        completed_environment_steps=completed_environment_steps,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(receipt)
    owner.acknowledge_post_update(receipt)
    return receipt


def test_checkpoint_schema_identity_rejects_wrong_fixed_width_shape():
    owner, _leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(owner, update_index=0, completed_environment_steps=48)
    content = owner.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("1" * 64)
    ).content

    absent_width = list(content.schema_identity)
    first_owner, first_fields = absent_width[0]
    absent_width[0] = (
        first_owner,
        first_fields + (("journal", "fixed", 0),),
    )
    payload = content.canonical_payload()
    payload["schema_identity"] = tuple(absent_width)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="width is absent"):
        D.PpoDrainCheckpointContent(
            **payload,
            checkpoint_frontier_sha256=D._canonical_sha256(payload),
        )

    foreign_width = list(content.schema_identity)
    foreign_width[0] = (
        first_owner,
        first_fields + (("journal", "scalar", 0, 4),),
    )
    payload["schema_identity"] = tuple(foreign_width)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign width"):
        D.PpoDrainCheckpointContent(
            **payload,
            checkpoint_frontier_sha256=D._canonical_sha256(payload),
        )


def test_checkpoint_restore_rejects_same_named_fixed_field_with_wrong_width():
    def schemas_with_width(width):
        schemas = list(D.DEFAULT_LEAF_SCHEMAS)
        first = schemas[0]
        schemas[0] = D.LeafDrainSchema(
            owner_kind=first.owner_kind,
            fields=first.fields
            + (
                D.DeviceDrainFieldSpec(
                    "journal_values",
                    cardinality="fixed",
                    fixed_width=width,
                ),
            ),
        )
        return tuple(schemas)

    source_schemas = schemas_with_width(4)
    source, _leaves = make_owner(
        2,
        schemas=source_schemas,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(source, update_index=0, completed_environment_steps=48)
    content = source.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("3" * 64)
    ).content
    verified = _ExternalVerifiedCheckpoint(root="4" * 64, content=content)
    authority = _ExternalCheckpointAuthority()
    authority.register(verified)
    target, _target_leaves = make_owner(
        2,
        schemas=schemas_with_width(5),
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="inventory"):
        target.restore_checkpoint(verified)


def _runner_projection_checkpoint_payload(projection, boundary_root):
    return {
        "schema_version": projection.schema_version,
        "kind": projection.kind,
        "num_envs": projection.num_envs,
        "device_type": projection.device_type,
        "device_index": projection.device_index,
        "owner_order": projection.owner_order,
        "schema_identity": projection.schema_identity,
        "checkpoint_boundary_sha256": boundary_root,
        "next_update_index": projection.next_update_index,
        "operation_sequence": projection.operation_sequence,
        "drain_sequence": projection.drain_sequence,
        "last_completed_environment_steps": (
            projection.last_completed_environment_steps
        ),
        "mutation_version_highwaters": projection.mutation_version_highwaters,
    }


def test_runner_frontier_projection_is_exact_latest_ack_and_checkpoint_shaped():
    owner, _leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    receipt = _ack_one(
        owner,
        update_index=0,
        completed_environment_steps=48,
    )
    projection = owner.require_owned_runner_frontier_projection(receipt)
    assert owner.require_owned_runner_frontier_projection(receipt) is projection
    assert projection.schema_version == D.SCHEMA_VERSION
    assert projection.kind == D.CHECKPOINT_KIND
    assert projection.num_envs == 2
    assert projection.device_type == "cpu"
    assert projection.device_index is None
    assert projection.owner_order == D.OWNER_ORDER
    assert projection.schema_identity == D._checkpoint_schema_identity(
        D.DEFAULT_LEAF_SCHEMAS
    )
    assert projection.next_update_index == 1
    assert projection.operation_sequence == 1
    assert projection.drain_sequence == 1
    assert projection.last_completed_environment_steps == 48
    assert projection.update_index == 0
    assert projection.completed_environment_steps == 48
    assert projection.next_update_index == projection.update_index + 1
    assert (
        projection.last_completed_environment_steps
        == projection.completed_environment_steps
    )
    assert projection.mutation_version_highwaters == tuple(
        (owner_kind, 3) for owner_kind in D.OWNER_ORDER
    )

    boundary = _CheckpointBoundary("a" * 64)
    snapshot = owner.snapshot_for_checkpoint_boundary(boundary)
    assert _runner_projection_checkpoint_payload(
        projection,
        boundary.root,
    ) == snapshot.content.canonical_payload()
    for hidden in (
        "content",
        "checkpoint_boundary_sha256",
        "checkpoint_frontier_sha256",
        "canonical_payload",
        "expected",
        "global_receipt",
        "source_receipt",
        "owner",
        "owner_identity",
        "root",
        "snapshot",
    ):
        assert not hasattr(projection, hidden)

    with pytest.raises(TypeError, match="owner-issued"):
        D.PpoDrainRunnerFrontierProjection()
    with pytest.raises(AttributeError, match="immutable"):
        projection.next_update_index = 9
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(projection)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(projection)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(projection)
    fabricated = object.__new__(D.PpoDrainRunnerFrontierProjection)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="owner-issued"):
        _ = fabricated.next_update_index
    real_payload = D._lookup_runner_frontier_projection(projection)
    assert real_payload is not None
    privately_minted_copy = D._mint_runner_frontier_projection(real_payload)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="owner-issued"):
        _ = privately_minted_copy.next_update_index


def test_runner_frontier_projection_rejects_preack_foreign_and_caller_copy():
    first, _first_leaves = make_owner(2)
    second, _second_leaves = make_owner(2)
    prepared = first.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    receipt = first.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="latest exact ACKed"):
        first.require_owned_runner_frontier_projection(receipt)
    first.mark_optimizer_returned(receipt)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="latest exact ACKed"):
        first.require_owned_runner_frontier_projection(receipt)
    first.acknowledge_post_update(receipt)

    foreign = _ack_one(
        second,
        update_index=0,
        completed_environment_steps=48,
    )
    assert foreign.owner_rows == receipt.owner_rows
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="latest exact ACKed"):
        first.require_owned_runner_frontier_projection(foreign)

    assembled = D.PreOptimizerPpoBoundaryReceipt(
        owner=first,
        operation_id=1,
        update_index=receipt.update_index,
        completed_environment_steps=receipt.completed_environment_steps,
        drain_sequence=receipt.drain_sequence,
        num_envs=receipt.num_envs,
        rows=receipt.owner_rows,
        token=D._RECEIPT_TOKEN,
    )
    assembled._mark_acknowledged()
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="latest exact ACKed"):
        first.require_owned_runner_frontier_projection(assembled)
    copied = copy.copy(receipt)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="latest exact ACKed"):
        first.require_owned_runner_frontier_projection(copied)


def test_runner_frontier_projection_retires_on_operation_and_tracks_latest_ack():
    owner, _leaves = make_owner(2)
    first_receipt = _ack_one(
        owner,
        update_index=0,
        completed_environment_steps=48,
    )
    first_projection = owner.require_owned_runner_frontier_projection(
        first_receipt
    )
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=1,
        completed_environment_steps=96,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="current"):
        _ = first_projection.operation_sequence
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="idle"):
        owner.require_owned_runner_frontier_projection(first_receipt)

    owner.abort_pre_optimizer_ppo_boundary(prepared)
    after_abort = owner.require_owned_runner_frontier_projection(first_receipt)
    assert after_abort.operation_sequence == 2
    assert after_abort is not first_projection

    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=1,
        completed_environment_steps=96,
    )
    second_receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.mark_optimizer_returned(second_receipt)
    owner.acknowledge_post_update(second_receipt)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="latest exact ACKed"):
        owner.require_owned_runner_frontier_projection(first_receipt)
    second_projection = owner.require_owned_runner_frontier_projection(
        second_receipt
    )
    assert second_projection.update_index == 1
    assert second_projection.next_update_index == 2
    assert second_projection.operation_sequence == 3
    assert second_projection.drain_sequence == 2


def test_runner_frontier_projection_rejects_poisoned_owner_and_retires_object():
    owner, _leaves = make_owner(2)
    first_receipt = _ack_one(
        owner,
        update_index=0,
        completed_environment_steps=48,
    )
    projection = owner.require_owned_runner_frontier_projection(first_receipt)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=1,
        completed_environment_steps=96,
    )
    receipt = owner.transfer_decode_pre_optimizer_ppo_boundary(prepared)
    owner.poison_optimizer_failure(receipt, reason="optimizer failed")
    with pytest.raises(D.ActionBallFullMdpPpoDrainPoisonedError):
        owner.require_owned_runner_frontier_projection(first_receipt)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="current"):
        _ = projection.drain_sequence


def test_runner_frontier_projection_registry_does_not_retain_owner_cycle():
    owner, _leaves = make_owner(2)
    receipt = _ack_one(
        owner,
        update_index=0,
        completed_environment_steps=48,
    )
    projection = owner.require_owned_runner_frontier_projection(receipt)
    owner_ref = weakref.ref(owner)
    projection_ref = weakref.ref(projection)
    del projection, receipt, owner, _leaves
    gc.collect()
    assert owner_ref() is None
    assert projection_ref() is None


def test_checkpoint_snapshot_is_idle_owner_issued_portable_frontier():
    owner, _leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(owner, update_index=0, completed_environment_steps=48)
    boundary = _CheckpointBoundary("1" * 64)

    snapshot = owner.snapshot_for_checkpoint_boundary(boundary)
    assert owner.snapshot_for_checkpoint_boundary(boundary) is snapshot
    assert owner.require_owned_checkpoint_snapshot(boundary, snapshot) is snapshot
    assert snapshot.next_update_index == 1
    assert snapshot.operation_sequence == 1
    assert snapshot.drain_sequence == 1
    assert snapshot.last_completed_environment_steps == 48
    assert snapshot.checkpoint_boundary_sha256 == "1" * 64
    assert snapshot.content.schema_identity == D._checkpoint_schema_identity(
        D.DEFAULT_LEAF_SCHEMAS
    )
    snapshot.content.validate_derived_root()
    with pytest.raises(TypeError, match="owner-issued"):
        D.PpoDrainCheckpointSnapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.content.next_update_index = 9


def test_checkpoint_snapshot_rejects_active_foreign_and_stale_boundary():
    owner, _leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    boundary = _CheckpointBoundary("2" * 64)
    prepared = owner.prepare_pre_optimizer_ppo_boundary(
        update_index=0,
        completed_environment_steps=48,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="no active"):
        owner.snapshot_for_checkpoint_boundary(boundary)
    owner.abort_pre_optimizer_ppo_boundary(prepared)
    snapshot = owner.snapshot_for_checkpoint_boundary(boundary)

    other, _other_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign"):
        other.require_owned_checkpoint_snapshot(boundary, snapshot)
    replacement = _CheckpointBoundary(boundary.root)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="boundary"):
        owner.require_owned_checkpoint_snapshot(replacement, snapshot)


def test_external_authority_restore_continues_frontiers_and_r03_ack_sequence():
    first, _first_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    first_receipt = _ack_one(
        first,
        update_index=0,
        completed_environment_steps=48,
    )
    assert first_receipt.drain_sequence == 1
    boundary = _CheckpointBoundary("3" * 64)
    snapshot = first.snapshot_for_checkpoint_boundary(boundary)
    external = _ExternalVerifiedCheckpoint(
        root="4" * 64,
        content=snapshot.content,
    )
    authority = _ExternalCheckpointAuthority()
    authority.register(external)

    restored, restored_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    restored.restore_checkpoint(external)
    restored.require_exact_leaf_bindings(
        {name: restored_leaves[name] for name in D.OWNER_ORDER}
    )
    restored_leaves["r03_strike_fact"].expected_ack_sequence = 2
    receipt = _ack_one(
        restored,
        update_index=1,
        completed_environment_steps=96,
    )
    assert receipt.drain_sequence == 2
    assert restored.next_update_index == 2


def test_restore_preserves_leaf_identity_join_as_separate_construction_gate():
    first, _first_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(first, update_index=0, completed_environment_steps=48)
    content = first.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("5" * 64)
    ).content
    external = _ExternalVerifiedCheckpoint(root="6" * 64, content=content)
    authority = _ExternalCheckpointAuthority()
    authority.register(external)

    schemas = D.DEFAULT_LEAF_SCHEMAS
    leaves = {
        schema.owner_kind: FakeLeaf(
            schema.owner_kind,
            num_envs=2,
            schema=schema,
        )
        for schema in schemas
    }
    restored = D.ActionBallFullMdpPpoDrainOwner(
        num_envs=2,
        device="cpu",
        leaves=leaves,
        diagnostic_allow_minimal_schemas=True,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
        checkpoint_restore_validator_factory=authority.validator_factory,
    )
    restored.restore_checkpoint(external)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="binding join"):
        restored.prepare_pre_optimizer_ppo_boundary(
            update_index=1,
            completed_environment_steps=96,
        )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="closed"):
        restored.require_exact_leaf_bindings(
            {name: leaves[name] for name in D.OWNER_ORDER}
        )

    second_authority = _ExternalCheckpointAuthority()
    second_authority.register(external)
    second, second_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=second_authority.validator_factory,
        join_leaf_bindings=False,
    )
    second.restore_checkpoint(external)
    assert second.require_exact_leaf_bindings(
        {name: second_leaves[name] for name in D.OWNER_ORDER}
    ) is None
    receipt = _ack_one(
        second,
        update_index=1,
        completed_environment_steps=96,
    )
    assert receipt.drain_sequence == 2


def test_restore_rejects_foreign_resealed_mapping_stale_schema_and_reuse():
    first, _first_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(first, update_index=0, completed_environment_steps=48)
    content = first.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("7" * 64)
    ).content
    authority = _ExternalCheckpointAuthority()
    external = _ExternalVerifiedCheckpoint(root="8" * 64, content=content)
    authority.register(external)

    foreign_owner, _foreign_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="rejected"):
        foreign_owner.restore_checkpoint(
            _ExternalVerifiedCheckpoint(root=external.root, content=content)
        )

    mapping_owner, _mapping_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="rejected"):
        mapping_owner.restore_checkpoint(content.canonical_payload())

    stale_fields = list(content.schema_identity)
    owner_kind, fields = stale_fields[-1]
    stale_fields[-1] = (
        owner_kind,
        fields + (("future_schema_field", "scalar", 0),),
    )
    stale_payload = content.canonical_payload()
    stale_payload["schema_identity"] = tuple(stale_fields)
    stale_content = D.PpoDrainCheckpointContent(
        **stale_payload,
        checkpoint_frontier_sha256=D._canonical_sha256(stale_payload),
    )
    stale = _ExternalVerifiedCheckpoint(root="9" * 64, content=stale_content)
    authority.register(stale)
    stale_owner, _stale_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="inventory"):
        stale_owner.restore_checkpoint(stale)

    reused_owner, _reused_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    reused_owner.restore_checkpoint(external)
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="already closed"):
        reused_owner.restore_checkpoint(external)


def test_restore_rejects_active_or_nonfresh_owner_and_self_consistent_forge():
    source, _source_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(source, update_index=0, completed_environment_steps=48)
    content = source.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("a" * 64)
    ).content
    authority = _ExternalCheckpointAuthority()
    external = _ExternalVerifiedCheckpoint(root="b" * 64, content=content)
    authority.register(external)

    active, _active_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    # Any construction read/join/runtime attempt closes the restore gate.
    assert active.next_update_index == 0
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="fresh idle"):
        active.restore_checkpoint(external)

    forged_payload = content.canonical_payload()
    forged_payload["next_update_index"] = 99
    forged = D.PpoDrainCheckpointContent(
        **forged_payload,
        checkpoint_frontier_sha256=D._canonical_sha256(forged_payload),
    )
    forged_external = _ExternalVerifiedCheckpoint(root="c" * 64, content=forged)
    # A self-consistent root is still rejected because external authority did
    # not register this exact VerifiedCheckpoint identity.
    forged_owner, _forged_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=authority.validator_factory,
        join_leaf_bindings=False,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="rejected"):
        forged_owner.restore_checkpoint(forged_external)


@pytest.mark.parametrize(
    ("updates", "failure"),
    (
        ({"operation_sequence": 0}, "operation sequence"),
        ({"next_update_index": 0}, "next update index"),
        ({"drain_sequence": 0}, "empty drain frontier"),
        ({"last_completed_environment_steps": -1}, "mutation highwaters"),
    ),
)
def test_restore_rejects_invalid_frontier_chronology(updates, failure):
    source, _source_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    _ack_one(source, update_index=0, completed_environment_steps=48)
    content = source.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("d" * 64)
    ).content
    invalid = _ExternalVerifiedCheckpoint(
        root="e" * 64,
        content=_content_with(content, **updates),
    )
    authority = _ExternalCheckpointAuthority()
    authority.register(invalid)

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match=failure):
        _restore_owner_for(authority).restore_checkpoint(invalid)


def test_restore_rejects_partial_mutation_highwaters():
    source, _source_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    content = source.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("f" * 64)
    ).content
    partial = tuple(
        (owner_kind, 0 if index == 0 else None)
        for index, owner_kind in enumerate(D.OWNER_ORDER)
    )
    invalid = _ExternalVerifiedCheckpoint(
        root="0" * 64,
        content=_content_with(
            content,
            mutation_version_highwaters=partial,
        ),
    )
    authority = _ExternalCheckpointAuthority()
    authority.register(invalid)

    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="partially absent"):
        _restore_owner_for(authority).restore_checkpoint(invalid)


@pytest.mark.parametrize("mismatch", ("boundary", "device", "num_envs"))
def test_restore_rejects_wrong_boundary_device_or_num_envs(mismatch):
    source, _source_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    content = source.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("1" * 64)
    ).content
    updates = {}
    if mismatch == "boundary":
        updates["checkpoint_boundary_sha256"] = "2" * 64
    elif mismatch == "device":
        updates["device_type"] = "cuda"
        updates["device_index"] = 0
    else:
        updates["num_envs"] = 3
    invalid = _ExternalVerifiedCheckpoint(
        root="3" * 64,
        content=_content_with(content, **updates),
    )
    authority = _ExternalCheckpointAuthority()
    if mismatch != "boundary":
        authority.register(invalid)
    owner = _restore_owner_for(authority)

    if mismatch == "boundary":
        # Boundary identity is externally meaningful.  The global owner binds
        # it into content; R10 must additionally require equality to the exact
        # VerifiedCheckpoint boundary.  This fixture authority deliberately
        # models that check and rejects the mismatched identity.
        with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="rejected"):
            owner.restore_checkpoint(invalid)
    else:
        with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="differs"):
            owner.restore_checkpoint(invalid)


def test_restore_projection_cannot_replay_across_fresh_owners():
    source, _source_leaves = make_owner(
        2,
        checkpoint_boundary_validator=_validate_checkpoint_boundary,
    )
    content = source.snapshot_for_checkpoint_boundary(
        _CheckpointBoundary("4" * 64)
    ).content
    external = _ExternalVerifiedCheckpoint(root="5" * 64, content=content)
    retained_projection = []

    def retaining_factory(mint_projection):
        def validate(candidate):
            assert candidate is external
            if not retained_projection:
                retained_projection.append(
                    mint_projection(
                        content=content,
                        external_checkpoint_root_sha256=external.root,
                    )
                )
            return retained_projection[0]

        return validate

    first, _first_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=retaining_factory,
        join_leaf_bindings=False,
    )
    first.restore_checkpoint(external)
    second, _second_leaves = make_owner(
        2,
        checkpoint_restore_validator_factory=retaining_factory,
        join_leaf_bindings=False,
    )
    with pytest.raises(D.ActionBallFullMdpPpoDrainError, match="foreign or reused"):
        second.restore_checkpoint(external)
