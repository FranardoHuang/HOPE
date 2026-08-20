"""Focused CPU tests for immutable-epoch lean Reward decoding."""

from __future__ import annotations

from dataclasses import replace
import importlib
import os
from pathlib import Path
import pickle
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
from test_action_ball_full_mdp_epoch_rowwise import (
    E,
    _RealSelectedResetHarness,
    _key,
    _ready_epoch,
    _terminal_reset_facts,
)
R = importlib.import_module(
    "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_lean_rewards"
)
TEST_DEVICE = os.environ.get("ACTION_BALL_LEAN_REWARD_TEST_DEVICE", "cpu")


def _tensor(data, *, dtype=None):
    return torch.tensor(data, dtype=dtype, device=TEST_DEVICE)


def _epoch(*, placement_gain=2.0, bind_selected_reset=False):
    owner, d05, _cadence, r06_owner, _playback, *_middle, physical_owner = (
        _ready_epoch(device=TEST_DEVICE)
    )
    selected_reset = None
    if bind_selected_reset:
        selected_reset = _RealSelectedResetHarness(owner)
        owner.bind_selected_reset_owner(selected_reset.owner)
    valid = torch.ones((2, 1), dtype=torch.bool, device=TEST_DEVICE)
    values = _tensor([[1], [2]], dtype=torch.int64)
    candidate = d05.candidate
    d05.candidate = replace(
        candidate,
        identity=replace(candidate.identity, shot_key=_key(values, valid)),
        task=replace(candidate.task, task_valid=valid),
        construction_admissible=valid,
    )
    producers = {
        "r03_strike_fact": object(),
        "physical_ball": physical_owner,
        "r06_landing_outcome": r06_owner,
        "r07_recovery": object(),
    }
    owner.bind_fact_owner("r03_strike_fact", producers["r03_strike_fact"])
    owner.bind_fact_owner("r07_recovery", producers["r07_recovery"])
    owner.prepare_after_command_rows()
    owner.settle_d05_transaction(d05.arm())
    # This focused decoder fixture represents a settled shot.  REVEAL and
    # LAUNCH are covered by the live producer tests and must not pay R07.
    publication = owner._publication
    record = publication.current
    owner._publication = E._Publication(
        replace(
            record,
            phase=torch.full_like(record.phase, E.PHASE_OUTCOME_SETTLED),
        ),
        publication.pending_log,
    )

    def publish(name, bits, values, faults=None):
        if faults is not None:
            owner.merge_runtime_owner_fault(name, faults)
        owner.publish_owner_facts(
            name,
            owner=producers[name],
            valid_bits=bits,
            source_step=torch.full(
                (2, 1), 8, dtype=torch.int64, device=TEST_DEVICE
            ),
            values=values,
        )

    values = torch.zeros(
        (2, 1, E.OWNER_FACT_F32_WIDTH),
        dtype=torch.float32,
        device=TEST_DEVICE,
    )
    # Perfect R03 target/achieved position/velocity/normal and ball at paddle.
    values[:, :, 6:9] = _tensor((0.0, 0.0, 1.0))
    values[:, :, 21:24] = _tensor((0.0, 0.0, 1.0))
    publish(
        "r03_strike_fact",
        torch.full(
            (2, 1),
            R.R03_PRESENT | R.R03_PHYSICALLY_VALID,
            dtype=torch.int64,
            device=TEST_DEVICE,
        ),
        values,
    )
    publish(
        "physical_ball",
        _tensor(
            [
                [R.PHYSICAL_PRESENT | R.PHYSICAL_SELECTED_CONTACT],
                [R.PHYSICAL_PRESENT],
            ],
            dtype=torch.int64,
        ),
        torch.zeros_like(values),
    )
    r06 = torch.zeros_like(values)
    r06[:, :, 0] = _tensor((1.0, 0.0)).reshape(2, 1)
    r06[:, :, 1] = 0.5
    r06[:, :, 2] = placement_gain
    publish(
        "r06_landing_outcome",
        torch.full(
            (2, 1),
            R.R06_PRESENT | R.R06_POLICY_ELIGIBLE | R.R06_SOURCE_VALID,
            dtype=torch.int64,
            device=TEST_DEVICE,
        ),
        r06,
    )
    r07 = torch.zeros_like(values)
    r07[:, :, 0] = _tensor((0.25, -0.5)).reshape(2, 1)
    r07[:, :, 2:4] = 1.0
    r07[:, :, 6] = 10.0
    publish(
        "r07_recovery",
        torch.full(
            (2, 1),
            R.R07_PRESENT | R.R07_NUMERICALLY_VALID,
            dtype=torch.int64,
            device=TEST_DEVICE,
        ),
        r07,
    )
    owner._test_selected_reset = selected_reset
    return owner


def _pay_all(
    graph, *, close_actual=True, manager_weights=None, manager_dt=None,
    dense_values=None,
):
    values = []
    actual = torch.zeros(graph.num_envs, dtype=torch.float32, device=TEST_DEVICE)
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        value = graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
        values.append(value)
        if manager_weights is None:
            term = value * graph._milestone_configured_income_scale[ordinal].to(
                torch.float32
            )
        else:
            term = value * float(manager_weights[ordinal])
            term = term * float(manager_dt)
        actual.add_(term)
    dense_values = dense_values or tuple(
        torch.ones(graph.num_envs, dtype=torch.float32, device=TEST_DEVICE)
        for _ in R.COMMON_DENSE_NAMES
    )
    for offset, value in enumerate(dense_values):
        ordinal = R.LIFECYCLE_PAYMENT_COUNT + offset
        values.append(graph.record_common_dense(ordinal, value))
        if manager_weights is None:
            term = value * graph._milestone_configured_income_scale[ordinal].to(
                torch.float32
            )
        else:
            term = value * float(manager_weights[ordinal]) * float(manager_dt)
        actual.add_(term)
    if close_actual:
        graph.close_milestone_actual_reward(actual)
    return values


class _ExactEnvRewardDispatcherRepresentation:
    """Focused mirror of the source-defined env dispatcher ABI."""

    def __init__(self, graph):
        self._test_reward_graph = graph

    def _action_ball_full_mdp_lean_reward_term(
        self, *, ordinal: int, scale: float | None = None, value=None
    ) -> torch.Tensor:
        if ordinal < R.LIFECYCLE_PAYMENT_COUNT:
            return self._test_reward_graph.pay(ordinal, scale=scale)
        return self._test_reward_graph.record_common_dense(ordinal, value)


def test_exact_fourteen_lifecycle_plus_six_dense_complete_once():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    values = _pay_all(graph)
    assert graph.completed_cycle_count == 1
    assert graph.actual_closed_cycle_count == 1
    assert graph.cycle_open is False
    for value in values:
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
    assert torch.equal(values[0], torch.ones(2, device=TEST_DEVICE))
    assert torch.equal(values[10], _tensor((1.0, 0.0)))
    assert torch.equal(values[11], _tensor((1.0, 0.0)))
    assert torch.equal(values[12], torch.ones(2, device=TEST_DEVICE))
    assert torch.equal(values[13], _tensor((0.25, -0.5)))


def test_r07_retired_row_cannot_replay_fact_or_mask_peer():
    owner = _epoch()
    publication = owner._publication
    record = publication.current
    phase = record.phase.clone()
    phase[0, 0] = E.PHASE_RETIRED
    before_peer = record.fact_f32[1, 0, E.OWNER_ORDER.index("r07_recovery")].clone()
    owner._publication = E._Publication(
        replace(record, phase=phase), publication.pending_log
    )

    values = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner))
    assert torch.equal(values[13], _tensor((0.0, -0.5)))
    after = owner.current()
    assert torch.equal(
        after.fact_f32[1, 0, E.OWNER_ORDER.index("r07_recovery")],
        before_peer,
    )


def test_selected_true_reset_clears_only_its_r07_row_and_peer_still_pays():
    owner = _epoch(bind_selected_reset=True)
    reset = owner._test_selected_reset
    before = owner.current()
    r07_slot = E.OWNER_ORDER.index("r07_recovery")
    peer_fact = before.fact_f32[1, 0, r07_slot].clone()
    peer_bits = before.fact_valid_bits[1, 0, r07_slot].clone()

    selected_index = _tensor([0], dtype=torch.int64)
    selected = _tensor([True, False], dtype=torch.bool)
    generation_before = before.reset_generation.clone()
    generation_after = generation_before + selected.to(torch.int64)
    overflow = torch.zeros_like(selected)
    terminal = _terminal_reset_facts(selected)
    top = reset.arm_preflight(
        selected_env_index=selected_index,
        selected_mask=selected,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal,
    )
    lease = owner.prepare_selected_true_reset(
        owner=reset.owner,
        top_preflight=top,
        selected_env_index=selected_index,
        selected_mask=selected,
        generation_before=generation_before,
        generation_after=generation_after,
        generation_overflow_fault=overflow,
        terminal_reset_facts_i64=terminal,
    )
    reset.arm_commit(lease)
    after = owner.commit_selected_true_reset(
        owner=reset.owner, prepared_reset=lease
    )
    assert after.phase[:, 0].tolist() == [
        E.PHASE_IDLE,
        E.PHASE_OUTCOME_SETTLED,
    ]
    assert after.fact_valid_bits[0, 0, r07_slot].item() == 0
    assert after.fact_f32[0, 0, r07_slot].eq(0).all()
    assert torch.equal(after.fact_valid_bits[1, 0, r07_slot], peer_bits)
    assert torch.equal(after.fact_f32[1, 0, r07_slot], peer_fact)

    values = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner))
    assert torch.equal(values[13], _tensor((0.0, -0.5)))


def test_real_consumers_reduce_primitive_eligibility_and_configured_income():
    owner = _epoch()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    cfg = {
        name: types.SimpleNamespace(weight=-2.0 if ordinal == 0 else 1.0)
        for ordinal, name in enumerate(R.MANAGER_NAMES)
    }
    graph.configure_milestone_configured_income(cfg, 0.25)
    weights = tuple(
        -2.0 if ordinal == 0 else 1.0 for ordinal in range(R.MANAGER_TERM_COUNT)
    )
    _pay_all(graph, manager_weights=weights, manager_dt=0.25)
    reward_i = owner.milestone.i64[: 20 * 4].reshape(20, 4)
    reward_f = owner.milestone.f64[: 20 * 7].reshape(20, 7)
    assert reward_i[0].tolist() == [2, 2, 2, 2]
    assert reward_f[0, 0].item() == 2.0
    assert reward_f[0, 3].item() == -1.0
    assert reward_f[0, 6].item() == 1.0
    assert reward_i[10].tolist() == [2, 2, 2, 1]
    assert reward_f[12, 0].item() == 1.0
    assert reward_f[12, 2].item() == 2.0
    assert reward_f[13, 0].item() == 0.0  # R07 raw score, not leaf-weighted payment.
    assert reward_f[13, 2].item() == -0.25


def test_actual_conservation_accepts_pinned_float32_two_multiply_order():
    owner = _epoch()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    weights = tuple(0.13 + ordinal * 0.017 for ordinal in range(20))
    dt = 0.031
    cfg = {
        name: types.SimpleNamespace(weight=weights[ordinal])
        for ordinal, name in enumerate(R.MANAGER_NAMES)
    }
    graph.configure_milestone_configured_income(cfg, dt)
    _pay_all(graph, manager_weights=weights, manager_dt=dt)
    milestone = owner.milestone
    violation_index = 20 * 4 + 3
    assert milestone.i64[violation_index].item() == 0
    start, end = owner.prepare_drain()
    materialized = owner.materialize_drain(start=start, end=end)
    E.milestone_tensors.decode_host_window(
        materialized.milestone_i64, materialized.milestone_f64
    )

    # A manager-weight or dt mutation is independent of the configured-term
    # scratch and must be rejected even when the window total looks plausible.
    for mutate_weight in (True, False):
        mutant_owner = _epoch()
        mutant = R.LeanActionEpochRewardGraph(epoch_owner=mutant_owner)
        mutant.configure_milestone_configured_income(cfg, dt)
        bad_weights = list(weights)
        bad_dt = dt
        if mutate_weight:
            bad_weights[3] += 0.01
        else:
            bad_dt += 0.001
        _pay_all(
            mutant,
            manager_weights=tuple(bad_weights),
            manager_dt=bad_dt,
        )
        assert mutant_owner.milestone.i64[violation_index].item() > 0


def test_ordinal_twelve_primitive_and_payment_are_independently_identified():
    owner_a = _epoch(placement_gain=2.0)
    owner_c = _epoch(placement_gain=3.0)
    values_a = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner_a))
    values_c = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner_c))
    assert torch.equal(values_a[11], values_c[11])
    assert torch.equal(values_a[12], torch.ones(2, device=TEST_DEVICE))
    assert torch.equal(values_c[12], torch.full((2,), 1.5, device=TEST_DEVICE))
    reward_a = owner_a.milestone.f64[: 20 * 7].reshape(20, 7)
    reward_c = owner_c.milestone.f64[: 20 * 7].reshape(20, 7)
    assert reward_a[12, 0].item() == reward_c[12, 0].item() == 1.0
    assert reward_a[12, 2].item() == 2.0
    assert reward_c[12, 2].item() == 3.0


def test_skipped_or_duplicate_ordinal_does_not_increment_completion():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    with pytest.raises(R.LeanRewardCycleError, match="expected 0"):
        graph.pay(1, scale=1.0)
    assert graph.completed_cycle_count == 0
    graph.pay(0, scale=1.0)
    with pytest.raises(R.LeanRewardCycleError, match="expected 1"):
        graph.pay(0, scale=1.0)
    assert graph.completed_cycle_count == 0


def test_actual_close_rejects_partial_missing_and_duplicate_cycles():
    partial = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    partial.pay(0, scale=1.0)
    with pytest.raises(R.LeanRewardCycleError, match="before all twenty"):
        partial.close_milestone_actual_reward(torch.zeros(2, device=TEST_DEVICE))
    assert partial.poisoned is True

    missing = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    _pay_all(missing, close_actual=False)
    with pytest.raises(R.LeanRewardCycleError, match="lacks its actual-buffer close"):
        missing.pay(0, scale=1.0)
    assert missing.poisoned is True

    duplicate = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    _pay_all(duplicate)
    with pytest.raises(R.LeanRewardCycleError, match="skipped, duplicated, or replayed"):
        duplicate.close_milestone_actual_reward(
            torch.zeros(2, device=TEST_DEVICE)
        )
    assert duplicate.poisoned is True


def test_producer_fault_suppresses_only_its_reward_row():
    owner = _epoch()
    owner.merge_runtime_owner_fault(
        "physical_ball",
        _tensor([[0], [32]], dtype=torch.int64),
        owner=owner._fact_owner_identities["physical_ball"],
    )
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    values = _pay_all(graph)
    assert torch.equal(values[10], _tensor((1.0, 0.0)))
    assert graph.completed_cycle_count == 1


def test_materializer_refuses_missing_real_isaac_import(monkeypatch):
    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("isaaclab")),
    )
    with pytest.raises(R.LeanRewardConstructionHold, match="RewardTermCfg import"):
        R.materialize_reward_manager_cfg(weights={}, r03_scales={})


def test_materializer_builds_exact_order_with_real_type_surface(monkeypatch):
    class RewardTermCfg:
        def __init__(self, *, func, weight, params):
            self.func, self.weight, self.params = func, weight, params

    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(RewardTermCfg=RewardTermCfg),
    )
    weights = dict(R.DIAGNOSTIC_N2_WEIGHTS)
    scales = {name: 0.5 for name in R.R03_NAMES}
    cfg = R.materialize_reward_manager_cfg(weights=weights, r03_scales=scales)
    assert tuple(cfg) == R.MANAGER_NAMES
    assert tuple(term.func for term in cfg.values()) == R.REWARD_TERM_CALLABLES
    assert all(type(term) is RewardTermCfg for term in cfg.values())
    assert all(
        term.params == {"scale": 0.5}
        for term in tuple(cfg.values())[: len(R.R03_NAMES)]
    )
    assert all(
        term.params == {}
        for term in tuple(cfg.values())[len(R.R03_NAMES):R.LIFECYCLE_PAYMENT_COUNT]
    )
    assert tuple(term.params["ordinal"] for term in tuple(cfg.values())[14:]) == tuple(range(14, 20))


def test_manager_cfg_and_each_callable_pickle_by_exact_module_global(monkeypatch):
    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(RewardTermCfg=types.SimpleNamespace),
    )
    weights = dict(R.DIAGNOSTIC_N2_WEIGHTS)
    scales = {name: 0.5 for name in R.R03_NAMES}
    cfg = R.materialize_reward_manager_cfg(weights=weights, r03_scales=scales)

    for name, function in zip(
        R.LIFECYCLE_MANAGER_NAMES, R.LIFECYCLE_REWARD_TERM_CALLABLES
    ):
        assert getattr(R, name) is function
        assert function.__module__ == R.__name__
        assert function.__name__ == name
        assert function.__qualname__ == name
        assert function.__closure__ is None
        assert pickle.loads(pickle.dumps(function)) is function

    restored = pickle.loads(pickle.dumps(cfg))
    assert tuple(restored) == R.MANAGER_NAMES
    assert tuple(term.func for term in restored.values()) == R.REWARD_TERM_CALLABLES


def test_module_global_functions_dispatch_exact_fourteen_lifecycle_groups():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    env = _ExactEnvRewardDispatcherRepresentation(graph)
    values = []
    for ordinal, name in enumerate(R.LIFECYCLE_MANAGER_NAMES):
        function = getattr(R, name)
        values.append(function(env, scale=1.0 if ordinal < 10 else None))
    for ordinal in range(14, 20):
        values.append(
            env._action_ball_full_mdp_lean_reward_term(
                ordinal=ordinal, value=torch.ones(2, device=TEST_DEVICE)
            )
        )
    assert graph.completed_cycle_count == 1
    assert torch.equal(values[0], torch.ones(2, device=TEST_DEVICE))
    assert torch.equal(values[10], _tensor((1.0, 0.0)))
    assert torch.equal(values[11], _tensor((1.0, 0.0)))
    assert torch.equal(values[12], torch.ones(2, device=TEST_DEVICE))
    assert torch.equal(values[13], _tensor((0.25, -0.5)))


def test_common_dense_reuses_motion_evaluator_and_changes_with_reference_error(
    monkeypatch,
):
    def evaluator(env, *, command_name, std):
        assert command_name == "motion"
        return torch.exp(-torch.square(env.reference - env.robot) / std**2)

    evaluator_module = types.SimpleNamespace(
        **{spec[1]: evaluator for spec in R.COMMON_DENSE_SPECS}
    )
    real_import = R.importlib.import_module
    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda name: evaluator_module
        if name.endswith(".rewards")
        else real_import(name),
    )
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    env = _ExactEnvRewardDispatcherRepresentation(graph)
    env.reference = torch.tensor([0.0, 1.0], device=TEST_DEVICE)
    env.robot = torch.tensor([0.0, 0.0], device=TEST_DEVICE)
    for ordinal in range(14):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    values = []
    for ordinal, spec in enumerate(R.COMMON_DENSE_SPECS, start=14):
        values.append(
            R.common_dense_reward(
                env, ordinal=ordinal, command_name="motion", std=spec[3]
            )
        )
    assert graph.completed_cycle_count == 1
    assert all(torch.isfinite(value).all() for value in values)
    assert all(value[0] > value[1] for value in values)


def test_dense_rows_ignore_lifecycle_paid_bits_and_exist_when_lifecycle_is_zero():
    owner = _epoch()
    owner._publication.current.fact_valid_bits.zero_()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    lifecycle = tuple(
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
        for ordinal in range(14)
    )
    assert all(not bool(value.any()) for value in lifecycle)
    owner._publication.current.reward_paid.logical_not_()
    dense = tuple(
        graph.record_common_dense(ordinal, torch.ones(2, device=TEST_DEVICE))
        for ordinal in range(14, 20)
    )
    assert all(torch.equal(value, torch.ones(2, device=TEST_DEVICE)) for value in dense)
    assert graph.completed_cycle_count == 1


@pytest.mark.parametrize("bad_ordinal", (15, 19))
def test_dense_skip_or_reorder_fails_before_actual_close(bad_ordinal):
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    for ordinal in range(14):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    with pytest.raises(R.LeanRewardCycleError, match="expected 14"):
        graph.record_common_dense(
            bad_ordinal, torch.ones(2, device=TEST_DEVICE)
        )
    assert graph.completed_cycle_count == 0
    assert graph.poisoned is True


def test_dense_duplicate_fails_before_actual_close():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    for ordinal in range(14):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    graph.record_common_dense(14, torch.ones(2, device=TEST_DEVICE))
    with pytest.raises(R.LeanRewardCycleError, match="expected 15"):
        graph.record_common_dense(14, torch.ones(2, device=TEST_DEVICE))
    assert graph.completed_cycle_count == 0
    assert graph.poisoned is True


def test_reward_function_rejects_instance_shadow_dispatcher():
    env = _ExactEnvRewardDispatcherRepresentation(
        R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    )
    env.__dict__[R.ENV_REWARD_DISPATCHER_NAME] = lambda **_kwargs: None
    with pytest.raises(R.LeanRewardConstructionHold, match="class-owned"):
        R.racket_position(env, scale=1.0)


def test_reward_function_rejects_inherited_foreign_dispatcher():
    class ForeignBase:
        def _action_ball_full_mdp_lean_reward_term(
            self, *, ordinal: int, scale: float | None = None
        ):
            del ordinal, scale
            return None

    class EnvWithForeignInheritedDispatcher(ForeignBase):
        pass

    with pytest.raises(R.LeanRewardConstructionHold, match="class-owned"):
        R.racket_position(EnvWithForeignInheritedDispatcher(), scale=1.0)


def test_reward_function_rejects_foreign_bound_dispatcher():
    def foreign_dispatcher(_self, *, ordinal: int, scale: float | None = None):
        del ordinal, scale
        return None

    class EnvReturningForeignBoundDispatcher:
        def _action_ball_full_mdp_lean_reward_term(
            self, *, ordinal: int, scale: float | None = None
        ):
            del ordinal, scale
            return None

        def __getattribute__(self, name):
            if name == R.ENV_REWARD_DISPATCHER_NAME:
                return types.MethodType(foreign_dispatcher, self)
            return object.__getattribute__(self, name)

    with pytest.raises(R.LeanRewardConstructionHold, match="binding differs"):
        R.racket_position(EnvReturningForeignBoundDispatcher(), scale=1.0)


def test_reward_function_rejects_non_function_class_descriptor():
    class EnvWithForeignDescriptor:
        _action_ball_full_mdp_lean_reward_term = staticmethod(
            lambda **_kwargs: None
        )

    with pytest.raises(R.LeanRewardConstructionHold, match="class-owned"):
        R.racket_position(EnvWithForeignDescriptor(), scale=1.0)


def test_diagnostic_bundle_has_no_caller_numeric_seam(monkeypatch):
    owner = _epoch()

    class RewardTermCfg:
        def __init__(self, *, func, weight, params):
            self.func, self.weight, self.params = func, weight, params

    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(RewardTermCfg=RewardTermCfg),
    )
    bundle = R.materialize_diagnostic_n2_reward_manager_cfg(epoch_owner=owner)
    assert type(bundle) is R.DiagnosticN2RewardManagerBundle
    assert type(bundle.graph) is R.LeanActionEpochRewardGraph
    assert tuple(bundle.manager_cfg) == R.MANAGER_NAMES
    assert bundle.profile_kind == R.DIAGNOSTIC_N2_REWARD_PROFILE_KIND
    assert bundle.diagnostic_unauthorized is True
    assert bundle.manager_cfg["common_on_table_outcome"].weight == 20.0
    assert bundle.manager_cfg["racket_position"].params["scale"] == 0.2


def test_private_carry_binds_manager_names_and_device_configured_income():
    source_owner = _epoch()
    source = R.LeanActionEpochRewardGraph(epoch_owner=source_owner)
    cfg = {
        name: types.SimpleNamespace(weight=0.5 + ordinal)
        for ordinal, name in enumerate(R.MANAGER_NAMES)
    }
    source.configure_milestone_configured_income(cfg, 0.02)
    _pay_all(source)
    source_owner.milestone.freeze_window_()
    source_owner.milestone.clear_window_()
    source_marker = object()
    source._lean_carry_coordinator = source_marker
    source_lease = types.SimpleNamespace(coordinator=source_marker, kind="capture")
    capture = source._lean_carry_capture(source_lease)
    assert capture.scalars[2] == tuple(R.MANAGER_NAMES)
    assert capture.scalars[3] == tuple(
        0.02 * (0.5 + ordinal) for ordinal in range(R.MANAGER_TERM_COUNT)
    )
    assert torch.equal(capture.tensors[0], source._milestone_configured_income_scale)

    target_owner = _epoch()
    target = R.LeanActionEpochRewardGraph(epoch_owner=target_owner)
    target.configure_milestone_configured_income(cfg, 0.02)
    target_marker = types.SimpleNamespace(_active_lease=None)
    target._lean_carry_coordinator = target_marker
    target_lease = types.SimpleNamespace(coordinator=target_marker, kind="prepare")
    target_marker._active_lease = target_lease
    host = (capture.tensors[0].detach().clone().contiguous(),)
    stage = target._lean_carry_stage(target_lease, capture.scalars, host)
    armed = type(stage)(stage.scalars, stage.staging, stage.targets, True)
    target._lean_carry_apply_scalars(target_lease, armed)
    assert target.completed_cycle_count == 1
    assert target.actual_closed_cycle_count == 1
    assert target.cycle_open is False
    _pay_all(target)
    assert target.completed_cycle_count == 2
    assert target.actual_closed_cycle_count == 2


def test_configure_after_root_registration_preserves_construction_identity():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    registered = graph._lean_carry_construction_views()
    assert len(registered) == 1
    assert registered[0] is graph._milestone_configured_income_scale
    registered_id = id(registered[0])
    cfg = {
        name: types.SimpleNamespace(weight=0.5 + ordinal)
        for ordinal, name in enumerate(R.MANAGER_NAMES)
    }

    graph.configure_milestone_configured_income(cfg, 0.02)

    current = graph._lean_carry_construction_views()
    assert id(current[0]) == registered_id
    assert current[0] is registered[0]
    assert torch.equal(
        current[0],
        torch.tensor(
            tuple(0.02 * (0.5 + ordinal) for ordinal in range(R.MANAGER_TERM_COUNT)),
            dtype=torch.float64,
            device=TEST_DEVICE,
        ),
    )


def test_source_has_no_leaf_callbacks_or_retired_authority_graph():
    source = (MDP / "action_ball_full_mdp_lean_rewards.py").read_text(encoding="utf-8")
    for marker in (
        "action_epoch_reward_facts_v1",
        "selected_contact_reward_view",
        "record_reward_payment",
        "action_ball_full_mdp_rewards",
        "receipt_sha256",
        "numeric_authority",
    ):
        assert marker not in source
