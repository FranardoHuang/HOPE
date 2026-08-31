"""Focused CPU tests for immutable-epoch lean Reward decoding."""

from __future__ import annotations

from dataclasses import replace
import importlib
import inspect
import math
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
TEST_DEVICE = torch.device(
    os.environ.get("ACTION_BALL_LEAN_REWARD_TEST_DEVICE", "cpu")
)


def _tensor(data, *, dtype=None):
    return torch.tensor(data, dtype=dtype, device=TEST_DEVICE)


def _paddle_telemetry_kwargs(ordinal, value):
    if ordinal != R.PADDLE_MOTION_PRIOR_FIRST_ORDINAL:
        return {}
    return {
        "paddle_playback_active": torch.ones(
            value.shape, dtype=torch.bool, device=value.device
        ),
        "paddle_error_components": torch.zeros(
            (value.shape[0], len(R.PADDLE_MOTION_PRIOR_SPECS)),
            dtype=value.dtype,
            device=value.device,
        ),
        "paddle_contact_scale": torch.ones(
            value.shape, dtype=value.dtype, device=value.device
        ),
    }


def _epoch(*, placement_gain=2.0, bind_selected_reset=False):
    owner, d05, _cadence, r06_owner, _playback, *_middle, physical_owner = (
        _ready_epoch(reward_age=7, device=TEST_DEVICE)
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
            settlement_step=torch.full_like(record.settlement_step, 8),
        ),
        publication.pending_log,
    )

    def publish(name, bits, values, faults=None, source_step=None):
        if faults is not None:
            owner.merge_runtime_owner_fault(name, faults)
        owner.publish_owner_facts(
            name,
            owner=producers[name],
            valid_bits=bits,
            source_step=(
                torch.full(
                    (2, 1), 8, dtype=torch.int64, device=TEST_DEVICE
                )
                if source_step is None
                else source_step
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
        source_step=_tensor([[8], [-1]], dtype=torch.int64),
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
        source_step=torch.full(
            (2, 1), 7, dtype=torch.int64, device=TEST_DEVICE
        ),
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
    dense_values = dense_values or (
        tuple(
            torch.ones(
                graph.num_envs, dtype=torch.float32, device=TEST_DEVICE
            )
            for _ in R.ALL_DENSE_SPECS
        )
        + tuple(
            torch.zeros(
                graph.num_envs, dtype=torch.float32, device=TEST_DEVICE
            )
            for _ in R.REGULARIZATION_SPECS
        )
    )
    for offset, value in enumerate(dense_values):
        ordinal = R.LIFECYCLE_PAYMENT_COUNT + offset
        values.append(
            graph.record_common_dense(
                ordinal, value, **_paddle_telemetry_kwargs(ordinal, value)
            )
        )
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
        joint_names = tuple(f"joint_{index}" for index in range(31))
        zeros = torch.zeros((graph.num_envs, 31), device=TEST_DEVICE)
        limits = torch.empty(
            (graph.num_envs, 31, 2), dtype=torch.float32, device=TEST_DEVICE
        )
        limits[:, :, 0] = -1.0
        limits[:, :, 1] = 1.0
        data = types.SimpleNamespace(
            joint_names=joint_names,
            soft_joint_pos_limits=limits.clone(),
            default_joint_pos=zeros.clone(),
            joint_pos_limits=limits,
            joint_pos=zeros.clone(),
        )
        action = types.SimpleNamespace(
            _joint_names=joint_names,
            _joint_ids=slice(None),
            _asset=types.SimpleNamespace(data=data),
            processed_actions=zeros.clone(),
            pre_clamp_qdes=zeros.clone(),
            nominal_projected_qdes=zeros.clone(),
            nominal_projection_span=torch.ones_like(zeros),
            pre_clamp_qdes_valid=torch.zeros(
                graph.num_envs, dtype=torch.bool, device=TEST_DEVICE
            ),
            nominal_projected_qdes_valid=torch.zeros(
                graph.num_envs, dtype=torch.bool, device=TEST_DEVICE
            ),
        )
        self.action_manager = types.SimpleNamespace(
            action=zeros.clone(),
            prev_action=zeros.clone(),
            get_term=lambda name: action
            if name == "joint_pos"
            else (_ for _ in ()).throw(KeyError(name)),
        )
        # The production seal resolves real IsaacLab evaluator modules once.
        # This focused CPU file deliberately stays dependency-light, so supply
        # inert callable surfaces only when the test's own monkeypatch did not
        # already provide the relevant module.
        current_import = R.importlib.import_module
        common_fallback = types.SimpleNamespace(
            **{
                spec.evaluator_name: (lambda *_args, **_kwargs: None)
                for spec in R.COMMON_DENSE_SPECS
            }
        )
        paddle_fallback = types.SimpleNamespace(
            _cmd=lambda *_args, **_kwargs: None,
            motion_racket_tracking_errors_now=lambda *_args, **_kwargs: None,
        )

        def focused_import(name):
            try:
                return current_import(name)
            except ModuleNotFoundError:
                if name.endswith(".rewards"):
                    return common_fallback
                if name.endswith(".hope_rewards"):
                    return paddle_fallback
                raise

        R.importlib.import_module = focused_import
        try:
            binding = R.seal_env_reward_hot_path(self, graph)
        finally:
            R.importlib.import_module = current_import
        self.__dict__[R.ENV_REWARD_HOT_PATH_ATTR] = (
            binding.bind_regularization(self)
        )

    def _action_ball_full_mdp_lean_reward_term(
        self,
        *,
        ordinal: int,
        scale: float | None = None,
        value=None,
        paddle_playback_active=None,
        paddle_error_components=None,
    ) -> torch.Tensor:
        if ordinal < R.LIFECYCLE_PAYMENT_COUNT:
            return self._test_reward_graph.pay(ordinal, scale=scale)
        return self._test_reward_graph.record_common_dense(
            ordinal,
            value,
            paddle_playback_active=paddle_playback_active,
            paddle_error_components=paddle_error_components,
        )


class _OldReferenceLeanActionEpochRewardGraph(R.LeanActionEpochRewardGraph):
    """The pre-cache R03 implementation retained only as a parity oracle."""

    def _r03(self, ordinal: int, scale: float) -> torch.Tensor:
        snapshot = self._snapshot()
        valid, source_step, fact, faults = snapshot.r03
        if fact.shape != (self.num_envs, E.OWNER_FACT_F32_WIDTH):
            raise R.LeanRewardCycleError("R03 epoch fact width differs")
        target_position = fact[:, 0:3]
        target_velocity = fact[:, 3:6]
        target_normal = fact[:, 6:9]
        ball_position = fact[:, 9:12]
        achieved_position = fact[:, 15:18]
        achieved_velocity = fact[:, 18:21]
        achieved_normal = fact[:, 21:24]
        consumer = R.R03_NAMES[ordinal]
        if consumer == "paddle_center_proximity":
            error = torch.linalg.vector_norm(
                achieved_position - ball_position, dim=-1
            )
        else:
            component = consumer.split("_", 1)[1].split("_", 1)[0]
            if component == "position":
                error = torch.linalg.vector_norm(
                    achieved_position - target_position, dim=-1
                )
            elif component == "velocity":
                error = torch.linalg.vector_norm(
                    achieved_velocity - target_velocity, dim=-1
                )
            else:
                cosine = torch.sum(
                    achieved_normal * target_normal, dim=-1
                ).clamp(-1.0, 1.0)
                error = torch.acos(cosine)
        finite = torch.isfinite(error)
        clean_error = torch.where(finite, error, torch.zeros_like(error))
        ratio_sq = torch.square(clean_error / scale)
        if consumer.endswith("_coarse") or consumer == "paddle_center_proximity":
            raw = torch.reciprocal(1.0 + ratio_sq)
        else:
            raw = torch.exp(-ratio_sq)
        present = torch.bitwise_and(valid, R.R03_PRESENT).ne(0) & faults.eq(0)
        admitted = (
            present
            & torch.bitwise_and(valid, R.R03_PHYSICALLY_VALID).ne(0)
            & snapshot.reward_cycle_fault.eq(0)
            & source_step.eq(snapshot.reward_cycle_age)
        )
        self._milestone.add_reward(
            ordinal,
            raw,
            raw,
            admitted,
            finite,
            self._milestone_configured_income_scale[ordinal],
        )
        return torch.where(admitted & finite, raw, torch.zeros_like(raw))


_R03_TEST_SCALES = (0.2, 1.1, 0.45, 0.55, 2.1, 0.9, 0.08, 0.4, 0.2, 0.13)


def _set_r03_parity_case(owner, case):
    publication = owner._publication
    record = publication.current
    slot = E.OWNER_ORDER.index("r03_strike_fact")
    facts = record.fact_f32.clone()
    valid_bits = record.fact_valid_bits.clone()
    source_step = record.fact_source_step.clone()
    faults = record.owner_fault_bits.clone()

    facts[0, 0, slot, 0:3] = _tensor((0.2, -0.1, 0.3))
    facts[0, 0, slot, 3:6] = _tensor((0.5, 0.1, -0.2))
    facts[0, 0, slot, 9:12] = _tensor((0.1, 0.2, -0.1))
    facts[0, 0, slot, 15:18] = _tensor((0.4, 0.2, 0.1))
    facts[0, 0, slot, 18:21] = _tensor((0.1, 0.3, -0.4))
    facts[0, 0, slot, 21:24] = _tensor((0.0, 0.6, 0.8))
    facts[1, 0, slot, 0:3] = _tensor((-0.3, 0.2, 0.1))
    facts[1, 0, slot, 3:6] = _tensor((-0.2, 0.4, 0.3))
    facts[1, 0, slot, 9:12] = _tensor((0.0, -0.2, 0.3))
    facts[1, 0, slot, 15:18] = _tensor((-0.1, -0.2, 0.5))
    facts[1, 0, slot, 18:21] = _tensor((0.3, 0.1, -0.1))
    facts[1, 0, slot, 21:24] = _tensor((0.0, -0.8, 0.6))

    if case == "invalid":
        valid_bits[0, 0, slot] = R.R03_PRESENT
        faults[1, 0, slot] = 32
    elif case == "nonfinite":
        facts[0, 0, slot, 15] = float("nan")
        facts[1, 0, slot, 18] = float("inf")
        facts[1, 0, slot, 21] = float("nan")
    elif case not in ("valid", "sticky"):
        raise AssertionError("unknown R03 parity case")

    owner._publication = E._Publication(
        replace(
            record,
            fact_valid_bits=valid_bits,
            fact_source_step=source_step,
            fact_f32=facts,
            owner_fault_bits=faults,
        ),
        publication.pending_log,
    )


def _configure_unique_income(graph):
    graph.configure_milestone_configured_income(
        {
            name: types.SimpleNamespace(weight=0.25 + ordinal * 0.125)
            for ordinal, name in enumerate(R.MANAGER_NAMES)
        },
        0.031,
    )


def _pay_r03(graph):
    return tuple(
        graph.pay(ordinal, scale=_R03_TEST_SCALES[ordinal])
        for ordinal in range(len(R.R03_NAMES))
    )


@pytest.mark.parametrize("case", ("valid", "invalid", "nonfinite", "sticky"))
def test_packed_r03_cache_matches_old_reference_for_all_fact_classes(case):
    old_owner = _epoch()
    new_owner = _epoch()
    _set_r03_parity_case(old_owner, case)
    _set_r03_parity_case(new_owner, case)
    old = _OldReferenceLeanActionEpochRewardGraph(epoch_owner=old_owner)
    new = R.LeanActionEpochRewardGraph(epoch_owner=new_owner)
    _configure_unique_income(old)
    _configure_unique_income(new)

    if case == "sticky":
        old_first = _pay_all(old)
        new_first = _pay_all(new)
        assert all(
            torch.equal(old_value, new_value)
            for old_value, new_value in zip(old_first, new_first)
        )
        old_owner.publish_reward_payment(8)
        new_owner.publish_reward_payment(8)

    old_values = _pay_r03(old)
    new_values = _pay_r03(new)

    assert all(
        torch.equal(old_value, new_value)
        for old_value, new_value in zip(old_values, new_values)
    )
    assert torch.equal(old_owner.milestone.i64, new_owner.milestone.i64)
    assert torch.equal(old_owner.milestone.f64, new_owner.milestone.f64)
    assert new._r03_cycle_cache is None


def test_packed_r03_owner_decode_and_shared_errors_run_once(monkeypatch):
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    counts = {"gather": 0, "vector_norm": 0, "acos": 0}
    gather = R.torch.gather
    vector_norm = R.torch.linalg.vector_norm
    acos = R.torch.acos

    def counted_gather(*args, **kwargs):
        counts["gather"] += 1
        return gather(*args, **kwargs)

    def counted_vector_norm(*args, **kwargs):
        counts["vector_norm"] += 1
        return vector_norm(*args, **kwargs)

    def counted_acos(*args, **kwargs):
        counts["acos"] += 1
        return acos(*args, **kwargs)

    monkeypatch.setattr(R.torch, "gather", counted_gather)
    monkeypatch.setattr(R.torch.linalg, "vector_norm", counted_vector_norm)
    monkeypatch.setattr(R.torch, "acos", counted_acos)

    values = _pay_r03(graph)

    assert all(torch.isfinite(value).all() for value in values)
    assert counts == {"gather": 17, "vector_norm": 3, "acos": 1}
    assert graph._r03_cycle_cache is None


def test_r03_cache_lifecycle_closes_and_reopens_with_a_new_cycle():
    owner = _epoch()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    actual = torch.zeros(graph.num_envs, dtype=torch.float32, device=TEST_DEVICE)

    value = graph.pay(0, scale=_R03_TEST_SCALES[0])
    actual.add_(value)
    first_snapshot = graph._cycle_snapshot
    assert first_snapshot is not None
    first_cache = graph._r03_cycle_cache
    assert first_cache is not None
    for ordinal in range(1, len(R.R03_NAMES)):
        value = graph.pay(ordinal, scale=_R03_TEST_SCALES[ordinal])
        actual.add_(value)
    assert graph._r03_cycle_cache is None
    for ordinal in range(10, R.LIFECYCLE_PAYMENT_COUNT):
        value = graph.pay(ordinal)
        actual.add_(value)
    assert graph._cycle_snapshot is None
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT, R.MANAGER_TERM_COUNT):
        dense_value = torch.ones(graph.num_envs, device=TEST_DEVICE)
        value = graph.record_common_dense(
            ordinal,
            dense_value,
            **_paddle_telemetry_kwargs(ordinal, dense_value),
        )
        actual.add_(value)
    graph.close_milestone_actual_reward(actual)
    assert graph._r03_cycle_cache is None
    owner.publish_reward_payment(8)

    graph.pay(0, scale=_R03_TEST_SCALES[0])

    second_snapshot = graph._cycle_snapshot
    assert second_snapshot is not None and second_snapshot is not first_snapshot
    assert torch.equal(
        second_snapshot.reward_cycle_age,
        first_snapshot.reward_cycle_age + 1,
    )
    assert (
        second_snapshot.reward_cycle_age.data_ptr()
        != first_snapshot.reward_cycle_age.data_ptr()
    )
    assert graph._r03_cycle_cache is not None
    assert graph._r03_cycle_cache is not first_cache


def test_r03_cache_invalidates_on_order_error_and_poison():
    order_error = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    order_error.pay(0, scale=_R03_TEST_SCALES[0])
    assert order_error._r03_cycle_cache is not None
    with pytest.raises(R.LeanRewardCycleError, match="expected 1"):
        order_error.pay(0, scale=_R03_TEST_SCALES[0])
    assert order_error._r03_cycle_cache is None
    assert order_error.poisoned is False

    poisoned = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    poisoned.pay(0, scale=_R03_TEST_SCALES[0])
    assert poisoned._r03_cycle_cache is not None
    with pytest.raises(R.LeanRewardConstructionHold, match="scale must"):
        poisoned.pay(1, scale=0.0)
    assert poisoned.poisoned is True
    assert poisoned._r03_cycle_cache is None


def test_r03_cache_hot_path_has_no_tensor_host_observation_api():
    source = "\n".join(
        inspect.getsource(member)
        for member in (
            R.LeanActionEpochRewardGraph._decode_r03_cycle,
            R.LeanActionEpochRewardGraph._r03,
            R.LeanActionEpochRewardGraph._invalidate_r03_cycle_cache,
        )
    )
    assert all(
        token not in source
        for token in (
            ".item(",
            ".cpu(",
            ".numpy(",
            ".tolist(",
            ".synchronize(",
            "torch.equal(",
        )
    )
    assert '.to(device="cpu"' not in source


def test_exact_reward28_cycle_completes_once(monkeypatch):
    owner = _epoch()
    before_commit = owner.commit_head
    before_version = owner._publication.current.version

    def reject_full_record_clone(_record):
        raise AssertionError("sealed Lean Reward cloned the full epoch record")

    monkeypatch.setattr(E.ActionEpochRecord, "clone", reject_full_record_clone)
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    values = _pay_all(graph)
    after = owner._publication.current
    assert owner.commit_head == before_commit + 15
    assert after.version == before_version + 15
    assert len(values) == R.MANAGER_TERM_COUNT == 28
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


def test_sticky_one_shots_do_not_repay_and_r07_requires_fresh_phase_tick():
    owner = _epoch()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)

    first = _pay_all(graph)
    assert owner.publish_reward_payment(8) is None
    payment = owner.project_current_reward_payment_rows()
    second = _pay_all(graph)
    record = owner.current()
    r07_slot = E.OWNER_ORDER.index("r07_recovery")
    owner.publish_owner_facts(
        "r07_recovery",
        owner=owner._fact_owner_identities["r07_recovery"],
        valid_bits=record.fact_valid_bits[:, :, r07_slot].clone(),
        source_step=torch.full(
            (2, 1), 9, dtype=torch.int64, device=TEST_DEVICE
        ),
        values=record.fact_f32[:, :, r07_slot].clone(),
    )
    third = _pay_all(graph)

    assert payment.valid.tolist() == [True, True]
    assert all(not bool(value.any()) for value in second[:14])
    assert all(not bool(value.any()) for value in third[:13])
    assert torch.equal(third[13], first[13])
    reward_i = owner.milestone.i64[
        : R.MANAGER_TERM_COUNT * 4
    ].reshape(R.MANAGER_TERM_COUNT, 4)
    assert reward_i[0, 1].item() == 2
    assert reward_i[10, 1].item() == 1
    assert reward_i[11, 1].item() == 2
    assert reward_i[12, 1].item() == 2
    assert reward_i[13, 1].item() == 4


def test_reward_payment_command_defers_the_only_projection_clone_to_r06_pull(
    monkeypatch,
):
    owner = _epoch()
    publication = owner._publication
    record = publication.current
    source_step = record.fact_source_step.clone()
    source_step[:, :, E.OWNER_ORDER.index("r06_landing_outcome")] = 3
    owner._publication = E._Publication(
        replace(
            record,
            fact_source_step=source_step,
            settlement_step=torch.full_like(record.settlement_step, 3),
        ),
        publication.pending_log,
    )

    values = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner))
    clone_calls = []
    clone_payment_rows = E.ActionEpochRewardPaymentRows.clone

    def counted_clone(value):
        clone_calls.append(value)
        return clone_payment_rows(value)

    monkeypatch.setattr(E.ActionEpochRewardPaymentRows, "clone", counted_clone)
    assert owner.publish_reward_payment(8) is None
    assert clone_calls == []
    payment = owner.project_current_reward_payment_rows()
    assert clone_calls == [owner._current_payment_rows]

    assert torch.equal(values[11], _tensor((1.0, 0.0)))
    assert torch.equal(values[12], torch.ones(2, device=TEST_DEVICE))
    assert payment.valid.tolist() == [True, True]
    assert payment.payment_step.tolist() == [8, 8]
    assert owner._undrained_row_fault_bits.tolist() == [0, 0]


def test_r06_rejects_missing_settlement_and_noncanonical_unpaid_sentinel():
    owner = _epoch()
    publication = owner._publication
    record = publication.current
    settlement_step = record.settlement_step.clone()
    payment_step = record.payment_step.clone()
    settlement_step[0, 0] = -1
    payment_step[1, 0] = -2
    owner._publication = E._Publication(
        replace(
            record,
            settlement_step=settlement_step,
            payment_step=payment_step,
        ),
        publication.pending_log,
    )

    values = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner))
    assert owner.publish_reward_payment(8) is None
    payment = owner.project_current_reward_payment_rows()

    assert not bool(values[11].any())
    assert not bool(values[12].any())
    assert payment.valid.tolist() == [False, False]
    assert payment.payment_step.tolist() == [-1, -1]
    assert owner.current().payment_step[:, 0].tolist() == [-1, -2]
    assert owner._undrained_row_fault_bits.tolist() == [
        E.ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY,
        E.ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY,
    ]


def test_reward_payment_before_local_settlement_faults_without_mutation():
    owner = _epoch()
    _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner))

    assert owner.publish_reward_payment(7) is None
    payment = owner.project_current_reward_payment_rows()

    assert payment.valid.tolist() == [False, False]
    assert payment.payment_step.tolist() == [-1, -1]
    assert owner.current().payment_step[:, 0].tolist() == [-1, -1]
    assert owner._undrained_row_fault_bits.tolist() == [
        E.ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY,
        E.ROW_FAULT_REWARD_PAYMENT_CHRONOLOGY,
    ]


def test_r06_ordinals_share_the_ordinal_zero_frozen_before_image():
    owner = _epoch()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    graph.pay(0, scale=1.0)

    record = owner.current()
    slot = E.OWNER_ORDER.index("r06_landing_outcome")
    live_facts = record.fact_f32[:, :, slot].clone()
    live_facts[:, :, 1] = 3.0
    live_facts[:, :, 2] = 4.0
    owner.publish_owner_facts(
        "r06_landing_outcome",
        owner=owner._fact_owner_identities["r06_landing_outcome"],
        valid_bits=record.fact_valid_bits[:, :, slot].clone(),
        source_step=record.fact_source_step[:, :, slot].clone(),
        values=live_facts,
    )

    for ordinal in range(1, 12):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    assert torch.equal(graph.pay(12), torch.ones(2, device=TEST_DEVICE))
    current = owner.current().fact_f32[:, :, slot]
    assert torch.equal(
        current[:, :, 1] * current[:, :, 2],
        torch.full((2, 1), 12.0, device=TEST_DEVICE),
    )


def test_r06_frozen_source_decodes_once_and_clears_after_second_row(monkeypatch):
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    calls = 0
    decode = graph._decode_r06_cycle

    def counted():
        nonlocal calls
        calls += 1
        return decode()

    monkeypatch.setattr(graph, "_decode_r06_cycle", counted)
    for ordinal in range(11):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    first = graph.pay(11)
    cache = graph._r06_cycle_cache
    second = graph.pay(12)

    assert calls == 1
    assert cache is not None
    assert graph._r06_cycle_cache is None
    assert torch.equal(first, _tensor((1.0, 0.0)))
    assert torch.equal(second, torch.ones(2, device=TEST_DEVICE))


def test_r06_pack_preserves_per_column_nonfinite_suppression_and_counts():
    owner = _epoch()
    publication = owner._publication
    record = publication.current
    slot = E.OWNER_ORDER.index("r06_landing_outcome")
    facts = record.fact_f32.clone()
    facts[0, 0, slot, 0] = float("nan")
    facts[1, 0, slot, 1] = float("inf")
    owner._publication = E._Publication(
        replace(record, fact_f32=facts), publication.pending_log
    )

    values = _pay_all(R.LeanActionEpochRewardGraph(epoch_owner=owner))
    rows = owner.milestone.i64[: R.MANAGER_TERM_COUNT * 4].reshape(
        R.MANAGER_TERM_COUNT, 4
    )

    assert torch.equal(values[11], torch.zeros(2, device=TEST_DEVICE))
    assert torch.equal(values[12], _tensor((1.0, 0.0)))
    assert rows[11].tolist() == [2, 2, 1, 0]
    assert rows[12].tolist() == [2, 2, 1, 1]


def test_reward_cycle_overflow_fault_suppresses_all_fresh_reward_families():
    owner = _epoch()
    limit = 2**63 - 1
    owner._reward_cycle_age.fill_(limit)
    publication = owner._publication
    record = publication.current
    source_step = record.fact_source_step.clone()
    source_step[:, :, E.OWNER_ORDER.index("r03_strike_fact")] = limit
    source_step[:, :, E.OWNER_ORDER.index("physical_ball")] = limit
    source_step[:, :, E.OWNER_ORDER.index("r07_recovery")] = limit - 1
    owner._publication = E._Publication(
        replace(
            record,
            reward_cycle_age=owner._reward_cycle_age,
            fact_source_step=source_step,
        ),
        publication.pending_log,
    )
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)

    first = _pay_all(graph)
    second = _pay_all(graph)

    assert all(not bool(value.any()) for value in first[:11])
    assert not bool(first[13].any())
    assert all(not bool(value.any()) for value in second[:11])
    assert not bool(second[13].any())
    assert owner.current().reward_cycle_fault.ne(0).all()


@pytest.mark.parametrize(
    "phase",
    (
        E.PHASE_REVEAL_COMMITTED,
        E.PHASE_LAUNCH_SETTLED,
        E.PHASE_RETIRED,
    ),
    ids=("reveal", "launch", "retired"),
)
def test_r07_non_outcome_row_cannot_pay_preloaded_fact_or_mask_peer(phase):
    owner = _epoch()
    publication = owner._publication
    record = publication.current
    row_phase = record.phase.clone()
    row_phase[0, 0] = phase
    before_peer = record.fact_f32[1, 0, E.OWNER_ORDER.index("r07_recovery")].clone()
    owner._publication = E._Publication(
        replace(record, phase=row_phase), publication.pending_log
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
    reward_i = owner.milestone.i64[
        : R.MANAGER_TERM_COUNT * 4
    ].reshape(R.MANAGER_TERM_COUNT, 4)
    reward_f = owner.milestone.f64[
        : R.MANAGER_TERM_COUNT * 7
    ].reshape(R.MANAGER_TERM_COUNT, 7)
    assert reward_i[0].tolist() == [2, 2, 2, 2]
    assert reward_f[0, 0].item() == 2.0
    assert reward_f[0, 3].item() == -1.0
    assert reward_f[0, 6].item() == 1.0
    assert reward_i[10].tolist() == [2, 1, 1, 1]
    assert reward_f[12, 0].item() == 1.0
    assert reward_f[12, 2].item() == 2.0
    assert reward_f[13, 0].item() == 0.0  # R07 raw score, not leaf-weighted payment.
    assert reward_f[13, 2].item() == -0.25


def test_actual_conservation_accepts_pinned_float32_two_multiply_order():
    owner = _epoch()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    weights = tuple(
        0.13 + ordinal * 0.017 for ordinal in range(R.MANAGER_TERM_COUNT)
    )
    dt = 0.031
    cfg = {
        name: types.SimpleNamespace(weight=weights[ordinal])
        for ordinal, name in enumerate(R.MANAGER_NAMES)
    }
    graph.configure_milestone_configured_income(cfg, dt)
    _pay_all(graph, manager_weights=weights, manager_dt=dt)
    milestone = owner.milestone
    violation_index = R.MANAGER_TERM_COUNT * 4 + 3
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
    reward_a = owner_a.milestone.f64[
        : R.MANAGER_TERM_COUNT * 7
    ].reshape(R.MANAGER_TERM_COUNT, 7)
    reward_c = owner_c.milestone.f64[
        : R.MANAGER_TERM_COUNT * 7
    ].reshape(R.MANAGER_TERM_COUNT, 7)
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
    with pytest.raises(R.LeanRewardCycleError, match="before all shared-contract"):
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


@pytest.mark.parametrize(("bad_row", "healthy_row"), ((0, 1), (1, 0)))
def test_producer_fault_suppresses_only_its_reward_row(bad_row, healthy_row):
    baseline_owner = _epoch()
    baseline_values = _pay_all(
        R.LeanActionEpochRewardGraph(epoch_owner=baseline_owner)
    )
    expected_r07 = baseline_values[13]
    baseline_reward_i = baseline_owner.milestone.i64[
        : R.MANAGER_TERM_COUNT * 4
    ].reshape(R.MANAGER_TERM_COUNT, 4)
    assert expected_r07.ne(0).all()
    assert expected_r07[0].ne(expected_r07[1])
    assert baseline_reward_i[13].tolist() == [2, 2, 2, 2]

    owner = _epoch()
    r07_slot = E.OWNER_ORDER.index("r07_recovery")
    before = owner.current()
    healthy_bits = before.fact_valid_bits[healthy_row, 0, r07_slot].clone()
    healthy_step = before.fact_source_step[healthy_row, 0, r07_slot].clone()
    healthy_fact = before.fact_f32[healthy_row, 0, r07_slot].clone()
    faults = torch.zeros((2, 1), dtype=torch.int64, device=TEST_DEVICE)
    faults[bad_row, 0] = 32
    owner.merge_runtime_owner_fault(
        "r07_recovery",
        faults,
        owner=owner._fact_owner_identities["r07_recovery"],
    )

    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    values = _pay_all(graph)
    expected_after_fault = expected_r07.clone()
    expected_after_fault[bad_row] = 0.0
    assert torch.equal(values[13], expected_after_fault)
    assert values[13].shape == (2,)
    assert values[13][healthy_row].eq(expected_r07[healthy_row])
    assert values[13][bad_row].eq(0.0)
    assert graph.completed_cycle_count == 1

    after = owner.current()
    assert torch.equal(
        after.fact_valid_bits[healthy_row, 0, r07_slot], healthy_bits
    )
    assert torch.equal(
        after.fact_source_step[healthy_row, 0, r07_slot], healthy_step
    )
    assert torch.equal(after.fact_f32[healthy_row, 0, r07_slot], healthy_fact)
    reward_i = owner.milestone.i64[
        : R.MANAGER_TERM_COUNT * 4
    ].reshape(R.MANAGER_TERM_COUNT, 4)
    assert reward_i[13, 0].eq(baseline_reward_i[13, 0])
    assert reward_i[13].tolist() == [2, 1, 1, 1]


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
    expected_body_names = ("torso", "left_elbow", "right_elbow")
    monkeypatch.setattr(
        R, "_a3_upper_except_held_wrist_body_names", lambda: expected_body_names
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
    dense_terms = tuple(cfg.values())[R.LIFECYCLE_PAYMENT_COUNT :]
    assert tuple(term.params["ordinal"] for term in dense_terms) == tuple(
        range(R.LIFECYCLE_PAYMENT_COUNT, R.MANAGER_TERM_COUNT)
    )
    assert R.MANAGER_TERM_COUNT == len(R.MANAGER_NAMES) == 28
    assert R.MANAGER_NAMES[-8:-4] == (
        "motion_racket_position",
        "motion_racket_velocity",
        "motion_racket_normal",
        "motion_racket_long_axis",
    )
    assert R.MANAGER_NAMES[-4:] == R.REGULARIZATION_NAMES
    assert tuple(cfg[name].weight for name in R.PADDLE_MOTION_PRIOR_NAMES) == (
        1.0,
        1.0,
        1.0,
        0.5,
    )
    assert all(
        cfg[name].params["contact_peak_scale"] == 4.0
        and cfg[name].params["contact_half_window_s"] == 0.12
        for name in R.PADDLE_MOTION_PRIOR_NAMES
    )
    assert tuple(cfg[name].weight for name in R.REGULARIZATION_NAMES) == (
        0.1,
        10.0,
        1.0,
        10.0,
    )
    body_names_by_term = {
        name: cfg[name].params.get("body_names")
        for name in R.COMMON_DENSE_NAMES
    }
    assert body_names_by_term == {
        "motion_global_anchor_pos": None,
        "motion_global_anchor_ori": None,
        "motion_body_pos": expected_body_names,
        "motion_body_ori": expected_body_names,
        "motion_body_lin_vel": expected_body_names,
        "motion_body_ang_vel": expected_body_names,
    }
    assert all(
        "body_names" not in cfg[name].params
        for name in R.PADDLE_MOTION_PRIOR_NAMES
    )


def test_manager_cfg_and_each_callable_pickle_by_exact_module_global(monkeypatch):
    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(RewardTermCfg=types.SimpleNamespace),
    )
    monkeypatch.setattr(
        R,
        "_a3_upper_except_held_wrist_body_names",
        lambda: ("torso", "right_elbow"),
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
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT, R.MANAGER_TERM_COUNT):
        dense_value = torch.ones(2, device=TEST_DEVICE)
        values.append(
            env._action_ball_full_mdp_lean_reward_term(
                ordinal=ordinal,
                value=dense_value,
                **_paddle_telemetry_kwargs(ordinal, dense_value),
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
    calls = []

    def evaluator(
        env, *, command_name, std, coarse_std=None, body_names=None
    ):
        assert command_name == "motion"
        calls.append(body_names)
        fine = torch.exp(-torch.square(env.reference - env.robot) / std**2)
        if coarse_std is None:
            return fine
        return 0.5 * (
            fine
            + torch.exp(
                -torch.square(env.reference - env.robot) / coarse_std**2
            )
        )

    evaluator_module = types.SimpleNamespace(
        **{spec.evaluator_name: evaluator for spec in R.COMMON_DENSE_SPECS}
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
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    values = []
    body_names = ("pelvis", "right_elbow")
    for ordinal, spec in enumerate(
        R.COMMON_DENSE_SPECS, start=R.LIFECYCLE_PAYMENT_COUNT
    ):
        values.append(
            R.common_dense_reward(
                env,
                ordinal=ordinal,
                command_name=spec.command_name,
                std=spec.std,
                coarse_std=spec.coarse_std,
                body_names=(body_names if spec.body_scope is not None else None),
            )
        )
    for ordinal in range(
        R.LIFECYCLE_PAYMENT_COUNT + len(R.COMMON_DENSE_SPECS),
        R.MANAGER_TERM_COUNT,
    ):
        dense_value = torch.ones(2, device=TEST_DEVICE)
        graph.record_common_dense(
            ordinal,
            dense_value,
            **_paddle_telemetry_kwargs(ordinal, dense_value),
        )
    assert graph.completed_cycle_count == 1
    assert all(torch.isfinite(value).all() for value in values)
    assert all(value[0] > value[1] for value in values)
    assert calls == [None, None, body_names, body_names, body_names, body_names]


@pytest.mark.parametrize(
    "reward_abi_mode", ("valid", "absent", "malformed", "noncontiguous_errors")
)
def test_paddle_motion_prior_dispatches_exact_specs_and_closes_cycle(
    monkeypatch, reward_abi_mode
):
    target_calls = 0
    playback_mask_calls = 0

    class Motion:
        def action_ball_full_mdp_playback_active_mask(self):
            nonlocal playback_mask_calls
            playback_mask_calls += 1
            if reward_abi_mode == "absent":
                raise RuntimeError("playback reward ABI unavailable")
            if reward_abi_mode == "malformed":
                return torch.ones((2, 1), dtype=torch.bool, device=TEST_DEVICE)
            return torch.tensor(
                [True, False], dtype=torch.bool, device=TEST_DEVICE
            )

    motion = Motion()
    command = types.SimpleNamespace(
        _motion=lambda: motion,
        time_to_strike=torch.tensor(
            [0.0, 0.0], dtype=torch.float32, device=TEST_DEVICE
        ),
    )
    exact_errors = torch.tensor(
        [[0.017, 0.731, 0.123, 0.456], [0.31, 2.7, 0.41, 0.93]],
        dtype=torch.float32,
        device=TEST_DEVICE,
    )

    def tracking_errors_now(_cmd):
        nonlocal target_calls
        target_calls += 1
        if reward_abi_mode == "noncontiguous_errors":
            return exact_errors.t().contiguous().t()
        return exact_errors

    evaluator_module = types.SimpleNamespace(
        _cmd=lambda _env, name: command
        if name == "racket_target"
        else (_ for _ in ()).throw(KeyError(name)),
        motion_racket_tracking_errors_now=tracking_errors_now,
    )
    real_import = R.importlib.import_module
    monkeypatch.setattr(
        R.importlib,
        "import_module",
        lambda name: evaluator_module
        if name.endswith(".hope_rewards")
        else real_import(name),
    )
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    env = _ExactEnvRewardDispatcherRepresentation(graph)
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    for ordinal in range(
        R.LIFECYCLE_PAYMENT_COUNT,
        R.LIFECYCLE_PAYMENT_COUNT + len(R.COMMON_DENSE_SPECS),
    ):
        graph.record_common_dense(
            ordinal, torch.ones(2, dtype=torch.float32, device=TEST_DEVICE)
        )
    first = R.LIFECYCLE_PAYMENT_COUNT + len(R.COMMON_DENSE_SPECS)
    values = []
    for ordinal, spec in enumerate(R.PADDLE_MOTION_PRIOR_SPECS, start=first):
        def call():
            return R.paddle_motion_prior_reward(
                env,
                ordinal=ordinal,
                command_name=spec.command_name,
                std=spec.std,
                coarse_std=spec.coarse_std,
                contact_peak_scale=spec.contact_peak_scale,
                contact_half_window_s=(
                    R.reward_contract.PADDLE_MOTION_PRIOR_CONTACT_HALF_WINDOW_S
                ),
            )
        if reward_abi_mode != "valid":
            with pytest.raises(
                (RuntimeError, R.LeanRewardCycleError),
                match="playback|paddle-error|poisoned|unavailable",
            ):
                call()
            assert graph.poisoned is True
            assert graph.completed_cycle_count == 0
            return
        values.append(call())
    assert graph._paddle_reward_cycle_cache is None
    for ordinal in range(R.REGULARIZATION_FIRST_ORDINAL, R.MANAGER_TERM_COUNT):
        graph.record_common_dense(
            ordinal, torch.zeros(2, dtype=torch.float32, device=TEST_DEVICE)
        )
    assert graph.completed_cycle_count == 1
    assert graph.poisoned is False
    assert all(torch.isfinite(value).all() for value in values)
    expected_kernels = torch.stack(
        tuple(
            R.paddle_prior.coarse_precision_kernel(
                exact_errors[:, column],
                precision_std=spec.std,
                coarse_std=spec.coarse_std,
            )
            for column, spec in enumerate(R.PADDLE_MOTION_PRIOR_SPECS)
        ),
        dim=1,
    )
    expected_values = expected_kernels.clone()
    expected_values[0].mul_(
        R.reward_contract.PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE
    )
    assert torch.equal(
        torch.stack(values, dim=1).view(torch.int32),
        expected_values.view(torch.int32),
    )
    assert playback_mask_calls == 1
    assert target_calls == 1
    owner_telemetry = graph.epoch_owner.milestone
    # Batched milestone reductions are committed only when RewardManager's
    # exact actual reward closes the cycle.  Use the already accumulated
    # configured value so this focused test exercises the production boundary.
    graph.close_milestone_actual_reward(
        owner_telemetry.open_step_configured_income.to(torch.float32)
    )
    payload = R.epoch_v1.milestone_tensors.decode_host_window(
        *(
            value.detach().cpu().contiguous()
            for value in owner_telemetry.pack_views()
        )
    ).as_json(R.MANAGER_NAMES)
    playback_rows = payload["paddle_motion_prior_playback"]["terms"]
    expected_unavailable = [0] * 4
    expected_playback = [1] * 4
    assert [
        row["telemetry_unavailable_count"] for row in playback_rows
    ] == expected_unavailable
    assert [row["playback_count"] for row in playback_rows] == expected_playback
    assert [row["finite_count"] for row in playback_rows] == expected_playback
    assert [row["error_finite_count"] for row in playback_rows] == expected_playback
    assert [row["domain_violation_count"] for row in playback_rows] == [0] * 4
    for column, row in enumerate(playback_rows):
        expected_sum = float(expected_kernels[0, column])
        assert row["kernel_sum"] == pytest.approx(expected_sum)
    expected_error = exact_errors[0].tolist()
    for row, error in zip(playback_rows, expected_error):
        assert row["error_sum"] == pytest.approx(
            error
        )
        assert row["error_sum_sq"] == pytest.approx(
            error * error
        )


def test_shared_reward_contract_keeps_only_supplied_upper_scope_without_held_wrist():
    contract = R.reward_contract
    upper = (
        "torso",
        contract.HELD_RACKET_WRIST_BODY_NAME,
        "right_elbow",
    )
    assert contract.upper_except_held_wrist_body_names(upper) == (
        "torso",
        "right_elbow",
    )
    with pytest.raises(ValueError, match="held racket wrist"):
        contract.upper_except_held_wrist_body_names(("torso", "right_elbow"))
    with pytest.raises(ValueError, match="held racket wrist"):
        contract.upper_except_held_wrist_body_names(
            ("torso", contract.HELD_RACKET_WRIST_BODY_NAME, "torso")
        )


def test_regularization_wrapper_uses_sealed_unbound_dispatcher():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    env = _ExactEnvRewardDispatcherRepresentation(graph)
    env.num_envs = graph.num_envs
    env.action_manager = types.SimpleNamespace(
        get_term=lambda _name: (_ for _ in ()).throw(
            AssertionError("regularization binding was repeated")
        )
    )
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT, R.REGULARIZATION_FIRST_ORDINAL):
        value = torch.ones(graph.num_envs, device=TEST_DEVICE)
        graph.record_common_dense(
            ordinal, value, **_paddle_telemetry_kwargs(ordinal, value)
        )

    actual = R.regularization_reward(
        env, ordinal=R.REGULARIZATION_FIRST_ORDINAL
    )
    assert torch.equal(actual, torch.zeros(graph.num_envs, device=TEST_DEVICE))


def test_regularization_seal_is_one_shot_and_unbound_consumer_fails_loud():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    env = _ExactEnvRewardDispatcherRepresentation(graph)
    binding = R._env_reward_hot_path(env)

    with pytest.raises(R.LeanRewardConstructionHold, match="already bound"):
        binding.bind_regularization(env)

    env.__dict__[R.ENV_REWARD_HOT_PATH_ATTR] = R.replace(
        binding, regularization=None
    )
    with pytest.raises(R.LeanRewardConstructionHold, match="not bound"):
        R.regularization_reward(
            env, ordinal=R.REGULARIZATION_FIRST_ORDINAL
        )


def test_regularization_binding_preserves_reset_rows_and_ordinal_dispatch(monkeypatch):
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    env = _ExactEnvRewardDispatcherRepresentation(graph)
    env.num_envs = graph.num_envs
    binding = R._env_reward_hot_path(env).regularization
    assert binding is not None

    # Reset semantics: action rate is zero, both barrier positions are neutral,
    # and stale projection buffers are ignored while their valid bits are false.
    binding.action.pre_clamp_qdes.fill_(float("nan"))
    binding.action.nominal_projected_qdes.fill_(float("nan"))
    binding.action.nominal_projection_span.fill_(float("nan"))
    binding.action.pre_clamp_qdes_valid.zero_()
    binding.action.nominal_projected_qdes_valid.zero_()

    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    for ordinal in range(
        R.LIFECYCLE_PAYMENT_COUNT, R.REGULARIZATION_FIRST_ORDINAL
    ):
        value = torch.ones(graph.num_envs, device=TEST_DEVICE)
        graph.record_common_dense(
            ordinal, value, **_paddle_telemetry_kwargs(ordinal, value)
        )

    original_barrier = R.regularization._soft_limit_barrier_v2_prepared
    original_projection = R.regularization.qdes_projection_penalty
    monkeypatch.setattr(
        R.regularization,
        "_soft_limit_barrier_v2_prepared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("later regularization row ran early")
        ),
    )
    monkeypatch.setattr(
        R.regularization,
        "qdes_projection_penalty",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("projection row ran early")
        ),
    )
    first = R.regularization_reward(
        env, ordinal=R.REGULARIZATION_FIRST_ORDINAL
    )
    assert torch.equal(first, torch.zeros_like(first))
    monkeypatch.setattr(
        R.regularization, "_soft_limit_barrier_v2_prepared", original_barrier
    )
    monkeypatch.setattr(
        R.regularization, "qdes_projection_penalty", original_projection
    )

    remaining = tuple(
        R.regularization_reward(env, ordinal=ordinal)
        for ordinal in range(R.REGULARIZATION_FIRST_ORDINAL + 1, R.MANAGER_TERM_COUNT)
    )
    assert all(torch.equal(value, torch.zeros_like(value)) for value in remaining)
    assert graph.completed_cycle_count == 1


def test_dense_rows_ignore_lifecycle_paid_bits_and_exist_when_lifecycle_is_zero():
    owner = _epoch()
    owner._publication.current.fact_valid_bits.zero_()
    graph = R.LeanActionEpochRewardGraph(epoch_owner=owner)
    lifecycle = tuple(
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
        for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT)
    )
    assert all(not bool(value.any()) for value in lifecycle)
    owner._publication.current.reward_paid.logical_not_()
    dense = []
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT, R.MANAGER_TERM_COUNT):
        dense_value = torch.ones(2, device=TEST_DEVICE)
        dense.append(
            graph.record_common_dense(
                ordinal,
                dense_value,
                **_paddle_telemetry_kwargs(ordinal, dense_value),
            )
        )
    dense = tuple(dense)
    assert all(torch.equal(value, torch.ones(2, device=TEST_DEVICE)) for value in dense)
    assert graph.completed_cycle_count == 1


@pytest.mark.parametrize("bad_ordinal", (15, 19))
def test_dense_skip_or_reorder_fails_before_actual_close(bad_ordinal):
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    with pytest.raises(R.LeanRewardCycleError, match="expected 14"):
        graph.record_common_dense(
            bad_ordinal, torch.ones(2, device=TEST_DEVICE)
        )
    assert graph.completed_cycle_count == 0
    assert graph.poisoned is True


def test_dense_duplicate_fails_before_actual_close():
    graph = R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    for ordinal in range(R.LIFECYCLE_PAYMENT_COUNT):
        graph.pay(ordinal, scale=1.0 if ordinal < 10 else None)
    graph.record_common_dense(14, torch.ones(2, device=TEST_DEVICE))
    with pytest.raises(R.LeanRewardCycleError, match="expected 15"):
        graph.record_common_dense(14, torch.ones(2, device=TEST_DEVICE))
    assert graph.completed_cycle_count == 0
    assert graph.poisoned is True


def test_sealed_reward_dispatcher_ignores_later_instance_shadow():
    env = _ExactEnvRewardDispatcherRepresentation(
        R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
    )
    env.__dict__[R.ENV_REWARD_DISPATCHER_NAME] = lambda **_kwargs: None
    value = R.racket_position(env, scale=1.0)
    assert torch.equal(value, torch.ones(2, device=TEST_DEVICE))


@pytest.mark.parametrize("descriptor", (None, staticmethod(lambda **_kwargs: None)))
def test_reward_hot_path_seal_requires_own_plain_dispatcher(descriptor):
    class InvalidEnv:
        pass

    if descriptor is not None:
        InvalidEnv._action_ball_full_mdp_lean_reward_term = descriptor
    with pytest.raises(R.LeanRewardConstructionHold, match="class-owned"):
        R.seal_env_reward_hot_path(
            InvalidEnv(), R.LeanActionEpochRewardGraph(epoch_owner=_epoch())
        )


def test_diagnostic_bundle_has_no_caller_numeric_seam(monkeypatch):
    owner = _epoch()

    class RewardTermCfg:
        def __init__(self, *, func, weight, params):
            self.func, self.weight, self.params = func, weight, params

    tracked = (
        "pelvis_link",
        R.reward_contract.HELD_RACKET_WRIST_BODY_NAME,
        "right_elbow_Link",
    )

    def import_focused_dependency(name):
        if name == "isaaclab.managers":
            return types.SimpleNamespace(RewardTermCfg=RewardTermCfg)
        if name == "whole_body_tracking.robots.agibot_a3":
            return types.SimpleNamespace(
                A3_TRACKED_BODIES=tracked,
                A3_UPPER_TRACKED=tracked[1:],
            )
        raise AssertionError("unexpected diagnostic bundle import: " + name)

    monkeypatch.setattr(R.importlib, "import_module", import_focused_dependency)
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
