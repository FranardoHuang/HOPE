"""CPU parity and hot-path guards for the engine-neutral Reward14 kernel."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
import textwrap

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "action_ball_full_mdp_portable_reward.py"
)
SPEC = importlib.util.spec_from_file_location("action_ball_full_mdp_portable_reward_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)


def _fixed_tape(batch: int, dtype: torch.dtype, case: str):
    row = torch.arange(batch, dtype=dtype)
    valid = torch.zeros((batch, reward.OWNER_COUNT), dtype=torch.int64)
    faults = torch.zeros_like(valid)
    facts = torch.zeros(
        (batch, reward.OWNER_COUNT, reward.OWNER_FACT_F32_WIDTH),
        dtype=dtype,
    )

    valid[:, 1] = reward.R03_PRESENT | reward.R03_PHYSICALLY_VALID

    r03 = facts[:, 1]
    r03[:, 0:3] = torch.stack((row * 0.01, row * -0.02, 0.8 + row * 0.001), dim=1)
    r03[:, 3:6] = torch.stack((1.0 + row * 0.003, row * -0.004, row * 0.002), dim=1)
    r03[:, 6:9] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    r03[:, 9:12] = r03[:, 0:3] + torch.tensor([0.02, -0.03, 0.04], dtype=dtype)
    r03[:, 15:18] = r03[:, 0:3] + torch.stack(
        ((row.remainder(7) - 3.0) * 0.03, row.remainder(5) * 0.02, row.remainder(3) * -0.01),
        dim=1,
    )
    r03[:, 18:21] = r03[:, 3:6] + torch.stack(
        (row.remainder(11) * -0.04, row.remainder(13) * 0.03, row.remainder(17) * 0.02),
        dim=1,
    )
    normal_y = 1.0 - row.remainder(9) * 0.2
    r03[:, 21:24] = torch.stack((row.remainder(4) * 0.1, normal_y, row.remainder(6) * -0.1), dim=1)

    if batch > 1:
        valid[1::11, 1] = reward.R03_PRESENT
        faults[2::13, 1] = 1

    if case == "degenerate":
        r03[:, 15:18] = r03[:, 0:3]
        r03[:, 18:21] = r03[:, 3:6]
        r03[:, 9:12] = r03[:, 15:18]
        r03[:, 6:9].zero_()
        r03[:, 21:24].zero_()
        r03[::3, 6:9] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        r03[::3, 21:24] = torch.tensor([0.0, -1.0, 0.0], dtype=dtype)
    elif case == "poison":
        r03[::5, 15] = float("nan")
        r03[1::5, 18] = float("inf")
        r03[2::5, 21] = float("-inf")
        r03[3::5, 9] = float("nan")
    elif case != "ordinary":
        raise AssertionError(f"unknown tape case {case!r}")
    return valid, faults, facts


def _old_reference(
    *,
    valid_bits: torch.Tensor,
    fact_f32: torch.Tensor,
    owner_fault_bits: torch.Tensor,
    step_dt: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Pre-deduplication loop, driven by the production ABI/spec names."""

    batch = int(valid_bits.shape[0])
    raw = torch.zeros(
        (batch, reward.LIFECYCLE_TERM_COUNT),
        dtype=fact_f32.dtype,
        device=fact_f32.device,
    )
    r03_bits = valid_bits[:, 1]
    r03 = fact_f32[:, 1]
    r03_admitted = (
        torch.bitwise_and(r03_bits, reward.R03_PRESENT).ne(0)
        & torch.bitwise_and(r03_bits, reward.R03_PHYSICALLY_VALID).ne(0)
        & owner_fault_bits[:, 1].eq(0)
    )
    target_position, target_velocity, target_normal = r03[:, 0:3], r03[:, 3:6], r03[:, 6:9]
    ball_position = r03[:, 9:12]
    achieved_position, achieved_velocity, achieved_normal = r03[:, 15:18], r03[:, 18:21], r03[:, 21:24]
    for ordinal, (name, scale, reciprocal) in enumerate(reward.R03_REWARD_SPECS):
        if name == "paddle_center_proximity":
            error = torch.linalg.vector_norm(achieved_position - ball_position, dim=-1)
        elif "position" in name:
            error = torch.linalg.vector_norm(achieved_position - target_position, dim=-1)
        elif "velocity" in name:
            error = torch.linalg.vector_norm(achieved_velocity - target_velocity, dim=-1)
        else:
            cosine = torch.sum(achieved_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
            error = torch.acos(cosine)
        finite = torch.isfinite(error)
        clean = torch.where(finite, error, torch.zeros_like(error))
        ratio_sq = torch.square(clean / scale)
        value = torch.reciprocal(1.0 + ratio_sq) if reciprocal else torch.exp(-ratio_sq)
        raw[:, ordinal] = torch.where(r03_admitted & finite, value, torch.zeros_like(value))

    return raw * weights * float(step_dt)


@pytest.mark.parametrize("batch", (1, 2, 64, 4096))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("case", ("ordinary", "degenerate", "poison"))
def test_reward14_unique_error_reuse_is_bitwise_old_reference(batch, dtype, case):
    valid, faults, facts = _fixed_tape(batch, dtype, case)
    valid_before = valid.clone()
    faults_before = faults.clone()
    facts_before = facts.clone()
    weights = torch.tensor(reward.LIFECYCLE_WEIGHTS, dtype=dtype)

    expected = _old_reference(
        valid_bits=valid,
        fact_f32=facts,
        owner_fault_bits=faults,
        step_dt=0.02,
        weights=weights,
    )
    actual = reward.lifecycle_reward14(
        valid_bits=valid,
        fact_f32=facts,
        owner_fault_bits=faults,
        step_dt=0.02,
        weights=weights,
    )

    assert actual.dtype == dtype and actual.device == facts.device
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.equal(valid, valid_before)
    assert torch.equal(faults, faults_before)
    assert torch.equal(facts.view(torch.uint8), facts_before.view(torch.uint8))


def test_reward14_executes_only_four_semantic_error_reductions_without_host_reads(monkeypatch):
    valid, faults, facts = _fixed_tape(64, torch.float64, "poison")
    weights = torch.tensor(reward.LIFECYCLE_WEIGHTS, dtype=facts.dtype)
    calls = {"vector_norm": 0, "sum": 0, "acos": 0}
    original_vector_norm = torch.linalg.vector_norm
    original_sum = torch.sum
    original_acos = torch.acos

    def counted_vector_norm(*args, **kwargs):
        calls["vector_norm"] += 1
        return original_vector_norm(*args, **kwargs)

    def counted_sum(*args, **kwargs):
        calls["sum"] += 1
        return original_sum(*args, **kwargs)

    def counted_acos(*args, **kwargs):
        calls["acos"] += 1
        return original_acos(*args, **kwargs)

    def forbidden_host_read(*_args, **_kwargs):
        raise AssertionError("Reward14 cached hot path performed a host read")

    with monkeypatch.context() as patch:
        patch.setattr(reward.torch.linalg, "vector_norm", counted_vector_norm)
        patch.setattr(reward.torch, "sum", counted_sum)
        patch.setattr(reward.torch, "acos", counted_acos)
        patch.setattr(torch.Tensor, "__bool__", forbidden_host_read)
        patch.setattr(torch.Tensor, "item", forbidden_host_read)
        patch.setattr(torch.Tensor, "cpu", forbidden_host_read)
        actual = reward.lifecycle_reward14(
            valid_bits=valid,
            fact_f32=facts,
            owner_fault_bits=faults,
            step_dt=0.02,
            weights=weights,
        )

    assert calls == {"vector_norm": 3, "sum": 1, "acos": 1}
    assert actual.shape == (64, reward.LIFECYCLE_TERM_COUNT)


def test_reward14_r03_reductions_stay_outside_the_reward_spec_loop():
    tree = ast.parse(textwrap.dedent(inspect.getsource(reward.lifecycle_reward14)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    spec_loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
    )
    forbidden_calls = {"vector_norm", "norm", "sum", "acos"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_calls
        for node in ast.walk(spec_loop)
    )
