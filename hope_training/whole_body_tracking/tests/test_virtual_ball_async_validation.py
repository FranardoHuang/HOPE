"""Simulator-free regression tests for opt-in asynchronous VirtualBall validation."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
HERE = Path(__file__).resolve().parent
MDP = (
    HERE.parent
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)


def _load_virtual_ball():
    name = "wbt_virtual_ball_async_validation_test"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(
        name, MDP / "virtual_ball.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _valid_contact_inputs():
    return {
        "exact_strike": torch.tensor([True, True, False]),
        "signed_face_ok": torch.tensor([True, False, True]),
        "geometry_contact": torch.tensor([True, True, True]),
        "contact_finite": torch.tensor([True, True, True]),
        "normal_speed_mps": torch.tensor([2.0, 2.0, 2.0]),
    }


def _valid_rollout_inputs():
    zeros = torch.zeros(2, 3)
    origin = torch.tensor([[0.5, 0.0, 0.9], [0.6, 0.1, 1.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return {
        "capture": torch.tensor([True, False]),
        "contact_origin_w_m": origin,
        "ball_velocity_w_mps": zeros.clone(),
        "contact_point_velocity_w_mps": zeros.clone(),
        "physical_face_normal_w": normal,
        "ball_spin_w_radps": zeros.clone(),
        "fallback_origin_w_m": torch.full((2, 3), 0.25),
    }


def test_cpu_async_opt_in_preserves_contact_classification():
    """The opt-in changes only CUDA synchronization, never the partition result."""

    vb = _load_virtual_ball()
    sync_capture, sync_reasons = vb.classify_action_ball_contact(
        **_valid_contact_inputs()
    )
    async_capture, async_reasons = vb.classify_action_ball_contact(
        **_valid_contact_inputs(), async_validate=True
    )
    assert torch.equal(async_capture, sync_capture)
    assert tuple(async_reasons) == tuple(sync_reasons)
    for name in sync_reasons:
        assert torch.equal(async_reasons[name], sync_reasons[name])


@pytest.mark.parametrize("async_validate", [False, True])
def test_cpu_captured_nonfinite_still_fails_with_precise_error(
    async_validate,
):
    """CPU remains an immediate negative-control oracle in both modes."""

    vb = _load_virtual_ball()
    inputs = _valid_rollout_inputs()
    inputs["contact_origin_w_m"][0, 0] = float("nan")
    with pytest.raises(
        RuntimeError,
        match="captured contact contains a non-finite rollout input",
    ):
        vb.finite_action_ball_rollout_inputs(
            **inputs, async_validate=async_validate
        )


@pytest.mark.parametrize("async_validate", [False, True])
def test_cpu_nonfinite_fallback_still_fails_with_precise_error(
    async_validate,
):
    vb = _load_virtual_ball()
    inputs = _valid_rollout_inputs()
    inputs["fallback_origin_w_m"][0, 0] = float("inf")
    with pytest.raises(
        RuntimeError,
        match="fallback rollout origin is non-finite",
    ):
        vb.finite_action_ball_rollout_inputs(
            **inputs, async_validate=async_validate
        )


def test_cpu_nonfinite_error_priority_keeps_captured_contact_first():
    """The earlier, more specific captured-row failure remains the CPU diagnostic."""

    vb = _load_virtual_ball()
    inputs = _valid_rollout_inputs()
    inputs["contact_origin_w_m"][0, 0] = float("nan")
    inputs["fallback_origin_w_m"][0, 0] = float("nan")
    with pytest.raises(
        RuntimeError,
        match="captured contact contains a non-finite rollout input",
    ):
        vb.finite_action_ball_rollout_inputs(
            **inputs, async_validate=True
        )


@pytest.mark.parametrize("async_validate", [False, True])
def test_cpu_validation_primitive_preserves_message_and_priority(
    async_validate,
):
    """The primitive used by otherwise-unreachable partition guards remains testable."""

    vb = _load_virtual_ball()
    with pytest.raises(RuntimeError, match="first partition failure"):
        vb._assert_tensor_validation(
            torch.tensor(False),
            "first partition failure",
            async_validate=async_validate,
        )


def test_cuda_opt_in_source_has_no_validation_only_host_sync():
    """Source guard for the hot CUDA path; CUDA behavior is exercised on the Pod."""

    vb = _load_virtual_ball()
    classifier = inspect.getsource(vb.classify_action_ball_contact)
    sanitizer = inspect.getsource(vb.finite_action_ball_rollout_inputs)
    primitive = inspect.getsource(vb._assert_tensor_validation)

    assert "async_validate: bool = False" in classifier
    assert "async_validate: bool = False" in sanitizer
    assert "bool(" not in classifier
    assert "torch.equal(" not in classifier
    assert "bool(" not in sanitizer
    assert "torch.equal(" not in sanitizer
    assert "async_validate and scalar.device.type == \"cuda\"" in primitive
    async_start = primitive.index(
        'async_validate and scalar.device.type == "cuda"'
    )
    async_return = primitive.index("return", async_start)
    sync_bool = primitive.index("bool(scalar)")
    assert async_start < async_return < sync_bool
    assert "assert_fn(scalar)" in primitive[async_start:async_return]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="asynchronous validation needs CUDA",
)
def test_cuda_async_validation_accepts_healthy_contact_and_rollout():
    """Exercise the supported Pod torch API on a healthy CUDA stream."""

    vb = _load_virtual_ball()
    contact = {
        name: value.cuda()
        for name, value in _valid_contact_inputs().items()
    }
    capture, _reasons = vb.classify_action_ball_contact(
        **contact, async_validate=True
    )
    rollout = {
        name: value.cuda()
        for name, value in _valid_rollout_inputs().items()
    }
    safe = vb.finite_action_ball_rollout_inputs(
        **rollout, async_validate=True
    )
    torch.cuda.synchronize()
    assert capture.device.type == "cuda"
    assert all(value.device.type == "cuda" for value in safe.values())
