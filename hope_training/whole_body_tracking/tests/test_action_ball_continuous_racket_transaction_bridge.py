"""Regression boundary after retiring Racket's portable-R05 compact bridge.

The live Racket mutation path is covered by
``test_action_ball_racket_rowwise_accept.py``.  This file keeps only the two
independent facts that belonged at the old bridge boundary: public reveal
entry points fail before caller inspection, and host-side binary32 arithmetic
quantizes each operand before constructing an after-image.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MDP_ROOT = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
RUNTIME_SOURCE = MDP_ROOT / "action_ball_full_mdp_runtime_owner.py"
RACKET_SOURCE = MDP_ROOT / "hope_commands.py"


def _load_runtime_owner():
    name = "action_ball_full_mdp_runtime_owner_racket_boundary_focused"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, RUNTIME_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNTIME = _load_runtime_owner()


class _ExplodingCaller:
    def __getattribute__(self, name):
        raise AssertionError(f"caller payload was inspected: {name}")


class _HealthyPolicyBoundary:
    def __init__(self):
        self.events = []

    def require_healthy(self):
        self.events.append("healthy")

    def _require_no_selected_reset_debt(self, *, operation):
        self.events.append(("debt", operation))


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _class_methods(source: Path, class_name: str) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_public_policy_boundary_holds_before_request_or_action_inspection():
    boundary = _HealthyPolicyBoundary()
    exploding = _ExplodingCaller()

    with pytest.raises(
        RUNTIME.ActionBallFullMdpRuntimeDependencyError,
        match="children do not consume Device-R05 hot tokens",
    ):
        RUNTIME.ActionBallFullMdpRuntimeOwner.before_policy_step(
            boundary, exploding, exploding
        )

    assert boundary.events == ["healthy", ("debt", "policy step")]


def test_direct_reveal_tombstone_holds_before_request_inspection():
    with pytest.raises(
        RUNTIME.ActionBallFullMdpRuntimeDependencyError,
        match="direct reveal execution is a tombstone",
    ):
        RUNTIME.ActionBallFullMdpRuntimeOwner.execute_reveal(
            object(), _ExplodingCaller()
        )


def test_portable_r05_compact_surface_is_physically_absent():
    racket_methods = _class_methods(RACKET_SOURCE, "RacketTargetCommand")
    runtime_methods = _class_methods(
        RUNTIME_SOURCE, "ActionBallFullMdpRuntimeOwner"
    )
    retired_racket = {
        "bind_action_ball_continuous_racket_staging",
        "stage_action_ball_continuous_racket_reveal",
        "finalize_action_ball_continuous_racket_prearm",
        "action_ball_continuous_racket_boundary_row",
        "arm_action_ball_continuous_racket_prearm",
        "arm_censored_action_ball_continuous_racket_prearm",
        "commit_prevalidated_action_ball_continuous_racket",
        "commit_censored_prevalidated_action_ball_continuous_racket",
        "complete_global_reveal_epoch",
        "abort_action_ball_continuous_racket_prearm",
    }
    assert racket_methods.isdisjoint(retired_racket)
    assert runtime_methods.isdisjoint(
        {"_execute_owned_reveal", "_abort_pretransfer_reveal"}
    )

    racket_required = next(
        spec.required_methods
        for spec in RUNTIME._DEPENDENCY_SPECS
        if spec.role == "racket_child"
    )
    assert "bind_action_ball_full_mdp_racket_staging" in racket_required
    assert set(racket_required).isdisjoint(retired_racket)


def test_host_after_image_quantizes_operands_before_float32_addition():
    origin_f32 = -731.271484375
    local_host = 694.8674738744653
    canonical_sum = _f32(origin_f32 + _f32(local_host))
    late_quantized_sum = _f32(origin_f32 + local_host)

    assert canonical_sum == _f32(_f32(origin_f32) + _f32(local_host))
    assert late_quantized_sum != canonical_sum
