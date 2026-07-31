"""Diagnostic-only receipt for IsaacLab ``push_by_setting_velocity`` (host-only).

The production event must remain IsaacLab-authored: disabled paths preserve its
physical writer, RNG and result semantics through one delegate call, while the
unauthorized ActionBall probe only reads root velocity around that call and
books device counters.  No Isaac installation is needed here; the HOPE event
module imports Isaac lazily.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import types
from pathlib import Path

import pytest
import torch


HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EVENT_PATH = (
    HERE
    / "../source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_push_events.py"
).resolve()
SPEC = importlib.util.spec_from_file_location(
    "push_velocity_diagnostic_under_test", EVENT_PATH
)
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)

RUNNER_PATH = (
    HERE
    / "../source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
).resolve()
MDP_INIT_PATH = (
    HERE
    / "../source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/__init__.py"
).resolve()

RANGE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.10, 0.10),
    "roll": (-0.26, 0.26),
    "pitch": (-0.26, 0.26),
    "yaw": (-0.39, 0.39),
}


class _Asset:
    def __init__(self, num_envs: int):
        self.data = types.SimpleNamespace(root_vel_w=torch.zeros((num_envs, 6)))


class _Env:
    def __init__(self, *, num_envs=4, diagnostic=True):
        self.num_envs = num_envs
        self.scene = {"robot": _Asset(num_envs)}
        self.cfg = types.SimpleNamespace(
            commands=types.SimpleNamespace(
                racket_target=types.SimpleNamespace(
                    target_mode="action_ball",
                    action_ball_diagnostic_unauthorized=diagnostic,
                )
            )
        )


def _delegate_with_delta(calls, delta):
    delta = torch.as_tensor(delta, dtype=torch.float32)

    def delegate(env, env_ids, *, velocity_range, asset_cfg=None):
        calls.append((env_ids, velocity_range, asset_cfg))
        ids = (
            torch.arange(env.num_envs)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long)
        )
        name = "robot" if asset_cfg is None else asset_cfg.name
        env.scene[name].data.root_vel_w[ids] += delta
        return "delegate-result"

    return delegate


def test_disabled_is_single_delegate_without_receipt_side_effects(monkeypatch):
    calls = []
    env = _Env(diagnostic=False)
    sentinel_scene = env.scene
    monkeypatch.setattr(
        EV,
        "_velocity_push_delegate",
        lambda: _delegate_with_delta(calls, [0.1] * 6),
    )
    result = EV.push_by_setting_velocity(
        env, torch.tensor([1]), velocity_range=RANGE
    )
    assert result == "delegate-result"
    assert len(calls) == 1
    assert calls[0][0].tolist() == [1]
    assert calls[0][1] is RANGE and calls[0][2] is None
    assert env.scene is sentinel_scene
    assert not hasattr(env, EV.PUSH_VELOCITY_DIAGNOSTIC_STATE_ATTR)


def test_explicit_asset_cfg_is_forwarded_once_and_observed_on_that_asset(monkeypatch):
    calls = []
    env = _Env(num_envs=3)
    env.scene["alternate_robot"] = _Asset(3)
    asset_cfg = types.SimpleNamespace(name="alternate_robot")
    delta = [0.1, -0.1, 0.05, 0.1, -0.1, 0.2]
    monkeypatch.setattr(
        EV, "_velocity_push_delegate", lambda: _delegate_with_delta(calls, delta)
    )

    result = EV.push_by_setting_velocity(
        env,
        torch.tensor([0, 2]),
        velocity_range=RANGE,
        asset_cfg=asset_cfg,
    )

    assert result == "delegate-result"
    assert len(calls) == 1
    assert calls[0][0].tolist() == [0, 2]
    assert calls[0][1] is RANGE
    assert calls[0][2] is asset_cfg
    assert torch.count_nonzero(env.scene["robot"].data.root_vel_w) == 0
    expected = torch.tensor(delta).expand(2, -1)
    assert torch.equal(
        env.scene["alternate_robot"].data.root_vel_w[torch.tensor([0, 2])],
        expected,
    )
    receipt = EV.consume_push_velocity_diagnostic_counters(env)
    assert receipt["event_call_count"] == 1
    assert receipt["env_application_count"] == 2


def test_public_mdp_import_order_leaves_hope_wrapper_as_final_symbol():
    """Dependency-light proof of Python's final wildcard binding in public mdp."""

    tree = ast.parse(MDP_INIT_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    upstream_index = next(
        index
        for index, node in enumerate(imports)
        if node.level == 0
        and node.module == "isaaclab.envs.mdp"
        and any(alias.name == "*" for alias in node.names)
    )
    hope_index = next(
        index
        for index, node in enumerate(imports)
        if node.level == 1
        and node.module == "hope_push_events"
        and any(alias.name == "*" for alias in node.names)
    )
    assert upstream_index < hope_index
    assert callable(EV.push_by_setting_velocity)
    assert EV.push_by_setting_velocity.__module__ == EV.__name__
    assert not hasattr(EV, "__all__")  # wildcard includes the public wrapper


@pytest.mark.parametrize("env_ids, expected", [(None, 4), (torch.tensor([1, 3]), 2)])
def test_enabled_none_or_subset_records_finite_in_range_delta(
    monkeypatch, env_ids, expected
):
    calls = []
    delta = [0.20, -0.15, 0.05, -0.20, 0.25, -0.30]
    env = _Env()
    monkeypatch.setattr(
        EV, "_velocity_push_delegate", lambda: _delegate_with_delta(calls, delta)
    )

    before = env.scene["robot"].data.root_vel_w.clone()
    result = EV.push_by_setting_velocity(env, env_ids, velocity_range=RANGE)
    after = env.scene["robot"].data.root_vel_w
    assert result == "delegate-result" and len(calls) == 1
    ids = torch.arange(4) if env_ids is None else env_ids
    assert torch.equal(after[ids] - before[ids], torch.tensor(delta).expand(expected, -1))

    state = getattr(env, EV.PUSH_VELOCITY_DIAGNOSTIC_STATE_ATTR)
    assert all(
        torch.is_tensor(value)
        for key, value in state.items()
        if key not in {"schema_version", "axes", "bounds"}
    )
    receipt = EV.consume_push_velocity_diagnostic_counters(env)
    assert receipt["event_call_count"] == 1
    assert receipt["env_application_count"] == expected
    assert receipt["delta_nonfinite_element_count"] == 0
    for index, axis in enumerate(EV.PUSH_VELOCITY_AXES):
        assert receipt["axes"][axis]["observed_delta_min"] == pytest.approx(delta[index])
        assert receipt["axes"][axis]["observed_delta_max"] == pytest.approx(delta[index])
        assert receipt["axes"][axis]["below_range_count"] == 0
        assert receipt["axes"][axis]["above_range_count"] == 0
    json.dumps(receipt, allow_nan=False)


def test_counter_accumulates_until_consume_then_clears(monkeypatch):
    calls = []
    env = _Env(num_envs=3)
    monkeypatch.setattr(
        EV,
        "_velocity_push_delegate",
        lambda: _delegate_with_delta(calls, [0.1, -0.1, 0.0, 0.1, -0.1, 0.2]),
    )
    EV.push_by_setting_velocity(env, torch.tensor([0, 2]), velocity_range=RANGE)
    # Simulate any reset-side mutation unrelated to the event ledger.  There is
    # intentionally no reset hook capable of clearing the counters.
    env.scene["robot"].data.root_vel_w.zero_()
    EV.push_by_setting_velocity(env, torch.tensor([1]), velocity_range=RANGE)
    receipt = EV.consume_push_velocity_diagnostic_counters(env)
    assert receipt["event_call_count"] == 2
    assert receipt["env_application_count"] == 3
    assert len(calls) == 2
    cleared = EV.consume_push_velocity_diagnostic_counters(env)
    assert cleared["event_call_count"] == 0
    assert cleared["env_application_count"] == 0
    assert all(
        values["observed_delta_min"] is None
        and values["observed_delta_max"] is None
        for values in cleared["axes"].values()
    )


def test_nonfinite_and_axis_bounds_are_booked_without_a_second_write(monkeypatch):
    calls = []
    env = _Env(num_envs=2)
    delta = [0.30, -0.30, float("nan"), 0.0, 0.40, -0.50]
    monkeypatch.setattr(
        EV, "_velocity_push_delegate", lambda: _delegate_with_delta(calls, delta)
    )
    EV.push_by_setting_velocity(env, None, velocity_range=RANGE)
    assert len(calls) == 1
    # The wrapper did not sanitize or rewrite the delegate's physical result.
    observed = env.scene["robot"].data.root_vel_w
    assert torch.isnan(observed[:, 2]).all()
    assert torch.allclose(observed[:, 0], torch.full((2,), 0.30))
    receipt = EV.consume_push_velocity_diagnostic_counters(env)
    assert receipt["delta_nonfinite_element_count"] == 2
    assert receipt["axes"]["x"]["above_range_count"] == 2
    assert receipt["axes"]["y"]["below_range_count"] == 2
    assert receipt["axes"]["pitch"]["above_range_count"] == 2
    assert receipt["axes"]["yaw"]["below_range_count"] == 2
    assert receipt["axes"]["z"]["observed_delta_min"] is None


def test_empty_selection_still_delegates_once_and_books_the_event(monkeypatch):
    calls = []
    env = _Env()
    monkeypatch.setattr(
        EV,
        "_velocity_push_delegate",
        lambda: _delegate_with_delta(calls, [0.0] * 6),
    )
    EV.push_by_setting_velocity(
        env, torch.tensor([], dtype=torch.long), velocity_range=RANGE
    )
    receipt = EV.consume_push_velocity_diagnostic_counters(env)
    assert len(calls) == 1
    assert receipt["event_call_count"] == 1
    assert receipt["env_application_count"] == 0


def test_runner_consumes_and_prints_once_even_when_dashboard_logs_are_disabled():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    log_body = source[source.index("    def log("):source.index("    def _consume_actual_joint_forbidden_diagnostic")]
    assert log_body.count("self._consume_push_velocity_diagnostic_update(step)") == 1
    method = source[
        source.index("    def _consume_push_velocity_diagnostic_update"):
        source.index("    def _notify_command_terms_rollout_end")
    ]
    assert "if not self._action_ball_diagnostic_unauthorized():" in method
    assert "consume_push_velocity_diagnostic_counters(env)" in method
    assert "HOPE_PUSH_VELOCITY_DIAGNOSTIC_UPDATE_JSON=" in method
    assert method.index("consume_push_velocity_diagnostic_counters(env)") < method.index(
        "_push_velocity_diagnostic_consumed_step = int(step)"
    )


def test_event_hot_path_has_no_device_to_host_value_sync():
    source = EVENT_PATH.read_text(encoding="utf-8")
    wrapper = source[
        source.index("def push_by_setting_velocity("):
        source.index("def consume_push_velocity_diagnostic_counters(")
    ]
    assert ".item(" not in wrapper
    assert ".tolist(" not in wrapper
    consumer = source[
        source.index("def consume_push_velocity_diagnostic_counters("):
        source.index("def push_combined_exclusive(")
    ]
    assert consumer.count(".tolist()") == 1
