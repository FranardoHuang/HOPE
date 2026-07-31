"""Runner integration tests for schema-4 ActionManager delay state."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from test_exact_resume_state import (
    _load_contract_module,
    _load_runner_module,
    _make_runner,
)


@pytest.fixture()
def runner_module(monkeypatch):
    return _load_runner_module(monkeypatch, _load_contract_module())


class _ActionManager:
    def __init__(self, ordered_terms):
        self._terms = dict(ordered_terms)
        self.active_terms = tuple(self._terms)

    def get_term(self, name):
        return self._terms[name]


class _DelayTerm:
    def __init__(self, values, *, enabled=True, runtime_state_required=None):
        self.control_step_action_delay_enabled = enabled
        self.action_runtime_state_required = (
            enabled
            if runtime_state_required is None
            else runtime_state_required
        )
        self.values = torch.as_tensor(values, dtype=torch.long).clone()
        self.validate_calls = 0
        self.load_calls = 0

    def action_delay_exact_resume_state_dict(self):
        return {
            "schema_version": 1,
            "lag_steps": self.values.detach().cpu().clone(),
        }

    def validate_action_delay_exact_resume_state_dict(
        self, state, *, strict=True
    ):
        self.validate_calls += 1
        if strict is not True:
            raise ValueError("strict required")
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "lag_steps",
        }:
            raise ValueError("malformed delay state")
        value = state["lag_steps"]
        if (
            state["schema_version"] != 1
            or not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != self.values.dtype
            or tuple(value.shape) != tuple(self.values.shape)
        ):
            raise ValueError("delay state tensor mismatch")

    def load_action_delay_exact_resume_state_dict(
        self, state, *, strict=True
    ):
        self.validate_action_delay_exact_resume_state_dict(
            state, strict=strict
        )
        self.values.copy_(state["lag_steps"])
        self.load_calls += 1

    def control_step_action_delay_runtime_receipt(self):
        histogram = {
            str(step): int(torch.sum(self.values == step).item())
            for step in range(3)
        }
        return {
            "schema_version": 1,
            "kind": (
                "whole_body_tracking."
                "policy_control_step_action_delay_receipt"
            ),
            "contract": {
                "enabled": True,
                "semantic_unit": "policy_control_step",
            },
            "num_envs": int(self.values.numel()),
            "initialized_env_count": int(self.values.numel()),
            "lag_histogram": histogram,
        }


class _IdentityOnlyTerm:
    control_step_action_delay_enabled = False


def _with_actions(inner, delay, *, reversed_order=False):
    rows = (
        (("aux", _IdentityOnlyTerm()), ("joint_pos", delay))
        if reversed_order
        else (("joint_pos", delay), ("aux", _IdentityOnlyTerm()))
    )
    inner.action_manager = _ActionManager(rows)


def test_schema4_capture_and_restore_complete_ordered_action_terms(
    runner_module, tmp_path
):
    source, source_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "source"), filled=True, counter=88
    )
    source_delay = _DelayTerm([0, 2, 1, 2])
    _with_actions(source_inner, source_delay)
    saved = source._capture_environment_resume_state()

    assert saved["schema_version"] == 4
    assert saved["active_action_term_names"] == ["joint_pos", "aux"]
    assert tuple(saved["action_terms"]) == ("joint_pos", "aux")
    assert saved["action_terms"]["joint_pos"]["capture_mode"] == "explicit_delay"
    assert saved["action_terms"]["aux"]["capture_mode"] == "identity_only"

    resumed, resumed_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "resumed"), filled=False, counter=0
    )
    resumed_delay = _DelayTerm([0, 0, 0, 0])
    _with_actions(resumed_inner, resumed_delay)
    resumed._restore_environment_resume_state(
        {"next_learning_iteration": 9, "environment_resume_state": saved}
    )
    assert resumed_inner.common_step_counter == 88
    assert torch.equal(resumed_delay.values, source_delay.values)
    assert resumed_delay.validate_calls == 2  # phase-one stage + loader recheck
    assert resumed_delay.load_calls == 1


def test_schema4_captures_no_delay_containment_runtime_state_without_delay_receipt(
    runner_module, tmp_path
):
    source, source_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "source"), filled=True, counter=89
    )
    source_containment = _DelayTerm(
        [-1, 0, 1, -1],
        enabled=False,
        runtime_state_required=True,
    )
    _with_actions(source_inner, source_containment)
    saved = source._capture_environment_resume_state()

    assert saved["schema_version"] == 4
    assert saved["action_terms"]["joint_pos"]["capture_mode"] == "explicit_delay"
    # Runtime-state capture is broader than the delay receipt: a (0,0)-delay containment run
    # must persist its latch but must not claim an actuator-delay distribution.
    source.training_contract_sha256 = "a" * 64
    assert source._emit_control_step_action_delay_runtime_receipt() is None

    resumed, resumed_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "resumed"), filled=False, counter=0
    )
    resumed_containment = _DelayTerm(
        [0, 0, 0, 0],
        enabled=False,
        runtime_state_required=True,
    )
    _with_actions(resumed_inner, resumed_containment)
    resumed._restore_environment_resume_state(
        {"next_learning_iteration": 9, "environment_resume_state": saved}
    )
    assert resumed_inner.common_step_counter == 89
    assert torch.equal(
        resumed_containment.values, source_containment.values
    )
    assert resumed_containment.validate_calls == 2
    assert resumed_containment.load_calls == 1


def test_action_runtime_state_required_flag_must_be_exact_bool(
    runner_module, tmp_path
):
    runner, inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "run"), filled=True, counter=1
    )
    malformed = _DelayTerm([0, 0], enabled=False)
    malformed.action_runtime_state_required = None
    _with_actions(inner, malformed)
    with pytest.raises(RuntimeError, match="runtime-state-required flag"):
        runner._capture_environment_resume_state()

    inconsistent = _DelayTerm(
        [0, 1], enabled=True, runtime_state_required=False
    )
    _with_actions(inner, inconsistent)
    with pytest.raises(RuntimeError, match="cannot be false while delay is enabled"):
        runner._capture_environment_resume_state()


def test_schema4_malformed_action_state_is_rejected_before_any_mutation(
    runner_module, tmp_path
):
    source, source_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "source"), filled=True, counter=91
    )
    _with_actions(source_inner, _DelayTerm([0, 1, 2, 0]))
    saved = source._capture_environment_resume_state()
    malformed = deepcopy(saved)
    malformed["action_terms"]["joint_pos"]["exact_state"][
        "lag_steps"
    ] = torch.tensor([1, 2], dtype=torch.long)

    resumed, resumed_inner, _, resumed_commands = _make_runner(
        runner_module, log_dir=str(tmp_path / "resumed"), filled=False, counter=17
    )
    delay = _DelayTerm([2, 2, 2, 2])
    _with_actions(resumed_inner, delay)
    command_before = resumed_commands["racket_target"]._curr_perturb_scale
    with pytest.raises(ValueError, match="tensor mismatch"):
        resumed._restore_environment_resume_state(
            {
                "next_learning_iteration": 9,
                "environment_resume_state": malformed,
            }
        )
    assert resumed_inner.common_step_counter == 17
    assert torch.equal(delay.values, torch.tensor([2, 2, 2, 2]))
    assert delay.load_calls == 0
    assert resumed_commands["racket_target"]._curr_perturb_scale == command_before


def test_enabled_delay_rejects_schema3_as_fresh_only_without_mutation(
    runner_module, tmp_path
):
    source, source_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "source"), filled=True, counter=33
    )
    saved4 = source._capture_environment_resume_state()
    saved3 = {
        key: value
        for key, value in saved4.items()
        if key not in ("active_action_term_names", "action_terms")
    }
    saved3["schema_version"] = 3

    resumed, resumed_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "resumed"), filled=False, counter=12
    )
    delay = _DelayTerm([0, 0, 0, 0], enabled=True)
    _with_actions(resumed_inner, delay)
    with pytest.raises(RuntimeError, match="fresh-only"):
        resumed._restore_environment_resume_state(
            {"next_learning_iteration": 4, "environment_resume_state": saved3}
        )
    assert resumed_inner.common_step_counter == 12
    assert delay.load_calls == 0


def test_schema3_original_keys_remain_compatible_when_delay_disabled(
    runner_module, tmp_path
):
    source, source_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "source"), filled=True, counter=44
    )
    saved4 = source._capture_environment_resume_state()
    saved3 = {
        key: value
        for key, value in saved4.items()
        if key not in ("active_action_term_names", "action_terms")
    }
    saved3["schema_version"] = 3

    resumed, resumed_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "resumed"), filled=False, counter=0
    )
    _with_actions(
        resumed_inner, _DelayTerm([0, 0, 0, 0], enabled=False)
    )
    resumed._restore_environment_resume_state(
        {"next_learning_iteration": 3, "environment_resume_state": saved3}
    )
    assert resumed_inner.common_step_counter == 44


def test_schema4_action_order_drift_fails_before_clock_commit(
    runner_module, tmp_path
):
    source, source_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "source"), filled=True, counter=55
    )
    _with_actions(source_inner, _DelayTerm([0, 1, 2, 0]))
    saved = source._capture_environment_resume_state()

    resumed, resumed_inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "resumed"), filled=False, counter=7
    )
    _with_actions(
        resumed_inner, _DelayTerm([0, 0, 0, 0]), reversed_order=True
    )
    with pytest.raises(RuntimeError, match="ordered action term identity"):
        resumed._restore_environment_resume_state(
            {"next_learning_iteration": 3, "environment_resume_state": saved}
        )
    assert resumed_inner.common_step_counter == 7


def test_first_reset_runtime_receipt_is_contract_bound_and_idempotent(
    runner_module, tmp_path, capsys
):
    runner, inner, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path / "run"), filled=False, counter=0
    )
    _with_actions(inner, _DelayTerm([0, 1, 2, 0]))
    runner.training_contract_sha256 = "a" * 64
    first = runner._emit_control_step_action_delay_runtime_receipt()
    second = runner._emit_control_step_action_delay_runtime_receipt()
    assert second is first
    assert first["training_contract_sha256"] == "a" * 64
    assert first["delay_terms"][0]["lag_histogram"] == {
        "0": 2,
        "1": 1,
        "2": 1,
    }
    output = capsys.readouterr().out
    assert output.count("HOPE_CONTROL_STEP_ACTION_DELAY_RUNTIME_JSON=") == 1
