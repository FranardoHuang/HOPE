from __future__ import annotations

import importlib.util
import random
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "my_on_policy_runner.py"
)


def _load_runner_module():
    """Load the runner with pure-Python stand-ins for Isaac/rsl_rl dependencies."""

    saved_modules = {}

    def install(name: str, module: types.ModuleType) -> None:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    rsl_rl = types.ModuleType("rsl_rl")
    rsl_rl_env = types.ModuleType("rsl_rl.env")
    rsl_rl_runners = types.ModuleType("rsl_rl.runners")
    rsl_rl_on_policy = types.ModuleType("rsl_rl.runners.on_policy_runner")

    class VecEnv:
        pass

    class OnPolicyRunner:
        def load(self, _path, load_optimizer=True, **_kwargs):
            del load_optimizer
            self.current_learning_iteration = self._base_load_iteration
            self._test_events.append("base_load")
            return self._base_infos

        def log(self, locs, width=80, pad=35):
            del width, pad
            self._test_events.append(("base_log", int(locs["it"])))

    rsl_rl_env.VecEnv = VecEnv
    rsl_rl_on_policy.OnPolicyRunner = OnPolicyRunner
    install("rsl_rl", rsl_rl)
    install("rsl_rl.env", rsl_rl_env)
    install("rsl_rl.runners", rsl_rl_runners)
    install("rsl_rl.runners.on_policy_runner", rsl_rl_on_policy)

    isaaclab_rl = types.ModuleType("isaaclab_rl")
    isaaclab_rl_rsl = types.ModuleType("isaaclab_rl.rsl_rl")
    isaaclab_rl_rsl.export_policy_as_onnx = lambda *_args, **_kwargs: None
    install("isaaclab_rl", isaaclab_rl)
    install("isaaclab_rl.rsl_rl", isaaclab_rl_rsl)

    exporter = types.ModuleType("whole_body_tracking.utils.exporter")
    exporter.attach_onnx_metadata = lambda *_args, **_kwargs: None
    exporter.export_motion_policy_as_onnx = lambda *_args, **_kwargs: False
    exporter.is_empirical_normalizer = lambda _value: False
    install("whole_body_tracking.utils.exporter", exporter)

    contract = types.ModuleType("whole_body_tracking.utils.training_contract")
    contract.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY = "lineage"
    contract.CHECKPOINT_CONTRACT_SCHEMA_KEY = "schema"
    contract.CHECKPOINT_CONTRACT_SHA_KEY = "sha"
    contract.CHECKPOINT_LAUNCH_CLAIM_SHA_KEY = "claim"
    contract.TRAINING_CONTRACT_SCHEMA_VERSION = 1
    contract.validate_training_launch_claim_sha256 = lambda value: value
    install("whole_body_tracking.utils.training_contract", contract)

    module_name = f"_runner_exact_resume_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


@pytest.fixture(scope="module")
def runner_module():
    return _load_runner_module()


class _Manager:
    def __init__(self, terms):
        self._terms = dict(terms)
        self.active_terms = tuple(self._terms)

    def get_term(self, name):
        return self._terms[name]


class _Env:
    def __init__(self, terms=(), events=None):
        self.unwrapped = self
        self.common_step_counter = 0
        self.command_manager = _Manager(terms)
        self._events = events

    def reset(self):
        if self._events is not None:
            self._events.append("reset")


def _bare_runner(runner_module, *, terms=(), events=None):
    runner = object.__new__(runner_module.MotionOnPolicyRunner)
    runner.env = _Env(terms, events=events)
    runner._test_events = events if events is not None else []
    return runner


def _term_type(term) -> str:
    return f"{type(term).__module__}.{type(term).__qualname__}"


def test_explicit_command_state_takes_priority_and_restores_strictly(runner_module):
    class ExplicitTerm:
        def __init__(self):
            self._curr_perturb_scale = 99.0
            self.loads = []

        def exact_resume_state_dict(self):
            return {"manifest_sha256": "abc", "level": 2}

        def load_exact_resume_state_dict(self, state, strict=True):
            assert state["manifest_sha256"] == "abc"
            self.loads.append((dict(state), strict))

    source_term = ExplicitTerm()
    source = _bare_runner(runner_module, terms=(("task", source_term),))
    captured = source._capture_environment_resume_state()

    assert captured["schema_version"] == 3
    assert captured["command_terms"]["task"] == {
        "capture_mode": "explicit",
        "term_type": _term_type(source_term),
        "exact_state": {"manifest_sha256": "abc", "level": 2},
    }
    assert "scalars" not in captured["command_terms"]["task"]

    restored_term = ExplicitTerm()
    target = _bare_runner(runner_module, terms=(("task", restored_term),))
    target._restore_environment_resume_state(
        {
            "next_learning_iteration": 4,
            "environment_resume_state": captured,
        }
    )
    assert restored_term.loads == [({"manifest_sha256": "abc", "level": 2}, True)]
    assert target.env.common_step_counter == 0


def test_explicit_hook_pair_and_identity_mismatches_fail_loud(runner_module):
    class GetterOnly:
        def exact_resume_state_dict(self):
            return {}

    runner = _bare_runner(runner_module, terms=(("task", GetterOnly()),))
    with pytest.raises(RuntimeError, match="as a pair"):
        runner._capture_environment_resume_state()

    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {"identity": "current"}

        def load_exact_resume_state_dict(self, state, strict=True):
            if strict and state["identity"] != "current":
                raise ValueError("identity mismatch")

    term = ExplicitTerm()
    runner = _bare_runner(runner_module, terms=(("task", term),))
    captured = runner._capture_environment_resume_state()
    captured["command_terms"]["task"]["exact_state"]["identity"] = "other"
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._restore_environment_resume_state(
            {"next_learning_iteration": 1, "environment_resume_state": captured}
        )

    runner_without_exact_term = _bare_runner(runner_module)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner_without_exact_term._restore_environment_resume_state(
            {"next_learning_iteration": 1, "environment_resume_state": captured}
        )


def test_schema2_legacy_attribute_scan_remains_compatible(runner_module):
    class LegacyTerm:
        def __init__(self):
            self._curr_perturb_scale = 0.0

    term = LegacyTerm()
    runner = _bare_runner(runner_module, terms=(("legacy", term),))
    runner._restore_environment_resume_state(
        {
            "next_learning_iteration": 9,
            "environment_resume_state": {
                "schema_version": 2,
                "common_step_counter": 123,
                "command_terms": {
                    "legacy": {
                        "scalars": {"_curr_perturb_scale": 0.75},
                        "tensors": {},
                        "tensor_dicts": {},
                    }
                },
            },
        }
    )
    assert runner.env.common_step_counter == 123
    assert term._curr_perturb_scale == pytest.approx(0.75)


def test_rng_round_trip_restores_python_numpy_and_torch(runner_module):
    runner = _bare_runner(runner_module)
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": cuda_states,
        "torch_cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.seed(404)
    np.random.seed(505)
    torch.manual_seed(606)
    runner._restore_exact_rng_state(state)

    actual = (random.random(), float(np.random.random()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_cuda_rng_count_and_shape_mismatch_fail_loud(runner_module, monkeypatch):
    runner = _bare_runner(runner_module)
    base_state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    monkeypatch.setattr(runner_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.zeros(4, dtype=torch.uint8), torch.zeros(4, dtype=torch.uint8)],
    )

    with pytest.raises(RuntimeError, match="device-count mismatch"):
        runner._restore_exact_rng_state(
            {
                **base_state,
                "torch_cuda_random_states": [torch.zeros(4, dtype=torch.uint8)],
                "torch_cuda_device_count": 1,
            }
        )

    with pytest.raises(RuntimeError, match="shape/dtype mismatch"):
        runner._restore_exact_rng_state(
            {
                **base_state,
                "torch_cuda_random_states": [
                    torch.zeros(3, dtype=torch.uint8),
                    torch.zeros(4, dtype=torch.uint8),
                ],
                "torch_cuda_device_count": 2,
            }
        )


def _configure_load_runner(runner_module, events):
    runner = _bare_runner(runner_module, events=events)
    runner.log_dir = "/tmp"
    runner._base_load_iteration = 4
    runner._base_infos = {"base": "infos"}
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    runner.num_steps_per_env = 24
    runner.alg = SimpleNamespace(schedule="fixed", learning_rate=1e-3)
    return runner


def test_load_restores_exact_rng_after_command_state_and_before_reset(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    exact_state = {
        "schema_version": 3,
        "next_learning_iteration": 5,
        "tot_timesteps": 88,
        "tot_time": 1.5,
    }
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: {
            "infos": {"hope_exact_resume_state": exact_state}
        },
    )
    runner._restore_environment_resume_state = (
        lambda _state: events.append("command_state") or (0, "checkpoint")
    )
    runner._restore_exact_rng_state = lambda _state: events.append("rng")

    assert runner.load("checkpoint.pt") == {"base": "infos"}
    assert events == ["base_load", "command_state", "rng", "reset"]
    assert runner.current_learning_iteration == 5


def test_legacy_load_does_not_guess_or_restore_rng(runner_module, monkeypatch):
    events = []
    runner = _configure_load_runner(runner_module, events)
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: {"infos": {}},
    )
    runner._restore_environment_resume_state = (
        lambda _state: events.append("legacy_command_state") or (0, "derived")
    )
    runner._restore_exact_rng_state = lambda _state: events.append("unexpected_rng")

    runner.load("legacy.pt")
    assert events == ["base_load", "legacy_command_state", "reset"]
    # Legacy checkpoints retain base rsl_rl iteration semantics.
    assert runner.current_learning_iteration == 4


def test_schema2_exact_state_restores_curriculum_but_not_rng(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    schema2_state = {
        "schema_version": 2,
        "next_learning_iteration": 5,
        "environment_resume_state": {
            "schema_version": 2,
            "common_step_counter": 120,
            "command_terms": {},
        },
    }
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: {
            "infos": {"hope_exact_resume_state": schema2_state}
        },
    )
    runner._restore_environment_resume_state = (
        lambda _state: events.append("schema2_command_state") or (120, "checkpoint")
    )
    runner._restore_exact_rng_state = lambda _state: events.append("unexpected_rng")

    runner.load("schema2.pt")
    assert events == ["base_load", "schema2_command_state", "reset"]
    assert runner.current_learning_iteration == 5


def test_rollout_end_hooks_are_exact_once_and_precede_all_receipts(runner_module):
    events = []

    class Term:
        metrics = {}

        def __init__(self, name):
            self.name = name

        def on_rollout_end(self, step):
            events.append((self.name, step))

    runner = _bare_runner(
        runner_module,
        terms=(("a", Term("a")), ("b", Term("b"))),
        events=events,
    )
    runner.disable_logs = True
    runner.writer = None
    runner._consume_exact_behavior_updates = (
        lambda step: events.append(("consume", step)) or {}
    )

    runner.log({"it": 7})
    runner.log({"it": 7})

    assert events == [
        ("a", 7),
        ("b", 7),
        ("base_log", 7),
        ("consume", 7),
        ("base_log", 7),
        ("consume", 7),
    ]


def test_rollout_end_exception_propagates_before_logging(runner_module):
    events = []

    class BrokenTerm:
        def on_rollout_end(self, _step):
            events.append("hook")
            raise LookupError("curriculum failure")

    runner = _bare_runner(
        runner_module,
        terms=(("broken", BrokenTerm()),),
        events=events,
    )
    runner.disable_logs = True
    runner.writer = None

    with pytest.raises(LookupError, match="curriculum failure"):
        runner.log({"it": 8})
    assert events == ["hook"]
