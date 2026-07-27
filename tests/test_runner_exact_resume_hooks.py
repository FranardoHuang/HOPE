from __future__ import annotations

import ast
import copy
import importlib.util
import io
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
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env
            self.cfg = dict(train_cfg or {})
            self.log_dir = log_dir
            self.device = device
            self.disable_logs = True
            self.current_learning_iteration = 0
            self.alg = SimpleNamespace(update=lambda: None)

        def load(self, _path, load_optimizer=True, **_kwargs):
            del load_optimizer
            self._base_load_source = _path
            if getattr(self, "_base_reloads_checkpoint", False):
                self._base_reloaded_checkpoint = torch.load(
                    _path, map_location="cpu", weights_only=False
                )
            self.current_learning_iteration = self._base_load_iteration
            self._test_events.append("base_load")
            return self._base_infos

        def log(self, locs, width=80, pad=35):
            del width, pad
            self._test_events.append(("base_log", int(locs["it"])))

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            del init_at_random_ep_len
            start = int(self.current_learning_iteration)
            for it in range(start, start + int(num_learning_iterations)):
                self.alg.update()
                self.current_learning_iteration = it
                # Mirror the installed RSL-RL contract: non-primary distributed ranks do not log.
                if not self.disable_logs:
                    self.log({"it": it})

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
    assert captured["active_term_names"] == ["task"]
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


def test_schema3_binds_complete_ordered_active_term_tuple(runner_module):
    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {}

        def load_exact_resume_state_dict(self, _state, strict=True):
            assert strict is True

    captured = _bare_runner(
        runner_module,
        terms=(("motion", ExplicitTerm()), ("racket", ExplicitTerm())),
    )._capture_environment_resume_state()

    reordered = _bare_runner(
        runner_module,
        terms=(("racket", ExplicitTerm()), ("motion", ExplicitTerm())),
    )
    with pytest.raises(RuntimeError, match="ordered command term identity mismatch"):
        reordered._restore_environment_resume_state(
            {"next_learning_iteration": 1, "environment_resume_state": captured}
        )

    missing = dict(captured)
    missing["command_terms"] = dict(captured["command_terms"])
    missing["command_terms"].pop("racket")
    target = _bare_runner(
        runner_module,
        terms=(("motion", ExplicitTerm()), ("racket", ExplicitTerm())),
    )
    with pytest.raises(RuntimeError, match="every active term in exact order"):
        target._restore_environment_resume_state(
            {"next_learning_iteration": 1, "environment_resume_state": missing}
        )


def test_action_ball_restore_orders_racket_then_motion_then_finalize_only_on_resume(
    runner_module,
):
    events = []

    class RacketTerm:
        def exact_resume_state_dict(self):
            return {"owner": "racket"}

        def load_exact_resume_state_dict(self, state, strict=True):
            assert state == {"owner": "racket"}
            assert strict is True
            events.append("racket")

    class MotionTerm:
        def exact_resume_state_dict(self):
            return {"owner": "motion"}

        def load_exact_resume_state_dict(self, state, strict=True):
            assert state == {"owner": "motion"}
            assert strict is True
            events.append("motion")

        def finalize_action_ball_exact_resume(self):
            events.append("finalize")

    source = _bare_runner(
        runner_module,
        terms=(("motion", MotionTerm()), ("racket_target", RacketTerm())),
    )
    source.env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(target_mode="action_ball")
        )
    )
    captured = source._capture_environment_resume_state()
    # Capturing/fresh training must never finalize a resume handoff.
    assert events == []

    target = _bare_runner(
        runner_module,
        terms=(("motion", MotionTerm()), ("racket_target", RacketTerm())),
    )
    target.env.cfg = source.env.cfg
    target._restore_environment_resume_state(
        {
            "next_learning_iteration": 1,
            "environment_resume_state": captured,
        }
    )
    assert events == ["racket", "motion", "finalize"]


def test_action_ball_restore_requires_finalize_before_mutating_owner(
    runner_module,
):
    events = []

    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {}

        def load_exact_resume_state_dict(self, _state, strict=True):
            assert strict is True
            events.append("loaded")

    source_motion = ExplicitTerm()
    source_motion.finalize_action_ball_exact_resume = lambda: None
    source = _bare_runner(
        runner_module,
        terms=(
            ("motion", source_motion),
            ("racket_target", ExplicitTerm()),
        ),
    )
    source.env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(target_mode="action_ball")
        )
    )
    captured = source._capture_environment_resume_state()
    events.clear()

    target = _bare_runner(
        runner_module,
        terms=(
            ("motion", ExplicitTerm()),
            ("racket_target", ExplicitTerm()),
        ),
    )
    target.env.cfg = source.env.cfg
    with pytest.raises(RuntimeError, match="requires Motion.*finalize"):
        target._restore_environment_resume_state(
            {
                "next_learning_iteration": 1,
                "environment_resume_state": captured,
            }
        )
    assert events == []


def test_schema3_does_not_skip_active_tuple_validation_without_manager(runner_module):
    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {}

        def load_exact_resume_state_dict(self, _state, strict=True):
            assert strict is True

    captured = _bare_runner(
        runner_module, terms=(("racket", ExplicitTerm()),)
    )._capture_environment_resume_state()
    target = _bare_runner(runner_module)
    target.env.command_manager = None

    with pytest.raises(RuntimeError, match="ordered command term identity mismatch"):
        target._restore_environment_resume_state(
            {"next_learning_iteration": 1, "environment_resume_state": captured}
        )
    # Strict structure/identity validation precedes all environment mutation.
    assert target.env.common_step_counter == 0


def test_task_first_requires_explicit_hooks_on_every_active_term(runner_module):
    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {}

        def load_exact_resume_state_dict(self, _state, strict=True):
            assert strict is True

    runner = _bare_runner(
        runner_module,
        terms=(("motion", ExplicitTerm()), ("racket", object())),
    )
    runner.env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(target_mode="task_first")
        )
    )
    with pytest.raises(RuntimeError, match="missing=\\['racket'\\]"):
        runner._capture_environment_resume_state()
    with pytest.raises(RuntimeError, match="missing=\\['racket'\\]"):
        runner._validate_task_first_exact_resume_terms()


def test_action_ball_requires_explicit_hooks_on_every_active_term(runner_module):
    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {}

        def load_exact_resume_state_dict(self, _state, strict=True):
            assert strict is True

    runner = _bare_runner(
        runner_module,
        terms=(("motion", ExplicitTerm()), ("racket", object())),
    )
    runner.env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(target_mode="action_ball")
        )
    )
    with pytest.raises(RuntimeError, match="action-ball.*missing=\\['racket'\\]"):
        runner._capture_environment_resume_state()
    with pytest.raises(RuntimeError, match="action-ball.*missing=\\['racket'\\]"):
        runner._validate_task_first_exact_resume_terms()


def test_task_first_capture_rechecks_manager_and_nonempty_active_tuple(runner_module):
    for manager, error in (
        (None, "requires a command manager"),
        (_Manager(()), "requires active command terms"),
    ):
        runner = _bare_runner(runner_module)
        runner.env.command_manager = manager
        runner.env.cfg = SimpleNamespace(
            commands=SimpleNamespace(
                racket_target=SimpleNamespace(target_mode="task_first")
            )
        )
        with pytest.raises(RuntimeError, match=error):
            runner._capture_environment_resume_state()


def test_task_first_constructor_rejects_legacy_term_before_first_rollout(runner_module):
    env = _Env((("racket", object()),))
    env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(target_mode="task_first")
        )
    )

    with pytest.raises(RuntimeError, match="missing=\\['racket'\\]"):
        runner_module.MotionOnPolicyRunner(env, {})


def test_required_exact_resume_constructor_rejects_evaluation_path(runner_module):
    with pytest.raises(ValueError, match="requires log_dir"):
        runner_module.MotionOnPolicyRunner(
            _Env(),
            {},
            log_dir=None,
            require_exact_resume_state=True,
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


def test_schema1_scalar_and_tensor_restore_remains_compatible(runner_module):
    class LegacyTerm:
        def __init__(self):
            self._curr_perturb_scale = 0.0
            self.bin_failed_count = torch.zeros(2)

    term = LegacyTerm()
    runner = _bare_runner(runner_module, terms=(("legacy", term),))
    runner._restore_environment_resume_state(
        {
            "next_learning_iteration": 2,
            "environment_resume_state": {
                "schema_version": 1,
                "common_step_counter": 17,
                "command_terms": {
                    "legacy": {
                        "scalars": {"_curr_perturb_scale": 0.5},
                        "tensors": {"bin_failed_count": torch.tensor([2.0, 3.0])},
                    }
                },
            },
        }
    )

    assert runner.env.common_step_counter == 17
    assert term._curr_perturb_scale == pytest.approx(0.5)
    assert torch.equal(term.bin_failed_count, torch.tensor([2.0, 3.0]))


def test_unknown_environment_schema_fails_even_without_command_manager(runner_module):
    runner = _bare_runner(runner_module)
    runner.env.command_manager = None

    with pytest.raises(RuntimeError, match="unsupported environment exact-resume schema 0"):
        runner._restore_environment_resume_state(
            {
                "next_learning_iteration": 2,
                "environment_resume_state": {
                    "schema_version": 0,
                    "common_step_counter": 17,
                },
            }
        )
    assert runner.env.common_step_counter == 0


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
    runner.training_contract_schema_version = 1
    runner.training_contract_sha256 = "a" * 64
    runner.training_contract_lineage_exact = True
    runner._checkpoint_byte_snapshot = lambda _path: io.BytesIO(b"checkpoint")
    return runner


def _complete_schema3_resume_state(*, next_iteration=5, environment_state=None):
    if environment_state is None:
        environment_state = {
            "schema_version": 3,
            "common_step_counter": 0,
            "active_term_names": [],
            "command_terms": {},
        }
    return {
        "schema_version": 3,
        "next_learning_iteration": next_iteration,
        "tot_timesteps": 88,
        "tot_time": 1.5,
        "algorithm_learning_rate": 1.0e-3,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": [],
        "torch_cuda_device_count": 0,
        "environment_resume_state": environment_state,
    }


def _install_fresh_adam_and_build_resumable_state(runner, *, amsgrad=False):
    live_parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    runner.alg.optimizer = torch.optim.Adam(
        [live_parameter], lr=1.0e-3, amsgrad=amsgrad
    )

    donor_parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    donor = torch.optim.Adam(
        [donor_parameter], lr=1.0e-3, amsgrad=amsgrad
    )
    donor_parameter.square().sum().backward()
    donor.step()
    return donor.state_dict()


def _complete_checkpoint(exact_state, optimizer_state, *, iteration=4):
    return {
        "iter": iteration,
        "optimizer_state_dict": optimizer_state,
        "infos": {
            "hope_exact_resume_state": exact_state,
            "schema": 1,
            "sha": "a" * 64,
            "lineage": 1,
        },
    }


def test_load_restores_exact_rng_after_command_state_and_before_reset(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    runner.require_exact_resume_state = True
    optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
    exact_state = _complete_schema3_resume_state()
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: _complete_checkpoint(
            exact_state, optimizer_state
        ),
    )
    runner._restore_environment_resume_state = (
        lambda _state: events.append("command_state") or (0, "checkpoint")
    )
    runner._restore_exact_rng_state = lambda _state: events.append("rng")

    assert runner.load("checkpoint.pt") == {"base": "infos"}
    assert events == ["base_load", "command_state", "rng", "reset"]
    assert runner.current_learning_iteration == 5


def test_action_ball_load_is_data_only_and_defers_true_reset_to_learn(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    runner.env.cfg = SimpleNamespace(
        commands=SimpleNamespace(
            racket_target=SimpleNamespace(target_mode="action_ball")
        )
    )
    runner.require_exact_resume_state = True
    optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
    exact_state = _complete_schema3_resume_state()
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: _complete_checkpoint(
            exact_state, optimizer_state
        ),
    )
    runner._restore_environment_resume_state = (
        lambda _state: events.append("command_state") or (0, "checkpoint")
    )
    runner._restore_exact_rng_state = lambda _state: events.append("rng")

    runner.load("checkpoint.pt")
    assert events == ["base_load", "command_state", "rng"]
    assert runner._action_ball_resume_reset_pending is True

    runner.disable_logs = True
    runner.writer = None
    runner.current_learning_iteration = 5
    runner.alg.update = lambda: events.append("update")
    runner.learn(num_learning_iterations=1)
    assert events == [
        "base_load",
        "command_state",
        "rng",
        "reset",
        "update",
    ]
    assert runner._action_ball_resume_reset_pending is False


def test_runner_source_parses_as_python38():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(RUNNER_PATH), feature_version=(3, 8))


def test_required_exact_resume_rejects_actor_only_and_optimizerless_checkpoints(
    runner_module, monkeypatch
):
    exact_state = _complete_schema3_resume_state()
    for corruption, load_optimizer in (
        ("none", False),
        ("missing", True),
        ("empty_state", True),
        ("missing_params", True),
        ("non_dict_group", True),
        ("nan_lr", True),
        ("bad_betas", True),
        ("fractional_step", True),
        ("negative_second_moment", True),
        ("amsgrad_max_below_second_moment", True),
    ):
        events = []
        runner = _configure_load_runner(runner_module, events)
        runner.require_exact_resume_state = True
        optimizer_state = _install_fresh_adam_and_build_resumable_state(
            runner, amsgrad=corruption == "amsgrad_max_below_second_moment"
        )
        checkpoint = _complete_checkpoint(exact_state, optimizer_state)
        if corruption == "missing":
            checkpoint.pop("optimizer_state_dict")
        elif corruption == "empty_state":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            checkpoint["optimizer_state_dict"]["state"] = {}
        elif corruption == "missing_params":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            checkpoint["optimizer_state_dict"]["param_groups"][0].pop("params")
        elif corruption == "non_dict_group":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            checkpoint["optimizer_state_dict"]["param_groups"] = [42]
        elif corruption == "nan_lr":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
        elif corruption == "bad_betas":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            checkpoint["optimizer_state_dict"]["param_groups"][0]["betas"] = "bad"
        elif corruption == "fractional_step":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            entry = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
            entry["step"] = torch.tensor(1.5)
        elif corruption == "negative_second_moment":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            entry = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
            entry["exp_avg_sq"].fill_(-1.0)
        elif corruption == "amsgrad_max_below_second_moment":
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer_state)
            entry = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
            entry["max_exp_avg_sq"].zero_()
        monkeypatch.setattr(
            runner_module.torch,
            "load",
            lambda *_args, _checkpoint=checkpoint, **_kwargs: _checkpoint,
        )

        with pytest.raises(RuntimeError, match="optimizer|actor-only"):
            runner.load("checkpoint.pt", load_optimizer=load_optimizer)
        # Strict envelope validation happens before policy/optimizer bytes are applied.
        assert events == []


def test_required_exact_resume_rejects_missing_legacy_incomplete_and_stale_state(
    runner_module, monkeypatch
):
    schema2 = {
        "schema_version": 2,
        "next_learning_iteration": 5,
        "environment_resume_state": {
            "schema_version": 2,
            "common_step_counter": 120,
            "command_terms": {},
        },
    }
    incomplete = _complete_schema3_resume_state()
    incomplete.pop("torch_random_state")
    invalid_lr = _complete_schema3_resume_state()
    invalid_lr["algorithm_learning_rate"] = float("nan")
    invalid_rng = _complete_schema3_resume_state()
    invalid_rng["torch_random_state"] = torch.zeros(1, dtype=torch.uint8)
    invalid_common_step_counter = _complete_schema3_resume_state()
    invalid_common_step_counter["environment_resume_state"][
        "common_step_counter"
    ] = "-7"
    cases = (
        (None, "requires hope_exact_resume_state schema 3"),
        (schema2, "requires hope_exact_resume_state schema 3"),
        (incomplete, "schema-3 state is incomplete"),
        (invalid_lr, "invalid algorithm_learning_rate"),
        (invalid_rng, "shape/dtype mismatch"),
        (invalid_common_step_counter, "common_step_counter"),
        (
            _complete_schema3_resume_state(next_iteration=99),
            "stale next_learning_iteration",
        ),
    )
    for state, error in cases:
        events = []
        runner = _configure_load_runner(runner_module, events)
        runner.require_exact_resume_state = True
        optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
        checkpoint = _complete_checkpoint(
            _complete_schema3_resume_state() if state is None else state,
            optimizer_state,
        )
        if state is None:
            checkpoint["infos"].pop("hope_exact_resume_state")
        monkeypatch.setattr(
            runner_module.torch,
            "load",
            lambda *_args, _checkpoint=checkpoint, **_kwargs: _checkpoint,
        )

        with pytest.raises(RuntimeError, match=error):
            runner.load("checkpoint.pt")
        assert events == []


def test_required_exact_resume_rechecks_contract_on_consumed_snapshot(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    runner.require_exact_resume_state = True
    optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
    checkpoint = _complete_checkpoint(
        _complete_schema3_resume_state(), optimizer_state
    )
    checkpoint["infos"]["sha"] = "b" * 64
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )

    with pytest.raises(RuntimeError, match="exact training contract"):
        runner.load("checkpoint.pt")
    assert events == []


def test_required_exact_resume_rejects_unvalidated_rnd_state(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    runner.require_exact_resume_state = True
    runner.alg.rnd = object()
    optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
    checkpoint = _complete_checkpoint(
        _complete_schema3_resume_state(), optimizer_state
    )
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )

    with pytest.raises(RuntimeError, match="exact RND resume"):
        runner.load("checkpoint.pt")
    assert events == []


def test_required_exact_resume_propagates_strict_term_state_failure(
    runner_module, monkeypatch
):
    class ExplicitTerm:
        def exact_resume_state_dict(self):
            return {"identity": "current"}

        def load_exact_resume_state_dict(self, state, strict=True):
            assert strict is True
            if state != {"identity": "current"}:
                raise ValueError("term state identity mismatch")

    events = []
    runner = _configure_load_runner(runner_module, events)
    runner.env.command_manager = _Manager((("task", ExplicitTerm()),))
    runner.require_exact_resume_state = True
    optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
    environment_state = runner._capture_environment_resume_state()
    environment_state["command_terms"]["task"]["exact_state"]["identity"] = "other"
    exact_state = _complete_schema3_resume_state(
        environment_state=environment_state
    )
    monkeypatch.setattr(
        runner_module.torch,
        "load",
        lambda *_args, **_kwargs: _complete_checkpoint(
            exact_state, optimizer_state
        ),
    )

    with pytest.raises(ValueError, match="term state identity mismatch"):
        runner.load("checkpoint.pt")
    assert events == ["base_load"]
    assert "reset" not in events


def test_required_exact_resume_uses_one_immutable_snapshot_for_base_load(
    runner_module, monkeypatch
):
    events = []
    runner = _configure_load_runner(runner_module, events)
    runner.require_exact_resume_state = True
    runner._base_reloads_checkpoint = True
    optimizer_state = _install_fresh_adam_and_build_resumable_state(runner)
    exact_state = _complete_schema3_resume_state()
    checkpoint = _complete_checkpoint(exact_state, optimizer_state)
    snapshot = io.BytesIO(b"one immutable checkpoint")
    runner._checkpoint_byte_snapshot = lambda _path: snapshot
    reads = []

    def load_snapshot(source, *_args, **_kwargs):
        assert source is snapshot
        reads.append(source.read())
        return checkpoint

    monkeypatch.setattr(runner_module.torch, "load", load_snapshot)
    runner._restore_exact_rng_state = lambda _state: None

    runner.load("checkpoint.pt")

    assert reads == [b"one immutable checkpoint", b"one immutable checkpoint"]
    assert runner._base_load_source is snapshot
    assert runner._base_reloaded_checkpoint is checkpoint


def test_outer_schema3_refuses_missing_or_legacy_nested_environment_state(
    runner_module, monkeypatch
):
    for nested, next_iteration in (
        (None, 5),
        (
            {
                "schema_version": 2,
                "common_step_counter": 0,
                "command_terms": {},
            },
            5,
        ),
        # Structural schema-3 validity is checked before the stale-state warm-start escape hatch.
        (None, 999),
    ):
        events = []
        runner = _configure_load_runner(runner_module, events)
        state = {
            "schema_version": 3,
            "next_learning_iteration": next_iteration,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_states": [],
            "torch_cuda_device_count": 0,
        }
        if nested is not None:
            state["environment_resume_state"] = nested
        monkeypatch.setattr(
            runner_module.torch,
            "load",
            lambda *_args, _state=state, **_kwargs: {
                "infos": {"hope_exact_resume_state": _state}
            },
        )
        with pytest.raises(RuntimeError, match="requires a schema-3"):
            runner.load("checkpoint.pt")
        assert events == ["base_load"]


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


def test_rollout_end_hooks_are_exact_once_without_rank0_logging(runner_module):
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
    runner.current_learning_iteration = 7
    runner.alg = SimpleNamespace(
        update=lambda: events.append("update") or {"loss": 0.0}
    )
    original_update = runner.alg.update
    runner.learn(num_learning_iterations=2)
    assert events == [
        "update",
        ("a", 7),
        ("b", 7),
        "update",
        ("a", 8),
        ("b", 8),
    ]
    assert runner.alg.update is original_update


def test_rollout_end_hooks_precede_receipts_without_double_invocation(runner_module):
    events = []

    class Term:
        def on_rollout_end(self, step):
            events.append(("hook", step))

    runner = _bare_runner(
        runner_module,
        terms=(("task", Term()),),
        events=events,
    )
    runner.disable_logs = False
    runner.writer = None
    runner.current_learning_iteration = 3
    runner.alg = SimpleNamespace(update=lambda: events.append("update"))
    runner._consume_exact_behavior_updates = (
        lambda step: events.append(("consume", step)) or {}
    )

    runner.learn(num_learning_iterations=1)

    assert events == [
        "update",
        ("hook", 3),
        ("base_log", 3),
        ("consume", 3),
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
    runner.current_learning_iteration = 8
    runner.alg = SimpleNamespace(update=lambda: events.append("update"))
    original_update = runner.alg.update

    with pytest.raises(LookupError, match="curriculum failure"):
        runner.learn(num_learning_iterations=1)
    assert events == ["update", "hook"]
    assert runner.alg.update is original_update
