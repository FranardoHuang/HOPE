"""Host-side contract tests for the task-first RacketTargetCommand runtime.

The Isaac module is intentionally not imported.  Small shipped functions/methods are extracted
from its AST and executed against dependency-light fakes; the optional face-cone test uses real
torch when the host provides it.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "hope_commands.py"
)
CURRICULUM_PATH = COMMAND_PATH.with_name("task_first_curriculum.py")
SOURCE = COMMAND_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(COMMAND_PATH))


def _load_curriculum():
    name = "task_first_curriculum_runtime_test"
    spec = importlib.util.spec_from_file_location(name, CURRICULUM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = _load_curriculum()


def _install_dependency_light_task_first_package(monkeypatch):
    """Expose only the pure task-first modules without executing package ``__init__`` files."""

    package_root = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking"
    package_paths = {
        "whole_body_tracking": package_root,
        "whole_body_tracking.tasks": package_root / "tasks",
        "whole_body_tracking.tasks.tracking": package_root / "tasks" / "tracking",
        "whole_body_tracking.tasks.tracking.mdp": (
            package_root / "tasks" / "tracking" / "mdp"
        ),
    }
    for name, path in package_paths.items():
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    for leaf in ("task_first_curriculum", "task_first_manifest"):
        monkeypatch.delitem(
            sys.modules,
            f"whole_body_tracking.tasks.tracking.mdp.{leaf}",
            raising=False,
        )


def _module_function(name, namespace):
    node = next(
        node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace[name]


def _runtime_class(method_names, namespace):
    source_class = next(
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommand"
    )
    methods = [
        node
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in set(method_names)
    ]
    assert {node.name for node in methods} == set(method_names)
    class_node = ast.ClassDef(
        name="RuntimeUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            class_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMMAND_PATH), "exec"), namespace)
    return namespace["RuntimeUnderTest"]


def _valid_recipe(**overrides):
    values = {
        "target_mode": "task_first",
        "face_command": True,
        "face_command_pairing": "shared_plus_y",
        "clip_names_per_clip": ("a",),
        "task_first_manifest_path": "cfg/task_first.json",
        "task_first_manifest_sha256": "a" * 64,
        "task_first_base_success_thresh_m": 0.10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_task_first_recipe_requires_no_ball_no_her_no_delay_and_face_command():
    check = _module_function(
        "_assert_task_first_recipe_is_coherent",
        {"math": __import__("math")},
    )
    check(_valid_recipe())
    forbidden = {
        "question_bank": "questions.npz",
        "cq_anchor_bank": "anchor.npz",
        "exam_bank": "exam.npz",
        "achieved_target_mix_prob": 0.1,
        "midswing_resample_prob": 0.1,
        "virtual_ball": True,
        "vb_metrics_only": True,
        "physical_ball": True,
        "shadow_ball": True,
        "shadow_table": True,
        "planner_revision_enabled": True,
        "target_delay_steps": 1,
        "target_delay_tts_mode": "uncompensated",
        "target_jitter_pos_per_s": 0.01,
        "target_jitter_vel_per_s": 0.01,
        "target_noise_white": 0.01,
        "target_noise_ar1_sigma": 0.01,
        "target_dropout_prob": 0.01,
        "target_post_strike_dropout_s": 0.01,
        "target_bias_per_swing": 0.01,
        "face_command": False,
    }
    for key, value in forbidden.items():
        with pytest.raises(ValueError, match="incoherent"):
            check(_valid_recipe(**{key: value}))

    # Every legacy mode returns before reading or changing any setting.
    check(SimpleNamespace(target_mode="uniform", virtual_ball=True))


def test_station_shift_moves_whole_action_and_base_axis_changes_relative_reach():
    station_targets = _module_function("_task_first_station_targets", {})
    origins = np.array([[10.0, 20.0, 0.0]])
    ref_racket = np.array([[0.70, -0.20, 0.95]])
    ref_base = np.array([[0.05, 0.02]])
    farther = np.array([[-0.10, 0.00]])

    # level 0: exact comfortable action, shifted -X as one rigid station.
    racket0, base0 = station_targets(
        origins,
        ref_racket,
        ref_base,
        farther,
        np.zeros((1, 3)),
        np.zeros((1, 2)),
    )
    assert np.allclose(racket0, [[10.60, 19.80, 0.95]])
    assert np.allclose(base0, [[9.95, 20.02]])

    # full position delta translates racket and base together; base delta alone changes reach.
    racket1, base1 = station_targets(
        origins,
        ref_racket,
        ref_base,
        farther,
        np.array([[0.08, -0.04, 0.03]]),
        np.array([[-0.02, 0.05]]),
    )
    assert np.allclose(racket1, [[10.68, 19.76, 0.98]])
    assert np.allclose(base1, [[10.01, 20.03]])
    reach0 = racket0[0, :2] - base0[0]
    reach1_without_base_delta = (
        racket1[0, :2] - (base1[0] - np.array([-0.02, 0.05]))
    )
    assert reach1_without_base_delta.tolist() == pytest.approx(reach0.tolist())


def test_face_cone_zero_is_exact_and_nonzero_is_area_uniform():
    sample = _module_function(
        "_task_first_face_cone_sample",
        {"torch": _FakeTorchCone, "math": __import__("math")},
    )
    center = _Tensor([[0.0, 0.0, 1.0], [0.3, 0.4, 0.5]], dtype=np.float64)
    center = center / _FakeTorchCone.norm(center, dim=-1, keepdim=True)
    exact = sample(center, _Tensor(np.zeros(2)), _Tensor([[0.2, 0.7], [0.9, 0.1]]))
    assert np.allclose(exact, center, atol=1.0e-12)

    count = 20000
    angle = 0.6
    repeated = _Tensor(np.repeat(center[:1], count, axis=0))
    draws = _Tensor(np.random.default_rng(7).random((count, 2)))
    faces = sample(repeated, _Tensor(np.full(count, angle)), draws)
    dots = np.sum(faces * repeated, axis=-1)
    assert float(dots.min()) >= float(np.cos(angle)) - 1.0e-10
    expected_mean = 0.5 * (1.0 + float(np.cos(angle)))
    assert float(dots.mean()) == pytest.approx(expected_mean, abs=0.004)
    assert np.allclose(np.linalg.norm(faces, axis=-1), np.ones(count), atol=1.0e-10)


class _Matrix:
    def __init__(self, values):
        self.values = deepcopy(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return deepcopy(self.values)

    def zero_(self):
        self.values = [[0 for _ in row] for row in self.values]

    def copy_(self, values):
        if hasattr(values, "tolist"):
            values = values.tolist()
        self.values = deepcopy(values)


class _Vector(_Matrix):
    def zero_(self):
        self.values = [0 for _ in self.values]


class _FakeTorchForResume:
    long = "long"
    bool = "bool"

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return deepcopy(values)


def _gate(min_attempts=2):
    return C.GateConfig(
        min_attempts=min_attempts,
        enter_success_lower_bound=0.0,
        exit_success_lower_bound=0.0,
        enter_unsafe_upper_bound=1.0,
        exit_unsafe_upper_bound=1.0,
        enter_dwell_updates=1,
        exit_dwell_updates=1,
        max_stall_updates=100,
        stall_policy="fail",
    )


def _rollout_runtime(action_count, attempts=1):
    names = tuple(f"action_{index:03d}" for index in range(action_count))
    runtime_type = _runtime_class(
        (
            "_task_first_counts_payload",
            "_task_first_on_rollout_end",
            "_task_first_exact_resume_state_dict",
            "_task_first_load_exact_resume_state_dict",
        ),
        {
            "json": json,
            "torch": _FakeTorchForResume,
            "OutcomeCounts": C.OutcomeCounts,
            "_TASK_FIRST_OUTCOME_NAMES": ("attempts", "successes", "unsafe_failures"),
            "_TASK_FIRST_STATE_SCHEMA_VERSION": 2,
            "_TASK_FIRST_STATE_KIND": "whole_body_tracking.RacketTargetCommand.task_first",
        },
    )
    runtime = runtime_type()
    runtime.device = "cpu"
    runtime.num_envs = 4
    runtime._task_first_action_order = names
    runtime._task_first_action_uids = tuple(range(1, action_count + 1))
    runtime._task_first_loaded_manifest = SimpleNamespace(file_sha256="a" * 64)
    runtime._task_first_manifest = SimpleNamespace(gate=_gate())
    runtime._task_first_curriculum = C.TaskFirstCurriculum(
        manifest_sha256="a" * 64,
        action_order=names,
        gate_config=runtime._task_first_manifest.gate,
    )
    runtime._task_first_outcome_type = C.OutcomeCounts
    runtime._task_first_window_counts = _Matrix(
        [
            [attempts] * action_count,
            [attempts] * action_count,
            [0] * action_count,
        ]
    )
    runtime._task_first_attempt_active = _Vector([True, True, False, True])
    runtime._task_first_attempt_action = _Vector([0, 0, -1, min(1, action_count - 1)])
    runtime._task_first_attempt_success = _Vector([True, False, False, True])
    runtime._task_first_recovery_pending = _Vector([False, False, True, False])
    runtime._task_first_recovery_action = _Vector([-1, -1, 0, -1])
    runtime._task_first_recovery_success = _Vector([False, False, True, False])
    runtime._task_first_restored_attempt_discard = _Vector([False] * 4)
    runtime._task_first_restored_recovery_discard = _Vector([False] * 4)
    runtime._task_first_last_rollout_step = None
    return runtime


@pytest.mark.parametrize("action_count", [1, 5, 93])
def test_exact_resume_structure_is_arbitrary_n_and_strict(action_count):
    runtime = _rollout_runtime(action_count)
    state = runtime._task_first_exact_resume_state_dict()
    assert len(state["action_order"]) == action_count
    assert len(state["window_counts"]["attempts"]) == action_count
    assert state["attempt_latches"]["action_slot"][2] == -1
    assert state["pending_recovery"]["action_slot"][2] == 0

    resumed = _rollout_runtime(action_count, attempts=0)
    resumed._task_first_load_exact_resume_state_dict(deepcopy(state), strict=True)
    assert resumed._task_first_exact_resume_state_dict() == state
    assert resumed._task_first_restored_attempt_discard.values == [True, True, False, True]
    assert resumed._task_first_restored_recovery_discard.values == [False, False, True, False]

    broken = deepcopy(state)
    broken["window_counts"]["unsafe_failures"][0] = state["window_counts"]["attempts"][0] + 1
    with pytest.raises(ValueError, match="disjoint"):
        resumed._task_first_load_exact_resume_state_dict(broken, strict=True)
    with pytest.raises(ValueError, match="strict=True"):
        resumed._task_first_load_exact_resume_state_dict(state, strict=False)

    overlapping = deepcopy(state)
    overlapping["attempt_latches"]["active"][2] = True
    overlapping["attempt_latches"]["action_slot"][2] = 0
    with pytest.raises(ValueError, match="cannot own"):
        resumed._task_first_load_exact_resume_state_dict(overlapping, strict=True)


def test_n93_evidence_accumulates_across_rollouts_then_advances_atomically(capsys):
    runtime = _rollout_runtime(93, attempts=1)
    runtime._task_first_on_rollout_end(10)
    pending_line = capsys.readouterr().out.strip()
    assert json.loads(pending_line)["status"] == "pending_evidence"
    assert runtime._task_first_window_counts.values[0] == [1] * 93
    assert all(
        runtime._task_first_curriculum.axis_level(action, "position") == 0.0
        for action in runtime._task_first_action_order
    )

    runtime._task_first_window_counts.values[0] = [2] * 93
    runtime._task_first_window_counts.values[1] = [2] * 93
    runtime._task_first_on_rollout_end(11)
    advanced_line = capsys.readouterr().out.strip()
    receipt = json.loads(advanced_line)
    assert receipt["status"] == "advanced"
    assert len(receipt["decisions"]) == 93
    assert runtime._task_first_window_counts.values == [[0] * 93 for _ in range(3)]
    assert all(
        runtime._task_first_curriculum.axis_level(action, "position") == 0.25
        for action in runtime._task_first_action_order
    )
    with pytest.raises(RuntimeError, match="exactly once"):
        runtime._task_first_on_rollout_end(11)


class _Tensor(np.ndarray):
    def __new__(cls, values, dtype=None):
        return np.asarray(values, dtype=dtype).view(cls)

    def to(self, device=None, dtype=None):
        return _Tensor(self, dtype=dtype)

    def clamp(self, min=None, max=None):
        lo = -np.inf if min is None else min
        hi = np.inf if max is None else max
        return _Tensor(np.clip(self, lo, hi))

    def unsqueeze(self, dim):
        return _Tensor(np.expand_dims(self, axis=dim))

    def add_(self, other):
        self[...] += np.asarray(other)
        return self

    def copy_(self, other):
        self[...] = np.asarray(other)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class _FakeTorchLedger:
    Tensor = _Tensor
    long = np.int64
    bool = np.bool_

    @staticmethod
    def as_tensor(values, dtype=None, device=None):
        return _Tensor(values, dtype=dtype)

    tensor = as_tensor

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return _Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def zeros_like(values):
        return _Tensor(np.zeros_like(values))

    @staticmethod
    def any(values):
        return np.any(values)

    @staticmethod
    def bincount(values, minlength=0):
        return _Tensor(np.bincount(np.asarray(values, dtype=np.int64), minlength=minlength))

    @staticmethod
    def where(values):
        return tuple(_Tensor(row, dtype=np.int64) for row in np.where(values))


class _FakeTorchCone(_FakeTorchLedger):
    @staticmethod
    def norm(values, dim=None, keepdim=False):
        return _Tensor(np.linalg.norm(values, axis=dim, keepdims=keepdim))

    @staticmethod
    def cos(values):
        return _Tensor(np.cos(values))

    @staticmethod
    def sin(values):
        return _Tensor(np.sin(values))

    @staticmethod
    def sqrt(values):
        return _Tensor(np.sqrt(values))

    @staticmethod
    def abs(values):
        return _Tensor(np.abs(values))

    @staticmethod
    def where(condition, left, right):
        return _Tensor(np.where(condition, left, right))

    @staticmethod
    def cross(left, right, dim=-1):
        return _Tensor(np.cross(left, right, axis=dim))


def test_wrap_defers_outcome_and_recovery_reset_charges_prior_action():
    runtime_type = _runtime_class(
        (
            "_task_first_reset_outcome_masks",
            "_task_first_book_outcomes",
            "_task_first_start_current_attempts",
            "_task_first_close_and_start_attempts",
            "_task_first_finish_recovery_holds",
        ),
        {
            "torch": _FakeTorchLedger,
            "_TASK_FIRST_UNSAFE_TERMINATIONS": (
                "base_fell_tilt",
                "base_too_low",
                "robot_hit_table",
            ),
        },
    )
    runtime = runtime_type()
    runtime.device = "cpu"
    runtime._task_first_action_order = tuple(f"a{i}" for i in range(5))
    runtime._task_first_window_counts = _Tensor(np.zeros((3, 5), dtype=np.int64))
    runtime._task_first_attempt_active = _Tensor([True, True, True, True], dtype=np.bool_)
    runtime._task_first_attempt_action = _Tensor([0, 1, 2, 3], dtype=np.int64)
    runtime._task_first_attempt_success = _Tensor([False, True, True, True], dtype=np.bool_)
    runtime._task_first_recovery_pending = _Tensor([False] * 4, dtype=np.bool_)
    runtime._task_first_recovery_action = _Tensor([-1] * 4, dtype=np.int64)
    runtime._task_first_recovery_success = _Tensor([False] * 4, dtype=np.bool_)
    runtime._task_first_restored_attempt_discard = _Tensor([False] * 4, dtype=np.bool_)
    runtime._task_first_restored_recovery_discard = _Tensor(
        [False] * 4, dtype=np.bool_
    )
    new_clip = _Tensor([4, 3, 2, 1], dtype=np.int64)
    runtime._motion = lambda: SimpleNamespace(_multiseg=True, clip_id=new_clip)
    masks = {
        "base_fell_tilt": _Tensor([True, False, False, False], dtype=np.bool_),
        "robot_hit_table": _Tensor([False, True, False, False], dtype=np.bool_),
    }
    runtime._env = SimpleNamespace(
        termination_manager=SimpleNamespace(
            terminated=_Tensor([True, True, False, False], dtype=np.bool_),
            time_outs=_Tensor([False, False, True, False], dtype=np.bool_),
            active_terms=tuple(masks),
            get_term=lambda name: masks[name],
        )
    )
    runtime._selected_bool = lambda value, ids: _Tensor(value)[ids]

    # The wrap itself books nothing.  The previous ownership/success latch moves into a pending
    # recovery transaction, and the newly selected task has not started yet.
    runtime._task_first_close_and_start_attempts(
        _Tensor([0, 1, 2, 3], dtype=np.int64),
        true_reset=False,
    )
    assert runtime._task_first_window_counts.tolist() == [[0] * 5 for _ in range(3)]
    assert runtime._task_first_recovery_action.tolist() == [0, 1, 2, 3]
    assert runtime._task_first_recovery_success.tolist() == [False, True, True, True]
    assert not runtime._task_first_attempt_active.any()

    # A fall/table hit/timeout in that hold closes the pending OLD action.  Only the safe recovery
    # of slot 3 can turn its strike pass into success.  Then the reset-selected clips start.
    runtime._task_first_close_and_start_attempts(
        _Tensor([0, 1, 2, 3], dtype=np.int64),
        true_reset=True,
    )
    assert runtime._task_first_window_counts[0].tolist() == [1, 1, 1, 1, 0]
    assert runtime._task_first_window_counts[1].tolist() == [0, 0, 0, 1, 0]
    assert runtime._task_first_window_counts[2].tolist() == [1, 1, 0, 0, 0]
    assert np.all(
        runtime._task_first_window_counts[1] + runtime._task_first_window_counts[2]
        <= runtime._task_first_window_counts[0]
    )
    # New attempt ownership is the newly selected clip only after old recovery close-out.
    assert runtime._task_first_attempt_action.tolist() == [4, 3, 2, 1]


def test_safe_recovery_expiry_books_success_then_starts_waiting_action():
    runtime_type = _runtime_class(
        (
            "_task_first_book_outcomes",
            "_task_first_start_current_attempts",
            "_task_first_close_and_start_attempts",
            "_task_first_finish_recovery_holds",
        ),
        {
            "torch": _FakeTorchLedger,
            "_TASK_FIRST_UNSAFE_TERMINATIONS": (
                "base_fell_tilt",
                "base_too_low",
                "robot_hit_table",
            ),
        },
    )
    runtime = runtime_type()
    runtime.device = "cpu"
    runtime.num_envs = 4
    runtime._task_first_action_order = tuple(f"a{i}" for i in range(5))
    runtime._task_first_window_counts = _Tensor(np.zeros((3, 5), dtype=np.int64))
    runtime._task_first_attempt_active = _Tensor([True] * 4, dtype=np.bool_)
    runtime._task_first_attempt_action = _Tensor([0, 1, 2, 3], dtype=np.int64)
    runtime._task_first_attempt_success = _Tensor([True, False, True, False], dtype=np.bool_)
    runtime._task_first_recovery_pending = _Tensor([False] * 4, dtype=np.bool_)
    runtime._task_first_recovery_action = _Tensor([-1] * 4, dtype=np.int64)
    runtime._task_first_recovery_success = _Tensor([False] * 4, dtype=np.bool_)
    runtime._task_first_restored_attempt_discard = _Tensor([False] * 4, dtype=np.bool_)
    runtime._task_first_restored_recovery_discard = _Tensor(
        [False] * 4, dtype=np.bool_
    )
    new_clip = _Tensor([4, 3, 2, 1], dtype=np.int64)
    runtime._motion = lambda: SimpleNamespace(_multiseg=True, clip_id=new_clip)

    runtime._task_first_attempt_success[:] = [True, False, True, False]
    runtime._task_first_close_and_start_attempts(
        _Tensor([0, 1, 2, 3], dtype=np.int64),
        true_reset=False,
    )
    runtime._task_first_finish_recovery_holds(
        _Tensor([False, False, False, False], dtype=np.bool_)
    )
    assert runtime._task_first_window_counts[0].tolist() == [1, 1, 1, 1, 0]
    assert runtime._task_first_window_counts[1].tolist() == [1, 0, 1, 0, 0]
    assert runtime._task_first_window_counts[2].tolist() == [0, 0, 0, 0, 0]
    assert runtime._task_first_attempt_action.tolist() == [4, 3, 2, 1]
    assert not runtime._task_first_recovery_pending.any()


def test_resume_mid_recovery_discards_transport_reset_without_evidence():
    runtime_type = _runtime_class(
        (
            "_task_first_exact_resume_state_dict",
            "_task_first_load_exact_resume_state_dict",
            "_task_first_reset_outcome_masks",
            "_task_first_book_outcomes",
            "_task_first_start_current_attempts",
            "_task_first_close_and_start_attempts",
        ),
        {
            "torch": _FakeTorchLedger,
            "_TASK_FIRST_STATE_SCHEMA_VERSION": 2,
            "_TASK_FIRST_STATE_KIND": "whole_body_tracking.RacketTargetCommand.task_first",
            "_TASK_FIRST_OUTCOME_NAMES": ("attempts", "successes", "unsafe_failures"),
            "_TASK_FIRST_UNSAFE_TERMINATIONS": (
                "base_fell_tilt",
                "base_too_low",
                "robot_hit_table",
            ),
        },
    )

    def make_runtime(*, pending):
        runtime = runtime_type()
        runtime.device = "cpu"
        runtime.num_envs = 1
        runtime._task_first_action_order = ("fh",)
        runtime._task_first_action_uids = (101,)
        runtime._task_first_loaded_manifest = SimpleNamespace(file_sha256="a" * 64)
        runtime._task_first_curriculum = C.TaskFirstCurriculum(
            manifest_sha256="a" * 64,
            action_order=("fh",),
            gate_config=_gate(),
        )
        runtime._task_first_window_counts = _Tensor(
            np.zeros((3, 1), dtype=np.int64)
        )
        runtime._task_first_attempt_active = _Tensor([False], dtype=np.bool_)
        runtime._task_first_attempt_action = _Tensor([-1], dtype=np.int64)
        runtime._task_first_attempt_success = _Tensor([False], dtype=np.bool_)
        runtime._task_first_recovery_pending = _Tensor([pending], dtype=np.bool_)
        runtime._task_first_recovery_action = _Tensor(
            [0 if pending else -1], dtype=np.int64
        )
        runtime._task_first_recovery_success = _Tensor(
            [pending], dtype=np.bool_
        )
        runtime._task_first_restored_attempt_discard = _Tensor(
            [False], dtype=np.bool_
        )
        runtime._task_first_restored_recovery_discard = _Tensor(
            [False], dtype=np.bool_
        )
        runtime._task_first_last_rollout_step = None
        runtime._motion = lambda: SimpleNamespace(
            _multiseg=True, clip_id=_Tensor([0], dtype=np.int64)
        )
        manager = SimpleNamespace(
            terminated=_Tensor([True], dtype=np.bool_),
            time_outs=_Tensor([False], dtype=np.bool_),
            active_terms=("robot_hit_table",),
            get_term=lambda name: _Tensor([True], dtype=np.bool_),
        )
        runtime._env = SimpleNamespace(termination_manager=manager)
        runtime._selected_bool = lambda value, ids: _Tensor(value)[ids]
        return runtime

    checkpoint = make_runtime(pending=True)._task_first_exact_resume_state_dict()
    resumed = make_runtime(pending=False)
    resumed._task_first_load_exact_resume_state_dict(checkpoint, strict=True)
    assert resumed._task_first_restored_recovery_discard.tolist() == [True]

    # The runner restores command state and then resets the non-serialized simulator.  Even if
    # that transport reset exposes a table-hit termination bit, the saved in-flight recovery is
    # discarded exactly once and cannot pollute the evidence window.
    resumed._task_first_close_and_start_attempts(
        _Tensor([0], dtype=np.int64), true_reset=True
    )
    assert resumed._task_first_window_counts.tolist() == [[0], [0], [0]]
    assert resumed._task_first_attempt_active.tolist() == [True]
    assert resumed._task_first_recovery_pending.tolist() == [False]


def test_runtime_source_pins_hard_contract_positive_speed_and_legacy_resume():
    assert "def task_first_hard_contract" in SOURCE
    assert "speed_delta_mps) >= speed - 1.0e-6" in SOURCE
    assert "motion_global_anchor_pos=None" in SOURCE
    assert "target_mode in (\"hitter_pure\", \"task_first\")" in SOURCE
    assert "self.target_normal_cmd[ids] = face" in SOURCE
    assert "self.racket_target_normal_w[ids] = face" in SOURCE
    assert "self._task_first_attempt_action[ids]" in SOURCE
    assert "self._task_first_window_counts.zero_()" in SOURCE
    assert "self.exact_resume_state_dict = self._task_first_exact_resume_state_dict" in SOURCE
    # Public callable hooks are dynamically installed only inside the task-first branch.  Legacy
    # targets therefore remain under the runner's historical heuristic resume scanner.
    method_names = {
        node.name
        for node in ast.walk(TREE)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "exact_resume_state_dict" not in method_names
    assert "load_exact_resume_state_dict" not in method_names

    cfg = next(
        node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "RacketTargetCommandCfg"
    )
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in cfg.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
        and node.target.id.startswith("task_first_")
    }
    assert defaults == {
        "task_first_manifest_path": "",
        "task_first_manifest_sha256": "",
        "task_first_base_success_thresh_m": 0.10,
    }


def test_runtime_hard_contract_exactly_equals_train_builder(monkeypatch):
    _install_dependency_light_task_first_package(monkeypatch)
    train_path = ROOT / "scripts" / "train.py"
    train_tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    train_node = next(
        node
        for node in train_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_task_first_manifest_contract"
    )
    train_namespace = {
        "_TASK_FIRST_LOADED_MANIFEST_ATTR": "_hope_task_first_loaded_manifest_v1",
        "_load_task_first_manifest_from_racket_cfg": lambda cfg: (_ for _ in ()).throw(
            AssertionError("preloaded manifest expected")
        ),
    }
    train_module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            train_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(train_module)
    exec(compile(train_module, str(train_path), "exec"), train_namespace)

    runtime_type = _runtime_class(("task_first_hard_contract",), {})
    runtime = runtime_type()
    actions = (
        SimpleNamespace(
            action_id="fh",
            action_uid=101,
            position_half_extent_m=(0.1, 0.2, 0.3),
            speed_delta_mps=0.4,
            face_cone_deg=12.0,
            station_center_shift_xy_m=(-0.1, 0.0),
            base_half_extent_xy_m=(0.05, 0.06),
        ),
        SimpleNamespace(
            action_id="bh",
            action_uid=202,
            position_half_extent_m=(0.2, 0.1, 0.15),
            speed_delta_mps=0.3,
            face_cone_deg=9.0,
            station_center_shift_xy_m=(0.0, 0.02),
            base_half_extent_xy_m=(0.04, 0.08),
        ),
    )
    gate = _gate(min_attempts=20)
    manifest = SimpleNamespace(
        manifest_id="manifest-v1",
        action_order=("fh", "bh"),
        actions=actions,
        gate=gate,
    )
    loaded = SimpleNamespace(
        source_path=Path("/review/task_first_manifest.json"),
        file_sha256="a" * 64,
        canonical_sha256="b" * 64,
        manifest=manifest,
    )
    motion_cfg = SimpleNamespace(
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=73,
        clip_switch_prob=0.0,
        speed_scale_range=(1.0, 1.0),
        event_timing_mode="disabled",
    )
    racket_cfg = SimpleNamespace(
        task_first_base_success_thresh_m=0.08,
        strike_success_pos_thresh=0.075,
        strike_success_vel_thresh=0.5,
        strike_success_normal_thresh_deg=15.0,
        clean_reference_strike_velocity=True,
        clean_strike_vel_window=2,
        wrist_body_name="right_wrist_yaw_Link",
        mount_offset=(0.0, 0.13, 0.0),
        mount_quat=(1.0, 0.0, 0.0, 0.0),
        mount_normal_axis=1,
        face_command_pairing="shared_plus_y",
    )
    env_cfg = SimpleNamespace(
        table_obstacle=True,
        table_obstacle_prim="{ENV_REGEX_NS}/TableObstacle",
        terminations=SimpleNamespace(
            robot_hit_table=SimpleNamespace(
                func="whole_body_tracking.tasks.tracking.mdp.robot_hit_table",
                params={
                    "filtered_sensor_cfg": SimpleNamespace(
                        name="racket_table_contact"
                    ),
                    "near_x": 0.78,
                    "surface_z": 0.76,
                    "force_threshold": 1.0,
                    "margin": 0.02,
                },
            )
        ),
    )
    runtime._task_first_enabled = True
    runtime._task_first_loaded_manifest = loaded
    runtime._task_first_action_uids = (101, 202)
    runtime.cfg = racket_cfg
    runtime._motion = lambda: SimpleNamespace(cfg=motion_cfg)
    runtime._env = SimpleNamespace(cfg=env_cfg)
    runtime_contract = runtime.task_first_hard_contract()
    train_namespace["_load_task_first_manifest_from_racket_cfg"] = lambda cfg: loaded
    train_contract = train_namespace["_task_first_manifest_contract"](
        racket_cfg, motion_cfg, env_cfg
    )
    assert runtime_contract == train_contract
