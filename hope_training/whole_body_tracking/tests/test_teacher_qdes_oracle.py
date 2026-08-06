"""Dependency-light contract tests for the Take061 teacher-q_des oracle."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts/train.py"


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def train(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "hydra", _module("hydra", main=lambda **kwargs: lambda fn: fn)
    )

    class FakeOmegaConf:
        @staticmethod
        def resolve(cfg):
            return None

        @staticmethod
        def set_struct(cfg, value):
            return None

    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        _module(
            "omegaconf",
            ListConfig=type("ListConfig", (list,), {}),
            OmegaConf=FakeOmegaConf,
        ),
    )
    spec = importlib.util.spec_from_file_location("train_teacher_oracle_test", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _resolve(train, cfg, **overrides):
    flags = dict(
        action_ball=True,
        diagnostic=True,
        dynamic_ready=True,
        shared_ready=False,
        num_envs=1,
        max_iterations=0,
    )
    flags.update(overrides)
    return train._resolve_teacher_qdes_oracle_request(cfg, **flags)


def test_request_is_noop_when_fields_are_absent(train):
    assert _resolve(train, {}) == (None, None)


def test_request_requires_fresh_dynamic_ready_single_env(train, tmp_path):
    output = tmp_path / "oracle.json"
    cfg = {
        "action_ball_teacher_qdes_oracle_output_path": str(output),
        "action_ball_teacher_qdes_oracle_episodes": 32,
        "checkpoint_path": None,
    }
    assert _resolve(train, cfg) == (str(output), 32)
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fresh path"):
        _resolve(train, cfg)
    output.unlink()
    with pytest.raises(RuntimeError, match="dynamic-ready only"):
        _resolve(train, cfg, shared_ready=True)
    with pytest.raises(RuntimeError, match="num_envs=1"):
        _resolve(train, cfg, num_envs=2)


def test_publish_is_canonical_no_clobber_and_failure_atomic(train, tmp_path, monkeypatch):
    output = tmp_path / "oracle.json"
    document = {"z": 1, "a": {"diagnostic_unauthorized": True}}
    receipt = train._publish_teacher_qdes_oracle(str(output), document)
    encoded = output.read_bytes()
    assert encoded == b'{"a":{"diagnostic_unauthorized":true},"z":1}\n'
    assert receipt["path"] == str(output)
    assert len(receipt["sha256"]) == 64
    with pytest.raises(RuntimeError, match="not fresh"):
        train._publish_teacher_qdes_oracle(str(output), document)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["oracle.json"]

    failed = tmp_path / "failed.json"
    monkeypatch.setattr(train.os, "fsync", lambda _descriptor: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(OSError, match="fsync"):
        train._publish_teacher_qdes_oracle(str(failed), document)
    assert not failed.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["oracle.json"]


def test_run_resets_once_then_preserves_32_episode_auto_reset_and_counters(
    train, monkeypatch
):
    class Tensor:
        def __init__(self, value):
            self.data = np.asarray(value)

        @property
        def shape(self):
            return self.data.shape

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.data.copy())

        def cpu(self):
            return self

        def tolist(self):
            return self.data.tolist()

        def item(self):
            return self.data.item()

        def abs(self):
            return Tensor(np.abs(self.data))

        def max(self):
            return Tensor(np.max(self.data))

        def sum(self):
            return Tensor(np.sum(self.data))

        @property
        def dtype(self):
            return self.data.dtype

        @property
        def device(self):
            return "fake"

        def ne(self, value):
            return Tensor(self.data != value)

        def __getitem__(self, key):
            return Tensor(self.data[key])

        def to(self, *, dtype):
            return Tensor(self.data.astype(dtype))

        def __sub__(self, other):
            return Tensor(self.data - other.data)

        def __add__(self, other):
            return Tensor(
                self.data + (other.data if isinstance(other, Tensor) else other)
            )

        def __truediv__(self, other):
            return Tensor(self.data / other.data)

        def __and__(self, other):
            return Tensor(self.data & other.data)

        def __or__(self, other):
            return Tensor(self.data | other.data)

        def __invert__(self):
            return Tensor(~self.data)

        def __gt__(self, other):
            return Tensor(self.data > other)

        def __lt__(self, other):
            return Tensor(self.data < other)

    fake_torch = _module(
        "torch",
        long=np.dtype(np.int64),
        any=lambda value: Tensor(np.any(value.data)),
        all=lambda value: Tensor(np.all(value.data)),
        isfinite=lambda value: Tensor(np.isfinite(value.data)),
        is_tensor=lambda value: isinstance(value, Tensor),
        bool=np.dtype(bool),
        where=lambda condition, first, second: Tensor(
            np.where(condition.data, first.data, second.data)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    action = SimpleNamespace(
        _scale=Tensor(np.ones(31)),
        _offset=Tensor(np.zeros(31)),
        _pre_clamp_qdes=Tensor(np.zeros((1, 31))),
        _resolved_table_contact_params=lambda: {
            "full_table_assembly": True,
            "attribution_diagnostic": True,
        },
    )
    # The split-ready WAIT reference IS the birth pose; the executable command for
    # those rows is the contract hold q_des.  The fake reproduces both channels so
    # the oracle's substitution is exercised, not stubbed away.
    hold_qdes = Tensor(np.full((1, 31), 0.5))
    wait_rows = {"value": True}
    # Post-reveal the teacher stays frozen at frame 0 for this many more steps; the
    # driver has to spread the split-ready -> frame 0 travel across them.  Zero is the
    # "no window published" case, where a revealed row lands on the reference at once.
    frozen_rows = {"value": 0}
    motion = SimpleNamespace(
        joint_pos=Tensor(np.full((1, 31), 0.25)),
        action_ball_diagnostic_split_ready_teacher=True,
        action_ball_split_ready_hold_command=lambda: (
            Tensor(np.array([wait_rows["value"]])),
            hold_qdes,
        ),
        action_ball_teacher_start_frozen_steps=lambda: Tensor(
            np.array([frozen_rows["value"]], dtype=np.int64)
        ),
    )
    racket = SimpleNamespace(
        cfg=SimpleNamespace(
            reference_guard_mode="metrics_only",
            strike_success_pos_thresh=0.075,
            strike_success_vel_thresh=0.5,
            strike_success_normal_thresh_deg=15.0,
        ),
        metrics={
            "exact_strike_hit_rate": Tensor([0.0]),
            "racket_pos_error_exact_strike": Tensor([0.0]),
            "racket_vel_error_exact_strike": Tensor([0.0]),
            "racket_normal_error_deg_exact_strike": Tensor([0.0]),
        },
        vb_fired=Tensor([False]),
    )
    racket.table_context = None
    racket.configure_table_guard_oracle_first_hit_export = lambda: None
    racket.set_table_guard_oracle_first_hit_context = lambda **context: setattr(
        racket, "table_context", context
    )
    racket.consume_table_guard_oracle_first_hit_rows = lambda: []
    counters = {
        "strike_opportunity_count": Tensor(1),
        "virtual_capture_count": Tensor(1),
        **{key: Tensor(0) for key in train._TEACHER_ORACLE_REJECT_KEYS},
    }
    racket.consume_exact_behavior_decision_counters = lambda: counters

    monkeypatch.setattr(
        train,
        "_consume_teacher_oracle_limit_exposure",
        lambda _base: {
            "projection": {
                "observed_sample_count": 64,
                "projected_sample_count": 0,
                "nonfinite_sample_count": 0,
                "hypothetical_unweighted_penalty_sum": 0.0,
                "max_normalized_projection_distance": 0.0,
                "mean_normalized_projection_distance": 0.0,
                "joints": [],
            },
            "soft_limit": {
                "qdes": {"observed_sample_count": 64},
                "actual": {"observed_sample_count": 64},
            },
        },
    )

    class Terminations:
        active_terms = ("action_ball_single_stroke_complete",)

        def __init__(self):
            self.complete = False

        def get_term(self, name):
            assert name == "action_ball_single_stroke_complete"
            return Tensor([self.complete])

    tm = Terminations()

    class Base:
        num_envs = 1
        max_episode_length = 2

        def __init__(self):
            self.reset_calls = 0
            self.command_manager = SimpleNamespace(
                get_term=lambda name: motion if name == "motion" else racket
            )
            self.action_manager = SimpleNamespace(get_term=lambda _name: action)
            self.termination_manager = tm

        def _reset_idx(self, env_ids):
            assert env_ids.tolist() == [0]
            self.reset_calls += 1
            action._pre_clamp_qdes = Tensor(np.zeros((1, 31)))
            racket.metrics["exact_strike_hit_rate"] = Tensor([0.0])
            racket.vb_fired = Tensor([False])
            tm.complete = False

    base = Base()

    class ResetNeeded(RuntimeError):
        """Dependency-light stand-in for gymnasium.error.ResetNeeded."""

    class Env:
        unwrapped = base
        step_count = 0

        def __init__(self):
            self.initial_reset_calls = 0
            self.reset_needed = True

        def reset(self):
            self.initial_reset_calls += 1
            base._reset_idx(Tensor([0]))
            self.reset_needed = False
            return object(), {}

        def step(self, raw):
            if self.reset_needed:
                raise ResetNeeded(
                    "Cannot call env.step() before calling env.reset()"
                )
            self.step_count += 1
            # Exercise both command channels: the first four control steps are the
            # pre-task WAIT (hold q_des), everything after is the teacher reference.
            wait_rows["value"] = self.step_count < 4
            action._pre_clamp_qdes = Tensor(raw.data.copy())
            if self.step_count % 2:
                # Real ordering: command.compute publishes this step's metrics only
                # after a nonterminal step. Episode one strikes; episode two does not.
                exact = self.step_count == 1
                racket.metrics.update(
                    exact_strike_hit_rate=Tensor([float(exact)]),
                    racket_pos_error_exact_strike=Tensor([0.1]),
                    racket_vel_error_exact_strike=Tensor([0.2]),
                    racket_normal_error_deg_exact_strike=Tensor([3.0]),
                )
                racket.vb_fired = Tensor([exact])
                return None, None, Tensor([False]), Tensor([False]), {}
            tm.complete = True
            base._reset_idx(Tensor([0]))
            # command.compute now belongs to the reset/new episode; it must not
            # be attributed to the just-finished terminal step.
            racket.metrics["exact_strike_hit_rate"] = Tensor([0.0])
            return None, None, Tensor([True]), Tensor([False]), {}

    sha = "a" * 64
    hard_contract = {
        "motion_clips": [{"sha256": sha}],
        "action_ball_ppo_runner_recipe": {"sha256": sha},
        "action_ball_training": {
            "effective_reward_recipe_sha256": sha,
            "preflight": {
                "action_order": ["take_061_unit04_bh"],
                "policy_contract_sha256": sha,
                "manifest": {"file_sha256": sha},
            },
            # S0 把 teacher-q_des oracle 的准入运行时从"不可变磁带"换成了
            # "在线求解器 + 精确问题答案复用": 决定性不再靠一条录好的磁带,
            # 而是靠 ExactCurriculumQuestionCache 对同一道题必然给同一个答案。
            # 门槛没有放松, 只是换了等价的决定性来源, 而且额外要求磁带确实
            # 不在场(immutable_tape is None)、目标观测噪声关死。
            "runtime": {"target_provider": {
                "source": "online_solver", "recipe": "current_lm",
                "validity_mask": [True, True, True],
                "target_observation_noise": False,
                "immutable_tape": None,
                "exact_question_answer_reuse": {"enabled": True},
            }, "reference_guard": {"contract_payload": {
                "counter_schema_sha256": sha,
                "counter_names": [
                    "reference_guard_sample_count",
                    "reference_guard_union_count",
                    "reference_guard_reference_only_count",
                    "reference_guard_reference_and_hard_count",
                ],
            }}},
            "policy_bootstrap": {"ready_source": {"identity": {
                "binding_sha256": sha,
                "rows": [{
                    "artifact": {"sha256": sha},
                    "nominal_hold_receipt": {"sha256": sha},
                }],
            }}},
        },
    }
    class BadResetEnv:
        unwrapped = base

        def reset(self):
            return None

        def step(self, _raw):
            raise AssertionError("malformed reset output must fail before step")

    with pytest.raises(RuntimeError, match="initial reset must return"):
        train._run_teacher_qdes_oracle(
            BadResetEnv(),
            cfg=SimpleNamespace(task=SimpleNamespace(name="TrackingFlat")),
            hard_contract=hard_contract,
            hard_contract_sha256=sha,
            episodes=32,
        )

    unreset = Env()
    with pytest.raises(ResetNeeded, match="before calling env.reset"):
        unreset.step(Tensor(np.zeros((1, 31))))
    assert unreset.initial_reset_calls == 0

    env = Env()
    result = train._run_teacher_qdes_oracle(
        env,
        cfg=SimpleNamespace(task=SimpleNamespace(name="TrackingFlat")),
        hard_contract=hard_contract,
        hard_contract_sha256=sha,
        episodes=32,
    )

    assert env.initial_reset_calls == 1
    assert base.reset_calls == 33
    assert "_reset_idx" not in vars(base)
    assert result["completion"] == {
        "requested": 32,
        "terminal": 32,
        "single_stroke": 32,
        "exact_strike_observed_nonterminal": 1,
        "pre_strike_or_same_step_unknown": 31,
        "control_steps": 64,
    }
    assert result["completion"]["exact_strike_observed_nonterminal"] == 1
    assert result["phase_by_termination"]["post_strike"]["action_ball_single_stroke_complete"] == 1
    assert result["phase_by_termination"]["pre_strike_or_same_step_unknown"]["action_ball_single_stroke_complete"] == 31
    assert result["exact_strike"]["position"]["values"] == [0.1]
    assert result["capture_rejection"] == {
        "opportunities": 1,
        "captures": 1,
        "rejects": {key: 0 for key in train._TEACHER_ORACLE_REJECT_KEYS},
        "conserved": True,
    }
    assert result["teacher_qdes"]["preclamp_max_abs_error_rad"] == 0.0
    # WAIT rows were driven by the hold q_des (0.5), revealed rows by the teacher
    # reference (0.25).  This fake publishes no post-reveal frozen window, so no row
    # is a bridge row and the receipt says so rather than implying one happened; the
    # three counts must still partition every control step.
    assert result["teacher_qdes"]["wait_hold_command_steps"] == 4
    assert result["teacher_qdes"]["bridge_ramp_command_steps"] == 0
    assert result["teacher_qdes"]["teacher_reference_command_steps"] == 60
    assert (
        result["teacher_qdes"]["wait_hold_command_steps"]
        + result["teacher_qdes"]["bridge_ramp_command_steps"]
        + result["teacher_qdes"]["teacher_reference_command_steps"]
        == result["completion"]["control_steps"]
    )
    # The reveal step this instrument did NOT send: hold q_des 0.5 -> reference 0.25.
    assert result["teacher_qdes"]["reveal_reference_step_max_abs_rad"] == 0.25
    assert result["teacher_qdes"]["raw_action_max_abs"] == 0.5
    assert result["bindings"]["hard_contract_sha256"] == sha
    assert result["schema_version"] == 2
    assert result["kind"] == "action_ball_teacher_qdes_dynamic_oracle_v2"
    assert all(row["table_first_hit"] is None for row in result["episodes"])
    assert result["measurement_contract"]["exact_strike_thresholds"] == {
        "position_error_m_strict_lt": 0.075,
        "velocity_error_mps_strict_lt": 0.5,
        "face_error_deg_strict_lt": 15.0,
    }
    assert result["safety_exposure"]["projection"]["observed_sample_count"] == 64
    assert result["safety_exposure"]["reference_guard"] == {
        "mode": "metrics_only",
        "available": False,
        "counter_schema_sha256": sha,
        "counter_names": [
            "reference_guard_sample_count",
            "reference_guard_union_count",
            "reference_guard_reference_only_count",
            "reference_guard_reference_and_hard_count",
        ],
        "counters": None,
        "sample_count": None,
        "union_count": None,
        "reference_only_count": None,
        "reference_and_hard_count": None,
    }


def test_oracle_steps_teacher_action_without_teleport_or_ppo():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    oracle = source.split("def _run_teacher_qdes_oracle", 1)[1].split(
        "def _publish_teacher_qdes_oracle", 1
    )[0]
    assert "raw = (command - offset) / scale" in oracle
    # The WAIT-phase reference is the birth pose itself; sending it as q_des is a
    # zero-torque command.  The oracle must route those rows through the contract
    # hold q_des instead, and must not silently fall back to the reference.
    assert "_teacher_qdes_oracle_control_command(" in oracle
    assert "env.step(raw)" in oracle
    assert "set_table_context(" in oracle
    assert "consume_table_rows()" in oracle
    assert 'action, "_resolved_table_contact_params"' in oracle
    assert "table_guard_params = prepare_table_guard()" in oracle
    assert '"table_first_hit"' in oracle
    assert oracle.count("initial_reset = env.reset()") == 1
    assert oracle.index("initial_reset = env.reset()") < oracle.index(
        "_install_teacher_qdes_prereset_capture"
    )
    assert oracle.index("initial_reset = env.reset()") < oracle.index("env.step(raw)")
    assert oracle.index("initial_reset = env.reset()") < oracle.index(
        "table_guard_params = prepare_table_guard()"
    )
    assert oracle.index("table_guard_params = prepare_table_guard()") < oracle.index(
        "configure_table_export()"
    )
    assert oracle.index("configure_table_export()") < oracle.index("env.step(raw)")
    for forbidden in ("write_joint_state_to_sim", "write_root_state_to_sim", "OnPolicyRunner"):
        assert forbidden not in oracle

    run = source.split("def _run(cfg):", 1)[1]
    assert run.index("env = gym.make") < run.index("oracle = _run_teacher_qdes_oracle")
    assert run.index("oracle = _run_teacher_qdes_oracle") < run.index(
        "runner = OnPolicyRunner"
    )
    oracle_enable = run.index("teacher-q_des oracle enabled")
    assert oracle_enable < run.index("env = gym.make")
    assert "env_cfg.table_contact_attribution_diagnostic = True" in run


def _ramp_fakes(monkeypatch):
    """Minimal torch/motion doubles for the reveal-bridge command law."""

    class T:
        def __init__(self, value):
            self.data = np.asarray(value)

        @property
        def shape(self):
            return self.data.shape

        @property
        def dtype(self):
            return self.data.dtype

        @property
        def device(self):
            return "fake"

        def sum(self):
            return T(np.sum(self.data))

        def item(self):
            return self.data.item()

        def to(self, *, dtype):
            return T(self.data.astype(dtype))

        def __getitem__(self, key):
            return T(self.data[key])

        def __add__(self, other):
            return T(self.data + (other.data if isinstance(other, T) else other))

        def __sub__(self, other):
            return T(self.data - other.data)

        def __truediv__(self, other):
            return T(self.data / other.data)

        def __and__(self, other):
            return T(self.data & other.data)

        def __invert__(self):
            return T(~self.data)

        def __gt__(self, other):
            return T(self.data > other)

        def __lt__(self, other):
            return T(self.data < other)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        _module(
            "torch",
            long=np.dtype(np.int64),
            bool=np.dtype(bool),
            any=lambda value: T(np.any(value.data)),
            all=lambda value: T(np.all(value.data)),
            isfinite=lambda value: T(np.isfinite(value.data)),
            is_tensor=lambda value: isinstance(value, T),
            where=lambda condition, first, second: T(
                np.where(condition.data, first.data, second.data)
            ),
        ),
    )
    return T


def _ramp_motion(T, *, wait, frozen, hold):
    return SimpleNamespace(
        action_ball_split_ready_hold_command=lambda: (
            T(np.array([wait])),
            T(np.full((1, 3), hold)),
        ),
        action_ball_teacher_start_frozen_steps=lambda: T(
            np.array([frozen], dtype=np.int64)
        ),
    )


def test_wait_rows_command_the_hold_qdes_not_the_reference(train, monkeypatch):
    T = _ramp_fakes(monkeypatch)
    reference = T(np.full((1, 3), 0.25))
    motion = _ramp_motion(T, wait=True, frozen=7, hold=0.5)

    command, wait_rows, bridge_rows = train._teacher_qdes_oracle_control_command(
        motion, reference, None
    )

    # The WAIT reference IS the birth pose; sending it commands zero torque.  A WAIT
    # row is never a bridge row no matter what the frozen clock says.
    assert np.allclose(command.data, 0.5)
    assert (wait_rows, bridge_rows) == (1, 0)


def test_reveal_ramp_lands_exactly_on_the_reference_when_the_window_closes(
    train, monkeypatch
):
    T = _ramp_fakes(monkeypatch)
    reference = T(np.full((1, 3), 2.0))
    previous = None
    sent = []
    # 4 frozen steps left, counting down to 0 -- five revealed steps in the window.
    for frozen in (4, 3, 2, 1, 0):
        motion = _ramp_motion(T, wait=False, frozen=frozen, hold=0.0)
        previous, _wait_rows, bridge_rows = train._teacher_qdes_oracle_control_command(
            motion, reference, previous
        )
        sent.append(float(previous.data[0, 0]))
        assert bridge_rows == (1 if frozen > 0 else 0)

    # Equal increments -- a straight line from the hold q_des onto measured frame 0,
    # with no step anywhere and no overshoot past it.
    assert sent == pytest.approx([0.4, 0.8, 1.2, 1.6, 2.0])
    # It lands on the reference EXACTLY as the clip starts advancing, so every later
    # step is the raw teacher reference with no residue and no mode switch.
    after = train._teacher_qdes_oracle_control_command(
        _ramp_motion(T, wait=False, frozen=0, hold=0.0), reference, previous
    )
    assert after[0].data == pytest.approx(reference.data)
    assert after[2] == 0


def test_unpublished_window_degenerates_to_one_step_and_reports_zero_bridge_rows(
    train, monkeypatch
):
    T = _ramp_fakes(monkeypatch)
    reference = T(np.full((1, 3), 2.0))
    motion = _ramp_motion(T, wait=False, frozen=0, hold=0.0)

    command, _wait_rows, bridge_rows = train._teacher_qdes_oracle_control_command(
        motion, reference, T(np.zeros((1, 3)))
    )

    # No window published: the driver cannot invent one, so it does what it always
    # did -- but the receipt must not be able to claim a bridge happened.
    assert command.data == pytest.approx(reference.data)
    assert bridge_rows == 0


def test_driver_refuses_a_frozen_clock_it_cannot_trust(train, monkeypatch):
    T = _ramp_fakes(monkeypatch)
    reference = T(np.full((1, 3), 2.0))

    negative = _ramp_motion(T, wait=False, frozen=-1, hold=0.0)
    with pytest.raises(RuntimeError, match="non-negative long row count"):
        train._teacher_qdes_oracle_control_command(negative, reference, None)

    floaty = _ramp_motion(T, wait=False, frozen=3, hold=0.0)
    floaty.action_ball_teacher_start_frozen_steps = lambda: T(np.array([3.0]))
    with pytest.raises(RuntimeError, match="non-negative long row count"):
        train._teacher_qdes_oracle_control_command(floaty, reference, None)

    missing = _ramp_motion(T, wait=False, frozen=3, hold=0.0)
    del missing.action_ball_teacher_start_frozen_steps
    with pytest.raises(RuntimeError, match="teacher-start frozen-step accessor"):
        train._teacher_qdes_oracle_control_command(missing, reference, None)
