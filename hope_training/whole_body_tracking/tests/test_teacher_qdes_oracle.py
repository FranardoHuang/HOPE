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


def test_run_captures_terminal_exact_and_preclamp_before_one_real_reset(train, monkeypatch):
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

        def ne(self, value):
            return Tensor(self.data != value)

        def __getitem__(self, key):
            return Tensor(self.data[key])

        def __sub__(self, other):
            return Tensor(self.data - other.data)

        def __truediv__(self, other):
            return Tensor(self.data / other.data)

        def __and__(self, other):
            return Tensor(self.data & other.data)

        def __or__(self, other):
            return Tensor(self.data | other.data)

        def __gt__(self, other):
            return Tensor(self.data > other)

    fake_torch = _module(
        "torch",
        all=lambda value: Tensor(np.all(value.data)),
        isfinite=lambda value: Tensor(np.isfinite(value.data)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    action = SimpleNamespace(
        _scale=Tensor(np.ones(31)),
        _offset=Tensor(np.zeros(31)),
        _pre_clamp_qdes=Tensor(np.zeros((1, 31))),
    )
    motion = SimpleNamespace(
        joint_pos=Tensor(np.full((1, 31), 0.25)),
        action_ball_diagnostic_split_ready_teacher=True,
    )
    racket = SimpleNamespace(
        metrics={
            "exact_strike_hit_rate": Tensor([0.0]),
            "racket_pos_error_exact_strike": Tensor([0.0]),
            "racket_vel_error_exact_strike": Tensor([0.0]),
            "racket_normal_error_deg_exact_strike": Tensor([0.0]),
        },
        vb_fired=Tensor([False]),
    )
    counters = {
        "strike_opportunity_count": Tensor(1),
        "virtual_capture_count": Tensor(1),
        **{key: Tensor(0) for key in train._TEACHER_ORACLE_REJECT_KEYS},
    }
    racket.consume_exact_behavior_decision_counters = lambda: counters

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

    class Env:
        unwrapped = base
        step_count = 0

        def step(self, raw):
            self.step_count += 1
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
            "runtime": {"target_provider": {
                "source": "immutable_tape", "recipe": "current_lm",
                "validity_mask": [True, True, True],
                "immutable_tape": {
                    "online_lm_calls": 0, "physical_rng_draws": 0,
                    "file_sha256": sha, "canonical_sha256": sha,
                    "base_question_sha256": sha,
                    "target_lineage": {
                        "target_producer_sha256": sha,
                        "target_column_sha256": sha,
                    },
                },
            }},
            "policy_bootstrap": {"ready_source": {"identity": {
                "binding_sha256": sha,
                "rows": [{
                    "artifact": {"sha256": sha},
                    "nominal_hold_receipt": {"sha256": sha},
                }],
            }}},
        },
    }
    result = train._run_teacher_qdes_oracle(
        Env(),
        cfg=SimpleNamespace(task=SimpleNamespace(name="TrackingFlat")),
        hard_contract=hard_contract,
        hard_contract_sha256=sha,
        episodes=2,
    )

    assert base.reset_calls == 2
    assert "_reset_idx" not in vars(base)
    assert result["completion"]["terminal"] == 2
    assert result["completion"]["exact_strike_observed_nonterminal"] == 1
    assert result["completion"]["pre_strike_or_same_step_unknown"] == 1
    assert result["phase_by_termination"]["post_strike"]["action_ball_single_stroke_complete"] == 1
    assert result["phase_by_termination"]["pre_strike_or_same_step_unknown"]["action_ball_single_stroke_complete"] == 1
    assert result["exact_strike"]["position"]["values"] == [0.1]
    assert result["teacher_qdes"]["preclamp_max_abs_error_rad"] == 0.0
    assert result["bindings"]["hard_contract_sha256"] == sha


def test_oracle_steps_teacher_action_without_teleport_or_ppo():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    oracle = source.split("def _run_teacher_qdes_oracle", 1)[1].split(
        "def _publish_teacher_qdes_oracle", 1
    )[0]
    assert "raw = (teacher - offset) / scale" in oracle
    assert "env.step(raw)" in oracle
    for forbidden in ("write_joint_state_to_sim", "write_root_state_to_sim", "env.reset(", "OnPolicyRunner"):
        assert forbidden not in oracle

    run = source.split("def _run(cfg):", 1)[1]
    assert run.index("env = gym.make") < run.index("oracle = _run_teacher_qdes_oracle")
    assert run.index("oracle = _run_teacher_qdes_oracle") < run.index(
        "runner = OnPolicyRunner"
    )
