"""Dependency-light tests for checkpoint-to-launch-claim binding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "source/whole_body_tracking/whole_body_tracking/utils"


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "training_contract_launch_claim_under_test", UTILS / "training_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_runner_module(monkeypatch, contract_module):
    class FakeOnPolicyRunner:
        def __init__(self, env, train_cfg, log_dir, device):
            self.logger_type = "tensorboard"
            self.saved = []

        def save(self, path, infos=None):
            self.saved.append((path, infos))

        def log(self, locs, width=80, pad=35):
            return None

    fake_rsl_rl = _module("rsl_rl")
    fake_rsl_rl.__path__ = []
    fake_runners = _module("rsl_rl.runners")
    fake_runners.__path__ = []
    fake_isaaclab_rl = _module("isaaclab_rl")
    fake_isaaclab_rl.__path__ = []
    fake_wbt = _module("whole_body_tracking")
    fake_wbt.__path__ = []
    fake_utils = _module("whole_body_tracking.utils")
    fake_utils.__path__ = []
    fake_tasks = _module("whole_body_tracking.tasks")
    fake_tasks.__path__ = []
    fake_tracking = _module("whole_body_tracking.tasks.tracking")
    fake_tracking.__path__ = []
    fake_mdp = _module("whole_body_tracking.tasks.tracking.mdp")
    fake_mdp.__path__ = []
    fake_hope_rewards = _module(
        "whole_body_tracking.tasks.tracking.mdp.hope_rewards",
        consume_base_decel_activation_counters=lambda env, command_name: {},
    )
    modules = {
        "torch": _module("torch", Tensor=type("Tensor", (), {})),
        "rsl_rl": fake_rsl_rl,
        "rsl_rl.env": _module("rsl_rl.env", VecEnv=type("VecEnv", (), {})),
        "rsl_rl.runners": fake_runners,
        "rsl_rl.runners.on_policy_runner": _module(
            "rsl_rl.runners.on_policy_runner", OnPolicyRunner=FakeOnPolicyRunner
        ),
        "isaaclab_rl": fake_isaaclab_rl,
        "isaaclab_rl.rsl_rl": _module(
            "isaaclab_rl.rsl_rl", export_policy_as_onnx=lambda *args, **kwargs: None
        ),
        "whole_body_tracking": fake_wbt,
        "whole_body_tracking.tasks": fake_tasks,
        "whole_body_tracking.tasks.tracking": fake_tracking,
        "whole_body_tracking.tasks.tracking.mdp": fake_mdp,
        "whole_body_tracking.tasks.tracking.mdp.hope_rewards": fake_hope_rewards,
        "whole_body_tracking.utils": fake_utils,
        "whole_body_tracking.utils.exporter": _module(
            "whole_body_tracking.utils.exporter",
            attach_onnx_metadata=lambda *args, **kwargs: None,
            export_motion_policy_as_onnx=lambda *args, **kwargs: False,
            is_empirical_normalizer=lambda value: False,
        ),
        "whole_body_tracking.utils.training_contract": contract_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "motion_runner_launch_claim_under_test", UTILS / "my_on_policy_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_embeds_exact_launch_claim_without_mutating_scientific_contract(monkeypatch):
    contract = _load_contract_module()
    runner_module = _load_runner_module(monkeypatch, contract)
    claim = "a" * 64
    contract_sha = "b" * 64
    runner = runner_module.MotionOnPolicyRunner(
        object(), {},
        training_contract_schema_version=3,
        training_contract_sha256=contract_sha,
        training_contract_lineage_exact=True,
        training_launch_claim_sha256=claim,
    )
    original_infos = {"keep": "value"}
    runner.save("model_1.pt", original_infos)
    _, saved_infos = runner.saved[-1]
    assert saved_infos == {
        "keep": "value",
        contract.CHECKPOINT_CONTRACT_SCHEMA_KEY: 3,
        contract.CHECKPOINT_CONTRACT_SHA_KEY: contract_sha,
        contract.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY: 1,
        contract.CHECKPOINT_LAUNCH_CLAIM_SHA_KEY: claim,
    }
    assert original_infos == {"keep": "value"}


@pytest.mark.parametrize(
    "claim", ["a" * 63, "A" * 64, " " + "a" * 64, "g" * 64, 7, True]
)
def test_runner_rejects_noncanonical_launch_claim(monkeypatch, claim):
    contract = _load_contract_module()
    runner_module = _load_runner_module(monkeypatch, contract)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        runner_module.MotionOnPolicyRunner(
            object(), {}, training_launch_claim_sha256=claim
        )


def test_absent_claim_writes_no_launch_key_and_train_reads_only_top_level(monkeypatch):
    contract = _load_contract_module()
    runner_module = _load_runner_module(monkeypatch, contract)
    runner = runner_module.MotionOnPolicyRunner(object(), {})
    runner.save("model_1.pt", {"keep": "value"})
    assert runner.saved[-1][1] == {"keep": "value"}

    train_source = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert '_get(cfg, "training_launch_claim_sha256")' in train_source
    assert "training_launch_claim_sha256=training_launch_claim_sha256" in train_source
    assert 'hard_contract["training_launch_claim_sha256"]' not in train_source


def test_live_logger_exports_and_consumes_post_swing_activation_totals_once(monkeypatch):
    contract = _load_contract_module()
    runner_module = _load_runner_module(monkeypatch, contract)

    class MotionTerm:
        metrics = {}

        def __init__(self):
            self.consume_calls = 0

        def consume_post_swing_activation_counters(self):
            self.consume_calls += 1
            return {
                "post_swing_replay_buffer_not_ready_reset_count": 11,
                "post_swing_replay_eligible_reset_count": 101,
                "post_swing_replay_random_not_selected_reset_count": 49,
                "post_swing_replay_selected_reset_count": 52,
                "post_swing_replay_started_reset_count": 52,
            }

    term = MotionTerm()
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            active_terms=["motion"], get_term=lambda name: term
        ),
        episode_length_buf=7,
        common_step_counter=123,
    )
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.env = SimpleNamespace(unwrapped=env)
    logged = {}
    runner._log_scalar = lambda tag, value, step: logged.setdefault(
        tag, (value, step)
    )

    runner._log_live_metrics(step=17)

    assert term.consume_calls == 1
    assert logged["Live/motion/post_swing_replay_buffer_not_ready_reset_count"] == (
        11.0,
        17,
    )
    assert logged["Live/motion/post_swing_replay_eligible_reset_count"] == (
        101.0,
        17,
    )
    assert logged["Live/motion/post_swing_replay_random_not_selected_reset_count"] == (
        49.0,
        17,
    )
    assert logged["Live/motion/post_swing_replay_selected_reset_count"] == (
        52.0,
        17,
    )
    assert logged["Live/motion/post_swing_replay_started_reset_count"] == (
        52.0,
        17,
    )


def test_live_logger_prefers_aggregate_v1_v2_activation_transaction(monkeypatch):
    contract = _load_contract_module()
    runner_module = _load_runner_module(monkeypatch, contract)

    class MotionTerm:
        metrics = {}

        def __init__(self):
            self.aggregate_calls = 0
            self.legacy_calls = 0

        def consume_training_activation_counters(self):
            self.aggregate_calls += 1
            return {
                "v1_velocity_mimic_eligible_sample_count": 4096,
                "v1_held_wrist_excluded_sample_count": 4096,
                "v2_strike_window_eligible_imitation_sample_count": 512,
                "v2_quarter_scaled_strike_window_imitation_sample_count": 512,
            }

        def consume_post_swing_activation_counters(self):
            self.legacy_calls += 1
            raise AssertionError("aggregate consumer must suppress the legacy fallback")

    term = MotionTerm()
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            active_terms=["motion"], get_term=lambda name: term
        ),
        episode_length_buf=7,
        common_step_counter=123,
    )
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.env = SimpleNamespace(unwrapped=env)
    logged = {}
    runner._log_scalar = lambda tag, value, step: logged.setdefault(
        tag, (value, step)
    )

    runner._log_live_metrics(step=19)

    assert term.aggregate_calls == 1
    assert term.legacy_calls == 0
    assert logged["Live/motion/v1_velocity_mimic_eligible_sample_count"] == (
        4096.0,
        19,
    )
    assert logged["Live/motion/v1_held_wrist_excluded_sample_count"] == (
        4096.0,
        19,
    )
    assert logged[
        "Live/motion/v2_strike_window_eligible_imitation_sample_count"
    ] == (512.0, 19)
    assert logged[
        "Live/motion/v2_quarter_scaled_strike_window_imitation_sample_count"
    ] == (512.0, 19)


def test_live_logger_consumes_base_decel_raw_ledger_once_with_stable_tags(monkeypatch):
    contract = _load_contract_module()
    runner_module = _load_runner_module(monkeypatch, contract)
    calls = []
    rewards_module = sys.modules[
        "whole_body_tracking.tasks.tracking.mdp.hope_rewards"
    ]

    def _consume(env, command_name):
        calls.append((env, command_name))
        return {
            "base_decel_eligible_sample_count": 101,
            "base_decel_raw_kernel_sum": 37.5,
            "base_decel_raw_kernel_nonzero_sample_count": 99,
        }

    rewards_module.consume_base_decel_activation_counters = _consume
    term = SimpleNamespace(
        metrics={},
        cfg=SimpleNamespace(base_decel_activation_enabled=True),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            active_terms=["racket_target"], get_term=lambda name: term
        ),
        episode_length_buf=7,
        common_step_counter=123,
    )
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.env = SimpleNamespace(unwrapped=env)
    logged = {}
    runner._log_scalar = lambda tag, value, step: logged.setdefault(
        tag, (value, step)
    )

    runner._log_live_metrics(step=23)

    assert calls == [(env, "racket_target")]
    assert logged["Live/racket_target/base_decel_eligible_sample_count"] == (
        101.0,
        23,
    )
    assert logged["Live/racket_target/base_decel_raw_kernel_sum"] == (
        37.5,
        23,
    )
    assert logged[
        "Live/racket_target/base_decel_raw_kernel_nonzero_sample_count"
    ] == (99.0, 23)
