"""Dependency-light tests for checkpoint-to-launch-claim binding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
