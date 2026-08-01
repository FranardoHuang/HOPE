from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import yaml


TRAINING_ROOT = Path(__file__).resolve().parents[1]
PPO_YAML = TRAINING_ROOT / "cfg" / "algo" / "ppo.yaml"
PPO_CFG = (
    TRAINING_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "ppo_cfg.py"
)


class _ConfigStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_ppo_cfg(monkeypatch):
    isaaclab_rl = types.ModuleType("isaaclab_rl")
    isaaclab_rl.__path__ = []
    rsl_rl = types.ModuleType("isaaclab_rl.rsl_rl")
    rsl_rl.RslRlPpoActorCriticCfg = _ConfigStub
    rsl_rl.RslRlPpoAlgorithmCfg = _ConfigStub
    monkeypatch.setitem(sys.modules, "isaaclab_rl", isaaclab_rl)
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", rsl_rl)

    spec = importlib.util.spec_from_file_location("ppo_cfg_under_test", PPO_CFG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_iteration_budget_is_finite_and_shared(monkeypatch):
    raw = yaml.safe_load(PPO_YAML.read_text())
    assert raw["runner"]["max_iterations"] == 25_000

    ppo_cfg = _load_ppo_cfg(monkeypatch)
    params = ppo_cfg.load_ppo_params(str(PPO_YAML))
    kwargs = ppo_cfg.runner_kwargs(params, "budget-default-test")
    assert kwargs["max_iterations"] == 25_000
    assert kwargs["policy"].noise_std_type == "scalar"


def test_explicit_runner_budget_still_overrides_default(monkeypatch):
    ppo_cfg = _load_ppo_cfg(monkeypatch)
    params = copy.deepcopy(ppo_cfg.load_ppo_params(str(PPO_YAML)))
    params["runner"]["max_iterations"] = 7

    kwargs = ppo_cfg.runner_kwargs(params, "budget-override-test")
    assert kwargs["max_iterations"] == 7


def test_explicit_log_std_policy_is_forwarded_without_changing_default(
    monkeypatch,
):
    ppo_cfg = _load_ppo_cfg(monkeypatch)
    params = copy.deepcopy(ppo_cfg.load_ppo_params(str(PPO_YAML)))
    params["policy"]["noise_std_type"] = "log"

    kwargs = ppo_cfg.runner_kwargs(params, "vendor-log-std-test")

    assert kwargs["policy"].noise_std_type == "log"


def test_legacy_yaml_without_noise_std_type_keeps_scalar_semantics(monkeypatch):
    ppo_cfg = _load_ppo_cfg(monkeypatch)
    params = copy.deepcopy(ppo_cfg.load_ppo_params(str(PPO_YAML)))
    del params["policy"]["noise_std_type"]

    kwargs = ppo_cfg.runner_kwargs(params, "legacy-scalar-test")

    assert kwargs["policy"].noise_std_type == "scalar"
