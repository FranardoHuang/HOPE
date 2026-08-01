"""Dependency-light checks for portable dynamic-ready PPO recipe identity."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts/train.py"
CONTRACT_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/training_contract.py"
)


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_train_module(monkeypatch):
    hydra = _module("hydra", main=lambda **kwargs: (lambda function: function))

    class FakeOmegaConf:
        @staticmethod
        def resolve(cfg):
            return None

        @staticmethod
        def set_struct(cfg, value):
            return None

    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        _module(
            "omegaconf",
            ListConfig=type("ListConfig", (list,), {}),
            OmegaConf=FakeOmegaConf,
        ),
    )
    package = _module("whole_body_tracking")
    package.__path__ = []
    utils = _module("whole_body_tracking.utils")
    utils.__path__ = []
    monkeypatch.setitem(sys.modules, "whole_body_tracking", package)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.utils", utils)
    contract_spec = importlib.util.spec_from_file_location(
        "whole_body_tracking.utils.training_contract", CONTRACT_PATH
    )
    assert contract_spec is not None and contract_spec.loader is not None
    contract = importlib.util.module_from_spec(contract_spec)
    contract_spec.loader.exec_module(contract)
    monkeypatch.setitem(
        sys.modules, "whole_body_tracking.utils.training_contract", contract
    )

    spec = importlib.util.spec_from_file_location(
        "portable_action_ball_train_under_test", TRAIN_PATH
    )
    assert spec is not None and spec.loader is not None
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return train


class _AgentCfg:
    def to_dict(self):
        return {
            "num_steps_per_env": 24,
            "empirical_normalization": True,
            "policy": {"class_name": "ActorCritic", "init_noise_std": 0.02},
            "algorithm": {"class_name": "PPO", "rnd_cfg": None},
        }


def _bootstrap(repo_root: Path) -> dict:
    config = repo_root / "configs/dynamic_ready"
    config.mkdir(parents=True)
    artifact = config / "candidate.json"
    receipt = config / "nominal_hold.json"
    artifact.write_bytes(b"identical-artifact")
    receipt.write_bytes(b"identical-receipt")
    return {
        "schema_version": 3,
        "ready_source": {
            "identity": {
                "schema_version": 2,
                "kind": "action_ball_dynamic_ready_runtime_binding_v2",
                "rows": [
                    {
                        "artifact": {
                            "path": str(artifact),
                            "sha256": "a" * 64,
                            "content_sha256": "b" * 64,
                        },
                        "nominal_hold_receipt": {
                            "path": str(receipt),
                            "sha256": "c" * 64,
                            "content_sha256": "d" * 64,
                        },
                        "physical_ready": {"joint_pos_rad": [0.0] * 31},
                    }
                ],
                "binding_sha256": "e" * 64,
            }
        },
    }


def _all_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _all_strings(key)
            yield from _all_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _all_strings(child)
    elif isinstance(value, str):
        yield value


def test_dynamic_ready_policy_recipe_sha_is_checkout_location_independent(
    tmp_path, monkeypatch
):
    train = _load_train_module(monkeypatch)
    checkout_a = tmp_path / "checkout_a"
    checkout_b = tmp_path / "checkout_b"
    bootstrap_a = _bootstrap(checkout_a)
    bootstrap_b = _bootstrap(checkout_b)

    recipe_a = train._action_ball_agent_recipe(
        _AgentCfg(),
        policy_bootstrap=bootstrap_a,
        policy_identity_repo_root=checkout_a,
    )
    recipe_b = train._action_ball_agent_recipe(
        _AgentCfg(),
        policy_bootstrap=bootstrap_b,
        policy_identity_repo_root=checkout_b,
    )

    assert recipe_a["sha256"] == recipe_b["sha256"]
    assert recipe_a["recipe"] == recipe_b["recipe"]
    signed_row = recipe_a["recipe"]["policy_initialization"]["ready_source"][
        "identity"
    ]["rows"][0]
    assert signed_row["artifact"]["path"] == (
        "configs/dynamic_ready/candidate.json"
    )
    for value in _all_strings(recipe_a["recipe"]["policy_initialization"]):
        assert not Path(value).is_absolute()
        assert ".." not in Path(value).parts
    assert bootstrap_a["ready_source"]["identity"]["rows"][0]["artifact"][
        "path"
    ] == str(checkout_a / "configs/dynamic_ready/candidate.json")

    changed_receipt = deepcopy(bootstrap_a)
    changed_receipt["ready_source"]["identity"]["rows"][0][
        "nominal_hold_receipt"
    ]["sha256"] = "f" * 64
    changed_recipe = train._action_ball_agent_recipe(
        _AgentCfg(),
        policy_bootstrap=changed_receipt,
        policy_identity_repo_root=checkout_a,
    )
    assert changed_recipe["sha256"] != recipe_a["sha256"]

    changed_ready = deepcopy(bootstrap_a)
    changed_ready["ready_source"]["identity"]["rows"][0]["physical_ready"][
        "joint_pos_rad"
    ][0] = 0.1
    changed_ready_recipe = train._action_ball_agent_recipe(
        _AgentCfg(),
        policy_bootstrap=changed_ready,
        policy_identity_repo_root=checkout_a,
    )
    assert changed_ready_recipe["sha256"] != recipe_a["sha256"]


def test_shared_ready_policy_recipe_remains_legacy_byte_equivalent(
    tmp_path, monkeypatch
):
    train = _load_train_module(monkeypatch)
    bootstrap = {
        "schema_version": 1,
        "ready_source": {"shared_ready_joint_pos_sha256": "a" * 64},
    }
    recipe = train._action_ball_agent_recipe(
        _AgentCfg(),
        policy_bootstrap=bootstrap,
        policy_identity_repo_root=tmp_path,
    )
    assert recipe["recipe"]["policy_initialization"] is bootstrap
