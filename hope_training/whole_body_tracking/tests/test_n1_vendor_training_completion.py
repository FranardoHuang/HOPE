"""Dependency-light tests for the N1 vendor natural-completion marker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts/train.py"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def train(monkeypatch):
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
    spec = importlib.util.spec_from_file_location(
        "train_n1_completion_under_test", TRAIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build(train, **overrides):
    fields = {
        "diagnostic_stage_present": True,
        "stage": "probe",
        "vendor_contract_present": True,
        "num_envs": 4096,
        "max_iterations": 5,
        "training_launch_claim_sha256": SHA_A,
        "training_contract_sha256": SHA_B,
        "vendor_runtime_training_contract_sha256": SHA_C,
    }
    fields.update(overrides)
    return train._build_n1_vendor_training_completion_payload(**fields)


def test_completion_payload_and_canonical_output(train, capsys):
    payload = _build(train)
    assert payload == {
        "cleanup_complete": True,
        "completed_ppo_updates": 5,
        "event": "hope_training_complete",
        "num_envs": 4096,
        "schema_version": 1,
        "stage": "probe",
        "training_contract_sha256": SHA_B,
        "training_launch_claim_sha256": SHA_A,
        "vendor_runtime_training_contract_sha256": SHA_C,
    }

    train._emit_n1_vendor_training_completion(payload)
    output = capsys.readouterr().out
    assert output.startswith("HOPE_TRAINING_COMPLETE_JSON=")
    encoded = output.removeprefix("HOPE_TRAINING_COMPLETE_JSON=").rstrip("\n")
    assert encoded == json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def test_ordinary_training_is_a_strict_noop(train, capsys):
    assert _build(
        train,
        diagnostic_stage_present=False,
        stage=None,
        vendor_contract_present=False,
        vendor_runtime_training_contract_sha256=None,
    ) is None
    train._emit_n1_vendor_training_completion(None)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            {"diagnostic_stage_present": False},
            "must be supplied together",
        ),
        (
            {"vendor_contract_present": False},
            "must be supplied together",
        ),
        ({"stage": None}, "must be one of"),
        ({"stage": "unknown"}, "must be one of"),
        ({"num_envs": True}, "exact integer"),
        ({"num_envs": "4096"}, "exact integer"),
        ({"max_iterations": 1.0}, "exact integer"),
        ({"training_launch_claim_sha256": None}, "64 lowercase hex"),
        ({"training_contract_sha256": "A" * 64}, "64 lowercase hex"),
        ({"vendor_runtime_training_contract_sha256": "c" * 63}, "64 lowercase hex"),
    ],
)
def test_half_bound_or_inexact_payload_fails_closed(train, overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _build(train, **overrides)


@pytest.mark.parametrize("stage", ["smoke", "probe", "push_evidence", "long"])
def test_exact_stage_allowlist(train, stage):
    assert _build(train, stage=stage)["stage"] == stage
