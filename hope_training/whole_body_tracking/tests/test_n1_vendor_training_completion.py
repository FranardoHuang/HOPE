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
    encoded = output[len("HOPE_TRAINING_COMPLETE_JSON=") :].rstrip("\n")
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


def test_completion_claim_uses_verified_exec_boundary_without_argv_cycle(train):
    assert train._resolve_n1_vendor_completion_launch_claim_sha256(
        configured_sha256=None,
        exec_boundary_sha256=SHA_A,
    ) == SHA_A
    assert train._resolve_n1_vendor_completion_launch_claim_sha256(
        configured_sha256=SHA_A,
        exec_boundary_sha256=SHA_A,
    ) == SHA_A
    assert train._resolve_n1_vendor_completion_launch_claim_sha256(
        configured_sha256=SHA_A,
        exec_boundary_sha256=None,
    ) == SHA_A


def test_completion_claim_rejects_bad_or_conflicting_exec_boundary(train):
    with pytest.raises(ValueError, match="64 lowercase hex"):
        train._resolve_n1_vendor_completion_launch_claim_sha256(
            configured_sha256=None,
            exec_boundary_sha256="bad",
        )
    with pytest.raises(ValueError, match="SHAs differ"):
        train._resolve_n1_vendor_completion_launch_claim_sha256(
            configured_sha256=SHA_A,
            exec_boundary_sha256=SHA_B,
        )


def test_effective_claim_reads_environment_only_for_complete_vendor_identity(train):
    resolve = train._resolve_effective_n1_vendor_training_launch_claim_sha256
    assert resolve(
        diagnostic_stage_present=True,
        vendor_contract_present=True,
        diagnostic_action_ball=False,
        configured_sha256=None,
        exec_boundary_sha256=SHA_A,
    ) == SHA_A
    for stage_present, contract_present in (
        (False, False),
        (True, False),
        (False, True),
    ):
        assert resolve(
            diagnostic_stage_present=stage_present,
            vendor_contract_present=contract_present,
            diagnostic_action_ball=False,
            configured_sha256=None,
            exec_boundary_sha256="malformed ambient value must not be read",
        ) is None


def test_effective_claim_also_binds_the_diagnostic_action_ball_exec_boundary(train):
    """诊断 ActionBall 没有正式 claim 路径,它的 claim 只能从 exec 边界的环境变量拿。

    两个 211 发射器都不发 ``n1_vendor_diagnostic_stage`` /
    ``vendor_runtime_training_contract_sha256``(诊断跑发不了),所以在 2026-08-07
    之前这个值恒为 None,checkpoint 的 infos 里永远没有 launch claim 键 —— 而发射器的
    scale4096 终局验收门恰恰要比对它。
    """

    resolve = train._resolve_effective_n1_vendor_training_launch_claim_sha256
    assert resolve(
        diagnostic_stage_present=False,
        vendor_contract_present=False,
        diagnostic_action_ball=True,
        configured_sha256=None,
        exec_boundary_sha256=SHA_A,
    ) == SHA_A
    # 同一个进程两边都给,且一致 -> 仍然是那个值;不一致 -> 当场炸。
    assert resolve(
        diagnostic_stage_present=False,
        vendor_contract_present=False,
        diagnostic_action_ball=True,
        configured_sha256=SHA_A,
        exec_boundary_sha256=SHA_A,
    ) == SHA_A
    with pytest.raises(ValueError, match="SHAs differ"):
        resolve(
            diagnostic_stage_present=False,
            vendor_contract_present=False,
            diagnostic_action_ball=True,
            configured_sha256=SHA_A,
            exec_boundary_sha256=SHA_B,
        )
    with pytest.raises(ValueError, match="64 lowercase hex"):
        resolve(
            diagnostic_stage_present=False,
            vendor_contract_present=False,
            diagnostic_action_ball=True,
            configured_sha256=None,
            exec_boundary_sha256="not a digest",
        )
    # 不是诊断 ActionBall、又没有完整 vendor 身份时,环境变量连读都不读。
    assert resolve(
        diagnostic_stage_present=False,
        vendor_contract_present=False,
        diagnostic_action_ball=False,
        configured_sha256=None,
        exec_boundary_sha256="malformed ambient value must not be read",
    ) is None
    # 这个准入位必须是真 bool,不接受 1/"true"/对象这种“看着像真”的东西。
    for impostor in (1, "true", object()):
        with pytest.raises(TypeError, match="exact bool"):
            resolve(
                diagnostic_stage_present=False,
                vendor_contract_present=False,
                diagnostic_action_ball=impostor,
                configured_sha256=None,
                exec_boundary_sha256=SHA_A,
            )


def test_diagnostic_action_ball_claim_is_computed_before_it_is_consumed(train):
    """``diagnostic_launch`` 必须在解析 claim 之前算出来,否则又变回恒 None。

    这条门看的是 train.py 的真实语句顺序:布尔的赋值必须早于把它喂给 resolver 的那次
    调用,并且 resolver 拿到的就是那个布尔。
    """

    source = TRAIN_PATH.read_text(encoding="utf-8")
    assignment = source.index("    diagnostic_launch = action_ball_launch_requested and (")
    consumption = source.index("            diagnostic_action_ball=diagnostic_launch,")
    assert assignment < consumption
    # 旧代码在 action_ball_launch_requested 分支里第二次赋值 diagnostic_launch;
    # 留着会让上面的顺序保证失效(claim 用的是尚未定型的值)。
    assert source.count("    diagnostic_launch = ") == 1
    assert source.count("diagnostic_action_ball=diagnostic_launch,") == 1


def test_runner_and_completion_share_one_effective_vendor_claim_source():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    assert source.count(
        "training_launch_claim_sha256=(\n"
        "            effective_training_launch_claim_sha256\n"
        "        ),"
    ) == 1
    assert source.count(
        "training_launch_claim_sha256=(\n"
        "                effective_training_launch_claim_sha256\n"
        "            ),"
    ) == 1


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
        ({"stage": "push_evidence"}, "must be one of"),
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


@pytest.mark.parametrize("stage", ["smoke", "probe", "long"])
def test_exact_stage_allowlist(train, stage):
    assert _build(train, stage=stage)["stage"] == stage
