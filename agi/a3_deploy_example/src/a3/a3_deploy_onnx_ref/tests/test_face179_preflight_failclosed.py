"""Dependency-light contract tests plus optional real runner/ONNX integration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/verify_face179_preflight_failclosed.py"
SPEC = importlib.util.spec_from_file_location("face179_preflight_under_test", MODULE_PATH)
PF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PF)


def test_mutations_are_narrow_and_non_mutating():
    original = {
        "training_contract_exact": "1",
        "stage1_normal_envelope_payload_sha256": "a" * 64,
        "unrelated": "keep",
    }
    assert PF.mutate_metadata(original, "metadata_stripped") == {}
    missing = PF.mutate_metadata(original, "missing_envelope")
    assert "stage1_normal_envelope_payload_sha256" not in missing
    assert missing["training_contract_exact"] == "1"
    inexact = PF.mutate_metadata(original, "training_contract_inexact")
    assert inexact["training_contract_exact"] == "0"
    assert inexact["stage1_normal_envelope_payload_sha256"] == "a" * 64
    masked = PF.mutate_metadata(original, "actor_leg_ref_mask_unsupported")
    assert masked["actor_leg_ref_mask"] == "1"
    assert masked["training_contract_exact"] == "1"
    assert original["training_contract_exact"] == "1"


def test_result_contract_requires_parsed_publishability_and_no_backend():
    accepted = subprocess.CompletedProcess(
        [], 0,
        stdout=(
            "backend_not_initialized=true obs_dim=179 "
            "publishable_model_contract=true training_contract_exact=1"
        ),
        stderr="",
    )
    PF.check_result(accepted, should_accept=True)
    with pytest.raises(RuntimeError, match="omitted parsed-contract"):
        PF.check_result(
            subprocess.CompletedProcess([], 0, stdout="backend_not_initialized=true obs_dim=179", stderr=""),
            should_accept=True,
        )
    with pytest.raises(RuntimeError, match="unexpectedly passed"):
        PF.check_result(accepted, should_accept=False)
    with pytest.raises(RuntimeError, match="touched or announced a backend"):
        PF.check_result(
            subprocess.CompletedProcess([], 3, stdout="", stderr="backend cfg: forbidden"),
            should_accept=False,
        )


def test_no_publish_is_separate_from_legacy_model_relaxation_in_source():
    main = (ROOT / "src/a3_deploy/a3_pingpong_main.cpp").read_text(encoding="utf-8")
    policy = (ROOT / "include/a3_pingpong/pp_policy.hpp").read_text(encoding="utf-8")
    onnx = (ROOT / "include/a3_pingpong/pp_onnx_policy.hpp").read_text(encoding="utf-8")
    assert "pcfg.diagnostic_no_publish = no_publish" in main
    assert (
        "pcfg.allow_legacy_model_diagnostic = allow_legacy_model_diagnostic" in main
    )
    assert "cfg.allow_legacy_model_diagnostic" in policy
    assert "onnx_(onnx_path, cfg.diagnostic_no_publish)" not in policy
    assert '" publishable_model_contract="' in main
    assert "pp->onnx().publishable_model_contract() ? \"true\" : \"false\"" in main
    assert '" training_contract_exact="' in main
    assert "pp->onnx().training_contract_exact() ? \"1\" : \"0\"" in main
    assert "bool publishable_model_contract() const" in onnx
    assert 'LookupMetaOptional(md, alloc, "actor_leg_ref_mask")' in onnx
    assert "C++ observation builder does not" in onnx
    assert "allow_legacy_model_diagnostic && model_preflight_only" in main


def test_optional_real_runner_rejects_mutated_models_before_backend():
    runner = os.environ.get("A3_PP_RUNNER_PATH")
    runtime_cfg = os.environ.get("A3_PP_RUNTIME_CFG")
    model = os.environ.get("A3_PP_ONNX_PATH")
    if not runner or not runtime_cfg or not model:
        pytest.skip(
            "set A3_PP_RUNNER_PATH, A3_PP_RUNTIME_CFG and A3_PP_ONNX_PATH for integration"
        )
    result = PF.run_suite(Path(runner), Path(runtime_cfg), Path(model))
    assert result["baseline"]["returncode"] == 0
    assert all(value["returncode"] != 0 for value in result["variants"].values())
    assert result["legacy_flag_with_preflight"]["returncode"] == 2
