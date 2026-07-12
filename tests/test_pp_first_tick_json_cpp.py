from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/cpp/pp_first_tick_json_core_test.cpp"
INCLUDE = (
    ROOT / "agi/a3_deploy_example/src/a3/a3_deploy_onnx_ref/include"
)


def _compiler_command(output: Path) -> list[str]:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler available")
    command = [
        compiler,
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(INCLUDE),
        str(SOURCE),
        "-o",
        str(output),
    ]
    if platform.system() == "Darwin":
        sdk_root = Path("/Library/Developer/CommandLineTools/SDKs")
        candidates = sorted(
            (
                path
                for path in sdk_root.glob("MacOSX*.sdk")
                if (path / "usr/include/c++/v1/algorithm").exists()
            ),
            reverse=True,
        )
        if candidates:
            sdk = candidates[0]
            command[1:1] = [
                "-stdlib=libc++",
                "-isysroot",
                str(sdk),
                "-isystem",
                str(sdk / "usr/include/c++/v1"),
            ]
    return command


def test_dependency_light_cpp_first_tick_contract(tmp_path: Path) -> None:
    binary = tmp_path / "pp-first-tick-core-test"
    subprocess.run(
        _compiler_command(binary),
        check=True,
        cwd=ROOT,
        env={**os.environ, "LC_ALL": "C"},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    subprocess.run([str(binary), str(run_dir.resolve())], check=True, cwd=ROOT)
    document = json.loads((run_dir / "first.json").read_text())
    assert document["artifact_kind"] == "gate3_first_tick_joined_source_diagnostic"
    assert document["evaluation_contract_exact"] is False
    payload = document["payload"]
    assert payload["evaluation_contract_exact"] is False
    exactness = payload["exactness"]
    assert exactness["planner_snapshot_exact"] is False
    assert exactness["native_sample_alignment_exact"] is False
    assert exactness["source_binary_binding_exact"] is False
    assert exactness["source_semantics_closure_exact"] is False
    assert exactness["runtime_artifact_closure_exact"] is False
    assert exactness["inexact_reasons"]
    assert "source_commit" not in payload

    def require_formal(candidate: dict[str, object]) -> None:
        allowed_formal_kinds = {"gate3_native_same_sample_first_tick_v2"}
        candidate_payload = candidate.get("payload")
        if not isinstance(candidate_payload, dict):
            raise ValueError("first-tick payload is missing")
        kind = candidate.get("artifact_kind")
        if kind not in allowed_formal_kinds or candidate_payload.get("artifact_kind") != kind:
            raise ValueError("diagnostic artifact kind is permanently inexact")
        candidate_exactness = candidate_payload.get("exactness")
        if not isinstance(candidate_exactness, dict):
            raise ValueError("first-tick exactness block is missing")
        exact_keys = {
            "planner_snapshot_exact",
            "native_sample_alignment_exact",
            "source_binary_binding_exact",
            "source_semantics_closure_exact",
            "runtime_artifact_closure_exact",
        }
        if (
            candidate.get("evaluation_contract_exact") is not True
            or candidate_payload.get("evaluation_contract_exact") is not True
            or any(candidate_exactness.get(key) is not True for key in exact_keys)
            or candidate_exactness.get("inexact_reasons") != []
        ):
            raise ValueError("inexact first-tick diagnostic is not Gate3 evidence")

    with pytest.raises(ValueError, match="permanently inexact"):
        require_formal(document)

    outer_only_forgery = json.loads(json.dumps(document))
    outer_only_forgery["evaluation_contract_exact"] = True
    with pytest.raises(ValueError, match="permanently inexact"):
        require_formal(outer_only_forgery)

    flags_only_forgery = json.loads(json.dumps(document))
    flags_only_forgery["evaluation_contract_exact"] = True
    forged_payload = flags_only_forgery["payload"]
    forged_payload["evaluation_contract_exact"] = True
    forged_payload["exactness"]["inexact_reasons"] = []
    for key in {
        "planner_snapshot_exact",
        "native_sample_alignment_exact",
        "source_binary_binding_exact",
        "source_semantics_closure_exact",
        "runtime_artifact_closure_exact",
    }:
        forged_payload["exactness"][key] = True
    with pytest.raises(ValueError, match="permanently inexact"):
        require_formal(flags_only_forgery)
