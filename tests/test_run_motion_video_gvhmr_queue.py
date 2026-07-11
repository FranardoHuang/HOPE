from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_motion_video_gvhmr_queue.py"
SPEC = importlib.util.spec_from_file_location("run_motion_video_gvhmr_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def _asset(asset_id: str, relpath: str) -> dict:
    return {
        "id": asset_id,
        "source_relpath": relpath,
        "sha256": hashlib.sha256(asset_id.encode()).hexdigest(),
    }


def test_select_assets_uses_manifest_processing_order():
    manifest = {
        "processing_order": ["b", "a"],
        "assets": [_asset("a", "one/a.mp4"), _asset("b", "two/b.mp4")],
    }
    assert [item["id"] for item in QUEUE.select_assets(manifest, None)] == ["b", "a"]


def test_select_assets_preserves_requested_order():
    manifest = {
        "processing_order": ["a", "b"],
        "assets": [_asset("a", "one/a.mp4"), _asset("b", "two/b.mp4")],
    }
    assert [item["id"] for item in QUEUE.select_assets(manifest, ["b", "a"])] == ["b", "a"]
    with pytest.raises(QUEUE.IntakeError, match="unique"):
        QUEUE.select_assets(manifest, ["a", "a"])
    with pytest.raises(QUEUE.IntakeError, match="absent"):
        QUEUE.select_assets(manifest, ["missing"])


def test_select_assets_rejects_gvhmr_output_alias():
    manifest = {
        "processing_order": ["first", "second"],
        "assets": [
            _asset("first", "family_a/shared.mp4"),
            _asset("second", "family_b/shared.mp4"),
        ]
    }
    with pytest.raises(QUEUE.IntakeError, match="output stem"):
        QUEUE.select_assets(manifest, None)


def test_completed_binding_requires_exact_output_bytes(tmp_path):
    output = tmp_path / "hmr4d_results.pt"
    output.write_bytes(b"result")
    asset = _asset("a", "one/a.mp4")
    contract = {"gvhmr_commit": "abc123", "static_camera": True, "tool": "sha"}
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"status": "pass", "result_sha256": QUEUE.sha256_file(output)}),
        encoding="utf-8",
    )
    binding = {
        "status": "complete",
        "source_sha256": asset["sha256"],
        "processing_contract": contract,
        "output_sha256": QUEUE.sha256_file(output),
        "structural_audit_path": str(audit),
        "structural_audit_sha256": QUEUE.sha256_file(audit),
    }
    assert QUEUE.completed_binding_matches(binding, asset, output, contract)
    changed_contract = dict(contract, static_camera=False)
    assert not QUEUE.completed_binding_matches(binding, asset, output, changed_contract)
    output.write_bytes(b"changed")
    assert not QUEUE.completed_binding_matches(binding, asset, output, contract)


def test_gpu_used_mib_parses_one_physical_gpu_row(monkeypatch):
    monkeypatch.setattr(QUEUE.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        QUEUE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="17272\n", stderr=""),
    )
    assert QUEUE.gpu_used_mib(1) == 17272


def test_gpu_used_mib_rejects_ambiguous_or_invalid_output(monkeypatch):
    monkeypatch.setattr(QUEUE.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        QUEUE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="100\n200\n", stderr=""),
    )
    with pytest.raises(QUEUE.IntakeError, match="expected one GPU"):
        QUEUE.gpu_used_mib(1)


def test_dependency_tree_fingerprint_binds_paths_sizes_and_bytes(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "nested" / "b.bin").write_bytes(b"bb")
    first = QUEUE.tree_fingerprint(tmp_path)
    assert first["files"] == 2
    assert first["bytes"] == 3
    assert first == QUEUE.tree_fingerprint(tmp_path)
    (tmp_path / "nested" / "b.bin").write_bytes(b"bc")
    assert QUEUE.tree_fingerprint(tmp_path)["sha256"] != first["sha256"]
