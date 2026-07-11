"""Behavioral tests for export-time atomic replacement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/utils/atomic_output.py"
)
SPEC = importlib.util.spec_from_file_location("atomic_output_under_test", MODULE_PATH)
AO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AO)


def _temps_for(final: Path) -> list[Path]:
    return list(final.parent.glob(f".{final.name}.*.tmp"))


def test_atomic_output_replaces_existing_only_after_success(tmp_path: Path):
    final = tmp_path / "policy.onnx"
    final.write_bytes(b"accepted-old-model")
    with AO.atomic_output_path(final) as temporary:
        assert temporary.parent == final.parent
        assert final.read_bytes() == b"accepted-old-model"
        temporary.write_bytes(b"complete-new-model")
    assert final.read_bytes() == b"complete-new-model"
    assert _temps_for(final) == []


def test_atomic_output_failure_preserves_existing_and_cleans_temp(tmp_path: Path):
    final = tmp_path / "policy.onnx"
    final.write_bytes(b"accepted-old-model")
    with pytest.raises(RuntimeError, match="injected export failure"):
        with AO.atomic_output_path(final) as temporary:
            temporary.write_bytes(b"half-model")
            raise RuntimeError("injected export failure")
    assert final.read_bytes() == b"accepted-old-model"
    assert _temps_for(final) == []


def test_atomic_output_empty_result_never_clobbers_destination(tmp_path: Path):
    final = tmp_path / "policy.onnx"
    final.write_bytes(b"accepted-old-model")
    with pytest.raises(RuntimeError, match="missing/empty"):
        with AO.atomic_output_path(final):
            pass
    assert final.read_bytes() == b"accepted-old-model"
    assert _temps_for(final) == []
