"""The Take061 phase chain must not mix ball-physics models."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "hope_training/whole_body_tracking/scripts/solve_take061_joint_ball_phase3.py"


def _load_phase3():
    spec = importlib.util.spec_from_file_location("take061_phase3_lineage_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ball_physics_lineage_accepts_exact_bytes(tmp_path):
    phase3 = _load_phase3()
    physics = tmp_path / "physics.yaml"
    physics.write_bytes(b"flight:\n  k_d: 0.1253\n")
    report = {
        "solver_sources": {"ball_physics_sha256": phase3.P1._sha256(physics)}
    }
    phase3._require_ball_physics_lineage("phase2", report, physics)


def test_ball_physics_lineage_rejects_different_or_missing_bytes(tmp_path):
    phase3 = _load_phase3()
    physics = tmp_path / "physics.yaml"
    physics.write_bytes(b"flight:\n  k_d: 0.1253\n")
    with pytest.raises(phase3.P1.ProducerError, match="SHA mismatch"):
        phase3._require_ball_physics_lineage(
            "phase2", {"solver_sources": {"ball_physics_sha256": "0" * 64}}, physics
        )
    with pytest.raises(phase3.P1.ProducerError, match="does not declare"):
        phase3._require_ball_physics_lineage("phase3", None, physics)
