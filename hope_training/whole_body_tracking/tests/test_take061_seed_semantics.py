"""Phase2/3 seeds are distinct from final matched/admitted results."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "hope_training/whole_body_tracking/scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_inverse(*, matched=False):
    return {
        "matched": matched,
        "admitted_count": 1,
        "solver_racket_velocity_w_mps": [0.1, 0.2, 0.3],
        "solver_signed_face_w": [1.0, 0.0, 0.0],
        "incoming_velocity_w_mps": [-3.0, 0.0, -0.5],
        "landing_aim_w_xy_m": [2.5, 0.0],
    }


def test_phase2_unmatched_solver_solution_is_a_valid_phase3_seed():
    phase3 = _load("take061_phase3_seed_test", "solve_take061_joint_ball_phase3.py")
    assert phase3.P2._solver_seed_valid(_valid_inverse(matched=False)) is True
    assert phase3.P2._solver_seed_valid({**_valid_inverse(), "admitted_count": 0}) is False
    assert phase3.P2._solver_seed_valid({**_valid_inverse(), "solver_signed_face_w": [float("nan"), 0, 0]}) is False


def test_phase2_teacher_solver_face_counterexample_does_not_reject_velocity_match():
    phase3 = _load("take061_phase3_face_counterexample", "solve_take061_joint_ball_phase3.py")
    face_error_deg = 38.92
    obsolete_face_threshold_deg = 15.0
    assert face_error_deg > obsolete_face_threshold_deg
    assert phase3.P2._phase2_velocity_matched(0.0070603, 0.1) is True
    assert phase3.P2._phase2_velocity_matched(0.100001, 0.1) is False


def test_phase3_requires_explicit_seed_valid_without_requiring_admitted():
    phase3 = _load("take061_phase3_seed_guard_test", "solve_take061_joint_ball_phase3.py")
    phase3._require_seed_valid("phase2", {"seed_valid": True, "admitted": False})
    with pytest.raises(phase3.P1.ProducerError, match="seed_valid=true"):
        phase3._require_seed_valid("phase2", {"seed_valid": False, "admitted": True})
    with pytest.raises(phase3.P1.ProducerError, match="seed_valid=true"):
        phase3._require_seed_valid("phase2", {})


def test_phase3_finite_solver_target_is_phase4_seed_before_mechanical_admission():
    phase3 = _load("take061_phase3_joint_seed_test", "solve_take061_joint_ball_phase3.py")
    kwargs = {
        "q_ref": np.zeros((57, 31)),
        "site": np.zeros((57, 3)),
        "velocity": np.zeros((57, 3)),
        "face": np.zeros((57, 3)),
        "solver": {
            "solver_admitted": True,
            "matched": False,
            "racket_velocity_w_mps": [0.1, 0.2, 0.3],
            "signed_face_w": [1.0, 0.0, 0.0],
        },
    }
    assert phase3._phase4_seed_valid(**kwargs) is True
    kwargs["q_ref"][0, 0] = np.nan
    assert phase3._phase4_seed_valid(**kwargs) is False


def test_phase4_exit_code_requires_final_robust_admission(monkeypatch):
    phase4 = _load("take061_phase4_seed_test", "materialize_take061_exact_face_phase4.py")
    monkeypatch.setattr(phase4, "parser", lambda: type("P", (), {"parse_args": lambda self: object()})())
    monkeypatch.setattr(
        phase4, "solve",
        lambda _args: {"exact_face_admitted": True, "robust_curriculum_center": False},
    )
    assert phase4.main() == 2
    monkeypatch.setattr(
        phase4, "solve",
        lambda _args: {"exact_face_admitted": True, "robust_curriculum_center": True},
    )
    assert phase4.main() == 0
