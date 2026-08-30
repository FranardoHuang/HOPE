import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "solve_take061_joint_ball_phase3.py"
SPEC = importlib.util.spec_from_file_location("take061_phase3", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_phase3_is_offline_diagnostic_semantic_kind():
    assert M.KIND == "take061_joint_ball_feasible_center_phase3_v1"


def test_phase3_defaults_keep_strict_task_match():
    parser = M.parser()
    velocity = parser.get_default("velocity_tolerance_mps")
    face = parser.get_default("face_tolerance_deg")
    assert velocity == 0.10
    assert face == 10.0

