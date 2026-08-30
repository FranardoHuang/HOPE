import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_take061_exact_face_phase4.py"
SPEC = importlib.util.spec_from_file_location("take061_phase4", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_phase4_has_new_diagnostic_action_identity():
    assert M.KIND == "take061_slow_block_exact_face_phase4_v1"


def test_phase4_robust_defaults_are_stricter_than_admission():
    p = M.parser()
    assert p.get_default("robust_qdes_margin_rad") == 0.02
    assert p.get_default("robust_velocity_error_mps") == 0.08
    assert p.get_default("robust_face_error_deg") == 8.0
    assert p.get_default("robust_exact_site_position_error_m") == 0.005
    assert p.get_default("robust_exact_site_velocity_error_mps") == 0.08
