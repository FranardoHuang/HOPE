import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "solve_take061_feasible_center_phase2.py"
SPEC = importlib.util.spec_from_file_location("take061_phase2", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_contact_taper_preserves_three_frame_contact_window():
    from scipy.signal import savgol_filter

    q = np.arange(31 * 12, dtype=np.float64).reshape(31, 12) / 100.0
    ready = np.zeros(12)
    out = M._smooth_with_contact_taper(q, ready, 15, 7, savgol_filter)
    np.testing.assert_array_equal(out[14:17], q[14:17])
    assert out.shape == q.shape
    assert np.isfinite(out).all()


def test_contact_taper_moves_far_frames_toward_ready():
    from scipy.signal import savgol_filter

    q = np.ones((31, 4), dtype=np.float64)
    ready = np.zeros(4)
    out = M._smooth_with_contact_taper(q, ready, 15, 7, savgol_filter)
    assert np.all(out[0] < q[0])
    np.testing.assert_array_equal(out[15], q[15])

