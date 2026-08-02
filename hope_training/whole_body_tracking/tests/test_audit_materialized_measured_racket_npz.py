from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "audit_materialized_measured_racket_npz.py"
SPEC = importlib.util.spec_from_file_location("measured_racket_auditor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_so3_audit_observes_twist_around_face_normal():
    face = np.asarray([[0.0, 1.0, 0.0]])
    measured_long = np.asarray([[1.0, 0.0, 0.0]])
    twisted_long = np.asarray([[0.0, 0.0, 1.0]])

    measured = MODULE._orientation(measured_long, face)
    twisted = MODULE._orientation(twisted_long, face)

    # Face-only error would be zero, while the full orientation exposes the 90 degree twist.
    np.testing.assert_allclose(MODULE._so3_error_deg(twisted, measured), [90.0])


def test_auditor_uses_diagonal_butt_to_blade_axis_not_site_local_x():
    np.testing.assert_allclose(
        MODULE.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL,
        [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        atol=0.0,
    )
    assert not np.array_equal(
        MODULE.ROBOT_BUTT_TO_BLADE_AXIS_LOCAL, np.asarray([1.0, 0.0, 0.0])
    )


def test_low_speed_does_not_fake_a_valid_velocity_direction():
    position = np.zeros((5, 3), dtype=np.float64)
    direction, relative, valid = MODULE._velocity_errors(position, position, 50.0)

    assert not valid.any()
    np.testing.assert_allclose(direction, 0.0)
    np.testing.assert_allclose(relative, 0.0)


def test_audit_report_publish_is_atomic_and_no_replace(tmp_path):
    path = tmp_path / "audit.json"
    MODULE._atomic_json_no_replace(path, {"admitted": True})
    assert path.is_file()
    try:
        MODULE._atomic_json_no_replace(path, {"admitted": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("audit completion report must be no-replace")
    assert list(tmp_path.iterdir()) == [path]
