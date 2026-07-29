"""Contract tests for the A3 stable-upper replacement producer.

Exact MuJoCo and Isaac integration remains Pod-only.  These tests protect the
pure replacement contract: only the lower twelve joints and root Z/roll/pitch
may change, while source yaw and every non-leg joint remain intact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_a3_stable_upper_motion.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "materialize_a3_stable_upper_motion", _SCRIPT
)
materializer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = materializer
_SPEC.loader.exec_module(materializer)


def _stable_contract() -> dict[str, object]:
    return {
        "root_height_m": 1.0684,
        "lower_joint_pos_rad": {
            name: 0.01 * (index + 1)
            for index, name in enumerate(materializer.LEG_JOINT_NAMES)
        },
        "provenance": {"source": "fixture"},
    }


def test_stable_replacement_preserves_nonleg_and_source_yaw() -> None:
    frames = 4
    q = np.arange(frames * 31, dtype=np.float32).reshape(frames, 31) / 100.0
    qd = -q.copy()
    root_pos = np.broadcast_to(
        np.asarray([0.2, -0.3, 0.92], dtype=np.float32), (frames, 3)
    ).copy()
    yaw = 0.7
    roll = 0.13
    pitch = -0.2
    # Compose only for a nontrivial source quaternion; the producer must retain
    # the extracted world-Z yaw and remove the tilt.
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    source_quat = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    root_quat = np.broadcast_to(source_quat, (frames, 4)).copy()

    q_out, qd_out, pos_out, quat_out, extracted_yaw = (
        materializer._replace_with_stable_stand(
            joint_pos=q,
            joint_vel=qd,
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            stable_contract=_stable_contract(),
        )
    )

    nonleg = np.asarray(materializer.NONLEG_JOINT_INDICES, dtype=np.int64)
    leg = np.asarray(materializer.LEG_JOINT_INDICES, dtype=np.int64)
    assert np.array_equal(q_out[:, nonleg], q[:, nonleg])
    assert np.array_equal(qd_out[:, nonleg], qd[:, nonleg])
    assert np.count_nonzero(qd_out[:, leg]) == 0
    assert np.array_equal(pos_out[:, :2], root_pos[:, :2])
    assert np.all(pos_out[:, 2] == np.float32(1.0684))
    assert extracted_yaw == pytest.approx(yaw, abs=2.0e-7)
    assert np.all(quat_out[:, 1:3] == 0.0)
    expected = np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float32,
    )
    assert np.array_equal(quat_out, np.broadcast_to(expected, quat_out.shape))


def test_stable_contract_rejects_missing_leg(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "root_height_m": 1.0684,
        "lower_joint_pos_rad": {
            name: 0.0 for name in materializer.LEG_JOINT_NAMES[1:]
        },
        "provenance": {"source": "fixture"},
    }
    path = tmp_path / "stable.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(
        materializer.StableUpperMaterializationError,
        match="twelve runtime leg joints",
    ):
        materializer._load_stable_contract(path)
