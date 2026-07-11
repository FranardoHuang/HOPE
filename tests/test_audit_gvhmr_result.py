from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_gvhmr_result.py"
SPEC = importlib.util.spec_from_file_location("audit_gvhmr_result", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _payload(n: int = 5) -> dict:
    return {
        "smpl_params_global": {
            "body_pose": np.zeros((n, 63), dtype=np.float32),
            "betas": np.zeros((n, 10), dtype=np.float32),
            "global_orient": np.zeros((n, 3), dtype=np.float32),
            "transl": np.zeros((n, 3), dtype=np.float32),
        }
    }


def test_validate_payload_accepts_finite_expected_shapes():
    report = AUDIT.validate_payload(_payload(), 5)
    assert report["actual_frames"] == 5
    assert report["shapes"]["body_pose"] == [5, 63]
    assert report["finite_elements"] == 5 * (63 + 10 + 3 + 3)


@pytest.mark.parametrize("shape", [(10,), (1, 10), (5, 10)])
def test_validate_payload_accepts_common_betas_shapes(shape):
    payload = _payload()
    payload["smpl_params_global"]["betas"] = np.zeros(shape, dtype=np.float32)
    AUDIT.validate_payload(payload, 5)


def test_validate_payload_rejects_frame_mismatch_and_nonfinite():
    with pytest.raises(AUDIT.ResultError, match="body_pose shape"):
        AUDIT.validate_payload(_payload(4), 5)
    payload = _payload()
    payload["smpl_params_global"]["transl"][2, 0] = np.nan
    with pytest.raises(AUDIT.ResultError, match="non-finite"):
        AUDIT.validate_payload(payload, 5)
