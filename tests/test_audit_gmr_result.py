from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_gmr_result.py"
SPEC = importlib.util.spec_from_file_location("audit_gmr_result", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _payload(n: int = 5) -> dict:
    return {
        "fps": 30.0,
        "root_pos": np.zeros((n, 3), dtype=np.float32),
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
        "dof_pos": np.zeros((n, 31), dtype=np.float32),
    }


def test_validate_payload_accepts_exact_diagnostic_contract():
    report = AUDIT.validate_payload(_payload(), 5)
    assert report["actual_frames"] == 5
    assert report["fps"] == 30.0
    assert report["root_rotation_convention"] == "xyzw"
    assert report["shapes"]["dof_pos"] == [5, 31]
    assert report["finite_elements"] == 5 * (3 + 4 + 31) + 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 50.0, "expected exactly 30"),
        ("root_pos", np.zeros((4, 3)), "root_pos shape"),
        ("root_rot", np.zeros((5, 3)), "root_rot shape"),
        ("dof_pos", np.zeros((5, 30)), "dof_pos shape"),
    ],
)
def test_validate_payload_rejects_wrong_fps_or_shape(field, value, message):
    payload = _payload()
    payload[field] = value
    with pytest.raises(AUDIT.ResultError, match=message):
        AUDIT.validate_payload(payload, 5)


def test_validate_payload_rejects_nonfinite():
    payload = _payload()
    payload["dof_pos"][2, 4] = np.nan
    with pytest.raises(AUDIT.ResultError, match="non-finite"):
        AUDIT.validate_payload(payload, 5)


def test_validate_payload_rejects_nonunit_root_quaternion():
    payload = _payload()
    payload["root_rot"][2] = 0.0
    with pytest.raises(AUDIT.ResultError, match="quaternion max norm error"):
        AUDIT.validate_payload(payload, 5)


@pytest.mark.parametrize(
    "line",
    [
        "[GMR] frame-0 warm-up: 22 rounds, final max|dq| 7.81e-05 rad, task error 2.325976",
        "[warm-up] pass 17 max|Δq|=9.0e-05",
        "warmup converged after 8 iterations; final_max_dq: 0.00002",
        "frame0 WARM_UP 4/200 maximum delta dq = 1e-05",
    ],
)
def test_default_warmup_parser_accepts_converged_final_evidence(line):
    result = AUDIT.parse_warmup_evidence(line)
    assert result["parser"] == "built-in"
    assert result["rounds"] <= 200
    assert result["max_dq"] < 1e-4


def test_warmup_parser_uses_last_event_and_fails_closed():
    text = "\n".join(
        [
            "[warm-up] pass 1 max_dq=1e-2",
            "[warm-up] pass 201 max_dq=9e-5",
        ]
    )
    with pytest.raises(AUDIT.ResultError, match="rounds=201"):
        AUDIT.parse_warmup_evidence(text)
    with pytest.raises(AUDIT.ResultError, match="no parseable"):
        AUDIT.parse_warmup_evidence("optimization completed without a bound warmup line")
    with pytest.raises(AUDIT.ResultError, match="required <"):
        AUDIT.parse_warmup_evidence("warm-up pass 7 max_dq=0.0001")


def test_custom_warmup_regex_requires_named_groups_and_parses():
    custom = r"FRAME_ZERO_DONE n=(?P<rounds>\d+) residual=(?P<max_dq>[0-9.e-]+)"
    result = AUDIT.parse_warmup_evidence(
        "FRAME_ZERO_DONE n=12 residual=3e-6", custom_regex=custom
    )
    assert result["parser"] == "custom_regex"
    assert result["rounds"] == 12
    with pytest.raises(AUDIT.ResultError, match="named groups"):
        AUDIT.parse_warmup_evidence("x", custom_regex=r"(?P<rounds>\d+)")


def test_main_binds_explicit_canonical_body_shape_contract(tmp_path):
    import pickle

    result = tmp_path / "result.pkl"
    log = tmp_path / "run.log"
    report = tmp_path / "report.json"
    with result.open("wb") as handle:
        pickle.dump(_payload(), handle)
    log.write_text("[warm-up] pass 7 max_dq=9e-5\n", encoding="utf-8")
    contract = "diagnostic_same_performer_coordinatewise_median_betas_v1"
    assert AUDIT.main(
        [
            "--result",
            str(result),
            "--expected-frames",
            "5",
            "--run-log",
            str(log),
            "--body-shape-contract",
            contract,
            "--json-out",
            str(report),
        ]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["body_shape_contract"] == contract
    assert payload["formal_eligible"] is False
