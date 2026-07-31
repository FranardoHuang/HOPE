"""Host-only adversarial tests for the A3 dual-envelope stress producer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_a3_vendor_dual_position_envelope.py"
SPEC = importlib.util.spec_from_file_location("a3_dual_envelope_probe_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "waist_roll_joint",
    "right_hip_pitch_joint",
    "waist_pitch_joint",
)
H_MECH = (
    (-1.0, 1.0),
    (-0.5, 0.7),
    (-1.1, 1.1),
    (-0.8, 0.6),
)
H_CTRL = (
    (-1.0, 1.0),
    (-0.476, 0.676),
    (-1.1, 1.1),
    (-0.772, 0.572),
)


def _tape():
    return PROBE.build_stress_tape(JOINT_NAMES, H_MECH, H_CTRL)


def _diagnostic():
    rows = []
    for joint in PROBE.STRESSED_JOINTS:
        rows.append(
            {
                "joint": joint,
                "max_abs_delta_qdot_rad_s": 3.0,
                "sides": {
                    side: {
                        "near_ctrl_edge_readback": 2,
                        "ctrl_penetration_readback": 1,
                        "ballistic_attempt_proxy": 2,
                        "capture_proxy": 1,
                        "ballistic_attempt_side_flip_proxy": 0,
                        "minimum_signed_ctrl_gap_rad": -0.001,
                        "minimum_signed_mechanical_gap_rad": 0.005,
                        "max_ctrl_penetration_dwell_readbacks": 1,
                        "nonfinite_readback_observed": False,
                    }
                    for side in PROBE.SIDES
                },
            }
        )
    return {
        "enabled": True,
        "physx_control_position_limits": {
            "enabled": True,
            "semantics": "kinematic H_ctrl proxy; not a PhysX constraint impulse getter",
            "joint_order": list(PROBE.STRESSED_JOINTS),
            "side_order": list(PROBE.SIDES),
            "ballistic_horizon_s": 0.02,
            "by_joint": rows,
        },
    }


def _observations(tape):
    rows = []
    for row in tape:
        direction = row["direction"]
        reserve = row["cage_reserve_rad"]
        if row["condition"] == "on":
            # Strictly inside H_ctrl after solver capture.
            q_after = row["h_ctrl_edge_rad"] - direction * 0.05 * reserve
        else:
            # Strictly inside H_mech but outside H_ctrl when the cage is absent.
            q_after = row["h_ctrl_edge_rad"] + direction * 0.30 * reserve
        rows.append(
            {
                "env_id": row["env_id"],
                "joint": row["joint"],
                "side": row["side"],
                "condition": row["condition"],
                "q_after_rad": q_after,
                "qdot_after_rad_s": 0.0,
                "q0_live_rad": PROBE._float32_round(row["q0_rad"]),
                "qdes_rad": PROBE._float32_round(row["q0_rad"]),
            }
        )
    return rows


def test_exact_formula_builds_eight_same_tape_on_off_cases():
    tape = _tape()
    assert len(tape) == 8
    assert [
        (row["joint"], row["side"], row["condition"])
        for row in tape
    ] == [
        (joint, side, condition)
        for joint in PROBE.STRESSED_JOINTS
        for side in PROBE.SIDES
        for condition in PROBE.CONDITIONS
    ]

    for row in tape:
        reserve = row["cage_reserve_rad"]
        assert abs(row["qdot0_rad_s"]) == pytest.approx(0.70 * reserve / 0.005)
        assert row["qdes_rad"] == row["q0_rad"]
        assert row["kinematic_mechanical_gap_rad"] == pytest.approx(0.40 * reserve)
        assert row["q0_rad"] + row["qdot0_rad_s"] * 0.005 == pytest.approx(
            row["kinematic_q_5ms_rad"]
        )

    for index in range(0, 8, 2):
        on = dict(tape[index])
        off = dict(tape[index + 1])
        assert on.pop("env_id") == index
        assert off.pop("env_id") == index + 1
        assert on.pop("condition") == "on"
        assert off.pop("condition") == "off"
        assert on == off


def test_runtime_schema_requires_four_exact_pair_aggregates():
    tape = _tape()
    runtime = PROBE.validate_runtime_result(
        tape,
        _observations(tape),
        _diagnostic(),
        physics_dt_s=0.005,
        live_limits_restored_exact=True,
    )
    assert runtime["all_rows_finite"] is True
    assert runtime["mechanical_penetration_count"] == 0
    assert len(runtime["observations"]) == 8
    assert runtime["aggregate_by_joint_side"] == [
        {
            "joint": joint,
            "side": side,
            "strict_5ms_kinematic_attempt_count": 2,
            "on_capture_count": 1,
            "off_post_ctrl_penetration_count": 1,
            "mechanical_penetration_count": 0,
            "existing_20ms_ballistic_attempt_proxy_count": 2,
            "existing_20ms_capture_proxy_count": 1,
            "post_ctrl_penetration_readback_count": 1,
        }
        for joint in PROBE.STRESSED_JOINTS
        for side in PROBE.SIDES
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda tape, obs, diag: obs[0].update(
                q_after_rad=tape[0]["h_ctrl_edge_rad"]
            ),
            "ON did not capture",
        ),
        (
            lambda tape, obs, diag: obs[1].update(
                q_after_rad=tape[1]["q0_rad"]
            ),
            "OFF did not enter",
        ),
        (
            lambda tape, obs, diag: obs[2].update(
                qdes_rad=tape[2]["qdes_rad"] + 1e-9
            ),
            "q_des differs",
        ),
        (
            lambda tape, obs, diag: obs[3].update(
                q0_live_rad=0.0,
                qdes_rad=0.0,
            ),
            "q0 differs",
        ),
        (
            lambda tape, obs, diag: diag["physx_control_position_limits"][
                "by_joint"
            ][0]["sides"]["lower"].update(capture_proxy=0),
            "expected attempt=2 capture=1 penetration=1",
        ),
        (
            lambda tape, obs, diag: diag["physx_control_position_limits"].update(
                ballistic_horizon_s=0.005
            ),
            "capture_proxy horizon must remain 20 ms",
        ),
    ),
)
def test_tampered_outcome_or_proxy_semantics_fail_closed(mutation, match):
    tape = _tape()
    observations = _observations(tape)
    diagnostic = _diagnostic()
    mutation(tape, observations, diagnostic)
    with pytest.raises(PROBE.DualEnvelopeProbeError, match=match):
        PROBE.validate_runtime_result(
            tape,
            observations,
            diagnostic,
            physics_dt_s=0.005,
            live_limits_restored_exact=True,
        )


def test_restore_failure_can_never_mint_pass():
    tape = _tape()
    runtime = PROBE.validate_runtime_result(
        tape,
        _observations(tape),
        _diagnostic(),
        physics_dt_s=0.005,
        live_limits_restored_exact=True,
    )
    common = {
        "source_commit": "a" * 40,
        "source_script_sha256": "b" * 64,
        "task": "Task",
        "motion_files": [],
        "tape": tape,
        "runtime": runtime,
        "live_limit_identity": {},
        "error": None,
    }
    passing = PROBE.build_receipt(
        **common,
        restore={"attempted": True, "exact_readback": True, "error": None},
    )
    assert passing["status"] == "PASS"
    unhashed = dict(passing)
    content_sha256 = unhashed.pop("content_sha256")
    assert content_sha256 == PROBE._sha256_bytes(
        PROBE._canonical_json_bytes(unhashed)
    )

    receipt = PROBE.build_receipt(
        **common,
        restore={
            "attempted": True,
            "exact_readback": False,
            "error": "tampered restore",
        },
    )
    assert receipt["status"] == "FAIL"
    assert receipt["training_authorized"] is False
    assert receipt["restore"]["exact_readback"] is False


def test_task_is_code_owned_and_cannot_be_overridden():
    with pytest.raises(SystemExit):
        PROBE._parse_args(
            [
                "--task",
                "Some-Other-Task-v0",
                "--motion-file",
                "/tmp/motion.npz",
                "--source-root",
                "/tmp/source",
                "--expected-source-commit",
                "a" * 40,
                "--output",
                "/tmp/out.json",
            ]
        )


def test_json_publication_is_canonical_and_no_clobber(tmp_path: Path):
    output = tmp_path / "receipt.json"
    payload = {"schema_version": 1, "status": "FAIL", "why": "test"}
    PROBE._write_json_exclusive(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert output.read_bytes() == PROBE._canonical_json_bytes(payload)
    with pytest.raises(FileExistsError):
        PROBE._write_json_exclusive(output, payload)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def test_exact_clean_head_rejects_source_tamper(tmp_path: Path):
    root = tmp_path / "source"
    script = root / "probe.py"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "probe@example.invalid")
    _git(root, "config", "user.name", "Probe Test")
    script.write_text("print('exact')\n", encoding="utf-8")
    _git(root, "add", "probe.py")
    _git(root, "commit", "-m", "exact")
    commit = _git(root, "rev-parse", "HEAD")

    assert PROBE._verify_clean_exact_checkout(root, commit, script_path=script) == commit
    script.write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="exactly clean"):
        PROBE._verify_clean_exact_checkout(root, commit, script_path=script)


def test_output_must_be_outside_source_and_isaaclab_and_absent(tmp_path: Path):
    source = tmp_path / "source"
    isaaclab = tmp_path / "IsaacLab"
    external = tmp_path / "receipts"
    source.mkdir()
    isaaclab.mkdir()
    external.mkdir()
    outside = external / "stress.json"
    assert PROBE._validate_output_path(outside, (source, isaaclab)) == outside
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="outside protected root"):
        PROBE._validate_output_path(source / "stress.json", (source, isaaclab))
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="outside protected root"):
        PROBE._validate_output_path(
            isaaclab / "stress.json", (source, isaaclab)
        )
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PROBE.DualEnvelopeProbeError, match="already exists"):
        PROBE._validate_output_path(outside, (source, isaaclab))
