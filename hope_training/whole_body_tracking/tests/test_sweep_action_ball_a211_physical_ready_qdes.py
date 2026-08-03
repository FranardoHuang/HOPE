from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/sweep_action_ball_a211_physical_ready_qdes.py"
)
SPEC = importlib.util.spec_from_file_location("_a211_ready_sweep_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def _artifact() -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    names[5] = sweep.JOINT_NAME
    unsigned = {
        "schema_version": 2,
        "kind": sweep.ARTIFACT_KIND,
        "action_id": "take061",
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "robot": {"family": "AgiBot A3", "joint_names": names},
        "physical_ready": {
            "root_pos_w_m": [0.0, 0.0, 1.0],
            "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_pos_rad": [0.0] * 31,
            "joint_vel_radps": [0.0] * 31,
        },
        "teacher_reference": {"motion_sha256": "1" * 64},
        "runtime_plant": {
            "default_joint_pos_rad": [0.0] * 31,
            "action_scale_rad": [1.0] * 31,
            "joint_stiffness": [50.0] * 31,
            "joint_effort_limits": [100.0] * 31,
            "qdes_joint_pos_limits": [[-0.5, 0.5] for _ in range(31)],
            "finite_projection_soft_envelope_inset_fraction": 0.1,
        },
        "hold_candidate": {
            "hold_qdes_joint_pos_rad": [0.0] * 31,
            "normalized_actor_action": [0.0] * 31,
            "hold_qdes_mode": "fresh_static_lp",
            "selected_hold_authority": {},
            "semantics": "old",
            "solver_report_role": "selected_hold_solution",
        },
        "sources": {"stable_motion": {"sha256": "1" * 64}},
    }
    return {**unsigned, "content_sha256": sweep.canonical_sha256(unsigned)}


def test_candidate_changes_only_hold_candidate_and_content_seal():
    base = _artifact()
    frozen = copy.deepcopy(base)
    candidate, metadata = sweep.derive_candidate(base, 0.12)
    assert base == frozen
    for key in base:
        if key not in {"hold_candidate", "content_sha256"}:
            assert candidate[key] == base[key]
    index = candidate["robot"]["joint_names"].index(sweep.JOINT_NAME)
    assert candidate["hold_candidate"]["hold_qdes_joint_pos_rad"][index] == 0.12
    assert candidate["hold_candidate"]["normalized_actor_action"][index] == 0.12
    assert metadata == {
        "candidate_id": "waist_roll_+0.12",
        "waist_roll_offset_rad": 0.12,
        "waist_roll_qdes_rad": 0.12,
        "maximum_initial_pd_effort_ratio": 0.06,
    }
    assert candidate["content_sha256"] != base["content_sha256"]
    sweep._verify_seal(candidate, name="candidate")


def test_candidate_refuses_qdes_outside_soft_envelope():
    with pytest.raises(sweep.SweepError, match="soft envelope"):
        sweep.derive_candidate(_artifact(), 0.41)


def _receipt(*, artifact_sha: str, content_sha: str, steps: int) -> dict:
    unsigned = {
        "schema_version": 1,
        "kind": sweep.RECEIPT_KIND,
        "verdict": "PASS",
        "artifact": {"sha256": artifact_sha, "content_sha256": content_sha},
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": True,
        "plant_contract_match": True,
        "active_terminations": list(sweep.HARD_TERMINATIONS),
        "completed_policy_steps": steps,
        "completed_physics_steps": steps * 4,
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "minimum_root_z_m": 1.0,
        "maximum_root_tilt_rad": 0.02,
        "joint_safety_telemetry": {
            "schema_version": 1,
            "complete": True,
            "current_actual_hard_edge_joint_count": 0,
            "substep_actual_hard_edge_joint_count": 0,
            "final_minimum_hard_gap_rad": 0.1,
        },
    }
    return {**unsigned, "content_sha256": sweep.canonical_sha256(unsigned)}


def _failed_receipt(
    *, artifact_sha: str, content_sha: str, completed_steps: int
) -> dict:
    receipt = _receipt(
        artifact_sha=artifact_sha,
        content_sha=content_sha,
        steps=completed_steps,
    )
    receipt["verdict"] = "FAIL"
    receipt["terminal_reasons"] = ["joint_actual_forbidden"]
    receipt["generic_terminated"] = True
    receipt["joint_safety_telemetry"][
        "current_actual_hard_edge_joint_count"
    ] = 1
    receipt.pop("content_sha256")
    receipt["content_sha256"] = sweep.canonical_sha256(receipt)
    return receipt


def test_receipt_requires_exact_four_substeps_per_policy_step():
    receipt = _receipt(artifact_sha="2" * 64, content_sha="3" * 64, steps=200)
    assert sweep.validate_receipt(
        receipt,
        artifact_sha="2" * 64,
        artifact_content_sha="3" * 64,
        policy_steps=200,
    )
    receipt["completed_physics_steps"] = 799
    receipt.pop("content_sha256")
    receipt["content_sha256"] = sweep.canonical_sha256(receipt)
    with pytest.raises(sweep.SweepError, match="structural"):
        sweep.validate_receipt(
            receipt,
            artifact_sha="2" * 64,
            artifact_content_sha="3" * 64,
            policy_steps=200,
        )


def test_consume_probe_accepts_published_candidate_fail_exit_two(tmp_path):
    receipt_path = tmp_path / "failed.json"
    receipt = _failed_receipt(
        artifact_sha="2" * 64,
        content_sha="3" * 64,
        completed_steps=62,
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    loaded, passed = sweep._consume_probe_result(
        SimpleNamespace(returncode=2),
        receipt_path=receipt_path,
        stage="full",
        candidate_id="waist_roll_+0.00",
        artifact_sha="2" * 64,
        artifact_content_sha="3" * 64,
        policy_steps=200,
    )
    assert loaded == receipt
    assert passed is False


def test_consume_probe_rejects_process_receipt_verdict_disagreement(tmp_path):
    receipt_path = tmp_path / "failed.json"
    receipt = _failed_receipt(
        artifact_sha="2" * 64,
        content_sha="3" * 64,
        completed_steps=62,
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sweep.SweepError, match="verdict differs"):
        sweep._consume_probe_result(
            SimpleNamespace(returncode=0),
            receipt_path=receipt_path,
            stage="full",
            candidate_id="waist_roll_+0.00",
            artifact_sha="2" * 64,
            artifact_content_sha="3" * 64,
            policy_steps=200,
        )


@pytest.mark.parametrize("failure_stage", ("short", "full"))
def test_run_continues_after_candidate_fail(
    tmp_path, monkeypatch, failure_stage
):
    root = tmp_path / "checkout"
    root.mkdir()
    work = tmp_path / "work"
    base_path = root / "base.json"
    base_path.write_text("{}", encoding="utf-8")
    base = _artifact()
    monkeypatch.setattr(sweep, "OFFSETS_RAD", (0.0, 0.04))
    monkeypatch.setattr(sweep, "verify_exact_source", lambda *_args: None)
    monkeypatch.setattr(
        sweep,
        "load_base_artifact",
        lambda *_args: (base_path, base),
    )

    def fake_run(command, *, cwd, check):
        assert cwd == str(root)
        assert check is False
        artifact_path = Path(command[command.index("--nominal-hold") + 1])
        artifact_sha = command[command.index("--nominal-hold-sha256") + 1]
        receipt_path = Path(
            command[command.index("--nominal-hold-receipt-out") + 1]
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        is_full = receipt_path.name.startswith("full")
        first_candidate = "waist_roll_+0.00" in str(receipt_path)
        fail_this_probe = first_candidate and (
            (failure_stage == "full" and is_full)
            or (failure_stage == "short" and not is_full)
        )
        if fail_this_probe:
            receipt = _failed_receipt(
                artifact_sha=artifact_sha,
                content_sha=artifact["content_sha256"],
                completed_steps=62,
            )
            returncode = 2
        else:
            steps = (
                sweep.FULL_POLICY_STEPS
                if is_full
                else sweep.SHORT_POLICY_STEPS
            )
            receipt = _receipt(
                artifact_sha=artifact_sha,
                content_sha=artifact["content_sha256"],
                steps=steps,
            )
            returncode = 0
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    result = sweep.run(
        SimpleNamespace(
            repo_root=str(root),
            source_commit="a" * 40,
            base_artifact_path="base.json",
            expected_base_artifact_sha256="b" * 64,
            python="/exact/python",
            device="cuda:0",
            work_dir=str(work),
        )
    )
    assert result["verdict"] == "PASS"
    assert result["selected_candidate_id"] == "waist_roll_+0.04"
    document = json.loads(
        Path(result["result"]["path"]).read_text(encoding="utf-8")
    )
    assert [row["full_pass"] for row in document["candidates"]] == [
        False,
        True,
    ]
    assert document["candidates"][0]["short_pass"] is (
        failure_stage != "short"
    )


def test_selection_rule_prioritizes_hard_gap_before_tilt():
    better_gap = {
        "candidate_id": "b",
        "maximum_initial_pd_effort_ratio": 0.5,
        "full_receipt": {
            "joint_safety_telemetry": {"final_minimum_hard_gap_rad": 0.2},
            "maximum_root_tilt_rad": 0.5,
            "minimum_root_z_m": 0.9,
        },
    }
    prettier_pose = {
        "candidate_id": "a",
        "maximum_initial_pd_effort_ratio": 0.1,
        "full_receipt": {
            "joint_safety_telemetry": {"final_minimum_hard_gap_rad": 0.1},
            "maximum_root_tilt_rad": 0.01,
            "minimum_root_z_m": 1.1,
        },
    }
    assert min((better_gap, prettier_pose), key=sweep._rank) is better_gap
