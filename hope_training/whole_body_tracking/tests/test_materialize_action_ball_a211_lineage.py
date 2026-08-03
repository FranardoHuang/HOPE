"""Fail-closed tests for the commit-required A211 lineage producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_action_ball_a211_lineage.py"
SPEC = importlib.util.spec_from_file_location("materialize_a211_lineage", SCRIPT)
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    raw = materializer.canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


def _sealed(value: dict) -> dict:
    return {**value, "content_sha256": materializer.canonical_sha256(value)}


def _live_safety(action_id: str, motion_sha: str, ticks: int) -> dict:
    names = ["joint_%02d" % index for index in range(31)]
    joint = {
        "schema_version": 1, "complete": True, "joint_order": names,
        "current_actual_hard_edge_joint_count": 0,
        "current_actual_hard_edge_joint_names": [],
        "substep_actual_hard_edge_joint_count": 0,
        "substep_actual_hard_edge_joint_names": [],
        "final_minimum_hard_gap_rad": 0.05,
        "preterminal_joint_pos_rad": [0.0] * 31,
        "preterminal_joint_vel_radps": [0.0] * 31,
        "final_joint_pos_rad": [0.0] * 31,
        "final_joint_vel_radps": [0.0] * 31,
        "hard_lower_rad": [-1.0] * 31,
        "hard_upper_rad": [1.0] * 31,
    }
    unsigned = {
        "schema_version": 1,
        "kind": materializer._L.FRAME0_LIVE_RECEIPT_KIND,
        "verdict": "PASS", "action_id": action_id,
        "motion_sha256": motion_sha,
        "teacher_reference_unchanged": True,
        "teacher_physical_birth_separated": False,
        "candidate_physical_birth_written": True,
        "candidate_hold_qdes_and_delay_history_installed": True,
        "plant_contract_match": True,
        "active_terminations": list(materializer._L.HARD_TERMINATION_UNION),
        "requested_duration_s": ticks * materializer._L.POLICY_DT_S,
        "completed_duration_s": ticks * materializer._L.POLICY_DT_S,
        "completed_policy_steps": ticks, "completed_physics_steps": ticks * 4,
        "terminal_reasons": [], "generic_terminated": False,
        "generic_truncated": False, "minimum_root_z_m": 0.9,
        "maximum_root_tilt_rad": 0.1, "both_feet_contact_fraction": 1.0,
        "joint_safety_telemetry": joint,
        "screenshots": [
            {"label": label, "sha256": ("%x" % (index + 1)) * 64}
            for index, label in enumerate((
                "raw_env_reset", "physical_ready_after_reset_write",
                "after_step_1", "after_step_10", "final",
            ))
        ],
    }
    return _sealed(unsigned)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _write_motion(path: Path) -> dict[str, list[float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_pos = np.asarray(
        [
            [[-0.125, 0.375, 0.8125], [1.0, 2.0, 3.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        ],
        dtype=np.float32,
    )
    root_quat = np.asarray(
        [
            [[0.5, 0.5, -0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    joint_pos = (
        np.arange(62, dtype=np.float32).reshape(2, 31) / np.float32(17.0)
    )
    np.savez(
        path,
        body_names=np.asarray(["pelvis_link", "torso_link"]),
        body_pos_w=root_pos,
        body_quat_w=root_quat,
        joint_pos=joint_pos,
    )
    return {
        "root_pos_w_m": root_pos[0, 0].tolist(),
        "root_quat_wxyz": root_quat[0, 0].tolist(),
        "root_lin_vel_w_mps": [0.0, 0.0, 0.0],
        "root_ang_vel_w_radps": [0.0, 0.0, 0.0],
        "joint_pos_rad": joint_pos[0].tolist(),
        "joint_vel_radps": [0.0] * 31,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / ".gitignore").write_text("vendor_assets/\n", encoding="utf-8")
    motion_path = root / "assets/motion.npz"
    frame0 = _write_motion(motion_path)
    motion_sha = _sha(motion_path.read_bytes())
    action_id = "take_061_unit04_bh"
    dynamic = _sealed(
        {
            "schema_version": 1,
            "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
            "action_id": action_id,
            "teacher_reference": {"motion_sha256": motion_sha},
        }
    )
    hold = _sealed(
        {
            "schema_version": 1,
            "kind": "nominal_hold_fixture",
            "verdict": "PASS",
            "action_id": action_id,
            "motion_sha256": motion_sha,
        }
    )
    dynamic_sha = _write_json(root / "configs/dynamic.json", dynamic)
    hold_sha = _write_json(root / "configs/hold.json", hold)
    frame0_artifact = _sealed(
        {
            "schema_version": 1,
            "kind": materializer._L.FRAME0_EXACT_ARTIFACT_KIND,
            "diagnostic_unauthorized": True,
            "source_kind": materializer._L.FRAME0_EXACT_SOURCE_KIND,
            "action_id": action_id,
            "motion_sha256": motion_sha,
            "task_close_ticks": 200,
            "policy_dt_s": materializer._L.POLICY_DT_S,
            "wait_schedule_canonical_sha256": materializer._L.WAIT_SCHEDULE[
                "canonical_sha256"
            ],
            "frame0": frame0,
        }
    )
    frame0_artifact_sha = _write_json(
        root / "configs/frame0_exact_artifact.json", frame0_artifact
    )
    for source_path in materializer._L.FRAME0_RECEIPT_PROBE_SOURCE_PATHS:
        path = root / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exact probe fixture\n", encoding="utf-8")
    _git(
        root, "add", ".gitignore", "assets/motion.npz", "configs/dynamic.json",
        "configs/hold.json", "configs/frame0_exact_artifact.json",
        *materializer._L.FRAME0_RECEIPT_PROBE_SOURCE_PATHS,
    )
    _git(root, "commit", "-m", "tracked frame0 artifact inputs")
    artifact_source_commit = _git(root, "rev-parse", "HEAD")
    live = _live_safety(action_id, motion_sha, 200)
    frame0_receipt = _sealed(
        {
            "schema_version": 1,
            "kind": materializer._L.FRAME0_EXACT_RECEIPT_KIND,
            "diagnostic_unauthorized": True,
            "source_kind": materializer._L.FRAME0_EXACT_SOURCE_KIND,
            "verdict": "PASS",
            "action_id": action_id,
            "motion_sha256": motion_sha,
            "artifact_file_sha256": frame0_artifact_sha,
            "artifact_content_sha256": frame0_artifact["content_sha256"],
            "artifact_source_commit": artifact_source_commit,
            "probe_source_commit": artifact_source_commit,
            "plant_template_file_sha256": "1" * 64,
            "plant_template_content_sha256": "2" * 64,
            "probe_input_file_sha256": "3" * 64,
            "probe_input_content_sha256": "4" * 64,
            "live_safety_evidence_file_sha256": _sha(materializer.canonical_bytes(live)),
            "live_safety_evidence_content_sha256": live["content_sha256"],
            "live_safety_evidence": live,
            "task_close_ticks": 200,
            "policy_dt_s": materializer._L.POLICY_DT_S,
            "wait_schedule_canonical_sha256": materializer._L.WAIT_SCHEDULE[
                "canonical_sha256"
            ],
        }
    )
    frame0_receipt_sha = _write_json(
        root / "configs/frame0_exact_receipt.json", frame0_receipt
    )
    _git(root, "add", "configs/frame0_exact_receipt.json")
    _git(root, "commit", "-m", "track frame0 exact receipt")
    source_commit = _git(root, "rev-parse", "HEAD")

    manifest = {
        "schema_version": 3,
        "action_order": [action_id],
        "mobility_mode": "no_move",
        "actions": [{
            "action_id": action_id, "action_uid": 7,
            "motion_path": "assets/motion.npz", "motion_sha256": motion_sha,
        }],
        "solver_profile_sha256": "1" * 64,
        "physics_profile_sha256": "2" * 64,
    }
    manifest_sha = _write_json(root / "vendor_assets/manifest.json", manifest)
    tape_unsigned = {
        "schema_version": 1,
        "kind": "action_ball_n1_immutable_single_question_tape",
        "diagnostic_unauthorized": True,
        "question": {"action_uid": 7, "motion_sha256": motion_sha, "physics_sha256": "2" * 64},
    }
    tape = {**tape_unsigned, "canonical_sha256": materializer.canonical_sha256(tape_unsigned)}
    tape_sha = _write_json(root / "vendor_assets/tape.json", tape)
    bundle = {
        "schema_version": 1,
        "artifact_type": "measured_action_ball_n1_diagnostic_bundle_v1",
        "action_id": action_id, "action_uid": 7, "measured_uid": "Take_061_unit04_BH",
        "target_recipe": "current_lm",
        "target_validity": {"order": ["position", "velocity", "face"], "mask": [True, True, True]},
        "immutable_tape": {"path": "vendor_assets/tape.json", "sha256": tape_sha},
        "motion": {"path": "assets/motion.npz", "sha256": motion_sha},
        "claims": {"diagnostic_unauthorized": True},
        "runtime_contract": {
            "physical_ball_semantics": materializer._L.PHYSICAL_BALL_SEMANTICS,
            "reset_inverse_solve": False, "target_source": "immutable_tape",
        },
    }
    bundle_sha = _write_json(root / "vendor_assets/bundle.json", bundle)
    pins = {
        "bundle": bundle_sha, "tape": tape_sha, "manifest": manifest_sha,
        "motion": motion_sha, "dynamic": dynamic_sha, "hold": hold_sha,
        "frame0_artifact": frame0_artifact_sha,
        "frame0_receipt": frame0_receipt_sha,
    }
    return root, pins, source_commit


def _argv(root: Path, pins: dict[str, str], commit: str, output: str) -> list[str]:
    return [
        "--repo-root", str(root), "--source-commit", commit,
        "--bundle-path", "vendor_assets/bundle.json", "--expected-bundle-sha256", pins["bundle"], "--bundle-explicit",
        "--immutable-tape-path", "vendor_assets/tape.json", "--expected-immutable-tape-sha256", pins["tape"], "--immutable-tape-explicit",
        "--action-manifest-path", "vendor_assets/manifest.json", "--expected-action-manifest-sha256", pins["manifest"], "--action-manifest-explicit",
        "--motion-path", "assets/motion.npz", "--expected-motion-sha256", pins["motion"],
        "--dynamic-ready-artifact-path", "configs/dynamic.json", "--expected-dynamic-ready-artifact-sha256", pins["dynamic"],
        "--dynamic-ready-nominal-receipt-path", "configs/hold.json", "--expected-dynamic-ready-nominal-receipt-sha256", pins["hold"],
        "--frame0-exact-artifact-path", "configs/frame0_exact_artifact.json", "--expected-frame0-exact-artifact-sha256", pins["frame0_artifact"],
        "--frame0-exact-receipt-path", "configs/frame0_exact_receipt.json", "--expected-frame0-exact-receipt-sha256", pins["frame0_receipt"],
        "--output", output,
    ]


def _recommit_frame0_artifact(
    root: Path, pins: dict[str, str], mutate
) -> str:
    artifact_path = root / "configs/frame0_exact_artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact.pop("content_sha256")
    mutate(artifact)
    artifact = _sealed(artifact)
    pins["frame0_artifact"] = _write_json(artifact_path, artifact)
    _git(root, "add", "configs/frame0_exact_artifact.json")
    _git(root, "commit", "-m", "replace resealed frame0 artifact")
    artifact_source_commit = _git(root, "rev-parse", "HEAD")

    receipt_path = root / "configs/frame0_exact_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("content_sha256")
    receipt.update(
        {
            "artifact_file_sha256": pins["frame0_artifact"],
            "artifact_content_sha256": artifact["content_sha256"],
            "artifact_source_commit": artifact_source_commit,
        }
    )
    receipt = _sealed(receipt)
    pins["frame0_receipt"] = _write_json(receipt_path, receipt)
    _git(root, "add", "configs/frame0_exact_receipt.json")
    _git(root, "commit", "-m", "bind replacement frame0 artifact")
    return _git(root, "rev-parse", "HEAD")


def test_explicit_fresh_chain_is_canonical_but_not_launchable_until_committed(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    output = "configs/a211/fresh_lineage.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 0
    lineage_path = root / output
    raw = lineage_path.read_bytes()
    lineage = json.loads(raw)
    assert raw == materializer.canonical_bytes(lineage) + b"\n"
    assert lineage["schema_version"] == 2
    assert lineage["actor_layout_identity"] == materializer._L._actor_layout_identity()
    assert lineage["bundle"] == {"path": "vendor_assets/bundle.json", "sha256": pins["bundle"]}
    pin = {"path": output, "sha256": _sha(raw)}
    with pytest.raises(materializer._L.LaunchRefused, match="not tracked"):
        materializer._L._validate_lineage(root, commit, pin)
    _git(root, "add", "-f", "vendor_assets/bundle.json", "vendor_assets/tape.json", "vendor_assets/manifest.json")
    _git(root, "add", output)
    _git(root, "commit", "-m", "commit fresh lineage closure")
    committed = _git(root, "rev-parse", "HEAD")
    accepted = materializer._L._validate_lineage(root, committed, pin)
    assert accepted["lineage_sha256"] == pin["sha256"]


def test_rejects_bad_tape_semantic_seal_without_creating_output(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    tape_path = root / "vendor_assets/tape.json"
    tape = json.loads(tape_path.read_text(encoding="utf-8"))
    tape["canonical_sha256"] = "0" * 64
    pins["tape"] = _write_json(tape_path, tape)
    output = "configs/a211/refused.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 2
    assert not (root / output).exists()


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("root_pos_w_m", 1),
        ("root_quat_wxyz", 2),
        ("joint_pos_rad", 7),
    ),
)
def test_rejects_resealed_frame0_state_that_differs_from_pinned_motion(
    tmp_path, field, index
):
    root, pins, _commit = _fixture(tmp_path)

    def mutate(artifact):
        artifact["frame0"][field][index] += 0.125

    commit = _recommit_frame0_artifact(root, pins, mutate)
    output = "configs/a211/wrong-resealed-state.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 2
    assert not (root / output).exists()


@pytest.mark.parametrize("mutation", ("extra", "missing", "integer_zero"))
def test_rejects_non_exact_frame0_payload_contract(tmp_path, mutation):
    root, pins, _commit = _fixture(tmp_path)

    def mutate(artifact):
        if mutation == "extra":
            artifact["frame0"]["units"] = "SI"
        elif mutation == "missing":
            artifact["frame0"].pop("root_quat_wxyz")
        else:
            artifact["frame0"]["root_lin_vel_w_mps"][0] = 0

    commit = _recommit_frame0_artifact(root, pins, mutate)
    output = "configs/a211/bad-payload-%s.json" % mutation
    assert materializer.main(_argv(root, pins, commit, output)) == 2
    assert not (root / output).exists()


def test_accepts_same_chain_after_inputs_are_tracked_at_source_commit(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    _git(root, "add", "-f", "vendor_assets/bundle.json", "vendor_assets/tape.json", "vendor_assets/manifest.json")
    _git(root, "commit", "-m", "track fresh closure")
    tracked_commit = _git(root, "rev-parse", "HEAD")
    argv = _argv(root, pins, tracked_commit, "configs/a211/tracked_lineage.json")
    argv = [value for value in argv if value not in ("--bundle-explicit", "--immutable-tape-explicit", "--action-manifest-explicit")]
    assert materializer.main(argv) == 0


def test_accepts_exact_tracked_pretty_dynamic_receipts(tmp_path):
    root, pins, _commit = _fixture(tmp_path)
    for key, relative in (
        ("dynamic", "configs/dynamic.json"),
        ("hold", "configs/hold.json"),
    ):
        path = root / relative
        document = json.loads(path.read_text(encoding="utf-8"))
        raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(raw)
        pins[key] = _sha(raw)
    _git(root, "add", "configs/dynamic.json", "configs/hold.json")
    _git(root, "commit", "-m", "preserve exact tracked pretty receipts")
    commit = _git(root, "rev-parse", "HEAD")
    assert materializer.main(
        _argv(root, pins, commit, "configs/a211/pretty_receipt_lineage.json")
    ) == 0


def test_refuses_ignored_lineage_destination(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    assert materializer.main(_argv(root, pins, commit, "vendor_assets/lineage.json")) == 2
