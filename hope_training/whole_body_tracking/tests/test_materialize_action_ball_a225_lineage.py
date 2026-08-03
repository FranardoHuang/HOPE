"""Fail-closed tests for the commit-required A225 lineage producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_action_ball_a225_lineage.py"
SPEC = importlib.util.spec_from_file_location("materialize_a225_lineage", SCRIPT)
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


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / ".gitignore").write_text("vendor_assets/\n", encoding="utf-8")
    motion_path = root / "assets/motion.npz"
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"fixture-motion\n")
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
    _git(root, "add", ".gitignore", "assets/motion.npz", "configs/dynamic.json", "configs/hold.json")
    _git(root, "commit", "-m", "tracked fixture inputs")
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
        "--output", output,
    ]


def test_explicit_fresh_chain_is_canonical_but_not_launchable_until_committed(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    output = "configs/a225/fresh_lineage.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 0
    lineage_path = root / output
    raw = lineage_path.read_bytes()
    lineage = json.loads(raw)
    assert raw == materializer.canonical_bytes(lineage) + b"\n"
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
    output = "configs/a225/refused.json"
    assert materializer.main(_argv(root, pins, commit, output)) == 2
    assert not (root / output).exists()


def test_accepts_same_chain_after_inputs_are_tracked_at_source_commit(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    _git(root, "add", "-f", "vendor_assets/bundle.json", "vendor_assets/tape.json", "vendor_assets/manifest.json")
    _git(root, "commit", "-m", "track fresh closure")
    tracked_commit = _git(root, "rev-parse", "HEAD")
    argv = _argv(root, pins, tracked_commit, "configs/a225/tracked_lineage.json")
    argv = [value for value in argv if value not in ("--bundle-explicit", "--immutable-tape-explicit", "--action-manifest-explicit")]
    assert materializer.main(argv) == 0


def test_refuses_ignored_lineage_destination(tmp_path):
    root, pins, commit = _fixture(tmp_path)
    assert materializer.main(_argv(root, pins, commit, "vendor_assets/lineage.json")) == 2
