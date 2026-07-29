"""Profile-identity regressions for the cross-engine physics audit."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
AUDIT_PATH = SCRIPTS / "audit_action_ball_cross_engine_physics.py"
PINNER = SCRIPTS / "pin_action_ball_profile_contracts.py"
SPEC = importlib.util.spec_from_file_location(
    "action_ball_cross_engine_physics_audit_under_test",
    AUDIT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _pinned_worktree() -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(PINNER),
            "--repo-root",
            str(REPO),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _promote_external_authority(raw: dict) -> dict:
    source_map_sha = AUDIT._canonical_sha256(
        raw["solver_implementation_source_sha256"]
    )
    raw["source_authority"] = {
        "schema_version": 1,
        "authority": "external_exact_commit_subset_blob_map_v1",
        "commit_binding": "external_preexec_immutable_launch_capsule_v1",
        "embedded_commit": False,
        "source_blob_map_sha256": source_map_sha,
    }
    return raw


def test_cross_engine_audit_rejects_checked_in_worktree_pseudorevision():
    raw = json.loads(
        (REPO / "configs/action_ball_profile_pins_20260728.json").read_text()
    )
    with pytest.raises(
        AUDIT.PhysicsEquivalenceError,
        match=r"source_rev.*self-reference.*forbidden",
    ):
        AUDIT._validate_formal_profile_pins(REPO, raw)


def test_cross_engine_audit_uses_same_formal_authority_gate_as_teacher_gate():
    worktree = _pinned_worktree()
    with pytest.raises(
        AUDIT.PhysicsEquivalenceError,
        match="source authority",
    ):
        AUDIT._validate_formal_profile_pins(REPO, worktree)

    validated = AUDIT._validate_formal_profile_pins(
        REPO,
        _promote_external_authority(worktree),
    )
    assert len(validated["solver_implementation_source_sha256"]) == 5
    assert validated["source_authority"]["authority"] == (
        "external_exact_commit_subset_blob_map_v1"
    )
