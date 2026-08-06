"""CPU regressions for the executable ActionBall profile pinner."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


WHOLE_BODY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WHOLE_BODY_ROOT.parents[1]
PINNER = WHOLE_BODY_ROOT / "scripts" / "pin_action_ball_profile_contracts.py"
MDP_REL = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
TABLE_GEOMETRY_REL = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py"
)
SOLVER_SOURCES = (
    "hope_commands.py",
    "continuous_questions.py",
    "racket_contact_geometry.py",
    "stroke_adapt_torch.py",
    "virtual_ball.py",
    "counter_rally.py",
    "counter_rally_torch.py",
)
# Solver profile v3 pins a per-symbol semantic surface, so the pinner also reads
# the surface module and ``strike_spec_torch.py`` (whose fixed-direction seed
# every inverse solve runs through and which used to be in no pin at all).  They
# are not in the seven-name byte map the document publishes as provenance.
SEMANTIC_SURFACE_SOURCES = (
    "action_ball_solver_semantic_surface.py",
    "strike_spec_torch.py",
)


def _copy_minimal_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    source_mdp = REPO_ROOT / MDP_REL
    target_mdp = root / MDP_REL
    target_mdp.mkdir(parents=True)
    for name in SOLVER_SOURCES + SEMANTIC_SURFACE_SOURCES:
        shutil.copy2(source_mdp / name, target_mdp / name)

    target_geometry = root / TABLE_GEOMETRY_REL
    target_geometry.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / TABLE_GEOMETRY_REL, target_geometry)

    venue = root / "configs" / "ball_physics_venue.yaml"
    venue.parent.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "configs" / "ball_physics_venue.yaml",
        venue,
    )
    return root


def _run(
    root: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PINNER),
            "--repo-root",
            str(root),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_real_cli_pins_exact_face_payload_and_all_seven_sources(tmp_path):
    root = _copy_minimal_repo(tmp_path)
    result = _run(root)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["schema_version"] == 1
    assert report["kind"] == "whole_body_tracking.action_ball.profile_pins"
    assert "source_rev" not in report
    assert "repo_root" not in report
    assert report["venue_yaml"] == "configs/ball_physics_venue.yaml"
    assert report["source_authority"]["authority"] == (
        "uncommitted_worktree_diagnostic_only_v1"
    )
    assert report["source_authority"]["commit_binding"] == "none"
    assert set(report["solver_implementation_source_sha256"]) == set(
        SOLVER_SOURCES
    )
    geometry = report["contact_geometry"]
    assert geometry["payload"]["kind"] == "exact_face_contact_v2"
    assert geometry["sha256"] == (
        report["solver_payload"]["contact_geometry"]["sha256"]
    )
    assert geometry["payload"] == (
        report["solver_payload"]["contact_geometry"]["payload"]
    )


def test_real_cli_rejects_tampered_declared_geometry_payload_sha(tmp_path):
    """Appending a second, lying GEOMETRY_SOURCE_SHA256 must not mint a pin.

    Two independent refusals now cover this, and either is a pass: the geometry
    self-seal check ("the declared SHA does not match the canonical payload"),
    and -- reached first, because the semantic surface is built before the
    contract -- the surface's duplicate-definition refusal.  A file that defines
    the same symbol twice cannot be digested per symbol without silently picking
    a winner, so the surface refuses instead of guessing.
    """

    root = _copy_minimal_repo(tmp_path)
    geometry_source = root / MDP_REL / "racket_contact_geometry.py"
    with geometry_source.open("a", encoding="utf-8") as handle:
        handle.write("\nGEOMETRY_SOURCE_SHA256 = '0' * 64\n")

    result = _run(root)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert (
        (
            "GEOMETRY_SOURCE_SHA256 does not match its canonical "
            "GEOMETRY_SOURCE_PAYLOAD"
        )
        in output
        or "defines GEOMETRY_SOURCE_SHA256 more than once" in output
    ), output


def test_source_rev_uses_historical_contracts_not_worktree(tmp_path):
    root = _copy_minimal_repo(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "add", "."],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "fixture"],
        check=True,
    )
    baseline = _run(root, "--source-rev", "HEAD")
    assert baseline.returncode == 0, baseline.stderr
    baseline_document = json.loads(baseline.stdout)
    assert "source_rev" not in baseline_document
    assert baseline_document["source_authority"]["authority"] == (
        "external_exact_commit_subset_blob_map_v1"
    )
    assert baseline_document["source_authority"]["commit_binding"] == (
        "external_preexec_immutable_launch_capsule_v1"
    )

    commands = root / MDP_REL / "hope_commands.py"
    source = commands.read_text(encoding="utf-8")
    source = source.replace(
        "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION = 2",
        "_ACTION_BALL_SOLVER_PROFILE_SCHEMA_VERSION = 77",
        1,
    )
    commands.write_text(source, encoding="utf-8")
    venue = root / "configs/ball_physics_venue.yaml"
    venue.write_text(
        venue.read_text(encoding="utf-8").replace(
            "k_d: 0.1261",
            "k_d: 9.1261",
            1,
        ),
        encoding="utf-8",
    )
    geometry = root / MDP_REL / "racket_contact_geometry.py"
    geometry.write_text(
        geometry.read_text(encoding="utf-8").replace(
            'EXACT_FACE_CONTACT_KIND = "exact_face_contact_v2"',
            'EXACT_FACE_CONTACT_KIND = "worktree_tamper"',
            1,
        ),
        encoding="utf-8",
    )

    reproduced = _run(root, "--source-rev", "HEAD")
    assert reproduced.returncode == 0, reproduced.stderr
    assert json.loads(reproduced.stdout) == baseline_document
