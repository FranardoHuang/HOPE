"""Fail-closed tests for the production C211 lineage materializer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

import pytest


WBT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = WBT_ROOT / "scripts/materialize_action_ball_c211_lineage.py"
SPEC = importlib.util.spec_from_file_location("materialize_c211_lineage", SCRIPT)
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)

# Reuse the launcher's exact C211 authority fixture.  This deliberately avoids
# maintaining a second hand-written copy of the large zero-handoff and passive
# hold schemas while the assertions below exercise the production materializer.
LAUNCHER_TEST = WBT_ROOT / "tests/test_launch_action_ball_c211_diagnostic.py"
HELPER_SPEC = importlib.util.spec_from_file_location(
    "_c211_launcher_fixture_for_materializer", LAUNCHER_TEST
)
helper = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = helper
HELPER_SPEC.loader.exec_module(helper)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, value: dict, *, canonical: bool = True) -> str:
    if canonical:
        raw = materializer.canonical_bytes(value) + b"\n"
    else:
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sealed(value: dict) -> dict:
    return {**value, "content_sha256": materializer.canonical_sha256(value)}


def _fixture(tmp_path: Path, *, include_bundle: bool) -> tuple[Path, dict, str]:
    root = (tmp_path / "repo").resolve()
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "C211 Test")
    (root / ".gitignore").write_text(
        "ignored/\n__pycache__/\n*.pyc\n", encoding="utf-8"
    )

    lineage = helper._lineage(root)
    bundle_path = root / lineage["bundle"]["path"]
    if not include_bundle:
        bundle_path.unlink()

    repo = Path(__file__).resolve().parents[3]
    runtime_paths = []
    for relative, markers in materializer._L.C211_ORACLE_HOOK_SOURCE_MARKERS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\n".join(markers) + b"\n")
        runtime_paths.append(relative)

    # The DR-L0 binding is code-owned: the manifest is only authoritative when
    # it resolves against the exact training-contract bytes in the source
    # commit.  Track the selected C leaf and its inherited parent as part of the
    # same fixture closure even though the binding validator consumes their
    # canonical paths from the manifest rather than parsing Hydra here.
    dr_source_paths = (
        materializer._L.TRAINING_CONTRACT_SOURCE,
        materializer._L.TASK_PROFILE_SOURCE,
        materializer._L.RETAINED_TASK_PROFILE_PARENT_SOURCE,
        materializer._L._FRAME0.DR_L0_MANIFEST_SOURCE,
    )
    for relative in dr_source_paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, destination)

    first_commit_paths = [
        ".gitignore",
        lineage["motion"]["path"],
        lineage["action_manifest"]["path"],
        lineage["initial_center_task_receipt"]["path"],
        lineage["dynamic_ready_artifact"]["path"],
        lineage["dynamic_ready_nominal_receipt"]["path"],
        lineage["teacher_frame0_artifact"]["path"],
        *runtime_paths,
        *dr_source_paths,
    ]
    _git(root, "add", *first_commit_paths)
    _git(root, "commit", "-m", "track split-ready C211 source authorities")
    authority_commit = _git(root, "rev-parse", "HEAD")
    dr_l0_manifest = materializer._L._FRAME0._dr_l0_manifest_binding(
        root,
        authority_commit,
        family="C",
        task_profile=materializer._L.TASK_PROFILE_ID,
    )
    lineage["dr_l0_manifest"] = dr_l0_manifest

    if include_bundle:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle.pop("content_sha256")
        bundle["dr_l0_manifest"] = dr_l0_manifest
        bundle = _sealed(bundle)
        lineage["bundle"]["sha256"] = _write(bundle_path, bundle)
        _git(root, "add", lineage["bundle"]["path"])
        _git(root, "commit", "-m", "bind C211 DR-L0 bundle closure")
        source_commit = _git(root, "rev-parse", "HEAD")
    else:
        source_commit = authority_commit
    assert not _git(root, "status", "--porcelain=v1")
    return root, lineage, source_commit


def _argv(
    root: Path,
    lineage: dict,
    source_commit: str,
    *,
    output: str,
    bundle_output: Optional[str] = None,
) -> list[str]:
    argv = [
        "--repo-root",
        str(root),
        "--source-commit",
        source_commit,
    ]
    if bundle_output is None:
        argv.extend(
            [
                "--bundle-path",
                lineage["bundle"]["path"],
                "--expected-bundle-sha256",
                lineage["bundle"]["sha256"],
            ]
        )
    else:
        argv.extend(["--bundle-output", bundle_output])
    for key, flag in (
        ("motion", "motion"),
        ("action_manifest", "action-manifest"),
        ("initial_center_task_receipt", "initial-center-task-receipt"),
        ("dynamic_ready_artifact", "dynamic-ready-artifact"),
        (
            "dynamic_ready_nominal_receipt",
            "dynamic-ready-raw-nominal-receipt",
        ),
        ("teacher_frame0_artifact", "teacher-frame0-artifact"),
    ):
        argv.extend(
            [
                "--%s-path" % flag,
                lineage[key]["path"],
                "--expected-%s-sha256" % flag,
                lineage[key]["sha256"],
            ]
        )
    argv.extend(["--output", output])
    return argv


def _run(argv: list[str]) -> dict:
    return materializer.materialize(materializer._parser().parse_args(argv))


def test_generated_bundle_and_lineage_are_commit_required_then_launcher_accepted(
    tmp_path,
):
    root, source, source_commit = _fixture(tmp_path, include_bundle=False)
    bundle_output = "outputs/c211.direct-ball.bundle.json"
    lineage_output = "outputs/c211.direct-ball.lineage.json"
    argv = _argv(
        root,
        source,
        source_commit,
        output=lineage_output,
        bundle_output=bundle_output,
    )

    result = _run(argv)
    assert result["status"] == "MATERIALIZED_COMMIT_REQUIRED"
    assert result["diagnostic_unauthorized"] is True
    assert result["launch_authorized"] is False
    assert result["bundle_mode"] == "materialized"
    bundle_raw = (root / bundle_output).read_bytes()
    lineage_raw = (root / lineage_output).read_bytes()
    bundle = json.loads(bundle_raw)
    lineage = json.loads(lineage_raw)
    assert bundle_raw == materializer.canonical_bytes(bundle) + b"\n"
    assert lineage_raw == materializer.canonical_bytes(lineage) + b"\n"
    assert bundle["content_sha256"] == materializer.canonical_sha256(
        {key: value for key, value in bundle.items() if key != "content_sha256"}
    )
    assert lineage["kind"] == materializer._L.LINEAGE_KIND
    assert lineage["bundle"] == result["bundle"]

    # Same clean source commit plus the two exact uncommitted outputs is
    # idempotent: no bytes are rewritten and no alternate digest is minted.
    repeated = _run(argv)
    assert repeated == result
    assert (root / bundle_output).read_bytes() == bundle_raw
    assert (root / lineage_output).read_bytes() == lineage_raw

    with pytest.raises(materializer._L.LaunchRefused, match="not tracked"):
        materializer._L._validate_lineage(
            root, source_commit, result["lineage"]
        )
    _git(root, "add", bundle_output, lineage_output)
    _git(root, "commit", "-m", "commit C211 production lineage closure")
    committed = _git(root, "rev-parse", "HEAD")
    accepted = materializer._L._validate_lineage(
        root, committed, result["lineage"]
    )
    assert accepted["lineage_sha256"] == result["lineage"]["sha256"]
    assert accepted["target_recipe"] == "outcome_dense_only"
    assert accepted["target_validity_mask"] == [False, False, False]


def test_tracked_bundle_mode_crosschecks_every_explicit_source_pin(tmp_path):
    root, source, source_commit = _fixture(tmp_path, include_bundle=True)
    bundle_path = root / source["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle.pop("content_sha256")
    bundle["motion"] = {
        "path": source["motion"]["path"],
        "sha256": "b" * 64,
    }
    bundle = _sealed(bundle)
    source["bundle"]["sha256"] = _write(bundle_path, bundle)
    _git(root, "add", source["bundle"]["path"])
    _git(root, "commit", "-m", "re-seal bundle with a disconnected motion pin")
    source_commit = _git(root, "rev-parse", "HEAD")

    argv = _argv(
        root,
        source,
        source_commit,
        output="outputs/rejected-lineage.json",
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="explicit source pin closure differs",
    ):
        _run(argv)
    assert not (root / "outputs/rejected-lineage.json").exists()


def test_tracked_canonical_bundle_mode_materializes_the_same_lineage_schema(tmp_path):
    root, source, source_commit = _fixture(tmp_path, include_bundle=True)
    result = _run(
        _argv(
            root,
            source,
            source_commit,
            output="outputs/from-tracked-bundle.lineage.json",
        )
    )
    lineage = json.loads(
        (root / result["lineage"]["path"]).read_text(encoding="utf-8")
    )
    assert result["bundle_mode"] == "tracked"
    assert result["bundle"] == source["bundle"]
    assert lineage["bundle"] == source["bundle"]
    assert lineage["trainability_contract"] == (
        "action_ball_c211_fixed_midpoint_learnability_v2"
    )


@pytest.mark.parametrize("failure", ("foreign", "noncanonical"))
def test_rejects_foreign_or_noncanonical_tracked_bundle(tmp_path, failure):
    root, source, _source_commit = _fixture(tmp_path, include_bundle=True)
    bundle_path = root / source["bundle"]["path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if failure == "foreign":
        bundle.pop("content_sha256")
        bundle["actor_contract"] = "action_ball_a225"
        bundle = _sealed(bundle)
        source["bundle"]["sha256"] = _write(bundle_path, bundle)
    else:
        source["bundle"]["sha256"] = _write(
            bundle_path, bundle, canonical=False
        )
    _git(root, "add", source["bundle"]["path"])
    _git(root, "commit", "-m", "track rejected C211 bundle form")
    source_commit = _git(root, "rev-parse", "HEAD")
    argv = _argv(
        root,
        source,
        source_commit,
        output="outputs/rejected-%s.json" % failure,
    )
    expected = "foreign ABI/lineage token" if failure == "foreign" else "canonical JSON"
    with pytest.raises(materializer.MaterializationError, match=expected):
        _run(argv)
    assert not (root / ("outputs/rejected-%s.json" % failure)).exists()


def test_different_existing_bundle_output_is_never_clobbered(tmp_path):
    root, source, source_commit = _fixture(tmp_path, include_bundle=False)
    bundle_output = "outputs/c211.bundle.json"
    occupied = root / bundle_output
    occupied.parent.mkdir(parents=True)
    original = b"do not overwrite this pre-existing output\n"
    occupied.write_bytes(original)
    argv = _argv(
        root,
        source,
        source_commit,
        output="outputs/c211.lineage.json",
        bundle_output=bundle_output,
    )
    with pytest.raises(materializer.MaterializationError, match="different bytes"):
        _run(argv)
    assert occupied.read_bytes() == original
    assert not (root / "outputs/c211.lineage.json").exists()


def test_rejects_dirty_source_even_when_the_worktree_digest_is_redeclared(tmp_path):
    root, source, source_commit = _fixture(tmp_path, include_bundle=True)
    manifest_path = root / source["action_manifest"]["path"]
    raw = manifest_path.read_bytes() + b"\n"
    manifest_path.write_bytes(raw)
    source["action_manifest"]["sha256"] = hashlib.sha256(raw).hexdigest()
    argv = _argv(
        root,
        source,
        source_commit,
        output="outputs/dirty-source.json",
    )
    with pytest.raises(materializer.MaterializationError, match="source checkout is dirty"):
        _run(argv)
    assert not (root / "outputs/dirty-source.json").exists()


def test_rejects_staged_output_when_worktree_bytes_no_longer_match_index(tmp_path):
    root, source, source_commit = _fixture(tmp_path, include_bundle=False)
    bundle_output = "outputs/staged.bundle.json"
    lineage_output = "outputs/staged.lineage.json"
    argv = _argv(
        root,
        source,
        source_commit,
        output=lineage_output,
        bundle_output=bundle_output,
    )
    _run(argv)
    correct = (root / lineage_output).read_bytes()
    (root / lineage_output).write_bytes(b"staged wrong lineage\n")
    _git(root, "add", lineage_output)
    (root / lineage_output).write_bytes(correct)
    assert _git(root, "show", ":" + lineage_output) == "staged wrong lineage"
    assert (root / lineage_output).read_bytes() == correct
    with pytest.raises(materializer.MaterializationError, match="source checkout is dirty"):
        _run(argv)
    assert _git(root, "show", ":" + lineage_output) == "staged wrong lineage"
    assert (root / lineage_output).read_bytes() == correct


def test_rejects_matching_staged_output_instead_of_treating_index_as_publication(
    tmp_path,
):
    root, source, source_commit = _fixture(tmp_path, include_bundle=False)
    bundle_output = "outputs/staged-matching.bundle.json"
    lineage_output = "outputs/staged-matching.lineage.json"
    argv = _argv(
        root,
        source,
        source_commit,
        output=lineage_output,
        bundle_output=bundle_output,
    )
    _run(argv)
    correct = (root / lineage_output).read_bytes()
    _git(root, "add", lineage_output)
    assert _git(root, "show", ":" + lineage_output).encode("utf-8") + b"\n" == correct

    with pytest.raises(materializer.MaterializationError, match="source checkout is dirty"):
        _run(argv)

    assert (root / lineage_output).read_bytes() == correct


def test_cross_output_rejection_does_not_create_destination_directories(tmp_path):
    root, source, source_commit = _fixture(tmp_path, include_bundle=False)
    parent = root / "never-created" / "nested"
    shared_output = "never-created/nested/c211.json"
    assert not parent.exists()
    argv = _argv(
        root,
        source,
        source_commit,
        output=shared_output,
        bundle_output=shared_output,
    )

    with pytest.raises(
        materializer.MaterializationError,
        match="bundle and lineage outputs must differ",
    ):
        _run(argv)

    assert not parent.exists()


def test_missing_launcher_runtime_authority_is_rejected_before_publication(tmp_path):
    root, source, _source_commit = _fixture(tmp_path, include_bundle=False)
    relative = next(iter(materializer._L.C211_ORACLE_HOOK_SOURCE_MARKERS))
    path = root / relative
    path.write_bytes(b"runtime marker deliberately absent\n")
    _git(root, "add", relative)
    _git(root, "commit", "-m", "remove C211 runtime authority")
    source_commit = _git(root, "rev-parse", "HEAD")
    argv = _argv(
        root,
        source,
        source_commit,
        output="outputs/no-authority.lineage.json",
        bundle_output="outputs/no-authority.bundle.json",
    )
    assert not (root / "outputs").exists()
    with pytest.raises(
        materializer.MaterializationError,
        match="runtime authority is absent",
    ):
        _run(argv)
    assert not (root / "outputs").exists()
    assert not (root / "outputs/no-authority.lineage.json").exists()
    assert not (root / "outputs/no-authority.bundle.json").exists()
