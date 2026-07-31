from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_REL = Path(
    "hope_training/whole_body_tracking/scripts/"
    "materialize_a3_vendor_identity_manifest.py"
)
REGISTRY_REL = Path(
    "hope_training/whole_body_tracking/scripts/a3_vendor_action_registry.py"
)
PIN_PRODUCER_REL = Path(
    "hope_training/whole_body_tracking/scripts/"
    "pin_action_ball_profile_contracts.py"
)
MDP_REL = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp"
)
TABLE_GEOMETRY_REL = Path(
    "hope_training/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/table_tennis/geometry.py"
)
VENUE_REL = Path("configs/ball_physics_venue.yaml")
SOURCE_MANIFEST_REL = Path(
    "configs/n1_contact_20260730_stable_v2/"
    "bh_loop_c.manifest.v3.775f74183e58.json"
)
SOURCE_MANIFEST_SHA = (
    "775f74183e58683df48f5f44084e89320736d1533a4d962f43f455664830d8e5"
)
SOURCE_PROTOTYPE_REL = Path(
    "configs/n1_contact_20260730_stable_v2/"
    "bh_loop_c.upper.prototype.v2.1726d7825f1c.json"
)
MOTION_REL = Path(
    "assets/motions/fivebind_20260727/bh_loop_c_upper_stable_v2.npz"
)
BLOCK_SOURCE_MANIFEST_REL = Path(
    "configs/n1_contact_20260730_stable_v2/"
    "bh_block.manifest.v3.7b16eef89878.json"
)
BLOCK_SOURCE_MANIFEST_SHA = (
    "7b16eef898780d388e71987ebd7332f5ebbffec72a7513042860d8196b87ddea"
)
BLOCK_SOURCE_PROTOTYPE_REL = Path(
    "configs/n1_contact_20260730_stable_v2/"
    "bh_block.upper.prototype.v2.edb3a600e4fc.json"
)
BLOCK_SOURCE_PROTOTYPE_SHA = (
    "edb3a600e4fcb35a9cb69b3741da5020d733132a3dd3d28b1272a34293481f2d"
)
BLOCK_MOTION_REL = Path(
    "assets/motions/fivebind_20260727/bh_block_upper_stable_v2.npz"
)
ACTION_SOURCES = {
    "bh_loop_c": {
        "manifest": SOURCE_MANIFEST_REL,
        "prototype": SOURCE_PROTOTYPE_REL,
        "motion": MOTION_REL,
    },
    "bh_block": {
        "manifest": BLOCK_SOURCE_MANIFEST_REL,
        "prototype": BLOCK_SOURCE_PROTOTYPE_REL,
        "motion": BLOCK_MOTION_REL,
    },
}
PROFILE_REL = Path(
    "configs/a3_vendor_identity_bootstrap_20260731/"
    "action_ball_profile_pins.v1.07e79f968a63.json"
)
PROFILE_SHA = (
    "07e79f968a6301f17a932775586868aa96be8c2df3bcf0358cab096280857f10"
)
SOLVER_SHA = (
    "146c4d6aa72cb06773a30f089e53acd5b4964c49ddfaf2d836675faa222c248a"
)
OLD_SOLVER_SHA = (
    "329ea0a33689303b08e84855ffcfd6fe541ef1c2537f9978be1f883dc202d80b"
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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_ascii_sha(value: object) -> str:
    return _sha(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_head_blob(relative: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relative.as_posix()}"],
        check=True,
        capture_output=True,
    ).stdout
    destination.write_bytes(raw)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        check=True,
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = tmp_path_factory.mktemp("a3-identity-repin-base")
    _copy(REPO_ROOT / SCRIPT_REL, repo / SCRIPT_REL)
    _copy(REPO_ROOT / REGISTRY_REL, repo / REGISTRY_REL)
    _copy_head_blob(PIN_PRODUCER_REL, repo / PIN_PRODUCER_REL)
    _copy_head_blob(
        MDP_REL / "action_ball_manifest.py",
        repo / MDP_REL / "action_ball_manifest.py",
    )
    for filename in SOLVER_SOURCES:
        _copy_head_blob(MDP_REL / filename, repo / MDP_REL / filename)
    _copy_head_blob(TABLE_GEOMETRY_REL, repo / TABLE_GEOMETRY_REL)
    _copy_head_blob(VENUE_REL, repo / VENUE_REL)
    _copy_head_blob(SOURCE_MANIFEST_REL, repo / SOURCE_MANIFEST_REL)
    _copy_head_blob(SOURCE_PROTOTYPE_REL, repo / SOURCE_PROTOTYPE_REL)
    _copy_head_blob(MOTION_REL, repo / MOTION_REL)
    _copy_head_blob(
        BLOCK_SOURCE_MANIFEST_REL,
        repo / BLOCK_SOURCE_MANIFEST_REL,
    )
    _copy_head_blob(
        BLOCK_SOURCE_PROTOTYPE_REL,
        repo / BLOCK_SOURCE_PROTOTYPE_REL,
    )
    _copy_head_blob(BLOCK_MOTION_REL, repo / BLOCK_MOTION_REL)

    profile_raw = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / PIN_PRODUCER_REL),
            "--repo-root",
            str(REPO_ROOT),
            "--source-rev",
            "HEAD",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert _sha(profile_raw) == PROFILE_SHA
    assert json.loads(profile_raw)["solver_profile_sha256"] == SOLVER_SHA
    (repo / PROFILE_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo / PROFILE_REL).write_bytes(profile_raw)

    _git(repo, "init", "-q")
    _commit_all(repo, "fixture")
    return repo


@pytest.fixture
def case_repo(base_repo: Path, tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "-q", str(base_repo), str(repo)], check=True
    )
    (repo / "configs/out").mkdir(parents=True)
    return repo


def _paths(repo: Path, action_id: str = "bh_loop_c") -> dict[str, Path]:
    return {
        "prototype": repo / f"configs/out/{action_id}.prototype.v2.json",
        "manifest": repo / f"configs/out/{action_id}.manifest.v3.json",
        "receipt": repo / f"configs/out/{action_id}.identity_repin.v1.json",
    }


def _command(
    repo: Path,
    *,
    commit: str | None = None,
    action_id: str | None = None,
    source_action_id: str | None = None,
) -> list[str]:
    selected_action = action_id or "bh_loop_c"
    source_action = source_action_id or selected_action
    source_manifest = ACTION_SOURCES[source_action]["manifest"]
    outputs = _paths(repo, selected_action)
    profile_sha = _sha((repo / PROFILE_REL).read_bytes())
    command = [
        sys.executable,
        str(repo / SCRIPT_REL),
    ]
    if action_id is not None:
        command.extend(("--action-id", action_id))
    command.extend([
        "--repo-root",
        str(repo),
        "--source-commit",
        commit or _git(repo, "rev-parse", "HEAD"),
        "--source-manifest",
        source_manifest.as_posix(),
        "--expected-source-manifest-sha256",
        _sha((repo / source_manifest).read_bytes()),
        "--profile-pins",
        PROFILE_REL.as_posix(),
        "--expected-profile-pins-sha256",
        profile_sha,
        "--prototype-output",
        outputs["prototype"].relative_to(repo).as_posix(),
        "--manifest-output",
        outputs["manifest"].relative_to(repo).as_posix(),
        "--receipt-output",
        outputs["receipt"].relative_to(repo).as_posix(),
    ])
    return command


def _run(
    repo: Path,
    *,
    commit: str | None = None,
    action_id: str | None = None,
    source_action_id: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(
            repo,
            commit=commit,
            action_id=action_id,
            source_action_id=source_action_id,
        ),
        capture_output=True,
        text=True,
    )


def test_materializes_cross_bound_three_output_identity_repin(case_repo: Path) -> None:
    result = _run(case_repo)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    outputs = _paths(case_repo)
    prototype = json.loads(outputs["prototype"].read_text())
    manifest = json.loads(outputs["manifest"].read_text())
    receipt = json.loads(outputs["receipt"].read_text())
    source_manifest = json.loads((case_repo / SOURCE_MANIFEST_REL).read_text())
    source_prototype = json.loads((case_repo / SOURCE_PROTOTYPE_REL).read_text())

    assert outputs["prototype"].read_bytes().endswith(b"\n")
    assert outputs["manifest"].read_bytes().endswith(b"\n")
    assert outputs["receipt"].read_bytes().endswith(b"\n")
    assert summary["action_id"] == "bh_loop_c"
    assert summary["action_registry_source_identity_sha256"] == receipt[
        "inputs"
    ]["action_registry"]["source_identity_sha256"]
    assert summary["solver_profile_sha256"] == SOLVER_SHA
    assert receipt["purpose"] == "identity_bootstrap_repin"
    assert receipt["authorization"] == {
        "contact_admission": False,
        "deployment": False,
        "dynamic_ready": False,
        "formal_bundle": False,
        "hardware": False,
        "identity_bootstrap_repin": True,
        "training": False,
    }
    assert all(receipt["invariants"].values())
    assert receipt["outputs"]["prototype"]["sha256"] == _sha(
        outputs["prototype"].read_bytes()
    )
    assert receipt["outputs"]["manifest"]["sha256"] == _sha(
        outputs["manifest"].read_bytes()
    )
    assert receipt["inputs"]["action_registry"] == {
        "path": REGISTRY_REL.as_posix(),
        "action_id": "bh_loop_c",
        "source_identity_sha256": summary[
            "action_registry_source_identity_sha256"
        ],
    }
    assert manifest["prototype"] == {
        "path": outputs["prototype"].relative_to(case_repo).as_posix(),
        "scope": "upper",
        "sha256": _sha(outputs["prototype"].read_bytes()),
    }
    assert prototype["scopes"] == source_prototype["scopes"]
    assert prototype["derived_sha256"] == source_prototype["derived_sha256"]
    assert manifest["actions"] == source_manifest["actions"]
    assert (
        manifest["counter_rally_objective"]
        == source_manifest["counter_rally_objective"]
    )


def test_materializes_bh_block_from_its_code_owned_stable_sources(
    case_repo: Path,
) -> None:
    assert _sha((case_repo / BLOCK_SOURCE_MANIFEST_REL).read_bytes()) == (
        BLOCK_SOURCE_MANIFEST_SHA
    )
    assert _sha((case_repo / BLOCK_SOURCE_PROTOTYPE_REL).read_bytes()) == (
        BLOCK_SOURCE_PROTOTYPE_SHA
    )
    result = _run(case_repo, action_id="bh_block")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    outputs = _paths(case_repo, "bh_block")
    manifest = json.loads(outputs["manifest"].read_text())
    prototype = json.loads(outputs["prototype"].read_text())
    receipt = json.loads(outputs["receipt"].read_text())
    source_manifest = json.loads(
        (case_repo / BLOCK_SOURCE_MANIFEST_REL).read_text()
    )
    source_prototype = json.loads(
        (case_repo / BLOCK_SOURCE_PROTOTYPE_REL).read_text()
    )

    assert summary["action_id"] == "bh_block"
    assert summary["action_registry_source_identity_sha256"] == receipt[
        "inputs"
    ]["action_registry"]["source_identity_sha256"]
    assert summary["solver_profile_sha256"] == SOLVER_SHA
    assert manifest["action_order"] == ["bh_block"]
    assert manifest["actions"] == source_manifest["actions"]
    assert manifest["prototype"] == {
        "path": outputs["prototype"].relative_to(case_repo).as_posix(),
        "scope": "upper",
        "sha256": _sha(outputs["prototype"].read_bytes()),
    }
    assert prototype["scopes"] == source_prototype["scopes"]
    assert prototype["derived_sha256"] == source_prototype["derived_sha256"]
    assert receipt["inputs"]["source_manifest"] == {
        "path": BLOCK_SOURCE_MANIFEST_REL.as_posix(),
        "sha256": BLOCK_SOURCE_MANIFEST_SHA,
    }
    assert receipt["inputs"]["source_prototype"] == {
        "path": BLOCK_SOURCE_PROTOTYPE_REL.as_posix(),
        "sha256": BLOCK_SOURCE_PROTOTYPE_SHA,
    }
    assert all(receipt["invariants"].values())


def test_loop_and_block_identity_outputs_can_coexist_with_current_producer(
    case_repo: Path,
) -> None:
    loop_result = _run(case_repo, action_id="bh_loop_c")
    assert loop_result.returncode == 0, loop_result.stderr
    loop_summary = json.loads(loop_result.stdout)
    _commit_all(case_repo, "materialize loop identity")

    block_result = _run(case_repo, action_id="bh_block")
    assert block_result.returncode == 0, block_result.stderr
    block_summary = json.loads(block_result.stdout)
    loop_receipt = json.loads(_paths(case_repo, "bh_loop_c")["receipt"].read_text())
    block_receipt = json.loads(_paths(case_repo, "bh_block")["receipt"].read_text())
    producer_sha = _sha((case_repo / SCRIPT_REL).read_bytes())

    assert loop_summary["producer"]["sha256"] == producer_sha
    assert block_summary["producer"]["sha256"] == producer_sha
    assert loop_receipt["inputs"]["producer"]["sha256"] == producer_sha
    assert block_receipt["inputs"]["producer"]["sha256"] == producer_sha
    assert loop_receipt["inputs"]["action_registry"]["action_id"] == "bh_loop_c"
    assert block_receipt["inputs"]["action_registry"]["action_id"] == "bh_block"
    assert all(path.exists() for path in _paths(case_repo, "bh_loop_c").values())
    assert all(path.exists() for path in _paths(case_repo, "bh_block").values())


def test_rejects_action_whose_planned_producer_path_is_cross_bound(
    case_repo: Path,
) -> None:
    registry_path = case_repo / REGISTRY_REL
    raw = registry_path.read_text()
    block_marker = '''identity_source_commit=None,
    identity_repin_producer=ArtifactPin(
        "hope_training/whole_body_tracking/scripts/"
        "materialize_a3_vendor_identity_manifest.py",
        None,
    ),'''
    replacement = '''identity_source_commit=None,
    identity_repin_producer=ArtifactPin(
        "hope_training/whole_body_tracking/scripts/other_identity_producer.py",
        None,
    ),'''
    assert block_marker in raw
    registry_path.write_text(raw.replace(block_marker, replacement, 1))
    _commit_all(case_repo, "cross-bind block identity producer")

    result = _run(case_repo, action_id="bh_block")
    assert result.returncode != 0
    assert "plans identity output from a different producer path" in result.stderr
    assert not any(path.exists() for path in _paths(case_repo, "bh_block").values())


def test_rejects_identity_producer_worktree_drift_before_publication(
    case_repo: Path,
) -> None:
    producer = case_repo / SCRIPT_REL
    producer.write_bytes(producer.read_bytes() + b"\n# producer drift\n")
    result = _run(case_repo, action_id="bh_block")
    assert result.returncode != 0
    assert "completely clean checkout" in result.stderr
    assert not any(path.exists() for path in _paths(case_repo, "bh_block").values())


@pytest.mark.parametrize(
    ("action_id", "source_action_id"),
    (("bh_block", "bh_loop_c"), ("bh_loop_c", "bh_block")),
)
def test_rejects_cross_action_stable_source_binding(
    case_repo: Path, action_id: str, source_action_id: str
) -> None:
    result = _run(
        case_repo,
        action_id=action_id,
        source_action_id=source_action_id,
    )
    assert result.returncode != 0
    assert "code-owned stable-v2 manifest blob" in result.stderr
    assert f"for action {action_id!r}" in result.stderr
    assert not any(
        path.exists() for path in _paths(case_repo, action_id).values()
    )


def test_rejects_cross_action_motion_even_when_manifest_is_resigned(
    case_repo: Path,
) -> None:
    path = case_repo / BLOCK_SOURCE_MANIFEST_REL
    manifest = json.loads(path.read_text())
    manifest["actions"][0]["motion_path"] = MOTION_REL.as_posix()
    manifest["actions"][0]["motion_sha256"] = _sha(
        (case_repo / MOTION_REL).read_bytes()
    )
    path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    _commit_all(case_repo, "cross-bind block to loop motion")
    result = _run(case_repo, action_id="bh_block")
    assert result.returncode != 0
    assert "code-owned stable-v2 manifest blob" in result.stderr
    assert "for action 'bh_block'" in result.stderr
    assert not any(
        output.exists() for output in _paths(case_repo, "bh_block").values()
    )


def test_rejects_unknown_action_before_publication(case_repo: Path) -> None:
    result = _run(
        case_repo,
        action_id="unknown_action",
        source_action_id="bh_loop_c",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not any(
        path.exists() for path in _paths(case_repo, "unknown_action").values()
    )


def test_output_prototype_cannot_retain_old_solver_provenance(case_repo: Path) -> None:
    result = _run(case_repo)
    assert result.returncode == 0, result.stderr
    raw = _paths(case_repo)["prototype"].read_text()
    prototype = json.loads(raw)
    assert OLD_SOLVER_SHA not in raw
    assert (
        prototype["provenance"]["profile_pins"]["solver_profile_sha256"]
        == SOLVER_SHA
    )
    assert prototype["provenance"]["profile_pins"]["sha256"] == PROFILE_SHA


def test_rejects_profile_payload_crossbinding_even_when_resigned(case_repo: Path) -> None:
    path = case_repo / PROFILE_REL
    profile = json.loads(path.read_text())
    profile["solver_payload"]["physics_profile_sha256"] = "0" * 64
    profile["solver_profile_sha256"] = _canonical_ascii_sha(
        profile["solver_payload"]
    )
    path.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n")
    _commit_all(case_repo, "tamper solver physics binding")
    result = _run(case_repo)
    assert result.returncode != 0
    assert "not the exact formal pinner output" in result.stderr
    assert not any(path.exists() for path in _paths(case_repo).values())


def test_rejects_fully_resigned_self_consistent_fake_physics_profile(
    case_repo: Path,
) -> None:
    path = case_repo / PROFILE_REL
    profile = json.loads(path.read_text())
    profile["physics_payload"]["virtual_ball_params"]["g"] = 123.0
    physics_sha = _canonical_ascii_sha(profile["physics_payload"])
    profile["physics_profile_sha256"] = physics_sha
    profile["solver_payload"]["physics_profile_sha256"] = physics_sha
    profile["solver_profile_sha256"] = _canonical_ascii_sha(
        profile["solver_payload"]
    )
    path.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n")
    _commit_all(case_repo, "fully resigned fake physics")
    result = _run(case_repo)
    assert result.returncode != 0
    assert "not the exact formal pinner output" in result.stderr
    assert not any(output.exists() for output in _paths(case_repo).values())


def test_rejects_nonformal_profile_authority(case_repo: Path) -> None:
    path = case_repo / PROFILE_REL
    profile = json.loads(path.read_text())
    profile["source_authority"]["authority"] = (
        "uncommitted_worktree_diagnostic_only_v1"
    )
    profile["source_authority"]["commit_binding"] = "none"
    path.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n")
    _commit_all(case_repo, "diagnostic authority")
    result = _run(case_repo)
    assert result.returncode != 0
    assert "not the exact formal pinner output" in result.stderr


def test_rejects_seven_source_map_drift(case_repo: Path) -> None:
    source = case_repo / MDP_REL / "continuous_questions.py"
    source.write_bytes(source.read_bytes() + b"\n# drift\n")
    _commit_all(case_repo, "solver source drift")
    result = _run(case_repo)
    assert result.returncode != 0
    assert "not the exact formal pinner output" in result.stderr


def test_rejects_action_registry_worktree_drift_before_publication(
    case_repo: Path,
) -> None:
    registry = case_repo / REGISTRY_REL
    registry.write_bytes(registry.read_bytes() + b"\n# uncommitted drift\n")
    result = _run(case_repo)
    assert result.returncode != 0
    assert REGISTRY_REL.as_posix() in result.stderr
    assert not any(output.exists() for output in _paths(case_repo).values())


def test_rejects_stable_prototype_numeric_drift_even_if_repinned(
    case_repo: Path,
) -> None:
    prototype_path = case_repo / SOURCE_PROTOTYPE_REL
    prototype = json.loads(prototype_path.read_text())
    prototype["scopes"]["upper"][0]["slack_b_xy_m"] += 0.001
    prototype["derived_sha256"] = _canonical_ascii_sha(prototype["scopes"])
    prototype_path.write_text(
        json.dumps(prototype, indent=2, sort_keys=True) + "\n"
    )
    manifest_path = case_repo / SOURCE_MANIFEST_REL
    manifest = json.loads(manifest_path.read_text())
    manifest["prototype"]["sha256"] = _sha(prototype_path.read_bytes())
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    _commit_all(case_repo, "numeric prototype drift")
    result = _run(case_repo)
    assert result.returncode != 0
    assert "code-owned stable-v2 manifest blob" in result.stderr


def test_any_spent_target_prevents_all_new_output_bytes(case_repo: Path) -> None:
    spent = _paths(case_repo)["receipt"]
    spent.write_bytes(b"spent\n")
    _commit_all(case_repo, "spent receipt target")
    result = _run(case_repo)
    assert result.returncode != 0
    assert spent.read_bytes() == b"spent\n"
    assert not _paths(case_repo)["prototype"].exists()
    assert not _paths(case_repo)["manifest"].exists()


def test_rejects_head_source_commit_mismatch_without_outputs(case_repo: Path) -> None:
    old_commit = _git(case_repo, "rev-parse", "HEAD")
    marker = case_repo / "tracked-marker.txt"
    marker.write_text("next\n")
    _commit_all(case_repo, "next")
    result = _run(case_repo, commit=old_commit)
    assert result.returncode != 0
    assert "producer requires HEAD=" in result.stderr
    assert not any(path.exists() for path in _paths(case_repo).values())


def test_rejects_dirty_checkout_before_publication(case_repo: Path) -> None:
    (case_repo / "untracked.txt").write_text("dirty\n")
    result = _run(case_repo)
    assert result.returncode != 0
    assert "completely clean checkout" in result.stderr
    assert not any(path.exists() for path in _paths(case_repo).values())
