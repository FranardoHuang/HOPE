from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_lean_training_queue.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lean_asset_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_module()


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _init_checkout(path: Path) -> str:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    assets = path / "assets"
    assets.mkdir()
    (assets / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    (path / "tracked.txt").write_text("exact\n", encoding="utf-8")
    _git(path, "add", "assets/.gitignore", "tracked.txt")
    _git(path, "commit", "-qm", "fixture")
    return _git(path, "rev-parse", "HEAD")


def _make_asset(root: Path) -> None:
    (root / "urdf").mkdir(parents=True)
    (root / "meshes").mkdir()
    (root / "config").mkdir()
    robot = ET.Element("robot", name="a3")
    for index in range(43):
        name = f"mesh{index:02d}.STL"
        (root / "meshes" / name).write_bytes(f"mesh-{index}\n".encode())
        for _ in range(2):
            visual = ET.SubElement(robot, "visual")
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "mesh", filename=f"../meshes/{name}")
    (root / "meshes" / "unused.STL").write_bytes(b"accepted-extra\n")
    (root / "config" / "joint.yaml").write_text("joints: 31\n", encoding="utf-8")
    ET.ElementTree(robot).write(
        root / "urdf" / "model.urdf", encoding="utf-8", xml_declaration=True
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> dict:
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    return {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "tree_content_sha256": Q._canonical_sha256({"files": rows}),
    }


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    donor = tmp_path / "donor"
    source_commit = _init_checkout(source)
    donor_commit = _init_checkout(donor)
    donor_asset = donor / "assets" / "agibot_a3"
    _make_asset(donor_asset)
    target = source / "assets" / "agibot_a3"
    shutil.copytree(donor_asset, target)
    expected = _inventory(donor_asset)
    assert expected["file_count"] == 46
    contract = {
        "target_relative_path": "assets/agibot_a3",
        "donor": {
            "checkout": str(donor),
            "commit": donor_commit,
            "relative_path": "assets/agibot_a3",
        },
        **expected,
        "symlinks_forbidden": True,
        "target_must_be_gitignored": True,
    }
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    spec = {
        "mode": "prepare",
        "pod": "pod2",
        "source": {"checkout": str(source), "commit": source_commit},
        "contract": contract,
        "receipt_path": str(tmp_path / "receipts" / "receipt.json"),
        "staging_path": str(tmp_path / "staging" / "asset.stage"),
        "lock_path": str(tmp_path / "locks" / "source.lock"),
        "entrypoint_relative_path": "hope_training/whole_body_tracking/scripts/train.py",
        "proc_root": str(proc_root),
        "urdf_relative_path": "urdf/model.urdf",
        "expected_unique_mesh_references": 43,
    }
    return source, donor_asset, target, spec


def _run(spec: dict) -> subprocess.CompletedProcess[str]:
    encoded = base64.b64encode(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    return subprocess.run(
        [sys.executable, "-c", Q.SOURCE_ASSET_PROGRAM, encoded],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def test_prepare_existing_exact_is_idempotent_and_doctor_consumes_receipt(tmp_path):
    _source, _donor, target, spec = _fixture(tmp_path)
    doctor = dict(spec, mode="doctor")
    missing_receipt = _run(doctor)
    assert missing_receipt.returncode == 2
    assert "receipt is missing" in missing_receipt.stderr

    inode = (target / "urdf" / "model.urdf").stat().st_ino
    first = _run(spec)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["receipt_state"] == "created"
    second = _run(spec)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["receipt_state"] == "existing_exact"
    assert (target / "urdf" / "model.urdf").stat().st_ino == inode
    verified = _run(doctor)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["materialized"] is False


def test_target_or_donor_drift_extra_and_symlink_fail_closed(tmp_path):
    _source, donor, target, spec = _fixture(tmp_path)
    assert _run(spec).returncode == 0

    (target / "extra.bin").write_bytes(b"extra")
    drift = _run(dict(spec, mode="doctor"))
    assert drift.returncode == 2 and "inventory drift" in drift.stderr
    (target / "extra.bin").unlink()

    mesh = donor / "meshes" / "mesh00.STL"
    mesh.write_bytes(b"changed")
    donor_drift = _run(spec)
    assert donor_drift.returncode == 2 and "inventory drift" in donor_drift.stderr
    mesh.unlink()
    mesh.symlink_to("mesh01.STL")
    donor_symlink = _run(spec)
    assert donor_symlink.returncode == 2 and "symlink" in donor_symlink.stderr
    mesh.unlink()
    os.mkfifo(mesh)
    donor_special = _run(spec)
    assert donor_special.returncode == 2 and "special file" in donor_special.stderr


def test_partial_staging_or_existing_wrong_target_is_never_overwritten(tmp_path):
    _source, _donor, target, spec = _fixture(tmp_path)
    shutil.rmtree(target)
    missing = _run(dict(spec, mode="doctor"))
    assert missing.returncode == 2 and "target runtime asset is missing" in missing.stderr
    staging = Path(spec["staging_path"])
    staging.mkdir(parents=True)
    marker = staging / "partial"
    marker.write_text("preserve", encoding="utf-8")
    partial = _run(spec)
    assert partial.returncode == 2 and "staging path already exists" in partial.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not target.exists()

    shutil.rmtree(staging)
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("do-not-overwrite", encoding="utf-8")
    wrong_target = _run(spec)
    assert wrong_target.returncode == 2 and "inventory drift" in wrong_target.stderr
    assert sentinel.read_text(encoding="utf-8") == "do-not-overwrite"


def test_wrong_donor_commit_and_target_symlink_fail_closed(tmp_path):
    _source, donor, target, spec = _fixture(tmp_path)
    wrong = json.loads(json.dumps(spec))
    wrong["contract"]["donor"]["commit"] = "0" * 40
    wrong_commit = _run(wrong)
    assert wrong_commit.returncode == 2 and "wrong commit" in wrong_commit.stderr

    shutil.rmtree(target)
    target.symlink_to(donor, target_is_directory=True)
    linked = _run(spec)
    assert linked.returncode == 2 and "symlink component" in linked.stderr
