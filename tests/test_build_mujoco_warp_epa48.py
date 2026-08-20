"""Host tests for the EPA48 supply chain, not the missing physics fixture."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_mujoco_warp_epa48.py"
PROVENANCE = ROOT / "configs/mujoco_warp_epa48_20260821/PROVENANCE.json"
HOST_RECEIPT = ROOT / "configs/mujoco_warp_epa48_20260821/HOST_BUILD_RECEIPT_SUMMARY.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("epa48_build_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_module()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_tracked_contract_is_exactly_two_source_changes() -> None:
    value = json.loads(PROVENANCE.read_text())
    upstream, fork = value["upstream"], value["fork"]
    assert upstream["release_tag"] == "v3.10.0.3"
    assert upstream["release_commit"] == "710c34ca96745a44bfb701cdbda89e1434845728"
    assert upstream["sdist"]["sha256"] == (
        "f22196465cb1350677f66d8b65aa23bf37d95e150ce3ba3c68ea934ba35e3070"
    )
    assert fork["version"] == "3.10.0.3+hope.epa48.1"
    assert fork["epa_horizon"] == 48
    assert set(fork["allowed_source_changes"]) == {
        "pyproject.toml",
        "mujoco_warp/_src/types.py",
    }
    patch = PROVENANCE.parent / fork["patch"]["path"]
    text = patch.read_text()
    assert M._sha256_file(patch) == fork["patch"]["sha256"]
    assert text.count("diff --git ") == 2
    assert '+version = "3.10.0.3+hope.epa48.1"' in text
    assert "+MJ_MAX_EPAHORIZON = 48" in text
    assert "scientific_status" not in value
    assert "reported telemetry" in value["build"]["policy"]


def test_tracked_host_build_receipt_is_supply_chain_only() -> None:
    value = json.loads(HOST_RECEIPT.read_text())
    assert value["schema_version"] == 1
    assert value["verdict"] == "PASS_BUILD_CHAIN_ONLY"
    assert value["full_build_receipt"]["schema_version"] == M.RECEIPT_SCHEMA_VERSION
    assert value["source"]["package_before_file_count"] == 281
    assert value["source"]["package_after_file_count"] == 281
    assert value["source"]["package_before_manifest_sha256"] != value["source"]["package_after_manifest_sha256"]
    assert value["builder"]["authority"] == "reported_telemetry_only"
    assert value["scientific_status"]["deterministic_epa_fixture"] == "BLOCKED_NOT_YET_CAPTURED"
    assert value["scientific_status"]["training_authorized"] is False


def test_wrong_sdist_sha_fails_before_extract(tmp_path: Path) -> None:
    bad = tmp_path / "mujoco_warp-3.10.0.3.tar.gz"
    bad.write_bytes(b"wrong")
    with pytest.raises(M.ForkError, match="sdist SHA mismatch"):
        M._prepare_source(bad, tmp_path / "out", M._load_provenance())
    assert not (tmp_path / "out").exists()


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("mujoco_warp-3.10.0.3/../../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(M.ForkError, match="unsafe sdist member"):
        M._safe_extract(archive_path, tmp_path / "out", "mujoco_warp-3.10.0.3")
    assert not (tmp_path / "escape").exists()


def _synthetic_patch_case(tmp_path: Path):
    top = "mujoco_warp-3.10.0.3"
    before = {
        "pyproject.toml": b'[project]\nversion = "3.10.0.3"\n',
        "mujoco_warp/__init__.py": b"# package\n",
        "mujoco_warp/_src/types.py": b"MJ_MAX_EPAHORIZON = 24\n",
    }
    after = dict(before)
    after["pyproject.toml"] = b'[project]\nversion = "3.10.0.3+hope.epa48.1"\n'
    after["mujoco_warp/_src/types.py"] = b"MJ_MAX_EPAHORIZON = 48\n"
    sdist = tmp_path / "mujoco_warp-3.10.0.3.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, data in before.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    patch = tmp_path / "fork.patch"
    patch.write_text(
        """--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,2 +1,2 @@
 [project]
-version = "3.10.0.3"
+version = "3.10.0.3+hope.epa48.1"
--- a/mujoco_warp/_src/types.py
+++ b/mujoco_warp/_src/types.py
@@ -1 +1 @@
-MJ_MAX_EPAHORIZON = 24
+MJ_MAX_EPAHORIZON = 48
"""
    )
    allowed = {
        name: {"before_sha256": _sha(before[name]), "after_sha256": _sha(after[name])}
        for name in ("pyproject.toml", "mujoco_warp/_src/types.py")
    }
    provenance = {
        "upstream": {
            "sdist": {
                "filename": sdist.name,
                "sha256": M._sha256_file(sdist),
                "top_level_directory": top,
            }
        },
        "fork": {
            "patch": {"path": patch.name, "sha256": M._sha256_file(patch)},
            "allowed_source_changes": allowed,
        },
    }
    return sdist, patch, provenance, after


def test_patch_changes_only_the_two_allowed_files(tmp_path: Path, monkeypatch) -> None:
    sdist, patch, provenance, expected = _synthetic_patch_case(tmp_path)
    monkeypatch.setattr(M, "PROVENANCE_PATH", patch.parent / "PROVENANCE.json")
    source, evidence = M._prepare_source(sdist, tmp_path / "out", provenance)
    assert evidence["changed_files"] == ["mujoco_warp/_src/types.py", "pyproject.toml"]
    assert evidence["sdist"]["sha256"] == provenance["upstream"]["sdist"]["sha256"]
    assert evidence["patch"]["sha256"] == provenance["fork"]["patch"]["sha256"]
    before = {
        "mujoco_warp/__init__.py": _sha(b"# package\n"),
        "mujoco_warp/_src/types.py": _sha(b"MJ_MAX_EPAHORIZON = 24\n")
    }
    after = {
        "mujoco_warp/__init__.py": _sha(b"# package\n"),
        "mujoco_warp/_src/types.py": _sha(b"MJ_MAX_EPAHORIZON = 48\n"),
    }
    assert evidence["package_before"] == {
        "file_count": len(before), "manifest_sha256": M._manifest_sha256(before)
    }
    assert evidence["package_after"] == {
        "file_count": len(after), "manifest_sha256": M._manifest_sha256(after)
    }
    for name, data in expected.items():
        assert (source / name).read_bytes() == data


def _wheel_source(root: Path, types_data: bytes) -> Path:
    source = root / "source"
    files = {
        "AUTHORS": b"authors\n",
        "LICENSE": b"license\n",
        "PKG-INFO": b"Metadata-Version: 2.4\nName: mujoco-warp\nVersion: 3.10.0.3\nRequires-Dist: absl-py\n\n",
        "mujoco_warp/__init__.py": b"# package\n",
        "mujoco_warp/_src/types.py": types_data,
        "mujoco_warp.egg-info/entry_points.txt": b"[console_scripts]\nmjwarp-testspeed = mujoco_warp.testspeed:main\n",
        "mujoco_warp.egg-info/top_level.txt": b"mujoco_warp\n",
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return source


def _write_wheel(
    path: Path,
    source: Path,
    *,
    version: str = "3.10.0.3+hope.epa48.1",
    extra=None,
    drop=None,
    duplicate=None,
) -> None:
    dist = "mujoco_warp-3.10.0.3+hope.epa48.1.dist-info"
    entries = {}
    for file in (source / "mujoco_warp").rglob("*"):
        if file.is_file():
            entries[file.relative_to(source).as_posix()] = file.read_bytes()
    entries.update(
        {
            f"{dist}/licenses/AUTHORS": (source / "AUTHORS").read_bytes(),
            f"{dist}/licenses/LICENSE": (source / "LICENSE").read_bytes(),
            f"{dist}/entry_points.txt": (source / "mujoco_warp.egg-info/entry_points.txt").read_bytes(),
            f"{dist}/top_level.txt": (source / "mujoco_warp.egg-info/top_level.txt").read_bytes(),
            f"{dist}/METADATA": f"Metadata-Version: 2.4\nName: mujoco-warp\nVersion: {version}\nRequires-Dist: absl-py\n\n".encode(),
            f"{dist}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        }
    )
    entries.update(extra or {})
    if drop:
        entries.pop(drop)
    record_name = f"{dist}/RECORD"
    rows = []
    for name, data in entries.items():
        digest = M.base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{name},sha256={digest},{len(data)}")
    entries[record_name] = ("\n".join(rows + [f"{record_name},,"]) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
        if duplicate:
            archive.writestr(duplicate, entries[duplicate])


def test_wheel_verifier_binds_version_types_bytes_and_wheel_sha(tmp_path: Path) -> None:
    provenance = M._load_provenance()
    types_data = b"MJ_MAX_EPAHORIZON = 48\n"
    source = _wheel_source(tmp_path, types_data)
    provenance["fork"]["allowed_source_changes"]["mujoco_warp/_src/types.py"][
        "after_sha256"
    ] = _sha(types_data)
    wheel = tmp_path / provenance["fork"]["expected_wheel_filename"]
    _write_wheel(wheel, source)
    evidence = M._verify_wheel(wheel, provenance, source)
    assert evidence["sha256"] == M._sha256_file(wheel)
    assert evidence["version"] == "3.10.0.3+hope.epa48.1"
    _write_wheel(wheel, source, extra={"mujoco_warp/_src/injected.py": b"PAYLOAD = True\n"})
    with pytest.raises(M.ForkError, match="payload set mismatch"):
        M._verify_wheel(wheel, provenance, source)


def test_wheel_rejects_missing_unsafe_and_duplicate_entries(tmp_path: Path) -> None:
    provenance = M._load_provenance()
    types_data = b"MJ_MAX_EPAHORIZON = 48\n"
    source = _wheel_source(tmp_path, types_data)
    provenance["fork"]["allowed_source_changes"]["mujoco_warp/_src/types.py"][
        "after_sha256"
    ] = _sha(types_data)
    wheel = tmp_path / provenance["fork"]["expected_wheel_filename"]
    _write_wheel(wheel, source, drop="mujoco_warp/__init__.py")
    with pytest.raises(M.ForkError, match="payload set mismatch"):
        M._verify_wheel(wheel, provenance, source)
    _write_wheel(wheel, source, extra={"../escape.py": b"escape\n"})
    with pytest.raises(M.ForkError, match="unsafe wheel member"):
        M._verify_wheel(wheel, provenance, source)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_wheel(wheel, source, duplicate="mujoco_warp/__init__.py")
    with pytest.raises(M.ForkError, match="duplicate ZIP entries"):
        M._verify_wheel(wheel, provenance, source)


def test_receipt_rejects_schema_payload_mismatch_but_treats_build_environment_as_reported() -> None:
    provenance = M._load_provenance()
    source = {"fresh_source_manifest": "1" * 64}
    wheel = {"sha256": "2" * 64}
    caller = {
        "reported_executable": "/builder/python",
        "resolved_executable": "/builder/python",
        "python_version": "3.12",
        "pip": {"version": "1", "root": "/builder"},
        "setuptools": {"version": "1", "root": "/builder"},
        "wheel": {"version": "1", "root": "/builder"},
        "executable_sha256": "3" * 64,
    }
    receipt = {
        "schema_version": M.RECEIPT_SCHEMA_VERSION,
        "verdict": "PASS_BUILD_CHAIN_ONLY",
        "fork_id": provenance["fork_id"],
        "source": source,
        "build": {
            "caller": caller,
            "pip_flags": ["--no-index", "--no-deps", "--no-build-isolation", "--no-cache-dir"],
        },
        "wheel": wheel,
    }
    M._validate_receipt(receipt, provenance, source, wheel)
    reported = json.loads(json.dumps(receipt))
    reported["build"]["caller"]["python_version"] = "reported environment may differ"
    reported["build"]["caller"]["executable_sha256"] = "4" * 64
    M._validate_receipt(reported, provenance, source, wheel)
    for field, forged in (
        ("schema_version", 999),
        ("schema_version", float(M.RECEIPT_SCHEMA_VERSION)),
        ("fork_id", "forged"),
        ("build", {"forged": True}),
        ("wheel", {"sha256": "4" * 64}),
    ):
        candidate = dict(receipt)
        candidate[field] = forged
        with pytest.raises(M.ForkError):
            M._validate_receipt(candidate, provenance, source, wheel)

def test_pip_build_is_explicitly_offline_no_deps_and_keeps_venv_entry(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    wheelhouse = tmp_path / "wheelhouse"
    captured = {}
    venv_entry = tmp_path / "venv" / "bin" / "python"
    venv_entry.parent.mkdir(parents=True)
    venv_entry.symlink_to(Path(sys.executable))

    def fake_identity(python):
        assert python == venv_entry
        return {"python": "reported-only"}

    monkeypatch.setattr(M, "_builder_identity", fake_identity)

    def fake_run(command, **kwargs):
        captured.update(command=command, environment=kwargs["env"])
        (wheelhouse / "only.whl").write_bytes(b"wheel")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    _, evidence = M._build_wheel(source, wheelhouse, venv_entry)
    assert captured["command"][0] == str(venv_entry)
    assert {"--no-index", "--no-deps", "--no-build-isolation"} <= set(captured["command"])
    assert captured["environment"]["PIP_NO_INDEX"] == "1"
    assert evidence["pip_flags"] == [
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
    ]


def test_existing_exact_artifact_root_is_not_modified(tmp_path: Path, monkeypatch) -> None:
    vendor = tmp_path / "vendor_assets"
    artifact = vendor / "mujoco_warp_epa48_1"
    artifact.mkdir(parents=True)
    sentinel = artifact / "preserve.bin"
    sentinel.write_bytes(b"do-not-clobber")
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        M.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )
    provenance = {"build": {"wheel_output_root": "vendor_assets/mujoco_warp_epa48_1"}}
    with pytest.raises(M.ForkError, match="refusing to overwrite"):
        M._validate_target(artifact, provenance)
    assert sentinel.read_bytes() == b"do-not-clobber"
    assert list(artifact.iterdir()) == [sentinel]

    stage = tmp_path / "publish-stage"
    raced_target = tmp_path / "raced-target"
    stage.mkdir()
    (stage / "payload").write_text("new")
    raced_target.mkdir()
    with pytest.raises(M.ForkError, match="refusing to replace"):
        M._atomic_rename_noreplace(stage, raced_target)
    assert (stage / "payload").read_text() == "new"
    assert raced_target.is_dir() and not list(raced_target.iterdir())


def test_artifact_verifier_rejects_extra_entries_and_symlink_wheel(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    wheelhouse = artifact / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    (artifact / M.RECEIPT_NAME).write_text("{}")
    wheel = wheelhouse / "fork.whl"
    wheel.write_bytes(b"wheel")
    (artifact / "extra").write_text("unexpected")
    monkeypatch.setattr(M, "_load_provenance", lambda: {})
    with pytest.raises(M.ForkError, match="exactly receipt and wheelhouse"):
        M.verify_artifact(tmp_path / "sdist", artifact)

    (artifact / "extra").unlink()
    wheel.unlink()
    target = tmp_path / "outside.whl"
    target.write_bytes(b"wheel")
    wheel.symlink_to(target)
    with pytest.raises(M.ForkError, match="exactly one regular wheel"):
        M.verify_artifact(tmp_path / "sdist", artifact)


def test_build_verifies_wheel_against_fresh_reconstructed_source(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    calls = []
    evidence = {"source": "pinned"}
    provenance = {"fork_id": "fork"}

    def fake_prepare(_sdist, destination, _provenance):
        destination.mkdir(parents=True)
        marker = destination / "marker"
        marker.write_text("pristine")
        calls.append(destination)
        return destination, evidence

    def fake_build(source, wheelhouse, _python):
        (source / "marker").write_text("mutated-by-build-backend")
        wheelhouse.mkdir()
        wheel = wheelhouse / "fork.whl"
        wheel.write_bytes(b"wheel")
        return wheel, {"builder": "reported"}

    def fake_verify(_wheel, _provenance, source):
        assert source == calls[1]
        assert (source / "marker").read_text() == "pristine"
        return {"sha256": "1" * 64}

    monkeypatch.setattr(M, "_load_provenance", lambda: provenance)
    monkeypatch.setattr(M, "_validate_target", lambda *_args: None)
    monkeypatch.setattr(M, "_prepare_source", fake_prepare)
    monkeypatch.setattr(M, "_build_wheel", fake_build)
    monkeypatch.setattr(M, "_verify_wheel", fake_verify)
    monkeypatch.setattr(M, "_validate_receipt", lambda *_args: None)
    monkeypatch.setattr(M, "_atomic_rename_noreplace", lambda source, target: source.rename(target))

    assert M.build_artifact(tmp_path / "source.tar.gz", artifact, Path(sys.executable)) == artifact
    assert len(calls) == 2
