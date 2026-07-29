from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "action_ball_runtime_inventory.py"
)
SPEC = importlib.util.spec_from_file_location("action_ball_runtime_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runtime_inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_inventory)


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _wheel_record_hash(raw: bytes) -> str:
    digest = hashlib.sha256(raw).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode(
        "ascii"
    )


def _write_dist_info(
    site_packages: Path,
    distribution_name: str,
    version: str,
    top_level_name: str,
    owned_files: list,
    requires_dist: list = None,
) -> Path:
    stem = distribution_name.replace("-", "_")
    metadata_path = site_packages / ("%s-%s.dist-info" % (stem, version))
    metadata_path.mkdir(exist_ok=True)
    metadata = metadata_path / "METADATA"
    metadata_lines = [
        "Metadata-Version: 2.1",
        "Name: %s" % distribution_name,
        "Version: %s" % version,
    ]
    for requirement in requires_dist or []:
        metadata_lines.append("Requires-Dist: %s" % requirement)
    metadata.write_bytes(("\n".join(metadata_lines) + "\n").encode("utf-8"))
    wheel = metadata_path / "WHEEL"
    wheel.write_bytes(
        b"Wheel-Version: 1.0\nGenerator: runtime-test\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    top_level = metadata_path / "top_level.txt"
    top_level.write_bytes((top_level_name + "\n").encode("utf-8"))
    record = metadata_path / "RECORD"
    all_files = list(owned_files) + [metadata, wheel, top_level]
    rows = []
    for path in sorted(all_files, key=lambda item: str(item)):
        raw = path.read_bytes()
        relative = os.path.relpath(str(path), str(site_packages)).replace(
            os.sep, "/"
        )
        rows.append(
            "%s,%s,%d" % (relative, _wheel_record_hash(raw), len(raw))
        )
    record_relative = os.path.relpath(
        str(record), str(site_packages)
    ).replace(os.sep, "/")
    rows.append("%s,," % record_relative)
    record.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return metadata_path


def _make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    venv = tmp_path / "venv"
    binary_dir = venv / "bin"
    binary_dir.mkdir(parents=True)
    final_python = tmp_path / "runtime" / "python3.10"
    final_python.parent.mkdir()
    final_python.write_bytes(b"#!/bin/sh\nexit 99\n")
    final_python.chmod(0o755)
    middle = binary_dir / "python3"
    middle.symlink_to(final_python)
    requested = binary_dir / "python"
    requested.symlink_to("python3")
    (venv / "pyvenv.cfg").write_bytes(
        b"home = /opt/python\ninclude-system-site-packages = false\n"
    )
    stdlib_root = tmp_path / "runtime" / "lib" / "python3.10"
    stdlib_root.mkdir(parents=True)

    site_packages = venv / "lib" / "python3.10" / "site-packages"
    site_packages.mkdir(parents=True)

    checkout = tmp_path / "IsaacLab"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "runtime@example.invalid")
    _git(checkout, "config", "user.name", "Runtime Test")
    (checkout / "README").write_text("fake IsaacLab\n", encoding="utf-8")
    editable_project = checkout / "source" / "isaaclab"
    editable_origin = editable_project / "isaaclab" / "__init__.py"
    editable_origin.parent.mkdir(parents=True)
    editable_origin.write_bytes(b"VERSION = '1.0-isaaclab'\n")
    _git(checkout, "add", "README", "source")
    _git(checkout, "commit", "-q", "-m", "fixture")
    _git(
        checkout,
        "remote",
        "add",
        "origin",
        "https://example.invalid/IsaacLab.git",
    )

    origins = {}
    descriptors = {}
    for name in runtime_inventory.MODULE_NAMES:
        version = "1.0-" + name
        if name == "isaaclab":
            origin = editable_origin
            pth = site_packages / "isaaclab-editable.pth"
            pth.write_bytes((str(editable_project) + "\n").encode("utf-8"))
            direct_url = site_packages / "isaaclab-1.0-isaaclab.dist-info"
            direct_url_path = direct_url / "direct_url.json"
            # _write_dist_info creates the parent; write direct_url immediately
            # afterwards and regenerate RECORD with the extra install artifacts.
            metadata_path = _write_dist_info(
                site_packages,
                "isaaclab",
                version,
                "isaaclab",
                [pth],
            )
            direct_url_path = metadata_path / "direct_url.json"
            direct_url_path.write_bytes(
                json.dumps(
                    {
                        "dir_info": {"editable": True},
                        "url": editable_project.as_uri(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            metadata_path = _write_dist_info(
                site_packages,
                "isaaclab",
                version,
                "isaaclab",
                [pth, direct_url_path],
            )
            descriptors[name] = {
                "name": name,
                "version": version,
                "metadata_path": str(metadata_path),
                "editable": True,
                "project_path": str(editable_project),
                "direct_url_path": str(direct_url_path),
                "top_level_names": [name],
            }
        else:
            origin = site_packages / name / "__init__.py"
            origin.parent.mkdir()
            origin.write_bytes(
                ("VERSION = %r\n" % version).encode("utf-8")
            )
            owned_files = [origin]
            if name == "isaacsim":
                carb = origin.parent / "carb_core.so"
                carb.write_bytes(b"fixture carb runtime\n")
                omni = origin.parent / "omni" / "runtime.py"
                omni.parent.mkdir()
                omni.write_bytes(b"OMNI_RUNTIME = True\n")
                owned_files.extend([carb, omni])
            metadata_path = _write_dist_info(
                site_packages,
                name,
                version,
                name,
                owned_files,
                ["transitive-runtime>=1.0"] if name == "torch" else None,
            )
            descriptors[name] = {
                "name": name,
                "version": version,
                "metadata_path": str(metadata_path),
                "editable": False,
                "project_path": None,
                "direct_url_path": None,
                "top_level_names": [name],
            }
        origins[name] = origin

    dependency_name = "transitive-runtime"
    dependency_version = "1.2"
    dependency_top_level = "transitive_runtime"
    dependency_origin = (
        site_packages / dependency_top_level / "__init__.py"
    )
    dependency_origin.parent.mkdir()
    dependency_origin.write_bytes(b"VERSION = '1.2'\n")
    dependency_metadata = _write_dist_info(
        site_packages,
        dependency_name,
        dependency_version,
        dependency_top_level,
        [dependency_origin],
    )
    descriptors[dependency_name] = {
        "name": dependency_name,
        "version": dependency_version,
        "metadata_path": str(dependency_metadata),
        "editable": False,
        "project_path": None,
        "direct_url_path": None,
        "top_level_names": [dependency_top_level],
    }
    origins[dependency_name] = dependency_origin

    def fake_probe(
        path: Path,
        import_roots,
        *,
        bootstrap_path=runtime_inventory.NOSITE_BOOTSTRAP,
        entrypoint_path=runtime_inventory.INVENTORY_ENTRYPOINT,
    ):
        assert path == requested
        nosite = runtime_inventory._load_nosite_bootstrap()
        bootstrap_binding = nosite.bind_regular_file(bootstrap_path)
        entrypoint_binding = nosite.bind_regular_file(entrypoint_path)
        command = nosite.build_exact_nosite_argv(
            python=path,
            bootstrap=bootstrap_path,
            bootstrap_sha256=bootstrap_binding["sha256"],
            entrypoint=entrypoint_path,
            entrypoint_sha256=entrypoint_binding["sha256"],
            import_roots=import_roots,
            entrypoint_argv=["_probe"],
        )
        root_paths = [row["path"] for row in import_roots]
        before_sys_path = [str(stdlib_root)]
        after_sys_path = before_sys_path + root_paths
        exact_flags = {
            "isolated": True,
            "no_site": True,
            "no_user_site": True,
            "ignore_environment": True,
            "dont_write_bytecode": True,
            "optimize": 0,
        }
        modules = []
        for name in runtime_inventory.MODULE_NAMES:
            modules.append(
                {
                    "name": name,
                    "version": "1.0-" + name,
                    "version_source": "distribution:" + name,
                    "distributions": [descriptors[name]],
                    "origin_path": str(origins[name]),
                }
            )
        resolved = sorted(
            descriptors.values(),
            key=lambda row: (row["metadata_path"], row["name"]),
        )
        torch_metadata = descriptors["torch"]["metadata_path"]
        dependency_metadata_path = descriptors[dependency_name][
            "metadata_path"
        ]
        return {
            "implementation": "cpython",
            "version": "3.10.14",
            "cache_tag": "cpython-310",
            "executable": str(requested),
            "prefix": str(venv),
            "base_prefix": str(tmp_path / "runtime"),
            "sys_path": after_sys_path,
            "site_package_paths": sorted(root_paths),
            "modules": modules,
            "marker_environment": {
                "platform_machine": "fixture",
                "python_full_version": "3.10.14",
                "python_version": "3.10",
                "sys_platform": "fixture",
            },
            "resolved_distributions": resolved,
            "dependency_edges": [
                {
                    "from_metadata_path": torch_metadata,
                    "requirement": "transitive-runtime>=1.0",
                    "to_metadata_path": dependency_metadata_path,
                    "to_name": dependency_name,
                    "to_version": dependency_version,
                }
            ],
            "optional_distributions": [
                {
                    "name": "tensordict",
                    "present": False,
                    "version": None,
                    "metadata_path": None,
                }
            ],
            "no_site_execution": {
                "outer": {
                    "schema_version": 1,
                    "kind": "action_ball_python_nosite_execution_v1",
                    "argv_contract_sha256": command.contract_sha256,
                    "bootstrap": bootstrap_binding,
                    "entrypoint": entrypoint_binding,
                    "import_roots": list(import_roots),
                    "flags": exact_flags,
                    "site_module_loaded_before_entrypoint": False,
                    "pth_files_executed": False,
                    "sys_path_before_import_roots": before_sys_path,
                    "sys_path_after_import_roots": after_sys_path,
                },
                "inner": {
                    "flags": exact_flags,
                    "site_module_loaded": False,
                    "pth_files_executed": False,
                    "sys_path": after_sys_path,
                },
            },
        }

    monkeypatch.setattr(runtime_inventory, "_run_probe", fake_probe)
    return {
        "requested": requested,
        "middle": middle,
        "final_python": final_python,
        "venv": venv,
        "site_packages": site_packages,
        "origins": origins,
        "descriptors": descriptors,
        "editable_project": editable_project,
        "checkout": checkout,
        # The editable IsaacLab source is explicit.  Its .pth remains present
        # and inventoried, but no-site execution never interprets it.
        "import_roots": [site_packages, editable_project],
    }


def _external_output(tmp_path: Path) -> Path:
    parent = tmp_path / "external"
    parent.mkdir()
    return parent / "receipt.json"


def _make_real_probe_roots(tmp_path: Path):
    site_packages = tmp_path / "probe-site"
    editable_source = tmp_path / "probe-isaaclab-source"
    site_packages.mkdir()
    editable_source.mkdir()
    marker = tmp_path / "executable-pth-ran"
    (site_packages / "malicious.pth").write_text(
        "import pathlib; pathlib.Path(%r).write_text('executed')\n"
        % str(marker),
        encoding="utf-8",
    )

    for module_name in runtime_inventory.MODULE_NAMES:
        if module_name == "isaaclab":
            origin = editable_source / "isaaclab" / "__init__.py"
            distribution_name = "isaaclab"
        else:
            origin = site_packages / module_name / "__init__.py"
            distribution_name = module_name
        origin.parent.mkdir(parents=True)
        origin.write_text(
            "VERSION = %r\n" % ("1.0-" + module_name),
            encoding="utf-8",
        )
        owned = [origin] if site_packages in origin.parents else []
        if module_name == "packaging":
            markers = origin.parent / "markers.py"
            markers.write_text(
                "\n".join(
                    [
                        "def default_environment():",
                        "    return {",
                        "      'python_version': '3.8',",
                        "      'python_full_version': '3.8.20',",
                        "      'sys_platform': 'fixture',",
                        "      'platform_machine': 'fixture',",
                        "    }",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            requirements = origin.parent / "requirements.py"
            requirements.write_text(
                "\n".join(
                    [
                        "class InvalidRequirement(Exception):",
                        "    pass",
                        "class Requirement:",
                        "    def __init__(self, value):",
                        "        raise InvalidRequirement(value)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            owned.extend([markers, requirements])
        _write_dist_info(
            site_packages,
            distribution_name,
            "1.0",
            module_name,
            owned,
        )
    return site_packages, editable_source, marker


def test_real_probe_requires_explicit_editable_root_and_never_executes_pth(
    tmp_path: Path,
):
    site_packages, editable_source, marker = _make_real_probe_roots(
        tmp_path
    )
    nosite = runtime_inventory._load_nosite_bootstrap()
    site_only = nosite.bind_import_roots([site_packages])
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="isolated Python runtime probe exited",
    ):
        runtime_inventory._run_probe(Path(sys.executable), site_only)
    assert marker.exists() is False

    explicit = nosite.bind_import_roots(
        [site_packages, editable_source]
    )
    payload = runtime_inventory._run_probe(
        Path(sys.executable), explicit
    )
    runtime_inventory._validate_probe_payload(
        payload, Path(sys.executable)
    )
    assert marker.exists() is False
    assert payload["no_site_execution"]["outer"]["import_roots"] == explicit
    assert payload["no_site_execution"]["outer"][
        "pth_files_executed"
    ] is False
    assert payload["no_site_execution"]["inner"][
        "site_module_loaded"
    ] is False


def test_mint_and_verify_exact_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = tmp_path / "external" / "runtime.json"
    output.parent.mkdir()

    receipt = runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )

    assert receipt["schema_version"] == 2
    assert receipt["kind"] == "action_ball_runtime_inventory_v2"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    python = receipt["content"]["python"]
    assert [row["kind"] for row in python["symlink_chain"]] == [
        "symlink",
        "symlink",
        "regular",
    ]
    assert python["symlink_chain"][0]["link_text"] == "python3"
    assert python["symlink_chain"][1]["link_text"] == str(
        fixture["final_python"]
    )
    assert [row["name"] for row in python["probe"]["modules"]] == list(
        runtime_inventory.MODULE_NAMES
    )
    site_row = next(
        row
        for row in python["site_packages"]
        if row["path"] == str(fixture["site_packages"])
    )
    assert [row["path"] for row in site_row["pth_files"]] == [
        str(fixture["site_packages"] / "isaaclab-editable.pth")
    ]
    distributions = python["distributions"]
    assert len(distributions) == len(runtime_inventory.MODULE_NAMES) + 1
    assert any(
        row["name"] == "transitive-runtime" for row in distributions
    )
    assert all(
        row["metadata_file"]["path"].endswith(".dist-info/METADATA")
        and row["wheel_file"]["path"].endswith(".dist-info/WHEEL")
        and row["record"]["path"].endswith(".dist-info/RECORD")
        for row in distributions
    )
    editable = next(row for row in distributions if row["editable"])
    assert editable["git_checkout"]["remote_v"]["byte_count"] > 0
    assert editable["git_checkout"]["status_porcelain_v2"]["byte_count"] == 0
    assert [row["witness"] for row in python["critical_record_witnesses"]] == [
        "carb",
        "omni",
    ]
    assert python["probe"]["optional_distributions"] == [
        {
            "name": "tensordict",
            "present": False,
            "version": None,
            "metadata_path": None,
        }
    ]
    no_site = python["probe"]["no_site_execution"]
    assert no_site["outer"]["flags"]["no_site"] is True
    assert no_site["inner"]["flags"]["no_site"] is True
    assert no_site["outer"]["pth_files_executed"] is False
    assert no_site["inner"]["pth_files_executed"] is False
    assert no_site["inner"]["site_module_loaded"] is False
    assert [row["path"] for row in no_site["outer"]["import_roots"]] == [
        str(fixture["site_packages"]),
        str(fixture["editable_project"]),
    ]
    assert runtime_inventory.verify_receipt(output) == receipt


def test_publish_is_no_clobber_and_receipt_is_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    receipt = runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    assert output.read_bytes() == runtime_inventory._canonical_json_bytes(receipt) + b"\n"
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="O_EXCL"):
        runtime_inventory.mint_receipt(
            fixture["requested"],
            fixture["checkout"],
            output,
            fixture["import_roots"],
        )


def test_verify_rejects_pth_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    (fixture["site_packages"] / "isaaclab-editable.pth").write_bytes(
        b"/different\n"
    )
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="differs"):
        runtime_inventory.verify_receipt(output)


def test_verify_rejects_python_link_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    replacement = fixture["final_python"].with_name("python-replacement")
    replacement.write_bytes(b"#!/bin/sh\nexit 98\n")
    replacement.chmod(0o755)
    fixture["middle"].unlink()
    fixture["middle"].symlink_to(replacement)
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="differs"):
        runtime_inventory.verify_receipt(output)


def test_verify_rejects_dirty_isaaclab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    (fixture["checkout"] / "untracked").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError, match="not exactly clean"
    ):
        runtime_inventory.verify_receipt(output)


def test_strict_schema_rejects_extra_and_duplicate_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    receipt = runtime_inventory.build_receipt(
        fixture["requested"],
        fixture["checkout"],
        fixture["import_roots"],
    )
    extra = dict(receipt)
    extra["invented"] = True
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="keys differ"):
        runtime_inventory.validate_receipt_document(extra)
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="duplicate"):
        runtime_inventory._strict_json_loads(
            b'{"schema_version":1,"schema_version":1}', "duplicate"
        )
    inner_flag_drift = json.loads(json.dumps(receipt))
    inner_flag_drift["content"]["python"]["probe"][
        "no_site_execution"
    ]["inner"]["flags"]["no_site"] = False
    inner_flag_drift["content_sha256"] = hashlib.sha256(
        runtime_inventory._canonical_json_bytes(
            inner_flag_drift["content"]
        )
    ).hexdigest()
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="inner interpreter flags",
    ):
        runtime_inventory.validate_receipt_document(inner_flag_drift)


def test_receipt_symlink_and_mode_change_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    output.chmod(0o644)
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="0600"):
        runtime_inventory.verify_receipt(output)
    output.chmod(0o600)
    link = tmp_path / "receipt-link.json"
    link.symlink_to(output)
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="without following"):
        runtime_inventory.verify_receipt(link)


def test_python_must_be_requested_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(runtime_inventory.RuntimeInventoryError, match="must name a symlink"):
        runtime_inventory.build_receipt(
            fixture["final_python"],
            fixture["checkout"],
            fixture["import_roots"],
        )


def test_cli_verify_emits_machine_readable_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = tmp_path / "external" / "receipt.json"
    output.parent.mkdir()
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    assert runtime_inventory.main(["verify", "--receipt", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["kind"] == runtime_inventory.RECEIPT_KIND
    assert result["receipt_path"] == str(output)


def test_record_closure_rejects_installed_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    fixture["origins"]["torch"].write_bytes(b"VERSION = 'tampered'\n")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="sha256 differs from RECORD",
    ):
        runtime_inventory.verify_receipt(output)


def test_transitive_dependency_and_critical_witness_drift_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    dependency_output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        dependency_output,
        fixture["import_roots"],
    )
    fixture["origins"]["transitive-runtime"].write_bytes(
        b"VERSION = 'tampered dependency'\n"
    )
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="sha256 differs from RECORD",
    ):
        runtime_inventory.verify_receipt(dependency_output)

    second = tmp_path / "second"
    second.mkdir()
    second_fixture = _make_fixture(second, monkeypatch)
    witness_output = _external_output(second)
    runtime_inventory.mint_receipt(
        second_fixture["requested"],
        second_fixture["checkout"],
        witness_output,
        second_fixture["import_roots"],
    )
    carb = second_fixture["origins"]["isaacsim"].parent / "carb_core.so"
    carb.write_bytes(b"tampered carb runtime\n")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="sha256 differs from RECORD",
    ):
        runtime_inventory.verify_receipt(witness_output)


def test_distribution_requires_metadata_wheel_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    metadata_path = Path(
        fixture["descriptors"]["torch"]["metadata_path"]
    )
    (metadata_path / "WHEEL").unlink()
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="WHEEL.*not an openable regular file",
    ):
        runtime_inventory.build_receipt(
            fixture["requested"],
            fixture["checkout"],
            fixture["import_roots"],
        )


def test_editable_direct_url_must_match_project_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    direct_url = Path(
        fixture["descriptors"]["isaaclab"]["direct_url_path"]
    )
    direct_url.write_bytes(
        json.dumps(
            {
                "dir_info": {"editable": True},
                "url": (tmp_path / "other-project").as_uri(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="differs from project_path",
    ):
        runtime_inventory.build_receipt(
            fixture["requested"],
            fixture["checkout"],
            fixture["import_roots"],
        )


def test_git_control_environment_cannot_redirect_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "invented.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "invented-worktree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "invented-index"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "invented-config"))
    receipt = runtime_inventory.build_receipt(
        fixture["requested"],
        fixture["checkout"],
        fixture["import_roots"],
    )
    assert (
        receipt["content"]["isaaclab_checkout"]["path"]
        == str(fixture["checkout"])
    )


def test_git_index_hiding_flags_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    _git(fixture["checkout"], "update-index", "--assume-unchanged", "README")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="assume-unchanged",
    ):
        runtime_inventory.build_receipt(
            fixture["requested"],
            fixture["checkout"],
            fixture["import_roots"],
        )


def test_verify_rejects_git_remote_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    _git(
        fixture["checkout"],
        "remote",
        "set-url",
        "origin",
        "https://example.invalid/retargeted.git",
    )
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError, match="differs"
    ):
        runtime_inventory.verify_receipt(output)


def test_verify_rejects_pyvenv_cfg_raw_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    output = _external_output(tmp_path)
    runtime_inventory.mint_receipt(
        fixture["requested"],
        fixture["checkout"],
        output,
        fixture["import_roots"],
    )
    (fixture["venv"] / "pyvenv.cfg").write_bytes(b"home = /different\n")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError, match="differs"
    ):
        runtime_inventory.verify_receipt(output)


def test_unlisted_distribution_file_and_metadata_symlink_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _make_fixture(tmp_path, monkeypatch)
    unlisted = fixture["origins"]["torch"].parent / "injected.py"
    unlisted.write_bytes(b"INJECTED = True\n")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="absent from RECORD",
    ):
        runtime_inventory.build_receipt(
            fixture["requested"],
            fixture["checkout"],
            fixture["import_roots"],
        )
    unlisted.unlink()
    metadata_path = Path(
        fixture["descriptors"]["torch"]["metadata_path"]
    )
    metadata = metadata_path / "METADATA"
    metadata.unlink()
    metadata.symlink_to(metadata_path / "WHEEL")
    with pytest.raises(
        runtime_inventory.RuntimeInventoryError,
        match="cannot bind the explicit no-site import roots",
    ):
        runtime_inventory.build_receipt(
            fixture["requested"],
            fixture["checkout"],
            fixture["import_roots"],
        )
