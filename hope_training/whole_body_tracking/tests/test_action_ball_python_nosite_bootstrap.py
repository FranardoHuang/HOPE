from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


WBT = Path(__file__).resolve().parents[1]
SCRIPT = WBT / "scripts" / "action_ball_python_nosite_bootstrap.py"
SPEC = importlib.util.spec_from_file_location(
    "action_ball_python_nosite_bootstrap_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def _fixture(tmp_path: Path):
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir(parents=True)
    root_b.mkdir()
    (root_a / "fixture_module.py").write_text(
        "VALUE = 'root-a-loaded'\n", encoding="utf-8"
    )
    marker = tmp_path / "pth-executed"
    (root_a / "executable.pth").write_text(
        "import pathlib; pathlib.Path(%r).write_text('bad')\n" % str(marker),
        encoding="utf-8",
    )
    (root_b / "data.txt").write_text("second-root\n", encoding="utf-8")
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "import fixture_module",
                "print(json.dumps({",
                "  'attestation': ACTION_BALL_NOSITE_ATTESTATION,",
                "  'argv': sys.argv,",
                "  'module_value': fixture_module.VALUE,",
                "  'site_loaded': 'site' in sys.modules,",
                "}, sort_keys=True, separators=(',', ':')))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    bootstrap = B.bind_regular_file(SCRIPT, label="bootstrap")
    entrypoint_binding = B.bind_regular_file(entrypoint, label="entrypoint")
    roots = B.bind_import_roots([root_a, root_b])
    command = B.build_exact_nosite_argv(
        python=Path(sys.executable),
        bootstrap=SCRIPT,
        bootstrap_sha256=bootstrap["sha256"],
        entrypoint=entrypoint,
        entrypoint_sha256=entrypoint_binding["sha256"],
        import_roots=roots,
        entrypoint_argv=["--fixture", "value"],
    )
    return {
        "root_a": root_a,
        "root_b": root_b,
        "marker": marker,
        "entrypoint": entrypoint,
        "bootstrap": bootstrap,
        "entrypoint_binding": entrypoint_binding,
        "roots": roots,
        "command": command,
    }


def test_builder_validator_and_execution_are_exact_and_no_site(tmp_path: Path):
    fixture = _fixture(tmp_path)
    command = fixture["command"]
    validated = B.validate_exact_nosite_argv(
        command.argv,
        expected_python=Path(sys.executable),
        expected_bootstrap=fixture["bootstrap"],
        expected_entrypoint=fixture["entrypoint_binding"],
        expected_import_roots=fixture["roots"],
        expected_entrypoint_argv=["--fixture", "value"],
        expected_contract_sha256=command.contract_sha256,
    )
    assert validated == command
    assert list(command.argv[1:5]) == ["-I", "-B", "-S", "-c"]

    completed = subprocess.run(
        list(command.argv),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # The bootstrap emits no success output; this one line belongs entirely to
    # the entrypoint.
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    result = json.loads(completed.stdout)
    assert result["argv"] == [
        str(fixture["entrypoint"]),
        "--fixture",
        "value",
    ]
    assert result["module_value"] == "root-a-loaded"
    assert result["site_loaded"] is False
    assert fixture["marker"].exists() is False
    attestation = result["attestation"]
    assert attestation["argv_contract_sha256"] == command.contract_sha256
    assert attestation["flags"] == {
        "dont_write_bytecode": True,
        "ignore_environment": True,
        "isolated": True,
        "no_site": True,
        "no_user_site": True,
        "optimize": 0,
    }
    assert attestation["pth_files_executed"] is False
    assert attestation["site_module_loaded_before_entrypoint"] is False
    assert attestation["import_roots"] == fixture["roots"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda argv: argv + ["extra"], "exactly 10"),
        (
            lambda argv: argv[:1] + ["-B", "-I"] + argv[3:],
            "flags/trampoline",
        ),
        (
            lambda argv: argv[:6] + argv[6:8] + ["0" * 64] + argv[9:],
            "contract SHA",
        ),
    ],
)
def test_validator_rejects_extra_reordered_or_drifted_tokens(
    tmp_path: Path, mutation, match: str
):
    fixture = _fixture(tmp_path)
    with pytest.raises(B.NoSiteBootstrapError, match=match):
        B.validate_exact_nosite_argv(mutation(list(fixture["command"].argv)))


def test_validator_rejects_source_entrypoint_and_root_drift(tmp_path: Path):
    fixture = _fixture(tmp_path)
    fixture["entrypoint"].write_text("print('drift')\n", encoding="utf-8")
    with pytest.raises(B.NoSiteBootstrapError, match="entrypoint source differs"):
        B.validate_exact_nosite_argv(fixture["command"].argv)

    second = _fixture(tmp_path / "second")
    (second["root_b"] / "data.txt").write_text("root drift\n", encoding="utf-8")
    with pytest.raises(B.NoSiteBootstrapError, match="import root 1 differs"):
        B.validate_exact_nosite_argv(second["command"].argv)


def test_validator_rejects_root_reorder_and_duplicate(tmp_path: Path):
    fixture = _fixture(tmp_path)
    with pytest.raises(B.NoSiteBootstrapError, match="order/bindings"):
        B.validate_exact_nosite_argv(
            fixture["command"].argv,
            expected_import_roots=list(reversed(fixture["roots"])),
        )

    contract = dict(fixture["command"].contract)
    contract["import_roots"] = [
        fixture["roots"][0],
        fixture["roots"][0],
    ]
    raw = B.canonical_json_bytes(contract)
    encoded = __import__("base64").b64encode(raw).decode("ascii")
    argv = list(fixture["command"].argv)
    argv[-2] = __import__("hashlib").sha256(raw).hexdigest()
    argv[-1] = encoded
    with pytest.raises(B.NoSiteBootstrapError, match="unique"):
        B.validate_exact_nosite_argv(argv)

    nested = fixture["root_a"] / "nested"
    nested.mkdir()
    (nested / "member.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(B.NoSiteBootstrapError, match="overlap or nest"):
        B.bind_import_roots([fixture["root_a"], nested])


def test_trampoline_rejects_bootstrap_byte_drift_before_exec(tmp_path: Path):
    fixture = _fixture(tmp_path)
    copied = tmp_path / "bootstrap-copy.py"
    copied.write_bytes(SCRIPT.read_bytes())
    binding = B.bind_regular_file(copied)
    command = B.build_exact_nosite_argv(
        python=Path(sys.executable),
        bootstrap=copied,
        bootstrap_sha256=binding["sha256"],
        entrypoint=fixture["entrypoint"],
        entrypoint_sha256=fixture["entrypoint_binding"]["sha256"],
        import_roots=fixture["roots"],
        entrypoint_argv=[],
    )
    copied.write_text("raise SystemExit('wrong bootstrap')\n", encoding="utf-8")
    completed = subprocess.run(
        list(command.argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode != 0
    assert "bootstrap bytes or identity drifted" in completed.stderr
    assert completed.stdout == ""
