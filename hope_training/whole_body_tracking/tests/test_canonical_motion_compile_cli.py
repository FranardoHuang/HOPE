"""Tests for the strict candidate-only canonical compiler CLI."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_motion_compile_cli as cli  # noqa: E402
from mujoco_motion_player import RUNTIME_JOINT_NAMES  # noqa: E402


_MOTION_IDS = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)
_SCOPES = ("upper", "full")
_POST_BUILD_GATES = (
    "strict_schema2_and_shared_ready_digest",
    "exact_vendor_mujoco_fk_playback",
    "joint_position_velocity_and_plant_specific_torque_screen",
    "self_collision_body_racket_ground_table_net_scan",
    "contact_opportunity_rescan_per_scope",
    "stationary_behavior_and_recovery_exam_per_motion",
    "registry_consumer_export_deploy_contract",
)
_CATCHABLE_TEST_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
    *((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _acceleration_receipt() -> dict:
    return {
        "acceleration_rad_s2": [float(index + 1) for index in range(31)],
        "all_positive": True,
        "joint_names": list(RUNTIME_JOINT_NAMES),
        "limiting_source_frame": [
            f"source.npz:f{index}" for index in range(31)
        ],
        "method": "test diagonal acceleration envelope",
        "minimum_effort_margin_nm": [
            float(index + 10) for index in range(31)
        ],
    }


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema2_receipts(
    motion_id: str,
    scope: str,
    output_sha256: str,
) -> tuple[dict, dict]:
    tool_path = Path(
        cli.cmc.build_schema2_candidate.__code__.co_filename
    ).resolve()
    hashes = {
        "input_sha256": _digest_text(f"input:{motion_id}:{scope}"),
        "ready_sha256": _digest_text("canonical-ready"),
        "mjcf_sha256": _digest_text("mjcf"),
        "body_order_sha256": _digest_text("body-order"),
        "tool_sha256": _sha256(tool_path),
        "output_npz_sha256": output_sha256,
    }
    shared = {
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "tool_id": "canonical_schema2_builder_v1",
        "hashes": hashes,
    }
    manifest = {
        **shared,
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "files": {
            "mjcf_path": "/test/mjcf.xml",
            "body_order_path": "/test/body-order.json",
            "tool_path": str(tool_path),
        },
    }
    report = {
        **shared,
        "hashes": dict(hashes),
        "status": "PASS",
        "frames": 2,
        "fps": 50.0,
    }
    return manifest, report


def _candidate_manifest(recipe_sha256: str, compiler_sha256: str) -> dict:
    outputs = []
    for motion_id in _MOTION_IDS:
        for scope in _SCOPES:
            filename = f"{motion_id}_{scope}_canonical_v2.npz"
            output_sha = hashlib.sha256(
                _candidate_bytes(motion_id, scope)
            ).hexdigest()
            schema2_manifest, schema2_report = _schema2_receipts(
                motion_id, scope, output_sha
            )
            outputs.append(
                {
                    "motion_id": motion_id,
                    "scope": scope,
                    "filename": filename,
                    "output_npz_sha256": output_sha,
                    "schema2_manifest": schema2_manifest,
                    "schema2_report": schema2_report,
                }
            )
    return {
        "schema_version": 1,
        "library_id": "test-canonical-library",
        "publication_class": "compiler_candidate",
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "training_authorized": False,
        "hardware_authorized": False,
        "recipe": {
            "path": "recipe.json",
            "sha256": recipe_sha256,
        },
        "compiler": {
            "path": str(Path(cli.cmc.__file__).resolve()),
            "sha256": compiler_sha256,
        },
        "output_matrix": {
            "motion_ids": list(_MOTION_IDS),
            "scopes": list(_SCOPES),
            "candidate_count": 10,
        },
        "outputs": outputs,
        "post_build_gates": [
            {"name": name, "status": "pending"}
            for name in _POST_BUILD_GATES
        ],
    }


def _candidate_bytes(motion_id: str, scope: str) -> bytes:
    return f"candidate:{motion_id}:{scope}".encode("ascii")


@pytest.fixture
def job_environment(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    recipe_path = repo / "recipe.json"
    recipe_path.write_text('{"recipe": "strict-test"}\n', encoding="utf-8")
    acceleration_path = tmp_path / "acceleration.json"
    acceleration_sha = _write_json(
        acceleration_path, _acceleration_receipt()
    )
    model_paths = {}
    model_hashes = {}
    for name in ("mjcf", "urdf", "body_order"):
        path = repo / f"{name}.asset"
        path.write_bytes(f"{name}-bytes".encode("ascii"))
        model_paths[name] = path
        model_hashes[name] = _sha256(path)
    fake_recipe = SimpleNamespace(
        path=recipe_path,
        model_paths=model_paths,
        model_hashes=model_hashes,
        sources=[
            SimpleNamespace(motion_id=motion_id)
            for motion_id in _MOTION_IDS
        ],
        raw={"post_build_gates": list(_POST_BUILD_GATES)},
    )
    monkeypatch.setattr(
        cli, "load_canonical_motion_recipe", lambda *args, **kwargs: fake_recipe
    )
    output = tmp_path / "candidate-output"
    args = [
        "--recipe",
        str(recipe_path),
        "--repo-root",
        str(repo),
        "--output",
        str(output),
        "--joint-acceleration-receipt",
        str(acceleration_path),
        "--joint-acceleration-receipt-sha256",
        acceleration_sha,
        "--full-root-position-lower",
        "0.0",
        "-0.4",
        "0.85",
        "-1.0",
        "-1.0",
        "-1.0",
        "--full-root-position-upper",
        "0.4",
        "0.1",
        "1.05",
        "1.0",
        "1.0",
        "1.0",
        "--full-root-velocity",
        "1.0",
        "1.0",
        "0.5",
        "2.0",
        "2.0",
        "2.0",
        "--full-root-acceleration",
        "10.0",
        "10.0",
        "5.0",
        "20.0",
        "20.0",
        "20.0",
        "--s0-full-grounding-offset-m",
        "0.075",
        "--samples-per-scaled-unit",
        "6.0",
        "--min-connector-intervals",
        "5",
        "--min-core-intervals",
        "5",
        "--grid-subdivisions",
        "4",
        "--search-workers",
        "8",
        "--search-parallel-backend",
        "process",
    ]
    return SimpleNamespace(
        repo=repo,
        recipe_path=recipe_path,
        acceleration_path=acceleration_path,
        acceleration_sha=acceleration_sha,
        fake_recipe=fake_recipe,
        output=output,
        args=args,
    )


def _install_fake_compiler(monkeypatch, environment):
    calls = []
    compiler_sha = _sha256(Path(cli.cmc.__file__).resolve())
    manifest = _candidate_manifest(
        _sha256(environment.recipe_path), compiler_sha
    )
    library = SimpleNamespace(manifest=manifest)

    def compile_library(recipe, *, options):
        calls.append(
            {
                "recipe": recipe,
                "options": options,
            }
        )
        return library

    def write_library(actual_library, output_directory):
        assert actual_library is library
        output = Path(output_directory)
        assert not output.exists()
        output.mkdir()
        (output / cli.cmc.BUILD_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for row in manifest["outputs"]:
            npz_path = output / row["filename"]
            npz_path.write_bytes(
                _candidate_bytes(row["motion_id"], row["scope"])
            )
            _write_json(
                npz_path.with_suffix(
                    npz_path.suffix + ".manifest.json"
                ),
                row["schema2_manifest"],
            )
            _write_json(
                npz_path.with_suffix(npz_path.suffix + ".report.json"),
                row["schema2_report"],
            )
        assert not (output / cli.RUN_RECEIPT_NAME).exists()
        return output

    monkeypatch.setattr(
        cli.cmc, "compile_loaded_canonical_motion_library", compile_library
    )
    monkeypatch.setattr(
        cli.cmc, "write_compiled_canonical_motion_library", write_library
    )
    return calls


def _real_writer_library(environment):
    compiler_sha = _sha256(Path(cli.cmc.__file__).resolve())
    manifest = _candidate_manifest(
        _sha256(environment.recipe_path), compiler_sha
    )
    motions = []
    for row in manifest["outputs"]:
        candidate = cli.cmc.Schema2Candidate(
            arrays={},
            npz_bytes=_candidate_bytes(
                row["motion_id"], row["scope"]
            ),
            manifest=row["schema2_manifest"],
            report=row["schema2_report"],
        )
        motions.append(
            SimpleNamespace(
                filename=row["filename"],
                schema2=candidate,
            )
        )
    return cli.cmc.CompiledLibrary(
        recipe=environment.fake_recipe,
        motions=tuple(motions),
        manifest=manifest,
    )


def test_normal_run_maps_every_option_and_writes_bound_receipt(
    job_environment, monkeypatch
):
    calls = _install_fake_compiler(monkeypatch, job_environment)

    result = cli.run(job_environment.args)

    assert len(calls) == 1
    call = calls[0]
    assert call["recipe"] is job_environment.fake_recipe
    options = call["options"]
    np.testing.assert_array_equal(
        options.joint_acceleration_limits_rad_s2,
        np.arange(1.0, 32.0),
    )
    np.testing.assert_array_equal(
        options.full_root_limits.position_lower,
        [0.0, -0.4, 0.85, -1.0, -1.0, -1.0],
    )
    np.testing.assert_array_equal(
        options.full_root_limits.position_upper,
        [0.4, 0.1, 1.05, 1.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        options.full_root_limits.velocity,
        [1.0, 1.0, 0.5, 2.0, 2.0, 2.0],
    )
    np.testing.assert_array_equal(
        options.full_root_limits.acceleration,
        [10.0, 10.0, 5.0, 20.0, 20.0, 20.0],
    )
    assert options.s0_full_grounding_offset_m == pytest.approx(0.075)
    assert options.samples_per_scaled_unit == pytest.approx(6.0)
    assert options.min_connector_intervals == 5
    assert options.min_core_intervals == 5
    assert options.grid_subdivisions == 4
    assert options.search_workers == 8
    assert options.search_parallel_backend == "process"
    assert options.face_config is None
    assert options.face_active_candidate_seeds == ()

    receipt_path = job_environment.output / cli.RUN_RECEIPT_NAME
    assert receipt_path.is_file()
    on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert on_disk == result
    assert result["publication_class"] == "compiler_candidate"
    assert result["training_authorized"] is False
    assert result["hardware_authorized"] is False
    assert (
        result["joint_acceleration_receipt"]["sha256"]
        == job_environment.acceleration_sha
    )
    manifest_path = (
        job_environment.output / cli.cmc.BUILD_MANIFEST_NAME
    )
    assert result["build_manifest"]["sha256"] == _sha256(manifest_path)
    assert result["recipe"]["sha256"] == _sha256(
        job_environment.recipe_path
    )
    assert result["job_args"]["search_parallel_backend"] == "process"
    assert result["job_args"]["face_config"] is None
    assert result["job_args"]["face_active_candidate_seeds"] == []
    assert len(result["published_outputs"]) == 10
    for published in result["published_outputs"]:
        for sidecar_key in ("manifest_sidecar", "report_sidecar"):
            sidecar = published[sidecar_key]
            sidecar_path = Path(sidecar["path"])
            assert sidecar_path.is_file()
            assert sidecar["sha256"] == _sha256(sidecar_path)
    assert result["termination_contract"]["parent_signals_handled"][:2] == [
        "SIGINT",
        "SIGTERM",
    ]
    assert (
        result["termination_contract"][
            "publication_on_listed_catchable_termination"
        ]
        .startswith("remove the requested output path")
    )
    assert "retained for manual audit" in result["termination_contract"][
        "termination_quarantine_retention"
    ]
    for name, binding in result["models"].items():
        assert binding["sha256"] == _sha256(
            job_environment.fake_recipe.model_paths[name]
        )
    receipt_payload = dict(result)
    expected_payload_sha = receipt_payload.pop(
        "run_receipt_payload_sha256"
    )
    canonical = json.dumps(
        receipt_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == expected_payload_sha


def test_real_writer_publishes_strict_ten_three_file_bundles(
    job_environment, monkeypatch
):
    library = _real_writer_library(job_environment)
    compile_calls = []

    def compile_library(recipe, *, options):
        compile_calls.append((recipe, options))
        return library

    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        compile_library,
    )

    result = cli.run(job_environment.args)

    assert len(compile_calls) == 1
    children = {
        path.name for path in job_environment.output.iterdir()
    }
    assert len(children) == 32
    assert cli.cmc.BUILD_MANIFEST_NAME in children
    assert cli.RUN_RECEIPT_NAME in children
    assert len(list(job_environment.output.glob("*.npz"))) == 10
    assert len(
        list(job_environment.output.glob("*.npz.manifest.json"))
    ) == 10
    assert len(
        list(job_environment.output.glob("*.npz.report.json"))
    ) == 10
    assert len(result["published_outputs"]) == 10
    for row, published in zip(
        library.manifest["outputs"],
        result["published_outputs"],
    ):
        npz_path = job_environment.output / row["filename"]
        manifest_sidecar = npz_path.with_suffix(
            npz_path.suffix + ".manifest.json"
        )
        report_sidecar = npz_path.with_suffix(
            npz_path.suffix + ".report.json"
        )
        assert json.loads(
            manifest_sidecar.read_text(encoding="utf-8")
        ) == row["schema2_manifest"]
        assert json.loads(
            report_sidecar.read_text(encoding="utf-8")
        ) == row["schema2_report"]
        assert published["manifest_sidecar"]["sha256"] == _sha256(
            manifest_sidecar
        )
        assert published["report_sidecar"]["sha256"] == _sha256(
            report_sidecar
        )


def test_dry_run_validates_but_does_not_compile_or_create_output(
    job_environment, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not compile or write")

    monkeypatch.setattr(
        cli.cmc, "compile_loaded_canonical_motion_library", forbidden
    )
    monkeypatch.setattr(
        cli.cmc, "write_compiled_canonical_motion_library", forbidden
    )

    result = cli.run([*job_environment.args, "--dry-run"])

    assert result["status"] == "PASS_INPUT_VALIDATION_ONLY"
    assert result["output_created"] is False
    assert result["training_authorized"] is False
    assert result["hardware_authorized"] is False
    assert not job_environment.output.exists()
    assert "portable_boundary" in result["termination_contract"]


def test_missing_acceleration_receipt_field_fails_closed(
    job_environment, monkeypatch
):
    receipt = _acceleration_receipt()
    del receipt["method"]
    receipt_sha = _write_json(job_environment.acceleration_path, receipt)
    args = list(job_environment.args)
    sha_index = args.index("--joint-acceleration-receipt-sha256") + 1
    args[sha_index] = receipt_sha
    compile_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *a, **k: compile_calls.append((a, k)),
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="receipt keys changed",
    ):
        cli.run(args)

    assert not compile_calls
    assert not job_environment.output.exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda receipt: receipt["acceleration_rad_s2"].__setitem__(
                0, float("nan")
            ),
            "non-finite",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "acceleration_rad_s2",
                receipt["acceleration_rad_s2"][:-1],
            ),
            r"shape \(31,\)",
        ),
    ],
)
def test_nonfinite_or_wrong_shape_acceleration_receipt_is_rejected(
    job_environment, mutate, match
):
    receipt = _acceleration_receipt()
    mutate(receipt)
    receipt_sha = _write_json(job_environment.acceleration_path, receipt)
    args = list(job_environment.args)
    args[args.index("--joint-acceleration-receipt-sha256") + 1] = (
        receipt_sha
    )

    with pytest.raises(cli.CanonicalMotionCompileCliError, match=match):
        cli.run(args)

    assert not job_environment.output.exists()


def test_nonfinite_cli_numeric_argument_is_rejected(job_environment):
    args = list(job_environment.args)
    args[args.index("--samples-per-scaled-unit") + 1] = "nan"

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="samples per scaled unit must be finite",
    ):
        cli.run(args)

    assert not job_environment.output.exists()


def test_wrong_root_vector_shape_is_rejected_by_parser(job_environment):
    args = list(job_environment.args)
    start = args.index("--full-root-velocity") + 1
    del args[start + 5]

    with pytest.raises(
        cli.CanonicalMotionCompileCliError, match="expected 6 arguments"
    ):
        cli.run(args)

    assert not job_environment.output.exists()


def test_receipt_sha_mismatch_fails_before_compile(
    job_environment, monkeypatch
):
    compile_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *a, **k: compile_calls.append((a, k)),
    )
    args = list(job_environment.args)
    args[args.index("--joint-acceleration-receipt-sha256") + 1] = "0" * 64

    with pytest.raises(
        cli.CanonicalMotionCompileCliError, match="SHA-256 mismatch"
    ):
        cli.run(args)

    assert not compile_calls
    assert not job_environment.output.exists()


def test_existing_output_is_never_overwritten(
    job_environment, monkeypatch
):
    job_environment.output.mkdir()
    sentinel = job_environment.output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    compile_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *a, **k: compile_calls.append((a, k)),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.run(job_environment.args)

    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert not compile_calls


def test_authorized_library_is_rejected_before_publication(
    job_environment, monkeypatch
):
    compiler_sha = _sha256(Path(cli.cmc.__file__).resolve())
    manifest = _candidate_manifest(
        _sha256(job_environment.recipe_path), compiler_sha
    )
    manifest["training_authorized"] = True
    library = SimpleNamespace(manifest=manifest)
    write_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *a, **k: library,
    )
    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        lambda *a, **k: write_calls.append((a, k)),
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="must be false",
    ):
        cli.run(job_environment.args)

    assert not write_calls
    assert not job_environment.output.exists()


def test_nested_schema2_report_authorization_is_rejected_before_publication(
    job_environment, monkeypatch
):
    compiler_sha = _sha256(Path(cli.cmc.__file__).resolve())
    manifest = _candidate_manifest(
        _sha256(job_environment.recipe_path), compiler_sha
    )
    manifest["outputs"][0]["schema2_report"][
        "hardware_authorized"
    ] = True
    library = SimpleNamespace(manifest=manifest)
    write_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *args, **kwargs: library,
    )
    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        lambda *args, **kwargs: write_calls.append((args, kwargs)),
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="report authorization .* must be false",
    ):
        cli.run(job_environment.args)

    assert not write_calls
    assert not job_environment.output.exists()


def test_schema2_tool_hash_must_bind_loaded_builder_before_publication(
    job_environment, monkeypatch
):
    compiler_sha = _sha256(Path(cli.cmc.__file__).resolve())
    manifest = _candidate_manifest(
        _sha256(job_environment.recipe_path), compiler_sha
    )
    first = manifest["outputs"][0]
    first["schema2_manifest"]["hashes"]["tool_sha256"] = "0" * 64
    first["schema2_report"]["hashes"]["tool_sha256"] = "0" * 64
    library = SimpleNamespace(manifest=manifest)
    write_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *args, **kwargs: library,
    )
    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        lambda *args, **kwargs: write_calls.append((args, kwargs)),
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="builder tool SHA-256 does not bind loaded code",
    ):
        cli.run(job_environment.args)

    assert not write_calls
    assert not job_environment.output.exists()


def test_schema2_input_hash_must_be_strict_before_publication(
    job_environment, monkeypatch
):
    compiler_sha = _sha256(Path(cli.cmc.__file__).resolve())
    manifest = _candidate_manifest(
        _sha256(job_environment.recipe_path), compiler_sha
    )
    first = manifest["outputs"][0]
    first["schema2_manifest"]["hashes"]["input_sha256"] = "not-a-digest"
    first["schema2_report"]["hashes"]["input_sha256"] = "not-a-digest"
    library = SimpleNamespace(manifest=manifest)
    write_calls = []
    monkeypatch.setattr(
        cli.cmc,
        "compile_loaded_canonical_motion_library",
        lambda *args, **kwargs: library,
    )
    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        lambda *args, **kwargs: write_calls.append((args, kwargs)),
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="manifest hash input_sha256 must be a 64-character",
    ):
        cli.run(job_environment.args)

    assert not write_calls
    assert not job_environment.output.exists()


def test_missing_published_npz_blocks_final_run_receipt(
    job_environment, monkeypatch
):
    _install_fake_compiler(monkeypatch, job_environment)

    def incomplete_writer(library, output_directory):
        output = Path(output_directory)
        output.mkdir()
        (output / cli.cmc.BUILD_MANIFEST_NAME).write_text(
            json.dumps(library.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        incomplete_writer,
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="missing, duplicate, or unexpected files",
    ):
        cli.run(job_environment.args)

    assert job_environment.output.exists()
    assert not (job_environment.output / cli.RUN_RECEIPT_NAME).exists()


def test_published_npz_must_match_sidecar_output_hash(
    job_environment, monkeypatch
):
    _install_fake_compiler(monkeypatch, job_environment)
    ordinary_writer = cli.cmc.write_compiled_canonical_motion_library

    def tampered_writer(library, output_directory):
        output = ordinary_writer(library, output_directory)
        first = library.manifest["outputs"][0]
        (output / first["filename"]).write_bytes(b"tampered-candidate")
        return output

    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        tampered_writer,
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="published output 0 SHA-256 mismatch",
    ):
        cli.run(job_environment.args)

    assert job_environment.output.exists()
    assert not (job_environment.output / cli.RUN_RECEIPT_NAME).exists()


def test_published_report_sidecar_must_equal_build_manifest(
    job_environment, monkeypatch
):
    _install_fake_compiler(monkeypatch, job_environment)
    ordinary_writer = cli.cmc.write_compiled_canonical_motion_library

    def mismatched_writer(library, output_directory):
        output = ordinary_writer(library, output_directory)
        first = library.manifest["outputs"][0]
        report_path = (
            output / first["filename"]
        ).with_suffix(".npz.report.json")
        report = dict(first["schema2_report"])
        report["frames"] = report["frames"] + 1
        _write_json(report_path, report)
        return output

    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        mismatched_writer,
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="report sidecar disagrees with BUILD_MANIFEST",
    ):
        cli.run(job_environment.args)

    assert job_environment.output.exists()
    assert not (job_environment.output / cli.RUN_RECEIPT_NAME).exists()


def test_published_sidecar_symlink_is_rejected_even_if_bytes_match(
    job_environment, monkeypatch, tmp_path
):
    _install_fake_compiler(monkeypatch, job_environment)
    ordinary_writer = cli.cmc.write_compiled_canonical_motion_library

    def symlinked_writer(library, output_directory):
        output = ordinary_writer(library, output_directory)
        first = library.manifest["outputs"][0]
        report_path = (
            output / first["filename"]
        ).with_suffix(".npz.report.json")
        external = tmp_path / "matching-external-report.json"
        _write_json(external, first["schema2_report"])
        report_path.unlink()
        report_path.symlink_to(external)
        return output

    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        symlinked_writer,
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="must be a real regular file, not a symlink",
    ):
        cli.run(job_environment.args)

    assert job_environment.output.exists()
    assert not (job_environment.output / cli.RUN_RECEIPT_NAME).exists()


def test_published_three_file_matrix_rejects_any_extra_file(
    job_environment, monkeypatch
):
    _install_fake_compiler(monkeypatch, job_environment)
    ordinary_writer = cli.cmc.write_compiled_canonical_motion_library

    def writer_with_extra(library, output_directory):
        output = ordinary_writer(library, output_directory)
        (output / "unexpected.txt").write_text(
            "not part of the compiler contract\n",
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        writer_with_extra,
    )

    with pytest.raises(
        cli.CanonicalMotionCompileCliError,
        match="missing, duplicate, or unexpected files",
    ):
        cli.run(job_environment.args)

    assert job_environment.output.exists()
    assert not (job_environment.output / cli.RUN_RECEIPT_NAME).exists()


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="atomic publication identity boundary requires POSIX signal masks",
)
def test_signal_after_atomic_publication_waits_for_identity_then_cleans(
    job_environment, monkeypatch
):
    _install_fake_compiler(monkeypatch, job_environment)
    ordinary_writer = cli.cmc.write_compiled_canonical_motion_library

    def writer_signals_after_publication(library, output_directory):
        published = ordinary_writer(library, output_directory)
        os.kill(os.getpid(), signal.SIGTERM)
        return published

    monkeypatch.setattr(
        cli.cmc,
        "write_compiled_canonical_motion_library",
        writer_signals_after_publication,
    )

    with pytest.raises(
        cli.CanonicalMotionCompileTerminated,
        match="SIGTERM",
    ):
        cli.run(job_environment.args)

    assert not job_environment.output.exists()
    assert not (job_environment.output / cli.RUN_RECEIPT_NAME).exists()
    quarantines = list(
        job_environment.output.parent.glob(
            f".{job_environment.output.name}.terminated-*"
        )
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / cli.cmc.BUILD_MANIFEST_NAME).is_file()
    assert not (quarantines[0] / cli.RUN_RECEIPT_NAME).exists()


def _pid_is_running(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        path = Path("/proc") / str(pid) / "stat"
        try:
            value = path.read_text(encoding="utf-8")
        except OSError:
            return False
        right_parenthesis = value.rfind(")")
        return (
            right_parenthesis >= 0
            and right_parenthesis + 2 < len(value)
            and value[right_parenthesis + 2] != "Z"
        )
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_linux_process_without_pidfd_is_never_signaled_by_raw_pid(
    monkeypatch,
):
    process = cli._ObservedProcess(
        pid=987654,
        linux_starttime=123,
        pidfd=None,
    )

    def forbidden_raw_kill(*args, **kwargs):
        raise AssertionError("Linux stable-handle failure must not use os.kill")

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.os, "kill", forbidden_raw_kill)

    cli._send_observed_signal(process, signal.SIGTERM)
    cli._send_observed_signal(process, signal.SIGKILL)


def test_descendant_rescan_upgrades_transient_identity_and_pidfd(
    monkeypatch,
):
    closed = []

    def record_close(process):
        if process.pidfd is not None:
            closed.append(process.pidfd)
            process.pidfd = None

    monkeypatch.setattr(cli, "_close_observed_process", record_close)

    unstable = cli._ObservedProcess(
        pid=111,
        linux_starttime=None,
        pidfd=None,
    )
    stable = cli._ObservedProcess(
        pid=111,
        linux_starttime=222,
        pidfd=333,
    )
    observed = {111: unstable}
    added = cli._merge_observed_descendants(observed, {111: stable})

    assert observed[111] is stable
    assert added == [stable]
    assert stable.pidfd == 333
    assert closed == []

    existing = cli._ObservedProcess(
        pid=444,
        linux_starttime=555,
        pidfd=None,
    )
    later_handle = cli._ObservedProcess(
        pid=444,
        linux_starttime=555,
        pidfd=666,
    )
    observed = {444: existing}
    added = cli._merge_observed_descendants(
        observed, {444: later_handle}
    )

    assert observed[444] is existing
    assert added == [existing]
    assert existing.pidfd == 666
    assert later_handle.pidfd is None
    assert closed == []


@pytest.mark.parametrize(
    "termination_signal",
    _CATCHABLE_TEST_SIGNALS,
    ids=lambda value: signal.Signals(value).name.lower(),
)
def test_parent_signal_reaps_process_pool_and_removes_owned_publication(
    tmp_path: Path, termination_signal: int,
):
    output = tmp_path / "terminated-output"
    ready_path = tmp_path / "parent-ready"
    pid_path = tmp_path / "worker-pids.json"
    script = textwrap.dedent(
        """
        import concurrent.futures
        import json
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace

        scripts, output_raw, ready_raw, pids_raw = sys.argv[1:]
        sys.path.insert(0, scripts)
        import canonical_motion_compile_cli as cli

        output = Path(output_raw)
        ready = Path(ready_raw)
        pids = Path(pids_raw)
        parsed = SimpleNamespace(dry_run=False)
        fake_job = SimpleNamespace(output_directory=output)
        manifest = {
            "publication_class": "compiler_candidate",
            "training_authorized": False,
            "hardware_authorized": False,
            "test_owner": "signal-cleanup-subprocess",
        }

        class Parser:
            def parse_args(self, argv):
                return parsed

        cli._parser = lambda: Parser()
        cli._validate_job = lambda args: fake_job
        cli._verify_unchanged_job_inputs = lambda job: None

        def blocked_build(job):
            cli._bind_termination_publication(output, manifest)
            output.mkdir()
            (output / cli.cmc.BUILD_MANIFEST_NAME).write_text(
                json.dumps(manifest, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            cli._mark_termination_publication_published(output)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=3
            ) as executor:
                futures = [
                    executor.submit(time.sleep, 300.0)
                    for _ in range(3)
                ]
                deadline = time.monotonic() + 10.0
                while len(executor._processes) < 3:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("workers did not start")
                    time.sleep(0.01)
                pids.write_text(
                    json.dumps(sorted(executor._processes)) + "\\n",
                    encoding="utf-8",
                )
                ready.write_text("ready\\n", encoding="utf-8")
                for future in futures:
                    future.result()

        cli._run_build = blocked_build
        raise SystemExit(cli.main([]))
        """
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(SCRIPTS),
            str(output),
            str(ready_path),
            str(pid_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 15.0
    while not ready_path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"signal-test parent exited early: {process.returncode}; "
                f"stdout={stdout!r}; stderr={stderr!r}"
            )
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("signal-test workers did not become ready")
        time.sleep(0.02)
    worker_pids = json.loads(pid_path.read_text(encoding="utf-8"))
    if len(worker_pids) != 3:
        process.kill()
        process.wait(timeout=5)
        pytest.fail(f"expected three workers, got {worker_pids}")
    if not all(_pid_is_running(pid) for pid in worker_pids):
        process.kill()
        process.wait(timeout=5)
        pytest.fail("one or more signal-test workers exited before signaling")

    process.send_signal(termination_signal)
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(
            "signal-test parent did not terminate; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        )

    assert process.returncode == 128 + termination_signal
    assert stdout == ""
    assert "TERMINATED:" in stderr
    deadline = time.monotonic() + 5.0
    while (
        any(_pid_is_running(pid) for pid in worker_pids)
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert not any(_pid_is_running(pid) for pid in worker_pids)
    assert not output.exists()
    assert not (output / cli.RUN_RECEIPT_NAME).exists()
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))
    quarantines = list(
        tmp_path.glob(f".{output.name}.terminated-{process.pid}-*")
    )
    assert len(quarantines) == 1
    quarantined_manifest = json.loads(
        (quarantines[0] / cli.cmc.BUILD_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert quarantined_manifest["test_owner"] == (
        "signal-cleanup-subprocess"
    )
    assert not (quarantines[0] / cli.RUN_RECEIPT_NAME).exists()


def test_signal_rescan_reaps_worker_started_after_first_snapshot(
    tmp_path: Path,
):
    ready_path = tmp_path / "rescan-ready"
    pid_path = tmp_path / "rescan-worker-pids.json"
    script = textwrap.dedent(
        """
        import json
        import multiprocessing
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace

        scripts, ready_raw, pids_raw = sys.argv[1:]
        sys.path.insert(0, scripts)
        import canonical_motion_compile_cli as cli

        ready = Path(ready_raw)
        pids = Path(pids_raw)
        parsed = SimpleNamespace(dry_run=False)
        fake_job = SimpleNamespace(output_directory=Path("unused"))
        fork = multiprocessing.get_context("fork")
        workers = []
        late_started = []

        class Parser:
            def parse_args(self, argv):
                return parsed

        cli._parser = lambda: Parser()
        cli._validate_job = lambda args: fake_job
        cli._verify_unchanged_job_inputs = lambda job: None

        def start_worker():
            worker = fork.Process(target=time.sleep, args=(300.0,))
            worker.start()
            workers.append(worker)
            pids.write_text(
                json.dumps([item.pid for item in workers]) + "\\n",
                encoding="utf-8",
            )

        ordinary_observer = cli._observed_descendants

        def observer_that_launches_after_snapshot(owner_pid):
            snapshot = ordinary_observer(owner_pid)
            if not late_started:
                late_started.append(True)
                start_worker()
            return snapshot

        cli._observed_descendants = observer_that_launches_after_snapshot

        def blocked_build(job):
            start_worker()
            ready.write_text("ready\\n", encoding="utf-8")
            while True:
                time.sleep(300.0)

        cli._run_build = blocked_build
        raise SystemExit(cli.main([]))
        """
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(SCRIPTS),
            str(ready_path),
            str(pid_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 15.0
    while not ready_path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"rescan-test parent exited early: {process.returncode}; "
                f"stdout={stdout!r}; stderr={stderr!r}"
            )
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("rescan-test initial worker did not become ready")
        time.sleep(0.02)

    process.send_signal(signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(
            "rescan-test parent did not terminate; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        )

    assert process.returncode == 128 + signal.SIGTERM
    assert stdout == ""
    assert "TERMINATED:" in stderr
    worker_pids = json.loads(pid_path.read_text(encoding="utf-8"))
    assert len(worker_pids) == 2
    deadline = time.monotonic() + 5.0
    while (
        any(_pid_is_running(pid) for pid in worker_pids)
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert not any(_pid_is_running(pid) for pid in worker_pids)


def test_writer_cleans_staging_when_termination_bypasses_exception(
    tmp_path: Path, monkeypatch
):
    motion = SimpleNamespace(
        schema2=object(),
        filename="candidate.npz",
    )
    library = cli.cmc.CompiledLibrary(
        recipe=None,
        motions=tuple(motion for _ in range(10)),
        manifest={},
    )
    output = tmp_path / "writer-interrupted"

    def terminate(*args, **kwargs):
        raise cli.CanonicalMotionCompileTerminated(signal.SIGTERM)

    monkeypatch.setattr(cli.cmc, "write_schema2_candidate", terminate)

    with pytest.raises(cli.CanonicalMotionCompileTerminated):
        cli.cmc.write_compiled_canonical_motion_library(library, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))


def test_termination_cleanup_refuses_replaced_output_directory(
    tmp_path: Path,
):
    output = tmp_path / "owned-output"
    displaced = tmp_path / "displaced-owned-output"
    manifest = {
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "hardware_authorized": False,
        "test_owner": "directory-identity-test",
    }

    with cli._termination_guard() as state:
        cli._bind_termination_publication(output, manifest)
        output.mkdir()
        (output / cli.cmc.BUILD_MANIFEST_NAME).write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cli._mark_termination_publication_published(output)

        output.rename(displaced)
        output.mkdir()
        (output / cli.cmc.BUILD_MANIFEST_NAME).write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replacement_marker = output / "replacement-owner.txt"
        replacement_marker.write_text("must survive\n", encoding="utf-8")

        with pytest.raises(
            cli.CanonicalMotionCompileCliError,
            match="replaced output directory",
        ):
            cli._cleanup_terminated_publication(state)

    assert displaced.is_dir()
    assert output.is_dir()
    assert replacement_marker.read_text(encoding="utf-8") == "must survive\n"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG is a Linux worker guarantee",
)
def test_linux_parent_sigkill_triggers_worker_parent_death_signal(
    tmp_path: Path,
):
    output = tmp_path / "hard-death-output"
    ready_path = tmp_path / "hard-death-ready"
    pid_path = tmp_path / "hard-death-worker-pids.json"
    script = textwrap.dedent(
        """
        import concurrent.futures
        import json
        import signal
        import sys
        import time
        from pathlib import Path
        from types import SimpleNamespace

        scripts, output_raw, ready_raw, pids_raw = sys.argv[1:]
        sys.path.insert(0, scripts)
        import canonical_motion_compile_cli as cli

        output = Path(output_raw)
        ready = Path(ready_raw)
        pids = Path(pids_raw)
        parsed = SimpleNamespace(dry_run=False)
        fake_job = SimpleNamespace(output_directory=output)

        class Parser:
            def parse_args(self, argv):
                return parsed

        cli._parser = lambda: Parser()
        cli._validate_job = lambda args: fake_job
        cli._verify_unchanged_job_inputs = lambda job: None

        def blocked_build(job):
            # Exercise the inheritance trap explicitly: Linux fork workers
            # must unblock SIGTERM before installing PR_SET_PDEATHSIG.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=2
            ) as executor:
                futures = [
                    executor.submit(time.sleep, 300.0)
                    for _ in range(2)
                ]
                deadline = time.monotonic() + 10.0
                while len(executor._processes) < 2:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("workers did not start")
                    time.sleep(0.01)
                pids.write_text(
                    json.dumps(sorted(executor._processes)) + "\\n",
                    encoding="utf-8",
                )
                ready.write_text("ready\\n", encoding="utf-8")
                for future in futures:
                    future.result()

        cli._run_build = blocked_build
        raise SystemExit(cli.main([]))
        """
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(SCRIPTS),
            str(output),
            str(ready_path),
            str(pid_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 15.0
    while not ready_path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"hard-death parent exited early: {process.returncode}; "
                f"stdout={stdout!r}; stderr={stderr!r}"
            )
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5)
            pytest.fail("hard-death workers did not become ready")
        time.sleep(0.02)
    worker_pids = json.loads(pid_path.read_text(encoding="utf-8"))
    if len(worker_pids) != 2:
        process.kill()
        process.wait(timeout=5)
        pytest.fail(f"expected two hard-death workers, got {worker_pids}")
    if not all(_pid_is_running(pid) for pid in worker_pids):
        process.kill()
        process.wait(timeout=5)
        pytest.fail("one or more hard-death workers exited before SIGKILL")

    process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        pytest.fail("hard-death parent did not exit after SIGKILL")

    assert process.returncode == -signal.SIGKILL
    deadline = time.monotonic() + 5.0
    while (
        any(_pid_is_running(pid) for pid in worker_pids)
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert not any(_pid_is_running(pid) for pid in worker_pids)
    assert not output.exists()
    assert not (output / cli.RUN_RECEIPT_NAME).exists()
