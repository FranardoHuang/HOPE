"""Host contracts for the standalone arbitrary-N bank-gate profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
for directory in (SCRIPTS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import canonical_motion_bank_gate as legacy_gate  # noqa: E402
import canonical_motion_compiler as compiler  # noqa: E402
import canonical_motion_generic_bank_gate as generic  # noqa: E402
import test_canonical_motion_arbitrary_bank as producer_tests  # noqa: E402
import test_canonical_motion_compiler as compiler_fixtures  # noqa: E402


def _compiled_fixture(tmp_path: Path, monkeypatch):
    fixture = producer_tests.BankFixture(tmp_path, 1, monkeypatch)
    loaded = fixture.load()
    monkeypatch.setattr(
        compiler,
        "build_schema2_candidate",
        compiler_fixtures._fake_schema2_builder,
    )
    monkeypatch.setattr(
        compiler,
        "build_canonical_geometry",
        compiler_fixtures._compiler_plumbing_geometry,
    )
    monkeypatch.setattr(
        compiler,
        "retime_path",
        compiler_fixtures._fast_marker_only_retime,
    )
    output = tmp_path / "compiled"
    producer_tests.arbitrary.compile_arbitrary_bank(
        loaded,
        output_directory=output,
        backend=compiler_fixtures.FakePlantBackend(),
    )
    return fixture, loaded, output


def _raw_complete_report(motion_id: str) -> dict:
    return {
        "schema_version": 1,
        "verdict": "PASS",
        "bank_gate_pass": True,
        "candidate_integrity_pass": True,
        "grounded_trace_status": (
            "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
        ),
        "publication_class": "post_build_diagnostic_only",
        "training_authorized": False,
        "hardware_authorized": False,
        "library_id": "fixture_arbitrary_1",
        "manifest": {"path": "/fixture/BUILD_MANIFEST.json", "sha256": "a" * 64},
        "bank_dir": "/fixture",
        "bound_inputs": {
            "swept_clearance_receipt": {"path": "/swept", "sha256": "b" * 64},
            "verifier_tools": {
                "bank_gate": {"path": "/legacy", "sha256": "c" * 64},
                "mujoco_motion_player": {
                    "path": "/player",
                    "sha256": "d" * 64,
                },
                "canonical_mujoco_dynamics_gate": {
                    "path": "/dynamics",
                    "sha256": "e" * 64,
                    "report_schema_version": 1,
                },
            },
        },
        "contracts": {
            "matrix": {
                "motion_ids": [motion_id],
                "scopes": ["upper", "full"],
                "count": 2,
            },
            "grounded_trace_status": (
                "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
            ),
            "swept_clearance": {"complete": True},
        },
        "aggregate": {
            "clip_count": 2,
            "swept_clearance_pass_count": 2,
            "swept_clearance_minimum_certified_lower_bound_m": 0.01,
            "time_law_artifact_count": 2,
            "grounded_lmr_pass_count": 2,
            "grounded_lmr_incomplete_count": 0,
        },
        "clips": [
            {
                "motion_id": motion_id,
                "scope": scope,
                "canonical_time_law": {"status": "PASS"},
                "grounded_left_midpoint_right": {"status": "PASS"},
            }
            for scope in ("upper", "full")
        ],
        "non_claims": [],
    }


def test_preflight_reopens_exact_pair_hash_and_compiled_timing(
    tmp_path: Path,
    monkeypatch,
):
    _fixture, loaded, output = _compiled_fixture(tmp_path, monkeypatch)

    manifest, matrix = generic._manifest_and_matrix(
        loaded,
        output / legacy_gate.MANIFEST_NAME,
        output,
    )

    assert manifest["output_matrix"]["candidate_count"] == 2
    assert tuple(matrix) == (
        (loaded.motion_ids[0], "upper"),
        (loaded.motion_ids[0], "full"),
    )


def test_preflight_missing_pair_and_bad_recovery_fail_closed(
    tmp_path: Path,
    monkeypatch,
):
    _fixture, loaded, output = _compiled_fixture(tmp_path, monkeypatch)
    missing = output / f"{loaded.motion_ids[0]}_full_canonical_v2.npz"
    missing.unlink()
    with pytest.raises(generic.GenericBankGateError, match="not the exact"):
        generic._manifest_and_matrix(
            loaded,
            output / legacy_gate.MANIFEST_NAME,
            output,
        )

    _fixture, loaded, output = _compiled_fixture(
        tmp_path / "timing", monkeypatch
    )
    manifest_path = output / legacy_gate.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["source_anchor_time_s"] = manifest["outputs"][0][
        "duration_s"
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        generic.GenericBankGateError,
        match="required recovery",
    ):
        generic._manifest_and_matrix(loaded, manifest_path, output)


def test_engine_matrix_view_is_complete_and_always_restored(
    tmp_path: Path,
    monkeypatch,
):
    fixture = producer_tests.BankFixture(tmp_path, 5, monkeypatch)
    loaded = fixture.load()
    old = (
        legacy_gate.MOTION_IDS,
        legacy_gate.EXPECTED_MATRIX,
        legacy_gate.EXPECTED_FILENAMES,
    )
    observed = {}

    def fake_verify(*args, **kwargs):
        del args
        observed["ids"] = legacy_gate.MOTION_IDS
        observed["matrix"] = legacy_gate.EXPECTED_MATRIX
        observed["recipe"] = kwargs["recipe_loader"](loaded.path)
        raise RuntimeError("fixture stop")

    monkeypatch.setattr(
        legacy_gate, "verify_canonical_motion_bank", fake_verify
    )
    with pytest.raises(RuntimeError, match="fixture stop"):
        generic._run_independent_engine(
            loaded=loaded,
            manifest_path=tmp_path / "BUILD_MANIFEST.json",
            bank_directory=tmp_path,
            mjcf_path=tmp_path / "model.xml",
            urdf_path=tmp_path / "model.urdf",
            body_order_path=tmp_path / "body.txt",
            expected_compiled_signature="a" * 64,
            swept_clearance_receipt_path=tmp_path / "swept.json",
            expected_swept_clearance_receipt_sha256="b" * 64,
            engine_recipe=generic._engine_recipe(loaded),
        )

    assert observed["ids"] == loaded.motion_ids
    assert observed["matrix"] == tuple(
        (motion_id, scope)
        for motion_id in loaded.motion_ids
        for scope in ("upper", "full")
    )
    assert observed["recipe"].raw["motion_specs"][-1] == {
        "motion_id": "s0_highpress",
        "scope_overrides": {
            "full": {"maximum_grounding_offset_m": 0.0}
        },
        "verifier_only_option_sentinel": True,
    }
    assert (
        legacy_gate.MOTION_IDS,
        legacy_gate.EXPECTED_MATRIX,
        legacy_gate.EXPECTED_FILENAMES,
    ) == old


def test_generic_projection_binds_wrapper_and_selected_registry(
    tmp_path: Path,
):
    del tmp_path
    motion_id = "take_000_unit00_bh"
    selected = MappingProxyType(
        {
            "scope": "full",
            "registry_sha256": "1" * 64,
            "alignment_sha256": "2" * 64,
            "canonical_ready_sha256": "3" * 64,
            "canonical_ready_fk_sha256": "4" * 64,
            "motion_ids": [motion_id],
            "npz_sha256": ["5" * 64],
            "build_manifest_sha256": ["6" * 64],
        }
    )

    report = generic._generic_v2_report(
        _raw_complete_report(motion_id),
        selected_registry_binding=selected,
    )

    assert report["schema_version"] == generic.REPORT_SCHEMA_VERSION == 2
    assert report["grounded_trace_status"] == "COMPLETE_PASS"
    assert report["selected_registry_binding"] == dict(selected)
    assert "swept_clearance_receipt" not in report["bound_inputs"]
    assert "swept_clearance" not in report["contracts"]
    assert "time_law_artifact_count" not in report["aggregate"]
    assert all(
        "canonical_time_law" not in row for row in report["clips"]
    )
    assert report["bound_inputs"]["verifier_tools"]["bank_gate"] == {
        "path": str(Path(generic.__file__).resolve()),
        "sha256": generic._sha256_file(Path(generic.__file__).resolve()),
    }


def test_generic_failure_report_keeps_v2_discriminator():
    report = generic._failure_report(RuntimeError("fixture failure"))

    assert report["schema_version"] == generic.REPORT_SCHEMA_VERSION == 2
    assert report["bank_gate_pass"] is False


def test_generic_projection_rejects_nonlegacy_engine_discriminator():
    raw = _raw_complete_report("take_000_unit00_bh")
    raw["schema_version"] = 2
    selected = {
        "scope": "full",
        "registry_sha256": "1" * 64,
        "alignment_sha256": "2" * 64,
        "canonical_ready_sha256": "3" * 64,
        "canonical_ready_fk_sha256": "4" * 64,
        "motion_ids": ["take_000_unit00_bh"],
        "npz_sha256": ["5" * 64],
        "build_manifest_sha256": ["6" * 64],
    }

    with pytest.raises(generic.GenericBankGateError, match="independent engine"):
        generic._generic_v2_report(
            raw,
            selected_registry_binding=selected,
        )


def test_registry_binding_rejects_wrong_order_before_engine(
    tmp_path: Path,
    monkeypatch,
):
    fixture, loaded, output = _compiled_fixture(tmp_path, monkeypatch)
    manifest, matrix = generic._manifest_and_matrix(
        loaded,
        output / legacy_gate.MANIFEST_NAME,
        output,
    )
    fake_registry = SimpleNamespace(
        schema_version=registry_schema_version(),
        bank_id=fixture.recipe["bank_id"],
        motion_ids=("other_motion",),
        canonical_ready_sha256=loaded.canonical_recipe.ready.sha256,
    )
    monkeypatch.setattr(
        generic.registry_module,
        "load_canonical_motion_bank_registry",
        lambda *args, **kwargs: fake_registry,
    )

    with pytest.raises(generic.GenericBankGateError, match="bank/order/ready"):
        generic._registry_binding(
            registry_path=tmp_path / "registry.json",
            expected_registry_sha256="a" * 64,
            repo_root=tmp_path,
            loaded=loaded,
            manifest=manifest,
            matrix=matrix,
        )


def registry_schema_version() -> int:
    return generic.registry_module.GENERIC_REGISTRY_SCHEMA_VERSION
