"""Tests for byte-preserving fresh-N5 5+2 bank materialization."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_stroke_prototypes as prototypes  # noqa: E402
import materialize_canonical_motion_composition as composition  # noqa: E402
import test_canonical_motion_bank_gate as bank_support  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path):
    base = bank_support.BankFixture(tmp_path)
    bank_support._attach_exact_time_law_artifacts(base)
    append = bank_support.AppendBankFixture(base)
    append.manifest["station_center_shift_xy_m"] = [0.0, 0.0]
    append.manifest["append_only_composition"][
        "station_center_shift_xy_m"
    ] = [0.0, 0.0]
    append.write_manifest()
    append.weighted_arc_tool_path = base.weighted_arc_tool_path
    append.ready_root_pos = base.ready_root_pos
    append.hashes = base.hashes
    bank_support._attach_exact_time_law_artifacts(append)
    output = tmp_path / "composed"
    return base, append, output


def _run(base, append, output: Path):
    return composition.materialize(
        base_manifest_path=base.manifest_path,
        expected_base_manifest_sha256=_sha(base.manifest_path),
        base_bank_dir=base.bank,
        append_manifest_path=append.manifest_path,
        expected_append_manifest_sha256=_sha(append.manifest_path),
        append_bank_dir=append.bank,
        output_directory=output,
    )


def test_materializes_exact_byte_preserving_7x2_bank(tmp_path: Path):
    base, append, output = _inputs(tmp_path)

    receipt = _run(base, append, output)

    assert receipt["verdict"] == "PASS_BYTE_PRESERVING_MATERIALIZATION"
    assert receipt["contract"]["candidate_count"] == 14
    assert receipt["contract"]["motion_bytes_modified"] is False
    assert receipt["artifact_file_count"] == 70
    assert len(list(output.iterdir())) == 72
    manifest = json.loads(
        (output / composition.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["output_matrix"] == {
        "motion_ids": list(composition.COMPOSED_MOTION_IDS),
        "scopes": list(composition.SCOPES),
        "candidate_count": 14,
    }
    selected = prototypes._fresh_n5_upper_outputs(manifest)
    assert tuple(row["motion_id"] for row in selected) == (
        "bh_loop_c",
        "v12_forehand_block",
        "bh_block",
        "s0_highpress",
        "fh_loop_high",
    )
    for source in (base, append):
        for row in source.outputs:
            names = (
                row["filename"],
                row["filename"] + ".manifest.json",
                row["filename"] + ".report.json",
                row["time_law_artifact"]["npz_filename"],
                row["time_law_artifact"]["manifest_filename"],
            )
            for name in names:
                assert (output / name).read_bytes() == (
                    source.bank / name
                ).read_bytes()


def test_missing_artifact_fails_before_publication(tmp_path: Path):
    base, append, output = _inputs(tmp_path)
    missing = (
        append.bank
        / append.outputs[0]["time_law_artifact"]["manifest_filename"]
    )
    missing.unlink()

    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="time-law strict reopen failed",
    ):
        _run(base, append, output)

    assert not output.exists()


def test_reordered_or_duplicate_output_rows_fail_closed(tmp_path: Path):
    base, append, output = _inputs(tmp_path)
    append.manifest["outputs"][0], append.manifest["outputs"][1] = (
        append.manifest["outputs"][1],
        append.manifest["outputs"][0],
    )
    append.write_manifest()

    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="changed order or identity",
    ):
        _run(base, append, output)

    assert not output.exists()

    append.manifest["outputs"][0] = append.manifest["outputs"][1]
    append.write_manifest()
    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="changed order or identity",
    ):
        _run(base, append, output)
    assert not output.exists()


def test_symlink_and_hash_drift_fail_closed(tmp_path: Path):
    base, append, output = _inputs(tmp_path)
    row = append.outputs[0]
    artifact = append.bank / row["filename"]
    external = tmp_path / "external.npz"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="regular non-symlink",
    ):
        _run(base, append, output)
    assert not output.exists()

    artifact.unlink()
    artifact.write_bytes(external.read_bytes() + b"drift")
    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="NPZ hash drifted",
    ):
        _run(base, append, output)
    assert not output.exists()


def test_manifest_sha_and_existing_output_are_no_clobber(tmp_path: Path):
    base, append, output = _inputs(tmp_path)
    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="SHA-256 mismatch",
    ):
        composition.materialize(
            base_manifest_path=base.manifest_path,
            expected_base_manifest_sha256="0" * 64,
            base_bank_dir=base.bank,
            append_manifest_path=append.manifest_path,
            expected_append_manifest_sha256=_sha(append.manifest_path),
            append_bank_dir=append.bank,
            output_directory=output,
        )
    assert not output.exists()

    output.mkdir()
    marker = output / "owner"
    marker.write_text("preexisting\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(base, append, output)
    assert marker.read_text(encoding="utf-8") == "preexisting\n"


def test_station_shift_is_never_silently_materialized(tmp_path: Path):
    base, append, output = _inputs(tmp_path)
    append.manifest["station_center_shift_xy_m"] = [-0.05, 0.0]
    append.manifest["append_only_composition"][
        "station_center_shift_xy_m"
    ] = [-0.05, 0.0]
    append.write_manifest()

    with pytest.raises(
        composition.CanonicalMotionCompositionError,
        match="station-shifted",
    ):
        _run(base, append, output)
    assert not output.exists()
