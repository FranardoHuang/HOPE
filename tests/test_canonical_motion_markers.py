"""Tests for the externally pinned, non-authorizing marker authority v2."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from canonical_motion_markers import (  # noqa: E402
    CANONICAL_MOTION_IDS,
    FULL_FRAME_CLASSIFICATION,
    LEGACY_AUTHORITY_PATH,
    LEGACY_AUTHORITY_SHA256,
    MARKER_AUTHORITY_REVIEW_STATUS,
    TEMPORAL_INDEX_CLASSIFICATION,
    MarkerSemanticsError,
    load_canonical_motion_markers,
    sha256_file,
)


_AUTHORITY = _ROOT / "configs" / "canonical_motion_marker_semantics_v2_20260724.json"
_AUTHORITY_SHA256 = "0bf36c204162bf63e044db129bdb869129f36fb7a6d191a7c83effc4ba300929"
_LEGACY_AUTHORITY = _ROOT / LEGACY_AUTHORITY_PATH


def _raw() -> dict:
    return json.loads(_AUTHORITY.read_text(encoding="utf-8"))


def _write_and_load(tmp_path: Path, raw: dict):
    path = tmp_path / "authority.json"
    path.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return load_canonical_motion_markers(
        path,
        expected_authority_sha256=sha256_file(path),
        repo_root=_ROOT,
    )


def _row(raw: dict, motion_id: str) -> dict:
    return next(row for row in raw["motions"] if row["motion_id"] == motion_id)


def test_loads_externally_pinned_v2_with_separated_semantics():
    assert sha256_file(_AUTHORITY) == _AUTHORITY_SHA256
    assert sha256_file(_LEGACY_AUTHORITY) == LEGACY_AUTHORITY_SHA256
    loaded = load_canonical_motion_markers(
        _AUTHORITY,
        expected_authority_sha256=_AUTHORITY_SHA256,
        repo_root=_ROOT,
    )
    assert loaded.review_status == MARKER_AUTHORITY_REVIEW_STATUS
    assert loaded.legacy_authority_sha256 == LEGACY_AUTHORITY_SHA256
    assert tuple(row.motion_id for row in loaded.rows) == CANONICAL_MOTION_IDS
    assert isinstance(loaded.rows, tuple)

    fh = loaded.row("fh_loop")
    assert fh.nominal_event == 61
    assert fh.ge50_seed == (47, 55)
    assert fh.ge80_seed == (48, 55)
    assert fh.preferred_seed == 49
    assert fh.historical_adv2c3_start == 40
    assert fh.construction_marker is None
    assert fh.frame_identity is not None
    assert fh.frame_identity.classification == TEMPORAL_INDEX_CLASSIFICATION
    assert fh.frame_identity.invariant_joint_count == 28
    assert fh.frame_identity.excluded_joint_indices == (26, 28, 30)
    assert fh.frame_identity.full_pose_identity_claimed is False

    block = loaded.row("bh_block")
    assert block.frame_identity is not None
    assert block.frame_identity.classification == FULL_FRAME_CLASSIFICATION
    assert block.frame_identity.invariant_joint_count == 31
    assert block.frame_identity.excluded_joint_indices == ()
    assert block.frame_identity.full_pose_identity_claimed is True

    synthetic = loaded.row("fh_block_syn")
    assert synthetic.nominal_event is None
    assert synthetic.preferred_seed is None
    assert synthetic.frame_identity is None
    assert synthetic.construction_marker is not None
    assert synthetic.construction_marker.annotation_frame == 44
    assert synthetic.construction_marker.donor_preferred_frame == 42
    assert synthetic.construction_marker.solve_span == (34, 48)
    assert not hasattr(synthetic, "certified_window")
    with pytest.raises(FrozenInstanceError):
        synthetic.preferred_seed = 42  # type: ignore[misc]


def test_authority_bytes_must_match_external_sha_pin():
    with pytest.raises(MarkerSemanticsError, match="SHA-256 mismatch"):
        load_canonical_motion_markers(
            _AUTHORITY,
            expected_authority_sha256="0" * 64,
            repo_root=_ROOT,
        )


def test_loader_hashes_and_parses_one_authority_byte_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "authority.json"
    original = _AUTHORITY.read_bytes()
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    original_read_bytes = Path.read_bytes
    authority_reads = 0

    def swap_after_snapshot(self):
        nonlocal authority_reads
        payload = original_read_bytes(self)
        if self == path:
            authority_reads += 1
            if authority_reads == 1:
                path.write_text('{"swapped": true}', encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", swap_after_snapshot)
    loaded = load_canonical_motion_markers(
        path,
        expected_authority_sha256=expected,
        repo_root=_ROOT,
    )
    assert loaded.row("fh_loop").historical_adv2c3_start == 40
    assert authority_reads == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.__setitem__("extra", 1), "keys changed"),
        (
            lambda raw: raw["authorization"].__setitem__("training_authorized", True),
            "training_authorized",
        ),
        (
            lambda raw: raw.__setitem__(
                "review_status", "REVIEWED_FOR_POST_RETIME_BEHAVIOR"
            ),
            "review_status",
        ),
        (
            lambda raw: raw["semantic_contract"].__setitem__(
                "contact_truth_observed", True
            ),
            "contact_truth_observed",
        ),
        (
            lambda raw: raw["semantic_contract"].__setitem__(
                "post_retime_behavior_rescan_required", False
            ),
            "post_retime_behavior_rescan_required",
        ),
        (
            lambda raw: raw["shared_evidence"][
                "frame_identity_receipts_summary"
            ].__setitem__("sha256", "0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda raw: raw["shared_evidence"]["legacy_timing_table"].__setitem__(
                "extra", "drift"
            ),
            "keys changed",
        ),
    ],
)
def test_top_level_and_authority_semantics_fail_closed(tmp_path, mutation, match):
    raw = _raw()
    mutation(raw)
    with pytest.raises(MarkerSemanticsError, match=match):
        _write_and_load(tmp_path, raw)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda raw: raw["motions"][0].__setitem__("extra", 1),
            "keys changed",
        ),
        (
            lambda raw: raw["motions"][0]["bound_recipe_source"].__setitem__(
                "extra", 1
            ),
            "keys changed",
        ),
        (
            lambda raw: raw["motions"][0]["bound_recipe_source"].__setitem__(
                "sha256", "0" * 64
            ),
            "SHA-256 mismatch",
        ),
        (
            lambda raw: raw["motions"][0]["nominal_event"].__setitem__("frame", True),
            "integer",
        ),
        (
            lambda raw: raw["motions"][0]["nominal_event"].__setitem__(
                "semantic_kind", "observed_ball_contact"
            ),
            "air-swing",
        ),
        (
            lambda raw: raw["motions"][0]["nominal_event"].__setitem__(
                "contact_truth_observed", True
            ),
            "contact_truth_observed",
        ),
        (
            lambda raw: raw["motions"][0]["nominal_event"].__setitem__(
                "event_source_sha256", None
            ),
            "string",
        ),
        (
            lambda raw: raw["motions"][0]["legacy_ge50_seed"].__setitem__(
                "span_inclusive", [55, 47]
            ),
            "precedes",
        ),
        (
            lambda raw: raw["motions"][0]["legacy_ge80_seed"].__setitem__(
                "span_inclusive", [46, 55]
            ),
            "contained",
        ),
        (
            lambda raw: raw["motions"][0]["legacy_preferred_seed"].__setitem__(
                "frame", 99
            ),
            "outside ge50",
        ),
        (
            lambda raw: raw["motions"][0]["legacy_scan_evidence"].__setitem__(
                "extra", 1
            ),
            "keys changed",
        ),
        (
            lambda raw: raw["motions"][0]["legacy_scan_evidence"].__setitem__(
                "input_source_sha256", "0" * 64
            ),
            "bind bound_recipe_source",
        ),
        (
            lambda raw: raw["motions"][0]["legacy_scan_evidence"][
                "scan_contract"
            ].__setitem__("cli_overrides", [False]),
            "string",
        ),
        (
            lambda raw: raw["motions"][0]["post_retime_behavior_gate"].__setitem__(
                "required", False
            ),
            "required",
        ),
        (
            lambda raw: raw["motions"][0]["post_retime_behavior_gate"].__setitem__(
                "behavior_promotion_authorized", True
            ),
            "behavior_promotion_authorized",
        ),
    ],
)
def test_row_nested_markers_and_legacy_scan_provenance_are_strict(
    tmp_path, mutation, match
):
    raw = _raw()
    mutation(raw)
    with pytest.raises(MarkerSemanticsError, match=match):
        _write_and_load(tmp_path, raw)


def test_five_rows_are_unique_and_ordered(tmp_path):
    raw = _raw()
    raw["motions"][0], raw["motions"][1] = raw["motions"][1], raw["motions"][0]
    with pytest.raises(MarkerSemanticsError, match="canonical order"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    raw["motions"][1] = copy.deepcopy(raw["motions"][0])
    with pytest.raises(MarkerSemanticsError, match="canonical order"):
        _write_and_load(tmp_path, raw)


def test_bool_is_never_accepted_as_integer(tmp_path):
    raw = _raw()
    raw["schema_version"] = True
    with pytest.raises(MarkerSemanticsError, match="integer"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    _row(raw, "fh_block_syn")["construction_marker"]["donor_preferred_frame"] = True
    with pytest.raises(MarkerSemanticsError, match="integer"):
        _write_and_load(tmp_path, raw)


def test_historical_start_is_read_from_v1_direct_record_never_formula(tmp_path):
    legacy = json.loads(_LEGACY_AUTHORITY.read_text(encoding="utf-8"))
    expected = {
        row["motion_id"]: row["historical_adv2c3_start"]["frame"]
        for row in legacy["motions"]
    }
    loaded = load_canonical_motion_markers(
        _AUTHORITY,
        expected_authority_sha256=_AUTHORITY_SHA256,
        repo_root=_ROOT,
    )
    assert {
        row.motion_id: row.historical_adv2c3_start for row in loaded.rows
    } == expected

    raw = _raw()
    fh = _row(raw, "fh_loop")
    fh["historical_adv2c3_comparator"]["frame"] = 7
    with pytest.raises(MarkerSemanticsError, match="v1 direct record"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    fh = _row(raw, "fh_loop")
    fh["historical_adv2c3_comparator"][
        "recorded_basis"
    ] = "explicit immutable v1 frame; no formula evaluated"
    loaded = _write_and_load(tmp_path, raw)
    assert loaded.row("fh_loop").historical_adv2c3_start == 40

    raw = _raw()
    fh = _row(raw, "fh_loop")
    fh["historical_adv2c3_comparator"]["recompute_forbidden"] = False
    with pytest.raises(MarkerSemanticsError, match="recompute_forbidden"):
        _write_and_load(tmp_path, raw)


def test_legacy_preferred_is_independently_nullable(tmp_path):
    raw = _raw()
    _row(raw, "fh_loop")["legacy_preferred_seed"]["frame"] = None
    loaded = _write_and_load(tmp_path, raw)
    fh = loaded.row("fh_loop")
    assert fh.preferred_seed is None
    assert fh.nominal_event == 61
    assert fh.ge50_seed == (47, 55)
    assert fh.ge80_seed == (48, 55)


def test_temporal_lineage_cannot_be_promoted_to_full_identity(tmp_path):
    raw = _raw()
    receipt = _row(raw, "fh_loop")["nominal_event"]["frame_identity_receipt"]
    receipt["classification"] = FULL_FRAME_CLASSIFICATION
    receipt["full_pose_identity_claimed"] = True
    receipt["invariant_joint_count"] = 31
    receipt["excluded_joint_indices"] = []
    with pytest.raises(MarkerSemanticsError, match="disagrees with summary"):
        _write_and_load(tmp_path, raw)


def test_receipt_is_content_addressed_and_must_match_summary(tmp_path):
    raw = _raw()
    receipt = _row(raw, "bh_block")["nominal_event"]["frame_identity_receipt"]
    receipt["sha256"] = "0" * 64
    with pytest.raises(MarkerSemanticsError, match="SHA-256 mismatch"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    receipt = _row(raw, "bh_block")["nominal_event"]["frame_identity_receipt"]
    receipt["event_source_frame"] = 43
    with pytest.raises(MarkerSemanticsError, match="frame identity"):
        _write_and_load(tmp_path, raw)


def test_synthetic_construction_cannot_create_behavior_truth(tmp_path):
    raw = _raw()
    synthetic = _row(raw, "fh_block_syn")
    synthetic["nominal_event"] = copy.deepcopy(_row(raw, "bh_block")["nominal_event"])
    with pytest.raises(MarkerSemanticsError, match="may not create nominal"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    synthetic = _row(raw, "fh_block_syn")
    synthetic["legacy_preferred_seed"]["frame"] = 42
    with pytest.raises(MarkerSemanticsError, match="preferred behavior"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    synthetic = _row(raw, "fh_block_syn")
    synthetic["construction_marker"]["behavior_nominal_created"] = True
    with pytest.raises(MarkerSemanticsError, match="behavior_nominal_created"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    synthetic = _row(raw, "fh_block_syn")
    synthetic["construction_marker"][
        "transfer_to_synthetic_forehand_behavior"
    ] = "allowed"
    with pytest.raises(MarkerSemanticsError, match="must equal 'forbidden'"):
        _write_and_load(tmp_path, raw)


def test_synthetic_ge80_and_construction_spans_are_independent_and_bounded(
    tmp_path,
):
    raw = _raw()
    synthetic = _row(raw, "fh_block_syn")
    synthetic["legacy_ge80_seed"]["span_inclusive"] = [38, 45]
    with pytest.raises(MarkerSemanticsError, match="contained in ge50"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    synthetic = _row(raw, "fh_block_syn")
    synthetic["construction_marker"]["solve_span_inclusive"] = [45, 48]
    with pytest.raises(MarkerSemanticsError, match="exceed"):
        _write_and_load(tmp_path, raw)


@pytest.mark.parametrize("bad_hash", ["0" * 63, "A" * 64, 7, True])
def test_provenance_hash_fields_have_exact_lowercase_sha_type(tmp_path, bad_hash):
    raw = _raw()
    raw["shared_evidence"]["legacy_timing_generator"]["sha256"] = bad_hash
    with pytest.raises(MarkerSemanticsError, match="SHA-256|string"):
        _write_and_load(tmp_path, raw)


def test_duplicate_json_keys_are_rejected_even_when_sha_is_pinned(tmp_path):
    path = tmp_path / "duplicate.json"
    text = _AUTHORITY.read_text(encoding="utf-8").replace(
        '"schema_version": 2,',
        '"schema_version": 2,\n  "schema_version": 2,',
        1,
    )
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(MarkerSemanticsError, match="duplicate JSON key"):
        load_canonical_motion_markers(
            path,
            expected_authority_sha256=digest,
            repo_root=_ROOT,
        )
