"""Tests for marker authority v3 and the v1-lineage fabrication guard.

Three things are being proved here, in plain language:

1. the four legacy rows plus the synthetic row behave exactly as before -- both
   under v2 and when carried into v3;
2. a recording made after v1 was frozen can be admitted, and its provenance
   says what it actually is (a tool scanned these exact bytes) rather than
   borrowing someone else's lineage; and
3. the fabrication guard is LOAD-BEARING: the same payload that the guard
   rejects is accepted the moment the guard is neutered.  If someone deletes
   ``_require_v1_content_agreement``, ``test_fabrication_guard_is_load_bearing``
   fails.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import canonical_motion_markers as markers  # noqa: E402
from canonical_motion_markers import (  # noqa: E402
    CANONICAL_MOTION_IDS,
    EVENT_KIND_DERIVED_CONTACT,
    EVENT_KIND_DONOR_CONSTRUCTION,
    EVENT_KIND_LEGACY_V1,
    LEGACY_AUTHORITY_SHA256,
    MARKER_AUTHORITY_PATH,
    MARKER_AUTHORITY_SHA256,
    MARKER_AUTHORITY_V3_PATH,
    MarkerSemanticsError,
    load_canonical_motion_markers,
    sha256_file,
)

_V2 = _ROOT / MARKER_AUTHORITY_PATH
_V3 = _ROOT / MARKER_AUTHORITY_V3_PATH
_NEW_MOTION_ID = "fh_loop_high"
_NEW_SOURCE = (
    "vendor_assets/motion_finalize_20260724/sources/SHADOW_fh_loop_high_yaw152.npz"
)
_NEW_SOURCE_SHA256 = (
    "7d045fcb036ffa668dede4607cfcc82e789a0db7ab86fd8df9dd52cfd5ac4153"
)
_OLD_FH_LOOP_SHA256 = (
    "faa8df8c552e4bd99134cefe5457f86e646499ff12737db160fa43eec763dcc1"
)
#: Scratch directory INSIDE the repo, because the loader refuses any evidence
#: path that escapes repo_root.  Removed in the fixture teardown.
_SCRATCH = _ROOT / "vendor_assets" / "motion_finalize_20260724" / "evidence" / (
    "_pytest_fabrication_scratch"
)

# The v2-era fields, frozen literally.  A behaviour change on an existing row
# has to edit this table, which is the point.
_EXPECTED_V2_ROWS = {
    "fh_loop": {
        "nominal_event": 61,
        "ge50_seed": (47, 55),
        "ge80_seed": (48, 55),
        "preferred_seed": 49,
        "construction_marker": None,
        "historical_adv2c3_start": 40,
        "bound_recipe_source_sha256": _OLD_FH_LOOP_SHA256,
        "kind": EVENT_KIND_LEGACY_V1,
        "anchor": (61, EVENT_KIND_LEGACY_V1),
        "window": ((48, 55), "legacy_ge80_seed"),
    },
    "bh_loop_c": {
        "nominal_event": 84,
        "ge50_seed": (88, 95),
        "ge80_seed": (88, 94),
        "preferred_seed": 89,
        "construction_marker": None,
        "historical_adv2c3_start": 56,
        "bound_recipe_source_sha256": (
            "d5338168e692c8a2c19fbfac8aeb56653fa79a1f45cebc6803a460835fbc1fba"
        ),
        "kind": EVENT_KIND_LEGACY_V1,
        "anchor": (84, EVENT_KIND_LEGACY_V1),
        "window": ((88, 94), "legacy_ge80_seed"),
    },
    "fh_block_syn": {
        "nominal_event": None,
        "ge50_seed": (39, 46),
        "ge80_seed": (40, 45),
        "preferred_seed": None,
        "construction_marker": (44, 42, (34, 48)),
        "historical_adv2c3_start": 29,
        "bound_recipe_source_sha256": (
            "55870b981584a458bfd479171046445845cb74171618b71338fd9dc9f66a5fe0"
        ),
        "kind": EVENT_KIND_DONOR_CONSTRUCTION,
        "anchor": (44, EVENT_KIND_DONOR_CONSTRUCTION),
        "window": ((40, 45), "legacy_ge80_seed"),
    },
    "bh_block": {
        "nominal_event": 44,
        "ge50_seed": (39, 46),
        "ge80_seed": (40, 45),
        "preferred_seed": 42,
        "construction_marker": None,
        "historical_adv2c3_start": 29,
        "bound_recipe_source_sha256": (
            "55870b981584a458bfd479171046445845cb74171618b71338fd9dc9f66a5fe0"
        ),
        "kind": EVENT_KIND_LEGACY_V1,
        "anchor": (44, EVENT_KIND_LEGACY_V1),
        "window": ((40, 45), "legacy_ge80_seed"),
    },
    "s0_highpress": {
        "nominal_event": 58,
        "ge50_seed": (52, 57),
        "ge80_seed": (53, 57),
        "preferred_seed": 54,
        "construction_marker": None,
        "historical_adv2c3_start": 38,
        "bound_recipe_source_sha256": (
            "2cd32da1864fa686aff544d29a84e988b91911503ae7f7680601f93345378c01"
        ),
        "kind": EVENT_KIND_LEGACY_V1,
        "anchor": (58, EVENT_KIND_LEGACY_V1),
        "window": ((53, 57), "legacy_ge80_seed"),
    },
}


def _load(path: Path, profile: str):
    return load_canonical_motion_markers(
        path,
        expected_authority_sha256=sha256_file(path),
        repo_root=_ROOT,
        profile=profile,
    )


def _write_and_load(tmp_path: Path, raw: dict, profile: str):
    path = tmp_path / "authority.json"
    path.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return _load(path, profile)


def _v2_raw() -> dict:
    return json.loads(_V2.read_text(encoding="utf-8"))


def _v3_raw() -> dict:
    return json.loads(_V3.read_text(encoding="utf-8"))


def _row(raw: dict, motion_id: str) -> dict:
    return next(row for row in raw["motions"] if row["motion_id"] == motion_id)


@contextmanager
def _scratch():
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        yield _SCRATCH
    finally:
        shutil.rmtree(_SCRATCH, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sha256_file(path)


# ---------------------------------------------------------------------------
# 1. the existing rows do not move
# ---------------------------------------------------------------------------


def test_v2_authority_bytes_are_untouched():
    assert sha256_file(_V2) == MARKER_AUTHORITY_SHA256
    assert (
        sha256_file(_ROOT / "configs/canonical_motion_marker_semantics_v1_20260724.json")
        == LEGACY_AUTHORITY_SHA256
    )


def test_v2_rows_still_produce_exactly_the_frozen_semantics():
    loaded = _load(_V2, "v2")
    assert loaded.profile == "v2"
    assert loaded.verbatim_prefix_authority_sha256 is None
    assert tuple(row.motion_id for row in loaded.rows) == CANONICAL_MOTION_IDS
    for motion_id, expected in _EXPECTED_V2_ROWS.items():
        row = loaded.row(motion_id)
        assert row.nominal_event == expected["nominal_event"]
        assert row.ge50_seed == expected["ge50_seed"]
        assert row.ge80_seed == expected["ge80_seed"]
        assert row.preferred_seed == expected["preferred_seed"]
        assert row.historical_adv2c3_start == expected["historical_adv2c3_start"]
        assert row.bound_recipe_source_sha256 == expected["bound_recipe_source_sha256"]
        if expected["construction_marker"] is None:
            assert row.construction_marker is None
        else:
            annotation, donor, span = expected["construction_marker"]
            assert row.construction_marker.annotation_frame == annotation
            assert row.construction_marker.donor_preferred_frame == donor
            assert row.construction_marker.solve_span == span
        # The v3 machinery is inert on a v2 document.
        assert row.event_provenance_kind == expected["kind"]
        assert row.derived_contact is None
        assert row.derived_seed is None
        assert row.post_retime_rescan is None
        assert row.contact_anchor() == expected["anchor"]
        assert row.search_window() == expected["window"]


def test_v3_prefix_rows_are_field_for_field_identical_to_v2():
    """The strongest 'nothing moved' statement available: same objects."""

    v2 = _load(_V2, "v2")
    v3 = _load(_V3, "v3")
    assert v3.profile == "v3"
    # v2's SHA-256 is a load-bearing input to v3, recomputed from bytes.
    assert v3.verbatim_prefix_authority_sha256 == MARKER_AUTHORITY_SHA256
    assert [row.motion_id for row in v3.rows[:5]] == list(CANONICAL_MOTION_IDS)
    names = [field.name for field in fields(v2.rows[0])]
    for old, new in zip(v2.rows, v3.rows):
        for name in names:
            assert getattr(old, name) == getattr(new, name), name
        assert old.contact_anchor() == new.contact_anchor()
        assert old.search_window() == new.search_window()
        assert old.authority_frames() == new.authority_frames()


def test_v3_refuses_a_restated_prefix_row(tmp_path):
    raw = _v3_raw()
    _row(raw, "fh_loop")["legacy_ge80_seed"]["span_inclusive"] = [49, 55]
    with pytest.raises(MarkerSemanticsError, match="verbatim carry-over"):
        _write_and_load(tmp_path, raw, "v3")


def test_v3_refuses_a_prefix_row_that_grows_a_derived_seed(tmp_path):
    raw = _v3_raw()
    _row(raw, "fh_loop")["derived_seed"] = copy.deepcopy(
        _row(raw, _NEW_MOTION_ID)["derived_seed"]
    )
    with pytest.raises(MarkerSemanticsError, match="derived_seed must be null"):
        _write_and_load(tmp_path, raw, "v3")


def test_v3_refuses_prefix_reordering_or_replacement(tmp_path):
    raw = _v3_raw()
    raw["motions"][0], raw["motions"][1] = raw["motions"][1], raw["motions"][0]
    with pytest.raises(MarkerSemanticsError, match="canonical order changed"):
        _write_and_load(tmp_path, raw, "v3")


# ---------------------------------------------------------------------------
# 2. the caller, not the document, picks the profile
# ---------------------------------------------------------------------------


def test_a_document_cannot_choose_its_own_validation_regime():
    # v3 bytes offered to a caller expecting v2, and vice versa.
    with pytest.raises(MarkerSemanticsError, match="schema_version must equal 2"):
        _load(_V3, "v2")
    with pytest.raises(MarkerSemanticsError, match="schema_version must equal 3"):
        _load(_V2, "v3")


def test_unknown_profile_fails_closed():
    with pytest.raises(MarkerSemanticsError, match="unknown marker authority profile"):
        _load(_V2, "v4")


def test_default_profile_is_v2_so_existing_callers_are_unchanged():
    loaded = load_canonical_motion_markers(
        _V2, expected_authority_sha256=sha256_file(_V2), repo_root=_ROOT
    )
    assert loaded.profile == "v2"


# ---------------------------------------------------------------------------
# 3. the new recording is admitted, honestly
# ---------------------------------------------------------------------------


def test_v3_admits_the_2026_07_27_forehand_recording():
    row = _load(_V3, "v3").row(_NEW_MOTION_ID)
    assert row.event_provenance_kind == EVENT_KIND_DERIVED_CONTACT
    assert row.bound_recipe_source_path == _NEW_SOURCE
    assert row.bound_recipe_source_sha256 == _NEW_SOURCE_SHA256

    # It states what it is.
    contact = row.derived_contact
    assert contact is not None
    assert contact.frame == 54
    assert contact.returnability_certified is False
    assert contact.semantic_kind == "derived_kinematic_carrier_speed_peak_frame"

    # It states what it is NOT.  No v1 lineage, no frame-identity receipt, no
    # legacy scan, no adv2c3 comparator -- none of those exist for this take.
    assert row.nominal_event is None
    assert row.frame_identity is None
    assert row.source_scan_sha256 is None
    assert row.historical_adv2c3_start is None
    assert row.ge50_seed is None and row.ge80_seed is None

    # Its evidence is closed on its OWN bytes rather than on an ancestor's.
    provenance = contact.provenance
    assert provenance.scan_input_source_sha256 == _NEW_SOURCE_SHA256
    assert provenance.retiming_stage == "source_timing"
    assert provenance.tool == (
        "hope_training/whole_body_tracking/scripts/canonical_derived_contact_scan.py"
    )
    assert provenance.tool_sha256 == sha256_file(_ROOT / provenance.tool)
    assert (
        sha256_file(_ROOT / provenance.scan_artifact_path)
        == provenance.scan_artifact_sha256
    )

    assert row.contact_anchor() == (54, "derived_source_timing_contact")
    assert row.search_window() == ((47, 57), "derived_seed")
    assert row.derived_seed.argmax_frame == 54
    assert row.derived_seed.certified_post_retime_window is False


def test_derived_row_must_be_scanned_on_the_bytes_it_binds(tmp_path):
    raw = _v3_raw()
    contact = _row(raw, _NEW_MOTION_ID)["event_provenance"]["derived_contact"]
    contact["scan_artifact"]["input_source_sha256"] = _OLD_FH_LOOP_SHA256
    with pytest.raises(
        MarkerSemanticsError, match="must equal this row's bound_recipe_source"
    ):
        _write_and_load(tmp_path, raw, "v3")


def test_derived_row_cannot_transcribe_a_frame_the_artifact_does_not_report(tmp_path):
    raw = _v3_raw()
    _row(raw, _NEW_MOTION_ID)["event_provenance"]["derived_contact"]["frame"] = 61
    with pytest.raises(
        MarkerSemanticsError, match="not the frame the bound scan artifact reports"
    ):
        _write_and_load(tmp_path, raw, "v3")


def test_derived_row_cannot_relabel_the_rule_it_ran(tmp_path):
    raw = _v3_raw()
    derivation = _row(raw, _NEW_MOTION_ID)["event_provenance"]["derived_contact"][
        "derivation"
    ]
    derivation["rule_id"] = "longest_contiguous_run_at_or_above_score_threshold"
    with pytest.raises(MarkerSemanticsError, match="rule_id disagrees"):
        _write_and_load(tmp_path, raw, "v3")


def test_derived_row_cannot_claim_a_tool_build_that_is_not_on_disk(tmp_path):
    raw = _v3_raw()
    derivation = _row(raw, _NEW_MOTION_ID)["event_provenance"]["derived_contact"][
        "derivation"
    ]
    derivation["tool_sha256"] = "0" * 64
    with pytest.raises(MarkerSemanticsError, match="does not match the tool on disk"):
        _write_and_load(tmp_path, raw, "v3")


def test_derived_row_may_not_claim_contact_truth_or_returnability(tmp_path):
    for key in ("contact_truth_observed", "behavior_authorized", "returnability_certified"):
        raw = _v3_raw()
        _row(raw, _NEW_MOTION_ID)["event_provenance"]["derived_contact"][key] = True
        with pytest.raises(MarkerSemanticsError, match="must be exactly False"):
            _write_and_load(tmp_path, raw, "v3")


def test_appended_row_may_not_pose_as_a_v1_air_swing_event(tmp_path):
    raw = _v3_raw()
    _row(raw, _NEW_MOTION_ID)["event_provenance"]["kind"] = EVENT_KIND_LEGACY_V1
    with pytest.raises(MarkerSemanticsError):
        _write_and_load(tmp_path, raw, "v3")


def test_appended_row_may_not_reuse_a_canonical_motion_id(tmp_path):
    raw = _v3_raw()
    _row(raw, _NEW_MOTION_ID)["motion_id"] = "fh_loop"
    with pytest.raises(MarkerSemanticsError, match="reuses a canonical motion id"):
        _write_and_load(tmp_path, raw, "v3")


def test_rescan_slot_exists_but_is_not_yet_a_promotion_path(tmp_path):
    raw = _v3_raw()
    _row(raw, "fh_loop")["post_retime_behavior_gate"]["rescan"] = {
        "status": "PASS",
        "scope": "upper",
    }
    with pytest.raises(MarkerSemanticsError, match="rescan must be null"):
        _write_and_load(tmp_path, raw, "v3")


# ---------------------------------------------------------------------------
# 4. the fabrication guard
# ---------------------------------------------------------------------------


def test_guard_rejects_a_motion_id_v1_never_recorded():
    with pytest.raises(MarkerSemanticsError, match="has no record for"):
        markers._require_v1_content_agreement(
            {},
            motion_id=_NEW_MOTION_ID,
            bound_sha256=_NEW_SOURCE_SHA256,
            nominal_frame=54,
            label="motions[fh_loop_high]",
        )


def test_guard_rejects_a_moved_nominal_frame():
    legacy = {"fh_loop": {
        "bound_recipe_source": {"sha256": _OLD_FH_LOOP_SHA256},
        "nominal_event": {"frame": 61},
    }}
    with pytest.raises(MarkerSemanticsError, match="disagrees with the immutable v1"):
        markers._require_v1_content_agreement(
            legacy,
            motion_id="fh_loop",
            bound_sha256=_OLD_FH_LOOP_SHA256,
            nominal_frame=52,
            label="motions[fh_loop]",
        )


def _repointed_authority(scratch: Path) -> dict:
    """Build the A4/B forgery: fh_loop re-aimed at the 2026-07-27 npz.

    Every content-addressed artefact is REGENERATED so that it agrees with the
    forgery.  That is the whole difficulty: the receipt, the summary and the
    authority are internally consistent, the v1 SHA-256 string is still present
    in both places the loader compares it, and only opening v1's own content
    reveals that fh_loop's source is not the file v1 recorded.
    """

    raw = _v2_raw()
    fh_loop = _row(raw, "fh_loop")
    fh_loop["bound_recipe_source"] = {
        "path": _NEW_SOURCE,
        "sha256": _NEW_SOURCE_SHA256,
    }
    # legacy_scan_evidence is an unverifiable remote claim, so a forger simply
    # retypes it.
    fh_loop["legacy_scan_evidence"]["input_source_sha256"] = _NEW_SOURCE_SHA256

    receipt_source = (
        _ROOT
        / "vendor_assets/motion_finalize_20260724/evidence/frame_maps"
        / "fh_loop_frame_identity_v1.json"
    )
    receipt = json.loads(receipt_source.read_text(encoding="utf-8"))
    receipt["input_bindings"]["bound_npz"]["sha256"] = _NEW_SOURCE_SHA256
    receipt["event_authority_contract"]["bound_recipe_source_sha256"] = (
        _NEW_SOURCE_SHA256
    )
    receipt["event_authority_contract"]["bound_recipe_source_path"] = _NEW_SOURCE
    receipt_path = scratch / "fh_loop_frame_identity_forged.json"
    receipt_sha = _write_json(receipt_path, receipt)
    receipt_rel = receipt_path.relative_to(_ROOT).as_posix()

    summary_ref = raw["shared_evidence"]["frame_identity_receipts_summary"]
    summary = json.loads((_ROOT / summary_ref["path"]).read_text(encoding="utf-8"))
    for entry in summary["receipts"]:
        if entry["motion_id"] == "fh_loop":
            entry["receipt_path"] = receipt_rel
            entry["receipt_sha256"] = receipt_sha
    summary_path = scratch / "frame_identity_receipts_summary_forged.json"
    summary_sha = _write_json(summary_path, summary)
    raw["shared_evidence"]["frame_identity_receipts_summary"] = {
        "path": summary_path.relative_to(_ROOT).as_posix(),
        "sha256": summary_sha,
    }

    binding = fh_loop["nominal_event"]["frame_identity_receipt"]
    binding["path"] = receipt_rel
    binding["sha256"] = receipt_sha
    binding["bound_source_sha256"] = _NEW_SOURCE_SHA256
    return raw


def test_fabrication_guard_blocks_repointing_a_v1_receipt_at_new_bytes(tmp_path):
    with _scratch() as scratch:
        raw = _repointed_authority(scratch)
        with pytest.raises(
            MarkerSemanticsError,
            match="disagree with the immutable v1 record",
        ):
            _write_and_load(tmp_path, raw, "v2")


def test_fabrication_guard_is_load_bearing(tmp_path, monkeypatch):
    """THIS TEST FAILS IF THE GUARD IS REMOVED.

    The forgery in ``_repointed_authority`` passes every other check in the
    loader -- receipt SHA, summary SHA, event/bound npz bindings, the two
    ``authority_v1_sha256`` string comparisons, the classification/coverage
    contract, all of it.  Neutering ``_require_v1_content_agreement`` therefore
    makes it LOAD, which is exactly how a 98-frame take could inherit a
    116-frame take's frame-61 marker.  If the guard ever stops being called,
    the ``pytest.raises`` above stops raising and this assertion documents why.
    """

    with _scratch() as scratch:
        raw = _repointed_authority(scratch)

        # Guard on: rejected.
        with pytest.raises(MarkerSemanticsError):
            _write_and_load(tmp_path / "on", raw, "v2")

        # Guard off: the very same bytes sail through.  Nothing else notices
        # that fh_loop is now bound to a recording v1 never saw.
        monkeypatch.setattr(
            markers,
            "_require_v1_content_agreement",
            lambda *args, **kwargs: None,
        )
        forged = _write_and_load(tmp_path / "off", raw, "v2")
        assert forged.row("fh_loop").bound_recipe_source_sha256 == _NEW_SOURCE_SHA256
        assert forged.row("fh_loop").nominal_event == 61


@pytest.fixture(autouse=True)
def _make_tmp_subdirs(tmp_path):
    (tmp_path / "on").mkdir(exist_ok=True)
    (tmp_path / "off").mkdir(exist_ok=True)
    yield
