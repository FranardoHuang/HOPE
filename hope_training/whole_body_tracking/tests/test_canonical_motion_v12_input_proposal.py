"""Fail-closed input proposal for the real v12 forehand-block take.

This test deliberately stops before compiler admission.  The source-local
marker evidence is exact and reproducible, but the seven-motion recipe remains
a fixture proposal until both the new authority path and a human-adopted
neutral-ready provenance schema are supported by the strict recipe loader.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from canonical_derived_contact_scan import scan_source  # noqa: E402
from canonical_motion_markers import (  # noqa: E402
    EVENT_KIND_DERIVED_CONTACT,
    MARKER_AUTHORITY_PROFILE_BY_PATH,
    MARKER_AUTHORITY_V3APPEND_V12_MOTION_IDS,
    MARKER_AUTHORITY_V3APPEND_V12_PATH,
    MarkerSemanticsError,
    load_canonical_motion_markers,
)
import canonical_motion_markers as marker_module  # noqa: E402
from canonical_motion_recipe import (  # noqa: E402
    MotionRecipeError,
    load_canonical_motion_recipe,
)


_SOURCE_REL = (
    "vendor_assets/motion_finalize_20260724/sources/SHADOW_v12_fh_block.npz"
)
_SOURCE_SHA256 = (
    "433dd363b686b8339def707941eb8048a3af200c0cdff6704b612e5f9184902c"
)
_SCAN_REL = "configs/canonical_motion_v12_forehand_block_contact_scan_20260729.json"
_SCAN_SHA256 = (
    "7180a916bc72d58829b4d232c40c7cff33351ce17d50ab74ff2fa82135f34bb3"
)
_BASE_AUTHORITY = (
    _ROOT / "configs" / "canonical_motion_marker_semantics_v3_20260727.json"
)
_AUTHORITY_REL = (
    "configs/canonical_motion_marker_semantics_v3append_v12_20260729.json"
)
_AUTHORITY = _ROOT / _AUTHORITY_REL
_AUTHORITY_SHA256 = (
    "bc1ea675050c1f5277ecfbb894217ae8fcb7bd53a0e1ca32569ea530d6e3ab98"
)
_BASE_RECIPE = (
    _ROOT / "configs" / "canonical_motion_library_v3append_20260727.json"
)
_RECIPE = (
    _ROOT
    / "configs"
    / "canonical_motion_library_v3append_v12_fixture_20260729.json"
)
_LOADABLE_RECIPE = (
    _ROOT / "configs" / "canonical_motion_library_v3append_v12_20260729.json"
)
_LIBRARY_ORDER = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
    "v12_forehand_block",
)
_DOWNSTREAM_N5_ORDER = (
    "bh_loop_c",
    "v12_forehand_block",
    "bh_block",
    "s0_highpress",
    "fh_loop_high",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v12_source_local_scan_is_exact_reproducible_and_non_authorizing():
    scan_path = _ROOT / _SCAN_REL
    scan = _json(scan_path)
    assert _sha256(scan_path) == _SCAN_SHA256
    assert scan["publication_class"] == "evidence_only_not_artifact_authorization"
    assert set(scan["authorization"].values()) == {False}
    assert set(scan["non_claims"].values()) == {False}
    assert scan["input_source"] == {
        "path": _SOURCE_REL,
        "sha256": _SOURCE_SHA256,
    }
    assert scan["source_materialization_provenance"] == {
        "claim_scope": (
            "Content-addressed SHADOW materialization provenance only; this "
            "binding does not upgrade the source to canonical, certified, or "
            "training-authorized status."
        ),
        "path": (
            "vendor_assets/motion_finalize_20260724/evidence/v12/"
            "SHADOW_v12_fh_block.provenance.json"
        ),
        "sha256": (
            "0399852dd72d7b7de36127880b2684b6a9905bb4a998c107305d48246e889203"
        ),
        "source_pkl": {
            "fps": 30,
            "frames": 100,
            "path": "v12/v12_forehand_block.native_betas.gmr.pkl",
            "sha256": (
                "d9ba1f896bef8eb8cb1c0f615cc661702bd04c17dfebcb3c44435ff174eb6d18"
            ),
        },
        "status": "SHADOW_probe_grade_not_canonical_schema2",
    }
    assert scan["derivation"]["thresholds"]["min_height_m"] == 0.78
    assert scan["derivation"]["thresholds"]["span_speed_fraction"] == 0.6
    assert scan["result"]["anchor_frame"] == 82
    assert scan["result"]["span_inclusive"] == [78, 86]
    assert scan["result"]["body_name"] == "right_wrist_yaw_Link"
    assert scan["result"]["body_position_point"] == "link_origin"
    assert scan["result"]["body_linear_velocity_point"] == "center_of_mass"

    source = _ROOT / _SOURCE_REL
    if not source.is_file():
        pytest.skip("ignored v12 source bundle is not restored")
    assert _sha256(source) == _SOURCE_SHA256
    rerun = scan_source(
        source,
        body_name="right_wrist_yaw_Link",
        min_height_m=0.78,
        span_speed_fraction=0.6,
    )
    assert rerun == scan["result"]


def test_v12_authority_keeps_old_rows_and_adds_one_derived_noncontact_row():
    base = _json(_BASE_AUTHORITY)
    authority = _json(_AUTHORITY)
    assert _sha256(_AUTHORITY) == _AUTHORITY_SHA256
    assert authority["schema_version"] == 3
    assert authority["authority_id"] == base["authority_id"]
    assert authority["review_status"] == base["review_status"]
    assert set(authority["authorization"].values()) == {False}
    assert authority["motions"][:6] == base["motions"]
    assert tuple(row["motion_id"] for row in authority["motions"]) == _LIBRARY_ORDER

    row = authority["motions"][-1]
    assert row["bound_recipe_source"] == {
        "path": _SOURCE_REL,
        "sha256": _SOURCE_SHA256,
    }
    assert row["event_provenance"]["kind"] == EVENT_KIND_DERIVED_CONTACT
    event = row["event_provenance"]["derived_contact"]
    assert event["frame"] == 82
    assert event["contact_truth_observed"] is False
    assert event["behavior_authorized"] is False
    assert event["returnability_certified"] is False
    assert event["frame_identity"]["claimed"] is False
    assert event["v1_lineage"]["claimed"] is False
    assert row["legacy_ge50_seed"] is None
    assert row["legacy_ge80_seed"] is None
    assert row["legacy_preferred_seed"] is None
    assert row["historical_adv2c3_comparator"] is None
    assert row["derived_seed"]["span_inclusive"] == [78, 86]
    assert row["derived_seed"]["argmax_frame"] == 82
    assert row["derived_seed"]["certified_post_retime_window"] is False
    assert row["post_retime_behavior_gate"] == {
        "required": True,
        "status": "PENDING_POST_RETIME_BEHAVIOR_RESCAN",
        "required_scopes": ["upper", "full"],
        "legacy_seed_is_certified_window": False,
        "behavior_promotion_authorized": False,
        "rescan": None,
    }
    for provenance in (event, row["derived_seed"]):
        assert provenance["scan_artifact"] == {
            "path": _SCAN_REL,
            "sha256": _SCAN_SHA256,
            "input_source_sha256": _SOURCE_SHA256,
        }


def test_seven_motion_recipe_is_an_explicit_fail_closed_fixture_proposal():
    base = _json(_BASE_RECIPE)
    recipe = _json(_RECIPE)
    assert recipe["publication_class"] == (
        "fixture_proposal_pending_neutral_ready_and_loader_registration"
    )
    assert recipe["training_authorized"] is False
    assert recipe["hardware_authorized"] is False
    assert recipe["motion_specs"][:6] == base["motion_specs"]
    assert tuple(row["motion_id"] for row in recipe["motion_specs"]) == _LIBRARY_ORDER
    assert tuple(recipe["required_output_matrix"]["motion_ids"]) == _LIBRARY_ORDER
    assert recipe["required_output_matrix"]["scopes"] == ["upper", "full"]
    assert recipe["required_output_matrix"]["candidate_count"] == 14
    assert recipe["marker_authority"] == {
        "path": _AUTHORITY_REL,
        "sha256": _AUTHORITY_SHA256,
    }
    assert recipe["motion_specs"][-1]["source_path"] == _SOURCE_REL
    assert recipe["motion_specs"][-1]["source_sha256"] == _SOURCE_SHA256

    # No fake donor or hash is used while the grounded neutral candidate is
    # still unadopted and does not fit the current exact-donor ready schema.
    ready = recipe["canonical_ready"]
    assert ready["path"].endswith("canonical_ready_v2_g1_neutral_arm.npz")
    assert ready["sha256"] is None
    assert ready["donor_motion_id"] is None
    assert ready["donor_source_frame"] is None
    assert ready["donor_source_sha256"] is None
    assert ", ".join(_DOWNSTREAM_N5_ORDER) in recipe["purpose"]

    # This older fixture remains deliberately non-loadable even after the
    # authority path is registered: it has no adopted neutral-ready identity.
    with pytest.raises(
        MotionRecipeError,
        match="only accepts publication_class='compiler_candidate'",
    ):
        load_canonical_motion_recipe(_RECIPE, repo_root=_ROOT)


def test_v12_authority_path_is_a_strict_registered_profile():
    assert MARKER_AUTHORITY_V3APPEND_V12_PATH == _AUTHORITY_REL
    assert MARKER_AUTHORITY_V3APPEND_V12_MOTION_IDS == (
        "fh_loop_high",
        "v12_forehand_block",
    )
    assert MARKER_AUTHORITY_PROFILE_BY_PATH[_AUTHORITY_REL] == "v3append_v12"
    profile = marker_module.MARKER_AUTHORITY_PROFILES["v3append_v12"]
    assert profile.path == _AUTHORITY_REL
    assert profile.appendable_rows is True
    assert profile.exact_appended_motion_ids == MARKER_AUTHORITY_V3APPEND_V12_MOTION_IDS


def test_registered_v12_profile_rejects_any_drop_or_reorder_before_evidence_io(
    tmp_path,
):
    raw = _json(_AUTHORITY)
    for mutate in (
        lambda value: value["motions"].pop(),
        lambda value: value["motions"].reverse(),
        lambda value: value["motions"].__setitem__(
            slice(5, 7), [value["motions"][6], value["motions"][5]]
        ),
    ):
        candidate = json.loads(json.dumps(raw))
        mutate(candidate)
        path = tmp_path / f"authority_{len(candidate['motions'])}_{id(mutate)}.json"
        path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
        with pytest.raises(
            MarkerSemanticsError,
            match="requires exact ordered motion ids",
        ):
            load_canonical_motion_markers(
                path,
                expected_authority_sha256=_sha256(path),
                repo_root=_ROOT,
                profile="v3append_v12",
            )


def test_old_five_content_cannot_be_replaced_inside_the_v12_authority():
    raw = _json(_AUTHORITY)
    unwrapped = [
        marker_module._unwrap_v3_prefix_row(row, row_index=index)[0]
        for index, row in enumerate(raw["motions"][:5])
    ]
    assert marker_module._require_verbatim_prefix(_ROOT, unwrapped) == (
        "0bf36c204162bf63e044db129bdb869129f36fb7a6d191a7c83effc4ba300929"
    )

    changed = json.loads(json.dumps(raw["motions"][:5]))
    changed[0]["bound_recipe_source"]["sha256"] = "0" * 64
    changed_unwrapped = [
        marker_module._unwrap_v3_prefix_row(row, row_index=index)[0]
        for index, row in enumerate(changed)
    ]
    with pytest.raises(MarkerSemanticsError, match="not a verbatim carry-over"):
        marker_module._require_verbatim_prefix(_ROOT, changed_unwrapped)


def test_loadable_seven_motion_recipe_uses_only_the_legal_legacy_ready():
    recipe = _json(_LOADABLE_RECIPE)
    base = _json(_BASE_RECIPE)
    assert recipe["publication_class"] == "compiler_candidate"
    assert recipe["training_authorized"] is False
    assert recipe["hardware_authorized"] is False
    assert recipe["marker_authority"] == {
        "path": _AUTHORITY_REL,
        "sha256": _AUTHORITY_SHA256,
    }
    assert recipe["canonical_ready"] == base["canonical_ready"]
    assert recipe["canonical_ready"] == {
        "path": (
            "vendor_assets/motion_finalize_20260724/ready/canonical_ready_v1.npz"
        ),
        "sha256": (
            "cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0"
        ),
        "donor_motion_id": "bh_loop_c",
        "donor_source_frame": 0,
        "donor_source_sha256": (
            "d5338168e692c8a2c19fbfac8aeb56653fa79a1f45cebc6803a460835fbc1fba"
        ),
        "endpoint_velocity_policy": "all_joint_root_body_velocities_exact_zero",
    }
    assert tuple(row["motion_id"] for row in recipe["motion_specs"]) == _LIBRARY_ORDER
    assert tuple(recipe["required_output_matrix"]["motion_ids"]) == _LIBRARY_ORDER
    assert recipe["required_output_matrix"]["candidate_count"] == 14
    assert recipe["motion_specs"][:6] == base["motion_specs"]
    assert recipe["motion_specs"][-1]["source_path"] == _SOURCE_REL
    assert recipe["motion_specs"][-1]["source_sha256"] == _SOURCE_SHA256
    assert "does not claim" in recipe["purpose"]
    assert "human-adopted" in recipe["purpose"]


def test_loadable_seven_motion_recipe_closes_when_full_evidence_is_restored():
    summary = (
        _ROOT
        / "vendor_assets"
        / "motion_finalize_20260724"
        / "evidence"
        / "frame_maps"
        / "frame_identity_receipts_summary_v1.json"
    )
    if not summary.is_file():
        pytest.skip("full ignored v3 authority evidence bundle is not restored")
    recipe = load_canonical_motion_recipe(_LOADABLE_RECIPE, repo_root=_ROOT)
    assert tuple(row.motion_id for row in recipe.sources) == _LIBRARY_ORDER
    assert recipe.marker_semantics.profile == "v3append_v12"
    assert recipe.source("v12_forehand_block").sha256 == _SOURCE_SHA256
    assert recipe.ready.sha256 == (
        "cb0a05ca9f7220686acfde1010c28ed04558fb2aa47ef2cfb2284d576ecd15b0"
    )


def test_v12_probe_provenance_remains_shadow_when_local_receipt_is_present():
    receipt = (
        _ROOT
        / "vendor_assets"
        / "motion_finalize_20260724"
        / "evidence"
        / "v12"
        / "SHADOW_v12_fh_block.provenance.json"
    )
    if not receipt.is_file():
        pytest.skip("ignored v12 source-provenance receipt is not restored")
    assert _sha256(receipt) == (
        "0399852dd72d7b7de36127880b2684b6a9905bb4a998c107305d48246e889203"
    )
    raw = _json(receipt)
    assert raw["status"] == "SHADOW_probe_grade_not_canonical_schema2"
    assert set(raw["not_authorized_for"]) == {
        "registry",
        "training",
        "judge",
        "hardware",
        "certificates",
    }
    assert raw["source_pkl"] == {
        "path": "v12/v12_forehand_block.native_betas.gmr.pkl",
        "sha256": (
            "d9ba1f896bef8eb8cb1c0f615cc661702bd04c17dfebcb3c44435ff174eb6d18"
        ),
        "frames": 100,
        "fps": 30,
    }
    assert raw["output"] == {
        "frames": 166,
        "fps": 50,
        "frame_formula": "round(((input_frames-1)/30)*50)+1",
    }


def test_strict_v3_authority_loader_closes_when_full_ignored_bundle_is_restored():
    summary = (
        _ROOT
        / "vendor_assets"
        / "motion_finalize_20260724"
        / "evidence"
        / "frame_maps"
        / "frame_identity_receipts_summary_v1.json"
    )
    if not summary.is_file():
        pytest.skip("full ignored v3 authority evidence bundle is not restored")
    authority = load_canonical_motion_markers(
        _AUTHORITY,
        expected_authority_sha256=_AUTHORITY_SHA256,
        repo_root=_ROOT,
        profile="v3append_v12",
    )
    assert tuple(row.motion_id for row in authority.rows) == _LIBRARY_ORDER
    v12 = authority.row("v12_forehand_block")
    assert v12.contact_anchor() == (82, "derived_source_timing_contact")
    assert v12.search_window() == ((78, 86), "derived_seed")
    assert v12.bound_recipe_source_sha256 == _SOURCE_SHA256
