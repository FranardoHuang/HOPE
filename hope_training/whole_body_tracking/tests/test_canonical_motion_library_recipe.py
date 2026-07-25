"""Contract tests for the canonical five-motion compiler recipe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[3]
_RECIPE = _REPO / "configs" / "canonical_motion_library_v2_20260724.json"


def _load():
    return json.loads(_RECIPE.read_text(encoding="utf-8"))


def _load_marker_authority(recipe):
    authority = recipe["marker_authority"]
    path = _REPO / authority["path"]
    assert _sha256(path) == authority["sha256"]
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_recipe_is_candidate_only_and_defines_exact_five_by_two_matrix():
    recipe = _load()
    assert recipe["publication_class"] == "compiler_candidate"
    assert recipe["training_authorized"] is False
    assert recipe["hardware_authorized"] is False
    assert recipe["entry_exit_search"]["historical_adv2c3_role"] == (
        "comparator_only_not_default"
    )
    assert recipe["entry_exit_search"]["candidate_eligibility"] == (
        "retain_legacy_ge80_seed_plus_symmetric_halo"
    )
    assert recipe["time_law"]["window_acceleration_allowed_through_end"] is True
    assert recipe["time_law"]["joint_velocity_limit_fraction"] == 1.0
    assert recipe["time_law"]["legacy_seed_marker_policy"] == (
        "search_and_retime_marker_only_never_output_behavior_window"
    )
    assert recipe["time_law"]["kinematic_window_policy"] == (
        "nonnegative_scalar_acceleration_through_exact_window_end"
    )
    assert recipe["time_law"]["acceleration_policy"] == (
        "grounded_torque_contact_screen_required_before_promotion_beyond_"
        "compiler_candidate"
    )

    expected = {
        "fh_loop",
        "bh_loop_c",
        "fh_block_syn",
        "bh_block",
        "s0_highpress",
    }
    specs = recipe["motion_specs"]
    assert len(specs) == 5
    assert {row["motion_id"] for row in specs} == expected
    matrix = recipe["required_output_matrix"]
    assert set(matrix["motion_ids"]) == expected
    assert matrix["scopes"] == ["upper", "full"]
    assert matrix["candidate_count"] == 10


def test_marker_authority_distinguishes_legacy_seeds_from_contact_truth():
    recipe = _load()
    authority = _load_marker_authority(recipe)
    assert authority["authority_id"] == (
        "canonical_motion_marker_semantics_v2_20260724"
    )
    rows = {row["motion_id"]: row for row in authority["motions"]}
    assert set(rows) == {
        "fh_loop",
        "bh_loop_c",
        "fh_block_syn",
        "bh_block",
        "s0_highpress",
    }
    for motion_id, row in rows.items():
        ge50 = row["legacy_ge50_seed"]["span_inclusive"]
        ge80 = row["legacy_ge80_seed"]["span_inclusive"]
        assert all(isinstance(value, int) for value in (*ge50, *ge80))
        # The stricter >=80% seed nests inside the >=50% seed.
        assert ge50[0] <= ge80[0] <= ge80[1] <= ge50[1]
        assert row["legacy_ge80_seed"]["certified_post_retime_window"] is False
        gate = row["post_retime_behavior_gate"]
        assert gate["required"] is True
        assert gate["legacy_seed_is_certified_window"] is False
        assert gate["behavior_promotion_authorized"] is False
        event = row["nominal_event"]
        if motion_id == "fh_block_syn":
            assert event is None
        else:
            # Event-to-window aliasing is forbidden: the reviewed air-swing
            # event may legally sit outside the legacy ge80 behavior seed
            # (it does for fh_loop, bh_loop_c, and s0_highpress).
            assert isinstance(event["frame"], int)
            assert event["contact_truth_observed"] is False
            assert event["behavior_authorized"] is False
    assert recipe["time_law"]["post_retime_behavior_opportunity_minimum_s"] > 0.0


def test_synthetic_block_is_multijoint_manifold_not_a_pi_joint_overlay():
    recipe = _load()
    specs = {row["motion_id"]: row for row in recipe["motion_specs"]}
    synthetic = specs["fh_block_syn"]
    backhand = specs["bh_block"]
    assert synthetic["source_sha256"] == backhand["source_sha256"]
    face = synthetic["face_manifold"]
    assert face["mode"] == "signed_raw_plus_y_flip"
    assert face["active_joints"] == "right_arm_7"
    assert face["site_position"] == "preserve_per_source_frame"
    assert face["orientation"] == "normal_hard_inplane_free"
    assert face["single_axis_pi_overlay_forbidden"] is True

    authority = _load_marker_authority(recipe)
    rows = {row["motion_id"]: row for row in authority["motions"]}
    construction = rows["fh_block_syn"]["construction_marker"]
    solve_span = construction["solve_span_inclusive"]
    ge80 = rows["fh_block_syn"]["legacy_ge80_seed"]["span_inclusive"]
    # The face solve covers the search halo around the retained ge80 core.
    assert solve_span[0] < ge80[0] <= ge80[1] < solve_span[1]
    assert construction["donor_motion_id"] == "bh_block"
    donor_event = rows["bh_block"]["nominal_event"]["frame"]
    assert construction["annotation_frame"] == donor_event
    assert rows["fh_block_syn"]["nominal_event"] is None
    assert rows["fh_block_syn"]["legacy_preferred_seed"]["frame"] is None


def test_tracked_model_contract_hashes_are_current():
    recipe = _load()
    model = recipe["model_contract"]
    for path_key, sha_key in (
        ("mjcf_path", "mjcf_sha256"),
        ("urdf_path", "urdf_sha256"),
        ("body_order_path", "body_order_sha256"),
    ):
        path = _REPO / model[path_key]
        assert path.is_file()
        assert _sha256(path) == model[sha_key]


def test_ignored_source_hashes_and_ready_donor_close_when_assets_are_present():
    recipe = _load()
    ready_path = _REPO / recipe["canonical_ready"]["path"]
    source_paths = [_REPO / row["source_path"] for row in recipe["motion_specs"]]
    if not ready_path.is_file() or not all(path.is_file() for path in source_paths):
        pytest.skip("ignored canonical motion source bundle is not restored")

    authority = _load_marker_authority(recipe)
    rows = {row["motion_id"]: row for row in authority["motions"]}
    assert _sha256(ready_path) == recipe["canonical_ready"]["sha256"]
    for row, path in zip(recipe["motion_specs"], source_paths):
        assert _sha256(path) == row["source_sha256"]
        bound = rows[row["motion_id"]]["bound_recipe_source"]
        assert bound["path"] == row["source_path"]
        assert bound["sha256"] == row["source_sha256"]
        with np.load(path, allow_pickle=False) as source:
            frame_count = int(source["joint_pos"].shape[0])
        assert rows[row["motion_id"]]["legacy_ge80_seed"]["span_inclusive"][1] < (
            frame_count
        )

    donor = next(
        row for row in recipe["motion_specs"] if row["motion_id"] == "bh_loop_c"
    )
    with np.load(_REPO / donor["source_path"], allow_pickle=False) as source:
        with np.load(ready_path, allow_pickle=False) as ready:
            np.testing.assert_array_equal(ready["joint_pos"], source["joint_pos"][0])
            np.testing.assert_array_equal(
                ready["root_pos_w"], source["body_pos_w"][0, 0]
            )
            np.testing.assert_array_equal(
                ready["root_quat_w"], source["body_quat_w"][0, 0]
            )
