"""Tests for canonical motion-library recipe schema 2."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from canonical_motion_markers import (  # noqa: E402
    MARKER_AUTHORITY_PATH,
)
from canonical_motion_recipe import (  # noqa: E402
    MotionRecipeError,
    _validate_output_matrix,
    _validate_recipe_contract,
    load_canonical_motion_recipe,
    sha256_file,
)


_RECIPE = _ROOT / "configs" / "canonical_motion_library_v2_20260724.json"
_MARKER_AUTHORITY_SHA256 = (
    "0bf36c204162bf63e044db129bdb869129f36fb7a6d191a7c83effc4ba300929"
)


def _raw() -> dict:
    return json.loads(_RECIPE.read_text(encoding="utf-8"))


@contextmanager
def _temporary_recipe(tmp_path: Path, raw: dict):
    name = f"canonical_motion_library_test_{tmp_path.name}.json"
    path = _ROOT / "configs" / name
    path.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _write_and_load(tmp_path: Path, raw: dict):
    with _temporary_recipe(tmp_path, raw) as path:
        return load_canonical_motion_recipe(path, repo_root=_ROOT)


def _spec(raw: dict, motion_id: str) -> dict:
    return next(row for row in raw["motion_specs"] if row["motion_id"] == motion_id)


def test_loads_schema2_and_closes_ready_source_and_marker_authority():
    recipe = load_canonical_motion_recipe(_RECIPE, repo_root=_ROOT)
    assert [row.motion_id for row in recipe.sources] == [
        "fh_loop",
        "bh_loop_c",
        "fh_block_syn",
        "bh_block",
        "s0_highpress",
    ]
    assert recipe.ready.source_segment == "bh_loop_c"
    assert recipe.ready.source_frame == 0
    assert recipe.source("fh_block_syn").sha256 == recipe.source("bh_block").sha256

    assert (
        recipe.marker_authority_path.relative_to(_ROOT).as_posix()
        == MARKER_AUTHORITY_PATH
    )
    assert recipe.marker_authority_sha256 == _MARKER_AUTHORITY_SHA256
    assert sha256_file(recipe.marker_authority_path) == (_MARKER_AUTHORITY_SHA256)
    assert recipe.marker_semantics.row("fh_loop").ge80_seed == (48, 55)
    synthetic = recipe.marker_semantics.row("fh_block_syn")
    assert synthetic.nominal_event is None
    assert synthetic.preferred_seed is None
    assert synthetic.construction_marker is not None

    source = recipe.source("fh_loop")
    assert not hasattr(source, "protected_source_span")
    assert not hasattr(source, "source_anchor_frame")
    assert not hasattr(source, "entry_exit_override")
    assert "protected_source_span" not in _spec(_raw(), "fh_loop")
    assert "source_anchor_frame" not in _spec(_raw(), "fh_loop")
    assert "entry_exit_override" not in _spec(_raw(), "fh_block_syn")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda row: row.__setitem__("training_authorized", True),
            "training_authorized",
        ),
        (
            lambda row: row["time_law"].__setitem__(
                "legacy_seed_marker_policy", "legacy_is_final_window"
            ),
            "never certify",
        ),
        (
            lambda row: row["time_law"].__setitem__(
                "joint_velocity_limit_fraction", 0.85
            ),
            "exact URDF",
        ),
        (
            lambda row: row["entry_exit_search"].__setitem__(
                "historical_adv2c3_role", "default"
            ),
            "comparator",
        ),
        (
            lambda row: row["entry_exit_search"].__setitem__(
                "candidate_eligibility", "preferred_only"
            ),
            "ge80",
        ),
        (
            lambda row: row["entry_exit_search"].__setitem__(
                "ranking_preference",
                [
                    "ordinary_nominal_event_if_available",
                    "opportunity_start",
                    "opportunity_end",
                ],
            ),
            "start/event/end",
        ),
        (
            lambda row: row["entry_exit_search"].__setitem__(
                "legacy_ge80_halo_source_frames", 0
            ),
            "integer",
        ),
        (
            lambda row: row.__setitem__("unknown_key", 1),
            "keys changed",
        ),
    ],
)
def test_top_level_contract_fails_closed(mutation, match):
    raw = _raw()
    mutation(raw)
    with pytest.raises(MotionRecipeError, match=match):
        _validate_recipe_contract(raw)


def test_output_matrix_cannot_silently_drop_a_motion_or_scope():
    raw = _raw()
    raw["required_output_matrix"]["motion_ids"].pop()
    with pytest.raises(MotionRecipeError, match="ordered five"):
        _validate_output_matrix(raw)

    raw = _raw()
    raw["required_output_matrix"]["scopes"].reverse()
    with pytest.raises(MotionRecipeError, match="upper/full"):
        _validate_output_matrix(raw)


def test_marker_authority_path_and_sha_are_explicit_and_fail_closed(tmp_path):
    raw = _raw()
    raw["marker_authority"]["sha256"] = "0" * 64
    with pytest.raises(MotionRecipeError, match="SHA-256 mismatch"):
        _write_and_load(tmp_path, raw)

    # v1 is immutable historical provenance and is not a loadable authority.
    # The recipe picks the validation profile from this path, so an unknown or
    # non-registered path must fail closed rather than fall back to a default.
    raw = _raw()
    raw["marker_authority"][
        "path"
    ] = "configs/canonical_motion_marker_semantics_v1_20260724.json"
    with pytest.raises(MotionRecipeError, match="registered marker authority"):
        _write_and_load(tmp_path, raw)


def test_source_bytes_must_match_recipe_and_marker_authority(tmp_path):
    raw = _raw()
    _spec(raw, "fh_loop")["source_sha256"] = "0" * 64
    with pytest.raises(MotionRecipeError, match="SHA-256 mismatch"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    fh = _spec(raw, "fh_loop")
    s0 = _spec(raw, "s0_highpress")
    fh["source_path"] = s0["source_path"]
    fh["source_sha256"] = s0["source_sha256"]
    with pytest.raises(MotionRecipeError, match="marker authority v2"):
        _write_and_load(tmp_path, raw)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "protected_source_span",
        "source_anchor_frame",
        "protected_span_status",
        "entry_exit_override",
    ],
)
def test_schema2_rejects_old_hard_coded_core_fields(tmp_path, forbidden_field):
    raw = _raw()
    target = _spec(
        raw,
        "fh_block_syn" if forbidden_field == "entry_exit_override" else "fh_loop",
    )
    target[forbidden_field] = (
        {"entry_frame": 34, "exit_frame": 48}
        if forbidden_field == "entry_exit_override"
        else [48, 55]
        if forbidden_field == "protected_source_span"
        else 50
    )
    with pytest.raises(MotionRecipeError, match="keys changed"):
        _write_and_load(tmp_path, raw)


def test_synthetic_geometry_has_no_duplicate_marker_or_entry_fields(
    tmp_path,
):
    raw = _raw()
    synthetic = _spec(raw, "fh_block_syn")
    synthetic["face_manifold"]["solve_span"] = [34, 48]
    with pytest.raises(MotionRecipeError, match="keys changed"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    synthetic = _spec(raw, "fh_block_syn")
    synthetic["face_manifold"]["single_axis_pi_overlay_forbidden"] = False
    with pytest.raises(MotionRecipeError, match="single-axis"):
        _write_and_load(tmp_path, raw)


def test_post_retime_behavior_rescan_gate_is_mandatory_and_ordered():
    raw = _raw()
    raw["post_build_gates"][4] = "legacy_seed_is_certified_window"
    with pytest.raises(MotionRecipeError, match="post_build_gates"):
        _validate_recipe_contract(raw)


@pytest.mark.parametrize("bad_hash", ["0" * 63, "A" * 64, 7, True])
def test_hash_fields_require_exact_lowercase_sha_type(tmp_path, bad_hash):
    raw = _raw()
    raw["marker_authority"]["sha256"] = bad_hash
    with pytest.raises(MotionRecipeError, match="SHA-256|string"):
        _write_and_load(tmp_path, raw)


def test_recipe_schema_and_nested_objects_use_exact_keys(tmp_path):
    raw = _raw()
    raw["marker_authority"]["extra"] = 1
    with pytest.raises(MotionRecipeError, match="keys changed"):
        _write_and_load(tmp_path, raw)

    raw = _raw()
    _spec(raw, "bh_block")["scope_overrides"] = {"upper": {}}
    with pytest.raises(MotionRecipeError, match="unapproved"):
        _write_and_load(tmp_path, raw)


def test_duplicate_json_keys_are_rejected(tmp_path):
    text = _RECIPE.read_text(encoding="utf-8").replace(
        '"schema_version": 2,',
        '"schema_version": 2,\n  "schema_version": 2,',
        1,
    )
    name = f"canonical_motion_library_duplicate_{tmp_path.name}.json"
    path = _ROOT / "configs" / name
    path.write_text(text, encoding="utf-8")
    try:
        with pytest.raises(MotionRecipeError, match="duplicate JSON key"):
            load_canonical_motion_recipe(path, repo_root=_ROOT)
    finally:
        path.unlink(missing_ok=True)


def test_recipe_copy_preserves_exact_marker_binding(tmp_path):
    raw = copy.deepcopy(_raw())
    loaded = _write_and_load(tmp_path, raw)
    canonical = hashlib.sha256((_ROOT / MARKER_AUTHORITY_PATH).read_bytes()).hexdigest()
    assert loaded.marker_authority_sha256 == canonical
