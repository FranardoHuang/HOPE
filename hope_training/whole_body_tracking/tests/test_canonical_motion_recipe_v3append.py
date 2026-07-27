"""Recipe-level admission of a sixth motion, and non-regression of the five.

The recipe loader used to say "exactly five motions" in four places.  That rule
was doing two jobs at once: keeping the canonical five from being swapped or
reordered (worth keeping) and forbidding this repository from ever holding a
sixth stroke (never the point).  It is now an ordered-prefix rule.
"""

from __future__ import annotations

import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from canonical_motion_markers import (  # noqa: E402
    EVENT_KIND_DERIVED_CONTACT,
    MARKER_AUTHORITY_PATH,
    MARKER_AUTHORITY_SHA256,
    sha256_file,
)
from canonical_motion_recipe import (  # noqa: E402
    MotionRecipeError,
    load_canonical_motion_recipe,
)

_V2_LIBRARY = _ROOT / "configs" / "canonical_motion_library_v2_20260724.json"
_V3_LIBRARY = _ROOT / "configs" / "canonical_motion_library_v3append_20260727.json"
_V2_LIBRARY_SHA256 = (
    "327d9f70dd674441308a6a03af6f4a21d20d5bf3d9010ffe2c7de0e8cc7e44fb"
)
_CANONICAL_FIVE = (
    "fh_loop",
    "bh_loop_c",
    "fh_block_syn",
    "bh_block",
    "s0_highpress",
)


@contextmanager
def _temporary_recipe(tmp_path: Path, raw: dict):
    # configs/ is the only directory the loader will accept a recipe from with
    # the repo_root convention the other tests use.
    path = _ROOT / "configs" / f"canonical_motion_library_test_{tmp_path.name}.json"
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


def _raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_library_and_v2_authority_bytes_did_not_move():
    assert sha256_file(_V2_LIBRARY) == _V2_LIBRARY_SHA256
    assert sha256_file(_ROOT / MARKER_AUTHORITY_PATH) == MARKER_AUTHORITY_SHA256


def test_v2_library_still_loads_with_the_same_five_motions():
    recipe = load_canonical_motion_recipe(_V2_LIBRARY, repo_root=_ROOT)
    assert tuple(row.motion_id for row in recipe.sources) == _CANONICAL_FIVE
    assert recipe.marker_semantics.profile == "v2"
    assert recipe.marker_authority_sha256 == MARKER_AUTHORITY_SHA256
    assert recipe.raw["required_output_matrix"]["candidate_count"] == 10


def test_v3append_library_admits_the_new_forehand_end_to_end():
    recipe = load_canonical_motion_recipe(_V3_LIBRARY, repo_root=_ROOT)
    ids = tuple(row.motion_id for row in recipe.sources)
    assert ids[:5] == _CANONICAL_FIVE
    assert ids[5] == "fh_loop_high"
    assert recipe.marker_semantics.profile == "v3"

    new_source = recipe.source("fh_loop_high")
    assert new_source.clip.n_frames == 98
    assert new_source.sha256 == (
        "7d045fcb036ffa668dede4607cfcc82e789a0db7ab86fd8df9dd52cfd5ac4153"
    )
    row = recipe.marker_semantics.row("fh_loop_high")
    assert row.event_provenance_kind == EVENT_KIND_DERIVED_CONTACT
    # Every asserted frame is inside the 98-frame clip, which is the check the
    # old "reuse fh_loop's markers" path would have failed on frame 61 only by
    # luck (61 < 98) and would have passed while meaning nothing.
    assert max(row.authority_frames()) < new_source.clip.n_frames
    assert row.contact_anchor() == (54, "derived_source_timing_contact")
    assert row.search_window() == ((47, 57), "derived_seed")


def test_the_canonical_five_keep_their_v2_sources_in_the_v3_library():
    v2 = load_canonical_motion_recipe(_V2_LIBRARY, repo_root=_ROOT)
    v3 = load_canonical_motion_recipe(_V3_LIBRARY, repo_root=_ROOT)
    for motion_id in _CANONICAL_FIVE:
        old = v2.source(motion_id)
        new = v3.source(motion_id)
        assert old.sha256 == new.sha256
        assert old.path == new.path
        old_row = v2.marker_semantics.row(motion_id)
        new_row = v3.marker_semantics.row(motion_id)
        assert old_row.contact_anchor() == new_row.contact_anchor()
        assert old_row.search_window() == new_row.search_window()
        assert old_row.authority_frames() == new_row.authority_frames()


def test_the_canonical_five_may_not_be_dropped_or_reordered(tmp_path):
    raw = _raw(_V3_LIBRARY)
    raw["motion_specs"][0], raw["motion_specs"][1] = (
        raw["motion_specs"][1],
        raw["motion_specs"][0],
    )
    raw["required_output_matrix"]["motion_ids"][0] = "bh_loop_c"
    raw["required_output_matrix"]["motion_ids"][1] = "fh_loop"
    with pytest.raises(MotionRecipeError, match="must begin with the canonical"):
        _write_and_load(tmp_path, raw)

    raw = _raw(_V3_LIBRARY)
    del raw["motion_specs"][5]
    with pytest.raises(MotionRecipeError):
        _write_and_load(tmp_path, raw)


def test_output_matrix_must_agree_with_the_motion_specs(tmp_path):
    raw = _raw(_V3_LIBRARY)
    raw["required_output_matrix"]["candidate_count"] = 10
    with pytest.raises(MotionRecipeError, match="candidate_count must be 12"):
        _write_and_load(tmp_path, raw)

    raw = _raw(_V3_LIBRARY)
    raw["required_output_matrix"]["motion_ids"].append("fh_loop_high_extra")
    raw["required_output_matrix"]["candidate_count"] = 14
    with pytest.raises(MotionRecipeError, match="disagree"):
        _write_and_load(tmp_path, raw)


def test_a_sixth_motion_cannot_ride_on_the_v2_authority(tmp_path):
    """v2 has no row for the new take, so the v2 authority must refuse it."""

    raw = _raw(_V3_LIBRARY)
    raw["marker_authority"] = {
        "path": MARKER_AUTHORITY_PATH,
        "sha256": MARKER_AUTHORITY_SHA256,
    }
    # Raised as MarkerSemanticsError (a ValueError) by MarkerSemantics.row;
    # the recipe loader does not wrap that call, which is pre-existing.
    with pytest.raises(ValueError, match="0 rows named 'fh_loop_high'"):
        _write_and_load(tmp_path, raw)


def test_a_new_motion_cannot_be_admitted_by_repointing_an_existing_row(tmp_path):
    """The move the untracked v3 library actually attempted, and why it failed.

    Swapping ``fh_loop``'s source for the 2026-07-27 take leaves the authority
    still describing the 2026-07-10 take, so the source no longer closes.
    """

    raw = _raw(_V2_LIBRARY)
    spec = next(row for row in raw["motion_specs"] if row["motion_id"] == "fh_loop")
    spec["source_path"] = (
        "vendor_assets/motion_finalize_20260724/sources/SHADOW_fh_loop_high_yaw152.npz"
    )
    spec["source_sha256"] = (
        "7d045fcb036ffa668dede4607cfcc82e789a0db7ab86fd8df9dd52cfd5ac4153"
    )
    with pytest.raises(MotionRecipeError, match="does not close against marker"):
        _write_and_load(tmp_path, raw)


def test_scope_overrides_still_gated_for_the_appended_motion(tmp_path):
    raw = _raw(_V3_LIBRARY)
    spec = copy.deepcopy(raw["motion_specs"][5])
    spec["scope_overrides"] = {"full": {"grounding_policy": "anything"}}
    raw["motion_specs"][5] = spec
    with pytest.raises(MotionRecipeError, match="unapproved scope override"):
        _write_and_load(tmp_path, raw)
