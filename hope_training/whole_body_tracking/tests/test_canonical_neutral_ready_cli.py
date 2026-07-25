"""Contract tests for the real neutral-ready source reconstruction adapter."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import canonical_face_manifold as face  # noqa: E402
import canonical_motion_recipe as recipe_module  # noqa: E402
import canonical_mujoco_dynamics_gate as dynamics_gate  # noqa: E402
import canonical_neutral_ready as neutral  # noqa: E402
import canonical_neutral_ready_cli as adapter  # noqa: E402


REPO = Path(__file__).resolve().parents[3]
RECIPE = REPO / "configs/canonical_motion_library_v2_20260724.json"
AUTHORITY = (
    REPO / "configs/canonical_motion_marker_semantics_v1_20260724.json"
)
RECIPE_SHA256 = "327d9f70dd674441308a6a03af6f4a21d20d5bf3d9010ffe2c7de0e8cc7e44fb"


def _source():
    return SimpleNamespace(
        sha256=(
            "55870b981584a458bfd479171046445845cb74171618b71338"
            "fd9dc9f66a5fe0"
        ),
    )


def _marker_row(
    motion_id: str,
    *,
    nominal: int | None,
    ge80: tuple[int, int] = (40, 45),
    construction=None,
):
    return SimpleNamespace(
        motion_id=motion_id,
        nominal_event=nominal,
        ge50_seed=(39, 46),
        ge80_seed=ge80,
        preferred_seed=None,
        construction_marker=construction,
        historical_adv2c3_start=29,
    )


def _marker_semantics(
    *,
    bh_nominal: int = 44,
    bh_ge80: tuple[int, int] = (40, 45),
    fh_ge80: tuple[int, int] = (40, 45),
    annotation: int = 44,
    donor_preferred: int = 42,
):
    rows = {
        "bh_block": _marker_row("bh_block", nominal=bh_nominal, ge80=bh_ge80),
        "fh_block_syn": _marker_row(
            "fh_block_syn",
            nominal=None,
            ge80=fh_ge80,
            construction=SimpleNamespace(
                annotation_frame=annotation,
                donor_preferred_frame=donor_preferred,
                solve_span=(34, 48),
            ),
        ),
    }
    return SimpleNamespace(row=lambda motion_id: rows[motion_id])


def _phase_binding(tmp_path: Path) -> adapter.BlockPhaseMapBinding:
    del tmp_path
    return adapter.load_block_phase_map_binding(
        AUTHORITY,
        hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
    )


def test_reviewed_block_phase_frames_keep_f42_and_f44_distinct(tmp_path):
    assert adapter.reviewed_block_phase_frames(
        _source(),
        _source(),
        _marker_semantics(),
        _phase_binding(tmp_path),
    ) == {
        "opportunity_start": 40,
        "construction_donor_preferred": 42,
        "nominal_event": 44,
        "opportunity_end": 45,
    }


def test_recipe_anchor_is_nominal_event_and_construction_seed_is_not_behavior(
    tmp_path,
):
    binding = _phase_binding(tmp_path)
    assert binding.synthetic_behavior_preferred is None
    assert binding.frames["construction_donor_preferred"] == 42
    with pytest.raises(
        adapter.NeutralReadyAdapterError,
        match="nominal event",
    ):
        adapter.reviewed_block_phase_frames(
            _source(),
            _source(),
            _marker_semantics(bh_nominal=42, annotation=42),
            binding,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.BlockPhaseMapBinding(
            authority_path=tmp_path / "irrelevant",
            expected_authority_sha256="0" * 64,
            source_motion_sha256="1" * 64,
            opportunity_start=40,
            construction_donor_preferred=44,
            nominal_event=44,
            opportunity_end=45,
            synthetic_behavior_preferred=None,
            review_status="REVIEWED_FOR_LEGACY_SEED_PROVENANCE_ONLY",
        )
    with pytest.raises(
        adapter.NeutralReadyAdapterError,
        match="ge80 seed spans disagree",
    ):
        adapter.reviewed_block_phase_frames(
            _source(),
            _source(),
            _marker_semantics(fh_ge80=(39, 45)),
            _phase_binding(tmp_path),
        )


def test_marker_authority_v2_keeps_block_span_and_face_annotation():
    raw = json.loads(RECIPE.read_text(encoding="utf-8"))
    authority_raw = json.loads(
        (REPO / raw["marker_authority"]["path"]).read_text(encoding="utf-8")
    )
    rows = {row["motion_id"]: row for row in authority_raw["motions"]}
    for motion_id in ("bh_block", "fh_block_syn"):
        assert rows[motion_id]["legacy_ge80_seed"]["span_inclusive"] == [40, 45]
    assert rows["bh_block"]["nominal_event"]["frame"] == 44
    construction = rows["fh_block_syn"]["construction_marker"]
    assert construction["annotation_frame"] == 44
    assert construction["donor_preferred_frame"] == 42
    assert construction["solve_span_inclusive"] == [34, 48]
    specs = {row["motion_id"]: row for row in raw["motion_specs"]}
    assert "solver_anchor_frame" not in specs["fh_block_syn"]["face_manifold"]


def test_recipe_decoder_uses_private_content_snapshot():
    raw = json.loads(RECIPE.read_text(encoding="utf-8"))
    required = [
        REPO / raw["canonical_ready"]["path"],
        *[
            REPO / raw["model_contract"][f"{name}_path"]
            for name in ("mjcf", "urdf", "body_order")
        ],
        *[REPO / row["source_path"] for row in raw["motion_specs"]],
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("private canonical recipe inputs are absent")
    recipe, bindings = adapter._snapshot_recipe_inputs(
        RECIPE,
        repo_root=REPO,
        expected_recipe_sha256=RECIPE_SHA256,
    )
    assert recipe.path == RECIPE
    assert len(recipe.sources) == 5
    assert recipe.source("bh_block").clip.n_frames == 104
    assert bindings[RECIPE] == hashlib.sha256(RECIPE.read_bytes()).hexdigest()
    assert bindings[recipe.ready.path] == recipe.ready.sha256


def test_recipe_decoder_requires_independent_recipe_pin():
    with pytest.raises(
        adapter.NeutralReadyAdapterError,
        match="independently reviewed pin",
    ):
        adapter._snapshot_recipe_inputs(
            RECIPE,
            repo_root=REPO,
            expected_recipe_sha256="0" * 64,
        )


def test_contact_seals_canonical_unit_normal_before_digest(tmp_path):
    source = tmp_path / "source.npz"
    np.savez(source, fixture=np.asarray([1], dtype=np.int64))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    class Backend:
        def site_pose(self, joint_pos, root_pos_w, root_quat_w):
            del joint_pos, root_pos_w, root_quat_w
            rotation = np.eye(3, dtype=np.float64)
            rotation[:, 1] *= 1.0 + 1.0e-10
            return np.zeros(3, dtype=np.float64), rotation

    row = adapter._contact(
        scope="upper",
        phase="opportunity_start",
        face_name="bh",
        frame=40,
        joint_pos=np.zeros(31, dtype=np.float64),
        root_pos_w=np.zeros(3, dtype=np.float64),
        root_quat_w=np.asarray([1.0, 0.0, 0.0, 0.0]),
        source_path=source,
        source_sha256=source_sha,
        backend=Backend(),
    )
    assert np.linalg.norm(row.signed_face_normal_w) == 1.0
    assert neutral.contact_pose_digest(row) == row.pose_content_sha256
    np.testing.assert_array_equal(
        neutral._unit(row.signed_face_normal_w, "test"),
        row.signed_face_normal_w,
    )


def test_real_16_contact_reconstruction_when_private_plant_is_available(
    tmp_path,
):
    raw = json.loads(RECIPE.read_text(encoding="utf-8"))
    required = [
        REPO / raw["canonical_ready"]["path"],
        REPO / raw["model_contract"]["mjcf_path"],
        REPO / raw["model_contract"]["urdf_path"],
    ]
    required.extend(
        REPO / row["source_path"] for row in raw["motion_specs"]
    )
    if not all(path.is_file() for path in required):
        pytest.skip("private canonical motion/plant assets are absent")
    pytest.importorskip("mujoco")

    recipe, _ = adapter._snapshot_recipe_inputs(
        RECIPE,
        repo_root=REPO,
        expected_recipe_sha256=RECIPE_SHA256,
    )
    backend = face.MujocoRightRacketBackend(
        recipe.model_paths["mjcf"],
        dynamics_gate.RUNTIME_JOINT_NAMES,
        urdf_path=recipe.model_paths["urdf"],
    )
    measured = adapter.measure_exact_model_binding(recipe, backend)
    loaded = adapter.load_real_neutral_ready_inputs(
        RECIPE,
        expected_recipe_sha256=RECIPE_SHA256,
        repo_root=REPO,
        backend=backend,
        model_binding=measured,
        phase_map_binding=_phase_binding(tmp_path),
    )
    assert len(loaded.contacts) == 16
    assert {
        (row.scope, row.phase, row.face_name)
        for row in loaded.contacts
    } == {
        (scope, phase, face_name)
        for scope in neutral.SCOPES
        for phase in neutral.PHASES
        for face_name in neutral.FACES
    }
    assert {
        row.phase: row.source_frame_index for row in loaded.contacts
    } == {
        "opportunity_start": 40,
        "construction_donor_preferred": 42,
        "nominal_event": 44,
        "opportunity_end": 45,
    }
    lookup = {
        (row.scope, row.phase, row.face_name): row
        for row in loaded.contacts
    }
    for scope in neutral.SCOPES:
        for phase in neutral.PHASES:
            bh = lookup[(scope, phase, "bh")]
            fh = lookup[(scope, phase, "fh")]
            assert neutral._angle(
                bh.signed_face_normal_w, fh.signed_face_normal_w
            ) == pytest.approx(3.141592653589793, abs=2.0e-5)
            assert (
                ((bh.site_pos_w - fh.site_pos_w) ** 2).sum() ** 0.5
                <= 2.0e-6
            )
    adapter.verify_real_contact_source_reconstruction(
        loaded.contact_source_proof,
        loaded.contacts,
        backend=backend,
        model_binding=measured,
    )
