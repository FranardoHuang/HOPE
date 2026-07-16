"""Dependency-light source guards for checkpoint-bound planner revision ONNX metadata."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"
STANDALONE = ROOT / "scripts/standalone_onnx_export.py"


def _source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    return source


def test_both_exporters_use_one_checkpoint_contract_canonicalizer():
    native = _source(EXPORTER)
    standalone = _source(STANDALONE)
    assert "planner_task_revision_metadata," in native
    assert "planner_revision_metadata_json = planner_task_revision_metadata(training_contract)" in (
        native
    )
    assert "planner_task_revision_metadata = _TC.planner_task_revision_metadata" in standalone
    assert "planner_revision_metadata_json = planner_task_revision_metadata(training_contract)" in (
        standalone
    )
    assert "bind_planner_task_revision_metadata(metadata, training_contract)" in native
    assert "bind_planner_task_revision_metadata(" in standalone


def test_native_export_refuses_environment_backfill_or_contract_mismatch():
    source = _source(EXPORTER)
    assert "runtime_planner_revision != checkpoint_planner_revision" in source
    assert 'training_contract.get("planner_task_revision")' in source
    assert "legacy/OFF checkpoints" in source
    assert 'actor_contract.name != "deploy_parity_face179"' in source


def test_standalone_strips_donor_claim_and_rebuilds_only_from_checkpoint():
    source = _source(STANDALONE)
    strip = source.index('key == "planner_task_revision"')
    rebuild = source.index("bind_planner_task_revision_metadata(")
    assert strip < rebuild
    assert "The donor is never authority for revision capability" in source
    assert "donor_planner_revision_claim == bound_planner_revision" in source
    assert "donor planner_task_revision metadata != checkpoint training contract" in source
    assert 'actor_contract == "deploy_parity_face179"' in source


def test_revision_export_keeps_checkpoint_bound_strike_layout_truth_source():
    native = _source(EXPORTER)
    standalone = _source(STANDALONE)
    assert 'metadata["clip_seg_lengths"]' in native
    assert 'metadata["clip_strike_phases"]' in native
    assert '"clip_seg_lengths": contract["motion_segment_lengths"]' in standalone
    assert 'training_contract.get("strike_phase_per_clip", [])' in standalone
    assert '"clip_strike_phases": ",".join' in standalone
