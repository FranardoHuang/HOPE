from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "consume_motion_post_gvhmr_exact.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("consume_motion_post_gvhmr_exact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONSUMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSUMER)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def binding(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": CONSUMER.sha256_file(path),
    }


def fixture_plan(tmp_path: Path) -> dict:
    state = tmp_path / "state"
    output = tmp_path / "gvhmr" / "asset.pt"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"exact-gvhmr-result")
    asset_id = "asset"
    source_sha = "1" * 64
    prereg_sha = "2" * 64
    intake_sha = "3" * 64
    review_sha = "4" * 64
    vector_sha = "5" * 64

    audit_path = state / "audits" / f"{asset_id}.json"
    audit_payload = {
        "schema_version": 1,
        "status": "pass",
        "result_path": str(output),
        "result_bytes": output.stat().st_size,
        "result_sha256": CONSUMER.sha256_file(output),
        "expected_frames": 8,
        "actual_frames": 8,
        "finite_elements": 632,
    }
    write_json(audit_path, audit_payload)

    record_path = tmp_path / "execution_record.json"
    record_payload = {
        "schema_version": 1,
        "status": "ready_for_exact_offline_gvhmr",
        "authorization_scope": CONSUMER.AUTHORIZATION_SCOPE,
        "prereg_sha256": prereg_sha,
        "manifest_sha256": intake_sha,
        "manual_event_review_sha256": review_sha,
        "state_root": str(state),
        "processing_order": [asset_id],
        "downstream_consumer": "none_beyond_structural_auditor",
    }
    write_json(record_path, record_payload)
    record_sha = CONSUMER.sha256_file(record_path)

    binding_path = state / "bindings" / f"{asset_id}.json"
    binding_payload = {
        "schema_version": 1,
        "status": "complete",
        "asset_id": asset_id,
        "source_sha256": source_sha,
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": CONSUMER.sha256_file(output),
        "returncode": 0,
        "audit_returncode": 0,
        "gvhmr_commit": "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
        "structural_audit_path": str(audit_path),
        "structural_audit_sha256": CONSUMER.sha256_file(audit_path),
        "processing_contract": {
            "manifest_sha256": intake_sha,
            "manual_event_review_sha256": review_sha,
            "prereg_sha256": prereg_sha,
            "execution_record_sha256": record_sha,
            "authorization_scope": CONSUMER.AUTHORIZATION_SCOPE,
            "source_consumption": CONSUMER.SOURCE_CONSUMPTION,
            "gvhmr_commit": "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
            "bound_source_snapshot_sha256": {asset_id: source_sha},
        },
    }
    write_json(binding_path, binding_payload)

    queue_path = state / "queue_state.json"
    queue_payload = {
        "schema_version": 2,
        "status": "complete",
        "authorization_scope": CONSUMER.AUTHORIZATION_SCOPE,
        "prereg_sha256": prereg_sha,
        "execution_record_sha256": record_sha,
        "asset_ids": [asset_id],
        "batch_id": "fixture_batch",
        "source_consumption": CONSUMER.SOURCE_CONSUMPTION,
        "bound_source_snapshots": {
            asset_id: {"snapshot_before": {"sha256": source_sha}}
        },
    }
    write_json(queue_path, queue_payload)

    artifact_path = tmp_path / "canonical_betas.json"
    artifact_payload = {
        "schema_version": 1,
        "body_shape_contract": CONSUMER.BODY_SHAPE_CONTRACT,
        "vector_sha256": vector_sha,
        "components": [float(index) / 10.0 for index in range(10)],
        "measured_height_m": None,
        "a3_calibrated": False,
    }
    write_json(artifact_path, artifact_payload)
    completion_path = tmp_path / "materialization_manifest.json"
    completion_payload = {
        "schema_version": 1,
        "status": "complete",
        "canonical_betas_artifact": {
            "path": str(artifact_path),
            "sha256": CONSUMER.sha256_file(artifact_path),
            "vector_sha256": vector_sha,
        },
    }
    write_json(completion_path, completion_payload)

    output_binding = binding(output)
    plan = {
        "batch_kind": "fixture",
        "intake_sha256": intake_sha,
        "manual_event_review_sha256": review_sha,
        "consumer": {"path": "scripts/consume_motion_post_gvhmr_exact.py", "sha256": "6" * 64},
        "upstream_gvhmr": {
            "preregistration": {"path": "fixture.json", "sha256": prereg_sha},
            "batch_id": "fixture_batch",
            "asset_ids": [asset_id],
            "state_root": str(state),
            "execution_record": binding(record_path),
            "queue_state": binding(queue_path),
            "assets": [
                {
                    "asset_id": asset_id,
                    "source_sha256": source_sha,
                    "frames": 8,
                    "ready_before_window_s": [0.0, 0.2],
                    "ready_after_window_s": [0.6, 0.8],
                    "output": output_binding,
                    "binding": binding(binding_path),
                    "structural_audit": binding(audit_path),
                }
            ],
        },
        "canonical_beta_donor": {
            "artifact": binding(artifact_path),
            "completion_manifest": binding(completion_path),
            "vector_sha256": vector_sha,
        },
        "semantic_guard": {"fixture": True},
        "downstream_gate": {"fixture": True},
        "handoff_output": {
            "root": str(tmp_path / "handoff"),
            "path": str(tmp_path / "handoff" / "handoff.json"),
        },
    }
    return plan


@pytest.mark.parametrize(
    "name,kind,asset_ids",
    [
        (
            "motion_post_gvhmr_s0_prereg_20260713.json",
            "s0_static_high_press",
            ["static_backhand_high_press"],
        ),
        (
            "motion_post_gvhmr_m0_prereg_20260713.json",
            "m0_lateral_teachers",
            [
                "lateral_step_left_1",
                "lateral_step_left_2",
                "lateral_step_right_1",
                "lateral_step_right_2",
            ],
        ),
    ],
)
def test_tracked_preregistrations_are_exact_and_static_valid(name, kind, asset_ids):
    path = ROOT / "configs" / name
    plan, _static, actual = CONSUMER.validate_plan(path, CONSUMER.sha256_file(path))
    assert actual == CONSUMER.sha256_file(path)
    assert plan["batch_kind"] == kind
    assert plan["upstream_gvhmr"]["asset_ids"] == asset_ids
    assert plan["consumer"]["sha256"] == CONSUMER.sha256_file(SCRIPT)


def test_s0_cannot_borrow_pull_paper_or_claim_effectiveness():
    plan = json.loads(
        (ROOT / "configs" / "motion_post_gvhmr_s0_prereg_20260713.json").read_text()
    )
    review = json.loads(
        (ROOT / "configs" / "motion_video_manual_event_review_20260713.json").read_text()
    )
    review_by_id = {row["asset_id"]: row for row in review["assets"]}
    plan["semantic_guard"]["pull_or_loop_question_paper_allowed"] = True
    with pytest.raises(CONSUMER.HandoffError, match="borrowed a pull/loop"):
        CONSUMER._validate_batch_semantics(plan, review_by_id)
    plan["semantic_guard"]["pull_or_loop_question_paper_allowed"] = False
    plan["semantic_guard"]["strike_effectiveness"] = 1.0
    with pytest.raises(CONSUMER.HandoffError, match="borrowed a pull/loop"):
        CONSUMER._validate_batch_semantics(plan, review_by_id)


def test_m0_requires_full_initial_foot_vector_and_rejects_feet_together():
    plan = json.loads(
        (ROOT / "configs" / "motion_post_gvhmr_m0_prereg_20260713.json").read_text()
    )
    review = json.loads(
        (ROOT / "configs" / "motion_video_manual_event_review_20260713.json").read_text()
    )
    review_by_id = {row["asset_id"]: row for row in review["assets"]}
    stance = plan["semantic_guard"]["terminal_stance_contract"]
    assert stance["must_preserve_components"] == ["lateral_separation", "fore_aft_stagger"]
    stance["feet_together_or_narrower_substitute_allowed"] = True
    with pytest.raises(CONSUMER.HandoffError, match="stance vector contract"):
        CONSUMER._validate_batch_semantics(plan, review_by_id)


def test_runtime_chain_inspects_and_publishes_once(tmp_path):
    plan = fixture_plan(tmp_path)
    evidence = CONSUMER.inspect_runtime_evidence(plan)
    assert evidence["asset_ids"] == ["asset"]
    assert evidence["results"][0]["finite_elements"] == 632
    handoff = CONSUMER.publish_handoff(plan, "7" * 64, evidence)
    payload = json.loads(handoff.read_text())
    assert payload["status"] == CONSUMER.HANDOFF_STATUS
    assert payload["runtime_evidence"]["results"][0]["gvhmr_output"] == plan["upstream_gvhmr"]["assets"][0]["output"]
    with pytest.raises(CONSUMER.HandoffError, match="already exists"):
        CONSUMER.publish_handoff(plan, "7" * 64, evidence)


def test_runtime_rejects_processing_lineage_mutation_even_when_rehashed(tmp_path):
    plan = fixture_plan(tmp_path)
    row = plan["upstream_gvhmr"]["assets"][0]
    path = Path(row["binding"]["path"])
    payload = json.loads(path.read_text())
    payload["processing_contract"]["execution_record_sha256"] = "8" * 64
    write_json(path, payload)
    row["binding"] = binding(path)
    with pytest.raises(CONSUMER.HandoffError, match="execution_record_sha256"):
        CONSUMER.inspect_runtime_evidence(plan)


def test_runtime_rejects_unexpected_binding_file(tmp_path):
    plan = fixture_plan(tmp_path)
    state = Path(plan["upstream_gvhmr"]["state_root"])
    write_json(state / "bindings" / "stale.json", {"status": "complete"})
    with pytest.raises(CONSUMER.HandoffError, match="unexpected JSON binding"):
        CONSUMER.inspect_runtime_evidence(plan)


def test_strict_json_rejects_duplicates_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n')
    with pytest.raises(CONSUMER.HandoffError, match="duplicate key"):
        CONSUMER.read_json(duplicate, "duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n')
    with pytest.raises(CONSUMER.HandoffError, match="non-finite"):
        CONSUMER.read_json(nonfinite, "nonfinite")
