from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "run_motion_video_canonical_gmr_queue.py"
SPEC = importlib.util.spec_from_file_location("run_motion_video_canonical_gmr_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def test_tracked_plan_and_tool_contract_are_self_consistent():
    plan_path = ROOT / "configs" / "motion_video_canonical_gmr_prereg_20260711.json"
    plan_sha = QUEUE.sha256_file(plan_path)
    plan = QUEUE.validate_plan(plan_path, plan_sha)
    assert plan["processing_order"] == [row["asset_id"] for row in plan["inputs"]]
    assert plan["processing_order"][0] == "franco_forehand_block"
    assert len(plan["processing_order"]) == 10
    assert plan["gmr_source_contract"]["loader_semantics"] == {
        "field": "smpl_params_global.betas",
        "selection": "betas[0].detach().cpu().numpy()[:10]",
        "selected_components": 10,
        "zero_padding": False,
        "height_formula_not_calibration": "1.66 + 0.1 * selected_betas[0]",
    }
    QUEUE.verify_tool_contract(plan)


def test_plan_hash_and_lineage_fail_closed(tmp_path):
    tracked = ROOT / "configs" / "motion_video_canonical_gmr_prereg_20260711.json"
    payload = json.loads(tracked.read_text(encoding="utf-8"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QUEUE.QueueError, match="plan sha256"):
        QUEUE.validate_plan(plan_path, "0" * 64)

    payload["body_shape_contract"] = "diagnostic_video_betas"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QUEUE.QueueError, match="body_shape_contract"):
        QUEUE.validate_plan(plan_path, QUEUE.sha256_file(plan_path))


def test_materialization_verifier_binds_every_input_and_vector(tmp_path):
    canonical = tmp_path / "canonical_betas.json"
    canonical.write_text("{}\n", encoding="utf-8")
    vector_sha = "1" * 64
    rows = []
    completion_rows = []
    for index in range(2):
        source = tmp_path / f"asset{index}.pt"
        source.write_bytes(f"source-{index}".encode())
        asset_id = f"asset{index}"
        binding = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": QUEUE.sha256_file(source),
        }
        rows.append(
            {
                "asset_id": asset_id,
                "frames": 5 + index,
                "canonical_vector_sha256": vector_sha,
                "input": binding,
            }
        )
        completion_rows.append(
            {
                "asset_id": asset_id,
                "output_path": str(source),
                "output_bytes": source.stat().st_size,
                "output_sha256": QUEUE.sha256_file(source),
                "frames": 5 + index,
                "output_canonical_vector_sha256": vector_sha,
                "frame_beta_max_abs_deviation_from_video_median": 0.0,
                "non_beta_bit_exact": True,
                "output_beta_contract": {
                    "shape": [5 + index, 10],
                    "dtype": "torch.float32",
                    "shape_contract": "frames_by_10",
                },
            }
        )
    completion = {
        "body_shape_contract": QUEUE.BODY_SHAPE_CONTRACT,
        "formal_eligible": False,
        "a3_calibrated": False,
        "canonical_betas_artifact": {
            "path": str(canonical),
            "sha256": QUEUE.sha256_file(canonical),
            "vector_sha256": vector_sha,
        },
        "results": completion_rows,
    }
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    plan = {
        "source_materialization": {
            "completion_manifest": {
                "path": str(completion_path),
                "bytes": completion_path.stat().st_size,
                "sha256": QUEUE.sha256_file(completion_path),
            },
            "canonical_betas_artifact": {
                "path": str(canonical),
                "bytes": canonical.stat().st_size,
                "sha256": QUEUE.sha256_file(canonical),
            },
            "canonical_vector_sha256": vector_sha,
        },
        "inputs": rows,
    }
    _, verified = QUEUE.verify_materialization(plan)
    assert [row["asset_id"] for row in verified] == ["asset0", "asset1"]
    completion["results"][1]["output_canonical_vector_sha256"] = "2" * 64
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    plan["source_materialization"]["completion_manifest"].update(
        bytes=completion_path.stat().st_size,
        sha256=QUEUE.sha256_file(completion_path),
    )
    with pytest.raises(QUEUE.QueueError, match="canonical_vector"):
        QUEUE.verify_materialization(plan)


def test_selection_preserves_requested_order_and_rejects_duplicates():
    rows = [{"asset_id": "a"}, {"asset_id": "b"}]
    assert QUEUE.select_rows(rows, None) == rows
    assert [row["asset_id"] for row in QUEUE.select_rows(rows, ["b", "a"])] == ["b", "a"]
    with pytest.raises(QUEUE.QueueError, match="unique"):
        QUEUE.select_rows(rows, ["a", "a"])
    with pytest.raises(QUEUE.QueueError, match="absent"):
        QUEUE.select_rows(rows, ["missing"])
