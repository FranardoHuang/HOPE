from __future__ import annotations

import copy
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_motion_handoff_canonical_betas.py"
SPEC = importlib.util.spec_from_file_location("materialize_motion_handoff_canonical_betas", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOTION)

S0 = ROOT / "configs" / "motion_canonical_betas_s0_prereg_20260713.json"
M0 = ROOT / "configs" / "motion_canonical_betas_m0_prereg_20260713.json"


class PickleTorchFixture:
    @staticmethod
    def is_tensor(value):
        return False

    @staticmethod
    def save(payload, handle):
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path, **_kwargs):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("path,kind,count", [(S0, "s0_static_high_press", 1), (M0, "m0_lateral_teachers", 4)])
def test_tracked_preregistrations_pass_static(path, kind, count):
    plan, actual = MOTION.validate_plan(path, MOTION.sha256_file(path))
    assert actual == MOTION.sha256_file(path)
    assert plan["batch_kind"] == kind
    assert len(plan["inputs"]) == count


def test_static_rejects_stance_extra_key_and_fabricated_numeric_reference(tmp_path):
    plan = json.loads(M0.read_text(encoding="utf-8"))
    plan["m0_stance_handoff"]["stance_passed"] = True
    bad = tmp_path / "bad-extra.json"
    write_json(bad, plan)
    with pytest.raises(MOTION.MotionBetaError, match="must equal"):
        MOTION.validate_plan(bad, MOTION.sha256_file(bad))

    plan = json.loads(M0.read_text(encoding="utf-8"))
    plan["m0_stance_handoff"]["robot_coordinate_numeric_reference_m"] = {
        "lateral_separation": 0.3,
        "fore_aft_stagger": 0.05,
    }
    bad = tmp_path / "bad-number.json"
    write_json(bad, plan)
    with pytest.raises(MOTION.MotionBetaError, match="must equal"):
        MOTION.validate_plan(bad, MOTION.sha256_file(bad))


def test_s0_rejects_effectiveness_or_pull_paper_claim(tmp_path):
    plan = json.loads(S0.read_text(encoding="utf-8"))
    plan["semantic_guard"]["strike_effectiveness"] = 1.0
    bad = tmp_path / "bad-effect.json"
    write_json(bad, plan)
    with pytest.raises(MOTION.MotionBetaError, match="overclaims"):
        MOTION.validate_plan(bad, MOTION.sha256_file(bad))

    plan = json.loads(S0.read_text(encoding="utf-8"))
    plan["semantic_guard"]["pull_or_loop_question_paper_allowed"] = True
    bad = tmp_path / "bad-paper.json"
    write_json(bad, plan)
    with pytest.raises(MOTION.MotionBetaError, match="overclaims"):
        MOTION.validate_plan(bad, MOTION.sha256_file(bad))


def make_runtime_fixture(
    tmp_path: Path, tracked_plan: Path = S0
) -> tuple[dict, list[dict], PickleTorchFixture]:
    plan = copy.deepcopy(json.loads(tracked_plan.read_text(encoding="utf-8")))
    torch = PickleTorchFixture()

    components = np.linspace(-0.45, 0.45, 10, dtype=np.float32)
    vector_sha = MOTION.base._vector_sha256(components)
    donor_path = tmp_path / "canonical_betas.json"
    donor = {
        "schema_version": 1,
        "body_shape_contract": MOTION.BODY_SHAPE_CONTRACT,
        "components": [float(value) for value in components],
        "vector_sha256": vector_sha,
        "measured_height_m": None,
        "a3_calibrated": False,
    }
    write_json(donor_path, donor)
    completion_path = tmp_path / "donor_completion.json"
    completion = {
        "status": "complete",
        "canonical_betas_artifact": {
            "path": str(donor_path),
            "sha256": MOTION.sha256_file(donor_path),
            "vector_sha256": vector_sha,
        },
    }
    write_json(completion_path, completion)
    plan["canonical_beta_donor"] = {
        "body_shape_contract": MOTION.BODY_SHAPE_CONTRACT,
        "reuse": "exact_existing_vector_no_reaggregation",
        "artifact": {
            "path": str(donor_path),
            "bytes": donor_path.stat().st_size,
            "sha256": MOTION.sha256_file(donor_path),
        },
        "completion_manifest": {
            "path": str(completion_path),
            "bytes": completion_path.stat().st_size,
            "sha256": MOTION.sha256_file(completion_path),
        },
        "vector_sha256": vector_sha,
    }

    payloads: list[dict] = []
    runtime_results: list[dict] = []
    for index, row in enumerate(plan["inputs"]):
        frames = row["frames"]
        source_path = tmp_path / f"source-{index}.pt"
        payload = {
            "smpl_params_global": {
                "betas": np.full((frames, 10), index, dtype=np.float32),
                "body_pose": np.arange(frames * 63, dtype=np.float32).reshape(frames, 63),
            },
            "tag": f"must-remain-bit-exact-{index}",
        }
        with source_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        source_binding = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": MOTION.sha256_file(source_path),
        }
        row["source_pt"] = source_binding
        payloads.append(payload)
        runtime_results.append(
            {
                "asset_id": row["asset_id"],
                "source_sha256": str(index) * 64,
                "frames": frames,
                "gvhmr_output": source_binding,
                "queue_binding": {
                    "path": f"/irrelevant-{index}",
                    "bytes": 1,
                    "sha256": "1" * 64,
                },
                "structural_audit": {
                    "path": f"/irrelevant-audit-{index}",
                    "bytes": 1,
                    "sha256": "2" * 64,
                },
                "finite_elements": frames * 63,
                "ready_before_window_s": row["ready_before_window_s"],
                "ready_after_window_s": row["ready_after_window_s"],
            }
        )
    plan["output_contract"]["output_root"] = str(tmp_path / "materialized")
    handoff = {
        "schema_version": 1,
        "status": MOTION.HANDOFF_STATUS,
        "plan_sha256": plan["upstream_post_gvhmr"]["preregistration"]["sha256"],
        "consumer_sha256": plan["upstream_post_gvhmr"]["consumer_sha256"],
        "batch_kind": plan["batch_kind"],
        "formal_eligible": False,
        "training_authorized": False,
        "hardware_authorized": False,
        "runtime_evidence": {
            "batch_id": plan["upstream_post_gvhmr"]["batch_id"],
            "asset_ids": plan["upstream_post_gvhmr"]["asset_ids"],
            "results": runtime_results,
            "canonical_beta_donor": {
                "artifact": plan["canonical_beta_donor"]["artifact"],
                "completion_manifest": plan["canonical_beta_donor"]["completion_manifest"],
                "vector_sha256": vector_sha,
                "new_batch_materialization_status": "not_run",
            },
        },
        "downstream_gate": {
            "next_authorized_stage": "canonical_beta_materialization_only",
            "statuses": {"canonical_beta_materialization": "not_run"},
        },
    }
    handoff_path = tmp_path / "handoff.json"
    write_json(handoff_path, handoff)
    plan["handoff"] = {
        "path": str(handoff_path),
        "bytes": handoff_path.stat().st_size,
        "sha256": MOTION.sha256_file(handoff_path),
    }
    return plan, payloads, torch


@pytest.mark.parametrize("tracked_plan,expected_count", [(S0, 1), (M0, 4)])
def test_inspect_and_consume_apply_exact_donor_only_and_are_no_clobber(
    tmp_path, monkeypatch, tracked_plan, expected_count
):
    plan, source_payloads, torch = make_runtime_fixture(tmp_path, tracked_plan)
    monkeypatch.setattr(MOTION.base, "_load_torch", lambda: torch)
    evidence = MOTION.inspect_inputs(plan)
    evidence["execution_fingerprint"] = {
        "python_executable": "/fixture/python",
        "python_version": "Python fixture",
        "pip_freeze_sha256": "f" * 64,
        "CUDA_VISIBLE_DEVICES": "",
        "cpu_only": True,
    }
    assert len(evidence["loaded"]) == expected_count
    assert MOTION.base._vector_sha256(evidence["canonical"]) == plan["canonical_beta_donor"]["vector_sha256"]

    manifest_path = MOTION.materialize(plan, tmp_path / "plan.json", "3" * 64, evidence)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["status"] == MOTION.RESULT_STATUS
    assert result["canonical_beta_artifact_semantics"].startswith("byte-exact donor copy")
    assert len(result["results"]) == expected_count
    for row, source_payload in zip(result["results"], source_payloads, strict=True):
        assert row["non_beta_bit_exact"] is True
        output = torch.load(Path(row["output_path"]))
        expected = np.broadcast_to(evidence["canonical"], (row["frames"], 10))
        assert np.array_equal(output["smpl_params_global"]["betas"], expected)
        assert np.array_equal(
            output["smpl_params_global"]["body_pose"],
            source_payload["smpl_params_global"]["body_pose"],
        )
        assert output["tag"] == source_payload["tag"]
    if expected_count == 4:
        assert result["m0_stance_handoff"]["foot_site_mapping"] is None
        assert result["m0_stance_handoff"]["stance_passed"] is None
    assert (manifest_path.parent / "canonical_betas.json").read_bytes() == evidence["donor_bytes"]

    with pytest.raises(MOTION.MotionBetaError, match="already exists"):
        MOTION.materialize(plan, tmp_path / "plan.json", "3" * 64, evidence)


def test_handoff_mutation_and_reordered_assets_fail_closed(tmp_path):
    plan, _source, _torch = make_runtime_fixture(tmp_path)
    handoff_path = Path(plan["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["runtime_evidence"]["results"][0]["ready_after_window_s"] = [1.9, 2.8]
    write_json(handoff_path, handoff)
    plan["handoff"]["bytes"] = handoff_path.stat().st_size
    plan["handoff"]["sha256"] = MOTION.sha256_file(handoff_path)
    with pytest.raises(MOTION.MotionBetaError, match="ready_after_window_s mismatch"):
        MOTION.validate_handoff(plan)

    plan, _source, _torch = make_runtime_fixture(tmp_path / "fresh")
    handoff_path = Path(plan["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["runtime_evidence"]["asset_ids"] = list(reversed(handoff["runtime_evidence"]["asset_ids"]))
    handoff["runtime_evidence"]["asset_ids"].append("unexpected")
    write_json(handoff_path, handoff)
    plan["handoff"]["bytes"] = handoff_path.stat().st_size
    plan["handoff"]["sha256"] = MOTION.sha256_file(handoff_path)
    with pytest.raises(MOTION.MotionBetaError, match="batch/asset order mismatch"):
        MOTION.validate_handoff(plan)
