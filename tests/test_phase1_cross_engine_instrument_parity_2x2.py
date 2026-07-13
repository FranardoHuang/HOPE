import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs" / "phase1_cross_engine_instrument_parity_2x2_prereg_20260711.json"
REVOCATION = (
    ROOT
    / "configs"
    / "phase1_cross_engine_instrument_parity_2x2_revocation_20260713.json"
)
VALIDATOR_PATH = ROOT / "scripts" / "validate_phase1_cross_engine_instrument_parity_2x2.py"
ADAPTER_PATH = ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "isaac_bank_exam_adapter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def numeric(values):
    return {"shape": [len(values)], "values": list(values)}


def state_document():
    return {
        "schema": "hope.cross-engine-state-instrumentation.v1",
        "base": {"root_state": numeric([0.0] * 13)},
        "racket": {
            "position_env_m": numeric([0.1, 0.2, 0.3]),
            "linear_velocity_world_mps": numeric([1.0, 2.0, 3.0]),
            "face_normal_signed_pre_orient_world": numeric([0.0, -1.0, 0.0]),
            "face_normal_raw_plus_y_world": numeric([0.0, 1.0, 0.0]),
            "analytic_face_normal_oriented_world": numeric([0.0, 1.0, 0.0]),
        },
        "incoming_ball": {
            "linear_velocity_world_mps": numeric([-2.0, 0.1, 0.2]),
            "spin_world_radps": numeric([0.0, 3.0, 0.0]),
        },
    }


def cell_document(prereg, engine, instrument, question_order, *, virtual_only=False):
    physical = instrument == "physical_truth"
    capability = (
        "physical_paddle_contact_and_post_contact_flight_v1"
        if physical
        else "analytic_counterfactual_contact_and_flight_v1"
    )
    outcome = (
        {
            "available": True,
            "capability": capability,
            "contacted": True,
            "net_clear": True,
            "landed_ok": True,
            "returned": True,
        }
        if physical
        else {
            "available": True,
            "capability": capability,
            "capture_gate": True,
            "net_clear": True,
            "on_opponent": True,
            "returned": True,
        }
    )
    return {
        "schema": "hope.cross-engine-instrument-cell.v1",
        "status": "complete",
        "engine": engine,
        "instrument": instrument,
        "instrument_capability": capability,
        "analytic_only": virtual_only if physical else True,
        "schedule_file_sha256": prereg["schedule"]["file_sha256"],
        "schedule_semantic_sha256": prereg["schedule"]["semantic_sha256"],
        "question_id_order_sha256": prereg["schedule"]["question_id_order_sha256"],
        "question_id_order": question_order,
        "checkpoint_sha256": prereg["target"]["checkpoint_sha256"],
        "training_contract_sha256": prereg["target"]["training_contract_sha256"],
        "exam_bank_sha256": prereg["target"]["exam_bank_sha256"],
        "evaluation_contract_exact": True,
        "fresh_lineage": True,
        "censored_attempts": 0,
        "numeric_ready_state": {
            "schema": "hope.cross-engine-state-instrumentation.v1",
            "sha256": ("1" if engine == "isaac" else "2") * 64,
            "root_state": numeric([0.0] * 13),
        },
        "attempts": [
            {
                "schedule_index": index,
                "question_id": question_id,
                "censored": False,
                "instrumentation": state_document(),
                "outcome": outcome,
            }
            for index, question_id in enumerate(question_order)
        ],
    }


def synthetic_evidence(tmp_path: Path, validator, *, omit=None, virtual_only=None):
    prereg = json.loads(PREREG.read_text())
    for name in ("validator", "isaac_evaluator", "isaac_adapter"):
        spec = prereg["tools"][name]
        spec["sha256"] = sha256(ROOT / spec["path"])
    prereg["target"]["actor_leg_ref_mask_provenance_epoch"] = 1
    prereg["target"]["actor_leg_ref_mask"] = False
    order = [f"forehand:{index:064x}" if index % 2 == 0 else f"backhand:{index:064x}" for index in range(100)]
    prereg["schedule"]["question_id_order_sha256"] = validator.canonical_sha256(order)
    config = tmp_path / "prereg.json"
    write_json(config, prereg)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    specs = []
    for key in sorted(validator.REQUIRED_CELLS):
        if key == omit:
            continue
        engine, instrument = key.split(":")
        document = cell_document(
            prereg,
            engine,
            instrument,
            order,
            virtual_only=(key == virtual_only),
        )
        path = artifact_root / f"{engine}_{instrument}.json"
        write_json(path, document)
        specs.append(
            {
                "engine": engine,
                "instrument": instrument,
                "artifact": {"path": path.name, "sha256": sha256(path)},
            }
        )
    evidence = tmp_path / "evidence.json"
    write_json(
        evidence,
        {
            "schema": "hope.cross-engine-instrument-parity-evidence.v1",
            "schema_version": 1,
            "preregistration_sha256": sha256(config),
            "cells": specs,
        },
    )
    return config, evidence, artifact_root


def test_numeric_isaac_state_extension_preserves_signed_pre_orient_normal():
    adapter = load_module(ADAPTER_PATH, "isaac_bank_exam_adapter_instrumentation")
    ready = adapter.numeric_ready_state_document(
        np.zeros(13), np.zeros(3), np.zeros(3), joint_names=["a", "b", "c"]
    )
    assert ready["legacy_ready_state_sha256"] == adapter.ready_state_sha256(
        np.zeros(13), np.zeros(3), np.zeros(3)
    )
    state = adapter.strike_state_instrumentation_document(
        observation_phase="exact_strike",
        base_root_state_env=np.zeros(13),
        racket_pos_env=np.zeros(3),
        racket_lin_vel_world=np.ones(3),
        racket_face_normal_signed_pre_orient_world=[0, -1, 0],
        racket_face_normal_raw_plus_y_world=[0, 1, 0],
        analytic_face_normal_oriented_world=[0, 1, 0],
        target_racket_pos_env=np.zeros(3),
        target_racket_lin_vel_world=np.zeros(3),
        target_face_normal_world=[0, 1, 0],
        incoming_ball_lin_vel_world=[-2, 0, 0],
        incoming_ball_spin_world=np.zeros(3),
        analytic_available=True,
        analytic_capture_gate=True,
        analytic_net_clear=True,
        analytic_on_opponent=True,
        analytic_landing_valid=True,
        analytic_landing_xy_env=[2.5, 0.0],
        physical_truth={
            "available": False,
            "capability": "incoming_flight_only_no_paddle_contact_phase_a",
            "reason": "no racket impulse",
        },
    )
    assert state["racket"]["face_normal_signed_pre_orient_world"]["values"] == [0.0, -1.0, 0.0]
    assert state["racket"]["analytic_face_normal_oriented_world"]["values"] == [0.0, 1.0, 0.0]
    assert state["physical_truth"]["available"] is False


def test_existing_scorecard_keeps_nested_instrumentation_json_only(tmp_path):
    adapter = load_module(ADAPTER_PATH, "isaac_bank_exam_adapter_scorecard_extension")
    schedule = [
        SimpleNamespace(
            schedule_index=0,
            clip=0,
            bank_row=7,
            question_id="forehand:" + "a" * 64,
            repeat=0,
            hold_steps=4,
            attempt_seed=9,
        )
    ]
    row = {
        "schedule_index": 0,
        "env_id": 0,
        "clip": 0,
        "side": "forehand",
        "bank_row": 7,
        "question_id": schedule[0].question_id,
        "repeat": 0,
        "hold_steps": 4,
        "attempt_seed": 9,
        "finalized": True,
        "censored": False,
        "physical_fall": False,
        "guard_reset": False,
        "reached_exact": True,
        "hit": True,
        "returned": True,
        "instrumentation": state_document(),
    }
    output_json = tmp_path / "scorecard.json"
    output_csv = tmp_path / "scorecard.csv"
    adapter.write_scorecard(
        output_json=output_json,
        output_csv=output_csv,
        metadata={"cross_engine_instrumentation": {"schema": adapter.CROSS_ENGINE_INSTRUMENTATION_SCHEMA}},
        records=[row],
        schedule=schedule,
        clip_names=["forehand"],
    )
    written = json.loads(output_json.read_text())
    assert written["schema"] == "hope.isaac-bank-exam.v1"
    assert written["attempts"][0]["instrumentation"]["schema"] == (
        "hope.cross-engine-state-instrumentation.v1"
    )
    assert "instrumentation" not in output_csv.read_text().splitlines()[0].split(",")


def test_checked_in_preregistration_is_frozen_and_explicitly_revoked():
    validator = load_module(VALIDATOR_PATH, "instrument_parity_validator_prereg")
    assert sha256(PREREG) == (
        "bd90f6f28ba578f452fe63184c7a0cafb4d8c511d478c3753c46b3bd58ba0175"
    )
    prereg = json.loads(PREREG.read_text())
    for spec in (prereg["forensic_input"], *prereg["tools"].values()):
        result = subprocess.run(
            ["git", "show", f"612f54d:{spec['path']}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert hashlib.sha256(result.stdout).hexdigest() == spec["sha256"]
    with pytest.raises(validator.ParityContractError, match="revoked for the current exact lane"):
        validator.validate_prereg(PREREG, ROOT)
    result = validator.validate_revocation(REVOCATION, ROOT)
    assert result["status"] == "revoked_for_current_exact_lane"
    assert result["instrument_parity_gate_closed"] is False
    assert result["preregistration_sha256"] == sha256(PREREG)
    assert set(result["formal_recovery_requirements"]) == {
        "post_epoch_checkpoint",
        "new_preregistration",
        "new_phase_b_contract",
        "rerun_all_four_cells",
    }


def test_evidence_missing_one_cell_fails_closed(tmp_path):
    validator = load_module(VALIDATOR_PATH, "instrument_parity_validator_missing")
    config, evidence, root = synthetic_evidence(
        tmp_path, validator, omit="isaac:physical_truth"
    )
    with pytest.raises(validator.ParityContractError, match="evidence incomplete"):
        validator.validate_evidence(config, evidence, root, ROOT)


def test_virtual_only_physical_cell_fails_closed(tmp_path):
    validator = load_module(VALIDATOR_PATH, "instrument_parity_validator_virtual")
    config, evidence, root = synthetic_evidence(
        tmp_path, validator, virtual_only="isaac:physical_truth"
    )
    with pytest.raises(validator.ParityContractError, match="virtual-only"):
        validator.validate_evidence(config, evidence, root, ROOT)


def test_complete_physical_and_analytic_2x2_is_the_only_close_path(tmp_path):
    validator = load_module(VALIDATOR_PATH, "instrument_parity_validator_complete")
    config, evidence, root = synthetic_evidence(tmp_path, validator)
    result = validator.validate_evidence(config, evidence, root, ROOT)
    assert result["status"] == "complete_four_cell_instrument_parity_evidence"
    assert result["instrument_parity_gate_closed"] is True
    assert len(result["accepted_cells"]) == 4
