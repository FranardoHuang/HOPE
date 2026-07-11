from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_phase1_plant_semantics_prereg.py"
MANIFEST = ROOT / "configs" / "phase1_plant_semantics_repair_prereg_20260711.json"

SPEC = importlib.util.spec_from_file_location("plant_semantics_prereg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_tracked_preregistration_and_repository_baseline_pass():
    data = _manifest()
    MODULE.validate_manifest(data)
    MODULE.verify_repository_baseline(data, ROOT)


def test_cli_reports_blocked_contract_without_launching_anything():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(MANIFEST), "--verify-repository-baseline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PLANT_SEMANTICS_PREREG_OK" in result.stdout
    assert "status=blocked_on_calibration_evidence" in result.stdout
    assert "minimum_training_arms=4" in result.stdout
    assert "evaluations_per_milestone=16" in result.stdout
    assert "hardware_commands_authorized=false" in result.stdout


@pytest.mark.parametrize("cell", ["SZ", "SP", "LZ", "LP"])
def test_no_current_cell_can_be_relabelled_calibrated_or_deployable(cell):
    data = _manifest()
    data["current_phase1_cells"][cell]["plant_semantics_calibrated"] = True
    with pytest.raises(MODULE.PlantPreregError, match="cannot be labelled calibrated"):
        MODULE.validate_manifest(data)

    data = _manifest()
    data["current_phase1_cells"][cell]["deployment_candidate"] = True
    with pytest.raises(MODULE.PlantPreregError, match="deployment candidate"):
        MODULE.validate_manifest(data)


def test_sp_cannot_masquerade_as_calibrated_control():
    data = _manifest()
    data["current_phase1_cells"]["SP"]["plant"] = "calibrated_friction"
    with pytest.raises(MODULE.PlantPreregError, match="SP plant semantics changed"):
        MODULE.validate_manifest(data)


def test_direct_numeric_mapping_cannot_be_enabled():
    data = _manifest()
    data["source_semantics"]["mapping_rule"]["same_number_allowed"] = True
    with pytest.raises(MODULE.PlantPreregError, match="same numeric"):
        MODULE.validate_manifest(data)


def test_legacy_directional_probe_cannot_gain_invented_raw_provenance():
    data = _manifest()
    data["legacy_frozen_probe"]["raw_artifact_sha256"] = "a" * 64
    with pytest.raises(MODULE.PlantPreregError, match="invented SHA"):
        MODULE.validate_manifest(data)


def test_ready_status_requires_every_evidence_binding():
    data = _manifest()
    data["status"] = "ready_for_semantics_correct_launch"
    data["evidence_bindings"] = {
        "measurement_protocol_sha256": "a" * 64,
    }
    with pytest.raises(MODULE.PlantPreregError, match="missing evidence bindings"):
        MODULE.validate_manifest(data)


def test_evidence_binding_must_be_lowercase_sha256():
    data = _manifest()
    data["evidence_bindings"] = {"measurement_protocol_sha256": "NOT-A-SHA"}
    with pytest.raises(MODULE.PlantPreregError, match="lowercase SHA-256"):
        MODULE.validate_manifest(data)


def test_minimum_axis_requires_two_paired_seeds_and_recomputed_counts():
    data = _manifest()
    data["minimum_training_axis"]["paired_seed_blocks"] = [1]
    data["minimum_training_axis"]["minimum_from_scratch_training_arms"] = 2
    data["minimum_training_axis"]["evaluations_per_milestone"] = 8
    with pytest.raises(MODULE.PlantPreregError, match="at least two"):
        MODULE.validate_manifest(data)

    data = _manifest()
    data["minimum_training_axis"]["evaluations_per_milestone"] = 8
    with pytest.raises(MODULE.PlantPreregError, match="evaluations_per_milestone must be 16"):
        MODULE.validate_manifest(data)


def test_q10_cannot_become_a_stop_or_promotion_rule():
    data = _manifest()
    data["checkpoint_decision_contract"]["q10_may_stop_or_promote"] = True
    with pytest.raises(MODULE.PlantPreregError, match="q10 must remain screen-only"):
        MODULE.validate_manifest(data)


def test_both_engine_legs_are_mandatory():
    data = _manifest()
    data["minimum_training_axis"]["evaluation_factors"]["engine_levels"] = ["isaac"]
    with pytest.raises(MODULE.PlantPreregError, match="both Isaac and MuJoCo"):
        MODULE.validate_manifest(data)


def test_baseline_sha_drift_fails_loud(tmp_path):
    data = copy.deepcopy(_manifest())
    source = tmp_path / "source.py"
    source.write_text("changed\n", encoding="utf-8")
    data["repository_baseline"]["audited_sources"] = [
        {"path": "source.py", "sha256": "0" * 64}
    ]
    with pytest.raises(MODULE.PlantPreregError, match="SHA mismatch"):
        MODULE.verify_repository_baseline(data, tmp_path)
