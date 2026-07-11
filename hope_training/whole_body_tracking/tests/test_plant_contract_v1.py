"""Dependency-light tests for the semantics-correct plant-contract compiler."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/plant_contract.py"
)
CLI = REPO / "scripts/compile_semantics_correct_plant_contract.py"
SPEC = importlib.util.spec_from_file_location("plant_contract_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PLANT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANT)


JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def _sha(character: str) -> str:
    return character * 64


def _support(*, load=(0.0, 220.0), speed=(0.0, 20.0), temperature=(15.0, 40.0)):
    poses = ["ready_stand", "forehand_contact", "backhand_contact"]
    return {
        "load_abs_Nm": list(load),
        "speed_abs_rad_s": list(speed),
        "temperature_C": list(temperature),
        "pose_ids": poses,
        "pose_ids_sha256": PLANT.canonical_sha256(poses),
    }


def _adapter(engine: str):
    common = {
        "engine": engine,
        "engine_version": "IsaacLab-2.1.0_PhysX-5.3.1"
        if engine == "physx"
        else "MuJoCo-3.x-pinned-by-solver-contract",
        "runtime_target": "isaac_training_and_companion_eval"
        if engine == "physx"
        else "agibot_vendor_mujoco_gate3_gate3b",
        "source_parameter_origin": "engine_specific_fit_to_shared_latent_model",
        "latent_model_sha256": _sha("1"),
        "threshold_contract_sha256": _sha("2"),
        "adapter_source_sha256": _sha("3") if engine == "physx" else _sha("4"),
        "fit_report_sha256": _sha("5") if engine == "physx" else _sha("6"),
        "runtime_probe_report_sha256": _sha("7") if engine == "physx" else _sha("8"),
        "runtime_probe_passed": True,
        "runtime_source_sha256": _sha("c") if engine == "physx" else _sha("d"),
        "runtime_instantiation_report_sha256": _sha("e")
        if engine == "physx"
        else _sha("f"),
        "probe_schedule_sha256": _sha("9"),
        "asset_sha256": _sha("a") if engine == "physx" else _sha("b"),
        "solver_contract_sha256": _sha("c") if engine == "physx" else _sha("d"),
        "physics_step_dt_s": 0.005,
        "policy_step_dt_s": 0.02,
        "control_decimation": 4,
        "integrator": "physx_tgs_pinned" if engine == "physx" else "implicitfast_pinned",
    }
    if engine == "physx":
        return {
            **common,
            "backend": "native_transmitted_force_coefficient",
            "parameter_semantics": "load_dependent_spatial_force_coefficient",
            "parameters": {
                "friction_coefficient": {
                    "units": "dimensionless",
                    "values": [0.015 + index * 0.001 for index in range(31)],
                }
            },
        }
    return {
        **common,
        "vendor_mjcf_path": "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/"
        "model/a3_pingpong/a3_pingpong.xml",
        "backend": "native_frictionloss_plus_damping",
        "parameter_semantics": "load_independent_coulomb_bound_plus_viscous",
        "parameters": {
            "frictionloss": {"units": "N*m", "values": [0.2] * 31},
            "damping": {"units": "N*m*s/rad", "values": [0.05] * 31},
        },
    }


def _draft():
    support = _support()
    return {
        "schema_version": 1,
        "contract_id": "fresh_SC_calibrated_v1",
        "status": "ready_for_semantics_correct_runtime",
        "lineage_role": "fresh_semantics_correct_calibrated_plant",
        "hardware_commands_authorized": False,
        "legacy_direct_number_proxy": False,
        "joint_order": {
            "names": JOINTS,
            "sha256": PLANT.canonical_sha256(JOINTS),
        },
        "physical_model": {
            "family": "load_affine_breakaway_plus_coulomb_plus_viscous",
            "units": {
                "load": "N*m",
                "speed": "rad/s",
                "generalized_torque": "N*m",
                "viscous": "N*m*s/rad",
                "temperature": "degC",
            },
            "latent_model_sha256": _sha("1"),
            "source_dataset_manifest_sha256": _sha("e"),
            "session_split_sha256": _sha("f"),
            "repeatability_report_sha256": _sha("0"),
            "threshold_contract_sha256": _sha("2"),
            "selection_report_sha256": _sha("a"),
            "support_envelope_sha256": PLANT.canonical_sha256(support),
            "support": support,
        },
        "cross_engine": {
            "probe_schedule_sha256": _sha("9"),
            "threshold_contract_sha256": _sha("2"),
            "equivalence_report_sha256": _sha("b"),
            "equivalence_passed": True,
            "same_latent_model_required": True,
            "parameter_equality_is_acceptance": False,
        },
        "adapters": {
            "physx": _adapter("physx"),
            "mujoco": _adapter("mujoco"),
        },
    }


def _contract():
    return PLANT.bind_contract_sha256(_draft())


def _rebind(value):
    return PLANT.bind_contract_sha256(value)


def test_ready_contract_is_content_addressed_and_compiles_both_engines():
    contract = _contract()
    normalized = PLANT.validate_plant_contract(contract)
    assert normalized["contract_sha256"] == PLANT.contract_payload_sha256(contract)
    assert normalized["joint_names"] == JOINTS

    request = _support(load=(10.0, 180.0), speed=(0.0, 12.0), temperature=(20.0, 35.0))
    physx = PLANT.prepare_runtime_adapter(
        contract, engine="physx", requested_support=request
    )
    mujoco = PLANT.prepare_runtime_adapter(
        contract, engine="mujoco", requested_support=request
    )
    assert physx["parameters"]["friction_coefficient"]["units"] == "dimensionless"
    assert mujoco["parameters"]["frictionloss"]["units"] == "N*m"
    assert mujoco["parameters"]["damping"]["units"] == "N*m*s/rad"
    assert mujoco["runtime_target"] == "agibot_vendor_mujoco_gate3_gate3b"
    assert mujoco["vendor_mjcf_path"].endswith("a3_pingpong/a3_pingpong.xml")
    assert physx["runtime_adapter_sha256"] != mujoco["runtime_adapter_sha256"]
    assert physx["hardware_commands_authorized"] is False


def test_nonzero_cross_unit_direct_number_copy_is_impossible_but_zero_is_common():
    with pytest.raises(PLANT.PlantContractError, match="no non-zero numeric conversion"):
        PLANT.zero_only_unit_conversion(
            [1.1971, 2.4276], source_units="N*m", target_units="dimensionless"
        )
    assert PLANT.zero_only_unit_conversion(
        [0.0, -0.0], source_units="dimensionless", target_units="N*m"
    ) == [0.0, 0.0]
    assert PLANT.zero_only_unit_conversion(
        [1.0, 2.0], source_units="N*m", target_units="N*m"
    ) == [1.0, 2.0]


def test_physx_and_mujoco_units_are_not_swappable():
    data = _draft()
    data["adapters"]["physx"]["parameters"]["friction_coefficient"]["units"] = "N*m"
    with pytest.raises(PLANT.PlantContractError, match="dimensionless"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["adapters"]["mujoco"]["parameters"]["frictionloss"]["units"] = "dimensionless"
    with pytest.raises(PLANT.PlantContractError, match="must be 'N\\*m'"):
        PLANT.validate_plant_contract(_rebind(data))


def test_legacy_proxy_and_unpassed_evidence_fail_closed():
    data = _draft()
    data["legacy_direct_number_proxy"] = True
    with pytest.raises(PLANT.PlantContractError, match="legacy direct-number"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["cross_engine"]["equivalence_passed"] = False
    with pytest.raises(PLANT.PlantContractError, match="has not passed"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["adapters"]["physx"]["runtime_probe_report_sha256"] = None
    with pytest.raises(PLANT.PlantContractError, match="lowercase SHA-256"):
        PLANT.validate_plant_contract(_rebind(data))


def test_engine_adapters_require_independent_fit_reports_and_shared_latent_model():
    data = _draft()
    data["adapters"]["mujoco"]["fit_report_sha256"] = data["adapters"]["physx"][
        "fit_report_sha256"
    ]
    with pytest.raises(PLANT.PlantContractError, match="distinct engine-specific fit reports"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["adapters"]["mujoco"]["vendor_mjcf_path"] = "generic.xml"
    with pytest.raises(PLANT.PlantContractError, match="Agibot Gate3/Gate3B asset"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["adapters"]["mujoco"]["latent_model_sha256"] = _sha("3")
    with pytest.raises(PLANT.PlantContractError, match="latent_model_sha256 drifted"):
        PLANT.validate_plant_contract(_rebind(data))


def test_digest_joint_order_nonfinite_and_unknown_keys_fail_closed():
    contract = _contract()
    contract["adapters"]["physx"]["parameters"]["friction_coefficient"]["values"][0] += 1.0
    with pytest.raises(PLANT.PlantContractError, match="contract_sha256"):
        PLANT.validate_plant_contract(contract)

    data = _draft()
    data["joint_order"]["names"][0], data["joint_order"]["names"][1] = (
        data["joint_order"]["names"][1],
        data["joint_order"]["names"][0],
    )
    with pytest.raises(PLANT.PlantContractError, match="does not bind names"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["adapters"]["mujoco"]["parameters"]["damping"]["values"][5] = float("nan")
    with pytest.raises(PLANT.PlantContractError, match="NaN/Inf"):
        PLANT.validate_plant_contract(_rebind(data))

    data = _draft()
    data["adapters"]["physx"]["copied_from_mujoco"] = True
    with pytest.raises(PLANT.PlantContractError, match="unknown=.*copied_from_mujoco"):
        PLANT.validate_plant_contract(_rebind(data))


def test_requested_support_must_be_subset_of_calibration_envelope():
    contract = _contract()
    request = _support(load=(0.0, 221.0))
    with pytest.raises(PLANT.PlantContractError, match="outside calibrated support"):
        PLANT.prepare_runtime_adapter(
            contract, engine="physx", requested_support=request
        )

    request = _support()
    request["pose_ids"] = ["unknown_pose"]
    request["pose_ids_sha256"] = PLANT.canonical_sha256(request["pose_ids"])
    with pytest.raises(PLANT.PlantContractError, match="out-of-support pose"):
        PLANT.prepare_runtime_adapter(
            contract, engine="mujoco", requested_support=request
        )


def test_cli_binds_verifies_and_prepares_without_launching(tmp_path):
    draft = tmp_path / "draft.json"
    bound = tmp_path / "bound.json"
    support = tmp_path / "support.json"
    runtime = tmp_path / "runtime.json"
    draft.write_text(json.dumps(_draft()), encoding="utf-8")
    support.write_text(json.dumps(_support(load=(5.0, 100.0))), encoding="utf-8")

    bind = subprocess.run(
        [sys.executable, str(CLI), "bind", str(draft), str(bound)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bind.returncode == 0, bind.stdout + bind.stderr
    assert "PLANT_CONTRACT_BIND_OK" in bind.stdout
    assert "hardware_commands_authorized=false" in bind.stdout

    verify = subprocess.run(
        [sys.executable, str(CLI), "verify", str(bound)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "PLANT_CONTRACT_VERIFY_OK" in verify.stdout

    prepare = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "prepare",
            str(bound),
            "--engine",
            "mujoco",
            "--requested-support",
            str(support),
            "--output",
            str(runtime),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stdout + prepare.stderr
    assert "PLANT_RUNTIME_ADAPTER_PREPARE_OK" in prepare.stdout
    prepared = json.loads(runtime.read_text(encoding="utf-8"))
    assert prepared["engine"] == "mujoco"
    assert prepared["hardware_commands_authorized"] is False
