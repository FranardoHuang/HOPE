"""Dependency-light contract tests for the A3 dynamic-ready producer.

The exact MuJoCo/HiGHS solve and Isaac closed-loop hold remain Pod integration
tests.  These tests protect the easy-to-regress order conversion around that
solve: the LP speaks MuJoCo post-root DoF order, while the training contract,
PD gains, qdes, and actor action all speak the A3 runtime joint order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "materialize_a3_dynamic_ready_contract",
    _SCRIPTS / "materialize_a3_dynamic_ready_contract.py",
)
materializer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = materializer
_SPEC.loader.exec_module(materializer)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_measured_identity_clip(
    path: Path, *, uid: str = "Take_061_unit04_BH", frames: int = 3
) -> Path:
    np.savez(
        path,
        joint_pos=np.zeros((frames, 31), dtype=np.float32),
        measured_racket_uid=np.asarray(uid),
        measured_racket_schema_version=np.asarray([4], dtype=np.int64),
        measured_racket_retarget_admitted=np.asarray([1], dtype=np.int64),
        measured_racket_joint_order_contract_id=np.asarray(
            materializer.EXPECTED_MEASURED_JOINT_ORDER_CONTRACT_ID
        ),
        measured_racket_joint_order_contract_sha256=np.asarray(
            materializer.EXPECTED_MEASURED_JOINT_ORDER_CONTRACT_SHA256
        ),
    )
    return path


def _measured_evidence_documents(
    *, uid: str, motion_sha: str, frames: int, bank_sha: str
) -> tuple[dict, dict]:
    bank = {
        "schema_version": 1,
        "kind": materializer.MEASURED_BANK_RECEIPT_KIND,
        "authorization": {
            "diagnostic_unauthorized": True,
            "training": False,
            "promotion": False,
            "deployment": False,
            "mechanical_admission": False,
        },
        "denominators": {"materialized_npz": 1},
        "actions": [
            {"uid": uid, "sha256": motion_sha, "frames": frames}
        ],
    }
    mechanical = {
        "schema_version": 1,
        "kind": materializer.MEASURED_MECHANICAL_AUDIT_KIND,
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
            "mechanical_admission": False,
        },
        "sources": {
            "bank_import_receipt": {
                "sha256": bank_sha,
                "kind": materializer.MEASURED_BANK_RECEIPT_KIND,
            }
        },
        "denominators": {"actions_expected": 1, "actions_audited": 1},
        "actions": [
            {
                "uid": uid,
                "sha256": motion_sha,
                "kinematic_limit_verdict": "PASS",
                "mechanical_verdict": "UNKNOWN",
                "mechanical_admitted": False,
            }
        ],
    }
    return bank, mechanical


def test_motion_frame0_normalizes_float32_quaternion_and_rejects_zero(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.npz"
    root_quat = np.asarray([0.92387956, 0.0, 0.38268343, 0.0], np.float32)
    np.savez(
        clip,
        joint_pos=np.zeros((2, 31), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=np.broadcast_to(root_quat, (2, 1, 4)).copy(),
    )
    _joint_pos, _root_pos, loaded_quat = materializer._load_motion_frame0(clip)
    assert np.linalg.norm(loaded_quat) == pytest.approx(1.0, abs=1.0e-15)
    root_quat_f64 = root_quat.astype(np.float64)
    expected_direction = root_quat_f64 / np.linalg.norm(root_quat_f64)
    assert loaded_quat == pytest.approx(expected_direction, abs=1.0e-15)

    zero_clip = tmp_path / "zero_clip.npz"
    np.savez(
        zero_clip,
        joint_pos=np.zeros((2, 31), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=np.zeros((2, 1, 4), dtype=np.float32),
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="root quaternion is degenerate",
    ):
        materializer._load_motion_frame0(zero_clip)


def test_materialize_scatter_gathers_runtime_and_mujoco_force_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 31
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    assert len(names) == count

    # Deliberately non-identity.  Value[runtime_row] is the corresponding
    # MuJoCo post-root row, matching ExactMujocoModelBinding.joint_dof_adrs-6.
    model_row_for_runtime = np.asarray(
        [
            19,
            25,
            0,
            20,
            26,
            1,
            21,
            27,
            2,
            22,
            28,
            3,
            5,
            12,
            23,
            29,
            4,
            6,
            13,
            24,
            30,
            7,
            14,
            8,
            15,
            9,
            16,
            10,
            17,
            11,
            18,
        ],
        np.int64,
    )
    assert np.array_equal(np.sort(model_row_for_runtime), np.arange(count))

    ready_q = np.zeros(count, np.float64)
    kp = np.arange(1.0, count + 1.0, dtype=np.float64)
    tau_runtime = 0.01 * np.arange(1.0, count + 1.0, dtype=np.float64)
    tau_model = np.empty(count, np.float64)
    tau_model[model_row_for_runtime] = tau_runtime
    runtime_lower = -kp
    runtime_upper = kp
    expected_lower_model = np.empty(count, np.float64)
    expected_upper_model = np.empty(count, np.float64)
    expected_lower_model[model_row_for_runtime] = runtime_lower
    expected_upper_model[model_row_for_runtime] = runtime_upper

    plant = {
        "joint_names": names,
        "kp": kp,
        "kd": np.ones(count, np.float64),
        "effort": np.full(count, 1000.0, np.float64),
        "velocity": np.full(count, 20.0, np.float64),
        "default_q": np.zeros(count, np.float64),
        "action_scale": np.ones(count, np.float64),
        "qdes_limits": np.column_stack(
            (-np.ones(count, np.float64), np.ones(count, np.float64))
        ),
        "physx_control_position_limits": {
            "schema_version": 1,
            "backend": "physx_root_view_dof_limits",
            "inset_fraction_per_side_hard_span": 0.02,
            "selected_joint_names": [
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ],
            "mechanical_joint_pos_limits": [[-2.0, 2.0]] * 31,
            "control_joint_pos_limits": [
                [-1.92, 1.92]
                if index in (5, 8, 19, 20)
                else [-2.0, 2.0]
                for index in range(31)
            ],
            "unselected_joint_count": 27,
            "unselected_limits_equal_mechanical": True,
            "articulation_mechanical_ledger_unchanged": True,
            "soft_qdes_ledger_unchanged": True,
        },
        "projection_inset": 0.0,
        "physics_dt": 0.005,
        "policy_dt": 0.02,
        "decimation": 4,
        "control_step_action_delay": {
            "schema_version": 1,
            "enabled": True,
            "semantic_unit": "policy_control_step",
            "sample_timing": "once_per_episode_reset",
            "distribution": "discrete_uniform_inclusive",
            "min_steps": 0,
            "max_steps": 2,
            "shared_across_all_31_joints": True,
            "history_fill": "safe_default_or_action_specific_hold",
        },
        "actuator_types": ["implicit"] * count,
        "armature": np.zeros(count, np.float64),
        "friction": np.zeros(count, np.float64),
        "friction_backend": "physx",
        "friction_semantics": "load_dependent_spatial_force_coefficient",
        "friction_units": "dimensionless",
    }
    runtime_contract = {
        "target_mode": "action_ball",
        "action_ball_training": {
            "preflight": {"ready_root_z_by_slot_m": [1.0]}
        },
    }
    stable_receipt = {
        "robot": {"exact_xml_model_name": "A3-test"},
    }
    identity = SimpleNamespace(
        ground_model_binding_sha256="a" * 64,
        compiled_model_sha256="b" * 64,
        path_model_binding_sha256="c" * 64,
        xml_model_name="A3-test",
    )
    backend = SimpleNamespace(
        model=SimpleNamespace(nv=37),
        _binding=SimpleNamespace(
            joint_dof_adrs=model_row_for_runtime + 6
        ),
        _qpos=lambda _ready: np.zeros(38, np.float64),
    )
    captured: dict[str, np.ndarray] = {}

    class FakeSolver:
        def __init__(self, _model: object, _config: object) -> None:
            pass

        def solve(
            self,
            _qpos: np.ndarray,
            _qvel: np.ndarray,
            _qacc: np.ndarray,
            _actuated: np.ndarray,
            effort_lower: np.ndarray,
            effort_upper: np.ndarray,
            _velocity: np.ndarray,
            **_kwargs: object,
        ) -> SimpleNamespace:
            captured["lower"] = np.asarray(effort_lower).copy()
            captured["upper"] = np.asarray(effort_upper).copy()
            return SimpleNamespace(
                feasible=True,
                actuator_generalized_force=tau_model.copy(),
                report={
                    "model_binding": "a" * 64,
                    "optimum_max_normalized_available_hold_torque": 0.01,
                },
            )

    paths = {
        "motion": Path("/tmp/motion.npz"),
        "receipt": Path("/tmp/receipt.json"),
        "runtime": Path("/tmp/training_contract.json"),
        "mjcf": Path("/tmp/a3.xml"),
    }
    shas = {
        "motion": "1" * 64,
        "receipt": "2" * 64,
        "runtime": "3" * 64,
        "mjcf": "4" * 64,
    }

    def fake_pinned(
        path_value: str, _expected: object, *, name: str
    ) -> tuple[Path, str]:
        key = {
            "stable motion": "motion",
            "stable receipt": "receipt",
            "runtime training contract": "runtime",
            "A3 MJCF": "mjcf",
        }[name]
        assert path_value == key
        return paths[key], shas[key]

    def fake_read_json(_path: Path, *, name: str) -> dict:
        if name == "stable receipt":
            return stable_receipt
        assert name == "runtime training contract"
        return runtime_contract

    monkeypatch.setattr(materializer, "_pinned_file", fake_pinned)
    monkeypatch.setattr(materializer, "_read_json", fake_read_json)
    monkeypatch.setattr(
        materializer, "_validate_stable_receipt", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        materializer,
        "_load_motion_frame0",
        lambda _path: (
            ready_q.copy(),
            np.asarray([0.0, 0.0, 1.0], np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], np.float64),
        ),
    )
    monkeypatch.setattr(materializer, "_runtime_plant", lambda _c: plant)
    monkeypatch.setattr(
        materializer,
        "_bind_action_runtime",
        lambda *_a, **_k: (
            -2.0 * np.ones(count),
            2.0 * np.ones(count),
            ready_q.copy(),
            plant["default_q"].copy(),
            plant["action_scale"].copy(),
        ),
    )
    monkeypatch.setattr(
        materializer, "_exact_model_identity", lambda *_a, **_k: identity
    )
    monkeypatch.setattr(
        materializer.grounded.MujocoGroundedReadyBackend,
        "load",
        lambda _identity: backend,
    )
    monkeypatch.setattr(
        materializer.torque_topp,
        "GroundContactConfig",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        materializer.torque_topp, "MujocoGroundContactLPSolver", FakeSolver
    )
    monkeypatch.setattr(
        materializer.torque_topp,
        "direct_actuator_contract_from_mujoco",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        materializer.torque_topp,
        "_resolve_grounded_actuator_limits",
        lambda _contract, _nv: (
            np.full(count, -1000.0),
            np.full(count, 1000.0),
            np.arange(6, 37, dtype=np.int64),
            {"fixture": True},
        ),
    )
    monkeypatch.setattr(materializer, "_sha256_file", lambda _path: "f" * 64)

    args = argparse.Namespace(
        action_id="bh_block",
        motion="motion",
        expected_motion_sha256=shas["motion"],
        stable_receipt="receipt",
        expected_stable_receipt_sha256=shas["receipt"],
        runtime_contract="runtime",
        expected_runtime_contract_sha256=shas["runtime"],
        mjcf="mjcf",
        expected_mjcf_sha256=shas["mjcf"],
        output="unused",
    )
    result = materializer._materialize(args)

    assert captured["lower"] == pytest.approx(expected_lower_model)
    assert captured["upper"] == pytest.approx(expected_upper_model)
    hold = result["hold_candidate"]
    assert hold["mujoco_row_for_runtime_joint"] == (
        model_row_for_runtime.tolist()
    )
    assert hold["actuator_generalized_force_mujoco_row_order_nm"] == (
        pytest.approx(tau_model)
    )
    assert hold["actuator_generalized_force_runtime_order_nm"] == (
        pytest.approx(tau_runtime)
    )
    assert hold["hold_qdes_joint_pos_rad"] == pytest.approx(
        tau_runtime / kp
    )
    assert result["schema_version"] == 2
    assert result["kind"] == "agibot_a3_action_dynamic_ready_candidate_v2"
    runtime_plant = result["runtime_plant"]
    assert runtime_plant["joint_names"] == list(names)
    assert runtime_plant["articulation_joint_names"] == list(names)
    assert runtime_plant["action_joint_ids"] == list(range(31))
    assert runtime_plant["joint_velocity_limits"] == [20.0] * 31
    assert runtime_plant["control_step_action_delay"]["max_steps"] == 2
    assert runtime_plant["physx_control_position_limits"] == plant[
        "physx_control_position_limits"
    ]


def test_exclusive_writer_never_clobbers_existing_bytes(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    materializer._write_exclusive(output, b"first\n")
    with pytest.raises(FileExistsError):
        materializer._write_exclusive(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def test_physx_control_position_limits_are_exact_and_fail_loud() -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    qdes = np.asarray([[-1.0, 1.0]] * 31, np.float64)
    block = {
        "schema_version": 1,
        "backend": "physx_root_view_dof_limits",
        "inset_fraction_per_side_hard_span": 0.02,
        "selected_joint_names": [
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ],
        "mechanical_joint_pos_limits": [[-2.0, 2.0]] * 31,
        "control_joint_pos_limits": [
            [-1.92, 1.92] if index in (5, 8, 19, 20) else [-2.0, 2.0]
            for index in range(31)
        ],
        "unselected_joint_count": 27,
        "unselected_limits_equal_mechanical": True,
        "articulation_mechanical_ledger_unchanged": True,
        "soft_qdes_ledger_unchanged": True,
    }
    assert materializer._physx_control_position_limits(
        block, joint_names=names, qdes_limits=qdes
    ) == block

    for mutate, message in (
        (
            lambda value: value.update(
                selected_joint_names=["right_ankle_roll_joint"]
            ),
            "identity",
        ),
        (
            lambda value: value.update(unselected_joint_count=28),
            "identity",
        ),
        (
            lambda value: value.pop("soft_qdes_ledger_unchanged"),
            "fields",
        ),
        (
            lambda value: value["control_joint_pos_limits"][0].__setitem__(
                0, -1.99
            ),
            "unselected",
        ),
    ):
        changed = json.loads(json.dumps(block))
        mutate(changed)
        with pytest.raises(
            materializer.DynamicReadyMaterializationError, match=message
        ):
            materializer._physx_control_position_limits(
                changed, joint_names=names, qdes_limits=qdes
            )


def test_measured_retarget_evidence_is_cross_bound_and_unknown_is_explicit(
    tmp_path: Path,
) -> None:
    uid = "Take_061_unit04_BH"
    clip = _write_measured_identity_clip(tmp_path / "measured.npz", uid=uid)
    motion_sha = _sha256(clip)
    bank_sha = "b" * 64
    bank, mechanical = _measured_evidence_documents(
        uid=uid,
        motion_sha=motion_sha,
        frames=3,
        bank_sha=bank_sha,
    )

    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="mechanical verdict is UNKNOWN",
    ):
        materializer._validate_measured_retarget_l0_evidence(
            motion_path=clip,
            motion_sha256=motion_sha,
            measured_uid=uid,
            bank_receipt=bank,
            bank_receipt_sha256=bank_sha,
            mechanical_audit=mechanical,
            allow_mechanical_unknown=False,
        )

    evidence = materializer._validate_measured_retarget_l0_evidence(
        motion_path=clip,
        motion_sha256=motion_sha,
        measured_uid=uid,
        bank_receipt=bank,
        bank_receipt_sha256=bank_sha,
        mechanical_audit=mechanical,
        allow_mechanical_unknown=True,
    )
    assert evidence["ready_pose_semantics"] == (
        "exact_original_measured_motion_frame0_no_transplant"
    )
    assert evidence["unknown_explicitly_accepted_for_sim_diagnostic"] is True
    assert evidence["training_authorized"] is False

    changed = json.loads(json.dumps(bank))
    changed["authorization"]["training"] = True
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="diagnostic-only schema-v4 import",
    ):
        materializer._validate_measured_retarget_l0_evidence(
            motion_path=clip,
            motion_sha256=motion_sha,
            measured_uid=uid,
            bank_receipt=changed,
            bank_receipt_sha256=bank_sha,
            mechanical_audit=mechanical,
            allow_mechanical_unknown=True,
        )

    changed = json.loads(json.dumps(mechanical))
    changed["actions"][0]["sha256"] = "0" * 64
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="exact selected motion",
    ):
        materializer._validate_measured_retarget_l0_evidence(
            motion_path=clip,
            motion_sha256=motion_sha,
            measured_uid=uid,
            bank_receipt=bank,
            bank_receipt_sha256=bank_sha,
            mechanical_audit=changed,
            allow_mechanical_unknown=True,
        )


def test_measured_direct_frame0_branch_bypasses_action_runtime_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uid = "Take_061_unit04_BH"
    clip = _write_measured_identity_clip(tmp_path / "measured.npz", uid=uid)
    motion_sha = _sha256(clip)
    bank_path = tmp_path / "BANK_IMPORT_RECEIPT.json"
    mechanical_path = tmp_path / "mechanical.json"
    runtime_path = tmp_path / "runtime.json"
    mjcf_path = tmp_path / "a3.xml"
    bank_placeholder_sha = "b" * 64
    bank, mechanical = _measured_evidence_documents(
        uid=uid,
        motion_sha=motion_sha,
        frames=3,
        bank_sha=bank_placeholder_sha,
    )
    bank_path.write_text(json.dumps(bank))
    actual_bank_sha = _sha256(bank_path)
    mechanical["sources"]["bank_import_receipt"]["sha256"] = actual_bank_sha
    mechanical_path.write_text(json.dumps(mechanical))
    runtime = {
        "target_mode": "action_ball",
        "action_ball_training": {
            "authorization": {
                "diagnostic_unauthorized": True,
                "formal_evidence_prohibited": True,
                "curriculum_promotion_prohibited": True,
                "exact_export_prohibited": True,
                "formal_judge_prohibited": True,
            },
            "motion_admission": {
                "diagnostic_unauthorized": True,
                "training_authorized": False,
            },
        },
    }
    runtime_path.write_text(json.dumps(runtime))
    mjcf_path.write_text("<mujoco model='A3-test'/>")

    plant = {
        "joint_names": tuple(materializer.grounded.RUNTIME_JOINT_NAMES),
        "physx_control_position_limits": {
            "mechanical_joint_pos_limits": [[-2.0, 2.0]] * 31
        },
    }
    monkeypatch.setattr(materializer, "_runtime_plant", lambda _c: plant)
    monkeypatch.setattr(
        materializer,
        "_load_motion_frame0",
        lambda _path: (
            np.zeros(31, np.float64),
            np.asarray([0.0, 0.0, 1.0], np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], np.float64),
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_bind_action_runtime",
        lambda *_a, **_k: pytest.fail(
            "measured direct-frame0 must not consume action-bound bootstrap"
        ),
    )

    class ReachedModelIdentity(RuntimeError):
        pass

    monkeypatch.setattr(
        materializer,
        "_derive_exact_model_identity",
        lambda **_kwargs: (_ for _ in ()).throw(ReachedModelIdentity()),
    )
    args = argparse.Namespace(
        action_id="take_061_unit04_bh",
        ready_source_kind=materializer.MEASURED_RETARGET_SOURCE_KIND,
        motion=str(clip),
        expected_motion_sha256=motion_sha,
        measured_uid=uid,
        measured_bank_receipt=str(bank_path),
        expected_measured_bank_receipt_sha256=actual_bank_sha,
        mechanical_audit=str(mechanical_path),
        expected_mechanical_audit_sha256=_sha256(mechanical_path),
        allow_mechanical_unknown_diagnostic=True,
        stable_receipt=None,
        expected_stable_receipt_sha256=None,
        runtime_contract=str(runtime_path),
        expected_runtime_contract_sha256=_sha256(runtime_path),
        mjcf=str(mjcf_path),
        expected_mjcf_sha256=_sha256(mjcf_path),
        output="unused",
    )
    with pytest.raises(ReachedModelIdentity):
        materializer._materialize(args)


def test_measured_plant_template_remains_negatively_authorized() -> None:
    contract = {
        "target_mode": "action_ball",
        "action_ball_training": {
            "authorization": {
                "diagnostic_unauthorized": True,
                "formal_evidence_prohibited": True,
                "curriculum_promotion_prohibited": True,
                "exact_export_prohibited": True,
                "formal_judge_prohibited": True,
            },
            "motion_admission": {
                "diagnostic_unauthorized": True,
                "training_authorized": False,
            },
        },
    }
    materializer._validate_diagnostic_plant_template(contract)
    contract["action_ball_training"]["motion_admission"][
        "training_authorized"
    ] = True
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="training_authorized=false",
    ):
        materializer._validate_diagnostic_plant_template(contract)


def test_mechanical_limit_inner_guard_and_parser_source_defaults() -> None:
    plant = {
        "physx_control_position_limits": {
            "mechanical_joint_pos_limits": [[-2.0, 3.0]] * 31
        }
    }
    lower, upper = materializer._hard_inner_from_mechanical_limits(plant)
    assert lower == pytest.approx([-1.9] * 31)
    assert upper == pytest.approx([2.9] * 31)

    parser = materializer._parser()
    common = [
        "--action-id",
        "x",
        "--motion",
        "m",
        "--expected-motion-sha256",
        "0" * 64,
        "--runtime-contract",
        "r",
        "--expected-runtime-contract-sha256",
        "1" * 64,
        "--mjcf",
        "j",
        "--expected-mjcf-sha256",
        "2" * 64,
        "--output",
        "o",
    ]
    assert parser.parse_args(common).ready_source_kind == (
        materializer.STABLE_UPPER_SOURCE_KIND
    )
    measured = parser.parse_args(
        [
            *common,
            "--ready-source-kind",
            materializer.MEASURED_RETARGET_SOURCE_KIND,
            "--allow-mechanical-unknown-diagnostic",
        ]
    )
    assert measured.ready_source_kind == materializer.MEASURED_RETARGET_SOURCE_KIND
    assert measured.allow_mechanical_unknown_diagnostic is True
