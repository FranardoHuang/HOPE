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
import os
import stat
import sys
from dataclasses import replace
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
        measured_racket_site_pos_w=np.zeros((frames, 3), dtype=np.float32),
        measured_racket_normal_w=np.tile(
            np.asarray([0.0, 1.0, 0.0], np.float32), (frames, 1)
        ),
        measured_racket_long_axis_w=np.tile(
            np.asarray([1.0, 0.0, 0.0], np.float32), (frames, 1)
        ),
        measured_racket_uid=np.asarray(uid),
        measured_racket_schema_version=np.asarray([4], dtype=np.int64),
        measured_racket_position_semantics=np.asarray(
            materializer.EXPECTED_MEASURED_RACKET_POSITION_SEMANTICS
        ),
        measured_racket_normal_semantics=np.asarray(
            materializer.EXPECTED_MEASURED_RACKET_NORMAL_SEMANTICS
        ),
        measured_racket_long_axis_semantics=np.asarray(
            materializer.EXPECTED_MEASURED_RACKET_LONG_AXIS_SEMANTICS
        ),
        measured_racket_robot_mount_normal_sign=np.asarray(
            [1], dtype=np.int8
        ),
        measured_racket_robot_butt_to_blade_axis_local=np.asarray(
            materializer.EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
            np.float64,
        ),
        measured_racket_robot_rigid_visual_mesh_sha256=np.asarray(
            materializer.EXPECTED_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        ),
        measured_racket_source_sha256=np.asarray("c" * 64),
        measured_racket_retarget_receipt_sha256=np.asarray("d" * 64),
        measured_racket_input_motion_sha256=np.asarray("f" * 64),
        measured_racket_manifest_sha256=np.asarray("1" * 64),
        measured_racket_catalog_sha256=np.asarray("2" * 64),
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
        "authorities": {
            "source_manifest": {"sha256": "1" * 64},
            "signed_catalog": {"sha256": "2" * 64},
        },
        "source_manifest": {"sha256": "1" * 64},
        "denominators": {"materialized_npz": 1},
        "actions": [
            {
                "uid": uid,
                "sha256": motion_sha,
                "frames": frames,
                "robot_mount_normal_sign": 1,
            }
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


def _physical_birth_seed_document(
    *, joint_names: tuple[str, ...] | None = None
) -> dict:
    names = joint_names or tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    joint_pos = np.linspace(-0.2, 0.2, 31, dtype=np.float64)
    seed = {
        "schema_version": 2,
        "kind": materializer.PHYSICAL_BIRTH_SEED_KIND,
        "action_id": "bh_loop_c",
        "robot": {"family": "AgiBot A3", "joint_names": list(names)},
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "physical_ready": {
            "joint_pos_rad": joint_pos.tolist(),
            "joint_vel_radps": [0.0] * 31,
            "root_pos_w_m": [0.15, -0.18, 1.0684],
            "root_quat_wxyz": [0.7394323349, 0.0, 0.0, 0.6732308865],
        },
    }
    seed["content_sha256"] = hashlib.sha256(
        materializer._canonical_json_bytes(seed)
    ).hexdigest()
    return seed


def _write_physical_birth_seed(path: Path) -> tuple[Path, dict]:
    seed = _physical_birth_seed_document()
    path.write_text(json.dumps(seed))
    return path, seed


def test_seed_hold_transport_requires_and_preserves_exact_pd_identities() -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    seed = _physical_birth_seed_document(joint_names=names)
    q = np.asarray(seed["physical_ready"]["joint_pos_rad"], np.float64)
    kp = np.linspace(40.0, 70.0, 31)
    default_q = np.linspace(-0.1, 0.1, 31)
    scale = np.full(31, 0.25)
    tau = np.linspace(-2.0, 2.0, 31)
    qdes = q + tau / kp
    action = (qdes - default_q) / scale
    seed["runtime_plant"] = {
        "joint_names": list(names),
        "joint_stiffness": kp.tolist(),
        "default_joint_pos_rad": default_q.tolist(),
        "action_scale_rad": scale.tolist(),
        "executed_qdes_lower_rad": (qdes - 0.1).tolist(),
        "executed_qdes_upper_rad": (qdes + 0.1).tolist(),
    }
    seed["hold_candidate"] = {
        "hold_qdes_joint_pos_rad": qdes.tolist(),
        "normalized_actor_action": action.tolist(),
        "actuator_generalized_force_runtime_order_nm": tau.tolist(),
    }

    loaded = materializer._load_seed_hold_transport(
        seed, joint_names=names
    )
    assert loaded["qdes"] == pytest.approx(qdes)
    assert loaded["normalized_action"] == pytest.approx(action)
    assert loaded["tau_runtime"] == pytest.approx(tau)

    seed["hold_candidate"]["normalized_actor_action"][0] += 0.01
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="qdes/action/tau identity is invalid",
    ):
        materializer._load_seed_hold_transport(seed, joint_names=names)


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


def test_direct_frame0_loader_preserves_stored_float32_quaternion_exactly(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "direct_clip.npz"
    root_quat = np.asarray(
        [0.966759086, 0.040494341, 0.252240956, -0.010565540],
        np.float32,
    )
    joint_pos = np.arange(62, dtype=np.float32).reshape(2, 31) / 100.0
    root_pos = np.asarray([[-0.001, -0.002, 0.891], [0.0, 0.0, 0.9]])
    np.savez(
        clip,
        joint_pos=joint_pos,
        body_pos_w=root_pos[:, None, :].astype(np.float32),
        body_quat_w=np.broadcast_to(root_quat, (2, 1, 4)).copy(),
    )

    loaded_q, loaded_root, loaded_quat = (
        materializer._load_motion_frame0_exact(clip)
    )

    assert np.array_equal(loaded_q, joint_pos[0].astype(np.float64))
    assert np.array_equal(loaded_root, root_pos[0].astype(np.float32))
    assert np.array_equal(loaded_quat, root_quat.astype(np.float64))
    assert not np.array_equal(
        loaded_quat,
        root_quat.astype(np.float64)
        / np.linalg.norm(root_quat.astype(np.float64)),
    )


@pytest.mark.parametrize("mount_sign", (-1, 1))
def test_independent_measured_racket_reference_preserves_signed_face_and_long_axis(
    mount_sign: int,
) -> None:
    motion_sha = "e" * 64
    axis_local = np.asarray(
        materializer.EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
        np.float64,
    )
    signed_face = np.asarray([0.0, 1.0, 0.0], np.float64)
    long_axis = np.asarray([1.0, 0.0, 0.0], np.float64)
    value = {
        "authority": "independent_schema_v4_measured_racket_channel",
        "motion_sha256": motion_sha,
        "frame_index": 0,
        "site_pos_w_m": [0.8, -0.2, 1.1],
        "signed_face_normal_w": signed_face.tolist(),
        "long_axis_w": long_axis.tolist(),
        "position_semantics": (
            materializer.EXPECTED_MEASURED_RACKET_POSITION_SEMANTICS
        ),
        "normal_semantics": (
            materializer.EXPECTED_MEASURED_RACKET_NORMAL_SEMANTICS
        ),
        "long_axis_semantics": (
            materializer.EXPECTED_MEASURED_RACKET_LONG_AXIS_SEMANTICS
        ),
        "robot_mount_normal_sign": mount_sign,
        "robot_butt_to_blade_axis_local": axis_local.tolist(),
        "robot_rigid_visual_mesh_sha256": (
            materializer.EXPECTED_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        ),
    }

    position, rotation, evidence = (
        materializer._independent_measured_racket_frame0_reference(
            value, expected_motion_sha256=motion_sha
        )
    )

    np.testing.assert_array_equal(position, np.asarray([0.8, -0.2, 1.1]))
    np.testing.assert_allclose(rotation @ axis_local, long_axis, atol=1.0e-12)
    np.testing.assert_allclose(
        rotation @ np.asarray([0.0, 1.0, 0.0]),
        mount_sign * signed_face,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-12)
    assert evidence["official_site_rotation_w"] == pytest.approx(rotation)


def test_materialize_scatter_gathers_runtime_and_mujoco_force_orders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    mjcf_source = tmp_path / "a3.xml"
    original_mjcf = b'<mujoco model="hashed"/>\n'
    mjcf_source.write_bytes(original_mjcf)
    mjcf_digest = hashlib.sha256(original_mjcf).hexdigest()
    pinned_mjcf, _ = materializer._pinned_file(
        mjcf_source, mjcf_digest, name="A3 MJCF"
    )
    paths = {
        "motion": Path("/tmp/motion.npz"),
        "receipt": Path("/tmp/receipt.json"),
        "runtime": Path("/tmp/training_contract.json"),
        "mjcf": pinned_mjcf,
    }
    shas = {
        "motion": "1" * 64,
        "receipt": "2" * 64,
        "runtime": "3" * 64,
        "mjcf": mjcf_digest,
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
        if key == "mjcf":
            mjcf_source.write_text('<mujoco model="replacement"/>\n')
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
    def ground_config(**kwargs: object) -> object:
        captured["ground_model_source_path"] = kwargs["model_source_path"]
        return object()

    monkeypatch.setattr(
        materializer.torque_topp, "GroundContactConfig", ground_config
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
    ground_model_path = Path(str(captured["ground_model_source_path"]))
    assert ground_model_path != mjcf_source
    assert ground_model_path.parent == mjcf_source.parent
    assert ground_model_path.read_bytes() == original_mjcf
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


def test_pinned_json_reads_hashed_bytes_after_path_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contract.json"
    original = b'{"generation": "hashed"}\n'
    source.write_bytes(original)
    pinned, digest = materializer._pinned_file(
        source, hashlib.sha256(original).hexdigest(), name="runtime contract"
    )

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"generation": "replacement"}\n')
    replacement.replace(source)

    assert digest == hashlib.sha256(original).hexdigest()
    assert str(pinned) == str(source)
    assert materializer._read_json(pinned, name="runtime contract") == {
        "generation": "hashed"
    }


def test_pinned_npz_reads_hashed_bytes_after_in_place_rewrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "motion.npz"
    root_quat = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    np.savez(
        source,
        joint_pos=np.zeros((2, 31), np.float32),
        body_pos_w=np.zeros((2, 1, 3), np.float32),
        body_quat_w=np.broadcast_to(root_quat, (2, 1, 4)).copy(),
    )
    pinned, _digest = materializer._pinned_file(
        source, _sha256(source), name="motion"
    )

    np.savez(
        source,
        joint_pos=np.full((2, 31), 7.0, np.float32),
        body_pos_w=np.full((2, 1, 3), 8.0, np.float32),
        body_quat_w=np.broadcast_to(root_quat, (2, 1, 4)).copy(),
    )

    joint_pos, root_pos, loaded_quat = (
        materializer._load_motion_frame0_exact(pinned)
    )
    assert np.array_equal(joint_pos, np.zeros(31, np.float64))
    assert np.array_equal(root_pos, np.zeros(3, np.float64))
    assert np.array_equal(loaded_quat, root_quat.astype(np.float64))


def test_exact_model_identity_uses_same_directory_hashed_mjcf_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "a3.xml"
    original = b'<mujoco model="hashed"/>\n'
    source.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    pinned, _actual = materializer._pinned_file(source, digest, name="A3 MJCF")
    receipt = {
        "inputs": {
            "exact_model": {
                "mjcf_sha256": digest,
                "joint_order": list(materializer.grounded.RUNTIME_JOINT_NAMES),
                "compiled_model_sha256": "a" * 64,
                "path_model_binding_sha256": "b" * 64,
                "ground_model_binding_sha256": "c" * 64,
                "xml_model_name": "A3-test",
            }
        }
    }

    identity = materializer._exact_model_identity(
        receipt, mjcf_path=pinned, mjcf_sha256=digest
    )
    pinned_model = Path(identity.mjcf_path)
    source.write_text('<mujoco model="replacement"/>\n')

    assert pinned_model.parent == source.parent
    assert pinned_model != source
    assert pinned_model.read_bytes() == original


def test_exclusive_writer_removes_temp_and_never_publishes_partial_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "candidate.json"
    before = set(tmp_path.iterdir())
    real_write = materializer.os.write
    calls = 0

    def fail_after_prefix(descriptor: int, payload: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(payload)[:3])
        raise OSError("injected write failure")

    monkeypatch.setattr(materializer.os, "write", fail_after_prefix)
    with pytest.raises(OSError, match="injected write failure"):
        materializer._write_exclusive(output, b"complete payload\n")

    assert not output.exists()
    assert set(tmp_path.iterdir()) == before


def test_exclusive_writer_cannot_overwrite_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "candidate.json"
    real_link = materializer.os.link
    real_write = materializer.os.write

    def race_link(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        descriptor = materializer.os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=dst_dir_fd,
        )
        try:
            real_write(descriptor, b"concurrent winner\n")
        finally:
            materializer.os.close(descriptor)
        real_link(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(materializer.os, "link", race_link)
    with pytest.raises(FileExistsError):
        materializer._write_exclusive(output, b"materializer bytes\n")

    assert output.read_bytes() == b"concurrent winner\n"
    assert set(tmp_path.iterdir()) == {output}


def test_exclusive_writer_fsyncs_file_then_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "candidate.json"
    real_fsync = materializer.os.fsync
    fsync_targets: list[str] = []

    def record_fsync(descriptor: int) -> None:
        fsync_targets.append(
            "directory"
            if stat.S_ISDIR(materializer.os.fstat(descriptor).st_mode)
            else "file"
        )
        real_fsync(descriptor)

    monkeypatch.setattr(materializer.os, "fsync", record_fsync)
    materializer._write_exclusive(output, b"complete\n")

    assert fsync_targets == ["file", "directory"]
    assert output.read_bytes() == b"complete\n"


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
    assert evidence["teacher_reference_semantics"] == (
        "exact_original_measured_motion_frame0"
    )
    assert evidence["physical_birth_authority"] == (
        "separate_content_pinned_composition"
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

    changed = json.loads(json.dumps(bank))
    changed["actions"][0]["robot_mount_normal_sign"] = -1
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="does not bind the exact selected motion",
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

    changed = json.loads(json.dumps(bank))
    changed["authorities"]["signed_catalog"]["sha256"] = "3" * 64
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="manifest/catalog authorities",
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
    seed_path, _seed = _write_physical_birth_seed(tmp_path / "seed.json")
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
        physical_birth_seed=str(seed_path),
        expected_physical_birth_seed_sha256=_sha256(seed_path),
        allow_mechanical_unknown_diagnostic=True,
        stable_receipt=None,
        expected_stable_receipt_sha256=None,
        runtime_contract=str(runtime_path),
        expected_runtime_contract_sha256=_sha256(runtime_path),
        mjcf=str(mjcf_path),
        expected_mjcf_sha256=_sha256(mjcf_path),
        output="unused",
    )
    missing_seed = argparse.Namespace(**vars(args))
    missing_seed.physical_birth_seed = None
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="requires --physical-birth-seed",
    ):
        materializer._materialize(missing_seed)

    wrong_seed_sha = argparse.Namespace(**vars(args))
    wrong_seed_sha.expected_physical_birth_seed_sha256 = "0" * 64
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="physical-birth numerical seed SHA-256 mismatch",
    ):
        materializer._materialize(wrong_seed_sha)

    with pytest.raises(ReachedModelIdentity):
        materializer._materialize(args)

    direct_q = np.linspace(-0.2, 0.2, 31, dtype=np.float64)
    direct_root = np.asarray([-0.001, -0.002, 0.891], np.float64)
    direct_quat = np.asarray(
        [0.9667590856552124, 0.040494341403245926,
         0.25224095582962036, -0.010565539821982384],
        np.float64,
    )
    direct_plant = {
        "joint_names": tuple(materializer.grounded.RUNTIME_JOINT_NAMES),
        "physx_control_position_limits": {
            "mechanical_joint_pos_limits": [[-2.0, 2.0]] * 31
        },
        "qdes_limits": np.asarray([[-1.0, 1.0]] * 31, np.float64),
        "projection_inset": 0.0,
        "effort": np.full(31, 100.0, np.float64),
        "kp": np.full(31, 50.0, np.float64),
    }
    identity = SimpleNamespace(ground_model_binding_sha256="a" * 64)
    backend = SimpleNamespace(
        model=SimpleNamespace(nv=37),
        _binding=SimpleNamespace(joint_dof_adrs=np.arange(6, 37)),
        _qpos=lambda ready: np.r_[
            ready.root_pos_w, ready.root_quat_wxyz, ready.joint_pos
        ],
    )
    static_calls: list[object] = []
    lp_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def audit_direct(*, ready: object, **_kwargs: object) -> dict:
        static_calls.append(ready)
        assert np.array_equal(ready.joint_pos, direct_q)
        assert np.array_equal(ready.root_pos_w, direct_root)
        assert ready.root_quat_wxyz == pytest.approx(
            direct_quat / np.linalg.norm(direct_quat), abs=1.0e-15
        )
        return {"gates": {"static_geometry": "PASS", "ground": "PASS"}}

    class ReachedQdesBoundedLP(RuntimeError):
        pass

    class DirectSolver:
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
        ) -> None:
            lp_calls.append((effort_lower.copy(), effort_upper.copy()))
            raise ReachedQdesBoundedLP()

    monkeypatch.setattr(materializer, "_runtime_plant", lambda _c: direct_plant)
    monkeypatch.setattr(
        materializer,
        "_load_motion_frame0_exact",
        lambda _path: (direct_q.copy(), direct_root.copy(), direct_quat.copy()),
    )
    monkeypatch.setattr(
        materializer, "_derive_exact_model_identity", lambda **_kwargs: identity
    )
    monkeypatch.setattr(
        materializer.grounded.MujocoGroundedReadyBackend,
        "load",
        lambda _identity: backend,
    )
    monkeypatch.setattr(materializer, "_audit_composed_physical_birth", audit_direct)
    monkeypatch.setattr(
        materializer.torque_topp, "GroundContactConfig", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        materializer.torque_topp, "MujocoGroundContactLPSolver", DirectSolver
    )
    monkeypatch.setattr(
        materializer.torque_topp,
        "direct_actuator_contract_from_mujoco",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        materializer.torque_topp,
        "_resolve_grounded_actuator_limits",
        lambda _contract, _nv: (
            np.full(31, -1000.0),
            np.full(31, 1000.0),
            np.arange(6, 37),
            {},
        ),
    )
    direct_args = argparse.Namespace(**vars(args))
    direct_args.physical_birth_composition_mode = (
        materializer.MEASURED_BIRTH_DIRECT_FRAME0_MODE
    )
    direct_args.physical_birth_seed = None
    direct_args.expected_physical_birth_seed_sha256 = None
    with pytest.raises(ReachedQdesBoundedLP):
        materializer._materialize(direct_args)
    assert len(static_calls) == 1
    assert len(lp_calls) == 1
    assert np.all(lp_calls[0][0] < 0.0)
    assert np.all(lp_calls[0][1] > 0.0)

    projected_calls: list[dict[str, object]] = []

    def compose_projected(**kwargs: object):
        projected_calls.append(kwargs)
        projected_q = direct_q.copy()
        projected_q[0] += 0.01
        return (
            projected_q,
            direct_root.copy(),
            direct_quat.copy(),
            {
                "semantics": materializer.MEASURED_BIRTH_PROJECTED_FRAME0_SEMANTICS,
                "historical_physical_birth_seed_consumed": False,
            },
            {
                "geometry_passed": True,
                "ground_dynamics_passed": True,
                "gates": {"static_ground_dynamics": "PASS"},
            },
        )

    monkeypatch.setattr(
        materializer,
        "_compose_measured_projected_frame0_physical_birth",
        compose_projected,
    )
    projected_args = argparse.Namespace(**vars(direct_args))
    projected_args.physical_birth_composition_mode = (
        materializer.MEASURED_BIRTH_PROJECTED_FRAME0_MODE
    )
    with pytest.raises(ReachedQdesBoundedLP):
        materializer._materialize(projected_args)
    assert len(projected_calls) == 1
    assert projected_calls[0]["backend"] is backend
    assert projected_calls[0]["identity"] is identity
    assert len(static_calls) == 1
    assert len(lp_calls) == 2

    original_motion_sha = motion_sha
    clip.write_bytes(clip.read_bytes() + b"teacher-drift")
    changed_teacher = argparse.Namespace(**vars(args))
    changed_teacher.expected_motion_sha256 = original_motion_sha
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="measured motion SHA-256 mismatch",
    ):
        materializer._materialize(changed_teacher)


def test_physical_birth_seed_composes_only_root_and_leg12() -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    seed_document = _physical_birth_seed_document()
    seed = materializer._load_physical_birth_seed(
        seed_document, joint_names=names
    )
    teacher_q = np.linspace(1.0, 2.0, 31, dtype=np.float64)
    teacher_root = np.asarray([-0.1, 0.2, 0.9], np.float64)
    # Exact Take_061_unit04_BH frame-0 float32 values.  Their promoted norm is
    # 1.0000000259060773, so a hidden normalization changes emitted bytes.
    teacher_quat = np.asarray(
        [0.966759086, 0.040494341, 0.252240956, -0.010565540],
        np.float32,
    ).astype(np.float64)
    assert np.linalg.norm(teacher_quat) == pytest.approx(
        1.0000000259060773, abs=1.0e-15
    )
    seed = materializer._align_seed_world_yaw_to_teacher(
        seed=seed,
        teacher_root_quat=teacher_quat,
        seed_foot_positions_w=[
            [0.05, -0.3, 0.0],
            [0.05, -0.06, 0.0],
        ],
    )
    ready_q, ready_root, ready_quat, provenance = (
        materializer._compose_measured_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            seed=seed,
        )
    )
    leg_indices = np.asarray(seed["leg_joint_indices"], np.int64)
    nonleg = np.ones(31, dtype=bool)
    nonleg[leg_indices] = False
    assert ready_q[leg_indices] == pytest.approx(
        seed["joint_pos_rad"][leg_indices]
    )
    assert np.array_equal(ready_q[nonleg], teacher_q[nonleg])
    assert ready_root == pytest.approx(seed["root_pos_w_m"])
    assert ready_quat == pytest.approx(seed["root_quat_wxyz"])
    assert provenance["teacher_nonleg_exactly_preserved"] is True
    assert provenance["teacher_and_physical_birth_differ"] is True
    assert len(provenance["leg_joint_indices"]) == 12
    assert len(provenance["nonleg_joint_indices"]) == 19
    assert (
        provenance["seed_world_yaw_alignment"]["semantics"]
        == materializer.MEASURED_SEED_YAW_ALIGNMENT_SEMANTICS
    )


def test_seed_world_yaw_alignment_preserves_tilt_and_support() -> None:
    seed = materializer._load_physical_birth_seed(
        _physical_birth_seed_document(),
        joint_names=materializer.grounded.RUNTIME_JOINT_NAMES,
    )
    original_q = np.asarray(seed["joint_pos_rad"], np.float64).copy()
    original_root_z = float(seed["root_pos_w_m"][2])
    feet = np.asarray(
        [[0.08, -0.31, 0.015], [0.12, -0.05, 0.021]], np.float64
    )
    teacher_yaw = -0.2
    teacher_quat = np.asarray(
        [
            np.cos(teacher_yaw / 2.0),
            0.0,
            0.0,
            np.sin(teacher_yaw / 2.0),
        ],
        np.float64,
    )

    aligned = materializer._align_seed_world_yaw_to_teacher(
        seed=seed,
        teacher_root_quat=teacher_quat,
        seed_foot_positions_w=feet,
    )
    evidence = aligned["seed_world_yaw_alignment"]

    assert materializer._root_yaw_rad(
        aligned["root_quat_wxyz"]
    ) == pytest.approx(teacher_yaw, abs=1.0e-12)
    assert evidence["aligned_minus_teacher_yaw_rad"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert evidence["seed_root_tilt_rad"] == pytest.approx(
        evidence["aligned_root_tilt_rad"], abs=1.0e-12
    )
    expected_feet = np.asarray(
        evidence["expected_aligned_seed_foot_positions_w_m"], np.float64
    )
    assert expected_feet[:, :2].mean(axis=0) == pytest.approx(
        feet[:, :2].mean(axis=0), abs=1.0e-12
    )
    assert expected_feet[:, 2] == pytest.approx(feet[:, 2], abs=1.0e-12)
    assert aligned["root_pos_w_m"][2] == pytest.approx(original_root_z)
    assert np.array_equal(aligned["joint_pos_rad"], original_q)


def test_full_seed_birth_preserves_all_seed_joints_and_exact_teacher() -> None:
    seed = materializer._load_physical_birth_seed(
        _physical_birth_seed_document(),
        joint_names=materializer.grounded.RUNTIME_JOINT_NAMES,
    )
    teacher_q = np.linspace(1.0, 2.0, 31, dtype=np.float64)
    teacher_root = np.asarray([-0.1, 0.2, 0.9], np.float64)
    teacher_quat = np.asarray([1.0, 0.0, 0.0, 0.0], np.float64)
    seed = materializer._align_seed_world_yaw_to_teacher(
        seed=seed,
        teacher_root_quat=teacher_quat,
        seed_foot_positions_w=[
            [0.05, -0.3, 0.0],
            [0.05, -0.06, 0.0],
        ],
    )

    ready_q, ready_root, ready_quat, provenance = (
        materializer._compose_measured_full_seed_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            seed=seed,
        )
    )

    assert np.array_equal(ready_q, seed["joint_pos_rad"])
    assert ready_root == pytest.approx(seed["root_pos_w_m"])
    assert ready_quat == pytest.approx(seed["root_quat_wxyz"])
    assert provenance["semantics"] == (
        materializer.MEASURED_BIRTH_FULL_SEED_SEMANTICS
    )
    assert provenance["seed_all_joints_exactly_preserved"] is True
    assert provenance["teacher_nonleg_exactly_preserved"] is False
    assert provenance["seed_joint_indices"] == list(range(31))
    assert provenance["physical_minus_teacher_joint_pos_rad"] == (
        ready_q - teacher_q
    ).tolist()


def test_scalar_hold_projection_finds_nearest_exact_boundary_both_directions() -> None:
    # The positive direction gets worse, like the observed 0807 waist gravity
    # slope; the feasible boundary is therefore opposite shortfall/kp.
    value, evidence = materializer.nearest_feasible_scalar_boundary(
        current=0.0,
        lower=-0.5,
        upper=0.5,
        initial_step=0.002,
        slack_at=lambda q: -0.1 - 5.0 * q,
    )
    assert value == pytest.approx(-0.02, abs=1.0e-12)
    assert evidence["selected_delta_rad"] == pytest.approx(-0.02)


def test_scalar_hold_projection_refuses_when_interval_has_no_root() -> None:
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="no contact-free hold boundary",
    ):
        materializer.nearest_feasible_scalar_boundary(
            current=0.0,
            lower=-0.5,
            upper=0.5,
            initial_step=0.002,
            slack_at=lambda q: -1.0,
        )


def test_direct_frame0_birth_preserves_exact_teacher_and_consumes_no_seed() -> None:
    teacher_q = np.linspace(-0.3, 0.3, 31, dtype=np.float64)
    teacher_root = np.asarray([-0.001, -0.002, 0.891], np.float64)
    teacher_quat = np.asarray(
        [0.9667590856552124, 0.040494341403245926,
         0.25224095582962036, -0.010565539821982384],
        np.float64,
    )

    ready_q, ready_root, ready_quat, provenance = (
        materializer._compose_measured_direct_frame0_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
        )
    )

    assert np.array_equal(ready_q, teacher_q)
    assert np.array_equal(ready_root, teacher_root)
    assert np.array_equal(ready_quat, teacher_quat)
    assert provenance["semantics"] == (
        materializer.MEASURED_BIRTH_DIRECT_FRAME0_SEMANTICS
    )
    assert provenance["teacher_and_physical_birth_differ"] is False
    assert provenance["historical_physical_birth_seed_consumed"] is False
    assert provenance["required_live_table_gate"] == (
        "isaac_action_ball_nominal_hold_v1"
    )


def test_projected_frame0_birth_preserves_root_nonleg_and_racket_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    leg = materializer.grounded._joint_indices(
        names, materializer.grounded.LEG_JOINT_NAMES
    )
    teacher_q = np.linspace(-0.3, 0.3, 31, dtype=np.float64)
    teacher_root = np.asarray([-0.001, -0.002, 0.891], np.float64)
    teacher_quat = np.asarray(
        [0.9667590856552124, 0.040494341403245926,
         0.25224095582962036, -0.010565539821982384],
        np.float64,
    )
    projected_q = teacher_q.copy()
    projected_q[leg] += np.linspace(-0.05, 0.05, 12)
    normalized_quat = teacher_quat / np.linalg.norm(teacher_quat)
    projected_state = materializer.grounded.ReadyState(
        projected_q,
        teacher_root,
        normalized_quat,
    )
    receipt = {
        "gates": {
            "joint_limits": "PASS",
            "double_support": "PASS",
            "support_margin": "PASS",
            "static_ground_dynamics": "PASS",
        },
        "static_geometry": {"support": {"margin_m": 7.5e-4}},
    }
    result = SimpleNamespace(
        candidate_id="G1S",
        geometry_passed=True,
        ground_dynamics_passed=True,
        state=projected_state,
        receipt=receipt,
        receipt_sha256="a" * 64,
    )
    backend = SimpleNamespace(joint_names=names)
    identity = SimpleNamespace()
    site_position = np.asarray([0.8, -0.26, 1.16], np.float64)
    site_rotation = np.eye(3, dtype=np.float64)
    captured: dict[str, object] = {}

    def solve_projected(donor, **kwargs):
        captured["donor"] = donor
        captured["projection_config"] = kwargs["projection_config"]
        return result

    monkeypatch.setattr(
        materializer.grounded,
        "solve_g1_support_edge_projection",
        solve_projected,
    )
    monkeypatch.setattr(
        materializer,
        "_exact_racket_site_pose",
        lambda *_args, **_kwargs: (site_position.copy(), site_rotation.copy()),
    )

    ready_q, ready_root, ready_quat, provenance, static = (
        materializer._compose_measured_projected_frame0_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            backend=backend,
            identity=identity,
        )
    )

    nonleg = materializer.grounded._joint_indices(
        names, materializer.grounded.UPPER_JOINT_NAMES
    )
    assert np.array_equal(ready_q[nonleg], teacher_q[nonleg])
    assert np.array_equal(ready_root, teacher_root)
    assert np.array_equal(ready_quat, teacher_quat)
    assert captured["donor"].root_quat_wxyz == pytest.approx(
        teacher_quat / np.linalg.norm(teacher_quat), abs=1.0e-15
    )
    assert captured["projection_config"].required_support_margin_m == 5.0e-4
    assert set(provenance["changed_joint_indices"]) == set(leg)
    assert provenance["teacher_root_exactly_preserved"] is True
    assert provenance["teacher_nonleg_exactly_preserved"] is True
    assert provenance["historical_physical_birth_seed_consumed"] is False
    assert provenance["racket_site_fidelity"]["position_bitwise_equal"] is True
    assert provenance["racket_site_fidelity"]["position_error_m"] == 0.0
    assert provenance["racket_site_fidelity"]["rotation_bitwise_equal"] is True
    assert static["grounded_ready_receipt"] == receipt
    assert static["realized_support_margin_m"] >= 5.0e-4


def test_projected_frame0_birth_rejects_racket_fk_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    teacher_q = np.zeros(31, np.float64)
    teacher_root = np.asarray([0.0, 0.0, 0.9], np.float64)
    teacher_quat = np.asarray([1.0, 0.0, 0.0, 0.0], np.float64)
    projected_q = teacher_q.copy()
    projected_q[0] = 0.05
    result = SimpleNamespace(
        candidate_id="G1S",
        geometry_passed=True,
        ground_dynamics_passed=True,
        state=materializer.grounded.ReadyState(
            projected_q, teacher_root, teacher_quat
        ),
        receipt={
            "gates": {},
            "static_geometry": {"support": {"margin_m": 5.0e-4}},
        },
        receipt_sha256="b" * 64,
    )
    poses = iter(
        [
            (np.zeros(3), np.eye(3)),
            (np.asarray([1.0e-9, 0.0, 0.0]), np.eye(3)),
        ]
    )
    monkeypatch.setattr(
        materializer.grounded,
        "solve_g1_support_edge_projection",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        materializer,
        "_exact_racket_site_pose",
        lambda *_args, **_kwargs: next(poses),
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="changed exact racket-site FK",
    ):
        materializer._compose_measured_projected_frame0_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            backend=SimpleNamespace(joint_names=names),
            identity=SimpleNamespace(),
        )


def test_whole_body_threshold_first_exact_frame0_seals_robust_single_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    teacher_q = np.zeros(31, np.float64)
    teacher_root = np.asarray([0.0, 0.0, 0.9], np.float64)
    # Exact Take_061_unit04_BH frame-0 float32 values.  Their promoted norm is
    # 1.0000000259060773, so a hidden normalization changes emitted bytes.
    teacher_quat = np.asarray(
        [0.966759086, 0.040494341, 0.252240956, -0.010565540],
        np.float32,
    ).astype(np.float64)
    assert np.linalg.norm(teacher_quat) == pytest.approx(
        1.0000000259060773, abs=1.0e-15
    )
    final_q = teacher_q.copy()
    final_state = materializer.grounded.ReadyState(
        final_q,
        teacher_root,
        teacher_quat / np.linalg.norm(teacher_quat),
    )
    slacks = {
        name: (
            materializer.whole_body_ready.DIRECT_FRAME0_ROBUST_MINIMUM_SLACKS[
                name
            ]
            + 0.1
        )
        for name in materializer.whole_body_ready.REQUIRED_SAFETY_SLACK_NAMES
    }
    fake_result = materializer.whole_body_ready.WholeBodySafeReadyResult(
        state=final_state,
        safety_slacks=slacks,
        normalized_safety_slacks=slacks,
        worst_normalized_safety_slack=0.01,
        stage1_locked_worst_normalized_slack=1.0e-6,
        changed_joint_mask=(False,) * 31,
        joint_delta_rad=(0.0,) * 31,
        root_position_delta_m=(0.0, 0.0, 0.0),
        root_rotation_delta_rad=(0.0, 0.0, 0.0),
        racket_position_delta_m=(0.01, -0.02, 0.03),
        racket_rotation_delta_rad=(0.0, 0.04, 0.0),
        evaluator_evidence={"exact_contact_lp_reused": True},
        optimizer_report={
            "safety_weighted_against_tracking": False,
            "global_optimum_claimed": False,
            "exact_measured_frame0_selected": True,
        },
    )
    backend = SimpleNamespace(
        position_lower=np.full(31, -2.0),
        position_upper=np.full(31, 2.0),
        vendor_key_state=lambda _index: materializer.grounded.ReadyState(
            np.zeros(31), teacher_root, teacher_quat
        ),
    )
    captured: dict[str, object] = {}
    selected_result = {"value": fake_result}

    def solve(measured, **kwargs):
        captured["measured"] = measured
        captured.update(kwargs)
        return selected_result["value"]

    motion_sha = "e" * 64
    measured_racket_frame0 = {
        "authority": "independent_schema_v4_measured_racket_channel",
        "motion_sha256": motion_sha,
        "frame_index": 0,
        "site_pos_w_m": [0.8, -0.2, 1.1],
        "signed_face_normal_w": [0.0, 1.0, 0.0],
        "long_axis_w": list(
            materializer.EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL
        ),
        "position_semantics": (
            materializer.EXPECTED_MEASURED_RACKET_POSITION_SEMANTICS
        ),
        "normal_semantics": (
            materializer.EXPECTED_MEASURED_RACKET_NORMAL_SEMANTICS
        ),
        "long_axis_semantics": (
            materializer.EXPECTED_MEASURED_RACKET_LONG_AXIS_SEMANTICS
        ),
        "robot_mount_normal_sign": 1,
        "robot_butt_to_blade_axis_local": list(
            materializer.EXPECTED_MEASURED_RACKET_BUTT_TO_BLADE_AXIS_LOCAL
        ),
        "robot_rigid_visual_mesh_sha256": (
            materializer.EXPECTED_MEASURED_RACKET_RIGID_VISUAL_MESH_SHA256
        ),
        "source_sha256": "c" * 64,
        "retarget_receipt_sha256": "d" * 64,
    }

    final_evaluation = materializer.whole_body_ready.SafetyEvaluation(
        slacks=slacks,
        racket_position_w=np.asarray([0.81, -0.22, 1.13], np.float64),
        racket_rotation_w=np.eye(3),
        evidence={
            "lp_feasible": True,
            "exact_state_lp_cache_hit": False,
            "evaluated_state_sha256": materializer.grounded.state_digest(
                final_state
            ),
            "required_minimum_normal_force_per_contact_n": (
                materializer.WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
            ),
            "required_minimum_normal_force_per_foot_n": (
                materializer.WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_FOOT_N
            ),
            "sole_minimum_distance_m": [0.0, 0.0],
            "exact_joint_position_lower_rad": [-2.0] * 31,
            "exact_joint_position_upper_rad": [2.0] * 31,
        },
    )

    monkeypatch.setattr(
        materializer,
        "_build_whole_body_safety_evaluator",
        lambda **_kwargs: (
            lambda _state: final_evaluation,
            {"table_near_x_m": 0.5},
        ),
    )
    monkeypatch.setattr(
        materializer.grounded.MujocoGroundedReadyBackend,
        "load",
        lambda _identity: backend,
    )
    monkeypatch.setattr(
        materializer.whole_body_ready,
        "solve_measured_conditioned_whole_body_safe_ready",
        solve,
    )
    ready_q, ready_root, ready_quat, provenance, static = (
        materializer._compose_measured_whole_body_safe_frame0_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            backend=backend,
            identity=SimpleNamespace(),
            plant={},
            hard_inner_lower=np.full(31, -1.0),
            hard_inner_upper=np.full(31, 1.0),
            runtime_contract={},
            measured_racket_frame0=measured_racket_frame0,
            motion_sha256=motion_sha,
        )
    )
    np.testing.assert_array_equal(ready_q, final_q)
    np.testing.assert_array_equal(ready_root, final_state.root_pos_w)
    np.testing.assert_array_equal(ready_quat, teacher_quat)
    assert not np.array_equal(ready_quat, final_state.root_quat_wxyz)
    assert captured["config"].movable_joint_names == names
    np.testing.assert_array_equal(
        captured["racket_reference_position_w"],
        np.asarray([0.8, -0.2, 1.1]),
    )
    np.testing.assert_allclose(
        captured["racket_reference_rotation_w"], np.eye(3), atol=1.0e-12
    )
    assert provenance["released_root_degrees_of_freedom"] == [
        "z",
        "roll",
        "pitch",
    ]
    assert provenance["released_joint_names"] == list(names)
    assert provenance["changed_joint_mask"] == [False] * 31
    assert provenance["exact_measured_frame0_selected"] is True
    assert provenance["racket_site_fidelity"]["position_error_m"] == pytest.approx(
        np.linalg.norm([0.01, -0.02, 0.03])
    )
    assert provenance["racket_site_fidelity"]["reference_authority"][
        "authority"
    ] == "independent_schema_v4_measured_racket_channel"
    assert provenance["safety_weighted_against_tracking"] is False
    assert static["exact_contact_lp_reused"] is False
    assert static["selected_hold_witness_authority"] == (
        "new_backend_new_solver_final_state_cache_miss"
    )
    assert static["all_safety_slacks_meet_original_and_locked_gate"] is True
    assert static["fresh_direct_robust_gate_passed"] is True
    assert static["safety_slacks"] == slacks
    assert static["evaluator_evidence"]["sole_minimum_distance_m"] == [
        0.0,
        0.0,
    ]
    assert static["evaluator_evidence"][
        "exact_joint_position_lower_rad"
    ] == [-2.0] * 31
    assert static["evaluator_evidence"][
        "exact_joint_position_upper_rad"
    ] == [2.0] * 31
    assert provenance["frame0_handoff"]["certified_transition_s"] == 0.0
    assert provenance["frame0_handoff"]["endpoints_bitwise_equal"] is True
    handoff = provenance["frame0_handoff"]
    assert handoff["physical_ready_state_sha256"] == (
        materializer._stored_ready_state_sha256(
            teacher_q, teacher_root, teacher_quat
        )
    )
    assert handoff["teacher_frame0_state_sha256"] == (
        handoff["physical_ready_state_sha256"]
    )
    assert handoff["mjcf_audit_state_sha256"] == (
        materializer.grounded.state_digest(final_state)
    )
    assert handoff["stored_root_quaternion_norm"] == pytest.approx(
        np.linalg.norm(teacher_quat), abs=0.0
    )
    np.testing.assert_array_equal(
        handoff["mjcf_audit_root_quat_wxyz"],
        final_state.root_quat_wxyz,
    )
    assert handoff["physical_ready_joint_velocity_exact_zero"] is True
    assert handoff["teacher_static_endpoint_joint_velocity_exact_zero"] is True
    assert handoff["measured_motion_velocity_channels_consumed"] is False
    assert handoff["not_a_motion_velocity_continuity_claim"] is True
    assert "zero_velocity_at_both_endpoints" not in handoff

    selected_result["value"] = replace(
        fake_result,
        optimizer_report={
            "safety_weighted_against_tracking": False,
            "global_optimum_claimed": False,
            "exact_measured_frame0_selected": False,
        },
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="learned-bridge fallback is bitwise equal",
    ):
        materializer._compose_measured_whole_body_safe_frame0_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            backend=backend,
            identity=SimpleNamespace(),
            plant={},
            hard_inner_lower=np.full(31, -1.0),
            hard_inner_upper=np.full(31, 1.0),
            runtime_contract={},
            measured_racket_frame0=measured_racket_frame0,
            motion_sha256=motion_sha,
        )


def test_whole_body_selected_hold_rejects_qdes_from_a_second_lp() -> None:
    ready_q = np.zeros(31, np.float64)
    ready_root = np.asarray([0.0, 0.0, 0.9], np.float64)
    ready_quat = np.asarray([1.0, 0.0, 0.0, 0.0], np.float64)
    state = materializer.grounded.ReadyState(ready_q, ready_root, ready_quat)
    identity = SimpleNamespace(ground_model_binding_sha256="a" * 64)
    rows = np.arange(31, dtype=np.int64)
    actuated = rows + 6
    kp = np.full(31, 100.0, np.float64)
    tau_model = np.linspace(-0.2, 0.2, 31, dtype=np.float64)
    tau_runtime = tau_model[rows]
    qdes = ready_q + tau_runtime / kp
    expected_vectors = {
        "executed_qdes_lower_rad": np.full(31, -1.0),
        "executed_qdes_upper_rad": np.full(31, 1.0),
        "model_tau_lower_mujoco_row_order_nm": np.full(31, -2.0),
        "model_tau_upper_mujoco_row_order_nm": np.full(31, 2.0),
        "runtime_tau_lower_runtime_order_nm": np.full(31, -1.5),
        "runtime_tau_upper_runtime_order_nm": np.full(31, 1.5),
        "runtime_tau_lower_mujoco_row_order_nm": np.full(31, -1.5),
        "runtime_tau_upper_mujoco_row_order_nm": np.full(31, 1.5),
        "effective_tau_lower_mujoco_row_order_nm": np.full(31, -1.5),
        "effective_tau_upper_mujoco_row_order_nm": np.full(31, 1.5),
    }
    report = {
        "model_binding": identity.ground_model_binding_sha256,
        "exact_state_lp_cache_hit": False,
        "normal_force_per_contact_n": [0.2] * 8,
        "normal_force_per_foot_n": [10.0, 11.0],
        "cop_interior_margin_per_foot_m": [0.01, 0.02],
    }
    witness = {
        "lp_feasible": True,
        "exact_state_lp_cache_hit": False,
        "required_minimum_normal_force_per_contact_n": (
            materializer.WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_CONTACT_N
        ),
        "required_minimum_normal_force_per_foot_n": (
            materializer.WHOLE_BODY_MINIMUM_NORMAL_FORCE_PER_FOOT_N
        ),
        "evaluated_state_sha256": materializer.grounded.state_digest(state),
        "evaluated_joint_pos_rad": ready_q.tolist(),
        "evaluated_root_pos_w_m": ready_root.tolist(),
        "evaluated_root_quat_wxyz": ready_quat.tolist(),
        "mujoco_row_for_runtime_joint": rows.tolist(),
        "mujoco_actuated_dof_indices": actuated.tolist(),
        "actuator_generalized_force_mujoco_row_order_nm": tau_model.tolist(),
        "actuator_generalized_force_runtime_order_nm": tau_runtime.tolist(),
        "hold_qdes_joint_pos_rad": qdes.tolist(),
        "normal_force_per_contact_n": [0.2] * 8,
        "normal_force_per_foot_n": [10.0, 11.0],
        "cop_interior_margin_per_foot_m": [0.01, 0.02],
        "equality_residual": 1.0e-9,
        "root_residual": 2.0e-9,
        "solver_report": report,
        **{name: value.tolist() for name, value in expected_vectors.items()},
    }
    static = {
        "selected_hold_witness_authority": (
            "new_backend_new_solver_final_state_cache_miss"
        ),
        "evaluator_evidence": witness,
    }

    selected = materializer._consume_whole_body_selected_hold_witness(
        static_birth_evidence=static,
        ready_q=ready_q,
        ready_root_pos=ready_root,
        ready_root_quat=ready_quat,
        identity=identity,
        kp=kp,
        model_row_for_runtime=rows,
        actuated=actuated,
        expected_vectors=expected_vectors,
    )
    np.testing.assert_array_equal(
        selected.actuator_generalized_force, tau_model
    )

    second_lp_witness = dict(witness)
    second_lp_witness["hold_qdes_joint_pos_rad"] = (
        qdes + 1.0e-3
    ).tolist()
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="qdes/torque is internally inconsistent",
    ):
        materializer._consume_whole_body_selected_hold_witness(
            static_birth_evidence={
                **static,
                "evaluator_evidence": second_lp_witness,
            },
            ready_q=ready_q,
            ready_root_pos=ready_root,
            ready_root_quat=ready_quat,
            identity=identity,
            kp=kp,
            model_row_for_runtime=rows,
            actuated=actuated,
            expected_vectors=expected_vectors,
        )


def test_whole_body_table_gate_is_runtime_bound_and_conservative() -> None:
    contract = {
        "action_ball_training": {
            "runtime": {
                "counter_rally": {
                    "objective_profile": {
                        "table_near_x_env_m": 0.5,
                        "table_half_width_m": 0.7625,
                        "table_surface_z_env_m": 0.76,
                    }
                }
            }
        }
    }
    assert materializer._whole_body_table_geometry(contract) == (
        0.5,
        0.7625,
        0.76,
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="runtime-bound table geometry",
    ):
        materializer._whole_body_table_geometry({})

    model = SimpleNamespace(
        ngeom=3,
        geom_bodyid=np.asarray([0, 1, 1], np.int64),
        geom_contype=np.asarray([1, 1, 1], np.int64),
        geom_rbound=np.asarray([0.0, 0.1, 0.05], np.float64),
    )
    data = SimpleNamespace(
        geom_xpos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.7],
                [0.3, 0.0, 1.0],
            ],
            np.float64,
        )
    )
    backend = SimpleNamespace(
        model=model,
        _data=data,
        _floor_geom=0,
        _install=lambda _state: None,
    )
    clearance = materializer._conservative_robot_table_clearance_m(
        backend,
        object(),
        table_near_x_m=0.5,
        table_half_width_m=0.7625,
        table_surface_z_m=0.76,
    )
    # First geom proves 0.20 m near-edge separation; second proves 0.19 m
    # top separation.  The conservative all-geom result is their minimum.
    assert clearance == pytest.approx(0.19, abs=1.0e-15)

    data.geom_xpos[1] = [0.55, 0.0, 0.7]
    assert materializer._conservative_robot_table_clearance_m(
        backend,
        object(),
        table_near_x_m=0.5,
        table_half_width_m=0.7625,
        table_surface_z_m=0.76,
    ) < 0.0


def test_whole_body_collision_bisection_midpoint_is_deducted_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tolerance = materializer.WHOLE_BODY_COLLISION_CLEARANCE_TOLERANCE_M
    raw_midpoint = (
        materializer.WHOLE_BODY_REQUIRED_COLLISION_CLEARANCE_M
        - 0.49 * tolerance
    )
    model = SimpleNamespace(
        ngeom=3,
        geom_rbound=np.asarray([0.0, 0.2, 0.2], np.float64),
    )
    data = SimpleNamespace(
        geom_xpos=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.1, 0.0, 0.5]],
            np.float64,
        )
    )
    backend = SimpleNamespace(model=model, _data=data, _floor_geom=0)
    monkeypatch.setattr(
        materializer.self_collision_audit,
        "geom_clearance",
        lambda *_args, **_kwargs: (raw_midpoint, False),
    )

    conservative, raw = materializer._whole_body_collision_clearance_m(
        backend,
        self_pairs=((1, 2),),
        unsupported_floor_geoms=(),
    )
    assert raw == raw_midpoint
    assert conservative == pytest.approx(raw_midpoint - tolerance)
    assert conservative < materializer.WHOLE_BODY_REQUIRED_COLLISION_CLEARANCE_M

def test_projected_frame0_birth_rejects_incomplete_static_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = tuple(materializer.grounded.RUNTIME_JOINT_NAMES)
    teacher_q = np.zeros(31, np.float64)
    teacher_root = np.asarray([0.0, 0.0, 0.9], np.float64)
    teacher_quat = np.asarray([1.0, 0.0, 0.0, 0.0], np.float64)
    monkeypatch.setattr(
        materializer.grounded,
        "solve_g1_support_edge_projection",
        lambda *_args, **_kwargs: SimpleNamespace(
            candidate_id="G1S",
            geometry_passed=True,
            ground_dynamics_passed=None,
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_exact_racket_site_pose",
        lambda *_args, **_kwargs: (np.zeros(3), np.eye(3)),
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="complete G1S static pass",
    ):
        materializer._compose_measured_projected_frame0_physical_birth(
            teacher_q=teacher_q,
            teacher_root_pos=teacher_root,
            teacher_root_quat=teacher_quat,
            backend=SimpleNamespace(joint_names=names),
            identity=SimpleNamespace(),
        )


def test_physical_birth_seed_rejects_leg_joint_mapping_drift() -> None:
    names = list(materializer.grounded.RUNTIME_JOINT_NAMES)
    names[0], names[2] = names[2], names[0]
    seed = _physical_birth_seed_document(joint_names=tuple(names))
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="joint mapping drifted",
    ):
        materializer._load_physical_birth_seed(
            seed, joint_names=materializer.grounded.RUNTIME_JOINT_NAMES
        )


def test_composed_physical_birth_fails_closed_on_current_ground_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = SimpleNamespace()
    backend = SimpleNamespace(foot_poses=lambda _ready: object())
    identity = SimpleNamespace()
    failed = SimpleNamespace(
        geometry_passed=True,
        ground_dynamics_passed=False,
        receipt={"gates": {"static_geometry": "PASS", "ground": "FAIL"}},
    )
    monkeypatch.setattr(
        materializer.grounded,
        "_audit_and_build_result",
        lambda *_a, **_k: failed,
    )
    monkeypatch.setattr(
        materializer.grounded, "GroundedReadyConfig", lambda: object()
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match="failed current-MJCF static gates",
    ):
        materializer._audit_composed_physical_birth(
            ready=ready,
            backend=backend,
            identity=identity,
            source={"fixture": True},
        )


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
    projected_seed = parser.parse_args(
        [
            *common,
            "--ready-source-kind",
            materializer.MEASURED_RETARGET_SOURCE_KIND,
            "--physical-birth-composition-mode",
            materializer.MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE,
        ]
    )
    assert projected_seed.physical_birth_composition_mode == (
        materializer.MEASURED_BIRTH_HOLDABLE_FULL_SEED_MODE
    )
    direct = parser.parse_args(
        [
            *common,
            "--ready-source-kind",
            materializer.MEASURED_RETARGET_SOURCE_KIND,
            "--physical-birth-composition-mode",
            materializer.MEASURED_BIRTH_DIRECT_FRAME0_MODE,
        ]
    )
    assert direct.physical_birth_composition_mode == (
        materializer.MEASURED_BIRTH_DIRECT_FRAME0_MODE
    )
    projected = parser.parse_args(
        [
            *common,
            "--ready-source-kind",
            materializer.MEASURED_RETARGET_SOURCE_KIND,
            "--physical-birth-composition-mode",
            materializer.MEASURED_BIRTH_PROJECTED_FRAME0_MODE,
        ]
    )
    assert projected.physical_birth_composition_mode == (
        materializer.MEASURED_BIRTH_PROJECTED_FRAME0_MODE
    )
    whole_body = parser.parse_args(
        [
            *common,
            "--ready-source-kind",
            materializer.MEASURED_RETARGET_SOURCE_KIND,
            "--physical-birth-composition-mode",
            materializer.MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE,
        ]
    )
    assert whole_body.physical_birth_composition_mode == (
        materializer.MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_MODE
    )


@pytest.mark.parametrize(
    ("birth_mode", "qdes_mode", "normal_reserve", "message"),
    [
        (
            materializer.MEASURED_BIRTH_DIRECT_FRAME0_MODE,
            materializer.FULL_SEED_QDES_SEED_TRANSPORT,
            0.0,
            "seed-transport qdes requires full-seed",
        ),
        (
            materializer.MEASURED_BIRTH_PROJECTED_FRAME0_MODE,
            materializer.FULL_SEED_QDES_SEED_TRANSPORT,
            0.0,
            "seed-transport qdes requires full-seed",
        ),
        (
            materializer.MEASURED_BIRTH_DIRECT_FRAME0_MODE,
            materializer.FULL_SEED_QDES_FRESH_STATIC_LP,
            20.0,
            "normal reserve is a finite non-negative full-seed",
        ),
        (
            materializer.MEASURED_BIRTH_PROJECTED_FRAME0_MODE,
            materializer.FULL_SEED_QDES_FRESH_STATIC_LP,
            20.0,
            "normal reserve is a finite non-negative full-seed",
        ),
    ],
)
def test_direct_frame0_rejects_seed_transport_and_robust_contact_modes(
    birth_mode: str, qdes_mode: str, normal_reserve: float, message: str
) -> None:
    args = argparse.Namespace(
        ready_source_kind=materializer.MEASURED_RETARGET_SOURCE_KIND,
        physical_birth_composition_mode=birth_mode,
        full_seed_hold_qdes_mode=qdes_mode,
        full_seed_minimum_normal_force_per_support_vertex_n=normal_reserve,
    )
    with pytest.raises(
        materializer.DynamicReadyMaterializationError,
        match=message,
    ):
        materializer._materialize(args)
