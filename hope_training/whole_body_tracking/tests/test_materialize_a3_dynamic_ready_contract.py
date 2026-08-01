"""Dependency-light contract tests for the A3 dynamic-ready producer.

The exact MuJoCo/HiGHS solve and Isaac closed-loop hold remain Pod integration
tests.  These tests protect the easy-to-regress order conversion around that
solve: the LP speaks MuJoCo post-root DoF order, while the training contract,
PD gains, qdes, and actor action all speak the A3 runtime joint order.
"""

from __future__ import annotations

import argparse
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
