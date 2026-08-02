#!/usr/bin/env python3
"""Build a diagnostic ActionBall physical-ready/hold-qdes candidate.

The measured teacher frame remains the reference.  The physical reset uses a
previously audited shared lower-body/root seed and overlays the measured
teacher's non-leg joints.  The resulting state is re-audited on the current
exact MJCF, then a double-support LP finds a gravity-supporting qdes inside the
runtime execution envelope.  The output is diagnostic-only and no-clobber.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import single_env as core


KIND = "a3_mujoco_action_specific_hold_candidate_v1"
SCHEMA_VERSION = 1
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(row) for row in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise core.ContractError(f"cannot hash source {path}: {exc}") from exc


def _require_expected_sha(path: Path, expected: str, label: str) -> str:
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise core.ContractError(f"expected {label} SHA-256 is invalid")
    actual = _file_sha(path)
    if actual != expected:
        raise core.ContractError(
            f"{label} SHA-256 mismatch: actual={actual}, expected={expected}"
        )
    return actual


def _load_solver_modules() -> tuple[Any, Any, Any]:
    scripts = str(SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import canonical_grounded_ready as grounded
        import canonical_torque_path_topp as torque
        import materialize_a3_dynamic_ready_contract as dynamic
    except ImportError as exc:
        raise core.ContractError(
            f"MuJoCo/SciPy grounded-hold dependencies are unavailable: {exc}"
        ) from exc
    return grounded, torque, dynamic


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise core.ContractError(f"{label} must contain {size} finite scalars")
    return out.copy()


def build_candidate(
    *,
    binding: core.PlantBinding,
    teacher_motion: Path | str,
    teacher_frame: int,
    seed_dynamic_ready: Path | str,
    expected_seed_sha256: str,
    mjcf_path: Path | str = core.DEFAULT_MJCF,
) -> dict[str, Any]:
    """Build one exact-model, runtime-envelope-constrained hold candidate."""

    grounded, torque, dynamic = _load_solver_modules()
    teacher_path = Path(teacher_motion).expanduser().resolve()
    teacher_payload, _teacher_action = core._teacher_frame_reset_payload(
        binding, teacher_path, teacher_frame
    )
    teacher_q = _vector(teacher_payload["joint_pos"], core.ACTION_DIM, "teacher q")

    seed_path = Path(seed_dynamic_ready).expanduser().resolve()
    seed_sha = _require_expected_sha(
        seed_path, expected_seed_sha256, "dynamic-ready seed"
    )
    seed, _seed_raw = core._load_strict_json(seed_path)
    if (
        seed.get("schema_version") != 2
        or seed.get("kind") != "agibot_a3_action_dynamic_ready_candidate_v2"
    ):
        raise core.ContractError("dynamic-ready seed kind/schema mismatch")
    robot = seed.get("robot")
    physical = seed.get("physical_ready")
    if (
        not isinstance(robot, dict)
        or tuple(robot.get("joint_names", ())) != binding.joint_names
        or not isinstance(physical, dict)
    ):
        raise core.ContractError("dynamic-ready seed joint order/physical ready is invalid")
    seed_q = _vector(physical.get("joint_pos_rad"), core.ACTION_DIM, "seed q")
    seed_root = _vector(physical.get("root_pos_w_m"), 3, "seed root position")
    seed_quat = _vector(physical.get("root_quat_wxyz"), 4, "seed root quaternion")
    seed_quat /= np.linalg.norm(seed_quat)

    leg_names = frozenset(str(name) for name in grounded.LEG_JOINT_NAMES)
    leg_indices = np.asarray(
        [index for index, name in enumerate(binding.joint_names) if name in leg_names],
        dtype=np.int64,
    )
    if leg_indices.shape != (12,):
        raise core.ContractError("canonical grounded-ready leg order is not 12-D")
    ready_q = teacher_q.copy()
    ready_q[leg_indices] = seed_q[leg_indices]

    mjcf = Path(mjcf_path).expanduser().resolve()
    mjcf_sha = _file_sha(mjcf)
    identity = dynamic._derive_exact_model_identity(
        mjcf_path=mjcf, mjcf_sha256=mjcf_sha
    )
    backend = grounded.MujocoGroundedReadyBackend.load(identity)
    ready = grounded.ReadyState(ready_q, seed_root, seed_quat)
    targets = backend.foot_poses(ready)
    config = grounded.GroundedReadyConfig()
    audit = grounded._audit_and_build_result(
        "action-specific-shared-lower-overlay",
        ready,
        targets,
        source={
            "mode": "shared_grounded_lower_root_plus_teacher_nonleg_overlay",
            "seed_sha256": seed_sha,
            "teacher_motion_sha256": teacher_payload["source_motion_sha256"],
            "teacher_frame": int(teacher_frame),
        },
        backend=backend,
        expected_model_identity=identity,
        config=config,
    )
    if not audit.geometry_passed or audit.ground_dynamics_passed is not True:
        raise core.ContractError(
            "action-specific physical ready failed exact static gates: "
            f"{dict(audit.receipt['gates'])}"
        )

    qpos = backend._qpos(ready)
    ground_config = torque.GroundContactConfig(
        expected_model_binding=identity.ground_model_binding_sha256,
        model_source_path=str(mjcf),
        expected_source_sha256=mjcf_sha,
    )
    solver = torque.MujocoGroundContactLPSolver(backend.model, ground_config)
    actuator_contract = torque.direct_actuator_contract_from_mujoco(
        backend.model,
        support_mode="ground",
        contact_mode="double_support_floor",
        fixed_lp_solver="scipy.optimize.linprog:highs",
    )
    model_lower, model_upper, actuated, _limit_report = (
        torque._resolve_grounded_actuator_limits(
            actuator_contract, int(backend.model.nv)
        )
    )
    model_row_for_runtime = (
        np.asarray(backend._binding.joint_dof_adrs, dtype=np.int64) - 6
    )
    if not np.array_equal(
        np.sort(model_row_for_runtime), np.arange(core.ACTION_DIM)
    ):
        raise core.ContractError("runtime-to-MuJoCo actuator rows are not bijective")

    runtime_lower = np.maximum(
        -binding.effort_limits,
        binding.stiffness * (binding.executed_qdes_limits[:, 0] - ready_q),
    )
    runtime_upper = np.minimum(
        binding.effort_limits,
        binding.stiffness * (binding.executed_qdes_limits[:, 1] - ready_q),
    )
    runtime_lower_model = np.empty(core.ACTION_DIM, dtype=np.float64)
    runtime_upper_model = np.empty(core.ACTION_DIM, dtype=np.float64)
    runtime_lower_model[model_row_for_runtime] = runtime_lower
    runtime_upper_model[model_row_for_runtime] = runtime_upper
    effective_lower = np.maximum(model_lower, runtime_lower_model)
    effective_upper = np.minimum(model_upper, runtime_upper_model)
    if np.any(effective_lower >= effective_upper):
        raise core.ContractError("runtime/model hold-torque envelope is empty")

    solution = solver.solve(
        qpos,
        np.zeros(int(backend.model.nv), dtype=np.float64),
        np.zeros(int(backend.model.nv), dtype=np.float64),
        actuated,
        effective_lower,
        effective_upper,
        np.full(int(backend.model.nv), 1.0e6, dtype=np.float64),
        path_tangent=np.zeros(int(backend.model.nv), dtype=np.float64),
        lp_objective=torque.GROUND_LP_OBJECTIVE_HOLD_MINIMAX,
    )
    if not solution.feasible:
        raise core.ContractError(
            "no action-specific static hold exists inside the runtime qdes envelope: "
            f"{solution.report.get('highs_message')}"
        )
    tau_model = np.asarray(solution.actuator_generalized_force, dtype=np.float64)
    tau_runtime = tau_model[model_row_for_runtime]
    hold_qdes = ready_q + tau_runtime / binding.stiffness
    hold_action = (hold_qdes - binding.default_joint_pos) / binding.action_scale
    _raw, decoded_qdes, clamps = binding.decode_action(hold_action)
    if clamps or not np.allclose(decoded_qdes, hold_qdes, rtol=0.0, atol=2.0e-10):
        raise core.ContractError("derived action-specific hold qdes is not exactly executable")

    static_geometry = audit.receipt["static_geometry"]
    solver_report = solution.report
    candidate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "diagnostic_unauthorized": True,
        "authorization": {
            "training": False,
            "promotion": False,
            "deployment": False,
            "hardware": False,
        },
        "joint_names": list(binding.joint_names),
        "sources": {
            "training_contract": {
                "path": binding.source_path,
                "sha256": binding.source_sha256,
            },
            "teacher_motion": {
                "path": teacher_payload["source_motion_path"],
                "sha256": teacher_payload["source_motion_sha256"],
                "uid": teacher_payload["source_motion_uid"],
                "frame": teacher_payload["source_frame_index"],
                "joint_order_contract_id": teacher_payload[
                    "source_joint_order_contract_id"
                ],
                "joint_order_contract_sha256": teacher_payload[
                    "source_joint_order_contract_sha256"
                ],
            },
            "shared_lower_root_seed": {
                "path": str(seed_path),
                "sha256": seed_sha,
                "source_action_id": seed.get("action_id"),
                "consumed_fields": ["physical_ready.root", "physical_ready.leg12"],
            },
            "root_mjcf": {"path": str(mjcf), "sha256": mjcf_sha},
        },
        "semantics": {
            "teacher_reference_unchanged": True,
            "physical_reset": "shared_grounded_lower_root_plus_teacher_nonleg",
            "controller_birth_target": "static_lp_hold_qdes",
            "history_fill": "same_static_lp_hold_action",
            "teacher_and_physical_reset_may_differ": True,
        },
        "physical_ready": {
            "joint_pos": ready_q.tolist(),
            "joint_vel": [0.0] * core.ACTION_DIM,
            "root_pos": seed_root.tolist(),
            "root_quat_wxyz": seed_quat.tolist(),
            "root_lin_vel_w": [0.0, 0.0, 0.0],
            "root_ang_vel_w": [0.0, 0.0, 0.0],
            "root_lin_vel_point": "link_origin",
            "leg_joint_indices": leg_indices.tolist(),
            "leg_joint_names": [binding.joint_names[index] for index in leg_indices],
            "nonleg_exact_teacher_q0": bool(
                np.array_equal(
                    np.delete(ready_q, leg_indices), np.delete(teacher_q, leg_indices)
                )
            ),
        },
        "hold": {
            "joint_qdes": hold_qdes.tolist(),
            "normalized_action": hold_action.tolist(),
            "actuator_force_runtime_nm": tau_runtime.tolist(),
            "maximum_abs_normalized_action": float(np.max(np.abs(hold_action))),
            "maximum_abs_actuator_force_nm": float(np.max(np.abs(tau_runtime))),
        },
        "static_evidence": {
            "grounded_ready_receipt_sha256": audit.receipt_sha256,
            "gates": _jsonable(audit.receipt["gates"]),
            "support_margin_m": float(static_geometry["support"]["margin_m"]),
            "maximum_foot_penetration_m": float(
                static_geometry["sole_floor"]["maximum_foot_penetration_m"]
            ),
            "lp": {
                "status": solver_report.get("status"),
                "solver": solver_report.get("solver"),
                "highs_status": solver_report.get("highs_status"),
                "root_residual": solver_report.get("root_residual"),
                "max_inequality_violation": solver_report.get(
                    "max_inequality_violation"
                ),
                "optimum_max_normalized_available_hold_torque": (
                    solver_report.get("optimum_max_normalized_available_hold_torque")
                ),
                "normal_force_per_foot_n": _jsonable(
                    solver_report.get("normal_force_per_foot_n")
                ),
                "model_binding": solver_report.get("model_binding"),
            },
        },
        "non_claims": [
            "not an Isaac or PhysX nominal-hold certificate",
            "not mechanical admission for the measured teacher",
            "not training, promotion, deployment, or hardware authorization",
        ],
    }
    candidate["content_sha256"] = core._sha256(core._canonical_json_bytes(candidate))
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--teacher-motion", required=True)
    parser.add_argument("--teacher-frame", type=int, default=0)
    parser.add_argument("--seed-dynamic-ready", required=True)
    parser.add_argument("--expected-seed-sha256", required=True)
    parser.add_argument("--mjcf", default=str(core.DEFAULT_MJCF))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    binding = core.load_plant_binding(args.contract)
    payload = build_candidate(
        binding=binding,
        teacher_motion=args.teacher_motion,
        teacher_frame=args.teacher_frame,
        seed_dynamic_ready=args.seed_dynamic_ready,
        expected_seed_sha256=args.expected_seed_sha256,
        mjcf_path=args.mjcf,
    )
    raw = core._canonical_json_bytes(payload)
    core._write_new_bytes(Path(args.output), raw)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "file_sha256": core._sha256(raw),
                "content_sha256": payload["content_sha256"],
                "diagnostic_unauthorized": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
