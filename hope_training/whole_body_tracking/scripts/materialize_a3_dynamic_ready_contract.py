#!/usr/bin/env python3
"""Materialize one action-specific AgiBot A3 dynamic-ready hold candidate.

The artifact separates the physical motion frame-0 pose from the implicit-PD
joint target required to hold that pose.  It is a deterministic candidate, not
an Isaac hold certificate and not training authorization.  A downstream
nominal-hold probe must validate it on the exact Isaac/PhysX plant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import canonical_grounded_ready as grounded
import canonical_torque_path_topp as torque_topp


SCHEMA_VERSION = 2
KIND = "agibot_a3_action_dynamic_ready_candidate_v2"
LP_OBJECTIVE = torque_topp.GROUND_LP_OBJECTIVE_HOLD_MINIMAX


class DynamicReadyMaterializationError(RuntimeError):
    """The requested dynamic-ready artifact cannot be produced exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise DynamicReadyMaterializationError(
            f"{name} must be 64 lowercase SHA-256 digits"
        )
    return digest


def _pinned_file(
    path_value: str | Path, expected_sha256: object, *, name: str
) -> tuple[Path, str]:
    path_input = Path(path_value).expanduser().absolute()
    try:
        path = path_input.resolve(strict=True)
    except OSError as exc:
        raise DynamicReadyMaterializationError(
            f"cannot resolve {name}: {exc}"
        ) from exc
    if path_input != path or path_input.is_symlink() or not path.is_file():
        raise DynamicReadyMaterializationError(
            f"{name} must be one regular file without symlink components"
        )
    expected = _require_sha256(expected_sha256, name=f"expected {name}")
    actual = _sha256_file(path)
    if actual != expected:
        raise DynamicReadyMaterializationError(
            f"{name} SHA-256 mismatch: {actual} != {expected}"
        )
    return path, actual


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DynamicReadyMaterializationError(
            f"cannot read {name} JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DynamicReadyMaterializationError(f"{name} must be one JSON object")
    return payload


def _validate_stable_receipt(
    receipt: Mapping[str, Any],
    *,
    motion_sha256: str,
) -> None:
    if (
        receipt.get("schema_version") != 2
        or receipt.get("artifact_class")
        != "diagnostic_a3_stable_upper_motion_v2"
        or receipt.get("verdict")
        != "PASS_DIAGNOSTIC_A3_STABLE_UPPER_WAIST_REBASED_REBUILD"
    ):
        raise DynamicReadyMaterializationError(
            "stable receipt is not the exact A3 stable-upper-v2 artifact class"
        )
    robot = receipt.get("robot")
    authorization = receipt.get("authorization")
    outputs = receipt.get("outputs")
    if (
        not isinstance(robot, Mapping)
        or robot.get("family") != "AgiBot A3"
        or not isinstance(authorization, Mapping)
        or any(
            authorization.get(name) is not False
            for name in (
                "training_authorized",
                "deployment_authorized",
                "hardware_authorized",
            )
        )
        or not isinstance(outputs, Mapping)
        or outputs.get("motion_sha256") != motion_sha256
    ):
        raise DynamicReadyMaterializationError(
            "stable receipt robot, authorization, or output binding is invalid"
        )
    seal = _require_sha256(
        receipt.get("receipt_payload_sha256"),
        name="stable receipt payload seal",
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_payload_sha256", None)
    actual_seal = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if actual_seal != seal:
        raise DynamicReadyMaterializationError(
            "stable receipt payload seal does not match its canonical content"
        )


def _plain_finite_vector(
    value: object, *, name: str, size: int
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise DynamicReadyMaterializationError(
            f"{name} must contain exactly {size} entries"
        )
    if any(
        isinstance(item, bool)
        or type(item) not in (int, float)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise DynamicReadyMaterializationError(
            f"{name} must contain plain finite numbers"
        )
    return np.asarray(value, np.float64)


def _plain_finite_matrix(
    value: object, *, name: str, rows: int, columns: int
) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise DynamicReadyMaterializationError(
            f"{name} must contain exactly {rows} rows"
        )
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (rows, columns) or not np.all(np.isfinite(matrix)):
        raise DynamicReadyMaterializationError(
            f"{name} must be a finite [{rows},{columns}] matrix"
        )
    return matrix


def _load_motion_frame0(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            joint_pos = np.asarray(archive["joint_pos"], np.float64)
            body_pos = np.asarray(archive["body_pos_w"], np.float64)
            body_quat = np.asarray(archive["body_quat_w"], np.float64)
    except Exception as exc:
        raise DynamicReadyMaterializationError(
            f"cannot load stable motion: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        joint_pos.ndim != 2
        or joint_pos.shape[0] < 2
        or joint_pos.shape[1] != 31
        or body_pos.ndim != 3
        or body_pos.shape[0] != joint_pos.shape[0]
        or body_pos.shape[1] < 1
        or body_pos.shape[2] != 3
        or body_quat.shape != (joint_pos.shape[0], body_pos.shape[1], 4)
        or not np.all(np.isfinite(joint_pos))
        or not np.all(np.isfinite(body_pos))
        or not np.all(np.isfinite(body_quat))
    ):
        raise DynamicReadyMaterializationError(
            "stable motion frame arrays are malformed or non-finite"
        )
    return joint_pos[0].copy(), body_pos[0, 0].copy(), body_quat[0, 0].copy()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_exclusive(path_value: str | Path, payload: bytes) -> Path:
    output = Path(path_value).expanduser().absolute()
    parent_input = output.parent
    try:
        parent = parent_input.resolve(strict=True)
    except OSError as exc:
        raise DynamicReadyMaterializationError(
            f"cannot resolve output parent: {exc}"
        ) from exc
    if parent_input != parent or not parent.is_dir() or not output.name:
        raise DynamicReadyMaterializationError(
            "output must have one concrete leaf under an existing real directory"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_descriptor = os.open(parent, parent_flags)
    try:
        descriptor = os.open(
            output.name,
            flags,
            0o644,
            dir_fd=parent_descriptor,
        )
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("exclusive write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    return output


def _exact_model_identity(
    receipt: Mapping[str, Any], *, mjcf_path: Path, mjcf_sha256: str
) -> grounded.ExactModelIdentity:
    try:
        exact = receipt["inputs"]["exact_model"]
    except (KeyError, TypeError) as exc:
        raise DynamicReadyMaterializationError(
            "stable receipt has no exact-model identity"
        ) from exc
    if not isinstance(exact, Mapping):
        raise DynamicReadyMaterializationError(
            "stable receipt exact-model identity must be an object"
        )
    if str(exact.get("mjcf_sha256")) != mjcf_sha256:
        raise DynamicReadyMaterializationError(
            "stable receipt and supplied MJCF have different SHA-256"
        )
    joint_order = tuple(str(value) for value in exact.get("joint_order", ()))
    if joint_order != grounded.RUNTIME_JOINT_NAMES:
        raise DynamicReadyMaterializationError(
            "stable receipt exact-model joint order is not the A3 runtime order"
        )
    return grounded.ExactModelIdentity(
        mjcf_path=str(mjcf_path),
        mjcf_sha256=mjcf_sha256,
        compiled_model_sha256=_require_sha256(
            exact.get("compiled_model_sha256"),
            name="compiled_model_sha256",
        ),
        path_model_binding_sha256=_require_sha256(
            exact.get("path_model_binding_sha256"),
            name="path_model_binding_sha256",
        ),
        ground_model_binding_sha256=_require_sha256(
            exact.get("ground_model_binding_sha256"),
            name="ground_model_binding_sha256",
        ),
        xml_model_name=str(exact.get("xml_model_name")),
    )


def _runtime_plant(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("schema_version") != 3:
        raise DynamicReadyMaterializationError(
            "runtime training contract schema_version must be 3"
        )
    names = tuple(str(value) for value in contract.get("joint_names", ()))
    articulation_names = tuple(
        str(value) for value in contract.get("articulation_joint_names", ())
    )
    action_joint_ids = contract.get("action_joint_ids")
    if (
        names != grounded.RUNTIME_JOINT_NAMES
        or articulation_names != names
        or action_joint_ids != list(range(len(names)))
    ):
        raise DynamicReadyMaterializationError(
            "runtime contract does not bind the exact full A3 action joint order"
        )
    count = len(names)
    kp = _plain_finite_vector(
        contract.get("joint_stiffness"), name="joint_stiffness", size=count
    )
    kd = _plain_finite_vector(
        contract.get("joint_damping"), name="joint_damping", size=count
    )
    effort = _plain_finite_vector(
        contract.get("joint_effort_limits"),
        name="joint_effort_limits",
        size=count,
    )
    velocity = _plain_finite_vector(
        contract.get("joint_velocity_limits"),
        name="joint_velocity_limits",
        size=count,
    )
    default_q = _plain_finite_vector(
        contract.get("default_joint_pos"),
        name="default_joint_pos",
        size=count,
    )
    action_scale = _plain_finite_vector(
        contract.get("action_scale"), name="action_scale", size=count
    )
    qdes_limits = _plain_finite_matrix(
        contract.get("qdes_joint_pos_limits"),
        name="qdes_joint_pos_limits",
        rows=count,
        columns=2,
    )
    actuator_types = contract.get("joint_actuator_types")
    armature = _plain_finite_vector(
        contract.get("joint_armature"), name="joint_armature", size=count
    )
    friction = _plain_finite_vector(
        contract.get("joint_friction_coefficients"),
        name="joint_friction_coefficients",
        size=count,
    )
    if (
        np.any(kp <= 0.0)
        or np.any(kd < 0.0)
        or np.any(effort <= 0.0)
        or np.any(velocity <= 0.0)
        or np.any(action_scale <= 0.0)
        or np.any(armature < 0.0)
        or np.any(friction < 0.0)
        or np.any(qdes_limits[:, 0] >= qdes_limits[:, 1])
        or actuator_types != ["implicit"] * count
        or contract.get("action_use_default_offset") is not True
        or contract.get("joint_friction_backend") != "physx"
        or contract.get("joint_friction_semantics")
        != "load_dependent_spatial_force_coefficient"
        or contract.get("joint_friction_units") != "dimensionless"
        or contract.get("qdes_clamp") is not True
        or contract.get("finite_preclamp_qdes_projection_enabled") is not True
    ):
        raise DynamicReadyMaterializationError(
            "runtime contract has an invalid A3 implicit-PD/qdes contract"
        )
    inset = contract.get("finite_projection_soft_envelope_inset_fraction")
    if (
        isinstance(inset, bool)
        or type(inset) not in (int, float)
        or not math.isfinite(float(inset))
        or not 0.0 <= float(inset) < 0.5
    ):
        raise DynamicReadyMaterializationError(
            "runtime finite projection inset must lie in [0,0.5)"
        )
    physics_dt = float(contract.get("physics_step_dt_s", float("nan")))
    policy_dt = float(contract.get("policy_step_dt_s", float("nan")))
    decimation = contract.get("control_decimation")
    if (
        not math.isfinite(physics_dt)
        or physics_dt <= 0.0
        or not math.isfinite(policy_dt)
        or policy_dt <= 0.0
        or isinstance(decimation, bool)
        or type(decimation) is not int
        or decimation <= 0
        or not math.isclose(
            policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise DynamicReadyMaterializationError(
            "runtime physics/policy/decimation timing is inconsistent"
        )
    delay = contract.get("control_step_action_delay")
    if (
        type(delay) is not dict
        or delay.get("schema_version") != 1
        or delay.get("semantic_unit") != "policy_control_step"
        or delay.get("sample_timing") != "once_per_episode_reset"
        or delay.get("distribution") != "discrete_uniform_inclusive"
        or type(delay.get("enabled")) is not bool
        or isinstance(delay.get("min_steps"), bool)
        or type(delay.get("min_steps")) is not int
        or isinstance(delay.get("max_steps"), bool)
        or type(delay.get("max_steps")) is not int
        or delay["min_steps"] < 0
        or delay["max_steps"] < delay["min_steps"]
        or delay.get("shared_across_all_31_joints") is not True
        or delay.get("history_fill")
        != "safe_default_or_action_specific_hold"
        or delay["enabled"] != (delay["max_steps"] > 0)
    ):
        raise DynamicReadyMaterializationError(
            "runtime control-step action delay contract is invalid"
        )
    return {
        "joint_names": names,
        "kp": kp,
        "kd": kd,
        "effort": effort,
        "velocity": velocity,
        "default_q": default_q,
        "action_scale": action_scale,
        "qdes_limits": qdes_limits,
        "projection_inset": float(inset),
        "physics_dt": physics_dt,
        "policy_dt": policy_dt,
        "decimation": decimation,
        "control_step_action_delay": dict(delay),
        "actuator_types": actuator_types,
        "armature": armature,
        "friction": friction,
        "friction_backend": contract.get("joint_friction_backend"),
        "friction_semantics": contract.get("joint_friction_semantics"),
        "friction_units": contract.get("joint_friction_units"),
    }


def _bind_action_runtime(
    contract: Mapping[str, Any],
    *,
    action_id: str,
    motion_sha256: str,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        training = contract["action_ball_training"]
        preflight = training["preflight"]
        bootstrap = training["policy_bootstrap"]
        admission = training["motion_admission"]
        action_runtime = training["runtime"]
    except (KeyError, TypeError) as exc:
        raise DynamicReadyMaterializationError(
            "runtime contract has no exact ActionBall N=1 binding"
        ) from exc
    if (
        not isinstance(preflight, Mapping)
        or not isinstance(bootstrap, Mapping)
        or not isinstance(admission, Mapping)
        or not isinstance(action_runtime, Mapping)
        or preflight.get("action_order") != [action_id]
        or bootstrap.get("action_order") != [action_id]
        or admission.get("motion_file_sha256") != [motion_sha256]
        or action_runtime.get("action_order") != [action_id]
        or bootstrap.get("joint_names") != list(grounded.RUNTIME_JOINT_NAMES)
    ):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall action order, motion, or joint binding drifted"
        )
    bindings = preflight.get("action_bindings")
    ready_source = bootstrap.get("ready_source")
    decoder = bootstrap.get("decoder")
    guard = bootstrap.get("hard_inner_guard")
    runtime_bindings = action_runtime.get("bindings")
    if (
        preflight.get("schema_version") != 1
        or bootstrap.get("schema_version") != 1
        or bootstrap.get("kind")
        != "action_ball_shared_ready_actor_bootstrap_v1"
        or bootstrap.get("action_count") != 1
        or admission.get("schema_version") != 1
        or not isinstance(bindings, list)
        or len(bindings) != 1
        or not isinstance(bindings[0], Mapping)
        or bindings[0].get("action_id") != action_id
        or bindings[0].get("action_slot") != 0
        or bindings[0].get("motion_sha256") != motion_sha256
        or not isinstance(ready_source, Mapping)
        or ready_source.get("motion_sha256_per_action") != [motion_sha256]
        or not isinstance(decoder, Mapping)
        or decoder.get("use_default_offset") is not True
        or not isinstance(runtime_bindings, list)
        or len(runtime_bindings) != 1
        or not isinstance(runtime_bindings[0], Mapping)
        or runtime_bindings[0].get("action_slot") != 0
        or runtime_bindings[0].get("motion_sha256") != motion_sha256
        or not isinstance(guard, Mapping)
        or guard.get("limit_source")
        != "articulation.data.joint_pos_limits"
        or guard.get("margin_fraction") != 0.02
        or guard.get("margin_rad") != 0.0
    ):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall N=1 bootstrap does not bind this action motion"
        )
    hard_lower = _plain_finite_vector(
        guard.get("hard_inner_lower"),
        name="policy_bootstrap.hard_inner_lower",
        size=size,
    )
    hard_upper = _plain_finite_vector(
        guard.get("hard_inner_upper"),
        name="policy_bootstrap.hard_inner_upper",
        size=size,
    )
    if np.any(hard_lower >= hard_upper):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall hard-inner guard is empty"
        )
    shared_ready = _plain_finite_vector(
        ready_source.get("shared_ready_joint_pos"),
        name="policy_bootstrap.shared_ready_joint_pos",
        size=size,
    )
    decoder_default = _plain_finite_vector(
        decoder.get("default_joint_pos"),
        name="policy_bootstrap.decoder.default_joint_pos",
        size=size,
    )
    decoder_scale = _plain_finite_vector(
        decoder.get("action_scale"),
        name="policy_bootstrap.decoder.action_scale",
        size=size,
    )
    return (
        hard_lower,
        hard_upper,
        shared_ready,
        decoder_default,
        decoder_scale,
    )


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    motion_path, motion_sha = _pinned_file(
        args.motion,
        args.expected_motion_sha256,
        name="stable motion",
    )
    receipt_path, receipt_sha = _pinned_file(
        args.stable_receipt,
        args.expected_stable_receipt_sha256,
        name="stable receipt",
    )
    runtime_path, runtime_sha = _pinned_file(
        args.runtime_contract,
        args.expected_runtime_contract_sha256,
        name="runtime training contract",
    )
    mjcf_path, mjcf_sha = _pinned_file(
        args.mjcf,
        args.expected_mjcf_sha256,
        name="A3 MJCF",
    )
    stable_receipt = _read_json(receipt_path, name="stable receipt")
    runtime_contract = _read_json(runtime_path, name="runtime training contract")
    _validate_stable_receipt(stable_receipt, motion_sha256=motion_sha)
    if runtime_contract.get("target_mode") != "action_ball":
        raise DynamicReadyMaterializationError(
            "runtime contract is not an ActionBall contract"
        )

    ready_q, ready_root_pos, ready_root_quat = _load_motion_frame0(motion_path)
    plant = _runtime_plant(runtime_contract)
    (
        hard_inner_lower,
        hard_inner_upper,
        bootstrap_ready,
        bootstrap_default,
        bootstrap_scale,
    ) = _bind_action_runtime(
        runtime_contract,
        action_id=str(args.action_id),
        motion_sha256=motion_sha,
        size=len(plant["joint_names"]),
    )
    if (
        not np.array_equal(bootstrap_ready, ready_q)
        or not np.array_equal(bootstrap_default, plant["default_q"])
        or not np.array_equal(bootstrap_scale, plant["action_scale"])
    ):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall bootstrap decoder or ready pose differs "
            "from the physical motion/runtime plant"
        )
    identity = _exact_model_identity(
        stable_receipt, mjcf_path=mjcf_path, mjcf_sha256=mjcf_sha
    )
    if (
        stable_receipt["robot"].get("exact_xml_model_name")
        != identity.xml_model_name
    ):
        raise DynamicReadyMaterializationError(
            "stable receipt robot model name differs from its exact-model identity"
        )
    try:
        ready_root_z = runtime_contract["action_ball_training"]["preflight"][
            "ready_root_z_by_slot_m"
        ]
    except (KeyError, TypeError) as exc:
        raise DynamicReadyMaterializationError(
            "runtime ActionBall preflight has no ready-root binding"
        ) from exc
    if (
        not isinstance(ready_root_z, list)
        or len(ready_root_z) != 1
        or not math.isclose(
            float(ready_root_z[0]),
            float(ready_root_pos[2]),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
    ):
        raise DynamicReadyMaterializationError(
            "runtime ActionBall ready-root height differs from motion frame 0"
        )
    backend = grounded.MujocoGroundedReadyBackend.load(identity)
    ready = grounded.ReadyState(ready_q, ready_root_pos, ready_root_quat)
    qpos = backend._qpos(ready)

    ground_config = torque_topp.GroundContactConfig(
        expected_model_binding=identity.ground_model_binding_sha256,
        model_source_path=str(mjcf_path),
        expected_source_sha256=mjcf_sha,
    )
    solver = torque_topp.MujocoGroundContactLPSolver(
        backend.model, ground_config
    )
    actuator_contract = torque_topp.direct_actuator_contract_from_mujoco(
        backend.model,
        support_mode="ground",
        contact_mode="double_support_floor",
        fixed_lp_solver="scipy.optimize.linprog:highs",
    )
    (
        model_tau_lower,
        model_tau_upper,
        actuated,
        actuator_limit_report,
    ) = torque_topp._resolve_grounded_actuator_limits(
        actuator_contract, int(backend.model.nv)
    )
    model_row_for_runtime = (
        np.asarray(backend._binding.joint_dof_adrs, np.int64) - 6
    )
    expected_model_rows = np.arange(len(plant["joint_names"]), dtype=np.int64)
    if (
        model_row_for_runtime.shape != expected_model_rows.shape
        or not np.array_equal(np.sort(model_row_for_runtime), expected_model_rows)
        or not np.array_equal(
            np.asarray(actuated, np.int64),
            expected_model_rows + 6,
        )
    ):
        raise DynamicReadyMaterializationError(
            "exact A3 runtime-to-MuJoCo actuator rows are not one full permutation"
        )

    qdes_limits = plant["qdes_limits"]
    inset = plant["projection_inset"]
    span = qdes_limits[:, 1] - qdes_limits[:, 0]
    projected_soft_lower = qdes_limits[:, 0] + inset * span
    projected_soft_upper = qdes_limits[:, 1] - inset * span
    executed_qdes_lower = np.maximum(projected_soft_lower, hard_inner_lower)
    executed_qdes_upper = np.minimum(projected_soft_upper, hard_inner_upper)
    if np.any(executed_qdes_lower >= executed_qdes_upper):
        raise DynamicReadyMaterializationError(
            "runtime projected-soft and hard-inner qdes envelopes do not intersect"
        )
    if np.any(ready_q <= executed_qdes_lower) or np.any(
        ready_q >= executed_qdes_upper
    ):
        raise DynamicReadyMaterializationError(
            "physical ready lies outside the executed qdes envelope"
        )
    runtime_tau_lower = np.maximum(
        -plant["effort"], plant["kp"] * (executed_qdes_lower - ready_q)
    )
    runtime_tau_upper = np.minimum(
        plant["effort"], plant["kp"] * (executed_qdes_upper - ready_q)
    )
    runtime_tau_lower_model = np.empty_like(runtime_tau_lower)
    runtime_tau_upper_model = np.empty_like(runtime_tau_upper)
    runtime_tau_lower_model[model_row_for_runtime] = runtime_tau_lower
    runtime_tau_upper_model[model_row_for_runtime] = runtime_tau_upper
    hold_tau_lower_model = np.maximum(
        model_tau_lower, runtime_tau_lower_model
    )
    hold_tau_upper_model = np.minimum(
        model_tau_upper, runtime_tau_upper_model
    )
    if np.any(hold_tau_lower_model >= 0.0) or np.any(
        hold_tau_upper_model <= 0.0
    ):
        raise DynamicReadyMaterializationError(
            "hold torque envelope must contain zero on both sides"
        )

    solution = solver.solve(
        qpos,
        np.zeros(int(backend.model.nv), np.float64),
        np.zeros(int(backend.model.nv), np.float64),
        actuated,
        hold_tau_lower_model,
        hold_tau_upper_model,
        np.full(int(backend.model.nv), 1.0e6, np.float64),
        path_tangent=np.zeros(int(backend.model.nv), np.float64),
        lp_objective=LP_OBJECTIVE,
    )
    if not solution.feasible:
        raise DynamicReadyMaterializationError(
            "no static double-support hold exists inside the executed qdes envelope"
        )
    tau_model = np.asarray(solution.actuator_generalized_force, np.float64)
    if tau_model.shape != (31,) or not np.all(np.isfinite(tau_model)):
        raise DynamicReadyMaterializationError(
            "ground LP returned a malformed hold torque"
        )
    tau_runtime = tau_model[model_row_for_runtime]
    hold_qdes = ready_q + tau_runtime / plant["kp"]
    tolerance = 1.0e-10
    if np.any(hold_qdes < executed_qdes_lower - tolerance) or np.any(
        hold_qdes > executed_qdes_upper + tolerance
    ):
        raise DynamicReadyMaterializationError(
            "derived hold qdes lies outside the executed qdes envelope"
        )
    normalized_action = (
        hold_qdes - plant["default_q"]
    ) / plant["action_scale"]
    if not np.all(np.isfinite(normalized_action)):
        raise DynamicReadyMaterializationError(
            "derived normalized hold action is non-finite"
        )

    model_binding = solution.report.get("model_binding")
    if model_binding != identity.ground_model_binding_sha256:
        raise DynamicReadyMaterializationError(
            "ground LP result lost the exact model binding"
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "action_id": str(args.action_id),
        "robot": {
            "family": "AgiBot A3",
            "joint_names": list(plant["joint_names"]),
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
            "isaac_nominal_hold_validated": False,
        },
        "sources": {
            "stable_motion": {
                "path": str(motion_path),
                "sha256": motion_sha,
                "frame_index": 0,
            },
            "stable_receipt": {
                "path": str(receipt_path),
                "sha256": receipt_sha,
            },
            "runtime_training_contract": {
                "path": str(runtime_path),
                "sha256": runtime_sha,
            },
            "mujoco_model": {
                "path": str(mjcf_path),
                "sha256": mjcf_sha,
                "compiled_model_sha256": identity.compiled_model_sha256,
                "path_model_binding_sha256": (
                    identity.path_model_binding_sha256
                ),
                "ground_model_binding_sha256": (
                    identity.ground_model_binding_sha256
                ),
                "xml_model_name": identity.xml_model_name,
            },
        },
        "physical_ready": {
            "root_pos_w_m": ready_root_pos.tolist(),
            "root_quat_wxyz": ready_root_quat.tolist(),
            "joint_pos_rad": ready_q.tolist(),
            "joint_vel_radps": [0.0] * 31,
        },
        "runtime_plant": {
            "joint_names": list(plant["joint_names"]),
            "articulation_joint_names": list(plant["joint_names"]),
            "action_joint_ids": list(range(31)),
            "joint_stiffness": plant["kp"].tolist(),
            "joint_damping": plant["kd"].tolist(),
            "joint_effort_limits": plant["effort"].tolist(),
            "joint_velocity_limits": plant["velocity"].tolist(),
            "joint_actuator_types": plant["actuator_types"],
            "joint_armature": plant["armature"].tolist(),
            "joint_friction_coefficients": plant["friction"].tolist(),
            "joint_friction_backend": plant["friction_backend"],
            "joint_friction_semantics": plant["friction_semantics"],
            "joint_friction_units": plant["friction_units"],
            "qdes_joint_pos_limits": qdes_limits.tolist(),
            "finite_projection_soft_envelope_inset_fraction": inset,
            "projected_soft_qdes_lower_rad": projected_soft_lower.tolist(),
            "projected_soft_qdes_upper_rad": projected_soft_upper.tolist(),
            "hard_inner_qdes_lower_rad": hard_inner_lower.tolist(),
            "hard_inner_qdes_upper_rad": hard_inner_upper.tolist(),
            "executed_qdes_envelope_semantics": (
                "intersection(projected_soft_qdes,policy_bootstrap_hard_inner)"
            ),
            "executed_qdes_lower_rad": executed_qdes_lower.tolist(),
            "executed_qdes_upper_rad": executed_qdes_upper.tolist(),
            "default_joint_pos_rad": plant["default_q"].tolist(),
            "action_scale_rad": plant["action_scale"].tolist(),
            "physics_step_dt_s": plant["physics_dt"],
            "policy_step_dt_s": plant["policy_dt"],
            "control_decimation": plant["decimation"],
            "control_step_action_delay": plant[
                "control_step_action_delay"
            ],
        },
        "hold_candidate": {
            "semantics": (
                "tau_pd=kp*(qdes-physical_q) at zero joint velocity; "
                "MuJoCo contact LP initializes the candidate and Isaac must "
                "validate it"
            ),
            "lp_objective": LP_OBJECTIVE,
            "actuator_generalized_force_runtime_order_nm": (
                tau_runtime.tolist()
            ),
            "actuator_generalized_force_mujoco_row_order_nm": (
                tau_model.tolist()
            ),
            "hold_qdes_joint_pos_rad": hold_qdes.tolist(),
            "normalized_actor_action": normalized_action.tolist(),
            "mujoco_row_for_runtime_joint": model_row_for_runtime.tolist(),
            "mujoco_actuated_dof_indices": (
                np.asarray(actuated, np.int64).tolist()
            ),
            "model_tau_lower_mujoco_row_order_nm": model_tau_lower.tolist(),
            "model_tau_upper_mujoco_row_order_nm": model_tau_upper.tolist(),
            "runtime_tau_lower_runtime_order_nm": runtime_tau_lower.tolist(),
            "runtime_tau_upper_runtime_order_nm": runtime_tau_upper.tolist(),
            "runtime_tau_lower_mujoco_row_order_nm": (
                runtime_tau_lower_model.tolist()
            ),
            "runtime_tau_upper_mujoco_row_order_nm": (
                runtime_tau_upper_model.tolist()
            ),
            "effective_tau_lower_mujoco_row_order_nm": (
                hold_tau_lower_model.tolist()
            ),
            "effective_tau_upper_mujoco_row_order_nm": (
                hold_tau_upper_model.tolist()
            ),
            "actuator_limit_contract": actuator_limit_report,
            "solver_report": solution.report,
        },
        "required_next_gate": {
            "kind": "isaac_action_ball_nominal_hold_v1",
            "minimum_horizon_semantics": "validated_t_hit_plus_reaction_margin",
            "zero_terminal_required": [
                "joint_qdes_forbidden",
                "joint_actual_forbidden",
                "robot_hit_table",
                "base_fell_tilt",
                "base_too_low",
            ],
        },
        "non_claims": [
            "not an Isaac or PhysX closed-loop hold certificate",
            "not a training policy bootstrap until the nominal hold gate passes",
            "not deployment or hardware authorization",
        ],
        "producer": {
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
            "grounded_ready_tool_path": str(Path(grounded.__file__).resolve()),
            "grounded_ready_tool_sha256": _sha256_file(
                Path(grounded.__file__).resolve()
            ),
            "torque_lp_tool_path": str(Path(torque_topp.__file__).resolve()),
            "torque_lp_tool_sha256": _sha256_file(
                Path(torque_topp.__file__).resolve()
            ),
        },
    }
    unsigned = dict(result)
    content_sha = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    result["content_sha256"] = content_sha
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--expected-motion-sha256", required=True)
    parser.add_argument("--stable-receipt", required=True)
    parser.add_argument("--expected-stable-receipt-sha256", required=True)
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--expected-mjcf-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _materialize(args)
    output = _write_exclusive(args.output, _pretty_json_bytes(result))
    print(
        json.dumps(
            {
                "output": str(output),
                "content_sha256": result["content_sha256"],
                "action_id": result["action_id"],
                "objective": LP_OBJECTIVE,
                "max_hold_utilization": result["hold_candidate"][
                "solver_report"
                ].get("optimum_max_normalized_available_hold_torque"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
