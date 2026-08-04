#!/usr/bin/env python3
"""Materialize an exact-frame0 MuJoCo hold candidate and fixed probe tape.

This adapter consumes only the threshold-first, exact-measured-frame0 branch
of the Isaac dynamic-ready artifact.  It copies the stored teacher/physical
q, root and quaternion without substituting a historical lower-body seed,
constructs exact-zero reset velocities, and reuses the artifact's sealed hold
qdes/action/torque witness.  The output remains diagnostic-only.

The adapter deliberately rejects the lexicographic fallback, any nonzero
handoff duration, non-equal teacher/physical endpoints, different motion or
action lineage, and every source/SHA mismatch.  It does not run MuJoCo, Isaac,
training, deployment, or hardware commands.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import action_specific_hold as legacy_hold
from . import single_env as core


KIND = core.EXACT_FRAME0_ACTION_SPECIFIC_HOLD_KIND
SCHEMA_VERSION = core.EXACT_FRAME0_ACTION_SPECIFIC_HOLD_SCHEMA_VERSION
THRESHOLD_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/check_table_obstacle_scene.py"
)
PHYSICAL_RESET_SEMANTICS = "exact_measured_frame0_threshold_first"
CONTROLLER_BIRTH_SEMANTICS = "artifact_fresh_static_lp_hold_qdes"
HISTORY_FILL_SEMANTICS = "same_artifact_hold_action"
NON_CLAIMS = (
    "not a new Isaac or PhysX nominal-hold certificate",
    "not training, promotion, deployment, or hardware authorization",
    "no historical dynamic-ready physical-birth seed is consumed",
)

_THRESHOLD_MODULE_NAME = "_mujoco_exact_frame0_threshold_validator"


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise core.ContractError(f"cannot hash exact-frame0 source {path}: {exc}") from exc


def _expected_file_sha256(path: Path, expected: str, label: str) -> str:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise core.ContractError(f"expected {label} SHA-256 is invalid")
    actual = _file_sha256(path)
    if actual != expected:
        raise core.ContractError(
            f"{label} SHA-256 mismatch: actual={actual}, expected={expected}"
        )
    return actual


def _load_threshold_validator() -> Any:
    current_sha256 = _file_sha256(THRESHOLD_VALIDATOR_PATH)
    existing = sys.modules.get(_THRESHOLD_MODULE_NAME)
    if existing is not None:
        if (
            getattr(existing, "__exact_source_sha256__", None)
            != current_sha256
        ):
            raise core.ContractError(
                "threshold validator source drifted after module import"
            )
        return existing
    spec = importlib.util.spec_from_file_location(
        _THRESHOLD_MODULE_NAME, THRESHOLD_VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise core.ContractError("cannot load exact-frame0 threshold validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_THRESHOLD_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_THRESHOLD_MODULE_NAME, None)
        raise
    if _file_sha256(THRESHOLD_VALIDATOR_PATH) != current_sha256:
        sys.modules.pop(_THRESHOLD_MODULE_NAME, None)
        raise core.ContractError(
            "threshold validator source changed while it was imported"
        )
    setattr(module, "__exact_source_sha256__", current_sha256)
    return module


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise core.ContractError(f"{label} must contain {size} finite scalars")
    return array.copy()


def _source_row(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sources = document.get("sources")
    row = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(row, Mapping):
        raise core.ContractError(f"exact-frame0 artifact source {name!r} is missing")
    return row


def _resolved_source(row: Mapping[str, Any], label: str) -> tuple[Path, str]:
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise core.ContractError(f"exact-frame0 artifact {label} path is invalid")
    path = Path(raw_path).expanduser().resolve()
    expected = row.get("sha256")
    return path, _expected_file_sha256(path, expected, label)


def _same_array(actual: Any, expected: Any, size: int, label: str) -> np.ndarray:
    actual_array = _vector(actual, size, f"{label} actual")
    expected_array = _vector(expected, size, f"{label} expected")
    if not np.array_equal(actual_array, expected_array):
        raise core.ContractError(f"exact-frame0 {label} endpoints differ")
    return actual_array


def validate_exact_frame0_artifact(
    *,
    binding: core.PlantBinding,
    artifact_path: Path | str,
    expected_artifact_sha256: str,
    teacher_motion: Path | str,
    expected_teacher_motion_sha256: str,
    expected_action_id: str,
    mjcf_path: Path | str,
    expected_mjcf_sha256: str,
) -> dict[str, Any]:
    """Replay the exact threshold-first schema and return its sealed projection."""

    artifact = Path(artifact_path).expanduser().resolve()
    artifact_sha = _expected_file_sha256(
        artifact, expected_artifact_sha256, "threshold-first artifact"
    )
    teacher_path = Path(teacher_motion).expanduser().resolve()
    teacher_sha = _expected_file_sha256(
        teacher_path, expected_teacher_motion_sha256, "teacher motion"
    )
    mjcf = Path(mjcf_path).expanduser().resolve()
    mjcf_sha = _expected_file_sha256(mjcf, expected_mjcf_sha256, "root MJCF")
    if not isinstance(expected_action_id, str) or not expected_action_id:
        raise core.ContractError("expected exact-frame0 action id is invalid")

    validator = _load_threshold_validator()
    try:
        nominal = validator._load_nominal_hold_input(
            artifact, expected_sha256=artifact_sha
        )
    except Exception as exc:
        raise core.ContractError(
            f"threshold-first exact-frame0 artifact is invalid: {exc}"
        ) from exc
    document = nominal.document
    composition = document.get("physical_birth_composition")
    handoff = document.get("frame0_handoff")
    if (
        not isinstance(composition, Mapping)
        or composition.get("semantics")
        != validator.MEASURED_BIRTH_WHOLE_BODY_SAFE_FRAME0_SEMANTICS
        or composition.get("exact_measured_frame0_selected") is not True
        or nominal.teacher_physical_separated is not False
        or not isinstance(handoff, Mapping)
        or handoff.get("kind") != "exact_frame0_zero_duration_handoff_v1"
        or handoff.get("selection_semantics")
        != "threshold_first_exact_frame0_direct"
        or handoff.get("certified_transition_s") != 0.0
        or handoff.get("required_min_wait_s") != 0.0
        or handoff.get("endpoints_bitwise_equal") is not True
        or handoff.get("physical_ready_joint_velocity_exact_zero") is not True
        or handoff.get("teacher_static_endpoint_joint_velocity_exact_zero")
        is not True
    ):
        raise core.ContractError(
            "artifact is not the zero-duration threshold-first exact-frame0 branch"
        )
    if nominal.action_id != expected_action_id:
        raise core.ContractError(
            "threshold-first artifact belongs to a different action id"
        )
    if nominal.motion_path != teacher_path or nominal.motion_sha256 != teacher_sha:
        raise core.ContractError(
            "threshold-first artifact and requested teacher motion differ"
        )

    runtime_source = _source_row(document, "runtime_training_contract")
    runtime_path, runtime_sha = _resolved_source(
        runtime_source, "runtime training contract"
    )
    if (
        runtime_path != Path(binding.source_path).expanduser().resolve()
        or runtime_sha != binding.source_sha256
    ):
        raise core.ContractError(
            "threshold-first artifact belongs to a different plant contract"
        )
    model_source = _source_row(document, "mujoco_model")
    model_path, model_sha = _resolved_source(model_source, "root MJCF")
    if model_path != mjcf or model_sha != mjcf_sha:
        raise core.ContractError(
            "threshold-first artifact and requested root MJCF differ"
        )

    teacher_mapping, _teacher_center = core._teacher_frame_reset_payload(
        binding, teacher_path, 0
    )
    if teacher_mapping["source_motion_sha256"] != teacher_sha:
        raise core.ContractError("teacher motion replay lost its exact SHA-256")
    physical = document.get("physical_ready")
    teacher = document.get("teacher_reference")
    hold = document.get("hold_candidate")
    if not all(isinstance(row, Mapping) for row in (physical, teacher, hold)):
        raise core.ContractError("threshold-first artifact projection is incomplete")
    assert isinstance(physical, Mapping)
    assert isinstance(teacher, Mapping)
    assert isinstance(hold, Mapping)

    joint_pos = _same_array(
        physical.get("joint_pos_rad"),
        teacher_mapping["joint_pos"],
        core.ACTION_DIM,
        "physical/teacher joint position",
    )
    _same_array(
        teacher.get("joint_pos_rad"),
        teacher_mapping["joint_pos"],
        core.ACTION_DIM,
        "artifact/motion teacher joint position",
    )
    root_pos = _same_array(
        physical.get("root_pos_w_m"),
        teacher_mapping["root_pos"],
        3,
        "physical/teacher root position",
    )
    _same_array(
        teacher.get("root_pos_w_m"),
        teacher_mapping["root_pos"],
        3,
        "artifact/motion teacher root position",
    )
    root_quat = _same_array(
        physical.get("root_quat_wxyz"),
        teacher_mapping["root_quat_wxyz"],
        4,
        "physical/teacher stored root quaternion",
    )
    _same_array(
        teacher.get("root_quat_wxyz"),
        teacher_mapping["root_quat_wxyz"],
        4,
        "artifact/motion teacher stored root quaternion",
    )
    if not math.isclose(
        float(np.linalg.norm(root_quat)), 1.0, rel_tol=0.0, abs_tol=2.0e-6
    ):
        raise core.ContractError("stored exact-frame0 root quaternion is invalid")
    joint_vel = _vector(
        physical.get("joint_vel_radps"), core.ACTION_DIM, "physical joint velocity"
    )
    teacher_static_vel = _vector(
        teacher.get("static_handoff_joint_vel_radps"),
        core.ACTION_DIM,
        "teacher static-handoff joint velocity",
    )
    if not np.array_equal(joint_vel, np.zeros(core.ACTION_DIM)) or not np.array_equal(
        teacher_static_vel, np.zeros(core.ACTION_DIM)
    ):
        raise core.ContractError("exact-frame0 static endpoint velocity is nonzero")

    hold_qdes = _vector(
        hold.get("hold_qdes_joint_pos_rad"), core.ACTION_DIM, "artifact hold qdes"
    )
    hold_action = _vector(
        hold.get("normalized_actor_action"), core.ACTION_DIM, "artifact hold action"
    )
    hold_force = _vector(
        hold.get("actuator_generalized_force_runtime_order_nm"),
        core.ACTION_DIM,
        "artifact hold force",
    )
    _raw_qdes, decoded_qdes, clamps = binding.decode_action(hold_action)
    expected_force = binding.stiffness * (hold_qdes - joint_pos)
    raw_force, applied_force, force_clamps = core.total_pd_effort(
        binding, hold_qdes, joint_pos, np.zeros(core.ACTION_DIM)
    )
    if (
        clamps != 0
        or not np.allclose(decoded_qdes, hold_qdes, rtol=0.0, atol=2.0e-10)
        or not np.allclose(hold_force, expected_force, rtol=0.0, atol=2.0e-10)
        or not np.allclose(raw_force, hold_force, rtol=0.0, atol=2.0e-10)
        or not np.array_equal(applied_force, raw_force)
        or force_clamps != 0
    ):
        raise core.ContractError(
            "threshold-first hold qdes/action/torque identity differs"
        )

    joint_names = tuple(str(name) for name in nominal.joint_names)
    if joint_names != binding.joint_names:
        raise core.ContractError("threshold-first artifact joint order differs")
    leg_names = frozenset(str(name) for name in validator._A3_LEG_JOINT_NAMES)
    leg_indices = tuple(
        index for index, name in enumerate(binding.joint_names) if name in leg_names
    )
    if len(leg_indices) != 12:
        raise core.ContractError("canonical exact-frame0 leg partition is not 12-D")

    content_sha = document.get("content_sha256")
    if (
        not isinstance(content_sha, str)
        or len(content_sha) != 64
        or any(character not in "0123456789abcdef" for character in content_sha)
    ):
        raise core.ContractError("threshold-first artifact content seal is invalid")
    return {
        "artifact_path": artifact,
        "artifact_sha256": artifact_sha,
        "artifact_content_sha256": content_sha,
        "action_id": nominal.action_id,
        "teacher_path": teacher_path,
        "teacher_sha256": teacher_sha,
        "teacher_mapping": teacher_mapping,
        "mjcf_path": mjcf,
        "mjcf_sha256": mjcf_sha,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos": root_pos,
        "root_quat_wxyz": root_quat,
        "hold_qdes": hold_qdes,
        "hold_action": hold_action,
        "hold_force": hold_force,
        "leg_indices": leg_indices,
        "frame0_handoff": copy.deepcopy(dict(handoff)),
        "static_evidence": copy.deepcopy(
            dict(document["physical_birth_static_evidence"])
        ),
        "threshold_validator_sha256": _file_sha256(THRESHOLD_VALIDATOR_PATH),
    }


def build_candidate(
    *,
    binding: core.PlantBinding,
    artifact_path: Path | str,
    expected_artifact_sha256: str,
    teacher_motion: Path | str,
    expected_teacher_motion_sha256: str,
    expected_action_id: str,
    mjcf_path: Path | str,
    expected_mjcf_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic exact-frame0 candidate from one artifact."""

    validated = validate_exact_frame0_artifact(
        binding=binding,
        artifact_path=artifact_path,
        expected_artifact_sha256=expected_artifact_sha256,
        teacher_motion=teacher_motion,
        expected_teacher_motion_sha256=expected_teacher_motion_sha256,
        expected_action_id=expected_action_id,
        mjcf_path=mjcf_path,
        expected_mjcf_sha256=expected_mjcf_sha256,
    )
    teacher = validated["teacher_mapping"]
    leg_indices = validated["leg_indices"]
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
                "path": legacy_hold._repo_relative_logical_path(
                    binding.source_path, "training_contract"
                ),
                "sha256": binding.source_sha256,
            },
            "teacher_motion": {
                "path": legacy_hold._repo_relative_logical_path(
                    validated["teacher_path"], "teacher_motion"
                ),
                "sha256": validated["teacher_sha256"],
                "uid": teacher["source_motion_uid"],
                "frame": 0,
                "joint_order_contract_id": teacher[
                    "source_joint_order_contract_id"
                ],
                "joint_order_contract_sha256": teacher[
                    "source_joint_order_contract_sha256"
                ],
            },
            "exact_frame0_threshold_first_artifact": {
                "path": legacy_hold._repo_relative_logical_path(
                    validated["artifact_path"],
                    "exact_frame0_threshold_first_artifact",
                ),
                "sha256": validated["artifact_sha256"],
                "content_sha256": validated["artifact_content_sha256"],
                "action_id": validated["action_id"],
                "motion_sha256": validated["teacher_sha256"],
                "physical_ready_state_sha256": validated["frame0_handoff"][
                    "physical_ready_state_sha256"
                ],
                "teacher_frame0_state_sha256": validated["frame0_handoff"][
                    "teacher_frame0_state_sha256"
                ],
                "consumed_fields": [
                    "physical_ready.joint_pos_rad",
                    "physical_ready.root_pos_w_m",
                    "physical_ready.root_quat_wxyz",
                    "physical_ready.joint_vel_radps",
                    "hold_candidate.hold_qdes_joint_pos_rad",
                    "hold_candidate.normalized_actor_action",
                    "hold_candidate.actuator_generalized_force_runtime_order_nm",
                    "frame0_handoff",
                ],
            },
            "root_mjcf": {
                "path": legacy_hold._repo_relative_logical_path(
                    validated["mjcf_path"], "root_mjcf"
                ),
                "sha256": validated["mjcf_sha256"],
            },
        },
        "semantics": {
            "teacher_reference_unchanged": True,
            "physical_reset": PHYSICAL_RESET_SEMANTICS,
            "controller_birth_target": CONTROLLER_BIRTH_SEMANTICS,
            "history_fill": HISTORY_FILL_SEMANTICS,
            "teacher_and_physical_reset_may_differ": False,
            "threshold_first_fallback_used": False,
            "certified_transition_s": 0.0,
        },
        "physical_ready": {
            "joint_pos": validated["joint_pos"].tolist(),
            "joint_vel": [0.0] * core.ACTION_DIM,
            "root_pos": validated["root_pos"].tolist(),
            "root_quat_wxyz": validated["root_quat_wxyz"].tolist(),
            "root_lin_vel_w": [0.0, 0.0, 0.0],
            "root_ang_vel_w": [0.0, 0.0, 0.0],
            "root_lin_vel_point": "link_origin",
            "leg_joint_indices": list(leg_indices),
            "leg_joint_names": [binding.joint_names[index] for index in leg_indices],
            "nonleg_exact_teacher_q0": True,
            "all_joints_exact_teacher_q0": True,
            "root_exact_teacher_frame0": True,
            "stored_quaternion_unchanged": True,
        },
        "hold": {
            "joint_qdes": validated["hold_qdes"].tolist(),
            "normalized_action": validated["hold_action"].tolist(),
            "actuator_force_runtime_nm": validated["hold_force"].tolist(),
            "maximum_abs_normalized_action": float(
                np.max(np.abs(validated["hold_action"]))
            ),
            "maximum_abs_actuator_force_nm": float(
                np.max(np.abs(validated["hold_force"]))
            ),
        },
        "static_evidence": {
            "authority": "sealed_threshold_first_exact_frame0_artifact",
            "threshold_validator": {
                "path": legacy_hold._repo_relative_logical_path(
                    THRESHOLD_VALIDATOR_PATH, "threshold_validator"
                ),
                "sha256": validated["threshold_validator_sha256"],
            },
            "artifact_content_sha256": validated["artifact_content_sha256"],
            "frame0_handoff": validated["frame0_handoff"],
            "source_static_evidence": validated["static_evidence"],
            "isaac_nominal_hold_validated": False,
        },
        "non_claims": list(NON_CLAIMS),
    }
    candidate["content_sha256"] = core._sha256(
        core._canonical_json_bytes(candidate)
    )
    return candidate


def rebuild_candidate_from_sources(
    binding: core.PlantBinding, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct a candidate solely from its four replayed sources."""

    sources = candidate.get("sources")
    if not isinstance(sources, Mapping):
        raise core.ContractError("exact-frame0 candidate sources are missing")

    def resolved(name: str) -> tuple[Path, Mapping[str, Any]]:
        row = sources.get(name)
        if not isinstance(row, Mapping):
            raise core.ContractError(f"exact-frame0 candidate source {name!r} is missing")
        return core._resolve_action_specific_hold_logical_path(
            row.get("path"), name
        ), row

    artifact_path, artifact_row = resolved(
        "exact_frame0_threshold_first_artifact"
    )
    teacher_path, teacher_row = resolved("teacher_motion")
    mjcf_path, mjcf_row = resolved("root_mjcf")
    return build_candidate(
        binding=binding,
        artifact_path=artifact_path,
        expected_artifact_sha256=artifact_row.get("sha256"),
        teacher_motion=teacher_path,
        expected_teacher_motion_sha256=teacher_row.get("sha256"),
        expected_action_id=artifact_row.get("action_id"),
        mjcf_path=mjcf_path,
        expected_mjcf_sha256=mjcf_row.get("sha256"),
    )


def materialize_candidate_and_tape(
    *,
    binding: core.PlantBinding,
    artifact_path: Path | str,
    expected_artifact_sha256: str,
    teacher_motion: Path | str,
    expected_teacher_motion_sha256: str,
    expected_action_id: str,
    mjcf_path: Path | str,
    expected_mjcf_sha256: str,
    candidate_output: Path | str,
    tape_output: Path | str,
) -> dict[str, Any]:
    """No-clobber materialize the candidate and its delay-zero probe tape."""

    candidate_path = Path(candidate_output).expanduser().resolve()
    tape_path = Path(tape_output).expanduser().resolve()
    if candidate_path == tape_path:
        raise core.ContractError("candidate and tape outputs must differ")
    if candidate_path.exists() or tape_path.exists():
        raise core.ContractError("exact-frame0 materializer refuses existing outputs")
    candidate = build_candidate(
        binding=binding,
        artifact_path=artifact_path,
        expected_artifact_sha256=expected_artifact_sha256,
        teacher_motion=teacher_motion,
        expected_teacher_motion_sha256=expected_teacher_motion_sha256,
        expected_action_id=expected_action_id,
        mjcf_path=mjcf_path,
        expected_mjcf_sha256=expected_mjcf_sha256,
    )
    candidate_raw = core._canonical_json_bytes(candidate)
    candidate_sha = hashlib.sha256(candidate_raw).hexdigest()
    wrote_candidate = False
    try:
        core._write_new_bytes(candidate_path, candidate_raw)
        wrote_candidate = True
        tape = core.build_probe_tape(
            binding,
            delay_steps=0,
            teacher_motion=teacher_motion,
            teacher_frame_index=0,
            hold_candidate=candidate_path,
            expected_hold_candidate_sha256=candidate_sha,
        )
        tape_sha = core.write_fixed_tape(tape_path, tape)
    except Exception:
        if wrote_candidate:
            try:
                candidate_path.unlink()
            except OSError:
                pass
        raise
    return {
        "schema_version": 1,
        "kind": "a3_mujoco_exact_frame0_candidate_tape_materialization_v1",
        "status": "EXACT_FRAME0_CANDIDATE_AND_TAPE_WRITTEN",
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "candidate_content_sha256": candidate["content_sha256"],
        "tape_path": str(tape_path),
        "tape_sha256": tape_sha,
        "artifact_sha256": expected_artifact_sha256,
        "teacher_motion_sha256": expected_teacher_motion_sha256,
        "action_id": expected_action_id,
        "delay_steps": 0,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--threshold-artifact", type=Path, required=True)
    parser.add_argument("--expected-threshold-artifact-sha256", required=True)
    parser.add_argument("--teacher-motion", type=Path, required=True)
    parser.add_argument("--expected-teacher-motion-sha256", required=True)
    parser.add_argument("--expected-action-id", required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--expected-mjcf-sha256", required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--tape-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = args.contract.expanduser().resolve()
    _expected_file_sha256(
        contract, args.expected_contract_sha256, "training contract"
    )
    binding = core.load_plant_binding(contract)
    result = materialize_candidate_and_tape(
        binding=binding,
        artifact_path=args.threshold_artifact,
        expected_artifact_sha256=args.expected_threshold_artifact_sha256,
        teacher_motion=args.teacher_motion,
        expected_teacher_motion_sha256=args.expected_teacher_motion_sha256,
        expected_action_id=args.expected_action_id,
        mjcf_path=args.mjcf,
        expected_mjcf_sha256=args.expected_mjcf_sha256,
        candidate_output=args.candidate_output,
        tape_output=args.tape_output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
