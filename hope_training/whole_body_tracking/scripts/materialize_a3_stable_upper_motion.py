#!/usr/bin/env python3
"""Materialize an A3 upper motion on the runtime's stable standing birth.

The historical upper clips keep useful waist/head/arm motion, but their
physical birth uses a deeply crouched, pitched lower-body pose that is not a
closed-loop hold state for the AgiBot A3 training plant.  This producer makes
the smallest coherent replacement:

* preserve every head/arm joint position and velocity;
* replace the twelve leg joint positions with the exact runtime default stand
  and set their velocities to zero;
* rebase each waist trajectory from its source frame-0 offset onto the exact
  runtime ready value, preserving every within-clip delta and velocity;
* preserve root X/Y and source yaw, while using the runtime stand height and an
  upright root;
* rebuild all schema-2 body kinematics with the exact pinned A3 MJCF;
* record the resulting racket-site change so downstream ball/task bindings
  must be rematerialized rather than silently reused.

This is a diagnostic asset producer, not a training or hardware authorization.
Isaac closed-loop hold, table contact, task alignment, policy and curriculum
evidence remain downstream gates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import canonical_schema2_builder as schema2
import materialize_grounded_upper_motion as base


TOOL_ID = "a3_stable_upper_motion_materializer_v2"
ARTIFACT_CLASS = "diagnostic_a3_stable_upper_motion_v2"
VERDICT = "PASS_DIAGNOSTIC_A3_STABLE_UPPER_WAIST_REBASED_REBUILD"

LEG_JOINT_NAMES = tuple(base.LEG_JOINT_NAMES)
LEG_JOINT_INDICES = tuple(base.LEG_JOINT_INDICES)
RUNTIME_JOINT_NAMES = tuple(base.RUNTIME_JOINT_NAMES)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
WAIST_JOINT_INDICES = tuple(
    RUNTIME_JOINT_NAMES.index(name) for name in WAIST_JOINT_NAMES
)
PRESERVED_JOINT_INDICES = tuple(
    index
    for index in range(31)
    if index not in set(LEG_JOINT_INDICES) | set(WAIST_JOINT_INDICES)
)

_STABLE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "root_height_m",
        "lower_joint_pos_rad",
        "waist_ready_pos_rad",
        "provenance",
    }
)


class StableUpperMaterializationError(RuntimeError):
    """The stable-upper replacement cannot satisfy its exact contract."""


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise StableUpperMaterializationError(
            f"{label} must be a real array with shape {shape}, got "
            f"{array.shape}/{array.dtype}"
        )
    if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
        raise StableUpperMaterializationError(f"{label} contains NaN/Inf")
    return array


def _load_stable_contract(path: Path) -> dict[str, Any]:
    payload = base._read_json(path, "A3 stable-stand contract")
    if frozenset(payload) != _STABLE_CONTRACT_KEYS:
        raise StableUpperMaterializationError(
            "A3 stable-stand contract keyset mismatch"
        )
    if payload["schema_version"] != 2:
        raise StableUpperMaterializationError(
            "A3 stable-stand contract schema_version must equal 2"
        )
    root_height = payload["root_height_m"]
    if (
        isinstance(root_height, bool)
        or not isinstance(root_height, (int, float))
        or not math.isfinite(float(root_height))
        or float(root_height) <= 0.0
    ):
        raise StableUpperMaterializationError(
            "A3 stable-stand root_height_m must be finite and positive"
        )
    lower = payload["lower_joint_pos_rad"]
    if not isinstance(lower, Mapping) or set(lower) != set(LEG_JOINT_NAMES):
        raise StableUpperMaterializationError(
            "A3 stable-stand lower_joint_pos_rad must contain exactly the "
            "twelve runtime leg joints"
        )
    values: dict[str, float] = {}
    for name in LEG_JOINT_NAMES:
        value = lower[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise StableUpperMaterializationError(
                f"A3 stable-stand {name} must be finite"
            )
        values[name] = float(value)
    waist = payload["waist_ready_pos_rad"]
    if not isinstance(waist, Mapping) or set(waist) != set(WAIST_JOINT_NAMES):
        raise StableUpperMaterializationError(
            "A3 stable-stand waist_ready_pos_rad must contain exactly the "
            "three runtime waist joints"
        )
    waist_values: dict[str, float] = {}
    for name in WAIST_JOINT_NAMES:
        value = waist[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise StableUpperMaterializationError(
                f"A3 stable-stand {name} ready value must be finite"
            )
        waist_values[name] = float(value)
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise StableUpperMaterializationError(
            "A3 stable-stand provenance must be a non-empty object"
        )
    return {
        "root_height_m": float(root_height),
        "lower_joint_pos_rad": values,
        "waist_ready_pos_rad": waist_values,
        "provenance": dict(provenance),
    }


def _validate_input_motion(
    arrays: Mapping[str, np.ndarray], *, strike_frame: int
) -> dict[str, Any]:
    if frozenset(arrays) not in schema2.ALLOWED_KEYSETS:
        raise StableUpperMaterializationError(
            f"input motion must have an exact schema-2 keyset, got {sorted(arrays)}"
        )
    q_raw = np.asarray(arrays["joint_pos"])
    if q_raw.ndim != 2 or q_raw.shape[1] != 31 or q_raw.shape[0] < 2:
        raise StableUpperMaterializationError(
            f"joint_pos must have shape (T>=2,31), got {q_raw.shape}"
        )
    frames = int(q_raw.shape[0])
    q = _finite_array(q_raw, (frames, 31), "motion joint_pos")
    qd = _finite_array(arrays["joint_vel"], (frames, 31), "motion joint_vel")
    body_pos = _finite_array(
        arrays["body_pos_w"], (frames, 32, 3), "motion body_pos_w"
    )
    body_quat = _finite_array(
        arrays["body_quat_w"], (frames, 32, 4), "motion body_quat_w"
    )
    _finite_array(
        arrays["body_lin_vel_w"], (frames, 32, 3), "motion body_lin_vel_w"
    )
    _finite_array(
        arrays["body_ang_vel_w"], (frames, 32, 3), "motion body_ang_vel_w"
    )
    body_names = tuple(
        str(value) for value in np.asarray(arrays["body_names"]).tolist()
    )
    if len(body_names) != 32 or len(set(body_names)) != 32:
        raise StableUpperMaterializationError(
            "motion body_names must contain 32 unique names"
        )
    if int(np.asarray(arrays["kinematics_schema_version"]).reshape(-1)[0]) != 2:
        raise StableUpperMaterializationError(
            "motion kinematics_schema_version must equal 2"
        )
    if str(np.asarray(arrays["body_pos_point"]).item()) != "link_origin":
        raise StableUpperMaterializationError(
            "motion body_pos_point must equal link_origin"
        )
    if str(np.asarray(arrays["body_lin_vel_point"]).item()) != "center_of_mass":
        raise StableUpperMaterializationError(
            "motion body_lin_vel_point must equal center_of_mass"
        )
    fps_array = np.asarray(arrays["fps"])
    if fps_array.size != 1:
        raise StableUpperMaterializationError("motion fps must be scalar")
    fps = float(fps_array.reshape(-1)[0])
    if not math.isfinite(fps) or fps <= 0.0:
        raise StableUpperMaterializationError("motion fps must be positive")
    if strike_frame < 0 or strike_frame >= frames:
        raise StableUpperMaterializationError(
            f"strike frame {strike_frame} lies outside [0,{frames - 1}]"
        )
    if not np.array_equal(q[0], q[-1]):
        raise StableUpperMaterializationError(
            "upper motion first/last joint_pos must be bitwise identical"
        )
    leg = np.asarray(LEG_JOINT_INDICES, dtype=np.int64)
    if not np.array_equal(q[:, leg], np.broadcast_to(q[0, leg], q[:, leg].shape)):
        raise StableUpperMaterializationError(
            "upper motion must have bitwise-constant leg joint positions"
        )
    root_pos = body_pos[:, 0]
    root_quat = body_quat[:, 0]
    if not np.array_equal(
        root_pos, np.broadcast_to(root_pos[0], root_pos.shape)
    ) or not np.array_equal(
        root_quat, np.broadcast_to(root_quat[0], root_quat.shape)
    ):
        raise StableUpperMaterializationError(
            "upper motion root pose must be bitwise constant"
        )
    if np.count_nonzero(qd[[0, -1]]) != 0:
        raise StableUpperMaterializationError(
            "upper motion joint_vel endpoints must be exact zero"
        )
    return {
        "frames": frames,
        "fps": fps,
        "strike_frame": int(strike_frame),
        "joint_pos": q,
        "joint_vel": qd,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_names": body_names,
    }


def _yaw_only_quaternion_wxyz(value: np.ndarray) -> tuple[np.ndarray, float]:
    quat = _finite_array(value, (4,), "source root quaternion").astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise StableUpperMaterializationError("source root quaternion is zero")
    w, x, y, z = quat / norm
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    result = np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )
    return result, float(yaw)


def _replace_with_stable_stand(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    stable_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    q = np.asarray(joint_pos)
    qd = np.asarray(joint_vel)
    root_pos = np.asarray(root_pos_w)
    root_quat = np.asarray(root_quat_wxyz)
    frames = int(q.shape[0])
    if (
        q.shape != (frames, 31)
        or qd.shape != q.shape
        or root_pos.shape != (frames, 3)
        or root_quat.shape != (frames, 4)
    ):
        raise StableUpperMaterializationError(
            "stable replacement input shapes do not match"
        )
    q_out = np.array(q, copy=True)
    qd_out = np.array(qd, copy=True)
    for name, index in zip(LEG_JOINT_NAMES, LEG_JOINT_INDICES):
        q_out[:, index] = np.asarray(
            stable_contract["lower_joint_pos_rad"][name], dtype=q_out.dtype
        )
        qd_out[:, index] = np.asarray(0.0, dtype=qd_out.dtype)
    for name, index in zip(WAIST_JOINT_NAMES, WAIST_JOINT_INDICES):
        ready = np.asarray(
            stable_contract["waist_ready_pos_rad"][name], dtype=q_out.dtype
        )
        q_out[:, index] = (q[:, index] - q[0, index]) + ready
    root_pos_out = np.array(root_pos, copy=True)
    root_pos_out[:, 2] = np.asarray(
        stable_contract["root_height_m"], dtype=root_pos_out.dtype
    )
    yaw_quat, yaw = _yaw_only_quaternion_wxyz(
        np.asarray(root_quat[0], dtype=np.float64)
    )
    root_quat_out = np.broadcast_to(
        yaw_quat.astype(root_quat.dtype), root_quat.shape
    ).copy()
    preserved = np.asarray(PRESERVED_JOINT_INDICES, dtype=np.int64)
    if not np.array_equal(q_out[:, preserved], q[:, preserved]):
        raise AssertionError(
            "stable replacement changed a preserved head/arm joint position"
        )
    if not np.array_equal(qd_out[:, preserved], qd[:, preserved]):
        raise AssertionError(
            "stable replacement changed a preserved head/arm joint velocity"
        )
    waist = np.asarray(WAIST_JOINT_INDICES, dtype=np.int64)
    source_waist_delta = q[:, waist] - q[[0], waist]
    output_waist_delta = q_out[:, waist] - q_out[[0], waist]
    if not np.array_equal(output_waist_delta, source_waist_delta):
        raise AssertionError("stable replacement changed a waist trajectory delta")
    if not np.array_equal(qd_out[:, waist], qd[:, waist]):
        raise AssertionError("stable replacement changed a waist joint velocity")
    if not np.array_equal(root_pos_out[:, :2], root_pos[:, :2]):
        raise AssertionError("stable replacement changed root X/Y")
    return q_out, qd_out, root_pos_out, root_quat_out, yaw


def _site_delta(before: base.SiteTrace, after: base.SiteTrace) -> dict[str, Any]:
    rows = (
        ("position_w", before.position_w, after.position_w, "m"),
        ("rotation_w", before.rotation_w, after.rotation_w, "matrix"),
        (
            "linear_velocity_w",
            before.linear_velocity_w,
            after.linear_velocity_w,
            "m/s",
        ),
        (
            "angular_velocity_w",
            before.angular_velocity_w,
            after.angular_velocity_w,
            "rad/s",
        ),
    )
    report: dict[str, Any] = {}
    for label, left, right, unit in rows:
        report[label] = {
            "bitwise_equal": bool(np.array_equal(left, right)),
            "maximum_abs_delta": float(np.max(np.abs(left - right))),
            "unit": unit,
            "before_sha256": base._hash_array(
                f"{base.SITE_NAME}.{label}", left
            ),
            "after_sha256": base._hash_array(
                f"{base.SITE_NAME}.{label}", right
            ),
        }
    return report


def materialize(args: argparse.Namespace) -> base.PublishedBundle:
    input_path, input_sha = base._pinned_regular_file(
        args.input_motion, args.expected_input_sha256, "input motion"
    )
    stable_path, stable_sha = base._pinned_regular_file(
        args.stable_stand_contract,
        args.expected_stable_stand_contract_sha256,
        "A3 stable-stand contract",
    )
    reference_candidate_path, reference_candidate_sha = base._pinned_regular_file(
        args.grounded_reference_candidate,
        args.expected_grounded_reference_candidate_sha256,
        "grounded reference candidate",
    )
    reference_receipt_path, reference_receipt_sha = base._pinned_regular_file(
        args.grounded_reference_receipt,
        args.expected_grounded_reference_receipt_sha256,
        "grounded reference receipt",
    )
    body_order_path, body_order_sha = base._pinned_regular_file(
        args.body_order, args.expected_body_order_sha256, "runtime body-order"
    )

    input_arrays = base._copy_npz(input_path, "input motion")
    reference_arrays = base._copy_npz(
        reference_candidate_path, "grounded reference candidate"
    )
    reference_receipt = base._read_json(
        reference_receipt_path, "grounded reference receipt"
    )
    stable = _load_stable_contract(stable_path)
    motion = _validate_input_motion(
        input_arrays, strike_frame=int(args.strike_frame)
    )
    exact_identity = base._validate_grounded_reference(
        reference_arrays,
        reference_receipt,
        candidate_sha256=reference_candidate_sha,
    )
    if tuple(motion["body_names"]) != base._body_order_names(body_order_path):
        raise StableUpperMaterializationError(
            "input motion body_names differs from pinned runtime body-order"
        )

    q_out, qd_out, root_pos_out, root_quat_out, source_yaw = (
        _replace_with_stable_stand(
            joint_pos=motion["joint_pos"],
            joint_vel=motion["joint_vel"],
            root_pos_w=motion["body_pos_w"][:, 0],
            root_quat_wxyz=motion["body_quat_w"][:, 0],
            stable_contract=stable,
        )
    )
    zeros3 = np.zeros((motion["frames"], 3), dtype=np.float64)
    candidate = schema2.build_schema2_candidate(
        joint_pos=q_out,
        joint_vel=qd_out,
        root_pos_w=root_pos_out,
        root_quat_wxyz=root_quat_out,
        root_lin_vel_w=zeros3,
        root_ang_vel_w=zeros3,
        fps=motion["fps"],
        mjcf_path=exact_identity.mjcf_path,
        input_sha256=input_sha,
        ready_sha256=stable_sha,
        body_order_path=body_order_path,
        migration_provenance=base._migration_provenance(input_arrays),
    )
    output = candidate.arrays
    preserved = np.asarray(PRESERVED_JOINT_INDICES, dtype=np.int64)
    waist = np.asarray(WAIST_JOINT_INDICES, dtype=np.int64)
    leg = np.asarray(LEG_JOINT_INDICES, dtype=np.int64)
    if not np.array_equal(
        output["joint_pos"][:, preserved],
        input_arrays["joint_pos"][:, preserved],
    ):
        raise StableUpperMaterializationError(
            "schema-2 rebuild changed preserved head/arm joint positions"
        )
    if not np.array_equal(
        output["joint_vel"][:, preserved],
        input_arrays["joint_vel"][:, preserved],
    ):
        raise StableUpperMaterializationError(
            "schema-2 rebuild changed preserved head/arm joint velocities"
        )
    if not np.array_equal(
        output["joint_pos"][:, waist] - output["joint_pos"][[0], waist],
        input_arrays["joint_pos"][:, waist]
        - input_arrays["joint_pos"][[0], waist],
    ):
        raise StableUpperMaterializationError(
            "schema-2 rebuild changed waist trajectory deltas"
        )
    if not np.array_equal(
        output["joint_vel"][:, waist], input_arrays["joint_vel"][:, waist]
    ):
        raise StableUpperMaterializationError(
            "schema-2 rebuild changed waist joint velocities"
        )
    if np.count_nonzero(output["joint_vel"][:, leg]) != 0:
        raise StableUpperMaterializationError(
            "schema-2 rebuild emitted non-zero leg velocity"
        )

    backend = base.grounded.MujocoGroundedReadyBackend.load(exact_identity)
    ready_state = base.grounded.ReadyState(
        np.asarray(output["joint_pos"][0], dtype=np.float64),
        np.asarray(output["body_pos_w"][0, 0], dtype=np.float64),
        np.asarray(output["body_quat_w"][0, 0], dtype=np.float64),
    )
    grounding_audit = base._audit_input_grounded_state(backend, ready_state)
    before_trace = base._model_site_trace(
        mujoco=backend._mujoco,
        model=backend.model,
        joint_pos=np.asarray(input_arrays["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(input_arrays["joint_vel"], dtype=np.float64),
        root_pos_w=np.asarray(input_arrays["body_pos_w"][:, 0], dtype=np.float64),
        root_quat_wxyz=np.asarray(
            input_arrays["body_quat_w"][:, 0], dtype=np.float64
        ),
    )
    after_trace = base._model_site_trace(
        mujoco=backend._mujoco,
        model=backend.model,
        joint_pos=np.asarray(output["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(output["joint_vel"], dtype=np.float64),
        root_pos_w=np.asarray(output["body_pos_w"][:, 0], dtype=np.float64),
        root_quat_wxyz=np.asarray(output["body_quat_w"][:, 0], dtype=np.float64),
    )
    site_change = _site_delta(before_trace, after_trace)
    strike = int(motion["strike_frame"])
    motion_filename = input_path.stem + ".a3_stable_upper_v2.npz"
    manifest_payload = base._pretty_json_bytes(candidate.manifest)
    report_payload = base._pretty_json_bytes(candidate.report)
    receipt = {
        "schema_version": 2,
        "tool_id": TOOL_ID,
        "artifact_class": ARTIFACT_CLASS,
        "verdict": VERDICT,
        "robot": {
            "family": "AgiBot A3",
            "exact_xml_model_name": exact_identity.xml_model_name,
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "inputs": {
            "motion": {"path": str(input_path), "sha256": input_sha},
            "stable_stand_contract": {
                "path": str(stable_path),
                "sha256": stable_sha,
            },
            "grounded_reference_candidate": {
                "path": str(reference_candidate_path),
                "sha256": reference_candidate_sha,
            },
            "grounded_reference_receipt": {
                "path": str(reference_receipt_path),
                "sha256": reference_receipt_sha,
            },
            "body_order": {
                "path": str(body_order_path),
                "sha256": body_order_sha,
            },
            "exact_model": base._jsonable(reference_receipt["exact_model"]),
        },
        "replacement": {
            "semantics": (
                "preserve head/arm motion and waist trajectory deltas; rebase "
                "waist frame-0 onto the AgiBot A3 runtime ready values; replace "
                "lower-body birth/reference with the runtime default stand; "
                "preserve source yaw while making the root upright"
            ),
            "source_root_z_m": float(motion["body_pos_w"][0, 0, 2]),
            "replacement_root_z_m": float(root_pos_out[0, 2]),
            "source_yaw_rad": source_yaw,
            "leg_joint_names": list(LEG_JOINT_NAMES),
            "leg_joint_indices": list(LEG_JOINT_INDICES),
            "waist_joint_names": list(WAIST_JOINT_NAMES),
            "waist_joint_indices": list(WAIST_JOINT_INDICES),
            "source_waist_ready_pos_rad": [
                float(value)
                for value in motion["joint_pos"][0, WAIST_JOINT_INDICES]
            ],
            "replacement_waist_ready_pos_rad": [
                float(value)
                for value in output["joint_pos"][0, WAIST_JOINT_INDICES]
            ],
            "stable_contract": base._jsonable(stable),
            "exact_a3_static_geometry_audit": grounding_audit,
        },
        "invariants": {
            "frame_count_before": motion["frames"],
            "frame_count_after": int(output["joint_pos"].shape[0]),
            "fps_before": motion["fps"],
            "fps_after": float(np.asarray(output["fps"]).reshape(-1)[0]),
            "strike_frame_before": strike,
            "strike_frame_after": strike,
            "head_arm_joint_pos_all_frames_bitwise_equal": True,
            "head_arm_joint_vel_all_frames_bitwise_equal": True,
            "waist_joint_delta_from_frame0_all_frames_bitwise_equal": True,
            "waist_joint_vel_all_frames_bitwise_equal": True,
            "root_xy_all_frames_bitwise_equal": True,
            "leg_joint_velocity_exact_zero": True,
            "right_racket_site_change_requires_task_rebind": site_change,
            "right_racket_site_speed_at_strike_before_mps": float(
                np.linalg.norm(before_trace.linear_velocity_w[strike])
            ),
            "right_racket_site_speed_at_strike_after_mps": float(
                np.linalg.norm(after_trace.linear_velocity_w[strike])
            ),
        },
        "outputs": {
            "motion_filename": motion_filename,
            "motion_sha256": candidate.output_sha256,
            "schema2_manifest_filename": base.SCHEMA2_MANIFEST_FILENAME,
            "schema2_manifest_sha256": base._sha256_bytes(manifest_payload),
            "schema2_report_filename": base.SCHEMA2_REPORT_FILENAME,
            "schema2_report_sha256": base._sha256_bytes(report_payload),
            "receipt_filename": base.RECEIPT_FILENAME,
            "completion_semantics": "exclusive_directory_receipt_written_last",
        },
        "producer": {
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": base._sha256_file(
                Path(__file__).resolve(), "stable-upper materializer"
            ),
            "schema2_tool_path": str(Path(schema2.__file__).resolve()),
            "schema2_tool_sha256": base._sha256_file(
                Path(schema2.__file__).resolve(), "schema-2 builder"
            ),
        },
        "non_claims": [
            "not an Isaac closed-loop hold certificate",
            "not a table-contact or physical-ball certificate",
            "not a task-alignment or policy-quality certificate",
            "not deployment or hardware authorization",
        ],
    }
    return base._publish_bundle(
        output_directory=Path(args.output_dir),
        motion_filename=motion_filename,
        motion_payload=candidate.npz_bytes,
        schema_manifest_payload=manifest_payload,
        schema_report_payload=report_payload,
        receipt=receipt,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-motion", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--stable-stand-contract", required=True)
    parser.add_argument("--expected-stable-stand-contract-sha256", required=True)
    parser.add_argument("--grounded-reference-candidate", required=True)
    parser.add_argument(
        "--expected-grounded-reference-candidate-sha256", required=True
    )
    parser.add_argument("--grounded-reference-receipt", required=True)
    parser.add_argument(
        "--expected-grounded-reference-receipt-sha256", required=True
    )
    parser.add_argument("--body-order", required=True)
    parser.add_argument("--expected-body-order-sha256", required=True)
    parser.add_argument("--strike-frame", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        published = materialize(_parser().parse_args(argv))
    except (
        StableUpperMaterializationError,
        base.GroundedUpperMaterializationError,
        schema2.Schema2BuildError,
    ) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(published.directory),
                "motion": str(published.motion),
                "motion_sha256": published.motion_sha256,
                "receipt": str(published.receipt),
                "receipt_sha256": published.receipt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
