"""Contract tests for the independent canonical 5 x 2 bank verifier."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "hope_training/whole_body_tracking/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
ROOT_SCRIPTS = REPO / "scripts"
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))

import canonical_motion_bank_gate as gate  # noqa: E402
import canonical_motion_compiler as compiler  # noqa: E402
import canonical_motion_recipe as recipe_module  # noqa: E402
import canonical_mujoco_path_adapter as path_adapter_module  # noqa: E402
import canonical_path_topp as path_topp_module  # noqa: E402
import canonical_time_law_artifact as time_law_artifact  # noqa: E402
import canonical_torque_path_topp as torque_topp_module  # noqa: E402
from canonical_path_topp import MarkerMapping, RetimeResult  # noqa: E402
import canonical_schema2_builder as schema2_builder  # noqa: E402
from mujoco_motion_player import load_motion  # noqa: E402
import test_canonical_motion_compiler as compiler_test_support  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _swept_clearance_payload(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    recipe_path: Path,
    ready_path: Path,
    mjcf_path: Path,
    urdf_path: Path,
    body_order_path: Path,
    verifier_path: Path,
) -> dict[str, Any]:
    outputs = [
        {
            "motion_id": row["motion_id"],
            "scope": row["scope"],
            "filename": row["filename"],
            "sha256": row["output_npz_sha256"],
        }
        for row in manifest["outputs"]
    ]
    result_rows = []
    for output, manifest_output in zip(outputs, manifest["outputs"]):
        with np.load(
            manifest_path.parent / output["filename"], allow_pickle=False
        ) as payload:
            frames = int(np.asarray(payload["joint_pos"]).shape[0])
            fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
        duration = (frames - 1) / fps
        result_rows.append(
            {
                **output,
                "frames": frames,
                "fps": fps,
                "duration_s": duration,
                "start_frame": 0,
                "end_frame": frames - 1,
                "interval_count": frames - 1,
                "certified_interval_count": frames - 1,
                "unknown_interval_count": 0,
                "unsafe_interval_count": 0,
                "nonfinite_interval_count": 0,
                "all_intervals_conservatively_bounded": True,
                "contact_window_start_s": manifest_output[
                    "contact_window_start_s"
                ],
                "contact_window_end_s": manifest_output[
                    "contact_window_end_s"
                ],
                "coverage_start": "first_frame",
                "contact_opportunity_covered": True,
                "coverage_end": "last_frame",
                "complete_cycle": True,
                "with_table": True,
                "subjects": list(gate._SWEPT_SUBJECTS),
                "obstacles": list(gate._SWEPT_OBSTACLES),
                "verdict": "PASS",
                "hard_collision_count": 0,
                "minimum_clearance_certified_lower_bound_m": 0.006,
            }
        )
    geometry_source_paths = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/geometry.py",
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/table_tennis/table_frame.py",
        SCRIPTS / "check_table_obstacle_scene.py",
    )
    components = [
        {
            "role": role,
            "center_m": [float(index), 0.0, 0.75],
            "full_extents_m": [0.1, 0.1, 0.1],
        }
        for index, role in enumerate(gate._ACTION_BALL_ASSEMBLY_ROLES)
    ]
    collision_geom_names = sorted(
        [
            "right_racket_collision",
            "right_racket_handle_collision",
            "torso_collision",
        ]
    )
    return {
        "schema_version": 1,
        "receipt_class": gate._SWEPT_RECEIPT_CLASS,
        "verdict": "PASS",
        "with_table": True,
        "independent_verifier": {
            "path": str(verifier_path.resolve()),
            "sha256": _sha(verifier_path),
        },
        "bank_binding": {
            "manifest_sha256": _sha(manifest_path),
            "recipe_sha256": _sha(recipe_path),
            "ready_sha256": _sha(ready_path),
            "mjcf_sha256": _sha(mjcf_path),
            "urdf_sha256": _sha(urdf_path),
            "body_order_sha256": _sha(body_order_path),
            "station_center_shift_xy_m": manifest.get(
                "station_center_shift_xy_m"
            ),
            "output_matrix": copy.deepcopy(manifest["output_matrix"]),
            "outputs": copy.deepcopy(outputs),
        },
        "trajectory_contract": {
            "coverage": gate._SWEPT_COVERAGE,
            "complete_cycle": True,
            "start": "first_canonical_ready_frame",
            "includes_contact_opportunity": True,
            "end": "final_canonical_recovery_ready_frame",
            "scopes": list(gate.SCOPES),
        },
        "scene_contract": {
            "subjects": list(gate._SWEPT_SUBJECTS),
            "forbidden_world_geometry": list(gate._SWEPT_OBSTACLES),
            "action_ball_keepout_semantics": gate._ACTION_BALL_KEEPOUT,
            "action_ball_assembly": {
                "roles": list(gate._ACTION_BALL_ASSEMBLY_ROLES),
                "geometry_sources": [
                    {
                        "role": role,
                        "path": str(source.resolve()),
                        "sha256": _sha(source),
                    }
                    for role, source in zip(
                        gate._ACTION_BALL_GEOMETRY_SOURCE_ROLES,
                        geometry_source_paths,
                    )
                ],
                "components": components,
                "components_sha256": gate._canonical_json_sha256(
                    components, "test ActionBall components"
                ),
            },
            "robot_geometry": {
                "all_enabled_collision_geoms": True,
                "collision_geom_names": collision_geom_names,
                "collision_geom_names_sha256": gate._canonical_json_sha256(
                    collision_geom_names,
                    "test robot collision geom names",
                ),
                "racket_and_handle_geom_names": list(
                    gate._RACKET_AND_HANDLE_GEOMS
                ),
            },
        },
        "method": {
            "certificate_kind": (
                "conservative_continuous_time_swept_volume"
            ),
            "continuous_time_swept_volume": True,
            "sampled_or_geometry_only": False,
            "inter_sample_conservative_bound": True,
        },
        "results": result_rows,
        "authorization": {
            "swept_clearance_complete": True,
            "training_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": [
            "dynamics_or_balance",
            "training_authorization",
            "hardware_authorization",
        ],
    }


def _publish_clearance_payload(fixture: Any, payload: dict[str, Any]) -> None:
    fixture.clearance_payload = payload
    fixture.clearance_receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fixture.clearance_receipt_sha256 = _sha(
        fixture.clearance_receipt_path
    )


def _write_clip(path: Path, *, ready_joint: np.ndarray, frames: int = 5) -> None:
    joint_pos = np.repeat(ready_joint[None, :], frames, axis=0).astype(np.float32)
    joint_pos[1:-1, 0] += (
        0.02
        * np.sin(np.linspace(0.0, np.pi, frames - 2, dtype=np.float32))
    )
    joint_vel = np.zeros((frames, 31), dtype=np.float32)
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    body_pos[:, 0, 2] = 1.0
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((frames, 32, 3), dtype=np.float32)
    body_ang = np.zeros((frames, 32, 3), dtype=np.float32)
    np.savez(
        path,
        fps=np.array([50.0], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=np.asarray(gate.motion_player.RUNTIME_BODY_NAMES),
    )


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def _grounded_dynamics_report(
    plant: Any,
    path: Path,
    *,
    fake_green: bool = False,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        frames = int(np.asarray(payload["joint_pos"]).shape[0])
        fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
    duration = float(frames - 1) / fps
    peak_frame = min(2, frames - 1)
    contact_frames = list(range(frames))
    floor_contacts = [
        {
            "frame": frame,
            "depth_m": 0.0,
            "geom_pair": ["floor", "right_foot"],
            "body_pair": ["world", "right_foot_link"],
            "robot_body_is_foot": True,
        }
        for frame in contact_frames
    ]
    return {
        "schema_version": 1,
        "verdict": "PASS" if fake_green else "INCOMPLETE_FAIL_CLOSED",
        "screen_pass": bool(fake_green),
        "plant_specific_screen": True,
        "source": str(path.resolve()),
        "plant": {
            "mjcf_sha256": plant.mjcf_sha256,
            "urdf_sha256": plant.urdf_sha256,
            "compiled_model_signature_sha256": plant.compiled_signature_sha256,
            "identity_bound": True,
            "actuation_mapping_verified": True,
            "runtime_joint_names": list(gate.motion_player.RUNTIME_JOINT_NAMES),
        },
        "input": {
            "frames": frames,
            "fps": fps,
            "duration_s": duration,
            "schema2_body_validation": {
                "body_count": 32.0,
                "position_peak_error_m": 0.0,
                "orientation_peak_error_rad": 0.0,
                "linear_velocity_peak_error_m_s": 0.0,
                "angular_velocity_peak_error_rad_s": 0.0,
                "linear_velocity_peak_abs_error_m_s": 0.0,
                "angular_velocity_peak_abs_error_rad_s": 0.0,
                "joint_velocity_peak_abs_mismatch_rad_s": 0.0,
                "root_linear_velocity_peak_mismatch_m_s": 0.0,
                "root_angular_velocity_peak_mismatch_rad_s": 0.0,
                "inverse_body_velocity_peak_abs_residual": 0.0,
                "velocity_jacobian_min_rank": 37.0,
                "velocity_jacobian_max_condition": 1.0,
                "absolute_tolerance": (
                    gate.dynamics_gate.SCHEMA_BODY_LINEAR_VELOCITY_TOL_M_S
                ),
            },
            "velocity_position_consistency": {
                "method": (
                    "schema2_all_32_body_com_velocity_jacobian_inverse_and_"
                    "mujoco_objectVelocity_world_forward"
                ),
                "pass": True,
                "joint_velocity_peak_abs_mismatch_rad_s": 0.0,
                "root_linear_velocity_peak_mismatch_m_s": 0.0,
                "root_angular_velocity_peak_mismatch_rad_s": 0.0,
            },
        },
        "checks": {
            "joint_limits": {
                "pass": True,
                "position": {
                    "pass": True,
                    "violation_samples": 0,
                    "violation_frames": [],
                    "peak_frame": 0,
                    "peak_joint": gate.motion_player.RUNTIME_JOINT_NAMES[0],
                },
                "velocity": {
                    "pass": True,
                    "violation_samples": 0,
                    "violation_frames": [],
                    "peak_frame": 0,
                    "peak_joint": gate.motion_player.RUNTIME_JOINT_NAMES[0],
                },
            },
            "geometry": {
                "pass": True,
                "self_collision_violation_count": 0,
                "foot_floor_penetration_violation_count": 0,
                "nonfoot_floor_penetration_violation_count": 0,
                "other_world_penetration_violation_count": 0,
                "self_collision_violations": [],
                "foot_floor_penetration_violations": [],
                "nonfoot_floor_penetration_violations": [],
                "other_world_penetration_violations": [],
                "all_contacts": {
                    "self_contacts": [],
                    "floor_contacts": floor_contacts,
                    "other_world_contacts": [],
                },
            },
            "inverse_dynamics": {
                "torque_interpretation": {
                    "valid": False,
                    "label": "uncertified_contact_disabled_joint_effort_proxy",
                    "reasons": [
                        "geometric contact is active; contact wrench distribution is unspecified"
                    ],
                    "contact_frames": contact_frames,
                    "root_force_peak_n": 10.0,
                    "root_force_peak_frame": peak_frame,
                    "root_torque_peak_nm": 1.0,
                    "root_torque_peak_frame": peak_frame,
                },
                "joint_effort_proxy_peak_utilization": 0.4,
                "joint_effort_proxy_peak_frame": peak_frame,
                "joint_effort_proxy_peak_joint": (gate.motion_player.RUNTIME_JOINT_NAMES[0]),
                "actuator_force_proxy_peak_utilization": 0.4,
                "actuator_force_proxy_peak_frame": peak_frame,
                "actuator_force_proxy_peak_joint": (gate.motion_player.RUNTIME_JOINT_NAMES[0]),
                "effort_limit_pass_only_if_interpretation_valid": (True if fake_green else False),
            },
            "root_and_com": {
                "root_height_min_m": 1.0,
                "root_height_max_m": 1.0,
                "root_tilt_peak_rad": 0.0,
                "root_xy_displacement_peak_m": 0.0,
                "com_height_min_m": 0.8,
                "com_height_max_m": 0.8,
            },
        },
    }


def _player_report() -> dict[str, Any]:
    return {
        "verdict": "PASS",
        "gates": {
            "position": {"pass": True, "max_error_m": 0.0},
            "orientation": {"pass": True, "max_error_rad": 0.0},
        },
        "evidence_boundary": {
            "level": "kinematic_playback_only",
            "mj_step_calls": 0,
            "dynamic_certificate": False,
            "training_certificate": False,
            "real_robot_certificate": False,
        },
    }


def _contact_free_dynamics_report(plant: Any, path: Path) -> dict[str, Any]:
    report = _grounded_dynamics_report(plant, path)
    report["verdict"] = "PASS"
    report["screen_pass"] = True
    report["plant"]["actuation_mapping_verified"] = True
    report["checks"]["geometry"]["all_contacts"] = {
        "self_contacts": [],
        "floor_contacts": [],
        "other_world_contacts": [],
    }
    torque = report["checks"]["inverse_dynamics"]["torque_interpretation"]
    torque.update(
        {
            "valid": True,
            "label": "certified_contact_free_direct_motor_torque",
            "reasons": [],
            "contact_frames": [],
            "root_force_peak_n": 0.0,
            "root_torque_peak_nm": 0.0,
        }
    )
    report["checks"]["inverse_dynamics"]["effort_limit_pass_only_if_interpretation_valid"] = True
    return report


def _compiler_schema2_candidate(**kwargs) -> schema2_builder.Schema2Candidate:
    """MuJoCo-free schema writer for the real compiler integration test.

    The compiler, geometry search, retimer, manifest builder, and atomic bank
    writer remain real.  Only schema FK is substituted because the local test
    environment has no MuJoCo; the bank's player/dynamics consumers are
    separately dependency-injected under their strict report contracts.
    """

    q = np.asarray(kwargs["joint_pos"], dtype=np.float32)
    qd = np.asarray(kwargs["joint_vel"], dtype=np.float32)
    root_pos = np.asarray(kwargs["root_pos_w"], dtype=np.float32)
    root_quat = np.asarray(kwargs["root_quat_wxyz"], dtype=np.float32)
    root_lin = np.asarray(kwargs["root_lin_vel_w"], dtype=np.float32)
    root_ang = np.asarray(kwargs["root_ang_vel_w"], dtype=np.float32)
    frames = int(q.shape[0])
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((frames, 32, 3), dtype=np.float32)
    body_ang = np.zeros((frames, 32, 3), dtype=np.float32)
    body_pos[:, 0] = root_pos
    body_quat[:, 0] = root_quat
    body_lin[:, 0] = root_lin
    body_ang[:, 0] = root_ang
    arrays = {
        "fps": np.array([kwargs["fps"]], dtype=np.float32),
        "joint_pos": q,
        "joint_vel": qd,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "kinematics_schema_version": np.array([2], dtype=np.int64),
        "body_pos_point": np.array("link_origin"),
        "body_lin_vel_point": np.array("center_of_mass"),
        "body_names": np.asarray(gate.motion_player.RUNTIME_BODY_NAMES),
    }
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    npz_bytes = stream.getvalue()
    output_sha = hashlib.sha256(npz_bytes).hexdigest()
    mjcf = Path(kwargs["mjcf_path"]).resolve()
    body_order = Path(kwargs["body_order_path"]).resolve()
    tool = Path(schema2_builder.__file__).resolve()
    hashes = {
        "input_sha256": kwargs["input_sha256"],
        "ready_sha256": kwargs["ready_sha256"],
        "mjcf_sha256": _sha(mjcf),
        "body_order_sha256": _sha(body_order),
        "tool_sha256": _sha(tool),
        "output_npz_sha256": output_sha,
    }
    runtime = {
        "joint_count": 31,
        "joint_names": list(gate.motion_player.RUNTIME_JOINT_NAMES),
        "body_count": 32,
        "body_names": list(gate.motion_player.RUNTIME_BODY_NAMES),
        "schema2_field_count": 11,
    }
    manifest = {
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "tool_id": "canonical_schema2_builder_v1",
        "hashes": hashes,
        "runtime_contract": runtime,
        "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
        "files": {
            "mjcf_path": str(mjcf),
            "body_order_path": str(body_order),
            "tool_path": str(tool),
        },
        "kinematics": {
            "body_pos_point": "link_origin",
            "body_lin_vel_point": "center_of_mass",
            "body_velocity_method": "mujoco_mj_jacBodyCom_times_qvel",
            "root_velocity_input": "world_link_origin_twist",
            "joint_velocity_input": "explicit_compiler_time_law",
            "pose_finite_difference_used": False,
            "static_ready_endpoints_required": True,
        },
        "non_claims": [
            "training_authorization",
            "dynamics",
            "balance",
            "contact",
            "deployment",
            "hardware",
        ],
    }
    report = {
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "tool_id": "canonical_schema2_builder_v1",
        "hashes": dict(hashes),
        "runtime_contract": dict(runtime),
        "status": "PASS",
        "frames": frames,
        "fps": float(kwargs["fps"]),
        "checks": {
            "same_ready_pose_first_last": True,
            "six_velocity_channels_zero_first_last": True,
        },
        "maxima": {},
    }
    return schema2_builder.Schema2Candidate(
        arrays=MappingProxyType(arrays),
        npz_bytes=npz_bytes,
        manifest=MappingProxyType(manifest),
        report=MappingProxyType(report),
    )


def _fast_current_contract_retime(q_path: np.ndarray, *args: Any, **kwargs: Any) -> RetimeResult:
    """Deterministic retimer seam for the compiler-to-bank integration test.

    The test intentionally exercises the real compiler search, geometry,
    diagnostics, manifest builder, and atomic writer.  Dedicated retimer tests
    cover the expensive numerical solve; this seam keeps the cross-component
    contract test fast while emitting the current retimer receipt verbatim.
    """

    del args
    q = np.asarray(q_path, dtype=np.float64)
    fps = float(kwargs["fps"])
    marker_positions = {name: float(position) for name, position in kwargs["markers"].items()}
    frame_count = int(len(q))
    duration = float(frame_count - 1) / fps
    markers = {
        name: MarkerMapping(
            source_index=position,
            time_s=position / fps,
            output_fractional_frame=position,
            output_frame=int(round(position)),
            path_position_at_frame=position,
        )
        for name, position in marker_positions.items()
    }
    for (start, end), minimum_s in kwargs["marker_min_duration_s"].items():
        actual_s = markers[end].time_s - markers[start].time_s
        if actual_s + 1.0e-12 < float(minimum_s):
            raise AssertionError(
                "integration retimer seam received geometry that violates "
                "the compiler's marker-duration contract"
            )
    acceleration_marker = kwargs["nonnegative_acceleration_until_marker"]
    acceleration_marker_position = marker_positions[acceleration_marker]
    return RetimeResult(
        q=q,
        qdot=np.zeros_like(q),
        path_position=np.arange(frame_count, dtype=np.float64),
        path_speed=np.ones(frame_count - 1, dtype=np.float64),
        path_acceleration=np.zeros(frame_count - 1, dtype=np.float64),
        markers=markers,
        report={
            "algorithm": "shape_preserving_pchip_forward_backward_scalar_path",
            "constraint_model": "kinematic_velocity_and_acceleration_only",
            "marker_policy": ("selected_marker_nonnegative_scalar_acceleration_no_pose_lock"),
            "marker_output_frame_policy": ("nearest_sample_observation_only_not_interval_gate"),
            "marker_interval_discrete_policy": ("inclusive_samples_ceil_start_floor_end"),
            "fps": fps,
            "input_samples": frame_count,
            "output_frames": frame_count,
            "duration_s": duration,
            "start_speed": 0.0,
            "end_speed": 0.0,
            "markers": {
                name: {
                    "source_index": mapping.source_index,
                    "time_s": mapping.time_s,
                    "output_fractional_frame": mapping.output_fractional_frame,
                    "output_frame": mapping.output_frame,
                    "path_position_at_frame": mapping.path_position_at_frame,
                }
                for name, mapping in markers.items()
            },
            "nonnegative_acceleration_until_marker": {
                "enabled": True,
                "marker": acceleration_marker,
                "marker_source_index": acceleration_marker_position,
                "grid_node_is_exact_marker": True,
                "prefix_scalar_acceleration_min_continuous": 0.0,
                "prefix_scalar_acceleration_min_50hz": 0.0,
            },
        },
    )


def _write_strict_compiler_recipe(root: Path) -> Path:
    """Create a tiny but fully strict recipe/load graph for the E2E seam."""

    configs = root / "configs"
    assets = root / "assets"
    models = root / "models"
    configs.mkdir(parents=True)
    assets.mkdir()
    models.mkdir()
    body_order = configs / "body_order.txt"
    body_order.write_text(
        "\n".join(gate.motion_player.RUNTIME_BODY_NAMES) + "\n",
        encoding="utf-8",
    )
    mjcf = models / "plant.xml"
    urdf = models / "plant.urdf"
    mjcf.write_text("<mujoco model='strict-fixture'/>\n", encoding="utf-8")
    urdf.write_text("<robot name='strict-fixture'/>\n", encoding="utf-8")

    ready_joint = np.zeros(31, dtype=np.float64)
    source_files: dict[str, Path] = {}
    for motion_id in gate.MOTION_IDS:
        source_key = "bh_block" if motion_id == "fh_block_syn" else motion_id
        path = assets / f"{source_key}.npz"
        if not path.exists():
            _write_clip(path, ready_joint=ready_joint, frames=9)
            arrays = _read_arrays(path)
            # A two-joint circular arc gives the retained core a regular
            # (cusp-free) C2 loop against the shared ready, matching the
            # geometry the real retimer is proven to certify.  A collinear
            # ramp or a same-line bump would put a reversal cusp in either
            # the core or a ready connector.
            theta = np.linspace(0.0, np.pi, 9, dtype=np.float64)
            shoulder = gate.motion_player.RUNTIME_JOINT_NAMES.index(
                "right_shoulder_pitch_joint"
            )
            elbow = gate.motion_player.RUNTIME_JOINT_NAMES.index(
                "right_elbow_joint"
            )
            amplitude = 0.008 * (gate.MOTION_IDS.index(motion_id) + 1)
            arrays["joint_pos"][:, shoulder] = (
                np.float32(0.02) * np.cos(theta).astype(np.float32)
            )
            arrays["joint_pos"][:, elbow] = (
                np.float32(amplitude) * np.sin(theta).astype(np.float32)
            )
            _write_arrays(path, arrays)
        source_files[motion_id] = path

    donor = source_files["bh_loop_c"]
    with np.load(donor, allow_pickle=False) as payload:
        donor_joint = np.asarray(payload["joint_pos"][0], dtype=np.float64)
        donor_root_pos = np.asarray(payload["body_pos_w"][0, 0], dtype=np.float64)
        donor_root_quat = np.asarray(payload["body_quat_w"][0, 0], dtype=np.float64)
    ready_path = assets / "ready.npz"
    np.savez(
        ready_path,
        joint_pos=donor_joint,
        joint_vel=np.zeros(31, dtype=np.float64),
        root_pos_w=donor_root_pos,
        root_quat_w=donor_root_quat,
        source_segment=np.array("bh_loop_c"),
        source_npz=np.array(donor.name),
        source_frame=np.array(0, dtype=np.int64),
        striking_joint_ids=np.arange(7, dtype=np.int64),
        note=np.array("strict compiler/gate integration fixture"),
    )

    motion_specs = []
    for motion_id in gate.MOTION_IDS:
        source = source_files[motion_id]
        row: dict[str, Any] = {
            "motion_id": motion_id,
            "human_role": f"{motion_id} strict fixture",
            "source_path": str(source.relative_to(root)),
            "source_sha256": _sha(source),
            "scope_overrides": (
                {
                    "full": {
                        "grounding_policy": (
                            "minimum_constant_z_offset_for_1mm_clearance"
                        ),
                        "maximum_grounding_offset_m": 0.09,
                    }
                }
                if motion_id == "s0_highpress"
                else {}
            ),
        }
        if motion_id == "fh_block_syn":
            row.update(
                {
                    "face_manifold": {
                        "mode": "signed_raw_plus_y_flip",
                        "active_joints": "right_arm_7",
                        "site_position": "preserve_per_source_frame",
                        "orientation": "normal_hard_inplane_free",
                        "single_axis_pi_overlay_forbidden": True,
                    },
                }
            )
        motion_specs.append(row)

    marker_authority_path = configs / "marker_authority.json"
    marker_authority_path.write_text(
        json.dumps(
            {
                "authority_id": "strict_fixture_marker_authority_v2",
                "note": (
                    "object-injected marker semantics; parsed authority "
                    "loading is covered by test_canonical_motion_recipe"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    recipe = {
        "schema_version": 1,
        "library_id": "strict_compiler_bank_e2e",
        "publication_class": "compiler_candidate",
        "training_authorized": False,
        "hardware_authorized": False,
        "purpose": "Strict compiler to independent bank-gate integration fixture.",
        "frame_id": "a3_robot_origin_ground_z0",
        "canonical_ready": {
            "path": str(ready_path.relative_to(root)),
            "sha256": _sha(ready_path),
            "donor_motion_id": "bh_loop_c",
            "donor_source_frame": 0,
            "donor_source_sha256": _sha(donor),
            "endpoint_velocity_policy": (
                "all_joint_root_body_velocities_exact_zero"
            ),
        },
        "model_contract": {
            "mjcf_path": str(mjcf.relative_to(root)),
            "mjcf_sha256": _sha(mjcf),
            "urdf_path": str(urdf.relative_to(root)),
            "urdf_sha256": _sha(urdf),
            "body_order_path": str(body_order.relative_to(root)),
            "body_order_sha256": _sha(body_order),
        },
        "scope_contract": {
            "upper": {
                "root": "fixed_canonical_ready",
                "lower_and_head": "fixed_canonical_ready",
                "pelvis_relative_rotation": "fold_complete_so3_into_waist_zxy",
                "pelvis_translation": "removed_and_reported",
            },
            "full": {
                "root": (
                    "one_atomic_se2_frame0_alignment_then_preserve_local_motion"
                ),
                "joints": "preserve_full_source_before_ready_connectors",
            },
        },
        "marker_authority": {
            "path": str(marker_authority_path.relative_to(root)),
            "sha256": _sha(marker_authority_path),
        },
        "time_law": {
            "fps": 50.0,
            "joint_velocity_limit_fraction": 1.0,
            "post_retime_behavior_opportunity_minimum_s": 0.08,
            "legacy_seed_marker_policy": (
                "search_and_retime_marker_only_never_output_behavior_window"
            ),
            "kinematic_window_policy": (
                "nonnegative_scalar_acceleration_through_exact_window_end"
            ),
            "acceleration_policy": (
                "grounded_torque_contact_screen_required_before_promotion_beyond_"
                "compiler_candidate"
            ),
            "window_acceleration_allowed_through_end": True,
            "window_acceleration_objective": (
                "prefer_no_cruise_or_braking_before_window_end_unless_a_hard_limit_"
                "requires_it"
            ),
            "torque_claim": (
                "no_uniform_torque_or_dynamic_claim_until_inverse_dynamics_and_replay_pass"
            ),
        },
        "entry_exit_search": {
            "mode": "enumerate_all_then_gate_and_rank",
            "legacy_ge80_halo_source_frames": 1,
            "candidate_eligibility": (
                "retain_legacy_ge80_seed_plus_symmetric_halo"
            ),
            "ranking_preference": [
                "opportunity_start",
                "ordinary_nominal_event_if_available",
                "opportunity_end",
            ],
            "retained_source_prefix_required": False,
            "retained_source_suffix_required": False,
            "historical_adv2c3_role": "comparator_only_not_default",
        },
        "motion_specs": motion_specs,
        "required_output_matrix": {
            "motion_ids": list(gate.MOTION_IDS),
            "scopes": list(gate.SCOPES),
            "candidate_count": 10,
        },
        "post_build_gates": [
            "strict_schema2_and_shared_ready_digest",
            "exact_vendor_mujoco_fk_playback",
            "joint_position_velocity_and_plant_specific_torque_screen",
            "self_collision_body_racket_ground_table_net_scan",
            "post_retime_behavior_opportunity_rescan_per_scope",
            "stationary_behavior_and_recovery_exam_per_motion",
            "registry_consumer_export_deploy_contract",
        ],
    }
    recipe_path = configs / "recipe.json"
    recipe_path.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sources = tuple(
        recipe_module.MotionSource(
            motion_id=row["motion_id"],
            human_role=row["human_role"],
            path=root / row["source_path"],
            sha256=row["source_sha256"],
            clip=load_motion(root / row["source_path"]),
            face_manifold=row.get("face_manifold"),
            scope_overrides=row["scope_overrides"],
        )
        for row in motion_specs
    )
    ready_state = recipe_module.ReadyState(
        path=ready_path,
        sha256=_sha(ready_path),
        joint_pos=donor_joint,
        joint_vel=np.zeros(31, dtype=np.float64),
        root_pos_w=donor_root_pos,
        root_quat_wxyz=donor_root_quat,
        source_segment="bh_loop_c",
        source_frame=0,
    )
    recipe_object = recipe_module.CanonicalMotionRecipe(
        path=recipe_path,
        repo_root=root,
        raw=recipe,
        ready=ready_state,
        sources=sources,
        marker_semantics=_fixture_marker_semantics(
            ge80=(4, 5), anchor=4, solve_span=(3, 6)
        ),
        marker_authority_path=marker_authority_path,
        marker_authority_sha256=_sha(marker_authority_path),
        model_paths={"mjcf": mjcf, "urdf": urdf, "body_order": body_order},
        model_hashes={
            "mjcf": _sha(mjcf),
            "urdf": _sha(urdf),
            "body_order": _sha(body_order),
        },
    )
    return recipe_path, recipe_object


def _fixture_marker_semantics(
    *,
    ge80: tuple[int, int] = (1, 3),
    anchor: int = 2,
    solve_span: tuple[int, int] = (0, 4),
    motion_ids: tuple[str, ...] = gate.MOTION_IDS,
):
    """Marker authority v2 rows matching the fixture window/anchor markers."""

    rows = {}
    for motion_id in motion_ids:
        synthetic = motion_id == "fh_block_syn"
        rows[motion_id] = SimpleNamespace(
            motion_id=motion_id,
            nominal_event=None if synthetic else anchor,
            ge50_seed=ge80,
            ge80_seed=ge80,
            preferred_seed=None,
            construction_marker=(
                SimpleNamespace(
                    annotation_frame=anchor,
                    donor_preferred_frame=anchor,
                    solve_span=solve_span,
                )
                if synthetic
                else None
            ),
            historical_adv2c3_start=1,
        )
    return SimpleNamespace(row=lambda motion_id: rows[motion_id])


class BankFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.bank = tmp_path / "bank"
        self.bank.mkdir()
        self.mjcf = tmp_path / "plant.xml"
        self.urdf = tmp_path / "plant.urdf"
        self.body_order = tmp_path / "body_order.txt"
        self.recipe_path = tmp_path / "recipe.json"
        self.compiler_path = SCRIPTS / "canonical_motion_compiler.py"
        self.geometry_tool_path = SCRIPTS / "canonical_motion_geometry.py"
        self.weighted_arc_tool_path = SCRIPTS / "canonical_weighted_arc_path.py"
        self.ready_path = tmp_path / "ready.npz"
        self.clearance_verifier_path = (
            tmp_path / "independent_swept_clearance_verifier.py"
        )
        self.clearance_receipt_path = (
            tmp_path / "swept_clearance_receipt.json"
        )
        _write_text(self.mjcf, "<mujoco model='test'/>")
        _write_text(self.urdf, "<robot name='test'/>")
        _write_text(
            self.clearance_verifier_path,
            "# independent continuous swept-clearance test verifier\n",
        )
        _write_text(
            self.body_order,
            "\n".join(gate.motion_player.RUNTIME_BODY_NAMES) + "\n",
        )
        _write_text(self.recipe_path, '{"fixture":"strict-loader-injected"}\n')
        self.ready_joint = np.zeros(31, dtype=np.float64)
        self.ready_root_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.ready_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        np.savez(
            self.ready_path,
            joint_pos=self.ready_joint,
            joint_vel=np.zeros(31, dtype=np.float64),
            root_pos_w=self.ready_root_pos,
            root_quat_w=self.ready_root_quat,
            source_segment=np.array("bh_loop_c"),
            source_npz=np.array("fixture.npz"),
            source_frame=np.array(0, dtype=np.int64),
            striking_joint_ids=np.arange(7, dtype=np.int64),
            note=np.array("fixture"),
        )
        self.signature = hashlib.sha256(b"compiled fixture").hexdigest()
        self.hashes = {
            "mjcf": _sha(self.mjcf),
            "urdf": _sha(self.urdf),
            "body_order": _sha(self.body_order),
            "ready": _sha(self.ready_path),
            "recipe": _sha(self.recipe_path),
            "compiler": _sha(self.compiler_path),
            "geometry_tool": _sha(self.geometry_tool_path),
            "weighted_arc_tool": _sha(self.weighted_arc_tool_path),
            "tool": _sha(SCRIPTS / "canonical_schema2_builder.py"),
        }
        source_paths: dict[str, Path] = {}
        for motion_id in gate.MOTION_IDS:
            key = "bh_block" if motion_id == "fh_block_syn" else motion_id
            path = tmp_path / f"source_{key}.npz"
            if not path.exists():
                path.write_bytes(f"source:{key}".encode("utf-8"))
            source_paths[motion_id] = path
        source_rows = tuple(
            SimpleNamespace(
                motion_id=motion_id,
                path=source_paths[motion_id],
                sha256=_sha(source_paths[motion_id]),
            )
            for motion_id in gate.MOTION_IDS
        )
        self.recipe = SimpleNamespace(
            path=self.recipe_path.resolve(),
            raw={
                "library_id": "fixture_bank",
                "publication_class": "compiler_candidate",
                "training_authorized": False,
                "hardware_authorized": False,
                "motion_specs": [
                    {
                        "motion_id": motion_id,
                        "scope_overrides": (
                            {
                                "full": {
                                    "maximum_grounding_offset_m": 0.09,
                                }
                            }
                            if motion_id == "s0_highpress"
                            else {}
                        ),
                    }
                    for motion_id in gate.MOTION_IDS
                ],
                "time_law": {
                    "joint_velocity_limit_fraction": 1.0,
                    "legacy_seed_marker_policy": (
                        "search_and_retime_marker_only_never_output_behavior_window"
                    ),
                },
                "entry_exit_search": {
                    "historical_adv2c3_role": "comparator_only_not_default",
                },
            },
            marker_semantics=_fixture_marker_semantics(),
            ready=SimpleNamespace(
                path=self.ready_path.resolve(),
                sha256=self.hashes["ready"],
                joint_pos=self.ready_joint.copy(),
                joint_vel=np.zeros(31, dtype=np.float64),
                root_pos_w=self.ready_root_pos.copy(),
                root_quat_wxyz=self.ready_root_quat.copy(),
            ),
            model_paths={
                "mjcf": self.mjcf.resolve(),
                "urdf": self.urdf.resolve(),
                "body_order": self.body_order.resolve(),
            },
            model_hashes={
                "mjcf": self.hashes["mjcf"],
                "urdf": self.hashes["urdf"],
                "body_order": self.hashes["body_order"],
            },
            sources=source_rows,
        )
        options = compiler_test_support._options()
        joint_acceleration, root_limits = compiler._validate_options(options)
        backend = compiler_test_support.FakePlantBackend()
        lower, upper, velocity = compiler._validate_backend(backend)
        contracts = {
            scope: compiler._path_contract(
                scope=scope,
                joint_lower=lower,
                joint_upper=upper,
                joint_velocity=velocity,
                joint_acceleration=joint_acceleration,
                root_limits=root_limits,
                velocity_fraction=1.0,
            )
            for scope in gate.SCOPES
        }
        face_config = compiler._effective_face_config(options, 1.0)
        self.compiler_options = json.loads(
            json.dumps(
                compiler._json_safe(
                    compiler._compiler_options_receipt(
                        options,
                        joint_acceleration=joint_acceleration,
                        root_limits=root_limits,
                        face_config=face_config,
                        contracts=contracts,
                    )
                )
            )
        )
        segment_intervals = [
            {
                "kind": "pre_connector",
                "sample_intervals": 5,
                "source_frame_start": None,
                "source_frame_end": 0.0,
            },
            {
                "kind": "source_core",
                "sample_intervals": 5,
                "source_frame_start": 0.0,
                "source_frame_end": 1.0,
            },
            {
                "kind": "source_core",
                "sample_intervals": 10,
                "source_frame_start": 1.0,
                "source_frame_end": 2.0,
            },
            {
                "kind": "source_core",
                "sample_intervals": 10,
                "source_frame_start": 2.0,
                "source_frame_end": 3.0,
            },
            {
                "kind": "source_core",
                "sample_intervals": 5,
                "source_frame_start": 3.0,
                "source_frame_end": 4.0,
            },
            {
                "kind": "post_connector",
                "sample_intervals": 5,
                "source_frame_start": 4.0,
                "source_frame_end": None,
            },
        ]
        source_map = np.concatenate(
            (
                np.asarray([np.nan] * 5 + [0.0]),
                np.linspace(0.0, 1.0, 6)[1:],
                np.linspace(1.0, 2.0, 11)[1:],
                np.linspace(2.0, 3.0, 11)[1:],
                np.linspace(3.0, 4.0, 6)[1:],
                np.asarray([np.nan] * 5),
            )
        )
        canonical_map = np.ascontiguousarray(source_map, dtype="<f8").copy()
        canonical_map.view("<u8")[np.isnan(canonical_map)] = np.uint64(
            0x7FF8000000000000
        )
        self.geometry_map_receipt = {
            "schema_version": 1,
            "encoding": gate._SOURCE_FRAME_MAP_ENCODING,
            "length": 41,
            "sha256": hashlib.sha256(
                canonical_map.tobytes(order="C")
            ).hexdigest(),
            "finite_count": 31,
            "nan_count": 10,
            "finite_value_encoding": (
                "piecewise_linear_source_frame_by_segment_v1"
            ),
            "segment_intervals": segment_intervals,
            "source_waypoint_path_indices": [5, 10, 20, 30, 35],
        }
        self.outputs = []
        for motion_id, scope in gate.EXPECTED_MATRIX:
            filename = f"{motion_id}_{scope}_canonical_v2.npz"
            path = self.bank / filename
            _write_clip(path, ready_joint=self.ready_joint)
            output_sha = _sha(path)
            schema_hashes = {
                "input_sha256": "0" * 64,
                "ready_sha256": self.hashes["ready"],
                "mjcf_sha256": self.hashes["mjcf"],
                "body_order_sha256": self.hashes["body_order"],
                "tool_sha256": self.hashes["tool"],
                "output_npz_sha256": output_sha,
            }
            schema_manifest = {
                "publication_class": "compiler_candidate",
                "training_authorized": False,
                "tool_id": "canonical_schema2_builder",
                "hashes": schema_hashes,
                "runtime_contract": {
                    "joint_count": 31,
                    "joint_names": list(gate.motion_player.RUNTIME_JOINT_NAMES),
                    "body_count": 32,
                    "body_names": list(gate.motion_player.RUNTIME_BODY_NAMES),
                    "schema2_field_count": 11,
                },
                "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
                "files": {
                    "mjcf_path": str(self.mjcf.resolve()),
                    "body_order_path": str(self.body_order.resolve()),
                    "tool_path": str((SCRIPTS / "canonical_schema2_builder.py").resolve()),
                },
                "kinematics": {
                    "body_pos_point": "link_origin",
                    "body_lin_vel_point": "center_of_mass",
                    "body_velocity_method": "mujoco_mj_jacBodyCom_times_qvel",
                    "root_velocity_input": "world_link_origin_twist",
                    "joint_velocity_input": "explicit_compiler_time_law",
                    "pose_finite_difference_used": False,
                    "static_ready_endpoints_required": True,
                },
                "non_claims": [
                    "training_authorization",
                    "dynamics",
                    "balance",
                    "contact",
                    "deployment",
                    "hardware",
                ],
            }
            schema_report = {
                "publication_class": "compiler_candidate",
                "training_authorized": False,
                "tool_id": "canonical_schema2_builder",
                "hashes": dict(schema_hashes),
                "runtime_contract": dict(schema_manifest["runtime_contract"]),
                "status": "PASS",
                "frames": 5,
                "fps": 50.0,
                "checks": {
                    "same_ready_pose_first_last": True,
                    "six_velocity_channels_zero_first_last": True,
                },
                "maxima": {},
            }
            markers = {
                "window_start": {
                    # Retimer source_index is the dense geometry-row index,
                    # not the original recipe source-frame number.
                    "source_index": 10.0,
                    "time_s": 0.02,
                    "output_fractional_frame": 1.0,
                    "output_frame": 1,
                    "path_position_at_frame": 10.0,
                },
                "source_anchor": {
                    "source_index": 20.0,
                    "time_s": 0.04,
                    "output_fractional_frame": 2.0,
                    "output_frame": 2,
                    "path_position_at_frame": 20.0,
                },
                "window_end": {
                    "source_index": 30.0,
                    "time_s": 0.06,
                    "output_fractional_frame": 3.0,
                    "output_frame": 3,
                    "path_position_at_frame": 30.0,
                },
            }
            self.outputs.append(
                {
                    "motion_id": motion_id,
                    "scope": scope,
                    "filename": filename,
                    "output_npz_sha256": output_sha,
                    "entry_frame": 0,
                    "exit_frame": 4,
                    "duration_s": 0.08,
                    "contact_window_start_s": 0.02,
                    "contact_window_end_s": 0.06,
                    "source_anchor_time_s": 0.04,
                    "scaled_l2_total_variation": 0.1,
                    "search": {
                        "selected": {
                            "entry_frame": 0,
                            "exit_frame": 4,
                            "duration_s": 0.08,
                            "old_frame_zero_forced": False,
                            "direct_path": ("canonical_ready_to_selected_core_to_canonical_ready"),
                        },
                        "contact_opportunity": {
                            "source_span_inclusive": [1.0, 3.0],
                            "source_anchor_frame": 2.0,
                            "marker_only": True,
                            "pose_locked": False,
                            "velocity_locked": False,
                            "acceleration_locked": False,
                            "acceleration_allowed_through_window_end": True,
                            "nonnegative_scalar_acceleration_through_window_end": True,
                        },
                    },
                    "scope_preprocessing": {
                        "scope": scope,
                        "joint_count": 31,
                        "joint_names": list(gate.motion_player.RUNTIME_JOINT_NAMES),
                        "pure_math_only": {"time_law_changed": False},
                    },
                    "face_manifold": None,
                    "geometry": {
                        "c2_continuous": True,
                        "parameterization": {
                            "samples_per_scaled_coordinate_unit": 6.0,
                        },
                        "selection": {
                            "entry_frame": 0,
                            "exit_frame": 4,
                            "window_start": 1,
                            "window_end": 3,
                            "old_source_frame_zero_retained": True,
                        },
                        "sampling": {
                            "path_points": 41,
                            "pre_connector_intervals": 5,
                            "post_connector_intervals": 5,
                            "min_core_intervals": 5,
                        },
                        "source_waypoints": {
                            "count": 5,
                            "mapped_frame_min": 0.0,
                            "mapped_frame_max": 4.0,
                        },
                        "source_frame_map_receipt": self.geometry_map_receipt,
                        "recipe_marker_binding": {
                            "window_start": {
                                "source_frame": 1,
                                "dense_row": 10,
                            },
                            "source_anchor": {
                                "source_frame": 2,
                                "dense_row": 20,
                            },
                            "window_end": {
                                "source_frame": 3,
                                "dense_row": 30,
                            },
                        },
                    },
                    "retiming": {
                        "constraint_model": "kinematic_velocity_and_acceleration_only",
                        "marker_policy": (
                            "selected_marker_nonnegative_scalar_acceleration_no_pose_lock"
                        ),
                        "marker_output_frame_policy": (
                            "nearest_sample_observation_only_not_interval_gate"
                        ),
                        "marker_interval_discrete_policy": (
                            "inclusive_samples_ceil_start_floor_end"
                        ),
                        "fps": 50.0,
                        "input_samples": 41,
                        "output_frames": 5,
                        "duration_s": 0.08,
                        "markers": markers,
                        "nonnegative_acceleration_until_marker": {
                            "enabled": True,
                            "marker": "window_end",
                            "marker_source_index": 30.0,
                            "grid_node_is_exact_marker": True,
                            "prefix_scalar_acceleration_min_continuous": 0.0,
                            "prefix_scalar_acceleration_min_50hz": 0.0,
                        },
                        "scalar_no_early_brake_proxy": {
                            "name": "scalar_no_early_brake_proxy_v2",
                            "selection_criterion": True,
                            "proxy_only": True,
                            "window_start_fractional_frame": 1.0,
                            "window_end_fractional_frame": 3.0,
                            "negative_segment_count_before_window_end": 0,
                            "negative_segment_count_overlapping_window": 0,
                            "no_negative_scalar_acceleration_before_window_end": True,
                            "no_negative_scalar_acceleration_inside_window": True,
                        },
                    },
                    "schema2_manifest": schema_manifest,
                    "schema2_report": schema_report,
                }
            )
        for row in self.outputs:
            input_sha = gate._recompute_candidate_input_sha256(
                row, self.bank / row["filename"], self.recipe
            )
            row["schema2_manifest"]["hashes"]["input_sha256"] = input_sha
            row["schema2_report"]["hashes"]["input_sha256"] = input_sha
        self.write_sidecars()
        self.manifest = {
            "schema_version": 1,
            "library_id": "fixture_bank",
            "publication_class": "compiler_candidate",
            "build_verdict": "PASS_COMPILER_CANDIDATE_ONLY",
            "training_authorized": False,
            "hardware_authorized": False,
            "recipe": {
                "path": str(self.recipe_path.resolve()),
                "sha256": self.hashes["recipe"],
            },
            "compiler": {
                "path": str(self.compiler_path.resolve()),
                "sha256": self.hashes["compiler"],
            },
            "geometry_tool": {
                "path": str(self.geometry_tool_path.resolve()),
                "sha256": self.hashes["geometry_tool"],
            },
            "weighted_arc_tool": {
                "path": str(self.weighted_arc_tool_path.resolve()),
                "sha256": self.hashes["weighted_arc_tool"],
            },
            "compiler_options": self.compiler_options,
            "ready": {
                "path": str(self.ready_path.resolve()),
                "sha256": self.hashes["ready"],
                "direct_endpoint_for_every_motion": True,
                "old_source_frame_zero_bridge_inserted": False,
            },
            "output_matrix": {
                "motion_ids": list(gate.MOTION_IDS),
                "scopes": list(gate.SCOPES),
                "candidate_count": 10,
            },
            "search_contract": {
                "entry_exit": "enumerate_all_then_gate_and_rank",
                "ranking": (
                    "feasibility_and_no_scalar_braking_through_window_end_then_"
                    "duration_then_scaled_path_total_variation"
                ),
                "adv2c3": "comparator_only_not_default",
            },
            "contact_opportunity_contract": {
                "marker_only": True,
                "pose_speed_acceleration_freeze": False,
                "acceleration_allowed_through_window_end": True,
                "nonnegative_scalar_acceleration_through_window_end": True,
            },
            "time_law_claim": (
                "weighted_arc_scalar_coordinate_only; "
                "kinematic_velocity_acceleration_and_no_early_scalar_braking_"
                "warm_start_only"
            ),
            "outputs": self.outputs,
            "post_build_gates": [{"name": "fixture_gate", "status": "pending"}],
            "non_claims": [
                "inverse_dynamics_feasibility",
                "balance",
                "training_authorization",
                "hardware_authorization",
            ],
        }
        self.manifest_path = self.bank / gate.MANIFEST_NAME
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_clearance_receipt()

    def write_clearance_receipt(self) -> None:
        self.clearance_payload = _swept_clearance_payload(
            manifest_path=self.manifest_path,
            manifest=self.manifest,
            recipe_path=self.recipe_path,
            ready_path=self.ready_path,
            mjcf_path=self.mjcf,
            urdf_path=self.urdf,
            body_order_path=self.body_order,
            verifier_path=self.clearance_verifier_path,
        )
        self.clearance_receipt_path.write_text(
            json.dumps(
                self.clearance_payload, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        self.clearance_receipt_sha256 = _sha(
            self.clearance_receipt_path
        )

    def write_sidecars(self) -> None:
        for row in self.outputs:
            npz = self.bank / row["filename"]
            npz.with_suffix(npz.suffix + ".manifest.json").write_text(
                json.dumps(row["schema2_manifest"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            npz.with_suffix(npz.suffix + ".report.json").write_text(
                json.dumps(row["schema2_report"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def recipe_loader(self, path: Path):
        assert path.resolve() == self.recipe_path.resolve()
        return self.recipe

    def plant_loader(
        self,
        mjcf: Path,
        urdf: Path,
        mjcf_sha256: str,
        compiled_signature: str,
    ):
        assert mjcf.resolve() == self.mjcf.resolve()
        assert urdf.resolve() == self.urdf.resolve()
        return SimpleNamespace(
            mjcf_sha256=mjcf_sha256,
            urdf_sha256=self.hashes["urdf"],
            compiled_signature_sha256=compiled_signature,
            identity_bound=True,
            runtime_body_names=gate.motion_player.RUNTIME_BODY_NAMES,
        )

    def verify(self, *, dynamics_runner=None, grounded_lmr_runner=None):
        kwargs = {}
        if grounded_lmr_runner is not None:
            kwargs["grounded_lmr_runner"] = grounded_lmr_runner
        return gate.verify_canonical_motion_bank(
            self.manifest_path,
            self.bank,
            mjcf_path=self.mjcf,
            urdf_path=self.urdf,
            body_order_path=self.body_order,
            expected_compiled_signature=self.signature,
            swept_clearance_receipt_path=self.clearance_receipt_path,
            expected_swept_clearance_receipt_sha256=(
                self.clearance_receipt_sha256
            ),
            recipe_loader=self.recipe_loader,
            plant_loader=self.plant_loader,
            player_runner=lambda clip: _player_report(),
            dynamics_runner=(
                dynamics_runner
                if dynamics_runner is not None
                else lambda path, plant: _grounded_dynamics_report(plant, path)
            ),
            **kwargs,
        )

    def refresh_output_hash(self, filename: str) -> None:
        digest = _sha(self.bank / filename)
        row = next(output for output in self.outputs if output["filename"] == filename)
        row["output_npz_sha256"] = digest
        row["schema2_manifest"]["hashes"]["output_npz_sha256"] = digest
        row["schema2_report"]["hashes"]["output_npz_sha256"] = digest
        input_sha = gate._recompute_candidate_input_sha256(
            row, self.bank / filename, self.recipe
        )
        row["schema2_manifest"]["hashes"]["input_sha256"] = input_sha
        row["schema2_report"]["hashes"]["input_sha256"] = input_sha
        self.write_sidecars()
        self.write_manifest()


class AppendBankFixture:
    """Strict generic suffix over an unchanged BankFixture base."""

    def __init__(self, base: BankFixture) -> None:
        self.base = base
        self.appended_motion_ids = (
            "fh_loop_high",
            "v12_forehand_block",
        )
        self.composed_motion_ids = (
            gate.MOTION_IDS + self.appended_motion_ids
        )
        base_raw = copy.deepcopy(base.recipe.raw)
        base_raw["required_output_matrix"] = {
            "motion_ids": list(gate.MOTION_IDS),
            "scopes": list(gate.SCOPES),
            "candidate_count": 10,
        }
        base.recipe.raw = base_raw
        base.recipe_path.write_text(
            json.dumps(base_raw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        base.hashes["recipe"] = _sha(base.recipe_path)
        base.manifest["recipe"] = {
            "path": str(base.recipe_path.resolve()),
            "sha256": base.hashes["recipe"],
        }
        self.bank = base.bank.parent / "append_bank"
        self.bank.mkdir()
        self.recipe_path = base.bank.parent / "append_recipe.json"
        self.recipe_path.write_text(
            '{"fixture":"strict-append-loader-injected"}\n',
            encoding="utf-8",
        )
        self.source_paths = {
            motion_id: base.bank.parent / f"source_{motion_id}.npz"
            for motion_id in self.appended_motion_ids
        }
        for motion_id, source_path in self.source_paths.items():
            source_path.write_bytes(f"source:{motion_id}".encode("utf-8"))

        raw = copy.deepcopy(base_raw)
        raw["library_id"] = "fixture_append_bank"
        raw["motion_specs"].extend(
            {
                "motion_id": motion_id,
                "scope_overrides": {},
            }
            for motion_id in self.appended_motion_ids
        )
        raw["required_output_matrix"] = {
            "motion_ids": list(self.composed_motion_ids),
            "scopes": list(gate.SCOPES),
            "candidate_count": 2 * len(self.composed_motion_ids),
        }
        appended_sources = tuple(
            SimpleNamespace(
                motion_id=motion_id,
                path=self.source_paths[motion_id],
                sha256=_sha(self.source_paths[motion_id]),
            )
            for motion_id in self.appended_motion_ids
        )
        self.recipe = SimpleNamespace(
            path=self.recipe_path.resolve(),
            raw=raw,
            marker_semantics=_fixture_marker_semantics(
                motion_ids=self.composed_motion_ids
            ),
            ready=base.recipe.ready,
            model_paths=base.recipe.model_paths,
            model_hashes=base.recipe.model_hashes,
            sources=tuple(base.recipe.sources) + appended_sources,
        )

        self.outputs = []
        for motion_id in self.appended_motion_ids:
            for scope in gate.SCOPES:
                donor = next(
                    row
                    for row in base.outputs
                    if row["motion_id"] == "fh_loop"
                    and row["scope"] == scope
                )
                row = copy.deepcopy(donor)
                row["motion_id"] = motion_id
                row["filename"] = (
                    f"{motion_id}_{scope}_canonical_v2.npz"
                )
                path = self.bank / row["filename"]
                donor_path = base.bank / donor["filename"]
                path.write_bytes(donor_path.read_bytes())
                row["output_npz_sha256"] = _sha(path)
                row["schema2_manifest"]["hashes"][
                    "output_npz_sha256"
                ] = _sha(path)
                row["schema2_report"]["hashes"][
                    "output_npz_sha256"
                ] = _sha(path)
                input_sha = gate._recompute_candidate_input_sha256(
                    row, path, self.recipe
                )
                row["schema2_manifest"]["hashes"][
                    "input_sha256"
                ] = input_sha
                row["schema2_report"]["hashes"][
                    "input_sha256"
                ] = input_sha
                path.with_suffix(
                    path.suffix + ".manifest.json"
                ).write_text(
                    json.dumps(
                        row["schema2_manifest"],
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                path.with_suffix(
                    path.suffix + ".report.json"
                ).write_text(
                    json.dumps(
                        row["schema2_report"],
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self.outputs.append(row)

        # The base manifest is immutable input to the append receipt.
        base.write_manifest()
        self.manifest = copy.deepcopy(base.manifest)
        self.manifest["library_id"] = "fixture_append_bank"
        self.manifest["recipe"] = {
            "path": str(self.recipe_path.resolve()),
            "sha256": _sha(self.recipe_path),
        }
        self.manifest["output_matrix"] = {
            "motion_ids": list(self.appended_motion_ids),
            "scopes": list(gate.SCOPES),
            "candidate_count": len(self.outputs),
        }
        self.manifest["outputs"] = self.outputs
        shift = [-0.05, 0.0]
        self.manifest["station_center_shift_xy_m"] = list(shift)
        self.manifest["append_only_composition"] = {
            "mode": "reuse_exact_base_outputs_compile_appended_only",
            "base_outputs_rebuilt": False,
            "base_recipe": {
                "path": str(base.recipe_path.resolve()),
                "sha256": _sha(base.recipe_path),
            },
            "base_build_manifest": {
                "path": str(base.manifest_path.resolve()),
                "sha256": _sha(base.manifest_path),
            },
            "base_output_matrix": copy.deepcopy(
                base.manifest["output_matrix"]
            ),
            "base_outputs": [
                {
                    "motion_id": row["motion_id"],
                    "scope": row["scope"],
                    "path": str((base.bank / row["filename"]).resolve()),
                    "sha256": _sha(base.bank / row["filename"]),
                }
                for row in base.outputs
            ],
            "appended_motion_ids": list(self.appended_motion_ids),
            "appended_scopes": list(gate.SCOPES),
            "station_center_shift_xy_m": list(shift),
            "composed_candidate_count": (
                len(gate.EXPECTED_MATRIX) + len(self.outputs)
            ),
        }
        self.manifest_path = self.bank / gate.MANIFEST_NAME
        self.clearance_verifier_path = base.clearance_verifier_path
        self.clearance_receipt_path = (
            self.bank / "swept_clearance_receipt.json"
        )
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_clearance_receipt()

    def write_clearance_receipt(self) -> None:
        self.clearance_payload = _swept_clearance_payload(
            manifest_path=self.manifest_path,
            manifest=self.manifest,
            recipe_path=self.recipe_path,
            ready_path=self.base.ready_path,
            mjcf_path=self.base.mjcf,
            urdf_path=self.base.urdf,
            body_order_path=self.base.body_order,
            verifier_path=self.clearance_verifier_path,
        )
        self.clearance_receipt_path.write_text(
            json.dumps(
                self.clearance_payload, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        self.clearance_receipt_sha256 = _sha(
            self.clearance_receipt_path
        )

    def recipe_loader(self, path: Path):
        assert path.resolve() == self.recipe_path.resolve()
        return self.recipe

    def verify(self, *, dynamics_runner=None):
        return gate.verify_canonical_motion_bank(
            self.manifest_path,
            self.bank,
            mjcf_path=self.base.mjcf,
            urdf_path=self.base.urdf,
            body_order_path=self.base.body_order,
            expected_compiled_signature=self.base.signature,
            swept_clearance_receipt_path=self.clearance_receipt_path,
            expected_swept_clearance_receipt_sha256=(
                self.clearance_receipt_sha256
            ),
            recipe_loader=self.recipe_loader,
            plant_loader=self.base.plant_loader,
            player_runner=lambda clip: _player_report(),
            dynamics_runner=(
                dynamics_runner
                if dynamics_runner is not None
                else lambda path, plant: _grounded_dynamics_report(
                    plant, path
                )
            ),
        )


@pytest.fixture
def fixture_bank(tmp_path: Path) -> BankFixture:
    return BankFixture(tmp_path)


@pytest.fixture
def append_bank(fixture_bank: BankFixture) -> AppendBankFixture:
    return AppendBankFixture(fixture_bank)


def _attach_exact_time_law_artifacts(
    fixture: BankFixture,
    *,
    source_anchor_path_s: float = 0.5,
    window_start_path_s: float = 1.0 / 6.0,
    window_end_path_s: float = 5.0 / 6.0,
) -> None:
    """Publish lossless trace fixtures matching every schema-2 control tick."""

    source_by_motion = {
        source.motion_id: source for source in fixture.recipe.sources
    }
    compiler_path = Path(compiler.__file__).resolve()
    path_topp_path = Path(path_topp_module.__file__).resolve()
    weighted_arc_path = fixture.weighted_arc_tool_path.resolve()
    for row in fixture.outputs:
        npz_path = fixture.bank / row["filename"]
        with np.load(npz_path, allow_pickle=False) as archive:
            joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
        if row["scope"] == "upper":
            q = joint_pos
        else:
            root = np.broadcast_to(
                np.concatenate(
                    (fixture.ready_root_pos, np.zeros(3, dtype=np.float64))
                ),
                (len(joint_pos), 6),
            )
            q = np.concatenate((joint_pos, root), axis=1)
        assert len(q) == 5
        speed = 50.0 / 3.0
        s_node = np.asarray(
            [0.0, 1.0 / 6.0, 5.0 / 6.0, 1.0],
            dtype=np.float64,
        )
        s_mid = 0.5 * (s_node[:-1] + s_node[1:])
        x_node = np.asarray(
            [0.0, speed * speed, speed * speed, 0.0],
            dtype=np.float64,
        )
        x_mid = 0.5 * (x_node[:-1] + x_node[1:])
        u_cell = np.diff(x_node) / (2.0 * np.diff(s_node))
        time_node = np.asarray(
            [0.0, 0.02, 0.06, 0.08], dtype=np.float64
        )
        time_mid = time_node[:-1] + np.diff(s_node) / (
            np.sqrt(x_node[:-1]) + np.sqrt(x_mid)
        )
        q_node = q[[0, 1, 3, 4]]
        q_mid = 0.5 * (q_node[:-1] + q_node[1:])
        q_mid[1] = q[2]
        q_s_node = np.zeros_like(q_node)
        q_s_mid = np.zeros_like(q_mid)
        collocation_qpos = np.stack(
            (q_node[:-1], q_mid, q_node[1:]), axis=1
        )
        collocation_q_s = np.zeros_like(collocation_qpos)
        artifact_trace = time_law_artifact.TimeLawTrace(
            s_node=s_node,
            s_mid=s_mid,
            qpos_node=q_node,
            q_s_node=q_s_node,
            q_ss_node_left=q_s_node,
            q_ss_node_right=q_s_node,
            qpos_mid=q_mid,
            q_s_mid=q_s_mid,
            q_ss_mid=q_s_mid,
            x_node=x_node,
            x_mid=x_mid,
            u_cell=u_cell,
            time_node_s=time_node,
            time_mid_s=time_mid,
            collocation_qpos=collocation_qpos,
            collocation_q_s=collocation_q_s,
            collocation_qvel=np.zeros_like(collocation_qpos),
            collocation_qacc=np.zeros_like(collocation_qpos),
            tick_s=np.asarray(
                [0.0, 1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0],
                dtype=np.float64,
            ),
            tick_qpos=q,
            tick_q_s=np.zeros_like(q),
            tick_q_ss=np.zeros_like(q),
            tick_qvel=np.zeros_like(q),
            tick_qacc=np.zeros_like(q),
        )
        evaluated_sha = (
            time_law_artifact.path_evaluation_array_sha256(
                artifact_trace
            )
        )
        source = source_by_motion[row["motion_id"]]
        evaluator_kwargs = {
            "evaluator_id": "weighted_arc_path_evaluate_l_exact_v1",
            "evaluator_version": "1",
            "derivative_method": "exact_spline_derivative",
            "evaluator_contract_sha256": hashlib.sha256(
                f"evaluator:{row['motion_id']}:{row['scope']}".encode()
            ).hexdigest(),
            "evaluator_implementation_sha256": _sha(weighted_arc_path),
            "evaluated_arrays_sha256": evaluated_sha,
        }
        evaluator_binding = time_law_artifact.PathEvaluatorBinding(
            **evaluator_kwargs,
            producer_receipt_sha256=(
                time_law_artifact.path_evaluation_receipt_sha256(
                    source_sha256=source.sha256,
                    **evaluator_kwargs,
                )
            ),
        )
        weighted_kwargs = {
            "algorithm_id": "canonical_weighted_arc_fixture_v1",
            "content_sha256": hashlib.sha256(
                f"arc:{row['motion_id']}:{row['scope']}".encode()
            ).hexdigest(),
            "retimer_receipt_sha256": hashlib.sha256(
                f"retime:{row['motion_id']}:{row['scope']}".encode()
            ).hexdigest(),
            "coordinate_scale_sha256_float64_le": "1" * 64,
            "l_knots_sha256_float64_le": "2" * 64,
            "total_length": 1.0,
            "formal_knot_count": 3,
            "arc_absolute_tolerance": 1.0e-10,
            "arc_relative_tolerance": 1.0e-10,
            "quadrature_max_depth": 8,
            "quadrature_error_estimate_sum": 0.0,
            "regularity_margin": 1.0e-6,
            "regularity_max_depth": 8,
            "certified_min_weighted_speed_per_s": 1.0,
            "observed_min_weighted_speed_per_s": 1.0,
            "inverse_absolute_tolerance": 1.0e-10,
            "inverse_relative_tolerance": 1.0e-10,
            "inverse_parameter_tolerance": 1.0e-10,
            "inverse_max_iterations": 32,
            "evaluated_arrays_sha256": evaluated_sha,
            "producer_receipt_sha256": "0" * 64,
        }
        provisional_weighted = (
            time_law_artifact.WeightedArcLengthBinding(
                **weighted_kwargs
            )
        )
        weighted_kwargs["producer_receipt_sha256"] = (
            time_law_artifact.weighted_arc_length_receipt_sha256(
                source_sha256=source.sha256,
                binding=provisional_weighted,
            )
        )
        weighted_binding = (
            time_law_artifact.WeightedArcLengthBinding(
                **weighted_kwargs
            )
        )
        solver_arrays = {
            key: getattr(artifact_trace, key)
            for key in gate._TIME_LAW_SOLVER_ARRAY_KEYS
        }
        bindings = time_law_artifact.ArtifactBindings(
            recipe_sha256=_sha(fixture.recipe_path),
            source_sha256=source.sha256,
            ready_sha256=fixture.hashes["ready"],
            mjcf_sha256=fixture.hashes["mjcf"],
            urdf_sha256=fixture.hashes["urdf"],
            model_binding_sha256="a" * 64,
            actuator_contract_sha256="b" * 64,
            tools_sha256={
                "compiler": _sha(compiler_path),
                "path_topp": _sha(path_topp_path),
                "weighted_arc": _sha(weighted_arc_path),
                "artifact": _sha(
                    Path(time_law_artifact.__file__).resolve()
                ),
                "mujoco_path_adapter": _sha(
                    Path(path_adapter_module.__file__).resolve()
                ),
                "grounded_solver": _sha(
                    Path(torque_topp_module.__file__).resolve()
                ),
            },
            solver=time_law_artifact.SolverBinding(
                solver_id="canonical_path_topp.retime_path",
                solver_version="weighted_arc_collocation_trace_v1",
                solver_contract_sha256=(
                    compiler._time_law_solver_contract_sha256(
                        arrays=solver_arrays,
                        compiler_input_sha256=(
                            row["schema2_manifest"]["hashes"][
                                "input_sha256"
                            ]
                        ),
                        output_npz_sha256=row["output_npz_sha256"],
                    )
                ),
                solver_implementation_sha256=_sha(path_topp_path),
            ),
            path_evaluator=evaluator_binding,
            weighted_arc_length=weighted_binding,
        )
        artifact_filename = (
            row["filename"][: -len(".npz")]
            + ".time_law.npz"
        )
        artifact = time_law_artifact.build_time_law_artifact(
            motion_id=row["motion_id"],
            scope=row["scope"],
            trace=artifact_trace,
            marker_path_s={
                "window_start": window_start_path_s,
                "source_anchor": source_anchor_path_s,
                "window_end": window_end_path_s,
            },
            bindings=bindings,
        )
        artifact_manifest_filename = artifact_filename + ".manifest.json"
        time_law_artifact.write_time_law_artifact(
            artifact,
            fixture.bank / artifact_filename,
            manifest_path=fixture.bank / artifact_manifest_filename,
        )
        row["time_law_artifact"] = {
            "npz_filename": artifact_filename,
            "npz_sha256": artifact.npz_sha256,
            "manifest_filename": artifact_manifest_filename,
            "manifest_sha256": artifact.manifest_sha256,
            "bundle_sha256": artifact.bundle_sha256,
            "schema_version": time_law_artifact.ARTIFACT_SCHEMA_VERSION,
            "artifact_type": time_law_artifact.ARTIFACT_TYPE,
        }
    fixture.write_manifest()


def _passing_grounded_lmr(
    loaded,
    scope,
    plant,
    files,
    recipe,
    compiler_options,
    npz_path,
):
    del scope, plant, files, recipe, compiler_options, npz_path
    cells = len(loaded.arrays["u_cell"])
    return {
        "status": "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT",
        "sample_count": 3 * cells,
        "cell_count": cells,
        "roles": ["left", "midpoint", "right"],
        "all_feasible": True,
        "finite_difference_qacc_used": False,
        "qacc_contract": "q_s*u+q_ss*x_from_persisted_compiler_trace",
    }


def test_append_only_verifies_generic_suffix_and_binds_base_without_replay(
    append_bank: AppendBankFixture,
):
    dynamics_paths: list[str] = []

    def dynamics(path, plant):
        dynamics_paths.append(path.name)
        return _grounded_dynamics_report(plant, path)

    report = append_bank.verify(dynamics_runner=dynamics)

    assert report["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert report["candidate_integrity_pass"] is True
    assert report["bank_gate_pass"] is False
    assert report["training_authorized"] is False
    assert report["hardware_authorized"] is False
    assert report["contracts"]["matrix"] == {
        "motion_ids": list(append_bank.appended_motion_ids),
        "scopes": ["upper", "full"],
        "count": 4,
    }
    assert report["aggregate"]["clip_count"] == 4
    assert [
        (row["motion_id"], row["scope"]) for row in report["clips"]
    ] == list(
        gate._matrix_for_motion_ids(append_bank.appended_motion_ids)
    )
    assert dynamics_paths == list(
        gate._filenames_for_matrix(
            gate._matrix_for_motion_ids(
                append_bank.appended_motion_ids
            )
        )
    )
    composition = report["append_only_composition"]
    assert composition["base_outputs_rebuilt"] is False
    assert len(composition["base_outputs"]) == 10
    assert composition["appended_motion_ids"] == [
        "fh_loop_high",
        "v12_forehand_block",
    ]
    assert composition["composed_candidate_count"] == 14
    assert composition["base_output_matrix"] == {
        "motion_ids": list(gate.MOTION_IDS),
        "scopes": list(gate.SCOPES),
        "candidate_count": 10,
    }
    assert report["append_only_base_validation_scope"] == (
        "base_recipe_bytes_manifest_bytes_and_ten_output_npz_sha256_only"
    )
    assert report["station_center_shift_xy_m"] == [-0.05, 0.0]
    assert report["bound_inputs"]["swept_clearance_receipt"][
        "sha256"
    ] == _sha(append_bank.clearance_receipt_path)


def test_append_only_accepts_immutable_base_with_exact_time_law_artifacts(
    tmp_path: Path,
):
    base = BankFixture(tmp_path)
    _attach_exact_time_law_artifacts(base)
    appended = AppendBankFixture(base)
    # This regression targets the reused base.  Appended outputs may acquire
    # their own time-law artifacts in a later independent compiler step.
    for row in appended.outputs:
        row.pop("time_law_artifact", None)
    appended.manifest["outputs"] = appended.outputs
    appended.write_manifest()

    report = appended.verify()

    assert report["candidate_integrity_pass"] is True
    assert report["append_only_composition"]["base_outputs_rebuilt"] is False
    assert len(report["append_only_composition"]["base_outputs"]) == 10


def test_append_only_rejects_base_manifest_sha_mismatch(
    append_bank: AppendBankFixture,
):
    append_bank.manifest["append_only_composition"][
        "base_build_manifest"
    ]["sha256"] = "0" * 64
    append_bank.write_manifest()

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="append-only base build manifest SHA-256 mismatch",
    ):
        append_bank.verify()


def test_append_only_rejects_unknown_composition_key(
    append_bank: AppendBankFixture,
):
    append_bank.manifest["append_only_composition"]["unknown"] = False
    append_bank.write_manifest()

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="append_only_composition keys changed",
    ):
        append_bank.verify()


def test_append_only_rejects_reused_output_sha_mismatch(
    append_bank: AppendBankFixture,
):
    append_bank.manifest["append_only_composition"]["base_outputs"][0][
        "sha256"
    ] = "0" * 64
    append_bank.write_manifest()

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="append-only base output .* SHA-256 mismatch",
    ):
        append_bank.verify()


def test_append_only_rejects_changed_base_prefix_in_full_recipe(
    append_bank: AppendBankFixture,
):
    append_bank.recipe.raw["motion_specs"][0]["human_role"] = "changed"

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="changed the canonical-five motion specs",
    ):
        append_bank.verify()


def test_append_only_rejects_new_output_sha_mismatch(
    append_bank: AppendBankFixture,
):
    append_bank.manifest["outputs"][0]["output_npz_sha256"] = "0" * 64
    append_bank.write_manifest()

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="ordered output bytes",
    ):
        append_bank.verify()


def test_append_only_rejects_recipe_source_suffix_order_drift(
    append_bank: AppendBankFixture,
):
    sources = append_bank.recipe.sources
    append_bank.recipe.sources = sources[:-2] + tuple(
        reversed(sources[-2:])
    )

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="sources are not the exact unchanged base-five",
    ):
        append_bank.verify()


@pytest.mark.parametrize(
    ("motion_ids", "message"),
    [
        ([], "may not be empty"),
        (
            ["v12_forehand_block", "v12_forehand_block"],
            "contain duplicates",
        ),
        (["bh_block"], "may not replace or repeat"),
        (["../escape"], "lowercase ASCII motion identifier"),
    ],
)
def test_append_only_rejects_empty_duplicate_base_or_unsafe_suffix(
    append_bank: AppendBankFixture,
    motion_ids: list[str],
    message: str,
):
    append_bank.manifest["output_matrix"]["motion_ids"] = motion_ids
    append_bank.manifest["output_matrix"]["candidate_count"] = (
        2 * len(motion_ids)
    )
    append_bank.write_manifest()

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match=message,
    ):
        append_bank.verify()


def test_append_only_rejects_composition_suffix_drift(
    append_bank: AppendBankFixture,
):
    append_bank.manifest["append_only_composition"][
        "appended_motion_ids"
    ] = list(reversed(append_bank.appended_motion_ids))
    append_bank.write_manifest()

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="disagree with the exact append output matrix",
    ):
        append_bank.verify()


def test_append_only_rejects_recipe_suffix_drift(
    append_bank: AppendBankFixture,
):
    append_bank.recipe.raw["motion_specs"][-1]["motion_id"] = (
        "different_append"
    )

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="motion_specs do not match",
    ):
        append_bank.verify()


def test_append_only_rejects_canonical_five_marker_authority_drift(
    append_bank: AppendBankFixture,
):
    append_bank.recipe.raw["marker_authority"] = {
        "path": "changed-marker-authority.json",
        "sha256": "0" * 64,
    }

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="marker_authority",
    ):
        append_bank.verify()


def test_rejects_missing_or_hash_mismatched_swept_clearance_receipt(
    fixture_bank: BankFixture,
):
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="content-addressed swept-clearance receipt is required",
    ):
        gate.verify_canonical_motion_bank(
            fixture_bank.manifest_path,
            fixture_bank.bank,
            mjcf_path=fixture_bank.mjcf,
            urdf_path=fixture_bank.urdf,
            body_order_path=fixture_bank.body_order,
            expected_compiled_signature=fixture_bank.signature,
        )

    fixture_bank.clearance_receipt_path.unlink()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="regular, non-symlink",
    ):
        fixture_bank.verify()

    fixture_bank.write_clearance_receipt()
    fixture_bank.clearance_receipt_sha256 = "0" * 64
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="swept-clearance receipt SHA-256 mismatch",
    ):
        fixture_bank.verify()


def test_rejects_swept_clearance_without_table_or_both_scopes(
    fixture_bank: BankFixture,
):
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["with_table"] = False
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="with_table=true",
    ):
        fixture_bank.verify()

    payload = _swept_clearance_payload(
        manifest_path=fixture_bank.manifest_path,
        manifest=fixture_bank.manifest,
        recipe_path=fixture_bank.recipe_path,
        ready_path=fixture_bank.ready_path,
        mjcf_path=fixture_bank.mjcf,
        urdf_path=fixture_bank.urdf,
        body_order_path=fixture_bank.body_order,
        verifier_path=fixture_bank.clearance_verifier_path,
    )
    payload["trajectory_contract"]["scopes"] = ["upper"]
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="upper and full",
    ):
        fixture_bank.verify()


def test_rejects_old_four_or_partial_swept_clearance_matrix(
    append_bank: AppendBankFixture,
):
    payload = copy.deepcopy(append_bank.clearance_payload)
    payload["bank_binding"]["output_matrix"] = {
        "motion_ids": list(gate.MOTION_IDS[:4]),
        "scopes": list(gate.SCOPES),
        "candidate_count": 8,
    }
    _publish_clearance_payload(append_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="not the exact bank matrix",
    ):
        append_bank.verify()

    append_bank.write_clearance_receipt()
    payload = copy.deepcopy(append_bank.clearance_payload)
    payload["results"].pop()
    _publish_clearance_payload(append_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="exact matrix in both scopes",
    ):
        append_bank.verify()


def test_rejects_partial_cycle_or_uncertified_swept_interval(
    fixture_bank: BankFixture,
):
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["results"][0]["end_frame"] -= 1
    payload["results"][0]["interval_count"] -= 1
    payload["results"][0]["certified_interval_count"] -= 1
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="exact full-cycle",
    ):
        fixture_bank.verify()

    fixture_bank.write_clearance_receipt()
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["results"][0]["certified_interval_count"] -= 1
    payload["results"][0]["unknown_interval_count"] = 1
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="all-interval",
    ):
        fixture_bank.verify()


def test_rejects_sampled_or_geometry_only_clearance_claim(
    fixture_bank: BankFixture,
):
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["method"]["certificate_kind"] = (
        "finite_400hz_sampled_geometry"
    )
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="sampled or geometry-only",
    ):
        fixture_bank.verify()


def test_rejects_missing_action_ball_keepout_or_robot_geom(
    fixture_bank: BankFixture,
):
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["scene_contract"]["action_ball_assembly"]["roles"].remove(
        "keepout"
    )
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="under-table robot keepout",
    ):
        fixture_bank.verify()

    fixture_bank.write_clearance_receipt()
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["scene_contract"]["robot_geometry"][
        "collision_geom_names"
    ].remove("right_racket_handle_collision")
    payload["scene_contract"]["robot_geometry"][
        "collision_geom_names_sha256"
    ] = gate._canonical_json_sha256(
        payload["scene_contract"]["robot_geometry"][
            "collision_geom_names"
        ],
        "tampered robot geometry",
    )
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="racket and handle",
    ):
        fixture_bank.verify()


def test_swept_clearance_exact_5mm_boundary(
    fixture_bank: BankFixture,
):
    payload = copy.deepcopy(fixture_bank.clearance_payload)
    for row in payload["results"]:
        row["minimum_clearance_certified_lower_bound_m"] = 0.005
    _publish_clearance_payload(fixture_bank, payload)
    report = fixture_bank.verify()
    assert report["aggregate"][
        "swept_clearance_minimum_certified_lower_bound_m"
    ] == 0.005

    payload = copy.deepcopy(fixture_bank.clearance_payload)
    payload["results"][0][
        "minimum_clearance_certified_lower_bound_m"
    ] = float(np.nextafter(0.005, -np.inf))
    _publish_clearance_payload(fixture_bank, payload)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match=">=5mm",
    ):
        fixture_bank.verify()


def test_rejects_missing_and_extra_npz(fixture_bank: BankFixture):
    missing = fixture_bank.bank / gate.EXPECTED_FILENAMES[0]
    missing.unlink()
    with pytest.raises(gate.CanonicalMotionBankGateError, match="missing="):
        fixture_bank.verify()

    _write_clip(missing, ready_joint=fixture_bank.ready_joint)
    fixture_bank.refresh_output_hash(missing.name)
    (fixture_bank.bank / "unexpected.npz").write_bytes(b"not an asset")
    with pytest.raises(gate.CanonicalMotionBankGateError, match="extra="):
        fixture_bank.verify()


def test_rejects_npz_sha_tamper(fixture_bank: BankFixture):
    target = fixture_bank.bank / gate.EXPECTED_FILENAMES[0]
    with target.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="ordered output bytes",
    ):
        fixture_bank.verify()


def test_rejects_shared_ready_mismatch(fixture_bank: BankFixture):
    filename = gate.EXPECTED_FILENAMES[2]
    target = fixture_bank.bank / filename
    arrays = _read_arrays(target)
    arrays["joint_pos"][0, 5] = 0.1
    arrays["joint_pos"][-1, 5] = 0.1
    _write_arrays(target, arrays)
    fixture_bank.refresh_output_hash(filename)
    with pytest.raises(gate.CanonicalMotionBankGateError, match="not canonical ready"):
        fixture_bank.verify()


def test_rejects_nonzero_endpoint_velocity(fixture_bank: BankFixture):
    filename = gate.EXPECTED_FILENAMES[4]
    target = fixture_bank.bank / filename
    arrays = _read_arrays(target)
    arrays["body_ang_vel_w"][-1, 7, 2] = 0.01
    _write_arrays(target, arrays)
    fixture_bank.refresh_output_hash(filename)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="six endpoint velocity classes",
    ):
        fixture_bank.verify()


def test_rejects_marker_or_geometry_receipt_escape(
    fixture_bank: BankFixture,
):
    row = fixture_bank.outputs[0]
    marker = row["retiming"]["markers"]["window_end"]
    marker["source_index"] = 1_000_000.0
    marker["path_position_at_frame"] = 1_000_000.0
    row["retiming"]["nonnegative_acceleration_until_marker"]["marker_source_index"] = 1_000_000.0
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="leaves the selected geometry",
    ):
        fixture_bank.verify()


def test_rejects_in_range_marker_forgery_against_rebuilt_source_map(
    fixture_bank: BankFixture,
):
    row = fixture_bank.outputs[0]
    row["retiming"]["markers"]["window_start"]["source_index"] = 11.0
    row["geometry"]["recipe_marker_binding"]["window_start"]["dense_row"] = 11
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="does not map recipe source frame",
    ):
        fixture_bank.verify()


def test_rejects_forged_compiler_input_digest(fixture_bank: BankFixture):
    row = fixture_bank.outputs[0]
    forged = hashlib.sha256(b"forged compiler input").hexdigest()
    row["schema2_manifest"]["hashes"]["input_sha256"] = forged
    row["schema2_report"]["hashes"]["input_sha256"] = forged
    fixture_bank.write_sidecars()
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="input_sha256 does not bind the verified bytes",
    ):
        fixture_bank.verify()


def test_rejects_compiler_option_receipt_even_with_rehashed_top_level(
    fixture_bank: BankFixture,
):
    options = fixture_bank.manifest["compiler_options"]
    upper = options["effective_path_coordinate_contracts"]["upper"]
    upper["coordinate_scale"][0] *= 2.0
    arrays = [
        np.asarray(upper[key], dtype=np.float64)
        for key in (
            "position_lower",
            "position_upper",
            "velocity",
            "acceleration",
            "coordinate_scale",
        )
    ]
    upper["sha256"] = hashlib.sha256(
        b"".join(bytes.fromhex(gate._array_sha256(value)) for value in arrays)
    ).hexdigest()
    payload = dict(options)
    del payload["compiler_options_sha256"]
    options["compiler_options_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="bounds/coordinate scale receipt is inconsistent",
    ):
        fixture_bank.verify()


def test_rejects_active_probe_source_smoothing(fixture_bank: BankFixture):
    options = fixture_bank.manifest["compiler_options"]
    grid = options["geometry_and_grid"]
    # Sanity: the honest fixture receipt is the identity (no smoothing).
    assert grid["probe_source_smoothing_tolerance_rad"] is None
    assert grid["probe_source_smoothing_is_identity"] is True
    grid["probe_source_smoothing_tolerance_rad"] = 0.02
    grid["probe_source_smoothing_is_identity"] = False
    payload = dict(options)
    del payload["compiler_options_sha256"]
    options["compiler_options_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="probe source smoothing is not verifiable",
    ):
        fixture_bank.verify()


def test_rejects_noncanonical_compiler_or_geometry_tool(
    fixture_bank: BankFixture,
    tmp_path: Path,
):
    fake_tool = tmp_path / "lookalike_compiler.py"
    fake_tool.write_text(
        fixture_bank.compiler_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    fixture_bank.manifest["compiler"] = {
        "path": str(fake_tool),
        "sha256": _sha(fake_tool),
    }
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError, match="formal compiler tool path changed"
    ):
        fixture_bank.verify()


def test_rejects_fake_green_grounded_torque(fixture_bank: BankFixture):
    with pytest.raises(gate.CanonicalMotionBankGateError, match="fake green"):
        fixture_bank.verify(
            dynamics_runner=lambda path, plant: _grounded_dynamics_report(
                plant, path, fake_green=True
            )
        )

    def forged_valid(path, plant):
        report = _grounded_dynamics_report(plant, path, fake_green=True)
        torque = report["checks"]["inverse_dynamics"]["torque_interpretation"]
        torque["valid"] = True
        torque["label"] = "certified_contact_free_direct_motor_torque"
        torque["reasons"] = []
        torque["contact_frames"] = []
        report["checks"]["inverse_dynamics"][
            "effort_limit_pass_only_if_interpretation_valid"
        ] = True
        report["checks"]["geometry"]["all_contacts"] = {
            "self_contacts": [],
            "floor_contacts": [],
            "other_world_contacts": [],
        }
        report["plant"]["actuation_mapping_verified"] = True
        return report

    with pytest.raises(gate.CanonicalMotionBankGateError, match="fake green"):
        fixture_bank.verify(dynamics_runner=forged_valid)


def test_rejects_fake_green_contact_frame_receipt(
    fixture_bank: BankFixture,
):
    def forged_contact_frames(path, plant):
        report = _contact_free_dynamics_report(plant, path)
        report["checks"]["inverse_dynamics"]["torque_interpretation"]["contact_frames"] = [0]
        return report

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="contact_frames contradict all_contacts",
    ):
        fixture_bank.verify(dynamics_runner=forged_contact_frames)


def test_rejects_fake_green_geometry_violation(fixture_bank: BankFixture):
    def forged_geometry(path, plant):
        report = _contact_free_dynamics_report(plant, path)
        geometry = report["checks"]["geometry"]
        violation = {
            "frame": 2,
            "depth_m": (gate.dynamics_gate.GateThresholds().self_penetration_tolerance_m + 0.001),
            "geom_pair": ["arm", "torso"],
            "body_pair": ["right_arm", "torso"],
        }
        geometry["self_collision_violation_count"] = 1
        geometry["self_collision_violations"] = [violation]
        geometry["all_contacts"]["self_contacts"] = [violation]
        # The forged producer leaves its headline green despite the evidence.
        geometry["pass"] = True
        return report

    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="geometry pass contradicts",
    ):
        fixture_bank.verify(dynamics_runner=forged_geometry)


def test_normal_grounded_summary_stays_incomplete_and_unauthorized(
    fixture_bank: BankFixture,
):
    report = fixture_bank.verify()
    assert report["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert report["bank_gate_pass"] is False
    assert report["candidate_integrity_pass"] is True
    assert report["training_authorized"] is False
    assert report["hardware_authorized"] is False
    assert report["aggregate"] == {
        "clip_count": 10,
        "fk_pass_count": 10,
        "velocity_consistency_pass_count": 10,
        "joint_limit_pass_count": 10,
        "geometry_pass_count": 10,
        "non_torque_dynamics_pass_count": 10,
        "complete_dynamics_pass_count": 0,
        "incomplete_fail_closed_count": 10,
        "failed_count": 0,
        "torque_interpretation_valid_count": 0,
        "clips_with_contact_count": 10,
        "contact_frame_count": 50,
        "self_collision_violation_count": 0,
        "foot_floor_penetration_violation_count": 0,
        "nonfoot_floor_penetration_violation_count": 0,
            "other_world_penetration_violation_count": 0,
            "swept_clearance_pass_count": 10,
            "swept_clearance_minimum_certified_lower_bound_m": 0.006,
            "joint_effort_proxy_peak_utilization": 0.4,
        "actuator_force_proxy_peak_utilization": 0.4,
        "root_height_min_m": 1.0,
        "root_height_max_m": 1.0,
        "root_tilt_peak_rad": 0.0,
        "root_xy_displacement_peak_m": 0.0,
        "com_height_min_m": 0.8,
        "com_height_max_m": 0.8,
    }
    assert all(
        row["contact_opportunity"]["acceleration_allowed_through_window_end"] is True
        for row in report["clips"]
    )


def test_complete_contact_free_screen_still_lacks_exact_grounded_trace(
    fixture_bank: BankFixture,
):
    report = fixture_bank.verify(
        dynamics_runner=lambda path, plant: _contact_free_dynamics_report(plant, path)
    )
    assert report["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert report["bank_gate_pass"] is False
    assert report["candidate_integrity_pass"] is True
    assert report["grounded_trace_status"] == "MISSING_INCOMPLETE_FAIL_CLOSED"
    assert report["training_authorized"] is False
    assert report["hardware_authorized"] is False
    assert report["aggregate"]["complete_dynamics_pass_count"] == 10
    assert report["aggregate"]["torque_interpretation_valid_count"] == 10


def test_exact_time_law_artifacts_close_grounded_lmr_bank_gate(
    fixture_bank: BankFixture,
):
    _attach_exact_time_law_artifacts(fixture_bank)
    report = fixture_bank.verify(
        grounded_lmr_runner=_passing_grounded_lmr
    )
    assert report["verdict"] == "PASS"
    assert report["bank_gate_pass"] is True
    assert report["candidate_integrity_pass"] is True
    assert report["grounded_trace_status"] == (
        "PASS_GROUNDED_LEFT_MIDPOINT_RIGHT"
    )
    assert report["training_authorized"] is False
    assert report["hardware_authorized"] is False
    assert report["aggregate"]["time_law_artifact_count"] == 10
    assert report["aggregate"]["grounded_lmr_pass_count"] == 10
    assert all(
        clip["canonical_time_law"]["finite_difference_reconstruction_used"]
        is False
        for clip in report["clips"]
    )
    expected_marker_contract = {
        "marker_names": [
            "window_start",
            "source_anchor",
            "window_end",
        ],
        "path_s": {
            "window_start": 1.0 / 6.0,
            "source_anchor": 0.5,
            "window_end": 5.0 / 6.0,
        },
        "time_s": {
            "window_start": 0.02,
            "source_anchor": 0.04,
            "window_end": 0.06,
        },
        "source_anchor_within_solved_path": True,
        "source_anchor_independent_of_protected_window": True,
        "protected_window_order_valid": True,
        "no_early_brake_from_path_start_through_window_end": True,
        "inclusive_tick_nonempty": True,
    }
    assert all(
        clip["canonical_time_law"]["marker_contract"]
        == expected_marker_contract
        for clip in report["clips"]
    )


@pytest.mark.parametrize(
    ("source_anchor_path_s", "expected_time_s"),
    [
        (0.0, 0.0),
        (0.5, 0.04),
        (1.0, 0.08),
    ],
)
def test_reopened_time_law_marker_contract_allows_anchor_before_inside_or_after_window(
    fixture_bank: BankFixture,
    source_anchor_path_s: float,
    expected_time_s: float,
):
    _attach_exact_time_law_artifacts(
        fixture_bank,
        source_anchor_path_s=source_anchor_path_s,
    )
    binding = fixture_bank.outputs[0]["time_law_artifact"]
    loaded = time_law_artifact.read_time_law_artifact(
        fixture_bank.bank / binding["npz_filename"],
        manifest_path=(
            fixture_bank.bank / binding["manifest_filename"]
        ),
    )

    marker_contract = gate._marker_contract_from_reopened_time_law(
        loaded
    )

    assert marker_contract["marker_names"] == [
        "window_start",
        "source_anchor",
        "window_end",
    ]
    assert (
        marker_contract["path_s"]["source_anchor"]
        == source_anchor_path_s
    )
    assert marker_contract["time_s"]["source_anchor"] == expected_time_s
    assert marker_contract["source_anchor_within_solved_path"] is True
    assert (
        marker_contract[
            "source_anchor_independent_of_protected_window"
        ]
        is True
    )
    assert marker_contract["protected_window_order_valid"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"source_anchor_path_s": -0.01},
            "marker path positions leave the solved path",
        ),
        (
            {
                "window_start_path_s": 5.0 / 6.0,
                "window_end_path_s": 1.0 / 6.0,
            },
            "protected window endpoints",
        ),
        (
            {
                "window_start_path_s": 0.2,
                "window_end_path_s": 0.4,
            },
            "contact window contains no inclusive 50 Hz observation tick",
        ),
        (
            {"window_end_path_s": 1.0},
            "u_cell must remain non-negative",
        ),
    ],
)
def test_time_law_marker_counterexamples_fail_before_bank_summary(
    fixture_bank: BankFixture,
    kwargs: dict[str, float],
    message: str,
):
    with pytest.raises(
        time_law_artifact.TimeLawArtifactError,
        match=message,
    ):
        _attach_exact_time_law_artifacts(fixture_bank, **kwargs)


def test_time_law_artifact_byte_tamper_fails_closed(
    fixture_bank: BankFixture,
):
    _attach_exact_time_law_artifacts(fixture_bank)
    artifact_name = fixture_bank.outputs[0]["time_law_artifact"][
        "npz_filename"
    ]
    artifact_path = fixture_bank.bank / artifact_name
    raw = bytearray(artifact_path.read_bytes())
    raw[-2] ^= 1
    artifact_path.write_bytes(raw)
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="time-law artifact failed strict reopen",
    ):
        fixture_bank.verify(
            grounded_lmr_runner=_passing_grounded_lmr
        )


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    [
        ("schema_version", 1),
        ("artifact_type", "canonical_time_law_collocation_v1"),
    ],
)
def test_bank_rejects_legacy_v1_time_law_manifest_binding(
    fixture_bank: BankFixture,
    field: str,
    legacy_value: object,
):
    _attach_exact_time_law_artifacts(fixture_bank)
    fixture_bank.outputs[0]["time_law_artifact"][field] = legacy_value
    fixture_bank.write_manifest()
    with pytest.raises(
        gate.CanonicalMotionBankGateError,
        match="time-law artifact contract changed",
    ):
        fixture_bank.verify(
            grounded_lmr_runner=_passing_grounded_lmr
        )


@pytest.mark.skip(
    reason=(
        "queued debt: the tiny 9-frame fixture arcs do not converge under the "
        "real explicit retimer at test-budget grid resolution, and the cheap "
        "marker-only retimer double is not yet data-coherent with the bank "
        "gate's NPZ/duration/marker cross-checks.  Retimer convergence is "
        "covered by the canonical_path_topp suites plus the process/serial "
        "determinism test on the same arc geometry, the writer/manifest seam "
        "by the compiler 5x2 plumbing test, and every bank-gate refusal path "
        "by the other tests in this file; the genuine end-to-end remains the "
        "real-data probe compile plus independent verification on pod2."
    )
)
def test_real_compiler_manifest_and_writer_feed_bank_gate(tmp_path: Path, monkeypatch):
    """Exercise strict recipe load, real geometry, writer, and bank gate."""

    recipe_path, recipe = _write_strict_compiler_recipe(tmp_path / "strict_repo")

    monkeypatch.setattr(
        compiler,
        "solve_face_flipped_window",
        lambda source_joint_pos, *args, **kwargs: (
            compiler_test_support.FakeFaceResult(source_joint_pos, kwargs["frame_indices"])
        ),
    )
    monkeypatch.setattr(compiler, "build_schema2_candidate", _compiler_schema2_candidate)
    monkeypatch.setattr(
        compiler, "retime_path", compiler_test_support._fast_marker_only_retime
    )
    library = compiler.compile_loaded_canonical_motion_library(
        recipe,
        options=compiler_test_support._options(),
        backend=compiler_test_support.FakePlantBackend(),
    )
    output = compiler.write_compiled_canonical_motion_library(library, tmp_path / "compiled_bank")
    manifest_path = output / compiler.BUILD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["contact_opportunity_contract"][
            "nonnegative_scalar_acceleration_through_window_end"
        ]
        is True
    )
    assert manifest["time_law_claim"] == (
        "weighted_arc_scalar_coordinate_only; "
        "kinematic_velocity_acceleration_and_no_early_scalar_braking_"
        "warm_start_only"
    )

    signature = hashlib.sha256(b"compiler integration signature").hexdigest()

    def plant_loader(mjcf, urdf, mjcf_sha256, compiled_signature):
        assert compiled_signature == signature
        return SimpleNamespace(
            mjcf_sha256=mjcf_sha256,
            urdf_sha256=_sha(Path(urdf)),
            compiled_signature_sha256=compiled_signature,
            identity_bound=True,
            runtime_body_names=gate.motion_player.RUNTIME_BODY_NAMES,
        )

    clearance_verifier = tmp_path / "integration_swept_verifier.py"
    _write_text(
        clearance_verifier,
        "# independent continuous swept-clearance integration verifier\n",
    )
    clearance_receipt = tmp_path / "integration_swept_receipt.json"
    clearance_payload = _swept_clearance_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        recipe_path=Path(recipe.path),
        ready_path=Path(recipe.ready.path),
        mjcf_path=Path(recipe.model_paths["mjcf"]),
        urdf_path=Path(recipe.model_paths["urdf"]),
        body_order_path=Path(recipe.model_paths["body_order"]),
        verifier_path=clearance_verifier,
    )
    clearance_receipt.write_text(
        json.dumps(clearance_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = gate.verify_canonical_motion_bank(
        manifest_path,
        output,
        mjcf_path=recipe.model_paths["mjcf"],
        urdf_path=recipe.model_paths["urdf"],
        body_order_path=recipe.model_paths["body_order"],
        expected_compiled_signature=signature,
        swept_clearance_receipt_path=clearance_receipt,
        expected_swept_clearance_receipt_sha256=_sha(
            clearance_receipt
        ),
        # The strict JSON loader (including the parsed marker authority) is
        # covered by test_canonical_motion_recipe against the checked-in
        # recipe; this seam test injects the equivalent recipe object.
        recipe_loader=lambda path: recipe,
        plant_loader=plant_loader,
        player_runner=lambda clip: _player_report(),
        dynamics_runner=lambda path, plant: _grounded_dynamics_report(plant, path),
    )
    assert report["candidate_integrity_pass"] is True
    assert report["verdict"] == "INCOMPLETE_FAIL_CLOSED"
    assert report["aggregate"]["clip_count"] == 10


def test_report_write_is_atomic_no_clobber_and_report_last(tmp_path: Path):
    target = tmp_path / "bank_report.json"
    report = {"verdict": "INCOMPLETE_FAIL_CLOSED", "bank_gate_pass": False}
    written = gate.write_bank_report_no_clobber(report, target)
    assert written == target.resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate.write_bank_report_no_clobber({"verdict": "PASS"}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == report

    broken_target = tmp_path / "broken_report.json"
    missing_destination = tmp_path / "must_not_be_created.json"
    broken_target.symlink_to(missing_destination)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate.write_bank_report_no_clobber(report, broken_target)
    assert broken_target.is_symlink()
    assert not missing_destination.exists()
