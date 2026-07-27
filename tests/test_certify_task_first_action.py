"""Fail-closed contracts for the task-first pre-run diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import certify_task_first_action as cert  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _binding(path: Path, base: Path):
    return {"path": path.relative_to(base).as_posix(), "sha256": _sha(path)}


def _ready(path: Path) -> None:
    np.savez(
        path,
        joint_pos=np.zeros(31, dtype=np.float64),
        joint_vel=np.zeros(31, dtype=np.float64),
        root_pos_w=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        source_segment=np.array("fixture"),
        source_npz=np.array("fixture.npz"),
        source_frame=np.array(0, dtype=np.int64),
        striking_joint_ids=np.arange(7, dtype=np.int64),
        note=np.array("test-only canonical ready"),
    )


def _ready_fk(path: Path, *, ready_sha256: str) -> None:
    body_names = [
        line.strip()
        for line in (REPO / "configs/a3_runtime_body_order.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    body_pos = np.zeros((32, 3), dtype=np.float32)
    body_pos[:, 2] = 1.0
    body_quat = np.zeros((32, 4), dtype=np.float32)
    body_quat[:, 0] = 1.0
    np.savez(
        path,
        canonical_ready_sha256=np.array(ready_sha256),
        body_names=np.asarray(body_names),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        kinematics_contract_version=np.array([1], dtype=np.int64),
    )


def _motion(path: Path, *, middle_scale: float) -> None:
    frames = 6
    joint_pos = np.zeros((frames, 31), dtype=np.float32)
    joint_pos[1:-1, 30] = (
        np.array([0.1, 0.2, 0.15, 0.05], dtype=np.float32) * middle_scale
    )
    joint_vel = np.zeros_like(joint_pos)
    body_pos = np.zeros((frames, 32, 3), dtype=np.float32)
    body_pos[:, :, 2] = 1.0
    body_quat = np.zeros((frames, 32, 4), dtype=np.float32)
    body_quat[:, :, 0] = 1.0
    body_lin = np.zeros((frames, 32, 3), dtype=np.float32)
    body_ang = np.zeros((frames, 32, 3), dtype=np.float32)
    body_names = [
        line.strip()
        for line in (REPO / "configs/a3_runtime_body_order.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    np.savez(
        path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        kinematics_schema_version=np.array([2], dtype=np.int64),
        body_pos_point=np.array("link_origin"),
        body_lin_vel_point=np.array("center_of_mass"),
        body_names=np.array(body_names),
    )


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"numpy-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _playback(motion_sha: str, mjcf_sha: str, *, frames: int = 6, speed: float = 2.0):
    arrays = {
        "site_pos_w": np.array(
            [[0.8 + 0.01 * frame, 0.0, 1.0] for frame in range(frames)],
            dtype="<f8",
        ),
        "site_normal_w": np.tile([1.0, 0.0, 0.0], (frames, 1)).astype("<f8"),
        "site_lin_vel_w": np.tile([speed, 0.0, 0.0], (frames, 1)).astype("<f8"),
        "site_ang_vel_w": np.tile([0.0, 0.0, 4.0], (frames, 1)).astype("<f8"),
    }
    receipts = {
        name: {"sha256": _array_digest(value), "dtype": "<f8", "shape": [frames, 3]}
        for name, value in arrays.items()
    }
    combined = hashlib.sha256()
    combined.update(b"right-racket-trajectory-v1\0")
    for name, value in arrays.items():
        combined.update(name.encode("ascii"))
        combined.update(b"\0")
        combined.update(_array_digest(value).encode("ascii"))
        combined.update(b"\0")
    body_names = [
        line.strip()
        for line in (REPO / "configs/a3_runtime_body_order.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    exact_fk_gate = {
        "pass": True,
        "threshold_m": 1.0e-4,
        "max_error_m": 0.0,
        "worst_frame": 0,
        "worst_body": body_names[0],
    }
    exact_orientation_gate = {
        "pass": True,
        "threshold_rad": 1.0e-4,
        "max_error_rad": 0.0,
        "worst_frame": 0,
        "worst_body": body_names[0],
    }

    def exact_site_gate(threshold: float, suffix: str):
        return {
            "pass": True,
            "threshold": threshold,
            f"max_error_{suffix}": 0.0,
            "worst_frame": 0,
        }

    return {
        "verdict": "PASS",
        "artifacts": {
            "motion_sha256": motion_sha,
            "mjcf_path": "model.xml",
            "mjcf_sha256": mjcf_sha,
            "racket_reference_path": None,
            "racket_reference_sha256": None,
        },
        "contract": {
            "schema": "exact schema-2 11/14 fields",
            "joint_columns": 31,
            "body_columns": 32,
            "joint_order": [f"joint_{index}" for index in range(31)],
            "body_order": body_names,
            "joint_mapping": "name_to_mjcf_qpos_address",
            "body_mapping": "name_to_mjcf_body_id",
            "racket_site": cert.RACKET_SITE_NAME,
            "racket_site_body": cert.RACKET_SITE_BODY,
            "racket_site_local_position_m": list(cert.RACKET_SITE_OFFSET_WRIST_M),
            "racket_normal_convention": "local +Y",
        },
        "evidence_boundary": {
            "level": "kinematic_playback_only",
            "mj_forward_calls": 2 * frames,
            "mj_step_calls": 0,
            "dynamic_certificate": False,
            "training_certificate": False,
            "deployment_certificate": False,
            "hardware_certificate": False,
            "real_robot_certificate": False,
            "racket_velocity_source": cert.RACKET_VELOCITY_SOURCE,
            "statement": "fixture kinematic diagnostics only",
        },
        "authorization": {
            "training": False,
            "deployment": False,
            "hardware": False,
        },
        "motion": {
            "path": "motion.npz",
            "frames": frames,
            "fps": 50.0,
            "duration_s": (frames - 1) / 50.0,
            "migration_provenance": False,
            "body_lin_vel_point": "center_of_mass",
        },
        "gates": {
            "position": exact_fk_gate,
            "orientation": exact_orientation_gate,
            "racket_site_position_vs_schema": exact_site_gate(1.0e-4, "m"),
            "racket_site_normal_vs_schema": exact_site_gate(1.0e-4, "rad"),
            "racket_site_linear_velocity_vs_schema": exact_site_gate(
                1.0e-3, "m_s"
            ),
            "racket_site_angular_velocity_vs_schema": exact_site_gate(
                1.0e-3, "rad_s"
            ),
            "racket_site_jacobian_vs_object_velocity": {
                "pass": True,
                "threshold_max_abs": 1.0e-9,
                "linear_max_abs_error": 0.0,
                "linear_worst_frame": 0,
                "angular_max_abs_error": 0.0,
                "angular_worst_frame": 0,
                "root_twist_max_abs_error": 0.0,
            },
            "table_contact": {
                "enabled": True,
                "pass": True,
                "obstacle_names": list(cert.OBSTACLES),
                "isaac_equivalent_obstacles": list(cert.OBSTACLES),
                "table_pose": {"near_x_m": 0.5, "surface_z_m": 0.76},
                "augmented_mjcf_sha256": "a" * 64,
                "strikes_table": False,
                "contact_frames": [],
                "max_penetration_m": 0.0,
                "worst": None,
                "per_obstacle": {},
            },
            "racket_external_reference": {
                "enabled": False,
                "pass": True,
                "path": None,
                "max_errors": None,
            },
        },
        "racket": {
            "array_receipts": receipts,
            "trajectory_sha256": combined.hexdigest(),
            "peaks": {
                "site_position_norm_max_m": float(
                    np.max(np.linalg.norm(arrays["site_pos_w"], axis=1))
                ),
                "site_normal_norm_error_max": 0.0,
                "site_linear_speed_max_m_s": speed,
                "site_linear_speed_peak_frame": 0,
                "site_angular_speed_max_rad_s": 4.0,
                "site_angular_speed_peak_frame": 0,
            },
            "per_frame": [
                {
                    "frame": frame,
                    "time_s": frame / 50.0,
                    "site_pos_w_m": arrays["site_pos_w"][frame].tolist(),
                    "site_local_plus_y_normal_w": arrays["site_normal_w"][frame].tolist(),
                    "site_lin_vel_w_m_s": arrays["site_lin_vel_w"][frame].tolist(),
                    "site_ang_vel_w_rad_s": arrays["site_ang_vel_w"][frame].tolist(),
                }
                for frame in range(frames)
            ],
        },
        "per_body_max": {
            name: {"position_m": 0.0, "orientation_rad": 0.0}
            for name in body_names
        },
    }


def _collision(
    action: str,
    scope: str,
    shift,
    motion_sha: str,
    mjcf_sha: str,
    urdf_sha: str,
    compiled_signature: str,
):
    component_names = (
        "self_collision",
        "foot_ground_penetration",
        "nonfoot_ground_collision",
        "table_top_collision",
        "net_collision",
        "net_post_collision",
    )
    passing_check = {
        "pass": True,
        "violation_sample_count": 0,
        "violation_contact_count": 0,
        "maximum_penetration_m": 0.0,
        "tolerance_m": 1.0e-5,
    }
    return {
        "schema_version": cert.SCHEMA_VERSION,
        "report_kind": cert.COLLISION_REPORT_KIND,
        "action_id": action,
        "scope": scope,
        "station_center_shift_xy_m": list(shift),
        "verdict": "PASS",
        "artifacts": {
            "motion": {"path": f"{scope}.npz", "sha256": motion_sha},
            "mjcf": {"path": "model.xml", "sha256": mjcf_sha},
            "urdf": {"path": "model.urdf", "sha256": urdf_sha},
            "compiled_model_signature_sha256": compiled_signature,
            "tool": {
                "path": "scripts/certify_task_first_action.py",
                "sha256": _sha(REPO / "scripts/certify_task_first_action.py"),
            },
        },
        "sampling": {
            "source_fps": 50.0,
            "substeps_per_source_interval": 8,
            "sample_hz": 400.0,
            "sample_count": 41,
            "entire_cycle": True,
            "interpolation": (
                "root_xyz_and_joint_linear_plus_shortest_arc_root_quaternion_slerp"
            ),
            "mj_forward_calls": 82,
            "mj_step_calls": 0,
        },
        "model": {
            "robot_collision_geom_count": 37,
            "racket_collision_geoms_included": list(cert.RACKET_COLLISION_GEOMS),
            "obstacle_names": list(cert.OBSTACLES),
            "table_legs_present": False,
        },
        "checks": {
            **{name: dict(passing_check) for name in component_names},
            "aggregate": {"pass": True},
        },
        "clearance": {
            "minimum_table_net_clearance_m": 0.02,
            "distance_query_cap_m": 0.10,
            "minimum": {
                "sample": 0,
                "time_s": 0.0,
                "robot_geom": "right_racket_collision",
                "obstacle": cert.TABLE_TOP,
                "distance_m": 0.02,
            },
        },
        "authorization": {
            "training_authorized": False,
            "deployment_authorized": False,
            "hardware_authorized": False,
        },
        "non_claims": ["fixture"],
    }


def _fixture(tmp_path: Path, monkeypatch):
    action = "fh_loop_high"
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    ready = tmp_path / "ready.npz"
    _ready(ready)
    ready_fk = tmp_path / "ready_fk.npz"
    _ready_fk(ready_fk, ready_sha256=_sha(ready))
    marker = tmp_path / "marker.json"
    _write_json(marker, {"fixture": True})
    mjcf = tmp_path / "model.xml"
    mjcf.write_bytes(b"<mujoco/>")
    urdf = tmp_path / "model.urdf"
    urdf.write_bytes(b"<robot/>")
    venue = tmp_path / "venue.yaml"
    venue.write_bytes((REPO / cert.CODE_ROOTED_BALL_PHYSICS).read_bytes())
    venue_profile = tmp_path / "venue_profile.json"
    venue_profile.write_bytes((REPO / cert.CODE_ROOTED_VENUE_PROFILE).read_bytes())
    compiled_signature = "f" * 64
    recipe = tmp_path / "recipe.json"
    _write_json(
        recipe,
        {
            "canonical_ready": {"path": ready.name, "sha256": _sha(ready)},
            "canonical_ready_fk": {
                "path": ready_fk.name,
                "sha256": _sha(ready_fk),
            },
            "marker_authority": {"path": marker.name, "sha256": _sha(marker)},
            "motion_specs": [
                {
                    "motion_id": action,
                    "source_path": source.name,
                    "source_sha256": _sha(source),
                }
            ],
            "model_contract": {
                "mjcf_path": mjcf.name,
                "mjcf_sha256": _sha(mjcf),
                "urdf_path": urdf.name,
                "urdf_sha256": _sha(urdf),
            },
        },
    )
    marker_row = SimpleNamespace(
        bound_recipe_source_sha256=_sha(source),
        post_retime_behavior_gate_status="PENDING_POST_RETIME_BEHAVIOR_RESCAN",
        contact_anchor=lambda: (2, "fixture"),
        search_window=lambda: ((1, 3), "fixture"),
    )
    monkeypatch.setattr(cert, "_load_marker_row", lambda *_: marker_row)

    motions = {}
    playbacks = {}
    for index, scope in enumerate(cert.SCOPES):
        motion = tmp_path / f"{scope}.npz"
        _motion(motion, middle_scale=1.0 + index)
        motions[scope] = motion
        playback = tmp_path / f"{scope}.playback.json"
        _write_json(playback, _playback(_sha(motion), _sha(mjcf)))
        playbacks[scope] = playback

    manifest = tmp_path / "BUILD_MANIFEST.json"
    outputs = []
    for scope in cert.SCOPES:
        outputs.append(
            {
                "motion_id": action,
                "scope": scope,
                "scope_preprocessing": {"algorithm": "fixture-scope-v1"},
                "filename": motions[scope].name,
                "output_npz_sha256": _sha(motions[scope]),
                "source_anchor_time_s": 0.04,
                "duration_s": 0.10,
                "search": {
                    "contact_opportunity": {
                        "source_anchor_frame": 2,
                        "source_span_inclusive": [1, 3],
                        "marker_only": True,
                        "pose_locked": False,
                        "velocity_locked": False,
                    }
                },
                "retiming": {
                    "markers": {
                        "source_anchor": {
                            "time_s": 0.04,
                            "output_fractional_frame": 2.0,
                            "output_frame": 2,
                        }
                    }
                },
            }
        )
    _write_json(manifest, {"recipe": {"sha256": _sha(recipe)}, "outputs": outputs})

    bank = tmp_path / "bank_report.json"
    _write_json(
        bank,
        {
            "schema_version": 1,
            "verdict": "INCOMPLETE_FAIL_CLOSED",
            "bank_gate_pass": False,
            "candidate_integrity_pass": True,
            "grounded_trace_status": "MISSING_INCOMPLETE_FAIL_CLOSED",
            "publication_class": "post_build_diagnostic_only",
            "training_authorized": False,
            "hardware_authorized": False,
            "library_id": "fixture_bank",
            "manifest": {"path": manifest.name, "sha256": _sha(manifest)},
            "bank_dir": str(tmp_path),
            "bound_inputs": {
                "recipe": {"path": recipe.name, "sha256": _sha(recipe)},
                "compiler": {
                    "path": "hope_training/whole_body_tracking/scripts/canonical_motion_compiler.py",
                    "sha256": _sha(
                        REPO
                        / "hope_training/whole_body_tracking/scripts/"
                        "canonical_motion_compiler.py"
                    ),
                },
                "geometry_tool": {
                    "path": "hope_training/whole_body_tracking/scripts/canonical_motion_geometry.py",
                    "sha256": _sha(
                        REPO
                        / "hope_training/whole_body_tracking/scripts/"
                        "canonical_motion_geometry.py"
                    ),
                },
                "compiler_options_sha256": "e" * 64,
                "ready": {"path": ready.name, "sha256": _sha(ready)},
                "mjcf": {"path": mjcf.name, "sha256": _sha(mjcf)},
                "urdf": {"path": urdf.name, "sha256": _sha(urdf)},
                "body_order": {
                    "path": "configs/a3_runtime_body_order.txt",
                    "sha256": _sha(REPO / "configs/a3_runtime_body_order.txt"),
                },
                "plant": {
                    "mjcf_sha256": _sha(mjcf),
                    "urdf_sha256": _sha(urdf),
                    "compiled_signature_sha256": compiled_signature,
                    "identity_bound": True,
                    "runtime_body_order": [
                        line.strip()
                        for line in (
                            REPO / "configs/a3_runtime_body_order.txt"
                        )
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    ],
                },
                "verifier_tools": {
                    "bank_gate": {
                        "path": "hope_training/whole_body_tracking/scripts/canonical_motion_bank_gate.py",
                        "sha256": _sha(
                            REPO
                            / "hope_training/whole_body_tracking/scripts/"
                            "canonical_motion_bank_gate.py"
                        )
                    },
                    "mujoco_motion_player": {
                        "path": "hope_training/whole_body_tracking/scripts/mujoco_motion_player.py",
                        "sha256": _sha(
                            REPO
                            / "hope_training/whole_body_tracking/scripts/"
                            "mujoco_motion_player.py"
                        )
                    },
                    "canonical_mujoco_dynamics_gate": {
                        "path": "hope_training/whole_body_tracking/scripts/canonical_mujoco_dynamics_gate.py",
                        "sha256": _sha(
                            REPO
                            / "hope_training/whole_body_tracking/scripts/"
                            "canonical_mujoco_dynamics_gate.py"
                        ),
                        "report_schema_version": 1,
                    },
                },
            },
            "contracts": {
                "matrix": {
                    "motion_ids": [action],
                    "scopes": list(cert.SCOPES),
                    "count": 2,
                },
                "shared_ready": True,
                "six_endpoint_velocity_classes_exact_zero": True,
                "contact_opportunity_is_marker_only": True,
                "acceleration_allowed_through_window_end": True,
                "nonnegative_scalar_acceleration_through_window_end": True,
                "adv2c3_role": "comparator_only_not_default",
                "grounded_inverse_dynamics": (
                    "incomplete_fail_closed_until_content_addressed_trace"
                ),
                "grounded_trace_status": "MISSING_INCOMPLETE_FAIL_CLOSED",
            },
            "aggregate": {
                "clip_count": 2,
                "fk_pass_count": 2,
                "velocity_consistency_pass_count": 2,
                "joint_limit_pass_count": 2,
                "geometry_pass_count": 2,
                "non_torque_dynamics_pass_count": 2,
                "complete_dynamics_pass_count": 0,
                "incomplete_fail_closed_count": 2,
                "failed_count": 0,
                "torque_interpretation_valid_count": 0,
                "clips_with_contact_count": 0,
                "contact_frame_count": 0,
                "self_collision_violation_count": 0,
                "foot_floor_penetration_violation_count": 0,
                "nonfoot_floor_penetration_violation_count": 0,
                "other_world_penetration_violation_count": 0,
                "joint_effort_proxy_peak_utilization": 0.0,
                "actuator_force_proxy_peak_utilization": 0.0,
                "root_height_min_m": 1.0,
                "root_height_max_m": 1.0,
                "root_tilt_peak_rad": 0.0,
                "root_xy_displacement_peak_m": 0.0,
                "com_height_min_m": 1.0,
                "com_height_max_m": 1.0,
            },
            "clips": [
                {
                    "motion_id": action,
                    "scope": scope,
                    "filename": motions[scope].name,
                    "sha256": _sha(motions[scope]),
                    "frames": 6,
                    "fps": 50.0,
                    "duration_s": 0.10,
                    "schema2_receipts": {},
                    "strict_schema2_and_ready": True,
                    "contact_opportunity": {},
                    "mujoco_fk": {"pass": True},
                    "plant_specific_dynamics": {"screen_pass": False},
                }
                for scope in cert.SCOPES
            ],
            "non_claims": ["fixture diagnostic only"],
        },
    )

    collision_paths = {}
    for scope in cert.SCOPES:
        collision_paths[scope] = []
        for index, shift in enumerate(
            cert.STATION_CENTER_SHIFT_CANDIDATES_XY_M
        ):
            path = tmp_path / f"{scope}.shift{index}.json"
            _write_json(
                path,
                _collision(
                    action,
                    scope,
                    shift,
                    _sha(motions[scope]),
                    _sha(mjcf),
                    _sha(urdf),
                    compiled_signature,
                ),
            )
            collision_paths[scope].append(path)

    plan = {
        "schema_version": cert.SCHEMA_VERSION,
        "plan_kind": cert.PLAN_KIND,
        "action_id": action,
        "bindings": {
            "source": _binding(source, tmp_path),
            "recipe": _binding(recipe, tmp_path),
            "build_manifest": _binding(manifest, tmp_path),
            "canonical_verifier_report": _binding(bank, tmp_path),
            "mjcf": _binding(mjcf, tmp_path),
            "venue_yaml": _binding(venue, tmp_path),
            "venue_profile": _binding(venue_profile, tmp_path),
        },
        "required_scopes": list(cert.SCOPES),
        "station_center_shift_candidates_xy_m": [
            list(row) for row in cert.STATION_CENTER_SHIFT_CANDIDATES_XY_M
        ],
        "selected_station_center_shift_xy_m": None,
        "station_selection_approval": None,
        "behavior_contact_evidence": None,
        "thresholds": {
            "source_anchor_time_min_s": 0.03,
            "source_anchor_time_max_s": 0.06,
            "t_hit_reference_s": 0.5,
            "t_cycle_min_s": 0.08,
            "t_cycle_max_s": 0.20,
            "blade_site_speed_min_m_s": 1.0,
            "blade_site_speed_max_m_s": 3.0,
            "shared_ready_pose_tolerance": 1.0e-6,
            "dense_collision_min_hz": 400.0,
            "minimum_table_net_clearance_m": 0.005,
        },
        "task_distribution": {
            "incoming_velocity_box_m_s": [
                [-4.0, -3.0],
                [-0.1, 0.1],
                [-0.2, 0.2],
            ],
            "spin_abs_max_rad_s": 5.0,
            "samples": 256,
            "seed": 7,
            "face_sign": 1.0,
            "capture_radius_m": 0.095,
            "minimum_approach_speed_m_s": 0.3,
            "minimum_legal_return_fraction": 0.5,
        },
        "scopes": {
            scope: {
                "motion": _binding(motions[scope], tmp_path),
                "playback_report": _binding(playbacks[scope], tmp_path),
                "collision_reports": [
                    _binding(path, tmp_path) for path in collision_paths[scope]
                ],
            }
            for scope in cert.SCOPES
        },
        "authorization_intent": "task_first_training_only_no_deployment_no_hardware",
    }
    return {
        "plan": plan,
        "source": source,
        "bank": bank,
        "playbacks": playbacks,
        "collisions": collision_paths,
        "motions": motions,
    }


def _evaluate(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    return fixture, cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def _fail_collision(fixture, tmp_path, *, scope: str, station_index: int) -> None:
    path = fixture["collisions"][scope][station_index]
    collision = json.loads(path.read_text(encoding="utf-8"))
    collision["verdict"] = "FAIL"
    collision["checks"]["table_top_collision"]["pass"] = False
    collision["checks"]["table_top_collision"]["violation_sample_count"] = 1
    collision["checks"]["table_top_collision"]["violation_contact_count"] = 1
    collision["checks"]["table_top_collision"]["maximum_penetration_m"] = 0.01
    collision["checks"]["aggregate"]["pass"] = False
    collision["clearance"]["minimum_table_net_clearance_m"] = -0.01
    collision["clearance"]["minimum"]["distance_m"] = -0.01
    _write_json(path, collision)
    fixture["plan"]["scopes"][scope]["collision_reports"][station_index] = _binding(
        path, tmp_path
    )


def _approve_station_selection(
    fixture,
    tmp_path,
    monkeypatch,
    selected,
) -> Path:
    fixture["plan"]["selected_station_center_shift_xy_m"] = list(selected)
    fixture["plan"]["station_selection_approval"] = None
    comparison_sha = cert._station_comparison_input_sha256(
        fixture["plan"]
    )
    approval = {
        "schema_version": 1,
        "kind": cert.STATION_SELECTION_APPROVAL_KIND,
        "action_id": fixture["plan"]["action_id"],
        "selected_station_center_shift_xy_m": list(selected),
        "comparison_input_sha256": comparison_sha,
        "approval_policy": (
            "independent_code_reviewed_station_selection_v1"
        ),
        "non_claims": [
            "station approval does not authorize training or hardware"
        ],
    }
    path = tmp_path / "station-selection-approval.json"
    _write_json(path, approval)
    fixture["plan"]["station_selection_approval"] = _binding(
        path, tmp_path
    )
    monkeypatch.setattr(
        cert,
        "TRUSTED_STATION_SELECTION_APPROVAL_SHA256",
        frozenset({_sha(path)}),
    )
    return path


def _install_behavior_contact_evidence(
    fixture,
    tmp_path,
    monkeypatch,
    *,
    accepted=(0.03, 0.06),
    t_hit=0.04,
) -> Path:
    artifact = tmp_path / "behavior-contact-evidence.bin"
    artifact.write_bytes(b"reviewed behavior/contact evidence\n")
    receipt = {
        "schema_version": 1,
        "kind": cert.BEHAVIOR_CONTACT_EVIDENCE_KIND,
        "authority_contract_sha256": (
            cert.BEHAVIOR_CONTACT_AUTHORITY_CONTRACT_SHA256
        ),
        "action_id": fixture["plan"]["action_id"],
        "accepted_t_hit_range_s": list(accepted),
        "measurements": [
            {
                "scope": scope,
                "motion_sha256": fixture["plan"]["scopes"][scope][
                    "motion"
                ]["sha256"],
                "t_hit_s": t_hit,
            }
            for scope in cert.SCOPES
        ],
        "evidence_artifact": _binding(artifact, tmp_path),
        "non_claims": [
            "behavior/contact evidence does not authorize training"
        ],
    }
    path = tmp_path / "behavior-contact-receipt.json"
    _write_json(path, receipt)
    fixture["plan"]["behavior_contact_evidence"] = _binding(path, tmp_path)
    monkeypatch.setattr(
        cert,
        "TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256",
        frozenset({_sha(path)}),
    )
    return path


def test_template_is_incomplete_and_uses_whole_station_center_translation():
    plan = cert.template_plan("fh_loop_high", "vendor_assets/source.npz", "a" * 64)
    assert plan["station_center_shift_candidates_xy_m"] == [
        [0.0, 0.0],
        [-0.05, 0.0],
        [-0.10, 0.0],
    ]
    assert plan["selected_station_center_shift_xy_m"] is None
    assert plan["thresholds"]["source_anchor_time_min_s"] is None
    assert plan["thresholds"]["source_anchor_time_max_s"] is None


def test_passing_external_reference_checks_never_authorize_smoke_or_training(
    tmp_path, monkeypatch
):
    _fixture_value, report = _evaluate(tmp_path, monkeypatch)
    assert report["diagnostic_reference_checks_pass"] is False
    assert report["diagnostic_smoke_authorized"] is False
    assert report["training_authorized"] is False
    assert report["deployment_authorized"] is False
    assert report["hardware_authorized"] is False
    assert report["verdict"] == "BLOCKED"
    assert report["publication_class"] == "diagnostic_only_blocked"
    assert report["admission_capability_minted"] is False
    assert "canonical_motion_admission" in report["admission_authority"]
    assert report["bindings"]["canonical_ready"]["sha256"] == _sha(
        tmp_path / "ready.npz"
    )
    assert report["bindings"]["canonical_ready_fk"]["sha256"] == _sha(
        tmp_path / "ready_fk.npz"
    )
    assert any(
        "external_playback" in row
        for row in report["diagnostic_smoke_blockers"]
    )
    assert any("grounded_collocation_trace_missing" in row for row in report["training_blockers"])
    for scope in cert.SCOPES:
        timing = report["scopes"][scope]["timing"]
        assert timing["post_retime_behavior_t_hit_measured"] is False
        assert timing["comparison_t_hit_reference_s"] == pytest.approx(0.5)
        assert timing["t_hit_gate_result"] is False
        assert timing["t_hit_acceptance_authority"] == (
            "code_pinned_action_specific_behavior_contact_evidence"
        )
        assert timing["compiler_anchor_substituted_for_t_hit"] is False


def test_cli_never_returns_success_for_untrusted_reference_receipts(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, fixture["plan"])
    output_path = tmp_path / "diagnostic.json"
    rc = cert.main(
        [
            "certify",
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            _sha(plan_path),
            "--out",
            str(output_path),
        ]
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert report["plan"]["sha256"] == _sha(plan_path)
    assert report["diagnostic_smoke_authorized"] is False
    assert report["training_authorized"] is False
    assert report["deployment_authorized"] is False
    assert report["hardware_authorized"] is False


def test_scan_exception_receipt_closes_every_authorization_field(
    tmp_path, monkeypatch
):
    def fail_scan(**_kwargs):
        raise cert.CertificationError("fixture scan failure")

    monkeypatch.setattr(cert, "scan_collisions", fail_scan)
    output_path = tmp_path / "collision-failure.json"
    rc = cert.main(
        [
            "scan-collisions",
            "--action-id",
            "fh_loop_high",
            "--scope",
            "upper",
            "--station-center-shift-xy-m",
            "0",
            "0",
            "--motion",
            str(tmp_path / "motion.npz"),
            "--expected-motion-sha256",
            "a" * 64,
            "--mjcf",
            str(tmp_path / "model.xml"),
            "--expected-mjcf-sha256",
            "b" * 64,
            "--urdf",
            str(tmp_path / "model.urdf"),
            "--expected-urdf-sha256",
            "c" * 64,
            "--expected-compiled-signature",
            "d" * 64,
            "--out",
            str(output_path),
        ]
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert report["diagnostic_smoke_authorized"] is False
    assert report["training_authorized"] is False
    assert report["deployment_authorized"] is False
    assert report["hardware_authorized"] is False


def test_handwritten_bank_pass_is_rejected_not_promoted(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    bank = json.loads(fixture["bank"].read_text(encoding="utf-8"))
    bank["verdict"] = "PASS"
    bank["bank_gate_pass"] = True
    _write_json(fixture["bank"], bank)
    fixture["plan"]["bindings"]["canonical_verifier_report"] = _binding(
        fixture["bank"], tmp_path
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="verdict/bank_gate_pass"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_forged_bank_training_authorization_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    bank = json.loads(fixture["bank"].read_text(encoding="utf-8"))
    bank["training_authorized"] = True
    _write_json(fixture["bank"], bank)
    fixture["plan"]["bindings"]["canonical_verifier_report"] = _binding(
        fixture["bank"], tmp_path
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="top-level contract"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_plain_screen_pass_never_substitutes_for_grounded_trace(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    bank = json.loads(fixture["bank"].read_text(encoding="utf-8"))
    for clip in bank["clips"]:
        clip["plant_specific_dynamics"]["screen_pass"] = True
    _write_json(fixture["bank"], bank)
    fixture["plan"]["bindings"]["canonical_verifier_report"] = _binding(
        fixture["bank"], tmp_path
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["training_authorized"] is False
    assert report["admission_capability_minted"] is False
    for scope in cert.SCOPES:
        selected = report["scopes"][scope]["stations"]["0.00,0.00"]
        assert selected["gates"]["grounded_dynamics"] is False


@pytest.mark.parametrize("target", ["bank", "playback", "collision"])
def test_external_json_reports_reject_unknown_schema_fields(
    tmp_path, monkeypatch, target
):
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "bank":
        path = fixture["bank"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["forged_extra"] = True
        _write_json(path, payload)
        fixture["plan"]["bindings"]["canonical_verifier_report"] = _binding(
            path, tmp_path
        )
    elif target == "playback":
        path = fixture["playbacks"]["upper"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["forged_extra"] = True
        _write_json(path, payload)
        fixture["plan"]["scopes"]["upper"]["playback_report"] = _binding(
            path, tmp_path
        )
    else:
        path = fixture["collisions"]["upper"][0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["forged_extra"] = True
        _write_json(path, payload)
        fixture["plan"]["scopes"]["upper"]["collision_reports"][0] = _binding(
            path, tmp_path
        )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="keys changed"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_playback_cannot_claim_pass_with_oversized_tolerance(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["playbacks"]["upper"]
    playback = json.loads(path.read_text(encoding="utf-8"))
    playback["gates"]["position"]["threshold_m"] = 1.0e9
    _write_json(path, playback)
    fixture["plan"]["scopes"]["upper"]["playback_report"] = _binding(
        path, tmp_path
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="oversized tolerance"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


@pytest.mark.parametrize("target", ["bank_playback_tool", "collision_tool"])
def test_external_report_tool_digest_must_match_code_root(
    tmp_path, monkeypatch, target
):
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "bank_playback_tool":
        path = fixture["bank"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["bound_inputs"]["verifier_tools"]["mujoco_motion_player"][
            "sha256"
        ] = "0" * 64
        _write_json(path, payload)
        fixture["plan"]["bindings"]["canonical_verifier_report"] = _binding(
            path, tmp_path
        )
    else:
        path = fixture["collisions"]["upper"][0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"]["tool"]["sha256"] = "0" * 64
        _write_json(path, payload)
        fixture["plan"]["scopes"]["upper"]["collision_reports"][0] = _binding(
            path, tmp_path
        )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="exact"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_nonfinite_json_constant_is_rejected_before_gate_claim(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    raw = fixture["bank"].read_text(encoding="utf-8")
    raw = raw.replace(
        '"joint_effort_proxy_peak_utilization": 0.0',
        '"joint_effort_proxy_peak_utilization": NaN',
    )
    fixture["bank"].write_text(raw, encoding="utf-8")
    fixture["plan"]["bindings"]["canonical_verifier_report"] = _binding(
        fixture["bank"], tmp_path
    )
    with pytest.raises(cert.CertificationError, match="non-finite JSON"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_comparison_never_auto_adopts_station_center(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["selected_station_center_shift_xy_m"] = None
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["selected_station_center_shift_xy_m"] is None
    assert "not_selected" in report["diagnostic_reference_blockers"][0]


def test_selected_station_requires_independent_code_pinned_approval(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["selected_station_center_shift_xy_m"] = [0.0, 0.0]
    with pytest.raises(
        cert.CertificationError, match="independent.*approval"
    ):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)

    approval_path = _approve_station_selection(
        fixture, tmp_path, monkeypatch, (0.0, 0.0)
    )
    fixture["plan"]["thresholds"]["source_anchor_time_min_s"] = 0.031
    with pytest.raises(
        cert.CertificationError, match="comparison binding drifted"
    ):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert approval_path.is_file()


def test_behavior_t_hit_requires_code_pin_and_action_specific_motion_binding(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    receipt_path = _install_behavior_contact_evidence(
        fixture, tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        cert,
        "TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256",
        frozenset(),
    )
    with pytest.raises(cert.CertificationError, match="trust set is empty"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)

    monkeypatch.setattr(
        cert,
        "TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256",
        frozenset({_sha(receipt_path)}),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["measurements"][0]["motion_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    fixture["plan"]["behavior_contact_evidence"] = _binding(
        receipt_path, tmp_path
    )
    monkeypatch.setattr(
        cert,
        "TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256",
        frozenset({_sha(receipt_path)}),
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(
        cert.CertificationError, match="different motion bytes"
    ):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_trusted_action_specific_behavior_t_hit_opens_only_that_gate(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    _install_behavior_contact_evidence(
        fixture,
        tmp_path,
        monkeypatch,
        accepted=(0.03, 0.06),
        t_hit=0.04,
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["training_authorized"] is False
    assert report["diagnostic_smoke_authorized"] is False
    assert report["behavior_contact_authority"] is not None
    assert not any(
        "post_retime_behavior_t_hit_rescan_pending" in blocker
        for blocker in report["diagnostic_smoke_blockers"]
    )
    for scope in cert.SCOPES:
        timing = report["scopes"][scope]["timing"]
        assert timing["post_retime_behavior_t_hit_measured"] is True
        assert timing["post_retime_behavior_t_hit_s"] == pytest.approx(0.04)
        assert timing["accepted_action_specific_t_hit_range_s"] == [
            0.03,
            0.06,
        ]
        assert timing["t_hit_gate_result"] is True
        assert (
            report["scopes"][scope]["stations"]["0.00,0.00"]["gates"][
                "post_retime_t_hit"
            ]
            is True
        )


def test_plan_cannot_rewrite_code_pinned_behavior_t_hit_range(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    receipt_path = _install_behavior_contact_evidence(
        fixture, tmp_path, monkeypatch
    )
    pinned_sha = _sha(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["accepted_t_hit_range_s"] = [0.0, 1000.0]
    _write_json(receipt_path, receipt)
    fixture["plan"]["behavior_contact_evidence"] = _binding(
        receipt_path, tmp_path
    )
    monkeypatch.setattr(
        cert,
        "TRUSTED_BEHAVIOR_CONTACT_EVIDENCE_SHA256",
        frozenset({pinned_sha}),
    )
    with pytest.raises(cert.CertificationError, match="not code-pinned"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


@pytest.mark.parametrize("selected_index", [1, 2])
def test_farther_station_requires_every_nearer_station_to_fail_upper_or_full(
    tmp_path, monkeypatch, selected_index
):
    fixture = _fixture(tmp_path, monkeypatch)
    for nearer_index in range(selected_index):
        _fail_collision(
            fixture,
            tmp_path,
            scope="upper",
            station_index=nearer_index,
        )
    selected = cert.STATION_CENTER_SHIFT_CANDIDATES_XY_M[selected_index]
    _approve_station_selection(
        fixture, tmp_path, monkeypatch, selected
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["selected_station_center_shift_xy_m"] == list(selected)
    assert report["diagnostic_reference_checks_pass"] is False
    assert "upper/post_retime_t_hit" in report["diagnostic_reference_blockers"]
    assert report["diagnostic_smoke_authorized"] is False


@pytest.mark.parametrize("selected_x", [-0.05, -0.10])
def test_farther_station_is_rejected_when_a_nearer_station_common_passes(
    tmp_path, monkeypatch, selected_x
):
    fixture = _fixture(tmp_path, monkeypatch)
    _approve_station_selection(
        fixture, tmp_path, monkeypatch, (selected_x, 0.0)
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="nearest upper/full"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_failed_selected_station_center_blocks_reference_checks(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    for station_index in range(len(cert.STATION_CENTER_SHIFT_CANDIDATES_XY_M)):
        _fail_collision(
            fixture,
            tmp_path,
            scope="upper",
            station_index=station_index,
        )
    _approve_station_selection(
        fixture, tmp_path, monkeypatch, (0.0, 0.0)
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["diagnostic_reference_checks_pass"] is False
    assert "upper/dense_collision" in report["diagnostic_reference_blockers"]


def test_low_anchor_blade_site_speed_blocks_reference_checks(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["playbacks"]["full"]
    _write_json(
        path,
        _playback(
            fixture["plan"]["scopes"]["full"]["motion"]["sha256"],
            fixture["plan"]["bindings"]["mjcf"]["sha256"],
            speed=0.5,
        ),
    )
    fixture["plan"]["scopes"]["full"]["playback_report"] = _binding(path, tmp_path)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["diagnostic_reference_checks_pass"] is False
    assert (
        report["scopes"]["full"]["stations"]["0.00,0.00"]["gates"][
            "physical_blade_site_speed"
        ]
        is False
    )
    assert report["training_authorized"] is False


def test_low_reference_return_fraction_blocks_reference_checks(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.0)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["diagnostic_reference_checks_pass"] is False
    for scope in cert.SCOPES:
        assert (
            report["scopes"][scope]["stations"]["0.00,0.00"]["gates"][
                "reference_returnability"
            ]
            is False
        )
    assert report["diagnostic_smoke_authorized"] is False
    assert report["training_authorized"] is False


@pytest.mark.parametrize("value", [float("nan"), -0.001, 1.001])
def test_reference_scorer_fraction_must_be_finite_probability(
    tmp_path, monkeypatch, value
):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: value)
    with pytest.raises(cert.CertificationError, match="reference return fraction"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_anchor_time_min_s", 0.0),
        ("source_anchor_time_max_s", 4.0),
        ("source_anchor_time_max_s", 0.50),
        ("t_hit_reference_s", 0.500001),
        ("t_hit_reference_s", 1.0e9),
        ("t_cycle_min_s", 0.0),
        ("t_cycle_max_s", 6.0),
        ("t_cycle_max_s", 2.0),
        ("blade_site_speed_min_m_s", 0.0),
        ("blade_site_speed_max_m_s", 21.0),
        ("blade_site_speed_max_m_s", 12.0),
        ("shared_ready_pose_tolerance", 0.01),
        ("dense_collision_min_hz", 399.999),
        ("minimum_table_net_clearance_m", 0.004999),
    ],
)
def test_plan_cannot_relax_code_reviewed_certification_thresholds(
    tmp_path, monkeypatch, field, value
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["thresholds"][field] = value
    with pytest.raises(cert.CertificationError, match="code-reviewed"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 255),
        ("capture_radius_m", 0.095001),
        ("minimum_approach_speed_m_s", 0.299999),
        ("minimum_legal_return_fraction", 0.499999),
        ("minimum_legal_return_fraction", 0.0),
    ],
)
def test_plan_cannot_relax_code_reviewed_returnability_thresholds(
    tmp_path, monkeypatch, field, value
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["task_distribution"][field] = value
    with pytest.raises(cert.CertificationError, match="physical domains"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_one_sample_collision_claim_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["collisions"]["upper"][0]
    collision = json.loads(path.read_text(encoding="utf-8"))
    collision["sampling"]["sample_count"] = 1
    collision["sampling"]["mj_forward_calls"] = 2
    _write_json(path, collision)
    fixture["plan"]["scopes"]["upper"]["collision_reports"][0] = _binding(
        path, tmp_path
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="entire cycle"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_playback_array_receipt_must_match_physical_site_rows(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["playbacks"]["upper"]
    playback = json.loads(path.read_text(encoding="utf-8"))
    playback["racket"]["per_frame"][2]["site_lin_vel_w_m_s"] = [99.0, 0.0, 0.0]
    _write_json(path, playback)
    fixture["plan"]["scopes"]["upper"]["playback_report"] = _binding(path, tmp_path)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="trajectory receipt"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_common_wrong_ready_pose_is_rejected_against_canonical_truth(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    path = fixture["motions"]["upper"]
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    payload["joint_pos"][[0, -1], 0] = 0.1
    ready_path = tmp_path / "ready.npz"
    ready = cert._canonical_ready_state(
        cert.Snapshot(
            path=ready_path,
            data=ready_path.read_bytes(),
            sha256=_sha(ready_path),
        )
    )
    ready_fk_path = tmp_path / "ready_fk.npz"
    ready_fk = cert._canonical_ready_fk_state(
        cert.Snapshot(
            path=ready_fk_path,
            data=ready_fk_path.read_bytes(),
            sha256=_sha(ready_fk_path),
        ),
        canonical_ready_sha256=_sha(ready_path),
        canonical_ready=ready,
    )
    gate = cert._motion_ready_truth_gate(
        payload, ready, ready_fk, tolerance=1.0e-6
    )
    assert gate["pass"] is False
    assert gate["joint_position_max_abs_error_rad"] == pytest.approx(0.1)


def test_ready_fk_must_bind_exact_canonical_ready_digest(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    ready_path = tmp_path / "ready.npz"
    ready_fk_path = tmp_path / "ready_fk.npz"
    ready = cert._canonical_ready_state(
        cert.Snapshot(
            path=ready_path,
            data=ready_path.read_bytes(),
            sha256=_sha(ready_path),
        )
    )
    with np.load(ready_fk_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    payload["canonical_ready_sha256"] = np.array("0" * 64)
    tampered = tmp_path / "wrong-ready-fk.npz"
    np.savez(tampered, **payload)
    with pytest.raises(cert.CertificationError, match="exact canonical-ready digest"):
        cert._canonical_ready_fk_state(
            cert.Snapshot(
                path=tampered,
                data=tampered.read_bytes(),
                sha256=_sha(tampered),
            ),
            canonical_ready_sha256=_sha(ready_path),
            canonical_ready=ready,
        )


def test_nonroot_ready_fk_error_blocks_shared_ready_gate(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    motion_path = fixture["motions"]["upper"]
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: np.asarray(archive[key]).copy() for key in archive.files}
    ready_path = tmp_path / "ready.npz"
    ready_fk_path = tmp_path / "ready_fk.npz"
    ready = cert._canonical_ready_state(
        cert.Snapshot(
            path=ready_path,
            data=ready_path.read_bytes(),
            sha256=_sha(ready_path),
        )
    )
    ready_fk = cert._canonical_ready_fk_state(
        cert.Snapshot(
            path=ready_fk_path,
            data=ready_fk_path.read_bytes(),
            sha256=_sha(ready_fk_path),
        ),
        canonical_ready_sha256=_sha(ready_path),
        canonical_ready=ready,
    )
    motion["body_pos_w"][[0, -1], 7, 1] += 0.01
    gate = cert._motion_ready_truth_gate(
        motion, ready, ready_fk, tolerance=1.0e-6
    )
    assert gate["pass"] is False
    assert gate["body_position_max_error_m"] == pytest.approx(0.01)


def test_canonical_ready_truth_is_composed_into_reference_gate(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    monkeypatch.setattr(
        cert,
        "_motion_ready_truth_gate",
        lambda *_args, **_kwargs: {
            "pass": False,
            "joint_position_max_abs_error_rad": 0.1,
            "root_position_max_error_m": 0.0,
            "root_orientation_max_error_rad": 0.0,
            "tolerance": 1.0e-6,
            "truth_source": "content_bound_canonical_ready_npz",
        },
    )
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["diagnostic_reference_checks_pass"] is False
    for scope in cert.SCOPES:
        assert (
            report["scopes"][scope]["stations"]["0.00,0.00"]["gates"][
                "shared_ready_return"
            ]
            is False
        )


def test_reference_scorer_receives_exact_venue_and_station_center_xy(
    tmp_path, monkeypatch
):
    captured = {}

    def score_reference_returns(**kwargs):
        captured.update(kwargs)
        return 0.625

    monkeypatch.setitem(
        sys.modules,
        "reference_return_gate",
        SimpleNamespace(score_reference_returns=score_reference_returns),
    )
    venue = tmp_path / "venue.yaml"
    venue.write_text("fixture: true\n", encoding="utf-8")
    result = cert._reference_return_fraction(
        state={
            "position_w_m": np.array([1.0, 2.0, 3.0]),
            "velocity_w_m_s": np.array([2.0, 0.0, 0.0]),
            "normal_w": np.array([1.0, 0.0, 0.0]),
        },
        station_center_shift_xy_m=(-0.05, 0.0),
        task={
            "incoming_velocity_box_m_s": [[-4.0, -3.0], [-0.1, 0.1], [-0.2, 0.2]],
            "spin_abs_max_rad_s": 0.0,
            "samples": 256,
            "seed": 7,
            "face_sign": 1.0,
            "venue_yaml_path": str(venue),
            "capture_radius_m": 0.095,
            "minimum_approach_speed_m_s": 0.3,
        },
    )
    assert result == pytest.approx(0.625)
    assert captured["p_contact_w"].tolist() == pytest.approx([0.95, 2.0, 3.0])
    assert captured["venue_yaml"] == str(venue)


def test_duplicate_station_report_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    reports = fixture["plan"]["scopes"]["upper"]["collision_reports"]
    reports[2] = dict(reports[0])
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="duplicates collision shift"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_same_motion_bytes_cannot_masquerade_as_upper_and_full(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["scopes"]["full"]["motion"] = dict(
        fixture["plan"]["scopes"]["upper"]["motion"]
    )
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="must be distinct"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_content_drift_is_rejected_before_any_gate_claim(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["source"].write_bytes(b"changed")
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="SHA-256 mismatch"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "incoming_velocity_box_m_s",
            [[-3.0, -3.0], [-0.1, 0.1], [-0.2, 0.2]],
            "zero/single-point",
        ),
        (
            "incoming_velocity_box_m_s",
            [[-0.5, -0.4], [-0.1, 0.1], [-0.2, 0.2]],
            "physical domains",
        ),
        (
            "incoming_velocity_box_m_s",
            [[-8.0, -7.5], [-0.1, 0.1], [-0.2, 0.2]],
            "physical domains",
        ),
        ("spin_abs_max_rad_s", 0.0, "physical domains"),
        ("spin_abs_max_rad_s", 100.0, "physical domains"),
    ],
)
def test_returnability_domain_cannot_bypass_venue_envelope(
    tmp_path, monkeypatch, field, value, match
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["task_distribution"][field] = value
    with pytest.raises(cert.CertificationError, match=match):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


@pytest.mark.parametrize("binding_name", ["venue_yaml", "venue_profile"])
def test_custom_physics_or_profile_bytes_are_not_accepted(
    tmp_path, monkeypatch, binding_name
):
    fixture = _fixture(tmp_path, monkeypatch)
    path = tmp_path / f"forged-{binding_name}.json"
    path.write_text('{"schema_version":"venue_profile_v1","physics":{}}\n')
    fixture["plan"]["bindings"][binding_name] = _binding(path, tmp_path)
    with pytest.raises(cert.CertificationError, match="exact code-rooted"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_quaternion_slerp_uses_shortest_antipodal_identity():
    lhs = np.array([1.0, 0.0, 0.0, 0.0])
    rhs = np.array([-1.0, 0.0, 0.0, 0.0])
    out = cert._slerp_wxyz(lhs, rhs, 0.5)
    assert abs(float(np.dot(out, lhs))) == pytest.approx(1.0)
