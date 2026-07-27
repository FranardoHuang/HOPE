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
    passing_gate = {"pass": True}
    return {
        "verdict": "PASS",
        "artifacts": {
            "motion_sha256": motion_sha,
            "mjcf_sha256": mjcf_sha,
        },
        "contract": {
            "racket_site": cert.RACKET_SITE_NAME,
            "racket_site_body": cert.RACKET_SITE_BODY,
            "racket_site_local_position_m": list(cert.RACKET_SITE_OFFSET_WRIST_M),
        },
        "evidence_boundary": {
            "level": "kinematic_playback_only",
            "mj_forward_calls": 2 * frames,
            "mj_step_calls": 0,
            "training_certificate": False,
            "racket_velocity_source": cert.RACKET_VELOCITY_SOURCE,
        },
        "authorization": {
            "training": False,
            "deployment": False,
            "hardware": False,
        },
        "gates": {
            "position": passing_gate,
            "orientation": passing_gate,
            "racket_site_position_vs_schema": passing_gate,
            "racket_site_normal_vs_schema": passing_gate,
            "racket_site_linear_velocity_vs_schema": passing_gate,
            "racket_site_angular_velocity_vs_schema": passing_gate,
            "racket_site_jacobian_vs_object_velocity": passing_gate,
            "table_contact": {"enabled": True, "pass": True},
        },
        "racket": {
            "array_receipts": receipts,
            "trajectory_sha256": combined.hexdigest(),
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
            **{name: {"pass": True} for name in component_names},
            "aggregate": {"pass": True},
        },
        "clearance": {"minimum_table_net_clearance_m": 0.02},
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
    marker = tmp_path / "marker.json"
    _write_json(marker, {"fixture": True})
    mjcf = tmp_path / "model.xml"
    mjcf.write_bytes(b"<mujoco/>")
    urdf = tmp_path / "model.urdf"
    urdf.write_bytes(b"<robot/>")
    venue = tmp_path / "venue.yaml"
    venue.write_text("fixture: true\n", encoding="utf-8")
    compiled_signature = "f" * 64
    recipe = tmp_path / "recipe.json"
    _write_json(
        recipe,
        {
            "canonical_ready": {"path": ready.name, "sha256": _sha(ready)},
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
            "manifest": {"path": manifest.name, "sha256": _sha(manifest)},
            "bound_inputs": {
                "recipe": {"path": recipe.name, "sha256": _sha(recipe)},
                "mjcf": {"path": mjcf.name, "sha256": _sha(mjcf)},
                "plant": {
                    "mjcf_sha256": _sha(mjcf),
                    "urdf_sha256": _sha(urdf),
                    "compiled_signature_sha256": compiled_signature,
                    "identity_bound": True,
                },
                "verifier_tools": {
                    "bank_gate": {
                        "sha256": _sha(
                            REPO
                            / "hope_training/whole_body_tracking/scripts/"
                            "canonical_motion_bank_gate.py"
                        )
                    },
                    "mujoco_motion_player": {
                        "sha256": _sha(
                            REPO
                            / "hope_training/whole_body_tracking/scripts/"
                            "mujoco_motion_player.py"
                        )
                    },
                    "canonical_mujoco_dynamics_gate": {
                        "sha256": _sha(
                            REPO
                            / "hope_training/whole_body_tracking/scripts/"
                            "canonical_mujoco_dynamics_gate.py"
                        )
                    },
                },
            },
            "clips": [
                {
                    "motion_id": action,
                    "scope": scope,
                    "sha256": _sha(motions[scope]),
                    "mujoco_fk": {"pass": True},
                    "plant_specific_dynamics": {"screen_pass": False},
                }
                for scope in cert.SCOPES
            ],
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
        },
        "required_scopes": list(cert.SCOPES),
        "station_center_shift_candidates_xy_m": [
            list(row) for row in cert.STATION_CENTER_SHIFT_CANDIDATES_XY_M
        ],
        "selected_station_center_shift_xy_m": [0.0, 0.0],
        "thresholds": {
            "source_anchor_time_min_s": 0.03,
            "source_anchor_time_max_s": 0.06,
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
            "spin_abs_max_rad_s": 0.0,
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
    collision["checks"]["aggregate"]["pass"] = False
    collision["clearance"]["minimum_table_net_clearance_m"] = -0.01
    _write_json(path, collision)
    fixture["plan"]["scopes"][scope]["collision_reports"][station_index] = _binding(
        path, tmp_path
    )


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
    assert report["diagnostic_reference_checks_pass"] is True
    assert report["diagnostic_smoke_authorized"] is False
    assert report["training_authorized"] is False
    assert report["deployment_authorized"] is False
    assert report["hardware_authorized"] is False
    assert report["verdict"] == "REFERENCE_CHECKS_PASS_UNAUTHORIZED"
    assert "external_playback" in report["diagnostic_smoke_blockers"][0]
    assert any("grounded_collocation_trace_missing" in row for row in report["training_blockers"])
    for scope in cert.SCOPES:
        timing = report["scopes"][scope]["timing"]
        assert timing["post_retime_behavior_t_hit_measured"] is False
        assert timing["historical_universal_0p5_gate_applied"] is False


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


def test_comparison_never_auto_adopts_station_center(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["selected_station_center_shift_xy_m"] = None
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["selected_station_center_shift_xy_m"] is None
    assert "not_selected" in report["diagnostic_reference_blockers"][0]


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
    fixture["plan"]["selected_station_center_shift_xy_m"] = list(selected)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["selected_station_center_shift_xy_m"] == list(selected)
    assert report["diagnostic_reference_checks_pass"] is True
    assert report["diagnostic_smoke_authorized"] is False


@pytest.mark.parametrize("selected_x", [-0.05, -0.10])
def test_farther_station_is_rejected_when_a_nearer_station_common_passes(
    tmp_path, monkeypatch, selected_x
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["selected_station_center_shift_xy_m"] = [selected_x, 0.0]
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.75)
    with pytest.raises(cert.CertificationError, match="nearest upper/full"):
        cert.certify_plan(fixture["plan"], base_dir=tmp_path)


def test_failed_selected_station_center_blocks_reference_checks(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["plan"]["selected_station_center_shift_xy_m"] = [0.0, 0.0]
    for station_index in range(len(cert.STATION_CENTER_SHIFT_CANDIDATES_XY_M)):
        _fail_collision(
            fixture,
            tmp_path,
            scope="upper",
            station_index=station_index,
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
    assert "full/physical_blade_site_speed" in report["diagnostic_reference_blockers"]
    assert report["training_authorized"] is False


def test_low_reference_return_fraction_blocks_reference_checks(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cert, "_reference_return_fraction", lambda **_: 0.0)
    report = cert.certify_plan(fixture["plan"], base_dir=tmp_path)
    assert report["diagnostic_reference_checks_pass"] is False
    assert "upper/reference_returnability" in report["diagnostic_reference_blockers"]
    assert "full/reference_returnability" in report["diagnostic_reference_blockers"]
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
    gate = cert._motion_ready_truth_gate(payload, ready, tolerance=1.0e-6)
    assert gate["pass"] is False
    assert gate["joint_position_max_abs_error_rad"] == pytest.approx(0.1)


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
    assert "upper/shared_ready_return" in report["diagnostic_reference_blockers"]
    assert "full/shared_ready_return" in report["diagnostic_reference_blockers"]


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


def test_quaternion_slerp_uses_shortest_antipodal_identity():
    lhs = np.array([1.0, 0.0, 0.0, 0.0])
    rhs = np.array([-1.0, 0.0, 0.0, 0.0])
    out = cert._slerp_wxyz(lhs, rhs, 0.5)
    assert abs(float(np.dot(out, lhs))) == pytest.approx(1.0)
