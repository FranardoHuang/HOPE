from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO
    / "hope_training/whole_body_tracking/scripts/"
    "mujoco_teacher_motion_native_ball_diagnostic.py"
)
PINNER = SCRIPT.with_name("pin_action_ball_profile_contracts.py")
SPEC = importlib.util.spec_from_file_location("teacher_native_ball_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_profile_pins(*, formal: bool) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(PINNER),
            "--repo-root",
            str(REPO),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    raw = json.loads(result.stdout)
    if formal:
        source_map_sha = GATE.sha256_bytes(
            GATE.canonical_json_bytes(
                raw["solver_implementation_source_sha256"]
            )
        )
        raw["source_authority"] = {
            "schema_version": 1,
            "authority": "external_exact_commit_subset_blob_map_v1",
            "commit_binding": (
                "external_preexec_immutable_launch_capsule_v1"
            ),
            "embedded_commit": False,
            "source_blob_map_sha256": source_map_sha,
        }
    return raw


def _pair() -> dict:
    return {
        "friction": [0.5, 0.5, 0.001, 0.0001, 0.0001],
        "solref": [0.002, 1.0],
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "condim": 3,
        "margin_m": 0.0,
        "gap_m": 0.0,
    }


def _material_raw() -> dict:
    return {
        "schema_version": 1,
        "certificate_type": "native_mujoco_pingpong_contact_material_v1",
        "certificate_id": "synthetic-test-only",
        "authorization": {
            "diagnostic_native_ab": True,
            "reviewed_by": "Unit Test",
        },
        "simulation": {
            "mujoco_version": "3.3.5",
            "timestep_s": 0.001,
            "integrator": "implicitfast",
        },
        "ball": {
            "radius_m": 0.02,
            "mass_kg": 0.0034,
            "inertia_coeff": 2.0 / 3.0,
        },
        "contact_pairs": {
            "ball_racket": _pair(),
            "ball_table": _pair(),
            "ball_net": _pair(),
        },
        "calibration": {
            "artifact_sha256": "1" * 64,
            "native_ball_racket_contact_verified": True,
            "native_ball_table_contact_verified": True,
        },
    }


def _material() -> dict:
    return GATE.validate_material_certificate(
        _material_raw(), expected_ball_radius_m=0.02
    )


def _passing_events() -> object:
    return GATE.NativeEvents(
        racket_contact_time_s=1.0,
        racket_contact_position_m=[0.2, 0.0, 1.2],
        incoming_ball_velocity_mps=[-3.0, 0.0, -0.5],
        contact_impulse_ns=0.02,
        net_crossing={
            "cleared": True,
            "ball_center_z_m": 1.05,
        },
        first_landing={
            "ball_center_m": [2.5, 0.1, 0.78],
            "descending": True,
            "native_contact": True,
        },
    )


def _reasons(events: object) -> list[str]:
    return GATE.evaluate_failure_reasons(
        events=events,
        target_contact_position=[0.2, 0.0, 1.2],
        target_incoming_velocity=[-3.0, 0.0, -0.5],
        expected_global_contact_time_s=1.0,
        contact_time_tolerance_s=0.02,
        contact_position_tolerance_m=0.02,
        incoming_velocity_tolerance_mps=0.1,
        opponent_near_x_m=1.87,
        opponent_far_x_m=3.24,
        table_half_width_m=0.7625,
    )


def test_checked_in_n5_named_manifest_fails_exact_n5_count_before_geometry_migration():
    path = REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
    raw, _receipt = GATE.read_json_exact(
        path, "manifest", expected_sha256=_sha(path)
    )
    with pytest.raises(GATE.GateError, match=r"action_count_mismatch.*N=5.*has 4"):
        GATE.validate_manifest(raw, expected_actions=5)


def test_manifest_t_hit_phase_and_cycle_are_one_frozen_timing_law():
    path = REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
    raw = json.loads(path.read_text())
    production_source = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
    )
    raw["racket_geometry_contract"] = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/"
            "racket_contact_geometry.py"
        ),
        "source_sha256": _sha(production_source),
        "geometry_source_sha256": (
            GATE.RACKET_GEOMETRY_SOURCE_SHA256
        ),
    }
    raw["actions"][0]["strike_phase"] += 0.01
    with pytest.raises(GATE.GateError, match="same frozen timing law"):
        GATE.validate_manifest(raw, expected_actions=4)


def test_physical_racket_geometry_binding_is_mandatory_and_versioned():
    with pytest.raises(GATE.GateError, match="must be an object"):
        GATE.validate_racket_geometry_binding(None)

    production_source = (
        REPO
        / "hope_training/whole_body_tracking/source/whole_body_tracking/"
        "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
    )
    binding = {
        "schema_version": 2,
        "semantics": "exact_face_contact_v2",
        "ball_target_point": "physical_ball_center_at_native_contact",
        "site_target_mapping": "site_target_from_ball_center",
        "face_velocity_mapping": (
            "site_linear_plus_omega_cross_face_center_offset"
        ),
        "source_path": (
            "hope_training/whole_body_tracking/source/whole_body_tracking/"
            "whole_body_tracking/tasks/tracking/mdp/racket_contact_geometry.py"
        ),
        "source_sha256": _sha(production_source),
        "geometry_source_sha256": GATE.RACKET_GEOMETRY_SOURCE_SHA256,
    }
    receipt = GATE.validate_racket_geometry_binding(binding)
    assert receipt["source_sha256"] == _sha(production_source)
    assert (
        receipt["geometry_source_sha256"]
        == GATE.RACKET_GEOMETRY_SOURCE_SHA256
    )
    assert receipt["red_legacy_colocation_error_m"] == pytest.approx(
        0.020040, abs=2.0e-6
    )
    assert receipt["black_legacy_colocation_error_m"] == pytest.approx(
        0.033232, abs=2.0e-6
    )


def test_material_certificate_requires_reviewed_native_contact_calibration():
    raw = _material_raw()
    raw["authorization"]["diagnostic_native_ab"] = False
    with pytest.raises(GATE.GateError, match="does not authorize"):
        GATE.validate_material_certificate(raw, expected_ball_radius_m=0.02)

    raw = _material_raw()
    raw["calibration"]["native_ball_racket_contact_verified"] = False
    with pytest.raises(GATE.GateError, match="ball-racket calibration"):
        GATE.validate_material_certificate(raw, expected_ball_radius_m=0.02)


def test_checked_in_profile_pins_reject_forbidden_worktree_revision():
    path = REPO / "configs/action_ball_profile_pins_20260728.json"
    raw, _receipt = GATE.read_json_exact(
        path, "profile pins", expected_sha256=_sha(path)
    )
    with pytest.raises(
        GATE.GateError, match=r"source_rev.*self-reference.*forbidden"
    ):
        GATE.validate_profile_pins(raw, manifest=None)


def test_profile_authority_binds_exact_five_source_blob_map_without_commit():
    diagnostic = _current_profile_pins(formal=False)
    with pytest.raises(GATE.GateError, match="source authority"):
        GATE.validate_profile_pins(diagnostic, manifest=None)

    raw = _current_profile_pins(formal=True)
    names = (
        "continuous_questions.py",
        "hope_commands.py",
        "racket_contact_geometry.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
    )
    profile = GATE.validate_profile_pins(raw, manifest=None)
    assert tuple(profile["solver_implementation_source_sha256"]) == names
    assert (
        profile["source_blob_map_sha256"]
        == raw["source_authority"]["source_blob_map_sha256"]
    )
    assert (
        profile["contact_geometry_sha256"]
        == GATE.RACKET_GEOMETRY_SOURCE_SHA256
    )

    raw["source_rev"] = "a" * 40
    with pytest.raises(
        GATE.GateError, match=r"source_rev.*self-reference.*forbidden"
    ):
        GATE.validate_profile_pins(raw, manifest=None)
    raw.pop("source_rev")
    raw["solver_implementation_source_sha256"].pop(
        "racket_contact_geometry.py"
    )
    raw["solver_payload"]["implementation_source_sha256"].pop(
        "racket_contact_geometry.py"
    )
    raw["solver_profile_sha256"] = GATE.sha256_bytes(
        GATE.canonical_json_bytes(raw["solver_payload"])
    )
    with pytest.raises(GATE.GateError, match="exact five solver files"):
        GATE.validate_profile_pins(raw, manifest=None)


def test_thin_shell_inertia_and_native_pair_parameters_are_explicit_in_scene():
    canonical = GATE.DEFAULT_MJCF.read_bytes()
    rows = GATE.table_scene.obstacle_geometry()
    xml_bytes, receipt = GATE.assemble_physical_scene_xml(
        canonical,
        obstacle_rows=rows,
        material=_material(),
    )
    text = xml_bytes.decode("utf-8")
    expected_inertia = (2.0 / 3.0) * 0.0034 * 0.02 * 0.02
    assert f'name="{GATE.BALL_JOINT_NAME}"' in text
    assert f'name="{GATE.BALL_GEOM_NAME}"' in text
    assert 'type="sphere"' in text
    assert f'name="teacher_ball_racket"' in text
    assert f'name="teacher_ball_table"' in text
    assert all(f'name="teacher_ball_{name}"' in text for name in GATE.NET_GEOM_NAMES)
    assert receipt["ball"]["diagonal_inertia_kg_m2"] == pytest.approx(
        expected_inertia, rel=0.0, abs=1.0e-18
    )
    assert receipt["physical_scene_xml_sha256"] == hashlib.sha256(
        xml_bytes
    ).hexdigest()


def test_reverse_integrator_only_constructs_incoming_initial_state_not_return_score():
    target_p = np.array([0.25, 0.1, 1.25])
    target_v = np.array([-3.2, 0.1, -0.4])
    spin = np.array([0.0, 0.0, 0.0])
    duration = 0.2
    p, v = GATE.reverse_free_flight(
        target_p,
        target_v,
        spin,
        duration_s=duration,
        k_d=0.0,
        k_m=0.0,
        gravity=9.81,
        max_step_s=0.0005,
    )
    predicted_v = v + np.array([0.0, 0.0, -9.81]) * duration
    predicted_p = (
        p
        + v * duration
        + 0.5 * np.array([0.0, 0.0, -9.81]) * duration * duration
    )
    assert np.allclose(predicted_p, target_p, atol=1.0e-10)
    assert np.allclose(predicted_v, target_v, atol=1.0e-10)


def test_formal_pass_requires_native_contact_net_crossing_and_native_first_landing():
    assert _reasons(_passing_events()) == []

    events = _passing_events()
    events.racket_contact_time_s = None
    assert "no_native_ball_racket_contact" in _reasons(events)

    events = _passing_events()
    events.net_crossing = None
    assert "no_post_hit_net_crossing" in _reasons(events)

    events = _passing_events()
    events.first_landing = None
    assert "no_native_first_table_landing" in _reasons(events)


def test_any_robot_table_contact_fails_even_for_feet_and_without_force_threshold():
    events = _passing_events()
    events.robot_obstacle_contacts.append(
        {
            "time_s": 0.0,
            "robot_geom": "left_foot_collision",
            "obstacle": "motion_table_top",
        }
    )
    assert "robot_hit_table_edge_or_net" in _reasons(events)


def test_non_racket_ball_contact_and_self_contact_fail_closed():
    events = _passing_events()
    events.ball_other_robot_contacts.append(
        {"time_s": 1.0, "other_geom": "right_hand_palm_collision"}
    )
    events.self_contacts.append(
        {"time_s": 0.5, "geoms": ["left_forearm", "torso"]}
    )
    reasons = _reasons(events)
    assert "ball_hit_non_racket_robot_geom" in reasons
    assert "robot_self_contact" in reasons


def test_script_has_native_authority_and_no_analytic_return_scorer_import():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "mujoco.mj_step(model, data)" in source
    assert "mujoco.mj_contactForce(" in source
    assert "from virtual_return_scorer" not in source
    assert "import virtual_return_scorer" not in source
    assert "ball_to_task_solver_executed" in source
    assert "analytic_or_counterfactual_return_scorer_executed" in source
    assert "formal_venue_physics_gate" in source
    assert "native_ball_racket_impulse_must_be_disabled_in_formal_referee" in source


def test_preflight_writes_no_clobber_receipt_for_current_n4_and_missing_material(
    tmp_path: Path,
):
    manifest = REPO / "configs/action_ball_n5_nomove_f20_20260728.json"
    pins = REPO / "configs/action_ball_profile_pins_20260728.json"
    out = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            _sha(manifest),
            "--profile-pins",
            str(pins),
            "--profile-pins-sha256",
            _sha(pins),
            "--preflight-only",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "BLOCKED"
    blockers = receipt["preflight"]["blockers"]
    assert any("action_count_mismatch" in reason for reason in blockers)
    assert "missing_pre_registered_native_mujoco_material_certificate" in blockers
    assert receipt["selector_executed"] is False
    assert receipt["ball_to_task_solver_executed"] is False
    assert receipt["analytic_or_counterfactual_return_scorer_executed"] is False

    second = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--preflight-only",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 2
    assert "refusing to overwrite" in second.stderr


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(GATE.GateError, match="duplicate JSON key"):
        GATE.read_json_exact(duplicate, "duplicate")

    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(GATE.GateError, match="non-finite JSON constant"):
        GATE.read_json_exact(nonfinite, "nonfinite")


def test_slerp_and_teacher_interpolation_remain_finite_at_endpoints():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array(
        [math.cos(0.25), 0.0, 0.0, math.sin(0.25)]
    )
    assert np.allclose(GATE.slerp_wxyz(q0, q1, 0.0), q0)
    assert np.allclose(GATE.slerp_wxyz(q0, q1, 1.0), q1)
