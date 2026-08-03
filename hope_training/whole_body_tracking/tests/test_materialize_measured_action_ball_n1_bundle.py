"""CPU-only contract tests for the measured VendorV2 N1 bundle materializer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_measured_action_ball_n1_bundle.py"
SPEC = importlib.util.spec_from_file_location("materialize_measured_n1", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def test_final_action_binding_is_exact_and_revoked_candidate_is_absent():
    assert module.ACTION_ID == "take_061_unit04_bh"
    assert module.MEASURED_UID == "Take_061_unit04_BH"
    assert module.ACTION_FACTS == {
        "action_uid": 5527597793770800,
        "motion_path": (
            "assets/motions/chingmu73_measured_v4_20260803/"
            "hope_Take_061_unit04_BH.npz"
        ),
        "motion_sha256": (
            "aab1953b9a857d0a7663a92d85fe4de5bd1d991d22249aa3d4d22ce7ef9fdd8e"
        ),
        "reference_t_hit_s": 0.96,
        "reference_t_cycle_s": 1.12,
        "reference_racket_site_speed_mps": 1.8901338577270508,
        "strike_phase": 0.8571,
        "family": "backhand",
    }


def test_prepare_does_not_bypass_full_solver_admission_preflight():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "skip_full_solver_preflight_for_immutable_tape=True" not in source
    assert "full_solver_preflight_support_source" in source


def test_five_recipes_keep_explicit_identity_when_masks_match():
    assert module.RECIPES == {
        "current_lm": (True, True, True),
        "analytic_full": (True, True, True),
        "analytic_no_velocity": (True, False, True),
        "teacher_pos_face_no_velocity": (True, False, True),
        "outcome_dense_only": (False, False, False),
    }
    assert module.RECIPES["current_lm"] == module.RECIPES["analytic_full"]
    assert (
        module.RECIPES["analytic_no_velocity"]
        == module.RECIPES["teacher_pos_face_no_velocity"]
    )


def test_fixed_question_projection_zeroes_widths_but_keeps_legal_support():
    profile = {
        "contact_offset_center_b_yaw_m": [1.0, 2.0, 3.0],
        "contact_offset_min_b_yaw_m": [0.0, 0.0, 0.0],
        "contact_offset_max_b_yaw_m": [4.0, 4.0, 4.0],
        "contact_offset_std_lower_initial_m": [0.1, 0.2, 0.3],
        "contact_offset_std_lower_max_m": [0.4, 0.5, 0.6],
        "contact_offset_std_upper_initial_m": [0.1, 0.2, 0.3],
        "contact_offset_std_upper_max_m": [0.4, 0.5, 0.6],
        "time_to_contact_center_s": 1.0,
        "time_to_contact_min_s": 0.5,
        "time_to_contact_max_s": 1.5,
        "time_to_contact_std_lower_initial_s": 0.1,
        "time_to_contact_std_lower_max_s": 0.2,
        "time_to_contact_std_upper_initial_s": 0.1,
        "time_to_contact_std_upper_max_s": 0.2,
        "incoming_speed_center_mps": 3.0,
        "incoming_speed_min_mps": 2.0,
        "incoming_speed_max_mps": 4.0,
        "spin_magnitude_center_radps": 0.0,
        "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 10.0,
        "base_spawn_center_w_xy_m": [0.1, 0.2],
        "base_spawn_min_w_xy_m": [-1.0, -1.0],
        "base_spawn_max_w_xy_m": [1.0, 1.0],
        "base_spawn_std_lower_initial_m": [0.1, 0.1],
        "base_spawn_std_lower_max_m": [0.2, 0.2],
        "base_spawn_std_upper_initial_m": [0.1, 0.1],
        "base_spawn_std_upper_max_m": [0.2, 0.2],
        "base_travel_center_b_yaw_xy_m": [0.0, 0.0],
        "base_travel_min_b_yaw_xy_m": [-1.0, -1.0],
        "base_travel_max_b_yaw_xy_m": [1.0, 1.0],
        "base_travel_std_lower_initial_m": [0.1, 0.1],
        "base_travel_std_lower_max_m": [0.2, 0.2],
        "base_travel_std_upper_initial_m": [0.1, 0.1],
        "base_travel_std_upper_max_m": [0.2, 0.2],
        "incoming_direction_tangent_u_neg_initial_deg": 3.0,
        "incoming_direction_tangent_u_neg_max_deg": 15.0,
        "incoming_direction_tangent_u_pos_initial_deg": 3.0,
        "incoming_direction_tangent_u_pos_max_deg": 15.0,
    }
    fixed = module._freeze_ball_profile(profile)
    assert fixed["contact_offset_min_b_yaw_m"] == [0.0, 0.0, 0.0]
    assert fixed["contact_offset_max_b_yaw_m"] == [4.0, 4.0, 4.0]
    assert fixed["time_to_contact_min_s"] == 0.5
    assert fixed["time_to_contact_max_s"] == 1.5
    assert fixed["incoming_speed_min_mps"] == 2.0
    assert fixed["incoming_speed_max_mps"] == 4.0
    assert fixed["base_spawn_min_w_xy_m"] == [-1.0, -1.0]
    assert fixed["base_spawn_max_w_xy_m"] == [1.0, 1.0]
    assert all(
        value == 0.0 or value == [0.0, 0.0, 0.0] or value == [0.0, 0.0]
        for key, value in fixed.items()
        if "std_lower_" in key or "std_upper_" in key
    )
    assert fixed["incoming_direction_tangent_u_neg_max_deg"] == 0.0


def test_tape_receipt_binds_recipe_mask_artifact_and_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "tape.npz"
    artifact.write_bytes(b"immutable tape")
    producers = {name: str(index) * 64 for index, name in enumerate(
        ("incoming_ball", "teacher_contact", "desired_contact", "landing_spin_task"), start=1
    )}
    receipt = _write(
        tmp_path / "receipt.json",
        {
            "schema_version": 1,
            "kind": module.TAPE_RECEIPT_KIND,
            "action_id": module.ACTION_ID,
            "target_recipe": "analytic_no_velocity",
            "target_validity": {"order": list(module.TARGET_ORDER), "mask": [True, False, True]},
            "artifact": {"path": "tape.npz", "sha256": _sha(artifact)},
            "row_count": 1,
            "per_column_producer_sha256": producers,
            "physical_ball_semantics": module.PHYSICAL_BALL_SEMANTICS,
            "reset_inverse_solve": False,
            "diagnostic_unauthorized": True,
        },
    )
    class FakeTape:
        canonical_sha256 = "5" * 64
        question_sha256 = "6" * 64
        source_receipt = SimpleNamespace(
            action_uid=module.ACTION_FACTS["action_uid"],
            action_slot=0,
            profile_sha256="a" * 64,
            motion_sha256=module.ACTION_FACTS["motion_sha256"],
            manifest_sha256="b" * 64,
            sampler_sha256="c" * 64,
            physics_sha256="d" * 64,
            solver_sha256="e" * 64,
            mobility_mode="no_move",
        )

        def target_lineage(self, recipe):
            return {
                "base_question_sha256": self.question_sha256,
                "target_recipe": recipe,
                "target_producer_sha256": (
                    producers["desired_contact"] if recipe == "analytic_no_velocity" else "9" * 64
                ),
                "target_column_sha256": "8" * 64,
                "target_validity_mask": list(module.RECIPES[recipe]),
                "tape_canonical_sha256": self.canonical_sha256,
            }

    fake_module = SimpleNamespace(
        TARGET_RECIPES=tuple(module.RECIPES),
        TARGET_VALIDITY_BY_RECIPE=module.RECIPES,
        load_immutable_n1_tape=lambda *_args, **_kwargs: FakeTape(),
    )
    monkeypatch.setattr(module, "_load_module", lambda *_args, **_kwargs: fake_module)
    pin, summary = module._validate_tape_receipt(
        tmp_path,
        receipt,
        _sha(receipt),
        action_id=module.ACTION_ID,
        action_uid=module.ACTION_FACTS["action_uid"],
        motion_sha=module.ACTION_FACTS["motion_sha256"],
        recipe="analytic_no_velocity",
    )
    assert pin["sha256"] == _sha(receipt)
    assert summary["artifact"]["sha256"] == _sha(artifact)
    assert summary["target_validity"]["mask"] == [True, False, True]
    assert summary["row_count"] == 1
    assert summary["selected_target_lineage"]["target_recipe"] == "analytic_no_velocity"


def test_mechanical_unknown_requires_explicit_diagnostic_acceptance(tmp_path: Path):
    motion_sha = "a" * 64
    report = _write(
        tmp_path / "mechanical.json",
        {
            "schema_version": 1,
            "kind": "measured_racket_mechanical_admission_audit_v1",
            "diagnostic_unauthorized": True,
            "actions": [
                {
                    "uid": module.MEASURED_UID,
                    "sha256": motion_sha,
                    "kinematic_limit_verdict": "PASS",
                    "mechanical_verdict": "UNKNOWN",
                    "mechanical_admitted": False,
                }
            ],
        },
    )
    with pytest.raises(module.BundleError, match="allow-mechanical-unknown"):
        module._mechanical_selection(
            tmp_path,
            report,
            _sha(report),
            motion_sha=motion_sha,
            action_uid=module.MEASURED_UID,
            allow_unknown=False,
        )
    _pin, selected = module._mechanical_selection(
        tmp_path,
        report,
        _sha(report),
        motion_sha=motion_sha,
        action_uid=module.MEASURED_UID,
        allow_unknown=True,
    )
    assert selected["mechanical_verdict"] == "UNKNOWN"
    assert selected["unknown_explicitly_accepted_for_sim_diagnostic"] is True


def test_alignment_receipt_requires_all_11_gates_and_diagonal_long_axis(tmp_path: Path):
    gates = {name: True for name in module.RACKET_ALIGNMENT_GATES}
    report = _write(
        tmp_path / "alignment.json",
        {
            "schema_version": 3,
            "kind": "materialized_measured_racket_fk_audit_v3",
            "uid": module.MEASURED_UID,
            "motion_sha256": module.ACTION_FACTS["motion_sha256"],
            "frames": 57,
            "finite": True,
            "admitted": True,
            "robot_butt_to_blade_axis_local": [math.sqrt(0.5), 0.0, math.sqrt(0.5)],
            "robot_rigid_visual_mesh_sha256": module.RACKET_MESH_SHA256,
            "gates": gates,
            "hit": {"frame": 48, "position_error_m": 0.001},
            "position_error_m": {"p95": 0.01},
            "face_error_deg": {"p95": 1.0},
            "long_axis_error_deg": {"p95": 1.0},
            "so3_error_deg": {"p95": 1.0},
            "authorization": {
                "diagnostic_unauthorized": True,
                "training": False,
                "promotion": False,
                "deployment": False,
            },
        },
    )
    _pin, summary = module._validate_racket_alignment(
        tmp_path,
        report,
        _sha(report),
        motion_sha=module.ACTION_FACTS["motion_sha256"],
        action_uid=module.MEASURED_UID,
        frame_count=57,
        strike_frame=48,
    )
    assert summary["all_11_gates_pass"] is True

    failed = json.loads(report.read_text())
    failed["gates"]["hit_velocity_direction_le_15_deg"] = False
    failed_path = _write(tmp_path / "alignment_failed.json", failed)
    with pytest.raises(module.BundleError, match="failed gate"):
        module._validate_racket_alignment(
            tmp_path,
            failed_path,
            _sha(failed_path),
            motion_sha=module.ACTION_FACTS["motion_sha256"],
            action_uid=module.MEASURED_UID,
            frame_count=57,
            strike_frame=48,
        )

    missing = json.loads(report.read_text())
    del missing["gates"]["hit_velocity_direction_observable"]
    missing_path = _write(tmp_path / "alignment_missing_gate.json", missing)
    with pytest.raises(module.BundleError, match="failed gate"):
        module._validate_racket_alignment(
            tmp_path,
            missing_path,
            _sha(missing_path),
            motion_sha=module.ACTION_FACTS["motion_sha256"],
            action_uid=module.MEASURED_UID,
            frame_count=57,
            strike_frame=48,
        )


def test_local_v4_bank_and_build_report_bind_the_frozen_action():
    root = Path(__file__).resolve().parents[3]
    bank = root / "assets/motions/chingmu73_measured_v4_20260803/BANK_IMPORT_RECEIPT.json"
    report = root / "configs/action_ball_chingmu73_measured_v4_f10_20260803.buildreport.json"
    source = root / "configs/action_ball_chingmu73_measured_v4_f10_20260803.json"
    if not (bank.exists() and report.exists() and source.exists()):
        pytest.skip("local measured-racket authority assets are unavailable")
    bank_pin, report_pin, evidence = module._validate_measured_provenance(
        root,
        bank_receipt_path=bank,
        bank_receipt_sha=_sha(bank),
        build_report_path=report,
        build_report_sha=_sha(report),
        source_manifest_sha=_sha(source),
        action_id=module.ACTION_ID,
        measured_uid=module.MEASURED_UID,
        motion_sha=module.ACTION_FACTS["motion_sha256"],
        frame_count=57,
        strike_frame=48,
    )
    assert bank_pin["sha256"] == _sha(bank)
    assert report_pin["sha256"] == _sha(report)
    assert evidence["bank_action_row"]["hit_frame_50"] == 48
    assert evidence["build_report_action"]["racket_authority"] == "measured_channel"
