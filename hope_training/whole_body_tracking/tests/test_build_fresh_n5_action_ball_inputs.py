from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SCRIPT = ROOT / "scripts" / "build_fresh_n5_action_ball_inputs.py"
SPEC = importlib.util.spec_from_file_location(
    "build_fresh_n5_action_ball_inputs_under_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def _canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _motion(path, *, yaw=math.pi / 2.0):
    frames = 7
    names = np.asarray(
        ["right_wrist_yaw_Link", "pelvis_link"],
        dtype=str,
    )
    positions = np.zeros((frames, 2, 3), dtype=np.float64)
    positions[:, 0] = np.asarray([0.70, -0.10, 1.05])
    positions[:, 0, 0] += np.arange(frames) * 0.01
    positions[:, 1] = np.asarray([1.20, 0.30, 0.92])
    quaternions = np.zeros((frames, 2, 4), dtype=np.float64)
    quaternions[:, 0, 0] = 1.0
    quaternions[:, 1, 0] = math.cos(yaw / 2.0)
    quaternions[:, 1, 3] = math.sin(yaw / 2.0)
    angular = np.zeros((frames, 2, 3), dtype=np.float64)
    np.savez(
        path,
        fps=np.asarray(50.0),
        body_names=names,
        body_pos_w=positions,
        body_quat_w=quaternions,
        body_ang_vel_w=angular,
        joint_pos=np.zeros((frames, 1), dtype=np.float64),
    )


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    mdp = repo / B.MDP_DIR_REL
    mdp.mkdir(parents=True)
    source_mdp = REPO_ROOT / B.MDP_DIR_REL
    for name in B.SOLVER_SOURCE_NAMES:
        shutil.copyfile(source_mdp / name, mdp / name)

    geometry = B._load_module(
        "fresh_n5_test_geometry_%s" % tmp_path.name,
        mdp / "racket_contact_geometry.py",
    )
    bank_dir = repo / "assets" / "bank"
    bank_dir.mkdir(parents=True)
    outputs = []
    upper_sha = {}
    for motion_id in B.FRESH_BANK_ORDER:
        for scope in ("upper", "full"):
            filename = "%s_%s_canonical_v2.npz" % (motion_id, scope)
            path = bank_dir / filename
            _motion(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            outputs.append(
                {
                    "motion_id": motion_id,
                    "scope": scope,
                    "filename": filename,
                    "output_npz_sha256": digest,
                }
            )
            if scope == "upper":
                upper_sha[motion_id] = digest
    bank = {
        "output_matrix": {
            "motion_ids": list(B.FRESH_BANK_ORDER),
            "scopes": ["upper", "full"],
            "candidate_count": 14,
        },
        "outputs": outputs,
    }
    bank_path = bank_dir / "BUILD_MANIFEST.json"
    _write_json(bank_path, bank)

    prototype_rows = []
    for index, action_id in enumerate(B.FRESH_N5_ACTION_ORDER):
        prototype_rows.append(
            {
                "motion_id": action_id,
                "scope": "upper",
                "clip_index": index,
                "family": B.FRESH_N5_FAMILIES[action_id],
                "npz_sha256": upper_sha[action_id],
                "contact_frame": 3,
                "strike_phase": 0.5,
                "face_sign": (
                    1.0 if B.FRESH_N5_FAMILIES[action_id] == "forehand" else -1.0
                ),
            }
        )
    scopes = {"upper": prototype_rows}
    prototype = {
        "schema_version": 2,
        "scopes": scopes,
        "derived_sha256": _canonical_sha(scopes),
    }
    prototype_path = repo / "configs" / "prototype.json"
    _write_json(prototype_path, prototype)

    venue_path = repo / "configs" / "ball_physics_venue.yaml"
    venue_path.parent.mkdir(parents=True, exist_ok=True)
    venue_path.write_text("fixture: true\n", encoding="utf-8")
    source_map = {
        name: hashlib.sha256((mdp / name).read_bytes()).hexdigest()
        for name in B.SOLVER_SOURCE_NAMES
    }
    physics_payload = {
        "kind": "fixture.physics",
        "venue_source": {
            "path": "configs/ball_physics_venue.yaml",
            "file_sha256": hashlib.sha256(venue_path.read_bytes()).hexdigest(),
        },
    }
    solver_payload = {"kind": "fixture.solver"}
    pins = {
        "solver_implementation_source_sha256": source_map,
        "contact_geometry": {
            "payload": geometry.GEOMETRY_SOURCE_PAYLOAD,
            "sha256": geometry.GEOMETRY_SOURCE_SHA256,
        },
        "physics_payload": physics_payload,
        "physics_profile_sha256": _canonical_sha(physics_payload),
        "solver_payload": solver_payload,
        "solver_profile_sha256": _canonical_sha(solver_payload),
        "venue_yaml_sha256": hashlib.sha256(venue_path.read_bytes()).hexdigest(),
    }
    pins_path = repo / "configs" / "pins.json"
    _write_json(pins_path, pins)
    incoming = {
        "sampling_spec": {
            "pooled_matchlike": {
                "trunc_gaussian": {
                    "variables": [
                        "vx",
                        "vy",
                        "vz",
                        "z_above_surface",
                        "w_norm",
                    ],
                    "mean": [-2.0, 0.0, -0.5, 0.3, 10.0],
                    "cov": np.eye(5).tolist(),
                    "clip_lo": [-5.0, -2.0, -3.0, 0.05, 0.0],
                    "clip_hi": [-0.1, 2.0, 2.0, 1.0, 100.0],
                }
            }
        }
    }
    incoming_path = repo / "assets" / "incoming_dist.json"
    _write_json(incoming_path, incoming)
    return {
        "repo": repo,
        "bank": bank_path,
        "prototype": prototype_path,
        "pins": pins_path,
        "venue": venue_path,
        "geometry": geometry,
        "incoming": incoming_path,
        "incoming_sha": hashlib.sha256(incoming_path.read_bytes()).hexdigest(),
    }


def _passing_inverse(**kwargs):
    count = int(kwargs["proposal_count"])
    return {
        "status": "PASS",
        "proposal_count": count,
        "legal_count": count - 7,
        "rejection_counts": {"net_not_cleared": 7},
        "selected_proposal_index": 11,
        "incoming_velocity_w_mps": [-2.5, 0.1, -0.4],
        "outgoing_velocity_w_mps": [3.0, -0.1, 1.2],
        "outgoing_spin_w_radps": [0.0, 2.0, 0.0],
        "legal_landing_w_xy_m": [2.5, 0.0],
        "legal_net_z_m": 1.1,
    }


def test_batch_exact_order_frame_hash_and_lossless_inverse_ledger(tmp_path):
    fixture = _fixture(tmp_path)
    document = B.build_batch_document(
        bank_manifest_path=fixture["bank"],
        prototype_path=fixture["prototype"],
        profile_pins_path=fixture["pins"],
        venue_yaml=fixture["venue"],
        repo_root=fixture["repo"],
        seed=17,
        proposal_count=100,
        incoming_dist_path=fixture["incoming"],
        incoming_dist_sha256=fixture["incoming_sha"],
        inverse_screen=_passing_inverse,
    )

    assert tuple(document["action_order"]) == B.FRESH_N5_ACTION_ORDER
    assert tuple(row["uid"] for row in document["units"]) == B.FRESH_N5_ACTION_ORDER
    assert not ({row["uid"] for row in document["units"]} & B.FRESH_N5_FORBIDDEN)
    assert document["selector_executed"] is False
    assert document["mobility_mode"] == "no_move"
    assert document["base_task_frame"] == "relative_about_actual_episode_spawn"
    assert document["source_bank"]["required_matrix"] == "ordered_complete_7x2"
    for index, screen in enumerate(document["inverse_screen"]["screens"]):
        assert screen["proposal_count"] == 100
        assert screen["legal_count"] == 93
        assert screen["rejection_counts"] == {"net_not_cleared": 7}
        assert screen["seed"] == 17 + 1009 * index

    first = document["units"][0]
    assert first["hit_frame_50"] == 3
    assert first["strike_phase"] == 0.5
    assert first["station_xy_hope_m"] == pytest.approx([0.7, -0.4625])
    site = np.asarray([0.73, -0.10, 1.05]) + np.asarray(
        fixture["geometry"].RACKET_SITE_OFFSET_WRIST_M
    )
    ball = site + np.asarray(fixture["geometry"].ball_center_from_site_local(-1))
    assert first["ball_pos_hit_hope_m"] == pytest.approx(
        [ball[0] - 0.5, ball[1] - 0.7625, ball[2] - 0.76]
    )
    assert first["contact_point_semantics"] == (
        "physical_ball_center_at_exact_teacher_strike"
    )


def test_complete_bank_hash_drift_and_prototype_order_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    last = fixture["bank"].parent / "v12_forehand_block_full_canonical_v2.npz"
    last.write_bytes(last.read_bytes() + b"drift")
    with pytest.raises(B.FreshN5BuildError, match="clip hash drifted"):
        B.build_batch_document(
            bank_manifest_path=fixture["bank"],
            prototype_path=fixture["prototype"],
            profile_pins_path=fixture["pins"],
            venue_yaml=fixture["venue"],
            repo_root=fixture["repo"],
            seed=1,
            proposal_count=64,
            incoming_dist_path=fixture["incoming"],
            incoming_dist_sha256=fixture["incoming_sha"],
            inverse_screen=_passing_inverse,
        )

    fixture = _fixture(tmp_path / "second")
    prototype = json.loads(fixture["prototype"].read_text(encoding="utf-8"))
    prototype["scopes"]["upper"][0], prototype["scopes"]["upper"][1] = (
        prototype["scopes"]["upper"][1],
        prototype["scopes"]["upper"][0],
    )
    prototype["derived_sha256"] = _canonical_sha(prototype["scopes"])
    _write_json(fixture["prototype"], prototype)
    with pytest.raises(B.FreshN5BuildError, match="action order drifted"):
        B.build_batch_document(
            bank_manifest_path=fixture["bank"],
            prototype_path=fixture["prototype"],
            profile_pins_path=fixture["pins"],
            venue_yaml=fixture["venue"],
            repo_root=fixture["repo"],
            seed=1,
            proposal_count=64,
            incoming_dist_path=fixture["incoming"],
            incoming_dist_sha256=fixture["incoming_sha"],
            inverse_screen=_passing_inverse,
        )


def test_no_legal_center_rejects_before_formal_output(tmp_path):
    fixture = _fixture(tmp_path)
    target = fixture["repo"] / "out" / "batch.json"

    def no_legal(**kwargs):
        return {
            "status": "NO_LEGAL_CENTER",
            "proposal_count": kwargs["proposal_count"],
            "legal_count": 0,
            "rejection_counts": {
                "landing_x_outside_opponent_table": kwargs["proposal_count"]
            },
        }

    with pytest.raises(B.FreshN5BuildError, match="no legal centre"):
        B.build_batch_document(
            bank_manifest_path=fixture["bank"],
            prototype_path=fixture["prototype"],
            profile_pins_path=fixture["pins"],
            venue_yaml=fixture["venue"],
            repo_root=fixture["repo"],
            seed=1,
            proposal_count=64,
            incoming_dist_path=fixture["incoming"],
            incoming_dist_sha256=fixture["incoming_sha"],
            inverse_screen=no_legal,
        )
    assert not target.exists()


def test_exclusive_writer_rejects_overwrite_and_broken_symlink(tmp_path):
    target = tmp_path / "artifact.json"
    B._write_exclusive(target, {"value": 1})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        B._write_exclusive(target, {"value": 2})
    assert target.read_bytes() == original

    occupied = tmp_path / "occupied.json"
    occupied.symlink_to(tmp_path / "missing-target")
    assert os.path.lexists(str(occupied))
    with pytest.raises(FileExistsError):
        B._write_exclusive(occupied, {"value": 3})


def test_physical_launch_map_is_exact_one_bounce_and_hash_bound(tmp_path):
    repo = tmp_path / "repo"
    artifacts = repo / "launch_artifacts"
    artifacts.mkdir(parents=True)
    actions = {}
    rows = []
    for index, action_id in enumerate(B.FRESH_N5_ACTION_ORDER):
        motion_sha = "%064x" % (index + 1)
        actions[action_id] = {
            "action_uid": index + 10,
            "motion_sha256": motion_sha,
        }
        artifact = artifacts / ("%s.json" % action_id)
        artifact.write_text("{}\n", encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        launch_payload = {
            "activation_time_s": 0.0,
            "position_w_m": [3.0, 0.0, 1.2],
            "velocity_w_mps": [-3.0, 0.0, -1.0],
            "spin_w_radps": [0.0, 0.0, 0.0],
            "required_incoming_table_bounces": 1,
        }
        case_launch = {
            **launch_payload,
            "state_sha256": _canonical_sha(launch_payload),
        }
        physical_payload = {
            "source": "pre_registered_native_shooting_receipt_v1",
            **launch_payload,
            "source_artifact_path": artifact.relative_to(repo).as_posix(),
            "source_artifact_sha256": artifact_sha,
        }
        rows.append(
            {
                "action_id": action_id,
                "action_uid": index + 10,
                "motion_sha256": motion_sha,
                "physical_ball_launch": {
                    **physical_payload,
                    "state_sha256": _canonical_sha(physical_payload),
                },
                "case_launches": {
                    "center": case_launch,
                    "support": case_launch,
                },
            }
        )
    path = repo / "launch_map.json"
    document = {
        "schema_version": 1,
        "artifact_type": "fresh_n5_physical_launch_map_v1",
        "action_order": list(B.FRESH_N5_ACTION_ORDER),
        "actions": rows,
    }
    _write_json(path, document)
    loaded = B._load_physical_launches(path, repo, actions)
    assert tuple(loaded) == B.FRESH_N5_ACTION_ORDER
    assert all(
        row["center"]["required_incoming_table_bounces"] == 1 for row in loaded.values()
    )

    document["actions"][0]["physical_ball_launch"][
        "source"
    ] = "recorded_pre_hit_state_v1"
    _write_json(path, document)
    with pytest.raises(B.FreshN5BuildError, match="state SHA mismatch"):
        B._load_physical_launches(path, repo, actions)


def test_solver_case_contract_is_exact_three_positive_three_negative():
    assert B.CASE_ROLES == (
        "center_positive_seed_0",
        "center_positive_seed_1",
        "support_positive",
        "negative_t_hit_offset",
        "negative_face_sign",
        "negative_ball_state_mismatch",
    )
    assert len(B.POSITIVE_CASE_ROLES) == 3
    assert set(B.NEGATIVE_REASONS) == set(B.CASE_ROLES[3:])


def test_fitted_gate_materialization_contracts_pin_exact_runtime_closure():
    gate = B._load_formal_fitted_gate(REPO_ROOT)
    venue_path = REPO_ROOT / "configs" / "ball_physics_venue.yaml"
    pins = {
        "geometry_source_sha256": gate.racket_geometry.GEOMETRY_SOURCE_SHA256,
        "raw": {
            "venue_yaml_sha256": hashlib.sha256(venue_path.read_bytes()).hexdigest(),
        },
    }
    fields = B._build_fitted_gate_contracts(
        gate=gate,
        repo_root=REPO_ROOT,
        profile_pins=pins,
    )

    geometry = fields["racket_geometry_contract"]
    assert geometry["schema_version"] == 2
    assert geometry["semantics"] == "exact_face_contact_v2"
    assert (
        geometry["geometry_source_sha256"]
        == gate.racket_geometry.GEOMETRY_SOURCE_SHA256
    )
    validated_geometry = gate.native_diag.validate_racket_geometry_binding(geometry)
    assert (
        validated_geometry["geometry_source_sha256"]
        == gate.racket_geometry.GEOMETRY_SOURCE_SHA256
    )

    physical = fields["physical_contact_contract"]
    assert physical["schema_version"] == gate.CONTACT_CONTRACT_VERSION
    assert physical["authority"] == gate.CONTACT_AUTHORITY
    assert set(physical["runtime_source_sha256"]) == set(gate.RUNTIME_SOURCE_PATHS)
    assert set(physical["runtime_execution_source_sha256"]) == set(
        gate.RUNTIME_EXECUTION_SOURCE_PATHS
    )
    assert set(physical["runtime_execution_data_sha256"]) == set(
        gate.RUNTIME_EXECUTION_DATA_PATHS
    )
    assert set(physical["selected_face_mesh_sha256"]) == set(gate.FACE_MESH_PIN_KEYS)
    assert physical["convergence_timestep_s"] == list(gate.DEFAULT_DT_S)


def test_one_bounce_shooter_uses_formal_forward_primitives_and_dual_dt():
    gate = B._load_formal_fitted_gate(REPO_ROOT)
    venue_path = REPO_ROOT / "configs" / "ball_physics_venue.yaml"
    venue = gate.load_venue_yaml(
        venue_path,
        hashlib.sha256(venue_path.read_bytes()).hexdigest(),
    )
    profile = {
        "center_surface_z_m": 0.78,
        "eroded_near_x_m": 0.525,
        "eroded_far_x_m": 3.215,
        "eroded_half_width_m": 0.7375,
        "net_x_m": 1.87,
        "net_top_z_m": 0.9325,
    }
    known = B._replay_one_bounce_launch(
        gate=gate,
        launch_position_m=[3.0, 0.0, 1.3],
        launch_velocity_mps=[-3.0, 0.0, -1.0],
        launch_spin_radps=[0.0, 0.0, 0.0],
        duration_s=0.7,
        venue=venue,
        table_profile=profile,
        step_s=0.0005,
    )
    assert known["bounce_count"] == 1
    solved = B.solve_one_bounce_launch(
        gate=gate,
        target_position_m=known["arrival_position_m"],
        target_velocity_mps=known["arrival_velocity_mps"],
        target_spin_radps=known["arrival_spin_radps"],
        time_to_contact_s=0.7,
        venue=venue,
        table_profile=profile,
        seed=9,
        proposal_count=4,
    )
    assert solved["status"] == "PASS"
    assert solved["proposal_count"] == 4
    assert solved["formal_forward_primitives"] == [
        "advance_fitted_flight",
        "swept_table_crossing",
        "fitted_contact",
    ]
    assert solved["selected_coarse_replay"]["step_s"] == 0.001
    assert solved["selected_fine_replay"]["step_s"] == 0.0005
    assert (
        solved["selected_fine_replay"]["bounce_count"]
        == solved["selected_coarse_replay"]["bounce_count"]
        == 1
    )
    assert solved["selected_residuals"]["arrival_position_m"] < 0.002
    assert solved["selected_residuals"]["arrival_velocity_mps"] < 0.02
    assert solved["selected_residuals"]["arrival_spin_radps"] < 0.05
