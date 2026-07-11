from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "screen_motion_gmr_phase_safety.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("screen_motion_gmr_phase_safety", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_binding(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


EXPECTED_IDS = [
    "franco_forehand_block",
    "franco_backhand_block",
    "franco_forehand_loop",
    "franco_backhand_loop_a",
    "franco_backhand_loop_b",
    "franco_backhand_loop_c",
    "v6_forehand_block",
    "v6_backhand_block",
    "v7_forehand_block",
    "v7_backhand_block",
]


def _schedule(count_per_side: int = 4) -> dict:
    value = {
        "algorithm": mod.ALGORITHM,
        "seed": "unit-test-paper",
        "count_per_side": count_per_side,
        "sides": ["forehand", "backhand"],
        "venue_contact_x_from_net_m": [-1.50, -0.92],
        "venue_contact_z_above_table_m": [0.22, 0.50],
        "incoming_vx_mps": [-2.85, -0.82],
        "incoming_vy_mps": [-0.46, 0.31],
        "incoming_vz_mps": [-2.02, 0.49],
        "spin_magnitude_radps": [0.0, 34.0],
        "forehand_contact_y_m": [-0.35, -0.155],
        "backhand_contact_y_m": [-0.155, 0.18],
        "table_geometry": {
            "surface_z_m": 0.76,
            "net_x_m": 1.87,
            "far_x_m": 3.24,
            "half_width_m": 0.7625,
            "net_height_m": 0.1525,
        },
    }
    questions = mod.build_questions(value)
    rows = [
        {
            "question_id": question.question_id,
            "side": question.side,
            "ball_pos_w_m": question.ball_pos_w.tolist(),
            "ball_vel_w_mps": question.ball_vel_w.tolist(),
            "ball_spin_w_radps": question.ball_spin_w.tolist(),
        }
        for question in questions
    ]
    value["expected_semantic_sha256"] = mod.canonical_sha256(rows)
    return value


def _blocked_manifest(tmp_path: Path) -> dict:
    mjcf = (
        REPO
        / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
        "a3_pingpong/a3_pingpong.xml"
    )
    physics = REPO / "configs/ball_physics_venue.yaml"
    venue_distribution = REPO / "configs/incoming_ball_venue.yaml"
    grounding_result = tmp_path / "canonical_grounding_result.json"
    grounding_result.write_text("{}\n", encoding="utf-8")
    tool_paths = {
        "screen": SCRIPT,
        "grounding_dependency": REPO / "scripts/ground_gmr_pkl.py",
        "self_collision_dependency": (
            REPO / "hope_training/whole_body_tracking/scripts/audit_self_collision.py"
        ),
        "motion_audit_dependency": (
            REPO / "hope_training/whole_body_tracking/scripts/audit_motion_npz.py"
        ),
        "virtual_return_dependency": (
            REPO / "hope_training/whole_body_tracking/scripts/virtual_return_scorer.py"
        ),
    }
    return {
        "schema_version": 1,
        "plan_id": "motion-video-gmr-phase-safety-20260711-v1",
        "status": "preregistered_blocked_on_canonical_grounding",
        "execution_ready": False,
        "scope": "unit test",
        "cpu_only": True,
        "CUDA_VISIBLE_DEVICES": "",
        "real_robot_commands_authorized": False,
        "input_mode": "explicit_manifest_only_no_directory_scan",
        "contact_phase_truth": None,
        "body_shape_contract": mod.BODY_SHAPE_CONTRACT,
        "canonical_grounding_result_manifest": _file_binding(grounding_result),
        "tool_contract": {name: _file_binding(path) for name, path in tool_paths.items()},
        "mjcf": {
            **_file_binding(mjcf),
            "compiled_kinematic_collision_sha256": "1" * 64,
        },
        "physics": _file_binding(physics),
        "venue_distribution": _file_binding(venue_distribution),
        "dense_safety_contract": {
            "substeps_per_source_interval": 8,
            "ground_penetration_tolerance_m": 0.0005,
            "self_collision_penetration_tolerance_m": 0.000001,
            "hard_racket_body_clearance_m": 0.005,
            "warning_racket_body_clearance_m": 0.02,
            "racket_body_clearance_groups": {
                "head_neck": ["head_yaw_collision", "head_pitch_collision"],
                "trunk": ["torso_collision", "pelvis_collision"],
            },
        },
        "question_schedule": _schedule(),
        "phase_selection_contract": {
            "minimum_racket_speed_mps": 0.3,
            "rank_order": [
                "exact_return_count",
                "intrinsic_return_count",
                "median_exact_return_margin_m",
                "dense_racket_body_clearance_m",
                "earlier_frame",
            ],
        },
        "frame_contract": {
            "returnability_enabled": False,
            "gmr_world_to_hope_table_transform_verified": False,
            "gmr_world_to_hope_matrix_4x4": None,
            "mirror_status": "unverified",
            "blockers": ["unit test keeps returnability blocked"],
        },
        "expected_asset_ids": EXPECTED_IDS,
        "inputs": [],
        "libraries": {
            "franco_two_blocks": [
                "franco_forehand_block",
                "franco_backhand_block",
            ],
            "franco_four_loop_a": [
                "franco_forehand_block",
                "franco_backhand_block",
                "franco_forehand_loop",
                "franco_backhand_loop_a",
            ],
        },
        "library_comparisons": [
            {"baseline": "franco_two_blocks", "candidate": "franco_four_loop_a"}
        ],
        "output_contract": {
            "result": str(tmp_path / "result.json"),
            "no_clobber": True,
        },
    }


def _write_manifest(tmp_path: Path, value: dict) -> tuple[Path, str]:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path, _sha(path)


def test_question_schedule_is_stable_balanced_and_bounded():
    schedule = _schedule(count_per_side=5)
    first = mod.build_questions(schedule)
    second = mod.build_questions(schedule)
    assert len(first) == 10
    assert [value.question_id for value in first] == [value.question_id for value in second]
    assert len({value.question_id for value in first}) == 10
    assert [value.side for value in first].count("forehand") == 5
    assert [value.side for value in first].count("backhand") == 5
    for question in first:
        assert -2.85 <= question.ball_vel_w[0] <= -0.82
        assert 0.37 <= question.ball_pos_w[0] <= 0.95
        assert 0.98 <= question.ball_pos_w[2] <= 1.26
        assert 0.0 <= np.linalg.norm(question.ball_spin_w) <= 34.0
        if question.side == "forehand":
            assert -0.35 <= question.ball_pos_w[1] < -0.155
        else:
            assert -0.155 <= question.ball_pos_w[1] <= 0.18


def test_question_schedule_seed_changes_atomic_ids():
    first = mod.build_questions(_schedule())
    changed = _schedule()
    changed["seed"] = "another-paper"
    second = mod.build_questions(changed)
    assert {value.question_id for value in first}.isdisjoint(
        {value.question_id for value in second}
    )


def test_slerp_shortest_arc_and_densify_endpoints():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    same_negated = -identity
    assert np.allclose(mod.slerp_xyzw(identity, same_negated, 0.5), identity)

    payload = {
        "fps": 30.0,
        "root_pos": np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]),
        "root_rot": np.stack([identity, identity]),
        "dof_pos": np.stack([np.zeros(31), np.ones(31)]),
    }
    dense, coordinate = mod.densify_payload(payload, 4)
    assert dense["root_pos"].shape == (5, 3)
    assert np.allclose(dense["root_pos"][0], payload["root_pos"][0])
    assert np.allclose(dense["root_pos"][-1], payload["root_pos"][-1])
    assert np.allclose(dense["dof_pos"][2], 0.5)
    assert np.allclose(coordinate, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert dense["fps"] == 120.0


def test_dense_danger_marks_both_interval_endpoints():
    coordinate = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    dangerous = np.zeros(9, dtype=bool)
    dangerous[2] = True
    assert mod.unsafe_source_mask(3, coordinate, dangerous).tolist() == [True, True, False]
    dangerous[:] = False
    dangerous[4] = True
    assert mod.unsafe_source_mask(3, coordinate, dangerous).tolist() == [False, True, False]


def test_wilson_lcb_domain_and_order():
    assert mod.wilson_lcb(0, 32) == 0.0
    assert 0.0 < mod.wilson_lcb(16, 32) < 0.5
    assert mod.wilson_lcb(30, 32) > mod.wilson_lcb(20, 32)
    with pytest.raises(mod.ScreenError):
        mod.wilson_lcb(33, 32)


def test_verified_rigid_frame_transform_moves_points_but_only_rotates_vectors():
    frame = {
        "returnability_enabled": True,
        "gmr_world_to_hope_matrix_4x4": [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    positions = np.array([[1.0, 2.0, 3.0]])
    normals = np.array([[1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 2.0, 0.0]])
    p, n, v = mod.apply_verified_frame_contract(positions, normals, velocities, frame)
    assert np.allclose(p, [[8.0, 21.0, 33.0]])
    assert np.allclose(n, [[0.0, 1.0, 0.0]])
    assert np.allclose(v, [[-2.0, 0.0, 0.0]])


def test_library_envelope_and_two_vs_four_delta():
    questions = mod.build_questions(_schedule(count_per_side=1))
    fh, bh = questions
    assets = [
        {
            "asset_id": "franco_forehand_block",
            "side": "forehand",
            "question_coverage": {},
        },
        {
            "asset_id": "franco_backhand_block",
            "side": "backhand",
            "question_coverage": {
                bh.question_id: {
                    "frame": 2,
                    "phase": 0.2,
                    "return_margin_m": 0.01,
                    "racket_body_clearance_m": 0.1,
                }
            },
        },
        {
            "asset_id": "franco_forehand_loop",
            "side": "forehand",
            "question_coverage": {
                fh.question_id: {
                    "frame": 3,
                    "phase": 0.3,
                    "return_margin_m": 0.02,
                    "racket_body_clearance_m": 0.2,
                }
            },
        },
        {
            "asset_id": "franco_backhand_loop_a",
            "side": "backhand",
            "question_coverage": {},
        },
    ]
    plan = {
        "libraries": {
            "franco_two_blocks": [
                "franco_forehand_block",
                "franco_backhand_block",
            ],
            "franco_four_loop_a": [
                "franco_forehand_block",
                "franco_backhand_block",
                "franco_forehand_loop",
                "franco_backhand_loop_a",
            ],
        },
        "library_comparisons": [
            {"baseline": "franco_two_blocks", "candidate": "franco_four_loop_a"}
        ],
    }
    libraries, comparisons = mod.aggregate_libraries(plan, assets, questions)
    assert libraries["franco_two_blocks"]["coverage_returned"] == 1
    assert libraries["franco_four_loop_a"]["coverage_returned"] == 2
    assert comparisons[0]["candidate_only_count"] == 1
    assert comparisons[0]["coverage_delta_count"] == 1


def test_blocked_prereg_validates_but_run_contract_rejects(tmp_path):
    manifest, expected = _write_manifest(tmp_path, _blocked_manifest(tmp_path))
    plan = mod.validate_manifest(
        manifest,
        expected,
        require_ready=False,
        verify_files=True,
    )
    assert plan["execution_ready"] is False
    with pytest.raises(mod.ScreenError, match="canonical grounding"):
        mod.validate_manifest(
            manifest,
            expected,
            require_ready=True,
            verify_files=True,
        )


def test_manifest_hash_and_tool_dependency_are_fail_closed(tmp_path):
    value = _blocked_manifest(tmp_path)
    manifest, expected = _write_manifest(tmp_path, value)
    with pytest.raises(mod.ScreenError, match="manifest sha256"):
        mod.validate_manifest(
            manifest,
            "0" * 64,
            require_ready=False,
            verify_files=True,
        )

    value["tool_contract"]["virtual_return_dependency"]["sha256"] = "f" * 64
    manifest, expected = _write_manifest(tmp_path, value)
    with pytest.raises(mod.ScreenError, match="local SHA mismatch"):
        mod.validate_manifest(
            manifest,
            expected,
            require_ready=False,
            verify_files=True,
        )


def test_blocked_prereg_cannot_guess_future_inputs(tmp_path):
    value = _blocked_manifest(tmp_path)
    value["inputs"] = [{}]
    manifest, expected = _write_manifest(tmp_path, value)
    with pytest.raises(mod.ScreenError, match="must not guess"):
        mod.validate_manifest(
            manifest,
            expected,
            require_ready=False,
            verify_files=True,
        )
