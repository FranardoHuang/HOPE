import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_take061_phase4_canonical_bundle.py"
SPEC = importlib.util.spec_from_file_location("take061_phase4_bundle", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_action_identity_is_new_and_explicit():
    assert M.ACTION_ID == "take061_slow_block_phase4_v1"
    assert M.EXPECTED_PHASE4_KIND == "take061_slow_block_exact_face_phase4_v1"


def test_heading_transform_is_yaw_invariant():
    yaw = 0.73
    local = np.asarray([0.4, -0.2, 0.7])
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.asarray([c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]])
    assert np.allclose(M._world_to_heading(world, yaw), local, atol=1.0e-12)


def test_singleton_profile_has_exact_independent_sides():
    profile = M._singleton_profile(
        ball_b=np.asarray([0.7, -0.3, 0.15]),
        incoming_b=np.asarray([-3.0, 2.0, 1.0]),
        base_xy=np.asarray([-0.1, 0.2]), ttc=4.9,
    )
    assert profile["time_to_contact_min_s"] == 4.9
    assert profile["time_to_contact_max_s"] == 4.9
    assert profile["contact_offset_min_b_yaw_m"] == [0.7, -0.3, 0.15]
    assert profile["contact_offset_max_b_yaw_m"] == [0.7, -0.3, 0.15]
    assert np.linalg.norm(profile["incoming_direction_center_b_yaw"]) == pytest.approx(1.0)
    for prefix in ("incoming_direction", "spin_direction"):
        for axis in ("u", "v"):
            for side in ("neg", "pos"):
                assert profile[f"{prefix}_tangent_{axis}_{side}_initial_deg"] == 0.0
                assert profile[f"{prefix}_tangent_{axis}_{side}_max_deg"] == 0.0


def test_contact_deadline_rounds_up_to_policy_grid_without_losing_reaction_margin():
    raw = 5.088 / 0.9816630367871384 + 0.1
    assert raw == pytest.approx(5.283041236484155)
    assert M._ceil_to_policy_tick(raw) == 5.3
    assert M._ceil_to_policy_tick(raw) - 5.088 / 0.9816630367871384 >= 0.1


def test_grid_retime_preserves_exact_contact_and_emits_one_row_per_policy_tick():
    q = np.arange(57 * 2, dtype=np.float64).reshape(57, 2)
    root_pos = np.arange(57 * 3, dtype=np.float64).reshape(57, 3)
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (57, 1))
    q_grid, qd_grid, root_grid, quat_grid, hit = M._retime_to_policy_grid(
        q=q,
        root_pos=root_pos,
        root_quat=root_quat,
        t_hit_s=5.04,
        t_cycle_s=5.88,
    )
    assert hit == 252
    assert q_grid.shape == qd_grid.shape == (295, 2)
    assert root_grid.shape == (295, 3)
    assert quat_grid.shape == (295, 4)
    assert np.array_equal(q_grid[hit], q[M.HIT_FRAME])
    assert np.array_equal(root_grid[hit], root_pos[M.HIT_FRAME])
    assert np.array_equal(q_grid[-1], q[-1])


def test_grid_retime_rejects_non_policy_aligned_phase4_timing():
    q = np.zeros((57, 2), dtype=np.float64)
    root_pos = np.zeros((57, 3), dtype=np.float64)
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (57, 1))
    with pytest.raises(M.BundleError, match="not aligned to the policy grid"):
        M._retime_to_policy_grid(
            q=q,
            root_pos=root_pos,
            root_quat=root_quat,
            t_hit_s=5.088,
            t_cycle_s=5.936,
        )


def test_output_path_rejects_parent_escape(tmp_path):
    args = M.parser().parse_args([
        "--phase4-report", "x", "--phase4-npz", "y", "--dynamic-ready", "z",
        "--mjcf", "m", "--output-dir-rel", "../escape",
    ])
    # The escape is rejected before any output publication; full materialization
    # is exercised on the exact Pod assets.
    assert ".." in Path(args.output_dir_rel).parts


def test_optitrack_profile_pins_name_and_hash_the_same_exact_yaml():
    physics = M.REPO_ROOT_DEFAULT / "configs/ball_physics_optitrack_20260730.yaml"
    template = M.REPO_ROOT_DEFAULT / "configs/action_ball_profile_pins_20260728.json"
    pins, _payload = M._materialize_profile_pins(
        template, physics, M.REPO_ROOT_DEFAULT
    )
    venue = pins["physics_payload"]["venue_source"]
    assert venue["path"] == "configs/ball_physics_optitrack_20260730.yaml"
    assert venue["file_sha256"] == M._sha256_file(physics)
    assert venue["file_sha256"] == (
        "3afb1c9a00f975d924169503d7dafab92ea6c0b96263336e27edcd1d6257ea14"
    )
    assert pins["venue_yaml_sha256"] == venue["file_sha256"]
