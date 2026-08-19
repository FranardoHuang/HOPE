"""Dependency-light parity for the shared FullMDP catalog source."""

from __future__ import annotations

import math
from pathlib import Path
import sys


MDP = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
)
if str(MDP) not in sys.path:
    sys.path.insert(0, str(MDP))

import action_ball_full_mdp_portable_catalog as catalog  # noqa: E402


def test_portable_catalog_preserves_legacy_columns_and_fresh_action_zero():
    legacy = catalog.load_action_ball_full_mdp_diagnostic_catalog_table()
    portable = catalog.load_portable_action_center_table()
    assert len(legacy.action_order) == 73
    assert len(portable.actions) == 73
    assert portable.manifest_file_sha256 == legacy.manifest_file_sha256
    assert portable.manifest_canonical_sha256 == legacy.manifest_canonical_sha256
    assert portable.fresh_action_slot == 0
    action = portable.fresh_action
    assert action.action_slot == 0
    assert action.action_id == legacy.action_order[0]
    assert action.action_uid == legacy.action_uids[0] == 6907688916670928
    assert action.motion_file == legacy.motion_files[0]
    assert action.motion_sha256 == legacy.motion_sha256[0]
    assert action.family == legacy.clip_family_per_clip[0] == "forehand"
    assert action.strike_phase == legacy.strike_phase_per_clip[0]
    assert action.mount_normal_sign == legacy.mount_normal_sign_per_clip[0]


def test_portable_center_rows_are_finite_and_keep_unique_identity():
    portable = catalog.load_portable_action_center_table()
    assert len({row.action_uid for row in portable.actions}) == 73
    assert tuple(row.action_slot for row in portable.actions) == tuple(range(73))
    for row in portable.actions:
        numeric = (
            *row.contact_offset_center_b_yaw_m,
            row.time_to_contact_center_s,
            *row.incoming_direction_center_b_yaw,
            row.incoming_speed_center_mps,
            *row.spin_direction_center_b_yaw,
            row.spin_magnitude_center_radps,
            *row.base_spawn_center_w_xy_m,
            *row.base_travel_center_b_yaw_xy_m,
            row.reference_t_hit_s,
            row.reference_t_cycle_s,
            *row.reference_racket_site_position_w_m,
            *row.reference_racket_quat_wxyz,
            *row.reference_racket_angular_velocity_w_radps,
            *row.reference_racket_site_velocity_w_mps,
            *row.reference_raw_face_normal_w,
            *row.reference_reach_offset_xy_m,
            *row.reference_base_root_quat_wxyz,
        )
        assert all(math.isfinite(value) for value in numeric)
        assert row.mount_normal_sign in (-1, 1)
        assert math.isclose(
            sum(value * value for value in row.reference_racket_quat_wxyz),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-5,
        )
        assert math.isclose(
            sum(value * value for value in row.reference_raw_face_normal_w),
            1.0,
            rel_tol=0.0,
            abs_tol=3.0e-6,
        )
    assert all(math.isfinite(value) for value in portable.landing_aim_center_w_xy_m)


def test_action_zero_reference_row_matches_frozen_cold_builder_values():
    action = catalog.load_portable_action_center_table().fresh_action
    expected = {
        "reference_racket_site_position_w_m": (
            0.6866140365600586,
            -0.5569199919700623,
            0.9527190327644348,
        ),
        "reference_racket_site_velocity_w_mps": (
            2.8326847553253174,
            1.0447025299072266,
            -1.9873850345611572,
        ),
        "reference_raw_face_normal_w": (
            0.8461580872535706,
            0.27271875739097595,
            0.45786359906196594,
        ),
        "reference_reach_offset_xy_m": (
            0.5465314984321594,
            -0.5838211178779602,
        ),
    }
    for name, values in expected.items():
        actual = getattr(action, name)
        assert len(actual) == len(values)
        assert all(
            math.isclose(got, want, rel_tol=0.0, abs_tol=2.0e-7)
            for got, want in zip(actual, values)
        )
