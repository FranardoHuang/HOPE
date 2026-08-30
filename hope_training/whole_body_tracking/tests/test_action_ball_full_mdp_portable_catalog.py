"""Dependency-light parity for the shared FullMDP catalog source."""

from __future__ import annotations

from dataclasses import replace
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


def test_runtime_math_pin_excludes_broad_racket_telemetry_module():
    assert catalog.PINNED_SOLVER_MATH_MODULE_NAMES == (
        "continuous_questions.py",
        "stroke_adapt_torch.py",
        "virtual_ball.py",
    )
    assert "hope_commands.py" not in catalog.PINNED_SOLVER_MATH_MODULE_NAMES


def test_portable_fresh_cadence_freezes_due_ticks_not_verdicts():
    table = catalog.load_portable_action_center_table()
    cadence = catalog.derive_portable_fresh_cadence(table)
    assert cadence.first_reveal_tick == 48
    assert cadence.maximum_task_close_ticks == 309
    assert cadence.cadence_ticks == 388
    assert cadence.reference_due_ticks == (48,)
    assert len(cadence.reference_due_ticks) == catalog.FRESH_REFERENCE_DUE_COUNT
    assert cadence.reference_due_ticks[-1] + cadence.cadence_ticks == 436
    assert (
        cadence.reference_due_ticks[-1] + cadence.cadence_ticks
        < cadence.episode_horizon_ticks
    )
    assert cadence.episode_horizon_ticks == 500
    assert not hasattr(cadence, "minimum_shot_reveal_ticks")

    drifted = replace(
        table,
        actions=tuple(
            replace(row, teacher_rate_min=row.teacher_rate_min * 2.0)
            for row in table.actions
        ),
    )
    try:
        catalog.derive_portable_fresh_cadence(drifted)
    except ValueError as exc:
        assert "frozen schedule" in str(exc)
    else:
        raise AssertionError("timing drift did not invalidate the frozen cadence")


def test_one_shot_reveal_launch_contact_close_fit_exact_n1_episode():
    table = catalog.load_portable_action_center_table()
    action = table.fresh_action
    cadence = catalog.derive_portable_fresh_cadence(table)
    reveal_tick = cadence.first_reveal_tick
    commit_and_full_horizon_launch_tick = reveal_tick + catalog.FRESH_HIDDEN_GAP_TICKS
    contact_delta_ticks = round(
        action.time_to_contact_center_s / catalog.FRESH_POLICY_STEP_S
    )
    contact_tick = commit_and_full_horizon_launch_tick + contact_delta_ticks
    close_tick = (
        commit_and_full_horizon_launch_tick + cadence.maximum_task_close_ticks
    )
    retirement_tick = reveal_tick + cadence.cadence_ticks

    assert contact_delta_ticks == 265
    assert (
        reveal_tick,
        commit_and_full_horizon_launch_tick,
        contact_tick,
        close_tick,
        retirement_tick,
    ) == (48, 50, 315, 359, 436)
    assert contact_tick < close_tick < retirement_tick < cadence.episode_horizon_ticks


def test_portable_catalog_preserves_legacy_columns_and_fresh_action_zero():
    legacy = catalog.load_action_ball_full_mdp_diagnostic_catalog_table()
    portable = catalog.load_portable_action_center_table()
    assert len(legacy.action_order) == 1
    assert len(portable.actions) == 1
    assert portable.manifest_file_sha256 == legacy.manifest_file_sha256
    assert portable.manifest_canonical_sha256 == legacy.manifest_canonical_sha256
    assert portable.fresh_action_slot == 0
    action = portable.fresh_action
    assert action.action_slot == 0
    assert action.action_id == legacy.action_order[0]
    assert action.action_uid == legacy.action_uids[0] == 4098890508575574
    assert action.motion_file == legacy.motion_files[0]
    assert action.motion_sha256 == legacy.motion_sha256[0]
    assert action.family == legacy.clip_family_per_clip[0] == "backhand"
    assert action.strike_phase == legacy.strike_phase_per_clip[0]
    assert action.mount_normal_sign == legacy.mount_normal_sign_per_clip[0]


def test_portable_center_rows_are_finite_and_keep_unique_identity():
    portable = catalog.load_portable_action_center_table()
    assert len({row.action_uid for row in portable.actions}) == 1
    assert tuple(row.action_slot for row in portable.actions) == (0,)
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
            0.727763295173645,
            -0.24545995891094208,
            1.0585051774978638,
        ),
        "reference_racket_site_velocity_w_mps": (
            0.5448840856552124,
            -0.009558722376823425,
            0.1934632658958435,
        ),
        "reference_raw_face_normal_w": (
            0.9005173444747925,
            -0.4245518445968628,
            0.09392757713794708,
        ),
        "reference_reach_offset_xy_m": (
            0.7295531630516052,
            -0.24445714056491852,
        ),
    }
    for name, values in expected.items():
        actual = getattr(action, name)
        assert len(actual) == len(values)
        assert all(
            math.isclose(got, want, rel_tol=0.0, abs_tol=2.0e-7)
            for got, want in zip(actual, values)
        )
