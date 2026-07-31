"""Host-only contract tests for the ActionBall four-substep table-contact latch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test_reward_flags_mdp import hope_actions_mod, terminations_mod


def _latch(num_envs: int = 2):
    return hope_actions_mod._PhysicsSubstepTableContactLatch(
        num_envs=num_envs,
        expected_apply_calls=4,
        device="cpu",
    )


def _run_policy_step(latch, pulse_substep: int | None):
    """Model apply0(skip), apply1..3(read substeps1..3), finalize(read substep4)."""

    latch.begin_policy_step()
    latch.record_apply(None)
    for substep in range(3):
        hit = torch.tensor(
            [substep == pulse_substep, False], dtype=torch.bool
        )
        latch.record_apply(hit)
    final = torch.tensor(
        [pulse_substep == 3, False], dtype=torch.bool
    )
    return latch.finalize(final)


def _clear_initial_reset_quarantine(latch):
    """Advance one clean substep so ordinary pulse tests model a live episode."""

    latch.begin_policy_step()
    latch.record_apply(None)
    latch.record_apply(torch.tensor([False, False]))
    latch.record_apply(torch.tensor([False, False]))
    latch.record_apply(torch.tensor([False, False]))
    latch.finalize(torch.tensor([False, False]))


@pytest.mark.parametrize("pulse_substep", range(4))
def test_each_of_four_physics_substep_pulses_is_sticky(pulse_substep):
    latch = _latch()
    _clear_initial_reset_quarantine(latch)
    assert _run_policy_step(latch, pulse_substep).tolist() == [True, False]


def test_first_apply_skips_previous_control_step_sensor_state():
    latch = _latch()
    latch.begin_policy_step()
    with pytest.raises(RuntimeError, match="first.*skip"):
        latch.record_apply(torch.tensor([True, False]))
    # The rejected stale sample was not incorporated.
    assert latch.hit.tolist() == [False, False]


def test_finalize_is_exact_counted_and_idempotent():
    latch = _latch()
    latch.begin_policy_step()
    latch.record_apply(None)
    with pytest.raises(RuntimeError, match="exactly 4 apply"):
        latch.finalize(torch.tensor([False, False]))

    latch = _latch()
    _clear_initial_reset_quarantine(latch)
    first = _run_policy_step(latch, 3)
    second = latch.finalize(torch.tensor([False, True]))
    assert first.data_ptr() == second.data_ptr()
    assert second.tolist() == [True, False]


def test_only_reset_clears_selected_episode_rows_and_no_cross_env_leak():
    latch = _latch()
    _clear_initial_reset_quarantine(latch)
    assert _run_policy_step(latch, 3).tolist() == [True, False]

    # Starting another policy step does not erase episode-sticky evidence.
    assert _run_policy_step(latch, None).tolist() == [True, False]
    latch.reset_envs(torch.tensor([0]))
    assert latch.hit.tolist() == [False, False]

    # A hit in env 1 remains isolated from env 0 and a partial reset clears only env 1.
    latch.begin_policy_step()
    latch.record_apply(None)
    latch.record_apply(torch.tensor([False, True]))
    latch.record_apply(torch.tensor([False, False]))
    latch.record_apply(torch.tensor([False, False]))
    assert latch.finalize(torch.tensor([False, False])).tolist() == [
        False,
        True,
    ]
    latch.reset_envs(torch.tensor([1]))
    assert latch.hit.tolist() == [False, False]


def test_first_post_reset_substep_is_quarantined_but_persistent_contact_is_not():
    latch = _latch()
    _clear_initial_reset_quarantine(latch)

    # First create a real table terminal so only that reset arms quarantine.
    assert _run_policy_step(latch, 3).tolist() == [True, False]
    latch.reset_envs(torch.tensor([0]))

    # A one-substep force carried over from that terminal pose is ignored.
    latch.begin_policy_step()
    latch.record_apply(None)
    latch.record_apply(torch.tensor([True, False]))
    latch.record_apply(torch.tensor([False, False]))
    latch.record_apply(torch.tensor([False, False]))
    assert latch.finalize(torch.tensor([False, False])).tolist() == [
        False,
        False,
    ]

    # After another reset, a contact that remains for a second substep is real
    # and must still terminate the new episode.
    assert _run_policy_step(latch, 3).tolist() == [True, False]
    latch.reset_envs(torch.tensor([0]))
    latch.begin_policy_step()
    latch.record_apply(None)
    latch.record_apply(torch.tensor([True, False]))
    latch.record_apply(torch.tensor([True, False]))
    latch.record_apply(torch.tensor([False, False]))
    assert latch.finalize(torch.tensor([False, False])).tolist() == [
        True,
        False,
    ]


def test_post_reset_quarantine_is_per_environment():
    latch = _latch()
    _clear_initial_reset_quarantine(latch)
    assert _run_policy_step(latch, 3).tolist() == [True, False]
    latch.reset_envs(torch.tensor([0]))

    latch.begin_policy_step()
    latch.record_apply(None)
    latch.record_apply(torch.tensor([True, True]))
    latch.record_apply(torch.tensor([False, False]))
    latch.record_apply(torch.tensor([False, False]))
    assert latch.finalize(torch.tensor([False, False])).tolist() == [
        False,
        True,
    ]


def test_non_table_reset_does_not_quarantine_first_substep():
    latch = _latch()
    latch.reset_envs(torch.tensor([0]))

    latch.begin_policy_step()
    latch.record_apply(None)
    latch.record_apply(torch.tensor([True, False]))
    latch.record_apply(torch.tensor([False, False]))
    latch.record_apply(torch.tensor([False, False]))
    assert latch.finalize(torch.tensor([False, False])).tolist() == [
        True,
        False,
    ]


def test_single_substep_latch_is_rejected():
    with pytest.raises(ValueError, match="at least two physics substeps"):
        hope_actions_mod._PhysicsSubstepTableContactLatch(
            num_envs=1,
            expected_apply_calls=1,
            device="cpu",
        )


def test_unfinalized_policy_step_cannot_be_overwritten():
    latch = _latch()
    latch.begin_policy_step()
    latch.record_apply(None)
    with pytest.raises(RuntimeError, match="not finalized"):
        latch.begin_policy_step()
    with pytest.raises(RuntimeError, match="cannot discard"):
        latch.reset_envs(torch.tensor([0]))


def test_float32_timestamp_accumulation_accepts_2000_real_substeps():
    is_consecutive = hope_actions_mod._consecutive_physics_timestamp_mask
    timestamp = torch.zeros(4, dtype=torch.float32)
    for _ in range(2000):
        next_timestamp = timestamp + 0.005
        assert is_consecutive(next_timestamp, timestamp, 0.005).all()
        timestamp = next_timestamp


@pytest.mark.parametrize("bad_delta", [0.0, 0.01])
def test_timestamp_guard_rejects_repeated_or_skipped_substep(bad_delta):
    is_consecutive = hope_actions_mod._consecutive_physics_timestamp_mask
    previous = torch.full((4,), 9.995, dtype=torch.float32)
    current = previous + bad_delta
    assert not is_consecutive(current, previous, 0.005).any()


def _clock_sensor(timestamp):
    value = torch.full((2,), timestamp, dtype=torch.float32)
    return SimpleNamespace(
        cfg=SimpleNamespace(update_period=0.0),
        _timestamp=value,
        _timestamp_last_update=value.clone(),
    )


def _clock_reader(sensor_times):
    sensors = {
        name: _clock_sensor(timestamp)
        for name, timestamp in sensor_times.items()
    }

    def assert_device(condition, message):
        if not bool(condition):
            raise RuntimeError(message)

    return SimpleNamespace(
        num_envs=2,
        _processed_actions=torch.zeros(2, 1),
        _table_contact_timestamp_sensors=None,
        _table_contact_timestamp_data_contract_validated=False,
        _safety_env=SimpleNamespace(
            scene=SimpleNamespace(sensors=sensors)
        ),
        _assert_table_contact_device=assert_device,
    )


def test_full_assembly_owns_no_contact_sensor_clock():
    method = (
        hope_actions_mod.ClampedJointPositionAction
        ._table_contact_sensor_timestamps
    )
    whole_body_cfg = SimpleNamespace(name="contact_forces")
    params = {
        "full_table_assembly": True,
        "full_table_filtered_sensor_cfgs": (),
        "sensor_cfg": whole_body_cfg,
        # Deliberately absent: full assembly must not read any pair-filtered clock.
        "filtered_sensor_cfg": SimpleNamespace(name="racket_table_contact"),
    }
    reader = _clock_reader({})
    with pytest.raises(RuntimeError, match="owns no ContactSensor clock"):
        method(reader, params, require_data_fresh=True)


def test_legacy_two_clock_path_keeps_private_baseline_and_freshness():
    method = (
        hope_actions_mod.ClampedJointPositionAction
        ._table_contact_sensor_timestamps
    )
    raw_cfg = SimpleNamespace(name="contact_forces")
    filtered_cfg = SimpleNamespace(name="racket_table_contact")
    params = {
        "full_table_assembly": False,
        "sensor_cfg": raw_cfg,
        "filtered_sensor_cfg": filtered_cfg,
    }
    reader = _clock_reader({raw_cfg.name: 2.0, filtered_cfg.name: 2.0})
    raw_sensor = reader._safety_env.scene.sensors[raw_cfg.name]
    filtered_sensor = reader._safety_env.scene.sensors[filtered_cfg.name]
    baseline = method(reader, params, require_data_fresh=False)
    assert baseline.data_ptr() != raw_sensor._timestamp.data_ptr()
    raw_sensor._timestamp.add_(0.005)
    filtered_sensor._timestamp.add_(0.005)
    raw_sensor._timestamp_last_update.copy_(raw_sensor._timestamp)
    filtered_sensor._timestamp_last_update.copy_(
        filtered_sensor._timestamp
    )
    assert baseline.tolist() == [2.0, 2.0]

    current = method(reader, params, require_data_fresh=True)
    assert current.data_ptr() != raw_sensor._timestamp.data_ptr()
    assert hope_actions_mod._consecutive_physics_timestamp_mask(
        current, baseline, 0.005
    ).all()


def test_full_assembly_sample_forwards_exact_body_and_proxy_contract(monkeypatch):
    body_names = tuple(f"a3_body_{index}" for index in range(32))
    captured = {}

    def sample_stub(_env, **kwargs):
        captured.update(kwargs)
        return torch.zeros(2, dtype=torch.bool)

    monkeypatch.setattr(
        terminations_mod,
        "sample_robot_table_contact_current",
        sample_stub,
    )
    params = {
        "sensor_cfg": SimpleNamespace(name="contact_forces"),
        "filtered_sensor_cfg": SimpleNamespace(name="contact_forces"),
        "full_table_filtered_sensor_cfgs": (),
        "expected_full_table_source_prim_paths": tuple(
            f"{{ENV_REGEX_NS}}/TablePart{index}" for index in range(5)
        ),
        "expected_full_robot_body_names": body_names,
        "asset_cfg": SimpleNamespace(name="robot"),
        "near_x": 0.5,
        "surface_z": 0.76,
        "full_table_assembly": True,
        "body_proxy_radius_m": 0.18,
        "foot_proxy_radius_m": 0.10,
        "wrist_proxy_radius_m": 0.08,
        "foot_body_names": ("left_foot", "right_foot"),
        "racket_body_name": "right_wrist_yaw_Link",
        "racket_blade_center_offset_wrist_m": (0.206194, 0.025474, 0.028020),
        "racket_blade_half_extents_m": (0.082, 0.008, 0.082),
    }

    def forbid_sensor_clock(*_args, **_kwargs):
        raise AssertionError("full pose guard touched a sensor clock")

    reader = SimpleNamespace(
        _safety_env=object(),
        _resolved_table_contact_params=lambda: params,
        _table_contact_sensor_timestamps=forbid_sensor_clock,
        _table_contact_last_sensor_timestamp=None,
        _table_contact_guard_physics_dt_s=None,
    )
    method = (
        hope_actions_mod.ClampedJointPositionAction
        ._sample_table_contact_current
    )
    assert method(reader).tolist() == [False, False]
    assert captured["expected_full_robot_body_names"] == body_names
    assert captured["full_table_filtered_sensor_cfgs"] == ()
    assert captured["racket_blade_half_extents_m"] == (0.082, 0.008, 0.082)
