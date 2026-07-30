"""Host-only contract tests for the ActionBall four-substep table-contact latch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from test_reward_flags_mdp import hope_actions_mod


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
    assert _run_policy_step(latch, 1).tolist() == [True, False]

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

    # A one-substep force carried over from the terminal pose is ignored.
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
        _safety_env=SimpleNamespace(
            scene=SimpleNamespace(sensors=sensors)
        ),
        _assert_table_contact_device=assert_device,
    )


def test_full_assembly_freshness_checks_every_exact_pair_sensor_clock():
    method = (
        hope_actions_mod.ClampedJointPositionAction
        ._table_contact_sensor_timestamps
    )
    cfgs = tuple(
        SimpleNamespace(name=name)
        for name in ("pair_pelvis", "pair_elbow", "racket_table_contact")
    )
    params = {
        "full_table_assembly": True,
        "all_body_filtered_sensor_cfgs": cfgs,
        # Deliberately absent from the scene: full assembly must not read the legacy broad clock.
        "sensor_cfg": SimpleNamespace(name="legacy_broad"),
        "filtered_sensor_cfg": cfgs[-1],
    }
    reader = _clock_reader({cfg.name: 1.25 for cfg in cfgs})
    got = method(reader, params, require_data_fresh=True)
    assert got.tolist() == [1.25, 1.25]

    reader = _clock_reader(
        {
            cfgs[0].name: 1.25,
            cfgs[1].name: 1.255,
            cfgs[2].name: 1.25,
        }
    )
    with pytest.raises(RuntimeError, match="different physics frames"):
        method(reader, params, require_data_fresh=True)
