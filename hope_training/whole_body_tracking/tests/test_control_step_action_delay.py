"""Host-only tests for the vendor A3 per-episode control-step action delay.

Run:
    python -m pytest \
      hope_training/whole_body_tracking/tests/test_control_step_action_delay.py -q
"""

from __future__ import annotations

import types

import pytest
import torch

from test_reward_flags_mdp import hope_actions_mod as A


_JOINTS = 31


def _action(
    *,
    num_envs: int = 3,
    min_steps: int = 0,
    max_steps: int = 0,
    explicit_delay_fields: bool = True,
    clamp: bool | None = None,
    use_default_offset: bool = True,
):
    names = tuple(f"joint_{index}" for index in range(_JOINTS))
    default = torch.linspace(-0.15, 0.15, _JOINTS).repeat(num_envs, 1)
    soft = torch.stack(
        (
            torch.full((num_envs, _JOINTS), -2.0),
            torch.full((num_envs, _JOINTS), 2.0),
        ),
        dim=-1,
    )
    asset = types.SimpleNamespace(
        joint_names=names,
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=default,
            soft_joint_pos_limits=soft,
        ),
    )
    if clamp is None:
        clamp = max_steps > 0
    cfg_values = {
        "asset_name": "robot",
        "scale": 0.25,
        "use_default_offset": use_default_offset,
        "clamp": clamp,
    }
    if explicit_delay_fields:
        cfg_values.update(
            control_step_action_delay_min=min_steps,
            control_step_action_delay_max=max_steps,
        )
    cfg = types.SimpleNamespace(**cfg_values)
    env = types.SimpleNamespace(
        scene={"robot": asset},
        num_envs=num_envs,
        device="cpu",
    )
    term = A.ClampedJointPositionAction(cfg, env)
    manager = types.SimpleNamespace(
        _action=torch.zeros(num_envs, _JOINTS),
        _prev_action=torch.zeros(num_envs, _JOINTS),
    )
    manager.get_term = lambda name: term if name == "joint_pos" else None
    env.action_manager = manager
    return term, env


def test_zero_delay_is_bitwise_legacy_path_and_draws_no_rng():
    legacy, _ = _action(explicit_delay_fields=False)
    zero, _ = _action(min_steps=0, max_steps=0)
    rng_before = torch.get_rng_state().clone()
    legacy.reset()
    zero.reset()
    assert torch.equal(torch.get_rng_state(), rng_before)

    for step in range(4):
        current = torch.arange(_JOINTS, dtype=torch.float32).repeat(3, 1)
        current = current + float(step)
        legacy.process_actions(current)
        zero.process_actions(current)
        assert torch.equal(zero.raw_actions, legacy.raw_actions)
        assert torch.equal(zero.processed_actions, legacy.processed_actions)
        assert torch.equal(zero.prev_raw_actions, legacy.prev_raw_actions)
        assert torch.equal(
            zero.prev_prev_raw_actions, legacy.prev_prev_raw_actions
        )
        assert torch.equal(
            zero.raw_action_history_valid, legacy.raw_action_history_valid
        )
    assert zero.control_step_action_delay_enabled is False


def test_fixed_two_step_impulse_is_measured_in_policy_steps():
    action, _ = _action(num_envs=1, min_steps=2, max_steps=2)
    action.reset()
    default = action._asset.data.default_joint_pos.clone()
    impulse = torch.linspace(-1.0, 1.0, _JOINTS).unsqueeze(0)
    zeros = torch.zeros_like(impulse)

    action.process_actions(impulse)
    assert torch.equal(action.processed_actions, default)
    action.process_actions(zeros)
    assert torch.equal(action.processed_actions, default)
    action.process_actions(zeros)
    assert torch.equal(action.processed_actions, default + 0.25 * impulse)
    # Raw-action regularization remains on actor output, not the delayed command.
    assert torch.equal(action.raw_actions, zeros)
    assert torch.equal(action.prev_raw_actions, zeros)
    assert torch.equal(action.prev_prev_raw_actions, impulse)


def test_lag_is_episode_fixed_partial_reset_only_mutates_selected_rows():
    torch.manual_seed(20260731)
    action, _ = _action(num_envs=8, min_steps=0, max_steps=2)
    action.reset()
    initial_lag = action.control_step_action_delay_lag_steps.clone()
    assert bool(torch.all((initial_lag >= 0) & (initial_lag <= 2)))
    action.process_actions(torch.ones(8, _JOINTS))
    assert torch.equal(action.control_step_action_delay_lag_steps, initial_lag)

    other_ids = torch.tensor([0, 1, 3, 4, 5, 6, 7])
    other_history = action._policy_action_delay._history[other_ids].clone()
    action.reset(torch.tensor([2]))
    assert torch.equal(
        action.control_step_action_delay_lag_steps[other_ids],
        initial_lag[other_ids],
    )
    assert torch.equal(
        action._policy_action_delay._history[other_ids], other_history
    )
    assert torch.count_nonzero(
        action._policy_action_delay._history[2]
    ).item() == 0


def test_one_scalar_lag_governs_the_complete_31_joint_row():
    action, _ = _action(num_envs=3, min_steps=0, max_steps=2)
    action.reset()
    action._policy_action_delay._lag_steps.copy_(torch.tensor([0, 1, 2]))
    previous = torch.stack(
        (
            torch.linspace(0.10, 0.20, _JOINTS),
            torch.linspace(0.20, 0.30, _JOINTS),
            torch.linspace(0.30, 0.40, _JOINTS),
        )
    )
    older = previous + 0.40
    action._policy_action_delay._history[:, 0, :].copy_(previous)
    action._policy_action_delay._history[:, 1, :].copy_(older)
    current = previous + 0.80
    action.process_actions(current)

    expected_raw_due = torch.stack((current[0], previous[1], older[2]))
    expected_qdes = (
        action._asset.data.default_joint_pos + 0.25 * expected_raw_due
    )
    assert torch.equal(action.processed_actions, expected_qdes)
    assert action.control_step_action_delay_lag_steps.shape == (3,)
    assert action.control_step_action_delay_contract()[
        "shared_across_all_31_joints"
    ] is True


def test_dynamic_ready_hold_fills_every_history_age_without_redraw():
    action, _ = _action(
        num_envs=2, min_steps=0, max_steps=2, clamp=True
    )
    action.reset()
    lag_before = action.control_step_action_delay_lag_steps.clone()
    ids = torch.tensor([1])
    normalized_hold = torch.linspace(-0.5, 0.5, _JOINTS).unsqueeze(0)
    physical_hold = (
        action._asset.data.default_joint_pos[ids] + 0.25 * normalized_hold
    )
    action.install_action_ball_dynamic_ready_state(
        ids, normalized_hold, physical_hold
    )
    assert torch.equal(action.control_step_action_delay_lag_steps, lag_before)
    expected = normalized_hold[:, None, :].expand(1, 2, _JOINTS)
    assert torch.equal(action._policy_action_delay._history[ids], expected)


def test_disabled_dynamic_ready_snapshot_keeps_legacy_keys_and_skips_delay_clone():
    action, _ = _action(
        num_envs=2, min_steps=0, max_steps=0, clamp=True
    )

    def forbidden(_ids):
        raise AssertionError("disabled delay snapshot cloned state")

    action._policy_action_delay.snapshot_rows = forbidden
    state = action.snapshot_action_ball_dynamic_ready_state(torch.tensor([0]))
    assert "policy_action_delay" not in state
    assert set(state) == {
        "manager_action",
        "manager_prev_action",
        "raw_actions",
        "processed_actions",
        "previous_processed_qdes",
        "pre_clamp_qdes",
        "nominal_projected_qdes",
        "nominal_projection_span",
        "prev_raw_actions",
        "prev_prev_raw_actions",
        "processed_qdes_valid",
        "previous_processed_qdes_valid",
        "pre_clamp_qdes_valid",
        "nominal_projected_qdes_valid",
        "raw_actions_valid",
        "prev_raw_actions_valid",
        "prev_prev_raw_actions_valid",
    }


def test_exact_resume_round_trip_reproduces_future_delayed_qdes():
    source, _ = _action(num_envs=3, min_steps=0, max_steps=2)
    resumed, _ = _action(num_envs=3, min_steps=0, max_steps=2)
    source.reset()
    resumed.reset()
    source._policy_action_delay._lag_steps.copy_(torch.tensor([0, 1, 2]))
    source.process_actions(torch.full((3, _JOINTS), 0.2))
    source.process_actions(torch.full((3, _JOINTS), -0.4))
    state = source.action_delay_exact_resume_state_dict()
    assert state["lag_steps"].device.type == "cpu"
    assert state["history"].device.type == "cpu"
    resumed.load_action_delay_exact_resume_state_dict(state, strict=True)

    for value in (0.7, -0.1, 0.3):
        future = torch.full((3, _JOINTS), value)
        source.process_actions(future)
        resumed.process_actions(future)
        assert torch.equal(resumed.processed_actions, source.processed_actions)
        assert torch.equal(
            resumed.control_step_action_delay_lag_steps,
            source.control_step_action_delay_lag_steps,
        )
        assert torch.equal(
            resumed._policy_action_delay._history,
            source._policy_action_delay._history,
        )


def test_enabled_delay_rejects_non_31d_non_default_offset_and_unclamped_qdes():
    with pytest.raises(ValueError, match="31-D"):
        A._EpisodeSharedPolicyActionDelay(
            num_envs=1,
            action_dim=30,
            min_steps=0,
            max_steps=2,
            device="cpu",
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="use_default_offset=True"):
        _action(
            min_steps=0,
            max_steps=2,
            use_default_offset=False,
        )
    with pytest.raises(ValueError, match="q_des clamp"):
        _action(
            min_steps=0,
            max_steps=2,
            clamp=False,
        )


def test_runtime_receipt_exposes_control_step_semantics_and_histogram():
    action, _ = _action(num_envs=3, min_steps=0, max_steps=2)
    action.reset()
    action._policy_action_delay._lag_steps.copy_(torch.tensor([0, 1, 2]))
    receipt = action.control_step_action_delay_runtime_receipt()
    assert receipt["contract"]["semantic_unit"] == "policy_control_step"
    assert receipt["contract"]["sample_timing"] == "once_per_episode_reset"
    assert receipt["initialized_env_count"] == 3
    assert receipt["lag_histogram"] == {"0": 1, "1": 1, "2": 1}


def test_zero_delay_receipt_is_explicit_shared_and_reset_accounted():
    action, _ = _action(num_envs=3, min_steps=0, max_steps=0)
    rng_before = torch.get_rng_state().clone()
    action.reset()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert not bool(action._policy_action_delay.episode_initialized.any())

    receipt = action.control_step_action_delay_runtime_receipt()
    assert receipt["contract"] == {
        "schema_version": 1,
        "enabled": False,
        "semantic_unit": "policy_control_step",
        "sample_timing": "once_per_episode_reset",
        "distribution": "discrete_uniform_inclusive",
        "min_steps": 0,
        "max_steps": 0,
        "shared_across_all_31_joints": True,
        "history_fill": "safe_default_or_action_specific_hold",
    }
    assert receipt["initialized_env_count"] == 3
    assert receipt["lag_histogram"] == {"0": 3}
