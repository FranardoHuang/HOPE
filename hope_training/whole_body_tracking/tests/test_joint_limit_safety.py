"""Fail-closed joint-position safety primitives.

These tests are host-only.  ``test_reward_flags_mdp`` supplies the lightweight Isaac Lab stubs
already used by the rest of the MDP unit suite.

The Pod 1-env Isaac smoke must additionally attest the real manager order, not merely repeat these
fakes: one policy step at ``step_dt=0.02`` has exactly four ``apply_actions`` records at fresh
articulation timestamps ``t + [0, .005, .010, .015]`` and one DoneTerm post-step readback at
``t + .020``; q/qdot must correspond to those timestamps, ``apply_call_count==decimation==4``,
``post_readback_count==1``, and the snapshot must be complete before the next policy step begins.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
import types

import pytest
import torch

from test_reward_flags_mdp import hope_actions_mod, terminations_mod
from test_training_launch_claim import _load_contract_module, _load_runner_module


def _limits(num_envs: int, joint_count: int, lo: float = -1.0, hi: float = 1.0):
    lower = torch.full((num_envs, joint_count), lo)
    upper = torch.full((num_envs, joint_count), hi)
    return torch.stack((lower, upper), dim=-1)


def _action_and_env(
    *,
    num_envs: int = 2,
    joint_count: int = 2,
    scale: float = 1.0,
    offset: torch.Tensor | None = None,
    clamp: bool = True,
    guard: bool = False,
    guard_policy_dt_s: float | None = 0.1,
    guard_margin_rad: float | None = 0.0,
    guard_margin_fraction: float | None = 0.0,
    runtime_step_dt: float = 0.1,
    decimation: int = 4,
    runtime_physics_dt: float | None = None,
    expected_decimation: int | None = 4,
    terminal_archive_capacity: int | None = 16,
):
    names = [f"j{index}" for index in range(joint_count)]
    offset = (
        torch.zeros(num_envs, joint_count)
        if offset is None
        else offset.clone()
    )
    limits = _limits(num_envs, joint_count)
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_names=names,
            default_joint_pos=offset,
            soft_joint_pos_limits=limits,
            # Keep a visible hard-vs-soft distinction: q_des clamps at +/-1.0, while terminal
            # crossing guards use the physical +/-1.2 envelope.
            joint_pos_limits=_limits(
                num_envs, joint_count, lo=-1.2, hi=1.2
            ),
            joint_pos=torch.zeros(num_envs, joint_count),
            joint_vel=torch.zeros(num_envs, joint_count),
            _sim_timestamp=0.0,
            _joint_pos=types.SimpleNamespace(timestamp=0.0),
            _joint_vel=types.SimpleNamespace(timestamp=0.0),
        )
    )
    cfg = types.SimpleNamespace(
        asset_name="robot",
        scale=scale,
        use_default_offset=True,
        clamp=clamp,
        pre_apply_limit_guard=guard,
        pre_apply_guard_policy_dt_s=guard_policy_dt_s,
        pre_apply_guard_margin_rad=guard_margin_rad,
        pre_apply_guard_margin_fraction=guard_margin_fraction,
        pre_apply_guard_expected_decimation=expected_decimation,
        pre_apply_guard_terminal_archive_capacity=terminal_archive_capacity,
    )
    env = types.SimpleNamespace(
        scene={"robot": asset},
        num_envs=num_envs,
        device="cpu",
        step_dt=runtime_step_dt,
        physics_dt=(
            runtime_step_dt / decimation
            if runtime_physics_dt is None
            else runtime_physics_dt
        ),
        cfg=types.SimpleNamespace(decimation=decimation),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    )
    action = hope_actions_mod.ClampedJointPositionAction(cfg, env)
    env.action_manager = types.SimpleNamespace(
        get_term=lambda name: action if name == "joint_pos" else None
    )
    return action, env, asset


def _set_sim_timestamp(asset, timestamp_s: float) -> None:
    asset.data._sim_timestamp = timestamp_s
    asset.data._joint_pos.timestamp = timestamp_s
    asset.data._joint_vel.timestamp = timestamp_s


def _finish_guarded_policy_step(action, asset):
    """Model the pinned Isaac order: 4 apply writes, then one post-step DoneTerm readback."""

    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    start = float(asset.data._sim_timestamp)
    physics_dt = float(action._pre_apply_guard_physics_dt_s)
    decimation = int(action._pre_apply_guard_decimation)
    for index in range(decimation):
        _set_sim_timestamp(asset, start + index * physics_dt)
        action.apply_actions()
    _set_sim_timestamp(asset, start + decimation * physics_dt)
    action.finalize_joint_safety_post_step_readback()


def _two_step_cross_reset_action_ball_ledger():
    action, env, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    receipts = ["a" * 64, "b" * 64]
    action_uids = torch.tensor([101, 202], dtype=torch.long)
    birth_generations = torch.tensor([7, 9], dtype=torch.long)
    swing_generations = torch.tensor([3, 4], dtype=torch.long)
    command = types.SimpleNamespace(
        action_ball_enabled=True,
        action_ball_episode_generation=birth_generations,
        action_ball_reset_generation=birth_generations,
        action_ball_swing_generation=swing_generations,
        action_ball_ordered_action_uids=(101, 202, 303),
        action_ball_action_uid_for_envs=lambda ids: action_uids[ids],
        action_ball_birth_receipt_sha256=lambda env_id: receipts[env_id],
        action_ball_hard_contract=lambda: {
            "action_uids": [101, 202, 303],
        },
    )
    env.command_manager = types.SimpleNamespace(
        active_terms=["racket_target"],
        get_term=lambda name: command if name == "racket_target" else None,
    )
    termination_masks = {
        name: torch.zeros(2, dtype=torch.bool)
        for name in (
            "base_fell_tilt",
            "base_too_low",
            "joint_actual_forbidden",
            "joint_qdes_forbidden",
            "robot_hit_table",
        )
    }
    env.termination_manager = types.SimpleNamespace(
        active_terms=list(termination_masks),
        terminated=torch.zeros(2, dtype=torch.bool),
        time_outs=torch.zeros(2, dtype=torch.bool),
        get_term=lambda name: termination_masks[name],
    )

    env.episode_length_buf[:] = torch.tensor([41, 17])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    env.episode_length_buf[0] = 42
    action.reset(env_ids=torch.tensor([0]))

    # The next policy step belongs to a new immutable birth for env 0.  Env 1 stays on the exact
    # previous identity, which exercises both sides of the runner's generation transition check.
    receipts[0] = "c" * 64
    action_uids[0] = 303
    birth_generations[0] = 8
    swing_generations[0] = 0
    env.episode_length_buf[:] = torch.tensor([0, 18])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    return action, env


def test_forbidden_zone_exact_edges_are_closed_and_nearby_interior_is_allowed():
    # Limits [-1, 1], per-side inset = 0.1 rad + 10% * 2 rad = 0.3 rad.
    # The allowed interval is OPEN (-0.7, 0.7).
    positions = torch.tensor([[-0.7000, -0.6999, 0.6999, 0.7000]])
    limits = _limits(1, 4)
    per_joint = terminations_mod.joint_position_forbidden_zone_per_joint(
        positions,
        limits,
        margin_rad=0.1,
        margin_fraction=0.1,
    )
    assert per_joint.tolist() == [[True, False, False, True]]
    assert terminations_mod.joint_position_forbidden_zone_mask(
        positions,
        limits,
        margin_rad=0.1,
        margin_fraction=0.1,
    ).tolist() == [True]


def test_forbidden_zone_nonfinite_and_invalid_envelopes_fail_safe():
    positions = torch.tensor(
        [
            [0.0, float("nan"), 0.0, 0.0],
            [float("inf"), 0.0, 0.0, 0.0],
        ]
    )
    limits = _limits(2, 4)
    limits[0, 2, 0] = float("nan")
    limits[0, 3] = torch.tensor([0.5, 0.5])  # zero travel
    limits[1, 2] = torch.tensor([1.0, -1.0])  # reversed
    # 0.6 rad on a 2-rad envelope consumes the remaining interval when added per side.
    limits[1, 3] = torch.tensor([-0.5, 0.5])
    per_joint = terminations_mod.joint_position_forbidden_zone_per_joint(
        positions,
        limits,
        margin_rad=0.6,
        margin_fraction=0.0,
    )
    assert per_joint.tolist() == [
        [False, True, True, True],
        [True, False, True, True],
    ]


@pytest.mark.parametrize(
    ("margin_rad", "margin_fraction"),
    [
        (-0.01, 0.0),
        (float("nan"), 0.0),
        (0.0, -0.01),
        (0.0, 0.5),
        (0.0, float("inf")),
        (True, 0.0),
        (0.0, False),
        ("0.0", 0.0),
    ],
)
def test_forbidden_zone_rejects_implicit_or_invalid_margins(
    margin_rad, margin_fraction
):
    with pytest.raises(ValueError, match="margin"):
        terminations_mod.joint_position_forbidden_zone_mask(
            torch.zeros(1, 1),
            _limits(1, 1),
            margin_rad=margin_rad,
            margin_fraction=margin_fraction,
        )


def test_forbidden_zone_rejects_broadcast_and_dtype_shape_shortcuts():
    position = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="exactly match"):
        terminations_mod.joint_position_forbidden_zone_mask(
            position,
            _limits(1, 2),  # broadcasting across environments is forbidden
            margin_rad=0.0,
            margin_fraction=0.0,
        )
    with pytest.raises(RuntimeError, match="floating"):
        terminations_mod.joint_position_forbidden_zone_mask(
            position.to(torch.int64),
            _limits(2, 2).to(torch.int64),
            margin_rad=0.0,
            margin_fraction=0.0,
        )
    with pytest.raises(RuntimeError, match="identical device and dtype"):
        terminations_mod.joint_position_forbidden_zone_mask(
            position,
            _limits(2, 2).to(torch.float64),
            margin_rad=0.0,
            margin_fraction=0.0,
        )


def test_pre_clamp_qdes_is_affine_target_and_clamp_cannot_hide_violation():
    offset = torch.tensor([[0.10, -0.20], [0.10, -0.20]])
    action, env, asset = _action_and_env(scale=2.0, offset=offset, clamp=True)
    normalized_actor_action = torch.tensor([[0.70, 0.00], [0.00, 0.00]])
    action.process_actions(normalized_actor_action)

    # env0 asks for 0.10 + 2*0.70 = 1.50 rad.  The PD target is clamped to 1.00, but the
    # safety source preserves 1.50; it is neither normalized raw action (0.70) nor clamped (1.00).
    assert action.pre_clamp_qdes[:, 0].tolist() == pytest.approx([1.50, 0.10])
    assert action.processed_actions[:, 0].tolist() == pytest.approx([1.00, 0.10])
    assert action.pre_clamp_qdes_valid.tolist() == [True, True]

    # Realized q is still harmless.  Only the pre-clamp termination exposes the unsafe command.
    asset.data.joint_pos.zero_()
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]


def test_pre_clamp_validity_is_reset_per_environment_and_nonfinite_fails_safe():
    action, env, _ = _action_and_env()
    action.process_actions(torch.tensor([[0.25, 0.0], [0.25, 0.0]]))
    action.reset(env_ids=torch.tensor([0]))
    assert action.pre_clamp_qdes_valid.tolist() == [False, True]
    assert action.pre_clamp_qdes[0].tolist() == [0.0, 0.0]
    # Stale/reset-invalid env0 never terminates.
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]

    action.process_actions(
        torch.tensor([[float("nan"), 0.0], [float("inf"), 0.0]])
    )
    # Evidence remains non-finite for the DoneTerm, while the target that apply_actions/PhysX sees
    # falls back to the preceding finite deploy-space target.
    assert torch.isnan(action.pre_clamp_qdes[0, 0])
    assert torch.isinf(action.pre_clamp_qdes[1, 0])
    assert torch.all(torch.isfinite(action.processed_actions))
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.0, 0.25])
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, True]


def test_nonfinite_first_qdes_after_reset_uses_finite_default_before_physics():
    offset = torch.tensor([[0.30, -0.20], [0.10, -0.40]])
    action, _, _ = _action_and_env(offset=offset)
    action.process_actions(torch.zeros(2, 2))
    action.reset(env_ids=torch.tensor([0]))
    action.process_actions(
        torch.tensor([[float("nan"), 0.0], [float("nan"), 0.0]])
    )
    # env0 has no valid history after reset -> current default 0.30.  env1 keeps its preceding
    # processed target 0.10.  Neither non-finite request reaches apply_actions.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.30, 0.10])
    assert torch.all(torch.isfinite(action.processed_actions))


def test_nonfinite_qdes_with_no_finite_fallback_fails_before_apply():
    action, _, asset = _action_and_env(num_envs=1, joint_count=1)
    asset.data.default_joint_pos[:] = float("nan")
    with pytest.raises(RuntimeError, match="no finite q_des fallback"):
        action.process_actions(torch.tensor([[float("nan")]]))


def test_pre_apply_guard_triggers_on_outward_but_not_inward_ballistic_velocity():
    action, env, asset = _action_and_env(guard=True)
    # Both are near the upper soft edge, but the terminal envelope is the physical +/-1.2 hard
    # limit.  Over one explicit 0.1-s policy horizon env0 moves outward to 1.25 and must brake;
    # env1 moves inward to 0.65 and remains eligible.
    asset.data.joint_pos[:, 0] = torch.tensor([0.95, 0.95])
    asset.data.joint_vel[:, 0] = torch.tensor([3.0, -3.0])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    assert action.pre_apply_crossing_violation_latch.tolist() == [True, False]
    assert action.pre_apply_joint_safety_latch.tolist() == [True, False]
    assert action.pre_apply_crossing_violation_joint_latch.tolist() == [
        [True, False],
        [False, False],
    ]
    assert action.pre_apply_crossing_violation_joint_count.tolist() == [
        [1, 0],
        [0, 0],
    ]
    # Derived brake target q-qdot*dt = 0.65.  The inward env keeps its nominal zero target.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.65, 0.0])
    assert torch.all(torch.isfinite(action.processed_actions))
    assert torch.all(action.processed_actions >= -1.0)
    assert torch.all(action.processed_actions <= 1.0)

    # The DoneTerm consumes the pre-physics sticky latch even though proposed qdes itself was safe.
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]


def test_pre_apply_guard_latches_raw_qdes_and_never_sends_boundary_request():
    action, _, asset = _action_and_env(guard=True)
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    action.process_actions(torch.tensor([[1.20, 0.0], [0.25, 0.0]]))

    assert action.pre_clamp_qdes[:, 0].tolist() == pytest.approx([1.20, 0.25])
    assert action.pre_apply_qdes_violation_latch.tolist() == [True, False]
    assert action.pre_apply_qdes_violation_joint_count.tolist() == [
        [1, 0],
        [0, 0],
    ]
    # The violating joint gets the derived stationary brake target q=0, not the soft edge.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.0, 0.25])
    assert torch.all(torch.isfinite(action.processed_actions))


def test_pre_apply_guard_soft_only_qdes_intrusion_clamps_but_does_not_terminate():
    action, env, asset = _action_and_env(guard=True)
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    # 1.10 is outside deploy soft +1.0 but still strictly inside physical hard +1.2.
    action.process_actions(torch.tensor([[1.10, 0.0], [0.25, 0.0]]))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [False, False]
    assert action.processed_actions[:, 0].tolist() == pytest.approx([1.0, 0.25])
    assert terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]


def test_pre_apply_guard_nonfinite_state_latches_and_uses_finite_brake_target():
    offset = torch.tensor([[0.30, -0.20], [0.10, -0.40]])
    action, _, asset = _action_and_env(offset=offset, guard=True)
    asset.data.joint_pos[0, 0] = float("nan")
    asset.data.joint_vel[1, 0] = float("inf")
    action.process_actions(torch.zeros(2, 2))
    assert action.pre_apply_crossing_violation_latch.tolist() == [True, True]
    assert torch.all(torch.isfinite(action.processed_actions))
    # q NaN falls back to default 0.30; qdot Inf becomes zero.  Both remain inside soft limits.
    assert action.processed_actions[:, 0].tolist() == pytest.approx([0.30, 0.0])


def test_pre_apply_guard_uses_closed_two_percent_physical_hard_inset():
    action, _, asset = _action_and_env(
        guard=True, guard_margin_fraction=0.02
    )
    hard = asset.data.joint_pos_limits[0, 0]
    inner_upper = hard[1] - 0.02 * (hard[1] - hard[0])
    asset.data.joint_pos[:, 0] = torch.stack(
        (inner_upper - 1.0e-4, inner_upper)
    )
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(2, 2))
    # The remaining interval is open: an exact 2%-inset edge is forbidden.
    assert action.pre_apply_crossing_violation_latch.tolist() == [False, True]


def test_pre_apply_guard_sticky_state_counts_and_partial_reset():
    action, _, asset = _action_and_env(guard=True)
    asset.data.joint_pos[:, 0] = torch.tensor([0.95, 0.0])
    asset.data.joint_vel[:, 0] = torch.tensor([3.0, 0.0])
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [True, False]

    # A subsequent safe proposal/state cannot erase forensic evidence before episode reset.
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [True, False]
    assert action.pre_apply_crossing_violation_joint_count[:, 0].tolist() == [1, 0]

    # Make env1 violate too, then prove partial reset clears only env0's latch/counters.
    action.process_actions(torch.tensor([[0.0, 0.0], [1.2, 0.0]]))
    _finish_guarded_policy_step(action, asset)
    assert action.pre_apply_joint_safety_latch.tolist() == [True, True]
    action.reset(env_ids=torch.tensor([0]))
    assert action.pre_apply_joint_safety_latch.tolist() == [False, True]
    assert action.pre_apply_qdes_violation_joint_count[0].tolist() == [0, 0]
    assert action.pre_apply_qdes_violation_joint_count[1, 0].item() == 1
    assert action.pre_apply_crossing_violation_joint_count[0].tolist() == [0, 0]


def test_pre_apply_guard_is_opt_in_and_legacy_finite_target_is_unchanged():
    action, _, asset = _action_and_env(guard=False)
    asset.data.joint_pos[:, 0] = 0.95
    asset.data.joint_vel[:, 0] = 1.0
    proposal = torch.tensor([[0.90, 0.0], [1.20, 0.0]])
    action.process_actions(proposal)
    # Guard-off preserves the old deploy clamp exactly: safe 0.90 stays 0.90; 1.20 clamps to 1.0.
    assert torch.equal(
        action.processed_actions,
        torch.tensor([[0.90, 0.0], [1.00, 0.0]]),
    )
    assert action.pre_apply_joint_safety_latch.tolist() == [False, False]
    assert action.pre_apply_qdes_violation_joint_count.sum().item() == 0
    assert action.pre_apply_crossing_violation_joint_count.sum().item() == 0


def test_legacy_apply_actions_dispatch_is_unchanged_and_ledger_stays_disabled(monkeypatch):
    dispatched = []

    def fake_apply(term):
        dispatched.append(term.processed_actions.detach().clone())

    monkeypatch.setattr(
        hope_actions_mod.JointPositionAction,
        "apply_actions",
        fake_apply,
        raising=False,
    )
    action, _, _ = _action_and_env(guard=False)
    action.process_actions(torch.tensor([[0.25, 0.0], [1.2, 0.0]]))
    expected = action.processed_actions.detach().clone()
    action.apply_actions()
    assert len(dispatched) == 1
    assert torch.equal(dispatched[0], expected)
    assert action.joint_safety_ledger_snapshot() == {
        "schema_version": 1,
        "enabled": False,
    }


def test_legacy_reset_delegates_original_ids_without_safety_archive(monkeypatch):
    action, _, _ = _action_and_env(guard=False)
    calls = []

    def fake_reset(term, env_ids=None):
        calls.append(env_ids)

    monkeypatch.setattr(
        hope_actions_mod.JointPositionAction,
        "reset",
        fake_reset,
        raising=False,
    )
    ids = torch.tensor([1])
    action.reset(env_ids=ids)
    assert len(calls) == 1
    assert calls[0] is ids
    assert action.joint_safety_ledger_snapshot() == {
        "schema_version": 1,
        "enabled": False,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clamp": False, "guard_policy_dt_s": 0.1},
        {"guard_policy_dt_s": None},
        {"guard_policy_dt_s": 0.2},
        {"guard_margin_rad": None},
        {"guard_margin_rad": -0.01},
        {"guard_margin_fraction": None},
        {"guard_margin_fraction": 0.5},
        {"expected_decimation": None},
        {"decimation": 2, "expected_decimation": 4},
        {"terminal_archive_capacity": None},
        {"terminal_archive_capacity": 0},
    ],
)
def test_pre_apply_guard_requires_explicit_runtime_matched_contract(kwargs):
    with pytest.raises(ValueError, match="pre_apply"):
        _action_and_env(guard=True, **kwargs)


def test_pre_apply_guard_rejects_soft_envelope_outside_physical_hard_limits():
    action, _, asset = _action_and_env(guard=True)
    asset.data.joint_pos_limits[:] = _limits(2, 2, lo=-0.5, hi=0.5)
    with pytest.raises(RuntimeError, match="soft deploy envelope"):
        action.process_actions(torch.zeros(2, 2))


def test_physics_substep_ledger_records_exact_decimation_and_fresh_timestamps():
    action, _, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    physics_dt = 0.025
    for index, q_value in enumerate((0.00, 0.10, 0.20, 0.30)):
        _set_sim_timestamp(asset, index * physics_dt)
        asset.data.joint_pos[:, 0] = q_value
        asset.data.joint_vel[:, 0] = 0.5
        action.apply_actions()
    _set_sim_timestamp(asset, 4 * physics_dt)
    asset.data.joint_pos[:, 0] = 0.40
    action.finalize_joint_safety_post_step_readback()

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["expected_apply_calls"] == 4
    assert snapshot["apply_call_count"] == 4
    assert snapshot["post_readback_count"] == 1
    assert snapshot["complete"] is True
    assert snapshot["record_kind"] == ("apply", "apply", "apply", "apply", "post")
    assert snapshot["call_index"] == (0, 1, 2, 3, 4)
    assert snapshot["timestamp_s"] == pytest.approx(
        (0.0, 0.025, 0.05, 0.075, 0.1)
    )
    assert snapshot["q"][:, 0, 0].tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )
    # hard upper is +1.2, so the final readback gap is 0.8.
    assert snapshot["hard_upper_gap"][-1, 0, 0].item() == pytest.approx(0.8)
    assert snapshot["env_valid"].all().item()

    # Snapshot tensors are detached clones, not a mutation channel into the live ledger.
    snapshot["q"].fill_(99.0)
    assert action.joint_safety_ledger_snapshot()["q"][0, 0, 0].item() == 0.0


def test_physics_substep_ledger_catches_crossing_and_bounced_actual_hard_edge():
    action, env, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))

    # apply0: q is inside hard +1.2 but q+qdot*0.025 crosses it.
    _set_sim_timestamp(asset, 0.0)
    asset.data.joint_pos[0, 0] = 1.15
    asset.data.joint_vel[0, 0] = 3.0
    action.apply_actions()
    assert action.physics_substep_hard_crossing_latch.tolist() == [True, False]
    assert action.processed_actions[0, 0].item() == pytest.approx(1.0)

    # apply1 observes an actual hard-edge breach.  Later readbacks bounce safely inside.
    _set_sim_timestamp(asset, 0.025)
    asset.data.joint_pos[0, 0] = 1.21
    asset.data.joint_vel[0, 0] = 0.0
    action.apply_actions()
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [True, False]
    for index in (2, 3):
        _set_sim_timestamp(asset, index * 0.025)
        asset.data.joint_pos.zero_()
        asset.data.joint_vel.zero_()
        action.apply_actions()
    _set_sim_timestamp(asset, 0.1)
    asset.data.joint_pos.zero_()
    asset.data.joint_vel.zero_()

    # Either DoneTerm may run first; both use the same idempotent final post-step readback.
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    actual = terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.0,
    )
    qdes = terminations_mod.pre_clamp_qdes_forbidden_zone(
        env,
        "joint_pos",
        "joint_pos_limits",
        0.0,
        0.0,
    )
    assert actual.tolist() == [True, False]  # sticky substep edge survives the safe bounce
    assert qdes.tolist() == [True, False]  # combined safety latch is visible pre-reset
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["complete"] is True
    assert snapshot["hard_crossing"][0, 0, 0].item()
    assert snapshot["actual_hard_edge"][1, 0, 0].item()
    assert snapshot["substep_crossing_joint_count"][0, 0].item() >= 1
    assert snapshot["substep_actual_joint_count"][0, 0].item() == 1


def test_physics_substep_ledger_rejects_stale_or_missing_apply_sequence():
    action, env, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    _set_sim_timestamp(asset, 0.0)
    action.apply_actions()
    # The second apply must observe timestamp + physics_dt, never the previous stale state.
    with pytest.raises(RuntimeError, match="stale or skipped"):
        action.apply_actions()
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["apply_call_count"] == 1
    assert snapshot["record_count"] == 1
    assert action.physics_substep_hard_crossing_joint_count.sum().item() == 0

    action2, env2, asset2 = _action_and_env(guard=True)
    action2.process_actions(torch.zeros(2, 2))
    _set_sim_timestamp(asset2, 0.0)
    with pytest.raises(RuntimeError, match="exactly 4 prior apply_actions"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env2,
            "joint_pos",
            "joint_pos_limits",
            0.0,
            0.0,
        )


def test_physics_substep_lazy_buffer_timestamp_failure_is_transactional():
    action, _, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    _set_sim_timestamp(asset, 0.0)
    asset.data._joint_vel.timestamp = -0.025
    with pytest.raises(RuntimeError, match="lazy-buffer timestamps"):
        action.apply_actions()
    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["apply_call_count"] == 0
    assert snapshot["record_count"] == 0
    assert action.physics_substep_hard_crossing_joint_count.sum().item() == 0
    assert action.physics_substep_actual_hard_edge_joint_count.sum().item() == 0


def test_update_accumulator_allows_multiple_nonterminal_steps_before_consume():
    action, _, asset = _action_and_env(guard=True)
    for _ in range(2):
        action.process_actions(torch.zeros(2, 2))
        _finish_guarded_policy_step(action, asset)

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["policy_step_sequence"] == 1
    assert snapshot["complete"] is True
    assert snapshot["terminal_archive_used"] == 0
    assert snapshot["policy_step_summary_used"] == 2
    assert [
        item["policy_step_sequence"]
        for item in snapshot["identity_bound_policy_steps"]
    ] == [0, 1]
    assert snapshot["identity_bound_policy_steps"][0]["complete"].all().item()
    assert snapshot["current_action_episode_identity"][
        "action_episode_sequence"
    ].tolist() == [0, 0]
    accumulated = snapshot["since_last_consume"]
    assert accumulated["policy_step_count"].tolist() == [2, 2]
    assert accumulated["complete_policy_step_count"].tolist() == [2, 2]
    assert accumulated["incomplete_policy_step_count"].tolist() == [0, 0]
    assert accumulated["apply_readback_count"].tolist() == [8, 8]
    assert accumulated["post_readback_count"].tolist() == [2, 2]
    assert accumulated["timestamp_invariant_pass_count"].tolist() == [2, 2]

    token, first = action.prepare_joint_safety_ledger_consume()
    assert first["since_last_consume"]["has_data"] is True
    action.acknowledge_joint_safety_ledger(token)
    second_token, second = action.prepare_joint_safety_ledger_consume()
    assert second["since_last_consume"]["has_data"] is False
    assert second["since_last_consume"]["consume_sequence"] == 1
    assert second["since_last_consume"]["policy_step_count"].sum().item() == 0
    assert second["terminal_archive_used"] == 0
    assert second["policy_step_summary_used"] == 0
    action.acknowledge_joint_safety_ledger(second_token)


def test_terminal_reset_before_consume_preserves_cpu_archive_and_birth_identity():
    action, env, asset = _action_and_env(guard=True)
    receipts = ("a" * 64, "b" * 64)
    action_uids = torch.tensor([101, 202], dtype=torch.long)
    command = types.SimpleNamespace(
        action_ball_enabled=True,
        action_ball_episode_generation=torch.tensor([7, 9], dtype=torch.long),
        action_ball_swing_generation=torch.tensor([3, 4], dtype=torch.long),
        action_ball_action_uid_for_envs=lambda ids: action_uids[ids],
        action_ball_birth_receipt_sha256=lambda env_id: receipts[env_id],
    )
    env.command_manager = types.SimpleNamespace(
        get_term=lambda name: command if name == "racket_target" else None
    )
    env.episode_length_buf[:] = torch.tensor([41, 17])
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[0, 0] = 1.21
    _finish_guarded_policy_step(action, asset)

    env.termination_manager = types.SimpleNamespace(
        terminated=torch.tensor([False, False]),
        time_outs=torch.tensor([False, False]),
    )
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    env.episode_length_buf[0] = 42
    action.reset(env_ids=torch.tensor([0]))
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [False, False]

    snapshot = action.joint_safety_ledger_snapshot()
    assert not snapshot["env_valid"][:, 0].any().item()
    assert snapshot["terminal_archive_used"] == 1
    assert snapshot["policy_step_summary_used"] == 1
    step_identity = snapshot["identity_bound_policy_steps"][0][
        "action_identity"
    ]
    assert step_identity["action_uid"].tolist() == [101, 202]
    assert step_identity["birth_generation"].tolist() == [7, 9]
    assert snapshot["terminal_archive_payload_bytes"] > 0
    archived = snapshot["terminal_archives"][0]
    assert archived["env_id"] == 0
    assert archived["reasons"] == ("unsafe", "reset")
    assert archived["reset_hook_observed"] is True
    assert archived["terminated"] is True
    assert archived["timed_out"] is False
    assert archived["action_episode_sequence"] == 0
    assert archived["episode_length"] == 42
    assert archived["episode_length_at_policy_start"] == 41
    assert archived["episode_length_at_reset_hook"] == 42
    assert archived["action_uid"] == 101
    assert archived["birth_generation"] == 7
    assert archived["swing_generation"] == 3
    assert archived["birth_receipt_sha256"] == receipts[0]
    assert archived["included_in_accumulator"] is True
    transcript = archived["transcript"]
    assert transcript["complete"] is True
    assert transcript["q"].device.type == "cpu"
    assert transcript["actual_hard_edge"][:, 0].any().item()
    assert transcript["joint_pos_timestamp_s"] == pytest.approx(
        (0.0, 0.025, 0.05, 0.075, 0.1)
    )

    # Export mutation cannot corrupt the retained single-consumer evidence.
    transcript["q"].fill_(99.0)
    assert (
        action.joint_safety_ledger_snapshot()["terminal_archives"][0][
            "transcript"
        ]["q"][0, 0].item()
        != 99.0
    )
    token, consumed = action.prepare_joint_safety_ledger_consume()
    assert consumed["terminal_archive_used"] == 1
    action.acknowledge_joint_safety_ledger(token)
    assert action.joint_safety_ledger_snapshot()["terminal_archive_used"] == 0


def test_partial_reset_archives_incomplete_row_and_other_row_can_complete():
    action, _, asset = _action_and_env(guard=True)
    action.process_actions(torch.zeros(2, 2))
    for index in range(2):
        _set_sim_timestamp(asset, index * 0.025)
        action.apply_actions()
    action.reset(env_ids=torch.tensor([0]))
    for index in range(2, 4):
        _set_sim_timestamp(asset, index * 0.025)
        action.apply_actions()
    _set_sim_timestamp(asset, 0.1)
    action.finalize_joint_safety_post_step_readback()

    snapshot = action.joint_safety_ledger_snapshot()
    accumulated = snapshot["since_last_consume"]
    assert accumulated["policy_step_count"].tolist() == [1, 1]
    assert accumulated["complete_policy_step_count"].tolist() == [0, 1]
    assert accumulated["incomplete_policy_step_count"].tolist() == [1, 0]
    assert snapshot["terminal_archive_used"] == 1
    assert snapshot["policy_step_summary_used"] == 1
    assert snapshot["identity_bound_policy_steps"][0]["complete"].tolist() == [
        False,
        True,
    ]
    archived = snapshot["terminal_archives"][0]
    assert archived["env_id"] == 0
    assert archived["reasons"] == ("reset",)
    assert archived["transcript"]["complete"] is False
    assert archived["transcript"]["apply_call_count"] == 2
    assert archived["transcript"]["post_readback_count"] == 0
    assert not snapshot["env_valid"][:, 0].any().item()
    assert snapshot["env_valid"][:, 1].all().item()


def test_terminal_archive_overflow_is_sticky_and_never_overwrites():
    action, _, asset = _action_and_env(
        guard=True, terminal_archive_capacity=1
    )
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[:, 0] = torch.tensor([1.21, -1.21])
    _finish_guarded_policy_step(action, asset)
    with pytest.raises(RuntimeError, match="archive overflow"):
        action.reset(env_ids=torch.tensor([0, 1]))

    snapshot = action.joint_safety_ledger_snapshot()
    assert snapshot["terminal_archive_capacity"] == 1
    assert snapshot["terminal_archive_used"] == 0
    assert snapshot["terminal_archive_overflow_latch"] is True
    assert snapshot["terminal_archive_overflow_count"] == 1
    with pytest.raises(RuntimeError, match="one-shot consume is disabled"):
        action.consume_joint_safety_ledger()
    after = action.joint_safety_ledger_snapshot()
    assert after["terminal_archive_used"] == 0
    assert after["terminal_archive_overflow_latch"] is True
    assert after["terminal_archive_overflow_count"] == 1
    with pytest.raises(RuntimeError, match="overflow is sticky"):
        action.process_actions(torch.zeros(2, 2))


def test_one_shot_consume_fails_closed_even_when_guard_is_disabled():
    action, _, _ = _action_and_env(guard=False)
    assert action.joint_safety_ledger_snapshot()["enabled"] is False

    with pytest.raises(RuntimeError, match="one-shot consume is disabled"):
        action.consume_joint_safety_ledger()


@pytest.mark.parametrize(
    "bad_ids",
    [
        torch.tensor([0.9]),
        torch.tensor([True]),
        [0.9],
        [True],
    ],
)
def test_joint_safety_reset_rejects_noninteger_environment_ids(bad_ids):
    action, _, _ = _action_and_env(guard=True)
    with pytest.raises(TypeError, match="integer"):
        action.reset(env_ids=bad_ids)


def test_runner_consumes_once_persists_every_step_and_keeps_cross_reset_identity(
    monkeypatch, tmp_path, capsys
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    prepare_calls = []
    acknowledge_calls = []
    original_prepare = action.prepare_joint_safety_ledger_consume
    original_acknowledge = action.acknowledge_joint_safety_ledger

    def counted_prepare():
        prepare_calls.append(1)
        return original_prepare()

    def counted_acknowledge(token):
        acknowledge_calls.append(token)
        return original_acknowledge(token)

    action.prepare_joint_safety_ledger_consume = counted_prepare
    action.acknowledge_joint_safety_ledger = counted_acknowledge
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0

    first = runner._consume_joint_safety_update(
        12, expected_action_term=action
    )
    second = runner._consume_joint_safety_update(
        12, expected_action_term=action
    )

    assert first is second
    assert prepare_calls == [1]
    assert len(acknowledge_calls) == 1
    assert first["consume_sequence"] == 0
    assert first["policy_step_count"] == 2
    assert len(first["per_policy_step_sparse_counters"]) == 2
    assert first["terminal_archive_count"] == 1
    assert first["terminal_reason_counts"] == {"reset": 1}
    artifact = tmp_path / first["artifact"]["path"]
    assert artifact.is_file()
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["status"] == "prepared_before_optimizer"
    archives = payload["terminal"]["entries"]
    assert len(archives) == 1
    assert archives[0]["storage"] == "full_forensic"
    archived = archives[0]["archive"]
    assert archived["action_uid"] == 101
    assert archived["birth_generation"] == 7
    assert archived["birth_receipt_sha256"] == "a" * 64
    identity = payload["identity"]
    assert identity["initial"]["action_uid"].tolist() == [101, 202]
    assert identity["changes"]["action_uid"]["index"].tolist() == [[1, 0]]
    assert identity["changes"]["action_uid"]["value"].tolist() == [303]
    assert (tmp_path / first["optimizer_commit"]["path"]).is_file()
    # The exact prepared generation is acknowledged only after its optimizer marker is durable.
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is False
    )
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HOPE_JOINT_SAFETY_UPDATE_JSON=")
    ]
    assert len(lines) == 1
    assert json.loads(lines[0].split("=", 1)[1]) == first


def test_protected_task_detection_covers_action_ball_and_upper_safe(monkeypatch):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    action_ball_cfg = types.SimpleNamespace(
        commands=types.SimpleNamespace(
            racket_target=types.SimpleNamespace(target_mode="action_ball")
        )
    )
    runner.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(cfg=action_ball_cfg)
    )
    assert runner._effective_reward_activation_task_kind() == "action_ball"

    upper_cfg_type = type("HOPEPingPongUpperSafeAgibotA3EnvCfg", (), {})
    upper_cfg = upper_cfg_type()
    upper_cfg.commands = types.SimpleNamespace(
        racket_target=types.SimpleNamespace(target_mode="virtual_ball")
    )
    runner.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(cfg=upper_cfg)
    )
    assert runner._effective_reward_activation_task_kind() == "upper_safe"


def test_learn_consumes_joint_safety_before_command_rollout_callback(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    callback_observations = []
    call_order = []
    command = env.command_manager.get_term("racket_target")

    def on_rollout_end(step):
        call_order.append("curriculum_callback")
        callback_observations.append(
            (
                step,
                action.joint_safety_ledger_snapshot()["since_last_consume"][
                    "has_data"
                ],
                list((tmp_path / "joint_safety_ledgers").glob("*.pt")),
            )
        )

    command.on_rollout_end = on_rollout_end

    effective_reward_module = types.ModuleType(
        "whole_body_tracking.utils.effective_reward_recipe"
    )

    class FakeRewardLedger:
        def __init__(self, *_args, **_kwargs):
            pass

        def begin_environment_step(self):
            return "fake-step"

        def observe_after_environment_step(self, *_args):
            pass

        def prepare_update(self, step, **_kwargs):
            return {
                "activation": {"ppo_update": step},
                "per_action": None,
                "safety": None,
            }

        def acknowledge_update(self, _prepared):
            pass

    effective_reward_module.EffectiveRewardActivationLedger = FakeRewardLedger
    effective_reward_module.ActionBoundRewardEvidenceLedger = FakeRewardLedger
    effective_reward_module.canonical_effective_reward_activation_json = (
        lambda record: json.dumps(record, sort_keys=True)
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        effective_reward_module,
    )

    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def one_update(_self, **_kwargs):
        _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", one_update, raising=False)
    wrapper = types.SimpleNamespace(unwrapped=env, step=lambda *_a, **_k: None)
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = wrapper
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0
    runner.current_learning_iteration = 12
    runner.alg = types.SimpleNamespace(
        update=lambda: call_order.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: "action_ball"
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False
    original_prepare = runner._prepare_joint_safety_update
    original_commit_marker = runner._persist_joint_safety_optimizer_commit
    original_directory_fsync = runner._joint_safety_fsync_directory
    original_acknowledge = action.acknowledge_joint_safety_ledger

    def ordered_prepare(*args, **kwargs):
        call_order.append("prepare_validate_persist")
        return original_prepare(*args, **kwargs)

    def ordered_commit_marker(prepared):
        call_order.append("durable_optimizer_marker")
        return original_commit_marker(prepared)

    def ordered_directory_fsync(directory):
        call_order.append("directory_fsync")
        return original_directory_fsync(directory)

    def ordered_acknowledge(token):
        call_order.append("ledger_ack")
        return original_acknowledge(token)

    runner._prepare_joint_safety_update = ordered_prepare
    runner._persist_joint_safety_optimizer_commit = ordered_commit_marker
    runner._joint_safety_fsync_directory = ordered_directory_fsync
    action.acknowledge_joint_safety_ledger = ordered_acknowledge

    runner.learn(num_learning_iterations=1)

    assert len(callback_observations) == 1
    step, has_data_at_callback, artifacts_at_callback = callback_observations[0]
    assert step == 12
    assert has_data_at_callback is False
    assert len(artifacts_at_callback) == 1
    assert call_order == [
        "prepare_validate_persist",
        "directory_fsync",
        # Action-bound Reward evidence has its own pre-optimizer durable
        # receipt and post-optimizer commit marker.
        "directory_fsync",
        "optimizer",
        "directory_fsync",
        "durable_optimizer_marker",
        "directory_fsync",
        "ledger_ack",
        "curriculum_callback",
    ]


def test_inflight_frozen_eval_fences_next_optimizer_and_sidecar_failure(
    monkeypatch, tmp_path
):
    """A published request cannot be judged after another PPO update."""

    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    calls = []

    class FailingSidecarTerm:
        def __init__(self):
            self.stage = None
            self.inflight_polls = 0

        def action_ball_frozen_evaluation_boundary(
            self, *, phase, runner_bindings=None, **_kwargs
        ):
            if phase == "poll" and self.stage is None:
                return {
                    "request_seq": 0,
                    "request_due": True,
                    "needs_global_reset": False,
                    "needs_ack_checkpoint": False,
                    "stage": None,
                    "requires_runner_binding": False,
                }
            if phase == "publish_request":
                assert runner_bindings == {
                    "policy_generation": 5,
                    "policy_state": "frozen",
                }
                self.stage = "published"
                calls.append("request_published")
                return {
                    "request_seq": 0,
                    "published": True,
                }
            if phase == "poll" and self.stage == "published":
                self.inflight_polls += 1
                if self.inflight_polls == 1:
                    return {
                        "request_seq": 0,
                        "request_due": False,
                        "needs_global_reset": False,
                        "needs_ack_checkpoint": False,
                        "stage": "published",
                        "requires_runner_binding": False,
                    }
                raise RuntimeError("sidecar liveness failed")
            raise AssertionError(
                f"unexpected frozen-eval phase {phase!r}"
            )

    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def two_updates(_self, *, num_learning_iterations, **_kwargs):
        for _ in range(num_learning_iterations):
            _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", two_updates, raising=False)
    monkeypatch.setattr(
        runner_mod.time,
        "sleep",
        lambda _seconds: calls.append("fenced_poll_wait"),
    )

    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(),
        step=lambda *_args, **_kwargs: None,
        reset=lambda: calls.append("env_reset"),
    )
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0
    runner.current_learning_iteration = 5
    runner.alg = types.SimpleNamespace(
        update=lambda: calls.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: None
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False
    term = FailingSidecarTerm()
    runner._action_ball_frozen_eval_term = lambda: term
    runner._notify_command_terms_rollout_end = (
        lambda step: calls.append(f"rollout_end:{step}")
    )
    runner._action_ball_control_checkpoint = (
        lambda *, purpose, **_kwargs: tmp_path / f"{purpose}.pt"
    )
    runner._frozen_eval_runner_bindings = (
        lambda *, policy_generation: {
            "policy_generation": policy_generation,
            "policy_state": "frozen",
        }
    )
    runner._action_ball_frozen_eval_poll_interval_s = 0.001

    with pytest.raises(RuntimeError, match="sidecar liveness failed"):
        runner.learn(num_learning_iterations=2)

    assert calls.count("optimizer") == 1
    assert calls == [
        "optimizer",
        "rollout_end:5",
        "request_published",
        "fenced_poll_wait",
    ]


def test_runner_rejects_sticky_overflow_before_consuming(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, _ = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    action._joint_safety_policy_step_summary_overflow_latch = True
    action._joint_safety_policy_step_summary_overflow_count = 1
    calls = []
    original_consume = action.consume_joint_safety_ledger

    def counted_consume():
        calls.append(1)
        return original_consume()

    action.consume_joint_safety_ledger = counted_consume
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0

    with pytest.raises(RuntimeError, match="overflow is already latched"):
        runner._consume_joint_safety_update(
            0, expected_action_term=action
        )
    assert calls == []
    assert not (tmp_path / "joint_safety_ledgers").exists()


def test_runner_rejects_terminal_archive_identity_drift(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    snapshot["terminal_archives"][0]["action_uid"] = 999
    action.prepare_joint_safety_ledger_consume = lambda: (token, snapshot)
    acknowledge_calls = []
    action.acknowledge_joint_safety_ledger = lambda value: acknowledge_calls.append(
        value
    )
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0

    with pytest.raises(
        RuntimeError, match="action_uid does not match its policy-step identity"
    ):
        runner._consume_joint_safety_update(
            3, expected_action_term=action
        )
    assert acknowledge_calls == []
    assert (
        action.joint_safety_ledger_snapshot()["since_last_consume"]["has_data"]
        is True
    )
    assert not list((tmp_path / "joint_safety_ledgers").glob("*.pt"))


def test_two_phase_consume_freezes_every_mutator_and_requires_exact_token():
    action, _, asset = _action_and_env(guard=True)
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)

    token, snapshot = action.prepare_joint_safety_ledger_consume()
    assert snapshot["since_last_consume"]["has_data"] is True
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.prepare_joint_safety_ledger_consume()
    with pytest.raises(RuntimeError, match="refusing process_actions"):
        action.process_actions(torch.zeros(2, 2))
    with pytest.raises(RuntimeError, match="refusing reset"):
        action.reset(env_ids=torch.tensor([0]))
    with pytest.raises(RuntimeError, match="refusing apply_actions"):
        action.apply_actions()
    with pytest.raises(RuntimeError, match="refusing finalize"):
        action.finalize_joint_safety_post_step_readback()
    with pytest.raises(RuntimeError, match="token does not match"):
        action.acknowledge_joint_safety_ledger((*token[:-1], "0" * 64))
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True

    action.acknowledge_joint_safety_ledger(token)
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is False
    with pytest.raises(RuntimeError, match="no prepared consume"):
        action.acknowledge_joint_safety_ledger(token)


def test_two_phase_ack_rejects_private_evidence_mutation_without_clearing():
    action, _, asset = _action_and_env(guard=True)
    action.process_actions(torch.zeros(2, 2))
    _finish_guarded_policy_step(action, asset)
    token, _ = action.prepare_joint_safety_ledger_consume()
    action._joint_safety_accumulator_policy_steps[0] += 1

    with pytest.raises(RuntimeError, match="evidence changed"):
        action.acknowledge_joint_safety_ledger(token)
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


def test_prepare_returns_borrowed_view_without_recursive_export_clone(
    monkeypatch,
):
    action, _, asset = _action_and_env(
        num_envs=256,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    for _ in range(4):
        action.process_actions(torch.zeros(256, 31))
        _finish_guarded_policy_step(action, asset)

    fingerprint_calls = []
    original_fingerprint = action._joint_safety_evidence_fingerprint

    def counted_fingerprint():
        fingerprint_calls.append(1)
        return original_fingerprint()

    def forbidden_export_clone(_value):
        raise AssertionError(
            "prepare must not recursively clone the monitoring export"
        )

    monkeypatch.setattr(
        action, "_joint_safety_evidence_fingerprint", counted_fingerprint
    )
    monkeypatch.setattr(
        action, "_joint_safety_export_clone", forbidden_export_clone
    )
    tracemalloc.start()
    start = time.perf_counter()
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    elapsed = time.perf_counter() - start
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert fingerprint_calls == [1]
    assert elapsed < 0.25
    assert peak_bytes < 512 * 1024
    assert "q" not in snapshot
    assert "qdot" not in snapshot
    assert snapshot["identity_bound_policy_steps"][0] is (
        action._joint_safety_policy_step_summaries[0]
    )
    assert (
        snapshot["since_last_consume"]["policy_step_count"].data_ptr()
        == action._joint_safety_accumulator_policy_steps.data_ptr()
    )
    assert (
        snapshot["since_last_consume"][
            "actual_hard_edge_joint_count"
        ].data_ptr()
        == action._joint_safety_accumulator_actual_hard_edge_joint_count.data_ptr()
    )

    action.acknowledge_joint_safety_ledger(token)
    assert fingerprint_calls == [1, 1]


def test_4096_by_24_prepare_has_bounded_python_peak_and_no_dense_clone(
    monkeypatch,
):
    action, _, asset = _action_and_env(
        num_envs=4096,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        terminal_archive_capacity=64,
    )
    zero_action = torch.zeros(4096, 31)
    for _ in range(24):
        action.process_actions(zero_action)
        _finish_guarded_policy_step(action, asset)

    monkeypatch.setattr(
        action,
        "_joint_safety_export_clone",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("prepare attempted a recursive dense export clone")
        ),
    )
    tracemalloc.start()
    start = time.perf_counter()
    token, snapshot = action.prepare_joint_safety_ledger_consume()
    elapsed = time.perf_counter() - start
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert elapsed < 1.0
    assert peak_bytes < 1024 * 1024
    assert len(snapshot["identity_bound_policy_steps"]) == 24
    assert snapshot["terminal_archive_used"] == 0
    assert (
        snapshot["identity_bound_policy_steps"][-1][
            "minimum_hard_gap"
        ].data_ptr()
        == action._joint_safety_policy_step_summaries[-1][
            "minimum_hard_gap"
        ].data_ptr()
    )
    action.acknowledge_joint_safety_ledger(token)


def test_runner_rejects_self_consistent_incomplete_readback_window(
    monkeypatch
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    _, snapshot = action.prepare_joint_safety_ledger_consume()
    summary = snapshot["identity_bound_policy_steps"][0]
    summary["complete"][0] = False
    summary["apply_readback_count"][0] = 3
    summary["timestamp_invariant_pass"][0] = False
    since = snapshot["since_last_consume"]
    since["complete_policy_step_count"][0] -= 1
    since["incomplete_policy_step_count"][0] += 1
    since["apply_readback_count"][0] -= 1
    since["timestamp_invariant_pass_count"][0] -= 1

    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.num_steps_per_env = 2
    contract = runner._joint_safety_runtime_contract(action)
    with pytest.raises(RuntimeError, match="incomplete"):
        runner._validate_joint_safety_update_snapshot(
            snapshot, step=0, contract=contract
        )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


@pytest.mark.parametrize(
    "corruption,match",
    [
        (
            lambda transcript: transcript["q"].__setitem__(
                (0, 0), float("nan")
            ),
            "actual-hard-edge mask|hard gaps",
        ),
        (
            lambda transcript: transcript.__setitem__("record_count", 999),
            "complete bound",
        ),
        (
            lambda transcript: transcript.__setitem__(
                "record_kind", ("forged",) * 5
            ),
            "kind/index",
        ),
        (
            lambda transcript: transcript[
                "substep_actual_joint_count"
            ].__setitem__(0, 99),
            "substep masks/counters",
        ),
        (
            lambda transcript: transcript.__setitem__(
                "joint_vel_timestamp_s",
                (0.0, 0.005, 0.010, 0.015, 99.0),
            ),
            "lazy-buffer timestamps",
        ),
    ],
)
def test_runner_deeply_rejects_terminal_transcript_corruption(
    monkeypatch, corruption, match
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    _, snapshot = action.prepare_joint_safety_ledger_consume()
    corruption(snapshot["terminal_archives"][0]["transcript"])
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.num_steps_per_env = 2
    contract = runner._joint_safety_runtime_contract(action)

    with pytest.raises(RuntimeError, match=match):
        runner._validate_joint_safety_update_snapshot(
            snapshot, step=0, contract=contract
        )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


def test_runner_persistence_failure_keeps_prepared_action_evidence(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env = _two_step_cross_reset_action_ball_ledger()
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 2
    runner.rank = 0
    runner._persist_joint_safety_update = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected disk full")
        )
    )

    with pytest.raises(OSError, match="disk full"):
        runner._prepare_joint_safety_update(
            0, expected_action_term=action
        )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True
    assert runner._joint_safety_pending_prepared["status"] == "snapshot_frozen"
    with pytest.raises(RuntimeError, match="prepared but not acknowledged"):
        action.process_actions(torch.zeros(2, 2))


def test_actual_hard_edge_is_durable_and_blocks_optimizer(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, asset = _action_and_env(
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
    )
    action.process_actions(torch.zeros(2, 2))
    asset.data.joint_pos[0, 0] = 1.21
    _finish_guarded_policy_step(action, asset)
    env.reset_terminated = torch.tensor([True, False])
    env.reset_time_outs = torch.tensor([False, False])
    action.reset(env_ids=torch.tensor([0]))

    effective_reward_module = types.ModuleType(
        "whole_body_tracking.utils.effective_reward_recipe"
    )

    class FakeRewardLedger:
        def __init__(self, *_args, **_kwargs):
            pass

        def observe_after_environment_step(self):
            pass

        def finish_update(self, step):
            return {"ppo_update": step}

    effective_reward_module.EffectiveRewardActivationLedger = FakeRewardLedger
    effective_reward_module.ActionBoundRewardEvidenceLedger = FakeRewardLedger
    effective_reward_module.canonical_effective_reward_activation_json = (
        lambda record: json.dumps(record, sort_keys=True)
    )
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        effective_reward_module,
    )
    base_runner = runner_mod.MotionOnPolicyRunner.__mro__[1]

    def one_update(_self, **_kwargs):
        _self.alg.update()

    monkeypatch.setattr(base_runner, "learn", one_update, raising=False)
    optimizer_calls = []
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(
        unwrapped=env, step=lambda *_a, **_k: None
    )
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 1
    runner.rank = 0
    runner.current_learning_iteration = 0
    runner.alg = types.SimpleNamespace(
        update=lambda: optimizer_calls.append("optimizer")
    )
    runner._effective_reward_activation_task_kind = lambda: "upper_safe"
    runner._action_ball_resume_reset_pending = False
    runner._rollout_update_wrapper_active = False

    with pytest.raises(RuntimeError, match="refusing PPO update"):
        runner.learn(num_learning_iterations=1)
    assert optimizer_calls == []
    artifacts = list(
        (tmp_path / "joint_safety_ledgers").glob("*.prepared.pt")
    )
    assert len(artifacts) == 1
    payload = torch.load(
        artifacts[0], map_location="cpu", weights_only=False
    )
    assert payload["status"] == "fatal_actual_hard_edge"
    assert payload["fatal_flags"]["actual_hard_edge_event_count"] > 0
    assert not list(
        (tmp_path / "joint_safety_ledgers").glob(
            "*.optimizer_commit.json"
        )
    )
    assert action.joint_safety_ledger_snapshot()["since_last_consume"][
        "has_data"
    ] is True


def test_4096_by_24_safe_compact_artifact_stays_within_budget(
    monkeypatch, tmp_path
):
    runner_mod = _load_runner_module(monkeypatch, _load_contract_module())
    action, env, asset = _action_and_env(
        num_envs=4096,
        joint_count=31,
        guard=True,
        guard_policy_dt_s=0.02,
        runtime_step_dt=0.02,
        terminal_archive_capacity=64,
    )
    for _ in range(24):
        action.process_actions(torch.zeros(4096, 31))
        _finish_guarded_policy_step(action, asset)
    runner = runner_mod.MotionOnPolicyRunner.__new__(
        runner_mod.MotionOnPolicyRunner
    )
    runner.env = types.SimpleNamespace(unwrapped=env)
    runner.log_dir = str(tmp_path)
    runner.num_steps_per_env = 24
    runner.rank = 0
    original_prepare = action.prepare_joint_safety_ledger_consume
    prepare_metrics = {}

    def measured_prepare():
        tracemalloc.start()
        start = time.perf_counter()
        token, snapshot = original_prepare()
        elapsed = time.perf_counter() - start
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        prepare_metrics.update(
            elapsed_s=elapsed,
            python_peak_bytes=peak_bytes,
        )
        assert "q" not in snapshot
        assert "qdot" not in snapshot
        assert "hard_lower_gap" not in snapshot
        assert (
            snapshot["identity_bound_policy_steps"][0][
                "minimum_hard_gap"
            ].data_ptr()
            == action._joint_safety_policy_step_summaries[0][
                "minimum_hard_gap"
            ].data_ptr()
        )
        assert (
            snapshot["since_last_consume"][
                "actual_hard_edge_joint_count"
            ].data_ptr()
            == action._joint_safety_accumulator_actual_hard_edge_joint_count.data_ptr()
        )
        return token, snapshot

    action.prepare_joint_safety_ledger_consume = measured_prepare

    record = runner._consume_joint_safety_update(
        0, expected_action_term=action
    )
    assert prepare_metrics["elapsed_s"] < 1.0
    assert prepare_metrics["python_peak_bytes"] < 1024 * 1024
    assert (
        record["artifact"]["size_bytes"]
        <= runner_mod._JOINT_SAFETY_NORMAL_ARTIFACT_MAX_BYTES
    )
    payload = torch.load(
        tmp_path / record["artifact"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    assert (
        payload["budgets"]["core_payload_bytes"]
        <= runner_mod._JOINT_SAFETY_CORE_PAYLOAD_MAX_BYTES
    )
    assert payload["terminal"]["archive_count"] == 0
    assert all(
        step["sparse_counters"][field]["nonzero_cells"] == 0
        for step in payload["policy_steps"]
        for field in (
            "qdes_joint_count",
            "policy_crossing_joint_count",
            "substep_hard_crossing_joint_count",
            "actual_hard_edge_joint_count",
        )
    )


def test_physics_substep_ledger_partial_reset_atomically_clears_rows_and_counters():
    action, _, asset = _action_and_env(guard=True)
    base = hope_actions_mod.JointPositionAction
    if not hasattr(base, "apply_actions"):
        base.apply_actions = lambda self: None
    action.process_actions(torch.zeros(2, 2))
    for index in range(4):
        _set_sim_timestamp(asset, index * 0.025)
        asset.data.joint_pos[:, 0] = torch.tensor([1.21, -1.21])
        asset.data.joint_vel.zero_()
        action.apply_actions()
    _set_sim_timestamp(asset, 0.1)
    action.finalize_joint_safety_post_step_readback()
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [True, True]

    action.reset(env_ids=torch.tensor([0]))
    snapshot = action.joint_safety_ledger_snapshot()
    assert not snapshot["env_valid"][:, 0].any().item()
    assert snapshot["env_valid"][:, 1].all().item()
    assert torch.isnan(snapshot["q"][:, 0]).all().item()
    assert action.physics_substep_actual_hard_edge_latch.tolist() == [False, True]
    assert action.physics_substep_actual_hard_edge_joint_count[0].sum().item() == 0
    assert action.physics_substep_actual_hard_edge_joint_count[1, 0].item() == 5


def test_physics_substep_contract_pins_decimation_times_physics_dt():
    with pytest.raises(ValueError, match=r"physics_dt \* decimation"):
        _action_and_env(
            guard=True,
            decimation=4,
            runtime_physics_dt=0.01,  # 4 * .01 != policy/runtime step_dt .1
        )


def test_actual_joint_position_uses_explicit_soft_envelope_and_travel_margin():
    _, env, asset = _action_and_env()
    # On [-1, 1], 10% of full travel is 0.2 rad per side.  Exact +/-0.8 is forbidden.
    asset.data.joint_pos[:] = torch.tensor([[0.7999, 0.0], [-0.8, 0.0]])
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    result = terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.1,
    )
    assert result.tolist() == [False, True]

    asset.data.joint_pos[0, 0] = float("nan")
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.1,
    ).tolist() == [True, True]


def test_actual_joint_position_limit_source_is_explicit_not_guessed():
    _, env, asset = _action_and_env()
    asset.data.joint_pos[:] = torch.tensor([[1.2, 0.0], [0.0, 0.0]])
    asset.data.joint_pos_limits = _limits(2, 2, lo=-2.0, hi=2.0)
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=slice(None))
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "soft_joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [True, False]
    assert terminations_mod.actual_joint_position_forbidden_zone(
        env,
        asset_cfg,
        "joint_pos_limits",
        0.0,
        0.0,
    ).tolist() == [False, False]


def test_done_terms_reject_reordered_or_partial_joint_contracts():
    action, env, asset = _action_and_env()
    action.process_actions(torch.zeros(2, 2))

    action._joint_names = list(reversed(action._joint_names))
    with pytest.raises(RuntimeError, match="joint-name order"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )
    action._joint_names = list(asset.data.joint_names)
    action._joint_ids = [1, 0]
    with pytest.raises(RuntimeError, match="identity order"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )

    bad_cfg = types.SimpleNamespace(name="robot", joint_ids=[1, 0])
    with pytest.raises(RuntimeError, match="identity order"):
        terminations_mod.actual_joint_position_forbidden_zone(
            env,
            bad_cfg,
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )
    non_integer_cfg = types.SimpleNamespace(name="robot", joint_ids=[0.0, 1.0])
    with pytest.raises(RuntimeError, match="integer joint_ids"):
        terminations_mod.actual_joint_position_forbidden_zone(
            env,
            non_integer_cfg,
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )


def test_done_terms_require_explicit_existing_limit_source_and_exact_runtime_shape():
    action, env, asset = _action_and_env()
    action.process_actions(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="limit_source"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "guessed_limits",
            0.0,
            0.0,
        )

    asset.data.soft_joint_pos_limits = _limits(1, 2)
    with pytest.raises(RuntimeError, match="num_envs, num_joints, 2"):
        terminations_mod.pre_clamp_qdes_forbidden_zone(
            env,
            "joint_pos",
            "soft_joint_pos_limits",
            0.0,
            0.0,
        )
