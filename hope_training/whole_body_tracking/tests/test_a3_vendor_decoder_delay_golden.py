"""Golden for the vendor A3 decoder composed with control-step action delay.

The A3 task does not define a second decoder class: its live action term is
``ClampedJointPositionAction`` with Isaac Lab's ``JointPositionAction`` affine
map configured by ``AGIBOT_A3_ACTION_SCALE``.  The dependency-light fixture
used by the action tests supplies that Isaac Lab base class; this test keeps the
HOPE action term and its production delay queue intact and installs the scale
row resolved from the production robot declaration.

Run on the training pod only::

    python -m pytest \
      hope_training/whole_body_tracking/tests/test_a3_vendor_decoder_delay_golden.py -q
"""

from __future__ import annotations

import types

import torch

from test_agibot_a3_vendor_training_authority import (
    _load_actuator_contract,
    _resolve,
)
from test_reward_flags_mdp import hope_actions_mod as A


# Actual A3 USD articulation/action order.  This is deliberately not the
# logical/controller CSV order in ``AGIBOT_A3_JOINT_NAMES``.
JOINT_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


# Independent vendor literals in the exact action order: (effort, nominal Kp).
# Expected values below are computed only as 0.25 * effort / Kp; they never read
# AGIBOT_A3_ACTION_SCALE or any live action-term tensor.
_LITERAL_EFFORT_KP = (
    (220.0, 80.0),
    (220.0, 80.0),
    (220.0, 85.0),
    (220.0, 120.0),
    (220.0, 120.0),
    (46.0, 50.0),
    (220.0, 80.0),
    (220.0, 80.0),
    (118.0, 50.0),
    (320.0, 250.0),
    (320.0, 250.0),
    (6.0, 40.0),
    (60.0, 40.0),
    (60.0, 40.0),
    (118.2, 50.0),
    (118.2, 50.0),
    (6.0, 40.0),
    (60.0, 40.0),
    (60.0, 40.0),
    (54.75, 50.0),
    (54.75, 50.0),
    (24.0, 30.0),
    (24.0, 30.0),
    (24.0, 30.0),
    (24.0, 30.0),
    (24.0, 30.0),
    (24.0, 30.0),
    (6.0, 20.0),
    (6.0, 20.0),
    (6.0, 20.0),
    (6.0, 20.0),
)


def _literal_expected_scale() -> torch.Tensor:
    return torch.tensor(
        [0.25 * effort / kp for effort, kp in _LITERAL_EFFORT_KP],
        dtype=torch.float32,
    )


def _production_scale_row() -> torch.Tensor:
    _, scale_patterns = _load_actuator_contract()
    return torch.tensor(
        [_resolve(scale_patterns, joint) for joint in JOINT_ORDER],
        dtype=torch.float32,
    )


def _ready_default() -> torch.Tensor:
    # Non-zero, non-symmetric ready offsets make offset loss/order swaps visible.
    index = torch.arange(len(JOINT_ORDER), dtype=torch.float32)
    return (
        0.003 * index
        - 0.041 * torch.where(index.remainder(2) == 0, 1.0, -1.0)
    ).unsqueeze(0)


def _action(*, lag_steps: int):
    default = _ready_default()
    limits = torch.stack(
        (
            torch.full_like(default, -4.0),
            torch.full_like(default, 4.0),
        ),
        dim=-1,
    )
    asset = types.SimpleNamespace(
        joint_names=JOINT_ORDER,
        data=types.SimpleNamespace(
            joint_names=JOINT_ORDER,
            default_joint_pos=default.clone(),
            soft_joint_pos_limits=limits,
        ),
    )
    cfg = types.SimpleNamespace(
        asset_name="robot",
        # The dependency-light Isaac Lab stub accepts a scalar at construction.
        # Replace its resolved runtime row below before any action is processed.
        scale=1.0,
        use_default_offset=True,
        clamp=True,
        control_step_action_delay_min=lag_steps,
        control_step_action_delay_max=lag_steps,
    )
    env = types.SimpleNamespace(scene={"robot": asset}, num_envs=1, device="cpu")
    term = A.ClampedJointPositionAction(cfg, env)
    term._scale = _production_scale_row().unsqueeze(0)
    manager = types.SimpleNamespace(
        _action=torch.zeros(1, len(JOINT_ORDER)),
        _prev_action=torch.zeros(1, len(JOINT_ORDER)),
    )
    manager.get_term = lambda name: term if name == "joint_pos" else None
    env.action_manager = manager
    return term


def _sentinel(step: int) -> torch.Tensor:
    index = torch.arange(len(JOINT_ORDER), dtype=torch.float32)
    sign = torch.where(index.remainder(3) == 0, -1.0, 1.0)
    # Every axis and every step is distinct; no symmetric row can hide a swap.
    return (sign * (0.013 * (index + 1.0) + 0.071 * step)).unsqueeze(0)


def _expected_qdes(default: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    return default + _literal_expected_scale().unsqueeze(0) * raw


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _assert_scale(joint: str, *, effort: float, kp: float) -> None:
    index = JOINT_ORDER.index(joint)
    _assert_exact(
        _literal_expected_scale()[index],
        torch.tensor(0.25 * effort / kp, dtype=torch.float32),
    )


def test_vendor_decoder_zero_lag_executes_each_identity_ordered_row_immediately():
    action = _action(lag_steps=0)
    action.reset()
    default = action._asset.data.default_joint_pos.clone()

    # Scale authority and the expected decoder are intentionally independent.
    _assert_exact(action._scale, _literal_expected_scale().unsqueeze(0))
    assert action._joint_names == list(JOINT_ORDER)
    _assert_scale("waist_yaw_joint", effort=220.0, kp=85.0)
    _assert_scale("waist_roll_joint", effort=46.0, kp=50.0)
    _assert_scale("waist_pitch_joint", effort=118.0, kp=50.0)
    _assert_scale("head_yaw_joint", effort=6.0, kp=40.0)
    _assert_scale("head_pitch_joint", effort=6.0, kp=40.0)
    for side in ("left", "right"):
        for axis, effort, kp in (
            ("pitch", 220.0, 80.0),
            ("roll", 220.0, 120.0),
            ("yaw", 220.0, 80.0),
        ):
            _assert_scale(f"{side}_hip_{axis}_joint", effort=effort, kp=kp)
        _assert_scale(f"{side}_knee_joint", effort=320.0, kp=250.0)
        _assert_scale(f"{side}_ankle_pitch_joint", effort=118.2, kp=50.0)
        _assert_scale(f"{side}_ankle_roll_joint", effort=54.75, kp=50.0)
        _assert_scale(f"{side}_shoulder_pitch_joint", effort=60.0, kp=40.0)
        _assert_scale(f"{side}_shoulder_roll_joint", effort=60.0, kp=40.0)
        _assert_scale(f"{side}_shoulder_yaw_joint", effort=24.0, kp=30.0)
        _assert_scale(f"{side}_elbow_joint", effort=24.0, kp=30.0)
        _assert_scale(f"{side}_wrist_roll_joint", effort=24.0, kp=30.0)
        for axis in ("pitch", "yaw"):
            _assert_scale(f"{side}_wrist_{axis}_joint", effort=6.0, kp=20.0)

    for step in range(1, 4):
        raw = _sentinel(step)
        action.process_actions(raw)
        _assert_exact(action.processed_actions, _expected_qdes(default, raw))


def test_vendor_decoder_two_step_lag_is_episode_fixed_and_reset_clears_old_qdes():
    action = _action(lag_steps=2)
    action.reset()
    default = action._asset.data.default_joint_pos.clone()
    lag_at_birth = action.control_step_action_delay_lag_steps.clone()

    assert lag_at_birth.tolist() == [2]
    assert lag_at_birth.shape == (1,)  # one lag for the complete 31-joint row
    contract = action.control_step_action_delay_contract()
    assert contract["sample_timing"] == "once_per_episode_reset"
    assert contract["shared_across_all_31_joints"] is True

    first, second, third, fourth = (_sentinel(step) for step in range(1, 5))
    expected_due = (
        torch.zeros_like(first),
        torch.zeros_like(first),
        first,
        second,
    )
    for raw, due in zip((first, second, third, fourth), expected_due):
        action.process_actions(raw)
        _assert_exact(action.processed_actions, _expected_qdes(default, due))
        assert torch.equal(action.control_step_action_delay_lag_steps, lag_at_birth)

    # Retire an episode with non-zero commands queued, then prove that the new
    # episode emits only the ready/default q_des until its own two-step history
    # matures.  No old actor row or old q_des may cross the reset boundary.
    action.reset()
    assert torch.equal(action.control_step_action_delay_lag_steps, lag_at_birth)
    assert torch.count_nonzero(action._policy_action_delay._history).item() == 0

    new_first, new_second, new_third = (_sentinel(step) for step in range(11, 14))
    action.process_actions(new_first)
    _assert_exact(action.processed_actions, default)
    action.process_actions(new_second)
    _assert_exact(action.processed_actions, default)
    action.process_actions(new_third)
    _assert_exact(action.processed_actions, _expected_qdes(default, new_first))
