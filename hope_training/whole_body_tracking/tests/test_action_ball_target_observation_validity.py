"""Fixed-question observation validity tests for 111/101/000 target recipes."""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_reward_flags_mdp import hope_observations_mod  # noqa: E402


class _Command:
    def __init__(self, validity):
        self._validity = tuple(validity)
        self.base_quat_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )
        self.base_pos_w = torch.tensor(
            [[2.0, -3.0, 0.8], [-1.0, 4.0, 1.2]]
        )
        self._target_position_w = torch.tensor(
            [[3.0, -1.0, 1.1], [0.5, 5.5, 0.9]]
        )
        self._position_b = torch.tensor(
            [[0.7, 0.2, 0.3], [-0.1, 0.4, -0.2]]
        )
        self._position_rel_racket_b = torch.tensor(
            [[0.1, -0.2, 0.05], [0.3, 0.1, -0.1]]
        )
        self._velocity_w = torch.tensor(
            [[2.0, -0.5, 0.1], [1.0, 0.25, -0.2]]
        )
        self._face_w = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        self.racket_target_vel_w = self._velocity_w.clone()
        self.racket_normal_raw_w = self._face_w.clone()
        self.target_normal_cmd = self._face_w.clone()
        self.cfg = types.SimpleNamespace(
            face_command=True,
            face_command_pairing="shared_plus_y",
        )

    def action_ball_target_component_valid(self, component):
        return self._validity[{"position": 0, "velocity": 1, "face": 2}[component]]

    def racket_target_pos_b(self):
        # The production command currently already applies a final mask.  Return the unmasked value
        # here to ensure the observation boundary independently enforces its ABI.
        return self._position_b.clone()

    def racket_target_pos_b_rel(self):
        return self._position_rel_racket_b.clone()

    def actor_racket_target_vel_w(self):
        return self._velocity_w.clone()

    def actor_target_normal_cmd(self):
        return self._face_w.clone()

    def racket_target_rel_base_w(self):
        # Reproduce the old 000 leak exactly: upstream masks absolute target to zero, then subtracts
        # base position.  The observation producer must erase this *after* the relative transform.
        absolute = (
            self._target_position_w
            if self.action_ball_target_component_valid("position")
            else torch.zeros_like(self._target_position_w)
        )
        return absolute - self.base_pos_w


def _env(command):
    return types.SimpleNamespace(
        command_manager=types.SimpleNamespace(
            get_term=lambda _name: command
        )
    )


@pytest.mark.parametrize(
    ("validity", "position_valid", "velocity_valid", "face_valid"),
    (
        ((True, True, True), True, True, True),
        ((True, False, True), True, False, True),
        ((False, False, False), False, False, False),
    ),
)
def test_actor_and_critic_target_observations_apply_final_validity_mask(
    validity,
    position_valid,
    velocity_valid,
    face_valid,
):
    command = _Command(validity)
    env = _env(command)

    position_values = (
        hope_observations_mod.racket_target_pos_b(env, "racket_target"),
        hope_observations_mod.racket_target_pos_rel_b(env, "racket_target"),
        hope_observations_mod.racket_target_rel_base(env, "racket_target"),
    )
    expected_positions = (
        command._position_b,
        command._position_rel_racket_b,
        command._target_position_w - command.base_pos_w,
    )
    for value, expected in zip(position_values, expected_positions):
        assert torch.equal(
            value,
            expected if position_valid else torch.zeros_like(expected),
        )

    velocity_values = (
        hope_observations_mod.racket_target_vel_w(env, "racket_target"),
        hope_observations_mod.racket_target_vel_heading(
            env, "racket_target"
        ),
        hope_observations_mod.racket_target_vel_w_live(
            env, "racket_target"
        ),
    )
    for value in velocity_values:
        assert torch.equal(
            value,
            command._velocity_w
            if velocity_valid
            else torch.zeros_like(command._velocity_w),
        )

    expected_face4 = torch.cat(
        (command._face_w, torch.zeros(2, 1)), dim=-1
    )
    face_values = (
        hope_observations_mod.racket_target_normal_cmd(
            env, "racket_target"
        ),
        hope_observations_mod.racket_target_normal_cmd_heading(
            env, "racket_target"
        ),
    )
    for value in face_values:
        assert torch.equal(
            value,
            expected_face4 if face_valid else torch.zeros_like(expected_face4),
        )
    critic_face = hope_observations_mod.racket_target_normal_w(
        env, "racket_target"
    )
    assert torch.equal(
        critic_face,
        command._face_w
        if face_valid
        else torch.zeros_like(command._face_w),
    )


def test_000_world_relative_position_does_not_leak_negative_base_position():
    command = _Command((False, False, False))
    leaked_upstream = command.racket_target_rel_base_w()
    assert torch.equal(leaked_upstream, -command.base_pos_w)

    observed = hope_observations_mod.racket_target_rel_base(
        _env(command), "racket_target"
    )
    assert torch.equal(observed, torch.zeros_like(leaked_upstream))
