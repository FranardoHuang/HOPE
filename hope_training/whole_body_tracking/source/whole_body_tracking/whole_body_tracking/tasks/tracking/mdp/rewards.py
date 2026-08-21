from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _apply_window_scale(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    window_scale: float,
    window_command_name: str | None,
    motion_command: MotionCommand,
) -> torch.Tensor:
    """V2 in-window imitation yield (reward_staged_design 2026-07-08 §③): scale the imitation
    reward by ``window_scale`` for envs inside the racket command's WIDE strike window (the window
    where the question-bank answer takes over), full value outside. 人话:触球窗内老师闭嘴(或小
    声),听题目的。``window_command_name=None`` (the default for every term) or scale 1.0 is a
    strict no-op — the default reward path is byte-identical. The mask is looked up BY NAME at
    call time so this base-BeyondMimic module never imports the HOPE racket command."""
    if window_command_name is None or window_scale == 1.0:
        return reward
    cmd = env.command_manager.get_term(window_command_name)
    window = getattr(cmd, "strike_window_wide", None)
    if window is None:  # racket command without split-window support: fall back to the base window
        window = cmd.strike_window
    motion_command.record_v2_strike_window_scale_activation(
        window, actual_window_scale=float(window_scale)
    )
    return torch.where(window, reward * float(window_scale), reward)


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float,
    window_scale: float = 1.0, window_command_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return _apply_window_scale(
        env, torch.exp(-error / std**2), window_scale, window_command_name, command
    )


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float,
    window_scale: float = 1.0, window_command_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return _apply_window_scale(
        env, torch.exp(-error / std**2), window_scale, window_command_name, command
    )


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None,
    window_scale: float = 1.0, window_command_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return _apply_window_scale(
        env,
        torch.exp(-error.mean(-1) / std**2),
        window_scale,
        window_command_name,
        command,
    )


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None,
    window_scale: float = 1.0, window_command_name: str | None = None,
    coarse_std: float | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    mean_error = error.mean(-1)
    reward = torch.exp(-mean_error / std**2)
    if coarse_std is not None:
        if not math.isfinite(coarse_std) or coarse_std <= std:
            raise ValueError("coarse_std must be finite and greater than std")
        # The narrow kernel preserves close-pose precision.  The broad kernel
        # keeps a usable gradient during the ready-to-frame0 transition rather
        # than letting one large all-body RMS error erase the whole term.
        reward = 0.5 * (
            reward + torch.exp(-mean_error / coarse_std**2)
        )
    return _apply_window_scale(
        env,
        reward,
        window_scale,
        window_command_name,
        command,
    )


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None,
    window_scale: float = 1.0, window_command_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    command.record_v1_velocity_mimic_activation(
        reward.shape[0],
        held_wrist_excluded=(
            body_names is not None
            and "right_wrist_yaw_Link" not in body_names
        ),
    )
    return _apply_window_scale(
        env, reward, window_scale, window_command_name, command
    )


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None,
    window_scale: float = 1.0, window_command_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return _apply_window_scale(
        env,
        torch.exp(-error.mean(-1) / std**2),
        window_scale,
        window_command_name,
        command,
    )


def terminated_by_term(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor:
    """1.0 on the step ONE named termination fired, 0.0 otherwise.

    The narrow sibling of IsaacLab's ``is_terminated`` (which charges for every terminal reason at
    once, and is what ``death_penalty`` uses).  This one is addressed to a single term so its
    penalty can be priced, ablated and read off independently — e.g. ``table_hit_penalty`` against
    ``robot_hit_table``.  Time-outs are excluded exactly as ``is_terminated`` excludes them,
    because ``TerminationManager.get_term`` reports the term's own mask and every non-``time_out``
    term here has ``time_out=False``.

    Fails loud on a missing term: a silently-zero penalty for a termination that is not actually
    wired would look identical to "the robot never hits the table".
    """
    tm = env.termination_manager
    if term_name not in tuple(tm.active_terms):
        raise RuntimeError(
            f"terminated_by_term({term_name!r}) but that termination is not active; "
            f"active terms are {tuple(tm.active_terms)}"
        )
    return tm.get_term(term_name).float()


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
