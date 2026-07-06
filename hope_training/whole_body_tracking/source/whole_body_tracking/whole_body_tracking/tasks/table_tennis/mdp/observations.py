"""Observation terms for the table-tennis environment (ball state in the robot base frame)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_position_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position relative to the robot base, expressed in the robot base frame. Shape ``(N, 3)``."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    rel_w = ball.data.root_pos_w - robot.data.root_pos_w
    return quat_rotate_inverse(robot.data.root_quat_w, rel_w)


def ball_velocity_b(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball linear velocity expressed in the robot base frame. Shape ``(N, 3)``."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    return quat_rotate_inverse(robot.data.root_quat_w, ball.data.root_lin_vel_w)


def ball_predicted_landing(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Predicted landing of the robot's outgoing shot, ``(N, 3) = [x_hope, y_hope, valid]``.

    Computed by :class:`~..table_tennis_env.TableTennisEnv` at the instant of a racket hit (forward-
    integrating the mocap-fitted flight model). Persists until the next hit / reset. ``valid`` is 1.0
    when a table-plane crossing was found. Privileged signal (use in the critic group). Returns zeros if
    the spin-aware physics layer is inactive (e.g. aerodynamics disabled)."""
    if not getattr(env, "_physics_ready", False):
        return torch.zeros(env.num_envs, 3, device=env.device)
    xy = env._predicted_landing_xy
    valid = env._predicted_landing_valid.float().unsqueeze(-1)
    return torch.cat([xy, valid], dim=-1)
