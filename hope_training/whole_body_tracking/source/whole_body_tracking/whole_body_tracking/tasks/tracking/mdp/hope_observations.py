"""HOPE racket-target observation terms.

These wrap :class:`RacketTargetCommand`. The actor (policy) group should use only the *desired*
quantities the planner provides at deploy time (HITTER actor observation, Table I):

* :func:`racket_target_pos_b`  — desired racket position relative to base (3)
* :func:`racket_target_vel_w`  — desired racket velocity in world frame (3)
* :func:`time_to_strike`       — time remaining until strike (1)
* :func:`base_target_pos_b`    — desired base XY position relative to base (2)

The desired racket *normal* and the *actual* racket state are privileged/critic-only or used by
the reward; they are intentionally NOT in the HITTER actor observation (the racket is never sensed
on hardware). :func:`swing_type` is provided for a unified forehand+backhand policy variant; the
HOPE default trains separate policies and does not need it.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand
from whole_body_tracking.tasks.tracking.mdp.stage1_question_bank import face_command_obs_vector

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


# --- actor (policy) observations: desired targets only ------------------------------------ #
def racket_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket pos rel-base (yaw frame). PRIVILEGED — uses world base position (`full` mode)."""
    return _cmd(env, command_name).racket_target_pos_b()


def racket_target_pos_rel_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket pos relative to the CURRENT racket (FK), yaw frame. DEPLOY-HONEST (no world
    base position; see :meth:`RacketTargetCommand.racket_target_pos_b_rel`). Used by the deploy-parity
    actor contract (legacy task name: `real_sensor_only`). A1: reads the ACTOR-visible target view
    (delayed/jittered when target latency is on; the live tensor otherwise)."""
    return _cmd(env, command_name).racket_target_pos_b_rel()


def racket_target_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket velocity, world frame. ACTOR term — A1: reads the ACTOR-visible view
    (delayed/jittered when target latency is on; the live tensor otherwise, byte-identical).
    The critic uses :func:`racket_target_vel_w_live`."""
    return _cmd(env, command_name).actor_racket_target_vel_w()


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Time remaining until the strike (s). NOT delayed by A1 target latency ON PURPOSE: the swing
    clock is generated robot-side by the deploy runner, not by the mocap link, so it carries no
    mocap/planner transport latency."""
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def base_target_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).base_target_pos_b()


def swing_type(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Forehand (+1) / backhand (-1). Only needed for a unified (single) policy. A1: delayed with
    the target when latency is on (the flag rides the same planner->runner message as the target)."""
    return _cmd(env, command_name).actor_swing_sign().unsqueeze(-1)


def racket_target_normal_cmd(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Stage-1 face-command channel, 4-D per env: [DEMANDED face normal (3, world frame, question
    bank / StrikeSpec n), rho placeholder (1, zero-filled)]. rho is the S3 spin-lane scalar,
    reserved now so the layout matches the frozen contract-day 175 -> 179 decision and no ladder
    retrain is needed later. NOT in the frozen 175-D contract — only wired into the actor when
    racket.face_command_obs is enabled. Normal is zeros when the question bank is off (the buffer
    always exists)."""
    return face_command_obs_vector(_cmd(env, command_name).target_normal_cmd)


# --- privileged (critic) observations: desired normal + actual racket state --------------- #
def racket_target_vel_w_live(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """TRUE live desired racket velocity (world). CRITIC/privileged term: the asymmetric critic
    keeps the undegraded target even when the actor's view is delayed/jittered (A1). Identical to
    :func:`racket_target_vel_w` when the A1 knobs are off."""
    return _cmd(env, command_name).racket_target_vel_w


def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _cmd(env, command_name).racket_target_normal_w


def racket_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket position relative to base (FK). Privileged — not sensed on hardware."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_pos_w - cmd.base_pos_w)


def racket_lin_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_lin_vel_w


def racket_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_normal_w


def episode_time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in the episode (seconds). HITTER critic privileged input."""
    # IsaacLab 2.1 calls observation terms once during ObservationManager._prepare_terms (dimension
    # probe) BEFORE ManagerBasedRLEnv allocates episode_length_buf — fall back to zeros there.
    buf = getattr(env, "episode_length_buf", None)
    if buf is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    left = (env.max_episode_length - buf).float() * env.step_dt
    return left.unsqueeze(-1)
