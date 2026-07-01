"""HOPE goal-tracking reward terms (HITTER r_goal).

These implement the racket/base target tracking rewards on top of the BeyondMimic imitation
reward (``r_imitation``, the ``motion_*`` terms already in ``rewards.py``) and the regularization
reward (``r_regularization``, ``action_rate_l2`` / ``joint_torques_l2`` / contact penalties).

Activation timing follows HITTER: the base-position reward is active **before** the strike; the
racket position/velocity/normal rewards are active only in a **short window around** the strike.
Because a ``RewardTermCfg`` weight is constant, the time gating is applied *inside* each term by
multiplying the exponential kernel by the command's ``pre_strike`` / ``strike_window`` mask.

The exponential kernel form (``exp(-error/std**2)``) mirrors the BeyondMimic motion-tracking
rewards. HITTER does not publish reward weights or kernel forms, so the weights in the env config
are HOPE choices to be tuned, not paper-sourced values.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def _dbg_log(cmd: RacketTargetCommand, name: str, raw: torch.Tensor, mask: torch.Tensor) -> None:
    """Log raw (pre-mask) and gated (post-mask) kernel values, held over the active mask.

    No-op unless ``cmd.cfg.debug_reward_logging`` is set. The held value lets the reset-mean report the
    in-window reward, and lets you see how much reward the time-gate is killing (gated vs raw) and whether
    the raw kernel still has any gradient at the current error scale (raw ~0 => std too tight).
    """
    if not cmd.cfg.debug_reward_logging:
        return
    cmd.metrics[f"dbg_{name}_raw"] = torch.where(mask, raw, cmd.metrics[f"dbg_{name}_raw"])
    cmd.metrics[f"dbg_{name}_gated"] = torch.where(mask, raw * mask.float(), cmd.metrics[f"dbg_{name}_gated"])


def racket_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket center position near strike using the target's swing-through trajectory."""
    cmd = _cmd(env, command_name)
    target_pos_now = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_pos_now), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "racket_pos", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def racket_position_tracking_static_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Ablation B: track the strike POINT itself (no swing-through), decoupling position from timing/velocity.

    Identical gating to ``racket_position_tracking_exp`` but compares against the bare ``racket_target_pos_w``
    instead of the moving swing-through point ``target - vel*t_to_strike``. Over a ±0.15 s window the
    swing-through point sweeps up to ~0.9 m at a 6 m/s target, so the standard term mostly rewards being on
    the moving line (timing/velocity); this variant gives a clean "get the paddle to the point" signal for
    early stable positioning. Select via ``rewards.racket_position_static: true`` in the task YAML.
    """
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_pos_w - cmd.racket_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "racket_pos", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def racket_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket linear velocity near the strike time (FK actual vs desired, world frame)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_target_vel_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "racket_vel", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def racket_normal_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track racket face-normal orientation near the strike time. ``std`` is in radians."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    raw = torch.exp(-(angle**2) / std**2)
    _dbg_log(cmd, "racket_normal", raw, cmd.strike_window)
    return raw * cmd.strike_window.float()


def base_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track desired base XY position before the strike (encourages repositioning footwork)."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w), dim=-1)
    raw = torch.exp(-error / std**2)
    _dbg_log(cmd, "base", raw, cmd.pre_strike)
    return raw * cmd.pre_strike.float()


def pre_strike_foot_slip(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize horizontal foot speed WHILE the foot is in contact, BEFORE the strike only.

    The robot was sliding/leaning to reach far racket targets while the base reward pinned it near spawn
    (foot_slip_speed high, foot_contact_frac low). This term teaches it to plant its feet and stabilize
    during the approach. It is gated by ``pre_strike`` ONLY (not ``strike_window``), so the strike swing's
    footwork is untouched. ``foot_slip_in_contact`` (sum over feet of horizontal speed * in_contact) is
    precomputed by the RacketTargetCommand each step (0 if the feet/contact sensor cannot be resolved).
    Returns a positive magnitude; the RewTerm weight is negative.
    """
    cmd = _cmd(env, command_name)
    return cmd.foot_slip_in_contact * cmd.pre_strike.float()


# ============================================================================================== #
# Footwork-to-strike (BASE-FREE). The legs move because moving the body REDUCES the racket->target
# distance (racket_progress), not because they track a base target. Footwork is penalized for being
# BAD (slip / drag / violent / unstable at strike), NOT for stepping — the feet are free to move.
# ============================================================================================== #
def racket_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """DENSE pre-strike reward for reducing the racket->target distance (prev - current, clamped). This
    is the base-free driver of whole-body footwork: the legs/waist/arms all get credit for moving the
    racket closer to the target, with NO base-position target. Gated to pre_strike (approach phase); the
    strike swing itself is scored by the racket pos/vel/normal terms. Positive when approaching; RewTerm
    weight is POSITIVE."""
    cmd = _cmd(env, command_name)
    return cmd.racket_progress * cmd.pre_strike.float()


def racket_strike_success(
    env: ManagerBasedRLEnv, command_name: str, std_pos: float, std_vel: float, std_normal: float
) -> torch.Tensor:
    """MULTIPLICATIVE strike success R_pos * R_vel * R_normal, gated to the strike window. Unlike the
    additive racket terms (which give partial credit for getting only position OR velocity right), the
    product is high ONLY when position AND velocity AND normal are all good at once — a true hit. RewTerm
    weight is POSITIVE."""
    # Each kernel already multiplies by strike_window internally, so the product is non-zero ONLY in the
    # window (no extra gate needed). The product is high only when pos AND vel AND normal are all good.
    rp = racket_position_tracking_exp(env, command_name, std_pos)
    rv = racket_velocity_tracking_exp(env, command_name, std_vel)
    rn = racket_normal_tracking_exp(env, command_name, std_normal)
    return rp * rv * rn


# --- footwork penalties (feet may STEP; we only punish BAD foot behaviour) --------------------- #
def foot_slip_sq(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot slip while in contact: sum over feet of contact * ||foot_xy_velocity||² (always on).
    A planted/landing foot should not skate. Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_slip_sq


def foot_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize excessive/violent foot velocity: sum over feet of ||foot_velocity||². Lets the foot step
    but discourages flailing. Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_vel_sq


def foot_drag(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot dragging: lateral foot speed while the foot is near the ground (skimming instead of
    lifting cleanly to step). Positive magnitude; RewTerm weight is negative."""
    return _cmd(env, command_name).foot_drag


def arm_overreach(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Anti-arm-only: penalize solving the target by maxing the arm out — fraction of ARM joints within
    10% of a position limit. Encourages using the body/legs to bring the target into a comfortable arm
    range instead of stretching. Positive in [0,1]; RewTerm weight is negative."""
    return _cmd(env, command_name).arm_overreach_frac


# --- strike-window stability (penalize wobble/bob/skate AT the hit; gated to the strike window) - #
def strike_proj_grav_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize base tilt (||projected_gravity_xy||) DURING the strike window — be upright at the hit."""
    cmd = _cmd(env, command_name)
    return cmd.proj_grav_xy * cmd.strike_window.float()


def strike_base_ang_vel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize base roll/pitch rate (||base_ang_vel_xy||) DURING the strike window."""
    cmd = _cmd(env, command_name)
    return cmd.base_ang_vel_xy_norm * cmd.strike_window.float()


def strike_foot_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize foot motion (sum ||foot_velocity||²) DURING the strike window — plant for the hit."""
    cmd = _cmd(env, command_name)
    return cmd.foot_vel_sq * cmd.strike_window.float()


def strike_vertical_bob(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize vertical base velocity (|base_lin_vel_z|) DURING the strike window — no bob at the hit."""
    cmd = _cmd(env, command_name)
    return cmd.vertical_speed * cmd.strike_window.float()
