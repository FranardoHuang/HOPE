"""C225-only causal strike and achieved-flight rewards.

These terms deliberately consume no desired racket contact position, velocity,
or face.  The strike bridge compares the causal incoming-ball centre with the
achieved centre of the selected rubber face at the single nominal strike tick.
The landing term grades only the analytic flight produced after an actual
selected-rubber contact; it is not an observed physical-ball outcome.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def action_ball_task_valid_mask(cmd) -> torch.Tensor:
    """Lazy bridge to the shared A/C eligibility helper.

    Keeping the import lazy preserves this reward module's dependency-light
    CPU tensor tests while runtime still has exactly one validation authority.
    """

    from .hope_rewards import action_ball_task_valid_mask as shared_mask

    return shared_mask(cmd)


def _selected_face_center_w(cmd) -> tuple[torch.Tensor, torch.Tensor]:
    """Return achieved selected-rubber centre and its finite-state mask.

    This is the position-only subset of the canonical exact-face geometry.  It
    imports the measured face offsets from that authority and performs no
    target reconstruction or inverse solve.
    """

    from . import racket_contact_geometry as geometry

    site = cmd.racket_pos_w
    quaternion = cmd.racket_quat_w
    action = cmd._action_ball_attempt_action
    if site.ndim != 2 or site.shape[-1] != 3:
        raise RuntimeError("C225 achieved racket site must have shape [num_envs,3]")
    if quaternion.shape != (site.shape[0], 4):
        raise RuntimeError("C225 achieved racket quaternion must have shape [num_envs,4]")
    if action.shape != (site.shape[0],):
        raise RuntimeError("C225 attempt action must have shape [num_envs]")
    if not cmd._action_ball_mount_signs:
        raise RuntimeError("C225 selected-rubber mount signs are unavailable")

    sign_table = torch.as_tensor(
        cmd._action_ball_mount_signs,
        device=site.device,
        dtype=site.dtype,
    )
    valid_action = (action >= 0) & (action < sign_table.numel())
    sign = sign_table[action.clamp(min=0, max=sign_table.numel() - 1)]
    x, z = geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M
    face_local = torch.stack(
        (
            torch.full_like(sign, float(x)),
            torch.where(
                sign > 0.0,
                torch.full_like(sign, float(geometry.RED_OUTER_Y_FROM_SITE_M)),
                torch.full_like(sign, float(geometry.BLACK_OUTER_Y_FROM_SITE_M)),
            ),
            torch.full_like(sign, float(z)),
        ),
        dim=-1,
    )

    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    finite = (
        valid_action
        & torch.isfinite(site).all(dim=-1)
        & torch.isfinite(quaternion).all(dim=-1)
        & torch.isfinite(norm[:, 0])
        & (norm[:, 0] > 1.0e-12)
    )
    q = quaternion / norm.clamp_min(1.0e-12)
    qv = q[:, 1:]
    uv = torch.cross(qv, face_local, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    face_offset_w = face_local + 2.0 * (q[:, :1] * uv + uuv)
    face_center_w = site + face_offset_w
    finite &= torch.isfinite(face_center_w).all(dim=-1)
    return face_center_w, finite


def c225_strike_ball_paddle_center_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.15,
) -> torch.Tensor:
    """One-shot Cauchy shaping on achieved paddle-centre/ball-centre distance.

    Eligibility is exactly one nominal strike tick per active swing.  It does
    not require contact, so a miss retains a bounded, non-zero tail.  The ball
    centre is the immutable incoming question propagated to contact time; the
    paddle centre is reconstructed only from the achieved site pose and the
    action's fixed selected-rubber mount sign.
    """

    scale = float(std)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("C225 strike proximity std must be finite and positive")
    cmd = _command(env, command_name)
    exact = cmd.metrics.get("exact_strike_hit_rate")
    if not isinstance(exact, torch.Tensor) or exact.shape != (cmd.racket_pos_w.shape[0],):
        raise RuntimeError("C225 strike proximity requires exact_strike_hit_rate [num_envs]")
    active = cmd._action_ball_attempt_active
    if not isinstance(active, torch.Tensor) or active.shape != exact.shape:
        raise RuntimeError("C225 strike proximity requires an active-attempt mask")
    ball_center_w = cmd._action_ball_ball_contact_target_w
    if ball_center_w.shape != cmd.racket_pos_w.shape:
        raise RuntimeError("C225 incoming ball centre must match achieved racket batch shape")

    face_center_w, finite_face = _selected_face_center_w(cmd)
    finite = finite_face & torch.isfinite(ball_center_w).all(dim=-1)
    distance = torch.linalg.vector_norm(face_center_w - ball_center_w, dim=-1)
    kernel = 1.0 / (1.0 + torch.square(distance / scale))
    eligible = (
        (exact > 0.5)
        & active
        & action_ball_task_valid_mask(cmd)
        & finite
        & torch.isfinite(kernel)
    )
    return torch.where(eligible, kernel, torch.zeros_like(kernel))


def c225_landing_outcome_actual_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    mode: str = "legal_base",
    base_frac: float = 0.6,
    off_table_frac: float = 0.5,
    settle_delay_s: float = 0.0,
) -> torch.Tensor:
    """Grade predicted landing only after a valid achieved selected-rubber hit.

    A legal opponent-table landing receives ``base_frac + (1-base)*kernel``.
    An opponent-side landing outside the table receives at most
    ``off_table_frac*kernel``.  Misses, invalid/non-finite flights, own-side or
    backwards landings, and trajectories that do not cross and clear the net
    receive zero.  The current C225 task has no physical observed-ball channel,
    so this function must never be described as observed outcome evidence.
    """

    if mode != "legal_base":
        raise ValueError("C225 landing mode must be 'legal_base'")
    base = float(base_frac)
    off = float(off_table_frac)
    delay = float(settle_delay_s)
    if not math.isfinite(base) or not 0.0 < base < 1.0:
        raise ValueError("C225 landing base_frac must be in (0,1)")
    if not math.isfinite(off) or not 0.0 < off < base:
        raise ValueError("C225 landing off_table_frac must be in (0,base_frac)")
    if not math.isfinite(delay) or delay != 0.0:
        raise ValueError("C225 rollout-zero landing requires settle_delay_s=0")

    cmd = _command(env, command_name)
    target_xy = getattr(cmd, "_vb_target_xy_per_env", None)
    if target_xy is None:
        target_xy = cmd._vb_target_xy.unsqueeze(0)
    landing_xy = cmd.vb_landing_xy
    if (
        target_xy.ndim != 2
        or target_xy.shape[1:] != landing_xy.shape[1:]
        or target_xy.shape[0] not in (1, landing_xy.shape[0])
    ):
        raise RuntimeError(
            "C225 landing target must be broadcastable as [1,2] or match [num_envs,2]"
        )
    sigma = float(cmd.cfg.vb_landing_sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("C225 landing sigma must be finite and positive")

    dist2 = torch.sum(torch.square(landing_xy - target_xy), dim=-1)
    kernel = torch.exp(-dist2 / (sigma**2))
    finite = torch.isfinite(landing_xy).all(dim=-1) & torch.isfinite(kernel)
    achieved_flight = (
        cmd.vb_fired
        & action_ball_task_valid_mask(cmd)
        & cmd.vb_landing_valid
        & cmd.vb_net_crossed
        & cmd.vb_net_clear
        & finite
    )
    opponent_plane = landing_xy[:, 0] > float(cmd._vb_net_x)
    legal = achieved_flight & opponent_plane & cmd.vb_on_opponent
    off_table = achieved_flight & opponent_plane & ~cmd.vb_on_opponent
    legal_value = base + (1.0 - base) * kernel
    off_table_value = off * kernel
    return torch.where(
        legal,
        legal_value,
        torch.where(off_table, off_table_value, torch.zeros_like(kernel)),
    )
