"""Reward terms for the table-tennis environment.

Only a small example ball-aware term lives here; the generic robot rewards (alive, action-rate, ...) are
reused from ``isaaclab.envs.mdp``. Add real match objectives (return success, ball-over-net, landing in
the opponent half, racket-to-ball tracking) here as the policy is developed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_above_surface(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """1.0 while the ball is above the table surface (HOPE z > 0), else 0.0. Shape ``(N,)``.

    A placeholder "ball in play" signal demonstrating how to read ball state in the HOPE frame
    (subtract the per-environment origin) for reward shaping."""
    ball: RigidObject = env.scene[asset_cfg.name]
    z_hope = ball.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z_hope > 0.0).float()


def landing_in_opponent_half(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (2.055, -0.7625),  # P2 half centre (HOPE frame)
    sigma: float = 0.3,
    in_bounds_bonus: float = 1.0,
    require_net_clearance: bool = True,
    net_clear_z: float = 0.1725,  # ball centre above which it clears the net (net top 0.1525 + ball R 0.02)
) -> torch.Tensor:
    """One-shot reward at each racket hit for the predicted landing of the outgoing shot. Shape ``(N,)``.

    Uses the landing point that :class:`~..table_tennis_env.TableTennisEnv` predicts at the hit instant
    (mocap-fitted flight model): a Gaussian on the distance to ``target_xy`` plus a bonus when the shot
    is predicted to land in-bounds on the opponent half. Fires only on the control step in which a hit
    occurred. The one-shot ``_hit_event`` is consumed once per step by ``TableTennisEnv.step`` (NOT here),
    so this and other per-hit terms may all read it. Returns zeros if the spin-aware physics is inactive.

    ``require_net_clearance`` (default on) makes this reward the *legal-return* objective rather than pure
    landing geometry: the RK4 landing predictor flies straight THROUGH the net plane (it ignores the net
    collider), so a shot that will actually hit the net can still have a "good" predicted landing. Gating
    on the predicted net crossing (``net_valid`` and centre height ``> net_clear_z``) zeroes the landing
    reward for any shot that would not clear the net — so the policy cannot farm landing reward with shots
    that hit the net. Pair with :func:`pass_net_margin` (the shaping gradient toward clearing)."""
    if not getattr(env, "_physics_ready", False):
        return torch.zeros(env.num_envs, device=env.device)

    hit = env._hit_event
    xy = env._predicted_landing_xy
    target = torch.tensor(target_xy, device=xy.device, dtype=xy.dtype)
    shaped = torch.exp(-torch.sum((xy - target) ** 2, dim=-1) / (sigma * sigma))

    valid = hit & env._predicted_landing_valid
    opp = hit & env._predicted_landing_in_opp
    if require_net_clearance:
        clears = env._predicted_net_valid & (env._predicted_net_z > net_clear_z)
        valid = valid & clears
        opp = opp & clears

    reward = torch.where(valid, shaped, torch.zeros_like(shaped))
    reward = reward + in_bounds_bonus * opp.float()
    return reward


def pass_net_margin(
    env: "ManagerBasedRLEnv",
    target_z: float = 0.2725,    # desired ball-CENTRE height at the net plane (net top 0.1525 + ~0.12 m)
    sigma: float = 0.10,
    clear_bonus: float = 0.5,
    net_clear_z: float = 0.1725,  # ball clears the net once its centre exceeds net top (0.1525) + ball R (0.02)
) -> torch.Tensor:
    """One-shot reward at each racket hit for clearing the net at a specified margin. Shape ``(N,)``.

    Mirrors PACE's ``reward_future_pass_net`` but uses the height the env's RK4 predictor reports where
    the outgoing shot crosses the net plane (``x = net_x``) under the SAME mocap-fitted gravity + drag +
    Magnus flight model that flies the ball — so the predicted clearance includes spin curvature, not a
    no-drag closed form. A Gaussian centred on ``target_z`` rewards passing at the desired margin (not
    merely "as high as possible"), plus a ``clear_bonus`` for actually being above the net. Fires only on
    the step of a hit (``_hit_event`` consumed once per step by ``TableTennisEnv.step``). Returns zeros if
    the spin-aware physics layer is inactive."""
    if not getattr(env, "_physics_ready", False):
        return torch.zeros(env.num_envs, device=env.device)

    hit = env._hit_event
    net_z = env._predicted_net_z
    err = net_z - target_z
    shaped = torch.exp(-(err * err) / (2.0 * sigma * sigma))

    valid = hit & env._predicted_net_valid
    reward = torch.where(valid, shaped, torch.zeros_like(shaped))
    reward = reward + clear_bonus * (valid & (net_z > net_clear_z)).float()
    return reward
