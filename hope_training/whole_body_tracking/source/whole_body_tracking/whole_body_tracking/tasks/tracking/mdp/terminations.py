from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def align_body_ids(
    sensor_names: list[str], asset_names: list[str],
    sensor_ids: list[int], asset_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Pair a contact-sensor body selection with an articulation body selection BY NAME.

    A ``ContactSensor``'s body order comes from the PhysX rigid-body view built over the matched
    prims; an ``Articulation``'s body order comes from the articulation's own link order.  They
    are not guaranteed to agree, and if they silently disagree the termination would read one
    body's contact force against another body's position — a bug that produces plausible-looking
    numbers forever.  So the pairing is done on names and the intersection is returned in a fixed
    (sensor) order.  Bodies the articulation does not expose are dropped, not guessed.
    """
    want = {sensor_names[i]: i for i in sensor_ids}
    have = {asset_names[i]: i for i in asset_ids}
    common = [n for n in (sensor_names[i] for i in sensor_ids) if n in have]
    if not common:
        raise RuntimeError(
            "robot_hit_table: sensor and articulation body selections do not overlap; "
            f"sensor={sorted(want)} asset={sorted(have)}"
        )
    return [want[n] for n in common], [have[n] for n in common]


def _aligned_body_ids(sensor, asset, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """``align_body_ids`` memoized on the sensor (the selections are fixed for a run)."""
    key = f"_hope_table_hit_ids__{sensor_cfg.name}__{asset_cfg.name}"
    cached = getattr(sensor, key, None)
    if cached is not None:
        return cached
    s_ids = sensor_cfg.body_ids
    a_ids = asset_cfg.body_ids
    s_ids = list(range(len(sensor.body_names))) if not isinstance(s_ids, list) else list(s_ids)
    a_ids = list(range(len(asset.body_names))) if not isinstance(a_ids, list) else list(a_ids)
    pair = align_body_ids(list(sensor.body_names), list(asset.body_names), s_ids, a_ids)
    setattr(sensor, key, pair)
    return pair


def table_hit_mask(
    body_pos_w: torch.Tensor,
    contact_force_w: torch.Tensor,
    env_origins: torch.Tensor,
    aabb_lo: torch.Tensor,
    aabb_hi: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Pure tensor kernel behind :func:`robot_hit_table`.  No env, no Isaac — so it is testable.

    ``body_pos_w`` (E, B, 3) and ``contact_force_w`` (E, B, 3) are WORLD-frame; ``env_origins``
    (E, 3) converts them to the env-local frame the table box is expressed in.  A body counts as
    a table strike when it is BOTH pushing (|f| > threshold) AND geometrically inside the table
    slab's inflated box.
    """
    p_local = body_pos_w - env_origins[:, None, :]
    inside = torch.all((p_local >= aabb_lo) & (p_local <= aabb_hi), dim=-1)
    pushing = torch.norm(contact_force_w, dim=-1) > float(force_threshold)
    return torch.any(inside & pushing, dim=-1)


def robot_hit_table(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    near_x: float,
    surface_z: float,
    force_threshold: float = 1.0,
    margin: float = 0.02,
) -> torch.Tensor:
    """The robot struck the table.  Terminal, exactly like falling over.

    人话:机器人打到桌子了 —— 跟摔倒一样,这一局直接结束。

    WHY IT IS A CONJUNCTION (contact AND geometry), not contact alone
    ----------------------------------------------------------------
    IsaacLab's ``ContactSensor`` can attribute a contact to a specific other prim
    (``force_matrix_w``), but only ONE sensor body may be filtered per environment — the sensor
    docstring is explicit that ``prim_path='{ENV_REGEX_NS}/Robot/.*'`` +
    ``filter_prim_paths_expr=['{ENV_REGEX_NS}/Table']`` "will not work", you would need one sensor
    per body.  So the un-filtered ``net_forces_w`` is all a whole-body sensor can give, and on its
    own it cannot tell "hit the table" from "the arm hit the FLOOR while falling".

    The geometric half fixes exactly that: the contacting body must also be inside the table
    slab's own axis-aligned box (inflated by ``margin`` so a contact resolved a few millimetres
    outside the surface still counts).  The box comes from
    ``table_tennis.table_frame.table_top_aabb_env`` — the SAME slab the collider is spawned from —
    so the sensor and the collider cannot drift apart.  A body on the floor at x < near_x fails
    the box test and is attributed to the fall guards, where it belongs.

    KNOWN GAP, stated rather than hidden: the collider is the table TOP slab only (the real table
    is a slab on legs and the repo has no leg geometry anywhere).  A racket that arrives under
    the overhang WITHOUT crossing the surface — e.g. driven in from behind the near edge below
    z = surface_z - TABLE_THICKNESS — touches nothing and is not caught.  What that costs in
    practice is small, because reaching such a pose from the ready stance means crossing the slab
    on the way; what it costs in principle is that this term proves "did not strike the table",
    not "stayed out of the table's volume".
    """
    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame

    sensor = env.scene.sensors[sensor_cfg.name]
    forces = getattr(sensor.data, "net_forces_w", None)
    if forces is None or forces.ndim != 3:
        raise RuntimeError(
            "robot_hit_table requires sensor net_forces_w shaped [env, body, 3]; got "
            f"{None if forces is None else tuple(forces.shape)}"
        )
    asset: Articulation = env.scene[asset_cfg.name]
    sensor_ids, asset_ids = _aligned_body_ids(sensor, asset, sensor_cfg, asset_cfg)

    f = forces[:, sensor_ids, :]
    p = asset.data.body_pos_w[:, asset_ids, :]

    lo, hi = tt_frame.table_top_aabb_env(near_x, surface_z, margin=margin)
    dev, dt = f.device, f.dtype
    lo_t = torch.tensor(lo, device=dev, dtype=dt)
    hi_t = torch.tensor(hi, device=dev, dtype=dt)
    return table_hit_mask(p, f, env.scene.env_origins, lo_t, hi_t, force_threshold)
