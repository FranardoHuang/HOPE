"""The table, expressed in the TRACKING task's env frame.  One place, one translation.

人话:同一张桌子,两套坐标写法,别再各写各的。

There are exactly two frames in which this repo places the ping-pong table, and they differ by a
pure translation:

* **HOPE frame** — :mod:`whole_body_tracking.tasks.table_tennis.geometry`'s own frame.  Near edge
  at ``x = 0``, ``y in [-TABLE_WIDTH, 0]``, table SURFACE at ``z = 0`` (floor at ``-TABLE_HEIGHT``).
* **Tracking env frame** — env-local, floor at ``z = 0``.  Near edge at
  ``RacketTargetCommandCfg.vb_table_near_x``, table CENTRED on ``y = 0``, surface at
  ``vb_table_surface_z``.

so::

    p_env = p_hope + (near_x, +TABLE_WIDTH/2, surface_z)

This module is the ONLY place that translation is written down.  ``geometry.py`` stays the single
source of truth for the table's DIMENSIONS and is deliberately not edited (it is sha256-pinned by
``configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json`` as the bound frame
source of the MuJoCo table/net audit).  The regulation table/net dimensions remain there.  This
adapter additionally owns only the already-preregistered conservative post-proxy width/height so
host-only safety kernels and Isaac builders do not duplicate those two proxy values.

Consumers: ``hope_env_cfg.attach_table_obstacle`` (the task obstacle, default ON),
``attach_shadow_ball_scene`` / ``attach_physical_ball_scene`` (the two metrics-only truth
instruments), and the ``robot_hit_table`` termination.  The in-memory MJCF table/net augmentation
in ``scripts/audit_motion_schema2_table_net_clearance.py`` derives the same numbers from the same
``geometry.py`` constants with the same ``(near_x, TABLE_WIDTH/2, surface_z)`` shift; it does not
import this module (that script must stay importable without the Isaac package tree), so a test
asserts the two agree instead.

Pure Python — no Isaac, no torch — so it is importable and testable on a bare host.
"""

from __future__ import annotations

import math

from whole_body_tracking.tasks.table_tennis import geometry

# Explicit post proxy already used by the frozen MuJoCo clearance preregistration.  Keep it in this
# pure shared frame module so Isaac scene construction and host-only safety kernels consume the
# same dimensions without importing Isaac Lab.
NET_POST_WIDTH: float = 0.02
NET_POST_HEIGHT: float = geometry.NET_HEIGHT + 0.02
TABLE_ASSEMBLY_ROLES: tuple[str, ...] = (
    "top",
    "keepout",
    "net",
    "post_left",
    "post_right",
)


def env_frame_offset(near_x: float, surface_z: float) -> tuple[float, float, float]:
    """The HOPE -> tracking-env translation ``(dx, dy, dz)``.  The only place it is written down."""
    return (float(near_x), geometry.TABLE_WIDTH / 2.0, float(surface_z))


def hope_to_env(
    point: tuple[float, float, float], near_x: float, surface_z: float
) -> tuple[float, float, float]:
    """Move one HOPE-frame landmark into the tracking env frame."""
    dx, dy, dz = env_frame_offset(near_x, surface_z)
    return (float(point[0]) + dx, float(point[1]) + dy, float(point[2]) + dz)


def table_top_center_env(near_x: float, surface_z: float) -> tuple[float, float, float]:
    """Centre of the table-top slab in the tracking env frame (its TOP face lands on surface_z)."""
    return hope_to_env(geometry.table_top_center(), near_x, surface_z)


def table_top_size() -> tuple[float, float, float]:
    """Full extents of the table-top slab.  Re-exported so a caller needs one import, not two."""
    return geometry.table_top_size()


def table_visual_origin_env(near_x: float, surface_z: float) -> tuple[float, float, float]:
    """Origin of the visual USD table mesh in the tracking env frame.

    The USD models the floor at its own local ``z = 0`` with the playing surface at local
    ``z = 0.76`` (== ``TABLE_HEIGHT``), centred horizontally on its local ``(x, y) = (0, 0)``, so
    its origin is the FLOOR point under the table centre — one ``TABLE_HEIGHT`` below the surface.
    """
    x, y, _ = table_top_center_env(near_x, surface_z)
    return (x, y, float(surface_z) - geometry.TABLE_HEIGHT)


def net_center_env(near_x: float, surface_z: float) -> tuple[float, float, float]:
    """Centre of the net slab in the tracking env frame."""
    return hope_to_env(geometry.net_center(), near_x, surface_z)


def net_post_center_env(
    near_x: float, surface_z: float, *, left: bool, post_height: float
) -> tuple[float, float, float]:
    """Centre of one net post in the tracking env frame.

    The post straddles the net line at ``|y| = TABLE_WIDTH/2 + NET_OVERHANG`` and stands from the
    table surface up through ``post_height``.
    """
    y = geometry.TABLE_WIDTH / 2.0 + geometry.NET_OVERHANG
    return (
        float(near_x) + geometry.NET_X,
        y if left else -y,
        float(surface_z) + float(post_height) / 2.0,
    )


def net_post_size() -> tuple[float, float, float]:
    """Full extents of the shared conservative net-post collision proxy."""

    return (NET_POST_WIDTH, NET_POST_WIDTH, NET_POST_HEIGHT)


def table_assembly_aabbs_env(
    near_x: float,
    surface_z: float,
    *,
    keepout_floor_z: float = 0.0,
    margin: float = 0.0,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]], ...
]:
    """AABBs for the conservative robot keep-out, top, net and two posts.

    Box order is :data:`TABLE_ASSEMBLY_ROLES`: real top, under-table keep-out, regulation net,
    then left/right post proxies.  The keep-out fills only the volume below the regulation top
    slab, from ``keepout_floor_z`` to its underside; it is a robot-safety proxy, not a model of
    individual leg geometry.
    """

    near = float(near_x)
    floor_z = float(keepout_floor_z)
    surface = float(surface_z)
    m = float(margin)
    if not all(math.isfinite(value) for value in (near, floor_z, surface, m)):
        raise ValueError("table assembly coordinates and margin must be finite")
    underside = surface - geometry.TABLE_THICKNESS
    if not floor_z < underside:
        raise ValueError(
            "table assembly keep-out floor must be below the top-slab underside"
        )
    if m < 0.0:
        raise ValueError("table assembly AABB margin must be non-negative")

    table_c = table_top_center_env(near, surface)
    table_s = geometry.table_top_size()
    net_c = net_center_env(near, surface)
    net_s = geometry.net_size()
    post_s = net_post_size()
    post_centers = (
        net_post_center_env(
            near, surface, left=True, post_height=NET_POST_HEIGHT
        ),
        net_post_center_env(
            near, surface, left=False, post_height=NET_POST_HEIGHT
        ),
    )

    def bounds(
        center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            tuple(float(c) - float(s) / 2.0 - m for c, s in zip(center, size)),
            tuple(float(c) + float(s) / 2.0 + m for c, s in zip(center, size)),
        )

    keepout_center = (table_c[0], table_c[1], (floor_z + underside) / 2.0)
    keepout_size = (table_s[0], table_s[1], underside - floor_z)
    return (
        bounds(table_c, table_s),
        bounds(keepout_center, keepout_size),
        bounds(net_c, net_s),
        *(bounds(center, post_s) for center in post_centers),
    )


def table_top_aabb_env(
    near_x: float, surface_z: float, margin: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(lo, hi)`` axis-aligned bounds of the table-top slab in the tracking env frame.

    ``margin`` inflates the box on every axis.  This is the volume the ``robot_hit_table``
    termination tests a contacting body against, derived from the SAME slab the collider is
    spawned from — the box and the collider therefore cannot drift apart.
    """
    cx, cy, cz = table_top_center_env(near_x, surface_z)
    sx, sy, sz = geometry.table_top_size()
    m = float(margin)
    return (
        (cx - sx / 2.0 - m, cy - sy / 2.0 - m, cz - sz / 2.0 - m),
        (cx + sx / 2.0 + m, cy + sy / 2.0 + m, cz + sz / 2.0 + m),
    )
