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
source of the MuJoCo table/net audit); this module is a pure consumer of it and adds no constant
of its own.

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

from whole_body_tracking.tasks.table_tennis import geometry


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
    table surface up through ``post_height``.  Only the MJCF audit uses posts today; the Isaac
    scenes spawn no post collider (they are visual-only there).
    """
    y = geometry.TABLE_WIDTH / 2.0 + geometry.NET_OVERHANG
    return (
        float(near_x) + geometry.NET_X,
        y if left else -y,
        float(surface_z) + float(post_height) / 2.0,
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
