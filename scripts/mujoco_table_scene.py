#!/usr/bin/env python3
"""The ping-pong table, for MuJoCo, as a thing the robot can actually hit.

人话:MuJoCo 里也得有真桌子(会挡、会撞),跟现实一样;桌子只在内存里加,
不写回厂商模型。模型自身几何如果要改,必须换新 identity 和证据链。默认关闭 —— 打开与否由 Franco 决定,
现有闸门读数不会被悄悄改掉。

WHY THIS FILE EXISTS
--------------------
The vendor model of record
(``agi/A3_MuJoCo_Sim/.../a3_pingpong/a3_pingpong.xml``) contains one ``floor`` plane and one
robot.  There is no table, no net, and no ball — and there never has been, on any branch.  That
Historical bytes are frozen by ``configs/a3_mujoco_identity_v1_20260724.json``; the current
URDF-thickness-corrected source is pinned by ``configs/a3_mujoco_identity_v2_20260803.json``.
``run_ready_to_strike_join_ladder_stage2.py`` asserts ``nbody==33, ngeom==79`` against the model,
and about ten consumers index its geoms by integer id.  Therefore geometry changes require a new
successor identity, while the table is still appended only to an in-memory copy.

``scripts/audit_motion_schema2_table_net_clearance.py`` already does that append — but with
``conaffinity=0``, i.e. measurement-only boxes, because that audit measures clearance with
``mj_geomDistance`` and must not have contacts perturbing the pose it measures.  This module is
the other half: the same four boxes, at the same pose, made **collidable**, plus the loader and
the contact probe that let the playback / dynamics / feasibility tools see them.  The formal
ActionBall learned-policy profile adds one fifth, robot-only under-table keep-out derived from the
training safety assembly; the historical four-box projection remains byte-identical.

ONE TABLE, TWO SIMULATORS
-------------------------
Dimensions come from ``table_tennis/geometry.py`` (sha256-pinned, deliberately not edited) and the
pose from ``table_tennis/table_frame.py`` (the single place the HOPE -> env translation is
written).  Nothing here retypes a constant.  Those two modules are loaded **by file path**, so the
``whole_body_tracking`` package ``__init__`` — which imports ``isaaclab_tasks`` — is never
touched and this module stays importable on a bare host.

NO LEGS
-------
Stated, not hidden: no profile invents physical table legs.  ``geometry.py`` defines no leg
constants, the repo contains no leg geometry, and the visible Isaac obstacle remains the top
slab.  The ActionBall policy profile instead projects the training robot-safety keep-out under the
slab with robot-only collision filtering; its collision-disabled fitted ball never sees that
volume.  The default legacy profile still has only top/net/posts.  This keeps both profiles honest:
historical consumers do not change, while the formal policy gate cannot tunnel a robot under the
training keep-out.

WHAT COLLIDES
-------------
The four boxes get ``contype=0 conaffinity=7`` — byte-for-byte the vendor ``floor`` geom's own
setting — so they collide with the 37 ``class="collision"`` robot geoms (``contype=1``) and with
nothing else: not the floor, not each other, and not the 41 visual-only geoms.

The Isaac task obstacle spawns the **table top only** and deliberately no net collider
(``attach_table_obstacle``: the net sits at x=1.87 m, out of reach behind the table).  Here all
four boxes are available because MuJoCo is the reality-matching sim and carries no training path,
but every reading this module produces is broken out per obstacle, so the Isaac-equivalent number
(table top alone) is always readable on its own.  Callers that want strict Isaac parity pass
``obstacles=("motion_table_top",)``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_MJCF = (
    REPO_ROOT
    / "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
_TABLE_TENNIS = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "tasks/table_tennis"
)
_GEOMETRY_PY = _TABLE_TENNIS / "geometry.py"
_TABLE_FRAME_PY = _TABLE_TENNIS / "table_frame.py"
_HOPE_COMMANDS_PY = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "tasks/tracking/mdp/hope_commands.py"
)
_AUDIT_PY = REPO_ROOT / "scripts/audit_motion_schema2_table_net_clearance.py"

#: Append order, and therefore geom-id order, of the four world boxes.  Must equal the audit
#: module's ``OBSTACLE_NAMES``; :func:`obstacle_geometry` asserts that it does.
OBSTACLE_NAMES = (
    "motion_table_top",
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)

#: The formal learned-policy ActionBall gate must reproduce the training task's
#: complete five-solid robot-safety assembly.  The extra box is deliberately a
#: robot-only conservative keep-out from the floor to the slab underside; it is
#: not a ball surface and it is not a claim about individual table-leg shapes.
ACTION_BALL_ROBOT_KEEPOUT_NAME = "motion_table_robot_keepout"
ACTION_BALL_POLICY_OBSTACLE_NAMES = (
    "motion_table_top",
    ACTION_BALL_ROBOT_KEEPOUT_NAME,
    "motion_net",
    "motion_net_post_left",
    "motion_net_post_right",
)
ACTION_BALL_POLICY_GEOMETRY_KIND = (
    "whole_body_tracking.action_ball.five_solid_robot_safety_geometry_v1"
)

#: The obstacle the Isaac task actually spawns.  ``robot_hit_table`` tests this box and no other.
ISAAC_EQUIVALENT_OBSTACLES = ("motion_table_top",)

#: Net-post height above the surface.  This is the audit's own post model (``NET_HEIGHT + 0.02``),
#: reproduced here from ``geometry.NET_HEIGHT`` — the ``0.02`` cap is the posts' only free number
#: and it lives in ``_expected_obstacle_geometry`` upstream; a test pins the two together.
_POST_CAP_M = 0.02
_POST_HALF_WIDTH_M = 0.01


class TableSceneError(RuntimeError):
    """Fail-closed source, derivation, or model-compilation error."""


# --------------------------------------------------------------------------- source loading ---


def _load_by_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TableSceneError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_geometry_and_frame() -> tuple[Any, Any]:
    """Load ``geometry`` and ``table_frame`` without importing the Isaac package tree.

    ``table_frame`` does ``from whole_body_tracking.tasks.table_tennis import geometry``, so the
    parent packages must exist as bare module objects before it executes.  They are created empty
    on purpose: touching the real ``whole_body_tracking/__init__.py`` would pull in
    ``isaaclab_tasks``, which does not exist on a host or in a plain MuJoCo venv.
    """

    geometry = _load_by_path("whole_body_tracking.tasks.table_tennis.geometry", _GEOMETRY_PY)
    for package in (
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.table_tennis",
    ):
        sys.modules.setdefault(package, types.ModuleType(package))
    sys.modules["whole_body_tracking.tasks.table_tennis"].geometry = geometry
    table_frame = _load_by_path(
        "whole_body_tracking.tasks.table_tennis.table_frame", _TABLE_FRAME_PY
    )
    return geometry, table_frame


def virtual_table_pose() -> tuple[float, float]:
    """``(near_x, surface_z)`` read out of the live ``RacketTargetCommandCfg`` defaults.

    ``hope_commands.py`` needs torch + Isaac to import, so the two defaults are read from its
    source with ``ast`` rather than retyped here.  If the trainer's table ever moves, every number
    this module produces moves with it.
    """

    try:
        tree = ast.parse(_HOPE_COMMANDS_PY.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise TableSceneError(f"cannot read tracking command source: {exc}") from exc
    found: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "RacketTargetCommandCfg":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or statement.value is None:
                continue
            target = statement.target
            if not isinstance(target, ast.Name):
                continue
            if target.id in ("vb_table_near_x", "vb_table_surface_z"):
                try:
                    found[target.id] = float(ast.literal_eval(statement.value))
                except (ValueError, TypeError) as exc:
                    raise TableSceneError(f"{target.id} is not a literal number") from exc
    missing = {"vb_table_near_x", "vb_table_surface_z"} - set(found)
    if missing:
        raise TableSceneError(
            f"RacketTargetCommandCfg is missing {sorted(missing)}; the table pose is unbound"
        )
    return found["vb_table_near_x"], found["vb_table_surface_z"]


# ------------------------------------------------------------------------- geometry derivation ---


def obstacle_geometry(
    near_x: float | None = None, surface_z: float | None = None
) -> dict[str, Any]:
    """The four world boxes, derived live from ``geometry.py`` + ``table_frame.py``.

    Shape matches what :func:`augment_mjcf_xml` consumes (and what the frozen prereg JSON stores),
    so the collidable and the inert paths cannot drift apart: both go through the same appender.
    """

    geometry, table_frame = load_geometry_and_frame()
    if near_x is None or surface_z is None:
        default_near_x, default_surface_z = virtual_table_pose()
        near_x = default_near_x if near_x is None else near_x
        surface_z = default_surface_z if surface_z is None else surface_z
    near_x = float(near_x)
    surface_z = float(surface_z)
    post_height = geometry.NET_HEIGHT + _POST_CAP_M
    rows = {
        "primitive": "axis_aligned_box_full_extents_m",
        "table_top": {
            "name": "motion_table_top",
            "center_mjcf_world_m": list(table_frame.table_top_center_env(near_x, surface_z)),
            "full_extents_m": list(table_frame.table_top_size()),
        },
        "net": {
            "name": "motion_net",
            "center_mjcf_world_m": list(table_frame.net_center_env(near_x, surface_z)),
            "full_extents_m": list(geometry.net_size()),
        },
        "net_posts": [
            {
                "name": f"motion_net_post_{side}",
                "center_mjcf_world_m": list(
                    table_frame.net_post_center_env(
                        near_x, surface_z, left=(side == "left"), post_height=post_height
                    )
                ),
                "full_extents_m": [_POST_HALF_WIDTH_M * 2.0, _POST_HALF_WIDTH_M * 2.0, post_height],
            }
            for side in ("left", "right")
        ],
        "source_semantics": (
            "derived live from table_tennis.geometry (dimensions) and table_tennis.table_frame "
            "(HOPE->env pose); no constant is restated here"
        ),
    }
    names = [rows["table_top"]["name"], rows["net"]["name"], *(p["name"] for p in rows["net_posts"])]
    if tuple(names) != OBSTACLE_NAMES:
        raise TableSceneError(f"obstacle append order changed: {names}")
    return rows


def _box_row_from_aabb(
    name: str,
    bounds: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],
) -> dict[str, Any]:
    lo, hi = bounds
    if len(lo) != 3 or len(hi) != 3:
        raise TableSceneError(f"{name} AABB must have three axes")
    center = []
    extents = []
    for low, high in zip(lo, hi):
        low = float(low)
        high = float(high)
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            raise TableSceneError(f"{name} AABB is not finite and non-empty")
        center.append(0.5 * (low + high))
        extents.append(high - low)
    return {
        "name": str(name),
        "center_mjcf_world_m": center,
        "full_extents_m": extents,
    }


def action_ball_policy_obstacle_geometry(
    near_x: float | None = None,
    surface_z: float | None = None,
    *,
    keepout_floor_z: float = 0.0,
) -> dict[str, Any]:
    """The exact five-solid robot-safety assembly used by ActionBall training.

    Unlike :func:`obstacle_geometry`, this formal-policy projection includes
    the conservative under-table keep-out.  Every box is derived from
    ``table_frame.table_assembly_aabbs_env`` so the MuJoCo gate consumes the
    same top/keep-out/net/post geometry as the Isaac task.  The legacy four-box
    projection remains byte-identical for its historical clearance consumers.
    """

    _geometry, table_frame = load_geometry_and_frame()
    if near_x is None or surface_z is None:
        default_near_x, default_surface_z = virtual_table_pose()
        near_x = default_near_x if near_x is None else near_x
        surface_z = default_surface_z if surface_z is None else surface_z
    near_x = float(near_x)
    surface_z = float(surface_z)
    keepout_floor_z = float(keepout_floor_z)
    roles = tuple(table_frame.TABLE_ASSEMBLY_ROLES)
    expected_roles = ("top", "keepout", "net", "post_left", "post_right")
    if roles != expected_roles:
        raise TableSceneError(
            f"training table assembly role order drifted: {roles}"
        )
    bounds = table_frame.table_assembly_aabbs_env(
        near_x,
        surface_z,
        keepout_floor_z=keepout_floor_z,
        margin=0.0,
    )
    if len(bounds) != len(ACTION_BALL_POLICY_OBSTACLE_NAMES):
        raise TableSceneError("training table assembly is not exactly five solids")
    derived_by_role = {
        role: _box_row_from_aabb(name, aabb)
        for role, name, aabb in zip(
            roles, ACTION_BALL_POLICY_OBSTACLE_NAMES, bounds
        )
    }
    legacy = obstacle_geometry(near_x, surface_z)
    rows_by_role = {
        "top": dict(legacy["table_top"]),
        "keepout": derived_by_role["keepout"],
        "net": dict(legacy["net"]),
        "post_left": dict(legacy["net_posts"][0]),
        "post_right": dict(legacy["net_posts"][1]),
    }
    # The four physical boxes are already the common table-frame derivation.
    # Reopen that equality before combining them with the fifth training box.
    for role in ("top", "net", "post_left", "post_right"):
        for field in ("center_mjcf_world_m", "full_extents_m"):
            actual = rows_by_role[role][field]
            expected = derived_by_role[role][field]
            if any(
                abs(float(a) - float(b)) > 1.0e-12
                for a, b in zip(actual, expected)
            ):
                raise TableSceneError(
                    f"legacy {role} geometry drifted from training assembly"
                )
    rows = {
        "primitive": "axis_aligned_box_full_extents_m",
        "table_top": rows_by_role["top"],
        "robot_keepout": rows_by_role["keepout"],
        "net": rows_by_role["net"],
        "net_posts": [
            rows_by_role["post_left"],
            rows_by_role["post_right"],
        ],
        "keepout_floor_z_m": keepout_floor_z,
        "source_semantics": (
            "exact projection of table_frame.table_assembly_aabbs_env; "
            "under-table keepout is robot-only and never a ball contact surface"
        ),
    }
    if tuple(
        row["name"] for row in action_ball_policy_obstacle_rows(rows)
    ) != ACTION_BALL_POLICY_OBSTACLE_NAMES:
        raise TableSceneError("ActionBall five-solid append order changed")
    return rows


def action_ball_policy_obstacle_rows(
    geometry_rows: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Normalize the exact top/keep-out/net/post row order."""

    posts = geometry_rows.get("net_posts")
    if not isinstance(posts, list) or len(posts) != 2:
        raise TableSceneError(
            "ActionBall geometry must contain exactly two net posts"
        )
    raw = (
        geometry_rows.get("table_top"),
        geometry_rows.get("robot_keepout"),
        geometry_rows.get("net"),
        *posts,
    )
    if any(not isinstance(row, Mapping) for row in raw):
        raise TableSceneError("ActionBall geometry must contain five box rows")
    rows = tuple(dict(row) for row in raw)
    names = tuple(str(row.get("name", "")) for row in rows)
    if names != ACTION_BALL_POLICY_OBSTACLE_NAMES:
        raise TableSceneError(
            f"ActionBall five-solid names/order drifted: {names}"
        )
    for row in rows:
        center = row.get("center_mjcf_world_m")
        extents = row.get("full_extents_m")
        if (
            not isinstance(center, (list, tuple))
            or not isinstance(extents, (list, tuple))
            or len(center) != 3
            or len(extents) != 3
            or any(
                not math.isfinite(float(value))
                for value in (*center, *extents)
            )
            or any(float(value) <= 0.0 for value in extents)
        ):
            raise TableSceneError(
                f"ActionBall obstacle {row['name']} has invalid center/extents"
            )
    return rows


def action_ball_policy_geometry_contract(
    geometry_rows: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical five-solid payload and its explicit SHA-256."""

    roles = ("top", "keepout", "net", "post_left", "post_right")
    obstacles = []
    for role, row in zip(
        roles, action_ball_policy_obstacle_rows(geometry_rows)
    ):
        obstacles.append(
            {
                "role": role,
                "name": row["name"],
                "center_mjcf_world_m": [
                    float(value) for value in row["center_mjcf_world_m"]
                ],
                "full_extents_m": [
                    float(value) for value in row["full_extents_m"]
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "kind": ACTION_BALL_POLICY_GEOMETRY_KIND,
        "primitive": "axis_aligned_box_full_extents_m",
        "obstacle_order": list(ACTION_BALL_POLICY_OBSTACLE_NAMES),
        "obstacles": obstacles,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "payload": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def action_ball_policy_obstacle_aabbs(
    geometry_rows: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return five AABBs as NumPy arrays for the continuous policy guard."""

    import numpy as np

    result = {}
    for row in action_ball_policy_obstacle_rows(geometry_rows):
        center = np.asarray(row["center_mjcf_world_m"], np.float64)
        half = 0.5 * np.asarray(row["full_extents_m"], np.float64)
        result[str(row["name"])] = (center - half, center + half)
    return result


def table_top_aabb(
    near_x: float | None = None, surface_z: float | None = None, margin: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(lo, hi)`` of the table-top slab — the same box ``robot_hit_table`` tests against.

    Pure geometry, no MuJoCo: this is what lets a host-side test assert that a pose known to be
    inside the table volume is detected and a legal pose is not.
    """

    _geometry, table_frame = load_geometry_and_frame()
    if near_x is None or surface_z is None:
        default_near_x, default_surface_z = virtual_table_pose()
        near_x = default_near_x if near_x is None else near_x
        surface_z = default_surface_z if surface_z is None else surface_z
    return table_frame.table_top_aabb_env(float(near_x), float(surface_z), margin=float(margin))


def point_penetration_m(
    point: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
) -> float:
    """Depth of ``point`` inside the axis-aligned box; ``0.0`` when outside or on the surface.

    The depth is the distance to the NEAREST face, which is what "how far into the table is it"
    means for a solid: a racket 1 cm below the 5 cm slab's top face is 1 cm in, not 4 cm in.
    """

    depths = []
    for axis in range(3):
        value = float(point[axis])
        low, high = float(lo[axis]), float(hi[axis])
        if not (low <= value <= high):
            return 0.0
        depths.append(min(value - low, high - value))
    return float(min(depths))


# ------------------------------------------------------------------------------ MJCF assembly ---


#: The exact attribute text the audit's appender emits for one inert obstacle geom, and what it
#: becomes when the box is made solid.  ``7`` is not a new number: it is byte-for-byte the vendor
#: ``floor`` geom's own ``conaffinity``, so the table joins collision on precisely the terms the
#: ground the robot already stands on does.  ``contype`` stays ``0`` (also the floor's value), so
#: the table is inert against the floor plane and against the other three boxes: the only pairs
#: that can ever be generated are robot-vs-table.
_INERT_ATTRS = b'contype="0" conaffinity="0"'
_SOLID_ATTRS = b'contype="0" conaffinity="7"'


def augment_mjcf_xml(
    canonical_xml: bytes, geometry_rows: Mapping[str, Any], *, collidable: bool
) -> bytes:
    """Append the four world boxes, optionally solid.  ONE appender, and it is not this one.

    The boxes are produced by ``audit_motion_schema2_table_net_clearance.augment_mjcf_xml``, which
    is **deliberately not modified**: that file is pinned by path, byte count and sha256 inside
    ``configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json`` as the table/net
    clearance certificate's validator of record, and the audit re-verifies its own bytes at
    runtime.  Editing it — even to add a keyword argument — invalidates that certificate and takes
    nine of its self-binding tests down with it.

    So the solid variant is derived from the inert one by a surgical, fail-closed rewrite of a
    single attribute inside each of the four appended geoms.  Everything else — the geometry, the
    append order, the number formatting, the serializer — is the audit's, unchanged, which is what
    makes "the two simulators cannot drift apart" true by construction rather than by convention.

    ``collidable=False`` returns the audit's bytes untouched.
    """

    audit = _load_by_path("_mjcf_table_augmenter", _AUDIT_PY)
    if tuple(audit.OBSTACLE_NAMES) != OBSTACLE_NAMES:
        raise TableSceneError("audit OBSTACLE_NAMES drifted from mujoco_table_scene")
    inert = audit.augment_mjcf_xml(canonical_xml, geometry_rows)
    if not collidable:
        return inert

    out = inert
    for name in OBSTACLE_NAMES:
        opening = b'<geom name="' + name.encode("ascii") + b'"'
        start = out.find(opening)
        if start < 0 or out.find(opening, start + 1) >= 0:
            raise TableSceneError(f"expected exactly one appended geom named {name}")
        end = out.find(b"/>", start)
        if end < 0:
            raise TableSceneError(f"appended geom {name} is not self-closing")
        element = out[start:end]
        if element.count(_INERT_ATTRS) != 1:
            raise TableSceneError(
                f"appended geom {name} no longer carries {_INERT_ATTRS!r}; the audit's attribute "
                "layout changed and the collidable rewrite must be re-derived, not guessed"
            )
        out = out[:start] + element.replace(_INERT_ATTRS, _SOLID_ATTRS) + out[end:]

    if len(out) != len(inert):
        raise TableSceneError("collidable rewrite changed the document length")
    differing = sum(1 for a, b in zip(inert, out) if a != b)
    if differing != len(OBSTACLE_NAMES):
        raise TableSceneError(
            f"collidable rewrite touched {differing} bytes, expected {len(OBSTACLE_NAMES)}"
        )
    return out


def append_action_ball_policy_keepout_xml(
    assembled_xml: bytes,
    geometry_rows: Mapping[str, Any],
    *,
    collidable: bool,
) -> bytes:
    """Append the fifth, robot-only keep-out to an already four-box scene.

    The fitted-ball assembler first installs the historical four boxes and its
    collision-disabled analytic ball.  This policy-only post-transform then
    adds the one training solid that must affect the robot but never the ball.
    """

    import xml.etree.ElementTree as ET

    if b"<!DOCTYPE" in assembled_xml or b"<!ENTITY" in assembled_xml:
        raise TableSceneError("assembled MJCF contains forbidden DTD/entity")
    try:
        root = ET.fromstring(assembled_xml)
    except ET.ParseError as exc:
        raise TableSceneError(
            f"cannot parse assembled MJCF for ActionBall keepout: {exc}"
        ) from exc
    worldbodies = root.findall("./worldbody")
    if len(worldbodies) != 1:
        raise TableSceneError("assembled MJCF must contain one worldbody")
    rows = action_ball_policy_obstacle_rows(geometry_rows)
    existing = [
        node.get("name")
        for node in root.iter("geom")
        if node.get("name")
    ]
    for name in OBSTACLE_NAMES:
        if existing.count(name) != 1:
            raise TableSceneError(
                f"assembled four-box scene must contain exactly one {name}"
            )
    if ACTION_BALL_ROBOT_KEEPOUT_NAME in existing:
        raise TableSceneError("ActionBall robot keepout is already present")
    keepout = rows[1]
    ET.SubElement(
        worldbodies[0],
        "geom",
        {
            "name": keepout["name"],
            "type": "box",
            "pos": " ".join(
                format(float(value), ".17g")
                for value in keepout["center_mjcf_world_m"]
            ),
            "size": " ".join(
                format(0.5 * float(value), ".17g")
                for value in keepout["full_extents_m"]
            ),
            "contype": "0",
            "conaffinity": "7" if collidable else "0",
            "group": "6",
        },
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _mesh_assets(canonical_xml: bytes, model_root: Path) -> dict[str, bytes]:
    """Read the MJCF's exact mesh closure so the model can be compiled from a string."""

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(canonical_xml)
    except ET.ParseError as exc:
        raise TableSceneError(f"cannot parse canonical MJCF: {exc}") from exc
    compiler = root.find("./compiler")
    meshdir = "" if compiler is None else (compiler.get("meshdir") or "")
    assets: dict[str, bytes] = {}
    for node in root.findall("./asset/mesh"):
        raw = node.get("file")
        if not raw:
            raise TableSceneError("canonical MJCF mesh node lacks a file attribute")
        relative = (Path(meshdir) / Path(raw)) if meshdir else Path(raw)
        key = relative.as_posix()
        if key in assets:
            continue
        try:
            assets[key] = (model_root / relative).read_bytes()
        except OSError as exc:
            raise TableSceneError(f"cannot read MJCF mesh {key}: {exc}") from exc
    if not assets:
        raise TableSceneError("canonical MJCF declares no meshes")
    return assets


@dataclass(frozen=True)
class TableScene:
    """A compiled, table-aware model plus the ids needed to read contacts off it."""

    model: Any
    obstacle_geom_ids: dict[str, int]
    obstacle_names: tuple[str, ...]
    obstacle_rows: dict[str, Any]
    collidable: bool
    augmented_xml_sha256: str
    canonical_xml_sha256: str
    near_x: float
    surface_z: float
    geom_index_shift: int

    def obstacle_of(self, geom_id: int) -> str | None:
        for name, value in self.obstacle_geom_ids.items():
            if int(value) == int(geom_id):
                return name
        return None


def load_table_scene(
    mujoco: Any,
    mjcf_path: Path | str = CANONICAL_MJCF,
    *,
    collidable: bool = True,
    obstacles: Sequence[str] | None = None,
    near_x: float | None = None,
    surface_z: float | None = None,
    action_ball_policy: bool = False,
) -> TableScene:
    """Compile the vendor model with the table appended in memory.  Never writes to disk.

    The default remains the historical four-box scene.  ``action_ball_policy``
    selects the five-solid training-equivalent assembly.  ``obstacles`` may
    mute named boxes while preserving the selected profile's geom-id shift.
    """

    obstacle_names = (
        ACTION_BALL_POLICY_OBSTACLE_NAMES
        if action_ball_policy
        else OBSTACLE_NAMES
    )
    if obstacles is None:
        obstacles = obstacle_names
    unknown = [name for name in obstacles if name not in obstacle_names]
    if unknown:
        raise TableSceneError(f"unknown obstacle name(s): {unknown}")
    path = Path(mjcf_path).expanduser().resolve()
    if not path.is_file():
        raise TableSceneError(f"MJCF not found: {path}")
    canonical_xml = path.read_bytes()
    if near_x is None or surface_z is None:
        default_near_x, default_surface_z = virtual_table_pose()
        near_x = default_near_x if near_x is None else near_x
        surface_z = default_surface_z if surface_z is None else surface_z
    rows = (
        action_ball_policy_obstacle_geometry(near_x, surface_z)
        if action_ball_policy
        else obstacle_geometry(near_x, surface_z)
    )
    augmented = augment_mjcf_xml(canonical_xml, rows, collidable=collidable)
    if action_ball_policy:
        augmented = append_action_ball_policy_keepout_xml(
            augmented, rows, collidable=collidable
        )
    assets = _mesh_assets(canonical_xml, path.parent)
    try:
        model = mujoco.MjModel.from_xml_string(augmented.decode("utf-8"), assets=assets)
    except Exception as exc:  # noqa: BLE001 - MuJoCo raises bare Exception subclasses
        raise TableSceneError(f"cannot compile the table-aware MJCF in memory: {exc}") from exc

    ids: dict[str, int] = {}
    for name in obstacle_names:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0:
            raise TableSceneError(f"augmented obstacle geom is missing after compile: {name}")
        if int(model.geom_bodyid[geom_id]) != 0:
            raise TableSceneError(f"obstacle {name} is not attached to the world body")
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise TableSceneError(f"obstacle {name} is not a box")
        ids[name] = geom_id
        if collidable and name not in obstacles:
            # Present but deselected: mute it so it cannot generate contacts.
            model.geom_conaffinity[geom_id] = 0

    live = [
        name
        for name in obstacle_names
        if int(model.geom_conaffinity[ids[name]]) != 0
    ]
    if collidable and tuple(live) != tuple(
        name for name in obstacle_names if name in obstacles
    ):
        raise TableSceneError(f"collidable selection did not take effect: live={live}")
    if not collidable and live:
        raise TableSceneError("inert augmentation must leave every obstacle at conaffinity=0")

    return TableScene(
        model=model,
        obstacle_geom_ids=ids,
        obstacle_names=tuple(obstacle_names),
        obstacle_rows=rows,
        collidable=bool(collidable),
        augmented_xml_sha256=hashlib.sha256(augmented).hexdigest(),
        canonical_xml_sha256=hashlib.sha256(canonical_xml).hexdigest(),
        near_x=float(near_x),
        surface_z=float(surface_z),
        geom_index_shift=len(obstacle_names),
        )


# ---------------------------------------------------------------------------- contact probing ---


@dataclass(frozen=True)
class TableContact:
    """One robot-geom / table-box overlap at one frame.  ``depth_m`` is positive when touching."""

    frame: int
    robot_geom: str
    obstacle: str
    depth_m: float
    position_m: tuple[float, float, float]


def frame_table_contacts(
    mujoco: Any,
    scene: TableScene,
    data: Any,
    frame: int = 0,
) -> list[TableContact]:
    """Read every robot-vs-table contact out of an already-posed ``mjData``.

    The caller is responsible for having written ``qpos`` and called ``mj_forward`` — collision
    detection runs inside ``mj_forward``'s position stage, so contacts are live by then and no
    ``mj_step`` is needed.  MuJoCo only emits a contact when the signed distance is below the
    pair's margin (zero here), so every contact returned is a real overlap and ``-contact.dist``
    is its penetration depth.
    """

    model = scene.model
    contacts: list[TableContact] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        obstacle = scene.obstacle_of(g1) or scene.obstacle_of(g2)
        if obstacle is None:
            continue
        robot_geom_id = g2 if scene.obstacle_of(g1) is not None else g1
        if scene.obstacle_of(robot_geom_id) is not None:
            continue  # obstacle-vs-obstacle can never happen, but never report it as a strike
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_id)
        depth = -float(contact.dist)
        if not math.isfinite(depth) or depth <= 0.0:
            continue
        contacts.append(
            TableContact(
                frame=int(frame),
                robot_geom=str(name) if name else f"geom_{robot_geom_id}",
                obstacle=obstacle,
                depth_m=depth,
                position_m=tuple(float(v) for v in contact.pos),
            )
        )
    return contacts


def summarize_contacts(contacts: Sequence[TableContact]) -> dict[str, Any]:
    """Per-clip rollup: does it strike the table, where, and how deep."""

    if not contacts:
        return {
            "strikes_table": False,
            "contact_frames": [],
            "max_penetration_m": 0.0,
            "worst": None,
            "per_obstacle": {},
        }
    worst = max(contacts, key=lambda c: c.depth_m)
    per_obstacle: dict[str, Any] = {}
    for name in ACTION_BALL_POLICY_OBSTACLE_NAMES:
        rows = [c for c in contacts if c.obstacle == name]
        if not rows:
            continue
        deepest = max(rows, key=lambda c: c.depth_m)
        per_obstacle[name] = {
            "contact_frames": sorted({c.frame for c in rows}),
            "max_penetration_m": deepest.depth_m,
            "worst_frame": deepest.frame,
            "worst_robot_geom": deepest.robot_geom,
        }
    return {
        "strikes_table": True,
        "contact_frames": sorted({c.frame for c in contacts}),
        "max_penetration_m": worst.depth_m,
        "worst": {
            "frame": worst.frame,
            "robot_geom": worst.robot_geom,
            "obstacle": worst.obstacle,
            "depth_m": worst.depth_m,
            "position_m": list(worst.position_m),
        },
        "per_obstacle": per_obstacle,
    }
