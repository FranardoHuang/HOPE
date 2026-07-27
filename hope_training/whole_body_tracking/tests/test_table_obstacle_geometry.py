"""The table has ONE pose and ONE set of dimensions, on both simulators.

Host-runnable (python 3.8, no torch, no Isaac, no MuJoCo): every module under test here is pure
Python, loaded by file path so the ``whole_body_tracking`` package ``__init__`` (which imports
``isaaclab_tasks``) is never touched.

What is pinned:

* the HOPE -> tracking-env translation is exactly ``(near_x, +TABLE_WIDTH/2, surface_z)``;
* the table-top slab's TOP face lands exactly on ``vb_table_surface_z`` and its extents are the
  ITTF constants from ``geometry.py`` — not a restatement of them;
* the box the ``robot_hit_table`` termination tests against is the same slab, only inflated;
* the numbers the MuJoCo-side in-memory MJCF augmentation uses
  (``scripts/audit_motion_schema2_table_net_clearance.py``, driven by the frozen prereg JSON)
  are the SAME numbers.  That is the check that keeps "one table" true across the two simulators:
  Isaac reads ``table_frame``, MuJoCo derives from ``geometry`` + the same shift, and this test
  asserts they land on the same millimetre.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TT = (_REPO / "hope_training/whole_body_tracking/source/whole_body_tracking"
       / "whole_body_tracking/tasks/table_tennis")
_PREREG = _REPO / "configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json"

# The tracking task's virtual-table pose (RacketTargetCommandCfg defaults).  Restated here on
# purpose: this file is the CHECK, so it must not read the value it is checking from the code that
# produces it.  test_virtual_table_pose_matches_the_command_cfg re-reads the cfg and compares.
NEAR_X = 0.5
SURFACE_Z = 0.76


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frame():
    geom = _load("whole_body_tracking.tasks.table_tennis.geometry", _TT / "geometry.py")
    # table_frame does `from whole_body_tracking.tasks.table_tennis import geometry`, so the
    # parent packages must exist as (empty) module objects before it is executed.
    import types
    for pkg in ("whole_body_tracking", "whole_body_tracking.tasks",
                "whole_body_tracking.tasks.table_tennis"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    sys.modules["whole_body_tracking.tasks.table_tennis"].geometry = geom
    tf = _load("whole_body_tracking.tasks.table_tennis.table_frame", _TT / "table_frame.py")
    return geom, tf


def _close(a, b, tol=1e-9):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(float(x) - float(y)) <= tol, f"component {i}: {x} != {y}"


# ------------------------------------------------------------------ the translation itself --- #
def test_offset_is_the_documented_translation(frame):
    geom, tf = frame
    _close(tf.env_frame_offset(NEAR_X, SURFACE_Z),
           (NEAR_X, geom.TABLE_WIDTH / 2.0, SURFACE_Z))


def test_hope_landmarks_land_where_the_tracking_frame_expects(frame):
    """Near edge -> near_x, table centred on y=0, surface -> surface_z."""
    geom, tf = frame
    near_left = tf.hope_to_env(geom.ORIGIN, NEAR_X, SURFACE_Z)
    _close(near_left, (NEAR_X, geom.TABLE_WIDTH / 2.0, SURFACE_Z))
    far_right = tf.hope_to_env(
        (geom.TABLE_LENGTH, -geom.TABLE_WIDTH, 0.0), NEAR_X, SURFACE_Z)
    _close(far_right, (NEAR_X + geom.TABLE_LENGTH, -geom.TABLE_WIDTH / 2.0, SURFACE_Z))
    # the two y corners are symmetric about 0 -> the table is centred on the robot's stance line
    assert abs(near_left[1] + far_right[1]) < 1e-12
    _close(tf.net_center_env(NEAR_X, SURFACE_Z),
           (NEAR_X + geom.NET_X, 0.0, SURFACE_Z + geom.NET_HEIGHT / 2.0))


# ------------------------------------------------------------------------- the collider ------ #
def test_slab_top_face_is_exactly_the_surface(frame):
    geom, tf = frame
    cx, cy, cz = tf.table_top_center_env(NEAR_X, SURFACE_Z)
    sx, sy, sz = tf.table_top_size()
    assert abs((cz + sz / 2.0) - SURFACE_Z) < 1e-12, "table TOP face must sit on vb_table_surface_z"
    _close((sx, sy, sz), (geom.TABLE_LENGTH, geom.TABLE_WIDTH, geom.TABLE_THICKNESS))
    _close((cx, cy), (NEAR_X + geom.TABLE_LENGTH / 2.0, 0.0))


def test_visual_mesh_origin_puts_its_surface_on_the_collider_surface(frame):
    geom, tf = frame
    vx, vy, vz = tf.table_visual_origin_env(NEAR_X, SURFACE_Z)
    # the USD's own playing surface is TABLE_HEIGHT above its origin
    assert abs((vz + geom.TABLE_HEIGHT) - SURFACE_Z) < 1e-12
    cx, cy, _ = tf.table_top_center_env(NEAR_X, SURFACE_Z)
    _close((vx, vy), (cx, cy))


def test_aabb_is_the_slab_inflated_and_nothing_else(frame):
    geom, tf = frame
    lo0, hi0 = tf.table_top_aabb_env(NEAR_X, SURFACE_Z, margin=0.0)
    _close(lo0, (NEAR_X, -geom.TABLE_WIDTH / 2.0, SURFACE_Z - geom.TABLE_THICKNESS))
    _close(hi0, (NEAR_X + geom.TABLE_LENGTH, geom.TABLE_WIDTH / 2.0, SURFACE_Z))
    m = 0.02
    lo, hi = tf.table_top_aabb_env(NEAR_X, SURFACE_Z, margin=m)
    _close(lo, tuple(v - m for v in lo0))
    _close(hi, tuple(v + m for v in hi0))


def test_the_box_covers_the_z_range_the_investigation_flagged(frame):
    """z ~ 0.65-0.69 m over the table is INSIDE the table, not empty air.

    This is the concrete failure that motivated the obstacle: commanded contact points below the
    0.76 m surface passed every check for thousands of iterations because nothing in the scene
    occupied that space.  With the 2 cm margin the slab spans z in [0.69, 0.78], so a body at
    0.69-0.71 m over the table footprint is inside the box; below that it is under the (legless)
    top slab, which is the documented gap in ``robot_hit_table``.
    """
    _geom, tf = frame
    lo, hi = tf.table_top_aabb_env(NEAR_X, SURFACE_Z, margin=0.02)
    x_mid = NEAR_X + 0.3
    for z in (0.69, 0.70, 0.71, 0.735, 0.76, 0.775):
        assert lo[2] <= z <= hi[2], f"z={z} should be inside the inflated slab"
        assert lo[0] <= x_mid <= hi[0]
    # and the pure air well above the table is NOT inside
    assert not (lo[2] <= 0.95 <= hi[2])


# ------------------------------------------------------- one table across the two simulators -- #
def test_mujoco_prereg_obstacles_equal_the_isaac_derivation(frame):
    """The MJCF augmentation and the Isaac scene place the same table, to the millimetre.

    ``scripts/audit_motion_schema2_table_net_clearance.py`` injects four inert world boxes into an
    in-memory copy of the vendor MJCF (the on-disk file is byte-pinned by
    ``configs/a3_mujoco_identity_v1_20260724.json`` and must not be edited).  Their numbers live in
    the frozen prereg below.  If this test fails, the two simulators disagree about where the
    table is — which is precisely the class of bug that "one source of truth" is supposed to make
    impossible.
    """
    geom, tf = frame
    plan = json.loads(_PREREG.read_text(encoding="utf-8"))
    obs = plan["obstacle_geometry"]
    assert obs["primitive"] == "axis_aligned_box_full_extents_m"

    _close(obs["table_top"]["center_mjcf_world_m"], tf.table_top_center_env(NEAR_X, SURFACE_Z))
    _close(obs["table_top"]["full_extents_m"], tf.table_top_size())

    _close(obs["net"]["center_mjcf_world_m"], tf.net_center_env(NEAR_X, SURFACE_Z))
    _close(obs["net"]["full_extents_m"], geom.net_size())

    post_h = geom.NET_HEIGHT + 0.02
    posts = {p["name"]: p for p in obs["net_posts"]}
    for name, left in (("motion_net_post_left", True), ("motion_net_post_right", False)):
        _close(posts[name]["center_mjcf_world_m"],
               tf.net_post_center_env(NEAR_X, SURFACE_Z, left=left, post_height=post_h))
        _close(posts[name]["full_extents_m"], (0.02, 0.02, post_h))


def test_prereg_still_binds_the_unmodified_geometry_module():
    """``geometry.py`` is sha256-pinned as the MuJoCo audit's bound frame source.

    The env-frame translation therefore lives in the NEW ``table_frame.py`` rather than being
    appended to ``geometry.py``: adding it there would have invalidated this pin (and, through it,
    the whole table/net clearance audit) for no gain.  This test is the reason that decision is
    safe to rely on.
    """
    import hashlib
    plan = json.loads(_PREREG.read_text(encoding="utf-8"))
    bound = plan["frame_sources"]["table_geometry"]
    data = (_REPO / bound["path"]).read_bytes()
    assert len(data) == int(bound["bytes"])
    assert hashlib.sha256(data).hexdigest() == bound["sha256"]


# ------------------------------------------------------------------ the cfg the tests assume -- #
def test_virtual_table_pose_matches_the_command_cfg():
    """NEAR_X / SURFACE_Z above are the live ``RacketTargetCommandCfg`` defaults, read as text.

    ``hope_commands.py`` cannot be imported on a bare host (torch + Isaac), so the two defaults are
    read out of the source.  A silent change to either would otherwise leave every number in this
    file testing a pose the trainer no longer uses.
    """
    src = (_REPO / "hope_training/whole_body_tracking/source/whole_body_tracking"
           / "whole_body_tracking/tasks/tracking/mdp/hope_commands.py").read_text(encoding="utf-8")
    assert f"vb_table_near_x: float = {NEAR_X}" in src
    assert f"vb_table_surface_z: float = {SURFACE_Z}" in src
