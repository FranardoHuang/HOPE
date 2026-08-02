"""The MuJoCo table is the SAME table as the Isaac one, and turning it on changes nothing else.

人话:MuJoCo 那张桌子必须和 Isaac 那张一模一样;而且默认关着的时候,老结果一个字节都不能变。

Four things are pinned here, one per requirement:

1. **byte-identity** — the existing (inert) call sites of ``augment_mjcf_xml`` still produce exact
   pinned bytes.  The pin was deliberately migrated once when the canonical racket collision mesh
   was corrected to the URDF rubber-face thickness; further appender drift remains fail-closed.
2. **one table** — the collidable boxes' pose and extent equal the ``table_frame`` derivation, and
   the sha256 pin that binds ``geometry.py`` as the audit's frame source still holds.
3. **it detects** — a robot pose inside the table volume is caught; a legal pose is not.
4. **no legs** — the documented gap is asserted, so it cannot be silently "fixed" on one simulator.

The MuJoCo-dependent tests skip on a bare host; everything else runs on python 3.8 with numpy only.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import mujoco_table_scene as ts  # noqa: E402

_CANONICAL_MJCF = ts.CANONICAL_MJCF
_PREREG = _REPO / "configs/motion_backhand_loop_b_table_net_clearance_prereg_20260715.json"

#: sha256 of ``augment_mjcf_xml(canonical, geometry)`` after the intentional canonical-racket
#: collision-thickness correction.  This remains the exact byte-identity proof for the two inert
#: call sites: the clearance audit's ``_compile_augmented_model`` and
#: ``tests/test_motion_backhand_loop_b_table_net_clearance.py``.  Historical certificates retain
#: their old source pin and are not silently promoted by this successor baseline.
_BASELINE_INERT_SHA256 = "558135ed4c112a08ee20f389ac28373dbbd643543ae1a037eac7f05972fb5219"
_BASELINE_INERT_BYTES = 49770
_CURRENT_COLLIDABLE_SHA256 = (
    "fac8d51c2f990bcbd10e07d5e3fa1294b90bcba649c3d05f126ee3a017c503b6"
)

try:
    import mujoco
except ImportError:  # pragma: no cover - host without MuJoCo
    mujoco = None

requires_mujoco = pytest.mark.skipif(mujoco is None, reason="MuJoCo is not installed")


def _audit():
    spec = importlib.util.spec_from_file_location(
        "_tn_audit", _REPO / "scripts/audit_motion_schema2_table_net_clearance.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_tn_audit"] = module
    spec.loader.exec_module(module)
    return module


def _canonical_xml():
    return _CANONICAL_MJCF.read_bytes()


def _close(a, b, tol=1e-9):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(float(x) - float(y)) <= tol, f"component {i}: {x} != {y}"


# ------------------------------------------------------- 1. the disk file is never touched ---
def test_the_vendor_mjcf_still_has_no_table_in_it():
    """The whole design rests on this: the table is added in memory, never on disk."""
    text = _canonical_xml().decode("utf-8").lower()
    assert "table" not in text
    assert "motion_table_top" not in text


# ------------------------------------------------------------------- 1. byte-identity proof ---
def test_existing_inert_call_sites_are_byte_identical():
    """Default ``collidable=False`` reproduces the current pinned bytes exactly.

    If this ever fails, a new successor source identity is required — that is the whole point of
    pinning the digest rather than merely re-deriving it.
    """
    audit = _audit()
    out = audit.augment_mjcf_xml(_canonical_xml(), audit._expected_obstacle_geometry())
    assert len(out) == _BASELINE_INERT_BYTES
    assert hashlib.sha256(out).hexdigest() == _BASELINE_INERT_SHA256


def test_frozen_prereg_geometry_still_produces_current_inert_bytes():
    """Reuse only the frozen obstacle geometry; its old model identity remains historical."""
    audit = _audit()
    plan = json.loads(_PREREG.read_text(encoding="utf-8"))
    out = audit.augment_mjcf_xml(_canonical_xml(), plan["obstacle_geometry"])
    assert hashlib.sha256(out).hexdigest() == _BASELINE_INERT_SHA256


def test_the_pinned_validator_script_is_untouched():
    """The audit script is pinned by path+bytes+sha256 as the clearance certificate's validator.

    It re-verifies its own bytes at runtime, so it cannot gain even a keyword argument.  That is
    why the solid variant is a post-transform of its output rather than a parameter on it.
    """
    plan = json.loads(_PREREG.read_text(encoding="utf-8"))
    pin = plan["validator"]
    assert pin["path"] == "scripts/audit_motion_schema2_table_net_clearance.py"
    data = (_REPO / pin["path"]).read_bytes()
    assert len(data) == int(pin["bytes"])
    assert hashlib.sha256(data).hexdigest() == pin["sha256"]
    # and it must never itself ask for solid geometry: it measures with mj_geomDistance, and
    # contacts would perturb the very pose it is measuring.
    assert b"collidable" not in data


def test_collidable_differs_from_inert_in_exactly_one_attribute():
    """Same names, same order, same pose, same size — only ``conaffinity`` moves."""
    rows = ts.obstacle_geometry()
    inert = ts.augment_mjcf_xml(_canonical_xml(), rows, collidable=False)
    collidable = ts.augment_mjcf_xml(_canonical_xml(), rows, collidable=True)
    # the inert path is the audit's own bytes, unmodified
    assert hashlib.sha256(inert).hexdigest() == _BASELINE_INERT_SHA256
    assert hashlib.sha256(collidable).hexdigest() == _CURRENT_COLLIDABLE_SHA256
    assert inert != collidable
    assert len(inert) == len(collidable), "the two variants must differ only in one digit"
    # exactly four characters differ, one per obstacle box, each a '0' -> '7' in conaffinity
    diffs = [i for i, (a, b) in enumerate(zip(inert, collidable)) if a != b]
    assert len(diffs) == 4
    for index in diffs:
        assert inert[index : index + 1] == b"0"
        assert collidable[index : index + 1] == b"7"
        # ...and each differing byte sits inside one of the four appended obstacle geoms, not in
        # any vendor geom (the vendor XML has conaffinity="0" geoms of its own).
        preceding = inert[:index].rsplit(b"<geom", 1)[-1]
        assert any(name.encode() in preceding for name in ts.OBSTACLE_NAMES), (
            f"byte {index} is not inside an appended obstacle geom"
        )
        assert preceding.endswith(b' contype="0" conaffinity="')


def test_collidable_reuses_the_vendor_floor_collision_class():
    """``conaffinity=7`` is the vendor floor's own value, not a number invented here."""
    xml = _canonical_xml().decode("utf-8")
    floor_line = next(line for line in xml.splitlines() if 'name="floor"' in line)
    assert 'conaffinity="7"' in floor_line
    # ...and the robot's collision class carries contype=1, which is what meets it.
    assert 'contype="1" conaffinity="7"' in xml


# ------------------------------------------------------ 2. one table, across both simulators ---
def test_derived_boxes_equal_the_table_frame_derivation():
    """Pose from ``table_frame``, dimensions from ``geometry`` — no constant restated."""
    geometry, table_frame = ts.load_geometry_and_frame()
    near_x, surface_z = ts.virtual_table_pose()
    rows = ts.obstacle_geometry()

    _close(rows["table_top"]["center_mjcf_world_m"],
           table_frame.table_top_center_env(near_x, surface_z))
    _close(rows["table_top"]["full_extents_m"], table_frame.table_top_size())
    _close(rows["table_top"]["full_extents_m"],
           (geometry.TABLE_LENGTH, geometry.TABLE_WIDTH, geometry.TABLE_THICKNESS))

    _close(rows["net"]["center_mjcf_world_m"], table_frame.net_center_env(near_x, surface_z))
    _close(rows["net"]["full_extents_m"], geometry.net_size())

    post_h = geometry.NET_HEIGHT + 0.02
    posts = {p["name"]: p for p in rows["net_posts"]}
    for name, left in (("motion_net_post_left", True), ("motion_net_post_right", False)):
        _close(posts[name]["center_mjcf_world_m"],
               table_frame.net_post_center_env(near_x, surface_z, left=left, post_height=post_h))
        _close(posts[name]["full_extents_m"], (0.02, 0.02, post_h))


def test_action_ball_policy_geometry_is_the_exact_five_solid_training_assembly():
    """Formal policy safety adds only the training task's robot keep-out."""

    _geometry, table_frame = ts.load_geometry_and_frame()
    near_x, surface_z = ts.virtual_table_pose()
    rows = ts.action_ball_policy_obstacle_geometry()
    ordered = ts.action_ball_policy_obstacle_rows(rows)
    assert tuple(row["name"] for row in ordered) == (
        ts.ACTION_BALL_POLICY_OBSTACLE_NAMES
    )
    expected_aabbs = table_frame.table_assembly_aabbs_env(
        near_x,
        surface_z,
        keepout_floor_z=0.0,
        margin=0.0,
    )
    for row, (lo, hi) in zip(ordered, expected_aabbs):
        expected_center = [
            0.5 * (float(low) + float(high))
            for low, high in zip(lo, hi)
        ]
        expected_extents = [
            float(high) - float(low)
            for low, high in zip(lo, hi)
        ]
        _close(row["center_mjcf_world_m"], expected_center, tol=1.0e-12)
        _close(row["full_extents_m"], expected_extents, tol=1.0e-12)
    contract = ts.action_ball_policy_geometry_contract(rows)
    assert contract["payload"]["obstacle_order"] == list(
        ts.ACTION_BALL_POLICY_OBSTACLE_NAMES
    )
    assert len(contract["sha256"]) == 64
    assert contract == ts.action_ball_policy_geometry_contract(rows)


def test_action_ball_keepout_closes_the_legacy_under_table_tunnel():
    """A point below the slab is legal in the legacy scene but not ActionBall."""

    point = (1.0, 0.0, 0.35)
    top_lo, top_hi = ts.table_top_aabb()
    assert ts.point_penetration_m(point, top_lo, top_hi) == 0.0
    rows = ts.action_ball_policy_obstacle_geometry()
    keepout = rows["robot_keepout"]
    center = keepout["center_mjcf_world_m"]
    half = [0.5 * value for value in keepout["full_extents_m"]]
    keepout_lo = [c - h for c, h in zip(center, half)]
    keepout_hi = [c + h for c, h in zip(center, half)]
    assert ts.point_penetration_m(
        point, keepout_lo, keepout_hi
    ) == pytest.approx(0.35)


def test_derived_boxes_equal_the_frozen_prereg_the_audit_uses():
    """The live derivation and the audit's frozen numbers are the same table.

    The inert audit path and the new collidable path must not be able to drift apart; this is the
    assertion that keeps them married.
    """
    rows = ts.obstacle_geometry()
    frozen = json.loads(_PREREG.read_text(encoding="utf-8"))["obstacle_geometry"]
    assert rows["primitive"] == frozen["primitive"]
    for key in ("table_top", "net"):
        _close(rows[key]["center_mjcf_world_m"], frozen[key]["center_mjcf_world_m"])
        _close(rows[key]["full_extents_m"], frozen[key]["full_extents_m"])
    live = {p["name"]: p for p in rows["net_posts"]}
    frz = {p["name"]: p for p in frozen["net_posts"]}
    assert set(live) == set(frz)
    for name in live:
        _close(live[name]["center_mjcf_world_m"], frz[name]["center_mjcf_world_m"])
        _close(live[name]["full_extents_m"], frz[name]["full_extents_m"])


def test_geometry_module_sha_pin_still_holds():
    """``geometry.py`` is sha256-pinned as the audit's bound frame source; we did not edit it."""
    bound = json.loads(_PREREG.read_text(encoding="utf-8"))["frame_sources"]["table_geometry"]
    data = (_REPO / bound["path"]).read_bytes()
    assert len(data) == int(bound["bytes"])
    assert hashlib.sha256(data).hexdigest() == bound["sha256"]


def test_table_pose_is_read_from_the_live_command_cfg():
    """If the trainer's table moves, every number here moves with it."""
    assert ts.virtual_table_pose() == (0.5, 0.76)
    src = (_REPO / "hope_training/whole_body_tracking/source/whole_body_tracking"
           / "whole_body_tracking/tasks/tracking/mdp/hope_commands.py").read_text("utf-8")
    assert "vb_table_near_x: float = 0.5" in src
    assert "vb_table_surface_z: float = 0.76" in src


def test_append_order_matches_the_audit_module():
    """The +4 geom-id shift the audit normalises depends on this order."""
    assert ts.OBSTACLE_NAMES == tuple(_audit().OBSTACLE_NAMES)
    assert _audit().ROBOT_GEOM_INDEX_SHIFT == len(ts.OBSTACLE_NAMES)


# ------------------------------------------------------------ 3. it detects the wrong poses ---
def test_the_slab_occupies_the_z_band_the_forehand_defect_lives_in():
    """The table volume is z in [0.71, 0.76] over the table footprint — not empty air."""
    lo, hi = ts.table_top_aabb()
    _close(lo, (0.5, -0.7625, 0.71))
    _close(hi, (3.24, 0.7625, 0.76))


def test_point_inside_the_table_is_detected_and_a_legal_point_is_not():
    lo, hi = ts.table_top_aabb()
    # dead centre of the slab: 25 mm below the surface, half the 50 mm thickness
    assert ts.point_penetration_m((1.87, 0.0, 0.735), lo, hi) == pytest.approx(0.025)
    # 10 mm below the surface -> 10 mm in (nearest face is the top face)
    assert ts.point_penetration_m((1.87, 0.0, 0.75), lo, hi) == pytest.approx(0.010)
    # legal: a strike point well above the surface
    assert ts.point_penetration_m((1.87, 0.0, 0.95), lo, hi) == 0.0
    # legal: behind the near edge, where the robot stands
    assert ts.point_penetration_m((0.0, 0.0, 0.735), lo, hi) == 0.0
    # legal: beside the table
    assert ts.point_penetration_m((1.87, 1.2, 0.735), lo, hi) == 0.0


def test_the_documented_no_legs_gap_is_real_and_is_not_silently_patched():
    """Under the overhang touches NOTHING, exactly as Isaac's robot_hit_table documents.

    Asserted rather than merely commented so that "someone adds legs to MuJoCo only" becomes a
    test failure instead of a silent divergence between the two simulators.
    """
    lo, _hi = ts.table_top_aabb()
    # z = 0.694 m is the bound forehand strike height from EXP-MOTION-CANONICAL-LIBRARY-20260723
    # section 7.4.  It is BELOW the slab's underside (0.71) -> it is under the overhang.
    assert 0.694 < lo[2]
    assert ts.point_penetration_m((1.0, 0.0, 0.694), lo, _hi) == 0.0
    rows = ts.obstacle_geometry()
    assert [r["name"] for r in [rows["table_top"], rows["net"], *rows["net_posts"]]] == list(
        ts.OBSTACLE_NAMES
    ), "no leg box may appear without a matching Isaac collider"
    assert not any("leg" in name for name in ts.OBSTACLE_NAMES)


# -------------------------------------------------------------- 3. the same, but in MuJoCo ---
@requires_mujoco
def test_compiled_model_gains_exactly_four_world_boxes():
    canonical = mujoco.MjModel.from_xml_path(str(_CANONICAL_MJCF))
    scene = ts.load_table_scene(mujoco, _CANONICAL_MJCF, collidable=True)
    assert int(scene.model.ngeom) == int(canonical.ngeom) + 4
    assert int(scene.model.nbody) == int(canonical.nbody)  # world boxes add no bodies
    assert int(canonical.nbody) == 33 and int(canonical.ngeom) == 79  # the pinned contract
    for name, gid in scene.obstacle_geom_ids.items():
        assert int(scene.model.geom_bodyid[gid]) == 0
        assert int(scene.model.geom_conaffinity[gid]) == 7
        assert int(scene.model.geom_contype[gid]) == 0


@requires_mujoco
def test_action_ball_policy_model_gains_exactly_five_world_solids():
    canonical = mujoco.MjModel.from_xml_path(str(_CANONICAL_MJCF))
    scene = ts.load_table_scene(
        mujoco,
        _CANONICAL_MJCF,
        collidable=True,
        action_ball_policy=True,
    )
    assert int(scene.model.ngeom) == int(canonical.ngeom) + 5
    assert scene.obstacle_names == ts.ACTION_BALL_POLICY_OBSTACLE_NAMES
    assert scene.geom_index_shift == 5
    for name in ts.ACTION_BALL_POLICY_OBSTACLE_NAMES:
        gid = scene.obstacle_geom_ids[name]
        assert int(scene.model.geom_bodyid[gid]) == 0
        assert int(scene.model.geom_contype[gid]) == 0
        assert int(scene.model.geom_conaffinity[gid]) == 7


@requires_mujoco
def test_compiled_box_pose_and_extent_equal_the_table_frame_derivation():
    scene = ts.load_table_scene(mujoco, _CANONICAL_MJCF, collidable=True)
    rows = ts.obstacle_geometry()
    by_name = {rows["table_top"]["name"]: rows["table_top"], rows["net"]["name"]: rows["net"]}
    by_name.update({p["name"]: p for p in rows["net_posts"]})
    for name, gid in scene.obstacle_geom_ids.items():
        _close(scene.model.geom_pos[gid], by_name[name]["center_mjcf_world_m"], tol=1e-12)
        _close(
            [2.0 * v for v in scene.model.geom_size[gid]],
            by_name[name]["full_extents_m"],
            tol=1e-12,
        )


@requires_mujoco
def test_a_pose_inside_the_table_is_detected_and_the_stand_pose_is_not():
    """The real detector, on the real model: robot in the table vs robot standing legally."""
    import numpy as np

    scene = ts.load_table_scene(mujoco, _CANONICAL_MJCF, collidable=True)
    data = mujoco.MjData(scene.model)

    # legal: the vendor 'stand' keyframe, robot behind the near edge at x ~ -0.04
    mujoco.mj_resetDataKeyframe(scene.model, data, 0)
    mujoco.mj_forward(scene.model, data)
    assert data.qpos[0] < 0.5, "the stand keyframe must be behind the table's near edge"
    assert ts.frame_table_contacts(mujoco, scene, data, 0) == []

    # illegal: same posture, teleported into the middle of the slab
    mujoco.mj_resetDataKeyframe(scene.model, data, 0)
    data.qpos[0] = 1.87           # mid-table in x
    data.qpos[1] = 0.0            # centred in y
    data.qpos[2] = 0.735 - 0.30   # drop the pelvis so the torso occupies the slab band
    mujoco.mj_forward(scene.model, data)
    contacts = ts.frame_table_contacts(mujoco, scene, data, 0)
    assert contacts, "a robot standing inside the table volume must be detected"
    assert any(c.obstacle == "motion_table_top" for c in contacts)
    assert max(c.depth_m for c in contacts) > 0.0
    summary = ts.summarize_contacts(contacts)
    assert summary["strikes_table"] is True
    assert summary["max_penetration_m"] > 0.0
    assert np.isfinite(summary["max_penetration_m"])


@requires_mujoco
def test_action_ball_policy_scene_rejects_robot_under_the_table():
    """The old four-box scene permits this translated stand; five-solid does not."""

    legacy = ts.load_table_scene(
        mujoco, _CANONICAL_MJCF, collidable=True
    )
    formal = ts.load_table_scene(
        mujoco,
        _CANONICAL_MJCF,
        collidable=True,
        action_ball_policy=True,
    )
    legacy_data = mujoco.MjData(legacy.model)
    formal_data = mujoco.MjData(formal.model)
    for model, data in (
        (legacy.model, legacy_data),
        (formal.model, formal_data),
    ):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[0] = 1.0
        data.qpos[1] = 0.0
        mujoco.mj_forward(model, data)
    legacy_contacts = ts.frame_table_contacts(
        mujoco, legacy, legacy_data, 0
    )
    formal_contacts = ts.frame_table_contacts(
        mujoco, formal, formal_data, 0
    )
    assert not any(
        contact.obstacle == ts.ACTION_BALL_ROBOT_KEEPOUT_NAME
        for contact in legacy_contacts
    )
    assert any(
        contact.obstacle == ts.ACTION_BALL_ROBOT_KEEPOUT_NAME
        for contact in formal_contacts
    )


@requires_mujoco
def test_collision_disabled_ball_passes_through_robot_only_keepout():
    """The fifth solid changes robot safety only, never the fitted ball path."""

    base = b"""<mujoco>
      <option timestep="0.001" gravity="0 0 0"/>
      <worldbody>
        <body name="test_ball" pos="0.30 0 0.35">
          <freejoint name="test_ball_joint"/>
          <geom name="test_ball_geom" type="sphere" size="0.02"
                mass="0.0034" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>"""
    rows = ts.action_ball_policy_obstacle_geometry()
    four = ts.augment_mjcf_xml(base, rows, collidable=True)
    five = ts.append_action_ball_policy_keepout_xml(
        four, rows, collidable=True
    )
    model = mujoco.MjModel.from_xml_string(five.decode("utf-8"))
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "test_ball_joint"
    )
    dof = int(model.jnt_dofadr[joint])
    data.qvel[dof : dof + 3] = (4.0, 0.0, 0.0)
    for _ in range(500):
        mujoco.mj_step(model, data)
        assert int(data.ncon) == 0
    qpos = int(model.jnt_qposadr[joint])
    assert data.qpos[qpos] == pytest.approx(2.30, abs=1.0e-9)
    assert data.qvel[dof] == pytest.approx(4.0, abs=1.0e-12)


@requires_mujoco
def test_inert_augmentation_generates_no_contacts_even_inside_the_table():
    """``collidable=False`` really is measurement-only: the audit's behaviour is unchanged."""
    scene = ts.load_table_scene(mujoco, _CANONICAL_MJCF, collidable=False)
    data = mujoco.MjData(scene.model)
    mujoco.mj_resetDataKeyframe(scene.model, data, 0)
    data.qpos[0] = 1.87
    data.qpos[1] = 0.0
    data.qpos[2] = 0.735 - 0.30
    mujoco.mj_forward(scene.model, data)
    assert ts.frame_table_contacts(mujoco, scene, data, 0) == []


@requires_mujoco
def test_isaac_parity_selection_mutes_the_net_and_posts():
    """``ISAAC_EQUIVALENT_OBSTACLES`` reproduces the Isaac scene, which spawns the slab only."""
    scene = ts.load_table_scene(
        mujoco, _CANONICAL_MJCF, collidable=True, obstacles=ts.ISAAC_EQUIVALENT_OBSTACLES
    )
    live = {
        name
        for name, gid in scene.obstacle_geom_ids.items()
        if int(scene.model.geom_conaffinity[gid]) != 0
    }
    assert live == {"motion_table_top"}


@requires_mujoco
def test_the_robot_still_binds_by_name_with_the_table_attached():
    """The +4 geom shift must not break the name-resolved bindings the tools rely on."""
    scene = ts.load_table_scene(mujoco, _CANONICAL_MJCF, collidable=True)
    canonical = mujoco.MjModel.from_xml_path(str(_CANONICAL_MJCF))
    assert int(scene.model.nq) == int(canonical.nq)
    assert int(scene.model.nv) == int(canonical.nv)
    assert int(scene.model.nu) == int(canonical.nu)
    for name in ("pelvis_link", "right_wrist_yaw_Link", "left_ankle_roll_Link"):
        assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name) == mujoco.mj_name2id(
            canonical, mujoco.mjtObj.mjOBJ_BODY, name
        )
    site = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
    assert site >= 0
    # geom ids shift by exactly +4, which is what ROBOT_GEOM_INDEX_SHIFT encodes
    canonical_floor = mujoco.mj_name2id(canonical, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    scene_floor = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert canonical_floor == scene_floor == 0
    assert sorted(scene.obstacle_geom_ids.values()) == [1, 2, 3, 4]
