"""Agibot A3 — HOPE ping-pong WBC (BeyondMimic + HITTER racket-target tracking).

This is the step-13 environment. It extends the A3 motion-tracking baseline
(:class:`AgibotA3FlatEnvCfg`) with the HITTER racket objective:

* a :class:`RacketTargetCommand` that samples the desired racket state (position/velocity/normal)
  and desired base XY each swing, and computes the actual racket state by FK through ``T_mount``;
* HOPE actor observations (desired racket pos rel-base, desired racket vel/normal world,
  time-to-strike, desired base XY rel-base) plus projected gravity, with privileged actual racket
  state on the critic;
* HITTER goal rewards (base-position before strike; racket pos/vel/normal in a window around strike),
  on top of the BeyondMimic imitation reward and the regularization reward;
* extended domain randomization for sim-to-real.

Default usage trains one unified forehand+backhand policy by passing two reference clips
(``registry_name`` + ``registry_name_2``). The swing-type observation is present on the actor so
one policy can condition on which clip/target family it is currently imitating.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.agibot_a3 import A3_FEET_BODIES, A3_HAND_BODIES, A3_UPPER_TRACKED
from whole_body_tracking.tasks.tracking.config.agibot_a3.flat_env_cfg import (
    A3_NON_FOOT_BODY_REGEX,
    AgibotA3FlatEnvCfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# SHADOW physical ball + table scene attachment (flag-gated, METRICS-ONLY; shadow_ball.py).
##


def attach_shadow_ball_scene(env_cfg, *, shadow_table: bool) -> None:
    """Attach the shadow-ball scene entities for ``RacketTargetCommandCfg.shadow_ball``.

    Adds to ``env_cfg.scene`` (per-env cloned assets via the ``{ENV_REGEX_NS}`` regex path, the
    same pattern as the table_tennis scene builders):

    * ``shadow_ball`` — one dynamic sphere per env. Radius/mass come from
      ``configs/ball_physics_venue.yaml`` (the fitted coated match ball: R=0.02 m, m=3.4 g), zero
      linear/angular damping and gyroscopic forces OFF so PhysX integrates gravity ONLY and the
      per-substep venue aero wrench supplies drag+Magnus (omega stays constant in flight, matching
      the fit — see scripts/isaac_ball_inloop_check.py). The collider is DISABLED unless
      ``shadow_table`` (pure flight, nothing to touch — the strike is applied analytically, never
      by PhysX contact).
    * ``shadow_table`` / ``shadow_table_visual`` (only when ``shadow_table=True``) — the
      table_tennis static table-top collider (invisible cuboid, multiplicative-restitution
      material) + the visual USD mesh, both placed at the TRACKING task's virtual-table pose:
      near edge at env-local ``x = vb_table_near_x``, surface at ``z = vb_table_surface_z``,
      centered on ``y = 0`` (hope_commands landmark convention). NO net collider: the virtual-ball
      reward model gates the net analytically, and a physical net would make the engine flight
      diverge from the analytic reference being cross-checked.

    Idempotent (train.py may call it after ``__post_init__`` already did — the same post-init
    override timing as ``face_command_obs``). METRICS-ONLY: nothing here is read by rewards/obs.
    """
    if bool(getattr(env_cfg, "table_robot_keepout", False)):
        raise RuntimeError(
            "shadow-ball physics cannot coexist with the ActionBall robot-only "
            "under-table keep-out proxy"
        )
    if getattr(env_cfg.scene, "shadow_ball", None) is not None:
        return  # already attached (cfg-flag path ran before the train.py override path)

    import yaml as _yaml

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    from whole_body_tracking.tasks.table_tennis import geometry as tt_geom
    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame
    from whole_body_tracking.tasks.table_tennis import table_tennis_env_cfg as tt_cfg
    from whole_body_tracking.tasks.tracking.mdp.virtual_ball import default_venue_yaml_path

    with open(default_venue_yaml_path(), "r") as fh:
        _ball_raw = _yaml.safe_load(fh)["ball"]
    ball_r = float(_ball_raw["radius"])   # 0.02 m
    ball_m = float(_ball_raw["mass"])     # 0.0034 kg (coated match ball)

    rt = env_cfg.commands.racket_target
    near_x = float(rt.vb_table_near_x)
    surface_z = float(rt.vb_table_surface_z)
    mats = tt_geom.BounceMaterials()

    env_cfg.scene.shadow_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShadowBall",
        # Spawn parked below the floor; the driver rewrites the root state every control step.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
        spawn=sim_utils.SphereCfg(
            radius=ball_r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,   # m/s
                max_angular_velocity=1.0e5,   # deg/s (PhysX uses deg/s here) ~ 1745 rad/s
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=False,  # omega constant in flight (venue-fit assumption)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_m),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=bool(shadow_table)),
            physics_material=tt_cfg._surface_material(
                mats.ball_restitution, mats.ball_static_friction, mats.ball_dynamic_friction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.05), roughness=0.6),
        ),
    )

    if shadow_table:
        # Static table-top collider at the virtual-table pose (top face at env-local surface_z),
        # same slab + multiplicative-restitution material as table_tennis.build_table_top_cfg.
        # Pose comes from table_frame.table_top_center_env — the shared HOPE->env translation, not a
        # locally re-derived expression (one source of truth for where the table is).
        env_cfg.scene.shadow_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ShadowTable",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tt_frame.table_top_center_env(near_x, surface_z)
            ),
            spawn=sim_utils.CuboidCfg(
                size=tt_geom.table_top_size(),
                visible=False,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=tt_cfg._surface_material(
                    mats.table_restitution, mats.table_static_friction, mats.table_dynamic_friction
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.32, 0.55), roughness=0.5),
            ),
        )
        # Visual-only USD table+net mesh overlaid on the invisible collider (no PhysX from it —
        # the base USD layer carries no colliders; see table_tennis_env_cfg notes). Its local
        # origin is the floor point under the table center (surface at local z=0.76).
        env_cfg.scene.shadow_table_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ShadowTableVisual",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tt_frame.table_visual_origin_env(near_x, surface_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(usd_path=tt_cfg._TABLE_USD_PATH),
        )


##
# PHYSICAL ball + table scene attachment — Phase A/B truth instrument (flag-gated,
# METRICS-ONLY; physical_ball.py). Distinct from the shadow-ball entities so both instruments
# can coexist.
##


def attach_physical_ball_scene(env_cfg) -> None:
    """Attach the physical-ball scene entities for ``physical_ball=True`` (Phase A/B).

    Adds to ``env_cfg.scene`` (per-env cloned assets via the ``{ENV_REGEX_NS}`` regex path, the
    table_tennis scene-builder pattern):

    * ``pb_ball`` — one dynamic sphere per env. Radius/mass from
      ``configs/ball_physics_venue.yaml`` (fitted coated match ball: R=0.02 m, m=3.4 g), zero
      linear/angular damping, gyroscopic forces OFF (omega constant in flight, the venue-fit
      assumption; the per-substep aero wrench supplies drag+Magnus — PhysX integrates gravity
      only; validated by scripts/isaac_ball_inloop_check.py, 17 mm systematic vs venue RK4).
      COLLIDER DISABLED + material restitution 0 (neutralized): PhysX never resolves the ball's
      contacts — the fitted CODE-DRIVEN table bounce is the single bounce authority. Phase A
      passes through the robot; Phase B adds its own blade-disc scan and fitted venue impulse
      while keeping the collider disabled, so PhysX cannot double-hit the ball. Scene-level CCD
      remains off because every ball collision pair is filtered and the sign-crossing scan owns
      anti-tunneling.
    * ``pb_table`` / ``pb_table_visual`` — the table_tennis static table-top collider (invisible
      cuboid) + visual USD mesh at the TRACKING task's virtual-table pose. FRAME RECONCILIATION:
      the table_tennis/HOPE frame has the near edge at x=0, y in [-TABLE_WIDTH, 0], surface z=0;
      the tracking env frame wants near edge x=vb_table_near_x, table CENTERED on y=0, surface
      z=vb_table_surface_z — i.e. a pure translation (dx, dy, dz) = (+near_x, +TABLE_WIDTH/2,
      +surface_z) applied to every table_tennis landmark (same shift attach_shadow_ball_scene
      uses). No net collider: Phase A only needs the bounce surface, and the vb reward model
      gates the net analytically.

    Idempotent (train.py may call it after ``__post_init__`` already did — the face_command_obs
    override timing). METRICS-ONLY: nothing here is read by rewards/obs.
    """
    if bool(getattr(env_cfg, "table_robot_keepout", False)):
        raise RuntimeError(
            "physical-ball physics cannot coexist with the ActionBall robot-only "
            "under-table keep-out proxy"
        )
    if getattr(env_cfg.scene, "pb_ball", None) is not None:
        return  # already attached (cfg-flag path ran before the train.py override path)

    import yaml as _yaml

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    from whole_body_tracking.tasks.table_tennis import geometry as tt_geom
    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame
    from whole_body_tracking.tasks.table_tennis import table_tennis_env_cfg as tt_cfg
    from whole_body_tracking.tasks.tracking.mdp.virtual_ball import default_venue_yaml_path

    with open(default_venue_yaml_path(), "r") as fh:
        _ball_raw = _yaml.safe_load(fh)["ball"]
    ball_r = float(_ball_raw["radius"])   # 0.02 m
    ball_m = float(_ball_raw["mass"])     # 0.0034 kg (coated match ball)

    rt = env_cfg.commands.racket_target
    near_x = float(rt.vb_table_near_x)
    surface_z = float(rt.vb_table_surface_z)
    mats = tt_geom.BounceMaterials()

    env_cfg.scene.pb_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PhysicalBall",
        # Spawn parked below the floor; the manager rewrites the root state every control step.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
        spawn=sim_utils.SphereCfg(
            radius=ball_r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,   # drag comes from the aero wrench only (no double-count)
                angular_damping=0.0,  # omega constant in flight (venue-fit assumption)
                max_linear_velocity=1000.0,   # m/s
                max_angular_velocity=1.0e5,   # deg/s (PhysX uses deg/s here) ~ 1745 rad/s
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_m),
            # Phase A/B collision filter: collider OFF -> no ball<->robot and no ball<->table
            # PhysX contact. The fitted code-driven table bounce owns the table; Phase B keeps
            # this filter and adds its own blade-disc scan/venue impulse. Restitution 0 is a
            # belt-and-suspenders neutralization if a future asset accidentally gains a pair.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            physics_material=tt_cfg._surface_material(0.0, mats.ball_static_friction, mats.ball_dynamic_friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.4),
        ),
    )

    # Real static table-top collider at the virtual-table pose (top face at env-local surface_z):
    # the table_tennis slab translated by (+near_x, +TABLE_WIDTH/2, +surface_z) — see docstring.
    # Restitution 0 here too (neutralized): even if a future asset gains a collider pair with it,
    # PhysX must never add bounce energy on top of the code-driven contact model.
    env_cfg.scene.pb_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PhysicalTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tt_frame.table_top_center_env(near_x, surface_z)
        ),
        spawn=sim_utils.CuboidCfg(
            size=tt_geom.table_top_size(),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=tt_cfg._surface_material(0.0, mats.table_static_friction, mats.table_dynamic_friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.32, 0.55), roughness=0.5),
        ),
    )
    # Visual-only USD table+net mesh overlaid on the invisible collider (no PhysX from it — the
    # base USD layer carries no colliders; see table_tennis_env_cfg notes). Its local origin is
    # the floor point under the table center (surface at local z=0.76).
    env_cfg.scene.pb_table_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PhysicalTableVisual",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tt_frame.table_visual_origin_env(near_x, surface_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=tt_cfg._TABLE_USD_PATH),
    )


##
# TABLE OBSTACLE — the table the ROBOT is not allowed to hit.  DEFAULT ON.
##


#: Prim name of the default table collider.  ``attach_table_obstacle`` may instead record the
#: shadow/physical table prim when one of those truth instruments already supplies the same slab.
TABLE_OBSTACLE_PRIM = "{ENV_REGEX_NS}/TableObstacle"
#: ActionBall-only conservative robot keep-out below the regulation top slab.  It fills the
#: table footprint from the tracking-scene floor to the slab underside, but is never enabled in a
#: scene with a dynamic/shadow physics ball.
TABLE_ROBOT_KEEPOUT_PRIM = "{ENV_REGEX_NS}/TableRobotKeepout"
TABLE_NET_PRIM = "{ENV_REGEX_NS}/TableNet"
TABLE_NET_POST_LEFT_PRIM = "{ENV_REGEX_NS}/TableNetPostLeft"
TABLE_NET_POST_RIGHT_PRIM = "{ENV_REGEX_NS}/TableNetPostRight"
#: Exact rigid-body order produced by the shipped A3 URDF: root followed by the 31 non-fixed joint
#: children.  The full-assembly pose keep-out reads these articulation bodies in this exact order.
#: Keeping the order explicit makes a changed A3 asset fail closed without constructing or reading
#: the broken 32-body filtered-contact matrix that previously dominated scene creation and
#: collection time.
#: Fixed visual/collision children (including every racket mesh) are merged into their parent body;
#: in particular the racket is carried by ``right_wrist_yaw_Link``.
TABLE_CONTACT_BODY_NAMES = (
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "head_yaw_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "head_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
)
#: Preserve the historic wrist sensor name for legacy top-only tasks.  ActionBall's full table
#: assembly deliberately installs no pair-filtered sensor; it uses live articulation pose plus
#: conservative geometry.
TABLE_CONTACT_SENSOR_NAME = "racket_table_contact"
TABLE_CONTACT_SENSOR_PRIM = "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link"
TABLE_FULL_CONTACT_SENSOR_ROLES = (
    "top",
    "keepout",
    "net",
    "post_left",
    "post_right",
)
TABLE_FULL_CONTACT_SENSOR_NAMES = (
    "table_top_robot_contact",
    "table_keepout_robot_contact",
    "table_net_robot_contact",
    "table_post_left_robot_contact",
    "table_post_right_robot_contact",
)
TABLE_FULL_CONTACT_SENSOR_PRIMS = (
    TABLE_OBSTACLE_PRIM,
    TABLE_ROBOT_KEEPOUT_PRIM,
    TABLE_NET_PRIM,
    TABLE_NET_POST_LEFT_PRIM,
    TABLE_NET_POST_RIGHT_PRIM,
)
#: Exact 43 collision-component OBB artifact: vendor ping-pong URDF collision bytes are folded
#: through 11 fixed children into the 32 live A3 rigid bodies.  The artifact also binds the
#: reviewed six-file Pod runtime USD tree; the DoneTerm refuses a file/body-order drift.
TABLE_COLLISION_PROXY_ARTIFACT_PATH = (
    "configs/a3_table_collision_proxy_20260731/"
    "a3_table_collision_components.v1.json"
)
TABLE_COLLISION_PROXY_ARTIFACT_SHA256 = (
    "23e2f5b30bbba909f1123dc41f6c010354122b9837b4ef133a1c285a2cd78ca8"
)
TABLE_CONTACT_RACKET_BODY_NAME = "right_wrist_yaw_Link"
#: Conservative OBB for the shipped ``right_racket_face_collision.STL`` in the wrist-yaw frame.
#: The centre is the pinned MuJoCo collision-geom offset.  The half extents are the measured
#: 0.081019/0.007001/0.081019 m mesh bounds rounded outward by about one millimetre.
TABLE_RACKET_BLADE_CENTER_OFFSET_WRIST_M = (0.206194, 0.025474, 0.028020)
TABLE_RACKET_BLADE_HALF_EXTENTS_M = (0.082, 0.008, 0.082)


def attach_table_obstacle(env_cfg, *, visual: bool = True) -> None:
    """Put the ping-pong table in the training scene as a solid obstacle.

    人话:训练场里补上真桌子。以前机器人是在"桌子不存在"的世界里学挥拍——指令让它去
    z≈0.65 m 抓球点,在仿真里那是空气,在现实里那是桌面下面。现在那里有实体,撞上就
    像摔倒一样结束这一局并扣分。

    WHAT THIS IS NOT.  The two other tables in this file (``shadow_table``, ``pb_table``) are
    METRICS-ONLY truth instruments: they exist so a ball's bounce can be measured against engine
    physics, they are flag-gated, and both are OFF in every training arm.  This one is a TASK
    OBJECT.  It is on by default and the robot is scored against it.

    GEOMETRY.  Reuses ``table_tennis.geometry`` / ``table_tennis_env_cfg`` verbatim: the ITTF top
    slab, regulation net and the already-preregistered 20 mm post proxies are translated through
    ``table_frame``.  The tracked visual USD contains table legs, but its physics layer applies one
    whole-mesh convex hull; using that hull would falsely fill free space.  ActionBall therefore
    adds a separately named, robot-only conservative keep-out under the top: same XY footprint,
    floor to slab underside, with no overlap.  This is a safety proxy, not a claim about leg shape.
    It is prohibited when a dynamic/shadow physics ball exists, so ball physics continues to use
    only the real top/net/post primitives.

    IDEMPOTENT, and it stands down for the truth instruments: when ``shadow_table`` or
    ``pb_table`` is already attached, that collider IS the table (same slab, same pose) and a
    second overlapping static box would only add PhysX pairs.  In that case this records the
    existing prim on the env cfg instead of spawning a duplicate.
    """
    scene = env_cfg.scene

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from whole_body_tracking.tasks.table_tennis import geometry as tt_geom
    from whole_body_tracking.tasks.table_tennis import table_frame as tt_frame
    from whole_body_tracking.tasks.table_tennis import table_tennis_env_cfg as tt_cfg

    rt = env_cfg.commands.racket_target
    near_x = float(rt.vb_table_near_x)
    surface_z = float(rt.vb_table_surface_z)
    mats = tt_geom.BounceMaterials()
    full_assembly = bool(getattr(env_cfg, "table_robot_keepout", False))

    def robot_proxy_rigid_props():
        """Use the pinned backend's GPU-replicable fixed-body representation.

        Collision-only static ``AssetBaseCfg`` cuboids exhibit pathological
        4096-environment construction scaling in the pinned Isaac Sim build.
        Gravity-free kinematic bodies keep the same infinite-mass collision
        response and use the normal batched replication path.  They still own
        no ContactReportAPI: the robot's existing whole-body
        ``contact_forces`` sensor remains the sole reporter.
        """

        if not full_assembly:
            return None
        return sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        )

    truth_tops = [
        (existing, prim)
        for existing, prim in (
        ("shadow_table", "{ENV_REGEX_NS}/ShadowTable"),
        ("pb_table", "{ENV_REGEX_NS}/PhysicalTable"),
        )
        if getattr(scene, existing, None) is not None
    ]
    if len(truth_tops) > 1:
        raise RuntimeError(
            "shadow and physical table truth instruments cannot coexist; "
            "they would create overlapping top colliders"
        )
    if full_assembly and truth_tops:
        raise RuntimeError(
            "ActionBall full table-contact assembly cannot reuse a shadow/physical truth top: "
            "the robot-only keep-out cannot coexist with a dynamic truth ball"
        )
    top_prim = truth_tops[0][1] if truth_tops else ""
    if top_prim and getattr(scene, "table_obstacle", None) is not None:
        # Hydra late overrides still run before Gym construction.  Retire the earlier default
        # top instead of leaving it stacked under the newly selected truth-instrument top.
        scene.table_obstacle = None
    if not top_prim:
        if getattr(scene, "table_obstacle", None) is None:
            scene.table_obstacle = AssetBaseCfg(
                prim_path=TABLE_OBSTACLE_PRIM,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=tt_frame.table_top_center_env(near_x, surface_z)
                ),
                spawn=sim_utils.CuboidCfg(
                    size=tt_geom.table_top_size(),
                    # Kinematic is the fast replicated fixed-collider representation on the
                    # pinned backend.  It does not request ContactReportAPI or pair-filter views.
                    rigid_props=robot_proxy_rigid_props(),
                    activate_contact_sensors=False,
                    # Invisible collision source + the tracked visual USD.  The visual base layer
                    # carries no PhysX collision API.
                    visible=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),
                    physics_material=tt_cfg._surface_material(
                        mats.table_restitution,
                        mats.table_static_friction,
                        mats.table_dynamic_friction,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.32, 0.55), roughness=0.5
                    ),
                ),
            )
        top_prim = TABLE_OBSTACLE_PRIM
    env_cfg.table_obstacle_prim = top_prim

    filter_prims = [top_prim]
    if full_assembly:
        dynamic_ball_present = (
            bool(getattr(env_cfg, "physical_ball", False))
            or bool(getattr(rt, "physical_ball", False))
            or bool(getattr(rt, "shadow_ball", False))
            or getattr(scene, "pb_ball", None) is not None
            or getattr(scene, "shadow_ball", None) is not None
        )
        if dynamic_ball_present:
            raise RuntimeError(
                "ActionBall robot-only table keep-out cannot coexist with a physical/shadow "
                "ball; it is a conservative robot safety proxy, not ball collision geometry"
            )
        underside_z = surface_z - float(tt_geom.TABLE_THICKNESS)
        if underside_z <= 0.0:
            raise ValueError(
                "table robot keep-out requires the slab underside above the tracking floor"
            )
        if getattr(scene, "table_robot_keepout", None) is None:
            top_center = tt_frame.table_top_center_env(near_x, surface_z)
            scene.table_robot_keepout = AssetBaseCfg(
                prim_path=TABLE_ROBOT_KEEPOUT_PRIM,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(top_center[0], top_center[1], underside_z / 2.0)
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(
                        float(tt_geom.TABLE_LENGTH),
                        float(tt_geom.TABLE_WIDTH),
                        underside_z,
                    ),
                    rigid_props=robot_proxy_rigid_props(),
                    activate_contact_sensors=False,
                    visible=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),
                    physics_material=tt_cfg._surface_material(
                        mats.table_restitution,
                        mats.table_static_friction,
                        mats.table_dynamic_friction,
                    ),
                ),
            )
        if getattr(scene, "table_net", None) is None:
            scene.table_net = AssetBaseCfg(
                prim_path=TABLE_NET_PRIM,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=tt_frame.net_center_env(near_x, surface_z)
                ),
                spawn=sim_utils.CuboidCfg(
                    size=tt_geom.net_size(),
                    rigid_props=robot_proxy_rigid_props(),
                    activate_contact_sensors=False,
                    visible=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),
                    physics_material=tt_cfg._surface_material(
                        mats.net_restitution,
                        mats.net_static_friction,
                        mats.net_dynamic_friction,
                    ),
                ),
            )
        post_size = tt_cfg.net_post_size()
        for attr, prim, left in (
            ("table_net_post_left", TABLE_NET_POST_LEFT_PRIM, True),
            ("table_net_post_right", TABLE_NET_POST_RIGHT_PRIM, False),
        ):
            if getattr(scene, attr, None) is None:
                setattr(
                    scene,
                    attr,
                    AssetBaseCfg(
                        prim_path=prim,
                        init_state=AssetBaseCfg.InitialStateCfg(
                            pos=tt_frame.net_post_center_env(
                                near_x,
                                surface_z,
                                left=left,
                                post_height=tt_cfg.NET_POST_HEIGHT,
                            )
                        ),
                        spawn=sim_utils.CuboidCfg(
                            size=post_size,
                            rigid_props=robot_proxy_rigid_props(),
                            activate_contact_sensors=False,
                            visible=False,
                            collision_props=sim_utils.CollisionPropertiesCfg(
                                collision_enabled=True
                            ),
                            physics_material=tt_cfg._surface_material(
                                mats.net_restitution,
                                mats.net_static_friction,
                                mats.net_dynamic_friction,
                            ),
                        ),
                    ),
                )
        filter_prims.extend(
            [
                TABLE_ROBOT_KEEPOUT_PRIM,
                TABLE_NET_PRIM,
                TABLE_NET_POST_LEFT_PRIM,
                TABLE_NET_POST_RIGHT_PRIM,
            ]
        )
    else:
        # A late Hydra override may turn the ActionBall-only assembly off after the leaf class's
        # default-on ``__post_init__`` already attached it.  Remove every full-assembly collider
        # here instead of leaving an unscored keep-out/net behind a legacy top-only contract.
        for attr in (
            "table_robot_keepout",
            "table_net",
            "table_net_post_left",
            "table_net_post_right",
        ):
            if getattr(scene, attr, None) is not None:
                setattr(scene, attr, None)
    env_cfg.table_obstacle_prims = tuple(filter_prims)

    truth_visual_present = any(
        getattr(scene, attr, None) is not None
        for attr in ("shadow_table_visual", "pb_table_visual")
    )
    if truth_visual_present and getattr(
        scene, "table_obstacle_visual", None
    ) is not None:
        scene.table_obstacle_visual = None
    if (
        visual
        and not truth_visual_present
        and getattr(scene, "table_obstacle_visual", None) is None
    ):
        scene.table_obstacle_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableObstacleVisual",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tt_frame.table_visual_origin_env(near_x, surface_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(usd_path=tt_cfg._TABLE_USD_PATH),
        )


def attach_table_contact_sensor(env_cfg) -> None:
    """Attach only the legacy top-only wrist pair sensor.

    ActionBall's full assembly deliberately installs *no* filtered ``ContactSensor``.  The pinned
    GPU backend expands five table sources by 32 robot filter expressions, emits invalid
    ``force_matrix_w`` data and makes 4096-environment construction/collection prohibitively
    expensive.  Full assembly instead reads live articulation pose and performs a conservative
    device-side geometric keep-out.
    """

    filter_prims = tuple(getattr(env_cfg, "table_obstacle_prims", ()))
    if not filter_prims:
        raise RuntimeError("table contact sensor cannot be attached without a table obstacle prim")
    full_assembly = bool(getattr(env_cfg, "table_robot_keepout", False))
    previous_sensor_names = tuple(
        getattr(env_cfg, "table_pair_contact_sensor_names", ())
    )
    sensor_names = () if full_assembly else (TABLE_CONTACT_SENSOR_NAME,)
    for stale_sensor_name in set(previous_sensor_names) - set(sensor_names):
        if getattr(env_cfg.scene, stale_sensor_name, None) is not None:
            setattr(env_cfg.scene, stale_sensor_name, None)
    if full_assembly:
        # Retire configs created by an older class-level/default-on pass even when the old run did
        # not publish ``table_pair_contact_sensor_names`` yet.
        for stale_sensor_name in (
            TABLE_CONTACT_SENSOR_NAME,
            *TABLE_FULL_CONTACT_SENSOR_NAMES,
        ):
            if getattr(env_cfg.scene, stale_sensor_name, None) is not None:
                setattr(env_cfg.scene, stale_sensor_name, None)
    else:
        setattr(
            env_cfg.scene,
            TABLE_CONTACT_SENSOR_NAME,
            ContactSensorCfg(
                prim_path=TABLE_CONTACT_SENSOR_PRIM,
                filter_prim_paths_expr=list(filter_prims),
                update_period=0.0,
            ),
        )
    env_cfg.table_pair_contact_sensor_names = tuple(sensor_names)


#: Numerical-zero tolerance (N) for an exact robot-body/table pair.  ActionBall is a no-touch
#: task: a light brush is not a cheaper legal action than a hard strike.  Pair-filter tensors are
#: exactly zero when the pair has no contact, so this tolerance rejects every resolved contact
#: while leaving only a tiny floating-point guard.  The independent teacher admission is stricter
#: still: it requires at least 5 mm continuous swept clearance over the full cycle.
TABLE_HIT_FORCE_THRESHOLD_N = 1.0e-6
#: Inflation (m) of the table box used to accept a contact.  PhysX resolves a contact a fraction
#: of a millimetre outside the surface; 2 cm is one racket-blade thickness of slack and is far
#: below the 5 cm slab, so it cannot reach anything that is not touching the table.
TABLE_HIT_MARGIN_M = 0.02


def table_hit_done_term():
    """The ``robot_hit_table`` termination.  ONE definition, used by every HOPE terminations cfg.

    Legacy top-only mode watches every non-foot body because its geometric discriminator is the
    table slab, not body identity; support-foot floor force remains sanctioned.  ActionBall full
    assembly instead watches the exact 32-body articulation, including both feet, through the
    component geometry: a stance foot remains far from every table AABB, while a foot actually
    entering the floor-to-slab keep-out is correctly terminal.

    Legacy top-only tasks keep the broad-origin plus exact-wrist combination.  ActionBall uses a
    conservative articulation-pose collision-component-OBB/table-AABB keep-out without reading
    contact sensors.  Its conservative broad phase can terminate before resolved physical contact
    and must be reported as a keep-out violation, not contact truth.  The merged wrist/racket body
    also keeps a live blade OBB, so the 21 cm wrist-to-racket offset is not lost.

    ``near_x``/``surface_z`` default to the ``RacketTargetCommandCfg`` defaults; every HOPE
    ``__post_init__`` rewrites them from the live cfg, so a run that moves the virtual table moves
    the collider and this box together.
    """
    return DoneTerm(
        func=mdp.robot_hit_table,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[A3_NON_FOOT_BODY_REGEX]),
            "filtered_sensor_cfg": SceneEntityCfg(TABLE_CONTACT_SENSOR_NAME),
            # ``ManagerTermBase`` resolves every SceneEntityCfg in ``params`` eagerly, including
            # fields ignored by the selected mode.  Keep inactive-mode bindings empty so legacy
            # scenes do not need the five full-table sensors and full scenes do not need the
            # legacy wrist sensor.
            "full_table_filtered_sensor_cfgs": (),
            "expected_full_table_source_prim_paths": (),
            "expected_full_robot_body_names": (),
            "asset_cfg": SceneEntityCfg("robot", body_names=[A3_NON_FOOT_BODY_REGEX]),
            "near_x": 0.5,
            "surface_z": 0.76,
            "force_threshold": TABLE_HIT_FORCE_THRESHOLD_N,
            "margin": TABLE_HIT_MARGIN_M,
            "full_table_assembly": False,
            "keepout_floor_z": 0.0,
            "collision_proxy_artifact_path": (
                TABLE_COLLISION_PROXY_ARTIFACT_PATH
            ),
            "collision_proxy_artifact_sha256": (
                TABLE_COLLISION_PROXY_ARTIFACT_SHA256
            ),
            "racket_body_name": TABLE_CONTACT_RACKET_BODY_NAME,
            "racket_blade_center_offset_wrist_m": (
                TABLE_RACKET_BLADE_CENTER_OFFSET_WRIST_M
            ),
            "racket_blade_half_extents_m": TABLE_RACKET_BLADE_HALF_EXTENTS_M,
            "action_name": "joint_pos",
            "require_substep_latch": False,
        },
    )


def table_hit_rew_term():
    """The ``table_hit_penalty`` reward.  Price of one table strike; ``robot_hit_table`` is the rule.

    Same family and same shape as ``death_penalty`` (``weight * dt`` charged once, on the terminal
    step) but addressed to ONE termination reason, so it can be priced, ablated and read off
    independently of falling.  DEFAULT weight 0.0 = IsaacLab skips the term entirely = the default
    path stays byte-equivalent; ``reward_pack=v2`` sets the real number.
    """
    return RewTerm(
        func=mdp.terminated_by_term, weight=0.0, params={"term_name": "robot_hit_table"}
    )


def apply_table_obstacle(env_cfg) -> None:
    """Wire the table, contact guard, termination and penalty, or take all four away.

    Called from every HOPE ``__post_init__`` AFTER the shadow/physical attachments, so it can see
    a truth-instrument table already in the scene and reuse that collider instead of stacking a
    second identical static box on top of it.

    The four pieces move together on purpose.  A table with no termination is a decoration; a
    termination with no contact guard can never report the full rule; a penalty naming a
    termination that was removed raises at the first step.  So this either installs all four or
    removes all four, and never leaves a half-configured scene.
    """
    T = env_cfg.terminations
    R = getattr(env_cfg, "rewards", None)
    if not getattr(env_cfg, "table_obstacle", False):
        if getattr(T, "robot_hit_table", None) is not None:
            T.robot_hit_table = None
        if R is not None and getattr(R, "table_hit_penalty", None) is not None:
            R.table_hit_penalty = None
        # Take the collider back out too — ``__post_init__`` runs with the DEFAULT (on), so by the
        # time a train.py ``task.table_obstacle=false`` override reaches here the slab is already
        # attached. Leaving it would give a "no-table" control arm a table it is not scored
        # against, which is worse than either honest option. Only the prims THIS function creates
        # are removed: shadow_table / pb_table belong to the metrics instruments and are theirs.
        table_sensor_names = tuple(
            getattr(
                env_cfg,
                "table_pair_contact_sensor_names",
                (TABLE_CONTACT_SENSOR_NAME,),
            )
        )
        for attr in (
            "table_obstacle",
            "table_obstacle_visual",
            "table_robot_keepout",
            "table_net",
            "table_net_post_left",
            "table_net_post_right",
            *table_sensor_names,
        ):
            if getattr(env_cfg.scene, attr, None) is not None:
                setattr(env_cfg.scene, attr, None)
        env_cfg.table_obstacle_prim = ""
        env_cfg.table_obstacle_prims = ()
        env_cfg.table_pair_contact_sensor_names = ()
        action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        if action_cfg is not None:
            action_cfg.table_contact_substep_guard = False
        return

    attach_table_obstacle(env_cfg)
    attach_table_contact_sensor(env_cfg)
    if getattr(T, "robot_hit_table", None) is None:
        T.robot_hit_table = table_hit_done_term()
    if R is not None and getattr(R, "table_hit_penalty", None) is None:
        R.table_hit_penalty = table_hit_rew_term()
    rt = env_cfg.commands.racket_target
    T.robot_hit_table.params["near_x"] = float(rt.vb_table_near_x)
    T.robot_hit_table.params["surface_z"] = float(rt.vb_table_surface_z)
    full_assembly = bool(getattr(env_cfg, "table_robot_keepout", False))
    T.robot_hit_table.params["full_table_assembly"] = full_assembly
    T.robot_hit_table.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names=(
            list(TABLE_CONTACT_BODY_NAMES)
            if full_assembly
            else [A3_NON_FOOT_BODY_REGEX]
        ),
    )
    T.robot_hit_table.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=(
            list(TABLE_CONTACT_BODY_NAMES)
            if full_assembly
            else [A3_NON_FOOT_BODY_REGEX]
        ),
    )
    T.robot_hit_table.params["filtered_sensor_cfg"] = SceneEntityCfg(
        "contact_forces" if full_assembly else TABLE_CONTACT_SENSOR_NAME
    )
    T.robot_hit_table.params["full_table_filtered_sensor_cfgs"] = ()
    T.robot_hit_table.params["expected_full_table_source_prim_paths"] = (
        tuple(env_cfg.table_obstacle_prims) if full_assembly else ()
    )
    T.robot_hit_table.params["expected_full_robot_body_names"] = (
        TABLE_CONTACT_BODY_NAMES if full_assembly else ()
    )
    T.robot_hit_table.params["require_substep_latch"] = full_assembly
    T.robot_hit_table.params["action_name"] = "joint_pos"
    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if action_cfg is not None:
        action_cfg.table_contact_substep_guard = full_assembly
        if full_assembly:
            if int(getattr(env_cfg, "decimation", -1)) != 4:
                raise ValueError(
                    "ActionBall table-contact assembly requires decimation=4"
                )
            action_cfg.table_contact_guard_termination_term = "robot_hit_table"
            action_cfg.table_contact_guard_expected_decimation = 4


##
# Commands: motion (imitation) + racket target.
##


@configclass
class HOPECommandsCfg(CommandsCfg):
    racket_target = mdp.RacketTargetCommandCfg(
        asset_name="robot",
        motion_command_name="motion",
        debug_vis=False,
        # Paddle face normal = racket-local +Y (blade is thin along Y; +Y is the red/hitting face).
        # Confirmed from the std-pingpang URDF + blade STL in reimplement.md Step 11 (the cfg default
        # of axis 2/+Z was a placeholder guess). sign=+1 -> red (forehand) face; use -1 for the
        # black face if you train a backhand-only policy.
        # NOTE: cfg/task/HOPEPingPong.yaml also sets mount_normal_axis and (via train.py) overrides
        # this for the Hydra path — keep the two in sync.
        mount_normal_axis=1,
        mount_normal_sign=1.0,
    )


##
# Observations: HITTER actor (desired targets only) + privileged critic (actual racket state).
##


@configclass
class HOPEObservationsCfg(ObservationsCfg):
    @configclass
    class HOPEPolicyCfg(ObservationsCfg.PolicyCfg):
        # Deployment alignment with HITTER (arXiv:2508.21043, Table — actor obs): world-frame base LINEAR
        # velocity is a CRITIC-ONLY (privileged) observation there, because a humanoid's floating-base
        # linear velocity is not cleanly measurable on hardware (it needs a fragile IMU+leg-odometry state
        # estimator). The BeyondMimic base PolicyCfg feeds it to the actor; remove it here so the actor
        # never depends on a quantity it cannot reliably get at deploy. base_ang_vel / projected_gravity
        # (both from the IMU) and joint pos/vel stay. The critic (HOPECriticCfg) keeps base_lin_vel.
        base_lin_vel = None
        # Appended after the BeyondMimic proprioceptive + motion terms.
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_target_pos_b = ObsTerm(func=mdp.base_target_pos_b, params={"command_name": "racket_target"})
        racket_target_pos_b = ObsTerm(
            func=mdp.racket_target_pos_b,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        # HITTER (arXiv:2508.21043, Table I): the racket NORMAL/orientation is NOT an actor observation —
        # it is a reward target only. The actor sees only desired racket pos (rel base) + desired racket
        # vel (world) + time-to-strike + desired base pos (rel base). The critic keeps the normal (below).
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        # Unified HITTER policy (forehand+backhand in one policy): the actor must know which swing it is
        # doing (forehand +1 / backhand -1), since the swing type selects the imitated clip and the target
        # region. (For a single-swing-type policy this is constant and can be removed.)
        swing_type = ObsTerm(func=mdp.swing_type, params={"command_name": "racket_target"})

    @configclass
    class HOPECriticCfg(ObservationsCfg.PrivilegedCfg):
        base_target_pos_b = ObsTerm(func=mdp.base_target_pos_b, params={"command_name": "racket_target"})
        racket_target_pos_b = ObsTerm(func=mdp.racket_target_pos_b, params={"command_name": "racket_target"})
        # A1: the CRITIC keeps the TRUE live target velocity even when the actor's view is
        # delayed/jittered (task.racket.target_delay_steps / target_jitter_*): the asymmetric critic
        # is privileged/sim-side. Identical value to mdp.racket_target_vel_w when the A1 knobs are off.
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w_live, params={"command_name": "racket_target"})
        racket_target_normal_w = ObsTerm(func=mdp.racket_target_normal_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})
        # actual racket state (FK) — privileged, never available on hardware
        racket_pos_b = ObsTerm(func=mdp.racket_pos_b, params={"command_name": "racket_target"})
        racket_lin_vel_w = ObsTerm(func=mdp.racket_lin_vel_w, params={"command_name": "racket_target"})
        racket_normal_w = ObsTerm(func=mdp.racket_normal_w, params={"command_name": "racket_target"})
        episode_time_left = ObsTerm(func=mdp.episode_time_left)

    policy: HOPEPolicyCfg = HOPEPolicyCfg()
    critic: HOPECriticCfg = HOPECriticCfg()


##
# Rewards: imitation (inherited) + goal (racket/base) + regularization.
# Weights are HOPE tuning choices (HITTER does not publish reward weights/kernels).
##


@configclass
class HOPERewardsCfg(RewardsCfg):
    # r_goal — racket state tracking, active only in the ±strike_window around the strike.
    # std values are set to the step-14 acceptance tolerances so reward ≈ exp(-1) at the threshold;
    # tune from here (reimplement.md §13.7 item 7). HITTER does not publish reward weights/kernels.
    racket_position = RewTerm(
        func=mdp.racket_position_tracking_exp,
        weight=4.0,
        params={"command_name": "racket_target", "std": 0.075},  # target < 7.5 cm
    )
    # Broad strike-window ranking signal.  It is declared but disabled in every historical task;
    # only the vendor ActionBall leaf opts in after the 2026-07-31 probe observed 97% of entry
    # misses beyond 20 cm.  The original 7.5 cm precision channel above remains unchanged.
    racket_position_coarse = RewTerm(
        func=mdp.racket_position_coarse_tracking_exp,
        weight=0.0,
        params={"command_name": "racket_target", "std": 0.30},
    )
    racket_velocity = RewTerm(
        func=mdp.racket_velocity_tracking_exp,
        weight=2.0,
        params={"command_name": "racket_target", "std": 0.5},  # target < 0.5 m/s
    )
    racket_normal = RewTerm(
        func=mdp.racket_normal_tracking_exp,
        weight=2.0,
        params={"command_name": "racket_target", "std": 0.262},  # radians, target < 15 deg
    )
    # r_goal — base repositioning, active only before the strike.
    base_position = RewTerm(
        func=mdp.base_position_tracking_exp,
        weight=1.0,
        params={"command_name": "racket_target", "std": 0.3},
    )
    # r_regularization — pre-strike foot-slip penalty (stability). Penalizes horizontal foot speed while
    # the foot is in contact, gated by pre_strike ONLY (the strike swing is untouched). Default weight is
    # overridden by cfg/task/HOPEPingPong.yaml `pre_strike_foot_slip_weight`.
    pre_strike_foot_slip = RewTerm(
        func=mdp.pre_strike_foot_slip,
        weight=-0.2,
        params={"command_name": "racket_target"},
    )
    # r_regularization — energy / torque smoothness (action_rate_l2 already inherited).
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # --- reward_staged_design 2026-07-08 flag-off terms (weight 0.0 = SKIPPED by the ---------------
    # RewardManager = byte-identical baseline; enabled per-arm from the task YAML / CLI) ------------
    # Constant guidance penalty (§② B2): -w * min(||racket_FK - target||, d_max) every
    # pre-strike + in-window step — a dense "which way to swing" gradient that survives outside
    # every exp kernel's responsive band. 人话:挥不到球也天天有"往哪挥"的工资单,小而恒。
    # Enable via rewards.racket_guidance_weight (NEGATIVE; design铁律: per-second cost <= 10-20%
    # of the measured imitation income — verify against the launch-log income accounting).
    racket_guidance = RewTerm(
        func=mdp.racket_guidance, weight=0.0,
        params={"command_name": "racket_target", "d_max": 0.5})
    # Face-angle guidance penalty (2026-07-10, M3c 死区解药): -w * min(angle(achieved_normal,
    # demanded_normal), theta_max) every pre-strike + in-window step — the face-channel twin of
    # racket_guidance. exp 拍面核在 ~3·std 外零梯度(翻面修复后 swing 33°/v5syn 反手 ~53° 全在
    # 死区),这条线性罚把反面的拍子一路拉回来。Enable via rewards.racket_face_guidance_weight
    # (NEGATIVE; 比值铁律同 racket_guidance: per-second cost <= 10-20% imitation income).
    racket_face_guidance = RewTerm(
        func=mdp.racket_face_guidance, weight=0.0,
        params={"command_name": "racket_target", "theta_max": 1.5707963})
    # Conditional, fixed-budget face guidance (2026-07-14; default OFF).  Unlike the historical
    # pre-strike-wide linear angle tax, this term spends a fixed cost only in the wide strike window:
    # outside readiness it is constant (so there is no face gradient or escape reward), then readiness
    # converts that cost into the signed-face error fraction.  Its function returns [0,1], so |weight|
    # is an auditable maximum per-window-step penalty.  Thresholds are
    # frozen to existing task contracts: position 7.5cm full / 9.5cm zero; velocity 0.5m/s full /
    # 1.0m/s zero; no penalty inside the 15-degree face tolerance.  Enable only through the
    # single causal-axis flag rewards.racket_face_conditional_guidance_weight (must be <= 0).
    racket_face_conditional_guidance = RewTerm(
        func=mdp.racket_face_conditional_guidance,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "theta_free": 0.262,
            "theta_max": 3.141592653589793,
            "pos_full": 0.075,
            "pos_zero": 0.095,
            "vel_full": 0.5,
            "vel_zero": 1.0,
        },
    )
    # R-b envelope-as-penalty (§⑥): per-step indicator of the tracking-envelope violation that
    # normally TERMINATES (anchor_pos | ee_body_pos, both z>0.25 m vs the reference — identical
    # expressions/threshold/body list as TerminationsCfg after the A3 __post_init__ re-pin).
    # 人话:跟丢参考不再判死,改成站在违规区里每秒扣钱。Enabled ONLY by train.py's
    # terminations.envelope_as_penalty override, which also REMOVES the two terminations and
    # switches on the tracking_loss accounting; the weight alone is never set by hand.
    tracking_envelope = RewTerm(
        func=mdp.tracking_envelope_violation, weight=0.0,
        params={"command_name": "motion", "threshold": 0.25,
                "body_names": A3_FEET_BODIES + A3_HAND_BODIES})


##
# Domain randomization (local sim-to-real reconstruction).
#
# HITTER publishes no DR recipe. Its controller gains are reported as heuristic fixed values, but
# mass/friction/push/observation-noise randomization is not specified there. The terms below are local
# sim-to-real choices informed by external implementations and the latest Agibot A3 training setting.
#
# Already provided by the base EventCfg: friction (physics_material, startup), CoM (startup),
# joint default pos (startup), external pushes (push_robot, interval). Observation noise comes from the
# per-term Unoise + enable_corruption on the policy observation group.
##


@configclass
class HOPEEventCfg(EventCfg):
    # Historical local default: no external push. HITTER publishes no push/DR prescription, so this
    # must not be described as paper alignment. Keep friction (physics_material) and CoM (base_com)
    # from the base EventCfg; disable the base interval push until an explicit recipe enables it.
    push_robot = None
    # F-axis interval FORCE push pair (default OFF = both None, byte-identical; see
    # HOPEForcePushCfg). Two terms on purpose: force_push fires the horizontal constant force,
    # force_push_sweep clears expired forces every control step — Isaac interval events are NOT
    # called per-step, so expiry needs its own high-frequency term or the force never clears.
    force_push = None
    force_push_sweep = None
    # 合并互斥推事件对(default OFF = both None, byte-identical;见 HOPEPushRobotCfg.
    # combined_exclusive)。单事件按 force_prob 抽签二选一(速度踢 / 持续力推),防两种随机推
    # 同帧叠加;力分支写同一本到期账本,所以合并模式同样需要每控制步的清扫兜底事件。
    combined_push = None
    combined_push_sweep = None

    # Local link-mass randomization (±15%); not a claimed HITTER setting.
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # Latest Agibot A3 Parkour training authority: startup-only draw, Kp log_uniform (0.8,1.2),
    # Kd (0.7,1.3). This is a local sim-to-real robustness choice. Set the event to None to disable.
    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


##
# Wave-P random base push (PACE/BeyondMimic-style shove; historical default OFF).
##


@configclass
class HOPEPushRobotCfg:
    """Wave-P 随机推撞开关组(默认全关 = 历史本地配方,``events.push_robot=None`` 逐位不变)。

    ``recipe='legacy_v1'`` 保留历史 vxy/等幅角速度踢。
    ``recipe='axis_box_6d_v2'`` 则直接绑定完整 x/y/z/roll/pitch/yaw 对称速度箱;
    两种拼写不得混用。A3 vendor ActionBall 叶子用同底盘建议幅值,但沿用
    PACE/Wave-P 已验证的 5–15 s 任务特异 cadence，不把 parkour 1–3 s 直接搬进击球窗。

    两条启用路径共用 ``training_contract`` 的 v1/v2 纯装配函数
    (fail-loud,单一来源):(i) cfg 直启 —— ``__post_init__`` 末尾消费本旗标组
    (``apply_push_robot_event``);(ii) YAML/CLI —— train.py 的 ``task.push.*`` 覆盖在
    ``__post_init__`` 之后运行(face_command_obs 时序),自己构造同款 EventTerm。
    """

    enable: bool = False
    recipe: str = "legacy_v1"
    interval_range_s: tuple[float, float] = (5.0, 15.0)
    vel_xy_mps: float = 0.0
    ang_vel_radps: float = 0.0
    ang_axes: str = "none"  # "none" | "yaw" | "rpy"
    velocity_range: dict = None
    # --- 合并互斥模式(Franco 2026-07-25:两种随机推合并抽签,防同帧叠加)------------------
    # True 时:速度推(本组幅度)与力推(force_push 组的 force_n/duration_s)合并成【一个】
    # interval 事件,每次触发按 force_prob 逐 env 抽签二选一,同一次触发绝不两种都来。两组
    # enable 都必须 =True(分支配方上膛),但两个 legacy 独立事件一个都不许再挂——合并 +
    # legacy 独立事件同时在场就是回到"同帧叠加",apply_combined_push_event fail-loud。默认
    # False = 两个 legacy 独立事件行为逐字节不变(各自独立时钟,可能同帧叠加——这正是合并
    # 模式要消掉的历史行为)。
    combined_exclusive: bool = False
    # 抽签出"力推"分支的概率(严格 0<p<1;0/1 等价单类型推,请直接用 legacy 单事件)。
    # combined_exclusive=False 时必须停在默认 0.5(关着的开关不许挂上膛参数)。
    force_prob: float = 0.5


def apply_push_robot_event(env_cfg) -> None:
    """Consume the ``push`` flag group: build ``events.push_robot`` when enabled.

    人话:把开关组翻译成真正的 interval push 事件;没开就什么都不动(所有在跑矩阵格 =
    push_robot None,行为逐位不变)。Idempotent — train.py 的 task.push 覆盖路径和 cfg
    直启路径可以都跑一遍,结果相同;矛盾的配方(enable=true 但幅度全零、轴组合不一致等)
    在 ``push_robot_event_block`` 里 fail-loud。
    """

    push = getattr(env_cfg, "push", None)
    if push is None or not push.enable:
        return
    if getattr(push, "combined_exclusive", False):
        # 合并互斥模式:本组幅度归合并事件抽签分发,这里绝不再挂独立 push_robot 事件(否则
        # 就是"合并 + legacy 独立事件同帧叠加"的回归)。合并事件必须已由
        # apply_combined_push_event 装好(__post_init__ 先跑它);没装好 = 半配置,直接炸。
        if getattr(env_cfg.events, "combined_push", None) is None:
            raise ValueError(
                "push.combined_exclusive=true but events.combined_push is not wired — "
                "apply_combined_push_event must run first; the legacy independent "
                "push_robot event is never built in combined mode"
            )
        return
    recipe = str(getattr(push, "recipe", "legacy_v1"))
    if recipe == "legacy_v1":
        if getattr(push, "velocity_range", None) is not None:
            raise ValueError(
                "push legacy_v1 cannot carry velocity_range; v1/v2 spellings may not mix"
            )
        from whole_body_tracking.utils.training_contract import push_robot_event_block

        block = push_robot_event_block(
            enable=True,
            interval_range_s=tuple(push.interval_range_s),
            vel_xy_mps=float(push.vel_xy_mps),
            ang_vel_radps=float(push.ang_vel_radps),
            ang_axes=str(push.ang_axes),
        )
    elif recipe == "axis_box_6d_v2":
        if (
            float(push.vel_xy_mps) != 0.0
            or float(push.ang_vel_radps) != 0.0
            or str(push.ang_axes) != "none"
        ):
            raise ValueError(
                "push axis_box_6d_v2 cannot carry legacy vxy/angular fields"
            )
        from whole_body_tracking.utils.training_contract import (
            push_robot_axis_box_event_block,
        )

        block = push_robot_axis_box_event_block(
            enable=True,
            interval_range_s=tuple(push.interval_range_s),
            velocity_range=push.velocity_range,
        )
    else:
        raise ValueError(f"unsupported push recipe {recipe!r}")
    env_cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        ),
        params={
            "velocity_range": {
                axis: (float(rng[0]), float(rng[1]))
                for axis, rng in block["velocity_range"].items()
            }
        },
    )


##
# F-axis interval FORCE push (matched-impulse companion of the Wave-P velocity push; default OFF).
##


@configclass
class HOPEForcePushCfg:
    """F 轴持续力推开关组(默认全关 = ``events.force_push``/``force_push_sweep`` 双 None,逐字节 no-op)。

    人话:开了以后,每隔 ``interval_range_s`` 秒(uniform 抽样)朝水平随机方向,对 pelvis_link
    施加幅度 ``force_n`` 牛的恒力,持续 ``duration_s`` 秒(默认 0.30 s = 15 个控制步 @ 50 Hz)
    后清零。与速度推档位(p02/p035/p05/p08)按同冲量对表:Δv_equiv = force_n × duration_s /
    m_robot;机器人总质量在运行时读 articulation 写进训练合同,臂的 ``force_n`` 由主控按真实
    质量在渲染前算好写死(配置面只有 force_n 与 duration_s,不搞自动换算魔法)。施力点 =
    pelvis link 原点,合同 application_point 诚实写 "pelvis_link_origin",不许标 COM
    (Yikang V9 教训)。

    两条启用路径共用 ``training_contract.force_push_event_block`` 的同一套校验/装配
    (fail-loud,单一来源):(i) cfg 直启 —— ``__post_init__`` 末尾消费本旗标组
    (``apply_force_push_event``);(ii) YAML/CLI —— train.py 的 ``task.force_push.*`` 覆盖在
    ``__post_init__`` 之后运行(face_command_obs 时序),自己构造同款事件对(施力 + 清扫)。
    """

    enable: bool = False
    interval_range_s: tuple[float, float] = (5.0, 15.0)
    force_n: float = 0.0
    duration_s: float = 0.30


def apply_force_push_event(env_cfg) -> None:
    """Consume the ``force_push`` flag group: build the force event + expiry sweeper when enabled.

    人话:把开关组翻译成两个事件——interval 触发的施力事件(``push_by_applying_wrench``)+
    每个控制步跑一次的到期清扫事件(``sweep_expired_force_pushes``)。Isaac 的 interval 事件
    不逐步调用,清零必须有高频兜底,否则恒力永远挂着。没开就什么都不动(两个事件都保持
    None,行为逐字节不变)。Idempotent — train.py 的 task.force_push 覆盖路径和 cfg 直启路径
    可以都跑一遍,结果相同;矛盾配方(enable=true 但 force_n=0、duration 不是整数个控制步等)
    在 ``force_push_event_block`` 里 fail-loud。
    """

    force_push = getattr(env_cfg, "force_push", None)
    if force_push is None or not force_push.enable:
        return
    push = getattr(env_cfg, "push", None)
    if push is not None and getattr(push, "combined_exclusive", False):
        # 合并互斥模式:力分支归合并事件抽签分发,这里绝不再挂独立 force_push 事件对(理由
        # 同 apply_push_robot_event 的合并让路守卫;清扫兜底由 combined_push_sweep 承担)。
        if getattr(env_cfg.events, "combined_push", None) is None:
            raise ValueError(
                "push.combined_exclusive=true but events.combined_push is not wired — "
                "apply_combined_push_event must run first; the legacy independent "
                "force_push event pair is never built in combined mode"
            )
        return
    from whole_body_tracking.utils.training_contract import force_push_event_block

    control_dt_s = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    block = force_push_event_block(
        enable=True,
        interval_range_s=tuple(force_push.interval_range_s),
        force_n=float(force_push.force_n),
        duration_s=float(force_push.duration_s),
        control_dt_s=control_dt_s,
    )
    env_cfg.events.force_push = EventTerm(
        func=mdp.push_by_applying_wrench,
        mode="interval",
        interval_range_s=(
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        ),
        params={
            "force_n": float(block["force_n"]),
            "duration_steps": int(block["duration_steps"]),
            "body_name": str(block["body_name"]),
        },
    )
    env_cfg.events.force_push_sweep = EventTerm(
        func=mdp.sweep_expired_force_pushes,
        mode="interval",
        interval_range_s=(control_dt_s, control_dt_s),
        params={},
    )


##
# Merged EXCLUSIVE push (合并互斥推; default OFF — Franco 2026-07-25 两种随机推合并抽签).
##


def apply_combined_push_event(env_cfg) -> None:
    """Consume the merged-exclusive spelling: build ONE sampling push event instead of the pair.

    人话:``push.combined_exclusive=True`` 时,把速度推(push 组幅度)与力推(force_push 组
    幅度)装配成【一个】interval 事件 ``events.combined_push``(每次触发按 force_prob 逐 env
    抽签二选一,防两种随机推同帧叠加)+ 每控制步的到期清扫兜底 ``events.combined_push_sweep``。
    两组 enable 都必须 =True(分支配方上膛),两个 legacy 独立事件必须全 None(合并 + 独立
    事件同时在场 = 同帧叠加回归,fail-loud);两组 interval_range_s 必须逐字相同(合并事件
    只有一个触发时钟,不许两种拼写)。默认 combined_exclusive=False 时本函数是严格 no-op
    (但 force_prob 偏离默认 0.5 会炸——关着的开关不许挂上膛参数),两个 legacy 独立事件
    行为逐字节不变。参数校验/装配走 ``training_contract.push_robot_event_block`` 的合并分支
    (与 schema-3 合同校验同一单一来源)。⚠ train.py 尚无 task.combined_push 覆盖面:经
    train.py 发射合并模式会在硬合同装配处因"半接线"fail-loud,属预期(后续波次接线)。
    """

    push = getattr(env_cfg, "push", None)
    if push is None:
        return
    force_prob = getattr(push, "force_prob", 0.5)
    if not getattr(push, "combined_exclusive", False):
        # 关着的开关不许挂上膛参数:combined 关着时 force_prob 必须停在默认 0.5。
        if float(force_prob) != 0.5:
            raise ValueError(
                "push.combined_exclusive=false may not carry a loaded force_prob "
                f"({force_prob!r}) — delete it or set combined_exclusive=true"
            )
        return
    if str(getattr(push, "recipe", "legacy_v1")) != "legacy_v1":
        raise ValueError(
            "push axis_box_6d_v2 cannot use combined_exclusive; the v2 recipe is "
            "one six-axis velocity event"
        )
    force_push = getattr(env_cfg, "force_push", None)
    if (
        force_push is None
        or not bool(getattr(push, "enable", False))
        or not bool(getattr(force_push, "enable", False))
    ):
        raise ValueError(
            "push.combined_exclusive=true requires BOTH branch recipes armed: "
            "push.enable=true AND force_push.enable=true (合并事件抽签二选一,缺一个"
            "分支就没得抽;只想要单类型推请关掉 combined_exclusive 用 legacy 单事件)"
        )
    events = getattr(env_cfg, "events", None)
    if (
        events is None
        or not hasattr(events, "combined_push")
        or not hasattr(events, "combined_push_sweep")
    ):
        raise ValueError(
            "combined push requires the events cfg to DECLARE the combined_push/"
            "combined_push_sweep slots (HOPEEventCfg does; an events cfg without the "
            "declared attributes would silently hide the event from the EventManager)"
        )
    legacy_live = [
        name
        for name in ("push_robot", "force_push", "force_push_sweep")
        if getattr(events, name, None) is not None
    ]
    if legacy_live:
        raise ValueError(
            "push.combined_exclusive=true forbids the legacy independent push events "
            f"{legacy_live} — 合并模式就是为了防同帧叠加,独立事件必须保持 None"
        )
    push_interval = tuple(float(v) for v in push.interval_range_s)
    force_interval = tuple(float(v) for v in force_push.interval_range_s)
    if push_interval != force_interval:
        raise ValueError(
            "combined push has exactly ONE trigger clock: push.interval_range_s and "
            "force_push.interval_range_s must be spelled identically, got "
            f"{push_interval!r} vs {force_interval!r}"
        )
    from whole_body_tracking.utils.training_contract import push_robot_event_block

    control_dt_s = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    block = push_robot_event_block(
        enable=True,
        interval_range_s=push_interval,
        vel_xy_mps=float(push.vel_xy_mps),
        ang_vel_radps=float(push.ang_vel_radps),
        ang_axes=str(push.ang_axes),
        combined_exclusive=True,
        force_prob=float(force_prob),
        force_n=float(force_push.force_n),
        duration_s=float(force_push.duration_s),
        control_dt_s=control_dt_s,
    )
    events.combined_push = EventTerm(
        func=mdp.push_combined_exclusive,
        mode="interval",
        interval_range_s=(
            float(block["interval_range_s"][0]),
            float(block["interval_range_s"][1]),
        ),
        params={
            "velocity_range": {
                axis: (float(rng[0]), float(rng[1]))
                for axis, rng in block["velocity_range"].items()
            },
            "force_n": float(block["force_n"]),
            "duration_steps": int(block["duration_steps"]),
            "force_prob": float(block["force_prob"]),
            "body_name": str(block["body_name"]),
        },
    )
    events.combined_push_sweep = EventTerm(
        func=mdp.sweep_expired_force_pushes,
        mode="interval",
        interval_range_s=(control_dt_s, control_dt_s),
        params={},
    )


##
# Environment configuration.
##


##
# deploy-parity variant — deploy-honest observation (no fabricated base pose).
#
# WHY: the `full` actor obs above depends on the robot's true world base pose through three terms
# (motion_anchor_pos_b, base_target_pos_b, racket_target_pos_b). The mocap streams the base pose at
# 300 Hz during play, but that link is not bridged into the deploy front-end, so those terms are
# fabricated at deploy (anchor_pos_b := 0, base_pos := nominal) -> the deployed policy
# sees a DIFFERENT observation distribution than training and the legs cannot balance. Making the
# actor base-position-free is a deliberate robustness choice (no mocap/VRPN dependency). AGI's reference
# policy transfers because its observation is real-sensor-only (IMU orientation + proprioception, no
# world base position). This variant copies that recipe for the HOPE actor. The privileged CRITIC
# group is unchanged (it may use base pose in sim — it is never deployed). The `full` cfgs above are
# untouched (kept for comparison / the old path).
##


@configclass
class HOPEObservationsDeployParityCfg(HOPEObservationsCfg):
    """Actor obs with every world-frame BASE-POSITION dependency removed (180 -> 175):

    * REMOVED  ``motion_anchor_pos_b`` (3)  — reference torso *position* error needs the world base pose.
    * REMOVED  ``base_target_pos_b``   (2)  — base-repositioning target needs the world base pose.
    * REFRAMED ``racket_target_pos_b`` (3)  — now ``target - current_racket`` (FK), base pose cancels.
    * KEPT     ``motion_anchor_ori_b`` (6, orientation-only / IMU), command, base_ang_vel, joint pos/vel,
               last action, projected_gravity, racket_target_vel_w, time_to_strike, swing_type.

    Every kept/reframed term is computable on hardware from IMU + joint encoders + the planner target.
    """

    @configclass
    class HOPEPolicyDeployParityCfg(HOPEObservationsCfg.HOPEPolicyCfg):
        # --- remove base-position-dependent terms (fabricated on hardware) ---
        motion_anchor_pos_b = None  # inherited from ObservationsCfg.PolicyCfg; needs world base position
        base_target_pos_b = None  # base-repositioning target; needs world base position
        # --- reframe racket target to be relative to the current racket (FK); no world base position ---
        racket_target_pos_b = ObsTerm(
            func=mdp.racket_target_pos_rel_b,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

    @configclass
    class HOPECriticDeployParityCfg(HOPEObservationsCfg.HOPECriticCfg):
        # Vestigial in the base-free deploy-parity task: the base target is never consumed by any reward
        # or actor obs and (base_couple_blend=0) is pure spawn+jitter noise — conditioning the value
        # function on it only adds variance. Removing it changes the CRITIC input dim (2026-07-03), so
        # every pre-change checkpoint fails a FULL strict load — train.py resume stays a loud error on
        # purpose; play.py (export) and eval_deterministic.py fall back to an actor-only tolerant load
        # (utils/ckpt_compat.py). The exported ACTOR / 175-D contract is untouched.
        base_target_pos_b = None

    policy: HOPEPolicyDeployParityCfg = HOPEPolicyDeployParityCfg()
    critic: HOPECriticDeployParityCfg = HOPECriticDeployParityCfg()


@configclass
class HOPEDeployParityRewardsCfg(HOPERewardsCfg):
    """FOOTWORK-TO-STRIKE reward — BASE-FREE. No base-position / base-target / base-arrival reward: the
    legs move because reducing the racket->target distance (``racket_progress``) takes whole-body motion.
    The feet are FREE to step/shift — only BAD foot behaviour is penalized (slip / drag / violent / unstable
    at the strike), never "both feet planted". Lower-body imitation is DROPPED (legs free to reach varied
    targets); upper-body + racket imitation is kept for swing style. All weights are STARTING POINTS — the
    footwork weights live here (not the task YAML), so tune them in this class. (Obs is the base-free
    deploy-parity layout from HOPEObservationsDeployParityCfg.)"""

    # --- BASE-FREE corrections: remove every base-position-dependent reward ---
    base_position = None  # inherited HITTER base-repositioning reward -> REMOVED (it needs a base target)
    motion_global_anchor_pos = None  # reference base-POSITION tracking -> REMOVED (it pins the base)

    # --- racket task: keep the additive pos/vel/normal (inherited, wide gradient) + a MULTIPLICATIVE
    #     success bonus that fires only when pos AND vel AND normal are all good at once (tight acceptance). ---
    racket_strike_success = RewTerm(
        func=mdp.racket_strike_success, weight=5.0,
        params={"command_name": "racket_target", "std_pos": 0.075, "std_vel": 0.5, "std_normal": 0.262},
    )
    # --- the BASE-FREE MOVEMENT DRIVER: dense pre-strike reward for closing the racket->target distance.
    #     Telescopes to weight * (distance reduced over the approach) -> the whole body moves to the target. ---
    racket_progress = RewTerm(func=mdp.racket_progress, weight=10.0, params={"command_name": "racket_target"})

    # --- upper-body-only imitation (legs DECOUPLED so footwork is free to adapt to the target) ---
    # swing-only since 2026-07-05: during hold the body refs (frozen crouch frame) fought
    # the stand joint reference -> splayed-feet crouch-stand; see hope_rewards wrappers.
    # Foot discipline (2026-07-05): hip yaw/roll + ankle roll held to the reference
    # footwork (hold-aware). Penalty; jiayi's tuned value is -0.3 (his own note: tune in
    # [-0.5,-0.1] if it taxes the lunge). Weight 0.0 HERE (merge-audit 2026-07-06 flag-off
    # default — un-ablated tunable); the jiayi lineages pin -0.3 in their task YAMLs
    # (`rewards: foot_orientation_weight`), other lineages adopt via an A/B arm.
    foot_orientation = RewTerm(func=mdp.foot_orientation_discipline, weight=0.0,
        params={"command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint"])})
    motion_body_pos = RewTerm(func=mdp.motion_body_pos_swing_only, weight=1.0,
        params={"command_name": "motion", "std": 0.3, "body_names": A3_UPPER_TRACKED})
    motion_body_ori = RewTerm(func=mdp.motion_body_ori_swing_only, weight=1.0,
        params={"command_name": "motion", "std": 0.4, "body_names": A3_UPPER_TRACKED})
    motion_body_lin_vel = RewTerm(func=mdp.motion_global_body_linear_velocity_error_exp, weight=1.0,
        params={"command_name": "motion", "std": 1.0, "body_names": A3_UPPER_TRACKED})
    motion_body_ang_vel = RewTerm(func=mdp.motion_global_body_angular_velocity_error_exp, weight=1.0,
        params={"command_name": "motion", "std": 3.14, "body_names": A3_UPPER_TRACKED})

    # --- footwork PENALTIES (the feet may step; punish only bad behaviour, NEVER reward "always planted") ---
    foot_slip_sq = RewTerm(func=mdp.foot_slip_sq, weight=-1.0, params={"command_name": "racket_target"})
    foot_velocity = RewTerm(func=mdp.foot_velocity, weight=-0.05, params={"command_name": "racket_target"})
    foot_drag = RewTerm(func=mdp.foot_drag, weight=-0.5, params={"command_name": "racket_target"})
    arm_overreach = RewTerm(func=mdp.arm_overreach, weight=-0.5, params={"command_name": "racket_target"})
    # Anti twist-instead-of-step (pre-strike): penalize |waist_yaw|+|waist_roll| deviation from neutral so
    # the policy cannot face a lateral target by twisting the torso with planted feet — it must STEP.
    # Weight is CLI-tunable via task.rewards.prestrike_waist_twist_weight. Raise if the torso still twists
    # (waist_twist_prestrike stays high / legs stay frozen); lower if it flattens the swing.
    prestrike_waist_twist = RewTerm(
        func=mdp.prestrike_waist_twist, weight=-1.0, params={"command_name": "racket_target"})

    # --- between-swing recovery: POSITIVE ready-stance reward during the pre-swing HOLD --------------
    # (2026-07-03 audit alignment) HITTER's recovery signal is positive-and-causal ("prepare for the next
    # target"), not a pile of penalties. During the hold the imitation reward already pulls the UPPER body
    # to the windup pose, but the legs/base had zero positive signal. hold_ready = exp(-(|v|^2+|w|^2)/std^2)
    # * feet_contact_frac, gated to motion.in_hold AND to target-within-reach (racket_target_distance <
    # reach): near targets -> stand ready pays; far targets -> the term is SILENT so it never out-earns
    # racket_progress for stepping (without the reach gate, planted stillness beats stepping ~1.5/step and
    # teaches freeze-then-rush). The swing itself is untouched (zero outside the hold). CLI-tunable via
    # task.rewards.hold_ready_weight / hold_ready_std / hold_ready_reach.
    hold_ready = RewTerm(
        func=mdp.hold_ready, weight=2.0,
        params={"command_name": "racket_target", "std": 0.5, "reach": 0.65})

    # --- P2.4 PACE-style smooth deceleration (G08, flag-gated, DEFAULT OFF) --------------------------
    # Pseudo base-velocity command proportional to the remaining PLANAR racket->target error:
    # v_des = clamp(v_gain*dist_xy, 0, v_max); reward = exp(-(|v_base_xy| - v_des)^2/std^2), gated to
    # pre_strike. Far target -> pays for moving at v_max (cooperates with racket_progress); at arrival
    # v_des -> 0 -> pays for a CALM base, killing the reactive rush-then-slam toward far targets.
    # REWARD-side only — the frozen 175-D actor obs contract is untouched.  The instrumentation probe
    # is also default OFF; train.py gives it weight 1.0 in BOTH explicit control/treatment arms.  Its
    # function always returns zero, so it runs in the same RewardManager phase without changing total
    # reward.  base_decel weight 0.0 = OFF (IsaacLab skips zero-weight terms); enable per-experiment via
    # task.rewards.base_decel_weight. CLI/yaml-tunable: base_decel_weight / _v_gain / _v_max / _std.
    # Watch metric: base_speed_xy_prestrike (should taper near targets instead of staying hot).
    base_decel_activation_probe = RewTerm(
        func=mdp.base_decel_activation_probe, weight=0.0,
        params={"command_name": "racket_target", "v_gain": 2.0, "v_max": 1.6, "std": 0.4})
    base_decel = RewTerm(
        func=mdp.base_decel_tracking, weight=0.0,
        params={"command_name": "racket_target", "v_gain": 2.0, "v_max": 1.6, "std": 0.4})

    # --- strike-window stability: be planted + upright + still AT the hit (gated to the strike window) ---
    strike_upright = RewTerm(func=mdp.strike_proj_grav_xy, weight=-2.0, params={"command_name": "racket_target"})
    strike_ang_vel = RewTerm(func=mdp.strike_base_ang_vel, weight=-0.5, params={"command_name": "racket_target"})
    strike_foot_vel = RewTerm(func=mdp.strike_foot_velocity, weight=-0.5, params={"command_name": "racket_target"})
    strike_vbob = RewTerm(func=mdp.strike_vertical_bob, weight=-1.0, params={"command_name": "racket_target"})

    # --- v2 蓝图替换候选(reward_redesign_20260725 §1.4/§3;DEFAULT weight=0.0 = IsaacLab 直接
    #     跳过 = 默认路径逐字节等价,消融臂经 task.rewards.* 起零)。替代关系不是叠加关系:
    #     臂里给这两条非零权重时,应同时把被替代的旧项关掉——
    #     * upright_exp(mjlab 收入型站正,权重用【正数】):站得越正每步发钱越多,有界 (0,1],
    #       顺带兼任 alive bonus;替代税型 upright 罚(flat_orientation_l2)。
    #     * hit_unstable_support(PACE 单条击球稳定,权重用【负数】):击球窗内单脚/无支撑记 1,
    #       只管"支撑够不够"这一个事实;替代上面的 strike 四件套
    #       (strike_upright/strike_ang_vel/strike_foot_vel/strike_vbob)。
    upright_exp = RewTerm(
        func=mdp.upright_exp,
        weight=0.0,
        params={"std": math.sqrt(0.2)},
    )
    hit_unstable_support = RewTerm(
        func=mdp.hit_unstable_support,
        weight=0.0,
        params={
            # 与 foot_soft_landing/foot_clearance 同款两脚 sensor 名单
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(A3_FEET_BODIES)),
            "command_name": "racket_target",
        },
    )

    # --- SIM2REAL FINE-TUNE (2026-07-02): explicit clipped-PD only. -----------------------------------
    # ``computed_torque`` is a pre-clip demand only for an EXPLICIT actuator model.  A3 currently uses
    # ImplicitActuatorCfg for every group, where the same tensor is not evidence of an actuator-side
    # pre-clip demand.  Keep the declaration so an explicitly-actuated research leaf can opt in, but
    # default it OFF.  The effective-recipe builder also checks the composed actuator backend and
    # forcibly removes this term from the active ledger on implicit A3, so a stale YAML -0.5 cannot
    # resurrect a counterfeit objective.
    arm_torque_saturation = RewTerm(
        func=mdp.arm_torque_saturation, weight=0.0, params={"command_name": "racket_target"})
    # CHANGE 3 — balance shaping (POSITION-based): penalize forward base/torso TILT (proj_grav_xy) DURING
    # the approach (pre_strike), so the CoM stays over the support base THROUGH the swing (strike_upright
    # covers the strike window). NOT an angular-velocity penalty (those are gameable / anti-swing).
    # CLI-tunable via task.rewards.prestrike_upright_weight.
    prestrike_upright = RewTerm(
        func=mdp.prestrike_proj_grav_xy, weight=-1.0, params={"command_name": "racket_target"})

    # --- always-on balance + safety regularizers (kept) ---
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # base tilt
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # roll/pitch rate
    base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)  # vertical bob
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)  # joint-velocity smoothness
    # (inherited & kept: racket_position/velocity/normal, pre_strike_foot_slip, action_rate_l2,
    #  joint_torques, joint_limit, undesired_contacts, motion_global_anchor_ori.)


@configclass
class HOPEDeployParityTerminationsCfg(TerminationsCfg):
    """Swing-only reference envelopes plus always-on absolute fall/sink guards."""

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only_hold_aware,
        params={"command_name": "motion", "threshold": 0.25, "ignore_hold": True},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori_hold_aware,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion",
                "threshold": 0.8, "ignore_hold": True},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only_hold_aware,
        params={"command_name": "motion", "threshold": 0.25,
                "body_names": A3_FEET_BODIES + A3_HAND_BODIES, "ignore_hold": True},
    )

    base_fell_tilt = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})  # ~40 deg, absolute
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})

    # HIT THE TABLE — terminal, in the same class as falling over.  人话:打到桌子 = 这局结束。
    # Declared here so it reads alongside the other absolute guards; built by the shared factory
    # so the base HOPE env (which inherits the plain TerminationsCfg) gets exactly the same term
    # installed in __post_init__ rather than a second, drifting copy.
    robot_hit_table = table_hit_done_term()


@configclass
class HOPEPingPongAgibotA3EnvCfg(AgibotA3FlatEnvCfg):
    obs_mode: str = "full"  # descriptive; the deploy-parity variant is HOPEPingPongDeployParityAgibotA3EnvCfg
    # Stage-1 face-command obs switch (DEFAULT OFF = frozen actor contract, 175-D on deploy-parity).
    # True -> append racket_target_normal_cmd (+4: demanded face normal (3) + zero-filled rho
    # placeholder (1), the contract-day 175 -> 179 layout — rho is reserved for the S3 spin lane so
    # no later contract change / ladder retrain) to the actor group in __post_init__. train.py
    # toggles this AFTER __post_init__ has run, so its racket.face_command_obs override attaches
    # the term itself (same ObsTerm, same tail position).
    face_command_obs: bool = False
    # R10c 站位锚观测开关(DEFAULT OFF = 契约逐位不变;franco 2026-07-09 拍板)。True -> actor
    # 观测再追加 station_anchor_err_b(+2:世界系出生点锚 − 当前 base XY,旋进 base 系),排在
    # 拍面通道之后 = 179 -> 181。人话:给策略一个世界系"该站哪"的常数锚,躯干漂了这 2 维自己
    # 变大(R9a 删缰绳后拍随躯干漂移挥空的任务通道解法)。⚠ 必须与 face_command_obs 同开:
    # 单开会得到 177 维、且与 jiayi Hitter 的 177(站位插在第 167 列)布局不同——评估器按维数
    # 认契约会静默错位,__post_init__ 直接报错拦住。train.py 的 racket.station_obs 覆盖同 face
    # 时序:__post_init__ 已跑完,由 train.py 自己挂 ObsTerm(同名同尾部位置)。
    station_obs: bool = False
    # PHYSICAL ball + table truth instrument (Phase A; DEFAULT OFF = byte-identical scene/env).
    # True -> __post_init__ sets commands.racket_target.physical_ball and attaches the pb_ball /
    # pb_table / pb_table_visual scene entities (attach_physical_ball_scene). The train.py
    # task.physical_ball override runs AFTER __post_init__ (face_command_obs timing), so it calls
    # the (idempotent) attach itself. METRICS-ONLY — see physical_ball.py; racket impulse=Phase B.
    physical_ball: bool = False
    # TABLE OBSTACLE — the real table, as a solid the robot must not hit. DEFAULT **ON**.
    # 人话:桌子默认在场上。以前不在,机器人是在"桌子不存在"的世界里学挥拍,指令让它去
    # 桌面下面(z≈0.65 m)也没人拦得住;现在那里是实体,撞上就终止 + 扣分,和摔倒同一档。
    # 与 shadow_table / pb_table(两个 METRICS-ONLY 量球用的桌子,默认全关)不同:那两个是
    # 量具,这个是任务物体。三者用同一份 table_tennis.geometry 尺寸与位姿,不存在第二张桌子;
    # 若量具桌已挂,本项不再重复生成碰撞体,直接沿用它。
    # 关掉它 = 回到旧世界(仅供做"有桌/无桌"消融对照)。
    table_obstacle: bool = True
    #: filled in by attach_table_obstacle — which prim actually carries the table collider.
    table_obstacle_prim: str = ""
    #: all collider prims filtered by the wrist safety sensor; populated atomically with the scene.
    table_obstacle_prims: tuple[str, ...] = ()
    #: ActionBall-only conservative robot keep-out + net/posts.  It must remain false for every
    #: physical/shadow-ball truth-instrument task because the under-table proxy is not ball geometry.
    table_robot_keepout: bool = False
    # Wave-P random base push (DEFAULT OFF = events.push_robot stays None, the historical local
    # recipe every running matrix cell trains with; not a HITTER literature claim). 人话:训练时每隔几秒随机推
    # 机器人一把,练抗扰平衡;见 HOPEPushRobotCfg。train.py 的 task.push.* 覆盖在
    # __post_init__ 之后运行并自己构造 EventTerm(face_command_obs 时序);这里的旗标由
    # __post_init__ 末尾的 apply_push_robot_event 消费(cfg 直启路径)。
    push: HOPEPushRobotCfg = HOPEPushRobotCfg()
    # F-axis interval FORCE push (DEFAULT OFF = events.force_push/force_push_sweep stay None,
    # byte-identical). 人话:训练时每隔几秒朝水平随机方向对 pelvis_link 施加持续 duration_s
    # 的恒力,与速度推同冲量可比(Δv_equiv = F·Δt/m 记进合同);见 HOPEForcePushCfg。
    # train.py 的 task.force_push.* 覆盖在 __post_init__ 之后运行并自己构造事件对
    # (face_command_obs 时序);这里的旗标由 __post_init__ 末尾的 apply_force_push_event
    # 消费(cfg 直启路径)。
    force_push: HOPEForcePushCfg = HOPEForcePushCfg()
    commands: HOPECommandsCfg = HOPECommandsCfg()
    observations: HOPEObservationsCfg = HOPEObservationsCfg()
    rewards: HOPERewardsCfg = HOPERewardsCfg()
    events: HOPEEventCfg = HOPEEventCfg()

    def __post_init__(self):
        # AgibotA3FlatEnvCfg sets the robot, action scale, motion anchor/body names, and the A3
        # contact/termination/CoM body names (all valid for the inherited HOPE* cfg subclasses).
        super().__post_init__()
        # Multi-swing ping-pong must learn physical recovery between clips. Reset-time RSI remains active,
        # but clip wrap never teleports the robot back to the next reference start state
        # (MotionCommandCfg.wrap_teleport already defaults to False; kept explicit here).
        self.commands.motion.wrap_teleport = False
        # Stage-1 face-command channel: appended LAST in the actor group (configclass attribute order),
        # so every existing term keeps its slot and the contract only grows at the tail. The frozen
        # 175-D/180-D contracts stay byte-identical while the switch is off.
        if self.face_command_obs:
            self.observations.policy.racket_target_normal_cmd = ObsTerm(
                func=mdp.racket_target_normal_cmd, params={"command_name": "racket_target"}
            )
        # R10c 站位锚通道:必须排在拍面通道之后(179 前缀逐位不变 -> pad_obs_cols 纯尾部扩列
        # 热启才成立)。⚠ 布局取舍:契约日 181 蓝图是 175+站位2+拍面3+ρ1(站位在前);这里为了
        # 现役 179 存档能零成本热启,采用「179 + 尾部站位 2」,站位在拍面后——契约日统一时再定
        # 最终顺序(见 actor_observation_contract.DEPLOY_PARITY_STATION181 注释)。
        if self.station_obs:
            if not self.face_command_obs:
                raise ValueError(
                    "HOPEPingPongAgibotA3EnvCfg.station_obs=True requires face_command_obs=True: "
                    "单开站位会得到 177 维、且与 Hitter 的 177(站位在第 167 列)布局不同,评估器"
                    "按维数认契约会静默错位。R10c 的形状只有 181(=179+尾部站位2)。"
                )
            self.observations.policy.station_anchor_err_b = ObsTerm(
                func=mdp.station_anchor_err_b, params={"command_name": "racket_target"}
            )
        # SHADOW physical ball + table (metrics-only measurement; defaults OFF = scene untouched).
        # NOTE: train.py's racket.shadow_ball/shadow_table override runs AFTER this __post_init__
        # (same timing as face_command_obs), so it calls attach_shadow_ball_scene itself; the
        # helper is idempotent so both paths compose.
        rt = self.commands.racket_target
        if getattr(rt, "shadow_table", False) and not getattr(rt, "shadow_ball", False):
            raise ValueError(
                "RacketTargetCommandCfg.shadow_table=True requires shadow_ball=True "
                "(the table exists only for the shadow ball to land on)."
            )
        if getattr(rt, "shadow_ball", False):
            attach_shadow_ball_scene(self, shadow_table=bool(rt.shadow_table))
        # PHYSICAL ball + table truth instrument (Phase A; defaults OFF = scene untouched). Either
        # switch spelling works: the env-level flag here, or racket_target.physical_ball directly
        # (kept in sync both ways so the descriptive cfg stays honest). train.py's top-level
        # task.physical_ball override runs AFTER this __post_init__ and calls the idempotent
        # attach itself.
        if self.physical_ball or getattr(rt, "physical_ball", False):
            self.physical_ball = True
            rt.physical_ball = True
            attach_physical_ball_scene(self)
        # TABLE OBSTACLE — DEFAULT ON. Runs AFTER the two truth-instrument attachments so it can
        # see their table and reuse it rather than spawning a duplicate slab in the same place.
        apply_table_obstacle(self)
        # 合并互斥推 (defaults OFF = events.combined_push/_sweep stay None, byte-identical).
        # MUST run BEFORE the two legacy appliers: in combined mode they only step aside when
        # the merged event is already wired — reversed order fails loud (半配置绝不静默).
        apply_combined_push_event(self)
        # Wave-P random base push (defaults OFF = events.push_robot stays None, byte-identical).
        # This consumes the cfg-flag spelling; train.py's task.push override runs AFTER this
        # __post_init__ and builds the term itself. Both share the same validator/assembly
        # (training_contract.push_robot_event_block), so the two paths cannot drift.
        apply_push_robot_event(self)
        # F-axis interval FORCE push (defaults OFF = events.force_push/force_push_sweep stay
        # None, byte-identical). Same two-path wiring as the velocity push: this consumes the
        # cfg-flag spelling, train.py's task.force_push override runs AFTER this __post_init__
        # and builds the term pair itself; both share training_contract.force_push_event_block.
        apply_force_push_event(self)


@configclass
class HOPEPingPongDeployParityAgibotA3EnvCfg(HOPEPingPongAgibotA3EnvCfg):
    """Deploy-parity variant: deploy-honest actor observation (no fabricated base pose) plus
    absolute balance rewards/terminations. The ``full`` HOPEPingPongAgibotA3EnvCfg is left intact."""

    obs_mode: str = "deploy_parity"
    observations: HOPEObservationsDeployParityCfg = HOPEObservationsDeployParityCfg()
    rewards: HOPEDeployParityRewardsCfg = HOPEDeployParityRewardsCfg()
    terminations: HOPEDeployParityTerminationsCfg = HOPEDeployParityTerminationsCfg()


@configclass
class HOPEPingPongRealSensorAgibotA3EnvCfg(HOPEPingPongDeployParityAgibotA3EnvCfg):
    """Backward-compatible alias for the deploy-parity variant.

    Older docs and scripts still refer to this env as ``real_sensor_only`` / ``RealSensor``.
    The actor contract is the same deploy-parity 175-D layout.
    """


##
# Tier-1 virtual-ball variant (rewardDesign.md) — REWARD-ONLY on top of deploy-parity.
#
# The observation is the UNCHANGED deploy-parity 175-D actor contract (sim-to-real alignment is
# frozen; the virtual ball is never observed — it exists only inside the reward). Per swing the
# command term samples a virtual incoming ball that arrives at the racket target at strike time;
# at the exact-strike frame the achieved racket FK state is pushed through the venue-fitted paddle
# contact model + a coarse landing rollout, and the one-shot virtual_* terms below score the
# predicted shot (net clearance / landing accuracy / outgoing topspin).
##


@configclass
class HOPEVirtualBallRewardsCfg(HOPEDeployParityRewardsCfg):
    """DeployParity reward stack + Tier-1 virtual-ball outcome terms.

    Weights follow rewardDesign.md: landing 30 / pass_net 20 / spin 5 (start of the 5->10 ramp),
    ordered clear-net-first below landing per the PACE/v0 precedent. racket_velocity/racket_normal
    drop 2.0 -> 0.5: the contact model now scores the whole (velocity, normal, timing) manifold
    directly, so vector-matching the commanded velocity becomes shaping, not the task. The approach
    gradient (racket_position 4.0, racket_progress 10.0, racket_strike_success 5.0) is kept — the
    virtual terms are zero until the paddle reaches the 9.5 cm capture gate at the strike frame.
    """

    virtual_pass_net = RewTerm(
        func=mdp.virtual_pass_net, weight=20.0, params={"command_name": "racket_target"})
    virtual_landing = RewTerm(
        func=mdp.virtual_landing, weight=30.0, params={"command_name": "racket_target"})
    virtual_spin = RewTerm(
        func=mdp.virtual_spin, weight=5.0, params={"command_name": "racket_target"})

    # --- v2 蓝图 L2 击中层(reward_redesign_20260725 §3.5;DEFAULT weight=0.0 = IsaacLab 直接
    #     跳过 = 默认路径逐字节等价)。人话:每挥拍最多发一次的"击中大奖"one-shot——判据就是
    #     现成的 vb_fired 捕获门(exact-strike 一步 & 拍面正确半球 & 位置/进拍速达标,一拍锁存
    #     不重发),与 virtual_* 三项的区别是本项只认"打上了",不看出球质量。权重用【正数】;
    #     量级 B 由 redesign §2 定权公式给出(B = m1*I*Tc*rho/p*,名义 ~850,probe 校准后冻结
    #     prereg),reward_pack=v2 的翻译层直写这里的 weight。
    strike_capture_bonus = RewTerm(
        func=mdp.strike_capture_bonus, weight=0.0, params={"command_name": "racket_target"})
    # v2 值封顶版一阶平滑罚(fresh 自杀区间的解;v1 无封顶 action_rate_l2 照旧,v2 包
    # 归零它并启用本项,weight −0.2/clamp 9.0 来自冻结表)。默认 0 = 跳过,字节等价。
    action_rate_clamped = RewTerm(
        func=mdp.action_rate_l2_clamped, weight=0.0, params={"value_clamp": 9.0})
    # 死亡罚(Franco 07-26:延付只拉长刷分周期没关死——摔死重生仍在"跳过等下一球"上
    # 套利;罚值须比上台大奖大:weight −1800 = 每次死亡实际 −36 > 满分上台券 33)。
    # is_terminated 只计真终止(摔倒/包络),timeout 截断不罚。默认 0 = 跳过,字节等价。
    death_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    # 撞桌罚 table_hit_penalty 是 death_penalty 的同族窄版(只认 robot_hit_table 一个终止原因),
    # 由 apply_table_obstacle() 在 __post_init__ 里随桌子一起装/一起撤 —— 见 table_hit_rew_term()。
    # 不写死在这里,是因为桌子默认开但 terminations 类不止一个,装配点只能有一个。

    # D6 source gate (2026-07-14, DEFAULT OFF): penalize only the normalized tail above 85% of
    # each *actual articulation* joint-speed limit.  This is not action-rate smoothing: it reads
    # realized qdot and the 31 runtime-ordered limits directly, and fails closed on bad limits or
    # joint-order drift.  A future ablation enables it with a non-positive weight through Hydra.
    joint_velocity_limit_hinge = RewTerm(
        func=mdp.joint_velocity_limit_hinge,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "margin": 0.85,
            "expected_joint_count": 31,
        },
    )
    # Measurement-only twin of the hinge.  Hydra raises this term to weight 1.0 whenever a qdot
    # arm (including weight-zero control) explicitly binds the hinge setting.  Its function returns
    # exact zeros, so it observes qdot/excess eligibility without changing the task reward.
    joint_velocity_limit_hinge_probe = RewTerm(
        func=mdp.joint_velocity_limit_hinge_probe,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "margin": 0.85,
            "expected_joint_count": 31,
        },
    )

    # Deploy-space recovery slew (DEFAULT OFF): unlike action_rate_l2, this reads q_des after the
    # configured affine transform and train==deploy clamp.  It charges only the exact 15 waist/leg
    # joints during the same swing's 0.20..1.55 s post-contact recovery, normalized by each joint's
    # physical 20-ms velocity allowance.  The zero-valued probe is enabled for every explicitly
    # configured control/treatment arm so reset-invalid first steps and tail activation are auditable.
    processed_qdes_slew_hinge = RewTerm(
        func=mdp.processed_qdes_slew_hinge,
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "command_name": "racket_target",
            "margin": 0.85,
            "recovery_start_s": 0.20,
            "recovery_end_s": 1.55,
        },
    )
    processed_qdes_slew_hinge_probe = RewTerm(
        func=mdp.processed_qdes_slew_hinge_probe,
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "command_name": "racket_target",
            "margin": 0.85,
            "recovery_start_s": 0.20,
            "recovery_end_s": 1.55,
        },
    )

    # Wave Q qbar (DEFAULT OFF): all-joint q_des position-limit barrier — Jiayi V14's 全关节
    # top-k qdes barrier idea with the top-k removed (Franco 2026-07-21).  Every one of the 31
    # deploy-space targets pays as soon as it enters the margin band (margin_frac of its motion
    # range) next to a position limit, on every control step in every phase (dense, no gate).
    # The zero-valued probe books above-margin joint counts and the max intrusion depth for every
    # explicitly configured control/treatment arm.  人话:目标角贴近限位就罚,全身 31 关节全程
    # 盯着,不挑"最狠的几个";默认全关,配了就必带探针记账。
    qdes_limit_barrier = RewTerm(
        func=mdp.qdes_limit_barrier,
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "margin_frac": 0.08,
        },
    )
    qdes_limit_barrier_probe = RewTerm(
        func=mdp.qdes_limit_barrier_probe,
        weight=0.0,
        params={
            "action_name": "joint_pos",
            "margin_frac": 0.08,
        },
    )

    # mjlab-ported foot-contact shaping (DEFAULT OFF, 纯加法):落地冲击 + 摆动相抬脚高度。
    # 人话:foot_soft_landing 罚"落地砸太重"(first-contact 步法向峰值力超 300 N 的部分,
    # 有界);foot_clearance 罚"腾空脚又低又快地扫"(|脚高-目标高| x 水平速度),给"允许
    # 跨步"臂用。分工:近亲 foot_slip_sq/foot_drag 管触地脚的水平蹭滑,这两项一个管竖直
    # 冲击、一个管腾空高度,互不重复;mjlab 的"速度指令门"(站立自动关)刻意不搬——常开。
    # CLI:task.rewards.foot_soft_landing_weight / foot_soft_landing_force_threshold_n /
    # foot_clearance_weight / foot_clearance_target_m(weight 必须 <= 0;显式 0 是对照)。
    foot_soft_landing = RewTerm(
        func=mdp.foot_soft_landing,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(A3_FEET_BODIES)),
            "force_threshold_n": 300.0,  # A3 整机 58.2 kg -> 静重约 571 N,单脚静态约 285 N
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,
        params={
            # 接触状态查 sensor、脚位姿/速度查 articulation,两份名单必须同序同名(函数内硬校验)
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(A3_FEET_BODIES)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(A3_FEET_BODIES)),
            "target_m": 0.08,  # ankle_roll 原点目标高;站立贴地原点约 0.07 m,跨步臂建议往 0.15 调
        },
    )

    # mjlab 档①第三项 (DEFAULT OFF, 纯加法):action_acc_l2 动作二阶平滑。
    # 人话:action_rate_l2 罚"步子迈多大"(一阶差分),这条罚"方向掉头多猛"(二阶差分)——
    # 高频抖(chatter)恰恰是一阶小、二阶大的信号,平滑轴上的正交新轴。核
    # ||a_t - 2a_{t-1} + a_{t-2}||²,raw 动作与 action_rate_l2 同源同量纲;a_{t-2} 由
    # ClampedJointPositionAction 自存(isaaclab ActionManager 只有 prev_action),reset
    # 清零 + 有效位保证复位后前两步不计费。CLI:task.rewards.action_acc_weight(必须 <= 0;
    # 显式 0 是对照)。剂量:二阶量纲大于一阶,起步取 action_rate 档位的 1/5~1/2
    # (mjlab 采纳文档 §4),别按一阶惯用值抄。
    action_acc_l2 = RewTerm(
        func=mdp.action_acc_l2,
        weight=0.0,
        params={"action_name": "joint_pos"},
    )

    # Wave B (DEFAULT OFF): two mutually exclusive lower-body stability hypotheses.  B1 pays a
    # bounded pose kernel on the exact twelve leg joints from the current v4rg motion command.
    # B2 is reference-free: it combines an absolute anti-collapse stance-width hinge with only
    # the realized leg-qdot tail above a free margin.  Both use a phase opportunity/same-attempt
    # support gate, never hit success.
    # Their zero-valued probes are raised to weight 1.0 only by explicit B0/B1/B2 overrides.
    lower_body_pose_imitation = RewTerm(
        func=mdp.lower_body_pose_imitation,
        weight=0.0,
        params={
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "std": 0.35,
            "support_pre_s": 0.30,
            "support_post_s": 0.40,
        },
    )
    lower_body_pose_imitation_probe = RewTerm(
        func=mdp.lower_body_pose_imitation_probe,
        weight=0.0,
        params={
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "std": 0.35,
            "support_pre_s": 0.30,
            "support_post_s": 0.40,
        },
    )
    lower_body_stability_bundle = RewTerm(
        func=mdp.lower_body_stability_bundle,
        weight=0.0,
        params={
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "min_stance_width_m": 0.22,
            "stance_scale_m": 0.05,
            "leg_velocity_margin_radps": 1.0,
            "leg_velocity_scale_radps": 0.5,
            "support_pre_s": 0.30,
            "support_post_s": 0.40,
        },
    )
    lower_body_stability_bundle_probe = RewTerm(
        func=mdp.lower_body_stability_bundle_probe,
        weight=0.0,
        params={
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "min_stance_width_m": 0.22,
            "stance_scale_m": 0.05,
            "leg_velocity_margin_radps": 1.0,
            "leg_velocity_scale_radps": 0.5,
            "support_pre_s": 0.30,
            "support_post_s": 0.40,
        },
    )

    # S1 (DEFAULT OFF): post-swing settle-debt bundle, a clean main-side redo of the Jiayi V13
    # post-swing debts idea (unmerged branch; margins/scales re-fixed by this repo's conventions,
    # not copied).  Five bounded tails — base linear/angular quiet, upright tilt, nominal pelvis
    # height, and ankle-roll horizontal slip — averaged only inside the same 0.20..1.55 s
    # same-attempt recovery window the processed_qdes_slew_hinge uses (one shared clock).  The
    # zero-valued probe is raised to weight 1.0 by any explicit S1 override so eligibility and
    # per-debt income stay auditable even at weight 0.  人话:挥完拍这一秒多要稳稳站好,五项
    # "没站稳的债"超过免费额度才扣钱;默认全关,配了就必带探针记账。
    post_swing_settle_debt = RewTerm(
        func=mdp.post_swing_settle_debt,
        weight=0.0,
        params={
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "base_lin_margin_mps": 0.30,
            "base_lin_scale_mps": 0.20,
            "base_ang_margin_radps": 0.50,
            "base_ang_scale_radps": 0.30,
            "tilt_margin_rad": 0.10,
            "tilt_scale_rad": 0.10,
            "nominal_root_z_m": 1.0684,
            "root_height_deadband_m": 0.05,
            "root_height_scale_m": 0.05,
            "foot_slip_margin_mps": 0.05,
            "foot_slip_scale_mps": 0.10,
            "recovery_start_s": 0.20,
            "recovery_end_s": 1.55,
        },
    )
    post_swing_settle_debt_probe = RewTerm(
        func=mdp.post_swing_settle_debt_probe,
        weight=0.0,
        params={
            "racket_command_name": "racket_target",
            "motion_command_name": "motion",
            "base_lin_margin_mps": 0.30,
            "base_lin_scale_mps": 0.20,
            "base_ang_margin_radps": 0.50,
            "base_ang_scale_radps": 0.30,
            "tilt_margin_rad": 0.10,
            "tilt_scale_rad": 0.10,
            "nominal_root_z_m": 1.0684,
            "root_height_deadband_m": 0.05,
            "root_height_scale_m": 0.05,
            "foot_slip_margin_mps": 0.05,
            "foot_slip_scale_mps": 0.10,
            "recovery_start_s": 0.20,
            "recovery_end_s": 1.55,
        },
    )

    racket_velocity = RewTerm(
        func=mdp.racket_velocity_tracking_exp,
        weight=0.5,
        params={"command_name": "racket_target", "std": 0.5},
    )
    racket_normal = RewTerm(
        func=mdp.racket_normal_tracking_exp,
        weight=0.5,
        params={"command_name": "racket_target", "std": 0.262},
    )


@configclass
class HOPEPingPongVirtualBallAgibotA3EnvCfg(HOPEPingPongDeployParityAgibotA3EnvCfg):
    """Deploy-parity env + Tier-1 virtual-ball rewards. Obs/terminations/DR inherited untouched."""

    obs_mode: str = "deploy_parity"
    rewards: HOPEVirtualBallRewardsCfg = HOPEVirtualBallRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Reward-only feature switch: enables the per-swing virtual-ball sampler and the at-strike
        # contact + coarse-landing evaluation in RacketTargetCommand (vb_* cfg fields hold the
        # venue-fit sampling boxes / gates; tune there, not here).
        self.commands.racket_target.virtual_ball = True
        # CLIMB-PHASE shaping width (2026-07-03): the E-champion warm start crosses the net plane
        # ~0.3-0.5 m BELOW the target height; at the v0 default sigma 0.10 the height kernel is
        # exp(-(0.5/0.1)^2) ~ 0 there — no gradient, and vb_warmE14k3 paid zero virtual reward for
        # 2.5k iters. 0.25 keeps a usable gradient down to the current operating band. Tighten
        # back toward 0.10 once virtual_net_clear_rate is healthy (>0.3 or so).
        self.commands.racket_target.vb_net_sigma = 0.25
        # CLIMB-PHASE landing kernel width (2026-07-04): landings start ~1.9 m short of the target
        # (exp(-(1.9/0.3)^2) = 0 — the v0 sigma has no reach); 1.0 pays 0.03 at the current band
        # and grows monotonically toward the target = dense "hit deeper" gradient (the kernel is
        # also ungated from net clearance during the climb — see hope_rewards.virtual_landing).
        # Tighten back toward 0.3 together with re-gating once the net terms carry the signal.
        self.commands.racket_target.vb_landing_sigma = 1.0


##
# HITTER-footwork variant (arXiv:2508.21043 §V-B-1 "Separate Commands for Base and Racket") —
# deploy-parity base + the base-position command channel restored (2026-07-05).
#
# WHY: the BASE-FREE deploy-parity policy self-selects walk-and-strike footwork toward deep
# world-frame racket targets ("chasing a point forward"); it cannot be commanded to a station.
# HITTER instead (a) commands the base to a world XY station, (b) fixes the striking plane
# RELATIVE to the robot (0.4 m in front on their G1; our analog = each clip's reference
# base→racket strike offset), sampling only the racket target's y/z spread, and (c) activates
# base tracking only PRE-STRIKE (mdp.base_position_tracking_exp is already gated that way).
#
# SIM2REAL CONTRACT (177-D actor = 175-D deploy-parity + base_target_pos_b(2) restored at its
# original slot between projected_gravity and racket_target_pos_b):
#   * base_target_pos_b is a RELATIVE Δxy in the yaw-heading frame — computable on hardware from
#     the mocap base position (300 Hz, position-only; hope-mocap-spec) + IMU yaw-align-at-engage.
#     No absolute world coordinates enter the obs; mocap dropout → feed Δ=0, which degrades
#     gracefully to "already at station" (today's BASE-FREE behavior).
#   * A1 target latency/jitter does NOT yet degrade the base channel (the racket channel does);
#     the base station demand is O(10 cm), obs Unoise covers mocap noise. Revisit if hardware
#     shows base-channel transport lag matters.
#   * The C++ runner (pp_policy.hpp build_obs_175) and mujoco_eval_onnx are 175-D and need the
#     177-D layout + a planner base-target input before this variant can deploy — verify with
#     scripts/verify_realsensor.py layout print after any obs change.
##


@configclass
class HOPEObservationsHitterCfg(HOPEObservationsDeployParityCfg):
    """Deploy-parity actor obs + the HITTER base-position command channel (175 -> 177)."""

    @configclass
    class HOPEPolicyHitterCfg(HOPEObservationsDeployParityCfg.HOPEPolicyDeployParityCfg):
        # Restore the base-repositioning target (Δxy, yaw-heading frame). Overriding the parent's
        # `= None` puts the term back at its ORIGINAL declaration slot (configclass inheritance
        # preserves attribute order): between projected_gravity and racket_target_pos_b.
        base_target_pos_b = ObsTerm(
            func=mdp.base_target_pos_b,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.03, n_max=0.03),  # ~mocap base-position noise at 300 Hz
        )

    @configclass
    class HOPECriticHitterCfg(HOPEObservationsDeployParityCfg.HOPECriticDeployParityCfg):
        # The critic conditions on the station too now that a reward consumes it.
        base_target_pos_b = ObsTerm(func=mdp.base_target_pos_b, params={"command_name": "racket_target"})

    policy: HOPEPolicyHitterCfg = HOPEPolicyHitterCfg()
    critic: HOPECriticHitterCfg = HOPECriticHitterCfg()


@configclass
class HOPEHitterRewardsCfg(HOPEDeployParityRewardsCfg):
    """Deploy-parity rewards + the HITTER base-repositioning goal reward restored.

    mdp.base_position_tracking_exp is PRE-STRIKE gated in the reward function itself (HITTER:
    "the base position tracking reward is activated only before the strike"). std 0.3 m matches
    the original HITTER-alignment tuning (HOPERewardsCfg.base_position).
    """

    base_position = RewTerm(
        func=mdp.base_position_tracking_exp,
        weight=1.0,
        params={"command_name": "racket_target", "std": 0.3},
    )


@configclass
class HOPEPingPongHitterAgibotA3EnvCfg(HOPEPingPongDeployParityAgibotA3EnvCfg):
    """Deploy-parity env + HITTER separate base/racket commands (obs 177-D; NOT deploy-compatible
    with the 175-D C++ runner until the runner/planner grow the base channel)."""

    obs_mode: str = "hitter_footwork"
    observations: HOPEObservationsHitterCfg = HOPEObservationsHitterCfg()
    rewards: HOPEHitterRewardsCfg = HOPEHitterRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # HITTER coupling: base station derived from the racket target at the clip's reference
        # reach — the striking plane is fixed relative to the COMMANDED base; the box x-span moves
        # the station, not the reach depth. Jitter ranges (base_target_*_range) train deliberate
        # station offsets (y-reach diversity); the yaml preset owns their spans.
        self.commands.racket_target.base_couple_mode = "reference_reach"


@configclass
class HOPEActionBallRewardsCfg(HOPEVirtualBallRewardsCfg):
    """VirtualBall/v2-capable outcome stack plus HITTER's commanded-base objective.

    Action-conditioned ball-first needs both parent lineages at once:

    * ``HOPEVirtualBallRewardsCfg`` declares the complete virtual-return ledger/reward surface
      consumed by ``reward_pack=v2`` (including the non-zero ``virtual_landing`` result term);
    * HITTER's native ``base_target_pos_b`` observation must have the matching pre-strike
      ``base_position`` reward instead of an unobserved or reward-free base goal.

    Keeping this as a new leaf class is deliberate.  The historical Hitter and VirtualBall task
    IDs retain their exact reward declarations and defaults.
    """

    base_position = RewTerm(
        func=mdp.base_position_tracking_exp,
        weight=1.0,
        params={"command_name": "racket_target", "std": 0.3},
    )

    # The -72 terminal charge is a HARD-SAFETY union, not a generic episode-reset tax.
    # Reference-consistency envelopes (anchor_pos/anchor_ori/ee_body_pos) remain valid
    # terminations during the center phase, but are independently classified by the runtime
    # ledger and receive no death charge.  The exact union is duplicated in the DoneTerm class
    # below and is checked again by the effective-reward/audit contracts.
    death_penalty = RewTerm(
        func=mdp.action_ball_safety_terminated,
        weight=0.0,
        params={
            "term_names": (
                "base_fell_tilt",
                "base_too_low",
                "joint_actual_forbidden",
                "joint_qdes_forbidden",
                "robot_hit_table",
            )
        },
    )

    # Fresh ActionBall soft-limit v2.  The q_des and actual-q channels are deliberately separate
    # objective terms (and therefore separate rows in the effective Reward receipt/activation
    # ledger): command clipping must not hide inertial actual-joint intrusion, while actual-q
    # tracking must not hide an exploitative command that happens not to be realized yet.
    #
    # At policy_dt=0.02, weight -40 and floor 0.25 charge at least -0.20 per intruding joint per
    # channel per step.  A one-joint micro-intrusion sustained for one second costs -10 in one
    # channel (-20 if both command and plant persist); a full-depth joint costs at most -0.80 per
    # channel per step.  Even the impossible single-step maximum (31 joints, both channels) is
    # -49.6, below generic hard death -72.  The launch ablation grid is -20/-40/-80; -40 is the
    # adopted middle dose, not an unreviewed escalation.
    qdes_limit_barrier = RewTerm(
        func=mdp.qdes_limit_barrier_v2,
        weight=-40.0,
        params={
            "action_name": "joint_pos",
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
        },
    )
    qdes_limit_barrier_probe = RewTerm(
        func=mdp.qdes_limit_barrier_v2_probe,
        weight=1.0,
        params={
            "action_name": "joint_pos",
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
        },
    )
    # A finite Gaussian proposal outside the drive's target envelope is projected rather than
    # reset.  This independent cost teaches the actor to keep its mean/noise inside that envelope;
    # the existing q_des barrier above still shapes the *executed* target near its soft edge.
    #
    # With shape_rate 4, a 10%-of-envelope projection costs 0.330 before weighting.  At
    # policy_dt=0.02 and the adopted weight -5 that is -0.033 per affected joint-step; the
    # asymptotic maximum is -0.10 per joint-step.  Since all 31 joints are summed, -5 is the
    # conservative main-run dose that cannot easily swamp early imitation/hit income.  A -20
    # strong-dose comparison may be preregistered separately; it is not tonight's default.
    qdes_projection_penalty = RewTerm(
        func=mdp.qdes_projection_penalty,
        weight=-5.0,
        params={
            "action_name": "joint_pos",
            "shape_rate": 4.0,
        },
    )
    # Override Isaac Lab's inherited ``joint_pos_limits`` tail: that legacy term is exactly zero
    # until actual q has already crossed the soft limit, so it cannot prevent "grazing" the band.
    joint_limit = RewTerm(
        func=mdp.actual_joint_limit_barrier_v2,
        weight=-40.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
            "expected_joint_count": 31,
        },
    )
    actual_joint_limit_barrier_probe = RewTerm(
        func=mdp.actual_joint_limit_barrier_v2_probe,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "margin_frac": 0.08,
            "penalty_floor": 0.25,
            "expected_joint_count": 31,
        },
    )


@configclass
class HOPEActionBallTerminationsCfg(HOPEDeployParityTerminationsCfg):
    """Action-ball hard joint-safety masks in addition to fall/table termination.

    The q_des guard uses the physical ``joint_pos_limits`` with a two-percent-per-side inset to
    select a finite brake target, while finite ActionBall proposals are projected and trained by
    the projection/barrier costs.  The actual-q Done term reserves reset for a non-finite state or
    a current/substep raw mechanical hard edge.  Recoverable occupancy of the two-percent inner
    band stays observable but is taught by the strong actual-q barrier instead of discarding the
    transition.  Fall and table contact remain independent hard terminations.
    """

    joint_qdes_forbidden = DoneTerm(
        func=mdp.pre_clamp_qdes_forbidden_zone,
        time_out=False,
        params={
            "action_name": "joint_pos",
            "limit_source": "joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
        },
    )
    joint_actual_forbidden = DoneTerm(
        func=mdp.actual_joint_position_forbidden_zone,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_source": "joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
        },
    )


@configclass
class HOPEPingPongActionBallAgibotA3EnvCfg(HOPEPingPongHitterAgibotA3EnvCfg):
    """Action-conditioned ball-first task source before the fresh N1 v2 tail is installed.

    The class supplies the clean 177-D ``hitter_footwork`` prefix and the complete VirtualBall
    reward/safety lineage.  After verifying the one-action manifest and fixed-194 v2 contract,
    ``train.py`` atomically installs table pose/twist, the demanded-face ``+4`` and the
    teacher-start clock.  Stable action UID and dense slot remain control-plane state and are not
    actor observations.  Multi-action N5/N73 fail closed until a separately versioned,
    fixed-width continuous future-motion intent exists.
    """

    obs_mode: str = "hitter_footwork"
    # This task has an analytic ball.  It may therefore use the conservative robot-only under-table
    # proxy without changing ball flight/bounce physics.
    table_robot_keepout: bool = True
    observations: HOPEObservationsHitterCfg = HOPEObservationsHitterCfg()
    rewards: HOPEActionBallRewardsCfg = HOPEActionBallRewardsCfg()
    terminations: HOPEActionBallTerminationsCfg = HOPEActionBallTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The analytic incoming ball installed by the action-ball transaction is also the one
        # scored by the VirtualBall outcome ledger.  This is not ``vb_metrics_only``: v2's
        # virtual_landing term is an active training reward.
        self.commands.racket_target.virtual_ball = True
        # Match the reviewed VirtualBall source defaults.  The action-ball solver owns the exact
        # incoming state and task; these two values only shape the shared net/landing evaluator.
        self.commands.racket_target.vb_net_sigma = 0.25
        self.commands.racket_target.vb_landing_sigma = 1.0
        # Fail closed before each physics step: the raw affine q_des plus fresh q/qdot
        # projections must stay inside the physical hard envelope.  Trigger plant-state braking
        # at five percent of hard travel, deliberately earlier than the existing two-percent
        # proximity/bootstrap envelope.  On the current A3 plant the finite nominal q_des
        # projection is already tighter, so this containment moves only the intervention point;
        # the raw mechanical-edge DoneTerm remains unchanged.  The action term independently
        # verifies the exact 20 ms policy horizon and four-substep runtime contract.
        self.actions.joint_pos.pre_apply_limit_guard = True
        self.actions.joint_pos.pre_apply_guard_policy_dt_s = 0.02
        self.actions.joint_pos.pre_apply_guard_expected_decimation = 4
        self.actions.joint_pos.pre_apply_guard_terminal_archive_capacity = 4096
        self.actions.joint_pos.pre_apply_guard_margin_rad = 0.0
        self.actions.joint_pos.pre_apply_guard_margin_fraction = 0.05
        # Finite actor proposals outside the physical hard-inner envelope are ordinary
        # constrained actions: execute the same nearest safe target projection used by deploy
        # parity and preserve the transition for the dense projection-distance penalty.  NaN/Inf
        # and actual/predicted physical crossings remain brake-and-terminate.
        self.actions.joint_pos.project_finite_preclamp_qdes_without_termination = True
        # Observe the same resolved robot_hit_table term at every physics substep.  The action term
        # records substeps 1..3 before the next write; the DoneTerm finalizes substep 4.
        self.actions.joint_pos.table_contact_substep_guard = True
        self.actions.joint_pos.table_contact_guard_termination_term = (
            "robot_hit_table"
        )
        self.actions.joint_pos.table_contact_guard_expected_decimation = 4
        self.terminations.robot_hit_table.params["require_substep_latch"] = True


##
# HITTER-PURE variant (2026-07-07) — faithful reproduction of the paper's MDP, replacing the
# accumulated HOPE machinery. Decision context: model_17400 (177-D hitter_footwork) deploys and
# stands on hardware but swings on ~1/10 served balls and misses — the trained distribution is
# clip-centered and narrow, the actor carries the 62-D reference stream (paper: CRITIC-only,
# Table I), and the face-normal target was locked to the reference clip (paper §IV-C: the racket
# plane is PERPENDICULAR TO ITS VELOCITY at impact). This variant re-aligns all three.
#
# vs the paper (arXiv:2508.21043), EXACT alignment:
#   * Actor obs = Table I structure sized for the A3 (110-D): ang vel, gravity, e_base,x,
#     Δbase target (world xy), racket target rel base (world), racket target vel (world),
#     time-to-strike, q/q̇/a_last. NO reference joints, NO swing_type, NO anchor terms.
#   * Separate commands (§V-B-1): base station sampled INDEPENDENTLY (paper Fig. 4: up to
#     ±0.75-0.8 m, 1 cm arrival in <0.8 s); racket target on a plane FIXED relative to the
#     commanded station (their 0.4 m on the G1; our A3 analog = the clips' blade reach 0.70 m),
#     only y/z sampled, per-swing-type non-overlapping regions.
#   * Normal target = velocity direction (§IV-C impact model) — the policy must LEARN the wrist
#     orientation (initial error 18-110°; expected to learn slowly, do NOT "fix" it by moving
#     the target back to the reference normal — that is how legal returns became 0%).
#   * Reward = dense upper-body imitation + sparse goal (racket pos/vel/normal in the strike
#     window; base position pre-strike only) + generic regularization. NO hold_ready, NO foot/
#     stability shaping, NO HER replay, NO base_decel (paper has none of them).
#   * 10 s episodes, multiple swings, swing type + targets resampled per swing, no hold phase.
#
# Deliberate departures (kept, with reasons):
#   * stand_start_prob 0.25 + no-teleport wraps (deploy-honest entry/transition; paper does not
#     document its reset scheme).
#   * DR keeps PD ±15% / link mass ±15% (sim2real; paper fixes PD).
#   * Tuned kernel widths from the 0625-0706 lineage (paper publishes no weights/stds).
#
# Deploy contract: 110-D `hitter_pure` — needs a NEW C++ obs builder + a planner that streams
# (station, racket target, vel, tts) CONTINUOUSLY (no engage-lock). See actor_observation_contract.
##


@configclass
class HOPEObservationsHitterPureCfg(HOPEObservationsCfg):
    """HITTER Table-I actor (110-D, world-frame targets + e_base,x); critic unchanged (privileged)."""

    @configclass
    class HOPEPolicyHitterPureCfg(ObservationsCfg.PolicyCfg):
        # --- remove every non-Table-I term from the BeyondMimic base actor ---
        command = None  # 62-D reference joint stream: CRITIC-ONLY in HITTER (Table I)
        motion_anchor_pos_b = None  # needs world base position; not in Table I
        motion_anchor_ori_b = None  # reference-coupled orientation error; Table I uses e_base,x instead
        base_lin_vel = None  # critic-only in HITTER (not measurable on hardware)
        # --- Table I goal terms (appended after the inherited proprioception) ---
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_forward_xy = ObsTerm(
            func=mdp.base_forward_xy,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        base_target_delta_xy = ObsTerm(
            func=mdp.base_target_delta_xy,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.03, n_max=0.03),  # ~mocap base-position noise
        )
        racket_target_rel_base = ObsTerm(
            func=mdp.racket_target_rel_base,
            params={"command_name": "racket_target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        racket_target_vel_w = ObsTerm(func=mdp.racket_target_vel_w, params={"command_name": "racket_target"})
        time_to_strike = ObsTerm(func=mdp.time_to_strike, params={"command_name": "racket_target"})

    @configclass
    class HOPECriticHitterPureCfg(HOPEObservationsCfg.HOPECriticCfg):
        # Table I checkmarks EVERY actor term in the critic column too — make the critic a strict
        # actor superset (audit 2026-07-07): the inherited critic lacked projected_gravity and the
        # world-frame goal view (it only had the yaw-heading-frame legacy accessors). Live,
        # noise-free variants; the privileged heading-frame/FK extras above are kept.
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_forward_xy = ObsTerm(func=mdp.base_forward_xy, params={"command_name": "racket_target"})
        base_target_delta_xy = ObsTerm(
            func=mdp.base_target_delta_xy, params={"command_name": "racket_target"}
        )
        racket_target_rel_base = ObsTerm(
            func=mdp.racket_target_rel_base, params={"command_name": "racket_target"}
        )

    policy: HOPEPolicyHitterPureCfg = HOPEPolicyHitterPureCfg()
    critic: HOPECriticHitterPureCfg = HOPECriticHitterPureCfg()
    # critic = HOPECriticCfg (reference joints, body poses T_B, base lin vel, time-left, live
    # targets + actual racket FK state) + the actor's world-frame goal terms above — a strict
    # superset of both the paper's critic and the actor. Privileged sim-side only, never deployed.


@configclass
class HOPEHitterPureRewardsCfg(RewardsCfg):
    """HITTER §V-B-2 faithful reward stack: r = w_i·r_imitation + w_g·r_goal + w_r·r_regularization.

    * r_imitation — dense, UPPER-BODY reference only (paper §V-A: B = bodies above the pelvis);
      the base is steered by the GOAL terms, not imitation (motion_global_anchor_pos removed).
    * r_goal — sparse, relatively high weights (paper): racket pos/vel/NORMAL tracking in the
      strike window; base position tracking PRE-STRIKE only (gated inside the fn).
    * r_regularization — generic energy/smoothness/safety only. NO hold_ready / foot shaping /
      waist twist / strike-window stability / torque saturation — the paper has none of them.

    Weights/stds are HOPE tuning (the paper publishes neither); the task YAML owns the numbers.
    """

    # --- imitation: upper-body only, swing-gated (hold refs are ready-stand; legs decoupled) ---
    motion_global_anchor_pos = None  # base position is a GOAL (base_position), not imitation
    motion_body_pos = RewTerm(func=mdp.motion_body_pos_swing_only, weight=1.0,
        params={"command_name": "motion", "std": 0.3, "body_names": A3_UPPER_TRACKED})
    motion_body_ori = RewTerm(func=mdp.motion_body_ori_swing_only, weight=1.0,
        params={"command_name": "motion", "std": 0.4, "body_names": A3_UPPER_TRACKED})
    motion_body_lin_vel = RewTerm(func=mdp.motion_body_lin_vel_swing_only, weight=1.0,
        params={"command_name": "motion", "std": 1.0, "body_names": A3_UPPER_TRACKED})
    motion_body_ang_vel = RewTerm(func=mdp.motion_body_ang_vel_swing_only, weight=1.0,
        params={"command_name": "motion", "std": 3.14, "body_names": A3_UPPER_TRACKED})

    # --- goal (sparse; strike-window / pre-strike gating lives inside the reward fns) ---
    racket_position = RewTerm(func=mdp.racket_position_tracking_exp, weight=14.0,
        params={"command_name": "racket_target", "std": 0.15})
    racket_velocity = RewTerm(func=mdp.racket_velocity_tracking_exp, weight=14.0,
        params={"command_name": "racket_target", "std": 0.6})
    racket_normal = RewTerm(func=mdp.racket_normal_tracking_exp, weight=5.0,
        params={"command_name": "racket_target", "std": 0.30})
    base_position = RewTerm(func=mdp.base_position_tracking_exp, weight=2.0,
        params={"command_name": "racket_target", "std": 0.20})
    racket_strike_success = RewTerm(func=mdp.racket_strike_success, weight=5.0,
        params={"command_name": "racket_target", "std_pos": 0.075, "std_vel": 0.5, "std_normal": 0.262})
    # OFF by default (not in the paper). Declared as a fallback shaping knob: if from-scratch
    # exploration cannot find the strike window over the wide station box, re-enable via
    # task.rewards (racket_progress telescopes to distance-reduced; weight 0.0 = skipped).
    racket_progress = RewTerm(func=mdp.racket_progress, weight=0.0, params={"command_name": "racket_target"})

    # --- regularization (generic only) ---
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    # (inherited & kept: motion_global_anchor_ori 0.5, action_rate_l2 -0.1, joint_limit -10,
    #  undesired_contacts -0.1.)

    # --- continuous-rally recovery terms (2026-07-07) — weight 0.0 = SKIPPED (RewardManager drops
    # zero-weight terms), so plain HitterPure stays byte-identical / paper-faithful. The Rally
    # variant (HOPEPingPongHitterPureRallyAgibotA3EnvCfg + its YAML) enables them:
    #   post_strike_brake — positive braking kernel through the follow-through ((~pre_strike) &
    #     (~strike_window)); arrests the walk-and-strike lunge momentum (deploy P7 drift fall).
    #   hold_ready — the 177-proven settle term (stillness x planted feet), in_hold-gated with the
    #     STATION reach gate (std 1.5 / reach 0.20 / "station": the YAML-proven numbers — the code
    #     defaults std 0.5/reach "racket" are dead/arm-gameable, see HOPEPingPongHitter.yaml notes).
    post_strike_brake = RewTerm(func=mdp.post_strike_brake, weight=0.0,
        params={"command_name": "racket_target", "std": 0.5})
    hold_ready = RewTerm(func=mdp.hold_ready, weight=0.0,
        params={"command_name": "racket_target", "std": 1.5, "reach": 0.20, "reach_mode": "station"})
    # Hold-only heading restoration. Plain HitterPure keeps this at zero; RallyV3 pairs it
    # with yawed stand starts so the policy sees and learns the deploy recovery state.
    hold_heading = RewTerm(func=mdp.hold_heading, weight=0.0,
        params={"command_name": "racket_target", "std": 0.6})
    # Foot discipline (declared 2026-07-07 after the pigeon-toe diagnosis): with lower-body
    # imitation absent (paper §V-A: B = above pelvis) the hip-yaw DOF are reward-free and the
    # policy toe-ins HARD while stepping — obs-CSV quantification on model_12200 in the AGI sim:
    # hip_yaw deviation p95 ±0.94 rad, max 1.69 (reference envelope ±0.41; ankle/hip_roll clean;
    # standing clean — it is ONLY the moving gait). Same pathology foot_orientation_discipline
    # was built for on 2026-07-05 (177 ran it at -0.3). Enable on a 12200 resume via
    # task.rewards.foot_orientation_weight=-0.3; gate: single-swing det >= 0.99 AND g25 oracle
    # 10/10 must both hold, then re-run the obs-CSV diag (hip_yaw p95 back inside ~0.41).
    foot_orientation = RewTerm(func=mdp.foot_orientation_discipline, weight=0.0,
        params={"command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint"]),
                "hold_gate": False})


@configclass
class HOPEHitterPureTerminationsCfg(HOPEDeployParityTerminationsCfg):
    """HitterPure reference envelopes are swing-only; absolute fall guards stay live.

    Held RSI intentionally combines default-stand joints/root height with the next clip's
    frozen windup body reference.  Applying the reference-relative torso/body/orientation
    envelopes to that mixed state can terminate a valid reset before the actor reaches a
    swing.  HitterPure's 110-D actor cannot observe the reference body stream, so these terms
    are explicitly ignored during hold.  ``base_fell_tilt`` and ``base_too_low`` are inherited
    unchanged and remain active on every step, including hold.
    """

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only_hold_aware,
        params={"command_name": "motion", "threshold": 0.25, "ignore_hold": True},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori_hold_aware,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
            "ignore_hold": True,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only_hold_aware,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "ignore_hold": True,
        },
    )


@configclass
class HOPEPingPongHitterPureAgibotA3EnvCfg(HOPEPingPongAgibotA3EnvCfg):
    """Faithful HITTER MDP on the A3 (110-D hitter_pure actor contract). Code defaults below MIRROR
    cfg/task/HOPEPingPongHitterPure.yaml so eval/verify scripts that bypass train.py see the same
    task — keep the two in sync (the YAML wins at train time)."""

    obs_mode: str = "hitter_pure"
    observations: HOPEObservationsHitterPureCfg = HOPEObservationsHitterPureCfg()
    rewards: HOPEHitterPureRewardsCfg = HOPEHitterPureRewardsCfg()
    terminations: HOPEHitterPureTerminationsCfg = HOPEHitterPureTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # HITTER episode: 10 s, multiple swings, no hold phase (consecutive strikes train the
        # between-swing recovery; deploy idle is the runner's static-stand handoff, not a policy
        # state). stand_start_prob keeps the deploy entry in-distribution (min-hold 25 = 0.5 s to
        # settle stand -> windup); post-swing buffer starts OFF (not in the paper).
        self.episode_length_s = 10.0
        self.commands.motion.hold_steps_range = (0, 0)
        self.commands.motion.post_swing_start_prob = 0.0

        # Mirror the YAML's latest Agibot split-gain DR exactly for verify/eval/export paths that
        # bypass train.py: Kd uncertainty is intentionally wider than Kp.
        self.events.randomize_pd_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_pd_gains.params["damping_distribution_params"] = (0.7, 1.3)

        C = self.commands.racket_target
        C.target_mode = "hitter_pure"
        C.normal_mode = "velocity"  # §IV-C: racket plane ⊥ velocity at impact (LEARNED, not ref-locked)
        # Unified forehand/backhand strikes with opposite physical paddle faces. This belongs in
        # the code default as well as the YAML because verify/export paths bypass train.py.
        C.mount_normal_sign_per_clip = (1.0, -1.0)
        C.strike_phase_per_clip = (0.47, 0.333)  # blade-speed-peak re-plane (hopex clips, 2026-07-02)
        C.strike_window_s = 0.12
        C.clean_reference_strike_velocity = True
        C.achieved_target_mix_prob = 0.0  # no HER in the paper
        # Independent STATION box (world xy around the env origin; paper Fig. 4 goes to ±0.75-0.8 m —
        # start at ±0.40 = the proven trained band, widen on resume once arrival is established).
        # X-PLANE LOCKED (2026-07-08): station x FIXED at spawn — mirrors HOPEPingPongHitterPure.yaml.
        C.base_target_x_range = (0.0, 0.0)
        C.base_target_y_range = (-0.40, 0.40)
        # STATION-RELATIVE racket boxes: x = the FIXED striking plane (blade reach of both clips
        # ≈ 0.70 m in front of the commanded station), y = per-swing non-overlapping bands centered
        # on each clip's natural lateral reach (fh −0.409 / bh +0.185), z = absolute height bands
        # centered on each clip's blade strike height (fh 0.82 / bh 1.03), half-width 0.15.
        # STRIKING-PLANE x FIX (2026-07-08): x 0.70 -> 0.51 (mirrors HOPEPingPongHitterPure.yaml).
        # 0.70 was 0.16-0.22 m TOO FAR vs the demo racket (pingpang_red_Link rel-station world x =
        # 0.484 fh / 0.542 bh), which forced the forward lunge/lean. 0.51 = demo midpoint so the
        # racket reaches the plane with the base AT the locked station.
        # TABLE-CLEARANCE FIX (2026-07-26): forehand z (0.67, 0.97) -> (0.78, 1.08). The striking
        # plane sits past the near table edge (vb_table_near_x 0.5), so the lowest commandable
        # contact is vb_table_surface_z 0.76 + ball radius 0.02 = 0.78; the old floor asked for
        # contact inside the table top. Span preserved (0.30 m). Enforced by
        # RacketTargetCommand._assert_contact_clears_table.
        C.racket_pos_range_per_clip = (
            ((0.51, 0.51), (-0.65, -0.15), (0.78, 1.08)),  # forehand
            ((0.51, 0.51), (-0.05, 0.45), (0.88, 1.18)),   # backhand
        )
        # Blade-replaned per-clip velocity boxes (world frame, 2026-07-02 lineage).
        C.racket_vel_range_per_clip = (
            ((1.05, 2.05), (0.96, 1.96), (0.31, 1.11)),    # forehand
            ((1.61, 2.61), (-1.21, -0.21), (0.00, 0.71)),  # backhand
        )


@configclass
class HOPEPingPongHitterPureRallyAgibotA3EnvCfg(HOPEPingPongHitterPureAgibotA3EnvCfg):
    """CONTINUOUS-RALLY variant of HitterPure (2026-07-07). ⚠ POST-MORTEM: the Gate-2.5 P7 fall
    this task targeted turned out to be the C++ runner's Δ=0-idle artifact, NOT a training gap —
    fixed deploy-side (pp_policy.hpp idle-anchor; model_12200 = ORACLE 10/10 with ZERO
    retraining). The first run of this task (model_18000, weights 1.0) traded single-swing strike
    0.994→0.866 and still failed deploy (P4b) — archived, not deployed. KEPT default-off as
    tooling for future genuine multi-swing robustness (e.g. station widening toward ±0.75 m);
    see the YAML header post-mortem for the lessons (brake ≤0.3, hold_ready reach 0.30, gate any
    candidate on single-swing det ≥0.95 FIRST).

    Mechanics (all existing machinery, unchanged facts): same 110-D contract/boxes/DR/
    terminations as Pure (strict warm-resume works); swing -> follow-through BRAKE -> 0.5-2.5 s
    HOLD (settle at the NEXT station — the wrap resamples target+station+clip BEFORE the hold)
    -> windup -> swing, 3-4 swings per 16 s episode, fh/bh 50/50 per wrap. Hold obs: tts frozen
    POSITIVE at the windup value (== the runner's idle clamp), ready-stand reference, imitation
    auto-zeroed (*_swing_only), base_position live toward the new station. Code defaults MIRROR
    cfg/task/HOPEPingPongHitterPureRally.yaml — edit BOTH."""

    def __post_init__(self):
        super().__post_init__()
        # 16 s: ~3-4 swing+hold cycles per episode (swing ~2.7 s + hold 0.5-2.5 s) — enough
        # consecutive cycles for drift to hurt WITHIN an episode, so the policy must learn to
        # cancel it; 10 s gave only ~2 held cycles.
        self.episode_length_s = 16.0
        # THE structural change: a real recovery window at EVERY wrap (and reset). 25-125 steps =
        # 0.5-2.5 s @50 Hz — inside the deploy envelope (runner hold_recover_s 2.5 s policy-active,
        # scripted P7 holds ~4.5 s). Deliberately NOT longer: holds pay no goal income, so hold
        # steps dilute strike-gradient sample efficiency.
        self.commands.motion.hold_steps_range = (25, 125)
        # A held RSI birth commands default-stand joints. Keep its root at the default stand height
        # too; using the windup crouch z puts the stand feet below the floor at reset.
        self.commands.motion.rsi_hold_root_stand_z = True
        # Recovery income (weights mirror the Rally YAML; 0.0 in plain HitterPure):
        self.rewards.post_strike_brake.weight = 1.0
        self.rewards.hold_ready.weight = 1.0
