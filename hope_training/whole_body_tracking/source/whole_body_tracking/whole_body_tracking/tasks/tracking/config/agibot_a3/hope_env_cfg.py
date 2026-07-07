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

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.agibot_a3 import A3_FEET_BODIES, A3_HAND_BODIES, A3_UPPER_TRACKED
from whole_body_tracking.tasks.tracking.config.agibot_a3.flat_env_cfg import AgibotA3FlatEnvCfg
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
    if getattr(env_cfg.scene, "shadow_ball", None) is not None:
        return  # already attached (cfg-flag path ran before the train.py override path)

    import yaml as _yaml

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    from whole_body_tracking.tasks.table_tennis import geometry as tt_geom
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
        env_cfg.scene.shadow_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ShadowTable",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(
                    near_x + tt_geom.TABLE_LENGTH / 2.0,
                    0.0,
                    surface_z - tt_geom.TABLE_THICKNESS / 2.0,
                )
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
                pos=(near_x + tt_geom.TABLE_LENGTH / 2.0, 0.0, surface_z - tt_geom.TABLE_HEIGHT),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(usd_path=tt_cfg._TABLE_USD_PATH),
        )


##
# PHYSICAL ball + table scene attachment — Phase A truth instrument (flag-gated, METRICS-ONLY;
# physical_ball.py). Distinct from the shadow-ball entities so both instruments can coexist.
##


def attach_physical_ball_scene(env_cfg) -> None:
    """Attach the physical-ball scene entities for ``physical_ball=True`` (Phase A).

    Adds to ``env_cfg.scene`` (per-env cloned assets via the ``{ENV_REGEX_NS}`` regex path, the
    table_tennis scene-builder pattern):

    * ``pb_ball`` — one dynamic sphere per env. Radius/mass from
      ``configs/ball_physics_venue.yaml`` (fitted coated match ball: R=0.02 m, m=3.4 g), zero
      linear/angular damping, gyroscopic forces OFF (omega constant in flight, the venue-fit
      assumption; the per-substep aero wrench supplies drag+Magnus — PhysX integrates gravity
      only; validated by scripts/isaac_ball_inloop_check.py, 17 mm systematic vs venue RK4).
      COLLIDER DISABLED + material restitution 0 (neutralized): PhysX never resolves the ball's
      contacts — the fitted CODE-DRIVEN table bounce is the single bounce authority, and the ball
      passes THROUGH the robot (one switch = the Phase A collision filter; the in-engine fitted
      racket impulse is PHASE B). NOTE: scene-level PhysX CCD (the only CCD knob in this Isaac
      Lab version, per table_tennis_env_cfg) is deliberately NOT enabled here — with the ball
      contactless it protects nothing and would touch the robot's solver path; Phase B enables it
      together with the collider.
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
    if getattr(env_cfg.scene, "pb_ball", None) is not None:
        return  # already attached (cfg-flag path ran before the train.py override path)

    import yaml as _yaml

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    from whole_body_tracking.tasks.table_tennis import geometry as tt_geom
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
            # Phase A collision filter: collider OFF -> no ball<->robot and no ball<->table PhysX
            # contact (the fitted code-driven bounce owns the table). Restitution 0 documents the
            # neutralization for Phase B, when the collider turns on for the racket impulse.
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
            pos=(
                near_x + tt_geom.TABLE_LENGTH / 2.0,
                0.0,
                surface_z - tt_geom.TABLE_THICKNESS / 2.0,
            )
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
            pos=(near_x + tt_geom.TABLE_LENGTH / 2.0, 0.0, surface_z - tt_geom.TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=tt_cfg._TABLE_USD_PATH),
    )


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
# Domain randomization (HITTER + standard sim-to-real reconstruction).
#
# HITTER publishes no DR table; it states PD gains are FIXED. The mass/friction/push/observation-noise
# terms below are standard BeyondMimic practice; PD-gain and motor-strength randomization are added for
# sim-to-real robustness and can be disabled to match HITTER exactly.
#
# Already provided by the base EventCfg: friction (physics_material, startup), CoM (startup),
# joint default pos (startup), external pushes (push_robot, interval). Observation noise comes from the
# per-term Unoise + enable_corruption on the policy observation group.
##


@configclass
class HOPEEventCfg(EventCfg):
    # HITTER alignment: no external push. HITTER's prose DR is mass/friction/restitution + perception
    # noise/delays only — there is no random shove. Keep friction (physics_material) and CoM (base_com)
    # from the base EventCfg; disable the base interval push.
    push_robot = None

    # link mass randomization (±10%) — HITTER prose randomizes link mass.
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
    # PD gain / motor strength randomization (±20%). NOTE: HITTER keeps PD fixed; this is a
    # sim-to-real robustness choice. Set to None to disable.
    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
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
    # REWARD-side only — the frozen 175-D actor obs contract is untouched. weight 0.0 = OFF (IsaacLab's
    # RewardManager skips zero-weight terms); enable per-experiment via task.rewards.base_decel_weight
    # (suggested trial 1.0). CLI/yaml-tunable: base_decel_weight / _v_gain / _v_max / _std.
    # Watch metric: base_speed_xy_prestrike (should taper near targets instead of staying hot).
    base_decel = RewTerm(
        func=mdp.base_decel_tracking, weight=0.0,
        params={"command_name": "racket_target", "v_gain": 2.0, "v_max": 1.6, "std": 0.4})

    # --- strike-window stability: be planted + upright + still AT the hit (gated to the strike window) ---
    strike_upright = RewTerm(func=mdp.strike_proj_grav_xy, weight=-2.0, params={"command_name": "racket_target"})
    strike_ang_vel = RewTerm(func=mdp.strike_base_ang_vel, weight=-0.5, params={"command_name": "racket_target"})
    strike_foot_vel = RewTerm(func=mdp.strike_foot_velocity, weight=-0.5, params={"command_name": "racket_target"})
    strike_vbob = RewTerm(func=mdp.strike_vertical_bob, weight=-1.0, params={"command_name": "racket_target"})

    # --- SIM2REAL FINE-TUNE (2026-07-02): survive AGI's EXPLICIT clipped-PD MuJoCo. ------------------
    # CHANGE 2 — torque-saturation penalty: penalize the mean over-limit fraction of the COMPUTED (pre-clip)
    # effort over the arm + waist joints so the policy stops demanding torque the explicit motor cannot
    # deliver (the elbow was at ~6.7x its 24 Nm limit in the failing trace). Modest weight to protect the
    # strike. CLI-tunable via task.rewards.arm_torque_saturation_weight. Watch metric: arm_torque_sat_frac.
    arm_torque_saturation = RewTerm(
        func=mdp.arm_torque_saturation, weight=-0.5, params={"command_name": "racket_target"})
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
    """Inherited reference-relative terminations + ABSOLUTE balance terminations, so a real fall/sink
    ends the episode regardless of the reference clip (the actual deploy failure mode)."""

    base_fell_tilt = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})  # ~40 deg, absolute
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})


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
    # PHYSICAL ball + table truth instrument (Phase A; DEFAULT OFF = byte-identical scene/env).
    # True -> __post_init__ sets commands.racket_target.physical_ball and attaches the pb_ball /
    # pb_table / pb_table_visual scene entities (attach_physical_ball_scene). The train.py
    # task.physical_ball override runs AFTER __post_init__ (face_command_obs timing), so it calls
    # the (idempotent) attach itself. METRICS-ONLY — see physical_ball.py; racket impulse=Phase B.
    physical_ball: bool = False
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
