"""Agibot (Zhiyuan) Expedition A3 — ping-pong configuration for BeyondMimic / HOPE WBC.

This file is written against the OFFICIAL Agibot A3 ping-pong assets shipped in the HOPE repo under
``agi/`` (the URDF ``agi/URDF/A3T2.5-URDF-std-pingpang/`` and the MuJoCo MJCF
``agi/A3_MuJoCo_Sim/.../a3_pingpong/a3_pingpong.xml`` — both are Agibot-provided, not stand-ins).
Names, link inertials, velocity limits, and the standing pose originate from those assets and the
legacy deploy package. Training actuator identity follows the latest vendor A3 training
configuration where it supersedes those older constants: in particular waist yaw Kp, waist pitch
effort, wrist pitch/yaw Kp and effort, and every 29-DoF armature value covered by the vendor table.
Head 40/2 remains the legacy deploy neck/head default (ExpandToBackend); the 2-DOF neck is not in
the 29-DOF policy view.

Nothing here touches the filesystem at import time: ``ArticulationCfg`` only stores the asset
path string, so the A3 task registers and imports fine *without* the asset present. The path is
only resolved when an environment is actually instantiated for training.

A3 active DOF (31, excluding hands): waist yaw/roll/pitch (3), neck yaw/pitch (2),
each arm 7 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), each leg 6
(hip pitch/roll/yaw, knee, ankle pitch/roll). The right arm holds the paddle.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

##
# Asset path — the OFFICIAL Agibot A3 ping-pong URDF (copied from agi/URDF/A3T2.5-URDF-std-pingpang/
# into assets/agibot_a3/, reimplement.md step 12). Isaac Lab spawns the URDF directly (as the G1
# config does with ``UrdfFileCfg``); for the MuJoCo/mjlab path use the official a3_pingpong.xml MJCF.
##
AGIBOT_A3_ASSET_ROOT = f"{ASSET_DIR}/agibot_a3"
AGIBOT_A3_URDF_PATH = f"{AGIBOT_A3_ASSET_ROOT}/urdf/model.urdf"  # official Agibot A3 ping-pong URDF

##
# Body / joint name constants (real names from the A3 ping-pong URDF). The rest of the HOPE
# code imports these so there is a single source of truth when the asset is swapped.
##
# NOTE the mixed casing — it is INTENTIONAL and matches the A3 URDF exactly: the root is
# "pelvis_link" (lowercase) while every other body uses "_Link" (capital L). MotionCommand does an
# exact-string lookup, so do not "normalize" these. Re-verify against the validated asset's link
# table when it arrives.
A3_ROOT_BODY = "pelvis_link"
A3_ANCHOR_BODY = "torso_Link"

# Bodies tracked by the BeyondMimic motion command (mirror of the G1 14-body set).
A3_TRACKED_BODIES = [
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
]

# Feet + hands; used for contact/termination exclusions.
A3_FEET_BODIES = ["left_ankle_roll_Link", "right_ankle_roll_Link"]
A3_HAND_BODIES = ["left_wrist_yaw_Link", "right_wrist_yaw_Link"]

# UPPER-body tracked bodies (torso + both arms) — the subset of A3_TRACKED_BODIES used by the
# footwork variant's imitation reward. The legs (pelvis/hip/knee/ankle) are intentionally EXCLUDED so
# the lower body is free to step/shift to reach different racket targets instead of copying the clip's
# fixed leg motion. Upper-body imitation still gives the swing its style.
A3_UPPER_TRACKED = [
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
]

# Joint order for reading the retargeted-motion CSV in scripts/csv_to_npz.py. This is the order of
# the *DOF columns* in the A3 retargeted CSV (columns 7: after base pos/quat), i.e. the order your
# GMR retargeting outputs — NOT the simulation articulation order (the npz stores joint_pos in the
# articulation order automatically). The default below follows the A3 controller_joint_names.yaml
# (agi/.../config/joint_names_*.yaml). IMPORTANT: if your GMR A3 retargeting emits a different
# column order, reorder this list to match it, or the npz joints will be scrambled.
AGIBOT_A3_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# Racket mount (right arm). See whole_body_tracking/tasks/tracking/mdp/hope_commands.py.
A3_WRIST_BODY = "right_wrist_yaw_Link"          # last actuated link of the paddle arm
A3_RACKET_BODY = "pingpang_red_Link"            # racket-center body (coincident with pingbang_ball_Link)
# Offset wrist_yaw -> canonical racket site, in the wrist_yaw local frame
# (metres).  This is the exact URDF pingpang_red_joint / MJCF right_racket site;
# right_hand_pingpang_joint is identity.
A3_MOUNT_OFFSET = (0.21021, 0.032078, 0.032036)


##
# Training actuator constants use the latest vendor A3 training configuration as authority. Values
# not superseded there retain the asset/deploy transcription. Action scale is always derived below as
# 0.25*effort/stiffness, so a vendor Kp or effort update cannot leave the policy decoder stale
# (target = action*action_scale + default_angle). Head 40/2 remains the legacy deploy default.
##
def _make_agibot_a3_spawn_cfg():
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        # Self-collision is OFF: in the official URDF the merged wrist body carries 4 overlapping
        # collision meshes (wrist + hand_pingpang + red/black blades, all coincident) with thin blade
        # hulls, which corrupts PhysX at sim start ("free(): corrupted unsorted chunks" -> Aborted).
        # WBC imitation does not need self-collision. NOTE: the official MJCF a3_pingpong.xml already
        # ships a clean collision setup (convex hulls + primitive racket/hand geoms + adjacent-body
        # <contact><exclude> list) — port that into the URDF to re-enable; it is NOT an Agibot blocker.
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )

    # A pre-converted USD bypasses the URDF importer entirely.  With no override, retain the
    # established URDF conversion path byte-for-byte at the configuration level.
    preconverted_usd_path = os.environ.get("HOPE_AGIBOT_A3_USD_PATH")
    if preconverted_usd_path:
        return sim_utils.UsdFileCfg(
            usd_path=preconverted_usd_path,
            activate_contact_sensors=True,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )

    return sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=AGIBOT_A3_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    )


AGIBOT_A3_CFG = ArticulationCfg(
    spawn=_make_agibot_a3_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        # Standing pose = a3.py ``default_angles`` (legacy Agibot deploy source,
        # a3_policy_parameters.hpp). This is used BOTH as the reset pose AND the action offset
        # (use_default_offset=True), so it must match the deploy action decoder exactly. Pelvis Z
        # 1.0684 m is the A3 MuJoCo stand-keyframe height for this (near-identical) leg pose; waist,
        # neck, shoulder_yaw and the wrists stay at 0.
        pos=(0.0, 0.0, 1.0684),
        joint_pos={
            ".*_hip_pitch_joint": -0.1311,
            ".*_knee_joint": 0.2468,
            ".*_ankle_pitch_joint": -0.1204,
            "left_hip_roll_joint": 0.0056,
            "right_hip_roll_joint": -0.0056,
            "left_hip_yaw_joint": -0.0348,
            "right_hip_yaw_joint": 0.0348,
            "left_ankle_roll_joint": -0.0078,
            "right_ankle_roll_joint": 0.0078,
            # arms — paddle-ready stance (right arm holds the racket)
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.12,
            "right_shoulder_roll_joint": -0.12,
            ".*_elbow_joint": 0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # ACTUATOR MODEL (Phase A, 2026-07-02): ALL groups are ImplicitActuatorCfg. AGI officially trains
    # with IsaacLab implicit PD and the real robot's backend is close to implicit PD; the earlier
    # IdealPDActuatorCfg (explicit) round was built on a falsified premise (implicit training already
    # clamps torque — effort_limit_sim is written into the PhysX drive max force; the "elbow 6.7x24Nm"
    # figure was the PRE-clip computed_effort) and IdealPD@200Hz added discrete-overshoot dynamics
    # (wrist kd*dt/I ~ 1.3-2.5) that empirically DEGRADED the backhand. Do not reintroduce IdealPD.
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim={
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_roll_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 320.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 12.0,
                ".*_hip_roll_joint": 12.0,
                ".*_hip_pitch_joint": 12.0,
                ".*_knee_joint": 14.6,
            },
            stiffness={  # latest vendor A3 training Kp
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 80.0,
                ".*_knee_joint": 250.0,
            },
            damping={  # latest vendor A3 training Kd
                ".*_hip_yaw_joint": 3.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_pitch_joint": 3.0,
                ".*_knee_joint": 8.0,
            },
            armature={  # latest vendor A3 training armature
                ".*_hip_yaw_joint": 0.066472,
                ".*_hip_roll_joint": 0.066472,
                ".*_hip_pitch_joint": 0.066472,
                ".*_knee_joint": 0.120340,
            },
            # IMPORTANT SEMANTICS (audited 2026-07-10): Isaac Lab 2.1 interprets ``friction`` as
            # a DIMENSIONLESS PhysX coefficient whose resisting force scales with transmitted
            # spatial force. These numeric values were copied from MuJoCo ``frictionloss`` (a
            # constant Coulomb torque in N m), so they are currently an UNCALIBRATED legacy
            # choice, not a unit-preserving port. Keep them immutable for existing checkpoint
            # lineage; schema-3 records the backend/semantics and formal MuJoCo evaluation fails
            # closed while any coefficient is non-zero. A future calibrated/zero-friction arm
            # must be a new training contract, never a silent resume.
            friction={
                ".*_hip_yaw_joint": 1.1971,
                ".*_hip_roll_joint": 1.1971,
                ".*_hip_pitch_joint": 1.1971,
                ".*_knee_joint": 2.4276,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={".*_ankle_pitch_joint": 118.2, ".*_ankle_roll_joint": 54.75},
            velocity_limit_sim={".*_ankle_pitch_joint": 10.8, ".*_ankle_roll_joint": 19.3},
            stiffness=50.0,  # latest vendor A3 training Kp (ankle)
            damping=2.0,     # latest vendor A3 training Kd (ankle)
            armature={".*_ankle_pitch_joint": 0.064449, ".*_ankle_roll_joint": 0.020129},
            friction={".*_ankle_pitch_joint": 1.4, ".*_ankle_roll_joint": 0.778},  # uncalibrated PhysX coeff; see legs
        ),
        # EXPLICIT PD (sim2real) — see the "feet" group note. effort_limit MUST be set (explicit-cfg
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim={"waist_yaw_joint": 220.0, "waist_roll_joint": 46.0, "waist_pitch_joint": 115.0},
            velocity_limit_sim={"waist_yaw_joint": 12.0, "waist_roll_joint": 22.7, "waist_pitch_joint": 9.2},
            stiffness={"waist_yaw_joint": 80.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0},
            damping={"waist_yaw_joint": 3.0, "waist_roll_joint": 2.0, "waist_pitch_joint": 2.0},
            armature={"waist_yaw_joint": 0.066472, "waist_roll_joint": 0.014623, "waist_pitch_joint": 0.088220},
            friction={"waist_yaw_joint": 1.1971, "waist_roll_joint": 0.69223, "waist_pitch_joint": 1.7},  # uncalibrated PhysX coeff
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=6.0,
            velocity_limit_sim=12.7,
            # neck/head kp=40, kd=2 from the deploy default (ExpandToBackend, A3 deploy example.md)
            stiffness=40.0,
            damping=2.0,
            armature={"head_yaw_joint": 0.0008100893338, "head_pitch_joint": 0.0008100893338},
            friction=0.1,  # uncalibrated PhysX coefficient (not 0.1 N m)
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_roll_joint": 24.0,
                ".*_wrist_pitch_joint": 24.0,
                ".*_wrist_yaw_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.6,
                ".*_shoulder_roll_joint": 13.6,
                ".*_shoulder_yaw_joint": 15.7,
                ".*_elbow_joint": 15.7,
                ".*_wrist_roll_joint": 15.7,
                ".*_wrist_pitch_joint": 12.7,
                ".*_wrist_yaw_joint": 12.7,
            },
            stiffness={  # latest vendor A3 training Kp
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 30.0,
                ".*_elbow_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
                ".*_wrist_pitch_joint": 30.0,
                ".*_wrist_yaw_joint": 30.0,
            },
            damping={  # latest vendor A3 training Kd
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_yaw_joint": 2.0,
            },
            armature={  # latest vendor A3 training armature
                ".*_shoulder_pitch_joint": 0.012085,
                ".*_shoulder_roll_joint": 0.012085,
                ".*_shoulder_yaw_joint": 0.004968,
                ".*_elbow_joint": 0.004968,
                ".*_wrist_roll_joint": 0.004968,
                ".*_wrist_pitch_joint": 0.004968,
                ".*_wrist_yaw_joint": 0.004968,
            },
            friction={  # uncalibrated PhysX coefficients copied numerically from MJCF frictionloss
                ".*_shoulder_pitch_joint": 0.6293,
                ".*_shoulder_roll_joint": 0.6293,
                ".*_shoulder_yaw_joint": 0.41197,
                ".*_elbow_joint": 0.41197,
                ".*_wrist_roll_joint": 0.41197,
                ".*_wrist_pitch_joint": 0.1,
                ".*_wrist_yaw_joint": 0.1,
            },
        ),
    },
)


# Per-joint action scale from the latest vendor training convention: 0.25 * base effort / base Kp.
# Domain-randomized gains do not alter this decoder scale.
AGIBOT_A3_ACTION_SCALE: dict[str, float] = {}
for _act in AGIBOT_A3_CFG.actuators.values():
    _eff = _act.effort_limit_sim
    _stiff = _act.stiffness
    _names = _act.joint_names_expr
    if not isinstance(_eff, dict):
        _eff = {n: _eff for n in _names}
    if not isinstance(_stiff, dict):
        _stiff = {n: _stiff for n in _names}
    for _n in _names:
        if _n in _eff and _n in _stiff and _stiff[_n]:
            AGIBOT_A3_ACTION_SCALE[_n] = 0.25 * _eff[_n] / _stiff[_n]
