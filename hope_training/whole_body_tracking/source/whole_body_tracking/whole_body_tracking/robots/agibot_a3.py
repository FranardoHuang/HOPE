"""Agibot (Zhiyuan) Expedition A3 — ping-pong configuration for BeyondMimic / HOPE WBC.

This file is written against the A3 ping-pong URDF that already ships in the HOPE repo at
``agi/0000014503_A3T2.5-URDF-std-pingpang-0519/`` (joint/link names and effort/velocity
limits are taken verbatim from its ``urdf/joints.txt``). The *names* are real; what is still
a PLACEHOLDER is the validated simulation asset (USD/MJCF), the verified link inertials, and
the official PD gains — all of which must come from Agibot (see reimplement.md step 2.1).

Nothing here touches the filesystem at import time: ``ArticulationCfg`` only stores the asset
path string, so the A3 task registers and imports fine *without* the asset present. The path is
only resolved when an environment is actually instantiated for training.

A3 active DOF (31, excluding hands): waist yaw/roll/pitch (3), neck yaw/pitch (2),
each arm 7 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), each leg 6
(hip pitch/roll/yaw, knee, ankle pitch/roll). The right arm holds the paddle.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

##
# Asset path (PLACEHOLDER until the validated Agibot A3 asset is dropped in).
#
# reimplement.md step 12 copies the A3 model into
# ``source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/``. Point this at the
# robot model file once it exists. Isaac Lab can spawn a URDF directly (as the G1 config does
# with ``UrdfFileCfg``); for the MuJoCo/mjlab path use the MJCF instead.
##
AGIBOT_A3_ASSET_ROOT = f"{ASSET_DIR}/agibot_a3"
AGIBOT_A3_URDF_PATH = f"{AGIBOT_A3_ASSET_ROOT}/urdf/model.urdf"  # TODO(asset): set to the real A3 ping-pong URDF/USD

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
# Offset wrist_yaw -> racket center, in the wrist_yaw local frame (meters). From the URDF
# pingbang_ball_joint origin; right_hand_pingpang_joint is xyz=0 rpy=0 so this equals the
# offset from right_wrist_yaw_Link directly.
A3_MOUNT_OFFSET = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)


##
# Heuristic PD gains. THESE ARE PLACEHOLDERS — Agibot must confirm the real impedance settings
# (reimplement.md step 2.1). Effort/velocity limits below ARE the real values from joints.txt.
# Stiffness/damping are reasonable starting points for a ~55 kg humanoid and should be replaced.
##
AGIBOT_A3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=AGIBOT_A3_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collision is OFF: the merged wrist body carries 4 overlapping collision meshes
            # (wrist + hand_pingpang + red/black blades, all coincident) with thin blade hulls, which
            # corrupts PhysX at sim start ("free(): corrupted unsorted chunks" -> Aborted) on this
            # placeholder asset. WBC imitation does not need self-collision; re-enable once a validated
            # A3 asset with clean collision geometry is dropped in (reimplement.md step 2.1).
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # TODO(asset): verify standing pelvis height against the real A3 model (≈0.93 m for a 1.73 m robot).
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            # legs — slight crouch ready stance
            ".*_hip_pitch_joint": -0.30,
            ".*_knee_joint": 0.60,
            ".*_ankle_pitch_joint": -0.30,
            # arms — paddle-ready stance (right arm holds the racket)
            "left_shoulder_pitch_joint": 0.25,
            "left_shoulder_roll_joint": 0.20,
            "left_elbow_joint": 0.80,
            "right_shoulder_pitch_joint": 0.25,
            "right_shoulder_roll_joint": -0.20,
            "right_elbow_joint": 0.80,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
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
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={".*_ankle_pitch_joint": 118.0, ".*_ankle_roll_joint": 54.0},
            velocity_limit_sim={".*_ankle_pitch_joint": 10.8, ".*_ankle_roll_joint": 19.3},
            stiffness=40.0,
            damping=2.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim={"waist_yaw_joint": 220.0, "waist_roll_joint": 46.0, "waist_pitch_joint": 118.0},
            velocity_limit_sim={"waist_yaw_joint": 12.0, "waist_roll_joint": 22.7, "waist_pitch_joint": 9.2},
            stiffness=200.0,
            damping=5.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=6.0,
            velocity_limit_sim=12.7,
            stiffness=10.0,
            damping=0.5,
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
                ".*_wrist_pitch_joint": 6.0,
                ".*_wrist_yaw_joint": 6.0,
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
            stiffness={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_roll_joint": 20.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 1.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_yaw_joint": 1.0,
            },
        ),
    },
)


# Per-joint action scale, computed like the G1 config: 0.25 * effort_limit / stiffness.
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
