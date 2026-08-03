from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class ActorObservationTerm:
    name: str
    dim: int
    deploy_source: str
    description: str


@dataclass(frozen=True)
class ActorObservationContract:
    name: str
    obs_mode: str
    total_dim: int
    terms: tuple[ActorObservationTerm, ...]
    legacy_names: tuple[str, ...] = ()

    @property
    def layout(self) -> tuple[tuple[str, int], ...]:
        return tuple((term.name, term.dim) for term in self.terms)


FULL = ActorObservationContract(
    name="full",
    obs_mode="full",
    total_dim=180,
    terms=(
        ActorObservationTerm("command", 62, "reference_clip", "reference joint positions and velocities"),
        ActorObservationTerm(
            "motion_anchor_pos_b",
            3,
            "sim_or_localizer",
            "reference torso position error; requires world base position",
        ),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "base_target_pos_b",
            2,
            "planner_plus_localizer",
            "desired base XY target relative to the current base",
        ),
        ActorObservationTerm(
            "racket_target_pos_b",
            3,
            "planner_plus_localizer",
            "desired racket position relative to the current base",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
        ActorObservationTerm("swing_type", 1, "reference_clip", "forehand (+1) or backhand (-1) flag"),
    ),
)

DEPLOY_PARITY = ActorObservationContract(
    name="deploy_parity",
    obs_mode="deploy_parity",
    total_dim=175,
    legacy_names=("real_sensor_only",),
    terms=(
        ActorObservationTerm("command", 62, "reference_clip", "reference joint positions and velocities"),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "racket_target_pos_b",
            3,
            "planner_plus_racket_fk",
            "desired racket position relative to the current racket FK; no world base position",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
        ActorObservationTerm("swing_type", 1, "reference_clip", "forehand (+1) or backhand (-1) flag"),
    ),
)

HITTER_FOOTWORK = ActorObservationContract(
    name="hitter_footwork",
    obs_mode="hitter_footwork",
    total_dim=177,
    terms=(
        ActorObservationTerm("command", 62, "reference_clip", "reference joint positions and velocities"),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "base_target_pos_b",
            2,
            "planner_plus_mocap",
            "desired base XY station relative to the current base (yaw-heading frame); mocap base "
            "position at deploy — relative Δ only, Δ=0 graceful fallback on mocap dropout",
        ),
        ActorObservationTerm(
            "racket_target_pos_b",
            3,
            "planner_plus_racket_fk",
            "desired racket position relative to the current racket FK; no world base position",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
        ActorObservationTerm("swing_type", 1, "reference_clip", "forehand (+1) or backhand (-1) flag"),
    ),
)


STAGE1_NATURAL_CLIP_SITE_V1 = ActorObservationContract(
    name="stage1_natural_clip_site_v1",
    obs_mode="stage1_natural_clip",
    total_dim=170,
    terms=(
        ActorObservationTerm(
            "command",
            62,
            "reference_clip",
            "current-phase natural-clip joint positions and velocities",
        ),
        ActorObservationTerm(
            "motion_anchor_pos_b",
            3,
            "sim_or_localizer",
            "reference torso position error in the base frame",
        ),
        ActorObservationTerm(
            "motion_anchor_ori_b",
            6,
            "imu_plus_reference_clip",
            "reference torso orientation error",
        ),
        ActorObservationTerm(
            "base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"
        ),
        ActorObservationTerm(
            "joint_pos", 31, "encoders", "joint position offset from default"
        ),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm(
            "actions", 31, "runtime_state", "previous policy action"
        ),
        ActorObservationTerm(
            "projected_gravity", 3, "imu", "gravity direction in the base frame"
        ),
    ),
)


# Fresh, intentionally warm-start-breaking successor to ``stage1_natural_clip_site_v1``.
# The order is semantic rather than historical: each achieved value is adjacent to the teacher
# value it is meant to track, and the current-state paddle pair is adjacent to the future-contact
# paddle pair.  All absolute base quantities use the canonical HOPE world frame; all four racket
# blocks use the current actual base position as origin and the current actual base yaw-heading as
# axes.  This fixed-width physical preview replaces categorical action identity.
STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2 = ActorObservationContract(
    name="stage1_natural_clip_paddle_world_v2",
    obs_mode="stage1_natural_clip_paddle_world",
    total_dim=225,
    terms=(
        ActorObservationTerm(
            "actual_base_now_world",
            15,
            "optitrack_plus_imu",
            "current base position(3), orientation-6D(6), world linear velocity(3), "
            "and world angular velocity(3) in canonical HOPE world",
        ),
        ActorObservationTerm(
            "teacher_base_now_world",
            15,
            "aligned_reference_clip",
            "current aligned teacher base position/orientation-6D/linear velocity/angular "
            "velocity in canonical HOPE world",
        ),
        ActorObservationTerm(
            "joint_pos",
            31,
            "encoders",
            "current joint position minus the nominal default position",
        ),
        ActorObservationTerm(
            "teacher_joint_pos",
            31,
            "reference_clip",
            "current teacher joint position minus the same nominal default position",
        ),
        ActorObservationTerm(
            "joint_vel",
            31,
            "encoders",
            "current joint velocity in actor joint order",
        ),
        ActorObservationTerm(
            "teacher_joint_vel",
            31,
            "reference_clip",
            "current teacher joint velocity in the same actor joint order",
        ),
        ActorObservationTerm(
            "actions",
            31,
            "runtime_state",
            "previous normalized policy action before the episode-fixed actuator delay",
        ),
        ActorObservationTerm(
            "racket_site_achieved_now_heading",
            9,
            "encoder_fk_plus_base_state",
            "actual official-site position(3), absolute linear velocity(3), and signed "
            "face normal(3) expressed in the current base yaw-heading frame",
        ),
        ActorObservationTerm(
            "racket_site_teacher_now_heading",
            9,
            "aligned_reference_clip",
            "current-phase teacher official-site position/velocity/signed-normal in the "
            "same current base yaw-heading frame",
        ),
        ActorObservationTerm(
            "racket_site_teacher_at_reference_hit_heading",
            9,
            "aligned_reference_clip_strike_landmark",
            "selected clip's nominal official-site position/velocity/signed-normal at its "
            "reference hit, in the same current base yaw-heading frame",
        ),
        ActorObservationTerm(
            "racket_contact_desired_at_t_hit_heading",
            9,
            "planner_contact_demand",
            "desired contact official-site position/velocity/signed-normal at time-to-contact, "
            "in the same current base yaw-heading frame; Stage-1 copies the teacher hit tuple",
        ),
        ActorObservationTerm(
            "desired_base_xy_world",
            2,
            "planner_plus_table_calibration",
            "desired base XY position in canonical HOPE world",
        ),
        ActorObservationTerm(
            "time_to_contact",
            1,
            "task_clock",
            "signed seconds remaining to the contact deadline",
        ),
        ActorObservationTerm(
            "time_to_teacher_start",
            1,
            "motion_phase_governor",
            "seconds until teacher playback leaves its ready hold",
        ),
    ),
)


# Tonight's fixed-midpoint N1 comparison keeps the historical 225-D robot,
# teacher, achieved-paddle, base-station, and clock ABI.  A and C are separate
# contracts because columns [212:221] have different physical meanings and
# must never be silently reinterpreted by a checkpoint or rollout receipt.
_ACTION_BALL_225_COMMON_PREFIX = STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[:10]
_ACTION_BALL_225_COMMON_SUFFIX = STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[-3:]

ACTION_BALL_A225 = ActorObservationContract(
    name="action_ball_a225",
    obs_mode="action_ball_a225",
    total_dim=225,
    terms=_ACTION_BALL_225_COMMON_PREFIX
    + (
        ActorObservationTerm(
            "task_desired_contact_position_heading",
            3,
            "action_ball_a225_atomic_desired_contact_snapshot",
            "task-derived desired official-site position at contact relative to the "
            "current base origin in current base yaw-heading axes",
        ),
        ActorObservationTerm(
            "task_desired_contact_velocity_heading",
            3,
            "action_ball_a225_atomic_desired_contact_snapshot",
            "task-derived desired official-site linear velocity at contact in current "
            "base yaw-heading axes",
        ),
        ActorObservationTerm(
            "task_desired_contact_face_heading",
            3,
            "action_ball_a225_atomic_desired_contact_snapshot",
            "task-derived desired signed official-site face normal at contact in current "
            "base yaw-heading axes",
        ),
    )
    + _ACTION_BALL_225_COMMON_SUFFIX,
)

ACTION_BALL_C225 = ActorObservationContract(
    name="action_ball_c225",
    obs_mode="action_ball_c225",
    total_dim=225,
    terms=_ACTION_BALL_225_COMMON_PREFIX
    + (
        ActorObservationTerm(
            "incoming_ball_contact_position_heading",
            3,
            "action_ball_c225_atomic_causal_question_snapshot",
            "incoming ball-centre position at contact relative to the current base "
            "origin in current base yaw-heading axes",
        ),
        ActorObservationTerm(
            "incoming_ball_contact_velocity_heading",
            3,
            "action_ball_c225_atomic_causal_question_snapshot",
            "incoming ball linear velocity at contact in current base yaw-heading axes",
        ),
        ActorObservationTerm(
            "incoming_ball_contact_spin_heading",
            3,
            "action_ball_c225_atomic_causal_question_snapshot",
            "incoming ball spin at contact in current base yaw-heading axes",
        ),
    )
    + _ACTION_BALL_225_COMMON_SUFFIX,
)


# Fresh public successor to the historical A225/C225 diagnostic ABI.  The
# actor-side teacher-base row is removed and the actual base row is split by
# hardware source: world pose/linear velocity come from localization while the
# only actor angular-velocity copy is the pelvis/body-frame IMU gyro.  Robot,
# teacher, task, and runtime groups are contiguous.  ``task_valid`` is last so
# WAIT is an explicit all-zero task boundary and TASK_ACTIVE becomes valid
# atomically without reinterpreting any preceding column.
_ACTION_BALL_211_COMMON_PREFIX = (
    ActorObservationTerm(
        "actual_base_pose_lin_vel_world",
        12,
        "optitrack_plus_fused_root_velocity",
        "actual base position3, orientation6D, and linear velocity3 in canonical "
        "HOPE world; angular velocity is not packed in this localizer row",
    ),
    ActorObservationTerm(
        "base_ang_vel_body",
        3,
        "imu_gyro",
        "bias-corrected pelvis angular velocity in the calibrated robot body frame",
    ),
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[2],  # joint_pos
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[4],  # joint_vel
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[6],  # actions
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[7],  # achieved racket
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[3],  # teacher_joint_pos
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[5],  # teacher_joint_vel
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[8],  # teacher racket now
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[9],  # teacher racket at hit
)
_ACTION_BALL_211_COMMON_SUFFIX = STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.terms[-3:]
_ACTION_BALL_211_VALIDITY = (
    ActorObservationTerm(
        "task_valid",
        1,
        "action_ball_task_active_atomic_mask",
        "one exactly when the task, ball question, base goal, and both task "
        "clocks are atomically valid; zero throughout WAIT",
    ),
)

ACTION_BALL_A211 = ActorObservationContract(
    name="action_ball_a211",
    obs_mode="action_ball_a211",
    total_dim=211,
    terms=_ACTION_BALL_211_COMMON_PREFIX
    + (
        ActorObservationTerm(
            "task_desired_contact_position_heading",
            3,
            "action_ball_a211_atomic_desired_contact_snapshot",
            "task-derived desired official-site position at contact relative to the "
            "current base origin in current base yaw-heading axes",
        ),
        ActorObservationTerm(
            "task_desired_contact_velocity_heading",
            3,
            "action_ball_a211_atomic_desired_contact_snapshot",
            "task-derived desired official-site linear velocity at contact in current "
            "base yaw-heading axes",
        ),
        ActorObservationTerm(
            "task_desired_contact_face_heading",
            3,
            "action_ball_a211_atomic_desired_contact_snapshot",
            "task-derived desired signed official-site face normal at contact in current "
            "base yaw-heading axes",
        ),
    )
    + _ACTION_BALL_211_COMMON_SUFFIX
    + _ACTION_BALL_211_VALIDITY,
)

ACTION_BALL_C211 = ActorObservationContract(
    name="action_ball_c211",
    obs_mode="action_ball_c211",
    total_dim=211,
    terms=_ACTION_BALL_211_COMMON_PREFIX
    + (
        ActorObservationTerm(
            "incoming_ball_contact_position_heading",
            3,
            "action_ball_c211_atomic_causal_question_snapshot",
            "incoming ball-centre position at contact relative to the current base "
            "origin in current base yaw-heading axes",
        ),
        ActorObservationTerm(
            "incoming_ball_contact_velocity_heading",
            3,
            "action_ball_c211_atomic_causal_question_snapshot",
            "incoming ball linear velocity at contact in current base yaw-heading axes",
        ),
        ActorObservationTerm(
            "incoming_ball_contact_spin_heading",
            3,
            "action_ball_c211_atomic_causal_question_snapshot",
            "incoming ball spin at contact in current base yaw-heading axes",
        ),
    )
    + _ACTION_BALL_211_COMMON_SUFFIX
    + _ACTION_BALL_211_VALIDITY,
)


def task_first_n_contract(action_count: int) -> ActorObservationContract:
    """Build the task-first actor layout for one exact local action-bank size.

    The four demanded-face dimensions and the N-way categorical action identity
    are appended to ``hitter_footwork``.  Keeping the N in the contract name
    makes a changed action bank a shape/contract change instead of a silent
    reinterpretation of an existing checkpoint.
    """

    count = int(action_count)
    if count <= 0 or count > 1024:
        raise ValueError(f"task-first action_count must be in [1,1024], got {action_count!r}")
    return ActorObservationContract(
        name=f"task_first_n{count}",
        obs_mode="hitter_footwork",
        total_dim=HITTER_FOOTWORK.total_dim + 4 + count,
        terms=HITTER_FOOTWORK.terms
        + (
            ActorObservationTerm(
                "racket_target_normal_cmd",
                4,
                "planner",
                "demanded racket face normal (world, 3) + reserved spin-rho scalar",
            ),
            ActorObservationTerm(
                "action_one_hot",
                count,
                "action_catalog",
                "categorical local action slot; stable action_uid is resolved through the "
                "catalog and is never treated as a numeric observation",
            ),
        ),
    )


def action_ball_n_contract(action_count: int) -> ActorObservationContract:
    """Build the action-conditioned ball-first layout for an exact action-bank size.

    The observation columns intentionally match ``task_first_n<N>`` so the two
    training producers can share observation assembly.  Their contract names
    remain distinct because producer semantics are part of checkpoint identity,
    even when the tensor shape and ordered terms happen to be identical.
    """

    if type(action_count) is not int or not 1 <= action_count <= 1024:
        raise ValueError(
            "action-ball action_count must be a plain integer in [1,1024], "
            f"got {action_count!r}"
        )
    layout = task_first_n_contract(action_count)
    return ActorObservationContract(
        name=f"action_ball_n{action_count}",
        obs_mode=layout.obs_mode,
        total_dim=layout.total_dim,
        terms=layout.terms,
    )


def action_ball_table_pose_n_contract(
    action_count: int,
) -> ActorObservationContract:
    """Build the table-aware ActionBall layout for an exact action-bank size.

    The existing Hitter-footwork task channels remain relative.  A separate
    table-pose block gives the actor its absolute 6-DoF placement with respect
    to the calibrated table: position (3) plus the continuous first-two-column
    rotation representation (6).  Keeping those concerns separate lets one
    policy generalize relative strike tasks without losing the information
    needed to avoid or reposition around the table.
    """

    if type(action_count) is not int or not 1 <= action_count <= 1024:
        raise ValueError(
            "action-ball table-pose action_count must be a plain integer in "
            f"[1,1024], got {action_count!r}"
        )
    count = action_count
    # The tensor layout remains the proven HITTER 177-D prefix, but this
    # contract deliberately replaces the old IMU pose authority.  At deploy,
    # every base-attitude-derived actor term comes from the same calibrated
    # mocap SE(3) history (plus measured-joint FK where needed).  An IMU may
    # remain outside the actor as an independent safety monitor.
    mocap_overrides = {
        "motion_anchor_ori_b": (
            "mocap_plus_joint_fk_plus_reference_clip",
            "reference torso orientation error; current torso orientation "
            "comes from calibrated mocap base pose plus measured waist FK",
        ),
        "base_ang_vel": (
            "mocap_pose_history",
            "pelvis angular velocity from a causal SO(3) difference/filter "
            "of calibrated mocap base orientation",
        ),
        "projected_gravity": (
            "mocap",
            "gravity direction in the base frame from calibrated mocap base "
            "orientation",
        ),
        "racket_target_pos_b": (
            "planner_plus_mocap_plus_racket_fk",
            "desired racket position relative to current racket FK in the "
            "mocap/table-aligned base heading frame",
        ),
    }
    table_pose_prefix = tuple(
        ActorObservationTerm(
            term.name,
            term.dim,
            *mocap_overrides.get(
                term.name, (term.deploy_source, term.description)
            ),
        )
        for term in HITTER_FOOTWORK.terms
    )
    return ActorObservationContract(
        name=f"action_ball_table_pose_n{count}",
        obs_mode=HITTER_FOOTWORK.obs_mode,
        total_dim=HITTER_FOOTWORK.total_dim + 3 + 6 + 4 + count,
        terms=table_pose_prefix
        + (
            ActorObservationTerm(
                "base_position_table",
                3,
                "mocap_plus_table_calibration",
                "base root position relative to the calibrated table-surface center",
            ),
            ActorObservationTerm(
                "base_orientation_table_6d",
                6,
                "mocap_plus_table_calibration",
                "base orientation in the table/world frame as the first two "
                "rotation-matrix columns",
            ),
            ActorObservationTerm(
                "racket_target_normal_cmd",
                4,
                "planner",
                "demanded racket face normal (world, 3) + reserved spin-rho scalar",
            ),
            ActorObservationTerm(
                "action_one_hot",
                count,
                "action_catalog",
                "categorical local action slot; stable action_uid is resolved through the "
                "catalog and is never treated as a numeric observation",
            ),
        ),
    )


def action_ball_table_pose_twist_n_contract(
    action_count: int,
) -> ActorObservationContract:
    """Build the preferred table-pose-and-twist ActionBall layout.

    This keeps the complete 177-D Hitter-footwork prefix and the relative task
    channels, then appends table-relative base position (3), continuous base
    orientation (6), and yaw-heading-frame base linear velocity (3) before the
    face-command/action-identity tail.  The explicit contract name prevents a
    191-D table-pose checkpoint from being silently reinterpreted as this
    ``193 + N`` layout.
    """

    if type(action_count) is not int or not 1 <= action_count <= 1024:
        raise ValueError(
            "action-ball table-pose-twist action_count must be a plain integer "
            f"in [1,1024], got {action_count!r}"
        )
    count = action_count
    deploy_overrides = {
        "motion_anchor_ori_b": (
            "optitrack_plus_joint_fk_plus_reference_clip",
            "reference torso orientation error; current torso orientation "
            "comes from calibrated OptiTrack base pose plus measured waist FK",
        ),
        "base_ang_vel": (
            "imu_gyro",
            "pelvis angular velocity in rad/s from the bias-corrected three-axis "
            "IMU gyroscope after the calibrated sensor-to-base rotation",
        ),
        "projected_gravity": (
            "optitrack",
            "gravity direction in the base frame from calibrated OptiTrack "
            "base orientation",
        ),
        "base_target_pos_b": (
            "planner_plus_optitrack",
            "desired base XY station relative to the current OptiTrack base "
            "position in the yaw-heading frame",
        ),
        "racket_target_pos_b": (
            "planner_plus_optitrack_plus_racket_fk",
            "desired racket position relative to current racket FK in the "
            "OptiTrack/table-aligned base heading frame",
        ),
    }
    deploy_prefix = tuple(
        ActorObservationTerm(
            term.name,
            term.dim,
            *deploy_overrides.get(
                term.name, (term.deploy_source, term.description)
            ),
        )
        for term in HITTER_FOOTWORK.terms
    )
    return ActorObservationContract(
        name=f"action_ball_table_pose_twist_n{count}",
        obs_mode=HITTER_FOOTWORK.obs_mode,
        total_dim=HITTER_FOOTWORK.total_dim + 3 + 6 + 3 + 4 + count,
        terms=deploy_prefix
        + (
            ActorObservationTerm(
                "base_position_table",
                3,
                "optitrack_plus_table_calibration",
                "base root position relative to the calibrated table-surface center",
            ),
            ActorObservationTerm(
                "base_orientation_table_6d",
                6,
                "optitrack_plus_table_calibration",
                "base orientation in the table/world frame as the first two "
                "rotation-matrix columns",
            ),
            ActorObservationTerm(
                "base_lin_vel_heading",
                3,
                "fused_root_com_velocity_estimator",
                "yaw-heading-frame root-rigid-body COM linear velocity from a causal "
                "fused estimator using OptiTrack position as the absolute anchor, "
                "the calibrated marker-to-root/COM offset, and optional IMU "
                "accelerometer propagation",
            ),
            ActorObservationTerm(
                "racket_target_normal_cmd",
                4,
                "planner",
                "demanded racket face normal (world, 3) + reserved spin-rho scalar",
            ),
            ActorObservationTerm(
                "action_one_hot",
                count,
                "action_catalog",
                "categorical local action slot; stable action_uid is resolved through the "
                "catalog and is never treated as a numeric observation",
            ),
        ),
    )


def action_ball_table_pose_twist_heading_task_n_contract(
    action_count: int,
) -> ActorObservationContract:
    """Build the frame-consistent table-pose/twist ActionBall layout.

    This is the preferred successor to
    :func:`action_ball_table_pose_twist_n_contract`.  Width and broad grouping
    stay ``193 + N``, but all three actor-visible racket-task vectors are now
    represented in one base yaw-heading frame:

    * target-minus-current-racket position;
    * demanded racket velocity;
    * demanded raw-A face normal.

    The canonical planner wire, solver, reward and physics remain in the
    table/world frame.  A new contract name is mandatory because an old 194-D
    checkpoint has the same shape but different velocity/normal semantics.
    """

    if type(action_count) is not int or not 1 <= action_count <= 1024:
        raise ValueError(
            "action-ball table-pose-twist-heading-task action_count must be a "
            f"plain integer in [1,1024], got {action_count!r}"
        )
    count = action_count
    deploy_overrides = {
        "motion_anchor_ori_b": (
            "optitrack_plus_joint_fk_plus_reference_clip",
            "reference torso orientation error; current torso orientation "
            "comes from calibrated OptiTrack base pose plus measured waist FK",
        ),
        "base_ang_vel": (
            "imu_gyro",
            "pelvis angular velocity in rad/s from the bias-corrected three-axis "
            "IMU gyroscope after the calibrated sensor-to-base rotation",
        ),
        "projected_gravity": (
            "optitrack",
            "gravity direction in the base frame from calibrated OptiTrack "
            "base orientation",
        ),
        "base_target_pos_b": (
            "planner_plus_optitrack",
            "desired base XY station relative to the current OptiTrack base "
            "position in the yaw-heading frame",
        ),
        "racket_target_pos_b": (
            "planner_plus_optitrack_plus_racket_fk",
            "desired racket position relative to current racket FK in the "
            "OptiTrack/table-aligned base heading frame",
        ),
    }
    deploy_prefix = []
    for term in HITTER_FOOTWORK.terms:
        if term.name == "racket_target_vel_w":
            deploy_prefix.append(
                ActorObservationTerm(
                    "racket_target_vel_heading",
                    3,
                    "planner_plus_optitrack",
                    "actor-visible demanded racket velocity rotated from the "
                    "canonical table/world frame into the base yaw-heading frame",
                )
            )
            continue
        deploy_prefix.append(
            ActorObservationTerm(
                term.name,
                term.dim,
                *deploy_overrides.get(
                    term.name, (term.deploy_source, term.description)
                ),
            )
        )
    return ActorObservationContract(
        name=f"action_ball_table_pose_twist_heading_task_n{count}",
        obs_mode=HITTER_FOOTWORK.obs_mode,
        total_dim=HITTER_FOOTWORK.total_dim + 3 + 6 + 3 + 4 + count,
        terms=tuple(deploy_prefix)
        + (
            ActorObservationTerm(
                "base_position_table",
                3,
                "optitrack_plus_table_calibration",
                "base root position relative to the calibrated table-surface center",
            ),
            ActorObservationTerm(
                "base_orientation_table_6d",
                6,
                "optitrack_plus_table_calibration",
                "base orientation in the table/world frame as the first two "
                "rotation-matrix columns",
            ),
            ActorObservationTerm(
                "base_lin_vel_heading",
                3,
                "fused_root_com_velocity_estimator",
                "yaw-heading-frame root-rigid-body COM linear velocity from a causal "
                "fused estimator using OptiTrack position as the absolute anchor, "
                "the calibrated marker-to-root/COM offset, and optional IMU "
                "accelerometer propagation",
            ),
            ActorObservationTerm(
                "racket_target_normal_cmd_heading",
                4,
                "planner_plus_optitrack",
                "demanded raw-A racket face normal rotated into the base "
                "yaw-heading frame (3) + reserved spin-rho scalar",
            ),
            ActorObservationTerm(
                "action_one_hot",
                count,
                "action_catalog",
                "categorical local action slot; stable action_uid is resolved through the "
                "catalog and is never treated as a numeric observation",
            ),
        ),
    )


def action_ball_table_pose_twist_heading_task_teacher_start_n_contract(
    action_count: int,
) -> ActorObservationContract:
    """Historical N-dependent teacher-start layout with categorical identity.

    The new scalar is inserted immediately before the final action one-hot so
    the categorical identity remains the final ``N`` columns.  This is a fresh
    contract; the old layout remains readable for historical checkpoints and
    receipts but is never silently warm-started under shifted offsets.
    """

    base = action_ball_table_pose_twist_heading_task_n_contract(action_count)
    return ActorObservationContract(
        name=(
            "action_ball_table_pose_twist_heading_task_teacher_start_n"
            f"{action_count}"
        ),
        obs_mode=base.obs_mode,
        total_dim=base.total_dim + 1,
        terms=base.terms[:-1]
        + (
            ActorObservationTerm(
                "time_to_teacher_start_s",
                1,
                "action_ball_motion_phase_governor",
                "live seconds until the frozen selected action's teacher "
                "leaves its ready frame",
            ),
            base.terms[-1],
        ),
    )


def action_ball_table_pose_twist_heading_task_teacher_start_v2_contract(
) -> ActorObservationContract:
    """Build the fixed-width ActionBall layout without a categorical action ID.

    Action identity remains frozen and audited in the sampler/solver/runtime
    control plane, but it is not an actor observation.  The policy therefore
    sees the same 194-D tensor shape for every action-bank size instead of an
    N-wide slot code that has no motion-similarity geometry.  The teacher
    trajectory itself carries the professional stroke content; no motion ID or
    synthetic intent code is needed.  This remains the N=1 launch contract only
    because the final ball/task/validity/history ABI has not yet been
    implemented.  N2/N3 may later diagnose a failed N73 run, but they are not a
    promotion prerequisite and do not justify adding an identity feature.
    """

    historical = (
        action_ball_table_pose_twist_heading_task_n_contract(1)
    )
    return ActorObservationContract(
        name=(
            "action_ball_table_pose_twist_heading_task_teacher_start_v2"
        ),
        obs_mode=historical.obs_mode,
        total_dim=194,
        terms=historical.terms[:-1]
        + (
            ActorObservationTerm(
                "time_to_teacher_start_s",
                1,
                "action_ball_motion_phase_governor",
                "live seconds until the frozen selected action's teacher "
                "leaves its ready frame",
            ),
        ),
    )


# Stage-1 face-command contract (2026-07-06): deploy_parity + the +4D face-command channel
# appended LAST — racket_target_normal_cmd = demanded face normal (3, world frame, from the
# question bank / planner) + spin-rho placeholder (1, zero-filled until the S3 spin tier).
# Matches train.py's face_command_obs wiring and the MuJoCo evaluator's 179-D support (bec7673).
# Contract-day note: the 181-D single cut (this + Hitter's +2 station dims) supersedes it later.
DEPLOY_PARITY_FACE179 = ActorObservationContract(
    name="deploy_parity_face179",
    obs_mode="deploy_parity",
    total_dim=179,
    terms=DEPLOY_PARITY.terms
    + (
        ActorObservationTerm(
            "racket_target_normal_cmd",
            4,
            "planner",
            "demanded racket face normal (world, 3) + spin-rho placeholder (zero until S3)",
        ),
    ),
)

# R10c station-anchor contract (2026-07-09, franco:"planner 的 p_base 应该加进去——就算不需要
# 移动,它也是一个锚"): face179 + station_anchor_err_b(2) appended LAST = 181.
# 人话:在 179(拍面通道)之后再追加 2 维世界系站位锚误差(出生点常数 − 当前 base XY,base 系),
# 躯干漂了这两个数自己变大 = R9a 删缰绳后丢世界系基准的任务通道解法。
# ⚠ 布局取舍(有意为之,和契约日蓝图不同):契约日 181 蓝图 = 175+站位2+拍面3+ρ1(站位在拍面
# 前,即 Hitter 177 的第 167 列位置)。这里为了现役 179 存档能走 pad_obs_cols 纯尾部扩列热启
# (第 0 步逐位相同),站位排在拍面后。契约日统一布局时再做一次插列手术或重训对齐——到时以
# 蓝图为准,这个契约名保留作训练侧过渡。
DEPLOY_PARITY_STATION181 = ActorObservationContract(
    name="deploy_parity_station181",
    obs_mode="deploy_parity",
    total_dim=181,
    terms=DEPLOY_PARITY_FACE179.terms
    + (
        ActorObservationTerm(
            "station_anchor_err_b",
            2,
            "planner_plus_mocap",
            "world-frame station anchor (spawn-point constant) minus current base XY, rotated into "
            "the yaw-heading base frame; mocap base position at deploy — relative Δ only, Δ=0 "
            "graceful fallback on mocap dropout (same recipe as Hitter's base_target_pos_b)",
        ),
    ),
)

HITTER_PURE = ActorObservationContract(
    name="hitter_pure",
    obs_mode="hitter_pure",
    total_dim=110,
    terms=(
        # HITTER (arXiv:2508.21043) Table I EXACT structure, sized for the A3's 31 joints:
        # proprioception + goal ONLY. The 62-D reference joint stream is CRITIC-ONLY in the paper
        # (and here); swing_type is not observed anywhere (deploy infers fh/bh outside the policy,
        # §V-B-3). Target vectors are WORLD-frame with the explicit base forward vector e_base,x.
        ActorObservationTerm("base_ang_vel", 3, "imu", "pelvis angular velocity in the body frame"),
        ActorObservationTerm("joint_pos", 31, "encoders", "joint position offset from default"),
        ActorObservationTerm("joint_vel", 31, "encoders", "joint velocities"),
        ActorObservationTerm("actions", 31, "runtime_state", "previous policy action"),
        ActorObservationTerm("projected_gravity", 3, "imu", "gravity direction in the base frame"),
        ActorObservationTerm(
            "base_forward_xy", 2, "imu_yaw_aligned", "base forward unit vector e_base,x (world xy)"
        ),
        ActorObservationTerm(
            "base_target_delta_xy",
            2,
            "planner_plus_mocap",
            "target base position minus current base position, world xy (Δ=0 on mocap dropout)",
        ),
        ActorObservationTerm(
            "racket_target_rel_base",
            3,
            "planner_plus_mocap",
            "target racket position relative to the base, world frame",
        ),
        ActorObservationTerm("racket_target_vel_w", 3, "planner", "desired racket velocity in the world frame"),
        ActorObservationTerm("time_to_strike", 1, "reference_clock", "time remaining until strike"),
    ),
)

CONTRACTS = {
    FULL.name: FULL,
    DEPLOY_PARITY.name: DEPLOY_PARITY,
    DEPLOY_PARITY_FACE179.name: DEPLOY_PARITY_FACE179,
    DEPLOY_PARITY_STATION181.name: DEPLOY_PARITY_STATION181,
    HITTER_FOOTWORK.name: HITTER_FOOTWORK,
    STAGE1_NATURAL_CLIP_SITE_V1.name: STAGE1_NATURAL_CLIP_SITE_V1,
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.name: (
        STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2
    ),
    ACTION_BALL_A225.name: ACTION_BALL_A225,
    ACTION_BALL_C225.name: ACTION_BALL_C225,
    ACTION_BALL_A211.name: ACTION_BALL_A211,
    ACTION_BALL_C211.name: ACTION_BALL_C211,
    HITTER_PURE.name: HITTER_PURE,
    FULL.obs_mode: FULL,
    DEPLOY_PARITY.obs_mode: DEPLOY_PARITY,
    HITTER_FOOTWORK.obs_mode: HITTER_FOOTWORK,
    STAGE1_NATURAL_CLIP_SITE_V1.obs_mode: STAGE1_NATURAL_CLIP_SITE_V1,
    STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.obs_mode: (
        STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2
    ),
    HITTER_PURE.obs_mode: HITTER_PURE,
    (
        "action_ball_table_pose_twist_heading_task_teacher_start_v2"
    ): action_ball_table_pose_twist_heading_task_teacher_start_v2_contract(),
    **{alias: DEPLOY_PARITY for alias in DEPLOY_PARITY.legacy_names},
}


def resolve_actor_observation_contract(name: str | None) -> ActorObservationContract | None:
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return None
    dynamic = re.fullmatch(r"task_first_n([1-9][0-9]*)", key)
    if dynamic is not None:
        return task_first_n_contract(int(dynamic.group(1)))
    dynamic = re.fullmatch(r"action_ball_n([1-9][0-9]*)", key)
    if dynamic is not None:
        return action_ball_n_contract(int(dynamic.group(1)))
    dynamic = re.fullmatch(
        (
            r"action_ball_table_pose_twist_heading_task_teacher_start_n"
            r"([1-9][0-9]*)"
        ),
        key,
    )
    if dynamic is not None:
        return (
            action_ball_table_pose_twist_heading_task_teacher_start_n_contract(
                int(dynamic.group(1))
            )
        )
    if key.startswith(
        "action_ball_table_pose_twist_heading_task_teacher_start_n"
    ):
        raise ValueError(
            "Invalid teacher-start table-pose-twist action-ball actor "
            f"observation contract {name!r}; expected "
            "action_ball_table_pose_twist_heading_task_teacher_start_n<N> "
            "with a base-10 N in [1,1024] and no leading zeros"
        )
    dynamic = re.fullmatch(
        r"action_ball_table_pose_twist_heading_task_n([1-9][0-9]*)", key
    )
    if dynamic is not None:
        return action_ball_table_pose_twist_heading_task_n_contract(
            int(dynamic.group(1))
        )
    if key.startswith("action_ball_table_pose_twist_heading_task_n"):
        raise ValueError(
            "Invalid frame-consistent table-pose-twist action-ball actor "
            f"observation contract {name!r}; expected "
            "action_ball_table_pose_twist_heading_task_n<N> with a base-10 N "
            "in [1,1024] and no leading zeros"
        )
    dynamic = re.fullmatch(
        r"action_ball_table_pose_twist_n([1-9][0-9]*)", key
    )
    if dynamic is not None:
        return action_ball_table_pose_twist_n_contract(int(dynamic.group(1)))
    if key.startswith("action_ball_table_pose_twist_n"):
        raise ValueError(
            f"Invalid table-pose-twist action-ball actor observation contract "
            f"{name!r}; expected action_ball_table_pose_twist_n<N> with a "
            "base-10 N in [1,1024] and no leading zeros"
        )
    dynamic = re.fullmatch(
        r"action_ball_table_pose_n([1-9][0-9]*)", key
    )
    if dynamic is not None:
        return action_ball_table_pose_n_contract(int(dynamic.group(1)))
    if key.startswith("action_ball_table_pose_n"):
        raise ValueError(
            f"Invalid table-pose action-ball actor observation contract "
            f"{name!r}; expected action_ball_table_pose_n<N> with a base-10 "
            "N in [1,1024] and no leading zeros"
        )
    if key.startswith("action_ball_n"):
        raise ValueError(
            f"Invalid action-ball actor observation contract {name!r}; expected "
            "action_ball_n<N> with a base-10 N in [1,1024] and no leading zeros"
        )
    if key not in CONTRACTS:
        known = ", ".join(sorted(CONTRACTS))
        raise ValueError(
            f"Unknown actor observation contract '{name}'. Known values: {known}, "
            "task_first_n<N>, action_ball_n<N>, "
            "action_ball_table_pose_twist_heading_task_teacher_start_v2, "
            "action_ball_table_pose_twist_heading_task_teacher_start_n<N>, "
            "action_ball_table_pose_twist_heading_task_n<N>, "
            "action_ball_table_pose_n<N>, "
            "action_ball_table_pose_twist_n<N>"
        )
    return CONTRACTS[key]


def policy_layout_from_env(env) -> tuple[tuple[str, int], ...]:
    observation_manager = env.observation_manager
    names = list(observation_manager.active_terms["policy"])
    dims = observation_manager.group_obs_term_dim["policy"]
    flat_dims = []
    for dim in dims:
        if isinstance(dim, (tuple, list)):
            flat_dims.append(int(dim[0]))
        else:
            flat_dims.append(int(dim))
    return tuple(zip(names, flat_dims))


def total_policy_dim_from_env(env) -> int:
    total = env.observation_manager.group_obs_dim["policy"]
    if isinstance(total, (tuple, list)):
        return int(total[0])
    return int(total)


def infer_actor_observation_contract(env) -> ActorObservationContract | None:
    layout = policy_layout_from_env(env)
    total_dim = total_policy_dim_from_env(env)
    for contract in (
        FULL,
        DEPLOY_PARITY,
        HITTER_FOOTWORK,
        STAGE1_NATURAL_CLIP_SITE_V1,
        STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2,
        ACTION_BALL_A225,
        ACTION_BALL_C225,
        ACTION_BALL_A211,
        ACTION_BALL_C211,
        DEPLOY_PARITY_FACE179,
        DEPLOY_PARITY_STATION181,
        HITTER_PURE,
        action_ball_table_pose_twist_heading_task_teacher_start_v2_contract(),
    ):
        if layout == contract.layout and total_dim == contract.total_dim:
            return contract
    return None


def validate_actor_observation_contract(env, expected: str | ActorObservationContract) -> ActorObservationContract:
    contract = (
        expected
        if isinstance(expected, ActorObservationContract)
        else resolve_actor_observation_contract(expected)
    )
    if contract is None:
        raise ValueError("Expected actor observation contract must not be empty.")
    layout = policy_layout_from_env(env)
    total_dim = total_policy_dim_from_env(env)
    if layout != contract.layout or total_dim != contract.total_dim:
        raise ValueError(
            "Actor observation contract mismatch.\n"
            f"Expected {contract.name} ({contract.total_dim}D): {format_layout(contract.layout)}\n"
            f"Actual   ({total_dim}D): {format_layout(layout)}"
        )
    return contract


def removed_terms_vs_full(contract: str | ActorObservationContract) -> tuple[ActorObservationTerm, ...]:
    resolved = contract if isinstance(contract, ActorObservationContract) else resolve_actor_observation_contract(contract)
    if resolved is None:
        return ()
    removed = []
    active = {term.name for term in resolved.terms}
    for term in FULL.terms:
        if term.name not in active:
            removed.append(term)
    return tuple(removed)


def format_layout(layout: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{name}:{dim}" for name, dim in layout)
