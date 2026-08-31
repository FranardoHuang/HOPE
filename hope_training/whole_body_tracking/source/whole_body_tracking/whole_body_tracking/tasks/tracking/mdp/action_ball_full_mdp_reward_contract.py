"""Dependency-light ordered RewardManager contract for fresh ActionBall FullMDP."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple


HELD_RACKET_WRIST_BODY_NAME = "right_wrist_yaw_Link"
UPPER_EXCEPT_HELD_WRIST = "upper_except_held_wrist"


def upper_except_held_wrist_body_names(
    upper_body_names: object,
) -> Tuple[str, ...]:
    """Return the configured upper-body order with the held wrist removed.

    The measured clips are a swing/style reference, not an executable lower-body
    oracle.  Keeping pelvis/legs in the imitation average makes balance compete
    with floating, wide-stance capture geometry.  The torso anchor remains a
    separate term; this scope leaves the policy free to find a stable stance.
    """

    if not isinstance(upper_body_names, (tuple, list)):
        raise ValueError("FullMDP upper body names must be one tuple/list")
    names = tuple(upper_body_names)
    if (
        not names
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
        or names.count(HELD_RACKET_WRIST_BODY_NAME) != 1
    ):
        raise ValueError("FullMDP upper body names lack one held racket wrist")
    return tuple(name for name in names if name != HELD_RACKET_WRIST_BODY_NAME)


class DenseRewardSpec(NamedTuple):
    """One fixed dense reward term without importing Torch or Isaac Lab."""

    manager_name: str
    evaluator_name: str
    manager_weight: float
    command_name: str
    std: float
    body_scope: Optional[str] = None
    coarse_std: Optional[float] = None
    contact_peak_scale: Optional[float] = None


class RegularizationRewardSpec(NamedTuple):
    """One backend-neutral continuous cost and its effective coefficient."""

    manager_name: str
    evaluator_name: str
    manager_weight: float
    effective_coefficient: float


REGULARIZATION_JOINT_COUNT = 31
REGULARIZATION_SOFT_LIMIT_MARGIN_FRAC = 0.02
REGULARIZATION_SOFT_LIMIT_PENALTY_FLOOR = 0.25
REGULARIZATION_PROJECTION_KNEE_FRAC = 0.05
REGULARIZATION_STANCE_EPS_FRAC = 0.005
REGULARIZATION_MARGIN_FLOOR_FRAC = 1.0e-6
REGULARIZATION_SPECS = (
    RegularizationRewardSpec("action_rate_l2", "action_rate_l2", 0.1, -0.1),
    RegularizationRewardSpec(
        "qdes_limit_barrier", "qdes_limit_barrier_v2", 10.0, -10.0
    ),
    RegularizationRewardSpec(
        "qdes_projection_penalty", "qdes_projection_penalty", 1.0, -1.0
    ),
    RegularizationRewardSpec(
        "joint_limit", "actual_joint_limit_barrier_v2", 10.0, -10.0
    ),
)
REGULARIZATION_NAMES = tuple(spec.manager_name for spec in REGULARIZATION_SPECS)

if any(
    spec.manager_weight <= 0.0
    or spec.effective_coefficient != -spec.manager_weight
    for spec in REGULARIZATION_SPECS
):
    raise RuntimeError("FullMDP regularization coefficients must be negative costs")


LIFECYCLE_MANAGER_NAMES = (
    "racket_position",
    "racket_velocity",
    "racket_normal",
    "racket_position_coarse",
    "racket_velocity_coarse",
    "racket_normal_coarse",
    "racket_position_precision",
    "racket_velocity_precision",
    "racket_normal_precision",
    "paddle_center_proximity",
    "physical_selected_contact",
    "common_on_table_outcome",
    "post_contact_placement_guidance",
    "common_recovery_reward_v1",
)

COMMON_DENSE_SPECS = (
    DenseRewardSpec(
        "motion_global_anchor_pos",
        "motion_global_anchor_position_error_exp",
        0.5,
        "motion",
        0.3,
    ),
    DenseRewardSpec(
        "motion_global_anchor_ori",
        "motion_global_anchor_orientation_error_exp",
        0.5,
        "motion",
        0.4,
    ),
    DenseRewardSpec(
        "motion_body_pos",
        "motion_relative_body_position_error_exp",
        1.0,
        "motion",
        0.3,
        body_scope=UPPER_EXCEPT_HELD_WRIST,
    ),
    DenseRewardSpec(
        "motion_body_ori",
        "motion_relative_body_orientation_error_exp",
        1.0,
        "motion",
        0.4,
        body_scope=UPPER_EXCEPT_HELD_WRIST,
        coarse_std=1.0,
    ),
    DenseRewardSpec(
        "motion_body_lin_vel",
        "motion_global_body_linear_velocity_error_exp",
        1.0,
        "motion",
        1.0,
        body_scope=UPPER_EXCEPT_HELD_WRIST,
    ),
    DenseRewardSpec(
        "motion_body_ang_vel",
        "motion_global_body_angular_velocity_error_exp",
        1.0,
        "motion",
        3.14,
        body_scope=UPPER_EXCEPT_HELD_WRIST,
    ),
)

# A matched fresh run falsified the first successor's unconditional 4x manager
# weights: it improved three aggregate paddle residuals but collapsed selected
# contact from 332 to 1 by update 423.  The fault was structural, not a request
# for another scalar search.  The same high-value paddle objective paid while
# Motion still held the easy ready pose, so the learner could strengthen a
# pre-task local optimum before the first dynamic teacher row appeared.
#
# Keep the original 1/1/1/.5 full-phase prior so ready/balance retains the
# already observed baseline economy.  During playback the same four kernels
# receive a smooth raised-cosine emphasis centred on the immutable contact
# time.  The multiplier is 1 at and outside +/-0.12 s and reaches 4 only at
# contact.  This is temporal credit assignment, not a Stage or success Gate:
# every objective remains present from rollout zero, while easy low-speed
# playback rows can no longer outvote the short contact neighbourhood 4:1.
PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE = 4.0
PADDLE_MOTION_PRIOR_CONTACT_HALF_WINDOW_S = 0.12
PADDLE_MOTION_PRIOR_SPECS = (
    DenseRewardSpec(
        "motion_racket_position",
        "motion_racket_position_tracking_cauchy",
        1.0,
        "racket_target",
        0.075,
        coarse_std=0.30,
        contact_peak_scale=PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE,
    ),
    DenseRewardSpec(
        "motion_racket_velocity",
        "motion_racket_velocity_tracking_cauchy",
        1.0,
        "racket_target",
        0.50,
        coarse_std=2.0,
        contact_peak_scale=PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE,
    ),
    DenseRewardSpec(
        "motion_racket_normal",
        "motion_racket_normal_tracking_cauchy",
        1.0,
        "racket_target",
        0.2617993877991494,
        coarse_std=1.0471975511965976,
        contact_peak_scale=PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE,
    ),
    DenseRewardSpec(
        "motion_racket_long_axis",
        "motion_racket_long_axis_tracking_cauchy",
        0.5,
        "racket_target",
        0.17453292519943295,
        coarse_std=0.6981317007977318,
        contact_peak_scale=PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE,
    ),
)

COMMON_DENSE_NAMES = tuple(spec.manager_name for spec in COMMON_DENSE_SPECS)
PADDLE_MOTION_PRIOR_NAMES = tuple(
    spec.manager_name for spec in PADDLE_MOTION_PRIOR_SPECS
)
MANAGER_NAMES = (
    LIFECYCLE_MANAGER_NAMES
    + COMMON_DENSE_NAMES
    + PADDLE_MOTION_PRIOR_NAMES
    + REGULARIZATION_NAMES
)
LIFECYCLE_PAYMENT_COUNT = len(LIFECYCLE_MANAGER_NAMES)
REWARD_TERM_COUNT = len(MANAGER_NAMES)

if len(set(MANAGER_NAMES)) != REWARD_TERM_COUNT:
    raise RuntimeError("FullMDP RewardManager names must be unique")


__all__ = [
    "HELD_RACKET_WRIST_BODY_NAME",
    "UPPER_EXCEPT_HELD_WRIST",
    "upper_except_held_wrist_body_names",
    "DenseRewardSpec",
    "RegularizationRewardSpec",
    "REGULARIZATION_JOINT_COUNT",
    "REGULARIZATION_SOFT_LIMIT_MARGIN_FRAC",
    "REGULARIZATION_SOFT_LIMIT_PENALTY_FLOOR",
    "REGULARIZATION_PROJECTION_KNEE_FRAC",
    "REGULARIZATION_STANCE_EPS_FRAC",
    "REGULARIZATION_MARGIN_FLOOR_FRAC",
    "LIFECYCLE_MANAGER_NAMES",
    "COMMON_DENSE_SPECS",
    "PADDLE_MOTION_PRIOR_CONTACT_PEAK_SCALE",
    "PADDLE_MOTION_PRIOR_CONTACT_HALF_WINDOW_S",
    "PADDLE_MOTION_PRIOR_SPECS",
    "COMMON_DENSE_NAMES",
    "PADDLE_MOTION_PRIOR_NAMES",
    "REGULARIZATION_SPECS",
    "REGULARIZATION_NAMES",
    "MANAGER_NAMES",
    "LIFECYCLE_PAYMENT_COUNT",
    "REWARD_TERM_COUNT",
]
