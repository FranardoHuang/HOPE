"""Dependency-light ordered RewardManager contract for fresh ActionBall FullMDP."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple


HELD_RACKET_WRIST_BODY_NAME = "right_wrist_yaw_Link"
TRACKED_EXCEPT_HELD_WRIST = "tracked_except_held_wrist"


def tracked_except_held_wrist_body_names(
    tracked_body_names: object,
) -> Tuple[str, ...]:
    """Return the configured tracked-body order with the held wrist removed."""

    if not isinstance(tracked_body_names, (tuple, list)):
        raise ValueError("FullMDP tracked body names must be one tuple/list")
    names = tuple(tracked_body_names)
    if (
        not names
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
        or names.count(HELD_RACKET_WRIST_BODY_NAME) != 1
    ):
        raise ValueError("FullMDP tracked body names lack one held racket wrist")
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
    scale_in_strike_window: Optional[float] = None


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
        body_scope=TRACKED_EXCEPT_HELD_WRIST,
    ),
    DenseRewardSpec(
        "motion_body_ori",
        "motion_relative_body_orientation_error_exp",
        1.0,
        "motion",
        0.4,
        body_scope=TRACKED_EXCEPT_HELD_WRIST,
        coarse_std=1.0,
    ),
    DenseRewardSpec(
        "motion_body_lin_vel",
        "motion_global_body_linear_velocity_error_exp",
        1.0,
        "motion",
        1.0,
        body_scope=TRACKED_EXCEPT_HELD_WRIST,
    ),
    DenseRewardSpec(
        "motion_body_ang_vel",
        "motion_global_body_angular_velocity_error_exp",
        1.0,
        "motion",
        3.14,
        body_scope=TRACKED_EXCEPT_HELD_WRIST,
    ),
)

# The four-channel cap is 70% of the six body-imitation cap (3.5 vs 5.0).
# This preserves a material official-paddle gradient after the held wrist is
# removed from the body average.  The older 0.20/0.20/0.20/0.10 values only
# had this role when all six body terms were separately multiplied by 0.15;
# copying them beside unscaled body terms diluted the bridge by about 6.7x.
# The contact target is constructed from the same measured motion row, so the
# prior remains full strength through the strike window instead of creating a
# timing hole before the one-tick lifecycle target is paid.
PADDLE_MOTION_PRIOR_SPECS = (
    DenseRewardSpec(
        "motion_racket_position",
        "motion_racket_position_tracking_cauchy",
        1.0,
        "racket_target",
        0.70,
        scale_in_strike_window=1.0,
    ),
    DenseRewardSpec(
        "motion_racket_velocity",
        "motion_racket_velocity_tracking_cauchy",
        1.0,
        "racket_target",
        4.0,
        scale_in_strike_window=1.0,
    ),
    DenseRewardSpec(
        "motion_racket_normal",
        "motion_racket_normal_tracking_cauchy",
        1.0,
        "racket_target",
        3.141592653589793,
        scale_in_strike_window=1.0,
    ),
    DenseRewardSpec(
        "motion_racket_long_axis",
        "motion_racket_long_axis_tracking_cauchy",
        0.5,
        "racket_target",
        1.0,
        scale_in_strike_window=1.0,
    ),
)

COMMON_DENSE_NAMES = tuple(spec.manager_name for spec in COMMON_DENSE_SPECS)
PADDLE_MOTION_PRIOR_NAMES = tuple(
    spec.manager_name for spec in PADDLE_MOTION_PRIOR_SPECS
)
MANAGER_NAMES = (
    LIFECYCLE_MANAGER_NAMES + COMMON_DENSE_NAMES + PADDLE_MOTION_PRIOR_NAMES
)
LIFECYCLE_PAYMENT_COUNT = len(LIFECYCLE_MANAGER_NAMES)
REWARD_TERM_COUNT = len(MANAGER_NAMES)

if len(set(MANAGER_NAMES)) != REWARD_TERM_COUNT:
    raise RuntimeError("FullMDP RewardManager names must be unique")


__all__ = [
    "HELD_RACKET_WRIST_BODY_NAME",
    "TRACKED_EXCEPT_HELD_WRIST",
    "tracked_except_held_wrist_body_names",
    "DenseRewardSpec",
    "LIFECYCLE_MANAGER_NAMES",
    "COMMON_DENSE_SPECS",
    "PADDLE_MOTION_PRIOR_SPECS",
    "COMMON_DENSE_NAMES",
    "PADDLE_MOTION_PRIOR_NAMES",
    "MANAGER_NAMES",
    "LIFECYCLE_PAYMENT_COUNT",
    "REWARD_TERM_COUNT",
]
