"""Engine-neutral ordered layout for the live ActionEpoch observation.

This module owns only column names, widths, and concatenation order.  Isaac
and MuJoCo supply their own live tensors; no owner, receipt, registry, or
runtime authority is defined here.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple


TASK_F32_WIDTH = 45
OWNER_FACT_F32_WIDTH = 32
REWARD_CONSUMER_COUNT = 14
EPOCH_IDLE_PHASE_INDEX = 0

ACTOR_CONTRACT_V1 = "action_ball_full_mdp_action_epoch_v1"
CRITIC_CONTRACT_V1 = "action_ball_full_mdp_action_epoch_critic_v1"
OBSERVATION_KIND_V1 = "action_ball_full_mdp_action_epoch_observation_v1"

DIRECT_FIELD_LAYOUT_V1 = (
    ("projected_gravity_b", 3),
    ("base_ang_vel_b", 3),
    ("joint_pos_rel", 31),
    ("joint_vel_rel", 31),
    ("last_action", 31),
    ("teacher_joint_pos_rel", 31),
    ("teacher_joint_vel_rel", 31),
)

ACTOR_LAYOUT_V1 = DIRECT_FIELD_LAYOUT_V1 + (
    ("motion_phase_one_hot", 5),
    ("epoch_task_f32", TASK_F32_WIDTH),
    ("epoch_clock_remaining_s", 5),
    ("epoch_phase_one_hot", 10),
    ("epoch_task_valid", 1),
    ("epoch_selected", 1),
    ("epoch_launch_succeeded", 1),
)

CRITIC_EXTENSION_LAYOUT_V1 = (
    ("physical_r03_r06_r07_fact_present", 4),
    ("physical_r03_r06_r07_fact_age_s", 4),
    ("physical_r03_r06_r07_fact_f32", 4 * OWNER_FACT_F32_WIDTH),
    ("physical_r03_r06_r07_fault_present", 4),
    ("reward_cycle_open", 1),
    ("reward_cycle_fault_present", 1),
    ("reward_due", REWARD_CONSUMER_COUNT),
    ("reward_paid", REWARD_CONSUMER_COUNT),
)

ACTOR_WIDTH_V1 = sum(width for _, width in ACTOR_LAYOUT_V1)
CRITIC_WIDTH_V1 = ACTOR_WIDTH_V1 + sum(
    width for _, width in CRITIC_EXTENSION_LAYOUT_V1
)


def concatenate_layout_rows(
    layout: Sequence[Tuple[str, int]], rows: Mapping[str, object]
):
    """Concatenate tensors in the one shared column order.

    Callers retain responsibility for producing and validating live tensors.
    Keeping this helper deliberately mechanical prevents a second backend from
    silently reordering otherwise same-named columns.
    """

    import torch

    return torch.cat(tuple(rows[name] for name, _ in layout), dim=1)


__all__ = [
    "TASK_F32_WIDTH",
    "OWNER_FACT_F32_WIDTH",
    "REWARD_CONSUMER_COUNT",
    "EPOCH_IDLE_PHASE_INDEX",
    "ACTOR_CONTRACT_V1",
    "CRITIC_CONTRACT_V1",
    "OBSERVATION_KIND_V1",
    "DIRECT_FIELD_LAYOUT_V1",
    "ACTOR_LAYOUT_V1",
    "CRITIC_EXTENSION_LAYOUT_V1",
    "ACTOR_WIDTH_V1",
    "CRITIC_WIDTH_V1",
    "concatenate_layout_rows",
]
