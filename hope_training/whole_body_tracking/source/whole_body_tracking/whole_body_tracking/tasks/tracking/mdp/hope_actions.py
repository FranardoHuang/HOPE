"""Deploy-faithful action terms.

ClampedJointPositionAction (2026-07-05): the C++ deploy runner clamps q_des to the
A3 joint limits before publishing (pp_joint_limits.hpp) — a SAFETY feature — but
training ran with no clamp (clip_actions=null, and PhysX implicit drives accept
out-of-range targets as "saturated torque, please"). The policy legitimately
learned to command PAST the ankle limit to buy kp-saturated torque when arresting
a forward tip (118 Nm requested -> clamp cut it to ~41 Nm on 34% of bare-hold
ticks in the Gate 2.5 P2 log = ~65% of the tipping-arrest torque silently
removed at deploy). Clamping the PROCESSED action (the joint-position target) in
training makes train == deploy so the policy learns torque strategies that
survive the runner's clamp.

DEFAULT ON since 2026-07-06 (franco ruling): jiayi found the unclamped P2 product
line CANNOT EVEN STAND in the MuJoCo gate — the policy's balance strategy leans
on out-of-range q_des torque the deploy runner will never grant. This is a
train==deploy correctness alignment, not a tunable: every future run trains
clamped. `clamp=False` remains available ONLY for explicit legacy-reproduction /
control arms (`actions: qdes_clamp: false` in the task YAML), and batch
comparisons must keep clamp state uniform within the batch.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class ClampedJointPositionAction(JointPositionAction):
    """JointPositionAction with an OPTIONAL q_des clamp to the articulation's (soft)
    joint position limits — mirrors the deploy runner's clamp when cfg.clamp=True;
    behaviorally identical to the stock action when cfg.clamp=False (default)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._clamp_enabled = bool(getattr(cfg, "clamp", False))
        if self._clamp_enabled:
            print("[hope_actions] q_des CLAMP ACTIVE: processed joint targets clamped to "
                  "joint limits (train==deploy, pp_joint_limits parity)", flush=True)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if not self._clamp_enabled:
            return
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        self._processed_actions = torch.clamp(
            self._processed_actions, min=limits[..., 0], max=limits[..., 1]
        )


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction
    # ON by default (franco 2026-07-06, after jiayi's P2-cannot-stand-in-MuJoCo finding).
    # Set `actions: qdes_clamp: false` in a task YAML ONLY for legacy-reproduction arms.
    clamp: bool = True
