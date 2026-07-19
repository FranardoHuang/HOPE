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

from collections.abc import Sequence

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
        # Keep deploy-space q_des history next to the action term that owns the affine transform
        # and safety clamp.  ActionManager.prev_action is the previous *normalized actor output*;
        # it cannot attest the target the PD controller actually received after offset/scale/clamp.
        # A separate validity bit is essential because JointAction.reset() clears only raw_actions
        # and intentionally leaves processed_actions untouched in Isaac Lab 2.1.
        self._previous_processed_qdes = torch.zeros_like(self._processed_actions)
        self._previous_processed_qdes_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._processed_qdes_valid = torch.zeros_like(
            self._previous_processed_qdes_valid
        )
        if self._clamp_enabled:
            print("[hope_actions] q_des CLAMP ACTIVE: processed joint targets clamped to "
                  "joint limits (train==deploy, pp_joint_limits parity)", flush=True)

    def process_actions(self, actions: torch.Tensor):
        # Snapshot the preceding deploy-space target before JointPositionAction overwrites it.
        # On the first action after reset the numeric snapshot is deliberately ignored by the
        # copied validity bit, so an episode boundary can never create a fictitious slew charge.
        self._previous_processed_qdes.copy_(self._processed_actions)
        self._previous_processed_qdes_valid.copy_(self._processed_qdes_valid)
        super().process_actions(actions)
        if self._clamp_enabled:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
            self._processed_actions = torch.clamp(
                self._processed_actions, min=limits[..., 0], max=limits[..., 1]
            )
        self._processed_qdes_valid.fill_(True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate deploy-space history for reset environments.

        Isaac's ActionManager resolves ``None`` to ``slice(None)`` before calling terms, but
        accepting ``None`` here as well keeps direct callers safe and matches the base API.
        """

        super().reset(env_ids=env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_qdes_valid[ids] = False
        self._previous_processed_qdes_valid[ids] = False

    @property
    def previous_processed_qdes(self) -> torch.Tensor:
        """Previous affine-transformed and clamp-applied joint-position target."""

        return self._previous_processed_qdes

    @property
    def previous_processed_qdes_valid(self) -> torch.Tensor:
        """Per-environment validity of :attr:`previous_processed_qdes`."""

        return self._previous_processed_qdes_valid


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction
    # ON by default (franco 2026-07-06, after jiayi's P2-cannot-stand-in-MuJoCo finding).
    # Set `actions: qdes_clamp: false` in a task YAML ONLY for legacy-reproduction arms.
    clamp: bool = True
