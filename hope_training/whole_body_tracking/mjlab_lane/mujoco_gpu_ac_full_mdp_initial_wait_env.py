"""First MuJoCo FullMDP slice: one real reset and the 229/399 WAIT view.

The plant, masked reset implementation, and ``sim.forward()`` remain the
tracked :class:`A3ReadyBallVecEnv` implementation.  This subclass only parks
the ball and projects the post-forward live robot tensors into the shared
ActionEpoch observation order.  Task, epoch, physical-fact, and reward rows
are zero because no question has been revealed in this slice.

Normal stepping is intentionally unavailable.  Reward, termination, lifecycle
writers, and ActionEpoch transitions must exist before this can become a
trainable VecEnv.
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from .a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
except ImportError:  # Direct execution with mjlab_lane on PYTHONPATH.
    from a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg


_HERE = Path(__file__).resolve().parent
_MDP = (
    _HERE.parent
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
if str(_MDP) not in sys.path:
    sys.path.insert(0, str(_MDP))

import action_ball_full_mdp_portable_observation as observation_contract


WAIT_BALL_PARK_HOPE = (0.0, 0.0, 10.0)
READY_HOLD_PHASE_INDEX = 4


class FullMdpInitialWaitVecEnv(A3ReadyBallVecEnv):
    """N=1 real plant reset projected into the live 229/399 contract."""

    def __init__(
        self,
        sim_cfg: SimCfg,
        task_cfg: TaskCfg,
        device: str,
        xml_path=None,
        ready_pose_path=None,
        seed: int = 0,
        capacity_probe: bool = True,
    ) -> None:
        if int(sim_cfg.nworld) != 1:
            raise ValueError("initial-WAIT FullMDP slice requires nworld=1")
        super().__init__(
            sim_cfg=sim_cfg,
            task_cfg=task_cfg,
            device=device,
            xml_path=xml_path,
            ready_pose_path=ready_pose_path,
            seed=seed,
            count_contacts=True,
            capacity_probe=capacity_probe,
        )

    def _serve(self, ids) -> None:
        """Keep every unrevealed ball at the canonical contact-free park."""

        torch = self._torch
        n = int(ids.numel())
        if n == 0:
            return
        park_hope = torch.tensor(
            WAIT_BALL_PARK_HOPE, dtype=self.qpos_init.dtype, device=self.device
        )
        park_scene = park_hope + self.hope_to_scene
        data = self.sim.data
        data.qpos[ids, self.b_q : self.b_q + 3] = park_scene.expand(n, 3)
        data.qpos[ids, self.b_q + 3 : self.b_q + 7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=self.qpos_init.dtype,
            device=self.device,
        ).expand(n, 4)
        data.qvel[ids, self.b_v : self.b_v + 6] = 0.0
        self.ball_age_buf[ids] = 0

    def _compute_obs(self, st=None):
        """Read the live post-forward plant and publish initial WAIT only."""

        torch = self._torch
        contact = self._con_geom[:]
        valid = self._con_idx < self._nacon[0]
        ball_contact = valid & (
            (contact[:, 0] == self._ball_gid)
            | (contact[:, 1] == self._ball_gid)
        )
        if bool(ball_contact.any()):
            raise RuntimeError("portable FullMDP initial-WAIT ball is in contact")
        st = st or self._state()
        joint_pos_rel = self._qpos_act() - self.q_ready.unsqueeze(0)
        zero_joint = torch.zeros_like(joint_pos_rel)
        phase = torch.zeros(
            (self.num_envs, 5), dtype=joint_pos_rel.dtype, device=self.device
        )
        phase[:, READY_HOLD_PHASE_INDEX] = 1.0

        actor_rows = {
            name: torch.zeros(
                (self.num_envs, width),
                dtype=joint_pos_rel.dtype,
                device=self.device,
            )
            for name, width in observation_contract.ACTOR_LAYOUT_V1
        }
        actor_rows.update(
            {
                "projected_gravity_b": st["proj_g"],
                "base_ang_vel_b": st["base_ang_b"],
                "joint_pos_rel": joint_pos_rel,
                "joint_vel_rel": self._qvel_act(),
                "last_action": self.actions,
                "teacher_joint_pos_rel": zero_joint,
                "teacher_joint_vel_rel": zero_joint,
                "motion_phase_one_hot": phase,
            }
        )
        actor_rows["epoch_phase_one_hot"][
            :, observation_contract.EPOCH_IDLE_PHASE_INDEX
        ] = 1.0
        policy = observation_contract.concatenate_layout_rows(
            observation_contract.ACTOR_LAYOUT_V1, actor_rows
        )
        critic_extension = observation_contract.concatenate_layout_rows(
            observation_contract.CRITIC_EXTENSION_LAYOUT_V1,
            {
                name: torch.zeros(
                    (self.num_envs, width),
                    dtype=joint_pos_rel.dtype,
                    device=self.device,
                )
                for name, width in observation_contract.CRITIC_EXTENSION_LAYOUT_V1
            },
        )
        critic = torch.cat((policy, critic_extension), dim=1)
        if tuple(policy.shape) != (
            self.num_envs,
            observation_contract.ACTOR_WIDTH_V1,
        ) or tuple(critic.shape) != (
            self.num_envs,
            observation_contract.CRITIC_WIDTH_V1,
        ):
            raise RuntimeError("portable FullMDP observation width differs")
        if not bool(torch.isfinite(policy).all()) or not bool(
            torch.isfinite(critic).all()
        ):
            raise RuntimeError("portable FullMDP initial-WAIT observation is nonfinite")
        self._obs_buf = policy
        self._critic_obs_buf = critic
        return policy

    def get_observations(self):
        """Return the stock RSL-RL TensorDict group surface."""

        from tensordict import TensorDict

        return TensorDict(
            {"policy": self._obs_buf, "critic": self._critic_obs_buf},
            batch_size=[self.num_envs],
        )

    def step(self, _actions):
        raise RuntimeError(
            "initial-WAIT slice has no FullMDP step/reward/termination producers"
        )


__all__ = [
    "WAIT_BALL_PARK_HOPE",
    "READY_HOLD_PHASE_INDEX",
    "FullMdpInitialWaitVecEnv",
]
