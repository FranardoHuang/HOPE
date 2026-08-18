"""First MuJoCo FullMDP slice: one real reset and the 229/399 WAIT view.

The plant, masked reset implementation, and ``sim.forward()`` remain the
tracked :class:`A3ReadyBallVecEnv` implementation.  This subclass only parks
the ball and projects the post-forward live robot tensors into the shared
ActionEpoch observation order.  Task, epoch, physical-fact, and reward rows
are zero because no question has been revealed in this slice.

The narrow WAIT transition below advances the real plant and closes only the
IDLE Reward20/termination/masked-reset path.  Reveal, launch, shot facts, and
the training runner remain unavailable.
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from .a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
    from .mujoco_gpu_ac_table_keepout import DeviceExactTableKeepout
except ImportError:  # Direct execution with mjlab_lane on PYTHONPATH.
    from a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
    from mujoco_gpu_ac_table_keepout import DeviceExactTableKeepout


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
FULLMDP_TRACKED_BODY_NAMES = (
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
)
FULLMDP_ANCHOR_BODY_NAME = "torso_Link"
FULLMDP_DENSE_REWARD_SPECS = (
    ("motion_global_anchor_pos", 0.5, 0.3),
    ("motion_global_anchor_ori", 0.5, 0.4),
    ("motion_body_pos", 1.0, 0.3),
    ("motion_body_ori", 1.0, 0.4),
    ("motion_body_lin_vel", 1.0, 1.0),
    ("motion_body_ang_vel", 1.0, 3.14),
)
FULLMDP_TERMINATION_BITS = {
    "time_out": 1,
    "base_fell_tilt": 2,
    "base_too_low": 4,
    "joint_qdes_forbidden": 8,
    "robot_hit_table": 16,
}


class FullMdpInitialWaitVecEnv(A3ReadyBallVecEnv):
    """Real WAIT reset/transition projected into the live 229/399 contract."""

    def __init__(
        self,
        sim_cfg: SimCfg,
        task_cfg: TaskCfg,
        device: str,
        xml_path=None,
        ready_pose_path=None,
        seed: int = 0,
        capacity_probe: bool = True,
        ready_pose_payload=None,
        ready_pose_source=None,
    ) -> None:
        if int(sim_cfg.nworld) <= 0:
            raise ValueError("initial-WAIT FullMDP slice requires positive nworld")
        if any(
            float(value) != 0.0
            for value in (
                task_cfg.reset_joint_noise_rad,
                task_cfg.reset_joint_vel_noise,
                task_cfg.reset_root_xy_noise_m,
                task_cfg.reset_root_yaw_noise_rad,
            )
        ):
            raise ValueError("initial-WAIT FullMDP slice requires deterministic reset")
        self._fullmdp_initialized = False
        super().__init__(
            sim_cfg=sim_cfg,
            task_cfg=task_cfg,
            device=device,
            xml_path=xml_path,
            ready_pose_path=ready_pose_path,
            ready_pose_payload=ready_pose_payload,
            ready_pose_source=ready_pose_source,
            seed=seed,
            count_contacts=True,
            capacity_probe=capacity_probe,
        )
        import mujoco

        if not self._robot_table_ok:
            raise RuntimeError("initial-WAIT FullMDP table-contact probe is unavailable")
        body_ids = tuple(
            int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}"))
            for name in FULLMDP_TRACKED_BODY_NAMES
        )
        if any(index < 0 for index in body_ids) or len(set(body_ids)) != len(body_ids):
            raise RuntimeError("FullMDP tracked body names do not resolve uniquely")
        self._fullmdp_body_ids = self._torch.tensor(
            body_ids, dtype=self._torch.long, device=self.device
        )
        roots = tuple(int(self.mj_model.body_rootid[index]) for index in body_ids)
        self._fullmdp_body_root_ids = self._torch.tensor(
            roots, dtype=self._torch.long, device=self.device
        )
        self._fullmdp_anchor_index = FULLMDP_TRACKED_BODY_NAMES.index(
            FULLMDP_ANCHOR_BODY_NAME
        )
        weights = [spec[1] for spec in FULLMDP_DENSE_REWARD_SPECS]
        self._fullmdp_dense_weights = self._torch.tensor(
            weights, dtype=self.qpos_init.dtype, device=self.device
        )
        self._epoch_phase = self._torch.zeros(
            self.num_envs, dtype=self._torch.long, device=self.device
        )
        self._epoch_task_valid = self._torch.zeros(
            self.num_envs, dtype=self._torch.bool, device=self.device
        )
        self._epoch_selected = self._torch.zeros_like(self._epoch_task_valid)
        self._epoch_launch_succeeded = self._torch.zeros_like(
            self._epoch_task_valid
        )
        self.reset_generation = self._torch.ones(
            self.num_envs, dtype=self._torch.long, device=self.device
        )
        self.last_terminal_bits = self._torch.zeros_like(self.reset_generation)
        self._cur_table_keepout = self._torch.zeros(
            self.num_envs, dtype=self._torch.bool, device=self.device
        )
        self._table_keepout = DeviceExactTableKeepout(
            mujoco=mujoco,
            model=self.mj_model,
            mjcf_path=self.env.xml_path,
            env_origins=self.env.scene.env_origins,
            device=self.device,
        )
        self._all_env_ids = self._torch.arange(
            self.num_envs, dtype=self._torch.long, device=self.device
        )
        self._snapshot_ready_teacher()
        self._fullmdp_initialized = True
        self._compute_obs()

    def _snapshot_ready_teacher(self) -> None:
        data = self.sim.data
        ids = self._fullmdp_body_ids
        self._teacher_body_pos = data.xpos[:, ids].detach().clone()
        self._teacher_body_quat = data.xquat[:, ids].detach().clone()
        body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        self._teacher_body_lin_vel = body_lin_vel.detach().clone()
        self._teacher_body_ang_vel = body_ang_vel.detach().clone()
        self._refresh_aligned_teacher_body_pose()

    def _refresh_aligned_teacher_body_pose(self) -> None:
        """Publish the next-step MotionCommand-style aligned pose cache."""

        data = self.sim.data
        ids = self._fullmdp_body_ids
        anchor = self._fullmdp_anchor_index
        aligned_pos, aligned_quat = self._aligned_teacher_body_pose(
            self._torch,
            self._teacher_body_pos,
            self._teacher_body_quat,
            data.xpos[:, ids][:, anchor],
            data.xquat[:, ids][:, anchor],
            anchor,
        )
        self._aligned_teacher_body_pos = aligned_pos.detach().clone()
        self._aligned_teacher_body_quat = aligned_quat.detach().clone()

    @staticmethod
    def _body_com_velocities_from_cvel(
        torch, cvel, xipos, subtree_root_com
    ):
        """Translate MuJoCo C-frame velocity to the body inertial COM.

        MuJoCo stores each body's world-axis spatial velocity at the center of
        mass of its kinematic root's subtree.  Isaac's ``body_lin_vel_w`` is
        instead the velocity of the body's own inertial COM.  This is the same
        point-velocity transform used by MuJoCo-Warp's passive-force and
        derivative kernels.
        """

        body_ang_vel = cvel[..., :3]
        body_lin_vel = cvel[..., 3:] - torch.cross(
            xipos - subtree_root_com, body_ang_vel, dim=-1
        )
        return body_lin_vel, body_ang_vel

    def _body_com_velocities_w(self):
        data = self.sim.data
        ids = self._fullmdp_body_ids
        roots = self._fullmdp_body_root_ids
        return self._body_com_velocities_from_cvel(
            self._torch,
            data.cvel[:, ids],
            data.xipos[:, ids],
            data.subtree_com[:, roots],
        )

    def _latch_post_forward_resolved_table_contacts(self) -> None:
        """Include contacts created by the final integration in this step."""

        torch = self._torch
        geom = self._con_geom[:]
        valid = self._con_idx < self._nacon[0]
        g0, g1 = geom[:, 0], geom[:, 1]
        table0 = g0.eq(self._table_gid)
        table1 = g1.eq(self._table_gid)
        partner = torch.where(table0, g1, g0).long()
        robot_table = valid & (table0 | table1) & self._is_robot_geom[partner]
        world = self._con_world[:].long().clamp_(0, self.num_envs - 1)
        self._cur_robot_table.scatter_add_(0, world, robot_table.float())

    def _after_physics_substep(self, substep_index) -> None:
        # MJWarp leaves derived poses at the pre-integration state.  Calls
        # 2..20 therefore expose post-states 1..19; the explicit final forward
        # below supplies post-state 20 without adding 20 redundant forwards.
        if substep_index > 0:
            self._cur_table_keepout |= self._table_keepout.sample(self.sim.data)

    def _latch_post_forward_table_keepout(self) -> None:
        self._cur_table_keepout |= self._table_keepout.sample(self.sim.data)

    @staticmethod
    def _quat_error_sq(torch, expected, actual):
        dot = torch.abs(torch.sum(expected * actual, dim=-1)).clamp(0.0, 1.0)
        return torch.square(2.0 * torch.acos(dot))

    @staticmethod
    def _quat_mul_wxyz(torch, left, right):
        lw, lx, ly, lz = left.unbind(dim=-1)
        rw, rx, ry, rz = right.unbind(dim=-1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )

    @staticmethod
    def _quat_apply_wxyz(torch, quaternion, vector):
        q_vector = quaternion[..., 1:]
        twice_cross = 2.0 * torch.cross(q_vector, vector, dim=-1)
        return vector + quaternion[..., :1] * twice_cross + torch.cross(
            q_vector, twice_cross, dim=-1
        )

    @classmethod
    def _aligned_teacher_body_pose(
        cls,
        torch,
        teacher_body_pos,
        teacher_body_quat,
        live_anchor_pos,
        live_anchor_quat,
        anchor_index,
    ):
        """Match MotionCommand's live-anchor x/y plus yaw alignment."""

        teacher_anchor_pos = teacher_body_pos[:, anchor_index]
        teacher_anchor_quat = teacher_body_quat[:, anchor_index]
        teacher_inverse = teacher_anchor_quat.clone()
        teacher_inverse[..., 1:].neg_()
        anchor_delta = cls._quat_mul_wxyz(
            torch, live_anchor_quat, teacher_inverse
        )
        sin_yaw = 2.0 * (
            anchor_delta[..., 0] * anchor_delta[..., 3]
            + anchor_delta[..., 1] * anchor_delta[..., 2]
        )
        cos_yaw = 1.0 - 2.0 * (
            torch.square(anchor_delta[..., 2])
            + torch.square(anchor_delta[..., 3])
        )
        yaw = torch.atan2(sin_yaw, cos_yaw)
        half_yaw = 0.5 * yaw
        delta_yaw = torch.zeros_like(anchor_delta)
        delta_yaw[..., 0] = torch.cos(half_yaw)
        delta_yaw[..., 3] = torch.sin(half_yaw)
        aligned_anchor_pos = live_anchor_pos.clone()
        aligned_anchor_pos[..., 2] = teacher_anchor_pos[..., 2]
        body_delta = teacher_body_pos - teacher_anchor_pos[:, None, :]
        expanded_yaw = delta_yaw[:, None, :].expand_as(teacher_body_quat)
        aligned_pos = aligned_anchor_pos[:, None, :] + cls._quat_apply_wxyz(
            torch, expanded_yaw, body_delta
        )
        aligned_quat = cls._quat_mul_wxyz(
            torch, expanded_yaw, teacher_body_quat
        )
        return aligned_pos, aligned_quat

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
        if self._fullmdp_initialized:
            actor_rows["epoch_phase_one_hot"] = torch.nn.functional.one_hot(
                self._epoch_phase, num_classes=10
            ).to(dtype=joint_pos_rel.dtype)
            actor_rows["epoch_task_valid"] = self._epoch_task_valid[:, None].to(
                dtype=joint_pos_rel.dtype
            )
            actor_rows["epoch_selected"] = self._epoch_selected[:, None].to(
                dtype=joint_pos_rel.dtype
            )
            actor_rows["epoch_launch_succeeded"] = self._epoch_launch_succeeded[
                :, None
            ].to(dtype=joint_pos_rel.dtype)
        else:
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

    def _fullmdp_reward20(self):
        torch = self._torch
        data = self.sim.data
        ids = self._fullmdp_body_ids
        body_pos = data.xpos[:, ids]
        body_quat = data.xquat[:, ids]
        body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        anchor = self._fullmdp_anchor_index
        raw = torch.stack(
            (
                torch.exp(
                    -torch.sum(
                        torch.square(
                            self._teacher_body_pos[:, anchor] - body_pos[:, anchor]
                        ),
                        dim=-1,
                    )
                    / FULLMDP_DENSE_REWARD_SPECS[0][2] ** 2
                ),
                torch.exp(
                    -self._quat_error_sq(
                        torch,
                        self._teacher_body_quat[:, anchor],
                        body_quat[:, anchor],
                    )
                    / FULLMDP_DENSE_REWARD_SPECS[1][2] ** 2
                ),
                torch.exp(
                    -torch.sum(
                        torch.square(self._aligned_teacher_body_pos - body_pos),
                        dim=-1,
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[2][2] ** 2
                ),
                torch.exp(
                    -self._quat_error_sq(
                        torch, self._aligned_teacher_body_quat, body_quat
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[3][2] ** 2
                ),
                torch.exp(
                    -torch.sum(
                        torch.square(self._teacher_body_lin_vel - body_lin_vel), dim=-1
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[4][2] ** 2
                ),
                torch.exp(
                    -torch.sum(
                        torch.square(self._teacher_body_ang_vel - body_ang_vel), dim=-1
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[5][2] ** 2
                ),
            ),
            dim=1,
        )
        configured = raw * self._fullmdp_dense_weights * self.step_dt
        terms = torch.zeros(
            (self.num_envs, 20), dtype=raw.dtype, device=self.device
        )
        terms[:, 14:] = configured
        reward = terms.sum(dim=1)
        if not bool(torch.isfinite(terms).all()):
            raise RuntimeError("FullMDP WAIT Reward20 is nonfinite")
        return reward, terms

    def _fullmdp_termination(self, st, requested_qdes):
        torch = self._torch
        timeout = self.episode_length_buf >= self.max_episode_length
        tilt = torch.acos((-st["proj_g"][:, 2]).clamp(-1.0, 1.0)) > 0.7
        low = st["base_pos"][:, 2] < 0.5
        qdes = ~torch.isfinite(requested_qdes).all(dim=1)
        keepout = self._cur_table_keepout
        resolved_table = self._cur_robot_table > 0
        bits = (
            timeout.to(torch.long) * FULLMDP_TERMINATION_BITS["time_out"]
            + tilt.to(torch.long) * FULLMDP_TERMINATION_BITS["base_fell_tilt"]
            + low.to(torch.long) * FULLMDP_TERMINATION_BITS["base_too_low"]
            + qdes.to(torch.long)
            * FULLMDP_TERMINATION_BITS["joint_qdes_forbidden"]
            + keepout.to(torch.long) * FULLMDP_TERMINATION_BITS["robot_hit_table"]
        )
        terminated = tilt | low | qdes | keepout | resolved_table
        # Isaac TerminationManager keeps timeout and physical termination as
        # independent masks.  RSL-RL needs the timeout bit even when a row also
        # falls or hits a guard in the same transition for value bootstrapping.
        truncated = timeout
        return terminated, truncated, bits, resolved_table

    def _clear_lifecycle(self, ids) -> None:
        self._epoch_phase[ids] = observation_contract.EPOCH_IDLE_PHASE_INDEX
        self._epoch_task_valid[ids] = False
        self._epoch_selected[ids] = False
        self._epoch_launch_succeeded[ids] = False
        self._cur_touched[ids] = 0.0
        self._cur_robot_table[ids] = 0.0
        self._cur_table_keepout[ids] = False

    def reset(self):
        observations, extras = super().reset()
        if self._fullmdp_initialized:
            self.reset_generation += 1
            self.last_terminal_bits.zero_()
            self._clear_lifecycle(self._all_env_ids)
            self._refresh_aligned_teacher_body_pose()
            self._compute_obs()
            observations = self.get_observations()
        return observations, extras

    def step(self, actions):
        torch = self._torch
        incoming = actions.to(self.device)
        requested_qdes = self.q_ready.unsqueeze(0) + self.act_scale * incoming
        finite_qdes = torch.isfinite(requested_qdes)
        safe_actions = torch.where(finite_qdes, incoming, self.actions)
        _st_before_forward, _tau_sq, _safe_qdes = self._advance_plant(safe_actions)
        # MuJoCo-Warp's step integrates qpos/qvel after its derived-tensor
        # forward pass.  Re-forward once so termination, Reward20, and the
        # returned observation all describe the same post-transition state.
        self.sim.forward()
        self._latch_post_forward_resolved_table_contacts()
        self._latch_post_forward_table_keepout()
        if self._cap_ok:
            self._probe_capacity("forward")
        st = self._state()
        terminated, truncated, terminal_bits, resolved_table_contact = self._fullmdp_termination(
            st, requested_qdes
        )
        reward, reward_terms = self._fullmdp_reward20()
        dones = terminated | truncated
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.last_terminal_bits[reset_ids] = terminal_bits[reset_ids]
            self._reset_idx(reset_ids)
            self.reset_generation[reset_ids] += 1
            self._clear_lifecycle(reset_ids)
        self._serve(self._all_env_ids)
        self.sim.forward()
        if self._cap_ok:
            self._probe_capacity("forward")
        self._refresh_aligned_teacher_body_pose()
        self._compute_obs()
        if self._cap_ok:
            self._capacity_gate(f"FullMDP WAIT step {self.common_step_counter}")
        extras = {
            "time_outs": truncated,
            "termination_bits": terminal_bits,
            "backend_resolved_table_contact": resolved_table_contact,
            "reward_terms": reward_terms,
            "reset_generation": self.reset_generation.clone(),
        }
        return self.get_observations(), reward, dones.long(), extras


__all__ = [
    "WAIT_BALL_PARK_HOPE",
    "READY_HOLD_PHASE_INDEX",
    "FULLMDP_TRACKED_BODY_NAMES",
    "FULLMDP_DENSE_REWARD_SPECS",
    "FULLMDP_TERMINATION_BITS",
    "FullMdpInitialWaitVecEnv",
]
