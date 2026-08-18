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
FULL_A_PHASE_IDLE = 0
FULL_A_PHASE_REVEAL_COMMITTED = 2
FULL_A_PHASE_LAUNCH_SETTLED = 5
FULL_A_PHASE_OUTCOME_SETTLED = 6
FULL_A_OUTCOME_NONE = 0
FULL_A_OUTCOME_FLIGHT_EXPIRED = 1
FULL_A_OUTCOME_BALL_DEAD = 2
FULL_A_FLIGHT_HORIZON_S = 1.0
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
        full_a_mode: bool = False,
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
        if type(full_a_mode) is not bool:
            raise TypeError("full_a_mode must be bool")
        self.full_a_mode = full_a_mode
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
        if self.full_a_mode:
            self._initialize_full_a_state()
        self._snapshot_ready_teacher()
        self._fullmdp_initialized = True
        self._compute_obs()

    def _initialize_full_a_state(self) -> None:
        """Allocate the one row-wise state used by the optional A slice."""

        torch = self._torch
        n = self.num_envs
        dtype = self.qpos_init.dtype
        self._epoch_task_f32 = torch.zeros(
            (n, observation_contract.TASK_F32_WIDTH), dtype=dtype, device=self.device
        )
        self._epoch_clock_ticks = torch.full(
            (n, 5), -1, dtype=torch.long, device=self.device
        )
        self._full_a_physical_present = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._full_a_physical_source_step = torch.full(
            (n,), -1, dtype=torch.long, device=self.device
        )
        self._full_a_physical_fact_f32 = torch.zeros(
            (n, observation_contract.OWNER_FACT_F32_WIDTH),
            dtype=dtype,
            device=self.device,
        )
        self._full_a_launch_state_f32 = torch.zeros((n, 13), dtype=dtype, device=self.device)
        self._full_a_observation_ordinal = torch.zeros(
            n, dtype=torch.long, device=self.device
        )
        self._full_a_selected_contact = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._full_a_ball_table_contact = torch.zeros_like(
            self._full_a_selected_contact
        )
        self._full_a_contact_center = torch.zeros((n, 3), dtype=dtype, device=self.device)
        self._full_a_outcome_code = torch.zeros(n, dtype=torch.long, device=self.device)
        self._clear_lifecycle(self._all_env_ids)

    def _full_a_reveal_rows(self, ids) -> None:
        """Publish one deterministic, live-geometry question for selected rows."""

        torch = self._torch
        n = int(ids.numel())
        if n == 0:
            return
        launch_pos = 0.5 * (self.serve_pos_lo + self.serve_pos_hi)
        launch_vel = 0.5 * (self.serve_vel_lo + self.serve_vel_hi)
        state = torch.zeros((n, 13), dtype=self.qpos_init.dtype, device=self.device)
        state[:, :3] = launch_pos + self.hope_to_scene
        state[:, 3] = 1.0
        state[:, 7:10] = launch_vel
        self._full_a_launch_state_f32[ids] = state

        origins = self.env.scene.env_origins[ids]
        racket_env = self.sim.data.site_xpos[ids, self.racket_sid] - origins
        base_env = self.sim.data.qpos[ids, self.root_qadr : self.root_qadr + 3] - origins
        speed_x = launch_vel[0]
        time_to_contact = torch.clamp(
            (racket_env[:, 0] - state[:, 0]) / speed_x,
            min=self.step_dt,
            max=0.8 * FULL_A_FLIGHT_HORIZON_S,
        )
        contact = state[:, :3] + state[:, 7:10] * time_to_contact[:, None]
        contact[:, 2] -= 0.5 * 9.81 * time_to_contact.square()
        contact_velocity = state[:, 7:10].clone()
        contact_velocity[:, 2] -= 9.81 * time_to_contact
        normal = -contact_velocity
        normal /= torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(1.0e-6)

        task = torch.zeros(
            (n, observation_contract.TASK_F32_WIDTH),
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        task[:, :5] = torch.stack(
            (
                time_to_contact,
                torch.ones_like(time_to_contact),
                time_to_contact,
                torch.full_like(time_to_contact, FULL_A_FLIGHT_HORIZON_S),
                torch.full_like(time_to_contact, self.step_dt),
            ),
            dim=1,
        )
        racket = task[:, 5:32]
        racket[:, 0:3] = contact
        racket[:, 3:6] = contact_velocity
        racket[:, 6:9] = normal
        racket[:, 9:12] = contact
        racket[:, 12:15] = contact_velocity
        racket[:, 15] = 1.0
        racket[:, 19:21] = base_env[:, :2]
        racket[:, 21:24] = state[:, 7:10]
        task[:, 32:] = state
        self._epoch_task_f32[ids] = task

        reveal = int(self.common_step_counter)
        launch = reveal + 1
        deadline = launch + max(1, int(round(FULL_A_FLIGHT_HORIZON_S / self.step_dt)))
        contact_tick = launch + torch.ceil(time_to_contact / self.step_dt).to(torch.long)
        self._epoch_clock_ticks[ids, 0] = reveal
        self._epoch_clock_ticks[ids, 1] = contact_tick
        self._epoch_clock_ticks[ids, 2] = launch
        self._epoch_clock_ticks[ids, 3] = deadline
        self._epoch_clock_ticks[ids, 4] = deadline + 1
        self._epoch_phase[ids] = FULL_A_PHASE_REVEAL_COMMITTED
        self._epoch_task_valid[ids] = True
        self._epoch_selected[ids] = True

    def _full_a_launch_rows(self, ids) -> None:
        state = self._full_a_launch_state_f32[ids]
        data = self.sim.data
        data.qpos[ids, self.b_q : self.b_q + 3] = state[:, :3]
        data.qpos[ids, self.b_q + 3 : self.b_q + 7] = state[:, 3:7]
        data.qvel[ids, self.b_v : self.b_v + 3] = state[:, 7:10]
        data.qvel[ids, self.b_v + 3 : self.b_v + 6] = state[:, 10:13]
        self.ball_age_buf[ids] = 0
        self._epoch_phase[ids] = FULL_A_PHASE_LAUNCH_SETTLED
        self._epoch_launch_succeeded[ids] = True

    def _full_a_prepare_step(self):
        idle = self._epoch_phase.eq(FULL_A_PHASE_IDLE)
        self._full_a_reveal_rows(idle.nonzero(as_tuple=False).squeeze(-1))
        launch = self._epoch_phase.eq(FULL_A_PHASE_REVEAL_COMMITTED) & (
            self._epoch_clock_ticks[:, 2] <= int(self.common_step_counter)
        )
        self._full_a_launch_rows(launch.nonzero(as_tuple=False).squeeze(-1))
        return idle, launch

    def _full_a_latch_ball_contacts(self) -> None:
        """Latch only contacts read from the live MuJoCo contact arrays."""

        torch = self._torch
        geom = self._con_geom[:]
        valid = self._con_idx < self._nacon[0]
        g0, g1 = geom[:, 0], geom[:, 1]
        ball0, ball1 = g0.eq(self._ball_gid), g1.eq(self._ball_gid)
        other = torch.where(ball0, g1, g0).long()
        ball = valid & (ball0 | ball1)
        world = self._con_world[:].long().clamp_(0, self.num_envs - 1)
        racket_count = torch.zeros(self.num_envs, device=self.device)
        table_count = torch.zeros_like(racket_count)
        racket_count.scatter_add_(0, world, (ball & self._geom_class[other].eq(1)).float())
        table_count.scatter_add_(0, world, (ball & self._geom_class[other].eq(2)).float())
        active = self._epoch_phase.eq(FULL_A_PHASE_LAUNCH_SETTLED)
        first_contact = active & racket_count.gt(0) & ~self._full_a_selected_contact
        center = self.sim.data.qpos[:, self.b_q : self.b_q + 3]
        self._full_a_contact_center.copy_(
            torch.where(first_contact[:, None], center, self._full_a_contact_center)
        )
        self._full_a_selected_contact |= active & racket_count.gt(0)
        self._full_a_ball_table_contact |= active & table_count.gt(0)

    def _full_a_publish_physical_fact(self) -> None:
        active = self._epoch_task_valid & self._epoch_launch_succeeded
        center = self.sim.data.qpos[:, self.b_q : self.b_q + 3]
        values = self._full_a_physical_fact_f32
        values[:, :3] = self._torch.where(active[:, None], center, self._torch.zeros_like(center))
        values[:, 3:6] = self._torch.where(
            self._full_a_selected_contact[:, None],
            self._full_a_contact_center,
            self._torch.zeros_like(self._full_a_contact_center),
        )
        values[:, 6:9] = values[:, 3:6]
        self._full_a_observation_ordinal += active.to(self._torch.long)
        values[:, 9] = self._torch.where(
            active,
            self._full_a_observation_ordinal.to(values.dtype),
            self._torch.zeros_like(values[:, 9]),
        )
        self._full_a_physical_present.copy_(active)
        self._full_a_physical_source_step.copy_(
            self._torch.where(
                active,
                self._torch.full_like(
                    self._full_a_physical_source_step, int(self.common_step_counter)
                ),
                self._torch.full_like(self._full_a_physical_source_step, -1),
            )
        )

    def _full_a_settle_outcome(self, st):
        torch = self._torch
        active = self._epoch_phase.eq(FULL_A_PHASE_LAUNCH_SETTLED)
        ball_hope = st["ball_pos"] - self.hope_to_scene
        dead = active & (
            (ball_hope[:, 2] < self.cfg.ball_dead_z_hope)
            | (ball_hope[:, 0] < self.cfg.ball_dead_x_lo_hope)
            | (ball_hope[:, 0] > self.cfg.ball_dead_x_hi_hope)
            | ~torch.isfinite(ball_hope).all(dim=1)
        )
        expired = active & (
            int(self.common_step_counter) >= self._epoch_clock_ticks[:, 3]
        )
        outcome = torch.where(
            dead,
            torch.full_like(self._full_a_outcome_code, FULL_A_OUTCOME_BALL_DEAD),
            torch.where(
                expired,
                torch.full_like(
                    self._full_a_outcome_code, FULL_A_OUTCOME_FLIGHT_EXPIRED
                ),
                torch.zeros_like(self._full_a_outcome_code),
            ),
        )
        settled = outcome.ne(FULL_A_OUTCOME_NONE)
        self._full_a_outcome_code.copy_(outcome)
        self._epoch_phase.copy_(
            torch.where(
                settled,
                torch.full_like(self._epoch_phase, FULL_A_PHASE_OUTCOME_SETTLED),
                self._epoch_phase,
            )
        )
        return settled, outcome

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
            if getattr(self, "full_a_mode", False) and self._fullmdp_initialized:
                self._full_a_latch_ball_contacts()

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
        if bool(ball_contact.any()) and not getattr(self, "full_a_mode", False):
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
            if self.full_a_mode:
                actor_rows["epoch_task_f32"] = self._epoch_task_f32
                actor_rows["epoch_clock_remaining_s"] = (
                    self._epoch_clock_ticks - int(self.common_step_counter)
                ).to(dtype=joint_pos_rel.dtype) * self.step_dt
        else:
            actor_rows["epoch_phase_one_hot"][
                :, observation_contract.EPOCH_IDLE_PHASE_INDEX
            ] = 1.0
        policy = observation_contract.concatenate_layout_rows(
            observation_contract.ACTOR_LAYOUT_V1, actor_rows
        )
        critic_rows = {
                name: torch.zeros(
                    (self.num_envs, width),
                    dtype=joint_pos_rel.dtype,
                    device=self.device,
                )
                for name, width in observation_contract.CRITIC_EXTENSION_LAYOUT_V1
            }
        if self._fullmdp_initialized and self.full_a_mode:
            present = torch.zeros(
                (self.num_envs, 4), dtype=torch.bool, device=self.device
            )
            present[:, 0] = self._full_a_physical_present
            age = torch.zeros(
                (self.num_envs, 4), dtype=joint_pos_rel.dtype, device=self.device
            )
            age[:, 0] = torch.where(
                self._full_a_physical_present,
                (
                    int(self.common_step_counter)
                    - self._full_a_physical_source_step
                ).to(joint_pos_rel.dtype) * self.step_dt,
                torch.zeros_like(age[:, 0]),
            )
            facts = torch.zeros(
                (self.num_envs, 4, observation_contract.OWNER_FACT_F32_WIDTH),
                dtype=joint_pos_rel.dtype,
                device=self.device,
            )
            facts[:, 0] = self._full_a_physical_fact_f32
            critic_rows["physical_r03_r06_r07_fact_present"] = present.to(
                joint_pos_rel.dtype
            )
            critic_rows["physical_r03_r06_r07_fact_age_s"] = age
            critic_rows["physical_r03_r06_r07_fact_f32"] = facts.reshape(
                self.num_envs, -1
            )
        critic_extension = observation_contract.concatenate_layout_rows(
            observation_contract.CRITIC_EXTENSION_LAYOUT_V1, critic_rows
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
        if getattr(self, "full_a_mode", False) and hasattr(self, "_epoch_task_f32"):
            self._epoch_task_f32[ids] = 0.0
            self._epoch_clock_ticks[ids] = -1
            self._full_a_physical_present[ids] = False
            self._full_a_physical_source_step[ids] = -1
            self._full_a_physical_fact_f32[ids] = 0.0
            self._full_a_launch_state_f32[ids] = 0.0
            self._full_a_observation_ordinal[ids] = 0
            self._full_a_selected_contact[ids] = False
            self._full_a_ball_table_contact[ids] = False
            self._full_a_contact_center[ids] = 0.0
            self._full_a_outcome_code[ids] = FULL_A_OUTCOME_NONE

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
        if getattr(self, "full_a_mode", False):
            return self._step_full_a(actions)
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

    def _step_full_a(self, actions):
        """Advance the one real reveal/launch/flight/outcome vertical slice."""

        torch = self._torch
        reveal_event, launch_event = self._full_a_prepare_step()
        contact_before = self._full_a_selected_contact.clone()
        incoming = actions.to(self.device)
        requested_qdes = self.q_ready.unsqueeze(0) + self.act_scale * incoming
        finite_qdes = torch.isfinite(requested_qdes)
        safe_actions = torch.where(finite_qdes, incoming, self.actions)
        self._advance_plant(safe_actions)
        self.sim.forward()
        self._latch_post_forward_resolved_table_contacts()
        self._latch_post_forward_table_keepout()
        self._full_a_latch_ball_contacts()
        if self._cap_ok:
            self._probe_capacity("forward")
        st = self._state()
        terminated, truncated, terminal_bits, resolved_table_contact = (
            self._fullmdp_termination(st, requested_qdes)
        )
        reward, reward_terms = self._fullmdp_reward20()
        self._full_a_publish_physical_fact()
        shot_terminal, outcome = self._full_a_settle_outcome(st)
        contact_event = self._full_a_selected_contact & ~contact_before
        terminated |= shot_terminal
        dones = terminated | truncated
        selected_reset = dones & self._epoch_selected
        terminal_phase = self._epoch_phase.clone()
        selected_contact = self._full_a_selected_contact.clone()
        table_contact = self._full_a_ball_table_contact.clone()
        physical_center = self._full_a_physical_fact_f32[:, :3].clone()
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.last_terminal_bits[reset_ids] = terminal_bits[reset_ids]
            self._reset_idx(reset_ids)
            self.reset_generation[reset_ids] += 1
            self._clear_lifecycle(reset_ids)
            self.sim.forward()
            if self._cap_ok:
                self._probe_capacity("forward")
        self._refresh_aligned_teacher_body_pose()
        self._compute_obs()
        if self._cap_ok:
            self._capacity_gate(f"FullMDP A step {self.common_step_counter}")
        extras = {
            "time_outs": truncated,
            "termination_bits": terminal_bits,
            "backend_resolved_table_contact": resolved_table_contact,
            "reward_terms": reward_terms,
            "reset_generation": self.reset_generation.clone(),
            "full_a_phase_before_reset": terminal_phase,
            "full_a_outcome_code": outcome,
            "full_a_selected_contact": selected_contact,
            "full_a_ball_table_contact": table_contact,
            "full_a_physical_current_center": physical_center,
            "full_a_reveal_event": reveal_event,
            "full_a_launch_event": launch_event,
            "full_a_flight_terminal_event": shot_terminal,
            "full_a_selected_reset_event": selected_reset,
            "full_a_contact_eligible_event": launch_event,
            "full_a_selected_contact_event": contact_event,
        }
        return self.get_observations(), reward, dones.long(), extras


__all__ = [
    "WAIT_BALL_PARK_HOPE",
    "READY_HOLD_PHASE_INDEX",
    "FULL_A_PHASE_IDLE",
    "FULL_A_PHASE_REVEAL_COMMITTED",
    "FULL_A_PHASE_LAUNCH_SETTLED",
    "FULL_A_PHASE_OUTCOME_SETTLED",
    "FULL_A_OUTCOME_NONE",
    "FULL_A_OUTCOME_FLIGHT_EXPIRED",
    "FULL_A_OUTCOME_BALL_DEAD",
    "FULLMDP_TRACKED_BODY_NAMES",
    "FULLMDP_DENSE_REWARD_SPECS",
    "FULLMDP_TERMINATION_BITS",
    "FullMdpInitialWaitVecEnv",
]
