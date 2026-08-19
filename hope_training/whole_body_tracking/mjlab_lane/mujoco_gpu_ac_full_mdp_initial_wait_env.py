"""MuJoCo FullMDP WAIT plus an explicit incomplete single-shot A slice.

The plant, masked reset implementation, and ``sim.forward()`` remain the
tracked :class:`A3ReadyBallVecEnv` implementation.  WAIT parks the ball and
projects the post-forward live robot tensors into the shared ActionEpoch
observation order.  The opt-in A slice additionally reveals and launches one
deterministic question, publishes live Physical and R03 FK facts, and computes
Reward20 ordinals 0..9 plus the six common motion terms.

The narrow WAIT transition below advances the real plant and closes only the
IDLE Reward20/termination/masked-reset path.  An observed generic racket
contact is classified against the action-zero mount at the same physics
substep, so Reward ordinal 10 is live; R06, R07, and Reward ordinals 11..13
remain explicitly unavailable.
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
import action_ball_full_mdp_portable_reward as portable_reward
import action_ball_full_mdp_portable_catalog as portable_catalog
import racket_contact_geometry as racket_contact_geometry


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
        self._fullmdp_racket_body_id = int(self.mj_model.site_bodyid[self.racket_sid])
        self._fullmdp_racket_root_id = int(
            self.mj_model.body_rootid[self._fullmdp_racket_body_id]
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
            self._full_a_catalog = portable_catalog.load_portable_action_center_table()
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
        self._full_a_owner_valid_bits = torch.zeros(
            (n, portable_reward.OWNER_COUNT), dtype=torch.long, device=self.device
        )
        self._full_a_owner_fault_bits = torch.zeros_like(
            self._full_a_owner_valid_bits
        )
        self._full_a_owner_source_step = torch.full_like(
            self._full_a_owner_valid_bits, -1
        )
        self._full_a_owner_fact_f32 = torch.zeros(
            (
                n,
                portable_reward.OWNER_COUNT,
                portable_reward.OWNER_FACT_F32_WIDTH,
            ),
            dtype=dtype,
            device=self.device,
        )
        self._full_a_physical_present = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._full_a_physical_source_step = self._full_a_owner_source_step[:, 0]
        self._full_a_physical_fact_f32 = self._full_a_owner_fact_f32[:, 0]
        self._full_a_r03_present = torch.zeros_like(self._full_a_physical_present)
        self._full_a_r03_physically_valid = torch.zeros_like(
            self._full_a_physical_present
        )
        self._full_a_r03_armed = torch.zeros_like(self._full_a_physical_present)
        self._full_a_r03_expected_source_step = torch.full(
            (n,), -1, dtype=torch.long, device=self.device
        )
        self._full_a_r03_fact_f32 = self._full_a_owner_fact_f32[:, 1]
        self._full_a_launch_state_f32 = torch.zeros((n, 13), dtype=dtype, device=self.device)
        self._full_a_observation_ordinal = torch.zeros(
            n, dtype=torch.long, device=self.device
        )
        self._full_a_racket_contact = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._full_a_ball_table_contact = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_contact_center = torch.zeros((n, 3), dtype=dtype, device=self.device)
        self._full_a_outcome_code = torch.zeros(n, dtype=torch.long, device=self.device)
        action = self._full_a_catalog.fresh_action
        self._full_a_action_slot = torch.full(
            (n,), action.action_slot, dtype=torch.long, device=self.device
        )
        self._full_a_action_uid = torch.full(
            (n,), action.action_uid, dtype=torch.long, device=self.device
        )
        self._full_a_mount_normal_sign = torch.full(
            (n,), action.mount_normal_sign, dtype=torch.int8, device=self.device
        )
        self._full_a_contact_classification_status = torch.zeros(
            n, dtype=torch.int8, device=self.device
        )
        self._full_a_generic_contact_event = torch.zeros_like(
            self._full_a_physical_present
        )
        self._full_a_selected_contact_event = torch.zeros_like(
            self._full_a_physical_present
        )
        self._full_a_opposite_contact_event = torch.zeros_like(
            self._full_a_physical_present
        )
        self._full_a_edge_contact_event = torch.zeros_like(
            self._full_a_physical_present
        )
        self._full_a_between_contact_event = torch.zeros_like(
            self._full_a_physical_present
        )
        self._full_a_invalid_contact_event = torch.zeros_like(
            self._full_a_physical_present
        )
        self._clear_lifecycle(self._all_env_ids)

    def _full_a_begin_control_step(self) -> None:
        self._full_a_contact_classification_status.zero_()
        self._full_a_generic_contact_event.zero_()
        self._full_a_selected_contact_event.zero_()
        self._full_a_opposite_contact_event.zero_()
        self._full_a_edge_contact_event.zero_()
        self._full_a_between_contact_event.zero_()
        self._full_a_invalid_contact_event.zero_()

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

        r03 = self._full_a_r03_fact_f32[ids]
        r03.zero_()
        r03[:, 0:3] = racket[:, 0:3]
        r03[:, 3:6] = racket[:, 3:6]
        r03[:, 6:9] = racket[:, 6:9]
        r03[:, 9:12] = racket[:, 9:12]
        r03[:, 12:15] = racket[:, 12:15]
        target_finite = torch.isfinite(r03[:, :15]).all(dim=1)
        normal_unit = torch.isclose(
            torch.linalg.vector_norm(r03[:, 6:9], dim=1),
            torch.ones(n, dtype=r03.dtype, device=r03.device),
            rtol=0.0,
            atol=1.0e-4,
        )
        self._full_a_r03_armed[ids] = True
        self._full_a_r03_physically_valid[ids] = target_finite & normal_unit

        reveal = int(self.common_step_counter)
        launch = reveal + 1
        deadline = launch + max(1, int(round(FULL_A_FLIGHT_HORIZON_S / self.step_dt)))
        contact_tick = launch + torch.ceil(time_to_contact / self.step_dt).to(torch.long)
        self._epoch_clock_ticks[ids, 0] = reveal
        self._epoch_clock_ticks[ids, 1] = contact_tick
        self._epoch_clock_ticks[ids, 2] = launch
        self._epoch_clock_ticks[ids, 3] = deadline
        self._epoch_clock_ticks[ids, 4] = deadline + 1
        self._full_a_r03_expected_source_step[ids] = reveal + 1
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
        first_contact = active & racket_count.gt(0) & ~self._full_a_racket_contact
        center = self.sim.data.qpos[:, self.b_q : self.b_q + 3]
        site = self.sim.data.site_xpos[:, self.racket_sid]
        rotation = self.sim.data.site_xmat[:, self.racket_sid].reshape(
            self.num_envs, 3, 3
        )
        classification = (
            racket_contact_geometry.torch_classify_observed_generic_racket_contact(
                observed_generic_contact=first_contact,
                ball_center_w_m=center,
                racket_site_position_w_m=site,
                racket_rotation_w_from_local=rotation,
                mount_normal_sign=self._full_a_mount_normal_sign,
            )
        )
        self._full_a_contact_center.copy_(
            torch.where(first_contact[:, None], center, self._full_a_contact_center)
        )
        self._full_a_contact_classification_status.copy_(
            torch.where(
                first_contact,
                classification["status"],
                self._full_a_contact_classification_status,
            )
        )
        self._full_a_generic_contact_event |= first_contact
        self._full_a_selected_contact_event |= classification["selected"]
        self._full_a_opposite_contact_event |= classification["opposite"]
        self._full_a_edge_contact_event |= classification[
            "edge_or_rim_ambiguous"
        ]
        self._full_a_between_contact_event |= classification[
            "between_planes_ambiguous"
        ]
        self._full_a_invalid_contact_event |= classification["invalid"]
        self._full_a_racket_contact |= active & racket_count.gt(0)
        self._full_a_ball_table_contact |= active & table_count.gt(0)

    def _full_a_racket_kinematics(self):
        """Return live scene-local racket site position, velocity, and raw +Y."""

        data = self.sim.data
        position = data.site_xpos[:, self.racket_sid]
        rotation = data.site_xmat[:, self.racket_sid].reshape(
            self.num_envs, 3, 3
        )
        angular = data.cvel[:, self._fullmdp_racket_body_id, :3]
        linear_at_subtree_com = data.cvel[
            :, self._fullmdp_racket_body_id, 3:
        ]
        subtree_com = data.subtree_com[:, self._fullmdp_racket_root_id]
        velocity = linear_at_subtree_com + self._torch.cross(
            angular, position - subtree_com, dim=1
        )
        origins = self.env.scene.env_origins
        normal = rotation[:, :, 1]
        normal = normal / self._torch.linalg.vector_norm(
            normal, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        return position - origins, velocity, normal

    def _full_a_publish_r03_fact(self):
        """Publish the armed question against one real post-physics racket FK."""

        due = self._full_a_r03_armed & (
            self._full_a_r03_expected_source_step
            <= int(self.common_step_counter)
        )
        position, velocity, normal = self._full_a_racket_kinematics()
        achieved_finite = (
            self._torch.isfinite(position).all(dim=1)
            & self._torch.isfinite(velocity).all(dim=1)
            & self._torch.isfinite(normal).all(dim=1)
        )
        safe = due & achieved_finite
        fact = self._full_a_r03_fact_f32
        zero = self._torch.zeros_like(position)
        fact[:, 15:18] = self._torch.where(safe[:, None], position, zero)
        fact[:, 18:21] = self._torch.where(safe[:, None], velocity, zero)
        fact[:, 21:24] = self._torch.where(safe[:, None], normal, zero)
        bits = (
            safe.to(self._torch.long) * portable_reward.R03_PRESENT
            + (safe & self._full_a_r03_physically_valid).to(self._torch.long)
            * portable_reward.R03_PHYSICALLY_VALID
        )
        self._full_a_owner_valid_bits[:, 1] = bits
        self._full_a_owner_fault_bits[:, 1] = (
            due & ~achieved_finite
        ).to(self._torch.long)
        self._full_a_owner_source_step[:, 1] = self._torch.where(
            safe,
            self._torch.full_like(
                self._full_a_r03_expected_source_step,
                int(self.common_step_counter),
            ),
            self._torch.full_like(self._full_a_r03_expected_source_step, -1),
        )
        self._full_a_r03_present.copy_(safe)
        # The current task remains armed for every next transition until its
        # selected reset, matching the Isaac Racket command tail rather than
        # freezing one early FK sample for the whole shot.
        self._full_a_r03_armed.copy_(self._epoch_task_valid)
        self._full_a_r03_expected_source_step.copy_(
            self._torch.where(
                self._epoch_task_valid,
                self._torch.full_like(
                    self._full_a_r03_expected_source_step,
                    int(self.common_step_counter) + 1,
                ),
                self._torch.full_like(self._full_a_r03_expected_source_step, -1),
            )
        )
        return safe, safe & self._full_a_r03_physically_valid

    def _full_a_publish_physical_fact(self) -> None:
        active = self._epoch_task_valid & self._epoch_launch_succeeded
        center = self.sim.data.qpos[:, self.b_q : self.b_q + 3]
        values = self._full_a_physical_fact_f32
        values[:, :3] = self._torch.where(active[:, None], center, self._torch.zeros_like(center))
        values[:, 3:6] = self._torch.where(
            self._full_a_racket_contact[:, None],
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
        self._full_a_owner_valid_bits[:, 0] = (
            active.to(self._torch.long) * portable_reward.PHYSICAL_PRESENT
            + self._full_a_selected_contact_event.to(self._torch.long)
            * portable_reward.PHYSICAL_SELECTED_CONTACT
        )
        self._full_a_owner_fault_bits[:, 0] = (
            self._full_a_invalid_contact_event.to(self._torch.long)
        )
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
            present = self._torch.bitwise_and(
                self._full_a_owner_valid_bits, 1
            ).ne(0)
            age = torch.where(
                present,
                (
                    int(self.common_step_counter)
                    - self._full_a_owner_source_step
                ).to(joint_pos_rel.dtype) * self.step_dt,
                torch.zeros_like(self._full_a_owner_source_step, dtype=joint_pos_rel.dtype),
            )
            critic_rows["physical_r03_r06_r07_fact_present"] = present.to(
                joint_pos_rel.dtype
            )
            critic_rows["physical_r03_r06_r07_fact_age_s"] = age
            critic_rows["physical_r03_r06_r07_fact_f32"] = self._full_a_owner_fact_f32.reshape(
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
        if getattr(self, "full_a_mode", False):
            terms[:, :14] = portable_reward.lifecycle_reward14(
                valid_bits=self._full_a_owner_valid_bits,
                fact_f32=self._full_a_owner_fact_f32,
                owner_fault_bits=self._full_a_owner_fault_bits,
                step_dt=self.step_dt,
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
            self._full_a_owner_valid_bits[ids] = 0
            self._full_a_owner_fault_bits[ids] = 0
            self._full_a_owner_source_step[ids] = -1
            self._full_a_owner_fact_f32[ids] = 0.0
            self._full_a_r03_present[ids] = False
            self._full_a_r03_physically_valid[ids] = False
            self._full_a_r03_armed[ids] = False
            self._full_a_r03_expected_source_step[ids] = -1
            self._full_a_launch_state_f32[ids] = 0.0
            self._full_a_observation_ordinal[ids] = 0
            self._full_a_racket_contact[ids] = False
            self._full_a_ball_table_contact[ids] = False
            self._full_a_contact_center[ids] = 0.0
            self._full_a_outcome_code[ids] = FULL_A_OUTCOME_NONE
            self._full_a_contact_classification_status[ids] = (
                racket_contact_geometry.OBSERVED_RUBBER_STATUS_NONE
            )
            self._full_a_generic_contact_event[ids] = False
            self._full_a_selected_contact_event[ids] = False
            self._full_a_opposite_contact_event[ids] = False
            self._full_a_edge_contact_event[ids] = False
            self._full_a_between_contact_event[ids] = False
            self._full_a_invalid_contact_event[ids] = False

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
        self._full_a_begin_control_step()
        reveal_event, launch_event = self._full_a_prepare_step()
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
        self._full_a_publish_physical_fact()
        r03_present_event, r03_valid_event = self._full_a_publish_r03_fact()
        reward, reward_terms = self._fullmdp_reward20()
        shot_terminal, outcome = self._full_a_settle_outcome(st)
        contact_event = self._full_a_generic_contact_event.clone()
        terminated |= shot_terminal
        dones = terminated | truncated
        selected_reset = dones & self._epoch_selected
        terminal_phase = self._epoch_phase.clone()
        racket_contact = self._full_a_racket_contact.clone()
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
            "full_a_racket_contact": racket_contact,
            "full_a_ball_table_contact": table_contact,
            "full_a_physical_current_center": physical_center,
            "full_a_reveal_event": reveal_event,
            "full_a_launch_event": launch_event,
            "full_a_flight_terminal_event": shot_terminal,
            "full_a_selected_reset_event": selected_reset,
            "full_a_racket_contact_eligible_event": launch_event,
            "full_a_racket_contact_event": contact_event,
            "full_a_action_slot": self._full_a_action_slot.clone(),
            "full_a_action_uid": self._full_a_action_uid.clone(),
            "full_a_mount_normal_sign": self._full_a_mount_normal_sign.clone(),
            "full_a_contact_classification_status": (
                self._full_a_contact_classification_status.clone()
            ),
            "full_a_selected_contact_event": (
                self._full_a_selected_contact_event.clone()
            ),
            "full_a_opposite_contact_event": (
                self._full_a_opposite_contact_event.clone()
            ),
            "full_a_edge_contact_event": self._full_a_edge_contact_event.clone(),
            "full_a_between_contact_event": (
                self._full_a_between_contact_event.clone()
            ),
            "full_a_invalid_contact_event": (
                self._full_a_invalid_contact_event.clone()
            ),
            "full_a_r03_present_event": r03_present_event,
            "full_a_r03_physically_valid_event": r03_valid_event,
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
