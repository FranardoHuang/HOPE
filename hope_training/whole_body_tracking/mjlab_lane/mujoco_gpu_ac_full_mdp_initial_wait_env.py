"""MuJoCo FullMDP WAIT plus an explicit incomplete single-shot A slice.

The plant, masked reset implementation, and ``sim.forward()`` remain the
tracked :class:`A3ReadyBallVecEnv` implementation.  WAIT parks the ball and
projects the post-forward live robot tensors into the shared ActionEpoch
observation order.  The opt-in A slice additionally reveals and launches one
deterministic question, publishes live Physical/R03/R06/R07 facts, and computes
their Reward20 terms plus the six common motion terms.

The narrow WAIT transition below advances the real plant and closes only the
IDLE Reward20/termination/masked-reset path.  An observed generic racket
contact is classified against the action-zero mount at the same physics
substep.  R06 consumes only a measured descending landing crossing and R07
consumes only the fixed post-outcome recovery window; neither is a readiness
claim for Full MuJoCo A.
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    from .a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
    from . import mujoco_full_mdp_portable_question as portable_question
    from . import mujoco_full_mdp_portable_outcome as portable_outcome
    from .mujoco_gpu_ac_table_keepout import DeviceExactTableKeepout
except ImportError:  # Direct execution with mjlab_lane on PYTHONPATH.
    from a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
    import mujoco_full_mdp_portable_question as portable_question
    import mujoco_full_mdp_portable_outcome as portable_outcome
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
FULL_A_PHASE_RECOVERY_SETTLED = 7
FULL_A_OUTCOME_NONE = portable_outcome.OUTCOME_NONE
FULL_A_OUTCOME_FLIGHT_EXPIRED = portable_outcome.OUTCOME_FLIGHT_EXPIRED
FULL_A_OUTCOME_BALL_DEAD = portable_outcome.OUTCOME_BALL_DEAD
FULL_A_OUTCOME_LEGAL_LANDING = portable_outcome.OUTCOME_LEGAL_LANDING
FULL_A_OUTCOME_OWN_TABLE_LANDING = portable_outcome.OUTCOME_OWN_TABLE_LANDING
FULL_A_OUTCOME_OUT = portable_outcome.OUTCOME_OUT
FULL_A_OUTCOME_INVALID = portable_outcome.OUTCOME_INVALID
FULL_A_FLIGHT_HORIZON_S = 1.0
FULL_A_RECOVERY_START_AGE_TICK = portable_outcome.RECOVERY_START_AGE_TICK
FULL_A_RECOVERY_END_AGE_TICK = portable_outcome.RECOVERY_END_AGE_TICK
FULL_A_PLACEMENT_BROAD_SIGMA_M = portable_outcome.PLACEMENT_BROAD_SIGMA_M
FULL_A_SUPPORT_FORCE_N = 10.0
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
        if full_a_mode and float(task_cfg.episode_length_s) < 10.0:
            raise ValueError(
                "Full-A requires the shared 10s episode horizon so one "
                "question, flight, and recovery can complete"
            )
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
            self._initialize_full_a_geometry(mujoco)
            self._initialize_full_a_state()
            self._full_a_teacher = portable_question.load_portable_motion_teacher(
                row=self._full_a_catalog.fresh_action,
                tracked_body_names=FULLMDP_TRACKED_BODY_NAMES,
                torch=self._torch,
                dtype=self.qpos_init.dtype,
                device=self.device,
            )
        self._snapshot_ready_teacher()
        self._fullmdp_initialized = True
        self._compute_obs()

    def _initialize_full_a_geometry(self, mujoco) -> None:
        """Bind R06/R07 numeric geometry to the compiled MuJoCo scene."""

        torch = self._torch
        net = int(
            mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "court_net"
            )
        )
        if self._table_gid < 0 or net < 0 or self._ball_gid < 0:
            raise RuntimeError("FullMDP landing geometry is unavailable")
        table_position = self.mj_model.geom_pos[self._table_gid]
        table_half_size = self.mj_model.geom_size[self._table_gid]
        net_position = self.mj_model.geom_pos[net]
        net_half_size = self.mj_model.geom_size[net]
        ball_radius = float(self.mj_model.geom_size[self._ball_gid][0])
        target = tuple(self._full_a_catalog.landing_aim_center_w_xy_m)
        x_min = float(table_position[0] - table_half_size[0])
        x_max = float(table_position[0] + table_half_size[0])
        y_min = float(table_position[1] - table_half_size[1])
        y_max = float(table_position[1] + table_half_size[1])
        net_x = float(net_position[0])
        table_surface = float(table_position[2] + table_half_size[2])
        net_top = float(net_position[2] + net_half_size[2])
        if not (
            ball_radius > 0.0
            and x_min < net_x < x_max
            and y_min < float(target[1]) < y_max
            and net_x < float(target[0]) < x_max
            and net_top > table_surface
        ):
            raise RuntimeError("FullMDP landing target/compiled geometry differs")
        dtype = self.qpos_init.dtype
        self._full_a_landing_target_xy = torch.tensor(
            target, dtype=dtype, device=self.device
        )
        self._full_a_target_positive_x = float(target[0]) > net_x
        self._full_a_table_bounds = (x_min, x_max, y_min, y_max)
        self._full_a_table_surface_z = table_surface
        self._full_a_net_x = net_x
        self._full_a_net_clear_z = net_top + ball_radius
        self._full_a_landing_plane_z = table_surface + ball_radius
        self._full_a_placement_broad_sigma = FULL_A_PLACEMENT_BROAD_SIGMA_M
        self._full_a_placement_narrow_sigma = 2.0 * ball_radius
        body_ids = self.mj_model.geom_bodyid
        foot_body_ids = (
            int(self._fullmdp_body_ids[3]),
            int(self._fullmdp_body_ids[6]),
        )
        foot_class = torch.zeros(
            int(self.mj_model.ngeom), dtype=torch.int8, device=self.device
        )
        for geom_id, body_id in enumerate(body_ids):
            if int(body_id) == foot_body_ids[0]:
                foot_class[geom_id] = 1
            elif int(body_id) == foot_body_ids[1]:
                foot_class[geom_id] = 2
        floor = int(
            mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "robot/floor"
            )
        )
        if floor < 0 or not bool(foot_class.ne(0).any()):
            raise RuntimeError("FullMDP recovery foot/floor geometry is unavailable")
        self._full_a_floor_geom_id = floor
        self._full_a_foot_geom_class = foot_class
        self._initialize_full_a_contact_force_probe()

    def _initialize_full_a_contact_force_probe(self) -> None:
        """Allocate the exact MuJoCo constraint-force output used by R07."""

        try:
            import mujoco_warp as mjwarp
            import warp as wp

            contact_ids_torch = self._torch.arange(
                self._naconmax,
                dtype=self._torch.int32,
                device=self.device,
            )
            contact_ids = wp.from_torch(contact_ids_torch, dtype=wp.int32)
            force = wp.zeros(
                self._naconmax,
                dtype=wp.spatial_vector,
                device=str(self.device),
            )
            force_torch = wp.to_torch(force)
        except Exception as exc:
            raise RuntimeError(
                "FullMDP R07 requires live MuJoCo contact normal forces"
            ) from exc
        if tuple(force_torch.shape) != (self._naconmax, 6):
            raise RuntimeError("MuJoCo contact-force output shape drifted")
        self._full_a_mjwarp = mjwarp
        self._full_a_contact_ids_wp = contact_ids
        self._full_a_contact_force_wp = force
        self._full_a_contact_force_f32 = force_torch

    def _full_a_contact_normal_force(self):
        """Return the live contact-frame normal force for every contact row."""

        self._full_a_mjwarp.contact_force(
            self.sim.wp_model,
            self.sim.wp_data,
            self._full_a_contact_ids_wp,
            False,
            self._full_a_contact_force_wp,
        )
        return self._full_a_contact_force_f32[:, 0]

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
        self._full_a_selected_racket_contact = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_contact_center = torch.zeros((n, 3), dtype=dtype, device=self.device)
        self._full_a_previous_ball_center = torch.zeros(
            (n, 3), dtype=dtype, device=self.device
        )
        self._full_a_previous_ball_center_valid = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_net_crossed = torch.zeros_like(self._full_a_racket_contact)
        self._full_a_net_clear = torch.zeros_like(self._full_a_racket_contact)
        self._full_a_landing_crossing_present = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_landing_crossing_xy = torch.zeros(
            (n, 2), dtype=dtype, device=self.device
        )
        self._full_a_landing_on_table = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_landing_opponent_bound = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_landing_on_opponent = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_r06_payment_event = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_outcome_code = torch.zeros(n, dtype=torch.long, device=self.device)
        self._full_a_recovery_origin_step = torch.full(
            (n,), -1, dtype=torch.long, device=self.device
        )
        self._full_a_recovery_ready_streak = torch.zeros(
            n, dtype=torch.long, device=self.device
        )
        self._full_a_recovery_ready_seen = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_recovery_expected_count = torch.zeros(
            n, dtype=torch.long, device=self.device
        )
        self._full_a_recovery_eligible_count = torch.zeros_like(
            self._full_a_recovery_expected_count
        )
        self._full_a_recovery_last_age = torch.full_like(
            self._full_a_recovery_expected_count, -1
        )
        self._full_a_recovery_sticky_fault = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_selected_reset_frame0_carry = torch.zeros_like(
            self._full_a_racket_contact
        )
        self._full_a_recovery_component_scales = torch.tensor(
            portable_outcome.RECOVERY_COMPONENT_SCALES,
            dtype=dtype,
            device=self.device,
        )
        self._full_a_recovery_ready_tolerances = torch.tensor(
            portable_outcome.RECOVERY_READY_TOLERANCES,
            dtype=dtype,
            device=self.device,
        )
        self._full_a_teacher_rate = torch.ones(n, dtype=dtype, device=self.device)
        self._full_a_scaled_t_hit_s = torch.zeros(n, dtype=dtype, device=self.device)
        self._full_a_scaled_t_cycle_s = torch.zeros(n, dtype=dtype, device=self.device)
        self._full_a_pre_swing_wait_s = torch.zeros(n, dtype=dtype, device=self.device)
        self._full_a_teacher_frame = torch.zeros(
            n, dtype=torch.long, device=self.device
        )
        self._full_a_motion_phase_code = torch.full(
            (n,), READY_HOLD_PHASE_INDEX, dtype=torch.long, device=self.device
        )
        self._full_a_teacher_joint_pos = self.q_ready.unsqueeze(0).expand(
            n, -1
        ).clone()
        self._full_a_teacher_joint_vel = torch.zeros_like(
            self._full_a_teacher_joint_pos
        )
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
        self._full_a_r06_payment_event.zero_()
        self._full_a_owner_valid_bits[:, 3:] = 0
        self._full_a_owner_fault_bits[:, 3:] = 0
        self._full_a_owner_source_step[:, 3:] = -1
        self._full_a_owner_fact_f32[:, 3:] = 0.0

    def _full_a_reveal_rows(self, ids) -> None:
        """Publish the slot-zero centre question from the live base frame."""

        torch = self._torch
        n = int(ids.numel())
        if n == 0:
            return
        self._full_a_selected_reset_frame0_carry[ids] = False
        builder = getattr(
            self, "_full_a_question_builder", portable_question.build_center_question
        )
        origins = self.env.scene.env_origins[ids]
        base_position = self.sim.data.qpos[
            ids, self.root_qadr : self.root_qadr + 3
        ] - origins
        base_quat = self.sim.data.qpos[
            ids, self.root_qadr + 3 : self.root_qadr + 7
        ]
        question = builder(
            torch=torch,
            row=self._full_a_catalog.fresh_action,
            base_position_scene=base_position,
            base_quat_wxyz=base_quat,
            contact_reference_root_z_scene=(
                self._full_a_teacher.contact_reference_root_z_scene
            ),
            step_dt=self.step_dt,
            table_surface_z_scene=float(self.hope_to_scene[2]),
        )
        task = question["task_f32"]
        if tuple(task.shape) != (n, observation_contract.TASK_F32_WIDTH):
            raise RuntimeError("portable centre question task width differs")
        self._epoch_task_f32[ids] = task
        self._full_a_launch_state_f32[ids] = question[
            "launch_state_f32"
        ]
        self._full_a_teacher_rate[ids] = question["teacher_rate"]
        self._full_a_scaled_t_hit_s[ids] = question["scaled_t_hit_s"]
        self._full_a_scaled_t_cycle_s[ids] = question["scaled_t_cycle_s"]
        self._full_a_pre_swing_wait_s[ids] = question["pre_swing_wait_s"]

        racket = task[:, 5:32]
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
        contact_tick = reveal + question["ttc_ticks"]
        launch_tick = contact_tick - question["launch_horizon_ticks"]
        deadline = contact_tick + max(
            1, int(round(FULL_A_FLIGHT_HORIZON_S / self.step_dt))
        )
        self._epoch_clock_ticks[ids, 0] = reveal
        self._epoch_clock_ticks[ids, 1] = contact_tick
        self._epoch_clock_ticks[ids, 2] = launch_tick
        self._epoch_clock_ticks[ids, 3] = deadline
        self._epoch_clock_ticks[ids, 4] = deadline + 1
        self._full_a_r03_expected_source_step[ids] = contact_tick
        self._epoch_phase[ids] = FULL_A_PHASE_REVEAL_COMMITTED
        self._epoch_task_valid[ids] = True
        self._epoch_selected[ids] = True

    def _full_a_launch_rows(self, ids) -> None:
        state = self._full_a_launch_state_f32[ids]
        data = self.sim.data
        data.qpos[ids, self.b_q : self.b_q + 3] = (
            self.env.scene.env_origins[ids] + state[:, :3]
        )
        data.qpos[ids, self.b_q + 3 : self.b_q + 7] = state[:, 3:7]
        data.qvel[ids, self.b_v : self.b_v + 3] = state[:, 7:10]
        data.qvel[ids, self.b_v + 3 : self.b_v + 6] = state[:, 10:13]
        self.ball_age_buf[ids] = 0
        self._epoch_phase[ids] = FULL_A_PHASE_LAUNCH_SETTLED
        self._epoch_launch_succeeded[ids] = True

    def _full_a_park_rows(self, ids) -> None:
        """Kinematically park revealed rows until their reverse-flight tick."""

        n = int(ids.numel())
        if n == 0:
            return
        torch = self._torch
        park_env = torch.tensor(
            WAIT_BALL_PARK_HOPE,
            dtype=self.qpos_init.dtype,
            device=self.device,
        ) + self.hope_to_scene
        data = self.sim.data
        data.qpos[ids, self.b_q : self.b_q + 3] = (
            self.env.scene.env_origins[ids] + park_env
        )
        data.qpos[ids, self.b_q + 3 : self.b_q + 7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            dtype=self.qpos_init.dtype,
            device=self.device,
        ).expand(n, 4)
        data.qvel[ids, self.b_v : self.b_v + 6] = 0.0
        self.ball_age_buf[ids] = 0

    def _full_a_prepare_step(self):
        idle = self._epoch_phase.eq(FULL_A_PHASE_IDLE)
        self._full_a_reveal_rows(idle.nonzero(as_tuple=False).squeeze(-1))
        launch = self._epoch_phase.eq(FULL_A_PHASE_REVEAL_COMMITTED) & (
            self._epoch_clock_ticks[:, 2] <= int(self.common_step_counter)
        )
        self._full_a_launch_rows(launch.nonzero(as_tuple=False).squeeze(-1))
        waiting = self._epoch_phase.eq(FULL_A_PHASE_REVEAL_COMMITTED)
        self._full_a_park_rows(waiting.nonzero(as_tuple=False).squeeze(-1))
        recovering = self._epoch_phase.eq(FULL_A_PHASE_OUTCOME_SETTLED)
        self._full_a_park_rows(recovering.nonzero(as_tuple=False).squeeze(-1))
        return idle, launch

    def _full_a_latch_ball_contacts(self) -> None:
        """Latch live contact plus first net/landing-plane crossings."""

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
        center_local = center - self.env.scene.env_origins
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
        newly_selected = classification["selected"] & ~self._full_a_selected_racket_contact
        self._full_a_selected_racket_contact |= classification["selected"]

        previous = self._full_a_previous_ball_center
        previous_valid = self._full_a_previous_ball_center_valid
        tracking = (
            active
            & self._full_a_selected_racket_contact
            & previous_valid
            & ~newly_selected
            & ~self._full_a_landing_crossing_present
        )
        (
            crosses_net,
            clears_net,
            landing,
            landing_xy,
            on_table,
            opponent_bound,
            on_opponent,
        ) = (
            portable_outcome.observe_flight_step(
                torch=torch,
                previous=previous,
                current=center_local,
                tracking=tracking,
                target_positive_x=self._full_a_target_positive_x,
                net_x=self._full_a_net_x,
                net_clear_z=self._full_a_net_clear_z,
                landing_plane_z=self._full_a_landing_plane_z,
                table_bounds=self._full_a_table_bounds,
            )
        )
        first_net = crosses_net & ~self._full_a_net_crossed
        self._full_a_net_crossed |= first_net
        self._full_a_net_clear |= first_net & clears_net
        self._full_a_landing_crossing_present |= landing
        self._full_a_landing_crossing_xy.copy_(
            torch.where(
                landing[:, None], landing_xy, self._full_a_landing_crossing_xy
            )
        )
        self._full_a_landing_on_table |= on_table
        self._full_a_landing_opponent_bound |= opponent_bound
        self._full_a_landing_on_opponent |= on_opponent

        keep = active & self._full_a_selected_racket_contact
        self._full_a_previous_ball_center.copy_(
            torch.where(keep[:, None], center_local, previous)
        )
        self._full_a_previous_ball_center_valid.copy_(
            (previous_valid | newly_selected) & active
        )

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
        # Physical owns the live flight image only.  Once R06 settles, retain
        # its last flight/contact fact for the critic instead of overwriting it
        # every recovery tick with the repeatedly parked ball.
        active = self._epoch_phase.eq(FULL_A_PHASE_LAUNCH_SETTLED)
        origins = self.env.scene.env_origins
        center = self.sim.data.qpos[:, self.b_q : self.b_q + 3] - origins
        contact_center = self._full_a_contact_center - origins
        values = self._full_a_physical_fact_f32
        first_selected = (
            self._full_a_selected_contact_event
            & self._full_a_physical_source_step.lt(0)
        )
        values[:, :3] = self._torch.where(active[:, None], center, values[:, :3])
        values[:, 3:6] = self._torch.where(
            first_selected[:, None],
            contact_center,
            values[:, 3:6],
        )
        values[:, 6:9] = values[:, 3:6]
        self._full_a_observation_ordinal += active.to(self._torch.long)
        values[:, 9] = self._torch.where(
            active,
            self._full_a_observation_ordinal.to(values.dtype),
            values[:, 9],
        )
        self._full_a_physical_present.copy_(active)
        self._full_a_owner_valid_bits[:, 0] = self._torch.bitwise_or(
            self._full_a_owner_valid_bits[:, 0],
            (
                active.to(self._torch.long) * portable_reward.PHYSICAL_PRESENT
                + self._full_a_selected_contact_event.to(self._torch.long)
                * portable_reward.PHYSICAL_SELECTED_CONTACT
            ),
        )
        self._full_a_owner_fault_bits[:, 0] = self._torch.bitwise_or(
            self._full_a_owner_fault_bits[:, 0],
            self._full_a_invalid_contact_event.to(self._torch.long),
        )
        self._full_a_physical_source_step.copy_(
            self._torch.where(
                first_selected,
                self._torch.full_like(
                    self._full_a_physical_source_step, int(self.common_step_counter)
                ),
                self._full_a_physical_source_step,
            )
        )

    def _full_a_settle_outcome(self, st):
        torch = self._torch
        active = self._epoch_phase.eq(FULL_A_PHASE_LAUNCH_SETTLED)
        ball_local = st["ball_pos"] - self.env.scene.env_origins
        finite = torch.isfinite(ball_local).all(dim=1) & ~self._full_a_invalid_contact_event
        now = int(self.common_step_counter)
        no_contact_deadline = (
            active
            & finite
            & ~self._full_a_selected_racket_contact
            & ~self._full_a_landing_crossing_present
            & (now >= self._epoch_clock_ticks[:, 3])
        )
        selected_crossing_horizon = (
            active
            & finite
            & self._full_a_selected_racket_contact
            & ~self._full_a_landing_crossing_present
            & (now >= self._epoch_clock_ticks[:, 4])
        )
        # Shared R06 owns two policy settlement clocks: a no-contact row closes
        # at the contact deadline, while a selected-contact row without a
        # landing crossing remains live through the crossing horizon.  A ball
        # leaving the broad housekeeping bounds is not a third settlement
        # authority and therefore cannot close either row early.
        expired = no_contact_deadline | selected_crossing_horizon
        settled, outcome = portable_outcome.classify_outcome(
            torch=torch,
            active=active,
            selected_contact=self._full_a_selected_racket_contact,
            finite=finite,
            landing_present=self._full_a_landing_crossing_present,
            landing_on_table=self._full_a_landing_on_table,
            landing_on_opponent=self._full_a_landing_on_opponent,
            net_crossed=self._full_a_net_crossed,
            net_clear=self._full_a_net_clear,
            dead=torch.zeros_like(active),
            expired=expired,
            codes=self._full_a_outcome_code,
        )
        self._full_a_outcome_code.copy_(
            torch.where(settled, outcome, self._full_a_outcome_code)
        )
        self._full_a_recovery_origin_step.copy_(
            torch.where(
                settled,
                self._epoch_clock_ticks[:, 3],
                self._full_a_recovery_origin_step,
            )
        )
        self._epoch_phase.copy_(
            torch.where(
                settled,
                torch.full_like(self._epoch_phase, FULL_A_PHASE_OUTCOME_SETTLED),
                self._epoch_phase,
            )
        )
        return settled, outcome

    def _full_a_publish_r06_fact(self, settled, outcome):
        """Publish one event-only R06 row from the observed shot outcome."""

        torch = self._torch
        present, source_valid, common, facts = portable_outcome.r06_rows(
            torch=torch,
            settled=settled,
            selected_contact=self._full_a_selected_racket_contact,
            invalid_outcome=outcome.eq(FULL_A_OUTCOME_INVALID),
            crossing_present=self._full_a_landing_crossing_present,
            crossing_xy=self._full_a_landing_crossing_xy,
            target_xy=self._full_a_landing_target_xy,
            opponent_bound=self._full_a_landing_opponent_bound,
            on_opponent=self._full_a_landing_on_opponent,
            net_crossed=self._full_a_net_crossed,
            net_clear=self._full_a_net_clear,
            broad_sigma=self._full_a_placement_broad_sigma,
            narrow_sigma=self._full_a_placement_narrow_sigma,
        )
        bits = (
            present.to(torch.long) * portable_reward.R06_PRESENT
            + source_valid.to(torch.long) * portable_reward.R06_POLICY_ELIGIBLE
            + source_valid.to(torch.long) * portable_reward.R06_SOURCE_VALID
        )
        self._full_a_r06_payment_event.copy_(present)
        self._full_a_owner_valid_bits[:, 2] = torch.where(
            present, bits, self._full_a_owner_valid_bits[:, 2]
        )
        self._full_a_owner_fault_bits[:, 2] = torch.where(
            present,
            (present & ~source_valid).to(torch.long),
            self._full_a_owner_fault_bits[:, 2],
        )
        self._full_a_owner_source_step[:, 2] = torch.where(
            present,
            torch.full_like(
                self._full_a_owner_source_step[:, 2], int(self.common_step_counter)
            ),
            self._full_a_owner_source_step[:, 2],
        )
        self._full_a_owner_fact_f32[:, 2] = torch.where(
            present[:, None], facts, self._full_a_owner_fact_f32[:, 2]
        )
        return present, source_valid, common

    def _full_a_recovery_foot_support(self, body_lin_vel=None):
        """Return the shared >=10N foot support gate and ankle slip velocity."""

        torch = self._torch
        geom = self._con_geom[:]
        valid = self._con_idx < self._nacon[0]
        g0, g1 = geom[:, 0], geom[:, 1]
        floor0, floor1 = g0.eq(self._full_a_floor_geom_id), g1.eq(
            self._full_a_floor_geom_id
        )
        other = torch.where(floor0, g1, g0).long()
        foot = self._full_a_foot_geom_class[other]
        world = self._con_world[:].long().clamp_(0, self.num_envs - 1)
        normal_force = self._full_a_contact_normal_force()
        finite_force = torch.isfinite(normal_force)
        safe_force = torch.where(
            finite_force, normal_force.clamp_min(0.0), torch.zeros_like(normal_force)
        )
        support_force = torch.zeros(
            (self.num_envs, 2), dtype=self.qpos_init.dtype, device=self.device
        )
        invalid_force = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        for index in (0, 1):
            rows = valid & (floor0 | floor1) & foot.eq(index + 1)
            support_force[:, index].scatter_add_(
                0, world, torch.where(rows, safe_force, torch.zeros_like(safe_force))
            )
            invalid_force.scatter_add_(
                0, world, (rows & ~finite_force).to(dtype=invalid_force.dtype)
            )
        if body_lin_vel is None:
            body_lin_vel, _body_ang_vel = self._body_com_velocities_w()
        slip = body_lin_vel[:, (3, 6), :2]
        return support_force.ge(FULL_A_SUPPORT_FORCE_N), slip, invalid_force.eq(0)

    def _full_a_recovery_component_errors(self):
        """Compute the shared thirteen recovery errors from live MuJoCo state."""

        torch = self._torch
        data = self.sim.data
        origins = self.env.scene.env_origins
        body_ids = self._fullmdp_body_ids
        body_pos = data.xpos[:, body_ids]
        body_quat = data.xquat[:, body_ids]
        body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        root_pos = body_pos[:, 0]
        root_quat = body_quat[:, 0]
        reference_body_pos = (
            self._full_a_teacher.body_pos_w[0].unsqueeze(0) + origins[:, None, :]
        )
        reference_body_quat = self._full_a_teacher.body_quat_w[0].unsqueeze(0)
        reference_joint = self._full_a_teacher.joint_pos[0].unsqueeze(0)
        upper = (7, 8, 9, 10, 11, 12, 13)
        foot_support, foot_slip, support_valid = self._full_a_recovery_foot_support(
            body_lin_vel
        )

        current_quat = body_quat[:, (0, *upper)]
        reference_quat = reference_body_quat[:, (0, *upper)]
        current_norm = torch.linalg.vector_norm(current_quat, dim=2)
        reference_norm = torch.linalg.vector_norm(reference_quat, dim=2)
        tiny = torch.finfo(body_quat.dtype).tiny
        quaternion_valid = (
            torch.isfinite(current_norm)
            & torch.isfinite(reference_norm)
            & current_norm.gt(tiny)
            & reference_norm.gt(tiny)
        ).all(dim=1)
        safe_current_quat = current_quat / current_norm.clamp_min(tiny).unsqueeze(2)
        safe_reference_quat = reference_quat / reference_norm.clamp_min(tiny).unsqueeze(2)

        root_angle_sq = self._quat_error_sq(
            torch, safe_reference_quat[:, 0], safe_current_quat[:, 0]
        )
        body_angle_sq = self._quat_error_sq(
            torch, safe_reference_quat[:, 1:], safe_current_quat[:, 1:]
        )
        errors = portable_outcome.recovery_errors(
            torch=torch,
            root_position=root_pos,
            reference_root_position=reference_body_pos[:, 0],
            root_orientation_error_sq=root_angle_sq,
            root_linear_velocity=body_lin_vel[:, 0],
            root_angular_velocity=body_ang_vel[:, 0],
            joint_position=self._qpos_act(),
            reference_joint_position=reference_joint,
            joint_velocity=self._qvel_act(),
            body_position=body_pos[:, upper],
            reference_body_position=reference_body_pos[:, upper],
            body_orientation_error_sq=body_angle_sq,
            body_linear_velocity=body_lin_vel[:, upper],
            body_angular_velocity=body_ang_vel[:, upper],
            foot_support=foot_support,
            foot_slip_xy=foot_slip,
        )
        return torch.where(
            (support_valid & quaternion_valid)[:, None],
            errors,
            torch.full_like(errors, float("nan")),
        )

    def _full_a_recovery_joint_limit_ok(self):
        """Match Isaac's 0.9 soft joint-position envelope."""

        torch = self._torch
        lower = self.jnt_lo + 0.05 * (self.jnt_hi - self.jnt_lo)
        upper = self.jnt_hi - 0.05 * (self.jnt_hi - self.jnt_lo)
        joint = self._qpos_act()
        return (
            torch.isfinite(joint).all(dim=1)
            & (joint >= lower).all(dim=1)
            & (joint <= upper).all(dim=1)
        )

    def _full_a_publish_r07_fact(self):
        """Publish one dense R07 cell only inside the exact 10..77 window."""

        torch = self._torch
        recovery = self._epoch_phase.eq(
            FULL_A_PHASE_OUTCOME_SETTLED
        ) & self._full_a_outcome_code.ne(FULL_A_OUTCOME_INVALID)
        age = int(self.common_step_counter) - self._full_a_recovery_origin_step
        expected = (
            recovery
            & (age >= FULL_A_RECOVERY_START_AGE_TICK)
            & (age <= FULL_A_RECOVERY_END_AGE_TICK)
        )
        errors = self._full_a_recovery_component_errors()
        hard_safety_ok = self._full_a_recovery_joint_limit_ok()
        eligible, valid, ready, facts = portable_outcome.r07_rows(
            torch=torch,
            expected=expected,
            age=age,
            errors=errors,
            hard_safety_ok=hard_safety_ok,
            scales=self._full_a_recovery_component_scales.unsqueeze(0),
            ready_tolerances=self._full_a_recovery_ready_tolerances.unsqueeze(0),
            weight=portable_outcome.RECOVERY_REWARD_WEIGHT,
        )
        expected_age = torch.where(
            self._full_a_recovery_expected_count.eq(0),
            torch.full_like(age, FULL_A_RECOVERY_START_AGE_TICK),
            self._full_a_recovery_last_age + 1,
        )
        sequence_fault = expected & age.ne(expected_age)
        self._full_a_recovery_sticky_fault |= sequence_fault | (expected & ~valid)
        clean_eligible = eligible & ~self._full_a_recovery_sticky_fault
        facts = torch.where(clean_eligible[:, None], facts, torch.zeros_like(facts))
        self._full_a_owner_valid_bits[:, 3] = (
            expected.to(torch.long) * portable_reward.R07_PRESENT
            + clean_eligible.to(torch.long)
            * portable_reward.R07_NUMERICALLY_VALID
        )
        self._full_a_owner_fault_bits[:, 3] = (
            expected & self._full_a_recovery_sticky_fault
        ).to(torch.long)
        self._full_a_owner_source_step[:, 3] = torch.where(
            clean_eligible,
            torch.full_like(
                self._full_a_owner_source_step[:, 3], int(self.common_step_counter)
            ),
            torch.full_like(self._full_a_owner_source_step[:, 3], -1),
        )
        self._full_a_owner_fact_f32[:, 3] = facts
        next_streak = torch.where(
            clean_eligible & ready,
            self._full_a_recovery_ready_streak + 1,
            torch.zeros_like(self._full_a_recovery_ready_streak),
        )
        self._full_a_recovery_ready_streak.copy_(next_streak)
        self._full_a_recovery_ready_seen |= next_streak >= 2
        self._full_a_recovery_expected_count += expected.to(torch.long)
        self._full_a_recovery_eligible_count += clean_eligible.to(torch.long)
        self._full_a_recovery_last_age.copy_(
            torch.where(expected, age, self._full_a_recovery_last_age)
        )
        return expected, clean_eligible

    def _full_a_finish_recovery(self, terminated, truncated):
        """Classify the completed fixed recovery window without truncating it."""

        torch = self._torch
        outcome_settled = self._epoch_phase.eq(FULL_A_PHASE_OUTCOME_SETTLED)
        invalid_outcome = outcome_settled & self._full_a_outcome_code.eq(
            FULL_A_OUTCOME_INVALID
        )
        recovery = outcome_settled & ~invalid_outcome
        age = int(self.common_step_counter) - self._full_a_recovery_origin_step
        expected_cells = (
            FULL_A_RECOVERY_END_AGE_TICK
            - FULL_A_RECOVERY_START_AGE_TICK
            + 1
        )
        complete_window = (
            self._full_a_recovery_expected_count.eq(expected_cells)
            & self._full_a_recovery_eligible_count.eq(expected_cells)
            & self._full_a_recovery_last_age.eq(FULL_A_RECOVERY_END_AGE_TICK)
            & ~self._full_a_recovery_sticky_fault
        )
        completion_due = (
            recovery
            & age.ge(FULL_A_RECOVERY_END_AGE_TICK)
            & ~terminated
            & ~truncated
        )
        if bool((completion_due & ~complete_window).any()):
            raise RuntimeError(
                "portable R07 recovery window is incomplete or faulted"
            )
        terminal, success, failure, timeout = portable_outcome.recovery_status(
            torch=torch,
            recovering=recovery,
            age=age,
            terminated=terminated,
            truncated=truncated,
            ready_seen=self._full_a_recovery_ready_seen,
            end_age=FULL_A_RECOVERY_END_AGE_TICK,
        )
        terminal |= invalid_outcome
        failure |= invalid_outcome
        self._epoch_phase.copy_(
            torch.where(
                terminal,
                torch.full_like(self._epoch_phase, FULL_A_PHASE_RECOVERY_SETTLED),
                self._epoch_phase,
            )
        )
        return terminal, success, failure, timeout

    def _snapshot_ready_teacher(self) -> None:
        data = self.sim.data
        ids = self._fullmdp_body_ids
        self._ready_teacher_body_pos = data.xpos[:, ids].detach().clone()
        self._ready_teacher_body_quat = data.xquat[:, ids].detach().clone()
        body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        self._ready_teacher_body_lin_vel = body_lin_vel.detach().clone()
        self._ready_teacher_body_ang_vel = body_ang_vel.detach().clone()
        self._teacher_body_pos = self._ready_teacher_body_pos.clone()
        self._teacher_body_quat = self._ready_teacher_body_quat.clone()
        self._teacher_body_lin_vel = self._ready_teacher_body_lin_vel.clone()
        self._teacher_body_ang_vel = self._ready_teacher_body_ang_vel.clone()
        self._refresh_aligned_teacher_body_pose()

    def _full_a_update_teacher(self) -> None:
        """Advance the selected measured teacher on the task's exact clock."""

        if not getattr(self, "full_a_mode", False):
            return
        torch = self._torch
        valid = self._epoch_task_valid
        frame0_carry = self._full_a_selected_reset_frame0_carry
        elapsed = torch.clamp(
            (
                int(self.common_step_counter) - self._epoch_clock_ticks[:, 0]
            ).to(dtype=self.qpos_init.dtype)
            * self.step_dt,
            min=0.0,
        )
        sampled = portable_question.sample_motion_teacher(
            torch,
            self._full_a_teacher,
            elapsed,
            self._full_a_teacher_rate,
            self._full_a_pre_swing_wait_s,
        )
        origins = self.env.scene.env_origins[:, None, :]
        # Isaac's Motion teacher switches atomically from the hidden physical
        # safe-ready tuple to measured frame zero at reveal.  The separate
        # diagnostic q_des oracle may ramp the command, but that ramp is never
        # a Motion observation or reward reference.  Keep frame zero in the
        # prepare phase until the rounded measured clock actually leaves it.
        playback_started = (
            valid
            & (elapsed + 1.0e-12 >= self._full_a_pre_swing_wait_s)
            & sampled["frame"].gt(0)
        )
        prepare = valid & ~playback_started
        swing = (
            playback_started
            & (
                elapsed
                <= self._full_a_pre_swing_wait_s
                + self._full_a_scaled_t_hit_s
                + 1.0e-12
            )
        )
        follow = (
            valid
            & ~prepare
            & ~swing
            & (
                elapsed
                < self._full_a_pre_swing_wait_s
                + self._full_a_scaled_t_cycle_s
            )
        )
        phase = torch.full_like(
            self._full_a_motion_phase_code, READY_HOLD_PHASE_INDEX
        )
        phase = torch.where(
            valid & ~prepare & ~swing & ~follow,
            torch.full_like(phase, 3),
            phase,
        )
        phase = torch.where(follow, torch.full_like(phase, 2), phase)
        phase = torch.where(swing, torch.full_like(phase, 1), phase)
        phase = torch.where(prepare, torch.zeros_like(phase), phase)
        self._full_a_motion_phase_code.copy_(phase)
        # Task-valid rows remain owned by this measured action through
        # recovery.  Shared Motion holds the completed action's measured frame
        # zero after the suffix; only task-invalid IDLE rows expose the
        # separately admitted physical-ready snapshot.
        measured = valid | frame0_carry
        completed = (valid & ~prepare & ~swing & ~follow) | frame0_carry
        frame0_joint = self._full_a_teacher.joint_pos[0].unsqueeze(0)
        frame0_body_pos = self._full_a_teacher.body_pos_w[0].unsqueeze(0)
        frame0_body_quat = self._full_a_teacher.body_quat_w[0].unsqueeze(0)
        measured_joint_pos = torch.where(
            completed[:, None], frame0_joint, sampled["joint_pos"]
        )
        measured_body_pos = torch.where(
            completed[:, None, None], frame0_body_pos, sampled["body_pos_w"]
        )
        measured_body_quat = torch.where(
            completed[:, None, None], frame0_body_quat, sampled["body_quat_w"]
        )
        self._full_a_teacher_frame.copy_(
            torch.where(
                valid & ~completed,
                sampled["frame"],
                torch.zeros_like(sampled["frame"]),
            )
        )
        self._full_a_teacher_joint_pos.copy_(
            torch.where(
                measured[:, None],
                measured_joint_pos,
                self.q_ready.unsqueeze(0).expand_as(self._full_a_teacher_joint_pos),
            )
        )
        measured_joint_vel = torch.where(
            (prepare | completed)[:, None],
            torch.zeros_like(sampled["joint_vel"]),
            sampled["joint_vel"],
        )
        self._full_a_teacher_joint_vel.copy_(
            torch.where(
                measured[:, None],
                measured_joint_vel,
                torch.zeros_like(self._full_a_teacher_joint_vel),
            )
        )
        self._teacher_body_pos.copy_(
            torch.where(
                measured[:, None, None],
                measured_body_pos + origins,
                self._ready_teacher_body_pos,
            )
        )
        self._teacher_body_quat.copy_(
            torch.where(
                measured[:, None, None],
                measured_body_quat,
                self._ready_teacher_body_quat,
            )
        )
        measured_body_lin_vel = torch.where(
            (prepare | completed)[:, None, None],
            torch.zeros_like(sampled["body_lin_vel_w"]),
            sampled["body_lin_vel_w"],
        )
        self._teacher_body_lin_vel.copy_(
            torch.where(
                measured[:, None, None],
                measured_body_lin_vel,
                self._ready_teacher_body_lin_vel,
            )
        )
        measured_body_ang_vel = torch.where(
            (prepare | completed)[:, None, None],
            torch.zeros_like(sampled["body_ang_vel_w"]),
            sampled["body_ang_vel_w"],
        )
        self._teacher_body_ang_vel.copy_(
            torch.where(
                measured[:, None, None],
                measured_body_ang_vel,
                self._ready_teacher_body_ang_vel,
            )
        )
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
        # Base construction calls reset() -> _compute_obs() before the
        # Full-A buffers can be installed.  That construction observation is
        # the ordinary deterministic WAIT surface; publish Full-A fields only
        # after the subclass has completed its atomic initialization.
        full_a_initialized = (
            getattr(self, "full_a_mode", False) and self._fullmdp_initialized
        )
        contact = self._con_geom[:]
        valid = self._con_idx < self._nacon[0]
        ball_contact = valid & (
            (contact[:, 0] == self._ball_gid)
            | (contact[:, 1] == self._ball_gid)
        )
        if bool(ball_contact.any()) and not full_a_initialized:
            raise RuntimeError("portable FullMDP initial-WAIT ball is in contact")
        st = st or self._state()
        joint_pos_rel = self._qpos_act() - self.q_ready.unsqueeze(0)
        zero_joint = torch.zeros_like(joint_pos_rel)
        phase = torch.zeros(
            (self.num_envs, 5), dtype=joint_pos_rel.dtype, device=self.device
        )
        if full_a_initialized:
            phase = torch.nn.functional.one_hot(
                self._full_a_motion_phase_code, num_classes=5
            ).to(dtype=joint_pos_rel.dtype)
            teacher_joint_pos_rel = (
                self._full_a_teacher_joint_pos - self.q_ready.unsqueeze(0)
            )
            teacher_joint_vel = self._full_a_teacher_joint_vel
        else:
            phase[:, READY_HOLD_PHASE_INDEX] = 1.0
            teacher_joint_pos_rel = zero_joint
            teacher_joint_vel = zero_joint

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
                "teacher_joint_pos_rel": teacher_joint_pos_rel,
                "teacher_joint_vel_rel": teacher_joint_vel,
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
            if full_a_initialized:
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
        if full_a_initialized:
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
            critic_rows["physical_r03_r06_r07_fault_present"] = (
                self._full_a_owner_fault_bits.ne(0).to(joint_pos_rel.dtype)
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
            payment_valid_bits = self._full_a_owner_valid_bits.clone()
            payment_valid_bits[:, 0] = self._torch.where(
                self._full_a_selected_contact_event,
                payment_valid_bits[:, 0],
                self._torch.bitwise_and(
                    payment_valid_bits[:, 0],
                    self._torch.full_like(
                        payment_valid_bits[:, 0],
                        ~portable_reward.PHYSICAL_SELECTED_CONTACT,
                    ),
                ),
            )
            payment_valid_bits[:, 2] = self._torch.where(
                self._full_a_r06_payment_event,
                payment_valid_bits[:, 2],
                self._torch.zeros_like(payment_valid_bits[:, 2]),
            )
            terms[:, :14] = portable_reward.lifecycle_reward14(
                valid_bits=payment_valid_bits,
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
            self._full_a_teacher_rate[ids] = 1.0
            self._full_a_scaled_t_hit_s[ids] = 0.0
            self._full_a_scaled_t_cycle_s[ids] = 0.0
            self._full_a_pre_swing_wait_s[ids] = 0.0
            self._full_a_teacher_frame[ids] = 0
            self._full_a_motion_phase_code[ids] = READY_HOLD_PHASE_INDEX
            self._full_a_teacher_joint_pos[ids] = self.q_ready
            self._full_a_teacher_joint_vel[ids] = 0.0
            if hasattr(self, "_ready_teacher_body_pos"):
                self._teacher_body_pos[ids] = self._ready_teacher_body_pos[ids]
                self._teacher_body_quat[ids] = self._ready_teacher_body_quat[ids]
                self._teacher_body_lin_vel[ids] = (
                    self._ready_teacher_body_lin_vel[ids]
                )
                self._teacher_body_ang_vel[ids] = (
                    self._ready_teacher_body_ang_vel[ids]
                )
            self._full_a_observation_ordinal[ids] = 0
            self._full_a_racket_contact[ids] = False
            self._full_a_ball_table_contact[ids] = False
            self._full_a_selected_racket_contact[ids] = False
            self._full_a_contact_center[ids] = 0.0
            self._full_a_previous_ball_center[ids] = 0.0
            self._full_a_previous_ball_center_valid[ids] = False
            self._full_a_net_crossed[ids] = False
            self._full_a_net_clear[ids] = False
            self._full_a_landing_crossing_present[ids] = False
            self._full_a_landing_crossing_xy[ids] = 0.0
            self._full_a_landing_on_table[ids] = False
            self._full_a_landing_opponent_bound[ids] = False
            self._full_a_landing_on_opponent[ids] = False
            self._full_a_r06_payment_event[ids] = False
            self._full_a_outcome_code[ids] = FULL_A_OUTCOME_NONE
            self._full_a_recovery_origin_step[ids] = -1
            self._full_a_recovery_ready_streak[ids] = 0
            self._full_a_recovery_ready_seen[ids] = False
            self._full_a_recovery_expected_count[ids] = 0
            self._full_a_recovery_eligible_count[ids] = 0
            self._full_a_recovery_last_age[ids] = -1
            self._full_a_recovery_sticky_fault[ids] = False
            self._full_a_selected_reset_frame0_carry[ids] = False
            self._full_a_contact_classification_status[ids] = (
                racket_contact_geometry.OBSERVED_RUBBER_STATUS_NONE
            )
            self._full_a_generic_contact_event[ids] = False
            self._full_a_selected_contact_event[ids] = False
            self._full_a_opposite_contact_event[ids] = False
            self._full_a_edge_contact_event[ids] = False
            self._full_a_between_contact_event[ids] = False
            self._full_a_invalid_contact_event[ids] = False

    def _full_a_apply_selected_reset(self, selected_reset):
        """Clear completed shot rows without terminating the Gym episode."""

        ids = selected_reset.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self._full_a_park_rows(ids)
            self.reset_generation[ids] += 1
            self._clear_lifecycle(ids)
            self._full_a_selected_reset_frame0_carry[ids] = True
        return ids

    def reset(self):
        observations, extras = super().reset()
        if self._fullmdp_initialized:
            self.reset_generation += 1
            self.last_terminal_bits.zero_()
            self._clear_lifecycle(self._all_env_ids)
            if self.full_a_mode:
                self._full_a_update_teacher()
            else:
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
        self._full_a_update_teacher()
        terminated, truncated, terminal_bits, resolved_table_contact = (
            self._fullmdp_termination(st, requested_qdes)
        )
        self._full_a_publish_physical_fact()
        r03_present_event, r03_valid_event = self._full_a_publish_r03_fact()
        outcome_event, outcome = self._full_a_settle_outcome(st)
        r06_present, r06_eligible, r06_common = self._full_a_publish_r06_fact(
            outcome_event, outcome
        )
        r07_present, r07_eligible = self._full_a_publish_r07_fact()
        reward, reward_terms = self._fullmdp_reward20()
        shot_terminal, recovery_success, recovery_failure, recovery_timeout = (
            self._full_a_finish_recovery(terminated, truncated)
        )
        contact_event = self._full_a_generic_contact_event.clone()
        selected_contact_event = self._full_a_selected_contact_event.clone()
        opposite_contact_event = self._full_a_opposite_contact_event.clone()
        edge_contact_event = self._full_a_edge_contact_event.clone()
        between_contact_event = self._full_a_between_contact_event.clone()
        invalid_contact_event = self._full_a_invalid_contact_event.clone()
        contact_classification_status = (
            self._full_a_contact_classification_status.clone()
        )
        landing_on_opponent = self._full_a_landing_on_opponent.clone()
        landing_opponent_bound = self._full_a_landing_opponent_bound.clone()
        shot_terminal |= invalid_contact_event
        recovery_failure |= invalid_contact_event
        self._epoch_phase.copy_(
            torch.where(
                invalid_contact_event,
                torch.full_like(
                    self._epoch_phase, FULL_A_PHASE_RECOVERY_SETTLED
                ),
                self._epoch_phase,
            )
        )
        dones = terminated | truncated
        # R07 completion closes the selected ActionEpoch row, not the Gym
        # episode.  Preserve robot/action/episode state and bootstrap across
        # the next reveal; only an actual safety/time-limit terminal owns the
        # expensive full environment reset.
        selected_reset = shot_terminal & self._epoch_selected & ~dones
        terminal_phase = self._epoch_phase.clone()
        outcome_state = self._full_a_outcome_code.clone()
        racket_contact = self._full_a_racket_contact.clone()
        table_contact = self._full_a_ball_table_contact.clone()
        physical_center = self._full_a_physical_fact_f32[:, :3].clone()
        selected_reset_ids = self._full_a_apply_selected_reset(selected_reset)
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.last_terminal_bits[reset_ids] = terminal_bits[reset_ids]
            self._reset_idx(reset_ids)
            self.reset_generation[reset_ids] += 1
            self._clear_lifecycle(reset_ids)
        if selected_reset_ids.numel() > 0 or reset_ids.numel() > 0:
            self.sim.forward()
            if self._cap_ok:
                self._probe_capacity("forward")
        self._full_a_update_teacher()
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
            "full_a_outcome_code": outcome_state,
            "full_a_racket_contact": racket_contact,
            "full_a_ball_table_contact": table_contact,
            "full_a_physical_current_center": physical_center,
            "full_a_reveal_event": reveal_event,
            "full_a_launch_event": launch_event,
            "full_a_flight_terminal_event": outcome_event,
            "full_a_selected_reset_event": selected_reset,
            "full_a_racket_contact_eligible_event": launch_event,
            "full_a_racket_contact_event": contact_event,
            "full_a_action_slot": self._full_a_action_slot.clone(),
            "full_a_action_uid": self._full_a_action_uid.clone(),
            "full_a_mount_normal_sign": self._full_a_mount_normal_sign.clone(),
            "full_a_contact_classification_status": (
                contact_classification_status
            ),
            "full_a_selected_contact_event": selected_contact_event,
            "full_a_opposite_contact_event": opposite_contact_event,
            "full_a_edge_contact_event": edge_contact_event,
            "full_a_between_contact_event": between_contact_event,
            "full_a_invalid_contact_event": invalid_contact_event,
            "full_a_r03_present_event": r03_present_event,
            "full_a_r03_physically_valid_event": r03_valid_event,
            "full_a_landing_crossing_event": (
                outcome_event & self._full_a_landing_crossing_present
            ),
            "full_a_landing_on_opponent": (
                landing_on_opponent
            ),
            "full_a_landing_opponent_bound": (
                landing_opponent_bound
            ),
            "full_a_r06_present_event": r06_present,
            "full_a_r06_eligible_event": r06_eligible,
            "full_a_r06_common_event": r06_common,
            "full_a_r07_present_event": r07_present,
            "full_a_r07_eligible_event": r07_eligible,
            "full_a_recovery_success_event": recovery_success,
            "full_a_recovery_failure_event": recovery_failure,
            "full_a_recovery_timeout_event": recovery_timeout,
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
    "FULL_A_SUPPORT_FORCE_N",
    "FULLMDP_TRACKED_BODY_NAMES",
    "FULLMDP_DENSE_REWARD_SPECS",
    "FULLMDP_TERMINATION_BITS",
    "FullMdpInitialWaitVecEnv",
]
