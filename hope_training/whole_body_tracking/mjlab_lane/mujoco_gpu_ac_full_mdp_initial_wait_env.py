"""MuJoCo FullMDP legacy WAIT plus the live single-shot Full-A slice.

The plant, masked reset implementation, and ``sim.forward()`` remain the
tracked :class:`A3ReadyBallVecEnv` implementation.  WAIT parks the ball and
keeps the legacy V1 observation ABI.  The opt-in A slice additionally reveals
and launches one deterministic question, publishes live Physical/R03/R06/R07
facts, computes their ordered lifecycle/motion reward terms, and
publishes the compact semantic V3 observation ABI.

The narrow WAIT transition below advances the real plant and closes only the
IDLE reward/termination/masked-reset path.  An observed generic racket
contact is classified against the action-zero mount at the same physics
substep.  R06 consumes only a measured descending landing crossing.  R07's
fixed accepted-task window is clocked from Motion task-close independently
of R06 settlement; neither fact is a readiness claim for Full MuJoCo A.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

try:
    from .a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
    from . import mujoco_full_mdp_portable_question as portable_question
    from . import mujoco_full_mdp_portable_outcome as portable_outcome
    from . import mujoco_full_mdp_teacher_replay as teacher_replay
    from .mujoco_gpu_ac_table_keepout import DeviceExactTableKeepout
except ImportError:  # Direct execution with mjlab_lane on PYTHONPATH.
    from a3_train_ppo import A3ReadyBallVecEnv, SimCfg, TaskCfg
    import mujoco_full_mdp_portable_question as portable_question
    import mujoco_full_mdp_portable_outcome as portable_outcome
    import mujoco_full_mdp_teacher_replay as teacher_replay
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
import action_ball_full_mdp_reward_contract as reward_contract
import action_ball_full_mdp_regularization as regularization
import action_ball_full_mdp_paddle_prior as paddle_prior
import racket_contact_geometry as racket_contact_geometry


WAIT_BALL_PARK_HOPE = (0.0, 0.0, 10.0)
RECOVER_HIDDEN_PHASE_INDEX = 3
READY_HOLD_PHASE_INDEX = 4
FULL_A_MOTION_PREPARE_PHASE_INDEX = 0
FULL_A_MOTION_SWING_PHASE_INDEX = 1
FULL_A_MOTION_FOLLOW_PHASE_INDEX = 2
FULL_A_PHASE_IDLE = 0
FULL_A_PHASE_REVEAL_COMMITTED = 2
FULL_A_PHASE_LAUNCH_SETTLED = 5
FULL_A_PHASE_OUTCOME_SETTLED = 6
FULL_A_PHASE_RETIRED = 8
FULL_A_OUTCOME_NONE = portable_outcome.OUTCOME_NONE
FULL_A_OUTCOME_FLIGHT_EXPIRED = portable_outcome.OUTCOME_FLIGHT_EXPIRED
FULL_A_OUTCOME_BALL_DEAD = portable_outcome.OUTCOME_BALL_DEAD
FULL_A_OUTCOME_LEGAL_LANDING = portable_outcome.OUTCOME_LEGAL_LANDING
FULL_A_OUTCOME_OWN_TABLE_LANDING = portable_outcome.OUTCOME_OWN_TABLE_LANDING
FULL_A_OUTCOME_OUT = portable_outcome.OUTCOME_OUT
FULL_A_OUTCOME_INVALID = portable_outcome.OUTCOME_INVALID
FULL_A_RECOVERY_START_AGE_TICK = portable_outcome.RECOVERY_START_AGE_TICK
FULL_A_RECOVERY_END_AGE_TICK = portable_outcome.RECOVERY_END_AGE_TICK
FULL_A_PLACEMENT_BROAD_SIGMA_M = portable_outcome.PLACEMENT_BROAD_SIGMA_M
FULL_A_BODY_ORIENTATION_COARSE_STD_RAD = 1.0
FULL_A_SUPPORT_FORCE_N = 10.0
FULL_A_POLICY_BOOTSTRAP_KIND = "a3_take061_dynamic_ready_head_v1"
FULL_A_FACT_INTEGRITY_R03_NONFINITE = 1 << 0
FULL_A_FACT_INTEGRITY_R06_SOURCE_INVALID = 1 << 1
FULL_A_FACT_INTEGRITY_R07_SEQUENCE = 1 << 2
FULL_A_FACT_INTEGRITY_R07_NONFINITE = 1 << 3
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
FULLMDP_UPPER_TRACKED_BODY_NAMES = (
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)
FULLMDP_ANCHOR_BODY_NAME = "torso_Link"
FULLMDP_DENSE_REWARD_SPECS = reward_contract.COMMON_DENSE_SPECS
FULLMDP_PADDLE_REWARD_SPECS = reward_contract.PADDLE_MOTION_PRIOR_SPECS
FULLMDP_TERMINATION_BITS = {
    "time_out": 1,
    "base_fell_tilt": 2,
    "base_too_low": 4,
    "joint_qdes_forbidden": 8,
    "robot_hit_table": 16,
}


class FullMdpInitialWaitVecEnv(A3ReadyBallVecEnv):
    """Real WAIT V1 plus the Full-A semantic 215/231 observation."""

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
        if full_a_mode and float(task_cfg.episode_length_s) != 30.0:
            raise ValueError(
                "Full-A requires the shared exact 30s/1500-tick horizon"
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
        self._fullmdp_anchor_body_id = int(body_ids[self._fullmdp_anchor_index])
        self._fullmdp_pelvis_body_id = int(body_ids[0])
        self._fullmdp_pelvis_root_id = int(roots[0])
        upper_non_wrist_names = (
            reward_contract.upper_except_held_wrist_body_names(
                FULLMDP_UPPER_TRACKED_BODY_NAMES
            )
        )
        self._fullmdp_upper_non_wrist_body_indices = self._torch.tensor(
            [FULLMDP_TRACKED_BODY_NAMES.index(name) for name in upper_non_wrist_names],
            dtype=self._torch.long,
            device=self.device,
        )
        weights = [spec.manager_weight for spec in FULLMDP_DENSE_REWARD_SPECS]
        self._fullmdp_dense_weights = self._torch.tensor(
            weights, dtype=self.qpos_init.dtype, device=self.device
        )
        self._fullmdp_paddle_weights = self._torch.tensor(
            [spec.manager_weight for spec in FULLMDP_PADDLE_REWARD_SPECS],
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        if any(
            spec.scale_during_playback is None
            or not math.isfinite(float(spec.scale_during_playback))
            or float(spec.scale_during_playback) <= 0.0
            for spec in FULLMDP_PADDLE_REWARD_SPECS
        ):
            raise RuntimeError("FullMDP paddle playback scales are unavailable")
        self._fullmdp_paddle_playback_scales = self._torch.tensor(
            [
                spec.scale_during_playback
                for spec in FULLMDP_PADDLE_REWARD_SPECS
            ],
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        self._fullmdp_paddle_precision_stds = self._torch.tensor(
            [spec.std for spec in FULLMDP_PADDLE_REWARD_SPECS],
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        if any(spec.coarse_std is None for spec in FULLMDP_PADDLE_REWARD_SPECS):
            raise RuntimeError("FullMDP paddle coarse widths are unavailable")
        self._fullmdp_paddle_coarse_stds = self._torch.tensor(
            [spec.coarse_std for spec in FULLMDP_PADDLE_REWARD_SPECS],
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        self._fullmdp_regularization_weights = self._torch.tensor(
            [spec.manager_weight for spec in reward_contract.REGULARIZATION_SPECS],
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        hard_span = self.jnt_hi - self.jnt_lo
        soft_lower = self.jnt_lo + 0.05 * hard_span
        soft_upper = self.jnt_hi - 0.05 * hard_span
        self._fullmdp_regularization_soft_limits = self._torch.stack(
            (soft_lower, soft_upper), dim=1
        )
        self._fullmdp_regularization_hard_limits = self._torch.stack(
            (self.jnt_lo, self.jnt_hi), dim=1
        )
        self._fullmdp_racket_long_axis_local = self._torch.tensor(
            racket_contact_geometry.RACKET_BUTT_TO_BLADE_AXIS_LOCAL,
            dtype=self.qpos_init.dtype,
            device=self.device,
        )
        self._fullmdp_mount_normal_sign = self._torch.ones(
            self.num_envs, dtype=self.qpos_init.dtype, device=self.device
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
            self._full_a_cadence = portable_catalog.derive_portable_fresh_cadence(
                self._full_a_catalog
            )
            if (
                float(self.step_dt) != portable_catalog.FRESH_POLICY_STEP_S
                or int(self.max_episode_length)
                != self._full_a_cadence.episode_horizon_ticks
            ):
                raise RuntimeError("Full-A policy clock/cadence horizon differs")
            self._full_a_teacher = portable_question.load_portable_motion_teacher(
                row=self._full_a_catalog.fresh_action,
                tracked_body_names=FULLMDP_TRACKED_BODY_NAMES,
                torch=self._torch,
                dtype=self.qpos_init.dtype,
                device=self.device,
            )
            self._bind_full_a_dynamic_ready_policy_prior(
                ready_pose_payload=ready_pose_payload,
                ready_pose_path=ready_pose_path,
            )
            self._initialize_full_a_geometry(mujoco)
            self._initialize_full_a_state()
            self.sim.forward()
        self._snapshot_ready_teacher()
        if self.full_a_mode:
            self._full_a_prime_cadence_readiness(self._all_env_ids)
            self._full_a_update_teacher()
        self._fullmdp_initialized = True
        self._compute_obs()
        # ``A3ReadyBallVecEnv.__init__`` necessarily observes the legacy WAIT
        # surface before the Full-A producers below exist.  No caller can see
        # that construction-only row; publish the final public width only after
        # the semantic state has been installed atomically.
        self.num_obs = int(self._obs_buf.shape[1])

    @property
    def action_contract_identity(self) -> dict:
        identity = super().action_contract_identity
        identity.update(
            {
                "full_a_reset_joint_source": "dynamic_ready.physical_ready.joint_pos_rad",
                "full_a_reset_root_source": "dynamic_ready.physical_ready.root_pose",
                "full_a_policy_bootstrap": FULL_A_POLICY_BOOTSTRAP_KIND,
            }
        )
        return identity

    def _bind_full_a_dynamic_ready_policy_prior(
        self, *, ready_pose_payload, ready_pose_path
    ) -> None:
        """Bind one action identity while keeping birth and teacher distinct."""

        if ready_pose_payload is None:
            source = Path(ready_pose_path) if ready_pose_path is not None else Path(
                self.pose["source"]
            )
            payload = source.read_bytes()
        else:
            payload = ready_pose_payload
        try:
            document = json.loads(payload.decode("utf-8"))
            normalized = document["hold_candidate"]["normalized_actor_action"]
            hold_qdes = document["hold_candidate"]["hold_qdes_joint_pos_rad"]
            stable_motion = document["sources"]["stable_motion"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Full-A dynamic-ready artifact core fields differ") from exc
        row = self._full_a_catalog.fresh_action
        if (
            document.get("action_id") != row.action_id
            or stable_motion.get("sha256") != row.motion_sha256
            or stable_motion.get("frame_index") != 0
            or not isinstance(normalized, list)
            or len(normalized) != self.num_actions
            or not isinstance(hold_qdes, list)
            or len(hold_qdes) != self.num_actions
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in (*normalized, *hold_qdes)
            )
        ):
            raise RuntimeError(
                "Full-A birth/actor prior and teacher do not bind one Take061 action"
            )
        action = self._torch.tensor(
            normalized, dtype=self.action_offset.dtype, device=self.device
        )
        expected_qdes = self._torch.tensor(
            hold_qdes, dtype=self.action_offset.dtype, device=self.device
        )
        decoded_qdes = self.action_offset + self.act_scale * action
        if not self._torch.allclose(
            decoded_qdes,
            expected_qdes,
            rtol=0.0,
            atol=teacher_replay.TEACHER_REPLAY_DECODER_ABI_ATOL_RAD,
        ):
            raise RuntimeError(
                "Full-A dynamic-ready actor prior does not decode to hold q_des"
            )
        self._full_a_policy_bootstrap_action = action
        self._full_a_policy_bootstrap_qdes = expected_qdes
        # The first guarded action is compared with the command that actually
        # sustains the physical birth pose, not the unrelated default stand.
        # Birth, actor prior and guard history therefore share one artifact.
        self._qdes_previous_executable.copy_(
            expected_qdes.unsqueeze(0).expand_as(
                self._qdes_previous_executable
            )
        )

    def _reset_idx(self, ids):
        """Reset at physical ready and seed the guard from its hold command."""

        super()._reset_idx(ids)
        if (
            getattr(self, "full_a_mode", False)
            and ids.numel() > 0
            and hasattr(self, "_full_a_policy_bootstrap_qdes")
        ):
            self._qdes_previous_executable[ids] = (
                self._full_a_policy_bootstrap_qdes
            )

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
        self._full_a_table_surface_center_scene = torch.tensor(
            (
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
                table_surface,
            ),
            dtype=dtype,
            device=self.device,
        )
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

    @staticmethod
    def _diagnostic_contact_field_torch_view(
        *, torch, warp, torch_array_type, value
    ):
        """Return the field's existing Torch storage without guessing by device.

        MJLab 1.5.3 publishes live MuJoCo-Warp fields as its exact
        ``mjlab.sim.sim_data.TorchArray`` class.  Its public indexing API
        delegates to the cached shared-memory Torch tensor; ``value[...]`` is
        therefore a zero-copy view rather than a host transfer.  Native
        contiguous Warp arrays instead use Warp's zero-copy Torch bridge.  No
        other object is accepted merely because it exposes a shape, device,
        similarly named attribute, or DLPack-like protocol.
        """

        if torch.is_tensor(value):
            return value
        if isinstance(value, torch_array_type):
            result = value[...]
            if torch.is_tensor(result):
                return result
            raise TypeError("MJLab TorchArray indexing did not return a Tensor")
        warp_array_type = getattr(warp, "array", None)
        if (
            isinstance(warp_array_type, type)
            and isinstance(value, warp_array_type)
        ):
            result = warp.to_torch(value)
            if torch.is_tensor(result):
                return result
        raise TypeError("generic contact field has no supported Torch view")

    def enable_diagnostic_first_generic_contact_patch(self) -> None:
        """Opt into one host contact patch for the N=1 teacher replay only.

        The training path never calls this method.  Consequently its hot loop
        owns no patch buffers, contact-row selection, host transfer, or device
        synchronization.  The diagnostic deliberately pays those costs once,
        at the first backend-observed generic ball/racket contact.
        """

        if not getattr(self, "full_a_mode", False) or self.num_envs != 1:
            raise RuntimeError(
                "generic contact patch requires one FullMDP diagnostic world"
            )
        if hasattr(self, "_diagnostic_first_generic_contact_patch"):
            raise RuntimeError("generic contact patch diagnostic already enabled")
        contact = self.sim.data.contact
        required = ("geom", "worldid", "pos", "frame", "dist")
        if any(not hasattr(contact, name) for name in required):
            raise RuntimeError("MuJoCo-Warp generic contact patch surface differs")
        try:
            import warp as wp
            from mjlab.sim.sim_data import TorchArray

            tensors = {
                name: self._diagnostic_contact_field_torch_view(
                    torch=self._torch,
                    warp=wp,
                    torch_array_type=TorchArray,
                    value=value,
                )
                for name in ("pos", "frame", "dist")
                for value in (getattr(contact, name),)
            }
            expected_device = self._torch.device(self.device)
            if any(value.device != expected_device for value in tensors.values()):
                raise ValueError("generic contact patch tensor device differs")
            self._diagnostic_contact_patch_tensors = tensors
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo-Warp generic contact patch tensor bridge differs"
            ) from exc
        self._diagnostic_first_generic_contact_patch = None
        self._table_keepout.enable_diagnostic_first_positive_witness()
        self._diagnostic_contact_patch_consumer = (
            self._capture_diagnostic_first_generic_contact_patch
        )
        self._diagnostic_first_table_terminal_source = None
        self._diagnostic_first_resolved_substep = None
        self._diagnostic_table_tick_first_positive = None
        self._diagnostic_table_tick_keepout = False
        self._diagnostic_table_tick_final_resolved = False
        self._diagnostic_table_tick_resolved_any_substep = False
        self._diagnostic_table_attribution_consumer = (
            self._capture_diagnostic_table_attribution
        )

    def _begin_diagnostic_table_attribution_tick(self) -> None:
        self._diagnostic_table_tick_first_positive = None
        self._diagnostic_table_tick_keepout = False
        self._diagnostic_table_tick_final_resolved = False
        self._diagnostic_table_tick_resolved_any_substep = False

    def _capture_diagnostic_table_attribution(
        self, *, keepout, resolved, resolved_is_final,
        substep_index, capture_boundary, keepout_witness=None,
    ) -> None:
        """Latch the first pre-reset table source in the opt-in N=1 replay."""

        if tuple(keepout.shape) != (1,) or tuple(resolved.shape) != (1,):
            raise RuntimeError("diagnostic table attribution shape differs")
        keepout_positive = bool(keepout[0].detach().cpu().item())
        resolved_positive = bool(resolved[0].detach().cpu().item())
        needs_keepout_witness = (
            keepout_positive
            and self._diagnostic_table_tick_first_positive is None
            and self._diagnostic_first_table_terminal_source is None
        )
        if needs_keepout_witness:
            if keepout_witness is None:
                keepout_witness = self._table_keepout.diagnostic_first_positive_witness(
                    self.sim.data
                )
            pose_keys = (
                ("root_position_env_m", 3),
                ("root_quaternion_wxyz", 4),
                ("owner_position_env_m", 3),
                ("owner_quaternion_wxyz", 4),
            )
            identity = keepout_witness.get("plant_identity", {})
            if (
                not isinstance(keepout_witness, dict)
                or keepout_witness.get("schema")
                != "action_ball_keepout_first_witness_v1"
                or keepout_witness.get("table_role")
                not in ("top", "keepout", "net", "post_left", "post_right")
                or keepout_witness.get("component_kind")
                not in ("body_proxy", "blade")
                or keepout_witness.get("reason") != "sat_overlap"
                or type(keepout_witness.get("component_index")) is not int
                or not 0 <= keepout_witness["component_index"] < 63
                or type(keepout_witness.get("owner_body_name")) is not str
                or not keepout_witness["owner_body_name"]
                or type(keepout_witness.get("component_id")) is not str
                or not keepout_witness["component_id"]
                or not isinstance(
                    keepout_witness.get("sat_signed_margin_m"), (int, float)
                )
                or not math.isfinite(float(keepout_witness["sat_signed_margin_m"]))
                or any(
                    not isinstance(keepout_witness.get(key), list)
                    or len(keepout_witness[key]) != length
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in keepout_witness[key]
                    )
                    for key, length in pose_keys
                )
                or not isinstance(identity, dict)
                or set(identity) != {
                    "root_mjcf_sha256",
                    "identity_manifest_sha256",
                    "portable_identity_sha256",
                    "verification_receipt_sha256",
                    "owner_local_frame_sha256",
                }
                or any(
                    type(value) is not str
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                    for value in identity.values()
                )
                or any(
                    type(keepout_witness.get(key)) is not str
                    or len(keepout_witness[key]) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in keepout_witness[key]
                    )
                    for key in (
                        "collision_artifact_sha256",
                        "collision_content_sha256",
                    )
                )
            ):
                raise RuntimeError("positive keepout witness is unknown")
        self._diagnostic_table_tick_keepout |= keepout_positive
        if resolved_is_final:
            if capture_boundary != "post_forward_final" or substep_index is not None:
                raise RuntimeError("diagnostic final resolved boundary differs")
            self._diagnostic_table_tick_final_resolved = resolved_positive
        else:
            if capture_boundary != "physics_substep_poststate":
                raise RuntimeError("diagnostic substep resolved boundary differs")
            self._diagnostic_table_tick_resolved_any_substep |= resolved_positive
            if resolved_positive and self._diagnostic_first_resolved_substep is None:
                self._diagnostic_first_resolved_substep = {
                    **teacher_replay.contact_capture_boundary(
                        transition_start_step=int(
                            self._diagnostic_contact_patch_transition_start_step
                        ),
                        capture_boundary=capture_boundary,
                        physics_substep_index=substep_index,
                        decimation=int(self.decimation),
                    ),
                    "backend_resolved_table_contact": True,
                }
        if (
            self._diagnostic_table_tick_first_positive is None
            and (keepout_positive or (resolved_is_final and resolved_positive))
        ):
            boundary = teacher_replay.contact_capture_boundary(
                transition_start_step=int(
                    self._diagnostic_contact_patch_transition_start_step
                ),
                capture_boundary=capture_boundary,
                physics_substep_index=substep_index,
                decimation=int(self.decimation),
            )
            self._diagnostic_table_tick_first_positive = {
                **boundary,
                "keepout_source": keepout_positive,
                "backend_resolved_table_contact": (
                    resolved_is_final and resolved_positive
                ),
            }
            if keepout_positive and keepout_witness is not None:
                self._diagnostic_table_tick_first_positive[
                    "keepout_witness"
                ] = dict(keepout_witness)

    def diagnostic_table_attribution_tick(self, terminal_bits) -> dict:
        """Finalize one source split only after the terminal bits are known."""

        if not hasattr(self, "_diagnostic_table_attribution_consumer"):
            raise RuntimeError("diagnostic table attribution is not enabled")
        if tuple(terminal_bits.shape) != (1,):
            raise RuntimeError("diagnostic table terminal bits shape differs")
        table_bit = int(FULLMDP_TERMINATION_BITS["robot_hit_table"])
        table_terminal = bool(
            terminal_bits[0].detach().cpu().to(self._torch.long).item() & table_bit
        )
        any_source = (
            self._diagnostic_table_tick_keepout
            or self._diagnostic_table_tick_final_resolved
        )
        if table_terminal != any_source:
            raise RuntimeError("diagnostic table terminal source is unknown")
        if table_terminal and self._diagnostic_table_tick_first_positive is None:
            raise RuntimeError("diagnostic table first-positive source is unknown")
        if table_terminal and self._diagnostic_first_table_terminal_source is None:
            self._diagnostic_first_table_terminal_source = dict(
                self._diagnostic_table_tick_first_positive
            )
        return {
            "keepout_source": self._diagnostic_table_tick_keepout,
            "backend_resolved_table_contact": (
                self._diagnostic_table_tick_final_resolved
            ),
            "resolved_any_substep": (
                self._diagnostic_table_tick_resolved_any_substep
            ),
            "first_resolved_substep": (
                None
                if self._diagnostic_first_resolved_substep is None
                else dict(self._diagnostic_first_resolved_substep)
            ),
            "first_table_terminal_source": (
                None
                if self._diagnostic_first_table_terminal_source is None
                else dict(self._diagnostic_first_table_terminal_source)
            ),
        }

    def _capture_diagnostic_first_generic_contact_patch(
        self, *, first_contact, classification, substep_index, capture_boundary
    ) -> None:
        """Capture the first live generic ball/racket row without reclassifying."""

        if self._diagnostic_first_generic_contact_patch is not None:
            return
        if tuple(first_contact.shape) != (1,) or not bool(first_contact[0]):
            return
        torch = self._torch
        geom = self._con_geom[:]
        valid = self._con_idx < self._nacon[0]
        g0, g1 = geom[:, 0], geom[:, 1]
        ball0, ball1 = g0.eq(self._ball_gid), g1.eq(self._ball_gid)
        partner = torch.where(ball0, g1, g0).long()
        rows = (
            valid
            & (ball0 | ball1)
            & self._geom_class[partner].eq(1)
            & self._con_world[:].long().eq(0)
        )
        selected = rows.nonzero(as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            raise RuntimeError(
                "generic contact event has no matching backend contact row"
            )
        index = int(selected[0].item())
        force = self._full_a_contact_normal_force()
        raw_force = self._full_a_contact_force_f32[index].detach().cpu()
        normal_force = float(force[index].detach().cpu().item())
        timestep = float(self.mj_model.opt.timestep)
        def host_values(name):
            value = self._diagnostic_contact_patch_tensors[name][index]
            return value.detach().cpu().reshape(-1).tolist()

        geom_ids = [int(value) for value in geom[index].detach().cpu().tolist()]
        raw_frame = [float(value) for value in host_values("frame")]
        raw_normal = raw_frame[:3]
        normal_sign = 1.0 if geom_ids[0] == self._ball_gid else -1.0
        boundary = teacher_replay.contact_capture_boundary(
            transition_start_step=int(
                self._diagnostic_contact_patch_transition_start_step
            ),
            capture_boundary=capture_boundary,
            physics_substep_index=substep_index,
            decimation=int(self.decimation),
        )
        self._diagnostic_first_generic_contact_patch = {
            "present": True,
            **boundary,
            "reset_generation": int(self.reset_generation[0].item()),
            "question_f32_sha256": teacher_replay.tensor_f32_sha256(
                self._epoch_task_f32[0]
            ),
            "contact_row_index": index,
            "geom_ids": geom_ids,
            "position_w_m": [float(value) for value in host_values("pos")],
            "frame_w_from_contact": raw_frame,
            "normal_ball_to_racket_w": [
                normal_sign * value for value in raw_normal
            ],
            "distance_m": float(host_values("dist")[0]),
            "contact_force_6d_backend": [
                float(value) for value in raw_force.reshape(-1).tolist()
            ],
            "normal_force_n_backend": normal_force,
            "normal_impulse_ns_derived": normal_force * timestep,
            "physics_timestep_s": timestep,
            "classification_status": int(
                classification["status"][0].detach().cpu().item()
            ),
            "selected": bool(classification["selected"][0].detach().cpu().item()),
            "opposite": bool(classification["opposite"][0].detach().cpu().item()),
            "edge_or_rim_ambiguous": bool(
                classification["edge_or_rim_ambiguous"][0].detach().cpu().item()
            ),
            "between_planes_ambiguous": bool(
                classification["between_planes_ambiguous"][0]
                .detach().cpu().item()
            ),
            "invalid": bool(classification["invalid"][0].detach().cpu().item()),
        }

    def diagnostic_first_generic_contact_patch(self) -> dict:
        """Return a detached diagnostic patch or an explicit no-contact row."""

        if not hasattr(self, "_diagnostic_first_generic_contact_patch"):
            raise RuntimeError("generic contact patch diagnostic is not enabled")
        patch = self._diagnostic_first_generic_contact_patch
        return {"present": False} if patch is None else dict(patch)

    def _initialize_full_a_state(self) -> None:
        """Allocate the one row-wise state used by the optional A slice."""

        torch = self._torch
        n = self.num_envs
        dtype = self.qpos_init.dtype
        # Reward14 is evaluated on every control step.  Materialize its immutable
        # configured weights once with the rest of the Full-A device state so the
        # portable kernel never performs a per-step host-to-device construction.
        self._full_a_lifecycle_reward_weights = torch.tensor(
            portable_reward.LIFECYCLE_WEIGHTS,
            dtype=dtype,
            device=self.device,
        )
        # WAIT/recovery parking uses these exact constants on every control
        # transition.  Cache only the immutable scene-frame offset and unit
        # quaternion; per-world origins remain live inputs in _full_a_park_rows.
        self._full_a_park_position_scene = torch.tensor(
            WAIT_BALL_PARK_HOPE,
            dtype=dtype,
            device=self.device,
        ) + self.hope_to_scene
        self._full_a_park_quaternion = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            dtype=dtype,
            device=self.device,
        )
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
        # One per-transition producer-fault ingress.  It is intentionally not
        # lifecycle state: begin-step clears it, while a same-transition Gym
        # reset/reveal must not erase evidence before the rollout ledger drains
        # the returned extras.
        self._full_a_fact_integrity_fault_bits = torch.zeros(
            n, dtype=torch.long, device=self.device
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
        self._full_a_next_reveal_tick = torch.full(
            (n,),
            int(self._full_a_cadence.first_reveal_tick),
            dtype=torch.long,
            device=self.device,
        )
        self._full_a_scheduled_ordinal = torch.full(
            (n,), -1, dtype=torch.long, device=self.device
        )
        self._full_a_cadence_ready_streak = torch.zeros(
            (n,), dtype=torch.long, device=self.device
        )
        # R07 already measures support once from the real contact-force rows on
        # every control transition.  The critic consumes that same sample;
        # observation packing must not launch a second contact-force pass.
        self._full_a_foot_supported_lr = torch.zeros(
            (n, 2), dtype=torch.bool, device=self.device
        )
        # Materialize the shared host ABI constants once.  Observation packing
        # then needs one flat multiply per newly allocated actor/critic suffix,
        # with the already-scaled actor reused as the critic prefix.
        self._full_a_actor_scale_v3 = torch.tensor(
            observation_contract.ACTOR_SCALE_FLAT_V3,
            dtype=dtype,
            device=self.device,
        )
        self._full_a_critic_extension_scale_v3 = torch.tensor(
            observation_contract.CRITIC_EXTENSION_SCALE_FLAT_V3,
            dtype=dtype,
            device=self.device,
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
            (n,), RECOVER_HIDDEN_PHASE_INDEX, dtype=torch.long, device=self.device
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
        self._fullmdp_mount_normal_sign.copy_(
            self._full_a_mount_normal_sign.to(dtype=self.qpos_init.dtype)
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

    def _full_a_reset_cadence_rows(self, ids) -> None:
        """Restart only the true-reset cadence state for selected Gym rows."""

        if ids.numel() == 0:
            return
        self._full_a_next_reveal_tick[ids] = int(
            self._full_a_cadence.first_reveal_tick
        )
        self._full_a_scheduled_ordinal[ids] = -1
        self._full_a_cadence_ready_streak[ids] = 0

    def _full_a_begin_control_step(self) -> None:
        self._full_a_fact_integrity_fault_bits.zero_()
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

        # This callpoint runs after the preceding transition's reward and
        # termination have settled, but before the returned observation.  The
        # accepted question therefore belongs to the current post-transition
        # boundary and is visible to the next policy action that can earn its
        # first imitation reward.
        reveal = int(self.common_step_counter)
        contact_tick = reveal + question["ttc_ticks"]
        launch_tick = contact_tick - question["launch_horizon_ticks"]
        task_close_offset = torch.ceil(
            (
                question["pre_swing_wait_s"]
                + question["scaled_t_cycle_s"]
            )
            / float(self.step_dt)
            - 1.0e-12
        ).to(dtype=torch.long)
        task_close_tick = reveal + task_close_offset
        # Epoch clocks are monotonic common-step boundaries.  The scheduler may
        # park its episode-relative pointer at horizon+1 after the final due;
        # that does not change this accepted shot's fixed cadence boundary.
        next_reveal_tick = reveal + int(self._full_a_cadence.cadence_ticks)
        self._epoch_clock_ticks[ids, 0] = reveal
        self._epoch_clock_ticks[ids, 1] = contact_tick
        self._epoch_clock_ticks[ids, 2] = launch_tick
        self._epoch_clock_ticks[ids, 3] = task_close_tick
        self._epoch_clock_ticks[ids, 4] = next_reveal_tick
        self._full_a_r03_expected_source_step[ids] = contact_tick
        self._epoch_phase[ids] = FULL_A_PHASE_REVEAL_COMMITTED
        self._epoch_task_valid[ids] = True
        self._epoch_selected[ids] = True
        self._full_a_cadence_ready_streak[ids] = 0

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
        data = self.sim.data
        data.qpos[ids, self.b_q : self.b_q + 3] = (
            self.env.scene.env_origins[ids] + self._full_a_park_position_scene
        )
        data.qpos[ids, self.b_q + 3 : self.b_q + 7] = (
            self._full_a_park_quaternion.expand(n, 4)
        )
        data.qvel[ids, self.b_v : self.b_v + 6] = 0.0
        self.ball_age_buf[ids] = 0

    def _full_a_prepare_step(self):
        """Freeze the due schedule and advance only already-visible shots."""

        torch = self._torch
        policy_tick = self.episode_length_buf + 1
        last_scheduled_ordinal = (
            len(self._full_a_cadence.reference_due_ticks) - 1
        )
        opportunity_available = self._full_a_scheduled_ordinal.lt(
            last_scheduled_ordinal
        )
        scheduled_due = (
            policy_tick.eq(self._full_a_next_reveal_tick)
            & opportunity_available
        )
        launch_pending = self._epoch_phase.eq(FULL_A_PHASE_REVEAL_COMMITTED)
        launch_tick = self._epoch_clock_ticks[:, 2]
        now = int(self.common_step_counter)
        missed_launch = launch_pending & launch_tick.lt(now)
        # Clock ticks name post-transition boundaries.  Installing at boundary
        # L makes L->L+1 the first integrated flight transition; installing at
        # L-1 would add one hidden step and move a H-tick tape by (H+1)*dt.
        launch = launch_pending & launch_tick.eq(now)
        self._full_a_launch_rows(launch.nonzero(as_tuple=False).squeeze(-1))
        waiting = self._epoch_phase.eq(FULL_A_PHASE_REVEAL_COMMITTED)
        recovering = self._epoch_phase.eq(FULL_A_PHASE_OUTCOME_SETTLED) | (
            self._epoch_phase.eq(FULL_A_PHASE_RETIRED)
        )
        idle = self._epoch_phase.eq(FULL_A_PHASE_IDLE)
        # These post-launch phases are disjoint and all require the identical
        # idempotent park write.  Select their union once: per-row chronology is
        # unchanged (reveal, then exact launch, then park-if-nonflight) while the
        # dynamic-shape selections fall from five to three per control step.
        park = waiting | recovering | idle
        self._full_a_park_rows(park.nonzero(as_tuple=False).squeeze(-1))
        return scheduled_due, launch, missed_launch

    def _full_a_settle_reveal(self, scheduled_due, dones):
        """Classify a frozen due using post-transition lifecycle availability."""

        torch = self._torch
        survived = ~dones
        available = self._epoch_phase.eq(FULL_A_PHASE_IDLE) | (
            self._epoch_phase.eq(FULL_A_PHASE_RETIRED)
        )
        # Availability belongs to the returned-observation boundary.  A row
        # busy at transition start may naturally retire during physics and
        # recovery settlement, then ACCEPT without losing this scheduled
        # opportunity.  R07 remains telemetry/reward, never admission.
        due = scheduled_due & survived
        reveal = due & available
        deferred = due & ~available
        due_terminal_overlap = scheduled_due & dones
        last_scheduled_ordinal = (
            len(self._full_a_cadence.reference_due_ticks) - 1
        )
        self._full_a_scheduled_ordinal += due.to(torch.long)
        exhausted = due & self._full_a_scheduled_ordinal.eq(
            last_scheduled_ordinal
        )
        future_tick = self._full_a_next_reveal_tick + int(
            self._full_a_cadence.cadence_ticks
        )
        self._full_a_next_reveal_tick.copy_(
            torch.where(
                exhausted,
                torch.full_like(
                    self._full_a_next_reveal_tick,
                    int(self._full_a_cadence.episode_horizon_ticks) + 1,
                ),
                torch.where(due, future_tick, self._full_a_next_reveal_tick),
            )
        )
        reveal_ids = reveal.nonzero(as_tuple=False).squeeze(-1)
        if reveal_ids.numel() > 0:
            # ACCEPT atomically replaces an IDLE/RETIRED row.  DEFER and a
            # terminal overlap are zero-write with respect to task/lifecycle.
            self._clear_lifecycle(reveal_ids)
            self._full_a_reveal_rows(reveal_ids)
        return reveal, due, deferred, due_terminal_overlap

    def _full_a_latch_ball_contacts(
        self, contact_census=None, diagnostic_substep_index=None,
        diagnostic_capture_boundary=None,
    ) -> None:
        """Consume one shared census, then latch net/landing-plane crossings."""

        torch = self._torch
        if contact_census is None:
            # Focused host tests may invoke this consumer directly.  Production
            # always passes the census already recorded by the base probe.
            contact_census = A3ReadyBallVecEnv._contact_census(self)
        racket_count = contact_census.ball_racket_by_world
        table_count = contact_census.ball_table_by_world
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
        patch_consumer = getattr(
            self, "_diagnostic_contact_patch_consumer", None
        )
        if patch_consumer is not None:
            patch_consumer(
                first_contact=first_contact,
                classification=classification,
                substep_index=diagnostic_substep_index,
                capture_boundary=diagnostic_capture_boundary,
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
        """Return official-site position/velocity, raw +Y, and long axis."""

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
        long_axis = self._torch.matmul(
            rotation, self._fullmdp_racket_long_axis_local
        )
        long_axis = long_axis / self._torch.linalg.vector_norm(
            long_axis, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        return position - origins, velocity, normal, long_axis

    def _full_a_publish_r03_fact(self, racket_kinematics=None):
        """Publish one exact-strike FK and retain it until task reset."""

        step = int(self.common_step_counter)
        exact = self._full_a_r03_armed & (
            self._full_a_r03_expected_source_step == step
        )
        # LAUNCH_SETTLED is the lifecycle authority.  ``task_valid`` and
        # ``launch_succeeded`` are written by the same transition, so checking
        # them again here would only add a same-source, always-agreeing gate.
        due = exact & self._epoch_phase.eq(FULL_A_PHASE_LAUNCH_SETTLED)
        if racket_kinematics is None:
            racket_kinematics = self._full_a_racket_kinematics()
        position, velocity, normal, _long_axis = racket_kinematics
        achieved_finite = (
            self._torch.isfinite(position).all(dim=1)
            & self._torch.isfinite(velocity).all(dim=1)
            & self._torch.isfinite(normal).all(dim=1)
        )
        safe = due & achieved_finite
        self._full_a_fact_integrity_fault_bits |= (
            (due & ~achieved_finite).to(self._torch.long)
            * FULL_A_FACT_INTEGRITY_R03_NONFINITE
        )
        fact = self._full_a_r03_fact_f32
        fact[:, 15:18] = self._torch.where(
            safe[:, None], position, fact[:, 15:18]
        )
        fact[:, 18:21] = self._torch.where(
            safe[:, None], velocity, fact[:, 18:21]
        )
        fact[:, 21:24] = self._torch.where(
            safe[:, None], normal, fact[:, 21:24]
        )
        bits = (
            safe.to(self._torch.long) * portable_reward.R03_PRESENT
            + (safe & self._full_a_r03_physically_valid).to(self._torch.long)
            * portable_reward.R03_PHYSICALLY_VALID
        )
        self._full_a_owner_valid_bits[:, 1] = self._torch.where(
            due, bits, self._full_a_owner_valid_bits[:, 1]
        )
        self._full_a_owner_fault_bits[:, 1] = self._torch.where(
            due,
            (due & ~achieved_finite).to(self._torch.long),
            self._full_a_owner_fault_bits[:, 1],
        )
        self._full_a_owner_source_step[:, 1] = self._torch.where(
            due,
            self._torch.where(
                safe,
                self._torch.full_like(
                    self._full_a_r03_expected_source_step, step
                ),
                self._torch.full_like(
                    self._full_a_r03_expected_source_step, -1
                ),
            ),
            self._full_a_owner_source_step[:, 1],
        )
        self._full_a_r03_present.copy_(safe)
        consumed = self._full_a_r03_armed & (
            self._full_a_r03_expected_source_step <= step
        )
        self._full_a_r03_armed.logical_and_(~consumed)
        self._full_a_r03_expected_source_step.masked_fill_(consumed, -1)
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
            & (now == self._epoch_clock_ticks[:, 3])
        )
        selected_crossing_horizon = (
            active
            & finite
            & self._full_a_selected_racket_contact
            & ~self._full_a_landing_crossing_present
            & (now == self._epoch_clock_ticks[:, 4])
        )
        # Shared R06 owns two policy settlement clocks.  A no-contact row closes
        # when Motion's selected suffix closes; a selected-contact row without
        # a landing crossing remains live until the next frozen opportunity.
        # R07 is independently clocked from task close below.  A ball leaving
        # broad housekeeping bounds is not a third settlement authority.
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
        source_invalid = present & ~source_valid
        self._full_a_fact_integrity_fault_bits |= (
            source_invalid.to(torch.long)
            * FULL_A_FACT_INTEGRITY_R06_SOURCE_INVALID
        )
        self._full_a_r06_payment_event.copy_(present)
        self._full_a_owner_valid_bits[:, 2] = torch.where(
            present, bits, self._full_a_owner_valid_bits[:, 2]
        )
        self._full_a_owner_fault_bits[:, 2] = torch.where(
            present,
            source_invalid.to(torch.long),
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
        support = support_force.ge(FULL_A_SUPPORT_FORCE_N)
        cache = getattr(self, "_full_a_foot_supported_lr", None)
        if cache is not None:
            cache.copy_(support)
        slip = body_lin_vel[:, (3, 6), :2]
        return support, slip, invalid_force.eq(0)

    def _full_a_recovery_component_errors(self, tracked_body_kinematics=None):
        """Compute the shared thirteen recovery errors from live MuJoCo state."""

        torch = self._torch
        origins = self.env.scene.env_origins
        if tracked_body_kinematics is None:
            data = self.sim.data
            body_ids = self._fullmdp_body_ids
            body_pos = data.xpos[:, body_ids]
            body_quat = data.xquat[:, body_ids]
            body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        else:
            body_pos, body_quat, body_lin_vel, body_ang_vel = (
                tracked_body_kinematics
            )
        root_pos = body_pos[:, 0]
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

    def _full_a_update_cadence_readiness(
        self, errors, hard_safety_ok, terminated, truncated
    ) -> None:
        """Maintain the critic's two-transition recovery-readiness telemetry."""

        torch = self._torch
        candidate = (
            self._epoch_phase.eq(FULL_A_PHASE_IDLE)
            | self._epoch_phase.eq(FULL_A_PHASE_OUTCOME_SETTLED)
            | self._epoch_phase.eq(FULL_A_PHASE_RETIRED)
        )
        numeric = torch.isfinite(errors).all(dim=1)
        ready = (
            candidate
            & numeric
            & hard_safety_ok
            & ~terminated
            & ~truncated
            & (errors <= self._full_a_recovery_ready_tolerances).all(dim=1)
        )
        self._full_a_cadence_ready_streak.copy_(
            torch.where(
                ready,
                self._full_a_cadence_ready_streak + 1,
                torch.zeros_like(self._full_a_cadence_ready_streak),
            )
        )

    def _full_a_prime_cadence_readiness(self, ids) -> None:
        """Install the reset-return tick-zero readiness sample for selected rows."""

        if ids.numel() == 0:
            return
        torch = self._torch
        errors = self._full_a_recovery_component_errors()
        hard_safety_ok = self._full_a_recovery_joint_limit_ok()
        ready = (
            torch.isfinite(errors).all(dim=1)
            & hard_safety_ok
            & (errors <= self._full_a_recovery_ready_tolerances).all(dim=1)
        )
        self._full_a_cadence_ready_streak[ids] = ready[ids].to(torch.long)

    def _full_a_recovery_clock(self):
        """Return Motion's current accepted-task mask and task-close age."""

        task_close = self._epoch_clock_ticks[:, 3]
        task_owned = self._epoch_task_valid
        age = int(self.common_step_counter) - task_close
        return task_owned, age

    def _full_a_publish_r07_fact(self, errors=None, hard_safety_ok=None):
        """Publish Motion's complete task-clock recovery tape at ages 10..77."""

        torch = self._torch
        task_owned, age = self._full_a_recovery_clock()
        expected = (
            task_owned
            & (age >= FULL_A_RECOVERY_START_AGE_TICK)
            & (age <= FULL_A_RECOVERY_END_AGE_TICK)
        )
        if errors is None:
            errors = self._full_a_recovery_component_errors()
        if hard_safety_ok is None:
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
        nonfinite_fault = expected & ~valid
        self._full_a_fact_integrity_fault_bits |= (
            sequence_fault.to(torch.long) * FULL_A_FACT_INTEGRITY_R07_SEQUENCE
            + nonfinite_fault.to(torch.long) * FULL_A_FACT_INTEGRITY_R07_NONFINITE
        )
        self._full_a_recovery_sticky_fault |= sequence_fault | nonfinite_fault
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
        """Retire only after R06 and Motion's complete 68-cell R07 tape."""

        torch = self._torch
        task_owned, age = self._full_a_recovery_clock()
        r06_settled = self._epoch_phase.eq(FULL_A_PHASE_OUTCOME_SETTLED)
        invalid_outcome = r06_settled & self._full_a_outcome_code.eq(
            FULL_A_OUTCOME_INVALID
        )
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
            task_owned
            & age.eq(FULL_A_RECOVERY_END_AGE_TICK)
            & ~terminated
            & ~truncated
        )
        completion_fault = completion_due & ~complete_window
        # Keep the hot path device-only, but do not translate an internal
        # chronology fault into a business recovery verdict.  The named event
        # returned below is accumulated by the ledger and rejected at its one
        # pre-optimizer host boundary.
        self._full_a_recovery_sticky_fault |= completion_fault
        # A Gym terminal interrupts an already-settled recovery immediately for
        # telemetry/reset, but it is never a nonterminating shot retirement.
        # Natural retirement remains the strict R06 + complete-68-cell join.
        recovery = r06_settled & (complete_window | terminated | truncated)
        terminal, success, failure, timeout = portable_outcome.recovery_status(
            torch=torch,
            recovering=recovery,
            age=age,
            terminated=terminated,
            truncated=truncated,
            ready_seen=self._full_a_recovery_ready_seen,
            end_age=FULL_A_RECOVERY_END_AGE_TICK,
        )
        terminal &= ~completion_fault
        invalid_terminal = terminal & invalid_outcome
        success &= ~invalid_outcome & ~completion_fault
        failure = (failure | invalid_terminal) & ~completion_fault
        timeout &= ~invalid_outcome & ~completion_fault
        # A physical/Gym terminal interrupts recovery and true-resets the row;
        # it is telemetry, not a nonterminating shot retirement.  Only the
        # naturally joined R06+68-cell window enters RETIRED.
        retired = terminal & ~terminated & ~truncated
        self._epoch_phase.copy_(
            torch.where(
                retired,
                torch.full_like(self._epoch_phase, FULL_A_PHASE_RETIRED),
                self._epoch_phase,
            )
        )
        return terminal, success, failure, timeout, completion_fault

    def _full_a_completed_action_epoch(self, shot_retired):
        """Close one same-row business epoch; this is evidence, not shot success."""

        r03 = portable_reward.R03_PRESENT | portable_reward.R03_PHYSICALLY_VALID
        r06 = (
            portable_reward.R06_PRESENT
            | portable_reward.R06_POLICY_ELIGIBLE
            | portable_reward.R06_SOURCE_VALID
        )
        expected_cells = (
            FULL_A_RECOVERY_END_AGE_TICK - FULL_A_RECOVERY_START_AGE_TICK + 1
        )
        return (
            shot_retired
            & self._epoch_launch_succeeded
            & self._full_a_selected_racket_contact
            & self._torch.bitwise_and(self._full_a_owner_valid_bits[:, 1], r03).eq(r03)
            & self._full_a_owner_fault_bits[:, 1].eq(0)
            & self._torch.bitwise_and(self._full_a_owner_valid_bits[:, 2], r06).eq(r06)
            & self._full_a_owner_fault_bits[:, 2].eq(0)
            & self._full_a_recovery_expected_count.eq(expected_cells)
            & self._full_a_recovery_eligible_count.eq(expected_cells)
            & self._full_a_recovery_last_age.eq(FULL_A_RECOVERY_END_AGE_TICK)
            & ~self._full_a_recovery_sticky_fault
        )

    def enable_diagnostic_direct_frame0_playback(self) -> None:
        """Enable one N=1 replay-only frame-zero plant intervention.

        The production FullMDP step never calls this method.  Allocation and
        state mutation exist only after the explicit teacher-replay CLI selects
        ``direct_frame0_playback``.
        """

        if (
            not getattr(self, "full_a_mode", False)
            or self.num_envs != teacher_replay.TEACHER_REPLAY_NUM_ENVS
            or hasattr(self, "_diagnostic_direct_frame0_consumed")
        ):
            raise RuntimeError("direct frame-zero diagnostic enable differs")
        self._diagnostic_direct_frame0_consumed = self._torch.zeros(
            self.num_envs, dtype=self._torch.bool, device=self.device
        )

    def _diagnostic_direct_frame0_preserved_state(self, ids):
        """Clone the ball, accepted task, and lifecycle around one intervention."""

        data = self.sim.data
        ball = (
            data.qpos[ids, self.b_q : self.b_q + 7].clone(),
            data.qvel[ids, self.b_v : self.b_v + 6].clone(),
            data.qacc_warmstart[ids, self.b_v : self.b_v + 6].clone(),
        )
        task = tuple(value[ids].clone() for value in (
            self._epoch_task_f32,
            self._full_a_launch_state_f32,
            self._full_a_action_slot,
            self._full_a_action_uid,
            self._full_a_mount_normal_sign,
            self._full_a_teacher_rate,
            self._full_a_scaled_t_hit_s,
            self._full_a_scaled_t_cycle_s,
            self._full_a_pre_swing_wait_s,
        ))
        lifecycle = tuple(value[ids].clone() for value in (
            self._epoch_phase,
            self._epoch_task_valid,
            self._epoch_selected,
            self._epoch_launch_succeeded,
            self._epoch_clock_ticks,
            self.reset_generation,
            self.episode_length_buf,
            self.ball_age_buf,
            self._full_a_physical_present,
            self._full_a_owner_valid_bits,
            self._full_a_owner_fault_bits,
            self._full_a_owner_source_step,
        ))
        return ball, task, lifecycle, int(self.common_step_counter)

    def _diagnostic_direct_frame0_boundary_report(self, ids):
        """Describe the exact actor boundary used by the frame-zero probe.

        Waiting rows are parked at the *start* of every production transition.
        The returned actor boundary is consequently one integrated policy step
        later: the still-logical park row has ``ball_age == 1`` and gravity has
        moved its free joint away from the canonical park bytes.  The probe may
        inspect that finite state, but it must preserve the bytes exactly; the
        next production ``_full_a_prepare_step`` will park it again.
        """

        torch = self._torch
        data = self.sim.data

        def host_bool(value):
            return bool(value.detach().cpu().item())

        def host_int(value):
            return int(value.detach().cpu().item())

        def host_float(value):
            result = float(value.detach().cpu().item())
            return result if math.isfinite(result) else None

        expected_ball_position = (
            self.env.scene.env_origins[ids] + self._full_a_park_position_scene
        )
        expected_ball_quaternion = self._full_a_park_quaternion.expand(
            ids.numel(), 4
        )
        ball_position = data.qpos[ids, self.b_q : self.b_q + 3]
        ball_quaternion = data.qpos[ids, self.b_q + 3 : self.b_q + 7]
        ball_velocity = data.qvel[ids, self.b_v : self.b_v + 6]
        quaternion_norm = torch.linalg.vector_norm(ball_quaternion, dim=1)
        frozen_steps = teacher_replay.remaining_teacher_frozen_steps(
            torch=torch,
            common_step=int(self.common_step_counter),
            reveal_tick=self._epoch_clock_ticks[ids, 0],
            pre_swing_wait_s=self._full_a_pre_swing_wait_s[ids],
            step_dt=float(self.step_dt),
        )
        checks = {
            "consumed_clear": host_bool(
                ~self._diagnostic_direct_frame0_consumed[ids].any()
            ),
            "task_valid": host_bool(self._epoch_task_valid[ids].all()),
            "phase_reveal_committed": host_bool(
                self._epoch_phase[ids]
                .eq(FULL_A_PHASE_REVEAL_COMMITTED)
                .all()
            ),
            "launch_not_succeeded": host_bool(
                ~self._epoch_launch_succeeded[ids].any()
            ),
            "physical_not_present": host_bool(
                ~self._full_a_physical_present[ids].any()
            ),
            "ball_age_one": host_bool(self.ball_age_buf[ids].eq(1).all()),
            "ball_position_finite": host_bool(
                torch.isfinite(ball_position).all()
            ),
            "ball_quaternion_finite": host_bool(
                torch.isfinite(ball_quaternion).all()
            ),
            "ball_quaternion_nonzero": host_bool(
                quaternion_norm.gt(0).all()
            ),
            "ball_velocity_finite": host_bool(
                torch.isfinite(ball_velocity).all()
            ),
            "frozen_steps_zero": host_bool(frozen_steps.eq(0).all()),
            "teacher_frame_zero": host_bool(
                self._full_a_teacher_frame[ids].eq(0).all()
            ),
            "motion_phase_prepare": host_bool(
                self._full_a_motion_phase_code[ids]
                .eq(FULL_A_MOTION_PREPARE_PHASE_INDEX)
                .all()
            ),
            "actuator_state_absent": int(self.mj_model.na) == 0,
        }
        return {
            "schema": "action_ball_direct_frame0_boundary_v2",
            "checks": checks,
            "actual": {
                "common_step": int(self.common_step_counter),
                "phase": host_int(self._epoch_phase[ids][0]),
                "motion_phase": host_int(
                    self._full_a_motion_phase_code[ids][0]
                ),
                "teacher_frame": host_int(self._full_a_teacher_frame[ids][0]),
                "ball_age": host_int(self.ball_age_buf[ids][0]),
                "frozen_steps": host_int(frozen_steps[0]),
                "mj_model_na": int(self.mj_model.na),
                "ball_position_max_abs_delta_from_park_m": host_float(
                    torch.max(torch.abs(ball_position - expected_ball_position))
                ),
                "ball_quaternion_max_abs_delta_from_park": host_float(
                    torch.max(
                        torch.abs(ball_quaternion - expected_ball_quaternion)
                    )
                ),
                "ball_quaternion_norm": host_float(quaternion_norm[0]),
                "ball_linear_velocity_max_abs_mps": host_float(
                    torch.max(torch.abs(ball_velocity[:, :3]))
                ),
                "ball_angular_velocity_max_abs_radps": host_float(
                    torch.max(torch.abs(ball_velocity[:, 3:6]))
                ),
            },
        }

    def _diagnostic_direct_frame0_table_state(self):
        """Read table safety at the installed-frame-zero boundary.

        This deliberately bypasses the production latches: the replay probe
        needs attribution for the state immediately after ``sim.forward()``,
        but must not manufacture lifecycle debt before the first real physics
        transition.
        """

        keepout = self._table_keepout.sample(self.sim.data)
        contact_census = A3ReadyBallVecEnv._contact_census(self)
        resolved_table_contact = contact_census.robot_table_by_world.gt(0)
        expected_shape = (self.num_envs,)
        if (
            tuple(keepout.shape) != expected_shape
            or tuple(resolved_table_contact.shape) != expected_shape
        ):
            raise RuntimeError(
                "direct frame-zero installed table-state shape differs"
            )
        return keepout, resolved_table_contact

    @staticmethod
    def _diagnostic_direct_frame0_rows_equal(torch, before, after):
        """Return one exact equality bit per selected environment."""

        if len(before) != len(after):
            raise RuntimeError("direct frame-zero preserved rows differ")
        comparisons = []
        for left, right in zip(before, after):
            comparisons.append(
                left.eq(right).reshape(left.shape[0], -1).all(dim=1)
            )
        return torch.stack(comparisons, dim=1).all(dim=1)

    def install_diagnostic_direct_frame0_playback(self, ids):
        """Atomically install measured Motion frame zero for a replay only.

        Root pose and the 31 runtime-ordered joints become frame zero, while
        root/joint velocities and solver/controller history are reset.  Ball,
        accepted task, and lifecycle bytes must remain unchanged across the
        write and the required derived-state ``sim.forward()``.
        """

        torch = self._torch
        data = self.sim.data
        consumed = getattr(self, "_diagnostic_direct_frame0_consumed", None)
        if (
            consumed is None
            or ids.ndim != 1
            or ids.dtype != torch.long
            or ids.numel() != 1
            or int(ids[0].detach().cpu().item()) != 0
        ):
            raise RuntimeError("direct frame-zero diagnostic boundary differs")
        boundary_report = self._diagnostic_direct_frame0_boundary_report(ids)
        if not all(boundary_report["checks"].values()):
            raise RuntimeError(
                "direct frame-zero diagnostic boundary differs: "
                + json.dumps(
                    boundary_report,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )

        before_ball, before_task, before_lifecycle, before_step = (
            self._diagnostic_direct_frame0_preserved_state(ids)
        )
        pelvis_index = FULLMDP_TRACKED_BODY_NAMES.index("pelvis_link")
        joint_q0 = self._full_a_teacher.joint_pos[0].unsqueeze(0)
        root_pos_q0 = (
            self._full_a_teacher.body_pos_w[0, pelvis_index].unsqueeze(0)
            + self.env.scene.env_origins[ids]
        )
        root_quat_q0 = self._full_a_teacher.body_quat_w[
            0, pelvis_index
        ].unsqueeze(0)
        frame0_action = (
            joint_q0 - self.action_offset.unsqueeze(0)
        ) / self.act_scale.unsqueeze(0)
        root_quat_norm = torch.linalg.vector_norm(root_quat_q0, dim=1)
        if (
            not bool(torch.isfinite(joint_q0).all())
            or not bool(torch.isfinite(root_pos_q0).all())
            or not bool(torch.isfinite(root_quat_q0).all())
            or not bool(torch.isfinite(self.act_scale).all())
            or bool(self.act_scale.eq(0).any())
            or not bool(torch.isfinite(frame0_action).all())
            or not bool(
                torch.isclose(
                    root_quat_norm,
                    torch.ones_like(root_quat_norm),
                    rtol=0.0,
                    atol=1.0e-5,
                ).all()
            )
            or not bool(
                (
                    joint_q0.ge(self.jnt_lo.unsqueeze(0))
                    & joint_q0.le(self.jnt_hi.unsqueeze(0))
                ).all()
            )
        ):
            raise RuntimeError("direct frame-zero target is invalid")
        current_joint = self._qpos_act()[ids].clone()
        current_root_pos = data.qpos[
            ids, self.root_qadr : self.root_qadr + 3
        ].clone()
        current_root_quat = data.qpos[
            ids, self.root_qadr + 3 : self.root_qadr + 7
        ].clone()

        def quat_error(observed):
            direct = torch.linalg.vector_norm(observed - root_quat_q0, dim=1)
            antipodal = torch.linalg.vector_norm(observed + root_quat_q0, dim=1)
            return torch.minimum(direct, antipodal)

        joint_error_before = torch.max(
            torch.abs(current_joint - joint_q0), dim=1
        ).values
        root_pos_error_before = torch.linalg.vector_norm(
            current_root_pos - root_pos_q0, dim=1
        )
        root_quat_error_before = quat_error(current_root_quat)

        data.qpos[ids, self.root_qadr : self.root_qadr + 3] = root_pos_q0
        data.qpos[ids, self.root_qadr + 3 : self.root_qadr + 7] = root_quat_q0
        if self._q_slice is not None:
            data.qpos[ids, self._q_slice] = joint_q0
        else:
            data.qpos[ids[:, None], self.q_adr_act[None, :]] = joint_q0
        data.qvel[ids, self.root_vadr : self.root_vadr + 6] = 0.0
        if self._v_slice is not None:
            data.qvel[ids, self._v_slice] = 0.0
        else:
            data.qvel[ids[:, None], self.v_adr_act[None, :]] = 0.0
        def clear_robot_warmstart():
            data.qacc_warmstart[
                ids, self.root_vadr : self.root_vadr + 6
            ] = 0.0
            if self._v_slice is not None:
                data.qacc_warmstart[ids, self._v_slice] = 0.0
            else:
                data.qacc_warmstart[
                    ids[:, None], self.v_adr_act[None, :]
                ] = 0.0

        clear_robot_warmstart()
        data.ctrl[ids] = 0.0

        self.actions[ids] = frame0_action
        self.last_actions[ids] = frame0_action
        self.action_nonfinite_buf[ids] = False
        self._qdes_previous_executable[ids] = joint_q0
        self._qdes_previous_executable_valid[ids] = True
        self._qdes_guard_terminal[ids] = False
        self._qdes_guard_intervention[ids] = False
        self._actual_hard_edge_latch[ids] = False
        self._qdes_reward_processed[ids] = joint_q0
        self._qdes_reward_pre_clamp[ids] = joint_q0
        self._qdes_reward_nominal_projected[ids] = joint_q0
        self._qdes_reward_operand_valid[ids] = False
        self._controller_trace_latest = None
        self.sim.forward()
        if self._cap_ok:
            self._probe_capacity("forward")
        # ``forward`` may refresh solver guesses.  The next physical transition
        # must start with no robot warm-start debt, while the parked ball keeps
        # precisely the solver history it had before this robot-only probe.
        clear_robot_warmstart()
        data.qacc_warmstart[ids, self.b_v : self.b_v + 6] = before_ball[2]
        data.ctrl[ids] = 0.0
        (
            installed_table_keepout,
            installed_resolved_table_contact,
        ) = self._diagnostic_direct_frame0_table_state()
        installed_table_keepout = installed_table_keepout[ids].clone()
        installed_resolved_table_contact = (
            installed_resolved_table_contact[ids].clone()
        )
        self._refresh_aligned_teacher_body_pose()
        self._compute_obs()

        after_ball, after_task, after_lifecycle, after_step = (
            self._diagnostic_direct_frame0_preserved_state(ids)
        )
        ball_unchanged = self._diagnostic_direct_frame0_rows_equal(
            torch, before_ball, after_ball
        )
        task_unchanged = self._diagnostic_direct_frame0_rows_equal(
            torch, before_task, after_task
        )
        lifecycle_unchanged = self._diagnostic_direct_frame0_rows_equal(
            torch, before_lifecycle, after_lifecycle
        ) & torch.full(
            (ids.numel(),),
            before_step == after_step,
            dtype=torch.bool,
            device=self.device,
        )
        joint_error_after = torch.max(
            torch.abs(self._qpos_act()[ids] - joint_q0), dim=1
        ).values
        root_pos_error_after = torch.linalg.vector_norm(
            data.qpos[ids, self.root_qadr : self.root_qadr + 3] - root_pos_q0,
            dim=1,
        )
        root_quat_error_after = quat_error(
            data.qpos[ids, self.root_qadr + 3 : self.root_qadr + 7]
        )
        frame0_pose_exact = (
            self._qpos_act()[ids].eq(joint_q0).all(dim=1)
            & data.qpos[ids, self.root_qadr : self.root_qadr + 3]
            .eq(root_pos_q0)
            .all(dim=1)
            & data.qpos[ids, self.root_qadr + 3 : self.root_qadr + 7]
            .eq(root_quat_q0)
            .all(dim=1)
        )
        robot_velocity_zero = (
            data.qvel[ids, self.root_vadr : self.root_vadr + 6]
            .eq(0)
            .all(dim=1)
            & self._qvel_act()[ids].eq(0).all(dim=1)
        )
        root_warmstart_zero = data.qacc_warmstart[
            ids, self.root_vadr : self.root_vadr + 6
        ].eq(0).all(dim=1)
        if self._v_slice is not None:
            joint_warmstart_zero = data.qacc_warmstart[
                ids, self._v_slice
            ].eq(0).all(dim=1)
        else:
            joint_warmstart_zero = data.qacc_warmstart[
                ids[:, None], self.v_adr_act[None, :]
            ].eq(0).all(dim=1)
        robot_qacc_warmstart_zero = (
            root_warmstart_zero & joint_warmstart_zero
        )
        ctrl_zero = data.ctrl[ids].eq(0).all(dim=1)
        controller_history_exact = (
            self.actions[ids].eq(frame0_action).all(dim=1)
            & self.last_actions[ids].eq(frame0_action).all(dim=1)
            & self._qdes_previous_executable[ids].eq(joint_q0).all(dim=1)
            & self._qdes_previous_executable_valid[ids]
            & ~self.action_nonfinite_buf[ids]
            & ~self._qdes_guard_terminal[ids]
            & ~self._qdes_guard_intervention[ids]
            & ~self._actual_hard_edge_latch[ids]
            & self._qdes_reward_processed[ids].eq(joint_q0).all(dim=1)
            & self._qdes_reward_pre_clamp[ids].eq(joint_q0).all(dim=1)
            & self._qdes_reward_nominal_projected[ids]
            .eq(joint_q0)
            .all(dim=1)
            & ~self._qdes_reward_operand_valid[ids]
        )
        consumed[ids] = True
        return {
            "applied": torch.ones(
                ids.numel(), dtype=torch.bool, device=self.device
            ),
            "joint_q0_error_max_before_rad": joint_error_before,
            "joint_q0_error_max_after_rad": joint_error_after,
            "root_position_q0_error_before_m": root_pos_error_before,
            "root_position_q0_error_after_m": root_pos_error_after,
            "root_quaternion_q0_error_before": root_quat_error_before,
            "root_quaternion_q0_error_after": root_quat_error_after,
            "ball_unchanged": ball_unchanged,
            "task_unchanged": task_unchanged,
            "lifecycle_unchanged": lifecycle_unchanged,
            "frame0_pose_exact": frame0_pose_exact,
            "robot_velocity_zero": robot_velocity_zero,
            "robot_qacc_warmstart_zero": robot_qacc_warmstart_zero,
            "ctrl_zero": ctrl_zero,
            "controller_history_exact": controller_history_exact,
            "teacher_cache_refreshed": torch.ones(
                ids.numel(), dtype=torch.bool, device=self.device
            ),
            "actuator_state_absent": torch.ones(
                ids.numel(), dtype=torch.bool, device=self.device
            ),
            "installed_frame0_table_keepout": installed_table_keepout,
            "installed_frame0_backend_resolved_table_contact": (
                installed_resolved_table_contact
            ),
            "frame0_joint_qdes": joint_q0.clone(),
        }

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
        (
            ready_racket_pos_scene,
            _ready_racket_velocity,
            ready_racket_raw_normal,
            ready_racket_long_axis,
        ) = self._full_a_racket_kinematics()
        self._ready_teacher_racket_site_pos_w = (
            ready_racket_pos_scene + self.env.scene.env_origins
        ).detach().clone()
        self._ready_teacher_racket_signed_normal_w = (
            ready_racket_raw_normal.detach().clone()
        )
        self._ready_teacher_racket_long_axis_w = (
            ready_racket_long_axis.detach().clone()
        )
        self._teacher_racket_site_pos_w = (
            self._ready_teacher_racket_site_pos_w.clone()
        )
        self._teacher_racket_site_lin_vel_w = self._torch.zeros_like(
            self._ready_teacher_racket_site_pos_w
        )
        self._teacher_racket_signed_normal_w = (
            self._ready_teacher_racket_signed_normal_w.clone()
        )
        self._teacher_racket_long_axis_w = (
            self._ready_teacher_racket_long_axis_w.clone()
        )
        self._refresh_aligned_teacher_body_pose()

    def _full_a_update_teacher(self) -> None:
        """Advance the selected measured teacher on the task's exact clock."""

        if not getattr(self, "full_a_mode", False):
            return
        torch = self._torch
        valid = self._epoch_task_valid
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
        # Before any reveal, joint and body teachers describe the same
        # reset-ready pose.  Reveal switches both to action frame zero for the
        # stationary pre-swing preparation window; mixing those authorities
        # makes orientation imitation contradictory.
        # Motion owns its accepted clip clock independently of the physical
        # ball outcome.  An early outcome starts recovery accounting but must
        # not truncate the measured follow-through teacher.
        motion_open = valid & (
            int(self.common_step_counter) <= self._epoch_clock_ticks[:, 3]
        )
        in_hold = motion_open & (
            elapsed <= self._full_a_pre_swing_wait_s + 1.0e-12
        )
        # Playback starts only when the measured teacher has actually left
        # frame zero.  This is the same observable edge used by Isaac Motion;
        # treating a positive sub-frame clock as playback would multiply the
        # paddle reward before the teacher pose changes.
        playback_started = motion_open & ~in_hold & sampled["frame"].gt(0)
        prepare = motion_open & ~playback_started
        swing = (
            playback_started
            & (
                elapsed
                <= self._full_a_pre_swing_wait_s
                + self._full_a_scaled_t_hit_s
                + 1.0e-12
            )
        )
        follow = motion_open & ~prepare & ~swing
        # Hidden rows always track the reset-ready teacher.  R07 readiness is
        # recovery telemetry/reward, not an actor phase authority or a task
        # exposure gate.
        hidden_phase = torch.full_like(
            self._full_a_motion_phase_code, READY_HOLD_PHASE_INDEX
        )
        phase = hidden_phase
        phase = torch.where(
            follow,
            torch.full_like(phase, FULL_A_MOTION_FOLLOW_PHASE_INDEX),
            phase,
        )
        phase = torch.where(
            swing,
            torch.full_like(phase, FULL_A_MOTION_SWING_PHASE_INDEX),
            phase,
        )
        phase = torch.where(
            prepare,
            torch.full_like(phase, FULL_A_MOTION_PREPARE_PHASE_INDEX),
            phase,
        )
        self._full_a_motion_phase_code.copy_(phase)
        completed = valid & ~motion_open
        frame0_joint = self._full_a_teacher.joint_pos[0].unsqueeze(0)
        frame0_body_pos = self._full_a_teacher.body_pos_w[0].unsqueeze(0)
        frame0_body_quat = self._full_a_teacher.body_quat_w[0].unsqueeze(0)
        frame0_racket_pos = (
            self._full_a_teacher.measured_racket_site_pos_w[0].unsqueeze(0)
        )
        frame0_racket_normal = (
            self._full_a_teacher.measured_racket_normal_w[0].unsqueeze(0)
        )
        frame0_racket_long_axis = (
            self._full_a_teacher.measured_racket_long_axis_w[0].unsqueeze(0)
        )
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
                valid & ~completed & ~in_hold,
                sampled["frame"],
                torch.zeros_like(sampled["frame"]),
            )
        )
        # Reveal starts one stationary frame-zero preparation window.  Joint
        # and body targets switch together; the clip clock itself remains
        # frozen until the window ends.
        measured_joint = valid
        self._full_a_teacher_joint_pos.copy_(
            torch.where(
                measured_joint[:, None],
                measured_joint_pos,
                self.q_ready.unsqueeze(0).expand_as(
                    self._full_a_teacher_joint_pos
                ),
            )
        )
        measured_joint_vel = torch.where(
            (prepare | completed)[:, None],
            torch.zeros_like(sampled["joint_vel"]),
            sampled["joint_vel"],
        )
        self._full_a_teacher_joint_vel.copy_(
            torch.where(
                (measured_joint & ~prepare)[:, None],
                measured_joint_vel,
                torch.zeros_like(self._full_a_teacher_joint_vel),
            )
        )
        inactive_body_pos = torch.where(
            completed[:, None, None],
            frame0_body_pos + origins,
            self._ready_teacher_body_pos,
        )
        inactive_body_quat = torch.where(
            completed[:, None, None],
            frame0_body_quat.expand_as(self._teacher_body_quat),
            self._ready_teacher_body_quat,
        )
        self._teacher_body_pos.copy_(
            torch.where(
                (valid & ~completed)[:, None, None],
                measured_body_pos + origins,
                inactive_body_pos,
            )
        )
        self._teacher_body_quat.copy_(
            torch.where(
                (valid & ~completed)[:, None, None],
                measured_body_quat,
                inactive_body_quat,
            )
        )
        measured_body_lin_vel = torch.where(
            (prepare | completed)[:, None, None],
            torch.zeros_like(sampled["body_lin_vel_w"]),
            sampled["body_lin_vel_w"],
        )
        self._teacher_body_lin_vel.copy_(
            torch.where(
                (valid & ~completed)[:, None, None],
                measured_body_lin_vel,
                torch.zeros_like(self._teacher_body_lin_vel),
            )
        )
        measured_body_ang_vel = torch.where(
            (prepare | completed)[:, None, None],
            torch.zeros_like(sampled["body_ang_vel_w"]),
            sampled["body_ang_vel_w"],
        )
        self._teacher_body_ang_vel.copy_(
            torch.where(
                (valid & ~completed)[:, None, None],
                measured_body_ang_vel,
                torch.zeros_like(self._teacher_body_ang_vel),
            )
        )
        selected_racket_pos = torch.where(
            completed[:, None],
            frame0_racket_pos,
            sampled["measured_racket_site_pos_w"],
        )
        selected_racket_normal = torch.where(
            completed[:, None],
            frame0_racket_normal,
            sampled["measured_racket_normal_w"],
        )
        selected_racket_long_axis = torch.where(
            completed[:, None],
            frame0_racket_long_axis,
            sampled["measured_racket_long_axis_w"],
        )
        self._teacher_racket_site_pos_w.copy_(
            torch.where(
                valid[:, None],
                selected_racket_pos + origins[:, 0],
                self._ready_teacher_racket_site_pos_w,
            )
        )
        measured_racket_velocity = torch.where(
            (prepare | completed)[:, None],
            torch.zeros_like(sampled["measured_racket_site_lin_vel_w"]),
            sampled["measured_racket_site_lin_vel_w"],
        )
        self._teacher_racket_site_lin_vel_w.copy_(
            torch.where(
                valid[:, None],
                measured_racket_velocity,
                torch.zeros_like(self._teacher_racket_site_lin_vel_w),
            )
        )
        self._teacher_racket_signed_normal_w.copy_(
            torch.where(
                valid[:, None],
                selected_racket_normal,
                self._ready_teacher_racket_signed_normal_w,
            )
        )
        self._teacher_racket_long_axis_w.copy_(
            torch.where(
                valid[:, None],
                selected_racket_long_axis,
                self._ready_teacher_racket_long_axis_w,
            )
        )
        self._refresh_aligned_teacher_body_pose()

    def _full_a_paddle_prior_playback_mask(self):
        """Return Motion's open-playback rows without guessing from task state."""

        return self._full_a_motion_phase_code.eq(
            FULL_A_MOTION_SWING_PHASE_INDEX
        ) | self._full_a_motion_phase_code.eq(
            FULL_A_MOTION_FOLLOW_PHASE_INDEX
        )

    def _refresh_aligned_teacher_body_pose(self) -> None:
        """Publish body and measured-paddle caches under one alignment rule."""

        data = self.sim.data
        ids = self._fullmdp_body_ids
        anchor = self._fullmdp_anchor_index
        (
            delta_yaw,
            teacher_anchor_pos,
            aligned_anchor_pos,
        ) = self._teacher_yaw_alignment(
            self._torch,
            self._teacher_body_pos,
            self._teacher_body_quat,
            data.xpos[:, ids][:, anchor],
            data.xquat[:, ids][:, anchor],
            anchor,
        )
        aligned_pos, aligned_quat = self._apply_teacher_yaw_alignment(
            self._torch,
            self._teacher_body_pos,
            self._teacher_body_quat,
            delta_yaw,
            teacher_anchor_pos,
            aligned_anchor_pos,
        )
        self._aligned_teacher_body_pos = aligned_pos.detach().clone()
        self._aligned_teacher_body_quat = aligned_quat.detach().clone()
        self._aligned_teacher_racket_site_pos_w = (
            aligned_anchor_pos
            + self._quat_apply_wxyz(
                self._torch,
                delta_yaw,
                self._teacher_racket_site_pos_w - teacher_anchor_pos,
            )
        )
        self._aligned_teacher_racket_site_lin_vel_w = self._quat_apply_wxyz(
            self._torch, delta_yaw, self._teacher_racket_site_lin_vel_w
        )
        self._aligned_teacher_racket_signed_normal_w = self._quat_apply_wxyz(
            self._torch, delta_yaw, self._teacher_racket_signed_normal_w
        )
        self._aligned_teacher_racket_signed_normal_w = (
            self._aligned_teacher_racket_signed_normal_w
            / self._torch.linalg.vector_norm(
                self._aligned_teacher_racket_signed_normal_w,
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-6)
        )
        self._aligned_teacher_racket_long_axis_w = self._quat_apply_wxyz(
            self._torch, delta_yaw, self._teacher_racket_long_axis_w
        )
        self._aligned_teacher_racket_long_axis_w = (
            self._aligned_teacher_racket_long_axis_w
            / self._torch.linalg.vector_norm(
                self._aligned_teacher_racket_long_axis_w,
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-6)
        )

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

    def _fullmdp_tracked_body_kinematics(self):
        """Materialize one post-forward body pose/COM-velocity tuple."""

        data = self.sim.data
        ids = self._fullmdp_body_ids
        body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        return (
            data.xpos[:, ids],
            data.xquat[:, ids],
            body_lin_vel,
            body_ang_vel,
        )

    def _latch_post_forward_resolved_table_contacts(
        self, contact_census=None
    ):
        """Include final-integration table facts from the shared census."""

        if contact_census is None:
            contact_census = A3ReadyBallVecEnv._contact_census(self)
        resolved = contact_census.robot_table_by_world
        self._cur_robot_table.add_(resolved)
        return resolved

    def _after_physics_substep(
        self, substep_index, contact_census=None
    ) -> None:
        # MJWarp leaves derived poses at the pre-integration state.  Calls
        # 2..20 therefore expose post-states 1..19; the explicit final forward
        # below supplies post-state 20 without adding 20 redundant forwards.
        if substep_index > 0:
            keepout = self._table_keepout.sample(self.sim.data)
            self._cur_table_keepout |= keepout
            if getattr(self, "full_a_mode", False) and self._fullmdp_initialized:
                if contact_census is None:
                    raise RuntimeError(
                        "FullMDP physics substep contact census is absent"
                    )
                self._full_a_latch_ball_contacts(
                    contact_census,
                    diagnostic_substep_index=substep_index,
                    diagnostic_capture_boundary="physics_substep_poststate",
                )
                table_consumer = getattr(
                    self, "_diagnostic_table_attribution_consumer", None
                )
                if table_consumer is not None:
                    table_consumer(
                        keepout=keepout,
                        resolved=contact_census.robot_table_by_world.gt(0),
                        resolved_is_final=False,
                        substep_index=substep_index,
                        capture_boundary="physics_substep_poststate",
                    )

    def _latch_post_forward_table_keepout(self):
        keepout = self._table_keepout.sample(self.sim.data)
        self._cur_table_keepout |= keepout
        return keepout

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
    def _teacher_yaw_alignment(
        cls,
        torch,
        teacher_body_pos,
        teacher_body_quat,
        live_anchor_pos,
        live_anchor_quat,
        anchor_index,
    ):
        """Return MotionCommand's one live-anchor x/y plus yaw transform."""

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
        return delta_yaw, teacher_anchor_pos, aligned_anchor_pos

    @classmethod
    def _apply_teacher_yaw_alignment(
        cls,
        torch,
        teacher_body_pos,
        teacher_body_quat,
        delta_yaw,
        teacher_anchor_pos,
        aligned_anchor_pos,
    ):
        """Apply one already-materialized MotionCommand alignment."""

        body_delta = teacher_body_pos - teacher_anchor_pos[:, None, :]
        expanded_yaw = delta_yaw[:, None, :].expand_as(teacher_body_quat)
        aligned_pos = aligned_anchor_pos[:, None, :] + cls._quat_apply_wxyz(
            torch, expanded_yaw, body_delta
        )
        aligned_quat = cls._quat_mul_wxyz(
            torch, expanded_yaw, teacher_body_quat
        )
        return aligned_pos, aligned_quat

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

        delta_yaw, teacher_anchor_pos, aligned_anchor_pos = (
            cls._teacher_yaw_alignment(
                torch,
                teacher_body_pos,
                teacher_body_quat,
                live_anchor_pos,
                live_anchor_quat,
                anchor_index,
            )
        )
        return cls._apply_teacher_yaw_alignment(
            torch,
            teacher_body_pos,
            teacher_body_quat,
            delta_yaw,
            teacher_anchor_pos,
            aligned_anchor_pos,
        )

    def _serve(self, ids) -> None:
        """Keep every unrevealed ball at the canonical contact-free park."""

        torch = self._torch
        n = int(ids.numel())
        if n == 0:
            return
        full_a_initialized = (
            getattr(self, "full_a_mode", False) and self._fullmdp_initialized
        )
        if full_a_initialized:
            # Full-A owns scene-local state in a multi-world batch.  Reuse the
            # construction-time constants, while keeping origins live so a
            # reset cannot park every row in world zero.
            park_scene = self._full_a_park_position_scene
            park_position_w = (
                self.env.scene.env_origins[ids] + park_scene
            )
        else:
            # Base construction and the legacy WAIT lane reach this override
            # before Full-A state exists.  Preserve its state math and write
            # ordering exactly.
            park_hope = torch.tensor(
                WAIT_BALL_PARK_HOPE,
                dtype=self.qpos_init.dtype,
                device=self.device,
            )
            park_scene = park_hope + self.hope_to_scene
            park_position_w = park_scene.expand(n, 3)
        data = self.sim.data
        data.qpos[ids, self.b_q : self.b_q + 3] = park_position_w
        if full_a_initialized:
            park_quaternion = self._full_a_park_quaternion
        else:
            park_quaternion = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=self.qpos_init.dtype,
                device=self.device,
            )
        data.qpos[ids, self.b_q + 3 : self.b_q + 7] = (
            park_quaternion.expand(n, 4)
        )
        data.qvel[ids, self.b_v : self.b_v + 6] = 0.0
        self.ball_age_buf[ids] = 0

    def _full_a_semantic_observation_v3(self, st):
        """Pack the live Full-A plant into the shared semantic 215/231 ABI."""

        torch = self._torch
        data = self.sim.data
        dtype = self.qpos_init.dtype
        origins = self.env.scene.env_origins
        root_pos_w = st["base_pos"]
        root_pos_scene = root_pos_w - origins
        root_quat_w = st["base_quat"]
        heading_xy = observation_contract.heading_xy_from_quat_wxyz(root_quat_w)

        def heading(value):
            return observation_contract.rotate_world_to_heading_xy(
                heading_xy, value
            )

        # Epoch deliberately retains the current task through RETIRED until a
        # later ACCEPT replaces it.  Actor visibility is narrower: Motion's
        # prepare/swing/follow phases expose the task, while recover/ready hide
        # it.  Derive that view from the existing canonical phase rather than
        # adding a second validity owner.
        current_task = self._epoch_task_valid
        task_visible = current_task & self._full_a_motion_phase_code.lt(
            RECOVER_HIDDEN_PHASE_INDEX
        )
        task_mask = task_visible[:, None]
        recovery_observable = (
            current_task
            & self._epoch_phase.eq(FULL_A_PHASE_OUTCOME_SETTLED)
            & self._full_a_outcome_code.ne(FULL_A_OUTCOME_INVALID)
        )

        def task_vector(value):
            rotated = heading(value)
            return torch.where(task_mask, rotated, torch.zeros_like(rotated))

        anchor = self._fullmdp_anchor_index
        live_anchor_pos = data.xpos[:, self._fullmdp_anchor_body_id]
        live_anchor_quat = data.xquat[:, self._fullmdp_anchor_body_id]
        anchor_pos, anchor_ori = observation_contract.relative_pose_6d(
            live_anchor_pos,
            live_anchor_quat,
            self._teacher_body_pos[:, anchor],
            self._teacher_body_quat[:, anchor],
        )
        root_com_lin_vel_w = self._body_com_velocities_from_cvel(
            torch,
            data.cvel[:, self._fullmdp_pelvis_body_id],
            data.xipos[:, self._fullmdp_pelvis_body_id],
            data.subtree_com[:, self._fullmdp_pelvis_root_id],
        )[0]

        (
            racket_pos_scene,
            racket_vel_w,
            racket_raw_normal_w,
            racket_long_axis_w,
        ) = self._full_a_racket_kinematics()
        # The measured-paddle reward and observation consume one aligned
        # teacher tuple and one actual FK tuple.  Ready rows own the raw +Y
        # face, while rows with a selected Epoch own its physical mount sign;
        # neither convention is a visibility mask for these all-phase fields.
        racket_face_sign = torch.where(
            current_task,
            self._fullmdp_mount_normal_sign,
            torch.ones_like(self._fullmdp_mount_normal_sign),
        )
        racket_signed_normal_w = (
            racket_raw_normal_w * racket_face_sign[:, None]
        )
        teacher_racket_pos_scene = (
            self._aligned_teacher_racket_site_pos_w - origins
        )
        task = self._epoch_task_f32
        base_goal_delta = torch.cat(
            (
                task[:, 24:26] - root_pos_scene[:, :2],
                torch.zeros_like(root_pos_scene[:, :1]),
            ),
            dim=1,
        )
        elapsed_s = (
            int(self.common_step_counter) - self._epoch_clock_ticks[:, 0]
        ).to(dtype=dtype) * float(self.step_dt)
        time_to_contact = (
            self._epoch_clock_ticks[:, 1] - int(self.common_step_counter)
        ).to(dtype=dtype) * float(self.step_dt)
        time_to_contact = torch.where(
            task_visible, time_to_contact, torch.zeros_like(time_to_contact)
        )
        time_to_teacher_start = torch.clamp(
            self._full_a_pre_swing_wait_s - elapsed_s, min=0.0
        )
        time_to_teacher_start = torch.where(
            task_visible,
            time_to_teacher_start,
            torch.zeros_like(time_to_teacher_start),
        )
        time_to_next_opportunity = (
            self._full_a_next_reveal_tick - self.episode_length_buf
        ).to(dtype=dtype) * float(self.step_dt)
        schedule_exhausted = self._full_a_scheduled_ordinal.ge(
            len(self._full_a_cadence.reference_due_ticks) - 1
        )
        time_to_next_opportunity = torch.where(
            schedule_exhausted,
            portable_catalog.FRESH_SCHEDULE_EXHAUSTED_TIME_TO_NEXT_OPPORTUNITY_S,
            time_to_next_opportunity,
        )

        learning_phase = torch.zeros_like(self._epoch_phase)
        for index, code in enumerate(
            (
                FULL_A_PHASE_IDLE,
                FULL_A_PHASE_REVEAL_COMMITTED,
                FULL_A_PHASE_LAUNCH_SETTLED,
                FULL_A_PHASE_OUTCOME_SETTLED,
                FULL_A_PHASE_RETIRED,
            )
        ):
            learning_phase = torch.where(
                self._epoch_phase.eq(code),
                torch.full_like(learning_phase, index),
                learning_phase,
            )

        actor_rows = {
            "projected_gravity_b": st["proj_g"],
            "base_ang_vel_b": st["base_ang_b"],
            "base_position_table": (
                root_pos_scene - self._full_a_table_surface_center_scene
            ),
            "base_heading_table_xy": heading_xy,
            "base_com_lin_vel_heading": heading(root_com_lin_vel_w),
            "joint_pos_rel": self._qpos_act() - self.action_offset.unsqueeze(0),
            "joint_vel": self._qvel_act(),
            "last_action": self.actions,
            "teacher_joint_pos_rel": (
                self._full_a_teacher_joint_pos - self.action_offset.unsqueeze(0)
            ),
            "teacher_joint_vel": self._full_a_teacher_joint_vel,
            "motion_anchor_pos_b": anchor_pos,
            "motion_anchor_ori_b6": anchor_ori,
            "motion_phase_one_hot": torch.nn.functional.one_hot(
                self._full_a_motion_phase_code, num_classes=5
            ).to(dtype=dtype),
            "motion_racket_pos_error_heading": heading(
                teacher_racket_pos_scene - racket_pos_scene
            ),
            "motion_racket_vel_error_heading": heading(
                self._aligned_teacher_racket_site_lin_vel_w - racket_vel_w
            ),
            "motion_racket_signed_normal_error_heading": heading(
                self._aligned_teacher_racket_signed_normal_w
                - racket_signed_normal_w
            ),
            "motion_racket_long_axis_error_heading": heading(
                self._aligned_teacher_racket_long_axis_w - racket_long_axis_w
            ),
            "racket_target_pos_error_heading": task_vector(
                task[:, 5:8] - racket_pos_scene
            ),
            "racket_target_vel_error_heading": task_vector(
                task[:, 8:11] - racket_vel_w
            ),
            "racket_target_normal_error_heading": task_vector(
                task[:, 11:14] - racket_raw_normal_w
            ),
            "base_goal_error_heading_xy": task_vector(base_goal_delta)[:, :2],
            "time_to_contact_s": time_to_contact[:, None],
            "time_to_teacher_start_s": time_to_teacher_start[:, None],
            "time_to_next_opportunity_s": time_to_next_opportunity[:, None],
            "epoch_learning_phase_one_hot": torch.nn.functional.one_hot(
                learning_phase, num_classes=5
            ).to(dtype=dtype),
            "task_valid": task_visible[:, None].to(dtype=dtype),
        }
        policy = observation_contract.concatenate_layout_rows(
            observation_contract.ACTOR_LAYOUT_V3, actor_rows
        )
        policy.mul_(self._full_a_actor_scale_v3)

        ball_pos_w = data.qpos[:, self.b_q : self.b_q + 3]
        ball_quat_w = data.qpos[:, self.b_q + 3 : self.b_q + 7]
        ball_lin_vel_w = data.qvel[:, self.b_v : self.b_v + 3]
        ball_ang_vel_w = observation_contract.quat_rotate_wxyz(
            ball_quat_w, data.qvel[:, self.b_v + 3 : self.b_v + 6]
        )
        live_ball = (
            current_task
            & self._epoch_launch_succeeded
            & self._epoch_phase.eq(FULL_A_PHASE_LAUNCH_SETTLED)
        )
        ball9 = torch.cat(
            (
                heading(ball_pos_w - root_pos_w),
                heading(ball_lin_vel_w),
                heading(ball_ang_vel_w),
            ),
            dim=1,
        )
        ball9 = torch.where(live_ball[:, None], ball9, torch.zeros_like(ball9))

        def current_latch(value):
            # These critic bits describe the current live flight, not the
            # retained history of a shot whose outcome has already closed.
            return torch.where(live_ball, value, torch.zeros_like(value))[
                :, None
            ].to(dtype=dtype)

        critic_rows = {
            "episode_time_remaining_s": (
                (self.max_episode_length - self.episode_length_buf)
                .clamp_min(0)
                .to(dtype=dtype)[:, None]
                * float(self.step_dt)
            ),
            "live_ball_center_rel_root_heading": ball9[:, :3],
            "live_ball_lin_vel_heading": ball9[:, 3:6],
            "live_ball_ang_vel_heading": ball9[:, 6:9],
            "selected_rubber_contact_latched": current_latch(
                self._full_a_selected_racket_contact
            ),
            "net_crossed_latched": current_latch(self._full_a_net_crossed),
            "net_clear_latched": current_latch(self._full_a_net_clear),
            "foot_supported_lr": torch.where(
                recovery_observable[:, None],
                self._full_a_foot_supported_lr.to(dtype=dtype),
                torch.zeros_like(self._full_a_foot_supported_lr, dtype=dtype),
            ),
            "cadence_ready_dwell_fraction": (
                torch.where(
                    recovery_observable,
                    self._full_a_cadence_ready_streak.clamp(0, 2).to(
                        dtype=dtype
                    ),
                    torch.zeros_like(
                        self._full_a_cadence_ready_streak, dtype=dtype
                    ),
                )[:, None]
                / 2.0
            ),
        }
        critic_extension = observation_contract.concatenate_layout_rows(
            observation_contract.CRITIC_EXTENSION_LAYOUT_V3, critic_rows
        )
        critic_extension.mul_(self._full_a_critic_extension_scale_v3)
        critic = torch.cat((policy, critic_extension), dim=1)
        if tuple(policy.shape) != (
            self.num_envs,
            observation_contract.ACTOR_WIDTH_V3,
        ) or tuple(critic.shape) != (
            self.num_envs,
            observation_contract.CRITIC_WIDTH_V3,
        ):
            raise RuntimeError("portable Full-A semantic observation width differs")
        # The update ledger scans both stored observation groups before PPO;
        # keep this path device-only so a fault remains attributable there.
        self._obs_buf = policy
        self._critic_obs_buf = critic
        return policy

    def _compute_obs(self, st=None):
        """Publish legacy WAIT V1 or the initialized Full-A semantic V3."""

        torch = self._torch
        # Base construction calls reset() -> _compute_obs() before the
        # Full-A buffers can be installed.  That construction observation is
        # the ordinary deterministic WAIT surface; publish Full-A fields only
        # after the subclass has completed its atomic initialization.
        full_a_initialized = (
            getattr(self, "full_a_mode", False) and self._fullmdp_initialized
        )
        if full_a_initialized:
            return self._full_a_semantic_observation_v3(st or self._state())

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
        critic_rows = {
                name: torch.zeros(
                    (self.num_envs, width),
                    dtype=joint_pos_rel.dtype,
                    device=self.device,
                )
                for name, width in observation_contract.CRITIC_EXTENSION_LAYOUT_V1
            }
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

    def _fullmdp_regularization_reward_terms(self):
        """Return four configured component rows from the shared tensor kernels."""

        raw = self._torch.stack(
            (
                regularization.action_rate_l2(self.actions, self.last_actions),
                regularization.soft_limit_barrier_v2(
                    self._qdes_reward_processed,
                    self._fullmdp_regularization_soft_limits,
                    self.action_offset,
                    self._fullmdp_regularization_hard_limits,
                ),
                regularization.qdes_projection_penalty(
                    self._qdes_reward_pre_clamp,
                    self._qdes_reward_nominal_projected,
                    self._qdes_reward_projection_span,
                    self._qdes_reward_operand_valid,
                    self._qdes_reward_operand_valid,
                ),
                regularization.soft_limit_barrier_v2(
                    self._qpos_act(),
                    self._fullmdp_regularization_soft_limits,
                    self.action_offset,
                    self._fullmdp_regularization_hard_limits,
                ),
            ),
            dim=1,
        )
        return raw * self._fullmdp_regularization_weights * self.step_dt

    def _fullmdp_reward(
        self,
        racket_kinematics=None,
        tracked_body_kinematics=None,
        *,
        return_paddle_error=False,
    ):
        """Return the shared ordered Reward28 graph and its exact term vector."""

        torch = self._torch
        if tracked_body_kinematics is None:
            data = self.sim.data
            ids = self._fullmdp_body_ids
            body_pos = data.xpos[:, ids]
            body_quat = data.xquat[:, ids]
            body_lin_vel, body_ang_vel = self._body_com_velocities_w()
        else:
            body_pos, body_quat, body_lin_vel, body_ang_vel = (
                tracked_body_kinematics
            )
        anchor = self._fullmdp_anchor_index
        non_wrist = self._fullmdp_upper_non_wrist_body_indices
        body_orientation_error = self._quat_error_sq(
            torch,
            self._aligned_teacher_body_quat[:, non_wrist],
            body_quat[:, non_wrist],
        ).mean(-1)
        raw = torch.stack(
            (
                torch.exp(
                    -torch.sum(
                        torch.square(
                            self._teacher_body_pos[:, anchor] - body_pos[:, anchor]
                        ),
                        dim=-1,
                    )
                    / FULLMDP_DENSE_REWARD_SPECS[0].std**2
                ),
                torch.exp(
                    -self._quat_error_sq(
                        torch,
                        self._teacher_body_quat[:, anchor],
                        body_quat[:, anchor],
                    )
                    / FULLMDP_DENSE_REWARD_SPECS[1].std**2
                ),
                torch.exp(
                    -torch.sum(
                        torch.square(
                            self._aligned_teacher_body_pos[:, non_wrist]
                            - body_pos[:, non_wrist]
                        ),
                        dim=-1,
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[2].std**2
                ),
                0.5
                * (
                    torch.exp(
                        -body_orientation_error
                        / FULLMDP_DENSE_REWARD_SPECS[3].std**2
                    )
                    + torch.exp(
                        -body_orientation_error
                        / FULLMDP_DENSE_REWARD_SPECS[3].coarse_std**2
                    )
                ),
                torch.exp(
                    -torch.sum(
                        torch.square(
                            self._teacher_body_lin_vel[:, non_wrist]
                            - body_lin_vel[:, non_wrist]
                        ),
                        dim=-1,
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[4].std**2
                ),
                torch.exp(
                    -torch.sum(
                        torch.square(
                            self._teacher_body_ang_vel[:, non_wrist]
                            - body_ang_vel[:, non_wrist]
                        ),
                        dim=-1,
                    ).mean(-1)
                    / FULLMDP_DENSE_REWARD_SPECS[5].std**2
                ),
            ),
            dim=1,
        )
        configured = raw * self._fullmdp_dense_weights * self.step_dt
        if racket_kinematics is None:
            racket_kinematics = self._full_a_racket_kinematics()
        (
            racket_pos_scene,
            racket_velocity,
            racket_raw_normal,
            racket_long_axis,
        ) = racket_kinematics
        racket_face_sign = self._fullmdp_mount_normal_sign
        if getattr(self, "full_a_mode", False):
            racket_face_sign = torch.where(
                self._epoch_task_valid,
                racket_face_sign,
                torch.ones_like(racket_face_sign),
            )
        racket_signed_normal = racket_raw_normal * racket_face_sign[:, None]
        teacher_racket_pos_scene = (
            self._aligned_teacher_racket_site_pos_w
            - self.env.scene.env_origins
        )
        paddle_error = paddle_prior.tracking_errors(
            racket_pos_scene,
            racket_velocity,
            racket_signed_normal,
            racket_long_axis,
            teacher_racket_pos_scene,
            self._aligned_teacher_racket_site_lin_vel_w,
            self._aligned_teacher_racket_signed_normal_w,
            self._aligned_teacher_racket_long_axis_w,
        )
        paddle_raw = paddle_prior.kernels(
            paddle_error,
            precision_stds=self._fullmdp_paddle_precision_stds,
            coarse_stds=self._fullmdp_paddle_coarse_stds,
        )
        if getattr(self, "full_a_mode", False):
            paddle_playback_mask = (
                FullMdpInitialWaitVecEnv
                ._full_a_paddle_prior_playback_mask(self)
            )
            if (
                type(paddle_playback_mask) is not torch.Tensor
                or tuple(paddle_playback_mask.shape) != (self.num_envs,)
                or paddle_playback_mask.device != self.device
                or paddle_playback_mask.dtype is not torch.bool
                or not paddle_playback_mask.is_contiguous()
            ):
                raise RuntimeError("FullMDP paddle playback Reward ABI differs")
            paddle_raw = torch.where(
                paddle_playback_mask[:, None],
                paddle_raw * self._fullmdp_paddle_playback_scales[None, :],
                paddle_raw,
            )
        paddle_configured = (
            paddle_raw
            * self._fullmdp_paddle_weights
            * self.step_dt
        )
        terms = torch.zeros(
            (self.num_envs, reward_contract.REWARD_TERM_COUNT),
            dtype=raw.dtype,
            device=self.device,
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
            payment_valid_bits[:, 1] = self._torch.where(
                self._full_a_r03_present,
                payment_valid_bits[:, 1],
                self._torch.zeros_like(payment_valid_bits[:, 1]),
            )
            payment_valid_bits[:, 2] = self._torch.where(
                self._full_a_r06_payment_event,
                payment_valid_bits[:, 2],
                self._torch.zeros_like(payment_valid_bits[:, 2]),
            )
            clean_recovery = self._epoch_phase.eq(
                FULL_A_PHASE_OUTCOME_SETTLED
            ) & self._full_a_outcome_code.ne(FULL_A_OUTCOME_INVALID)
            payment_valid_bits[:, 3] = self._torch.where(
                clean_recovery,
                payment_valid_bits[:, 3],
                self._torch.zeros_like(payment_valid_bits[:, 3]),
            )
            terms[:, : reward_contract.LIFECYCLE_PAYMENT_COUNT] = (
                portable_reward.lifecycle_reward14(
                    valid_bits=payment_valid_bits,
                    fact_f32=self._full_a_owner_fact_f32,
                    owner_fault_bits=self._full_a_owner_fault_bits,
                    step_dt=self.step_dt,
                    weights=self._full_a_lifecycle_reward_weights,
                )
            )
        common_start = reward_contract.LIFECYCLE_PAYMENT_COUNT
        paddle_start = common_start + len(FULLMDP_DENSE_REWARD_SPECS)
        paddle_end = paddle_start + len(FULLMDP_PADDLE_REWARD_SPECS)
        terms[:, common_start:paddle_start] = configured
        terms[:, paddle_start:paddle_end] = paddle_configured
        terms[:, paddle_end:] = self._fullmdp_regularization_reward_terms()
        reward = terms.sum(dim=1)
        # Do not synchronize or poison the CUDA context here.  The bound
        # FullMdpUpdateLedger accumulates per-row reward finiteness on device
        # and performs the single, attributed host reduction before PPO's
        # optimizer step.
        if return_paddle_error:
            return reward, terms, paddle_error
        return reward, terms

    def _fullmdp_termination(self, st, requested_qdes):
        torch = self._torch
        timeout = self.episode_length_buf >= self.max_episode_length
        tilt = torch.acos((-st["proj_g"][:, 2]).clamp(-1.0, 1.0)) > 0.7
        low = st["base_pos"][:, 2] < 0.5
        # ``_advance_plant`` keeps the projection/brake executable for every
        # guard verdict.  In projection mode only a non-finite request is a
        # q_des safety violation; predicted/current hard-inner crossings are
        # recoverable telemetry.  A raw mechanical-edge readback from the
        # current state or any physics substep is latched into Full-A evidence,
        # but the exact Isaac FullMDP manager does not turn that evidence into
        # a Done bit.
        qdes_safety = self._qdes_guard_terminal
        keepout = self._cur_table_keepout
        resolved_table = self._cur_robot_table > 0
        terminated = tilt | low | qdes_safety | keepout | resolved_table
        # RSL-RL bootstraps only a pure time-limit truncation.  A row that also
        # hits a physical/safety terminal must not receive gamma*V merely
        # because the horizon ended on the same control transition.
        truncated = timeout & ~terminated
        table_terminal = keepout | resolved_table
        bits = (
            truncated.to(torch.long) * FULLMDP_TERMINATION_BITS["time_out"]
            + tilt.to(torch.long) * FULLMDP_TERMINATION_BITS["base_fell_tilt"]
            + low.to(torch.long) * FULLMDP_TERMINATION_BITS["base_too_low"]
            + qdes_safety.to(torch.long)
            * FULLMDP_TERMINATION_BITS["joint_qdes_forbidden"]
            + table_terminal.to(torch.long)
            * FULLMDP_TERMINATION_BITS["robot_hit_table"]
        )
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
            self._full_a_motion_phase_code[ids] = RECOVER_HIDDEN_PHASE_INDEX
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
                self._teacher_racket_site_pos_w[ids] = (
                    self._ready_teacher_racket_site_pos_w[ids]
                )
                self._teacher_racket_site_lin_vel_w[ids] = 0.0
                self._teacher_racket_signed_normal_w[ids] = (
                    self._ready_teacher_racket_signed_normal_w[ids]
                )
                self._teacher_racket_long_axis_w[ids] = (
                    self._ready_teacher_racket_long_axis_w[ids]
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
            self._full_a_recovery_ready_streak[ids] = 0
            self._full_a_recovery_ready_seen[ids] = False
            self._full_a_recovery_expected_count[ids] = 0
            self._full_a_recovery_eligible_count[ids] = 0
            self._full_a_recovery_last_age[ids] = -1
            self._full_a_recovery_sticky_fault[ids] = False
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
            if self.full_a_mode:
                self._full_a_fact_integrity_fault_bits.zero_()
            self._clear_lifecycle(self._all_env_ids)
            if self.full_a_mode:
                self._full_a_reset_cadence_rows(self._all_env_ids)
                self.sim.forward()
                self._full_a_prime_cadence_readiness(self._all_env_ids)
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
        _st_before_forward, _tau_sq, requested_qdes = self._advance_plant(actions)
        # MuJoCo-Warp's step integrates qpos/qvel after its derived-tensor
        # forward pass.  Re-forward once so termination, Reward28, and the
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
        reward, reward_terms = self._fullmdp_reward()
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
        if getattr(self, "_diagnostic_contact_patch_consumer", None) is not None:
            self._diagnostic_contact_patch_transition_start_step = int(
                self.common_step_counter
            )
            self._begin_diagnostic_table_attribution_tick()
        self._full_a_begin_control_step()
        scheduled_due_event, launch_event, missed_launch_event = (
            self._full_a_prepare_step()
        )
        _st_before_forward, _tau_sq, requested_qdes = self._advance_plant(actions)
        self.sim.forward()
        final_contact_census = A3ReadyBallVecEnv._contact_census(self)
        final_resolved_count = self._latch_post_forward_resolved_table_contacts(
            final_contact_census
        )
        final_keepout = self._latch_post_forward_table_keepout()
        table_consumer = getattr(
            self, "_diagnostic_table_attribution_consumer", None
        )
        if table_consumer is not None:
            table_consumer(
                keepout=final_keepout,
                resolved=final_resolved_count.gt(0),
                resolved_is_final=True,
                substep_index=None,
                capture_boundary="post_forward_final",
            )
        self._full_a_latch_ball_contacts(
            final_contact_census,
            diagnostic_substep_index=None,
            diagnostic_capture_boundary="post_forward_final",
        )
        if self._cap_ok:
            self._probe_capacity("forward")
        st = self._state()
        tracked_body_kinematics = self._fullmdp_tracked_body_kinematics()
        # The cached teacher is the one published in the observation that
        # produced ``actions``.  Keep it fixed through physics and reward;
        # advance it only at the post-transition observation boundary below.
        terminated, truncated, terminal_bits, resolved_table_contact = (
            self._fullmdp_termination(st, requested_qdes)
        )
        recovery_errors = self._full_a_recovery_component_errors(
            tracked_body_kinematics
        )
        recovery_hard_safety_ok = (
            self._full_a_recovery_joint_limit_ok() & ~terminated & ~truncated
        )
        self._full_a_update_cadence_readiness(
            recovery_errors,
            recovery_hard_safety_ok,
            terminated,
            truncated,
        )
        self._full_a_publish_physical_fact()
        racket_kinematics = self._full_a_racket_kinematics()
        r03_present_event, r03_valid_event = self._full_a_publish_r03_fact(
            racket_kinematics
        )
        outcome_event, outcome = self._full_a_settle_outcome(st)
        r06_present, r06_eligible, r06_common = self._full_a_publish_r06_fact(
            outcome_event, outcome
        )
        r07_present, r07_eligible = self._full_a_publish_r07_fact(
            recovery_errors, recovery_hard_safety_ok
        )
        paddle_prior_playback = (
            FullMdpInitialWaitVecEnv
            ._full_a_paddle_prior_playback_mask(self)
            .clone()
        )
        reward, reward_terms, paddle_prior_error = self._fullmdp_reward(
            racket_kinematics,
            tracked_body_kinematics,
            return_paddle_error=True,
        )
        (
            recovery_terminal,
            recovery_success,
            recovery_failure,
            recovery_timeout,
            recovery_completion_fault,
        ) = self._full_a_finish_recovery(terminated, truncated)
        contact_event = self._full_a_generic_contact_event.clone()
        selected_contact_event = self._full_a_selected_contact_event.clone()
        opposite_contact_event = self._full_a_opposite_contact_event.clone()
        edge_contact_event = self._full_a_edge_contact_event.clone()
        between_contact_event = self._full_a_between_contact_event.clone()
        invalid_contact_event = self._full_a_invalid_contact_event.clone()
        actual_hard_edge_event = self._actual_hard_edge_latch.clone()
        qdes_guard_intervention_event = self._qdes_guard_intervention.clone()
        contact_classification_status = (
            self._full_a_contact_classification_status.clone()
        )
        landing_on_opponent = self._full_a_landing_on_opponent.clone()
        landing_opponent_bound = self._full_a_landing_opponent_bound.clone()
        landing_crossing_event = (
            outcome_event & self._full_a_landing_crossing_present
        ).clone()
        dones = terminated | truncated
        shot_retired = recovery_terminal & ~dones
        completed_action_epoch = self._full_a_completed_action_epoch(shot_retired)
        # Only a true Gym reset advances reset_generation.  A retired shot is
        # retained until the next frozen due tick produces a real ACCEPT.
        selected_reset = dones.clone()
        outcome_state = self._full_a_outcome_code.clone()
        racket_contact = self._full_a_racket_contact.clone()
        table_contact = self._full_a_ball_table_contact.clone()
        physical_center = self._full_a_physical_fact_f32[:, :3].clone()
        (
            reveal_event,
            reveal_due_event,
            reveal_deferred_event,
            due_terminal_overlap_event,
        ) = self._full_a_settle_reveal(
            scheduled_due_event, dones
        )
        terminal_phase = self._epoch_phase.clone()
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.last_terminal_bits[reset_ids] = terminal_bits[reset_ids]
            self._reset_idx(reset_ids)
            self.reset_generation[reset_ids] += 1
            self._clear_lifecycle(reset_ids)
            self._full_a_reset_cadence_rows(reset_ids)
        if reset_ids.numel() > 0:
            self.sim.forward()
            if self._cap_ok:
                self._probe_capacity("forward")
            self._full_a_prime_cadence_readiness(reset_ids)
        self._full_a_update_teacher()
        self._compute_obs()
        if self._cap_ok:
            self._capacity_gate(f"FullMDP A step {self.common_step_counter}")
        extras = {
            "time_outs": truncated,
            "termination_bits": terminal_bits,
            "backend_resolved_table_contact": resolved_table_contact,
            "reward_terms": reward_terms,
            "full_a_paddle_prior_playback": paddle_prior_playback,
            "full_a_paddle_prior_error": paddle_prior_error,
            "reset_generation": self.reset_generation.clone(),
            "full_a_phase_before_reset": terminal_phase,
            "full_a_outcome_code": outcome_state,
            "full_a_fact_integrity_fault_bits": (
                self._full_a_fact_integrity_fault_bits.clone()
            ),
            "full_a_racket_contact": racket_contact,
            "full_a_ball_table_contact": table_contact,
            "full_a_physical_current_center": physical_center,
            "full_a_scheduled_due_event": scheduled_due_event,
            "full_a_due_terminal_overlap_event": due_terminal_overlap_event,
            "full_a_reveal_event": reveal_event,
            "full_a_reveal_due_event": reveal_due_event,
            "full_a_reveal_deferred_event": reveal_deferred_event,
            "full_a_launch_event": launch_event,
            "full_a_missed_launch_event": missed_launch_event,
            "full_a_flight_terminal_event": outcome_event,
            "full_a_shot_retired_event": shot_retired,
            "full_a_completed_action_epoch_event": completed_action_epoch,
            "full_a_selected_reset_event": selected_reset,
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
            "full_a_actual_hard_edge_event": actual_hard_edge_event,
            "full_a_qdes_guard_intervention_event": (
                qdes_guard_intervention_event
            ),
            "full_a_r03_present_event": r03_present_event,
            "full_a_r03_physically_valid_event": r03_valid_event,
            "full_a_landing_crossing_event": landing_crossing_event,
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
            "full_a_recovery_completion_fault_event": (
                recovery_completion_fault
            ),
        }
        return self.get_observations(), reward, dones.long(), extras


__all__ = [
    "WAIT_BALL_PARK_HOPE",
    "RECOVER_HIDDEN_PHASE_INDEX",
    "READY_HOLD_PHASE_INDEX",
    "FULL_A_PHASE_IDLE",
    "FULL_A_PHASE_REVEAL_COMMITTED",
    "FULL_A_PHASE_LAUNCH_SETTLED",
    "FULL_A_PHASE_OUTCOME_SETTLED",
    "FULL_A_PHASE_RETIRED",
    "FULL_A_OUTCOME_NONE",
    "FULL_A_OUTCOME_FLIGHT_EXPIRED",
    "FULL_A_OUTCOME_BALL_DEAD",
    "FULL_A_FACT_INTEGRITY_R03_NONFINITE",
    "FULL_A_FACT_INTEGRITY_R06_SOURCE_INVALID",
    "FULL_A_FACT_INTEGRITY_R07_SEQUENCE",
    "FULL_A_FACT_INTEGRITY_R07_NONFINITE",
    "FULL_A_SUPPORT_FORCE_N",
    "FULLMDP_TRACKED_BODY_NAMES",
    "FULLMDP_DENSE_REWARD_SPECS",
    "FULLMDP_TERMINATION_BITS",
    "FullMdpInitialWaitVecEnv",
]
