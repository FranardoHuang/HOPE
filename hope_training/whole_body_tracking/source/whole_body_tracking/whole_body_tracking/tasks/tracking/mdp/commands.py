from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Loads one or more motion clips into a single concatenated time axis.

    Passing several files (HITTER unified policy: forehand + backhand) concatenates them along the time
    dimension and records per-clip ``seg_start`` / ``seg_len`` so the command can step/wrap/strike within
    one clip ("segment") at a time, selected per-env by swing type. A single file behaves exactly as
    before: one segment spanning the whole motion, ``time_step_total`` unchanged.
    """

    def __init__(self, motion_file, body_indexes: Sequence[int], device: str = "cpu"):
        files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        assert len(files) >= 1, "MotionLoader needs at least one motion file"
        jp, jv, bp, bq, bl, ba = [], [], [], [], [], []
        seg_lens = []
        self.fps = None
        for f in files:
            assert os.path.isfile(f), f"Invalid file path: {f}"
            data = np.load(f)
            if self.fps is None:
                self.fps = data["fps"]
            jp.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
            jv.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
            bp.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            bq.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            bl.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
            ba.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
            seg_lens.append(jp[-1].shape[0])
        self.joint_pos = torch.cat(jp, dim=0)
        self.joint_vel = torch.cat(jv, dim=0)
        self._body_pos_w = torch.cat(bp, dim=0)
        self._body_quat_w = torch.cat(bq, dim=0)
        self._body_lin_vel_w = torch.cat(bl, dim=0)
        self._body_ang_vel_w = torch.cat(ba, dim=0)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        # Per-clip segment boundaries on the concatenated time axis.
        self.num_segments = len(seg_lens)
        self.seg_len = torch.tensor(seg_lens, dtype=torch.long, device=device)
        self.seg_start = torch.zeros(self.num_segments, dtype=torch.long, device=device)
        if self.num_segments > 1:
            self.seg_start[1:] = torch.cumsum(self.seg_len, dim=0)[:-1]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Unified multi-clip (HITTER forehand+backhand) support. With one clip these are inert and the
        # behaviour below is byte-identical to the single-clip path. clip_id[env] selects which segment
        # (swing type) the env is currently imitating.
        self._multiseg = self.motion.num_segments > 1
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Robust per-step "this env just wrapped to a new swing" signal, consumed by the racket-target
        # command to resample its target. Replaces a time_steps<prev heuristic that fails when a clip
        # wrap jumps the index to a HIGHER segment start (forehand->backhand on the concatenated axis).
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Pre-swing hold state (see cfg.hold_steps_range): while hold_counter > 0 the reference
        # clock is frozen at the swing's first frame ("waiting for the ball"). _update_command
        # decrements it. in_hold is exposed for rewards/metrics.
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # True only while _resample_command is being invoked from an intra-episode clip WRAP
        # (as opposed to a true episode reset) — wraps skip the RSI teleport (cfg.wrap_teleport).
        self._resampling_from_wrap = False
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["in_hold"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"reference_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def in_hold(self) -> torch.Tensor:
        """Bool mask: env is in the pre-swing hold (reference frozen at the swing's first frame)."""
        return self.hold_counter > 0

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        anchor_pos_err = self.anchor_pos_w - self.robot_anchor_pos_w
        anchor_rot_err = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        anchor_lin_vel_err = self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w
        anchor_ang_vel_err = self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w

        self.metrics["error_anchor_pos"] = torch.norm(anchor_pos_err, dim=-1)
        self.metrics["error_anchor_rot"] = anchor_rot_err
        self.metrics["error_anchor_lin_vel"] = torch.norm(anchor_lin_vel_err, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(anchor_ang_vel_err, dim=-1)
        self.metrics["error_anchor_rot_deg"] = anchor_rot_err * (180.0 / math.pi)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        joint_pos_err = self.joint_pos - self.robot_joint_pos
        joint_vel_err = self.joint_vel - self.robot_joint_vel
        self.metrics["error_joint_pos"] = torch.norm(joint_pos_err, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(joint_vel_err, dim=-1)
        self.metrics["error_joint_pos_mean_abs"] = torch.mean(torch.abs(joint_pos_err), dim=-1)
        self.metrics["error_joint_pos_max_abs"] = torch.max(torch.abs(joint_pos_err), dim=-1).values
        self.metrics["error_joint_vel_mean_abs"] = torch.mean(torch.abs(joint_vel_err), dim=-1)
        self.metrics["error_joint_vel_max_abs"] = torch.max(torch.abs(joint_vel_err), dim=-1).values

        # Log anchor states in an env-origin-relative frame so cross-env averages remain meaningful.
        anchor_ref_rel = self.anchor_pos_w - self._env.scene.env_origins
        anchor_robot_rel = self.robot_anchor_pos_w - self._env.scene.env_origins
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"reference_anchor_pos_{axis}"] = anchor_ref_rel[:, axis_idx]
            self.metrics[f"robot_anchor_pos_{axis}"] = anchor_robot_rel[:, axis_idx]
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = self.anchor_lin_vel_w[:, axis_idx]
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = self.robot_anchor_lin_vel_w[:, axis_idx]

        self.metrics["reference_anchor_speed"] = torch.norm(self.anchor_lin_vel_w, dim=-1)
        self.metrics["robot_anchor_speed"] = torch.norm(self.robot_anchor_lin_vel_w, dim=-1)
        if self._multiseg:
            seg_start = self.motion.seg_start[self.clip_id]
            seg_len = self.motion.seg_len[self.clip_id].clamp(min=2)
            self.metrics["motion_phase"] = (self.time_steps - seg_start).float() / (seg_len - 1).float()
        else:
            self.metrics["motion_phase"] = self.time_steps.float() / max(self.motion.time_step_total - 1, 1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self._multiseg:
            # HITTER unified policy: each new swing uniformly samples the swing TYPE (clip) and starts at
            # that clip's first frame (reference-state-init at the swing start). The adaptive failure-bin
            # curriculum is single-clip BeyondMimic machinery and is bypassed here.
            n = len(env_ids)
            if n > 0:
                new_clip = torch.randint(0, self.motion.num_segments, (n,), device=self.device)
                self.clip_id[env_ids] = new_clip
                self.time_steps[env_ids] = self.motion.seg_start[new_clip]
            # Report the REAL clip-sampling distribution (repurpose the bin-sampling metrics for clips):
            # entropy of the per-clip env fraction (1.0 = balanced), and the most-sampled clip + its share.
            counts = torch.bincount(self.clip_id, minlength=self.motion.num_segments).float()
            probs = counts / counts.sum().clamp(min=1.0)
            H = -(probs * (probs + 1e-12).log()).sum()
            self.metrics["sampling_entropy"][:] = H / math.log(max(self.motion.num_segments, 2))
            pmax, imax = probs.max(dim=0)
            self.metrics["sampling_top1_prob"][:] = pmax
            self.metrics["sampling_top1_bin"][:] = imax.float() / max(self.motion.num_segments, 1)
            return
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)

        # Pre-swing HOLD (Phase A): freeze the reference at the swing's first frame for a random
        # number of control steps ("the ball is not reaching yet"). Applies to resets AND wraps.
        lo, hi = self.cfg.hold_steps_range
        self.hold_counter[env_ids_t] = torch.randint(int(lo), int(hi) + 1, (len(env_ids_t),), device=self.device)

        # Intra-episode clip WRAP: no teleport (deploy case) — the policy must physically carry
        # the body from the previous swing's end into the new swing's windup. The imitation
        # targets are anchor-relative, so the new reference re-anchors to the robot where it is.
        # Teleporting at a wrap (legacy RSI) requires BOTH wrap_teleport=True AND rsi_on_wrap=True
        # (rsi_on_wrap=False is the HOPE task-YAML knob that disables mid-episode RSI teleports).
        if self._resampling_from_wrap and not (self.cfg.wrap_teleport and self.cfg.rsi_on_wrap):
            return

        # TRUE episode reset: a fraction starts from the DEFAULT STAND (deploy entry), the rest
        # keeps the legacy RSI teleport onto the (noised) reference frame.
        stand_mask = torch.zeros(len(env_ids_t), dtype=torch.bool, device=self.device)
        if not self._resampling_from_wrap and self.cfg.stand_start_prob > 0.0:
            stand_mask = torch.rand(len(env_ids_t), device=self.device) < self.cfg.stand_start_prob
        stand_ids = env_ids_t[stand_mask]
        rsi_ids = env_ids_t[~stand_mask]

        if len(stand_ids) > 0:
            default_root = self.robot.data.default_root_state[stand_ids].clone()
            default_root[:, :3] += self._env.scene.env_origins[stand_ids]
            default_root[:, 7:] = 0.0  # zero lin/ang velocity
            self.robot.write_root_state_to_sim(default_root, env_ids=stand_ids)
            self.robot.write_joint_state_to_sim(
                self.robot.data.default_joint_pos[stand_ids],
                torch.zeros_like(self.robot.data.default_joint_vel[stand_ids]),
                env_ids=stand_ids,
            )
            # Give the stand-started envs time to travel stand -> windup before the clip runs.
            self.hold_counter[stand_ids] = torch.clamp(
                self.hold_counter[stand_ids], min=int(self.cfg.stand_start_min_hold)
            )

        if len(rsi_ids) == 0:
            return
        env_ids = rsi_ids

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        # Pre-swing HOLD: held envs keep the reference frozen at the swing's first frame
        # ("waiting for the ball"); everyone else advances the clip clock.
        held = self.hold_counter > 0
        self.hold_counter = torch.clamp(self.hold_counter - 1, min=0)
        self.metrics["in_hold"] = held.float()
        self.time_steps += (~held).long()
        if self._multiseg:
            # Wrap at the END of the env's current clip/segment, not the global concatenated end.
            seg_end = self.motion.seg_start[self.clip_id] + self.motion.seg_len[self.clip_id]
            env_ids = torch.where(self.time_steps >= seg_end)[0]
        else:
            env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(env_ids) > 0:
            self.just_resampled[env_ids] = True
        # Wrap-path resample: skips the RSI teleport (cfg.wrap_teleport=False or cfg.rsi_on_wrap=False)
        # so the policy physically transitions swing -> swing. True resets go through reset()/manager.
        self._resampling_from_wrap = True
        try:
            self._resample_command(env_ids)
        finally:
            self._resampling_from_wrap = False

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Reference-state initialization (RSI) at clip wrap. HOPE ping-pong task YAMLs set this to
    # false so the robot must physically recover between swings instead of being teleported to
    # the next clip start mid-episode. NOTE: since the Phase A redesign below, wraps only teleport
    # when BOTH rsi_on_wrap AND wrap_teleport are true — with wrap_teleport's default of False,
    # wraps never teleport unless both knobs are explicitly enabled (legacy BeyondMimic behavior).
    rsi_on_wrap: bool = True

    # --- Phase A (2026-07-02): swing ENTRY / TRANSITION / WAITING coverage --------------------
    # Deploy enters every swing from a NOMINAL STAND, waits at the windup while the ball is not
    # yet reaching, and must physically transition between swings — none of which the pure-RSI
    # scheme ever produced (teleport at every episode start AND every clip wrap). These knobs
    # close that gap; the imitation targets are anchor-RELATIVE (re-anchored to the robot's
    # current xy+yaw every step), so no-teleport starts/wraps are well-posed.
    # Fraction of TRUE episode resets that start from the robot's DEFAULT STAND pose (zero
    # velocities) instead of teleporting onto the reference clip frame (RSI).
    stand_start_prob: float = 0.25
    # Teleport the robot onto the new clip's start frame at intra-episode wraps (legacy RSI
    # behavior). False = the policy must physically transition swing->swing (the deploy case).
    wrap_teleport: bool = False
    # Pre-swing HOLD: on every swing (re)start, freeze the reference at the clip's first frame
    # for U[lo,hi] control steps (50 Hz). While held, time_to_strike sits at its per-clip
    # maximum — exactly the deploy runner's clamped "waiting for the ball" pairing.
    hold_steps_range: tuple[int, int] = (0, 100)
    # Stand-started envs get at least this much hold (they must travel stand -> windup first).
    stand_start_min_hold: int = 25

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
