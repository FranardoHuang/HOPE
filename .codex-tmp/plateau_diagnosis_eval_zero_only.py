from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device='cuda:0').app

import gymnasium as gym
import torch
import yaml

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import whole_body_tracking.tasks  # noqa: F401

RUN_DIR = Path('/home/dongc1/workspace/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-06-25_08-10-54_pathA_basecouple')
OUT_DIR = Path('/home/dongc1/workspace/HOPE/.codex-tmp')
TASK_ID = 'HOPE-PingPong-AgibotA3-v0'
CHECKPOINT = str(RUN_DIR / 'model_4400.pt')
NUM_ENVS = 128
MIN_EXACT_SAMPLES = 256
MAX_STEPS = 900


def qstats(vals):
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p25': float(np.quantile(arr, 0.25)),
        'p75': float(np.quantile(arr, 0.75)),
        'p90': float(np.quantile(arr, 0.90)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_env_cfg():
    with open(RUN_DIR / 'params' / 'env.pkl', 'rb') as f:
        return pickle.load(f)


def load_agent_cfg():
    with open(RUN_DIR / 'params' / 'agent.yaml', 'r') as f:
        return yaml.safe_load(f)


def maybe_joint_limits(robot, joint_ids):
    limits = getattr(robot.data, 'soft_joint_pos_limits', None)
    if limits is None:
        limits = getattr(robot.data, 'joint_pos_limits', None)
    if limits is None:
        return None, None
    if limits.dim() == 3:
        limits = limits[0]
    if not isinstance(joint_ids, slice):
        limits = limits[joint_ids]
    return limits[:, 0], limits[:, 1]


def maybe_vel_limits(robot, joint_ids):
    for getter_name in ('get_dof_max_velocities', 'get_dof_velocity_limits'):
        getter = getattr(robot.root_physx_view, getter_name, None)
        if getter is not None:
            vals = torch.as_tensor(getter(), device=robot.device)
            if vals.dim() > 1:
                vals = vals[0]
            if not isinstance(joint_ids, slice):
                vals = vals[joint_ids]
            return vals
    return None


def maybe_effort_limits(robot, joint_ids):
    for getter_name in ('get_dof_max_forces', 'get_dof_effort_limits'):
        getter = getattr(robot.root_physx_view, getter_name, None)
        if getter is not None:
            vals = torch.as_tensor(getter(), device=robot.device)
            if vals.dim() > 1:
                vals = vals[0]
            if not isinstance(joint_ids, slice):
                vals = vals[joint_ids]
            return vals
    return None


def build_right_arm_indices(joint_names):
    return [i for i, n in enumerate(joint_names) if n.startswith('right_') and any(k in n for k in ('shoulder', 'elbow', 'wrist'))]


def evaluate_variant(name: str, zero_perturb: bool):
    print(f'[eval] start {name}', flush=True)
    env_cfg = load_env_cfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.sim.device = 'cuda:0'
    if zero_perturb:
        env_cfg.commands.racket_target.ref_perturb_pos = (0.0, 0.0, 0.0)
        env_cfg.commands.racket_target.ref_perturb_vel = (0.0, 0.0, 0.0)
        env_cfg.commands.racket_target.ref_perturb_normal = 0.0
    agent_cfg = load_agent_cfg()
    agent_cfg['device'] = 'cuda:0'

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg['device'])
    runner.load(CHECKPOINT)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene['robot']
    action_term = uenv.action_manager.get_term('joint_pos')
    joint_ids = action_term._joint_ids
    joint_names = list(action_term._joint_names)
    right_arm_idx = build_right_arm_indices(joint_names)
    joint_lower, joint_upper = maybe_joint_limits(robot, joint_ids)
    vel_limits = maybe_vel_limits(robot, joint_ids)
    effort_limits = maybe_effort_limits(robot, joint_ids)
    step_dt = float(uenv.step_dt)
    racket_pos_std = float(env_cfg.rewards.racket_position.params['std'])
    racket_pos_weight = float(env_cfg.rewards.racket_position.weight)
    success_pos_thresh = float(env_cfg.commands.racket_target.strike_success_pos_thresh)

    obs = env.get_observations().to(agent_cfg['device'])
    total_env_steps = 0
    moving_reward_global_sum = 0.0
    static_reward_global_sum = 0.0
    in_window_count = 0
    moving_reward_window_sum = 0.0
    static_reward_window_sum = 0.0
    completed_ep_lengths = []

    samples = {k: [] for k in [
        'time_to_strike','static_pos_err','moving_pos_err','vel_err','normal_err_deg','base_xy_err',
        'abs_x','abs_y','abs_z','signed_x','signed_y','signed_z','perturb_pos_norm','perturb_pos_y',
        'perturb_vel_norm','target_normal_perturb_deg','moving_reward_value','static_reward_value',
        'right_arm_raw_abs','right_arm_raw_gt1','right_arm_target_actual_abs','right_arm_target_outside_limits',
        'right_arm_actual_near_limits','right_arm_vel_sat','right_arm_tau_sat'
    ]}

    try:
        for step in range(MAX_STEPS):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions.to(uenv.device))
            if step % 150 == 0:
                print(f'[eval] {name} step {step}', flush=True)

            total_env_steps += NUM_ENVS
            cmd = uenv.command_manager.get_term('racket_target')
            in_win = cmd.strike_window
            moving_target = cmd.racket_target_pos_w - cmd.racket_target_vel_w * cmd.time_to_strike.unsqueeze(-1)
            static_err_vec = cmd.racket_pos_w - cmd.racket_target_pos_w
            moving_err_vec = cmd.racket_pos_w - moving_target
            static_err = torch.norm(static_err_vec, dim=-1)
            moving_err = torch.norm(moving_err_vec, dim=-1)
            moving_rew = torch.exp(-(moving_err ** 2) / (racket_pos_std ** 2)) * in_win.float()
            static_rew = torch.exp(-(static_err ** 2) / (racket_pos_std ** 2)) * in_win.float()
            moving_reward_global_sum += float(moving_rew.sum().item())
            static_reward_global_sum += float(static_rew.sum().item())
            if in_win.any():
                in_window_count += int(in_win.sum().item())
                moving_reward_window_sum += float(moving_rew[in_win].sum().item())
                static_reward_window_sum += float(static_rew[in_win].sum().item())
            if dones.any():
                completed_ep_lengths.extend(float(x) for x in uenv.episode_length_buf[dones].detach().cpu().numpy().tolist())

            exact = torch.abs(cmd.time_to_strike) <= (0.5 * uenv.step_dt + 1e-6)
            if exact.any():
                idx = torch.where(exact)[0]
                vel_err = torch.norm(cmd.racket_lin_vel_w - cmd.racket_target_vel_w, dim=-1)
                cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
                normal_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)
                base_xy_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.base_target_pos_w, dim=-1)
                origins = uenv.scene.env_origins
                ref_static = origins + cmd._ref_racket_pos_rel.unsqueeze(0)
                perturb_pos = cmd.racket_target_pos_w - ref_static
                perturb_vel = cmd.racket_target_vel_w - cmd._ref_racket_vel_w.unsqueeze(0)
                ref_n = cmd._ref_racket_normal_w.unsqueeze(0)
                target_n_cos = torch.sum(cmd.racket_target_normal_w * ref_n, dim=-1).clamp(-1.0, 1.0)
                target_n_deg = torch.acos(target_n_cos) * (180.0 / math.pi)

                samples['time_to_strike'].extend(cmd.time_to_strike[idx].detach().cpu().numpy().tolist())
                samples['static_pos_err'].extend(static_err[idx].detach().cpu().numpy().tolist())
                samples['moving_pos_err'].extend(moving_err[idx].detach().cpu().numpy().tolist())
                samples['vel_err'].extend(vel_err[idx].detach().cpu().numpy().tolist())
                samples['normal_err_deg'].extend(normal_err_deg[idx].detach().cpu().numpy().tolist())
                samples['base_xy_err'].extend(base_xy_err[idx].detach().cpu().numpy().tolist())
                samples['abs_x'].extend(static_err_vec[idx, 0].abs().detach().cpu().numpy().tolist())
                samples['abs_y'].extend(static_err_vec[idx, 1].abs().detach().cpu().numpy().tolist())
                samples['abs_z'].extend(static_err_vec[idx, 2].abs().detach().cpu().numpy().tolist())
                samples['signed_x'].extend(static_err_vec[idx, 0].detach().cpu().numpy().tolist())
                samples['signed_y'].extend(static_err_vec[idx, 1].detach().cpu().numpy().tolist())
                samples['signed_z'].extend(static_err_vec[idx, 2].detach().cpu().numpy().tolist())
                samples['perturb_pos_norm'].extend(torch.norm(perturb_pos[idx], dim=-1).detach().cpu().numpy().tolist())
                samples['perturb_pos_y'].extend(perturb_pos[idx, 1].detach().cpu().numpy().tolist())
                samples['perturb_vel_norm'].extend(torch.norm(perturb_vel[idx], dim=-1).detach().cpu().numpy().tolist())
                samples['target_normal_perturb_deg'].extend(target_n_deg[idx].detach().cpu().numpy().tolist())
                samples['moving_reward_value'].extend(moving_rew[idx].detach().cpu().numpy().tolist())
                samples['static_reward_value'].extend(static_rew[idx].detach().cpu().numpy().tolist())

                raw = action_term.raw_actions[idx]
                proc = action_term.processed_actions[idx]
                jpos = robot.data.joint_pos[idx] if isinstance(joint_ids, slice) else robot.data.joint_pos[idx][:, joint_ids]
                jvel = robot.data.joint_vel[idx] if isinstance(joint_ids, slice) else robot.data.joint_vel[idx][:, joint_ids]
                jtgt_gap = (proc - jpos).abs()
                if right_arm_idx:
                    ridx = torch.as_tensor(right_arm_idx, device=uenv.device)
                    samples['right_arm_raw_abs'].extend(raw[:, ridx].abs().detach().cpu().numpy().tolist())
                    samples['right_arm_raw_gt1'].extend((raw[:, ridx].abs() > 1.0).float().detach().cpu().numpy().tolist())
                    samples['right_arm_target_actual_abs'].extend(jtgt_gap[:, ridx].detach().cpu().numpy().tolist())
                    if joint_lower is not None and joint_upper is not None:
                        lower = joint_lower[ridx]
                        upper = joint_upper[ridx]
                        outside = ((proc[:, ridx] < lower) | (proc[:, ridx] > upper)).float()
                        half_span = ((upper - lower) * 0.5).clamp(min=1e-6)
                        dist = torch.minimum(jpos[:, ridx] - lower, upper - jpos[:, ridx]).clamp(min=0.0)
                        near = ((dist / half_span) < 0.1).float()
                        samples['right_arm_target_outside_limits'].extend(outside.detach().cpu().numpy().tolist())
                        samples['right_arm_actual_near_limits'].extend(near.detach().cpu().numpy().tolist())
                    if vel_limits is not None:
                        vlim = vel_limits[ridx].clamp(min=1e-6)
                        vsat = ((jvel[:, ridx].abs() / vlim) > 0.9).float()
                        samples['right_arm_vel_sat'].extend(vsat.detach().cpu().numpy().tolist())
                    tau = getattr(robot.data, 'applied_torque', None)
                    if tau is not None and effort_limits is not None:
                        tau = tau[idx] if isinstance(joint_ids, slice) else tau[idx][:, joint_ids]
                        elim = effort_limits[ridx].clamp(min=1e-6)
                        tsat = ((tau[:, ridx].abs() / elim) > 0.9).float()
                        samples['right_arm_tau_sat'].extend(tsat.detach().cpu().numpy().tolist())

            if len(samples['static_pos_err']) >= MIN_EXACT_SAMPLES:
                print(f'[eval] {name} reached {len(samples["static_pos_err"])} exact samples at step {step}', flush=True)
                break
    finally:
        env.close()

    def per_joint_mean(key):
        arr = np.asarray(samples[key], dtype=np.float64)
        if arr.size == 0:
            return None
        return {joint_names[right_arm_idx[i]]: float(arr[:, i].mean()) for i in range(arr.shape[1])}

    static_arr = np.asarray(samples['static_pos_err'], dtype=np.float64)
    vel_arr = np.asarray(samples['vel_err'], dtype=np.float64)
    normal_arr = np.asarray(samples['normal_err_deg'], dtype=np.float64)
    mean_ep_len = float(np.mean(completed_ep_lengths)) if completed_ep_lengths else None
    moving_step_mean = moving_reward_global_sum / total_env_steps
    static_step_mean = static_reward_global_sum / total_env_steps

    result = {
        'variant': name,
        'num_envs': NUM_ENVS,
        'steps_run': step + 1,
        'exact_samples': int(len(samples['static_pos_err'])),
        'mean_completed_episode_length_steps': mean_ep_len,
        'step_dt': step_dt,
        'success_thresholds': {'position_m': success_pos_thresh, 'velocity_mps': 0.5, 'normal_deg': 15.0, 'base': None},
        'pass_rates': {
            'position_7p5cm': float(np.mean(static_arr < success_pos_thresh)),
            'position_10cm': float(np.mean(static_arr < 0.10)),
            'position_12p5cm': float(np.mean(static_arr < 0.125)),
            'position_15cm': float(np.mean(static_arr < 0.15)),
            'velocity_0p5': float(np.mean(vel_arr < 0.5)),
            'normal_15deg': float(np.mean(normal_arr < 15.0)),
            'composite_exact': float(np.mean((static_arr < success_pos_thresh) & (vel_arr < 0.5) & (normal_arr < 15.0))),
            'base_threshold_applicable': None,
            'timing_mask_applicable': 1.0,
        },
        'errors': {
            'time_to_strike_exact_s': qstats(samples['time_to_strike']),
            'racket_pos_error_static_exact': qstats(samples['static_pos_err']),
            'racket_pos_error_moving_exact': qstats(samples['moving_pos_err']),
            'racket_pos_error_abs_x_exact': qstats(samples['abs_x']),
            'racket_pos_error_abs_y_exact': qstats(samples['abs_y']),
            'racket_pos_error_abs_z_exact': qstats(samples['abs_z']),
            'racket_pos_error_signed_x_exact': qstats(samples['signed_x']),
            'racket_pos_error_signed_y_exact': qstats(samples['signed_y']),
            'racket_pos_error_signed_z_exact': qstats(samples['signed_z']),
            'racket_vel_error_exact': qstats(samples['vel_err']),
            'racket_normal_error_deg_exact': qstats(samples['normal_err_deg']),
            'base_xy_error_exact': qstats(samples['base_xy_err']),
        },
        'reward_alignment': {
            'racket_position_weight': racket_pos_weight,
            'racket_position_std': racket_pos_std,
            'moving_reward_step_mean': moving_step_mean,
            'static_reward_step_mean': static_step_mean,
            'moving_reward_episode_contrib_est': None if mean_ep_len is None else moving_step_mean * step_dt * mean_ep_len * racket_pos_weight,
            'static_reward_episode_contrib_est': None if mean_ep_len is None else static_step_mean * step_dt * mean_ep_len * racket_pos_weight,
            'window_sample_moving_reward_mean': None if in_window_count == 0 else moving_reward_window_sum / in_window_count,
            'window_sample_static_reward_mean': None if in_window_count == 0 else static_reward_window_sum / in_window_count,
            'exact_moving_reward_mean': float(np.mean(samples['moving_reward_value'])) if samples['moving_reward_value'] else None,
            'exact_static_reward_mean': float(np.mean(samples['static_reward_value'])) if samples['static_reward_value'] else None,
        },
        'perturbation': {
            'pos_norm_stats': qstats(samples['perturb_pos_norm']),
            'vel_norm_stats': qstats(samples['perturb_vel_norm']),
            'normal_deg_stats': qstats(samples['target_normal_perturb_deg']),
            'corr_pos_perturb_norm_vs_static_error': corr(samples['perturb_pos_norm'], samples['static_pos_err']),
            'corr_abs_y_perturb_vs_abs_y_error': corr(np.abs(samples['perturb_pos_y']), samples['abs_y']),
        },
        'control': {
            'right_arm': {
                'joint_names': [joint_names[i] for i in right_arm_idx],
                'mean_abs_raw_action': per_joint_mean('right_arm_raw_abs'),
                'raw_action_abs_gt1_rate': per_joint_mean('right_arm_raw_gt1'),
                'mean_abs_target_minus_actual': per_joint_mean('right_arm_target_actual_abs'),
                'target_outside_pos_limits_rate': per_joint_mean('right_arm_target_outside_limits'),
                'actual_near_pos_limits_rate': per_joint_mean('right_arm_actual_near_limits'),
                'vel_limit_sat_rate_gt90pct': per_joint_mean('right_arm_vel_sat'),
                'torque_limit_sat_rate_gt90pct': per_joint_mean('right_arm_tau_sat'),
            }
        },
    }
    out = OUT_DIR / f'plateau_eval_{name}.json'
    out.write_text(json.dumps(result, indent=2))
    print(f'[eval] wrote {out}', flush=True)
    return result


try:
    evaluate_variant('zero_perturb', zero_perturb=True)
finally:
    app.close()
