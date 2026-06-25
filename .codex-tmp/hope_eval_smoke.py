from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device='cuda:0').app

import gymnasium as gym
import pickle
import torch
import yaml

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import whole_body_tracking.tasks

run_dir = '/home/dongc1/workspace/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/agibot_a3_hope/2026-06-25_08-10-54_pathA_basecouple'
with open(f'{run_dir}/params/env.pkl', 'rb') as f:
    env_cfg = pickle.load(f)
with open(f'{run_dir}/params/agent.yaml', 'r') as f:
    agent_cfg = yaml.safe_load(f)

env_cfg.scene.num_envs = 64
env_cfg.sim.device = 'cuda:0'

env = gym.make('HOPE-PingPong-AgibotA3-v0', cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg['device'])
runner.load(f'{run_dir}/model_4400.pt')
policy = runner.get_inference_policy(device=env.unwrapped.device)
obs = env.get_observations().to(agent_cfg['device'])
for i in range(30):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
    if i in (0, 1, 2, 10, 20, 29):
        cmd = env.unwrapped.command_manager.get_term('racket_target')
        mask = torch.abs(cmd.time_to_strike) <= (0.5 * env.unwrapped.step_dt + 1e-6)
        print('step', i, 'exact', int(mask.sum().item()), 'mean_exact_pos_metric', float(cmd.metrics['racket_pos_error_exact_strike'].mean().item()), 'action_abs_max', float(cmd.metrics['action_abs_max'].mean().item()))
term = env.unwrapped.action_manager.get_term('joint_pos')
print('action_term', type(term), len(term._joint_names), term._joint_names[:8])
env.close()
app.close()
