"""Headless probe: verify the corrected exact-strike success metric vs the RAW conditional pass rate.

Loads a trained checkpoint + local motion clip(s) (no wandb), steps the env, and each step pairs the
per-env exact-strike mask with the instantaneous racket errors to recompute the RAW conditional
exact-strike pass rates. It then compares those to the LOGGED metric tensors:

    hope_isaac_py scripts/probe_metric.py task=HOPEPingPong algo=ppo headless=true num_envs=512 \
        checkpoint=logs/rsl_rl/agibot_a3_hope/<run>/model_4400.pt \
        motion_file=../motions/preprocessed/hope_forehand.npz \
        motion_file_2=../motions/preprocessed/hope_backhand.npz steps=800

Confirms: (1) logged strike_composite_success_exact ~= raw exact composite, (2) pos/vel/normal logged
separately, (3) exact-strike timing stays aligned for unified multi-clip policies.
"""
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides, _registry_clip_name, resolve_motion_sources


def _run(cfg, simulation_app):
    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking.tasks as _wbt_tasks  # registers the gym tasks
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    _ = _wbt_tasks

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task, _registry_clip_name(cfg))
    env_cfg.sim.device = str(cfg.device)
    motion_files, _ = resolve_motion_sources(cfg, cwd=pathlib.Path.cwd())
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    # HER achieved-target replay is TRAIN-ONLY (the probe must measure the pure box distribution).
    if hasattr(env_cfg.commands, "racket_target") and hasattr(
        env_cfg.commands.racket_target, "achieved_target_mix_prob"
    ):
        env_cfg.commands.racket_target.achieved_target_mix_prob = 0.0

    # R14 retiming is TRAIN-ONLY (the probe must measure the native-speed reference clock).
    if hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "speed_scale_range"):
        env_cfg.commands.motion.speed_scale_range = (1.0, 1.0)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(cfg.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd = env.unwrapped.command_manager.get_term("racket_target")
    pos_thr = float(cmd.cfg.strike_success_pos_thresh)
    vel_thr = float(cmd.cfg.strike_success_vel_thresh)
    nrm_thr = float(cmd.cfg.strike_success_normal_thresh_deg)
    step_dt = float(env.unwrapped.step_dt)

    # Accumulate exact-strike samples for the logged mask and the live command timing. The latter uses
    # the command term's per-clip time_to_strike, so unified forehand/backhand probes do not assume one
    # scalar strike phase for all clips.
    acc = {k: dict(n=0, pos=0, vel=0, nrm=0, comp=0) for k in ("logged", "live")}
    n_steps = int(cfg.get("steps", 800))

    def tally(which, mask, m):
        k = int(mask.sum())
        if not k:
            return
        pp = m["racket_pos_error"][mask] < pos_thr
        vp = m["racket_vel_error"][mask] < vel_thr
        npass = m["racket_normal_error_deg"][mask] < nrm_thr
        a = acc[which]
        a["n"] += k
        a["pos"] += int(pp.sum()); a["vel"] += int(vp.sum()); a["nrm"] += int(npass.sum())
        a["comp"] += int((pp & vp & npass).sum())

    obs = env.get_observations().to(agent_cfg.device)
    step = 0
    while simulation_app.is_running() and step < n_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
        m = cmd.metrics
        tally("logged", m["exact_strike_hit_rate"] > 0.5, m)
        fresh_tts = cmd.time_to_strike
        tally("live", fresh_tts.abs() <= (0.5 * step_dt + 1e-6), m)
        step += 1

    def rate(a, key):
        return acc[a][key] / acc[a]["n"] if acc[a]["n"] else float("nan")

    logged_comp = float(cmd.metrics["strike_composite_success_exact"][0])
    logged_pos = float(cmd.metrics["strike_pos_pass_exact"][0])
    logged_vel = float(cmd.metrics["strike_vel_pass_exact"][0])
    logged_nrm = float(cmd.metrics["strike_normal_pass_exact"][0])
    logged_decN = float(cmd.metrics["exact_strike_sample_count_decayed"][0])

    print("\n" + "=" * 72, flush=True)
    print(f"PROBE over {step} steps, {num_envs} envs  |  exact samples logged={acc['logged']['n']} "
          f"live={acc['live']['n']}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'metric':<22}{'RAW logged':>12}{'RAW live':>12}{'LOGGED':>12}", flush=True)
    print(f"{'pos < %.3g m' % pos_thr:<22}{rate('logged','pos'):>12.4f}{rate('live','pos'):>12.4f}{logged_pos:>12.4f}", flush=True)
    print(f"{'vel < %.3g m/s' % vel_thr:<22}{rate('logged','vel'):>12.4f}{rate('live','vel'):>12.4f}{logged_vel:>12.4f}", flush=True)
    print(f"{'normal < %.3g deg' % nrm_thr:<22}{rate('logged','nrm'):>12.4f}{rate('live','nrm'):>12.4f}{logged_nrm:>12.4f}", flush=True)
    print(f"{'composite (all 3)':<22}{rate('logged','comp'):>12.4f}{rate('live','comp'):>12.4f}{logged_comp:>12.4f}", flush=True)
    print("-" * 72, flush=True)
    print(f"decayed exact-strike sample count (logged): {logged_decN:.1f}", flush=True)
    print("=" * 72 + "\n", flush=True)

    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        _run(cfg, simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
