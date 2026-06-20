"""Hydra eval/export entry for HOPE Agibot A3 WBC (106B-Final-Project style).

    python scripts/play.py task=HOPEPingPong algo=ppo num_envs=2 \
        wandb_path=<entity>/hope_wbc/<run_id>

Loads a trained policy (from a wandb run, or the latest local checkpoint), runs it, and exports
policy.onnx next to the checkpoint. Mirrors the legacy scripts/rsl_rl/play.py mechanics; reuses the
task-YAML override mapping from scripts/train.py.
"""

import os
import sys

# allow `from train import _apply_task_overrides` (sibling script; no isaaclab imported at its top)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import OmegaConf

from train import _apply_task_overrides


def _run_play(cfg, simulation_app):
    import pathlib

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.sim.device = str(cfg.device)

    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name)))
    agent_cfg.device = str(cfg.device)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # resolve the checkpoint + reference motion
    wandb_path = cfg.wandb_path
    if wandb_path:
        import wandb

        wandb_path = str(wandb_path)
        run_path = "/".join(wandb_path.split("/")[:-1]) if "model" in wandb_path else wandb_path
        api = wandb.Api()
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        fname = wandb_path.split("/")[-1] if "model" in wandb_path else max(
            files, key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        wandb_run.file(str(fname)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{fname}"
        print(f"[INFO] Loading model checkpoint from: {run_path}/{fname}")
        if cfg.motion_file is not None:
            env_cfg.commands.motion.motion_file = str(cfg.motion_file)
        else:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is not None:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
            else:
                print("[WARN] No motion artifact in the run; pass motion_file=... if replay fails.")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        reg = cfg.registry_name if cfg.registry_name is not None else cfg.task.get("registry_name")
        if cfg.motion_file is not None:
            env_cfg.commands.motion.motion_file = str(cfg.motion_file)
        elif reg is not None:
            import wandb

            reg = str(reg)
            if ":" not in reg:
                reg += ":latest"
            art = wandb.Api().artifact(reg)
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    log_dir = os.path.dirname(resume_path)
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export the policy to ONNX next to the checkpoint (step 15)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped, ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir, filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, str(wandb_path) if wandb_path else "none", export_model_dir)
    print(f"[INFO] Exported ONNX policy to: {export_model_dir}")

    obs, _ = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if cfg.video:
            timestep += 1
            if timestep == int(cfg.video_length):
                break
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    try:
        _run_play(cfg, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
