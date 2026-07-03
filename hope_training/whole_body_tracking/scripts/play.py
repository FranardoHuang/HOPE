"""Hydra eval/export entry for HOPE Agibot A3 WBC (106B-Final-Project style).

    python scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
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

from train import (
    _apply_task_overrides,
    _is_noneish,
    _normalize_registry_name,
    _registry_clip_name,
    resolve_motion_sources,
)


def _run_play(cfg, simulation_app):
    import pathlib

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.tasks.tracking.actor_observation_contract import (
        validate_actor_observation_contract,
    )
    from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task, _registry_clip_name(cfg))
    env_cfg.sim.device = str(cfg.device)

    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name)))
    agent_cfg.device = str(cfg.device)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # resolve the checkpoint + reference motion
    wandb_path = cfg.wandb_path
    checkpoint = cfg.get("checkpoint", None)
    if wandb_path and not checkpoint:
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
            mf = cfg.motion_file
            env_cfg.commands.motion.motion_file = str(mf) if isinstance(mf, str) else [str(x) for x in mf]
        else:
            arts = [a for a in wandb_run.used_artifacts() if a.type == "motions"]
            if arts:
                arts.sort(key=lambda a: (0 if "forehand" in a.name.lower() else 1 if "backhand" in a.name.lower() else 2, a.name))
                motion_files = [str(pathlib.Path(art.download()) / "motion.npz") for art in arts]
                env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
            else:
                print("[WARN] No motion artifact in the run; pass motion_file=... if replay fails.")
    else:
        if checkpoint:
            resume_path = str(checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        reg = cfg.registry_name if cfg.registry_name is not None else cfg.task.get("registry_name")
        if cfg.motion_file is not None:
            # accept a list (unified 2-clip forehand+backhand) or a single path
            mf = cfg.motion_file
            env_cfg.commands.motion.motion_file = str(mf) if isinstance(mf, str) else [str(x) for x in mf]
        elif reg is not None:
            import wandb

            api = wandb.Api()

            def _dl(r):
                r = str(r)
                if ":" not in r:
                    r += ":latest"
                art = api.artifact(r)
                print(f"[play.py] motion clip: {r} -> {art.source_qualified_name} (digest {art.digest[:12]})", flush=True)
                return str(pathlib.Path(art.download()) / "motion.npz")

            # clip0 = registry_name (forehand); clip1 = registry_name_2 (backhand) if set. MUST match
            # train.py so the exported ONNX bakes the FULL concatenated motion (T = sum of seg_lens).
            # Without this, a unified policy exports forehand-only and backhand time_steps clamp to the
            # last forehand frame -> frozen reference -> dead backhand swing in deployment / sim2sim.
            motion_files = [_dl(reg)]
            reg2 = cfg.get("registry_name_2", None) or cfg.task.get("registry_name_2", None)
            if reg2 is not None and str(reg2).strip() and str(reg2).lower() != "none":
                motion_files.append(_dl(reg2))
                print(f"[play.py] UNIFIED 2-clip export: clip0={reg}  clip1={reg2}", flush=True)
            env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    expected_contract = cfg.task.get("actor_obs_contract", None)
    if expected_contract is not None:
        contract = validate_actor_observation_contract(env.unwrapped, str(expected_contract))
        print(
            "[play.py] actor observation contract validated: "
            f"{contract.name} ({contract.total_dim}D, obs_mode={contract.obs_mode})",
            flush=True,
        )
    log_dir = os.path.dirname(resume_path)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export the policy to ONNX next to the checkpoint (step 15)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped, ppo_runner.alg.policy,
        normalizer=getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None),
        path=export_model_dir, filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, str(wandb_path) if wandb_path else "none", export_model_dir)
    print(f"[INFO] Exported ONNX policy to: {export_model_dir}")

    # Manual video capture: grab env.render() each step and encode to mp4 with imageio
    # (imageio-ffmpeg). Avoids gym RecordVideo's vec-env / flush quirks and reports exactly
    # how many frames were captured so a black/empty render is obvious instead of silent.
    frames = []
    # This rsl_rl/IsaacLab version returns a TensorDict from get_observations() (NOT an (obs, extras)
    # tuple) and the inference policy consumes the whole TensorDict — mirror the runner's rollout loop.
    obs = env.get_observations().to(agent_cfg.device)
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(env.unwrapped.device))
        if cfg.video:
            frame = env.unwrapped.render()
            if frame is not None:
                frames.append(frame)
            timestep += 1
            if timestep >= int(cfg.video_length):
                break
        # non-video: keep stepping until the Isaac Sim window is closed (live viewing)

    if cfg.video:
        import numpy as np

        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, "play.mp4")
        valid = [np.asarray(f) for f in frames if f is not None and getattr(f, "size", 0) > 0]
        print(f"[INFO] captured {len(frames)} frames ({len(valid)} non-empty)", flush=True)
        if valid:
            import imageio

            imageio.mimsave(video_path, valid, fps=30)
            print(f"[INFO] wrote video -> {video_path}", flush=True)
        else:
            print(
                "[ERROR] env.render() returned no usable frames. Check that AppLauncher got "
                "enable_cameras=True (it ties to video) and render_mode='rgb_array'.",
                flush=True,
            )

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
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
