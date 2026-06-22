"""Hydra training entry for HOPE Agibot A3 WBC (106B-Final-Project style).

Pick the task/algo YAML on the command line and override any field:

    python scripts/train.py task=HOPEPingPong algo=ppo headless=true \
        registry_name=<entity>/wandb-registry-motions/hope_forehand

    python scripts/train.py task=TrackingFlat algo=ppo num_envs=2048 max_iterations=20000 \
        registry_name=<entity>/wandb-registry-motions/hope_forehand

Tune by editing cfg/task/*.yaml (env / reward / racket / DR) and cfg/algo/ppo.yaml (PPO). This
script reuses BeyondMimic's training mechanics (Isaac Lab + rsl_rl + the wandb motion registry);
the legacy `scripts/rsl_rl/train.py --task=... --registry_name=...` still works too.
"""

import sys

import hydra
from omegaconf import OmegaConf


def dump_pickle(filename: str, data):
    """Compatibility helper for IsaacLab builds that no longer expose dump_pickle."""
    import os
    import pickle

    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# --------------------------------------------------------------------------- #
# Task YAML -> Isaac Lab env cfg overrides (only keys present in the YAML are applied).
# --------------------------------------------------------------------------- #
def _get(node, key, default=None):
    try:
        return node.get(key, default)
    except Exception:
        return default


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


def _set_attr(obj, attr, val, cast):
    if val is not None:
        setattr(obj, attr, cast(val))


def _set_range(obj, attr, val):
    if val is not None:
        setattr(obj, attr, (float(val[0]), float(val[1])))


def _set_reward(rewards, name, weight, std):
    if not hasattr(rewards, name):
        return
    term = getattr(rewards, name)
    if weight is not None:
        term.weight = float(weight)
    if std is not None:
        term.params["std"] = float(std)


def _apply_task_overrides(env_cfg, task):
    """Apply cfg/task/<name>.yaml overrides (incl. the composed base/ groups) onto the env cfg."""
    # env base (num_envs is applied earlier via parse_env_cfg)
    env = _get(task, "env")
    if env is not None:
        if _get(env, "env_spacing") is not None:
            env_cfg.scene.env_spacing = float(env.env_spacing)
        if _get(env, "episode_length_s") is not None:
            env_cfg.episode_length_s = float(env.episode_length_s)

    # sim base (control frequency = 1 / (dt * decimation))
    sim = _get(task, "sim")
    if sim is not None:
        if _get(sim, "dt") is not None:
            env_cfg.sim.dt = float(sim.dt)
        if _get(sim, "decimation") is not None:
            env_cfg.decimation = int(sim.decimation)
            env_cfg.sim.render_interval = env_cfg.decimation  # keep render in step with decimation

    rw = _get(task, "rewards")
    if rw is not None:
        R = env_cfg.rewards
        _set_reward(R, "racket_position", _get(rw, "racket_position_weight"), _get(rw, "racket_position_std"))
        _set_reward(R, "racket_velocity", _get(rw, "racket_velocity_weight"), _get(rw, "racket_velocity_std"))
        _set_reward(R, "racket_normal", _get(rw, "racket_normal_weight"), _get(rw, "racket_normal_std"))
        _set_reward(R, "base_position", _get(rw, "base_position_weight"), _get(rw, "base_position_std"))
        if hasattr(R, "joint_torques") and _get(rw, "joint_torques_weight") is not None:
            R.joint_torques.weight = float(rw.joint_torques_weight)

    rk = _get(task, "racket")
    if rk is not None and hasattr(env_cfg.commands, "racket_target"):
        C = env_cfg.commands.racket_target
        _set_attr(C, "strike_phase", _get(rk, "strike_phase"), float)
        _set_attr(C, "strike_window_s", _get(rk, "strike_window_s"), float)
        _set_attr(C, "strike_success_pos_thresh", _get(rk, "strike_success_pos_thresh"), float)
        _set_range(C, "racket_pos_x_range", _get(rk, "pos_x_range"))
        _set_range(C, "racket_pos_y_range", _get(rk, "pos_y_range"))
        _set_range(C, "racket_pos_z_range", _get(rk, "pos_z_range"))
        _set_range(C, "racket_vel_x_range", _get(rk, "vel_x_range"))
        _set_range(C, "racket_vel_y_range", _get(rk, "vel_y_range"))
        _set_range(C, "racket_vel_z_range", _get(rk, "vel_z_range"))
        _set_range(C, "base_target_x_range", _get(rk, "base_target_x_range"))
        _set_range(C, "base_target_y_range", _get(rk, "base_target_y_range"))
        _set_attr(C, "normal_mode", _get(rk, "normal_mode"), str)
        _set_attr(C, "forehand_on_negative_y", _get(rk, "forehand_on_negative_y"), _as_bool)
        _set_attr(C, "mount_normal_axis", _get(rk, "mount_normal_axis"), int)
        _set_attr(C, "mount_normal_sign", _get(rk, "mount_normal_sign"), float)

    dr = _get(task, "domain_rand")
    if dr is not None and hasattr(env_cfg, "events"):
        E = env_cfg.events
        mr = _get(dr, "link_mass_range")
        if mr is not None and hasattr(E, "randomize_link_mass"):
            E.randomize_link_mass.params["mass_distribution_params"] = (float(mr[0]), float(mr[1]))
        if hasattr(E, "randomize_pd_gains"):
            pr = _get(dr, "pd_gain_range")
            if pr is None:
                E.randomize_pd_gains = None  # disable
            else:
                E.randomize_pd_gains.params["stiffness_distribution_params"] = (float(pr[0]), float(pr[1]))
                E.randomize_pd_gains.params["damping_distribution_params"] = (float(pr[0]), float(pr[1]))


# --------------------------------------------------------------------------- #
# Training (runs after the simulator is launched).
# --------------------------------------------------------------------------- #
def _run(cfg):
    import os
    import pathlib
    from datetime import datetime

    import gymnasium as gym
    import torch

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) env cfg (gym registry) + task YAML overrides
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)

    # 2) PPO runner cfg from cfg.algo
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)
    if cfg.logger is not None:
        agent_cfg.logger = str(cfg.logger)
    if agent_cfg.logger in {"wandb", "neptune"} and cfg.log_project_name:
        agent_cfg.wandb_project = str(cfg.log_project_name)
        agent_cfg.neptune_project = str(cfg.log_project_name)

    # 3) motion file from the wandb registry (forehand/backhand clip)
    registry_name = cfg.registry_name if cfg.registry_name is not None else cfg.task.registry_name
    registry_name = str(registry_name)
    if ":" not in registry_name:
        registry_name += ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # 4) logging dir (same layout as scripts/rsl_rl/train.py so export/eval are unchanged)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Task: {task_id} | experiment: {agent_cfg.experiment_name} | log: {log_dir}")

    # 5) build env, wrap, run
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % int(cfg.video_interval) == 0,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    runner.add_git_repo_to_log(__file__)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so the kit app does not try to
    # parse Hydra's `task=...`/`algo=...` overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video)
    )
    simulation_app = app_launcher.app
    # Print the traceback BEFORE closing the app: Isaac's simulation_app.close() hard-exits the
    # process (os._exit), which otherwise swallows any exception from _run and makes a real failure
    # look like a clean "exit 0" with the log truncated at startup.
    failed = False
    try:
        _run(cfg)
    except Exception:
        import traceback
        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
