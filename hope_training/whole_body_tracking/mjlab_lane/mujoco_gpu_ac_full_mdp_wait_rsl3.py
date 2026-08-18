"""Run one upstream RSL-RL 3 update on the real MuJoCo WAIT environment.

This is an engineering callpoint, not a complete ActionBall task: question
reveal, contact, outcome, and recovery remain absent while the epoch is IDLE.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat


RSL_RL_VERSION = "3.1.2"
NUM_STEPS_PER_ENV = 24
READY_POSE_SHA256 = "ab6b7e41ff129f91238835c533c8d589e68cc21f7e6184d639e95d8938d38069"


def _ready_pose_input() -> tuple[bytes, str]:
    raw = os.environ.get("ACTIONBALL_READY_POSE")
    if not raw:
        raise RuntimeError("MuJoCo WAIT ready-pose path is not bound")
    path = Path(raw)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError("MuJoCo WAIT ready-pose path differs") from exc
    try:
        row = os.fstat(fd)
        resolved = path.resolve(strict=True)
        payload = b""
        while chunk := os.read(fd, 1024 * 1024):
            payload += chunk
        current = path.stat(follow_symlinks=False)
        if (
            not path.is_absolute()
            or not stat.S_ISREG(row.st_mode)
            or row.st_nlink != 1
            or current.st_nlink != 1
            or resolved != path
            or (row.st_dev, row.st_ino) != (current.st_dev, current.st_ino)
            or hashlib.sha256(payload).hexdigest() != READY_POSE_SHA256
        ):
            raise RuntimeError("MuJoCo WAIT ready-pose path differs")
        return payload, str(path)
    except OSError as exc:
        raise RuntimeError("MuJoCo WAIT ready-pose path differs") from exc
    finally:
        os.close(fd)


def build_train_cfg() -> dict:
    """Return the same RSL3 PPO surface used by the Isaac FullMDP run."""

    return {
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "save_interval": 1,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 0.02,
            "noise_std_type": "log",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def _wait_module():
    module = importlib.import_module("mujoco_gpu_ac_full_mdp_initial_wait_env")
    expected = Path(__file__).with_name(
        "mujoco_gpu_ac_full_mdp_initial_wait_env.py"
    ).resolve()
    actual = Path(getattr(module, "__file__", "")).resolve()
    if actual != expected:
        raise RuntimeError("MuJoCo WAIT environment import origin differs")
    return module


def _rsl3_runner():
    import torch

    distribution = importlib.metadata.distribution("rsl-rl-lib")
    if distribution.version != RSL_RL_VERSION:
        raise RuntimeError(
            f"MuJoCo WAIT requires RSL-RL {RSL_RL_VERSION}, got {distribution.version}"
        )
    module = importlib.import_module("rsl_rl.runners.on_policy_runner")
    expected = Path(
        distribution.locate_file("rsl_rl/runners/on_policy_runner.py")
    ).resolve()
    runner = getattr(module, "OnPolicyRunner", None)
    actual = Path(getattr(module, "__file__", "")).resolve()
    source = Path(inspect.getsourcefile(runner) or "").resolve()
    if actual != expected or source != expected:
        raise RuntimeError("MuJoCo WAIT RSL-RL import origin differs")
    ppo_module = importlib.import_module("rsl_rl.algorithms.ppo")
    actor_module = importlib.import_module("rsl_rl.modules.actor_critic")
    recurrent_module = importlib.import_module("rsl_rl.modules.actor_critic_recurrent")
    storage_module = importlib.import_module("rsl_rl.storage.rollout_storage")
    _require_rsl3_preconstruction(
        distribution,
        module,
        ppo_module,
        actor_module,
        recurrent_module,
        storage_module,
        torch,
    )
    return distribution.version, runner, distribution


def _require_rsl3_preconstruction(
    distribution,
    runner_module,
    ppo_module,
    actor_module,
    recurrent_module,
    storage_module,
    torch_module,
) -> None:
    expected_modules = (
        (runner_module, "rsl_rl/runners/on_policy_runner.py"),
        (ppo_module, "rsl_rl/algorithms/ppo.py"),
        (actor_module, "rsl_rl/modules/actor_critic.py"),
        (recurrent_module, "rsl_rl/modules/actor_critic_recurrent.py"),
        (storage_module, "rsl_rl/storage/rollout_storage.py"),
    )
    if any(
        Path(getattr(module, "__file__", "")).resolve()
        != Path(distribution.locate_file(relative)).resolve()
        for module, relative in expected_modules
    ) or not (
        runner_module.PPO is ppo_module.PPO
        and runner_module.ActorCritic is actor_module.ActorCritic
        and runner_module.ActorCriticRecurrent is recurrent_module.ActorCriticRecurrent
        and ppo_module.RolloutStorage is storage_module.RolloutStorage
        and ppo_module.optim.Adam is torch_module.optim.Adam
    ):
        raise RuntimeError("MuJoCo WAIT RSL-RL preconstruction origin differs")


def _require_rsl3_runtime(distribution, runner, torch_module) -> None:
    try:
        alg = runner.alg
        runtime = (
            (alg, "rsl_rl/algorithms/ppo.py"),
            (alg.policy, "rsl_rl/modules/actor_critic.py"),
            (alg.storage, "rsl_rl/storage/rollout_storage.py"),
        )
        optimizer = alg.optimizer
    except AttributeError as exc:
        raise RuntimeError("MuJoCo WAIT RSL-RL runtime origin differs") from exc
    if any(
        Path(inspect.getsourcefile(type(value)) or "").resolve()
        != Path(distribution.locate_file(relative)).resolve()
        for value, relative in runtime
    ) or type(optimizer) is not torch_module.optim.Adam:
        raise RuntimeError("MuJoCo WAIT RSL-RL runtime origin differs")


def main() -> int:
    import torch

    version, runner_type, distribution = _rsl3_runner()
    wait = _wait_module()
    ready_pose_payload, ready_pose_source = _ready_pose_input()
    torch.manual_seed(0)
    task = wait.TaskCfg(
        action_scale_mode="vendor",
        reset_joint_noise_rad=0.0,
        reset_joint_vel_noise=0.0,
        reset_root_xy_noise_m=0.0,
        reset_root_yaw_noise_rad=0.0,
    )
    env = wait.FullMdpInitialWaitVecEnv(
        wait.SimCfg(nworld=2),
        task,
        device="cuda:0",
        seed=0,
        ready_pose_payload=ready_pose_payload,
        ready_pose_source=ready_pose_source,
    )
    initial = env.get_observations()
    if (
        env.num_actions != 31
        or tuple(initial["policy"].shape) != (2, 229)
        or tuple(initial["critic"].shape) != (2, 399)
        or not bool(torch.isfinite(initial["policy"]).all())
        or not bool(torch.isfinite(initial["critic"]).all())
    ):
        raise RuntimeError("MuJoCo WAIT initial RSL3 surface differs")

    runner = runner_type(env, build_train_cfg(), log_dir=None, device="cuda:0")
    _require_rsl3_runtime(distribution, runner, torch)
    runner.disable_logs = True
    updates = 0
    original_update = runner.alg.update

    def counted_update():
        nonlocal updates
        updates += 1
        return original_update()

    runner.alg.update = counted_update
    runner.learn(1, init_at_random_ep_len=False)
    final = env.get_observations()
    storage = runner.alg.storage
    if (
        updates != 1
        or env.common_step_counter != NUM_STEPS_PER_ENV
        or storage.step != 0
        or not runner.alg.optimizer.state
        or not bool(torch.isfinite(storage.rewards).all())
        or tuple(final["policy"].shape) != (2, 229)
        or tuple(final["critic"].shape) != (2, 399)
        or not bool(torch.isfinite(final["policy"]).all())
        or not bool(torch.isfinite(final["critic"]).all())
    ):
        raise RuntimeError("MuJoCo WAIT RSL3 update evidence differs")
    print(
        "ACTION_BALL_MUJOCO_WAIT_RSL3_JSON="
        + json.dumps(
            {
                "diagnostic_unauthorized": True,
                "rsl_rl_version": version,
                "ppo_update_calls": updates,
                "environment_steps": env.common_step_counter,
                "transitions": env.common_step_counter * env.num_envs,
                "policy_width": final["policy"].shape[1],
                "critic_width": final["critic"].shape[1],
                "task_lifecycle": "idle_wait_only",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
