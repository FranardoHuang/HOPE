"""Gym-registry PPO runner cfg for the Agibot A3 table-tennis task.

Backs the registry ``rsl_rl_cfg_entry_point`` used by the dedicated, motion-free trainer
``scripts/train_table_tennis.py`` (the shared ``scripts/train.py`` / ``scripts/rsl_rl/train.py`` are
hardwired to the WBC *tracking* task: they require a wandb motion registry and set
``env_cfg.commands.motion.motion_file``, which the table-tennis scene does not have).

Hyperparameters come from ``cfg/algo/ppo.yaml`` via :mod:`whole_body_tracking.utils.ppo_cfg` — the same
single source of truth the WBC tasks use. Tune by editing that YAML (or set ``WBT_AGIBOT_A3_PPO_CFG``).
``experiment_name`` is the ``logs/rsl_rl/<name>/`` directory.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs

_KW = runner_kwargs(load_ppo_params(), "agibot_a3_table_tennis")


@configclass
class TableTennisAgibotA3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = _KW["num_steps_per_env"]
    max_iterations = _KW["max_iterations"]
    save_interval = _KW["save_interval"]
    experiment_name = _KW["experiment_name"]
    empirical_normalization = _KW["empirical_normalization"]
    policy = _KW["policy"]
    algorithm = _KW["algorithm"]
