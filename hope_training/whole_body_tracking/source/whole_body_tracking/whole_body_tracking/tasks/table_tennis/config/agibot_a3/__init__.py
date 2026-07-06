import gymnasium as gym

from . import agents, table_tennis_env_cfg

##
# Register the Agibot A3 table-tennis match environment.
##

gym.register(
    id="HOPE-TableTennis-AgibotA3-v0",
    entry_point="whole_body_tracking.tasks.table_tennis.table_tennis_env:TableTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": table_tennis_env_cfg.AgibotA3TableTennisEnvCfg,
        # PPO runner cfg for the motion-free trainer scripts/train_table_tennis.py. The shared
        # scripts/train.py path is not wired for this task (it requires a wandb motion registry).
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:TableTennisAgibotA3PPORunnerCfg",
    },
)
