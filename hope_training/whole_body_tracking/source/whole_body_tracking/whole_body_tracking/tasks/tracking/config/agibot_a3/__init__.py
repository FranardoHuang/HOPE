import gymnasium as gym

from . import agents, flat_env_cfg, hope_env_cfg

##
# Register Gym environments.
##

# Plain BeyondMimic motion tracking on the A3 (baseline).
gym.register(
    id="Tracking-Flat-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AgibotA3FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:AgibotA3FlatPPORunnerCfg",
    },
)

# HOPE ping-pong WBC with racket-target tracking (step 13/14).
gym.register(
    id="HOPE-PingPong-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# HOPE ping-pong WBC — deploy-parity actor observation (no fabricated base pose).
# Same task/reward family; the actor obs drops every world-frame base-position dependency (180 -> 175)
# and adds absolute balance rewards/terminations. The `full` env above is unchanged.
gym.register(
    id="HOPE-PingPong-DeployParity-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongDeployParityAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Backward-compatible alias for older docs/scripts that still say `RealSensor`.
gym.register(
    id="HOPE-PingPong-RealSensor-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongRealSensorAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# HOPE ping-pong WBC — deploy-parity obs + Tier-1 virtual-ball outcome rewards (rewardDesign.md).
# REWARD-ONLY variant: identical 175-D actor contract; the virtual ball lives only in the reward.
gym.register(
    id="HOPE-PingPong-VirtualBall-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongVirtualBallAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# HOPE ping-pong WBC — HITTER separate base/racket commands (arXiv:2508.21043 §V-B-1).
# Deploy-parity base + base_target_pos_b actor obs restored (175 -> 177) + pre-strike base
# tracking reward + reference-reach base/racket coupling. NOT deploy-compatible with the
# 175-D C++ runner until it grows the base channel (see hope_env_cfg comments).
gym.register(
    id="HOPE-PingPong-Hitter-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongHitterAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)
