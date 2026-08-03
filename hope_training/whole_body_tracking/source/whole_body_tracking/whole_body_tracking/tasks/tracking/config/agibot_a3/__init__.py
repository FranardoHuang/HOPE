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

# HOPE ping-pong WBC — action-conditioned ball-first launch source.  The registered class keeps
# the native 177-D HITTER prefix and adds the complete VirtualBall/v2 reward lineage.  The exact
# +4 face tail and +N action identity are appended only by train.py after manifest/order preflight;
# the task YAML therefore cannot silently choose an action-bank size.
gym.register(
    id="HOPE-PingPong-ActionBall-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongActionBallAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Fixed-question A225 four-arm learnability diagnostic.  This is intentionally
# separate from both the historical fixed-194 ActionBall task and the
# construction-only A225/C225 leaves.  Its explicit 318-D privileged critic is
# mandatory; official train entry points reject any symmetric fallback.
gym.register(
    id="HOPE-PingPong-ActionBall-A225Learnability-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_env_cfg.HOPEPingPongActionBallA225LearnabilityAgibotA3EnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Fixed-midpoint incoming-ball-direct C225 diagnostic.  This leaf has its own
# 318-D privileged critic and fresh normalizer/checkpoint identities; it never
# falls back to the actor or reuses the A225 lineage.
gym.register(
    id="HOPE-PingPong-ActionBall-C225Learnability-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_env_cfg.HOPEPingPongActionBallC225LearnabilityAgibotA3EnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Historical ball-free natural-motion Stage 1 V1 (170-D actor, window-only paddle reward).
# Current source retains this Gym id for provenance but rejects construction because its retired
# adaptive-sigma controller is no longer available.  Production launches use the v1 Gym version
# below (task profile VendorV2); never remap this id to V2 semantics.
gym.register(
    id="HOPE-PingPong-Stage1NaturalClip-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_env_cfg.HOPEPingPongStage1NaturalClipAgibotA3EnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Current ball-free natural-motion Stage 1 V2.  This is a new observation/reward ABI:
# 225-D actor, 318-D critic, dense full-phase official-paddle learning, fixed broad capture kernels
# and windowed precision overlays.  The versioned Gym id prevents old checkpoints/configs from
# silently inheriting these meanings.
gym.register(
    id="HOPE-PingPong-Stage1NaturalClip-AgibotA3-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_env_cfg.HOPEPingPongStage1NaturalClipV2AgibotA3EnvCfg
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# HOPE ping-pong WBC — HITTER-PURE faithful reproduction (2026-07-07, arXiv:2508.21043).
# Actor = Table I exact (110-D hitter_pure: NO reference stream, NO swing_type, world-frame
# targets + e_base,x); independent station sampling + station-relative fixed striking plane;
# face-normal target = velocity direction (§IV-C); no hold / HER / foot shaping. Needs a NEW
# C++ obs builder + continuously-streaming planner targets before deploy (see hope_env_cfg).
gym.register(
    id="HOPE-PingPong-HitterPure-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongHitterPureAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# HITTER-PURE + CONTINUOUS RALLY (2026-07-07): same 110-D contract/boxes, adds the between-swing
# hold/recovery window (0.5-2.5 s at every wrap), follow-through braking + hold settle income, and
# 16 s episodes — targets the deploy Gate-2.5 P7 walked-forward drift fall. Warm-resume a
# HitterPure checkpoint with checkpoint_path=... (strict load works; identical obs/critic layout).
gym.register(
    id="HOPE-PingPong-HitterPureRally-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongHitterPureRallyAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)
