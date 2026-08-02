"""Isolated, diagnostic-only building blocks for native MuJoCo training.

This package deliberately does not import Isaac Lab or ``rsl_rl``.  It now
contains the single-environment plant/action runner and a diagnostic physical-
ball N1 wrapper.  Reward, VecEnv, PPO and canonical training authorization
remain outside it.

The implementation is not imported here so ``python -m ...single_env`` has a
single, warning-free module execution path.
"""

__all__ = ["n1_ball_core", "physical_ball_scene", "single_env", "vec_env"]
