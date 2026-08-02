"""Isolated, diagnostic-only building blocks for native MuJoCo training.

This package deliberately does not import Isaac Lab or ``rsl_rl``.  The first
deliverable is the single-environment plant/action runner; balls, rewards,
observations, PPO and canonical training authorization remain outside it.

The implementation is not imported here so ``python -m ...single_env`` has a
single, warning-free module execution path.
"""

__all__ = ["single_env"]
