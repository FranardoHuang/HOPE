"""Fresh full-MDP pre-reward DoneTerm callable.

The callable delegates publication to the lease-bound top runtime owner via
``FreshFullMdpRewardGraph``.  It is not a time-limit termination, and it never
calls R03 or R07 directly.  The current manager configuration does not consume
this module, so all launch claims remain false.
"""

from __future__ import annotations

import torch

try:
    from . import action_ball_full_mdp_rewards as _rewards
except ImportError:  # dependency-light direct module diagnostics
    import action_ball_full_mdp_rewards as _rewards


SCHEMA_VERSION = 1
TIME_OUT = False
RUNTIME_INTEGRATED = False
MANAGER_CONFIG_INTEGRATED = False
LAUNCH_AUTHORIZED = False


def _control_step(env: object, attr: str) -> int:
    value = getattr(env, attr, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        # A CUDA scalar conversion here would create a hidden synchronization;
        # the environment must publish its ordinary host control-step integer.
        raise _rewards.FreshFullMdpRewardCycleError(
            f"env.{attr} must be an exact nonnegative host int"
        )
    return value


def fresh_full_mdp_pre_reward_done_term(
    env: object,
    *,
    graph_attr: str = "action_ball_full_mdp_reward_graph",
    control_step_attr: str = "common_step_counter",
) -> torch.Tensor:
    """Publish R03+R07 through the top lease and return terminal rows.

    A manager config must install this term with ``time_out=False``.  Ordinary
    misses and falls are represented by the owners' facts and are not changed
    into infrastructure errors by this wrapper.
    """

    graph = getattr(env, graph_attr, None)
    if type(graph) is not _rewards.FreshFullMdpRewardGraph:
        raise _rewards.FreshFullMdpRewardConstructionHold(
            f"env.{graph_attr} is not the exact fresh Reward graph"
        )
    return graph.begin_pre_reward(
        control_step=_control_step(env, control_step_attr)
    )


__all__ = [
    "SCHEMA_VERSION",
    "TIME_OUT",
    "RUNTIME_INTEGRATED",
    "MANAGER_CONFIG_INTEGRATED",
    "LAUNCH_AUTHORIZED",
    "fresh_full_mdp_pre_reward_done_term",
]
