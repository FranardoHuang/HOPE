"""Probe-only action jerk and implicit-PD effort observability.

Both cfg fields are ``None`` by default.  An explicit boolean flag installs a weight-one
RewardTerm whose function returns exact zeros, so metrics cannot be pruned as zero-weight rewards
and the default vendor N1 manager/hot loop remains unchanged.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _Term, _apply_legacy_v1, _make_env_cfg, train_mod


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG_SRC = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
).read_text(encoding="utf-8")
RUNNER_SRC = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
).read_text(encoding="utf-8")


def _jerk_env(*, valid=(True, False)):
    action = types.SimpleNamespace(
        raw_actions=torch.tensor([[1.0, -2.0], [0.3, 0.1]]),
        prev_raw_actions=torch.tensor([[0.5, -1.0], [0.2, 0.0]]),
        prev_prev_raw_actions=torch.tensor([[0.25, -0.25], [0.1, -0.1]]),
        raw_action_history_valid=torch.tensor(valid, dtype=torch.bool),
    )
    env = types.SimpleNamespace(
        common_step_counter=7,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
    )
    return env, action


def test_action_acc_probe_returns_zero_and_books_raw_and_clamped_metrics_once():
    env, action = _jerk_env()
    value = hope_rewards_mod.action_acc_jerk_probe(env, value_clamp=0.1)
    assert torch.equal(value, torch.zeros(2))
    state = getattr(env, hope_rewards_mod._ACTION_ACC_PROBE_STATE_ATTR)
    assert state["observed_sample_count"].item() == 2
    assert state["history_valid_sample_count"].item() == 1
    assert state["above_clamp_sample_count"].item() == 1
    assert state["raw_jerk_square_sum"].item() == pytest.approx(0.125)
    assert state["clamped_jerk_square_sum"].item() == pytest.approx(0.1)

    # Same RewardManager step is idempotent.  A later step gets a new charge.
    hope_rewards_mod.action_acc_jerk_probe(env, value_clamp=0.1)
    assert state["observed_sample_count"].item() == 2
    env.common_step_counter += 1
    action.raw_actions[0] = torch.tensor([10.0, 10.0])
    hope_rewards_mod.action_acc_jerk_probe(env, value_clamp=0.1)
    assert state["observed_sample_count"].item() == 4
    assert state["above_clamp_sample_count"].item() == 2

    snapshot = hope_rewards_mod.consume_action_acc_jerk_probe_counters(env)
    assert snapshot["history_valid_sample_count"].item() == 2
    assert all(value.item() == 0 for value in state.values())


def _implicit_env(*, implicit=True):
    q_des = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    data = types.SimpleNamespace(
        joint_names=["j0", "j1"],
        joint_pos=torch.zeros(2, 2),
        joint_vel=torch.tensor([[0.0, 2.0], [0.0, 0.0]]),
        joint_stiffness=torch.full((2, 2), 10.0),
        joint_damping=torch.full((2, 2), 2.0),
        joint_effort_limits=torch.tensor([[10.0, 5.0], [10.0, 5.0]]),
    )
    asset = types.SimpleNamespace(
        data=data,
        actuators={
            "all": types.SimpleNamespace(
                joint_indices=[0, 1], is_implicit_model=implicit
            )
        },
    )
    action = types.SimpleNamespace(
        processed_actions=q_des,
        _asset=asset,
        _joint_ids=[0, 1],
    )
    env = types.SimpleNamespace(
        common_step_counter=11,
        action_manager=types.SimpleNamespace(get_term=lambda name: action),
    )
    return env


def test_implicit_pd_proxy_uses_live_gains_sent_qdes_and_post_step_state():
    env = _implicit_env()
    value = hope_rewards_mod.implicit_pd_post_step_effort_proxy_probe(env)
    assert torch.equal(value, torch.zeros(2))
    state = getattr(env, hope_rewards_mod._IMPLICIT_PD_PROXY_STATE_ATTR)
    # Analytic ratios: [[1.0, 0.8], [2.0, 0.0]].
    assert state["observed_joint_sample_count"].item() == 4
    assert state["valid_joint_sample_count"].item() == 4
    assert state["above_soft_ratio_joint_count"].item() == 2
    assert state["above_limit_joint_count"].item() == 1
    assert state["utilization_ratio_sum"].item() == pytest.approx(3.8)
    assert state["excess_over_limit_ratio_sum"].item() == pytest.approx(1.0)
    assert state["peak_utilization_ratio"].item() == pytest.approx(2.0)

    hope_rewards_mod.implicit_pd_post_step_effort_proxy_probe(env)
    assert state["observed_joint_sample_count"].item() == 4
    snapshot = hope_rewards_mod.consume_implicit_pd_post_step_effort_proxy_counters(env)
    assert snapshot["peak_utilization_ratio"].item() == pytest.approx(2.0)
    assert all(value.item() == 0 for value in state.values())


def test_implicit_pd_proxy_refuses_explicit_or_unproven_backend():
    with pytest.raises(RuntimeError, match="complete implicit-actuator ownership"):
        hope_rewards_mod.implicit_pd_post_step_effort_proxy_probe(
            _implicit_env(implicit=False)
        )


def _probe_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.action_acc_jerk_probe = None
    cfg.rewards.implicit_pd_post_step_effort_proxy_probe = None
    return cfg


def _apply_probe(rewards):
    cfg = _probe_cfg()
    applied = _apply_legacy_v1(cfg, {"rewards": rewards})
    return cfg, applied


def test_probe_flags_are_whitelisted_and_default_path_constructs_nothing():
    for key in (
        "action_acc_jerk_probe",
        "implicit_pd_post_step_effort_proxy_probe",
    ):
        assert key in train_mod._REWARD_KEYS
    cfg, applied = _apply_probe({"racket_position_weight": 14.0})
    assert cfg.rewards.action_acc_jerk_probe is None
    assert cfg.rewards.implicit_pd_post_step_effort_proxy_probe is None
    assert not [line for line in applied if "probe" in line]


@pytest.mark.parametrize(
    ("key", "attr", "func_name"),
    [
        ("action_acc_jerk_probe", "action_acc_jerk_probe", "action_acc_jerk_probe"),
        (
            "implicit_pd_post_step_effort_proxy_probe",
            "implicit_pd_post_step_effort_proxy_probe",
            "implicit_pd_post_step_effort_proxy_probe",
        ),
    ],
)
def test_true_installs_weight_one_zero_reward_probe_and_false_leaves_none(
    key, attr, func_name
):
    cfg, applied = _apply_probe({key: True})
    term = getattr(cfg.rewards, attr)
    assert isinstance(term, _Term)
    assert term.weight == 1.0
    assert term.func.__name__ == func_name
    assert f"rewards.{attr}.enabled=True" in applied

    cfg, applied = _apply_probe({key: False})
    assert getattr(cfg.rewards, attr) is None
    assert f"rewards.{attr}.enabled=False" in applied


@pytest.mark.parametrize("bad", [2, "maybe", [], {}])
def test_probe_flags_require_literal_boolean(bad):
    with pytest.raises((train_mod._OverrideError, TypeError, ValueError)):
        _apply_probe({"action_acc_jerk_probe": bad})


def test_cfg_and_runner_keep_probe_boundary_explicit():
    # DeployParity + HitterPure each declare unconstructed slots.
    assert ENV_CFG_SRC.count("action_acc_jerk_probe = None") == 2
    assert ENV_CFG_SRC.count("implicit_pd_post_step_effort_proxy_probe = None") == 2
    assert 'if "action_acc_jerk_probe" in active_reward_terms' in RUNNER_SRC
    assert 'if "implicit_pd_post_step_effort_proxy_probe" in active_reward_terms' in RUNNER_SRC
    source = hope_rewards_mod.implicit_pd_post_step_effort_proxy_probe.__doc__
    assert "post-policy-step analytic demand proxy" in source
    assert "actual torque" in source
    assert "substeps" in source
