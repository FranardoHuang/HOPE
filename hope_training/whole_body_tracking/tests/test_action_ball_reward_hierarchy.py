from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_reward_flags_mdp import (
    _fake_env,
    _fake_racket_cmd,
    hope_commands_mod,
    hope_rewards_mod,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_action_ball_reward_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("reward_hierarchy_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_vendor_v2_static_hierarchy_and_live_error_counterfactual():
    task = ROOT / "cfg/task/HOPEPingPongActionBallA3VendorV2.yaml"
    audit = MODULE.build_audit(
        task,
        observed_errors=(0.6340, 1.9595, math.radians(56.21)),
    )
    assert audit["all_static_hierarchy_checks_pass"] is True
    assert all(audit["system_recipe"]["checks"].values())
    assert audit["system_recipe"]["checks"] == {
        "full_body_mimic": True,
        "measured_racket_teacher": True,
        "three_channel_monotonic_adaptive_fine": True,
        "action_ball_target_mode": True,
        "ball_outcome_enabled": True,
        "table_obstacle_enabled": True,
        "complete_reward_pack": True,
    }
    summary = audit["catalog_summary"]
    assert summary["action_count"] == 73
    assert summary["longest_action"] == "Take_062_unit11_BH"
    assert summary["longest_action_frames"] == 133
    assert summary["all_actions_strict_order_at_fine_acceptance"] is True
    assert math.isclose(summary["action_prior_cap_max"], 3.6575)
    assert audit["constants"]["channel_window_steps"] == {
        "position": 3,
        "velocity": 11,
        "normal": 11,
    }
    assert audit["constants"]["adaptive_sigma_bounds"] == {
        "pos": {"min": 0.075, "max": 0.5},
        "vel": {"min": 0.5, "max": 3.0},
        "normal": {"min": 0.262, "max": 2.1},
    }
    assert audit["constants"]["motion_racket_scale_in_strike_window"] == 1.0
    assert (
        audit["constants"]["motion_racket_long_axis_scale_in_strike_window"]
        == 1.0
    )
    assert math.isclose(summary["target_income_at_fine_acceptance"], 4.030, abs_tol=0.001)
    assert math.isclose(
        summary["target_initial_income_at_fine_acceptance"], 4.310, abs_tol=0.001
    )
    assert math.isclose(summary["broad_one_sigma_income"], 1.95)
    assert math.isclose(summary["target_kernel_plus_progress_upper"], 5.485)
    for row in audit["actions"].values():
        assert row["action_prior_undiscounted_cap"] < row[
            "ball_target_income_at_fine_acceptance"
        ]
        assert math.isclose(row["ball_target_broad_envelope_floor"], 1.95)
        assert row["legal_landing_event_min"] == 6.0
    observed = audit["frozen_observed_exact_strike_counterfactual"]
    assert observed["v1_window_kernel_income"] < 0.01
    assert math.isclose(observed["v2_window_kernel_income"], 2.664, abs_tol=0.001)
    assert math.isclose(
        observed["v2_window_kernel_income_at_initial_sigma"], 2.873, abs_tol=0.001
    )


def test_cauchy_tail_is_live_and_toward_step_improves_reward():
    scale = 0.7
    error = 1.4
    step = 1.0e-4
    original = MODULE.cauchy(error, scale)
    toward = MODULE.cauchy(error - step, scale)
    away = MODULE.cauchy(error + step, scale)
    assert toward > original > away
    finite_difference = (toward - away) / (2.0 * step)
    assert math.isfinite(finite_difference)
    assert finite_difference > 0.1


def test_runtime_cauchy_terms_keep_far_error_signal_and_channel_windows():
    position_window = torch.tensor([True, True, False])
    wide_window = torch.tensor([True, True, True])
    cmd = _fake_racket_cmd(
        3,
        window=wide_window,
        window_pos=position_window,
        window_wide=wide_window,
    )
    cmd.racket_pos_w[:, 0] = torch.tensor([0.0, 1.4, 1.4])
    cmd.racket_lin_vel_w[:, 0] = torch.tensor([0.0, 8.0, 8.0])
    cmd.racket_normal_w = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    )
    env = _fake_env(racket_target=cmd)

    position = hope_rewards_mod.racket_position_coarse_tracking_cauchy(
        env, "racket_target", std=0.7
    )
    velocity = hope_rewards_mod.racket_velocity_coarse_tracking_cauchy(
        env, "racket_target", std=4.0
    )
    normal = hope_rewards_mod.racket_normal_coarse_tracking_cauchy(
        env, "racket_target", std=math.pi
    )

    torch.testing.assert_close(position, torch.tensor([1.0, 0.2, 0.0]))
    torch.testing.assert_close(velocity, torch.tensor([1.0, 0.2, 0.2]))
    torch.testing.assert_close(normal, torch.tensor([1.0, 0.5, 0.5]))


def test_measured_long_axis_reward_supervises_wrist_twist_including_contact_pin(monkeypatch):
    cmd = _fake_racket_cmd(
        3,
        window=torch.tensor([True, False, False]),
        window_wide=torch.tensor([True, False, False]),
    )
    cmd.racket_long_axis_w = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    target = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
    monkeypatch.setattr(
        hope_rewards_mod,
        "_stage1_aligned_clip_long_axis_target",
        lambda _cmd: target,
    )
    env = _fake_env(racket_target=cmd)
    reward = hope_rewards_mod.motion_racket_long_axis_tracking_cauchy(
        env,
        "racket_target",
        std=1.0,
        scale_in_strike_window=1.0,
    )
    assert reward[0].item() == 1.0
    assert math.isclose(
        reward[1].item(), 1.0 / (1.0 + (math.pi / 2.0) ** 2), abs_tol=1.0e-6
    )
    assert math.isclose(
        reward[2].item(), 1.0 / (1.0 + math.pi**2), abs_tol=1.0e-6
    )


def _adaptive_resume_command():
    cfg = SimpleNamespace(
        adaptive_sigma=True,
        adaptive_sigma_monotonic=True,
        adaptive_sigma_normal=True,
        adaptive_sigma_source="ball_exact_strike",
        sigma_pos_min=0.075,
        sigma_pos_max=0.50,
        sigma_vel_min=0.50,
        sigma_vel_max=3.0,
        sigma_normal_min=0.262,
        sigma_normal_max=2.10,
        sigma_update_every=500,
        sigma_ema_scale=1.0,
        exact_success_decay=0.99,
        exact_success_min_count=32.0,
        strike_window_pos_s=0.02,
        strike_window_wide_s=0.10,
    )
    terms = {
        "racket_position": SimpleNamespace(params={"std": 0.50}),
        "racket_velocity": SimpleNamespace(params={"std": 3.0}),
        "racket_normal": SimpleNamespace(params={"std": 2.10}),
        "racket_strike_success": SimpleNamespace(
            params={"std_pos": 0.50, "std_vel": 3.0, "std_normal": 2.10}
        ),
    }
    manager = SimpleNamespace(get_term_cfg=lambda name: terms[name])
    command = hope_commands_mod.RacketTargetCommand.__new__(
        hope_commands_mod.RacketTargetCommand
    )
    command.cfg = cfg
    command._env = SimpleNamespace(reward_manager=manager)
    command._adaptive_sigma_pos = 0.20
    command._adaptive_sigma_vel = 1.25
    command._adaptive_sigma_normal = 0.80
    command._exact_n_acc = 41.5
    command._exact_pos_err_sum = 7.0
    command._exact_vel_err_sum = 12.0
    command._exact_nrm_err_sum = 5.0
    return command, terms


def test_action_ball_adaptive_sigma_state_roundtrips_driver_and_live_reward_widths():
    command, terms = _adaptive_resume_command()
    state = command._action_ball_adaptive_sigma_state_dict()
    staged = command._action_ball_stage_adaptive_sigma_state(state)
    command._adaptive_sigma_pos = 0.50
    command._adaptive_sigma_vel = 3.0
    command._adaptive_sigma_normal = 2.10
    command._exact_n_acc = 0.0
    command._exact_pos_err_sum = 0.0
    command._exact_vel_err_sum = 0.0
    command._exact_nrm_err_sum = 0.0
    command._action_ball_commit_adaptive_sigma_state(staged)

    assert command._adaptive_sigma_pos == 0.20
    assert command._adaptive_sigma_vel == 1.25
    assert command._adaptive_sigma_normal == 0.80
    assert command._exact_n_acc == 41.5
    assert command._exact_pos_err_sum == 7.0
    assert command._exact_vel_err_sum == 12.0
    assert command._exact_nrm_err_sum == 5.0
    assert terms["racket_position"].params["std"] == 0.20
    assert terms["racket_velocity"].params["std"] == 1.25
    assert terms["racket_normal"].params["std"] == 0.80
    assert terms["racket_strike_success"].params == {
        "std_pos": 0.20,
        "std_vel": 1.25,
        "std_normal": 0.80,
    }


def test_action_ball_adaptive_sigma_resume_rejects_out_of_bounds_width():
    command, _ = _adaptive_resume_command()
    state = command._action_ball_adaptive_sigma_state_dict()
    state["sigma"]["normal"] = 2.11
    with pytest.raises(ValueError, match="normal sigma is outside"):
        command._action_ball_stage_adaptive_sigma_state(state)
