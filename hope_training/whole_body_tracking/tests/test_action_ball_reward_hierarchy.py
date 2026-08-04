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
TASK_RECEIPT = (
    ROOT.parents[1]
    / "configs/action_ball_n1_measured_20260803"
    / "fresh_tape_seed0_20260803_take061_robust20n_r4_splitready"
    / "current_lm.target.task_receipt.v5.f64f52137ad8.json"
)
SPEC = importlib.util.spec_from_file_location("reward_hierarchy_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_vendor_v2_catalog_arithmetic_is_explicitly_partial():
    task = ROOT / "cfg/task/HOPEPingPongActionBallA3VendorV2.yaml"
    audit = MODULE.build_audit(
        task,
        observed_errors=(0.6340, 1.9595, math.radians(56.21)),
    )
    assert audit["all_static_hierarchy_checks_pass"] is False
    assert audit["partial_catalog_arithmetic_checks_pass"] is True
    assert audit["hierarchy_authority"]["n73_authorized"] is False
    assert audit["selected_n1_wall_clock"]["complete"] is False
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
    assert summary["partial_catalog_frame_only_order_at_fine_acceptance"] is True
    assert math.isclose(
        summary["native_catalog_frame_only_imitation_cap_max"], 3.6575
    )
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
        assert row["native_catalog_frame_only_imitation_cap"] < row[
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


def test_a211_candidate_prices_window_and_progress_below_landing():
    task = ROOT / "cfg/task/HOPEPingPongActionBallA211VendorV2N1Learnability.yaml"
    audit = MODULE.build_audit(
        task,
        observed_errors=(0.6340, 1.9595, math.radians(56.21)),
        task_receipt_path=TASK_RECEIPT,
    )

    assert audit["all_static_hierarchy_checks_pass"] is True
    assert audit["partial_catalog_arithmetic_checks_pass"] is True
    assert audit["system_recipe"]["fine_width_mode"] == "static_rollout0"
    assert audit["system_recipe"]["checks"] == {
        "full_body_mimic": True,
        "measured_racket_teacher": True,
        "three_channel_static_fine": True,
        "action_ball_target_mode": True,
        "ball_outcome_enabled": True,
        "table_obstacle_enabled": True,
        "complete_reward_pack": True,
    }
    summary = audit["catalog_summary"]
    assert math.isclose(
        summary["native_catalog_frame_only_imitation_cap_max"], 3.6575
    )
    assert math.isclose(
        summary["target_income_at_fine_acceptance"], 4.665611620979069
    )
    assert math.isclose(
        summary["target_initial_income_at_fine_acceptance"], 4.665611620979069
    )
    assert math.isclose(summary["target_kernel_plus_progress_upper"], 6.16825)
    assert audit["constants"]["adaptive_sigma_bounds"] == {
        "pos": {"min": 0.5, "max": 0.5},
        "vel": {"min": 3.0, "max": 3.0},
        "normal": {"min": 2.1, "max": 2.1},
    }
    taxonomy = audit["active_reward_taxonomy"]
    # 1.0 -> 0.25：原值折扣收入 +1.9869 = task-valid mimic 1.77331 的 112%，
    # 即“站着不动”胜过“学动作”。详见 exp §5.6 偏离记录第 5 条。
    assert taxonomy["scientific_groups"]["balance"]["upright_exp"] == 0.25
    assert "base_position" not in taxonomy["scientific_groups"]["target"]
    assert taxonomy["scientific_groups"]["target"]["racket_progress"] == 10.0
    assert taxonomy["scientific_groups"]["strike"]["strike_capture_bonus"] == 25.0
    assert taxonomy["scientific_groups"]["outcome"] == {
        "virtual_pass_net": 20.0,
        "virtual_landing_dense": 20.0,
        "virtual_landing": 700.0,
    }
    timing = audit["selected_n1_wall_clock"]
    assert timing["teacher_rate"] == pytest.approx(0.8513476051357717)
    assert timing["pre_swing_wait_s"] == pytest.approx(0.7123759904781779)
    assert timing["scaled_t_cycle_s"] == pytest.approx(1.3155613444421261)
    assert timing["episode_contact_tick"] == {"earliest": 97, "latest": 117}
    assert timing["mimic_undiscounted_cap_envelope"]["max_wait"] == pytest.approx(
        3.52
    )
    assert timing["mimic_reward_support_steps_conservative"][
        "task_valid_swing_upper"
    ] == 103
    assert timing["mimic_task_reveal_discounted_cap"] == pytest.approx(
        1.7733077595610476
    )
    layers = audit["selected_n1_layer_accounting"]
    assert layers["undiscounted_per_swing"]["strict_order_proved"] is True
    eligible = layers["task_reveal_discounted_eligible"]
    assert eligible["motion_below_target_lower"] is True
    assert eligible["target_upper_below_landing"] is True
    assert eligible["target_guidance_upper_components"] == pytest.approx(
        {
            "window_kernel_max": 2.078760003085006,
            "base_position_prestrike_max": 0.0,
            "racket_progress_theoretical_telescoping_upper_assumed_at_t0": 0.93,
        }
    )
    assert eligible["target_guidance_lower_at_fine_acceptance"] == pytest.approx(
        1.8515134629687275
    )
    assert eligible["target_guidance_conservative_upper"] == pytest.approx(
        3.008760003085006
    )
    assert eligible["legal_landing_floor"] == pytest.approx(3.332093573945011)
    assert eligible["strict_order_proved"] is True
    assert "runtime ledger" in layers["runtime_validation_requirement"]
    termination = audit["termination_arbitrage_monitor"]
    assert {
        key: termination[key]
        for key in (
            "death_penalty_weight",
            "death_penalty_one_step_income",
            "legal_landing_plus_same_episode_death_floor",
            "legal_landing_plus_same_episode_death_max",
        )
    } == pytest.approx(
        {
            "death_penalty_weight": -10.0,
            "death_penalty_one_step_income": -0.2,
            # death -300->-10 后:8.4 - 0.2 = 8.2(原 8.4 - 6.0 = 2.4)。"打成一次再摔"
            # 从净亏边缘变成明确净正。详见 exp §5.6 偏离记录第 7 条。
            "legal_landing_plus_same_episode_death_floor": 8.2,
            "legal_landing_plus_same_episode_death_max": 13.8,
        }
    )
    assert termination["positive_success_then_fall_net_is_possible"] is True
    assert "post_contact_fall_or_termination" in termination[
        "required_runtime_stratum"
    ]
    observed = audit["frozen_observed_exact_strike_counterfactual"]
    assert math.isclose(observed["v2_window_kernel_income"], 1.8813873682328477)
    assert math.isclose(
        observed["v2_window_kernel_income_at_initial_sigma"], 1.8813873682328477
    )


def test_c211_leaf_has_no_hit_bonus_and_passes_task_valid_discounted_order():
    task = ROOT / "cfg/task/HOPEPingPongActionBallC211VendorV2N1Learnability.yaml"
    resolved, _chain = MODULE._resolved_task_document(task)
    assert resolved["racket"]["action_ball_target_source"] == "direct_ball"
    audit = MODULE.build_audit(
        task,
        observed_errors=(0.6340, 1.9595, math.radians(56.21)),
        task_receipt_path=TASK_RECEIPT,
    )

    assert audit["kind"] == "action_ball_c211_reward_hierarchy_v1"
    assert audit["all_static_hierarchy_checks_pass"] is True
    assert audit["partial_catalog_arithmetic_checks_pass"] is True
    assert audit["system_recipe"]["fine_width_mode"] == "not_applicable_c211"
    assert audit["system_recipe"]["checks"] == {
        "full_body_mimic": True,
        "measured_racket_teacher": True,
        "no_desired_contact_target": True,
        "action_ball_target_mode": True,
        "ball_outcome_enabled": True,
        "table_obstacle_enabled": True,
        "complete_reward_pack": True,
    }
    constants = audit["constants"]
    assert constants["strike_one_shot_peak"] == pytest.approx(4.8)
    assert constants["legal_landing_min"] == pytest.approx(8.4)
    assert constants["legal_landing_max"] == pytest.approx(14.0)
    assert constants["off_table_max"] == pytest.approx(7.0)
    summary = audit["catalog_summary"]
    assert summary["action_count"] == 73
    assert summary["native_catalog_frame_only_imitation_cap_max"] == pytest.approx(
        3.6575
    )
    assert summary["partial_catalog_frame_only_order"] is True
    for row in audit["actions"].values():
        assert (
            row["native_catalog_frame_only_imitation_cap"]
            < row["strike_proximity_one_shot_peak"]
            < row["legal_landing_event_min"]
        )
        assert row["opponent_side_off_table_event_max"] < row[
            "legal_landing_event_min"
        ]
    counterfactual = audit["frozen_paddle_ball_distance_counterfactual"]
    assert counterfactual["income"] > 0.0
    assert counterfactual["signed_derivative_wrt_distance"] < 0.0
    assert counterfactual["nonzero_tail"] is True
    assert counterfactual["velocity_and_face_errors_unused"] is True
    taxonomy = audit["active_reward_taxonomy"]["scientific_groups"]
    assert taxonomy["target"] == {}
    assert taxonomy["strike"] == {
        "c225_strike_ball_paddle_center_proximity": 240.0
    }
    assert "strike_capture_bonus" not in taxonomy["strike"]
    assert taxonomy["outcome"] == {"virtual_landing": 700.0}
    layers = audit["selected_n1_layer_accounting"]
    assert layers["undiscounted_per_swing"]["strict_order"] is True
    assert layers["discounted_strict_order"] is True
    eligible = layers["task_reveal_discounted_eligible"]
    assert eligible == pytest.approx(
        {
            "contact_tick": 92,
            "mimic_cap": 1.7733077595610476,
            "strike_proximity_peak": 1.9040534708257204,
            "legal_landing_floor": 3.332093573945011,
            "strict_order": True,
        }
    )


def test_candidate_keeps_a3_gamma_099_instead_of_changing_gae_horizon():
    assert MODULE.PPO_GAMMA_DEFAULT == pytest.approx(0.99)


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
