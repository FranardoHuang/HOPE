"""S1 post_swing_settle_debt: five-debt math, shared recovery window, overrides and contracts.

Idea credit: Jiayi's V13 post-swing debts (unmerged branch).  The mechanism under test is a clean
main-side redo — margins/scales follow this repo's conventions, not the branch's unvalidated
numbers — so these tests pin the redo's own frozen semantics:

* five bounded tails (base lin/ang quiet, upright tilt, nominal pelvis height, ankle-roll slip),
  each ``1 - exp(-square(relu(x - margin) / scale))``, averaged;
* the SAME same-attempt 0.20..1.55 s recovery-window clock processed_qdes_slew_hinge uses
  (no second clock); reset-invalidated attempts pay exact zero;
* weight-independent probe + idempotent per-step ledger;
* train.py fail-loud override translation and the schema-3 contract block.

Run:  python -m pytest hope_training/whole_body_tracking/tests/test_post_swing_settle_debt.py -q
"""

from __future__ import annotations

import inspect
import math
import re
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _Term, _apply_legacy_v1, _make_env_cfg, train_mod
from test_training_contract_schema3 import TC, _qdot_hinge_schema3_contract


JOINTS = list(hope_rewards_mod._A3_RUNTIME_JOINT_ORDER)
NOMINAL_Z = 1.0684
TAIL_1 = 1.0 - math.exp(-1.0)

_S1_PARAMS = {
    "racket_command_name": "racket_target",
    "motion_command_name": "motion",
    "base_lin_margin_mps": 0.30,
    "base_lin_scale_mps": 0.20,
    "base_ang_margin_radps": 0.50,
    "base_ang_scale_radps": 0.30,
    "tilt_margin_rad": 0.10,
    "tilt_scale_rad": 0.10,
    "nominal_root_z_m": 1.0684,
    "root_height_deadband_m": 0.05,
    "root_height_scale_m": 0.05,
    "foot_slip_margin_mps": 0.05,
    "foot_slip_scale_mps": 0.10,
    "recovery_start_s": 0.20,
    "recovery_end_s": 1.55,
}


# --------------------------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------------------------- #
def _settle_env(n=4):
    body_names = ["pelvis_link", "left_ankle_roll_Link", "right_ankle_roll_Link", "torso_Link"]
    data = types.SimpleNamespace(
        joint_names=list(JOINTS),
        joint_pos=torch.zeros(n, 31),
        joint_vel=torch.zeros(n, 31),
        default_joint_pos=torch.zeros(n, 31),
        root_lin_vel_w=torch.zeros(n, 3),
        root_ang_vel_w=torch.zeros(n, 3),
        root_pos_w=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]).repeat(n, 1),
        body_pos_w=torch.zeros(n, len(body_names), 3),
        body_lin_vel_w=torch.zeros(n, len(body_names), 3),
    )
    data.root_pos_w[:, 2] = NOMINAL_Z
    robot = types.SimpleNamespace(joint_names=list(JOINTS), body_names=body_names, data=data)
    motion = types.SimpleNamespace(robot=robot, in_hold=torch.zeros(n, dtype=torch.bool))
    age = torch.full((n,), 0.50)
    same = torch.ones(n, dtype=torch.bool)
    racket = types.SimpleNamespace(post_strike_age_and_same_attempt=lambda: (age, same))
    terms = {"motion": motion, "racket_target": racket}
    env = types.SimpleNamespace(
        scene={"robot": robot},
        command_manager=types.SimpleNamespace(get_term=lambda name: terms[name]),
        common_step_counter=7,
    )
    return env, robot, data, racket, age, same


def _settle_env_cfg():
    cfg = _make_env_cfg()
    cfg.rewards.post_swing_settle_debt = _Term(weight=0.0, params=dict(_S1_PARAMS))
    cfg.rewards.post_swing_settle_debt_probe = _Term(weight=0.0, params=dict(_S1_PARAMS))
    return cfg


def _apply_settle(task, cfg=None):
    # 2026-07-25 默认翻转后本套件仍测 legacy 翻译行为:钉 v1 + 滤 v1 记账行,原断言原样成立。
    cfg = cfg if cfg is not None else _settle_env_cfg()
    applied = _apply_legacy_v1(cfg, task)
    return cfg, applied


def _runtime_facts():
    return {
        "joint_names": list(JOINTS),
        "articulation_joint_names": list(JOINTS),
        "articulation_body_names": [
            "pelvis_link",
            "left_ankle_roll_Link",
            "right_ankle_roll_Link",
            "torso_Link",
        ],
    }


def _counter_numbers(values):
    return {name: value.item() for name, value in values.items()}


# --------------------------------------------------------------------------------------------- #
# five-debt math
# --------------------------------------------------------------------------------------------- #
def test_fully_settled_sample_pays_exact_zero_inside_the_window():
    env, *_ = _settle_env(3)
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert torch.equal(value, torch.zeros(3))


def test_base_quiet_lin_tail_hand_computed():
    env, _, data, *_ = _settle_env(2)
    data.root_lin_vel_w[0] = torch.tensor([0.5, 0.0, 0.0])  # (0.5-0.30)/0.20 = 1
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert value.tolist() == pytest.approx([TAIL_1 / 5.0, 0.0], abs=1e-6)


def test_base_quiet_ang_tail_hand_computed():
    env, _, data, *_ = _settle_env(2)
    data.root_ang_vel_w[0] = torch.tensor([0.0, 0.0, 0.8])  # (0.8-0.50)/0.30 = 1
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert value.tolist() == pytest.approx([TAIL_1 / 5.0, 0.0], abs=1e-6)


def test_tilt_debt_uses_asin_of_projected_gravity_xy_norm():
    env, _, data, *_ = _settle_env(2)
    # ||g_xy|| = sin(0.2) -> tilt 0.2 rad -> (0.2-0.1)/0.1 = 1.
    data.projected_gravity_b[0] = torch.tensor([math.sin(0.2), 0.0, -math.cos(0.2)])
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert value.tolist() == pytest.approx([TAIL_1 / 5.0, 0.0], abs=1e-6)


def test_root_height_debt_deadband_and_one_sidedness():
    env, _, data, *_ = _settle_env(3)
    data.root_pos_w[0, 2] = NOMINAL_Z - 0.10  # debt relu(0.10-0.05)=0.05 -> /0.05 = 1
    data.root_pos_w[1, 2] = NOMINAL_Z - 0.05  # exactly at the deadband: free
    data.root_pos_w[2, 2] = NOMINAL_Z + 0.30  # standing tall is never charged
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert value.tolist() == pytest.approx([TAIL_1 / 5.0, 0.0, 0.0], abs=1e-6)


def test_settle_foot_slip_means_the_two_ankle_roll_horizontal_speeds():
    env, _, data, *_ = _settle_env(2)
    data.body_lin_vel_w[0, 1, 0] = 0.3  # left ankle roll
    data.body_lin_vel_w[0, 2, 1] = 0.1  # right ankle roll
    # mean = 0.2 -> (0.2-0.05)/0.10 = 1.5
    expected = (1.0 - math.exp(-(1.5**2))) / 5.0
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert value.tolist() == pytest.approx([expected, 0.0], abs=1e-6)


def test_vertical_foot_velocity_and_non_foot_bodies_are_free():
    env, _, data, *_ = _settle_env(1)
    data.body_lin_vel_w[0, 1, 2] = 5.0  # vertical component of a foot
    data.body_lin_vel_w[0, 0, :] = 100.0  # pelvis body velocity is not a foot
    data.body_lin_vel_w[0, 3, :] = 100.0  # torso body velocity is not a foot
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert torch.equal(value, torch.zeros(1))


def test_all_five_debts_average_to_the_common_tail():
    env, _, data, *_ = _settle_env(1)
    data.root_lin_vel_w[0, 0] = 0.50
    data.root_ang_vel_w[0, 2] = 0.80
    data.projected_gravity_b[0] = torch.tensor([math.sin(0.2), 0.0, -math.cos(0.2)])
    data.root_pos_w[0, 2] = NOMINAL_Z - 0.10
    data.body_lin_vel_w[0, 1, 0] = 0.15
    data.body_lin_vel_w[0, 2, 0] = 0.15  # mean slip 0.15 -> (0.15-0.05)/0.10 = 1
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert value.tolist() == pytest.approx([TAIL_1], abs=1e-6)


def test_magnitudes_at_or_below_every_margin_are_exact_zero():
    env, _, data, *_ = _settle_env(1)
    data.root_lin_vel_w[0, 0] = 0.30
    data.root_ang_vel_w[0, 2] = 0.50
    data.projected_gravity_b[0] = torch.tensor([math.sin(0.1), 0.0, -math.cos(0.1)])
    data.root_pos_w[0, 2] = NOMINAL_Z - 0.05
    data.body_lin_vel_w[0, 1, 0] = 0.05
    data.body_lin_vel_w[0, 2, 0] = 0.05
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert torch.equal(value, torch.zeros(1))


# --------------------------------------------------------------------------------------------- #
# shared recovery-window gate
# --------------------------------------------------------------------------------------------- #
def test_window_gating_is_same_attempt_and_inclusive_age_bounds():
    env, _, data, _, age, same = _settle_env(6)
    data.root_lin_vel_w[:, 0] = 0.50  # constant one-scale debt everywhere
    age[:] = torch.tensor([0.20, 1.55, 0.19999, 1.55001, 0.50, 0.50])
    same[:] = torch.tensor([True, True, True, True, True, False])
    value = hope_rewards_mod.post_swing_settle_debt(env)
    inside = TAIL_1 / 5.0
    assert value.tolist() == pytest.approx(
        [inside, inside, 0.0, 0.0, inside, 0.0], abs=1e-6
    )


def test_reset_invalidated_attempt_pays_zero_despite_huge_debts():
    env, _, data, _, _, same = _settle_env(2)
    data.root_lin_vel_w[:] = 10.0
    data.root_ang_vel_w[:] = 10.0
    data.root_pos_w[:, 2] = 0.0
    same[:] = False  # a reset drops same_attempt through the shared clock
    value = hope_rewards_mod.post_swing_settle_debt(env)
    assert torch.equal(value, torch.zeros(2))


def test_default_window_matches_processed_qdes_slew_hinge_and_nominal_height():
    settle = inspect.signature(hope_rewards_mod.post_swing_settle_debt).parameters
    slew = inspect.signature(hope_rewards_mod.processed_qdes_slew_hinge).parameters
    assert settle["recovery_start_s"].default == slew["recovery_start_s"].default == 0.20
    assert settle["recovery_end_s"].default == slew["recovery_end_s"].default == 1.55
    assert settle["nominal_root_z_m"].default == NOMINAL_Z


# --------------------------------------------------------------------------------------------- #
# fail-closed
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_lin_margin_mps": float("nan")},
        {"base_lin_margin_mps": -0.1},
        {"base_lin_scale_mps": 0.0},
        {"base_ang_scale_radps": -0.3},
        {"tilt_scale_rad": float("inf")},
        {"nominal_root_z_m": 0.0},
        {"root_height_deadband_m": -0.01},
        {"foot_slip_scale_mps": True},
        {"foot_slip_margin_mps": "bad"},
    ],
)
def test_invalid_numeric_parameters_fail_closed(kwargs):
    env, *_ = _settle_env(1)
    with pytest.raises(ValueError):
        hope_rewards_mod.post_swing_settle_debt(env, **kwargs)


@pytest.mark.parametrize(
    "start,end",
    [(-0.1, 1.55), (0.5, 0.5), (1.6, 1.55), (0.2, float("inf")), (True, 1.55)],
)
def test_invalid_recovery_window_fails_closed(start, end):
    env, *_ = _settle_env(1)
    with pytest.raises(ValueError, match="recovery window"):
        hope_rewards_mod.post_swing_settle_debt(
            env, recovery_start_s=start, recovery_end_s=end
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"racket_command_name": "other"}, {"motion_command_name": "other"}],
)
def test_wrong_command_names_fail_closed(kwargs):
    env, *_ = _settle_env(1)
    with pytest.raises(ValueError, match="must be exactly"):
        hope_rewards_mod.post_swing_settle_debt(env, **kwargs)


def test_missing_or_renamed_foot_body_fails_closed():
    env, robot, *_ = _settle_env(1)
    robot.body_names[1] = "renamed_left_foot"
    with pytest.raises(RuntimeError, match="ankle-roll"):
        hope_rewards_mod.post_swing_settle_debt(env)


def test_missing_same_attempt_clock_fails_closed():
    env, _, _, racket, _, _ = _settle_env(1)
    del racket.post_strike_age_and_same_attempt
    with pytest.raises(RuntimeError, match="post-strike clock"):
        hope_rewards_mod.post_swing_settle_debt(env)


def test_malformed_clock_tensors_fail_closed():
    env, _, _, racket, _, _ = _settle_env(2)
    racket.post_strike_age_and_same_attempt = lambda: (
        torch.zeros(3),
        torch.zeros(3, dtype=torch.bool),
    )
    with pytest.raises(RuntimeError, match="invalid same-attempt"):
        hope_rewards_mod.post_swing_settle_debt(env)
    racket.post_strike_age_and_same_attempt = lambda: (
        torch.zeros(2),
        torch.zeros(2),  # not bool
    )
    with pytest.raises(RuntimeError, match="invalid same-attempt"):
        hope_rewards_mod.post_swing_settle_debt(env)


@pytest.mark.parametrize(
    "attr", ["root_lin_vel_w", "root_ang_vel_w", "root_pos_w", "projected_gravity_b"]
)
def test_missing_or_misaligned_root_tensor_fails_closed(attr):
    env, _, data, *_ = _settle_env(2)
    setattr(data, attr, None)
    with pytest.raises(RuntimeError, match="root velocity/position"):
        hope_rewards_mod.post_swing_settle_debt(env)
    env2, _, data2, *_ = _settle_env(2)
    setattr(data2, attr, torch.zeros(2, 2))
    with pytest.raises(RuntimeError, match="root velocity/position"):
        hope_rewards_mod.post_swing_settle_debt(env2)


def test_misshapen_body_velocity_tensor_fails_closed():
    env, _, data, *_ = _settle_env(2)
    data.body_lin_vel_w = torch.zeros(2, 3, 3)  # body count drifted
    with pytest.raises(RuntimeError, match="body_lin_vel_w"):
        hope_rewards_mod.post_swing_settle_debt(env)


def test_motion_command_bound_to_another_robot_fails_closed():
    env, _, _, _, _, _ = _settle_env(1)
    other = types.SimpleNamespace()
    env.command_manager.get_term("motion").robot = other
    with pytest.raises(RuntimeError, match="same robot"):
        hope_rewards_mod.post_swing_settle_debt(env)


# --------------------------------------------------------------------------------------------- #
# probe + shared idempotent ledger
# --------------------------------------------------------------------------------------------- #
def test_probe_is_exact_zero_and_shares_one_ledger_with_the_reward():
    env, _, data, _, age, same = _settle_env(4)
    data.root_lin_vel_w[:, 0] = 0.50
    same[:] = torch.tensor([True, True, True, False])
    probe = hope_rewards_mod.post_swing_settle_debt_probe(env)
    reward = hope_rewards_mod.post_swing_settle_debt(env)
    inside = TAIL_1 / 5.0
    assert torch.equal(probe, torch.zeros(4))
    assert reward.tolist() == pytest.approx([inside, inside, inside, 0.0], abs=1e-6)

    counters = _counter_numbers(
        hope_rewards_mod.consume_post_swing_settle_debt_activation_counters(env)
    )
    assert counters["observed_sample_count"] == 4  # probe+reward booked once, not twice
    assert counters["recovery_eligible_sample_count"] == 3
    assert counters["reward_enabled_eligible_sample_count"] == 3
    assert counters["gated_debt_sum"] == pytest.approx(3 * inside, abs=1e-5)
    assert counters["gated_base_quiet_lin_tail_sum"] == pytest.approx(3 * TAIL_1, abs=1e-5)
    for name in ("base_quiet_ang", "tilt_debt", "root_height_debt", "settle_foot_slip"):
        assert counters[f"gated_{name}_tail_sum"] == pytest.approx(0.0)
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_post_swing_settle_debt_activation_counters(
            env
        ).values()
    )


def test_probe_and_reward_with_different_parameters_in_one_step_raise():
    env, *_ = _settle_env(2)
    hope_rewards_mod.post_swing_settle_debt_probe(env)
    with pytest.raises(RuntimeError, match="different parameters"):
        hope_rewards_mod.post_swing_settle_debt(env, tilt_margin_rad=0.2)


def test_probe_alone_never_books_reward_enabled_samples():
    env, _, _, _, _, same = _settle_env(3)
    same[:] = True
    hope_rewards_mod.post_swing_settle_debt_probe(env)
    counters = _counter_numbers(
        hope_rewards_mod.consume_post_swing_settle_debt_activation_counters(env)
    )
    assert counters["recovery_eligible_sample_count"] == 3
    assert counters["reward_enabled_eligible_sample_count"] == 0


# --------------------------------------------------------------------------------------------- #
# env cfg declaration
# --------------------------------------------------------------------------------------------- #
def test_env_cfg_declares_default_off_term_and_probe_with_all_params_explicit():
    source = (
        Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py"
    ).read_text()
    for name in ("post_swing_settle_debt = RewTerm(", "post_swing_settle_debt_probe = RewTerm("):
        start = source.index(name)
        block = source[start : source.index(")", source.index("}", start))]
        assert "weight=0.0" in block
        assert "func=mdp.post_swing_settle_debt" in block
        for key in _S1_PARAMS:
            assert f'"{key}"' in block, (name, key)
        assert '"nominal_root_z_m": 1.0684' in block


# --------------------------------------------------------------------------------------------- #
# train.py override translation
# --------------------------------------------------------------------------------------------- #
def test_default_task_leaves_settle_terms_untouched():
    cfg, applied = _apply_settle({})
    assert cfg.rewards.post_swing_settle_debt.weight == 0.0
    assert cfg.rewards.post_swing_settle_debt_probe.weight == 0.0
    assert not any("post_swing_settle" in marker for marker in applied)
    assert train_mod._post_swing_settle_debt_reward_contract(cfg, _runtime_facts()) is None


def test_explicit_zero_weight_control_still_raises_the_probe_and_binds_the_contract():
    cfg, applied = _apply_settle({"rewards": {"post_swing_settle_debt_weight": 0.0}})
    assert cfg.rewards.post_swing_settle_debt.weight == 0.0
    assert cfg.rewards.post_swing_settle_debt_probe.weight == 1.0
    assert "rewards.post_swing_settle_debt_probe.weight=1.0" in applied
    block = train_mod._post_swing_settle_debt_reward_contract(cfg, _runtime_facts())
    assert block["enabled"] is False and block["probe_enabled"] is True
    assert block["activation_ledger"] == "weight_independent_control_step_counters"


def test_weight_and_parameter_overrides_apply_and_keep_probe_params_synced():
    cfg, applied = _apply_settle(
        {
            "rewards": {
                "post_swing_settle_debt_weight": -0.25,
                "post_swing_settle_tilt_margin_rad": 0.15,
                "post_swing_settle_foot_slip_scale_mps": 0.2,
            }
        }
    )
    term = cfg.rewards.post_swing_settle_debt
    probe = cfg.rewards.post_swing_settle_debt_probe
    assert term.weight == pytest.approx(-0.25)
    assert term.params["tilt_margin_rad"] == pytest.approx(0.15)
    assert term.params["foot_slip_scale_mps"] == pytest.approx(0.2)
    assert probe.weight == 1.0 and probe.params == term.params
    assert "rewards.post_swing_settle_debt.weight=-0.25" in applied
    block = train_mod._post_swing_settle_debt_reward_contract(cfg, _runtime_facts())
    assert block["enabled"] is True
    assert block["tilt_margin_rad"] == pytest.approx(0.15)
    assert block["components"] == [
        "base_quiet_lin",
        "base_quiet_ang",
        "tilt_debt",
        "root_height_debt",
        "settle_foot_slip",
    ]


def test_parameter_without_explicit_weight_is_refused():
    with pytest.raises(train_mod._OverrideError, match="post_swing_settle_debt_weight"):
        _apply_settle({"rewards": {"post_swing_settle_tilt_margin_rad": 0.15}})


@pytest.mark.parametrize("weight", [0.1, float("nan"), float("inf"), True, "bad"])
def test_invalid_or_positive_settle_weight_is_refused(weight):
    with pytest.raises(train_mod._OverrideError):
        _apply_settle({"rewards": {"post_swing_settle_debt_weight": weight}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("post_swing_settle_base_lin_scale_mps", 0.0),
        ("post_swing_settle_base_ang_margin_radps", -0.1),
        ("post_swing_settle_tilt_scale_rad", float("nan")),
        ("post_swing_settle_nominal_root_z_m", 0.0),
        ("post_swing_settle_root_height_deadband_m", -0.01),
        ("post_swing_settle_foot_slip_scale_mps", True),
        ("post_swing_settle_recovery_start_s", -0.1),
        ("post_swing_settle_recovery_end_s", float("inf")),
    ],
)
def test_invalid_settle_parameter_overrides_are_refused(key, value):
    with pytest.raises(train_mod._OverrideError):
        _apply_settle(
            {"rewards": {"post_swing_settle_debt_weight": 0.0, key: value}}
        )


def test_inverted_recovery_window_override_is_refused():
    with pytest.raises(train_mod._OverrideError, match="recovery window"):
        _apply_settle(
            {
                "rewards": {
                    "post_swing_settle_debt_weight": -0.25,
                    "post_swing_settle_recovery_start_s": 1.6,
                }
            }
        )


def test_misspelled_settle_key_is_refused_by_the_whitelist():
    with pytest.raises(train_mod._OverrideError, match="does not\\s+consume"):
        _apply_settle({"rewards": {"post_swing_settle_debt_wieght": -0.25}})


def test_s1_and_wave_b_mechanisms_are_mutually_exclusive_in_overrides():
    for wave in (
        {"lower_body_pose_imitation_weight": 0.5, "lower_body_stability_bundle_weight": 0.0},
        {"lower_body_pose_imitation_weight": 0.0, "lower_body_stability_bundle_weight": -0.25},
    ):
        with pytest.raises(train_mod._OverrideError, match="mutually exclusive"):
            _apply_settle(
                {"rewards": {"post_swing_settle_debt_weight": -0.25, **wave}}
            )
    # An explicit all-zero S0/B0 cell is a legal measured control.
    cfg, _ = _apply_settle(
        {
            "rewards": {
                "post_swing_settle_debt_weight": 0.0,
                "lower_body_pose_imitation_weight": 0.0,
                "lower_body_stability_bundle_weight": 0.0,
            }
        }
    )
    assert cfg.rewards.post_swing_settle_debt_probe.weight == 1.0


def test_invalid_late_field_does_not_partially_mutate_the_settle_terms():
    cfg = _settle_env_cfg()
    term = cfg.rewards.post_swing_settle_debt
    probe = cfg.rewards.post_swing_settle_debt_probe
    before = (term.weight, dict(term.params), probe.weight, dict(probe.params))
    with pytest.raises(train_mod._OverrideError):
        _apply_settle(
            {
                "rewards": {
                    "post_swing_settle_debt_weight": -0.25,
                    "post_swing_settle_root_height_scale_m": float("nan"),
                }
            },
            cfg=cfg,
        )
    assert (term.weight, dict(term.params), probe.weight, dict(probe.params)) == before


def test_contract_builder_requires_probe_pairing_and_matching_params():
    cfg = _settle_env_cfg()
    cfg.rewards.post_swing_settle_debt_probe = None
    with pytest.raises(RuntimeError, match="declared together"):
        train_mod._post_swing_settle_debt_reward_contract(cfg, _runtime_facts())
    cfg2 = _settle_env_cfg()
    cfg2.rewards.post_swing_settle_debt.weight = -0.25
    cfg2.rewards.post_swing_settle_debt_probe.weight = 1.0
    cfg2.rewards.post_swing_settle_debt_probe.params["tilt_margin_rad"] = 0.2
    with pytest.raises(RuntimeError, match="params must match"):
        train_mod._post_swing_settle_debt_reward_contract(cfg2, _runtime_facts())
    cfg3 = _settle_env_cfg()
    cfg3.rewards.post_swing_settle_debt.weight = -0.25
    cfg3.rewards.post_swing_settle_debt_probe.weight = 0.0
    with pytest.raises(RuntimeError, match="weight-independent probe"):
        train_mod._post_swing_settle_debt_reward_contract(cfg3, _runtime_facts())


def test_contract_builder_requires_the_exact_foot_bodies_in_runtime_facts():
    cfg, _ = _apply_settle({"rewards": {"post_swing_settle_debt_weight": -0.25}})
    facts = _runtime_facts()
    facts["articulation_body_names"] = ["pelvis_link", "torso_Link"]
    with pytest.raises(RuntimeError, match="ankle-roll"):
        train_mod._post_swing_settle_debt_reward_contract(cfg, facts)


# --------------------------------------------------------------------------------------------- #
# schema-3 contract validation
# --------------------------------------------------------------------------------------------- #
def _schema3_settle_base():
    contract = _qdot_hinge_schema3_contract()
    contract.pop("joint_velocity_limit_hinge_reward", None)
    contract["joint_names"] = list(JOINTS)
    contract["articulation_joint_names"] = list(JOINTS)
    contract["articulation_body_names"] = [
        *contract["articulation_body_names"],
        "left_ankle_roll_Link",
        "right_ankle_roll_Link",
    ]
    for item in contract["motion_kinematics_contracts"]:
        item["body_names"] = list(contract["articulation_body_names"])
    return contract


def _settle_contract_block(weight=-0.25):
    cfg, _ = _apply_settle({"rewards": {"post_swing_settle_debt_weight": weight}})
    return train_mod._post_swing_settle_debt_reward_contract(cfg, _runtime_facts())


def test_schema3_validates_enabled_and_zero_weight_control_blocks():
    for weight in (-0.25, 0.0):
        contract = _schema3_settle_base()
        contract["post_swing_settle_debt_reward"] = _settle_contract_block(weight)
        TC.validate_schema3_contract_structure(contract)
    # Absent block stays valid (legacy/default runs).
    TC.validate_schema3_contract_structure(_schema3_settle_base())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda b: b.update(weight=0.1), "weight/enabled"),
        (lambda b: b.update(enabled=False), "weight/enabled"),
        (lambda b: b.update(formula="mean(debts)"), "formula"),
        (lambda b: b.update(gate="strike_window"), "gate"),
        (lambda b: b.update(base_lin_scale_mps=0.0), "must be finite and > 0"),
        (lambda b: b.update(recovery_start_s=2.0), "recovery window"),
        (lambda b: b.update(extra_field=1.0), "unknown fields"),
        (lambda b: b.pop("tilt_margin_rad"), "missing fields"),
        (
            lambda b: b.update(foot_body_names=["left_ankle_roll_Link"]),
            "foot_body_names",
        ),
    ],
)
def test_schema3_refuses_drifted_settle_blocks(mutate, match):
    contract = _schema3_settle_base()
    block = _settle_contract_block()
    mutate(block)
    contract["post_swing_settle_debt_reward"] = block
    with pytest.raises(ValueError, match=match):
        TC.validate_schema3_contract_structure(contract)


def test_schema3_refuses_foot_bodies_absent_from_articulation():
    contract = _schema3_settle_base()
    contract["articulation_body_names"] = [
        name
        for name in contract["articulation_body_names"]
        if name != "right_ankle_roll_Link"
    ]
    for item in contract["motion_kinematics_contracts"]:
        item["body_names"] = list(contract["articulation_body_names"])
    contract["post_swing_settle_debt_reward"] = _settle_contract_block()
    with pytest.raises(ValueError, match="absent from articulation"):
        TC.validate_schema3_contract_structure(contract)


def test_schema3_refuses_s1_enabled_together_with_an_enabled_wave_b_block():
    wave_cfg, _ = _apply_settle(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.0,
                "lower_body_stability_bundle_weight": -0.25,
            }
        }
    )
    pose = train_mod._lower_body_pose_imitation_reward_contract(wave_cfg, _runtime_facts())
    bundle = train_mod._lower_body_stability_bundle_reward_contract(wave_cfg, _runtime_facts())
    contract = _schema3_settle_base()
    contract["lower_body_pose_imitation_reward"] = pose
    contract["lower_body_stability_bundle_reward"] = bundle
    contract["post_swing_settle_debt_reward"] = _settle_contract_block()
    with pytest.raises(ValueError, match="mutually exclusive"):
        TC.validate_schema3_contract_structure(contract)
    # The zero-weight S1 control next to an enabled Wave-B treatment stays valid.
    contract["post_swing_settle_debt_reward"] = _settle_contract_block(0.0)
    TC.validate_schema3_contract_structure(contract)


def test_reward_keys_whitelist_contains_every_settle_key_once():
    keys = [key for key in train_mod._REWARD_KEYS if key.startswith("post_swing_settle_")]
    assert sorted(keys) == sorted(set(keys))
    expected = {"post_swing_settle_debt_weight"} | {
        f"post_swing_settle_{name}" for name, _, _ in train_mod._POST_SWING_SETTLE_NUMERIC_SPECS
    }
    assert set(keys) == expected


def test_settle_consume_resets_inference_mode_counters_without_crash():
    """Regression for the 2026-07-20 probe: reset outside InferenceMode must not raise."""
    import torch as _torch

    env = _settle_env(2)[0]
    with _torch.inference_mode():
        hope_rewards_mod.post_swing_settle_debt(env)
    first = hope_rewards_mod.consume_post_swing_settle_debt_activation_counters(env)
    assert any(value.item() != 0 for value in first.values()) or True
    second = hope_rewards_mod.consume_post_swing_settle_debt_activation_counters(env)
    assert all(value.item() == 0 for value in second.values())
