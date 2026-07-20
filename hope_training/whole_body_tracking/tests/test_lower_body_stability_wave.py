"""Wave-B lower-body mechanisms: exact phase/joint math, overrides, ledgers and contracts."""

from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
import torch

from test_reward_flags_mdp import hope_rewards_mod
from test_reward_flags_overrides import _apply, _make_env_cfg, train_mod
from test_training_contract_schema3 import TC, _qdot_hinge_schema3_contract


JOINTS = list(hope_rewards_mod._A3_RUNTIME_JOINT_ORDER)
LEG_IDS = [i for i, name in enumerate(JOINTS) if name in hope_rewards_mod._LOWER_BODY_LEG_JOINT_NAMES]


def _wave_env(n=4):
    q = torch.zeros(n, 31)
    qd = torch.zeros_like(q)
    default_q = torch.zeros_like(q)
    body_names = ["pelvis_link", "left_ankle_roll_Link", "right_ankle_roll_Link", "torso_Link"]
    body_pos = torch.zeros(n, len(body_names), 3)
    body_pos[:, 1, 1] = 0.15
    body_pos[:, 2, 1] = -0.15
    robot = types.SimpleNamespace(
        joint_names=list(JOINTS),
        body_names=body_names,
        data=types.SimpleNamespace(
            joint_names=list(JOINTS),
            joint_pos=q,
            joint_vel=qd,
            default_joint_pos=default_q,
            body_pos_w=body_pos,
        ),
    )
    reference = torch.zeros_like(q)
    motion = types.SimpleNamespace(
        robot=robot,
        joint_pos=reference,
        motion=types.SimpleNamespace(joint_pos=torch.zeros(20, 31)),
        in_hold=torch.zeros(n, dtype=torch.bool),
    )
    age = torch.zeros(n)
    same = torch.zeros(n, dtype=torch.bool)
    racket = types.SimpleNamespace(
        time_to_strike=torch.full((n,), 0.1),
        pre_strike=torch.ones(n, dtype=torch.bool),
        base_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        post_strike_age_and_same_attempt=lambda: (age, same),
    )
    terms = {"motion": motion, "racket_target": racket}
    env = types.SimpleNamespace(
        scene={"robot": robot},
        command_manager=types.SimpleNamespace(get_term=lambda name: terms[name]),
        common_step_counter=7,
    )
    return env, robot, motion, racket, age, same


def _counter_numbers(values):
    return {name: value.item() for name, value in values.items()}


def test_b1_pose_imitation_phase_boundaries_exact_12_joints_and_ledger():
    env, robot, motion, racket, age, same = _wave_env(4)
    racket.time_to_strike[:] = torch.tensor([0.30, 0.30001, -0.2, -0.2])
    racket.pre_strike[:] = torch.tensor([True, True, False, False])
    age[:] = torch.tensor([0.0, 0.0, 0.40, 0.40])
    same[:] = torch.tensor([False, False, True, False])
    motion.joint_pos[:, LEG_IDS] = 1.0
    # Arm/head/waist errors are deliberately enormous and must never enter the 12-leg mean.
    non_leg = [index for index in range(31) if index not in LEG_IDS]
    robot.data.joint_pos[:, non_leg] = 1000.0

    probe = hope_rewards_mod.lower_body_pose_imitation_probe(env, std=1.0)
    reward = hope_rewards_mod.lower_body_pose_imitation(env, std=1.0)
    expected = math.exp(-1.0)
    assert torch.equal(probe, torch.zeros(4))
    assert reward.tolist() == pytest.approx([expected, 0.0, expected, 0.0])

    counters = _counter_numbers(
        hope_rewards_mod.consume_lower_body_wave_activation_counters(env)
    )
    assert counters["pose/observed_sample_count"] == 4
    assert counters["pose/support_eligible_sample_count"] == 2
    assert counters["pose/reward_enabled_eligible_sample_count"] == 2
    assert counters["pose/gated_kernel_sum"] == pytest.approx(2.0 * expected)
    assert counters["pose/gated_joint_abs_error_mean_sum"] == pytest.approx(2.0)
    assert counters["pose/gated_reference_motion_l1_mean_sum"] == pytest.approx(2.0)
    assert all(
        value.item() == 0
        for value in hope_rewards_mod.consume_lower_body_wave_activation_counters(env).values()
    )


def test_b1_gate_excludes_hold_and_is_not_exact_hit_success_conditioned():
    env, _, motion, racket, age, same = _wave_env(3)
    motion.joint_pos[:, LEG_IDS] = 0.5
    racket.time_to_strike[:] = torch.tensor([0.1, -0.1, -0.1])
    racket.pre_strike[:] = torch.tensor([True, False, False])
    motion.in_hold[:] = torch.tensor([True, False, False])
    age[:] = torch.tensor([0.0, 0.2, 0.2])
    same[:] = torch.tensor([False, True, False])
    # The fake exposes no hit/completion flag at all; phase opportunity + same-attempt clock is enough.
    value = hope_rewards_mod.lower_body_pose_imitation(env, std=0.5)
    assert value.tolist() == pytest.approx([0.0, math.exp(-1.0), 0.0])


def test_b2_bundle_components_base_yaw_stance_and_no_arm_velocity():
    env, robot, motion, racket, _, _ = _wave_env(3)
    # A huge motion-reference discrepancy must be irrelevant to reference-free B2.
    motion.joint_pos[:] = 1000.0
    motion.motion.joint_pos = torch.zeros(20, 7)  # malformed reference is also irrelevant to B2.
    robot.data.joint_vel[:, LEG_IDS] = 1.5
    robot.data.joint_vel[:, :19] = 100.0  # arms/waist/head are outside the 12-leg tail.

    # env0: yaw=0, signed stance 0.17 (one 0.05-m scale below 0.22).
    robot.data.body_pos_w[0, 1, :2] = torch.tensor([0.0, 0.085])
    robot.data.body_pos_w[0, 2, :2] = torch.tensor([0.0, -0.085])
    # env1: yaw=+90deg, the base-lateral direction is world -X; signed width is exactly 0.22.
    half = math.sqrt(0.5)
    racket.base_quat_w[1] = torch.tensor([half, 0.0, 0.0, half])
    robot.data.body_pos_w[1, 1, :2] = torch.tensor([-0.11, 0.0])
    robot.data.body_pos_w[1, 2, :2] = torch.tensor([0.11, 0.0])
    # env2: comfortably wide.
    robot.data.body_pos_w[2, 1, 1] = 0.15
    robot.data.body_pos_w[2, 2, 1] = -0.15

    probe = hope_rewards_mod.lower_body_stability_bundle_probe(env)
    reward = hope_rewards_mod.lower_body_stability_bundle(env)
    tail = 1.0 - math.exp(-1.0)
    assert torch.equal(probe, torch.zeros(3))
    assert reward.tolist() == pytest.approx(
        [tail, tail / 2.0, tail / 2.0], abs=1.0e-6
    )

    counters = _counter_numbers(
        hope_rewards_mod.consume_lower_body_wave_activation_counters(env)
    )
    assert counters["bundle/observed_sample_count"] == 3
    assert counters["bundle/support_eligible_sample_count"] == 3
    assert counters["bundle/reward_enabled_eligible_sample_count"] == 3
    assert counters["bundle/narrow_or_crossed_sample_count"] == 1
    assert counters["bundle/gated_stance_tail_sum"] == pytest.approx(tail, abs=1.0e-6)
    assert counters["bundle/gated_leg_velocity_tail_sum"] == pytest.approx(3 * tail)


def test_b2_uses_the_same_inclusive_pre_and_reset_aware_post_boundaries():
    env, robot, _, racket, age, same = _wave_env(5)
    robot.data.joint_vel[:, LEG_IDS] = 1.5
    racket.time_to_strike[:] = torch.tensor([0.30, 0.30001, -0.1, -0.1, -0.1])
    racket.pre_strike[:] = torch.tensor([True, True, False, False, False])
    age[:] = torch.tensor([0.0, 0.0, 0.40, 0.40001, 0.20])
    same[:] = torch.tensor([False, False, True, True, False])

    leg_tail_half = (1.0 - math.exp(-1.0)) / 2.0
    value = hope_rewards_mod.lower_body_stability_bundle(env)
    assert value.tolist() == pytest.approx(
        [leg_tail_half, 0.0, leg_tail_half, 0.0, 0.0], abs=1.0e-6
    )


@pytest.mark.parametrize("mutation", ["runtime_order", "reference_width", "missing_foot"])
def test_wave_mechanisms_fail_closed_on_joint_and_body_contract_drift(mutation):
    env, robot, motion, _, _, _ = _wave_env(2)
    if mutation == "runtime_order":
        # A permutation of the same 31 names is legal (live articulation is
        # breadth-first); real drift is a wrong or duplicated name.
        robot.data.joint_names[0] = "not_a_joint"
        with pytest.raises(RuntimeError, match="exact 31-joint"):
            hope_rewards_mod.lower_body_pose_imitation(env)
        robot.data.joint_names[0] = robot.data.joint_names[1]
        with pytest.raises(RuntimeError, match="exact 31-joint"):
            hope_rewards_mod.lower_body_pose_imitation(env)
    elif mutation == "reference_width":
        motion.motion.joint_pos = torch.zeros(20, 30)
        with pytest.raises(RuntimeError, match="31-column"):
            hope_rewards_mod.lower_body_pose_imitation(env)
    else:
        robot.body_names[1] = "renamed_left_foot"
        with pytest.raises(RuntimeError, match="ankle-roll bodies"):
            hope_rewards_mod.lower_body_stability_bundle(env)


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


def test_default_is_byte_contract_compatible_and_explicit_b0_is_measured():
    default, applied = _apply({})
    assert default.rewards.lower_body_pose_imitation.weight == 0.0
    assert default.rewards.lower_body_pose_imitation_probe.weight == 0.0
    assert default.rewards.lower_body_stability_bundle.weight == 0.0
    assert default.rewards.lower_body_stability_bundle_probe.weight == 0.0
    assert train_mod._lower_body_pose_imitation_reward_contract(default, _runtime_facts()) is None
    assert train_mod._lower_body_stability_bundle_reward_contract(default, _runtime_facts()) is None
    assert not any("lower_body_" in marker for marker in applied)

    b0, _ = _apply(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.0,
                "lower_body_stability_bundle_weight": 0.0,
            }
        }
    )
    assert b0.rewards.lower_body_pose_imitation_probe.weight == 1.0
    assert b0.rewards.lower_body_stability_bundle_probe.weight == 1.0
    pose = train_mod._lower_body_pose_imitation_reward_contract(b0, _runtime_facts())
    bundle = train_mod._lower_body_stability_bundle_reward_contract(b0, _runtime_facts())
    assert pose["enabled"] is False and bundle["enabled"] is False
    assert pose["probe_enabled"] is True and bundle["probe_enabled"] is True
    assert pose["activation_ledger"] == "weight_independent_control_step_counters"
    assert bundle["activation_ledger"] == "weight_independent_control_step_counters"
    assert pose["joint_names"] == JOINTS[-12:]
    assert bundle["uses_motion_reference"] is False
    assert bundle["components"] == [
        "stance_width_lower_hinge",
        "twelve_leg_realized_qdot_tail",
    ]


def test_b1_b2_overrides_are_mutually_exclusive_and_leave_baseline_foot_term_unchanged():
    baseline = _make_env_cfg().rewards.foot_orientation.weight
    b1, _ = _apply(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.5,
                "lower_body_pose_imitation_std": 0.30,
                "lower_body_stability_bundle_weight": 0.0,
            }
        }
    )
    assert b1.rewards.lower_body_pose_imitation.weight == pytest.approx(0.5)
    assert b1.rewards.lower_body_stability_bundle.weight == 0.0
    assert b1.rewards.foot_orientation.weight == baseline

    b2, _ = _apply(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.0,
                "lower_body_stability_bundle_weight": -0.25,
                "lower_body_stability_min_stance_width_m": 0.22,
            }
        }
    )
    assert b2.rewards.lower_body_pose_imitation.weight == 0.0
    assert b2.rewards.lower_body_stability_bundle.weight == pytest.approx(-0.25)
    assert b2.rewards.foot_orientation.weight == baseline

    with pytest.raises(train_mod._OverrideError, match="mutually exclusive"):
        _apply(
            {
                "rewards": {
                    "lower_body_pose_imitation_weight": 0.5,
                    "lower_body_stability_bundle_weight": -0.25,
                }
            }
        )


@pytest.mark.parametrize(
    "rewards",
    [
        {"lower_body_pose_imitation_weight": 0.5},
        {"lower_body_stability_bundle_weight": -0.25},
        {"lower_body_pose_imitation_std": 0.30},
        {"lower_body_stability_min_stance_width_m": 0.22},
    ],
)
def test_wave_cells_require_both_weights_so_both_probes_and_contracts_are_bound(rewards):
    with pytest.raises(train_mod._OverrideError, match="requires both"):
        _apply({"rewards": rewards})


def test_invalid_late_wave_field_does_not_partially_mutate_either_mechanism():
    cfg = _make_env_cfg()
    pose = cfg.rewards.lower_body_pose_imitation
    pose_probe = cfg.rewards.lower_body_pose_imitation_probe
    bundle = cfg.rewards.lower_body_stability_bundle
    bundle_probe = cfg.rewards.lower_body_stability_bundle_probe
    before = (
        pose.weight,
        dict(pose.params),
        pose_probe.weight,
        dict(pose_probe.params),
        bundle.weight,
        dict(bundle.params),
        bundle_probe.weight,
        dict(bundle_probe.params),
    )
    with pytest.raises(train_mod._OverrideError):
        _apply(
            {
                "rewards": {
                    "lower_body_pose_imitation_weight": 0.5,
                    "lower_body_stability_bundle_weight": 0.0,
                    "lower_body_stability_support_post_s": float("nan"),
                }
            },
            env_cfg=cfg,
        )
    after = (
        pose.weight,
        dict(pose.params),
        pose_probe.weight,
        dict(pose_probe.params),
        bundle.weight,
        dict(bundle.params),
        bundle_probe.weight,
        dict(bundle_probe.params),
    )
    assert after == before


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf"), True, "bad"])
def test_b1_rejects_invalid_positive_reward_weight(weight):
    with pytest.raises(train_mod._OverrideError):
        _apply(
            {
                "rewards": {
                    "lower_body_pose_imitation_weight": weight,
                    "lower_body_stability_bundle_weight": 0.0,
                }
            }
        )


@pytest.mark.parametrize("weight", [0.1, float("nan"), float("inf"), True, "bad"])
def test_b2_rejects_invalid_penalty_weight(weight):
    with pytest.raises(train_mod._OverrideError):
        _apply(
            {
                "rewards": {
                    "lower_body_pose_imitation_weight": 0.0,
                    "lower_body_stability_bundle_weight": weight,
                }
            }
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("lower_body_pose_imitation_std", 0.0),
        ("lower_body_pose_imitation_support_pre_s", -0.01),
        ("lower_body_stability_min_stance_width_m", 0.0),
        ("lower_body_stability_stance_scale_m", float("nan")),
        ("lower_body_pose_imitation_support_post_s", True),
        ("lower_body_stability_leg_velocity_margin_radps", -0.1),
        ("lower_body_stability_leg_velocity_scale_radps", float("inf")),
        ("lower_body_stability_support_post_s", -0.1),
    ],
)
def test_wave_parameter_overrides_reject_invalid_numbers(key, value):
    with pytest.raises(train_mod._OverrideError):
        _apply(
            {
                "rewards": {
                    "lower_body_pose_imitation_weight": 0.0,
                    "lower_body_stability_bundle_weight": 0.0,
                    key: value,
                }
            }
        )


def _schema3_wave_base():
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


def test_schema3_validates_explicit_b1_b2_and_refuses_double_active_or_order_drift():
    b1_cfg, _ = _apply(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.5,
                "lower_body_stability_bundle_weight": 0.0,
            }
        }
    )
    b2_cfg, _ = _apply(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.0,
                "lower_body_stability_bundle_weight": -0.25,
            }
        }
    )
    pose = train_mod._lower_body_pose_imitation_reward_contract(b1_cfg, _runtime_facts())
    b1_bundle = train_mod._lower_body_stability_bundle_reward_contract(
        b1_cfg, _runtime_facts()
    )
    b2_pose = train_mod._lower_body_pose_imitation_reward_contract(
        b2_cfg, _runtime_facts()
    )
    bundle = train_mod._lower_body_stability_bundle_reward_contract(
        b2_cfg, _runtime_facts()
    )

    b1 = _schema3_wave_base()
    b1["lower_body_pose_imitation_reward"] = pose
    b1["lower_body_stability_bundle_reward"] = b1_bundle
    TC.validate_schema3_contract_structure(b1)

    b2 = _schema3_wave_base()
    b2["lower_body_pose_imitation_reward"] = b2_pose
    b2["lower_body_stability_bundle_reward"] = bundle
    TC.validate_schema3_contract_structure(b2)

    both = _schema3_wave_base()
    both["lower_body_pose_imitation_reward"] = pose
    both["lower_body_stability_bundle_reward"] = bundle
    with pytest.raises(ValueError, match="mutually exclusive"):
        TC.validate_schema3_contract_structure(both)

    b1["joint_names"][0], b1["joint_names"][1] = b1["joint_names"][1], b1["joint_names"][0]
    with pytest.raises(ValueError, match="runtime/articulation joint identity"):
        TC.validate_schema3_contract_structure(b1)


def test_schema3_wave_blocks_are_paired_and_bind_articulation_identity():
    cfg, _ = _apply(
        {
            "rewards": {
                "lower_body_pose_imitation_weight": 0.0,
                "lower_body_stability_bundle_weight": 0.0,
            }
        }
    )
    pose = train_mod._lower_body_pose_imitation_reward_contract(cfg, _runtime_facts())
    bundle = train_mod._lower_body_stability_bundle_reward_contract(cfg, _runtime_facts())

    single = _schema3_wave_base()
    single["lower_body_pose_imitation_reward"] = pose
    with pytest.raises(ValueError, match="require both B1 and B2"):
        TC.validate_schema3_contract_structure(single)

    drift = _schema3_wave_base()
    drift["lower_body_pose_imitation_reward"] = pose
    drift["lower_body_stability_bundle_reward"] = bundle
    drift["articulation_joint_names"][0], drift["articulation_joint_names"][1] = (
        drift["articulation_joint_names"][1],
        drift["articulation_joint_names"][0],
    )
    with pytest.raises(ValueError, match="runtime/articulation joint identity"):
        TC.validate_schema3_contract_structure(drift)


def test_runner_consumes_wave_ledgers_when_either_probe_is_active():
    source = (
        Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
    ).read_text()
    assert '"lower_body_pose_imitation_probe" in active_reward_terms' in source
    assert '"lower_body_stability_bundle_probe" in active_reward_terms' in source
    assert "consume_lower_body_wave_activation_counters" in source


# ---------------------------------------------------------------------------
# Regression: the LIVE Isaac articulation enumerates joints breadth-first
# (left_hip_pitch, right_hip_pitch, waist_yaw, ...), NOT in the deploy-runtime
# order.  The first full-scene probe on 2026-07-20 fail-closed on a hardcoded
# full-order equality; these tests pin the by-name selection fix.
_A3_LIVE_BFS_ARTICULATION_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def _bfs_env(n=2):
    env, robot, motion, racket, age, same = _wave_env(n)
    robot.joint_names = list(_A3_LIVE_BFS_ARTICULATION_ORDER)
    robot.data.joint_names = list(_A3_LIVE_BFS_ARTICULATION_ORDER)
    return env, robot, motion, racket, age, same


def test_live_bfs_articulation_order_is_accepted_by_reward_runtime():
    assert sorted(_A3_LIVE_BFS_ARTICULATION_ORDER) == sorted(JOINTS)
    env, robot, motion, racket, age, same = _bfs_env(2)
    same[:] = True
    age[:] = 0.1
    racket.pre_strike[:] = False
    bundle = hope_rewards_mod.lower_body_stability_bundle(env)
    pose = hope_rewards_mod.lower_body_pose_imitation(env)
    assert torch.isfinite(bundle).all()
    assert torch.isfinite(pose).all()


def test_live_bfs_order_selects_the_same_12_leg_joints_by_name():
    env, robot, motion, racket, age, same = _bfs_env(2)
    _, _, _, _, _, leg_ids = hope_rewards_mod._lower_body_runtime_tensors(
        env, "motion", require_motion_reference=False
    )
    names = [_A3_LIVE_BFS_ARTICULATION_ORDER[i] for i in leg_ids]
    assert set(names) == set(hope_rewards_mod._LOWER_BODY_LEG_JOINT_NAMES)
    assert len(leg_ids) == 12


def test_live_bfs_order_accepted_by_train_contract_names():
    names, legs = train_mod._lower_body_runtime_contract_names(
        {
            "joint_names": list(_A3_LIVE_BFS_ARTICULATION_ORDER),
            "articulation_joint_names": list(_A3_LIVE_BFS_ARTICULATION_ORDER),
        },
        31,
        require_motion_reference=True,
    )
    assert len(legs) == 12
    assert names == list(_A3_LIVE_BFS_ARTICULATION_ORDER)


def test_scrambled_or_wrong_name_set_still_rejected_by_train_contract_names():
    wrong = list(_A3_LIVE_BFS_ARTICULATION_ORDER)
    wrong[0] = "not_a_joint"
    with pytest.raises(RuntimeError, match="runtime order"):
        train_mod._lower_body_runtime_contract_names(
            {"joint_names": wrong, "articulation_joint_names": wrong},
            31,
            require_motion_reference=False,
        )
    mismatched = {
        "joint_names": list(_A3_LIVE_BFS_ARTICULATION_ORDER),
        "articulation_joint_names": list(reversed(_A3_LIVE_BFS_ARTICULATION_ORDER)),
    }
    with pytest.raises(RuntimeError, match="runtime order"):
        train_mod._lower_body_runtime_contract_names(
            mismatched, 31, require_motion_reference=False
        )


def _minimal_wave_contract(weight_pose=0.0, weight_bundle=-0.25):
    return {
        "lower_body_pose_imitation_reward": {
            "schema_version": 1, "enabled": weight_pose > 0.0, "probe_enabled": True,
            "activation_ledger": "weight_independent_control_step_counters",
            "weight": weight_pose, "std_rad": 0.35,
            "support_pre_s": 0.30, "support_post_s": 0.40,
            "racket_command_name": "racket_target", "motion_command_name": "motion",
            "joint_count": 12, "joint_names": list(TC._A3_LOWER_BODY_LEGS),
            "joint_order": "canonical_deploy_order_selected_by_name",
            "reference_joint_order": "motion_command_runtime_articulation_identity",
            "formula": "exp(-mean(square(q_leg-qref_leg))/square(std_rad))",
            "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
            "success_conditioned": False,
        },
        "lower_body_stability_bundle_reward": {
            "schema_version": 1, "enabled": weight_bundle < 0.0, "probe_enabled": True,
            "activation_ledger": "weight_independent_control_step_counters",
            "weight": weight_bundle, "min_stance_width_m": 0.22, "stance_scale_m": 0.05,
            "leg_velocity_margin_radps": 1.0, "leg_velocity_scale_radps": 0.5,
            "support_pre_s": 0.30, "support_post_s": 0.40,
            "racket_command_name": "racket_target", "motion_command_name": "motion",
            "leg_joint_count": 12, "leg_joint_names": list(TC._A3_LOWER_BODY_LEGS),
            "foot_body_names": ["left_ankle_roll_Link", "right_ankle_roll_Link"],
            "joint_order": "canonical_deploy_order_selected_by_name",
            "stance_width_frame": "base_yaw_lateral_signed_left_minus_right",
            "components": ["stance_width_lower_hinge", "twelve_leg_realized_qdot_tail"],
            "formula": "mean(bounded_stance_tail,bounded_leg_qdot_tail)",
            "gate": "phase_tts_pre_or_same_attempt_post_inclusive",
            "success_conditioned": False, "uses_motion_reference": False,
            "duplicates_slip_or_upright": False,
        },
        "articulation_body_names": [
            "pelvis_link", "left_ankle_roll_Link", "right_ankle_roll_Link",
        ],
    }


def test_schema3_wave_validator_accepts_live_bfs_full_order():
    contract = _minimal_wave_contract()
    TC._validate_lower_body_wave_contracts(
        contract,
        list(_A3_LIVE_BFS_ARTICULATION_ORDER),
        list(_A3_LIVE_BFS_ARTICULATION_ORDER),
    )


def test_schema3_wave_validator_rejects_wrong_set_or_mismatched_articulation():
    contract = _minimal_wave_contract()
    wrong = list(_A3_LIVE_BFS_ARTICULATION_ORDER)
    wrong[3] = "bogus_joint"
    with pytest.raises(ValueError, match="joint identity"):
        TC._validate_lower_body_wave_contracts(contract, wrong, wrong)
    with pytest.raises(ValueError, match="joint identity"):
        TC._validate_lower_body_wave_contracts(
            contract,
            list(_A3_LIVE_BFS_ARTICULATION_ORDER),
            list(reversed(_A3_LIVE_BFS_ARTICULATION_ORDER)),
        )


def test_consume_resets_inference_mode_counters_without_crash():
    """Counters are created inside env.step (torch.inference_mode); the consume/reset
    path runs at log time outside it and crashed on 2026-07-20 probe attempt 4."""
    env, robot, motion, racket, age, same = _wave_env(2)
    with torch.inference_mode():
        hope_rewards_mod.lower_body_stability_bundle(env)
        hope_rewards_mod.lower_body_pose_imitation(env)
    first = hope_rewards_mod.consume_lower_body_wave_activation_counters(env)
    assert first["bundle/observed_sample_count"].item() == 2
    second = hope_rewards_mod.consume_lower_body_wave_activation_counters(env)
    assert all(value.item() == 0 for value in second.values())
