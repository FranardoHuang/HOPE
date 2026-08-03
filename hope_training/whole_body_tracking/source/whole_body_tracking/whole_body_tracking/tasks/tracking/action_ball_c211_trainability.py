"""Fail-closed runtime contract for the fresh incoming-ball C211 leaf."""

from __future__ import annotations

try:
    from whole_body_tracking.tasks.tracking.action_ball_a211_trainability import (
        _TrainabilityContract,
        _validate_cfg,
        _validate_runner,
        _validate_runtime,
        _validate_wrapped_env,
        layout_names,
    )
except ModuleNotFoundError:
    # Dependency-free host contract tests load this file directly without
    # importing the Isaac package (whose __init__ requires isaaclab_tasks).
    import importlib.util
    from pathlib import Path
    import sys

    _sibling_path = Path(__file__).with_name("action_ball_a211_trainability.py")
    _sibling_spec = importlib.util.spec_from_file_location(
        "action_ball_a211_trainability_host", _sibling_path
    )
    if _sibling_spec is None or _sibling_spec.loader is None:
        raise RuntimeError("cannot load sibling A211 trainability contract")
    _sibling = importlib.util.module_from_spec(_sibling_spec)
    sys.modules[_sibling_spec.name] = _sibling
    _sibling_spec.loader.exec_module(_sibling)
    _TrainabilityContract = _sibling._TrainabilityContract
    _validate_cfg = _sibling._validate_cfg
    _validate_runner = _sibling._validate_runner
    _validate_runtime = _sibling._validate_runtime
    _validate_wrapped_env = _sibling._validate_wrapped_env
    layout_names = _sibling.layout_names


C211_ACTOR_CONTRACT = "action_ball_c211"
C211_CRITIC_CONTRACT = "action_ball_c211_critic_v1"
C211_TRAINABILITY_CONTRACT = "action_ball_c211_fixed_midpoint_learnability_v1"
C211_ACTOR_NORMALIZER_IDENTITY = "action_ball_c211_actor_norm_v1"
C211_CRITIC_NORMALIZER_IDENTITY = "action_ball_c211_critic_norm_v1"
C211_REWARD_CONTRACT = "action_ball_c211_achieved_outcome_reward_v2"


def c211_reward_contract_facts() -> dict:
    """Return the exact C211 achieved-outcome reward economics."""

    return {
        "identity": C211_REWARD_CONTRACT,
        "desired_contact_position_velocity_face_consumed": False,
        "task_valid_required": True,
        "strike_bridge": {
            "term": "c225_strike_ball_paddle_center_proximity",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_c225_rewards."
                "c225_strike_ball_paddle_center_proximity"
            ),
            "weight": 220.0,
            "std_m": 0.15,
            "kernel": "cauchy_inverse_quadratic",
            "eligibility": "task_valid_active_swing_single_exact_strike_tick",
            "miss_retains_gradient": True,
        },
        "economics": {
            "policy_dt_s": 0.02,
            "compatible_swing_motion_static_max": 3.6575,
            "strike_bridge_post_dt_peak": 4.4,
            "legal_landing_post_dt_min": 6.0,
            "ordering": "motion_lt_strike_peak_lt_legal_landing",
        },
        "landing": {
            "term": "virtual_landing",
            "callable": (
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_c225_rewards.c225_landing_outcome_actual_contact"
            ),
            "weight": 500.0,
            "evidence_source": "analytic_prediction_from_achieved_selected_rubber_contact",
            "observed_physical_landing_available": False,
            "eligibility": (
                "task_valid_and_actual_selected_rubber_contact_and_finite_landing_plane_"
                "and_net_crossed_and_net_clear"
            ),
            "legal_opponent_table": "0.6_plus_0.4_gaussian",
            "opponent_side_off_table": "0.5_times_same_gaussian",
            "miss_or_invalid_or_hypothetical": 0.0,
            "sigma_m": 1.0,
        },
        "legacy_duplicate_outcome_terms_active": False,
        "rollout0_required_priors": [
            "upright_exp",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
            "motion_racket_position",
            "motion_racket_velocity",
            "motion_racket_normal",
            "motion_racket_long_axis",
        ],
    }


C211_ACTOR_LAYOUT = (
    ("actual_base_now_world", 15),
    ("joint_pos", 31),
    ("teacher_joint_pos", 31),
    ("joint_vel", 31),
    ("teacher_joint_vel", 31),
    ("actions", 31),
    ("racket_site_achieved_now_heading", 9),
    ("racket_site_teacher_now_heading", 9),
    ("racket_site_teacher_at_reference_hit_heading", 9),
    ("incoming_ball_contact_position_heading", 3),
    ("incoming_ball_contact_velocity_heading", 3),
    ("incoming_ball_contact_spin_heading", 3),
    ("desired_base_xy_world", 2),
    ("time_to_contact", 1),
    ("time_to_teacher_start", 1),
    ("task_valid", 1),
)

C211_CRITIC_LAYOUT = (
    ("command", 62),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("body_pos", 42),
    ("body_ori", 84),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", 31),
    ("joint_vel", 31),
    ("actions", 31),
    ("racket_site_teacher_at_reference_hit_heading", 9),
    ("incoming_ball_contact_position_heading", 3),
    ("incoming_ball_contact_velocity_heading", 3),
    ("incoming_ball_contact_spin_heading", 3),
    ("desired_base_xy_world", 2),
    ("time_to_contact", 1),
    ("time_to_teacher_start", 1),
    ("task_valid", 1),
)

C211_ACTOR_WIDTH = sum(dim for _name, dim in C211_ACTOR_LAYOUT)
C211_CRITIC_WIDTH = sum(dim for _name, dim in C211_CRITIC_LAYOUT)
assert C211_ACTOR_WIDTH == 211
assert C211_CRITIC_WIDTH == 319


def _extra_runtime_facts() -> dict:
    return {
        "contact_target_absent": True,
        "c225_reward_contract": c211_reward_contract_facts(),
    }


_C211 = _TrainabilityContract(
    label="C211",
    actor_contract=C211_ACTOR_CONTRACT,
    critic_contract=C211_CRITIC_CONTRACT,
    trainability_contract=C211_TRAINABILITY_CONTRACT,
    actor_normalizer_identity=C211_ACTOR_NORMALIZER_IDENTITY,
    critic_normalizer_identity=C211_CRITIC_NORMALIZER_IDENTITY,
    actor_layout=C211_ACTOR_LAYOUT,
    critic_layout=C211_CRITIC_LAYOUT,
    extra_runtime_facts=_extra_runtime_facts,
)


def validate_action_ball_c211_cfg_trainability(env_cfg, *, entrypoint: str) -> None:
    _validate_cfg(env_cfg, entrypoint=entrypoint, contract=_C211)


def validate_action_ball_c211_runtime(env) -> dict | None:
    return _validate_runtime(env, contract=_C211)


def validate_action_ball_c211_wrapped_env(env) -> dict | None:
    return _validate_wrapped_env(env, contract=_C211)


def validate_action_ball_c211_runner(runner) -> dict | None:
    return _validate_runner(runner, contract=_C211)
