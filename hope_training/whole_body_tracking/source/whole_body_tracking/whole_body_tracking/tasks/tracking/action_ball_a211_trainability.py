"""Fail-closed runtime contract for the fresh A211 learnability leaf.

A211 removes the actor-only 15-D teacher-base row from A225, splits actual base
state into a 12-D world localizer row plus one 3-D body-frame IMU gyro row, and
appends the causal one-bit ``task_valid`` admission signal.  Its privileged
critic also appends that signal, producing a fresh 211/319 actor/critic ABI.
No 225, interim 210, or pre-IMU A211 lineage is consumable by this contract.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple


A211_ACTOR_CONTRACT = "action_ball_a211"
A211_CRITIC_CONTRACT = "action_ball_a211_critic_v1"
A211_TRAINABILITY_CONTRACT = "action_ball_a211_fixed_question_learnability_v2"
A211_ACTOR_NORMALIZER_IDENTITY = "action_ball_a211_actor_norm_v2"
A211_CRITIC_NORMALIZER_IDENTITY = "action_ball_a211_critic_norm_v1"

A211_ACTOR_LAYOUT = (
    ("actual_base_pose_lin_vel_world", 12),
    ("base_ang_vel_body", 3),
    ("joint_pos", 31),
    ("joint_vel", 31),
    ("actions", 31),
    ("racket_site_achieved_now_heading", 9),
    ("teacher_joint_pos", 31),
    ("teacher_joint_vel", 31),
    ("racket_site_teacher_now_heading", 9),
    ("racket_site_teacher_at_reference_hit_heading", 9),
    ("task_desired_contact_position_heading", 3),
    ("task_desired_contact_velocity_heading", 3),
    ("task_desired_contact_face_heading", 3),
    ("desired_base_xy_world", 2),
    ("time_to_contact", 1),
    ("time_to_teacher_start", 1),
    ("task_valid", 1),
)

A211_CRITIC_LAYOUT = (
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
    ("task_desired_contact_position_heading", 3),
    ("task_desired_contact_velocity_heading", 3),
    ("task_desired_contact_face_heading", 3),
    ("desired_base_xy_world", 2),
    ("time_to_contact", 1),
    ("time_to_teacher_start", 1),
    ("task_valid", 1),
)

A211_ACTOR_WIDTH = sum(dim for _name, dim in A211_ACTOR_LAYOUT)
A211_CRITIC_WIDTH = sum(dim for _name, dim in A211_CRITIC_LAYOUT)
assert A211_ACTOR_WIDTH == 211
assert A211_CRITIC_WIDTH == 319

_LEGACY_MODES = {
    "action_ball_a225",
    "action_ball_c225",
    "action_ball_a210",
    "action_ball_c210",
}


def action_ball_211_wait_contract_facts() -> dict:
    return {
        "identity": "action_ball_pre_task_wait_schedule_v1",
        "policy_dt_s": 0.02,
        "seed": 20260804,
        "min_wait_ticks": 5,
        "max_wait_ticks": 25,
        "episode_horizon_ticks": 500,
        "required_active_ticks": 200,
        "schedule_canonical_sha256": (
            "58aa7bb62406d301df619caf7026af8d595f4b8cd9594ea8441b4c89997d400e"
        ),
        "task_valid_actor_and_critic": True,
        "wait_task_ball_base_and_clocks_masked": True,
        "wait_remaining_observed": False,
    }


def action_ball_211_question_source_contract_facts() -> dict:
    """Separate the current diagnostic tape from the final curriculum ABI."""

    return {
        "identity": "action_ball_211_question_source_scope_v1",
        "current_immutable_tape": {
            "scope": "diagnostic_n1_early_fixed_band_only",
            "final_curriculum_frozen": False,
        },
        "final_curriculum": {
            "source": "pregenerated_cached_band_question_bank",
            "generation": "offline_before_rollout",
            "reset_selection": "index_one_bank_row",
            "online_inverse_solves_per_reset": 0,
            "online_inverse_solves_per_step": 0,
            "wait_remaining_observed": False,
        },
    }


def _validate_wait_cfg(env_cfg, *, label: str) -> dict:
    facts = action_ball_211_wait_contract_facts()
    command_cfg = getattr(
        getattr(env_cfg, "commands", None), "racket_target", None
    )
    expected = {
        "action_ball_task_wait_enabled": True,
        "action_ball_task_wait_policy_dt_s": facts["policy_dt_s"],
        "action_ball_task_wait_seed": facts["seed"],
        "action_ball_task_wait_min_wait_ticks": facts["min_wait_ticks"],
        "action_ball_task_wait_max_wait_ticks": facts["max_wait_ticks"],
        "action_ball_task_wait_episode_horizon_ticks": facts["episode_horizon_ticks"],
        "action_ball_task_wait_required_active_ticks": facts["required_active_ticks"],
    }
    for attribute, expected_value in expected.items():
        actual = getattr(command_cfg, attribute, None)
        if type(expected_value) is bool:
            valid = type(actual) is bool and actual is expected_value
        elif type(expected_value) is float:
            valid = type(actual) in (int, float) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            valid = type(actual) is int and actual == expected_value
        if not valid:
            raise RuntimeError(
                f"{label} WAIT runtime fact {attribute} must be "
                f"{expected_value!r}, got {actual!r}"
            )
    return facts


class _TrainabilityContract(NamedTuple):
    label: str
    actor_contract: str
    critic_contract: str
    trainability_contract: str
    actor_normalizer_identity: str
    critic_normalizer_identity: str
    actor_layout: tuple[tuple[str, int], ...]
    critic_layout: tuple[tuple[str, int], ...]
    extra_runtime_facts: object = None

    @property
    def actor_width(self) -> int:
        return sum(dim for _name, dim in self.actor_layout)

    @property
    def critic_width(self) -> int:
        return sum(dim for _name, dim in self.critic_layout)


_A211 = _TrainabilityContract(
    label="A211",
    actor_contract=A211_ACTOR_CONTRACT,
    critic_contract=A211_CRITIC_CONTRACT,
    trainability_contract=A211_TRAINABILITY_CONTRACT,
    actor_normalizer_identity=A211_ACTOR_NORMALIZER_IDENTITY,
    critic_normalizer_identity=A211_CRITIC_NORMALIZER_IDENTITY,
    actor_layout=A211_ACTOR_LAYOUT,
    critic_layout=A211_CRITIC_LAYOUT,
)


def _cfg_from(value):
    cfg = getattr(value, "cfg", None)
    if cfg is not None:
        return cfg
    unwrapped = getattr(value, "unwrapped", None)
    if unwrapped is not None and unwrapped is not value:
        return _cfg_from(unwrapped)
    return value


def _layout_from_manager(manager, group: str, *, label: str):
    try:
        names = tuple(str(name) for name in manager.active_terms[group])
        raw_dims = manager.group_obs_term_dim[group]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{label} trainability requires an explicit {group!r} observation group"
        ) from exc
    dims = []
    for value in raw_dims:
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise RuntimeError(
                    f"{label} {group} observation term has non-vector shape {value!r}"
                )
            value = value[0]
        dims.append(int(value))
    return tuple(zip(names, dims))


def _total_from_manager(manager, group: str, *, label: str) -> int:
    try:
        value = manager.group_obs_dim[group]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{label} trainability requires an explicit {group!r} observation width"
        ) from exc
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError(
                f"{label} {group} observation group has non-vector shape {value!r}"
            )
        value = value[0]
    return int(value)


def _validate_cfg(env_cfg, *, entrypoint: str, contract: _TrainabilityContract):
    mode = str(getattr(env_cfg, "obs_mode", "") or "")
    if mode in _LEGACY_MODES:
        raise RuntimeError(
            f"{entrypoint}: legacy {mode} is not consumable by the fresh A211/C211 ABI"
        )
    if mode != contract.actor_contract:
        return
    marker = getattr(env_cfg, "action_ball_211_trainability_contract", None)
    if marker != contract.trainability_contract:
        raise RuntimeError(
            f"{entrypoint}: {contract.actor_contract} is construction-only unless "
            f"the exact trainability contract {contract.trainability_contract!r} is present"
        )
    if getattr(env_cfg, "action_ball_211_construction_only", None) is not False:
        raise RuntimeError(
            f"{entrypoint}: trainable {contract.label} must explicitly disable "
            "its construction-only marker"
        )
    if getattr(env_cfg, "critic_obs_contract", None) != contract.critic_contract:
        raise RuntimeError(
            f"{entrypoint}: {contract.label} critic contract must be "
            f"{contract.critic_contract!r}"
        )
    observations = getattr(env_cfg, "observations", None)
    if observations is None or getattr(observations, "critic", None) is None:
        raise RuntimeError(
            f"{entrypoint}: {contract.label} requires an explicit privileged critic "
            "group; symmetric actor fallback is forbidden"
        )
    _validate_wait_cfg(env_cfg, label=contract.label)


def _validate_runtime(env, *, contract: _TrainabilityContract) -> dict | None:
    cfg = _cfg_from(env)
    if str(getattr(cfg, "obs_mode", "") or "") != contract.actor_contract:
        return None
    _validate_cfg(cfg, entrypoint="runtime", contract=contract)
    manager = getattr(getattr(env, "unwrapped", env), "observation_manager", None)
    if manager is None:
        raise RuntimeError(f"{contract.label} runtime lacks an ObservationManager")
    actor_layout = _layout_from_manager(manager, "policy", label=contract.label)
    critic_layout = _layout_from_manager(manager, "critic", label=contract.label)
    actor_width = _total_from_manager(manager, "policy", label=contract.label)
    critic_width = _total_from_manager(manager, "critic", label=contract.label)
    if actor_layout != contract.actor_layout or actor_width != contract.actor_width:
        raise RuntimeError(
            f"{contract.label} actor runtime ABI mismatch: "
            f"layout={actor_layout!r} width={actor_width}"
        )
    if critic_layout != contract.critic_layout or critic_width != contract.critic_width:
        raise RuntimeError(
            f"{contract.label} critic runtime ABI mismatch: "
            f"layout={critic_layout!r} width={critic_width}"
        )
    facts = {
        "trainability_contract": contract.trainability_contract,
        "actor_contract": contract.actor_contract,
        "critic_contract": contract.critic_contract,
        "actor_width": actor_width,
        "critic_width": critic_width,
        "actor_normalizer_identity": contract.actor_normalizer_identity,
        "critic_normalizer_identity": contract.critic_normalizer_identity,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
        "task_valid_required": True,
        "task_wait_contract": _validate_wait_cfg(cfg, label=contract.label),
        "question_source_contract": action_ball_211_question_source_contract_facts(),
    }
    if callable(contract.extra_runtime_facts):
        facts.update(contract.extra_runtime_facts())
    return facts


def _validate_wrapped_env(env, *, contract: _TrainabilityContract) -> dict | None:
    cfg = _cfg_from(env)
    if str(getattr(cfg, "obs_mode", "") or "") != contract.actor_contract:
        return None
    facts = _validate_runtime(getattr(env, "unwrapped", env), contract=contract)
    actor_width = getattr(env, "num_obs", None)
    critic_width = getattr(env, "num_privileged_obs", None)
    if type(actor_width) is not int or actor_width != contract.actor_width:
        raise RuntimeError(
            f"{contract.label} RSL wrapper actor width must be {contract.actor_width}, "
            f"got {actor_width!r}"
        )
    if type(critic_width) is not int or critic_width != contract.critic_width:
        raise RuntimeError(
            f"{contract.label} RSL wrapper must expose a real "
            f"{contract.critic_width}-D privileged critic; "
            f"symmetric fallback value={critic_width!r}"
        )
    return facts


def first_linear_input_width(module) -> int | None:
    if module is None:
        return None
    in_features = getattr(module, "in_features", None)
    if type(in_features) is int:
        return in_features
    children = getattr(module, "children", None)
    if not callable(children):
        return None
    for child in children():
        width = first_linear_input_width(child)
        if width is not None:
            return width
    return None


def _validate_runner(runner, *, contract: _TrainabilityContract) -> dict | None:
    env = getattr(runner, "env", None)
    if env is None:
        return None
    facts = _validate_wrapped_env(env, contract=contract)
    if facts is None:
        return None
    algorithm = getattr(runner, "alg", None)
    policy = getattr(algorithm, "policy", None)
    if policy is None:
        policy = getattr(algorithm, "actor_critic", None)
    actor_width = getattr(policy, "num_actor_obs", None)
    critic_width = getattr(policy, "num_critic_obs", None)
    if type(actor_width) is not int:
        actor_width = first_linear_input_width(getattr(policy, "actor", None))
    if type(critic_width) is not int:
        critic_width = first_linear_input_width(getattr(policy, "critic", None))
    if actor_width != contract.actor_width or critic_width != contract.critic_width:
        raise RuntimeError(
            f"{contract.label} runner network ABI mismatch: actor={actor_width!r} "
            f"critic={critic_width!r}; symmetric fallback is forbidden"
        )
    if getattr(runner, "empirical_normalization", None) is not True:
        raise RuntimeError(
            f"{contract.label} requires fresh empirical actor and critic normalizers"
        )
    actor_attribute, actor_normalizer, _ = runner._resolve_runtime_normalizer("actor")
    critic_attribute, critic_normalizer, _ = runner._resolve_runtime_normalizer("critic")
    if actor_normalizer is None or critic_normalizer is None:
        raise RuntimeError(
            f"{contract.label} requires fresh actor and critic empirical normalizers"
        )
    if actor_normalizer is critic_normalizer:
        raise RuntimeError(
            f"{contract.label} actor and critic normalizers must be distinct objects"
        )
    facts = dict(facts)
    facts.update(
        {
            "runner_actor_width": actor_width,
            "runner_critic_width": critic_width,
            "actor_normalizer_attribute": actor_attribute,
            "critic_normalizer_attribute": critic_attribute,
        }
    )
    return facts


def validate_action_ball_211_cfg_trainability(env_cfg, *, entrypoint: str) -> None:
    _validate_cfg(env_cfg, entrypoint=entrypoint, contract=_A211)


def validate_action_ball_211_runtime(env) -> dict | None:
    return _validate_runtime(env, contract=_A211)


def validate_action_ball_211_wrapped_env(env) -> dict | None:
    return _validate_wrapped_env(env, contract=_A211)


def validate_action_ball_211_runner(runner) -> dict | None:
    return _validate_runner(runner, contract=_A211)


def layout_names(layout: Iterable[tuple[str, int]]) -> tuple[str, ...]:
    return tuple(name for name, _dim in layout)
