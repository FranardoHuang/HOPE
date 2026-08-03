"""Fail-closed training contract for the fixed-question A225 diagnostic.

The generic A225/C225 construction leaves remain non-trainable because RSL-RL
otherwise treats an absent privileged group as a request for a symmetric
actor/critic input.  One dedicated A225 learnability leaf is explicitly enabled
only when it supplies the exact 225/318-D layouts and fresh normalizer identities
below; C225 remains construction-only.  The helpers are shared by construction
and runner entry points so no path can silently take the symmetric fallback.
"""

from __future__ import annotations

from typing import Iterable


A225_ACTOR_CONTRACT = "action_ball_a225"
A225_CRITIC_CONTRACT = "action_ball_a225_critic_v1"
A225_TRAINABILITY_CONTRACT = "action_ball_a225_fixed_question_learnability_v1"
A225_ACTOR_NORMALIZER_IDENTITY = "action_ball_a225_actor_norm_v1"
A225_CRITIC_NORMALIZER_IDENTITY = "action_ball_a225_critic_norm_v1"

A225_ACTOR_LAYOUT = (
    ("actual_base_now_world", 15),
    ("teacher_base_now_world", 15),
    ("joint_pos", 31),
    ("teacher_joint_pos", 31),
    ("joint_vel", 31),
    ("teacher_joint_vel", 31),
    ("actions", 31),
    ("racket_site_achieved_now_heading", 9),
    ("racket_site_teacher_now_heading", 9),
    ("racket_site_teacher_at_reference_hit_heading", 9),
    ("task_desired_contact_position_heading", 3),
    ("task_desired_contact_velocity_heading", 3),
    ("task_desired_contact_face_heading", 3),
    ("desired_base_xy_world", 2),
    ("time_to_contact", 1),
    ("time_to_teacher_start", 1),
)

# This is an A225-owned ABI even though its scalar width equals the historical
# Stage-1 critic.  The three desired-contact rows use the atomic A225 task
# snapshot, and its normalizer/checkpoint lineage is new.
A225_CRITIC_LAYOUT = (
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
)

A225_ACTOR_WIDTH = sum(dim for _name, dim in A225_ACTOR_LAYOUT)
A225_CRITIC_WIDTH = sum(dim for _name, dim in A225_CRITIC_LAYOUT)
assert A225_ACTOR_WIDTH == 225
assert A225_CRITIC_WIDTH == 318


def _layout_from_manager(manager, group: str) -> tuple[tuple[str, int], ...]:
    try:
        names = tuple(str(name) for name in manager.active_terms[group])
        raw_dims = manager.group_obs_term_dim[group]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"A225 trainability requires an explicit {group!r} observation group"
        ) from exc
    dims = []
    for value in raw_dims:
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise RuntimeError(
                    f"A225 {group} observation term has non-vector shape {value!r}"
                )
            value = value[0]
        dims.append(int(value))
    return tuple(zip(names, dims))


def _total_from_manager(manager, group: str) -> int:
    try:
        value = manager.group_obs_dim[group]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"A225 trainability requires an explicit {group!r} observation width"
        ) from exc
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError(
                f"A225 {group} observation group has non-vector shape {value!r}"
            )
        value = value[0]
    return int(value)


def _cfg_from(value):
    cfg = getattr(value, "cfg", None)
    if cfg is not None:
        return cfg
    unwrapped = getattr(value, "unwrapped", None)
    if unwrapped is not None and unwrapped is not value:
        return _cfg_from(unwrapped)
    return value


def validate_action_ball_225_cfg_trainability(env_cfg, *, entrypoint: str) -> None:
    """Reject construction-only A/C leaves before scene or runner creation."""

    mode = str(getattr(env_cfg, "obs_mode", "") or "")
    if mode not in (A225_ACTOR_CONTRACT, "action_ball_c225"):
        return
    if mode == "action_ball_c225":
        raise RuntimeError(
            f"{entrypoint}: action_ball_c225 remains construction-only and is not "
            "part of the A225 four-arm learnability matrix"
        )
    marker = getattr(env_cfg, "action_ball_225_trainability_contract", None)
    if marker != A225_TRAINABILITY_CONTRACT:
        raise RuntimeError(
            f"{entrypoint}: action_ball_a225 is construction-only unless the exact "
            f"trainability contract {A225_TRAINABILITY_CONTRACT!r} is present"
        )
    if getattr(env_cfg, "action_ball_225_construction_only", None) is not False:
        raise RuntimeError(
            f"{entrypoint}: trainable A225 must explicitly disable its construction-only marker"
        )
    if getattr(env_cfg, "critic_obs_contract", None) != A225_CRITIC_CONTRACT:
        raise RuntimeError(
            f"{entrypoint}: A225 critic contract must be {A225_CRITIC_CONTRACT!r}"
        )
    observations = getattr(env_cfg, "observations", None)
    if observations is None or getattr(observations, "critic", None) is None:
        raise RuntimeError(
            f"{entrypoint}: A225 requires an explicit privileged critic group; "
            "symmetric actor fallback is forbidden"
        )


def validate_action_ball_225_runtime(env) -> dict | None:
    """Validate the instantiated ObservationManager before RSL wrapping."""

    cfg = _cfg_from(env)
    if str(getattr(cfg, "obs_mode", "") or "") != A225_ACTOR_CONTRACT:
        return None
    validate_action_ball_225_cfg_trainability(cfg, entrypoint="runtime")
    manager = getattr(getattr(env, "unwrapped", env), "observation_manager", None)
    if manager is None:
        raise RuntimeError("A225 runtime lacks an ObservationManager")
    actor_layout = _layout_from_manager(manager, "policy")
    critic_layout = _layout_from_manager(manager, "critic")
    actor_width = _total_from_manager(manager, "policy")
    critic_width = _total_from_manager(manager, "critic")
    if actor_layout != A225_ACTOR_LAYOUT or actor_width != A225_ACTOR_WIDTH:
        raise RuntimeError(
            f"A225 actor runtime ABI mismatch: layout={actor_layout!r} width={actor_width}"
        )
    if critic_layout != A225_CRITIC_LAYOUT or critic_width != A225_CRITIC_WIDTH:
        raise RuntimeError(
            f"A225 critic runtime ABI mismatch: layout={critic_layout!r} width={critic_width}"
        )
    return {
        "trainability_contract": A225_TRAINABILITY_CONTRACT,
        "actor_contract": A225_ACTOR_CONTRACT,
        "critic_contract": A225_CRITIC_CONTRACT,
        "actor_width": actor_width,
        "critic_width": critic_width,
        "actor_normalizer_identity": A225_ACTOR_NORMALIZER_IDENTITY,
        "critic_normalizer_identity": A225_CRITIC_NORMALIZER_IDENTITY,
        "fresh_normalizers_required": True,
        "symmetric_critic_fallback_forbidden": True,
    }


def validate_action_ball_225_wrapped_env(env) -> dict | None:
    """Reject RSL's missing-privileged-observation symmetric fallback."""

    cfg = _cfg_from(env)
    if str(getattr(cfg, "obs_mode", "") or "") != A225_ACTOR_CONTRACT:
        return None
    facts = validate_action_ball_225_runtime(getattr(env, "unwrapped", env))
    actor_width = getattr(env, "num_obs", None)
    critic_width = getattr(env, "num_privileged_obs", None)
    if type(actor_width) is not int or actor_width != A225_ACTOR_WIDTH:
        raise RuntimeError(
            f"A225 RSL wrapper actor width must be {A225_ACTOR_WIDTH}, got {actor_width!r}"
        )
    if type(critic_width) is not int or critic_width != A225_CRITIC_WIDTH:
        raise RuntimeError(
            "A225 RSL wrapper must expose a real 318-D privileged critic; "
            f"symmetric fallback value={critic_width!r}"
        )
    return facts


def first_linear_input_width(module) -> int | None:
    """Return the first linear-like ``in_features`` without importing torch."""

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


def validate_action_ball_225_runner(runner) -> dict | None:
    """Validate network input widths immediately after runner construction."""

    env = getattr(runner, "env", None)
    if env is None:
        return None
    facts = validate_action_ball_225_wrapped_env(env)
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
    if actor_width != A225_ACTOR_WIDTH or critic_width != A225_CRITIC_WIDTH:
        raise RuntimeError(
            "A225 runner network ABI mismatch: "
            f"actor={actor_width!r} critic={critic_width!r}; symmetric fallback is forbidden"
        )
    if getattr(runner, "empirical_normalization", None) is not True:
        raise RuntimeError(
            "A225 requires fresh empirical actor and critic normalizers"
        )
    actor_attribute, actor_normalizer, _ = runner._resolve_runtime_normalizer("actor")
    critic_attribute, critic_normalizer, _ = runner._resolve_runtime_normalizer("critic")
    if actor_normalizer is None or critic_normalizer is None:
        raise RuntimeError("A225 requires fresh actor and critic empirical normalizers")
    if actor_normalizer is critic_normalizer:
        raise RuntimeError("A225 actor and critic normalizers must be distinct objects")
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


def layout_names(layout: Iterable[tuple[str, int]]) -> tuple[str, ...]:
    return tuple(name for name, _dim in layout)
