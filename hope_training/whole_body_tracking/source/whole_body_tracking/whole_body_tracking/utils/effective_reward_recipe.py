"""Dependency-light receipts for the effective reward recipe.

The input is the *already composed* environment configuration, not a task YAML
or a list of Hydra overrides.  This module deliberately does not import Hydra,
OmegaConf, Isaac Lab, or Torch.  It only records active reward terms (non-None
terms whose finite numeric weight is non-zero) and hashes their effective
callable, weight, and parameters.

The SHA-256 covers the canonical JSON encoding of::

    {"schema_version": 1, "terms": [...]}

It does not cover the convenience ``sha256`` field returned in the receipt.
Values which cannot be represented without guessing at their semantics fail
closed instead of falling back to ``repr()``, whose output may contain process
addresses or other unstable state.

The runtime ledger at the end of this module is likewise Isaac-Lab independent.
It accepts a tiny tensor-operations adapter so host tests can use NumPy while
the trainer supplies Torch lazily.  It deliberately reads the narrow
``RewardManager._step_reward`` cache contract and validates that cache against
``RewardManager._reward_buf`` on every environment step.  If that versioned
contract is unavailable, it fails before claiming activation evidence.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence


EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION = 1
EFFECTIVE_REWARD_ACTIVATION_SCHEMA_VERSION = 1
REWARD_TERM_ROLE_OBJECTIVE = "objective"
REWARD_TERM_ROLE_DIAGNOSTIC_PROBE = "diagnostic_probe"
ACTION_BALL_ADOPTED_STEP_DT_S = 0.02
ACTION_BALL_ADOPTED_DEATH_WEIGHT = -3600.0
ACTION_BALL_ADOPTED_DEATH_PER_TERMINATION = -72.0
ACTION_BALL_ADOPTED_SOFT_LIMIT_WEIGHT = -40.0
ACTION_BALL_REWARD_GROUP_TAXONOMY_SCHEMA_VERSION = 1
ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY = "mjlab_balance_stability"
ACTION_BALL_REWARD_GROUP_BEYONDMIMIC = "beyondmimic_imitation"
ACTION_BALL_REWARD_GROUP_HOPE_TASK = "hope_hit_landing_task"
ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY = "immutable_safety"
ACTION_BALL_REWARD_GROUP_ORDER = (
    ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY,
    ACTION_BALL_REWARD_GROUP_BEYONDMIMIC,
    ACTION_BALL_REWARD_GROUP_HOPE_TASK,
    ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()
ACTION_BALL_HARD_SAFETY_TERMINATION_TERMS = (
    "base_fell_tilt",
    "base_too_low",
    "joint_actual_forbidden",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
ACTION_BALL_REFERENCE_ENVELOPE_TERMINATION_TERMS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
)


def _taxonomy_specs(group, expected_weight_sign, rows, *, adjustability):
    return {
        name: {
            "group": group,
            "source": source,
            "expected_weight_sign": expected_weight_sign,
            "expected_contribution": (
                "zero" if name.endswith("_probe") else expected_weight_sign
            ),
            "adjustability": adjustability,
            "causal_axis": axis,
        }
        for name, source, axis in rows
    }


_ACTION_BALL_REWARD_TERM_TAXONOMY = {}
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY,
        "positive",
        (
            ("upright_exp", "MJLab", "base_uprightness"),
            ("hold_ready", "HOPE balance adaptation", "ready_state_quality"),
            ("base_decel", "HOPE balance adaptation", "base_deceleration_error"),
            ("post_strike_brake", "HOPE balance adaptation", "post_strike_speed"),
            ("hold_heading", "HOPE balance adaptation", "ready_heading_error"),
        ),
        adjustability="preregistered_scientific",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY,
        "negative",
        (
            ("pre_strike_foot_slip", "HOPE stability", "prestrike_foot_slip"),
            ("joint_torques", "BeyondMimic regularization", "joint_torque"),
            ("action_rate_l2", "BeyondMimic regularization", "action_rate"),
            ("action_rate_clamped", "MJLab-aligned", "clamped_action_rate"),
            ("action_acc_l2", "MJLab", "action_acceleration"),
            ("undesired_contacts", "BeyondMimic regularization", "undesired_contact"),
            ("foot_slip_sq", "MJLab", "stance_foot_slip"),
            ("foot_velocity", "HOPE stability", "foot_speed"),
            ("foot_drag", "HOPE stability", "foot_drag"),
            ("foot_soft_landing", "MJLab", "foot_landing_impact"),
            ("foot_clearance", "MJLab", "swing_foot_clearance"),
            ("arm_overreach", "HOPE stability", "arm_overreach"),
            ("prestrike_waist_twist", "HOPE stability", "prestrike_waist_twist"),
            ("prestrike_upright", "HOPE stability", "prestrike_tilt"),
            ("strike_upright", "HOPE stability", "strike_tilt"),
            ("strike_ang_vel", "HOPE stability", "strike_base_angular_speed"),
            ("strike_foot_vel", "HOPE stability", "strike_foot_speed"),
            ("strike_vbob", "HOPE stability", "strike_vertical_bob"),
            ("hit_unstable_support", "PACE/MJLab-aligned", "strike_support"),
            ("arm_torque_saturation", "HOPE sim2real", "arm_torque_saturation"),
            ("upright", "BeyondMimic regularization", "base_tilt"),
            ("base_ang_vel_xy", "BeyondMimic regularization", "base_roll_pitch_rate"),
            ("base_lin_vel_z", "BeyondMimic regularization", "base_vertical_speed"),
            ("joint_vel", "BeyondMimic regularization", "joint_speed"),
            ("foot_orientation", "HOPE stability", "foot_orientation_error"),
            (
                "processed_qdes_slew_hinge",
                "HOPE recovery stability",
                "recovery_qdes_slew",
            ),
            (
                "lower_body_stability_bundle",
                "HOPE stability",
                "lower_body_stability_debt",
            ),
            (
                "post_swing_settle_debt",
                "HOPE recovery stability",
                "post_swing_settle_debt",
            ),
        ),
        adjustability="preregistered_scientific",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY,
        "positive",
        (
            (
                "base_decel_activation_probe",
                "HOPE balance diagnostic",
                "base_deceleration_error",
            ),
            (
                "processed_qdes_slew_hinge_probe",
                "HOPE recovery diagnostic",
                "recovery_qdes_slew",
            ),
            (
                "lower_body_stability_bundle_probe",
                "HOPE stability diagnostic",
                "lower_body_stability_debt",
            ),
            (
                "post_swing_settle_debt_probe",
                "HOPE recovery diagnostic",
                "post_swing_settle_debt",
            ),
        ),
        adjustability="diagnostic_only",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_BEYONDMIMIC,
        "positive",
        (
            ("motion_global_anchor_pos", "BeyondMimic", "anchor_position_error"),
            ("motion_global_anchor_ori", "BeyondMimic", "anchor_orientation_error"),
            ("motion_body_pos", "BeyondMimic", "body_position_error"),
            ("motion_body_ori", "BeyondMimic", "body_orientation_error"),
            ("motion_body_lin_vel", "BeyondMimic", "body_linear_velocity_error"),
            ("motion_body_ang_vel", "BeyondMimic", "body_angular_velocity_error"),
            (
                "lower_body_pose_imitation",
                "BeyondMimic-derived",
                "lower_body_pose_error",
            ),
        ),
        adjustability="preregistered_scientific",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_BEYONDMIMIC,
        "positive",
        (
            (
                "lower_body_pose_imitation_probe",
                "BeyondMimic-derived diagnostic",
                "lower_body_pose_error",
            ),
        ),
        adjustability="diagnostic_only",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_HOPE_TASK,
        "positive",
        (
            ("racket_position", "HOPE", "racket_position_error"),
            ("racket_velocity", "HOPE", "racket_velocity_error"),
            ("racket_normal", "HOPE", "signed_racket_face_error"),
            ("base_position", "HOPE/HITTER", "base_task_position_error"),
            ("racket_progress", "HOPE/HITTER", "racket_target_progress"),
            ("racket_strike_success", "HOPE", "joint_strike_quality"),
            ("strike_capture_bonus", "HOPE", "strike_capture"),
            ("virtual_pass_net", "HOPE", "virtual_net_clearance"),
            ("virtual_landing", "HOPE", "virtual_landing_error"),
            ("virtual_spin", "HOPE", "virtual_spin_error"),
        ),
        adjustability="preregistered_scientific",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_HOPE_TASK,
        "negative",
        (
            ("racket_guidance", "HOPE", "racket_position_error"),
            ("racket_face_guidance", "HOPE", "signed_racket_face_error"),
            (
                "racket_face_conditional_guidance",
                "HOPE",
                "conditional_signed_racket_face_error",
            ),
        ),
        adjustability="preregistered_scientific",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY,
        "negative",
        (
            ("death_penalty", "HOPE hard safety", "unsafe_termination"),
            ("table_hit_penalty", "HOPE hard safety", "table_contact"),
            ("joint_limit", "HOPE soft safety", "actual_joint_soft_limit"),
            ("qdes_limit_barrier", "HOPE soft safety", "qdes_joint_soft_limit"),
            (
                "joint_velocity_limit_hinge",
                "HOPE soft safety",
                "actual_joint_velocity_soft_limit",
            ),
            (
                "tracking_envelope",
                "BeyondMimic hard-envelope adaptation",
                "tracking_envelope_violation",
            ),
        ),
        adjustability="immutable_safety",
    )
)
_ACTION_BALL_REWARD_TERM_TAXONOMY.update(
    _taxonomy_specs(
        ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY,
        "positive",
        (
            (
                "qdes_limit_barrier_probe",
                "HOPE soft-safety diagnostic",
                "qdes_joint_soft_limit",
            ),
            (
                "actual_joint_limit_barrier_probe",
                "HOPE soft-safety diagnostic",
                "actual_joint_soft_limit",
            ),
            (
                "joint_velocity_limit_hinge_probe",
                "HOPE soft-safety diagnostic",
                "actual_joint_velocity_soft_limit",
            ),
        ),
        adjustability="diagnostic_only",
    )
)

ACTION_BALL_REWARD_GROUP_TAXONOMY_AUTHORITY = {
    "schema_version": ACTION_BALL_REWARD_GROUP_TAXONOMY_SCHEMA_VERSION,
    "group_order": list(ACTION_BALL_REWARD_GROUP_ORDER),
    "groups": {
        ACTION_BALL_REWARD_GROUP_MJLAB_STABILITY: {
            "human_name": "MJLab-derived balance and stability shaping",
            "layer": "scientific",
        },
        ACTION_BALL_REWARD_GROUP_BEYONDMIMIC: {
            "human_name": "BeyondMimic action imitation",
            "layer": "scientific",
        },
        ACTION_BALL_REWARD_GROUP_HOPE_TASK: {
            "human_name": "HOPE hit, landing, and task objectives",
            "layer": "scientific",
        },
        ACTION_BALL_REWARD_GROUP_IMMUTABLE_SAFETY: {
            "human_name": "immutable terminal and soft-limit safety",
            "layer": "safety",
        },
    },
    "terms": _ACTION_BALL_REWARD_TERM_TAXONOMY,
}
ACTION_BALL_REWARD_GROUP_TAXONOMY_AUTHORITY_SHA256 = hashlib.sha256(
    json.dumps(
        ACTION_BALL_REWARD_GROUP_TAXONOMY_AUTHORITY,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


class RewardRecipeError(ValueError):
    """The effective reward configuration cannot produce a stable receipt."""


class RewardRecipeMismatchError(RewardRecipeError):
    """The effective reward recipe does not match its expected SHA-256."""


class RewardActivationLedgerError(RuntimeError):
    """Runtime reward activation cannot be attested without guessing."""


def _get_member(value, name, default=_MISSING):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_rewards_node(cfg):
    """Accept an env cfg, ``task`` wrapper, or the rewards node itself."""

    rewards = _get_member(cfg, "rewards")
    if rewards is not _MISSING:
        if rewards is None:
            raise RewardRecipeError("cfg.rewards is None")
        return rewards

    task = _get_member(cfg, "task")
    if task is not _MISSING:
        rewards = _get_member(task, "rewards")
        if rewards is _MISSING:
            raise RewardRecipeError("cfg.task has no rewards node")
        if rewards is None:
            raise RewardRecipeError("cfg.task.rewards is None")
        return rewards

    return cfg


def _extract_environment_node(cfg):
    task = _get_member(cfg, "task")
    if task is not _MISSING and _get_member(task, "scene") is not _MISSING:
        return task
    return cfg


def _actuator_cfg_backend(value):
    """Return implicit|explicit|None without importing Isaac Lab."""

    identities = []
    for candidate in (value, _get_member(value, "class_type", None)):
        if candidate is None:
            continue
        cls = candidate if isinstance(candidate, type) else type(candidate)
        identities.append(
            "{}.{}".format(
                getattr(cls, "__module__", ""),
                getattr(cls, "__qualname__", getattr(cls, "__name__", "")),
            ).lower()
        )
    joined = " ".join(identities)
    if "implicitactuator" in joined:
        return "implicit"
    if any(
        marker in joined
        for marker in (
            "idealpdactuator",
            "dcmotor",
            "explicitactuator",
            "remotizedpdactuator",
        )
    ):
        return "explicit"
    return None


def disable_incompatible_backend_reward_terms(cfg):
    """Disable explicit-only Reward terms on a composed implicit A3 backend.

    Task YAML is applied after the environment config's ``__post_init__`` and
    can therefore resurrect a stale non-zero torque-saturation weight.  Reward
    receipt construction is the final pre-scene composition boundary used by
    training and by runtime hard-contract recapture, so the compatibility
    decision is made here and mutates the composed term to zero before
    RewardManager sees it.
    """

    rewards = _extract_rewards_node(cfg)
    term = _get_member(rewards, "arm_torque_saturation", None)
    if term is None:
        return ()
    weight = _get_member(term, "weight", 0.0)
    if (
        type(weight) not in (int, float)
        or isinstance(weight, bool)
        or not math.isfinite(float(weight))
    ):
        raise RewardRecipeError(
            "arm_torque_saturation weight must be finite before backend compatibility"
        )
    if float(weight) == 0.0:
        return ()

    env = _extract_environment_node(cfg)
    scene = _get_member(env, "scene", None)
    robot = None if scene is None else _get_member(scene, "robot", None)
    actuators = None if robot is None else _get_member(robot, "actuators", None)
    if actuators is None:
        # Dependency-light reward-only tools have no plant configuration and
        # cannot make a backend claim.  Formal training cfgs always have one.
        return ()
    if not isinstance(actuators, Mapping):
        raise RewardRecipeError("robot actuator cfgs must be a mapping")
    relevant = {
        str(name): actuator
        for name, actuator in actuators.items()
        if str(name) in {"arms", "waist"}
    }
    if set(relevant) != {"arms", "waist"}:
        raise RewardRecipeError(
            "arm_torque_saturation requires explicit arms+waist actuator cfgs"
        )
    backends = {
        name: _actuator_cfg_backend(actuator)
        for name, actuator in relevant.items()
    }
    if any(value is None for value in backends.values()):
        raise RewardRecipeError(
            "arm_torque_saturation actuator backend is unresolved: "
            f"{backends!r}"
        )
    if any(value == "implicit" for value in backends.values()):
        term.weight = 0.0
        return (
            {
                "name": "arm_torque_saturation",
                "status": "disabled_not_in_active_recipe",
                "reason": (
                    "ImplicitActuator does not expose a proven explicit "
                    "pre-clip demand through computed_torque"
                ),
                "actuator_backends": dict(sorted(backends.items())),
            },
        )
    if set(backends.values()) != {"explicit"}:
        raise RewardRecipeError(
            f"arm_torque_saturation backend mix is unsupported: {backends!r}"
        )
    return ()


def _object_items(value, *, context):
    if isinstance(value, Mapping):
        return list(value.items())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in dataclasses.fields(value)]
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise RewardRecipeError(
            f"{context} must be a mapping, dataclass, or attribute object"
        ) from exc
    return [(name, item) for name, item in attributes.items() if not name.startswith("_")]


def _callable_identity(func, *, term_name):
    if isinstance(func, str):
        if not func or func.strip() != func or any(char.isspace() for char in func):
            raise RewardRecipeError(
                f"reward term {term_name!r} has an invalid callable identity string"
            )
        return func

    # Bound methods and arbitrary callable instances can hide mutable instance
    # state which a module/qualname string would not capture.
    if inspect.ismethod(func) and getattr(func, "__self__", None) is not None:
        raise RewardRecipeError(
            f"reward term {term_name!r} uses a bound method; callable state is not stable"
        )
    if not (inspect.isfunction(func) or inspect.isbuiltin(func) or inspect.isclass(func)):
        raise RewardRecipeError(
            f"reward term {term_name!r} func must be a named function, class, or identity string"
        )

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        raise RewardRecipeError(
            f"reward term {term_name!r} callable has no stable module-qualified identity"
        )
    return f"{module}.{qualname}"


def reward_term_runtime_role(term_name, func):
    """Classify an active term without letting a probe masquerade as an objective.

    HOPE's measurement-only RewardManager terms use the explicit ``*_probe``
    convention for both the config key and the function name.  Requiring both
    sides makes the classification fail closed on a typo or a renamed callable:
    one suffix alone is not enough evidence to decide whether a term is allowed
    to influence the optimized objective.
    """

    if type(term_name) is not str or not term_name or term_name.strip() != term_name:
        raise RewardRecipeError("runtime reward term name must be a non-empty trimmed string")
    callable_identity = _callable_identity(func, term_name=term_name)
    name_is_probe = term_name.endswith("_probe")
    callable_is_probe = callable_identity.rsplit(".", 1)[-1].endswith("_probe")
    if name_is_probe != callable_is_probe:
        raise RewardRecipeError(
            f"reward term {term_name!r} has ambiguous probe identity: "
            f"callable={callable_identity!r}"
        )
    role = (
        REWARD_TERM_ROLE_DIAGNOSTIC_PROBE
        if name_is_probe
        else REWARD_TERM_ROLE_OBJECTIVE
    )
    return role, callable_identity


def _action_ball_taxonomy_weight_sign(weight):
    if weight > 0.0:
        return "positive"
    if weight < 0.0:
        return "negative"
    return "zero"


def build_action_ball_reward_group_taxonomy(recipe_terms):
    """Bind every active composed term to one authoritative Reward group.

    ``recipe_terms`` must be the already-composed active recipe, never a task
    YAML or nominal reward-pack table.  An active term absent from the authority
    map is rejected; silently filing it under ``other`` would recreate the
    historical "configured but not actually understood" failure mode.
    """

    if not isinstance(recipe_terms, Sequence) or isinstance(
        recipe_terms, (str, bytes, bytearray)
    ):
        raise RewardRecipeError("action-ball taxonomy requires a recipe term sequence")
    active = []
    seen = set()
    for index, term in enumerate(recipe_terms):
        if not isinstance(term, Mapping) or set(term) != {
            "name",
            "callable",
            "weight",
            "params",
        }:
            raise RewardRecipeError(
                f"action-ball taxonomy recipe term {index} has an invalid field set"
            )
        name = term["name"]
        if type(name) is not str or not name or name in seen:
            raise RewardRecipeError(
                "action-ball taxonomy requires unique non-empty term names"
            )
        seen.add(name)
        spec = _ACTION_BALL_REWARD_TERM_TAXONOMY.get(name)
        if spec is None:
            raise RewardRecipeError(
                f"active ActionBall Reward term {name!r} has no authoritative taxonomy"
            )
        weight = _plain_finite_float(
            term["weight"], context=f"taxonomy term {name!r} weight", nonzero=True
        )
        actual_sign = _action_ball_taxonomy_weight_sign(weight)
        if actual_sign != spec["expected_weight_sign"]:
            raise RewardRecipeError(
                f"active ActionBall Reward term {name!r} has {actual_sign} weight, "
                f"expected {spec['expected_weight_sign']}"
            )
        callable_identity = term["callable"]
        if type(callable_identity) is not str or not callable_identity:
            raise RewardRecipeError(
                f"active ActionBall Reward term {name!r} has no callable identity"
            )
        name_is_probe = name.endswith("_probe")
        callable_is_probe = callable_identity.rsplit(".", 1)[-1].endswith(
            "_probe"
        )
        if name_is_probe != callable_is_probe:
            raise RewardRecipeError(
                f"active ActionBall Reward term {name!r} has ambiguous probe identity"
            )
        role = (
            REWARD_TERM_ROLE_DIAGNOSTIC_PROBE
            if name_is_probe
            else REWARD_TERM_ROLE_OBJECTIVE
        )
        recipe_term_sha256 = hashlib.sha256(
            json.dumps(
                dict(term),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        active.append(
            {
                "name": name,
                "callable": callable_identity,
                "weight": weight,
                "role": role,
                "recipe_term_sha256": recipe_term_sha256,
                **spec,
            }
        )
    active.sort(key=lambda item: item["name"])
    document = {
        "schema_version": ACTION_BALL_REWARD_GROUP_TAXONOMY_SCHEMA_VERSION,
        "authority_sha256": (
            ACTION_BALL_REWARD_GROUP_TAXONOMY_AUTHORITY_SHA256
        ),
        "group_order": list(ACTION_BALL_REWARD_GROUP_ORDER),
        "active_terms": active,
    }
    document["sha256"] = hashlib.sha256(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return document


def _linear_quantile(values, probability):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise RewardActivationLedgerError("Reward group quantile input is non-finite")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return _normalized_output_float(
            ordered[lower], context=f"Reward group p{probability}"
        )
    fraction = position - lower
    return _normalized_output_float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction,
        context=f"Reward group p{probability}",
    )


def validate_action_ball_reward_causal_probes(
    recipe_terms, probes, *, step_dt=ACTION_BALL_ADOPTED_STEP_DT_S
):
    """Validate one callable-produced, single-axis worsening probe per objective.

    This validator deliberately does not fabricate probes from weights.  Each
    row must bind the actual callable, two distinct state receipts, one frozen
    context receipt, and exactly the taxonomy's causal axis.  Worsening that
    axis must make the weighted contribution strictly lower and non-zero.
    """

    taxonomy = build_action_ball_reward_group_taxonomy(recipe_terms)
    step_dt = _plain_finite_float(
        step_dt, context="causal probe step_dt", positive=True
    )
    if not isinstance(probes, Sequence) or isinstance(
        probes, (str, bytes, bytearray)
    ):
        raise RewardRecipeError("causal probes must be a sequence")
    by_name = {}
    for row in probes:
        if not isinstance(row, Mapping) or set(row) != {
            "term_name",
            "callable",
            "changed_axes",
            "frozen_context_sha256",
            "baseline_state_sha256",
            "worsened_state_sha256",
            "baseline_raw",
            "worsened_raw",
        }:
            raise RewardRecipeError("causal probe row has an invalid field set")
        name = row["term_name"]
        if type(name) is not str or not name or name in by_name:
            raise RewardRecipeError("causal probe term names must be unique")
        by_name[name] = row
    expected = {
        row["name"]: row
        for row in taxonomy["active_terms"]
        if row["role"] == REWARD_TERM_ROLE_OBJECTIVE
    }
    if set(by_name) != set(expected):
        raise RewardRecipeError(
            "causal probes must cover every active objective term exactly once"
        )
    results = []
    for name in sorted(expected):
        spec = expected[name]
        row = by_name[name]
        if (
            row["callable"] != spec["callable"]
            or row["changed_axes"] != [spec["causal_axis"]]
        ):
            raise RewardRecipeError(
                f"causal probe {name!r} does not isolate its authoritative axis/callable"
            )
        for field in (
            "frozen_context_sha256",
            "baseline_state_sha256",
            "worsened_state_sha256",
        ):
            if type(row[field]) is not str or _SHA256_RE.fullmatch(row[field]) is None:
                raise RewardRecipeError(
                    f"causal probe {name!r}.{field} must be SHA-256"
                )
        if row["baseline_state_sha256"] == row["worsened_state_sha256"]:
            raise RewardRecipeError(
                f"causal probe {name!r} baseline/worsened states are identical"
            )
        baseline_raw = _plain_finite_float(
            row["baseline_raw"], context=f"causal probe {name!r}.baseline_raw"
        )
        worsened_raw = _plain_finite_float(
            row["worsened_raw"], context=f"causal probe {name!r}.worsened_raw"
        )
        baseline_weighted = baseline_raw * spec["weight"] * step_dt
        worsened_weighted = worsened_raw * spec["weight"] * step_dt
        delta = worsened_weighted - baseline_weighted
        if not math.isfinite(delta) or delta >= 0.0:
            raise RewardRecipeError(
                f"causal probe {name!r} worsening did not produce a strict negative "
                "weighted delta"
            )
        results.append(
            {
                "term_name": name,
                "group": spec["group"],
                "causal_axis": spec["causal_axis"],
                "baseline_weighted": _normalized_output_float(
                    baseline_weighted,
                    context=f"causal probe {name!r}.baseline_weighted",
                ),
                "worsened_weighted": _normalized_output_float(
                    worsened_weighted,
                    context=f"causal probe {name!r}.worsened_weighted",
                ),
                "weighted_delta": _normalized_output_float(
                    delta, context=f"causal probe {name!r}.weighted_delta"
                ),
                "frozen_context_sha256": row["frozen_context_sha256"],
                "baseline_state_sha256": row["baseline_state_sha256"],
                "worsened_state_sha256": row["worsened_state_sha256"],
            }
        )
    report = {
        "schema_version": 1,
        "taxonomy_sha256": taxonomy["sha256"],
        "step_dt_s": step_dt,
        "coverage": "every_active_objective_exactly_once",
        "probes": results,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return report


def _type_identity(value):
    cls = value if isinstance(value, type) else type(value)
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise RewardRecipeError(f"{cls!r} has no stable type identity")
    if "<locals>" in qualname:
        raise RewardRecipeError(f"{module}.{qualname} is a local type and is not stable")
    return f"{module}.{qualname}"


def _stable_json_value(value, *, context, active_ids):
    """Normalize supported values without ever consulting an unstable repr."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RewardRecipeError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, enum.Enum):
        return {"__enum__": f"{_type_identity(value)}.{value.name}"}
    if isinstance(value, slice):
        return {
            "__slice__": [
                _stable_json_value(value.start, context=f"{context}.start", active_ids=active_ids),
                _stable_json_value(value.stop, context=f"{context}.stop", active_ids=active_ids),
                _stable_json_value(value.step, context=f"{context}.step", active_ids=active_ids),
            ]
        }

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in active_ids:
            raise RewardRecipeError(f"{context} contains a reference cycle")
        active_ids.add(object_id)
        try:
            normalized = {}
            for key, item in value.items():
                if type(key) is not str or not key:
                    raise RewardRecipeError(f"{context} contains a non-string or empty key")
                normalized[key] = _stable_json_value(
                    item, context=f"{context}.{key}", active_ids=active_ids
                )
            return normalized
        finally:
            active_ids.remove(object_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        object_id = id(value)
        if object_id in active_ids:
            raise RewardRecipeError(f"{context} contains a reference cycle")
        active_ids.add(object_id)
        try:
            return [
                _stable_json_value(
                    item, context=f"{context}[{index}]", active_ids=active_ids
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_ids.remove(object_id)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        object_id = id(value)
        if object_id in active_ids:
            raise RewardRecipeError(f"{context} contains a reference cycle")
        active_ids.add(object_id)
        try:
            fields = {
                field.name: _stable_json_value(
                    getattr(value, field.name),
                    context=f"{context}.{field.name}",
                    active_ids=active_ids,
                )
                for field in dataclasses.fields(value)
            }
            return {"__config_type__": _type_identity(value), "fields": fields}
        finally:
            active_ids.remove(object_id)

    raise RewardRecipeError(
        f"{context} has unsupported type {_type_identity(value)}; "
        "stable reward receipts do not fall back to repr()"
    )


def _normalized_term(term_name, term):
    func = _get_member(term, "func")
    weight = _get_member(term, "weight")
    params = _get_member(term, "params", {})

    if func is _MISSING:
        raise RewardRecipeError(f"reward term {term_name!r} has no func")
    if weight is _MISSING:
        raise RewardRecipeError(f"reward term {term_name!r} has no weight")
    if type(weight) not in (int, float) or isinstance(weight, bool):
        raise RewardRecipeError(f"reward term {term_name!r} weight must be a finite number")
    normalized_weight = float(weight)
    if not math.isfinite(normalized_weight):
        raise RewardRecipeError(f"reward term {term_name!r} weight must be finite")
    if normalized_weight == 0.0:
        return None

    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise RewardRecipeError(f"reward term {term_name!r} params must be a mapping")

    return {
        "name": term_name,
        "callable": _callable_identity(func, term_name=term_name),
        "weight": normalized_weight,
        "params": _stable_json_value(
            params, context=f"reward term {term_name!r} params", active_ids=set()
        ),
    }


def effective_reward_recipe(cfg):
    """Return the normalized, hashable recipe payload from a composed cfg."""

    disable_incompatible_backend_reward_terms(cfg)
    rewards = _extract_rewards_node(cfg)
    terms = []
    seen_names = set()
    for raw_name, term in _object_items(rewards, context="rewards node"):
        if type(raw_name) is not str or not raw_name or raw_name.strip() != raw_name:
            raise RewardRecipeError("reward term names must be non-empty strings without padding")
        if raw_name in seen_names:
            raise RewardRecipeError(f"duplicate reward term name {raw_name!r}")
        seen_names.add(raw_name)
        if term is None:
            continue
        normalized = _normalized_term(raw_name, term)
        if normalized is not None:
            terms.append(normalized)
    terms.sort(key=lambda item: item["name"])
    return {
        "schema_version": EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION,
        "terms": terms,
    }


def canonical_effective_reward_recipe_json(recipe):
    """Encode a normalized recipe payload as stable, compact JSON."""

    if not isinstance(recipe, Mapping):
        raise RewardRecipeError("recipe must be a mapping")
    if set(recipe) != {"schema_version", "terms"}:
        raise RewardRecipeError("recipe must contain exactly schema_version and terms")
    try:
        return json.dumps(
            recipe,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RewardRecipeError("recipe is not canonical JSON data") from exc


def effective_reward_recipe_sha256(cfg):
    """Return SHA-256 of the effective recipe's canonical JSON bytes."""

    recipe = effective_reward_recipe(cfg)
    encoded = canonical_effective_reward_recipe_json(recipe).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_expected_reward_recipe_sha256(actual_sha256, expected_sha256):
    """Fail loudly unless an expected SHA-256 exactly matches the actual recipe."""

    if not isinstance(actual_sha256, str) or not _SHA256_RE.fullmatch(actual_sha256):
        raise RewardRecipeError("actual reward recipe SHA-256 must be 64 lowercase hex characters")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise RewardRecipeError("expected reward recipe SHA-256 must be 64 lowercase hex characters")
    if actual_sha256 != expected_sha256:
        raise RewardRecipeMismatchError(
            "effective reward recipe SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def build_effective_reward_receipt(cfg, expected_sha256=None):
    """Build the normalized receipt and optionally enforce a preregistered SHA."""

    recipe = effective_reward_recipe(cfg)
    canonical_json = canonical_effective_reward_recipe_json(recipe)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if expected_sha256 is not None:
        validate_expected_reward_recipe_sha256(digest, expected_sha256)
    return {
        "schema_version": recipe["schema_version"],
        "terms": recipe["terms"],
        "sha256": digest,
    }


class _TorchRewardActivationTensorOps:
    """Small Torch facade used only when a runtime ledger is constructed."""

    def __init__(self, torch_module):
        self._torch = torch_module

    def is_tensor(self, value):
        return self._torch.is_tensor(value)

    @staticmethod
    def detach(value):
        return value.detach()

    def as_tensor_like(self, values, like):
        return self._torch.as_tensor(values, dtype=like.dtype, device=like.device)

    def isfinite(self, value):
        return self._torch.isfinite(value)

    def logical_not(self, value):
        return self._torch.logical_not(value)

    def greater(self, left, right):
        return self._torch.gt(left, right)

    def abs(self, value):
        return self._torch.abs(value)

    def sum(self, value, axis=None):
        return self._torch.sum(value) if axis is None else self._torch.sum(value, dim=axis)

    def count_nonzero(self, value, axis=None):
        return (
            self._torch.count_nonzero(value)
            if axis is None
            else self._torch.count_nonzero(value, dim=axis)
        )

    def max(self, value, axis=None):
        return (
            self._torch.max(value)
            if axis is None
            else self._torch.max(value, dim=axis).values
        )

    def maximum(self, left, right):
        return self._torch.maximum(left, right)

    def stack(self, values, axis=0):
        return self._torch.stack(tuple(values), dim=axis)

    @staticmethod
    def to_host_list(value):
        return value.detach().cpu().tolist()

    @staticmethod
    def to_host_scalar(value):
        return value.detach().cpu().item()


def _default_reward_activation_tensor_ops():
    # Importing Torch here, rather than at module import, preserves the host-only
    # effective-recipe tooling and tests.
    try:
        import torch
    except ImportError as exc:
        raise RewardActivationLedgerError(
            "runtime reward activation requires Torch or an explicit tensor_ops adapter"
        ) from exc
    return _TorchRewardActivationTensorOps(torch)


def _plain_finite_float(value, *, context, positive=False, nonzero=False):
    if type(value) not in (int, float) or isinstance(value, bool):
        raise RewardActivationLedgerError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RewardActivationLedgerError(f"{context} must be finite")
    if positive and result <= 0.0:
        raise RewardActivationLedgerError(f"{context} must be positive")
    if nonzero and result == 0.0:
        raise RewardActivationLedgerError(f"{context} must be non-zero")
    return result


def _normalized_output_float(value, *, context):
    result = _plain_finite_float(value, context=context)
    # Canonicalize negative zero so equivalent probe receipts have identical
    # bytes across tensor backends.
    return 0.0 if result == 0.0 else result


class EffectiveRewardActivationLedger:
    """Accumulate verified per-term reward evidence across one PPO rollout.

    This is intentionally a narrow adapter for the Isaac Lab cache contract
    used by ActionBall and UpperSafe:

    * ``active_terms`` and ``get_term_cfg(name)`` bind names/functions/weights;
    * ``_step_reward[:, i]`` is ``raw_i * weight_i`` after each ``env.step``;
    * ``_reward_buf`` is ``sum_i(raw_i * weight_i * step_dt)``.

    The last identity is checked for every environment sample without moving
    tensors to the host.  Host conversion happens only in ``finish_update``.
    Eligibility is never inferred from a zero/nonzero reward: a term-specific
    opportunity mask is a different fact, so every generic entry records it as
    unknown.
    """

    _SUPPORTED_TASK_KINDS = ("action_ball", "upper_safe")

    def __init__(
        self,
        env,
        *,
        task_kind,
        expected_environment_step_count,
        tensor_ops=None,
        closure_rtol=1.0e-5,
        closure_atol=1.0e-7,
    ):
        if task_kind not in self._SUPPORTED_TASK_KINDS:
            raise RewardActivationLedgerError(
                f"runtime reward activation is not verified for task_kind={task_kind!r}"
            )
        if (
            type(expected_environment_step_count) is not int
            or expected_environment_step_count <= 0
        ):
            raise RewardActivationLedgerError(
                "expected_environment_step_count must be a positive plain integer"
            )
        self._env = env
        self._task_kind = task_kind
        self._expected_step_count = expected_environment_step_count
        self._ops = (
            tensor_ops
            if tensor_ops is not None
            else _default_reward_activation_tensor_ops()
        )
        self._closure_rtol = _plain_finite_float(
            closure_rtol, context="reward closure rtol", positive=True
        )
        self._closure_atol = _plain_finite_float(
            closure_atol, context="reward closure atol", positive=True
        )

        manager = getattr(env, "reward_manager", None)
        if manager is None:
            raise RewardActivationLedgerError(
                f"{task_kind} runtime reward activation requires reward_manager"
            )
        self._manager = manager
        get_term_cfg = getattr(manager, "get_term_cfg", None)
        if not callable(get_term_cfg):
            raise RewardActivationLedgerError(
                "verified RewardManager adapter requires get_term_cfg(name)"
            )
        self._get_term_cfg = get_term_cfg

        raw_names = getattr(manager, "active_terms", None)
        if not isinstance(raw_names, (list, tuple)):
            raise RewardActivationLedgerError(
                "verified RewardManager adapter requires an ordered active_terms list"
            )
        names = tuple(raw_names)
        if (
            not names
            or any(type(name) is not str or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise RewardActivationLedgerError(
                "RewardManager active_terms must be unique non-empty strings"
            )
        self._all_names = names

        step_reward = getattr(manager, "_step_reward", None)
        if not self._ops.is_tensor(step_reward):
            raise RewardActivationLedgerError(
                "verified RewardManager adapter requires tensor _step_reward"
            )
        shape = tuple(int(size) for size in getattr(step_reward, "shape", ()))
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != len(names):
            raise RewardActivationLedgerError(
                "RewardManager _step_reward shape must be [num_envs, len(active_terms)]"
            )
        self._num_envs = shape[0]

        reward_buf = getattr(manager, "_reward_buf", None)
        if not self._ops.is_tensor(reward_buf):
            raise RewardActivationLedgerError(
                "verified RewardManager adapter requires tensor _reward_buf"
            )
        if tuple(int(size) for size in getattr(reward_buf, "shape", ())) != (
            self._num_envs,
        ):
            raise RewardActivationLedgerError(
                "RewardManager _reward_buf shape must be [num_envs]"
            )

        self._step_dt = _plain_finite_float(
            getattr(env, "step_dt", None), context="environment step_dt", positive=True
        )
        self._all_term_bindings = []
        active_metadata = []
        runtime_recipe_terms = []
        for index, name in enumerate(names):
            cfg = get_term_cfg(name)
            weight = _plain_finite_float(
                getattr(cfg, "weight", None),
                context=f"reward term {name!r} weight",
            )
            func = getattr(cfg, "func", _MISSING)
            if func is _MISSING:
                raise RewardActivationLedgerError(
                    f"reward term {name!r} has no runtime func"
                )
            normalized_term = _normalized_term(name, cfg)
            binding = {
                "name": name,
                "weight": weight,
                "func": func,
                "recipe_term": normalized_term,
            }
            self._all_term_bindings.append(binding)
            if weight != 0.0:
                if normalized_term is None:
                    raise RewardActivationLedgerError(
                        f"active reward term {name!r} normalized to inactive"
                    )
                try:
                    role, callable_identity = reward_term_runtime_role(name, func)
                except RewardRecipeError as exc:
                    raise RewardActivationLedgerError(str(exc)) from exc
                active_metadata.append(
                    {
                        "index": index,
                        "name": name,
                        "weight": weight,
                        "role": role,
                        "callable": callable_identity,
                        "recipe_term": normalized_term,
                    }
                )
                runtime_recipe_terms.append(normalized_term)
        if not active_metadata:
            raise RewardActivationLedgerError(
                f"{task_kind} has no finite non-zero runtime reward terms"
            )
        self._active_metadata = tuple(active_metadata)
        runtime_recipe_terms.sort(key=lambda item: item["name"])
        runtime_recipe = {
            "schema_version": EFFECTIVE_REWARD_RECIPE_SCHEMA_VERSION,
            "terms": runtime_recipe_terms,
        }
        self._recipe_sha256 = hashlib.sha256(
            canonical_effective_reward_recipe_json(runtime_recipe).encode("utf-8")
        ).hexdigest()
        self._recipe_term_sha256 = {
            item["name"]: hashlib.sha256(
                json.dumps(
                    item,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            for item in runtime_recipe_terms
        }
        self._active_indices = tuple(item["index"] for item in active_metadata)
        self._inactive_indices = tuple(
            index
            for index in range(len(self._all_term_bindings))
            if index not in self._active_indices
        )
        self._weight_vector = self._ops.as_tensor_like(
            [item["weight"] for item in active_metadata], step_reward
        )

        self._last_finished_update = None
        self._pending_update = None
        self._update_start_common_step_counter = self._read_common_step_counter()
        self._reset_update_accumulators()

    def _read_common_step_counter(self):
        value = getattr(self._env, "common_step_counter", None)
        if type(value) is not int or value < 0:
            raise RewardActivationLedgerError(
                "runtime reward activation requires a nonnegative plain "
                "environment common_step_counter"
            )
        return value

    def _reset_update_accumulators(self):
        self._environment_step_count = 0
        self._observed_sample_count = 0
        self._weighted_sums = None
        self._raw_sums = None
        self._nonzero_counts = None
        self._raw_recomposition_violation_counts = None
        self._raw_recomposition_max_abs_errors = None
        self._active_nonfinite_counts = None
        self._cache_nonfinite_count = None
        self._reward_buf_nonfinite_count = None
        self._total_closure_violation_count = None
        self._total_closure_max_abs_error = None
        self._total_reward_sum = None

    @staticmethod
    def _accumulate(current, value):
        return value if current is None else current + value

    def _validate_runtime_bindings(self, step_reward):
        raw_names = getattr(self._manager, "active_terms", None)
        if not isinstance(raw_names, (list, tuple)) or tuple(raw_names) != self._all_names:
            raise RewardActivationLedgerError(
                "RewardManager active_terms changed during a PPO rollout"
            )
        shape = tuple(int(size) for size in getattr(step_reward, "shape", ()))
        if shape != (self._num_envs, len(self._all_names)):
            raise RewardActivationLedgerError(
                "RewardManager _step_reward shape changed during a PPO rollout"
            )
        for binding in self._all_term_bindings:
            cfg = self._get_term_cfg(binding["name"])
            current_weight = _plain_finite_float(
                getattr(cfg, "weight", None),
                context=f"reward term {binding['name']!r} runtime weight",
            )
            if current_weight != binding["weight"]:
                raise RewardActivationLedgerError(
                    f"reward term {binding['name']!r} weight changed during a PPO rollout"
                )
            if getattr(cfg, "func", _MISSING) is not binding["func"]:
                raise RewardActivationLedgerError(
                    f"reward term {binding['name']!r} callable changed during a PPO rollout"
                )
            if _normalized_term(binding["name"], cfg) != binding["recipe_term"]:
                raise RewardActivationLedgerError(
                    f"reward term {binding['name']!r} parameters changed during a PPO rollout"
                )

    def observe_after_environment_step(self):
        """Book the post-``env.step`` RewardManager cache without host sync."""

        if self._pending_update is not None:
            raise RewardActivationLedgerError(
                "runtime reward activation has an unacknowledged prepared update"
            )
        step_reward = getattr(self._manager, "_step_reward", None)
        reward_buf = getattr(self._manager, "_reward_buf", None)
        if not self._ops.is_tensor(step_reward) or not self._ops.is_tensor(reward_buf):
            raise RewardActivationLedgerError(
                "RewardManager reward cache disappeared during a PPO rollout"
            )
        self._validate_runtime_bindings(step_reward)
        if tuple(int(size) for size in getattr(reward_buf, "shape", ())) != (
            self._num_envs,
        ):
            raise RewardActivationLedgerError(
                "RewardManager _reward_buf shape changed during a PPO rollout"
            )

        step_reward = self._ops.detach(step_reward)
        reward_buf = self._ops.detach(reward_buf)
        if self._inactive_indices:
            inactive_nonzero = self._ops.count_nonzero(
                step_reward[:, list(self._inactive_indices)]
            )
            if self._host_nonnegative_int(
                inactive_nonzero, context="inactive reward cache nonzero count"
            ):
                raise RewardActivationLedgerError(
                    "zero-weight RewardManager cache column produced a non-zero value"
                )
        active_rate = step_reward[:, list(self._active_indices)]
        weighted = active_rate * self._step_dt
        raw = weighted / (self._weight_vector * self._step_dt)
        recomposed = raw * self._weight_vector * self._step_dt
        recompose_error = self._ops.abs(recomposed - weighted)
        recompose_tolerance = (
            self._ops.abs(weighted) * self._closure_rtol + self._closure_atol
        )

        self._weighted_sums = self._accumulate(
            self._weighted_sums, self._ops.sum(weighted, axis=0)
        )
        self._raw_sums = self._accumulate(
            self._raw_sums, self._ops.sum(raw, axis=0)
        )
        self._nonzero_counts = self._accumulate(
            self._nonzero_counts, self._ops.count_nonzero(weighted, axis=0)
        )
        self._active_nonfinite_counts = self._accumulate(
            self._active_nonfinite_counts,
            self._ops.count_nonzero(
                self._ops.logical_not(self._ops.isfinite(active_rate)), axis=0
            ),
        )
        self._raw_recomposition_violation_counts = self._accumulate(
            self._raw_recomposition_violation_counts,
            self._ops.count_nonzero(
                self._ops.greater(recompose_error, recompose_tolerance), axis=0
            ),
        )
        step_recompose_max = self._ops.max(recompose_error, axis=0)
        self._raw_recomposition_max_abs_errors = (
            step_recompose_max
            if self._raw_recomposition_max_abs_errors is None
            else self._ops.maximum(
                self._raw_recomposition_max_abs_errors, step_recompose_max
            )
        )

        # Verify the private cache semantics against the manager's actual output.
        # Summing every column also catches a stale zero-weight column.
        all_weighted = step_reward * self._step_dt
        cache_total = self._ops.sum(all_weighted, axis=1)
        closure_error = self._ops.abs(cache_total - reward_buf)
        closure_scale = self._ops.sum(self._ops.abs(all_weighted), axis=1)
        closure_tolerance = (
            closure_scale * self._closure_rtol + self._closure_atol
        )
        self._cache_nonfinite_count = self._accumulate(
            self._cache_nonfinite_count,
            self._ops.count_nonzero(
                self._ops.logical_not(self._ops.isfinite(step_reward))
            ),
        )
        self._reward_buf_nonfinite_count = self._accumulate(
            self._reward_buf_nonfinite_count,
            self._ops.count_nonzero(
                self._ops.logical_not(self._ops.isfinite(reward_buf))
            ),
        )
        self._total_closure_violation_count = self._accumulate(
            self._total_closure_violation_count,
            self._ops.count_nonzero(
                self._ops.greater(closure_error, closure_tolerance)
            ),
        )
        step_closure_max = self._ops.max(closure_error)
        self._total_closure_max_abs_error = (
            step_closure_max
            if self._total_closure_max_abs_error is None
            else self._ops.maximum(
                self._total_closure_max_abs_error, step_closure_max
            )
        )
        self._total_reward_sum = self._accumulate(
            self._total_reward_sum, self._ops.sum(reward_buf)
        )

        self._environment_step_count += 1
        self._observed_sample_count += self._num_envs

    def _host_vector(self, value, *, context, integer=False):
        if value is None:
            raise RewardActivationLedgerError(
                f"{context} is unavailable because no environment step was observed"
            )
        values = self._ops.to_host_list(value)
        if not isinstance(values, list) or len(values) != len(self._active_metadata):
            raise RewardActivationLedgerError(f"{context} has an unexpected tensor shape")
        if integer:
            normalized = []
            for item in values:
                if type(item) not in (int, float) or int(item) != item or int(item) < 0:
                    raise RewardActivationLedgerError(
                        f"{context} contains a nonnegative-integer violation"
                    )
                normalized.append(int(item))
            return normalized
        return [
            _normalized_output_float(item, context=f"{context}[{index}]")
            for index, item in enumerate(values)
        ]

    def _host_nonnegative_int(self, value, *, context):
        item = self._ops.to_host_scalar(value)
        if type(item) not in (int, float) or int(item) != item or int(item) < 0:
            raise RewardActivationLedgerError(
                f"{context} must be a nonnegative integer"
            )
        return int(item)

    def _host_float(self, value, *, context):
        return _normalized_output_float(
            self._ops.to_host_scalar(value), context=context
        )

    def prepare_update(self, ppo_update):
        """Freeze and validate one update without consuming its accumulators."""

        if type(ppo_update) is not int or ppo_update < 0:
            raise RewardActivationLedgerError(
                "ppo_update must be a nonnegative plain integer"
            )
        if self._pending_update is not None:
            raise RewardActivationLedgerError(
                "runtime reward activation already has a prepared update"
            )
        if self._last_finished_update is not None and ppo_update != (
            self._last_finished_update + 1
        ):
            raise RewardActivationLedgerError(
                "PPO update sequence is not contiguous in runtime reward ledger"
            )
        if self._environment_step_count != self._expected_step_count:
            raise RewardActivationLedgerError(
                "runtime reward ledger observed "
                f"{self._environment_step_count} env.step calls for PPO update "
                f"{ppo_update}, expected {self._expected_step_count}"
            )
        end_common_step_counter = self._read_common_step_counter()
        if (
            end_common_step_counter - self._update_start_common_step_counter
            != self._environment_step_count
        ):
            raise RewardActivationLedgerError(
                "environment common_step_counter delta does not equal the exact "
                "runtime reward ledger step count"
            )

        weighted_sums = self._host_vector(
            self._weighted_sums, context="weighted sums"
        )
        raw_sums = self._host_vector(self._raw_sums, context="raw sums")
        nonzero_counts = self._host_vector(
            self._nonzero_counts, context="nonzero counts", integer=True
        )
        nonfinite_counts = self._host_vector(
            self._active_nonfinite_counts,
            context="active nonfinite counts",
            integer=True,
        )
        raw_violation_counts = self._host_vector(
            self._raw_recomposition_violation_counts,
            context="raw recomposition violation counts",
            integer=True,
        )
        raw_max_errors = self._host_vector(
            self._raw_recomposition_max_abs_errors,
            context="raw recomposition max errors",
        )
        cache_nonfinite_count = self._host_nonnegative_int(
            self._cache_nonfinite_count, context="reward cache nonfinite count"
        )
        reward_buf_nonfinite_count = self._host_nonnegative_int(
            self._reward_buf_nonfinite_count,
            context="reward buffer nonfinite count",
        )
        closure_violation_count = self._host_nonnegative_int(
            self._total_closure_violation_count,
            context="total reward closure violation count",
        )
        closure_max_error = self._host_float(
            self._total_closure_max_abs_error,
            context="total reward closure max error",
        )
        total_reward_sum = self._host_float(
            self._total_reward_sum, context="total reward sum"
        )

        if cache_nonfinite_count or reward_buf_nonfinite_count:
            raise RewardActivationLedgerError(
                "non-finite value observed in RewardManager runtime cache"
            )
        if closure_violation_count:
            raise RewardActivationLedgerError(
                "RewardManager _step_reward does not close to _reward_buf under "
                "raw*weight*step_dt semantics"
            )
        if any(nonfinite_counts):
            raise RewardActivationLedgerError(
                "non-finite value observed in an active reward term"
            )
        if any(raw_violation_counts):
            raise RewardActivationLedgerError(
                "raw reward recovery failed weighted=raw*weight*step_dt validation"
            )

        terms = []
        for index, metadata in enumerate(self._active_metadata):
            if (
                metadata["role"] == REWARD_TERM_ROLE_DIAGNOSTIC_PROBE
                and nonzero_counts[index] != 0
            ):
                raise RewardActivationLedgerError(
                    f"diagnostic reward probe {metadata['name']!r} produced a "
                    "non-zero weighted contribution"
                )
            terms.append(
                {
                    "name": metadata["name"],
                    "callable": metadata["callable"],
                    "role": metadata["role"],
                    "weight": metadata["weight"],
                    "recipe_term_sha256": self._recipe_term_sha256[
                        metadata["name"]
                    ],
                    "observed_environment_step_count": self._environment_step_count,
                    "observed_sample_count": self._observed_sample_count,
                    "nonzero_sample_count": nonzero_counts[index],
                    "weighted_sum": weighted_sums[index],
                    "raw_sum": raw_sums[index],
                    "raw_recovery": "validated_weighted_eq_raw_times_weight_times_step_dt",
                    "raw_recomposition_max_abs_error": raw_max_errors[index],
                    "eligibility": "unknown",
                    "eligibility_reason": "term_specific_mask_unavailable",
                }
            )
        terms.sort(key=lambda item: item["name"])
        objective_names = [
            item["name"]
            for item in terms
            if item["role"] == REWARD_TERM_ROLE_OBJECTIVE
        ]
        diagnostic_probe_names = [
            item["name"]
            for item in terms
            if item["role"] == REWARD_TERM_ROLE_DIAGNOSTIC_PROBE
        ]
        record = {
            "event": "hope_effective_reward_activation_update",
            "schema_version": EFFECTIVE_REWARD_ACTIVATION_SCHEMA_VERSION,
            "recipe_sha256": self._recipe_sha256,
            "task_kind": self._task_kind,
            "ppo_update": ppo_update,
            "environment_step_count": self._environment_step_count,
            "expected_environment_step_count": self._expected_step_count,
            "num_envs": self._num_envs,
            "observed_sample_count": self._observed_sample_count,
            "step_dt_s": self._step_dt,
            "common_step_counter_start": self._update_start_common_step_counter,
            "common_step_counter_end": end_common_step_counter,
            "objective_term_names": objective_names,
            "diagnostic_probe_term_names": diagnostic_probe_names,
            "reward_cache_contract": {
                "source": "isaaclab_reward_manager_private_step_cache",
                "step_cache_semantics": "raw_times_weight",
                "weighted_semantics": "raw_times_weight_times_step_dt",
                "total_reward_closure": "validated",
                "max_abs_error": closure_max_error,
            },
            "total_weighted_reward_sum": total_reward_sum,
            "terms": terms,
        }
        # Keep the validated window frozen until a durable optimizer marker
        # exists.  Persistence failure must leave the exact evidence pending
        # and cannot silently open another rollout window.
        canonical = canonical_effective_reward_activation_json(record)
        self._pending_update = {
            "record": record,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "ppo_update": ppo_update,
            "end_common_step_counter": end_common_step_counter,
        }
        return record

    def acknowledge_update(self, record):
        """Consume exactly the prepared record after its durable commit marker."""

        pending = self._pending_update
        if pending is None or record is not pending["record"]:
            raise RewardActivationLedgerError(
                "runtime reward activation acknowledgement token is stale/foreign"
            )
        canonical = canonical_effective_reward_activation_json(record)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != pending[
            "sha256"
        ]:
            raise RewardActivationLedgerError(
                "prepared runtime reward activation was mutated before acknowledgement"
            )
        self._last_finished_update = pending["ppo_update"]
        self._update_start_common_step_counter = pending[
            "end_common_step_counter"
        ]
        self._pending_update = None
        self._reset_update_accumulators()

    def finish_update(self, ppo_update):
        """Compatibility transaction: prepare, then immediately acknowledge."""

        record = self.prepare_update(ppo_update)
        self.acknowledge_update(record)
        return record


class ActionBoundRewardEvidenceLedger:
    """Per-action and negative-transition evidence layered on activation truth.

    The wrapper freezes action identity and termination masks before ``env.step``
    can reset an environment, then reads the RewardManager cache and the new
    termination masks after that same step.  Only device reductions are kept
    for dense per-action accounting; per-environment rows are retained solely
    for the two soft-limit terms, terminal death transitions, and completed
    ``(env_id, reset_generation)`` RewardManager episode segments.
    """

    _SOFT_LIMIT_TERM_NAMES = ("joint_limit", "qdes_limit_barrier")
    _SOFT_LIMIT_CALLABLES = {
        "joint_limit": "actual_joint_limit_barrier_v2",
        "qdes_limit_barrier": "qdes_limit_barrier_v2",
    }
    _HARD_SAFETY_TERMINATION_TERMS = (
        ACTION_BALL_HARD_SAFETY_TERMINATION_TERMS
    )
    _REFERENCE_ENVELOPE_TERMINATION_TERMS = (
        ACTION_BALL_REFERENCE_ENVELOPE_TERMINATION_TERMS
    )
    _REQUIRED_TERMINATION_TERMS = (
        _HARD_SAFETY_TERMINATION_TERMS
        + _REFERENCE_ENVELOPE_TERMINATION_TERMS
    )

    def __init__(
        self,
        env,
        *,
        expected_environment_step_count,
        action_contract,
        action_identity_provider,
        termination_snapshot_provider,
        tensor_ops=None,
    ):
        self._activation = EffectiveRewardActivationLedger(
            env,
            task_kind="action_ball",
            expected_environment_step_count=expected_environment_step_count,
            tensor_ops=tensor_ops,
        )
        self._env = env
        self._ops = self._activation._ops
        self._num_envs = self._activation._num_envs
        self._expected_step_count = expected_environment_step_count
        self._identity_provider = action_identity_provider
        self._termination_provider = termination_snapshot_provider
        if not callable(action_identity_provider) or not callable(
            termination_snapshot_provider
        ):
            raise RewardActivationLedgerError(
                "action-bound Reward evidence requires callable identity/termination providers"
            )
        if not isinstance(action_contract, Mapping):
            raise RewardActivationLedgerError(
                "action-bound Reward evidence requires an action contract"
            )
        order = action_contract.get("action_order")
        uids = action_contract.get("action_uids")
        manifest = action_contract.get("manifest")
        if (
            not isinstance(order, (list, tuple))
            or not order
            or any(type(value) is not str or not value for value in order)
            or len(order) != len(set(order))
            or not isinstance(uids, (list, tuple))
            or len(uids) != len(order)
            or any(type(value) is not int or value < 0 for value in uids)
            or len(uids) != len(set(uids))
            or not isinstance(manifest, Mapping)
            or not _SHA256_RE.fullmatch(str(manifest.get("file_sha256", "")))
        ):
            raise RewardActivationLedgerError(
                "action-bound Reward action order/UID/manifest contract is invalid"
            )
        self._action_order = tuple(order)
        self._action_uids = tuple(uids)
        self._uid_to_action = dict(zip(self._action_uids, self._action_order))
        self._manifest_sha256 = str(manifest["file_sha256"])
        metadata = {
            item["name"]: item for item in self._activation._active_metadata
        }
        if "death_penalty" not in metadata:
            raise RewardActivationLedgerError(
                "action-bound negative evidence requires death_penalty"
            )
        for name in self._SOFT_LIMIT_TERM_NAMES:
            item = metadata.get(name)
            if (
                item is None
                or float(item["weight"]) >= 0.0
                or item["callable"].rsplit(".", 1)[-1]
                != self._SOFT_LIMIT_CALLABLES[name]
            ):
                raise RewardActivationLedgerError(
                    "action-bound negative evidence requires the exact active "
                    f"negative {name}/{self._SOFT_LIMIT_CALLABLES[name]} term"
                )
        death = metadata["death_penalty"]
        if (
            float(death["weight"]) >= 0.0
            or death["callable"].rsplit(".", 1)[-1]
            != "action_ball_safety_terminated"
            or tuple(death["recipe_term"]["params"].get("term_names", ()))
            != self._HARD_SAFETY_TERMINATION_TERMS
        ):
            raise RewardActivationLedgerError(
                "death_penalty must be the exact active ActionBall hard-safety union"
            )
        generic_death = [
            item["name"]
            for item in metadata.values()
            if item["callable"].rsplit(".", 1)[-1] == "is_terminated"
        ]
        hard_safety_death = [
            item["name"]
            for item in metadata.values()
            if item["callable"].rsplit(".", 1)[-1]
            == "action_ball_safety_terminated"
        ]
        terminal_specific = [
            item["name"]
            for item in metadata.values()
            if item["name"] == "table_hit_penalty"
            or item["callable"].rsplit(".", 1)[-1] == "terminated_by_term"
        ]
        if generic_death or hard_safety_death != ["death_penalty"] or terminal_specific:
            raise RewardActivationLedgerError(
                "terminal Reward must be exactly one hard-safety-union death "
                "with no generic/reason-specific stack"
            )
        self._metadata = metadata
        if not math.isclose(
            self._activation._step_dt,
            ACTION_BALL_ADOPTED_STEP_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RewardActivationLedgerError(
                "action-bound negative evidence requires the adopted ActionBall "
                f"step_dt={ACTION_BALL_ADOPTED_STEP_DT_S}"
            )
        for name in self._SOFT_LIMIT_TERM_NAMES:
            if not math.isclose(
                float(metadata[name]["weight"]),
                ACTION_BALL_ADOPTED_SOFT_LIMIT_WEIGHT,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RewardActivationLedgerError(
                    "action-bound negative evidence requires the adopted "
                    f"{name} weight={ACTION_BALL_ADOPTED_SOFT_LIMIT_WEIGHT}"
                )
        if not math.isclose(
            float(death["weight"]),
            ACTION_BALL_ADOPTED_DEATH_WEIGHT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RewardActivationLedgerError(
                "action-bound negative evidence requires the adopted "
                f"death_penalty weight={ACTION_BALL_ADOPTED_DEATH_WEIGHT}"
            )
        self._term_names = tuple(item["name"] for item in self._activation._active_metadata)
        self._term_indices = {
            item["name"]: offset
            for offset, item in enumerate(self._activation._active_metadata)
        }
        try:
            self._reward_group_taxonomy = (
                build_action_ball_reward_group_taxonomy(
                    [
                        item["recipe_term"]
                        for item in self._activation._active_metadata
                    ]
                )
            )
        except RewardRecipeError as exc:
            raise RewardActivationLedgerError(str(exc)) from exc
        active_taxonomy = {
            item["name"]: item
            for item in self._reward_group_taxonomy["active_terms"]
        }
        self._expected_contribution_by_term = {
            name: row["expected_contribution"]
            for name, row in active_taxonomy.items()
        }
        self._group_objective_indices = {
            group: tuple(
                self._term_indices[name]
                for name in self._term_names
                if active_taxonomy[name]["group"] == group
                and active_taxonomy[name]["role"]
                == REWARD_TERM_ROLE_OBJECTIVE
            )
            for group in ACTION_BALL_REWARD_GROUP_ORDER
        }
        self._group_objective_term_names = {
            group: tuple(
                sorted(
                    name
                    for name in self._term_names
                    if active_taxonomy[name]["group"] == group
                    and active_taxonomy[name]["role"]
                    == REWARD_TERM_ROLE_OBJECTIVE
                )
            )
            for group in ACTION_BALL_REWARD_GROUP_ORDER
        }
        self._group_probe_term_names = {
            group: tuple(
                sorted(
                    name
                    for name in self._term_names
                    if active_taxonomy[name]["group"] == group
                    and active_taxonomy[name]["role"]
                    == REWARD_TERM_ROLE_DIAGNOSTIC_PROBE
                )
            )
            for group in ACTION_BALL_REWARD_GROUP_ORDER
        }
        self._open_step = None
        self._step_rows = []
        self._per_action = None
        self._pending = None
        self._pending_sha256 = None
        self._reward_device = self._device_key(
            self._activation._manager._step_reward
        )
        self._manager_term_names = tuple(self._activation._all_names)
        self._max_episode_length_s = _plain_finite_float(
            getattr(env, "max_episode_length_s", None),
            context="environment max_episode_length_s",
            positive=True,
        )
        reward_buf = self._activation._manager._reward_buf
        self._episode_term_sums = {
            name: self._clone(reward_buf * 0)
            for name in self._manager_term_names
        }
        self._episode_reward_buf_sums = self._clone(reward_buf * 0)
        self._episode_step_counts = self._ops.as_tensor_like(
            [0] * self._num_envs,
            self._activation._manager._step_reward[:, 0],
        )
        self._episode_reset_generations = None
        self._episode_action_uids = None
        self._episode_external_reset_pending = set()
        self._completed_episode_segments = []
        self._episode_reset_batches = []
        self._episode_closure_stats = self._new_episode_closure_stats()
        self._closed = False
        self._original_reward_manager_reset = None
        self._reward_manager_reset_wrapper = None
        self._require_manager_episode_sums(
            require_zero=True, context="ActionBall Reward ledger construction"
        )
        self._install_reward_manager_reset_hook()

    @staticmethod
    def _clone(value):
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        clone = getattr(value, "clone", None)
        if callable(clone):
            return clone()
        copy = getattr(value, "copy", None)
        if callable(copy):
            return copy()
        raise RewardActivationLedgerError("evidence tensor cannot be cloned")

    @staticmethod
    def _tensor_any(value):
        result = value.any()
        item = getattr(result, "item", None)
        return bool(item() if callable(item) else result)

    @staticmethod
    def _device_key(value):
        device = getattr(value, "device", None)
        return ("host", None) if device is None else ("device", str(device))

    @staticmethod
    def _dtype_key(value):
        return str(getattr(value, "dtype", ""))

    def _require_evidence_tensor(
        self, tensor, *, field, dtype, shape=None
    ):
        expected_shape = (
            (self._num_envs,) if shape is None else tuple(shape)
        )
        if (
            not self._ops.is_tensor(tensor)
            or tuple(getattr(tensor, "shape", ())) != expected_shape
            or self._device_key(tensor) != self._reward_device
        ):
            raise RewardActivationLedgerError(
                f"{field} must be shaped {expected_shape} on the original Reward device"
            )
        key = self._dtype_key(tensor)
        accepted = (
            {"torch.int64", "int64"}
            if dtype == "int64"
            else {"torch.bool", "bool"}
        )
        if key not in accepted:
            raise RewardActivationLedgerError(
                f"{field} must have exact {dtype} dtype, got {key!r}"
            )
        return tensor

    @staticmethod
    def _cast_like(value, like):
        to = getattr(value, "to", None)
        if callable(to):
            return to(dtype=like.dtype)
        astype = getattr(value, "astype", None)
        if callable(astype):
            return astype(like.dtype)
        raise RewardActivationLedgerError("boolean evidence cannot cast to reward dtype")

    @staticmethod
    def _new_episode_closure_stats():
        return {
            "environment_step_count": 0,
            "reset_batch_count": 0,
            "completed_episode_count": 0,
            "manager_episode_sum_comparison_count": 0,
            "dashboard_term_comparison_count": 0,
            "reward_buf_term_sum_comparison_count": 0,
            "manager_clear_comparison_count": 0,
            "max_abs_manager_episode_sum_error": 0.0,
            "max_abs_dashboard_normalization_error": 0.0,
            "max_abs_reward_buf_vs_term_sum_error": 0.0,
            "max_abs_manager_clear_error": 0.0,
        }

    def _update_episode_error(self, field, error):
        error = _normalized_output_float(error, context=field)
        self._episode_closure_stats[field] = max(
            self._episode_closure_stats[field], abs(error)
        )

    def _tensor_max_abs_error(self, actual, expected, *, context):
        if (
            not self._ops.is_tensor(actual)
            or not self._ops.is_tensor(expected)
            or tuple(getattr(actual, "shape", ()))
            != tuple(getattr(expected, "shape", ()))
        ):
            raise RewardActivationLedgerError(
                f"{context} tensors have incompatible shapes"
            )
        difference = self._ops.abs(actual - expected)
        tolerance = (
            self._activation._closure_atol
            + self._activation._closure_rtol * self._ops.abs(expected)
        )
        if self._tensor_any(difference > tolerance):
            raise RewardActivationLedgerError(
                f"{context} violates the episode-segmented Reward closure"
            )
        return self._host_float(
            self._ops.max(difference), name=f"{context} max_abs_error"
        )

    def _require_manager_episode_sums(self, *, require_zero=False, context):
        episode_sums = getattr(
            self._activation._manager, "_episode_sums", None
        )
        if (
            not isinstance(episode_sums, Mapping)
            or tuple(episode_sums) != self._manager_term_names
        ):
            raise RewardActivationLedgerError(
                f"{context} requires ordered RewardManager _episode_sums "
                "for every active_terms entry"
            )
        result = {}
        reward_buf = self._activation._manager._reward_buf
        for name in self._manager_term_names:
            value = episode_sums[name]
            if (
                not self._ops.is_tensor(value)
                or tuple(getattr(value, "shape", ()))
                != (self._num_envs,)
                or self._device_key(value) != self._reward_device
                or self._dtype_key(value) != self._dtype_key(reward_buf)
            ):
                raise RewardActivationLedgerError(
                    f"{context} RewardManager _episode_sums[{name!r}] "
                    "must match the Reward buffer shape/device/dtype"
                )
            cloned = self._clone(value)
            result[name] = cloned
        stacked = self._ops.stack(
            [result[name] for name in self._manager_term_names], axis=1
        )
        if self._tensor_any(~self._ops.isfinite(stacked)):
            raise RewardActivationLedgerError(
                f"{context} RewardManager _episode_sums contains a non-finite value"
            )
        if require_zero and self._tensor_any(stacked != 0):
            raise RewardActivationLedgerError(
                f"{context} found a non-zero pre-existing RewardManager episode sum"
            )
        return result

    def _normalize_reset_env_ids(self, raw_env_ids):
        if raw_env_ids is None:
            values = list(range(self._num_envs))
        elif isinstance(raw_env_ids, slice):
            values = list(range(self._num_envs))[raw_env_ids]
        elif type(raw_env_ids) is int:
            values = [raw_env_ids]
        elif self._ops.is_tensor(raw_env_ids):
            values = self._ops.to_host_list(raw_env_ids)
        elif isinstance(raw_env_ids, (list, tuple)):
            values = list(raw_env_ids)
        else:
            raise RewardActivationLedgerError(
                "RewardManager.reset env_ids cannot be normalized without guessing"
            )
        if (
            any(type(value) is not int for value in values)
            or len(values) != len(set(values))
            or any(value < 0 or value >= self._num_envs for value in values)
        ):
            raise RewardActivationLedgerError(
                "RewardManager.reset env_ids are invalid, duplicate, or out of range"
            )
        return tuple(sorted(values))

    def _dashboard_values(self, extras, *, context):
        if not isinstance(extras, Mapping):
            raise RewardActivationLedgerError(
                f"{context} RewardManager.reset did not return a dashboard mapping"
            )
        result = {}
        for name in self._manager_term_names:
            key = f"Episode_Reward/{name}"
            if key not in extras:
                raise RewardActivationLedgerError(
                    f"{context} RewardManager.reset omitted {key!r}"
                )
            raw = extras[key]
            if self._ops.is_tensor(raw):
                raw = self._ops.to_host_scalar(raw)
            result[name] = _normalized_output_float(
                raw, context=f"{context} {key}"
            )
        return result

    def _install_reward_manager_reset_hook(self):
        manager = self._activation._manager
        original = getattr(manager, "reset", None)
        if not callable(original):
            raise RewardActivationLedgerError(
                "episode-segmented Reward closure requires callable "
                "RewardManager.reset(env_ids)"
            )
        self._original_reward_manager_reset = original

        def reset_with_episode_closure(*args, **kwargs):
            if self._closed:
                raise RewardActivationLedgerError(
                    "RewardManager reset hook was used after ledger close"
                )
            if len(args) > 1 or (
                args and "env_ids" in kwargs
            ) or set(kwargs) - {"env_ids"}:
                raise RewardActivationLedgerError(
                    "RewardManager.reset call shape changed during ActionBall training"
                )
            raw_env_ids = (
                args[0] if args else kwargs.get("env_ids", None)
            )
            env_ids = self._normalize_reset_env_ids(raw_env_ids)
            before = self._require_manager_episode_sums(
                context="RewardManager.reset pre-clear"
            )
            step_reward = self._clone(
                self._activation._manager._step_reward
            )
            reward_buf = self._clone(
                self._activation._manager._reward_buf
            )
            result = original(*args, **kwargs)
            after = self._require_manager_episode_sums(
                context="RewardManager.reset post-clear"
            )
            if not env_ids:
                return result
            dashboard = self._dashboard_values(
                result, context="RewardManager.reset"
            )
            capture = {
                "env_ids": env_ids,
                "manager_episode_sums": {
                    name: self._clone(before[name][list(env_ids)])
                    for name in self._manager_term_names
                },
                "step_reward": self._clone(
                    step_reward[list(env_ids), :]
                ),
                "reward_buf": self._clone(reward_buf[list(env_ids)]),
                "dashboard": dashboard,
            }
            cleared = self._ops.stack(
                [
                    after[name][list(env_ids)]
                    for name in self._manager_term_names
                ],
                axis=1,
            )
            error = self._tensor_max_abs_error(
                cleared,
                cleared * 0,
                context="RewardManager reset clear",
            )
            self._update_episode_error(
                "max_abs_manager_clear_error", error
            )
            self._episode_closure_stats[
                "manager_clear_comparison_count"
            ] += len(env_ids) * len(self._manager_term_names)
            if self._open_step is not None:
                already = {
                    env_id
                    for row in self._open_step["reset_captures"]
                    for env_id in row["env_ids"]
                }
                if already.intersection(env_ids):
                    raise RewardActivationLedgerError(
                        "one environment was Reward-reset more than once in one step"
                    )
                self._open_step["reset_captures"].append(capture)
            else:
                self._close_external_reset(capture)
            return result

        try:
            manager.reset = reset_with_episode_closure
        except Exception as exc:
            raise RewardActivationLedgerError(
                "cannot install the required RewardManager reset closure hook"
            ) from exc
        if getattr(manager, "reset", None) is not reset_with_episode_closure:
            raise RewardActivationLedgerError(
                "RewardManager reset closure hook did not install exactly"
            )
        self._reward_manager_reset_wrapper = reset_with_episode_closure

    def _ensure_episode_identity(self, identity):
        if self._episode_reset_generations is None:
            self._episode_reset_generations = self._clone(
                identity["reset_generation"]
            )
            self._episode_action_uids = self._clone(identity["action_uid"])
            return
        if self._episode_external_reset_pending:
            env_ids = sorted(self._episode_external_reset_pending)
            if self._tensor_any(
                self._episode_reset_generations[env_ids]
                != identity["reset_generation"][env_ids]
            ):
                raise RewardActivationLedgerError(
                    "external Reward reset generation did not commit exactly once"
                )
            self._episode_action_uids[env_ids] = identity["action_uid"][
                env_ids
            ]
            self._episode_external_reset_pending.clear()
        mismatch = (
            self._episode_reset_generations
            != identity["reset_generation"]
        ) | (self._episode_action_uids != identity["action_uid"])
        if self._tensor_any(mismatch):
            raise RewardActivationLedgerError(
                "live action identity drifted outside a captured Reward reset"
            )

    def _validate_open_manager_episode_sums(self, *, context):
        current = self._require_manager_episode_sums(context=context)
        actual = self._ops.stack(
            [current[name] for name in self._manager_term_names], axis=1
        )
        expected = self._ops.stack(
            [
                self._episode_term_sums[name]
                for name in self._manager_term_names
            ],
            axis=1,
        )
        error = self._tensor_max_abs_error(
            actual, expected, context=context
        )
        self._update_episode_error(
            "max_abs_manager_episode_sum_error", error
        )
        self._episode_closure_stats[
            "manager_episode_sum_comparison_count"
        ] += self._num_envs * len(self._manager_term_names)

    def _record_dashboard_batch(
        self, capture, *, reset_generations, administrative
    ):
        env_ids = capture["env_ids"]
        terms = []
        for name in self._manager_term_names:
            manager_values = self._ops.to_host_list(
                capture["manager_episode_sums"][name]
            )
            expected = (
                sum(float(value) for value in manager_values)
                / len(env_ids)
                / self._max_episode_length_s
            )
            actual = capture["dashboard"][name]
            error = abs(actual - expected)
            tolerance = self._activation._closure_atol + (
                self._activation._closure_rtol * abs(expected)
            )
            if error > tolerance:
                raise RewardActivationLedgerError(
                    f"Episode_Reward/{name} dashboard normalization "
                    "does not close to RewardManager _episode_sums"
                )
            self._update_episode_error(
                "max_abs_dashboard_normalization_error", error
            )
            self._episode_closure_stats[
                "dashboard_term_comparison_count"
            ] += 1
            terms.append(
                {
                    "name": name,
                    "reward_manager_episode_sum_mean": (
                        sum(float(value) for value in manager_values)
                        / len(env_ids)
                    ),
                    "expected_dashboard_value": expected,
                    "actual_dashboard_value": actual,
                    "abs_error": error,
                }
            )
        self._episode_reset_batches.append(
            {
                "env_ids": list(env_ids),
                "reset_generations": list(reset_generations),
                "administrative_reset": bool(administrative),
                "normalization_divisor_s": self._max_episode_length_s,
                "terms": terms,
                "status": "PASS",
            }
        )
        self._episode_closure_stats["reset_batch_count"] += 1

    def _append_completed_segment(
        self,
        *,
        env_id,
        reset_generation,
        action_uid,
        terminated,
        timed_out,
        administrative,
        expected_term_values,
        manager_term_values,
        reward_buf_sum,
        step_count,
    ):
        term_sum_total = sum(expected_term_values)
        error = abs(term_sum_total - reward_buf_sum)
        tolerance = self._activation._closure_atol + (
            self._activation._closure_rtol * abs(reward_buf_sum)
        )
        if error > tolerance:
            raise RewardActivationLedgerError(
                "episode sum(reward_buf) does not close to all Reward term sums"
            )
        self._update_episode_error(
            "max_abs_reward_buf_vs_term_sum_error", error
        )
        self._episode_closure_stats[
            "reward_buf_term_sum_comparison_count"
        ] += 1
        self._completed_episode_segments.append(
            {
                "env_id": env_id,
                "reset_generation": reset_generation,
                "action_uid": action_uid,
                "step_count": step_count,
                "terminated": bool(terminated),
                "timed_out": bool(timed_out),
                "administrative_reset": bool(administrative),
                "reward_buf_sum": reward_buf_sum,
                "all_term_sum": term_sum_total,
                "reward_buf_vs_all_terms_abs_error": error,
                "local_term_sums": expected_term_values,
                "reward_manager_episode_sums": manager_term_values,
            }
        )
        self._episode_closure_stats["completed_episode_count"] += 1

    def _zero_episode_envs(self, env_ids):
        for name in self._manager_term_names:
            self._episode_term_sums[name][list(env_ids)] = 0
        self._episode_reward_buf_sums[list(env_ids)] = 0
        self._episode_step_counts[list(env_ids)] = 0

    def _close_external_reset(self, capture):
        identity = self._validate_identity(self._identity_provider())
        if self._episode_reset_generations is None:
            self._ensure_episode_identity(identity)
        env_ids = capture["env_ids"]
        generations = self._ops.to_host_list(
            self._episode_reset_generations[list(env_ids)]
        )
        action_uids = self._ops.to_host_list(
            self._episode_action_uids[list(env_ids)]
        )
        step_counts = self._ops.to_host_list(
            self._episode_step_counts[list(env_ids)]
        )
        reward_sums = self._ops.to_host_list(
            self._episode_reward_buf_sums[list(env_ids)]
        )
        local_by_term = {
            name: self._ops.to_host_list(
                self._episode_term_sums[name][list(env_ids)]
            )
            for name in self._manager_term_names
        }
        manager_by_term = {
            name: self._ops.to_host_list(
                capture["manager_episode_sums"][name]
            )
            for name in self._manager_term_names
        }
        for offset, env_id in enumerate(env_ids):
            expected = [
                float(local_by_term[name][offset])
                for name in self._manager_term_names
            ]
            observed = [
                float(manager_by_term[name][offset])
                for name in self._manager_term_names
            ]
            for name, left, right in zip(
                self._manager_term_names, observed, expected
            ):
                error = abs(left - right)
                tolerance = self._activation._closure_atol + (
                    self._activation._closure_rtol * abs(right)
                )
                if error > tolerance:
                    raise RewardActivationLedgerError(
                        f"external reset _episode_sums[{name!r}] does not "
                        "close to the open local segment"
                    )
                self._update_episode_error(
                    "max_abs_manager_episode_sum_error", error
                )
                self._episode_closure_stats[
                    "manager_episode_sum_comparison_count"
                ] += 1
            self._append_completed_segment(
                env_id=env_id,
                reset_generation=int(generations[offset]),
                action_uid=int(action_uids[offset]),
                terminated=False,
                timed_out=False,
                administrative=True,
                expected_term_values=expected,
                manager_term_values=observed,
                reward_buf_sum=float(reward_sums[offset]),
                step_count=int(step_counts[offset]),
            )
        self._record_dashboard_batch(
            capture,
            reset_generations=[int(value) for value in generations],
            administrative=True,
        )
        self._zero_episode_envs(env_ids)
        self._episode_reset_generations[list(env_ids)] += 1
        self._episode_external_reset_pending.update(env_ids)

    def abort_environment_step(self):
        """Discard an interrupted transaction; training must stop afterwards."""

        self._open_step = None

    def close(self):
        """Restore RewardManager.reset after the runner leaves ``learn``."""

        if self._closed:
            return
        manager = self._activation._manager
        if (
            self._reward_manager_reset_wrapper is not None
            and getattr(manager, "reset", None)
            is self._reward_manager_reset_wrapper
        ):
            manager.reset = self._original_reward_manager_reset
        self._closed = True

    def _validate_identity(self, value):
        if not isinstance(value, Mapping) or set(value) != {
            "action_uid",
            "reset_generation",
            "swing_generation",
            "birth_receipt_sha256",
        }:
            raise RewardActivationLedgerError(
                "frozen action identity has a schema mismatch"
            )
        tensors = {}
        for field in ("action_uid", "reset_generation", "swing_generation"):
            tensor = self._require_evidence_tensor(
                value[field],
                field=f"frozen action identity {field}",
                dtype="int64",
            )
            tensors[field] = self._clone(tensor)
        valid = tensors["action_uid"] == int(self._action_uids[0])
        for uid in self._action_uids[1:]:
            valid = valid | (tensors["action_uid"] == int(uid))
        invalid = (
            ~valid
            | (tensors["reset_generation"] < 0)
            | (tensors["swing_generation"] < 0)
        )
        if self._tensor_any(invalid):
            raise RewardActivationLedgerError(
                "frozen action identity contains an invalid UID/generation"
            )
        receipts = value["birth_receipt_sha256"]
        if (
            not isinstance(receipts, tuple)
            or len(receipts) != self._num_envs
            or any(
                type(receipt) is not str
                or _SHA256_RE.fullmatch(receipt) is None
                for receipt in receipts
            )
        ):
            raise RewardActivationLedgerError(
                "frozen action birth receipts are not env-aligned SHA-256 values"
            )
        return {**tensors, "birth_receipt_sha256": receipts}

    def _validate_termination_snapshot(self, value):
        if not isinstance(value, Mapping) or set(value) != {
            "term_order",
            "terminated",
            "time_outs",
            "reason_masks",
        }:
            raise RewardActivationLedgerError(
                "termination snapshot has a schema mismatch"
            )
        order = value["term_order"]
        reasons = value["reason_masks"]
        if (
            not isinstance(order, tuple)
            or len(order) != len(set(order))
            or any(type(name) is not str or not name for name in order)
            or not set(self._REQUIRED_TERMINATION_TERMS).issubset(set(order))
            or not isinstance(reasons, Mapping)
            or set(reasons) != set(order)
        ):
            raise RewardActivationLedgerError(
                "termination snapshot term order/reason set is incomplete"
            )
        tensors = {}
        for field, tensor in (
            ("terminated", value["terminated"]),
            ("time_outs", value["time_outs"]),
        ):
            tensor = self._require_evidence_tensor(
                tensor,
                field=f"termination snapshot {field}",
                dtype="bool",
            )
            tensors[field] = self._clone(tensor)
        cloned_reasons = {}
        for name in order:
            tensor = self._require_evidence_tensor(
                reasons[name],
                field=f"termination reason {name!r}",
                dtype="bool",
            )
            cloned_reasons[name] = self._clone(tensor)
        return {
            "term_order": order,
            **tensors,
            "reason_masks": cloned_reasons,
        }

    def begin_environment_step(self):
        if (
            self._open_step is not None
            or self._pending is not None
            or self._activation._pending_update is not None
        ):
            raise RewardActivationLedgerError(
                "action-bound Reward evidence has an open/pending transaction"
            )
        identity = self._validate_identity(self._identity_provider())
        self._ensure_episode_identity(identity)
        self._validate_open_manager_episode_sums(
            context="pre-step open episode"
        )
        pre = self._validate_termination_snapshot(self._termination_provider())
        marker = object()
        self._open_step = {
            "marker": marker,
            "step_index": len(self._step_rows),
            "common_step_counter": self._activation._read_common_step_counter(),
            "identity": identity,
            "pre": pre,
            "reset_captures": [],
        }
        return marker

    def observe_after_environment_step(self, token):
        if (
            self._open_step is None
            or token is not self._open_step["marker"]
        ):
            raise RewardActivationLedgerError(
                "action-bound Reward step token is stale or foreign"
            )
        frozen = self._open_step
        post_identity = self._validate_identity(self._identity_provider())
        post = self._validate_termination_snapshot(self._termination_provider())
        if post["term_order"] != frozen["pre"]["term_order"]:
            raise RewardActivationLedgerError(
                "termination term order changed inside one environment step"
            )
        common_end = self._activation._read_common_step_counter()
        if common_end != frozen["common_step_counter"] + 1:
            raise RewardActivationLedgerError(
                "action-bound Reward step did not advance common_step_counter once"
            )
        manager = self._activation._manager
        step_reward = self._ops.detach(manager._step_reward)
        reward_buf = self._ops.detach(manager._reward_buf)
        all_weighted = step_reward * self._activation._step_dt
        active_rate = step_reward[:, list(self._activation._active_indices)]
        weighted = active_rate * self._activation._step_dt
        raw = weighted / (
            self._activation._weight_vector * self._activation._step_dt
        )
        death_index = self._term_indices["death_penalty"]
        death_raw = raw[:, death_index]
        safety_mask = post["reason_masks"][
            self._HARD_SAFETY_TERMINATION_TERMS[0]
        ]
        for name in self._HARD_SAFETY_TERMINATION_TERMS[1:]:
            safety_mask = safety_mask | post["reason_masks"][name]
        safety_numeric = self._cast_like(safety_mask, death_raw)
        done = post["terminated"] | post["time_outs"]
        expected_generation = (
            frozen["identity"]["reset_generation"]
            + self._cast_like(done, frozen["identity"]["reset_generation"])
        )
        identity_invalid = (
            post_identity["reset_generation"] != expected_generation
        ) | (
            (~done)
            & (
                post_identity["action_uid"]
                != frozen["identity"]["action_uid"]
            )
        )
        if self._tensor_any(identity_invalid):
            raise RewardActivationLedgerError(
                "post-step action/reset identity does not match the Done reset edge"
            )
        captured_env_ids = tuple(
            sorted(
                env_id
                for capture in frozen["reset_captures"]
                for env_id in capture["env_ids"]
            )
        )
        captured_set = set(captured_env_ids)
        for env_id in range(self._num_envs):
            if (
                env_id not in captured_set
                and post_identity["birth_receipt_sha256"][env_id]
                != frozen["identity"]["birth_receipt_sha256"][env_id]
            ):
                raise RewardActivationLedgerError(
                    "action birth receipt changed without a captured reset"
                )
        captured_mask = self._ops.as_tensor_like(
            [
                env_id in captured_set
                for env_id in range(self._num_envs)
            ],
            done,
        )
        if self._tensor_any(done != captured_mask):
            raise RewardActivationLedgerError(
                "Done environments and RewardManager reset-hook coverage differ"
            )
        post_invalid = (
            self._ops.abs(death_raw - safety_numeric) > 1.0e-6
        )
        for reason in post["reason_masks"].values():
            post_invalid = post_invalid | (reason & ~done)
        # Required safety terms are non-timeout DoneTerms.  A coincident
        # timeout must not turn a hard/table/fall event into a timeout-only row
        # that escapes the generic death charge and terminal transcript.
        for name in self._HARD_SAFETY_TERMINATION_TERMS:
            post_invalid = post_invalid | (
                post["reason_masks"][name] & ~post["terminated"]
            )
        for name in self._SOFT_LIMIT_TERM_NAMES:
            index = self._term_indices[name]
            post_invalid = post_invalid | (raw[:, index] < 0)
            post_invalid = post_invalid | (weighted[:, index] > 0)
        for name, index in self._term_indices.items():
            expected = self._expected_contribution_by_term[name]
            if expected == "positive":
                post_invalid = post_invalid | (weighted[:, index] < 0)
            elif expected == "negative":
                post_invalid = post_invalid | (weighted[:, index] > 0)
        if self._tensor_any(post_invalid):
            raise RewardActivationLedgerError(
                "post-step Reward/termination evidence violates death, reason-mask, "
                "or soft-limit sign invariants"
            )

        current_episode_sums = self._require_manager_episode_sums(
            context="post-step Reward closure"
        )
        expected_term_columns = {
            name: (
                self._episode_term_sums[name]
                + all_weighted[:, term_index]
            )
            for term_index, name in enumerate(self._manager_term_names)
        }
        expected_term_matrix = self._ops.stack(
            [
                expected_term_columns[name]
                for name in self._manager_term_names
            ],
            axis=1,
        )
        expected_reward_sums = self._episode_reward_buf_sums + reward_buf
        expected_step_counts = self._episode_step_counts + 1
        expected_post_manager = self._clone(expected_term_matrix)
        if captured_env_ids:
            expected_post_manager[list(captured_env_ids), :] = 0
        actual_post_manager = self._ops.stack(
            [
                current_episode_sums[name]
                for name in self._manager_term_names
            ],
            axis=1,
        )
        manager_error = self._tensor_max_abs_error(
            actual_post_manager,
            expected_post_manager,
            context="post-step open/reset RewardManager _episode_sums",
        )
        self._update_episode_error(
            "max_abs_manager_episode_sum_error", manager_error
        )
        self._episode_closure_stats[
            "manager_episode_sum_comparison_count"
        ] += self._num_envs * len(self._manager_term_names)

        reset_generations_by_capture = {}
        for capture in frozen["reset_captures"]:
            env_ids = list(capture["env_ids"])
            capture_weighted = (
                capture["step_reward"] * self._activation._step_dt
            )
            expected_step_weighted = all_weighted[env_ids, :]
            self._tensor_max_abs_error(
                capture_weighted,
                expected_step_weighted,
                context="reset-hook current-step term cache",
            )
            self._tensor_max_abs_error(
                capture["reward_buf"],
                reward_buf[env_ids],
                context="reset-hook current-step reward_buf cache",
            )
            captured_manager_matrix = self._ops.stack(
                [
                    capture["manager_episode_sums"][name]
                    for name in self._manager_term_names
                ],
                axis=1,
            )
            capture_error = self._tensor_max_abs_error(
                captured_manager_matrix,
                expected_term_matrix[env_ids, :],
                context="completed episode RewardManager _episode_sums",
            )
            self._update_episode_error(
                "max_abs_manager_episode_sum_error", capture_error
            )
            self._episode_closure_stats[
                "manager_episode_sum_comparison_count"
            ] += len(env_ids) * len(self._manager_term_names)
            expected_host = self._ops.to_host_list(
                expected_term_matrix[env_ids, :]
            )
            manager_host = self._ops.to_host_list(
                captured_manager_matrix
            )
            reward_host = self._ops.to_host_list(
                expected_reward_sums[env_ids]
            )
            step_count_host = self._ops.to_host_list(
                expected_step_counts[env_ids]
            )
            generation_host = self._ops.to_host_list(
                frozen["identity"]["reset_generation"][env_ids]
            )
            action_uid_host = self._ops.to_host_list(
                frozen["identity"]["action_uid"][env_ids]
            )
            terminated_host = self._ops.to_host_list(
                post["terminated"][env_ids]
            )
            timeout_host = self._ops.to_host_list(
                post["time_outs"][env_ids]
            )
            for offset, env_id in enumerate(env_ids):
                self._append_completed_segment(
                    env_id=env_id,
                    reset_generation=int(generation_host[offset]),
                    action_uid=int(action_uid_host[offset]),
                    terminated=bool(terminated_host[offset]),
                    timed_out=bool(timeout_host[offset]),
                    administrative=False,
                    expected_term_values=[
                        float(value)
                        for value in expected_host[offset]
                    ],
                    manager_term_values=[
                        float(value) for value in manager_host[offset]
                    ],
                    reward_buf_sum=float(reward_host[offset]),
                    step_count=int(step_count_host[offset]),
                )
            reset_generations_by_capture[id(capture)] = [
                int(value) for value in generation_host
            ]
        for name in self._manager_term_names:
            self._episode_term_sums[name] = self._clone(
                expected_term_columns[name]
            )
        self._episode_reward_buf_sums = self._clone(
            expected_reward_sums
        )
        self._episode_step_counts = self._clone(expected_step_counts)
        if captured_env_ids:
            self._zero_episode_envs(captured_env_ids)
        for capture in frozen["reset_captures"]:
            self._record_dashboard_batch(
                capture,
                reset_generations=reset_generations_by_capture[id(capture)],
                administrative=False,
            )
        self._episode_reset_generations = self._clone(
            post_identity["reset_generation"]
        )
        self._episode_action_uids = self._clone(
            post_identity["action_uid"]
        )
        self._episode_closure_stats["environment_step_count"] += 1

        # Only after every action/termination/negative-sign invariant has
        # passed may the aggregate activation ledger book this step.
        self._activation.observe_after_environment_step()

        if self._per_action is None:
            self._per_action = {
                uid: {
                    "observed_sample_count": None,
                    "terms": {
                        name: {
                            "nonzero_sample_count": None,
                            "raw_sum": None,
                            "weighted_sum": None,
                            "terminated_nonzero_sample_count": None,
                        }
                        for name in self._term_names
                    },
                }
                for uid in self._action_uids
            }
        action_uid = frozen["identity"]["action_uid"]
        for uid in self._action_uids:
            mask = action_uid == int(uid)
            row = self._per_action[uid]
            observed = mask.sum()
            row["observed_sample_count"] = (
                observed
                if row["observed_sample_count"] is None
                else row["observed_sample_count"] + observed
            )
            for name, term_index in self._term_indices.items():
                raw_term = raw[:, term_index]
                weighted_term = weighted[:, term_index]
                term_row = row["terms"][name]
                values = {
                    "nonzero_sample_count": ((weighted_term != 0) & mask).sum(),
                    "raw_sum": (raw_term * mask).sum(),
                    "weighted_sum": (weighted_term * mask).sum(),
                    "terminated_nonzero_sample_count": (
                        (weighted_term != 0) & mask & post["terminated"]
                    ).sum(),
                }
                for field, value in values.items():
                    term_row[field] = (
                        value
                        if term_row[field] is None
                        else term_row[field] + value
                    )
        reward_groups = {}
        for group in ACTION_BALL_REWARD_GROUP_ORDER:
            indices = self._group_objective_indices[group]
            if not indices:
                reward_groups[group] = None
                continue
            contributions = weighted[:, list(indices)]
            zeros = contributions * 0
            positive = self._ops.maximum(contributions, zeros)
            reward_groups[group] = {
                "weighted": self._clone(
                    self._ops.sum(contributions, axis=1)
                ),
                "nonzero_term_count": self._clone(
                    self._ops.count_nonzero(contributions, axis=1)
                ),
                "positive_weighted": self._clone(
                    self._ops.sum(positive, axis=1)
                ),
                "negative_weighted": self._clone(
                    self._ops.sum(contributions - positive, axis=1)
                ),
            }
        self._step_rows.append(
            {
                "step_index": frozen["step_index"],
                "common_step_counter": frozen["common_step_counter"],
                "identity": frozen["identity"],
                "pre": frozen["pre"],
                "common_step_counter_end": common_end,
                "post": post,
                "death_raw": self._clone(death_raw),
                "death_weighted": self._clone(weighted[:, death_index]),
                "reward_groups": reward_groups,
            }
        )
        self._open_step = None

    def _host_int(self, value, *, name):
        raw = self._ops.to_host_scalar(value)
        if type(raw) not in (int, float) or int(raw) != raw or int(raw) < 0:
            raise RewardActivationLedgerError(
                f"{name} did not reduce to a nonnegative integer"
            )
        return int(raw)

    def _host_float(self, value, *, name):
        return _normalized_output_float(
            self._ops.to_host_scalar(value), context=name
        )

    @staticmethod
    def _transition_id(ppo_update, transition):
        identity = {
            "ppo_update": ppo_update,
            "common_step_counter": transition["common_step_counter"],
            "joint_policy_step_sequence": transition[
                "joint_policy_step_sequence"
            ],
            "env_id": transition["env_id"],
            "action_uid": transition["action_uid"],
            "reset_generation": transition["reset_generation"],
            "swing_generation": transition["swing_generation"],
            "birth_receipt_sha256": transition["birth_receipt_sha256"],
            "termination_terms": list(transition["termination_terms"]),
            "rising_termination_terms": list(
                transition["rising_termination_terms"]
            ),
            "pre_terminal_reason_mask": transition[
                "pre_terminal_reason_mask"
            ],
            "post_terminal_reason_mask": transition[
                "post_terminal_reason_mask"
            ],
        }
        return hashlib.sha256(
            json.dumps(
                identity,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _reason_classes(terms):
        term_set = set(terms)
        classes = []
        if "robot_hit_table" in term_set:
            classes.append("table_hit")
        if term_set & {"base_fell_tilt", "base_too_low"}:
            classes.append("fall")
        if term_set & {
            "joint_qdes_forbidden",
            "joint_actual_forbidden",
        }:
            classes.append("hard_limit")
        if term_set & set(
            ACTION_BALL_REFERENCE_ENVELOPE_TERMINATION_TERMS
        ):
            classes.append("reference_envelope")
        recognized = set(ACTION_BALL_HARD_SAFETY_TERMINATION_TERMS) | set(
            ACTION_BALL_REFERENCE_ENVELOPE_TERMINATION_TERMS
        )
        if term_set - recognized:
            classes.append("other_termination")
        return classes

    @staticmethod
    def _primary_reason_class(classes):
        for name in (
            "table_hit",
            "fall",
            "hard_limit",
            "reference_envelope",
            "other_termination",
        ):
            if name in classes:
                return name
        raise RewardActivationLedgerError(
            "terminal transition has no classified termination reason"
        )

    def _prepared_reward_group_rows(self):
        samples = {
            uid: {
                group: {
                    "weighted": [],
                    "nonzero_term_count": [],
                    "positive_weighted": [],
                    "negative_weighted": [],
                }
                for group in ACTION_BALL_REWARD_GROUP_ORDER
            }
            for uid in self._action_uids
        }
        for step in self._step_rows:
            action_uids = self._ops.to_host_list(
                step["identity"]["action_uid"]
            )
            group_host = {}
            for group in ACTION_BALL_REWARD_GROUP_ORDER:
                row = step["reward_groups"][group]
                group_host[group] = (
                    None
                    if row is None
                    else {
                        field: self._ops.to_host_list(value)
                        for field, value in row.items()
                    }
                )
            for env_id, raw_uid in enumerate(action_uids):
                uid = int(raw_uid)
                if uid not in samples:
                    raise RewardActivationLedgerError(
                        "Reward group sample has an unknown action UID"
                    )
                for group in ACTION_BALL_REWARD_GROUP_ORDER:
                    row = group_host[group]
                    if row is None:
                        continue
                    for field in samples[uid][group]:
                        value = row[field][env_id]
                        if field == "nonzero_term_count":
                            if (
                                type(value) not in (int, float)
                                or int(value) != value
                                or int(value) < 0
                            ):
                                raise RewardActivationLedgerError(
                                    "Reward group nonzero count is invalid"
                                )
                            samples[uid][group][field].append(int(value))
                        else:
                            samples[uid][group][field].append(
                                _normalized_output_float(
                                    value,
                                    context=(
                                        f"Reward group {group}.{field}"
                                    ),
                                )
                            )

        by_action = {}
        for action_id, uid in zip(self._action_order, self._action_uids):
            rows = []
            total_positive = math.fsum(
                math.fsum(samples[uid][group]["positive_weighted"])
                for group in ACTION_BALL_REWARD_GROUP_ORDER
            )
            total_negative = math.fsum(
                math.fsum(samples[uid][group]["negative_weighted"])
                for group in ACTION_BALL_REWARD_GROUP_ORDER
            )
            if total_positive < 0.0 or total_negative > 0.0:
                raise RewardActivationLedgerError(
                    "Reward group signed totals violate contribution polarity"
                )
            for group in ACTION_BALL_REWARD_GROUP_ORDER:
                group_samples = samples[uid][group]
                values = group_samples["weighted"]
                positive = math.fsum(group_samples["positive_weighted"])
                negative = math.fsum(group_samples["negative_weighted"])
                eligible = len(values)
                if eligible and not self._group_objective_indices[group]:
                    raise RewardActivationLedgerError(
                        "Reward group without objective terms retained samples"
                    )
                rows.append(
                    {
                        "group": group,
                        "objective_term_names": list(
                            self._group_objective_term_names[group]
                        ),
                        "diagnostic_probe_term_names": list(
                            self._group_probe_term_names[group]
                        ),
                        "eligibility": (
                            "reward_manager_evaluated_active_group_terms"
                        ),
                        "eligible_sample_count": eligible,
                        "nonzero_sample_count": sum(
                            value > 0
                            for value in group_samples[
                                "nonzero_term_count"
                            ]
                        ),
                        "weighted_sum": _normalized_output_float(
                            math.fsum(values),
                            context=f"{action_id}.{group}.weighted_sum",
                        ),
                        "weighted_p5": _linear_quantile(values, 0.05),
                        "weighted_p50": _linear_quantile(values, 0.50),
                        "weighted_p95": _linear_quantile(values, 0.95),
                        "positive_weighted_sum": (
                            _normalized_output_float(
                                positive,
                                context=(
                                    f"{action_id}.{group}.positive_weighted_sum"
                                ),
                            )
                        ),
                        "negative_weighted_sum": (
                            _normalized_output_float(
                                negative,
                                context=(
                                    f"{action_id}.{group}.negative_weighted_sum"
                                ),
                            )
                        ),
                        "positive_return_fraction": (
                            None
                            if total_positive == 0.0
                            else _normalized_output_float(
                                positive / total_positive,
                                context=(
                                    f"{action_id}.{group}.positive_fraction"
                                ),
                            )
                        ),
                        "negative_return_fraction": (
                            None
                            if total_negative == 0.0
                            else _normalized_output_float(
                                negative / total_negative,
                                context=(
                                    f"{action_id}.{group}.negative_fraction"
                                ),
                            )
                        ),
                    }
                )
            by_action[action_id] = {
                "positive_weighted_sum": _normalized_output_float(
                    total_positive,
                    context=f"{action_id}.positive_weighted_sum",
                ),
                "negative_weighted_sum": _normalized_output_float(
                    total_negative,
                    context=f"{action_id}.negative_weighted_sum",
                ),
                "groups": rows,
            }
        return by_action

    def _validate_prepared_records(self, activation, per_action, safety):
        activation_terms = {
            row["name"]: row for row in activation["terms"]
        }
        action_rows = per_action["actions"]
        if sum(row["observed_sample_count"] for row in action_rows) != activation[
            "observed_sample_count"
        ]:
            raise RewardActivationLedgerError(
                "per-action observed samples do not partition activation"
            )
        for name, aggregate in activation_terms.items():
            rows = [
                {term["name"]: term for term in action["terms"]}[name]
                for action in action_rows
            ]
            if (
                sum(row["observed_sample_count"] for row in rows)
                != aggregate["observed_sample_count"]
                or sum(row["nonzero_sample_count"] for row in rows)
                != aggregate["nonzero_sample_count"]
                or not math.isclose(
                    sum(float(row["raw_sum"]) for row in rows),
                    float(aggregate["raw_sum"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
                or not math.isclose(
                    sum(float(row["weighted_sum"]) for row in rows),
                    float(aggregate["weighted_sum"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
            ):
                raise RewardActivationLedgerError(
                    f"per-action term {name!r} does not close to activation"
                )

        action_by_id = {row["action_id"]: row for row in action_rows}
        if per_action.get("reward_group_taxonomy") != self._reward_group_taxonomy:
            raise RewardActivationLedgerError(
                "per-action Reward group taxonomy drifted from composed recipe"
            )
        taxonomy_by_name = {
            row["name"]: row
            for row in self._reward_group_taxonomy["active_terms"]
        }
        for action_id, action in action_by_id.items():
            groups = action.get("reward_groups")
            if (
                not isinstance(groups, list)
                or [row.get("group") for row in groups]
                != list(ACTION_BALL_REWARD_GROUP_ORDER)
            ):
                raise RewardActivationLedgerError(
                    f"per-action Reward groups for {action_id!r} are incomplete"
                )
            term_rows = {row["name"]: row for row in action["terms"]}
            total_positive = 0.0
            total_negative = 0.0
            positive_fractions = []
            negative_fractions = []
            partition = set()
            for group_row in groups:
                group = group_row["group"]
                objective_names = group_row.get("objective_term_names")
                probe_names = group_row.get("diagnostic_probe_term_names")
                expected_objectives = list(
                    self._group_objective_term_names[group]
                )
                expected_probes = list(self._group_probe_term_names[group])
                if (
                    objective_names != expected_objectives
                    or probe_names != expected_probes
                    or set(objective_names) & set(probe_names)
                ):
                    raise RewardActivationLedgerError(
                        f"per-action Reward group {group!r} term partition drifted"
                    )
                partition.update(objective_names)
                partition.update(probe_names)
                eligible = group_row.get("eligible_sample_count")
                nonzero = group_row.get("nonzero_sample_count")
                expected_eligible = (
                    action["observed_sample_count"]
                    if objective_names
                    else 0
                )
                if (
                    group_row.get("eligibility")
                    != "reward_manager_evaluated_active_group_terms"
                    or type(eligible) is not int
                    or eligible != expected_eligible
                    or type(nonzero) is not int
                    or nonzero < 0
                    or nonzero > eligible
                ):
                    raise RewardActivationLedgerError(
                        f"per-action Reward group {group!r} eligibility is invalid"
                    )
                weighted = _plain_finite_float(
                    group_row.get("weighted_sum"),
                    context=f"{action_id}.{group}.weighted_sum",
                )
                positive = _plain_finite_float(
                    group_row.get("positive_weighted_sum"),
                    context=f"{action_id}.{group}.positive_weighted_sum",
                )
                negative = _plain_finite_float(
                    group_row.get("negative_weighted_sum"),
                    context=f"{action_id}.{group}.negative_weighted_sum",
                )
                expected_weighted = math.fsum(
                    float(term_rows[name]["weighted_sum"])
                    for name in objective_names
                )
                if (
                    positive < 0.0
                    or negative > 0.0
                    or not math.isclose(
                        weighted,
                        positive + negative,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-7,
                    )
                    or not math.isclose(
                        weighted,
                        expected_weighted,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-7,
                    )
                ):
                    raise RewardActivationLedgerError(
                        f"per-action Reward group {group!r} does not close"
                    )
                quantiles = [
                    group_row.get("weighted_p5"),
                    group_row.get("weighted_p50"),
                    group_row.get("weighted_p95"),
                ]
                if eligible:
                    if (
                        any(
                            type(value) not in (int, float)
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            for value in quantiles
                        )
                        or not (
                            float(quantiles[0])
                            <= float(quantiles[1])
                            <= float(quantiles[2])
                        )
                    ):
                        raise RewardActivationLedgerError(
                            f"per-action Reward group {group!r} quantiles are invalid"
                        )
                elif quantiles != [None, None, None]:
                    raise RewardActivationLedgerError(
                        f"empty per-action Reward group {group!r} has quantiles"
                    )
                total_positive += positive
                total_negative += negative
                positive_fractions.append(
                    group_row.get("positive_return_fraction")
                )
                negative_fractions.append(
                    group_row.get("negative_return_fraction")
                )
            if partition != set(taxonomy_by_name):
                raise RewardActivationLedgerError(
                    "per-action Reward groups do not partition active taxonomy"
                )
            if not math.isclose(
                total_positive,
                float(action["positive_weighted_sum"]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            ) or not math.isclose(
                total_negative,
                float(action["negative_weighted_sum"]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            ):
                raise RewardActivationLedgerError(
                    "per-action signed Reward group totals do not close"
                )
            for total, fractions, label in (
                (total_positive, positive_fractions, "positive"),
                (total_negative, negative_fractions, "negative"),
            ):
                if total == 0.0:
                    if any(value is not None for value in fractions):
                        raise RewardActivationLedgerError(
                            f"undefined {label} Reward fractions must be null"
                        )
                elif (
                    any(
                        type(value) not in (int, float)
                        or isinstance(value, bool)
                        or not 0.0 <= float(value) <= 1.0
                        for value in fractions
                    )
                    or not math.isclose(
                        math.fsum(float(value) for value in fractions),
                        1.0,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-7,
                    )
                ):
                    raise RewardActivationLedgerError(
                        f"per-action {label} Reward fractions do not close"
                    )
        soft = safety["soft_limit_by_action_term"]
        if len(soft) != len(action_rows) * len(self._SOFT_LIMIT_TERM_NAMES):
            raise RewardActivationLedgerError(
                "soft-limit evidence lacks exact action-by-term coverage"
            )
        for row in soft:
            term = {
                item["name"]: item
                for item in action_by_id[row["action_id"]]["terms"]
            }[row["term_name"]]
            if (
                row["observed_sample_count"] != term["observed_sample_count"]
                or row["eligible_sample_count"] != term["observed_sample_count"]
                or row["active_sample_count"] != term["nonzero_sample_count"]
                or not math.isclose(
                    float(row["raw_sum"]),
                    float(term["raw_sum"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
                or not math.isclose(
                    float(row["weighted_sum"]),
                    float(term["weighted_sum"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
                or float(row["raw_sum"]) < 0.0
                or float(row["weighted_sum"]) > 0.0
                or row["effective"] is not True
                or row["terminal_reward"] is not False
            ):
                raise RewardActivationLedgerError(
                    "soft-limit evidence does not bind nonterminal negative activation"
                )

        grouped = {
            action_id: {"count": 0, "raw": 0.0, "weighted": 0.0}
            for action_id in self._action_order
        }
        for transition in safety["terminal_transitions"]:
            row = grouped[transition["action_id"]]
            row["count"] += int(transition["death_activation"]["active"])
            row["raw"] += float(transition["death_raw_value"])
            row["weighted"] += float(
                transition["death_weighted_contribution"]
            )
        for action_id, totals in grouped.items():
            death = {
                row["name"]: row
                for row in action_by_id[action_id]["terms"]
            }["death_penalty"]
            if (
                totals["count"] != death["nonzero_sample_count"]
                or not math.isclose(
                    totals["raw"],
                    float(death["raw_sum"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
                or not math.isclose(
                    totals["weighted"],
                    float(death["weighted_sum"]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-7,
                )
            ):
                raise RewardActivationLedgerError(
                    "hard-safety transitions do not close to per-action death"
                )

    def _prepare_action_ball_conservation(self, ppo_update, activation):
        if (
            self._episode_closure_stats["environment_step_count"]
            != self._expected_step_count
        ):
            raise RewardActivationLedgerError(
                "episode-segmented closure lacks exact environment-step coverage"
            )
        if self._episode_external_reset_pending:
            raise RewardActivationLedgerError(
                "external Reward reset identity has not reached the next "
                "captured environment boundary"
            )
        self._validate_open_manager_episode_sums(
            context="pre-optimizer open episode"
        )
        local_terms = {
            name: self._ops.to_host_list(self._episode_term_sums[name])
            for name in self._manager_term_names
        }
        reward_sums = self._ops.to_host_list(
            self._episode_reward_buf_sums
        )
        step_counts = self._ops.to_host_list(self._episode_step_counts)
        generations = self._ops.to_host_list(
            self._episode_reset_generations
        )
        action_uids = self._ops.to_host_list(self._episode_action_uids)
        open_segments = []
        for env_id in range(self._num_envs):
            term_values = [
                float(local_terms[name][env_id])
                for name in self._manager_term_names
            ]
            term_total = sum(term_values)
            reward_total = float(reward_sums[env_id])
            error = abs(term_total - reward_total)
            tolerance = self._activation._closure_atol + (
                self._activation._closure_rtol * abs(reward_total)
            )
            if error > tolerance:
                raise RewardActivationLedgerError(
                    "open episode sum(reward_buf) does not close to all term sums"
                )
            self._update_episode_error(
                "max_abs_reward_buf_vs_term_sum_error", error
            )
            self._episode_closure_stats[
                "reward_buf_term_sum_comparison_count"
            ] += 1
            open_segments.append(
                {
                    "env_id": env_id,
                    "reset_generation": int(generations[env_id]),
                    "action_uid": int(action_uids[env_id]),
                    "step_count": int(step_counts[env_id]),
                    "reward_buf_sum": reward_total,
                    "all_term_sum": term_total,
                    "reward_buf_vs_all_terms_abs_error": error,
                    "local_term_sums": term_values,
                    "status": "OPEN_NOT_E2",
                }
            )
        completed = [
            {
                **row,
                "segment_key": [
                    row["env_id"],
                    row["reset_generation"],
                ],
            }
            for row in self._completed_episode_segments
        ]
        e2_eligible = any(
            row["step_count"] > 0 and not row["administrative_reset"]
            for row in completed
        )
        dashboard_status = (
            "PASS"
            if self._episode_reset_batches
            else "NOT_OBSERVED_NO_RESET"
        )
        receipt = {
            "event": "hope_reward_episode_segmented_closure_update",
            "schema_version": 1,
            "status": "PASS",
            "evidence_source": "live_isaac_reward_manager",
            "capture_mode": "reward_manager_reset_pre_clear_hook",
            "task_kind": "action_ball",
            "ppo_update": ppo_update,
            "recipe_sha256": activation["recipe_sha256"],
            "step_dt_s": activation["step_dt_s"],
            "max_episode_length_s": self._max_episode_length_s,
            "num_envs": self._num_envs,
            "segment_key_fields": ["env_id", "reset_generation"],
            "all_reward_manager_term_names": list(
                self._manager_term_names
            ),
            "completed_episode_count": len(completed),
            "completed_episode_segments": completed,
            "reset_batches": list(self._episode_reset_batches),
            "open_episode_count": len(open_segments),
            "open_episode_segments": open_segments,
            "dashboard_normalization": {
                "status": dashboard_status,
                "reset_batch_count": len(self._episode_reset_batches),
                "reason": (
                    None
                    if self._episode_reset_batches
                    else "no RewardManager.reset batch occurred in this PPO update"
                ),
            },
            "checks": {
                **self._episode_closure_stats,
                "status": "PASS",
                "all_step_reward_buf_equals_all_term_sums": "PASS",
                "all_episode_sums_equal_captured_term_sums": "PASS",
                "all_observed_dashboard_values_normalized_exactly": (
                    "PASS"
                    if self._episode_reset_batches
                    else "NOT_OBSERVED_NO_RESET"
                ),
                "all_reset_episode_sums_cleared": "PASS",
                "exact_environment_step_coverage": "PASS",
            },
            "e2_eligible": e2_eligible,
            "e2_ineligible_reason": (
                None
                if e2_eligible
                else "no non-administrative completed live episode segment in this update"
            ),
        }
        canonical_effective_reward_activation_json(receipt)
        return receipt

    def prepare_update(self, ppo_update, *, joint_first_policy_step_sequence):
        if self._open_step is not None or self._pending is not None:
            raise RewardActivationLedgerError(
                "cannot prepare Reward evidence with an open/pending transaction"
            )
        if (
            type(joint_first_policy_step_sequence) is not int
            or joint_first_policy_step_sequence < 0
            or len(self._step_rows) != self._expected_step_count
            or self._per_action is None
        ):
            raise RewardActivationLedgerError(
                "Reward evidence lacks exact joint-step/action coverage"
            )
        activation = self._activation.prepare_update(ppo_update)
        reward_groups_by_action = self._prepared_reward_group_rows()
        action_rows = []
        for action_id, uid in zip(self._action_order, self._action_uids):
            accumulated = self._per_action[uid]
            observed = self._host_int(
                accumulated["observed_sample_count"],
                name=f"{action_id}.observed_sample_count",
            )
            term_rows = []
            for name in sorted(self._term_names):
                term = accumulated["terms"][name]
                term_rows.append(
                    {
                        "name": name,
                        "observed_sample_count": observed,
                        "nonzero_sample_count": self._host_int(
                            term["nonzero_sample_count"],
                            name=f"{action_id}.{name}.nonzero",
                        ),
                        "raw_sum": self._host_float(
                            term["raw_sum"], name=f"{action_id}.{name}.raw"
                        ),
                        "weighted_sum": self._host_float(
                            term["weighted_sum"],
                            name=f"{action_id}.{name}.weighted",
                        ),
                    }
                )
            action_rows.append(
                {
                    "action_id": action_id,
                    "action_uid": uid,
                    "observed_sample_count": observed,
                    "positive_weighted_sum": reward_groups_by_action[
                        action_id
                    ]["positive_weighted_sum"],
                    "negative_weighted_sum": reward_groups_by_action[
                        action_id
                    ]["negative_weighted_sum"],
                    "reward_groups": reward_groups_by_action[action_id][
                        "groups"
                    ],
                    "terms": term_rows,
                }
            )
        per_action = {
            "event": "hope_effective_reward_activation_by_action_update",
            "schema_version": 2,
            "recipe_sha256": activation["recipe_sha256"],
            "task_kind": "action_ball",
            "ppo_update": ppo_update,
            "step_dt_s": activation["step_dt_s"],
            "manifest_sha256": self._manifest_sha256,
            "action_order": list(self._action_order),
            "reward_group_taxonomy": self._reward_group_taxonomy,
            "actions": action_rows,
        }
        action_rows_by_id = {
            row["action_id"]: row for row in action_rows
        }
        soft_rows = []
        for action_id, uid in zip(self._action_order, self._action_uids):
            terms_by_name = {
                row["name"]: row
                for row in action_rows_by_id[action_id]["terms"]
            }
            accumulated_terms = self._per_action[uid]["terms"]
            for name in self._SOFT_LIMIT_TERM_NAMES:
                term = terms_by_name[name]
                soft_rows.append(
                    {
                        "action_id": action_id,
                        "action_uid": uid,
                        "term_name": name,
                        "observed_sample_count": term[
                            "observed_sample_count"
                        ],
                        "eligible_sample_count": term[
                            "observed_sample_count"
                        ],
                        "active_sample_count": term[
                            "nonzero_sample_count"
                        ],
                        "raw_sum": term["raw_sum"],
                        "weighted_sum": term["weighted_sum"],
                        "terminated_active_sample_count": self._host_int(
                            accumulated_terms[name][
                                "terminated_nonzero_sample_count"
                            ],
                            name=f"{action_id}.{name}.terminated_active",
                        ),
                        "step_dt_s": activation["step_dt_s"],
                        "effective": True,
                        "terminal_reward": False,
                    }
                )

        transitions = []
        termination_order = None
        death_weight = float(self._metadata["death_penalty"]["weight"])
        for step_index, step in enumerate(self._step_rows):
            pre = step["pre"]
            post = step["post"]
            if termination_order is None:
                termination_order = post["term_order"]
            elif post["term_order"] != termination_order:
                raise RewardActivationLedgerError(
                    "termination term order drifted inside one PPO update"
                )
            identity_host = {
                field: self._ops.to_host_list(step["identity"][field])
                for field in (
                    "action_uid",
                    "reset_generation",
                    "swing_generation",
                )
            }
            pre_masks = {
                name: self._ops.to_host_list(pre["reason_masks"][name])
                for name in termination_order
            }
            post_masks = {
                name: self._ops.to_host_list(post["reason_masks"][name])
                for name in termination_order
            }
            terminated = self._ops.to_host_list(post["terminated"])
            time_outs = self._ops.to_host_list(post["time_outs"])
            death_raw = self._ops.to_host_list(step["death_raw"])
            death_weighted = self._ops.to_host_list(
                step["death_weighted"]
            )
            for env_id in range(self._num_envs):
                if not bool(terminated[env_id]):
                    continue
                terms = sorted(
                    name
                    for name in termination_order
                    if bool(post_masks[name][env_id])
                )
                if not terms:
                    raise RewardActivationLedgerError(
                        "terminated sample has no post-step reason mask"
                    )
                rising_terms = sorted(
                    name
                    for name in terms
                    if not bool(pre_masks[name][env_id])
                )
                if not rising_terms:
                    raise RewardActivationLedgerError(
                        "terminated sample has no rising terminal reason edge"
                    )
                uid = int(identity_host["action_uid"][env_id])
                reason_classes = self._reason_classes(terms)
                hard_safety = any(
                    name in reason_classes
                    for name in ("table_hit", "fall", "hard_limit")
                )
                transition = {
                    "action_id": self._uid_to_action[uid],
                    "action_uid": uid,
                    "env_id": env_id,
                    "common_step_counter": step[
                        "common_step_counter_end"
                    ],
                    "joint_policy_step_sequence": (
                        joint_first_policy_step_sequence + step_index
                    ),
                    "reset_generation": int(
                        identity_host["reset_generation"][env_id]
                    ),
                    "swing_generation": int(
                        identity_host["swing_generation"][env_id]
                    ),
                    "birth_receipt_sha256": step["identity"][
                        "birth_receipt_sha256"
                    ][env_id],
                    "reason_classes": reason_classes,
                    "primary_reason_class": self._primary_reason_class(
                        reason_classes
                    ),
                    "termination_terms": terms,
                    "rising_termination_terms": rising_terms,
                    "timed_out_same_step": bool(time_outs[env_id]),
                    "pre_terminal_reason_mask": {
                        name: bool(pre_masks[name][env_id])
                        for name in termination_order
                    },
                    "post_terminal_reason_mask": {
                        name: bool(post_masks[name][env_id])
                        for name in termination_order
                    },
                    "death_raw_value": float(death_raw[env_id]),
                    "death_weighted_contribution": float(
                        death_weighted[env_id]
                    ),
                    "death_activation": {
                        "term_name": "death_penalty",
                        "eligible": True,
                        "active": hard_safety,
                        "raw": float(death_raw[env_id]),
                        "weighted": float(death_weighted[env_id]),
                        "step_dt_s": activation["step_dt_s"],
                        "effective": True,
                    },
                    "reason_specific_penalties": [],
                }
                expected_raw = 1.0 if hard_safety else 0.0
                expected_weighted = (
                    ACTION_BALL_ADOPTED_DEATH_PER_TERMINATION
                    if hard_safety
                    else 0.0
                )
                if not math.isclose(
                    transition["death_raw_value"],
                    expected_raw,
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                ) or not math.isclose(
                    transition["death_weighted_contribution"],
                    expected_weighted,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ) or not math.isclose(
                    death_weight * activation["step_dt_s"],
                    ACTION_BALL_ADOPTED_DEATH_PER_TERMINATION,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise RewardActivationLedgerError(
                        "terminal transition hard-safety death eligibility is inconsistent"
                    )
                transition["transition_id"] = self._transition_id(
                    ppo_update, transition
                )
                transitions.append(transition)
        safety = {
            "event": "hope_reward_safety_transition_update",
            "schema_version": 2,
            "recipe_sha256": activation["recipe_sha256"],
            "ppo_update": ppo_update,
            "step_dt_s": activation["step_dt_s"],
            "coverage": "complete_update",
            "manifest_sha256": self._manifest_sha256,
            "action_order": list(self._action_order),
            "soft_limit_term_names": list(self._SOFT_LIMIT_TERM_NAMES),
            "hard_safety_termination_term_names": list(
                self._HARD_SAFETY_TERMINATION_TERMS
            ),
            "reference_envelope_termination_term_names": list(
                self._REFERENCE_ENVELOPE_TERMINATION_TERMS
            ),
            "termination_term_order": list(termination_order or ()),
            "soft_limit_by_action_term": soft_rows,
            "terminal_transitions": transitions,
        }
        self._validate_prepared_records(activation, per_action, safety)
        action_ball_conservation = (
            self._prepare_action_ball_conservation(ppo_update, activation)
        )
        for record in (
            activation,
            per_action,
            safety,
            action_ball_conservation,
        ):
            canonical_effective_reward_activation_json(record)
        prepared = {
            "ppo_update": ppo_update,
            "activation": activation,
            "per_action": per_action,
            "safety": safety,
            "action_ball_conservation": action_ball_conservation,
            "status": "frozen_validated_before_optimizer",
        }
        self._pending_sha256 = hashlib.sha256(
            canonical_effective_reward_activation_json(prepared).encode(
                "utf-8"
            )
        ).hexdigest()
        self._pending = prepared
        return prepared

    def acknowledge_update(self, prepared):
        if prepared is not self._pending:
            raise RewardActivationLedgerError(
                "Reward evidence acknowledgement token is stale/foreign"
            )
        digest = hashlib.sha256(
            canonical_effective_reward_activation_json(prepared).encode(
                "utf-8"
            )
        ).hexdigest()
        if digest != self._pending_sha256:
            raise RewardActivationLedgerError(
                "prepared action-bound Reward evidence was mutated before acknowledgement"
            )
        self._activation.acknowledge_update(prepared["activation"])
        self._step_rows = []
        self._per_action = None
        self._completed_episode_segments = []
        self._episode_reset_batches = []
        self._episode_closure_stats = self._new_episode_closure_stats()
        self._pending_sha256 = None
        self._pending = None


def canonical_effective_reward_activation_json(record):
    """Return the canonical one-line encoding for a runtime activation record."""

    if not isinstance(record, Mapping):
        raise RewardActivationLedgerError("runtime reward activation record must be a mapping")
    try:
        return json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RewardActivationLedgerError(
            "runtime reward activation record is not canonical finite JSON"
        ) from exc
