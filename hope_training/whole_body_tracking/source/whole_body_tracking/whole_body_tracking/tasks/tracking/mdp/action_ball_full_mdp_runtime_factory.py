"""Code-owned construction root for the fresh ActionBall full-MDP graph.

The environment calls :func:`construct_action_ball_full_mdp_runtime_graph`
exactly once after IsaacLab has constructed CommandManager and before it
constructs ActionManager or ObservationManager.  Launchers, YAML and callers
are deliberately absent from this API.

The sole constructible path is the generic-N, unauthorized single-action lean graph.
It builds every identity off-side, materializes real manager configurations,
and publishes only once.  Missing causal runtime producers remain explicit
HOLDs; a fixture, wrapper, receipt, or source digest cannot satisfy them.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import json
import math
import sys
import threading
from typing import Mapping, NoReturn
import weakref

import torch

try:
    from . import action_ball_full_mdp_reward_contract as reward_contract
except ImportError:
    import action_ball_full_mdp_reward_contract as reward_contract


RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True

INSTALLED_REWARD_GRAPH_KIND = (
    "action_ball_full_mdp_installed_reward_graph_receipt_v2"
)
REWARD_GRAPH_ATTR = "action_ball_full_mdp_reward_graph"
INSTALLED_REWARD_RECEIPT_ATTR = (
    "_action_ball_full_mdp_installed_reward_graph_receipt"
)
NUMERIC_REWARD_FACTORY_HOLD_REASONS = (
    "constructed_runtime_reward_graph_producer_absent",
    "real_four_shot_unit_income_phase_support_producer_absent",
    "launcher_owned_finite_candidate_set_producer_absent",
    "fourteen_live_reward_consumers_not_factory_bound",
)

# This is the executable critical path, not an inventory of every receipt,
# projection or helper object.  Each entry is (stage, direct prerequisites).
# One stage may construct several tightly coupled owners, but it may publish
# nothing to ``env`` until the complete component registry can be installed.
# A Protocol-shaped object or test fixture cannot satisfy a stage.
FULL_MDP_RUNTIME_FACTORY_DAG = (
    ("verified_split_asset", ()),
    ("command_owners", ()),
    (
        "independent_reset_genesis",
        ("verified_split_asset", "command_owners"),
    ),
    ("scene_question_cadence", ("command_owners", "independent_reset_genesis")),
    ("hot_leaf_owners", ("scene_question_cadence",)),
    ("action_epoch_owner", ("independent_reset_genesis", "hot_leaf_owners")),
    ("device_r05_owner", ("action_epoch_owner", "hot_leaf_owners")),
    ("top_runtime_owner", ("device_r05_owner", "action_epoch_owner")),
    ("reward_manager_cfg", ("top_runtime_owner", "action_epoch_owner")),
    ("observation_manager_cfg", ("top_runtime_owner", "action_epoch_owner")),
    ("termination_manager_cfg", ("top_runtime_owner",)),
    (
        "live_physx_fact_subscriber",
        (
            "top_runtime_owner",
            "reward_manager_cfg",
            "observation_manager_cfg",
            "termination_manager_cfg",
        ),
    ),
    (
        "atomic_component_install",
        (
            "live_physx_fact_subscriber",
        ),
    ),
    (
        "causal_step_runtime",
        ("atomic_component_install", "top_runtime_owner"),
    ),
    (
        "first_optimizer_update",
        ("causal_step_runtime", "action_epoch_owner"),
    ),
)

FIRST_UNRESOLVED_PRODUCTION_NODE = (
    "row_wise_shot_close_and_four_shot_carry_state_absent"
)

UNRESOLVED_PRODUCTION_NODES = (
    FIRST_UNRESOLVED_PRODUCTION_NODE,
    "post_physics_physical_r06_runtime_unverified",
    "real_gym_first_optimizer_update_unverified",
)


class ActionBallFullMdpRuntimeFactoryHold(RuntimeError):
    """The production graph lacks one or more reviewed constructor nodes."""


class ActionBallFullMdpRewardMaterializationHold(
    ActionBallFullMdpRuntimeFactoryHold
):
    """The numeric authority cannot be installed into the exact live graph."""


@dataclass(frozen=True)
class _OffsideConstructionSeed:
    """Cold factory-owned inputs retained before any environment install."""

    motion_owner: object
    racket_owner: object
    racket_cfg: object
    motion_parent_authority: object
    motion_parent_receipt: object
    reset_genesis_authority: object
    reset_genesis_receipt: object
    num_envs: int
    device: object


@dataclass(frozen=True)
class _OffsideDeviceProfileBundle:
    """Real diagnostic profile retained with the unpublished genesis."""

    seed: _OffsideConstructionSeed
    profile_spec: object
    profile_owner: object
    profile_receipt: object


@dataclass(frozen=True)
class _OffsideQuestionInputs:
    """Profile root retained while the live scene/Physical root is built."""

    profile: _OffsideDeviceProfileBundle


@dataclass(frozen=True)
class _OffsidePhysicalSceneInputs:
    """Exact live-scene roots retained before Physical can consume genesis."""

    question: _OffsideQuestionInputs
    scene_port: object
    diagnostic_capacity_binding: object
    physical_owner: object
    physical_question_core: object


@dataclass(frozen=True)
class _OffsideRecurringQuestionInputs:
    """Reusable D05 composer joined to the cold Physical/profile graph."""

    physical: _OffsidePhysicalSceneInputs
    recurring_question_bundle: object


@dataclass(frozen=True)
class _OffsideColdLeafInputs:
    """Real Racket-owned R03 and diagnostic R06 retained before R07."""

    recurring: _OffsideRecurringQuestionInputs
    r03_owner: object
    r06_owner: object


@dataclass(frozen=True)
class _OffsideLeanRuntimeInputs:
    """Complete diagnostic identities retained before one atomic install."""

    cold: _OffsideColdLeafInputs
    r07_owner: object
    r07_plant_fact_adapter: object
    epoch_owner: object
    device_r05_owner: object
    reward_graph: object
    reward_manager_cfg: dict[str, object]
    observation_source: object
    observation_manager_cfg: dict[str, object]
    termination_manager_cfg: dict[str, object]
    lean_runtime_owner: object


@dataclass(frozen=True)
class ActionBallFullMdpInstalledRewardGraphView:
    """Non-capability projection of one exact installed manager graph."""

    schema_version: int
    kind: str
    numeric_authority_sha256: str
    numeric_materialization_sha256: str
    resolved_graph_receipt_sha256: str
    installed_manager_graph_sha256: str
    ordered_manager_names: tuple[str, ...]
    ordered_payment_consumers: tuple[str, ...]
    diagnostic_unauthorized: bool
    runtime_integrated: bool
    launch_authorized: bool


class ActionBallFullMdpInstalledRewardGraphReceipt:
    """Opaque factory receipt bound to the graph and installed cfg identities."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NoReturn:
        del cls
        raise TypeError("installed Reward graph receipts are factory-issued only")

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("installed Reward graph receipts are immutable")

    def __copy__(self) -> NoReturn:
        raise TypeError("installed Reward graph receipts cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("installed Reward graph receipts cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("installed Reward graph receipts cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("installed Reward graph receipts cannot be serialized")


@dataclass(frozen=True)
class _InstalledRewardGraphRecord:
    installation_identity: object
    env_identity: object
    graph: object
    manager_cfg: dict[str, object]
    numeric_authority_owner: object
    numeric_authority_receipt: object
    view: ActionBallFullMdpInstalledRewardGraphView


_INSTALLED_REWARD_REGISTRY_LOCK = threading.RLock()
_INSTALLED_REWARD_REGISTRY: weakref.WeakKeyDictionary[
    ActionBallFullMdpInstalledRewardGraphReceipt,
    _InstalledRewardGraphRecord,
] = weakref.WeakKeyDictionary()


def _mint_installed_reward_graph_receipt(
    record: _InstalledRewardGraphRecord,
) -> ActionBallFullMdpInstalledRewardGraphReceipt:
    receipt = object.__new__(ActionBallFullMdpInstalledRewardGraphReceipt)
    with _INSTALLED_REWARD_REGISTRY_LOCK:
        _INSTALLED_REWARD_REGISTRY[receipt] = record
    return receipt


def _reward_dependencies():
    try:
        budget = importlib.import_module("action_ball_fresh_reward_budget")
        rewards = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_rewards"
        )
        config = importlib.import_module(
            "whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg"
        )
        managers = importlib.import_module("isaaclab.managers")
    except Exception as exc:
        raise ActionBallFullMdpRewardMaterializationHold(
            "exact Reward budget/graph/config dependencies are unavailable"
        ) from exc
    return budget, rewards, config, managers.RewardTermCfg


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ActionBallFullMdpRewardMaterializationHold(
            f"{label} must be an exact numeric-authority mapping"
        )
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _path(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ActionBallFullMdpRewardMaterializationHold(
                f"numeric authority lacks exact path {path}"
            )
        current = current[component]
    return current


def _fraction(
    value: object,
    *,
    label: str,
    positive: bool = True,
) -> tuple[Fraction, float]:
    item = _mapping(value, label=label)
    if frozenset(item) != frozenset(("numerator", "denominator")):
        raise ActionBallFullMdpRewardMaterializationHold(
            f"{label} must contain only numerator/denominator"
        )
    numerator = item["numerator"]
    denominator = item["denominator"]
    if (
        isinstance(numerator, bool)
        or type(numerator) is not int
        or isinstance(denominator, bool)
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            f"{label} must be an exact finite fraction"
        )
    exact = Fraction(numerator, denominator)
    if positive and exact <= 0:
        raise ActionBallFullMdpRewardMaterializationHold(
            f"{label} must be strictly positive"
        )
    host = float(exact)
    if not math.isfinite(host) or (positive and host <= 0.0):
        raise ActionBallFullMdpRewardMaterializationHold(
            f"{label} cannot materialize to a finite positive host value"
        )
    return exact, host


def _positive_host_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ActionBallFullMdpRewardMaterializationHold(label + " must be a host number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ActionBallFullMdpRewardMaterializationHold(label + " must be finite positive")
    return result


def _require_owner_numeric_profiles(
    graph: object,
    selected: Mapping[str, object],
) -> None:
    owners = getattr(graph, "_owners", None)
    r06_profile = getattr(getattr(owners, "r06", None), "profile", None)
    r07_profile = getattr(getattr(owners, "r07", None), "profile", None)
    placement = _mapping(
        selected.get("placement_profile"),
        label="selected_numeric_parameters.placement_profile",
    )
    for field in ("alpha_broad", "sigma_broad_m", "sigma_narrow_m"):
        _exact, expected = _fraction(
            placement.get(field),
            label=f"selected_numeric_parameters.placement_profile.{field}",
        )
        actual = getattr(r06_profile, field, None)
        if type(actual) is not float or actual.hex() != expected.hex():
            raise ActionBallFullMdpRewardMaterializationHold(
                f"live R06 placement profile differs at {field}"
            )
    recovery = _path(
        selected,
        "manager_weights.recovery.recovery_pose",
    )
    _exact, expected_recovery = _fraction(
        recovery,
        label="selected_numeric_parameters.manager_weights.recovery.recovery_pose",
    )
    actual_recovery = getattr(r07_profile, "reward_weight", None)
    if (
        type(actual_recovery) is not float
        or actual_recovery.hex() != expected_recovery.hex()
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "live R07 owner reward weight differs from numeric authority"
        )


def _consume_numeric_authority(
    *,
    budget: object,
    numeric_authority_owner: object,
    numeric_authority_receipt: object,
) -> tuple[Mapping[str, object], str, str, str]:
    if type(numeric_authority_owner) is not budget.NumericRewardAuthorityOwner:
        raise ActionBallFullMdpRewardMaterializationHold(
            "numeric authority owner exact class/source identity differs"
        )
    if type(numeric_authority_receipt) is not budget.NumericRewardAuthorityReceipt:
        raise ActionBallFullMdpRewardMaterializationHold(
            "numeric authority receipt must be the exact opaque owner-issued type"
        )
    try:
        consumed = numeric_authority_owner.consume(numeric_authority_receipt)
    except Exception as exc:
        raise ActionBallFullMdpRewardMaterializationHold(
            "numeric authority receipt is foreign, replayed, or not owner-issued"
        ) from exc
    authority = _mapping(consumed, label="consumed numeric authority")
    if (
        authority.get("kind") != budget.NUMERIC_AUTHORITY_KIND
        or authority.get("scope") != budget.NUMERIC_AUTHORITY_SCOPE
        or authority.get("diagnostic_unauthorized")
        is not budget.NUMERIC_DIAGNOSTIC_UNAUTHORIZED
        or authority.get("runtime_integrated")
        is not budget.NUMERIC_RUNTIME_INTEGRATED
        or authority.get("launch_authorized")
        is not budget.NUMERIC_LAUNCH_AUTHORIZED
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "owner-issued numeric authority flags/scope differ"
        )
    payload = _mapping(authority.get("authority_payload"), label="authority_payload")
    authority_sha = _sha256(
        authority.get("authority_sha256"), label="numeric authority SHA"
    )
    if budget.canonical_sha256(payload) != authority_sha:
        raise ActionBallFullMdpRewardMaterializationHold(
            "owner-issued numeric authority content SHA differs"
        )
    materialization_sha = _sha256(
        authority.get("materialization_sha256"),
        label="numeric materialization SHA",
    )
    inputs = _mapping(payload.get("input_receipts"), label="authority input receipts")
    resolved_graph_sha = _sha256(
        inputs.get("resolved_graph_receipt_sha256"),
        label="resolved graph receipt SHA",
    )
    return payload, authority_sha, materialization_sha, resolved_graph_sha


def _manager_cfg_rows(
    manager_cfg: object,
    *,
    templates: tuple[object, ...],
    reward_term_type: type,
    graph_attr: str,
) -> tuple[dict[str, object], ...]:
    if type(manager_cfg) is not dict:
        raise ActionBallFullMdpRewardMaterializationHold(
            "materialized RewardManager config must be an ordered plain dict"
        )
    names = tuple(term.manager_name for term in templates)
    if (
        tuple(manager_cfg) != names
        or names != reward_contract.MANAGER_NAMES
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "installed RewardManager graph differs from the shared order"
        )
    rows: list[dict[str, object]] = []
    for index, template in enumerate(templates):
        term = manager_cfg[template.manager_name]
        if type(term) is not reward_term_type:
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed RewardTermCfg type differs: {template.manager_name}"
            )
        if term.func is not template.func:
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed Reward callable differs: {template.manager_name}"
            )
        weight = term.weight
        if isinstance(weight, bool) or type(weight) not in (int, float):
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed Reward weight is not a host number: {template.manager_name}"
            )
        weight_float = float(weight)
        if not math.isfinite(weight_float) or weight_float <= 0.0:
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed Reward weight is not finite positive: {template.manager_name}"
            )
        params = term.params
        lifecycle = index < reward_contract.LIFECYCLE_PAYMENT_COUNT
        if type(params) is not dict or (
            lifecycle and params.get("graph_attr") != graph_attr
        ) or (not lifecycle and "graph_attr" in params):
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed Reward graph binding differs: {template.manager_name}"
            )
        expected_param_keys = {"graph_attr"} if lifecycle else set()
        if template.scale_source is not None:
            expected_param_keys.add("std")
        expected_param_keys.update(name for name, _value in template.fixed_func_params)
        if set(params) != expected_param_keys:
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed Reward params differ: {template.manager_name}"
            )
        if any(
            params[name] != expected
            for name, expected in template.fixed_func_params
        ):
            raise ActionBallFullMdpRewardMaterializationHold(
                f"installed fixed Reward params differ: {template.manager_name}"
            )
        normalized_params = {}
        for name, value in sorted(params.items()):
            if isinstance(value, bool) or type(value) not in (
                str,
                int,
                float,
                tuple,
            ):
                raise ActionBallFullMdpRewardMaterializationHold(
                    f"installed Reward param is not canonical: {template.manager_name}.{name}"
                )
            if type(value) in (int, float):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ActionBallFullMdpRewardMaterializationHold(
                        f"installed Reward param is nonfinite: {template.manager_name}.{name}"
                    )
                normalized_params[name] = {"float_hex": numeric.hex()}
            elif type(value) is tuple:
                if (
                    name != "body_names"
                    or not value
                    or any(type(item) is not str or not item for item in value)
                    or len(set(value)) != len(value)
                    or reward_contract.HELD_RACKET_WRIST_BODY_NAME in value
                ):
                    raise ActionBallFullMdpRewardMaterializationHold(
                        "installed Reward body scope is not canonical: "
                        + template.manager_name
                    )
                normalized_params[name] = list(value)
            else:
                normalized_params[name] = value
        rows.append(
            {
                "manager_name": template.manager_name,
                "payment_consumer": template.payment_consumer,
                "owner_role": template.owner_role,
                "callable_module": template.func.__module__,
                "callable_qualname": template.func.__qualname__,
                "manager_weight_float_hex": weight_float.hex(),
                "params": normalized_params,
            }
        )
    return tuple(rows)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_unique_manager_seam(env: object) -> None:
    if (
        getattr(
            env,
            "_action_ball_full_mdp_manager_construction_state",
            None,
        )
        != "command_manager_ready"
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "full-MDP runtime factory was called outside its unique manager seam"
        )
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None or not callable(
        getattr(command_manager, "get_term", None)
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "full-MDP runtime factory has no constructed CommandManager"
        )
    try:
        motion = command_manager.get_term("motion")
        racket = command_manager.get_term("racket_target")
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "full-MDP runtime factory has no exact Motion/Racket command owners"
        ) from exc
    if motion is None or racket is None or motion is racket:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "full-MDP Motion/Racket command identities are missing or aliased"
        )
    if hasattr(env, "action_manager") or hasattr(env, "observation_manager"):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "full-MDP runtime factory was called after action/observation construction"
        )


def _require_single_action_lean_factory_env(env: object) -> None:
    """Admit only the code-owned disposable canary before minting genesis.

    Checkpoint/cold-restore modes are rejected by ``train.py`` and the env
    constructor before this seam.  We still reject any retained marker here;
    absence is the expected EnvCfg state, never evidence of save authority.
    The canary module's exact false flags are the local no-save declaration.
    """

    try:
        canary = importlib.import_module(
            "action_ball_full_mdp_canary_target_profile"
        )
        split_asset = importlib.import_module(
            "whole_body_tracking.tasks.tracking.config.agibot_a3."
            "action_ball_full_mdp_split_asset"
        )
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact diagnostic scene/profile/command dependencies are unavailable"
        ) from exc
    require_split_asset = getattr(
        split_asset,
        "require_action_ball_full_mdp_split_asset",
        None,
    )
    direct_require_split_asset = getattr(
        split_asset,
        "require_action_ball_full_mdp_split_asset",
        None,
    )
    if (
        not callable(require_split_asset)
        or require_split_asset is not direct_require_split_asset
        or getattr(require_split_asset, "__module__", None)
        != split_asset.__name__
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact v3 split-asset consumer is unavailable"
        )
    try:
        verified_model_path = require_split_asset()
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "v3 split-rubber asset failed independent reconstruction before "
            "reset genesis and runtime-owner construction"
        ) from exc
    # These modules necessarily constructed the cfg and CommandManager terms
    # already.  Resolve those resident production identities instead of
    # importing a second package graph at the manager seam.
    scene = sys.modules.get(
        "whole_body_tracking.tasks.tracking.config.agibot_a3."
        "action_ball_full_mdp_ball_scene"
    )
    commands = sys.modules.get(
        "whole_body_tracking.tasks.tracking.mdp.commands"
    )
    hope_commands = sys.modules.get(
        "whole_body_tracking.tasks.tracking.mdp.hope_commands"
    )
    config = sys.modules.get(
        "whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg"
    )
    if (
        scene is None
        or commands is None
        or hope_commands is None
        or config is None
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact diagnostic scene/profile/command/config dependencies are unavailable "
            "before ObservationManager"
        )

    cfg = getattr(env, "cfg", None)
    command_cfg = getattr(cfg, "commands", None)
    motion_cfg = getattr(command_cfg, "motion", None)
    racket_cfg = getattr(command_cfg, "racket_target", None)
    scene_spec = getattr(cfg, "action_ball_full_mdp_ball_scene_spec", None)
    robot_spawn = getattr(
        getattr(getattr(cfg, "scene", None), "robot", None),
        "spawn",
        None,
    )
    if getattr(robot_spawn, "usd_path", None) != verified_model_path:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "live EnvCfg robot spawn differs from the reconstructed v3 split asset"
        )
    try:
        motion = env.command_manager.get_term("motion")
        racket = env.command_manager.get_term("racket_target")
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "diagnostic canary lacks exact constructed command owners"
        ) from exc

    failures: list[str] = []
    cfg_roles = {
        getattr(
            config,
            "HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg",
            None,
        ): "A",
        getattr(
            config,
            "HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg",
            None,
        ): "C",
    }
    expected_cfg_role = cfg_roles.get(type(cfg))
    family_resolver = getattr(config, "action_ball_full_mdp_family_role", None)
    if expected_cfg_role is None or not callable(family_resolver):
        failures.append("env_cfg_is_not_exact_registered_full_mdp_leaf")
    else:
        try:
            live_cfg_role = family_resolver(cfg)
        except Exception:
            failures.append("live_env_cfg_family_authority_rejected")
        else:
            if live_cfg_role != expected_cfg_role:
                failures.append("live_env_cfg_family_authority_differs")
    if type(motion) is not commands.MotionCommand:
        failures.append("motion_owner_exact_type_differs")
    if type(racket) is not hope_commands.RacketTargetCommand:
        failures.append("racket_owner_exact_type_differs")
    if type(motion_cfg) is not commands.MotionCommandCfg:
        failures.append("motion_cfg_exact_type_differs")
    if type(racket_cfg) is not hope_commands.RacketTargetCommandCfg:
        failures.append("racket_cfg_exact_type_differs")
    live_racket_cfg = getattr(racket, "cfg", None)
    if type(live_racket_cfg) is not hope_commands.RacketTargetCommandCfg:
        failures.append("live_racket_cfg_exact_type_differs")
    if getattr(racket_cfg, "target_mode", None) != "action_ball_full_mdp":
        failures.append("racket_target_mode_is_not_fresh_full_mdp")
    if getattr(live_racket_cfg, "target_mode", None) != "action_ball_full_mdp":
        failures.append("live_racket_target_mode_is_not_fresh_full_mdp")
    if type(getattr(env, "num_envs", None)) is not int or env.num_envs <= 0:
        failures.append("num_envs_must_be_positive_exact_int")
    cfg_num_envs = getattr(getattr(cfg, "scene", None), "num_envs", None)
    if type(cfg_num_envs) is not int or cfg_num_envs != getattr(
        env, "num_envs", None
    ):
        failures.append("cfg_env_num_envs_differ")
    if type(scene_spec) is not scene.ActionBallFullMdpDiagnosticBallSceneSpec:
        failures.append("scene_is_not_exact_diagnostic_n2_spec")
    else:
        if scene_spec.flight_capacity != 2:
            failures.append("diagnostic_scene_capacity_differs")
        if scene_spec.formal_capacity_receipt_sha256 is not None:
            failures.append("diagnostic_scene_claims_formal_capacity")
        if hasattr(scene_spec, "capacity_receipt_sha256"):
            failures.append("diagnostic_scene_exposes_capacity_receipt")
        if getattr(cfg, "action_ball_full_mdp_scene_capacity", None) != 2:
            failures.append("cfg_scene_capacity_differs")
        if (
            scene_spec.capacity_authority_kind
            != getattr(
                cfg,
                "action_ball_full_mdp_scene_capacity_authority_kind",
                None,
            )
        ):
            failures.append("diagnostic_scene_capacity_authority_differs")
        if getattr(cfg, "action_ball_full_mdp_capacity_receipt_sha256", None) != "":
            failures.append("cfg_claims_formal_capacity_receipt")
        if (
            getattr(cfg, "action_ball_full_mdp_scene_spec_sha256", None)
            != scene_spec.canonical_sha256
        ):
            failures.append("cfg_scene_spec_binding_differs")
    # Keep the configured mode distinct from the value retained by Motion at
    # construction.  A combined reason hid which exact source differed.
    if (
        getattr(racket_cfg, "action_ball_diagnostic_unauthorized", None)
        is not True
    ):
        failures.append(
            "racket_cfg.action_ball_diagnostic_unauthorized_is_not_true"
        )
    if (
        getattr(live_racket_cfg, "action_ball_diagnostic_unauthorized", None)
        is not True
    ):
        failures.append(
            "live_racket_cfg.action_ball_diagnostic_unauthorized_is_not_true"
        )
    if getattr(motion, "_canonical_diagnostic_unauthorized", None) is not True:
        failures.append(
            "motion_owner._canonical_diagnostic_unauthorized_is_not_true"
        )
    if (
        RUNTIME_INTEGRATED is not False
        or LAUNCH_AUTHORIZED is not False
        or DIAGNOSTIC_UNAUTHORIZED is not True
        or scene.RUNTIME_INTEGRATED is not False
        or scene.LAUNCH_AUTHORIZED is not False
        or getattr(cfg, "action_ball_full_mdp_runtime_construction_status", None)
        != "HOLD"
    ):
        failures.append("factory_scene_or_cfg_authorization_flags_differ")
    if (
        canary.CANARY_SAVE_CHECKPOINTS is not False
        or canary.DIAGNOSTIC_UNAUTHORIZED is not True
        or canary.FORMAL_PROFILE is not False
        or canary.FORMAL_LAUNCH_AUTHORIZED is not False
        or canary.PRODUCTION_INTEGRATED is not False
        or canary.RUNTIME_INTEGRATED is not False
        or canary.LAUNCH_AUTHORIZED is not False
    ):
        failures.append("canary_no_save_or_authorization_flags_differ")
    if getattr(cfg, "checkpoint_path", None) is not None:
        failures.append("checkpoint_resume_marker_present")
    if getattr(cfg, "checkpoint_tolerant", False) is not False:
        failures.append("checkpoint_tolerant_marker_present")
    if getattr(env, "full_mdp_cold_restore_dormant", False) is not False:
        failures.append("cold_restore_marker_present")
    if getattr(env, "_action_ball_r10_cold_restore_capsule", None) is not None:
        failures.append("cold_restore_capsule_present")
    if failures:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "fresh full-MDP single_action_lean preflight rejected before "
            "reset-genesis issue and before ObservationManager: "
            + ",".join(failures)
        )


def _construct_offside_seed(env: object) -> _OffsideConstructionSeed:
    """Construct the real genesis root without publishing a partial graph.

    This function intentionally consumes neither genesis projection.  The env
    and Device-R05 projections remain available to the eventual complete
    bundle, and a failure in any later stage leaves ``env`` byte-for-byte
    untouched by this factory.
    """

    try:
        import torch
        genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
        cadence = importlib.import_module("action_ball_motion_cadence_device")
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "independent reset-genesis producer is unavailable"
        ) from exc
    try:
        motion = env.command_manager.get_term("motion")
        racket = env.command_manager.get_term("racket_target")
        # IsaacLab configclass construction does not retain object identity
        # between the EnvCfg template and the CommandTerm.  The cold numeric
        # producers must consume the exact cfg retained by the live owner.
        racket_cfg = getattr(racket, "cfg")
        num_envs = getattr(env, "num_envs")
        device = torch.device(getattr(env, "device"))
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "factory seam lacks exact command identities or environment shape"
        ) from exc
    if type(num_envs) is not int or num_envs < 1:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "factory env num_envs must be a positive exact int"
        )
    # Motion parses this profile during CommandManager construction.  Check it
    # before issuing any one-shot capability; a known typed pending stage must
    # not allocate even cold authority state.
    if getattr(motion, "_action_ball_continuous_motion_profile", None) is None:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "fresh full-MDP production graph remains HOLD before "
            "ObservationManager; motion_cadence_pre_command_manager_absent: "
            "Motion was already constructed without the fresh cadence "
            "profile; no reset genesis or component identity was installed"
        )
    try:
        parent, parent_receipt, expected_profile = (
            cadence.build_action_ball_full_mdp_diagnostic_motion_profile()
        )
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "fresh full-MDP diagnostic Motion parent authority is unavailable"
        ) from exc
    if (
        type(parent) is not cadence.DiagnosticMotionParentScheduleAuthority
        or type(parent_receipt) is not cadence.DiagnosticMotionProfileReceipt
        or type(expected_profile) is not dict
        or getattr(motion, "_action_ball_continuous_motion_profile", None)
        != expected_profile
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "Motion retained cadence differs from the code-owned parent source"
        )
    issue = genesis.issue_action_ball_full_mdp_reset_genesis(
        num_envs=num_envs,
        device=device,
    )
    if (
        type(issue) is not genesis.ActionBallFullMdpResetGenesisIssue
        or type(issue.authority)
        is not genesis.ActionBallFullMdpResetGenesisAuthority
        or type(issue.receipt) is not genesis.ActionBallFullMdpResetGenesisReceipt
    ):
        if (
            type(getattr(issue, "authority", None))
            is genesis.ActionBallFullMdpResetGenesisAuthority
            and type(getattr(issue, "receipt", None))
            is genesis.ActionBallFullMdpResetGenesisReceipt
        ):
            genesis.discard_unpublished_action_ball_full_mdp_reset_genesis(
                authority=issue.authority,
                receipt=issue.receipt,
            )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "independent reset-genesis issuer returned a foreign capability"
        )
    return _OffsideConstructionSeed(
        motion_owner=motion,
        racket_owner=racket,
        racket_cfg=racket_cfg,
        motion_parent_authority=parent,
        motion_parent_receipt=parent_receipt,
        reset_genesis_authority=issue.authority,
        reset_genesis_receipt=issue.receipt,
        num_envs=num_envs,
        device=device,
    )


def _require_precommand_motion_cadence(seed: _OffsideConstructionSeed) -> None:
    """Cold-bind Motion to its exact parent before any command compute.

    Motion parses and allocates this profile inside CommandManager
    construction, but its first ``_update_command`` also requires the one
    code-owned parent schedule to be bound.  This call consumes that exact
    parent/receipt pair; a source digest or retained-profile comparison alone
    cannot replace the live binding.
    """

    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")

    def discard_genesis() -> None:
        genesis.discard_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )

    if getattr(seed.motion_owner, "_action_ball_continuous_motion_profile", None) is None:
        discard_genesis()
        raise ActionBallFullMdpRuntimeFactoryHold(
            "motion_cadence_pre_command_manager_absent: Motion was already "
            "constructed without the fresh cadence profile; reset genesis "
            "was discarded and no environment install occurred"
        )
    bind = getattr(
        seed.motion_parent_authority,
        "bind_exact_parent_schedule",
        None,
    )
    direct_bind = getattr(
        type(seed.motion_parent_authority),
        "bind_exact_parent_schedule",
        None,
    )
    if (
        not callable(bind)
        or not callable(direct_bind)
        or getattr(bind, "__self__", None) is not seed.motion_parent_authority
        or getattr(bind, "__func__", None) is not direct_bind
    ):
        discard_genesis()
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact Motion parent-schedule cold binder is absent"
        )
    try:
        bind(seed.motion_owner, seed.motion_parent_receipt)
    except BaseException as exc:
        discard_genesis()
        raise ActionBallFullMdpRuntimeFactoryHold(
            "Motion parent-schedule cold binding failed; unpublished reset "
            "genesis was discarded before downstream owner construction"
        ) from exc


def _construct_offside_device_profile(
    seed: _OffsideConstructionSeed,
) -> _OffsideDeviceProfileBundle:
    """Construct the real canary profile off-side or discard genesis."""

    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    try:
        canary = importlib.import_module(
            "action_ball_full_mdp_canary_target_profile"
        )
        profile = importlib.import_module("action_ball_device_profile_authority")
        if (
            profile.DIAGNOSTIC_UNAUTHORIZED is not True
            or profile.PRODUCTION_INTEGRATED is not False
            or profile.RUNTIME_INTEGRATED is not False
            or profile.LAUNCH_AUTHORIZED is not False
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "diagnostic profile authority authorization flags differ"
            )
        # The owner retains only its private record and opaque receipt.  This
        # constructor module intentionally exposes no profile registry, so an
        # off-side bundle that is dropped at HOLD cannot leave global state.
        if any("REGISTRY" in name for name in vars(profile)):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "diagnostic profile authority unexpectedly owns a global registry"
            )
        spec = canary.build_action_ball_full_mdp_canary_target_profile(
            racket_cfg=seed.racket_cfg
        )
        owner, receipt = profile.construct_device_profile_authority(
            spec,
            device=seed.device,
            expected_support_size=3,
        )
        if (
            type(spec) is not profile.FrozenDeviceTargetProfileSpec
            or type(owner) is not profile.DeviceProfileAuthorityOwner
            or type(receipt) is not profile.DeviceProfileReceipt
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "diagnostic canary profile constructor returned a foreign capability"
            )
    except Exception as exc:
        genesis.discard_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        if isinstance(exc, ActionBallFullMdpRuntimeFactoryHold):
            raise
        raise ActionBallFullMdpRuntimeFactoryHold(
            "diagnostic canary profile construction failed; unpublished reset "
            "genesis was discarded and no environment install occurred"
        ) from exc
    return _OffsideDeviceProfileBundle(
        seed=seed,
        profile_spec=spec,
        profile_owner=owner,
        profile_receipt=receipt,
    )


def _construct_offside_question_inputs(
    bundle: _OffsideDeviceProfileBundle,
) -> _OffsideQuestionInputs:
    """Retain only cold reusable roots; no first-shot one-shot authority."""

    return _OffsideQuestionInputs(profile=bundle)


def _construct_offside_physical_scene_inputs(
    question: _OffsideQuestionInputs,
) -> _OffsidePhysicalSceneInputs:
    """Bind the exact K=2 live scene without consuming reset genesis.

    The scene port reads the real K-body scene.  The factory does not fabricate
    an observe-all post-physics request: its absent-producer fault would be
    true by construction and it has no runtime consumer.

    Physical consumes the third exact clone-only device projection from the
    same genesis authority/receipt.  It never accepts a caller-authored host
    tuple and leaves the env/Device-R05 projections unconsumed for the later
    atomic graph.  Failure after entering the Physical constructor cannot be
    rolled back.  The code-owned genesis issuer instead retires the
    unpublished env/R05 projections and releases its retained record; it
    never claims that the already-issued Physical clone was undone.
    """

    seed = question.profile.seed
    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    try:
        import torch

        capacity = importlib.import_module(
            "action_ball_full_mdp_diagnostic_capacity"
        )
        scene = importlib.import_module(
            "whole_body_tracking.tasks.tracking.config.agibot_a3."
            "action_ball_full_mdp_ball_scene"
        )
        physical = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_physical_flight_device"
        )
        physical_question = importlib.import_module(
            "action_ball_physical_question_device"
        )
        live_scene = getattr(seed.motion_owner, "_env", None)
        live_scene = getattr(live_scene, "scene", None)
        if live_scene is None:
            raise ActionBallFullMdpRuntimeFactoryHold(
                "exact diagnostic K=2 live scene is unavailable from Motion"
            )
        env_origins = getattr(live_scene, "env_origins", None)
        scene_spec = getattr(
            getattr(seed.motion_owner, "_env", None),
            "cfg",
            None,
        )
        scene_spec = getattr(
            scene_spec, "action_ball_full_mdp_ball_scene_spec", None
        )
        binding = capacity.construct_diagnostic_n2_capacity_binding(scene_spec)
        port = scene.IsaacLabPhysicalFlightScenePort(
            scene=live_scene,
            spec=scene_spec,
            env_origins=env_origins,
        )
        state = port.read_state_env()
        if (
            type(binding) is not capacity.DiagnosticN2CapacityBinding
            or type(port) is not scene.IsaacLabPhysicalFlightScenePort
            or tuple(state.shape) != (seed.num_envs, 2, 13)
            or state.dtype != torch.float32
            or state.device != seed.device
            or hasattr(binding, "capacity_receipt_sha256")
            or getattr(
                getattr(seed.motion_owner, "_env", None).cfg,
                "action_ball_full_mdp_capacity_receipt_sha256",
                None,
            )
            != ""
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "diagnostic K=2 scene-port ABI differs"
            )
    except BaseException as exc:
        genesis.discard_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        if isinstance(exc, ActionBallFullMdpRuntimeFactoryHold):
            raise
        raise ActionBallFullMdpRuntimeFactoryHold(
            "diagnostic K=2 scene-port construction failed; unpublished "
            "reset genesis was discarded and no environment install occurred"
        ) from exc
    try:
        physical_owner = physical.ActionBallPhysicalFlightDeviceOwner(
            num_envs=seed.num_envs,
            scene_body_names=tuple(port.spec.scene_entity_names),
            scene_port=port,
            diagnostic_n2_capacity_binding=binding,
            reset_genesis_authority=seed.reset_genesis_authority,
            reset_genesis_receipt=seed.reset_genesis_receipt,
        )
        snapshot = physical_owner.scene_snapshot()
        physical_question_core = (
            physical_question.
            construct_diagnostic_n2_no_save_physical_question_numeric_core(
                physical_flight_owner=physical_owner,
                motion_owner=seed.motion_owner,
                racket_owner=seed.racket_owner,
            )
        )
        if (
            type(physical_owner)
            is not physical.ActionBallPhysicalFlightDeviceOwner
            or type(physical_question_core)
            is not physical_question.PhysicalQuestionNumericCore
            or type(snapshot) is not physical.PhysicalFlightSceneSnapshotNK
            or tuple(snapshot.state_env_f32.shape) != (seed.num_envs, 2, 13)
            or snapshot.state_env_f32.dtype != torch.float32
            or snapshot.state_env_f32.device != seed.device
            or getattr(physical_owner, "_genesis_world_reset_identity", None)
            is None
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "diagnostic Physical owner/scene snapshot ABI differs"
            )
    except BaseException as exc:
        # Physical may already have consumed its one-shot projection.  Retire
        # the remaining unpublished consumers without claiming rollback.
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "diagnostic Physical construction failed after entering its "
            "one-shot genesis boundary; remaining unpublished genesis "
            "consumers were retired, no projected-genesis rollback was "
            "claimed, and no environment install occurred"
        ) from exc
    return _OffsidePhysicalSceneInputs(
        question=question,
        scene_port=port,
        diagnostic_capacity_binding=binding,
        physical_owner=physical_owner,
        physical_question_core=physical_question_core,
    )


def _construct_offside_recurring_question_bundle(
    physical: _OffsidePhysicalSceneInputs,
) -> _OffsideRecurringQuestionInputs:
    """Build the reusable cold table after Physical exists, without env install."""

    seed = physical.question.profile.seed
    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    try:
        question_owner = importlib.import_module(
            "action_ball_full_mdp_canary_question_owner"
        )
        recurring = question_owner.construct_recurring_d05_internal_question_bundle(
            profile_owner=physical.question.profile.profile_owner,
            profile_receipt=physical.question.profile.profile_receipt,
            racket_owner=seed.racket_owner,
            physical_owner=physical.physical_question_core,
        )
        if type(recurring) is not question_owner.RecurringD05InternalQuestionBundle:
            raise ActionBallFullMdpRuntimeFactoryHold(
                "diagnostic recurring question bundle exact type differs"
            )
    except BaseException as exc:
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "recurring D05 question-bundle construction failed after Physical "
            "consumed its projection; remaining unpublished genesis consumers "
            "were retired and no environment install occurred"
        ) from exc
    return _OffsideRecurringQuestionInputs(
        physical=physical,
        recurring_question_bundle=recurring,
    )


def _construct_offside_r03_r06(
    env: object,
    recurring: _OffsideRecurringQuestionInputs,
) -> _OffsideColdLeafInputs:
    """Take the live Racket child and construct R06 from the same Physical."""

    physical = recurring.physical
    seed = physical.question.profile.seed
    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    try:
        strike = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_strike_fact_device"
        )
        landing = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_landing_outcome_device"
        )
        getter = getattr(seed.racket_owner, "action_ball_full_mdp_r03_owner")
        direct_getter = getattr(
            type(seed.racket_owner), "action_ball_full_mdp_r03_owner", None
        )
        r03_owner = getter()
        if (
            not callable(getter)
            or not callable(direct_getter)
            or getattr(getter, "__self__", None) is not seed.racket_owner
            or getattr(getter, "__func__", None) is not direct_getter
            or type(r03_owner) is not strike.ActionBallStrikeFactDeviceCoordinator
            or r03_owner
            is not getattr(
                seed.racket_owner,
                "_action_ball_strike_fact_device_coordinator",
                None,
            )
            or getattr(r03_owner, "num_envs", None) != seed.num_envs
            or getattr(r03_owner, "device", None) != seed.device
            or getattr(r03_owner, "_observation_projection_mode", None)
            != strike.OBSERVATION_PROJECTION_MODE_FRESH_FULL_MDP
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "live Racket-owned exact fresh R03 identity differs"
            )
        r06_owner = landing.construct_diagnostic_n2_no_save_r06(
            env=env,
            physical_owner=physical.physical_owner,
            diagnostic_n2_capacity_binding=(
                physical.diagnostic_capacity_binding
            ),
        )
        if (
            type(r06_owner)
            is not landing.ActionBallLandingOutcomeDeviceCoordinator
            or getattr(r06_owner, "diagnostic_n2_no_save", None) is not True
            or getattr(r06_owner, "diagnostic_unauthorized", None) is not True
            or getattr(r06_owner, "num_envs", None) != seed.num_envs
            or getattr(r06_owner, "device", None) != seed.device
            or getattr(r06_owner, "flight_slot_capacity", None) != 2
            or getattr(r06_owner, "mailbox_capacity", None) != 2
            or hasattr(r06_owner, "runtime_binding")
            or hasattr(r06_owner, "payment_authority")
            or hasattr(r06_owner, "capacity_authority")
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "real diagnostic R06 owner ABI differs"
            )
    except BaseException as exc:
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        if isinstance(exc, ActionBallFullMdpRuntimeFactoryHold):
            raise
        raise ActionBallFullMdpRuntimeFactoryHold(
            "real cold Racket-owned R03 / diagnostic R06 construction failed "
            "after Physical consumed its projection; remaining unpublished "
            "genesis consumers were retired and no environment install occurred"
        ) from exc
    return _OffsideColdLeafInputs(
        recurring=recurring,
        r03_owner=r03_owner,
        r06_owner=r06_owner,
    )


def _construct_offside_r07(
    env: object,
    inputs: _OffsideColdLeafInputs,
    epoch_owner: object,
) -> tuple[object, object]:
    """Construct the exact live-fact R07 bundle without publishing to env."""

    seed = inputs.recurring.physical.question.profile.seed
    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    recovery = importlib.import_module(
        "action_ball_continuous_recovery_device"
    )
    try:
        bundle = recovery.construct_action_ball_full_mdp_diagnostic_n2_recovery_owner(
            env=env,
            motion_owner=seed.motion_owner,
            action_epoch_owner=epoch_owner,
            motion_parent_authority=seed.motion_parent_authority,
            motion_parent_receipt=seed.motion_parent_receipt,
        )
    except recovery.ContinuousRecoveryConstructionHold as exc:
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact R07 diagnostic constructor remains HOLD; remaining "
            "unpublished genesis consumers were retired: " + str(exc)
        ) from exc
    except BaseException as exc:
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact R07 diagnostic constructor failed outside its reviewed HOLD; "
            "remaining unpublished genesis consumers were retired"
        ) from exc
    if (
        type(bundle) is not recovery.DiagnosticN2ContinuousRecoveryBundle
        or type(bundle.owner) is not recovery.ContinuousRecoveryDeviceCoordinator
        or type(bundle.plant_fact_adapter)
        is not recovery.DiagnosticN2ContinuousRecoveryPlantFactAdapter
        or getattr(bundle.plant_fact_adapter, "_env", None) is not env
        or getattr(bundle.plant_fact_adapter, "_motion_owner", None)
        is not seed.motion_owner
    ):
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact R07 diagnostic bundle identity differs; remaining genesis "
            "consumers were retired"
        )
    motion_bind = getattr(
        seed.motion_owner,
        "bind_action_ball_continuous_r07_ready_projection",
        None,
    )
    direct_motion_bind = getattr(
        type(seed.motion_owner),
        "bind_action_ball_continuous_r07_ready_projection",
        None,
    )
    ready_validator = getattr(
        bundle,
        "require_owned_motion_ready_projection",
        None,
    )
    direct_ready_validator = getattr(
        type(bundle),
        "require_owned_motion_ready_projection",
        None,
    )
    if (
        not callable(motion_bind)
        or not callable(direct_motion_bind)
        or getattr(motion_bind, "__self__", None) is not seed.motion_owner
        or getattr(motion_bind, "__func__", None) is not direct_motion_bind
        or not callable(ready_validator)
        or not callable(direct_ready_validator)
        or getattr(ready_validator, "__self__", None) is not bundle
        or getattr(ready_validator, "__func__", None)
        is not direct_ready_validator
    ):
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact Motion/R07 readiness cold binding API differs; remaining "
            "genesis consumers were retired"
        )
    try:
        motion_bind(
            bundle,
            require_owned_ready_projection=ready_validator,
        )
    except BaseException as exc:
        genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
            authority=seed.reset_genesis_authority,
            receipt=seed.reset_genesis_receipt,
        )
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact Motion/R07 readiness cold binding failed; remaining genesis "
            "consumers were retired"
        ) from exc
    return bundle, bundle.plant_fact_adapter


def _construct_offside_lean_runtime(
    env: object,
    inputs: _OffsideColdLeafInputs,
) -> _OffsideLeanRuntimeInputs:
    """Build one [N,1] epoch and bind exact live writers before publication."""

    seed = inputs.recurring.physical.question.profile.seed
    genesis = importlib.import_module("action_ball_full_mdp_reset_genesis")
    try:
        epoch_module = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_epoch"
        )
        cadence = importlib.import_module("action_ball_motion_cadence_device")
        device_r05 = importlib.import_module(
            "action_ball_continuous_runtime_transaction_device"
        )
        lean = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_runtime"
        )
        lean_rewards = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_rewards"
        )
        lean_observations = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp."
            "action_ball_full_mdp_lean_observation_cfg"
        )
        terminations = importlib.import_module(
            "whole_body_tracking.tasks.tracking.mdp.terminations"
        )
        epoch_genesis = (
            seed.reset_genesis_authority.
            require_owned_action_epoch_genesis(
                seed.reset_genesis_receipt,
                device=seed.device,
                num_envs=seed.num_envs,
            )
        )
        if (
            type(epoch_genesis)
            is not genesis.ActionBallFullMdpActionEpochGenesisProjection
            or epoch_genesis.world_reset_identity is not getattr(
                inputs.recurring.physical.physical_owner,
                "_genesis_world_reset_identity",
                None,
            )
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "ActionEpoch genesis projection or world identity differs"
            )
        epoch_owner = epoch_module.ActionEpochOwner(
            num_envs=seed.num_envs,
            device=seed.device,
            shot_slot_capacity=1,
            initial_reset_generation=epoch_genesis.reset_generations,
        )
        # This is the canonical first whole-environment reset already minted
        # by the independent genesis authority.  Adopt it once without
        # manufacturing a second reset generation or an empty dummy epoch.
        epoch_owner.activate_reset_genesis(
            selected_mask=torch.ones(
                (seed.num_envs,),
                dtype=torch.bool,
                device=seed.device,
            ),
            reset_generation=epoch_genesis.reset_generations,
        )
        cadence_owner = cadence.construct_production_motion_cadence_authority(
            motion_owner=seed.motion_owner
        )
        inputs.r03_owner.bind_action_epoch_owner(epoch_owner)
        inputs.r03_owner.bind_action_epoch_racket_owner(seed.racket_owner)
        d05_owner = (
            device_r05.construct_action_ball_full_mdp_device_r05(
                inputs.recurring.physical.question.profile.profile_owner,
                inputs.recurring.physical.question.profile.profile_receipt,
                seed=20260804,
                genesis_authority=seed.reset_genesis_authority,
                genesis_receipt=seed.reset_genesis_receipt,
                cadence_authority=cadence_owner,
                question_authority=inputs.recurring.recurring_question_bundle,
                epoch_owner=epoch_owner,
                motion_owner=seed.motion_owner,
                racket_owner=seed.racket_owner,
                physical_owner=inputs.recurring.physical.physical_owner,
                journal_capacity=64,
                max_reveal_epochs_per_drain=64,
            )
        )
        # Every selected-reset leaf consumes its exact Device-R05 genesis
        # while the construction window is still open.  No source digest or
        # same-writer receipt substitutes for these bound method identities.
        inputs.recurring.physical.physical_owner.bind_device_r05_reset_owner(
            d05_owner,
            prepared_reset_validator=(
                d05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                d05_owner.require_owned_true_reset_receipt
            ),
        )
        # R06 keeps formal reset authority fail-closed.  This disposable
        # diagnostic graph instead binds the exact D05/Physical/Epoch/world-
        # reset construction through its dedicated non-formal seam.
        inputs.r06_owner.bind_diagnostic_n2_device_r05_reset_owner(
            d05_owner,
            prepared_reset_validator=(
                d05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                d05_owner.require_owned_true_reset_receipt
            ),
        )
        inputs.recurring.physical.physical_owner.bind_r06_owner(
            inputs.r06_owner
        )
        seed.motion_owner.bind_action_ball_continuous_motion_selected_reset(
            d05_owner,
            prepared_reset_validator=(
                d05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                d05_owner.require_owned_true_reset_receipt
            ),
            diagnostic=False,
        )
        seed.racket_owner.bind_action_ball_continuous_racket_selected_reset(
            d05_owner,
            prepared_reset_validator=(
                d05_owner.require_owned_prepared_true_reset
            ),
            r05_receipt_validator=(
                d05_owner.require_owned_true_reset_receipt
            ),
            diagnostic=False,
        )
        r06_binder = getattr(
            inputs.r06_owner,
            "bind_action_ball_full_mdp_epoch_owner",
            None,
        )
        direct_r06_binder = getattr(
            type(inputs.r06_owner),
            "bind_action_ball_full_mdp_epoch_owner",
            None,
        )
        if (
            not callable(r06_binder)
            or getattr(r06_binder, "__self__", None) is not inputs.r06_owner
            or getattr(r06_binder, "__func__", None) is not direct_r06_binder
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "exact R06 ActionEpoch cold binder is absent"
        )
        r06_binder(epoch_owner)
        r07_owner, r07_adapter = _construct_offside_r07(
            env, inputs, epoch_owner
        )
        reward_bundle = (
            lean_rewards.materialize_diagnostic_n2_reward_manager_cfg(
                epoch_owner=epoch_owner
            )
        )
        # ``profile_kind`` and ``diagnostic_unauthorized`` remain producer
        # telemetry, not a same-writer admission gate.  The Lean owner below
        # independently joins graph -> ActionEpoch identity; the environment
        # installation seam validates the manager cfg and component registry.
        runtime_owner = lean.ActionBallFullMdpLeanRuntimeOwner(
            env=env,
            runtime_lease=env.action_ball_full_mdp_construction_lease(),
            epoch_owner=epoch_owner,
            r05_runtime=d05_owner,
            motion=seed.motion_owner,
            racket=seed.racket_owner,
            physical_ball=inputs.recurring.physical.physical_owner,
            r06_landing_outcome=inputs.r06_owner,
            r03_strike_fact=inputs.r03_owner,
            r07_recovery=r07_owner,
            reward_graph=reward_bundle.graph,
        )
        epoch_owner.bind_selected_reset_owner(runtime_owner)
        # This is the final Device-R05 construction seam.  It closes only
        # after all four leaf validators and the ActionEpoch reset writer are
        # exact-bound to this one coordinator.
        d05_owner.bind_true_reset_authority(runtime_owner)
        observation_bundle = (
            lean_observations.materialize_observation_manager_cfg(
                env=env,
                runtime_owner=runtime_owner,
            )
        )
        observation_source = observation_bundle.source
        observation_manager_cfg = observation_bundle.manager_cfg
        termination_manager_cfg = (
            terminations.
            materialize_action_ball_full_mdp_lean_termination_manager_cfg(
                env.cfg.terminations
            )
        )
        if (
            type(epoch_owner) is not epoch_module.ActionEpochOwner
            or type(d05_owner) is not device_r05.DeviceR05Owner
            or type(runtime_owner) is not lean.ActionBallFullMdpLeanRuntimeOwner
            or runtime_owner.epoch_owner is not epoch_owner
            or type(observation_bundle)
            is not lean_observations.DiagnosticN2ObservationManagerBundle
            or type(observation_source)
            is not lean_observations.LeanActionEpochObservationSource
            or getattr(observation_source, "_env", None) is not env
            or getattr(observation_source, "_runtime_owner", None)
            is not runtime_owner
            or type(observation_manager_cfg) is not dict
            or tuple(observation_manager_cfg) != ("policy", "critic")
            or type(termination_manager_cfg) is not dict
            or tuple(termination_manager_cfg)
            != terminations.ACTION_BALL_FULL_MDP_LEAN_TERMINATION_MANAGER_ORDER
        ):
            raise ActionBallFullMdpRuntimeFactoryHold(
                "lean ActionEpoch runtime exact identities differ"
            )
    except BaseException as exc:
        # D05 may already have consumed the final genesis projection.  Never
        # describe this discarded graph as rolled back or safe to retry.
        try:
            genesis.retire_failed_unpublished_action_ball_full_mdp_reset_genesis(
                authority=seed.reset_genesis_authority,
                receipt=seed.reset_genesis_receipt,
            )
        except Exception:
            pass
        if isinstance(exc, ActionBallFullMdpRuntimeFactoryHold):
            raise
        raise ActionBallFullMdpRuntimeFactoryHold(
            "lean runtime construction failed after cold binding began; the "
            "graph is discarded and must not be retried"
        ) from exc
    return _OffsideLeanRuntimeInputs(
        cold=inputs,
        r07_owner=r07_owner,
        r07_plant_fact_adapter=r07_adapter,
        epoch_owner=epoch_owner,
        device_r05_owner=d05_owner,
        reward_graph=reward_bundle.graph,
        reward_manager_cfg=reward_bundle.manager_cfg,
        observation_source=observation_source,
        observation_manager_cfg=observation_manager_cfg,
        termination_manager_cfg=termination_manager_cfg,
        lean_runtime_owner=runtime_owner,
    )


def _install_lean_runtime_graph(env: object, graph: _OffsideLeanRuntimeInputs) -> None:
    """Install the live fact producer, then publish one complete env graph.

    The PhysX subscriber is deliberately the last off-side construction step:
    it is an engine side effect that cannot be honestly rolled back.  All
    R03/R06/R07/Reward/Observation/Termination owners already exist before it
    is installed.  A later env-publication failure discards the whole env
    fail-stop; it is never reported as a subscriber rollback.
    """

    env_module = importlib.import_module(
        "whole_body_tracking.tasks.tracking.full_mdp_env"
    )
    cold = graph.cold
    physical = cold.recurring.physical
    seed = physical.question.profile.seed
    components = env_module.FullMdpLeanRuntimeComponents(
        epoch_owner=graph.epoch_owner,
        device_r05_owner=graph.device_r05_owner,
        motion_owner=seed.motion_owner,
        racket_owner=seed.racket_owner,
        physical_owner=physical.physical_owner,
        r03_owner=cold.r03_owner,
        r06_owner=cold.r06_owner,
        r07_owner=graph.r07_owner,
        r07_plant_fact_adapter=graph.r07_plant_fact_adapter,
        reward_graph=graph.reward_graph,
        lean_runtime_owner=graph.lean_runtime_owner,
        observation_source=graph.observation_source,
    )
    scene_port = physical.scene_port
    install_live_physx = getattr(
        scene_port,
        "install_action_epoch_live_physx_fact_owner",
        None,
    )
    direct_install_live_physx = getattr(
        type(scene_port),
        "install_action_epoch_live_physx_fact_owner",
        None,
    )
    if (
        not callable(install_live_physx)
        or not callable(direct_install_live_physx)
        or getattr(install_live_physx, "__self__", None) is not scene_port
        or getattr(install_live_physx, "__func__", None)
        is not direct_install_live_physx
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact live PhysX ActionEpoch installer is absent"
        )
    shutdown_live_physx = getattr(
        scene_port,
        "shutdown_action_epoch_live_physx_fact_owner",
        None,
    )
    direct_shutdown_live_physx = getattr(
        type(scene_port),
        "shutdown_action_epoch_live_physx_fact_owner",
        None,
    )
    if (
        not callable(shutdown_live_physx)
        or not callable(direct_shutdown_live_physx)
        or getattr(shutdown_live_physx, "__self__", None) is not scene_port
        or getattr(shutdown_live_physx, "__func__", None)
        is not direct_shutdown_live_physx
    ):
        raise ActionBallFullMdpRuntimeFactoryHold(
            "exact live PhysX ActionEpoch shutdown is absent"
        )
    try:
        live_stage = importlib.import_module("omni.usd").get_context().get_stage()
    except Exception as exc:
        raise ActionBallFullMdpRuntimeFactoryHold(
            "live USD stage is unavailable for the PhysX fact producer"
        ) from exc
    # Physical's exact binder already joined the Racket selected-rubber
    # producer and the scene writer.  No caller paths, venue numerics, digest
    # or verdict can stand in for the engine subscriber installed here.
    try:
        install_live_physx(stage=live_stage)
        env.install_action_ball_full_mdp_lean_runtime_graph(
            env.action_ball_full_mdp_construction_lease(),
            genesis_authority=seed.reset_genesis_authority,
            genesis_receipt=seed.reset_genesis_receipt,
            components=components,
            reward_manager_cfg=graph.reward_manager_cfg,
            observation_source=graph.observation_source,
            observation_manager_cfg=graph.observation_manager_cfg,
            termination_manager_cfg=graph.termination_manager_cfg,
            live_physx_shutdown=shutdown_live_physx,
        )
    except BaseException as install_error:
        # This is a real resource release, not a rollback of already-consumed
        # one-shot owners.  Preserve the construction/publication failure as
        # primary while still exposing an unacknowledged unsubscribe as cause.
        try:
            shutdown_live_physx()
        except BaseException as shutdown_error:
            raise install_error from shutdown_error
        raise


def materialize_action_ball_full_mdp_reward_manager(
    env: object,
    *,
    numeric_authority_owner: object,
    numeric_authority_receipt: object,
    reward_graph: object,
) -> ActionBallFullMdpInstalledRewardGraphReceipt:
    """Consume one authority and atomically install the shared Reward graph.

    This does not construct any of the four absent production producers and it
    never converts a diagnostic authority or graph into runtime/launch
    authority.  It is the narrow transaction used once those owners exist.
    Failure after one-shot consumption deliberately burns the authority while
    rolling back every environment/config mutation.
    """

    _require_unique_manager_seam(env)
    budget, rewards, config, reward_term_type = _reward_dependencies()
    if tuple(budget.NUMERIC_PRODUCTION_HOLD_REASONS) != (
        NUMERIC_REWARD_FACTORY_HOLD_REASONS
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "numeric production HOLD reason set differs"
        )
    if type(reward_graph) is not rewards.FreshFullMdpRewardGraph:
        raise ActionBallFullMdpRewardMaterializationHold(
            "fourteen-lifecycle-consumer graph exact class/source identity differs"
        )
    if reward_graph.active_cycle is not None:
        raise ActionBallFullMdpRewardMaterializationHold(
            "fourteen-lifecycle-consumer graph has an open Reward cycle"
        )
    cfg = getattr(env, "cfg", None)
    template = getattr(cfg, "rewards", None)
    blockers = tuple(config.action_ball_full_mdp_reward_template_blockers(template))
    if blockers:
        raise ActionBallFullMdpRewardMaterializationHold(
            "dedicated shared Reward template differs: " + ",".join(blockers)
        )
    if hasattr(env, REWARD_GRAPH_ATTR) or hasattr(env, INSTALLED_REWARD_RECEIPT_ATTR):
        raise ActionBallFullMdpRewardMaterializationHold(
            "Reward graph or installed-graph receipt was already installed"
        )
    templates = tuple(config.ACTION_BALL_FULL_MDP_REWARD_TERM_TEMPLATES)
    if (
        tuple(term.manager_name for term in templates)
        != reward_contract.MANAGER_NAMES
        or tuple(
            term.payment_consumer
            for term in templates[: reward_contract.LIFECYCLE_PAYMENT_COUNT]
        )
        != tuple(rewards.ORDERED_CONSUMERS)
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "dedicated Reward template does not bind the exact lifecycle fourteen"
        )

    payload, authority_sha, materialization_sha, resolved_graph_sha = (
        _consume_numeric_authority(
            budget=budget,
            numeric_authority_owner=numeric_authority_owner,
            numeric_authority_receipt=numeric_authority_receipt,
        )
    )
    selected = _mapping(
        payload.get("selected_numeric_parameters"),
        label="selected_numeric_parameters",
    )
    treatment = _mapping(selected.get("treatment_gain"), label="treatment_gain")
    gain_a, _host_a = _fraction(treatment.get("a"), label="treatment_gain.a")
    gain_c, _host_c = _fraction(
        treatment.get("c"), label="treatment_gain.c", positive=False
    )
    if gain_a != Fraction(1, 1) or gain_c != Fraction(0, 1):
        raise ActionBallFullMdpRewardMaterializationHold(
            "numeric placement treatment must remain exactly A=1,C=0"
        )
    _require_owner_numeric_profiles(reward_graph, selected)

    materialized: dict[str, object] = {}
    for index, term_template in enumerate(templates):
        params = (
            {"graph_attr": REWARD_GRAPH_ATTR}
            if index < reward_contract.LIFECYCLE_PAYMENT_COUNT
            else {}
        )
        if term_template.weight_source == config.ACTION_BALL_FULL_MDP_WEIGHT_SOURCE:
            if term_template.manager_weight is not None:
                raise ActionBallFullMdpRewardMaterializationHold(
                    f"numeric Reward template carries a shell weight: {term_template.manager_name}"
                )
            _exact_weight, manager_weight = _fraction(
                _path(payload, term_template.manager_weight_path),
                label=term_template.manager_weight_path,
            )
        elif (
            index == reward_contract.LIFECYCLE_PAYMENT_COUNT - 1
            and term_template.weight_source
            == config.ACTION_BALL_FULL_MDP_FIXED_WEIGHT_SOURCE
            and term_template.manager_weight == 1.0
        ):
            manager_weight = 1.0
        elif (
            index >= reward_contract.LIFECYCLE_PAYMENT_COUNT
            and term_template.weight_source
            in (
                config.ACTION_BALL_FULL_MDP_COMMON_DENSE_WEIGHT_SOURCE,
                config.ACTION_BALL_FULL_MDP_PADDLE_MOTION_PRIOR_WEIGHT_SOURCE,
                config.ACTION_BALL_FULL_MDP_REGULARIZATION_WEIGHT_SOURCE,
            )
            and term_template.manager_weight is not None
        ):
            manager_weight = _positive_host_number(
                term_template.manager_weight,
                label=term_template.manager_name + " fixed weight",
            )
        else:
            raise ActionBallFullMdpRewardMaterializationHold(
                f"Reward template weight source differs: {term_template.manager_name}"
            )
        if term_template.scale_source is not None:
            _exact_scale, params["std"] = _fraction(
                _path(payload, term_template.scale_source),
                label=term_template.scale_source,
            )
        for name, fixed_value in term_template.fixed_func_params:
            if name in params:
                raise ActionBallFullMdpRewardMaterializationHold(
                    f"duplicate fixed Reward param: {term_template.manager_name}.{name}"
                )
            params[name] = fixed_value
        materialized[term_template.manager_name] = reward_term_type(
            func=term_template.func,
            weight=manager_weight,
            params=params,
        )

    rows = _manager_cfg_rows(
        materialized,
        templates=templates,
        reward_term_type=reward_term_type,
        graph_attr=REWARD_GRAPH_ATTR,
    )
    installed_manager_graph_sha = _canonical_sha256(
        {
            "schema_version": 2,
            "kind": INSTALLED_REWARD_GRAPH_KIND,
            "numeric_authority_sha256": authority_sha,
            "resolved_graph_receipt_sha256": resolved_graph_sha,
            "ordered_terms": rows,
        }
    )
    diagnostic = bool(
        payload.get("diagnostic_unauthorized")
        or reward_graph.diagnostic_unauthorized
    )
    view = ActionBallFullMdpInstalledRewardGraphView(
        schema_version=2,
        kind=INSTALLED_REWARD_GRAPH_KIND,
        numeric_authority_sha256=authority_sha,
        numeric_materialization_sha256=materialization_sha,
        resolved_graph_receipt_sha256=resolved_graph_sha,
        installed_manager_graph_sha256=installed_manager_graph_sha,
        ordered_manager_names=tuple(materialized),
        ordered_payment_consumers=tuple(
            term.payment_consumer for term in templates
        ),
        diagnostic_unauthorized=diagnostic,
        runtime_integrated=False,
        launch_authorized=False,
    )
    record = _InstalledRewardGraphRecord(
        installation_identity=object(),
        env_identity=env,
        graph=reward_graph,
        manager_cfg=materialized,
        numeric_authority_owner=numeric_authority_owner,
        numeric_authority_receipt=numeric_authority_receipt,
        view=view,
    )
    installed_receipt = _mint_installed_reward_graph_receipt(record)

    old_template = template
    try:
        setattr(env, REWARD_GRAPH_ATTR, reward_graph)
        setattr(env, INSTALLED_REWARD_RECEIPT_ATTR, installed_receipt)
        cfg.rewards = materialized
    except Exception as exc:
        # Roll back unconditionally.  A hostile descriptor may mutate state
        # and then raise, so a local "assignment completed" boolean is not a
        # sufficient transaction witness.
        try:
            cfg.rewards = old_template
        except Exception:
            pass
        try:
            delattr(env, INSTALLED_REWARD_RECEIPT_ATTR)
        except Exception:
            pass
        try:
            delattr(env, REWARD_GRAPH_ATTR)
        except Exception:
            pass
        with _INSTALLED_REWARD_REGISTRY_LOCK:
            _INSTALLED_REWARD_REGISTRY.pop(installed_receipt, None)
        raise ActionBallFullMdpRewardMaterializationHold(
            "RewardManager transaction failed and was rolled back"
        ) from exc
    return installed_receipt


def require_owned_action_ball_full_mdp_installed_reward_graph(
    receipt: object,
    *,
    env: object,
    reward_graph: object,
) -> ActionBallFullMdpInstalledRewardGraphView:
    """Reverse-derive and authenticate the installed shared Reward graph."""

    if type(receipt) is not ActionBallFullMdpInstalledRewardGraphReceipt:
        raise ActionBallFullMdpRewardMaterializationHold(
            "installed Reward graph receipt must be exact opaque factory identity"
        )
    with _INSTALLED_REWARD_REGISTRY_LOCK:
        record = _INSTALLED_REWARD_REGISTRY.get(receipt)
    if (
        record is None
        or record.env_identity is not env
        or record.graph is not reward_graph
        or getattr(env, REWARD_GRAPH_ATTR, None) is not reward_graph
        or getattr(env, INSTALLED_REWARD_RECEIPT_ATTR, None) is not receipt
        or getattr(getattr(env, "cfg", None), "rewards", None)
        is not record.manager_cfg
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "installed Reward graph receipt is foreign or no longer installed"
        )
    _budget, rewards, config, reward_term_type = _reward_dependencies()
    templates = tuple(config.ACTION_BALL_FULL_MDP_REWARD_TERM_TEMPLATES)
    if (
        tuple(term.manager_name for term in templates)
        != reward_contract.MANAGER_NAMES
        or tuple(
            term.payment_consumer
            for term in templates[: reward_contract.LIFECYCLE_PAYMENT_COUNT]
        )
        != tuple(rewards.ORDERED_CONSUMERS)
    ):
        raise ActionBallFullMdpRewardMaterializationHold(
            "installed Reward template payment order drifted"
        )
    rows = _manager_cfg_rows(
        record.manager_cfg,
        templates=templates,
        reward_term_type=reward_term_type,
        graph_attr=REWARD_GRAPH_ATTR,
    )
    actual_sha = _canonical_sha256(
        {
            "schema_version": 2,
            "kind": INSTALLED_REWARD_GRAPH_KIND,
            "numeric_authority_sha256": record.view.numeric_authority_sha256,
            "resolved_graph_receipt_sha256": (
                record.view.resolved_graph_receipt_sha256
            ),
            "ordered_terms": rows,
        }
    )
    if actual_sha != record.view.installed_manager_graph_sha256:
        raise ActionBallFullMdpRewardMaterializationHold(
            "installed Reward graph differs from its reverse-derived receipt"
        )
    return record.view


def construct_action_ball_full_mdp_runtime_graph(env: object) -> None:
    """Construct and install one production graph, or fail before observation.

    One independent genesis feeds the ActionEpoch/Physical/D05 graph.  Every
    manager config and the exact top owner is constructed off-side; then the
    live scene subscriber is installed and the environment publishes genesis,
    components, all three manager configs and its teardown boundary in one
    mutation.  Failure drains the subscriber and discards the environment; no
    receipt, digest or caller verdict can substitute for a missing producer.
    """

    _require_unique_manager_seam(env)
    _require_single_action_lean_factory_env(env)
    seed = _construct_offside_seed(env)
    _require_precommand_motion_cadence(seed)
    bundle = _construct_offside_device_profile(seed)
    question_inputs = _construct_offside_question_inputs(bundle)
    physical_scene_inputs = _construct_offside_physical_scene_inputs(
        question_inputs
    )
    recurring_inputs = _construct_offside_recurring_question_bundle(
        physical_scene_inputs
    )
    cold_leaf_inputs = _construct_offside_r03_r06(env, recurring_inputs)
    lean_graph = _construct_offside_lean_runtime(env, cold_leaf_inputs)
    _install_lean_runtime_graph(env, lean_graph)


__all__ = [
    "ActionBallFullMdpInstalledRewardGraphReceipt",
    "ActionBallFullMdpInstalledRewardGraphView",
    "ActionBallFullMdpRewardMaterializationHold",
    "ActionBallFullMdpRuntimeFactoryHold",
    "DIAGNOSTIC_UNAUTHORIZED",
    "FIRST_UNRESOLVED_PRODUCTION_NODE",
    "FULL_MDP_RUNTIME_FACTORY_DAG",
    "INSTALLED_REWARD_GRAPH_KIND",
    "LAUNCH_AUTHORIZED",
    "NUMERIC_REWARD_FACTORY_HOLD_REASONS",
    "RUNTIME_INTEGRATED",
    "UNRESOLVED_PRODUCTION_NODES",
    "construct_action_ball_full_mdp_runtime_graph",
    "materialize_action_ball_full_mdp_reward_manager",
    "require_owned_action_ball_full_mdp_installed_reward_graph",
]
