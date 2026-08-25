"""Fresh-only IsaacLab environment with one top ActionBall runtime owner.

IsaacLab 2.3.2 does not expose a callback between ``scene.update`` and the
manager-level termination/reward pass.  Its physics callback is invoked before
the physics step and therefore cannot publish same-transition contact or
landing facts.  This module deliberately leaves the default
``ManagerBasedRLEnv`` untouched and provides one opt-in, executable-bound subclass
for the fresh full-MDP lineage.

The copied ``step`` body is reviewed against the exact build_2 IsaacLab source.
Construction still fails if the upstream file bytes, upstream
``ManagerBasedRLEnv.step`` AST, or its required call order changes.  This local
module does not hash itself: after module execution it cold-binds the exact
exported environment class and every direct executable method.  The original
constructor's cold preflight rejects a replaced module export, class, method or
instance override instead of treating a newly recomputed self-digest as
provenance.  This is a trusted-process accidental-mutation check, not launch
authorization or a boundary against arbitrary code with module-write access.

The lean owner is bound once to the exact class exported by its loaded module,
the environment-minted lease and the construction-owned component graph.
Runtime dispatch retains those bound methods and does not repeatedly scan
Python module/class dictionaries.

The installed top owner has the only fresh callpoints: one policy-step entry
before action processing (the owner, not this environment, decides whether a
reveal is due), one publication after every ``scene.update``, and one selected
true-reset entry before native manager bookkeeping.  The environment never
authors reveal facts and never calls a Motion, Racket, R05, R06 or physical
leaf directly.  The seams themselves never call ``item``, ``cpu``, ``numpy``,
``tolist`` or a random API.  Owner failure or host-visible
clock/identity/version drift poisons the environment.  Physics and
owner-internal writes cannot be rolled back honestly, so failure atomicity is
fail-stop rather than a fabricated rollback.  This module exposes no second
owner-construction lane.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import types
from typing import Callable, NamedTuple

import torch

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from whole_body_tracking.tasks.tracking.mdp import (
    action_ball_full_mdp_reward_contract as reward_contract,
)


PINNED_ISAACLAB_RELEASE = "v2.3.2"
PINNED_ISAACLAB_COMMIT = "8320e0be5c0f2def58d5b19d308c6d2539d47cb2"
PINNED_MANAGER_BASED_RL_ENV_FILE_SHA256 = (
    "974977825dbbd916460b4f9bb12176ab2cde41026fadc05e1de02d8fbbc94725"
)
PINNED_MANAGER_BASED_ENV_FILE_SHA256 = (
    "74d495925e264521b05a92296bbaf98adcabb7d94f3787ab3ed61f4d0680a198"
)
FULL_MDP_RUNTIME_FACTORY_MODULE = (
    "whole_body_tracking.tasks.tracking.mdp."
    "action_ball_full_mdp_runtime_factory"
)
FULL_MDP_RUNTIME_FACTORY_QUALNAME = (
    "construct_action_ball_full_mdp_runtime_graph"
)
FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE = (
    "whole_body_tracking.tasks.tracking.mdp."
    "action_ball_full_mdp_lean_runtime"
)
FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_QUALNAME = (
    "ActionBallFullMdpLeanRuntimeOwner"
)
FULL_MDP_DIAGNOSTIC_RUNTIME_DEPENDENCY_KIND = (
    "action_ball_epoch_runtime_dependencies_v1"
)
FULL_MDP_CONFIG_MODULE = (
    "whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg"
)
FULL_MDP_DIAGNOSTIC_CONFIG_TYPES = (
    (
        "HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg",
        "A",
        "HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0",
    ),
    (
        "HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg",
        "C",
        "HOPE-PingPong-ActionBall-FullMdpC-AgibotA3-v0",
    ),
)
_FULL_MDP_TERMINATION_REASON_BITS = (
    ("time_out", 1),
    ("base_fell_tilt", 2),
    ("base_too_low", 4),
    ("joint_qdes_forbidden", 8),
    ("robot_hit_table", 16),
)
FULL_MDP_GYM_ENTRY_POINT = (
    "whole_body_tracking.tasks.tracking.full_mdp_env:"
    "ActionBallFullMdpManagerBasedRLEnv"
)
class FullMdpPhysicsEventPhase(IntEnum):
    """Integer event phase shared by strike, landing and recovery owners."""

    POST_SCENE_UPDATE = 1


class FullMdpPhysicsSubstepStamp(NamedTuple):
    """Exact post-update location of one physics transition.

    ``control_step`` is the one-based manager step whose counters, Termination
    and Reward have not run yet.  ``physics_substep`` is zero-based within that
    control step.  ``sim_step`` is IsaacLab's one-based global physics counter
    after the simulator and scene buffers have both advanced.
    """

    control_step: int
    physics_substep: int
    physics_substeps_per_control: int
    sim_step: int
    event_phase: FullMdpPhysicsEventPhase

    def exact_tuple(self) -> tuple[int, int, int, int, int]:
        """Return the integer-only portable stamp without tensor conversion."""

        return (
            self.control_step,
            self.physics_substep,
            self.physics_substeps_per_control,
            self.sim_step,
            int(self.event_phase),
        )


class FullMdpPrePhysicsSubstepStamp(NamedTuple):
    """Exact pre-write location for one Physical launch opportunity."""

    control_step: int
    physics_substep: int
    physics_substeps_per_control: int
    sim_step_before: int


@dataclass(frozen=True, slots=True)
class FullMdpLeanRuntimeComponents:
    """One diagnostic epoch graph installed before ObservationManager.

    The top owner is retained here because the factory constructs it exactly
    once while all leaf identities are still off-side.  ``create_from_env``
    later retrieves this same object after the environment seals; it is not a
    second constructor.  There is intentionally no separate drain role: the
    lean top owns the single packed epoch drain itself.
    """

    epoch_owner: object
    device_r05_owner: object
    motion_owner: object
    racket_owner: object
    physical_owner: object
    r03_owner: object
    r06_owner: object
    r07_owner: object
    r07_plant_fact_adapter: object
    reward_graph: object
    lean_runtime_owner: object
    observation_source: object

    def __post_init__(self) -> None:
        values = (
            self.epoch_owner,
            self.device_r05_owner,
            self.motion_owner,
            self.racket_owner,
            self.physical_owner,
            self.r03_owner,
            self.r06_owner,
            self.r07_owner,
            self.r07_plant_fact_adapter,
            self.reward_graph,
            self.lean_runtime_owner,
            self.observation_source,
        )
        if any(value is None for value in values):
            raise FullMdpPostPhysicsOwnerMissingError(
                "lean full-MDP component registry contains a missing role"
            )
        if len({id(value) for value in values}) != len(values):
            raise FullMdpPostPhysicsProtocolError(
                "lean full-MDP component registry aliases distinct roles"
            )


class FullMdpNativeResetContext(NamedTuple):
    """Read-only native boundary coordinates, never reset authority or facts."""

    common_step_counter: int
    sim_step_counter: int
    decimation: int


class FullMdpSelectedResetEvent:
    """Opaque environment-issued identity for one selected native reset.

    Instances have no readable payload.  The exact environment that minted an
    event retains its device selection and generation after-image in a private,
    single-active-event registry.  Constructing another instance therefore
    grants no reset authority.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class FullMdpSelectedResetProjection:
    """Clone-only device projection consumed by the construction-bound top.

    ``generation_after`` increments exactly the selected mask.  The environment
    advances its live ledger only after the lean top accepts the projection and
    returns successfully.
    """

    reset_event_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    generation_overflow_fault: torch.Tensor
    terminal_reset_facts_i64: torch.Tensor


@dataclass(frozen=True, slots=True)
class FullMdpResetGenesisProjection:
    """Independent world-reset chronology projected onto the env device."""

    world_reset_identity: object
    reset_generations: torch.Tensor


@dataclass(slots=True)
class _FullMdpSelectedResetRecord:
    event: FullMdpSelectedResetEvent
    reset_event_identity: object
    selected_env_index: torch.Tensor
    selected_mask: torch.Tensor
    generation_before: torch.Tensor
    generation_after: torch.Tensor
    generation_overflow_fault: torch.Tensor
    terminal_reset_facts_i64: torch.Tensor
    projected: bool = False


@dataclass(frozen=True, slots=True)
class _FullMdpResetGenesisInstall:
    authority: object
    receipt: object


@dataclass(slots=True)
class _FullMdpResetCallpointAuthority:
    """Single-use exact tensor identity minted by a reviewed env callpoint."""

    env_ids: torch.Tensor
    source: str


FullMdpRuntimeOwnerFactory = Callable[
    ["ActionBallFullMdpManagerBasedRLEnv", object],
    object,
]


class FullMdpUpstreamSourceDriftError(RuntimeError):
    """The installed IsaacLab step is not the reviewed Pod1 implementation."""


class FullMdpPostPhysicsOwnerMissingError(RuntimeError):
    """The fresh environment was constructed without its physical-fact owner."""


class FullMdpPostPhysicsProtocolError(RuntimeError):
    """The post-physics owner or dispatch sequence violated its contract."""


class FullMdpPostPhysicsPoisonedError(RuntimeError):
    """A prior partial physics transition makes continuation untrustworthy."""


class FullMdpUnsupportedRuntimeError(RuntimeError):
    """The fresh seam was requested from unsupported extension mode."""


@dataclass(frozen=True, slots=True)
class FullMdpManagerConstructionFailureSnapshot:
    """Traceback-free summary retained by a failed cold environment."""

    exception_type: str
    message: str
    phase: str


def _manager_construction_failure_snapshot(
    exc: BaseException,
    *,
    phase: str,
) -> FullMdpManagerConstructionFailureSnapshot:
    """Retain no exception/context/traceback graph on the failed env."""

    return FullMdpManagerConstructionFailureSnapshot(
        exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
        message=str(exc),
        phase=phase,
    )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return ""


def _manager_step_node(source_text: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        raise FullMdpUpstreamSourceDriftError(
            "pinned ManagerBasedRLEnv source is not valid Python"
        ) from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRLEnv"
    ]
    if len(classes) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "expected exactly one ManagerBasedRLEnv class"
        )
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    ]
    if len(methods) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "expected exactly one ManagerBasedRLEnv.step method"
        )
    return methods[0]


def _manager_load_managers_node(source_text: str) -> ast.FunctionDef:
    """Return the one reviewed upstream manager-construction method."""

    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        raise FullMdpUpstreamSourceDriftError(
            "pinned ManagerBasedRLEnv source is not valid Python"
        ) from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRLEnv"
    ]
    if len(classes) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "expected exactly one ManagerBasedRLEnv class"
        )
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "load_managers"
    ]
    if len(methods) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "expected exactly one ManagerBasedRLEnv.load_managers method"
        )
    return methods[0]


def _statement_call_index(statements: list[ast.stmt], name: str) -> int:
    matches = [
        index
        for index, statement in enumerate(statements)
        if any(
            isinstance(node, ast.Call) and _call_name(node) == name
            for node in ast.walk(statement)
        )
    ]
    if len(matches) != 1:
        raise FullMdpUpstreamSourceDriftError(
            f"expected one {name} statement, found {matches!r}"
        )
    return matches[0]


def _assert_pinned_upstream_manager_order(source_text: str) -> None:
    """Keep the copied manager loader at its reviewed IsaacLab 2.3.2 order."""

    method = _manager_load_managers_node(source_text)
    command_index = _statement_call_index(method.body, "CommandManager")
    parent_index = _statement_call_index(method.body, "load_managers")
    termination_index = _statement_call_index(method.body, "TerminationManager")
    reward_index = _statement_call_index(method.body, "RewardManager")
    curriculum_index = _statement_call_index(method.body, "CurriculumManager")
    spaces_index = _statement_call_index(
        method.body, "self._configure_gym_env_spaces"
    )
    startup_index = _statement_call_index(
        method.body, "self.event_manager.apply"
    )
    if not (
        command_index
        < parent_index
        < termination_index
        < reward_index
        < curriculum_index
        < spaces_index
        < startup_index
    ):
        raise FullMdpUpstreamSourceDriftError(
            "upstream manager construction order differs from the reviewed "
            "command/base/termination/reward/curriculum/spaces/startup order"
        )


def _assert_pinned_upstream_step_order(source_text: str) -> None:
    """Reject a legal upstream reorder even before the byte pin is considered."""

    step = _manager_step_node(source_text)
    loops = [
        statement
        for statement in step.body
        if isinstance(statement, ast.For)
        and isinstance(statement.iter, ast.Call)
        and _call_name(statement.iter) == "range"
        and len(statement.iter.args) == 1
        and _dotted_name(statement.iter.args[0]) == "self.cfg.decimation"
    ]
    if len(loops) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "upstream step must have one cfg.decimation physics loop"
        )
    loop = loops[0]
    if loop.orelse:
        raise FullMdpUpstreamSourceDriftError(
            "upstream physics loop unexpectedly has an else branch"
        )
    apply_index = _statement_call_index(
        loop.body, "self.action_manager.apply_action"
    )
    write_index = _statement_call_index(loop.body, "self.scene.write_data_to_sim")
    sim_index = _statement_call_index(loop.body, "self.sim.step")
    recorder_index = _statement_call_index(
        loop.body, "self.recorder_manager.record_post_physics_decimation_step"
    )
    render_index = _statement_call_index(loop.body, "self.sim.render")
    update_index = _statement_call_index(loop.body, "self.scene.update")
    if not (
        apply_index
        < write_index
        < sim_index
        < recorder_index
        < render_index
        < update_index
    ):
        raise FullMdpUpstreamSourceDriftError(
            "upstream physics order differs from "
            "apply/write/sim/recorder/render/update"
        )
    if update_index != len(loop.body) - 1:
        raise FullMdpUpstreamSourceDriftError(
            "scene.update is no longer the final upstream physics-loop statement"
        )

    process_index = _statement_call_index(
        step.body, "self.action_manager.process_action"
    )
    loop_index = step.body.index(loop)
    termination_index = _statement_call_index(
        step.body, "self.termination_manager.compute"
    )
    reward_index = _statement_call_index(step.body, "self.reward_manager.compute")
    if not process_index < loop_index < termination_index < reward_index:
        raise FullMdpUpstreamSourceDriftError(
            "upstream manager order differs from action/physics/termination/reward"
        )
    common_counter_indices = [
        index
        for index, statement in enumerate(step.body)
        if isinstance(statement, ast.AugAssign)
        and _dotted_name(statement.target) == "self.common_step_counter"
        and isinstance(statement.op, ast.Add)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value == 1
    ]
    if len(common_counter_indices) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "upstream common_step_counter increment differs"
        )
    if not loop_index < common_counter_indices[0] < termination_index:
        raise FullMdpUpstreamSourceDriftError(
            "upstream common-step clock no longer follows all physics substeps"
        )


def _validate_pinned_upstream_source_bytes(source_bytes: bytes) -> None:
    """Validate exact bytes plus reviewed step and manager-loader methods."""

    file_sha = hashlib.sha256(source_bytes).hexdigest()
    if file_sha != PINNED_MANAGER_BASED_RL_ENV_FILE_SHA256:
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedRLEnv source SHA drift: "
            f"expected {PINNED_MANAGER_BASED_RL_ENV_FILE_SHA256}, got {file_sha}"
        )
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedRLEnv source is not UTF-8"
        ) from exc
    _assert_pinned_upstream_step_order(source_text)
    _assert_pinned_upstream_manager_order(source_text)


def _assert_runtime_uses_pinned_upstream_step() -> None:
    methods = (
        ManagerBasedRLEnv.__init__,
        ManagerBasedRLEnv.load_managers,
        ManagerBasedRLEnv.step,
        ManagerBasedRLEnv._reset_idx,
    )
    source_path_value = inspect.getsourcefile(ManagerBasedRLEnv)
    method_source_values = tuple(inspect.getsourcefile(method) for method in methods)
    if source_path_value is None or any(
        value is None for value in method_source_values
    ):
        raise FullMdpUpstreamSourceDriftError(
            "cannot resolve live ManagerBasedRLEnv source"
        )
    source_path = Path(source_path_value).resolve()
    method_source_paths = tuple(
        Path(value).resolve() for value in method_source_values if value is not None
    )
    if not source_path.is_file() or any(
        value != source_path for value in method_source_paths
    ):
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedRLEnv lifecycle methods do not resolve to one source file"
        )
    if any(
        method.__module__ != "isaaclab.envs.manager_based_rl_env"
        for method in methods
    ):
        raise FullMdpUpstreamSourceDriftError(
            "live ManagerBasedRLEnv lifecycle was replaced outside its pinned module"
        )
    _validate_pinned_upstream_source_bytes(source_path.read_bytes())


def _validate_pinned_manager_based_env_source_bytes(
    source_bytes: bytes,
) -> None:
    """Pin the exact parent loader whose body the local override invokes."""

    actual_file_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_file_sha != PINNED_MANAGER_BASED_ENV_FILE_SHA256:
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedEnv source SHA drift: expected "
            f"{PINNED_MANAGER_BASED_ENV_FILE_SHA256}, got {actual_file_sha}"
        )
    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedEnv source is not valid UTF-8 Python"
        ) from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedEnv"
    ]
    methods = (
        []
        if len(classes) != 1
        else [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_managers"
        ]
    )
    if len(methods) != 1:
        raise FullMdpUpstreamSourceDriftError(
            "expected exactly one ManagerBasedEnv.load_managers method"
        )
    recorder_index = _statement_call_index(methods[0].body, "RecorderManager")
    action_index = _statement_call_index(methods[0].body, "ActionManager")
    observation_index = _statement_call_index(
        methods[0].body, "ObservationManager"
    )
    if not recorder_index < action_index < observation_index:
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedEnv manager order differs from recorder/action/observation"
        )


def _assert_runtime_uses_pinned_manager_based_env() -> None:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    if ManagerBasedRLEnv.__mro__[1] is not ManagerBasedEnv:
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedRLEnv no longer directly derives from ManagerBasedEnv"
        )
    methods = (
        ManagerBasedEnv.__init__,
        ManagerBasedEnv.load_managers,
        ManagerBasedEnv.reset,
    )
    class_path_value = inspect.getsourcefile(ManagerBasedEnv)
    method_path_values = tuple(inspect.getsourcefile(method) for method in methods)
    if class_path_value is None or any(value is None for value in method_path_values):
        raise FullMdpUpstreamSourceDriftError(
            "cannot resolve live ManagerBasedEnv source"
        )
    class_path = Path(class_path_value).resolve()
    method_paths = tuple(
        Path(value).resolve() for value in method_path_values if value is not None
    )
    if not class_path.is_file() or any(value != class_path for value in method_paths):
        raise FullMdpUpstreamSourceDriftError(
            "ManagerBasedEnv lifecycle methods do not resolve to one source"
        )
    if any(method.__module__ != "isaaclab.envs.manager_based_env" for method in methods):
        raise FullMdpUpstreamSourceDriftError(
            "live ManagerBasedEnv lifecycle was replaced outside its module"
        )
    _validate_pinned_manager_based_env_source_bytes(class_path.read_bytes())


def _cold_local_plain_callable_defaults(
    function: types.FunctionType,
) -> tuple[
    tuple[tuple[type, object], ...] | None,
    tuple[tuple[str, type, object], ...] | None,
]:
    """Freeze the finite plain-literal default state without a self-digest."""

    plain_types = (type(None), bool, int, float, str, bytes)
    positional = function.__defaults__
    if positional is not None and (
        type(positional) is not tuple
        or any(type(value) not in plain_types for value in positional)
    ):
        raise FullMdpUpstreamSourceDriftError(
            "fresh full-MDP direct method has non-plain positional defaults"
        )
    keyword = function.__kwdefaults__
    if keyword is not None and (
        type(keyword) is not dict
        or any(
            type(name) is not str or type(value) not in plain_types
            for name, value in keyword.items()
        )
    ):
        raise FullMdpUpstreamSourceDriftError(
            "fresh full-MDP direct method has non-plain keyword defaults"
        )
    positional_snapshot = (
        None
        if positional is None
        else tuple((type(value), value) for value in positional)
    )
    keyword_snapshot = (
        None
        if keyword is None
        else tuple(
            sorted(
                (name, type(value), value)
                for name, value in keyword.items()
            )
        )
    )
    return positional_snapshot, keyword_snapshot


def _assert_runtime_uses_pinned_local_step(instance: object | None = None) -> None:
    """Reject post-import replacement of the executable local env surface.

    The expected operands are the module, class and function objects captured
    once after this module finished defining the class.  No digest is derived
    from the same local source being admitted.  The original checker and
    expected operands remain inside the trusted Python module boundary.
    """

    module_object = sys.modules.get(__name__)
    pinned_module = _PINNED_LOCAL_FULL_MDP_MODULE
    pinned_class = _PINNED_LOCAL_FULL_MDP_ENV_CLASS
    if (
        module_object is not pinned_module
        or not isinstance(module_object, types.ModuleType)
        or ActionBallFullMdpManagerBasedRLEnv is not pinned_class
        or vars(module_object).get(
            "ActionBallFullMdpManagerBasedRLEnv", _ABSENT
        )
        is not pinned_class
    ):
        raise FullMdpUpstreamSourceDriftError(
            "loaded fresh full-MDP class is not its cold-bound module export"
        )

    module_source_value = inspect.getsourcefile(module_object)
    class_source_value = inspect.getsourcefile(pinned_class)
    if module_source_value is None or class_source_value is None:
        raise FullMdpUpstreamSourceDriftError(
            "cannot resolve cold-bound fresh full-MDP env source"
        )
    module_source = Path(module_source_value).resolve()
    if (
        not module_source.is_file()
        or Path(class_source_value).resolve() != module_source
    ):
        raise FullMdpUpstreamSourceDriftError(
            "cold-bound fresh full-MDP class does not share its module source"
        )

    live_class_dict = vars(pinned_class)
    for (
        name,
        pinned_function,
        pinned_code,
        pinned_defaults,
    ) in _PINNED_LOCAL_FULL_MDP_DIRECT_METHODS:
        live_function = live_class_dict.get(name, _ABSENT)
        source_value = (
            inspect.getsourcefile(live_function)
            if type(live_function) is types.FunctionType
            else None
        )
        if (
            type(pinned_function) is not types.FunctionType
            or type(pinned_code) is not types.CodeType
            or type(live_function) is not types.FunctionType
            or live_function is not pinned_function
            or live_function.__code__ is not pinned_code
            or _cold_local_plain_callable_defaults(live_function)
            != pinned_defaults
            or live_function.__globals__ is not globals()
            or source_value is None
            or Path(source_value).resolve() != module_source
        ):
            raise FullMdpUpstreamSourceDriftError(
                f"cold-bound fresh full-MDP method {name!r} was replaced"
            )

    for (
        name,
        pinned_descriptor,
        pinned_function,
        pinned_code,
        pinned_defaults,
    ) in _PINNED_LOCAL_FULL_MDP_PROPERTY_GETTERS:
        live_descriptor = live_class_dict.get(name, _ABSENT)
        live_function = (
            live_descriptor.fget
            if type(live_descriptor) is property
            else None
        )
        source_value = (
            inspect.getsourcefile(live_function)
            if type(live_function) is types.FunctionType
            else None
        )
        if (
            type(pinned_descriptor) is not property
            or live_descriptor is not pinned_descriptor
            or type(pinned_function) is not types.FunctionType
            or type(pinned_code) is not types.CodeType
            or type(live_function) is not types.FunctionType
            or live_function is not pinned_function
            or live_function.__code__ is not pinned_code
            or _cold_local_plain_callable_defaults(live_function)
            != pinned_defaults
            or live_function.__globals__ is not globals()
            or source_value is None
            or Path(source_value).resolve() != module_source
        ):
            raise FullMdpUpstreamSourceDriftError(
                f"cold-bound fresh full-MDP getter {name!r} was replaced"
            )

    if instance is not None:
        _assert_runtime_instance_uses_pinned_local_step(instance)


def _assert_runtime_instance_uses_pinned_local_step(instance: object) -> None:
    """Reject per-instance shadows before the constructor mints its lease."""

    if type(instance) is not _PINNED_LOCAL_FULL_MDP_ENV_CLASS:
        raise FullMdpUpstreamSourceDriftError(
            "fresh full-MDP instance does not use the cold-bound class"
        )
    instance_dict = vars(instance)
    protected_names = {
        name
        for name, _function, _code, _defaults
        in _PINNED_LOCAL_FULL_MDP_DIRECT_METHODS
    }
    protected_names.update(
        name
        for name, _descriptor, _function, _code, _defaults
        in _PINNED_LOCAL_FULL_MDP_PROPERTY_GETTERS
    )
    rebound = tuple(sorted(protected_names.intersection(instance_dict)))
    if rebound:
        raise FullMdpUpstreamSourceDriftError(
            "fresh full-MDP instance overrides cold-bound executable names: "
            + ",".join(rebound)
        )


def _require_owner_factory(
    factory: FullMdpRuntimeOwnerFactory | None,
) -> FullMdpRuntimeOwnerFactory:
    if factory is None or not callable(factory):
        raise FullMdpPostPhysicsOwnerMissingError(
            "fresh full-MDP env requires one post-physics owner factory: "
            "the top runtime owner"
        )
    return factory


def _resolve_action_ball_full_mdp_runtime_graph_builder() -> types.FunctionType:
    """Resolve the one code-owned graph builder, never a caller/config hook."""

    try:
        module_object = importlib.import_module(FULL_MDP_RUNTIME_FACTORY_MODULE)
    except Exception as exc:
        raise FullMdpPostPhysicsOwnerMissingError(
            "fresh full-MDP production runtime factory module is unavailable; "
            "construction remains HOLD"
        ) from exc
    if not isinstance(module_object, types.ModuleType):
        raise FullMdpPostPhysicsProtocolError(
            "fresh full-MDP runtime factory import is not one module"
        )
    builder = vars(module_object).get(FULL_MDP_RUNTIME_FACTORY_QUALNAME)
    if (
        not isinstance(builder, types.FunctionType)
        or builder.__module__ != FULL_MDP_RUNTIME_FACTORY_MODULE
        or builder.__qualname__ != FULL_MDP_RUNTIME_FACTORY_QUALNAME
        or builder.__globals__ is not vars(module_object)
    ):
        raise FullMdpPostPhysicsProtocolError(
            "fresh full-MDP runtime graph builder is absent, wrapped or foreign"
        )
    return builder


def _require_exact_lean_owner_factory(
    factory: FullMdpRuntimeOwnerFactory,
) -> None:
    """Require the one code-owned lean owner factory."""

    try:
        module_object = importlib.import_module(
            FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE
        )
    except Exception as exc:
        raise FullMdpPostPhysicsOwnerMissingError(
            "single-action lean code-owned top-owner factory is unavailable"
        ) from exc
    owner_type = vars(module_object).get(
        FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_QUALNAME
    )
    descriptor = (
        vars(owner_type).get("create_from_env")
        if type(owner_type) is type
        else None
    )
    if (
        type(owner_type) is not type
        or owner_type.__module__ != FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE
        or owner_type.__qualname__
        != FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_QUALNAME
        or vars(module_object).get(FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_QUALNAME)
        is not owner_type
        or type(descriptor) is not classmethod
        or not isinstance(factory, types.MethodType)
        or factory.__self__ is not owner_type
        or factory.__func__ is not descriptor.__func__
    ):
        raise FullMdpPostPhysicsOwnerMissingError(
            "single-action lean rejects a foreign, wrapped or caller-selected "
            "top-owner factory before simulator construction"
        )


def _require_single_action_lean_cfg(
    cfg: object,
    *,
    owner_factory: FullMdpRuntimeOwnerFactory,
) -> None:
    """Admit only the two exact code-owned disposable EnvCfg leaves.

    This is not readiness or launch authorization.  The fixed pre-manager
    builder repeats the live Command/scene/canary checks after CommandManager
    construction.
    """

    try:
        config_module = importlib.import_module(FULL_MDP_CONFIG_MODULE)
        gym = importlib.import_module("gymnasium")
        canary = importlib.import_module(
            "action_ball_full_mdp_canary_target_profile"
        )
        factory_module = importlib.import_module(
            FULL_MDP_RUNTIME_FACTORY_MODULE
        )
    except Exception as exc:
        raise FullMdpPostPhysicsOwnerMissingError(
            "single-action lean code-owned cfg/builder closure is unavailable"
        ) from exc
    exact_types = {
        vars(config_module).get(name): (role, task_id)
        for name, role, task_id in FULL_MDP_DIAGNOSTIC_CONFIG_TYPES
    }
    identity = exact_types.get(type(cfg))
    if identity is None:
        raise FullMdpPostPhysicsOwnerMissingError(
            "single-action lean may reach the builder only for one exact "
            "registered A/C full-MDP EnvCfg type"
        )
    role, task_id = identity
    try:
        registration = gym.spec(task_id)
    except Exception as exc:
        raise FullMdpPostPhysicsOwnerMissingError(
            "exact diagnostic full-MDP Gym registration is unavailable"
        ) from exc
    scene = getattr(cfg, "scene", None)
    commands = getattr(cfg, "commands", None)
    racket = getattr(commands, "racket_target", None)
    failures: list[str] = []
    if (
        getattr(registration, "entry_point", None) != FULL_MDP_GYM_ENTRY_POINT
        or getattr(registration, "kwargs", {}).get("env_cfg_entry_point")
        is not type(cfg)
    ):
        failures.append("gym_registration_differs")
    if getattr(cfg, "action_ball_full_mdp_family_role", None) != role:
        failures.append("family_role_rewritten")
    if getattr(cfg, "obs_mode", None) != "action_ball_full_mdp":
        failures.append("obs_mode_differs")
    if getattr(racket, "target_mode", None) != "action_ball_full_mdp":
        failures.append("target_mode_differs")
    if getattr(racket, "action_ball_diagnostic_unauthorized", None) is not True:
        failures.append("diagnostic_unauthorized_differs")
    if type(getattr(scene, "num_envs", None)) is not int or scene.num_envs <= 0:
        failures.append("num_envs_must_be_positive_exact_int")
    if getattr(cfg, "action_ball_full_mdp_scene_capacity", None) != 2:
        failures.append("diagnostic_scene_capacity_differs")
    if getattr(cfg, "action_ball_full_mdp_capacity_receipt_sha256", None) != "":
        failures.append("formal_capacity_receipt_present")
    if (
        getattr(cfg, "action_ball_full_mdp_runtime_construction_status", None)
        != "HOLD"
    ):
        failures.append("cfg_runtime_status_is_not_hold")
    if getattr(cfg, "checkpoint_path", None) is not None:
        failures.append("checkpoint_resume_present")
    if getattr(cfg, "checkpoint_tolerant", False) is not False:
        failures.append("checkpoint_tolerant_present")
    if (
        getattr(factory_module, "RUNTIME_INTEGRATED", None) is not False
        or getattr(factory_module, "LAUNCH_AUTHORIZED", None) is not False
        or getattr(factory_module, "DIAGNOSTIC_UNAUTHORIZED", None) is not True
    ):
        failures.append("factory_authorization_flags_differ")
    if (
        getattr(canary, "CANARY_NUM_ENVS", None) != 2
        or getattr(canary, "CANARY_SAVE_CHECKPOINTS", None) is not False
        or getattr(canary, "DIAGNOSTIC_UNAUTHORIZED", None) is not True
        or getattr(canary, "FORMAL_PROFILE", None) is not False
        or getattr(canary, "FORMAL_LAUNCH_AUTHORIZED", None) is not False
    ):
        failures.append("canary_no_save_or_authorization_flags_differ")
    # Resolve the fixed module-owned function now; a cfg field or caller kwarg
    # cannot select a different pre-manager builder.
    _resolve_action_ball_full_mdp_runtime_graph_builder()
    _require_exact_lean_owner_factory(owner_factory)
    if failures:
        raise FullMdpPostPhysicsOwnerMissingError(
            "exact single_action_lean admission failed: "
            + ",".join(failures)
        )


@dataclass(frozen=True, slots=True)
class _LeanOwnerExecutableBinding:
    """Bound methods retained from the exact installed lean owner."""

    owner_type: type
    before_policy_step_function: types.FunctionType
    before_physics_substep_function: types.FunctionType
    publish_function: types.FunctionType
    after_reward_close_function: types.FunctionType
    after_command_compute_before_observation_function: types.FunctionType
    selected_true_reset_function: types.FunctionType


def _require_standalone_simulation_app() -> None:
    # SimulationApp sets this exact flag to False.  Extension mode defers
    # manager loading until reset_async, so an eager owner cannot be bound
    # safely within this constructor.
    if getattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", None) is not False:
        raise FullMdpUnsupportedRuntimeError(
            "fresh full-MDP env supports standalone SimulationApp only; "
            "extension-mode deferred manager loading is rejected"
        )


_ABSENT = object()


@dataclass(slots=True)
class _ProtectedValueSnapshot:
    """Host-visible identity/metadata/version snapshot of a manager tensor.

    This is a scoped accidental-mutation guard, not the authority for arbitrary
    owner behavior.  Inference tensors have no version counter and ``.data`` or
    raw-storage writes can bypass it.  Construction-time executable binding and
    the independent manager seam checks are the trust boundary; no asynchronous
    CUDA content assert is used here.
    """

    name: str
    present: bool
    reference: torch.Tensor | None
    metadata: tuple[object, ...]
    version: int | None

    @classmethod
    def capture(cls, name: str, value: object) -> "_ProtectedValueSnapshot":
        if value is _ABSENT:
            return cls(
                name=name,
                present=False,
                reference=None,
                metadata=(),
                version=None,
            )
        if not isinstance(value, torch.Tensor):
            raise FullMdpPostPhysicsProtocolError(
                f"protected manager value {name!r} must be a tensor or absent"
            )
        metadata = (
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
            str(value.layout),
            bool(value.requires_grad),
        )
        version = None if torch.is_inference(value) else int(value._version)
        return cls(
            name=name,
            present=True,
            reference=value,
            metadata=metadata,
            version=version,
        )

    def assert_unchanged(self, value: object) -> None:
        if not self.present:
            if value is not _ABSENT:
                raise FullMdpPostPhysicsProtocolError(
                    f"protected manager value {self.name!r} was installed by owner"
                )
            return
        if not isinstance(value, torch.Tensor) or value is not self.reference:
            raise FullMdpPostPhysicsProtocolError(
                f"protected manager tensor {self.name!r} changed identity"
            )
        metadata = (
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
            str(value.layout),
            bool(value.requires_grad),
        )
        if metadata != self.metadata:
            raise FullMdpPostPhysicsProtocolError(
                f"protected manager tensor {self.name!r} changed metadata"
            )
        if self.version is not None and int(value._version) != self.version:
            raise FullMdpPostPhysicsProtocolError(
                f"protected manager tensor {self.name!r} changed version"
            )


@dataclass(slots=True)
class _ProtectedManagerState:
    common_step_counter: int
    sim_step_counter: int
    values: tuple[_ProtectedValueSnapshot, ...]


class _ControlStepDispatch:
    """Single-control-step sequence guard with no tensor or simulator access."""

    __slots__ = (
        "control_step",
        "decimation",
        "sim_step_before",
        "next_substep",
        "pending_stamp",
        "pending_exact_tuple",
        "last_committed_exact_tuple",
    )

    def __init__(
        self, *, control_step: int, decimation: int, sim_step_before: int
    ) -> None:
        if type(control_step) is not int or control_step < 1:
            raise FullMdpPostPhysicsProtocolError(
                "control_step must be a positive plain integer"
            )
        if type(decimation) is not int or decimation < 1:
            raise FullMdpPostPhysicsProtocolError(
                "decimation must be a positive plain integer"
            )
        if type(sim_step_before) is not int or sim_step_before < 0:
            raise FullMdpPostPhysicsProtocolError(
                "sim_step_before must be a nonnegative plain integer"
            )
        self.control_step = control_step
        self.decimation = decimation
        self.sim_step_before = sim_step_before
        self.next_substep = 0
        self.pending_stamp: FullMdpPhysicsSubstepStamp | None = None
        self.pending_exact_tuple: tuple[int, int, int, int, int] | None = None
        self.last_committed_exact_tuple: (
            tuple[int, int, int, int, int] | None
        ) = None

    def prepare(
        self, *, physics_substep: int, sim_step: int
    ) -> FullMdpPhysicsSubstepStamp:
        if self.pending_stamp is not None or self.pending_exact_tuple is not None:
            raise FullMdpPostPhysicsProtocolError(
                "previous post-physics publication was not committed"
            )
        if type(physics_substep) is not int or type(sim_step) is not int:
            raise FullMdpPostPhysicsProtocolError(
                "physics_substep and sim_step must be plain integers"
            )
        if physics_substep != self.next_substep:
            raise FullMdpPostPhysicsProtocolError(
                "physics substep was skipped, duplicated or reordered"
            )
        expected_sim_step = self.sim_step_before + physics_substep + 1
        if sim_step != expected_sim_step:
            raise FullMdpPostPhysicsProtocolError(
                "post-physics publication did not follow the expected sim step"
            )
        stamp = FullMdpPhysicsSubstepStamp(
            control_step=self.control_step,
            physics_substep=physics_substep,
            physics_substeps_per_control=self.decimation,
            sim_step=sim_step,
            event_phase=FullMdpPhysicsEventPhase.POST_SCENE_UPDATE,
        )
        self.pending_stamp = stamp
        self.pending_exact_tuple = stamp.exact_tuple()
        return stamp

    def commit(self, stamp: FullMdpPhysicsSubstepStamp) -> None:
        if (
            stamp is not self.pending_stamp
            or self.pending_exact_tuple is None
            or stamp.exact_tuple() != self.pending_exact_tuple
        ):
            raise FullMdpPostPhysicsProtocolError(
                "post-physics commit does not match the sealed exact stamp"
            )
        self.last_committed_exact_tuple = self.pending_exact_tuple
        self.pending_stamp = None
        self.pending_exact_tuple = None
        self.next_substep += 1

    def finish(self) -> None:
        if self.pending_stamp is not None or self.pending_exact_tuple is not None:
            raise FullMdpPostPhysicsProtocolError(
                "final post-physics publication is still pending"
            )
        if self.next_substep != self.decimation:
            raise FullMdpPostPhysicsProtocolError(
                "final physics substep was skipped"
            )


class ActionBallFullMdpManagerBasedRLEnv(ManagerBasedRLEnv):
    """Executable-bound fresh environment with one exact top runtime owner."""

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        *,
        full_mdp_runtime_owner_factory: FullMdpRuntimeOwnerFactory | None = None,
        full_mdp_cold_restore_dormant: bool = False,
        **kwargs,
    ) -> None:
        if type(self) is not ActionBallFullMdpManagerBasedRLEnv:
            raise FullMdpUnsupportedRuntimeError(
                "fresh full-MDP env rejects subclasses of its pinned step owner"
            )
        if type(full_mdp_cold_restore_dormant) is not bool:
            raise FullMdpUnsupportedRuntimeError(
                "full_mdp_cold_restore_dormant must be an exact bool"
            )
        if full_mdp_cold_restore_dormant:
            # R10 restore is not allowed to construct a live ordinary env and
            # patch it afterwards.  This failure deliberately precedes source
            # inspection, super(), simulator allocation, reset, observation,
            # and noise so the unsupported mode has no hidden fresh side effect.
            raise FullMdpUnsupportedRuntimeError(
                "dormant full-MDP cold restore remains HOLD before simulator, "
                "reset, observation and noise construction"
            )
        factory = _require_owner_factory(full_mdp_runtime_owner_factory)
        _require_single_action_lean_cfg(cfg, owner_factory=factory)
        _require_standalone_simulation_app()
        _assert_runtime_uses_pinned_upstream_step()
        _assert_runtime_uses_pinned_manager_based_env()
        _assert_runtime_uses_pinned_local_step()
        _assert_runtime_instance_uses_pinned_local_step(self)
        # This lease must already exist when the pinned base constructor calls
        # our ``load_managers`` override.  It is environment-owned and never a
        # launcher/train argument.  The second reference is retained only to
        # detect a base callback replacing the identity before ``super``
        # returns; no new lease is minted at the post-super boundary.
        lease = object()
        self._action_ball_full_mdp_runtime_lease = lease
        self._action_ball_full_mdp_runtime_lease_identity_at_mint = lease
        self._action_ball_full_mdp_manager_construction_state = "armed"
        self._action_ball_full_mdp_runtime_graph_builder_invocations = 0
        self._action_ball_full_mdp_base_construction_state = "entered"
        try:
            super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
            self._action_ball_full_mdp_base_construction_state = "returned"
            if self._action_ball_full_mdp_manager_construction_state != (
                "base_managers_complete"
            ):
                raise FullMdpPostPhysicsProtocolError(
                    "fresh full-MDP base construction returned without one "
                    "complete manager construction"
                )
            self._seal_action_ball_full_mdp_after_base_construction(lease)
            owner = factory(self, lease)
            executable_binding = self._validate_lean_owner_install(
                owner, expected_lease=lease
            )
            self._full_mdp_runtime_owner = owner
            self._full_mdp_before_policy_step = (
                executable_binding.before_policy_step_function.__get__(
                    owner, executable_binding.owner_type
                )
            )
            self._full_mdp_before_physics_substep = (
                executable_binding.before_physics_substep_function.__get__(
                    owner, executable_binding.owner_type
                )
            )
            self._full_mdp_post_physics_publish = (
                executable_binding.publish_function.__get__(
                    owner, executable_binding.owner_type
                )
            )
            self._full_mdp_after_reward_close = (
                executable_binding.after_reward_close_function.__get__(
                    owner, executable_binding.owner_type
                )
            )
            self._full_mdp_after_command_compute_before_observation = (
                executable_binding.
                after_command_compute_before_observation_function.__get__(
                    owner, executable_binding.owner_type
                )
            )
            self._full_mdp_selected_true_reset = (
                executable_binding.selected_true_reset_function.__get__(
                    owner, executable_binding.owner_type
                )
            )
        except BaseException as exc:
            # Preserve the owner-construction failure even if a partially
            # initialized environment cannot finish cleanup.  External
            # one-shot authorities may already have been consumed, so this is
            # a cold-discard boundary, never a fabricated rollback.
            if self._action_ball_full_mdp_manager_construction_state != "failed":
                self._action_ball_full_mdp_manager_construction_state = "failed"
                self._action_ball_full_mdp_manager_construction_failure = (
                    _manager_construction_failure_snapshot(
                        exc,
                        phase="runtime_owner_construction",
                    )
                )
            self._poison_action_ball_full_mdp_construction_installs()
            try:
                self.close()
            except BaseException:
                pass
            raise
        self._full_mdp_active_dispatch: _ControlStepDispatch | None = None
        self._full_mdp_last_after_reward_close_control_step = 0
        self._full_mdp_post_physics_poison: (
            tuple[str, tuple[int, int, int, int, int] | None] | None
        ) = None

    def load_managers(self) -> None:
        """Construct the fresh owner graph at the sole reviewed base seam.

        IsaacLab's ObservationManager calls observation terms during its own
        constructor to infer shapes.  The complete production graph must
        therefore be installed after CommandManager has produced the exact
        Motion/Racket identities but before the base loader constructs Action
        and Observation managers.  A failed attempt is sticky and cannot be
        retried against partially constructed managers.
        """

        state = getattr(
            self, "_action_ball_full_mdp_manager_construction_state", None
        )
        if state != "armed":
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP manager construction was replayed or previously failed"
            )
        self._action_ball_full_mdp_manager_construction_state = "entered"
        try:
            common_step_counter = getattr(self, "common_step_counter", None)
            episode_length_buf = getattr(self, "episode_length_buf", None)
            if type(common_step_counter) is not int or common_step_counter != 0:
                raise FullMdpPostPhysicsProtocolError(
                    "IsaacLab 8320 common-step clock was not initialized before managers"
                )
            if (
                not isinstance(episode_length_buf, torch.Tensor)
                or tuple(episode_length_buf.shape) != (self.cfg.scene.num_envs,)
                or episode_length_buf.dtype != torch.long
                or episode_length_buf.device != torch.device(self.cfg.sim.device)
            ):
                raise FullMdpPostPhysicsProtocolError(
                    "IsaacLab 8320 episode-length buffer was not initialized before managers"
                )
            from isaaclab.envs.manager_based_env import ManagerBasedEnv
            from isaaclab.managers import (
                CommandManager,
                CurriculumManager,
                RewardManager,
                TerminationManager,
            )

            self.command_manager = CommandManager(self.cfg.commands, self)
            print("[INFO] Command Manager: ", self.command_manager)
            self._action_ball_full_mdp_manager_construction_state = (
                "command_manager_ready"
            )
            self._assert_action_ball_full_mdp_runtime_lease_identity()

            if self._action_ball_full_mdp_runtime_graph_builder_invocations != 0:
                raise FullMdpPostPhysicsProtocolError(
                    "fresh full-MDP runtime graph builder cannot be replayed"
                )
            self._action_ball_full_mdp_runtime_graph_builder_invocations = 1
            builder = _resolve_action_ball_full_mdp_runtime_graph_builder()
            result = builder(self)
            self._assert_action_ball_full_mdp_runtime_lease_identity()
            if result is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "fresh full-MDP runtime graph builder must install in place "
                    "and return None"
                )
            if type(
                getattr(
                    self,
                    "_action_ball_full_mdp_reset_genesis_install",
                    None,
                )
            ) is not _FullMdpResetGenesisInstall:
                raise FullMdpPostPhysicsOwnerMissingError(
                    "fresh full-MDP runtime factory did not install the "
                    "independent reset genesis authority and receipt"
                )
            self._capture_action_ball_full_mdp_runtime_components()
            self._action_ball_full_mdp_manager_construction_state = (
                "runtime_graph_ready"
            )

            # This is the exact parent call made by the pinned upstream
            # ManagerBasedRLEnv loader.  It creates recorder/action/observation
            # managers in that order without constructing CommandManager twice.
            ManagerBasedEnv.load_managers(self)

            self.termination_manager = TerminationManager(
                self.cfg.terminations, self
            )
            print("[INFO] Termination Manager: ", self.termination_manager)
            self.reward_manager = RewardManager(self.cfg.rewards, self)
            print("[INFO] Reward Manager: ", self.reward_manager)
            self.curriculum_manager = CurriculumManager(
                self.cfg.curriculum, self
            )
            print("[INFO] Curriculum Manager: ", self.curriculum_manager)
            self._configure_gym_env_spaces()
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")
        except BaseException as exc:
            self._action_ball_full_mdp_manager_construction_state = "failed"
            self._action_ball_full_mdp_manager_construction_failure = (
                _manager_construction_failure_snapshot(
                    exc,
                    phase="manager_graph_construction",
                )
            )
            self._poison_action_ball_full_mdp_construction_installs()
            raise
        self._action_ball_full_mdp_manager_construction_state = (
            "base_managers_complete"
        )

    def _assert_action_ball_full_mdp_runtime_lease_identity(self) -> None:
        lease = getattr(self, "_action_ball_full_mdp_runtime_lease", _ABSENT)
        minted = getattr(
            self,
            "_action_ball_full_mdp_runtime_lease_identity_at_mint",
            _ABSENT,
        )
        if lease is _ABSENT or minted is _ABSENT or lease is not minted:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP construction lease identity changed"
            )

    def _poison_action_ball_full_mdp_construction_installs(self) -> None:
        """Drop env publication refs without claiming external rollback."""

        if hasattr(self, "_action_ball_full_mdp_components"):
            self._action_ball_full_mdp_components = _ABSENT
        if hasattr(self, "_action_ball_full_mdp_reset_genesis_install"):
            self._action_ball_full_mdp_reset_genesis_install = _ABSENT
        if hasattr(self, "_action_ball_full_mdp_lean_reward_graph"):
            self._action_ball_full_mdp_lean_reward_graph = _ABSENT
        if hasattr(self, "_action_ball_full_mdp_lean_observation_source"):
            self._action_ball_full_mdp_lean_observation_source = _ABSENT
        for manager_name in (
            "observation_manager",
            "termination_manager",
            "reward_manager",
        ):
            if hasattr(self, manager_name):
                setattr(self, manager_name, _ABSENT)
        for name in (
            "_full_mdp_runtime_owner",
            "_full_mdp_before_policy_step",
            "_full_mdp_before_physics_substep",
            "_full_mdp_post_physics_publish",
            "_full_mdp_after_reward_close",
            "_full_mdp_after_command_compute_before_observation",
            "_full_mdp_selected_true_reset",
        ):
            if hasattr(self, name):
                setattr(self, name, _ABSENT)

    def _seal_action_ball_full_mdp_after_base_construction(
        self, expected_lease: object
    ) -> None:
        if self._action_ball_full_mdp_manager_construction_state != (
            "base_managers_complete"
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP construction cannot seal before base managers"
            )
        self._assert_action_ball_full_mdp_runtime_lease_identity()
        if expected_lease is not self._action_ball_full_mdp_runtime_lease:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP super-return lease identity changed"
            )
        self._capture_action_ball_full_mdp_runtime_components()
        self._capture_action_ball_full_mdp_reset_genesis()
        self._action_ball_full_mdp_manager_construction_state = "sealed"

    def action_ball_full_mdp_construction_lease(self) -> object:
        """Return the env-owned lease only at the synchronous builder seam."""

        if self._action_ball_full_mdp_manager_construction_state != (
            "command_manager_ready"
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP construction lease is unavailable outside "
                "the runtime graph builder"
            )
        self._assert_action_ball_full_mdp_runtime_lease_identity()
        return self._action_ball_full_mdp_runtime_lease

    def _require_action_ball_full_mdp_install_phase(
        self, lease: object
    ) -> None:
        if self._action_ball_full_mdp_manager_construction_state != (
            "command_manager_ready"
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP construction install is outside the exact "
                "runtime graph builder phase"
            )
        self._assert_action_ball_full_mdp_runtime_lease_identity()
        if lease is not self._action_ball_full_mdp_runtime_lease:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP construction install received a foreign lease"
            )

    def _capture_action_ball_full_mdp_reset_genesis(self) -> None:
        install = getattr(
            self, "_action_ball_full_mdp_reset_genesis_install", None
        )
        if type(install) is not _FullMdpResetGenesisInstall:
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP construction has no reset genesis authority"
            )
        projector = getattr(
            install.authority,
            "require_owned_full_mdp_reset_genesis",
            None,
        )
        if not callable(projector):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset genesis projector is not callable"
            )
        projection = projector(
            install.receipt,
            device=torch.device(self.device),
            num_envs=self.num_envs,
        )
        if type(projection) is not FullMdpResetGenesisProjection:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset genesis projection type differs"
            )
        if projection.world_reset_identity is None:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset genesis world identity is absent"
            )
        generations = projection.reset_generations
        if (
            type(generations) is not torch.Tensor
            or generations.dtype != torch.int64
            or generations.ndim != 1
            or generations.shape != (self.num_envs,)
            or generations.device != torch.device(self.device)
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset genesis tensor differs from device int64[N]"
            )
        # The authority is independent of this environment ledger.  At this
        # cold boundary the consumer must reject a forged/exhausted projection;
        # the production issuer's fixed-one construction is not a substitute
        # for validating foreign authority implementations.
        if not bool(
            torch.all(
                torch.logical_and(
                    generations >= 1,
                    generations < torch.iinfo(torch.int64).max,
                )
            )
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset genesis has no positive int64 continuation"
            )
        self._action_ball_full_mdp_reset_generation = generations.clone()
        self._action_ball_full_mdp_world_reset_identity = (
            projection.world_reset_identity
        )
        self._action_ball_full_mdp_active_reset_record = None
        # The independent genesis is itself the canonical first all-env reset.
        # Consume that boundary at the first native reset callpoint; minting a
        # second selected-reset event there would advance env chronology past
        # the ActionEpoch genesis before any epoch exists.
        self._action_ball_full_mdp_lean_genesis_reset_pending = (
            type(self._action_ball_full_mdp_components)
            is FullMdpLeanRuntimeComponents
        )

    def install_action_ball_full_mdp_lean_runtime_graph(
        self,
        lease: object,
        *,
        genesis_authority: object,
        genesis_receipt: object,
        components: FullMdpLeanRuntimeComponents,
        reward_manager_cfg: dict[str, object],
        observation_source: object,
        observation_manager_cfg: dict[str, object],
        termination_manager_cfg: dict[str, object],
        live_physx_shutdown: object,
    ) -> None:
        """Publish the complete diagnostic graph in one construction mutation."""

        self._require_action_ball_full_mdp_install_phase(lease)
        if type(components) is not FullMdpLeanRuntimeComponents:
            raise FullMdpPostPhysicsProtocolError(
                "lean graph requires the exact component registry"
            )
        if genesis_authority is None or genesis_receipt is None:
            raise FullMdpPostPhysicsOwnerMissingError(
                "lean graph lacks its independent reset genesis"
            )
        try:
            lean_observations = importlib.import_module(
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_full_mdp_lean_observation_cfg"
            )
        except ImportError:  # pragma: no cover - focused direct imports.
            lean_observations = importlib.import_module(
                "action_ball_full_mdp_lean_observation_cfg"
            )
        source_type = lean_observations.LeanActionEpochObservationSource
        source_observe = getattr(observation_source, "observe", None)
        if (
            type(observation_source) is not source_type
            or observation_source is not components.observation_source
            or getattr(observation_source, "_env", None) is not self
            or getattr(observation_source, "_runtime_owner", None)
            is not components.lean_runtime_owner
            or getattr(observation_source, "_epoch_owner", None)
            is not components.epoch_owner
            or not callable(source_observe)
            or getattr(source_observe, "__self__", None) is not observation_source
            or getattr(source_observe, "__func__", None)
            is not source_type.observe
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean graph requires its exact env-bound observation source"
            )
        if (
            type(live_physx_shutdown) is not types.MethodType
            or live_physx_shutdown.__self__ is None
            or live_physx_shutdown.__func__
            is not getattr(
                type(live_physx_shutdown.__self__),
                "shutdown_action_epoch_live_physx_fact_owner",
                None,
            )
            or tuple(inspect.signature(live_physx_shutdown).parameters)
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean graph requires the exact bound live PhysX shutdown"
            )
        if (
            type(reward_manager_cfg) is not dict
            or tuple(reward_manager_cfg) != reward_contract.MANAGER_NAMES
            or type(observation_manager_cfg) is not dict
            or tuple(observation_manager_cfg) != ("policy", "critic")
            or type(termination_manager_cfg) is not dict
            or tuple(termination_manager_cfg)
            != (
                "time_out",
                "base_fell_tilt",
                "base_too_low",
                "joint_qdes_forbidden",
                "robot_hit_table",
            )
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean graph manager configs differ from exact Reward/Observation/Termination replacements"
            )
        components.reward_graph.configure_milestone_configured_income(
            reward_manager_cfg, self.step_dt
        )
        if (
            hasattr(self, "_action_ball_full_mdp_components")
            or hasattr(self, "_action_ball_full_mdp_reset_genesis_install")
            or hasattr(self, "_full_mdp_runtime_owner")
            or hasattr(self, "_action_ball_full_mdp_lean_observation_source")
            or hasattr(self, "_action_ball_full_mdp_live_physx_shutdown")
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean graph cannot replace a partial or complete install"
            )
        old_rewards = self.cfg.rewards
        old_observations = self.cfg.observations
        old_terminations = self.cfg.terminations
        published_names = (
            "_action_ball_full_mdp_reset_genesis_install",
            "_action_ball_full_mdp_components",
            "_action_ball_full_mdp_lean_reward_graph",
            "_action_ball_full_mdp_lean_observation_source",
            "_action_ball_full_mdp_live_physx_shutdown",
        )
        try:
            self.cfg.rewards = reward_manager_cfg
            self.cfg.observations = observation_manager_cfg
            self.cfg.terminations = termination_manager_cfg
            self.__dict__.update(
                {
                    published_names[0]: _FullMdpResetGenesisInstall(
                        authority=genesis_authority,
                        receipt=genesis_receipt,
                    ),
                    published_names[1]: components,
                    published_names[2]: components.reward_graph,
                    published_names[3]: observation_source,
                    published_names[4]: live_physx_shutdown,
                }
            )
        except BaseException as install_error:
            # A config descriptor is allowed to mutate and then raise.  Restore
            # every manager field independently so one hostile/failed setter
            # cannot prevent the other two restores or leave an env publication
            # behind.  Identity checks below distinguish an actual rollback
            # from a best-effort claim.
            for name, old_value in (
                ("rewards", old_rewards),
                ("observations", old_observations),
                ("terminations", old_terminations),
            ):
                try:
                    setattr(self.cfg, name, old_value)
                except BaseException:
                    pass
            for name in published_names:
                self.__dict__.pop(name, None)
            if (
                getattr(self.cfg, "rewards", _ABSENT) is not old_rewards
                or getattr(self.cfg, "observations", _ABSENT)
                is not old_observations
                or getattr(self.cfg, "terminations", _ABSENT)
                is not old_terminations
            ):
                raise FullMdpPostPhysicsProtocolError(
                    "lean graph manager-config rollback was incomplete"
                ) from install_error
            raise

    def _capture_action_ball_full_mdp_runtime_components(self) -> None:
        components = getattr(self, "_action_ball_full_mdp_components", None)
        if type(components) is not FullMdpLeanRuntimeComponents:
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP construction has no exact component registry"
            )
        try:
            motion_term = self.command_manager.get_term("motion")
            racket_term = self.command_manager.get_term("racket_target")
        except Exception as exc:
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP command owners are not installed"
            ) from exc
        if (
            motion_term is not components.motion_owner
            or racket_term is not components.racket_owner
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP command-manager identities differ from the registry"
            )

    def _action_ball_full_mdp_lean_observe_term(
        self, *, group: str
    ) -> torch.Tensor:
        """Resolve the privately installed source for one manager term call.

        Isaac config classes deep-copy term parameters, so a live source (and
        its retained environment/Kit graph) must never enter manager config.
        The term supplies only the code-owned group name; this resolver joins
        it to the exact source installed in the same atomic graph mutation.
        """

        if group not in ("policy", "critic"):
            raise FullMdpPostPhysicsProtocolError(
                "lean observation term group must be policy or critic"
            )
        if self._action_ball_full_mdp_manager_construction_state not in (
            "runtime_graph_ready",
            "base_managers_complete",
            "sealed",
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean observation term ran outside its installed graph lifecycle"
            )
        try:
            lean_observations = importlib.import_module(
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_full_mdp_lean_observation_cfg"
            )
        except ImportError:  # pragma: no cover - focused direct imports.
            lean_observations = importlib.import_module(
                "action_ball_full_mdp_lean_observation_cfg"
            )
        components = getattr(self, "_action_ball_full_mdp_components", _ABSENT)
        source = getattr(
            self, "_action_ball_full_mdp_lean_observation_source", _ABSENT
        )
        source_type = lean_observations.LeanActionEpochObservationSource
        observe = getattr(source, "observe", None)
        if (
            type(components) is not FullMdpLeanRuntimeComponents
            or type(source) is not source_type
            or source is not components.observation_source
            or getattr(source, "_env", None) is not self
            or getattr(source, "_runtime_owner", None)
            is not components.lean_runtime_owner
            or getattr(source, "_epoch_owner", None) is not components.epoch_owner
            or not callable(observe)
            or getattr(observe, "__self__", None) is not source
            or getattr(observe, "__func__", None) is not source_type.observe
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean observation source/runtime/component identity differs"
            )
        return observe(group)

    def _action_ball_full_mdp_lean_reward_term(
        self,
        *,
        ordinal: int,
        scale: float | None = None,
        value: torch.Tensor | None = None,
        paddle_playback_active: torch.Tensor | None = None,
        paddle_error_components: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pay one manager ordinal through the atomically installed graph.

        RewardManager configuration may contain only code-owned scalars.  The
        live graph therefore remains private to this environment and is joined
        to a term call only after the same lifecycle and component-identity
        checks used by the private observation resolver.
        """

        if type(ordinal) is not int:
            raise FullMdpPostPhysicsProtocolError(
                "lean Reward term ordinal must be one exact int"
            )
        if self._action_ball_full_mdp_manager_construction_state not in (
            "runtime_graph_ready",
            "base_managers_complete",
            "sealed",
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean Reward term ran outside its installed graph lifecycle"
            )
        try:
            lean_rewards = importlib.import_module(
                "whole_body_tracking.tasks.tracking.mdp."
                "action_ball_full_mdp_lean_rewards"
            )
        except ImportError:  # pragma: no cover - focused direct imports.
            lean_rewards = importlib.import_module(
                "action_ball_full_mdp_lean_rewards"
            )
        components = getattr(self, "_action_ball_full_mdp_components", _ABSENT)
        graph = getattr(
            self, "_action_ball_full_mdp_lean_reward_graph", _ABSENT
        )
        graph_type = lean_rewards.LeanActionEpochRewardGraph
        pay = getattr(graph, "pay", None)
        if (
            type(components) is not FullMdpLeanRuntimeComponents
            or type(graph) is not graph_type
            or graph is not components.reward_graph
            or graph.epoch_owner is not components.epoch_owner
            or not callable(pay)
            or getattr(pay, "__self__", None) is not graph
            or getattr(pay, "__func__", None) is not graph_type.pay
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean Reward graph/epoch/component identity differs"
            )
        if ordinal < lean_rewards.LIFECYCLE_PAYMENT_COUNT:
            if (
                value is not None
                or paddle_playback_active is not None
                or paddle_error_components is not None
            ):
                raise FullMdpPostPhysicsProtocolError(
                    "lifecycle Reward term cannot inject a dense value"
                )
            return graph_type.pay(graph, ordinal, scale=scale)
        if scale is not None or type(value) is not torch.Tensor:
            raise FullMdpPostPhysicsProtocolError(
                "common dense Reward term requires one tensor and no scale"
            )
        paddle_row = (
            lean_rewards.PADDLE_MOTION_PRIOR_FIRST_ORDINAL
            <= ordinal
            < lean_rewards.REGULARIZATION_FIRST_ORDINAL
        )
        if (
            paddle_playback_active is not None
            and type(paddle_playback_active) is not torch.Tensor
        ) or (
            paddle_error_components is not None
            and type(paddle_error_components) is not torch.Tensor
        ) or (
            not paddle_row
            and (
                paddle_playback_active is not None
                or paddle_error_components is not None
            )
        ):
            raise FullMdpPostPhysicsProtocolError(
                "dense Reward paddle telemetry binding differs"
            )
        return graph_type.record_common_dense(
            graph,
            ordinal,
            value,
            paddle_playback_active=paddle_playback_active,
            paddle_error_components=paddle_error_components,
        )

    def _require_action_ball_full_mdp_lease(self, lease: object) -> None:
        if self._action_ball_full_mdp_manager_construction_state != "sealed":
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP runtime lease is not sealed"
            )
        self._assert_action_ball_full_mdp_runtime_lease_identity()
        if lease is not getattr(
            self, "_action_ball_full_mdp_runtime_lease", _ABSENT
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP component getter received a foreign lease"
            )

    @property
    def action_ball_full_mdp_runtime_lease(self) -> object:
        if self._action_ball_full_mdp_manager_construction_state != "sealed":
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP runtime lease is not sealed"
            )
        self._assert_action_ball_full_mdp_runtime_lease_identity()
        lease = getattr(self, "_action_ball_full_mdp_runtime_lease", _ABSENT)
        if lease is _ABSENT:
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP runtime lease is not installed"
            )
        return lease

    def action_ball_full_mdp_ppo_drain_owner(self, lease: object) -> object:
        self._require_action_ball_full_mdp_lease(lease)
        components = self._action_ball_full_mdp_components
        if type(components) is not FullMdpLeanRuntimeComponents:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP drain lost the exact lean component registry"
            )
        return components.lean_runtime_owner

    def action_ball_full_mdp_lean_runtime_owner(
        self, lease: object
    ) -> object:
        """Return the exact owner installed by the diagnostic factory.

        This getter is intentionally sealed-only.  Its consumer is the
        code-owned ``create_from_env`` classmethod, which therefore cannot
        construct or substitute a second top owner.
        """

        self._require_action_ball_full_mdp_lease(lease)
        components = self._action_ball_full_mdp_components
        if type(components) is not FullMdpLeanRuntimeComponents:
            raise FullMdpPostPhysicsOwnerMissingError(
                "sealed environment has no exact lean runtime registry"
            )
        return components.lean_runtime_owner

    def action_ball_full_mdp_lean_reward_graph(
        self, lease: object
    ) -> object:
        """Return the sealed code-owned lean Reward graph identity."""

        self._require_action_ball_full_mdp_lease(lease)
        components = self._action_ball_full_mdp_components
        if type(components) is not FullMdpLeanRuntimeComponents:
            raise FullMdpPostPhysicsOwnerMissingError(
                "sealed environment has no exact lean Reward graph"
            )
        graph = self._action_ball_full_mdp_lean_reward_graph
        if graph is not components.reward_graph:
            raise FullMdpPostPhysicsProtocolError(
                "sealed lean Reward graph identity differs from components"
            )
        return graph

    def close(self) -> None:
        """Drain the live PhysX subscriber before IsaacLab scene teardown."""

        state_key = "_action_ball_full_mdp_base_construction_state"
        failed_key = "_action_ball_full_mdp_partial_close_failed_terminal_steps"
        guard_key = "_action_ball_full_mdp_partial_close_in_progress"
        base_state = self.__dict__.get(
            state_key, "not_started"
        )
        if base_state == "partial_close_failed":
            failed = self.__dict__.get(failed_key, ())
            raise FullMdpPostPhysicsProtocolError(
                "partial full-MDP simulator cleanup failed at "
                f"{','.join(failed) or '<unknown>'}; cold process exit is "
                "required and stale simulator resources will not be touched"
            )
        if base_state in ("not_started", "partial_close_complete"):
            return
        if base_state not in ("entered", "returned"):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP base-construction lifecycle state differs"
            )
        is_partial = base_state == "entered"
        if is_partial:
            if self.__dict__.get(guard_key, False):
                return
            self.__dict__[guard_key] = True
        try:
            shutdown = self.__dict__.pop(
                "_action_ball_full_mdp_live_physx_shutdown", None
            )
            shutdown_error: BaseException | None = None
            if shutdown is not None:
                try:
                    shutdown()
                except BaseException as exc:
                    shutdown_error = exc
            if not is_partial:
                try:
                    super().close()
                except BaseException as close_error:
                    if shutdown_error is not None:
                        raise close_error from shutdown_error
                    raise
                if shutdown_error is not None:
                    raise shutdown_error
                return

            cleanup_error = shutdown_error
            sim = self.__dict__.get("sim")
            # Mirror the pinned RL/base destructor order, skipping fields that
            # construction never reached.  The guard makes __del__ re-entry a
            # no-op without publishing a durable in-progress state.
            for name in (
                "command_manager", "reward_manager", "termination_manager",
                "curriculum_manager", "viewport_camera_controller",
                "action_manager", "observation_manager", "event_manager",
                "recorder_manager", "scene",
            ):
                if name not in self.__dict__:
                    continue
                try:
                    delattr(self, name)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            stage_methods = ()
            if sim is not None:
                sim_cfg = getattr(self.__dict__.get("cfg"), "sim", None)
                if getattr(sim_cfg, "create_stage_in_memory", False):
                    stage_methods = ("detach_stage", "stop", "clear")
                    try:
                        base_module = importlib.import_module(
                            "isaaclab.envs.manager_based_env"
                        )
                        if base_module.get_isaac_sim_version().major < 5:
                            stage_methods = ()
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
            terminal_steps = (
                *stage_methods,
                "clear_all_callbacks",
                "clear_instance",
            ) if sim is not None else ()
            failed_steps = []
            for method_name in terminal_steps:
                try:
                    if method_name == "detach_stage":
                        base_module = importlib.import_module(
                            "isaaclab.envs.manager_based_env"
                        )
                        base_module.omni.physx.get_physx_simulation_interface().detach_stage()
                    else:
                        getattr(sim, method_name)()
                except BaseException as exc:
                    failed_steps.append(method_name)
                    if cleanup_error is None:
                        cleanup_error = exc
            if failed_steps:
                self.__dict__[failed_key] = tuple(failed_steps)
                state = "partial_close_failed"
            else:
                self.__dict__.pop(failed_key, None)
                if "_window" in self.__dict__:
                    self.__dict__["_window"] = None
                state = "partial_close_complete"
            self.__dict__[state_key] = state
            self.__dict__["_is_closed"] = not failed_steps
            if cleanup_error is not None:
                raise cleanup_error
        finally:
            if is_partial:
                self.__dict__.pop(guard_key, None)

    @property
    def full_mdp_runtime_owner(self) -> object:
        owner = getattr(self, "_full_mdp_runtime_owner", _ABSENT)
        if owner is _ABSENT:
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP top runtime owner is not installed"
            )
        return owner

    @property
    def action_ball_r10_checkpoint_adapter(self) -> object | None:
        owner = self.full_mdp_runtime_owner
        return getattr(owner, "action_ball_r10_checkpoint_adapter", None)

    @property
    def action_ball_r10_cold_restore_capsule(self) -> None:
        """A normally constructed fresh env has no cold-restore capsule."""

        return None

    def _validate_lean_owner_install(
        self,
        owner: object,
        *,
        expected_lease: object,
    ) -> _LeanOwnerExecutableBinding:
        """Bind the diagnostic factory identity without treating a hash as truth."""

        try:
            module_object = importlib.import_module(
                FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE
            )
        except Exception as exc:
            raise FullMdpPostPhysicsOwnerMissingError(
                "lean diagnostic runtime owner module is unavailable"
            ) from exc
        owner_type = vars(module_object).get(
            FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_QUALNAME
        )
        components = getattr(self, "_action_ball_full_mdp_components", None)
        if (
            type(owner_type) is not type
            or owner_type.__module__
            != FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_MODULE
            or owner_type.__qualname__
            != FULL_MDP_DIAGNOSTIC_RUNTIME_OWNER_QUALNAME
            or type(owner) is not owner_type
            or type(components) is not FullMdpLeanRuntimeComponents
            or components.lean_runtime_owner is not owner
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean diagnostic owner is foreign, duplicated, or not factory-installed"
            )
        if (
            owner.full_mdp_runtime_env is not self
            or owner.full_mdp_runtime_lease is not expected_lease
            or owner.epoch_owner is not components.epoch_owner
            or owner.diagnostic_dependency_kind
            != FULL_MDP_DIAGNOSTIC_RUNTIME_DEPENDENCY_KIND
            or owner.diagnostic_unauthorized is not True
            or owner.launch_authorized is not False
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean diagnostic owner binding or authorization state differs"
            )
        expected_identities = (
            ("r05_runtime", components.device_r05_owner),
            ("motion", components.motion_owner),
            ("racket", components.racket_owner),
            ("physical_ball", components.physical_owner),
            ("r06_landing_outcome", components.r06_owner),
            ("r03_strike_fact", components.r03_owner),
            ("r07_recovery", components.r07_owner),
        )
        identities = owner.component_identities
        if (
            type(identities) is not tuple
            or len(identities) != len(expected_identities)
            or any(
                type(actual) is not tuple
                or len(actual) != 2
                or actual[0] != expected[0]
                or actual[1] is not expected[1]
                for actual, expected in zip(identities, expected_identities)
            )
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean diagnostic owner component identities differ"
            )
        namespace = vars(owner_type)
        functions: list[types.FunctionType] = []
        for name in (
            "before_policy_step",
            "before_physics_substep",
            "publish_post_physics_substep",
            "after_reward_close",
            "after_command_compute_before_observation",
            "selected_true_reset",
        ):
            function = namespace.get(name)
            bound = getattr(owner, name, None)
            if (
                not isinstance(function, types.FunctionType)
                or not isinstance(bound, types.MethodType)
                or bound.__self__ is not owner
                or bound.__func__ is not function
            ):
                raise FullMdpPostPhysicsProtocolError(
                    f"lean diagnostic owner {name!r} API is shadowed or foreign"
                )
            functions.append(function)
        return _LeanOwnerExecutableBinding(
            owner_type=owner_type,
            before_policy_step_function=functions[0],
            before_physics_substep_function=functions[1],
            publish_function=functions[2],
            after_reward_close_function=functions[3],
            after_command_compute_before_observation_function=functions[4],
            selected_true_reset_function=functions[5],
        )

    def _protected_manager_values(self) -> tuple[tuple[str, object], ...]:
        termination_manager = self.termination_manager
        values: list[tuple[str, object]] = [
            ("episode_length_buf", self.episode_length_buf),
            ("reset_buf", getattr(self, "reset_buf", _ABSENT)),
            (
                "reset_terminated",
                getattr(self, "reset_terminated", _ABSENT),
            ),
            ("reset_time_outs", getattr(self, "reset_time_outs", _ABSENT)),
            ("termination_manager.terminated", termination_manager.terminated),
            ("termination_manager.time_outs", termination_manager.time_outs),
        ]
        command_names = tuple(self.command_manager.active_terms)
        if len(command_names) != len(set(command_names)) or any(
            type(name) is not str for name in command_names
        ):
            raise FullMdpPostPhysicsProtocolError(
                "command manager active term names are not unique strings"
            )
        for name in command_names:
            term = self.command_manager.get_term(name)
            values.extend(
                (
                    (f"command.{name}.time_left", term.time_left),
                    (f"command.{name}.command_counter", term.command_counter),
                )
            )
        return tuple(values)

    def _protected_manager_state(self) -> _ProtectedManagerState:
        if type(self.common_step_counter) is not int:
            raise FullMdpPostPhysicsProtocolError(
                "common_step_counter stopped being a plain integer"
            )
        if type(self._sim_step_counter) is not int:
            raise FullMdpPostPhysicsProtocolError(
                "_sim_step_counter stopped being a plain integer"
            )
        return _ProtectedManagerState(
            common_step_counter=self.common_step_counter,
            sim_step_counter=self._sim_step_counter,
            values=tuple(
                _ProtectedValueSnapshot.capture(name, value)
                for name, value in self._protected_manager_values()
            ),
        )

    def _assert_protected_manager_state_unchanged(
        self, protected: _ProtectedManagerState
    ) -> None:
        if (
            self.common_step_counter != protected.common_step_counter
            or self._sim_step_counter != protected.sim_step_counter
        ):
            raise FullMdpPostPhysicsProtocolError(
                "post-physics owner mutated a protected manager clock"
            )
        current = self._protected_manager_values()
        if tuple(name for name, _ in current) != tuple(
            snapshot.name for snapshot in protected.values
        ):
            raise FullMdpPostPhysicsProtocolError(
                "post-physics owner changed the protected manager topology"
            )
        for snapshot, (_, value) in zip(protected.values, current):
            snapshot.assert_unchanged(value)

    def _poison(
        self,
        *,
        reason: str,
        exact_stamp: tuple[int, int, int, int, int] | None,
    ) -> None:
        if self._full_mdp_post_physics_poison is None:
            self._full_mdp_post_physics_poison = (reason, exact_stamp)

    def _assert_step_may_start(self) -> None:
        poison = self._full_mdp_post_physics_poison
        if poison is not None:
            stamp_text = (
                "none"
                if poison[1] is None
                else repr(poison[1])
            )
            raise FullMdpPostPhysicsPoisonedError(
                "fresh full-MDP env is poisoned; cold reconstruction is required: "
                f"reason={poison[0]}; stamp={stamp_text}"
            )
        if self._full_mdp_active_dispatch is not None:
            self._poison(reason="reentrant_step", exact_stamp=None)
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP env.step is not reentrant"
            )
        if not hasattr(self, "_full_mdp_runtime_owner"):
            self._poison(reason="missing_top_runtime_owner", exact_stamp=None)
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP first operation has no installed top runtime owner"
            )

    def _before_policy_step(
        self, *, control_step: int, action: torch.Tensor
    ) -> None:
        """Enter the one top scheduler; only it may decide reveal due/request."""

        if not hasattr(self, "_full_mdp_before_policy_step"):
            self._poison(reason="missing_before_policy_step", exact_stamp=None)
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP top owner has no before-policy callpoint"
            )
        try:
            result = self._full_mdp_before_policy_step(control_step, action)
            if result is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "top runtime owner before_policy_step must return None"
                )
            if self._full_mdp_post_physics_poison is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "top runtime owner poisoned the before-policy boundary"
                )
        except Exception as exc:
            self._poison(
                reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                exact_stamp=None,
            )
            if isinstance(exc, FullMdpPostPhysicsProtocolError):
                raise
            raise FullMdpPostPhysicsProtocolError(
                "top runtime owner before_policy_step failed; environment is poisoned"
            ) from exc
        except BaseException:
            self._poison(reason="base_exception", exact_stamp=None)
            raise

    def _publish_post_physics_substep(
        self,
        dispatch: _ControlStepDispatch,
        *,
        physics_substep: int,
    ) -> None:
        stamp: FullMdpPhysicsSubstepStamp | None = None
        expected_exact: tuple[int, int, int, int, int] | None = None
        try:
            stamp = dispatch.prepare(
                physics_substep=physics_substep,
                sim_step=self._sim_step_counter,
            )
            expected_exact = dispatch.pending_exact_tuple
            protected = self._protected_manager_state()
            result = self._full_mdp_post_physics_publish(stamp)
            if result is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "post-physics owner must return None"
                )
            self._assert_protected_manager_state_unchanged(protected)
            if self._full_mdp_post_physics_poison is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "post-physics owner attempted a reentrant or poisoned step"
                )
            dispatch.commit(stamp)
        except Exception as exc:
            self._poison(
                reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                exact_stamp=expected_exact,
            )
            if isinstance(exc, FullMdpPostPhysicsProtocolError):
                raise
            raise FullMdpPostPhysicsProtocolError(
                "post-physics owner failed; environment is poisoned"
            ) from exc
        except BaseException:
            self._poison(reason="base_exception", exact_stamp=expected_exact)
            raise

    def _after_reward_close(self, *, control_step: int) -> None:
        """Run the exact top-owned close immediately after RewardManager.

        Reward-cycle existence, payment ordering and close receipts belong to
        the construction-bound top/graph implementation.  This env seam owns
        only executable identity, once-per-control-step sequencing and sticky
        fail-stop behavior; it never fabricates a receipt or treats an absent
        cycle as success.
        """

        try:
            if type(control_step) is not int or control_step < 1:
                raise FullMdpPostPhysicsProtocolError(
                    "after-Reward control_step must be a positive plain integer"
                )
            last = getattr(
                self,
                "_full_mdp_last_after_reward_close_control_step",
                None,
            )
            if type(last) is not int or control_step != last + 1:
                raise FullMdpPostPhysicsProtocolError(
                    "after-Reward close was skipped, duplicated, or replayed"
                )
            result = self._full_mdp_after_reward_close(control_step)
            if result is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "top runtime owner after_reward_close must return None"
                )
            if self._full_mdp_post_physics_poison is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "top runtime owner poisoned the after-Reward boundary"
                )
            self._full_mdp_last_after_reward_close_control_step = control_step
        except Exception as exc:
            self._poison(
                reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                exact_stamp=None,
            )
            if isinstance(exc, FullMdpPostPhysicsProtocolError):
                raise
            raise FullMdpPostPhysicsProtocolError(
                "top runtime owner after_reward_close failed; environment is poisoned"
            ) from exc
        except BaseException:
            self._poison(reason="base_exception", exact_stamp=None)
            raise

    def _reset_non_action_ball_commands(self, env_ids: torch.Tensor) -> dict:
        """Reset unrelated command terms without legacy Motion/Racket sampling."""

        components = getattr(self, "_action_ball_full_mdp_components", None)
        if type(components) is not FullMdpLeanRuntimeComponents:
            raise FullMdpPostPhysicsOwnerMissingError(
                "fresh full-MDP reset lost its component registry"
            )
        names = tuple(self.command_manager.active_terms)
        if len(names) != len(set(names)) or any(type(name) is not str for name in names):
            raise FullMdpPostPhysicsProtocolError(
                "command manager active term names are not unique strings"
            )
        excluded = (components.motion_owner, components.racket_owner)
        extras: dict[str, object] = {}
        for name in names:
            term = self.command_manager.get_term(name)
            if term is excluded[0] or term is excluded[1]:
                continue
            metrics = term.reset(env_ids=env_ids)
            if type(metrics) is not dict:
                raise FullMdpPostPhysicsProtocolError(
                    f"command term {name!r} reset metrics must be one exact dict"
                )
            for metric_name, metric_value in metrics.items():
                if type(metric_name) is not str:
                    raise FullMdpPostPhysicsProtocolError(
                        "command reset metric names must be exact strings"
                    )
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        return extras

    def _mint_action_ball_full_mdp_selected_reset_event(
        self,
        env_ids: torch.Tensor,
    ) -> FullMdpSelectedResetEvent:
        callpoint = getattr(
            self, "_action_ball_full_mdp_reset_callpoint_authority", None
        )
        if (
            type(callpoint) is not _FullMdpResetCallpointAuthority
            or callpoint.env_ids is not env_ids
            or callpoint.source not in ("step_nonzero", "reset_all_arange")
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP selected reset lacks an owner-issued callpoint authority"
            )
        self._action_ball_full_mdp_reset_callpoint_authority = None
        if getattr(
            self, "_action_ball_full_mdp_active_reset_record", _ABSENT
        ) is not None:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP already has one active selected-reset event"
            )
        generations = getattr(
            self, "_action_ball_full_mdp_reset_generation", None
        )
        if (
            type(generations) is not torch.Tensor
            or generations.dtype != torch.int64
            or generations.shape != (self.num_envs,)
            or generations.device != torch.device(self.device)
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP live reset-generation ledger is unavailable"
            )
        if env_ids.shape[0] < 1:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP selected reset cannot be empty"
            )
        # Production supplies ``reset_buf.nonzero`` (automatic reset) or
        # ``arange`` (whole-environment reset), both sorted/unique/in-range.
        # Preserve that producer contract.  Device-R05/top owns the independent
        # mask/index equivalence check before any owner commit; a delayed CUDA
        # assertion here would not be a same-batch safety barrier.
        index = env_ids.clone()
        selected_mask = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=torch.device(self.device)
        )
        selected_mask.index_fill_(0, index, True)
        generation_before = generations.clone()
        generation_overflow_fault = torch.logical_and(
            selected_mask,
            generation_before == torch.iinfo(torch.int64).max,
        )
        safe_increment = torch.logical_and(
            selected_mask, ~generation_overflow_fault
        ).to(torch.int64)
        generation_after = generation_before + safe_increment
        termination_manager = self.termination_manager
        reason_names = tuple(
            name for name, _bit in _FULL_MDP_TERMINATION_REASON_BITS
        )
        cached_reasons = getattr(termination_manager, "_term_dones", None)
        if (
            tuple(termination_manager.active_terms) != reason_names
            or type(cached_reasons) is not torch.Tensor
            or cached_reasons.dtype != torch.bool
            or cached_reasons.shape != (self.num_envs, len(reason_names))
            or cached_reasons.device != torch.device(self.device)
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP terminal reset reason order differs"
            )
        pure_timeout = getattr(self, "reset_time_outs", None)
        if (
            type(pure_timeout) is not torch.Tensor
            or pure_timeout.dtype != torch.bool
            or pure_timeout.shape != (self.num_envs,)
            or pure_timeout.device != torch.device(self.device)
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP canonical pure-timeout tensor differs"
            )
        reason_bits = torch.zeros(
            (self.num_envs,),
            dtype=torch.int64,
            device=torch.device(self.device),
        )
        for column, (term_name, bit) in enumerate(
            _FULL_MDP_TERMINATION_REASON_BITS
        ):
            # ``TerminationManager.compute`` has already produced this exact
            # fixed buffer.  Read its retained result once; do not call a term
            # function or query the manager again during reset or PPO drain.
            reason = (
                pure_timeout
                if term_name == "time_out"
                else cached_reasons[:, column]
            )
            if (
                type(reason) is not torch.Tensor
                or reason.dtype != torch.bool
                or reason.shape != (self.num_envs,)
                or reason.device != torch.device(self.device)
            ):
                raise FullMdpPostPhysicsProtocolError(
                    "fresh full-MDP terminal reset reason tensor differs"
                )
            reason_bits.bitwise_or_(
                reason.to(dtype=torch.int64) * int(bit)
            )
        episode_tick = self.episode_length_buf
        if (
            type(self.common_step_counter) is not int
            or self.common_step_counter < 1
            or type(episode_tick) is not torch.Tensor
            or episode_tick.dtype != torch.int64
            or episode_tick.shape != (self.num_envs,)
            or episode_tick.device != torch.device(self.device)
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP terminal reset clock differs"
            )
        unselected_i64 = torch.full_like(episode_tick, -1)
        selected_common_step = torch.full_like(
            episode_tick, self.common_step_counter
        )
        terminal_reset_facts_i64 = torch.stack(
            (
                torch.where(
                    selected_mask, selected_common_step, unselected_i64
                ),
                torch.where(selected_mask, episode_tick, unselected_i64),
                torch.where(
                    selected_mask, reason_bits, torch.zeros_like(reason_bits)
                ),
            ),
            dim=1,
        ).contiguous()
        event = FullMdpSelectedResetEvent()
        record = _FullMdpSelectedResetRecord(
            event=event,
            reset_event_identity=object(),
            selected_env_index=index,
            selected_mask=selected_mask,
            generation_before=generation_before,
            generation_after=generation_after,
            generation_overflow_fault=generation_overflow_fault,
            terminal_reset_facts_i64=terminal_reset_facts_i64,
        )
        self._action_ball_full_mdp_active_reset_record = record
        return event

    def _authorize_action_ball_full_mdp_reset_callpoint(
        self,
        env_ids: torch.Tensor,
        *,
        source: str,
    ) -> None:
        if type(source) is not str or source not in (
            "step_nonzero",
            "reset_all_arange",
        ):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset callpoint source differs"
            )
        if getattr(
            self, "_action_ball_full_mdp_reset_callpoint_authority", None
        ) is not None:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset callpoint authority is already active"
            )
        self._action_ball_full_mdp_reset_callpoint_authority = (
            _FullMdpResetCallpointAuthority(env_ids=env_ids, source=source)
        )

    def _project_action_ball_full_mdp_lean_selected_reset_event(
        self,
        event: FullMdpSelectedResetEvent,
    ) -> FullMdpSelectedResetProjection:
        """Project the env-owned reset facts to the installed lean top once."""

        record = getattr(
            self, "_action_ball_full_mdp_active_reset_record", None
        )
        components = getattr(self, "_action_ball_full_mdp_components", None)
        if (
            type(components) is not FullMdpLeanRuntimeComponents
            or type(record) is not _FullMdpSelectedResetRecord
            or record.event is not event
            or record.projected
        ):
            raise FullMdpPostPhysicsProtocolError(
                "lean selected-reset event is stale, foreign, or replayed"
            )
        record.projected = True
        return FullMdpSelectedResetProjection(
            reset_event_identity=record.reset_event_identity,
            selected_env_index=record.selected_env_index.clone(),
            selected_mask=record.selected_mask.clone(),
            generation_before=record.generation_before.clone(),
            generation_after=record.generation_after.clone(),
            generation_overflow_fault=(
                record.generation_overflow_fault.clone()
            ),
            terminal_reset_facts_i64=(
                record.terminal_reset_facts_i64.clone()
            ),
        )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Commit one opaque top reset, then the pinned native bookkeeping."""

        if not isinstance(env_ids, torch.Tensor):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP selected env_ids must be one tensor"
            )
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP selected env_ids must be one-dimensional int64"
            )
        if str(env_ids.device) != str(self.device):
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP selected env_ids must remain on the env device"
            )
        if not hasattr(self, "_full_mdp_selected_true_reset"):
            self._poison(reason="missing_selected_true_reset", exact_stamp=None)
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP top owner has no selected-reset callpoint"
            )
        try:
            components = self._action_ball_full_mdp_components
            if type(components) is not FullMdpLeanRuntimeComponents:
                raise FullMdpPostPhysicsProtocolError(
                    "fresh full-MDP reset lost the exact lean component registry"
                )
            genesis_reset = getattr(
                self,
                "_action_ball_full_mdp_lean_genesis_reset_pending",
                False,
            )
            if genesis_reset:
                callpoint = getattr(
                    self,
                    "_action_ball_full_mdp_reset_callpoint_authority",
                    None,
                )
                if (
                    type(callpoint) is not _FullMdpResetCallpointAuthority
                    or callpoint.env_ids is not env_ids
                    or callpoint.source != "reset_all_arange"
                    or env_ids.shape != (self.num_envs,)
                ):
                    raise FullMdpPostPhysicsProtocolError(
                        "lean genesis reset requires the first canonical all-env callpoint"
                    )
                self._action_ball_full_mdp_reset_callpoint_authority = None
                self._action_ball_full_mdp_lean_genesis_reset_pending = False
            else:
                event = self._mint_action_ball_full_mdp_selected_reset_event(
                    env_ids
                )
            if not genesis_reset:
                projection = (
                    self._project_action_ball_full_mdp_lean_selected_reset_event(
                        event
                    )
                )
                result = self._full_mdp_selected_true_reset(event, projection)
                if result is not None:
                    raise FullMdpPostPhysicsProtocolError(
                        "lean selected_true_reset must return None"
                    )
                record = self._action_ball_full_mdp_active_reset_record
                if (
                    type(record) is not _FullMdpSelectedResetRecord
                    or record.event is not event
                    or not record.projected
                ):
                    raise FullMdpPostPhysicsProtocolError(
                        "lean selected-reset record changed during settlement"
                    )
                self._action_ball_full_mdp_reset_generation.copy_(
                    record.generation_after
                )
                self._action_ball_full_mdp_active_reset_record = None
            # Exact IsaacLab 2.3.2 native order, except that the two fresh
            # ActionBall command owners have already reset under the top owner.
            self.curriculum_manager.compute(env_ids=env_ids)
            self.scene.reset(env_ids)
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(
                    mode="reset",
                    env_ids=env_ids,
                    global_env_step_count=env_step_count,
                )
            self.extras["log"] = dict()
            info = self.observation_manager.reset(env_ids)
            self.extras["log"].update(info)
            info = self.action_manager.reset(env_ids)
            self.extras["log"].update(info)
            info = self.reward_manager.reset(env_ids)
            self.extras["log"].update(info)
            info = self.curriculum_manager.reset(env_ids)
            self.extras["log"].update(info)
            info = self._reset_non_action_ball_commands(env_ids)
            self.extras["log"].update(info)
            info = self.event_manager.reset(env_ids)
            self.extras["log"].update(info)
            info = self.termination_manager.reset(env_ids)
            self.extras["log"].update(info)
            info = self.recorder_manager.reset(env_ids)
            self.extras["log"].update(info)
            self.episode_length_buf[env_ids] = 0
            if not genesis_reset:
                if type(record) is not _FullMdpSelectedResetRecord:
                    raise FullMdpPostPhysicsProtocolError(
                        "completed lean selected-reset record differs"
                    )
                terminal_facts = record.terminal_reset_facts_i64
                components.epoch_owner.milestone.close_episodes(
                    record.selected_mask,
                    terminal_facts[:, 1],
                    terminal_facts[:, 2],
                )
            if genesis_reset:
                # IsaacLab's reset path does not compute commands before its
                # first ObservationManager pass.  Produce that one canonical
                # initial command epoch here, after every manager reset and
                # before the base reset computes its returned observation.
                self.command_manager.compute(dt=self.step_dt)
                result = (
                    self._full_mdp_after_command_compute_before_observation(0)
                )
                if result is not None:
                    raise FullMdpPostPhysicsProtocolError(
                        "lean genesis after-command boundary must return None"
                    )
        except Exception as exc:
            self._poison(
                reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                exact_stamp=None,
            )
            if isinstance(exc, FullMdpPostPhysicsProtocolError):
                raise
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP selected reset failed; environment is poisoned"
            ) from exc
        except BaseException:
            self._poison(reason="base_exception", exact_stamp=None)
            raise

    def reset(
        self,
        seed: int | None = None,
        env_ids: object = None,
        options: dict | None = None,
    ) -> tuple[object, dict]:
        """Authorize only the canonical whole-environment reset callpoint.

        Fresh full-MDP partial/manual reset has no independent selection
        authority and is rejected.  Automatic selected resets are exclusively
        minted by this class's pinned ``step`` body from ``reset_buf.nonzero``.
        """

        self._assert_step_may_start()

        if env_ids is not None:
            raise FullMdpUnsupportedRuntimeError(
                "fresh full-MDP manual partial reset is unsupported"
            )
        if options is not None and type(options) is not dict:
            raise FullMdpPostPhysicsProtocolError(
                "fresh full-MDP reset options must be an exact dict or None"
            )
        canonical_env_ids = torch.arange(
            self.num_envs,
            dtype=torch.int64,
            device=torch.device(self.device),
        )
        self._authorize_action_ball_full_mdp_reset_callpoint(
            canonical_env_ids,
            source="reset_all_arange",
        )
        return super().reset(
            seed=seed,
            env_ids=canonical_env_ids,
            options=options,
        )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Run the pinned IsaacLab step through the one top runtime owner."""

        self._assert_step_may_start()
        dispatch: _ControlStepDispatch | None = None
        try:
            action_on_device = action.to(self.device)
            self._before_policy_step(
                control_step=self.common_step_counter + 1,
                action=action_on_device,
            )
            # process actions
            self.action_manager.process_action(action_on_device)

            self.recorder_manager.record_pre_step()

            # check if we need to do rendering within the physics loop
            # note: checked here once to avoid multiple checks within the loop
            is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

            decimation = self.cfg.decimation
            dispatch = _ControlStepDispatch(
                control_step=self.common_step_counter + 1,
                decimation=decimation,
                sim_step_before=self._sim_step_counter,
            )
            self._full_mdp_active_dispatch = dispatch
            try:
                # perform physics stepping
                for physics_substep in range(decimation):
                    self._sim_step_counter += 1
                    # set actions into buffers
                    self.action_manager.apply_action()
                    before_physics = self._full_mdp_before_physics_substep
                    protected = self._protected_manager_state()
                    pre_stamp = FullMdpPrePhysicsSubstepStamp(
                        control_step=dispatch.control_step,
                        physics_substep=physics_substep,
                        physics_substeps_per_control=dispatch.decimation,
                        sim_step_before=self._sim_step_counter - 1,
                    )
                    result = before_physics(pre_stamp)
                    if result is not None:
                        raise FullMdpPostPhysicsProtocolError(
                            "lean pre-physics boundary must return None"
                        )
                    self._assert_protected_manager_state_unchanged(protected)
                    # set actions into simulator
                    self.scene.write_data_to_sim()
                    # simulate
                    self.sim.step(render=False)
                    self.recorder_manager.record_post_physics_decimation_step()
                    # render between steps only if the GUI or an RTX sensor needs it
                    # note: we assume the render interval to be the shortest accepted rendering interval.
                    #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                    if (
                        self._sim_step_counter % self.cfg.sim.render_interval
                        == 0
                        and is_rendering
                    ):
                        self.sim.render()
                    # update buffers at sim dt
                    self.scene.update(dt=self.physics_dt)
                    # publish same-transition physical facts before any manager clock.
                    self._publish_post_physics_substep(
                        dispatch, physics_substep=physics_substep
                    )
                dispatch.finish()
            finally:
                self._full_mdp_active_dispatch = None

            if self._full_mdp_post_physics_poison is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "post-physics dispatch poisoned the current step"
                )

            # post-step:
            # -- update env counters (used for curriculum generation)
            self.episode_length_buf += 1  # step in current episode (per env)
            self.common_step_counter += 1  # total step (common for all envs)
            # -- check terminations
            self.reset_buf = self.termination_manager.compute()
            self.reset_terminated = self.termination_manager.terminated
            raw_time_outs = self.termination_manager.time_outs
            # A horizon/plant overlap is a real terminal transition, not a
            # truncation that RSL may bootstrap through.  Keep the manager's
            # raw timeout tensor intact for telemetry; publish only the
            # disjoint Gymnasium/RSL timeout partition to learning consumers.
            pure_timeout = torch.logical_and(
                raw_time_outs, ~self.reset_terminated
            )
            self.reset_time_outs = pure_timeout
            # -- reward computation
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            components = self._action_ball_full_mdp_components
            if type(components) is not FullMdpLeanRuntimeComponents:
                raise FullMdpPostPhysicsProtocolError(
                    "fresh full-MDP step lost the exact lean component registry"
                )
            components.reward_graph.close_milestone_actual_reward(
                self.reward_buf
            )
            components.epoch_owner.milestone.add_step_return(self.reward_buf)
            self._after_reward_close(control_step=self.common_step_counter)

            if len(self.recorder_manager.active_terms) > 0:
                # update observations for recording if needed
                self.obs_buf = self.observation_manager.compute()
                self.recorder_manager.record_post_step()

            # -- reset envs that terminated/timed-out and log the episode information
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._authorize_action_ball_full_mdp_reset_callpoint(
                    reset_env_ids,
                    source="step_nonzero",
                )
                # trigger recorder terms for pre-reset calls
                self.recorder_manager.record_pre_reset(reset_env_ids)

                self._reset_idx(reset_env_ids)
                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if (
                    self.sim.has_rtx_sensors()
                    and self.cfg.num_rerenders_on_reset > 0
                ):
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()

                # trigger recorder terms for post-reset calls
                self.recorder_manager.record_post_reset(reset_env_ids)

            # -- update command
            self.command_manager.compute(dt=self.step_dt)
            after_command = (
                self._full_mdp_after_command_compute_before_observation
            )
            result = after_command(self.common_step_counter)
            if result is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "lean after-command boundary must return None"
                )
            if self._full_mdp_post_physics_poison is not None:
                raise FullMdpPostPhysicsProtocolError(
                    "lean after-command boundary poisoned the environment"
                )
            # -- step interval events
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
            # -- compute observations
            # note: done after reset to get the correct observations for reset envs
            self.obs_buf = self.observation_manager.compute(update_history=True)

            # return observations, rewards, resets and extras
            return (
                self.obs_buf,
                self.reward_buf,
                self.reset_terminated,
                self.reset_time_outs,
                self.extras,
            )
        except BaseException as exc:
            exact_stamp = (
                None
                if dispatch is None
                else dispatch.last_committed_exact_tuple
            )
            self._poison(
                reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                exact_stamp=exact_stamp,
            )
            raise


# Capture the executable local surface exactly once, after the class body has
# completed.  These are process-local object identities, not evidence derived
# from re-reading or hashing the same source that defined them.  Capturing every
# direct function and property getter also makes newly added direct callpoints
# protected without maintaining a second hand-written allowlist.
_PINNED_LOCAL_FULL_MDP_MODULE = sys.modules[__name__]
_PINNED_LOCAL_FULL_MDP_ENV_CLASS = ActionBallFullMdpManagerBasedRLEnv
_PINNED_LOCAL_FULL_MDP_DIRECT_METHODS = tuple(
    (
        name,
        value,
        value.__code__,
        _cold_local_plain_callable_defaults(value),
    )
    for name, value in vars(ActionBallFullMdpManagerBasedRLEnv).items()
    if type(value) is types.FunctionType
)
_PINNED_LOCAL_FULL_MDP_PROPERTY_GETTERS = tuple(
    (
        name,
        descriptor,
        descriptor.fget,
        descriptor.fget.__code__,
        _cold_local_plain_callable_defaults(descriptor.fget),
    )
    for name, descriptor in vars(ActionBallFullMdpManagerBasedRLEnv).items()
    if type(descriptor) is property
    and type(descriptor.fget) is types.FunctionType
)
