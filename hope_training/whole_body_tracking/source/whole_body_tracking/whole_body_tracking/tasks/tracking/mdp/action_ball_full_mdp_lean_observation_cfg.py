"""Compact ObservationManager ABI for the diagnostic ActionEpoch runtime.

One direct runtime publication supplies real robot proprioception, current
action and Motion teacher state.  This manager independently gathers the
canonical task, clocks, lifecycle, Physical/R03/R06/R07 facts and Reward ledger
from the current fixed-slot ActionEpoch record.  There is no legacy observation
prefix, capacity padding, receipt adapter, or source digest.

Isaac Lab invokes each term once while constructing ObservationManager.  Those
two calls are explicitly shape-only and may return zeros.  Every term config
contains only its code-owned group name; the call resolves the exact source
atomically retained by the environment instead of embedding a live Kit object
in config state.  Every later call requires the direct runtime publication and
finite semantic tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Mapping

import torch

try:
    from . import action_ball_full_mdp_epoch as epoch_v1
    from . import action_ball_full_mdp_lean_runtime as lean_runtime
except ImportError:  # Focused source-file tests avoid the Isaac package tree.
    import action_ball_full_mdp_epoch as epoch_v1
    import action_ball_full_mdp_lean_runtime as lean_runtime


DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

DIRECT_VIEW_METHOD = "action_epoch_observation_v1"
ENV_TERM_METHOD = "_action_ball_full_mdp_lean_observe_term"
MANAGER_GROUP_ORDER = ("policy", "critic")
_CONSTRUCTION_STATE = "runtime_graph_ready"
ACTOR_CONTRACT_V1 = "action_ball_full_mdp_action_epoch_v1"
CRITIC_CONTRACT_V1 = "action_ball_full_mdp_action_epoch_critic_v1"
OBSERVATION_KIND_V1 = "action_ball_full_mdp_action_epoch_observation_v1"

# This is a deliberately small, named diagnostic ABI for the exact 31-DoF A3
# plant and current 31-wide action.  Widths are derived only from these fields.
# A future robot or observation change edits the named layout, never a magic
# total inherited from the superseded R08 contract.
ACTOR_LAYOUT_V1 = (
    ("projected_gravity_b", 3),
    ("base_ang_vel_b", 3),
    ("joint_pos_rel", 31),
    ("joint_vel_rel", 31),
    ("last_action", 31),
    ("teacher_joint_pos_rel", 31),
    ("teacher_joint_vel_rel", 31),
    ("motion_phase_one_hot", 5),
    ("epoch_task_f32", epoch_v1.TASK_F32_WIDTH),
    ("epoch_clock_remaining_s", 5),
    ("epoch_phase_one_hot", 10),
    ("epoch_task_valid", 1),
    ("epoch_selected", 1),
    # Preserve the frozen one-bit actor ABI with a public lifecycle fact.
    # Construction admissibility is D05-private and is not a published shot fact.
    ("epoch_launch_succeeded", 1),
)

CRITIC_EXTENSION_LAYOUT_V1 = (
    ("physical_r03_r06_r07_fact_present", 4),
    ("physical_r03_r06_r07_fact_age_s", 4),
    (
        "physical_r03_r06_r07_fact_f32",
        4 * epoch_v1.OWNER_FACT_F32_WIDTH,
    ),
    ("physical_r03_r06_r07_fault_present", 4),
    ("reward_cycle_open", 1),
    ("reward_cycle_fault_present", 1),
    ("reward_due", epoch_v1.REWARD_CONSUMER_COUNT),
    ("reward_paid", epoch_v1.REWARD_CONSUMER_COUNT),
)

ACTOR_WIDTH_V1 = sum(width for _, width in ACTOR_LAYOUT_V1)
CRITIC_WIDTH_V1 = ACTOR_WIDTH_V1 + sum(
    width for _, width in CRITIC_EXTENSION_LAYOUT_V1
)

_DIRECT_FIELD_LAYOUT = ACTOR_LAYOUT_V1[:7]
_FACT_OWNER_KINDS = (
    "physical_ball",
    "r03_strike_fact",
    "r06_landing_outcome",
    "r07_recovery",
)
_REQUIRED_COMPONENTS = (
    "motion",
    "racket",
    *_FACT_OWNER_KINDS,
)


class LeanObservationError(RuntimeError):
    """The direct observation source is malformed, stale, or out of order."""


class LeanObservationConstructionHold(LeanObservationError):
    """A real provider or exact runtime source required before Manager is absent."""


@dataclass(frozen=True)
class DirectActionEpochObservationFacts:
    """Real non-epoch facts published by the construction-bound lean owner."""

    projected_gravity_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    joint_pos_rel: torch.Tensor
    joint_vel_rel: torch.Tensor
    last_action: torch.Tensor
    teacher_joint_pos_rel: torch.Tensor
    teacher_joint_vel_rel: torch.Tensor
    motion_phase_code: torch.Tensor
    current_task_slot: torch.Tensor
    current_action_uid: torch.Tensor
    current_rng_counter: torch.Tensor
    transaction_epoch: int
    transaction_version: int
    common_step: int
    motion_owner: object
    racket_owner: object
    physical_owner: object
    r03_owner: object
    r06_owner: object
    r07_owner: object
    diagnostic_unauthorized: bool = True


def _exact_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.device != device
        or value.dtype != dtype
    ):
        raise LeanObservationError(
            f"{label} must be {list(shape)} {dtype} on {device}"
        )
    return value


def _assert_async_all(predicate: torch.Tensor, *, label: str) -> None:
    """Schedule a device assertion without a per-step tensor-to-host decode."""

    condition = torch.all(predicate)
    async_assert = getattr(torch, "_assert_async", None)
    if callable(async_assert):
        async_assert(condition, label)
    else:  # pragma: no cover - supported Torch builds provide _assert_async.
        torch._assert(condition, label)


def _gather_selected(value: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    index = slot.reshape(slot.shape[0], 1, *([1] * (value.ndim - 2)))
    index = index.expand(slot.shape[0], 1, *value.shape[2:])
    return torch.gather(value, 1, index).squeeze(1)


def build_direct_action_epoch_observation_facts(
    *,
    runtime_owner: lean_runtime.ActionBallFullMdpLeanRuntimeOwner,
    record: epoch_v1.ActionEpochRecord,
) -> DirectActionEpochObservationFacts:
    """Read the exact Isaac/Motion buffers for the lean owner's direct method.

    ``ActionBallFullMdpLeanRuntimeOwner.action_epoch_observation_v1`` delegates
    here.  Keeping extraction beside the named layout prevents that top owner
    from guessing Motion private storage or duplicating the 31-DoF contract.
    No legacy observation function, token, receipt or manager cache is read.
    """

    if type(runtime_owner) is not lean_runtime.ActionBallFullMdpLeanRuntimeOwner:
        raise LeanObservationError("direct fact builder requires exact lean owner")
    if type(record) is not epoch_v1.ActionEpochRecord:
        raise LeanObservationError("direct fact builder requires exact epoch record")
    env = runtime_owner.full_mdp_runtime_env
    epoch_owner = runtime_owner.epoch_owner
    if type(epoch_owner) is not epoch_v1.ActionEpochOwner:
        raise LeanObservationError("direct fact builder lost exact epoch owner")
    current = epoch_owner.current()
    if current.epoch != record.epoch or current.version != record.version:
        raise LeanObservationError("direct fact builder received stale epoch clone")

    components = runtime_owner.component_identities
    if type(components) is not tuple:
        raise LeanObservationError("direct fact builder component surface differs")
    by_name = dict(components)
    try:
        from .commands import MotionCommand
    except ImportError:  # Runtime may import this module from the flat source root.
        from commands import MotionCommand
    motion = by_name.get("motion")
    if type(motion) is not MotionCommand:
        raise LeanObservationError("direct fact builder requires exact MotionCommand")
    robot = getattr(motion, "robot", None)
    data = getattr(robot, "data", None)
    action_manager = getattr(env, "action_manager", None)
    action = getattr(action_manager, "action", None)
    phase = getattr(motion, "_action_ball_continuous_canonical_phase", None)
    common_step = getattr(env, "common_step_counter", None)
    if type(common_step) is not int or common_step < 0:
        raise LeanObservationError("direct fact builder common step differs")

    dtype = getattr(by_name.get("r06_landing_outcome"), "dtype", None)
    device = epoch_owner.device
    n = epoch_owner.num_envs
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise LeanObservationError("direct fact builder R06 dtype differs")

    def exact(name: str, value: object, width: int) -> torch.Tensor:
        result = _exact_tensor(
            value,
            label="live " + name,
            shape=(n, width),
            device=device,
            dtype=dtype,
        )
        _assert_async_all(torch.isfinite(result), label=name + " is nonfinite")
        return result.detach().clone()

    projected_gravity = exact(
        "projected_gravity_b", getattr(data, "projected_gravity_b", None), 3
    )
    base_ang_vel = exact(
        "base_ang_vel_b", getattr(data, "root_ang_vel_b", None), 3
    )
    joint_pos = exact("joint_pos", getattr(data, "joint_pos", None), 31)
    default_joint_pos = exact(
        "default_joint_pos", getattr(data, "default_joint_pos", None), 31
    )
    joint_vel = exact("joint_vel", getattr(data, "joint_vel", None), 31)
    teacher_joint_pos = exact("teacher_joint_pos", motion.joint_pos, 31)
    teacher_joint_vel = exact("teacher_joint_vel", motion.joint_vel, 31)
    last_action = exact("last_action", action, 31)
    motion_phase = _exact_tensor(
        phase,
        label="live Motion canonical phase",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    _assert_async_all(
        (motion_phase >= 0) & (motion_phase < 5),
        label="Motion canonical phase is outside 0..4",
    )
    slot = record.current_task_slot.detach().clone()
    row = slot[:, None]
    return DirectActionEpochObservationFacts(
        projected_gravity_b=projected_gravity,
        base_ang_vel_b=base_ang_vel,
        joint_pos_rel=joint_pos - default_joint_pos,
        joint_vel_rel=joint_vel,
        last_action=last_action,
        teacher_joint_pos_rel=teacher_joint_pos - default_joint_pos,
        teacher_joint_vel_rel=teacher_joint_vel,
        motion_phase_code=motion_phase.detach().clone(),
        current_task_slot=slot,
        current_action_uid=torch.gather(record.action_uid, 1, row)
        .squeeze(1)
        .detach()
        .clone(),
        current_rng_counter=torch.gather(record.rng_counter, 1, row)
        .squeeze(1)
        .detach()
        .clone(),
        transaction_epoch=record.epoch,
        transaction_version=record.version,
        common_step=common_step,
        motion_owner=motion,
        racket_owner=by_name.get("racket"),
        physical_owner=by_name.get("physical_ball"),
        r03_owner=by_name.get("r03_strike_fact"),
        r06_owner=by_name.get("r06_landing_outcome"),
        r07_owner=by_name.get("r07_recovery"),
    )


class LeanActionEpochObservationSource:
    """Construction-bound, two-group observation source with one-step cache."""

    def __init__(
        self,
        *,
        env: object,
        runtime_owner: lean_runtime.ActionBallFullMdpLeanRuntimeOwner,
    ) -> None:
        if type(runtime_owner) is not lean_runtime.ActionBallFullMdpLeanRuntimeOwner:
            raise LeanObservationConstructionHold(
                "observation source requires the exact lean runtime owner"
            )
        if runtime_owner.full_mdp_runtime_env is not env:
            raise LeanObservationConstructionHold(
                "lean runtime owner belongs to another environment"
            )
        epoch_owner = runtime_owner.epoch_owner
        if type(epoch_owner) is not epoch_v1.ActionEpochOwner:
            raise LeanObservationConstructionHold(
                "lean runtime owner lost its exact ActionEpoch owner"
            )
        method = getattr(runtime_owner, DIRECT_VIEW_METHOD, None)
        if (
            not callable(method)
            or getattr(method, "__self__", None) is not runtime_owner
            or getattr(method, "__func__", None)
            is not vars(lean_runtime.ActionBallFullMdpLeanRuntimeOwner).get(
                DIRECT_VIEW_METHOD
            )
        ):
            raise LeanObservationConstructionHold(
                "lean runtime owner has no exact " + DIRECT_VIEW_METHOD
            )

        components = runtime_owner.component_identities
        if type(components) is not tuple or any(
            type(row) is not tuple or len(row) != 2 for row in components
        ):
            raise LeanObservationConstructionHold(
                "lean runtime component identity surface differs"
            )
        by_name = dict(components)
        missing = tuple(
            name for name in _REQUIRED_COMPONENTS if by_name.get(name) is None
        )
        if missing:
            raise LeanObservationConstructionHold(
                "direct observation providers are absent: " + ",".join(missing)
            )
        r06 = by_name["r06_landing_outcome"]
        dtype = getattr(r06, "dtype", None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise LeanObservationConstructionHold(
                "R06 dtype must be an explicit floating torch dtype"
            )
        if getattr(r06, "device", None) != epoch_owner.device:
            raise LeanObservationConstructionHold(
                "R06 and ActionEpoch devices differ"
            )
        if getattr(r06, "num_envs", None) != epoch_owner.num_envs:
            raise LeanObservationConstructionHold(
                "R06 and ActionEpoch environment counts differ"
            )
        step_dt = getattr(env, "step_dt", None)
        if isinstance(step_dt, bool) or type(step_dt) not in (int, float):
            raise LeanObservationConstructionHold(
                "environment step_dt must be an explicit host number"
            )
        step_dt = float(step_dt)
        if not 0.0 < step_dt < float("inf"):
            raise LeanObservationConstructionHold(
                "environment step_dt must be finite and positive"
            )

        self._env = env
        self._runtime_owner = runtime_owner
        self._epoch_owner = epoch_owner
        self._components = {
            "motion_owner": by_name["motion"],
            "racket_owner": by_name["racket"],
            "physical_owner": by_name["physical_ball"],
            "r03_owner": by_name["r03_strike_fact"],
            "r06_owner": r06,
            "r07_owner": by_name["r07_recovery"],
        }
        self._num_envs = epoch_owner.num_envs
        self._device = epoch_owner.device
        self._dtype = dtype
        self._step_dt = step_dt
        self._widths = {"policy": ACTOR_WIDTH_V1, "critic": CRITIC_WIDTH_V1}
        self._shape_probed: set[str] = set()
        self._cached_key: tuple[int, int, int] | None = None
        self._cached_actor: torch.Tensor | None = None
        self._cached_critic: torch.Tensor | None = None
        self._semantic_publications = 0

    @property
    def group_widths(self) -> Mapping[str, int]:
        return dict(self._widths)

    @property
    def semantic_publication_count(self) -> int:
        return self._semantic_publications

    def _shape_probe(self, group: str) -> torch.Tensor:
        state = getattr(
            self._env, "_action_ball_full_mdp_manager_construction_state", None
        )
        if state != _CONSTRUCTION_STATE or group in self._shape_probed:
            raise LeanObservationError(
                "shape-only observation probe is unavailable outside first manager construction"
            )
        self._shape_probed.add(group)
        return torch.zeros(
            (self._num_envs, self._widths[group]),
            dtype=self._dtype,
            device=self._device,
        )

    def _validate_direct(
        self,
        view: object,
        *,
        record: epoch_v1.ActionEpochRecord,
        common_step: int,
    ) -> DirectActionEpochObservationFacts:
        if type(view) is not DirectActionEpochObservationFacts:
            raise LeanObservationError("direct runtime observation fact type differs")
        if view.diagnostic_unauthorized is not True:
            raise LeanObservationError(
                "diagnostic observation cannot claim runtime authorization"
            )
        if (
            view.transaction_epoch != record.epoch
            or view.transaction_version != record.version
            or view.common_step != common_step
        ):
            raise LeanObservationError("direct runtime observation chronology is stale")
        for name, expected in self._components.items():
            if getattr(view, name) is not expected:
                raise LeanObservationError(
                    "direct runtime observation provider identity differs: " + name
                )
        for name, width in _DIRECT_FIELD_LAYOUT:
            tensor = _exact_tensor(
                getattr(view, name),
                label="direct " + name,
                shape=(self._num_envs, width),
                device=self._device,
                dtype=self._dtype,
            )
            _assert_async_all(torch.isfinite(tensor), label=name + " is nonfinite")
        phase = _exact_tensor(
            view.motion_phase_code,
            label="direct motion_phase_code",
            shape=(self._num_envs,),
            device=self._device,
            dtype=torch.int64,
        )
        _assert_async_all(
            (phase >= 0) & (phase < 5), label="Motion phase is outside 0..4"
        )
        slot = _exact_tensor(
            view.current_task_slot,
            label="direct current_task_slot",
            shape=(self._num_envs,),
            device=self._device,
            dtype=torch.int64,
        )
        _assert_async_all(
            (slot >= 0) & (slot < self._epoch_owner.shot_slot_capacity),
            label="direct current_task_slot is outside epoch capacity",
        )
        epoch_slot = record.current_task_slot
        row = epoch_slot[:, None]
        epoch_uid = torch.gather(record.action_uid, 1, row).squeeze(1)
        epoch_rng = torch.gather(record.rng_counter, 1, row).squeeze(1)
        uid = _exact_tensor(
            view.current_action_uid,
            label="direct current_action_uid",
            shape=(self._num_envs,),
            device=self._device,
            dtype=torch.int64,
        )
        rng = _exact_tensor(
            view.current_rng_counter,
            label="direct current_rng_counter",
            shape=(self._num_envs,),
            device=self._device,
            dtype=torch.int64,
        )
        _assert_async_all(slot == epoch_slot, label="current task slot differs")
        _assert_async_all(uid == epoch_uid, label="current action UID differs")
        _assert_async_all(rng == epoch_rng, label="current RNG counter differs")
        return view

    def _pack(
        self,
        view: DirectActionEpochObservationFacts,
        record: epoch_v1.ActionEpochRecord,
        common_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot = record.current_task_slot
        task = _gather_selected(record.task.task_f32, slot)
        task_valid = _gather_selected(record.task.task_valid, slot)
        phase = _gather_selected(record.phase, slot)
        selected = _gather_selected(record.selected_mask, slot)
        launch_succeeded = _gather_selected(record.launch_succeeded, slot)
        clock_rows = torch.stack(
            tuple(
                _gather_selected(getattr(record.clocks, name), slot)
                for name in (
                    "reveal_tick",
                    "contact_tick",
                    "launch_tick",
                    "deadline_tick",
                    "next_reveal_tick",
                )
            ),
            dim=1,
        )
        clock_remaining = (
            clock_rows - common_step
        ).to(dtype=self._dtype) * self._step_dt
        phase_one_hot = torch.nn.functional.one_hot(
            torch.clamp(phase, min=0, max=9), num_classes=10
        ).to(dtype=self._dtype)
        motion_phase = torch.nn.functional.one_hot(
            view.motion_phase_code, num_classes=5
        ).to(dtype=self._dtype)
        actor_parts = [
            getattr(view, name) for name, _ in _DIRECT_FIELD_LAYOUT
        ]
        actor_parts.extend(
            (
                motion_phase,
                task,
                clock_remaining,
                phase_one_hot,
                task_valid[:, None].to(dtype=self._dtype),
                selected[:, None].to(dtype=self._dtype),
                launch_succeeded[:, None].to(dtype=self._dtype),
            )
        )
        actor = torch.cat(actor_parts, dim=1)

        owner_indices = torch.tensor(
            tuple(epoch_v1.OWNER_ORDER.index(name) for name in _FACT_OWNER_KINDS),
            dtype=torch.int64,
            device=self._device,
        )
        selected_valid_bits = _gather_selected(record.fact_valid_bits, slot)
        selected_source_step = _gather_selected(record.fact_source_step, slot)
        selected_fact_f32 = _gather_selected(record.fact_f32, slot)
        selected_fault_bits = _gather_selected(record.owner_fault_bits, slot)
        valid_bits = selected_valid_bits.index_select(1, owner_indices)
        source_step = selected_source_step.index_select(1, owner_indices)
        facts = selected_fact_f32.index_select(1, owner_indices)
        faults = selected_fault_bits.index_select(1, owner_indices)
        present = valid_bits.ne(0)
        fact_age = torch.where(
            present,
            torch.clamp(common_step - source_step, min=0).to(self._dtype)
            * self._step_dt,
            torch.zeros_like(source_step, dtype=self._dtype),
        )
        facts = torch.where(
            present[:, :, None], facts, torch.zeros_like(facts)
        ).reshape(self._num_envs, -1)
        critic_extension = torch.cat(
            (
                present.to(self._dtype),
                fact_age,
                facts,
                faults.ne(0).to(self._dtype),
                record.reward_cycle_open[:, None].to(self._dtype),
                record.reward_cycle_fault.ne(0)[:, None].to(self._dtype),
                record.reward_due.to(self._dtype),
                record.reward_paid.to(self._dtype),
            ),
            dim=1,
        )
        critic = torch.cat((actor, critic_extension), dim=1)
        if actor.shape != (self._num_envs, ACTOR_WIDTH_V1):
            raise LeanObservationError("named actor layout width differs")
        if critic.shape != (self._num_envs, CRITIC_WIDTH_V1):
            raise LeanObservationError("named critic layout width differs")
        _assert_async_all(torch.isfinite(actor), label="packed actor is nonfinite")
        _assert_async_all(torch.isfinite(critic), label="packed critic is nonfinite")
        return actor.detach().clone(), critic.detach().clone()

    def _semantic(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shape_probed != set(MANAGER_GROUP_ORDER):
            raise LeanObservationError(
                "runtime observation ran before both manager shape probes"
            )
        if self._epoch_owner.poisoned:
            raise LeanObservationError("ActionEpoch owner is poisoned")
        common_step = getattr(self._env, "common_step_counter", None)
        if type(common_step) is not int or common_step < 0:
            raise LeanObservationError("environment common_step_counter differs")
        record = self._epoch_owner.current()
        key = (common_step, record.epoch, record.version)
        if (
            self._cached_key == key
            and self._cached_actor is not None
            and self._cached_critic is not None
        ):
            return self._cached_actor, self._cached_critic
        view = getattr(self._runtime_owner, DIRECT_VIEW_METHOD)(record)
        checked = self._validate_direct(
            view, record=record, common_step=common_step
        )
        actor, critic = self._pack(checked, record, common_step)
        self._cached_key = key
        self._cached_actor = actor
        self._cached_critic = critic
        self._semantic_publications += 1
        return actor, critic

    def observe(self, group: str) -> torch.Tensor:
        if group not in MANAGER_GROUP_ORDER:
            raise LeanObservationError("observation group must be policy or critic")
        if group not in self._shape_probed:
            return self._shape_probe(group)
        actor, critic = self._semantic()
        value = actor if group == "policy" else critic
        return value.detach().clone()


@dataclass(frozen=True)
class DiagnosticN2ObservationManagerBundle:
    """Off-side source plus deepcopy-safe manager configuration.

    The bundle itself deliberately retains the live source for the one atomic
    environment installation.  Only ``manager_cfg`` is assigned to the Isaac
    config tree, and therefore no live environment/runtime/Kit object is
    reachable from an ``ObservationTermCfg.params`` mapping.
    """

    source: LeanActionEpochObservationSource
    manager_cfg: dict[str, object]


def _term(
    env: object,
    *,
    group: str,
) -> torch.Tensor:
    if type(group) is not str or group not in MANAGER_GROUP_ORDER:
        raise LeanObservationError("observation group must be policy or critic")
    try:
        instance_state = vars(env)
    except TypeError as exc:
        raise LeanObservationConstructionHold(
            "ObservationManager env has no exact resolver state"
        ) from exc
    if ENV_TERM_METHOD in instance_state:
        raise LeanObservationConstructionHold(
            "ObservationManager env resolver is shadowed on the instance"
        )
    env_type = type(env)
    direct = getattr(env_type, ENV_TERM_METHOD, None)
    binder = getattr(direct, "__get__", None)
    if not callable(direct) or not callable(binder):
        raise LeanObservationConstructionHold(
            "ObservationManager env has no exact lean source resolver"
        )
    bound = binder(env, env_type)
    if (
        not callable(bound)
        or getattr(bound, "__self__", None) is not env
        or getattr(bound, "__func__", None) is not direct
    ):
        raise LeanObservationConstructionHold(
            "ObservationManager env lean source resolver binding differs"
        )
    return bound(group=group)


def materialize_observation_manager_cfg(
    *,
    env: object,
    runtime_owner: lean_runtime.ActionBallFullMdpLeanRuntimeOwner,
) -> DiagnosticN2ObservationManagerBundle:
    """Return the live source and deepcopy-safe ObservationManager config."""

    source = LeanActionEpochObservationSource(env=env, runtime_owner=runtime_owner)
    try:
        managers = importlib.import_module("isaaclab.managers")
        group_type = managers.ObservationGroupCfg
        term_type = managers.ObservationTermCfg
    except Exception as exc:
        raise LeanObservationConstructionHold(
            "Isaac ObservationManager config surface is unavailable"
        ) from exc

    result: dict[str, object] = {}
    for group_name in MANAGER_GROUP_ORDER:
        group = group_type()
        group.concatenate_terms = True
        group.enable_corruption = False
        group.history_length = None
        group.flatten_history_dim = True
        group.action_epoch = term_type(
            func=_term,
            params={"group": group_name},
        )
        result[group_name] = group
    return DiagnosticN2ObservationManagerBundle(
        source=source,
        manager_cfg=result,
    )


def installed_observation_facts(env: object) -> dict[str, object]:
    """Authenticate the live manager/source and return its small public ABI.

    The diagnostic observation implementation owns this check.  Keeping it
    beside ``LeanActionEpochObservationSource`` avoids routing the one-term
    ActionEpoch layout through the legacy 8k-line training-contract registry.
    """

    try:
        lease = env.action_ball_full_mdp_runtime_lease
        owner = env.full_mdp_runtime_owner
    except AttributeError as exc:
        raise LeanObservationError(
            "installed observation lacks its runtime owner or lease"
        ) from exc
    getter = getattr(env, "action_ball_full_mdp_lean_runtime_owner", None)
    getter_descriptor = getattr(
        type(env), "action_ball_full_mdp_lean_runtime_owner", None
    )
    epoch_owner = getattr(owner, "epoch_owner", None)
    if (
        lease is None
        or not callable(getter)
        or getattr(getter, "__self__", None) is not env
        or getattr(getter, "__func__", None) is not getter_descriptor
        or getter(lease) is not owner
        or type(owner) is not lean_runtime.ActionBallFullMdpLeanRuntimeOwner
        or owner.full_mdp_runtime_env is not env
        or owner.full_mdp_runtime_lease is not lease
        or owner.diagnostic_unauthorized is not True
        or owner.launch_authorized is not False
        or owner.poisoned
        or type(epoch_owner) is not epoch_v1.ActionEpochOwner
        or epoch_owner.poisoned
    ):
        raise LeanObservationError(
            "installed observation owner/env/lease/epoch join differs"
        )

    manager = getattr(env, "observation_manager", None)
    expected_active = {
        "policy": ["action_epoch"],
        "critic": ["action_epoch"],
    }
    cfgs = getattr(manager, "_group_obs_term_cfgs", None)
    if (
        getattr(manager, "active_terms", None) != expected_active
        or getattr(manager, "group_obs_term_dim", None)
        != {"policy": [(ACTOR_WIDTH_V1,)], "critic": [(CRITIC_WIDTH_V1,)]}
        or getattr(manager, "group_obs_dim", None)
        != {"policy": (ACTOR_WIDTH_V1,), "critic": (CRITIC_WIDTH_V1,)}
        or getattr(manager, "group_obs_concatenate", None)
        != {"policy": True, "critic": True}
        or type(cfgs) is not dict
        or tuple(cfgs) != MANAGER_GROUP_ORDER
        or any(type(cfgs[group]) is not list or len(cfgs[group]) != 1 for group in MANAGER_GROUP_ORDER)
    ):
        raise LeanObservationError(
            "installed ActionEpoch ObservationManager layout differs"
        )
    for group in MANAGER_GROUP_ORDER:
        cfg = cfgs[group][0]
        if (
            getattr(cfg, "func", None) is not _term
            or getattr(cfg, "params", None) != {"group": group}
            or getattr(cfg, "noise", None) is not None
            or getattr(cfg, "history_length", None) != 0
        ):
            raise LeanObservationError(
                "installed ActionEpoch observation callpoint differs"
            )

    env_state = vars(env)
    direct_resolver = getattr(type(env), ENV_TERM_METHOD, None)
    resolver = getattr(env, ENV_TERM_METHOD, None)
    installed = getattr(env, "_action_ball_full_mdp_components", None)
    source = getattr(env, "_action_ball_full_mdp_lean_observation_source", None)
    components = owner.component_identities
    if (
        ENV_TERM_METHOD in env_state
        or not callable(direct_resolver)
        or not callable(resolver)
        or getattr(resolver, "__self__", None) is not env
        or getattr(resolver, "__func__", None) is not direct_resolver
        or source is not getattr(installed, "observation_source", None)
        or getattr(installed, "lean_runtime_owner", None) is not owner
        or getattr(installed, "epoch_owner", None) is not epoch_owner
        or type(source) is not LeanActionEpochObservationSource
        or source._env is not env
        or source._runtime_owner is not owner
        or source._epoch_owner is not epoch_owner
        or source._num_envs != epoch_owner.num_envs
        or source._device != epoch_owner.device
        or source.group_widths
        != {"policy": ACTOR_WIDTH_V1, "critic": CRITIC_WIDTH_V1}
        or type(components) is not tuple
        or any(type(row) is not tuple or len(row) != 2 for row in components)
    ):
        raise LeanObservationError(
            "installed ActionEpoch observation resolver/source differs"
        )
    by_name = dict(components)
    expected_sources = {
        "motion_owner": "motion",
        "racket_owner": "racket",
        "physical_owner": "physical_ball",
        "r03_owner": "r03_strike_fact",
        "r06_owner": "r06_landing_outcome",
        "r07_owner": "r07_recovery",
    }
    if (
        type(source._components) is not dict
        or tuple(source._components) != tuple(expected_sources)
        or any(
            source._components[source_name] is not by_name.get(component_name)
            for source_name, component_name in expected_sources.items()
        )
    ):
        raise LeanObservationError(
            "installed ActionEpoch observation component identities differ"
        )
    return {
        "actor_obs_contract": ACTOR_CONTRACT_V1,
        "actor_obs_mode": "action_ball_full_mdp",
        "actor_obs_total_dim": ACTOR_WIDTH_V1,
        "actor_obs_term_names": ["action_epoch"],
        "actor_obs_term_dims": [ACTOR_WIDTH_V1],
        "critic_obs_contract": CRITIC_CONTRACT_V1,
        "critic_obs_total_dim": CRITIC_WIDTH_V1,
        "critic_obs_term_names": ["action_epoch"],
        "critic_obs_term_dims": [CRITIC_WIDTH_V1],
        "fresh_full_mdp_observation_kind": OBSERVATION_KIND_V1,
        "fresh_full_mdp_diagnostic_unauthorized": True,
        "fresh_full_mdp_launch_authorized": False,
        "fresh_full_mdp_no_capacity_receipt_or_sha_authority": True,
    }


__all__ = [
    "DIAGNOSTIC_UNAUTHORIZED",
    "RUNTIME_INTEGRATED",
    "LAUNCH_AUTHORIZED",
    "DIRECT_VIEW_METHOD",
    "ENV_TERM_METHOD",
    "MANAGER_GROUP_ORDER",
    "ACTOR_CONTRACT_V1",
    "CRITIC_CONTRACT_V1",
    "OBSERVATION_KIND_V1",
    "ACTOR_LAYOUT_V1",
    "CRITIC_EXTENSION_LAYOUT_V1",
    "ACTOR_WIDTH_V1",
    "CRITIC_WIDTH_V1",
    "DirectActionEpochObservationFacts",
    "build_direct_action_epoch_observation_facts",
    "LeanObservationError",
    "LeanObservationConstructionHold",
    "LeanActionEpochObservationSource",
    "DiagnosticN2ObservationManagerBundle",
    "materialize_observation_manager_cfg",
    "installed_observation_facts",
]
