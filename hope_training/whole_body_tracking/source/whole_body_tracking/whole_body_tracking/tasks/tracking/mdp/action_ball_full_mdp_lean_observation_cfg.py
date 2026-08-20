"""Semantic ObservationManager ABI for the diagnostic ActionEpoch runtime.

One direct runtime publication supplies the robot, Motion, Racket, Physical,
R06 and R07 state that can change the next transition.  Raw task packets,
owner fact rows, fault bits and Reward ledgers are deliberately not policy or
critic inputs.  Infrastructure faults fail-stop instead of becoming features.

Isaac Lab invokes each term once while constructing ObservationManager.  Those
two calls are explicitly shape-only and may return zeros.  Every term config
contains only its code-owned group name; the call resolves the exact source
atomically retained by the environment instead of embedding a live Kit object
in config state.  Every later call requires the direct runtime publication and
finite semantic tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
import importlib
from typing import Mapping

import torch

import action_ball_full_mdp_row_identity as row_identity

try:
    from . import action_ball_full_mdp_epoch as epoch_v1
    from . import action_ball_full_mdp_lean_runtime as lean_runtime
    from . import action_ball_full_mdp_portable_observation as portable_observation
except ImportError:  # Focused source-file tests avoid the Isaac package tree.
    import action_ball_full_mdp_epoch as epoch_v1
    import action_ball_full_mdp_lean_runtime as lean_runtime
    import action_ball_full_mdp_portable_observation as portable_observation


DIAGNOSTIC_UNAUTHORIZED = True
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False

DIRECT_VIEW_METHOD = "semantic_action_epoch_observation_v2"
ENV_TERM_METHOD = "_action_ball_full_mdp_lean_observe_term"
MANAGER_GROUP_ORDER = ("policy", "critic")
_CONSTRUCTION_STATE = "runtime_graph_ready"
ACTOR_CONTRACT_V2 = portable_observation.ACTOR_CONTRACT_V2
CRITIC_CONTRACT_V2 = portable_observation.CRITIC_CONTRACT_V2
OBSERVATION_KIND_V2 = portable_observation.OBSERVATION_KIND_V2

# This is a deliberately small, named diagnostic ABI for the exact 31-DoF A3
# plant and current 31-wide action.  Widths are derived only from these fields.
# A future robot or observation change edits the named layout, never a magic
# total inherited from the superseded R08 contract.
ACTOR_LAYOUT_V2 = portable_observation.ACTOR_LAYOUT_V2
CRITIC_EXTENSION_LAYOUT_V2 = portable_observation.CRITIC_EXTENSION_LAYOUT_V2
ACTOR_WIDTH_V2 = portable_observation.ACTOR_WIDTH_V2
CRITIC_WIDTH_V2 = portable_observation.CRITIC_WIDTH_V2

_DIRECT_FLOAT_LAYOUT = tuple(
    (name, width)
    for name, width in ACTOR_LAYOUT_V2 + CRITIC_EXTENSION_LAYOUT_V2
    if name
    not in (
        "motion_phase_one_hot",
        "epoch_learning_phase_one_hot",
        "task_valid",
    )
)
_REQUIRED_COMPONENTS = (
    "motion",
    "racket",
    "physical_ball",
    "r06_landing_outcome",
    "r07_recovery",
)


class LeanObservationError(RuntimeError):
    """The direct observation source is malformed, stale, or out of order."""


class LeanObservationConstructionHold(LeanObservationError):
    """A real provider or exact runtime source required before Manager is absent."""


@dataclass(frozen=True)
class DirectActionEpochObservationFacts:
    """Real semantic tensors published by the construction-bound lean owner."""

    actor_rows: Mapping[str, torch.Tensor]
    critic_rows: Mapping[str, torch.Tensor]
    motion_phase_code: torch.Tensor
    task_valid: torch.Tensor
    transaction_epoch: int
    transaction_version: int
    common_step: int
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
        async_assert(condition)
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
    """Project the bound Isaac owners into the semantic 203/219 ABI."""

    if type(runtime_owner) is not lean_runtime.ActionBallFullMdpLeanRuntimeOwner:
        raise LeanObservationError("direct fact builder requires exact lean owner")
    epoch_owner = runtime_owner.epoch_owner
    if type(record) is not epoch_v1.ActionEpochRecord:
        raise LeanObservationError("direct fact builder requires exact ActionEpoch record")
    env = runtime_owner.full_mdp_runtime_env
    parts = dict(runtime_owner.component_identities)
    try:
        from .commands import MotionCommand
        from whole_body_tracking.tasks.table_tennis import geometry as table_geometry
    except ImportError:
        from commands import MotionCommand
        from whole_body_tracking.tasks.table_tennis import geometry as table_geometry

    motion = parts["motion"]
    if type(motion) is not MotionCommand:
        raise LeanObservationError("direct fact builder requires exact MotionCommand")
    racket, physical = parts["racket"], parts["physical_ball"]
    r06, r07 = parts["r06_landing_outcome"], parts["r07_recovery"]
    data, device = motion.robot.data, epoch_owner.device
    dtype, n = r06.dtype, epoch_owner.num_envs
    step = env.common_step_counter
    motion_view = motion.require_owned_action_ball_continuous_motion_observation(
        motion.action_ball_continuous_motion_observation_projection()
    )
    if motion_view.motion_owner is not motion or motion_view.common_step != step:
        raise LeanObservationError("Motion observation chronology differs")

    root_pos_w, root_quat_w = data.root_pos_w, data.root_quat_w
    root_pos_env = root_pos_w - env.scene.env_origins
    # The installed ActionBall scene fixes every table to env-local/world axes;
    # env origins translate but never rotate it.  A rotatable table therefore
    # requires a new real pose producer rather than reinterpretation here.
    table_center = root_pos_w.new_tensor(
        (
            float(racket.cfg.vb_table_near_x)
            + 0.5 * float(table_geometry.TABLE_LENGTH),
            0.0,
            float(racket.cfg.vb_table_surface_z),
        )
    )
    anchor_pos, anchor_ori = portable_observation.relative_pose_6d(
        motion.robot_anchor_pos_w,
        motion.robot_anchor_quat_w,
        motion.anchor_pos_w,
        motion.anchor_quat_w,
    )
    task_valid = motion_view.task_valid
    task_mask = task_valid[:, None]
    base_heading = portable_observation.heading_xy_from_quat_wxyz(root_quat_w)

    def heading(value: torch.Tensor) -> torch.Tensor:
        return portable_observation.rotate_world_to_heading_xy(base_heading, value)

    def task_vector(value: torch.Tensor) -> torch.Tensor:
        value = heading(value)
        return torch.where(task_mask, value, torch.zeros_like(value))

    # FullMDP questions and R03 rewards both use the raw mount +Y (A-frame)
    # normal.  The per-action mount sign selects the physical rubber/contact
    # identity; multiplying it into this residual would reverse the learning
    # direction for negative-sign actions.
    actual_normal = racket.racket_normal_raw_w
    actor_target_pos_w = racket.actor_racket_target_pos_w()
    actor_target_vel_w = racket.actor_racket_target_vel_w()
    actor_target_normal_w = racket.actor_target_normal_cmd()
    base_goal_delta = torch.cat(
        (
            racket.base_target_pos_w - root_pos_w[:, :2],
            torch.zeros_like(root_pos_w[:, :1]),
        ),
        dim=1,
    )
    actor_rows = {
        "projected_gravity_b": data.projected_gravity_b,
        "base_ang_vel_b": data.root_ang_vel_b,
        "base_position_table": root_pos_env - table_center,
        "base_heading_table_xy": base_heading,
        "base_com_lin_vel_heading": heading(data.root_lin_vel_w),
        "joint_pos_rel": data.joint_pos - data.default_joint_pos,
        "joint_vel": data.joint_vel,
        "last_action": env.action_manager.action,
        "teacher_joint_pos_rel": motion.joint_pos - data.default_joint_pos,
        "teacher_joint_vel": motion.joint_vel,
        "motion_anchor_pos_b": anchor_pos,
        "motion_anchor_ori_b6": anchor_ori,
        "racket_target_pos_error_heading": task_vector(
            actor_target_pos_w - racket.racket_pos_w
        ),
        "racket_target_vel_error_heading": task_vector(
            actor_target_vel_w - racket.racket_lin_vel_w
        ),
        "racket_target_normal_error_heading": task_vector(
            actor_target_normal_w - actual_normal
        ),
        "base_goal_error_heading_xy": task_vector(base_goal_delta)[:, :2],
        "time_to_contact_s": torch.where(
            task_mask,
            motion_view.time_to_contact_remaining_s[:, None],
            torch.zeros_like(task_mask, dtype=dtype),
        ),
        "time_to_teacher_start_s": torch.where(
            task_mask,
            motion_view.time_to_teacher_start_remaining_s[:, None],
            torch.zeros_like(task_mask, dtype=dtype),
        ),
        "time_to_next_opportunity_s": motion_view.time_to_next_reveal_s[:, None],
    }

    slot = record.current_task_slot
    current_key = row_identity.ActionEpochShotKey(
        **{
            field.name: _gather_selected(
                getattr(record.identity.shot_key, field.name), slot
            )
            for field in dataclass_fields(row_identity.ActionEpochShotKey)
        }
    )
    current_publication = _gather_selected(record.publication_ordinal, slot)
    r06_projection = r06.action_ball_full_mdp_observation_projection()
    r06_view = r06.require_owned_action_epoch_current_flight_observation(
        r06_projection,
        current_shot_key=current_key,
        current_publication_ordinal=current_publication,
    )
    if (
        r06_view.r06_owner is not r06
        or r06_view.publication_identity is not r06_projection
    ):
        raise LeanObservationError("R06 current-flight observation owner differs")
    flight_slot = _exact_tensor(
        r06_view.flight_slot,
        label="R06 current flight_slot",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    contact = _exact_tensor(
        r06_view.contact_valid,
        label="R06 current contact_valid",
        shape=(n,),
        device=device,
        dtype=torch.bool,
    )
    crossed = _exact_tensor(
        r06_view.net_crossed,
        label="R06 current net_crossed",
        shape=(n,),
        device=device,
        dtype=torch.bool,
    )
    clear = _exact_tensor(
        r06_view.net_clear,
        label="R06 current net_clear",
        shape=(n,),
        device=device,
        dtype=torch.bool,
    )
    live = flight_slot.ge(0)
    _assert_async_all(
        live | (~contact & ~crossed & ~clear),
        label="absent R06 current flight has retained latches",
    )
    _assert_async_all(~clear | crossed, label="R06 net-clear lacks crossing")
    _assert_async_all(~crossed | contact, label="R06 net crossing lacks contact")

    scene_state = physical.scene_port.read_state_env()
    flight_capacity = getattr(r06, "flight_slot_capacity", None)
    if (
        type(scene_state) is not torch.Tensor
        or scene_state.ndim != 3
        or scene_state.shape[0] != n
        or type(flight_capacity) is not int
        or flight_capacity < 1
        or scene_state.shape[1] != flight_capacity
        or scene_state.shape[2] != 13
        or scene_state.device != device
        or scene_state.dtype != dtype
    ):
        raise LeanObservationError("Physical flight scene ABI differs")
    _assert_async_all(
        (flight_slot >= -1) & (flight_slot < flight_capacity),
        label="R06 current flight_slot is outside Physical capacity",
    )
    flight_index = flight_slot.clamp(0, flight_capacity - 1)
    env_index = torch.arange(n, dtype=torch.int64, device=device)
    ball = scene_state[env_index, flight_index]
    ball9 = torch.cat(
        (
            heading(ball[:, :3] - root_pos_env),
            heading(ball[:, 7:10]),
            heading(ball[:, 10:13]),
        ),
        dim=1,
    )
    ball9 = torch.where(live[:, None], ball9, torch.zeros_like(ball9))

    contact = contact[:, None]
    crossed = crossed[:, None]
    clear = clear[:, None]

    ready = r07.action_epoch_observation_state()
    postphysics_valid = _exact_tensor(
        ready.postphysics_valid,
        label="R07 observation postphysics_valid",
        shape=(n,),
        device=device,
        dtype=torch.bool,
    )
    source_step = _exact_tensor(
        ready.source_step,
        label="R07 observation source_step",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    ready_generation = _exact_tensor(
        ready.reset_generation,
        label="R07 observation reset_generation",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    motion_generation = _exact_tensor(
        motion_view.reset_generation,
        label="Motion observation reset_generation",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    ready_control_tick = _exact_tensor(
        ready.control_tick,
        label="R07 observation control_tick",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    motion_control_tick = _exact_tensor(
        motion_view.control_tick,
        label="Motion observation control_tick",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    ready_streak = _exact_tensor(
        ready.ready_streak,
        label="R07 observation streak",
        shape=(n,),
        device=device,
        dtype=torch.int64,
    )
    published_support = _exact_tensor(
        ready.foot_supported_lr,
        label="R07 observation foot_supported_lr",
        shape=(n, 2),
        device=device,
        dtype=torch.bool,
    )
    same_generation = ready_generation == motion_generation
    can_advance_generation = (
        ready_generation >= 0
    ) & (ready_generation < torch.iinfo(torch.int64).max)
    safe_generation_base = torch.where(
        can_advance_generation,
        ready_generation,
        torch.zeros_like(ready_generation),
    )
    next_generation = safe_generation_base + can_advance_generation.to(
        dtype=torch.int64
    )
    reset_boundary = (
        can_advance_generation & (motion_generation == next_generation)
    )
    cold_genesis = (
        ~postphysics_valid
        & source_step.eq(-1)
        & ready_generation.eq(-1)
        & ready_control_tick.eq(-1)
        & ready_streak.eq(0)
        & torch.all(~published_support, dim=1)
    )
    same_epoch_publication = (
        postphysics_valid
        & source_step.ge(0)
        & same_generation
        & ready_control_tick.eq(motion_control_tick)
    )
    _assert_async_all(
        (motion_generation >= 0)
        & (cold_genesis | reset_boundary | same_epoch_publication),
        label="R07 observation chronology differs",
    )
    if type(ready.required_dwell) is not int or ready.required_dwell < 1:
        raise LeanObservationError("R07 required dwell differs")
    # A selected true reset happens after this step's real post-physics
    # publication and before its returned observation.  The new Motion
    # generation is therefore the sole row-wise reset-boundary fact: expose
    # zero dwell for exactly those +1 rows while preserving every peer byte.
    # No new R07 publication or post-physics capability is fabricated.
    observation_streak = torch.where(
        same_epoch_publication, ready_streak, torch.zeros_like(ready_streak)
    )
    support = torch.where(
        same_epoch_publication[:, None],
        published_support,
        torch.zeros_like(published_support),
    )
    dwell = (
        observation_streak.clamp(0, ready.required_dwell).to(dtype)[:, None]
        / float(ready.required_dwell)
    )
    remaining = (
        (env.max_episode_length - env.episode_length_buf)
        .clamp_min(0)
        .to(dtype)[:, None]
        * float(env.step_dt)
    )
    critic_rows = {
        "episode_time_remaining_s": remaining,
        "live_ball_center_rel_root_heading": ball9[:, :3],
        "live_ball_lin_vel_heading": ball9[:, 3:6],
        "live_ball_ang_vel_heading": ball9[:, 6:9],
        "selected_rubber_contact_latched": contact.to(dtype),
        "net_crossed_latched": crossed.to(dtype),
        "net_clear_latched": clear.to(dtype),
        "foot_supported_lr": support.to(dtype),
        "cadence_ready_dwell_fraction": dwell,
    }

    epoch_uid = _gather_selected(record.action_uid, slot)
    _assert_async_all(
        ~task_valid | (motion_view.action_uid == epoch_uid),
        label="Motion/Epoch current action UID differs",
    )
    return DirectActionEpochObservationFacts(
        actor_rows=actor_rows,
        critic_rows=critic_rows,
        motion_phase_code=motion_view.phase,
        task_valid=task_valid,
        transaction_epoch=record.epoch,
        transaction_version=record.version,
        common_step=step,
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
        self._num_envs = epoch_owner.num_envs
        self._device = epoch_owner.device
        self._dtype = dtype
        self._actor_scale_v2 = torch.tensor(
            portable_observation.ACTOR_SCALE_FLAT_V2,
            dtype=dtype,
            device=self._device,
        ).reshape(1, ACTOR_WIDTH_V2)
        self._critic_extension_scale_v2 = torch.tensor(
            portable_observation.CRITIC_EXTENSION_SCALE_FLAT_V2,
            dtype=dtype,
            device=self._device,
        ).reshape(1, CRITIC_WIDTH_V2 - ACTOR_WIDTH_V2)
        self._widths = {"policy": ACTOR_WIDTH_V2, "critic": CRITIC_WIDTH_V2}
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
        if type(view.actor_rows) is not dict or type(view.critic_rows) is not dict:
            raise LeanObservationError("direct semantic row mappings differ")
        rows = {**view.actor_rows, **view.critic_rows}
        for name, width in _DIRECT_FLOAT_LAYOUT:
            _exact_tensor(
                rows.get(name),
                label="direct " + name,
                shape=(self._num_envs, width),
                device=self._device,
                dtype=self._dtype,
            )
        _exact_tensor(
            view.task_valid,
            label="direct task_valid",
            shape=(self._num_envs,),
            device=self._device,
            dtype=torch.bool,
        )
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
        return view

    def _pack(
        self,
        view: DirectActionEpochObservationFacts,
        record: epoch_v1.ActionEpochRecord,
        common_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del common_step
        slot = record.current_task_slot
        phase = _gather_selected(record.phase, slot)
        allowed_phase = (
            phase.eq(epoch_v1.PHASE_IDLE)
            | phase.eq(epoch_v1.PHASE_REVEAL_COMMITTED)
            | phase.eq(epoch_v1.PHASE_LAUNCH_SETTLED)
            | phase.eq(epoch_v1.PHASE_OUTCOME_SETTLED)
            | phase.eq(epoch_v1.PHASE_RETIRED)
        )
        _assert_async_all(allowed_phase, label="ActionEpoch public phase differs")
        learning_phase = torch.zeros_like(phase)
        for index, code in enumerate(
            (
                epoch_v1.PHASE_IDLE,
                epoch_v1.PHASE_REVEAL_COMMITTED,
                epoch_v1.PHASE_LAUNCH_SETTLED,
                epoch_v1.PHASE_OUTCOME_SETTLED,
                epoch_v1.PHASE_RETIRED,
            )
        ):
            learning_phase = torch.where(
                phase.eq(code), torch.full_like(phase, index), learning_phase
            )
        phase_one_hot = torch.nn.functional.one_hot(
            learning_phase, num_classes=5
        ).to(dtype=self._dtype)
        motion_phase = torch.nn.functional.one_hot(
            view.motion_phase_code, num_classes=5
        ).to(dtype=self._dtype)
        actor_rows = dict(view.actor_rows)
        actor_rows.update(
            {
                "motion_phase_one_hot": motion_phase,
                "epoch_learning_phase_one_hot": phase_one_hot,
                # Motion owns whether the retained Epoch task is currently
                # visible to the actor.  Epoch deliberately keeps its payload
                # valid through RETIRED until the next ACCEPT, so its payload
                # bit is not an actor-visibility mask.
                "task_valid": view.task_valid[:, None].to(dtype=self._dtype),
            }
        )
        actor = portable_observation.concatenate_layout_rows(
            ACTOR_LAYOUT_V2, actor_rows
        )
        actor.mul_(self._actor_scale_v2)
        critic_extension = portable_observation.concatenate_layout_rows(
            CRITIC_EXTENSION_LAYOUT_V2,
            view.critic_rows,
        )
        critic_extension.mul_(self._critic_extension_scale_v2)
        critic = torch.cat((actor, critic_extension), dim=1)
        if actor.shape != (self._num_envs, ACTOR_WIDTH_V2):
            raise LeanObservationError("named actor layout width differs")
        if critic.shape != (self._num_envs, CRITIC_WIDTH_V2):
            raise LeanObservationError("named critic layout width differs")
        _assert_async_all(torch.isfinite(actor), label="packed actor is nonfinite")
        _assert_async_all(torch.isfinite(critic), label="packed critic is nonfinite")
        return actor, critic

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
        return actor if group == "policy" else critic


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
        != {"policy": [(ACTOR_WIDTH_V2,)], "critic": [(CRITIC_WIDTH_V2,)]}
        or getattr(manager, "group_obs_dim", None)
        != {"policy": (ACTOR_WIDTH_V2,), "critic": (CRITIC_WIDTH_V2,)}
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
        != {"policy": ACTOR_WIDTH_V2, "critic": CRITIC_WIDTH_V2}
        or type(components) is not tuple
        or any(type(row) is not tuple or len(row) != 2 for row in components)
    ):
        raise LeanObservationError(
            "installed ActionEpoch observation resolver/source differs"
        )
    return {
        "actor_obs_contract": ACTOR_CONTRACT_V2,
        "actor_obs_mode": "action_ball_full_mdp",
        "actor_obs_total_dim": ACTOR_WIDTH_V2,
        "actor_obs_term_names": ["action_epoch"],
        "actor_obs_term_dims": [ACTOR_WIDTH_V2],
        "critic_obs_contract": CRITIC_CONTRACT_V2,
        "critic_obs_total_dim": CRITIC_WIDTH_V2,
        "critic_obs_term_names": ["action_epoch"],
        "critic_obs_term_dims": [CRITIC_WIDTH_V2],
        "fresh_full_mdp_observation_kind": OBSERVATION_KIND_V2,
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
    "ACTOR_CONTRACT_V2",
    "CRITIC_CONTRACT_V2",
    "OBSERVATION_KIND_V2",
    "ACTOR_LAYOUT_V2",
    "CRITIC_EXTENSION_LAYOUT_V2",
    "ACTOR_WIDTH_V2",
    "CRITIC_WIDTH_V2",
    "DirectActionEpochObservationFacts",
    "build_direct_action_epoch_observation_facts",
    "LeanObservationError",
    "LeanObservationConstructionHold",
    "LeanActionEpochObservationSource",
    "DiagnosticN2ObservationManagerBundle",
    "materialize_observation_manager_cfg",
    "installed_observation_facts",
]
