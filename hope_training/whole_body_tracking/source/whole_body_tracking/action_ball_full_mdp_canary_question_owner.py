#!/usr/bin/env python3
"""Recurring generic-N question composer for the full-MDP lean path.

The cold bundle binds the exact Racket, timing, profile, and Physical owners.
Every reveal consumes only D05's ephemeral authenticated context and returns
one device-resident round bank.  There is no parallel fixed-tape question
owner, public cadence input, question receipt, or private RNG family.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

try:
    import action_ball_continuous_runtime_transaction_device as _r05
    import action_ball_device_profile_authority as _profile
    import action_ball_physical_question_device as _physical
    import action_ball_full_mdp_diagnostic_action_timing as _timing
except ImportError:  # Installed package import.
    from . import action_ball_continuous_runtime_transaction_device as _r05
    from . import action_ball_device_profile_authority as _profile
    from . import action_ball_physical_question_device as _physical
    from . import action_ball_full_mdp_diagnostic_action_timing as _timing


INTEGRATION_STATUS = "d05_internal_question_source_construction_hold"
RUNTIME_INTEGRATED = False
LAUNCH_AUTHORIZED = False
DIAGNOSTIC_UNAUTHORIZED = True
FORMAL_AUTHORIZED = False

CANARY_SAVE_CHECKPOINTS = False

CONSTRUCTION_REASON_INVALID_PRODUCER = 12
CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL = 13
PRODUCER_FAULT_DIAGNOSTIC_SOURCE = 1 << 57
PRODUCER_FAULT_STATIC_ROW_BINDING = 1 << 59

_I64_MAX = (1 << 63) - 1


class CanaryQuestionError(RuntimeError):
    """Base error for the canary question composition boundary."""


class CanaryQuestionConstructionHold(CanaryQuestionError):
    """A required fresh causal source does not exist yet."""


class RecurringD05InternalQuestionBundle:
    """Cold-static, recurring-hot composer bound directly inside D05.

    It owns no RNG, cadence, question receipt, or bank registry.  Each call
    consumes one ephemeral D05 context containing authenticated cadence,
    profile, three fixed rounds of six full-N uniform question draws
    (``[N, 3, 6]``), and D05-reserved candidate IDs.  D05 consumes the
    nineteenth draw separately for the final cell selection.
    """

    __slots__ = (
        "_racket_owner",
        "_num_envs",
        "_physical_owner",
        "_timing_table",
        "_contact_position_env_m",
        "_reference_quat",
        "_reference_omega",
        "_reference_site_speed",
        "_reference_normal",
        "_base_yaw_quat",
        "_face_reach_offset_xy",
        "_prototype_direction",
        "_prototype_speed_min",
        "_prototype_speed_max",
        "_policy_step_s",
        "_episode_length_s",
        "_venue",
        "_question_cfg",
        "_surface_z_m",
        "_net_x_m",
        "_net_top_z_m",
        "_integration_step_s",
        "_integration_steps",
    )

    def __init__(
        self,
        *,
        racket_owner: object,
        num_envs: int,
        physical_owner: _physical.PhysicalQuestionNumericCore,
        timing_table: _timing.DiagnosticActionTimingStaticTableProjection,
        contact_position_env_m: torch.Tensor,
        reference_quat: torch.Tensor,
        reference_omega: torch.Tensor,
        reference_site_speed: torch.Tensor,
        reference_normal: torch.Tensor,
        base_yaw_quat: torch.Tensor,
        face_reach_offset_xy: torch.Tensor,
        prototype_direction: torch.Tensor,
        prototype_speed_min: torch.Tensor,
        prototype_speed_max: torch.Tensor,
        policy_step_s: float,
        episode_length_s: float,
        venue: object,
        question_cfg: object,
        surface_z_m: float,
        net_x_m: float,
        net_top_z_m: float,
        integration_step_s: float,
        integration_steps: int,
    ) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise CanaryQuestionError(
                "recurring question num_envs must be a positive exact int"
            )
        self._racket_owner = racket_owner
        self._num_envs = num_envs
        self._physical_owner = physical_owner
        self._timing_table = timing_table
        self._contact_position_env_m = contact_position_env_m
        self._reference_quat = reference_quat
        self._reference_omega = reference_omega
        self._reference_site_speed = reference_site_speed
        self._reference_normal = reference_normal
        self._base_yaw_quat = base_yaw_quat
        self._face_reach_offset_xy = face_reach_offset_xy
        self._prototype_direction = prototype_direction
        self._prototype_speed_min = prototype_speed_min
        self._prototype_speed_max = prototype_speed_max
        self._policy_step_s = policy_step_s
        self._episode_length_s = episode_length_s
        self._venue = venue
        self._question_cfg = question_cfg
        self._surface_z_m = surface_z_m
        self._net_x_m = net_x_m
        self._net_top_z_m = net_top_z_m
        self._integration_step_s = integration_step_s
        self._integration_steps = integration_steps

    def compose_r05_candidate_bank_inside_prepare(
        self,
        internal_context: object,
    ) -> _r05.DeviceQuestionProjection:
        (
            cadence_receipt,
            cadence,
            profile,
            device,
            support,
            draw_u01,
            candidate_identity,
            construction_mask,
            bank_sequence,
        ) = _r05._consume_internal_question_context(internal_context, self)
        return _compose_recurring_question_projection(
            bundle=self,
            cadence_receipt=cadence_receipt,
            cadence=cadence,
            profile=profile,
            device=device,
            support=support,
            draw_u01=draw_u01,
            candidate_identity=candidate_identity,
            construction_mask=construction_mask,
            bank_sequence=bank_sequence,
        )


def _compose_recurring_question_projection(
    *,
    bundle: RecurringD05InternalQuestionBundle,
    cadence_receipt: object,
    cadence: _r05.DeviceCadenceProjection,
    profile: _r05.DeviceProfileProjection,
    device: torch.device,
    support: int,
    draw_u01: torch.Tensor,
    candidate_identity: torch.Tensor,
    construction_mask: torch.Tensor,
    bank_sequence: int,
) -> _r05.DeviceQuestionProjection:
    """Device-hot recurring composition; no receipt or caller chronology."""

    try:
        from whole_body_tracking.tasks.tracking.mdp import (
            action_ball_exact_face_timing_device as exact_face,
        )
        from whole_body_tracking.tasks.tracking.mdp import (
            continuous_questions as questions,
        )
    except ImportError as exc:
        raise CanaryQuestionConstructionHold(
            "recurring question numeric kernels are unavailable"
        ) from exc

    k = getattr(cadence, "selected_count", None)
    rounds = _r05.INTERNAL_QUESTION_REDRAW_ROUNDS
    if (
        type(device) is not torch.device
        or type(k) is not int
        or k < 1
        or k > bundle._num_envs
        or type(support) is not int
        or support != 3
        or support != profile.support_size
        or bundle._question_cfg.max_redraw_rounds != rounds
    ):
        raise CanaryQuestionError("recurring question row/support binding differs")
    selected_env_index = _require_tensor(
        getattr(cadence, "selected_env_index", None),
        name="D05 due environment index",
        device=device,
        dtype=torch.int64,
        shape=(k,),
    )
    _require_tensor(
        construction_mask,
        name="D05 construction mask",
        device=device,
        dtype=torch.bool,
        shape=(k,),
    )
    _require_tensor(
        draw_u01,
        name="D05 question draws",
        device=device,
        dtype=torch.float32,
        shape=(k, rounds, _r05.INTERNAL_QUESTION_DRAW_WIDTH),
    )
    _require_tensor(
        candidate_identity,
        name="D05 candidate identity",
        device=device,
        dtype=torch.int64,
        shape=(k, rounds, support),
    )
    action_count = bundle._contact_position_env_m.shape[0]
    slots = cadence.action_slot
    slot_fault = slots.lt(0) | slots.ge(action_count)
    safe_slots = torch.clamp(slots, min=0, max=action_count - 1)

    def gather(value: torch.Tensor) -> torch.Tensor:
        return torch.index_select(value, 0, safe_slots).contiguous()

    timing = bundle._timing_table
    action_uid = gather(timing.action_uid)
    time_to_contact_ticks = gather(timing.time_to_contact_ticks)
    contact = gather(bundle._contact_position_env_m)
    reference_quat = gather(bundle._reference_quat)
    reference_omega = gather(bundle._reference_omega)
    reference_site_speed = gather(bundle._reference_site_speed)
    reference_normal = gather(bundle._reference_normal)
    base_quat = gather(bundle._base_yaw_quat)
    reach_offset = gather(bundle._face_reach_offset_xy)
    teacher_rate_min = gather(timing.teacher_rate_min)
    teacher_rate_max = gather(timing.teacher_rate_max)
    reference_t_hit = gather(timing.reference_t_hit_s)
    reference_t_cycle = gather(timing.reference_t_cycle_s)
    reaction_margin = gather(timing.reaction_margin_s)
    mount_sign = gather(timing.mount_normal_sign)

    from whole_body_tracking.tasks.tracking.mdp import continuous_questions

    defaults = continuous_questions.ContinuousQuestionCfg()
    velocity_box = torch.tensor(
        defaults.vel_range,
        dtype=torch.float32,
        device=device,
    )
    velocity = velocity_box[:, 0] + draw_u01[:, :, :3] * (
        velocity_box[:, 1] - velocity_box[:, 0]
    )
    spin = (draw_u01[:, :, 3:6] * 2.0 - 1.0) * float(
        defaults.spin_abs_max
    )
    target = (
        profile.targets_xy_m.unsqueeze(0)
        .unsqueeze(0)
        .expand(k, rounds, support, 2)
        .contiguous()
    )
    contact_cells = _expand_round_cells(contact, rounds, support)
    velocity_cells = _expand_round_values(velocity, support)
    spin_cells = _expand_round_values(spin, support)
    normal_cells = _expand_round_cells(reference_normal, rounds, support)
    base_cells = _expand_round_cells(base_quat, rounds, support)
    sign_cells = _expand_round_cells(mount_sign, rounds, support)

    solver = questions.solve_proposals_device(
        _flat_round_cells(_expand_round_cells(slots, rounds, support)),
        _flat_round_cells(contact_cells),
        _flat_round_cells(velocity_cells),
        _flat_round_cells(spin_cells),
        _flat_round_cells(target),
        _flat_round_cells(normal_cells),
        protos=SimpleNamespace(
            v_hat_b=bundle._prototype_direction,
            speed_min=bundle._prototype_speed_min,
            speed_max=bundle._prototype_speed_max,
            face_sign=timing.mount_normal_sign,
        ),
        base_quat=_flat_round_cells(base_cells),
        prm=bundle._venue,
        surface_z=bundle._surface_z_m,
        net_x=bundle._net_x_m,
        net_top_z=bundle._net_top_z_m,
        cfg=bundle._question_cfg,
        h=bundle._integration_step_s,
        n_steps=bundle._integration_steps,
    )
    solver_ok = solver.ok.reshape(k, rounds, support).unsqueeze(-1)
    contact_room = cadence.episode_tick.le(_I64_MAX - time_to_contact_ticks)
    safe_episode = torch.where(
        contact_room, cadence.episode_tick, torch.zeros_like(cadence.episode_tick)
    )
    contact_tick_row = safe_episode + time_to_contact_ticks
    ttc = time_to_contact_ticks.to(torch.float32) * bundle._policy_step_s
    exact = exact_face.solve_exact_face_timing_device(
        ball_contact_w_m=torch.where(
            solver_ok.reshape(-1, 1),
            solver.p_contact,
            _flat_round_cells(contact_cells),
        ),
        racket_face_center_velocity_w_mps=torch.where(
            solver_ok.reshape(-1, 1),
            solver.v_racket,
            torch.zeros_like(solver.v_racket),
        ),
        solved_raw_a_normal_w=torch.where(
            solver_ok.reshape(-1, 1),
            solver.n_racket,
            _flat_round_cells(normal_cells),
        ),
        mount_normal_sign=_flat_round_cells(sign_cells),
        reference_racket_quat_wxyz=_flat_round_cells(
            _expand_round_cells(reference_quat, rounds, support)
        ),
        reference_racket_angular_velocity_w_radps=_flat_round_cells(
            _expand_round_cells(reference_omega, rounds, support)
        ),
        reference_racket_site_speed_mps=_flat_round_cells(
            _expand_round_cells(reference_site_speed, rounds, support)
        ),
        teacher_rate_min=_flat_round_cells(
            _expand_round_cells(teacher_rate_min, rounds, support)
        ),
        teacher_rate_max=_flat_round_cells(
            _expand_round_cells(teacher_rate_max, rounds, support)
        ),
        time_to_contact_s=_flat_round_cells(
            _expand_round_cells(ttc, rounds, support)
        ),
        reference_t_hit_s=_flat_round_cells(
            _expand_round_cells(reference_t_hit, rounds, support)
        ),
        reference_t_cycle_s=_flat_round_cells(
            _expand_round_cells(reference_t_cycle, rounds, support)
        ),
        reaction_margin_s=_flat_round_cells(
            _expand_round_cells(reaction_margin, rounds, support)
        ),
        attempt_close_margin_s=_flat_round_cells(
            _expand_round_cells(
                torch.full_like(ttc, bundle._policy_step_s), rounds, support
            )
        ),
        episode_length_s=_flat_round_cells(
            _expand_round_cells(
                torch.full_like(ttc, bundle._episode_length_s), rounds, support
            )
        ),
    )
    horizon_receipt = bundle._physical_owner.issue_horizon_for_test(
        _physical.PhysicalQuestionCandidateBatch(
            candidate_identity=candidate_identity,
            contact_position_env_m=contact_cells,
            incoming_linear_velocity_world_mps=velocity_cells,
            incoming_angular_velocity_world_radps=spin_cells,
        )
    )
    horizon = bundle._physical_owner.project_horizon_for_test(horizon_receipt)
    if type(horizon) is not _physical.PhysicalQuestionHorizonView:
        raise CanaryQuestionError("Physical horizon projection exact type differs")
    max_horizon = _require_tensor(
        horizon.max_feasible_motion_ticks,
        name="Physical max feasible motion ticks",
        device=device,
        dtype=torch.int64,
        shape=(k, rounds, support),
    )
    contact_tick = _expand_round_cells(contact_tick_row, rounds, support)
    chosen_horizon = torch.minimum(
        max_horizon,
        contact_tick.clamp(min=0),
    ).contiguous()
    launch_tick = (contact_tick - chosen_horizon).contiguous()
    pending_elapsed = _expand_round_cells(
        cadence.pending_elapsed_s.to(torch.float64), rounds, support
    )
    task_duration_raw = (
        exact.pre_swing_wait_s.reshape(k, rounds, support).to(torch.float64)
        + exact.scaled_t_cycle_s.reshape(k, rounds, support).to(torch.float64)
        - pending_elapsed
    )
    task_duration_finite = torch.isfinite(task_duration_raw)
    task_duration = torch.where(
        task_duration_finite,
        task_duration_raw.clamp(min=0.0),
        torch.zeros_like(task_duration_raw),
    )
    task_close_offset = torch.ceil(
        task_duration / float(bundle._policy_step_s) - 1.0e-12
    ).to(torch.int64)
    task_close_room = task_duration_finite & cadence.reveal_tick.reshape(
        k, 1, 1
    ).le(_I64_MAX - task_close_offset)
    safe_task_epoch = torch.where(
        task_close_room,
        cadence.reveal_tick.reshape(k, 1, 1),
        torch.zeros_like(task_close_offset),
    )
    task_close_tick = (safe_task_epoch + task_close_offset).contiguous()
    physical = bundle._physical_owner.finalize_exact_ticks_for_test(
        horizon_receipt,
        candidate_identity=candidate_identity,
        contact_tick=contact_tick,
        launch_tick=launch_tick,
    )
    reason = _reason_precedence(
        solver.proposals.reason_code.reshape(k, rounds, support),
        exact.construction_reason.reshape(k, rounds, support),
        physical.construction_reason,
    )
    admitted_before_suffix = reason.eq(
        _r05.QUESTION_CONSTRUCTION_REASON_ADMITTED
    )
    full_suffix_fits = cadence.next_reveal_tick.reshape(k, 1, 1).gt(
        task_close_tick
    )
    reason = torch.where(
        admitted_before_suffix
        & task_close_room
        & ~full_suffix_fits,
        torch.full_like(
            reason,
            CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL,
        ),
        reason,
    )
    active_task_chronology = reason.eq(
        _r05.QUESTION_CONSTRUCTION_REASON_ADMITTED
    ) | reason.eq(CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL)
    task_close_tick = torch.where(
        active_task_chronology,
        task_close_tick,
        torch.full_like(task_close_tick, -1),
    ).contiguous()
    structural_fault = torch.where(
        construction_mask,
        cadence.cadence_producer_fault,
        torch.zeros_like(cadence.cadence_producer_fault),
    )
    structural_fault = torch.bitwise_or(
        structural_fault,
        (
            construction_mask
            & (
                selected_env_index.lt(0)
                | selected_env_index.ge(bundle._num_envs)
                | selected_env_index.unsqueeze(1).eq(
                    selected_env_index.unsqueeze(0)
                ).triu(diagonal=1).any(dim=1)
                | selected_env_index.unsqueeze(1).eq(
                    selected_env_index.unsqueeze(0)
                ).tril(diagonal=-1).any(dim=1)
                |
                slot_fault
                | ~contact_room
                | action_uid.le(0)
                | time_to_contact_ticks.le(0)
            )
        ).to(torch.int64)
        * PRODUCER_FAULT_STATIC_ROW_BINDING,
    )
    round_producer_fault = torch.zeros(
        (k, rounds), dtype=torch.int64, device=device
    )
    for fault in (
        solver.producer_fault_bits.reshape(k, rounds, support),
        exact.producer_fault_bits.reshape(k, rounds, support),
        physical.producer_fault,
    ):
        round_producer_fault = torch.bitwise_or(
            round_producer_fault,
            fault.ne(0).any(dim=2).to(torch.int64)
            * PRODUCER_FAULT_DIAGNOSTIC_SOURCE,
        )
    round_producer_fault = torch.where(
        construction_mask[:, None],
        round_producer_fault,
        torch.zeros_like(round_producer_fault),
    )
    structural_fault = structural_fault.contiguous()
    round_producer_fault = round_producer_fault.contiguous()
    reason = torch.where(
        round_producer_fault.ne(0).unsqueeze(2)
        | structural_fault.ne(0).reshape(k, 1, 1),
        torch.full_like(reason, CONSTRUCTION_REASON_INVALID_PRODUCER),
        reason,
    ).contiguous()
    reason = torch.where(
        construction_mask[:, None, None],
        reason,
        torch.full_like(reason, CONSTRUCTION_REASON_INVALID_PRODUCER),
    ).contiguous()
    installable = reason.eq(-1).unsqueeze(-1)
    motion = torch.stack(
        (
            _flat_round_cells(
                _expand_round_cells(ttc, rounds, support)
            ),
            exact.teacher_rate,
            exact.scaled_t_hit_s,
            exact.scaled_t_cycle_s,
            exact.pre_swing_wait_s,
        ),
        dim=1,
    ).reshape(
        k, rounds, support, len(_r05.MOTION_TASK_F32_FIELDS)
    )
    base_goal = _expand_round_cells(
        contact[:, :2] - reach_offset, rounds, support
    )
    racket = torch.cat(
        (
            exact.racket_site_target_w_m.reshape(k, rounds, support, 3),
            exact.racket_site_velocity_w_mps.reshape(k, rounds, support, 3),
            solver.n_racket.reshape(k, rounds, support, 3),
            solver.p_contact.reshape(k, rounds, support, 3),
            solver.v_racket.reshape(k, rounds, support, 3),
            exact.racket_command_quat_wxyz.reshape(k, rounds, support, 4),
            base_goal,
            solver.v_ball_in.reshape(k, rounds, support, 3),
            solver.w_ball_in.reshape(k, rounds, support, 3),
        ),
        dim=3,
    )
    bank = _r05.DeviceR05CandidateRoundBank(
        candidate_identity=candidate_identity,
        construction_reason=reason,
        producer_fault=round_producer_fault,
        motion_task_f32=torch.where(
            installable & torch.isfinite(motion), motion, torch.zeros_like(motion)
        ).contiguous(),
        racket_task_f32=torch.where(
            installable & torch.isfinite(racket), racket, torch.zeros_like(racket)
        ).contiguous(),
        physical_state_f32=torch.where(
            installable & torch.isfinite(physical.physical_state_f32),
            physical.physical_state_f32,
            torch.zeros_like(physical.physical_state_f32),
        ).contiguous(),
    )
    return _r05.DeviceQuestionProjection(
        cadence_receipt_identity=cadence_receipt,
        bank_identity=object(),
        bank_sequence=bank_sequence,
        bank=None,
        producer_fault=structural_fault,
        selected_count=k,
        support_size=support,
        chronology=None,
        round_bank=bank,
        round_chronology=_r05.DeviceQuestionRoundChronology(
            action_uid=_expand_round_cells(action_uid, rounds, support),
            contact_tick=contact_tick,
            launch_tick=launch_tick,
            chosen_horizon_ticks=chosen_horizon,
            task_close_tick=task_close_tick,
        ),
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device != device
        or value.dtype is not dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
    ):
        raise CanaryQuestionError(
            f"{name} must be contiguous {dtype} on {device} with shape {shape}"
        )
    return value


def _require_exact_bound_method(
    owner: object,
    *,
    owner_type: type,
    method_name: str,
) -> None:
    if type(owner) is not owner_type:
        raise CanaryQuestionError(f"{method_name} owner exact type differs")
    method = getattr(owner, method_name, None)
    declared = getattr(owner_type, method_name, None)
    if (
        type(method) is not MethodType
        or method.__self__ is not owner
        or method.__func__ is not declared
    ):
        raise CanaryQuestionError(f"{method_name} owner method differs")


def _expand_round_cells(
    value: torch.Tensor, round_count: int, support_size: int
) -> torch.Tensor:
    return value.reshape(value.shape[0], 1, 1, *value.shape[1:]).expand(
        value.shape[0], round_count, support_size, *value.shape[1:]
    ).contiguous()


def _expand_round_values(
    value: torch.Tensor, support_size: int
) -> torch.Tensor:
    return value.unsqueeze(2).expand(
        value.shape[0], value.shape[1], support_size, *value.shape[2:]
    ).contiguous()


def _flat_round_cells(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(
        value.shape[0] * value.shape[1] * value.shape[2], *value.shape[3:]
    )


def _reason_precedence(
    solver_reason: torch.Tensor,
    exact_face_reason: torch.Tensor,
    physical_reason: torch.Tensor,
) -> torch.Tensor:
    admitted = torch.full_like(
        solver_reason, _r05.QUESTION_CONSTRUCTION_REASON_ADMITTED
    )
    reason = torch.where(solver_reason.ne(-1), solver_reason, admitted)
    reason = torch.where(
        reason.eq(-1) & exact_face_reason.ne(-1), exact_face_reason, reason
    )
    return torch.where(
        reason.eq(-1) & physical_reason.ne(-1), physical_reason, reason
    ).contiguous()


def _quat_rotate_wxyz(
    quaternion: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    scalar = quaternion[:, :1]
    xyz = quaternion[:, 1:]
    uv = torch.cross(xyz, vector, dim=1)
    uuv = torch.cross(xyz, uv, dim=1)
    return vector + 2.0 * (scalar * uv + uuv)


def _base_yaw_quaternion(base_quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = base_quat.unbind(dim=1)
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    half = yaw * 0.5
    return torch.stack(
        (
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ),
        dim=1,
    ).contiguous()


def _float32_ceil_device(value: torch.Tensor) -> torch.Tensor:
    rounded = value.to(dtype=torch.float32)
    return torch.where(
        rounded.to(torch.float64) < value,
        torch.nextafter(rounded, torch.full_like(rounded, float("inf"))),
        rounded,
    ).contiguous()


def _float32_floor_device(value: torch.Tensor) -> torch.Tensor:
    rounded = value.to(dtype=torch.float32)
    return torch.where(
        rounded.to(torch.float64) > value,
        torch.nextafter(rounded, torch.full_like(rounded, float("-inf"))),
        rounded,
    ).contiguous()


def construct_recurring_d05_internal_question_bundle(
    *,
    profile_owner: _profile.DeviceProfileAuthorityOwner,
    profile_receipt: _profile.DeviceProfileReceipt,
    racket_owner: object,
    physical_owner: _physical.PhysicalQuestionNumericCore,
) -> RecurringD05InternalQuestionBundle:
    """Construct one cold-static bundle for all recurring D05 reveals.

    The recurring bundle will consume only D05's ephemeral internal context:
    authenticated cadence/profile plus 18 fixed-width transactional RNG draws
    shaped ``[selected, 3, 6]``.  D05 separately consumes the final nineteenth
    selection draw.  It will not mint a question receipt, accept a caller
    cadence, or recreate timing/incoming owners per reveal.  Contact position
    and FK rows come from Racket's closure-private sealed
    Motion table.  Timing/profile rows are materialized once.  The returned
    composer performs only device gathers and numerical kernels.
    """

    _require_exact_bound_method(
        profile_owner,
        owner_type=_profile.DeviceProfileAuthorityOwner,
        method_name="require_owned_r05_profile",
    )
    _require_exact_bound_method(
        physical_owner,
        owner_type=_physical.PhysicalQuestionNumericCore,
        method_name="issue_horizon_for_test",
    )
    _require_exact_bound_method(
        physical_owner,
        owner_type=_physical.PhysicalQuestionNumericCore,
        method_name="project_horizon_for_test",
    )
    _require_exact_bound_method(
        physical_owner,
        owner_type=_physical.PhysicalQuestionNumericCore,
        method_name="finalize_exact_ticks_for_test",
    )
    try:
        from whole_body_tracking.tasks.tracking.mdp import commands
        from whole_body_tracking.tasks.tracking.mdp import continuous_questions
        from whole_body_tracking.tasks.tracking.mdp import hope_commands
        from whole_body_tracking.tasks.tracking.mdp import racket_contact_geometry
        from whole_body_tracking.tasks.tracking.mdp import virtual_ball
    except ImportError as exc:
        raise CanaryQuestionConstructionHold(
            "recurring question cold dependencies are unavailable"
        ) from exc
    _require_exact_bound_method(
        racket_owner,
        owner_type=hope_commands.RacketTargetCommand,
        method_name="_motion",
    )
    _require_exact_bound_method(
        racket_owner,
        owner_type=hope_commands.RacketTargetCommand,
        method_name=(
            "initialize_action_ball_full_mdp_racket_action_reference_cold"
        ),
    )
    _require_exact_bound_method(
        racket_owner,
        owner_type=hope_commands.RacketTargetCommand,
        method_name=(
            "project_action_ball_full_mdp_racket_action_reference_static_table"
        ),
    )
    env = getattr(racket_owner, "_env", None)
    command_manager = getattr(env, "command_manager", None)
    get_term = getattr(command_manager, "get_term", None)
    motion_name = getattr(getattr(racket_owner, "cfg", None), "motion_command_name", None)
    if (
        type(motion_name) is not str
        or not motion_name
        or not callable(get_term)
        or getattr(get_term, "__self__", None) is not command_manager
    ):
        raise CanaryQuestionError(
            "recurring Racket has no exact environment Motion source"
        )
    cached_motion_owner = getattr(racket_owner, "_motion_term", None)
    # Bind through Racket's exact resolver before any cold source can be
    # materialized.  A second manager read proves a non-idempotent/rebound
    # manager cannot install Motion B after this preflight accepted Motion A.
    motion_owner = racket_owner._motion()
    manager_motion_owner = get_term(motion_name)
    if motion_owner is not manager_motion_owner:
        detail = "cached" if cached_motion_owner is not None else "resolved"
        raise CanaryQuestionError(
            f"recurring {detail} Motion differs from CommandManager source"
        )
    motion_time_steps = getattr(motion_owner, "time_steps", None)
    racket_target_position = getattr(racket_owner, "racket_target_pos_w", None)
    if (
        type(motion_owner) is not commands.MotionCommand
        or getattr(motion_owner, "_env", None) is not env
        or type(motion_time_steps) is not torch.Tensor
        or motion_time_steps.dtype is not torch.int64
        or motion_time_steps.ndim != 1
        or not motion_time_steps.is_contiguous()
    ):
        raise CanaryQuestionError(
            "recurring Motion construction-bound row tensor differs"
        )
    num_envs = motion_time_steps.shape[0]
    if num_envs < 1:
        raise CanaryQuestionError(
            "recurring Motion construction-bound row count is empty"
        )
    _require_tensor(
        racket_target_position,
        name="recurring Racket construction-bound target rows",
        device=motion_time_steps.device,
        dtype=torch.float32,
        shape=(num_envs, 3),
    )
    if (
        type(getattr(env, "num_envs", None)) is not int
        or getattr(env, "num_envs") != num_envs
        or type(getattr(motion_owner, "num_envs", None)) is not int
        or getattr(motion_owner, "num_envs") != num_envs
        or type(getattr(racket_owner, "num_envs", None)) is not int
        or getattr(racket_owner, "num_envs") != num_envs
    ):
        raise CanaryQuestionError(
            "recurring owner metadata differs from construction-bound rows"
        )
    # The exact factory invokes this once after CommandManager exists.  Only
    # after the manager-owned Motion identity and live row cardinality close do
    # we allow Racket to bind its lazy cache and materialize the cold FK table.
    # This call accepts no caller rows, digest, builder, or cardinality verdict.
    racket_owner.initialize_action_ball_full_mdp_racket_action_reference_cold()
    if getattr(racket_owner, "_motion_term", None) is not motion_owner:
        raise CanaryQuestionError(
            "recurring Racket rebound the CommandManager Motion source"
        )
    profile = profile_owner.require_owned_r05_profile(profile_receipt)
    if type(profile) is not _r05.DeviceProfileProjection:
        raise CanaryQuestionError("recurring profile projection exact type differs")
    static = (
        hope_commands.RacketTargetCommand.
        project_action_ball_full_mdp_racket_action_reference_static_table(
            racket_owner
        )
    )
    timing = _timing.construct_action_ball_full_mdp_diagnostic_action_timing_static_table(
        racket_owner=racket_owner
    )
    if type(static) is not hope_commands.RacketActionReferenceStaticTableProjection:
        raise CanaryQuestionError("Racket recurring static table exact type differs")
    if type(timing) is not _timing.DiagnosticActionTimingStaticTableProjection:
        raise CanaryQuestionError("timing recurring static table exact type differs")
    device = profile.targets_xy_m.device
    if device != motion_time_steps.device:
        raise CanaryQuestionError(
            "recurring profile device differs from construction-bound rows"
        )
    action_count = static.reference_racket_site_position_w_m.shape[0]
    if action_count < 1 or timing.teacher_rate_min.shape != (action_count,):
        raise CanaryQuestionError("recurring cold table row count differs")
    face_offsets = torch.tensor(
        [
            racket_contact_geometry.face_center_from_site_local(int(sign))
            for sign in timing.mount_normal_sign.detach().cpu().tolist()
        ],
        dtype=torch.float32,
        device=device,
    )
    face_offset_w = _quat_rotate_wxyz(
        static.reference_racket_quat_wxyz, face_offsets
    )
    contact = (
        static.reference_racket_site_position_w_m + face_offset_w
    ).contiguous()
    face_velocity = static.reference_racket_site_velocity_w_mps + torch.cross(
        static.reference_racket_angular_velocity_w_radps,
        face_offset_w,
        dim=1,
    )
    # ExactFace's coupled rate equation consumes the native racket *site*
    # speed separately from the rigid-point face-centre search envelope.
    site_speed64 = torch.linalg.vector_norm(
        static.reference_racket_site_velocity_w_mps.to(torch.float64), dim=1
    )
    face_speed64 = torch.linalg.vector_norm(face_velocity.to(torch.float64), dim=1)
    if not bool(
        (
            torch.isfinite(site_speed64)
            & site_speed64.gt(0.0)
            & torch.isfinite(face_speed64)
            & face_speed64.gt(0.0)
        ).all().item()
    ):
        raise CanaryQuestionError(
            "Racket recurring reference site/face-centre speed is invalid"
        )
    base_yaw = _base_yaw_quaternion(static.reference_base_root_quat_wxyz)
    inverse = base_yaw.clone()
    inverse[:, 1:].neg_()
    direction = (
        _quat_rotate_wxyz(inverse, face_velocity).to(torch.float64)
        / face_speed64.unsqueeze(1)
    ).to(torch.float32).contiguous()
    # Reference-direction search band only.  Uniform teacher time scaling
    # scales ``v_site`` and ``omega`` together, so the exact rigid-point
    # face-centre vector scales as a whole.  ExactFace later remains the sole
    # authority for teacher-rate admission after the face normal changes.
    speed_min = _float32_ceil_device(
        timing.teacher_rate_min.to(torch.float64) * face_speed64
    )
    speed_max = _float32_floor_device(
        timing.teacher_rate_max.to(torch.float64) * face_speed64
    )
    cfg = getattr(racket_owner, "cfg", None)
    step_s = getattr(env, "step_dt", None)
    max_episode_length = getattr(env, "max_episode_length", None)
    if (
        type(step_s) is not float
        or type(max_episode_length) is not int
    ):
        raise CanaryQuestionError("recurring environment chronology differs")
    venue = virtual_ball.load_venue_params()
    question_cfg = continuous_questions.ContinuousQuestionCfg(
        fixed_direction=True,
        n_iters=int(cfg.cq_n_iters),
        tol_m=float(cfg.cq_tol_m),
        speed_budget=float(cfg.cq_speed_budget),
        max_redraw_rounds=int(cfg.cq_max_redraw_rounds),
    )
    return RecurringD05InternalQuestionBundle(
        racket_owner=racket_owner,
        num_envs=num_envs,
        physical_owner=physical_owner,
        timing_table=timing,
        contact_position_env_m=contact,
        reference_quat=static.reference_racket_quat_wxyz,
        reference_omega=static.reference_racket_angular_velocity_w_radps,
        reference_site_speed=site_speed64.to(torch.float32).contiguous(),
        reference_normal=static.reference_raw_face_normal_w,
        base_yaw_quat=base_yaw,
        face_reach_offset_xy=(
            static.reference_reach_offset_xy_m + face_offset_w[:, :2]
        ).contiguous(),
        prototype_direction=direction,
        prototype_speed_min=speed_min,
        prototype_speed_max=speed_max,
        policy_step_s=step_s,
        episode_length_s=step_s * max_episode_length,
        venue=venue,
        question_cfg=question_cfg,
        surface_z_m=float(cfg.vb_table_surface_z) + float(venue.ball_radius),
        net_x_m=float(getattr(racket_owner, "_vb_net_x")),
        net_top_z_m=(
            float(getattr(racket_owner, "_vb_net_top_z"))
            + float(venue.ball_radius)
        ),
        integration_step_s=float(cfg.vb_rollout_h),
        integration_steps=int(cfg.vb_rollout_steps),
    )


__all__ = (
    "CANARY_SAVE_CHECKPOINTS",
    "CONSTRUCTION_REASON_INVALID_PRODUCER",
    "CONSTRUCTION_REASON_FULL_SUFFIX_CROSSES_NEXT_REVEAL",
    "CanaryQuestionConstructionHold",
    "CanaryQuestionError",
    "DIAGNOSTIC_UNAUTHORIZED",
    "RecurringD05InternalQuestionBundle",
    "FORMAL_AUTHORIZED",
    "INTEGRATION_STATUS",
    "LAUNCH_AUTHORIZED",
    "PRODUCER_FAULT_DIAGNOSTIC_SOURCE",
    "PRODUCER_FAULT_STATIC_ROW_BINDING",
    "RUNTIME_INTEGRATED",
    "construct_recurring_d05_internal_question_bundle",
)
