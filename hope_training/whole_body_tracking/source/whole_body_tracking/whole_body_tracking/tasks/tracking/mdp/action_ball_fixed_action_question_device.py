"""Pure device-batch fixed-action ball-to-task construction.

The caller owns action selection, cadence, RNG, task-frame alignment and
lifecycle.  This module owns the one numerical mapping that both simulator
adapters must share:

``frozen action/reference + incoming ball + landing aim -> task + launch``.

All positions and orientations supplied here are already expressed in the
accepted shot's frozen task frame.  The kernel never selects another action,
reconstructs that frame, publishes to a scene, or creates a receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

try:
    from . import action_ball_exact_face_timing_device as _exact_face
    from . import continuous_questions as _questions
    from . import racket_contact_geometry as _racket_geometry
except ImportError:  # Dependency-light direct import.
    import action_ball_exact_face_timing_device as _exact_face
    import continuous_questions as _questions
    import racket_contact_geometry as _racket_geometry

try:
    import action_ball_physical_question_device as _physical
except ImportError as exc:  # pragma: no cover - installed source layout guard.
    raise ImportError("shared Physical question kernel is unavailable") from exc


CONSTRUCTION_REASON_ADMITTED = -1
CONSTRUCTION_REASON_INVALID_PRODUCER = 12


@dataclass(frozen=True)
class DeviceFixedActionQuestionResult:
    """One flat device batch and every independent construction fact."""

    motion_task_f32: torch.Tensor
    racket_task_f32: torch.Tensor
    physical_state_f32: torch.Tensor
    contact_tick: torch.Tensor
    launch_tick: torch.Tensor
    chosen_horizon_ticks: torch.Tensor
    construction_reason: torch.Tensor
    admitted: torch.Tensor
    solver_reason: torch.Tensor
    exact_face_reason: torch.Tensor
    physical_horizon_reason: torch.Tensor
    physical_reason: torch.Tensor
    solver_producer_fault: torch.Tensor
    exact_face_producer_fault: torch.Tensor
    physical_horizon_producer_fault: torch.Tensor
    physical_producer_fault: torch.Tensor
    producer_fault: torch.Tensor
    solver_residual_m: torch.Tensor


def _flat_row_count(value: torch.Tensor, *, width: int, name: str) -> int:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_rank = 1 if width == 0 else 2
    if value.ndim != expected_rank or (width and value.shape[1] != width):
        suffix = "" if width == 0 else f",{width}"
        raise ValueError(f"{name} must have shape (B{suffix})")
    if value.shape[0] < 1 or not value.is_contiguous():
        raise ValueError(f"{name} must be a non-empty contiguous flat batch")
    return int(value.shape[0])


def _require_float_peer(
    value: torch.Tensor,
    *,
    rows: int,
    width: int,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if _flat_row_count(value, width=width, name=name) != rows:
        raise ValueError(f"{name} row count differs")
    if value.dtype != dtype or value.device != device or not dtype.is_floating_point:
        raise ValueError(f"{name} dtype/device differs")


def _reason_precedence(
    solver_reason: torch.Tensor,
    exact_face_reason: torch.Tensor,
    physical_reason: torch.Tensor,
) -> torch.Tensor:
    reason = solver_reason.clone()
    reason = torch.where(
        reason.eq(CONSTRUCTION_REASON_ADMITTED)
        & exact_face_reason.ne(CONSTRUCTION_REASON_ADMITTED),
        exact_face_reason,
        reason,
    )
    return torch.where(
        reason.eq(CONSTRUCTION_REASON_ADMITTED)
        & physical_reason.ne(CONSTRUCTION_REASON_ADMITTED),
        physical_reason,
        reason,
    ).contiguous()


@torch.no_grad()
def solve_fixed_action_question_device(
    *,
    action_slot: torch.Tensor,
    candidate_identity: torch.Tensor,
    contact_position_env_m: torch.Tensor,
    incoming_linear_velocity_world_mps: torch.Tensor,
    incoming_angular_velocity_world_radps: torch.Tensor,
    landing_aim_xy_m: torch.Tensor,
    reference_raw_a_normal_w: torch.Tensor,
    base_yaw_quat_wxyz: torch.Tensor,
    reference_racket_quat_wxyz: torch.Tensor,
    reference_racket_angular_velocity_w_radps: torch.Tensor,
    reference_racket_site_speed_mps: torch.Tensor,
    mount_normal_sign: torch.Tensor,
    base_goal_xy_m: torch.Tensor,
    time_to_contact_s: torch.Tensor,
    reveal_tick: torch.Tensor,
    contact_tick: torch.Tensor,
    teacher_rate_min: torch.Tensor,
    teacher_rate_max: torch.Tensor,
    reference_t_hit_s: torch.Tensor,
    reference_t_cycle_s: torch.Tensor,
    reaction_margin_s: torch.Tensor,
    attempt_close_margin_s: torch.Tensor,
    episode_length_s: torch.Tensor,
    prototype_direction_b: torch.Tensor,
    prototype_speed_min_mps: torch.Tensor,
    prototype_speed_max_mps: torch.Tensor,
    prototype_face_sign: torch.Tensor,
    venue_params: object,
    question_config: object,
    physical_params: _physical.PhysicalQuestionFlightParams,
    physical_config: _physical.PhysicalQuestionNumericConfig,
    table_surface_z_m: float,
    net_x_m: float,
    net_top_z_m: float,
    integration_step_s: float,
    integration_steps: int,
) -> DeviceFixedActionQuestionResult:
    """Solve one already task-aligned flat batch without host value reads."""

    rows = _flat_row_count(
        contact_position_env_m, width=3, name="contact_position_env_m"
    )
    dtype = contact_position_env_m.dtype
    device = contact_position_env_m.device
    if not dtype.is_floating_point:
        raise TypeError("fixed-action question tensors must use floating dtype")
    for name, value, width in (
        ("incoming_linear_velocity_world_mps", incoming_linear_velocity_world_mps, 3),
        ("incoming_angular_velocity_world_radps", incoming_angular_velocity_world_radps, 3),
        ("landing_aim_xy_m", landing_aim_xy_m, 2),
        ("reference_raw_a_normal_w", reference_raw_a_normal_w, 3),
        ("base_yaw_quat_wxyz", base_yaw_quat_wxyz, 4),
        ("reference_racket_quat_wxyz", reference_racket_quat_wxyz, 4),
        (
            "reference_racket_angular_velocity_w_radps",
            reference_racket_angular_velocity_w_radps,
            3,
        ),
        ("reference_racket_site_speed_mps", reference_racket_site_speed_mps, 0),
        ("mount_normal_sign", mount_normal_sign, 0),
        ("base_goal_xy_m", base_goal_xy_m, 2),
        ("time_to_contact_s", time_to_contact_s, 0),
        ("teacher_rate_min", teacher_rate_min, 0),
        ("teacher_rate_max", teacher_rate_max, 0),
        ("reference_t_hit_s", reference_t_hit_s, 0),
        ("reference_t_cycle_s", reference_t_cycle_s, 0),
        ("reaction_margin_s", reaction_margin_s, 0),
        ("attempt_close_margin_s", attempt_close_margin_s, 0),
        ("episode_length_s", episode_length_s, 0),
    ):
        _require_float_peer(
            value,
            rows=rows,
            width=width,
            name=name,
            dtype=dtype,
            device=device,
        )
    for name, value in (
        ("action_slot", action_slot),
        ("candidate_identity", candidate_identity),
        ("reveal_tick", reveal_tick),
        ("contact_tick", contact_tick),
    ):
        if (
            _flat_row_count(value, width=0, name=name) != rows
            or value.dtype is not torch.int64
            or value.device != device
        ):
            raise ValueError(f"{name} int64/device binding differs")

    solver = _questions.solve_proposals_device(
        action_slot,
        contact_position_env_m,
        incoming_linear_velocity_world_mps,
        incoming_angular_velocity_world_radps,
        landing_aim_xy_m,
        reference_raw_a_normal_w,
        protos=SimpleNamespace(
            v_hat_b=prototype_direction_b,
            speed_min=prototype_speed_min_mps,
            speed_max=prototype_speed_max_mps,
            face_sign=prototype_face_sign,
        ),
        base_quat=base_yaw_quat_wxyz,
        prm=venue_params,
        surface_z=table_surface_z_m,
        net_x=net_x_m,
        net_top_z=net_top_z_m,
        cfg=question_config,
        h=integration_step_s,
        n_steps=integration_steps,
    )
    solver_ok = solver.ok.unsqueeze(-1)
    exact = _exact_face.solve_exact_face_timing_device(
        ball_contact_w_m=torch.where(
            solver_ok, solver.p_contact, contact_position_env_m
        ),
        racket_face_center_velocity_w_mps=torch.where(
            solver_ok, solver.v_racket, torch.zeros_like(solver.v_racket)
        ),
        solved_raw_a_normal_w=torch.where(
            solver_ok, solver.n_racket, reference_raw_a_normal_w
        ),
        mount_normal_sign=mount_normal_sign,
        reference_racket_quat_wxyz=reference_racket_quat_wxyz,
        reference_racket_angular_velocity_w_radps=(
            reference_racket_angular_velocity_w_radps
        ),
        reference_racket_site_speed_mps=reference_racket_site_speed_mps,
        teacher_rate_min=teacher_rate_min,
        teacher_rate_max=teacher_rate_max,
        time_to_contact_s=time_to_contact_s,
        reference_t_hit_s=reference_t_hit_s,
        reference_t_cycle_s=reference_t_cycle_s,
        reaction_margin_s=reaction_margin_s,
        attempt_close_margin_s=attempt_close_margin_s,
        episode_length_s=episode_length_s,
    )

    physical_batch = _physical.PhysicalQuestionCandidateBatch(
        candidate_identity=candidate_identity,
        contact_position_env_m=contact_position_env_m,
        incoming_linear_velocity_world_mps=incoming_linear_velocity_world_mps,
        incoming_angular_velocity_world_radps=(
            incoming_angular_velocity_world_radps
        ),
    )
    # Discovery and exact finalization share one transient numeric record.  No
    # receipt or adapter-owned identity is created at this pure boundary.
    physical = _physical.solve_max_final_segment_device(
        physical_batch,
        candidate_identity=candidate_identity,
        reveal_tick=reveal_tick,
        contact_tick=contact_tick,
        params=physical_params,
        config=physical_config,
    )
    chosen_horizon = physical.chosen_horizon_ticks
    launch_tick = physical.launch_tick

    solver_reason = solver.proposals.reason_code.contiguous()
    exact_reason = exact.construction_reason.contiguous()
    physical_reason = physical.construction_reason.contiguous()
    reason = _reason_precedence(solver_reason, exact_reason, physical_reason)
    producer_fault = torch.bitwise_or(
        torch.bitwise_or(solver.producer_fault_bits, exact.producer_fault_bits),
        physical.producer_fault,
    ).contiguous()
    reason = torch.where(
        producer_fault.ne(0),
        torch.full_like(reason, CONSTRUCTION_REASON_INVALID_PRODUCER),
        reason,
    ).contiguous()
    admitted = reason.eq(CONSTRUCTION_REASON_ADMITTED).contiguous()
    installable = admitted.unsqueeze(-1)
    motion = torch.stack(
        (
            time_to_contact_s,
            exact.teacher_rate,
            exact.scaled_t_hit_s,
            exact.scaled_t_cycle_s,
            exact.pre_swing_wait_s,
        ),
        dim=1,
    )
    racket = torch.cat(
        (
            exact.racket_site_target_w_m,
            exact.racket_site_velocity_w_mps,
            solver.n_racket,
            solver.p_contact,
            solver.v_racket,
            exact.racket_command_quat_wxyz,
            base_goal_xy_m,
            solver.v_ball_in,
            solver.w_ball_in,
        ),
        dim=1,
    )
    motion = torch.where(
        installable & torch.isfinite(motion), motion, torch.zeros_like(motion)
    ).contiguous()
    racket = torch.where(
        installable & torch.isfinite(racket), racket, torch.zeros_like(racket)
    ).contiguous()
    physical_state = torch.where(
        installable & torch.isfinite(physical.physical_state_f32),
        physical.physical_state_f32,
        torch.zeros_like(physical.physical_state_f32),
    ).contiguous()
    return DeviceFixedActionQuestionResult(
        motion_task_f32=motion,
        racket_task_f32=racket,
        physical_state_f32=physical_state,
        # Reuse the Physical leaf's already-cloned clock fact.  Returning the
        # caller tensor would let later adapter mutation rewrite this frozen
        # numerical result through shared storage.
        contact_tick=physical.contact_tick,
        launch_tick=launch_tick,
        chosen_horizon_ticks=chosen_horizon,
        construction_reason=reason,
        admitted=admitted,
        solver_reason=solver_reason,
        exact_face_reason=exact_reason,
        physical_horizon_reason=physical.horizon_construction_reason,
        physical_reason=physical_reason,
        solver_producer_fault=solver.producer_fault_bits.contiguous(),
        exact_face_producer_fault=exact.producer_fault_bits.contiguous(),
        physical_horizon_producer_fault=(
            physical.horizon_producer_fault.contiguous()
        ),
        physical_producer_fault=physical.producer_fault.contiguous(),
        producer_fault=producer_fault,
        solver_residual_m=solver.resid_m.contiguous(),
    )


@torch.no_grad()
def construct_reference_center_question_device(
    *,
    candidate_identity: torch.Tensor,
    contact_position_env_m: torch.Tensor,
    incoming_linear_velocity_world_mps: torch.Tensor,
    incoming_angular_velocity_world_radps: torch.Tensor,
    landing_aim_xy_m: torch.Tensor,
    reference_raw_a_normal_w: torch.Tensor,
    reference_racket_quat_wxyz: torch.Tensor,
    reference_racket_angular_velocity_w_radps: torch.Tensor,
    reference_racket_site_velocity_w_mps: torch.Tensor,
    reference_racket_site_speed_mps: torch.Tensor,
    mount_normal_sign: torch.Tensor,
    base_goal_xy_m: torch.Tensor,
    time_to_contact_s: torch.Tensor,
    reveal_tick: torch.Tensor,
    contact_tick: torch.Tensor,
    teacher_rate_min: torch.Tensor,
    teacher_rate_max: torch.Tensor,
    reference_t_hit_s: torch.Tensor,
    reference_t_cycle_s: torch.Tensor,
    reaction_margin_s: torch.Tensor,
    attempt_close_margin_s: torch.Tensor,
    episode_length_s: torch.Tensor,
    venue_params: object,
    question_config: object,
    physical_params: _physical.PhysicalQuestionFlightParams,
    physical_config: _physical.PhysicalQuestionNumericConfig,
    table_surface_z_m: float,
    net_x_m: float,
    net_top_z_m: float,
    integration_step_s: float,
    integration_steps: int,
) -> DeviceFixedActionQuestionResult:
    """Publish the exact zero-width centre whose answer is the teacher.

    The curriculum centre is chosen for this action, rather than asking the
    policy to imitate one paddle state while a second inverse solve commands a
    different face.  This leaf therefore performs only the forward physics
    check that the teacher answer opens legal landing; non-zero curriculum
    widths continue to require ``solve_fixed_action_question_device``.
    """

    rows = _flat_row_count(
        contact_position_env_m, width=3, name="contact_position_env_m"
    )
    dtype = contact_position_env_m.dtype
    device = contact_position_env_m.device
    if not dtype.is_floating_point:
        raise TypeError("reference-centre tensors must use floating dtype")
    for name, value, width in (
        ("incoming_linear_velocity_world_mps", incoming_linear_velocity_world_mps, 3),
        ("incoming_angular_velocity_world_radps", incoming_angular_velocity_world_radps, 3),
        ("landing_aim_xy_m", landing_aim_xy_m, 2),
        ("reference_raw_a_normal_w", reference_raw_a_normal_w, 3),
        ("reference_racket_quat_wxyz", reference_racket_quat_wxyz, 4),
        ("reference_racket_angular_velocity_w_radps", reference_racket_angular_velocity_w_radps, 3),
        ("reference_racket_site_velocity_w_mps", reference_racket_site_velocity_w_mps, 3),
        ("reference_racket_site_speed_mps", reference_racket_site_speed_mps, 0),
        ("mount_normal_sign", mount_normal_sign, 0),
        ("base_goal_xy_m", base_goal_xy_m, 2),
        ("time_to_contact_s", time_to_contact_s, 0),
        ("teacher_rate_min", teacher_rate_min, 0),
        ("teacher_rate_max", teacher_rate_max, 0),
        ("reference_t_hit_s", reference_t_hit_s, 0),
        ("reference_t_cycle_s", reference_t_cycle_s, 0),
        ("reaction_margin_s", reaction_margin_s, 0),
        ("attempt_close_margin_s", attempt_close_margin_s, 0),
        ("episode_length_s", episode_length_s, 0),
    ):
        _require_float_peer(
            value,
            rows=rows,
            width=width,
            name=name,
            dtype=dtype,
            device=device,
        )
    for name, value in (
        ("candidate_identity", candidate_identity),
        ("reveal_tick", reveal_tick),
        ("contact_tick", contact_tick),
    ):
        if (
            _flat_row_count(value, width=0, name=name) != rows
            or value.dtype is not torch.int64
            or value.device != device
        ):
            raise ValueError(f"{name} int64/device binding differs")

    face_offset_local = torch.zeros((rows, 3), dtype=dtype, device=device)
    face_offset_local[:, 0] = (
        _racket_geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M[0]
    )
    face_offset_local[:, 2] = (
        _racket_geometry.FACE_AREA_CENTER_XZ_FROM_SITE_M[1]
    )
    face_offset_local[:, 1] = torch.where(
        mount_normal_sign > 0.0,
        torch.full_like(
            mount_normal_sign, _racket_geometry.RED_OUTER_Y_FROM_SITE_M
        ),
        torch.full_like(
            mount_normal_sign, _racket_geometry.BLACK_OUTER_Y_FROM_SITE_M
        ),
    )
    quat_xyz = reference_racket_quat_wxyz[:, 1:]
    twice_cross = 2.0 * torch.cross(
        quat_xyz, face_offset_local, dim=1
    )
    face_offset_world = (
        face_offset_local
        + reference_racket_quat_wxyz[:, :1] * twice_cross
        + torch.cross(quat_xyz, twice_cross, dim=1)
    )
    reference_face_velocity = (
        reference_racket_site_velocity_w_mps
        + torch.cross(
            reference_racket_angular_velocity_w_radps,
            face_offset_world,
            dim=1,
        )
    ).contiguous()
    exact = _exact_face.solve_exact_face_timing_device(
        ball_contact_w_m=contact_position_env_m,
        racket_face_center_velocity_w_mps=reference_face_velocity,
        solved_raw_a_normal_w=reference_raw_a_normal_w,
        mount_normal_sign=mount_normal_sign,
        reference_racket_quat_wxyz=reference_racket_quat_wxyz,
        reference_racket_angular_velocity_w_radps=(
            reference_racket_angular_velocity_w_radps
        ),
        reference_racket_site_speed_mps=reference_racket_site_speed_mps,
        teacher_rate_min=teacher_rate_min,
        teacher_rate_max=teacher_rate_max,
        time_to_contact_s=time_to_contact_s,
        reference_t_hit_s=reference_t_hit_s,
        reference_t_cycle_s=reference_t_cycle_s,
        reaction_margin_s=reaction_margin_s,
        attempt_close_margin_s=attempt_close_margin_s,
        episode_length_s=episode_length_s,
    )

    normal_norm = torch.linalg.vector_norm(
        reference_raw_a_normal_w, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    physical_normal = (
        reference_raw_a_normal_w / normal_norm
    ) * mount_normal_sign[:, None]
    outgoing_velocity, outgoing_spin = _questions.predict_paddle_contact(
        incoming_linear_velocity_world_mps,
        reference_face_velocity,
        physical_normal,
        incoming_angular_velocity_world_radps,
        venue_params,
    )
    landing = _questions.coarse_landing(
        contact_position_env_m,
        outgoing_velocity,
        outgoing_spin,
        venue_params,
        surface_z=table_surface_z_m,
        net_x=net_x_m,
        h=integration_step_s,
        n_steps=integration_steps,
    )
    residual = torch.linalg.vector_norm(
        landing["land_xy"] - landing_aim_xy_m, dim=1
    )
    land_ok = (
        landing["land_valid"]
        & torch.isfinite(landing["land_xy"]).all(dim=1)
    )
    residual_ok = torch.isfinite(residual) & residual.lt(
        float(question_config.tol_m)
    )
    net_ok = (
        landing["net_valid"]
        & torch.isfinite(landing["net_z"])
        & landing["net_z"].gt(float(net_top_z_m))
    )
    normal_speed = -torch.sum(
        (incoming_linear_velocity_world_mps - reference_face_velocity)
        * physical_normal,
        dim=1,
    )
    contact_ok = (
        torch.isfinite(normal_speed)
        & normal_speed.ge(_questions.CONTACT_NORMAL_SPEED_MIN_MPS)
        & normal_speed.le(_questions.CONTACT_NORMAL_SPEED_MAX_MPS)
    )
    center_ok = land_ok & residual_ok & net_ok & contact_ok
    float_inputs_finite = (
        torch.isfinite(contact_position_env_m).all(dim=1)
        & torch.isfinite(incoming_linear_velocity_world_mps).all(dim=1)
        & torch.isfinite(incoming_angular_velocity_world_radps).all(dim=1)
        & torch.isfinite(landing_aim_xy_m).all(dim=1)
        & torch.isfinite(reference_raw_a_normal_w).all(dim=1)
        & torch.isfinite(reference_racket_quat_wxyz).all(dim=1)
        & torch.isfinite(reference_racket_angular_velocity_w_radps).all(dim=1)
        & torch.isfinite(reference_racket_site_velocity_w_mps).all(dim=1)
        & torch.isfinite(reference_racket_site_speed_mps)
        & torch.isfinite(mount_normal_sign)
        & torch.isfinite(base_goal_xy_m).all(dim=1)
        & torch.isfinite(time_to_contact_s)
        & torch.isfinite(teacher_rate_min)
        & torch.isfinite(teacher_rate_max)
        & torch.isfinite(reference_t_hit_s)
        & torch.isfinite(reference_t_cycle_s)
        & torch.isfinite(reaction_margin_s)
        & torch.isfinite(attempt_close_margin_s)
        & torch.isfinite(episode_length_s)
    )
    solver_fault = torch.where(
        float_inputs_finite,
        torch.zeros(rows, dtype=torch.int64, device=device),
        torch.full(
            (rows,),
            _questions.PRODUCER_FAULT_NONFINITE_PROPOSAL,
            dtype=torch.int64,
            device=device,
        ),
    ).contiguous()
    solver_reason = torch.zeros(
        rows, dtype=torch.int64, device=device
    )
    solver_reason = torch.where(
        land_ok, torch.full_like(solver_reason, 1), solver_reason
    )
    solver_reason = torch.where(
        land_ok & residual_ok & ~net_ok,
        torch.full_like(solver_reason, 5),
        solver_reason,
    )
    solver_reason = torch.where(
        land_ok & residual_ok & net_ok & ~contact_ok,
        torch.full_like(solver_reason, 7),
        solver_reason,
    )
    solver_reason = torch.where(
        center_ok,
        torch.full_like(solver_reason, CONSTRUCTION_REASON_ADMITTED),
        solver_reason,
    ).contiguous()

    physical_batch = _physical.PhysicalQuestionCandidateBatch(
        candidate_identity=candidate_identity,
        contact_position_env_m=contact_position_env_m,
        incoming_linear_velocity_world_mps=incoming_linear_velocity_world_mps,
        incoming_angular_velocity_world_radps=(
            incoming_angular_velocity_world_radps
        ),
    )
    physical = _physical.solve_max_final_segment_device(
        physical_batch,
        candidate_identity=candidate_identity,
        reveal_tick=reveal_tick,
        contact_tick=contact_tick,
        params=physical_params,
        config=physical_config,
    )
    exact_reason = exact.construction_reason.contiguous()
    physical_reason = physical.construction_reason.contiguous()
    reason = _reason_precedence(
        solver_reason, exact_reason, physical_reason
    )
    producer_fault = torch.bitwise_or(
        torch.bitwise_or(solver_fault, exact.producer_fault_bits),
        physical.producer_fault,
    ).contiguous()
    reason = torch.where(
        producer_fault.ne(0),
        torch.full_like(reason, CONSTRUCTION_REASON_INVALID_PRODUCER),
        reason,
    ).contiguous()
    admitted = reason.eq(CONSTRUCTION_REASON_ADMITTED).contiguous()
    installable = admitted.unsqueeze(-1)
    motion = torch.stack(
        (
            time_to_contact_s,
            exact.teacher_rate,
            exact.scaled_t_hit_s,
            exact.scaled_t_cycle_s,
            exact.pre_swing_wait_s,
        ),
        dim=1,
    )
    racket = torch.cat(
        (
            exact.racket_site_target_w_m,
            exact.racket_site_velocity_w_mps,
            reference_raw_a_normal_w,
            contact_position_env_m,
            reference_face_velocity,
            exact.racket_command_quat_wxyz,
            base_goal_xy_m,
            incoming_linear_velocity_world_mps,
            incoming_angular_velocity_world_radps,
        ),
        dim=1,
    )
    motion = torch.where(
        installable & torch.isfinite(motion), motion, torch.zeros_like(motion)
    ).contiguous()
    racket = torch.where(
        installable & torch.isfinite(racket), racket, torch.zeros_like(racket)
    ).contiguous()
    physical_state = torch.where(
        installable & torch.isfinite(physical.physical_state_f32),
        physical.physical_state_f32,
        torch.zeros_like(physical.physical_state_f32),
    ).contiguous()
    return DeviceFixedActionQuestionResult(
        motion_task_f32=motion,
        racket_task_f32=racket,
        physical_state_f32=physical_state,
        contact_tick=physical.contact_tick,
        launch_tick=physical.launch_tick,
        chosen_horizon_ticks=physical.chosen_horizon_ticks,
        construction_reason=reason,
        admitted=admitted,
        solver_reason=solver_reason,
        exact_face_reason=exact_reason,
        physical_horizon_reason=physical.horizon_construction_reason,
        physical_reason=physical_reason,
        solver_producer_fault=solver_fault,
        exact_face_producer_fault=exact.producer_fault_bits.contiguous(),
        physical_horizon_producer_fault=(
            physical.horizon_producer_fault.contiguous()
        ),
        physical_producer_fault=physical.producer_fault.contiguous(),
        producer_fault=producer_fault,
        solver_residual_m=residual.contiguous(),
    )


__all__ = (
    "CONSTRUCTION_REASON_ADMITTED",
    "CONSTRUCTION_REASON_INVALID_PRODUCER",
    "DeviceFixedActionQuestionResult",
    "construct_reference_center_question_device",
    "solve_fixed_action_question_device",
)
