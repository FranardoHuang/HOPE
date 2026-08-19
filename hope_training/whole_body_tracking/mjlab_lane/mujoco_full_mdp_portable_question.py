"""Portable fixed-action question and teacher construction for MuJoCo FullMDP.

This module deliberately owns no sampler, lifecycle, or outcome authority.  It
turns the already pinned slot-zero catalog row into the same three cold pieces
consumed by the MuJoCo lane:

* one centre question expressed from the live base yaw;
* one venue-model reverse-integrated launch state; and
* one measured motion teacher with the catalog's canonical retiming.

The reverse flight implementation is imported lazily from ``physical_ball``.
That helper is the engine-neutral Torch kernel used by Isaac's physical-ball
manager; keeping the import lazy lets host tests inject a small counterexample
kernel without importing Isaac or a GPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path


_JOINT_ORDER_CONTRACT_ID = "a3-gmr-dof-pos-to-runtime-articulation-v1"
_JOINT_ORDER_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "a3_joint_order_bijection_v1.json"
)


@dataclass(frozen=True)
class PortableMotionTeacher:
    """Device-resident measured reference for one catalog action."""

    fps: float
    strike_frame: int
    contact_reference_root_z_scene: float
    joint_pos: object
    joint_vel: object
    body_pos_w: object
    body_quat_w: object
    body_lin_vel_w: object
    body_ang_vel_w: object


def _plain_finite(value, label):
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(label + " must be one plain number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def _quat_mul_wxyz(torch, left, right):
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_apply_wxyz(torch, quaternion, vector):
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.cross(
        xyz, twice_cross, dim=-1
    )


def _yaw_quaternion(torch, yaw, dtype, device):
    result = torch.zeros((yaw.shape[0], 4), dtype=dtype, device=device)
    result[:, 0] = torch.cos(0.5 * yaw)
    result[:, 3] = torch.sin(0.5 * yaw)
    return result


def _yaw_from_quaternion(torch, quaternion):
    return torch.atan2(
        2.0
        * (
            quaternion[:, 0] * quaternion[:, 3]
            + quaternion[:, 1] * quaternion[:, 2]
        ),
        1.0
        - 2.0
        * (
            torch.square(quaternion[:, 2])
            + torch.square(quaternion[:, 3])
        ),
    )


def load_portable_motion_teacher(
    *, row, tracked_body_names, torch, dtype, device
):
    """Load the sealed measured motion selected by one portable catalog row."""

    import numpy as np

    with open(row.motion_file, "rb") as source:
        payload = source.read()
    if hashlib.sha256(payload).hexdigest() != row.motion_sha256:
        raise ValueError("portable teacher motion bytes differ from catalog")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_names",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "kinematics_schema_version",
            "body_pos_point",
            "body_lin_vel_point",
            "measured_racket_robot_mount_normal_sign",
            "measured_racket_joint_order_contract_id",
            "measured_racket_joint_order_contract_sha256",
        }
        if required.difference(data.files):
            raise ValueError("portable teacher NPZ schema differs")
        fps_values = np.asarray(data["fps"]).reshape(-1)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        names = tuple(str(value) for value in data["body_names"].tolist())
        try:
            local_order_sha = hashlib.sha256(
                _JOINT_ORDER_CONTRACT.read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise ValueError("portable teacher joint-order contract is absent") from exc
        if (
            fps_values.size != 1
            or not np.isfinite(fps_values).all()
            or float(fps_values[0]) <= 0.0
            or joint_pos.ndim != 2
            or joint_pos.shape[1] != 31
            or joint_vel.shape != joint_pos.shape
            or len(set(tracked_body_names)) != len(tracked_body_names)
            or any(names.count(name) != 1 for name in tracked_body_names)
            or int(
                np.asarray(
                    data["measured_racket_robot_mount_normal_sign"]
                ).reshape(-1)[0]
            )
            != int(row.mount_normal_sign)
            or int(np.asarray(data["kinematics_schema_version"]).reshape(-1)[0])
            != 2
            or str(np.asarray(data["body_pos_point"]).reshape(-1)[0])
            != "link_origin"
            or str(np.asarray(data["body_lin_vel_point"]).reshape(-1)[0])
            != "center_of_mass"
            or str(
                np.asarray(
                    data["measured_racket_joint_order_contract_id"]
                ).reshape(-1)[0]
            )
            != _JOINT_ORDER_CONTRACT_ID
            or str(
                np.asarray(
                    data["measured_racket_joint_order_contract_sha256"]
                ).reshape(-1)[0]
            )
            != local_order_sha
        ):
            raise ValueError("portable teacher identity/shape differs")
        body_indices = [names.index(name) for name in tracked_body_names]
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)[
            :, body_indices
        ]
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)[
            :, body_indices
        ]
        body_lin = np.asarray(data["body_lin_vel_w"], dtype=np.float32)[
            :, body_indices
        ]
        body_ang = np.asarray(data["body_ang_vel_w"], dtype=np.float32)[
            :, body_indices
        ]
        frames = joint_pos.shape[0]
        if (
            body_pos.shape != (frames, len(body_indices), 3)
            or body_quat.shape != (frames, len(body_indices), 4)
            or body_lin.shape != body_pos.shape
            or body_ang.shape != body_pos.shape
            or not np.allclose(
                np.linalg.norm(body_quat, axis=2),
                1.0,
                rtol=0.0,
                atol=2.0e-5,
            )
            or not all(
                np.isfinite(value).all()
                for value in (
                    joint_pos,
                    joint_vel,
                    body_pos,
                    body_quat,
                    body_lin,
                    body_ang,
                )
            )
        ):
            raise ValueError("portable teacher arrays differ or are nonfinite")
        strike = round(float(row.strike_phase) * (frames - 1))
        fps = float(fps_values[0])
        if tracked_body_names.count("pelvis_link") != 1:
            raise ValueError("portable teacher root body identity differs")
        pelvis_index = tracked_body_names.index("pelvis_link")
        if (
            not 0 <= strike < frames
            or not math.isclose(
                strike / fps,
                float(row.reference_t_hit_s),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            or not math.isclose(
                (frames - 1) / fps,
                float(row.reference_t_cycle_s),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        ):
            raise ValueError("portable teacher strike frame differs")

        def tensor(value):
            # np.load arrays may be read-only.  Copy before crossing the Torch
            # seam so the device tensor never aliases the archive buffer.
            return torch.tensor(
                np.array(value, copy=True), dtype=dtype, device=device
            )

        return PortableMotionTeacher(
            fps=fps,
            strike_frame=int(strike),
            contact_reference_root_z_scene=float(
                body_pos[strike, pelvis_index, 2]
            ),
            joint_pos=tensor(joint_pos),
            joint_vel=tensor(joint_vel),
            body_pos_w=tensor(body_pos),
            body_quat_w=tensor(body_quat),
            body_lin_vel_w=tensor(body_lin),
            body_ang_vel_w=tensor(body_ang),
        )


def sample_motion_teacher(torch, teacher, elapsed_s, teacher_rate, pre_swing_wait_s):
    """Sample MotionCommand's rounded measured-frame clock for each row."""

    active_s = torch.clamp(elapsed_s - pre_swing_wait_s, min=0.0)
    frame = torch.round(active_s * teacher_rate * float(teacher.fps)).to(
        torch.long
    )
    frame.clamp_(0, int(teacher.joint_pos.shape[0]) - 1)
    started = elapsed_s >= pre_swing_wait_s
    joint_vel = teacher.joint_vel[frame] * teacher_rate[:, None]
    body_lin = teacher.body_lin_vel_w[frame] * teacher_rate[:, None, None]
    body_ang = teacher.body_ang_vel_w[frame] * teacher_rate[:, None, None]
    return {
        "frame": frame,
        "started": started,
        "joint_pos": teacher.joint_pos[frame],
        "joint_vel": torch.where(started[:, None], joint_vel, torch.zeros_like(joint_vel)),
        "body_pos_w": teacher.body_pos_w[frame],
        "body_quat_w": teacher.body_quat_w[frame],
        "body_lin_vel_w": torch.where(
            started[:, None, None], body_lin, torch.zeros_like(body_lin)
        ),
        "body_ang_vel_w": torch.where(
            started[:, None, None], body_ang, torch.zeros_like(body_ang)
        ),
    }


def step_diagnostic_split_ready_qdes_bridge(
    *,
    torch,
    previous_qdes,
    frame0_qdes,
    frozen_steps,
):
    """Take one exact diagnostic q_des recurrence step toward frame zero.

    Isaac's split-ready Motion reference switches to measured frame zero at
    reveal and freezes there until playback.  Its diagnostic command consumer
    sends ``previous + (frame0 - previous) / (frozen_steps + 1)``.  This helper
    mirrors only that command recurrence.  It is deliberately not a Motion
    teacher/body sampler and currently has no production MuJoCo consumer.
    """

    if (
        tuple(previous_qdes.shape) != tuple(frame0_qdes.shape)
        or previous_qdes.ndim != 2
        or tuple(frozen_steps.shape) != (previous_qdes.shape[0],)
        or frozen_steps.dtype != torch.long
    ):
        raise ValueError("diagnostic q_des bridge tensors differ")
    torch._assert_async(
        torch.all(frozen_steps >= 0),
        "diagnostic q_des bridge frozen steps are negative",
    )
    span = (frozen_steps + 1).to(dtype=previous_qdes.dtype)[:, None]
    blended = previous_qdes + (frame0_qdes - previous_qdes) / span
    # Preserve an exact endpoint instead of accepting one ulp of subtract/add
    # cancellation when no frozen teacher step remains.
    qdes = torch.where(
        frozen_steps[:, None] == 0,
        frame0_qdes,
        blended,
    )
    torch._assert_async(torch.isfinite(qdes).all())
    return qdes


def build_center_question(
    *,
    torch,
    row,
    base_position_scene,
    base_quat_wxyz,
    contact_reference_root_z_scene,
    step_dt,
    table_surface_z_scene,
    back_integrate=None,
    venue_params=None,
    geometry=None,
    serve_horizon_s=None,
    backint_h=None,
    plane_margin=None,
):
    """Build the slot-centre task and exact final-ballistic launch state.

    ``base_position_scene`` and every position returned in ``task_f32`` are
    environment-local scene coordinates, matching the shared portable task
    ABI.  The MuJoCo consumer adds its row's ``env_origin`` only when writing
    the launch state into world-space qpos.

    The first reverse integration discovers the final segment above the table.
    The second integrates the exact integer number of control ticks that the
    lifecycle can schedule.  This mirrors the shared Physical manager's
    discover-then-serve rule and removes the old midpoint/linear launch.
    """

    if base_position_scene.ndim != 2 or tuple(base_position_scene.shape[1:]) != (3,):
        raise ValueError("portable question base position shape differs")
    if tuple(base_quat_wxyz.shape) != (base_position_scene.shape[0], 4):
        raise ValueError("portable question base quaternion shape differs")
    dt = _plain_finite(step_dt, "step_dt")
    if dt <= 0.0:
        raise ValueError("step_dt must be positive")
    shared_ball = None
    if (
        back_integrate is None
        or serve_horizon_s is None
        or backint_h is None
        or plane_margin is None
    ):
        import physical_ball

        shared_ball = physical_ball
    if back_integrate is None:
        back_integrate = shared_ball.back_integrate_incoming
    if serve_horizon_s is None:
        serve_horizon_s = shared_ball.SERVE_HORIZON_S
    if backint_h is None:
        backint_h = shared_ball.SERVE_BACKINT_H
    if plane_margin is None:
        plane_margin = shared_ball.SERVE_PLANE_MARGIN
    if venue_params is None:
        import virtual_ball

        venue_params = virtual_ball.load_venue_params()
    if geometry is None:
        import racket_contact_geometry

        geometry = racket_contact_geometry

    dtype = base_position_scene.dtype
    device = base_position_scene.device
    count = base_position_scene.shape[0]
    live_yaw = _yaw_from_quaternion(torch, base_quat_wxyz)
    live_yaw_q = _yaw_quaternion(torch, live_yaw, dtype, device)

    def repeated(values):
        return torch.tensor(values, dtype=dtype, device=device).expand(count, -1)

    base_goal = base_position_scene.clone()
    base_goal[:, :2] = repeated(row.base_spawn_center_w_xy_m)
    base_goal[:, 2] = _plain_finite(
        contact_reference_root_z_scene,
        "contact_reference_root_z_scene",
    )
    base_travel = repeated((*row.base_travel_center_b_yaw_xy_m, 0.0))
    base_goal += _quat_apply_wxyz(torch, live_yaw_q, base_travel)
    contact_offset = repeated(row.contact_offset_center_b_yaw_m)
    contact = base_goal + _quat_apply_wxyz(
        torch, live_yaw_q, contact_offset
    )
    incoming_direction = _quat_apply_wxyz(
        torch, live_yaw_q, repeated(row.incoming_direction_center_b_yaw)
    )
    incoming_velocity = incoming_direction * float(row.incoming_speed_center_mps)
    spin_direction = _quat_apply_wxyz(
        torch, live_yaw_q, repeated(row.spin_direction_center_b_yaw)
    )
    incoming_spin = spin_direction * float(row.spin_magnitude_center_radps)

    reference_root = repeated(row.reference_base_root_quat_wxyz)
    reference_yaw = _yaw_from_quaternion(torch, reference_root)
    delta_yaw_q = _yaw_quaternion(
        torch, live_yaw - reference_yaw, dtype, device
    )
    reference_quat = repeated(row.reference_racket_quat_wxyz)
    racket_quat = _quat_mul_wxyz(torch, delta_yaw_q, reference_quat)
    local_selected_normal = repeated(
        geometry.face_normal_local(int(row.mount_normal_sign))
    )
    racket_normal = _quat_apply_wxyz(
        torch, racket_quat, local_selected_normal
    )
    racket_normal = racket_normal / torch.linalg.vector_norm(
        racket_normal, dim=1, keepdim=True
    ).clamp_min(1.0e-8)

    # Selected ball centre relative to the official racket site comes from the
    # same dependency-light geometry module as the catalog's reference FK.
    ball_radius = float(geometry.BALL_RADIUS_M)
    if abs(float(venue_params.ball_radius) - ball_radius) > 1.0e-8:
        raise ValueError("venue and selected-face ball radius differ")
    ball_from_site_local = repeated(
        geometry.ball_center_from_site_local(int(row.mount_normal_sign))
    )
    racket_site_target = contact - _quat_apply_wxyz(
        torch, racket_quat, ball_from_site_local
    )

    reference_site_velocity = _quat_apply_wxyz(
        torch, delta_yaw_q, repeated(row.reference_racket_site_velocity_w_mps)
    )
    reference_omega = _quat_apply_wxyz(
        torch,
        delta_yaw_q,
        repeated(row.reference_racket_angular_velocity_w_radps),
    )
    face_from_site_local = repeated(
        geometry.face_center_from_site_local(int(row.mount_normal_sign))
    )
    face_from_site = _quat_apply_wxyz(
        torch, racket_quat, face_from_site_local
    )
    face_velocity = reference_site_velocity + torch.cross(
        reference_omega, face_from_site, dim=1
    )
    reference_speed = _plain_finite(
        row.reference_racket_site_speed_mps,
        "reference_racket_site_speed_mps",
    )
    rate_min = float(row.teacher_rate_min)
    rate_max = float(row.teacher_rate_max)
    required_speed = math.sqrt(
        sum(
            float(component) * float(component)
            for component in row.reference_racket_site_velocity_w_mps
        )
    )
    teacher_rate_value = geometry.canonical_teacher_rate_from_site_speed(
        required_speed,
        reference_speed,
        rate_min,
        rate_max,
    )
    teacher_rate = torch.full(
        (count,), teacher_rate_value, dtype=dtype, device=device
    )

    ttc_ticks = int(round(float(row.time_to_contact_center_s) / dt))
    if ttc_ticks < 1:
        raise ValueError("portable centre question has no contact tick")
    ttc = torch.full((count,), ttc_ticks * dt, dtype=dtype, device=device)
    scaled_hit = float(row.reference_t_hit_s) / teacher_rate
    scaled_cycle = float(row.reference_t_cycle_s) / teacher_rate
    pre_wait = ttc - scaled_hit
    if bool(
        (
            (pre_wait < float(row.reaction_margin_s) - 1.0e-6)
            | (pre_wait > 1.0 + 1.0e-6)
        ).any()
    ):
        raise ValueError("portable centre pre-swing wait is outside contract")

    discovery_horizon = torch.minimum(
        ttc,
        torch.full_like(
            ttc,
            _plain_finite(serve_horizon_s, "serve_horizon_s"),
        ),
    )
    _, _, maximum_horizon = back_integrate(
        contact,
        incoming_velocity,
        incoming_spin,
        discovery_horizon,
        venue_params,
        h=_plain_finite(backint_h, "backint_h"),
        surface_z=float(table_surface_z_scene),
        margin=_plain_finite(plane_margin, "plane_margin"),
    )
    horizon_ticks = torch.floor(maximum_horizon / dt + 1.0e-6).to(torch.long)
    if bool((horizon_ticks < 1).any()):
        raise ValueError("portable centre question has no launchable final segment")
    chosen_horizon = horizon_ticks.to(dtype=dtype) * dt
    launch_pos, launch_vel, actual_horizon = back_integrate(
        contact,
        incoming_velocity,
        incoming_spin,
        chosen_horizon,
        venue_params,
        h=_plain_finite(backint_h, "backint_h"),
        surface_z=float(table_surface_z_scene),
        margin=_plain_finite(plane_margin, "plane_margin"),
    )
    if bool((torch.abs(actual_horizon - chosen_horizon) > 1.0e-4).any()):
        raise ValueError("portable centre integer launch horizon truncated")

    task = torch.zeros((count, 45), dtype=dtype, device=device)
    task[:, :5] = torch.stack(
        (ttc, teacher_rate, scaled_hit, scaled_cycle, pre_wait), dim=1
    )
    racket = task[:, 5:32]
    racket[:, 0:3] = racket_site_target
    racket[:, 3:6] = reference_site_velocity
    racket[:, 6:9] = racket_normal
    racket[:, 9:12] = contact
    racket[:, 12:15] = face_velocity
    racket[:, 15:19] = racket_quat
    racket[:, 19:21] = base_goal[:, :2]
    racket[:, 21:24] = incoming_velocity
    racket[:, 24:27] = incoming_spin
    launch = task[:, 32:45]
    launch[:, 0:3] = launch_pos
    launch[:, 3] = 1.0
    launch[:, 7:10] = launch_vel
    launch[:, 10:13] = incoming_spin
    return {
        "task_f32": task,
        "launch_state_f32": launch,
        "ttc_ticks": torch.full(
            (count,), ttc_ticks, dtype=torch.long, device=device
        ),
        "launch_horizon_ticks": horizon_ticks,
        "teacher_rate": teacher_rate,
        "scaled_t_hit_s": scaled_hit,
        "scaled_t_cycle_s": scaled_cycle,
        "pre_swing_wait_s": pre_wait,
    }


__all__ = [
    "PortableMotionTeacher",
    "load_portable_motion_teacher",
    "sample_motion_teacher",
    "step_diagnostic_split_ready_qdes_bridge",
    "build_center_question",
]
