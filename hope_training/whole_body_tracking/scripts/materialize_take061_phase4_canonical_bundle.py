#!/usr/bin/env python3
"""Convert the Phase-4 Take061 diagnostic into a small runtime bundle.

The expensive retarget/search remains an offline producer.  This tool consumes
one already admitted Phase-4 report/NPZ and emits only the three artifacts the
ActionBall runtime understands: a schema-2 motion, a one-row schema-2 stroke
prototype, and a schema-3 action manifest.  It does not grant motion admission
or launch/train anything.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = HERE.parents[2]
MDP = HERE.parent / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MDP))

import canonical_schema2_builder as schema2
import action_ball_manifest as manifest_contract


KIND = "take061_phase4_canonical_action_bundle_v1"
ACTION_ID = "take061_slow_block_phase4_v1"
FAMILY = "backhand"
SCOPE = "full"
HIT_FRAME = 48
POLICY_STEP_S = 0.02
EXPECTED_PHASE4_KIND = "take061_slow_block_exact_face_phase4_v1"


class BundleError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               BundleError(f"{label} contains {token}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BundleError(f"{label} must be one JSON object")
    return value


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise BundleError(f"{label} must be finite with shape {shape}, got {result.shape}")
    return result


def _unit(value: Any, label: str) -> np.ndarray:
    result = _finite(value, (3,), label)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise BundleError(f"{label} has zero norm")
    return result / norm


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_to_heading(value: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray((c * value[0] + s * value[1],
                       -s * value[0] + c * value[1], value[2]), dtype=np.float64)


def _orthogonal_tangents(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(direction @ trial)) > 0.95:
        trial = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    u = np.cross(direction, trial)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    v /= np.linalg.norm(v)
    return u, v


def _rebuild_noncyclic_schema2(*, q: np.ndarray, qd: np.ndarray,
                               root_pos: np.ndarray, root_quat: np.ndarray,
                               fps: float, mjcf: Path, body_order: Path) -> tuple[bytes, Mapping[str, Any]]:
    """Use the canonical FK implementation without its cyclic-endpoint policy.

    Measured FullMDP clips are intentionally non-cyclic.  Reusing the builder's
    model binding/FK/serializer keeps wire semantics exact while avoiding a
    fabricated recovery segment that was never plant-certified by Phase 4.
    """
    mujoco = schema2._load_mujoco(None)
    model_path = mjcf.resolve()
    if model_path.suffix.lower() == ".mjb":
        model = mujoco.MjModel.from_binary_path(str(model_path))
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    body_names = schema2._read_body_order(body_order.resolve())
    # A runtime scene may contain a second free joint for the ball.  Bind the
    # robot root by the canonical first body instead of guessing "the only"
    # free joint; every scalar joint/body is still resolved by exact name.
    def resolve(obj_type: Any, count: int, name: str) -> int:
        direct = int(mujoco.mj_name2id(model, obj_type, name))
        if direct >= 0:
            return direct
        matches = [index for index in range(count)
                   if (mujoco.mj_id2name(model, obj_type, index) or "").endswith("/" + name)]
        if len(matches) != 1:
            raise BundleError(f"runtime model cannot uniquely resolve {name!r}")
        return int(matches[0])

    root_body_id = resolve(mujoco.mjtObj.mjOBJ_BODY, int(model.nbody), body_names[0])
    free_ids = [jid for jid in range(int(model.njnt))
                if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
                and int(model.jnt_bodyid[jid]) == root_body_id]
    if root_body_id < 0 or len(free_ids) != 1:
        raise BundleError("runtime model does not have one free joint on canonical root body")
    root_joint_id = int(free_ids[0])
    joint_ids = [resolve(mujoco.mjtObj.mjOBJ_JOINT, int(model.njnt), name)
                 for name in schema2.RUNTIME_JOINT_NAMES]
    body_ids = [resolve(mujoco.mjtObj.mjOBJ_BODY, int(model.nbody), name)
                for name in body_names]
    if min(joint_ids) < 0 or len(set(joint_ids)) != 31:
        raise BundleError("runtime model does not close all 31 named joints")
    if min(body_ids) < 0 or len(set(body_ids)) != 32:
        raise BundleError("runtime model does not close canonical 32-body order")
    binding = schema2._ModelBinding(
        root_joint_id=root_joint_id, root_body_id=root_body_id,
        root_qpos_adr=int(model.jnt_qposadr[root_joint_id]),
        root_dof_adr=int(model.jnt_dofadr[root_joint_id]),
        joint_ids=np.asarray(joint_ids, dtype=np.int64),
        joint_qpos_adrs=np.asarray([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int64),
        joint_dof_adrs=np.asarray([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int64),
        body_ids=np.asarray(body_ids, dtype=np.int64),
    )
    schema2._validate_joint_limits(model, binding, q)
    zeros = np.zeros((len(q), 3), dtype=np.float64)
    quat, quat_error = schema2._continuous_unit_quaternions(root_quat)
    body_pos, body_quat, body_lin, body_ang, twist_error, flips = schema2._rebuild_bodies(
        mujoco, model, binding, q, qd, root_pos, quat, zeros, zeros
    )
    arrays = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": q.astype(np.float32),
        "joint_vel": qd.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
        schema2.SCHEMA_KEY: np.asarray([schema2.KINEMATICS_SCHEMA_VERSION], dtype=np.int64),
        schema2.POS_POINT_KEY: np.asarray(schema2.BODY_POS_POINT),
        schema2.LIN_VEL_POINT_KEY: np.asarray(schema2.BODY_LIN_VEL_POINT),
        schema2.BODY_NAMES_KEY: np.asarray(body_names),
    }
    if frozenset(arrays) not in schema2.ALLOWED_KEYSETS:
        raise BundleError("internal schema-2 keyset mismatch")
    payload = schema2._deterministic_npz(arrays)
    return payload, {
        "frames": len(q), "fps": fps, "body_names": list(body_names),
        "quaternion_norm_error_max": quat_error,
        "root_twist_error_max": twist_error, "quaternion_sign_flips": flips,
        "cyclic_endpoint_required": False,
    }


def _prototype(*, motion_sha: str, frames: int, fps: float,
               ball_b: np.ndarray, face_b: np.ndarray,
               face_velocity_b: np.ndarray, mount_sign: int) -> Mapping[str, Any]:
    speed = float(np.linalg.norm(face_velocity_b))
    velocity_hat = face_velocity_b / speed
    elevation = math.degrees(math.asin(float(np.clip(velocity_hat[2], -1.0, 1.0))))
    row = {
        "motion_id": ACTION_ID, "scope": SCOPE, "family": FAMILY,
        "clip_index": 0, "npz_sha256": motion_sha, "frames": frames,
        "t_prepare_s": HIT_FRAME / fps, "t_prepare_min_s": HIT_FRAME / fps,
        "t_prepare_max_s": HIT_FRAME / fps,
        "band_b_x": [float(ball_b[0]), float(ball_b[0])],
        "band_b_y": [float(ball_b[1]), float(ball_b[1])],
        "band_z_w": [float(ball_b[2]), float(ball_b[2])],
        "slack_b_xy_m": 0.0, "slack_z_w_m": 0.0,
        "p_contact_b": ball_b.tolist(), "n_hat_b": face_b.tolist(),
        "face_sign": float(mount_sign), "priority": 0, "enabled": True,
        "strike_phase": HIT_FRAME / (frames - 1), "contact_frame": HIT_FRAME,
        "contact_window_frames": [HIT_FRAME, HIT_FRAME],
        "racket_face_center_velocity_hat_b": velocity_hat.tolist(),
        "racket_face_center_elevation_deg": elevation,
        "racket_face_center_window_dir_cone_deg": 0.0,
        "racket_face_center_speed_nominal_mps": speed,
        "racket_face_center_speed_max_mps": speed,
        "racket_face_center_speed_min_mps": speed,
        "racket_face_center_v_star_cap_mps": speed,
        "racket_face_center_v_dir_tol_deg": 0.0,
        "racket_face_center_cos_normal_velocity": float(velocity_hat @ face_b),
    }
    geometry_sha = manifest_contract._exact_face_geometry_source_sha256()
    document = {
        "schema_version": 2,
        "prototype_set_id": "take061_phase4_canonical_n1_v1",
        "velocity_contract": {
            "direction_and_speed_point": "selected_rubber_face_center",
            "policy_control_point": "official_racket_site",
            "mapping": "v_face_center=v_site+omega_world_cross_r_face_center_from_site_world",
            "site_velocity_authority": "centered_position_fd_half_window_2_clamped_per_clip",
            "angular_velocity_authority": "npz_body_ang_vel_w_at_right_wrist_yaw_Link",
            "direction_frame_authority": "canonical_ready_root_yaw_at_frame_0",
            "geometry_source_sha256": geometry_sha,
        },
        "contact_rule": "phase4_exact_face_center_singleton",
        "provenance": {"kind": KIND, "diagnostic_unauthorized": True},
        "scopes": {SCOPE: [row]},
    }
    document["derived_sha256"] = manifest_contract._prototype_canonical_sha256(document["scopes"])
    return document


def _singleton_profile(*, ball_b: np.ndarray, incoming_b: np.ndarray,
                       base_xy: np.ndarray, ttc: float) -> Mapping[str, Any]:
    direction = _unit(incoming_b, "incoming heading velocity")
    speed = float(np.linalg.norm(incoming_b))
    u, v = _orthogonal_tangents(direction)
    zero3, zero2 = [0.0, 0.0, 0.0], [0.0, 0.0]
    result = {
        "contact_offset_center_b_yaw_m": ball_b.tolist(),
        "contact_offset_std_lower_initial_m": zero3, "contact_offset_std_lower_max_m": zero3,
        "contact_offset_std_upper_initial_m": zero3, "contact_offset_std_upper_max_m": zero3,
        "contact_offset_min_b_yaw_m": ball_b.tolist(), "contact_offset_max_b_yaw_m": ball_b.tolist(),
        "time_to_contact_center_s": ttc, "time_to_contact_std_lower_initial_s": 0.0,
        "time_to_contact_std_lower_max_s": 0.0, "time_to_contact_std_upper_initial_s": 0.0,
        "time_to_contact_std_upper_max_s": 0.0, "time_to_contact_min_s": ttc,
        "time_to_contact_max_s": ttc,
        "incoming_direction_center_b_yaw": direction.tolist(),
        "incoming_direction_tangent_u_b_yaw": u.tolist(),
        "incoming_direction_tangent_v_b_yaw": v.tolist(),
        "incoming_inbound_axis_b_yaw": [-1.0, 0.0, 0.0], "incoming_inbound_min_cosine": 0.2,
        "incoming_speed_center_mps": speed, "incoming_speed_std_lower_initial_mps": 0.0,
        "incoming_speed_std_lower_max_mps": 0.0, "incoming_speed_std_upper_initial_mps": 0.0,
        "incoming_speed_std_upper_max_mps": 0.0,
        # Schema v3 registers the project's 0.4x lower support contract even
        # for an exact-centre diagnostic.  Sampler widths remain zero, so this
        # bundle constructs only the measured centre until rematerialized.
        "incoming_speed_min_mps": 0.4 * speed,
        "incoming_speed_max_mps": speed,
        "spin_direction_center_b_yaw": [0.0, 1.0, 0.0],
        "spin_direction_tangent_u_b_yaw": [0.0, 0.0, 1.0],
        "spin_direction_tangent_v_b_yaw": [1.0, 0.0, 0.0],
        "spin_magnitude_center_radps": 0.0, "spin_magnitude_std_lower_initial_radps": 0.0,
        "spin_magnitude_std_lower_max_radps": 0.0, "spin_magnitude_std_upper_initial_radps": 0.0,
        "spin_magnitude_std_upper_max_radps": 0.0, "spin_magnitude_min_radps": 0.0,
        "spin_magnitude_max_radps": 0.0,
        "base_spawn_center_w_xy_m": base_xy.tolist(),
        "base_spawn_std_lower_initial_m": zero2, "base_spawn_std_lower_max_m": zero2,
        "base_spawn_std_upper_initial_m": zero2, "base_spawn_std_upper_max_m": zero2,
        "base_spawn_min_w_xy_m": base_xy.tolist(), "base_spawn_max_w_xy_m": base_xy.tolist(),
        "base_travel_center_b_yaw_xy_m": zero2, "base_travel_std_lower_initial_m": zero2,
        "base_travel_std_lower_max_m": zero2, "base_travel_std_upper_initial_m": zero2,
        "base_travel_std_upper_max_m": zero2, "base_travel_min_b_yaw_xy_m": zero2,
        "base_travel_max_b_yaw_xy_m": zero2,
    }
    for prefix in ("incoming_direction", "spin_direction"):
        for axis in ("u", "v"):
            for side in ("neg", "pos"):
                result[f"{prefix}_tangent_{axis}_{side}_initial_deg"] = 0.0
                result[f"{prefix}_tangent_{axis}_{side}_max_deg"] = 0.0
    return result


def _ceil_to_policy_tick(seconds: float) -> float:
    """Keep the reveal-owned contact deadline on the FullMDP policy grid.

    Rounding upward preserves the Phase4 teacher-rate lower bound and the full
    reaction margin.  Rounding to nearest/down could make the exact-rate centre
    infeasible even though the offline motion/contact witness is unchanged.
    """

    if not math.isfinite(seconds) or seconds <= 0.0:
        raise BundleError("time-to-contact must be positive and finite")
    ticks = math.ceil(seconds / POLICY_STEP_S - 1.0e-12)
    return ticks * POLICY_STEP_S


def _materialize_profile_pins(
    template_path: Path, ball_physics: Path, repo_root: Path
) -> tuple[Mapping[str, Any], bytes]:
    template = deepcopy(dict(_read_json(template_path, "profile-pins template")))
    solver = template.get("solver_payload")
    physics = template.get("physics_payload")
    if not isinstance(solver, dict) or not isinstance(physics, dict):
        raise BundleError("profile-pins template lacks solver/physics payload")
    implementation = solver.get("implementation_source_sha256")
    if not isinstance(implementation, dict):
        raise BundleError("profile-pins template lacks implementation source map")
    source_paths = {
        "continuous_questions.py": MDP / "continuous_questions.py",
        "hope_commands.py": MDP / "hope_commands.py",
        "stroke_adapt_torch.py": MDP / "stroke_adapt_torch.py",
        "virtual_ball.py": MDP / "virtual_ball.py",
    }
    if set(implementation) != set(source_paths):
        raise BundleError("profile-pins implementation source keyset drift")
    for name, path in source_paths.items():
        implementation[name] = _sha256_file(path)
    venue = physics.get("venue_source")
    if not isinstance(venue, dict):
        raise BundleError("profile-pins physics venue_source missing")
    resolved_physics = ball_physics.resolve(strict=True)
    try:
        venue["path"] = resolved_physics.relative_to(repo_root).as_posix()
    except ValueError:
        venue["path"] = str(resolved_physics)
    venue["file_sha256"] = _sha256_file(resolved_physics)
    physics_sha = _sha256_bytes(json.dumps(
        physics, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    solver["physics_profile_sha256"] = physics_sha
    solver_sha = _sha256_bytes(json.dumps(
        solver, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    template["physics_profile_sha256"] = physics_sha
    template["solver_profile_sha256"] = solver_sha
    template["solver_implementation_source_sha256"] = implementation
    template["venue_yaml_sha256"] = venue["file_sha256"]
    return template, _json_bytes(template)


def materialize(args: argparse.Namespace) -> Mapping[str, Any]:
    root = args.repo_root.resolve(strict=True)
    report_path, npz_path, ready_path = args.phase4_report.resolve(), args.phase4_npz.resolve(), args.dynamic_ready.resolve()
    report = _read_json(report_path, "Phase4 report")
    ready = _read_json(ready_path, "dynamic ready")
    if (report.get("kind") != EXPECTED_PHASE4_KIND or report.get("diagnostic_unauthorized") is not True
            or report.get("exact_face_admitted") is not True or report.get("robust_curriculum_center") is not True):
        raise BundleError("Phase4 input is not the robust exact-face diagnostic")
    if report.get("artifact_payloads", {}).get("npz_sha256") != _sha256_file(npz_path):
        raise BundleError("Phase4 NPZ SHA does not match its report")
    with np.load(npz_path, allow_pickle=False) as archive:
        q = _finite(archive["q_ref"], (57, 31), "q_ref")
        qd = _finite(archive["qdot"], (57, 31), "qdot")
        site = _finite(archive["racket_site"], (57, 3), "racket_site")
        site_vel = _finite(archive["racket_velocity"], (57, 3), "racket_velocity")
        npz_ball = _finite(archive["ball_center_w_m"], (3,), "NPZ ball center")
        npz_site_target = _finite(
            archive["exact_racket_site_target_w_m"], (3,), "NPZ exact site target"
        )
        npz_site_velocity = _finite(
            archive["exact_racket_site_velocity_w_mps"], (3,), "NPZ exact site velocity"
        )
    physical = ready.get("physical_ready")
    if not isinstance(physical, Mapping):
        raise BundleError("dynamic ready lacks physical_ready")
    root0 = _finite(physical.get("root_pos_w_m"), (3,), "ready root position")
    quat0 = _finite(physical.get("root_quat_wxyz"), (4,), "ready root quaternion")
    frames = len(q)
    root_pos = np.repeat(root0[None, :], frames, axis=0)
    root_quat = np.repeat(quat0[None, :], frames, axis=0)
    effective_fps = (frames - 1) / float(report["timing"]["t_cycle_s"])
    motion_bytes, fk = _rebuild_noncyclic_schema2(
        q=q, qd=qd, root_pos=root_pos, root_quat=root_quat, fps=effective_fps,
        mjcf=args.mjcf, body_order=args.body_order,
    )
    motion_sha = _sha256_bytes(motion_bytes)
    yaw = _yaw_from_wxyz(quat0)
    exact = report["exact_face"]
    action_ball = report["action_ball"]
    ball_w = _finite(exact["ball_center_w_m"], (3,), "ball center")
    if not np.allclose(npz_ball, ball_w, rtol=0.0, atol=1.0e-7):
        raise BundleError("Phase4 report and NPZ ball center disagree")
    position_error = float(np.linalg.norm(npz_site_target - site[HIT_FRAME]))
    velocity_error = float(np.linalg.norm(npz_site_velocity - site_vel[HIT_FRAME]))
    if abs(position_error - float(exact["site_position_error_m"])) > 1.0e-6:
        raise BundleError("Phase4 exact-site position witness does not recompute")
    if abs(velocity_error - float(exact["site_velocity_error_mps"])) > 1.0e-6:
        raise BundleError("Phase4 exact-site velocity witness does not recompute")
    if (action_ball.get("analytic_landing_valid") is not True
            or action_ball.get("analytic_net_crossing_valid") is not True
            or float(action_ball.get("analytic_landing_error_m", math.inf)) >= 0.02):
        raise BundleError("Phase4 analytic contact continuation entrance is not valid")
    ball_b = _world_to_heading(ball_w - root0, yaw)
    incoming_w = _finite(action_ball["incoming_velocity_w_mps"], (3,), "incoming velocity")
    incoming_b = _world_to_heading(incoming_w, yaw)
    face_w = _unit(report["continuous_solver"]["signed_face_w"], "solver signed face")
    face_b = _world_to_heading(face_w, yaw)
    omega_w = _finite(exact["reference_racket_angular_velocity_w_radps"], (3,), "racket omega")
    offset_w = _finite(exact["face_center_from_site_w_m"], (3,), "face offset")
    face_velocity_w = site_vel[HIT_FRAME] + np.cross(omega_w, offset_w)
    face_velocity_b = _world_to_heading(face_velocity_w, yaw)
    prototype = _prototype(motion_sha=motion_sha, frames=frames, fps=effective_fps,
                           ball_b=ball_b, face_b=face_b,
                           face_velocity_b=face_velocity_b,
                           mount_sign=int(exact["mount_normal_sign"]))
    prototype_bytes = _json_bytes(prototype)
    prototype_sha = _sha256_bytes(prototype_bytes)
    profile_pins, profile_pins_bytes = _materialize_profile_pins(
        args.profile_pins_template, args.ball_physics, root
    )
    ball_physics_file_sha = _sha256_file(args.ball_physics)
    out_rel = Path(args.output_dir_rel)
    if out_rel.is_absolute() or ".." in out_rel.parts:
        raise BundleError("output-dir-rel must be normalized and repo-relative")
    motion_rel = out_rel / "take061_slow_block_phase4_v1.motion.npz"
    prototype_rel = out_rel / "take061_slow_block_phase4_v1.prototype.v2.json"
    manifest_rel = out_rel / "take061_slow_block_phase4_v1.action_ball.v3.json"
    profile_pins_rel = out_rel / "take061_slow_block_phase4_v1.profile_pins.v1.json"
    aim = _finite(action_ball["landing_aim_w_xy_m"], (2,), "landing aim")
    t_hit = float(report["timing"]["t_hit_s"])
    t_cycle = float(report["timing"]["t_cycle_s"])
    reaction = 0.1
    teacher_rate_min = float(exact["teacher_rate"])
    if not 0.0 < teacher_rate_min <= 1.0:
        raise BundleError("Phase4 exact-face teacher rate must lie in (0,1]")
    raw_ttc = t_hit / teacher_rate_min + reaction
    ttc = _ceil_to_policy_tick(raw_ttc)
    profile = _singleton_profile(ball_b=ball_b, incoming_b=incoming_b,
                                 base_xy=root0[:2], ttc=ttc)
    action_uid = manifest_contract.derive_action_ball_action_uid(ACTION_ID, FAMILY, motion_sha)
    solver_sha = profile_pins["solver_profile_sha256"]
    physics_sha = profile_pins["physics_profile_sha256"]
    manifest = {
        "schema_version": 3, "manifest_id": "take061_phase4_canonical_n1_v1",
        "mobility_mode": "no_move",
        "prototype": {"path": prototype_rel.as_posix(), "sha256": prototype_sha, "scope": SCOPE},
        "solver_profile_sha256": solver_sha, "physics_profile_sha256": physics_sha,
        "landing_aim": {"center_w_xy_m": aim.tolist(), "std_lower_initial_m": [0.0, 0.0],
            "std_lower_max_m": [0.0, 0.0], "std_upper_initial_m": [0.0, 0.0],
            "std_upper_max_m": [0.0, 0.0], "min_w_xy_m": aim.tolist(), "max_w_xy_m": aim.tolist()},
        "action_order": [ACTION_ID],
        "actions": [{"action_id": ACTION_ID, "action_uid": action_uid,
            "motion_path": motion_rel.as_posix(), "motion_sha256": motion_sha,
            "strike_phase": HIT_FRAME / (frames - 1), "reference_t_hit_s": t_hit,
            "reference_t_cycle_s": t_cycle,
            "reference_racket_site_speed_mps": float(np.linalg.norm(site_vel[HIT_FRAME])),
            "reaction_margin_s": reaction, "teacher_rate_min": teacher_rate_min,
            "teacher_rate_max": 1.0,
            "family": FAMILY, "mount_normal_sign": int(exact["mount_normal_sign"]),
            "ball_profile": profile}],
        "curriculum": {"min_proposals": 256, "min_safe_closed": 256,
            "target_failure_rate": 0.1, "failure_band_half_width": 0.025,
            "min_solver_admit_rate": 0.95, "min_install_rate": 0.95,
            "min_start_rate": 0.95, "min_close_rate": 0.95,
            "max_other_unsafe_rate": 0.02, "confidence_z": 1.96, "max_center_failures": 8},
        "holdout": {"seed": 20260830, "samples_per_action": 768,
                    "split_id": "take061_phase4_exact_center_v1"},
        "notes": "Diagnostic exact-center successor bundle; not motion admission or training authority.",
    }
    manifest_bytes = _json_bytes(manifest)
    binding = {
        "schema_version": 1, "kind": KIND, "diagnostic_unauthorized": True,
        "action_identity": {"action_id": ACTION_ID, "action_uid": action_uid,
                            "motion_sha256": motion_sha},
        "runtime_overrides": {
            "task.racket.action_ball_manifest_path": str(root / manifest_rel),
            "task.racket.action_ball_manifest_sha256": _sha256_bytes(manifest_bytes),
            "task.motion.motion_file": str(root / motion_rel),
            "fitted_ball_profile_pins": str(root / profile_pins_rel),
            "fitted_ball_profile_pins_sha256": _sha256_bytes(profile_pins_bytes),
            "HOPE_BALL_PHYSICS_YAML": str(args.ball_physics.resolve()),
        },
        "timing": {"t_hit_s": t_hit, "t_cycle_s": t_cycle, "fps": effective_fps,
                   "time_to_contact_raw_s": raw_ttc,
                   "time_to_contact_policy_grid_s": ttc,
                   "policy_step_s": POLICY_STEP_S},
        "exact_center": {"ball_center_w_m": ball_w.tolist(),
            "ball_center_b_yaw_m": ball_b.tolist(), "incoming_velocity_w_mps": incoming_w.tolist(),
            "landing_aim_w_xy_m": aim.tolist(),
            "solver_profile_sha256": solver_sha, "physics_profile_sha256": physics_sha,
            "geometry_source_sha256": exact["geometry_source_sha256"]},
        "construction_checks": {"phase4_robust": True,
            "analytic_landing_valid": action_ball["analytic_landing_valid"],
            "analytic_net_crossing_valid": action_ball["analytic_net_crossing_valid"],
            "exact_site_position_error_m": position_error,
            "exact_site_velocity_error_mps": velocity_error,
            "analytic_landing_error_m": action_ball["analytic_landing_error_m"],
            "fk": fk},
        "inputs": {"phase4_report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
            "phase4_npz": {"path": str(npz_path), "sha256": _sha256_file(npz_path)},
            "dynamic_ready": {"path": str(ready_path), "sha256": _sha256_file(ready_path)},
            "mjcf": {"path": str(args.mjcf.resolve()), "sha256": _sha256_file(args.mjcf)},
            "ball_physics": {
                "path": str(args.ball_physics.resolve()),
                "sha256": ball_physics_file_sha,
            }},
        "outputs": {"motion": {"path": motion_rel.as_posix(), "sha256": motion_sha},
            "prototype": {"path": prototype_rel.as_posix(), "sha256": prototype_sha},
            "manifest": {"path": manifest_rel.as_posix(), "sha256": _sha256_bytes(manifest_bytes)},
            "profile_pins": {"path": profile_pins_rel.as_posix(), "sha256": _sha256_bytes(profile_pins_bytes)}},
        "non_claims": ["formal motion admission", "backend contact", "policy", "deployment"],
    }
    binding_path = root / out_rel / "take061_slow_block_phase4_v1.binding.json"
    output_payloads = {root / motion_rel: motion_bytes, root / prototype_rel: prototype_bytes,
                       root / manifest_rel: manifest_bytes, root / profile_pins_rel: profile_pins_bytes}
    for path in output_payloads:
        if path.exists():
            raise BundleError(f"refusing to overwrite {path}")
    for path, payload in output_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    loaded = manifest_contract.load_action_ball_manifest(
        root / manifest_rel, expected_sha256=_sha256_bytes(manifest_bytes),
        verify_referenced_assets=True, repo_root=root,
    )
    binding["construction_checks"]["strict_manifest_and_assets"] = loaded.referenced_assets is not None
    binding_bytes = _json_bytes(binding)
    if binding_path.exists():
        raise BundleError(f"refusing to overwrite {binding_path}")
    fd = os.open(binding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(binding_bytes)
    return binding


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    result.add_argument("--phase4-report", type=Path, required=True)
    result.add_argument("--phase4-npz", type=Path, required=True)
    result.add_argument("--dynamic-ready", type=Path, required=True)
    result.add_argument("--mjcf", type=Path, required=True)
    result.add_argument("--body-order", type=Path, default=schema2.DEFAULT_BODY_ORDER_PATH)
    result.add_argument("--ball-physics", type=Path, default=REPO_ROOT_DEFAULT / "configs/ball_physics_venue.yaml")
    result.add_argument("--profile-pins-template", type=Path,
                        default=REPO_ROOT_DEFAULT / "configs/action_ball_profile_pins_20260728.json")
    result.add_argument("--output-dir-rel", required=True)
    return result


def main() -> int:
    result = materialize(parser().parse_args())
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
