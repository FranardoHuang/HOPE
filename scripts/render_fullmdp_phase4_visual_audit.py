#!/usr/bin/env python3
"""Render the sealed Phase4 teacher against its exact MuJoCo task.

This is a developer visual diagnostic, not a training, contact, landing, or
deployment gate.  It intentionally renders the physical reset pose, the
retargeted robot-FK teacher, and the solved contact ball in one fixed camera so
pose/task mismatches are visible before a long run is launched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
WBT_SCRIPTS = REPO / "hope_training" / "whole_body_tracking" / "scripts"
LANE = REPO / "hope_training" / "whole_body_tracking" / "mjlab_lane"
MDP = (
    REPO
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
)
for path in (WBT_SCRIPTS, LANE, MDP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mujoco_motion_player as motion_player  # noqa: E402
import mujoco_full_mdp_portable_question as portable_question  # noqa: E402
import action_ball_full_mdp_portable_catalog as portable_catalog  # noqa: E402
import racket_contact_geometry as racket_geometry  # noqa: E402


@dataclass(frozen=True)
class _VisualBinding:
    """Addresses needed to render the robot inside a namespaced scene."""

    root_qpos_adr: int
    root_dof_adr: int
    joint_ids: np.ndarray
    joint_qpos_adrs: np.ndarray
    joint_dof_adrs: np.ndarray
    body_ids: np.ndarray


def _resolve_unique_suffix(mujoco, model, object_type, count: int, name: str) -> int:
    """Resolve either the canonical name or its unique scene namespace suffix."""

    direct = int(mujoco.mj_name2id(model, object_type, name))
    if direct >= 0:
        return direct
    matches = [
        index
        for index in range(count)
        if (mujoco.mj_id2name(model, object_type, index) or "").endswith("/" + name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot uniquely resolve canonical MuJoCo name {name!r}: {matches}"
        )
    return int(matches[0])


def _bind_namespaced_robot(mujoco, model) -> _VisualBinding:
    """Bind the A3 portion of a court+robot+ball model by canonical names."""

    joint_ids = np.asarray(
        [
            _resolve_unique_suffix(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.njnt),
                name,
            )
            for name in motion_player.RUNTIME_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    body_ids = np.asarray(
        [
            _resolve_unique_suffix(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.nbody),
                name,
            )
            for name in motion_player.RUNTIME_BODY_NAMES
        ],
        dtype=np.int64,
    )
    if len(set(joint_ids.tolist())) != len(joint_ids):
        raise RuntimeError("runtime joint names do not bind bijectively")
    if len(set(body_ids.tolist())) != len(body_ids):
        raise RuntimeError("runtime body names do not bind bijectively")
    joint_types = np.asarray(model.jnt_type)[joint_ids]
    if np.any(joint_types != int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise RuntimeError("all 31 A3 runtime joints must be scalar hinges")
    root_candidates = np.flatnonzero(
        (np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE))
        & (np.asarray(model.jnt_bodyid) == int(body_ids[0]))
    )
    if root_candidates.shape != (1,):
        raise RuntimeError("canonical pelvis must own exactly one free joint")
    root_joint = int(root_candidates[0])
    return _VisualBinding(
        root_qpos_adr=int(model.jnt_qposadr[root_joint]),
        root_dof_adr=int(model.jnt_dofadr[root_joint]),
        joint_ids=joint_ids,
        joint_qpos_adrs=np.asarray(model.jnt_qposadr, dtype=np.int64)[joint_ids],
        joint_dof_adrs=np.asarray(model.jnt_dofadr, dtype=np.int64)[joint_ids],
        body_ids=body_ids,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_no_clobber(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quat_apply(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)


def _root_tilt(quaternion: np.ndarray) -> float:
    matrix = motion_player.quaternion_wxyz_to_matrix(quaternion)
    return math.acos(float(np.clip(matrix[2, 2], -1.0, 1.0)))


def _set_robot_state(mujoco, model, data, binding, state: dict) -> None:
    root = binding.root_qpos_adr
    data.qpos[root : root + 3] = state["root_pos"]
    data.qpos[root + 3 : root + 7] = state["root_quat"]
    data.qpos[binding.joint_qpos_adrs] = state["joint_pos"]
    data.qvel[:] = 0.0
    data.qvel[binding.joint_dof_adrs] = state["joint_vel"]
    mujoco.mj_forward(model, data)

    root_slice = slice(binding.root_dof_adr, binding.root_dof_adr + 6)
    jacp = np.zeros((3, int(model.nv)), np.float64)
    jacr = np.zeros((3, int(model.nv)), np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, int(binding.body_ids[0]))
    matrix = np.vstack((jacp[:, root_slice], jacr[:, root_slice]))
    nonroot = np.concatenate((jacp @ data.qvel, jacr @ data.qvel))
    root_linear = np.asarray(state["root_lin_vel"], np.float64).copy()
    root_angular = np.asarray(state["root_ang_vel"], np.float64)
    if state["body_lin_vel_point"] == motion_player.BODY_LIN_VEL_POINT:
        offset = (
            np.asarray(data.xipos[binding.body_ids[0]], np.float64)
            - np.asarray(data.xpos[binding.body_ids[0]], np.float64)
        )
        root_linear -= np.cross(root_angular, offset)
    target = np.concatenate((root_linear, root_angular))
    data.qvel[root_slice] = np.linalg.solve(matrix, target - nonroot)
    mujoco.mj_forward(model, data)


def _teacher_state(clip, frame: int, yaw: np.ndarray, translation: np.ndarray) -> dict:
    root_pos = _quat_apply(yaw, np.asarray(clip.body_pos_w[frame, 0], np.float64))
    root_quat = _quat_mul(yaw, np.asarray(clip.body_quat_w[frame, 0], np.float64))
    return {
        "root_pos": root_pos + translation,
        "root_quat": root_quat,
        "root_lin_vel": _quat_apply(
            yaw, np.asarray(clip.body_lin_vel_w[frame, 0], np.float64)
        ),
        "root_ang_vel": _quat_apply(
            yaw, np.asarray(clip.body_ang_vel_w[frame, 0], np.float64)
        ),
        "joint_pos": np.asarray(clip.joint_pos[frame], np.float64),
        "joint_vel": np.asarray(clip.joint_vel[frame], np.float64),
        "body_lin_vel_point": clip.body_lin_vel_point,
    }


def _ready_state(payload: dict) -> dict:
    ready = payload["physical_ready"]
    return {
        "root_pos": np.asarray(ready["root_pos_w_m"], np.float64),
        "root_quat": np.asarray(ready["root_quat_wxyz"], np.float64),
        "root_lin_vel": np.zeros(3, np.float64),
        "root_ang_vel": np.zeros(3, np.float64),
        "joint_pos": np.asarray(ready["joint_pos_rad"], np.float64),
        "joint_vel": np.asarray(ready["joint_vel_radps"], np.float64),
        "body_lin_vel_point": motion_player.BODY_LIN_VEL_POINT,
    }


def _joint_margin(model, data, binding) -> float:
    joint_ids = np.asarray(binding.joint_ids, np.int64)
    limited = np.asarray(model.jnt_limited)[joint_ids].astype(bool)
    ranges = np.asarray(model.jnt_range)[joint_ids]
    values = np.asarray(data.qpos)[binding.joint_qpos_adrs]
    margins = np.minimum(values - ranges[:, 0], ranges[:, 1] - values)
    return float(np.min(margins[limited]))


def _obstacle_contacts(mujoco, model, data) -> list[dict]:
    obstacles = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in ("court_table_top", "court_net")
    }
    robot = {
        geom
        for geom in range(int(model.ngeom))
        if int(model.geom_bodyid[geom]) != 0
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("robot/")
    }
    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & obstacles and pair & robot:
            rows.append(
                {
                    "geom1": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ),
                    "geom2": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ),
                }
            )
    return rows


def _frame_receipt(mujoco, model, data, binding, site_id, task, label, frame):
    jacp = np.zeros((3, int(model.nv)), np.float64)
    jacr = np.zeros((3, int(model.nv)), np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    site_position = np.asarray(data.site_xpos[site_id], np.float64).copy()
    site_velocity = jacp @ np.asarray(data.qvel, np.float64)
    site_matrix = np.asarray(data.site_xmat[site_id], np.float64).reshape(3, 3)
    raw_normal = site_matrix @ np.asarray(racket_geometry.face_normal_local(1))
    selected_ball = site_position + site_matrix @ np.asarray(
        racket_geometry.ball_center_from_site_local(1), np.float64
    )
    target_site = task[5:8]
    target_velocity = task[8:11]
    target_normal = task[11:14]
    target_ball = task[14:17]
    root_q = np.asarray(
        data.qpos[binding.root_qpos_adr + 3 : binding.root_qpos_adr + 7],
        np.float64,
    )
    normal_error = math.acos(
        float(np.clip(np.dot(raw_normal, target_normal), -1.0, 1.0))
    )
    return {
        "label": label,
        "frame_index": frame,
        "racket_site_w_m": site_position.tolist(),
        "site_target_error_m": float(np.linalg.norm(site_position - target_site)),
        "site_velocity_target_error_mps": float(
            np.linalg.norm(site_velocity - target_velocity)
        ),
        "selected_ball_target_error_m": float(
            np.linalg.norm(selected_ball - target_ball)
        ),
        "raw_normal_target_error_rad": normal_error,
        "root_tilt_rad": _root_tilt(root_q),
        "joint_limit_margin_min_rad": _joint_margin(model, data, binding),
        "robot_table_or_net_contacts": _obstacle_contacts(mujoco, model, data),
    }


def _render(renderer, data, camera, label: str, frame: int | None):
    renderer.update_scene(data, camera=camera)
    image = renderer.render().copy()
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(image)
        ImageDraw.Draw(pil).rectangle((8, 8, 440, 48), fill=(0, 0, 0))
        ImageDraw.Draw(pil).text(
            (18, 18), f"{label}  frame={frame if frame is not None else 'reset'}",
            fill=(255, 255, 255),
        )
        image = np.asarray(pil)
    except ImportError:
        pass
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjb", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-fps", type=int, default=25)
    args = parser.parse_args()
    if args.out.exists() or args.video_fps <= 0:
        raise SystemExit("output must be fresh and video-fps positive")
    args.out.mkdir(parents=True, exist_ok=False)

    binding_json = json.loads(args.binding.read_text(encoding="utf-8"))
    plant = binding_json["phase4_plant_mjb"]
    if _sha256(args.mjb) != plant["sha256"] or args.mjb.stat().st_size != plant["size_bytes"]:
        raise RuntimeError("render MJB differs from the Phase4 plant binding")
    ready_path = Path(binding_json["inputs"]["dynamic_ready"]["path"])
    if _sha256(ready_path) != binding_json["inputs"]["dynamic_ready"]["sha256"]:
        raise RuntimeError("dynamic-ready bytes differ")
    motion_path = REPO / binding_json["outputs"]["motion"]["path"]
    if _sha256(motion_path) != binding_json["outputs"]["motion"]["sha256"]:
        raise RuntimeError("motion bytes differ")

    import imageio.v2 as imageio
    import mujoco
    import torch

    table = portable_catalog.load_portable_action_center_table()
    row = table.fresh_action
    clip = motion_player.load_motion(motion_path)
    ready_payload = json.loads(ready_path.read_text(encoding="utf-8"))
    base = _ready_state(ready_payload)
    question = portable_question.build_center_question(
        torch=torch,
        row=row,
        base_position_scene=torch.tensor(base["root_pos"][None], dtype=torch.float32),
        base_quat_wxyz=torch.tensor(base["root_quat"][None], dtype=torch.float32),
        contact_reference_root_z_scene=float(clip.body_pos_w[0, 0, 2]),
        step_dt=portable_catalog.FRESH_POLICY_STEP_S,
        table_surface_z_scene=0.76,
    )
    task = question["task_f32"][0].cpu().numpy().astype(np.float64)
    yaw = question["teacher_source_to_task_yaw_wxyz"][0].cpu().numpy().astype(np.float64)
    translation = question["teacher_source_to_task_translation_scene"][0].cpu().numpy().astype(np.float64)

    model = mujoco.MjModel.from_binary_path(str(args.mjb))
    data = mujoco.MjData(model)
    model_binding = _bind_namespaced_robot(mujoco, model)
    site_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/right_racket")
    )
    ball_joint = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    )
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.0
    camera.elevation = -17.0
    camera.distance = 4.0
    camera.lookat[:] = (1.35, -0.05, 0.9)
    renderer = mujoco.Renderer(model, height=720, width=960)

    def stage(state, label, frame):
        _set_robot_state(mujoco, model, data, model_binding, state)
        data.qpos[ball_qpos : ball_qpos + 3] = task[14:17]
        data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        receipt = _frame_receipt(
            mujoco, model, data, model_binding, site_id, task, label, frame
        )
        return _render(renderer, data, camera, label, frame), receipt

    strike = int(binding_json["timing"]["policy_grid_hit_frame"])
    keyframes = {
        "reset_physical_ready": (base, None),
        "teacher_frame0_before_start": (_teacher_state(clip, 0, yaw, translation), 0),
        "playback_early": (_teacher_state(clip, min(32, strike), yaw, translation), min(32, strike)),
        "contact_frame": (_teacher_state(clip, strike, yaw, translation), strike),
        "recovery_final": (_teacher_state(clip, clip.n_frames - 1, yaw, translation), clip.n_frames - 1),
    }
    frame_receipts = {}
    for label, (state, frame) in keyframes.items():
        image, receipt = stage(state, label, frame)
        path = args.out / f"{label}.png"
        imageio.imwrite(path, image)
        receipt["path"] = path.name
        receipt["sha256"] = _sha256(path)
        frame_receipts[label] = receipt

    frames = []
    reset_image, _ = stage(base, "reset_physical_ready", None)
    frames.extend([reset_image] * max(1, args.video_fps // 2))
    stride = max(1, int(round(float(clip.fps) / args.video_fps)))
    for frame in range(0, clip.n_frames, stride):
        image, _ = stage(
            _teacher_state(clip, frame, yaw, translation), "teacher_playback", frame
        )
        frames.append(image)
    video_path = args.out / "phase4_teacher_task_fixed_camera.mp4"
    imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=8)
    renderer.close()

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    receipt = {
        "kind": "action_ball_full_mdp_phase4_visual_audit_v2",
        "diagnostic_unauthorized": True,
        "source_commit": source_commit,
        "mjb": {"sha256": plant["sha256"], "size_bytes": plant["size_bytes"]},
        "action": {
            "action_id": row.action_id,
            "action_uid": row.action_uid,
            "motion_sha256": row.motion_sha256,
            "frames": clip.n_frames,
            "strike_frame": strike,
        },
        "question": {
            "ttc_ticks": int(question["ttc_ticks"][0]),
            "teacher_rate": float(question["teacher_rate"][0]),
            "pre_swing_wait_s": float(question["pre_swing_wait_s"][0]),
            "task_f32": task.tolist(),
        },
        "fixed_camera": {
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
            "distance": camera.distance,
            "lookat": camera.lookat.tolist(),
        },
        "frames": frame_receipts,
        "video": {
            "path": video_path.name,
            "fps": args.video_fps,
            "frames": len(frames),
            "sha256": _sha256(video_path),
        },
        "non_claims": [
            "not a policy rollout",
            "not dynamic contact or landing success",
            "not promotion, export, deployment, or hardware evidence",
        ],
    }
    _write_json_no_clobber(args.out / "visual_audit_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
