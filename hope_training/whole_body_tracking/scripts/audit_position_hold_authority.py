#!/usr/bin/env python3
"""这个姿态,位置控制器**撑得住吗**?—— 只问腰、双臂、脖子这一类关节。

人话:A3 的脚踩在地上,地面给的力可以帮腿分担重量;但**腰以上没有任何东西
能帮忙** —— 两只脚都不在腰关节带动的那截身体里,所以脚底的支撑力在腰上产生的
力矩恒为零。于是「撑住这个姿态,腰要出多大力」根本没有解算的余地:
它**就等于** MuJoCo 的 `qfrc_bias`(零速度、零加速度下的重力项)。

而策略这一侧只会发位置指令 `q_des`,实际力矩是 `kp * (q_des - q) - kd * qd`。
静止时 `qd = 0`,所以位置指令能产生的力矩上限是 `kp × (q_des 还能走多远)`,
再被电机限幅一刀。两者一比,就能直接回答"这个姿态能不能被 hold 住",
而且**不需要解任何 LP** —— 这是一条必要条件,不满足就一定 hold 不住。

本工具是**未授权诊断**:只出报告,不写 artifact、不改门限、不授权上机。

用法(pod;需要 mujoco;不需要 GPU):
    CUDA_VISIBLE_DEVICES= /workspace/hope_isaac_venv/bin/python \\
        hope_training/whole_body_tracking/scripts/audit_position_hold_authority.py \\
        --runtime-contract configs/.../bh_loop_c.shared_ready.training_contract.json \\
        --mjcf agi/.../a3_pingpong.xml \\
        --motion assets/motions/chingmu73_measured_v4_20260803/hope_Take_061_unit04_BH.npz \\
        --frames all --json out.json
    ... --library assets/motions/chingmu73_measured_v4_20260803 --frames 0
    ... --ready-artifact configs/.../take061...dynamic_ready.v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import materialize_a3_dynamic_ready_contract as dynamic  # noqa: E402


class PositionHoldAuditError(RuntimeError):
    """Any fail-closed refusal of this diagnostic."""


def _parse_frames(spec: str, frames: int) -> tuple[int, ...]:
    text = str(spec).strip().lower()
    if text == "all":
        return tuple(range(frames))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            low_text, _, high_text = part.partition("-")
            low, high = int(low_text), int(high_text)
            if low > high:
                raise PositionHoldAuditError(f"frame range {part!r} runs backwards")
            out.extend(range(low, high + 1))
        else:
            out.append(int(part))
    if not out:
        raise PositionHoldAuditError("no frames selected")
    for index in out:
        if not 0 <= index < frames:
            raise PositionHoldAuditError(
                f"frame {index} is outside this clip's 0..{frames - 1}"
            )
    return tuple(dict.fromkeys(out))


def executed_qdes_envelope(plant) -> tuple[np.ndarray, np.ndarray]:
    """The exact envelope the live runtime projects every ``q_des`` into."""

    hard_lower, hard_upper = dynamic._hard_inner_from_mechanical_limits(plant)
    qdes_limits = np.asarray(plant["qdes_limits"], np.float64)
    inset = float(plant["projection_inset"])
    span = qdes_limits[:, 1] - qdes_limits[:, 0]
    lower = np.maximum(qdes_limits[:, 0] + inset * span, hard_lower)
    upper = np.minimum(qdes_limits[:, 1] - inset * span, hard_upper)
    if np.any(lower >= upper):
        raise PositionHoldAuditError(
            "runtime projected-soft and hard-inner qdes envelopes do not intersect"
        )
    return lower, upper


def position_command_torque_interval(
    *,
    kp: np.ndarray,
    effort: np.ndarray,
    q_rad: np.ndarray,
    qdes_lower: np.ndarray,
    qdes_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Widest torque a pure position command can hold at ``q_rad``.

    ``tau = kp * (q_des - q)`` at zero velocity, capped by the motor limit.
    """

    kp = np.asarray(kp, np.float64)
    effort = np.asarray(effort, np.float64)
    q_rad = np.asarray(q_rad, np.float64)
    lower = np.maximum(-effort, kp * (np.asarray(qdes_lower, np.float64) - q_rad))
    upper = np.minimum(effort, kp * (np.asarray(qdes_upper, np.float64) - q_rad))
    return lower, upper


class _Model:
    """The exact MJCF plus the runtime-order row map, loaded once."""

    def __init__(self, mjcf: Path, joint_names: Sequence[str]):
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        dof_row: dict[str, int] = {}
        qpos_row: dict[str, int] = {}
        for joint in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if self.model.jnt_type[joint] in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                dof_row[name] = int(self.model.jnt_dofadr[joint])
                qpos_row[name] = int(self.model.jnt_qposadr[joint])
        missing = [name for name in joint_names if name not in dof_row]
        if missing:
            raise PositionHoldAuditError(
                f"exact MJCF is missing runtime joints {missing}"
            )
        self.dof_row = np.asarray([dof_row[n] for n in joint_names], np.int64)
        self.qpos_row = np.asarray([qpos_row[n] for n in joint_names], np.int64)
        self.contact_free = dynamic.contact_free_actuated_rows(
            self.model, self.dof_row
        )

    def required_hold_torque(self, q31, root_pos, root_quat) -> np.ndarray:
        qpos = np.zeros(int(self.model.nq), np.float64)
        qpos[:3] = np.asarray(root_pos, np.float64)
        quat = np.asarray(root_quat, np.float64)
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm <= 0.0:
            raise PositionHoldAuditError("root quaternion is not usable")
        qpos[3:7] = quat / norm
        qpos[self.qpos_row] = np.asarray(q31, np.float64)
        bias = dynamic.static_hold_required_generalized_force(self.model, qpos)
        return bias[self.dof_row]


def audit_pose(
    *,
    model: _Model,
    plant,
    q31: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    qdes_lower: np.ndarray,
    qdes_upper: np.ndarray,
) -> dict[str, Any]:
    """One pose: which contact-free joints cannot be held, and by how much."""

    kp = np.asarray(plant["kp"], np.float64)
    effort = np.asarray(plant["effort"], np.float64)
    required = model.required_hold_torque(q31, root_pos, root_quat)
    tau_lower, tau_upper = position_command_torque_interval(
        kp=kp, effort=effort, q_rad=q31,
        qdes_lower=qdes_lower, qdes_upper=qdes_upper,
    )
    records = dynamic.contact_free_hold_torque_shortfall(
        joint_names=list(plant["joint_names"]),
        contact_free=model.contact_free,
        required_nm=required,
        tau_lower_nm=tau_lower,
        tau_upper_nm=tau_upper,
        kp=kp,
        ready_q_rad=np.asarray(q31, np.float64),
        executed_qdes_lower_rad=qdes_lower,
        executed_qdes_upper_rad=qdes_upper,
        motor_effort_nm=effort,
    )
    gain_short = [
        r for r in records
        if r["binding_side"] != "pose_outside_executed_qdes_envelope"
    ]
    outside = [
        r for r in records
        if r["binding_side"] == "pose_outside_executed_qdes_envelope"
    ]
    return {
        "holdable_by_position_command": not records,
        "unreachable": records,
        "gain_short_joints": [r["joint"] for r in gain_short],
        "pose_outside_envelope_joints": [r["joint"] for r in outside],
        "worst_shortfall_nm": (
            max(abs(float(r["shortfall_nm"])) for r in gain_short)
            if gain_short else 0.0
        ),
    }


def _clip_frames(path: Path):
    payload = np.load(path)
    joint_pos = np.asarray(payload["joint_pos"], np.float64)
    if "body_pos_w" in payload:
        root_pos = np.asarray(payload["body_pos_w"][:, 0], np.float64)
        root_quat = np.asarray(payload["body_quat_w"][:, 0], np.float64)
    else:
        root_pos = np.asarray(payload["root_pos_w"], np.float64)
        root_quat = np.asarray(payload["root_quat_w"], np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 31:
        raise PositionHoldAuditError(f"{path.name} is not a 31-joint clip")
    return joint_pos, root_pos, root_quat


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    blocked = [r for r in rows if not r["holdable_by_position_command"]]
    gain: dict[str, int] = {}
    outside: dict[str, int] = {}
    for row in blocked:
        for name in row.get("gain_short_joints", ()):
            gain[name] = gain.get(name, 0) + 1
        for name in row.get("pose_outside_envelope_joints", ()):
            outside[name] = outside.get(name, 0) + 1
    return {
        "poses": len(rows),
        "holdable": len(rows) - len(blocked),
        "not_holdable": len(blocked),
        "poses_with_a_gain_shortfall": sum(
            1 for r in blocked if r.get("gain_short_joints")
        ),
        "poses_with_a_joint_outside_the_qdes_envelope": sum(
            1 for r in blocked if r.get("pose_outside_envelope_joints")
        ),
        "gain_short_joint_counts": dict(sorted(gain.items())),
        "outside_envelope_joint_counts": dict(sorted(outside.items())),
        "worst_shortfall_nm": (
            max(float(r["worst_shortfall_nm"]) for r in rows) if rows else 0.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--motion")
    parser.add_argument("--library")
    parser.add_argument("--ready-artifact", action="append", default=[])
    parser.add_argument("--frames", default="0")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if not args.motion and not args.library and not args.ready_artifact:
        raise PositionHoldAuditError(
            "choose at least one of --motion / --library / --ready-artifact"
        )

    contract = json.loads(Path(args.runtime_contract).read_text())
    plant = dynamic._runtime_plant(contract)
    names = list(plant["joint_names"])
    model = _Model(Path(args.mjcf), names)
    qdes_lower, qdes_upper = executed_qdes_envelope(plant)

    report: dict[str, Any] = {
        "kind": "a3_position_hold_authority_audit_v1",
        "diagnostic_unauthorized": True,
        "semantics": dynamic.CONTACT_FREE_HOLD_TORQUE_SEMANTICS,
        "contact_free_joints": [
            n for n, free in zip(names, model.contact_free) if bool(free)
        ],
        "ground_loaded_joints": [
            n for n, free in zip(names, model.contact_free) if not bool(free)
        ],
    }

    def run_clip(path: Path) -> dict[str, Any]:
        joint_pos, root_pos, root_quat = _clip_frames(path)
        selected = _parse_frames(args.frames, int(joint_pos.shape[0]))
        rows = []
        for index in selected:
            row = audit_pose(
                model=model, plant=plant, q31=joint_pos[index],
                root_pos=root_pos[index], root_quat=root_quat[index],
                qdes_lower=qdes_lower, qdes_upper=qdes_upper,
            )
            row["frame"] = int(index)
            rows.append(row)
        return {"frames": rows, "summary": _summarize(rows)}

    if args.motion:
        report["motion"] = {"path": str(args.motion), **run_clip(Path(args.motion))}
    if args.library:
        clips = sorted(Path(args.library).glob("hope_*.npz"))
        if not clips:
            raise PositionHoldAuditError(f"no hope_*.npz under {args.library}")
        per_clip = {}
        flat: list[dict[str, Any]] = []
        for clip in clips:
            got = run_clip(clip)
            per_clip[clip.name] = got["summary"]
            flat.extend(got["frames"])
        report["library"] = {
            "path": str(args.library),
            "clips": len(clips),
            "per_clip": per_clip,
            "summary": _summarize(flat),
        }
    if args.ready_artifact:
        arts = {}
        for path_text in args.ready_artifact:
            doc = json.loads(Path(path_text).read_text())
            physical = doc["physical_ready"]
            row = audit_pose(
                model=model, plant=plant,
                q31=np.asarray(physical["joint_pos_rad"], np.float64),
                root_pos=np.asarray(physical["root_pos_w_m"], np.float64),
                root_quat=np.asarray(physical["root_quat_wxyz"], np.float64),
                qdes_lower=qdes_lower, qdes_upper=qdes_upper,
            )
            row["birth_semantics"] = (
                doc.get("ready_source", {}).get("physical_birth_semantics")
            )
            arts[Path(path_text).name] = row
        report["ready_artifacts"] = arts

    text = json.dumps(report, indent=1, sort_keys=True, allow_nan=False)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
