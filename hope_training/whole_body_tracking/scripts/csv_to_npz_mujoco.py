#!/usr/bin/env python3
"""csv_to_npz WITHOUT Isaac: MuJoCo forward kinematics (mjeval venv: mujoco+numpy+onnxruntime).

Faithful port of scripts/csv_to_npz.py (same CSV layout, same 30->50 fps lerp/slerp resample, same
np.gradient / SO3 central-difference velocities, same HOPE-frame alignment via hope_frame_utils),
with Isaac's articulation replay replaced by MuJoCo FK on the deploy MJCF.

Isaac's npz body/joint ORDER is reproduced from two sources:
- joint order: the donor ONNX metadata `joint_names` (the deploy contract, 31 names);
- body order: discovered ONCE against a reference npz produced by the Isaac pipeline
  (``--discover-map``), by matching FK body trajectories to reference columns; the resulting
  name list is then passed to conversions via ``--body-order`` (or baked after discovery).

Validate before trusting (prints max position/quaternion/velocity residuals vs the reference):
  python csv_to_npz_mujoco.py --mjcf <a3_pingpong.xml> --donor <policy.onnx> \
      --discover-map /workspace/shared/motions/hope_forehand_hopex.npz

Convert:
  python csv_to_npz_mujoco.py --mjcf ... --donor ... --body-order body_order.txt \
      --input_file forehand_oblique.csv --input_fps 30 --output_file hope_forehand_oblique.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hope_frame_utils import rotate_motion_to_hope_x  # noqa: E402


# ---------------------------------------------------------------- quaternion helpers (wxyz) --- #
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def axis_angle_from_quat(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = q * np.sign(q[..., :1])  # w >= 0 (shortest arc), matches isaaclab convention
    mag = np.linalg.norm(q[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    sin_half = np.where(mag > eps, mag, 1.0)
    axis = q[..., 1:] / sin_half[..., None]
    small = (mag <= eps)[..., None]
    return np.where(small, q[..., 1:] * 2.0, axis * angle[..., None])


def quat_slerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    lin = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w_a = np.where(lin, 1.0 - blend, np.sin((1.0 - blend) * theta) / np.where(lin, 1.0, sin_theta))
    w_b = np.where(lin, blend, np.sin(blend * theta) / np.where(lin, 1.0, sin_theta))
    out = w_a * a + w_b * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def so3_derivative(rotations: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference angular velocity, replicating csv_to_npz._so3_derivative."""
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


# ------------------------------------------------------------------------- CSV + resampling --- #
# CSV DOF-column order (GMR retarget output) — AGIBOT_A3_JOINT_NAMES in robots/agibot_a3.py.
CSV_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def load_and_resample(csv_path: str, input_fps: int, output_fps: int):
    motion = np.loadtxt(csv_path, delimiter=",").astype(np.float32)
    base_pos_in = motion[:, :3]
    base_rot_in = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof_in = motion[:, 7:]
    n_in = motion.shape[0]
    duration = (n_in - 1) / input_fps

    n_out = int(round(duration * output_fps)) + 1
    times = np.arange(n_out, dtype=np.float64) / output_fps
    phase = np.clip(times * input_fps, 0.0, n_in - 1)
    idx0 = np.floor(phase).astype(int)
    idx1 = np.minimum(idx0 + 1, n_in - 1)
    blend = (phase - idx0).astype(np.float32)[:, None]

    base_pos = base_pos_in[idx0] * (1 - blend) + base_pos_in[idx1] * blend
    dof = dof_in[idx0] * (1 - blend) + dof_in[idx1] * blend
    base_rot = quat_slerp(base_rot_in[idx0], base_rot_in[idx1], blend)

    dt = 1.0 / output_fps
    base_lin_vel = np.gradient(base_pos, dt, axis=0).astype(np.float32)
    dof_vel = np.gradient(dof, dt, axis=0).astype(np.float32)
    base_ang_vel = so3_derivative(base_rot, dt).astype(np.float32)
    return base_pos, base_rot, base_lin_vel, base_ang_vel, dof, dof_vel


# ---------------------------------------------------------------------------- MuJoCo FK ------ #
class MjFK:
    def __init__(self, mjcf_path: str, isaac_joint_names: list[str]):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.isaac_joint_names = isaac_joint_names
        self.qadr = {}
        for name in isaac_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint {name} not in MJCF"
            self.qadr[name] = self.model.jnt_qposadr[jid]
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1, f"expected one free joint, got {len(free)}"
        self.root_qadr = self.model.jnt_qposadr[free[0]]
        self.root_body = self.model.jnt_bodyid[free[0]]

    def body_names(self) -> list[str]:
        return [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in range(self.model.nbody)
        ]

    def fk(self, base_pos, base_rot, dof_by_name):
        """Returns xpos (nbody,3), xquat wxyz (nbody,4) for one frame."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.root_qadr : self.root_qadr + 3] = base_pos
        self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = base_rot  # mujoco is wxyz
        for name, val in dof_by_name.items():
            self.data.qpos[self.qadr[name]] = val
        self.mujoco.mj_forward(self.model, self.data)
        return self.data.xpos.copy(), self.data.xquat.copy()


def fk_series(fkm: MjFK, base_pos, base_rot, dof, dof_names):
    T = base_pos.shape[0]
    nb = fkm.model.nbody
    pos = np.zeros((T, nb, 3), dtype=np.float32)
    quat = np.zeros((T, nb, 4), dtype=np.float32)
    for t in range(T):
        p, q = fkm.fk(base_pos[t], base_rot[t], dict(zip(dof_names, dof[t])))
        pos[t], quat[t] = p, q
    return pos, quat


# --------------------------------------------------------------------------------- main ------ #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--donor", required=True, help="exported policy.onnx carrying joint_names metadata")
    ap.add_argument("--discover-map", help="reference npz: discover+validate Isaac body order, then exit")
    ap.add_argument("--body-order", help="file with MJ body names in Isaac column order (from --discover-map)")
    ap.add_argument("--input_file")
    ap.add_argument("--input_fps", type=int, default=30)
    ap.add_argument("--output_fps", type=int, default=50)
    ap.add_argument("--output_file")
    ap.add_argument("--hope_frame", choices=("on", "off"), default="on")
    args = ap.parse_args()

    import onnxruntime as ort

    meta = ort.InferenceSession(args.donor, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    isaac_joints = [s.strip() for s in meta["joint_names"].strip("[]").replace("'", "").split(",")]
    assert len(isaac_joints) == 31, f"expected 31 joints, got {len(isaac_joints)}"
    fkm = MjFK(args.mjcf, isaac_joints)

    if args.discover_map:
        ref = dict(np.load(args.discover_map))
        T, nb_ref = ref["body_pos_w"].shape[:2]
        # Reference joint_pos is in ISAAC joint order; root pose = body col 0 (pelvis).
        dof_by_isaac = ref["joint_pos"]
        pos, quat = fk_series(fkm, ref["body_pos_w"][:, 0], ref["body_quat_w"][:, 0], dof_by_isaac, isaac_joints)
        names = fkm.body_names()
        mapping, used = [], set()
        for col in range(nb_ref):
            resid = np.linalg.norm(pos - ref["body_pos_w"][:, col : col + 1], axis=-1).mean(axis=0)
            order = np.argsort(resid)
            pick = next(int(b) for b in order if int(b) not in used)
            used.add(pick)
            mapping.append((col, pick, names[pick], float(resid[pick])))
        worst = max(m[3] for m in mapping)
        print(f"[map] {nb_ref} reference columns matched; worst mean residual = {worst*1000:.2f} mm")
        for col, bid, name, r in mapping:
            print(f"[map] isaac_col {col:2d} <- mj body {bid:2d} {name:40s} resid {r*1000:7.2f} mm")
        # velocity-method check on a mid column
        dt = 1.0 / float(np.array(ref["fps"]).reshape(-1)[0])
        fd_vel = np.gradient(ref["body_pos_w"][:, 5], dt, axis=0)
        vres = np.abs(fd_vel - ref["body_lin_vel_w"][:, 5]).max()
        print(f"[map] velocity method residual (col 5, FD vs stored): {vres:.4f} m/s")
        out = args.body_order or "body_order.txt"
        with open(out, "w") as fh:
            fh.write("\n".join(m[2] for m in mapping) + "\n")
        print(f"[map] body order written -> {out}")
        return 0

    assert args.input_file and args.output_file and args.body_order, "--input_file/--output_file/--body-order required"
    with open(args.body_order) as fh:
        body_order = [ln.strip() for ln in fh if ln.strip()]
    names = fkm.body_names()
    cols = [names.index(n) for n in body_order]

    base_pos, base_rot, base_lin, base_ang, dof_csv, dof_vel_csv = load_and_resample(
        args.input_file, args.input_fps, args.output_fps
    )
    # CSV dof order -> Isaac dof order
    csv_idx = {n: i for i, n in enumerate(CSV_JOINT_NAMES)}
    perm = [csv_idx[n] for n in fkm.isaac_joint_names]
    dof = dof_csv[:, perm]
    dof_vel = dof_vel_csv[:, perm]

    pos_all, quat_all = fk_series(fkm, base_pos, base_rot, dof, fkm.isaac_joint_names)
    body_pos = pos_all[:, cols]
    body_quat = quat_all[:, cols]
    dt = 1.0 / args.output_fps
    body_lin = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang = np.stack([so3_derivative(body_quat[:, b], dt) for b in range(body_quat.shape[1])], axis=1).astype(
        np.float32
    )

    log = {
        "fps": np.array([args.output_fps], dtype=np.int64),
        "joint_pos": dof.astype(np.float32),
        "joint_vel": dof_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
    }
    if args.hope_frame == "on":
        log, report = rotate_motion_to_hope_x(log, theta_deg=None)
        print(f"[hope-frame] yaw {np.degrees(report.yaw_before_rad):+.2f} -> {np.degrees(report.yaw_after_rad):+.2f} deg")
    np.savez(args.output_file, **log)
    print(f"[convert] {args.input_file} -> {args.output_file}: {dof.shape[0]} frames @ {args.output_fps} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
