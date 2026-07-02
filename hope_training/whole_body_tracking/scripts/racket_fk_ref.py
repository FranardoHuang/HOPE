"""Pure-numpy racket forward-kinematics reference for the 175-D deploy_parity obs.

racket_pos_pelvis(q_isaac_31) returns the racket (pingpang_red_Link) position
expressed in the PELVIS frame, from the 10-revolute-joint chain
pelvis_link -> right_wrist_yaw_Link plus the fixed pingpang mount offset.

World position is then:  base_pos_w + R(base_quat_w) @ racket_pos_pelvis(q).

This is the ground-truth reference the C++ pp_racket_fk.hpp must match.

Run inside the `grasping` distrobox:
    cd .../whole_body_tracking && source setup_train_env.sh
    hope_isaac_py scripts/racket_fk_ref.py
Gate: max FK error vs Isaac's actual pingpang_red_Link world pos < 1e-4 m.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Rotation helpers (double precision, right-handed).
# ---------------------------------------------------------------------------


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rot_rpy(r, p, y):
    """URDF fixed-axis roll-pitch-yaw = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _rot_axis(axis, q):
    if axis == "x":
        return _rot_x(q)
    if axis == "y":
        return _rot_y(q)
    if axis == "z":
        return _rot_z(q)
    raise ValueError(axis)


# ---------------------------------------------------------------------------
# The 10-joint chain: (isaac q-index, origin_xyz, origin_rpy, axis)
# ---------------------------------------------------------------------------
_CHAIN = [
    # #   idx  origin_xyz                                                   origin_rpy                       axis
    (2,  (0.0, 0.0, 0.0),                                                (0.0, 0.0, 0.0),                 "z"),  # waist_yaw
    (5,  (0.0, 0.0, 0.0),                                                (0.0, 0.0, 0.0),                 "x"),  # waist_roll
    (8,  (-0.0199999999998296, 0.0, 0.00500000000000012),               (0.0, 0.0, 0.0),                 "y"),  # waist_pitch
    (13, (0.03030000000207, -0.148224512812557, 0.283817906204843),     (-0.0872664625997163, 0.0, 0.0), "y"),  # r_shoulder_pitch
    (18, (0.0, -0.0589999999982601, 0.0),                               (0.0872664625997163, 0.0, 0.0),  "x"),  # r_shoulder_roll
    (22, (0.0, -0.0100000000003387, -0.147499999999977),                (0.0, 0.0, 0.0),                 "z"),  # r_shoulder_yaw
    (24, (0.00999999997356363, 0.0, -0.132999999990091),                (0.0, 0.0, 0.0),                 "y"),  # r_elbow
    (26, (0.129, 0.0, -0.00999999999980283),                            (0.0, 0.0, 0.0),                 "x"),  # r_wrist_roll
    (28, (0.0600000000000003, 0.0, 0.0),                                (0.0, 0.0, 0.0),                 "y"),  # r_wrist_pitch
    (30, (0.046, 0.0, 0.0),                                             (0.0, 0.0, 0.0),                 "z"),  # r_wrist_yaw
]

# Fixed mount offset: pingpang_red_Link origin relative to right_wrist_yaw_Link
# (right_hand_pingpang_joint is identity, so this is applied at wrist_yaw frame).
_MOUNT = np.array([0.21021, 0.032078, 0.032036], dtype=np.float64)


def racket_pos_pelvis(q_isaac_31) -> np.ndarray:
    """Racket position in the pelvis frame for a 31-vec of Isaac-order joint angles."""
    q = np.asarray(q_isaac_31, dtype=np.float64).reshape(-1)
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    for idx, xyz, rpy, axis in _CHAIN:
        # T_i = Translate(origin_xyz) * Rot_rpy(origin_rpy) * Rot_axis(q_i)
        Ri = _rot_rpy(rpy[0], rpy[1], rpy[2]) @ _rot_axis(axis, float(q[idx]))
        ti = np.array(xyz, dtype=np.float64)
        # accumulate: T <- T * T_i
        t = t + R @ ti
        R = R @ Ri
    # apply fixed mount point at the wrist_yaw frame
    return t + R @ _MOUNT


# ---------------------------------------------------------------------------
# Validation against Isaac. Must be run under the Isaac python (hope_isaac_py)
# so the AppLauncher / isaaclab imports resolve. The AppLauncher MUST be created
# before any isaaclab.* import (below), so all Isaac imports live inside _main.
# ---------------------------------------------------------------------------
def _quat_to_R(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _main():
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--clip", type=str, default=None,
                        help="run a single clip only (path to motion.npz); avoids the slow in-process "
                             "rebuild of a second Isaac env. Omit to run both hopex clips.")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    if not getattr(args, "headless", False):
        args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  (registers the gym ids)

    task_id = "HOPE-PingPong-DeployParity-AgibotA3-v0"
    clips = [args.clip] if args.clip else [
        "artifacts/hope_forehand_hopex/motion.npz",
        "artifacts/hope_backhand_hopex/motion.npz",
    ]

    max_err = 0.0
    q0_dump = None
    try:
        for clip in clips:
            cfg = parse_env_cfg(task_id, num_envs=args.num_envs)
            cfg.commands.motion.motion_file = clip
            env = gym.make(task_id, cfg=cfg)
            unwrapped = env.unwrapped
            device = unwrapped.device
            act_dim = env.action_space.shape[-1]

            env.reset()
            clip_err = 0.0
            for step in range(args.steps):
                act = 0.3 * torch.randn(unwrapped.num_envs, act_dim, device=device)
                env.step(act)

                robot = unwrapped.scene["robot"]
                q = robot.data.joint_pos.detach().cpu().numpy()           # (E,31) Isaac order
                base_pos = robot.data.root_pos_w.detach().cpu().numpy()   # (E,3)
                base_quat = robot.data.root_quat_w.detach().cpu().numpy()  # (E,4) wxyz

                cmd = unwrapped.command_manager.get_term("racket_target")
                gt = cmd.racket_pos_w.detach().cpu().numpy()              # (E,3) ground truth

                for e in range(unwrapped.num_envs):
                    p_pelvis = racket_pos_pelvis(q[e])
                    p_world = base_pos[e] + _quat_to_R(base_quat[e]) @ p_pelvis
                    err = float(np.linalg.norm(p_world - gt[e]))
                    clip_err = max(clip_err, err)
                    max_err = max(max_err, err)
                if q0_dump is None:
                    q0_dump = q[0].copy()

            print(f"[{clip}] max FK err over {args.steps} steps x {unwrapped.num_envs} envs = {clip_err:.3e} m")
            env.close()

        print(f"\n=== overall max FK error = {max_err:.3e} m ===")
        print("PASS" if max_err < 1e-4 else "FAIL (>=1e-4)")

        if q0_dump is not None:
            print("\n# sample q + expected racket_pos_pelvis (for C++ diff):")
            print("q_sample =", np.array2string(q0_dump, precision=10, separator=", ", max_line_width=100000))
            print("racket_pos_pelvis =", np.array2string(racket_pos_pelvis(q0_dump), precision=10, separator=", "))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    _main()
