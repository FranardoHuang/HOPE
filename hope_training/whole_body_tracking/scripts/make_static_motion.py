"""Generate a valid BeyondMimic motion .npz WITHOUT a retargeting pipeline or wandb.

This bootstraps a "stand at the default pose" reference clip so the WBC training loop
(scripts/train.py task=TrackingFlat motion_file=...) can run end to end on a machine that
has no motion data and no wandb account. It is a placeholder reference (a static stand),
NOT a real forehand/backhand swing; use it only to prove the training pipeline and to get
a first PPO loop / checkpoint / ONNX export.

The output schema matches scripts/csv_to_npz.py exactly (the canonical generator):
    fps           : [int]
    joint_pos     : [T, num_joints]   (robot native joint order)
    joint_vel     : [T, num_joints]
    body_pos_w    : [T, num_bodies, 3]
    body_quat_w   : [T, num_bodies, 4]  (wxyz)
    body_lin_vel_w: [T, num_bodies, 3]
    body_ang_vel_w: [T, num_bodies, 3]

Usage (inside the GPU/Isaac env, after source setup_train_env.sh):
    hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
        --output_file ../motions/a3_stand.npz --frames 600 --fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate a static default-pose motion npz (no CSV, no wandb).")
parser.add_argument("--robot", type=str, default="agibot_a3", choices=["g1", "agibot_a3"])
parser.add_argument("--output_file", type=str, required=True, help="Output .npz path.")
parser.add_argument("--frames", type=int, default=600, help="Number of frames (>= episode_length_s * fps).")
parser.add_argument("--fps", type=int, default=50, help="Motion fps (BeyondMimic canonical is 50).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless: this is a data-gen utility, never needs a window.
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.agibot_a3 import AGIBOT_A3_CFG
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

_ROBOT_CFG = {"g1": G1_CYLINDER_CFG, "agibot_a3": AGIBOT_A3_CFG}[args_cli.robot]


@configclass
class GenSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(GenSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]

    # Pin the robot at its default root + default joint pose, then let FK settle so
    # body_pos_w/body_quat_w reflect the real default configuration.
    root_state = robot.data.default_root_state.clone()
    root_state[:, :2] += scene.env_origins[:, :2]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    for _ in range(3):
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

    # Single-frame snapshot of the default pose (env 0).
    f_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()          # [J]
    f_body_pos = robot.data.body_pos_w[0].detach().cpu().numpy()          # [B, 3]
    f_body_quat = robot.data.body_quat_w[0].detach().cpu().numpy()        # [B, 4] wxyz

    T = int(args_cli.frames)
    J = f_joint_pos.shape[0]
    B = f_body_pos.shape[0]

    log = {
        "fps": np.array([args_cli.fps]),
        "joint_pos": np.repeat(f_joint_pos[None, :], T, axis=0).astype(np.float32),
        "joint_vel": np.zeros((T, J), dtype=np.float32),
        "body_pos_w": np.repeat(f_body_pos[None, ...], T, axis=0).astype(np.float32),
        "body_quat_w": np.repeat(f_body_quat[None, ...], T, axis=0).astype(np.float32),
        "body_lin_vel_w": np.zeros((T, B, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((T, B, 3), dtype=np.float32),
    }

    out = os.path.abspath(args_cli.output_file)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **log)
    print(f"[make_static_motion] robot={args_cli.robot} frames={T} joints={J} bodies={B} fps={args_cli.fps}")
    print(f"[make_static_motion] wrote static default-pose motion -> {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
