"""Verification harness for the deploy-parity actor observation contract.

Checks (pick one with --check):

  layout   A) build the `full` and deploy-parity envs, print each policy-obs term + dim, and
              assert full==180 / deploy_parity==175 and that the base-position terms are gone.
  rollout  B) build the deploy-parity env, step it, and assert the observation + reward stay finite
              (no NaN/Inf) for --steps steps.
  golden   D) build the deploy-parity env, capture {policy obs (175) + the raw state quantities the
              C++ deploy obs builder needs} to an .npz — the deploy-parity golden for pp_obs_builder.
  onnx     C) (no Isaac needed) load an exported policy .onnx and assert its actor input last-dim ==
              175. Run this AFTER exporting with: python scripts/play.py task=HOPEPingPongDeployParity
              algo=ppo checkpoint=<warm/trained ckpt>  (play.py writes policy.onnx next to the ckpt).

Isaac checks need a local reference clip: pass --motion-file <.../motion.npz> (the same npz train.py
pulls from the wandb registry; any HOPE clip works for a layout/finite check).

Examples:
    python scripts/verify_realsensor.py --check layout   --motion-file /path/motion.npz --headless
    python scripts/verify_realsensor.py --check rollout  --motion-file /path/motion.npz --headless --steps 100
    python scripts/verify_realsensor.py --check golden   --motion-file /path/motion.npz --headless --out logs/realsensor_golden.npz
    python scripts/verify_realsensor.py --check onnx     --onnx logs/.../policy.onnx
"""

from __future__ import annotations

import argparse
import sys

FULL_ID = "HOPE-PingPong-AgibotA3-v0"
DEPLOY_ID = "HOPE-PingPong-DeployParity-AgibotA3-v0"
EXPECT_FULL = 180
EXPECT_REAL = 175


# --------------------------------------------------------------------------- #
# C) ONNX input-dim check — pure onnx, no Isaac. Gated BEFORE the AppLauncher.
# --------------------------------------------------------------------------- #
def check_onnx(path: str, expect: int) -> int:
    import onnx

    model = onnx.load(path)
    shapes = {}
    for inp in model.graph.input:
        dims = [d.dim_value if (d.dim_value > 0) else d.dim_param for d in inp.type.tensor_type.shape.dim]
        shapes[inp.name] = dims
    print(f"[onnx] {path}")
    for name, dims in shapes.items():
        print(f"  input '{name}': {dims}")
    ok = any(isinstance(dims[-1], int) and dims[-1] == expect for dims in shapes.values() if dims)
    print(f"[onnx] expect an actor input whose last dim == {expect}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--check")
_pre.add_argument("--onnx")
_pre.add_argument("--expect-dim", type=int, default=EXPECT_REAL)
_known, _ = _pre.parse_known_args()
if _known.check == "onnx":
    if not _known.onnx:
        print("--onnx PATH is required for --check onnx")
        sys.exit(2)
    sys.exit(check_onnx(_known.onnx, _known.expect_dim))


# --------------------------------------------------------------------------- #
# Isaac path (layout / rollout / golden).
# --------------------------------------------------------------------------- #
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--check", choices=["layout", "rollout", "golden"], required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--motion-file", default=None, help="local reference clip (motion.npz) for the command term")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--out", default="logs/realsensor_golden.npz")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402  (registers the gym ids)


def build_env(task_id: str, num_envs: int):
    cfg = parse_env_cfg(task_id, num_envs=num_envs)
    if args.motion_file:
        cfg.commands.motion.motion_file = args.motion_file
    return gym.make(task_id, cfg=cfg)


def policy_layout(env):
    om = env.unwrapped.observation_manager
    dims = om.group_obs_term_dim["policy"]  # list[tuple]
    total = om.group_obs_dim["policy"]
    total = total[0] if isinstance(total, (tuple, list)) else total
    names = getattr(om, "active_terms", {}).get("policy", None)
    if names is None:
        names = [f"term{i}" for i in range(len(dims))]
    flat = [(d[0] if isinstance(d, (tuple, list)) else d) for d in dims]
    return names, flat, int(total)


def print_layout(tag, names, dims):
    print(f"\n[{tag}] policy observation terms (order = obs layout):")
    o = 0
    for n, d in zip(names, dims):
        print(f"  [{o:3d}:{o + d:3d}] {n:24s} ({d})")
        o += d
    print(f"  TOTAL = {o}")
    return o


def run():
    if args.check == "layout":
        results = {}
        for tag, tid, expect in [("full", FULL_ID, EXPECT_FULL), ("deploy_parity", DEPLOY_ID, EXPECT_REAL)]:
            env = build_env(tid, 1)
            names, dims, total = policy_layout(env)
            got = print_layout(tag, names, dims)
            results[tag] = (set(names), got, expect)
            env.close()
        # assertions
        for tag, (names, got, expect) in results.items():
            assert got == expect, f"[{tag}] policy obs dim {got} != expected {expect}"
        removed = results["full"][0] - results["deploy_parity"][0]
        print(f"\n[layout] removed from actor: {sorted(removed)}")
        assert "motion_anchor_pos_b" in removed, "motion_anchor_pos_b should be removed from the actor"
        assert "base_target_pos_b" in removed, "base_target_pos_b should be removed from the actor"
        print("[layout] PASS — full=180, deploy_parity=175, base-position terms removed.")

    elif args.check == "rollout":
        env = build_env(DEPLOY_ID, args.num_envs)
        obs, _ = env.reset()
        n_act = env.action_space.shape[-1]
        bad = 0
        for t in range(args.steps):
            act = torch.zeros((args.num_envs, n_act), device=env.unwrapped.device)
            obs, rew, term, trunc, _ = env.step(act)
            pol = obs["policy"] if isinstance(obs, dict) else obs
            if not torch.isfinite(pol).all():
                bad += 1
                print(f"  step {t}: NON-FINITE obs ({(~torch.isfinite(pol)).sum().item()} elems)")
            if not torch.isfinite(rew).all():
                bad += 1
                print(f"  step {t}: NON-FINITE reward")
        env.close()
        print(f"[rollout] {args.steps} steps, deploy-parity env, {args.num_envs} envs: "
              f"{'PASS (all finite)' if bad == 0 else f'FAIL ({bad} non-finite events)'}")
        assert bad == 0

    elif args.check == "golden":
        env = build_env(DEPLOY_ID, 1)
        obs, _ = env.reset()
        act = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)
        obs, _, _, _, _ = env.step(act)
        om = env.unwrapped.observation_manager
        pol = (obs["policy"] if isinstance(obs, dict) else obs)[0].detach().cpu().numpy()
        names, dims, total = policy_layout(env)
        # raw state quantities the C++ deploy obs builder reconstructs the obs from:
        cmd = env.unwrapped.command_manager.get_term("racket_target")
        robot = env.unwrapped.scene["robot"]
        golden = dict(
            policy_obs=pol.astype(np.float32),
            term_names=np.array(names),
            term_dims=np.array(dims),
            base_quat_w=robot.data.root_quat_w[0].cpu().numpy().astype(np.float32),
            base_ang_vel_b=robot.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32),
            joint_pos=robot.data.joint_pos[0].cpu().numpy().astype(np.float32),
            joint_vel=robot.data.joint_vel[0].cpu().numpy().astype(np.float32),
            default_joint_pos=robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32),
            racket_pos_w=cmd.racket_pos_w[0].cpu().numpy().astype(np.float32),
            racket_target_pos_w=cmd.racket_target_pos_w[0].cpu().numpy().astype(np.float32),
            racket_target_vel_w=cmd.racket_target_vel_w[0].cpu().numpy().astype(np.float32),
        )
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        np.savez(args.out, **golden)
        env.close()
        print(f"[golden] wrote {args.out}  (policy obs dim = {total})")
        print("[golden] use this to validate the C++ pp_obs_builder real_sensor variant offline: feed the "
              "raw quantities into the builder and compare to policy_obs (per the realsensor_obs_reference layout).")
        assert total == EXPECT_REAL


try:
    run()
finally:
    simulation_app.close()
