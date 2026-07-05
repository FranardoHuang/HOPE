"""One-shot build gate for the HITTER-footwork task variant (2026-07-05).

Builds HOPE-PingPong-Hitter-AgibotA3-v0 headless with 4 envs, validates the 177-D
hitter_footwork actor contract (term order + dims), steps a few frames, and prints
the base-station coupling sanity numbers (station − racket_target_xy should equal
−ref_reach_offset per clip, up to the sampled jitter).

    hope_isaac_py scripts/verify_hitter_task.py --motion-file artifacts/hope_forehand_hopex/motion.npz \
        artifacts/hope_backhand_hopex/motion.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion-file", nargs="+", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import whole_body_tracking.tasks  # noqa: F401  (registers the gym tasks)
from whole_body_tracking.tasks.tracking.actor_observation_contract import (
    validate_actor_observation_contract,
)

TASK = "HOPE-PingPong-Hitter-AgibotA3-v0"

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=4)
env_cfg.commands.motion.motion_file = (
    args.motion_file if len(args.motion_file) > 1 else args.motion_file[0]
)
env = gym.make(TASK, cfg=env_cfg)

# 1) contract gate: loud failure prints the ACTUAL layout if term order/dims drifted.
contract = validate_actor_observation_contract(env.unwrapped, "hitter_footwork")
print(f"[hitter-gate] contract OK: {contract.name} {contract.total_dim}D")

# 1b) print BOTH obs-group layouts with running offsets (the warm-start weight surgery needs the
# exact column index of base_target_pos_b in actor AND critic input layers).
om = env.unwrapped.observation_manager
for group in ("policy", "critic"):
    names = om.active_terms[group]
    dims = [t[0] if isinstance(t, tuple) else int(t) for t in om.group_obs_term_dim[group]]
    off = 0
    print(f"[hitter-gate] {group} layout ({sum(dims)}D):")
    for nm, d in zip(names, dims):
        print(f"    [{off:3d}:{off + d:3d}] {nm} ({d})")
        off += d

# 2) coupling sanity: base station − racket_xy == −ref_reach_offset[clip] ± jitter.
cmd = env.unwrapped.command_manager.get_term("racket_target")
assert cmd.cfg.base_couple_mode == "reference_reach", cmd.cfg.base_couple_mode
motion = env.unwrapped.command_manager.get_term("motion")
env.reset()
delta = cmd.base_target_pos_w - cmd.racket_target_pos_w[:, :2]
reach = cmd._ref_reach_offset_xy_per_clip[motion.clip_id]
jitter = delta + reach  # should be inside base_target_*_range
print(f"[hitter-gate] station-racket delta: {delta.cpu().numpy()}")
print(f"[hitter-gate] ref reach offsets   : {reach.cpu().numpy()}")
print(f"[hitter-gate] residual jitter     : {jitter.cpu().numpy()}")
xr = cmd.cfg.base_target_x_range
yr = cmd.cfg.base_target_y_range
ok = (
    (jitter[:, 0] >= xr[0] - 1e-5).all()
    and (jitter[:, 0] <= xr[1] + 1e-5).all()
    and (jitter[:, 1] >= yr[0] - 1e-5).all()
    and (jitter[:, 1] <= yr[1] + 1e-5).all()
)
print(f"[hitter-gate] jitter within ranges x{xr} y{yr}: {bool(ok)}")

# 3) a few random steps: finite obs incl. the new channel.
obs, _ = env.reset()
for _ in range(10):
    act = 0.05 * torch.randn(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    obs, *_ = env.step(act)
pol = obs["policy"]
print(f"[hitter-gate] policy obs shape {tuple(pol.shape)} finite={bool(torch.isfinite(pol).all())}")
assert pol.shape[-1] == 177
print("[hitter-gate] ALL PASS" if ok else "[hitter-gate] JITTER RANGE FAIL")
env.close()
app.close()
