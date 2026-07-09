"""One-shot build gate for the HITTER-PURE task variant (2026-07-07).

Builds HOPE-PingPong-HitterPure-AgibotA3-v0 headless with a few envs and validates the paper
alignment (arXiv:2508.21043) layer by layer:

  1. actor contract: 110-D `hitter_pure` (Table I exact — NO reference stream, NO swing_type),
     with both group layouts printed (offsets feed the R2 C++ obs builder).
  2. sampling (§V-B-1): station sampled independently inside base_target_*_range; racket target
     on the STATION-RELATIVE fixed plane (x == 0.70), per-clip non-overlapping y bands, absolute
     z bands; velocity inside the per-clip boxes.
  3. normal target (§IV-C): parallel to the sampled racket velocity (NOT the reference normal).
  4. reward stack: no hold/foot/HER-era terms; goal + upper-body imitation + generic reg only.
  5. no hold phase: hold_counter == 0 except stand-start settles (<= stand_start_min_hold).
  6. 10 random steps -> finite 110-D obs.

    hope_isaac_py scripts/verify_hitter_pure.py --motion-file artifacts/hope_forehand_hopex/motion.npz \
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

TASK = "HOPE-PingPong-HitterPure-AgibotA3-v0"
NUM_ENVS = 16

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
env_cfg.commands.motion.motion_file = (
    args.motion_file if len(args.motion_file) > 1 else args.motion_file[0]
)
env = gym.make(TASK, cfg=env_cfg)
failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"[pure-gate] {'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


# 1) contract gate + both layouts with running offsets.
contract = validate_actor_observation_contract(env.unwrapped, "hitter_pure")
print(f"[pure-gate] contract OK: {contract.name} {contract.total_dim}D")
om = env.unwrapped.observation_manager
for group in ("policy", "critic"):
    names = om.active_terms[group]
    dims = [t[0] if isinstance(t, tuple) else int(t) for t in om.group_obs_term_dim[group]]
    off = 0
    print(f"[pure-gate] {group} layout ({sum(dims)}D):")
    for nm, d in zip(names, dims):
        print(f"    [{off:3d}:{off + d:3d}] {nm} ({d})")
        off += d

# 2) sampling: station independent + station-relative fixed plane + per-clip bands.
cmd = env.unwrapped.command_manager.get_term("racket_target")
motion = env.unwrapped.command_manager.get_term("motion")
check("target_mode == hitter_pure", cmd.cfg.target_mode == "hitter_pure", cmd.cfg.target_mode)
env.reset()
origins = env.unwrapped.scene.env_origins
tol = 1e-4

station = cmd.base_target_pos_w - origins[:, :2]
xr, yr = cmd.cfg.base_target_x_range, cmd.cfg.base_target_y_range
check(
    "station inside independent box",
    bool(
        (station[:, 0] >= xr[0] - tol).all() and (station[:, 0] <= xr[1] + tol).all()
        and (station[:, 1] >= yr[0] - tol).all() and (station[:, 1] <= yr[1] + tol).all()
    ),
    f"x{xr} y{yr}, sampled x[{station[:,0].min():.3f},{station[:,0].max():.3f}] "
    f"y[{station[:,1].min():.3f},{station[:,1].max():.3f}]",
)

rel = cmd.racket_target_pos_w[:, :2] - cmd.base_target_pos_w  # station-relative xy
z_abs = cmd.racket_target_pos_w[:, 2] - origins[:, 2]
clip = motion.clip_id
boxes = cmd._pos_range_per_clip_t  # (2, 3, 2)
lo, hi = boxes[clip, :, 0], boxes[clip, :, 1]
check(
    "racket target on the FIXED station-relative plane (x)",
    bool((rel[:, 0] - lo[:, 0]).abs().max() < 1e-3),
    f"x_rel mean {rel[:, 0].mean():.3f} (expect {lo[0, 0]:.2f})",
)
check(
    "racket y inside the per-clip station-relative band",
    bool(((rel[:, 1] >= lo[:, 1] - tol) & (rel[:, 1] <= hi[:, 1] + tol)).all()),
)
check(
    "racket z inside the per-clip absolute band",
    bool(((z_abs >= lo[:, 2] - tol) & (z_abs <= hi[:, 2] + tol)).all()),
)
fh, bh = clip == 0, clip == 1
if fh.any() and bh.any():
    check(
        "fh/bh y bands non-overlapping (fh below bh)",
        bool(rel[fh, 1].max() < rel[bh, 1].min() + 1e-6),
        f"fh max {rel[fh, 1].max():.3f} < bh min {rel[bh, 1].min():.3f}",
    )
else:
    print(f"[pure-gate] WARN: only one clip sampled at reset (fh={int(fh.sum())}, bh={int(bh.sum())}) — rerun for the band check")

vlo, vhi = cmd._vel_range_per_clip_t[clip, :, 0], cmd._vel_range_per_clip_t[clip, :, 1]
vel = cmd.racket_target_vel_w
check(
    "racket velocity inside the per-clip box",
    bool(((vel >= vlo - tol) & (vel <= vhi + tol)).all()),
)

# 3) normal target parallel to the velocity (paper §IV-C), NOT the reference normal.
ncos = torch.nn.functional.cosine_similarity(cmd.racket_target_normal_w, vel, dim=-1)
check("normal ∥ velocity", bool((ncos > 0.9999).all()), f"min cos {ncos.min():.6f}")

# 4) reward stack audit. NOTE: zero-weight terms stay LISTED in active_terms (the RewardManager
# skips them at compute time) — racket_progress is declared at 0.0 as a fallback knob, so it is
# checked by weight, not absence.
rm = env.unwrapped.reward_manager
active = set(rm.active_terms)
check(
    "racket_progress declared but OFF (weight 0.0)",
    "racket_progress" in active and float(rm.get_term_cfg("racket_progress").weight) == 0.0,
)
must_absent = {
    "hold_ready", "base_decel", "pre_strike_foot_slip", "foot_slip_sq",
    "foot_velocity", "foot_drag", "foot_orientation", "arm_overreach", "arm_torque_saturation",
    "prestrike_waist_twist", "prestrike_upright", "strike_upright", "strike_ang_vel",
    "strike_foot_vel", "strike_vbob", "motion_global_anchor_pos",
}
must_present = {
    "racket_position", "racket_velocity", "racket_normal", "base_position", "racket_strike_success",
    "motion_global_anchor_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel",
    "motion_body_ang_vel", "action_rate_l2", "joint_limit", "undesired_contacts", "joint_torques",
    "joint_vel", "upright", "base_ang_vel_xy", "base_lin_vel_z",
}
check("no hold/foot/HER-era reward terms", not (active & must_absent), str(sorted(active & must_absent)))
check("goal+imitation+reg terms present", must_present <= active, str(sorted(must_present - active)))
check("no HER replay", float(cmd.cfg.achieved_target_mix_prob) == 0.0)

# 5) no hold phase (only stand-start settles allowed).
hc = motion.hold_counter
check(
    "hold_counter <= stand_start_min_hold",
    bool((hc <= int(motion.cfg.stand_start_min_hold)).all()),
    f"max {int(hc.max())}",
)
check("hold_steps_range == (0, 0)", tuple(motion.cfg.hold_steps_range) == (0, 0), str(motion.cfg.hold_steps_range))

# 6) random steps -> finite 110-D obs.
obs, _ = env.reset()
for _ in range(10):
    act = 0.05 * torch.randn(
        env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device
    )
    obs, *_ = env.step(act)
pol = obs["policy"]
check("policy obs 110-D and finite", pol.shape[-1] == 110 and bool(torch.isfinite(pol).all()), str(tuple(pol.shape)))

# NOTE: app.close() can swallow the process exit code (observed exiting 0 regardless) — the
# authoritative verdict is the ALL PASS / FAILURES line above, not $?.
print(f"[pure-gate] {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
env.close()
app.close()
raise SystemExit(0 if not failures else 1)
