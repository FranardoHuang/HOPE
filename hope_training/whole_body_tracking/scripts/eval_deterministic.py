"""Headless eval of a HOPE A3 WBC checkpoint at one or more ACTION-NOISE scales.

Mirrors scripts/probe_metric.py's proven headless rollout, but (1) sets up the UNIFIED 2-clip
motion (forehand + backhand) exactly like scripts/train.py, (2) dumps the FULL cmd.metrics dict
averaged over envs + the last `tail` steps, plus a termination/episode-length tally, and (3) can
inject a small fraction of the LEARNED per-joint std as action noise to probe whether the
deterministic mean policy needs dithering for robustness.

The base policy is rsl_rl's get_inference_policy() -> the distribution MEAN (deterministic, the
exported-ONNX / deployed path). noise_scale=s adds  s * learned_std * N(0,1)  to the mean
(s=0 -> pure mean; s=1.0 -> full training-rollout noise). Pass several scales to sweep them in
ONE sim process. `base_couple_blend` optionally overrides the weak base->racket Y coupling.

    hope_isaac_py scripts/eval_deterministic.py task=HOPEPingPong algo=ppo headless=true \
        num_envs=128 +steps=1200 +tail=400 +noise_scales=0.0,0.05,0.10,0.20 \
        checkpoint=.../model_32200.pt 'motion_file=[.../fh.npz,.../bh.npz]'
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import OmegaConf

from isaac_bank_exam_adapter import policy_observation_tensor
from train import _apply_task_overrides, _registry_clip_name
from vendor_a3_eval_profile import (
    DETERMINISTIC_RANKING_PROFILE,
    apply_vendor_a3_eval_profile,
)


def _resolve_motion_files(cfg):
    """Replicate train.py: clip0 = registry_name (forehand), clip1 = registry_name_2 (backhand)."""
    import pathlib
    import wandb

    def _get(c, k, default=None):
        try:
            v = c.get(k, default) if hasattr(c, "get") else getattr(c, k, default)
        except Exception:
            v = default
        return v

    # explicit local override wins. Accept a Hydra list ([a,b]), a python list/ListConfig, or a
    # comma-separated string — robust to however Hydra parsed the override.
    mf = cfg.get("motion_file", None)
    if mf:
        if isinstance(mf, (list, tuple)) or mf.__class__.__name__ == "ListConfig":
            files = [str(s) for s in mf if str(s)]
        else:
            files = [s for s in str(mf).split(",") if s]
        if files:
            print(f"[eval] motion files (explicit): {files}", flush=True)
            return files

    reg = cfg.registry_name if cfg.registry_name is not None else cfg.task.registry_name
    reg = str(reg)
    if ":" not in reg:
        reg += ":latest"
    api = wandb.Api()
    files = [str(pathlib.Path(api.artifact(reg).download()) / "motion.npz")]
    reg2 = _get(cfg, "registry_name_2", None) or _get(cfg.task, "registry_name_2", None)
    if reg2 is not None and str(reg2).strip() and str(reg2).lower() != "none":
        reg2 = str(reg2)
        if ":" not in reg2:
            reg2 += ":latest"
        files.append(str(pathlib.Path(api.artifact(reg2).download()) / "motion.npz"))
        print(f"[eval] UNIFIED 2-clip: clip0={reg}  clip1={reg2}", flush=True)
    return files


REPORT_ROWS = [
    ("strike_composite_success_exact", "strike_composite_success_exact"),
    ("  forehand (end-to-end)", "strike_composite_success_exact_forehand"),
    ("  backhand (end-to-end)", "strike_composite_success_exact_backhand"),
    ("racket_vel_error_exact_strike", "racket_vel_error_exact_strike"),
    ("racket_pos_error_exact_strike", "racket_pos_error_exact_strike"),
    ("strike_vel_pass_exact", "strike_vel_pass_exact"),
    ("strike_pos_pass_exact", "strike_pos_pass_exact"),
    ("strike_normal_pass_exact", "strike_normal_pass_exact"),
    # UNCONDITIONAL swing accounting (Phase A + 2026-07-03): falls count against these, unlike the
    # conditional composite above. NOTE: the episode-timeout boundary swing deflates completion — with
    # ~2.7 swings per 10 s episode a PERFECT policy reads ~0.6-0.7, not 1.0; compare runs, not to 1.0.
    ("swing_completion_rate", "swing_completion_rate"),
    ("pre_strike_fall_rate", "pre_strike_fall_rate"),
    ("post_strike_fall_rate", "post_strike_fall_rate"),
    ("  pre-strike falls forehand", "pre_strike_fall_rate_forehand"),
    ("  pre-strike falls backhand", "pre_strike_fall_rate_backhand"),
    ("  post-strike falls forehand", "post_strike_fall_rate_forehand"),
    ("  post-strike falls backhand", "post_strike_fall_rate_backhand"),
    # Auditability: MUST read 0.0000 in a gate eval (HER replay is train-only; forced off above).
    ("achieved_replay_frac", "achieved_replay_frac"),
    ("base_target_offset_norm", "base_target_offset_norm"),
    ("base_pos_error", "base_pos_error"),
    ("base_pos_error_pre_strike", "base_pos_error_pre_strike"),
    # CONTINUOUS RALLY drift gate (2026-07-07): per-swing displacement EMAs (completed swings only)
    # + follow-through braking speed + cumulative drift from origin. Rally candidate thresholds:
    # drift_fwd <= ~0.10 m/swing, dist_from_origin tail-mean <= ~0.35 m, post_strike_fall <= 1%.
    ("base_drift_per_swing", "base_drift_per_swing"),
    ("base_drift_fwd_per_swing", "base_drift_fwd_per_swing"),
    ("base_station_offset_at_swing_start", "base_station_offset_at_swing_start"),
    ("post_strike_base_speed_xy", "post_strike_base_speed_xy"),
    ("base_dist_from_origin", "base_dist_from_origin"),
    # Heading recovery is meaningful only when the corresponding count reaches the command
    # term's exact_success_min_count. A zero value with zero count means "not measured".
    ("base_heading_hold_expiry_count", "base_heading_hold_expiry_count"),
    ("base_heading_abs_at_swing_start", "base_heading_abs_at_swing_start"),
    ("heading_recovery_count (gate first)", "heading_recovery_count"),
    ("heading_recovery_spawn_yaw", "heading_recovery_spawn_yaw"),
    ("heading_recovery_expiry_yaw", "heading_recovery_expiry_yaw"),
    ("base_roll_deg", "base_roll_deg"),
    ("base_pitch_deg", "base_pitch_deg"),
    ("foot_slip_speed", "foot_slip_speed"),
    ("foot_contact_frac", "foot_contact_frac"),
    ("joint_torque_abs_max", "joint_torque_abs_max"),
]


def _run(cfg, simulation_app):
    import torch
    import gymnasium as gym

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import whole_body_tracking.tasks  # noqa: F401  -- registers the gym tasks
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)
    # Exact-strike pass/composite metrics are gated by a decayed min-sample guard
    # (RacketTargetCommandCfg.exact_success_min_count, default 50): the exact-strike frame fires only
    # ~once per swing per env, so with too few envs the EMA never reaches that count and the rates are
    # FORCED to 0 even when the policy hits perfectly (the *_error_exact_strike values are still valid).
    # ~48 envs is the floor to cross the guard; use 128/256 for a stable read.
    if num_envs < 256:
        print(
            f"[WARN] num_envs={num_envs} is low. exact-strike pass/composite metrics may be forced to 0 "
            "by exact_success_min_count. Use num_envs=128 or 256 for standard eval.",
            flush=True,
        )

    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    _apply_task_overrides(env_cfg, cfg.task, _registry_clip_name(cfg))
    vendor_eval_receipt = apply_vendor_a3_eval_profile(
        env_cfg, cfg.task, profile=DETERMINISTIC_RANKING_PROFILE
    )
    if vendor_eval_receipt is not None:
        print(
            "[eval] VENDOR_A3_EVAL_PROFILE_JSON "
            + json.dumps(
                vendor_eval_receipt,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
    env_cfg.sim.device = str(cfg.device)

    # HER achieved-target replay is TRAIN-ONLY: the eval gate must score the pure box target
    # distribution (deploy-matched, comparable across checkpoints). Without this, ~30% of eval targets
    # would be jittered copies of states this very policy just produced — systematically easier than
    # box targets — inflating the ranking metrics. Opt back in with +allow_achieved_replay=true.
    if hasattr(env_cfg.commands, "racket_target") and hasattr(
        env_cfg.commands.racket_target, "achieved_target_mix_prob"
    ):
        if not bool(cfg.get("allow_achieved_replay", False)):
            env_cfg.commands.racket_target.achieved_target_mix_prob = 0.0
            print("[eval] HER achieved-target replay DISABLED (achieved_target_mix_prob=0; "
                  "pass +allow_achieved_replay=true to keep the training mixture)", flush=True)

    # R14 retiming is TRAIN-ONLY: the eval gate scores the native-speed reference clock
    # (deploy-matched, comparable across checkpoints).
    if hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "speed_scale_range"):
        if tuple(env_cfg.commands.motion.speed_scale_range) != (1.0, 1.0):
            env_cfg.commands.motion.speed_scale_range = (1.0, 1.0)
            print("[eval] R14 retiming DISABLED (speed_scale_range=(1.0, 1.0))", flush=True)

    motion_files = _resolve_motion_files(cfg)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    # optional eval-time override of the weak base->racket Y coupling (set BEFORE env build)
    blend_override = cfg.get("base_couple_blend", None)
    if blend_override is not None:
        try:
            env_cfg.commands.racket_target.base_couple_blend = float(blend_override)
            print(f"[eval] base_couple_blend override -> {float(blend_override)}", flush=True)
        except Exception as e:
            print(f"[eval] WARN could not set base_couple_blend: {e}", flush=True)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        **runner_kwargs(OmegaConf.to_container(cfg.algo, resolve=True), str(cfg.task.experiment_name))
    )
    agent_cfg.device = str(cfg.device)
    dev = agent_cfg.device

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=dev)
    # Shape-tolerant load: pre-2026-07-03 checkpoints have the old (2-dim-wider) critic; eval only
    # needs the actor, so fall back to an actor-preserving partial load instead of dying.
    from whole_body_tracking.utils.ckpt_compat import load_actor_tolerant

    load_actor_tolerant(runner, str(cfg.checkpoint))
    det_policy = runner.get_inference_policy(device=env.unwrapped.device)  # MEAN action

    # learned per-action std (state-independent parameter on the ActorCritic)
    std_vec = None
    for attr in ("std", "action_std"):
        s = getattr(runner.alg.policy, attr, None)
        if s is not None:
            try:
                std_vec = s.detach().reshape(-1).to(dev)
                break
            except Exception:
                pass
    print(f"[eval] learned std: "
          f"{'mean=%.4f min=%.4f max=%.4f' % (float(std_vec.mean()), float(std_vec.min()), float(std_vec.max())) if std_vec is not None else 'NOT FOUND'}",
          flush=True)

    cmd = env.unwrapped.command_manager.get_term("racket_target")
    n_steps = int(cfg.get("steps", 1200))
    tail = int(cfg.get("tail", 400))

    # noise scales (fractions of the learned std). 0 = pure deterministic mean.
    raw = cfg.get("noise_scales", None)
    if raw is None:
        raw = cfg.get("noise_scale", 0.0)
    if isinstance(raw, (list, tuple)) or raw.__class__.__name__ == "ListConfig":
        scales = [float(x) for x in raw]
    else:
        scales = [float(x) for x in str(raw).split(",") if str(x).strip() != ""]

    def make_policy(scale):
        if scale <= 0.0 or std_vec is None:
            return det_policy

        def p(obs):
            mean = det_policy(obs)
            return mean + scale * std_vec * torch.randn_like(mean)

        return p

    def run_rollout(scale):
        policy = make_policy(scale)
        keys = None
        sums = {}
        nacc = 0
        n_term_early = 0
        n_term_timeout = 0
        ep_len_sum = 0.0
        ep_len_n = 0
        term_by_type = {}
        elen_ctr = torch.zeros(num_envs, device=dev)
        # reset INSIDE inference_mode: after a prior rollout's inference-mode stepping the env
        # buffers are "inference tensors", and reset()'s inplace writes are only legal in-mode.
        with torch.inference_mode():
            env.reset()
            obs = policy_observation_tensor(env.get_observations(), device=dev)
        step = 0
        while simulation_app.is_running() and step < n_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions.to(env.unwrapped.device))
                obs = policy_observation_tensor(obs, device=dev)
            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            for k, v in log.items():
                if "Termination" in k:
                    try:
                        term_by_type[k] = term_by_type.get(k, 0.0) + float(v)
                        term_by_type[k + "::n"] = term_by_type.get(k + "::n", 0) + 1
                    except Exception:
                        pass
            elen_ctr += 1
            d = dones.bool() if hasattr(dones, "bool") else dones
            nd = int(d.sum())
            if nd:
                ep_len_sum += float(elen_ctr[d].sum()); ep_len_n += nd
                elen_ctr[d] = 0
            try:
                tm = env.unwrapped.termination_manager
                time_outs = tm.time_outs.bool()
                n_term_timeout += int((d & time_outs).sum())
                n_term_early += int((d & ~time_outs).sum())
            except Exception:
                pass
            if step >= n_steps - tail:
                m = cmd.metrics
                if keys is None:
                    keys = [k for k, v in m.items() if hasattr(v, "float")]
                    sums = {k: 0.0 for k in keys}
                for k in keys:
                    sums[k] += float(m[k].float().mean())
                nacc += 1
            step += 1
        avg = {k: sums[k] / nacc for k in keys} if (keys and nacc) else {}
        type_means = {k: term_by_type[k] / max(term_by_type.get(k + "::n", 1), 1)
                      for k in term_by_type if not k.endswith("::n")}
        term_total = n_term_early + n_term_timeout
        return dict(
            avg=avg, type_means=type_means,
            terminated_rate=(n_term_early / term_total) if term_total else float("nan"),
            n_term_early=n_term_early, n_term_timeout=n_term_timeout,
            mean_ep_len=(ep_len_sum / ep_len_n) if ep_len_n else float("nan"),
        )

    results = []
    for sc in scales:
        print(f"\n[eval] >>> rollout noise_scale={sc} x learned_std", flush=True)
        results.append((sc, run_rollout(sc)))

    def eebp(r):
        for k, v in r["type_means"].items():
            if "ee_body_pos" in k:
                return v
        return float("nan")

    ck = os.path.basename(str(cfg.checkpoint))
    blab = blend_override if blend_override is not None else "default"
    print("\n" + "=" * 96, flush=True)
    print(f"EVAL | ckpt={ck} | blend={blab} | {num_envs} envs x {n_steps} steps (tail {tail})", flush=True)
    print("-" * 96, flush=True)
    print(f"{'metric':34s}" + "".join(f"{('ns=' + str(sc)):>15s}" for sc, _ in results), flush=True)
    print("-" * 96, flush=True)
    for label, key in REPORT_ROWS:
        print(f"{label:34s}" + "".join(f"{r['avg'].get(key, float('nan')):15.4f}" for _, r in results), flush=True)
    print(f"{'mean_episode_length':34s}" + "".join(f"{r['mean_ep_len']:15.2f}" for _, r in results), flush=True)
    print(f"{'terminated_rate':34s}" + "".join(f"{r['terminated_rate']:15.4f}" for _, r in results), flush=True)
    print(f"{'ee_body_pos term frac (envlog)':34s}" + "".join(f"{eebp(r):15.4f}" for _, r in results), flush=True)
    print(f"{'early/timeout dones':34s}" + "".join(f"{(str(r['n_term_early']) + '/' + str(r['n_term_timeout'])):>15s}" for _, r in results), flush=True)
    print("=" * 96 + "\n", flush=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="play")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=False)
    simulation_app = app_launcher.app
    try:
        _run(cfg, simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
