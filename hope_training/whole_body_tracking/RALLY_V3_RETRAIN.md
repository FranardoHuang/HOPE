# Rally V3 retrain — autonomous heading recovery without wrecking strike precision

**Goal:** add autonomous post-swing heading self-recovery (robot re-squares to world +x during the
recovery hold, so the deploy runner re-engages with NO operator re-stand) to the 0.9931 HitterPure
striker — **without** the v2 precision collapse (0.9931 → 0.80).

**Warm-start (strict):**
`logs/rsl_rl/agibot_a3_hope_hitter_pure/2026-07-07_22-31-59_footfix/model_13200.pt` (composite 0.9931).

Root cause of the v2 collapse (design review + adversarial critique, 2026-07-08):
1. **`post_strike_brake` via GAE** was the killer — proven by model_18000 (collapsed 0.994→0.866 with
   *square* stands, *no* heading term, just brake+hold_ready). Brake income on the follow-through
   back-props into the swing action and damps commitment. → **brake = 0**.
2. **`foot_orientation -0.8` antagonizes the turn** (penalizes the hip-yaw deviation needed to
   re-square, and the hold reference is the square stand). → **hold-gate it off** (swing discipline kept).
3. **The recovery metric was dilution-blind.** → added spawn-conditioned `heading_recovery_expiry_yaw`.

## What changed in the repo (already applied)

| File | Change |
|---|---|
| `mdp/hope_rewards.py` | `foot_orientation_discipline(..., hold_gate=False)` — zeros the penalty during `in_hold` when True. No-op on hold-free tasks / default. |
| `scripts/train.py` | plumbs yaml `rewards.foot_orientation_hold_gate` → `foot_orientation.params["hold_gate"]`. |
| `mdp/hope_commands.py` | **additive** spawn-yaw-conditioned recovery metric: `heading_recovery_spawn_yaw`, `heading_recovery_expiry_yaw` (only holds that start >0.30 rad; expiry = the gate). No reward/termination logic touched. |
| `scripts/eval_deterministic.py` | REPORT_ROWS now print the two recovery metrics + the pooled one. |
| `cfg/task/HOPEPingPongHitterPureRallyV3.yaml` | the recipe (reuses the v2 gym task / env cfg). |
| `scripts/pp_recovery_eval.sh` | the G2 in-sim recovery eval (concentrates yawed holds). |

Recipe vs v2: brake 0.3→**0**, hold_ready 1.0→**0**, hold_heading 1.0→**0.10** (ramped), yaw
±0.9→**±0.2** (ramped), stand_prob 0.35→0.25, episode 16→10, **foot_orientation hold-gated**.

## Training — staged curriculum (all inside the `grasping` distrobox)

`cd .../hope_training/whole_body_tracking && source setup_train_env.sh` first. The operator supplies
`num_envs` (lineage default 4096). Trust-region anti-forgetting is applied via CLI on every stage:
`algo.algorithm.entropy_coef=0.006` (keeps std from re-inflating; desired_kl stays 0.01).

**Stage 1** — establish "brake=0 + hold-gate + hold_heading 0.10 + yaw ±0.2" holds precision (yaml as-is):
```
hope_isaac_py scripts/train.py task=HOPEPingPongHitterPureRallyV3 algo=ppo headless=true \
  num_envs=4096 algo.algorithm.entropy_coef=0.006 \
  checkpoint_path=logs/rsl_rl/agibot_a3_hope_hitter_pure/2026-07-07_22-31-59_footfix/model_13200.pt
```
Run ~400–800 iters, gating G1 every ~200 (below). Expected: composite stays ≥0.95 (it should barely
move — the perturbation is one 0.10 term, hold-disjoint from the swing).

**Stages 2→4** — ramp yaw + hold_heading, resuming from the last **G1-passing** checkpoint of the
previous stage. Edit the two values in the yaml (safest vs CLI list-override), then resume:

| Stage | `motion.stand_start_yaw_range` | `rewards.hold_heading_weight` |
|---|---|---|
| 2 | `[-0.35, 0.35]` | `0.30` |
| 3 | `[-0.6, 0.6]`  | `0.40` |
| 4 | `[-0.9, 0.9]`  | `0.50` |

Advance a stage **only after** G1 holds AND G2 `heading_recovery_expiry_yaw` is trending down. If G1
fails 3 consecutive gated checkpoints in a stage → roll back and halve that stage's `hold_heading_weight`.

> ⚠ `train.py` has **no in-train gate** — you kill training, eval on-disk checkpoints, and resume/roll
> back by hand. `save_interval=100`. This is manual over 1–2 days; budget ~2–3× a single blind retrain.

## Gates

**G1 — precision (HARD, first, every ~200 iters).** The v2 killer; check before anything else:
```
hope_isaac_py scripts/eval_deterministic.py task=HOPEPingPongHitterPure algo=ppo headless=true \
  num_envs=256 +steps=1200 +tail=400 +noise_scales=0.0 checkpoint=<rally_v3_ckpt>
```
PASS = `strike_composite_success_exact` ≥ 0.95 AND backhand ≥ 0.93 AND `pre_strike_fall_rate` ≤ 0.02.
Roll back to the last passing checkpoint on failure (allow a 2-eval grace for the resume value-refit dip).

**G2 — recovery (in-sim proxy).**
```
bash scripts/pp_recovery_eval.sh <rally_v3_ckpt>
```
PASS = `heading_recovery_expiry_yaw (GATE)` ≪ `spawn` AND < ~0.30 rad. (Proxy only — see below.)

**G3 — recovery (REAL, decisive).** `agi/a3_deploy_example/scripts/pp_gate3_rally.sh`
`PP_EXTRA_ARGS="--vel-box-center"` → **rescues == 0**. Isaac does NOT reproduce the deploy 30–55°
over-rotation, so only this MuJoCo/deploy gate proves transfer.

## Ship
First checkpoint clearing **G1 (2-in-a-row) + G2 + G3**. Export via the 110-D deploy chain, sync, parity.

## Hedge
The over-rotation is an AGI-MuJoCo plant artifact and was once fixed deploy-side with idle-anchor
(memory `hope-110-deploy-fixes-0708`). Keep that lever in parallel — if G3 stalls, the deploy-side fix
may close it without more GPU.
