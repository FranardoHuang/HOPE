# Footwork-to-strike reward redesign — BASE-FREE

**Goal.** Train a policy where the **lower body actively moves** to make the racket target reachable, then
stabilizes for the strike — **without any base position, base target, or base-arrival reward/observation**.
Built on the base-free `real_sensor_only` observation ([REALSENSOR_OBS_REDESIGN.md](REALSENSOR_OBS_REDESIGN.md)).
Variant: `task=HOPEPingPongRealSensor` / gym `HOPE-PingPong-RealSensor-AgibotA3-v0`. The `full` path is untouched.

## How movement is driven without a base target

The old design moved the body by **tracking an explicit base-XY target** (`base_position` reward) — which
needs the world base position, is fabricated on hardware, and is exactly what we removed. The replacement:

> **`racket_progress`** = `previous_racket_distance − current_racket_distance` (clamped), rewarded before
> the strike. `racket_distance = ‖racket_FK − racket_target‖` is **frame-invariant** (no base position).

Summed over the approach this **telescopes** to `weight × (distance the racket closed)`. So the policy is
paid for *any* whole-body motion that brings the racket toward the target — legs, waist, and arms all get
credit — with **no base target**. When the target is far/lateral and arm-only reach runs out, the cheapest
way to keep earning progress is to **step or shift the body**. The legs learn footwork as a *consequence*
of reducing racket error, not by copying a clip or chasing a base point.

Two guardrails keep it from cheating:
- **`arm_overreach`** penalty (fraction of arm joints within 10% of a limit) — discourages solving the
  target by maxing the arm out instead of moving the body.
- **Lower-body imitation is dropped** (below) — the legs aren't pinned to the clip's fixed leg motion, so
  they're free to adopt whatever stance/step reaches the sampled target.

## Reward terms (all weights are STARTING POINTS, in `HOPERealSensorRewardsCfg`)

| group | term | weight | notes |
|---|---|---:|---|
| **racket task** | `racket_position` / `velocity` / `normal` | 14 / 10 / 5 | inherited additive kernels (wide gradient), strike-window gated |
| | **`racket_strike_success`** | **+5** | **multiplicative** `R_pos·R_vel·R_normal` (tight acceptance std 0.075 / 0.5 / 0.262) — fires only on a *true* hit |
| **movement** | **`racket_progress`** | **+10** | dense pre-strike, base-free movement driver (telescopes to distance closed) |
| **anti-arm-only** | `arm_overreach` | −0.5 | arm joints near a limit |
| **footwork** (penalty only — feet may step) | `foot_slip_sq` | −1.0 | `Σ contact·‖foot_xy_vel‖²` |
| | `foot_velocity` | −0.05 | `Σ‖foot_vel‖²` (gentle — allow stepping, discourage flailing) |
| | `foot_drag` | −0.5 | lateral foot speed while near the ground (skimming) |
| | `pre_strike_foot_slip` | −0.4 | inherited (linear, pre-strike) |
| **strike-window stability** | `strike_upright` | −2.0 | `‖proj_grav_xy‖·strike_window` |
| | `strike_ang_vel` / `strike_foot_vel` / `strike_vbob` | −0.5 / −0.5 / −1.0 | wobble / foot motion / vertical bob at the hit |
| **balance + safety** (always on) | `upright` / `base_ang_vel_xy` / `base_lin_vel_z` / `joint_vel` | −1.0 / −0.05 / −0.5 / −1e-4 | |
| | `action_rate_l2` / `joint_torques` / `joint_limit` / `undesired_contacts` | inherited | |
| **terminations** | `bad_orientation` (0.7 rad) · `root_height_below_minimum` (0.5 m) | — | no foot-contact termination (recoverable) |

**REMOVED (the base-free correction):** `base_position` reward, `motion_global_anchor_pos` (reference
base-position tracking), and the previous **strong `feet_contact` "both feet planted" reward** (it prevents
movement). There is **no reward for keeping both feet planted** — only penalties for *bad* foot behaviour.

## Lower-body imitation decoupled

Every motion-body reward takes a `body_names` subset, so the footwork variant tracks **upper body only**
(`A3_UPPER_TRACKED` = torso + both arms). The legs (pelvis/hip/knee/ankle) are **not** imitated — they're
free to step to the sampled target. Upper-body + racket imitation still gives the swing its style.
`motion_global_anchor_ori` (torso orientation) is kept; `motion_global_anchor_pos` (base position) is dropped.

## Curriculum (widen the racket range so footwork is forced)

Phases share one task; widen the racket-target range per phase via the Hydra CLI (documented in the YAML),
warm-starting each from the previous checkpoint:
- **Phase 0** (this YAML's defaults): small range → learn a **stable strike**; legs barely need to move.
- **Phase 1**: wider lateral + vertical → `racket_progress` starts needing body/leg shift.
- **Phase 2**: full range + varied `time_to_strike` → arm-only reach is insufficient → the policy must
  **step / shift**. (Exact CLI overrides in `cfg/task/HOPEPingPongRealSensor.yaml`.)

## W&B metrics (logged each iter via the command's `metrics`, surfaced as `Live/racket_target/...`)

`racket_target_distance`, `racket_progress`, `racket_progress_prestrike`, `proj_grav_xy`, `base_ang_vel_xy`,
`base_vertical_speed`, `foot_slip_sq`, `foot_vel_mean`, `foot_lift_rate`, `foot_vel_at_strike`,
`arm_overreach_frac`, `arm_joint_vel_max`, **`leg_joint_vel_max`** + **`leg_moving_prestrike`** (do the legs
actually move before the strike?), plus the inherited `racket_pos/vel/normal_error`, `base_height`,
`base_roll/pitch_deg`. Reward shares appear as `Live/Reward/<term>` (e.g. `racket_strike_success`,
`racket_progress`); termination reasons as `Live/Termination/<term>`.

## Commands

**Gate first** (the obs-layout field-order check — host + Isaac):
```bash
python scripts/realsensor_obs_reference.py
python scripts/verify_realsensor.py --check layout --motion-file <…/motion.npz> --headless   # full=180, real=175
```

**First (small) training — Phase 0:**
```bash
python scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true num_envs=4096 \
  registry_name=${WANDB_REGISTRY_ORG}/wandb-registry-motions/hope_forehand \
  registry_name_2=${WANDB_REGISTRY_ORG}/wandb-registry-motions/hope_backhand
# (optional warm-start the swing: checkpoint_path=logs/rsl_rl/warmstart/model_15200_realsensor.pt)
```

**Phase 1 / 2** — warm-start from the Phase-0 checkpoint and widen the range (CLI in the YAML header), e.g.:
```bash
python scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true \
  checkpoint_path=logs/rsl_rl/agibot_a3_hope_realsensor/<phase0_run>/model_<N>.pt \
  racket.racket_pos_y_abs_range=[0.05,0.30] racket.pos_z_range=[0.75,1.05] \
  'racket.pos_range_per_clip.forehand.y=[-0.35,-0.12]' 'racket.pos_range_per_clip.backhand.y=[0.12,0.35]'
```

**Eval:**
```bash
python scripts/play.py task=HOPEPingPongRealSensor algo=ppo num_envs=4 \
  checkpoint=logs/rsl_rl/agibot_a3_hope_realsensor/<run>/model_<N>.pt
# watch Live/racket_target/leg_moving_prestrike (legs moving?) + racket_progress_prestrike (closing distance?)
```

## Tuning order (if the first run misbehaves)
- **Legs don't move / arm-only reach** → raise `racket_progress` (10→15) and/or `arm_overreach` (−0.5→−1.0);
  check `leg_moving_prestrike` rises as you widen the range (Phase 1).
- **Skating / dragging at the strike** → raise `foot_slip_sq` / `strike_foot_vel`.
- **Wobbly hit** → raise `strike_upright` / `strike_vbob`.
- **Footwork too jumpy** → raise `foot_velocity` slightly (but keep it small — it must still allow steps).

> Do NOT add a base target or a strong both-feet-contact reward to fix any of the above — those were the
> sim-to-real failures this redesign removes.
