# `real_sensor_only` observation redesign — HOPE ping-pong WBC

**Goal.** Make the *training* observation match what the real A3 can honestly produce. The deployed
HOPE policy failed sim-to-real because three actor observations depend on the robot's **world base
pose**, which has no localizer on hardware and is therefore *fabricated* at deploy (`anchor_pos_b := 0`,
`base_pos := nominal`). The policy then sees a different observation distribution than training and the
legs cannot balance. AGI's reference policy transfers precisely because its observation is
**real-sensor-only** (IMU orientation + proprioception, no world base position). This change copies that
recipe for the HOPE actor.

**Scope.** Training-side only. AGI's sim/backend are untouched. The old (`full`) path is intact; the new
mode is an additive variant. No hardware testing here — this is the training redesign + deploy-parity
preparation.

---

## Old vs new actor observation

| # | term | dim | `full` | `real_sensor_only` | honest on hardware? |
|---|------|----:|:------:|:------------------:|---------------------|
| 1 | `command` (ref clip future joint pos/vel) | 62 | ✓ | ✓ | yes — baked clip (deploy already provides) |
| 2 | `motion_anchor_pos_b` | 3 | ✓ | **removed** | NO — reference torso *position* error needs world base pose |
| 3 | `motion_anchor_ori_b` | 6 | ✓ | ✓ | yes — relative **orientation** only (IMU + clip); see yaw note |
| 4 | `base_ang_vel` | 3 | ✓ | ✓ | yes — pelvis IMU gyro |
| 5 | `joint_pos` | 31 | ✓ | ✓ | yes — encoders |
| 6 | `joint_vel` | 31 | ✓ | ✓ | yes — encoders |
| 7 | `actions` (last) | 31 | ✓ | ✓ | yes |
| 8 | `projected_gravity` | 3 | ✓ | ✓ | yes — pelvis IMU |
| 9 | `base_target_pos_b` | 2 | ✓ | **removed** | NO — base-repositioning target needs world base pose |
| 10 | `racket_target_pos_b` | 3 | ✓ (base-rel) | ✓ **reframed** | yes — now racket-FK-relative (base pose cancels) |
| 11 | `racket_target_vel_w` | 3 | ✓ | ✓ | yes — planner velocity (world); no base-pose dependency |
| 12 | `time_to_strike` | 1 | ✓ | ✓ | yes — swing clock |
| 13 | `swing_type` | 1 | ✓ | ✓ | yes — planner forehand/backhand flag |
| | **actor total** | | **180** | **175** | |

> Note: `base_lin_vel` was already removed from the `full` actor (`base_lin_vel = None`) in the prior
> HITTER alignment, so it is not in either layout. The **critic** (privileged) group is **unchanged** —
> it may use world base pose in sim because it is never deployed.

### Removed (2 terms, −5 dims)
- **`motion_anchor_pos_b`** (3) — the reference torso *position* relative to the robot; needs the true
  world base position. At deploy it is fabricated to 0 ("you are perfectly on track" — a lie), which is
  exactly why the deployed legs got no real balance feedback.
- **`base_target_pos_b`** (2) — desired base XY relative to the current base; a base-repositioning target
  that requires knowing the world base position.

### Replaced / reframed (1 term, same 3 dims)
- **`racket_target_pos_b`**: `quat_rotate_inverse(yaw_quat(base_quat), target_w − base_pos_w)`
  → `quat_rotate_inverse(yaw_quat(base_quat), target_w − racket_pos_w)`. Because `quat_rotate_inverse`
  is linear in its vector argument, the world base position **cancels**:
  `Rᵀ(target − racket) = (target rel base) − (racket FK rel base)`, both computable on hardware from the
  planner target + racket forward kinematics. Verified numerically (`realsensor_obs_reference.py`,
  cancellation error ≈ 1e-15). The kept old weight columns for this slot are reused as a warm init.

### Final actor dimension: **175**

### Deploy honesty
Every kept/reframed term is computable on the real robot from **IMU + joint encoders + the planner
target**, with **no world base position**. One residual: the **yaw** reference. `motion_anchor_ori_b` and
the `racket_target_pos_b` yaw-frame both use the base yaw; the pelvis IMU yaw is unreferenced (drifts).
This is the *same* yaw situation AGI's working policy has, and the deploy already handles it via the
`use_imu_yaw_for_targets` convention (identity yaw / robot-heading). Pitch/roll are fully honest from the
IMU. No term needs a base *position*, which was the actual sim-to-real break.

---

## Balance rewards + terminations (training-only; not in the deployed obs)

> ⚠️ **SUPERSEDED — the reward was redesigned to base-free footwork-to-strike.** `base_position` and the
> strong `feet_contact` reward below were **removed** (base-free correction); the current reward is in
> **[FOOTWORK_REWARD_REDESIGN.md](FOOTWORK_REWARD_REDESIGN.md)** (racket-progress driven movement, multiplicative
> strike success, footwork penalties, upper-body-only imitation). The obs redesign on this page is unchanged.
> The list below is kept only as the obs-stage history.

The old policy had **no absolute balance objective** — balance was implicit in tracking a reference-
relative anchor that is fabricated on hardware. Added to the `real_sensor_only` variant (gentle weights,
tune from here):

- `upright` = `flat_orientation_l2` (−1.0) — penalize base tilt (projected-gravity xy)
- `base_ang_vel_xy` = `ang_vel_xy_l2` (−0.05) — damp roll/pitch rate
- `base_lin_vel_z` = `lin_vel_z_l2` (−0.5) — penalize vertical bob (anti sink/hop)
- `joint_vel` = `joint_vel_l2` (−1e-4) — mild smoothness (plus the inherited `action_rate_l2` −0.1)
- **`feet_contact`** = both-feet-in-contact stance reward (**+0.5**, always on) — returns the fraction of
  feet in contact (1.0 both planted / 0.5 one foot unloaded / 0.0 airborne), so unloading a single foot
  costs half the reward even *without* a full lift-off. Whole-foot contact (A3 has no toe/heel bodies).
- **`foot_slip_all`** = un-gated foot-slip penalty (**−0.2**, whole episode) — extends the inherited
  `pre_strike_foot_slip` (−0.4, pre-strike only) through the strike + follow-through so it can't skate.
- **Absolute terminations** (not reference-relative): `bad_orientation` (limit 0.7 rad ≈ 40°) and
  `root_height_below_minimum` (0.5 m) — a real fall/sink ends the episode regardless of the clip. **No
  foot-contact termination** (reward/penalty only) so the policy can always recover.

> **Why stance contact is added now, not after a first run:** the observed hardware failure is lower-body
> support — the policy can satisfy `upright`/height while quietly unloading one foot or skating in sim.
> `feet_contact` + always-on `foot_slip_all` make the planted stance an explicit, always-on objective.

> ⚠️ **Risk + the next lever (read before tuning).** The inherited `base_position` and
> `motion_global_anchor_pos` rewards are training-only, but they still **pull the lower body toward the old
> reference/base behavior** (the crouch-and-shift that fails on hardware). If the first RealSensor run
> **still crouches or unloads the feet, the next lever is NOT more `upright`** — it is to **reduce/remove
> the leg/base imitation** (`base_position`, and the leg portion of `motion_global_anchor_pos` /
> `motion_body_*`) and replace it with **legs-near-official-stand + `feet_contact`/`foot_slip`**. I.e. stop
> imitating the reference legs; hold the legs near the proven A3 stand pose and let the stance-contact
> terms own balance, while imitation drives only the upper body (arms/waist).

DR is unchanged (mass ±15%, friction, CoM, **PD-gain ±20%** already on, obs noise; push disabled).

---

## Files changed (old path untouched)

| file | change |
|------|--------|
| `…/mdp/hope_commands.py` | + `RacketTargetCommand.racket_target_pos_b_rel()` (FK-relative, base pose cancels) |
| `…/mdp/hope_observations.py` | + `racket_target_pos_rel_b()` obs fn |
| `…/config/agibot_a3/hope_env_cfg.py` | + `HOPEObservationsRealSensorCfg`, `HOPERealSensorRewardsCfg`, `HOPERealSensorTerminationsCfg`, `HOPEPingPongRealSensorAgibotA3EnvCfg`; + `obs_mode` field |
| `…/config/agibot_a3/__init__.py` | + gym id `HOPE-PingPong-RealSensor-AgibotA3-v0` |
| `cfg/task/HOPEPingPongRealSensor.yaml` | new task (copy of HOPEPingPong.yaml; only `name`/`gym_task` differ — **keep in sync**) |
| `scripts/warm_start_realsensor.py` | partial warm-start 180→175 (column-remap actor input + normalizer; reset optimizer) |
| `scripts/realsensor_obs_reference.py` | numpy layout + reframe self-test (deploy-parity spec) |
| `scripts/verify_realsensor.py` | A layout / B rollout / C onnx-dim / D golden-capture |

---

## Commands

**0) Verify the obs math (host, no Isaac):**
```bash
python scripts/realsensor_obs_reference.py        # prints 175 layout + cancellation self-test (PASS)
```

**1) Verify the live obs layout (Isaac) — the deploy-parity GATE (must show full=180, real_sensor=175):**
```bash
python scripts/verify_realsensor.py --check layout  --motion-file <…/motion.npz> --headless
python scripts/verify_realsensor.py --check rollout --motion-file <…/motion.npz> --headless --steps 100
```

**2) Partial warm-start from model_15200:**
```bash
python scripts/warm_start_realsensor.py --old-ckpt \
  logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch/model_15200.pt --dry-run
python scripts/warm_start_realsensor.py --old-ckpt \
  logs/rsl_rl/agibot_a3_hope/2026-06-28_23-01-24_phase050_perclippos_scratch/model_15200.pt \
  --out logs/rsl_rl/warmstart/model_15200_realsensor.pt
```

**3) First training (warm-started; drop `checkpoint_path=` to train from scratch):**
```bash
python scripts/train.py task=HOPEPingPongRealSensor algo=ppo headless=true \
  registry_name=${WANDB_REGISTRY_ORG}/wandb-registry-motions/hope_forehand \
  registry_name_2=${WANDB_REGISTRY_ORG}/wandb-registry-motions/hope_backhand \
  checkpoint_path=logs/rsl_rl/warmstart/model_15200_realsensor.pt
```

**4) First eval + ONNX export (play.py writes policy.onnx next to the checkpoint):**
```bash
python scripts/play.py task=HOPEPingPongRealSensor algo=ppo num_envs=4 \
  checkpoint=logs/rsl_rl/agibot_a3_hope_realsensor/<run>/model_<N>.pt
python scripts/verify_realsensor.py --check onnx --onnx \
  logs/rsl_rl/agibot_a3_hope_realsensor/<run>/policy.onnx        # asserts actor input == 175
```

**5) Deploy-parity golden (for the future C++ pp_obs_builder real_sensor variant):**
```bash
python scripts/verify_realsensor.py --check golden --motion-file <…/motion.npz> --headless \
  --out logs/realsensor_golden.npz
```

---

## Next (NOT done here — deploy side)
The C++ `pp_obs_builder` must get a `real_sensor_only` variant matching the 175-D layout above (drop
`motion_anchor_pos_b` + `base_target_pos_b`; reframe `racket_target_pos_b` to `target − racket_FK` in the
base-yaw frame). Validate it offline against `logs/realsensor_golden.npz` before any hardware run. Do this
only after the retrained policy passes the sim → MuJoCo sim2sim ladder.
