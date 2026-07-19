# Run Training

Status: Draft

For GPU runs on the shared team pod (3× RTX 5090, per-user clones and folders), see
[run_on_runpod.md](run_on_runpod.md); the commands below are identical there after
`source /workspace/<name>/env.sh`.

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

2026-07-01 update:

- `HOPEPingPong` now defaults to unified forehand+backhand HITTER training: `registry_name_2` enabled, `target_mode: uniform`, per-clip 3-D blade-centered position and velocity target boxes (`pos_range_per_clip` / `vel_range_per_clip`; this supersedes the earlier fixed hit plane `x=0.4` with (y,z)-only sampling), actor `swing_type`, and no actor racket-normal observation.
- Local Step 9-12 motion products are first-class training inputs. Pass `motion_file=<forehand.npz>` plus optional `motion_file_2=<backhand.npz>` to skip WandB entirely; omit local files to use `registry_name` / `registry_name_2`.
- `setup_train_env.sh` is portable again: it reads optional `setup_train_env.local.sh` and overridable `HOPE_ISAAC_*` paths, with auto-detection for known `/workspace/...` Isaac layouts when present.

2026-07-02 update (verified on the current shared RunPod):

- Local two-clip `HOPEPingPong` smoke and a registry-backed WandB pipeline smoke both ran on the copied Agibot A3 URDF asset (31 actuated DOF). The registry smoke run `6xus13ga` finished at https://wandb.ai/BerkeleyPingPong/hope_wbc/runs/6xus13ga, but the `hope_forehand:v4` / `hope_backhand:v4` motion artifacts it used were later verified to face world +Y rather than HOPE +X. Treat that run as pipeline-only evidence.
- Until future verified registry artifacts are uploaded, pass the corrected local `_hopex.npz` clips explicitly via `motion_file=` / `motion_file_2=` instead of relying on the v4 registry aliases. The local v5 clips are R15 ablation inputs only, not product/default replacements.
- HOPE task YAMLs set `motion.wrap_teleport: false` (also the code default): a mid-episode clip wrap resamples the reference clip/time and racket target without teleporting the simulated robot. Episode reset still uses RSI.

2026-07-03 realignment:

- The default training task is now `HOPEPingPongDeployParity` (gym id `HOPE-PingPong-DeployParity-AgibotA3-v0`); `HOPEPingPongRealSensor` is a backward-compat alias for the same task. Its actor observation is 175-D deploy-parity: it removes `motion_anchor_pos_b` (3) and `base_target_pos_b` (2) and reframes `racket_target_pos_b` racket-FK-relative. Layout reference: `scripts/realsensor_obs_reference.py`; checks: `scripts/verify_realsensor.py`.
- `task=HOPEPingPong` remains available only as the legacy 180-D full-obs comparison path; it is NOT deploy-honest and cannot deploy.

2026-07-05 R15 v5 correction + strike-annotation registry:

- Product/default train and replay configs remain on the hopex/registry route: `cfg/train.yaml`
  and `cfg/play.yaml` keep `motion_file: null`, and `HOPEPingPongDeployParity.yaml` keeps
  `strike_phase_per_clip: [0.47, 0.333]`. Do not edit those defaults to v5 for R15.
- `cfg/strike_annotations.yaml` is the contact-phase source of truth for reference clips (all 6
  adjudicated 2026-07-05; the hopex values are a speed-peak CONVENTION — its source videos are
  ball-less dry swings). `scripts/analyze_strike_phase.py` applies annotations first and reports
  the speed peak only as a diagnostic candidate (known trap: post-contact whip / pre-contact
  pull-up). R15 overrides are now `strike_phase_per_clip=[0.673,0.362]` with the regenerated
  backhand boxes (see NOW.md).
- `scripts/play.py` uses the same local motion resolver as training when `motion_file` or
  `motion_file_2` is set, so R15 replay/export honors both local clips.

2026-07-04 update (deploy-parity robustness flags, all default OFF):

- `motion.clip_switch_prob` (default `0.0`; try `0.002` ≈ one switch per 3-4 swings): each control
  step that fraction of envs aborts its swing operator-style — the reference jumps to a random
  clip's FIRST frame with a fresh pre-swing hold and a fresh target; the robot is untouched (no
  teleport). This is deploy parity for `pp_reference_clock.hpp`, which flips `clip_id` mid-swing
  whenever the planner re-sides the target; training previously only switched at clip END — the
  root cause of the venue falls at 准备/正手/反手 switches. Aborted swings do NOT enter the A8
  post-swing buffer and slightly deflate completion-rate metrics. Watch: `clip_switch_count`.
- P2.4 `base_decel` reward (PACE-style pre-strike base-speed shaping, default OFF via
  `rewards.base_decel_weight: 0.0`): see Reward Shaping below.
- `motion.speed_scale_range` (R14 retiming, default `[1.0, 1.0]` = OFF; ablation trial
  `[0.8, 1.2]`): per-swing reference playback speed, resampled at every swing entry (wrap,
  clip switch, reset). At speed s the clip clock advances s frames per control step (float shadow
  clock, round() indexing = deploy-clock parity), reference joint/body/anchor velocities read ×s,
  `time_to_strike` runs ÷s (computed from the float clock so the exact-strike detector still fires
  once per swing), and provisional racket velocity targets scale ×s (uniform boxes,
  `reference_perturbed`, and the HER clamp box). A schema-3 question bank is different: its
  demanded racket velocity is an absolute inverse-physics answer for the unchanged incoming ball,
  so the final bank assignment deliberately overwrites the provisional speed-scaled target. This
  makes `question_bank` compatible with fixed `motion.speed_scale_per_clip` or a speed range without
  silently slowing the required return. Positions/normals are speed-invariant. Retiming is TRAIN-ONLY:
  play/eval force it back to `[1.0, 1.0]`. Deploy note: the runner's `swing_speed` knob retimes
  the clock but does NOT scale reference/target velocities — enabling it for an R14-trained
  policy requires adding those two scalings. Watch metric: `playback_speed`.
- A1v2 actor-view sensor defects (`racket:` block; modeled on the venue mocap fit — occlusion gaps
  concentrate at contacts, re-lock after contact carries a fresh bias): `target_dropout_prob`
  (per-step frame loss, hold-last), `target_post_strike_dropout_s` (forced hold-last window after
  each strike; venue ~0.03 s), `target_bias_per_swing` (3-D Gaussian position bias resampled at
  each strike edge, held constant within a swing). They degrade ONLY the actor-visible target
  view; rewards, critic, and metrics keep the true target. Same block as the A1 latency/noise
  family (`target_delay_steps`, `target_jitter_pos_per_s`, `target_jitter_vel_per_s`,
  `midswing_resample_prob`, `target_noise_white`, `target_noise_ar1_sigma`,
  `target_noise_ar1_rho`) — every one of these defaults off.
- `racket.target_delay_tts_mode` makes the actor-visible planner tuple time-coherent when
  `target_delay_steps > 0`. `live` is the historical/default behavior: delayed position, velocity,
  face and sign are paired with the live countdown. `source_timestamp_compensated` delays all five
  fields atomically and subtracts `target_delay_steps * policy_dt` from that delayed countdown;
  `uncompensated` delays the same tuple but intentionally leaves its old countdown untouched as a
  matched negative control. Only the policy observation switches to this actor countdown; critic,
  Reward gates and truth metrics retain live `time_to_strike`. Reset backfills the complete tuple,
  while dropout holds the complete tuple. The selected mode and delay are checkpoint hard-contract
  fields. See [atomic planner tuple](../DEFINITIONS.md#atomic-planner-tuple-timing).
- `task.planner_revision` is the opt-in replacement for the old “freeze target after engage” and
  `midswing_resample_prob` ablation. When enabled it configures both motion and racket command
  terms together; a half-configured clock-only or target-only path is rejected. One physical ball
  keeps immutable question/Reward/critic truth while the actor receives atomic position, velocity,
  signed-normal and TTS revisions. A checkpoint-bound
  [`phase governor`](../DEFINITIONS.md#phase-governor) may change reference rate without reversing
  phase or exceeding its frozen rate, acceleration, target-delta or deadline-delta limits. The
  legacy hold clocks are forced to zero because the revision task's initial TTS is the sole
  preparation clock. For `phase_governor_v1`, `racket.target_delay_steps` must remain `0`.
  Positive delay fails before launch because the legacy ring delays only the actor, not the motion
  governor; it becomes legal only after both consume one coupled transport tuple in the same tick.
- An enabled revision block must include a complete `initial_tts_mixture`, not merely a broad min/
  max range. The intended launch family has four explicit strata: a sub-0.5-second stress band,
  an exact 0.5-second point mass, a 0.5–0.9-second fast-deployment band and a longer-arrival band.
  The four weights sum exactly to one and their support exactly equals
  `initial_tts_range_s`. Thus 0.5 seconds is a separately counted required baseline, not the
  minimum preparation time. Per-component, below/exact/above-0.5 and total sample counts must
  partition exactly in every accepted full-scene probe and checkpoint contract. See
  [`initial TTS mixture`](../DEFINITIONS.md#initial-tts-mixture).
- This path does not claim a 0.5-second return from source tests. Behavior must be measured with
  the immutable K100 [`0.5-second timing exam`](../DEFINITIONS.md#timing-exam-0p5): frame 0 has
  zero reference velocity, every attempt remains in the denominator and a fixed clock multiplier
  is explicitly inexact. A separate TOPP run must also certify the chosen reference/action path;
  the current safe heuristic upper bound is not a proof of a global minimum or a 0.5-second
  feasible trajectory.
- A task-revision run is pruneable only when exactly one command term provides the behavior
  ledger and the runner emits one canonical `HOPE_EXACT_BEHAVIOR_UPDATE_JSON=...` record per PPO
  update. Completion is `swing_completion_count / swing_outcome_count`: both counters close on the
  same attempt-end event, so a start in one 100-update window and an outcome in the next cannot
  invalidate either window. `swing_start_count` and `strike_opportunity_count` remain raw
  diagnostics. Physical fall accepts only exact boolean termination reasons and is split into
  mutually exclusive pre/post counts; guard or timeout reset is separate. A numeric truthy tensor,
  duplicate provider, missing update, duplicate update or zero denominator makes the behavior
  decision unavailable and the trainer continues—it never manufactures zero or stops the arm.
- In planner-revision mode, ready eligibility is exactly the first metrics sample after a new
  active `(control_epoch, task_id)` is installed. It is not the install function itself and is not
  `motion.in_hold OR new task`: legacy hold clocks are zero because time-to-strike (TTS, remaining
  time before contact) is the sole preparation clock. Same-ball `task_revision` updates therefore
  do not duplicate ready samples. The log must emit `ready_phase_sample_count`,
  `ready_planner_task_entry_sample_count`, `ready_planner_legacy_hold_violation_count` and
  `ready_foot_sensor_unavailable_sample_count`; a nonzero planner legacy-hold violation or an
  unexplained zero denominator blocks pruning. Missing foot sensors are unavailable measurements,
  never fabricated zero contact/slip. Historical receipts created without these witnesses cannot
  be backfilled.
- Before launching a pruneable successor, run a clean detached full-scene probe that proves a
  finite checkpoint, exact source/contract binding, nonzero conserved task-entry and ready
  denominators, zero planner legacy-hold violations and explicit sensor availability. The Pod2
  CPU-only direct probe (`4/4` on exact `0ebd14a6…a8dd`) checks source mechanics only. It does not replace the full-scene
  probe or the two complete disjoint 100-update windows required for any ranking or stop.
- `rewards.free_wrist_ori_mimic` (R16, default `false`): drop `right_wrist_yaw_Link` (the racket
  mount) from the `motion_body_ori` / `motion_body_ang_vel` body lists — the wrist's ORIENTATION
  stops being imitated while position/linear-velocity mimic keep the swing path. Rationale
  (franco): the video pipeline's wrist orientation is unreliable (GVHMR), so mimicking it caps
  face quality; freed, the face is shaped by the `racket_normal` reward (and by ball-outcome
  rewards on the VirtualBall stack — the arm with a real learning signal for the face). Note this
  codebase has NO joint-level imitation rewards — body-level `motion_body_ori` on the wrist link
  IS the face mimic, so the flag is config-level (body-list filtering in `train.py`).
- ⚠ Override-whitelist rule: task-yaml keys under `task.motion` / `task.racket` are translated
  through explicit whitelists (`_MOTION_KEYS` / `_RACKET_KEYS` in `scripts/train.py`) and any
  unconsumed key RAISES at startup. Adding a new key to a task yaml therefore requires extending
  the whitelist in the SAME commit — 018467a added `clip_switch_prob` to the yaml only and broke
  every task-yaml startup until the 74c129e hotfix.

`TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied Agibot A3 URDF asset (31 actuated DOF), including WandB logging, checkpoint save, and ONNX export. This proves the pipeline can run; it is NOT an accepted quality baseline. G04/G05 remain Partial, and G06/G07 are not accepted until sim-to-sim and dry-run deployment gates record verification.

## Training-critical change barrier

Before changing any contract that affects a live trainer, checkpoint consumer, evaluator, pruning
rule, planner-policy tuple, motion timing law, or recurring Pod command, first set the existing
training automation to `PAUSED` and verify that state. Pausing the automation does not itself stop
trainers; it prevents a stale recurring turn from inspecting, attesting, pruning, launching or
signalling against a contract while that contract is being changed.

Keep the automation paused while implementation, independent review, source tests, documentation,
and any required one-shot runtime gate are incomplete. Resume the **same** automation only after the
verified change is on `main`, the operation document contains the exact command, and every in-flight
one-shot mutation has an unambiguous no-clobber state. Never create a second automation to work around
this barrier. Read-only operator inspection remains allowed, but it cannot publish receipts, signal a
process, retry a run, or be reported as a behavior verdict.

This branch adds:

- a scrubbed `setup_train_env.sh` as the training shell setup source of truth (site paths are now overridable env vars).
- source-first `HOPE_WBT_PYTHONPATH` ordering, so local `whole_body_tracking` edits beat stale installed copies.
- richer live `Live/...` telemetry in WandB/TensorBoard from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.
- `HOPE-TableTennis-AgibotA3-v0`, a first-pass Isaac Lab table/net/ball/A3 physics scene for G04 visualization and future G08 returner/spin experiments, now with a tracked Purdue PACE USD table/net visual overlay and Purdue-style table/ball contact materials.
- updated `HOPEPingPong` target/reward defaults with unified forehand/backhand sampling, per-clip blade-centered uniform racket target boxes, per-clip strike timing, conditional exact-strike metrics, and debug reward logging hooks.
- canonical WandB motion uploads where every motion artifact contains `motion.npz`, regardless of the source filename.
- HOPE +X motion alignment in `scripts/csv_to_npz.py --robot agibot_a3` (`--hope_frame auto`) before local save/upload.
- `scripts/check_motion_target_alignment.py`, a no-Isaac gate for frame-0 yaw, +X-dominant strike velocity, and target/reference center alignment.
- `motion.wrap_teleport` (default `false`; kept explicit in the HOPE task YAMLs) controlling the mid-episode RSI teleport on clip wrap, plus a `racket_progress` resample-spike fix. (The branch's original `rsi_on_wrap` knob was dropped 2026-07-03 in favor of main's equivalent `wrap_teleport`.)
- explicit `wandb.finish()` before Isaac `simulation_app.close()`, so WandB runs finish and sync before Isaac can hard-exit the process.

## Entry Files

- `hope_training/whole_body_tracking/README.md`
- `hope_training/whole_body_tracking/scripts/train.py`
- `hope_training/whole_body_tracking/scripts/play.py`
- `hope_training/whole_body_tracking/scripts/play_table_tennis.py`
- `hope_training/whole_body_tracking/scripts/probe_metric.py`
- `hope_training/whole_body_tracking/cfg/train.yaml`
- `hope_training/whole_body_tracking/cfg/play.yaml`
- `hope_training/whole_body_tracking/cfg/strike_annotations.yaml`
- `hope_training/whole_body_tracking/setup_train_env.sh`

## Environment Setup

This runs in the GPU/Isaac environment (Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10, CUDA GPU), not the ROS environment. `grasping` is the maintainer's EXAMPLE distrobox name — substitute your own box.

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

The script must be SOURCED (not executed) in every new GPU/Isaac terminal. It defines the `hope_isaac_py` launcher, sets `HOPE_WBT_PYTHONPATH`, and exports placeholder WandB variables for optional registry/logging use.

The script is scrubbed of site-specific paths. It reads overridable env vars with placeholder defaults:

- `HOPE_ISAAC_PYTHON` — the Isaac Lab Python interpreter `hope_isaac_py` wraps.
- `HOPE_ISAACLAB_ROOT` — your Isaac Lab checkout.
- `HOPE_ISAAC_VENV_SITE` — optional extra `site-packages` to inject (e.g. to provide `hydra`/`omegaconf`).

Set these for your machine in a git-ignored `setup_train_env.local.sh` next to the script; `setup_train_env.sh` auto-sources it if present and auto-detects known `/workspace/...` Isaac layouts.

On the current shared RunPod (verified 2026-07-02), the actual Isaac install is the venv at `/workspace/hope_isaac_venv` with Isaac Lab at `/workspace/IsaacLab`; point the `setup_train_env.local.sh` overrides at that install. The legacy `/workspace/isaacsim/python.sh`, `/opt/drone_venv`, and `hope-motion-py310` paths are not used for Isaac training. If another machine has different paths, update the local override and this doc together.

A from-scratch Isaac Sim 4.5.0 / Isaac Lab 2.1.0 / Python 3.10 install is NOT documented here and is the single biggest reproducibility gap. Follow the official [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) first, then point `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` at it.

### Blackwell (RTX 50-series, sm_120) torch fix

Isaac Sim 4.5.0 ships `torch 2.5.1+cu124`, which has **no sm_120 kernels**. On a Blackwell GPU
(e.g. RTX 5090) `import torch` "works" and `torch.cuda.is_available()` is `True`, but any real CUDA op
fails with `no kernel image is available for execution on the device`, so training crashes after Kit
startup. Fix (verified 2026-06-26 on RTX 5090):

```bash
# inside the Isaac env (hope-isaac-py310)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4    # Isaac Sim 4.5 requires numpy<2; the torch upgrade may pull numpy 2.x
```

`isaaclab*` carry a `torch==2.5.1` pin in metadata but are editable installs imported at runtime, so the
upgrade does not break them. Rollback: `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124
--index-url https://download.pytorch.org/whl/cu124`. Verify with a real kernel, not just `is_available()`:

```bash
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print((x@x).sum().item())"
```

### EULA

The first Kit launch needs the NVIDIA Omniverse EULA. Accept it non-interactively for headless runs:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### No-WandB training (local motion override)

`scripts/train.py` can load the motion clip from a local `.npz` instead of the WandB registry. Pass
`motion_file=<path>` (and `motion_file_2=<path>` for a unified forehand/backhand policy); when set it
skips WandB entirely, so use it with `logger=tensorboard` for an account-free run. Resolution is
LOCAL-FIRST (`resolve_motion_sources` in `train.py`): explicit `motion_file=` / `motion_file_2=` always
win and bypass the registry; only when no local files are given are `registry_name` / `registry_name_2`
downloaded from WandB. Back-compat: a local `.npz` path (or a directory containing `motion.npz`) passed
as `registry_name=` / `registry_name_2=` is rewritten to `motion_file=` and stays registry-free. If you
have no motion data at all, generate a placeholder "stand at default pose" clip (pipeline proof only,
not a real swing):

```bash
hope_isaac_py scripts/make_static_motion.py --robot agibot_a3 \
  --output_file ../motions/a3_stand.npz --frames 600 --fps 50

hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  num_envs=1024 max_iterations=60 algo.runner.save_interval=25 \
  logger=tensorboard run_name=stand_bootstrap \
  motion_file=$(pwd)/../motions/a3_stand.npz
```

For a local unified HITTER smoke after the video pipeline has produced both clips:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=hope_local_unified_smoke
```

### Motion kinematics schema 2 preflight

Fresh formal runs require each NPZ to bind three facts: body positions are
link origins, body linear velocities are COM-point velocities, and
`body_names` gives the complete articulation column order. Every clip must
also have one finite positive scalar FPS, all clips must share it, and it must
equal the policy rate (`1 / env.step_dt`, currently 50 Hz). Schema 1 and
untagged clips remain loadable only for diagnostic checkpoint compatibility;
they cannot produce an exact schema-v3 checkpoint/ONNX.

For MuJoCo conversion, discover the body order once against a trusted Isaac
reference, then reuse the emitted file. The GMR source `dof_pos` and runtime
`joint_pos` orders are intentionally different; the converter validates their
content-bound bijection and requires complete donor ONNX `joint_names`,
`articulation_joint_names`, and identity `action_joint_ids` metadata:

```bash
python scripts/csv_to_npz_mujoco.py \
  --mjcf /path/to/a3_pingpong.xml --donor /path/to/policy.onnx \
  --joint-order-contract configs/a3_joint_order_bijection_v1.json \
  --discover-map /path/to/trusted_isaac_motion.npz \
  --body-order /path/to/body_order.txt
```

Run `python scripts/a3_joint_order_contract.py` first. A successful source gate
still prints `schema2_materialization_authorized=false`; each new private motion
family needs its own content-bound/no-clobber conversion preregistration.

Migrate a legacy V5/MuJoCo clip whose stored velocity is the derivative of
link position:

```bash
python scripts/migrate_motion_kinematics.py \
  --input /path/to/legacy.npz --output /path/to/migrated_comv.npz \
  --source-point link_origin --mjcf /path/to/a3_pingpong.xml \
  --body-order /path/to/body_order.txt
```

`--body-order` describes the source NPZ columns; it is not automatically the
current articulation order. If a real Kit preflight reports a different
runtime body order, capture that current order and rerun with
`--target-body-order /path/to/current_runtime_body_order.txt`. The migration
then permutes all four body pose/velocity arrays by body name before converting
link-origin velocity to COM velocity. Never relabel the metadata without
reordering the arrays.

For a legacy Isaac clip already carrying COM velocity, use
`--source-point center_of_mass --body-order ...` and omit `--mjcf`. Never
infer the point or body order from a filename. Interpolation-only retiming
outputs are explicitly tagged `link_origin` and are not formal training
inputs; use the FK output mode to regenerate COM velocity.

Fresh schema-v3 training must also choose the joint-friction plant explicitly.
The checked-in A3 actuator config preserves historical, uncalibrated PhysX
coefficients. Because those dimensionless/load-dependent values have no exact
MuJoCo `frictionloss` equivalent, they remain a diagnostic control. Launch the
cross-engine-exact zero-friction control from scratch with:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongVirtualBall algo=ppo \
  task.plant.zero_joint_friction=true \
  motion_file=/abs/path/forehand_schema2_comv.npz \
  motion_file_2=/abs/path/backhand_schema2_comv.npz \
  ++task.racket.question_bank=/abs/path/schema3_train.npz \
  headless=true
```

The flag is absent/false by default and never changes an existing checkpoint.
The saved `training_contract.json` records the expanded per-joint coefficients;
any legacy warm-start remains exact-ineligible even when the new run selects
zero friction. Use a paired fresh `zero_joint_friction=true` versus
as-configured run when measuring the plant effect, and label the as-configured
cell diagnostic until a physically calibrated PhysX/MuJoCo mapping exists.

Do not use that binary flag to launch the future calibrated `SC` cell. The
unit-explicit plant-contract v1 preparation and its evidence checklist are in
[`prepare_semantics_correct_plant.md`](prepare_semantics_correct_plant.md).
Current training has no `SC` adapter hook. A reviewed future launch must bind
the prepared contract into the hard contract before `gym.make`; its final
MuJoCo evidence must instantiate the adapter in the Agibot vendor
Gate3/Gate3B runtime and bind the vendor MJCF/runtime/31-joint report. A
generic MuJoCo wrapper or current `contract-proxy` result cannot fill that
role.

### Serialized Kit boot for multi-GPU hosts

Do not let several Isaac/Kit processes initialize concurrently on one Pod.
`scripts/launch_kit_training_locked.sh` holds the host boot lock only until a
reliable log marker appears, launches the child in its own process group, and
records the exact PID/PGID and command in `<log>.launch`. After the marker, the
lock is released so already-booted training jobs may run concurrently:

```bash
source /workspace/codexschema/env.sh
KIT_BOOT_MARKER='Learning iteration' KIT_BOOT_TIMEOUT_S=900 \
  scripts/launch_kit_training_locked.sh /abs/path/arm/run.log \
  env CUDA_VISIBLE_DEVICES=0 /workspace/hope_isaac_venv/bin/python \
  scripts/train.py task=HOPEPingPongVirtualBall algo=ppo device=cuda:0 \
  headless=true logger=tensorboard run_name=arm
```

For a `max_iterations=0` mechanism smoke, use a marker such as
`[train.py] hard training contract:` instead. A process that exits before its
required marker is a failed boot even when its exit code is zero. A boot
timeout sends TERM, then KILL if necessary, only to the recorded arm PGID;
never replace that cleanup with a broad `pkill`.

The frozen 2026-07-11 Phase-1 recipes use
`scripts/launch_phase1_20260711.sh`. It verifies every parent checkpoint,
motion and train-bank SHA before invoking the locked launcher. Run the exact
179-D construction gate first, then start the three lanes assigned to each
Pod:

```bash
scripts/launch_phase1_20260711.sh smoke   # Pod 1; inspect contract, then wait for clean exit
scripts/launch_phase1_20260711.sh pod1    # M3 old/S1 + fresh schema-v3 seed 1
scripts/launch_phase1_20260711.sh pod2    # M2 old/S1 + fresh schema-v3 seed 2
```

Set `PHASE1_DRY_RUN=1` to validate inputs and print shell-escaped commands
without starting Kit. The causal continuations deliberately use the legacy
motion diagnostic flag and remain `training_contract_exact=0`; the two fresh
seeds use runtime-order schema-2 motion, a strict schema-v3 train bank, no
checkpoint and `zero_joint_friction=true`.

Those first six processes occupy the six cards but do **not** fill the measured
breadth capacity. The 2026-07-08 rule is three to four 4096-env jobs per GPU;
the 2026-07-11 Phase-1 target is 24 jobs. The scale-out roles are deliberately
layered so each host can be audited at two, three and four jobs per card:

```bash
# Run this launcher from a detached/current control worktree, but point every
# training command at the clean frozen 6d93bcb checkout.
export PHASE1_REPO_ROOT=/workspace/codexschema/nohope
export PHASE1_STAGGER_S=75
EVAL=/workspace/codexschema/nohope_eval_08e438e  # historical directory name; verify the live HEAD
test -z "$(git -C "$EVAL" status --porcelain)"
git -C "$EVAL" rev-parse HEAD

bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod1_scaleout_2
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod1_scaleout_3
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod1_scaleout_4

bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod2_scaleout_2
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod2_scaleout_3
bash "$EVAL/hope_training/whole_body_tracking/scripts/launch_phase1_20260711.sh" pod2_scaleout_4
```

The two Pods may launch the same layer in parallel; one Pod still serializes its
own three Kit boots. Verify all six new first iterations/contracts before
starting the next layer. The authoritative assignment and run names are in
`configs/phase1_scaleout_matrix_20260711.json`. Scale-out roles refuse a dirty
checkout or a training commit other than
`6d93bcb16c422a2f42748c2dc99432559653480b`.
If a layer stops on its second/third boot, preserve that failed arm and rerun
the same role: a still-live, command-identical ready arm is verified and
skipped. If an earlier arm has already completed, use
`PHASE1_ONLY_ARM=<exact_run_name>` to launch only the reviewed remaining arm.

Do not wait for terminal checkpoints to discover whether an ablation works.
The initial missing curves are frozen in
`configs/phase1_checkpoint_curve_initial_pod{1,2}_20260711.json`. The following
command is a **historical record only**; those manifests predate the mandatory
screen-policy/job-contract schema and must not be passed to the checked-in new
worker. Their successful/failed states are already preserved:

```bash
python3 "$EVAL/hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py" \
  --manifest "$EVAL/configs/phase1_checkpoint_curve_initial_pod1_20260711.json" \
  --judge-script "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" \
  --state-dir /workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/initial_pod1 \
  --max-active-cpu 9
```

The live 2026-07-11 paper deliberately remains on the clean detached evaluator
commit and judge SHA below. The current branch has since hardened both worker
and judge; mixing that latest `judge.sh` with manifests frozen to the old SHA
is correctly rejected. Do not update the live manifest SHA in place.

```bash
EVAL=/workspace/codexschema/nohope_eval_08e438e
RUNTIME_MANIFESTS=/workspace/codexschema/phase1_fresh_20260711/runtime_manifests
test "$(git -C "$EVAL" rev-parse HEAD)" = 46a0ce24524fdb843e55fe82ba4c045f2adc090f
test -z "$(git -C "$EVAL" status --porcelain)"
test "$(sha256sum "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" | awk '{print $1}')" = \
  1a00702935096b063435c3f0bd23e75f76f13e1298c87310d1cec3c26cca8529
```

The runtime manifest copies are the corrected 20998/split/SP-inexact files
from this repository; copy and hash-check them **before** launching a worker,
never while that worker is alive. This lets the historical clean evaluator run
the corrected queue without editing its worktree.

The worker starts the next judge only after the prior judge reaches its CPU-only
MuJoCo phase. It sets OpenMP/MKL/OpenBLAS/NumExpr to one thread, records exact
checkpoint and evaluator commit/hashes, requires clean frozen training/eval
worktrees, refuses stale failed state, and never signals a training process.
`judge.sh` shares the training launcher's Kit boot lock; only CPU exam phases
overlap. Observation-normalizer sidecars preserve finite zero std
dimensions only because the bound runtime divisor is `std + eps` with
`eps=0.01`; negative/non-finite std or a non-positive divisor remains fatal.
Before taking the Kit lock, `judge.sh` now activates the CPU evaluator and
requires both the graph loader and runtime. The current Pods use
`onnx==1.22.0` and `onnxruntime==1.27.0`:

```bash
/workspace/hope_mjeval_venv/bin/python -m pip install 'onnx==1.22.0'
/workspace/hope_mjeval_venv/bin/python - <<'PY'
import onnx, onnxruntime
print(onnx.__version__, onnxruntime.__version__)
PY
```

`onnxruntime` alone is insufficient because formal normalization preflight
inspects and checks the graph through `onnx`. Exact A3 plant comparison uses
no arbitrary fixed tolerance: exact float64 equality passes; otherwise the
bound metadata must be a canonical finite float32 value and the MJCF value
must map to the same float32 grid point. This accepts serialization-only
armature (`2.71e-9`) and ankle-effort (`3.0517578e-6`, `118.2` versus
`118.199996948...`) residues while a neighboring float32 value still fails.
Do not put report-only formatting changes in `venue_ball_sampler.py`: that
module's complete SHA is part of every schema-v3 bank physics contract. Final
artifact exactness is overlaid by `mujoco_eval_onnx.py` while rendering the
denominator report, leaving the bank/scorer source bytes immutable.
Long-run milestones, paired stopping rules and peak
density are specified in
`docs/research/phase1_ablation_acceleration_2026-07-11.md`.
The first historical repair manifests intentionally produced a full clean plus
5%-noise record. Do not copy that cost into every milestone. The fresh
preflight retry manifests
`configs/phase1_checkpoint_curve_fresh_retry_pod{1,2}_20260711.json` use one
fixed clean (`ns=0`) schedule with `K=20` (10 questions per side) to establish
direction. A stop or promotion still requires a separately pre-registered
50-per-side clean paper; noise and full-paper cells are reserved for survivors.

The ongoing original-arm milestones are frozen in
`configs/phase1_checkpoint_curve_cadence_pod{1,2}_20260711.json`. A cadence
worker may be started before later files exist:

```bash
python3 "$EVAL/hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py" \
  --manifest "$RUNTIME_MANIFESTS/phase1_checkpoint_curve_cadence_pod1_20260711.json" \
  --judge-script "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" \
  --state-dir /workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/cadence_pod1 \
  --max-active-cpu 6 --wait-for-checkpoints
```

In wait mode each pre-registered path must appear and keep the same size/mtime
for five seconds before hashing. Before every launch the worker rechecks the
judge SHA and both clean commits; it never scans arbitrary `model_*.pt` files
or changes a running trainer. Jobs are ordered so paired causal milestones are
consumed together and future fresh milestones follow every 2000 iterations.

The additional 18 scale-out arms have deterministic manifests generated from
the actual run-directory bindings rather than hand-copied paths:

```bash
python3 hope_training/whole_body_tracking/scripts/generate_phase1_scaleout_curve_manifests.py --check
```

Run two independent wait queues per Pod so a causal terminal checkpoint cannot
block an already-ready fresh milestone:

```bash
for queue in causal fresh; do
  state="/workspace/codexschema/phase1_fresh_20260711/checkpoint_curves/scaleout_${queue}_pod1"
  mkdir -p "$state"
  nohup setsid python3 "$EVAL/hope_training/whole_body_tracking/scripts/phase1_checkpoint_curve_worker.py" \
    --manifest "$RUNTIME_MANIFESTS/phase1_checkpoint_curve_scaleout_${queue}_pod1_20260711.json" \
    --judge-script "$EVAL/hope_training/whole_body_tracking/scripts/judge.sh" \
    --state-dir "$state" --max-active-cpu 6 --wait-for-checkpoints \
    >"$state/worker.log" 2>&1 </dev/null &
  pid=$!
  printf 'queue=%s pid=%s pgid=%s\n' "$queue" "$pid" \
    "$(ps -o pgid= -p "$pid" | tr -d ' ')"
done
```

Do not add `wait` to that loop: both workers must remain independent. The
original seed-1/2 cadence is split the same way. Its causal manifest remains
`phase1_checkpoint_curve_cadence_podN_20260711.json`; the Pod1 fresh-only
manifest starts at 4000, while Pod2 starts at 6000, each with a separate state
directory. This prevents an original causal terminal from blocking later
original `SZ` milestones.

Use the matching `pod2` manifests on Pod 2. The four files cover exactly the
18 newly launched arms and 142 clean q10 jobs: causal seed 2 at
`18000/19000/20000/20998`, and fresh at `2000/4000/.../16000/16999`.
They are milestone-major direction screens only. Their metadata explicitly
sets `screen_only=true` and never authorizes stop/promotion; use a separately
frozen q50 schedule for decisions. `SZ` is the only formal target, `SP` is an
inexact non-target plant diagnostic (non-zero PhysX friction has no exact
MuJoCo `frictionloss` equivalent), and causal plus `LZ/LP` remain inexact
diagnostics. Generated inexact jobs carry only the whitelisted
`--exam-extra --allow-inexact-contract` escape.

2026-07-13 runtime note: after these curves existed, the human owner separately authorized an
operational resource prune. Eight repeatedly collapsed trainer runs were stopped after checkpoint/
contract/log preservation, while eight continued. This does **not** reinterpret the manifest as a
q10 stopping protocol and does not change either `screen_only=true` or any q50
`whole_arm_stop_allowed=false` field. The exact runtime decision and retained artifacts are in
[EXP-P1-FACE-PLANT-SCALEOUT](../experiments/2026-07/EXP-P1-FACE-PLANT-SCALEOUT.md); process ownership
and exact-PGID procedure are in
[run_on_runpod.md](run_on_runpod.md#已登记-phase-1-实验臂的算力释放).

The checked-in curve worker requires `screen_policy` on every manifest,
requires `schedule_k == 2 * attempts_per_side`, compares that schedule (and
optional seed/noise constants) with every job, and records both the complete
manifest SHA and canonical screen-policy-plus-job contract SHA in state. Only
the latter gates per-job reuse, so appending an unrelated later job does not
invalidate a completed result; any change to that job or its screen policy is
rejected rather than silently skipped. The historical `initial`/`fresh_retry`
manifests predate this
schema discipline; do not restart them with the new worker without first
migrating them to an explicit screen policy and a new state directory. Current
live workers remain on their pinned clean eval checkout until they exit; never
edit that checkout underneath them.

For continuations that resume at iteration 16999 and execute 4000 updates, the
runner's terminal checkpoint is `model_20998.pt`; `model_20999.pt` is never
written. Do not infer a terminal filename by adding 4000 to the resume label.
The checked-in manifests and their deterministic generator encode 20998 and a
regression rejects 20999.

### Causal-triangle slot refill (2026-07-11)

The four second-wave followups are frozen in
`configs/phase1_causal_followups_20260711.json`. They fill only naturally idle
trainer slots and do not edit the original 24 recipes: Pod1 GPU1/GPU0 run M3
S1-only guidance-0 seed1/2; Pod2 GPU0/GPU1 run M2 S1+guidance-`-0.95`
seed1/2. Pod1 M3 seed2 additionally requires exact predecessor PGID `1310472`
to be absent plus a stable M3-old 20998 terminal. The gate is read-only and
never signals that predecessor.

Deploy the config and launcher under the external control root, never inside
the live training checkout, then verify the bytes explicitly:

```bash
CONTROL=/workspace/codexschema/phase1_fresh_20260711/control/causal_followups_v1
CONFIG="$CONTROL/phase1_causal_followups_20260711.json"
LAUNCHER="$CONTROL/launch_phase1_causal_followups_20260711.py"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = \
  050d6047fee280feb5754ec568c043fb20e468f81ef049b7420f90ec81a0efc8
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = \
  ca69e1cb90668060f150a518d9cee254f3883a80a07683c4fdfe1f3e4e071b08
```

Run read-only validation first, one exact arm at a time:

```bash
/usr/bin/python3 "$LAUNCHER" \
  --config "$CONFIG" \
  --expected-config-sha256 050d6047fee280feb5754ec568c043fb20e468f81ef049b7420f90ec81a0efc8 \
  --expected-launcher-sha256 ca69e1cb90668060f150a518d9cee254f3883a80a07683c4fdfe1f3e4e071b08 \
  --pod pod1 --arm phase1_M3_S1_only_guidance0_seed1 validate
```

Only after validation passes, replace the final `validate` with `launch`.
Repeat with the arm's registered Pod; do not edit its GPU from the command
line. The launcher rechecks clean train `6d93bcb...` and eval `46a0ce2...`,
all artifact/tool SHAs, GPU compute/trainer count and free memory, atomically
claims a never-used run directory, starts one isolated trainer PGID, validates
the emitted hard-contract, materializes the five q10 jobs and starts one
isolated checkpoint worker. On a post-start failure it may signal only those
new, sidecar-and-`/proc`-bound PGIDs. It contains no broad kill, checkout
mutation or real-robot path.

On the first read-only Pod validation, this capacity gate correctly prevented
all writes but exposed a driver reporting detail: `nvidia-smi` returned every
compute PID twice. Launcher `ca69e1cb...` de-duplicates PID rows before
counting unique compute/trainer processes; three unique trainers still allow
the fourth slot, while four unique processes still fail closed. Do not deploy
or authorize the superseded `dca9b9df...` launcher.

The original `model_16999.pt` is only an SHA-bound parent reference. Never copy
it into the new training run beside the new hard-contract sidecar: doing so
would launder checkpoint lineage. New cadence starts at 17000. Every q10 job is
screen-only and cannot stop/promote; the generated q50 file has no jobs and
remains inactive until its preregistered paired-evidence trigger is recorded.

The first followup 17k states were produced by eval `46a0ce2`'s legacy worker
SHA `8b980359...`. Their commands/results are correct, but that worker predates
the screen-policy/job-contract state binding. Replace only these four idle
workers with the external hardened worker; do not switch or edit either Git
worktree:

```bash
CONTROL=/workspace/codexschema/phase1_fresh_20260711/control/causal_followups_v1
HARD_CONFIG="$CONTROL/phase1_curve_worker_hardening_20260711.json"
HARD_TOOL="$CONTROL/replace_phase1_curve_workers_20260711.py"
HARD_WORKER="$CONTROL/phase1_checkpoint_curve_worker_hardened_21e3015.py"
test "$(sha256sum "$HARD_CONFIG" | awk '{print $1}')" = \
  d270ebb2d2e3fe45510cc1638f64841e9715f0cdccdd9fc983a61e42d5655a58
test "$(sha256sum "$HARD_TOOL" | awk '{print $1}')" = \
  d0678af285af42e16ec133e8d739ff3ce3cec0e8e3e4e39a5a973c0cc1a621ad
test "$(sha256sum "$HARD_WORKER" | awk '{print $1}')" = \
  21e301533328cad2a6684acced85fec6bb6854225eb18ca673247386f059f0eb

/usr/bin/python3 "$HARD_TOOL" \
  --config "$HARD_CONFIG" \
  --expected-config-sha256 d270ebb2d2e3fe45510cc1638f64841e9715f0cdccdd9fc983a61e42d5655a58 \
  --expected-tool-sha256 d0678af285af42e16ec133e8d739ff3ce3cec0e8e3e4e39a5a973c0cc1a621ad \
  --pod pod1 validate
```

Use `pod2` separately. `validate` is read-only and must show both exact legacy
worker PGIDs as single-member and childless. If either has a judge child, stop:
the tool does not wait for or signal it. Only then replace the final word with
`replace`. The Pod transaction rechecks both workers before its first signal,
sends TERM only to those two exact worker PGIDs (never KILL), freezes the old
17k state/sidecar/final log, starts the SHA-pinned standalone worker with the
same manifest and a never-used state directory, and rejudges 17k. Completion
requires rc=0 plus exact manifest/job/job-contract SHAs. It never manages a
trainer or judge; old evidence remains immutable beside a correction sidecar.

This one-time correction completed on 2026-07-11. Hardened worker PGIDs are
Pod1 `1416771/1416784` and Pod2 `198759/198771`; correction-sidecar SHAs are
`2faf88de...ffe3`, `1d6f8ba3...bae9`, `0dd02fae...d165`, and
`45f4334d...0ad`. All four 17k jobs were rejudged rc=0 with manifest, job spec
and job contract SHAs present. Do **not** rerun `replace`: its legacy-worker
precondition is intentionally no longer true. For current monitoring, read
each `checkpoint_cadence_q10.worker.hardened.launch.json` and manage only its
recorded PGID.

The six older global workers were separately replaced under
`configs/phase1_global_curve_worker_hardening_result_20260711.json`. Their
current PGIDs are Pod1 `1432280/1432292/1432304` and Pod2
`200706/200718/200730`. Do not rerun that replacement transaction either;
monitor the recorded launch sidecars and signal only an exact worker PGID if a
later, separately authorized repair requires it.

Before copying or launching any curve manifest, run:

```bash
python3 scripts/validate_phase1_queue_governance.py
```

For a separately supplied milestone-major manifest, validate it explicitly:

```bash
python3 scripts/validate_phase1_queue_governance.py \
  --manifest /absolute/path/to/manifest.json \
  --require-readiness-barrier
```

The validator requires q10 K=20/10 per side, screen-only/no-stop/no-promotion,
ordered milestones and barriers. It rejects q50 from the generic worker; q50
must use a preregistered paired runner.

### Paired terminal q50 runner

Do not turn a q10 trigger into an ad-hoc `judge.sh` command. The M3 terminal
paper is executed by `scripts/run_phase1_paired_bank_q50.py`, which separates
`prepare` (materialize one immutable schedule, start nothing) from `run`
(require the prepared runtime-contract SHA and validate both complete ledgers).
The accepted v2 bytes are runner
`095e476fd36fb68d500cb39ea7f71f6fee9b729209187d51599582c72c22198b`
and execution config
`550ca88988c88e94e626aed3e489cbedf981d2b32cde1bab9601ebacae05988b`.
It forces causal/inexact/non-formal semantics, K=100, 50 per side, one shared
schedule JSON, exact question order and zero censored attempts. It never
signals a process.

The 2026-07-11 M3 paper is already complete; do not rerun its no-clobber state
root. Schedule file SHA is `69f73458...7f25`, semantic schedule SHA is
`949eb196...8fc0`, runtime-contract SHA is `ca7a688a...17b2`, and paired-result
SHA is `e9bb07d3...f56e`. M3-old versus M3-S1 FH/BH/aggregate return was
`0.62/0.22/0.42` versus `1.00/1.00/1.00`, with 9 versus 0 physical falls.
The result selects S1 only in this same legacy swing-family causal paper and
the completed Isaac companion does not reproduce the ranking: both old and S1
score `0.99` aggregate on the same order. Therefore no cross-engine selection
gate closes. Full paths, the preserved fail-closed attempts and all hashes are
in `configs/phase1_M3_terminal_q50_result_20260711.json` and
`configs/phase1_M3_terminal_q50_isaac_result_20260711.json`.

The fresh exact wrapper is
`scripts/run_phase1_fresh_exact_paired_bank_q50.py`. It additionally requires
fresh lineage, a shared schema-3 hard-contract SHA and no inexact escape. The
accepted seed1 model-2000/model-4000 state root is already complete and must
not be reused. Runtime-contract SHA is `a756023d...4661`, schedule semantic
SHA is `7dc6af82...ff3e`, and paired-result SHA is `b95ba6c4...0478`.
Returns were `0.66/1.00/0.83` versus `0.00/1.00/0.50`; retain model 2000 but
continue the arm. Both cells' post-strike guard resets mean this is not a
continuity/deploy gate. Full paths and hashes are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_result_20260711.json`.

The completed fresh/exact Isaac companion consumed that same schedule file
and semantic SHA. Both checkpoints scored `0.99` aggregate (`0.98/1.00` by
side), one guard reset and zero physical falls; it does not reproduce the
MuJoCo separation. Do not interpret the earlier-checkpoint final tie-break as
cross-engine validation. Runtime-contract SHA is `63580328...b8120`, paired
result SHA is `65c08723...c18e`, and the full bindings are in
`configs/phase1_SZ_seed1_2000_vs_4000_q50_isaac_result_20260711.json`.

The current 10-second, no-wrap-teleport task does carry the robot state between
clips, but its complete-clip timing is slower than the conservative venue
A-B-A intervals. Do not claim that this pool proves arbitrary-time continuous
play, and do not change its live recipe. The offline reproduction command,
timing gap and separate `T0/T1` event-driven design are in
`docs/research/phase1_continuous_rally_timing_2026-07-11.md`.

Schema 3 has two validation levels. Structural validation is sufficient to
export a hash-bound diagnostic checkpoint whose motion is explicitly inexact;
it never promotes the metadata exact flag. A checkpoint whose embedded lineage
flag claims exactness must additionally pass the formal schema-2 motion/body
order gate. Removing or moving `params/training_contract.json` is not an
escape: a checkpoint that claims a binding while its adjacent sidecar is
missing is rejected. `judge.sh` likewise reads only that adjacent sidecar to
restore zero friction and the actor layout.

For a diagnostic sidecar (`motion_kinematics_exact=false` or the explicit
legacy face pairing), `judge.sh` adds `--allow-inexact-contract` to MuJoCo and
prints that decision in its preflight. A fresh exact candidate receives no
escape. Legacy schema-1/2 or missing-contract runs are also diagnostic. The
Isaac export subprocess activates the requested venv and then sources this
checkout's `setup_train_env.sh`, replacing `PYTHONPATH` with
`HOPE_WBT_PYTHONPATH`; never let a user-specific Pod env select another
checkout's task package.

`hydra`, `omegaconf`, and `rsl_rl` are NOT in the package `setup.py` `install_requires`; they must be importable from the Isaac Lab Python (provide via Isaac Lab itself or `HOPE_ISAAC_VENV_SITE`). Install the package into that Python:

```bash
hope_isaac_py -m pip install -e source/whole_body_tracking
```

Quick sanity check (expect `hydra` 1.3.2):

```bash
hope_isaac_py -c "import hydra, omegaconf; print(hydra.__version__)"
```

## Fresh Machine Entry

For a new computer, start from `docs/START_HERE.md`, then use this operation doc for Isaac training setup and [setup_local_sync.md](setup_local_sync.md) for ignored/private assets. Use [../../reimplement.md](../../reimplement.md) only as the long-form runbook when a gate or operation doc points at a specific step, such as the A3 URDF copy in Step 12.7 or the motion pipeline in steps 9-12.

Minimum order for a training machine:

1. Install Isaac Sim/Lab and set `HOPE_ISAAC_PYTHON` / `HOPE_ISAACLAB_ROOT` in `setup_train_env.local.sh`.
2. Source `hope_training/whole_body_tracking/setup_train_env.sh`.
3. Restore or create the A3 Isaac URDF asset and motion references listed below.
4. Run the smoke commands in this doc before starting a long training run.

## WandB Setup

WandB is optional for local motion-file training, but useful for shared/internal run logging and registry-backed motion distribution. If you use the registry, WandB needs two DISTINCT identities; they MUST differ or motion-registry reads fail with `Unable to find organization for entity ...`.

Current shared RunPod values (verified 2026-07-02) — export them in the git-ignored `setup_train_env.local.sh` so sourcing `setup_train_env.sh` picks them up:

- `WANDB_ENTITY=BerkeleyPingPong` — team/entity for run logging.
- `WANDB_REGISTRY_ORG=dongc_1-university-of-california-berkeley-org` — org for the motion registry.
- `WANDB_PROJECT=hope_wbc` — training project.
- `WANDB_MOTION_PROJECT=csv_to_npz` — motion upload project.
- `WANDB_DIR=/workspace/yikang/nohope/hope_training/wandb` — local W&B cache.

On any other machine:

```bash
wandb login
export WANDB_ENTITY=your-wandb-team
export WANDB_REGISTRY_ORG=your-wandb-org
export WANDB_PROJECT=hope_wbc
```

Run `wandb login` before registry-backed training. The API key is stored outside git (observed in `/root/.netrc`); never write it into repo files. No WandB account or testing on a fresh box? Pass `logger=tensorboard` and `motion_file=...`; this explicit smoke path needs no login or registry.

## Local Assets Needed For This Task

Before smoke tests or training, the A3 Isaac asset must exist at the path expected by `robots/agibot_a3.py`:

```text
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/agibot_a3/urdf/model.urdf
```

If it is missing, create it from the tracked Agibot ping-pong URDF package:

```bash
cd ~/workspace/HOPE
python3 scripts/prepare_a3_isaac_asset.py --force
python3 scripts/prepare_a3_isaac_asset.py --check
```

The script copies meshes/config from `agi/URDF/A3T2.5-URDF-std-pingpang/`, rewrites `package://.../meshes` URDF references to `../meshes/...`, and checks that the generated URDF references existing meshes including `right_hand_pingpang_Link.STL`, `pingpang_red_Link.STL`, `pingpang_black_Link.STL`, and `pingbang_ball_Link.STL`.

Motion references are also task-local setup. Registry-backed runs can use the WandB names from `cfg/task/*.yaml`, while local generated `.npz` files take precedence when passed explicitly:

- Internal registry paths such as `registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand"`.
- Local ignored `.npz` motion files under `hope_training/motions/preprocessed/` or `hope_training/whole_body_tracking/artifacts/`, passed as `motion_file=...` and optional `motion_file_2=...`, with the exact paths recorded in G05 when used.
- To create those files from manual videos, run the `reimplement.md` Step 9-12 flow: raw video -> GVHMR -> GMR (`--robot agibot_a3`) -> `scripts/csv_to_npz.py --robot agibot_a3 --output_file ../motions/preprocessed/<name>.npz`. Use `--upload_wandb` only if you also want registry artifacts.

The currently verified clips (2026-07-02) are the corrected HOPE +X local files
`hope_training/motions/preprocessed/hope_forehand_hopex.npz` and
`hope_training/motions/preprocessed/hope_backhand_hopex.npz`, passed as `motion_file=` / `motion_file_2=`. The older `hope_forehand:v4` / `hope_backhand:v4` registry artifacts face world +Y and fail the alignment gate below.

Optional R15 v5 ablation clips live on the team RunPod at `/workspace/shared/motions/hope_forehand_v5.npz` and `/workspace/shared/motions/hope_backhand_v5.npz`. Copy them into `hope_training/motions/preprocessed/` only for the R15 arm, then pass all phase and box changes on the CLI:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand_v5.npz \
  motion_file_2=../motions/preprocessed/hope_backhand_v5.npz \
  'task.racket.strike_phase_per_clip=[0.673,0.345]' \
  'task.racket.pos_range_per_clip.forehand.x=[0.29,0.49]' \
  'task.racket.pos_range_per_clip.forehand.y=[-0.63,-0.43]' \
  'task.racket.pos_range_per_clip.forehand.z=[0.74,0.94]' \
  'task.racket.vel_range_per_clip.forehand.x=[0.74,1.74]' \
  'task.racket.vel_range_per_clip.forehand.y=[0.71,1.71]' \
  'task.racket.vel_range_per_clip.forehand.z=[1.20,2.20]' \
  'task.racket.pos_range_per_clip.backhand.x=[0.60,0.80]' \
  'task.racket.pos_range_per_clip.backhand.y=[0.12,0.32]' \
  'task.racket.pos_range_per_clip.backhand.z=[0.81,1.01]' \
  'task.racket.vel_range_per_clip.backhand.x=[2.60,3.60]' \
  'task.racket.vel_range_per_clip.backhand.y=[0.50,1.50]' \
  'task.racket.vel_range_per_clip.backhand.z=[1.66,2.66]' \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=r15_v5_local_smoke
```

Before copying these values into any config, rerun `python scripts/analyze_strike_phase.py --clip forehand:../motions/preprocessed/hope_forehand_v5.npz --clip backhand:../motions/preprocessed/hope_backhand_v5.npz`; it should print `task.racket.strike_phase_per_clip=[0.673,0.345]`. Video/GVHMR face normals are wrist +Y proxies and are marked unreliable.

Use registry paths only after `scripts/check_motion_target_alignment.py --clip ...` passes for those downloaded artifacts. Do not commit generated logs, checkpoints, WandB caches, or motion artifacts unless the asset policy changes.

## Video-To-Motion Doc Map

Use this order when generating new reference clips from manually imported videos:

1. Restore local-only motion tooling and model/checkpoint assets: [setup_local_sync.md](setup_local_sync.md) steps 6-8.
2. Run the long-form command sequence: [../../reimplement.md](../../reimplement.md) steps 9-12.
3. Confirm local outputs exist:
   `hope_training/motions/preprocessed/hope_forehand.npz` and
   `hope_training/motions/preprocessed/hope_backhand.npz`.
4. Scrub the source video frame-by-frame for ball contact, record the result in `cfg/strike_annotations.yaml`, and rerun `scripts/analyze_strike_phase.py`; do not promote the speed peak by itself.
5. Replay with `scripts/replay_npz.py --motion_file ...`, then train with `motion_file=... motion_file_2=...`.
6. Add `--upload_wandb` / `registry_name=...` only for shared registry runs.

## Smoke Test

`TrackingFlat` needs a reference motion but no motion registry and no WandB account, so it is the cleanest smoke test once you have a local `.npz`:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke
```

Success means the env builds, PPO prints learning iterations, and rewards remain finite.

Before any HOPE task smoke, run the no-Isaac motion/target gate (it defaults to the local `_hopex` clips; pass `--clip name:path.npz` to check other files, e.g. registry downloads):

```bash
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPong.yaml
python scripts/check_motion_target_alignment.py --yaml cfg/task/HOPEPingPongRealSensor.yaml
```

Both commands passed on 2026-07-02 using `hope_forehand_hopex.npz` / `hope_backhand_hopex.npz` against the pre-merge branch YAML values; re-run them against the merged uniform-target config before the next long run. The same check intentionally fails on the old v4 registry downloads because frame-0 yaw is 82.03/85.92 deg and strike velocity is +Y-dominant.

For R15 v5, use `scripts/analyze_strike_phase.py` with `cfg/strike_annotations.yaml` instead of the +X-dominance gate: the forehand contact is hand-verified at frame 37 / phase 0.673, while the later speed peak is the known whip trap.

Local corrected-clips smoke for the unified HOPE task:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand_hopex.npz \
  motion_file_2=../motions/preprocessed/hope_backhand_hopex.npz \
  num_envs=32 max_iterations=1 logger=tensorboard run_name=smoke_hopex_local
```

Registry + WandB smoke `6xus13ga` on 2026-07-02 finished and synced `model_0.pt`, ONNX, config, diff, output log, and summary, but used the later-rejected v4 +Y-facing motions. Keep it as pipeline evidence only, not motion-quality evidence.

## Table-Tennis Physics Scene Smoke Test

This is a G04 scene/physics check, not the accepted G05 WBC baseline. It loads
`HOPE-TableTennis-AgibotA3-v0`, serves a ball from the P2 half toward the P1-side A3, and verifies the
table/net/ball frame plus drag/Magnus hooks.

```bash
hope_isaac_py scripts/play_table_tennis.py
hope_isaac_py scripts/play_table_tennis.py --num_envs 9
hope_isaac_py scripts/play_table_tennis.py --fix_base
hope_isaac_py scripts/play_table_tennis.py --enable_aero
hope_isaac_py scripts/play_table_tennis.py --magnus 0.1
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
```

Expected behavior: by default the ball arcs under PhysX gravity/contact only, bounces on the table, and travels toward the robot. With `--enable_aero` or `--magnus`, the HOPE drag/Magnus callback also affects flight. With `--fix_base`, the pelvis stays pinned for a stable visualization. Without it, the robot may drift or fall because no balance/return policy exists yet.

Default table-tennis scene behavior follows Purdue PACE parity: the ball uses PhysX gravity plus contacts with ball mass `3.4 g`, ball/table restitution/friction `0.9/0.1` and `0.95/0.4`, and aero drag off. Pass `--enable_aero` to use the HOPE-calibrated drag callback; `--magnus` also enables aero and adds spin.

## Baseline Training Commands

Plain tracking first:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  logger=tensorboard run_name=forehand_tracking
```

HOPE racket task, unified forehand+backhand policy from local Step 9-12 motions:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=../motions/preprocessed/hope_backhand.npz \
  logger=tensorboard run_name=hope_local_unified
```

Registry-backed equivalent:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  registry_name="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_forehand" \
  registry_name_2="$WANDB_REGISTRY_ORG/wandb-registry-motions/hope_backhand" \
  run_name=hope_registry_unified
```

Useful overrides:

```bash
num_envs=4096 max_iterations=20000 seed=1
```

`HOPEPingPong` defaults to a unified policy: clip 0 comes from `registry_name` / `motion_file`, clip 1 comes from `registry_name_2` / `motion_file_2`, and the actor receives `swing_type`. The HOPE task YAMLs also set `motion.wrap_teleport: false` (the code default), so a mid-episode clip wrap resamples the reference clip/time and racket target without teleporting the simulated robot; episode reset still uses RSI.

Resume / curriculum hand-off (added on `train_1`): `checkpoint_path=<model.pt>` loads weights + optimizer
from a prior run and CONTINUES training (the iteration counter resumes). Use it to apply a staged config
change — e.g. tightening `racket_velocity_std` — without throwing away progress:

```bash
checkpoint_path=logs/rsl_rl/agibot_a3_hope/<run>/model_2000.pt
```

Single-swing policy, if you deliberately want only one clip:

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongDeployParity algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  motion_file_2=null logger=tensorboard run_name=hope_forehand_local_smoke
```

At startup, `scripts/train.py` prints:

- the imported `whole_body_tracking` path,
- the composed env-cfg source file,
- every applied task override from `cfg/task/<name>.yaml`,
- the post-override racket reward stds and target-sampling knobs.

If a YAML key targets a missing env-cfg attribute, training now raises instead of silently ignoring the
override. Treat these printed lines as part of the G05 verification record.

### ppo.yaml deltas on this branch

- `max_iterations: 300000000000` is a train-FOREVER sentinel. Always pass `max_iterations=` on the CLI and stop manually when `strike_success` plateaus.
- `save_interval` 500 -> 100.
- `entropy_coef` is 0.01; treat `cfg/algo/ppo.yaml` as the source of truth for the current value.

### Racket Target Sampling

`HOPEPingPongDeployParity.yaml` defaults to `racket.target_mode: uniform`, matching the HITTER
structure. The command term samples per-clip 3-D blade-centered position AND velocity boxes
(`pos_range_per_clip` / `vel_range_per_clip`); the earlier fixed hit plane `x=0.4` with (y,z)-only
sampling is superseded. The imitated clip supplies the motion prior and, in the unified policy, the
swing type.

```yaml
target_mode: uniform
pos_range_per_clip:
  forehand: {x: [0.58, 0.78], y: [-0.64, -0.24], z: [0.72, 0.92]}
  backhand: {x: [0.56, 0.76], y: [-0.07,  0.33], z: [0.93, 1.13]}
vel_range_per_clip:
  forehand: {x: [1.05, 2.05], y: [ 0.96, 1.96], z: [0.31, 1.11]}
  backhand: {x: [1.61, 2.61], y: [-1.21, -0.21], z: [0.00, 0.71]}
strike_phase_per_clip: [0.47, 0.333]
```

Strike phases are blade-speed-peak detected per clip version (`scripts/analyze_strike_phase.py`). The
values above come from the 2026-07-02 blade re-detect on the re-grounded `_hopex` v3 clips; the old
`[0.36, 0.74]` values were for the v1 clips.

For two local clips, `MotionLoader` concatenates the files in order: clip 0 is forehand and clip 1 is
backhand. Keep `motion_file` / `motion_file_2` ordered the same way as `strike_phase_per_clip`.

`target_mode: reference_perturbed` remains available as a NON-default option (it was the pre-merge
default on the `rsi-on-wrap-progress-fix` branch). It centers the initial target on each imitated clip's
own PER-CLIP strike-frame racket FK state (selected by `motion.clip_id`), so the teacher action and
training target start aligned, and widens the distribution only when the success-gated exact-strike
metric advances `ref_perturb_scale` (`ref_perturb_curriculum_start: 0.05`,
`ref_perturb_pos: [0.15, 0.20, 0.15]`, `ref_perturb_vel: [1.0, 1.0, 0.8]`). Use it only for controlled
comparisons against the uniform default.

For the real-sensor footwork variant, `racket_progress` is zeroed on motion/target resample steps and its previous-distance baseline is reset. This prevents clip wrap or reset from contributing a fixed progress penalty/reward that the policy cannot control.

## Live Training Telemetry

`MotionOnPolicyRunner` (`utils/my_on_policy_runner.py`) logs a `Live/...` dashboard to WandB/TensorBoard every PPO iteration. Namespaces:

- `Live/<command_term>/<metric>` — per-axis command tracking (reference vs robot anchor pos/vel per x/y/z, joint error mean/max, `motion_phase`, racket pos/vel/normal per axis, `time_to_strike_s`, `pre_strike_flag`, `strike_window_flag`, `racket_speed`, ...).
- `Live/Reward/<term>` — per-reward-term contributions.
- `Live/Termination/*`, `Live/Action/*`, `Live/Env/*`.

The real "is it learning to hit" signal is the exact-strike metric group:
`strike_pos_pass_exact`, `strike_vel_pass_exact`, `strike_normal_pass_exact`,
`strike_composite_success_exact`, and `exact_strike_sample_count_decayed`. These are conditional
sample-weighted pass rates at the exact strike frame. `strike_success` and the `*_at_strike` metrics are
still useful, but episode-wide errors are diluted by the long non-strike phase, so do not judge progress
from them.

## Reward Shaping (strike_success=0 fix)

The reward kernel is `exp(-||err||^2 / std^2)`. With `std` set to the final acceptance tolerance, the reward is ~0 for any early error (a 50 cm error gives `exp(-44) ~ 0`), so there is no gradient and `strike_success` stays stuck at 0. The target-sampling fix above handles unreachable targets; the reward shaping here handles too-narrow early rewards.

The current `HOPEPingPongDeployParity.yaml` values try to keep useful gradients in the observed error
band while preventing non-hit rewards from dominating:

- `racket_position_weight: 14.0`, `racket_position_std: 0.20`
- `racket_velocity_weight: 10.0`, `racket_velocity_std: 1.0` (curriculum-tightened from 1.8 as the observed velocity error fell; the plan is 1.0 -> 0.8 -> 0.5)
- `racket_normal_weight: 5.0`, `racket_normal_std: 0.30`
- `base_position_weight: 2.5`, `base_position_std: 0.25` — legacy `HOPEPingPong` task only; the deploy-parity default REMOVES the base_position term entirely (base-free footwork: dense `racket_progress` plus pre-strike stability penalties)
- regularization: `joint_torques_weight: -0.00003`, `action_rate_weight: -0.10`, `joint_limit_weight: -10.0`, and `undesired_contacts_weight: -0.1`
- `base_decel_weight: 0.0` (P2.4, OFF by default; trial weight 1.0) — PACE-style pre-strike base
  speed shaping: `exp(-(||v_base_xy|| - v_des)^2 / base_decel_std^2) * pre_strike` with
  `v_des = clamp(base_decel_v_gain * planar_dist(racket→target), 0, base_decel_v_max)` (defaults
  `v_gain 2.0 /s`, `v_max 1.6 m/s`, `std 0.4 m/s`). Uses racket→target planar distance, NOT base
  position (deploy-parity obs and the base-free reward structure stay untouched); gated dead at
  and after the strike frame so it never commands a speed-up toward the swung-through old target.
  Speed-magnitude-only v1; the v2 spec (fitted accel/decel envelope, direction term, time budget,
  stroke-amplitude coupling) is `docs/motion_and_contract_v3.md` §5. Watch:
  `base_speed_xy_prestrike`.
- `joint_velocity_limit_hinge_weight: 0.0` and `joint_velocity_limit_hinge_margin: 0.85`
  ([关节速度限位铰链惩罚](../DEFINITIONS.md#qdot-limit-hinge)，VirtualBall only，默认关闭)：
  `mean(relu(abs(qd)/joint_velocity_limits - margin)^2)`。启用 weight 必须 `<= 0`；实现读取
  `robot.data.joint_vel` 和同一 31-joint articulation order 的实际 `joint_vel_limits`，不是
  `action_rate_weight` 的别名。启动会打印两个 applied marker；关节重排/缺失、零/非有限 limit 或
  每个 environment 的 limit 不一致都 fail closed。
- `racket_face_conditional_guidance_weight: 0.0`
  ([不逃离就绪区的固定预算 Reward](../DEFINITIONS.md#conditional-face-guidance)，默认关闭)：
  只在 wide strike window 内收费；位置误差用 `9.5→7.5 cm`、完整拍速向量误差用
  `1.0→0.5 m/s` 形成连续就绪门。未就绪时成本固定为 1；进入门后按就绪度把这份成本换成拍面误差
  （15° 内为 0，180° 为 1）。因此位置或拍速越就绪，成本只会不变或下降，不能靠故意退到门外免罚；
  门外拍面梯度为零。函数输出 `[0,1]`，weight 必须 `<=0`，所以 `|weight|` 是每个时间窗 step 的
  硬预算。开启时会记录 `face_conditional_guidance_gate`、
  `face_conditional_guidance_error_fraction` 与 `face_conditional_guidance_cost_fraction`；`+200` 若
  gate 全程为零，说明没有真正获得拍面纠偏信号。公式、配对与 `+200/+500/+1000` 门见
  [实验卷宗](../experiments/2026-07/EXP-P1-CONDITIONAL-FACE-GUIDANCE.md)。

  首轮只能通过 paired lean YAML 点火：control/treatment 均显式固定历史 static guidance 为 `0`，
  treatment 只设置 `++task.rewards.racket_face_conditional_guidance_weight=-0.4`，control 设 `0.0`；
  两格必须使用包含该 hard-contract 字段的同一 source。不要拿旧 source checkpoint 当 control，
  也不要同时扫 gate 或 weight。

qdot-limit treatment 尚未选择采用的负 weight，也没有 machine prereg，因此不要直接把下面写成临时
CLI 点火。未来 paired manifest 必须让 control/treatment 从同一父 checkpoint 各自启动，并同时冻结：
exact `task.rewards.joint_velocity_limit_hinge_weight`、margin、完整 argv、source commit、outer
`training_launch_claim_sha256`，以及 emitted hard contract 的
`joint_velocity_limit_hinge_reward` 和 31 项 `joint_names/joint_velocity_limits`。不得从 treatment 的
中间 checkpoint 再派生 control，也不得用 action-rate 代替这一轴。

### 恢复/等待窗随机横向躯干推力（source 已接线，当前 `NO-LAUNCH`）

这和 qdot-limit 完全不同：qdot-limit 是关节速度惩罚；
[恢复窗随机横向躯干推力](../DEFINITIONS.md#lateral-balance-perturbation)只在 post-strike recovery 或
pre-swing hold 且非 strike window 时，给 `torso_link` COM 施加 WORLD-Y 外力。Hydra 入口只有：

```text
+task.lateral_perturbation.enabled=true
+task.lateral_perturbation.cell=L0|L1
+task.lateral_perturbation.seed=<exact uint32>
```

`L0` 是同随机机会的零推力对照；`L1` 是冻结的 `0.04–0.08 m/s` 归一化冲量、`0.10 s` pulse、每
`0.50 s` 一个机会、eligible 后选择概率 `0.5`。命令行不能改 body/frame/XZ force/torque/强度/时长；
disabled 时不能同时提供 cell/seed。启用后的 checkpoint hard contract 会绑定 cell/seed、resolved tick、
共同随机题、hard-safety、Isaac backend/显式 COM transform、全部 active EventManager term 的 exact typed
参数值与 manifest SHA，以及 metric schema。pinned `SceneEntityCfg` 的 selector/resolved ids、EventTermCfg 全
行为字段与 plain module function source identity 会绑定；未知 config、decorated/method func、非有限/callable/
opaque 参数或任一 interval term 在首次 submit 前拒绝，每步前后重验 attach 后漂移。日志输出 opportunity/
selected/commanded/backend-accepted/zero-overwrite 整数、
abandoned 与三本 impulse 账、实际整机质量 min/mean/max。`backend_accepted_*` 只证同步 setter/scene-write
提交边界，不是 solver-consumed 证据。

目前不要把上述三行加入任何 queue：exact Isaac full-scene、solver dynamics response、同 GPU throughput 与
no-host-sync 门尚未通过，机器预注册仍为 `launch_authorized=false`。现在只允许运行 dependency-light source
回归：

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_lateral_perturbation.py \
  hope_training/whole_body_tracking/tests/test_isaac_lateral_perturbation.py \
  hope_training/whole_body_tracking/tests/test_reward_flags_overrides.py
```

full-scene 首次 canary 必须改走
[专用 probe 操作页](run_lateral_perturbation_runtime_probe.md)，不能用 trainer 偷跑。

These stds are DECOUPLED from acceptance thresholds: the position metric still reports true success only
below `strike_success_pos_thresh = 0.075 m`, velocity below `0.5 m/s`, and racket-normal error below
`15 deg`.

Optional later precision pass: once the exact-strike pass rates are non-trivial, tighten the racket
position/velocity stds and resume from the checkpoint.

## Domain Randomization (deploy-parity task)

2026-07-03 note: the deploy-parity task RE-ENABLES `pd_gain_range: [0.85, 1.15]` (+/-15% PD gains)
alongside +/-15% link-mass randomization (added in the 2026-07-02 sim2real fine-tune; a deliberate,
documented departure from HITTER, which keeps PD gains fixed). External push disturbance remains
disabled.

## Evaluate And Export

`play.py` exports the policy to `<checkpoint_dir>/exported/policy.onnx`.

```bash
hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="../motions/preprocessed/hope_forehand.npz" \
  motion_file_2="../motions/preprocessed/hope_backhand.npz" \
  headless=false
```

For a formal 179-D face actor, keep the exact train bank configured in the export environment.
The native exporter refuses to create a 179 ONNX unless the live bank is exact schema 3, split
`train`, checkpoint/SHA-bound, `shared_plus_y`, and `mount_plusY_A`; it derives and hashes the
per-clip raw-A demanded-normal envelope from the bank bytes during metadata attachment. The
checkpoint contract, live export configuration and envelope payload must all agree on exact
`mount_normal_sign_per_clip=[+1,-1]`. A raw-A row is wire-representable only when
`sign[clip] * raw_A.x > 1e-6`: the external schema-2 physical-B normal remains positive-x, while
the backhand actor/bank raw-A normal is negative-x. Every row must also satisfy
`raw_A_row · reference_A > 1e-6`, identical to the deploy runtime gate; a merely positive
near-boundary dot fails export. An older 179
ONNX that lacks the envelope metadata is intentionally rejected by the current C++ loader.

The Isaac-free standalone exporter has the same rule and cannot copy the envelope from its donor.
Pass the exact train NPZ explicitly:

Use `--plan` for a genuine zero-write preflight. Plan mode loads the checkpoint with
`weights_only=True`, requires a non-negative integer `checkpoint_iteration`, checks actor and
normalizer finiteness, validates the donor ONNX, motions, harvest, train bank, training contract and
formal face-179 envelope, then exits before creating `--out`, a temporary file, an ONNX graph or an
artifact. `--help` and `--contract-import-smoke` remain lightweight commands and are not export
preflights.

```bash
python scripts/standalone_onnx_export.py \
  --ckpt /abs/run/model_<N>.pt \
  --fh /abs/forehand.npz --bh /abs/backhand.npz \
  --donor /abs/same-config-donor/policy.onnx \
  --harvest /abs/same-donor-harvest.npz \
  --train-bank /abs/s1_<family>_v3_train.npz \
  --out /abs/run/exported --run-path <label> --bake-obs-norm --plan
```

Plan success prints one JSON object. Require its `checkpoint_iteration` to equal the requested
checkpoint and require `artifact_written=false`, `graph_export_not_executed=true`, `input_dim=179`,
`output_dim=31`, `materials_validated=true`, `train_bank_validated=true`, and
`formal_face179_materials_validated=true`. The `would_write` path is descriptive only. Verify that a
missing output remains absent or an existing output (including `policy.onnx`) remains byte-identical.
Remove `--plan` only for the subsequent real export, and keep W/Y in separate new output directories.

Focused source regression (`97 passed in 0.38s` on 2026-07-19):

```bash
python3 -m pytest -q \
  hope_training/whole_body_tracking/tests/test_standalone_onnx_export_plan.py \
  hope_training/whole_body_tracking/tests/test_export_obs_norm_contract.py \
  hope_training/whole_body_tracking/tests/test_export_planner_task_revision_contract.py \
  hope_training/whole_body_tracking/tests/test_stage1_normal_envelope.py \
  hope_training/whole_body_tracking/tests/test_training_contract_schema3.py
```

This path runs the same schema-3 bank loader and motion/anchor validation before deriving the
envelope, then verifies the bank SHA against the checkpoint-side training contract. Do not pass an
exam bank, omit `--train-bank`, or reuse a donor's `stage1_*` labels for a 179 artifact.
`--contract-import-smoke` is a dependency-light probe that exits before ONNX/Torch imports and
asserts that neither `whole_body_tracking/__init__.py` nor Isaac modules were loaded.

The standalone exporter validates the checkpoint contract/binding, donor, both motions, harvest,
train bank and derived envelope before producing a graph. It writes a same-directory owned temp,
checks the ONNX and metadata round trip, fsyncs, and atomically replaces `policy.onnx`. Any
validation, export, checker or save failure leaves an existing final model byte-identical and
removes the temp; do not replace this path with a direct write to the final filename.

Headless video:

```bash
hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
  checkpoint="logs/rsl_rl/agibot_a3_hope/<RUN>/model_<N>.pt" \
  motion_file="../motions/preprocessed/hope_forehand.npz" \
  motion_file_2="../motions/preprocessed/hope_backhand.npz" \
  headless=true video=true
```

From a WandB run:

```bash
hope_isaac_py scripts/play.py task=HOPEPingPongDeployParity algo=ppo num_envs=2 \
  wandb_path="$WANDB_ENTITY/hope_wbc/<RUN_ID>" headless=false
```

### Shared schema-v3 BankExam (Isaac + MuJoCo)

Do not pass an exam bank through a training Hydra override.  The saved
`RacketTargetCommand` continues to own its train-split bank; the evaluator loads
the exam split independently and installs only the current immutable questions.

Materialize one balanced paper first.  Both simulator cells must consume this
exact JSON (same schedule SHA, question order, hold values and attempt seeds):

```bash
python scripts/materialize_bank_exam_schedule.py \
  --exam-bank /abs/path/s1_<family>_v3_exam.npz \
  --per-clip-quota 10 --schedule-seed 0 --hold-range 0 100 \
  --output /abs/path/canary.schedule.json
```

Isaac single-ball cell (`K=20` creates one environment per immutable item):

```bash
hope_isaac_py scripts/isaac_bank_exam.py \
  task=HOPEPingPongVirtualBall headless=true device=cuda:0 \
  +run_dir=/abs/path/to/training_run \
  checkpoint=/abs/path/to/training_run/model_16999.pt \
  +exam_bank=/abs/path/s1_<family>_v3_exam.npz \
  +schedule_json=/abs/path/canary.schedule.json \
  +per_clip_quota=10 +schedule_seed=0 +noise_scale=0.0 \
  +output_dir=/abs/path/isaac_canary
```

Historical M3f/M2/G1 checkpoints are ruler canaries, not formal lineage; add
`+allow_inexact_contract=true` to Isaac and `--allow-inexact-contract` to
MuJoCo, and keep the resulting `evaluation_contract_exact=false` label.
Re-exporting or resuming an old checkpoint cannot turn it exact.

For a raw normalized ONNX, the sidecar must reproduce the saved runner formula
`(obs - mean) / (std + eps)`. Zero std entries are valid constant-feature
statistics only when `eps` makes every divisor strictly positive. The loader
therefore requires finite `std>=0`, finite `eps>=0`, and elementwise
`std+eps>0`; never delete the sidecar or feed raw observations to get around a
normalization error.

The BankExam entry point resolves the current checkout's dependency-light
`stage1_question_bank.py` automatically, exports that exact path to the
sampler process and binds the loader SHA into the execution contract. Do not
install Isaac packages into the MuJoCo environment or rely on a stale
`HOPE_STAGE1_QB` value.

MuJoCo consumes the same paper and uses the same authoritative NumPy scorer:

```bash
python scripts/mujoco_eval_onnx.py \
  --onnx /abs/path/to/exported/policy.onnx \
  --motion-files /abs/path/forehand.npz /abs/path/backhand.npz \
  --target-source bank --exam-bank /abs/path/s1_<family>_v3_exam.npz \
  --exam-schedule-json /abs/path/canary.schedule.json \
  --noise-scales 0.0 --seed 0 --qdes-clamp --hold-ref stand \
  --allow-inexact-contract \
  --out-dir /abs/path/mujoco_canary
```

Omit both inexact flags for a fresh schema-v3 checkpoint/ONNX. The evaluator
hashes the exam bank before and after loading; any mid-load replacement is
fatal. The shared schedule installer is formal/fail-closed by default and
accepts an inexact sampler only through the explicit historical diagnostic
flag above.

For a valid cell, the raw ledger must contain all `K` rows in schedule order.
Physical falls, guard resets and episode timeouts remain failed attempts in the
same denominator; an external step-cap truncation invalidates the whole cell.
The versioned hold contract is `H` ready-stand policy actions followed by raw
clip frame 0; it is part of the schedule SHA, not an evaluator-local guess.
Before comparing rates, assert that Isaac and MuJoCo report the same bank SHA,
schedule SHA and ordered question IDs.  Only the `noise_scale=0` canary
survivors advance to 50 questions per side, continuous play and 5% action
noise.

For the separate carry-state ruler, add `--exam-continuity-diagnostic` to the
MuJoCo command and keep `--allow-inexact-contract`. It consumes the same finite
paper but does not reset robot/action state between questions, always stamps
the result inexact, and reports `continuity.return_and_recover_rate`. The
denominator excludes only the terminal paper row, which has no scheduled next
opportunity. Do not call the one-environment-per-question Isaac adapter a
continuous test; its physical next-ball/serve timeline is a separate pending
implementation.

### T1 post-strike event mode is source-ready, not launch-ready

Commit `be5d7cf` adds the training-side scheduler and hard-contract fields;
it does not authorize a T1 run. Do not invent a schedule JSON or point a live
trainer at `event_timing_mode=post_strike_t1`. The frozen preregistration must
continue to fail launch validation until a reviewed materializer, immutable
screen/decision schedules, continuous judges, self-hit gate, fresh baseline
and semantics-correct plant are rebound in a new launch preregistration.

Dependency-light verification is:

```bash
/Users/Franco/opt/anaconda3/envs/fast/bin/python -m pytest -q \
  hope_training/whole_body_tracking/tests/test_event_timing_scheduler.py

sha=$(shasum -a 256 configs/phase1_event_timing_t0_t1_prereg_20260711.json | awk '{print $1}')
python3 scripts/validate_phase1_event_timing_prereg.py \
  --prereg configs/phase1_event_timing_t0_t1_prereg_20260711.json \
  --expected-prereg-sha256 "$sha" --mode design-check
```

Running the same validator with `--mode launch-check` must return 1 for the
frozen preregistration. The runtime field/schema contract is documented in
`docs/interfaces/t1_event_training_contract.md`.

### Signed-face single-seed rescue funnel

Do not reuse the old 24-arm launcher or replicate four seeds after the signed-face failure. The
machine-preregistered A/B/C/D hot-start/fresh × face-guidance-off/on funnel has its own clean source
commit, no-clobber L1 completion record, exact parent/asset bindings and SSH-interruption recovery
rules. Run it only through
[`run_phase1_signed_face_rescue_funnel.md`](run_phase1_signed_face_rescue_funnel.md).

L1 is a `512 env × 25 update` launch-integrity smoke on one seed. Hot cells must save lineage `0`,
fresh cells lineage `1`, and all four must emit one common hard-contract SHA. L2 is designed as
`4096 env × 1001 update`, but v6 rejects every L2 validate/plan/launch before runtime writes until a
separate immutable signed-face directional checkpoint paper path/SHA and reviewed v7 activation
exist. This launcher starts no judge, promotes no checkpoint and buys no additional seed.

The first production preflight (`control/v1`) was rejected before any run claim because its audit
looked only for top-level tensors and top-level contract keys. RSL-RL stores weights recursively and
the contract tuple under checkpoint `infos`. The v2 launch then preserved a pre-learning A-cell
failure because it did not pass the detached worktree's source-first Python environment to the
child. v3 then overreached by importing IsaacLab before `SimulationApp`; v4 proved the ignored A3
asset does not follow a detached worktree. v5 then proved the old train bank was bound to a different
physics/source-family contract and failed before learning. Preserve v1-v5; run only `control/v6`, whose
recursive audit, deterministic environment, module-origin, restored-asset tree and no-clobber rebound
train-bank report closure are recorded in the dedicated operation page.

## First-Loop Rule

Before setting a baseline quality target, record:

1. Isaac asset path.
2. Environment start command, including `source setup_train_env.sh`.
3. Random rollout result.
4. First training command.
5. Checkpoint path, ONNX export path, and WandB run ID when WandB logging is used.
6. Failure mode or first metric.
7. Whether the run is only pipeline viability or an accepted quality baseline.

把完整的设计/run/结果/决定写入
[`../experiments/`](../experiments/README.md) 下对应的实验记录，并更新 G05；随后只在
[`../PROGRESS.md`](../PROGRESS.md) 追加一条带链接的简短记录。如果已采用 setting 或逐动作
成绩表发生变化，还要更新 [`../NOW.md`](../NOW.md)。
