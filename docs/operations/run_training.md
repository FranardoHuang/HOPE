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
  once per swing), and the racket velocity target scales ×s (uniform boxes, reference_perturbed,
  and the HER clamp box). Positions/normals are speed-invariant. Retiming is TRAIN-ONLY:
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

## First-Loop Rule

Before setting a baseline quality target, record:

1. Isaac asset path.
2. Environment start command, including `source setup_train_env.sh`.
3. Random rollout result.
4. First training command.
5. Checkpoint path, ONNX export path, and WandB run ID when WandB logging is used.
6. Failure mode or first metric.
7. Whether the run is only pipeline viability or an accepted quality baseline.

Write the result to G05 and [../PROGRESS.md](../PROGRESS.md).
