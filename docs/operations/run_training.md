# Run Training

Status: Draft

## Current State

The training scaffold exists under:

- `hope_training/whole_body_tracking`

2026-07-01 update:

- `HOPEPingPong` now defaults to unified forehand+backhand HITTER training: `registry_name_2` enabled, `target_mode: uniform`, per-clip 3-D blade-centered position and velocity target boxes (`pos_range_per_clip` / `vel_range_per_clip`; this supersedes the earlier fixed hit plane `x=0.4` with (y,z)-only sampling), actor `swing_type`, and no actor racket-normal observation.
- Local Step 9-12 motion products are first-class training inputs. Pass `motion_file=<forehand.npz>` plus optional `motion_file_2=<backhand.npz>` to skip WandB entirely; omit local files to use `registry_name` / `registry_name_2`.
- `setup_train_env.sh` is portable again: it reads optional `setup_train_env.local.sh` and overridable `HOPE_ISAAC_*` paths, with auto-detection for the current `/workspace/...` Isaac layout when present.

2026-07-03 realignment:

- The default training task is now `HOPEPingPongDeployParity` (gym id `HOPE-PingPong-DeployParity-AgibotA3-v0`); `HOPEPingPongRealSensor` is a backward-compat alias for the same task. Its actor observation is 175-D deploy-parity: it removes `motion_anchor_pos_b` (3) and `base_target_pos_b` (2) and reframes `racket_target_pos_b` racket-FK-relative. Layout reference: `scripts/realsensor_obs_reference.py`; checks: `scripts/verify_realsensor.py`.
- `task=HOPEPingPong` remains available only as the legacy 180-D full-obs comparison path; it is NOT deploy-honest and cannot deploy.

`TrackingFlat` and `HOPEPingPong` forehand training have run end-to-end on the copied Agibot A3 URDF asset (31 actuated DOF), including WandB logging, checkpoint save, and ONNX export. This proves the pipeline can run; it is NOT an accepted quality baseline. G04/G05 remain Partial, and G06/G07 are not accepted until sim-to-sim and dry-run deployment gates record verification.

This branch adds:

- a scrubbed `setup_train_env.sh` as the training shell setup source of truth (site paths are now overridable env vars).
- source-first `HOPE_WBT_PYTHONPATH` ordering, so local `whole_body_tracking` edits beat stale installed copies.
- richer live `Live/...` telemetry in WandB/TensorBoard from command, reward, termination, action, and env state.
- ONNX export from training/eval inside the container.
- `HOPE-TableTennis-AgibotA3-v0`, a first-pass Isaac Lab table/net/ball/A3 physics scene for G04 visualization and future G08 returner/spin experiments, now with a tracked Purdue PACE USD table/net visual overlay and Purdue-style table/ball contact materials.
- updated `HOPEPingPong` target/reward defaults with unified forehand/backhand sampling, per-clip blade-centered uniform racket target boxes, per-clip strike timing, conditional exact-strike metrics, and debug reward logging hooks.

## Entry Files

- `hope_training/whole_body_tracking/README.md`
- `hope_training/whole_body_tracking/scripts/train.py`
- `hope_training/whole_body_tracking/scripts/play.py`
- `hope_training/whole_body_tracking/scripts/play_table_tennis.py`
- `hope_training/whole_body_tracking/scripts/probe_metric.py`
- `hope_training/whole_body_tracking/cfg/train.yaml`
- `hope_training/whole_body_tracking/cfg/play.yaml`
- `hope_training/whole_body_tracking/setup_train_env.sh`

## Environment Setup

This runs in the GPU/Isaac environment (Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10, CUDA GPU), not the ROS environment. `grasping` is the maintainer's EXAMPLE distrobox name — substitute your own box.

```bash
distrobox enter grasping
cd ~/workspace/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
```

The script must be SOURCED (not executed) in every new GPU/Isaac terminal. It defines the `hope_isaac_py` launcher, sets `HOPE_WBT_PYTHONPATH`, and exports placeholder WandB variables for optional registry/logging use.

The script is now scrubbed of site-specific paths. It reads overridable env vars with placeholder defaults:

- `HOPE_ISAAC_PYTHON` — the Isaac Lab Python interpreter `hope_isaac_py` wraps.
- `HOPE_ISAACLAB_ROOT` — your Isaac Lab checkout.
- `HOPE_ISAAC_VENV_SITE` — optional extra `site-packages` to inject (e.g. to provide `hydra`/`omegaconf`).

Set these for your machine in a git-ignored `setup_train_env.local.sh` next to the script; `setup_train_env.sh` auto-sources it if present.

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
skips WandB entirely, so use it with `logger=tensorboard` for an account-free run. If you have no motion
data at all, generate a placeholder "stand at default pose" clip (pipeline proof only, not a real swing):

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

- `WANDB_ENTITY` — your team, used for run logging.
- `WANDB_REGISTRY_ORG` — your org, used for the motion registry.
- `WANDB_PROJECT=hope_wbc` — the run project.

Replace any placeholder with your own; never commit a private identity. This branch ships `your-wandb-team` / `your-wandb-org` as placeholders.

```bash
wandb login
export WANDB_ENTITY=your-wandb-team
export WANDB_REGISTRY_ORG=your-wandb-org
export WANDB_PROJECT=hope_wbc
```

No WandB account or testing on a fresh box? Pass `logger=tensorboard` and `motion_file=...`; this explicit smoke path needs no login or registry.

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

Do not commit generated logs, checkpoints, WandB caches, or motion artifacts unless the asset policy changes.

## Video-To-Motion Doc Map

Use this order when generating new reference clips from manually imported videos:

1. Restore local-only motion tooling and model/checkpoint assets: [setup_local_sync.md](setup_local_sync.md) steps 6-8.
2. Run the long-form command sequence: [../../reimplement.md](../../reimplement.md) steps 9-12.
3. Confirm local outputs exist:
   `hope_training/motions/preprocessed/hope_forehand.npz` and
   `hope_training/motions/preprocessed/hope_backhand.npz`.
4. Replay with `scripts/replay_npz.py --motion_file ...`, then train with `motion_file=... motion_file_2=...`.
5. Add `--upload_wandb` / `registry_name=...` only for shared registry runs.

## Smoke Test

`TrackingFlat` needs a reference motion but no motion registry and no WandB account, so it is the cleanest smoke test once you have a local `.npz`:

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  motion_file=../motions/preprocessed/hope_forehand.npz \
  num_envs=32 max_iterations=3 logger=tensorboard run_name=smoke
```

Success means the env builds, PPO prints learning iterations, and rewards remain finite.

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
- `racket_velocity_weight: 10.0`, `racket_velocity_std: 1.0` (curriculum-tightened from 1.8; the plan is 1.0 -> 0.8 -> 0.5)
- `racket_normal_weight: 5.0`, `racket_normal_std: 0.30`
- `base_position_weight: 2.5`, `base_position_std: 0.25` — legacy `HOPEPingPong` task only; the deploy-parity default REMOVES the base_position term entirely (base-free footwork: dense `racket_progress` plus pre-strike stability penalties)
- regularization: `joint_torques_weight: -0.00003`, `action_rate_weight: -0.10`, `joint_limit_weight: -10.0`, and `undesired_contacts_weight: -0.1`

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
